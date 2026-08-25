"""The ONE save-vs-preview dispatch every run surface calls.

`_workers._OperatorWorker._run_body` and `_command.EngineExecutor.do_run_operator` used to be
two hand-written copies of this control flow, and each copy's PREVIEW branch independently
forgot ``operator_kwargs`` (fixed 2026-08-05 in the worker, 2026-08-06 in the executor): a
panel or command value reached the console line and not the pixels. One function, one
``parameters`` argument turned into ``operator_kwargs`` ONCE and used by BOTH branches, so a
caller cannot pass different arguments to the two — that property is the whole point.

What stays with the caller, on purpose: Qt signals and the stop-event plumbing (the worker),
console sentences and the result dict (the executor), and a stopped run's detail —
:func:`squidxplorer._measure.verdict` returns STOPPED with an empty detail because how a run
was stopped is the caller's sentence to write.

The engine is reached through :data:`_RUNNER` — a declared :class:`~squidxplorer._runner.Runner`
carrying the run as a :class:`~squidxplorer._runspec.RunSpec` — not through module attributes;
the arm bodies live in :class:`~squidxplorer._runner.InProcessRunner`, which still resolves
``write_plate`` / ``run_plate`` through the package so the parity tests' monkeypatches hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from squidxplorer._engine import N_FOVS_LOOP_DEFAULT
from squidxplorer._runner import InProcessRunner, Runner

#: THE substitution point: a second runner (process pool, remote box) replaces this object.
_RUNNER: Runner = InProcessRunner()


@dataclass(frozen=True)
class DispatchResult:
    """What one dispatched run came to, before any surface words it."""

    #: The verdict, in ``_measure``'s words; STOPPED's detail is the caller's sentence.
    outcome: str
    detail: str
    #: FIELDS that produced pixels — the manifest's ``n_fields_written`` on the save branch,
    #: the streamed count on the preview branch. verdict() only reads zero of it.
    landed: int
    stopped: bool
    #: Regions where at least one field raised and was skipped, as strings.
    skipped_regions: frozenset
    #: The save branch's manifest (``write_plate``'s or the acquisition writer's); None on a preview.
    manifest: Optional[dict]
    #: Where the save landed on disk, off the manifest; None on a preview.
    out_path: Optional[str] = None


def run_operator_once(reader, *, operator: str, save: bool, owed: int, out_dir=None,
                      regions=None, n_fovs=N_FOVS_LOOP_DEFAULT, workers=None,
                      parameters: Optional[dict] = None,
                      tiff: bool = False, on_well: Optional[Callable] = None,
                      on_error: Optional[Callable] = None,
                      stop: Optional[Callable[[], bool]] = None,
                      preview_z_level: Optional[int] = None) -> DispatchResult:
    """Run ``operator`` once — persisted when ``save``, streamed-and-dropped when not.

    ``parameters`` is the single source of the run's ``operator_kwargs``: both branches take
    the one dict built here, so a preview cannot run at the defaults while its save runs at
    the panel's values. ``on_well``/``on_error``/``stop`` are likewise passed to (or polled
    on) both branches. ``owed`` counts target WELLS; ``landed`` counts FIELDS — verdict()
    reads only zero of ``landed``, so the units never sit over each other.

    ``stopped`` is read off the save branch's manifest or the preview's mid-stream poll, and
    a final poll of ``stop`` on either branch — a stop requested in the run's tail is still a
    stopped run, not a finished one. Never off ``complete``: a skipped-well run is PARTIAL,
    not CANCELLED.
    """
    from squidxplorer._engine import operator_saves_copy
    from squidxplorer._measure import verdict
    from squidxplorer._runspec import RunSpec, write_runspec

    operator_kwargs = dict(parameters or {}) or None
    skipped_regions: set = set()

    # A 2D tab's PREVIEW of a depth-keeping per-FOV operator runs on ONE plane, the one in
    # view (Julio, 2026-08-25: "if it's on 2d mode it runs it only on that one") - the same
    # solve over a 1-plane stack, sub-second where the full stack is minutes. Declaration-
    # driven, never a name match; a save always runs full depth (computed BEFORE the copy
    # arm below flips `save`, so a copy-writing save can never be z-cropped).
    z_restrict: Optional[int] = None
    if preview_z_level is not None and not save:
        from squidxplorer import is_region_operator
        from squidxplorer._engine import operator_reduces_depth

        if not is_region_operator(operator) and not operator_reduces_depth(operator):
            z_restrict = int(preview_z_level)

    # An operator whose save artifact is a registered COPY of the acquisition (declared via
    # operator_saves_copy) saves by running the engine with copy=True — the operator writes
    # the copy itself; there is no plate to write, so the HCS layout is never demanded.
    copy_out = None
    if save and operator_saves_copy(operator):
        operator_kwargs = {**(operator_kwargs or {}), "copy": True}
        save = False
        src = getattr(reader, "source_id", None)
        if src:
            from squidxplorer._register import registered_copy_root

            copy_out = str(registered_copy_root(src))

    def _on_error(region, fov, exc):
        skipped_regions.add(str(region))
        if on_error is not None:
            on_error(region, fov, exc)

    # Captured AFTER the copy arm mutated the kwargs, so the spec is what actually ran.
    spec = RunSpec.capture(reader, operator=operator, operator_kwargs=operator_kwargs,
                           regions=regions, n_fovs=n_fovs)

    manifest: Optional[dict] = None
    if save:
        manifest = _RUNNER.run_save(reader, spec, out_dir=out_dir, tiff=tiff, workers=workers,
                                    on_well=on_well, on_error=_on_error, stop=stop)
        landed = int(manifest.get("n_fields_written") or 0)
        stopped = bool(manifest.get("stopped"))
        # Provenance beside the save — successful or partial alike, at whatever root the
        # writer's own manifest names. Nonfatal by construction (write_runspec warns).
        out_root = manifest.get("path") or manifest.get("plate")
        if out_root:
            written = write_runspec(spec, out_root, result=manifest)
            if written is not None:
                manifest["runspec"] = str(written)
    else:
        landed, stopped = _RUNNER.run_preview(reader, spec, workers=workers, on_well=on_well,
                                              on_error=_on_error, stop=stop,
                                              z_level=z_restrict)
    if not stopped and stop is not None and stop():
        stopped = True                  # requested between the last field and here
    outcome, detail = verdict(landed, owed, len(skipped_regions), stopped)
    out_path = (manifest or {}).get("path") or (manifest or {}).get("plate") or copy_out
    return DispatchResult(outcome=outcome, detail=detail, landed=landed, stopped=stopped,
                          skipped_regions=frozenset(skipped_regions), manifest=manifest,
                          out_path=str(out_path) if out_path else None)
