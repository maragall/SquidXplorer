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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from squidxplorer._engine import N_FOVS_LOOP_DEFAULT


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
                      stop: Optional[Callable[[], bool]] = None) -> DispatchResult:
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
    # Lazy, and through the package: the parity tests monkeypatch these on `squidxplorer`.
    import squidxplorer
    from squidxplorer._measure import verdict

    operator_kwargs = dict(parameters or {}) or None
    skipped_regions: set = set()

    # An operator whose save artifact is a registered COPY of the acquisition (declared via
    # operator_saves_copy) saves by running the engine with copy=True — the operator writes
    # the copy itself; there is no plate to write, so the HCS layout is never demanded.
    from squidxplorer._engine import operator_saves_copy

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

    manifest: Optional[dict] = None
    if save:
        from squidxplorer import _acq_output, _fused_output

        # Declaration-driven writer choice, and an explicit out_dir (the GUI's chosen folder,
        # the CLI's --out) is THE destination for every writer; beside-the-source is only the
        # default. A per-FOV intensity operator over an on-disk acquisition saves in the
        # acquisition's own format, full resolution (z-collapsing or z-keeping alike; only a run
        # owing every FOV, n_fovs=None, qualifies). A region operator's fused mosaic saves in
        # the stitcher's format (a Squid-style OME-TIFF per region, re-openable by open_reader).
        # Everything else keeps the OME-Zarr plate.
        acq_dst = _acq_output.acquisition_format_dst(reader, operator) if n_fovs is None else None
        fused_dst = (_fused_output.fused_format_dst(reader, operator)
                     if n_fovs is None or n_fovs is N_FOVS_LOOP_DEFAULT else None)
        if acq_dst is not None:
            manifest = _acq_output.write_acquisition_planes(
                reader, operator, out_dir or acq_dst, regions=regions,
                operator_kwargs=operator_kwargs,
                workers=workers, on_well=on_well, on_error=_on_error, stop=stop)
        elif fused_dst is not None:
            manifest = _fused_output.write_fused_acquisition(
                reader, operator, out_dir or fused_dst, regions=regions,
                operator_kwargs=operator_kwargs,
                workers=workers, on_well=on_well, on_error=_on_error, stop=stop)
        else:
            manifest = squidxplorer.write_plate(
                reader, out_dir, operator=operator, n_fovs=n_fovs, workers=workers, tiff=tiff,
                on_well=on_well, stop=stop, on_error=_on_error, regions=regions,
                operator_kwargs=operator_kwargs)
        landed = int(manifest.get("n_fields_written") or 0)
        stopped = bool(manifest.get("stopped"))
    else:
        # PREVIEW: the same engine over the same arguments, writing nothing to disk.
        stream = squidxplorer.run_plate(
            reader, operator=operator, workers=workers, n_fovs=n_fovs, on_error=_on_error,
            regions=regions, operator_kwargs=operator_kwargs)
        landed, stopped = 0, False
        try:
            for region, fov, image in stream:
                if stop is not None and stop():
                    stopped = True      # deliver nothing computed after the request to stop
                    break
                landed += 1
                if on_well is not None:
                    on_well(region, fov, image)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()                 # shut the engine's pool down NOW, not at GC
    if not stopped and stop is not None and stop():
        stopped = True                  # requested between the last field and here
    outcome, detail = verdict(landed, owed, len(skipped_regions), stopped)
    out_path = (manifest or {}).get("path") or (manifest or {}).get("plate") or copy_out
    return DispatchResult(outcome=outcome, detail=detail, landed=landed, stopped=stopped,
                          skipped_regions=frozenset(skipped_regions), manifest=manifest,
                          out_path=str(out_path) if out_path else None)
