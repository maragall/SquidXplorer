"""SquidXplorer CLI (IMA-186) — run a post-processing operator over an HCS acquisition, headless.

This is the same engine the GUI drives, exposed for high-throughput/batch use: point it at a Squid
acquisition and it iterates the chosen operator over every well of the plate and writes a navigable
multiscale OME-Zarr plate. No GUI, no FIJI.

Structured after Cephla's stitcher CLI: a declarative pydantic ``ProcessParameters`` model (its
field docstrings become ``--help`` text) + ``CliApp.run`` + a thin ``run()`` that opens the reader
and drives the shared command layer. Keeping the "what to run" as data (parameters) and "how to
run" as the shared engine means a new operator is a new ``--projector`` value, not new CLI plumbing.
(The flag keeps its name for compatibility; it selects any OPERATOR, not only a z-reduction.)

    squidxplorer <acquisition>                          # MIP every well -> <acquisition>.hcs/plate.ome.zarr
    squidxplorer <acquisition> --projector stitch       # a REGION operator, same flag
    squidxplorer <acquisition> --wells B2,B3            # named subset
    squidxplorer <acquisition> --projector cellpose --param min_area_px=80
    squidxplorer <acquisition> --tiff                   # also write the per-plane TIFF export

THE OPERATOR SURFACE IS GENERATED, NOT LISTED
---------------------------------------------
``--projector`` accepts anything ``list_operators`` reports — every projector AND every region
operator — because that is the exact set ``EngineExecutor.do_run_operator`` accepts and
``write_plate`` dispatches on. ``--param name=value`` is validated against the operator's own
``params`` declaration, and ``--help`` prints that declaration. Neither is a hand-written table:
an operator installed from another package through the ``squidxplorer.operators`` entry-point group
appears in both with no edit to this file. (This docstring used to claim that and the code did
not: the validator only knew ``available_projectors()``.)

EXIT CODES — the whole point of a batch surface
-----------------------------------------------
``for d in */; do squidxplorer "$d" || echo "FAILED: $d"; done`` has to work, so "the run finished"
and "the run produced a plate" are DIFFERENT exit codes:

    0   every target was written
    1   nothing was written: refused, failed, or every target was skipped
    2   bad usage (argparse/pydantic rejected the command line)
    3   PARTIAL — a plate exists, but at least one target was skipped
    130 interrupted (Ctrl-C); whatever had landed is on disk and marked incomplete
"""

from __future__ import annotations

import ast
import itertools
import logging
import signal
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import CliApp, CliPositionalArg

from squidxplorer._logpane import get_logger

logger = get_logger()

# --- exit codes ---------------------------------------------------------------------------------
#
# 2 is deliberately skipped for our own outcomes: argparse/pydantic already exit 2 on a bad command
# line, and a batch script must be able to tell "you typed it wrong" from "the data was bad".

EXIT_OK = 0
EXIT_NOTHING = 1
EXIT_USAGE = 2
EXIT_PARTIAL = 3
EXIT_INTERRUPTED = 130


def operator_defaults(operator: str) -> dict:
    """``{name: default}`` an operator declares it can be run with — ``{}`` when it declares none.

    Read off the registry's ``params`` declaration (``squidxplorer._engine.Param``), which is the same
    declaration ``Operator.bind`` applies and ``list_operators`` reports — for every operator,
    region ones included, since they are entries in the same table. A region operator that declares
    no params still passes its kwargs through to ``stitch_plate``, where an unknown one is refused.
    """
    from squidxplorer import operator_params

    # ASKED, not looked up: an operator CHAIN ('bgsub+spot') is not a table key, and it declares
    # its parts' parameters namespaced `<step>.<param>` — so `--param spot.min_area_px=80` has to
    # be checkable here or the chain would be runnable with parameters the CLI called unknown.
    try:
        return {p.name: p.default for p in operator_params(operator)}
    except (KeyError, TypeError, ValueError):
        return {}


def _operator_catalogue() -> str:
    """The ``--help`` line for ``--param``: every operator with the parameters it declares."""
    from squidxplorer import is_region_operator, operator_params, runnable_operators

    entries = []
    for name in runnable_operators():
        declared = ", ".join(f"{p.name}={p.default!r}" for p in operator_params(name))
        # A region operator's remaining kwargs go straight to `stitch_plate`, which refuses an
        # unknown one there; say so rather than imply the declared list is exhaustive.
        if is_region_operator(name):
            declared = ", ".join(filter(None, (declared, "**stitcher kwargs")))
        entries.append(f"{name}({declared})")
    return "; ".join(entries)


def _parse_param(pair: str) -> tuple[str, object]:
    """``"min_area_px=80"`` -> ``("min_area_px", 80)``. Values are Python literals, else strings."""
    name, sep, raw = str(pair).partition("=")
    if not sep or not name.strip():
        raise ValueError(f"--param wants name=value, got {pair!r}")
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw                    # a bare word is a string: --param method=phase
    return name.strip(), value


class ProcessParameters(BaseModel, use_attribute_docstrings=True):
    """Run a post-processing operator over an HCS acquisition (high-throughput, headless)."""

    input_folder: CliPositionalArg[str]
    """A Squid HCS acquisition folder on this machine (the latest Cephla acquisition format)."""

    projector: str = Field(default="mip", description=(
        "Operator to run over every well. A z-reduction ('mip' = maximum intensity projection), a "
        "plane operator, or a REGION operator ('stitch' fuses a well's FOVs into one mosaic). "
        "Anything the engine can run is accepted, including an operator installed from another "
        "package — see --param for the full table."))

    # The table in this help text is BUILT from the registry at import time (after squidxplorer's
    # built-ins and its entry-point plugins have registered), never written out here — so an
    # operator from another package documents itself in `--help` on the day it is installed.
    param: list[str] = Field(default_factory=list, description=(
        "Operator parameter, name=value, repeatable: --param min_area_px=80 --param "
        "split_touching=False. Values are Python literals (numbers, True/False, quoted strings); a "
        "bare word stays a string. Checked against the operator's own declaration before anything "
        "is opened. Declared parameters per operator: " + _operator_catalogue()))

    output_folder: Optional[str] = None
    """Directory to receive ``<acquisition-name>.hcs/`` (plate.ome.zarr). Defaults to a sibling of
    the input acquisition. The output can be hundreds of GB on a large plate — aim it at a disk with
    room."""

    workers: Optional[int] = None
    """Projection worker threads. Default: all usable cores (the engine is memory-bandwidth-bound, so
    more workers mainly helps on cold/network storage)."""

    tiff: bool = False
    """Also write the individual per-plane TIFF export (Squid filename convention). This is a SECOND,
    uncompressed copy — roughly doubles on-disk size — so it's off by default."""

    n_fovs: Optional[int] = 1
    """FOVs to project per well. 1 (default) keeps the historical one-FOV-per-well behaviour.
    Pass 0 for EVERY FOV in every well (the multi-FOV mosaic, IMA-187) — note this multiplies
    both compute and output size by the FOV count, so a 36-FOV plate is ~36x the work."""

    wells: Optional[str] = None
    """Run only these wells, comma-separated and in this order: --wells B2,B3. A name that is not
    in the acquisition is refused BY NAME before any output directory is made. Combines with
    --limit (the named list is truncated to the first N)."""

    limit: Optional[int] = None
    """Process only the first N wells — a quick SLICE of the plate (subset preview) so you can test
    the operator without committing the whole plate's compute + disk. Default: every well."""

    overwrite: bool = False
    """Allow writing into an <acquisition>.hcs that already holds a plate. OFF by default: a re-run
    republishes each field over the old one, so a second run with a narrower --wells/--limit leaves
    a plate that DECLARES fewer wells than are on disk. Nothing here resumes an interrupted run —
    it either refuses or overwrites."""

    odon: bool = False
    """Also write an Odon samplesheet next to the plate and launch Odon on it (IMA-212).

    Odon is a separately-installed GPL-3 desktop viewer — SquidXplorer never bundles it. Install it
    from https://github.com/alexcoulton/odon/releases, or set $ODON_BIN. Note Odon has no
    well-plate model, so it shows the fields as a flat mosaic, and it ignores our channel colors."""

    verbose: bool = False
    """Show debug-level logging."""

    @field_validator("n_fovs")
    @classmethod
    def _n_fovs(cls, v):
        # 0 is the CLI spelling of "all". pydantic-settings maps flags to scalars, so a
        # sentinel int is cleaner here than accepting the literal string "all" or None.
        if v is not None and v < 0:
            raise ValueError(f"n_fovs must be >= 0 (0 = every FOV), got {v}")
        return v

    @field_validator("limit")
    @classmethod
    def _positive_limit(cls, v):
        if v is not None and v < 1:
            raise ValueError(f"limit must be >= 1, got {v}")
        return v

    @field_validator("input_folder")
    @classmethod
    def _exists(cls, v: str) -> str:
        p = Path(v).expanduser()
        if not p.is_dir():
            raise ValueError(f"input_folder {v!r} is not an existing directory")
        return str(p.resolve())

    @field_validator("projector")
    @classmethod
    def _known_operator(cls, v: str) -> str:
        # Validate UP FRONT: otherwise the name is only resolved lazily inside project_plate, after
        # write_plate has already written an empty plate skeleton to disk, then crashes with a raw
        # traceback. A clean CLI error before any output is the safe behavior.
        #
        # Against the RUNNABLE set — every entry of the one operator table. Validating against
        # `available_projectors()` alone made the CLI strictly narrower than the command layer it
        # fronts: `--projector stitch` was refused as "unknown" by a CLI whose executor accepts it
        # and whose write_plate dispatches on it.
        from squidxplorer import runnable_operators

        runnable = runnable_operators()
        if v in runnable:
            return v
        # Not a registered name, and it may still be an operator CHAIN ('bgsub+mip') — a legal
        # projector everywhere the engine takes one. So the engine is asked to RESOLVE it, exactly
        # as `EngineExecutor.do_run_operator` does, and for the same reason: a membership test
        # against the table is not the whole answer, and deciding otherwise here would make
        # the CLI narrower than the command layer it fronts a second time. The chain refusals
        # (`_compose`: a z-reducer that is not last, a repeated step) arrive with their own
        # sentence, which names the fix.
        from squidxplorer._engine import _resolve_operator

        try:
            _resolve_operator(v)
        except (KeyError, TypeError):
            raise ValueError(
                f"unknown operator {v!r}; this application can run: {runnable}, or a chain of "
                "those joined with '+' (e.g. 'bgsub+mip')") from None
        return v

    @model_validator(mode="after")
    def _known_parameters(self):
        """Refuse a --param the chosen operator does not declare, before the reader is opened.

        The engine refuses it too (``Operator.bind``), and later — after ``write_plate`` has made
        the output tree. Same refusal, asked sooner, in the words of the flag the user typed.
        """
        pairs = dict(_parse_param(p) for p in self.param)
        declared = operator_defaults(self.projector)
        if declared:
            unknown = [k for k in pairs if k not in declared]
            if unknown:
                raise ValueError(
                    f"{self.projector!r} does not take {', '.join(unknown)}; it takes: "
                    f"{', '.join(declared) or 'no parameters'}")
        elif pairs and self.projector not in _region_operator_names():
            raise ValueError(
                f"{self.projector!r} declares no parameters, so it cannot be given "
                f"{', '.join(pairs)}")
        return self

    def parameters(self) -> dict:
        """``--param`` as the ``RunOperator.parameters`` dict the command layer already carries."""
        return dict(_parse_param(p) for p in self.param)

    def named_wells(self) -> Optional[list[str]]:
        """``--wells`` as a list, or None. Order and duplicates are the user's; we keep the order."""
        if not self.wells:
            return None
        names = [w.strip() for w in self.wells.split(",") if w.strip()]
        if not names:
            raise ValueError("--wells was given but names no wells")
        return names


def _region_operator_names() -> set:
    from squidxplorer import available_region_operators

    return set(available_region_operators())


def _resolve_regions(params: ProcessParameters, reader) -> Optional[list[str]]:
    """The explicit ``regions`` list for the command, or None for the whole plate.

    ``--wells`` then ``--limit``, both expressed through the command layer's ``regions`` field —
    documented there as "the ONE way to express a subset". Unknown names are NOT filtered out here:
    they go to the command layer, which refuses them by name (``unknown_region``), because silently
    running 2 of the 3 wells somebody asked for is the wrong kind of helpful.
    """
    all_regions = list(reader.metadata["regions"])
    regions = params.named_wells()
    if regions is not None:
        logger.info("WELLS: %d named (%s)", len(regions), ", ".join(regions[:8])
                    + (" …" if len(regions) > 8 else ""))
    if params.limit is not None:
        source = regions if regions is not None else all_regions
        regions = source[: params.limit]
        logger.info("SLICE: first %d of %d wells (%s%s)", len(regions), len(source),
                    ", ".join(regions[:8]), " …" if len(regions) > 8 else "")
    return regions


def _check_output(out_dir: Path, overwrite: bool) -> None:
    """Refuse to write into an <acquisition>.hcs that already holds a plate, unless asked.

    A re-run does not merge: each field is republished over the old one and the plate group's well
    list is rewritten from THIS run's target set, so re-running with a narrower --wells/--limit
    leaves a store that declares fewer wells than are on disk. That used to happen silently and
    exit 0. This is a GUARD, not a resume: --overwrite proceeds exactly as before.
    """
    from squidxplorer._output import is_incomplete

    plate = out_dir / "plate.ome.zarr"
    if not plate.exists() or overwrite:
        return
    state = ("an INTERRUPTED plate (nothing here resumes it — the run starts over)"
             if is_incomplete(plate) else "a finished plate")
    raise SystemExit(
        f"{plate} already holds {state}.\n"
        f"Re-running republishes each field over the old one and rewrites the plate's well list "
        f"from this run's targets, so a narrower run leaves a store that under-describes itself.\n"
        f"Pass --overwrite to do it anyway, or aim --output-folder somewhere else.")


class _Progress:
    """A thread-safe ``on_well`` that says a well landed. Runs on WRITER threads (several at once).

    The reason this exists: ``write_plate`` has taken ``on_well`` since IMA-230 and the headless
    surface passed None, so a multi-hour plate printed one line at the start and one at the end.

    It also keeps ``wells`` — the set of regions that produced at least one field — because this
    callback is the ONLY place the CLI can learn it. The manifest counts fields written
    (``n_fields_written``) against wells targeted (``n_wells``), two different units, and printing
    one over the other is how ``16/4 wells written`` reached the console.
    """

    def __init__(self, n_targets: int) -> None:
        self.n_targets = int(n_targets)
        self.wells: set = set()
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    def __call__(self, region, fov, _image) -> None:
        with self._lock:
            n = next(self._counter)
            self.wells.add(str(region))
        # n counts FIELDS, not wells (a multi-FOV run writes several per well), so the target is
        # named as wells rather than pretending to be a percentage it cannot compute.
        logger.info("  [%d] wrote %s fov %s%s", n, region, fov,
                    f" (target: {self.n_targets} well(s))" if self.n_targets else "")


def run(params: ProcessParameters, *, stop=None) -> dict:
    """Open the acquisition and write the operator's OME-Zarr plate; return write_plate's manifest.

    The operator run itself goes through the SHARED command layer (``squidxplorer._command``) — the
    exact surface the GUI drives — so "it works from the CLI" and "the button works" stop being
    two separate questions. This function keeps only what is genuinely CLI-shaped around that one
    command: the plate-format resolution, the multi-FOV warning, the ``--wells``/``--limit`` plate
    slice (expressed as the command's explicit ``regions`` list), the overwrite guard, and the Odon
    hand-off, which is a post-write launch of a separate program and not an operator at all.

    The returned manifest carries ``outcome`` (``"ok"`` / ``"partial"`` / ``"stopped"``) — what
    :func:`main` turns into the process exit code. ``run`` itself raises ``SystemExit`` only for a
    refusal (nothing ran); a run that ran and skipped everything RETURNS, so a library caller still
    gets the manifest and the skip list.
    """
    from squidxplorer import open_reader
    from squidxplorer._command import CANCELLED, CommandBus, EngineExecutor, RunOperator
    from squidxplorer._output import incomplete_reason
    from squidxplorer._plate_shape import PlateShapeError, resolve_plate_format

    # An OME-Zarr plate is a legal INPUT (`reader.SquidZarrReader`), so "process this folder" can
    # be aimed at a store some earlier run left half-written. Refuse it here, in the same words the
    # plate window uses, rather than project a subset of a plate and report it as the plate: every
    # count downstream -- targets, wells written, the exit code -- would be honest about the run
    # and wrong about the sample. Deleting the marker is the override, exactly as in the GUI.
    why = incomplete_reason(params.input_folder)
    if why is not None:
        raise SystemExit(
            f"{params.input_folder} is an INCOMPLETE plate: {why}.\n"
            f"Re-run the write that produced it, or delete its .squidxplorer-incomplete marker to "
            f"process what did land.")

    reader = open_reader(params.input_folder)
    # What plate is this? Resolved by the module that OWNS the question: declared format ->
    # SQUIDXPLORER_WELLPLATE_FORMAT (documented there as the headless/CLI override) -> inferred from
    # the well ids. Every Squid format is supported, including a glass slide / freeform tissue
    # acquisition, which reports "glass slide" and used to be refused as 'unknown'. The only
    # refusal left is the one _plate_shape itself raises: ids that fit NO format at all.
    try:
        fmt = resolve_plate_format(reader.metadata)
    except PlateShapeError as exc:
        raise SystemExit(str(exc)) from exc
    logger.info("plate format: %s", fmt)
    # Multi-FOV policy (IMA-187): n_fovs=0 on the CLI means "every FOV"; anything else is an
    # explicit count. Only warn about discarding FOVs when we are actually discarding them.
    fpr = reader.metadata["fovs_per_region"]
    n_fovs = None if params.n_fovs == 0 else params.n_fovs
    multi = sum(1 for r in fpr if len(fpr[r]) > 1)
    if multi and n_fovs is not None:
        logger.warning("%d well(s) have >1 FOV — projecting %d per well; pass --n-fovs 0 to "
                       "project every FOV.", multi, n_fovs)
    elif n_fovs is None:
        total = sum(len(v) for v in fpr.values())
        logger.info("projecting ALL %d FOV(s) across %d well(s)", total, len(fpr))
    name = Path(params.input_folder).name
    out_parent = (Path(params.output_folder).expanduser() if params.output_folder
                  else Path(params.input_folder).parent)
    out_dir = out_parent / f"{name}.hcs"
    _check_output(out_dir, params.overwrite)
    regions = _resolve_regions(params, reader)
    n_targets = len(regions) if regions is not None else len(reader.metadata["regions"])
    logger.info("running '%s' over %s -> %s", params.projector, name, out_dir)

    # Drive the SHARED command surface. The reader is already open, so hand it to the executor
    # rather than making it re-open the folder. A refusal comes back as a value with a code — the
    # CLI turns it into a clean SystemExit instead of a traceback, the same failure the GUI shows
    # as a status-line sentence. `on_well`/`stop` are the surface's own, not command fields: a
    # command is serialisable and a callback is not.
    progress = _Progress(n_targets)
    bus = CommandBus(EngineExecutor(params.input_folder, reader=reader,
                                    on_well=progress, stop=stop))
    result = bus.execute(RunOperator(
        operator=params.projector, regions=regions, save=True,
        output_folder=str(out_parent), n_fovs=n_fovs, workers=params.workers, tiff=params.tiff,
        parameters=params.parameters(),
    ))
    if not result.ok:
        if result.refusal == CANCELLED:
            # A second Ctrl-C, or an operator that raised KeyboardInterrupt: the bus turns it into
            # a refusal (it is not a crash), and the exit code says INTERRUPTED, not FAILED.
            logger.error("%s: %s", params.projector, result.message)
            raise SystemExit(EXIT_INTERRUPTED)
        raise SystemExit(f"{params.projector}: {result.message}")
    manifest = dict(result.data["manifest"])
    skipped = list(result.data.get("skipped") or [])
    outcome = str(result.data.get("outcome") or "ok")
    detail = str(result.data.get("detail") or "")
    manifest["skipped"] = skipped
    manifest["outcome"] = outcome
    manifest["detail"] = detail
    manifest["n_targets"] = int(result.data.get("n_targets") or 0)
    # Wells that produced at least one field, counted by the callback that saw each one land. The
    # manifest cannot answer this: `n_wells` is how many wells the run TARGETED.
    manifest["n_wells_written"] = len(progress.wells)

    # SAY WHICH ONE IT WAS. "done:" over an empty plate is the line this whole exit-code change
    # exists to stop printing, so the verdict picks the level and the word.
    #
    # EACH COUNT AGAINST ITS OWN TOTAL. This printed `n_fields_written`/`n_wells` — FIELDS over
    # WELLS — labelled "wells written", so a healthy 4-well 4-FOV plate read "16/4 wells written"
    # and a run that lost a quarter of the plate read "12/4", a numerator above its denominator.
    # `_progress` carries the same warning ten lines up and this line did the pretending anyway.
    line = ("%s (%d/%d fields written across %d/%d wells, %d pyramid level(s))%s" % (
        manifest["plate"], manifest["n_fields_written"], manifest["n_fields"],
        manifest["n_wells_written"], manifest["n_wells"], manifest["levels"],
        f"  + TIFFs at {manifest['tiff']}" if manifest["tiff"] else ""))
    if outcome == "ok":
        logger.info("done: %s", line)
    else:
        logger.error("%s — %s: %s", outcome.upper(), detail, line)
    if skipped:
        # NOT "due to read errors": per-well fault isolation catches whatever the operator raised
        # (a missing plane, a bad --param for a region operator, a numeric blow-up), and each one
        # was already logged by name as a SKIP line. Naming a cause here that the summary does not
        # know is how a wrong diagnosis gets read off the last line of a run.
        logger.warning("%d well(s) SKIPPED — see the SKIP line for each: %s",
                       len(skipped), ", ".join(skipped[:15]) + (" …" if len(skipped) > 15 else ""))

    # IMA-212: hand the finished plate to Odon. Deliberately AFTER the plate is fully written
    # and recorded in the manifest, so a missing binary costs the user nothing — the output is
    # already on disk and complete. The samplesheet is derived by walking that output, not from
    # `regions`/`n_fovs`, so it cannot disagree with what was actually written.
    if params.odon:
        from squidxplorer._odon import launch_odon, write_samplesheet

        samplesheet = write_samplesheet(out_dir)
        manifest["odon_samplesheet"] = str(samplesheet)
        try:
            launch_odon(samplesheet)
        except FileNotFoundError as exc:
            raise SystemExit(f"{exc}\n\nThe plate itself is written: {manifest['plate']}") from exc

    return manifest


def exit_code(manifest: dict) -> int:
    """The process exit code for a finished run. The batch surface's whole contract.

    ``ok`` -> 0. ``stopped`` -> 130 (Ctrl-C). Everything else is 1 when NOTHING landed and 3 when
    a plate exists but is short of what was asked for, so ``squidxplorer d || echo FAILED`` catches
    both and a script that wants to keep a partial plate can tell them apart.
    """
    outcome = manifest.get("outcome", "ok")
    if outcome == "ok":
        return EXIT_OK
    if outcome == "stopped":
        return EXIT_INTERRUPTED
    return EXIT_NOTHING if not manifest.get("n_fields_written") else EXIT_PARTIAL


@contextmanager
def interrupt_stop():
    """Yield a ``stop()`` predicate wired to SIGINT: the first Ctrl-C asks for a clean partial stop.

    A second one is left to Python's default handler, so a wedged run is still killable. The engine
    drains its in-flight writes and the store keeps its ``.squidxplorer-incomplete`` marker, which is
    what makes an interrupted plate tellable from a finished one.

    A context manager because ``main()`` is importable and tests call it: the previous handler is
    always put back, so nothing this process does afterwards inherits our Ctrl-C.
    """
    stopping = threading.Event()

    def handler(_signum, _frame):
        if stopping.is_set():
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        stopping.set()
        logger.warning("interrupted — stopping after the wells already in flight "
                       "(Ctrl-C again to abort now)")

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except ValueError:          # not the main thread (a test, an embedded caller): no handler
        yield None
        return
    try:
        yield stopping.is_set
    finally:
        signal.signal(signal.SIGINT, previous)


def main(args: Optional[list[str]] = None) -> int:
    """Parse, run, and RETURN AN EXIT CODE. The console script does ``sys.exit(main())``.

    Returning the code (rather than exiting 0 unconditionally, which is what this did) is the
    whole fix for a batch loop: ``for d in */; do squidxplorer "$d" || note "$d"; done`` could not
    detect a plate that wrote nothing, because every path here ended in an implicit None.
    """
    from pydantic import ValidationError

    argv = list(sys.argv[1:] if args is None else args)
    try:
        params = CliApp.run(ProcessParameters, cli_args=argv)
    except ValidationError as exc:
        # A clean sentence, not a pydantic traceback: an unknown --projector or a bad --param is a
        # USAGE error and must read like one (and exit 2, like every other usage error).
        for err in exc.errors():
            where = ".".join(str(p) for p in err["loc"]) or "argument"
            print(f"squidxplorer: {where}: {err['msg']}", file=sys.stderr)
        return EXIT_USAGE
    logging.basicConfig(level=logging.DEBUG if params.verbose else logging.INFO)
    with interrupt_stop() as stop:
        manifest = run(params, stop=stop)
    return exit_code(manifest)


if __name__ == "__main__":
    sys.exit(main())
