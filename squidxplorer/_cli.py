"""SquidXplorer CLI: run a post-processing operator over an HCS acquisition, headless.

Exit codes: 0 every target written, 1 nothing written, 2 bad usage, 3 partial, 130 interrupted.
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

# 2 is skipped for our own outcomes: argparse/pydantic already exit 2 on a bad command line.
EXIT_OK = 0
EXIT_NOTHING = 1
EXIT_USAGE = 2
EXIT_PARTIAL = 3
EXIT_INTERRUPTED = 130


def operator_defaults(operator: str) -> dict:
    """``{name: default}`` an operator declares it can be run with; ``{}`` when it declares none."""
    from squidxplorer import operator_params

    # Asked, not looked up: a chain ('bgsub+spot') is not a table key and declares
    # namespaced params.
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
        # A region operator's remaining kwargs go straight to `stitch_plate`.
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

    param: list[str] = Field(default_factory=list, description=(
        "Operator parameter, name=value, repeatable: --param min_area_px=80 --param "
        "split_touching=False. Values are Python literals (numbers, True/False, quoted strings); a "
        "bare word stays a string. Checked against the operator's own declaration before anything "
        "is opened. Declared parameters per operator: " + _operator_catalogue()))

    output_folder: Optional[str] = None
    """Directory to receive ``<acquisition-name>.hcs/`` (plate.ome.zarr)."""

    workers: Optional[int] = None
    """Projection worker threads."""

    tiff: bool = False
    """Also write the individual per-plane TIFF export (Squid filename convention)."""

    n_fovs: Optional[int] = 1
    """FOVs to project per well: 1 (default) for one per well, 0 for EVERY FOV."""

    wells: Optional[str] = None
    """Run only these wells, comma-separated and in this order: --wells B2,B3."""

    limit: Optional[int] = None
    """Process only the first N wells — a quick SLICE of the plate (subset preview) so you can test
    the operator without committing the whole plate's compute + disk."""

    overwrite: bool = False
    """Allow writing into an <acquisition>.hcs that already holds a plate."""

    odon: bool = False
    """Also write an Odon samplesheet next to the plate and launch Odon on it."""

    verbose: bool = False
    """Show debug-level logging."""

    @field_validator("n_fovs")
    @classmethod
    def _n_fovs(cls, v):
        # 0 is the CLI spelling of "all".
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
        # Validate up front, before write_plate has written an output skeleton. Against the
        # runnable set (every entry of the one operator table), not projectors alone.
        from squidxplorer import runnable_operators

        runnable = runnable_operators()
        if v in runnable:
            return v
        # Not a registered name; may still be an operator CHAIN ('bgsub+mip'), so let the
        # engine resolve it exactly as `EngineExecutor.do_run_operator` does.
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
        """Refuse a --param the chosen operator does not declare, before the reader is opened."""
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

    Unknown names are not filtered here; the command layer refuses them by name.
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
    """Refuse to write into an <acquisition>.hcs that already holds a plate, unless asked."""
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
    """A thread-safe ``on_well`` progress callback; also collects which wells produced a field."""

    def __init__(self, n_targets: int) -> None:
        self.n_targets = int(n_targets)
        self.wells: set = set()
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    def __call__(self, region, fov, _image) -> None:
        with self._lock:
            n = next(self._counter)
            self.wells.add(str(region))
        # n counts FIELDS, not wells, so the target is named as wells rather than a percentage.
        logger.info("  [%d] wrote %s fov %s%s", n, region, fov,
                    f" (target: {self.n_targets} well(s))" if self.n_targets else "")


def run(params: ProcessParameters, *, stop=None) -> dict:
    """Open the acquisition and write the operator's OME-Zarr plate; return write_plate's manifest.

    The run goes through the shared command layer (``squidxplorer._command``), the exact surface
    the GUI drives. Raises ``SystemExit`` only for a refusal; a run that skipped everything
    still returns the manifest.
    """
    from squidxplorer import open_reader
    from squidxplorer._command import CANCELLED, CommandBus, EngineExecutor, RunOperator
    from squidxplorer._output import incomplete_reason
    from squidxplorer._plate_shape import PlateShapeError, resolve_plate_format

    # An OME-Zarr plate is a legal INPUT, so refuse a half-written store up front.
    why = incomplete_reason(params.input_folder)
    if why is not None:
        raise SystemExit(
            f"{params.input_folder} is an INCOMPLETE plate: {why}.\n"
            f"Re-run the write that produced it, or delete its .squidxplorer-incomplete marker to "
            f"process what did land.")

    reader = open_reader(params.input_folder)
    try:
        fmt = resolve_plate_format(reader.metadata)
    except PlateShapeError as exc:
        raise SystemExit(str(exc)) from exc
    logger.info("plate format: %s", fmt)
    # n_fovs=0 on the CLI means "every FOV"; only warn when FOVs are actually discarded.
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

    # Drive the shared command surface; a refusal comes back as a value with a code.
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
    # Wells that produced at least one field; the manifest's n_wells is wells TARGETED.
    manifest["n_wells_written"] = len(progress.wells)

    # Each count against its own total: fields/fields and wells/wells, never mixed.
    line = ("%s (%d/%d fields written across %d/%d wells, %d pyramid level(s))%s" % (
        manifest["plate"], manifest["n_fields_written"], manifest["n_fields"],
        manifest["n_wells_written"], manifest["n_wells"], manifest["levels"],
        f"  + TIFFs at {manifest['tiff']}" if manifest["tiff"] else ""))
    if outcome == "ok":
        logger.info("done: %s", line)
    else:
        logger.error("%s — %s: %s", outcome.upper(), detail, line)
    if skipped:
        logger.warning("%d well(s) SKIPPED — see the SKIP line for each: %s",
                       len(skipped), ", ".join(skipped[:15]) + (" …" if len(skipped) > 15 else ""))

    # Odon hand-off, deliberately AFTER the plate is fully written and recorded.
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
    """The process exit code for a finished run: 0 ok, 130 stopped, 1 nothing landed, 3 partial."""
    outcome = manifest.get("outcome", "ok")
    if outcome == "ok":
        return EXIT_OK
    if outcome == "stopped":
        return EXIT_INTERRUPTED
    return EXIT_NOTHING if not manifest.get("n_fields_written") else EXIT_PARTIAL


@contextmanager
def interrupt_stop():
    """Yield a ``stop()`` predicate wired to SIGINT: first Ctrl-C stops cleanly, second aborts."""
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
    """Parse, run, and RETURN AN EXIT CODE. The console script does ``sys.exit(main())``."""
    from pydantic import ValidationError

    argv = list(sys.argv[1:] if args is None else args)
    try:
        params = CliApp.run(ProcessParameters, cli_args=argv)
    except ValidationError as exc:
        # A clean sentence and exit 2, not a pydantic traceback.
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
