"""One named command surface, shared by the GUI, the CLI and scripts. No Qt.

Commands are serialisable pydantic models dispatched by :class:`CommandBus` to an executor's
``do_<kind>``; every command returns a :class:`CommandResult`, and a failure is a named refusal.
"""

from __future__ import annotations

from typing import ClassVar, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from squidxplorer import _run_scope

from squidxplorer._logpane import get_logger

logger = get_logger("command")

__all__ = [
    "Command",
    "CommandResult",
    "CommandBus",
    "EngineExecutor",
    "OpenAcquisition",
    "ListOperators",
    "Describe",
    "RunOperator",
    "StopRun",
    "Metrics",
    "COMMANDS",
    "parse_command",
    "REFUSALS",
    "NO_ACQUISITION",
    "UNKNOWN_OPERATOR",
    "UNAVAILABLE_OPERATOR",
    "UNKNOWN_REGION",
    "EMPTY_SCOPE",
    "BAD_SCOPE",
    "BUSY",
    "NO_RUN",
    "NOT_SUPPORTED_HERE",
    "UNKNOWN_COMMAND",
    "BAD_COMMAND",
    "NO_DISK_SPACE",
    "FAILED",
    "CANCELLED",
]

# --- refusal codes -----------------------------------------------------------------------------

NO_ACQUISITION = "no_acquisition"        # nothing is open; open one first
UNKNOWN_OPERATOR = "unknown_operator"    # not in the engine registry (the answer lists what is)
UNAVAILABLE_OPERATOR = "unavailable_operator"   # registered, but a `requires=` package is missing
UNKNOWN_REGION = "unknown_region"        # a named region is not in this acquisition
EMPTY_SCOPE = "empty_scope"              # the scope resolved to nothing — never widen it silently
BAD_SCOPE = "bad_scope"                  # not one of _run_scope.RUN_SCOPES
BUSY = "busy"                            # a run is already in flight — RETRYABLE
NO_RUN = "no_run"                        # asked to stop nothing
NOT_SUPPORTED_HERE = "not_supported_here"   # this executor cannot express this command
UNKNOWN_COMMAND = "unknown_command"      # no such kind
BAD_COMMAND = "bad_command"              # the kind exists; the payload does not validate
NO_DISK_SPACE = "no_disk_space"          # the estimated write does not fit
FAILED = "failed"                        # the work ran and raised — the detail carries the name
CANCELLED = "cancelled"                  # stopped by the operator or the user; distinct from FAILED

REFUSALS = (NO_ACQUISITION, UNKNOWN_OPERATOR, UNAVAILABLE_OPERATOR, UNKNOWN_REGION,
            EMPTY_SCOPE, BAD_SCOPE, BUSY, NO_RUN, NOT_SUPPORTED_HERE, UNKNOWN_COMMAND,
            BAD_COMMAND, NO_DISK_SPACE, FAILED, CANCELLED)


# --- the result --------------------------------------------------------------------------------

class CommandResult(BaseModel):
    """What every command returns. Frozen and serialisable, like the command."""

    model_config = ConfigDict(frozen=True)

    command: str
    ok: bool
    status: Literal["completed", "started", "refused"]
    #: One of :data:`REFUSALS` when ``status == "refused"``, else None.
    refusal: Optional[str] = None
    #: A sentence for a human — the status line, the terminal, the log panel.
    message: str = ""
    #: Everything a machine reads: region lists, the manifest, the operator table, the metrics.
    data: dict = Field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.ok)

    def raise_for_refusal(self) -> "CommandResult":
        """Raise on a refusal, for a script that wants an exception."""
        if self.status == "refused":
            raise RuntimeError(f"{self.command} refused ({self.refusal}): {self.message}")
        return self


def _refuse(command: str, code: str, message: str, **data) -> CommandResult:
    logger.info("%s refused (%s): %s", command, code, message)
    return CommandResult(command=command, ok=False, status="refused", refusal=code,
                         message=message, data=data)


def _done(command: str, message: str, **data) -> CommandResult:
    return CommandResult(command=command, ok=True, status="completed", message=message, data=data)


def _started(command: str, message: str, **data) -> CommandResult:
    return CommandResult(command=command, ok=True, status="started", message=message, data=data)


# --- the commands ------------------------------------------------------------------------------

class Command(BaseModel):
    """Base for every command. Frozen, and ``extra="forbid"``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ClassVar[str] = ""

    def describe(self) -> str:
        """One line naming this command and its target — for the log, BEFORE anything runs."""
        return self.kind


class OpenAcquisition(Command):
    """Open an acquisition folder. The first command in any session."""

    kind: ClassVar[str] = "open_acquisition"
    type: Literal["open_acquisition"] = "open_acquisition"

    path: str
    """A Squid acquisition folder (any of the four layouts ``open_reader`` handles)."""

    def describe(self) -> str:
        return f"open {self.path}"


class ListOperators(Command):
    """List every runnable operator with its full declaration, availability included."""

    kind: ClassVar[str] = "list_operators"
    type: Literal["list_operators"] = "list_operators"


class Describe(Command):
    """What is open, and what could a run be aimed at?"""

    kind: ClassVar[str] = "describe"
    type: Literal["describe"] = "describe"


class RunOperator(Command):
    """Run one registered operator over a named target set."""

    kind: ClassVar[str] = "run_operator"
    type: Literal["run_operator"] = "run_operator"

    operator: str
    """A name from ``list_operators``; refused by name if it is not registered."""

    scope: str = _run_scope.SCOPE_SELECTION
    """How to resolve the target when ``regions`` is not given. One of ``_run_scope.RUN_SCOPES``."""

    regions: Optional[list[str]] = None
    """Explicit wells, in this order. ``None`` defers to ``scope``; an empty list is refused."""

    save: bool = False
    """Persist a navigable OME-Zarr plate. ``False`` (the default) is preview: write nothing."""

    output_folder: Optional[str] = None
    """Where the ``<acquisition>.hcs`` goes when ``save``. Required headless."""

    n_fovs: Optional[int] = None
    """FOVs per well. ``None`` = every FOV (the mosaic path)."""

    workers: Optional[int] = None
    """Worker threads. ``None`` = the engine's default. Only the headless surface honours this."""

    tiff: bool = False
    """When ``save``, also write the per-plane TIFF export (a second copy, so off by default)."""

    parameters: dict = Field(default_factory=dict)
    """Operator keyword arguments, passed through unchanged."""

    @field_validator("operator")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("operator must be a non-empty name")
        return str(v)

    def describe(self) -> str:
        where = (f"{len(self.regions)} named region(s)" if self.regions is not None
                 else f"scope {self.scope!r}")
        return (f"run {self.operator} on {where}"
                + (" (save)" if self.save else " (preview — not saved)"))


class StopRun(Command):
    """Stop the run in flight. A no-op is a named refusal (``no_run``)."""

    kind: ClassVar[str] = "stop_run"
    type: Literal["stop_run"] = "stop_run"


class Metrics(Command):
    """The wall-clock / peak-RSS record: the n-algorithms comparison table (:mod:`squidxplorer._measure`)."""

    kind: ClassVar[str] = "metrics"
    type: Literal["metrics"] = "metrics"

    operator: Optional[str] = None
    """Restrict to one operator's runs. ``None`` = the whole table."""


#: kind -> model. The registry a serialised command is parsed against.
COMMANDS: dict = {c.kind: c for c in (OpenAcquisition, ListOperators, Describe, RunOperator,
                                      StopRun, Metrics)}

def parse_command(payload) -> Command:
    """Build a command from a dict (or pass one straight through). Raises ``KeyError``/``ValueError``."""
    if isinstance(payload, Command):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"a command is a dict or a Command, got {type(payload).__name__}")
    data = dict(payload)
    kind = data.pop("kind", None) or data.pop("type", None)
    data.pop("type", None)
    if kind is None:
        raise KeyError(f"a command needs a 'kind'; known kinds: {sorted(COMMANDS)}")
    try:
        model = COMMANDS[str(kind)]
    except KeyError:
        raise KeyError(f"unknown command {kind!r}; known kinds: {sorted(COMMANDS)}") from None
    return model(**data)


# --- the bus -----------------------------------------------------------------------------------

class CommandBus:
    """Dispatches a command to an executor's ``do_<kind>``; ``execute`` never raises."""

    def __init__(self, executor) -> None:
        self.executor = executor

    @property
    def surface(self) -> str:
        return getattr(self.executor, "surface", type(self.executor).__name__)

    def supports(self, kind: str) -> bool:
        return callable(getattr(self.executor, f"do_{kind}", None))

    def supported(self) -> list[str]:
        """Which commands THIS surface can express. The honest edge of the migration, readable."""
        return sorted(k for k in COMMANDS if self.supports(k))

    def execute(self, payload) -> CommandResult:
        try:
            command = parse_command(payload)
        except KeyError as exc:
            return _refuse("?", UNKNOWN_COMMAND, str(exc).strip("'\""))
        except Exception as exc:            # noqa: BLE001 - pydantic validation, or a bad type
            kind = (payload.get("kind") or payload.get("type") or "?") if isinstance(payload, dict) else "?"
            return _refuse(str(kind), BAD_COMMAND, f"{type(exc).__name__}: {exc}")

        handler = getattr(self.executor, f"do_{command.kind}", None)
        if not callable(handler):
            return _refuse(command.kind, NOT_SUPPORTED_HERE,
                           f"the {self.surface} surface cannot run {command.kind!r} — it can run: "
                           f"{', '.join(self.supported()) or 'nothing'}")
        logger.info("%s: %s", self.surface, command.describe())
        try:
            result = handler(command)
        except KeyboardInterrupt:
            # Ctrl-C is a BaseException, so the `except Exception` below never sees it.
            logger.warning("%s interrupted by the user", command.kind)
            return _refuse(command.kind, CANCELLED, "interrupted (Ctrl-C) — nothing more was run")
        except Exception as exc:            # noqa: BLE001 - an executor bug is a refusal, not a crash
            logger.exception("%s raised out of %s", command.kind, self.surface)
            return _refuse(command.kind, FAILED, f"{type(exc).__name__}: {exc}")
        if not isinstance(result, CommandResult):
            return _refuse(command.kind, FAILED,
                           f"the {self.surface} surface returned {type(result).__name__}, not a "
                           "CommandResult — every command returns a result")
        return result


# --- scope, resolved ONCE, by the existing owner -------------------------------------------------

def resolve_target(command: "RunOperator", *, selection=None, current_region=None,
                   known_regions=None, total: Optional[int] = None):
    """Turn a :class:`RunOperator`'s target declaration into ``(regions, refusal_or_None)``.

    ``regions is None`` in the result means the whole dataset.
    """
    kind = command.kind
    if command.regions is not None:
        regions = [str(r) for r in command.regions]
        if not regions:
            return None, _refuse(kind, EMPTY_SCOPE,
                                 "an empty region list is not 'everything' — say so with "
                                 "regions=null or scope='whole dataset' if that is what you mean")
    else:
        regions, problem = _run_scope.resolve_run_scope(
            command.scope, selection=selection, current_region=current_region)
        if problem:
            code = BAD_SCOPE if command.scope not in _run_scope.RUN_SCOPES else EMPTY_SCOPE
            return None, _refuse(kind, code, problem)
    if regions is not None and known_regions is not None:
        known = set(str(r) for r in known_regions)
        unknown = [r for r in regions if r not in known]
        if unknown:
            return None, _refuse(kind, UNKNOWN_REGION,
                                 f"{len(unknown)} region(s) are not in this acquisition: "
                                 f"{unknown[:3]}", unknown=unknown)
    return regions, None


# --- the headless executor: the CLI and any script ----------------------------------------------

class EngineExecutor:
    """Runs commands against the engine, synchronously, with no Qt and no window."""

    surface = "engine"

    def __init__(self, path: Optional[str] = None, *, reader=None, selection=None,
                 on_well=None, stop=None) -> None:
        self._path = str(path) if path else None
        self._reader = reader
        #: The headless stand-in for the plate selection.
        self.selection: list = list(selection or [])
        #: ``on_well(region, fov, image)`` after each well lands; ``stop() -> bool`` polled before
        #: each one. ``on_well`` runs on a writer thread and must be thread-safe.
        self.on_well = on_well
        self.stop = stop

    # -- state -------------------------------------------------------------------------
    @property
    def reader(self):
        if self._reader is None and self._path:
            from squidxplorer import open_reader

            self._reader = open_reader(self._path)
        return self._reader

    def _meta(self):
        r = self.reader
        return None if r is None else r.metadata

    # -- commands ----------------------------------------------------------------------
    def do_open_acquisition(self, cmd: OpenAcquisition) -> CommandResult:
        from squidxplorer import open_reader

        self._path = cmd.path
        self._reader = open_reader(cmd.path)
        meta = self._reader.metadata
        regions = list(meta["regions"])
        return _done(cmd.kind, f"opened {cmd.path} — {len(regions)} region(s)",
                     path=cmd.path, n_regions=len(regions), regions=regions,
                     channels=[c["name"] for c in meta["channels"]],
                     wellplate_format=str(meta.get("wellplate_format", "")))

    def do_list_operators(self, cmd: ListOperators) -> CommandResult:
        from squidxplorer import (is_region_operator, operator_available, operator_consumes,
                              operator_extra, operator_params, operator_produces,
                              operator_requires, runnable_operators)

        names = runnable_operators()

        def _row(name, kind, consumes, produces, params, requires, extra, available):
            ok, why = available
            return {"name": name, "kind": kind, "consumes": consumes, "produces": produces,
                    "params": params, "requires": list(requires), "extra": extra,
                    "available": ok, "unavailable_reason": why}

        def _kind(name):
            if is_region_operator(name):
                return "region-operator"
            return "z-reducer" if operator_consumes(name) else "plane-op"

        rows = [_row(n, _kind(n), sorted(operator_consumes(n)), operator_produces(n),
                     {p.name: p.default for p in operator_params(n)},
                     operator_requires(n), operator_extra(n), operator_available(n))
                for n in names]
        blocked = [r["name"] for r in rows if not r["available"]]
        detail = f" ({len(blocked)} unavailable: {', '.join(blocked)})" if blocked else ""
        return _done(cmd.kind, f"{len(names)} operator(s): {', '.join(names)}{detail}",
                     operators=rows, names=names, unavailable=blocked)

    def do_describe(self, cmd: Describe) -> CommandResult:
        meta = self._meta()
        if meta is None:
            return _refuse(cmd.kind, NO_ACQUISITION,
                           "nothing is open — run open_acquisition first")
        regions = list(meta["regions"])
        return _done(cmd.kind, f"{self._path} — {len(regions)} region(s)",
                     surface=self.surface, path=self._path, regions=regions,
                     n_regions=len(regions),
                     channels=[c["name"] for c in meta["channels"]],
                     frame_shape=list(meta["frame_shape"]),
                     pixel_size_um=meta.get("pixel_size_um"),
                     wellplate_format=str(meta.get("wellplate_format", "")),
                     selection=list(self.selection),
                     current_region=None,
                     scopes=list(_run_scope.RUN_SCOPES))

    def do_metrics(self, cmd: Metrics) -> CommandResult:
        from squidxplorer._measure import METRICS, compare, compare_table

        rows = compare(METRICS, operators=[cmd.operator] if cmd.operator else None)
        runs = [m.as_dict() for m in METRICS
                if cmd.operator is None or m.operator == cmd.operator]
        return _done(cmd.kind, compare_table(METRICS), table=rows, runs=runs)

    def do_run_operator(self, cmd: RunOperator) -> CommandResult:
        """Run the operator to completion and return the manifest (or the streamed count).

        ``data["outcome"]`` carries the verdict: ``"ok"``, ``"partial"`` or ``"stopped"``.
        """
        from squidxplorer import runnable_operators
        from squidxplorer._dispatch import run_operator_once
        from squidxplorer._measure import measure_run

        meta = self._meta()
        if meta is None:
            return _refuse(cmd.kind, NO_ACQUISITION,
                           "nothing is open — run open_acquisition first")
        runnable = runnable_operators()
        if cmd.operator not in runnable:
            # Resolve for the refusal: a chain expression gets the engine's own explanation.
            from squidxplorer._engine import _resolve_operator

            try:
                _resolve_operator(cmd.operator)
            except (KeyError, TypeError):
                return _refuse(cmd.kind, UNKNOWN_OPERATOR,
                               f"{cmd.operator!r} is not a runnable operator — this application can "
                               f"run: {', '.join(runnable)}", available=runnable)
            except ValueError as exc:
                return _refuse(cmd.kind, BAD_COMMAND, str(exc), operator=cmd.operator)
        # Registered but not runnable on this machine: refuse before resolving the target.
        from squidxplorer import operator_available

        ok, why = operator_available(cmd.operator)
        if not ok:
            return _refuse(cmd.kind, UNAVAILABLE_OPERATOR, why, operator=cmd.operator)
        all_regions = list(meta["regions"])
        regions, refusal = resolve_target(cmd, selection=self.selection,
                                          known_regions=all_regions, total=len(all_regions))
        if refusal is not None:
            return refusal
        target = _run_scope.describe_run_target(regions, total=len(all_regions))
        n_targets = len(all_regions) if regions is None else len(regions)

        out_dir = None
        if cmd.save:
            if not cmd.output_folder:
                return _refuse(cmd.kind, BAD_COMMAND,
                               "save=true needs an output_folder — a headless run has no dialog "
                               "to ask, and the output can be hundreds of GB")
            from pathlib import Path

            out_dir = Path(cmd.output_folder).expanduser() / f"{Path(self._path).name}.hcs"

        def on_error(region, fov, exc):
            logger.warning("SKIP well %s (fov %s): %s: %s", region, fov, type(exc).__name__, exc)

        with measure_run(cmd.operator, target or "no target", n_targets=n_targets) as run:
            run.note(surface=self.surface, save=cmd.save, acquisition=self._path)
            # the ONE save-vs-preview dispatch; this executor only words the result
            result = run_operator_once(
                self.reader, operator=cmd.operator, save=cmd.save, owed=n_targets,
                out_dir=out_dir, regions=regions, n_fovs=cmd.n_fovs, workers=cmd.workers,
                parameters=cmd.parameters, tiff=cmd.tiff,
                on_well=self.on_well, on_error=on_error, stop=self.stop)
            landed = result.landed
            if cmd.save:
                # A save may carry NO writer manifest (register: the artifact is its own
                # stitched_ copy, reported by the operator) — word that, never crash.
                data = {"manifest": {k: (str(v) if hasattr(v, "__fspath__") else v)
                                     for k, v in (result.manifest or {}).items()}}
            else:
                data = {"n_fields": landed}   # PREVIEW headless: computed, counted, nothing retained
            outcome, detail = result.outcome, result.detail
            if result.stopped:
                # `landed` counts FIELDS; `n_targets` counts WELLS -- never put one over the other.
                owed = int((result.manifest or {}).get("n_fields") or 0)
                got = f"{landed} of {owed} field(s)" if owed else f"{landed} field(s)"
                detail = f"stopped after {got} across {n_targets} target well(s)"
            run.finish(outcome, detail)
            metrics = run
        data["n_landed"] = landed
        data["skipped"] = sorted(result.skipped_regions)
        data["metrics"] = metrics.metrics.as_dict() if metrics.metrics else None
        data["regions"] = regions
        data["target"] = target
        # The verdict, in `_measure`'s words; `ok` only means "the command ran".
        data["outcome"] = outcome
        data["detail"] = detail
        data["n_targets"] = n_targets
        return _done(cmd.kind, metrics.metrics.line() if metrics.metrics else "done", **data)
