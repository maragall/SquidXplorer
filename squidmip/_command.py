"""ONE named command surface, shared by the GUI, the CLI and a script. No Qt.

Julio: "make sure that our GUI has an amazing API so that AI can interact with it... The logger
and the API and the CLI are amazing to design really well and deeply so that the agent can program
our tool cosmically." And: "do all the logger, cli, api infrastructure that forces us to think of
scalable representations for how this code communicates with itself."

THE PROBLEM THIS SOLVES
-----------------------
There were TWO surfaces. The GUI called methods on a 7,000-line ``PlateWindow``; the CLI called
the engine directly. Both reached the same operators, by different routes, with different scope
rules and different failure behaviour — a GUI refusal was a sentence in a status label, a CLI
refusal was a traceback. The consequences are the ones ``docs/NAUTILUS.md`` records under
"How do I drive the app? — NOT ANSWERED":

* An agent (or a test, or a script) can drive ONE of them, and whichever it drives is not the one
  the user uses. "It works from the CLI" has never been evidence that the button works.
* Every new operator had to be wired twice, and the two wirings drifted. This codebase has already
  shipped that exact defect at a smaller scale — two operator registries, different labels,
  different ``save`` defaults, drifted in production (see ``_explore.RUN_SCOPES``).

So there is one layer. The GUI is a CALLER of it and so is the CLI.

WHAT A COMMAND IS
-----------------
A command is **named, declarative and serialisable**: a pydantic model with a ``kind``
discriminator, not a method call and certainly not a Qt signal. That is what makes it usable by
something that is not in this process — an agent, a saved workflow, a test fixture, a replay of
what the user did. ``RunOperator(operator="mip", scope="selected wells", save=False)`` is a value
you can print, diff, store next to a CSAT response, and hand back to the app to run again.

pydantic and not a plain dict for the reason ``_acquisition.Acquisition`` is pydantic: the thing
crossing the boundary is validated ONCE, at the boundary, so a typo is a named refusal at the
door rather than an ``AttributeError`` eight frames deep inside a QThread.

EVERY COMMAND NAMES ITS TARGET
------------------------------
Acquisition, regions, operator, parameters — explicitly, on the command. The established
convention "nothing selected = everything" is preserved exactly, and it is preserved by CALLING
:func:`squidmip._explore.resolve_run_scope`, which is the existing owner of scope resolution.
There is no second resolver in this module. A command carries either an explicit ``regions`` list
or a ``scope`` NAME, and the executor hands the live state (selection, current region, parked
subset) to the one resolver that already knows the rules.

EVERY COMMAND RETURNS A RESULT, AND A FAILURE IS A NAMED REFUSAL
----------------------------------------------------------------
:class:`CommandResult` always comes back. A refusal carries a CODE from :data:`REFUSALS` plus a
sentence for a human, and it is a returned value — never an exception escaping into a Qt slot,
where Qt's habit of swallowing exceptions turns "the run refused itself" into "the button did
nothing". Codes and not just sentences because an agent has to BRANCH on the reason: retrying an
``empty_scope`` is pointless and retrying a ``busy`` is exactly right.

HOW THE TWO CALLERS STAY ONE SURFACE
------------------------------------
The bus dispatches ``kind`` to ``executor.do_<kind>``. An executor that does not implement a
command refuses it BY NAME (:data:`NOT_SUPPORTED_HERE`) naming itself, so the edges of the
migration are visible in the result rather than hidden in a fork. That is the deliberate design:
a half-migrated surface that says which half it is beats a complete one that is a facade.

    EngineExecutor   headless, synchronous. The CLI and any script. Returns ``completed``.
    WindowExecutor   the GUI (``squidmip._gui_commands``). Starts a QThread; returns ``started``.

``started`` vs ``completed`` is a REAL difference and it is in the result rather than papered
over: a GUI run must not block the event loop, so the honest thing a command can say is that the
run began. A caller that needs the end of it watches the metrics/activity log, which is the same
record either way.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from squidmip import _explore

logger = logging.getLogger("squidmip.command")

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
]

# --- refusal codes -----------------------------------------------------------------------------
#
# A refusal names ITSELF. The sentence is for the user; the code is for the caller, which has to
# decide whether to retry, ask the user something, or stop. Getting this wrong is how an agentic
# loop becomes a random walk: "it didn't work" is not a branch.

NO_ACQUISITION = "no_acquisition"        # nothing is open; open one first
UNKNOWN_OPERATOR = "unknown_operator"    # not in the engine registry (the answer lists what is)
UNKNOWN_REGION = "unknown_region"        # a named region is not in this acquisition
EMPTY_SCOPE = "empty_scope"              # the scope resolved to nothing — never widen it silently
BAD_SCOPE = "bad_scope"                  # not one of _explore.RUN_SCOPES
BUSY = "busy"                            # a run is already in flight — RETRYABLE
NO_RUN = "no_run"                        # asked to stop nothing
NOT_SUPPORTED_HERE = "not_supported_here"   # this executor cannot express this command
UNKNOWN_COMMAND = "unknown_command"      # no such kind
BAD_COMMAND = "bad_command"              # the kind exists; the payload does not validate
NO_DISK_SPACE = "no_disk_space"          # the estimated write does not fit
FAILED = "failed"                        # the work ran and raised — the detail carries the name

REFUSALS = (NO_ACQUISITION, UNKNOWN_OPERATOR, UNKNOWN_REGION, EMPTY_SCOPE, BAD_SCOPE, BUSY,
            NO_RUN, NOT_SUPPORTED_HERE, UNKNOWN_COMMAND, BAD_COMMAND, NO_DISK_SPACE, FAILED)


# --- the result --------------------------------------------------------------------------------

class CommandResult(BaseModel):
    """What EVERY command returns. Frozen and serialisable, like the command.

    ``ok`` and ``status`` are not redundant: a run that STARTED is ok and is not finished, and a
    caller that treats "started" as "the pixels are on disk" is the bug this distinction prevents.
    """

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
        """For a script that WANTS an exception. The GUI never calls this — a refusal must reach a
        Qt slot as a value. Offered here so a CLI/notebook caller does not invent its own check."""
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
    """Base for every command. Frozen, and ``extra="forbid"``.

    Forbidding extras is the point of having a schema at all: an agent that writes
    ``{"kind": "run_operator", "operator": "mip", "region": "B2"}`` (singular, a plausible guess)
    must be TOLD, not silently run over the whole plate because the misspelled key was ignored.
    Silently ignoring an unknown field is how a scope mistake costs somebody an afternoon of
    compute.
    """

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
    """What can this application run? The question an agent asks first, and the one that makes a
    new operator discoverable without anybody updating a document.

    Answered off the ENGINE registries (``available_projectors`` /
    ``available_region_operators``), never off the GUI's card table — a card is presentation, an
    engine entry is capability, and confusing the two already shipped a button that did nothing.
    """

    kind: ClassVar[str] = "list_operators"
    type: Literal["list_operators"] = "list_operators"


class Describe(Command):
    """What is open, and what could a run be aimed at? The state an agent needs to build a target.

    Deliberately a COMMAND rather than an attribute on some object: it must be answerable
    identically from the GUI and headless, and a caller that reaches into ``window._meta`` is
    back to two surfaces.
    """

    kind: ClassVar[str] = "describe"
    type: Literal["describe"] = "describe"


class RunOperator(Command):
    """Run one registered operator over a named target set. THE command.

    Target, explicitly, in the established order of precedence:

    * ``regions`` — an explicit list wins over everything. This is the ONE way to express a
      subset, and it is what a reproducible/agent-issued command should carry.
    * ``scope`` — otherwise a scope NAME from :data:`squidmip._explore.RUN_SCOPES`, resolved
      against the caller's live state by ``_explore.resolve_run_scope``. The default
      ``"selected wells"`` keeps the established convention exactly: nothing selected = everything.
    """

    kind: ClassVar[str] = "run_operator"
    type: Literal["run_operator"] = "run_operator"

    operator: str
    """A name from ``list_operators`` — a projector ("mip", "bgsub") or a region operator
    ("stitch"). Refused BY NAME against the engine registry if it is not one."""

    scope: str = _explore.SCOPE_SELECTION
    """How to resolve the target when ``regions`` is not given. One of ``_explore.RUN_SCOPES``."""

    regions: Optional[list[str]] = None
    """Explicit wells, in this order. ``None`` defers to ``scope``. An empty LIST is not the same
    as ``None`` and is refused as an empty scope — running the whole plate because a caller sent
    ``[]`` is hours of compute nobody asked for."""

    save: bool = False
    """Persist a navigable OME-Zarr plate. ``False`` (the default) is PREVIEW: compute and show,
    write nothing. The default is the one that cannot fill a disk."""

    output_folder: Optional[str] = None
    """Where the ``<acquisition>.hcs`` goes when ``save``. Required headless — a headless run has
    no dialog to ask, and guessing a hundreds-of-GB destination is not ours to do."""

    n_fovs: Optional[int] = None
    """FOVs per well. ``None`` = every FOV (the mosaic path, and what the GUI runs). ``1`` is the
    historical one-FOV-per-well."""

    workers: Optional[int] = None
    """Worker threads. ``None`` = the engine's own default (CPUs usable by this process). Only the
    headless surface honours this; the GUI pins its own worker count for responsiveness."""

    tiff: bool = False
    """When ``save``, also write the uncompressed per-plane TIFF export (Squid filename
    convention). A SECOND copy — roughly doubles on-disk size — so off by default. Ignored on a
    preview (there is nothing on disk to duplicate)."""

    parameters: dict = Field(default_factory=dict)
    """Operator keyword arguments, passed through unchanged (e.g. the stitcher's ``register`` /
    ``blend_px``). A dict and not a typed model per operator on purpose: the registry scales to n
    algorithms, and a schema here would have to be edited for every one of them — which is the
    engine edit ``add_projector`` exists to avoid."""

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
    """Stop the run in flight. A no-op is a NAMED refusal (``no_run``), not a silent success —
    "stop" that returns ok over nothing teaches a caller that stopping worked."""

    kind: ClassVar[str] = "stop_run"
    type: Literal["stop_run"] = "stop_run"


class Metrics(Command):
    """The wall-clock / peak-RSS record: the n-algorithms comparison table (:mod:`squidmip._measure`).

    The third consumer of the one measurement, and the one that answers "which of these two
    implementations of the same operator should we keep".
    """

    kind: ClassVar[str] = "metrics"
    type: Literal["metrics"] = "metrics"

    operator: Optional[str] = None
    """Restrict to one operator's runs. ``None`` = the whole table."""


#: kind -> model. The registry a serialised command is parsed against, and the answer to "what can
#: I say to this application" — which is a question an agent has to be able to ask.
COMMANDS: dict = {c.kind: c for c in (OpenAcquisition, ListOperators, Describe, RunOperator,
                                      StopRun, Metrics)}

AnyCommand = Union[OpenAcquisition, ListOperators, Describe, RunOperator, StopRun, Metrics]


def parse_command(payload) -> Command:
    """Build a command from a dict (or pass one straight through). Raises ``KeyError``/``ValueError``.

    The entry point for anything outside this process: JSON on a socket, a stored workflow step, an
    agent's tool call. ``kind`` and ``type`` are both accepted as the discriminator — ``kind`` is
    what a human writes and ``type`` is the field pydantic serialises — so a round-trip through
    ``model_dump()`` and back is lossless.
    """
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
    """Dispatches a command to an executor's ``do_<kind>`` and guarantees a :class:`CommandResult`.

    The guarantee is the load-bearing part. ``execute`` NEVER raises: a bad payload, an unknown
    kind, an executor that does not implement the command, and an executor method that itself
    blows up all come back as refusals with a code. The GUI calls this from a Qt slot, where a
    raised exception is swallowed by Qt and the user sees a button that did nothing.
    """

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
                   parked_subset=None, known_regions=None, total: Optional[int] = None):
    """Turn a :class:`RunOperator`'s target declaration into ``(regions, refusal_or_None)``.

    The ONLY place a command's target is worked out, and it does not re-implement any of it:
    ``_explore.resolve_run_scope`` owns the scope rules and ``_explore.describe_run_target`` owns
    the sentence. This function adds exactly two things the resolver does not do — an explicit
    ``regions`` list wins, and a named region that is not in the acquisition is refused by name
    rather than becoming a bare ``KeyError`` several frames later.

    ``regions is None`` in the result means THE WHOLE DATASET, which is the historical plate-wide
    path and is preserved unchanged.
    """
    kind = command.kind
    if command.regions is not None:
        regions = [str(r) for r in command.regions]
        if not regions:
            return None, _refuse(kind, EMPTY_SCOPE,
                                 "an empty region list is not 'everything' — say so with "
                                 "regions=null or scope='whole dataset' if that is what you mean")
    else:
        regions, problem = _explore.resolve_run_scope(
            command.scope, selection=selection, current_region=current_region,
            parked_subset=parked_subset)
        if problem:
            code = BAD_SCOPE if command.scope not in _explore.RUN_SCOPES else EMPTY_SCOPE
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
    """Runs commands against the ENGINE, synchronously, with no Qt and no window.

    This is what the CLI drives and what a script or a test drives. It is deliberately the same
    command vocabulary the GUI answers, so "does the button work" and "does the CLI work" stop
    being two questions.
    """

    surface = "engine"

    def __init__(self, path: Optional[str] = None, *, reader=None, selection=None) -> None:
        self._path = str(path) if path else None
        self._reader = reader
        #: The headless stand-in for the plate selection. ``resolve_run_scope`` reads it exactly
        #: as it reads the GUI's, so "selected wells" means the same thing on both surfaces.
        self.selection: list = list(selection or [])
        self.last_metrics = None

    # -- state -------------------------------------------------------------------------
    @property
    def reader(self):
        if self._reader is None and self._path:
            from squidmip import open_reader

            self._reader = open_reader(self._path)
        return self._reader

    def _meta(self):
        r = self.reader
        return None if r is None else r.metadata

    # -- commands ----------------------------------------------------------------------
    def do_open_acquisition(self, cmd: OpenAcquisition) -> CommandResult:
        from squidmip import open_reader

        self._path = cmd.path
        self._reader = open_reader(cmd.path)
        meta = self._reader.metadata
        regions = list(meta["regions"])
        return _done(cmd.kind, f"opened {cmd.path} — {len(regions)} region(s)",
                     path=cmd.path, n_regions=len(regions), regions=regions,
                     channels=[c["name"] for c in meta["channels"]],
                     wellplate_format=str(meta.get("wellplate_format", "")))

    def do_list_operators(self, cmd: ListOperators) -> CommandResult:
        from squidmip import (available_projectors, available_region_operators,
                              projector_consumes)

        projectors = available_projectors()
        region_ops = available_region_operators()
        rows = [{"name": n, "kind": "z-reducer" if projector_consumes(n) else "plane-op",
                 "consumes": sorted(projector_consumes(n))} for n in projectors]
        rows += [{"name": n, "kind": "region-operator", "consumes": ["fov"]} for n in region_ops]
        names = sorted(set(projectors) | set(region_ops))
        return _done(cmd.kind, f"{len(names)} operator(s): {', '.join(names)}",
                     operators=rows, names=names)

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
                     current_region=None, parked_subset=[],
                     scopes=list(_explore.RUN_SCOPES))

    def do_metrics(self, cmd: Metrics) -> CommandResult:
        from squidmip._measure import METRICS, compare, compare_table

        rows = compare(METRICS, operators=[cmd.operator] if cmd.operator else None)
        runs = [m.as_dict() for m in METRICS
                if cmd.operator is None or m.operator == cmd.operator]
        return _done(cmd.kind, compare_table(METRICS), table=rows, runs=runs)

    def do_run_operator(self, cmd: RunOperator) -> CommandResult:
        """Run the operator to COMPLETION and return the manifest (or the streamed count).

        Synchronous on purpose: this surface has no event loop to keep responsive, and a caller
        that gets ``completed`` back knows the pixels exist. The GUI's executor returns
        ``started`` instead, and the difference is in the result rather than hidden.
        """
        from squidmip import (available_projectors, available_region_operators, project_plate,
                              stitch_plate, write_plate)
        from squidmip._measure import OK, PARTIAL, measure_run

        meta = self._meta()
        if meta is None:
            return _refuse(cmd.kind, NO_ACQUISITION,
                           "nothing is open — run open_acquisition first")
        runnable = sorted(set(available_projectors()) | set(available_region_operators()))
        if cmd.operator not in runnable:
            return _refuse(cmd.kind, UNKNOWN_OPERATOR,
                           f"{cmd.operator!r} is not a runnable operator — this application can "
                           f"run: {', '.join(runnable)}", available=runnable)
        all_regions = list(meta["regions"])
        regions, refusal = resolve_target(cmd, selection=self.selection,
                                          known_regions=all_regions, total=len(all_regions))
        if refusal is not None:
            return refusal
        target = _explore.describe_run_target(regions, total=len(all_regions))
        n_targets = len(all_regions) if regions is None else len(regions)

        out_dir = None
        if cmd.save:
            if not cmd.output_folder:
                return _refuse(cmd.kind, BAD_COMMAND,
                               "save=true needs an output_folder — a headless run has no dialog "
                               "to ask, and the output can be hundreds of GB")
            from pathlib import Path

            out_dir = Path(cmd.output_folder).expanduser() / f"{Path(self._path).name}.hcs"

        skipped: list[str] = []

        def on_error(region, fov, exc):
            skipped.append(str(region))
            logger.warning("SKIP well %s (fov %s): %s: %s", region, fov, type(exc).__name__, exc)

        region_op = cmd.operator in set(available_region_operators())
        with measure_run(cmd.operator, target or "no target", n_targets=n_targets) as run:
            run.note(surface=self.surface, save=cmd.save, acquisition=self._path)
            if cmd.save:
                manifest = write_plate(self.reader, out_dir, projector=cmd.operator,
                                       n_fovs=cmd.n_fovs, workers=cmd.workers, tiff=cmd.tiff,
                                       on_error=on_error, regions=regions,
                                       operator_kwargs=cmd.parameters or None)
                landed = int(manifest.get("n_fields_written") or 0)
                data = {"manifest": {k: (str(v) if hasattr(v, "__fspath__") else v)
                                     for k, v in manifest.items()}}
            else:
                if region_op:
                    stream = stitch_plate(self.reader, workers=1, operator=cmd.operator,
                                          n_fovs=None, on_error=on_error, regions=regions,
                                          **(cmd.parameters or {}))
                else:
                    stream = project_plate(self.reader, projector=cmd.operator, workers=cmd.workers,
                                           n_fovs=cmd.n_fovs, on_error=on_error, regions=regions)
                landed = 0
                for _region, _fov, _image in stream:
                    landed += 1          # PREVIEW headless: computed, counted, nothing retained
                data = {"n_fields": landed}
            # "It returned" is not "it worked". A run where every well raised (flat-field with no
            # profile is the routine case) still reaches here, because per-well fault isolation is
            # what keeps one bad file from aborting a plate — and the GUI already shipped a "✓"
            # over an empty plate once. Landed == 0 is a partial result however politely we got here.
            if landed == 0 and n_targets:
                run.finish(PARTIAL, f"produced nothing — all {n_targets} target(s) skipped")
            elif skipped:
                run.finish(PARTIAL, f"{len(set(skipped))} well(s) skipped")
            else:
                run.finish(OK)
            metrics = run
        data["n_landed"] = landed
        self.last_metrics = metrics.metrics
        data["skipped"] = sorted(set(skipped))
        data["metrics"] = metrics.metrics.as_dict() if metrics.metrics else None
        data["regions"] = regions
        data["target"] = target
        return _done(cmd.kind, metrics.metrics.line() if metrics.metrics else "done", **data)
