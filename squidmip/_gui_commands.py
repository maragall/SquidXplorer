"""The GUI half of the ONE command surface: a :class:`WindowExecutor` over ``PlateWindow``.

The command vocabulary, the result type, the refusal codes and the scope resolution all live in
:mod:`squidmip._command` and are SHARED with the headless :class:`~squidmip._command.EngineExecutor`
that the CLI drives. This module adds nothing to that vocabulary; it only teaches the window to
answer it. That is the whole point — Julio: "make sure that our GUI has an amazing API so that AI
can interact with it" — an agent, a test, or a script says the same command to the GUI it says to
the CLI, and the GUI is a CALLER of the layer rather than a second surface beside it.

WHY THIS IS A SEPARATE MODULE FROM ``_viewer``
----------------------------------------------
``_viewer.py`` is a 7,000-line module several agents edit at once. Every line of the translation
that CAN live outside it does, so the window's own diff is only the two lines that build a bus and
hand it out. The executor reads the window's public state (its selection, its open region, its
parked subset) and calls its public ``run_operator`` — it does not reach past the methods the GUI
itself uses, so it cannot drift from what the buttons do.

``started`` IS NOT ``completed``
--------------------------------
A GUI run is a QThread — it MUST NOT block the event loop, so the honest thing the command can
return is that the run BEGAN (``status="started"``). The end of it arrives on the metrics/activity
log, which is the same record the headless run writes, so a caller that needs completion watches
one place regardless of which surface launched the work. Papering this over — pretending a GUI
command completes synchronously — would either freeze the window or lie about the result.

THE EDGES, NAMED
----------------
``stop_run`` with nothing running, a run refused by the window's own disk/busy guard: these come
back as named refusals, not as a silent no-op or a status-label sentence the caller cannot read.
Where the window's behaviour cannot yet be expressed as a command, the executor says so
(``not_supported_here``) rather than faking it — an honest edge beats a facade.
"""

from __future__ import annotations

import logging
from typing import Optional

from squidmip import _explore
from squidmip._command import (
    BUSY,
    CommandResult,
    Describe,
    ListOperators,
    Metrics,
    NO_ACQUISITION,
    NO_RUN,
    OpenAcquisition,
    RunOperator,
    StopRun,
    UNKNOWN_OPERATOR,
    _done,
    _refuse,
    _started,
    resolve_target,
)

logger = logging.getLogger("squidmip.command")


class WindowExecutor:
    """Answers the shared command vocabulary against a live ``PlateWindow``.

    Constructed with the window; the window builds one and wraps it in a
    :class:`~squidmip._command.CommandBus` (``window.commands``). Everything an agent can do to the
    GUI, it does through that bus.
    """

    surface = "gui"

    def __init__(self, window) -> None:
        self._window = window

    # -- state the window already owns -------------------------------------------------
    def _meta(self):
        return getattr(self._window, "_meta", None)

    def _has_acquisition(self) -> bool:
        return getattr(self._window, "_reader", None) is not None and self._meta() is not None

    # -- introspection: SHARED with the engine, so both surfaces answer identically ----
    def do_list_operators(self, cmd: ListOperators) -> CommandResult:
        from squidmip._command import EngineExecutor

        return EngineExecutor.do_list_operators(self, cmd)   # pure registry read; no window state

    def do_metrics(self, cmd: Metrics) -> CommandResult:
        from squidmip._command import EngineExecutor

        return EngineExecutor.do_metrics(self, cmd)          # reads the process-wide METRICS log

    def do_describe(self, cmd: Describe) -> CommandResult:
        w = self._window
        meta = self._meta()
        if not self._has_acquisition():
            return _refuse(cmd.kind, NO_ACQUISITION,
                           "no acquisition is open in the window — drop one, or run "
                           "open_acquisition")
        regions = list(meta["regions"])
        return _done(cmd.kind, f"{len(regions)} region(s) open in the window",
                     surface=self.surface,
                     path=getattr(w, "_acq_name", None),
                     regions=regions, n_regions=len(regions),
                     channels=[c["name"] for c in meta["channels"]],
                     frame_shape=list(meta["frame_shape"]),
                     pixel_size_um=meta.get("pixel_size_um"),
                     wellplate_format=str(meta.get("wellplate_format", "")),
                     # the THREE live pieces a run's scope is resolved from — the same state the
                     # scope selector reads, exposed so an agent can build a target it can predict
                     selection=list(getattr(w, "_selected_regions", []) or []),
                     current_region=getattr(w, "_current_well", None),
                     parked_subset=list(w.parked_subset()) if hasattr(w, "parked_subset") else [],
                     scopes=list(_explore.RUN_SCOPES),
                     busy=self._busy())

    # -- opening -----------------------------------------------------------------------
    def do_open_acquisition(self, cmd: OpenAcquisition) -> CommandResult:
        self._window.ingest(cmd.path)
        if not self._has_acquisition():
            # ingest refuses in the readout (e.g. "this is already a written plate"); surface that
            # sentence as the refusal rather than a bare failure.
            msg = getattr(self._window, "_readout", None)
            return _refuse(cmd.kind, NO_ACQUISITION,
                           (msg.text() if msg is not None else "the window did not open that path"))
        regions = list(self._meta()["regions"])
        return _done(cmd.kind, f"opened {cmd.path} — {len(regions)} region(s)",
                     path=cmd.path, n_regions=len(regions), regions=regions)

    # -- running -----------------------------------------------------------------------
    def _busy(self) -> bool:
        return _explore.operator_busy(getattr(self._window, "_worker", None),
                                      getattr(self._window, "_retired", []) or [])

    def do_run_operator(self, cmd: RunOperator) -> CommandResult:
        """Resolve the target ONCE (shared resolver), then drive the window's own ``run_operator``.

        Returns ``started``: the run is a QThread. The command's scope decision is turned into an
        EXPLICIT region list here so the window's ``run_operator`` never re-consults its scope
        selector — one resolution, not two. A whole-dataset run becomes the full ordered region
        list, which the engine treats as the whole plate.
        """
        from squidmip._viewer import operator_label, runnable_operators

        w = self._window
        if not self._has_acquisition():
            return _refuse(cmd.kind, NO_ACQUISITION,
                           "no acquisition is open in the window — open one first")
        if cmd.operator not in runnable_operators():
            return _refuse(cmd.kind, UNKNOWN_OPERATOR,
                           f"{cmd.operator!r} is not a runnable operator — this window can run: "
                           f"{', '.join(runnable_operators())}", available=runnable_operators())
        if self._busy():
            return _refuse(cmd.kind, BUSY,
                           "a run is already in flight — stop it or let it finish first")

        # ONE resolution, by the shared helper, against the window's live state — exactly the
        # state the GUI's own scope selector reads.
        all_regions = list(self._meta()["regions"])
        regions, refusal = resolve_target(
            cmd, selection=getattr(w, "_selected_regions", []),
            current_region=getattr(w, "_current_well", None),
            parked_subset=w.parked_subset() if hasattr(w, "parked_subset") else [],
            known_regions=all_regions, total=len(all_regions))
        if refusal is not None:
            return refusal
        # The window's run_operator re-consults its scope selector when regions is None; passing
        # the full explicit list makes the command's decision win and keeps the run to exactly the
        # wells it resolved. (This is the subset code path; its output equals the whole plate.)
        run_regions = all_regions if regions is None else regions

        label = operator_label(cmd.operator)
        target = _explore.describe_run_target(regions, total=len(all_regions))

        worker_before = getattr(w, "_worker", None)
        # save=True needs an output_folder — the window would otherwise raise a file dialog, which
        # a programmatic caller cannot answer. Pass it straight through as the dialog's answer.
        w.run_operator(cmd.operator, out_parent=cmd.output_folder, regions=run_regions,
                       save=cmd.save,
                       operator_kwargs=cmd.parameters or None)
        started = getattr(w, "_worker", None) is not worker_before and getattr(w, "_worker", None) is not None
        readout = getattr(w, "_readout", None)
        message = readout.text() if readout is not None else (target or label)
        if not started:
            # run_operator declined inside the window (disk guard, an empty resolved set, a
            # cancelled save dialog). The reason is in the readout; surface it as a refusal rather
            # than reporting a run that never began.
            from squidmip._command import FAILED

            return _refuse(cmd.kind, FAILED, message or "the run did not start")
        return _started(cmd.kind, message or f"{label}: {target}",
                        operator=cmd.operator, regions=regions, target=target, save=cmd.save)

    def do_stop_run(self, cmd: StopRun) -> CommandResult:
        """Stop the run in flight. Nothing running is a NAMED refusal, not a cheerful no-op."""
        w = self._window
        if not self._busy():
            return _refuse(cmd.kind, NO_RUN, "no run is in flight to stop")
        stopper = getattr(w, "_stop_worker", None)
        if callable(stopper):
            stopper()
        elif getattr(w, "_worker", None) is not None:
            w._worker.stop()
        return _done(cmd.kind, "asked the run to stop")


def install_command_bus(window):
    """Build the window's command bus. The ENTIRE footprint this feature has in ``PlateWindow``
    is a call to this and holding the returned bus — everything else is in this module."""
    from squidmip._command import CommandBus

    return CommandBus(WindowExecutor(window))
