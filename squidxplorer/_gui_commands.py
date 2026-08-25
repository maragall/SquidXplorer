"""The GUI half of the one command surface: a :class:`WindowExecutor` over ``PlateWindow``.

The vocabulary, result type, refusal codes and scope resolution live in
:mod:`squidxplorer._command`, shared with the headless executor. A GUI run returns
``status="started"`` (it is a QThread); completion arrives on the metrics/activity log.
"""

from __future__ import annotations


from squidxplorer import _run_scope
# The operator vocabulary comes from the Qt-free registry, never from the window.
from squidxplorer._operations import operator_label, runnable_operators
from squidxplorer._command import (
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
    UNAVAILABLE_OPERATOR,
    UNKNOWN_OPERATOR,
    _done,
    _refuse,
    _started,
    resolve_target,
)

from squidxplorer._logpane import get_logger

logger = get_logger("command")


class WindowExecutor:
    """Answers the shared command vocabulary against a live ``PlateWindow``."""

    surface = "gui"

    def __init__(self, window) -> None:
        self._window = window

    def _meta(self):
        return getattr(self._window, "_meta", None)

    def _has_acquisition(self) -> bool:
        return getattr(self._window, "_reader", None) is not None and self._meta() is not None

    # Introspection is shared with the engine, so both surfaces answer identically.
    def do_list_operators(self, cmd: ListOperators) -> CommandResult:
        from squidxplorer._command import EngineExecutor

        return EngineExecutor.do_list_operators(self, cmd)   # pure registry read; no window state

    def do_metrics(self, cmd: Metrics) -> CommandResult:
        from squidxplorer._command import EngineExecutor

        return EngineExecutor.do_metrics(self, cmd)          # reads the process-wide METRICS log

    def do_describe(self, cmd: Describe) -> CommandResult:
        w = self._window
        meta = self._meta()
        if not self._has_acquisition():
            return _refuse(cmd.kind, NO_ACQUISITION,
                           "no acquisition is open in the window - drop one, or run "
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
                     # the live pieces a run's scope is resolved from
                     selection=list(getattr(w, "_selected_regions", []) or []),
                     current_region=getattr(w, "_current_well", None),
                     scopes=list(_run_scope.RUN_SCOPES),
                     busy=self._busy())

    def do_open_acquisition(self, cmd: OpenAcquisition) -> CommandResult:
        self._window.ingest(cmd.path)
        if not self._has_acquisition():
            # Surface ingest's own readout sentence as the refusal.
            msg = getattr(self._window, "_readout", None)
            return _refuse(cmd.kind, NO_ACQUISITION,
                           (msg.text() if msg is not None else "the window did not open that path"))
        regions = list(self._meta()["regions"])
        return _done(cmd.kind, f"opened {cmd.path} - {len(regions)} region(s)",
                     path=cmd.path, n_regions=len(regions), regions=regions)

    def _busy(self) -> bool:
        return _run_scope.operator_busy(getattr(self._window, "_worker", None),
                                      getattr(self._window, "_retired", []) or [])

    def do_run_operator(self, cmd: RunOperator) -> CommandResult:
        """Resolve the target once (shared resolver), then drive the window's own ``run_operator``. Returns ``started``."""
        w = self._window
        if not self._has_acquisition():
            return _refuse(cmd.kind, NO_ACQUISITION,
                           "no acquisition is open in the window - open one first")
        if cmd.operator not in runnable_operators():
            return _refuse(cmd.kind, UNKNOWN_OPERATOR,
                           f"{cmd.operator!r} is not a runnable operator - this window can run: "
                           f"{', '.join(runnable_operators())}", available=runnable_operators())
        # Registered but not installable: refuse the same way the headless executor does.
        from squidxplorer import operator_available

        avail_ok, avail_why = operator_available(cmd.operator)
        if not avail_ok:
            return _refuse(cmd.kind, UNAVAILABLE_OPERATOR, avail_why, operator=cmd.operator)
        if self._busy():
            return _refuse(cmd.kind, BUSY,
                           "a run is already in flight - stop it or let it finish first")

        # One resolution, by the shared helper, against the window's live state.
        all_regions = list(self._meta()["regions"])
        regions, refusal = resolve_target(
            cmd, selection=getattr(w, "_selected_regions", []),
            current_region=getattr(w, "_current_well", None),
            known_regions=all_regions, total=len(all_regions))
        if refusal is not None:
            return refusal
        # Pass the full explicit list so the window never re-consults its scope selector.
        run_regions = all_regions if regions is None else regions

        label = operator_label(cmd.operator)
        target = _run_scope.describe_run_target(regions, total=len(all_regions))

        worker_before = getattr(w, "_worker", None)
        # save=True needs an output_folder, or the window would raise a file dialog.
        w.run_operator(cmd.operator, out_parent=cmd.output_folder, regions=run_regions,
                       save=cmd.save,
                       operator_kwargs=cmd.parameters or None)
        started = getattr(w, "_worker", None) is not worker_before and getattr(w, "_worker", None) is not None
        readout = getattr(w, "_readout", None)
        message = readout.text() if readout is not None else (target or label)
        if not started:
            # run_operator declined inside the window; the reason is in the readout.
            from squidxplorer._command import FAILED

            return _refuse(cmd.kind, FAILED, message or "the run did not start")
        return _started(cmd.kind, message or f"{label}: {target}",
                        operator=cmd.operator, regions=regions, target=target, save=cmd.save)

    def do_stop_run(self, cmd: StopRun) -> CommandResult:
        """Stop the run in flight. Nothing running is a named refusal, not a cheerful no-op."""
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
    """Build the window's command bus; the window only calls this and holds the returned bus."""
    from squidxplorer._command import CommandBus

    return CommandBus(WindowExecutor(window))
