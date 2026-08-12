"""The timepoint control, one class shared by the plate and by every window.

``playback=False`` is a plain QSlider; ``playback=True`` wraps napari's dims playback.
"""
from __future__ import annotations

from typing import Callable, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget


class NoPlaybackError(RuntimeError):
    """``play()`` on a bar built without playback. A programming error, never a user gesture."""


class TimePointBar(QWidget):
    """A labelled slider over the acquisition's timepoints, optionally with playback.

    ``on_change(time_point)`` fires only for a user gesture (a playback step counts as one),
    never for a programmatic :meth:`set_count` or :meth:`set_time_point`.
    """

    def __init__(self, on_change: Optional[Callable[[int], None]] = None, parent=None,
                 playback: bool = False) -> None:
        super().__init__(parent)
        self._on_change = on_change
        self._count = 1
        self._muted = False          # True while WE move the slider, so we do not echo ourselves
        self._playback = None

        self.label = QLabel("time_point 1 / 1")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.label)

        if playback:
            self._playback = self._build_playback()
            # napari's own scrollbar, exposed under the name the plain skin uses.
            self.slider = self._playback.dim_slider.slider
            row.addWidget(self._playback, 1)
        else:
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(0)
            self.slider.setPageStep(1)
            self.slider.valueChanged.connect(self._on_slider_moved)
            row.addWidget(self.slider, 1)

        self.set_count(1)

    def _build_playback(self):
        """napari's dims playback, on the time axis. Imported here so a plain bar costs no napari."""
        from squidxplorer._region_nav import AxisPlayback

        class _TimeAxisPlayback(AxisPlayback):
            def __init__(self, bar):
                self._bar = bar
                super().__init__(axis_label="time_point", noun="timepoint", parent=bar)
                self.setToolTip(
                    "Step through TIMEPOINTS of this acquisition.\nPress play to walk them; "
                    "right-click play for frames per second and loop mode."
                )

            def _on_step(self, index: int) -> None:
                self._bar._on_slider_moved(int(index))

        return _TimeAxisPlayback(self)

    @property
    def count(self) -> int:
        """How many timepoints this acquisition has."""
        return self._count

    @property
    def time_point(self) -> int:
        if self._playback is not None:
            return int(self._playback.index)
        return int(self.slider.value())

    @property
    def playback(self):
        """The playback engine, or ``None`` on a bar that cannot play. Never a stub."""
        return self._playback

    def set_count(self, n_time_points: int) -> None:
        """Size the bar to the acquisition, and hide it when there is nothing to navigate."""
        self._count = max(1, int(n_time_points or 1))
        self._muted = True
        try:
            if self._playback is not None:
                self._playback.set_count(self._count)
            else:
                self.slider.setMaximum(self._count - 1)
                if self.slider.value() > self._count - 1:
                    self.slider.setValue(0)
        finally:
            self._muted = False
        # Hidden rather than never built, so every call site stays unconditional.
        self.setVisible(self._count > 1)
        self._refresh_label()

    def set_time_point(self, time_point: int) -> None:
        """Move the bar WITHOUT calling back. For following someone else, not for a gesture."""
        want = self._clamp(time_point)
        self._muted = True
        try:
            if self._playback is not None:
                self._playback._follow(want)
            else:
                self.slider.setValue(want)
        finally:
            self._muted = False
        self._refresh_label()

    def set_time_point_from_user(self, time_point: int) -> None:
        """As if the user had dragged it: moves the bar AND fires ``on_change``."""
        want = self._clamp(time_point)
        if self._playback is not None:
            if self._playback.index == want:
                # A gesture that changes nothing must not reload a mosaic.
                return
            self._playback.set_index_from_user(want)
            return
        self.slider.setValue(want)

    def frame_done(self) -> None:
        """This timepoint is on screen; playback may request the next. A no-op without playback."""
        if self._playback is not None:
            self._playback.frame_done()

    @property
    def is_playing(self) -> bool:
        return bool(self._playback is not None and self._playback.is_playing)

    @property
    def fps(self) -> float:
        return float(self._playback.fps) if self._playback is not None else 0.0

    def play(self, fps: Optional[float] = None) -> None:
        if self._playback is None:
            raise NoPlaybackError(
                "this timepoint bar was built without playback, so there is nothing to start. "
                "The plate's bar is deliberately one of them: its preview cache is not keyed by "
                "timepoint, so animating it would show timepoint 0's pixels under another "
                "timepoint's label."
            )
        self._playback.play(fps)

    def stop(self) -> None:
        if self._playback is not None:
            self._playback.stop()

    def shutdown(self) -> None:
        """Stop and JOIN napari's animation thread. Qt aborts the process without this."""
        if self._playback is not None:
            self._playback.shutdown()

    def on_problem(self, sink: Callable[[str], None]) -> None:
        """Where a playback refusal or stall is shown to the USER."""
        if self._playback is not None:
            self._playback.on_problem(sink)

    def _clamp(self, time_point: int) -> int:
        return max(0, min(int(time_point), self._count - 1))

    def _on_slider_moved(self, value: int) -> None:
        self._refresh_label()
        if self._muted or self._on_change is None:
            return
        self._on_change(int(value))

    def _refresh_label(self) -> None:
        # 1-based for the human, 0-based for the data.
        self.label.setText(f"time_point {self.time_point + 1} / {self._count}")
