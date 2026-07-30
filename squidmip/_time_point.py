"""The timepoint control, shared by the plate and by every window.

Task 4b, 2026-07-29. This closes a SILENT CORRECTNESS BUG rather than adding a feature.

Every consumer read timepoint 0 and presented it as the whole dataset, so a 40-timepoint plate was
indistinguishable from a 1-timepoint plate, with no error anywhere. The zarr we write is TCZYX and
holds every frame, `reader.read` has always taken a timepoint, and `project_well` threads one. The
loss was entirely in three preview reads that hardcoded ``[0, :, 0]``. Nothing caught it because
every fixture on this machine was ``Nt = 1``, so the bug was invisible by construction until
`multi_time_point_dataset` existed.

**Why one widget class and not two.** `tests/test_time_point_slider.py` pins that the plate and the
windows use the SAME type. A second implementation is how the two drift into disagreeing about what
timepoint you are looking at, which is worse than having no slider: you would be comparing two
frames and told they were one. The plate and each window own their own INSTANCE (each window
navigates independently, which is the whole point of the decentralization) but there is one
definition of what the control is.

**The name is Squid's.** `time_point`, not `t`. The naming law (Julio, 2026-07-29): Squid models the
physical world, we model the processing of what it recorded, and the language must agree. A
timepoint is a physical thing that exists in the microscope, so it takes Squid's exact spelling.
`squidmip/_address.py` uses `time_point` for the same reason.

**Hidden, not absent, when there is one timepoint.** A slider with a single position is clutter, but
building it and hiding it keeps every call site unconditional, so no caller has to ask whether the
control exists before talking to it. `isHidden()` is then the honest question, and it is what the
tests ask.

**What this does NOT do**, and both are deliberate decisions recorded elsewhere:

* The timepoint is not part of the result cache key. Julio: besides preview, nobody doing HCS
  applies an operator to one timepoint, and computing one alone buys latency rather than throughput
  since all of them are needed eventually. `Extent` can already express it if that changes.
* There is no time-reduction operator. Collapsing time destroys what the time was acquired for; the
  real operation on that axis is playback and export.
"""
from __future__ import annotations

from typing import Callable, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget


class TimePointBar(QWidget):
    """A labelled slider over the acquisition's timepoints.

    ``on_change(time_point)`` fires only for a USER gesture, never for a programmatic
    :meth:`set_count` or :meth:`set_time_point`. That distinction is the same one the contrast sink
    makes elsewhere in this codebase, and for the same reason: treating our own write as a user
    gesture is how a control ends up fighting the thing it is supposed to follow.
    """

    def __init__(self, on_change: Optional[Callable[[int], None]] = None, parent=None) -> None:
        super().__init__(parent)
        self._on_change = on_change
        self._count = 1
        self._muted = False          # True while WE move the slider, so we do not echo ourselves

        self.label = QLabel("time_point 1 / 1")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setPageStep(1)
        self.slider.valueChanged.connect(self._on_slider_moved)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.label)
        row.addWidget(self.slider, 1)

        self.set_count(1)

    # -- state ---------------------------------------------------------------------------

    @property
    def count(self) -> int:
        """How many timepoints this acquisition has."""
        return self._count

    @property
    def time_point(self) -> int:
        return int(self.slider.value())

    def set_count(self, n_time_points: int) -> None:
        """Size the bar to the acquisition, and hide it when there is nothing to navigate."""
        self._count = max(1, int(n_time_points or 1))
        self._muted = True
        try:
            self.slider.setMaximum(self._count - 1)
            if self.slider.value() > self._count - 1:
                self.slider.setValue(0)
        finally:
            self._muted = False
        # Hidden rather than never built: see the module docstring. Every call site stays
        # unconditional, and `isHidden()` becomes the one honest question about it.
        self.setVisible(self._count > 1)
        self._refresh_label()

    def set_time_point(self, time_point: int) -> None:
        """Move the bar WITHOUT calling back. For following someone else, not for a gesture."""
        self._muted = True
        try:
            self.slider.setValue(max(0, min(int(time_point), self._count - 1)))
        finally:
            self._muted = False
        self._refresh_label()

    def set_time_point_from_user(self, time_point: int) -> None:
        """As if the user had dragged it: moves the bar AND fires ``on_change``."""
        self.slider.setValue(max(0, min(int(time_point), self._count - 1)))

    # -- internals -----------------------------------------------------------------------

    def _on_slider_moved(self, value: int) -> None:
        self._refresh_label()
        if self._muted or self._on_change is None:
            return
        self._on_change(int(value))

    def _refresh_label(self) -> None:
        # 1-based for the human, 0-based for the data. Squid's timepoint folders are 0-based, so the
        # label says which of how many rather than pretending the index is the name.
        self.label.setText(f"time_point {self.time_point + 1} / {self._count}")
