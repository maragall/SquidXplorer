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

**PLAYBACK, 2026-08-05.** The second bullet said what the real operation on this axis is, and now
half of it exists: ``TimePointBar(playback=True)`` walks the timepoints with a play button, an fps
control and loop modes.

None of that machinery is new and none of it is a ``QTimer``. It is
:class:`squidmip._region_nav.AxisPlayback` — napari's own ``Dims`` + ``QtDims``, already driving the
REGION slider, with the axis passed in. A hand-rolled timer gets two things wrong that this does
not: it runs on the GUI event loop (napari's ``AnimationThread`` does not), and it free-runs, so at
10 fps it queues ten region re-reads for every one that completes and playback degrades into a
backlog that grows the longer you watch. napari drops the frames it cannot draw in time instead,
which is the CORRECT behaviour for an axis whose every step costs a mosaic load.

**Why one class with a ``playback`` flag rather than two widgets.** `tests/test_plate_contract.py`
pins that the plate and the windows use the SAME type, for the reason at the top of this file, and
that has not changed. What differs is which SKIN the one class wears:

* ``playback=False`` (the plate) — the plain ``QSlider``, byte-identical to what shipped in 4b.
* ``playback=True`` (a region window) — napari's dims widget IS the slider, so the position has
  exactly ONE owner (napari's ``Dims``) rather than a QSlider and a Dims hand-synced.

The plate does NOT get playback. The reason USED to be a correctness blocker: ``_PreviewWorker``
read frame 0 whatever the bar said, and its persistent cell cache was keyed ``(token, region)``
with no timepoint, so an animated plate would have served timepoint 0's pixels labelled timepoint
1. Both were fixed on 2026-08-05 — the cell is keyed ``(token, t, region)`` and the preview reads
the bar's timepoint (`_platecache.py`, `docs/plate-contract.md`) — and a revisited timepoint is now
a cache hit rather than a re-read, which is exactly what playback would lean on.

So what is left is a PRODUCT decision, not a defect: nobody has asked the plate to animate, and a
button that exists because it became possible is not a reason to ship one. ``playback=False``
stands, and ``play()`` on the plate's bar still raises rather than no-ops.
"""
from __future__ import annotations

from typing import Callable, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget


class NoPlaybackError(RuntimeError):
    """``play()`` on a bar built without playback. A programming error, never a user gesture."""


class TimePointBar(QWidget):
    """A labelled slider over the acquisition's timepoints, optionally with playback.

    ``on_change(time_point)`` fires only for a USER gesture, never for a programmatic
    :meth:`set_count` or :meth:`set_time_point`. That distinction is the same one the contrast sink
    makes elsewhere in this codebase, and for the same reason: treating our own write as a user
    gesture is how a control ends up fighting the thing it is supposed to follow. A PLAYBACK step
    counts as a gesture: it must reload the mosaic exactly as a drag does, and there is one loading
    path, not a second one for animation.
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
            # napari's OWN scrollbar, exposed under the name the plain skin uses. It is the same
            # position control, so callers and tests that ask a bar what its slider says get an
            # answer in both skins rather than having to know which one they were handed.
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
        """napari's dims playback, on the time axis. Imported here, not at module scope.

        This module is deliberately importable without napari — `tests/test_time_point_bar.py`
        exists because the GUI-level test file for this control aborts the interpreter on import,
        and that file can only stay light if the control stays light. A bar built WITHOUT playback
        must therefore cost no napari import at all.
        """
        from squidmip._region_nav import AxisPlayback

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

    # -- state ---------------------------------------------------------------------------

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
        """The playback engine, or ``None`` on a bar that cannot play. Never a stub.

        ``None`` rather than a do-nothing object so that "this bar cannot play" is a question with
        an answer, instead of a play button that looks pressed and moves nothing.
        """
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
        # Hidden rather than never built: see the module docstring. Every call site stays
        # unconditional, and `isHidden()` becomes the one honest question about it.
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
                # napari's Dims does not re-announce a step it is already on, and neither does the
                # QSlider skin: a gesture that changes nothing must not reload a mosaic.
                return
            self._playback.set_index_from_user(want)
            return
        self.slider.setValue(want)

    # -- playback (only on a bar built with it) ------------------------------------------

    def frame_done(self) -> None:
        """This timepoint is on screen; playback may request the next. A no-op without playback.

        Unconditional at the call site on purpose, exactly like the hidden-not-absent rule above:
        the loader says "the frame landed" and does not have to ask whether anybody is animating.
        """
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

    # -- internals -----------------------------------------------------------------------

    def _clamp(self, time_point: int) -> int:
        return max(0, min(int(time_point), self._count - 1))

    def _on_slider_moved(self, value: int) -> None:
        self._refresh_label()
        if self._muted or self._on_change is None:
            return
        self._on_change(int(value))

    def _refresh_label(self) -> None:
        # 1-based for the human, 0-based for the data. Squid's timepoint folders are 0-based, so the
        # label says which of how many rather than pretending the index is the name.
        self.label.setText(f"time_point {self.time_point + 1} / {self._count}")
