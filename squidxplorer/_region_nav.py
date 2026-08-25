"""Region navigation: the single owner of "which region is current", plus napari's own playback
machinery reused to walk it (and, via ``AxisPlayback``, the time axis too)."""

from __future__ import annotations

from typing import Callable, Optional, Sequence


class RegionCursor:
    """Which region is current. Not a QObject: a Qt Signal swallows exceptions raised in a
    slot, and this class must never fail to move part of the UI without saying so."""

    def __init__(self) -> None:
        self._order: list[str] = []
        self._index: Optional[int] = None
        self._activated = False
        self._subs: list[Callable[[int, str], None]] = []
        self._order_subs: list[Callable[[list], None]] = []
        self._problem: Optional[Callable[[str], None]] = None

    # -- reading ------------------------------------------------------------------------
    @property
    def regions(self) -> list[str]:
        return list(self._order)

    @property
    def count(self) -> int:
        return len(self._order)

    @property
    def index(self) -> Optional[int]:
        return self._index

    @property
    def region(self) -> Optional[str]:
        if self._index is None:
            return None
        return self._order[self._index]

    @property
    def activated(self) -> bool:
        """True once the user explicitly opened a region (double-click), not just because
        something had to be displayed; ``_selection_regions`` scopes operator runs to this."""
        return self._activated

    def position_of(self, region: str) -> Optional[int]:
        try:
            return self._order.index(region)
        except ValueError:
            return None

    # -- subscribing --------------------------------------------------------------------
    def subscribe(self, callback: Callable[[int, str], None]) -> None:
        """``callback(index, region)`` whenever the current region CHANGES."""
        self._subs.append(callback)

    def subscribe_order(self, callback: Callable[[list], None]) -> None:
        """``callback(regions)`` whenever the region order is re-scoped (separate from
        ``subscribe``: a re-scope may keep the same region, but the slider's length must still
        change)."""
        self._order_subs.append(callback)

    def on_problem(self, sink: Callable[[str], None]) -> None:
        """Where a failing subscriber is reported. Without one, the failure is re-raised."""
        self._problem = sink

    def _announce(self) -> None:
        idx, reg = self._index, self.region
        if idx is None or reg is None:
            return
        failures: list[str] = []
        for cb in list(self._subs):
            try:
                cb(idx, reg)
            except Exception as exc:               # noqa: BLE001 - reported, never swallowed
                if self._problem is None:
                    raise
                failures.append(f"{type(exc).__name__}: {exc}")
        for text in failures:
            # never log-and-continue: a failed subscriber must not silently desync the UI
            self._problem(f"region navigation: a subscriber failed - {text}")   # type: ignore[misc]

    # -- moving -------------------------------------------------------------------------
    def set_order(self, order: Sequence[str]) -> None:
        """Re-scope the cursor to *order*, staying on the same region if it survives (snapping
        to 0 would move the red frame off what the user is looking at)."""
        was = self.region
        self._order = [str(r) for r in order]
        if not self._order:
            self._index = None
            self._activated = False
            self._announce_order()
            return
        if was is not None and was in self._order:
            self._index = self._order.index(was)
            self._announce_order()
            return                                  # same region: the frame must NOT move
        self._index = 0
        self._activated = False
        self._announce_order()
        self._announce()

    def _announce_order(self) -> None:
        for cb in list(self._order_subs):
            cb(list(self._order))

    def set_index(self, index: int) -> None:
        """Move to *index*. Raises on out-of-range rather than clamping (clamping would
        silently desync caller and cursor)."""
        if not self._order:
            raise IndexError("no regions loaded; nothing to select")
        i = int(index)
        if not (0 <= i < len(self._order)):
            raise IndexError(f"region index {i} out of range 0..{len(self._order) - 1}")
        if i == self._index:
            return                                  # no re-announce: subscribers reload mosaics
        self._index = i
        self._announce()

    def set_region(self, region: str) -> None:
        pos = self.position_of(str(region))
        if pos is None:
            raise KeyError(f"{region!r} is not in the current region order")
        self.set_index(pos)

    def step(self, delta: int) -> None:
        """Move by *delta*, wrapping. Wrapping is what makes playback loop."""
        if not self._order:
            raise IndexError("no regions loaded; nothing to step through")
        base = 0 if self._index is None else self._index
        self.set_index((base + int(delta)) % len(self._order))

    def activate(self, region: str) -> None:
        """The user explicitly opened *region* (a double-click on the plate)."""
        self.set_region(region)
        self._activated = True

    def deactivate(self) -> None:
        """Nothing is explicitly open any more. Does NOT navigate — the frame stays put."""
        self._activated = False


# napari's Qt access path has moved before; a missing symbol must fail loudly at construction,
# not become a slider that appears and does nothing.

REQUIRED_PLAYBACK_BINDINGS: tuple[tuple[str, str], ...] = (
    ("napari.components", "Dims"),
    ("napari._qt.widgets.qt_dims", "QtDims"),
    ("napari._qt.widgets.qt_dims_slider", "QtDimSliderWidget"),
    ("napari._qt.widgets.qt_dims_slider", "AnimationThread"),
)


class NapariPlaybackError(RuntimeError):
    """napari's playback machinery has moved, been renamed, or been removed."""


def verify_playback_bindings(modules: Optional[dict] = None) -> None:
    """Fail loudly if the playback API we drive is missing. ``modules`` is a test seam."""
    import importlib

    missing: list[str] = []
    for dotted, attr in REQUIRED_PLAYBACK_BINDINGS:
        try:
            mod = modules[dotted] if modules and dotted in modules else importlib.import_module(dotted)
        except Exception as exc:                    # pragma: no cover - reported, not swallowed
            missing.append(f"{dotted} (import failed: {exc!r})")
            continue
        if not hasattr(mod, attr):
            missing.append(f"{dotted}.{attr}")
    if missing:
        raise NapariPlaybackError(
            "napari's playback machinery has moved under us, so the region slider cannot "
            "play. Missing: " + ", ".join(missing) + "\n"
            "This is a hard failure on purpose - a play button that silently does nothing is "
            "the failure mode this control was built to replace."
        )


#: napari refuses ndisplay < 2; one region axis plus two dummy displayed axes gets exactly one
#: slider.
_DUMMY_DISPLAYED_AXES = 2

#: napari's own default; changeable via its own fps popup (right-click play).
DEFAULT_FPS = 10


def _napari_stylesheet() -> str:
    """napari's own dark stylesheet, so the embedded dims widget matches; returns "" (cosmetic
    only) if napari has moved the accessor."""
    try:
        from napari.qt import get_stylesheet

        return get_stylesheet("dark")
    except Exception:                               # noqa: BLE001 - cosmetic only
        return ""


try:                                                # pragma: no cover - import shape only
    from qtpy.QtWidgets import QWidget as _QWidgetBase
except Exception:                                   # pragma: no cover
    _QWidgetBase = object                           # type: ignore[assignment,misc]


class AxisPlayback(_QWidgetBase):                   # type: ignore[misc,valid-type]
    """napari's dims slider, play button, fps popup and off-thread animation, over one axis.
    Generic here: the ``Dims``/``QtDims`` machinery, the frame gate, the stall watchdog. Left to
    the subclass: what a step means (``_on_step``) and why playing might be refused
    (``_refusal``)."""

    #: How long playback may sit gated before we assume the renderer is never coming back. Not
    #: a speed limit: region loads have been measured up to ~20s under load, well below this.
    STALL_GRACE_S = 180.0

    def __init__(self, *, axis_label: str, noun: str, parent=None) -> None:
        verify_playback_bindings()

        from napari.components import Dims
        from napari._qt.widgets.qt_dims import QtDims
        from qtpy.QtWidgets import QHBoxLayout

        super().__init__(parent)
        self._noun = str(noun)
        self._problem: Optional[Callable[[str], None]] = None
        self._echo = False                          # guards the owner -> widget -> owner loop
        self._stalled_since: Optional[float] = None

        self._dims = Dims(
            ndim=1 + _DUMMY_DISPLAYED_AXES,
            ndisplay=_DUMMY_DISPLAYED_AXES,
            range=((0, 0, 1), (0, 1, 1), (0, 1, 1)),
            axis_labels=(str(axis_label), "y", "x"),
        )
        self._qt_dims = QtDims(self._dims)
        self._dims.events.current_step.connect(self._on_dims_step)

        from qtpy.QtCore import QTimer

        self._stall_timer = QTimer(self)
        self._stall_timer.setInterval(1000)
        self._stall_timer.timeout.connect(self._watch_for_stall)

        # napari's stylesheet applies to napari's own QMainWindow, not a QtDims parented into
        # ours, so ask napari for it explicitly rather than hand-write a copy.
        self._qt_dims.setStyleSheet(_napari_stylesheet())

        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(8, 2, 8, 2)
        self._row.setSpacing(8)
        self._row.addWidget(self._qt_dims, 1)

    # -- the napari widgets, exposed so tests assert they really are napari's ------------
    @property
    def qt_dims(self):
        return self._qt_dims

    @property
    def dim_slider(self):
        """napari's ``QtDimSliderWidget`` for the axis this control walks."""
        return self._qt_dims.slider_widgets[0]

    @property
    def fps(self) -> float:
        return float(self.dim_slider.fps)

    @property
    def index(self) -> int:
        return int(self._dims.current_step[0])

    @property
    def count(self) -> int:
        rng = self._dims.range[0]
        return int(rng.stop) + 1 if hasattr(rng, "stop") else int(rng[1]) + 1

    @property
    def is_playing(self) -> bool:
        return bool(self._qt_dims.is_playing)

    def on_problem(self, sink: Callable[[str], None]) -> None:
        """Where a refusal is shown to the USER. Never log-and-continue."""
        self._problem = sink

    def _say(self, text: str) -> None:
        if self._problem is not None:
            self._problem(text)

    # -- the axis ------------------------------------------------------------------------
    def set_count(self, n: int) -> None:
        n = max(0, int(n))
        top = max(0, n - 1)
        self._echo = True
        try:
            self._dims.range = ((0, top, 1), (0, 1, 1), (0, 1, 1))
            if self.index > top:
                self._dims.set_current_step(0, top)
        finally:
            self._echo = False
        # napari hides a singleton slider, and with one position that is the correct look: there
        # is nothing to step through.
        self._qt_dims.setVisible(n > 1)

    def _follow(self, index: int) -> None:
        """Somebody else moved. Move the widget to match, without echoing back."""
        if self.index == int(index):
            return
        self._echo = True
        try:
            self._dims.set_current_step(0, int(index))
        finally:
            self._echo = False

    def _on_dims_step(self, event=None) -> None:
        if self._echo:
            return
        self._on_step(self.index)

    def _on_step(self, index: int) -> None:
        """The axis moved for a reason that is NOT us following somebody. Subclass hook."""

    def set_index_from_user(self, index: int) -> None:
        """What a drag on the slider does."""
        self._dims.set_current_step(0, int(index))

    # napari's QtDims debounces playback on the render: it drops a requested frame while
    # ``dims._play_ready`` is False. Reused rather than defeated, because a free-running timer
    # would queue mosaic loads faster than they finish. The gate closes on step and reopens on
    # ``frame_done()``, so playback self-limits to load speed and never runs ahead.

    def frame_done(self) -> None:
        """The current frame is fully on screen. Lets playback request the next one."""
        self._dims._play_ready = True
        self._stalled_since = None

    def _watch_for_stall(self) -> None:
        import time as _time

        if not self.is_playing:
            self._stalled_since = None
            return
        if self._dims._play_ready:
            self._stalled_since = None
            return
        if self._stalled_since is None:
            self._stalled_since = _time.monotonic()
            return
        waited = _time.monotonic() - self._stalled_since
        if waited >= self.STALL_GRACE_S:
            self.stop()
            self._stalled_since = None
            self._say(
                f"playback stopped: the {self._noun} has not finished loading after "
                f"{waited:.0f} s, so the next frame was never requested. Step through "
                f"{self._noun}s manually."
            )

    # -- playback (napari's) ------------------------------------------------------------
    def _refusal(self) -> Optional[str]:
        """Why playing would be pointless, in the user's words, or ``None`` to go ahead."""
        n = self.count
        if n <= 1:
            return (f"there is one {self._noun} here; there is nothing to play through.")
        return None

    def play(self, fps: Optional[float] = None) -> None:
        """Start napari's animation on this axis; refuses out loud rather than silently doing
        nothing."""
        refusal = self._refusal()
        if refusal is not None:
            self._say(refusal)
            return
        # loop_mode is explicit: napari's default comes from a user-wide setting that could be
        # "once", which would stop the axis after a single step.
        self._dims._play_ready = True               # arm the gate for the first frame
        self._stalled_since = None
        self._qt_dims.play(0, fps=float(fps) if fps is not None else self.fps, loop_mode="loop")
        if self._stall_timer is not None:
            self._stall_timer.start()

    def stop(self) -> None:
        self._qt_dims.stop()
        if self._stall_timer is not None:
            self._stall_timer.stop()

    def shutdown(self) -> None:
        """Stop and join napari's animation thread; Qt aborts the process if a QThread is
        destroyed while still running."""
        try:
            self._qt_dims.stop()
            thread = getattr(self._qt_dims, "_animation_thread", None)
            if thread is not None and thread.isRunning():
                thread.wait(2000)
        except RuntimeError:
            pass            # the C++ widget is already gone; nothing left to join

    def closeEvent(self, event):                    # noqa: N802 - Qt naming
        self.shutdown()
        super().closeEvent(event)


class RegionSlider(AxisPlayback):
    """The region slider: napari's dims slider/play/fps control, plus the cursor it is a view
    of and the region-id label napari's dims cannot know."""

    def __init__(self, parent=None) -> None:
        from qtpy.QtWidgets import QLabel

        super().__init__(axis_label="region", noun="region", parent=parent)
        self._cursor: Optional[RegionCursor] = None

        # napari's dims shows the index, not the region id, so add a label for it.
        self._label = QLabel("")
        self._label.setMinimumWidth(120)
        self._label.setStyleSheet("color:#c9d1d9;font-size:12px;border:none;")
        self._row.addWidget(self._label)
        self.setToolTip(
            "Step through REGIONS. A region is a mosaic of FOVs, and it is the unit you "
            "navigate by.\nPress play to walk them; right-click play for frames per second "
            "and loop mode."
        )

    # -- wiring -------------------------------------------------------------------------
    def bind(self, cursor: RegionCursor) -> None:
        """Make this slider a view of *cursor*; it holds no region state of its own."""
        self._cursor = cursor
        cursor.subscribe(self._on_cursor)
        # order changes (e.g. an exploration tab) change the region count; subscribing keeps
        # length in sync without a private copy.
        cursor.subscribe_order(self._on_order)
        self._on_order(cursor.regions)

    def _on_cursor(self, index: int, region: str) -> None:
        self._follow(index)
        self.set_region_label(region, index, self._cursor.count if self._cursor else 0)

    def _on_order(self, regions: list) -> None:
        self.set_count(len(regions))
        if self._cursor is not None and self._cursor.index is not None:
            self._on_cursor(self._cursor.index, self._cursor.region or "")
        else:
            self.set_region_label(None, 0, 0)

    def set_region_label(self, region: Optional[str], index: int, count: int) -> None:
        self._label.setText("" if not region else f"{region}   ({index + 1} of {count})")

    def _on_step(self, index: int) -> None:
        """The slider moved for a reason that is not the cursor. Tell the cursor; it owns it."""
        if self._cursor is None:
            return
        self._cursor.set_index(index)

    def _refusal(self):
        """Refuse in terms of regions, not steps; count comes from the cursor (the owner), not
        the dims range."""
        n = self._cursor.count if self._cursor is not None else 0
        if n == 0:
            return "no regions loaded - open an acquisition before playing through regions."
        if n == 1:
            return "this acquisition has one region; there is nothing to play through."
        return None
