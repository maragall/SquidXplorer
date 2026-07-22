"""Region navigation: ONE owner of "which region is current", and napari's own playback.

THE NAVIGATION UNIT IS THE REGION. A region is a MOSAIC of FOVs, never a single FOV. The
viewer used to navigate with ndviewer_light's FOV slider, which (a) does not exist at all once
napari is the viewer — ``_detail`` is ``None``, so from dc0f288 onward there was no navigation
control on screen whatsoever — and (b) labelled its positions ``r:0`` in raw mode, which is the
IMA-270 area: the label is a display concern and nothing here parses it back.

TWO THINGS LIVE HERE.

``RegionCursor`` is the single owner. Before this, "which region is current" existed in three
places that were hand-synced: ``PlateOverview._sel`` (the red frame), ``_mosaic_region`` (what
pane 2 is showing) and ``_current_well`` (what the user opened). Moving one and forgetting
another is this project's dominant defect shape — the FOV slider vs the red box, the plate's
contrast vs ndv's, ``_push_index`` vs the plate index, the loupe's compositor vs the plate's,
all the same bug wearing different hats. So the cursor is not "a better place to keep it": it
is the ONLY place it is kept, and the red frame, the slider and the mosaic are all subscribers
that cannot hold an opinion of their own.

``RegionSlider`` is **napari's own dims slider**, not a QSlider with a QTimer bolted on. Julio:
"we are not reinventing anything, we're just picking solutions out there and bundling them into
a suite." napari ships ``QtDims``/``QtDimSliderWidget`` with a play button, an fps spin box, a
loop-mode selector and an ``AnimationThread`` that keeps the timer off the GUI event loop so a
mouseover cannot lag playback. That is exactly the FPS controller the brief asks for.

WHY THE REGION IS NOT AN AXIS OF THE IMAGE ARRAY, which would have got the slider for free from
napari's viewer instead of a second ``Dims`` model:

  1. Regions are RAGGED. The 10x tissue set is 27 FOVs in ``manual0`` and 28 in ``manual1``, so
     their fused mosaics differ in extent. A single ``(region, z, y, x)`` array has to pad every
     region to the largest, which invents black borders that are indistinguishable from a
     genuinely empty edge of a mosaic.
  2. Each region sits at its OWN stage position. Placement is per LAYER
     (``scale_translate_from_bbox_um`` -> ``layer.translate``), and a layer has one translate.
     Fold regions into an axis and every region after the first draws at region 0's stage
     coordinates — silently wrong against anything else keyed to stage micrometres.
  3. napari's ``Dims`` is a plain model with no dependency on a layer. Driving a second one is
     supported use, costs nothing, and keeps the region axis out of the image entirely.

So we take napari's playback machinery and give it a napari ``Dims`` of its own. Nothing about
the animation, the fps control or the loop modes is ours.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

# --------------------------------------------------------------------------------------
# The single owner
# --------------------------------------------------------------------------------------


class RegionCursor:
    """Which region is current. Pure Python — no Qt, no napari, so it is testable directly.

    Deliberately NOT a QObject with a pyqtSignal. A Qt signal swallows exceptions raised in a
    slot (the whole reason ``launch_fast.py`` installs an excepthook), and the one thing this
    class must never do is fail to move half of the UI without saying so.
    """

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
        """True once the USER explicitly opened a region (double-click), not merely because a
        plate was loaded and something had to be on screen.

        This distinction is load-bearing: ``_selection_regions`` scopes an operator run to the
        activated region, so collapsing it into "a region is displayed" would silently narrow
        every run on a freshly-opened plate to region 0.
        """
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
        """``callback(regions)`` whenever the region ORDER is re-scoped.

        Separate from ``subscribe`` because a re-scope that keeps you on the same region is not
        a navigation — the red frame must not move — but the slider's LENGTH still has to change
        or it addresses regions that are no longer there.
        """
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
            # NEVER log-and-continue. One subscriber blowing up must not silently leave the red
            # frame on a different region from the mosaic; the user gets a sentence.
            self._problem(f"region navigation: a subscriber failed — {text}")   # type: ignore[misc]

    # -- moving -------------------------------------------------------------------------
    def set_order(self, order: Sequence[str]) -> None:
        """Re-scope the cursor to *order* (the plate, or an exploration tab's subset).

        Stays on the SAME region when it survives the re-scope. Snapping back to index 0 would
        move the red frame off the region the user is looking at, which is the exact class of
        silent divergence this object exists to make impossible.
        """
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
        """Move to *index*. Out of range RAISES rather than clamping.

        Clamping is a silent failure: the caller believes it moved to 99, the cursor sits at 2,
        and the two disagree forever after.
        """
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


# --------------------------------------------------------------------------------------
# Binding assertions for the parts of napari's playback we drive
# --------------------------------------------------------------------------------------
# Same policy as ``_napari_view.verify_napari_bindings``: napari's Qt access path moved twice
# between 0.5 and 0.8, and a symbol that vanishes must become a NAMED failure at construction
# rather than a slider that appears and does nothing.

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
            "This is a hard failure on purpose — a play button that silently does nothing is "
            "the failure mode this control was built to replace."
        )


# --------------------------------------------------------------------------------------
# The slider: napari's QtDims over a Dims model of our own
# --------------------------------------------------------------------------------------

#: napari's Dims refuses ndisplay < 2, and only NON-displayed axes get a slider. So the model
#: is 3-D — one region axis plus two dummy displayed axes — and exactly one slider appears.
_DUMMY_DISPLAYED_AXES = 2

#: Default playback rate. napari's own default is 10 fps and the user changes it in napari's
#: own fps popup (right-click the play button), so this is a starting point, not a policy.
DEFAULT_FPS = 10


def _napari_stylesheet() -> str:
    """napari's OWN dark stylesheet, so the embedded dims widget looks like napari's.

    Raises nothing and returns "" only if napari has moved the accessor — in which case the
    control still WORKS and merely looks wrong, which is the right way round for a cosmetic
    dependency. The functional bindings are asserted separately in ``verify_playback_bindings``.
    """
    try:
        from napari.qt import get_stylesheet

        return get_stylesheet("dark")
    except Exception:                               # noqa: BLE001 - cosmetic only
        return ""


try:                                                # pragma: no cover - import shape only
    from PyQt5.QtWidgets import QWidget as _QWidgetBase
except Exception:                                   # pragma: no cover
    _QWidgetBase = object                           # type: ignore[assignment,misc]


class RegionSlider(_QWidgetBase):                   # type: ignore[misc,valid-type]
    """The REGION slider — napari's dims slider, its play button and its fps control.

    Replaces the FOV slider as the navigation control. Everything about the animation is
    napari's ``AnimationThread``: it runs the timer on its own thread precisely so mouseovers
    and other events cannot make playback stutter, which is the first thing a hand-rolled
    ``QTimer`` loop gets wrong.
    """

    def __init__(self, parent=None) -> None:
        verify_playback_bindings()

        from napari.components import Dims
        from napari._qt.widgets.qt_dims import QtDims
        from PyQt5.QtWidgets import QHBoxLayout, QLabel

        super().__init__(parent)
        self._cursor: Optional[RegionCursor] = None
        self._problem: Optional[Callable[[str], None]] = None
        self._echo = False                          # guards the cursor -> slider -> cursor loop
        self._stalled_since: Optional[float] = None

        self._dims = Dims(
            ndim=1 + _DUMMY_DISPLAYED_AXES,
            ndisplay=_DUMMY_DISPLAYED_AXES,
            range=((0, 0, 1), (0, 1, 1), (0, 1, 1)),
            axis_labels=("region", "y", "x"),
        )
        self._qt_dims = QtDims(self._dims)
        self._dims.events.current_step.connect(self._on_dims_step)

        from PyQt5.QtCore import QTimer

        self._stall_timer = QTimer(self)
        self._stall_timer.setInterval(1000)
        self._stall_timer.timeout.connect(self._watch_for_stall)

        # napari's dims widgets are styled ENTIRELY by napari's stylesheet, which is applied to
        # napari's own QMainWindow and therefore does not reach a QtDims parented into our
        # window. Without it the control renders as an unstyled white box, an invisible slider
        # handle and a blank square where the play button should be — verified on screen, which
        # is the only way this was ever going to be caught. Ask napari for its own stylesheet
        # rather than hand-writing one: a hand-written copy is a second styling rule to keep in
        # step with napari's theme, which is this project's standing defect shape.
        self._qt_dims.setStyleSheet(_napari_stylesheet())

        # The region ID, which napari's dims cannot know: its axis label says "region" and its
        # spin box says the INDEX. The user navigates by region id ("manual0"), so show it.
        self._label = QLabel("")
        self._label.setMinimumWidth(120)
        self._label.setStyleSheet("color:#c9d1d9;font-size:12px;border:none;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 2, 8, 2)
        lay.setSpacing(8)
        lay.addWidget(self._qt_dims, 1)
        lay.addWidget(self._label)
        self.setToolTip(
            "Step through REGIONS. A region is a mosaic of FOVs, and it is the unit you "
            "navigate by.\nPress play to walk them; right-click play for frames per second "
            "and loop mode."
        )

    # -- the napari widgets, exposed so tests assert they really are napari's ------------
    @property
    def qt_dims(self):
        return self._qt_dims

    @property
    def dim_slider(self):
        """napari's ``QtDimSliderWidget`` for the region axis."""
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

    # -- wiring -------------------------------------------------------------------------
    def bind(self, cursor: RegionCursor) -> None:
        """Make this slider a VIEW of *cursor*. The slider holds no region state of its own."""
        self._cursor = cursor
        cursor.subscribe(self._on_cursor)
        # A re-scope (an exploration tab) changes the region COUNT. Subscribing to the order is
        # how the slider stays the right LENGTH without ever holding its own copy of the order.
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
        # napari hides a singleton slider, and with one region that is the correct look: there
        # is nothing to step through. The label still says which region is on screen.
        self._qt_dims.setVisible(n > 1)

    def _follow(self, index: int) -> None:
        """The cursor moved. Move the widget to match, without echoing back."""
        if self.index == int(index):
            return
        self._echo = True
        try:
            self._dims.set_current_step(0, int(index))
        finally:
            self._echo = False

    def _on_dims_step(self, event=None) -> None:
        if self._echo or self._cursor is None:
            return
        self._cursor.set_index(self.index)

    def set_index_from_user(self, index: int) -> None:
        """What a drag on the slider does. Named for what it means, so tests read as gestures."""
        self._dims.set_current_step(0, int(index))

    # -- the frame gate -----------------------------------------------------------------
    # napari's playback is DEBOUNCED ON THE RENDER, not free-running: ``QtDims._set_frame``
    # drops a requested frame while ``dims._play_ready`` is False, and napari's canvas sets it
    # back to True when the draw completes. "If the timer plays faster than the canvas can
    # draw, this will drop the intermediate frames, keeping the effective frame rate constant
    # even if the canvas cannot keep up."
    #
    # That is exactly the behaviour this feature needs, so it is REUSED rather than defeated:
    # a region change costs ~1 s of mosaic loading (measured), so a free-running 10 fps timer
    # would queue ten loads for every one that finishes and the viewer would fall further
    # behind the slider the longer you played. Here the gate is closed by the region change and
    # opened by ``frame_done()`` when the mosaic is actually on screen, so playback self-limits
    # to the rate the data can be loaded at and never runs ahead of the picture.

    #: How long playback may sit gated before we assume the renderer is never coming back.
    #: A stall must be a SENTENCE, not a play button that looks pressed and does nothing.
    #:
    #: This is NOT a speed limit and must not be set near the expected load time. A region
    #: change on the 10x tissue set measured 3-6 s idle and was seen at ~20 s with the plate
    #: preview and another GL process competing; at the 20 s this first shipped with, playback
    #: stopped itself mid-plate on a machine that was merely busy. It exists only to catch a
    #: renderer that never calls back AT ALL.
    STALL_GRACE_S = 180.0

    def frame_done(self) -> None:
        """The current region is fully on screen. Lets playback request the next one."""
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
                f"playback stopped: the region has not finished loading after {waited:.0f} s, "
                "so the next frame was never requested. Step through regions manually."
            )

    # -- playback (napari's) ------------------------------------------------------------
    def play(self, fps: Optional[float] = None) -> None:
        """Start napari's animation on the region axis.

        Refuses OUT LOUD rather than doing nothing: a play button that silently ignores you is
        the same dead-control failure as the "Focus reference plane" button that never ran.
        """
        n = self._cursor.count if self._cursor is not None else 0
        if n == 0:
            self._say("no regions loaded — open an acquisition before playing through regions.")
            return
        if n == 1:
            self._say("this acquisition has one region; there is nothing to play through.")
            return
        # loop_mode is passed EXPLICITLY because the default is not ours to depend on: it comes
        # from napari's USER-WIDE ``application.playback_loop_mode`` setting, so on a machine
        # configured to "once" the region axis would advance one step and stop. Walking a plate
        # is a loop, so say so rather than inherit whatever the user last set for a movie.
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
        """Stop and JOIN napari's animation thread.

        Qt aborts the process on "QThread: Destroyed while thread is still running", so the
        thread has to be joined before the widget goes away — a window close during playback
        would otherwise take the app down.
        """
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
