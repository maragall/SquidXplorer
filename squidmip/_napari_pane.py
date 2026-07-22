"""Pane 2: the napari mosaic viewer, with a VISIBLE fallback to ndviewer_light.

Kept separate from ``_napari_view`` so that module stays importable (and testable) with no Qt
and no napari at all. Everything Qt lives here.

The fallback is the point of this module as much as the canvas is. napari can fail to construct
for reasons that have nothing to do with our code — no GL context, a Qt binding clash, a napari
upgrade that moved a symbol. When that happens the user must end up with a WORKING viewer and a
sentence saying what happened. This project has six confirmed silent failures, most recently a
plane that rendered blank because an ``IsADirectoryError`` was logged and swallowed; a viewer
that quietly degrades is the same defect wearing a different hat.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QLabel, QSizePolicy, QHBoxLayout, QVBoxLayout, QWidget

from squidmip._napari_view import MosaicLayers, resolve_viewer

# Camera-settle debounce. The measured pan cost (22.6 ms median) is per SETTLED move; a drag
# emits camera events far faster than that, and fetching per event is the mechanism behind
# napari issue #1942 — each event starts a fetch the next event invalidates, so the queue grows
# faster than it drains and the canvas falls behind the cursor. 120 ms is the interval: long
# enough that a continuous drag (events every ~16 ms at 60 Hz) coalesces into ONE fetch, short
# enough to sit under the ~150 ms at which a pause stops feeling like a response to your own
# action. It is a QUIET-period debounce, not a rate limit: nothing is fetched until the camera
# has actually stopped, so a long drag costs one fetch, not one per 120 ms.
SETTLE_MS = 120


class SettleCoalescer:
    """Fire *callback* only once the camera has been quiet for ``interval``.

    Clock-injected so the policy is unit-testable without a Qt event loop or real sleeping —
    the timing rule is the thing worth testing, and a test that sleeps is a test nobody runs.
    """

    def __init__(self, interval_s: float, callback: Callable[[], None],
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._interval = float(interval_s)
        self._callback = callback
        self._clock = clock
        self._last: Optional[float] = None
        self.fired = 0

    def notify(self) -> None:
        """A camera event arrived. Restarts the quiet period."""
        self._last = self._clock()

    def poll(self) -> bool:
        """Fire if the camera has been quiet long enough. Returns whether it fired."""
        if self._last is None:
            return False
        if (self._clock() - self._last) < self._interval:
            return False
        self._last = None
        self.fired += 1
        self._callback()
        return True

    @property
    def pending(self) -> bool:
        return self._last is not None


# --- napari control-widget constructors -------------------------------------------------
# Imported lazily and one per function so a rename in any single napari version costs that
# ONE widget, not the whole control column. Binding is asserted by tests/test_napari_view.py
# rather than trusted -- the _voxel_scale precedent (a patch that bound, ran, and did nothing
# for its entire life) is why nothing here is assumed.

def _colormap_for(channel_name: str):
    """napari colormap for a channel, from Squid's authoritative palette.

    ``_channels`` owns the palette and the name normalisation; this does not restate either.
    Falls back to grey rather than raising: an unrecognised channel must still be VISIBLE here
    (``_channels.resolve_channels`` is the place that refuses to guess a colour, and it runs on
    the acquisition, not on the render).
    """
    try:
        from napari.utils import Colormap

        from squidmip._channels import fallback_color

        hex_color = fallback_color(channel_name)
        if not hex_color:
            return "gray"
        h = hex_color.lstrip("#")
        rgb = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        return Colormap([[0.0, 0.0, 0.0, 1.0], [*rgb, 1.0]], name=f"squid-{channel_name}")
    except Exception:
        return "gray"


class MosaicPane(QWidget):
    """Pane 2. Hosts the napari canvas, or a message saying why it could not be built."""

    def __init__(self, parent=None, show_docks: bool = True) -> None:
        super().__init__(parent)
        self.show_docks = bool(show_docks)
        self.mosaic: Optional[MosaicLayers] = None
        self._viewer = None
        self._native_window = None
        self.ndisplay_button: Optional[QWidget] = None
        self.layer_tree: Optional[QWidget] = None
        self._button_source = None               # keeps napari's row alive; see _install_ndisplay
        self.canvas: Optional[QWidget] = None
        self.failure: Optional[str] = None
        self._settle: Optional[SettleCoalescer] = None
        self._timer: Optional[QTimer] = None
        self._on_settle: Optional[Callable[[], None]] = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._banner = QLabel("")
        self._banner.setAlignment(Qt.AlignCenter)
        self._banner.setWordWrap(True)
        self._banner.setStyleSheet(
            "background:#5a2d2d;color:#ffd7d7;padding:6px 10px;font-size:12px;"
        )
        self._banner.hide()
        lay.addWidget(self._banner)

        try:
            from squidmip._napari_view import build_pane

            canvas, mosaic, viewer = build_pane()
            self._viewer = viewer
            # DO NOT reparent the canvas here. It is the QMainWindow's CENTRAL WIDGET, and
            # setParent() on it rips it out of napari's own window -- which _embed_native_window
            # then embeds, gutted. The docks and layer controls still came along, so the pane
            # looked alive while the mosaic had nowhere to paint: reported as "canvas is still
            # showing blank for the array, so I can't test the central viewer". The canvas is
            # kept only as a HANDLE (and for the bare-canvas fallback, which is the one path
            # allowed to reparent it, because there the QMainWindow is unusable anyway).
            self.canvas = canvas
            self.mosaic = mosaic

            # THE REAL NAPARI WINDOW, not a canvas plus controls I arranged myself.
            #
            # Julio: "You're not showing me a napari window. You're showing me maybe a napari
            # array viewer with controls that you made when napari already has embedded controls
            # and knows how to read data. I don't understand why you're inventing the wheel."
            # And: "the controls show on the left side... I just don't think that napari has the
            # toggle on and off like that. Are those the actual napari controls, or are you doing
            # a modification of them?"
            #
            # They WERE napari's real widget classes -- but laid out by me, in my own container,
            # at the bottom. napari docks them on the LEFT, with its own layer buttons and its own
            # theme. So it looked like a knock-off of napari built out of napari's own parts.
            #
            # I originally stripped the napari Window to honour "watch out for feature bloat".
            # That was the wrong reading: the Window is not the bloat, it is where contrast
            # behaviour, blending, the dims sliders, the ndisplay (2D/3D) button, the layer
            # controls AND the stylesheet all live. Use it.
            self._install_ndisplay_button(lay)
            self._install_layer_tree()
            self._embed_native_window(lay)
            self._install_camera_settle()
        except Exception as exc:                 # noqa: BLE001 - reported, never swallowed
            self.failure = f"{type(exc).__name__}: {exc}"
            msg = QLabel(
                "napari viewer unavailable — falling back to ndviewer_light.\n"
                f"{self.failure}"
            )
            msg.setAlignment(Qt.AlignCenter)
            msg.setWordWrap(True)
            msg.setStyleSheet("color:#ffd7d7;background:#3a2020;padding:12px;")
            lay.addWidget(msg, 1)

    # -- the 2D/3D toggle, where a short pane still shows it -----------------------------
    def _install_ndisplay_button(self, lay) -> None:
        """Lift NAPARI'S OWN ndisplay button into a fixed row at the top of the pane.

        Julio has asked for a visible 3D toggle twice, and the button was never missing: a probe
        of the embedded window found ``QtViewerButtons.ndisplayButton`` present and visible at
        y=752 inside a 900 px host — the last row of the left dock column, under a layer list
        that grows with every layer added. On a small monitor it is simply below the fold. So
        this does not BUILD a button (PartSeg's ``QtNDisplayButton`` does not exist in napari
        0.6.6 anyway); it constructs napari's own button row, takes the one button out of it and
        puts it where a short pane still shows it.

        Reparenting a BUTTON is not the canvas trap: the canvas is the QMainWindow's central
        widget and pulling it out guts the window (506c813). A button is a leaf in a dock.
        The napari row that produced it is kept alive on ``self._button_source`` because napari's
        check-state sync (``viewer.dims.events.ndisplay`` -> ``setChecked``) is a closure owned by
        that row; drop the row and the button silently stops following the viewer.

        There is exactly one owner of 2D/3D — ``viewer.dims.ndisplay``. This button and the one
        napari docks read and write that same property, so they cannot disagree.
        """
        try:
            from napari._qt.widgets.qt_viewer_buttons import QtViewerButtons
            from napari.qt import get_current_stylesheet
        except Exception as exc:                 # noqa: BLE001 - said out loud, never swallowed
            self.say(f"napari's 2D/3D button could not be mounted ({exc}); "
                     "use the one at the bottom of napari's left column.")
            return

        self._button_source = QtViewerButtons(self._viewer)
        btn = self._button_source.ndisplayButton

        row = QWidget(self)
        rl = QHBoxLayout(row)
        rl.setContentsMargins(6, 4, 6, 4)
        rl.setSpacing(6)
        rl.addWidget(btn)
        rl.addStretch(1)
        # napari's icons live in napari's stylesheet, which is applied to napari's own window.
        # Outside it the button would render as an empty square -- a control that is technically
        # visible and reads as broken. get_current_stylesheet is public and in napari.qt.__all__.
        try:
            row.setStyleSheet(get_current_stylesheet())
        except Exception:                        # noqa: BLE001 - cosmetic only
            pass
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(row)
        self.ndisplay_button = btn

    # -- the grouped layer tree ---------------------------------------------------------
    def _install_layer_tree(self) -> None:
        """Dock the PROCESSING LAYER -> CHANNELS tree next to napari's own layer list.

        24 flat rows (5 operators x 4 channels + 4 raw) is unusable, and napari 0.6.6 has no
        groups to fix it with. ``squidmip._layer_tree`` explains that in full; this is only the
        mounting.

        IT REPLACES NAPARI'S FLAT LAYER LIST, which is hidden below.

        The earlier version mounted this ALONGSIDE the flat list, arguing the two could not
        conflict because both write ``layer.visible``. Julio killed that argument in one line:
        "Why do we have the layer list tab in our napari variant if we don't want the number of
        layers to explode precisely?"

        He is right and the old reasoning missed the point. The problem was never that the two
        surfaces disagree - it is that the flat list SHOWS THE EXPLOSION. Five operators x four
        channels is 24 rows, and keeping a tab that displays all 24 defeats the entire reason the
        grouped tree exists. Both shipped precedents do the same thing: PartSeg deletes the dock
        outright, and napari-experimental replaces the layer-list UI rather than adding to it.

        napari's LAYER CONTROLS dock stays - that is the contrast/gamma/colormap panel, it is a
        different surface from the list, and it is the one that must keep owning contrast.

        Mounted through ``Window.add_dock_widget``, napari's own public API, so the tree is a
        napari dock in napari's dock area with napari's styling -- rather than a panel of mine
        bolted to the side, which is the shape that got rejected. ``tabify`` puts it in the same
        tab group as the layer list instead of stealing vertical space on a small monitor.
        """
        if self.mosaic is None:
            return
        try:
            from squidmip._layer_tree import MosaicTree

            tree = MosaicTree(self.mosaic)
            self._viewer.window.add_dock_widget(
                tree, name="mosaic layers", area="left", tabify=True,
            )
            self._hide_flat_layer_list()
        except Exception as exc:                 # noqa: BLE001 - said out loud, never swallowed
            self.say(
                f"the grouped layer tree could not be mounted ({type(exc).__name__}: {exc}); "
                "napari's flat layer list is still there."
            )
            return
        self.layer_tree = tree

    def _hide_flat_layer_list(self) -> None:
        """Hide napari's own flat layer list, leaving the grouped tree as the layer surface.

        HIDDEN, not deleted. PartSeg calls ``deleteLater()``; hiding is reversible, survives a
        napari version that reorganises its docks, and leaves the widget alive so napari's own
        code can still reference it. Deleting a dock napari believes it owns is a good way to
        find out which of its actions assumed otherwise.

        ``dockLayerList`` is PRIVATE napari surface, so a failure here is reported and the tree
        still mounts: an extra tab is untidy, an unmounted tree is a regression.
        """
        try:
            qt_viewer = getattr(self._viewer.window, "_qt_viewer", None)
            dock = getattr(qt_viewer, "dockLayerList", None) if qt_viewer is not None else None
            if dock is None:
                self.say("napari's flat layer list could not be found, so it is still showing.")
                return
            dock.setVisible(False)
            self.flat_layer_list_hidden = True
        except Exception as exc:                 # noqa: BLE001 - cosmetic; never lose the tree
            self.say(f"napari's flat layer list could not be hidden ({type(exc).__name__}: {exc}).")

    # -- the native napari window -------------------------------------------------------
    def _embed_native_window(self, lay) -> None:
        """Put napari's own QMainWindow inside pane 2.

        Falls back to the bare canvas if the private handle moves between napari versions -- and
        SAYS SO on the banner rather than degrading quietly. `_qt_window` is private, so it is
        asserted, not trusted: the _voxel_scale precedent (a patch that bound, ran and did nothing
        for its whole life) is why nothing here is assumed to work.
        """
        qt_window = getattr(self._viewer.window, "_qt_window", None)
        if qt_window is None or not hasattr(qt_window, "setParent"):
            self.say(
                "napari's native window could not be embedded (napari changed _qt_window); "
                "showing the bare canvas instead — controls will look wrong."
            )
            if self.canvas is not None:
                lay.addWidget(self.canvas, 1)
            return
        qt_window.setParent(self)
        qt_window.setWindowFlags(Qt.Widget)      # a child widget, not a top-level window
        qt_window.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mb = getattr(qt_window, "menuBar", None)
        if callable(mb):
            # Keep napari's docks and controls; drop only the menu bar, which duplicates our own
            # chrome and is the one part that genuinely is bloat inside a pane.
            try:
                mb().setVisible(False)
            except Exception:                     # noqa: BLE001 - cosmetic only
                pass
        # The canvas is the QMainWindow's CENTRAL widget; napari's docks are siblings of it.
        # Embedded, the docks claimed all the space and the canvas collapsed to nothing --
        # Julio: "Now all I see are the controls, and they are eclipsing the actual mosaic. It
        # just looks like an empty gray canvas." Give the central widget a floor so the mosaic
        # always has room, and let the docks take what is left.
        central = qt_window.centralWidget() if hasattr(qt_window, "centralWidget") else None
        if central is not None:
            central.setMinimumSize(360, 360)
            central.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        qt_window.setMinimumHeight(560)
        if not self.show_docks:
            # THE SIDE PANE gets the canvas and none of napari's control docks.
            #
            # Two reasons, and the first is the one that matters. Contrast and channel
            # VISIBILITY have exactly one owner, and it is the CENTRE viewer — Julio: "the
            # channel toggling and contrast adjustment for the plate view should happen from our
            # central viewer window". A second full layer-controls surface in a second viewer is
            # the same duplication as a second contrast slider, just wearing napari's own
            # clothes: two widgets that can move one quantity and disagree.
            #
            # The second is size. In a 380 px column the docks take essentially all of it and the
            # canvas collapses to a strip — measured on screen, not reasoned about: the mosaic
            # was ~40 px wide beside a full-height layer list. "Controls eclipsing content" is a
            # complaint this project has already had twice.
            #
            # The layers still EXIST and are still linked; only their control widgets are hidden.
            from PyQt5.QtWidgets import QDockWidget, QStatusBar

            for dock in qt_window.findChildren(QDockWidget):
                dock.hide()
            # napari's status bar too ("Ready ... activity"). It reports on the viewer it belongs
            # to, and in a side-pane tab it is a second status line sitting under a mosaic, six
            # pixels from the window's real one. Two status lines is two places to look.
            for bar in qt_window.findChildren(QStatusBar):
                bar.hide()
            if central is not None:
                central.setMinimumSize(180, 180)   # a narrow column is still a usable canvas
            qt_window.setMinimumHeight(220)
        lay.addWidget(qt_window, 1)
        self._native_window = qt_window

    # -- camera settle ------------------------------------------------------------------
    def _install_camera_settle(self) -> None:
        assert self.mosaic is not None
        self._settle = SettleCoalescer(SETTLE_MS / 1000.0, self._fire_settle)
        self._timer = QTimer(self)
        self._timer.setInterval(max(10, SETTLE_MS // 4))
        self._timer.timeout.connect(self._settle.poll)
        camera = self.mosaic.model.camera
        camera.events.zoom.connect(lambda e: self._note_camera())
        camera.events.center.connect(lambda e: self._note_camera())

    def _note_camera(self) -> None:
        if self._settle is None or self._timer is None:
            return
        self._settle.notify()
        if not self._timer.isActive():
            self._timer.start()

    def _fire_settle(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        if self._on_settle is not None:
            self._on_settle()

    def on_camera_settled(self, callback: Callable[[], None]) -> None:
        """Register the work that may only run once the camera has stopped."""
        self._on_settle = callback

    # -- banner -------------------------------------------------------------------------
    def say(self, text: str) -> None:
        """Show a message to the user. Never log-and-continue."""
        if not text:
            self._banner.hide()
            return
        self._banner.setText(text)
        self._banner.show()

    @property
    def ok(self) -> bool:
        return self.mosaic is not None


#: Qt platform plugins that ship no OpenGL. napari's canvas is vispy/GL, so constructing it
#: under one of these does not raise — it SEGFAULTS the process ("QOpenGLWidget is not supported
#: on this platform", "does not support createPlatformOpenGLContext"). Every headless gate here
#: (pytest, tools/acceptance.py, tools/walkthrough.py) runs offscreen, so without this check
#: wiring napari into PlateWindow would take the whole suite down with a signal 11 rather than a
#: test failure. Falling back with a stated reason is the only honest option: there is genuinely
#: no GL to render into.
_NO_GL_PLATFORMS = ("offscreen", "minimal", "vnc")


def gl_available(env: Optional[dict] = None) -> tuple[bool, str]:
    """Whether a GL-capable Qt platform is in use. Returns ``(ok, reason_if_not)``."""
    src = os.environ if env is None else env
    platform = str(src.get("QT_QPA_PLATFORM", "")).strip().lower()
    if platform in _NO_GL_PLATFORMS:
        return False, f"Qt platform {platform!r} provides no OpenGL context"
    return True, ""


def make_pane(readout: Optional[Callable[[str], None]] = None, *, show_docks: bool = True):
    """Build pane 2 honouring ``SQUIDMIP_VIEWER``.

    Returns ``(widget_or_None, mode, message)``:

    * ``mode == "napari"`` — the napari mosaic pane, and ``widget`` is it.
    * ``mode == "ndv"``    — the caller should build ndviewer_light instead. ``message`` says
      whether that was ASKED FOR or is a FALLBACK, and the caller must surface it.

    The default is napari. The fallback stays reachable with ``SQUIDMIP_VIEWER=ndv`` so a bad
    napari path never leaves the window without a viewer during a visual-feedback round.
    """
    if resolve_viewer() != "napari":
        return None, "ndv", "ndviewer_light selected by SQUIDMIP_VIEWER."

    ok, why = gl_available()
    if not ok:
        return None, "ndv", f"napari needs OpenGL ({why}) — using ndviewer_light."

    pane = MosaicPane(show_docks=show_docks)
    if pane.ok:
        return pane, "napari", ""

    reason = pane.failure or "unknown error"
    pane.deleteLater()
    return None, "ndv", f"napari viewer unavailable ({reason}) — fell back to ndviewer_light."
