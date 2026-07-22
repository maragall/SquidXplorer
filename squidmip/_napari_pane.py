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

def _qt_viewer_buttons(model):
    from napari._qt.widgets.qt_viewer_buttons import QtViewerButtons
    return QtViewerButtons(model)


def _qt_dims(model):
    from napari._qt.widgets.qt_dims import QtDims
    return QtDims(model.dims)


def _qt_layer_list(model):
    from napari._qt.containers import QtLayerList
    return QtLayerList(model.layers)


def _qt_layer_controls(model):
    from napari._qt.layer_controls import QtLayerControlsContainer
    return QtLayerControlsContainer(model)


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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.mosaic: Optional[MosaicLayers] = None
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

            canvas, mosaic = build_pane()
            canvas.setParent(self)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.canvas = canvas
            self.mosaic = mosaic

            # napari's OWN controls, mounted next to the canvas.
            #
            # Julio, driving the GUI: "The contrast sliders should be napari, and they control the
            # GUI. Napari has lots of awesome embedded functionality" / "I don't see the napari
            # comtrols" / "we definitely need 3d rendering".
            #
            # We embed a bare QtViewer with no napari Window, which is what keeps napari's menus,
            # docks and plugin surface out (his standing warning: "Watch out for feature bloat").
            # But the layer list, the layer controls, the dims sliders and the ndisplay button all
            # live on the Window in a normal napari app, so a bare QtViewer has none of them --
            # that is why the canvas looked bare. Mount the four widgets DIRECTLY: they are the
            # real napari controls, not a reimplementation, and they carry for free the three
            # things he asked for -- per-channel contrast, the z slider, and the 2D/3D toggle.
            # BELOW the canvas, not beside it. Pane 2 is roughly a third of the window, so a
            # side column left the canvas a ~140 px sliver -- Julio: "I cannot see whether my
            # image registers, stitches, and refines correctly". Stacking vertically gives the
            # canvas the FULL pane width, which is the axis that matters for a wide mosaic.
            lay.addWidget(canvas, 1)
            side = self._build_controls(mosaic.model)
            if side is not None:
                lay.addWidget(side, 0)

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

    # -- napari's own controls ----------------------------------------------------------
    def _build_controls(self, viewer_model) -> Optional[QWidget]:
        """The real napari widgets: layer list, layer controls, dims sliders, ndisplay button.

        Each is added independently and a failure of one must not cost the others -- a missing
        z slider is bad, losing the whole control column because one widget's constructor moved
        between napari versions is worse. Every failure is REPORTED on the banner, never
        swallowed: this project has six confirmed silent failures.
        """
        col = QWidget(self)
        # 260, not 320: pane 2 is one third of a three-pane window, and at 320 the control
        # column left the canvas a ~130 px sliver -- seen in a screenshot of the running GUI,
        # which is the only way that was ever going to be caught.
        # Scrollable, and tall enough to lay out. At a fixed 300 px the napari layer-controls
        # container overlapped its own rows (opacity on top of blending on top of contrast
        # limits) and the contrast slider could not be grabbed -- seen in a screenshot of the
        # running GUI. A scroll area means the strip never dominates the canvas AND never
        # squashes its contents; the user drags the splitter for more.
        col.setMinimumHeight(240)
        v = QVBoxLayout(col)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        added, failed = [], []

        def add(label, factory, stretch=0):
            try:
                w = factory()
                v.addWidget(w, stretch)
                added.append(label)
            except Exception as exc:             # noqa: BLE001 - reported below, never swallowed
                failed.append(f"{label}: {type(exc).__name__}: {exc}")

        # QtViewerButtons carries the ndisplay (2D/3D) toggle -- 3D rendering is a stated
        # customer requirement, and napari does it natively with per-layer scale, so the
        # anisotropy fix from IMA-255 (dz_um/pixel_size_um) applies through layer.scale.
        add("viewer buttons", lambda: _qt_viewer_buttons(viewer_model))
        # QtDims is the per-axis slider set -- this is the z control.
        add("dims sliders", lambda: _qt_dims(viewer_model))
        add("layer list", lambda: _qt_layer_list(viewer_model), 1)
        # QtLayerControlsContainer is where per-channel contrast, colormap, blending and
        # opacity live. It replaces the plate's duplicate slider row entirely.
        add("layer controls", lambda: _qt_layer_controls(viewer_model), 1)

        if failed:
            self.say("some napari controls could not be built: " + "; ".join(failed))
        # napari's DARK THEME. Its stylesheet is applied by the napari Window, which we do not
        # construct (that is what keeps napari's menus/docks out). So the real napari widgets
        # rendered with default Qt styling -- Julio: "The controls are so weird, and they have a
        # light gray color. It looks like you're not calling the napari." They ARE napari's
        # widgets; they were simply unstyled. Apply the theme ourselves.
        try:
            from napari._qt.qt_resources import get_stylesheet
            col.setStyleSheet(get_stylesheet("dark"))
        except Exception as exc:                 # noqa: BLE001 - reported, never swallowed
            failed.append(f"napari theme: {type(exc).__name__}: {exc}")

        if not added:
            return None
        from PyQt5.QtWidgets import QScrollArea
        scroll = QScrollArea(self)
        scroll.setWidget(col)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(240)
        scroll.setMaximumHeight(360)
        return scroll

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


def make_pane(readout: Optional[Callable[[str], None]] = None):
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

    pane = MosaicPane()
    if pane.ok:
        return pane, "napari", ""

    reason = pane.failure or "unknown error"
    pane.deleteLater()
    return None, "ndv", f"napari viewer unavailable ({reason}) — fell back to ndviewer_light."
