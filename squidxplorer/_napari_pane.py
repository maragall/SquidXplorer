"""Pane 2: the napari mosaic viewer. Everything Qt lives here; ``_napari_view`` stays Qt-free."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox, QLabel, QPushButton, QSizePolicy, QHBoxLayout, QVBoxLayout, QWidget,
)

from squidxplorer._napari_view import _DEFAULT_MAX_3D_TEXTURE, MosaicLayers, resolve_viewer

# Camera-settle debounce: a quiet-period debounce, not a rate limit. A continuous drag
# coalesces into ONE fetch once the camera stops.
SETTLE_MS = 120


class SettleCoalescer:
    """Fire *callback* only once the camera has been quiet for ``interval``. Clock-injected."""

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


def _channel_entry(channel_name: str, channels):
    """The channel record for *channel_name* (matched by name or display_name), or None.

    Entries are ``DisplayChannel`` records in real metadata and plain dicts in older callers;
    both answer ``.get``, so the discriminator is the protocol, never the type.
    """
    for entry in channels or ():
        get = getattr(entry, "get", None)
        if get is None:
            continue
        if str(get("name")) == str(channel_name) \
                or str(get("display_name") or "") == str(channel_name):
            return entry
    return None


def _acquisition_color(channel_name: str, channels) -> "str | None":
    """The resolved ``display_color`` for *channel_name* in an acquisition's channel list."""
    entry = _channel_entry(channel_name, channels)
    return entry.get("display_color") if entry is not None else None


def _colormap_for(channel_name: str, channels=None):
    """napari colormap for a channel: the measured stain LUT first (a color channel recorded
    gray — see ``_stain``), then the acquisition's own resolved ``display_color`` (the reader's
    RGB component channels carry pure primaries there — the name palette cannot know them),
    else Squid's name palette; grey for an unrecognised channel."""
    try:
        from napari.utils import Colormap

        from squidxplorer._channels import fallback_color

        entry = _channel_entry(channel_name, channels)
        lut = entry.get("display_lut") if entry is not None else None
        if lut:
            return Colormap([[*row[:3], 1.0] for row in lut],
                            name=f"squid-stain-{channel_name}")
        hex_color = _acquisition_color(channel_name, channels) or fallback_color(channel_name)
        if not hex_color:
            return "gray"
        h = hex_color.lstrip("#")
        rgb = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        return Colormap([[0.0, 0.0, 0.0, 1.0], [*rgb, 1.0]], name=f"squid-{channel_name}")
    except Exception:
        return "gray"


#: Tooltip on napari's own 2D/3D button (napari drops multiscale layers to their coarsest
#: level in 3D, so full native resolution needs an ROI crop).
NDISPLAY_TOOLTIP = (
    "3D view (napari).\n"
    "Renders this region's z-stack at the finest resolution that fits the GPU's single 3D "
    "texture (about 2048 px per axis).\n"
    "For FULL native resolution, draw an ROI and open it in its own window: a crop fits in the "
    "texture where the whole region cannot."
)


def apply_ndisplay_tooltip(btn) -> None:
    """Put :data:`NDISPLAY_TOOLTIP` on napari's 2D/3D button. A tooltip and nothing else."""
    if btn is not None:
        btn.setToolTip(NDISPLAY_TOOLTIP)


#: The last ``(value, measured)`` pair announced, so the limit is stated when it is learned.
_MAX_3D_TEXTURE_SAID: "Optional[tuple[int, bool]]" = None


def max_3d_texture_line(value: int, *, measured: bool) -> str:
    """The sentence naming the GPU's 3D texture limit, and whether it was read or assumed."""
    where = "read from the GPU" if measured else (
        f"NOT read from the GPU, assuming {_DEFAULT_MAX_3D_TEXTURE}")
    return (f"GL_MAX_3D_TEXTURE_SIZE = {int(value)} px ({where}). A 3D view is capped to this per "
            f"axis; native 3D needs an ROI no larger than {int(value)} px.")


def _say_max_3d_texture(value: int, *, measured: bool) -> int:
    """Announce the limit when it changes, and return it unchanged."""
    global _MAX_3D_TEXTURE_SAID

    pair = (int(value), bool(measured))
    if pair != _MAX_3D_TEXTURE_SAID:
        _MAX_3D_TEXTURE_SAID = pair
        from squidxplorer._logpane import get_logger

        get_logger("napari_pane").info("%s", max_3d_texture_line(value, measured=measured))
    return int(value)


class MosaicPane(QWidget):
    """Pane 2. Hosts the napari canvas, or a message saying why it could not be built."""

    def __init__(self, parent=None, show_docks: bool = True) -> None:
        super().__init__(parent)
        self.show_docks = bool(show_docks)
        self.mosaic: Optional[MosaicLayers] = None
        self._viewer = None
        self._native_window = None
        self.ndisplay_button: Optional[QWidget] = None
        self.detect_button: Optional[QWidget] = None
        self.detect_channel: Optional[QComboBox] = None   # channel-aware cellpose picker
        self.layer_tree: Optional[QWidget] = None
        self.view_controls_dock = None           # the window's chip block, once docked (below)
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
            from squidxplorer._napari_view import build_pane

            canvas, mosaic, viewer = build_pane()
            self._viewer = viewer
            # Do NOT reparent the canvas: it is the QMainWindow's central widget, and pulling
            # it out guts the embedded window. Kept only as a handle.
            self.canvas = canvas
            self.mosaic = mosaic

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

    def _install_ndisplay_button(self, lay) -> None:
        """Build the top control row, keeping napari's own ndisplay button alive (hidden)."""
        row = QWidget(self)
        rl = QHBoxLayout(row)
        rl.setContentsMargins(6, 4, 6, 4)
        rl.setSpacing(6)

        try:
            from napari._qt.widgets.qt_viewer_buttons import QtViewerButtons
            from napari.qt import get_current_stylesheet

            self._button_source = QtViewerButtons(self._viewer)
            btn = self._button_source.ndisplayButton
            # 3D is the ROI native popout, not an embedded toggle. Keep the button object
            # alive for napari's ndisplay state-sync closure, but never show it.
            btn.hide()
            # napari's icons live in napari's stylesheet; without it the button renders empty.
            try:
                row.setStyleSheet(get_current_stylesheet())
            except Exception:                    # noqa: BLE001 - cosmetic only
                pass
            self.ndisplay_button = btn
        except Exception as exc:                 # noqa: BLE001 - said out loud, never swallowed
            self.say(f"napari's 2D/3D button could not be mounted ({exc}); "
                     "use the one at the bottom of napari's left column.")

        # The analysis-operator trigger, built outside the try above so a napari upgrade
        # cannot take the operator off the screen. Channel choice is the user's: the nuclei
        # signal is not always in 405.
        rl.addWidget(QLabel("Detect on:", row))
        self.detect_channel = QComboBox(row)
        self.detect_channel.setToolTip(
            "Which channel cellpose segments. Pick the one that actually carries the signal "
            "(a nuclear stain for nuclei); a blank channel finds nothing."
        )
        self.detect_channel.setMinimumWidth(150)
        rl.addWidget(self.detect_channel)

        self.detect_button = QPushButton("Detect nuclei", row)
        self.detect_button.setToolTip(
            "Segment the chosen channel with Cellpose, and overlay the mask and the centroids "
            "on this canvas."
        )
        self.detect_button.setEnabled(False)
        rl.addWidget(self.detect_button)

        rl.addStretch(1)
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        #: Exposed so the window can fold this strip into its collapsible operators box
        #: (UI feedback 2026-08-17: "Move to controls window") — the detect trigger is an
        #: operator control, and the canvas keeps the pixels.
        self.detect_row = row
        lay.addWidget(row)
        apply_ndisplay_tooltip(btn)      # a tooltip only: this stays NAPARI's 3D button
        self.ndisplay_button = btn

        # Max-res 3D: serve the full-res volume while ndisplay == 3, restore the pyramid in
        # 2D. One owner of 2D/3D (viewer.dims.ndisplay), so the event catches every button.
        try:
            self._viewer.dims.events.ndisplay.connect(self._on_ndisplay_changed)
            # A region change while in 3D re-adds mosaics as multiscale; re-apply the swap.
            self._viewer.layers.events.inserted.connect(self._reapply_3d_on_insert)
        except Exception:                    # noqa: BLE001 - the 2D pane still works without it
            pass

    def _on_ndisplay_changed(self, event=None) -> None:
        """Follow the 2D/3D toggle: fill the GPU texture budget in 3D, fast pyramid in 2D."""
        if self.mosaic is None or self._viewer is None:
            return
        try:
            self.mosaic._max_3d_texture = self._live_max_3d_texture()
            self.mosaic.render_max_res_3d(self._viewer.dims.ndisplay == 3)
        except Exception:                    # noqa: BLE001 - never break the toggle itself
            pass

    def _live_max_3d_texture(self) -> int:
        """The GPU's real GL_MAX_3D_TEXTURE_SIZE from the live canvas, or the stated fallback."""
        for getter in (
            lambda: self._viewer.window._qt_viewer.canvas.max_texture_sizes[1],
            lambda: getattr(self.canvas, "max_texture_sizes", None)[1],
        ):
            try:
                v = getter()
                if v:
                    return _say_max_3d_texture(int(v), measured=True)
            except Exception:                # noqa: BLE001 - try the next path
                continue
        return _say_max_3d_texture(_DEFAULT_MAX_3D_TEXTURE, measured=False)

    def _reapply_3d_on_insert(self, event=None) -> None:
        if self.mosaic is None or self._viewer is None:
            return
        try:
            if self._viewer.dims.ndisplay == 3:
                self.mosaic.render_max_res_3d(True)
        except Exception:                    # noqa: BLE001
            pass

    def _install_layer_tree(self) -> None:
        """Dock the grouped layer tree, replacing napari's flat layer list (hidden below)."""
        if self.mosaic is None:
            return
        try:
            from squidxplorer._layer_tree import MosaicTree

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
        """Hide (never delete) napari's own flat layer list; the grouped tree is the layer surface."""
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

    def _embed_native_window(self, lay) -> None:
        """Put napari's own QMainWindow inside pane 2, falling back (stated) to the bare canvas."""
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
            # Keep napari's docks and controls; drop only the menu bar.
            try:
                mb().setVisible(False)
            except Exception:                     # noqa: BLE001 - cosmetic only
                pass
        # Give the central widget a floor so the docks cannot collapse the canvas to nothing.
        central = qt_window.centralWidget() if hasattr(qt_window, "centralWidget") else None
        if central is not None:
            central.setMinimumSize(360, 360)
            central.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        qt_window.setMinimumHeight(560)
        if not self.show_docks:
            # The side pane gets the canvas and none of napari's control docks: a second
            # layer-controls surface is a second owner of contrast, and in a narrow column
            # the docks collapse the canvas. The layers still exist and are still linked.
            from qtpy.QtWidgets import QDockWidget, QStatusBar

            for dock in qt_window.findChildren(QDockWidget):
                dock.hide()
            # napari's status bar too: a second status line under the window's real one.
            for bar in qt_window.findChildren(QStatusBar):
                bar.hide()
            if central is not None:
                central.setMinimumSize(180, 180)   # a narrow column is still a usable canvas
            qt_window.setMinimumHeight(220)
        lay.addWidget(qt_window, 1)
        self._native_window = qt_window

    def dock_view_controls(self, widget: QWidget) -> bool:
        """Dock *widget* (the window's "2D / 3D · ROI" chip block) at the TOP of napari's left
        column, above the layer controls (UI feedback 2026-08-19: the chips belong "on the left
        column, where the controls are", freeing the viewer's top edge).

        Returns whether it docked; a False sends the caller to its own fallback (the window
        body), so the chips are never lost. Failure is stated, never swallowed.
        """
        if self._viewer is None or self._native_window is None or not self.show_docks:
            return False
        try:
            dock = self._viewer.window.add_dock_widget(
                widget, name="2D / 3D · ROI", area="left")
        except Exception as exc:                 # noqa: BLE001 - the caller has a fallback
            self.say(f"the view controls could not be docked ({type(exc).__name__}: {exc}); "
                     "they are in the window body instead.")
            return False
        try:
            self._hoist_left_dock(dock)
        except Exception as exc:                 # noqa: BLE001 - in-column, just not on top
            from squidxplorer._logpane import get_logger

            get_logger("napari_pane").debug(
                "the view controls docked but could not be hoisted above the layer "
                "controls: %s", exc)
        self.view_controls_dock = dock
        return True

    def _hoist_left_dock(self, dock) -> None:
        """Put *dock* FIRST in the left column by re-adding every other left dock below it.

        Qt appends docks, so "insert above" is spelled remove-and-re-add. Visibility is kept per
        dock: napari's flat layer list is hidden on purpose (`_hide_flat_layer_list`) and a
        re-add must not resurrect it.
        """
        from qtpy.QtWidgets import QDockWidget

        qt_window = self._native_window
        others = [(d, d.isVisibleTo(qt_window))
                  for d in qt_window.findChildren(QDockWidget)
                  if d is not dock and qt_window.dockWidgetArea(d) == Qt.LeftDockWidgetArea]
        for d, _ in others:
            qt_window.removeDockWidget(d)
        for d, was_visible in others:
            qt_window.addDockWidget(Qt.LeftDockWidgetArea, d)
            d.setVisible(was_visible)

    def _install_camera_settle(self) -> None:
        assert self.mosaic is not None
        self._settle = SettleCoalescer(SETTLE_MS / 1000.0, self._fire_settle)
        self._timer = QTimer(self)
        self._timer.setInterval(max(10, SETTLE_MS // 4))
        self._timer.timeout.connect(self._settle.poll)
        camera = self.mosaic.model.camera
        # Keep a handle on each connection so shutdown() can disconnect it; napari's emitter
        # holds a strong ref to these lambdas, so the pane cannot be collected otherwise.
        self._cam_cbs = []
        for emitter in (camera.events.zoom, camera.events.center):
            cb = lambda e: self._note_camera()
            emitter.connect(cb)
            self._cam_cbs.append((emitter, cb))

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

    @property
    def canvas_widget(self):
        """The widget the mosaic PIXELS are drawn in — vispy's GL widget. ``None`` if unreachable.

        NOT :attr:`canvas`, which is napari's ``QtViewer`` and MUST NOT be reparented (see the
        note in ``__init__``: "Do NOT reparent the canvas"). This is one level further in,
        and the distinction matters twice over:

        * its coordinate system IS the canvas coordinate system, so an overlay positioned against
          it needs no conversion from a napari mouse event's ``pos``;
        * napari itself parents a plain ``QWidget`` to exactly this widget (``QtWelcomeWidget``,
          ``napari/_qt/qt_viewer.py``), so a child over the GL surface is a supported arrangement
          rather than something we are getting away with.

        ADDING A CHILD is not reparenting: nothing here calls ``setParent`` on anything napari
        owns. Returns ``None`` — and the caller must say so — rather than falling back to the
        QtViewer, which would put an overlay in the wrong coordinate space at the wrong size:
        a loupe confidently in the wrong place.
        """
        viewer = self._viewer
        if viewer is None:
            return None
        try:
            return viewer.window._qt_viewer.canvas.native
        except Exception:                        # noqa: BLE001 - napari moved it; answer None
            return None

    def shutdown(self) -> None:
        """Tear down the napari Viewer this pane owns (GL context, QMainWindow, subscriptions).

        deleteLater() alone leaks the Viewer via napari's instance registry. Idempotent.
        """
        if self._timer is not None:
            self._timer.stop()
        for emitter, cb in getattr(self, "_cam_cbs", ()):
            try:
                emitter.disconnect(cb)
            except (TypeError, RuntimeError, ValueError):
                pass                                  # already gone, or napari changed the emitter
        self._cam_cbs = []
        viewer = self._viewer
        self._viewer = None
        if viewer is not None:
            try:
                viewer.close()                        # napari: closes the window + drops the registry entry
            except Exception as exc:                  # a teardown error must not mask the dispose,
                from squidxplorer._logpane import get_logger   # but it is NOT swallowed — it is named
                get_logger("napari_pane").warning(
                    "napari viewer close failed during pane shutdown: %s", exc)


#: Qt platform plugins that ship no OpenGL: constructing the vispy canvas under one of these
#: segfaults the process rather than raising.
_NO_GL_PLATFORMS = ("offscreen", "minimal", "vnc")


def gl_available(env: Optional[dict] = None) -> tuple[bool, str]:
    """Whether a GL-capable Qt platform is in use. Returns ``(ok, reason_if_not)``."""
    src = os.environ if env is None else env
    platform = str(src.get("QT_QPA_PLATFORM", "")).strip().lower()
    if platform in _NO_GL_PLATFORMS:
        return False, f"Qt platform {platform!r} provides no OpenGL context"
    return True, ""


def model_pane_class():
    """The ONE test adapter at the pane seam: a pane whose napari CANVAS is absent but whose
    model (``napari.components.ViewerModel``, Qt-free) and ``MosaicLayers`` are real.

    Everything downstream of the pane is production code; what a ModelPane cannot prove is only
    that a layer was PAINTED. conftest's ``napari_pane_stub``, GATE 3 and the walkthrough all
    take it from here — two adapters (the vispy ``MosaicPane``, this) make the seam real, and a
    hand-synced reimplementation of ``MosaicLayers`` is exactly the drift this replaces
    (``StubLayer``'s last lie let four sites drift with the suite green; see CLAUDE.md).
    """
    from qtpy.QtWidgets import QWidget

    from napari.components import ViewerModel

    from squidxplorer._napari_view import MosaicLayers

    class ModelPane(QWidget):
        ok = True

        def __init__(self):
            super().__init__()
            self._viewer = ViewerModel()
            self.mosaic = MosaicLayers(self._viewer)
            self.detect_channel = None
            self.detect_button = None
            self.said = []
            self.shutdowns = 0

        def say(self, text):
            self.said.append(text)

        def shutdown(self):
            """COUNTS, rather than no-ops. The real ``MosaicPane.shutdown`` is what closes the
            napari Viewer and drops it from napari's instance registry, and it went uncalled for
            long enough to leak a GL context per closed window. A pane that silently accepted the
            call could not tell the difference between "disposed" and "never disposed" — and
            ``RegionViewer.dispose`` wraps every teardown in ``except Exception``, so a MISSING
            method here would be swallowed and read as success."""
            self.shutdowns += 1
            self._viewer = None

    return ModelPane


def make_pane(readout: Optional[Callable[[str], None]] = None, *, show_docks: bool = True):
    """Build pane 2, the napari mosaic.

    Returns ``(widget_or_None, mode, message)``; ``mode`` is ``"napari"`` or ``"unavailable"``,
    and an unavailable pane comes with the exact reason. There is no fallback viewer.
    """
    # resolve_viewer stays the single reader of SQUIDXPLORER_VIEWER (and warns on retired values).
    resolve_viewer()

    ok, why = gl_available()
    if not ok:
        return None, "unavailable", f"napari needs OpenGL, and {why}. No mosaic can be drawn here."

    # napari's TOASTS GO TO THE LOGGER (UI feedback 2026-08-17: "Move to logger"): the in-canvas
    # overlay ("Inconsistent units across layers…") covers pixels and vanishes; the log keeps it.
    # Process-wide and idempotent; a napari that moved these settings degrades to the toasts.
    try:
        import logging

        from squidxplorer._logpane import get_logger
        from napari.settings import get_settings
        from napari.utils.notifications import (
            notification_manager, NotificationSeverity)

        get_settings().application.gui_notification_level = NotificationSeverity.NONE

        def _to_log(notification, _log=get_logger("napari")):
            _log.log(logging.WARNING if str(notification.severity) in ("warning", "error")
                     else logging.INFO, "%s", notification.message)

        if _to_log.__code__ not in {getattr(cb, "__code__", None)
                                    for cb in notification_manager.callbacks}:
            notification_manager.callbacks.append(_to_log)
    except Exception as exc:                     # noqa: BLE001 - napari moved it: toasts remain
        get_logger("napari_pane").debug("could not route napari toasts to the log: %s", exc)

    pane = MosaicPane(show_docks=show_docks)
    if pane.ok:
        return pane, "napari", ""

    reason = pane.failure or "unknown error"
    pane.deleteLater()
    return None, "unavailable", f"napari viewer could not be built ({reason}). There is no mosaic."
