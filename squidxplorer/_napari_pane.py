"""Pane 2: the napari mosaic viewer. Everything Qt lives here; ``_napari_view`` stays Qt-free."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

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


#: Layer-controls rows the app does not need on screen (hero declutter, team feedback
#: 2026-08-25: "minimize most of the Napari-Native tools"; Julio, live: "Layer controls,
#: too much height"). Matched against the form's own label text, lowercased. What a
#: life-science user touches stays: contrast limits, auto-contrast, colormap. Everything
#: else goes: rendering choices the app sets itself (blending, opacity), z policy the
#: app's own slider and operators own (projection mode, depiction), and knobs no user of
#: this app has needed (interpolation, gamma). gamma and opacity stay IDENTITY_PROPS:
#: the model still mirrors them; only their napari rows are gone.
NATIVE_HIDDEN_ROWS = ("blending:", "projection mode:", "interpolation:",
                      "gamma:", "opacity:", "depiction:",
                      # A Shapes layer's styling: the ROI rectangle's look is the app's, not
                      # the user's (Julio, 2026-08-25, "realstate not being allocated").
                      "edge width:", "edge color:", "face color:", "display text:")

#: The row that marks a SHAPES form; with it, the shape-tool button grid is chrome too.
_SHAPES_SIGNATURE_ROW = "edge width:"


def hide_native_rows(controls_widget) -> "list[str]":
    """Hide the :data:`NATIVE_HIDDEN_ROWS` of one layer-controls form; returns what it hid.

    Idempotent, and matched by the LABEL TEXT of the form's own rows, never by napari
    widget attribute names, so a napari rename degrades to rows staying visible rather
    than an AttributeError."""
    from qtpy.QtWidgets import QFormLayout

    lay = controls_widget.layout() if controls_widget is not None else None
    if not isinstance(lay, QFormLayout):
        return []
    # Roughly HALF the resting height comes from chrome, not rows (Julio, live
    # 2026-08-25): squeeze the form's own vertical spacing and margins too.
    lay.setVerticalSpacing(2)
    m = lay.contentsMargins()
    lay.setContentsMargins(m.left(), 2, m.right(), 2)
    hidden = []
    is_shapes = False
    for i in range(lay.rowCount()):
        item = lay.itemAt(i, QFormLayout.LabelRole)
        label = item.widget() if item is not None else None
        text = str(label.text()).strip().lower() if hasattr(label, "text") else ""
        is_shapes = is_shapes or text == _SHAPES_SIGNATURE_ROW
        if text not in NATIVE_HIDDEN_ROWS:
            continue
        try:
            lay.setRowVisible(i, False)          # Qt >= 6.4: hides the whole row
        except AttributeError:
            field = lay.itemAt(i, QFormLayout.FieldRole)
            label.hide()
            if field is not None and field.widget() is not None:
                field.widget().hide()
        hidden.append(text)
    grid = getattr(controls_widget, "button_grid", None)
    if is_shapes and grid is not None:
        # The shape-tool row (select / add rectangle / ...): the app sets the layer's mode
        # itself, so the grid is chrome on a Shapes form. An Image form keeps its own.
        for j in range(grid.count()):
            w = grid.itemAt(j).widget() if grid.itemAt(j) is not None else None
            if w is not None:
                w.hide()
        hidden.append("shape tools")
    return hidden


def fit_controls_container(container) -> None:
    """Cap napari's layer-controls container at its CURRENT page's need. The container is
    a QStackedWidget whose hint is its TALLEST page (an Image form, 289 px measured), so a
    Shapes page - or a dieted Image page - sat over a blank band."""
    current = container.currentWidget() if hasattr(container, "currentWidget") else None
    if current is None:
        return
    lay = current.layout()
    if lay is not None:
        lay.activate()
    container.setMaximumHeight(max(1, int(current.sizeHint().height())))


def fit_dock_to_content(dock) -> int:
    """Make *dock* exactly as tall as its content wants (title bars are slimmed to 0 here).

    THE LEFT-COLUMN MECHANISM (Julio, 2026-08-25, screenshot: "realstate not being allocated
    efficiently"; measured on 2888349: ~130 px blank under the operators band, ~80 px under
    the layer controls, the layer list squeezed to two rows, the log band pushed out of its
    dock). A QDockWidget ignores its child's size policy: the dock area hands spare height to
    every dock, and ours top-align their content and paint the rest blank. So every dock the
    app adds is FIXED at its content's hint, and the layer list is the one stretch consumer
    (`stretch_dock`). Returns the height set."""
    w = dock.widget()
    if w is None:
        return 0
    lay = w.layout()
    if lay is not None:
        lay.activate()
    h = min(int(w.sizeHint().height()), int(w.maximumHeight()))
    tb = dock.titleBarWidget()
    if tb is not None and tb.maximumHeight() < 16777215:
        h += int(tb.maximumHeight())
    h = max(0, h)
    if dock.minimumHeight() != h or dock.maximumHeight() != h:
        dock.setFixedHeight(h)
    return h


class _DockFitter(QObject):
    """Keeps a dock fitted to its content across every relayout of that content (a fold
    collapsing, a param slot inserting, the log band re-capping): one deferred refit per
    LayoutRequest, so the hint is read after the layout pass, never inside it."""

    def __init__(self, dock) -> None:
        super().__init__(dock)
        self._dock = dock
        self._pending = False
        content = dock.widget()
        if content is not None:
            content.installEventFilter(self)

    def eventFilter(self, obj, event):           # noqa: N802 - Qt naming
        if event.type() == QEvent.Type.LayoutRequest and not self._pending:
            self._pending = True
            QTimer.singleShot(0, self._refit)
        return False

    def _refit(self) -> None:
        self._pending = False
        try:
            fit_dock_to_content(self._dock)
        except RuntimeError:                     # the dock is gone
            pass


def watch_dock_fit(dock) -> None:
    """Fit *dock* to its content now and on every content relayout (idempotent)."""
    if getattr(dock, "_squid_fitter", None) is None:
        dock._squid_fitter = _DockFitter(dock)
    fit_dock_to_content(dock)


def stretch_dock(dock) -> None:
    """Make *dock* the column's ONE stretch consumer: the layer list takes what the fitted
    docks leave, instead of a scrollbar over two rows."""
    from qtpy.QtWidgets import QSizePolicy

    dock.setMinimumHeight(0)
    dock.setMaximumHeight(16777215)
    dock.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    w = dock.widget()
    if w is not None:
        w.setMinimumHeight(0)
        w.setMaximumHeight(16777215)
        w.setSizePolicy(w.sizePolicy().horizontalPolicy(), QSizePolicy.Expanding)


def hoist_left_dock(qt_window, dock) -> None:
    """Put *dock* FIRST in the left column by re-adding every other left dock below it.

    Qt appends docks, so "insert above" is spelled remove-and-re-add. Visibility is kept per
    dock: napari's flat layer list is hidden on purpose (`_hide_flat_layer_list`) and a
    re-add must not resurrect it.
    """
    from qtpy.QtWidgets import QDockWidget

    others = [(d, d.isVisibleTo(qt_window))
              for d in qt_window.findChildren(QDockWidget)
              if d is not dock and qt_window.dockWidgetArea(d) == Qt.LeftDockWidgetArea]
    for d, _ in others:
        qt_window.removeDockWidget(d)
    for d, was_visible in others:
        qt_window.addDockWidget(Qt.LeftDockWidgetArea, d)
        d.setVisible(was_visible)


def append_left_dock(qt_window, widget, *, name: str):
    """Dock *widget* LAST in the left column (Qt appends), title slimmed to nothing and kept
    slim. The plate view + log slots ride this (Julio, 2026-08-25: under the layer controls
    and the layer toggles, not above)."""
    from qtpy.QtWidgets import QDockWidget

    dock = QDockWidget(name, qt_window)
    dock.setObjectName(name)
    dock.setWidget(widget)
    dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
    qt_window.addDockWidget(Qt.LeftDockWidgetArea, dock)
    slim_dock_title(dock)
    dock.visibilityChanged.connect(lambda vis, d=dock: keep_dock_slim(d) if vis else None)
    return dock


def slim_dock_title(dock) -> None:
    """Replace *dock*'s title bar with a zero-height widget: one window, so docks neither
    float nor close, and the ~20 px per title goes to the hero surfaces instead."""
    bar = QWidget(dock)
    bar.setFixedHeight(0)
    dock.setTitleBarWidget(bar)


def keep_dock_slim(dock) -> None:
    """Re-slim *dock* if something gave it a title bar back. napari's QtViewerDockWidget
    re-installs its QtCustomTitleBar on EVERY visibilityChanged(True) (measured: the two
    visible docks came back with 20 px titles right after show), so the pane connects this
    after napari's own handler."""
    tb = dock.titleBarWidget()
    if tb is None or tb.maximumHeight() > 0:
        slim_dock_title(dock)


def minimize_native_chrome(qt_window, controls_widgets=()) -> "list[str]":
    """Hide/slim the napari-native chrome the app does not need, and NAME what was hidden.

    Three cuts, each measured on the embedded window (probe 2026-08-25): napari's own
    status bar (27 px; the app's log panel is the one status surface), every dock's title
    bar (20 px each; one window, docks neither float nor close), and the layer-controls
    rows in :data:`NATIVE_HIDDEN_ROWS`. Returns the inventory; empty means nothing was
    found to hide, which a caller should say rather than assume."""
    from qtpy.QtWidgets import QDockWidget, QStatusBar

    hidden: "list[str]" = []
    for bar in qt_window.findChildren(QStatusBar):
        if bar.isVisibleTo(qt_window):
            hidden.append("status bar")
        bar.hide()
    for dock in qt_window.findChildren(QDockWidget):
        tb = dock.titleBarWidget()
        if tb is None or tb.maximumHeight() > 0:
            slim_dock_title(dock)
            hidden.append(f"dock title: {dock.windowTitle() or dock.objectName()}")
    for w in controls_widgets:
        hidden.extend(f"layer-controls row: {t}" for t in hide_native_rows(w))
    return hidden


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

        # NO BANNER STRIP (Julio, 2026-08-25: "I don't like the red strip that appears
        # above the window when I run an operator. That should appear in the logger.").
        # `say` routes through this readout: refusal-shaped text at WARNING, status at
        # INFO, `.text()` the seam tools/gates and tests assert on. The collapsed log
        # band's own latest-line display is what keeps a refusal noticed.
        from squidxplorer._logpane import StatusReadout, get_logger

        self.readout = StatusReadout(get_logger("view"))
        #: Recording seam (tests and gates assert on it); never a pixel.
        self.said: "list[str]" = []

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
                "napari viewer unavailable - falling back to ndviewer_light.\n"
                f"{self.failure}"
            )
            msg.setAlignment(Qt.AlignCenter)
            msg.setWordWrap(True)
            msg.setStyleSheet("color:#ffd7d7;background:#3a2020;padding:12px;")
            lay.addWidget(msg, 1)

    def _install_ndisplay_button(self, lay) -> None:
        """Keep napari's own ndisplay button alive (hidden) and follow its 2D/3D state.

        The visible control row that lived here — the Detect-nuclei channel picker and button —
        was shelved 2026-08-24 with the spot/cellpose operators; the hidden button object stays
        because napari's ndisplay state-sync closure holds it.
        """
        try:
            from napari._qt.widgets.qt_viewer_buttons import QtViewerButtons

            self._button_source = QtViewerButtons(self._viewer)
            btn = self._button_source.ndisplayButton
            # 3D is the ROI native popout, not an embedded toggle. Keep the button object
            # alive for napari's ndisplay state-sync closure, but never show it.
            btn.hide()
            apply_ndisplay_tooltip(btn)      # a tooltip only: this stays NAPARI's 3D button
            self.ndisplay_button = btn
        except Exception as exc:                 # noqa: BLE001 - said out loud, never swallowed
            self.say(f"napari's 2D/3D button could not be mounted ({exc}); "
                     "use the one at the bottom of napari's left column.")

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
            self.layer_tree_dock = self._viewer.window.add_dock_widget(
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
                "showing the bare canvas instead - controls will look wrong."
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
        if self.show_docks:
            self._minimize_native_chrome(qt_window)

    #: What the chrome diet hid on this pane, for tests and the log. Empty until the embed.
    native_chrome_hidden: "list[str]" = []

    def _minimize_native_chrome(self, qt_window) -> None:
        """Hero declutter (2026-08-25): hide the napari chrome the app does not need, and
        keep hiding the layer-controls rows as napari builds new per-layer controls (its
        container makes one widget per layer, lazily, so a one-shot pass would miss every
        layer added after open)."""
        from squidxplorer._logpane import get_logger

        container = None
        try:
            container = self._viewer.window._qt_viewer.controls
        except Exception as exc:                 # noqa: BLE001 - napari moved it: say so
            get_logger("napari_pane").debug(
                "the layer-controls container is unreachable (%s); its rows stay.", exc)
        widgets = []
        if container is not None:
            try:
                widgets = [container.widget(i) for i in range(container.count())]
                container.currentChanged.connect(self._diet_current_controls)
            except Exception as exc:             # noqa: BLE001 - degrade to the one-shot pass
                get_logger("napari_pane").debug(
                    "layer-controls diet cannot follow new layers (%s).", exc)
        self._controls_container = container
        self.native_chrome_hidden = minimize_native_chrome(qt_window, widgets)
        if container is not None:
            try:
                fit_controls_container(container)
            except Exception:                    # noqa: BLE001 - cosmetic, never fatal
                pass
        self._balance_left_column()
        # Slim titles must STAY slim: napari re-installs its title bar on every
        # visibilityChanged(True); connected here (after napari's own handler) so ours
        # runs last in the same emission.
        from qtpy.QtWidgets import QDockWidget

        for dock in qt_window.findChildren(QDockWidget):
            try:
                dock.visibilityChanged.connect(
                    lambda vis, d=dock: keep_dock_slim(d) if vis else None)
            except Exception:                    # noqa: BLE001 - cosmetic, never fatal
                pass
        get_logger("napari_pane").debug(
            "napari chrome minimized: %s", ", ".join(self.native_chrome_hidden) or "nothing")

    def _diet_current_controls(self, index: int) -> None:
        """Apply the row diet to the controls widget napari just switched to (new layers
        get fresh controls widgets; this keeps the diet on all of them)."""
        container = getattr(self, "_controls_container", None)
        if container is None:
            return
        try:
            hide_native_rows(container.widget(int(index)))
            fit_controls_container(container)
        except Exception:                        # noqa: BLE001 - cosmetic, never fatal
            pass

    def dock_plate_slots(self, widget: QWidget) -> bool:
        """Dock the plate view + log host LAST in napari's left column, under the layer
        controls and the layer list. False sends the caller to the window body."""
        if self._viewer is None or self._native_window is None or not self.show_docks:
            return False
        try:
            self.plate_slots_dock = append_left_dock(self._native_window, widget,
                                                     name="plate · log")
        except Exception as exc:                 # noqa: BLE001 - the caller has a fallback
            self.say(f"the plate and log slots could not be docked ({type(exc).__name__}: "
                     f"{exc}); they are in the window body instead.")
            return False
        watch_dock_fit(self.plate_slots_dock)
        self._balance_left_column()
        return True

    #: The grouped layer tree's dock, once mounted: the left column's one stretch consumer.
    layer_tree_dock = None

    def _balance_left_column(self) -> None:
        """Fixed docks at their content, the layer list stretching (see fit_dock_to_content).
        Idempotent; called whenever a dock joins the column and at the chrome diet."""
        try:
            controls_dock = self._viewer.window._qt_viewer.dockLayerControls
        except Exception:                        # noqa: BLE001 - napari moved it: leave it
            controls_dock = None
        if controls_dock is not None:
            try:
                watch_dock_fit(controls_dock)
            except Exception:                    # noqa: BLE001 - cosmetic, never fatal
                pass
        tree_dock = self.layer_tree_dock
        if tree_dock is not None:
            try:
                stretch_dock(tree_dock)
            except Exception:                    # noqa: BLE001 - cosmetic, never fatal
                pass

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
        try:
            slim_dock_title(dock)                # the chips are their own identity; no title bar
            dock.visibilityChanged.connect(
                lambda vis, d=dock: keep_dock_slim(d) if vis else None)
        except Exception:                        # noqa: BLE001 - cosmetic, never fatal
            pass
        self.view_controls_dock = dock
        watch_dock_fit(dock)
        self._balance_left_column()
        return True

    def _hoist_left_dock(self, dock) -> None:
        hoist_left_dock(self._native_window, dock)

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
        """Tell the user via the LOGGER (the banner strip is retired, 2026-08-25); the
        collapsed log band shows the latest line so this is still seen without expanding."""
        if text:
            self.said.append(str(text))
            del self.said[:-500]                 # a seam, not a history
        self.readout.setText(text)

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


def attach_async_slice_apply(viewer):
    """The Qt half a headless ``ViewerModel`` lacks: APPLY async slice responses.

    napari's ONLY consumer of ``_layer_slicer.events.ready`` is ``QtViewer._on_slice_ready``,
    so a bare ``ViewerModel`` computes async slices that never land on the layer. This mirrors
    that handler verbatim, ALWAYS marshalled to the main thread: an inline apply from the
    slicing thread reaches Qt-connected listeners and aborts the process (measured: SIGABRT
    mid-suite). A caller with no running event loop pumps ``QApplication.processEvents`` to
    drain the queued applies. Returns the handler; a napari that moved the seam degrades to
    sync-only slicing with a named log line, never a crash.
    """
    try:
        from superqt.utils import ensure_main_thread

        @ensure_main_thread
        def _apply(event) -> None:
            for weak_layer, response in event.value.items():
                layer = weak_layer()
                if layer is None:
                    continue
                layer._update_slice_response(response)
                layer._update_loaded_slice_id(response.request_id)
                layer.events.set_data()
                layer._refresh_sync(data_displayed=False, thumbnail=True,
                                    highlight=True, extent=True)

        viewer._layer_slicer.events.ready.connect(_apply)
        return _apply
    except AttributeError as exc:                # napari moved the slicer seam
        from squidxplorer._logpane import get_logger

        get_logger("napari_pane").warning(
            "async slice responses cannot be applied on this napari (%s); the headless "
            "pane stays synchronous.", exc)
        return None


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
            from squidxplorer._logpane import StatusReadout, get_logger

            self._viewer = ViewerModel()
            # The QtViewer half: without it, async slices compute and never land.
            self._async_apply = attach_async_slice_apply(self._viewer)
            self.mosaic = MosaicLayers(self._viewer)
            self.said = []
            # The same seam the real pane has: say() IS a log line (banner retired
            # 2026-08-25), and .readout.text() is what harnesses assert on.
            self.readout = StatusReadout(get_logger("view"))
            self.shutdowns = 0
            self._on_settle = None

        def on_camera_settled(self, callback):
            # The real pane debounces camera events into this; headless harnesses (conftest,
            # GATE 3) fire pane._on_settle() themselves so the brick-refine chain is reachable.
            self._on_settle = callback

        def say(self, text):
            self.said.append(text)
            self.readout.setText(text)

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
