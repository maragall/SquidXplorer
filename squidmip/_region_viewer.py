"""Decentralized viewer windows: one INDEPENDENT napari window per selection.

WHY THIS EXISTS. The app was one locked window that owned a plate view, a central napari pane
and a right "exploration" pane, wired together in a splitter so the whole thing moved as a slab.
Spencer's brief (2026-07-23 call) is the opposite: the plate is the ROOT, and clicking a
selection opens an INDEPENDENT napari window that floats on the desktop. Many wells become ONE
window with a region slider, not many windows. Every open window is tracked by ID in an "Open
View list" so the user can raise it. That is what this module builds.

Nothing here reinvents napari. Each window is a ``MosaicPane`` — the same full napari window the
central pane was — placed in its own ``QMainWindow``. Navigation is the same ``RegionCursor`` +
``RegionSlider`` the central pane used. The mosaic load is the same ``_MosaicWorker`` fusing FOVs
off the GUI thread. The only new thing is that these pieces are now instanced PER WINDOW instead
of once for a locked central pane, and a registry tracks the windows.

The reader is stateless (``reader.read(region, fov, channel, z)`` is a pure keyed read), so every
window SHARES the one reader/meta the root opened. No window reopens the dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger("squidmip.regionviewer")

#: Cross-window LUT clipboard for Julio's "sync windows = copy/paste LUTs": one window's per-channel
#: (contrast_limits, colormap) is stashed here by "Copy LUTs" and applied by "Paste LUTs" in any
#: other window (or the plate). A parameter file on the desktop is the same idea; this is the
#: in-session GUI form of it. Keyed by channel name -> {"clim": (lo, hi), "cmap": <name>}.
_LUT_CLIPBOARD: "dict[str, dict]" = {}


@dataclass(frozen=True)
class View:
    """The ONE thing an operator targets: a named set of regions.

    Spencer, 2026-07-23: "operators should really only work on Views ... we need the option to copy
    the whole plate if we're going to do something like decon the whole plate." A plate selection, a
    whole plate, an open window, and an ROI child are ALL Views — same shape, different origin. This
    is the data model that de-convolutes "run on selection vs window vs plate": there is only "run on
    a View's regions". Operators are per-View, not homogeneous-across-windows.

    ``kind`` records the origin so a UI can label it ('window' | 'plate' | 'selection' | 'roi');
    ``window_id`` is set when the View is backed by an open window (else None). Building the tab /
    selector UI over ``PlateWindow.available_views()`` is Spencer's operate-on-views lane; this
    model + the engine hook (``run_on_view``) is the plumbing under it."""
    id: str
    name: str
    regions: tuple
    kind: str = "window"
    window_id: Optional[int] = None
    roi_bbox: Optional[tuple] = None
    parent_id: Optional[int] = None


def _level_shape(level: Any) -> "Optional[tuple[int, int]]":
    """The (height, width) of one pyramid level, or None if it has no 2-D+ shape."""
    shp = getattr(level, "shape", None)
    if not shp or len(shp) < 2:
        return None
    return int(shp[-2]), int(shp[-1])


def _crop_levels_to_bbox(levels: "list", region_bbox_um: "Sequence[float]",
                         roi_bbox_um: "Sequence[float]"):
    """Crop a LAZY multiscale pyramid to an ROI box, returning ``(cropped_levels, cropped_bbox_um)``
    or ``None`` if the ROI does not overlap the region.

    Both boxes are ``(x0, y0, x1, y1)`` in stage micrometres — the same space ``mosaic_bbox_um``
    speaks. The levels are lazy (dask), so slicing them reads NOTHING; napari then materialises only
    the ROI sub-array. That is the whole point of an ROI child: read a corner, not the region. The
    returned bbox is derived from level 0's integer crop so placement lands exactly on the ROI."""
    try:
        x0, y0, x1, y1 = (float(v) for v in region_bbox_um)
        rx0, ry0, rx1, ry1 = (float(v) for v in roi_bbox_um)
    except Exception:                                    # noqa: BLE001 - malformed box, skip crop
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    # Clip the ROI to the region: a box dragged past the edge still crops to what exists.
    rx0, rx1 = max(min(rx0, rx1), x0), min(max(rx0, rx1), x1)
    ry0, ry1 = max(min(ry0, ry1), y0), min(max(ry0, ry1), y1)
    if rx1 - rx0 <= 0 or ry1 - ry0 <= 0:
        return None
    span_x, span_y = x1 - x0, y1 - y0
    out: list = []
    l0 = None
    for lvl in levels:
        shp = _level_shape(lvl)
        if shp is None:
            continue
        h, w = shp
        sx, sy = w / span_x, h / span_y
        c0 = int(max(0, min(w - 1, round((rx0 - x0) * sx))))
        c1 = int(max(c0 + 1, min(w, round((rx1 - x0) * sx))))
        r0 = int(max(0, min(h - 1, round((ry0 - y0) * sy))))
        r1 = int(max(r0 + 1, min(h, round((ry1 - y0) * sy))))
        out.append(lvl[..., r0:r1, c0:c1])
        if l0 is None:
            l0 = (c0, c1, r0, r1, w, h)                  # level 0 defines the returned bbox
    if not out or l0 is None:
        return None
    c0, c1, r0, r1, w0, h0 = l0
    nbbox = (x0 + (c0 / w0) * span_x, y0 + (r0 / h0) * span_y,
             x0 + (c1 / w0) * span_x, y0 + (r1 / h0) * span_y)
    try:
        from squidmip._mosaic_source import strictly_decreasing_levels
        out = strictly_decreasing_levels(out)
    except Exception:                                    # noqa: BLE001 - a 1-level pyramid is fine
        pass
    return out, nbbox

#: Debounce before a settled region is fused, matching the central pane's 140 ms. The red frame /
#: slider move instantly; only the expensive fuse waits for the slider to stop, so a drag across
#: ten regions fuses ONE mosaic instead of ten. See _region_nav for why the region is not an axis.
_REGION_LOAD_DEBOUNCE_MS = 140

#: Processing layer key for the raw fused mosaic (mirrors _viewer's "raw"). Operators that write
#: an OME-Zarr will add their own op key as a second visibility layer; not needed for exploration.
_RAW_OP = "raw"


class RegionViewer(QMainWindow):
    """ONE independent napari window over a subset of regions.

    Owns its own napari pane, its own region cursor + slider, and its own mosaic-load pipeline.
    Shares the app's single ``reader``/``meta`` (stateless reads). Closing it stops its worker and
    joins its slider's animation thread so a close during playback cannot abort the process.
    """

    closed = pyqtSignal(object)   # emits self, so the registry can drop it

    def __init__(
        self,
        reader: Any,
        meta: dict,
        regions: Sequence[str],
        *,
        window_id: int,
        title: Optional[str] = None,
        parent: Optional[QWidget] = None,
        manager: Optional["ViewerManager"] = None,
        roi_bbox: Optional[tuple] = None,
        operator_specs: Optional[Sequence] = None,
        run_operator: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self._reader = reader
        self._meta = meta
        self._regions = [str(r) for r in regions]
        self.window_id = int(window_id)
        self._worker = None
        self._pending_region: Optional[str] = None
        self._load_timer: Optional[QTimer] = None
        self._pane = None
        self._slider = None
        self._cursor = None
        self._native3d = None      # keeps a spawned 3D popout viewer alive
        # OPERATOR CONTROLS AT EACH LEVEL (the deck: "Operators for this window"; Julio, 2026-07-23:
        # "I don't see operator controls like the powerpoint specified at each level"). This is not a
        # contradiction of "operators work on Views" -- it IS that: the window's operator control runs
        # the SAME registry on THIS view's regions. Selecting where to run stitching = pick the view,
        # run it here. The manager also lets an ROI open a CHILD window (the view tree).
        self._manager = manager
        self._operator_specs = list(operator_specs or [])
        self._run_operator = run_operator
        # An ROI child carries the parent's ROI box (deck: "ROI -> child window"). Cropping the load
        # to it lands with the loader work; today it scopes the title + is recorded for that step.
        self._roi_bbox = roi_bbox
        self._roi_layer = None     # the napari Shapes layer this window draws ROI rectangles on

        # Name the window by the regions it holds (the deck shows the slider as "<> A1, B6, C3"),
        # not "N regions" — Julio: "'2 regions' is a bad name". Truncate a long list so the title
        # bar stays readable, keeping the count only as an overflow tail.
        label = title or self._region_label(self._regions)
        if self._roi_bbox is not None:
            label = f"ROI · {label}"
        self.setWindowTitle(f"[{self.window_id}] {label}")
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        # A modest, cascaded window — the deck's windows are small tiles, not full-screen slabs.
        # Cascade by ID so several opened in a row do not land exactly on top of one another.
        self.resize(860, 720)
        off = 28 * ((self.window_id - 1) % 8)
        self.move(120 + off, 90 + off)

        self._build()

    @staticmethod
    def _region_label(regions: "list[str]", limit: int = 3) -> str:
        if not regions:
            return "(empty)"
        if len(regions) <= limit:
            return ", ".join(regions)
        return ", ".join(regions[:limit]) + f", +{len(regions) - limit}"

    # -- construction -------------------------------------------------------------------
    def _build(self) -> None:
        from squidmip._napari_pane import make_pane
        from squidmip._region_nav import RegionCursor, RegionSlider

        central = QWidget(self)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        pane, mode, message = make_pane(show_docks=True)
        if pane is None or not getattr(pane, "ok", False):
            # No napari here. Say why, out loud, in the window — never a blank floater.
            msg = QLabel(f"napari viewer unavailable — {message}")
            msg.setAlignment(Qt.AlignCenter)
            msg.setWordWrap(True)
            msg.setStyleSheet("color:#ffd7d7;background:#3a2020;padding:16px;font-size:13px;")
            lay.addWidget(msg, 1)
            self.setCentralWidget(central)
            return
        self._pane = pane

        # DECK LAYOUT for a child window (2026-07-23 deck, per-window slide): a TOP ROW of two
        # panels — [2D / 3D + ROI tools] on the left, [Operators for THIS window] on the right —
        # over the mosaic viewer (full well), with the region slider at the bottom. The ROI
        # rectangles are drawn INSIDE the mosaic and open child windows (the next level of the tree).
        lay.addWidget(self._build_top_row(), 0)
        lay.addWidget(pane, 1)

        # THE REGION SLIDER — napari's own dims slider driven by our region cursor. One owner of
        # "which region is current"; the slider and the loader are subscribers, never opinions.
        self._cursor = RegionCursor()
        self._cursor.on_problem(self._say)
        self._cursor.subscribe(self._on_region_changed)
        self._slider = RegionSlider()
        self._slider.on_problem(self._say)
        self._slider.bind(self._cursor)
        lay.addWidget(self._slider)

        self.setCentralWidget(central)

        # Seed the cursor: this announces region 0 to the loader, so the first mosaic loads now.
        self._cursor.set_order(self._regions)
        if self._cursor.index is None and self._regions:
            self._cursor.set_index(0)

    # -- the deck's per-window top row --------------------------------------------------
    _BOX_QSS = "QFrame{background:#0d1117;border:1px solid #232b3a;border-radius:5px;}"
    _TITLE_QSS = "color:#8b949e;font-size:10px;font-weight:700;border:none;"
    _CHIP_QSS = (
        "QPushButton{background:#161b22;color:#c9d1d9;border:1px solid #30363d;"
        "border-radius:4px;padding:3px 9px;font-size:11px;}"
        "QPushButton:hover{background:#21262d;}"
        "QPushButton:checked{background:#1f6feb;color:#ffffff;border-color:#1f6feb;}"
        "QPushButton:disabled{color:#586069;border-color:#20262e;}"
    )

    def _titled_box(self, title: str) -> "tuple[QFrame, QVBoxLayout]":
        box = QFrame(self)
        box.setStyleSheet(self._BOX_QSS)
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 5, 8, 6)
        v.setSpacing(4)
        lab = QLabel(title)
        lab.setStyleSheet(self._TITLE_QSS)
        v.addWidget(lab)
        return box, v

    def _chip(self, text: str, tip: str, slot, *, checkable: bool = False) -> QPushButton:
        b = QPushButton(text)
        b.setToolTip(tip)
        b.setCheckable(checkable)
        b.setCursor(Qt.PointingHandCursor)
        b.setStyleSheet(self._CHIP_QSS)
        b.clicked.connect(lambda _=False: slot())
        return b

    def _build_top_row(self) -> QWidget:
        """[ 2D / 3D + ROI ]   [ Operators for this window ] — the deck's per-window header."""
        row = QWidget(self)
        row.setStyleSheet("background:#0b0e14;")
        h = QHBoxLayout(row)
        h.setContentsMargins(6, 6, 6, 2)
        h.setSpacing(6)

        # LEFT: the "2D 3D'" toggle + native-3D popout + ROI tools. The toggle drives the embedded
        # pane's ndisplay (which already renders 3D at max texture res, contrast preserved); the
        # popout is the single-FOV native volume for when the fused mosaic exceeds the GPU texture.
        view_box, vv = self._titled_box("2D / 3D · ROI")
        r1 = QHBoxLayout(); r1.setSpacing(4)
        self._btn_2d = self._chip("2D", "Show the mosaic in 2D.", lambda: self._set_ndisplay(2))
        # 3D is ONE thing: a NATIVE-resolution popout of this view (never the whole fused mosaic,
        # which exceeds the GPU texture and renders blocky). 2D just keeps the mosaic. No embedded
        # 3D toggle, no separate "native" button -- one behaviour, so the cases don't explode.
        self._btn_3d = self._chip("3D", "Open this view in 3D at NATIVE resolution (the region if it "
                                  "fits the GPU texture, else draw an ROI to pick the spot).",
                                  self._open_3d)
        r1.addWidget(self._btn_2d); r1.addWidget(self._btn_3d)
        r1.addStretch(1)
        vv.addLayout(r1)
        r2 = QHBoxLayout(); r2.setSpacing(4)
        r2.addWidget(self._chip("▭ new", "Draw an ROI rectangle inside the mosaic.", self._new_roi))
        r2.addWidget(self._chip("⊙ select", "Select ROIs: click one, then press Delete to remove it.",
                                self._select_rois))
        r2.addWidget(self._chip("✕ clear", "Remove all ROIs in this window.", self._clear_rois))
        r2.addWidget(self._chip("→ window", "Open the drawn ROI(s) as child window(s) — the next "
                                "level of the view tree.", self._open_roi_children))
        r2.addStretch(1)
        vv.addLayout(r2)
        view_box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        h.addWidget(view_box, 0)

        # RIGHT: contrast sync. Operators DELIBERATELY do NOT live in the window (Spencer, 2026-07-23:
        # "operators should really only work on Views", picked centrally, not per window). What the
        # window carries is LUT sync — the GUI form of "sync windows by copy-pasting a parameter
        # file", and Spencer scoped sync to LUTs specifically (FF correction is an operation, not a
        # LUT). Annotation (Julio's lane) lands beside these next.
        op_box, ov = self._titled_box("Operators for this window")
        # Row 1: pick an operator, Run it on THIS view (a dropdown, per Julio's "hierarchies should be
        # drop-down menus"). Runs the same registry the plate uses, scoped to this window's regions.
        opr = QHBoxLayout(); opr.setSpacing(4)
        self._op_combo = QComboBox()
        self._op_combo.setStyleSheet(self._CHIP_QSS + "QComboBox{min-width:120px;}")
        for spec in self._operator_specs:
            self._op_combo.addItem(str(spec[1]), spec[0])   # label shown, key as data
        if self._op_combo.count() == 0:
            self._op_combo.addItem("no operators", None)
            self._op_combo.setEnabled(False)
        opr.addWidget(self._op_combo, 1)
        opr.addWidget(self._chip("Run", "Run the selected operator on THIS view's regions.",
                                 self._run_view_operator))
        ov.addLayout(opr)
        # Row 2: contrast sync (copy/paste LUTs) — window <-> window <-> plate.
        sync = QHBoxLayout(); sync.setSpacing(4)
        sync.addWidget(self._chip("⧉ Copy LUTs", "Copy this window's per-channel contrast + colormap.",
                                  self._copy_luts))
        sync.addWidget(self._chip("⤓ Paste LUTs", "Apply the copied LUTs to this window's channels.",
                                  self._paste_luts))
        sync.addStretch(1)
        ov.addLayout(sync)
        h.addWidget(op_box, 1)

        row.setMaximumHeight(108)
        return row

    def _run_view_operator(self) -> None:
        """Run the operator picked in this window's dropdown on THIS view's regions — "select where to
        run stitching" = pick this view, Run. Uses the app's real engine (no reimplementation)."""
        if self._run_operator is None:
            self._say("the operator engine isn't connected to this window.")
            return
        key = self._op_combo.currentData() if getattr(self, "_op_combo", None) is not None else None
        if not key:
            self._say("no operator selected.")
            return
        regions = list(self._regions)
        try:
            self._run_operator(key, regions=regions)
            self._say(f"running {self._op_combo.currentText()} on {self._region_label(regions)}.")
        except Exception as exc:                          # noqa: BLE001 - named to the window
            self._say(f"could not start {self._op_combo.currentText()}: {exc}")

    def _napari_viewer(self):
        """The live napari ``Viewer`` behind this window's pane, or None if unavailable."""
        pane = self._pane
        if pane is None or not getattr(pane, "ok", False):
            return None
        mosaic = getattr(pane, "mosaic", None)
        v = getattr(mosaic, "model", None) if mosaic is not None else None
        return v if v is not None else getattr(pane, "_viewer", None)

    def _set_ndisplay(self, n: int) -> None:
        v = self._napari_viewer()
        if v is None:
            self._say(f"cannot switch to {n}D — the napari viewer isn't available here.")
            return
        try:
            v.dims.ndisplay = int(n)
        except Exception as exc:                         # noqa: BLE001 - named, never silent
            self._say(f"could not switch to {n}D: {exc}")

    # -- ROI -> child window (the next level of the tree) --------------------------------
    @staticmethod
    def _sync_roi_width(viewer, layer, screen_px: float = 3.0) -> None:
        """Keep the ROI border a ~constant thickness ON SCREEN as you zoom (Julio: "ROI width should
        react to zoom level"). napari's edge_width is in DATA units, so the world width for a given
        screen thickness is screen_px / camera.zoom (zoom = screen px per data unit)."""
        try:
            zoom = float(getattr(viewer.camera, "zoom", 1.0)) or 1.0
            w = max(1e-6, float(screen_px) / zoom)
            layer.edge_width = w
            layer.current_edge_width = w
        except Exception:                                # noqa: BLE001 - width is cosmetic
            pass

    def _roi_shapes_layer(self, create: bool = False):
        """This window's ROI Shapes layer (creating it, zoom-reactive, on first use if asked)."""
        v = self._napari_viewer()
        if v is None:
            return None, None
        layer = self._roi_layer
        if layer is None or layer not in list(v.layers):
            if not create:
                return v, None
            layer = v.add_shapes(name="ROIs", edge_color="#58a6ff", face_color="transparent")
            self._roi_layer = layer
            self._sync_roi_width(v, layer)
            try:                                         # border reacts to zoom from here on
                v.camera.events.zoom.connect(
                    lambda e=None, vv=v, ly=layer: self._sync_roi_width(vv, ly))
            except Exception:                            # noqa: BLE001
                pass
        return v, layer

    def _new_roi(self) -> None:
        """Start drawing an ROI rectangle inside the mosaic (deck: boxes inside the well view)."""
        v, layer = self._roi_shapes_layer(create=True)
        if v is None or layer is None:
            self._say("ROI needs the napari viewer, which isn't available here.")
            return
        try:
            v.layers.selection.active = layer
            layer.mode = "add_rectangle"
            self._say("Draw an ROI rectangle, then '→ window' to open it as a child window.")
        except Exception as exc:                         # noqa: BLE001
            self._say(f"could not start an ROI: {exc}")

    def _select_rois(self) -> None:
        """Enter select mode so an ROI can be clicked and deleted (Julio: "how do I delete ROIs")."""
        v, layer = self._roi_shapes_layer(create=False)
        if v is None or layer is None:
            self._say("draw an ROI first with '▭ new'.")
            return
        try:
            v.layers.selection.active = layer
            layer.mode = "select"
            self._say("Select mode: click an ROI, then press Delete/Backspace to remove it.")
        except Exception as exc:                         # noqa: BLE001
            self._say(f"could not enter select mode: {exc}")

    def _clear_rois(self) -> None:
        """Remove every ROI in this window."""
        v, layer = self._roi_shapes_layer(create=False)
        if v is None or layer is None or not list(getattr(layer, "data", []) or []):
            self._say("no ROIs to clear.")
            return
        try:
            layer.data = []
            self._say("cleared all ROIs.")
        except Exception as exc:                         # noqa: BLE001
            self._say(f"could not clear ROIs: {exc}")

    def _open_roi_children(self) -> None:
        """Open the drawn ROI(s) as child window(s) — "first dev step: one ROI, one child window"."""
        v = self._napari_viewer()
        layer = self._roi_layer
        rects = list(getattr(layer, "data", []) or []) if layer is not None else []
        if v is None or layer is None or layer not in list(v.layers) or not rects:
            self._say("no ROI to open — draw one with '▭ ROI' first.")
            return
        if self._manager is None:
            self._say(f"{len(rects)} ROI(s) drawn, but this window has no manager to open children.")
            return
        opened = 0
        for rect in rects:
            bbox = None
            try:
                arr = np.asarray(rect)
                ys, xs = arr[:, -2], arr[:, -1]        # world coords are (..., y, x)
                bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            except Exception:                            # noqa: BLE001 - a shapeless ROI still opens
                pass
            child = self._manager.open_child(
                self._regions, roi_bbox=bbox, parent_id=self.window_id)
            if child is not None:
                opened += 1
        self._say(f"opened {opened} ROI child window(s).")

    # -- copy/paste LUTs: sync windows without a parameter file --------------------------
    def _per_channel_luts(self) -> "dict[str, dict]":
        out: "dict[str, dict]" = {}
        pane = self._pane
        mosaic = getattr(pane, "mosaic", None) if pane is not None else None
        if mosaic is None:
            return out
        for c in (self._meta or {}).get("channels", []):
            name = c["name"]
            layer = mosaic.find(_RAW_OP, name)
            if layer is None:
                continue
            lut: dict = {}
            try:
                lut["clim"] = tuple(layer.contrast_limits)
            except Exception:                            # noqa: BLE001
                lut["clim"] = None
            try:
                cmap = layer.colormap
                lut["cmap"] = getattr(cmap, "name", cmap)
            except Exception:                            # noqa: BLE001
                lut["cmap"] = None
            out[name] = lut
        return out

    def _copy_luts(self) -> None:
        caught = self._per_channel_luts()
        if not caught:
            self._say("no channels on screen to copy LUTs from.")
            return
        _LUT_CLIPBOARD.clear()
        _LUT_CLIPBOARD.update(caught)
        self._say(f"copied LUTs for {len(caught)} channel(s) — paste them into another window.")

    def _paste_luts(self) -> None:
        if not _LUT_CLIPBOARD:
            self._say("no copied LUTs yet — use '⧉ Copy LUTs' in another window first.")
            return
        pane = self._pane
        mosaic = getattr(pane, "mosaic", None) if pane is not None else None
        if mosaic is None:
            self._say("no mosaic here to paste LUTs onto.")
            return
        applied = 0
        for ch, lut in _LUT_CLIPBOARD.items():
            layer = mosaic.find(_RAW_OP, ch)
            if layer is None:
                continue
            try:
                if lut.get("clim") is not None:
                    layer.contrast_limits = tuple(lut["clim"])
                if lut.get("cmap") is not None:
                    layer.colormap = lut["cmap"]
                applied += 1
            except Exception:                            # noqa: BLE001 - a missing channel is skipped
                pass
        self._say(f"pasted LUTs onto {applied} channel(s).")

    # -- navigation ---------------------------------------------------------------------
    def _on_region_changed(self, index: int, region: str) -> None:
        """Current region moved. Debounce the fuse; the slider label already moved instantly."""
        if getattr(self, "_load_timer", None) is None:
            self._load_timer = QTimer(self)
            self._load_timer.setSingleShot(True)
            self._load_timer.timeout.connect(
                lambda: self._load_mosaic(self._pending_region))
        self._pending_region = region
        self._load_timer.start(_REGION_LOAD_DEBOUNCE_MS)

    def _load_mosaic(self, region: Optional[str]) -> None:
        """Fuse one region's FOVs into this window's napari pane, one layer per channel."""
        pane = self._pane
        if pane is None or not getattr(pane, "ok", False):
            return
        if self._reader is None or self._meta is None or not region:
            return
        from squidmip._viewer import _MosaicWorker

        prior = self._worker
        if prior is not None and prior.isRunning():
            prior.stop()
            prior.wait(2000)

        pane.mosaic.remove_op(_RAW_OP)
        channels = [c["name"] for c in self._meta["channels"]]
        w = _MosaicWorker(self._reader, self._meta, region, channels, z_index=0, parent=self)
        w.ready.connect(lambda r, ch, levels, bbox: self._on_plane(r, ch, levels, bbox))
        w.problem.connect(self._say)
        w.finished_count.connect(lambda n: self._on_done(region, n))
        self._worker = w
        w.start()

    def _on_plane(self, region: str, channel: str, levels, bbox_um) -> None:
        pane = self._pane
        if pane is None or not getattr(pane, "ok", False):
            return
        if self._cursor is not None and self._cursor.region != region:
            return                                  # a later region won the race; drop this one
        from squidmip._napari_pane import _colormap_for

        # ROI CHILD: crop the lazy pyramid to the ROI box before adding, so napari materialises only
        # the ROI corner (read a corner, not the whole region). A window with no ROI box adds the
        # full region unchanged. The crop also adjusts bbox_um so placement lands on the ROI.
        add_levels, add_bbox = levels, bbox_um
        if self._roi_bbox is not None and bbox_um is not None:
            cropped = _crop_levels_to_bbox(levels, bbox_um, self._roi_bbox)
            if cropped is not None:
                add_levels, add_bbox = cropped
            else:
                self._say("ROI does not overlap this region — showing the whole region.")

        pane.mosaic.add_mosaic(
            _RAW_OP, channel, add_levels,
            colormap=_colormap_for(channel),
            multiscale=True,
            bbox_um=add_bbox,
            z_scale_um=(self._meta or {}).get("dz_um"),
        )

    def _on_done(self, region: str, n: int) -> None:
        pane = self._pane
        if pane is None or not getattr(pane, "ok", False):
            return
        if n == 0:
            pane.say(f"{region}: no mosaic could be built (see the message above).")
            self._frame_done()
            return
        pane.say("")
        try:
            pane.mosaic.show_op(_RAW_OP)
            pane.mosaic.model.reset_view()
        except Exception:                            # noqa: BLE001 - view framing is cosmetic
            pass
        self._frame_done()

    def _frame_done(self) -> None:
        """Open the playback gate: this region is on screen, the next may be requested."""
        if self._slider is not None:
            self._slider.frame_done()

    # -- 2D -> 3D, per window -----------------------------------------------------------
    def _roi_center_fov(self, region: str) -> Optional[int]:
        """The FOV nearest the ROI box's centre (stage um), so an ROI's 3D lands on the tissue you
        boxed. None (region centre) when there is no ROI."""
        if self._roi_bbox is None:
            return None
        x0, y0, x1, y1 = self._roi_bbox
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        positions = (self._meta or {}).get("fov_positions_um") or {}
        fovs = ((self._meta or {}).get("fovs_per_region") or {}).get(region) or []
        best, best_d = None, None
        for f in fovs:
            p = positions.get((region, int(f)))
            if p is None:
                continue
            d = (p[0] - cx) ** 2 + (p[1] - cy) ** 2
            if best_d is None or d < best_d:
                best, best_d = int(f), d
        return best

    def _open_3d(self) -> None:
        """3D = THIS view at NATIVE resolution, read STRAIGHT FROM THE READER (gallery-view recipe).

        Why not the 2D pyramid: its level 0 is itself CAPPED to the fused-plane budget
        (``_MAX_FUSED_PX``), so cropping the pyramid is already downsampled -- that was the "still
        downsampled" bug. One FOV's raw z-stack IS native and fits the GPU texture, so we read the FOV
        under the ROI (or the region centre) directly and carry the EXACT on-screen contrast so 3D
        matches 2D. (Native fusion across the FOVs a large ROI spans is the next step; one native FOV
        is the honest max-res primitive today -- simple, and never downsampled.)"""
        region = self._cursor.region if self._cursor is not None else (
            self._regions[0] if self._regions else None)
        if region is None or self._reader is None or self._meta is None:
            self._say("no region to render in 3D.")
            return
        mosaic = getattr(self._pane, "mosaic", None) if self._pane is not None else None
        contrast_by: dict = {}
        colormap_by: dict = {}
        if mosaic is not None:
            for c in (self._meta or {}).get("channels", []):
                name = c["name"]
                layer = mosaic.find(_RAW_OP, name)
                if layer is None:
                    continue
                try:
                    contrast_by[name] = tuple(layer.contrast_limits)   # EXACT window on screen
                except Exception:                    # noqa: BLE001
                    pass
                try:
                    cmap = layer.colormap
                    colormap_by[name] = getattr(cmap, "name", cmap)
                except Exception:                    # noqa: BLE001
                    pass
        fov = self._roi_center_fov(region)           # ROI -> its FOV; else region centre (None)
        from squidmip._napari3d import open_native_3d

        try:
            self._native3d = open_native_3d(
                self._reader, self._meta, region, fov=fov,
                contrast_by_channel=contrast_by or None,
                colormap_by_channel=colormap_by or None,
            )
        except Exception as exc:                     # noqa: BLE001 - named to the window, never silent
            self._say(f"3D could not open: {exc}")

    def _say(self, text: str) -> None:
        if self._pane is not None and getattr(self._pane, "ok", False):
            self._pane.say(text)
        elif text:
            log.warning("[window %s] %s", self.window_id, text)

    # -- render-halt: a window not being manipulated must not keep drawing ----------------
    def set_active(self, active: bool) -> None:
        """Halt draw/refresh on windows the user is not touching (Spencer's memory brief).

        A window that is not the active one stops its playback so it is not fusing regions in the
        background and competing for the GPU with the window the user is actually looking at.
        """
        if active or self._slider is None:
            return
        try:
            if self._slider.is_playing:
                self._slider.stop()
        except Exception:                            # noqa: BLE001 - best effort
            pass

    def changeEvent(self, event):                    # noqa: N802 - Qt naming
        from PyQt5.QtCore import QEvent

        if event.type() == QEvent.ActivationChange:
            self.set_active(self.isActiveWindow())
        super().changeEvent(event)

    # -- teardown -----------------------------------------------------------------------
    def closeEvent(self, event):                     # noqa: N802 - Qt naming
        try:
            if self._worker is not None and self._worker.isRunning():
                self._worker.stop()
                self._worker.wait(2000)
        except Exception:                            # noqa: BLE001
            pass
        try:
            if self._slider is not None:
                self._slider.shutdown()
        except Exception:                            # noqa: BLE001
            pass
        self.closed.emit(self)
        super().closeEvent(event)


class ViewerManager(QObject):
    """Registry of open :class:`RegionViewer` windows, keyed by a monotonic ID.

    The root plate window owns one of these. It is the single source of "what windows are open",
    so the Open View list is a pure VIEW of it and can never drift from the real set of windows.
    Memory is polled here (not per window) so one warning speaks for the whole app.
    """

    windowsChanged = pyqtSignal()          # the set of open windows changed
    memoryChanged = pyqtSignal(float)      # process RSS as a fraction 0..1 of total RAM
    viewFocused = pyqtSignal(object)       # a window was opened/raised -> its regions (list[str])

    def __init__(self, reader: Any = None, meta: Optional[dict] = None,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._reader = reader
        self._meta = meta
        self._windows: "dict[int, RegionViewer]" = {}
        self._next_id = 1
        self._focused_id: Optional[int] = None    # which view is active (its plate hue reads brighter)
        # Set by the root PlateWindow so every window's "Operators for this window" dropdown is the
        # SAME registry + the SAME run_operator (the CLI engine), scoped to that view.
        self.operator_specs: "list" = []
        self.run_operator: Optional[Any] = None

        self._mem_timer = QTimer(self)
        self._mem_timer.setInterval(2000)
        self._mem_timer.timeout.connect(self._poll_memory)
        self._mem_timer.start()

    def set_dataset(self, reader: Any, meta: dict) -> None:
        self._reader, self._meta = reader, meta

    @property
    def windows(self) -> "list[RegionViewer]":
        return list(self._windows.values())

    def views(self) -> "list[View]":
        """Every open window as a :class:`View` (a named region-set) — the unit an operator targets.

        Spencer's operate-on-views tab UI binds to this + ``PlateWindow.available_views`` (which adds
        the whole-plate and current-selection Views). One list, one concept, no per-surface rules."""
        out: "list[View]" = []
        for win in self.windows:
            roi = getattr(win, "_roi_bbox", None)
            out.append(View(
                id=f"w{win.window_id}", name=win.windowTitle(),
                regions=tuple(win._regions),
                kind="roi" if roi is not None else "window",
                window_id=win.window_id, roi_bbox=roi))
        return out

    def view_for(self, window_id: int) -> "Optional[View]":
        for v in self.views():
            if v.window_id == int(window_id):
                return v
        return None

    def open(self, regions: Sequence[str], *, title: Optional[str] = None) -> Optional[RegionViewer]:
        """Open ONE independent window over *regions*. Many regions => one window with a slider."""
        if self._reader is None or self._meta is None:
            log.warning("open() called before a dataset was loaded; ignoring.")
            return None
        regions = [str(r) for r in regions if r]
        if not regions:
            return None
        return self._spawn(regions, title=title)

    def open_child(self, regions: Sequence[str], *, roi_bbox: Optional[tuple] = None,
                   parent_id: Optional[int] = None) -> Optional[RegionViewer]:
        """Open a CHILD window from an ROI drawn in a parent window (the next level of the tree).

        Structurally the child is a window over the same regions carrying the ROI box; cropping the
        load to the box lands with the loader work. Titled so the Open View list shows the nesting."""
        regions = [str(r) for r in regions if r]
        if not regions:
            return None
        base = RegionViewer._region_label(regions)
        title = f"{base}  ◂ view {parent_id}" if parent_id is not None else base
        return self._spawn(regions, title=title, roi_bbox=roi_bbox)

    def _spawn(self, regions: "list[str]", *, title: Optional[str] = None,
               roi_bbox: Optional[tuple] = None) -> Optional[RegionViewer]:
        if self._reader is None or self._meta is None:
            log.warning("open() called before a dataset was loaded; ignoring.")
            return None
        wid = self._next_id
        self._next_id += 1
        win = RegionViewer(
            self._reader, self._meta, regions, window_id=wid, title=title,
            manager=self, roi_bbox=roi_bbox,
            operator_specs=self.operator_specs, run_operator=self.run_operator,
        )
        win.closed.connect(self._on_window_closed)
        self._windows[wid] = win
        self._focused_id = wid
        win.show()
        win.raise_()
        win.activateWindow()
        self.windowsChanged.emit()
        self.viewFocused.emit(list(win._regions))       # highlight its regions on the plate
        return win

    @property
    def focused_id(self) -> Optional[int]:
        """The window id of the active view (its plate hue reads brighter), or None."""
        return self._focused_id

    def clear_focus(self) -> None:
        """No view is selected -> clear the plate wash. Emitting empty regions makes the plate's hue
        refresh find no focused view and paint nothing."""
        self._focused_id = None
        self.viewFocused.emit([])

    def focus(self, window_id: int) -> None:
        win = self._windows.get(int(window_id))
        if win is not None:
            self._focused_id = int(window_id)
            win.showNormal()
            win.raise_()
            win.activateWindow()
            self.viewFocused.emit(list(win._regions))   # move the plate wash onto this view

    def close(self, window_id: int) -> None:
        win = self._windows.get(int(window_id))
        if win is not None:
            win.close()

    def close_all(self) -> None:
        for win in list(self._windows.values()):
            win.close()

    def _on_window_closed(self, win: "RegionViewer") -> None:
        wid = getattr(win, "window_id", -1)
        self._windows.pop(wid, None)
        if self._focused_id == wid:
            self._focused_id = None
        self.windowsChanged.emit()

    def _poll_memory(self) -> None:
        frac = _process_memory_fraction()
        if frac is not None:
            self.memoryChanged.emit(frac)


class OpenViewList(QWidget):
    """The "Open View list": every open window by ID, plus a live memory bar.

    Clicking a row raises that window to the front of the desktop — the meeting's "give it an ID,
    click it to pop it forward". A flat list of IDs is dev-step one; parent/child nesting (a
    selection's regions, then its ROIs) is the next step and slots onto the same tree.
    """

    def __init__(self, manager: ViewerManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._manager = manager

        # DARK THEME. Without an explicit stylesheet this widget renders WHITE against the dark app
        # (Julio: "Open views window still white") — QTreeWidget/QProgressBar do not inherit the
        # app palette on macOS. Match the plate's palette (#0b0e14 bg, #c9d1d9 text) here.
        self.setStyleSheet(
            "QWidget{background:#0b0e14;color:#c9d1d9;}"
            "QTreeWidget{background:#0d1117;border:1px solid #232b3a;border-radius:4px;"
            "outline:none;}"
            "QTreeWidget::item{padding:4px 6px;}"
            "QTreeWidget::item:selected{background:#1f6feb;color:#ffffff;}"
            "QPushButton{background:#161b22;color:#c9d1d9;border:1px solid #30363d;"
            "border-radius:4px;padding:4px 10px;}"
            "QPushButton:hover{background:#21262d;}"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # "Window navigator", not "Open views": Julio + Spencer (2026-07-23) decoupled the window
        # list from operators — it navigates windows (click to raise), it does not run anything.
        header = QLabel("Window navigator")
        header.setStyleSheet("color:#c9d1d9;font-size:13px;font-weight:600;border:none;")
        lay.addWidget(header)

        self._tree = QTreeWidget(self)
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(False)
        # The plate wash STRICTLY follows the navigator selection: select a row -> wash that view;
        # deselect (or nothing selected) -> no wash. Julio: "nothing is selected, and I still see a
        # region purple washed." itemActivated (double-click) also raises the window.
        self._tree.itemActivated.connect(self._on_activated)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._syncing = False   # guards refresh()'s programmatic selection from re-emitting
        lay.addWidget(self._tree, 1)

        row = QHBoxLayout()
        row.setSpacing(6)
        close_btn = QPushButton("Close view")
        close_btn.clicked.connect(self._close_selected)
        row.addWidget(close_btn)
        row.addStretch(1)
        lay.addLayout(row)

        self._mem_label = QLabel("Memory")
        self._mem_label.setStyleSheet("color:#8b949e;font-size:11px;border:none;")
        lay.addWidget(self._mem_label)
        self._mem_bar = QProgressBar(self)
        self._mem_bar.setRange(0, 100)
        self._mem_bar.setTextVisible(True)
        self._mem_bar.setFixedHeight(14)
        lay.addWidget(self._mem_bar)

        manager.windowsChanged.connect(self.refresh)
        manager.memoryChanged.connect(self._on_memory)
        self.refresh()

    def refresh(self) -> None:
        # Rebuild the rows, then sync the row selection to the manager's focused view (guarded so the
        # programmatic selection does not re-fire _on_selection_changed). No focused view => nothing
        # selected => no wash, which is exactly the state Julio saw violated.
        self._syncing = True
        try:
            self._tree.clear()
            focused = self._manager.focused_id
            for win in self._manager.windows:
                item = QTreeWidgetItem([win.windowTitle()])
                item.setData(0, Qt.UserRole, int(win.window_id))
                self._tree.addTopLevelItem(item)
                if focused is not None and int(win.window_id) == int(focused):
                    item.setSelected(True)
                    self._tree.setCurrentItem(item)
            if focused is None:
                self._tree.clearSelection()
        finally:
            self._syncing = False

    def _on_selection_changed(self) -> None:
        """Row selection IS the wash: selected -> wash that view; none -> clear the wash."""
        if self._syncing:
            return
        items = self._tree.selectedItems()
        if items:
            wid = items[0].data(0, Qt.UserRole)
            if wid is not None:
                self._manager.focus(int(wid))
        else:
            self._manager.clear_focus()

    def _on_activated(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        wid = item.data(0, Qt.UserRole)
        if wid is not None:
            self._manager.focus(int(wid))

    def _close_selected(self) -> None:
        item = self._tree.currentItem()
        if item is not None:
            wid = item.data(0, Qt.UserRole)
            if wid is not None:
                self._manager.close(int(wid))

    def _on_memory(self, frac: float) -> None:
        pct = max(0, min(100, int(round(frac * 100))))
        self._mem_bar.setValue(pct)
        # Warn out loud past 85%: Spencer wanted a memory bar AND a warning, not a silent cap.
        warn = pct >= 85
        self._mem_label.setText("Memory — HIGH, close a view" if warn else "Memory")
        color = "#f85149" if warn else "#3fb950"
        self._mem_bar.setStyleSheet(
            "QProgressBar{background:#161b22;border:1px solid #30363d;border-radius:3px;}"
            f"QProgressBar::chunk{{background:{color};border-radius:3px;}}"
        )


def _process_memory_fraction() -> Optional[float]:
    """This process's RSS as a fraction of total system RAM, or None if it can't be measured.

    Tries psutil (accurate, cross-platform incl. the Windows target); falls back to resource +
    a best-effort total. Returns None rather than a fake number when neither is available — a
    memory bar that invents a value is worse than one that is honestly absent.
    """
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        return float(proc.memory_info().rss) / float(psutil.virtual_memory().total)
    except Exception:                                # noqa: BLE001 - psutil optional
        pass
    try:
        import resource

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports ru_maxrss in bytes, Linux in kilobytes.
        import sys

        rss = float(rss_kb) if sys.platform == "darwin" else float(rss_kb) * 1024.0
        import os

        total = float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        return rss / total if total > 0 else None
    except Exception:                                # noqa: BLE001
        return None
