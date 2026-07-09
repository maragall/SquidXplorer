"""HCS viewer — a post-acquisition, well-plate viewer for Squid acquisitions (IMA-185).

A single professional Qt window, isolated from the Squid acquisition software. This tool runs on
acquisitions that are ALREADY on disk (post-acquisition), so there is no live-follow machinery — it
opens a completed scan and lets you navigate it and apply post-processing operators to it.

    drop a Squid acquisition folder
      -> TOP-LEFT: the PROCESS console — a Cellpose-style stack of operators to run on the plate
               (MIP, Record z-stack; more land here as cards) plus a "to be added" roadmap and an
               "Open CLI" button (a standalone stub window; the visible seam to IMA-186's headless
               CLI). Operators gather any parameters through dialogs — MIP prompts for a destination,
               Record prompts for scope + folder — so the pane is self-contained, no tabs.
      -> BOTTOM-LEFT (<= half the display): a low-resolution PLATE OVERVIEW — one cell per well, laid
               out in true plate row-major (A,B,...,Z,AA,...). Each well is HUE-CODED by its PROCESSING
               status (Hongquan Li's record-zstack-viewer palette): grey = not processed, amber =
               processing, blue = done, red-x = failed. The CURRENT well in view is a red box; the
               cursor's well (as you move around) is a red dot. Wheel-zoom + drag-pan; double-click
               opens a well.
      -> RIGHT (>= half): ndviewer_light EMBEDDED (dark-themed) — the per-FOV 4D detail, full height.
               DOUBLE-CLICK a well and its RAW z-stack (all z, all channels) opens here by pointing
               ndviewer at the acquisition's existing TIFFs (register_image with the raw paths) — zero
               bytes copied, nothing written to disk. The z / t sliders are the real acquisition axes.

The plate is the spatial navigator; ndviewer handles the per-FOV z-stack. "Processing" here means
post-processing: MIP is operator #1, and more operators stack behind the same menu (the moment a
second operator lands this is a general HCS viewer, not just a MIP tool).

Design notes:
- ndviewer_light is the embedded detail viewer (its LightweightViewer QWidget + push API); PyQt5 to
  match its stack. PyQt5 is imported here, never in squidmip/__init__, so the pipeline stays Qt-free.
- Nothing is written to the user's disk: the detail view reads the acquisition's own read-only
  TIFFs. Memory is NOT one-well-at-a-time on the plate side: the MIP run retains one downsampled
  88x88xC float32 tile PER WELL (_OperatorWorker._raw) for the final global-contrast montage, so
  the plate-side footprint is O(n_wells x C) (~190 MB for a 1536wp, C=4), plus a grid-sized RGB
  canvas (~36 MB) and a transient float32 montage buffer at run end. Bounded by the plate format
  (<=1536 wells), not by z/frame size. What IS one-well-at-a-time is project_plate's producer
  (workers x one ~139 MB well) and the detail viewer's LRU-bounded decoded planes.
- Hit-testing / cell fitting are pure functions (unit-testable); widgets run headless under
  QT_QPA_PLATFORM=offscreen.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QLabel, QMainWindow,
    QPlainTextEdit, QPushButton, QScrollArea, QSplitter, QTabBar, QTabWidget,
    QVBoxLayout, QWidget,
)

from squidmip._engine import _default_workers
from squidmip._montage import _area_downsample, _hex_to_rgb01, _window
from squidmip._output import parse_well_id

_CELL = 88                 # per-well px in the low-res overview (1536wp -> ~4224x2816)
_HDR, _COLH = 46, 30       # left / top label margins (px)
_PAD = 16                  # breathing room around the plate
_VIEWER_WORKERS = min(6, _default_workers())   # adapt to the machine, but CAP at 6: the producer's
                           # peak RAM is ~workers x one-well (~139 MB each on a 1536wp), and projection
                           # throughput scales only sublinearly past ~6 threads — so more workers buys
                           # little speed for linearly more memory. 6 balances both, leaves GUI cores.
_BG = "#070a0f"
_GRID, _RED, _MUTED, _ACCENT = QColor(0, 0, 0), QColor("#ff2d2d"), QColor("#8b98ad"), QColor("#58a6ff")

# Processing-status hue coding, adopted from Hongquan Li's record-zstack-viewer plate navigator.
# Deliberately colorblind-safe (blue/amber, never red/green) with a shape cue for failure (the x).
_STATUS = {
    "empty":      QColor("#b7bcc4"),   # not yet processed
    "processing": QColor("#f59e0b"),   # amber — running now
    "done":       QColor("#3b82f6"),   # blue — MIP computed
    "failed":     QColor("#ef4444"),   # red outline + x cross
}
_NDV_DARK = (  # ndviewer defaults to light; theme its Qt chrome dark (bg AND text) to match
    "QWidget{background:#0b0e14;color:#e6edf3;}"
    "QLabel{color:#e6edf3;background:transparent;}"
    "QSlider::groove:horizontal{background:#232b3a;height:4px;border-radius:2px;}"
    "QSlider::handle:horizontal{background:#58a6ff;width:12px;margin:-5px 0;border-radius:6px;}"
    "QPushButton{background:#131824;color:#e6edf3;border:1px solid #232b3a;border-radius:6px;padding:3px 8px;}"
)

# Tab bar for the top-left pane ONLY (its bar sits at the top of that pane, like the plate pane's
# title bar — NOT a global strip across the window). Home tab = the operator list; operators open
# their own UI tab beside it.
_TABS_DARK = (
    "QTabWidget{background:#070a0f;}"
    "QTabWidget::pane{border:1px solid #c9d1d9;background:#070a0f;top:-1px;}"  # thin white outline
    "QTabBar{background:#070a0f;}"                                            # black strip, not white
    "QTabBar::tab{background:#0b0e14;color:#8b98ad;padding:6px 13px;border:1px solid #232b3a;"
    "border-bottom:none;margin-right:2px;font-weight:700;font-size:12px;}"
    "QTabBar::tab:selected{background:#131b2b;color:#e6edf3;}"
)
_CARD_QSS = (   # an operator "card" in the Process pane (Cellpose-style pick-an-operation)
    "QPushButton{background:#0d1420;color:#e6edf3;border:1px solid #232b3a;border-radius:10px;"
    "text-align:left;padding:9px 13px;font-size:13px;}"
    "QPushButton:hover{border-color:#58a6ff;background:#111a2b;}"
    "QPushButton:disabled{color:#57606a;border-color:#1a2130;}"
)
_BTN_QSS = (
    "QPushButton{background:#131824;color:#e6edf3;border:1px solid #232b3a;border-radius:8px;"
    "padding:7px 12px;font-weight:700;} QPushButton:hover{border-color:#58a6ff;}"
    "QPushButton:disabled{color:#57606a;}"
)
_COMBO_QSS = "QComboBox{background:#0d1420;color:#e6edf3;border:1px solid #232b3a;border-radius:6px;padding:5px 8px;}"
@dataclass(frozen=True)
class Operation:
    """One post-processing operator declared in ONE place — the 'operation template'. Adding a feature
    is a single entry here plus one ``_build_<x>_tab`` method; the console builds a card + menu item
    from ``label``/``blurb``, clicking it opens the tab that ``build_tab`` (a PlateWindow method name)
    returns, and every status text (progress, done) derives from ``label``. A flat record, not a
    subclass tree — no new operator ever edits scattered texts or the dispatch."""
    key: str
    label: str
    blurb: str
    build_tab: str        # name of the PlateWindow method that builds this operator's UI tab

# The operator registry. MIP is operator #1; append an Operation + write its `_build_*_tab` and both
# the console cards and the Process-well-plates menu grow automatically.
_OPERATIONS = (
    Operation("mip", "Maximum Intensity Projection",
              "Collapse each well's z-stack to one max-intensity image; save a navigable OME-Zarr plate.",
              "_build_mip_tab"),
    Operation("reference", "Reference plane",
              "Pick each well's sharpest z-plane (Tenengrad autofocus); save a navigable OME-Zarr plate.",
              "_build_reference_tab"),
    Operation("record", "Record z-stack",
              "Save raw z-stacks to disk as multi-page TIFFs — no FIJI round-trip.",
              "_build_record_tab"),
)
_OPERATIONS_BY_KEY = {op.key: op for op in _OPERATIONS}

# Roadmap slots shown (disabled) in the Process console so the direction is visible: hand-off to
# Minerva Author, and a locally-run agent (Nautilus) that builds the operator you ask it for.
_TO_BE_ADDED = [
    ("Open in Minerva Author", "Hand the processed plate to Minerva Author for authoring."),
    ("Ask Nautilus", "Ask Nautilus (local agent) to build the operator you need."),
]


# --- pure geometry (unit-testable, no Qt display) -------------------------------------------

def well_at(rows, cols, by_rc, px: float, py: float, cell_disp: float) -> Optional[dict]:
    """Map a plate pixel (px, py) at *cell_disp* px/well to a cell, or None if out of bounds.

    ``by_rc`` maps (row_index, col_index) -> well_id for acquired wells (else the cell is 'empty').
    Pixels are relative to the plate's top-left (label margins already removed by the caller).
    """
    if px < 0 or py < 0:
        return None
    ci, ri = int(px // cell_disp), int(py // cell_disp)
    if ci >= len(cols) or ri >= len(rows):
        return None
    return {"row_index": ri, "col_index": ci, "row": rows[ri], "col": cols[ci],
            "well_id": by_rc.get((ri, ci))}


def _fit_cell(a: np.ndarray) -> np.ndarray:
    """Resize a 2D plane to EXACTLY (_CELL, _CELL) for the montage tile.

    Area-downsample when larger (the common case: a ~768px tile -> 88); nearest-upscale a tiny
    frame so the tile shape is always (_CELL, _CELL) (guards the <88px-frame crash the review found).
    """
    if a.shape == (_CELL, _CELL):
        return a
    if a.shape[0] >= _CELL and a.shape[1] >= _CELL:
        return _area_downsample(a, _CELL, _CELL)
    yi = (np.arange(_CELL) * a.shape[0]) // _CELL
    xi = (np.arange(_CELL) * a.shape[1]) // _CELL
    return a[yi][:, xi].astype(np.float32)


# The Squid well-plate formats we fit a plate to (well count -> (rows, cols)). An acquisition whose
# format isn't one of these falls back to a present-only grid (see _plate_grid).
_PLATE_DIMS = {4: (2, 2), 6: (2, 3), 12: (3, 4), 24: (4, 6), 96: (8, 12),
               384: (16, 24), 1536: (32, 48)}


def _row_letter(i: int) -> str:
    """0->A, 25->Z, 26->AA, ... (plate row labels)."""
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _plate_grid(wellplate_format) -> Optional[tuple[list, list]]:
    """Full (rows, cols) label grid for a Squid wellplate format, so the plate view shows every
    position evenly spaced (present wells fill; absent stay blank) rather than collapsing gaps.
    Returns None for an unknown/absent format (caller falls back to present-only)."""
    import re
    m = re.search(r"(\d+)", str(wellplate_format or ""))
    dims = _PLATE_DIMS.get(int(m.group(1))) if m else None
    if not dims:
        return None
    nr, nc = dims
    return [_row_letter(i) for i in range(nr)], [str(c) for c in range(1, nc + 1)]


def resolve_plate_root(path) -> tuple[Path, bool]:
    """(path, is_plate): is_plate True when *path* already holds an OME-zarr plate (not a raw
    acquisition); False for a raw acquisition (the case this viewer opens)."""
    p = Path(path)
    if (p / "plate.ome.zarr").is_dir() or (p.name.endswith(".zarr") and (p / "zarr.json").exists()):
        return p, True
    return p, False


class _RunningContrast:
    """Per-channel global contrast that updates as wells stream in (histogram over tiles so far)."""

    def __init__(self, n_ch: int, dmax: float, pct=(1.0, 99.8), bins=512):
        self._bins, self._dmax, self._pct = bins, max(1.0, float(dmax)), pct
        self._hist = [np.zeros(bins, dtype=np.int64) for _ in range(n_ch)]

    def add(self, ch: int, tile: np.ndarray):
        idx = np.clip((tile.ravel() / self._dmax * self._bins).astype(int), 0, self._bins - 1)
        self._hist[ch] += np.bincount(idx, minlength=self._bins)

    def window(self, ch: int) -> tuple[float, float]:
        h = self._hist[ch]
        tot = h.sum()
        if tot == 0:
            return 0.0, self._dmax
        cdf = np.cumsum(h) / tot
        lo = np.searchsorted(cdf, self._pct[0] / 100.0) / self._bins * self._dmax
        hi = np.searchsorted(cdf, self._pct[1] / 100.0) / self._bins * self._dmax
        return float(lo), float(max(hi, lo + 1))


# --- plate overview widget (one cell per well; hue-coded status; fit-to-view) ---------------

class PlateOverview(QWidget):
    """The low-res plate: an RGB canvas of MIP tiles, plus a per-well status hue and a red box."""

    hovered = pyqtSignal(str)              # region id (or "" off-plate), for the window's readout
    wellActivated = pyqtSignal(str, int)   # (well_id, fov_index) double-clicked -> load in ndviewer

    def __init__(self, rows, cols, wells: dict):
        """``wells``: (row_index, col_index) -> well_id for every acquired well (drawn grey until
        processed). Tiles/status arrive as an operator runs."""
        super().__init__()
        self._rows, self._cols = list(rows), list(cols)
        self._nr, self._nc = len(self._rows), len(self._cols)
        self._by_rc: dict[tuple, str] = dict(wells)            # every acquired well (for status + hit-test)
        self._status: dict[tuple, str] = {rc: "empty" for rc in wells}
        self._tiles: set[tuple] = set()                        # cells that have a MIP tile painted
        self._canvas = QImage(self._nc * _CELL, self._nr * _CELL, QImage.Format_RGB888)
        self._canvas.fill(QColor(_BG))
        self._final = None            # crisp global-contrast montage swapped in when the run ends
        self._scaled = None           # cached pixmap of (final|canvas) scaled to the current zoom;
        self._scaled_cd = None        # rebuilt only when zoom (cd) or the source image changes — so
        #                               a hover/pan repaint blits 1:1 instead of re-resampling 12 MP.
        self._cd = float(_CELL)       # displayed px/well (fit baseline, then wheel-zoomed)
        self._ox = self._oy = _PAD    # top-left of the plate within the widget (pan-able)
        self._hover = None
        self._sel = None              # well selected from the ndviewer FOV slider
        self._press = None            # (x, y, ox, oy) at left-press, for drag-to-pan
        self._panning = False
        self._user_view = False       # True once the user wheel-zooms/pans (stop auto-fitting)
        self.setMouseTracking(True)
        self.setMinimumSize(240, 200)

    def _fit(self):
        """Reset the view: the whole plate fits the widget, centered (zoom = 1)."""
        if self._nr == 0 or self._nc == 0:
            return
        w, h = self.width(), self.height()
        self._cd = max(2.0, min((w - _HDR - 2 * _PAD) / self._nc, (h - _COLH - 2 * _PAD) / self._nr))
        self._ox = max(_PAD, (w - _HDR - self._nc * self._cd) / 2)
        self._oy = max(_PAD, (h - _COLH - self._nr * self._cd) / 2)

    def _fit_cd(self) -> float:
        w, h = self.width(), self.height()
        return max(2.0, min((w - _HDR - 2 * _PAD) / self._nc, (h - _COLH - 2 * _PAD) / self._nr))

    # -- data in --
    def add_tile(self, ri: int, ci: int, well_id: str, rgb: np.ndarray):
        if (ri, ci) not in self._by_rc:    # ignore a stale tile from a retired run / foreign cell
            return
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        img = QImage(rgb.data, _CELL, _CELL, 3 * _CELL, QImage.Format_RGB888)
        p = QPainter(self._canvas)
        p.drawImage(ci * _CELL, ri * _CELL, img)
        p.end()
        self._tiles.add((ri, ci))
        self._scaled = None           # source changed -> the scaled cache is stale
        self.update()

    def set_status(self, ri: int, ci: int, state: str):
        if (ri, ci) not in self._status:   # never let a foreign/stale key leak into the status map
            return
        self._status[(ri, ci)] = state
        self.update()

    def set_all_status(self, state: str):
        for rc in self._status:
            self._status[rc] = state
        self.update()

    def set_final(self, img: QImage):
        self._final = img
        self._scaled = None           # source changed -> the scaled cache is stale
        self.update()

    def select(self, ri: int, ci: int):
        """Move the red box to a well (driven by the ndviewer FOV slider)."""
        self._sel = (ri, ci)
        self.update()

    def resizeEvent(self, e):
        self._user_view = False       # a resize re-fits (drop any zoom/pan)
        self._fit()
        self.update()

    # -- mouse: wheel-zoom anchored at cursor + left-drag pan (Hongquan's navigator gestures) --
    def _cell(self, x, y):
        px, py = x - (self._ox + _HDR), y - (self._oy + _COLH)
        return well_at(self._rows, self._cols, self._by_rc, px, py, self._cd)

    def wheelEvent(self, e):
        mx, my = e.x() - (self._ox + _HDR), e.y() - (self._oy + _COLH)    # cursor in plate px
        new_cd = self._cd * (1.0015 ** e.angleDelta().y())
        new_cd = max(self._fit_cd(), min(self._fit_cd() * 40, new_cd))    # never zoom out past fit
        scale = new_cd / self._cd
        self._ox = e.x() - _HDR - mx * scale         # keep the point under the cursor fixed
        self._oy = e.y() - _COLH - my * scale
        self._cd = new_cd
        self._user_view = True
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press = (e.x(), e.y(), self._ox, self._oy)
            self._panning = False

    def mouseMoveEvent(self, e):
        if self._press is not None and (e.buttons() & Qt.LeftButton):
            dx, dy = e.x() - self._press[0], e.y() - self._press[1]
            if abs(dx) + abs(dy) > 3:
                self._panning = True
            if self._panning:
                self._ox, self._oy = self._press[2] + dx, self._press[3] + dy
                self._user_view = True
                self.update()
                return
        c = self._cell(e.x(), e.y())                 # hover (only when not dragging)
        new_hover = (c["row_index"], c["col_index"]) if c else None
        if new_hover == self._hover:                 # still the same cell -> no repaint (kills the
            return                                   # per-pixel repaint storm; only cross-cell moves repaint)
        self._hover = new_hover
        self.hovered.emit((c["well_id"] or (c["row"] + c["col"] + "  ·  empty")) if c else "")
        self.update()

    def mouseReleaseEvent(self, e):
        self._press = None
        self._panning = False

    def leaveEvent(self, e):
        self._hover = None
        self.hovered.emit("")
        self.update()

    def mouseDoubleClickEvent(self, e):
        c = self._cell(e.x(), e.y())
        if c and c["well_id"]:
            self.wellActivated.emit(c["well_id"], 0)   # 1 FOV/well (IMA-183); IMA-187 will pick the FOV

    # -- paint --
    def paintEvent(self, _):
        if not self._user_view:          # auto-fit until the user first zooms/pans
            self._fit()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor(_BG))
        cd, nr, nc = self._cd, self._nr, self._nc
        ax, ay = self._ox + _HDR, self._oy + _COLH   # plate top-left (after label margins)
        W, H = nc * cd, nr * cd
        # Blit the montage from a cached pixmap scaled to the current zoom. The expensive smooth
        # resample runs ONCE per zoom/source-change (not every repaint) — pan/hover just re-blit.
        w, h = max(1, int(W)), max(1, int(H))
        if self._scaled is None or self._scaled_cd != cd or self._scaled.width() != w or self._scaled.height() != h:
            self._scaled = QPixmap.fromImage(self._final or self._canvas).scaled(
                w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self._scaled_cd = cd
        p.drawPixmap(int(ax), int(ay), self._scaled)

        # per-well PROCESSING status = one hue-coded DOT: grey = not processed, amber = processing,
        # NOTHING when done (the MIP image speaks for itself), red x = failed. No filled/hollow variant
        # (that was confusing). The dot is a readable ABSOLUTE size — capped so it doesn't balloon on a
        # small plate's big cells, only shrinking when cells get tiny. Hover dot below shares this size.
        d = min(max(3.0, cd * 0.36), 15.0)
        for (ri, ci), state in self._status.items():
            if state == "done":
                continue                                # done = the MIP image alone (no marker)
            x0, y0 = ax + ci * cd, ay + ri * cd
            ex, ey = int(x0 + (cd - d) / 2), int(y0 + (cd - d) / 2)
            if state == "failed":                       # red x within the dot box
                p.setPen(QPen(_STATUS["failed"], max(1.5, min(cd * 0.09, 3.0))))
                p.drawLine(ex, ey, ex + int(d), ey + int(d))
                p.drawLine(ex + int(d), ey, ex, ey + int(d))
            else:                                       # processing = amber, not-processed = grey
                p.setPen(Qt.NoPen)
                p.setBrush(_STATUS["processing"] if state == "processing" else _STATUS["empty"])
                p.drawEllipse(ex, ey, int(d), int(d))
        p.setBrush(Qt.NoBrush)

        p.setPen(QPen(_GRID, 3))       # black grid lines between wells (room for multi-FOV, IMA-187)
        for c in range(nc + 1):
            p.drawLine(int(ax + c * cd), int(ay), int(ax + c * cd), int(ay + H))
        for r in range(nr + 1):
            p.drawLine(int(ax), int(ay + r * cd), int(ax + W), int(ay + r * cd))
        # Column/row labels THIN OUT as cells shrink so they never overlap (a 48-col 1536wp would
        # otherwise cram "1..48" into a few px). Always draw the hovered row/col so hover still reads.
        p.setFont(QFont("Helvetica Neue", 11, QFont.DemiBold))
        cstep = max(1, int(np.ceil(22.0 / cd)))
        rstep = max(1, int(np.ceil(18.0 / cd)))
        for c in range(nc):
            hov = bool(self._hover and self._hover[1] == c)
            if c % cstep and not hov:
                continue
            p.setPen(_ACCENT if hov else _MUTED)
            p.drawText(int(ax + c * cd), int(self._oy), int(cd), _COLH, Qt.AlignCenter, str(self._cols[c]))
        for r in range(nr):
            hov = bool(self._hover and self._hover[0] == r)
            if r % rstep and not hov:
                continue
            p.setPen(_ACCENT if hov else _MUTED)
            p.drawText(int(self._ox), int(ay + r * cd), _HDR, int(cd), Qt.AlignCenter, str(self._rows[r]))
        if self._sel is not None:          # the CURRENT well in the detail viewer = a red BOX
            ri, ci = self._sel
            p.setPen(QPen(_RED, 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(int(ax + ci * cd), int(ay + ri * cd), int(cd), int(cd))
        if self._hover is not None:        # where the cursor is, moving around the plate = a red DOT
            ri, ci = self._hover           # SAME geometry as the status dots -> overlays them exactly
            x0, y0 = ax + ci * cd, ay + ri * cd
            ex, ey = int(x0 + (cd - d) / 2), int(y0 + (cd - d) / 2)
            p.setPen(Qt.NoPen)
            p.setBrush(_RED)
            p.drawEllipse(ex, ey, int(d), int(d))
        # a fine outer white frame around the whole plate view
        p.setPen(QPen(QColor("#c9d1d9"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)
        p.end()


# --- operator worker: stream a projection over the plate, fill row-major -------------------

class _OperatorWorker(QThread):
    """Runs an operator (MIP) over the plate AND persists it as a navigable multiscale OME-Zarr plate
    (``write_plate``), filling one thumbnail per well as each is written. Projection + pyramid write
    run in write_plate's bounded producer/writer pools; our ``_on_well`` renders the plate tile and
    is called FROM THOSE WRITER THREADS, several at once — so the shared running-contrast, the ``_raw``
    tile store, and the done-counter are guarded by ``_lock`` (the expensive per-channel downsample
    happens OUTSIDE the lock, so downsampling still parallelises). Memory stays O(engine + write
    workers) wells in flight plus one 88x88xC native-dtype tile per well retained in ``_raw`` for the
    final montage. The written ``plate.ome.zarr`` is the durable, re-openable artifact.
    """

    tileReady = pyqtSignal(int, int, str, object)   # (row_index, col_index, well_id, rgb tile)
    progress = pyqtSignal(int, int)                 # (done, total)
    finalReady = pyqtSignal(object)                 # final global-contrast montage (H, W, 3) uint8
    writtenReady = pyqtSignal(str)                  # path of the written plate.ome.zarr
    wellFailed = pyqtSignal(int, int)               # (ri, ci) of a well SKIPPED on a read error
    failed = pyqtSignal(str)                        # whole-run failure (not a per-well skip)
    finished_ok = pyqtSignal()

    def __init__(self, operator: str, reader, meta, fov_index: dict, nr: int, nc: int, out_dir: str):
        super().__init__()
        self._operator = operator
        self._reader, self._meta = reader, meta
        self._fov_index, self._nr, self._nc = fov_index, nr, nc
        self._out_dir = out_dir
        self._channels = [c["name"] for c in meta["channels"]]
        self._colors = np.stack([_hex_to_rgb01(c["display_color"]) for c in meta["channels"]])
        self._dtype = np.dtype(meta["dtype"])
        self._contrast = _RunningContrast(len(self._channels), float(np.iinfo(self._dtype).max))
        self._raw: dict[tuple, np.ndarray] = {}   # (ri,ci) -> (C, _CELL, _CELL) tiles, for the final montage
        self._lock = threading.Lock()             # guards _contrast/_raw/_done (on_well runs on writer threads)
        self._done = 0
        self._stop = threading.Event()            # set by the window to end the run cleanly

    def stop(self):
        """Ask the run to stop; write_plate polls this and abandons after in-flight wells drain."""
        self._stop.set()

    def _on_well(self, region, fov, image):
        """Called per written well (on a write_plate WRITER THREAD): render the plate thumbnail."""
        info = self._fov_index[region]
        ri, ci, well_id = *info["rc"], info["well_id"]
        well = image[0, :, 0]  # (C, Y, X)
        tiles = [_fit_cell(well[c_i]) for c_i in range(len(self._channels))]  # downsample OUTSIDE lock
        raw = np.empty((len(tiles), _CELL, _CELL), self._dtype)              # native dtype (half the RAM)
        rgb = np.zeros((_CELL, _CELL, 3), np.float32)
        with self._lock:                          # shared contrast/raw/counter -> serialize (cheap part)
            for c_i, ds in enumerate(tiles):
                raw[c_i] = ds
                self._contrast.add(c_i, ds)
                lo, hi = self._contrast.window(c_i)
                rgb += _window(ds, lo, hi)[:, :, None] * self._colors[c_i][None, None, :]
            self._raw[(ri, ci)] = raw
            self._done += 1
            done = self._done
        self.tileReady.emit(ri, ci, well_id, (np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        self.progress.emit(done, len(self._meta["regions"]))

    def _on_error(self, region, fov, exc):
        """A well's projection failed (corrupt/missing plane): SKIP it, mark its dot failed, keep the
        run alive. One bad file must not abort a whole-plate run."""
        info = self._fov_index.get(region)
        if info is not None:
            self.wellFailed.emit(*info["rc"])

    def run(self):
        try:
            from squidmip import write_plate      # persist + project in one bounded, streaming pass

            write_plate(self._reader, self._out_dir, n_fovs=1, workers=_VIEWER_WORKERS,
                        projector=self._operator, tiff=False, on_well=self._on_well,
                        stop=self._stop.is_set, on_error=self._on_error)
            if self._stop.is_set():
                return  # window closing / re-opening; drop out cleanly (no final/written emit)
            self.finalReady.emit(self._final_montage())
            self.writtenReady.emit(str(Path(self._out_dir) / "plate.ome.zarr"))
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _final_montage(self) -> np.ndarray:
        wins = [self._contrast.window(ch) for ch in range(len(self._channels))]
        canvas = np.zeros((self._nr * _CELL, self._nc * _CELL, 3), np.float32)
        for (ri, ci), raw in self._raw.items():
            y0, x0 = ri * _CELL, ci * _CELL
            for ch in range(raw.shape[0]):
                lo, hi = wins[ch]
                canvas[y0:y0 + _CELL, x0:x0 + _CELL] += _window(raw[ch], lo, hi)[:, :, None] * self._colors[ch][None, None, :]
        # clip/scale IN PLACE — avoids 3 grid-sized float32 copies (a ~430 MB transient on a 1536wp)
        np.clip(canvas, 0, 1, out=canvas)
        canvas *= 255
        return canvas.astype(np.uint8)


class _PreviewWorker(QThread):
    """Fast RAW preview so the plate shows imagery the moment it opens — before any operator runs
    (the "downsample the plate before opening" step). Reads ONE representative z-plane per channel
    per well (not the whole stack), area-downsamples, and composites by display colour. Cheap
    relative to a full projection; parallel reads; one well's planes at a time. Status stays 'empty'
    (grey frame) — this is a preview, not a processed result. A later operator overwrites each tile.
    """

    tileReady = pyqtSignal(int, int, str, object)   # (row_index, col_index, well_id, rgb tile)

    def __init__(self, reader, meta, fov_index: dict, order: list):
        super().__init__()
        self._reader, self._meta = reader, meta
        self._fov_index, self._order = fov_index, order
        self._channels = [c["name"] for c in meta["channels"]]
        self._colors = np.stack([_hex_to_rgb01(c["display_color"]) for c in meta["channels"]])
        self._contrast = _RunningContrast(len(self._channels), float(np.iinfo(np.dtype(meta["dtype"])).max))
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            from concurrent.futures import ThreadPoolExecutor
            zs = self._meta["z_levels"]
            z_mid = zs[len(zs) // 2]      # a mid-stack plane is a fair single-plane preview

            def load(region):
                fov = self._meta["fovs_per_region"][region][0]
                return region, [_fit_cell(self._reader.read(region, fov, ch, z_mid).astype(np.float32))
                                for ch in self._channels]

            with ThreadPoolExecutor(max_workers=_VIEWER_WORKERS) as ex:
                for region, tiles in ex.map(load, self._order):   # row-major order preserved
                    if self._stop.is_set():
                        return
                    rgb = np.zeros((_CELL, _CELL, 3), np.float32)
                    for c_i, ds in enumerate(tiles):
                        self._contrast.add(c_i, ds)
                        lo, hi = self._contrast.window(c_i)
                        rgb += _window(ds, lo, hi)[:, :, None] * self._colors[c_i][None, None, :]
                    ri, ci = self._fov_index[region]["rc"]
                    self.tileReady.emit(ri, ci, region, (np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        except Exception:
            pass   # preview is best-effort; the operator run is the authoritative result


# --- main window: plate overview | embedded ndviewer ----------------------------------------

class PlateWindow(QMainWindow):
    def __init__(self, initial_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("HCS viewer")
        self.resize(1600, 950)
        self._worker = None           # the operator (MIP) run
        self._preview = None          # the raw preview fill on open
        self._retired = []            # workers asked to stop; kept alive until they actually finish
        self._overview = None
        self._reader = None
        self._meta = None
        self._fov_index = {}
        self._pushed = set()          # wells whose raw z-stack is already registered in the detail viewer
        self._final_arr = None        # keep the final montage array alive for its QImage

        # Process-well-plates menu (operators). MIP is #1; disabled until an acquisition is open.
        self._op_actions = {}
        proc_menu = self.menuBar().addMenu("&Process well-plates")
        for op in _OPERATIONS:
            act = QAction("&" + op.label, self)
            act.setEnabled(False)
            act.triggered.connect(lambda _=False, k=op.key: self._activate_operator(k))
            proc_menu.addAction(act)
            self._op_actions[op.key] = act

        self._acq_name = ""           # acquisition folder name, shown as the Process-pane title
        self._current_well = None     # the well currently shown in the detail viewer (for Record)
        self._recording = False       # guards _run_record against re-entry during its processEvents
        self._acq_path = None         # the opened acquisition dir (persist writes next to it)
        self._processed_plate = None  # path of the written plate.ome.zarr once an operator persists it
        self._plate_mode = "raw"      # what the plate view is showing — shown in the plate-pane title
        self._op_tabs = {}            # key -> operator-UI widget currently open as a tab in _left_tabs

        # THREE-PANE layout. Tabs live ONLY inside the top-left pane (their bar sits at the pane's top,
        # like the plate pane's title bar) — never a global strip across the window:
        #   top-left  = the PROCESS console, a QTabWidget: a "Process wells" home tab (operator list),
        #               and one tab per operator you open (MIP -> where-to-save UI; Record -> recorder
        #               UI). The right pane stays a plain singleton, so operator UIs live here.
        #   bottom-left = the HCS PLATE view (<= half the display wide); its title bar names the plate.
        #   right     = the ndviewer_light array viewer, full height (a singleton — no tabs).

        # top-left: the process console (build the home tab first — it owns self._readout, which
        # _make_detail_viewer writes to if ndviewer is unavailable).
        self._left_tabs = QTabWidget()
        self._left_tabs.setStyleSheet(_TABS_DARK)
        self._left_tabs.setTabsClosable(True)
        self._left_tabs.tabCloseRequested.connect(self._close_op_tab)
        self._left_tabs.addTab(self._build_process_pane(), "Process wells")
        self._left_tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)  # home tab isn't closable

        # right: the ndviewer array viewer directly (a singleton — no tab)
        self._detail = self._make_detail_viewer()
        if self._detail is not None:   # connect the FOV slider -> red box ONCE (not per ingest)
            slider = getattr(self._detail, "_fov_slider", None)
            if slider is not None:
                slider.valueChanged.connect(self._on_fov_slider)
            self._right_widget = self._detail
        else:
            ph = QLabel("ndviewer_light unavailable — pip install ndviewer_light")
            ph.setAlignment(Qt.AlignCenter)
            ph.setStyleSheet("color:#8b98ad;")
            self._right_widget = ph

        # bottom-left: plate view (drop target until an acquisition opens). Its FIXED title bar names
        # the wellplate we're on (the acquisition) — the plate's identity lives with the plate.
        self._plate_title = QLabel("well plate")   # plate name; shows the hovered well (large) on hover
        self._plate_title.setStyleSheet(
            "color:#e6edf3;font-size:17px;font-weight:800;padding:9px 14px;"
            "background:#0b0e14;border-bottom:1px solid #232b3a;")
        self._drop = QLabel("Drop a Squid acquisition folder here\n\n"
                            "then pick an operator in  Process wells ▸")
        self._drop.setAlignment(Qt.AlignCenter)
        self._drop.setStyleSheet("color:#8b98ad;font-size:16px;border:2px dashed #232b3a;border-radius:12px;margin:24px;")
        plate_host = QWidget()
        plate_host.setStyleSheet(f"background:{_BG};")
        self._left_l = QVBoxLayout(plate_host)
        self._left_l.setContentsMargins(0, 0, 0, 0)
        self._left_l.setSpacing(0)
        self._left_l.addWidget(self._plate_title)
        self._left_l.addWidget(self._drop, 1)    # the plate overview replaces this on ingest

        left_col = QSplitter(Qt.Vertical)
        left_col.setStyleSheet("QSplitter::handle{background:#232b3a;height:1px;}")
        left_col.setChildrenCollapsible(False)
        left_col.addWidget(self._left_tabs)
        left_col.addWidget(plate_host)
        left_col.setStretchFactor(0, 0)
        left_col.setStretchFactor(1, 1)
        left_col.setSizes([340, 610])

        # the right (array viewer) pane gets a thin white outline via a 1px-margin frame
        right_frame = QFrame()
        right_frame.setStyleSheet("QFrame{background:#6e7681;}")   # shows through the 1px margin = outline
        rfl = QVBoxLayout(right_frame)
        rfl.setContentsMargins(1, 1, 1, 1)
        rfl.setSpacing(0)
        rfl.addWidget(self._right_widget)

        outer = QSplitter(Qt.Horizontal)
        outer.setStyleSheet("QSplitter::handle{background:#232b3a;width:1px;}")
        outer.setChildrenCollapsible(False)
        outer.addWidget(left_col)
        outer.addWidget(right_frame)
        outer.setSizes([760, 760])                 # divider fixed at the middle of the window
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 1)
        self._split = outer
        self.setCentralWidget(outer)

        self.setAcceptDrops(True)
        if initial_path:
            self.ingest(initial_path)

    # -- top-left process console: pick an operator (params via dialogs), open the CLI stub ---------
    def _build_process_pane(self) -> QWidget:
        """The 'Process wells' console: a compact one-line status, a scrollable stack of operator cards
        (MIP, Record z-stack) plus a 'to be added' roadmap, and an 'Open CLI' button. Operators gather
        any parameters through dialogs, so this pane is self-contained — no tabs."""
        pane = QWidget()
        pane.setStyleSheet(f"background:{_BG};")
        v = QVBoxLayout(pane)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(7)

        # a single small subtitle (also the live status line — tests read this)
        self._readout = QLabel("Drop a Squid acquisition, then pick an operator to run on the plate.")
        self._readout.setStyleSheet("color:#8b98ad;font-size:12px;")
        self._readout.setWordWrap(True)
        v.addWidget(self._readout)

        def _section(text):
            lab = QLabel(text)
            lab.setStyleSheet("color:#57606a;font-size:10px;font-weight:800;letter-spacing:1.5px;padding-top:8px;")
            return lab

        stack = QWidget()
        sv = QVBoxLayout(stack)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(7)
        sv.addWidget(_section("OPERATORS"))
        self._op_cards = {}
        for op in _OPERATIONS:
            card = QPushButton(f"{op.label}\n{op.blurb}")
            card.setEnabled(False)                         # enabled once an acquisition loads
            card.setCursor(Qt.PointingHandCursor)
            card.setStyleSheet(_CARD_QSS)
            card.setMinimumHeight(54)
            card.clicked.connect(lambda _=False, k=op.key: self._activate_operator(k))
            sv.addWidget(card)
            self._op_cards[op.key] = card
        sv.addWidget(_section("TO BE ADDED"))
        for label, blurb in _TO_BE_ADDED:
            soon = QPushButton(f"{label}\n{blurb}")
            soon.setEnabled(False)
            soon.setStyleSheet(_CARD_QSS + "QPushButton:disabled{color:#57606a;border-style:dashed;}")
            soon.setMinimumHeight(46)
            sv.addWidget(soon)
        sv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        scroll.setWidget(stack)
        v.addWidget(scroll, 1)

        cli_btn = QPushButton("⌨  Open CLI")               # opens a CLI tab within this pane
        cli_btn.setStyleSheet(_BTN_QSS)
        cli_btn.clicked.connect(lambda: self._open_op_tab("cli", "CLI", self._build_cli_tab))
        v.addWidget(cli_btn)
        return pane

    # -- operator UIs live as tabs INSIDE the top-left pane (home tab + one per opened operator) ----
    def _open_op_tab(self, key: str, title: str, builder):
        """Open (or focus) an operator's UI as a tab beside 'Process wells'. Built lazily, once."""
        w = self._op_tabs.get(key)
        if w is None:
            w = builder()
            self._op_tabs[key] = w
            self._left_tabs.addTab(w, title)
        self._left_tabs.setCurrentWidget(w)

    def _close_op_tab(self, index: int):
        if index == 0:                                     # 'Process wells' home tab — never closable
            return
        w = self._left_tabs.widget(index)
        self._left_tabs.removeTab(index)
        for k, v in list(self._op_tabs.items()):
            if v is w:
                del self._op_tabs[k]
        w.deleteLater()

    def _activate_operator(self, key: str):
        """Operator card / menu clicked: open the operator's UI tab. Fully generic — driven by the
        Operation template, so a new operator needs no edit here (just a registry entry + build_tab)."""
        if self._reader is None or self._overview is None:
            self._readout.setText("open an acquisition first")
            return
        op = _OPERATIONS_BY_KEY.get(key)
        if op is not None:
            self._open_op_tab(op.key, op.label, getattr(self, op.build_tab))

    def _op_tab_shell(self, title: str, blurb: str) -> tuple:
        """A standard operator-UI tab body: title + blurb, returns (widget, vbox) to fill."""
        w = QWidget()
        w.setStyleSheet(f"background:{_BG};color:#e6edf3;")
        v = QVBoxLayout(w)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)
        t = QLabel(title); t.setStyleSheet("font-size:16px;font-weight:800;")
        v.addWidget(t)
        b = QLabel(blurb); b.setWordWrap(True); b.setStyleSheet("color:#8b98ad;font-size:12px;")
        v.addWidget(b)
        return w, v

    def _build_mip_tab(self) -> QWidget:
        return self._build_run_tab(_OPERATIONS_BY_KEY["mip"])

    def _build_reference_tab(self) -> QWidget:
        return self._build_run_tab(_OPERATIONS_BY_KEY["reference"])

    def _build_run_tab(self, op) -> QWidget:
        """Generic projector-operator tab (MIP, Reference plane, …): pick a destination, run over the
        whole plate → a navigable OME-Zarr plate. ONE builder for every z-reduction operator — a new
        one needs no new tab code. Per-tab state lives in a closure (no per-operator instance attrs)."""
        w, v = self._op_tab_shell(op.label, op.blurb + " Pick a destination with room — output can be large.")
        state = {"dir": None}
        dir_lbl = QLabel("(no folder chosen)"); dir_lbl.setWordWrap(True)
        dir_lbl.setStyleSheet("color:#8b98ad;font-size:12px;")
        run = QPushButton("▶  Run on the whole plate"); run.setStyleSheet(_BTN_QSS); run.setEnabled(False)

        def pick():
            d = QFileDialog.getExistingDirectory(self, f"Save {op.label} plate to folder")
            if not d:
                return
            state["dir"] = d
            ok, est_gb, _ = self._check_disk(Path(d) / f"{self._acq_name}.hcs")
            dir_lbl.setText(f"{d}\n~{est_gb:.0f} GB needed" + ("" if ok else "  ⚠ not enough free space"))
            run.setEnabled(True)

        pick_btn = QPushButton("Choose output folder…"); pick_btn.setStyleSheet(_BTN_QSS)
        pick_btn.clicked.connect(pick)
        run.clicked.connect(lambda: self.run_operator(op.key, out_parent=state["dir"]))
        v.addWidget(pick_btn); v.addWidget(dir_lbl); v.addWidget(run); v.addStretch(1)
        return w

    def _build_record_tab(self) -> QWidget:
        axis = "time-lapse (T)" if (self._meta and self._meta.get("n_t", 1) > 1) else "focus sweep (Z)"
        w, v = self._op_tab_shell(
            "Record video (.mp4)",
            f"Assemble each well's {axis} into an .mp4 — post-acquisition, no FIJI. One movie per well.")
        v.addWidget(QLabel("Scope"))
        self._rec_scope = QComboBox(); self._rec_scope.setStyleSheet(_COMBO_QSS)
        self._rec_scope.addItems(["Current well only", "Every well on the plate"])
        v.addWidget(self._rec_scope)
        v.addWidget(QLabel("Playback fps"))
        self._rec_fps = QComboBox(); self._rec_fps.setStyleSheet(_COMBO_QSS)
        self._rec_fps.addItems(["2", "5", "10", "15"]); self._rec_fps.setCurrentText("5")
        v.addWidget(self._rec_fps)
        self._rec_z = QCheckBox("Record Z focus sweep (instead of time)")  # opt-in; default T
        self._rec_z.setStyleSheet("color:#e6edf3;")
        v.addWidget(self._rec_z)
        self._rec_dir = None
        pick = QPushButton("Choose output folder…"); pick.setStyleSheet(_BTN_QSS)
        pick.clicked.connect(self._pick_rec_dir)
        v.addWidget(pick)
        self._rec_dir_lbl = QLabel("(no folder chosen)"); self._rec_dir_lbl.setWordWrap(True)
        self._rec_dir_lbl.setStyleSheet("color:#8b98ad;font-size:12px;")
        v.addWidget(self._rec_dir_lbl)
        self._rec_run = QPushButton("⏺  Record .mp4"); self._rec_run.setStyleSheet(_BTN_QSS)
        self._rec_run.setEnabled(False); self._rec_run.clicked.connect(self._record_run)
        v.addWidget(self._rec_run)
        v.addStretch(1)
        return w

    def _pick_rec_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Save .mp4(s) to folder")
        if d:
            self._rec_dir = d; self._rec_dir_lbl.setText(d); self._rec_run.setEnabled(True)

    def _record_run(self):
        if self._rec_scope.currentIndex() == 1:
            wells = list(self._fov_index)
        elif self._current_well is not None:
            wells = [self._current_well]
        else:
            self._readout.setText("double-click a well first, or choose 'Every well on the plate'")
            return
        self._run_record(wells, self._rec_dir, int(self._rec_fps.currentText()),
                         record_z=self._rec_z.isChecked())

    def _build_cli_tab(self) -> QWidget:
        """The headless-CLI stub as a tab (IMA-186 wires it; this shows the equivalent commands)."""
        term = QPlainTextEdit(); term.setReadOnly(True)
        term.setStyleSheet("QPlainTextEdit{background:#05070b;color:#8bffd0;border:none;"
                           "font-family:'SF Mono','Menlo',monospace;font-size:12px;padding:12px;}")
        name = self._acq_name or "<acquisition>"
        term.setPlainText(
            "HCS viewer — CLI  (preview)\n"
            "──────────────────────────────\n"
            "The same operators, run headlessly (IMA-186 — the `squidmip` command):\n\n"
            f"  $ squidmip \"{name}\"                        # MIP every well -> navigable OME-zarr\n"
            f"  $ squidmip \"{name}\" --projector reference  # sharpest-plane per well\n"
            f"  $ squidmip \"{name}\" --workers 8 --tiff     # tune + also export TIFFs\n\n"
            "# this tab is a preview; run the command in a terminal.\n")
        return term

    def _run_record(self, wells, out, fps, record_z=False):
        """Record each well's video → ``<out>/<well>.mp4`` (time-lapse if there's a T series, else a
        Z focus-sweep), composited by display colour at *fps*. Post-acquisition: reads existing frames.

        Snapshot the inputs before the loop: it pumps processEvents (to update status), which can
        deliver a re-drop -> ingest() that nulls self._reader/_meta mid-loop. Iterating over locals
        means a concurrent open can't corrupt an in-flight record; _recording blocks re-entry."""
        if self._reader is None or self._recording:
            return
        from squidmip._video import default_axis, well_movie_frames, write_mp4
        reader, meta = self._reader, self._meta       # `out`, `wells`, `fps` are local args
        axis = default_axis(meta, record_z)
        self._recording = True
        try:
            for i, well in enumerate(wells, 1):
                fov = meta["fovs_per_region"][well][0]
                write_mp4(well_movie_frames(reader, well, fov, axis=axis),
                          Path(out) / f"{well}.mp4", fps)
                self._readout.setText(f"recording {i}/{len(wells)} · {well}.mp4 …")
                QApplication.processEvents()
            self._readout.setText(f"✓ recorded {len(wells)} .mp4(s) → {out}")
        except Exception as e:
            self._readout.setText(f"record failed: {e}")
        finally:
            self._recording = False

    def _enable_operators(self, flag: bool):
        for a in self._op_actions.values():
            a.setEnabled(flag)
        for c in getattr(self, "_op_cards", {}).values():
            c.setEnabled(flag)

    def _make_detail_viewer(self):
        try:
            from ndviewer_light.core import LightweightViewer
            v = LightweightViewer(None)   # empty -> push mode (we register raw z-planes on demand)
            v.setStyleSheet(_NDV_DARK)    # ndviewer defaults to light; match the plate view
            # The PLATE is the spatial navigator (double-click a well) — so hide ndviewer's own FOV
            # slider and its "n per well" FOV-subset control. Both navigate FOVs, which here is exactly
            # what the plate does; keeping them duplicates navigation and clutters the z-stack detail.
            # (go_to_well_fov still drives the hidden FOV slider programmatically, keeping z/channel and
            # the red box in sync.) Guarded by getattr so an older ndviewer without them still loads.
            for attr in ("_fov_slider_container", "_subset_container"):
                w = getattr(v, attr, None)
                if w is not None:
                    w.hide()
            return v
        except Exception as e:
            self._readout.setText(f"ndviewer_light unavailable: {e}")
            return None

    # -- drag & drop --
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.ingest(urls[0].toLocalFile())

    # -- open an acquisition (no processing yet — that's the Process menu) --
    def ingest(self, path: str):
        from squidmip import open_reader
        from squidmip._output import plate_metadata

        p, is_plate = resolve_plate_root(path)
        if is_plate:
            self._readout.setText("this is already a written plate — drop a raw Squid acquisition")
            return
        # stop any in-flight run/preview and clear prior state before opening a new acquisition
        self._stop_worker()
        self._stop_preview()
        self._reader = self._meta = None
        self._fov_index = {}
        self._pushed = set()
        self._current_well = None
        self._final_arr = None
        self._enable_operators(False)
        if self._overview is not None:
            self._overview.setParent(None)
            self._overview.deleteLater()
            self._overview = None
        self._readout.setText("scanning acquisition …")
        QApplication.processEvents()
        try:
            reader = open_reader(str(p))
            meta = reader.metadata
        except Exception as e:   # not a Squid acquisition / unreadable -> report, don't crash the app
            self._readout.setText(f"not a readable Squid acquisition: {e}")
            self._drop.show()
            return
        self._reader, self._meta = reader, meta
        self._acq_name = Path(p).name
        self._acq_path = Path(p)
        self._processed_plate = None

        # Order wells in TRUE plate row-major (A,B,...,Z,AA,...). NOT lexicographic ("AA" < "B").
        # This parses region ids as well ids — guard it: a readable acquisition whose regions are
        # NOT well-plate ids (glass slide, manual/coordinate names, "R2C3", "0") must report, not
        # crash. parse_well_id / plate_metadata raise ValueError on those. (contract: never crash.)
        try:
            plate = plate_metadata(meta["regions"], field_count=1)["plate"]
            present_rows = [r["name"] for r in plate["rows"]]
            present_cols = [c["name"] for c in plate["columns"]]
        except (ValueError, KeyError) as e:
            self._reader = self._meta = None
            self._readout.setText(
                f"not a well-plate acquisition — regions like {list(meta['regions'])[:3]} aren't "
                f"well ids (e.g. B2); the HCS viewer needs a well plate. ({type(e).__name__})")
            self._drop.show()
            return
        # Prefer the FULL plate-format grid (every position, evenly spaced — no collapsed gaps).
        # Fall back to present-only when the format is unknown or the wells don't fit it.
        grid = _plate_grid(meta.get("wellplate_format"))
        if grid and set(present_rows) <= set(grid[0]) and set(present_cols) <= set(grid[1]):
            rows, cols = grid
        else:
            rows, cols = present_rows, present_cols
        row_of = {r: i for i, r in enumerate(rows)}   # plate order: A=0,B=1,...,Z=25,AA=26,...
        col_of = {c: i for i, c in enumerate(cols)}

        def _rc(region):
            rr, cc = parse_well_id(region)
            return (row_of[rr], col_of[cc])
        order = sorted(meta["regions"], key=_rc)
        wells = {}
        for idx, region in enumerate(order):
            rc = _rc(region)
            self._fov_index[region] = {"idx": idx, "well_id": region, "rc": rc}
            wells[rc] = region

        self._overview = PlateOverview(rows, cols, wells)
        self._overview.hovered.connect(self._on_hover)
        self._overview.wellActivated.connect(self.activate_well)
        self._plate_mode = "raw"                     # a freshly-opened plate shows raw previews
        self._plate_title.setText(f"{self._acq_name}   ·   raw")   # bottom-left plate-pane title
        self._drop.hide()
        self._left_l.addWidget(self._overview, 1)   # fills the pane and self-fits — no scrollbars

        if self._detail is not None:
            # push mode over the RAW acquisition: full z (real z-stack) and full frame; the detail
            # reads the acquisition's own TIFFs (register_image) — nothing copied.
            h, w = meta["frame_shape"]
            self._detail.start_acquisition([c["name"] for c in meta["channels"]], meta["n_z"],
                                           h, w, [f"{r}:0" for r in order])

        self._enable_operators(True)

        # fast RAW preview: fill the plate with downsampled thumbnails immediately (grey dots),
        # in the SAME row-major order the operator will later process them in.
        self._preview = _PreviewWorker(reader, meta, self._fov_index, order)
        self._preview.tileReady.connect(self._on_preview_tile)
        self._preview.start()
        if order:                     # populate the detail pane right away so ndviewer isn't blank
            self.activate_well(order[0], 0)
        # top-left = LIVE STATUS (what's happening / what's shown); the plate name is the pane title
        # Multi-FOV policy (IMA-191): current scope is one FOV per well (a well = a condition, exactly
        # Nick's "n=1 per condition"). When wells hold >1 FOV we SAMPLE the first and say so — honest
        # interim until high-throughput stitching (stitch -> MIP -> one composite/well) lands.
        multi = sum(1 for r in order if len(meta["fovs_per_region"][r]) > 1)
        note = (f" · {multi} multi-FOV well(s): sampling 1 FOV/well (stitching TBD)" if multi else "")
        self._readout.setText(f"live · {len(self._fov_index)} wells · double-click to open{note}")

    # -- run a post-processing operator over the whole plate (persists a navigable OME-Zarr plate) --
    def run_operator(self, key: str, out_parent: Optional[str] = None):
        if self._reader is None or self._overview is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self._readout.setText("already processing — let the current run finish first")
            return
        label = _OPERATIONS_BY_KEY[key].label
        # Ask WHERE to persist: the output can be hundreds of GB, so let the user aim it at a disk
        # with room (e.g. an external drive) rather than silently filling the acquisition's disk.
        # Tests pass out_parent to skip the modal dialog.
        if out_parent is None:
            out_parent = QFileDialog.getExistingDirectory(self, f"Save {label} plate to folder")
            if not out_parent:
                return
        out_dir = Path(out_parent) / f"{self._acq_name}.hcs"
        ok, est_gb, msg = self._check_disk(out_dir)
        if not ok:
            self._readout.setText(msg)
            return
        self._stop_preview()                                 # the operator supersedes the raw preview
        self._overview.set_all_status("processing")          # amber across the plate
        self._plate_mode = label                             # plate now shows this operator's result
        self._plate_title.setText(f"{self._acq_name}   ·   {label}")
        self._worker = _OperatorWorker(key, self._reader, self._meta, self._fov_index,
                                       self._overview._nr, self._overview._nc, str(out_dir))
        self._worker.tileReady.connect(self._on_tile)
        self._worker.progress.connect(
            lambda d, t: self._readout.setText(f"● {label} · {d}/{t} wells → {out_dir.name} (~{est_gb:.0f} GB)"))
        self._worker.finalReady.connect(self._set_final)
        self._worker.writtenReady.connect(self._on_written)
        self._worker.wellFailed.connect(                     # a skipped well -> red x, run continues
            lambda ri, ci: self._overview.set_status(ri, ci, "failed") if self._overview else None)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished_ok.connect(lambda: self._readout.setText(
            f"✓ {label} · {len(self._fov_index)} wells → {out_dir.name}  (re-openable OME-Zarr)"))
        self._worker.start()

    def _check_disk(self, out_dir) -> tuple[bool, float, str]:
        """Estimate the persisted plate size and refuse if it won't fit (with headroom). Returns
        (ok, estimate_GB, message). Estimate = per-well MIP (C·Y·X·itemsize) × 1.34 (the exact 4/3
        geometric sum of the 2× pyramid tail), UNCOMPRESSED. We do NOT discount for zstd: real
        fluorescence compresses unpredictably (often <1.2×), so assuming compression would under-
        estimate and let a multi-hour run fill the disk mid-write — the exact failure this guards. An
        over-estimate only ever asks for a roomier disk, which is the safe way to be wrong."""
        import shutil
        m = self._meta
        ny, nx = m["frame_shape"]
        est = int(len(self._fov_index) * len(m["channels"]) * ny * nx
                  * np.dtype(m["dtype"]).itemsize * 1.34)
        gb = 1024 ** 3
        try:
            free = shutil.disk_usage(Path(out_dir).parent).free
        except OSError:
            return True, est / gb, ""      # can't stat the disk — don't block
        if est > free * 0.9:
            return False, est / gb, (f"MIP would persist ~{est/gb:.0f} GB to {Path(out_dir).parent} "
                                     f"but only {free/gb:.0f} GB free — free space or pick another disk.")
        return True, est / gb, ""

    def _on_written(self, plate_path: str):
        """The operator finished persisting: remember the written plate (re-openable artifact)."""
        self._processed_plate = plate_path

    def _on_preview_tile(self, ri, ci, well_id, rgb):
        if self._overview is not None:                       # raw thumbnail only; status stays grey
            self._overview.add_tile(ri, ci, well_id, rgb)

    def _on_tile(self, ri, ci, well_id, rgb):
        if self._overview is None:
            return
        self._overview.add_tile(ri, ci, well_id, rgb)
        self._overview.set_status(ri, ci, "done")           # blue

    def _on_failed(self, msg):
        if self._overview is not None:
            for rc, state in list(self._overview._status.items()):
                if state == "processing":
                    self._overview.set_status(*rc, "failed")  # red x on wells that didn't finish
        self._readout.setText(f"failed: {msg}")

    def _set_final(self, rgb):
        if self._overview is None:
            return
        self._final_arr = np.ascontiguousarray(rgb)
        h, w, _ = self._final_arr.shape
        self._overview.set_final(QImage(self._final_arr.data, w, h, 3 * w, QImage.Format_RGB888))

    # -- navigation links --
    def _on_hover(self, text: str):
        # BOTTOM-LEFT plate title bar: "<acq>  ·  <mode>" (mode = raw / the operator that processed it),
        # plus the hovered well when the cursor is over the plate.
        base = f"{self._acq_name or 'well plate'}   ·   {self._plate_mode}"
        self._plate_title.setText(f"{base}   ·   {text}" if text else base)

    def activate_well(self, well_id: str, fov_index: int):
        """Double-click -> open the well's RAW z-stack in the embedded ndviewer. Registers the raw
        TIFF paths lazily (once per well): the detail is the true z-stack, zero bytes copied."""
        if self._detail is None or self._reader is None or well_id not in self._fov_index:
            return
        self._current_well = well_id
        if self._overview is not None:                 # current well at view = the red BOX
            self._overview.select(*self._fov_index[well_id]["rc"])
        idx = self._fov_index[well_id]["idx"]
        if well_id not in self._pushed:
            fov = self._meta["fovs_per_region"][well_id][0]
            for z_i, z in enumerate(self._meta["z_levels"]):
                for ch in (c["name"] for c in self._meta["channels"]):
                    try:
                        path = self._reader.plane_path(well_id, fov, ch, z)
                        self._detail.register_image(0, idx, z_i, ch, str(path))
                    except (KeyError, IndexError, OSError, RuntimeError):
                        continue   # a genuinely-missing plane / closed viewer shouldn't block the rest
            self._pushed.add(well_id)
        row, col = parse_well_id(well_id)
        try:
            self._detail.go_to_well_fov(f"{row}{col}", fov_index)
        except Exception:
            pass

    def _on_fov_slider(self, flat_idx: int):
        """ndviewer FOV slider moved -> move the red box on the plate to that well."""
        if self._detail is None or self._overview is None:
            return
        labels = getattr(self._detail, "_fov_labels", None)
        if not labels or not (0 <= flat_idx < len(labels)):
            return
        info = self._fov_index.get(labels[flat_idx].split(":")[0])
        if info:
            self._overview.select(*info["rc"])

    def _retire(self, w):
        """Retire a worker thread WITHOUT ever destroying a running QThread (that aborts the app).
        Disconnect its signals first so a tile already queued before the stop can't paint onto a
        freshly-opened plate (the cross-plate corruption the review found); then keep a reference
        alive until it actually finishes (stop() returns after the current item, which is bounded)."""
        if w is None:
            return
        for name in ("tileReady", "progress", "finalReady", "writtenReady", "wellFailed", "failed", "finished_ok"):
            sig = getattr(w, name, None)
            if sig is not None:
                try:
                    sig.disconnect()
                except TypeError:
                    pass             # nothing connected — fine
        if w.isRunning():
            w.stop()
            self._retired.append(w)
            w.finished.connect(lambda: self._retired.remove(w) if w in self._retired else None)

    def _stop_worker(self):
        self._retire(self._worker)
        self._worker = None

    def _stop_preview(self):
        self._retire(self._preview)
        self._preview = None

    def closeEvent(self, e):
        self._stop_worker()          # stop the run cleanly; nothing on disk to clean up (no cache)
        self._stop_preview()
        for w in list(self._retired):
            w.wait()                 # join before exit — never leave a QThread running at teardown
        super().closeEvent(e)


def main(dataset_path: str = None):
    path = dataset_path or (sys.argv[1] if len(sys.argv) > 1 else None)
    app = QApplication.instance() or QApplication(sys.argv)
    win = PlateWindow(path)
    win.show()
    if not app.property("_squidmip_test"):
        sys.exit(app.exec_())
    return win


if __name__ == "__main__":
    main()
