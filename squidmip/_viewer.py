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
               opens a well; PRESS-AND-HOLD raises a loupe (IMA-208) that overlays the well's real
               pixels — read from the acquisition's TIFFs, or from the written pyramid once an
               operator has persisted one — magnified relative to the current plate zoom and capped
               at native resolution, with a µm scale bar when the pixel size is known.
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
  TIFFs. Memory is NOT one-well-at-a-time on the plate side: PlateOverview retains the whole plate
  with its CHANNEL AXIS intact — one (C, nr*88, nc*88) NATIVE-DTYPE store per displayed layer — so
  a channel toggle or a contrast drag recomposites from pixels already in RAM instead of re-reading
  or re-projecting anything. That is ~95 MB for a 1536wp at C=4 uint16 (native dtype, so half what
  float32 would cost), and it MULTIPLIES per layer: raw + one operator layer is ~190 MB. Allocated
  lazily, only for a layer that actually receives tiles. On top sits a grid-sized RGB canvas (~36 MB)
  per layer and one transient float32 buffer during a full-resolution recomposite. Bounded by the
  plate format (<=1536 wells), not by z/frame size. What IS one-well-at-a-time is project_plate's
  producer (workers x one ~139 MB well) and the detail viewer's LRU-bounded decoded planes.
- Hit-testing / cell fitting are pure functions (unit-testable); widgets run headless under
  QT_QPA_PLATFORM=offscreen.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import (
    Qt, QRectF, QThread, QTimer, pyqtSignal,
)
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPalette, QPen, QPixmap, QRegion
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMenu, QPlainTextEdit, QPushButton, QScrollArea, QSlider, QSpinBox,
    QSplitter, QStackedWidget, QStyleFactory, QTabBar, QVBoxLayout, QWidget,
)

#: The main window's logger. The log panel taps the stdlib ROOT logger, so anything logged here
#: appears in the bottom-right panel for free — the reason a failure the user triggers (a spot
#: detection that raised, a region that would not fuse) MUST go through this and not only into an
#: in-widget banner: a banner the user has already clicked past leaves no trace, and "the logger
#: didn't show it" is the exact gap this closes.
log = logging.getLogger("squidmip.viewer")

from squidmip import _explore, _qtstyle
from squidmip._engine import _default_workers, available_projectors
from squidmip._layers import OperationStack
from squidmip._minerva import MINERVA_HOME_ENV as _MINERVA_HOME_ENV
from squidmip._montage import _area_downsample, _hex_to_rgb01, composite
from squidmip._output import parse_well_id
from squidmip._activity import ActivityLog
from squidmip._logpane import LogBus
from squidmip._logpanel import LogPanel
from squidmip._plate import PlateBuildError, build_plate, display_well_id
from squidmip._plate_shape import PlateShapeError
from squidmip._qt_tabs import _DetachTabBar, _DetachTabs, _FloatWindow  # noqa: F401 (re-export)
from squidmip._qtstyle import dark_palette as _dark_palette
from squidmip._qtstyle import hline as _hline
from squidmip._terminal import _CmdEdit, _ProcTerminal, _Terminal  # noqa: F401 (re-export)
from squidmip._measure import (
    FAILED as _MEASURE_FAILED, OK as _MEASURE_OK, PARTIAL as _MEASURE_PARTIAL,
    STOPPED as _MEASURE_STOPPED, measure_run,
)
from squidmip._region_nav import RegionCursor, RegionSlider
from squidmip._spots import LAYER_KEY as _SPOTS_LAYER_KEY

# (`_SUPPORTED_PLATES` and `resolve_plate_format` used to live here. `build_plate` (IMA-214) is now
#  the single format-resolution path — override > measured > declared > inferred — so both were dead
#  leftovers that could only ever disagree with it. Deleted rather than left as a second opinion.)
_CELL = 88                # per-well px in the low-res overview (1536wp -> ~4224x2816)
_PUSH_PX = 512             # per-well px pushed to the ndviewer scan-slider (downsampled -> bounded RAM)
_HDR, _COLH = 46, 30       # left / top label margins (px)
_PAD = 16                  # breathing room around the plate
_VIEWER_WORKERS = min(6, _default_workers())   # adapt to the machine, but CAP at 6: the producer's
                           # peak RAM is ~workers x one-well (~139 MB each on a 1536wp), and projection
                           # throughput scales only sublinearly past ~6 threads — so more workers buys
                           # little speed for linearly more memory. 6 balances both, leaves GUI cores.
# Chrome (colours, stylesheets, palette) is defined ONCE in `squidmip._qtstyle` and aliased here
# so existing call sites keep their short private names. These are NOT second definitions: change
# a colour in _qtstyle and every widget in the window moves with it.
_BG = _qtstyle.BG
_GRID, _RED, _MUTED, _ACCENT = _qtstyle.GRID, _qtstyle.RED, _qtstyle.MUTED, _qtstyle.ACCENT
_SEL_FILL = _qtstyle.SEL_FILL
#: The plate's region highlight — a MORE TRANSPARENT light-blue wash than _SEL_FILL (Julio). Shown
#: on the manually-picked wells AND on the regions of the open view you click (highlight_regions).
_VIEW_WASH = QColor(88, 166, 255, 40)   # ~16% alpha light blue


def _view_hue(view_id: int, *, focused: bool = False) -> QColor:
    """A STABLE, distinct hue per open view/thread, so the plate colour-codes which wells belong to
    which window (Julio: "colour hueing the different view threads"). The golden-ratio hue step keeps
    successive views far apart on the wheel; the focused view is more opaque so it reads as active."""
    h = (0.13 + 0.61803398875 * int(view_id)) % 1.0     # golden-ratio walk => maximally spread hues
    c = QColor.fromHsvF(h, 0.62, 1.0, 0.34 if focused else 0.20)
    return c
_MIN_PREVIEW_BOX_PX = 4    # smallest FOV box (of _CELL) the RAW preview will bother mosaicking
#                            (IMA-253): below this a field is a speck, and reading one plane per
#                            field to draw specks is pure cost. The operator path is unaffected.
_CLICK_SLOP = 3                       # px of travel below which a Shift-drag counts as a click
#                                        (matches the pan threshold, so the two gestures agree)
_CONTROL_BLUE = _qtstyle.CONTROL_BLUE

_EMPTY_BODY_PX = _qtstyle.EMPTY_BODY_PX   # the legibility floor; see squidmip/_qtstyle.py
_EMPTY_HEAD_PX = _qtstyle.EMPTY_HEAD_PX

# The empty exploration pane's copy (IMA-260). Framed as an EXAMPLE of what you might do, never as
# an instruction: Julio asked for "example usage", so the pane shows one concrete path and then
# Control Well first (Julio's stated priority), Shift-drag second. Plain sentences, no jargon,
# and no hedging: the previous copy said "here is an example", "for example" and "these are only
# examples" in four consecutive paragraphs, which reads as apologetic rather than instructive.
# Julio: "The exploration pane message is really unprofessional and unlike AI."
#
# It also described the WRONG ROLE. Operator results belong in the plate view and the centre
# viewer as toggleable layers -- pane 3 is SUPPLEMENTARY (3D rendering, decon previews, fields
# worth keeping in view). Copy that promises results will "land here" teaches the wrong model.
_EMPTY_EXPLORE_HEAD = "Exploration"
_EMPTY_EXPLORE_LEDE = (
    "A second viewer, for a subset of the plate. Operator results appear as layers in the plate "
    "and the centre viewer \u2014 not here.")
#: PRIMARY line, WELL PLATE only. Julio: "You say control well, but that feature is only for our
#: well plate acquisition. For tissue acquisition we could print the user 'open in exploration
#: pane'." A control well is a plate concept -- on a glass slide with hand-drawn regions there is
#: nothing to control against, and naming a gesture the user cannot perform is worse than silence.
_EMPTY_EXPLORE_PRIMARY = (
    "Right-click a well and choose Control Well to pin it here, so you can compare the rest "
    "against it.")
#: PRIMARY line for a SLIDE / tissue acquisition, where the unit is a region, not a well.
_EMPTY_EXPLORE_PRIMARY_SLIDE = (
    "Double-click a region on the slide and choose Open in exploration pane to bring it here.")
_EMPTY_EXPLORE_SECONDARY = (
    "Hold Shift and drag across the plate to open a subset in its own tab, with a slider to "
    "step through it.")
_EMPTY_EXPLORE_SECONDARY_SLIDE = (
    "Hold Shift and drag to open several regions in one tab, with a slider to step through "
    "them.")
_EMPTY_EXPLORE_CODA = (
    "Use it for 3D volume rendering, deconvolution previews, and fields worth keeping in view.")
_EXPLORE_W = 380                      # pane 3's width on open, in px (see PlateWindow.__init__)

_STATUS = _qtstyle.STATUS   # processing-status hue coding; see squidmip/_qtstyle.py
_NDV_DARK = _qtstyle.NDV_DARK
_TABS_DARK = _qtstyle.TABS_DARK
_CARD_QSS = _qtstyle.CARD_QSS
_BTN_QSS = _qtstyle.BTN_QSS
_COMBO_QSS = _qtstyle.COMBO_QSS
_CHECK_QSS = _qtstyle.CHECK_QSS
_TERM_QSS = _qtstyle.TERM_QSS
_MENU_QSS = _qtstyle.MENU_QSS
_ANSI_RE = _qtstyle.ANSI_RE


def _signal_names(cls) -> tuple:
    """Every pyqtSignal declared on *cls* or its bases, by attribute name.

    ``pyqtSignal`` is a class attribute until Qt binds it per-instance, so the class object is
    where the declarations are discoverable. Excludes ``finished``/``started`` — QThread's own,
    which the retire path connects deliberately and must not tear down.
    """
    from PyQt5.QtCore import pyqtSignal as _sig
    seen, out = set(), []
    for klass in cls.__mro__:
        for name, value in vars(klass).items():
            if name in seen or name in ("finished", "started"):
                continue
            if isinstance(value, _sig) or type(value).__name__ in ("pyqtSignal", "unbound_signal"):
                seen.add(name)
                out.append(name)
    return tuple(out)


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
    runnable: bool = True
    """Whether the ENGINE can run this key (`runnable_operators()`), as opposed to the card
    merely existing. The two registries are deliberately not the same set - a card is
    presentation, an engine entry is capability - but that was written in a comment and
    enforced nowhere, so a card whose key the engine does not know produced a button that
    silently did nothing. Declaring it here makes the divergence checkable, and
    test_every_card_declares_whether_it_is_a_runnable_operator checks it."""

# The operator registry. MIP is operator #1; append an Operation + write its `_build_*_tab` and both
# the console cards and the Process-well-plates menu grow automatically.
_OPERATIONS = (
    Operation("mip", "Maximum Intensity Projection",
              "Collapse each well's z-stack to one max-intensity image; save a navigable OME-Zarr plate.",
              "_build_mip_tab"),
    Operation("stitch", "Stitch (register + fuse)",
              "Register every FOV of a well against its neighbours and fuse one seamless mosaic "
              "per well, instead of trusting the stage coordinates alone.",
              "_build_stitch_tab"),
    # NOT an operator: an export hand-off. Handing "minerva" to the engine dies with a raw
    # KeyError: unknown projector 'minerva'. Declared, rather than left to be rediscovered.
    Operation("minerva", "Open in Minerva Author",
              "Export the selected FOVs to Minerva-ingestable OME-TIFFs and open Minerva Author on them.",
              "_build_minerva_tab", runnable=False),
    # IMA-223/224/225 -- the PLANE-OPS. Unlike mip/stitch these keep z at full depth, so they get
    # _build_plane_op_tab (preview only) rather than _build_run_tab: write_plate's _validate_image
    # accepts Z == 1 only and would fail LOUD on save. Loud is correct; offering the button is not.
    # The blurb said "the microscope's Gaussian PSF ... no explicit kernel". Both halves were
    # false and had been since IMA-247 deleted the reimplementation: the kernel is a VECTORIAL
    # PSF computed from the acquisition's own optics (NA 0.3 on this scope), and it is very much
    # explicit. A card that describes the wrong algorithm is how a user picks the wrong operator.
    Operation("decon", "Deconvolution (Richardson-Lucy)",
              "Sharpen against a vectorial PSF computed from this acquisition's own optics (NA, "
              "emission wavelength, pixel size, z-step) -- not an assumed Gaussian. Richardson-Lucy "
              "is semi-convergent, so the iteration count is chosen by eye against a turbo x-z / "
              "y-z view rather than defaulted.",
              "_build_decon_tab"),
    Operation("bgsub", "Background subtraction",
              "Remove the smooth out-of-focus haze from every plane with a rolling ball (ImageJ's "
              "algorithm). A LAYER: the raw is untouched on disk and one toggle away.",
              "_build_bgsub_tab"),
    Operation("flatfield", "Flat-field correction",
              "Divide out the objective's illumination profile so the corners match the centre. "
              "Needs an illumination profile (.npy) from the stitcher or estimated from the plate.",
              "_build_flatfield_tab"),
)
_OPERATIONS_BY_KEY = {op.key: op for op in _OPERATIONS}

# The operator "Save this subset to disk…" runs. This used to be spelled `_OPERATIONS[0].key`,
# which made a PRESENTATION edit (reordering the cards) silently change which operator the save
# button RUNS. Named, so the two cannot be confused.
_SAVE_OPERATOR = "mip"

#: Registry key of the AGAVE 3D tab. Deliberately NOT an Operation key: the 3D view produces no
#: plate result, so it is never offered in the "Process well plates" menu and never routes a
#: result into pane 3 (results belong in the plate view and the centre viewer, as layers).
AGAVE_KEY = "agave3d"

# Roadmap cards shown under "TO BE ADDED", as (label, blurb). Empty: everything currently on the
# roadmap that we're willing to advertise has shipped as a real Operation above. Add an entry when
# a next operator (e.g. the Nautilus agent) is close enough to promise.
_TO_BE_ADDED: list = []


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


def cells_in_rect(rows, cols, by_rc, x0: float, y0: float, x1: float, y1: float,
                  cell_disp: float) -> list:
    """Every ACQUIRED cell whose square meets the drag rect (x0,y0)-(x1,y1), row-major sorted.

    Same plate-pixel space as ``well_at`` (label margins already removed by the caller). The rect
    is NORMALIZED first, so an up-left drag selects exactly what the equivalent down-right drag
    does. Out-of-grid edges clamp instead of inventing cells, and a cell is returned only when
    ``by_rc`` holds a well there — a marquee over a sparse plate never selects the un-acquired
    positions the grey dots mark.

        by_rc = {(0,0):A1, (1,1):B2}          drag (0,0)->(39,39) at 20px/cell
        +-------+-------+
        |  A1   |  A2   |   -> [(0,0), (1,1)]   A2/B1 are plate positions, not acquisitions
        |  (B1) |  B2   |
        +-------+-------+
    """
    if cell_disp <= 0:
        return []
    lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)      # normalize: any drag direction is equal
    lo_y, hi_y = (y0, y1) if y0 <= y1 else (y1, y0)
    if hi_x < 0 or hi_y < 0:                             # entirely above/left of the plate
        return []
    c0, c1 = int(max(0.0, lo_x) // cell_disp), int(max(0.0, hi_x) // cell_disp)
    r0, r1 = int(max(0.0, lo_y) // cell_disp), int(max(0.0, hi_y) // cell_disp)
    c1, r1 = min(c1, len(cols) - 1), min(r1, len(rows) - 1)   # clamp at the far edge
    return [(ri, ci) for ri in range(r0, r1 + 1) for ci in range(c0, c1 + 1) if (ri, ci) in by_rc]


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


def _fit_box(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a 2D plane to EXACTLY (h, w) — the arbitrary-target sibling of :func:`_fit_cell`.

    Used to place one FOV into its box inside a multi-FOV mosaic cell (IMA-187), where each box
    is a fraction of _CELL and generally not square. Same policy as _fit_cell: area-downsample
    when shrinking (the normal case — a 2084px frame into a ~15px box), nearest-sample when
    upscaling, so a tiny synthetic frame in a test can never crash the render path.
    """
    h, w = max(1, int(h)), max(1, int(w))
    if a.shape == (h, w):
        return a
    if a.shape[0] >= h and a.shape[1] >= w:
        return _area_downsample(a, h, w)
    yi = (np.arange(h) * a.shape[0]) // h
    xi = (np.arange(w) * a.shape[1]) // w
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


# Pane 3's identity and label rules live in ``_explore`` (no Qt, no napari), and are re-exported
# here under their historical names so every existing caller and test is unchanged. They MOVED
# rather than being copied: two spellings of "what is this tab called" is the same
# two-representations-of-one-truth defect this file already carries scars from.
exploration_tab_key = _explore.exploration_tab_key
exploration_tab_label = _explore.exploration_tab_label


def operator_layer_key(op_key: str, tab_key: Optional[str]) -> str:
    """Layer id an operator's results are filed under.

    Plate-wide runs keep the bare operator key ("mip") so existing behaviour is byte-identical.
    A run scoped to an exploration tab gets "<op>@<tab_key>" — without this, two tabs running
    the same operator both write into PlateOverview._op_canvas["mip"] and silently overwrite
    each other's tiles."""
    return f"{op_key}@{tab_key}" if tab_key else op_key


def runnable_operators() -> list[str]:
    """Every operator ``run_operator`` can stream live (IMA-226) — read off the ENGINE registry,
    never off ``_OPERATIONS``.

    The two lists are not the same set and never were:

    * ``reference`` is a registered projector with no card, so ``_OPERATIONS_BY_KEY[key].label``
      raised a bare ``KeyError`` out of the event loop the moment anything asked to run it.
    * ``minerva`` is a card that is NOT an operator — it is an export hand-off. Handing its key to
      the engine dies with a raw ``KeyError: unknown projector 'minerva'`` in the status line.

    Both are cured by asking the engine what it can run. A z-reducer and a region operator stream
    through the SAME ``_OperatorWorker`` (it already branches ``project_plate``/``stitch_plate`` on
    ``available_region_operators``), so both belong in one list.
    """
    from squidmip import available_projectors, available_region_operators

    return sorted(set(available_projectors()) | set(available_region_operators()))


def operator_label(key: str) -> str:
    """Human label for an operator: its card's if it has one, else the registry name itself.

    A card is presentation, not capability (IMA-226) — an operator with no card must still be
    runnable and must still name itself in the status line and the layer stack."""
    op = _OPERATIONS_BY_KEY.get(key)
    return op.label if op is not None else key


class _RunningContrast:
    """Per-channel global contrast that updates as wells stream in (histogram over tiles so far).

    Each channel also carries an auto/manual LATCH (IMA-206). The histogram keeps growing while a
    run streams, so an untouched channel keeps auto-scaling; the first time the user drags that
    channel's contrast it latches MANUAL and the next well to land can no longer stomp the window
    they just set. ``set_auto`` unlatches it back onto the running histogram.
    """

    def __init__(self, n_ch: int, dmax: float, pct=(1.0, 99.8), bins=512):
        self._bins, self._dmax, self._pct = bins, max(1.0, float(dmax)), pct
        self._hist = [np.zeros(bins, dtype=np.int64) for _ in range(n_ch)]
        self._manual: dict[int, tuple[float, float]] = {}   # ch -> the window the USER latched
        # ch -> the window the OWNING VIEWER (ndviewer_light) resolved and is rendering with.
        # Deliberately NOT the same dict as _manual: see set_followed.
        self._followed: dict[int, tuple[float, float]] = {}

    @property
    def dmax(self) -> float:
        return self._dmax

    def add(self, ch: int, tile: np.ndarray):
        idx = np.clip((tile.ravel() / self._dmax * self._bins).astype(int), 0, self._bins - 1)
        self._hist[ch] += np.bincount(idx, minlength=self._bins)

    def set_manual(self, ch: int, lo: float, hi: float):
        """Latch *ch* to a user-set window (hi is kept above lo so _window never divides by zero)."""
        self._manual[ch] = (float(lo), float(max(hi, lo + 1)))

    def set_auto(self, ch: int):
        """Unlatch *ch* — it goes back to following the running histogram."""
        self._manual.pop(ch, None)

    def is_manual(self, ch: int) -> bool:
        """Did the USER latch this channel? Never true merely because the viewer autoscaled."""
        return ch in self._manual

    def set_followed(self, ch: int, lo: float, hi: float):
        """Record the window the OWNING VIEWER resolved for *ch* (IMA-261).

        THIS IS NOT A LATCH, AND THE DISTINCTION IS THE WHOLE POINT
        ------------------------------------------------------------
        The first version of this recorded ndv's window by calling ``set_manual``, which read as
        "the user has taken manual control of this channel". It was wrong twice over, and both
        showed on screen:

          * ndv autoscales on its own, at open, before the user has touched anything — so every
            channel came up latched MANUAL and the plate's running histogram was permanently
            overridden. Auto-contrast was dead from the first frame.
          * ``resolve`` puts a manual latch above everything, so under SCOPE_PER_REGION every cell
            resolved to ndv's one global window. All 1536 wells were painted identically while the
            plate still drew the amber "wells NOT comparable" badge over the top. The control did
            nothing and the caveat was a lie.

        A followed window is a SINK recording what the owner is showing. A manual latch is a
        POLICY decision, and only the user makes it — the sink never writes policy back into the
        model. Same numbers, different authority, and the authority is what ``resolve`` reads.
        """
        self._followed[ch] = (float(lo), float(max(hi, lo + 1)))

    def clear_followed(self, ch: int):
        self._followed.pop(ch, None)

    def is_followed(self, ch: int) -> bool:
        return ch in self._followed

    def resolve(self, ch: int, auto: tuple[float, float],
                follow: bool = True) -> tuple[float, float]:
        """THE precedence rule, in one place (IMA-242, extended by IMA-261).

            user latch  >  the owning viewer's window  >  whatever the caller computed

        Every renderer derives its own *auto* window legitimately — the plate from the running
        histogram, a per-region cell from exact percentiles over that cell, the loupe from a
        well's coarse plane. What they must NOT each decide for themselves is whether the user's
        gesture outranks that, because a renderer that forgets to ask is a renderer where the
        control silently does nothing. One rule, one place, three callers.

        *follow* is how a caller says "I am not rendering the viewer's global view". Only
        SCOPE_PER_REGION passes False, and it means exactly what the user asked for by choosing
        per-region: derive this cell's window from this cell's pixels. A user's explicit latch
        still wins even then — that is a decision about the channel, not about the scope.
        """
        if ch in self._manual:
            return self._manual[ch]
        if follow and ch in self._followed:
            return self._followed[ch]
        return auto

    def window(self, ch: int) -> tuple[float, float]:
        """(lo, hi) for this channel: user latch, else the viewer's, else the running histogram."""
        return self.resolve(ch, self._auto_window(ch))

    def _auto_window(self, ch: int) -> tuple[float, float]:
        """The FLUORESCENCE window for *ch* from the running histogram, ignoring any latch.

        This is the maragall/stitcher rule (``_contrast.auto_contrast``): background peak to BLACK,
        99.9th percentile on top — the SAME rule the viewer windows use. It replaces a plain
        (1st, 99.8th) percentile low end, which lands INSIDE the fluorescence background so the
        whole field lifts off black and saturates (``_contrast`` module docstring: "a percentile
        window washes fluorescence out"). The plate used to get the good window only by FOLLOWING
        the central pane; with the pane gone (decentralized root) the plate must carry the rule
        itself, and it already keeps the per-channel histogram the rule needs.

        A DEGENERATE window (hi <= lo) is returned DELIBERATELY for a blank/flat channel —
        ``_window``'s ``span <= 0`` guard renders that black, the honest answer when there is no
        contrast. Blank wells are normal on a partially acquired plate and must not read as signal.
        """
        h = self._hist[ch].astype(np.float64)
        tot = h.sum()
        if tot == 0:
            return 0.0, self._dmax
        centers = (np.arange(self._bins) + 0.5) / self._bins * self._dmax
        cdf = np.cumsum(h) / tot
        mode_val = float(centers[int(np.argmax(h))])                 # background peak = the mode
        # std of the BACKGROUND (bins at or below the median), computed from the histogram.
        med_bin = min(int(np.searchsorted(cdf, 0.5)), self._bins - 1)
        bg = h[: med_bin + 1]
        bg_tot = float(bg.sum())
        if bg_tot > 0:
            bc = centers[: med_bin + 1]
            bg_mean = float((bc * bg).sum() / bg_tot)
            bg_std = float(np.sqrt(max(0.0, ((bc - bg_mean) ** 2 * bg).sum() / bg_tot)))
        else:
            bg_std = abs(mode_val) * 0.1
        lo = mode_val + 2.0 * bg_std                                 # push background to black
        hi = float(centers[min(int(np.searchsorted(cdf, 0.999)), self._bins - 1)])   # 99.9th pct
        if hi <= lo:
            return lo, lo                                            # degenerate -> black
        return float(lo), float(hi)

    def windows(self) -> list[tuple[float, float]]:
        return [self.window(ch) for ch in range(len(self._hist))]


# --- contrast scope (IMA-207): how wide a net each contrast window is computed over ------------
#
#   GLOBAL      one window per channel across the whole plate. Wells stay COMPARABLE — a dim well
#               looks dim — but a dim region can be crushed to black beside a bright one.
#   PER_REGION  one window per channel PER CELL. Every region fills its own range, so a dim and a
#               bright region are both readable at once — at the cost of comparability: two wells
#               that look identical may differ by orders of magnitude. That is why the active
#               scope is drawn INTO the plate (see paintEvent) rather than living only in a
#               dropdown that a screenshot would crop out.
#
# Scope is a DISPLAY control, NOT a run parameter. Flipping it re-composites from the native-dtype
# tiles PlateOverview already retains (IMA-206's per-layer store); it never re-runs the plate,
# because a 1536wp run is minutes and that would make the control unusable.
#
# PER_FOV is deliberately absent. It slots into `_scoped_windows`' bucketing when someone wants
# per-field windows inside a mosaic cell — no other change.

_PCT = (1.0, 99.8)   # clip the darkest 1% / brightest 0.2% so hot pixels don't crush the window


def _pct_window(a: np.ndarray, pct=_PCT) -> tuple[float, float]:
    """EXACT percentile window over *a*.

    Exactness is the point. ``_RunningContrast`` quantizes to a bin ~dmax/bins wide (~128 counts on
    uint16), so a dim region spanning a few hundred counts collapses into two or three bins and its
    window comes out garbage — precisely the region PER_REGION exists to rescue. The histogram
    stays the live during-run approximation; this is what the final render uses.

    A degenerate result (hi <= lo) is returned as-is on purpose: ``_window`` renders it black.
    """
    if a.size == 0:
        return 0.0, 0.0
    lo, hi = np.percentile(a, pct)
    return float(lo), float(hi)


# --- loupe (IMA-208): press-and-hold magnifier over the plate ------------------------------
#
# The plate montage CANNOT be the loupe's source: a tile is _CELL (88) px per well, a ~47x
# downsample of a ~4168px field (see _fit_cell), so magnifying it yields interpolation, not
# pixels. The loupe therefore reads the real data behind whatever layer is on screen — the
# acquisition's TIFFs in raw mode, a windowed read of the written pyramid otherwise.
#
# Magnification is derived from the CURRENT plate zoom (so it is dynamic, per the spec) and
# capped at native resolution (so it never invents detail):
#
#     s_plate = cd / well_px            screen px per image px, at the current plate zoom
#     s_loupe = min(1.0, MAG * s_plate) screen px per image px inside the inset (cap = native)
#     M       = s_loupe / s_plate       actual magnification, in (1, MAG]
#     L       = coarsest pyramid level whose own pixels are still >= s_loupe
#
# Reading level L instead of level 0 is what keeps this cheap: L is chosen so the level's
# pixels land ~1:1 on the inset's screen pixels, so the crop is a few hundred px per side no
# matter how far out the plate is zoomed.

_LOUPE_PX = 240            # inset size on screen (px)
_LOUPE_MAG = 8.0           # target magnification over the plate's current scale
_LOUPE_HOLD_MS = 350       # press-and-hold dwell before the loupe arms
_LOUPE_SLOP = 3            # cursor may drift this many px while arming (matches the pan threshold)
_LOUPE_CACHE = 8           # decoded crops kept (small: a crop is a few MB, not a whole well)
_LOUPE_MAX_CROP = 2 * _LOUPE_PX   # ceiling on the RETURNED array's side, in px
# Why a ceiling at all, when level selection is supposed to bound the read: a source can run OUT
# of levels. Raw TIFFs have no pyramid on disk (n_levels == 1), and a written field below
# _PYRAMID_MIN_YX collapses to level 0 alone, so loupe_level clamps to 0 and the crop becomes
# inset/s_loupe — the WHOLE field. Measured on the 2084 px synthetic plate at fit: a 1826 px
# crop, 4 channels, 26.7 MB, composited ON THE GUI THREAD at ~118 ms per cursor move, and the
# worker's LRU — keyed on (well, level, y0, x0, h, w), i.e. a new key for every pixel of motion
# — held eight of them (213 MB). A 4168 px field is 4x worse on both counts. The fix is decimation, not truncation: the requested RECTANGLE
# still defines the region the inset covers (truncating it would silently change the
# magnification), but a source returns it at no more than this many samples per side. The inset
# is 240 px on screen; beyond 2x that, nobody can see the difference.


def _fov_of_well(well_id, fovs_per_region=None) -> int:
    """The FOV index the plate addresses for ``well_id`` — the single seam for multi-FOV.

    The plate hit-test resolves a WELL, never a FOV, so today this is always 0: the viewer is
    one-FOV-per-well (the library folded IMA-187, the viewer has not). Everything that needs a
    FOV goes through here rather than writing a bare ``0``, so when the plate grows FOV
    sub-cells there is one place to change and ``test_fov_seam_is_single_fov`` fails loudly
    instead of the loupe silently magnifying FOV 0 of every position."""
    if fovs_per_region:
        fovs = fovs_per_region.get(well_id)
        if fovs:
            return int(fovs[0])
    return 0


def loupe_scale(cd: float, well_px: int, mag: float = _LOUPE_MAG,
                inset_px: int = _LOUPE_PX) -> tuple[float, float]:
    """(s_loupe, M) for a plate showing ``cd`` screen px per well of ``well_px`` image px.

    ``s_loupe`` is clamped to 1.0 — one screen pixel per level-0 image pixel is as far as
    honest magnification goes; past that we would be upsampling, which is the very thing
    the montage already does badly. ``M`` is what the user actually gains, in [1, mag].

    Two lower clamps, both learned the hard way:

    * Once the user has wheel-zoomed the plate PAST native (a well drawn bigger than its own
      pixel count), the 1.0 cap alone would put the inset BELOW the plate's own scale — a
      loupe that shrinks what it points at. Floor at the plate's scale; the caller labels
      that case "native", since there is no detail left to reveal.
    * A fixed target magnification does not survive a 1536-well plate. At fit, a well is ~10
      screen px, so 8x fills only ~85 px of a 240 px inset and the rest would have to come
      from neighbouring wells. Floor at ``inset_px / well_px`` so the inset shows AT MOST one
      whole well — which is also what the gesture means: look closely at *this* well. On a
      1536wp that yields ~22x rather than 8x, still derived entirely from the plate's zoom."""
    well_px = max(1, int(well_px))
    s_plate = max(1e-9, float(cd) / well_px)
    fill_well = float(inset_px) / well_px             # scale at which one well fills the inset
    # Order matters: cap at native FIRST, then floor at the plate's own scale. Capping last
    # would drag a plate that is already past native back down to 1.0 and demagnify.
    s_loupe = max(s_plate, min(1.0, max(mag * s_plate, fill_well)))
    return s_loupe, s_loupe / s_plate


def loupe_level(s_loupe: float, n_levels: int) -> int:
    """Coarsest pyramid level whose native resolution still satisfies ``s_loupe``.

    Level L is downsampled by 2**L, so its pixels carry scale 1/2**L relative to level 0. We
    want the largest L with 2**-L >= s_loupe, i.e. L <= log2(1/s_loupe). Clamped into the
    levels that actually exist (a small field writes a single level — see _PYRAMID_MIN_YX)."""
    s = min(1.0, max(1e-9, float(s_loupe)))
    return int(max(0, min(int(np.floor(np.log2(1.0 / s))), max(0, int(n_levels) - 1))))


def loupe_crop_px(s_loupe: float, level: int, inset_px: int = _LOUPE_PX) -> int:
    """Image pixels to read AT ``level`` to fill an ``inset_px`` square inset."""
    eff = max(1e-9, float(s_loupe) * (2 ** int(level)))   # screen px per level-``level`` px
    return int(max(1, np.ceil(inset_px / eff)))


def loupe_decimation(crop_px: int, max_px: int = _LOUPE_MAX_CROP) -> int:
    """Power-of-two stride that brings a ``crop_px``-wide read down to <= ``max_px`` samples.

    Applied by the SOURCE, after the rectangle is fixed: the region the inset covers is set by
    the crop rect and must not change, only the sample count within it."""
    step = 1
    while crop_px // step > max(1, int(max_px)):
        step *= 2
    return step


def loupe_clamp_crop(y0: int, x0: int, h: int, w: int, ny: int, nx: int):
    """Fit a crop rect inside a ``ny`` x ``nx`` field: shift the ORIGIN in, keep the extent.

    Every source must do this, and the reason it is a free function rather than four lines
    repeated per source is IMA-208's primary bug: ``_ZarrLoupeSource`` clamped and
    ``_RawLoupeSource`` did not, so raw mode — the DEFAULT on every folder open — passed a
    negative origin straight into a numpy slice. ``a[-427:1399]`` is not an error, it is an
    EMPTY array, so the inset said "no pixels here" over the ~75% of every well whose crop
    starts left of or above the field. Clamping the origin (rather than truncating the extent,
    which would return a 1 px sliver at an edge) keeps the inset full near the field border."""
    ny, nx = max(1, int(ny)), max(1, int(nx))
    h, w = max(1, min(int(h), ny)), max(1, min(int(w), nx))
    return max(0, min(int(y0), ny - h)), max(0, min(int(x0), nx - w)), h, w


def loupe_um_per_screen_px(pixel_size_um, s_loupe: float):
    """µm per SCREEN pixel inside the inset, or None when the pixel size isn't trustworthy.

    Returns None rather than a guess. ``_output.py`` writes 1.0 into the multiscales scale for
    BOTH "unknown" and a genuine 1.0 µm/px, so a computed plate cannot distinguish them (see
    TODOS.md) — callers pass None for that case. A microscopy tool that displays a confidently
    wrong micron figure is worse than one that admits it doesn't know."""
    if pixel_size_um is None:
        return None
    p = float(pixel_size_um)
    if not np.isfinite(p) or p <= 0:
        return None
    return p / max(1e-9, float(s_loupe))


def _nice_scale_um(rough: float) -> float:
    """Round a scale-bar length to a 1/2/5 x 10^n figure, the way a microscope overlay would."""
    rough = max(1e-6, float(rough))
    decade = 10.0 ** np.floor(np.log10(rough))
    for step in (1.0, 2.0, 5.0, 10.0):
        if rough <= step * decade:
            return step * decade
    return 10.0 * decade


def _fmt_um(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:g} mm"
    return f"{v:g} µm" if v >= 1 else f"{v * 1000:g} nm"


# IMA-242: `_composite_rgb` and `_percentile_window` used to live here — a second compositor and a
# second percentile rule, each a hand-synced twin of `composite` and `_pct_window`. They had drifted
# apart in exactly the way that shape always drifts:
#
#   * `_percentile_window` widened a degenerate window to (lo, lo + 1); `_pct_window` deliberately
#     does NOT, because +1 is one DATA unit and (v - lo)/1 clips to 1.0 — a blank or saturated
#     channel rendered FULL WHITE and read as signal. The loupe had the bug the plate had fixed.
#   * `_composite_rgb` took no channel mask, so unticking a channel removed it from the plate and
#     left it in the loupe.
#   * Neither consulted the manual latch, so dragging a contrast slider moved the plate and left
#     the loupe showing the old window forever.
#
# Both are now gone. `composite` is the one compositor, `_pct_window` the one percentile rule, and
# `_RunningContrast.resolve` the one place the manual-outranks-auto precedence is decided.


_LOUPE_WIN_LOCK = threading.Lock()   # guards the per-source window memo (worker thread writes)


class _LoupeSource:
    """Where the loupe's real pixels come from for the layer currently on the plate.

    Availability is per (source, WELL) — never per layer key. A layer key cannot express what
    is actually on disk: ``OperationStack.add`` dedupes by key, so a saved run and a later
    unsaved preview collapse into one "mip" layer while ``_processed_plate`` still points at
    the older save. Ask the source about the specific well instead."""

    n_levels = 1
    well_px = 1
    pixel_size_um = None

    def available(self, well_id) -> tuple[bool, str]:
        """(ok, reason-if-not). ``reason`` is shown to the user verbatim."""
        return False, "no pixel source"

    def read_crop(self, well_id, level, y0, x0, h, w):
        """(C, y, x) crop at ``level``, CLAMPED into the field (see loupe_clamp_crop) and
        decimated to at most _LOUPE_MAX_CROP samples per side. Runs on the worker thread."""
        raise NotImplementedError

    def coarse(self, well_id):
        """A small whole-field (C, y, x) plane used ONLY to derive the contrast window."""
        raise NotImplementedError

    def window(self, well_id):
        """Per-channel contrast window for a well, mirroring the tile's rule.

        Computed HERE, on the loupe worker thread, and memoised per well — never on the GUI
        thread. It used to be derived in ``_on_loupe_crop`` by calling ``coarse()``, which for
        raw meant decoding a whole TIFF plane inside a paint-driven slot AND touching the same
        plane cache the worker was writing (two threads, no lock, one well's pixels labelled as
        another's). One owner, one thread."""
        with _LOUPE_WIN_LOCK:
            cache = self.__dict__.setdefault("_win_cache", {})
            hit = cache.get(well_id)
        if hit is not None:
            return hit
        coarse = self.coarse(well_id)
        win = [_pct_window(coarse[c]) for c in range(coarse.shape[0])]
        with _LOUPE_WIN_LOCK:
            cache[well_id] = win
        return win


class _RawLoupeSource(_LoupeSource):
    """Raw-acquisition source: the loupe works the moment a folder is open, before any operator.

    Reads the same representative plane per channel that _PreviewWorker already reads, so the
    inset shows exactly the data the raw plate tile was built from. Individual TIFFs hold one
    plane per file and aren't tiled, so a crop means decoding that plane — hence the one-well
    plane cache. Bounded to a single well's channels (~C x frame bytes)."""

    def __init__(self, reader, meta, fov_of):
        self._reader, self._meta, self._fov_of = reader, meta, fov_of
        ny, nx = meta["frame_shape"]
        self.well_px = int(min(ny, nx))
        self.n_levels = 1                      # raw TIFFs have no pyramid ON DISK
        self.pixel_size_um = meta.get("pixel_size_um")
        self._channels = [c["name"] for c in meta["channels"]]
        zs = meta["z_levels"]
        self._z = zs[len(zs) // 2]             # mid plane, as the preview does
        self._lock = threading.RLock()         # _planes is touched by the worker AND the GUI thread
        self._cache_key = None
        self._cache = None
        self._coarse: dict[str, np.ndarray] = {}

    def available(self, well_id) -> tuple[bool, str]:
        if well_id in self._meta["regions"]:
            return True, ""
        return False, "no image for this well"

    def _planes(self, well_id):
        """The well's (C, y, x) planes, decoded once and cached.

        Held under a lock for the whole check-decode-publish sequence. Unsynchronised, the two
        callers (worker thread reading a crop, GUI thread deriving a window) could interleave
        between the key test and the store and hand back ANOTHER well's pixels labelled as the
        well under the cursor — a wrong-image bug in a microscopy tool, not a glitch. The GUI
        thread no longer calls in at all (see _LoupeSource.window), but the lock stays: the class
        must be correct for its callers, not for today's call sites."""
        with self._lock:
            if self._cache_key != well_id:
                fov = self._fov_of(well_id)
                planes = np.stack([
                    np.asarray(self._reader.read(well_id, fov, ch, self._z))
                    for ch in self._channels])
                self._cache, self._cache_key = planes, well_id
            return self._cache

    def read_crop(self, well_id, level, y0, x0, h, w):
        """Level is always 0 here — raw has no pyramid — so the whole burden of bounding the
        work falls on decimation. At plate fit the rect IS most of the field (2084 px on the
        synthetic plate); area-averaging it down to <= _LOUPE_MAX_CROP happens HERE, on the
        worker thread, so what crosses to the GUI thread to be composited is a 456 px square
        (3.3 MB, 11 ms) instead of a 1826 px one (26.7 MB, 118 ms) — which is also what the
        worker's LRU then caches."""
        p = self._planes(well_id)
        ny, nx = p.shape[-2], p.shape[-1]
        y0, x0, h, w = loupe_clamp_crop(y0, x0, h, w, ny, nx)   # NEGATIVE origin -> empty slice
        crop = p[:, y0:y0 + h, x0:x0 + w]
        step = loupe_decimation(max(h, w))
        if step == 1:
            return crop
        oh, ow = max(1, h // step), max(1, w // step)
        # float32, not _area_downsample's float64 default: the compositor casts to float32 anyway,
        # and this array crosses a thread boundary and sits in the worker's LRU.
        return np.stack([_area_downsample(crop[c], oh, ow).astype(np.float32, copy=False)
                         for c in range(crop.shape[0])])

    def coarse(self, well_id):
        if well_id not in self._coarse:
            p = self._planes(well_id)
            self._coarse[well_id] = np.stack(
                [_area_downsample(p[c], _CELL, _CELL) for c in range(p.shape[0])])
        return self._coarse[well_id]


class _ZarrLoupeSource(_LoupeSource):
    """Written-plate source: a WINDOWED tensorstore read of one pyramid level.

    Deliberately NOT _ComputedPlateWorker._read, which pulls a whole plane (~139 MB per well at
    level 0 on a 1536wp) — right for its one-pass streaming job, ruinous for a gesture that
    re-reads as the cursor moves. Arrays are chunked (1, 1, 1, <=1024, <=1024) (_zarr_store)
    precisely so a viewer can read a region, so a loupe crop touches a handful of chunks.

    ``written`` is the set of wells this run has actually persisted. It grows as wells land, so
    the loupe works on completed wells DURING a long run, and a subset save / failed well is
    reported as "not written yet" instead of magnifying some other well's pixels."""

    def __init__(self, base, path_of, fov_of, levels, well_px, pixel_size_um, written=None):
        self._base = str(base)
        self._path_of, self._fov_of = path_of, fov_of
        self._levels = list(levels) if levels is not None else None   # None -> discover on first use
        self.n_levels = max(1, len(self._levels)) if self._levels else 1
        self.well_px = int(well_px)
        self.pixel_size_um = pixel_size_um
        self._written = written                # None = every well (a plate opened from disk)
        self._handles: dict[tuple, object] = {}
        self._coarse: dict[str, np.ndarray] = {}

    def mark_written(self, well_id):
        """A well just landed on disk. Availability grows DURING a run — which is exactly when
        someone is watching the plate fill and wants to glance at what already finished."""
        if self._written is not None:
            self._written.add(well_id)

    def available(self, well_id) -> tuple[bool, str]:
        if self._written is not None and well_id not in self._written:
            return False, "not written yet"
        if self._path_of(well_id) is None:
            return False, "no image for this well"
        return True, ""

    def _resolve_levels(self, well_id):
        """Read the field's multiscales once, to learn how many pyramid levels exist.

        Deferred because a run that is still writing hasn't declared its levels yet — and how
        many there are depends on the field size (_PYRAMID_MIN_YX collapses small fields to a
        single level, which is exactly what the test fixtures hit)."""
        if self._levels is not None:
            return self._levels
        field = f"{self._base}/{self._path_of(well_id)}/{self._fov_of(well_id)}"
        try:
            ome = json.loads((Path(field) / "zarr.json").read_text())["attributes"]["ome"]
            self._levels = [ds["path"] for ds in ome["multiscales"][0]["datasets"]]
        except Exception:
            self._levels = ["0"]               # a field always has a full-res array 0
        self.n_levels = max(1, len(self._levels))
        return self._levels

    def _open(self, well_id, level):
        levels = self._resolve_levels(well_id)
        level = max(0, min(int(level), len(levels) - 1))
        key = (well_id, level)
        if key not in self._handles:
            import tensorstore as ts
            path = f"{self._base}/{self._path_of(well_id)}/{self._fov_of(well_id)}/{levels[level]}"
            self._handles[key] = ts.open(
                {"driver": "zarr3", "kvstore": {"driver": "file", "path": path}}).result()
        return self._handles[key]

    def read_crop(self, well_id, level, y0, x0, h, w):
        arr = self._open(well_id, level)
        ny, nx = arr.shape[-2], arr.shape[-1]
        # Clamp the ORIGIN so the window stays whole near an edge (shift it in), rather than
        # truncating the extent — clamping y0 to ny-1 first would return a 1px sliver.
        y0, x0, h, w = loupe_clamp_crop(y0, x0, h, w, ny, nx)
        # A field below _PYRAMID_MIN_YX writes level 0 alone, so level selection cannot bound
        # this read; stride it in tensorstore so the I/O itself shrinks, not just the result.
        step = loupe_decimation(max(h, w))
        return np.asarray(
            arr[0, :, 0, y0:y0 + h:step, x0:x0 + w:step].read().result())

    def coarse(self, well_id):
        if well_id not in self._coarse:
            arr = self._open(well_id, self.n_levels - 1)          # coarsest level = cheapest
            self._coarse[well_id] = np.asarray(arr[0, :, 0].read().result())
        return self._coarse[well_id]


class _LoupeWorker(QThread):
    """Serves loupe crops off the GUI thread, coalescing to the NEWEST request.

    Only the latest cursor position matters: if the user sweeps across three wells while a read
    is in flight, the two intermediate reads are worthless. One pending slot (overwritten by
    each new request) IS the coalescing. Results carry the generation they were asked for, so a
    late arrival for a stale position is dropped by the widget rather than flashing."""

    ready = pyqtSignal(int, str, object, object, object)  # (gen, well, crop|None, window|None, err)

    def __init__(self, source: _LoupeSource):
        super().__init__()
        self._source = source
        self._cv = threading.Condition()
        self._pending = None
        self._stop = False
        self._cache: dict[tuple, np.ndarray] = {}
        self._order: list[tuple] = []

    def request(self, gen, well_id, level, y0, x0, h, w):
        with self._cv:
            self._pending = (gen, well_id, level, y0, x0, h, w)
            self._cv.notify()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

    def _cached(self, key):
        hit = self._cache.get(key)
        if hit is not None:
            self._order.remove(key)
            self._order.append(key)
        return hit

    def _store(self, key, val):
        self._cache[key] = val
        self._order.append(key)
        while len(self._order) > _LOUPE_CACHE:
            self._cache.pop(self._order.pop(0), None)

    def run(self):
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                gen, well_id, level, y0, x0, h, w = self._pending
                self._pending = None
            key = (well_id, level, y0, x0, h, w)
            try:
                crop = self._cached(key)
                if crop is None:
                    crop = self._source.read_crop(well_id, level, y0, x0, h, w)
                    self._store(key, crop)
                # The contrast window belongs on this side too: deriving it on the GUI thread
                # meant a paint-driven slot could decode a whole TIFF plane (IMA-208).
                try:
                    win = self._source.window(well_id)
                except Exception:
                    win = None                        # the widget falls back to a flat window
                self.ready.emit(gen, well_id, crop, win, None)
            except Exception as e:                    # a racing writer / deleted plate / bad path
                self.ready.emit(gen, well_id, None, None, f"{type(e).__name__}: {e}")


# --- plate overview widget (one cell per well; hue-coded status; fit-to-view) ---------------

class PlateOverview(QWidget):
    """The low-res plate: an RGB canvas of MIP tiles, a per-well status hue, a red box, and a
    press-and-hold LOUPE that overlays real acquisition pixels for the well under the cursor
    (IMA-208 — the montage itself is far too coarse to magnify; see the loupe block above).

    The RGB canvas is only what is CURRENTLY shown. What the widget actually owns (IMA-206) is a
    per-layer ``(C, nr*_CELL, nc*_CELL)`` native-dtype store — the plate with its channel axis
    still intact — plus a channel mask and a per-channel contrast window. Producers hand over
    per-channel tiles and this widget does all compositing, so toggling a channel or dragging its
    contrast re-composites from the retained pixels: no reader I/O, no re-projection, ever.
    """

    hovered = pyqtSignal(str)              # region id (or "" off-plate), for the window's readout
    wellActivated = pyqtSignal(str, int)   # (well_id, fov_index) double-clicked -> load in ndviewer
    selectionChanged = pyqtSignal(list)    # acquired well ids the operator picked (row-major)
    marqueeSelected = pyqtSignal(list)     # ...and specifically by a Shift-DRAG: opens an exploration
                                           # tab (IMA-205). Shift+CLICK refines the selection one well
                                           # at a time and deliberately does NOT fire it — otherwise
                                           # every corrective click spawns another tab.

    def __init__(self, rows, cols, wells: dict, layout: Optional[dict] = None):
        """``wells``: (row_index, col_index) -> well_id for every acquired well (drawn grey until
        processed). Tiles/status arrive as an operator runs.

        ``layout`` (IMA-253) is ``{(row_index, col_index): (x, y, w, h)}`` in GRID UNITS for a
        holder whose cells are placed by real geometry rather than by a uniform pitch -- a freeform
        tissue slide, where each region's cell is its own mosaic's bounding box. ``None`` is the
        uniform grid every well plate is, and keeps the single-blit fast path a 1536-well plate
        needs. Cells absent from the map fall back to their nominal ``(c, r, 1, 1)`` square, which
        is what an EMPTY slot (no stage coordinates to place it by) can honestly be drawn as.
        """
        super().__init__()
        self._rows, self._cols = list(rows), list(cols)
        self._layout: Optional[dict] = ({tuple(k): tuple(float(v) for v in val)
                                         for k, val in layout.items()} if layout else None)
        self._nr, self._nc = len(self._rows), len(self._cols)
        self._by_rc: dict[tuple, str] = dict(wells)            # every acquired well (for status + hit-test)
        self._status: dict[tuple, str] = {rc: "empty" for rc in wells}
        self._tiles: set[tuple] = set()                        # cells that have a tile painted (any layer)
        self._tiles_by_layer: dict[str, set] = {}              # layer -> cells with an image there
        self._canvas = QImage(self._nc * _CELL, self._nr * _CELL, QImage.Format_RGB888)
        self._canvas.fill(QColor(_BG))
        self._final = None            # global-contrast recomposite of the ACTIVE layer (or None)
        # Layer stack render: the base ("raw") is self._canvas; each operator draws into its own
        # per-layer canvas/final. self._active is the layer the plate currently shows (LayersTab picks
        # it via set_active_layer). Keeps memory to one small montage-canvas per layer used.
        self._op_canvas: dict[str, QImage] = {}
        self._op_final: dict[str, QImage] = {}
        self._final_arr: dict[str, np.ndarray] = {}   # keeps each recomposited RGB alive: QImage
        #                                               WRAPS the numpy buffer, it does not copy it
        self._active = "raw"
        # --- the channel axis (IMA-206) — set_channels declares it; empty until then -----------
        self._labels: list[str] = []      # channel display names, for the channel bar
        self._colors = None               # (C, 3) float RGB, the RESOLVED display_color per channel
        self._dtype = np.uint16           # store dtype: the acquisition's native dtype (half the RAM)
        self._store: dict[str, np.ndarray] = {}   # layer -> (C, nr*_CELL, nc*_CELL), allocated lazily
        # --- what a contrast change is allowed to touch (IMA-261) ------------------------------
        # A contrast window is a POINT transform, so it commutes with subsampling: windowing the
        # display-sized thumbnail is bit-identical to windowing the whole plate and subsampling
        # afterwards (see squidmip._montage._window_lut). The only thing that must be re-derived
        # per tick is therefore the composite of the DISPLAY-SIZED buffer. These two caches hold
        # everything upstream of that, so a drag re-reads no store and re-percentiles nothing:
        self._disp: dict[str, tuple] = {}      # layer -> (step, contiguous (C, h, w) thumbnail)
        self._cell_auto: dict[str, dict] = {}  # layer -> {(ri, ci): [per-channel AUTO window]}
        #                                        SCOPE_PER_REGION's exact percentiles, which depend
        #                                        on the PIXELS only — never on the contrast.
        self._mask = None                 # (C,) bool: which channels composite into the plate
        self._contrast = None             # _RunningContrast: global per-channel window + auto/manual
        #                                   re-composites from the store above — it never re-runs.
        self._full = QTimer(self)         # coalesces the full-res recomposite behind a gesture
        self._full.setSingleShot(True)    # (a drag repaints at DISPLAY res; full-res lands once)
        self._full.timeout.connect(self._on_full_timeout)
        self._scaled = None           # cached pixmap of (final|canvas) scaled to the current zoom;
        self._scaled_cd = None        # rebuilt only when zoom (cd) or the source image changes — so
        #                               a hover/pan repaint blits 1:1 instead of re-resampling 12 MP.
        self._cd = float(_CELL)       # displayed px/well (fit baseline, then wheel-zoomed)
        self._ox = self._oy = _PAD    # top-left of the plate within the widget (pan-able)
        self._hover = None
        self._sel = None              # well selected from the ndviewer FOV slider
        # SELECTION (IMA-221) is a DIFFERENT concept from _sel above: _sel is "the one well the
        # detail viewer is showing" (red box, driven by the FOV slider); _selection is "the set the
        # operator picked" (tint, driven by Shift-gestures). Never merge them — the red box must
        # survive selecting, and selecting must survive scrubbing.
        self._selection: set = set()  # acquired (row_index, col_index) the user picked. A SET:
        #                               paintEvent membership-tests it once per cell, 1536x on a 1536wp.
        self._view_hues: list = []    # [(rc_set, QColor)] per open view — plate colour-codes threads
        self._marquee = None          # (x0, y0, x1, y1) widget px while a Shift-drag is in flight
        self._marquee_add = False     # this drag unions (Shift+Alt) rather than replaces
        self._ctrl_click = None       # (x, y) of a Cmd/Ctrl-press, committed as a TOGGLE on release
        self._press = None            # (x, y, ox, oy) at left-press, for drag-to-pan
        self._panning = False
        self._user_view = False       # True once the user wheel-zooms/pans (stop auto-fitting)
        self._boxes: dict = {}        # (region, fov) -> (top, left, h, w) in cell px; {} = single-FOV
        self._boxed_regions: set = set()   # regions whose cell holds a LETTERBOXED mosaic, not one tile
        # -- carrier geometry (IMA-220, redrawn for IMA-253: geometry, not a photograph) --
        self._carrier = None          # the _plate.PlateGeometry to draw the holder outline from
        self._carrier_slide = False   # slot-shaped cells (a slide carrier) vs round wells
        self._slides = None           # [(x, y, w, h), ...] in GRID UNITS: real glass slides drawn
        #                               behind a tissue acquisition (IMA-265, _slide_art). None on
        #                               a well plate and on a carrier with no stage coordinates.
        self._tile_rgn = None         # cached QRegion of cells that HAVE an image, at pan origin
        self._tile_rgn_key = None     # (cd, active layer, n tiled cells) the cached region was built for
        # -- loupe (IMA-208) --
        self._loupe_src = None        # _LoupeSource for the ACTIVE layer, or None (loupe disabled)
        self._loupe_worker = None
        self._loupe = None            # armed/live state dict, or None when idle
        self._loupe_gen = 0           # bumped per request; late results for older gens are dropped
        self._loupe_img = None        # QImage currently shown in the inset
        self._loupe_note = ""         # user-visible reason when the loupe can't show pixels
        self._loupe_win = {}          # well_id -> per-channel window, mirroring the tile's rule
        self._loupe_colors = None     # (C, 3) float RGB, set with the source
        self._hold = QTimer(self)
        self._hold.setSingleShot(True)
        self._hold.setInterval(_LOUPE_HOLD_MS)
        self._hold.timeout.connect(self._arm_loupe)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)   # so focusOutEvent can actually fire (see below)
        self.setMinimumSize(240, 200)

    # -- loupe wiring --
    def set_loupe_source(self, source, colors=None):
        """Point the loupe at the data behind the ACTIVE layer. ``None`` disables the gesture.

        Called whenever what the plate is showing changes identity — a new acquisition, an
        operator run persisting, a layer switch, a preview superseding a saved run. Re-pointing
        is what stops a stale run's pixels appearing under a newer run's tiles."""
        self._dismiss_loupe()
        if self._loupe_worker is not None:
            self._loupe_worker.stop()
            self._loupe_worker.wait(2000)
            self._loupe_worker = None
        self._loupe_src = source
        self._loupe_colors = colors
        self._loupe_win.clear()
        if source is not None:
            self._loupe_worker = _LoupeWorker(source)
            self._loupe_worker.ready.connect(self._on_loupe_crop)
            self._loupe_worker.start()

    def _arm_loupe(self):
        """Hold timer fired: the press became a loupe. Only reachable while still ARMED."""
        self._hold.stop()             # a pending fire must never re-arm and blank a LIVE loupe:
                                      # _arm_loupe clears _loupe_img, so a second arm 350 ms after
                                      # the first shows an empty inset until the re-read lands.
        if self._press is None or self._panning:
            return
        x, y = self._press[0], self._press[1]
        c = self._cell(x, y)
        if not c or not c["well_id"]:
            return
        self._loupe = {"well": c["well_id"], "x": x, "y": y}
        self._loupe_img, self._loupe_note = None, ""
        self._request_loupe(x, y)
        self.update()

    def _dismiss_loupe(self):
        if self._loupe is not None or self._loupe_img is not None or self._loupe_note:
            self._loupe = self._loupe_img = None
            self._loupe_note = ""
            self.update()

    def _loupe_geometry(self, x, y):
        """Map a widget point to (well_id, level, crop rect, s_loupe, M) — or None if off-plate."""
        src = self._loupe_src
        c = self._cell(x, y)
        if src is None or not c or not c["well_id"]:
            return None
        s_loupe, mag = loupe_scale(self._cd, src.well_px)
        level = loupe_level(s_loupe, src.n_levels)
        crop = loupe_crop_px(s_loupe, level)
        # cursor -> position within the cell -> image px at level 0 -> image px at ``level``
        ax, ay = self._ox + _HDR, self._oy + _COLH
        fy = (y - (ay + c["row_index"] * self._cd)) / max(1e-9, self._cd)
        fx = (x - (ax + c["col_index"] * self._cd)) / max(1e-9, self._cd)
        span = max(1, src.well_px >> level)
        cy, cx = int(fy * span), int(fx * span)
        # Clamp HERE as well as in the source. Two reasons beyond belt-and-braces: the request
        # that reaches the worker is then always a rectangle that exists, and a hold near a field
        # edge produces the SAME key as the cursor drifts, so the LRU hits instead of decoding a
        # fresh full-field crop per pixel of motion.
        y0, x0, h, w = loupe_clamp_crop(cy - crop // 2, cx - crop // 2, crop, crop, span, span)
        return c["well_id"], level, (y0, x0, h, w), s_loupe, mag

    def _request_loupe(self, x, y):
        geo = self._loupe_geometry(x, y)
        if geo is None:                    # dragged onto the margin / an un-acquired cell
            if self._loupe_img is not None or not self._loupe_note:
                self._loupe_img, self._loupe_note = None, "no well here"
                self.update()
            return
        if self._loupe_worker is None:
            return
        well, level, (y0, x0, h, w), _s, _m = geo
        ok, why = self._loupe_src.available(well)
        if not ok:
            self._loupe_img, self._loupe_note = None, why
            self.update()
            return
        self._loupe_gen += 1
        self._loupe_worker.request(self._loupe_gen, well, level, y0, x0, h, w)

    def _on_loupe_crop(self, gen, well_id, crop, window, error):
        """A crop landed. Drop it unless it is the newest request and the loupe is still up.

        Everything expensive already happened on the worker thread: this slot only windows and
        colours a <= _LOUPE_MAX_CROP square. It must stay that way — it runs inside the paint
        loop of a widget the user is dragging across."""
        if gen != self._loupe_gen or self._loupe is None:
            return
        if error is not None or crop is None or crop.size == 0:
            self._loupe_img, self._loupe_note = None, error or "no pixels here"
            self.update()
            return
        # Mirror the TILE's contrast rule on the WELL's pixels (computed by the source, per well)
        # — never percentiles of the crop under the cursor, which would make brightness lurch as
        # the cursor moves and make the inset look like different data.
        #
        # That AUTO window is then resolved through the plate's one contrast model (IMA-242), so a
        # channel the user latched with the slider shows the user's window here too. Before, the
        # loupe kept its own memo and the inset went on displaying the pre-drag contrast — two
        # representations of one truth, never synced.
        auto = window if window is not None else self._loupe_win.get(well_id)
        if auto is None:
            auto = [(0.0, 1.0)] * crop.shape[0]
        self._loupe_win[well_id] = auto              # memo the AUTO window, never the resolved one
        win = ([self._contrast.resolve(c, auto[c]) for c in range(len(auto))]
               if self._contrast is not None else list(auto))
        colors = self._loupe_colors
        if colors is None:
            colors = np.ones((crop.shape[0], 3), np.float32)
        # The same compositor the plate uses, with the same channel mask: unticking a channel must
        # remove it from the inset as well, or the loupe contradicts the plate it sits on top of.
        planes = np.stack([crop[c].astype(np.float32) for c in range(crop.shape[0])])
        mask = self._mask if (self._mask is not None
                              and len(self._mask) == crop.shape[0]) else None
        rgb = composite(planes, colors, win, mask)
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()   # copy: rgb is transient
        self._loupe_img, self._loupe_note = img, ""
        self.update()

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

    def _canvas_for(self, layer: str) -> QImage:
        if layer == "raw":
            return self._canvas
        cv = self._op_canvas.get(layer)
        if cv is None:
            cv = QImage(self._nc * _CELL, self._nr * _CELL, QImage.Format_RGB888)
            cv.fill(QColor(_BG))
            self._op_canvas[layer] = cv
        return cv

    def _active_source(self) -> QImage:
        return self._final or self._canvas_for(self._active)

    # -- the channel axis: store, mask, per-channel contrast (IMA-206) --
    def set_channels(self, labels, colors: np.ndarray, dtype=np.uint16, pct=(1.0, 99.8)):
        """Declare the acquisition's channels — the per-channel store/mask/contrast start here.

        *colors* is the (C, 3) float RGB of the RESOLVED ``display_color`` (the acquisition's YAML
        first, the wavelength fallback map second — resolve_channels already settled that), so the
        plate is tinted exactly the way every other compositing site tints it.
        """
        self._labels = [str(x) for x in labels]
        self._colors = np.asarray(colors, dtype=np.float32)
        self._dtype = np.dtype(dtype)
        self._mask = np.ones(len(self._labels), dtype=bool)     # every channel on by default (OV8)
        dmax = float(np.iinfo(self._dtype).max) if self._dtype.kind in "ui" else 1.0
        self._contrast = _RunningContrast(len(self._labels), dmax, pct=pct)
        self._store.clear()

    def _store_for(self, layer: str) -> Optional[np.ndarray]:
        """The layer's (C, H, W) plate store, allocated on first tile (one per layer, lazily —
        each layer that supports toggling costs its own ~95 MB at 1536wp x 4ch uint16)."""
        if self._colors is None:
            return None
        st = self._store.get(layer)
        if st is None:
            st = np.zeros((len(self._colors), self._nr * _CELL, self._nc * _CELL), self._dtype)
            self._store[layer] = st
        return st

    def channel_windows(self) -> list:
        """The effective (lo, hi) per channel — the latched manual window, else the running one."""
        return self._contrast.windows() if self._contrast is not None else []

    def _invalidate_pixels(self, layer: str, rc=None):
        """The layer's STORE changed. Drop everything derived from its pixels.

        Called from the one place that writes pixels (``add_tile``) and from ``reset_layer``.
        These caches are keyed on PIXELS alone, so a contrast change must never come through
        here — keeping the two invalidations apart is what lets a drag re-read nothing.

        *rc* narrows the per-cell percentile drop to the one cell that was written; the display
        thumbnail is a whole-plate copy, so it always goes.
        """
        self._disp.pop(layer, None)
        if rc is None:
            self._cell_auto.pop(layer, None)
        else:
            self._cell_auto.get(layer, {}).pop(rc, None)

    def _disp_store(self, layer: str, store: np.ndarray, step: int) -> np.ndarray:
        """A C-CONTIGUOUS (C, h, w) thumbnail of *store* at subsampling *step*, cached.

        This is the "precompute once at ingest, remap cheaply per tick" half of IMA-261. Building
        it costs one strided copy of the whole store (~2.6 ms on a 1536-well plate); it is
        rebuilt only when the pixels change or the zoom changes, so a continuous contrast drag
        pays for it ZERO times and every tick composites straight out of it.

        Contiguity is not a detail: ``store[:, ::4, ::4]`` is a strided VIEW, and both the table
        lookup and the BLAS reduce in ``composite`` would silently materialise their own copy of
        it on every single tick.
        """
        if step <= 1:
            return store          # already the thing itself; caching it would just double the RAM
        hit = self._disp.get(layer)
        if hit is not None and hit[0] == step:
            return hit[1]
        thumb = np.ascontiguousarray(store[:, ::step, ::step])
        self._disp[layer] = (step, thumb)
        return thumb

    # PER-REGION CONTRAST IS GONE, AND THAT IS THE POINT.
    #
    # `_cell_auto_windows`, `_cell_windows` and `_composite_per_region` lived here: one contrast
    # window per WELL, each cell stretched to its own percentiles. Julio: "the contrast should be
    # only global, I don't understand why there's a per region contrast... I don't think that
    # there's any scientific basis." He is right. It makes a dim well readable next to a bright
    # one, which is a presentation trick, and it costs the one thing a plate view exists for:
    # two wells that look identical may differ by orders of magnitude, so the plate can no longer
    # be read as relative signal. The amber "wells NOT comparable" badge was an admission that
    # the picture was misleading, printed on top of the misleading picture.
    #
    # Deleting it also removes the `follow=False` branch, which is why napari's contrast did not
    # reach the plate: per-region DELIBERATELY ignored the owning viewer's window. There is now
    # one window per channel, owned by napari, and the plate follows it. One owner, one value.

    def set_channel_color(self, ch: int, rgb) -> bool:
        """Re-tint one channel to the colour the CENTRE VIEWER is using, and repaint.

        The plate keeps a ``(C, 3)`` LUT table resolved once from the acquisition's
        ``display_color``. Left alone it is a second, stale answer to "what colour is this
        channel", and recolouring a layer in napari made the two panes disagree about the same
        channel. This is the sink half: napari decides, the plate follows, and nothing is re-read
        -- the composite is rebuilt from the native-dtype tiles already retained.
        """
        if self._colors is None or not (0 <= ch < len(self._colors)):
            return False
        new_rgb = np.asarray(rgb, dtype=np.float32)
        if np.allclose(self._colors[ch], new_rgb):
            return False
        self._colors[ch] = new_rgb
        self._refresh()
        return True

    def set_channel_visible(self, ch: int, on: bool):
        """Toggle a channel in/out of the plate composite. Recomposites from the RETAINED store —
        no reader I/O, no re-projection (that is the whole point of keeping the channel axis)."""
        if self._mask is None or not (0 <= ch < len(self._mask)):
            return
        self._mask[ch] = bool(on)
        self._refresh()

    def set_channel_window(self, ch: int, lo: float, hi: float):
        """Re-window one channel and repaint. LATCHES that channel manual (D4) so the wells still
        streaming in can't stomp the window the user just set."""
        if self._contrast is None or not (0 <= ch < len(self._mask)):
            return
        self._contrast.set_manual(ch, lo, hi)
        self._refresh()

    def follow_channel_window(self, ch: int, lo: float, hi: float):
        """Render *ch* with the window the OWNING ARRAY VIEWER resolved, and repaint (IMA-261).

        The sink half of the one-owner contract. It does NOT latch the channel manual: ndv
        autoscales on its own, at open and on every data change, so recording that as a user
        gesture would kill the plate's own auto-contrast before the user had touched anything,
        and would outrank the per-region scope the user explicitly selected. See
        ``_RunningContrast.set_followed``.
        """
        if self._contrast is None or not (0 <= ch < len(self._mask)):
            return
        self._contrast.set_followed(ch, lo, hi)
        self._refresh()

    def set_channel_auto(self, ch: int):
        """Unlatch a channel: it goes back to auto-scaling off the running histogram."""
        if self._contrast is None or not (0 <= ch < len(self._mask)):
            return
        self._contrast.set_auto(ch)
        self._refresh()

    def _refresh(self):
        """A user gesture: repaint NOW at display resolution, then land full-res once it settles.

        The invariant is that no gesture ever touches the full (C, 2816, 4224) canvas — a slider
        drag composites the sub-sampled view the screen can actually show (a few thousand px),
        and the single full-res pass is coalesced behind the last event.
        """
        self.recomposite(quick=True)
        self._full.start(150)
        self._refresh_loupe()

    def _refresh_loupe(self):
        """Re-render the loupe inset under the contrast that just changed (IMA-242).

        The inset holds a rendered QImage, so repainting the plate alone would leave it showing
        the PRE-drag contrast until the cursor happened to move — the plate and the magnifier of
        the plate disagreeing about the same pixels. Re-issuing the request is cheap: the worker
        memoises crops, so this re-colours the bytes it already has and re-reads nothing.
        """
        if self._loupe is None or self._loupe_worker is None:
            return
        try:
            self._request_loupe(self._loupe["x"], self._loupe["y"])
        except Exception:
            pass          # a contrast drag must never fail because the loupe could not re-render

    def _on_full_timeout(self):
        """The coalescing timer fired. Guarded: a pending recomposite must not outlive the widget
        (the plate is torn down and rebuilt on every open, and the timer is queued, not immediate)."""
        try:
            self.recomposite(quick=False)
        except RuntimeError:
            pass   # the C++ widget went away while the full-res pass was still pending

    def recomposite(self, layer: Optional[str] = None, *, quick: bool = False):
        """Rebuild a layer's plate image from its store, at the current mask + windows.

        ``quick=True`` composites a strided view at roughly the on-screen resolution (cheap enough
        to run on every slider tick); the default walks the whole store — the end-of-stream pass.
        """
        layer = layer or self._active
        store = self._store.get(layer)
        if store is None or self._colors is None:
            return
        step = max(1, int(round(_CELL / max(1.0, self._cd)))) if quick else 1
        rgb = composite(self._disp_store(layer, store, step), self._colors,
                        self.channel_windows(), self._mask)
        self._final_arr[layer] = rgb          # hold the buffer: the QImage below only wraps it
        h, w, _ = rgb.shape
        self.set_final(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888), layer)

    def hideEvent(self, e):
        self._full.stop()   # a plate on its way out must not repaint from a queued timer
        super().hideEvent(e)

    def reset_layer(self, layer: str):
        """Forget a layer's retained pixels — its store, its recomposite and its painted canvas.

        Called before a run streams into a layer that already has one. Without it a mosaic re-run
        that lands FEWER fields (a smaller selection, a failed well) would leave the previous run's
        FOVs standing inside the same cell, blended into the new ones with no way to tell which is
        which — and the ~95 MB store would never be freed either.
        """
        self._store.pop(layer, None)
        self._invalidate_pixels(layer)
        self._final_arr.pop(layer, None)
        self._op_final.pop(layer, None)
        self._tiles_by_layer.pop(layer, None)
        if layer == "raw":
            self._canvas.fill(QColor(_BG))
        else:
            self._op_canvas.pop(layer, None)
        if layer == self._active:
            self._final = None
            self._scaled = None
            self.update()

    # -- data in --
    def add_tile(self, ri: int, ci: int, well_id: str, tile: np.ndarray, layer: str = "raw",
                 box=None):
        """Take one PER-CHANNEL tile ``(C, h, w)`` (native dtype), retain it in the layer's store,
        feed the running contrast, and re-composite that whole cell.

        ``box`` is ``(top, left, h, w)`` in cell px for a multi-FOV mosaic (IMA-187): the tile is
        one FIELD inside the cell, so it is written at that offset and the cell is re-composited
        around it, which is what makes the seams between neighbouring FOVs update as they land.
        ``box=None`` is the historical single-tile path — one field fills the cell at (0, 0).

        CONTRAST IS FED THE TILE, NEVER THE STORE SLICE. A mosaic cell is zero-padded wherever no
        FOV lands (margins, gaps between fields); feeding those zeros to the running histogram
        would pin the 1st percentile at 0 for the whole plate and silently wash every well out.
        Only real acquired pixels get a vote.
        """
        if (ri, ci) not in self._by_rc:    # ignore a stale tile from a retired run / foreign cell
            return
        store = self._store_for(layer)
        if store is None:                  # no channel axis declared yet -> nothing to composite
            return
        tile = np.asarray(tile)
        y0, x0 = ri * _CELL, ci * _CELL
        top, left = (int(box[0]), int(box[1])) if box is not None else (0, 0)
        th, tw = tile.shape[1], tile.shape[2]      # place by ACTUAL shape: a field smaller than
        store[:, y0 + top:y0 + top + th,           # the cell must not broadcast-crash
              x0 + left:x0 + left + tw] = tile
        self._invalidate_pixels(layer, (ri, ci))   # these pixels are new: nothing derived survives
        for c_i in range(tile.shape[0]):
            self._contrast.add(c_i, tile[c_i])     # real FOV pixels only — see the docstring
        wins = self.channel_windows()     # ONE window per channel, owned by napari (see above)
        cell = composite(store[:, y0:y0 + _CELL, x0:x0 + _CELL], self._colors, wins, self._mask)
        img = QImage(cell.data, _CELL, _CELL, 3 * _CELL, QImage.Format_RGB888)
        p = QPainter(self._canvas_for(layer))
        p.drawImage(x0, y0, img)           # drawImage COPIES, so `cell` may die after p.end()
        p.end()
        if self._op_final.pop(layer, None) is not None:   # a new tile supersedes the old recomposite
            self._final_arr.pop(layer, None)              # -> back to the streamed canvas
            if layer == self._active:
                self._final = None
        self._tiles.add((ri, ci))
        self._tiles_by_layer.setdefault(layer, set()).add((ri, ci))   # per-layer: drives the grey dots
        if layer == self._active:         # only the shown layer needs a repaint / cache rebuild
            self._scaled = None
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

    def set_final(self, img: QImage, layer: str = "raw"):
        self._op_final[layer] = img
        if layer == self._active:
            self._final = img
            self._scaled = None       # source changed -> the scaled cache is stale
            self.update()

    def set_active_layer(self, layer: str):
        """Show a layer (LayersTab toggle/reorder). Swaps in its montage + streamed canvas."""
        self._active = layer
        self._final = self._op_final.get(layer)   # None for "raw" -> falls back to the base canvas
        self._scaled = None
        self.update()
        if layer in self._store:     # bring it up to the CURRENT mask/windows: its canvas was blitted
            self.recomposite(layer)  # cell-by-cell, with whatever mask was set when each tile landed

    def drop_layer(self, layer: str):
        """Forget a layer entirely and FREE its canvas (IMA-205: an exploration tab's layers die
        with the tab). ``_canvas_for`` lazily allocates a full plate-sized RGB888 image per layer
        (nc*_CELL x nr*_CELL — tens of MB on a 1536wp), so without this a closed tab's montage
        stays resident forever. Falls back to the base layer if the dropped one was showing.

        The per-channel STORE (IMA-206) is the bigger half — ~95 MB of retained (C, H, W) pixels
        per layer — so it goes too. Dropping the canvas while the store survived would look like a
        fix and leak the majority of the memory."""
        if layer == "raw":
            return
        self._op_canvas.pop(layer, None)
        self._op_final.pop(layer, None)
        self._store.pop(layer, None)
        self._invalidate_pixels(layer)
        self._final_arr.pop(layer, None)
        self._tiles_by_layer.pop(layer, None)
        self._tiles = set().union(*self._tiles_by_layer.values()) if self._tiles_by_layer else set()
        if self._active == layer:
            self.set_active_layer("raw")
        else:
            self.update()

    def status_snapshot(self) -> dict:
        """Copy of the per-well status map — the window saves one per exploration tab so a tab's
        amber/failed dots follow its own run, not whatever ran last (IMA-205)."""
        return dict(self._status)

    def set_status_map(self, status: dict):
        """Restore a snapshot. Foreign keys are ignored, same as ``set_status``."""
        for rc, state in status.items():
            if rc in self._status:
                self._status[rc] = state
        self.update()

    def select(self, ri: int, ci: int):
        """Move the red box to a well (driven by the ndviewer FOV slider)."""
        self._sel = (ri, ci)
        self.update()

    def resizeEvent(self, e):
        self._user_view = False       # a resize re-fits (drop any zoom/pan)
        self._fit()
        self.update()

    # -- mouse: wheel-zoom anchored at cursor, left-drag pan (Hongquan's navigator gestures),
    #    and press-and-hold loupe (IMA-208). The left button now means three different things
    #    depending on TIMING, so the rules live here as a diagram rather than as scattered flags:
    #
    #                        ┌───────────────────────────────────────────┐
    #                        │                  IDLE                     │
    #                        └──────────────────┬────────────────────────┘
    #          left-press on an acquired cell   │   (off-plate / empty: never arms)
    #                                           ▼
    #                        ┌───────────────────────────────────────────┐
    #                        │  ARMED   _hold running (_LOUPE_HOLD_MS)   │
    #                        │  cursor must stay within _LOUPE_SLOP px   │
    #                        └───┬───────────────────────┬───────────────┘
    #          move > slop       │                       │  timer fires
    #          (kill the timer)  │                       │
    #                            ▼                       ▼
    #                  ┌──────────────────┐   ┌──────────────────────────┐
    #                  │       PAN        │   │          LOUPE           │
    #                  │  drag the plate  │   │  inset follows cursor;   │
    #                  │  (unchanged)     │   │  pan is DEAD while up;   │
    #                  └────────┬─────────┘   │  hover + wheel suppressed│
    #                           │             └────────────┬─────────────┘
    #                           │ release                  │ release / dragged off the widget
    #                           │                          │ / leave / focus-out
    #                           ▼                          ▼
    #                        ┌───────────────────────────────────────────┐
    #                        │                  IDLE                     │
    #                        └───────────────────────────────────────────┘
    #
    #    Two edges worth stating because they are easy to regress:
    #      * SLOW PAN stays a pan. Press, dwell past the timer, then drag — the timer only runs
    #        while the cursor is still, and any move past the slop kills it. A press that has
    #        already become a loupe is dismissed on release, so the next drag pans normally.
    #      * DOUBLE-CLICK must cancel the timer. Qt delivers press/release/dblclick/release, and
    #        the second press re-arms; without the cancel you would open the detail viewer AND
    #        raise a loupe from one gesture.
    def _cell(self, x, y):
        if self._layout is not None:
            return self._freeform_cell(x, y)
        px, py = x - (self._ox + _HDR), y - (self._oy + _COLH)
        return well_at(self._rows, self._cols, self._by_rc, px, py, self._cd)

    def _freeform_cell(self, x, y):
        """Hit-test a geometrically placed holder: the first cell whose own rect contains (x, y).

        Placed cells are tested FIRST and in reverse paint order, so a click in the small area
        where two regions' bounding boxes overlap resolves to the one drawn on top — the same
        last-one-wins rule ``_fov_at`` uses inside a mosaic. Nominal (empty-slot) rects are only
        consulted when no real region claims the point, so an empty slot can never shadow a region
        that overlaps it. Freeform holders have a handful of cells, so a linear scan is free.
        """
        placed = [rc for rc in self._by_rc if rc in self._layout]
        for rc in list(reversed(placed)) + [rc for rc in self._by_rc if rc not in self._layout]:
            rx, ry, rw, rh = self._cell_rect(*rc)
            if rx <= x < rx + rw and ry <= y < ry + rh:
                ri, ci = rc
                return {"row_index": ri, "col_index": ci, "row": self._rows[ri],
                        "col": self._cols[ci], "well_id": self._by_rc.get(rc)}
        return None

    def _cells_in(self, x0, y0, x1, y1) -> list:
        """Widget px -> acquired cells, via the pure helper (same margin removal as _cell)."""
        if self._layout is not None:
            lo_x, hi_x = min(x0, x1), max(x0, x1)
            lo_y, hi_y = min(y0, y1), max(y0, y1)
            hits = []
            for rc in self._by_rc:
                rx, ry, rw, rh = self._cell_rect(*rc)
                if rx < hi_x and rx + rw > lo_x and ry < hi_y and ry + rh > lo_y:
                    hits.append(rc)
            return sorted(hits)
        ox, oy = self._ox + _HDR, self._oy + _COLH
        return cells_in_rect(self._rows, self._cols, self._by_rc,
                             x0 - ox, y0 - oy, x1 - ox, y1 - oy, self._cd)

    # -- selection API (IMA-221) --
    def selected_wells(self) -> list:
        """The selection as acquired well ids, in plate row-major order."""
        return [self._by_rc[rc] for rc in sorted(self._selection)]

    def clear_selection(self):
        """Drop the whole selection and tell listeners (used on re-ingest)."""
        if self._selection:
            self._selection = set()
            self.selectionChanged.emit([])
            self.update()

    def select_all(self):
        """Select every occupied well (the Select all button and Cmd/Ctrl-A)."""
        self._selection = set(self._by_rc.keys())
        self.selectionChanged.emit(self.selected_wells())
        self.update()

    def highlight_regions(self, region_ids):
        """Move the blue wash onto *region_ids* — used when the user clicks an OPEN VIEW so the
        plate shows which regions that window holds. Same wash the manual selection uses."""
        want = set(region_ids or [])
        self._selection = {rc for rc, rid in self._by_rc.items() if rid in want}
        self.selectionChanged.emit(self.selected_wells())
        self.update()

    def set_view_hues(self, entries):
        """Colour-code the OPEN VIEWS on the plate: *entries* is a list of ``(region_ids, QColor)``,
        one per open window/thread. Each view's wells get that view's hue, so overlapping/adjacent
        views are told apart at a glance (Julio's "hue the different view threads"). Painted UNDER
        the blue focus/selection wash, which still marks the one active view. Empty list clears it."""
        hues = []
        for region_ids, color in (entries or []):
            rcs = {rc for rc, rid in self._by_rc.items() if rid in set(region_ids or [])}
            if rcs:
                hues.append((rcs, color))
        self._view_hues = hues
        self.update()

    def wheelEvent(self, e):
        if self._marquee is not None:
            return          # a marquee owns the drag; zooming would slide the plate under the rect
        if self._loupe is not None:      # zooming the plate under a live loupe would fight it
            return
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
        if e.button() != Qt.LeftButton:
            return
        # SHIFT owns every selection gesture (IMA-221). Keeping selection off the plain click is
        # what makes double-click safe: Qt delivers press+release BEFORE mouseDoubleClickEvent, so
        # a plain-click toggle would silently flip a well every time you opened one. (Ctrl is out:
        # on macOS Ctrl+click is right-click and Qt maps Cmd -> ControlModifier.)
        if e.modifiers() & Qt.ShiftModifier:
            self._marquee = (e.x(), e.y(), e.x(), e.y())
            self._marquee_add = bool(e.modifiers() & Qt.AltModifier)   # Shift+Alt = union
            self._press = None                                          # ...so this drag never pans
            self._panning = False
            self.update()
            return          # a Shift-drag is a selection, never a pan and never a loupe
        # Cmd/Ctrl-click = TOGGLE this well in the batch selection (Linux-file-manager add/remove).
        # On macOS Cmd maps to ControlModifier, and a real Ctrl+click is a right-click (not
        # LeftButton), so this only ever fires for the intended gesture. Committed on RELEASE so a
        # cmd-drag can still not-select if the user changes their mind, and so it never pans.
        if e.modifiers() & Qt.ControlModifier:
            self._ctrl_click = (e.x(), e.y())
            self._press = None
            self._panning = False
            return
        self._press = (e.x(), e.y(), self._ox, self._oy)
        self._panning = False
        c = self._cell(e.x(), e.y())
        if self._loupe_src is not None and c and c["well_id"]:   # ARM (never off-plate/empty)
            self._hold.start()

    def mouseMoveEvent(self, e):
        if self._loupe is not None:                  # LOUPE: the inset tracks; panning is dead
            # Drag off the widget and the loupe must go. leaveEvent CANNOT do this: Qt grabs the
            # mouse for the duration of a press, so no leave is delivered until the button comes
            # up — the inset used to stay pinned over the neighbouring pane showing stale pixels.
            # The grab is also why this works: move events keep arriving, with coordinates
            # outside rect(), which is the signal.
            if not self.rect().contains(e.x(), e.y()):
                self._hold.stop()
                self._dismiss_loupe()
                return
            self._loupe["x"], self._loupe["y"] = e.x(), e.y()
            self._request_loupe(e.x(), e.y())
            self.update()
            return
        if self._marquee is not None and (e.buttons() & Qt.LeftButton):
            x0, y0, _, _ = self._marquee          # grow the rubber band; emit NOTHING until release
            self._marquee = (x0, y0, e.x(), e.y())
            self.update()
            return
        if self._press is not None and (e.buttons() & Qt.LeftButton):
            dx, dy = e.x() - self._press[0], e.y() - self._press[1]
            if abs(dx) + abs(dy) > 3:
                self._panning = True
                self._hold.stop()                    # moved -> this press is a pan, not a hold
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
        if c and c["well_id"]:
            enc = display_well_id(c["well_id"])
            text = c["well_id"] if enc == c["well_id"] else f'{c["well_id"]} ({enc})'
        elif c:
            text = c["row"] + c["col"] + "  ·  empty"
        else:
            text = ""
        self.hovered.emit(text)
        self.update()

    def mouseReleaseEvent(self, e):
        self._hold.stop()
        had_loupe = self._loupe is not None
        # Only the LEFT release commits a selection. The gesture is opened by a left press, but Qt
        # delivers a release for whichever button went up — so a right-click while a Shift-drag is
        # in flight would otherwise silently toggle/replace the selection.
        if self._marquee is not None and e.button() == Qt.LeftButton:
            x0, y0, x1, y1 = self._marquee
            add, self._marquee, self._marquee_add = self._marquee_add, None, False
            dragged = abs(x1 - x0) + abs(y1 - y0) > _CLICK_SLOP
            if not dragged:                                     # Shift+CLICK -> toggle ONE well
                hit = self._cell(x1, y1)
                if hit and hit["well_id"]:
                    self._selection ^= {(hit["row_index"], hit["col_index"])}
                self.selectionChanged.emit(self.selected_wells())
            else:
                # Shift-DRAG opens a WINDOW over the boxed regions (the meeting's "shift-drag a box
                # -> a floating view"). It does NOT leave a persistent wash on the plate: you see
                # that set in the new window's region slider, so a lingering highlight is just the
                # "stays selected forever" clutter Julio flagged. Emit the window request, then
                # clear the wash. Shift+Alt still UNIONS into the batch selection instead of opening.
                boxed = [self._by_rc[rc] for rc in sorted(set(self._cells_in(x0, y0, x1, y1)))]
                if add:
                    self._selection |= set(self._cells_in(x0, y0, x1, y1))
                    self.selectionChanged.emit(self.selected_wells())
                else:
                    self.marqueeSelected.emit(boxed)            # open a window over the box
                    if self._selection:                         # drop any lingering batch wash
                        self._selection = set()
                        self.selectionChanged.emit([])
            self.update()
            self._press = None
            self._panning = False
            self._dismiss_loupe()
            return
        # Cmd/Ctrl-click TOGGLE (Linux-style add/remove to the batch selection).
        if self._ctrl_click is not None and e.button() == Qt.LeftButton:
            px, py, self._ctrl_click = *self._ctrl_click, None
            hit = self._cell(px, py)
            if hit and hit["well_id"]:
                self._selection ^= {(hit["row_index"], hit["col_index"])}
                self.selectionChanged.emit(self.selected_wells())
                self.update()
            self._press = None
            self._panning = False
            self._dismiss_loupe()
            return
        # Plain CLICK (no modifier, no pan, no loupe) = select ONLY this well, or clear on empty.
        # This is the deselect path that was missing: without it a batch selection could never be
        # dropped by clicking, so it "stayed selected forever". A plain DRAG still pans (guarded by
        # _panning), and a hold that raised the loupe does not select (had_loupe).
        if (self._press is not None and not self._panning and not had_loupe
                and e.button() == Qt.LeftButton):
            hit = self._cell(e.x(), e.y())
            new_sel = {(hit["row_index"], hit["col_index"])} if hit and hit["well_id"] else set()
            if new_sel != self._selection:
                self._selection = new_sel
                self.selectionChanged.emit(self.selected_wells())
                self.update()
        self._press = None
        self._panning = False
        self._dismiss_loupe()                        # release always dismisses

    def leaveEvent(self, e):
        self._hold.stop()                            # cursor left mid-hold: release may never come
        self._dismiss_loupe()
        self._hover = None
        # Drop any in-flight marquee too. If the grab is lost mid-drag (modal dialog, alt-tab) no
        # release ever arrives, and a stranded _marquee both paints a dashed rect forever and makes
        # wheelEvent's mid-marquee guard disable zoom permanently.
        self._marquee = None
        self._marquee_add = False
        self.hovered.emit("")
        self.update()

    def keyPressEvent(self, e):
        """Keyboard selection, Linux-file-manager style. Cmd/Ctrl-A selects every well; Escape
        clears. Focus is ClickFocus, so these arrive once the user has clicked the plate."""
        if (e.modifiers() & Qt.ControlModifier) and e.key() == Qt.Key_A:
            self.select_all()
            return
        if e.key() == Qt.Key_Escape:
            self.clear_selection()
            return
        super().keyPressEvent(e)

    def set_mosaic_boxes(self, boxes: dict):
        """Adopt the per-FOV cell boxes so a double-click can resolve WHICH FOV was hit.

        Also tells the paint path WHICH cells hold a letterboxed mosaic rather than a
        cell-filling single tile (see :meth:`_cell_source`), so the two can never disagree about
        the same cell: they read one dict.
        """
        self._boxes = dict(boxes or {})
        self._boxed_regions = {r for r, _f in self._boxes}
        self.update()

    # -- carrier geometry (IMA-220 -> IMA-253: DRAWN, not blitted) --
    def set_carrier(self, plate, images_dir=None):
        """Adopt *plate*'s geometry so the holder can be DRAWN behind the cells.

        This used to blit Squid's carrier PHOTOGRAPH. It no longer does, and the reason is
        registration, not taste. A PNG lives in its own pixel space and has to be brought into the
        cell grid's space through three calibration constants (``a1_x_pixel``, ``a1_x_mm``,
        ``mm_per_pixel``) that must agree with the geometry the cells are laid out from. When they
        disagree, nothing raises — you get a plausible picture with the wells in the wrong places,
        which is exactly what shipped, and it is unfixable in general for a FREEFORM holder because
        there is no photograph of "two tissues wherever the operator happened to put them".

        Drawing the outline, the slot/well boundaries and the empty-vs-occupied state from
        :class:`~squidmip._plate.PlateGeometry` puts the holder in the SAME coordinate system as
        the cells, so it cannot misalign, it cannot vanish on pan or zoom (there is no separately
        positioned blit to drift), and an acquisition with no artwork on disk renders identically
        to one with artwork. ``plate.art()`` and the whole PNG registry stay in ``_plate.py`` for
        an optional skin; they are simply not on this path.

        *images_dir* is accepted and ignored, so callers that passed one still work.
        """
        self._carrier = getattr(plate, "geometry", None) if plate is not None else None
        try:
            from squidmip._plate import SlideCarrier
            self._carrier_slide = isinstance(plate, SlideCarrier)
        except Exception:
            self._carrier_slide = False
        # NO true-scale SLIDE ART (Julio, 2026-07-23). The slide-art layout drew glass slides at
        # true micron scale (a 25 mm slide dwarfing an 8 mm tissue) and placed the mosaics at their
        # real stage positions — which stacked two tissues into a tall, tiny, uneven column and
        # "looked like shite". The plate now keeps its EVEN carrier layout (``even_carrier_layout``,
        # equal cells side by side) set at construction, so this no longer overrides ``self._layout``
        # and draws no slide bodies. Even, horizontal, non-overlapping cells beat true-scale slides
        # for a browse view, whatever each tissue's geometry.
        self._slides = None
        self.update()

    # -- cell rectangles: the ONE place a (row, col) becomes widget pixels (IMA-253) --
    def _cell_rect(self, ri: int, ci: int) -> tuple:
        """Widget-pixel ``(x, y, w, h)`` of cell (ri, ci) at the current zoom/pan.

        Uniform plates return the historical ``(ax + ci*cd, ay + ri*cd, cd, cd)`` exactly. A
        freeform holder returns the region's own rectangle: its mosaic's bounding box, scaled by
        the same single transform for every region, so relative offset and relative scale are
        preserved and two regions of different size get different-sized cells.
        """
        cd = self._cd
        ax, ay = self._ox + _HDR, self._oy + _COLH
        if self._layout is not None:
            r = self._layout.get((ri, ci))
            if r is not None:
                return (ax + r[0] * cd, ay + r[1] * cd, r[2] * cd, r[3] * cd)
        return (ax + ci * cd, ay + ri * cd, cd, cd)

    def _cell_source(self, ri: int, ci: int) -> tuple:
        """The sub-rectangle of the montage canvas that ``_cell_rect(ri, ci)`` shows.

        The store keeps every cell as one ``_CELL`` x ``_CELL`` square. A MOSAIC is LETTERBOXED into
        it (``_placement.cell_boxes`` centres it, preserving aspect), so the bars must be excluded
        or the mosaic would be stretched back into them. A single tile — one FOV, or a region
        operator's already-fused result — FILLS the block, so the whole block is the source. Which
        of the two a cell holds is read from ``self._boxes``, the same dict the tiles were placed
        by, so the blit and the pixels cannot disagree.

        Since the cell rect and the letterbox come from the SAME aspect ratio, the inner box is
        recoverable from the rect alone: no second bookkeeping table to fall out of sync.
        """
        full = (ci * _CELL, ri * _CELL, _CELL, _CELL)
        if self._layout is None or self._by_rc.get((ri, ci)) not in self._boxed_regions:
            return full
        r = self._layout.get((ri, ci))
        if r is None or not (r[2] > 0 and r[3] > 0):
            return full
        a = r[2] / r[3]                                    # target aspect == mosaic aspect
        iw = _CELL * min(1.0, a)
        ih = _CELL * min(1.0, 1.0 / a)
        return (ci * _CELL + (_CELL - iw) / 2.0, ri * _CELL + (_CELL - ih) / 2.0, iw, ih)

    def _tiled_region(self) -> "QRegion":
        """The cells that HAVE an image on the active layer, as a QRegion at pan origin (0, 0).

        The montage canvas is opaque _BG wherever no tile landed, so blitting it whole would paint
        the carrier art out. Clipping the blit to this region is what lets the background show
        through the empty wells. Cached and only translated on pan: a full 1536wp is 1536 rects,
        and rebuilding that union on every hover repaint would be the one thing that makes the
        plate feel slow.
        """
        cells = self._tiles_by_layer.get(self._active, set())
        key = (self._cd, self._active, len(cells))
        if self._tile_rgn is None or self._tile_rgn_key != key:
            cd = self._cd
            rgn = QRegion()
            for ri, ci in cells:
                rgn = rgn.united(QRegion(int(ci * cd), int(ri * cd),
                                         int(cd) + 1, int(cd) + 1))   # +1: no hairline seams
            self._tile_rgn, self._tile_rgn_key = rgn, key
        return self._tile_rgn

    def _fov_at(self, c: dict, e) -> int:
        """FOV index under the cursor within cell *c*, or 0 when there is no mosaic to resolve.

        Inverts the placement transform: find where the click landed inside the cell (in _CELL
        units), then pick the FOV whose box contains it. Boxes overlap by ~9% at the seams, so
        the LAST match wins — matching the draw order in ``_OperatorWorker._on_well``, where
        later FOVs paint over earlier ones. Without that agreement a click in a seam would open
        a different FOV than the one visibly on top.
        """
        region = c["well_id"]
        if not region or not self._boxes:
            return 0
        ri, ci = c["row_index"], c["col_index"]
        rx, ry, rw, rh = self._cell_rect(ri, ci)
        sx, sy, sw, sh = self._cell_source(ri, ci)
        if not (rw > 0 and rh > 0):
            return 0
        # position within the cell, normalised to the _CELL-px space the boxes live in. Going via
        # the cell's SOURCE rect is what keeps the hit-test agreeing with the blit on a freeform
        # holder, where the drawn rect is the mosaic's box and not the whole square block.
        fx = (e.x() - rx) / rw * sw + (sx - ci * _CELL)
        fy = (e.y() - ry) / rh * sh + (sy - ri * _CELL)
        hit = 0
        for (r, fov), (top, left, h, w) in self._boxes.items():
            if r == region and top <= fy < top + h and left <= fx < left + w:
                hit = fov
        return hit

    def focusOutEvent(self, e):
        # Only reachable because __init__ sets ClickFocus: with the default NoFocus this widget
        # never held focus, so this handler was dead code pretending to cover "window
        # deactivated mid-hold". A press now focuses the plate, so losing focus is a real signal.
        self._hold.stop()
        self._dismiss_loupe()
        super().focusOutEvent(e)

    def mouseDoubleClickEvent(self, e):
        # Qt sends press/release/dblclick — the second press already re-armed the hold timer, so
        # kill it here or one double-click both opens the well AND raises a loupe.
        self._hold.stop()
        self._dismiss_loupe()
        c = self._cell(e.x(), e.y())
        if c and c["well_id"]:
            self.wellActivated.emit(c["well_id"], self._fov_at(c, e))

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
        tiled = self._tiles_by_layer.get(self._active, set())
        # THE HOLDER (IMA-253), behind everything: drawn from the plate's own geometry, in the
        # cells' own coordinate system. No photograph, so nothing to calibrate and nothing to
        # drift on pan/zoom; see set_carrier.
        self._paint_carrier(p, tiled)
        if self._layout is not None:
            # FREEFORM: each region's cell is its own rectangle, so the montage is blitted per
            # cell rather than as one grid-aligned image. A handful of regions, one drawImage each.
            src = self._active_source()
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            for rc in sorted(tiled):
                if rc not in self._by_rc:
                    continue
                p.drawImage(QRectF(*self._cell_rect(*rc)), src, QRectF(*self._cell_source(*rc)))
        else:
            # Blit the montage from a cached pixmap scaled to the current zoom. The expensive
            # smooth resample runs ONCE per zoom/source-change (not every repaint) — pan/hover
            # just re-blit.
            w, h = max(1, int(W)), max(1, int(H))
            if (self._scaled is None or self._scaled_cd != cd
                    or self._scaled.width() != w or self._scaled.height() != h):
                self._scaled = QPixmap.fromImage(self._active_source()).scaled(
                    w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                self._scaled_cd = cd
            if len(tiled) < nr * nc:
                # The montage canvas is opaque _BG wherever no tile landed, so let it cover only
                # the cells that actually have pixels — otherwise it paints the holder out.
                p.save()
                p.setClipRegion(self._tiled_region().translated(int(ax), int(ay)))
                p.drawPixmap(int(ax), int(ay), self._scaled)
                p.restore()
            else:
                p.drawPixmap(int(ax), int(ay), self._scaled)

        # per-cell DOT over the WHOLE plate grid (so a sparse acquisition still shows the full plate
        # shape — e.g. 32x48 for 1536, 16x24 for 384 — with grey dots on the un-acquired wells):
        # amber = processing, red x = failed, GREY = no image on the active layer, no dot once a cell
        # HAS an image (the image speaks for itself). Dot size is a capped absolute size.
        d = min(max(3.0, cd * 0.36), 15.0)
        active_tiles = self._tiles_by_layer.get(self._active, set())
        for ri in range(nr):
            for ci in range(nc):
                state = self._status.get((ri, ci), "empty")
                has_img = (ri, ci) in active_tiles
                x0, y0, cw, ch = self._cell_rect(ri, ci)
                ex, ey = int(x0 + (cw - d) / 2), int(y0 + (ch - d) / 2)
                if state == "processing":                   # amber dot
                    p.setPen(Qt.NoPen)
                    p.setBrush(_STATUS["processing"])
                    p.drawEllipse(ex, ey, int(d), int(d))
                elif state == "failed":                     # red x within the dot box
                    p.setPen(QPen(_STATUS["failed"], max(1.5, min(cd * 0.09, 3.0))))
                    p.drawLine(ex, ey, ex + int(d), ey + int(d))
                    p.drawLine(ex + int(d), ey, ex, ey + int(d))
                elif not has_img:                           # grey dot: an empty plate position
                    p.setPen(Qt.NoPen)
                    p.setBrush(_STATUS["empty"])
                    p.drawEllipse(ex, ey, int(d), int(d))
                # else: has an image on the active layer -> no dot
        p.setBrush(Qt.NoBrush)

        if self._view_hues:           # PER-VIEW HUES (under the focus wash): each open window/thread
            p.setPen(Qt.NoPen)         # tints its wells in its own colour, so views are told apart.
            for rcs, color in self._view_hues:
                p.setBrush(color)
                for ri, ci in rcs:
                    rx, ry, rw, rh = self._cell_rect(ri, ci)
                    p.drawRect(int(rx), int(ry), int(rw), int(rh))
            p.setBrush(Qt.NoBrush)

        if self._selection:            # SELECTED / focused-view wells = a light blue wash. More
            p.setPen(Qt.NoPen)         # transparent than before (Julio), and it FOLLOWS the open
            p.setBrush(_VIEW_WASH)     # view you click (highlight_regions), as well as manual picks.
            for ri, ci in self._selection:
                rx, ry, rw, rh = self._cell_rect(ri, ci)
                p.drawRect(int(rx), int(ry), int(rw), int(rh))
            p.setBrush(Qt.NoBrush)

        if self._layout is None:
            p.setPen(QPen(_GRID, 3))   # black grid lines between wells (multi-FOV mosaics sit INSIDE a cell)
            for c in range(nc + 1):
                p.drawLine(int(ax + c * cd), int(ay), int(ax + c * cd), int(ay + H))
            for r in range(nr + 1):
                p.drawLine(int(ax), int(ay + r * cd), int(ax + W), int(ay + r * cd))
        # (a freeform holder has no shared grid lines to draw — its cells are individually placed
        #  rectangles, and _paint_carrier already outlined each one.)
        p.setFont(QFont("Helvetica Neue", 11, QFont.DemiBold))
        if self._layout is not None:
            # A freeform region is named, not numbered, and the gutter is sized for "A".."AF" —
            # "manual0" gets sliced in half there. Its own cell is the only place wide enough and
            # the only place that is unambiguous when cells are individually positioned.
            for rc, region in self._by_rc.items():
                rx, ry, rw, _rh = self._cell_rect(*rc)
                p.setPen(_ACCENT if self._hover == rc else _MUTED)
                p.drawText(QRectF(rx, ry - _COLH, max(rw, 60.0), _COLH),
                           int(Qt.AlignCenter), str(region))
        # Column/row labels THIN OUT as cells shrink so they never overlap (a 48-col 1536wp would
        # otherwise cram "1..48" into a few px). Always draw the hovered row/col so hover still
        # reads. Skipped entirely for a freeform holder: its rows and columns are an internal
        # bookkeeping key, not something on the glass, and the names are already on the cells.
        cstep = max(1, int(np.ceil(22.0 / cd)))
        rstep = max(1, int(np.ceil(18.0 / cd)))
        for c in range(nc if self._layout is None else 0):
            hov = bool(self._hover and self._hover[1] == c)
            if c % cstep and not hov:
                continue
            p.setPen(_ACCENT if hov else _MUTED)
            p.drawText(int(ax + c * cd), int(self._oy), int(cd), _COLH, Qt.AlignCenter, str(self._cols[c]))
        for r in range(nr if self._layout is None else 0):
            hov = bool(self._hover and self._hover[0] == r)
            if r % rstep and not hov:
                continue
            p.setPen(_ACCENT if hov else _MUTED)
            p.drawText(int(self._ox), int(ay + r * cd), _HDR, int(cd), Qt.AlignCenter, str(self._rows[r]))
        if self._sel is not None:          # the CURRENT well in the detail viewer = a red BOX
            p.setPen(QPen(_RED, 2))
            p.setBrush(Qt.NoBrush)
            sx, sy, sw, sh = self._cell_rect(*self._sel)
            p.drawRect(int(sx), int(sy), int(sw), int(sh))
        if self._hover is not None:        # where the cursor is, moving around the plate = a red DOT
            ri, ci = self._hover           # SAME geometry as the status dots -> overlays them exactly
            x0, y0, hw, hh = self._cell_rect(ri, ci)
            ex, ey = int(x0 + (hw - d) / 2), int(y0 + (hh - d) / 2)
            p.setPen(Qt.NoPen)
            p.setBrush(_RED)
            p.drawEllipse(ex, ey, int(d), int(d))
        if self._marquee is not None:      # live drag rectangle while Shift-dragging
            mx0, my0, mx1, my1 = self._marquee
            p.setPen(QPen(_ACCENT, 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(int(min(mx0, mx1)), int(min(my0, my1)),
                       int(abs(mx1 - mx0)), int(abs(my1 - my0)))
        if self._loupe is not None:        # press-and-hold magnifier, over everything else
            self._paint_loupe(p)
        # a fine outer white frame around the whole plate view
        p.setPen(QPen(QColor("#c9d1d9"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)
        p.end()

    def _paint_carrier(self, p: QPainter, tiled: set):
        """Draw the sample holder: body outline, per-cell boundary, empty vs occupied (IMA-253).

        Everything here comes out of the geometry the cells themselves are placed from, so there is
        exactly one coordinate system and the holder cannot drift out of register with the wells —
        which is the failure a separately-positioned photograph kept producing, and could not
        avoid. It also means an acquisition with no artwork on disk renders IDENTICALLY to one with
        artwork, because artwork is no longer consulted.

        Three states, deliberately distinct, because "which slots are empty" was the exact question
        the photograph answered badly:

            occupied, imaged   the pixels themselves (drawn over this)
            occupied, waiting  a solid accent-tinted boundary + fill
            empty slot         a DASHED, dim boundary and no fill

        Skipped entirely below a few px per cell: at 1536-well zoom the boundaries are smaller than
        the status dots, so drawing 1536 of them would cost a repaint and show nothing.
        """
        if self._carrier is None:
            return
        cd = self._cd
        ax, ay = self._ox + _HDR, self._oy + _COLH
        if self._slides is not None:
            # SLIDE ACQUISITION (IMA-265): real glass slides at true size, side by side, drawn by
            # _slide_art from the same grid units the tissue cells are placed in. No generic
            # carrier body -- the slides ARE the holder, and the tissue mosaics paint on top of
            # them through the ordinary cell path (so every gesture is untouched).
            from squidmip._slide_art import paint_slides
            slide_rects_px = [(ax + s[0] * cd, ay + s[1] * cd, s[2] * cd, s[3] * cd)
                              for s in self._slides]
            paint_slides(p, slide_rects_px)
            if cd < 6.0:
                return
            self._paint_carrier_cells(p, tiled)
            return
        # The holder BODY: the union of every cell rectangle, padded by the margin the geometry
        # implies (half a pitch beyond the outer cell centres on a well plate).
        rects = [self._cell_rect(r, c) for r in range(self._nr) for c in range(self._nc)]
        if not rects:
            return
        bx0 = min(r[0] for r in rects)
        by0 = min(r[1] for r in rects)
        bx1 = max(r[0] + r[2] for r in rects)
        by1 = max(r[1] + r[3] for r in rects)
        pad = max(4.0, cd * 0.18)
        body = QRectF(bx0 - pad, by0 - pad, (bx1 - bx0) + 2 * pad, (by1 - by0) + 2 * pad)
        p.setBrush(QColor(28, 32, 40))
        p.setPen(QPen(QColor(90, 100, 116), 2))
        p.drawRoundedRect(body, min(10.0, pad), min(10.0, pad))
        # An orientation cue instead of a picture of one: the A1 / first-slot corner is chamfered,
        # the way a real plate's notched corner reads.
        p.setPen(QPen(QColor(120, 132, 150), 2))
        ch = min(14.0, pad * 2.0)
        p.drawLine(int(body.left()), int(body.top() + ch), int(body.left() + ch), int(body.top()))
        if cd < 6.0:                     # boundaries smaller than the status dots: not worth it
            return
        self._paint_carrier_cells(p, tiled)

    def _paint_carrier_cells(self, p: QPainter, tiled: set):
        """The per-cell occupied/empty boundaries, shared by the well plate and the slide holder.

        A cell that already has imaged pixels is left alone (the pixels speak); an occupied but
        un-imaged cell gets a solid accent-tinted boundary; an empty slot gets a dashed dim one.
        The SHAPE differs: a well is round, a slide slot / tissue region is rectangular.
        """
        cd = self._cd
        for ri in range(self._nr):
            for ci in range(self._nc):
                rx, ry, rw, rh = self._cell_rect(ri, ci)
                occupied = (ri, ci) in self._by_rc
                if occupied and (ri, ci) in tiled:
                    continue             # the acquired pixels will cover it; do not tint them
                if occupied:
                    p.setPen(QPen(_ACCENT, max(1.0, min(cd * 0.03, 2.0))))
                    p.setBrush(QColor(56, 139, 253, 40))
                else:
                    p.setPen(QPen(QColor(74, 84, 100), 1, Qt.DashLine))
                    p.setBrush(Qt.NoBrush)
                if self._carrier_slide:  # a slide slot is a rectangle; a well is round
                    p.drawRect(QRectF(rx, ry, rw, rh))
                else:
                    # Well diameter relative to pitch, straight from sample_formats.csv, so a 96wp
                    # reads as fat wells and a 1536wp as pinpricks — the real difference between them.
                    g = self._carrier
                    f = (g.cell_size_um / g.pitch_x_um) if g.pitch_x_um else 0.8
                    f = float(min(max(f, 0.15), 1.0))
                    p.drawEllipse(QRectF(rx + rw * (1 - f) / 2, ry + rh * (1 - f) / 2, rw * f, rh * f))
        p.setBrush(Qt.NoBrush)

    def _paint_loupe(self, p: QPainter):
        """The inset: real pixels, a µm scale bar when the pixel size is known, or the reason
        there are no pixels. Offset from the cursor so the hand never covers what it points at,
        and clamped inside the widget so it stays whole at the plate's edges."""
        x, y = self._loupe["x"], self._loupe["y"]
        s = _LOUPE_PX
        bx = x + 18 if x + 18 + s < self.width() else x - 18 - s
        by = y + 18 if y + 18 + s < self.height() else y - 18 - s
        bx = int(max(2, min(bx, self.width() - s - 2)))
        by = int(max(2, min(by, self.height() - s - 2)))
        p.fillRect(bx, by, s, s, QColor("#05070b"))
        if self._loupe_img is not None:
            p.save()
            p.setClipRect(bx, by, s, s)
            p.drawPixmap(bx, by, QPixmap.fromImage(self._loupe_img).scaled(
                s, s, Qt.KeepAspectRatioByExpanding, Qt.FastTransformation))   # 1:1-ish: no smoothing
            p.restore()
        else:
            p.setPen(_MUTED)
            p.setFont(QFont("Helvetica Neue", 11))
            p.drawText(bx, by, s, s, Qt.AlignCenter | Qt.TextWordWrap,
                       self._loupe_note or "reading …")
        geo = self._loupe_geometry(x, y)
        if geo is not None and self._loupe_img is not None:
            _w, _l, _r, s_loupe, mag = geo
            um_px = loupe_um_per_screen_px(getattr(self._loupe_src, "pixel_size_um", None), s_loupe)
            p.setFont(QFont("Helvetica Neue", 10, QFont.DemiBold))
            if um_px is None:
                # No trustworthy pixel size: say so rather than draw a bar that would be fiction.
                p.setPen(_MUTED)
                p.drawText(bx + 8, by + s - 10, "scale unknown")
            else:
                target = _nice_scale_um(um_px * (s * 0.4))     # ~40% of the inset, rounded to 1/2/5
                bar = int(round(target / um_px))
                p.setPen(QPen(QColor("#e6edf3"), 2))
                p.drawLine(bx + 10, by + s - 14, bx + 10 + bar, by + s - 14)
                p.setPen(QColor("#e6edf3"))
                p.drawText(bx + 10, by + s - 18, f"{_fmt_um(target)}")
            p.setPen(_ACCENT)
            label = f"{self._loupe['well']}  ·  {mag:.1f}×" if mag >= 1.05 else \
                    f"{self._loupe['well']}  ·  native"
            p.drawText(bx + 8, by + 16, label)
        p.setPen(QPen(QColor("#c9d1d9"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(bx, by, s, s)


def _mosaic_boxes(meta: dict) -> dict:
    """``{(region, fov): (top, left, h, w)}`` — every FOV's box inside its _CELL thumbnail.

    Pure geometry, delegated to :mod:`squidmip._placement` (which is Qt-free and unit-tested
    against exact pixel offsets). Returns ``{}`` when the acquisition has no stage positions or
    no pixel size, which is the signal for the caller to keep the historical single-tile path —
    a mosaic is simply not derivable without both, and guessing would draw a wrong picture.

    Placement failures for ONE region are contained: that region falls back to single-tile
    rendering rather than aborting a whole-plate run. The reader has already fail-loud checked
    the CSV/filename agreement, so anything reaching here is a genuine per-region oddity
    (e.g. a region with images but no coordinate rows).
    """
    from squidmip._placement import cell_boxes, fov_offsets_px

    positions = meta.get("fov_positions_um") or {}
    if not positions or meta.get("pixel_size_um") in (None, 0):
        return {}
    frame_shape = meta["frame_shape"]
    out: dict = {}
    for region in meta["regions"]:
        fovs = meta["fovs_per_region"][region]
        if len(fovs) < 2:
            continue                     # a single-FOV well fills its cell; no mosaic needed
        try:
            offsets = fov_offsets_px(positions, region, fovs, meta.get("pixel_size_um"))
            for fov, box in cell_boxes(offsets, frame_shape, _CELL).items():
                out[(region, fov)] = box
        except (KeyError, ValueError):
            continue                     # this region renders single-tile; the rest still mosaic
    return out


def region_mosaic_extent_px(meta: dict, regions: Optional[list] = None) -> Optional[tuple]:
    """Full-resolution ``(height, width)`` bounding box of the mosaics a REGION operator will fuse.

    IMA-245. A region operator (``available_region_operators()``) yields ONE fused mosaic per
    region, whose extent is the bounding box of the region's coordinate-placed frames — NOT the
    frame shape. Anything that has to size a surface for that result (the array viewer's canvas,
    the push planes fed into it) must ask this, or it sizes a mosaic as a frame.

    Returns the max extent over *regions* (``None`` = every region in the acquisition), because
    one array viewer serves the whole run and its canvas is declared once. Returns ``None`` when
    the acquisition carries no stage positions / no pixel size — the same "not derivable, do not
    guess" signal :func:`_mosaic_boxes` returns ``{}`` for.
    """
    from squidmip._placement import fov_offsets_px, mosaic_extent_px

    positions = meta.get("fov_positions_um") or {}
    if not positions or meta.get("pixel_size_um") in (None, 0):
        return None
    frame_shape = meta["frame_shape"]
    scoped = list(meta["regions"]) if regions is None else list(regions)
    best = None
    for region in scoped:
        fovs = meta["fovs_per_region"].get(region) or []
        if not fovs:
            continue
        try:
            offsets = fov_offsets_px(positions, region, fovs, meta.get("pixel_size_um"))
            h, w = mosaic_extent_px(offsets, frame_shape)
        except (KeyError, ValueError):
            continue                     # this region contributes nothing; the rest still size it
        best = (h, w) if best is None else (max(best[0], h), max(best[1], w))
    return best


def push_shape_for(meta: dict, region_op: bool, regions: Optional[list] = None) -> tuple:
    """The ``(height, width)`` of every plane pushed into the array viewer for this run.

    A per-FOV operator pushes a FRAME, so the surface is the frame shape. A REGION operator pushes
    a whole-region MOSAIC, so the surface is the mosaic EXTENT. Either is scaled into a ``_PUSH_PX``
    box PRESERVING ASPECT — a freeform 27-FOV strip is not a square, and squashing it into one
    (which is what a fixed ``(_PUSH_PX, _PUSH_PX)`` surface did) is how it arrives unrecognisable.

    Aspect is preserved the same way :func:`squidmip._placement.cell_boxes` preserves it for the
    plate thumbnail, so the plate and the array viewer describe one geometry rather than two.
    Falls back to the square when the extent is not derivable (no stage positions / pixel size);
    the caller reports that fallback rather than showing a silently squashed mosaic.
    """
    extent = region_mosaic_extent_px(meta, regions) if region_op else None
    if extent is None:                              # per-FOV op, or a region op with no geometry
        extent = tuple(meta["frame_shape"])
    mh, mw = int(extent[0]), int(extent[1])
    s = min(_PUSH_PX / mh, _PUSH_PX / mw, 1.0)     # never UPSCALE: a push is a bounded thumbnail
    return (max(1, int(round(mh * s))), max(1, int(round(mw * s))))


def _fit_letterboxed(a: np.ndarray, h: int, w: int, dtype) -> np.ndarray:
    """Scale a 2D plane into an exactly ``(h, w)`` canvas, aspect preserved and centred.

    The array viewer's canvas is declared ONCE per run (``start_acquisition``), so every push has
    to be that exact shape — while two regions of one acquisition can have differently shaped
    mosaics. Letterboxing is the only way to satisfy both without stretching one of them, and it
    is the policy the plate cell already uses (``cell_boxes`` centres a mosaic in its cell).
    """
    h, w = max(1, int(h)), max(1, int(w))
    s = min(h / a.shape[0], w / a.shape[1])
    ih = max(1, min(h, int(round(a.shape[0] * s))))
    iw = max(1, min(w, int(round(a.shape[1] * s))))
    out = np.zeros((h, w), dtype)
    out[(h - ih) // 2:(h - ih) // 2 + ih,
        (w - iw) // 2:(w - iw) // 2 + iw] = _fit_box(a, ih, iw)
    return out


# --- channel bar: one row per channel, under the plate overview -----------------------------

class _ChannelBar(QWidget):
    """Per-channel VISIBILITY for the plate, one compact row per channel, plus a contrast READOUT.

    A row is  <color dot> [x] <name>  …  <lo – hi>.  The checkbox masks that channel out of the
    plate composite; PlateOverview recomposites from its retained per-channel store, so nothing
    is re-read and nothing is re-projected.

    THERE ARE NO CONTRAST SLIDERS HERE, AND THERE MUST NOT BE (IMA-261)
    -------------------------------------------------------------------
    This strip used to carry a low/high QSlider pair and an "auto" button per channel —
    duplicating the contrast control the embedded ndviewer_light array viewer already has, two
    hand-widths apart on the same screen. Two controls over one quantity is the shape this
    project has now shipped a defect in four times, and it had already gone wrong here: the plate
    followed these sliders, the array viewer followed its own, and the SAME channel was displayed
    at two different windows side by side.

    Contrast therefore has exactly ONE owner — the central array viewer — and this strip only
    REPORTS the window that owner resolved (``set_window``, driven by
    ``LightweightViewer.contrastChanged`` → ``PlateWindow._on_detail_contrast``). A readout is not
    a second control surface: it cannot be dragged, it cannot disagree, and it is what makes the
    sync visible on screen instead of merely asserted in a commit message.
    """

    def __init__(self, labels, colors: np.ndarray, overview: PlateOverview):
        super().__init__()
        self._overview = overview
        self._rows = []           # per channel: (checkbox, contrast readout label)
        self.setStyleSheet(f"background:{_BG};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 5, 12, 6)
        lay.setSpacing(3)
        for c_i, label in enumerate(labels):
            row = QHBoxLayout()
            row.setSpacing(8)
            dot = QLabel("\u25cf")   # the channel's own LUT color, so the strip reads as a legend too
            dot.setStyleSheet("color:rgb({});".format(",".join(str(int(v * 255)) for v in colors[c_i])))
            # NO CHECKBOX. Julio: "there shouldn't be any controls for the plate view. It just
            # reacts to toggles and contrast adjustments in napari." Visibility is owned by
            # napari's eye icons and arrives here through `on_user_visibility`; this label only
            # dims to show the answer.
            box = QLabel(str(label))
            box.setStyleSheet("color:#e6edf3;")
            win = QLabel("\u2014")
            win.setStyleSheet("color:#8b949e;")   # dimmer than a control: this REPORTS, never sets
            win.setToolTip("contrast window, owned by the array viewer on the right \u2014 set it there")
            row.addWidget(dot)
            row.addWidget(box)
            row.addStretch(1)
            row.addWidget(win)
            lay.addLayout(row)
            self._rows.append((box, win))

    def set_window(self, ch: int, lo: float, hi: float):
        """Show the window the CENTRAL VIEWER resolved for *ch*. Display only — it sets nothing."""
        if 0 <= ch < len(self._rows):
            self._rows[ch][1].setText(f"{lo:g} \u2013 {hi:g}")

    def set_visible_state(self, ch: int, on: bool):
        """Show whether napari has this channel on. Display only — it toggles nothing."""
        if 0 <= ch < len(self._rows):
            self._rows[ch][0].setStyleSheet("color:#e6edf3;" if on else "color:#4a5364;")


# --- operator worker: stream a projection over the plate, fill row-major -------------------

class _OperatorWorker(QThread):
    """Runs an operator (MIP) over the plate AND persists it as a navigable multiscale OME-Zarr plate
    (``write_plate``), filling one thumbnail per well as each is written. Projection + pyramid write
    run in write_plate's bounded producer/writer pools; our ``_on_well`` renders the plate tile and
    is called FROM THOSE WRITER THREADS, several at once — so only the done-counter needs ``_lock``
    (the expensive per-channel downsample happens OUTSIDE it, so downsampling still parallelises).
    The worker is deliberately THIN: it emits one native-dtype tile per FIELD — the whole 88x88 cell
    for a single-FOV well, or just that FOV's sub-cell box for a mosaic (IMA-187) — and keeps no
    pixels of its own. PlateOverview owns the per-channel store, the contrast and the compositing,
    so the channel toggle works on every entry path, not only after a run. Memory stays O(engine +
    write workers) wells in flight. The written ``plate.ome.zarr`` is the durable, re-openable artifact.
    """

    tileReady = pyqtSignal(int, int, str, object, object)   # (ri, ci, well_id, (C,h,w) native tile,
    #                                                          box=(top,left,h,w) in cell px | None)
    progress = pyqtSignal(int, int)                 # (done, total)
    streamEnded = pyqtSignal()                      # every well landed -> recomposite the whole plate
    writtenReady = pyqtSignal(str)                  # path of the written plate.ome.zarr
    wellFailed = pyqtSignal(int, int)               # (ri, ci) of a well SKIPPED on a read error
    pushReady = pyqtSignal(int, object)             # (fov_idx, [per-channel ~512px plane]) for the slider
    # FULL-RESOLUTION result pixels, per FOV, for the napari layer group (Defect 3). Separate
    # from pushReady because that one is the ~512px ndviewer slider feed: a downsampled,
    # letterboxed preview. A processing LAYER has to be the operator's actual output, in the
    # raw mosaic's frame, or the before/after toggle compares a thumbnail against a pyramid.
    resultReady = pyqtSignal(str, int, object)      # (region, fov, (C, Y, X) native dtype)
    failed = pyqtSignal(str)                        # whole-run failure (not a per-well skip)
    finished_ok = pyqtSignal()

    def __init__(self, operator: str, reader, meta, fov_index: dict, out_dir: str,
                 regions=None, save: bool = True, n_fovs=1, operator_kwargs=None):
        # No nr/nc: the worker no longer builds a plate-sized montage of its own (IMA-206 moved the
        # canvas into PlateOverview), so it has no use for the plate's shape.
        super().__init__()
        self._operator = operator
        self._reader, self._meta = reader, meta
        self._fov_index = fov_index
        self._out_dir = out_dir
        self._regions = regions          # None = whole plate; a list = subset preview (those wells only)
        self._save = save                # False = PREVIEW: compute + push to the viewer, write NOTHING
        self._n_fovs = n_fovs            # None = every FOV per well -> coordinate-placed mosaic tiles
        # Per-run parameters for a REGION operator (registration on/off, registration channel,
        # feather width, blunder thresholds, channel subset) -- the stitcher panel in pane 1
        # sets these. Carried on BOTH branches of run(): a setting tuned on a preview and then
        # dropped on the save would be thrown away at exactly the moment it is written to disk.
        # A projector has no equivalent seam (its parameters are baked in at registration), and
        # write_plate refuses these for one by name rather than accepting and dropping them.
        self._operator_kwargs = dict(operator_kwargs or {})
        # Per-region FOV boxes inside the _CELL thumbnail (IMA-187). Computed ONCE up front from
        # the reader's stage positions, because every arriving FOV needs its box and the geometry
        # never changes mid-run. Empty dict => no positions => the historical single-tile path.
        # IMA-222: a REGION operator (stitch) returns ONE fused mosaic per well, not one array
        # per FOV, so there are no per-FOV sub-boxes to composite into -- the mosaic IS the cell.
        # A non-empty _boxes here would slot a whole-well mosaic into a single FOV's sub-rectangle.
        from squidmip import available_region_operators

        self._region_op = self._operator in available_region_operators()
        self._boxes = {} if (self._region_op or n_fovs == 1) else _mosaic_boxes(meta)
        # IMA-245: the shape of what this run PUSHES to the array viewer. A region operator pushes
        # a whole-region mosaic, so the surface is the mosaic extent (aspect preserved), not the
        # frame. Computed here, once, and read back by the window through `push_shape` — the window
        # declares the viewer's canvas from the SAME number the worker fills, so the producer and
        # the consumer cannot describe two different rectangles.
        self._push_shape = push_shape_for(meta, self._region_op, regions)
        # True when a region run wanted the mosaic extent and could not derive it (no stage
        # positions / no pixel size), so the push falls back to the square frame surface. The
        # window turns this into a readout line: a squashed mosaic must not look like a correct one.
        self._push_shape_estimated = bool(self._region_op
                                          and region_mosaic_extent_px(meta, regions) is None)
        self._total = len(regions) if regions is not None else len(meta["regions"])
        self._channels = [c["name"] for c in meta["channels"]]
        self._dtype = np.dtype(meta["dtype"])
        self._lock = threading.Lock()             # guards _done (on_well runs on writer threads)
        self._done = 0
        self._seen_fovs: dict[tuple, set] = {}    # (ri,ci) -> FOVs composited so far, for progress
        self._failed_regions: set = set()         # regions whose fields raised (IMA-226: report it)
        self._stop = threading.Event()            # set by the window to end the run cleanly

    @property
    def mosaic_boxes(self) -> dict:
        """``{(region, fov): (top, left, h, w)}`` this run composites into ({} = single-tile path).

        The plate view must hit-test against the SAME boxes the worker paints into, so it reads
        them from here rather than recomputing — a second `_mosaic_boxes(meta)` call would be a
        second chance to disagree, and a disagreement opens a different FOV than the one clicked.
        """
        return self._boxes

    @property
    def push_shape(self) -> tuple:
        """``(h, w)`` of every plane this run pushes to the array viewer (IMA-245).

        Read by the window to size ``start_acquisition``. Same reasoning as ``mosaic_boxes``:
        recomputing it there would be a second chance to disagree, and a disagreement here is a
        black viewer — the push is rejected for the wrong shape and the rejection is invisible.
        """
        return self._push_shape

    @property
    def push_shape_estimated(self) -> bool:
        """True when a REGION run could not derive its mosaic extent and fell back to the square
        frame surface. The window reports it; a squashed mosaic must not pass for a correct one."""
        return self._push_shape_estimated

    @property
    def landed(self) -> int:
        """Wells that actually produced pixels. IMA-226: a live run whose every well raised used to
        finish "✓ · 1 well" with an empty plate behind it — flat-field with no illumination profile
        raises per field, `_on_error` painted the dots red, and the success message printed anyway.
        The status line must not claim a result the plate does not have."""
        with self._lock:
            return self._done

    @property
    def skipped(self) -> int:
        """Regions where at least one field raised and was skipped."""
        with self._lock:
            return len(self._failed_regions)

    def stop(self):
        """Ask the run to stop; write_plate polls this and abandons after in-flight wells drain."""
        self._stop.set()

    def _on_well(self, region, fov, image):
        """Called per written FIELD (on a write_plate WRITER THREAD): composite the plate thumbnail.

        Single-FOV (the historical path) fills the whole _CELL tile. Multi-FOV composites this
        FOV into its coordinate-derived box inside the SAME cell, accumulating across the calls
        for that region — so a 36-FOV well is built from 36 arrivals rather than 36 overwrites.
        Compositing is at THUMBNAIL scale throughout: the cell is _CELL x _CELL no matter how
        many FOVs land in it, so a mosaic well costs the same memory as a single-FOV well.
        """
        info = self._fov_index[region]
        ri, ci, well_id = *info["rc"], info["well_id"]
        well = image[0, :, 0]  # (C, Y, X)
        box = self._boxes.get((region, fov))
        n_c = len(self._channels)

        # Downsample OUTSIDE the lock (the expensive part stays parallel). Single-FOV fills the
        # whole cell; a mosaic FOV is fitted to its own sub-cell box and carries that box along,
        # so the widget can slot it in at the right offset without knowing the geometry.
        if box is None:
            tiles = [_fit_cell(well[c_i]) for c_i in range(n_c)]
            bh = bw = _CELL
        else:
            top, left, bh, bw = box
            tiles = [_fit_box(well[c_i], bh, bw) for c_i in range(n_c)]
        raw = np.empty((n_c, bh, bw), self._dtype)   # native dtype (half the RAM)
        for c_i, ds in enumerate(tiles):
            raw[c_i] = ds
        with self._lock:                          # shared counter -> serialize (the cheap part)
            seen = self._seen_fovs.setdefault((ri, ci), set())
            was_empty = not seen
            seen.add(fov)
            if was_empty:                          # count WELLS, not fields, so the bar still
                self._done += 1                    # reads "n of n wells" on a 36-FOV plate
            done = self._done
        # per-channel + its box; the widget windows, places and composites (IMA-206 + IMA-187)
        self.tileReady.emit(ri, ci, well_id, raw, box)
        self.progress.emit(done, self._total)
        # feed the ndviewer growing slider: one ~512px plane per channel, in memory (register_array),
        # so scrubbing the processed wells is instant and z-collapsed (nz=1). Downsampled -> bounded.
        # ...at `push_shape`: the frame square for a per-FOV operator, the aspect-preserved mosaic
        # extent for a REGION operator (IMA-245). Squashing a region mosaic into the frame square
        # is what put a whole-well stitch into the array viewer as an unreadable rectangle.
        ph, pw = self._push_shape
        push = [_fit_letterboxed(well[c_i], ph, pw, self._dtype)
                for c_i in range(len(self._channels))]
        self.pushReady.emit(info["idx"], push)
        # The operator's own pixels, undownsampled. `well` is a view into `image`; the slot
        # copies what it keeps and drops the rest, so a plate-wide run does not accumulate.
        self.resultReady.emit(region, fov, well)

    def _on_error(self, region, fov, exc):
        """A well's projection failed (corrupt/missing plane): SKIP it, mark its dot failed, keep the
        run alive. One bad file must not abort a whole-plate run."""
        with self._lock:
            self._failed_regions.add(region)
        info = self._fov_index.get(region)
        if info is not None:
            self.wellFailed.emit(*info["rc"])

    def run(self):
        # TIME AND MEASURE THIS RUN. The same measurement the CLI's EngineExecutor makes, at the
        # GUI's own operator-run path, into the same METRICS log — so the comparison table sees
        # both surfaces' runs and the one line per run reaches the log panel (measure_run logs at
        # INFO to the root logger, which the panel is a sink of). One measurement, three consumers.
        target = _explore.describe_run_target(self._regions, total=self._total) or self._operator
        with measure_run(self._operator, target, n_targets=self._total) as _run_metrics:
            _run_metrics.note(surface="gui", save=self._save)
            self._run_body(_run_metrics)

    def _run_body(self, _run_metrics):
        try:
            projector = self._operator
            if self._save:
                # write_plate picks its own stream from the operator, so a region operator (stitch)
                # persists through the same call: both twins yield (region, fov, (T,C,1,Y,X)), and
                # the disk guard sizes a region write from real mosaic extents rather than frames.
                from squidmip import write_plate  # persist + project in one bounded, streaming pass

                write_plate(self._reader, self._out_dir, n_fovs=self._n_fovs, workers=_VIEWER_WORKERS,
                            projector=projector, tiff=False, on_well=self._on_well,
                            stop=self._stop.is_set, on_error=self._on_error, regions=self._regions,
                            operator_kwargs=self._operator_kwargs or None)
                if self._stop.is_set():
                    _run_metrics.finish(_MEASURE_STOPPED, "stopped by the window")
                    return  # window closing / re-opening; drop out cleanly (no final/written emit)
                self.streamEnded.emit()
                self.writtenReady.emit(str(Path(self._out_dir) / "plate.ome.zarr"))
            else:
                # PREVIEW: run the engine over the subset and push each result to the plate + slider,
                # writing NOTHING to disk (so testing an operator on a few wells costs no disk + only
                # the subset's compute). Same math as the saved run — a faithful preview.
                if self._region_op:
                    # IMA-222: a region operator's unit of work is the WELL, so stitch_plate yields
                    # one fused mosaic per region. It mirrors project_plate's contract exactly
                    # (bounded in-flight window, regions=, on_error=, and the same
                    # (region, fov, (T, C, 1, Y, X)) yield), so the loop below is UNCHANGED.
                    # workers=1: peak memory is workers x one fused mosaic (~0.9 GB on a 27-FOV 10x
                    # well), not the ~139 MB of one projected FOV. Saving takes the write_plate
                    # branch above, which dispatches to stitch_plate itself.
                    from squidmip import stitch_plate

                    stream = stitch_plate(self._reader, workers=1, operator=projector,
                                          n_fovs=None, on_error=self._on_error,
                                          regions=self._regions, **self._operator_kwargs)
                else:
                    from squidmip import project_plate

                    stream = project_plate(self._reader, workers=_VIEWER_WORKERS, projector=projector,
                                           n_fovs=self._n_fovs, on_error=self._on_error,
                                           regions=self._regions)
                try:
                    for region, fov, image in stream:
                        if self._stop.is_set():
                            _run_metrics.finish(_MEASURE_STOPPED, "stopped by the window")
                            return
                        self._on_well(region, fov, image)
                finally:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                if self._stop.is_set():
                    _run_metrics.finish(_MEASURE_STOPPED, "stopped by the window")
                    return
                self.streamEnded.emit()
            # Name the OUTCOME for the record, the same rule the status line follows: landed==0 is
            # a partial result however politely we got here (per-well fault isolation keeps one bad
            # file from aborting a plate), and a skip on any well is partial too.
            if self.landed == 0 and self._total:
                _run_metrics.finish(_MEASURE_PARTIAL,
                                    f"produced nothing — all {self._total} target(s) skipped")
            elif self.skipped:
                _run_metrics.finish(_MEASURE_PARTIAL, f"{self.skipped} well(s) skipped")
            else:
                _run_metrics.finish(_MEASURE_OK)
            self.finished_ok.emit()
        except Exception as e:
            # measure_run records this as failed with the exception name and re-raises; catch it
            # here so the QThread still ends via `failed` rather than an unhandled thread exception.
            _run_metrics.finish(_MEASURE_FAILED, f"{type(e).__name__}: {e}")
            self.failed.emit(f"{type(e).__name__}: {e}")


class _MinervaWorker(QThread):
    """Export the selection to Minerva-ingestable files, then start Minerva Author (IMA-228).

    Two stages with deliberately different failure semantics::

        export  ──ok──▶  launch  ──ok──▶  exported(paths) + launched(True)
           │                │
           │                └──fail──▶  exported(paths) + launched(False)   ← files still good
           └──fail──▶  failed(msg)                                          ← nothing written

    A launch failure must NEVER invalidate a successful export: Minerva Author lives in a
    separate checkout that may not be installed, and the OME-TIFF on disk is the deliverable.
    The user always gets the story path either way, because Minerva has no deep link — the
    file is picked by hand in its "Select File" dialog.
    """
    progress = pyqtSignal(int, int)          # (done, total) FOVs exported
    exported = pyqtSignal(object)            # [(ome_path, story_path), ...]
    launched = pyqtSignal(bool)              # did a Minerva server end up answering?
    failed = pyqtSignal(str)
    finished_ok = pyqtSignal()

    def __init__(self, reader, selection, out_dir, projector: str, t: int = 0, launch: bool = True):
        super().__init__()
        self._reader = reader
        self._selection = list(selection)
        self._out_dir = out_dir
        self._projector = projector
        self._t = t
        self._launch = launch
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        from squidmip import _minerva
        try:
            pairs = []

            def on_progress(done, total):
                self.progress.emit(done, total)

            # Export REGION by REGION — the export unit is a fused mosaic per region, so this
            # is also the finest granularity a stop can act on. A stop between regions takes
            # effect promptly; every file already written stays on disk and is reported.
            grouped = _minerva.group_selection(self._selection)
            for i, (region, fovs) in enumerate(grouped.items()):
                if self._stop.is_set():
                    break
                pairs.extend(
                    _minerva.export_selection(
                        self._reader, [(region, f) for f in fovs], self._out_dir,
                        t=self._t, projector=self._projector,
                    )
                )
                on_progress(i + 1, len(grouped))
            self.exported.emit(pairs)
            if pairs and self._launch and not self._stop.is_set():
                # should_stop: the liveness wait is up to 90 s and closeEvent joins this thread.
                # Without it, closing mid-poll froze the GUI for the rest of the wait (84 s).
                self.launched.emit(
                    _minerva.launch_minerva(pairs[0][1], should_stop=self._stop.is_set))
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class _MosaicWorker(QThread):
    """Fuse one region's FOVs into a mosaic per channel, OFF the GUI thread.

    A 28-FOV tissue region is ~940 MB of TIFF to read. Doing that in ``ingest`` would freeze the
    window for seconds on open, which is precisely the "opens instantly" property IMA-260 bought.
    Results arrive per channel so the first channel paints while the rest are still being read.
    """

    ready = pyqtSignal(str, str, object, object)   # region, channel, LEVELS (pyramid), bbox_um|None
    #                                                (no contrast window: napari owns contrast)
    problem = pyqtSignal(str)
    finished_count = pyqtSignal(int)

    def __init__(self, reader, meta, region, channels, z_index=0, parent=None):
        super().__init__(parent)
        self._reader, self._meta = reader, meta
        self._region = region
        self._channels = list(channels)
        self._z_index = int(z_index)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        from squidmip._mosaic_source import fuse_region_pyramid, mosaic_bbox_um

        try:
            bbox = mosaic_bbox_um(self._meta, self._region)
        except Exception as exc:                    # noqa: BLE001
            self.problem.emit(f"mosaic placement failed: {exc}")
            bbox = None

        n = 0
        for ch in self._channels:
            if self._stop.is_set():
                break
            try:
                # A LAZY MULTISCALE PYRAMID of (z, y, x) levels — the same shape of data the
                # written-OME-Zarr path has always handed napari via open_pyramid. napari fetches
                # only the clipped visible region of the level matching the current zoom, so a
                # fit-to-window view costs a coarse level (~0.9 MB) instead of a full-resolution
                # fused plane (54.9 MB on the real 10x region), per channel, per z step.
                #
                # napari's own dimension slider is still the z control: every level keeps the z
                # axis at full length and only y/x are coarsened. Only the visible (level, z) is
                # ever materialised, and _mosaic_source's bounded cache keeps a revisited one.
                res = fuse_region_pyramid(self._reader, self._meta, self._region, ch)
            except Exception as exc:                # noqa: BLE001 - reported, never swallowed
                self.problem.emit(f"{self._region}/{ch}: {type(exc).__name__}: {exc}")
                continue
            if res is None:
                self.problem.emit(
                    f"{self._region}: no stage positions / pixel size — mosaic not derivable."
                )
                continue
            # NO contrast window on the wire. ima-nav-controls added one here, computing
            # napari's own calc_data_range off-thread because it cost ~940 ms/channel on the GUI
            # thread. That measurement was taken against a FLAT level-0 stack; with the multiscale
            # pyramid napari autoscales from the small level it actually renders, so the cost it
            # was routing around is gone. Passing a window again would put a second contrast
            # decision on the wire for no measured gain.
            levels, _step, _nz = res
            self.ready.emit(self._region, ch, levels, bbox)
            n += 1
        self.finished_count.emit(n)

class _FocusWorker(QThread):
    """Rank one FOV's z planes by Tenengrad sharpness, OFF the GUI thread.

    The reference-plane scan reads every z plane of one FOV — ~40-50 TIFFs on the tissue set.
    It used to run inside the button's ``clicked`` slot, which froze the window solid for the
    whole scan with no progress and no cancel; it was the only long operation in the app without
    a QThread. Planes are area-downsampled to 512 px before scoring, so the cost is dominated by
    the reads rather than the metric.
    """

    ready = pyqtSignal(int, str)          # (z index of the sharpest plane, a note or "")
    problem = pyqtSignal(str)

    def __init__(self, reader, meta, region, fov, channel, parent=None):
        super().__init__(parent)
        self._reader, self._meta = reader, meta
        self._region, self._fov, self._channel = region, fov, channel
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        from squidmip.projection import _tenengrad

        best_z_i, best_f, read = 0, -1.0, 0
        failures = []
        for z_i, z in enumerate(self._meta["z_levels"]):
            if self._stop.is_set():
                return
            try:
                plane = self._reader.read(self._region, self._fov, self._channel, z)
            except Exception as exc:              # noqa: BLE001 - counted and REPORTED below
                failures.append(f"z={z}: {type(exc).__name__}")
                continue
            read += 1
            f = _tenengrad(_area_downsample(plane, 512, 512).astype(np.float32))
            if f > best_f:
                best_f, best_z_i = f, z_i
        if read == 0:
            # NEVER return a default. Reporting "focused on z=0" when nothing could be read is
            # the log-and-continue failure this project has six confirmed instances of.
            self.problem.emit(
                f"{self._region}:{self._fov} — not one z plane of {self._channel} could be "
                f"read, so there is no sharpest plane. ({'; '.join(failures[:3])})")
            return
        note = ("" if not failures else
                f" ({len(failures)} of {len(self._meta['z_levels'])} planes were unreadable "
                f"and were skipped)")
        self.ready.emit(int(best_z_i), note)


class _SpotWorker(QThread):
    """Run spot detection on the plane pane 2 is CURRENTLY showing, off the GUI thread.

    Spencer: *"responsiveness is important. And an indicator when its working."* Both halves are
    structural here rather than cosmetic:

    * **Responsive.** Same ``QThread`` shape as ``_MosaicWorker``/``_PreviewWorker``. The click
      handler builds this and calls ``start()``, which returns immediately; every pixel is
      touched on this thread. Measured on region ``manual0`` of the 10x tissue slide
      (5731 x 4793 fused mosaic, 405 nm): ~7.3 s total, of which the watershed is ~4.8 s.
    * **Indicator.** ``progress(done, total)`` counts STAGES of the recipe, matching the
      ``pyqtSignal(int, int)`` convention every other worker here uses, so an existing indicator
      binds to it unchanged. That signal has no text channel, so the stage NAME goes out
      separately on ``stageChanged(str)`` rather than being smuggled into an int.
    * **Cancellable.** ``stop()`` sets an Event that ``detect_spots`` polls between stages. The
      cancel is honoured at the next stage boundary (worst case one watershed), and a cancelled
      run emits ``cancelled`` and NO result — never a half-finished mask presented as an answer.

    The plane is taken from the layer already on the canvas, not re-read from disk, so what was
    counted is exactly what the user is looking at.
    """

    progress = pyqtSignal(int, int)                # (stages done, stages total) — the convention
    stageChanged = pyqtSignal(str)                 # the TEXT channel progress(int,int) cannot carry
    ready = pyqtSignal(str, str, object, object, object, int)
    # ^ (region, channel, labels (H,W) int32, centroids (N,2) float, bbox_um|None, count)
    problem = pyqtSignal(str)                      # a NAMED failure: "<region>/<channel>: ..."
    cancelled = pyqtSignal()
    finished_count = pyqtSignal(str, str, int)     # (region, channel, count) — the run's answer

    def __init__(self, region, channel, data, z_index, bbox_um, params=None, parent=None):
        super().__init__(parent)
        self._region, self._channel = region, channel
        self._data, self._z = data, z_index
        self._bbox_um = bbox_um
        self._params = params
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        from squidmip._spots import SpotDetectionCancelled, detect_spots, preferred_segmenter

        where = f"{self._region}/{self._channel}"
        algorithm = preferred_segmenter()
        try:
            plane = _full_res_plane(self._data, self._z)
            log.info("%s: detecting nuclei with %s on a %s plane", where, algorithm, plane.shape)

            res = detect_spots(
                plane, self._params, algorithm=algorithm,
                on_stage=lambda name, done, total: (self.stageChanged.emit(name),
                                                    self.progress.emit(done, total)),
                should_stop=self._stop.is_set,
            )
        except SpotDetectionCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:                   # noqa: BLE001 - NAMED, never swallowed
            # Log AND banner. The banner shows it where the user is looking; the log gives it a
            # permanent, copyable line in the panel — a user who clicked the banner away still has
            # the record. (This is the "logger didn't show the detect-nuclei error" gap.)
            log.error("%s: spot detection failed — %s: %s", where, type(exc).__name__, exc)
            self.problem.emit(f"{where}: spot detection failed — {type(exc).__name__}: {exc}")
            return

        self.progress.emit(len(_spot_stages()), len(_spot_stages()))
        self.stageChanged.emit("done")
        # Success goes to the LOG too, not only the napari readout — the user saw the count in the
        # viewer but nothing in the panel. A run that produced a number the user can act on should
        # leave a copyable line in the log like every other operator.
        log.info("%s: %d nuclei detected (%s)", where, res.count, algorithm)
        self.ready.emit(self._region, self._channel, res.labels, res.centroids,
                        self._bbox_um, res.count)
        self.finished_count.emit(self._region, self._channel, res.count)


def _spot_stages():
    """The stage list, imported lazily so ``_viewer`` keeps no second copy of the denominator."""
    from squidmip._spots import STAGES

    return STAGES


class _FlatfieldWorker(QThread):
    """Estimate an illumination profile LIVE from plate tiles, off the GUI thread.

    All processing comes from maragall/: this calls ``_flatfield.estimate_profile`` which is
    ``tilefusion.flatfield.estimate_flatfield_channel`` (the numpy BaSiC port), NOT a reimplemented
    estimator. The BaSiC solve is seconds-to-minutes, so it must not run on the GUI thread. Reads a
    SPREAD sample of FOVs across the plate (decorrelated content makes the low-rank/sparse split
    better than the first N tiles of one well). Fails to the LOG by name, never silently.
    """

    done = pyqtSignal(object)     # FlatfieldProfile
    problem = pyqtSignal(str)
    stage = pyqtSignal(str)

    def __init__(self, reader, meta, channel, *, max_tiles=48, use_darkfield=False, parent=None):
        super().__init__(parent)
        self._reader, self._meta, self._channel = reader, meta, channel
        self._max_tiles = int(max_tiles)
        self._use_dark = bool(use_darkfield)

    def run(self):                                    # pragma: no cover - Qt thread
        try:
            from squidmip._flatfield import estimate_profile

            meta = self._meta
            z0 = (meta.get("z_levels") or [0])[0]
            fpr = meta.get("fovs_per_region") or {}
            pairs = [(region, int(fov))
                     for region in (meta.get("regions") or [])
                     for fov in (fpr.get(region) or [])]
            if not pairs:
                self.problem.emit("no FOVs to estimate a flat-field from.")
                return
            # Spread the sample across the plate, not the first N of one well.
            step = max(1, len(pairs) // self._max_tiles)
            sample = pairs[::step][: self._max_tiles]
            tiles = []
            for region, fov in sample:
                try:
                    tiles.append(np.asarray(self._reader.read(region, fov, self._channel, int(z0))))
                except Exception:                     # noqa: BLE001 - one bad tile is not fatal
                    continue
                self.stage.emit(f"read {len(tiles)}/{len(sample)} tiles for {self._channel}…")
            if len(tiles) < 3:
                self.problem.emit(
                    f"flat-field estimate needs at least 3 readable tiles for {self._channel}, "
                    f"got {len(tiles)}.")
                return
            self.stage.emit(f"estimating illumination (tilefusion BaSiC) from {len(tiles)} tiles…")
            profile = estimate_profile(np.stack(tiles), use_darkfield=self._use_dark)
            log.info("flat-field: estimated a %s profile from %d tiles (tilefusion BaSiC)",
                     self._channel, len(tiles))
            self.done.emit(profile)
        except Exception as exc:                      # noqa: BLE001 - NAMED to the log, not swallowed
            log.error("flat-field estimate failed for %s: %s", self._channel, exc)
            self.problem.emit(f"{type(exc).__name__}: {exc}")


def _full_res_plane(data, z_index):
    """The FULL-RESOLUTION 2-D plane behind a napari layer's ``data``, whatever shape it is in.

    A napari layer's ``data`` is one of three things here, and counting cells on the wrong one
    gives a wrong number that looks entirely plausible:

    * a **list of pyramid levels** (a multiscale mosaic — level 0 is full resolution, and every
      later level has fewer, larger-looking nuclei). ALWAYS level 0: counting a 4x-downsampled
      level would merge touching nuclei and silently under-report.
    * a **(z, y, x) stack** — take the z the user is actually looking at.
    * a plain **(y, x)** plane.

    Only the ONE plane asked for is ever materialised; a lazy pyramid stays lazy until the
    ``np.asarray`` at the end.
    """
    # A pyramid arrives as a list/tuple whose ELEMENTS are arrays (level 0 is full resolution). A
    # plain nested Python list whose elements are lists/scalars is NOT a pyramid — it merely
    # encodes one array — so it is converted whole. `ndim` on the first element is the
    # discriminator: >=2 means "element is an array" (pyramid); otherwise it is a nested list.
    if isinstance(data, (list, tuple)):
        if not data:
            raise ValueError("the layer holds an EMPTY multiscale pyramid — nothing to count.")
        data = data[0] if getattr(data[0], "ndim", 0) >= 2 else np.asarray(data)

    # Trust ``.ndim`` when present (keeps a lazy dask/zarr level lazy until the final asarray). When
    # it is ABSENT — a container whose ndim defaulted to 2 is exactly what let a (z, y, x) stack
    # skip the reduction and reach the raise as a 3-D "plane" — materialise once and read the real
    # ndim, so the shape of the container never decides whether the z reduction runs.
    ndim = getattr(data, "ndim", None)
    if ndim is None:
        data = np.asarray(data)
        ndim = data.ndim

    # A 3-D (z, y, x) stack is indexed at the z the user is looking at. Anything with MORE leading
    # axes is genuinely ambiguous — we cannot know which is z, which is channel, which is time — so
    # it is REFUSED by name rather than silently counting the middle of the wrong axis.
    if ndim == 3:
        n_z = int(data.shape[0])
        z = n_z // 2 if z_index is None else int(z_index)
        data = data[min(max(z, 0), n_z - 1)]

    plane = np.asarray(data)
    if plane.ndim != 2:
        raise ValueError(
            f"expected a 2-D plane to count on, got shape {plane.shape!r}. The layer's data is "
            "neither a pyramid level list, a (z, y, x) stack, nor a (y, x) plane."
        )
    return plane


class _PreviewWorker(QThread):
    """Fast RAW preview so the plate shows imagery the moment it opens — before any operator runs
    (the "downsample the plate before opening" step). Reads ONE representative z-plane per channel
    per FOV (not the whole stack), area-downsamples, and hands the per-channel tile to the plate.
    Cheap relative to a full projection; parallel reads. Status stays 'empty' (grey frame) — this is
    a preview, not a processed result. A later operator overwrites each tile. Like the other
    producers it keeps the CHANNEL AXIS intact all the way to the widget, so the channel toggle
    works on a freshly-opened acquisition, before any operator has run.

    A REGION IS A MOSAIC, NOT A FOV (IMA-253/IMA-249). This used to read exactly one representative
    FOV per region and stretch it over the region's whole cell, so the real 10x tissue acquisition
    showed two lone frames pretending to be two 27- and 28-FOV mosaics, and the mosaic only ever
    appeared *after* an operator run. It now composites every FOV of a region into that region's
    cell at its coordinate-derived box (``_placement.cell_boxes`` — the same geometry the operator
    path uses, so preview and result describe one layout).

    The cost is driven by the REAL FOV COUNT PER REGION, which is the only way both datasets stay
    fast: the 1536-well fixture is 1536 regions x 1 FOV, so it reads 1536 planes per channel exactly
    as before, takes the identical single-tile code path (``box=None``), and cannot get slower. The
    tissue slide is 2 regions x ~27 FOVs, so it reads 55. Work is emitted per FOV as it lands, so
    cells fill progressively and the UI never blocks on a whole mosaic.
    """

    tileReady = pyqtSignal(int, int, str, object, object)   # (ri, ci, well_id, tile, box|None)
    #: This is the raw fill, not an operator run. ``_explore.operator_busy`` reads it so a retired
    #: preview still draining cannot make the next operator run refuse itself.
    IS_PREVIEW = True

    streamEnded = pyqtSignal()                      # preview complete -> recomposite the whole plate
    failed = pyqtSignal(str)                         # a preview that could not finish NAMES why —
    #                                                 a bare `except: pass` left the plate frozen
    #                                                 half-grey, indistinguishable from "loading".

    def __init__(self, reader, meta, fov_index: dict, order: list, mosaic: bool = True):
        super().__init__()
        self._reader, self._meta = reader, meta
        self._fov_index, self._order = fov_index, order
        self._channels = [c["name"] for c in meta["channels"]]
        self._dtype = np.dtype(meta["dtype"])
        self._mosaic = bool(mosaic)
        self._stop = threading.Event()

    def _plan(self) -> list:
        """``[(region, fov, box|None), ...]`` — the read list, in plate order, FOVs in stage order.

        ``box=None`` means "this FOV fills its cell": a single-FOV region, or an acquisition with no
        stage coordinates / no pixel size, where a mosaic is not derivable and guessing one would
        draw a wrong picture. That is the historical path, and it stays byte-identical for it.

        A region whose FOVs are so widely spread that each one lands in fewer than
        ``_MIN_PREVIEW_BOX_PX`` cell pixels also previews single-tile. Reading N planes to paint N
        specks is all cost and no picture — and at that scale the "mosaic" and the single tile are
        visually the same thing anyway. The operator path still mosaics it; this is a PREVIEW
        budget, not a change to the geometry.
        """
        from squidmip._placement import cell_boxes, fov_offsets_px

        positions = self._meta.get("fov_positions_um") or {}
        px = self._meta.get("pixel_size_um")
        plan: list = []
        for region in self._order:
            fovs = list(self._meta["fovs_per_region"][region])
            boxes: dict = {}
            if self._mosaic and len(fovs) > 1 and positions and px not in (None, 0):
                try:
                    boxes = cell_boxes(fov_offsets_px(positions, region, fovs, px),
                                       self._meta["frame_shape"], _CELL)
                except (KeyError, ValueError):
                    boxes = {}       # this region previews single-tile; the rest still mosaic
                if any(min(b[2], b[3]) < _MIN_PREVIEW_BOX_PX for b in boxes.values()):
                    boxes = {}
            if boxes:
                plan.extend((region, f, boxes[f]) for f in fovs if f in boxes)
            else:
                plan.append((region, fovs[0], None))
        return plan

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            from concurrent.futures import ThreadPoolExecutor
            zs = self._meta["z_levels"]
            z_mid = zs[len(zs) // 2]      # a mid-stack plane is a fair single-plane preview
            plan = self._plan()

            def load(item):
                region, fov, box = item
                h, w = (_CELL, _CELL) if box is None else (box[2], box[3])
                fit = _fit_cell if box is None else (lambda a: _fit_box(a, h, w))
                return region, box, [fit(self._reader.read(region, fov, ch, z_mid)
                                         .astype(np.float32)) for ch in self._channels]

            with ThreadPoolExecutor(max_workers=_VIEWER_WORKERS) as ex:
                for region, box, tiles in ex.map(load, plan):   # plate order preserved
                    if self._stop.is_set():
                        return
                    ri, ci = self._fov_index[region]["rc"]
                    self.tileReady.emit(ri, ci, region,
                                        np.stack(tiles).astype(self._dtype), box)
            if not self._stop.is_set():
                self.streamEnded.emit()   # the running window is mature now -> one clean recomposite
        except Exception as exc:
            # Preview is best-effort, but best-effort is not SILENT. Finalise the tiles that did
            # land (so a partial mosaic still paints) and then name the failure — the old bare
            # `except: pass` stranded the plate half-grey forever, and streamEnded never fired so
            # the status line kept claiming the load was still in progress.
            if not self._stop.is_set():
                self.streamEnded.emit()
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _ComputedPlateWorker(QThread):
    """Read a previously-written OME-Zarr plate back into the viewer (no recompute).

    Streams each well from disk: a coarse pyramid level -> the plate thumbnail, and a ~512px level ->
    the ndviewer slider (register_array). Bounded (one well in flight); reads via tensorstore off the
    GUI thread so opening a big computed plate never freezes the window. Emits per-channel tiles, so
    a reopened plate is windowed GLOBALLY by the widget's running contrast exactly like a live run —
    it used to take percentiles per well, which made a dim well and a bright well look identical and
    silently broke the one thing a plate overview is for (comparing wells at a glance)."""

    tileReady = pyqtSignal(int, int, str, object)   # (ri, ci, well_id, (C, cell, cell) native tile)
    pushReady = pyqtSignal(int, object)             # (fov_idx, [per-channel ~512px plane])
    progress = pyqtSignal(int, int)
    streamEnded = pyqtSignal()                      # plate fully loaded -> recomposite globally
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, base, wells, coarse_lvl, push_lvl, dtype):
        super().__init__()
        self._base = base                 # plate.ome.zarr path
        self._wells = wells               # [(well_id, wellpath, fov, ri, ci, flat_idx)]
        self._coarse, self._push = coarse_lvl, push_lvl
        self._dtype = dtype
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _read(self, wellpath, fov, level):
        import tensorstore as ts
        path = f"{self._base}/{wellpath}/{fov}/{level}"
        arr = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": path}}).result()
        return np.asarray(arr[0, :, 0].read().result())   # (C, y, x) at t=0, z=0

    def run(self):
        try:
            n = len(self._wells)
            for i, (wid, wpath, fov, ri, ci, idx) in enumerate(self._wells, 1):
                if self._stop.is_set():
                    return
                coarse = self._read(wpath, fov, self._coarse)             # thumbnail source (C,y,x)
                tile = np.stack([_fit_cell(plane.astype(np.float32)) for plane in coarse])
                self.tileReady.emit(ri, ci, wid, tile.astype(self._dtype))
                push_src = self._read(wpath, fov, self._push)             # detail-slider source (C,Y,X)
                # ...at the declared push canvas exactly (IMA-245): a pyramid level smaller than
                # _PUSH_PX used to be pushed at its own size, which the viewer silently refused.
                push = [_fit_letterboxed(push_src[c], _PUSH_PX, _PUSH_PX, self._dtype)
                        for c in range(push_src.shape[0])]
                self.pushReady.emit(idx, push)
                self.progress.emit(i, n)
            if not self._stop.is_set():
                self.streamEnded.emit()   # every well in the store -> one global-window recomposite
                self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# --- main window: plate overview | embedded ndviewer ----------------------------------------

class _ExplorationTab(QWidget):
    """One 'exploration' tab: a saved FOV/region subset plus the operator UI scoped to it (IMA-205).

    Multi-instance by design — one per selection — which is why it does NOT reuse an operator
    tab's fixed key. Identity is content-addressed (``exploration_tab_key``), so re-selecting the
    same wells focuses this tab instead of opening a second copy of it.

        selection {B2,B3,B4}
              |
              v  exploration_tab_key(acq, regions) -> "exp:1a2b3c"
        _ExplorationTab(regions)  --run--> run_operator(op, regions=..., tab_key="exp:1a2b3c")
              |                                              |
              |                                    layer "<op>@exp:1a2b3c"
              +--close--> stop run, drop layers, free canvases
    """

    def __init__(self, regions: list, tab_key: str, parent=None):
        super().__init__(parent)
        self.cursor = _explore.SubsetCursor(regions)
        self.regions = self.cursor.regions
        self.tab_key = tab_key
        self.status: dict = {}      # this tab's plate dots, restored when it becomes active
        self.sync_note = None       # set by _build_exploration_tab; the "not synced yet" banner
        self.sync_pending = False   # True while this tab is in front but the view still shows a run
        # THE TAB'S OWN VIEWER — built by pane 2's constructor, embedded in this widget. "The
        # right pane is essentially a copy of the central pane, but it occurs on a subset."
        self.viewer = None          # the MosaicPane, or None with the reason said on screen
        self.slider = None          # the slider UNDER the viewer: one stop per region
        self.region_label = None    # "region 1 of 3 · B2"
        self.progress = None        # what a preview run scoped to this tab has computed so far
        self.minerva_btn = None
        self.mosaic_worker = None   # the fuse-this-region thread currently feeding self.viewer
        self.tiles: dict = {}       # region -> the cell canvas a multi-FOV run is filling in
        self.plate_layer = None     # the PLATE layer the run displayed here writes into

    def dispose(self):
        """Free the tab's viewer and stop its mosaic read.

        A napari viewer is a GL context and tens of MB; leaking one per Shift-drag kills a
        session after twenty selections. Called from ``_discard_exploration`` — the ONE teardown
        path — so a tab close, a float close and app exit all free it identically.
        """
        w = self.mosaic_worker
        self.mosaic_worker = None
        if w is not None and w.isRunning():
            w.stop()
            w.wait(2000)
        pane = self.viewer
        self.viewer = None
        if pane is not None:
            # Close the napari Viewer FIRST. deleteLater() on the Qt wrapper does not close it —
            # napari holds every Viewer in its own instance registry — so without this the GL
            # context and its ~tens of MB leaked once per Shift-drag (the very leak this docstring
            # names). MosaicPane.shutdown() is idempotent and no-ops when no viewer was built.
            if hasattr(pane, "shutdown"):
                pane.shutdown()
            pane.setParent(None)
            pane.deleteLater()

    def set_sync_pending(self, pending: bool):
        """Say out loud that this tab is in FRONT but the plate/detail beside it still belong to a
        run that is finishing. A tab which silently shows someone else's wells is the whole bug —
        the banner is the honest state until _on_run_drained catches the view up."""
        self.sync_pending = bool(pending)
        if self.sync_note is not None:
            self.sync_note.setVisible(self.sync_pending)

    def shutdown(self):
        """Called by _close_op_tab (duck-typed, like the CLI terminal's). The window does the real
        teardown in _discard_exploration — this exists so the hasattr(w, 'shutdown') path is safe."""
        return


def _make_mosaic_pane(show_docks: bool = True):
    """Build a napari mosaic viewer, or report why it could not be built.

    Returns ``(pane_or_None, mode, message)``. Import failures are caught here rather than at
    module import so that a machine without napari still opens the window with ndviewer_light —
    and with a VISIBLE sentence saying so, never a silent downgrade.
    """
    try:
        from squidmip._napari_pane import make_pane

        return make_pane(show_docks=show_docks)
    except Exception as exc:                     # noqa: BLE001 - surfaced, not swallowed
        return None, "ndv", f"napari viewer unavailable ({type(exc).__name__}: {exc}) — using ndviewer_light."


class PlateWindow(QMainWindow):
    #: The in-flight operator result for the region pane 2 is showing (Defect 3), or None.
    #: A CLASS default rather than an __init__ assignment so ``_on_result`` can use plain
    #: attribute access: a bare ``getattr(self, ..., None)`` on a QObject whose __init__ has
    #: not run raises out of Qt's own attribute machinery instead of returning the default.
    _result_acc = None

    def __init__(self, initial_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("MIP tool")
        self.resize(1600, 950)
        self._worker = None           # the operator (MIP) run
        self._preview = None          # the raw preview fill on open
        self._minerva = None          # the Minerva export + Author launch (IMA-228)
        self._retired = []            # workers asked to stop; kept alive until they actually finish
        self._overview = None
        self._reader = None
        self._meta = None
        self._mosaic_worker = None    # fuses a region's FOVs for pane 2, off the GUI thread
        # THE SINGLE OWNER of "which region is current". The red ROI frame on the plate, the
        # region slider, and the mosaic in pane 2 are all VIEWS of this one value — none of them
        # keeps a copy. Before it there were three copies, hand-synced: PlateOverview._sel,
        # _mosaic_region and _current_well. Both `_mosaic_region` and `_current_well` are now
        # PROPERTIES that read the cursor, so an assignment cannot create a fourth.
        self._cursor = RegionCursor()
        self._cursor.subscribe(self._on_region_changed)
        self._cursor.on_problem(lambda msg: self._readout.setText(msg))
        # THE communication backbone, built once and owned here.
        # * _log_bus attaches to the stdlib ROOT logger, so every orchestrated library (tilefusion,
        #   petakit, bgsub, and the per-run measurement line) appears in the panel with no wiring.
        # * _activity is the single registry of in-flight work the panel's header reads.
        # * commands is the ONE command surface (squidmip._command) — the GUI is now a CALLER of the
        #   same layer the CLI drives, so an agent/test/script says one command to both.
        self._log_bus = LogBus()
        self._log_bus.install()
        self._activity = ActivityLog()
        from squidmip._gui_commands import install_command_bus
        self.commands = install_command_bus(self)
        self._spot_worker = None      # spot detection on the visible mosaic, off the GUI thread
        self._spot_counts = {}        # (region, channel) -> nuclei counted. PER-REGION, not global.
        self._fov_index = {}
        self._selected_regions = []   # wells picked on the plate (IMA-221); scopes an operator run
        self._pushed = set()          # wells whose raw z-stack is already registered in the detail viewer

        # DECENTRALIZED VIEWER (Spencer, 2026-07-23 call). The plate is the ROOT; a selection opens
        # an INDEPENDENT napari window that floats on the desktop, tracked by ID in the Open View
        # list. Many wells become ONE window with a region slider, not many windows. Every window
        # shares this one stateless reader/meta — nothing reopens the dataset. See _region_viewer.
        from squidmip._region_viewer import ViewerManager
        self._viewer_manager = ViewerManager(parent=self)
        # Operator controls appear AT EACH LEVEL (the deck; Julio 2026-07-23: "I don't see operator
        # controls like the powerpoint specified at each level"). Every window's "Operators for this
        # window" dropdown is the SAME registry + run_operator (the CLI engine), scoped to that view,
        # so "select where to run stitching" = pick the view, Run. Only runnable operators appear
        # (minerva/Gallery View are terminals that stay on the root's stack).
        self._viewer_manager.operator_specs = [
            (op.key, op.label) for op in _OPERATIONS if op.runnable]
        self._viewer_manager.run_operator = self.run_operator
        # The plate wash shows ONLY the view you CLICK (Julio), coloured by that view's own hue so
        # different view threads are told apart. Not all views at once — that clutters the plate.
        # viewFocused fires on open/raise; windowsChanged clears it when the focused view closes.
        self._viewer_manager.viewFocused.connect(lambda _regions: self._refresh_view_hues())
        self._viewer_manager.windowsChanged.connect(self._refresh_view_hues)

        # File menu: a reliable "Open acquisition folder" (drag-drop can be blocked on Windows by the
        # GL child pane or an elevation mismatch, so this is the always-works path).
        file_menu = self.menuBar().addMenu("&File")
        open_act = QAction("&Open acquisition folder…", self)
        open_act.triggered.connect(self._open_acquisition_dialog)
        file_menu.addAction(open_act)
        open_hcs = QAction("Open a &computed MIP (.hcs)…", self)
        open_hcs.triggered.connect(self._open_computed)
        file_menu.addAction(open_hcs)

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
        self._current_well = None     # a PROPERTY over self._cursor — see below. Kept as an
        #                               assignment so every existing call site still reads.
        self._current_fov = 0         # the FOV of that region on screen (IMA-250: autofocus ranks IT)
        self._acq_path = None         # the opened acquisition dir (persist writes next to it)
        self._processed_plate = None  # path of the written plate.ome.zarr once an operator persists it
        self._plate_mode = "raw"      # what the plate view is showing — shown in the plate-pane title
        self._plate_format = None     # the format the plate is laid out with (declared or inferred)
        self._plate_format_override = None   # manual override; also read from SQUIDMIP_WELLPLATE_FORMAT
        self._op_stack = OperationStack()   # the toggleable layer stack (base + applied operators)
        self._active_op_key = None    # operator whose tiles are streaming into its layer right now
        self._layers_tab = None       # the Layers tab widget, once opened
        self._order = []              # well order = the detail's FOV-slider order
        self._op_tabs = {}            # key -> operator-UI widget currently open as a tab in _left_tabs
        self._floating = {}           # key -> _FloatWindow holding that operator's UI detached
                                      # (a key lives in exactly ONE of the two dicts, never both)
        self._push_index = None       # global plate idx -> current run's slider position (None = identity)
        # IMA-245: the (h, w) canvas the array viewer was last declared with, and the sticky reason
        # a push could not be shown. None = no array canvas declared (the raw path registers file
        # paths, not arrays), which is the signal for _on_push to skip the shape check.
        self._push_shape = None
        self._push_problem = None
        self._readout_base = ""
        self._dropped_pushes = 0
        self._active_exploration = None   # the exploration tab currently in front, if any
        self._tabs_muted = False      # suppress _on_tab_changed during bulk teardown (ingest)
        self._run_out_dir = None      # output dir of the in-flight SAVE run (for partial cleanup)
        self._run_tab_key = None      # exploration tab that owns the in-flight run's LAYER, if any
        self._run_view_tab_key = None # side-pane tab the in-flight run is DISPLAYED in, if any
        self._run_label = ""          # the in-flight run's operator label, and where it is going —
        self._run_dest = ""           # one source for the status line AND the side-pane tab
        self._pending_resync = False  # a tab switch was deferred because a run was live (IMA-205 bugs)
        self._runs_settled = 0        # monotonic: bumped once a run's TERMINAL cascade has run — the
        #                               tiles, the streamEnded recomposite AND _on_run_drained. It is
        #                               the honest "done" signal a test must wait on: QThread.finished
        #                               (hence _busy()==False) fires BEFORE Qt dispatches the queued
        #                               tileReady/streamEnded/finished slots to the main thread, so a
        #                               test that waited on `not _busy()` was reading state its own
        #                               event loop had not yet applied (IMA-258 flakes).
        self._loupe_sources = {}      # layer key -> _LoupeSource backing that layer's pixels (IMA-208)

        # THREE HORIZONTAL PANES on one monitor (IMA-237). Tabs live inside a pane (their bar sits at
        # the pane's top, like the plate pane's title bar) — never a global strip across the window.
        # Any detachable tab can be DRAGGED OUT of its bar into a free-floating window (ImageJ-style;
        # see _detach_tab, which serves BOTH bars):
        #   PANE 1 = plate view + the controls with the tabs. A vertical split: on top the PROCESS
        #            console, a QTabWidget with a "Process wells" home tab (operator list) and one tab
        #            per operator you open (MIP -> where-to-save UI; Record -> recorder UI); below it
        #            the HCS PLATE view, whose title bar names the plate.
        #   PANE 2 = the initial viewer: the ndviewer_light array viewer, full height. ONE widget
        #            instance (never tabbed), but NOT plate-fixed: its FOV slider FOLLOWS the active
        #            exploration tab, which re-points it at that tab's subset; no exploration tab
        #            restores the whole plate (_on_tab_changed). ndviewer's only retarget seam is
        #            start_acquisition, which resets the viewer, so computed frames do not survive a
        #            switch; raw plane paths are re-registered so it isn't black.
        #   PANE 3 = the EXPLORATION pane: one tab per Shift-dragged FOV subset (IMA-205/221).
        #
        # Pane 3 is VISIBLE FROM OPEN (IMA-260), reversing IMA-237's reveal-on-first-drag. The
        # saving of a fifth of the monitor bought undiscoverability: you cannot find a pane that is
        # not there, so nobody found the Shift-drag that was the only way to make it appear. It now
        # opens showing EXAMPLE USAGE (_build_explore_empty) and swaps to the tab bar the moment it
        # holds real content — a pane that teaches costs its width back immediately.
        # Exploration tabs moved OUT of the process console to get here: the console is pane 1, and
        # pane 1 is not where the user asked exploration to live.

        # top-left: the process console (build the home tab first — it owns self._readout, which
        # _make_detail_viewer writes to if ndviewer is unavailable).
        self._left_tabs = _DetachTabs(self._detach_tab)
        # Dark the tab widget's own canvas (the strip behind/beside the tabs rendered white in macOS
        # light mode). Scope a Fusion style + dark palette to THIS widget subtree only — NOT the app,
        # which would bleed into the embedded ndviewer and hide its per-channel colour swatches.
        self._fusion_style = QStyleFactory.create("Fusion")   # keep a ref: setStyle doesn't own it
        if self._fusion_style is not None:
            self._left_tabs.setStyle(self._fusion_style)
        # LIGHT-BLUE tab-scroll arrows on BLACK boxes (Julio). The ‹ › scroller buttons draw their
        # arrow in the palette's ButtonText, so force that light blue on the tab widget; the QSS
        # keeps the button boxes black.
        _tabs_pal = _dark_palette()
        _tabs_pal.setColor(QPalette.ButtonText, QColor("#58a6ff"))
        self._left_tabs.setPalette(_tabs_pal)
        self._left_tabs.setAutoFillBackground(True)
        self._left_tabs.setStyleSheet(
            _TABS_DARK
            + "QTabBar QToolButton{background:#000000;border:1px solid #30363d;}"
              "QTabBar QToolButton:hover{background:#161b22;}")
        self._left_tabs.setTabsClosable(True)
        self._left_tabs.tabCloseRequested.connect(self._close_op_tab)
        self._left_tabs.currentChanged.connect(self._on_tab_changed)
        self._left_tabs.addTab(self._build_process_pane(), "Operators")
        self._left_tabs.tabBar().setTabButton(0, QTabBar.RightSide, None)  # home tab isn't closable

        # PANE 3: the exploration pane. Same _DetachTabs class as the console (one detach seam, not
        # two), but every tab is detachable — it has no permanent home tab to protect.
        self._explore_tabs = _DetachTabs(self._detach_tab, first_detachable=0)
        if self._fusion_style is not None:
            self._explore_tabs.setStyle(self._fusion_style)
        self._explore_tabs.setPalette(_dark_palette())
        self._explore_tabs.setAutoFillBackground(True)
        self._explore_tabs.setStyleSheet(_TABS_DARK)
        self._explore_tabs.setTabsClosable(True)
        # ELIDE long tab titles. A preview tab is named after its operator ("Maximum Intensity
        # Projection - C3-C5 (3)"), and a QTabBar's size hint includes every tab's full width: the
        # third such tab pushed the WHOLE WINDOW 260 px wider than it was asked to be, on a small
        # monitor, which is the "controls eclipsing content" failure arriving from the one
        # direction nobody watches. Measured on screen, not reasoned about.
        self._explore_tabs.setElideMode(Qt.ElideRight)
        self._explore_tabs.tabBar().setExpanding(False)
        self._explore_tabs.tabBar().setUsesScrollButtons(True)
        self._explore_tabs.tabCloseRequested.connect(
            lambda i: self._close_op_tab(i, self._explore_tabs))
        self._explore_tabs.currentChanged.connect(self._on_tab_changed)
        # Pane 3 is a two-page STACK, not a bare tab bar: page 0 is the example-usage empty state,
        # page 1 is the tab bar. One widget owns "what pane 3 shows", so the empty copy and the
        # tabs can never be on screen together and can never both be off it. _explore_tabs.isHidden()
        # stays truthful for free — QStackedWidget hides the page that is not current — which is
        # exactly what "there are no exploration tabs" means, and what callers already read.
        self._explore_empty = self._build_explore_empty()
        self._explore_pane = QStackedWidget()
        self._explore_pane.setStyleSheet(f"background:{_BG};")
        self._explore_pane.addWidget(self._explore_empty)   # page 0: example usage
        self._explore_pane.addWidget(self._explore_tabs)    # page 1: real content
        # Wide enough to set 24 px copy without one word per line — the legibility floor is a floor
        # on the TEXT, and text you have to read a syllable at a time is not legible either.
        self._explore_pane.setMinimumWidth(360)

        # NO CENTRAL VIEWER (decentralized, 2026-07-23). The locked central napari pane is gone:
        # viewing now happens in INDEPENDENT windows spawned from the plate (see _region_viewer),
        # each its own napari viewer. The root is just the plate + the Open View list + the log.
        # These stay defined-as-None because dozens of methods guard on them (``_load_mosaic``,
        # ``_on_result``, ``activate_well`` all early-return when they are None), so a stray call
        # from a menu operator no-ops instead of crashing rather than needing every call site cut
        # in one pass. Operator-result display migrates onto the windows next (Phase C).
        self._mosaic_pane = None
        self._detail = None
        self._right_widget = None

        # bottom-left: plate view (drop target until an acquisition opens). Its FIXED title bar names
        # the wellplate we're on (the acquisition) — the plate's identity lives with the plate.
        self._plate_title = QLabel("well plate")   # plate name; shows the hovered well (large) on hover
        self._plate_title.setStyleSheet(           # the BAR below now carries background + border
            "color:#e6edf3;font-size:17px;font-weight:800;padding:9px 14px;border:none;")
        # NO CONTRAST CONTROL HERE. Julio: "there shouldn't be any controls for the plate
        # view. It just reacts to toggles and contrast adjustments in napari." The scope
        # dropdown that used to sit here is gone with per-region contrast itself.
        plate_title_bar = QWidget()
        plate_title_bar.setStyleSheet("background:#0b0e14;border-bottom:1px solid #232b3a;")
        _tb = QHBoxLayout(plate_title_bar)
        _tb.setContentsMargins(0, 0, 12, 0)
        _tb.setSpacing(8)
        _tb.addWidget(self._plate_title, 1)
        self._drop = QLabel("Drop a Squid acquisition folder here\n\n"
                            "then pick an operator in  Process wells")
        self._drop.setAlignment(Qt.AlignCenter)
        self._drop.setStyleSheet("color:#8b98ad;font-size:16px;border:2px dashed #232b3a;border-radius:12px;margin:24px;")
        plate_host = QWidget()
        plate_host.setStyleSheet(f"background:{_BG};")
        self._left_l = QVBoxLayout(plate_host)
        self._left_l.setContentsMargins(0, 0, 0, 0)
        self._left_l.setSpacing(0)
        self._left_l.addWidget(plate_title_bar)

        # SELECTION BAR (the deck's "Selection" label): shows which wells operators will run on
        # ("run on selected wells"), and a Select all button. Operators default to this selection.
        sel_bar = QWidget()
        sel_bar.setStyleSheet("background:#0b0e14;border-bottom:1px solid #232b3a;")
        _sb = QHBoxLayout(sel_bar)
        _sb.setContentsMargins(12, 5, 12, 5)
        _sb.setSpacing(8)
        _sel_cap = QLabel("Selection:")
        _sel_cap.setStyleSheet("color:#8b98ad;font-size:12px;border:none;")
        self._selection_label = QLabel("none — click wells, or Select all")
        self._selection_label.setStyleSheet("color:#c9d1d9;font-size:12px;border:none;")
        self._select_all_btn = QPushButton("Select all")
        self._select_all_btn.setCursor(Qt.PointingHandCursor)
        self._select_all_btn.setStyleSheet(
            "QPushButton{background:#161b22;color:#c9d1d9;border:1px solid #30363d;"
            "border-radius:4px;padding:3px 10px;font-size:12px;}"
            "QPushButton:hover{background:#21262d;}")
        self._select_all_btn.clicked.connect(self._select_all_wells)
        # OPEN the current selection as ONE window (for a shift-/Cmd-CLICK selection, which unlike a
        # shift-DRAG has no release gesture to open on). Julio: "how do I open a new window with them
        # after selecting them with the shift click?"
        self._open_sel_btn = QPushButton("Open view")
        self._open_sel_btn.setCursor(Qt.PointingHandCursor)
        self._open_sel_btn.setStyleSheet(
            "QPushButton{background:#1f6feb;color:#ffffff;border:1px solid #1f6feb;"
            "border-radius:4px;padding:3px 10px;font-size:12px;}"
            "QPushButton:hover{background:#388bfd;}")
        self._open_sel_btn.clicked.connect(self._open_selected_view)
        # Copy/paste LUTs TO THE PLATE (Julio: "we have to be able to copy and paste luts to the
        # plate"). Shares the one _LUT_CLIPBOARD windows use, so a window's contrast pastes onto the
        # plate and vice versa — the plate is a View with controls like any window.
        _lut_qss = ("QPushButton{background:#161b22;color:#c9d1d9;border:1px solid #30363d;"
                    "border-radius:4px;padding:3px 10px;font-size:12px;}"
                    "QPushButton:hover{background:#21262d;}")
        self._plate_copy_lut_btn = QPushButton("⧉ LUTs")
        self._plate_copy_lut_btn.setToolTip("Copy the plate's per-channel contrast.")
        self._plate_copy_lut_btn.setCursor(Qt.PointingHandCursor)
        self._plate_copy_lut_btn.setStyleSheet(_lut_qss)
        self._plate_copy_lut_btn.clicked.connect(self._plate_copy_luts)
        self._plate_paste_lut_btn = QPushButton("⤓ LUTs")
        self._plate_paste_lut_btn.setToolTip("Apply the copied LUTs to the plate.")
        self._plate_paste_lut_btn.setCursor(Qt.PointingHandCursor)
        self._plate_paste_lut_btn.setStyleSheet(_lut_qss)
        self._plate_paste_lut_btn.clicked.connect(self._plate_paste_luts)
        _sb.addWidget(_sel_cap)
        _sb.addWidget(self._selection_label, 1)
        _sb.addWidget(self._plate_copy_lut_btn)
        _sb.addWidget(self._plate_paste_lut_btn)
        _sb.addWidget(self._select_all_btn)
        _sb.addWidget(self._open_sel_btn)
        self._left_l.addWidget(sel_bar)

        self._left_l.addWidget(self._drop, 1)    # the plate overview replaces this on ingest

        # THE REGION SLIDER — the navigation control, replacing the FOV slider. It lives in the
        # PLATE pane, directly under the plate, because the thing it moves is the red ROI frame
        # drawn on that plate. Under napari there was previously no navigation control on screen
        # at all: the FOV slider belonged to ndviewer_light, which is not constructed when napari
        # is the viewer.
        # NO region slider on the root plate. The deck puts the region slider ("<> A1, B6, C3") in
        # each spawned WINDOW, not on the plate — navigation is per window now. Building napari's
        # QtDims here also loaded napari icons with no napari viewer registered, which is the
        # "theme_dark:/playback-forward.svg not found" warning spam. Playback/frame_done paths
        # guard on None, so leaving it unbuilt is safe.
        self._region_slider = None

        # "Focus reference plane" was a control UNDER the old central viewer. It has no central
        # viewer to drive now (per-FOV autofocus belongs on a window, Phase C), but its setEnabled
        # callers still exist, so keep the button as a hidden orphan rather than chase every call
        # site. _sync_focus_button leaves it hidden.
        self._focus_btn = QPushButton("Focus reference plane")
        self._focus_btn.clicked.connect(self._focus_reference_plane)
        self._focus_btn.hide()

        # THE ROOT IS JUST THE PLATE (decentralized, 2026-07-23). The central viewer and the
        # exploration pane are gone from the layout; the plate column IS the window. Selections
        # open independent napari windows (the Views dock, added below), and the log lives in a
        # bottom dock — Julio: "the logger on the bottom of the GUI". This replaces the locked
        # 3-pane grid that Spencer asked us to dismantle.
        self._log_panel = LogPanel(self._log_bus, self._activity)
        self._log_panel.start()

        # THE DECK LAYOUT (2026-07-23 image): ONE COMPACT PORTRAIT (h>w) window — a top row of two
        # small panels [Open View list | Operators (bulk)] over a big Wellplate view below. NOT OS
        # docks spread across a wide window (that was wrong): the deck is a single tidy rectangle.
        from squidmip._region_viewer import OpenViewList
        self._open_views = OpenViewList(self._viewer_manager, self)

        top_row = QSplitter(Qt.Horizontal)
        top_row.setStyleSheet("QSplitter{background:#0b0e14;}"
                              "QSplitter::handle{background:#232b3a;width:1px;}")
        top_row.addWidget(self._open_views)     # top-left: "Open View list 'selectable'"
        top_row.addWidget(self._left_tabs)      # top-right: "Operators (bulk) to selection"
        top_row.setSizes([280, 280])
        top_row.setHandleWidth(6)
        # The top row is a COMPACT strip — the plate is the star, not these two small panels. A
        # fixed max height stops the operator cards' size hint from ballooning it into the "super
        # thick" top that squashed the plate. Its OWN panels scroll inside this height.
        top_row.setMaximumHeight(240)
        top_row.setMinimumHeight(150)

        root = QWidget()
        root.setStyleSheet(f"background:{_BG};")
        rv = QVBoxLayout(root)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(1)
        rv.addWidget(top_row, 0)                # compact strip, keeps its height
        rv.addWidget(plate_host, 1)             # the Wellplate view fills the rest
        self._split = top_row
        self.setCentralWidget(root)

        # LOCK THE WHOLE ROOT DARK so macOS LIGHT theme cannot whiten the framing (Julio: "make
        # sure my mac's light theme doesn't whiten the framing"). The old code scoped Fusion+dark to
        # just the tab subtree to protect the embedded ndviewer's colour swatches — but ndviewer and
        # the central pane are gone, so the whole root can go dark. napari's windows are SEPARATE
        # top-levels with their own stylesheet, so this does not touch them.
        if self._fusion_style is not None:
            self.setStyle(self._fusion_style)
        self.setPalette(_dark_palette())
        self.setStyleSheet("QMainWindow{background:#0b0e14;}")
        self.statusBar().setStyleSheet(
            "QStatusBar{background:#0b0e14;color:#8b98ad;} QStatusBar::item{border:0px;}")
        self.menuBar().setStyleSheet(
            "QMenuBar{background:#0b0e14;color:#c9d1d9;} "
            "QMenuBar::item:selected{background:#1f6feb;}")

        # THE LOG IS ITS OWN SMALL WINDOW (Julio: "logger on a small separate window, so we can
        # follow this compact h>w layout"). A top-level QMainWindow kept alive on self; toggle it
        # from the View menu. Not a dock — a dock would widen the compact root.
        self._log_window = QMainWindow(self)
        self._log_window.setWindowFlag(Qt.Window, True)
        self._log_window.setWindowTitle("Log")
        self._log_window.setCentralWidget(self._log_panel)
        if self._fusion_style is not None:
            self._log_window.setStyle(self._fusion_style)
        self._log_window.setPalette(_dark_palette())
        self._log_window.setStyleSheet("QMainWindow{background:#0b0e14;}")
        self._log_window.resize(760, 240)

        self._sync_explore_pane()                  # keeps the (now-hidden) op-tab stack coherent

        # FIXED SIZE on any display (Julio): 596 wide x 850 tall. A hard setFixedSize so the compact
        # portrait shape is identical on every monitor and never balloons — the plate dominates
        # below the capped top strip.
        self.setFixedSize(596, 850)

        # The log window opens alongside the root and is toggled from the View menu.
        view_menu = self.menuBar().addMenu("&View")
        self._log_act = QAction("&Log window", self)
        self._log_act.setCheckable(True)
        self._log_act.setChecked(True)
        self._log_act.toggled.connect(self._log_window.setVisible)
        view_menu.addAction(self._log_act)
        self._log_window.show()

        self.setAcceptDrops(True)
        if initial_path:
            self.ingest(initial_path)

    # -- the Operators panel (top-right): a scrollable list of operator blocks ----------------------
    def _build_process_pane(self) -> QWidget:
        """The Operators panel: JUST a scrollable list of operator blocks — no header, no footer
        (Julio, 2026-07-23). Each block opens that operator; operators apply to the plate SELECTION
        (Cmd/Ctrl-A picks the whole plate). Minerva and Gallery View are here as the deck's terminal
        operators. Status moved to the window status bar; the old 'run on' scope combo and the
        raw/3D/MIP footer buttons are kept as hidden orphans so their many callers still resolve —
        they migrate onto the operator tabs and the windows in the operator phase."""
        # Status line — tests and many methods read self._readout; it now lives in the status bar,
        # not as a pane header. Created here because _build_process_pane runs during __init__.
        self._readout = QLabel("Drop a Squid acquisition, then pick an operator.")
        self._readout.setStyleSheet("color:#8b98ad;font-size:12px;")
        self.statusBar().addWidget(self._readout, 1)

        # Hidden orphans (referenced elsewhere; not shown — no header/footer).
        self._scope_run = QComboBox()
        self._scope_run.addItems(list(_explore.RUN_SCOPES))
        self._scope_run.hide()
        self._raw_btn = QPushButton("Return to raw view")
        self._raw_btn.clicked.connect(self._return_to_raw)
        self._raw_btn.hide()
        self._native3d_btn = QPushButton("3D native (napari)…")
        self._native3d_btn.clicked.connect(self._open_native_3d)
        self._native3d_btn.hide()

        pane = QWidget()
        pane.setStyleSheet(f"background:{_BG};")
        v = QVBoxLayout(pane)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(0)

        stack = QWidget()
        sv = QVBoxLayout(stack)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(8)
        self._op_cards = {}
        # TERMINAL operators on TOP of the stack (Julio, 2026-07-23: "I need minerva author and the
        # gallery view to be on the top of the stack"). Gallery View first, then Minerva Author, then
        # the processing operators. Gallery View is a button (gathers windows), Minerva is an
        # Operation card; the rest follow in registry order, minus minerva (already placed).
        gv = QPushButton("Gallery View\nArrange the open viewer windows into a gallery")
        gv.setEnabled(False)
        gv.setCursor(Qt.PointingHandCursor)
        gv.setStyleSheet(_CARD_QSS)
        gv.setMinimumHeight(54)
        gv.clicked.connect(self._open_gallery_view)
        sv.addWidget(gv)
        self._op_cards["galleryview"] = gv

        _minerva = [op for op in _OPERATIONS if op.key == "minerva"]
        ordered = _minerva + [op for op in _OPERATIONS if op.key != "minerva"]
        for op in ordered:
            card = QPushButton(f"{op.label}\n{op.blurb}")
            card.setEnabled(False)                         # enabled once an acquisition loads
            card.setCursor(Qt.PointingHandCursor)
            card.setStyleSheet(_CARD_QSS)
            card.setMinimumHeight(54)
            card.clicked.connect(lambda _=False, k=op.key: self._activate_operator(k))
            sv.addWidget(card)
            self._op_cards[op.key] = card
        sv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        scroll.setWidget(stack)
        v.addWidget(scroll, 1)
        return pane

    def _open_gallery_view(self):
        """Gallery View terminal operator (slide 2): "prepares a gallery view instance using the
        selected Napari windows, with current views". The window-assembly lands with the operator
        phase; for now report what it will gather so the block is never a silent dead control."""
        n = len(self._viewer_manager.windows) if hasattr(self, "_viewer_manager") else 0
        if n == 0:
            self._readout.setText("Gallery View: open some viewer windows first, then gather them.")
            return
        self._readout.setText(
            f"Gallery View: {n} open window(s) will be arranged into a gallery "
            "(assembly lands with the operator phase).")

    def _open_native_3d(self):
        """Popout napari 3D on the current region's centre FOV at native resolution (gallery-view
        recipe). Carries the embedded layers' current contrast and colormap so the volume matches
        what is on screen. Fails to the LOG by name, never silently."""
        if self._reader is None or self._meta is None:
            self._readout.setText("No acquisition open — drop one before opening the 3D view.")
            return
        region = getattr(self, "_mosaic_region", None) or self._cursor.region
        if region is None:
            self._readout.setText("No region is open to render in 3D.")
            return
        contrast, colormap = {}, {}
        pane = getattr(self, "_mosaic_pane", None)
        mosaic = getattr(pane, "mosaic", None) if pane is not None else None
        if mosaic is not None:
            op = mosaic.visible_op()
            if op is not None and op != getattr(self, "SPOTS_OP", None):
                for ch in mosaic.channels(op):
                    ly = mosaic.find(op, ch)
                    if ly is None:
                        continue
                    try:
                        contrast[ch] = tuple(float(x) for x in ly.contrast_limits)
                        colormap[ch] = ly.colormap
                    except Exception:                # noqa: BLE001 - carry what we can
                        pass
        try:
            from squidmip._napari3d import open_native_3d

            open_native_3d(self._reader, self._meta, region,
                           contrast_by_channel=contrast, colormap_by_channel=colormap)
            log.info("opened native napari 3D popout for region %s", region)
        except Exception as exc:                     # noqa: BLE001 - NAMED, to the log and readout
            log.error("native 3D view failed for region %s: %s", region, exc)
            self._readout.setText(f"3D native view failed: {exc}")

    # -- operator UIs live as tabs INSIDE pane 1 (home tab + one per opened operator); exploration
    # -- tabs live in pane 3. Both bars share every path below — *tabs* says which one. -----------
    def _build_explore_empty(self) -> QWidget:
        """Pane 3 with nothing in it: EXAMPLE USAGE, not a blank strip (IMA-260).

        The pane is visible from open, so 'empty' is a state a user will actually look at, and a
        blank column teaches nothing — the Shift-drag and the right-click that fill this pane are
        both invisible gestures with no button anywhere. So the empty state names one concrete
        path (right-click -> Control Well, the primary), then a second (Shift-drag), then says
        plainly that these are only examples. It is illustration, not instruction: the user asked
        to be shown a way in, not told what to do.

        Every string here is sized at or above the project's legibility floor — see
        _EMPTY_BODY_PX. Copy nobody can read from their chair is a blank pane with extra steps.
        """
        w = QWidget()
        w.setStyleSheet(f"background:{_BG};color:#e6edf3;")
        v = QVBoxLayout(w)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(10)

        head = QLabel(_EMPTY_EXPLORE_HEAD)
        head.setWordWrap(True)
        head.setStyleSheet(f"color:#e6edf3;font-size:{_EMPTY_HEAD_PX}px;font-weight:800;")
        v.addWidget(head)

        # The gestures DIFFER by acquisition kind, so the copy does too: "Control Well" is a
        # plate concept and does not exist on a glass slide. Naming a gesture the user cannot
        # perform is worse than saying nothing.
        slide = self.is_slide_acquisition()
        primary = _EMPTY_EXPLORE_PRIMARY_SLIDE if slide else _EMPTY_EXPLORE_PRIMARY
        secondary = _EMPTY_EXPLORE_SECONDARY_SLIDE if slide else _EMPTY_EXPLORE_SECONDARY
        for text, color in ((_EMPTY_EXPLORE_LEDE, "#c3ccd9"),
                            (primary, "#e6edf3"),
                            (secondary, "#c3ccd9"),
                            (_EMPTY_EXPLORE_CODA, "#8b98ad")):
            lab = QLabel(text)
            lab.setWordWrap(True)
            lab.setStyleSheet(f"color:{color};font-size:{_EMPTY_BODY_PX}px;line-height:150%;")
            v.addWidget(lab)
        v.addStretch(1)
        return w

    def is_slide_acquisition(self) -> bool:
        """Is this a glass slide / hand-drawn tissue acquisition rather than a well plate?

        Read from the RESOLVED plate format, not guessed from region names: `_plate_shape`
        already owns that inference (and its manual override), and a second rule here would be
        another two-answers-to-one-question. Unknown counts as a slide, because the copy it
        selects names no gesture the user might not have.
        """
        from squidmip._plate_shape import GLASS_SLIDE, normalize_plate_format

        meta = self._meta
        if meta is None:
            return False
        fmt = normalize_plate_format(meta.get("wellplate_format"), strict=False)
        return fmt is None or fmt == GLASS_SLIDE

    def explore_empty_text(self) -> str:
        """Everything pane 3 is currently SAYING while empty — '' once it holds content.

        One reader for the whole empty state, so a check cannot pass by finding a label that is on
        the widget but not on the screen: it returns text only while the empty page is the page
        pane 3 is showing."""
        if self._explore_pane.currentWidget() is not self._explore_empty:
            return ""
        head = self._explore_empty.findChildren(QLabel)
        return "\n".join(lab.text() for lab in head)

    def _sync_explore_pane(self):
        """Show the tab bar once pane 3 holds a tab, and the EXAMPLE COPY whenever it does not.

        Pane 3 keeps its width either way (IMA-260) — it is a permanent third column, so this is a
        page swap inside it, never a collapse. Both directions matter: the copy has to come back
        when the last tab closes, or a user who explores once and tidies up is left with the blank
        strip the empty state exists to prevent."""
        page = self._explore_tabs if self._explore_tabs.count() > 0 else self._explore_empty
        if self._explore_pane.currentWidget() is not page:
            self._explore_pane.setCurrentWidget(page)

    def _open_op_tab(self, key: str, title: str, builder, tabs=None):
        """Open (or focus) a UI as a tab. Built lazily, once. *tabs* is the bar it belongs in —
        the process console by default, pane 3 for exploration tabs.
        If the UI is currently detached (see _detach_tab), focus its floating window instead —
        never rebuild: for the CLI that would mean a second live shell."""
        tabs = self._left_tabs if tabs is None else tabs
        win = self._floating.get(key)
        if win is not None:
            win.raise_()
            win.activateWindow()
            return
        w = self._op_tabs.get(key)
        if w is None:
            w = builder()
            self._op_tabs[key] = w
            tabs.addTab(w, title)
            self._sync_explore_pane()
        tabs.setCurrentWidget(w)

    def _close_op_tab(self, index: int, tabs=None):
        tabs = self._left_tabs if tabs is None else tabs
        if index == 0 and tabs is self._left_tabs:         # 'Process wells' home tab — never closable
            return
        w = tabs.widget(index)
        tabs.removeTab(index)
        self._dispose_tab_widget(w)
        self._sync_explore_pane()

    def _dispose_tab_widget(self, w):
        """The ONE teardown path for an operator UI — tab close, float close, and app exit all
        route here so they can't drift: registry pop, stale-ref clear, shell kill, delete.

        An exploration tab owns MORE than a widget (a possibly-live run and a set of plate layers),
        so its extra teardown hangs off this same path rather than off the tab-close caller — a
        float-close or an app exit must free it exactly as a tab close does (IMA-209 + IMA-205)."""
        if isinstance(w, _ExplorationTab):                 # stop its run + free its layers FIRST
            self._discard_exploration(w)
        for k, v in list(self._op_tabs.items()):
            if v is w:
                del self._op_tabs[k]
        if w is self._layers_tab:                          # drop the stale ref so refresh no-ops
            self._layers_tab = None
            self._layers_box = None
        if hasattr(w, "shutdown"):                         # a live terminal — kill its shell first
            w.shutdown()
        w.deleteLater()

    # -- drag a tab out -> free-floating window (IMA-209); Re-dock returns it ---------------------
    def _detach_tab(self, index: int, tabs=None):
        """Detach the tab at `index` of `tabs` into a _FloatWindow. ALL detach logic lives here (the
        drag in _DetachTabBar is a thin, deferred caller) so the offscreen tests drive it directly.
        Returns the new window, or None when the tab can't detach (home tab / unregistered).

        ONE implementation serves both bars (IMA-237): pane 3's tabs float out through this exact
        path, and re-dock to the bar they came from. *tabs* defaults to the process console, so
        IMA-209's callers and tests are unchanged."""
        tabs = self._left_tabs if tabs is None else tabs
        if index <= 0 and tabs is self._left_tabs:   # the process console's home tab never detaches
            return None
        if index < 0:
            return None
        w = tabs.widget(index)
        key = next((k for k, v in self._op_tabs.items() if v is w), None)
        if key is None:
            return None
        title = tabs.tabText(index)
        tabs.removeTab(index)
        del self._op_tabs[key]
        # _layers_tab is deliberately NOT cleared: the widget lives on in the float and
        # _refresh_layers_tab writes into it directly, so a floating Layers keeps updating.
        # `*_` is load-bearing: on_redock is connected to QPushButton.clicked, which passes
        # `checked=False` and would land on a bare `lambda k=key:` AS k — so the Re-dock button
        # called _redock(False), found no such key in _floating, and returned silently. The button
        # had been dead since IMA-209 because every test called _redock(key) directly instead of
        # clicking it. Swallow the signal's argument and keep the key bound.
        win = _FloatWindow(title, w,
                           on_close=lambda *_, k=key: self._on_float_closed(k),
                           on_redock=lambda *_, k=key: self._redock(k))
        win._home_tabs = tabs        # re-dock returns it to the bar it was dragged out of
        self._floating[key] = win
        win.show()
        self._sync_explore_pane()    # pane 3 collapses if that was its last tab
        return win

    def _on_float_closed(self, key: str):
        """User closed the floating window: same fate as closing the tab."""
        win = self._floating.pop(key, None)
        if win is None:
            return
        w = win.take_content()
        if w is not None:
            self._dispose_tab_widget(w)
        self._sync_explore_pane()

    def _redock(self, key: str):
        """Re-dock button: return the floated widget to the tab bar — the SAME object, so a live
        CLI keeps its shell and history (close-and-reopen would kill both)."""
        win = self._floating.pop(key, None)
        if win is None:
            return
        title = win._tab_title
        # `is None`, never `or`: an EMPTY QTabWidget is falsy in PyQt, so `_home_tabs or _left_tabs`
        # sent every re-dock from a just-emptied pane 3 into the process console instead.
        tabs = getattr(win, "_home_tabs", None)
        if tabs is None:
            tabs = self._left_tabs                           # back to the bar it came from
        w = win.take_content()                             # empties the window: its close is plain
        win.close()
        win.deleteLater()
        if w is None:
            return
        self._op_tabs[key] = w
        tabs.addTab(w, title)
        self._sync_explore_pane()
        tabs.setCurrentWidget(w)

    def _discard_exploration(self, tab: "_ExplorationTab"):
        """Tear down one exploration tab's work: stop its run if it owns the live one, then drop
        every layer it produced and FREE the plate canvases behind them.

        Without this the worker keeps computing into a layer nobody can reach, and each abandoned
        layer keeps a full plate-sized RGB canvas resident (tens of MB on a 1536wp) — silent
        growth on the app's headline gesture, with no error anywhere."""
        stopped = False
        if tab.tab_key in (self._run_tab_key, self._run_view_tab_key) and self._busy():
            self._stop_worker()          # _retire: disconnects signals, then lets the thread drain
            self._note_partial_output()  # a stopped SAVE run leaves a half-written .hcs on disk
            self._run_tab_key = self._run_view_tab_key = None
            stopped = True
        if self._active_exploration is tab:
            # BUG 1: the tab in front is being deleted. Leaving _active_exploration pointing at it
            # strands the whole view — _on_tab_changed would later park status onto a dead widget,
            # and _push_index / the FOV slider stay scoped to a subset nobody can see. Drop the ref
            # NOW and ask for a re-sync; if a run is still draining, _on_run_drained does it once
            # the thread is actually gone (a stopped run keeps _busy() True for a while, which is
            # exactly why the deferred path exists).
            self._active_exploration = None
            self._request_resync()
        gone = self._op_stack.remove_suffix(f"@{tab.tab_key}")
        if self._overview is not None:
            for layer in gone:
                self._overview.drop_layer(layer)
        if self._active_op_key in gone:
            self._active_op_key = None
            self._plate_mode = "raw"
            if self._acq_name:
                self._plate_title.setText(f"{self._acq_name}   ·   raw")
        self._refresh_layers_tab()
        tab.dispose()                    # free the tab's OWN viewer + stop its mosaic read
        if stopped:
            self._readout.setText(f"stopped {exploration_tab_label(tab.regions)} — tab closed mid-run")

    def _note_partial_output(self):
        """A save run stopped mid-write leaves a partial `.hcs`. Drop an INCOMPLETE marker in it so
        a later 'Open a computed MIP…' can refuse it instead of presenting a truncated plate as a
        finished one (resolve_plate_root only looks for plate.ome.zarr, which a partial still has)."""
        out = self._run_out_dir
        self._run_out_dir = None
        if not out:
            return
        try:
            p = Path(out)
            if p.exists():
                (p / "INCOMPLETE").write_text(
                    "This plate was stopped mid-write and is NOT complete.\n"
                    "Re-run the operator to produce a full plate.\n")
        except OSError:
            pass       # best-effort: never let cleanup bookkeeping break teardown

    def _close_exploration_tabs(self):
        """Close every exploration tab. Called on ingest: a tab's regions belong to the acquisition
        it was opened from, and _fov_index is about to be rebuilt for a different plate.

        Muted: each removeTab emits currentChanged, and letting _on_tab_changed re-point the detail
        at the OUTGOING acquisition mid-teardown is pure waste (ingest rebuilds it all anyway)."""
        self._tabs_muted = True
        try:
            for i in range(self._explore_tabs.count() - 1, -1, -1):
                if isinstance(self._explore_tabs.widget(i), _ExplorationTab):
                    self._close_op_tab(i, self._explore_tabs)
            # ...and the ones dragged out into floating windows (IMA-209). A float is off the tab
            # bar but NOT off the plate: it still owns layers and can still own the live run.
            for key, win in list(self._floating.items()):
                if isinstance(win.content(), _ExplorationTab):
                    self._floating.pop(key, None)
                    w = win.take_content()
                    win.close()
                    win.deleteLater()
                    if w is not None:
                        self._dispose_tab_widget(w)
        finally:
            self._tabs_muted = False

    def open_exploration_tab(self, regions) -> Optional[str]:
        """Open (or focus) the exploration tab for ``regions``. Returns its key, or None.

        The UI entry point is IMA-221's Shift-drag marquee, via ``_on_marquee_selected``; it is also
        callable programmatically (and by tests). Identity is content-addressed, so dragging the
        same wells twice focuses the existing tab rather than opening a duplicate."""
        if self._reader is None or self._overview is None:
            self._readout.setText("open an acquisition first")
            return None
        regions = list(dict.fromkeys(regions))            # de-dupe, keep first-seen order
        if not regions:
            self._readout.setText("empty selection — nothing to explore")
            return None
        unknown = [r for r in regions if r not in self._fov_index]
        if unknown:
            self._readout.setText(f"{len(unknown)} region(s) are not in this acquisition: {unknown[:3]}")
            return None
        key = exploration_tab_key(self._acq_name, regions)
        # PANE 3 (IMA-237), not the process console: the Shift-drag that opens this tab is also what
        # REVEALS the exploration pane, which is why it is the gesture and not a menu item.
        self._open_op_tab(key, exploration_tab_label(regions),
                          lambda: self._build_exploration_tab(regions, key),
                          tabs=self._explore_tabs)
        return key

    def _open_preview_tab(self, op_key: str, op_label: str, regions) -> Optional[str]:
        """Open (or focus) the side-pane tab a preview run of ``op_key`` streams its results into.

        Identity is content-addressed on acquisition + OPERATOR + region set, so two preview runs
        over one selection are two tabs side by side — which is the point: "preview runs can open
        a tab on the exploration pane so that they look at how it is behaving." Re-running the
        SAME operator on the SAME wells reuses its tab rather than accumulating duplicates.
        """
        key = _explore.preview_tab_key(self._acq_name, op_key, regions)
        self._open_op_tab(key, _explore.preview_tab_label(op_label, regions),
                          lambda: self._build_exploration_tab(regions, key),
                          tabs=self._explore_tabs)
        return key

    def _run_tab(self) -> Optional["_ExplorationTab"]:
        """The side-pane tab the in-flight run is streaming into, if any."""
        if not self._run_view_tab_key:
            return None
        w = self._op_tabs.get(self._run_view_tab_key)
        return w if isinstance(w, _ExplorationTab) else None

    def _on_progress(self, done: int, total: int):
        """A run advanced. Says so in pane 1's status line AND, when the run belongs to a side-pane
        tab, in that tab — where the user is actually watching the result appear."""
        self._run_readout(f"● {self._run_label} · {done}/{total} wells{self._run_dest}")
        # Feed the activity registry the log panel's header reads — this is what turns "the GUI is
        # doing something" into a visible line. Advanced from THIS slot (the GUI thread), never
        # from the worker: the panel writes a QLabel and a worker thread must not.
        self._activity.advance("operator-run", done, total)
        tab = self._run_tab()
        if tab is not None and tab.progress is not None:
            tab.progress.setText(_explore.progress_sentence(self._run_label, done, total))

    def _on_run_tile(self, ri, ci, well_id, tile, box=None):
        """One computed FIELD landed — put it on the run's side-pane tab as a REAL LAYER.

        Julio: "layers don't update in the napari mosaic... you instantiate an actual layer to be
        in the napari interface." So each region of the run becomes its own layer group the moment
        its first field arrives, and later fields of that region update it — rather than the tab
        sitting empty until the run ends and then being handed finished data.

        A field for a region this tab is not scoped to is DROPPED: the tab claims a subset, and
        painting a foreign region on it would make that claim false. (It cannot normally happen —
        the run and the tab have the same region list — but the tab's claim is not left to luck.)
        """
        tab = self._run_tab()
        if tab is None or tab.viewer is None or self._meta is None:
            return
        region = next((r for r in tab.regions
                       if tuple(self._fov_index[r]["rc"]) == (ri, ci)), None)
        if region is None:
            return
        from squidmip._mosaic_source import mosaic_bbox_um
        from squidmip._napari_pane import _colormap_for

        arr = np.asarray(tile)
        if box is not None:
            # A multi-FOV region arrives field by field, each with its box inside the region's
            # cell. Accumulate into ONE canvas per region so the layer fills in as the run walks
            # the region, instead of one layer per field (36 FOVs x 4 channels = 144 layers).
            canvas = tab.tiles.get(region)
            if canvas is None or canvas.shape[0] != arr.shape[0]:
                canvas = np.zeros((arr.shape[0], _CELL, _CELL), arr.dtype)
                tab.tiles[region] = canvas
            top, left, bh, bw = box
            canvas[:, top:top + bh, left:left + bw] = arr[:, :bh, :bw]
            arr = canvas
        try:
            bbox = mosaic_bbox_um(self._meta, region)
        except Exception as exc:                     # noqa: BLE001 - said, never swallowed
            tab.viewer.say(f"{region}: could not place the result ({exc}); showing it unplaced.")
            bbox = None
        op = _explore.subset_layer_op(self._run_label, region)
        for c_i, channel in enumerate(c["name"] for c in self._meta["channels"]):
            if c_i >= arr.shape[0]:
                break
            tab.viewer.mosaic.add_mosaic(
                op, channel, arr[c_i],
                colormap=_colormap_for(channel),
                bbox_um=bbox,
            )

    def _current_exploration(self) -> Optional["_ExplorationTab"]:
        """The exploration tab the plate and viewer follow: pane 3's FRONT tab, or None when pane 3
        is empty (IMA-237).

        Before pane 3 existed, "which tab is in front" was a single question with a single answer,
        because exploration tabs shared the process console's bar. Now the console and pane 3 are
        side by side and both are visible at once, so scope is owned by pane 3 alone — opening the
        Layers tab in pane 1 must not silently un-scope the viewer beside it."""
        if self._explore_tabs.count() == 0:
            return None
        w = self._explore_tabs.currentWidget()
        return w if isinstance(w, _ExplorationTab) else None

    def _on_tab_changed(self, index: int = -1, force: bool = False):
        """The plate + detail follow the ACTIVE tab (IMA-205).

        An exploration tab claims to be scoped to its subset, so the plate's status dots and the
        detail's FOV slider have to agree with it — otherwise the tab says '4 wells' while the
        viewer beside it lists all 1536, and scrubbing lands on wells the tab never selected.

        A LIVE run is the one thing we won't retarget under: the worker is pushing into the slider
        this call would rebuild. So the switch is DEFERRED, not dropped (``_request_resync``) —
        dropping it is what left the front tab lying about what the viewer shows (BUG 2), because
        nothing re-emits ``currentChanged`` when the run later drains.

        ``force=True`` re-runs the sync from ``_on_run_drained`` even when there is no outgoing
        exploration tab to park — after a mid-run tab close there ISN'T one, and that is precisely
        the case that has to fall back to the whole plate (BUG 1).

        Honest limitation: ndviewer's only retarget seam is ``start_acquisition``, which RESETS the
        viewer. Computed frames pushed via register_array are in-memory and do not survive the
        switch; we re-register the subset's RAW plane paths (cheap — paths only) so the pane shows
        real imagery rather than black. Re-run the operator in the tab to recompute its frames."""
        if self._reader is None or self._overview is None or self._tabs_muted:
            return
        if _explore.operator_busy(self._worker, self._retired):
            # Defer for an OPERATOR RUN only. Never for the raw preview: `_setup_raw_detail`
            # re-scopes and restarts the preview itself, so a streaming preview is not a reason
            # to postpone -- and postponing on it is what stranded the restore. Closing a tab
            # while the preview streamed (which is most of the time on a real plate) left the
            # viewer scoped to a subset whose tab no longer existed, until some unrelated thread
            # happened to exit. See _explore.operator_busy: this is the third gate that was
            # asking "is any producer alive" when the question is "is a RUN alive".
            self._request_resync()   # never retarget the slider a live run is pushing into — LATER
            return
        w = self._current_exploration()      # pane 3 owns scope now — not the index we were handed
        prev = self._active_exploration
        if prev is not None and self._overview is not None:
            prev.status = self._overview.status_snapshot()      # park the outgoing tab's dots
        if w is not None:
            self._active_exploration = w
            self._setup_raw_detail(order=w.regions)
            self._overview.set_all_status("empty")
            self._overview.set_status_map(w.status)
            top = next((ly.key for ly in reversed(self._op_stack.layers())
                        if ly.key.endswith(f"@{w.tab_key}")), None)
            # A PREVIEW tab owns no plate layer keyed to itself — its run's results are filed
            # under the plate-wide key on purpose (see run_operator), because the tab shows them
            # in its OWN viewer. Falling straight to "raw" for it would flip the plate back to
            # the raw preview the instant a preview run opened its tab: running an operator would
            # visibly UNDO itself on the plate. ``plate_layer`` is the layer a run displayed in
            # THIS tab wrote into, so the tab can name it instead of the window having to
            # remember which run is whose. A tab that never hosted a run has None and keeps the
            # historical `top or "raw"`.
            self._overview.set_active_layer(top or w.plate_layer or "raw")
            w.set_sync_pending(False)
            # NB: do NOT reset _push_index here — _setup_raw_detail just built the subset map for
            # this tab's slider, and clearing it would send register_image straight back to global
            # plate indices (the exact off-by-a-lot this whole path exists to prevent).
        else:
            if prev is None and not force:
                return                   # home -> operator tab: the plate is already plate-wide
            self._active_exploration = None
            self._setup_raw_detail(order=None)
            self._overview.set_all_status("empty")
            self._overview.set_active_layer(self._active_op_key or "raw")

    def _request_resync(self):
        """Remember that the plate/detail need to catch up with the front tab once the run drains.

        Both IMA-205 bugs are the same missing edge: a tab switch that arrives while a run is live
        is silently discarded, and no later event re-delivers it. The pending flag IS that later
        event; ``_on_run_drained`` fires it as soon as the last worker thread actually exits."""
        self._pending_resync = True
        w = self._current_exploration()
        if w is not None:
            # say so IN THE TAB rather than in _readout: the run's progress writes _readout on every
            # well, so a note there would be gone before the user could read it.
            w.set_sync_pending(True)
        if not _explore.operator_busy(self._worker, self._retired):
            # NOTHING IS RUNNING, SO NOTHING WILL EVER DELIVER THIS. `_on_run_drained` is the only
            # other caller, and it fires on QThread.finished -- so with no live thread the flag was
            # set and then sat there forever. Closing an exploration tab on an idle window left the
            # viewer scoped to the subset of a tab that no longer exists: the plate came back with
            # ['B3:0'] instead of ['B2:0', 'B3:0'], one well silently missing.
            #
            # It looked like a flake (~50% in isolation) because the RAW PREVIEW worker is usually
            # still streaming when a tab is closed by hand. When it was, its finish delivered the
            # resync and everything worked; when it had already finished, the restore was lost.
            # The bug was never in the timing -- deferral is simply only correct while something is
            # running.
            #
            # Delivered on the event loop rather than inline: this is called from the middle of tab
            # DISPOSAL, and re-entering _on_tab_changed there would rescope against a half-torn-down
            # tab. A zero timer runs after the current stack unwinds, and processEvents() delivers
            # it, so it stays deterministic for the tests too.
            QTimer.singleShot(0, self._deliver_pending_resync)

    def _deliver_pending_resync(self):
        """Deliver a deferred tab switch. Idempotent, and re-defers if a run started meanwhile."""
        if not self._pending_resync or _explore.operator_busy(self._worker, self._retired):
            return
        self._pending_resync = False
        self._on_tab_changed(force=True)

    def _on_run_drained(self):
        """A worker thread has exited. Deliver any tab switch that was deferred while it ran.

        Fires on QThread.finished, so it also covers a run that was STOPPED (closing a tab mid-run)
        — ``_stop_worker`` returns immediately but the thread keeps going until its current well is
        done, and ``_busy()`` stays True for all of that window."""
        if _explore.operator_busy(self._worker, self._retired):
            return                       # another operator run is still draining — wait for it
        # No operator run is in flight now — clear the activity header. end() is a no-op if it was
        # already cleared, so a failed/stopped run that never reached here does not leave it stuck.
        self._activity.end("operator-run")
        self._run_tab_key = self._run_view_tab_key = None
        # A genuine drain: every worker has exited AND (finished being FIFO-queued after this
        # worker's tileReady/streamEnded) their terminal slots have already run on this thread.
        # Bump BEFORE the pending-resync branch so a run with no deferred switch still counts.
        self._runs_settled += 1
        if not self._pending_resync:
            return
        self._pending_resync = False
        self._on_tab_changed(force=True)

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

    def _build_stitch_tab(self) -> QWidget:
        """maragall/stitcher's control surface, in pane 1 (IMA-decon-stitch-ui).

        This used to be `_build_run_tab` -- a destination picker and a "first N wells"
        spinner, with NO stitcher controls at all. Julio: "Right now I'm blocked in testing
        the post-processing because Stitcher doesn't have that maragall/Stitcher interface
        embedded in our top-left subpane." What a user tunes on a registration/fusion run
        (registration on/off, registration channel, feather width, blunder thresholds,
        which channels to fuse) now lives in `_op_panels.StitcherPanel` and travels to both
        the preview and the saved run through `operator_kwargs`.
        """
        from squidmip._op_panels import StitcherPanel

        return StitcherPanel(self)

    def _build_decon_tab(self) -> QWidget:
        """The RL semi-convergence loop's controls (IMA-252 + IMA-decon-stitch-ui).

        The controls are here in pane 1; the picture they produce -- the deconvolved 2-D
        image in turbo with the x-z and y-z strips concatenated -- opens as a tab in PANE 3
        via :meth:`publish_qc_result`. It was `_build_plane_op_tab` (a preview button and
        nothing else), which gave no way to choose an iteration count at all.
        """
        from squidmip._op_panels import DeconQCPanel

        return DeconQCPanel(self)

    # -- the host surface the pane-1 operator panels use -----------------------------------
    #
    # Deliberately three small methods rather than handing a panel the whole window: if a
    # panel starts needing more than this, that is a coupling worth seeing in a diff.

    def say(self, text: str) -> None:
        """Put an operator panel's sentence in the window's status line."""
        if text:
            self._run_readout(text)

    def explore_scopes(self) -> list:
        """``[(label, regions), ...]`` for every subset currently parked in pane 3.

        These become SCOPE VALUES on the pane-1 panels, not buttons over in pane 3. A UI
        audit found two operator registries launching the same operators from panes 1 and 3
        with different labels and different `save` defaults, and they had already diverged
        in production; a third caller would have made that worse rather than better.
        """
        scopes = []
        for i in range(self._explore_tabs.count()):
            w = self._explore_tabs.widget(i)
            if isinstance(w, _ExplorationTab):
                scopes.append((exploration_tab_label(w.regions), list(w.regions)))
        for win in self._floating.values():                # detached tabs count too
            w = win.content()
            if isinstance(w, _ExplorationTab):
                scopes.append((exploration_tab_label(w.regions), list(w.regions)))
        return scopes

    def publish_qc_result(self, widget: QWidget, title: str) -> None:
        """Show *widget* as a result tab in PANE 3.

        THE seam between the pane-1 controls and pane 3. It is deliberately one method wide
        and it introduces no new tab machinery: `_open_op_tab` with `tabs=self._explore_tabs`
        is exactly how exploration tabs already get there, so the pane-3 owner has nothing
        to merge. Keyed by title so re-running the same subject reuses its tab instead of
        stacking a new one per iteration.
        """
        self._open_op_tab(f"qc:{title}", title, lambda w=widget: w, tabs=self._explore_tabs)

    def _build_bgsub_tab(self) -> QWidget:
        return self._build_plane_op_tab(_OPERATIONS_BY_KEY["bgsub"])

    def _build_flatfield_tab(self) -> QWidget:
        # The one plane-op that cannot run without an argument: a flat-field with no illumination
        # profile has no sane default (an identity field would silently do nothing while the UI
        # said "corrected"), so the operator raises until one is loaded. The chooser is that load.
        return self._build_plane_op_tab(_OPERATIONS_BY_KEY["flatfield"], profile_chooser=True)

    def _build_plane_op_tab(self, op, profile_chooser: bool = False) -> QWidget:
        """Generic PLANE-OP tab (IMA-223/224/225): preview on a subset, never save.

        A plane-op maps plane -> plane and does NOT consume z (IMA-210), so its output keeps the
        z-stack at full depth -- and write_plate's _validate_image accepts Z == 1 only. So this
        builder deliberately omits the "Run on the whole plate" / destination half of
        _build_run_tab: there is nothing to write yet. The moment the OME-Zarr writer learns
        Z > 1, this method can simply forward to _build_run_tab and disappear.

        The preview path itself is unchanged and needs no worker edit: _OperatorWorker's save=False
        branch streams project_plate, and _on_well already indexes image[0, :, 0] -- for a plane-op
        that is the FIRST z-plane, corrected, which is exactly what a preview should show.
        """
        w, v = self._op_tab_shell(op.label, op.blurb)
        v.addWidget(_hline())

        state = {"profile": None}
        if profile_chooser:
            prof_lbl = QLabel("(no illumination profile loaded)")
            prof_lbl.setWordWrap(True)
            prof_lbl.setStyleSheet("color:#8b98ad;font-size:12px;")

            def load_profile():
                path, _ = QFileDialog.getOpenFileName(
                    self, "Load illumination profile", "", "Illumination profile (*.npy)")
                if not path:
                    return
                from squidmip import FlatfieldProfile
                from squidmip._flatfield import set_profile
                try:
                    profile = FlatfieldProfile.from_npy(path)
                except Exception as exc:                     # bad file -> say so, keep the tab alive
                    prof_lbl.setText(f"could not load {Path(path).name}: {exc}")
                    return
                frame = tuple(self._reader.metadata["frame_shape"]) if self._reader else None
                if frame is not None and profile.shape != frame:
                    prof_lbl.setText(f"profile is {profile.shape}, this acquisition's frames are "
                                     f"{frame} -- wrong profile for this plate")
                    return
                set_profile(profile)
                state["profile"] = path
                prof_lbl.setText(f"{Path(path).name}  {profile.shape}")
                prev.setEnabled(True)

            pick_prof = QPushButton("Load illumination profile (.npy)…")
            pick_prof.setStyleSheet(_BTN_QSS)
            pick_prof.clicked.connect(load_profile)
            v.addWidget(pick_prof)

            # ESTIMATE LIVE from the plate (maragall/stitcher's tilefusion BaSiC), no .npy needed.
            # Julio: flat-field computation comes from maragall/stitcher and must run from tiles.
            est_row = QHBoxLayout(); est_row.setSpacing(6)
            est_row.addWidget(QLabel("channel"))
            est_channel = QComboBox(); est_channel.setStyleSheet(_COMBO_QSS)
            est_channel.addItems([c["name"] for c in (self._meta or {}).get("channels", [])])
            est_row.addWidget(est_channel, 1)
            est_row.addWidget(QLabel("tiles"))
            est_tiles = QSpinBox(); est_tiles.setRange(3, 256); est_tiles.setValue(48)
            est_tiles.setStyleSheet(_COMBO_QSS)
            est_row.addWidget(est_tiles)
            v.addLayout(est_row)

            est_btn = QPushButton("Estimate from plate")
            est_btn.setStyleSheet(_BTN_QSS)
            est_btn.setToolTip("Estimate the illumination profile LIVE from a spread of plate tiles "
                               "with the stitcher's BaSiC estimator (tilefusion). No .npy required.")

            def estimate_from_plate():
                if self._reader is None or self._meta is None:
                    prof_lbl.setText("no acquisition open to estimate a flat-field from.")
                    return
                ch = est_channel.currentText()
                est_btn.setEnabled(False)
                prof_lbl.setText(f"estimating illumination for {ch} from the plate…")
                w = _FlatfieldWorker(self._reader, self._meta, ch,
                                     max_tiles=est_tiles.value(), parent=self)

                def _ok(profile):
                    from squidmip._flatfield import set_profile
                    set_profile(profile)
                    state["profile"] = f"estimated:{ch}"
                    prof_lbl.setText(f"estimated from plate ({ch})  {profile.shape}")
                    prev.setEnabled(True)
                    est_btn.setEnabled(True)

                def _bad(msg):
                    prof_lbl.setText(str(msg))
                    est_btn.setEnabled(True)

                w.done.connect(_ok)
                w.problem.connect(_bad)
                w.stage.connect(lambda s: prof_lbl.setText(str(s)))
                self._flatfield_worker = w            # keep a ref so it is not GC'd mid-run
                w.start()

            est_btn.clicked.connect(estimate_from_plate)
            v.addWidget(est_btn)
            v.addWidget(prof_lbl)
            v.addWidget(_hline())

        prev_lbl = QLabel("Preview (subset)")
        prev_lbl.setStyleSheet("color:#57606a;font-size:10px;font-weight:800;letter-spacing:1.5px;padding-top:6px;")
        v.addWidget(prev_lbl)
        n_wells = max(1, len(self._order))
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(QLabel("First"))
        spin = QSpinBox(); spin.setRange(1, n_wells); spin.setValue(min(4, n_wells))
        spin.setStyleSheet(_COMBO_QSS)
        row.addWidget(spin); row.addWidget(QLabel("wells")); row.addStretch(1)
        v.addLayout(row)

        prev = QPushButton("Preview"); prev.setStyleSheet(_BTN_QSS)
        prev.setEnabled(not profile_chooser)          # flat-field waits for its profile
        prev.clicked.connect(
            lambda: self.run_operator(op.key, out_parent=None,
                                      regions=self._order[:spin.value()], save=False))
        v.addWidget(prev)

        note = QLabel("Preview only: this operator keeps the z-stack at full depth, and the "
                      "OME-Zarr writer accepts one z per field today, so there is nothing to "
                      "save yet. The raw acquisition is never modified.")
        note.setWordWrap(True); note.setStyleSheet("color:#8b98ad;font-size:11px;")
        v.addWidget(note)
        v.addStretch(1)
        return w

    def _build_run_tab(self, op) -> QWidget:
        """Generic projector-operator tab (MIP, …): pick a destination, run over the whole plate → a
        navigable OME-Zarr plate. ONE builder for every z-reduction operator — a new one needs no new
        tab code. Per-tab state lives in a closure (no per-operator instance attrs)."""
        w, v = self._op_tab_shell(op.label, op.blurb + " Pick a destination with room — output can be large.")
        state = {"dir": None}
        dir_lbl = QLabel("(no folder chosen)"); dir_lbl.setWordWrap(True)
        dir_lbl.setStyleSheet("color:#8b98ad;font-size:12px;")
        run = QPushButton("Run"); run.setStyleSheet(_BTN_QSS); run.setEnabled(False)

        # RUN ON — the target the operator iterates over (Julio: the per-tool "run on" choice, not a
        # master-pane one). The decentralized model adds OPEN VIEWS: run the operator over the
        # regions currently held by the independent windows, not just the plate selection.
        TARGET_PLATE, TARGET_SELECTION, TARGET_OPEN = "Whole plate", "Selected wells", "Open views"
        run_row = QHBoxLayout(); run_row.setSpacing(6)
        _rl = QLabel("Run on"); _rl.setStyleSheet("color:#8b98ad;font-size:12px;")
        target = QComboBox(); target.setStyleSheet(_COMBO_QSS)
        target.addItems([TARGET_SELECTION, TARGET_OPEN, TARGET_PLATE])
        target.setToolTip(
            "What the operator iterates over.\n"
            f"{TARGET_SELECTION} — the wells picked on the plate (all if none).\n"
            f"{TARGET_OPEN} — every region held by the open viewer windows.\n"
            f"{TARGET_PLATE} — every region of the acquisition.")
        run_row.addWidget(_rl); run_row.addWidget(target, 1)

        def pick():
            d = QFileDialog.getExistingDirectory(self, f"Save {op.label} plate to folder")
            if not d:
                return
            state["dir"] = d
            ok, est_gb, _ = self._check_disk(Path(d) / f"{self._acq_name}.hcs")
            dir_lbl.setText(f"{d}\n~{est_gb:.0f} GB needed" + ("" if ok else "  (not enough free space)"))
            run.setEnabled(True)

        pick_btn = QPushButton("Choose output folder…"); pick_btn.setStyleSheet(_BTN_QSS)
        pick_btn.clicked.connect(pick)

        def do_run():
            choice = target.currentText()
            if choice == TARGET_PLATE:
                regions = None                       # None = whole dataset (run_operator's contract)
            elif choice == TARGET_OPEN:
                regions = self._open_views_regions()
                if not regions:
                    self._readout.setText("Run on open views: no windows are open — open some first.")
                    return
            else:                                    # selected wells (all if none selected)
                regions = self._selected_regions or None
            self.run_operator(op.key, out_parent=state["dir"], regions=regions)

        v.addWidget(_hline())
        run.clicked.connect(do_run)
        v.addLayout(run_row)
        v.addWidget(pick_btn); v.addWidget(dir_lbl); v.addWidget(run)

        # PREVIEW on a subset — test the operator on the first N wells without committing the whole
        # plate's compute + disk. Default: don't save (compute + push to the viewer only).
        v.addWidget(_hline())
        prev_lbl = QLabel("Preview (subset)")
        prev_lbl.setStyleSheet("color:#57606a;font-size:10px;font-weight:800;letter-spacing:1.5px;padding-top:6px;")
        v.addWidget(prev_lbl)
        n_wells = max(1, len(self._order))
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(QLabel("First"))
        spin = QSpinBox(); spin.setRange(1, n_wells); spin.setValue(min(4, n_wells))
        spin.setStyleSheet(_COMBO_QSS)
        row.addWidget(spin); row.addWidget(QLabel("wells")); row.addStretch(1)
        v.addLayout(row)
        save_cb = QCheckBox("Save previews to disk"); save_cb.setStyleSheet(_CHECK_QSS)
        v.addWidget(save_cb)
        prev = QPushButton("Preview"); prev.setStyleSheet(_BTN_QSS); prev.setEnabled(False)

        def do_preview():
            save = save_cb.isChecked()
            dest = None
            if save:
                dest = state["dir"] or QFileDialog.getExistingDirectory(self, f"Save {op.label} preview to folder")
                if not dest:
                    return
            # "first N wells" is just one way to build a region list, so the prefix policy lives
            # here (in the UI that owns the spinner) rather than as a second subset parameter.
            self.run_operator(op.key, out_parent=dest, regions=self._order[:spin.value()], save=save)

        prev.clicked.connect(do_preview)
        v.addWidget(prev)
        v.addStretch(1)
        # both run buttons enable once an acquisition is open (the tab is only reachable then, but be safe)
        for b in (run, prev):
            b.setEnabled(self._reader is not None)
        return w

    def _make_explore_viewer(self):
        """Build a viewer for ONE side-pane tab. Returns ``(pane_or_None, mode, message)``.

        DELEGATES to pane 2's constructor. The right pane is "a copy of the central pane, but it
        occurs on a subset", so a second viewer implementation here would be exactly the
        duplication this project keeps failing on — and it would also be a second embedding
        path, which is how the control well ended up in a floating window: ``napari.Viewer``
        builds a real QMainWindow, and one that nobody reparents IS a top-level window the
        moment anything shows it. ``MosaicPane._embed_native_window`` is the one place that
        knows to reparent the WINDOW (never the canvas, which is that window's central widget).

        It exists as a named method purely so tests can swap in a recording stub: napari's
        canvas needs OpenGL and the headless gate has none.
        """
        return _make_mosaic_pane(show_docks=False)

    def _build_exploration_tab(self, regions: list, tab_key: str) -> QWidget:
        """One side-pane tab: A VIEWER ON THIS SUBSET, a slider under it, and the Minerva hand-off.

        This tab is a RESULT SURFACE, not a control surface. Julio: "we have the controls for the
        whole dataset on the left, but those controls are repeated for the subset on the right
        pane. Maybe it's not a good idea for there to be repetition of knowledge in our user
        interface" — and "this is just a supplementary pane that augments the processing by
        showing preview results and how that reflects on our viewer."

        So the per-operator preview buttons that used to live here are GONE. They were a second
        operator catalogue (``runnable_operators()``) beside pane 1's (``_OPERATIONS``), with
        different labels and a different ``save`` default, and the comment they carried recorded
        that the two had already drifted in production. Running an operator on this subset is now
        a SCOPE on pane 1's one control panel (``_explore.SCOPE_SUBSET``), which reads the subset
        this pane owns.

        Minerva stays, because it is not an operator: it is an export of WHAT IS DISPLAYED HERE.
        """
        w = _ExplorationTab(regions, tab_key)
        regions = w.regions
        w.setStyleSheet(f"background:{_BG};color:#e6edf3;")
        v = QVBoxLayout(w)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(6)

        # -- what this tab is scoped to. One compact line: the pane is a fifth of a SMALL monitor,
        # and chrome that eclipses the viewer is the complaint this layout is answering.
        listing = QLabel(", ".join(regions))       # the tab must LIST exactly what it is scoped to
        listing.setWordWrap(True)
        listing.setStyleSheet(f"color:#c3ccd9;font-size:{_EMPTY_BODY_PX - 1}px;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        scroll.setWidget(listing)
        scroll.setMaximumHeight(46)
        v.addWidget(scroll)
        w.listing = listing                        # tests assert the tab lists exactly its regions

        note = QLabel("A run is still finishing — the plate and viewer beside this tab still show "
                      "it. They will switch to this subset when it is done.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:#d29922;font-size:{_EMPTY_BODY_PX - 1}px;")
        note.setVisible(False)
        v.addWidget(note)
        w.sync_note = note
        w.set_sync_pending(w.sync_pending)

        # -- THE VIEWER. Same constructor as pane 2, embedded here, never a separate window.
        pane, _mode, msg = self._make_explore_viewer()
        if pane is not None:
            w.viewer = pane
            pane.setParent(w)
            v.addWidget(pane, 1)
        else:
            # NO SILENT FAILURE. A tab with no viewer is not "a tab with fewer features", it is a
            # pane that cannot do its job, and the user has to be told which and why.
            dead = QLabel(msg or "no viewer could be built for this subset.")
            dead.setWordWrap(True)
            dead.setAlignment(Qt.AlignCenter)
            dead.setStyleSheet(
                f"color:#ffd7d7;background:#3a2020;padding:10px;font-size:{_EMPTY_BODY_PX}px;")
            v.addWidget(dead, 1)

        # -- THE SLIDER UNDER IT. "There should be a slider under." One stop per region of the
        # subset: the unit this pane shows is a REGION (a mosaic of FOVs), never a single field.
        w.region_label = QLabel("")
        w.region_label.setStyleSheet(f"color:#c3ccd9;font-size:{_EMPTY_BODY_PX - 1}px;")
        v.addWidget(w.region_label)
        w.slider = QSlider(Qt.Horizontal)
        w.slider.setMinimum(0)
        w.slider.setMaximum(max(0, len(regions) - 1))
        w.slider.setEnabled(len(regions) > 1)
        w.slider.setStyleSheet(_NDV_DARK)
        w.slider.valueChanged.connect(lambda i, t=w: self._on_explore_slider(t, i))
        v.addWidget(w.slider)
        save_btn = QPushButton("Save this subset to disk…")
        save_btn.setStyleSheet(_BTN_QSS)
        save_btn.clicked.connect(
            lambda: self.run_operator(_SAVE_OPERATOR, regions=regions, save=True, tab_key=tab_key))
        v.addWidget(save_btn)

        # -- what a preview run scoped to this tab has computed so far.
        w.progress = QLabel("")
        w.progress.setWordWrap(True)
        w.progress.setStyleSheet(f"color:#8b98ad;font-size:{_EMPTY_BODY_PX - 1}px;")
        v.addWidget(w.progress)

        minerva = QPushButton("Open in Minerva Author")
        minerva.setStyleSheet(_BTN_QSS)
        minerva.setCursor(Qt.PointingHandCursor)
        minerva.setToolTip(
            "Fuse each region of this subset into one mosaic, write it as an OME-TIFF plus a "
            "Minerva story, and start Minerva Author on it.")
        minerva.clicked.connect(lambda _=False, t=w: self._export_subset_to_minerva(t))
        v.addWidget(minerva)
        w.minerva_btn = minerva

        self._sync_explore_region(w)
        self._load_explore_region(w)
        return w

    # -- the side pane's viewer: aim it at one region of its subset --------------------------------
    def _sync_explore_region(self, tab: "_ExplorationTab"):
        """Make the tab's label agree with its cursor. The cursor is the only owner of 'which
        region is in front'; the label and the slider are both told from it."""
        if tab.region_label is None:
            return
        n = len(tab.cursor)
        tab.region_label.setText(
            f"region {tab.cursor.index + 1} of {n} · {tab.cursor.region}")

    def _on_explore_slider(self, tab: "_ExplorationTab", index: int):
        """The slider under a side-pane viewer moved."""
        if not tab.cursor.set_index(index):
            return                       # no move: do not restart a mosaic read on a stray event
        self._sync_explore_region(tab)
        self._load_explore_region(tab)

    def _load_explore_region(self, tab: "_ExplorationTab"):
        """Fuse the cursor's region and put it on THIS TAB's viewer, one layer per channel.

        The same ``_MosaicWorker`` pane 2 uses — a region is a mosaic of FOVs and there is one
        implementation of assembling it. Already-loaded regions stay on the canvas: the tab
        accumulates its subset as layers, so scrubbing back is instant and the pane keeps
        showing what was selected rather than emptying itself.
        """
        if tab.viewer is None or self._reader is None or self._meta is None:
            return
        region = tab.cursor.region
        if tab.cursor.is_loaded(region):
            return
        prior = tab.mosaic_worker
        if prior is not None and prior.isRunning():
            prior.stop()
            prior.wait(2000)
        tab.viewer.say(f"loading {region} …")
        channels = [c["name"] for c in self._meta["channels"]]
        op = _explore.subset_layer_op("raw", region)
        wk = _MosaicWorker(self._reader, self._meta, region, channels, parent=self)
        wk.ready.connect(
            lambda r, ch, plane, bbox, t=tab, o=op: self._on_explore_plane(t, o, r, ch, plane, bbox))
        wk.problem.connect(lambda m, t=tab: t.viewer.say(m) if t.viewer is not None else None)
        wk.finished_count.connect(
            lambda n, t=tab, r=region: self._on_explore_region_done(t, r, n))
        tab.mosaic_worker = wk
        wk.start()

    def _on_explore_plane(self, tab, op, region, channel, levels, bbox_um):
        """Same contract as ``_on_mosaic_plane``: ``_MosaicWorker`` emits a LAZY PYRAMID.

        ``levels`` is the list napari's ``multiscale=True`` wants, highest resolution first. The
        side pane is a copy of pane 2 on a subset, so it gets the same pyramid on the same terms —
        a tab that took level 0 would put the full 5731x4793 mosaic on screen per region and undo
        the memory win exactly where the user opens the most viewers.
        """
        if tab.viewer is None:
            return
        from squidmip._napari_pane import _colormap_for

        tab.viewer.mosaic.add_mosaic(
            op, channel, levels,
            colormap=_colormap_for(channel),
            multiscale=True,
            bbox_um=bbox_um,
            z_scale_um=(self._meta or {}).get("dz_um"),
        )

    def _on_explore_region_done(self, tab, region, n):
        if tab.viewer is None:
            return
        if n == 0:
            tab.viewer.say(f"{region}: no mosaic could be built (see the message above).")
            return
        tab.cursor.mark_loaded(region)
        tab.viewer.say("")
        self._apply_centre_contrast(tab)     # the centre viewer owns contrast; this pane follows

    # -- the subset this pane owns, read by pane 1's scope selector -------------------------------
    def parked_subset(self) -> list:
        """The regions parked in the side pane — its FRONT tab's subset, or ``[]``.

        ONE owner, ONE reader. The side pane owns the subset (it is what the user put there);
        pane 1's scope selector reads it here when a run is aimed at ``SCOPE_SUBSET``. Neither
        keeps its own copy, which is the whole point of deleting pane 3's operator buttons.
        """
        tab = self._current_exploration()
        return list(tab.regions) if tab is not None else []

    def _export_subset_to_minerva(self, tab: "_ExplorationTab"):
        """Minerva Author on THIS TAB's subset — one fused mosaic per region.

        The export contract is ``_minerva.export_selection``'s and is not touched here: a region
        is fused into ONE OME-TIFF (Minerva lays out exactly one image and reads only
        ``series[0]``), and a FOV subset of a region is the crop of that region's mosaic, still
        one file. All this decides is WHAT is exported, and the answer is what this pane is
        showing — not whatever happens to be highlighted on the plate.
        """
        try:
            selection = _explore.subset_selection(
                tab.regions, (self._meta or {}).get("fovs_per_region"))
        except ValueError as exc:                     # named, in the status line, nothing exported
            self._readout.setText(f"cannot export to Minerva: {exc}")
            return

        def _landed(pairs, t=tab):
            """Put the story paths IN THE TAB, next to the mosaics they were made from.

            Minerva Author has no local deep link — verified, not assumed: its own front-end
            bundle reads only ``?story=`` and ``?image=``, and both route to Minerva CLOUD
            (loadCloudStory / openMinervaImage), never to a path on this machine. So the user
            always has to pick the file by hand in Author's "Select File" browser, which opens at
            $HOME — and ~/minerva_export is one click from there. The one thing we can do is make
            sure they are never hunting for the name, so it is written where they are looking."""
            if t.progress is None:
                return
            if not pairs:
                t.progress.setText("nothing was exported.")
                return
            t.progress.setText(
                "exported. In Minerva Author choose Select File and pick:\n"
                + "\n".join(str(story) for _ome, story in pairs))

        self.run_minerva_export(selection=selection, on_exported=_landed)

    def _build_minerva_tab(self) -> QWidget:
        """Minerva Author hand-off (IMA-228): export the SELECTION, then open Author on it.

        Scope comes from :meth:`minerva_selection` — the plate's selected FOVs/wells, else the
        well open in the detail viewer, which means every FOV of it. One file pair per FOV
        (Minerva opens one 2D image at a time and SquidMIP has no stitcher).
        """
        op = _OPERATIONS_BY_KEY["minerva"]
        w, v = self._op_tab_shell(
            op.label,
            "Writes an OME-TIFF plus a Minerva story for every selected FOV, then starts Minerva "
            "Author. Minerva has no deep link, so pick the .story.json below in its “Select File” "
            "dialog — the colours and contrast are already applied.",
        )
        state = {"dir": None, "pairs": []}

        dir_lbl = QLabel("(defaults to a minerva_export folder in your home directory)")
        dir_lbl.setWordWrap(True)
        dir_lbl.setStyleSheet("color:#8b98ad;font-size:12px;")

        # Projection mode — the salesperson tool (squid2minerva convert.py) offers --mip/--z, so
        # hardcoding one here would be a capability regression. Driven by the projector registry.
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(QLabel("Projection"))
        proj = QComboBox(); proj.setStyleSheet(_COMBO_QSS)
        proj.addItems(available_projectors())
        proj.setCurrentText("mip")
        row.addWidget(proj); row.addStretch(1)

        launch_cb = QCheckBox("Open Minerva Author after exporting")
        launch_cb.setStyleSheet(_CHECK_QSS)
        launch_cb.setChecked(True)

        path_lbl = QLabel("")
        path_lbl.setWordWrap(True)
        path_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path_lbl.setStyleSheet("color:#8b98ad;font-size:11px;")
        copy_btn = QPushButton("Copy story path"); copy_btn.setStyleSheet(_BTN_QSS); copy_btn.hide()
        reveal_btn = QPushButton("Show in folder"); reveal_btn.setStyleSheet(_BTN_QSS); reveal_btn.hide()

        def pick():
            d = QFileDialog.getExistingDirectory(self, "Save the Minerva export to folder")
            if not d:
                return
            state["dir"] = d
            dir_lbl.setText(d)

        def on_exported(pairs):
            state["pairs"] = pairs
            if not pairs:
                return
            path_lbl.setText("\n".join(str(story) for _, story in pairs))
            copy_btn.show(); reveal_btn.show()

        def do_copy():
            if state["pairs"]:
                QApplication.clipboard().setText("\n".join(str(s) for _, s in state["pairs"]))
                self._readout.setText("story path copied")

        def do_reveal():
            if state["pairs"]:
                from squidmip._minerva import reveal
                reveal(state["pairs"][0][1])

        pick_btn = QPushButton("Choose output folder…"); pick_btn.setStyleSheet(_BTN_QSS)
        pick_btn.clicked.connect(pick)
        run = QPushButton("Export the selected FOVs"); run.setStyleSheet(_BTN_QSS)
        run.clicked.connect(lambda: self.run_minerva_export(
            out_dir=state["dir"], projector=proj.currentText(),
            launch=launch_cb.isChecked(), on_exported=on_exported,
        ))
        copy_btn.clicked.connect(do_copy)
        reveal_btn.clicked.connect(do_reveal)

        v.addWidget(pick_btn); v.addWidget(dir_lbl)
        v.addLayout(row); v.addWidget(launch_cb); v.addWidget(run)
        v.addWidget(_hline()); v.addWidget(path_lbl); v.addWidget(copy_btn); v.addWidget(reveal_btn)
        v.addStretch(1)
        run.setEnabled(self._reader is not None)
        return w

    def minerva_selection(self) -> list:
        """The ``[(region, fov), ...]`` the user actually selected — never a silent stand-in.

        The requirement is "open minerva-author with the selected region(s)", so this reads the
        selection instead of inventing one. Exactly two sources, in order:

        1. :meth:`selected_region_fovs` — **this window's** selection. ``PlateOverview`` is
           display-only: it maps grid cells to well ids and emits them, and ``PlateWindow`` is
           where they land (``_on_selection_changed`` -> ``_selected_regions``) because
           expanding a well to its FOVs needs ``fovs_per_region``, which only this side has.
           So we call our own method directly. The previous version probed the overview too and
           fell back to ``PlateOverview.selected_wells()``; the overview never had a
           ``selected_region_fovs`` and the fallback was what made the export appear to work at
           all — a duck-typed chain standing in for reading the selection from its owner.
        2. The region open in the detail viewer (``_current_well``): every FOV of it.

        Note the unit. The pairs are ``(region, fov)`` but the export groups them BY REGION and
        fuses each into one mosaic — a region is a mosaic containing an array of FOVs, never a
        FOV. Selecting a whole region yields all its FOVs here and one fused mosaic downstream.

        Nothing selected returns ``[]`` — the caller says so rather than exporting fov 0 of 36
        and calling it "the selected well".
        """
        fovs_per_region = (self._meta or {}).get("fovs_per_region", {}) or {}

        def expand(regions) -> list:
            out = []
            for region in regions:
                out.extend((str(region), int(f)) for f in fovs_per_region.get(str(region), []))
            return out

        sel = [(str(r), int(f)) for r, f in self.selected_region_fovs()
               if int(f) in fovs_per_region.get(str(r), [])]
        if sel:
            return sel
        if self._current_well:
            return expand([self._current_well])
        return []

    def run_minerva_export(self, out_dir=None, projector: str = "mip", launch: bool = True,
                           on_exported=None, t: int = 0, selection=None):
        """Export the user's selection for Minerva Author and (optionally) open it.

        Runs off the GUI thread: projecting a well is real I/O plus compute, and starting
        Minerva Author polls a port for up to 90 s. Tests call this directly with launch=False.
        *selection* overrides :meth:`minerva_selection` (tests and future callers).
        """
        if self._reader is None or self._meta is None:
            self._readout.setText("open an acquisition first")
            return
        if self._minerva is not None and self._minerva.isRunning():
            self._readout.setText("already exporting — let the current export finish first")
            return

        sel = list(selection) if selection is not None else self.minerva_selection()
        if not sel:
            self._readout.setText(
                "nothing selected — pick the well or FOVs to export "
                "(double-click a well on the plate), then export again")
            return

        # The export unit is a REGION (one fused mosaic each), so count regions, not FOVs.
        regions = list(dict.fromkeys(r for r, _ in sel))
        what = (f"{len(regions)} mosaic{'s' if len(regions) != 1 else ''} "
                f"({', '.join(regions)}, {len(sel)} FOVs)")
        n_t = self._meta.get("n_t", 1) or 1
        t_note = f" (t={t} of {n_t})" if n_t > 1 else ""
        self._minerva = w = _MinervaWorker(
            self._reader, sel, out_dir, projector, t=t, launch=launch)

        def on_launched(ok):
            if ok:
                self._readout.setText(
                    f"✓ Minerva Author open — pick a .story.json ({what}{t_note} exported)")
            else:
                self._readout.setText(
                    f"✓ exported {what}{t_note} — Minerva Author not found "
                    f"(set ${_MINERVA_HOME_ENV} to an explorer checkout)")

        def on_exported_readout(pairs):
            # Report what LANDED, not what was asked for: a stop mid-export writes fewer.
            if not pairs:
                self._readout.setText("nothing exported")
                return
            done = regions[: len(pairs)]
            note = "" if len(pairs) == len(regions) else f" of {len(regions)} (stopped)"
            self._readout.setText(
                f"✓ exported {len(pairs)} mosaic{'s' if len(pairs) != 1 else ''}{note} from "
                f"{', '.join(done)}{t_note} → {Path(pairs[0][0]).parent}")

        w.progress.connect(
            lambda d, n: self._readout.setText(f"● Minerva export · {d}/{n} mosaics"))
        if on_exported is not None:
            w.exported.connect(on_exported)
        w.exported.connect(on_exported_readout)
        w.launched.connect(on_launched)
        w.failed.connect(lambda m: self._readout.setText(f"Minerva export failed: {m}"))
        self._readout.setText(f"● Minerva export · {what}{t_note} …")
        w.start()

    def _build_layers_tab(self) -> QWidget:
        """The Layers tab: the OperationStack as a list of toggleable, reorderable layers. The topmost
        enabled layer is what the plate shows. Base 'raw' plus each operator you have run."""
        w = QWidget(); w.setStyleSheet(f"background:{_BG};color:#e6edf3;")
        self._layers_box = QVBoxLayout(w)
        self._layers_box.setContentsMargins(14, 12, 14, 12); self._layers_box.setSpacing(6)
        self._layers_tab = w
        self._refresh_layers_tab()
        return w

    def _refresh_layers_tab(self):
        box = getattr(self, "_layers_box", None)
        if self._layers_tab is None or box is None:
            return
        while box.count():                       # rebuild from the current stack
            item = box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        title = QLabel("Layers (top shows on the plate)")
        title.setStyleSheet("font-size:14px;font-weight:800;")
        box.addWidget(title)
        for ly in reversed(self._op_stack.layers()):   # topmost first
            base = ly.key == "raw"
            row = QWidget(); h = QHBoxLayout(row); h.setContentsMargins(0, 0, 0, 0)
            cb = QCheckBox(ly.label + ("  (base)" if base else ""))
            cb.setChecked(ly.enabled); cb.setStyleSheet("color:#e6edf3;")
            cb.toggled.connect(lambda on, k=ly.key: self._on_layer_toggle(k, on))
            up = QPushButton("↑"); up.setStyleSheet(_BTN_QSS); up.setFixedWidth(34)
            up.clicked.connect(lambda _=False, k=ly.key: self._on_layer_move(k, +1))
            dn = QPushButton("↓"); dn.setStyleSheet(_BTN_QSS); dn.setFixedWidth(34)
            dn.clicked.connect(lambda _=False, k=ly.key: self._on_layer_move(k, -1))
            if base:
                # IMA-227: raw is the layer every transform is recoverable TO — "each transform is
                # a LAYER, the raw is never destroyed". OperationStack refuses to disable or
                # reorder it; the controls must SAY so rather than accepting a click the model
                # then ignores, which reads as a broken checkbox.
                cb.setEnabled(False)
                cb.setToolTip("The base layer. Untick the transforms above to see it.")
                up.setEnabled(False); dn.setEnabled(False)
            h.addWidget(cb, 1); h.addWidget(up); h.addWidget(dn)
            box.addWidget(row)
        box.addStretch(1)

    def _on_layer_toggle(self, key, enabled):
        self._op_stack.toggle(key, enabled)
        self._apply_layers()

    def _on_layer_move(self, key, delta):
        self._op_stack.move(key, delta)
        self._apply_layers()
        self._refresh_layers_tab()

    def _apply_layers(self):
        """Show the topmost enabled layer on the plate; keep the title in sync.

        ``top_enabled()`` cannot be None now that raw is undisableable, but this used to no-op on
        None and leave the plate showing a layer the tab said was OFF. Fall back to raw explicitly
        instead of silently doing nothing: the plate must never render something no enabled layer
        accounts for."""
        top = self._op_stack.top_enabled()
        if top is None:
            top = next((ly for ly in self._op_stack.layers() if ly.key == "raw"), None)
        if top is not None and self._overview is not None:
            self._overview.set_active_layer(top.key)
            self._plate_mode = "raw" if top.key == "raw" else top.label
            self._plate_title.setText(f"{self._acq_name}   ·   {self._plate_mode}")
            self._update_loupe_source()

    # -- loupe sources (IMA-208) --------------------------------------------------------------
    # One source per LAYER, registered when that layer's pixels get a real home, and dropped the
    # moment they don't. This is the "run identity" the review insisted on: the layer KEY alone
    # can't be trusted, because OperationStack.add dedupes by key — save a MIP, return to raw,
    # then run an unsaved preview and the same "mip" key now shows preview tiles while
    # _processed_plate still names the older save. Re-registering on every transition is what
    # keeps the inset showing the same run the tiles came from.

    def _release_loupe_sources(self):
        """Drop every source AND join the read thread that serves them.

        The one call every "the plate is being replaced" path must make. Assigning
        ``self._loupe_sources = {}`` (which _open_computed did) only forgets the sources: the
        _LoupeWorker QThread lives on the OVERVIEW, so the old overview walked off with a running
        thread and its ~35 MB plane cache on every plate open — confirmed still isRunning() after
        the overview was replaced. Only PlateOverview.set_loupe_source(None) stops and joins it."""
        if self._overview is not None:
            self._overview.set_loupe_source(None)
        self._loupe_sources = {}

    def _set_loupe_source(self, layer_key, source):
        self._loupe_sources[layer_key] = source
        self._update_loupe_source()

    def _drop_loupe_source(self, layer_key):
        self._loupe_sources.pop(layer_key, None)
        self._update_loupe_source()

    def _update_loupe_source(self):
        """Point the plate at the source for whatever layer is on screen right now."""
        if self._overview is None:
            return
        active = getattr(self._overview, "_active", "raw")
        source = self._loupe_sources.get(active)
        if source is self._overview._loupe_src:
            return                                   # unchanged: don't churn the worker thread
        colors = None
        if self._meta and self._meta.get("channels"):
            colors = np.stack([_hex_to_rgb01(c["display_color"]) for c in self._meta["channels"]])
        self._overview.set_loupe_source(source, colors)

    def _build_cli_tab(self) -> QWidget:
        """A LIVE, interactive shell in the pane: run the `squidmip` batch CLI (IMA-186) right here.
        Pre-seeded with the how-to (MIP every well; `--tiff` -> FIJI-openable TIFFs). `squidmip` is
        aliased to this app's interpreter so it runs regardless of PATH/conda. Falls back to a static
        command preview where a PTY isn't available (e.g. Windows)."""
        # Input must be a RAW acquisition folder; if the current path is a computed .hcs plate (or
        # none), show a placeholder rather than a wrong path.
        p = str(self._acq_path) if self._acq_path else ""
        acq = p if (p and ".hcs" not in p and not p.endswith(".ome.zarr")) else "<your acquisition folder>"
        py = sys.executable
        win = sys.platform == "win32"
        banner = [
            "==========================================================",
            "  Process a whole plate from the command line",
            "==========================================================",
            "",
            "  Same MIP as the buttons, on every well. Copy a line and press Enter.",
            "",
            "  - Flatten every well + save FIJI-openable TIFFs:",
            f'      python -m squidmip "{acq}" --tiff',
            "",
            "  - Try just the first 8 wells first (quick, little disk):",
            f'      python -m squidmip "{acq}" --limit 8 --tiff',
            "",
            "  - Choose where to save:",
            f'      python -m squidmip "{acq}" --limit 8 --tiff --output-folder ~/Downloads',
            "",
            "  - All options:   python -m squidmip --help",
            "",
        ]
        # The terminals put the venv's Scripts/bin on PATH, so the `squidmip` console script resolves
        # directly — no alias needed (doskey is unreliable in a piped cmd.exe anyway).
        setup: list = []
        cwd = str(self._acq_path.parent) if self._acq_path else str(Path.home())
        if not win:                              # Unix: a real PTY terminal
            try:
                t = _Terminal(cwd, banner, setup_cmds=setup)
                if t._fd is not None:
                    return t
            except Exception:
                pass
        try:                                     # Windows (+ Unix fallback): a QProcess shell
            t = _ProcTerminal(cwd, banner, setup)
            if t.running():
                return t
        except Exception:
            pass
        term = QPlainTextEdit(); term.setReadOnly(True)   # last resort: static, copy-paste preview
        term.setStyleSheet(_TERM_QSS)
        term.setPlainText(
            "Process a whole plate from the command line\n"
            "──────────────────────────────\n"
            "Open a terminal, then paste (no conda needed — this is the app's own Python):\n\n"
            f'    "{py}" -m squidmip "{acq}" --limit 8 --tiff --output-folder ~/Downloads\n\n'
            "This flattens the first 8 wells (MIP) and saves TIFFs you can open in FIJI.\n"
            "Drop --limit 8 to do the whole plate. Add --help to see all options.\n")
        return term

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
            # Use ndviewer's OWN FOV slider as the scan navigator (upstreamed control — no external
            # slider). Its valueChanged drives the red box (_on_fov_slider), and the plate's double-click
            # drives it back (go_to_well_fov); both stay in sync. Hide only the "n per well" subset
            # control (an IMA-191 extra that would just clutter the z-stack detail here).
            sub = getattr(v, "_subset_container", None)
            if sub is not None:
                sub.hide()
            return v
        except Exception as e:
            self._readout.setText(f"ndviewer_light unavailable: {e}")
            return None

    # -- drag & drop --
    def _open_acquisition_dialog(self):
        """File > Open: pick a Squid acquisition folder (the reliable alternative to drag-drop)."""
        d = QFileDialog.getExistingDirectory(self, "Open a Squid acquisition folder")
        if d:
            self.ingest(d)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):   # some Windows setups require dragMove to also accept, or drop is refused
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            e.acceptProposedAction()
            self.ingest(urls[0].toLocalFile())

    # -- open an acquisition (no processing yet — that's the Process menu) --
    def ingest(self, path: str):
        from squidmip import open_reader

        p, is_plate = resolve_plate_root(path)
        if is_plate:
            self._readout.setText("this is already a written plate — drop a raw Squid acquisition")
            return
        # stop any in-flight run/preview/export and clear prior state before opening a new
        # acquisition. _stop_minerva matters as much as the other two: a Minerva worker left
        # running holds the OLD reader and would keep exporting (and launching) against an
        # acquisition the window no longer shows.
        self._stop_worker()
        self._stop_preview()
        self._stop_minerva()
        self._stop_mosaic_worker()   # it holds the OLD reader, and it is joined, not drained
        # Exploration tabs belong to the acquisition they were opened from: their region sets and
        # layer keys point at a _fov_index that is about to be rebuilt for a different plate.
        self._close_exploration_tabs()
        self._active_exploration = None
        self._push_index = None
        self._run_tab_key = self._run_view_tab_key = None
        self._reader = self._meta = None
        self._fov_index = {}
        self._selected_regions = []   # wells picked on the plate (IMA-221); scopes an operator run
        self._pushed = set()
        self._current_well = None
        self._current_fov = 0
        self._enable_operators(False)
        if self._overview is not None:
            self._release_loupe_sources()   # join the read thread BEFORE dropping its owner
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
        # Resolve the layout format ONCE: an explicit override wins, then the declared field, then
        # inference from the well ids (IMA-219 — two real acquisitions carry no format at all).
        # Never fatal: an un-inferable plate keeps the declared value and falls through the guard.
        # Resolve the sample holder ONCE (IMA-214). build_plate handles wells AND slides: a slide
        # carrier is a Plate whose cells are the freeform region ids, so a glass-slide/tissue
        # acquisition reaches this widget by the same path a 384wp does. It also reconciles a
        # declared format against the MEASURED stage pitch, so a mis-declared plate cannot lay out
        # at the wrong scale.
        try:
            plate = build_plate(meta, override=self._plate_format_override)
        except (PlateShapeError, PlateBuildError) as e:
            self._readout.setText(f"cannot lay out this acquisition: {e}")
            self._drop.show()
            return
        self._plate = plate
        self._plate_format = fmt = plate.format_name
        self._reader, self._meta = reader, meta
        self._acq_name = Path(p).name
        self._acq_path = Path(p)
        self._processed_plate = None
        self._sync_focus_button()                    # 2D acquisition -> no reference-plane button
        self._populate_detect_channels()             # channel-aware cellpose picker
        self._viewer_manager.set_dataset(reader, meta)   # every spawned window shares this reader
        rows, cols, wells, order = plate.viewer_grid()
        for idx, region in enumerate(order):
            self._fov_index[region] = {"idx": idx, "well_id": region, "rc": plate.cell_index(region)}

        self._order = order                          # well order = the detail's FOV-slider order
        # A freeform holder places its cells by GEOMETRY (IMA-253): the plate hands over one
        # rectangle per region, in grid units, and the overview draws exactly those. A well plate
        # returns None here and keeps the uniform grid it has always had.
        cl = plate.cell_layout() if hasattr(plate, "cell_layout") else None
        layout = ({plate.cell_index(cid): rect for cid, rect in cl.items()} if cl else None)
        self._overview = PlateOverview(rows, cols, wells, layout=layout)
        # Carrier art behind the cells (IMA-220). Hand over the PLATE, not its name: `plate` is what
        # build_plate RESOLVED (measured pitch beat the 2x2's mis-declared "384 well plate"), so the
        # background can only ever be drawn at the same scale the grid is laid out at.
        self._overview.set_carrier(plate)
        self._selected_regions = []                  # a new acquisition starts with nothing picked
        self._overview.hovered.connect(self._on_hover)
        self._overview.wellActivated.connect(self.activate_well)
        self._overview.selectionChanged.connect(self._on_selection_changed)
        self._overview.marqueeSelected.connect(self._on_marquee_selected)
        self._plate_mode = "raw"                     # a freshly-opened plate shows raw previews
        self._plate_title.setText(f"{self._acq_name}   ·   raw")   # bottom-left plate-pane title
        self._op_stack.reset()                       # fresh layer stack (base only)
        self._active_op_key = None
        if getattr(self, "_raw_btn", None):
            self._raw_btn.hide()                     # raw view on open -> nothing to return from
        self._refresh_layers_tab()
        self._drop.hide()
        self._left_l.addWidget(self._overview, 1)   # fills the pane and self-fits — no scrollbars
        self._declare_channel_axis(meta["channels"], meta["dtype"])

        self._setup_raw_detail()
        # Hand the plate's region order to the SINGLE OWNER. Announcing it is what puts the red
        # ROI frame on region 0, sizes the region slider, and loads pane 2's mosaic — one move,
        # not three calls that could each be forgotten on some path.
        #
        # Cleared first so the announce always happens: re-opening an acquisition whose region
        # ids match the previous one would otherwise be a no-op move and pane 2 would keep the
        # OLD plate's mosaic on screen.
        self._cursor.set_order([])
        self._cursor.set_order(order)

        self._enable_operators(True)

        # The loupe works from the moment the folder opens — the raw layer's real pixels are the
        # acquisition's own TIFFs, the same planes the preview below is about to downsample. No
        # operator run is required to look closely at a well.
        self._loupe_sources = {"raw": _RawLoupeSource(
            reader, meta, lambda w: _fov_of_well(w, meta.get("fovs_per_region")))}
        self._update_loupe_source()

        # The mosaic geometry is known the moment the acquisition opens — it is pure arithmetic on
        # coordinates.csv — so hand it to the plate NOW rather than waiting for an operator run
        # (IMA-249: it was only ever set from run_operator, which is why the plate looked like a
        # grid of lone frames until something was run). The preview below composites into exactly
        # these boxes.
        self._overview.set_mosaic_boxes(_mosaic_boxes(meta))

        # fast RAW preview: fill the plate with downsampled thumbnails immediately (grey dots),
        # in the SAME row-major order the operator will later process them in.
        self._preview = _PreviewWorker(reader, meta, self._fov_index, order)
        self._preview_order = list(order)
        self._preview.tileReady.connect(self._on_preview_tile)
        self._preview.streamEnded.connect(lambda: self._recomposite("raw"))
        self._preview.failed.connect(self._on_preview_failed)
        self._preview.start()   # (the detail already landed on order[0] via _setup_raw_detail)
        # top-left = STATUS (what's happening / what's shown); the plate name is the pane title.
        # "live" is retired from user-facing copy: this is POST-ACQUISITION review, and calling
        # a loaded plate "live" reads as a running scope. The phrasing is operator/stitcher
        # iteration.
        # Multi-FOV policy (IMA-187): an operator run processes EVERY FOV and composites them into
        # the well's cell by stage coordinate. The raw preview above is still one FOV per well (it
        # reads a single plane per well precisely to stay fast), so say which one you're looking at.
        multi = sum(1 for r in order if len(meta["fovs_per_region"][r]) > 1)
        note = (f" · {multi} multi-FOV region(s), previewing as mosaics" if multi else "")
        # NOT "live". This is a POST-ACQUISITION tool: nothing here is streaming off a scope --
        # the acquisition is finished and on disk, and calling it live invited exactly the wrong
        # mental model of what the operators below are doing. What the line has to say is what is
        # loaded and how to open it -- including the region slider, which is new and otherwise
        # undiscoverable.
        self._readout.setText(
            f"{len(self._fov_index)} wells loaded · slide the region slider or double-click to "
            f"open one{note}")

    def _make_region_slider(self):
        """Build the region slider, or say why there is none. NEVER a silent absence.

        It is napari's own dims slider (play button, fps popup, loop modes and an animation
        thread), driven from a napari ``Dims`` model that carries only the region axis — see
        ``_region_nav`` for why the region is not an axis of the image array instead.
        """
        try:
            slider = RegionSlider(self)
        except Exception as exc:                     # noqa: BLE001 - reported, never swallowed
            self._region_slider_failure = f"{type(exc).__name__}: {exc}"
            return None
        self._region_slider_failure = None
        slider.bind(self._cursor)
        slider.on_problem(lambda msg: self._readout.setText(msg))
        return slider

    # -- the current region: ONE value, three views ------------------------------------------
    #
    # `_mosaic_region` and `_current_well` are properties over `self._cursor` rather than fields.
    # That is the whole point: an assignment to either cannot create a second copy that drifts
    # out of step with the red frame. Every one of this project's 4+ confirmed instances of that
    # defect was a field somebody forgot to update on one path out of five.

    @property
    def _mosaic_region(self) -> Optional[str]:
        """The region pane 2 is showing. Read-only: the cursor decides, this reports."""
        return self._cursor.region

    @property
    def _current_well(self) -> Optional[str]:
        """The region the USER opened, or None if they have only ever been shown one.

        Not the same question as ``_mosaic_region``: ``_selection_regions`` scopes an operator
        run to this, so "a plate was loaded and something had to be on screen" must not count.
        """
        return self._cursor.region if self._cursor.activated else None

    @_current_well.setter
    def _current_well(self, value: Optional[str]) -> None:
        if value is None:
            self._cursor.deactivate()          # nothing open; the frame does NOT move
        else:
            self._cursor.activate(value)

    def _on_region_changed(self, index: int, region: str):
        """THE current region moved. Everything that shows it follows from here, and nowhere else.

        Order matters. The red ROI frame moves FIRST so the plate never lags the slider by the
        length of a mosaic load — the frame and the slider must never disagree, and a mosaic that
        takes a second to arrive would otherwise leave them disagreeing for that second.
        """
        if self._overview is not None:
            info = self._fov_index.get(region)
            if info is not None:
                self._overview.select(*info["rc"])          # THE RED FRAME
        if self._region_slider is not None:
            self._region_slider.setToolTip(
                f"region {index + 1} of {self._cursor.count}: {region}\n"
                "Press play to walk the regions; right-click play for frames per second.")
        # RESPONSIVE REGION SLIDER (viewport rendering). Fusing a region's mosaic is the expensive
        # step: each tick stops the prior _MosaicWorker (waiting up to 2 s) and starts a new one, so
        # dragging across ten regions queued ten fuses and stalled. The RED FRAME above already moved
        # instantly; only the mosaic load needs to wait for the slider to SETTLE. Debounce it: the
        # last region the slider lands on is the only one we fuse. A short delay is imperceptible when
        # you stop, and turns a drag from ten blocking loads into one.
        if getattr(self, "_region_load_timer", None) is None:
            self._region_load_timer = QTimer(self)
            self._region_load_timer.setSingleShot(True)
            self._region_load_timer.timeout.connect(
                lambda: self._load_mosaic(region=self._pending_region))
        self._pending_region = region
        self._region_load_timer.start(140)

    def _load_mosaic(self, region: Optional[str] = None, op: str = "raw"):
        """Show one region's fused MOSAIC in pane 2, one napari layer per channel.

        The unit displayed is a mosaic, never a single FOV (IMA-265). This runs on OPEN, before
        any operator: a raw acquisition has no pyramid on disk, so the region's FOVs are fused
        by stage position. Once an operator has written an OME-Zarr, ``_load_mosaic_zarr`` shows
        that pyramid lazily instead, as a SECOND processing layer, so the before/after toggle is
        just a visibility flip.
        """
        pane = getattr(self, "_mosaic_pane", None)
        if pane is None or not getattr(pane, "ok", False):
            return
        if self._reader is None or self._meta is None:
            return
        region = region or self._cursor.region
        if region is None:
            return

        prior = getattr(self, "_mosaic_worker", None)
        if prior is not None and prior.isRunning():
            prior.stop()
            prior.wait(2000)

        # THE Z SLIDER IS GLOBAL ACROSS THE PLATE. Replacing the layers resets napari's dims to
        # step 0, so the z you were inspecting silently snapped back to the bottom of the stack
        # every time you moved to another region — which is the opposite of "the plate composites
        # with the z and t sliders". Remember it here and restore it once the new layers are in.
        self._pending_dims_step = self._napari_dims_step()
        # A run in flight is counting the OLD region. Let it finish and you get a mask for B2
        # drawn over B3's mosaic, with B2's number in the readout — a plausible-looking lie.
        self._stop_spots()
        pane.mosaic.remove_op(op)
        # Drop the previous region's overlays for the same reason. `remove_op` is a no-op when
        # nothing has been counted yet.
        pane.mosaic.remove_op(self.SPOTS_OP)
        channels = [c["name"] for c in self._meta["channels"]]
        z_now = 0
        if self._pending_dims_step and self._napari_z_axis() is not None:
            z_now = int(self._pending_dims_step[self._napari_z_axis()])
        w = _MosaicWorker(self._reader, self._meta, region, channels, z_index=z_now, parent=self)
        w.ready.connect(lambda r, ch, levels, bbox:
                        self._on_mosaic_plane(op, r, ch, levels, bbox))
        w.problem.connect(lambda msg: pane.say(msg))
        w.finished_count.connect(lambda n: self._on_mosaic_done(op, region, n))
        self._mosaic_worker = w
        w.start()

    def _on_mosaic_plane(self, op: str, region: str, channel: str, levels, bbox_um):
        """One channel of the mosaic arrived, as a LAZY PYRAMID. Add it as a napari layer.

        ``levels`` is always the list napari's ``multiscale=True`` contract wants — highest
        resolution first — even when the mosaic is too small to have a second rung, so there is
        one code path here and no sniffing of what arrived.
        """
        pane = getattr(self, "_mosaic_pane", None)
        if pane is None or not getattr(pane, "ok", False):
            return
        if getattr(self, "_mosaic_region", None) != region:
            return                                  # a later region won the race; drop this one
        from squidmip._napari_pane import _colormap_for

        # NO contrast_limits: napari autoscales, and napari OWNS contrast.
        #
        # What must NOT come back is _pct_window's percentile window. Two things were wrong with it.
        # First it duplicated napari's job — napari computes its own percentile autoscale and
        # exposes it on the layer, so passing ours meant two percentile rules over one quantity,
        # which is this project's most-repeated defect shape. Second it made the composite
        # unreadable: a window like 561 -> 576..4032 sends every mid-tone tissue pixel to full
        # intensity in THAT channel, and with additive blending four saturated channels sum to
        # white. Julio, repeatedly: "Channel blending still sucks."
        #
        # Julio: "Napari has so many pre-built features that you're not leveraging." This is one.
        # napari autoscales on add and the user retunes with the layer's own contrast slider,
        # which is also the single owner the plate now follows. ima-nav-controls measured that
        # autoscale at ~940 ms/channel and moved it off-thread; against the PYRAMID napari
        # autoscales from the small level it renders, so the cost is gone rather than relocated.
        # multiscale=True is what makes the pyramid a pyramid. Without it napari treats the list
        # as one array to stack, or takes level 0 and renders exactly as slowly as before — the
        # levels would exist and buy nothing.
        pane.mosaic.add_mosaic(
            op, channel, levels,
            colormap=_colormap_for(channel),
            multiscale=True,
            bbox_um=bbox_um,
            z_scale_um=(self._meta or {}).get("dz_um"),
        )

    # -- napari's own z / t dimension sliders, which are GLOBAL across the plate ---------------
    def _napari_dims(self):
        """napari's ``Dims`` model, or None when napari is not the viewer.

        This is THE z slider — the one commit 19cd491 made real by handing napari a lazy
        ``(z, y, x)`` stack. Nothing here builds a second one.
        """
        pane = getattr(self, "_mosaic_pane", None)
        if pane is None or not getattr(pane, "ok", False):
            return None
        return getattr(pane.mosaic.model, "dims", None)

    def _napari_z_axis(self) -> Optional[int]:
        """Index of the z axis in napari's dims, or None when the data has no z axis.

        napari puts the displayed axes LAST, so with a ``(z, y, x)`` layer z is ``ndim - 3``.
        Derived rather than hard-coded to 0: adding a t axis would shift it, and a hard-coded
        index would then quietly drive the wrong slider.
        """
        dims = self._napari_dims()
        if dims is None or int(getattr(dims, "ndim", 0)) < 3:
            return None
        return int(dims.ndim) - 3

    def _napari_dims_step(self):
        dims = self._napari_dims()
        return tuple(dims.current_step) if dims is not None else None

    def _restore_dims_step(self):
        """Put the global z (and t) back where the user left it after a region change."""
        want = getattr(self, "_pending_dims_step", None)
        self._pending_dims_step = None
        dims = self._napari_dims()
        if dims is None or not want:
            return
        for axis, step in enumerate(want[: int(dims.ndim)]):
            top = int(dims.nsteps[axis]) - 1
            if 0 <= int(step) <= top and int(dims.current_step[axis]) != int(step):
                dims.set_current_step(axis, int(step))

    def _on_mosaic_done(self, op: str, region: str, n: int):
        pane = getattr(self, "_mosaic_pane", None)
        if pane is None or not getattr(pane, "ok", False):
            return
        btn = getattr(pane, "detect_button", None)
        if n == 0:
            pane.say(f"{region}: no mosaic could be built (see the message above).")
            # The frame gate must open even on failure, or playback stops dead on the first
            # region that cannot be fused and the play button just looks stuck.
            self._region_frame_done()
            if btn is not None:
                btn.setEnabled(False)   # nothing to count; an enabled button here does nothing
            return
        pane.say("")
        if btn is not None:
            btn.setEnabled(True)        # there is now a region on the canvas to run the operator on
        try:
            pane.mosaic.show_op(op)
            pane.mosaic.model.reset_view()
        except Exception:                            # noqa: BLE001 - view framing is cosmetic
            pass
        self._restore_dims_step()
        self._bind_napari_contrast()
        self._adopt_centre_view()
        self._region_frame_done()

    def _region_frame_done(self):
        """Tell the region slider this region is on screen, so playback may request the next.

        napari's playback is debounced on the render for exactly this reason; wiring our load
        into that gate is what stops a 10 fps timer queueing ten mosaic loads per completed one.
        """
        if self._region_slider is not None:
            self._region_slider.frame_done()

    # -- the analysis operator: spot detection on what pane 2 is showing -------------------

    #: Processing-layer key the spot-detection result layers are filed under. A DISTINCT op from
    #: "raw"/"stitched" so the layer tree groups the analysis overlays on their own and
    #: ``show_op`` never has to choose between the mosaic and the mask drawn over it.
    #: Read off ``_spots.LAYER_KEY`` rather than restated, so the UI and the engine registry
    #: cannot drift apart on the spelling.
    SPOTS_OP = _SPOTS_LAYER_KEY

    def _spot_source_layer(self):
        """The (channel, layer) the count will be run on: the first VISIBLE mosaic channel.

        Returns ``(None, None)`` when there is nothing to count. Deliberately reads the CANVAS
        rather than the metadata: the number in the readout has to describe the picture the user
        is looking at, or the two disagree and the readout is the one that lies.
        """
        pane = getattr(self, "_mosaic_pane", None)
        if pane is None or not getattr(pane, "ok", False) or pane.mosaic is None:
            return None, None
        op = pane.mosaic.visible_op()
        if op is None or op == self.SPOTS_OP:
            return None, None
        # CHANNEL-AWARE: if the "Detect on" dropdown names a channel, that is authoritative -- the
        # user picked the channel that carries the signal, which need not be the visible one (405
        # is blank on the tissue set). Segmentation reads layer.data, so visibility is irrelevant.
        combo = getattr(pane, "detect_channel", None)
        chosen = combo.currentText().strip() if combo is not None and combo.count() else ""
        if chosen:
            layer = pane.mosaic.find(op, chosen)
            if layer is not None:
                return chosen, layer
        # fallback: the first VISIBLE channel, as before.
        for channel in pane.mosaic.channels(op):
            layer = pane.mosaic.find(op, channel)
            if layer is not None and getattr(layer, "visible", False):
                return channel, layer
        return None, None

    def _current_z_index(self):
        """Which z napari is showing, or None for a 2-D layer. napari OWNS the z slider."""
        pane = getattr(self, "_mosaic_pane", None)
        try:
            dims = pane.mosaic.model.dims
            if dims.ndim < 3:
                return None
            return int(dims.current_step[0])
        except Exception:                            # noqa: BLE001 - absence is not a failure
            return None

    def _on_detect_nuclei(self):
        """Run spot detection on the visible channel. Returns IMMEDIATELY; the work is off-thread."""
        pane = getattr(self, "_mosaic_pane", None)
        region = getattr(self, "_mosaic_region", None)
        channel, layer = self._spot_source_layer()
        if pane is None or region is None or layer is None:
            if pane is not None:
                pane.say("nothing to count: no region mosaic is visible in this pane yet.")
            return

        prior = getattr(self, "_spot_worker", None)
        if prior is not None and prior.isRunning():
            # A second click CANCELS the run in flight rather than queueing another one. Two
            # segmentations racing to write the same layer is the "two representations of one
            # truth" defect with a thread attached.
            prior.stop()
            pane.say(f"{region}/{channel}: cancelling the run in flight…")
            return

        bbox_um = None
        try:
            from squidmip._mosaic_source import mosaic_bbox_um

            bbox_um = mosaic_bbox_um(self._meta, region)
        except Exception as exc:                     # noqa: BLE001 - said, not swallowed
            pane.say(f"{region}: mosaic placement unavailable ({exc}); the overlay will be "
                     "drawn in pixel coordinates and will NOT line up with the mosaic.")

        w = _SpotWorker(region, channel, layer.data, self._current_z_index(), bbox_um,
                        parent=self)
        w.ready.connect(self._on_spots_ready)
        w.problem.connect(lambda msg: pane.say(msg))
        w.stageChanged.connect(
            lambda name: pane.say(f"{region}/{channel}: counting nuclei — {name}…"))
        w.cancelled.connect(lambda: pane.say(f"{region}/{channel}: spot detection cancelled."))
        w.finished_count.connect(self._on_spots_done)
        w.finished.connect(self._on_spot_worker_finished)
        self._spot_worker = w
        pane.say(f"{region}/{channel}: counting nuclei…")
        w.start()

    def _on_spot_worker_finished(self):
        """Re-enable the button however the run ended — ok, failed, or cancelled."""
        pane = getattr(self, "_mosaic_pane", None)
        btn = getattr(pane, "detect_button", None) if pane is not None else None
        if btn is not None:
            btn.setEnabled(True)

    def _on_spots_ready(self, region, channel, labels, centroids, bbox_um, count):
        """The result landed. Put it ON THE CANVAS as real napari layers."""
        pane = getattr(self, "_mosaic_pane", None)
        if pane is None or not getattr(pane, "ok", False):
            return
        if getattr(self, "_mosaic_region", None) != region:
            return                                   # the user moved on; drop a stale result
        from squidmip._spots import centroid_layer_name, mask_layer_name

        # The MASK: a real Labels layer, so napari gives it its own label colormap, transparent
        # background and click-to-pick. add_image would render it as a near-black gradient.
        pane.mosaic.add_labels(self.SPOTS_OP, mask_layer_name(channel), labels,
                               bbox_um=bbox_um)
        # The CENTROIDS: a Points layer, with the per-object record (Fractal's feature-table
        # contract) riding on `features`, keyed by label value.
        pane.mosaic.add_points(
            self.SPOTS_OP, centroid_layer_name(channel), centroids,
            bbox_um=bbox_um, shape=labels.shape,
            features={"label": np.arange(1, len(centroids) + 1, dtype=np.int32)},
        )

    def _on_spots_done(self, region, channel, count):
        """The NUMBER — per region, in the status readout, which is what Spencer asked for."""
        pane = getattr(self, "_mosaic_pane", None)
        if pane is not None:
            pane.say(f"{region} · {channel}: {count} nuclei")
        counts = getattr(self, "_spot_counts", None)
        if counts is None:
            counts = self._spot_counts = {}
        counts[(region, channel)] = int(count)       # per-region tally, for the plate readout
        self._readout.setText(f"{region} · {channel} · {count} nuclei detected")

    def _bind_napari_contrast(self):
        """Point the EXISTING contrast sink (IMA-261) at napari instead of ndviewer_light.

        No new contrast model, and no second sink: this reuses ``_on_detail_contrast``, which
        already lands in the plate's FOLLOW path via ``follow_channel_window`` rather than its
        manual latch. That distinction is the one that matters — treating an owner's autoscale as
        a user gesture is what latched every channel MANUAL on open, killed the plate's running
        auto-contrast from the first frame, and left SCOPE_PER_REGION painting every well under
        one global window while the amber "wells NOT comparable" badge lied over the top.

        ``MosaicLayers.on_user_contrast`` additionally filters out OUR OWN writes (the percentile
        window set at add time, and link propagation), so only a real change of the owner's
        resolved window arrives here.
        """
        pane = getattr(self, "_mosaic_pane", None)
        if pane is None or not getattr(pane, "ok", False) or self._meta is None:
            return
        if getattr(self, "_napari_contrast_bound", False):
            return          # connect ONCE: the pane outlives every ingest, like the detail viewer
        index = {c["name"]: i for i, c in enumerate(self._meta["channels"])}

        def _sink(channel: str, lo: float, hi: float):
            ch = index.get(channel)
            if ch is not None:
                self._on_detail_contrast(ch, lo, hi)
            self._push_contrast_to_side_pane(channel, lo, hi)

        pane.mosaic.on_user_contrast(_sink)

        # ...and the eye icons, for exactly the same reason. Julio: "there shouldn't be any
        # controls for the plate view. It just reacts to toggles and contrast adjustments in
        # napari." The plate's own checkboxes are gone; this is what replaces them.
        def _vis_sink(channel: str, on: bool):
            ch = index.get(channel)
            if ch is None:
                return
            self._overview.set_channel_visible(ch, on)

        pane.mosaic.on_user_visibility(_vis_sink)

        # ...and the LUT. Julio: "I change channel colormap in napari and plate view doesn't
        # react." Same sink shape: napari owns the colour, the plate follows it.
        def _cmap_sink(channel: str, rgb):
            ch = index.get(channel)
            if ch is not None:
                self._overview.set_channel_color(ch, rgb)

        pane.mosaic.on_user_colormap(_cmap_sink)
        self._napari_contrast_bound = True

    def _adopt_centre_view(self):
        """PULL what napari resolved for every channel, and make the plate show the same.

        Julio, with a screenshot: "Look at contrast difference between napari window and plate
        view." This is why they differed, and it is not the event sink being broken.

        The sink (`on_user_contrast`) only reports a USER gesture -- deliberately, because napari
        autoscales on every `add_image` and treating that as a gesture latched every channel
        MANUAL before anyone had touched anything. But that filter also swallows the ONE moment
        that matters most: the window napari picks when a region is first shown. So the plate kept
        painting from its own running percentile histogram, napari painted from its autoscale, and
        the two panes disagreed from the first frame until the user happened to drag a slider.

        An EVENT tells you about a change; the initial state is not a change. So this pulls the
        current value instead of waiting to be told, at the one point where the layers are known
        to exist. It lands in the FOLLOW path, so it still is not a user latch, and the same is
        done for the colormap -- napari resolves the LUT per layer and the plate must tint to
        match it, not to its own copy of `display_color`.
        """
        pane = getattr(self, "_mosaic_pane", None)
        if pane is None or not getattr(pane, "ok", False) or self._meta is None:
            return
        if self._overview is None:
            return
        for i, c in enumerate(self._meta["channels"]):
            window = pane.mosaic.contrast(c["name"])
            if window is not None:
                self._overview.follow_channel_window(i, float(window[0]), float(window[1]))
            rgb = pane.mosaic.channel_rgb(c["name"])
            if rgb is not None:
                self._overview.set_channel_color(i, rgb)
            visible = pane.mosaic.channel_visible(c["name"])
            if visible is not None:
                self._overview.set_channel_visible(i, bool(visible))

    def _centre_contrast(self) -> dict:
        """The centre viewer's per-channel window — the ONE contrast value per channel.

        Read from ``MosaicLayers``, not remembered here. A remembered copy is a second answer to
        a question that already has an owner, and this file has shipped four bugs of exactly that
        shape."""
        pane = getattr(self, "_mosaic_pane", None)
        if pane is None or not getattr(pane, "ok", False) or self._meta is None:
            return {}
        out = {}
        for c in self._meta["channels"]:
            window = pane.mosaic.contrast(c["name"])
            if window is not None:
                out[c["name"]] = window
        return out

    def _push_contrast_to_side_pane(self, channel: str, lo: float, hi: float):
        """Every side-pane viewer FOLLOWS the centre viewer's contrast for that channel.

        Julio: "the channel toggling and contrast adjustment for the plate view should happen
        from our central viewer window." A side-pane tab is a second napari viewer, and a second
        viewer that autoscales independently is a second owner of one quantity however few
        widgets it shows. napari cannot link layers across viewers, so the link is made here —
        one direction only, centre -> side, and written inside ``programmatic()`` so a followed
        value can never be mistaken for the user having dragged the side pane's own slider and
        bounce back."""
        for w in list(self._op_tabs.values()):
            if not isinstance(w, _ExplorationTab) or w.viewer is None:
                continue
            try:
                with w.viewer.mosaic.programmatic():
                    w.viewer.mosaic.set_contrast(channel, lo, hi)
            except KeyError:
                continue          # this tab is not showing that channel yet — nothing to follow

    def _apply_centre_contrast(self, tab: "_ExplorationTab"):
        """Bring a side-pane viewer up to the centre viewer's current windows.

        Called after a region's layers land in a tab. Without it a tab opened after the user had
        already tuned contrast would show its own autoscale instead — the follower would be
        correct only for changes made from that moment on, which is the same half-life defect as
        a subscription that dies when its layers are rebuilt."""
        if tab.viewer is None:
            return
        for channel, (lo, hi) in self._centre_contrast().items():
            try:
                with tab.viewer.mosaic.programmatic():
                    tab.viewer.mosaic.set_contrast(channel, lo, hi)
            except KeyError:
                continue

    def _setup_raw_detail(self, order: Optional[list] = None):
        """Point the detail viewer at the RAW acquisition: full z-stack, full frame, FOV slider.

        ``order=None`` is the whole plate (open / 'Return to raw view'). An exploration tab passes
        its own region subset so the slider lists exactly the wells that tab is scoped to.
        Registers each well's raw plane PATHS up front (cheap — paths only, no image I/O) so
        scrubbing shows a real (lazily read + cached) image per well instead of black."""
        if self._detail is None or self._reader is None:
            return
        meta, reader = self._meta, self._reader
        order = self._order if order is None else list(order)
        h, w = meta["frame_shape"]
        channels = [c["name"] for c in meta["channels"]]
        # pixel_size_um/dz_um are what make ndv's 3D button render this z-stack with the
        # right geometry (IMA-255). This is the ONLY call site that declares a real n_z —
        # the processed/mosaic ones below declare n_z=1, where a volume is meaningless — so
        # it is the only one that needs them. Omitting them renders the stack isotropic:
        # on the tissue set that is dz 1.5um against pixel 0.752um, i.e. 2x squashed in z.
        # Passed positionally-by-keyword and NOT guarded: a stale ndviewer_light without
        # these parameters must fail loudly here rather than silently drop back to
        # isotropic. See tests/test_viewer_3d.py.
        self._detail.start_acquisition(channels, meta["n_z"], h, w, [f"{r}:0" for r in order],
                                       pixel_size_um=meta.get("pixel_size_um"),
                                       dz_um=meta.get("dz_um"))
        self._push_shape = None       # raw mode registers PATHS, not arrays — no array canvas here
        self._push_problem = None
        # Re-scope the RAW preview to the same wells the slider now lists. Without this the
        # producer (a full-plate _PreviewWorker) and the consumer (_push_index, built from
        # `order`) describe different well lists, and every push outside the subset is discarded.
        # That is the bug that made the FOV slider stop advancing after an exploration tab was
        # opened: the slider showed only the well that had already loaded.
        if getattr(self, "_preview", None) is not None and order != getattr(self, "_preview_order", None):
            self._stop_preview()
            self._preview = _PreviewWorker(reader, meta, self._fov_index, order)
            self._preview_order = list(order)
            self._preview.tileReady.connect(self._on_preview_tile)
            self._preview.streamEnded.connect(lambda: self._recomposite("raw"))
            self._preview.failed.connect(self._on_preview_failed)
            self._preview.start()
        self._pushed = set()
        # The raw slider is 1:1 with `order`, so pushes must map into THAT, not the plate index.
        # Identity when the slider IS the whole plate; a subset map otherwise.
        #
        # REGRESSION GUARD (found on the live GUI): this map is consumed by _on_push, which DROPS
        # any push it cannot translate. That is correct for a stale run, but it silently de-scoped
        # the main view: opening an exploration tab on one well left _push_index == {0: 0} while
        # the full-plate preview kept emitting global indices 1..N-1, so every other well was
        # discarded and the FOV slider stopped advancing past the well that was clicked. The map
        # and the producer must describe the SAME well list, so record it and re-scope the raw
        # preview to match rather than leaving the two to disagree.
        self._push_order = list(order)
        self._push_index = (None if order == self._order
                            else {self._fov_index[r]["idx"]: pos for pos, r in enumerate(order)})
        if hasattr(self._detail, "register_images_bulk"):
            entries = []
            for pos, well in enumerate(order):
                w_idx = pos            # position IN THIS SLIDER (== plate idx only for a full plate)
                fov = meta["fovs_per_region"][well][0]
                for z_i, z in enumerate(meta["z_levels"]):
                    for ch in channels:
                        try:
                            path, page = reader.plane_ref(well, fov, ch, z)   # (file, page) — OME-safe
                            entries.append((0, w_idx, z_i, ch, path, page))
                        except (KeyError, IndexError, OSError):
                            continue
            self._detail.register_images_bulk(entries)
            self._pushed.update(order)   # every well is registered; double-click just navigates
        if order:                        # land on the first well so the viewer isn't blank
            self.activate_well(order[0], 0)

    def _return_to_raw(self):
        """Stop previewing/processing and restore the raw downsampled view across the whole plate."""
        if self._reader is None or self._overview is None:
            return
        self._stop_worker()
        self._active_op_key = None
        if getattr(self, "_raw_btn", None):
            self._raw_btn.hide()                             # nothing to return from now
        self._plate_mode = "raw"
        self._plate_title.setText(f"{self._acq_name}   ·   raw")
        self._overview.set_active_layer("raw")
        # The raw preview is itself a MOSAIC now (IMA-253), so returning to it restores the
        # acquisition's own boxes rather than clearing them — clearing them broke both the paint
        # (a mosaic redrawn as if it filled its cell) and the double-click FOV hit-test.
        self._overview.set_mosaic_boxes(_mosaic_boxes(self._meta))
        self._update_loupe_source()                          # back to the acquisition's own pixels
        for rc in list(self._overview._status):
            self._overview.set_status(*rc, "empty")
        self._refresh_layers_tab()
        self._setup_raw_detail()
        # resume the raw thumbnail fill — the operator run stopped the preview partway, so re-run it to
        # finish downsampling every well's raw tile (idempotent: it just re-renders the raw layer).
        self._stop_preview()
        self._preview = _PreviewWorker(self._reader, self._meta, self._fov_index, self._order)
        self._preview_order = list(self._order)
        self._preview.tileReady.connect(self._on_preview_tile)
        self._preview.streamEnded.connect(lambda: self._recomposite("raw"))
        self._preview.failed.connect(self._on_preview_failed)
        self._preview.start()
        self._readout.setText("raw view")

    def _open_computed(self):
        """Open a previously-written .hcs plate (OME-Zarr) and VISUALISE it — no recompute.

        Reads the plate/well/image OME metadata, lays out the plate, and streams each well from disk
        (a coarse pyramid level -> plate thumbnail, a ~512px level -> the ndviewer slider). Read-only."""
        import json
        d = QFileDialog.getExistingDirectory(self, "Open a computed .hcs plate")
        if not d:
            return
        base = Path(d)
        zroot = base / "plate.ome.zarr"
        if not (zroot / "zarr.json").exists():
            zroot = base if (base / "zarr.json").exists() and base.name.endswith(".zarr") else zroot
        if not (zroot / "zarr.json").exists():
            self._readout.setText("not an .hcs plate — pick a folder containing plate.ome.zarr")
            return
        # A run stopped mid-write leaves a real-looking plate.ome.zarr with only some wells in it.
        # Refuse it by name rather than silently presenting a truncated plate as a finished one.
        if (base / "INCOMPLETE").exists():
            self._readout.setText(
                f"{base.name} was stopped mid-write and is incomplete — re-run the operator "
                f"(delete the INCOMPLETE marker to open it anyway)")
            return
        try:
            plate = json.loads((zroot / "zarr.json").read_text())["attributes"]["ome"]["plate"]
            rows = [r["name"] for r in plate["rows"]]
            cols = [c["name"] for c in plate["columns"]]
            wells_meta = sorted(plate["wells"], key=lambda w: (w["rowIndex"], w["columnIndex"]))
            w0 = wells_meta[0]["path"]

            fov_fallbacks: list = []          # wells whose own image id could not be read (per-well)

            def _fov_path(well_path, default=None):
                """Each well declares its OWN first image; do not assume well 0's id fits all.

                Reusing well 0's fov path for every well silently renders the wrong image on a
                plate whose wells carry differing image ids. No dataset produces that today, so
                it stayed latent — but the loupe reads through this same mapping, and a loupe
                that magnifies a different well than the one under the cursor is precisely the
                failure the FOV seam exists to prevent. So the per-well fallback is RECORDED (see
                fov_fallbacks) and named to the user, never silently substituted."""
                try:
                    meta_w = json.loads((zroot / well_path / "zarr.json").read_text())
                    return meta_w["attributes"]["ome"]["well"]["images"][0]["path"]
                except Exception as exc:
                    if default is not None:      # a per-well lookup (not well 0's own bootstrap read)
                        fov_fallbacks.append((well_path, f"{type(exc).__name__}: {exc}"))
                    return default

            fov0 = _fov_path(w0)
            ome0 = json.loads((zroot / w0 / fov0 / "zarr.json").read_text())["attributes"]["ome"]
            levels = [ds["path"] for ds in ome0["multiscales"][0]["datasets"]]
            ms0 = ome0["multiscales"][0]
            # Pixel size, recovered from the level-0 coordinate transform. The writer collapses an
            # unknown pixel size to 1.0 (_output.py), so a plate reporting exactly 1.0 is
            # AMBIGUOUS — treat it as unknown and let the loupe say so rather than draw a scale
            # bar that might be fiction. See TODOS.md for the writer-side fix.
            px_um = None
            try:
                sc = ms0["datasets"][0]["coordinateTransformations"][0]["scale"]
                cand = float(sc[-1])
                px_um = cand if cand > 0 and abs(cand - 1.0) > 1e-9 else None
            except Exception:
                px_um = None
            chans = ome0.get("omero", {}).get("channels", [])
            channels = [{"name": c.get("label", f"ch{i}"), "display_color": "#" + c["color"].lstrip("#")}
                        for i, c in enumerate(chans)]
        except Exception as e:
            self._readout.setText(f"could not read plate metadata: {e}")
            return
        if not channels:
            self._readout.setText("plate has no channel metadata (omero) — cannot open")
            return

        self._stop_worker()
        self._stop_preview()
        self._release_loupe_sources()             # a new plate: no source (and no thread) survives
        self._acq_name, self._acq_path = base.name, base
        self._processed_plate = str(zroot)
        self._reader = None                       # a computed plate has no raw reader
        self._meta = {"channels": channels, "z_levels": [0], "n_z": 1, "n_t": 1,
                      "pixel_size_um": px_um,
                      "regions": [f"{rows[w['rowIndex']]}{cols[w['columnIndex']]}" for w in wells_meta]}
        # a computed plate replaces the whole session: drop exploration tabs (their regions belong
        # to the raw acquisition) and go back to identity push indexing over the full plate.
        self._close_exploration_tabs()
        self._active_exploration = None
        self._push_index = None
        self._run_tab_key = self._run_view_tab_key = None
        wells_rc, self._fov_index, self._order, worker_wells = {}, {}, [], []
        well_paths, well_fovs = {}, {}
        for idx, w in enumerate(wells_meta):
            ri, ci = w["rowIndex"], w["columnIndex"]
            wid = f"{rows[ri]}{cols[ci]}"
            fov = _fov_path(w["path"], fov0)          # per-well, not well 0's for everyone
            wells_rc[(ri, ci)] = wid
            self._fov_index[wid] = {"rc": (ri, ci), "idx": idx, "well_id": wid}
            self._order.append(wid)
            well_paths[wid], well_fovs[wid] = w["path"], fov
            worker_wells.append((wid, w["path"], fov, ri, ci, idx))

        if self._overview is not None:
            self._overview.setParent(None); self._overview.deleteLater()
        self._overview = PlateOverview(rows, cols, wells_rc)
        # A written plate carries no stage coordinates and no declared format, so build_plate falls
        # through to inferring the format from the well ids — which is the right and only evidence
        # here. It can fail (a plate whose wells fit no standard format); carrier art is decoration,
        # so a failure means "no background", never "cannot open the plate".
        try:
            self._plate = build_plate(self._meta, override=self._plate_format_override)
        except (PlateShapeError, PlateBuildError):
            self._plate = None
        self._overview.set_carrier(self._plate)
        self._overview.hovered.connect(self._on_hover)
        self._overview.wellActivated.connect(self.activate_well)
        self._active_op_key = "computed"
        if getattr(self, "_raw_btn", None):
            self._raw_btn.hide()                      # a computed plate has no raw to return to
        self._plate_mode = "computed MIP"
        self._plate_title.setText(f"{self._acq_name}   ·   computed MIP")
        self._op_stack.reset(); self._op_stack.add("computed", "computed MIP")
        self._overview.set_active_layer("computed")
        self._refresh_layers_tab()
        self._drop.hide()
        self._left_l.addWidget(self._overview, 1)
        self._declare_channel_axis(channels, np.uint16)
        self._enable_operators(False)             # no raw data -> operators stay disabled

        if self._detail is not None:
            self._detail.start_acquisition([c["name"] for c in channels], 1, _PUSH_PX, _PUSH_PX,
                                           [f"{w}:0" for w in self._order])
        # A written plate is read back per FOV, so its pushes are frames at the push square.
        self._push_shape = (_PUSH_PX, _PUSH_PX)
        self._push_problem = None
        self._dropped_pushes = 0
        # One or more wells could not declare their own image id, so they fell back to well 0's. On
        # a uniform plate that is harmless; on a heterogeneous one the loupe would magnify the wrong
        # field. We cannot tell which from here, so NAME it rather than hide it. It rides the success
        # line below because the plain setText calls in this method would drop a sticky suffix.
        fov_warn = ""
        if fov_fallbacks:
            shown = ", ".join(wp for wp, _ in fov_fallbacks[:3])
            more = f" (+{len(fov_fallbacks) - 3} more)" if len(fov_fallbacks) > 3 else ""
            fov_warn = (f"  ·  {len(fov_fallbacks)} well(s) could not read their own image id and "
                        f"fell back to well 0's [{shown}{more}] — the loupe may magnify the wrong "
                        f"field for them")
        # Every well came from disk, so the loupe is available across the whole plate here.
        try:
            import tensorstore as _ts
            _a = _ts.open({"driver": "zarr3", "kvstore": {"driver": "file",
                          "path": f"{zroot}/{w0}/{fov0}/{levels[0]}"}}).result()
            _well_px = int(min(_a.shape[-2], _a.shape[-1]))
        except Exception:
            _well_px = _PUSH_PX
        self._set_loupe_source("computed", _ZarrLoupeSource(
            str(zroot), path_of=well_paths.get, fov_of=well_fovs.get,
            levels=levels, well_px=_well_px, pixel_size_um=px_um, written=None))
        coarse_lvl = levels[-1]                                   # coarsest -> tiny thumbnail
        push_lvl = levels[min(3, len(levels) - 1)]                # ~512px level for the detail slider
        self._worker = _ComputedPlateWorker(str(zroot), worker_wells, coarse_lvl, push_lvl, np.uint16)
        self._worker.tileReady.connect(self._on_tile)
        self._worker.pushReady.connect(self._on_push)
        self._worker.streamEnded.connect(lambda: self._recomposite("computed"))
        self._worker.progress.connect(
            lambda i, n: self._readout.setText(f"loading computed plate — {i}/{n} wells"))
        self._worker.failed.connect(self._on_failed)
        self._worker.finished_ok.connect(
            lambda: self._readout.setText(
                f"✓ computed MIP · {len(self._order)} wells (read-only){fov_warn}"))
        self._readout.setText(f"loading computed plate · {len(self._order)} wells …")
        self._worker.start()

    # -- run a post-processing operator over the whole plate (persists a navigable OME-Zarr plate) --
    def _busy(self) -> bool:
        """True while ANY operator run is still alive — including one that was STOPPED but is
        still draining.

        ``_stop_worker`` clears ``self._worker`` immediately, while ``_retire`` keeps the thread
        running (and referenced in ``self._retired``) until it finishes its in-flight well —
        destroying a running QThread aborts the app, so it cannot do otherwise. Checking
        ``self._worker`` alone therefore lets a second run start against the same reader the
        moment a tab is closed, which is exactly the routine path IMA-205 introduces."""
        if self._worker is not None and self._worker.isRunning():
            return True
        return any(w.isRunning() for w in self._retired)

    def run_operator(self, key: str, out_parent: Optional[str] = None,
                     regions: Optional[list] = None, save: bool = True,
                     tab_key: Optional[str] = None, operator_kwargs: Optional[dict] = None):
        """Run a projector operator (MIP / reference) over the plate, or over a subset of it.

        ``regions=None`` runs the whole plate. A list runs exactly those regions, in that order —
        this is the ONE way to express a subset (the old ``preview_limit=N`` was a prefix-only
        special case of it; the preview spinner now passes ``self._order[:n]`` itself).

        save=False is PREVIEW: compute + stream results into the plate + ndviewer slider, writing
        NOTHING to disk (no folder, no disk-space cost). save=True persists a navigable OME-Zarr;
        combined with a subset it saves just those regions. Tests pass out_parent to skip the dialog.

        ``tab_key`` scopes the run to an exploration tab: results are filed under the layer
        ``<op>@<tab_key>`` so two tabs running the same operator do not overwrite each other.
        """
        if self._reader is None or self._overview is None:
            return
        if _explore.operator_busy(self._worker, self._retired):
            # NOT ``_busy()``: that also counts a retired RAW PREVIEW, and opening a side-pane tab
            # restarts the preview, so the very next operator run refused itself over a thread the
            # user never started. See _explore.operator_busy.
            self._readout.setText("already processing — let the current run finish first")
            return
        # IMA-226: gate on the ENGINE registry, not on the card table. `_OPERATIONS_BY_KEY[key]`
        # raised a bare KeyError for `reference` (a registered projector with no card) and let
        # `minerva` (a card that is not an operator) through to die inside the engine instead.
        # Refuse BY NAME here, in the readout, the same way an unknown region is refused below.
        if key not in runnable_operators():
            self._readout.setText(
                f"'{key}' is not a runnable operator — this viewer can run: "
                f"{', '.join(runnable_operators())}")
            return
        label = operator_label(key)
        # Scope the run. An explicit `regions` list still wins (the preview spinner builds one, and
        # so do tests). Otherwise the SCOPE SELECTOR on this pane decides — one control panel, one
        # place a run is aimed. Its default value is "selected wells", which resolves to the plate
        # selection and, with nothing selected, to the whole plate: byte-for-byte the behaviour
        # that existed before the selector, so nothing silently changes under an existing user.
        from_selection = False
        if regions is None:
            scope_value = (self._scope_run.currentText()
                           if getattr(self, "_scope_run", None) is not None
                           else _explore.SCOPE_SELECTION)
            regions, problem = _explore.resolve_run_scope(
                scope_value,
                selection=self._selected_regions,
                current_region=self._current_well,
                parked_subset=self.parked_subset(),
            )
            if problem:
                # A scope the user CHOSE but that has nothing behind it. Say it and stop; widening
                # it to the whole plate would be hours of compute nobody asked for.
                self._readout.setText(problem)
                return
            from_selection = (regions is not None and scope_value == _explore.SCOPE_SELECTION)
        if regions is not None:
            regions = list(regions)
            if not regions:
                self._readout.setText("empty selection — nothing to run")
                return
            unknown = [r for r in regions if r not in self._fov_index]
            if unknown:      # fail NAMED, not with a bare KeyError out of the status loop below
                self._readout.setText(
                    f"{len(unknown)} region(s) are not in this acquisition: {unknown[:3]}")
                return
        if regions is None:
            scope = "the whole plate"
        elif from_selection:
            scope = f"{len(regions)} selected well(s)"
        else:
            scope = f"{len(regions)} well" + ("s" if len(regions) != 1 else "")

        # CONFIRM THE RESOLVED TARGET SET, by name, before the QThread starts (Defect 2).
        # The selector names the RULE ("selected wells"); this names the ANSWER. They differ
        # whenever the live state the rule reads is not what the user pictures, which is the
        # entire failure mode -- and the one the deleted per-panel scope combo made worse by
        # showing a THIRD, stale answer.
        self._resolved_target = _explore.describe_run_target(regions, total=len(self._order))
        if self._resolved_target:
            self._readout.setText(self._resolved_target)
        # A PREVIEW RUN OPENS A TAB IN THE SIDE PANE. Julio: "the exploration pane can obviously
        # visualize preliminary results as the user processes... preview runs can open a TAB on
        # the exploration pane so that they look at how it is behaving, a.k.a. look at the
        # results." Keyed by operator as well as by region set, so a second preview run opens a
        # SECOND tab and the two can be compared instead of one stealing the other's canvas.
        #
        # A saved run does not (it is not a preview), and neither does a plate-wide one: the side
        # pane shows a SUBSET, and the whole dataset is what pane 2 is already looking at.
        #
        # NOTE the tab this opens is where the run is DISPLAYED, which is a different question
        # from ``tab_key`` — the tab whose LAYER the results are filed under on the plate. They
        # are kept apart deliberately: folding them together would silently re-key every
        # pane-1 preview's plate layer from "mip" to "mip@preview:…", changing what the layer
        # stack and the before/after toggle show for a feature that is only about the side pane.
        self._run_view_tab_key = None
        if not save and regions is not None and tab_key is None:
            self._run_view_tab_key = self._open_preview_tab(key, label, regions)
        elif tab_key is not None:
            self._run_view_tab_key = tab_key
        out_dir = est_gb = None
        if save:
            # Ask WHERE to persist: output can be hundreds of GB, so let the user aim it at a roomy
            # disk rather than silently filling the acquisition's. Tests pass out_parent.
            if out_parent is None:
                out_parent = QFileDialog.getExistingDirectory(self, f"Save {label} plate to folder")
                if not out_parent:
                    return
            out_dir = Path(out_parent) / f"{self._acq_name}.hcs"
            # Estimate the bytes THIS RUN writes: a subset writes len(regions)/n_wells of a plate.
            # Previously the guard was computed plate-wide and then skipped entirely for subsets
            # (`if not ok and regions is None`), so a 500-well subset save got no check at all.
            ok, est_gb, msg = self._check_disk(out_dir, regions=regions)
            if not ok:
                self._readout.setText(msg)
                return
        self._stop_preview()                                 # the operator supersedes the raw preview
        if regions is not None:                              # amber only the wells we'll actually run
            for r in regions:
                self._overview.set_status(*self._fov_index[r]["rc"], "processing")
        else:
            self._overview.set_all_status("processing")      # amber across the plate
        self._plate_mode = label                             # plate now shows this operator's result
        self._plate_title.setText(f"{self._acq_name}   ·   {label}"
                                 + (f"   ·   {exploration_tab_label(regions)}" if tab_key else ""))
        layer_key = operator_layer_key(key, tab_key)
        self._active_op_key = layer_key                      # tiles stream into this layer
        _view_tab = self._run_tab()
        if _view_tab is not None:      # ...and the tab showing this run can name that layer later
            _view_tab.plate_layer = layer_key
        if getattr(self, "_raw_btn", None):
            self._raw_btn.show()                             # now there's a processed view to return from
        stack_label = label if not tab_key else f"{label} · {exploration_tab_label(regions)}"
        self._op_stack.add(layer_key, stack_label)           # push the operator layer onto the stack
        self._overview.set_active_layer(layer_key)           # show it
        # Loupe source for this run. A SAVED run gets a zarr source whose written-well set grows
        # as wells land (so the loupe works mid-run on what's finished); a PREVIEW writes nothing,
        # so the layer gets no source and the gesture reports that rather than magnifying the
        # previous run's pixels through the same reused layer key.
        if save and out_dir is not None:
            ny, nx = self._meta["frame_shape"]
            fovs = self._meta.get("fovs_per_region")
            self._set_loupe_source(layer_key, _ZarrLoupeSource(
                str(Path(out_dir) / "plate.ome.zarr"),
                path_of=lambda w: "/".join(str(x) for x in parse_well_id(w)),
                fov_of=lambda w: _fov_of_well(w, fovs),
                levels=None,                                 # discovered from the first written field
                well_px=min(ny, nx), pixel_size_um=self._meta.get("pixel_size_um"),
                written=set()))
        else:
            self._drop_loupe_source(layer_key)
        self._refresh_layers_tab()
        # switch the detail to processed mode: z collapsed (nz=1 -> ndv drops the z-slider), frames at
        # the push size. The slider lists THIS RUN's regions — for a subset that is the subset, not the
        # whole plate (it used to always be self._order, so a subset preview built a 1536-entry slider
        # of which 4 were ever filled).
        run_order = self._order if regions is None else regions
        # _OperatorWorker emits the GLOBAL plate index (fov_index[region]["idx"]) with every push.
        # The slider we just built is indexed 0..len(run_order)-1, so translate on the way in —
        # without this every push on a subset run lands at the wrong slot or out of range.
        self._push_index = (None if regions is None
                            else {self._fov_index[r]["idx"]: pos for pos, r in enumerate(run_order)})
        # n_fovs=None = EVERY FOV in each well (IMA-187). Anything else (the historical 1) makes
        # `_boxes` empty in the worker and the plate falls back to one thumbnail per well, which is
        # the whole feature not rendering. The overview then adopts the worker's boxes so a
        # double-click on a mosaic cell resolves the FOV under the cursor instead of always 0.
        # Built BEFORE start_acquisition (IMA-245): the worker owns this run's push geometry and the
        # viewer's canvas is declared from it, so there is one rectangle, not two that agree by luck.
        self._worker = _OperatorWorker(key, self._reader, self._meta, self._fov_index,
                                       str(out_dir) if out_dir else "", regions=regions, save=save,
                                       n_fovs=None, operator_kwargs=operator_kwargs)
        self._overview.set_mosaic_boxes(self._worker.mosaic_boxes)
        # IMA-245: size the array viewer to what this run actually pushes. A REGION operator
        # (stitch, coordinate) pushes one FUSED MOSAIC per region, so the canvas is the mosaic
        # extent — declaring the frame square here handed the viewer a rectangle the mosaic does
        # not have, and the reported symptom was a black central viewer with no error anywhere.
        self._push_shape = self._worker.push_shape
        self._push_problem = None                            # sticky readout warning (see _on_push)
        self._dropped_pushes = 0                             # per RUN: this run's unrouted pushes
        if self._worker.push_shape_estimated:
            self._note_push_problem(
                "no stage positions / pixel size — the array viewer is sized as a frame, so the "
                "fused mosaic is shown squashed to that shape")
        if self._detail is not None:
            ph, pw = self._push_shape
            self._detail.start_acquisition([c["name"] for c in self._meta["channels"]], 1,
                                           ph, pw, [f"{r}:0" for r in run_order])
        self._run_out_dir = str(out_dir) if (save and out_dir) else None   # for partial-output cleanup
        self._run_tab_key = tab_key
        # A re-run must not composite on top of the LAST run's pixels: with a mosaic, a run that
        # lands fewer FOVs would otherwise leave the previous run's fields standing in the same
        # cell, blended into the new ones. Drop this layer's store before the first tile arrives.
        # ...keyed by the LAYER, not the bare operator key: an exploration tab files its results
        # under "<op>@<tab_key>" (IMA-205), and resetting "mip" from a tab run would wipe the
        # plate-wide layer instead of the tab's own.
        self._overview.reset_layer(layer_key)
        dest = f" → {out_dir.name}" if save else " (preview — not saved)"
        # This run's identity, read back by _on_progress and _on_run_tile. Held as state rather
        # than captured in a lambda because the side-pane tab has to be told the same two things
        # the status line is, and one source is how they stay in agreement.
        self._run_label, self._run_dest = label, dest
        self._worker.tileReady.connect(self._on_tile)
        self._worker.pushReady.connect(self._on_push)
        self._worker.resultReady.connect(self._on_result)
        self._worker.progress.connect(self._on_progress)
        self._worker.streamEnded.connect(lambda k=layer_key: self._recomposite(k))
        self._worker.writtenReady.connect(self._on_written)
        self._worker.wellFailed.connect(                     # a skipped well -> red x, run continues
            lambda ri, ci: self._overview.set_status(ri, ci, "failed") if self._overview else None)
        self._worker.failed.connect(self._on_failed)
        # IMA-226: report what the plate ACTUALLY got. A run where every well raised (flat-field
        # with no profile is the routine case) still reaches finished_ok — the per-well on_error
        # path is what keeps one bad file from aborting a plate — and used to print "✓" over an
        # empty plate. Landed==0 is a failure however politely the engine returned.
        def _done_msg(w=self._worker):
            if w.landed == 0:
                self._run_readout(
                    f"⚠ {label} · {scope} produced nothing — all {w.skipped or self._worker._total} "
                    f"well(s) were skipped (see the red markers)")
            elif w.skipped:
                self._run_readout(
                    f"✓ {label} · {scope}{dest} — {w.skipped} well(s) skipped"
                    + ("  (re-openable OME-Zarr)" if save else ""))
            else:
                self._run_readout(
                    f"✓ {label} · {scope}{dest}" + ("  (re-openable OME-Zarr)" if save else ""))

        self._worker.finished_ok.connect(_done_msg)
        # a run that FINISHED wrote a complete plate — forget the path so a later stop can never
        # retroactively flag it incomplete
        self._worker.finished_ok.connect(lambda: setattr(self, "_run_out_dir", None))
        # QThread.finished (not finished_ok): it fires for a FAILED or STOPPED run too, and a tab
        # switch deferred during any of those still has to be delivered. _retire only disconnects
        # the worker's own signals, so this survives a stop.
        self._worker.finished.connect(self._on_run_drained)
        # Announce the run to the activity registry the log panel's header reads. Keyed
        # "operator-run" (re-entrant by key: a new run replaces the old entry rather than stacking).
        # Ended in _on_run_drained, which fires on ok/failed/stopped alike — so the header cannot be
        # left showing a run that is over.
        self._activity.start("operator-run", f"{label} · {scope}", total=len(run_order))
        self._run_readout(f"● {label} · {scope}{dest} …")
        self._worker.start()

    def _check_disk(self, out_dir, regions: Optional[list] = None) -> tuple[bool, float, str]:
        """Estimate the persisted plate size and refuse if it won't fit (with headroom). Returns
        (ok, estimate_GB, message). Estimate = per-well projection (T·C·Y·X·itemsize) × 1.34 (the exact
        4/3 geometric sum of the 2× pyramid tail), UNCOMPRESSED. The projection collapses Z only, so
        every timepoint is preserved — a time-lapse plate writes n_t as many bytes, so n_t MUST be in
        the estimate (omitting it under-counts n_t× and lets a multi-hour time-lapse run fill the disk
        mid-write — the exact failure this guards). We do NOT discount for zstd: real fluorescence
        compresses unpredictably (often <1.2×), so assuming compression would under-estimate. An
        over-estimate only ever asks for a roomier disk, which is the safe way to be wrong."""
        import shutil
        m = self._meta
        ny, nx = m["frame_shape"]
        # Count FIELDS, not wells: the run projects every FOV (n_fovs=None), so a 36-FOV plate
        # writes 36x what a per-well count predicts. Under-counting here is how a multi-hour run
        # fills the disk mid-write with the guard reporting "plenty of room".
        # regions=None means the whole plate; a subset writes only ITS wells' fields, so count over
        # exactly those — the guard must still RUN for subsets (500 selected wells is not a rounding
        # error, and it used to be skipped entirely).
        scoped = list(self._fov_index) if regions is None else [r for r in regions if r in self._fov_index]
        n_fields = sum(len(m["fovs_per_region"][r]) for r in scoped)
        est = int(n_fields * m.get("n_t", 1) * len(m["channels"]) * ny * nx
                  * np.dtype(m["dtype"]).itemsize * 1.34)
        gb = 1024 ** 3
        try:
            free = shutil.disk_usage(Path(out_dir).parent).free
        except OSError:
            return True, est / gb, ""      # can't stat the disk — don't block
        if est > free * 0.9:
            what = "MIP" if regions is None else f"this {len(scoped)}-well run"
            return False, est / gb, (f"{what} would persist ~{est/gb:.0f} GB to {Path(out_dir).parent} "
                                     f"but only {free/gb:.0f} GB free — free space or pick another disk.")
        return True, est / gb, ""

    def _on_written(self, plate_path: str):
        """The operator finished persisting: remember the written plate (re-openable artifact)."""
        self._processed_plate = plate_path

    def _on_preview_tile(self, ri, ci, well_id, tile, box=None):
        """One preview FIELD landed. ``box`` slots it into the region's mosaic (IMA-253); ``None``
        is the single-tile path, where the field fills the cell."""
        if self._overview is not None:                       # raw preview fills the base ("raw") layer
            self._overview.add_tile(ri, ci, well_id, tile, layer="raw", box=box)

    def _on_preview_failed(self, message: str):
        """The raw preview aborted before it finished. Name it in the status line instead of
        leaving a half-grey plate that looks identical to one still loading."""
        self._readout.setText(f"the raw preview could not finish: {message}")

    def _on_tile(self, ri, ci, well_id, tile, box=None):
        """A field landed. ``box`` is None for the single-tile producers (_ComputedPlateWorker emits
        a 4-arg signal, which PyQt matches against this default) and a sub-cell box for a mosaic."""
        if self._overview is None:
            return
        layer = self._active_op_key or "raw"
        self._overview.add_tile(ri, ci, well_id, tile, layer=layer, box=box)
        self._on_run_tile(ri, ci, well_id, tile, box)       # ...and onto the run's side-pane tab
        self._overview.set_status(ri, ci, "done")           # blue
        src = self._loupe_sources.get(layer)                 # this well is now on disk -> loupe-able
        if isinstance(src, _ZarrLoupeSource):
            src.mark_written(well_id)

    def _on_result(self, region, fov, planes):
        """An operator's FULL-RESOLUTION pixels -> a toggleable napari LAYER GROUP (Defect 3).

        Julio: "what if we want to see stitched AND deconvolved AND background subed. That's
        why we need the toggles." Before this, no operator's output reached pane 2's napari at
        all: every result went to ``_on_push`` -> ``register_array``, the ndviewer slider, and
        that was the whole of "the result is visible". The group toggle UI (``_layer_tree``)
        was already built and mounted; it had nothing to show.

        ONE REGION AT A TIME, deliberately. Pane 2 shows the open region (``_mosaic_region``),
        and the raw path already drops planes for any other region (see ``_on_mosaic_plane``).
        Holding full-resolution mosaics for every well of a plate run would be gigabytes of
        layers the user cannot look at, so a result for a region that is not on screen is
        dropped here rather than accumulated -- the same rule as raw, for the same reason.

        The accumulator is per (operator, region): a plane-op emits one result per FOV and the
        layer cannot be drawn until the region is whole, while a region operator emits the
        fused region in one go. ``_op_result`` owns that difference so this slot does not.
        """
        pane = getattr(self, "_mosaic_pane", None)
        if pane is None or not getattr(pane, "ok", False):
            return                              # no napari in this window; ndviewer path stands
        if region != getattr(self, "_mosaic_region", None):
            return                              # not the region on screen -- see the docstring
        op = self._active_op_key
        if not op:
            return
        acc = self._result_acc
        if acc is None or (acc.op, acc.region) != (op, region):
            from squidmip import available_region_operators
            from squidmip._op_result import RegionResultAccumulator

            acc = RegionResultAccumulator(
                op, region, self._meta, [c["name"] for c in self._meta["channels"]],
                region_operator=(op in available_region_operators()),
            )
            self._result_acc = acc
        try:
            acc.add(int(fov), np.asarray(planes))
        except ValueError as exc:
            # NO SILENT FAILURES: a result that cannot be placed is said out loud. It must not
            # abort the run -- the pixels are still written and still on the slider.
            self._readout.setText(f"result not shown as a layer: {exc}")
            self._result_acc = None
            return
        if not acc.complete():
            return
        self._result_acc = None
        try:
            result = acc.result()
        except ValueError as exc:
            self._readout.setText(f"result not shown as a layer: {exc}")
            return
        self._add_result_layers(result)

    def _add_result_layers(self, result):
        """One layer per channel, all under the operator's group, over the raw mosaic.

        ``add_mosaic`` keys the group off ``result.op`` and ``_register_channel`` links
        contrast per CHANNEL across every group, so flipping between raw and this operator
        preserves the window -- which is the difference between a comparison and two unrelated
        pictures. ``bbox_um`` is the raw mosaic's own bbox, so the layers land in register.
        """
        from squidmip._napari_pane import _colormap_for

        pane = self._mosaic_pane
        if pane is None:
            # No central viewer any more (decentralized root). Operator results will target the
            # spawned windows in Phase C; until then, say so rather than crash on a None pane.
            self._readout.setText(
                f"{result.op}: result computed — per-window result display lands next.")
            return
        for channel in result.channels:
            pane.mosaic.add_mosaic(
                result.op, channel, result.plane(channel),
                colormap=_colormap_for(channel),
                bbox_um=result.bbox_um,
                z_scale_um=(self._meta or {}).get("dz_um"),
            )
        self._readout.setText(
            f"{result.op} — {len(result.channels)} layer(s) added; toggle it against raw in "
            f"the mosaic layers panel")

    def _on_push(self, fov_idx, planes):
        """A computed result's bounded planes -> the array viewer (in-memory register_array, LRU
        bounded). z collapsed (nz=1). One push per FOV for a per-FOV operator, one per REGION —
        the fused mosaic — for a region operator (IMA-245).

        ``fov_idx`` is the GLOBAL plate index. The slider is built from the CURRENT RUN's regions,
        so for a subset run it is only len(regions) long and the global index has to be translated
        (``_push_index``). Dropping an untranslatable push is deliberate: a push whose position we
        cannot resolve belongs to a run whose slider is gone, and guessing would paint one well's
        image onto another well's slot.

        NOTHING here is dropped silently (IMA-245). Every way a push can fail to land — no viewer,
        a viewer with no ``register_array``, an index this run's slider has no slot for, a plane
        whose shape is not the canvas we declared, or a rejection from the viewer itself — counts
        into ``_dropped_pushes`` AND says so in the readout. A black viewer with no error is what
        made the reported defect take a human to find; the swallowed ``except Exception: pass``
        below it was the last place that could have spoken and did not."""
        if self._detail is None:
            self._drop_push("there is no array viewer in this window to show the result in")
            return
        if not hasattr(self._detail, "register_array"):
            # The routine cause: an ndviewer_light build without the register_array push API. Every
            # computed result is then unshowable, which looks exactly like a viewer that is black.
            self._drop_push("this ndviewer_light build has no register_array — computed results "
                            "cannot reach the array viewer (upgrade ndviewer_light)")
            return
        pos = fov_idx if self._push_index is None else self._push_index.get(fov_idx)
        if pos is None:
            self._drop_push(f"a result for plate index {fov_idx} has no slot in this run's "
                            f"viewer — it belongs to a run whose slider is gone")
            return
        want = getattr(self, "_push_shape", None)
        channels = [c["name"] for c in self._meta["channels"]]
        for c_i, plane in enumerate(planes):
            got = tuple(np.asarray(plane).shape)
            if want is not None and got != tuple(want):
                # The producer and the declared canvas disagree — the defect class this whole file
                # keeps meeting. Say which two numbers disagree; do not push a plane the viewer
                # will reject without telling anyone.
                self._drop_push(f"the result is {got[0]}x{got[1]} but the array viewer was "
                                f"declared {want[0]}x{want[1]}")
                return
            try:
                self._detail.register_array(0, pos, 0, channels[c_i], plane)
            except Exception as e:      # one bad push must not break the run — but it must be said
                self._drop_push(f"the array viewer rejected the result: {type(e).__name__}: {e}")
                return

    def _drop_push(self, why: str):
        """Count an unrouted push and put the reason in the readout (IMA-245)."""
        self._dropped_pushes = getattr(self, "_dropped_pushes", 0) + 1
        self._note_push_problem(why)

    def _note_push_problem(self, why: str):
        """Make ``why`` a STICKY suffix on this run's readout, so a later progress/success line
        cannot overwrite it. A run that finished computing but could not display its result is not
        a success, and the '✓' must not be the last word on it."""
        if getattr(self, "_push_problem", None) == why:
            return
        self._push_problem = why
        self._run_readout(getattr(self, "_readout_base", self._readout.text()))

    def _run_readout(self, text: str):
        """Set the run's status line, re-appending any push problem this run has hit."""
        self._readout_base = text
        why = getattr(self, "_push_problem", None)
        self._readout.setText(text + (f"   ·   ⚠ {why}" if why else ""))

    def _on_failed(self, msg):
        if self._overview is not None:
            for rc, state in list(self._overview._status.items()):
                if state == "processing":
                    self._overview.set_status(*rc, "failed")  # red x on wells that didn't finish
        self._readout.setText(f"failed: {msg}")

    def _recomposite(self, layer: str):
        """End of a producer's stream: rebuild that layer once at full resolution, now that the
        running global window has seen every well (early wells were windowed by a young histogram)."""
        if self._overview is not None:
            self._overview.recomposite(layer)

    def _declare_channel_axis(self, channels, dtype):
        """Declare the plate's channel axis: labels, LUT colours and dtype.

        Colors are the RESOLVED ``display_color`` — resolve_channels already applied the precedence
        (the acquisition's YAML first, the wavelength fallback map second), so the plate is tinted
        exactly like every other compositing site.
        """
        if self._overview is None:
            return
        colors = np.stack([_hex_to_rgb01(c["display_color"]) for c in channels])
        self._overview.set_channels([c.get("display_name") or c["name"] for c in channels],
                                    colors, dtype)
        # NO STRIP UNDER THE PLATE. Julio: "Take out the window below plate view, it's
        # unnecessary." It had already lost its controls (napari owns visibility and contrast),
        # which left a row of labels restating what napari's own layer list shows two panes away.
        # A readout that duplicates a control surface is still duplication; it just cannot be
        # clicked. The channel axis is still declared above -- that is the plate's data, not a
        # widget.
        # A fresh plate must ALREADY agree with the array viewer, not merely agree from the next
        # gesture on: the viewer keeps whatever window it had, and a plate that waited for the
        # user to touch the slider would open showing a different window from the one on screen.
        self._adopt_detail_contrast()

    def _adopt_detail_contrast(self):
        """Pull the array viewer's CURRENT per-channel windows onto the plate (IMA-261)."""
        get = getattr(self._detail, "channel_windows", None) if self._detail is not None else None
        if get is None:
            return
        for ch, (lo, hi) in get().items():
            self._on_detail_contrast(ch, lo, hi)

    # -- navigation links --
    # -- selection (IMA-221): the widget picks wells, THIS window knows what a well contains ----
    def _on_selection_changed(self, wells: list):
        """PlateOverview is display-only — it maps grid cells to well ids and nothing more. The
        metadata lives here, so the expansion to (region, fov) happens here too.

            PlateOverview            PlateWindow
            [cells] --wells--> [_order sort] --fovs_per_region--> [(region, fov), ...]

        Today every well yields one FOV (the viewer is 1-FOV: write_plate(n_fovs=1)), so the pairs
        read [(B3, 0)]. The PAIR shape is the point: when per-FOV selection becomes possible (it
        needs FOV geometry that metadata doesn't carry yet) consumers don't change.
        """
        picked = set(wells)
        self._selected_regions = [w for w in self._order if w in picked]   # plate row-major
        self._update_selection_label()

    def _update_selection_label(self):
        """Show the current selection in the Selection bar ("run on selected wells")."""
        lbl = getattr(self, "_selection_label", None)
        if lbl is None:
            return
        sel = self._selected_regions
        if not sel:
            lbl.setText("none — click wells, or Select all")
        elif len(sel) <= 6:
            lbl.setText(f"{', '.join(sel)}  ({len(sel)})")
        else:
            lbl.setText(f"{', '.join(sel[:6])}, +{len(sel) - 6}  ({len(sel)})")

    def _select_all_wells(self):
        if self._overview is not None:
            self._overview.select_all()

    def _open_selected_view(self):
        """Open ONE window over the wells picked on the plate. This is the "open" gesture for a
        shift-/Cmd-CLICK selection, which (unlike a shift-DRAG) has no release to open on."""
        regions = list(self._selected_regions or [])
        if not regions:
            self._readout.setText("Pick wells first (shift/Cmd-click or Select all), then Open view.")
            return
        if self._viewer_manager.open(regions) is None:
            self._readout.setText("Open an acquisition before opening a view.")

    def _plate_channels(self) -> list:
        return [c["name"] for c in (self._meta or {}).get("channels", [])]

    def _plate_copy_luts(self):
        """Copy the plate's per-channel contrast into the shared LUT clipboard (window <-> plate)."""
        from squidmip._region_viewer import _LUT_CLIPBOARD
        ov = self._overview
        names = self._plate_channels()
        wins = ov.channel_windows() if ov is not None else []
        if not names or not wins:
            self._readout.setText("no plate channels to copy LUTs from.")
            return
        _LUT_CLIPBOARD.clear()
        for i, name in enumerate(names):
            if i < len(wins) and wins[i] is not None:
                lo, hi = float(wins[i][0]), float(wins[i][1])
                _LUT_CLIPBOARD[name] = {"clim": (lo, hi), "cmap": None}
        self._readout.setText(f"copied plate LUTs for {len(_LUT_CLIPBOARD)} channel(s).")

    def _plate_paste_luts(self):
        """Apply the shared LUT clipboard to the plate's per-channel contrast."""
        from squidmip._region_viewer import _LUT_CLIPBOARD
        ov = self._overview
        if not _LUT_CLIPBOARD:
            self._readout.setText("no copied LUTs yet — copy from a window or the plate first.")
            return
        if ov is None:
            return
        applied = 0
        for i, name in enumerate(self._plate_channels()):
            lut = _LUT_CLIPBOARD.get(name)
            if lut and lut.get("clim") is not None:
                lo, hi = lut["clim"]
                try:
                    ov.set_channel_window(i, float(lo), float(hi))
                    applied += 1
                except Exception:                        # noqa: BLE001 - one bad channel is skipped
                    pass
        self._readout.setText(f"pasted LUTs onto {applied} plate channel(s).")

    def _highlight_view_regions(self, regions):
        """A view was clicked/opened — move the plate's blue wash onto its regions."""
        if self._overview is not None:
            self._overview.highlight_regions(regions)

    def _refresh_view_hues(self):
        """Wash the plate for the CLICKED view only, in that view's own hue (Julio: "the washes only
        show when I click the view"). Different views get different hues so threads are told apart,
        but only one wash shows at a time. Cleared when there is no focused view (it was closed)."""
        if self._overview is None:
            return
        mgr = self._viewer_manager
        focused = mgr.focused_id
        v = mgr.view_for(focused) if focused is not None else None
        entries = [(v.regions, _view_hue(v.window_id, focused=True))] if v is not None else []
        self._overview.set_view_hues(entries)

    def available_views(self) -> list:
        """Every View an operator could target, UNIFIED (Spencer's operate-on-views UI binds here).

        A View is just a named region-set (see ``_region_viewer.View``), so "run on the selection",
        "run on this window", and "decon the whole plate" stop being three code paths and become one:
        run on a View's regions. "Copy the whole plate" and "select all regions" are Views too — the
        whole-plate View below IS the copy. Order: whole plate, current selection (if any), then each
        open window / ROI child. The plate's existing status highlight (amber -> done) lights a View's
        wells as the run processes them, which is the "processed wells highlight on the plate" ask."""
        from squidmip._region_viewer import View

        views: list = []
        if getattr(self, "_order", None):
            views.append(View(id="plate", name="Whole plate",
                              regions=tuple(self._order), kind="plate"))
        sel = list(getattr(self, "_selected_regions", None) or [])
        if sel:
            ordered = tuple(r for r in self._order if r in set(sel)) or tuple(sel)
            views.append(View(id="selection", name=f"Selection ({len(ordered)})",
                              regions=ordered, kind="selection"))
        views.extend(self._viewer_manager.views())
        return views

    def run_on_view(self, key: str, view) -> None:
        """Run operator ``key`` on a View's regions — the operate-on-views ENGINE hook (Julio's lane;
        the selector UI is Spencer's). Reuses ``run_operator`` unchanged, so the plate's amber->done
        status lights exactly this View's wells as they process."""
        regions = list(getattr(view, "regions", None) or [])
        if not regions:
            self._readout.setText("this view has no regions to run on.")
            return
        self.run_operator(key, regions=regions)

    def _open_views_regions(self) -> list:
        """The union of regions held by the open independent windows, in first-seen order — the
        iteration set for an operator run 'on open views' (the decentralized bulk target)."""
        seen: set = set()
        out: list = []
        for win in getattr(self._viewer_manager, "windows", []):
            for r in getattr(win, "_regions", []):
                if r not in seen:
                    seen.add(r)
                    out.append(r)
        return out

    def _on_marquee_selected(self, wells: list):
        """Shift-DRAG released on the plate -> open an INDEPENDENT napari window for that subset.

        The decentralized flow (Spencer, 2026-07-23): a selection opens a floating napari window,
        and MANY wells become ONE window with a region slider to step through them — not one window
        per well, which "is really not what anybody wants". The window is tracked by ID in the Open
        View list. Shift+CLICK still refines the selection; the drag-release is the "open" gesture.

        An empty drag (over blank plate) is a miss, not a request: return quietly rather than
        writing 'empty selection' over whatever the readout is saying."""
        if not wells:
            return
        ordered = [w for w in self._order if w in set(wells)] or list(wells)  # plate row-major
        win = self._viewer_manager.open(ordered)
        if win is None:
            self._readout.setText("Open an acquisition before opening a view.")

    def selected_region_fovs(self) -> list:
        """The current selection as (region, fov) pairs — the payload IMA-205 will consume."""
        per = (self._meta or {}).get("fovs_per_region", {})
        return [(r, f) for r in self._selected_regions for f in (per.get(r) or [0])]

    def _on_hover(self, text: str):
        # BOTTOM-LEFT plate title bar: "<acq>  ·  <mode>" (mode = raw / the operator that processed it),
        # plus the hovered well when the cursor is over the plate.
        base = f"{self._acq_name or 'well plate'}   ·   {self._plate_mode}"
        self._plate_title.setText(f"{base}   ·   {text}" if text else base)

    def _slider_pos(self, well_id: str) -> Optional[int]:
        """Where ``well_id`` sits in the detail's CURRENT FOV slider, or None if it isn't in it.

        The slider is whole-plate by default (position == plate index) but an exploration tab
        scopes it to a subset, where the two diverge. Everything that hands ndviewer an index —
        register_image, register_array, go-to — has to translate through here."""
        info = self._fov_index.get(well_id)
        if info is None:
            return None
        if self._push_index is None:
            return info["idx"]
        return self._push_index.get(info["idx"])

    def activate_well(self, well_id: str, fov_index: int):
        """Double-click -> show the well in the ndviewer. In RAW mode (no operator run yet) push the
        well's raw z-stack lazily (the true z-stack, zero bytes copied). In PROCESSED mode (an operator
        has run, the slider already holds the results) just navigate the slider to that well."""
        if well_id not in self._fov_index:
            return
        # Resolve the slider position BEFORE moving anything. The red frame says "this is the well
        # you are looking at"; if the detail's slider does not contain the well (an exploration tab
        # scopes it to a subset) we cannot show it, and moving the frame anyway is how you get a red
        # frame on one well and another well's pixels beside it — silently.
        idx = self._slider_pos(well_id) if self._detail is not None else None
        if self._detail is not None and idx is None:
            self._readout.setText(
                f"{well_id} is not in this tab's subset — switch to 'Process wells' to open it")
            return
        self._current_fov = fov_index                  # the FOV ON SCREEN (IMA-250 (b))
        # ONE move. The cursor drives the red frame, the region slider and pane 2's mosaic
        # together, so they cannot disagree. This used to be three statements on three different
        # code paths, and under napari (`_detail is None`) it returned before ANY of them ran:
        # a double-click loaded the mosaic and left the red frame on the previous region.
        try:
            self._cursor.activate(well_id)
        except KeyError:
            self._readout.setText(f"{well_id} is not in the current region order")
            return
        if self._detail is None:
            # Decentralized root: double-click opens ONE independent window on this region (the
            # single-region case of the shift-drag gesture). Many regions -> shift-drag a box.
            win = self._viewer_manager.open([well_id])
            if win is None:
                self._readout.setText("Open an acquisition before opening a view.")
            return
        if self._active_op_key is not None or self._reader is None:   # processed/computed: already pushed
            self._detail.go_to_well_fov(well_id, fov_index)
            return
        if well_id not in self._pushed:
            fov = self._meta["fovs_per_region"][well_id][0]
            for z_i, z in enumerate(self._meta["z_levels"]):
                for ch in (c["name"] for c in self._meta["channels"]):
                    try:
                        path, page = self._reader.plane_ref(well_id, fov, ch, z)
                        self._detail.register_image(0, idx, z_i, ch, path, page)
                    except (KeyError, IndexError, OSError, RuntimeError):
                        continue   # a genuinely-missing plane / closed viewer shouldn't block the rest
            self._pushed.add(well_id)
        # Region ids are not necessarily well ids: a slide carrier's are freeform ("R2C3",
        # "region_A", "tissue-1"). This used to rebuild the id as f"{row}{col}" from
        # parse_well_id, which RAISES on all of those, inside a bare except that swallowed it -
        # so a double-click moved the red box and never navigated, silently. The id is only a
        # label for the detail viewer's well/FOV combo, so it is passed through untouched.
        self._detail.go_to_well_fov(well_id, fov_index)

    def _on_fov_slider(self, flat_idx: int):
        """ndviewer_light's own slider moved -> move the CURSOR (which moves the red frame).

        This is the FALLBACK viewer's slider; under napari the navigation control is
        ``_region_slider``. Both land in the same cursor, so there is still exactly one owner.

        The labels are ``f"{region}:0"`` in raw mode (IMA-270's ``r:0``), so the region id is
        everything before the colon. The label is a DISPLAY string and this is the only place it
        is read back; nothing downstream parses it.
        """
        if self._detail is None or self._overview is None:
            return
        labels = getattr(self._detail, "_fov_labels", None)
        if not labels or not (0 <= flat_idx < len(labels)):
            return
        region = labels[flat_idx].split(":")[0]
        if self._cursor.position_of(region) is not None:
            self._cursor.set_region(region)

    def _on_detail_contrast(self, ch: int, lo: float, hi: float):
        """The CENTRAL ARRAY VIEWER re-windowed channel *ch*. Make the plate show that window.

        This is the whole of the cross-repo sync (IMA-261), and it is deliberately one-way:
        ndviewer_light owns contrast, the plate follows. `(lo, hi)` are the numbers ndv handed its
        own canvas — not a re-derivation from a slider position, not a percentile recomputed here
        — so "the plate and the viewer show the same window" is true by construction rather than
        by two rules being kept in step.

        It lands in the plate's FOLLOW path, NOT in its manual latch. ndv autoscales by itself —
        at open, and again whenever the displayed data changes — so treating each broadcast as a
        user gesture latched every channel MANUAL before anyone had touched anything: the plate's
        running auto-contrast was dead from the first frame, and SCOPE_PER_REGION painted every
        well under ndv's one global window while the plate still drew the amber "wells NOT
        comparable" badge over the top. A sink records what the owner resolved; only the user sets
        policy. `_RunningContrast.resolve` is still the single precedence rule.
        """
        if self._overview is None:
            return
        n_ch = len(self._overview._labels)
        if not (0 <= ch < n_ch):
            return          # ndv drew a channel the plate does not have (RGB mode, or a re-ingest)
        self._overview.follow_channel_window(ch, float(lo), float(hi))

    def _sync_focus_button(self):
        """Show the reference-plane button only for a z-stack. A 2D acquisition (one z level) has
        nothing to focus, so the button would jump the slider to the plane it is already on."""
        btn = getattr(self, "_focus_btn", None)
        if btn is None:
            return
        btn.setVisible(len((self._meta or {}).get("z_levels", [])) > 1)

    def _populate_detect_channels(self):
        """Fill the 'Detect on' dropdown with this acquisition's channels, defaulting to the one
        most likely to carry nuclei. Channel-aware cellpose: the user segments the channel that has
        signal, not whatever happens to be visible (405 is blank on the tissue set)."""
        pane = getattr(self, "_mosaic_pane", None)
        combo = getattr(pane, "detect_channel", None) if pane is not None else None
        if combo is None:
            return
        names = [c["name"] for c in (self._meta or {}).get("channels", [])]
        prev = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(names)
        # Prefer a previously chosen channel if it still exists, else a 405/nuclei/DAPI-looking one.
        pick = prev if prev in names else next(
            (n for n in names if any(t in n.lower() for t in ("405", "dapi", "hoechst", "nuclei"))),
            names[0] if names else "")
        if pick:
            combo.setCurrentText(pick)
        combo.blockSignals(False)

    def _focus_reference_plane(self):
        """Jump THE z SLIDER to the current FOV's sharpest plane (Tenengrad autofocus).

        A BUTTON, not a plate operator: it is per-FOV, on demand, and nothing is saved.

        Two things were wrong with the previous version and both are fixed here.

        1. IT WAS PERMANENTLY DEAD under napari. It required ``self._detail``, which is ``None``
           whenever napari is the viewer (dc0f288), so the click returned immediately and told
           the user to "double-click a well first" — a well they had just double-clicked. A
           visible button with zero function is worse than no button. The z slider it must move
           is NAPARI'S OWN, the one 19cd491 made real behind a lazy ``(z, y, x)`` stack.
        2. IT RANKED EVERY PLANE ON THE GUI THREAD. ~40-50 TIFF reads inside a ``clicked`` slot,
           with no worker, no progress and no cancel: the window froze solid for the duration.
           It was the only long operation in the app without a QThread. Now it is one.

        The region comes from the CURSOR, so "which region" is the same value the red frame and
        the region slider show — there is no separate notion of "the well the detail viewer has".
        """
        if self._reader is None or self._meta is None:
            self._readout.setText("open an acquisition before focusing a reference plane")
            return
        well = self._cursor.region
        if well is None:
            self._readout.setText("no region is selected — nothing to focus")
            return
        z_levels = list(self._meta.get("z_levels") or [])
        if len(z_levels) <= 1:
            self._readout.setText(
                f"{well}: this acquisition has a single z plane, so there is no reference "
                "plane to find.")
            return
        prior = getattr(self, "_focus_worker", None)
        if prior is not None and prior.isRunning():
            self._readout.setText("still ranking planes for the last request — one at a time.")
            return
        # The FOV IN VIEW, not the region's first one (IMA-250). This is a per-FOV autofocus, so
        # ranking field 0 while the viewer shows field 12 reports the sharpest plane of pixels the
        # user is not looking at. Falls back to the region's first FOV when the one on screen is
        # not one of its own (a freshly-scoped detail, or a region with a single field).
        fovs = self._meta["fovs_per_region"][well]
        fov = self._current_fov if self._current_fov in fovs else fovs[0]
        chan = self._meta["channels"][0]["name"]        # rank on one representative channel
        self._focus_btn.setEnabled(False)
        self._readout.setText(f"{well}:{fov} — ranking {len(z_levels)} planes for focus …")
        w = _FocusWorker(self._reader, self._meta, well, fov, chan, parent=self)
        w.ready.connect(lambda z_i, note, r=well, f=fov: self._on_reference_plane(r, f, z_i, note))
        w.problem.connect(self._on_focus_problem)
        self._focus_worker = w
        w.start()

    def _on_focus_problem(self, msg: str):
        self._focus_btn.setEnabled(True)
        self._readout.setText(f"focus reference plane: {msg}")

    def _on_reference_plane(self, well: str, fov: int, z_index: int, note: str = ""):
        """The sharpest plane is known. MOVE THE Z SLIDER to it."""
        self._focus_btn.setEnabled(True)
        moved = self._set_z_index(z_index)
        if not moved:
            self._readout.setText(
                f"{well}:{fov} sharpest plane is z={z_index}, but no z slider could be moved — "
                "the viewer is showing a single plane.")
            return
        self._readout.setText(
            f"{well}:{fov} focused on reference plane z={z_index} (sharpest){note}")

    def _set_z_index(self, z_index: int) -> bool:
        """Drive THE z slider to *z_index*. Returns whether a slider actually moved.

        There is one z control per viewer and this finds it rather than owning a second one:
        napari's ``dims`` when napari is the viewer, ndviewer_light's ``set_current_index`` in
        the fallback. The boolean is the point — a "focused" message printed over a slider that
        never moved is exactly the silent failure this method exists to end.
        """
        axis = self._napari_z_axis()
        if axis is not None:
            dims = self._napari_dims()
            top = int(dims.nsteps[axis]) - 1
            if top < 1:
                return False
            dims.set_current_step(axis, max(0, min(int(z_index), top)))
            return int(dims.current_step[axis]) == max(0, min(int(z_index), top))
        setter = getattr(self._detail, "set_current_index", None) if self._detail else None
        if setter is None:
            return False
        setter("z_level", int(z_index))
        return True

    def _retire(self, w):
        """Retire a worker thread WITHOUT ever destroying a running QThread (that aborts the app).
        Disconnect its signals first so a tile already queued before the stop can't paint onto a
        freshly-opened plate (the cross-plate corruption the review found); then keep a reference
        alive until it actually finishes (stop() returns after the current item, which is bounded).

        The signal list is DISCOVERED from the worker class, not hardcoded. It used to be a literal
        tuple of names, which silently failed open: a worker declaring a signal absent from that
        tuple kept it connected through teardown and could paint onto the next plate — the very bug
        this method exists to prevent, re-armed by every new worker. Introspection makes a new
        worker correct by construction."""
        if w is None:
            return
        for name in _signal_names(type(w)):
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
            # _busy() counts EVERY retired thread (the raw preview included), so a deferred tab
            # switch can only be delivered once the last one exits — hook them all, not just the
            # operator run, or the resync waits for an event that never comes.
            w.finished.connect(self._on_run_drained)

    def _stop_worker(self):
        self._retire(self._worker)
        self._worker = None

    def _stop_preview(self):
        self._retire(self._preview)
        self._preview = None

    def _stop_minerva(self):
        self._retire(self._minerva)
        self._minerva = None

    def _join_retired(self, msec: int = 3000) -> None:
        """WAIT for every deferred worker, at the one moment deferring is not allowed: teardown.

        ``_retire`` is right for the normal case — it disconnects the signals and lets the thread
        drain in the background so the GUI never blocks on a stop. But every worker is parented to
        this window, so once the window is destroyed Qt destroys a QThread that is still running
        and the PROCESS ABORTS. Deferring is only safe while the parent outlives the thread.

        So closing joins instead. Bounded: ``stop()`` returns after the current item, and a thread
        that misses the deadline is detached from this window so its destruction is no longer tied
        to ours — a slow worker must not hang the close, and it must not abort the app either.
        """
        for w in list(self._retired):
            try:
                if w.isRunning():
                    w.stop()
                    if not w.wait(msec):
                        w.setParent(None)     # outlived us: cut it loose rather than abort
            except RuntimeError:              # already destroyed by Qt
                pass
        self._retired.clear()

    def _stop_mosaic_worker(self):
        """Stop the pane-2 fuse and WAIT for it, before the window that owns it is destroyed.

        This one is not like the others: ``_retire`` lets a thread drain in the background, which
        is right for a worker whose owner outlives it. ``_MosaicWorker`` is parented to this
        window, so when Qt destroys the window it destroys a QThread that is still running and
        the process ABORTS. Only the replace path in ``_load_mosaic`` ever stopped it, so a close
        (or a second ingest) mid-fuse killed the app. It went unnoticed while a fuse was fast;
        the multiscale pyramid made the fuse long enough to still be running on close.
        """
        workers = [getattr(self, "_mosaic_worker", None)]
        # ...and one per EXPLORATION TAB. Each tab fuses its own subset with its own worker, and
        # like pane 2's it was only ever stopped when REPLACED. They accumulate: one per tab per
        # region visited, all parented to this window, all still running when it is destroyed.
        for tab in list(getattr(self, "_op_tabs", {}).values()):
            workers.append(getattr(tab, "mosaic_worker", None))
        for w in workers:
            if w is None:
                continue
            try:
                w.stop()
                w.wait(2000)
            except RuntimeError:      # already destroyed by Qt; nothing left to join
                pass
        self._mosaic_worker = None
        for tab in list(getattr(self, "_op_tabs", {}).values()):
            if getattr(tab, "mosaic_worker", None) is not None:
                tab.mosaic_worker = None

    def showEvent(self, e):
        """Take a GUI slot the moment this window becomes VISIBLE.

        The cap cannot live in ``main()`` alone: every proof script and debug launcher builds a
        ``PlateWindow`` directly and never goes through it, which is exactly how Julio's screen
        filled up. This is the one call every visible window makes, whoever constructed it.

        A refusal closes the window rather than raising: an exception out of showEvent leaves a
        half-built top-level on screen, which is the state we are trying to prevent.
        """
        if _gui_cap_applies() and getattr(self, "_gui_slot", None) is None:
            try:
                self._gui_slot = acquire_gui_slot()
            except GuiAlreadyOpen as exc:
                print(f"squidmip-view: {exc}", file=sys.stderr)
                self._gui_slot = None
                QTimer.singleShot(0, self.close)   # unwind out of showEvent first, then close
                return
        super().showEvent(e)
    def _stop_spots(self):
        self._retire(getattr(self, "_spot_worker", None))
        self._spot_worker = None

    def closeEvent(self, e):
        release_gui_slot(getattr(self, "_gui_slot", None))   # let the next window open
        self._gui_slot = None
        self._stop_worker()          # stop the run cleanly; nothing on disk to clean up (no cache)
        self._stop_preview()
        self._stop_mosaic_worker()   # JOINED, not drained: it is parented to this window
        self._join_retired()         # ...and so is everything _retire deferred
        self._stop_spots()           # never leave the segmentation thread running at teardown
        self._stop_minerva()         # files already written stay; only the launch poll is abandoned
        for key in list(self._floating):   # floated tabs are top-levels of their own — Qt won't
            win = self._floating.pop(key)  # close them for us, and each may hold a live shell
            w = win.take_content()
            if w is not None:
                self._dispose_tab_widget(w)
            win.close()
        self._release_loupe_sources()   # joins the loupe read thread
        for w in list(self._op_tabs.values()):
            if hasattr(w, "shutdown"):
                w.shutdown()         # kill any live embedded terminal's shell
        for w in list(self._retired):
            w.wait()                 # join before exit — never leave a QThread running at teardown
        panel = getattr(self, "_log_panel", None)
        if panel is not None:
            panel.stop()             # stop the memory-poll timer before the widget is torn down
        bus = getattr(self, "_log_bus", None)
        if bus is not None:
            bus.uninstall()          # detach from the root logger so a closed window stops logging
        super().closeEvent(e)


def _rss_mb() -> tuple:
    """(peak_MB, current_MB_or_None). Peak = the OS high-water mark (ru_maxrss), so it is exact even
    without sampling. Current RSS needs psutil (optional). Returns (0, None) where resource is absent."""
    peak = 0.0
    try:
        import resource
        m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = m / (1024 * 1024) if sys.platform == "darwin" else m / 1024   # darwin: bytes, linux: KB
    except Exception:
        pass
    cur = None
    try:
        import os as _os

        import psutil
        cur = psutil.Process(_os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    return peak, cur


def _install_footprint_monitor(app, win):
    """Track the process memory footprint and PRINT THE PEAK when the GUI closes or crashes.

    A light QTimer prints a live line every few seconds so you can watch the footprint as you drive
    the GUI (open a plate, run MIP, scrub FOVs); the peak is the OS high-water mark, so the final
    number is exact regardless of sampling. Wired to app-quit (normal close), atexit, and the
    excepthook (crash) — so a peak is always reported. Unix only (no-ops where `resource` is absent)."""
    import atexit

    state = {"peak": 0.0, "done": False}

    def _live():
        peak, cur = _rss_mb()
        state["peak"] = max(state["peak"], peak)
        cur_s = f", current {cur:.0f} MB" if cur is not None else ""
        print(f"[footprint] peak {state['peak']:.0f} MB{cur_s}", flush=True)

    def _final(reason: str):
        if state["done"]:
            return
        state["done"] = True
        peak, _ = _rss_mb()
        state["peak"] = max(state["peak"], peak)
        print(f"\n[footprint] FINAL peak RSS: {state['peak']:.0f} MB  ({reason})", flush=True)

    timer = QTimer()
    timer.timeout.connect(_live)
    timer.start(5000)
    win._footprint_timer = timer            # keep a reference alive
    app.aboutToQuit.connect(lambda: _final("window closed"))
    atexit.register(lambda: _final("process exit"))
    _orig_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        _final(f"CRASH: {exc_type.__name__}: {exc}")
        _orig_hook(exc_type, exc, tb)

    sys.excepthook = _hook


# --- how many GUI windows may be open AT ONCE, across processes (Julio, IMA-window-cap) -----
#
# Every agent proof run opened another PlateWindow and left it there, until the screen was full
# and swap was at 5.8 GB of 7. Nothing in the app said no, so the cap has to live HERE, at the
# one place a real instance starts -- not in whatever script happened to launch it.
#
# flock on a slot file, deliberately NOT a pidfile. A pidfile must be cleaned up, and a GUI that
# is killed or crashes never cleans up; that is precisely how these runs ended, so a pidfile
# would have wedged the app permanently shut. The kernel drops an flock when the holder dies
# however it dies, so a crashed window frees its slot with no recovery path to get wrong.

DEFAULT_MAX_GUI = 1


class GuiAlreadyOpen(RuntimeError):
    """Refusing to open another GUI window: the cap is already used up."""


class _GuiSlot:
    """A held slot. Keep the reference alive: closing ``fd`` releases the lock."""

    __slots__ = ("fd", "path")

    def __init__(self, fd: int, path: Path) -> None:
        self.fd = fd
        self.path = path


def gui_slot_limit() -> int:
    """How many GUI windows may be open at once. ``SQUIDMIP_MAX_GUI`` overrides."""
    raw = os.environ.get("SQUIDMIP_MAX_GUI")
    if not raw:
        return DEFAULT_MAX_GUI
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_GUI


def _gui_lock_dir() -> Path:
    d = Path(os.environ.get("SQUIDMIP_GUI_LOCK_DIR")
             or (Path.home() / ".cache" / "squidmip"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def acquire_gui_slot() -> _GuiSlot:
    """Take one of the ``gui_slot_limit()`` slots, or raise :class:`GuiAlreadyOpen`.

    Returns a handle whose lifetime IS the reservation -- hold it for as long as the window
    lives. Never blocks: a GUI that hangs waiting for another GUI to exit is a worse bug than
    the one this prevents.
    """
    import fcntl

    limit = gui_slot_limit()
    lock_dir = _gui_lock_dir()
    for slot in range(limit):
        path = lock_dir / f"gui-{slot}.lock"
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)          # somebody else holds this slot; try the next one
            continue
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        return _GuiSlot(fd, path)

    raise GuiAlreadyOpen(
        f"a SquidMIP window is already open ({limit} allowed at once). Close it first, or "
        f"raise the cap with SQUIDMIP_MAX_GUI=<n>. Lock dir: {lock_dir}"
    )


def release_gui_slot(handle: Optional[_GuiSlot]) -> None:
    """Give the slot back. Idempotent, and safe on an already-closed handle."""
    if handle is None:
        return
    try:
        os.close(handle.fd)       # closing the fd releases the flock
    except OSError:
        pass                      # already closed (or the holder died) -- the lock is gone either way


def _gui_cap_applies() -> bool:
    """The cap guards REAL windows only.

    Offscreen runs (the test suite, and every automated proof) never put anything on Julio's
    screen, and capping them would serialise the suite for no benefit.
    """
    return os.environ.get("QT_QPA_PLATFORM") != "offscreen"


def main(dataset_path: str = None):
    path = dataset_path or (sys.argv[1] if len(sys.argv) > 1 else None)
    slot = None
    if _gui_cap_applies():
        try:
            slot = acquire_gui_slot()
        except GuiAlreadyOpen as e:
            print(f"squidmip-view: {e}", file=sys.stderr)
            sys.exit(1)
    app = QApplication.instance() or QApplication(sys.argv)
    win = PlateWindow(path)
    _install_footprint_monitor(app, win)
    win._gui_slot = slot                  # the reservation lives as long as the window
    win.show()
    if not app.property("_squidmip_test"):
        try:
            sys.exit(app.exec_())
        finally:
            release_gui_slot(slot)
    return win


if __name__ == "__main__":
    main()
