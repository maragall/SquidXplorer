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
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import (  # QThread/pyqtSignal: kept for tests that build a stub worker as V.QThread
    Qt, QThread, QTimer, pyqtSignal,  # noqa: F401 (see tests/test_viewer.py::_BlockingWorker)
)
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMenu, QPlainTextEdit, QPushButton, QScrollArea, QSlider, QSpinBox,
    QSplitter, QStackedWidget, QStyleFactory, QTabBar, QVBoxLayout, QWidget,
)

#: The one Fusion QStyle for this process, created on first use. It must NOT be per-window.
#:
#: ``QWidget.setStyle()`` does not take ownership: the widget keeps a bare pointer and Qt never
#: tells it when the style dies. Holding the only reference on the window meant the style was
#: freed with the window's ``__dict__``, before ``~QWidget`` ran -- and every widget destructor
#: calls ``style()->unpolish(this)``. Building and dropping windows therefore corrupted the heap
#: and segfaulted the process, which is what killed whole pytest runs (macOS SIGSEGV here, Windows
#: 0xC0000005) and took the summary line with them. Measured 2026-07-28: the interpreter died on
#: the 6th window; with the style kept alive, 40 survived. Pinned by tests/test_window_lifetime.py.
#:
#: A style is stateless chrome and Qt shares one app-wide by default, so one per process is also
#: simply the right object count. ``None`` if this Qt build has no Fusion, which callers guard for.
_FUSION_STYLE = None
_FUSION_STYLE_MADE = False

#: The one QApplication for this process, PINNED here for the life of the process.
#:
#: Same class of bug as the style above, one level up. A QApplication is a process singleton that
#: every widget, every QStyle and every posted event points back at, and PyQt gives its ownership
#: to PYTHON: the last Python reference dying deletes it, whatever is still standing. Callers keep
#: that reference in a place with a much shorter life than the process. ``main()`` held it in a
#: local and then handed the WINDOW back to its caller, and every test module holds it in a
#: module-scoped pytest fixture, whose cache pytest clears at the last test's teardown -- which is
#: exactly where whole suites died, after the last test passed and before the summary printed.
#: ``tests/test_channel_bar.py`` had already hit this and pinned its own copy in a global.
#:
#: Measured 2026-07-28 on a parametrised window fixture, 60 build/destroy cycles per run, 12 runs
#: per configuration: unpinned, the summary line printed 0 times out of 12; with the QApplication
#: pinned and NOTHING else changed, 12 out of 12. Pinned by tests/test_window_lifetime.py.
#:
#: One per process is also simply the right object count: ``QApplication.instance()`` is Qt saying
#: so. Nothing here creates a second one.
_APP = None


def qt_app(argv=None):
    """The process's QApplication, created if needed, and held so Python cannot free it early.

    Every entry point should go through this instead of ``QApplication.instance() or
    QApplication([])``, which binds the app to whatever local or fixture cache happens to be at
    hand. Returns the existing instance untouched when Qt already has one, so a host application
    that made its own is adopted rather than duplicated.
    """
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication(list(argv) if argv is not None else [])
    return _APP


def _fusion_style():
    """The shared Fusion style, or None. Created lazily: a QStyle needs a live QApplication."""
    global _FUSION_STYLE, _FUSION_STYLE_MADE
    if not _FUSION_STYLE_MADE:
        _FUSION_STYLE = QStyleFactory.create("Fusion")
        _FUSION_STYLE_MADE = True
    return _FUSION_STYLE


#: The main window's logger. The log panel taps the stdlib ROOT logger, so anything logged here
#: appears in the bottom-right panel for free — the reason a failure the user triggers (a spot
#: detection that raised, a region that would not fuse) MUST go through this and not only into an
#: in-widget banner: a banner the user has already clicked past leaves no trace, and "the logger
#: didn't show it" is the exact gap this closes.
from squidmip._logpane import get_logger

log = get_logger("viewer")

from squidmip import _explore, _qtstyle
from squidmip.contract import field_path
from squidmip._engine import available_projectors
from squidmip._layers import OperationStack
from squidmip._minerva import MINERVA_HOME_ENV as _MINERVA_HOME_ENV
from squidmip._montage import _hex_to_rgb01
from squidmip._output import parse_well_id
from squidmip._activity import ActivityLog
from squidmip._address import Extent
from squidmip._logpane import LogBus, ViewLog
from squidmip._logpanel import LogPanel
from squidmip._plate import PlateBuildError, build_plate
from squidmip._plate_shape import PlateShapeError
from squidmip._qt_tabs import _DetachTabBar, _DetachTabs, _FloatWindow  # noqa: F401 (re-export)
from squidmip._qtstyle import dark_palette as _dark_palette
from squidmip._qtstyle import hline as _hline
from squidmip._terminal import _CmdEdit, _ProcTerminal, _Terminal  # noqa: F401 (re-export)
from squidmip._region_nav import RegionCursor, RegionSlider
from squidmip._spots import LAYER_KEY as _SPOTS_LAYER_KEY

# The PLATE OVERVIEW and the plate geometry under it moved to `squidmip._plate_overview` (gap 6,
# 2026-07-29): 2,459 lines, cut along the seam this file already had a comment for. Re-exported
# here under their historical names so every caller, and the ~40 tests that reach in through
# `from squidmip import _viewer as V`, are unchanged. The arrows run one way only: that module
# imports nothing from this one. (The `_qtstyle` colour aliases are NOT re-exported: both modules
# alias the same single definition in `squidmip._qtstyle`, which is where they belong.)
from squidmip._plate_overview import (  # noqa: F401 (re-exports)
    _CELL, _CLICK_SLOP, _COLH, _HDR, _LOUPE_CACHE, _LOUPE_HOLD_MS, _LOUPE_MAG, _LOUPE_MAX_CROP,
    _LOUPE_PX, _LOUPE_SLOP, _LOUPE_WIN_LOCK, _PAD, _PCT, _PLATE_DIMS, _PUSH_PX, _TILE_CACHE_BYTES,
    _TILE_QUEUE_MAX, _VIEW_WASH,
    PlateOverview, _LoupeSource, _LoupeWorker, _RawLoupeSource, _RunningContrast, _TileFetcher,
    _ZarrLoupeSource, _box_union, _deep_zoom_enabled, _fit_box, _fit_cell, _fit_letterboxed,
    _fmt_um, _fov_of_well, _mosaic_boxes, _nice_scale_um, _pct_window, _plate_grid, _row_letter,
    cells_in_rect, loupe_clamp_crop, loupe_crop_px, loupe_decimation, loupe_level, loupe_scale,
    loupe_um_per_screen_px, push_shape_for, region_mosaic_extent_px, resolve_plate_root, well_at,
)

# The eight QThread workers moved to `squidmip._workers` (gap 6, 2026-07-29): 949 lines whose only
# common property is the Qt threading rule, that a background thread may not touch a widget, which
# is why none of them needed the window. Re-exported here under their historical names: several
# tests swap one out with `monkeypatch.setattr(V, "_OperatorWorker", ...)` and `PlateWindow`
# resolves the name in THIS module's namespace, so the spies keep working. `_region_viewer` also
# imports three of them from here inside function bodies.
from squidmip._workers import (  # noqa: F401 (re-exports)
    _CACHE_AUTO, _MIN_PREVIEW_BOX_PX, _VIEWER_WORKERS,
    _ComputedPlateWorker, _FlatfieldWorker, _FocusWorker, _MinervaWorker, _MosaicWorker,
    _OperatorWorker, _PreviewWorker, _SpotWorker, _full_res_mip, _full_res_plane, _spot_stages,
)

# (`_SUPPORTED_PLATES` and `resolve_plate_format` used to live here. `build_plate` (IMA-214) is now
#  the single format-resolution path — override > measured > declared > inferred — so both were dead
#  leftovers that could only ever disagree with it. Deleted rather than left as a second opinion.)

# Chrome (colours, stylesheets, palette) is defined ONCE in `squidmip._qtstyle` and aliased here
# so existing call sites keep their short private names. These are NOT second definitions: change
# a colour in _qtstyle and every widget in the window moves with it.
_BG = _qtstyle.BG
_GRID, _RED, _MUTED, _ACCENT = _qtstyle.GRID, _qtstyle.RED, _qtstyle.MUTED, _qtstyle.ACCENT
_SEL_FILL = _qtstyle.SEL_FILL


def _view_hue(view_id: int, *, focused: bool = False) -> QColor:
    """A STABLE, distinct hue per open view/thread, so the plate colour-codes which wells belong to
    which window (Julio: "colour hueing the different view threads"). The golden-ratio hue step keeps
    successive views far apart on the wheel; the focused view is more opaque so it reads as active."""
    h = (0.13 + 0.61803398875 * int(view_id)) % 1.0     # golden-ratio walk => maximally spread hues
    c = QColor.fromHsvF(h, 0.62, 1.0, 0.34 if focused else 0.20)
    return c


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
# The line below used to name "Right-click a well and choose Control Well". `set_control_well`
# and `PlateOverview._context_menu` were deleted wholesale in 2b8fbc5 (Decentralize GUI), so it
# taught a gesture the user cannot perform, which the comment below calls worse than silence.
# Corrected 2026-07-28 to name Shift-drag, which is the gesture that actually exists.
_EMPTY_EXPLORE_PRIMARY = (
    "Shift-drag across the plate to open the wells you select in their own window, so you can "
    "compare them side by side.")
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

#: ``font-size: 12px`` in a stylesheet, with the number as group 1. Only ``px`` — a ``pt`` size
#: is already device-independent and must NOT be scaled a second time.
_QSS_FONT_PX_RE = re.compile(r"(?<=font-size:)\s*(\d+(?:\.\d+)?)\s*px", re.IGNORECASE)


def _scale_qss_fonts(qss: str, scale: float) -> str:
    """Multiply every ``font-size:Npx`` in *qss* by *scale*, leaving the rest untouched.

    Only the font size moves. Scaling paddings and borders too was tried and looked wrong: the
    dark cards are drawn with 1 px hairlines that turn into muddy 2-3 px rules the moment they
    are multiplied, and Qt already scales the whole sheet by the device pixel ratio.

    Sizes floor at 8 px so shrinking the window toward the minimum cannot produce type nobody
    can read.
    """
    def _sub(m):
        return f"{max(8, round(float(m.group(1)) * scale))}px"

    return _QSS_FONT_PX_RE.sub(_sub, qss)


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
#: Height of the top strip when you are working the plate: the plate is the star, and a fixed cap
#: stops the operator cards' size hint ballooning it into the "super thick" top that squashed the
#: plate (Julio).
_TOP_ROW_COMPACT_PX = 240

#: ...and its height while the Log tab is in front. A console you cannot read is not a console: 240
#: px is about ten lines, which is a status light rather than a log. When you deliberately select
#: the Log tab you are reading it, not watching the plate, so the strip earns more room and gives it
#: straight back when you leave. This is the layout half of the same fix as `format_console`, which
#: shortened the LINE; this shortens the number of lines you have to scroll.
_TOP_ROW_READING_PX = 520

_SAVE_OPERATOR = "mip"

# Roadmap cards shown under "TO BE ADDED", as (label, blurb). Empty: everything currently on the
# roadmap that we're willing to advertise has shipped as a real Operation above. Add an entry when
# a next operator (e.g. the Nautilus agent) is close enough to promise.
_TO_BE_ADDED: list = []


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


def _action_label(key: str, operator_kwargs: Optional[dict] = None) -> str:
    """What the console calls one action: ``decon(sigma=2.0)``, or bare ``mip`` with no parameters.

    The REGISTRY key, not the card's human label, and the parameters spelled out. A console line
    has to be enough to reproduce the run from, and "Deconvolution" is not: two runs at different
    sigmas would print identically, which is precisely the mixed-recipe plate Task 3 is about.
    Sorted so the same call always renders the same string, i.e. so it can be compared by eye.
    """
    if not operator_kwargs:
        return str(key)
    args = ", ".join(f"{k}={operator_kwargs[k]}" for k in sorted(operator_kwargs))
    return f"{key}({args})"


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
        # PIN THE QAPPLICATION, whoever created it. A window cannot outlive its application, and
        # the application is routinely held only by a caller's local or a pytest fixture cache.
        # This is the one call every window makes, however it was constructed -- the same argument
        # showEvent makes for the GUI slot cap. See the _APP comment at the top of this module.
        qt_app()
        self.setWindowTitle("SquidXplorer")
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
        # THIS WINDOW'S LOGGER (Task 1). The root plate is VIEW 0: it is the root of the view tree,
        # and ViewerManager hands out 1 upward, so the two numberings cannot collide. Every action
        # it logs carries that id and, where it has one, an address.
        self.view_id = 0
        self.log = ViewLog(log, self.view_id)
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
        self._fusion_style = _fusion_style()   # process-wide: setStyle does NOT take ownership
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

        # THE ONE GLOBAL CONSOLE, AS A FIXED TAB (Task 1, 2026-07-29). It was a separate top-level
        # QMainWindow; Spencer logged that it "opens over the main window" on every launch, and the
        # fix is not to position it but to stop it being a window. Julio: "making the logger global
        # will force you to abstract the data layers cleanly" — a floating window per app is not
        # what makes it global, ONE console printing every window's actions with an address is, and
        # a fixed tab beside the operators is where the user already is. Fixed = never closable and
        # never detachable: a console you can lose is not a console, and _FIXED_TABS below is what
        # the close/detach paths check.
        self._log_panel = LogPanel(self._log_bus, self._activity)
        self._log_panel.start()
        self._left_tabs.addTab(self._log_panel, "Log")
        self._left_tabs.tabBar().setTabButton(1, QTabBar.RightSide, None)

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
        # open independent napari windows (the Views dock, added below), and the log is the fixed
        # "Log" tab built with _left_tabs above — Julio: "the logger on the bottom of the GUI".
        # This replaces the locked 3-pane grid that Spencer asked us to dismantle.

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
        top_row.setMaximumHeight(_TOP_ROW_COMPACT_PX)
        top_row.setMinimumHeight(150)
        self._top_row = top_row     # _on_tab_changed grows it while the console is being read

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

        self._sync_explore_pane()                  # keeps the (now-hidden) op-tab stack coherent

        # 596 x 850 stays the DEFAULT portrait shape (Julio): the plate dominates below the
        # capped top strip, and the window opens identically on every monitor. It is no longer
        # a setFixedSize, for two reasons.
        #
        # 1. Spencer: the root has to be resizable, and the type has to come up with it.
        # 2. A hard 850 stopped fitting the moment enable_hidpi() landed. Those are LOGICAL
        #    pixels, so on a 200%-scaled display 850 is 1700 physical -- taller than a 1080p
        #    screen. A fixed size that cannot fit on the monitor is not a compact shape, it is
        #    a window with its lower half off the bottom of the display.
        #
        # So: open at the design size, clamped to what the screen can actually show, and let
        # the user take it from there. The minimum keeps the top strip's controls from
        # collapsing into each other.
        self.setMinimumSize(420, 520)
        self.resize(*self._default_root_size())

        # The console is a tab now, so the View menu RAISES it rather than toggling a window. Not
        # checkable: there is no state to toggle, and a menu item that can hide the one global
        # console would put the app back where Spencer found it.
        view_menu = self.menuBar().addMenu("&View")
        self._log_act = QAction("&Log", self)
        self._log_act.triggered.connect(self.show_log)
        view_menu.addAction(self._log_act)

        self.setAcceptDrops(True)
        if initial_path:
            self.ingest(initial_path)

    # -- root window sizing + type that follows it -------------------------------------------------

    #: The portrait shape the layout was designed against. Also the denominator for the UI scale:
    #: at exactly this width the type is the size the stylesheets literally say.
    _DESIGN_W, _DESIGN_H = 596, 850

    def _default_root_size(self) -> tuple:
        """The design size, shrunk to fit the screen it will actually open on.

        In LOGICAL pixels, so this compares like with like under ``enable_hidpi()``. Leaves a
        margin for the taskbar and title bar rather than filling the work area exactly.
        """
        w, h = self._DESIGN_W, self._DESIGN_H
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            w = min(w, max(self.minimumWidth(), avail.width() - 40))
            h = min(h, max(self.minimumHeight(), avail.height() - 80))
        return w, h

    def _ui_scale(self) -> float:
        """How much bigger the window is than the shape the type was written for.

        Width-driven: the portrait root grows mostly sideways, and tying type to height would
        make a tall thin window shout. Clamped so a very wide window does not produce absurd
        type and a minimum-size one stays legible.
        """
        return max(0.85, min(2.0, self.width() / float(self._DESIGN_W)))

    def _rescale_fonts(self) -> None:
        """Re-apply every descendant stylesheet with its ``font-size`` multiplied by the scale.

        Done by rewriting the stylesheets rather than by setting a widget font, because a QSS
        ``font-size`` on a child beats any font inherited from a parent -- and this GUI sets one
        inline at 79 call sites. Each widget's ORIGINAL stylesheet is cached the first time it
        is seen, so scaling is always computed from the authored value and never compounds.
        """
        scale = self._ui_scale()
        if abs(scale - getattr(self, "_applied_ui_scale", 0.0)) < 0.02:
            return                              # sub-pixel churn; not worth restyling the tree
        self._applied_ui_scale = scale

        cache = self.__dict__.setdefault("_qss_original", {})
        for w in [self] + self.findChildren(QWidget):
            key = id(w)
            if key not in cache:
                qss = w.styleSheet()
                if "font-size:" not in qss:
                    cache[key] = None           # nothing to scale here; remember and skip it
                    continue
                cache[key] = qss
            base = cache[key]
            if base is None:
                continue
            w.setStyleSheet(_scale_qss_fonts(base, scale))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale_fonts()

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

    #: How many tabs at the head of the process console are FIXED: [0] Operators, [1] Log. They
    #: cannot close and cannot detach, so their indices never move and a plain `index < _FIXED_TABS`
    #: is a sound test. The log is fixed because it is THE one global console (Task 1): a console
    #: the user can close is a console that is missing when the thing worth reading happens.
    _FIXED_TABS = 2

    def show_log(self) -> None:
        """Bring the one global console to the front. The View menu's action, and the call any
        code should make instead of reaching for a window that no longer exists."""
        panel = getattr(self, "_log_panel", None)
        if panel is None:
            return
        if panel.collapsed:
            panel.set_collapsed(False)
        self._left_tabs.setCurrentWidget(panel)

    def _close_op_tab(self, index: int, tabs=None):
        tabs = self._left_tabs if tabs is None else tabs
        if index < self._FIXED_TABS and tabs is self._left_tabs:   # Operators + Log: never closable
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
        if index < self._FIXED_TABS and tabs is self._left_tabs:
            return None                      # Operators + Log are fixed: neither detaches
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

    def _sync_top_row_height(self) -> None:
        """Give the top strip room while the Log tab is in front, and take it back afterwards.

        The strip is capped at ``_TOP_ROW_COMPACT_PX`` because the plate is the star. But the one
        global console lives in that strip now, and 240 px is roughly ten lines, which is a status
        light rather than a log. Selecting the Log tab is the user saying they are reading it, so it
        earns ``_TOP_ROW_READING_PX`` for as long as that is true.

        Deliberately not a remembered setting and not a drag handle. It follows the tab, so there is
        no state to get stuck in a shape the user did not ask for, which is the failure mode the
        placement-mode indicator elsewhere guards against for the same reason.
        """
        row = getattr(self, "_top_row", None)
        tabs = getattr(self, "_left_tabs", None)
        panel = getattr(self, "_log_panel", None)
        if row is None or tabs is None or panel is None:
            return
        reading_log = tabs.currentWidget() is panel
        row.setMaximumHeight(_TOP_ROW_READING_PX if reading_log else _TOP_ROW_COMPACT_PX)

    def _on_tab_changed(self, index: int = -1, force: bool = False):
        """The plate + detail follow the ACTIVE tab (IMA-205).

        An exploration tab claims to be scoped to its subset, so the plate's status dots and the
        detail's FOV slider have to agree with it — otherwise the tab says '4 wells' while the
        viewer beside it lists all 1536, and scrubbing lands on wells the tab never selected.

        A LIVE run is the one thing we won't retarget under: the worker is pushing into the slider
        this call would rebuild. So the switch is DEFERRED, not dropped (``_request_resync``) —
        dropping it is what left the front tab lying about what the viewer shows (BUG 2), because
        nothing re-emits ``currentChanged`` when the run later drains.

        It also sizes the top strip: see ``_sync_top_row_height``. Reading the console and working
        the plate want different amounts of room, and the tab you selected says which you are doing.

        ``force=True`` re-runs the sync from ``_on_run_drained`` even when there is no outgoing
        exploration tab to park — after a mid-run tab close there ISN'T one, and that is precisely
        the case that has to fall back to the whole plate (BUG 1).

        Honest limitation: ndviewer's only retarget seam is ``start_acquisition``, which RESETS the
        viewer. Computed frames pushed via register_array are in-memory and do not survive the
        switch; we re-register the subset's RAW plane paths (cheap — paths only) so the pane shows
        real imagery rather than black. Re-run the operator in the tab to recompute its frames."""
        self._sync_top_row_height()
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
        # ...and close the console's started/done pair. This fires on ok, failed and STOPPED alike,
        # which is why the pair is closed here and not on finished_ok: an action that starts and
        # then says nothing is indistinguishable from one still running, and a stopped run is
        # exactly the case that would have gone quiet. A run that landed nothing is reported as a
        # failure however politely the engine returned.
        action = getattr(self, "_run_action", None)
        if action is not None:
            self._run_action = None
            elapsed = time.monotonic() - getattr(self, "_run_began", time.monotonic())
            landed = getattr(self._worker, "landed", None)
            if landed == 0:
                self.log.failed(action, f"produced nothing after {elapsed:.1f} s",
                                address=self._run_address)
            else:
                self.log.done(action, elapsed, address=self._run_address)
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
        # DEEP ZOOM: arm the tile overlay for this acquisition. Fail-quiet by contract — an
        # acquisition with no usable stage positions keeps the montage and nothing else changes.
        if self._overview.set_tile_source(reader, meta):
            g = self._overview._ladder.geometry
            log.info("deep zoom armed: %d rungs, %.3f-%.1f um/px, %d tiles at fit",
                     len(g), g.levels[0].scale_um_per_px, g.levels[-1].scale_um_per_px,
                     g.worst_case_tiles)
        else:
            log.info("deep zoom not armed (no usable stage positions) — montage only")
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
            # No region slider on the root any more (2b8fbc5 moved it into each spawned window),
            # so naming it here sent users looking for a control that is not on screen.
            f"{len(self._fov_index)} wells loaded · double-click a well, or Shift-drag to open "
            f"several{note}")

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
            # A BOUND METHOD, never a lambda closing over ``self``. PyQt keeps a lambda alive in a
            # slot proxy parented to the timer, and the timer is parented to this window: the
            # closure's ``self`` closes the loop window -> timer -> proxy -> lambda -> window. That
            # cycle means dropping the last reference to a window does NOT destroy it; the cyclic
            # collector does, later, from arbitrary code, with a debounce still pending. A bound
            # method of a QObject is connected by reference to the receiver instead, so this cycle
            # does not form. Measured on its own it was enough to stop the segfault (40 windows
            # survived, against a crash by window 21 without it); it is kept alongside the
            # ``stop()`` in closeEvent because the two remove different halves of the hazard, the
            # zombie window and the armed callback. See tests/test_window_lifetime.py.
            self._region_load_timer.timeout.connect(self._fire_region_load)
        self._pending_region = region
        self._region_load_timer.start(140)

    def _fire_region_load(self):
        """The debounce elapsed: fuse the region the slider actually settled on."""
        self._load_mosaic(region=self._pending_region)

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
                          "path": field_path(zroot, w0, fov0, levels[0])}}).result()
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
        # FLAT-FIELD needs an illumination profile. Without one, _correct_with_active raises per
        # field and the plate fills with red x's (Julio: "flatfield shows as x's"). If none is
        # active, AUTO-ESTIMATE one from a spread sample of plate tiles (tilefusion BaSiC, off-thread)
        # and re-run once it lands. The estimate uses the first channel; the flat-field tab lets the
        # user pick a different channel and re-estimate.
        if key == "flatfield":
            import squidmip._flatfield as _ff
            if _ff.active_profile() is None:
                if getattr(self, "_ff_est_worker", None) is not None and self._ff_est_worker.isRunning():
                    self._readout.setText("flat-field: estimating an illumination profile…")
                    return
                chan = self._meta["channels"][0]["name"]
                w = _FlatfieldWorker(self._reader, self._meta, chan, parent=self)
                w.stage.connect(self._readout.setText)
                w.problem.connect(lambda m: self._readout.setText(f"flat-field estimate failed: {m}"))

                def _profile_ready(profile, k=key, regs=regions, sv=save, op=out_parent, tk=tab_key):
                    _ff.set_profile(profile)
                    self._readout.setText("flat-field: profile ready — running.")
                    self.run_operator(k, out_parent=op, regions=regs, save=sv, tab_key=tk)

                w.done.connect(_profile_ready)
                self._ff_est_worker = w
                self._readout.setText(f"flat-field: estimating an illumination profile from {chan} "
                                      "(tilefusion BaSiC)…")
                w.start()
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
        # NOTE: _raw_btn is a hidden ORPHAN (never added to a layout since the central pane was
        # removed), so .show() made it POP UP AS A FLOATING WINDOW — Julio: "a 'return to raw view'
        # window pops up. That I don't get." Return-to-raw is handled by the layer stack / plate mode
        # now, so we no longer surface this stray button.
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
        # ...and to the ONE GLOBAL CONSOLE, as a started/done pair carrying this view's id:
        #     [0] A1  decon(sigma=2.0) · 1 well  started
        #     [0] A1  decon(sigma=2.0) · 1 well  done in 1.4 s
        # A run over exactly ONE region has an address. A run over many is a SET of extents, which
        # a single Extent cannot say (region_id is one region, deliberately — see _address.py), so
        # rather than invent a sentinel region_id the plural case names its count and carries the
        # view id alone. Task 2, where a cached result carries its OWN extent, is where the set
        # belongs: the run's answer is one extent per cell, not one extent for the run.
        self._run_action = f"{_action_label(key, operator_kwargs)} · {scope}"
        self._run_address = (Extent(region_id=regions[0])
                             if regions is not None and len(regions) == 1 else None)
        self._run_began = time.monotonic()
        self.log.started(self._run_action, address=self._run_address)
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
        """Wash the plate for the SELECTED views in the navigator, each in its own hue (Linux
        multi-select). The active (first) view reads brighter. Empty selection -> no wash (Julio:
        "the washes only show when I click the view")."""
        if self._overview is None:
            return
        mgr = self._viewer_manager
        focused = mgr.focused_id
        entries = []
        for wid in mgr.selected_ids:
            v = mgr.view_for(wid)
            if v is not None:
                entries.append((v.regions, _view_hue(v.window_id, focused=(v.window_id == focused))))
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
        timer = getattr(self, "_region_load_timer", None)
        if timer is not None:
            # A PENDING SINGLE-SHOT MUST NOT OUTLIVE THE CLOSE. The region-slider debounce is armed
            # for 140 ms and nothing disarmed it, so a window closed within that window kept a live
            # timer whose timeout called back into a torn-down window -- measured directly: with
            # windows built, opened, closed and dropped in a loop, ``_load_mosaic`` was observed
            # running on an already-closed window, and the process segfaulted a window later.
            # ``PlateOverview.hideEvent`` already stops its own coalescing timer for exactly this
            # reason; this one was simply missed.
            timer.stop()
        self._stop_worker()          # stop the run cleanly; nothing on disk to clean up (no cache)
        self._stop_preview()
        self._stop_mosaic_worker()   # JOINED, not drained: it is parented to this window
        self._join_retired()         # ...and so is everything _retire deferred
        self._stop_spots()           # never leave the segmentation thread running at teardown
        self._stop_minerva()         # files already written stay; only the launch poll is abandoned
        ov = getattr(self, "_overview", None)
        if ov is not None:
            ov.clear_tile_source()   # joins the tile fetcher; a live QThread blocks a clean exit
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
    try:
        import fcntl
    except ModuleNotFoundError:
        # Windows has no fcntl: skip the single-instance lock and launch anyway. The cap is a
        # nicety (it stops a second window on Unix), not core behaviour, and crashing the whole
        # GUI over a missing lock primitive would be a worse bug. release_gui_slot tolerates fd=-1.
        return _GuiSlot(-1, None)

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
        f"a SquidXplorer window is already open ({limit} allowed at once). Close it first, or "
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


def enable_hidpi() -> None:
    """Draw at the display's scale factor. MUST run before the QApplication exists.

    Qt5 does not scale unless asked. On a 200%-scaled display (Spencer's workstation:
    system DPI 192) the app is DPI-*aware* but not DPI-*scaling*, so the window occupies its
    logical pixel count in PHYSICAL pixels -- it renders at half the size of every other
    application on the screen, which is the "everything is too small" report. Every ``px`` in
    the stylesheets is a logical pixel once this is on, so fonts, icons and paddings all come
    up together instead of needing 79 call sites edited.

    Setting these after a QApplication has been constructed is a silent no-op, which is why
    this is a named function called at the one point that owns startup rather than a line
    buried in ``main``.
    """
    for attr in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        flag = getattr(Qt, attr, None)
        if flag is not None:                # Qt6 always scales and drops both attributes
            QApplication.setAttribute(flag, True)


def main(dataset_path: str = None):
    path = dataset_path or (sys.argv[1] if len(sys.argv) > 1 else None)
    slot = None
    if _gui_cap_applies():
        try:
            slot = acquire_gui_slot()
        except GuiAlreadyOpen as e:
            print(f"squidmip-view: {e}", file=sys.stderr)
            sys.exit(1)
    if QApplication.instance() is None:     # only the process that CREATES the app may set these
        enable_hidpi()
    app = qt_app(sys.argv)       # pinned process-wide: main() returns the WINDOW, not the app
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
