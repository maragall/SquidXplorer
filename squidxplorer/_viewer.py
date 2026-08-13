"""HCS viewer — a post-acquisition, well-plate viewer for Squid acquisitions.

One Qt main window: the plate overview on top, operator console and log below.
Viewing happens in independent RegionViewer windows spawned from the plate.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy.QtCore import (  # QThread/Signal: kept for tests that build a stub worker as V.QThread
    Qt, QThread, QTimer, Signal,  # noqa: F401 (see tests/test_viewer.py::_BlockingWorker)
)
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QStyleFactory, QTabBar, QVBoxLayout, QWidget,
)

#: The one Fusion QStyle for this process. setStyle() does not take ownership, so a
#: per-window style is freed before ~QWidget runs and corrupts the heap.
_FUSION_STYLE = None
_FUSION_STYLE_MADE = False

#: The one QApplication, pinned for the life of the process: PyQt deletes it when the
#: last Python reference dies, whatever widgets are still standing.
_APP = None


def qt_app(argv=None):
    """The process's QApplication, created if needed, and held so Python cannot free it early."""
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


# The log panel taps the stdlib root logger, so anything logged here appears in it.
from squidxplorer._fontscale import rescale_fonts, scale_qss_fonts, window_screen
from squidxplorer._logpane import get_logger

log = get_logger("viewer")

from squidxplorer import _ingest, _measure, _qtstyle, _run_scope
from squidxplorer.contract import field_path
from squidxplorer._engine import available_plane_operators
from squidxplorer._minerva import MINERVA_HOME_ENV as _MINERVA_HOME_ENV
from squidxplorer._minerva import NEEDS_INTERNET_NOTE as _MINERVA_INTERNET_NOTE
from squidxplorer._minerva import MINERVA_URL as _MINERVA_URL
from squidxplorer._gallery import GalleryScope as _GalleryScope
from squidxplorer._montage import _hex_to_rgb01
from squidxplorer._output import incomplete_reason, parse_well_id
from squidxplorer._activity import ActivityLog
from squidxplorer._address import Extent
from squidxplorer._logpane import LogBus, ViewLog
from squidxplorer._logpanel import LogPanel
from squidxplorer._plate import PlateBuildError, build_plate
from squidxplorer._plate_shape import PlateShapeError
from squidxplorer._qt_tabs import _DetachTabBar, _DetachTabs, _FloatWindow  # noqa: F401 (re-export)
from squidxplorer._qtstyle import dark_palette as _dark_palette
from squidxplorer._qtstyle import hline as _hline
from squidxplorer._qtstyle import operator_card as _operator_card
from squidxplorer._time_point import TimePointBar
from squidxplorer._region_nav import RegionCursor
from squidxplorer._run import OperatorRun

# Plate overview and geometry live in `_plate_overview`; re-exported under their
# historical names so callers and tests reaching through `_viewer` are unchanged.
from squidxplorer._plate_overview import (  # noqa: F401 (re-exports)
    _CELL, _CLICK_SLOP, _COLH, _HDR, _LOUPE_CACHE, _LOUPE_HOLD_MS, _LOUPE_MAG, _LOUPE_MAX_CROP,
    _LOUPE_PX, _LOUPE_SLOP, _LOUPE_WIN_LOCK, _PAD, _PCT, _PUSH_PX, _TILE_CACHE_BYTES,
    _TILE_QUEUE_MAX, _VIEW_WASH, _FRAME_MIN_GRID, _SEL_FRAME,
    PlateOverview, _LoupeSource, _LoupeWorker, _RawLoupeSource, _RunningContrast, _TileFetcher,
    _ZarrLoupeSource, _box_union, _deep_zoom_enabled, _fit_box, _fit_cell,
    _fmt_um, _fov_of_well, _mosaic_boxes, _nice_scale_um, _pct_window,
    cells_in_rect, content_box, frames_for_grid, loupe_clamp_crop, loupe_crop_px, loupe_decimation,
    loupe_level, loupe_scale, loupe_um_per_screen_px,
    resolve_plate_root, selection_frame_pen_px, well_at,
)
from squidxplorer._plate import _row_letter  # noqa: F401 (re-export)

# QThread workers live in `_workers`; re-exported so monkeypatched spies keep working.
from squidxplorer._workers import (  # noqa: F401 (re-exports)
    _CACHE_AUTO, _MIN_PREVIEW_BOX_PX, _VIEWER_WORKERS,
    _ComputedPlateWorker, _FlatfieldWorker, _FocusWorker, _MinervaRenderWorker, _MinervaWorker,
    _MosaicWorker, _OperatorWorker, _PreviewWorker, _SpotWorker, _VideoWorker, _full_res_mip,
    _full_res_plane, _spot_stages,
)

# The operator registry lives in `_operations`; re-exported for this module's call sites.
from squidxplorer._operations import (  # noqa: F401 (re-exports)
    _OPERATIONS, _OPERATIONS_BY_KEY, _SAVE_OPERATOR, _TO_BE_ADDED, Operation, OperationStack,
    _action_label, operator_label, operator_layer_key, operator_name, result_kind,
    runnable_operators,
)

# Chrome (colours, stylesheets, palette) is defined once in `_qtstyle` and aliased here.
_BG = _qtstyle.BG
_GRID, _RED, _MUTED, _ACCENT = _qtstyle.GRID, _qtstyle.RED, _qtstyle.MUTED, _qtstyle.ACCENT
_SEL_FILL = _qtstyle.SEL_FILL


def _view_hue(view_id: int, *, focused: bool = False) -> QColor:
    """A stable, distinct hue per open view, so the plate colour-codes wells per window."""
    h = (0.13 + 0.61803398875 * int(view_id)) % 1.0     # golden-ratio walk spreads hues
    c = QColor.fromHsvF(h, 0.62, 1.0, 0.34 if focused else 0.20)
    return c


_CONTROL_BLUE = _qtstyle.CONTROL_BLUE


_STATUS = _qtstyle.STATUS   # processing-status hue coding
_TABS_DARK = _qtstyle.TABS_DARK
_CARD_QSS = _qtstyle.CARD_QSS
_BTN_QSS = _qtstyle.BTN_QSS
_COMBO_QSS = _qtstyle.COMBO_QSS
_CHECK_QSS = _qtstyle.CHECK_QSS
_MENU_QSS = _qtstyle.MENU_QSS
_ANSI_RE = _qtstyle.ANSI_RE

#: ``font-size: 12px`` in a stylesheet; only ``px`` — ``pt`` must not be scaled twice.
_QSS_FONT_PX_RE = re.compile(r"(?<=font-size:)\s*(\d+(?:\.\d+)?)\s*px", re.IGNORECASE)


def _scale_qss_fonts(qss: str, scale: float) -> str:
    """Alias for `_fontscale.scale_qss_fonts`, kept under this module's historical name."""
    return scale_qss_fonts(qss, scale)

    return _QSS_FONT_PX_RE.sub(_sub, qss)


def _signal_names(cls) -> tuple:
    """Every Signal declared on *cls* or its bases, excluding QThread's own finished/started."""
    from qtpy.QtCore import Signal as _sig
    seen, out = set(), []
    for klass in cls.__mro__:
        for name, value in vars(klass).items():
            if name in seen or name in ("finished", "started"):
                continue
            if isinstance(value, _sig) or type(value).__name__ in ("Signal", "unbound_signal"):
                seen.add(name)
                out.append(name)
    return tuple(out)


#: Band under the plate: default height at the design size, and the hard ceiling the
#: operator cards' size hint cannot argue with. The plate keeps stretch factor 1.
_BAND_DEFAULT_PX = 365
_BAND_MAX_PX = 520

#: Operator-over-Log split inside the band's right column (starting position).
_RIGHT_COL_SIZES = [215, 165]


# --- the main window: the plate on top, the Open View list and the console below --------------


class PlateWindow(QMainWindow):
    #: The latest run's identity and books (:class:`squidxplorer._run.OperatorRun`), or None before
    #: the first run. A CLASS default rather than an __init__ assignment so the run slots can use
    #: plain attribute access: a bare ``getattr(self, ..., None)`` on a QObject whose __init__ has
    #: not run raises out of Qt's own attribute machinery instead of returning the default.
    _run = None

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
        self._minerva_render = None   # the render.py exhibit render, the no-file-picking path
        self._retired = []            # workers asked to stop; kept alive until they actually finish
        self._overview = None
        self._reader = None
        self._meta = None
        # THE SINGLE OWNER of "which region is current". The red ROI frame on the plate and every
        # spawned window's mosaic are VIEWS of this one value — none of them keeps a copy. Before
        # it there were three copies, hand-synced: PlateOverview._sel,
        # _mosaic_region and _current_well. Both `_mosaic_region` and `_current_well` are now
        # PROPERTIES that read the cursor, so an assignment cannot create a fourth.
        self._cursor = RegionCursor()
        self._cursor.subscribe(self._on_region_changed)
        self._cursor.on_problem(lambda msg: self._readout.setText(msg))
        # THE communication backbone, built once and owned here.
        # * _log_bus attaches to the stdlib ROOT logger, so every orchestrated library (tilefusion,
        #   petakit, bgsub, and the per-run measurement line) appears in the panel with no wiring.
        # * _activity is the single registry of in-flight work the panel's header reads.
        # * commands is the ONE command surface (squidxplorer._command) — the GUI is now a CALLER of the
        #   same layer the CLI drives, so an agent/test/script says one command to both.
        self._log_bus = LogBus()
        self._log_bus.install()
        # THIS WINDOW'S LOGGER (Task 1). The root plate is VIEW 0: it is the root of the view tree,
        # and ViewerManager hands out 1 upward, so the two numberings cannot collide. Every action
        # it logs carries that id and, where it has one, an address.
        self.view_id = 0
        self.log = ViewLog(log, self.view_id)
        self._activity = ActivityLog()
        from squidxplorer._gui_commands import install_command_bus
        self.commands = install_command_bus(self)
        self._fov_index = {}
        self._selected_regions = []   # wells picked on the plate (IMA-221); scopes an operator run

        # DECENTRALIZED VIEWER (Spencer, 2026-07-23 call). The plate is the ROOT; a selection opens
        # an INDEPENDENT napari window that floats on the desktop, tracked by ID in the Open View
        # list. Many wells become ONE window with a region slider, not many windows. Every window
        # shares this one stateless reader/meta — nothing reopens the dataset. See _region_viewer.
        from squidxplorer._region_viewer import ViewerManager
        self._viewer_manager = ViewerManager(parent=self)
        # Operator controls appear AT EACH LEVEL (the deck; Julio 2026-07-23: "I don't see operator
        # controls like the powerpoint specified at each level"). Every window's "Operators for this
        # window" dropdown is the SAME registry + run_operator (the CLI engine), scoped to that view,
        # so "select where to run stitching" = pick the view, Run. Only runnable operators appear
        # (minerva is a terminal that stays on the root's stack; Gallery View is a View-menu
        # window-management command and was never an operator at all).
        self._viewer_manager.operator_specs = [
            (op.key, op.label) for op in _OPERATIONS if op.runnable]
        self._viewer_manager.run_operator = self.run_operator
        # The plate wash shows ONLY the view you CLICK (Julio), coloured by that view's own hue so
        # different view threads are told apart. Not all views at once — that clutters the plate.
        # viewFocused fires on open/raise; windowsChanged clears it when the focused view closes.
        self._viewer_manager.viewFocused.connect(lambda _regions: self._refresh_view_hues())
        self._viewer_manager.windowsChanged.connect(self._refresh_view_hues)
        # THE PLATE FOLLOWS THE WINDOWS' napari (Task 8.1). Julio: "there shouldn't be any controls
        # for the plate view. It just reacts to toggles and contrast adjustments in napari." With no
        # central pane left, the napari the plate must react to is the one inside each window, so
        # the binding happens the moment a window is spawned. Windows whose ids are already bound
        # are skipped, see _bind_window_contrast.
        self._followed_windows: set = set()
        self._viewer_manager.windowOpened.connect(self._bind_window_contrast)

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
        # EVERY OTHER RUNNABLE OPERATOR, off the ENGINE registry. A card is presentation and the
        # engine is capability (`_operations.runnable_operators`), and the gap between the two was
        # a capability the GUI could not reach at all: `spot` and `cellpose` declare four
        # parameters each and appeared in no menu, no card and no dropdown, so not one of those
        # parameters was settable anywhere. These open a panel built from the declaration
        # (`_param_panel.GenericOperatorPanel`), so this submenu needs no edit when an operator is
        # added — including one discovered from another package through `squidxplorer._plugins`.
        self._declared_menu = proc_menu.addMenu("&From their declaration")
        self._declared_menu.setToolTip(
            "Operators with no hand-written panel. Their controls are built from the params= they "
            "declare.")
        self._declared_menu.setEnabled(False)
        for key in runnable_operators():
            if key in _OPERATIONS_BY_KEY:
                continue
            act = QAction(operator_label(key), self)
            act.triggered.connect(lambda _=False, k=key: self._activate_operator(k))
            self._declared_menu.addAction(act)

        self._acq_name = ""           # acquisition folder name, shown as the Process-pane title
        self._current_well = None     # a PROPERTY over self._cursor — see below. Kept as an
        #                               assignment so every existing call site still reads.
        self._current_fov = 0         # the FOV of that region on screen (IMA-250: autofocus ranks IT)
        self._acq_path = None         # the opened acquisition dir (persist writes next to it)
        self._processed_plate = None  # path of the written plate.ome.zarr once an operator persists it
        self._plate_mode = "raw"      # what the plate view is showing — shown in the plate-pane title
        self._plate_format = None     # the format the plate is laid out with (declared or inferred)
        self._plate_format_override = None   # manual override; also read from SQUIDXPLORER_WELLPLATE_FORMAT
        self._op_stack = OperationStack()   # the toggleable layer stack (base + applied operators)
        self._active_op_key = None    # operator whose tiles are streaming into its layer right now
        self._layers_tab = None       # the Layers tab widget, once opened
        self._order = []              # well order = the detail's FOV-slider order
        self._op_tabs = {}            # key -> operator-UI widget currently open as a tab in _left_tabs
        self._floating = {}           # key -> _FloatWindow holding that operator's UI detached
                                      # (a key lives in exactly ONE of the two dicts, never both)
        self._gallery = None          # the ONE open GalleryWindow, or None (see _open_gallery_view)
        self._readout_base = ""
        self._tabs_muted = False      # suppress _on_tab_changed during bulk teardown (ingest)
        self._pending_resync = False  # a tab switch was deferred because a run was live (IMA-205 bugs)
        self._runs_settled = 0        # monotonic: bumped once a run's TERMINAL cascade has run — the
        #                               tiles, the streamEnded recomposite AND _on_run_drained. It is
        #                               the honest "done" signal a test must wait on: QThread.finished
        #                               (hence _busy()==False) fires BEFORE Qt dispatches the queued
        #                               tileReady/streamEnded/finished slots to the main thread, so a
        #                               test that waited on `not _busy()` was reading state its own
        #                               event loop had not yet applied (IMA-258 flakes).
        self._loupe_sources = {}      # layer key -> _LoupeSource backing that layer's pixels (IMA-208)

        # WHAT IS ON SCREEN, and it is no longer a grid of panes. The plate is on top, full width;
        # under it a BAND holding [Window Navigator | Operators over the Log]. Viewing happens in
        # INDEPENDENT windows spawned from the plate (`_region_viewer`), never in a pane locked
        # inside this one. Tabs live inside the band's right column (their bar sits at the top of
        # it) and any tab but the Operators home tab can be DRAGGED OUT into a free-floating window
        # (ImageJ-style; see `_detach_tab`).
        #
        # There WAS a third pane, "exploration" (IMA-205/221/237/260): one tab per Shift-dragged FOV
        # subset, each embedding a second napari mosaic. 2b8fbc5 ("Decentralize GUI") replaced the
        # gesture that filled it — a Shift-drag opens an independent window now — and the pane was
        # removed on 2026-08-05 along with the tab, its slider, its Minerva button and its scope.
        # The one thing that was still routed into it, the deconvolution QC composite, goes to
        # `_left_tabs` (see `publish_qc_result`), which is on screen.

        # the band's right column: the process console (build the home tab first — it owns
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

        # THE ONE GLOBAL CONSOLE, STACKED UNDER THE OPERATORS RATHER THAN BEHIND THEM.
        #
        # 2026-07-29 Task 1 made it a FIXED TAB of `_left_tabs`. It had been a separate top-level
        # QMainWindow; Spencer logged that it "opens over the main window" on every launch, and the
        # fix chosen was not to position it but to stop it being a window. That decision was about
        # WINDOW vs NOT-WINDOW. Tab vs stacked panel was never argued, and the tab bar is the only
        # reason Operators and Log alternate instead of both being on screen.
        #
        # 2026-08-03, Julio, with a drawing: "I think that we should modify the layout of our main
        # window" — Operator above, Log below, both visible, and the Log gains an option to open in
        # a new window. So the panel comes OUT of the tab bar and goes into `_right_col`, the
        # vertical splitter built with the layout further down. It was written for this: its own
        # docstring calls it "the bottom-right log panel", and its collapse toggle and
        # setMinimumWidth(0) are the machinery of a panel stacked under something else.
        #
        # THE INVARIANT THAT REPLACES `_FIXED_TABS = 2`, and it is what the tests pin: the panel
        # exists for the life of the window and is reachable from View > Log in EVERY state —
        # docked, collapsed or floated. What changes is where it is, never whether it is. Floating
        # RELOCATES the console; nothing destroys it. That is why `_float_log`'s close handler
        # re-docks instead of routing through `_dispose_tab_widget` the way an operator float does.
        self._log_panel = LogPanel(self._log_bus, self._activity)
        self._log_panel.start()
        self._log_panel.float_requested.connect(self._float_log)

        # NO CENTRAL VIEWER (decentralized, 2026-07-23; the guards finally cut, 2026-08-06).
        # Viewing happens in INDEPENDENT windows spawned from the plate (see _region_viewer), each
        # its own napari viewer. The root is just the plate + the Open View list + the log.
        #
        # ``self._mosaic_pane = None`` used to sit here, with a note saying the sentinel stayed so
        # that "dozens of methods guard on it" could no-op "rather than needing every call site cut
        # in one pass". That pass has now happened: the sentinel and every method gated on it are
        # gone. Nothing is left to guard, so there is nothing to define. Do not reintroduce a
        # window-owned pane — a second surface showing a region is what the decentralization
        # removed, and a permanently-None one is worse than none, because it reads as a feature.
        #
        # Every finished run also goes to a file, because METRICS is a bounded in-memory deque and
        # a measurement that dies with the process cannot answer "is this slower than last month".
        # Idempotent, so eight windows in one process still attach one sink; and it writes under
        # the per-user cache root, which the test suite already redirects.
        _measure.persist_runs()
        self._right_widget = None

        # THE PLATE, on top and full width (drop target until an acquisition opens). Its FIXED
        # title bar names the wellplate we're on (the acquisition) — the plate's identity lives
        # with the plate.
        self._plate_title = QLabel("well plate")   # plate name; shows the hovered well on hover
        # SMALL, 2026-08-03. Spencer: "Test metadata label (e.g. '10x laser Z-stack') too large,
        # taking up excess space." This is that label — it reads
        # "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945   ·   raw" and it is the drawing's
        # "Filename ..." strip. It was 17 px / weight 800 / 9 px padding and rendered a 38 px band
        # across the window at the design size, 59 px at 1280 wide (`rescale_fonts` scales it with
        # the window). At 12 px / 600 / 3 px it is a caption on the plate pane instead of a headline
        # over it, which is the whole of the complaint: it is metadata, the plate is the content.
        #
        # NOT DELETED, which the drawing's missing "Filename ..." box could be read as asking for.
        # It is the only thing on screen that says WHICH acquisition is open and what the plate is
        # showing ("· raw" / "· computed MIP" / the operator's name), it is the hover readout for
        # the well under the cursor, and it has four writers (`_on_hover` and three mode switches).
        # Absorbed into the plate pane as a thin caption, not removed.
        self._plate_title.setStyleSheet(           # the BAR below now carries background + border
            "color:#c9d1d9;font-size:12px;font-weight:600;padding:3px 12px;border:none;")
        # NO CONTRAST CONTROL HERE. Julio: "there shouldn't be any controls for the plate
        # view. It just reacts to toggles and contrast adjustments in napari." The scope
        # dropdown that used to sit here is gone with per-region contrast itself.
        plate_title_bar = QWidget()
        plate_title_bar.setStyleSheet("background:#0b0e14;border-bottom:1px solid #232b3a;")
        _tb = QHBoxLayout(plate_title_bar)
        _tb.setContentsMargins(0, 0, 12, 0)
        _tb.setSpacing(8)
        _tb.addWidget(self._plate_title, 1)
        # THE VIEW COMBO: which layer the plate draws. Inline in the title bar, deliberately --
        # Julio, 2026-08-06: *"we have a little 'view' dropdown in the plate master window to
        # select if we want to view the plate raw, stitched, decon etc. But it is tiny, like don't
        # add more rows of buttons."*
        #
        # It exists because of the shelving one message earlier. A copied LUT is a LOOK, not
        # pixels, so the plate stays on raw until something is bulk-processed over it -- and once
        # something has been, the user needs a way to say so. That used to be answered implicitly,
        # by whichever window last toggled a layer; with the live follow gone, it needs a control,
        # and this is the smallest honest one. Populated from the plate's own `_op_stack` (see
        # `_refresh_view_combo`), which is the same stack the Layers tab draws, so it can only
        # offer layers this plate really holds pixels for.
        self._view_combo = QComboBox()
        self._view_combo.setStyleSheet(
            "QComboBox{background:#0b0e14;color:#8b949e;border:1px solid #232b3a;"
            "border-radius:4px;padding:1px 6px;font-size:11px;}"
            "QComboBox::drop-down{border:none;width:14px;}")
        self._view_combo.setToolTip(
            "Which layer the plate thumbnails draw. Only layers this plate has actually "
            "processed appear here — a result computed in one window is that window's, and does "
            "not give the plate pixels for its other wells.")
        self._view_combo.currentIndexChanged.connect(self._on_view_combo)
        _tb.addWidget(QLabel("view:"))
        _tb.addWidget(self._view_combo)
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
        #
        # NOT a duplicate of the automatic window->plate tap in `_bind_window_contrast`, and the
        # difference is authority, not numbers. That tap lands in `follow_channel_window` ->
        # `_RunningContrast.set_followed`, which `resolve` OUTRANKS with the per-region scope and
        # drops entirely when the user picks SCOPE_PER_REGION. This paste lands in
        # `set_channel_window` -> `set_manual`, the user latch, which wins over everything
        # including per-region scope. The tap is "show me what that window resolved"; this is
        # "pin the plate here". Also the ONLY way OUT of the plate: the tap is one-directional.
        _lut_qss = ("QPushButton{background:#161b22;color:#c9d1d9;border:1px solid #30363d;"
                    "border-radius:4px;padding:3px 10px;font-size:12px;}"
                    "QPushButton:hover{background:#21262d;}")
        self._plate_copy_lut_btn = QPushButton("⧉ LUTs")
        self._plate_copy_lut_btn.setToolTip(
            "PLATE → clipboard: the plate's per-channel contrast. Paste it into a window with "
            "that window's '⤓ Paste LUTs' — the clipboard is shared. This is the only way out "
            "of the plate; the automatic sync only runs window → plate.")
        self._plate_copy_lut_btn.setCursor(Qt.PointingHandCursor)
        self._plate_copy_lut_btn.setStyleSheet(_lut_qss)
        self._plate_copy_lut_btn.clicked.connect(self._plate_copy_luts)
        self._plate_paste_lut_btn = QPushButton("⤓ LUTs")
        self._plate_paste_lut_btn.setToolTip(
            "clipboard → PLATE: apply the copied contrast. This LATCHES each channel manual, so "
            "it survives per-region scope and the wells still streaming in cannot stomp it — "
            "unlike the automatic window → plate sync, which the plate is free to outrank.")
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
        # NO region slider on the root plate, and NO ATTRIBUTE FOR ONE. The deck puts the region
        # slider ("<> A1, B6, C3") in each spawned WINDOW, not on the plate — navigation is per
        # window now. Building napari's QtDims here also loaded napari icons with no napari viewer
        # registered, which is the "theme_dark:/playback-forward.svg not found" warning spam.
        #
        # It was `self._region_slider = None` until 2026-08-06, with `_make_region_slider` (the only
        # thing that could ever produce a non-None one) called from nowhere, a tooltip branch in
        # `_on_region_changed` guarding on it, and a `_region_slider_failure` string written in two
        # places and read in none. That is `_mosaic_pane` again, one release later and one
        # attribute over: a `None` that reads as "the feature is off right now" when the truth is
        # "there is no feature". `tests/test_nav_wiring.py` pins the ABSENCE of the attribute for
        # the same reason `tests/test_plate_follows_windows.py` pins `_mosaic_pane`'s.

        # NO "Focus reference plane" button here. It was a control UNDER the old central viewer,
        # kept on as a "hidden orphan" so its setEnabled callers would still resolve — and a
        # QPushButton built with no parent is a TOP-LEVEL WINDOW, so `_sync_focus_button`'s
        # setVisible(z_levels > 1) un-hid it as a bare 178x30 titleless window beside the root on
        # every multi-z acquisition. That is the stray window Julio kept seeing. The reference plane
        # is per-window now, on each window's own z-slider (d07db43,
        # `RegionViewer._focus_reference_plane`), so the whole chain here was dead but for the
        # orphan's own clicked signal. Deleted rather than re-hidden; pinned by
        # tests/test_no_orphan_windows.py.

        # THE ROOT IS JUST THE PLATE (decentralized, 2026-07-23). The central viewer is gone from
        # the layout (and the exploration pane with it); the plate column IS the window. Selections
        # open independent napari windows (the Views dock, added below), and the log sits UNDER the
        # operators in `_right_col` — Julio: "the logger on the bottom of the GUI". This replaces
        # the locked 3-pane grid that Spencer asked us to dismantle.

        # THE V2 DECK (2026-08-03 drawing, ...-main-window-layout-v2-plate-dominant.png): the PLATE
        # on top, full width, dominant; a BAND under it holding [Window Navigator | Operator over
        # Log]. Spencer: the plate view should take roughly half the window's real estate, and at
        # the bottom of the window it was losing prominence to text-heavy panels. Measured before
        # this change, offscreen: the plate was 32.8% of a 596x850 window and 35.6% of 1280x900.
        from squidxplorer._region_viewer import OpenViewList
        self._open_views = OpenViewList(self._viewer_manager, self)

        # THE STATUS BLOCK LEAVES THE NAVIGATOR AND GOES INTO THE LOG. Julio, verbatim: "the status
        # bar and memory bar should be moved to inside the logger so that we save space." The v2
        # drawing has no "Status bars" box at all. Those bars are internals of `OpenViewList`: the
        # memory caption + bar, and the run-progress caption + bar it grew for "where the memory bar
        # is, there should also be a loading bar" — so the move is a reparent, NOT a rebuild: the
        # navigator keeps driving them off ViewerManager's signals and the log panel just shows
        # them. See `OpenViewList.take_status_row` and `LogPanel.adopt_status_row`.
        #
        # NOT MOVED, and named so the omission is a decision rather than an oversight: the
        # QMainWindow `statusBar()` carrying `self._readout`. It is the guidance line ("2 wells
        # loaded · double-click a well…"), it is asserted by ~25 tests as the window's one reply
        # channel, and neither panel of the drawing shows it. It stays where it is.
        self._log_panel.adopt_status_row(*self._open_views.take_status_row())

        # THE RIGHT COLUMN IS A VERTICAL SPLIT: Operator on top, Log beneath (Julio's 2026-08-03
        # drawing). Both visible at once, which is the whole request; the tab bar that made them
        # alternate now carries only the Operators home tab and the user's detachable operator tabs.
        #
        # A SPLITTER, not a fixed 50/50 layout, for two reasons. The plate is held at about half the
        # window (see _BAND_DEFAULT_PX) and a handle is how the user takes some of that back. And
        # it is the drag affordance `_sync_top_row_height` existed to AVOID needing: that method
        # grew the strip while the Log TAB was in front and shrank it afterwards, inferring intent
        # from a tab selection. There is no tab selection now, and an automatic height swap would
        # fight a user who has just dragged the boundary where they want it. It is deleted, not
        # adapted.
        #
        # setChildrenCollapsible(False) IS THE INVARIANT IN CODE: a splitter will happily let you
        # drag a child to zero, and a console dragged to zero is a console you have lost — exactly
        # what `_FIXED_TABS` was protecting when the log was a tab. The pressure valve is the
        # panel's own collapse toggle (`▸ Log`), which drops it to its header and hands the space
        # to the operators without ever taking the console off screen.
        right_col = QSplitter(Qt.Vertical)
        right_col.setStyleSheet("QSplitter{background:#0b0e14;}"
                                "QSplitter::handle{background:#232b3a;height:1px;}")
        right_col.setHandleWidth(6)
        right_col.setChildrenCollapsible(False)
        right_col.addWidget(self._left_tabs)    # Operator, on top
        right_col.addWidget(self._log_panel)    # Log, beneath
        right_col.setStretchFactor(0, 3)
        right_col.setStretchFactor(1, 2)
        right_col.setSizes(list(_RIGHT_COL_SIZES))
        self._right_col = right_col

        band = QSplitter(Qt.Horizontal)
        band.setStyleSheet("QSplitter{background:#0b0e14;}"
                           "QSplitter::handle{background:#232b3a;width:1px;}")
        band.addWidget(self._open_views)        # band left: the Window Navigator
        band.addWidget(right_col)               # band right: Operators over the one global console
        # 280/280 -> 230/360. The navigator's contents are a tree of short window titles plus two
        # buttons; the operator cards are the widget actually starved of width, and _qtstyle.py
        # records that every blurb elides in the ~300 px it gets. Splitter sizes are hints, so this
        # is a default and not a constraint — and the navigator's two side-by-side buttons set a
        # minimum width that wins over this ratio at the 596 px design size (measured: 313/277).
        band.setSizes([230, 360])
        band.setHandleWidth(6)
        band.setMinimumHeight(150)
        self._band = band

        # THE CAP GOES ON A PLAIN HOST, NOT ON THE SPLITTER, AND THAT IS A BUG FIX.
        #
        # `band.setMaximumHeight(...)` DOES NOT WORK. QSplitterPrivate::recalc calls
        # setMaximumSize() on the splitter itself out of its children's maximums every time a child
        # is added or its geometry changes, so it overwrites any cap set from outside. MEASURED on
        # 83c486c, offscreen, a 596x850 window: the splitter's `maximumHeight()` reads 16777215
        # (QWIDGETSIZE_MAX) and the strip rendered 479 px tall against a 240 px cap. The only thing
        # that ever re-applied it was `_sync_top_row_height` firing on `currentChanged` — and the
        # next recalc dropped it again.
        #
        # A plain QWidget does not rewrite its own maximum, so the cap holds here for real. It is
        # the ceiling, not the size: `_BAND_DEFAULT_PX` is what the band actually opens at, and this
        # is what stops the operator cards' size hint arguing it upwards. Its OWN panels scroll.
        band_host = QWidget()
        _th = QVBoxLayout(band_host)
        _th.setContentsMargins(0, 0, 0, 0)
        _th.setSpacing(0)
        _th.addWidget(band)
        band_host.setMaximumHeight(_BAND_MAX_PX)
        band_host.setMinimumHeight(150)
        self._band_host = band_host

        root = QWidget()
        root.setStyleSheet(f"background:{_BG};")
        rv = QVBoxLayout(root)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(1)
        # The timepoint bar sits UNDER the plate, next to what it navigates, and it is built
        # unconditionally then hidden when there is one timepoint (see _time_point). Every call site
        # therefore stays unconditional: nothing has to ask whether the control exists.
        self._time_point_bar = TimePointBar(on_change=self._on_time_point_changed)
        self._time_point_bar.set_count(1)

        # THE WHOLE BODY IS ONE VERTICAL SPLITTER, TOP TO BOTTOM: the plate, then the band. One
        # splitter rather than a splitter nested in a splitter — same orientation twice would put
        # two indistinguishable horizontal handles on screen and make "which one moves the plate"
        # a guess.
        #
        # It had a THIRD child between them until 2026-08-05: the exploration pane, hidden unless
        # it held a tab. The only thing left routed into it was the decon QC composite, and that
        # now opens as a tab in `_left_tabs` — where the operator that produces it lives, and where
        # a user is already looking when they press Run.
        body = QSplitter(Qt.Vertical)
        body.setStyleSheet("QSplitter{background:#0b0e14;}"
                           "QSplitter::handle{background:#232b3a;height:1px;}")
        body.setHandleWidth(6)
        body.addWidget(plate_host)              # index 0: THE PLATE, on top, full width, dominant
        body.addWidget(band_host)               # index 1: navigator | operator over log
        # A CONSOLE YOU CAN DRAG TO NOTHING IS A CONSOLE YOU HAVE LOST — the same invariant
        # `_right_col` buys, one level up: the band must not be draggable to zero either, or the
        # navigator and the operators go with it.
        body.setChildrenCollapsible(False)
        # THE STRETCH IS WHAT KEEPS THE PLATE DOMINANT AS THE WINDOW GROWS. Qt hands a resize
        # delta out by stretch factor, so with the band at 0 the extra height is the PLATE's and its
        # share only ever goes up from the design ratio. `setSizes` sets the starting position.
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 0)
        body.setSizes([self._DESIGN_H - _BAND_DEFAULT_PX, _BAND_DEFAULT_PX])
        plate_host.setMinimumHeight(160)        # the band cannot squeeze the plate out of existence
        self._body = body

        rv.addWidget(body, 1)                   # plate over band, in the drawing's order
        rv.addWidget(self._time_point_bar, 0)   # hidden unless n_t > 1
        self._split = band
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

        # 596 x 850 stays the DEFAULT portrait shape (Julio): the plate dominates below the
        # band, and the window opens identically on every monitor. It is no longer
        # a setFixedSize, for two reasons.
        #
        # 1. Spencer: the root has to be resizable, and the type has to come up with it.
        # 2. A hard 850 stopped fitting the moment enable_hidpi() landed. Those are LOGICAL
        #    pixels, so on a 200%-scaled display 850 is 1700 physical -- taller than a 1080p
        #    screen. A fixed size that cannot fit on the monitor is not a compact shape, it is
        #    a window with its lower half off the bottom of the display.
        #
        # So: open at the design size, clamped to what the screen can actually show, and let
        # the user take it from there. The minimum keeps the band's controls from
        # collapsing into each other.
        self.setMinimumSize(420, 520)
        self.resize(*self._default_root_size())

        # The View menu RAISES the console rather than toggling it. NEITHER action is checkable:
        # there is no state to toggle, and a menu item that can HIDE the one global console would
        # put the app back where Spencer found it. This menu is the other half of the invariant in
        # `show_log` — the console is reachable from here whether it is docked, collapsed or
        # floated, which is what makes "you cannot lose it" survive the log becoming floatable.
        view_menu = self.menuBar().addMenu("&View")
        self._log_act = QAction("&Log", self)
        self._log_act.triggered.connect(self.show_log)
        view_menu.addAction(self._log_act)
        # Julio's drawing: "Log (option to open in a new window)". The panel's header carries the
        # same gesture as a ⧉ button; this is the discoverable duplicate, and it doubles as the way
        # back if the float is somehow off-screen (it raises rather than building a second). It sits
        # DIRECTLY under "Log" because the two are one pair — where the console is, and where you
        # would rather it were. The separator below keeps them from reading as a list with Gallery
        # View, which is a different kind of thing entirely.
        self._log_float_act = QAction("Log in a &New Window", self)
        self._log_float_act.triggered.connect(self._float_log)
        view_menu.addAction(self._log_float_act)
        view_menu.addSeparator()
        # Gallery View lives HERE, not in the Operators stack: it does not transform pixels and
        # writes nothing, so it is not a runnable operator and has no card. It IS gated on an
        # acquisition in the only way that matters -- with none open it says so and opens nothing
        # (the action stays enabled so the menu does not go silently grey on a window-management
        # command). See _open_gallery_view.
        self._gallery_act = QAction("&Gallery View…", self)
        self._gallery_act.triggered.connect(self._open_gallery_view)
        view_menu.addAction(self._gallery_act)

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

        The screen is THIS WINDOW's, not ``primaryScreen()``: on a laptop + external monitor the
        primary is whichever the OS names first, so the height floor below was being taken from a
        display the window may not be on. At construction there is no window handle yet and the
        answer is still the primary one, which is the old behaviour; from the first re-read on it
        is the display the user actually has the plate on.
        """
        w, h = self._DESIGN_W, self._DESIGN_H
        screen = window_screen(self)
        if screen is not None:
            avail = screen.availableGeometry()
            w = min(w, max(self.minimumWidth(), avail.width() - 40))
            # HEIGHT GROWS, WIDTH DOES NOT. The design height is a FLOOR here, not a ceiling: the
            # plate is the tall thing in this window and every pixel of height goes to it, so on a
            # screen with room to spare we take it. Width stays at the design number because past
            # it the plate does not grow, only the gutters either side of it do -- which is what
            # opening this window maximised looked like, and why that was reverted (2026-07-31).
            h = max(self.minimumHeight(), avail.height() - 80)
        return w, h

    def _ui_scale(self) -> float:
        """How much bigger the window is than the shape the type was written for."""
        from squidxplorer._fontscale import ui_scale
        return ui_scale(self, self._DESIGN_W)

    def _rescale_fonts(self) -> None:
        """Re-apply every descendant stylesheet with its ``font-size`` multiplied by the scale.

        The body moved to `_fontscale.rescale_fonts` so the CHILD windows (`RegionViewer`, the Log
        window) get the same behaviour. They are separate TOP-LEVEL windows, so they were never in
        this walk of `findChildren` and their type stayed put while the root's grew.
        """
        rescale_fonts(self, self._DESIGN_W)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale_fonts()

    # -- the Operators panel (band, upper right): a scrollable list of operator blocks -------------
    def _build_process_pane(self) -> QWidget:
        """The Operators panel: JUST a scrollable list of operator blocks — no header, no footer
        (Julio, 2026-07-23). Each block opens that operator; operators apply to the plate SELECTION
        (Cmd/Ctrl-A picks the whole plate). Minerva is here as the deck's terminal operator; Gallery
        View is NOT (it arranges windows, see the View menu). Status moved to the window status bar;
        the old 'run on' scope combo and the raw/3D/MIP footer buttons are kept as hidden orphans so
        their many callers still resolve — they migrate onto the operator tabs and the windows in
        the operator phase."""
        # Status line — tests and many methods read self._readout; it now lives in the status bar,
        # not as a pane header. Created here because _build_process_pane runs during __init__.
        self._readout = QLabel("Drop a Squid acquisition, then pick an operator.")
        self._readout.setStyleSheet("color:#8b98ad;font-size:12px;")
        self.statusBar().addWidget(self._readout, 1)

        # Hidden orphans (referenced elsewhere; not shown — no header/footer).
        self._scope_run = QComboBox()
        self._scope_run.addItems(list(_run_scope.RUN_SCOPES))
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
        # TERMINAL operator on TOP of the stack (Julio, 2026-07-23: "I need minerva author and the
        # gallery view to be on the top of the stack"): Minerva Author, then the processing
        # operators in registry order, minus minerva (already placed).
        #
        # GALLERY VIEW IS NOT HERE ANY MORE (Julio, 2026-08-02: "I guess I don't understand how
        # this can be treated as an operator in bulk"). He is right, and it was a category error:
        # an operator in this codebase is something the engine runs over regions to produce derived
        # data, declared by a `consumes` frozenset -- and "arrange the open windows in a grid" eats
        # no axis and produces no pixels. It was never in `_OPERATIONS`, but it sat in this stack
        # with the same card, the same styling and the same `_enable_operators` gate, which is what
        # made it read as one. It is a WINDOW-MANAGEMENT command, so it is a View-menu action now.
        _minerva = [op for op in _OPERATIONS if op.key == "minerva"]
        ordered = _minerva + [op for op in _OPERATIONS if op.key != "minerva"]
        for op in ordered:
            # ELIDED, not shortened: the blurb is where the registry says what the operator
            # actually does, and this pane is ~300 px wide, so the plain QPushButton was cutting
            # every description off mid-word at the card's edge. See _qtstyle.operator_card.
            card = _operator_card(op.label, op.blurb)
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

    def gallery_scope(self):
        """The scope a gallery would open on RIGHT NOW: the plate selection, else the whole thing.

        The selection plumbing is the one that already exists — ``selected_region_fovs()``, which is
        fed by ``PlateOverview.selected_wells()`` through ``_on_selection_changed`` and by the
        shift-drag marquee. A gallery therefore inherits the marquee, Cmd/Ctrl-A, and shift-click
        refinement for free, and there is no second selection mechanism to keep in step. The pairs
        it returns are ``(region, fov)``, which is exactly the ``{region: [fov, ...]}`` mapping
        ``run_plate(regions=...)`` takes — so a cropped well stays cropped all the way through.

        Returns ``None`` (never an empty gallery) when no acquisition is open.
        """
        if self._meta is None:
            return None
        sel = self.selected_region_fovs()
        if sel:
            return _GalleryScope.from_region_fovs(self._meta, sel, time_point=self.time_point)
        return _GalleryScope.whole(self._meta, time_point=self.time_point)

    def _open_gallery_view(self):
        """Tile the selected Regions side by side, one row each, one column per channel.

        The port of hongquanli/gallery-view's Region view ("Add Region view: stitched per-region
        MIPs", #7), adapted rather than imported for the same reason ``_napari3d`` adapts its 3-D
        recipe: gallery-view pins napari <0.6 and we run 0.6.6. See :mod:`squidxplorer._gallery` for
        which of its decisions were taken and which two were diverged from.

        SUBSET-NATIVE, and that is the whole design rather than an option on it: the scope is
        :meth:`gallery_scope`, i.e. the plate selection when there is one and the whole acquisition
        when there is not. One code path, two scopes.

        ONE gallery at a time. A second click RESCOPES and raises the open one instead of stacking
        a second window on the first, because "gallery of the current selection" is a question with
        one answer, and two galleries side by side is what the gallery itself is for.
        """
        scope = self.gallery_scope()
        if scope is None:
            self._readout.setText("Open an acquisition before opening the Gallery View.")
            return
        if scope.is_empty():
            self._readout.setText(
                "Gallery View: this acquisition has no regions with FOVs to tile.")
            return

        from squidxplorer._gallery_window import GalleryWindow

        title = self._acq_name or "acquisition"
        win = getattr(self, "_gallery", None)
        if win is not None and win.isVisible():
            win.rescope(scope, title=title)
        else:
            win = GalleryWindow(self._reader, self._meta, scope, title=title, parent=None)
            self._gallery = win
            win.resize(min(1400, 220 + 180 * max(1, len(scope.channels))), 900)
            win.show()
        win.raise_()
        win.activateWindow()
        msg = f"Gallery View: {scope.describe(self._meta)}"
        self._readout.setText(msg)
        self.log.info("%s", msg)

    def _open_native_3d(self):
        """Popout napari 3D on the current region's centre FOV at native resolution (gallery-view
        recipe). Fails to the LOG by name, never silently.

        It carries NO contrast or colormap. It used to harvest both off ``self._mosaic_pane``'s
        layers, which have never existed: the pane was pinned to None on 2026-07-23 and the harvest
        could only ever produce two empty dicts. ``open_native_3d`` defaults both to None and
        resolves the acquisition's own ``display_color`` and an autoscale, which is what this call
        has actually been doing all along. A window's on-screen LUTs reach 3D through
        ``RegionViewer``, which has the layers.
        """
        if self._reader is None or self._meta is None:
            self._readout.setText("No acquisition open — drop one before opening the 3D view.")
            return
        region = getattr(self, "_mosaic_region", None) or self._cursor.region
        if region is None:
            self._readout.setText("No region is open to render in 3D.")
            return
        try:
            from squidxplorer._napari3d import open_native_3d

            open_native_3d(self._reader, self._meta, region)
            log.info("opened native napari 3D popout for region %s", region)
        except Exception as exc:                     # noqa: BLE001 - NAMED, to the log and readout
            log.error("native 3D view failed for region %s: %s", region, exc)
            self._readout.setText(f"3D native view failed: {exc}")

    # -- operator UIs live as tabs in the band's right column: the Operators home tab, one tab
    # -- per operator you open, and any result a panel publishes. ---------------------------------
    def _open_op_tab(self, key: str, title: str, builder, tabs=None):
        """Open (or focus) a UI as a tab. Built lazily, once. *tabs* is the bar it belongs in;
        there is one bar (the band's right column) and it is the default.
        If the UI is currently detached (see _detach_tab), focus its floating window instead —
        never rebuild, so a widget's live state survives."""
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
        tabs.setCurrentWidget(w)

    #: How many tabs at the head of the process console are FIXED: [0] Operators. It cannot close
    #: and cannot detach, so the indices above it never move and a plain `index < _FIXED_TABS` is a
    #: sound test. It was 2 while the Log was tab [1]; the Log is now a sibling panel in
    #: `_right_col` and is protected by being always on screen instead of by this counter. Note
    #: `_DetachTabBar(first_detachable=1)` already agreed with 1.
    _FIXED_TABS = 1

    #: Registry key for the floated log in `_floating`. Not an entry in `_op_tabs`: the log is not
    #: an operator UI and must never be routed through `_dispose_tab_widget`, which deletes.
    _LOG_FLOAT_KEY = "__log__"

    def show_log(self) -> None:
        """Bring the one global console to the front, wherever it currently is.

        THE INVARIANT: the panel exists for the life of the window and is reachable from View > Log
        in every state — docked, collapsed or floated. This is the method that makes that true, so
        it has to answer for all three:

        * floated  -> raise and activate its window;
        * collapsed -> expand it;
        * docked   -> make sure the strip is showing it (it always is: it is a splitter child that
          cannot be collapsed to zero).

        It used to end in ``_left_tabs.setCurrentWidget(panel)``, which was the whole of it while
        the log was a tab. There is no tab to select now.
        """
        panel = getattr(self, "_log_panel", None)
        if panel is None:
            return
        win = self._floating.get(self._LOG_FLOAT_KEY)
        if win is not None:
            if panel.collapsed:
                panel.set_collapsed(False)
            win.show()
            win.raise_()
            win.activateWindow()
            return
        if panel.collapsed:
            panel.set_collapsed(False)
        panel.setVisible(True)

    # -- the console in a window of its own (Julio: "Log (option to open in a new window)") --------
    def _float_log(self):
        """Open the one global console in its own window, and give it back on Re-dock.

        THIS PARTLY REVERSES 2026-07-29 Task 1, deliberately, and the difference is the whole
        justification. The `_log_window` that was deleted was constructed and shown on EVERY
        launch, which is why Spencer saw it "open over the main window" every time. This is a user
        gesture on an always-present panel: docked by default, a window only when asked for.

        It reuses `_FloatWindow` rather than hand-rolling a second float, which matters: the old
        `_log_window` was one of the four widgets handed a Python-owned Fusion QStyle that ~QWidget
        then unpolished after GC (the segfault pinned by tests/test_window_lifetime.py), and
        `_FloatWindow` explicitly refuses that style for that reason (_qt_tabs.py:94-97). The
        hazard is fixed AT THE SEAM a new float uses, not merely absent.

        Its close handler RE-DOCKS. An operator float's close disposes the widget through
        `_dispose_tab_widget`; doing that to the console would delete a live sink on the
        process-wide root logger and lose it for good, which is the one outcome that would make
        this the wrong call.
        """
        panel = getattr(self, "_log_panel", None)
        if panel is None:
            return None
        win = self._floating.get(self._LOG_FLOAT_KEY)
        if win is not None:                     # already out: raise it, never build a second
            win.raise_()
            win.activateWindow()
            return win
        if panel.collapsed:
            panel.set_collapsed(False)          # a floated console that shows only its header is a
                                                # window with nothing in it
        key = self._LOG_FLOAT_KEY
        win = _FloatWindow("Log", panel,
                           on_close=lambda *_: self._redock_log(),
                           on_redock=lambda *_: self._redock_log())
        win._home_tabs = None                   # it has no tab bar to go home to; _redock_log knows
        self._floating[key] = win
        win.show()
        return win

    def _redock_log(self):
        """Put the console back in `_right_col`, under the operators. Idempotent."""
        win = self._floating.pop(self._LOG_FLOAT_KEY, None)
        if win is None:
            return
        panel = win.take_content()              # the SAME widget: the log's scrollback survives
        win.close()
        win.deleteLater()
        if panel is None:
            return
        col = getattr(self, "_right_col", None)
        if col is None:                         # no layout to return to (never in a built window)
            return
        col.addWidget(panel)                    # index 1: _left_tabs is still index 0
        panel.setVisible(True)
        col.setSizes(list(_RIGHT_COL_SIZES))    # the same split it opened at, not a second guess

    def _close_op_tab(self, index: int, tabs=None):
        tabs = self._left_tabs if tabs is None else tabs
        if index < self._FIXED_TABS and tabs is self._left_tabs:   # the Operators home tab
            return
        w = tabs.widget(index)
        tabs.removeTab(index)
        self._dispose_tab_widget(w)

    def _dispose_tab_widget(self, w):
        """The ONE teardown path for an operator UI — tab close, float close, and app exit all
        route here so they can't drift: registry pop, stale-ref clear, shell kill, delete."""
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

        *tabs* is the bar the tab is in and defaults to the one there is, so IMA-209's callers
        and tests are unchanged."""
        tabs = self._left_tabs if tabs is None else tabs
        if index < self._FIXED_TABS and tabs is self._left_tabs:
            return None                      # the Operators home tab is fixed: it never detaches
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
        return win

    def _on_float_closed(self, key: str):
        """User closed the floating window: same fate as closing the tab."""
        win = self._floating.pop(key, None)
        if win is None:
            return
        w = win.take_content()
        if w is not None:
            self._dispose_tab_widget(w)

    def _redock(self, key: str):
        """Re-dock button: return the floated widget to the tab bar — the SAME object, so its
        live state survives (close-and-reopen would kill it)."""
        win = self._floating.pop(key, None)
        if win is None:
            return
        title = win._tab_title
        # `is None`, never `or`: an EMPTY QTabWidget is falsy in PyQt, so `_home_tabs or _left_tabs`
        # sent every re-dock from a just-emptied bar to the wrong place.
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
        tabs.setCurrentWidget(w)

    def _on_progress(self, done: int, total: int):
        """A run advanced by a WELL. Feeds the log panel's activity header.

        It no longer writes the status line. ``_on_unit_progress`` does, because it is the finer
        and therefore the more useful count, and because two slots writing one QLabel is the
        "two representations of one truth" defect ``squidxplorer._activity`` was written to avoid —
        here it would have flickered between wells and FOVs on every field.
        """
        # Feed the activity registry the log panel's header reads — this is what turns "the GUI is
        # doing something" into a visible line. Advanced from THIS slot (the GUI thread), never
        # from the worker: the panel writes a QLabel and a worker thread must not.
        self._activity.advance("operator-run", done, total)

    def _on_unit_progress(self, report):
        """A run advanced by one ENGINE UNIT (a FOV, or a region for a region operator).

        This is the answer to Julio's 2026-08-03 report. The well counter above cannot see inside a
        well, so a decon over one region sat at ``0/1`` for 433 seconds; this one counts the 27 FOVs
        the engine actually iterates, and carries a time-remaining estimate with them.

        It owns pane 1's status line, and it forwards the same immutable report to the window that
        ASKED for the run — the region window, which had no progress affordance at all.
        """
        run = self._run
        self._run_readout(f"● {report.sentence()}{run.dest}")
        self._tell_requester(run.requester, "operator_progress", report)
        # ...and to the ONE bar next to the memory bar, which is where Julio asked to see a run
        # WHEREVER it was started from ("in bulk or in a specific window"). The requester above is
        # told only when the run came from a region window; this covers both, and the preview.
        self._publish_progress(report)

    def _publish_progress(self, report) -> None:
        """Hand a ``ProgressReport`` (or None for idle) to the window navigator's work bar.

        Routed through the ViewerManager rather than reaching into ``self._open_views`` because the
        manager is the thing that outlives the navigator widget: the panel can be closed and rebuilt,
        and a producer must not have to know whether it currently exists.
        """
        mgr = getattr(self, "_viewer_manager", None)
        if mgr is not None:
            mgr.set_run_progress(report)

    def _clear_progress_if_idle(self) -> None:
        """Take the work bar down, but ONLY once nothing is left running.

        Asked on every worker's ``finished``, and both questions are needed. ``operator_busy``
        deliberately does not count the raw preview (it opts out with ``IS_PREVIEW``, see
        ``_run_scope.operator_busy``), so an operator run ending while a preview is still filling the
        plate would otherwise hide a bar that has live work behind it.
        """
        if _run_scope.operator_busy(self._worker, self._retired):
            return
        preview = getattr(self, "_preview", None)
        if preview is not None and preview.isRunning():
            return
        self._publish_progress(None)

    @staticmethod
    def _tell_requester(requester, method: str, *args) -> None:
        """Call one of the four ``operator_*`` callbacks on the window that asked, if it has it.

        Duck-typed and forgiving in ONE direction only: a window that does not implement a callback
        is skipped, and a window that RAISES inside one is logged and skipped. One window's failure
        must not abort a run, or take down the three other windows waiting to be told — the same
        rule ``_deliver_result_to_windows`` already follows.
        """
        if requester is None:
            return
        tell = getattr(requester, method, None)
        if tell is None:
            return
        try:
            tell(*args)
        except Exception as exc:                 # noqa: BLE001 - one window's failure is its own
            log.warning("view %s could not take %s: %s",
                        getattr(requester, "window_id", "?"), method, exc)

    @property
    def time_point(self) -> int:
        """Which timepoint the plate is showing. 0 when there is nothing to navigate."""
        bar = getattr(self, "_time_point_bar", None)
        return bar.time_point if bar is not None else 0

    def _on_time_point_changed(self, time_point: int) -> None:
        """A user moved the plate's timepoint. Re-read the plate at that timepoint.

        Only a USER gesture reaches here: TimePointBar does not echo its own programmatic moves,
        which is what stops this looping when we set the bar from an ingest.
        """
        # `say`, not `_say`. There is no `_say` on this window and there never was, so THIS LINE
        # RAISED AttributeError ON EVERY USER DRAG of the timepoint bar — the first statement in
        # the only slot the bar calls, so nothing below it ran either: the plate never re-read at
        # the new timepoint and the exception surfaced out of Qt's slot dispatch. Caught 2026-08-05
        # while proving the Minerva export follows the slider; the export could not follow a bar
        # that could not be moved.
        self.say(f"time_point {time_point + 1} of {self._time_point_bar.count}")
        # Tell the PLATE, which is what the loupe reads its timepoint from. This comment used to
        # claim the loupe needed no invalidation "because it caches coarse tiles per (well,
        # timepoint)" — true of the cache and irrelevant, because nothing passed a timepoint in:
        # every read defaulted to frame 0 whatever the slider said. The plate THUMBNAILS are a
        # separate matter: they were streamed by a worker reading at a fixed timepoint, so showing
        # a new one means asking again rather than filtering what already arrived.
        if self._overview is not None:
            self._overview.set_time_point(time_point)
        if self._reader is not None:
            # `_return_to_raw` restarts the preview, and since 2026-08-05 the preview reads the
            # bar's timepoint and caches its cells under it. Before that this call re-read the
            # whole plate to repaint frame 0's pixels: the request was honest and the answer was
            # the same picture. Now it is a re-read the first time a timepoint is visited and a
            # cache replay every time after, including stepping back.
            self._return_to_raw()

    # `_sync_top_row_height` WAS HERE, and it is deleted rather than adapted (2026-08-03). It swung
    # the strip's cap between 240 and 520 px while the Log TAB was in front, inferring "the user is
    # reading the console" from a tab selection. The log is no longer a tab, so there is nothing
    # left to read the intent from — and its own docstring said it was "deliberately not a
    # remembered setting and not a drag handle", which is precisely what `_right_col`'s splitter
    # handle now is. `_BAND_MAX_PX` is the single cap and the boundary between
    # Operator and Log is dragged, not guessed.

    def _on_tab_changed(self, index: int = -1, force: bool = False):
        """Put the plate back on the whole dataset once a run has drained.

        It used to do much more: an exploration tab claimed to be scoped to its subset, so
        selecting one re-pointed the plate's status dots and the FOV slider at that subset and
        selecting away restored the plate. The tabs are gone (2026-08-05) and only the RESTORE
        half was ever plate-wide, so that is all this is now.

        A LIVE run is the one thing we won't retarget under: the worker is pushing into the slider
        this call would rebuild. So the switch is DEFERRED, not dropped (``_request_resync``) —
        dropping it is what left the view lying about what it shows (BUG 2), because nothing
        re-emits ``currentChanged`` when the run later drains.

        ``force=True`` is what ``_on_run_drained`` and ``_deliver_pending_resync`` call it with,
        and it is the only way past the guard below: selecting an operator tab does not touch the
        plate, and never did once there was no subset-scoped tab to select away from.

        It used to size the band too (``_sync_top_row_height``, deleted 2026-08-03): the Log
        was a tab, so the tab you selected said whether you were reading the console or working the
        plate. Both are on screen at once now and the boundary is a splitter handle."""
        if self._reader is None or self._overview is None or self._tabs_muted:
            return
        if _run_scope.operator_busy(self._worker, self._retired):
            # Defer for an OPERATOR RUN only, never for the raw preview: postponing on a streaming
            # preview is what stranded the restore. See _run_scope.operator_busy -- this is the
            # third gate that was asking "is any producer alive" when the question is "is a RUN
            # alive".
            self._request_resync()
            return
        if not force:
            return                   # a plain tab selection: the plate is already plate-wide
        self._overview.set_all_status("empty")
        self._apply_layers()

    def _request_resync(self):
        """Remember that the plate needs to catch up once the run drains.

        Both IMA-205 bugs are the same missing edge: a sync that arrives while a run is live is
        silently discarded, and no later event re-delivers it. The pending flag IS that later
        event; ``_on_run_drained`` fires it as soon as the last worker thread actually exits."""
        self._pending_resync = True
        if not _run_scope.operator_busy(self._worker, self._retired):
            # NOTHING IS RUNNING, SO NOTHING WILL EVER DELIVER THIS. `_on_run_drained` is the only
            # other caller, and it fires on QThread.finished -- so with no live thread the flag was
            # set and then sat there forever, and the plate never came back to the whole dataset.
            # The bug was never in the timing -- deferral is simply only correct while something is
            # running.
            #
            # Delivered on the event loop rather than inline: a zero timer runs after the current
            # stack unwinds, and processEvents() delivers it, so it stays deterministic for the
            # tests too.
            QTimer.singleShot(0, self._deliver_pending_resync)

    def _deliver_pending_resync(self):
        """Deliver a deferred tab switch. Idempotent, and re-defers if a run started meanwhile."""
        if not self._pending_resync or _run_scope.operator_busy(self._worker, self._retired):
            return
        self._pending_resync = False
        self._on_tab_changed(force=True)

    def _on_run_drained(self, worker=None):
        """A worker thread has exited. Deliver any tab switch that was deferred while it ran.

        Fires on QThread.finished, so it also covers a run that was STOPPED (closing a tab mid-run)
        — ``_stop_worker`` returns immediately but the thread keeps going until its current well is
        done, and ``_busy()`` stays True for all of that window.

        ``worker`` IS THE THREAD THAT EXITED, and it decides whether this is the RUN's drain or
        merely some thread's. ``run_operator`` retires the raw preview on its way in ("the operator
        supersedes the raw preview", below), and ``_retire`` hooks EVERY retired thread's
        ``finished`` here on purpose — a deferred re-sync must wait for the last of them. But the
        preview's exit is not the run's drain, and treating it as one closed the run's books early:
        observed on 2026-08-03, 5 runs in 200, as ``_close_requester_pair`` running while the
        operator worker's own ``runProgress`` and ``failed`` were still sitting in the queue. The
        window that asked was then unsubscribed (the run's requester cleared) before a single unit
        report reached it, so its bar's ONLY frame was the drain's final "2 of 2", and a failed run
        was reported as "produced nothing" instead of naming its cause. ``operator_busy`` cannot
        catch this: the run's thread has already exited, so it is honestly not busy — what has not
        happened yet is the DELIVERY of its terminal signals, which only its own ``finished`` can
        stand behind (queued after them, from the same thread).
        """
        # The work bar comes down here and not on ``finished_ok``, for the reason the console pair
        # below is closed here: this slot fires on ok, failed and STOPPED alike, and a bar that is
        # only taken down on success is a bar left running over a dead run.
        self._clear_progress_if_idle()
        if _run_scope.operator_busy(self._worker, self._retired):
            return                       # another operator run is still draining — wait for it
        if not getattr(worker, "IS_PREVIEW", False):
            # No operator run is in flight now — clear the activity header. end() is a no-op if it
            # was already cleared, so a failed/stopped run that never reached here does not leave
            # it stuck.
            self._activity.end("operator-run")
            # ...and close the console's started/done pair. This fires on ok, failed and STOPPED
            # alike, which is why the pair is closed here and not on finished_ok: an action that
            # starts and then says nothing is indistinguishable from one still running, and a
            # stopped run is exactly the case that would have gone quiet. A run that landed nothing
            # is reported as a failure however politely the engine returned.
            # A region whose FOVs were only PARTLY read never reached `acc.complete()`, so its
            # result was still sitting in the run's books with nothing left running to finish it.
            # Resolved here, before the books are closed, so the two lines below cannot report a
            # run as done when it produced no layer. See `OperatorRun.settle_stranded`.
            stranded = self._settle_stranded_results()
            run = self._run
            if run is not None and run.action is not None:
                action, run.action = run.action, None
                elapsed = time.monotonic() - run.began
                landed = getattr(self._worker, "landed", None)
                outcome, why = run.close(landed, stranded, elapsed)
                if outcome == _measure.OK:
                    self.log.done(action, elapsed, address=run.address)
                else:
                    self.log.failed(action, why, address=run.address)
                self._close_requester_pair(landed, elapsed)
        # A genuine drain: every worker has exited AND (finished being FIFO-queued after this
        # worker's tileReady/streamEnded) their terminal slots have already run on this thread.
        # Bump BEFORE the pending-resync branch so a run with no deferred switch still counts.
        self._runs_settled += 1
        if not self._pending_resync:
            return
        self._pending_resync = False
        self._on_tab_changed(force=True)

    def _close_requester_pair(self, landed, elapsed: float) -> None:
        """Tell the window that ASKED that its run is over — exactly once, whatever the outcome.

        Called from ``_on_run_drained``, which fires on ``QThread.finished``: success, failure and
        a STOPPED run all reach it. That is deliberate and it is the whole safety property of the
        region window's bar. A bar that is only taken down on success is a bar left spinning over
        a dead run, which teaches the user that the indicator lies — the exact failure
        ``squidxplorer._activity``'s docstring names.

        A run that landed nothing is reported as a FAILURE however politely the engine returned,
        the same rule the status line and the console line above already follow.
        """
        run = self._run
        requester, run.requester = run.requester, None
        action = run.label or "the operator"
        reason, run.error = run.error, None
        run.label = None
        if requester is None:
            return
        # THE RUN'S FINAL COUNT, read from the worker rather than waited for. The worker's last
        # ``runProgress`` and its ``finished`` are two signals queued from the same thread, and the
        # window is torn out of the run by whichever arrives first — so the last unit's report was
        # being dropped on a fast run, and the bar's last visible frame was "1 of 2". Asking the
        # worker for its own tally here is not a second source of truth: it IS the source the
        # signal carries, read directly instead of through a race.
        final = getattr(self._worker, "progress_report", None)
        if final is not None:
            self._tell_requester(requester, "operator_progress", final)
        if landed == 0 or reason:
            # A run that landed nothing is a FAILURE however politely the engine returned, and it
            # says so with whatever cause was captured rather than a bare "nothing happened".
            reason = reason or f"produced nothing after {elapsed:.1f} s"
            self._tell_requester(requester, "operator_failed", action, reason)
        else:
            self._tell_requester(requester, "operator_done", action, float(elapsed))

    def _refresh_view_combo(self) -> None:
        """Offer exactly the layers this plate HOLDS PIXELS FOR, raw first.

        Derived from ``_op_stack`` on every call and never bookkept: the stack is the same thing
        the Layers tab draws and the same thing ``_apply_layers`` renders from, so the combo cannot
        offer a layer the plate cannot show. A run that has not happened yet contributes nothing,
        which is the honest answer -- and the reason it is populated from the stack rather than
        from ``runnable_operators()``, which would list every operator the build can run and make
        the plate look broken for the ones nobody has run.
        """
        combo = getattr(self, "_view_combo", None)
        if combo is None:
            return
        keys = ["raw"] + [ly.key for ly in self._op_stack.layers() if ly.key != "raw"]
        current = self._overview._active if self._overview is not None else "raw"
        blocked = combo.blockSignals(True)
        try:
            combo.clear()
            for key in keys:
                combo.addItem(operator_label(key) if key != "raw" else "raw", key)
            idx = keys.index(current) if current in keys else 0
            combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(blocked)
        combo.setEnabled(len(keys) > 1)

    def _on_view_combo(self, _index: int) -> None:
        """The user picked a layer to look at. Routed through the stack, not straight to the pane.

        ``_apply_layers`` is what decides which layer is drawn (the topmost ENABLED one), and it is
        also what the Layers tab writes through. Setting ``_active`` here instead would be a second
        surface racing the first over one quantity -- the defect this file names most often.
        """
        combo = getattr(self, "_view_combo", None)
        if combo is None or self._overview is None:
            return
        key = combo.currentData()
        if not key:
            return
        for layer in self._op_stack.layers():
            if layer.key != "raw":
                self._op_stack.toggle(layer.key, layer.key == key)
        self._apply_layers()
        self._refresh_layers_tab()

    def operator_kwargs_for(self, key: str) -> dict:
        """THE parameters an operator would run with RIGHT NOW, off its live panel.

        Julio, 2026-08-06: *"Controls should be in the 'operators for this window' to show the UI
        controls for the operator in the dropdown and apply the newly set parameters."*

        The second half is what was missing. A window's own Run button called
        ``run_operator(key, regions=…, save=…, requester=self)`` with **no** ``operator_kwargs``,
        while the plate's panel Run passed ``self.kwargs()`` -- so tuning the stitcher's blend
        width, its outlier thresholds or its z handling on the plate and then pressing Run in the
        window ran the DEFAULTS, with every control on screen saying otherwise. That is the same
        defect shape as `_workers._OperatorWorker`'s preview branch and `_command.EngineExecutor`'s,
        both fixed on 2026-08-05/06: two entry points to one run must pass the same arguments.

        ONE READER, and it reads the PANEL rather than a copy of its values. Both panel families
        already answer ``kwargs()`` -- the hand-written ones (``StitcherPanel``) and the ones
        generated from an operator's declared ``params``
        (``_param_panel.GenericOperatorPanel``) -- so this needs no per-operator case and a plugin's
        operator is covered with no edit here.

        ``{}`` when the operator's tab has never been opened, which means "run with the declared
        defaults" and is exactly right: there is no panel, so there is nothing the user has set.
        """
        panel = (getattr(self, "_op_tabs", None) or {}).get(str(key))
        if panel is None:
            return {}
        reader = getattr(panel, "kwargs", None)
        if not callable(reader):
            return {}
        try:
            kwargs = dict(reader() or {})
        except Exception as exc:                 # noqa: BLE001 - a refused setting, NAMED
            log.warning("%s panel could not report its parameters: %s: %s",
                        key, type(exc).__name__, exc)
            return {}
        # The stitcher's z handling lives on its own combo rather than in `kwargs()` (which is
        # `stitch_region`'s keyword set), and `StitcherPanel._run` adds it on the way out. It is a
        # PARAMETER of the run either way, so it is added here too -- otherwise the window's Run
        # silently reverted the one control this round of feedback was about.
        combo = getattr(panel, "z_operator_combo", None)
        if combo is not None and "z_operator" not in kwargs:
            kwargs["z_operator"] = combo.currentText()
        return kwargs

    def operator_params_text(self, key: str) -> str:
        """A ONE-LINE summary of what *key* is currently set to, for a window to print.

        Julio: *"The control button should print a small text to it's side saying what the UI
        parameters are set to. When I modify in the plate window, the printed values should change
        in the roi window."*

        Derived from :meth:`operator_kwargs_for` on every call, never cached and never mirrored
        into the window: a second copy of a value the user is actively editing is precisely how the
        printed text and the run come to disagree, which is the thing being fixed. The window asks
        when it repaints, so the answer cannot be stale.
        """
        kwargs = self.operator_kwargs_for(key)
        if not kwargs:
            return "defaults"
        parts = []
        for name in sorted(kwargs):
            value = kwargs[name]
            if isinstance(value, float):
                value = f"{value:g}"
            elif isinstance(value, (list, tuple)):
                value = f"{len(value)} selected"
            parts.append(f"{name}={value}")
        return " · ".join(parts)

    def _activate_operator(self, key: str):
        """Operator card / menu clicked: open the operator's UI tab.

        Two sources, in this order, and NEITHER of them is silent:

        1. a HAND-WRITTEN panel, named by the ``Operation`` template's ``build_tab``. These do more
           than parameter entry (``StitcherPanel`` converts units and refuses a plane-op;
           ``DeconQCPanel`` runs a QC loop and publishes a picture into pane 3), so they win.
        2. otherwise a panel built FROM THE DECLARATION — :class:`squidxplorer._param_panel
           .GenericOperatorPanel` over the operator's ``params``. This is how ``spot``, ``cellpose``
           and an operator discovered from somebody else's package get real controls without an
           edit here.

        This method used to end at step 1 with a bare ``if op is not None:``, so a key the card
        table did not know made the click land on NOTHING: no tab, no error, no line in the
        readout. Silence was the bug. Every path below now opens a panel or says why it cannot.
        """
        if self._reader is None or self._overview is None:
            self._readout.setText("open an acquisition first")
            return
        op = _OPERATIONS_BY_KEY.get(key)
        if op is not None:
            self._open_op_tab(op.key, op.label, getattr(self, op.build_tab))
            return
        from squidxplorer._param_panel import GenericOperatorPanel, panel_refusal

        why = panel_refusal(key)
        if why:
            self._readout.setText(why)
            return
        self._open_op_tab(key, operator_label(key),
                          lambda k=key: GenericOperatorPanel(self, k))

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
        # The other z-reduction. `_build_run_tab` is ONE builder for every z-reducer, so the
        # focus-reference-plane operator needs no tab code of its own -- only this hand-off.
        return self._build_run_tab(_OPERATIONS_BY_KEY["reference"])

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
        from squidxplorer._op_panels import StitcherPanel

        return StitcherPanel(self)

    def _build_decon_tab(self) -> QWidget:
        """The RL semi-convergence loop's controls (IMA-252 + IMA-decon-stitch-ui).

        The controls are here; the picture they produce -- the deconvolved 2-D image in turbo
        with the x-z and y-z strips concatenated -- opens as a tab beside them via
        :meth:`publish_qc_result`. It was `_build_plane_op_tab` (a preview button and nothing
        else), which gave no way to choose an iteration count at all.
        """
        from squidxplorer._op_panels import DeconQCPanel

        return DeconQCPanel(self)

    # -- the host surface the pane-1 operator panels use -----------------------------------
    #
    # Deliberately three small methods rather than handing a panel the whole window: if a
    # panel starts needing more than this, that is a coupling worth seeing in a diff.

    def say(self, text: str) -> None:
        """Put an operator panel's sentence in the window's status line."""
        if text:
            self._run_readout(text)

    def publish_qc_result(self, widget: QWidget, title: str) -> None:
        """Show *widget* as a result tab beside the operators, and bring it to the front.

        THE seam between an operator panel and the window. Keyed by title, so re-running the same
        subject reuses its tab instead of stacking one per iteration.

        IT LANDS IN `_left_tabs`, WHICH IS ON SCREEN. It used to go to the exploration pane's tab
        bar, and for six weeks after 2b8fbc5 that bar had no parent at all: the decon QC composite
        — the turbo x-y / x-z / y-z picture the whole iterate-and-look loop exists for — was
        computed, put in a tab, and shown to nobody. Pressing Run silently produced nothing the
        user could see. The pane is gone now, and the result goes where the controls that asked
        for it already are.
        """
        self._open_op_tab(f"qc:{title}", title, lambda w=widget: w)

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
        z-stack at full depth. This builder omits the "Run on the whole plate" / destination half
        of _build_run_tab because write_plate's _validate_image accepted Z == 1 only, so there was
        nothing to write -- and it said "the moment the OME-Zarr writer learns Z > 1, this method
        can simply forward to _build_run_tab and disappear."

        THAT MOMENT HAS PASSED. IMA-277 taught _validate_image that a plane-op's full-depth result
        is a real result, and per-plane fusion taught stitch_region to fuse every z. So the save
        path exists and only this card has not been given it: preview-only is now a GUI GAP TO
        CLOSE, not a contract. Do not cite Z == 1 to justify it.

        The preview path itself is unchanged and needs no worker edit: _OperatorWorker's save=False
        branch streams the per-FOV loop, and _on_well already indexes image[0, :, 0] -- for a plane-op
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
                from squidxplorer import FlatfieldProfile
                from squidxplorer._flatfield import set_profiles
                # EVERY CHANNEL, not plane 0. A stored profile is (C, Y, X) with one genuinely
                # different gain field per channel; `from_npy(path)` defaults to channel 0, so
                # this button used to correct 488, 561 and 638 with the 405 field — 99.8% of
                # pixels changed, by up to 1799 counts, on the 10x set. per_channel_from_npy is
                # the one place a channel NAME becomes a plane index of that file.
                names = [c["name"] for c in (self._meta or {}).get("channels", [])]
                if not names:
                    prof_lbl.setText("no acquisition open, so nothing says which channel each "
                                     "field in the file belongs to. Open a plate first.")
                    return
                try:
                    profiles = FlatfieldProfile.per_channel_from_npy(path, names)
                except Exception as exc:                     # bad file -> say so, keep the tab alive
                    prof_lbl.setText(f"could not load {Path(path).name}: {exc}")
                    return
                frame = tuple(self._reader.metadata["frame_shape"]) if self._reader else None
                shapes = sorted({p.shape for p in profiles.values()})
                if frame is not None and shapes != [frame]:
                    prof_lbl.setText(f"profile is {shapes[0]}, this acquisition's frames are "
                                     f"{frame} -- wrong profile for this plate")
                    return
                set_profiles(profiles)
                state["profile"] = path
                prof_lbl.setText(f"{Path(path).name}  {len(profiles)} channel(s)  {shapes[0]}")
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
                    # Installed for THE CHANNEL IT WAS ESTIMATED FROM, and only that one. The
                    # worker reads tiles of `ch` and nothing else, so it has measured nothing
                    # about the other channels; a run over them now refuses BY NAME instead of
                    # correcting them with this field.
                    from squidxplorer._flatfield import active_profiles, set_profile
                    set_profile(profile, channel=ch)
                    state["profile"] = f"estimated:{ch}"
                    others = [c["name"] for c in (self._meta or {}).get("channels", [])
                              if c["name"] not in active_profiles()]
                    missing = (f"  (no profile yet for {', '.join(others)} — estimate each one)"
                               if others else "")
                    prof_lbl.setText(f"estimated from plate ({ch})  {profile.shape}{missing}")
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
        """Generic plane-operator tab (MIP, …): pick a destination, run over the whole plate → a
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
            f"{TARGET_OPEN} — every region held by the open viewer windows. Picking it prints the "
            "exact window list, and each window's regions, to the log console.\n"
            f"{TARGET_PLATE} — every region of the acquisition.")
        run_row.addWidget(_rl); run_row.addWidget(target, 1)

        # PRINT THE TARGET WHEN IT IS CHOSEN, not only when Run is pressed. "Open views" is the one
        # target whose meaning is invisible from the combo: the other two name a surface the user is
        # looking at, this one names a set of windows scattered across the desktop, deduplicated.
        target.currentTextChanged.connect(
            lambda choice: (self._print_open_views_target(f"Run {op.label}")
                            if choice == TARGET_OPEN else None))

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
                # Prints the block again at launch, deliberately: the log is the record of what was
                # run, and the state may have moved since the target was picked (a window closed).
                regions = self._print_open_views_target(f"Run {op.label}")
                if not regions:
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

    def _build_minerva_tab(self) -> QWidget:
        """Minerva Author hand-off (IMA-228): export the SELECTION, then open Author on it.

        Scope comes from :meth:`minerva_selection` — the plate's selected wells (all of their
        FOVs, or the fields a Shift+Alt box picked inside a mosaic), else the well open in the
        detail viewer, which means every FOV of it.

        ONE FILE PAIR PER REGION, not per FOV. This docstring said "one file pair per FOV
        (Minerva opens one 2D image at a time and SquidXplorer has no stitcher)" long after both
        halves of that stopped being true: there IS a stitcher (the region-operator seam
        ``export_selection`` fuses through), and a region's FOVs become ONE mosaic because
        Minerva lays out exactly one image (``"Layout": {"Grid": [["i0"]]}``, hardcoded) and
        opens only ``series[0]``. A FOV subset is that mosaic CROPPED, still one file.

        The timepoint is the one the window is showing — ``run_minerva_export`` reads
        ``self.time_point`` — so there is no control for it here and none is missing.
        """
        op = _OPERATIONS_BY_KEY["minerva"]
        w, v = self._op_tab_shell(
            op.label,
            "Writes an OME-TIFF plus a Minerva story for every selected region, at the timepoint "
            "the plate is showing, then starts Minerva Author. Zoom into a well and Shift+Alt-drag "
            "a box to export only the fields inside it - the mosaic is cropped to them. Author’s "
            "editor cannot be pointed at a file, so pick the .story.json below in its “Select "
            "File” dialog - the colours and contrast are already applied. To skip that step "
            "entirely, render a viewer instead (button below the paths).",
        )
        state = {"dir": None, "pairs": []}

        dir_lbl = QLabel("(defaults to a minerva_export folder in your home directory)")
        dir_lbl.setWordWrap(True)
        dir_lbl.setStyleSheet("color:#8b98ad;font-size:12px;")

        # Projection mode — the salesperson tool (squid2minerva convert.py) offers --mip/--z, so
        # hardcoding one here would be a capability regression. Driven by the operator registry.
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(QLabel("Projection"))
        proj = QComboBox(); proj.setStyleSheet(_COMBO_QSS)
        proj.addItems(available_plane_operators())
        proj.setCurrentText("mip")
        row.addWidget(proj); row.addStretch(1)

        # "channels need to be set to specific colors" - the colours ON SCREEN, which the export's
        # own defaults (acquisition display_color + 1/99.9 percentiles) do not know about. Checked
        # by default because matching what you are looking at is the request; harmless with no view
        # open, because on_screen_luts() returns None there and the defaults apply unchanged.
        luts_cb = QCheckBox("Match the LUTs of the focused view window")
        luts_cb.setStyleSheet(_CHECK_QSS)
        luts_cb.setChecked(True)
        luts_cb.setToolTip(
            "Use the contrast and colour you have on screen in the focused view window instead of "
            "the acquisition's channel colours and an automatic 1/99.9 percentile stretch.\n\n"
            "With no view window open there is nothing on screen to match and the automatic "
            "values are used. A channel that is not in that window keeps the automatic values "
            "too, and so does a channel showing a multi-stop colormap (viridis, turbo): Minerva "
            "stores one colour per channel and cannot hold a gradient.")

        launch_cb = QCheckBox("Open Minerva Author after exporting")
        launch_cb.setStyleSheet(_CHECK_QSS)
        launch_cb.setChecked(True)

        path_lbl = QLabel("")
        path_lbl.setWordWrap(True)
        path_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path_lbl.setStyleSheet("color:#8b98ad;font-size:11px;")
        copy_btn = QPushButton("Copy story path"); copy_btn.setStyleSheet(_BTN_QSS); copy_btn.hide()
        reveal_btn = QPushButton("Show in folder"); reveal_btn.setStyleSheet(_BTN_QSS); reveal_btn.hide()
        # THE ZERO-CLICK DESTINATION. A separate button and not a replacement for the Author
        # launch: Author is the EDITOR (waypoints, story text, masks) and needs its Select File
        # click because its server has no route, flag or URL that opens a file; render.py is the
        # VIEWER and needs none. Julio said "viewer". Both are offered; neither is assumed.
        render_btn = QPushButton("Render a Minerva viewer (no file picking)")
        render_btn.setStyleSheet(_BTN_QSS); render_btn.hide()
        render_btn.setToolTip(
            "Runs Minerva's own render.py on what you just exported and opens the finished "
            "exhibit. No Select File step.\n\n"
            "It is a viewer, not an editor: no waypoints, story text or masks.\n"
            "It writes a JPEG pyramid, so it is lossy; the OME-TIFF is untouched.\n"
            "Measured on this machine: about 2 s for a 2048x2048 4-channel crop and about "
            "132 s for a whole 11535x9635 4-channel region, plus about 13 s once per session "
            "while Minerva's renderer loads.\n"
            + _MINERVA_INTERNET_NOTE)

        def pick():
            d = QFileDialog.getExistingDirectory(self, "Save the Minerva export to folder")
            if not d:
                return
            state["dir"] = d
            dir_lbl.setText(d)

        def on_exported(pairs):
            state["pairs"] = pairs
            if not pairs:
                # An export that wrote NOTHING must not leave the previous one's paths on screen
                # with live Copy / Show in folder / Render buttons under them. `state["pairs"]` was
                # already emptied above, so the buttons had quietly become no-ops while still
                # naming files — a control that looks armed and does nothing.
                path_lbl.setText("")
                copy_btn.hide(); reveal_btn.hide(); render_btn.hide()
                return
            path_lbl.setText("\n".join(str(story) for _, story in pairs))
            copy_btn.show(); reveal_btn.show(); render_btn.show()

        def do_render():
            if state["pairs"]:
                self.run_minerva_render(state["pairs"])

        def do_copy():
            if state["pairs"]:
                QApplication.clipboard().setText("\n".join(str(s) for _, s in state["pairs"]))
                self._readout.setText("story path copied")

        def do_reveal():
            if state["pairs"]:
                from squidxplorer._minerva import reveal
                reveal(state["pairs"][0][1])

        pick_btn = QPushButton("Choose output folder…"); pick_btn.setStyleSheet(_BTN_QSS)
        pick_btn.clicked.connect(pick)
        # Named for the UNIT that lands: one fused mosaic per selected region, cropped to the
        # fields you boxed. "Export the selected FOVs" promised N files and wrote one per region.
        run = QPushButton("Export the selection (one mosaic per region)")
        run.setStyleSheet(_BTN_QSS)
        run.clicked.connect(lambda: self.run_minerva_export(
            out_dir=state["dir"], z_operator=proj.currentText(),
            launch=launch_cb.isChecked(), on_exported=on_exported,
            luts=self.on_screen_luts() if luts_cb.isChecked() else None,
        ))
        copy_btn.clicked.connect(do_copy)
        reveal_btn.clicked.connect(do_reveal)
        render_btn.clicked.connect(do_render)

        net_lbl = QLabel(_MINERVA_INTERNET_NOTE)
        net_lbl.setWordWrap(True)
        net_lbl.setStyleSheet("color:#8b98ad;font-size:11px;")

        v.addWidget(pick_btn); v.addWidget(dir_lbl)
        v.addLayout(row); v.addWidget(luts_cb); v.addWidget(launch_cb); v.addWidget(run)
        v.addWidget(_hline()); v.addWidget(path_lbl); v.addWidget(copy_btn); v.addWidget(reveal_btn)
        v.addWidget(render_btn); v.addWidget(net_lbl)
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

        A region the user boxed only PART of yields only those FOVs, and downstream that is the
        same one mosaic CROPPED to them. Nothing here decides that: ``selected_region_fovs``
        reads the plate's own FOV subsets, and this method's job is only to fall back to the
        detail viewer's well when the plate has no selection at all.

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

    def on_screen_luts(self) -> "Optional[dict]":
        """The per-channel LUTs of the view window the user is looking at, or ``None``.

        "Channels need to be set to specific colors" means the colours ON SCREEN, and the plate
        does not have them: the export's own defaults are the acquisition's ``display_color`` plus
        1/99.9 percentiles, neither of which knows that the user recoloured a layer or dragged a
        contrast slider. A :class:`RegionViewer` does know - ``_per_channel_luts`` reads it back
        off the napari layers - so this is the one hop from that window to the exporter.

        WHICH window: the manager's focused one, which is the window whose regions the plate is
        already highlighting (``viewFocused``) and the one the navigator shows as current. Picking
        "the first open window" instead would silently follow a window the user is not looking at.

        ``None`` when there is no view open, no focused window, or that window has no layers yet - 
        and ``None`` is not a failure. It is the plate-level export, and the percentile defaults
        are the right answer for it precisely because there is no screen to match.
        """
        mgr = getattr(self, "_viewer_manager", None)
        if mgr is None:
            return None
        # focused_id and windows are PROPERTIES on ViewerManager, not methods. They were called
        # with parentheses here, which raised TypeError on the int/list they return; the broad
        # except below swallowed it, so this returned None every single time and the on-screen
        # LUTs never reached Minerva. Found 2026-08-03, an hour after the feature shipped, by an
        # agent auditing an unrelated change. The except stays, because a window mid-teardown is
        # genuinely not an error, but it no longer hides a call-signature mistake: the non-None
        # path is now pinned by a test.
        try:
            wid = mgr.focused_id
            win = next((w for w in mgr.windows if getattr(w, "window_id", None) == wid), None)
            if win is None:
                return None
            luts = win._per_channel_luts()
        except Exception:                     # noqa: BLE001 - a window mid-teardown is not an error
            return None
        return luts or None

    def run_minerva_export(self, out_dir=None, z_operator: str = "mip", launch: bool = True,
                           on_exported=None, time_point=None, selection=None, luts=None):
        """Export the user's selection for Minerva Author and (optionally) open it.

        Runs off the GUI thread: projecting a well is real I/O plus compute, and starting
        Minerva Author polls a port for up to 90 s. Tests call this directly with launch=False.
        *selection* overrides :meth:`minerva_selection` (tests and future callers). *luts* is
        passed straight through to ``export_selection``: ``None`` means the percentile defaults,
        exactly as before this parameter existed. Deciding whether to match the screen belongs to
        the caller (the Minerva tab's checkbox calls :meth:`on_screen_luts`), not here - so this
        method has no opinion and stays trivially testable in both states.

        *t* is ``None`` by default, meaning THE TIMEPOINT THE WINDOW IS SHOWING. It used to
        default to the literal ``0`` and both GUI call sites took the default, so a
        multi-timepoint acquisition exported frame 0 whatever the timepoint bar said — the pixels
        on screen and the pixels in the OME-TIFF were different images, and nothing said so. An
        explicit *t* still wins, which is what keeps the CLI and the tests able to name one.
        """
        if self._reader is None or self._meta is None:
            self._readout.setText("open an acquisition first")
            return
        time_point = self.time_point if time_point is None else int(time_point)
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
        # ...but a region can now be CROPPED to some of its fields, and that changes the file that
        # lands. Say how many are cropped rather than letting "3 mosaics" mean either thing.
        per = (self._meta.get("fovs_per_region") or {})
        cropped = [r for r in regions
                   if 0 < len({f for rr, f in sel if rr == r}) < len(per.get(r) or [])]
        what = (f"{len(regions)} mosaic{'s' if len(regions) != 1 else ''} "
                f"({', '.join(regions)}, {len(sel)} FOVs"
                + (f", {len(cropped)} cropped" if cropped else "") + ")")
        n_t = self._meta.get("n_t", 1) or 1
        t_note = f" (t={time_point} of {n_t})" if n_t > 1 else ""
        self._minerva = w = _MinervaWorker(
            self._reader, sel, out_dir, z_operator, time_point=time_point, launch=launch, luts=luts)

        def on_launched(ok):
            if ok:
                # The URL is named and not just implied. Exactly ONE tab is opened now (Minerva
                # Author opens its own on a cold start, so we no longer open a second), and the
                # one way that leaves the user with none is Author's webbrowser call failing to
                # find a browser - in which case it returns False, the server serves anyway, and
                # this line is the address to paste.
                self._readout.setText(
                    f"✓ Minerva Author open at {_MINERVA_URL} - pick a .story.json "
                    f"({what}{t_note} exported)")
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
            # The SUCCESS line says a mosaic was cropped, not just the in-flight one. This is the
            # line that stays on screen and the only one a user reads after the export, and a crop
            # that reads identically to a whole region is the same silent difference the filename
            # suffix (`_2fov`) exists to prevent on disk.
            crop = [r for r in cropped if r in done]
            crop_note = (f", {len(crop)} cropped to the FOVs you boxed" if crop else "")
            self._readout.setText(
                f"✓ exported {len(pairs)} mosaic{'s' if len(pairs) != 1 else ''}{note} from "
                f"{', '.join(done)}{t_note}{crop_note} → {Path(pairs[0][0]).parent}")

        w.progress.connect(
            lambda d, n: self._readout.setText(f"● Minerva export · {d}/{n} mosaics"))
        if on_exported is not None:
            w.exported.connect(on_exported)
        w.exported.connect(on_exported_readout)
        w.launched.connect(on_launched)
        w.failed.connect(lambda m: self._readout.setText(f"Minerva export failed: {m}"))
        self._readout.setText(f"● Minerva export · {what}{t_note} …")
        w.start()

    def run_minerva_render(self, pairs, threads=None, open_when_done: bool = True):
        """Render exported ``(ome, story)`` pairs into Minerva exhibits and open the first one.

        The zero-click destination. ``run_minerva_export`` hands the user to Minerva Author, which
        cannot be pointed at a file and so still needs its "Select File" click; this hands them a
        finished, already-coloured Minerva VIEWER instead. Both exist because they are different
        programs: Author edits, ``render.py`` renders. See
        :func:`squidxplorer._minerva.render_exhibit` for the costs, which are real and measured.

        Runs off the GUI thread. A render is minutes, not seconds.
        """
        if not pairs:
            self._readout.setText("export something first - there is nothing to render")
            return
        if getattr(self, "_minerva_render", None) is not None and self._minerva_render.isRunning():
            self._readout.setText("already rendering - let the current render finish first")
            return
        n = len(pairs)
        self._minerva_render = w = _MinervaRenderWorker(pairs, threads=threads)

        def on_rendered(indexes):
            if not indexes:
                return                       # `failed` says why; an empty success is not a message
            note = "" if len(indexes) == n else f" of {n}"
            if open_when_done:
                from squidxplorer._minerva import open_exhibit
                open_exhibit(indexes[0])
            self._readout.setText(
                f"✓ rendered {len(indexes)} Minerva viewer{'s' if len(indexes) != 1 else ''}{note} "
                f"→ {Path(indexes[0]).parent}. {_MINERVA_INTERNET_NOTE}")

        w.progress.connect(
            lambda d, tot: self._readout.setText(f"● Minerva render · {d}/{tot} exhibits"))
        w.rendered.connect(on_rendered)
        # Named in the status line, because render.py runs as a script under a FOREIGN venv: its
        # failure is an exit code plus stderr, and if we do not print it nothing does.
        w.failed.connect(lambda m: self._readout.setText(f"Minerva render failed: {m}"))
        self._readout.setText(
            f"● Minerva render · {n} exhibit{'s' if n != 1 else ''} - this takes minutes …")
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
        # THE VIEW COMBO IS THE SAME JOB, so it is refreshed here rather than at each of the eight
        # call sites: both are surfaces over one `_op_stack`, and a surface that only agrees with
        # the stack at the sites somebody remembered to update is the drift this method exists to
        # prevent. Called first, so it still happens for a window with no Layers tab built yet.
        self._refresh_view_combo()
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

        THE ONE WRITER of the plate's shown layer: every path that changes it — a run start, a
        toggle, a reorder, return-to-raw, opening a computed plate, the post-run resync — writes
        the stack and calls this. Four sites used to call ``set_active_layer`` directly, each
        recomputing the answer the stack already holds, and the direct writes are how the Layers
        tab and the plate came to disagree.

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

    # The loupe-source bookkeeping lives in `_ingest`; these forward so callers are unchanged.
    def _release_loupe_sources(self):
        _ingest.release_loupe_sources(self)

    def _set_loupe_source(self, layer_key, source):
        _ingest.set_loupe_source(self, layer_key, source)

    def _drop_loupe_source(self, layer_key):
        _ingest.drop_loupe_source(self, layer_key)

    def _update_loupe_source(self):
        _ingest.update_loupe_source(self)

    def _enable_operators(self, flag: bool):
        for a in self._op_actions.values():
            a.setEnabled(flag)
        for c in getattr(self, "_op_cards", {}).values():
            c.setEnabled(flag)
        menu = getattr(self, "_declared_menu", None)
        if menu is not None:                       # the uncarded operators gate on the same flag
            menu.setEnabled(flag)

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

    def ingest(self, path: str):
        """Open a raw Squid acquisition folder. The pipeline lives in `_ingest.ingest`."""
        _ingest.ingest(self, path)

    # -- the current region: ONE value, three views ------------------------------------------
    #
    # `_mosaic_region` and `_current_well` are properties over `self._cursor` rather than fields.
    # That is the whole point: an assignment to either cannot create a second copy that drifts
    # out of step with the red frame. Every one of this project's 4+ confirmed instances of that
    # defect was a field somebody forgot to update on one path out of five.

    @property
    def _mosaic_region(self) -> Optional[str]:
        """The region this plate is CURRENT on. Read-only: the cursor decides, this reports."""
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

        The plate's job here is the red ROI frame. The pixels are somebody else's: every open
        ``RegionViewer`` runs its own cursor and fuses its own mosaic, so nothing on this window
        has to wait for a load, and there is nothing here to debounce.
        """
        if self._overview is not None:
            info = self._fov_index.get(region)
            if info is not None:
                self._overview.select(*info["rc"])          # THE RED FRAME

    # NO MOSAIC LOAD, NO SPOT DETECTION, ON THE PLATE (2026-08-06).
    #
    # ``_load_mosaic``, ``_on_mosaic_plane``, ``_on_mosaic_done``, ``_region_frame_done``, the four
    # napari-dims helpers (``_napari_dims`` / ``_napari_z_axis`` / ``_napari_dims_step`` /
    # ``_restore_dims_step``), ``_adopt_centre_view`` and the whole spot-detection chain
    # (``SPOTS_OP``, ``_spot_source_layer``, ``_current_z_index``, ``_on_detect_nuclei``,
    # ``_on_spot_worker_finished``, ``_on_spots_ready``, ``_on_spots_done``) lived here. Every one
    # of them opened by resolving ``self._mosaic_pane``, which has been unconditionally ``None``
    # since 2b8fbc5 (2026-07-23), so every one of them returned at its first line. Measured on the
    # real fixture through ``OpenAcquisition``: 2 of 2 ``_load_mosaic`` calls returned at the guard,
    # 0 ``_MosaicWorker`` objects were built, and the 140 ms debounce QTimer that fed it armed 3
    # times into a slot that could only return. The timer went with them.
    #
    # ``_on_detect_nuclei`` had no call site at all: its only entry point was the pane's own
    # ``detect_button``. The live homes for both jobs are ``RegionViewer._load_mosaic`` and
    # ``RegionViewer._detect_nuclei``, each on a window that actually has napari layers.
    #
    # ``_bind_napari_contrast`` went with them for the same reason at one remove: it swept every
    # open window and re-offered it to ``_bind_window_contrast`` below, and its only caller was
    # ``_on_mosaic_done``. The LIVE binding is ``ViewerManager.windowOpened`` -> that method,
    # connected in ``__init__``, which is the one that has been doing the work.

    def _bind_window_contrast(self, win):
        """Make the plate a SINK of ONE window's napari pane (contrast, eye icons, colormap).

        No new contrast model, and no second sink: this reuses ``_on_detail_contrast``, which
        already lands in the plate's FOLLOW path via ``follow_channel_window`` rather than its
        manual latch. That distinction is the one that matters — treating an owner's autoscale as
        a user gesture is what latched every channel MANUAL on open, killed the plate's running
        auto-contrast from the first frame, and left SCOPE_PER_REGION painting every well under
        one global window while the amber "wells NOT comparable" badge lied over the top.

        ``MosaicLayers.on_user_contrast`` additionally filters out OUR OWN writes (the percentile
        window set at add time, and link propagation), so only a real change of the owner's
        resolved window arrives here.

        Many windows, one plate: whichever window the user last gestured in is the one the plate
        shows. That is deliberate. A window IS a view of a subset of this plate, so its resolved
        window is the honest thing to paint the plate with, and the alternative — the plate
        following only one privileged window — reintroduces the central pane the decentralization
        removed.
        """
        if win is None or self._meta is None:
            return
        pane = getattr(win, "_pane", None)
        mosaic = getattr(pane, "mosaic", None)
        if pane is None or not getattr(pane, "ok", False) or mosaic is None:
            return          # a window that came up without napari has nothing to follow
        bound = getattr(self, "_followed_windows", None)
        if bound is None:
            bound = self._followed_windows = set()
        wid = getattr(win, "window_id", None)
        if wid in bound:
            return          # subscribe ONCE per window: MosaicLayers keeps a list of callbacks

        # THE PLATE NO LONGER FOLLOWS A WINDOW'S LOOK LIVE (2026-08-06).
        #
        # Three subscriptions used to live here -- contrast, eye icons and colormap -- each per
        # CHANNEL, each landing on the plate the instant the user moved it in any window. Plus
        # `_adopt_window_view`, which PULLED the same three at bind time because "an event tells
        # you about a CHANGE; the initial state is not a change".
        #
        # Julio, 2026-08-06: *"we're shelving the interactive contrast synch. What we do is that
        # whichever lookup table we have for the window, we copy it and it reflects on the plate,
        # with whichever channels were turned on on the window. And the plate image shouldn't
        # change unless we paste a LUT. That's the pragmatic fix to this annoying contrast sync
        # logic."*
        #
        # He is right, and the reason is in the docstring the deleted code carried: *"Many windows,
        # one plate: whichever window the user last gestured in is the one the plate shows."* That
        # sentence is the whole defect. "Last gestured in" is not a thing a user tracks, and with
        # several windows open the plate's look was decided by a history with no surface anywhere
        # -- so the plate could go dark, or take one window's window onto another's wells, with
        # nothing on screen having changed. Every fix made it a longer rule with more exceptions,
        # which is the shape of a model that is wrong rather than incomplete.
        #
        # An explicit copy/paste has none of it: it names its source, it happens when asked, and it
        # is already the model the windows use between themselves through the same `_LUT_CLIPBOARD`
        # (`Copy LUTs` / `Paste LUTs`, and the plate's own pair). See `_plate_paste_luts`, which is
        # now the ONE place the plate's look changes.
        #
        # What is deliberately KEPT is `on_user_op` below: which processing LAYER the plate draws
        # is a different quantity from how it is windowed, it has exactly one honest answer at a
        # time, and it is not what this feedback was about.

        # ...and the PROCESSING LAYER. Julio: "after I click an operator layer in our window, the
        # thumbnails don't update." The three sinks above are all per CHANNEL, so picking an
        # operator in a window's layer tree changed that window and nothing else -- the plate went
        # on showing whatever layer the last RUN left active. The op key a window files a result
        # under IS the plate's layer key (`_on_result` passes `_active_op_key` straight through to
        # `deliver_result` -> `add_mosaic`), so no translation is needed and none is invented.
        def _op_sink(op: str, on: bool):
            self._follow_window_layer(str(op), bool(on))

        sub = getattr(mosaic, "on_user_op", None)            # same slot-abort hazard as above
        if callable(sub):
            sub(_op_sink)
        bound.add(wid)

    def _follow_window_layer(self, layer_key: str, on: bool) -> None:
        """A window showed or hid a processing layer: put the plate on the same one.

        Routed through the plate's OWN layer stack rather than straight to
        ``PlateOverview.set_active_layer``, so a window's toggle and the Layers tab's checkbox are
        one gesture with one rule -- the plate renders the topmost ENABLED layer -- instead of two
        surfaces racing to set ``_active``. ``raw`` is the floor and cannot be disabled
        (``OperationStack.toggle``), so hiding raw in a window is a no-op on the plate, which is
        the existing rule stated in one place rather than a new one added here.

        A key the plate has no layer for is IGNORED AND SAID SO. A window can carry operator
        layers this plate never ran (an ROI child re-uses its parent's groups), and quietly
        toggling nothing would be indistinguishable from a broken sink.
        """
        if self._overview is None:
            return
        if layer_key not in [ly.key for ly in self._op_stack.layers()]:
            log.debug("window layer %r has no plate layer to follow (plate has %s)",
                      layer_key, [ly.key for ly in self._op_stack.layers()])
            return
        # A REGION OPERATOR'S LAYER IS REFUSED HERE, BY DECLARATION, NOT SHOWN AS BLACK.
        #
        # Having a layer in `_op_stack` and the PLATE having its pixels are two different facts,
        # and this followed the first while rendering the second. A region operator
        # (`consumes={"fov"}` -- stitch) hands back ONE fused mosaic per well, delivered to the
        # window that asked; there are no per-FOV tiles, so nothing is ever pushed into
        # `PlateOverview._store` and nothing ever will be. Switching `_active` onto it left the
        # plate rendering an empty canvas -- Julio, 2026-08-06: *"Right now, I can see all my
        # windows showing tissue, yet my plate view shows black."*
        #
        # THE DECLARATION, not `has_pixels`. "No pixels yet" and "no pixels ever" are different
        # answers and only the second is a reason to refuse: an operator's tiles stream in over
        # the course of a run, so a has-them-right-now test would refuse the plate its own layer
        # for as long as the first well took, and then leave it refused if the user toggled during
        # that window. `is_region_operator` asks the registry the question that is actually being
        # asked, and never compares a name (`tests/test_operator_declaration.py` fails the build
        # on one).
        #
        # Only the SWITCH-ON is refused. Turning a layer off must always be honoured, or a plate
        # left on a layer could never be brought back to raw.
        from squidxplorer import is_region_operator
        from squidxplorer._operations import operator_name

        if on and is_region_operator(operator_name(layer_key)):
            log.debug("plate stays on %r: %r is a region operator, so its result is one fused "
                      "mosaic delivered to the window that asked and the plate has no per-well "
                      "tiles for it", self._overview._active, layer_key)
            return
        self._op_stack.toggle(layer_key, on)
        self._apply_layers()             # -> set_active_layer + title + loupe source
        self._refresh_layers_tab()       # the tab's checkboxes must not lie about the stack

    def _return_to_raw(self):
        """Stop previewing/processing and restore the raw downsampled view across the whole plate."""
        if self._reader is None or self._overview is None:
            return
        self._stop_worker()
        self._active_op_key = None
        if getattr(self, "_raw_btn", None):
            self._raw_btn.hide()                             # nothing to return from now
        # Showing raw IS "every transform off" under the stack's one rule (topmost enabled
        # renders). Setting the overview to raw directly left the stack claiming a transform the
        # plate was not showing, and the next toggle snapped the plate back onto it.
        for ly in self._op_stack.layers():
            if ly.key != "raw":
                self._op_stack.toggle(ly.key, False)
        self._apply_layers()
        # The raw preview is itself a MOSAIC now (IMA-253), so returning to it restores the
        # acquisition's own boxes rather than clearing them — clearing them broke both the paint
        # (a mosaic redrawn as if it filled its cell) and the double-click FOV hit-test.
        self._overview.set_mosaic_boxes(_mosaic_boxes(self._meta))
        self._update_loupe_source()                          # back to the acquisition's own pixels
        for rc in list(self._overview._status):
            self._overview.set_status(*rc, "empty")
        self._refresh_layers_tab()
        # resume the raw thumbnail fill — the operator run stopped the preview partway, so re-run it to
        # finish downsampling every well's raw tile (idempotent: it just re-renders the raw layer).
        self._stop_preview()
        self._start_preview(self._reader, self._meta, self._order)
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
        # A run that was stopped, or that lost a well to `on_error`, leaves a real-looking
        # plate.ome.zarr with only some of its wells in it. Refuse it by name rather than present a
        # truncated plate as a finished one.
        #
        # THE marker is the one `write_plate` writes, read through `_output.incomplete_reason`.
        # This used to test `base / "INCOMPLETE"` -- a second name for the same fact, whose only
        # writer (`_note_partial_output`) had no callers, so the guard was dead and every stopped
        # save opened as a finished acquisition. The store settles this itself, so the refusal also
        # holds for a plate this window never wrote and for one whose process was killed.
        why = incomplete_reason(zroot)
        if why is not None:
            self._readout.setText(
                f"{base.name} is INCOMPLETE — {why}. Re-run the operator, or delete "
                f"{zroot.name}/.squidxplorer-incomplete to open it anyway.")
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
            self._overview.shutdown()   # both read threads; a deleteLater on a live one aborts
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
        self._overview.activeLayerChanged.connect(lambda _k: self._update_loupe_source())
        self._overview.set_time_point(self.time_point)
        self._active_op_key = "computed"
        if getattr(self, "_raw_btn", None):
            self._raw_btn.hide()                      # a computed plate has no raw to return to
        self._op_stack.reset(); self._op_stack.add("computed", "computed MIP")
        self._apply_layers()
        self._refresh_layers_tab()
        self._drop.hide()
        self._left_l.addWidget(self._overview, 1)
        self._declare_channel_axis(channels, np.uint16)
        self._enable_operators(False)             # no raw data -> operators stay disabled

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
        self._worker = _ComputedPlateWorker(str(zroot), worker_wells, coarse_lvl,
                                           np.uint16, self.time_point)
        self._worker.tileReady.connect(self._on_tile)
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
                     operator_kwargs: Optional[dict] = None,
                     requester: Optional[Any] = None):
        """Run a plane operator (MIP / reference) over the plate, or over a subset of it.

        ``requester`` IS THE COMPLETION CALLBACK, and its absence was the root fault Julio
        reported on 2026-07-29. This method starts a QThread and returns; a ``RegionViewer``
        calling in from its own "Operators for this window" dropdown therefore never heard back,
        could not render the result it had asked for, and showed no ``raw`` / ``flatfield`` /
        ``stitched`` to pick between. Given a requester, this window calls
        ``operator_started(action)`` here, ``deliver_result(op, result, visible=...)`` as each
        region completes, and exactly one of ``operator_done`` / ``operator_failed`` when the
        thread drains -- on success, on failure and on a stop alike, because an action that starts
        and then says nothing is indistinguishable from one still running.

        ``regions=None`` runs the whole plate. A list runs exactly those regions, in that order —
        this is the ONE way to express a subset (the old ``preview_limit=N`` was a prefix-only
        special case of it; the preview spinner now passes ``self._order[:n]`` itself).

        save=False is PREVIEW: compute + stream results into the plate + ndviewer slider, writing
        NOTHING to disk (no folder, no disk-space cost). save=True persists a navigable OME-Zarr;
        combined with a subset it saves just those regions. Tests pass out_parent to skip the dialog.

        """
        # The user asked for a run HERE. Everything below it — scope resolution, the disk estimate,
        # the plate statuses, worker construction — is time they spend waiting, so the first-paint
        # clock starts before all of it rather than at ``worker.start()``.
        t0 = time.perf_counter()
        if self._reader is None or self._overview is None:
            return
        if _run_scope.operator_busy(self._worker, self._retired):
            # NOT ``_busy()``: that also counts a retired RAW PREVIEW, so an operator run could
            # refuse itself over a thread the user never started. See _run_scope.operator_busy.
            self._readout.setText("already processing — let the current run finish first")
            return
        # IMA-226: gate on the ENGINE registry, not on the card table. `_OPERATIONS_BY_KEY[key]`
        # raised a bare KeyError for a registered operator with no card (`reference` then, `spot`
        # and `decon3d` now) and let `minerva` (a card that is not an operator) through to die
        # inside the engine instead.
        # Refuse BY NAME here, in the readout, the same way an unknown region is refused below.
        if key not in runnable_operators():
            self._readout.setText(
                f"'{key}' is not a runnable operator — this viewer can run: "
                f"{', '.join(runnable_operators())}")
            return
        # REGISTERED, and this machine cannot run it: a declared `requires=` package is missing
        # (2026-08-05). Refused in the readout, in the operator's own words, BEFORE the worker
        # starts. Previously the run started, every well raised the same ImportError from a lazy
        # import, `_on_error` recorded each as a per-well skip, and the readout said "done".
        from squidxplorer import operator_available

        _ok, _why = operator_available(key)
        if not _ok:
            self._readout.setText(_why)
            return
        # FLAT-FIELD needs an illumination profile PER CHANNEL. With none at all, the operator
        # raises per field and the plate fills with red x's (Julio: "flatfield shows as x's"). If
        # nothing is installed, AUTO-ESTIMATE one from a spread sample of plate tiles (tilefusion
        # BaSiC, off-thread) and re-run once it lands.
        #
        # The estimate is for the FIRST CHANNEL and is installed for that channel only: the worker
        # reads that channel's tiles and has measured nothing about the others, so a run over them
        # now refuses by name rather than correcting them with this field (which was wrong by up
        # to 1799 counts on the 10x set). The flat-field tab estimates the remaining channels.
        if key == "flatfield":
            import squidxplorer._flatfield as _ff
            if not _ff.active_profiles():
                if getattr(self, "_ff_est_worker", None) is not None and self._ff_est_worker.isRunning():
                    self._readout.setText("flat-field: estimating an illumination profile…")
                    return
                chan = self._meta["channels"][0]["name"]
                w = _FlatfieldWorker(self._reader, self._meta, chan, parent=self)
                w.stage.connect(self._readout.setText)
                w.problem.connect(lambda m: self._readout.setText(f"flat-field estimate failed: {m}"))

                def _profile_ready(profile, k=key, regs=regions, sv=save, op=out_parent, c=chan):
                    _ff.set_profile(profile, channel=c)
                    self._readout.setText("flat-field: profile ready — running.")
                    self.run_operator(k, out_parent=op, regions=regs, save=sv)

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
                           else _run_scope.SCOPE_SELECTION)
            regions, problem = _run_scope.resolve_run_scope(
                scope_value,
                selection=self._selected_regions,
                current_region=self._current_well,
            )
            if problem:
                # A scope the user CHOSE but that has nothing behind it. Say it and stop; widening
                # it to the whole plate would be hours of compute nobody asked for.
                self._readout.setText(problem)
                return
            from_selection = (regions is not None and scope_value == _run_scope.SCOPE_SELECTION)
        if regions is not None:
            # A MAPPING SURVIVES AS A MAPPING. `regions` has three shapes (see
            # `projection.scope_wells`) and `{region: [fov, ...]}` -- the FOV subset an ROI window
            # asks for -- is the one that carries the field lists. `list(regions)` over a dict
            # yields its KEYS, so this line silently widened every ROI run back to whole wells,
            # one call before the worker. It is the same defect `scope_wells` was extracted to fix
            # in the per-FOV loop, surviving one level up: the checks below want NAMES, and taking
            # the names by flattening threw away the rest of the request.
            names = list(regions)                 # keys for a mapping, items for a sequence
            if not isinstance(regions, dict):
                regions = names
            if not names:
                self._readout.setText("empty selection — nothing to run")
                return
            unknown = [r for r in names if r not in self._fov_index]
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
        resolved_target = _run_scope.describe_run_target(regions, total=len(self._order))
        if resolved_target:
            self._readout.setText(resolved_target)
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
        # THE OPERATOR SUPERSEDES THE RAW PREVIEW ONLY WHERE IT ACTUALLY LANDS.
        #
        # Julio, from the running GUI: "mip layer causes incomplete thumbnail ... incomplete
        # render, likely a process getting stuck and losing sync with downsampling."
        #
        # MEASURED (real 10x tissue set, 2 regions): open the plate and run mip on ONE region while
        # the raw preview is still walking the plate, and the other well ends the session with NO
        # thumbnail at all -- `PlateOverview.shown_cells()` returns {(0,0)} out of {(0,0), (0,1)}
        # and `_tiles_by_layer` has no "raw" entry whatsoever. The preview is the per-channel
        # DOWNSAMPLE pass, and `_retire` (see its docstring) disconnects its signals before
        # stopping it, so the tiles already in flight are dropped as well as the reads not yet
        # made. Nothing restarts it: only the return-to-raw path does.
        #
        # A plate-wide run really does supersede the preview -- every well gets an operator tile,
        # so continuing to read the same planes twice is pure cost. A SUBSET run does not: the
        # layer stack's own rule is that "a layer sits OVER the base, it does not replace it"
        # (`_plate_overview.underlay_cells`), and the base is what fills every well outside the
        # run. Stopping the pass that produces the base is how the base ends up missing.
        #
        # So the stop is scoped to the runs that genuinely replace it, which is the same
        # `regions is None` distinction the amber status below already makes.
        if regions is None:
            self._stop_preview()
        if regions is not None:                              # amber only the wells we'll actually run
            for r in regions:
                self._overview.set_status(*self._fov_index[r]["rc"], "processing")
        else:
            self._overview.set_all_status("processing")      # amber across the plate
        layer_key = operator_layer_key(key, None)
        self._active_op_key = layer_key                      # tiles stream into this layer
        # NOTE: _raw_btn is a hidden ORPHAN (never added to a layout since the central pane was
        # removed), so .show() made it POP UP AS A FLOATING WINDOW — Julio: "a 'return to raw view'
        # window pops up. That I don't get." Return-to-raw is handled by the layer stack / plate mode
        # now, so we no longer surface this stray button.
        self._op_stack.add(layer_key, label)                 # push the operator layer onto the stack
        self._apply_layers()                                 # show it: topmost enabled renders
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
        # n_fovs=None = EVERY FOV in each well (IMA-187). Anything else (the historical 1) makes
        # `_boxes` empty in the worker and the plate falls back to one thumbnail per well, which is
        # the whole feature not rendering. The overview then adopts the worker's boxes so a
        # double-click on a mosaic cell resolves the FOV under the cursor instead of always 0.
        run_order = self._order if regions is None else regions
        self._worker = _OperatorWorker(key, self._reader, self._meta, self._fov_index,
                                       str(out_dir) if out_dir else "", regions=regions, save=save,
                                       n_fovs=None, operator_kwargs=operator_kwargs)
        self._overview.set_mosaic_boxes(self._worker.mosaic_boxes)
        # A re-run must not composite on top of the LAST run's pixels: with a mosaic, a run that
        # lands fewer FOVs would otherwise leave the previous run's fields standing in the same
        # cell, blended into the new ones. Drop this layer's store before the first tile arrives.
        # Keyed by the LAYER rather than the bare operator key: `operator_layer_key` is the one
        # place that decides the two are the same thing today.
        self._overview.reset_layer(layer_key)
        dest = f" → {out_dir.name}" if save else " (preview — not saved)"
        self._worker.tileReady.connect(self._on_tile)
        self._worker.resultReady.connect(self._on_result)
        self._worker.progress.connect(self._on_progress)
        self._worker.runProgress.connect(self._on_unit_progress)
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
        # Whether the plate this run is writing ended up whole is NOT tracked here. `write_plate`
        # settles it on the store itself (`_output.INCOMPLETE_MARKER`, kept unless every field
        # this run owed landed) as the last act of the write, which is the only place that knows
        # the answer -- a GUI flag set from `finished_ok` cannot see a well the engine skipped, and
        # cannot be set at all by a process that was killed. See `_open_computed`.
        # QThread.finished (not finished_ok): it fires for a FAILED or STOPPED run too, and a tab
        # switch deferred during any of those still has to be delivered. _retire only disconnects
        # the worker's own signals, so this survives a stop.
        # Connected bare, so the slot is entered with ``worker=None`` — "the RUN's own thread
        # exited", which is the one drain allowed to close the run's books. Only ``_retire`` names
        # the thread, because only there can it be the superseded raw preview.
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
        # THE RUN'S BOOKS, in one object. The requester is genuinely held (it was once dropped on
        # the floor, so ``operator_started`` / ``operator_progress`` / ``operator_done`` /
        # ``operator_failed`` on the asking window were never called); it is cleared in
        # ``_on_run_drained``, which fires on ok / failed / STOPPED alike.
        #
        # The address is `next(iter(...))`, NOT `regions[0]`. `regions` has three shapes and one
        # of them is the mapping `{region: [fov, ...]}` an ROI window sends (see
        # `projection.scope_wells`), where integer indexing is a KeyError on the key `0`. `len()`
        # and iteration are the operations that mean the same thing for both shapes.
        self._run = OperatorRun(
            key=key, layer_key=layer_key, label=label,
            action=f"{_action_label(key, operator_kwargs)} · {scope}",
            dest=dest,
            address=(Extent(region_id=next(iter(regions)))
                     if regions is not None and len(regions) == 1 else None),
            requester=requester,
            # Is this run only PART of each well? A mapping means explicit fields (see `_on_tile`).
            is_partial=isinstance(regions, dict),
            t0=t0)
        self._tell_requester(requester, "operator_started", label)
        self.log.started(self._run.action, address=self._run.address)
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

    def _start_preview(self, reader, meta, order):
        """Start the raw preview over *order* (`_ingest.start_preview` is the one builder).

        This forwarder is the ONE place the plate's timepoint reaches the pixels. Every entry
        path — the first ingest, the tab re-scope, and `_return_to_raw`, which is what a
        timepoint change calls — must preview the frame the bar is showing, and reading
        `self.time_point` here rather than at three call sites makes that true by construction.
        """
        return _ingest.start_preview(self, reader, meta, order, time_point=self.time_point)

    def _on_tile(self, ri, ci, well_id, tile, box=None):
        """A field landed. ``box`` is None for the single-tile producers (_ComputedPlateWorker emits
        a 4-arg signal, which PyQt matches against this default) and a sub-cell box for a mosaic."""
        if self._overview is None:
            return
        # A PARTIAL-REGION RESULT DOES NOT BECOME THE REGION'S THUMBNAIL. Julio, 2026-08-06:
        # *"Just make sure that the thumbnail of the whole region stays after I stitch (replacing
        # it by the ROI's thumbnail)."*
        #
        # A REGION operator emits ONE array per well and the plate pastes it as the whole cell, so
        # an ROI-scoped stitch -- four fields of twenty-seven -- overwrote the well's thumbnail
        # with a picture of one corner of it. That is a wrong answer about the sample, not a rough
        # one: the plate is the NAVIGATOR, and a cell that shows a corner while claiming to be the
        # well makes every other well unfindable by eye.
        #
        # The run knows: `regions` arrived as the mapping `{region: [fov, ...]}` (see
        # `projection.scope_wells`), which is precisely the statement "this is part of a well".
        # The window that asked still gets the layer -- `deliver_result` places it at its own
        # bbox_um inside the region, which is where it belongs.
        if self._run is not None and self._run.is_partial and self._worker is not None \
                and not getattr(self._worker, "IS_PREVIEW", False):
            log.debug("plate keeps %s's whole-region thumbnail: this run covers part of it",
                      well_id)
            return
        layer = self._active_op_key or "raw"
        self._overview.add_tile(ri, ci, well_id, tile, layer=layer, box=box)
        # FIRST PAINT stops here, one line after the tile is actually on the plate. Reported for
        # the OPERATOR run only: this slot also serves the raw preview and the reopened-plate
        # worker, and neither is the wait being measured. The recorder keeps the first report and
        # drops the rest, so this needs no "have I already done this" flag of its own.
        w = self._worker
        if self._run is not None and w is not None and not getattr(w, "IS_PREVIEW", False):
            report = getattr(w, "report_first_paint", None)
            if report is not None:
                report(time.perf_counter() - self._run.t0)
        self._overview.set_status(ri, ci, "done")           # blue
        src = self._loupe_sources.get(layer)                 # this well is now on disk -> loupe-able
        if isinstance(src, _ZarrLoupeSource):
            src.mark_written(well_id)

    def _on_result(self, region, fov, planes):
        """An operator's FULL-RESOLUTION pixels -> a toggleable napari LAYER GROUP (Defect 3).

        Julio: "what if we want to see stitched AND deconvolved AND background subed. That's
        why we need the toggles." Before this, no operator's output reached any napari at
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

        WHICH REGIONS ARE ACCUMULATED changed on 2026-07-29. It used to be ``_mosaic_region``
        alone, the plate's own central pane, which is permanently ``None`` since the
        decentralization -- so this slot returned at its first line for every result in the app
        and no window could ever show an operator layer. It is now every region SOME surface is
        showing, which is ``_result_regions``.
        """
        op = self._active_op_key
        run = self._run
        if not op or run is None:
            return
        if str(region) not in self._result_regions():
            return                              # nobody is looking at it -- see the docstring
        accs = run.accs
        acc = accs.get(str(region))
        if acc is None or acc.op != op:
            from squidxplorer import is_region_operator
            from squidxplorer._op_result import RegionResultAccumulator

            acc = RegionResultAccumulator(
                op, region, self._meta, [c["name"] for c in self._meta["channels"]],
                # The REGISTRY name, not the layer key: a namespaced layer key ("stitch@…") is
                # in no registry, so asking with it accumulated a stitch as if it were a per-FOV
                # operator. `operator_name` strips the namespace; see `operator_layer_key`.
                region_operator=is_region_operator(operator_name(op)),
            )
            accs[str(region)] = acc
        try:
            # asanyarray: a region operator's pixels arrive as a `PlacedArray` carrying the
            # `Placement` that fused them, and `asarray` would return a base ndarray and drop it
            # one call before the accumulator reads it. See `RegionResultAccumulator._bbox`.
            acc.add(int(fov), np.asanyarray(planes))
        except ValueError as exc:
            # NO SILENT FAILURES: a result that cannot be placed is said out loud. It must not
            # abort the run -- the pixels are still written and still on the slider.
            self._readout.setText(f"result not shown as a layer: {exc}")
            accs.pop(str(region), None)
            return
        if not acc.complete():
            return
        accs.pop(str(region), None)
        try:
            result = acc.result()
        except ValueError as exc:
            self._readout.setText(f"result not shown as a layer: {exc}")
            return
        self._deliver_operator_result(op, result)

    def _settle_stranded_results(self) -> int:
        """Settle the run's books at the drain: :meth:`squidxplorer._run.OperatorRun.settle_stranded`
        owns the logic and the story; this forwards, says the refusals out loud, and logs them."""
        run = self._run
        if run is None:
            return 0
        stranded = run.settle_stranded(self._deliver_operator_result)
        for line in stranded:
            log.warning("%s", line)
        if stranded:
            self._run_readout("  ".join(stranded))
        return len(stranded)

    def _result_regions(self) -> set:
        """Every region a surface is SHOWING right now: one entry per open window.

        This is the memory bound on the run's accumulators and on the layers themselves. Holding
        full-resolution mosaics for every well of a plate run would be gigabytes of layers nobody
        can look at, so a result for a region no surface is showing is dropped rather than
        accumulated -- the same rule the raw path follows, for the same reason. The honest bound is
        no longer "one region" but "one per open window", because that is how many regions the user
        can actually be looking at.

        The plate window itself is NOT a surface and contributes nothing here. It used to add
        ``_mosaic_region`` when its own pane was ok, and that pane was never ok.
        """
        regions: set = set()
        mgr = getattr(self, "_viewer_manager", None)
        for win in (mgr.windows if mgr is not None else []):
            try:
                here = win.current_region()
            except Exception:                   # noqa: BLE001 - a window mid-teardown has none
                continue
            if here:
                regions.add(str(here))
        return regions

    def _deliver_operator_result(self, op: str, result) -> None:
        """THE COMPLETION PATH: one region's finished result, to the surfaces that asked for it.

        Julio, 2026-07-29: "the layers such as 'raw', 'flatfield', 'stitched', in the window that I
        decided to compute, are simply not available when I run an operator on the window." They
        were not available because the result never left this class: it went to
        ``_add_result_layers``, which paints the plate's own central pane, and that pane has been
        ``None`` since the decentralization. So the run happened, the pixels were written, and the
        window that asked for them gained nothing.

        One declaration, several sinks. The ``Result`` is built once -- by the accumulator, which
        is the only place that knows what it fused -- and handed to every open window, so no sink
        re-derives what the result is. ``_add_result_layers`` -- the branch that painted the
        plate's own pane -- was deleted on 2026-08-06 along with the pane it needed; the windows
        are the sinks, and they always were.
        """
        # FILE IT IN THE CROSS-WINDOW RESULT CACHE, whose docstring has promised exactly this
        # since it was written and which had no production call site at all: "two windows over the
        # same node running the same chain at the same version hit the SAME entry: results
        # cross-propagate for free". The propagation below only reaches windows that are open NOW;
        # a window opened a minute later saw nothing and the only way to get the layer was to run
        # the operator again. `ViewerManager._replay_cached_results` is the reader.
        #
        # One store for the whole process. Both windows share one reader object and one
        # interpreter (DEFAULT_MAX_GUI=1 with an flock refuses a second process), so this is
        # same-process reuse, not IPC, and nothing has to be serialised.
        from squidxplorer._recipe import acquisition_version, cache_operator_result

        cache_operator_result(op, result, acquisition_version(self._reader))
        added = self._deliver_to_views(op, result)
        if added:
            self._readout.setText(
                f"{op} · {result.region_id} — {added} layer(s) added; toggle it against raw in "
                f"the mosaic layers panel")
        else:
            # NO SILENT FAILURES: a computed result with nowhere to land is not a success.
            self._readout.setText(
                f"{op} · {result.region_id}: computed, but no open view is showing "
                f"{result.region_id}, so there was nowhere to put the layer")

    def _deliver_to_views(self, op: str, result) -> int:
        """Propagate one result to every open window, VISIBLE only in the window that ASKED.

        Julio's second sentence: "even if we have a cache of operations, when it propagates to
        other windows, it adds a layer, but it doesn't toggle it." The rule this settles on is one
        sentence long: **the window that asked shows the result; every other window gains it dark.**

        Asking is the consent, so the requester gets ``visible=True`` -- and it keeps it even if
        the user has clicked another window since, because a feature that depends on which window
        happens to be in front when a thread finishes is a race, not a behaviour. Every other
        window did not ask, so it gets ``visible=False``: it gains the layer, so the layer is there
        to toggle, and what someone is looking at does not change under them. That is strictly
        stronger than "unfocused windows get it dark", which is the requirement.
        """
        mgr = getattr(self, "_viewer_manager", None)
        if mgr is None:
            return 0
        requester = self._run.requester if self._run is not None else None
        added = 0
        for win in mgr.windows:
            deliver = getattr(win, "deliver_result", None)
            if deliver is None:
                continue
            try:
                added += int(deliver(op, result, visible=(win is requester)) or 0)
            except Exception as exc:            # noqa: BLE001 - one window's failure is its own
                log.warning("view %s could not take the %s result for %s: %s",
                            getattr(win, "window_id", "?"), op, result.region_id, exc)
        return added

    def _run_readout(self, text: str):
        """Set the run's status line. Kept as its own method because every run-status caller goes
        through it; it used to re-append a sticky push-failure suffix, and the only producer of one
        was the array-viewer feed that no window has had since ndviewer_light was removed."""
        self._readout_base = text
        self._readout.setText(text)

    def _on_failed(self, msg):
        # Remember WHY, for the requester's ``operator_failed`` line. ``_on_run_drained`` fires on
        # QThread.finished and cannot see the exception; without this the asking window would be
        # told "produced nothing" for a run that named its own cause here. The reopened-plate
        # worker shares this slot and has no run to file the reason under.
        if self._run is not None:
            self._run.error = str(msg)
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
        """Show the current selection in the Selection bar ("run on selected wells").

        A partly-boxed well is NAMED as one — ``A1 (4/27 FOVs)`` — because a Shift+Alt box inside
        a mosaic changes what an export writes, and a label that reads the same either way makes
        a crop indistinguishable from a whole region until the file lands.
        """
        lbl = getattr(self, "_selection_label", None)
        if lbl is None:
            return
        sel = self._selected_regions
        ov = getattr(self, "_overview", None)
        subsets = ov.fov_subsets() if ov is not None else {}
        per = (self._meta or {}).get("fovs_per_region", {})

        def name(region):
            fovs = subsets.get(region)
            if not fovs:
                return str(region)
            return f"{region} ({len(fovs)}/{len(per.get(region) or fovs)} FOVs)"

        if not sel:
            lbl.setText("none — click wells, or Select all")
        elif len(sel) <= 6:
            lbl.setText(f"{', '.join(name(r) for r in sel)}  ({len(sel)})")
        else:
            lbl.setText(
                f"{', '.join(name(r) for r in sel[:6])}, +{len(sel) - 6}  ({len(sel)})")

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
        from squidxplorer._region_viewer import _LUT_CLIPBOARD
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
        """Apply the shared LUT clipboard to the plate: contrast, colour, AND channel on/off.

        THIS IS THE ONLY WAY THE PLATE'S LOOK CHANGES (2026-08-06). Julio: *"whichever lookup table
        we have for the window, we copy it and it reflects on the plate, with whichever channels
        were turned on on the window. And the plate image shouldn't change unless we paste a LUT.
        That's the pragmatic fix to this annoying contrast sync logic."*

        The live subscriptions this replaced are described in ``_bind_window_contrast``. The short
        version: the plate followed whichever window the user last gestured in, per channel, for
        three quantities at once, and "last gestured in" is not a thing a user tracks. With several
        windows open, the plate's look was decided by a history nobody could see, and the only way
        to find out what it would do was to do it.

        A copy/paste pair has none of that. It is explicit, it names its source, it happens when
        asked, and it is the model the windows already use between themselves through the same
        ``_LUT_CLIPBOARD``. What was a rule with exceptions becomes a gesture with a result.

        VISIBILITY TRAVELS WITH THE WINDOW, not as a separate step: ``lut["on"]`` is what the
        source window had lit. Written through ``set_channel_visible``, so the plate's
        never-go-black floor still refuses the last lit channel -- pasting from a window with
        everything switched off must not empty the navigator.
        """
        from squidxplorer._region_viewer import _LUT_CLIPBOARD
        ov = self._overview
        if not _LUT_CLIPBOARD:
            self._readout.setText("no copied LUTs yet — copy from a window or the plate first.")
            return
        if ov is None:
            return
        applied = channels = 0
        skipped: list = []
        for i, name in enumerate(self._plate_channels()):
            lut = _LUT_CLIPBOARD.get(name)
            if not lut:
                continue
            if lut.get("clim") is not None:
                lo, hi = float(lut["clim"][0]), float(lut["clim"][1])
                # A DEGENERATE WINDOW IS NOT PASTED. Julio, 2026-08-06: *"when I copy luts, my
                # thumbnails saturate the yellow channel."*
                #
                # `_contrast.auto_contrast` returns hi <= lo for a BLANK channel, deliberately --
                # `add_mosaic` states the rule in full: "A degenerate window is passed through, NOT
                # widened, because widening it to (lo, lo+1) renders a blank channel as full white,
                # i.e. as signal." The plate's `_RunningContrast.set_manual` does exactly that
                # widening (`max(hi, lo + 1)`), because a running histogram must never divide by
                # zero. Both are right on their own side of the seam; what was missing is that a
                # channel with no signal in the WINDOW must not be latched onto the plate at all.
                # On the 10x set 561 is the empty channel -- its thumbnail is black in the window --
                # so a paste handed the plate a one-count window and every tile went full yellow.
                #
                # Skipped and SAID, not silently widened: the user asked for a paste and one
                # channel did not take it.
                if hi <= lo:
                    skipped.append(name)
                    continue
                try:
                    ov.set_channel_window(i, lo, hi)
                    applied += 1
                except Exception:                        # noqa: BLE001 - one bad channel is skipped
                    pass
            if lut.get("rgb") is not None:
                try:
                    ov.set_channel_color(i, lut["rgb"])
                except Exception:                        # noqa: BLE001
                    pass
            if lut.get("on") is not None:
                try:
                    ov.set_channel_visible(i, bool(lut["on"]))
                    channels += 1
                except Exception:                        # noqa: BLE001
                    pass
        self._readout.setText(
            f"pasted LUTs onto {applied} plate channel(s)"
            + (f", and {channels} channel on/off state(s)" if channels else "")
            + (f" — {', '.join(skipped)} has no signal in that window, so its contrast was left "
               f"alone rather than pasted as a one-count window." if skipped else "."))

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
        from squidxplorer._region_viewer import View

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

    def _print_open_views_target(self, action: str) -> "Optional[list]":
        """Say WHICH windows an "Open views" run is aimed at, and return what it resolved to.

        Julio, 2026-08-03: "it has to print which windows and subsets thereof are selected." The
        selector only names the RULE — "Open views" — and the rule is not the answer: which windows
        are open, what each holds, and how the overlap between them collapses are three facts a user
        cannot infer from three words in a combo box.

        Called when the user PICKS the target AND again when they press Run. Printing only at launch
        would be printing it after the decision, and a plate-scale run is minutes of compute.

        The block goes to the log console via this window's ``ViewLog`` — the existing addressed
        channel, which is monospace, scrollable and copyable, and already interleaves every stream.
        Only the headline goes to the status line, which is one line high.

        Returns ``None`` when nothing would run, having said why.
        """
        views = self._open_view_targets()
        block = _run_scope.describe_view_target(views, action=action)
        if block is None:
            self._readout.setText(
                f"Run on open views: {len(views)} open window(s) hold no regions between them — "
                f"nothing to run." if views else
                "Run on open views: no windows are open — open some first.")
            return None
        self.log.info("%s", block)
        self._readout.setText(block.splitlines()[0])
        return _run_scope.distinct_view_regions(views)

    def _open_view_targets(self) -> list:
        """Every open window as a View — the target set BEFORE it is flattened to regions.

        The sibling ``_open_views_regions`` throws the windows away, which is correct for the run
        and wrong for the print: an operator UI that cannot name which windows it is about to run
        on makes the user infer it. This is what ``_run_scope.describe_view_target`` reads.

        ``_open_view_targets`` and not ``_open_views``, which is already taken by the navigator
        WIDGET (``self._open_views = OpenViewList(...)``). Same three words, two different things,
        and the shorter name belongs to the one that shipped first.
        """
        mgr = getattr(self, "_viewer_manager", None)
        return list(mgr.views()) if mgr is not None else []

    def _open_views_regions(self) -> list:
        """The union of regions held by the open independent windows, in first-seen order — the
        iteration set for an operator run 'on open views' (the decentralized bulk target).

        Derived from the SAME Views the print reads, through the SAME flattener
        (``_run_scope.distinct_view_regions``). That is load-bearing rather than tidy: if the printed
        distinct-region count were computed by a second dedup, the block could disagree with what
        actually runs, which is worse than not printing it at all.
        """
        return _run_scope.distinct_view_regions(self._open_view_targets())

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
        """The current selection as (region, fov) pairs — the payload IMA-205 will consume.

        A region the user boxed only PART of contributes only those fields. That is the whole of
        "run on a subset of the acquisition": the engine has always cropped to the FOVs it is
        handed (``run_plate(regions={region: [fov, ...]})`` derives the mosaic canvas from
        those positions alone), and this method was the place that threw the finer answer away by
        expanding every selected well to all of its fields before anyone downstream could see it.

        ``PlateOverview.fov_subsets`` is the only source of a subset and it publishes STRICT ones
        only, so the fallback below is not a guess: a region absent from it is selected whole.
        ``or [0]`` still covers an acquisition whose metadata lists no FOVs for a region at all.
        """
        per = (self._meta or {}).get("fovs_per_region", {})
        ov = getattr(self, "_overview", None)
        subsets = ov.fov_subsets() if ov is not None else {}
        out = []
        for r in self._selected_regions:
            fovs = subsets.get(r) or per.get(r) or [0]
            out.extend((r, f) for f in fovs)
        return out

    def _on_hover(self, text: str):
        # BOTTOM-LEFT plate title bar: "<acq>  ·  <mode>" (mode = raw / the operator that processed it),
        # plus the hovered well when the cursor is over the plate.
        base = f"{self._acq_name or 'well plate'}   ·   {self._plate_mode}"
        self._plate_title.setText(f"{base}   ·   {text}" if text else base)

    def activate_well(self, well_id: str, fov_index: int):
        """Double-click a well -> open ONE independent window on that region.

        The single-region case of the shift-drag gesture. It used to have a second half — register
        the well's raw z-planes into an embedded ndviewer_light and move its FOV slider — and that
        half has been unreachable since the module was deleted on 2026-07-30 (`self._detail` was
        assigned `None` once and never anything else), so it is gone with it.
        """
        if well_id not in self._fov_index:
            return
        self._current_fov = fov_index                  # the FOV ON SCREEN (IMA-250 (b))
        # ONE move. The cursor drives the red frame and everything else that reads it, together,
        # so they cannot disagree. This used to be three statements on three different code paths,
        # and it returned before ANY of them ran.
        try:
            self._cursor.activate(well_id)
        except KeyError:
            self._readout.setText(f"{well_id} is not in the current region order")
            return
        win = self._viewer_manager.open([well_id])
        if win is None:
            self._readout.setText("Open an acquisition before opening a view.")

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

    # NO reference-plane chain here. `_focus_reference_plane`, `_on_focus_problem`,
    # `_on_reference_plane` and `_set_z_index` were removed on 2026-07-29: the only entry
    # point into all four was the orphan `_focus_btn`'s clicked signal (see __init__), and
    # the reference plane moved onto each window's own z-slider in d07db43. `_FocusWorker`
    # itself STAYS: `RegionViewer._focus_reference_plane` imports it by name.

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
            # WHICH thread exited is passed on, because the slot needs it: a retired RAW PREVIEW
            # exiting mid-run must deliver a deferred re-sync WITHOUT closing the run's books (see
            # _on_run_drained). Hooking them all and then treating them all alike is what made the
            # run's outcome depend on when the superseded preview happened to stop.
            w.finished.connect(lambda w=w: self._on_run_drained(w))

    def _stop_flatfield(self):
        """Retire both flat-field estimators. Neither was joined anywhere before 2026-08-06.

        Two slots — ``_flatfield_worker`` (the toolbar's "estimate from plate") and
        ``_ff_est_worker`` (the auto-estimate a `flatfield` run does for you) — both parented to
        this window and both absent from `closeEvent`. A BaSiC solve is seconds-to-minutes
        (`_FlatfieldWorker`'s own docstring), so closing the plate during one destroyed a running,
        parented QThread: SIGABRT. They can go through `_retire` now only because
        `_FlatfieldWorker` grew a `stop()`; `_retire` calls it.
        """
        for slot in ("_flatfield_worker", "_ff_est_worker"):
            self._retire(getattr(self, slot, None))
            setattr(self, slot, None)

    def _stop_worker(self):
        self._retire(self._worker)
        self._worker = None

    def _stop_preview(self):
        _ingest.stop_preview(self)

    def _stop_minerva(self):
        self._retire(self._minerva)
        self._minerva = None
        # The render worker is retired here too and not on its own line: it is the same feature and
        # it is the LONGER of the two (a measured 132 s for one real region against an at-most 90 s
        # port poll), so a close that abandons the launch wait but not the render would still hold
        # the window for two minutes. Its stop() terminates the child render.py process.
        self._retire(getattr(self, "_minerva_render", None))
        self._minerva_render = None

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

    # NO MOSAIC WORKER TO JOIN. ``_stop_mosaic_worker`` lived here: it joined ``self._mosaic_worker``
    # (written only by the deleted ``_load_mosaic``) plus one ``tab.mosaic_worker`` per exploration
    # tab, and the exploration pane went in ae6217e. Nothing in this process assigns either
    # attribute any more, so it joined two empty slots on every ingest and every close. The fuse
    # this window can still cause is ``RegionViewer``'s, which each window joins in its own
    # ``closeEvent``. Do not add a plate-owned QThread back without a join: an unjoined QThread
    # parented to a window aborts the process when Qt destroys it.

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
                print(f"squidxplorer-view: {exc}", file=sys.stderr)
                self._gui_slot = None
                QTimer.singleShot(0, self.close)   # unwind out of showEvent first, then close
                return
        super().showEvent(e)

    #: Session flag behind the close-all confirmation. True (the default) = ask. Class-level on
    #: purpose: "don't show me this again" is about the application, not one window. It does not
    #: reach disk (_prefs went with the 2026-08-13 kill list), so the dialog returns next launch.
    _warn_close_all = True

    def _open_view_count(self) -> int:
        """How many region windows are open right now. 0 when there is no manager."""
        mgr = getattr(self, "_viewer_manager", None)
        try:
            return 0 if mgr is None else len(mgr.windows)
        except Exception:                        # noqa: BLE001 - a torn-down manager: none open
            return 0

    def _confirm_close_all(self, n: int) -> bool:
        """Ask before the plate takes *n* region windows down with it. True = go ahead.

        Julio, 2026-08-06: *"closing the plate window should close all the other windows make sure
        that you pop up the warning, with the 'don't show me this again'."*

        Both halves matter and they pull against each other, which is why this is a dialog rather
        than either extreme. Leaving the windows open was the old behaviour and it is a trap: Qt
        quits when the LAST top-level closes, and a `RegionViewer` is a top-level, so closing the
        plate left a headless remainder holding the single-instance flock -- the next launch was
        then refused by a process with no plate to find. But closing several windows is not
        undoable either, and a window may be mid-run.

        Never shown when there is nothing to confirm (no open views) or under the test harness,
        where a modal dialog would hang the suite with no one to dismiss it.
        """
        from qtpy.QtWidgets import QApplication, QCheckBox, QMessageBox

        if n <= 0:
            return True
        app = QApplication.instance()
        if app is not None and app.property("_squidxplorer_test"):
            return True
        if not PlateWindow._warn_close_all:
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Close SquidXplorer")
        box.setText(f"Closing the plate will also close {n} open view window(s).")
        box.setInformativeText(
            "The plate is what the views belong to, so they go with it and the application quits.\n\n"
            "Anything already written to disk stays; a run still in flight is stopped.")
        box.setStandardButtons(QMessageBox.Cancel | QMessageBox.Close)
        box.setDefaultButton(QMessageBox.Cancel)
        never = QCheckBox("Don't show me this again")
        box.setCheckBox(never)
        if box.exec() != QMessageBox.Close:
            return False
        if never.isChecked():
            PlateWindow._warn_close_all = False
        return True

    def closeEvent(self, e):
        # THE PLATE TAKES ITS VIEWS WITH IT (2026-08-06), after confirming.
        #
        # `RegionViewer` is a top-level window and nothing sets `quitOnLastWindowClosed`, so Qt's
        # default applies: the process lives until the LAST top-level closes. Closing the plate
        # therefore used to leave a plateless remainder still holding the single-instance flock,
        # and the next launch was refused by a process the user could no longer see the plate of.
        # A view is a view OF this plate -- it reads through the plate's reader and follows the
        # plate's runs -- so it cannot outlive it.
        #
        # FIRST, before any teardown below: `_confirm_close_all` can cancel, and cancelling has to
        # leave the window exactly as it was. Every line under this point retires threads and
        # uninstalls the log bus, none of which is reversible.
        views = self._open_view_count()
        if not self._confirm_close_all(views):
            e.ignore()
            return
        if views:
            mgr = self._viewer_manager
            for wid in [int(w.window_id) for w in mgr.windows]:
                try:
                    mgr.close(wid)               # a no-op for an id a parent already took with it
                except Exception as exc:         # noqa: BLE001 - one view must not block the quit
                    log.warning("view %s did not close with the plate: %s: %s",
                                wid, type(exc).__name__, exc)
        # NO REGION DEBOUNCE TO DISARM. A single-shot QTimer used to be armed here for 140 ms by
        # `_on_region_changed` and stopped at this point, because a pending one fires into a
        # torn-down window (measured: a segfault a window later). Both the timer and the
        # `_load_mosaic` it fired are gone; the hazard is gone with them. Any timer added to this
        # window in future must be stopped HERE, the way `PlateOverview.hideEvent` stops its own.
        release_gui_slot(getattr(self, "_gui_slot", None))   # let the next window open
        self._gui_slot = None
        self._stop_worker()          # stop the run cleanly; nothing on disk to clean up (no cache)
        self._stop_preview()
        self._stop_flatfield()       # BEFORE _join_retired, so its threads are in that list
        self._join_retired()         # everything _retire deferred
        self._stop_minerva()         # files already written stay; only the launch poll is abandoned
        ov = getattr(self, "_overview", None)
        if ov is not None:
            ov.clear_tile_source()   # joins the tile fetcher; a live QThread blocks a clean exit
        # The console float first, and NOT through the loop below: that loop disposes each float's
        # content, and disposing the log panel would delete a widget `panel.stop()` is about to be
        # called on. Re-docking returns it to `_right_col`, where it is destroyed with the window
        # like any other child. It is a no-op when the log is not floated.
        #
        # NOTE for to-do/2026-08-03-window-lifetime-design.md, which has not decided whether child
        # windows outlive the plate: floats are ALREADY swept here, unlike RegionViewers, so the log
        # float lands on the safe side today by construction. If that document later chooses
        # "windows are peers that outlive the plate", the log float must be explicitly excluded —
        # the panel is a live sink on the process-wide root logger and the bus is uninstalled a few
        # lines below, so a surviving log window would be a console attached to nothing.
        self._redock_log()
        for key in list(self._floating):   # floated tabs are top-levels of their own — Qt won't
            win = self._floating.pop(key)  # close them for us, and each may hold a live shell
            w = win.take_content()
            if w is not None:
                self._dispose_tab_widget(w)
            win.close()
        gallery = getattr(self, "_gallery", None)
        if gallery is not None:
            # A gallery is a top-level of its own with a LIVE QThread fusing cells into it. Left
            # open it would be a window drawing into a closed plate's reader; left running it would
            # be a QThread at teardown, which is the one thing this method exists to prevent. It is
            # swept here with the log float rather than treated as a peer window: see the note
            # above for what to-do/2026-08-03-window-lifetime-design.md may later change.
            self._gallery = None
            gallery.close()
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
    """(peak_MB, current_MB_or_None). See squidxplorer._footprint, which owns the platform branches.

    This used to read the peak from ``resource`` alone, which is POSIX-only, so the footprint line
    printed ``peak 0 MB`` forever on Windows -- the platform v1 ships to. The peak now comes from
    whichever high-water mark the platform keeps (``ru_maxrss`` or ``PeakWorkingSetSize``)."""
    from squidxplorer._footprint import rss_mb

    return rss_mb()


def _install_footprint_monitor(app, win):
    """Track the process memory footprint and PRINT THE PEAK when the GUI closes or crashes.

    A light QTimer prints a live line every few seconds so you can watch the footprint as you drive
    the GUI (open a plate, run MIP, scrub FOVs); the peak is the OS high-water mark, so the final
    number is exact regardless of sampling. Wired to app-quit (normal close), atexit, and the
    excepthook (crash) — so a peak is always reported. Every platform: the peak comes from
    squidxplorer._footprint, which reads whichever high-water mark the OS keeps."""
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
    """How many GUI windows may be open at once. ``SQUIDXPLORER_MAX_GUI`` overrides."""
    raw = os.environ.get("SQUIDXPLORER_MAX_GUI")
    if not raw:
        return DEFAULT_MAX_GUI
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_GUI


def _gui_lock_dir() -> Path:
    d = Path(os.environ.get("SQUIDXPLORER_GUI_LOCK_DIR")
             or (Path.home() / ".cache" / "squidxplorer"))
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
        f"raise the cap with SQUIDXPLORER_MAX_GUI=<n>. Lock dir: {lock_dir}"
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

    # SHARED GL CONTEXTS. Qt6 migration, step 3. Every open window owns its own napari canvas, so
    # this process creates N OpenGL contexts rather than one. Without AA_ShareOpenGLContexts they
    # cannot share textures, and the symptom is not a clean error: it is a canvas that renders
    # black, or renders once and then stops, in whichever window happened to be created second.
    # That is the decentralised-windows architecture's own failure mode, so it matters more here
    # than it would in a single-canvas app.
    #
    # Set here rather than in `main` for the same reason as the two above: once a QApplication
    # exists this is a SILENT no-op. It applies under Qt5 as well, so it is not gated on the
    # binding and does not become dead code when PyQt5 is eventually dropped.
    share = getattr(Qt, "AA_ShareOpenGLContexts", None)
    if share is not None:
        QApplication.setAttribute(share, True)

    # FRACTIONAL SCALE FACTORS. AA_EnableHighDpiScaling above only says "scale"; it does not say
    # BY WHAT. Qt5's default rounding policy is Round, so a display at 125%, 150% or 175% -- the
    # three settings Windows actually ships and the ones a 4K laptop panel defaults to -- is
    # snapped to 100% or 200%. At 150% snapped down to 100% every window comes up two-thirds of
    # its intended size with the type to match, which is Spencer's "rendered at about 3x2 inches"
    # on a 4K monitor: not a missing high-DPI fix, a rounded one. macOS never showed it because
    # Retina is exactly 2x and rounds to itself.
    #
    # PassThrough honours the fraction as-is. It is the Qt6 DEFAULT, so this line only changes
    # Qt5 behaviour and is a no-op once PyQt5 is dropped -- which is also why the Qt6 build
    # Spencer tested looked correct without anyone fixing anything.
    policy = getattr(Qt, "HighDpiScaleFactorRoundingPolicy", None)
    setter = getattr(QApplication, "setHighDpiScaleFactorRoundingPolicy", None)
    if policy is not None and setter is not None:
        setter(policy.PassThrough)


def _startup_splash(app):
    """A small "starting" window, shown before the slow constructor runs. None under tests.

    Returns the splash so the caller can `finish(win)` it; returning None rather than a dummy
    keeps the caller's `if splash is not None` honest about the headless case.

    Deliberately plain: a QPixmap-less QSplashScreen with a styled message. No logo file, because
    an asset that fails to load produces a blank rectangle, which looks exactly like the hang this
    is here to rule out.
    """
    try:
        if app.property("_squidxplorer_test") or QApplication.platformName() in ("offscreen", "minimal"):
            return None                   # nothing to show, and nobody to see it

        from qtpy.QtWidgets import QSplashScreen

        splash = QSplashScreen()
        splash.setStyleSheet("QSplashScreen{background:#0d1117;color:#c9d1d9;"
                             "border:1px solid #30363d;font-size:13px;}")
        splash.resize(360, 90)
        splash.showMessage(
            "SquidXplorer\n\nStarting up: loading the viewer and reading the acquisition.\n"
            "The first launch takes a few seconds.",
            Qt.AlignCenter,
        )
        splash.show()
        app.processEvents()               # without this it never paints before the blocking call
        return splash
    except Exception:                     # noqa: BLE001 - a splash must never stop the app opening
        return None


def main(dataset_path: str = None):
    path = dataset_path or (sys.argv[1] if len(sys.argv) > 1 else None)
    slot = None
    if _gui_cap_applies():
        try:
            slot = acquire_gui_slot()
        except GuiAlreadyOpen as e:
            print(f"squidxplorer-view: {e}", file=sys.stderr)
            sys.exit(1)
    if QApplication.instance() is None:     # only the process that CREATES the app may set these
        enable_hidpi()
    app = qt_app(sys.argv)       # pinned process-wide: main() returns the WINDOW, not the app

    # SAY IT IS LOADING. Spencer, 2026-07-27: "startup needs some indication that work is
    # happening. Right now silence is indistinguishable from a crash", corroborated by a launch
    # that reported an empty window title and a null window handle for several seconds while
    # napari imported.
    #
    # It has to come BEFORE `PlateWindow(path)`, because that call is the slow part: napari's
    # import happens inside it, so anything hung off the window itself appears only once the wait
    # is already over. A splash is the one thing that can be on screen during a blocking
    # constructor.
    splash = _startup_splash(app)
    win = PlateWindow(path)
    _install_footprint_monitor(app, win)
    win._gui_slot = slot                  # the reservation lives as long as the window

    # FULL HEIGHT, DESIGN WIDTH -- not maximised. `showMaximized()` was tried first, on Spencer's
    # "start full screen and let me close it down", and Julio caught it immediately on a laptop:
    # "aspect ratio is good in height, but too much width". That is the layout telling the truth.
    # This root is a PORTRAIT window (596 x 850): a plate that takes the top half and wants to be
    # taller, so height is the dimension it can actually use and width past the design number just
    # pads empty gutters around the plate. Maximising is the right default for a document window
    # and the wrong one for this.
    #
    # So `_default_root_size` takes the whole usable height and leaves the width alone, which
    # satisfies both readings of the request: it fills the screen in the direction that helps, and
    # it is still an ordinary resizable window the user can drag to any shape from there.
    win.show()
    if splash is not None:
        splash.finish(win)
    if not app.property("_squidxplorer_test"):
        try:
            # `exec()`, not `exec_()`: PyQt6 removed every trailing-underscore alias, so `exec_`
            # is an AttributeError there -- the window would paint and the process would die on
            # the next line. PyQt5 has both, so this spelling is the one that works on either
            # binding. The suite never caught it because tests set `_squidxplorer_test` and return
            # the window instead of entering the event loop, so this line has no coverage by
            # construction: it is the one statement only a real launch executes.
            sys.exit(app.exec())
        finally:
            release_gui_slot(slot)
    return win


if __name__ == "__main__":
    main()
