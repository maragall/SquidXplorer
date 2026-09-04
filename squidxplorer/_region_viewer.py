"""Decentralized viewer windows: one INDEPENDENT napari window per selection."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from qtpy.QtCore import QObject, Qt, QTimer, Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from squidxplorer import _bitdepth
from squidxplorer import _measure
from squidxplorer._time_point import TimePointBar
from squidxplorer._address import Address, Extent
from squidxplorer._logpane import ViewLog, get_logger
from squidxplorer._fontscale import rescale_fonts, window_screen
from squidxplorer._worker_lifecycle import launch as _launch_worker

log = get_logger("regionviewer")

_sip = None
try:                                                     # pragma: no cover - binding detail
    import importlib

    import qtpy as _qtpy

    if _qtpy.API_NAME.startswith("PyQt"):
        _sip = importlib.import_module(f"{_qtpy.API_NAME}.sip")
except Exception:                                        # pragma: no cover
    try:
        import sip as _sip
    except ImportError:
        _sip = None

# `_lut_clipboard` keeps the NON-clipboard LUT helpers (per-channel read, apply, match-to-raw);
# the copy/paste clipboard itself was shelved on 2026-08-19 (Julio: "Shelf the LUT logic
# completely").
from squidxplorer import _lut_clipboard, _mosaic_playback, _roi_tools, _volume_view

# The ROI edge-colour cycle is defined once, in `_roi_tools`; historical alias.
from squidxplorer._roi_tools import ROI_COLORS as _ROI_COLORS  # noqa: E402,F401


@dataclass(frozen=True)
class View:
    """The ONE thing an operator targets: a named set of regions."""
    id: str
    name: str
    regions: tuple
    kind: str = "window"
    window_id: Optional[int] = None
    roi_bbox: Optional[tuple] = None
    parent_id: Optional[int] = None


_GLOBAL = "global"
_INHERIT = "inherit"

_SETTING_BASELINE = {
    "tenengrad_focus": _GLOBAL,
    "channel_visibility": _GLOBAL,
    "luts": _INHERIT,
}


def _alive(widget) -> bool:
    """Is this Qt object still there on the C++ side?

    ``sip.isdeleted`` is the only reliable question — a deleted QWidget is a live PYTHON object
    whose every attribute access raises, so ``is not None`` answers yes about a corpse.
    """
    if widget is None:
        return False
    if _sip is not None:                         # the module-level handle, resolved once at import
        try:
            return not _sip.isdeleted(widget)
        except Exception:                        # noqa: BLE001 - not a sip object: fall through
            pass
    try:
        widget.objectName()                      # PySide, or a non-sip object: ask Qt directly
        return True
    except RuntimeError:
        return False


def _copy_setting(value: Any) -> Any:
    """A private copy of one setting's value, so two windows can never share a mutable one."""
    if isinstance(value, dict):
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in value.items()}
    return value


def _unknown(name: str) -> KeyError:
    return KeyError(
        f"{name!r} is not a global-default setting. The global defaults are "
        f"{sorted(_SETTING_BASELINE)}; anything describing WHAT a window is looking at "
        "(2D/3D, the region, z, time_point) is per-window and does not belong here."
    )


@dataclass
class ViewDefaults:
    """The global defaults a NEW window opens with. Owned by :class:`ViewerManager`."""

    tenengrad_focus: bool = False
    channel_visibility: "dict[str, bool]" = field(default_factory=dict)
    luts: "dict[str, dict]" = field(default_factory=dict)

    def get(self, name: str) -> Any:
        if name not in _SETTING_BASELINE:
            raise _unknown(name)
        return _copy_setting(getattr(self, name))

    def set(self, name: str, value: Any) -> None:
        if name not in _SETTING_BASELINE:
            raise _unknown(name)
        setattr(self, name, _copy_setting(value))

    def snapshot(self) -> "dict[str, Any]":
        """Every default as a private copy, ready to become one window's baseline."""
        return {n: _copy_setting(getattr(self, n)) for n in _SETTING_BASELINE}


class ViewSettings:
    """ONE window's global-default settings: the baseline it opened with plus the user's overrides."""

    def __init__(self, baseline: "Optional[dict[str, Any]]" = None) -> None:
        base = ViewDefaults().snapshot() if baseline is None else baseline
        self._baseline: "dict[str, Any]" = {
            n: _copy_setting(base.get(n, getattr(ViewDefaults(), n))) for n in _SETTING_BASELINE}
        self._values: "dict[str, Any]" = {n: _copy_setting(v) for n, v in self._baseline.items()}
        self._overridden: "set[str]" = set()

    def get(self, name: str) -> Any:
        """This window's current value for *name*, as a private copy."""
        if name not in _SETTING_BASELINE:
            raise _unknown(name)
        return _copy_setting(self._values[name])

    def baseline(self, name: str) -> Any:
        """What this window OPENED with for *name* -- what :meth:`reset` goes back to."""
        if name not in _SETTING_BASELINE:
            raise _unknown(name)
        return _copy_setting(self._baseline[name])

    def set(self, name: str, value: Any) -> bool:
        """Change *name* FOR THIS WINDOW, and report whether it is now diverged on it."""
        if name not in _SETTING_BASELINE:
            raise _unknown(name)
        self._values[name] = _copy_setting(value)
        if self._values[name] == self._baseline[name]:
            self._overridden.discard(name)
        else:
            self._overridden.add(name)
        return name in self._overridden

    @property
    def diverged(self) -> "tuple[str, ...]":
        """The settings this window has overridden, sorted, so a label reads the same every time."""
        return tuple(sorted(self._overridden))

    def is_diverged(self, name: "Optional[str]" = None) -> bool:
        """Diverged on *name*, or on anything at all when *name* is None."""
        if name is None:
            return bool(self._overridden)
        if name not in _SETTING_BASELINE:
            raise _unknown(name)
        return name in self._overridden

    def reset(self, name: "Optional[str]" = None) -> "tuple[str, ...]":
        """Drop the overrides, back to what this window opened with. Returns what actually moved."""
        if name is not None and name not in _SETTING_BASELINE:
            raise _unknown(name)
        names = self.diverged if name is None else (
            (name,) if name in self._overridden else ())
        for n in names:
            self._values[n] = _copy_setting(self._baseline[n])
            self._overridden.discard(n)
        return names

    def adopt(self) -> None:
        """Take the current values as the new baseline, so nothing reads as diverged any more."""
        self._baseline = {n: _copy_setting(v) for n, v in self._values.items()}
        self._overridden.clear()

    def snapshot(self) -> "dict[str, Any]":
        """Every current value as a private copy."""
        return {n: _copy_setting(v) for n, v in self._values.items()}


def _level_shape(level: Any) -> "Optional[tuple[int, int]]":
    """The (height, width) of one pyramid level, or None if it has no 2-D+ shape."""
    shp = getattr(level, "shape", None)
    if not shp or len(shp) < 2:
        return None
    return int(shp[-2]), int(shp[-1])


def _crop_levels_to_bbox(levels: "list", region_bbox_um: "Sequence[float]",
                         roi_bbox_um: "Sequence[float]"):
    """Crop a LAZY multiscale pyramid to an ROI box, returning ``(cropped_levels, cropped_bbox_um)``"""
    try:
        x0, y0, x1, y1 = (float(v) for v in region_bbox_um)
        rx0, ry0, rx1, ry1 = (float(v) for v in roi_bbox_um)
    except Exception:                                    # noqa: BLE001 - malformed box, skip crop
        return None
    if x1 <= x0 or y1 <= y0:
        return None
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
            l0 = (c0, c1, r0, r1, w, h)
    if not out or l0 is None:
        return None
    c0, c1, r0, r1, w0, h0 = l0
    nbbox = (x0 + (c0 / w0) * span_x, y0 + (r0 / h0) * span_y,
             x0 + (c1 / w0) * span_x, y0 + (r1 / h0) * span_y)
    try:
        from squidxplorer._mosaic_source import strictly_decreasing_levels
        out = strictly_decreasing_levels(out)
    except Exception:                                    # noqa: BLE001 - a 1-level pyramid is fine
        pass
    return out, nbbox

_REGION_LOAD_DEBOUNCE_MS = 140

_RAW_OP = "raw"

# `full_res_level` / `_DEFAULT_MAX_3D_TEXTURE` moved out with their only users: the 3D cluster
# (`_volume_view`) and the ROI cluster (`_roi_tools`) import them at their own use sites.


#: THE one spacing of the left column: margins, gaps between chips, gaps between slots.
COLUMN_PX = 6
#: A view chip's fixed height and font, in pixels (Julio, 2026-08-25, live on 862 px: "The
#: SquidXplorer buttons at the top of the left dock should be smaller as well, this will give
#: the log a bit more height"). A QFont, never a stylesheet font: a stylesheet font resolves
#: at polish, after the height was fixed from the wrong metrics.
CHIP_PX = 22
CHIP_FONT_PX = 11


class RegionViewer(QMainWindow):
    """ONE independent napari window over a subset of regions."""

    closed = Signal(object)
    regionsChanged = Signal(object)   # emits self: this window ADOPTED a region it was not opened
    #                                   over, so anything that published its region set — the
    #                                   navigator row, the plate's per-view wash — is now stale.

    _op_action: Optional[str] = None
    _op_address: Any = None
    _result_region: Optional[str] = None
    _op_progress: Any = None
    open_clock: Any = None
    #: Has :meth:`dispose` already run? Same class-default rule as the attributes above, and here
    #: it is load-bearing rather than defensive: ``dispose`` is reachable from ``closeEvent`` on a
    #: window whose ``__init__`` raised partway, and a bare read of a missing attribute there
    #: would replace the real error with an attribute error out of Qt's machinery.
    _disposed: bool = False
    #: The :class:`~squidxplorer._view_deck.ViewDeck` holding this view as a tab, or None when it
    #: is a free-standing window. THE ONE ANSWER to "where does this view live" — written only by
    #: the deck's dock/undock, read by everything that needs to act on the window a view is in.
    #: Same class-default rule as above: it is read from slots.
    _host = None

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
        parent_id: Optional[int] = None,
        settings: "Optional[ViewSettings]" = None,
        fovs: bool = False,
    ) -> None:
        super().__init__(parent)
        self._reader = reader
        self._meta = meta
        #: WHAT THIS WINDOW WAS OPENED OVER. Historical and immutable — not "where it can go now",
        #: which is the cursor's order and is read through the ``_regions`` property below. The two
        #: were one field until the plate became a navigator: a window can now ADOPT a region it was
        #: not opened over, and a field kept in step with the cursor by hand is the second copy that
        #: ``_region_nav``'s "one cursor, no second copy" rule exists to forbid.
        self._seed_regions = [str(r) for r in regions]
        self.window_id = int(window_id)
        self._worker = None
        self._load_gen = 0
        self._retired_workers: list = []
        self._shown_region: Optional[str] = None
        self._pending_region: Optional[str] = None
        self._load_timer: Optional[QTimer] = None
        self._time_load_timer: Optional[QTimer] = None
        self._pane = None
        self._slider = None
        self._cursor = None
        self._native3d = None
        self._focus_worker = None
        self._video_worker = None
        self._png_worker = None
        self._manager = manager
        if manager is not None:
            # A LATER region can prove the dataset holds bigger numbers than the one this window
            # opened on -- the 14-bit set reads 3437 at C3 and 16380 at E7. Layers already built
            # carry a slider bounded by the old ceiling and cannot reach the new pixels, so every
            # window listens for the rise rather than only the one that triggered it.
            manager.depthChanged.connect(self._on_depth_changed)
        self._operator_specs = list(operator_specs or [])
        self._render_mode = "2d"
        self._run_operator = run_operator
        self.parent_id = parent_id
        self.settings = settings if settings is not None else ViewSettings()
        self._settings_applied = False
        self._roi_bbox = roi_bbox
        self._roi_layer = None
        # A FOVs VIEW walks this region's fields with the CAMERA. It loads nothing an ordinary
        # window would not: the mosaic is fused once and the slider only re-points the camera at
        # one field's box, which is why a step is instant and why playback on this axis is honest.
        # See `squidxplorer/_fov_nav.py` for why that is not a reversal of `_region_nav`'s "the
        # navigation unit is the region".
        self._fov_mode = bool(fovs)
        self._fov_slider = None    # the FOV axis, built ONLY in a FOVs view; None means "no axis"
        self._fov_layer = None     # the napari Shapes layer this window draws FOV rectangles on
        self._fov_boxes_cache: dict = {}   # the current region's FOV boxes; see _draw_fov_boxes
        if self._fov_mode and self._roi_bbox is not None:
            # No precedence rule and no fallback. An ROI view is CROPPED to one box and a FOVs
            # view walks every field of the whole region; a window that claimed to be both would
            # have to pick one silently, and the user would be looking at the other.
            raise ValueError(
                "a FOVs view walks the whole region's fields and an ROI view is cropped to one "
                "box; a window cannot be both. Open the FOVs view from the parent window rather "
                "than from inside an ROI child.")
        self._op_action: Optional[str] = None
        self._op_address: Any = None
        self._result_region: Optional[str] = None

        # THE SEED, not the live set: this runs before ``_build`` makes the cursor, and the name a
        # window is born with describes what it was OPENED over. Keeping it derived from the live
        # set would rename the window under the user the first time the plate navigated it
        # somewhere new, and the title is the only visible join between a log line and a window.
        self._derived_name = title or self._view_label(self._seed_regions)
        # THE ONE PLACE A KIND PREFIXES THE LABEL. The two are mutually exclusive by the refusal
        # above, so this is an elif and not a precedence question.
        if self._fov_mode:
            self._derived_name = f"FOVs · {self._derived_name}"
        elif self._roi_bbox is not None:
            self._derived_name = f"ROI · {self._derived_name}"
        self._display_name = self._derived_name
        self._refresh_title()
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.log = ViewLog(log, self.window_id)

        self.resize(*self._default_view_size())
        off = 28 * ((self.window_id - 1) % 8)
        home = self._home_screen()
        origin = home.availableGeometry().topLeft() if home is not None else None
        ox, oy = (origin.x(), origin.y()) if origin is not None else (0, 0)
        self.move(ox + 120 + off, oy + 90 + off)

        self._build()

    _OPEN_W, _OPEN_H = 860, 720

    def _home_screen(self):
        """The display this window belongs on: the PLATE's, else this window's, else the primary."""
        opener = self._manager.parent() if self._manager is not None else None
        return window_screen(opener if opener is not None else self)

    def _default_view_size(self) -> tuple:
        """The design size, floored at a third of the screen and capped to fit on it."""
        w, h = self._OPEN_W, self._OPEN_H
        screen = self._home_screen()
        if screen is None:
            return w, h
        avail = screen.availableGeometry()
        w = min(max(w, avail.width() // 3), max(1, avail.width() - 40))
        h = min(max(h, avail.height() // 3), max(1, avail.height() - 80))
        return int(w), int(h)

    @staticmethod
    def _view_label(regions: "list[str]", limit: int = 3) -> str:
        if not regions:
            return "(empty)"
        if len(regions) <= limit:
            return ", ".join(regions)
        return ", ".join(regions[:limit]) + f", +{len(regions) - limit}"

    @property
    def display_name(self) -> str:
        """The window's LABEL, without the ``[wid]`` bracket. Mutable; the id is not."""
        return self._display_name

    def _refresh_title(self) -> None:
        """Render identity + label into the title bar. The ONE place the two are joined."""
        self.setWindowTitle(f"[{self.window_id}] {self._display_name}")

    def set_display_name(self, name: "Optional[str]") -> bool:
        """Rename this window. Returns False for a blank name, which is a refusal, not a reset."""
        if name is None:
            self._display_name = self._derived_name
            self._refresh_title()
            return True
        text = str(name).strip()
        if not text:
            return False
        self._display_name = text
        self._refresh_title()
        return True

    def _build(self) -> None:
        from squidxplorer._napari_pane import make_pane
        from squidxplorer._region_nav import RegionCursor, RegionSlider

        central = QWidget(self)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        pane, mode, message = make_pane(show_docks=True)
        if pane is None or not getattr(pane, "ok", False):
            msg = QLabel(f"napari viewer unavailable - {message}")
            msg.setAlignment(Qt.AlignCenter)
            msg.setWordWrap(True)
            msg.setStyleSheet("color:#ffd7d7;background:#3a2020;padding:16px;font-size:13px;")
            lay.addWidget(msg, 1)
            self.setCentralWidget(central)
            return
        self._pane = pane
        try:
            model = getattr(getattr(pane, "mosaic", None), "_model", None)
            if model is not None:
                model.text_overlay.visible = True
                model.text_overlay.font_size = 12
                model.cursor.events.position.connect(self._on_cursor_position)
        except Exception as exc:                          # noqa: BLE001 - a readout, never fatal
            log.debug("view %s could not wire the FOV readout: %s", self.window_id, exc)

        # THE CANVAS LOUPE (shift-left-click). Built here because this is where the napari Viewer
        # and the GL widget first become reachable; it costs one object until it is raised, and it
        # builds neither its source nor its worker until then.
        self._loupe = None
        try:
            self._install_canvas_loupe(pane)
        except Exception as exc:                          # noqa: BLE001 - a magnifier, never fatal
            log.debug("view %s could not wire the canvas loupe: %s", self.window_id, exc)

        # EVERYTHING WINDOW-SCOPED lives in napari's LEFT column, above the layer controls (UI
        # feedback 2026-08-19: "free up the viewer space to the top" — no full-width top dock;
        # "the operators for this window row should also be on the left vertical dock"): the
        # 2D/3D·ROI chip block on top, this view's operator panel under it. The right-edge dock
        # keeps ONLY the bulk-processing cards. A pane that cannot dock (headless ModelPane, a
        # napari without dock areas) keeps the column in the window body so every control stays
        # actuatable.
        # ONE plain column, everything visible (Julio, 2026-08-25: "the plate shouldn't be
        # collapsible, the dock shouldn't be collapsible... the operators shouldn't
        # collapse... it should display its buttons in a way that it's pleasing, appropriately
        # sized"): chips, the operators row with its controls under it, napari's layer
        # controls, the layer list (THE stretch), the plate slot and the log slot, one spacing.
        left_col = QWidget()
        self._left_col = left_col
        lv = QVBoxLayout(left_col)
        lv.setContentsMargins(COLUMN_PX, COLUMN_PX, COLUMN_PX, COLUMN_PX)
        lv.setSpacing(COLUMN_PX)
        lv.addWidget(self._build_view_controls(), 0)
        lv.addWidget(self.operator_panel(), 0)
        take = getattr(pane, "native_column_widgets", None)
        controls, tree = take() if callable(take) else (None, None)
        if controls is not None:
            lv.addWidget(controls, 0)
        if tree is not None:
            tree.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            lv.addWidget(tree, 1)
        else:
            lv.addStretch(1)
        self._plate_log_host = QWidget()
        self._plate_log_slot = QVBoxLayout(self._plate_log_host)
        self._plate_log_slot.setContentsMargins(0, 0, 0, 0)
        self._plate_log_slot.setSpacing(COLUMN_PX)
        lv.addWidget(self._plate_log_host, 0)
        dock = getattr(pane, "dock_left_column", None)
        if not (callable(dock) and dock(left_col)):
            lay.addWidget(left_col, 0)
        # The RUN/MOVIE PROGRESS BAR stays in the window body: a run's progress must be visible
        # while the operator dock is collapsed to its grip. Hidden, it costs zero height.
        self._op_progress = QProgressBar()
        self._op_progress.setTextVisible(True)
        self._op_progress.setFixedHeight(16)
        self._op_progress.setStyleSheet(self._PROGRESS_QSS)
        self._op_progress.hide()
        lay.addWidget(self._op_progress, 0)
        lay.addWidget(pane, 1)

        self._cursor = RegionCursor()
        self._cursor.on_problem(self._say)
        self._cursor.subscribe(self._on_region_changed)
        self._slider = RegionSlider()
        self._slider.on_problem(self._say)
        self._slider.bind(self._cursor)
        lay.addWidget(self._slider)

        # THE FOV AXIS, and only in a FOVs view. Built here rather than unconditionally-and-hidden
        # (which is what the region slider and the timepoint bar do) because this one is not free:
        # it costs a napari `Dims`, a `QtDims`, a `QTimer` and an `AnimationThread`, and window
        # open time is a tracked complaint. `None` therefore means "this window has no FOV axis".
        if self._fov_mode:
            from squidxplorer._fov_nav import FovSlider

            self._fov_slider = FovSlider(on_change=self._on_fov_changed)
            self._fov_slider.on_problem(self._say)
            lay.addWidget(self._fov_slider)

        self._time_point_bar = TimePointBar(on_change=self._on_time_point_changed, playback=True)
        self._time_point_bar.on_problem(self._say)
        self._time_point_bar.set_count(int((self._meta or {}).get("n_t", 1) or 1))
        lay.addWidget(self._time_point_bar)

        self.setCentralWidget(central)

        # Derived display color must be labeled: said once per open, pinned on the raw group tooltip.
        try:
            from squidxplorer._channels import color_note

            note = color_note((self._meta or {}).get("channels"))
            if note:
                pane.mosaic.set_color_note(_RAW_OP, note)
                self._say(note)
        except Exception as exc:                          # noqa: BLE001 - a label, never fatal
            log.debug("view %s could not state color provenance: %s", self.window_id, exc)

        # SEED the cursor: this announces region 0 to the loader, so the first mosaic loads now.
        # Reads ``_seed_regions`` and not ``_regions``, because ``_regions`` reads back OUT of the
        # cursor — at this instant it would be answering from the empty order it is about to be
        # given, and seeding a cursor from itself is a no-op that leaves the window blank.
        self._cursor.set_order(self._seed_regions)
        if self._cursor.index is None and self._seed_regions:
            self._cursor.set_index(0)

    _BOX_QSS = "QFrame{background:#0d1117;border:1px solid #232b3a;border-radius:5px;}"
    _CHIP_QSS = (
        "QPushButton{background:#161b22;color:#c9d1d9;border:1px solid #30363d;"
        "border-radius:4px;padding:1px 6px;}"
        "QPushButton:hover{background:#21262d;}"
        "QPushButton:checked{background:#1f6feb;color:#ffffff;border-color:#1f6feb;}"
        "QPushButton:disabled{color:#586069;border-color:#20262e;}"
    )
    _COMBO_CHIP_QSS = (
        "QComboBox{background:#161b22;color:#c9d1d9;border:1px solid #30363d;"
        "border-radius:4px;padding:3px 9px;font-size:11px;min-width:120px;}"
        "QComboBox:hover{background:#21262d;color:#c9d1d9;}"
        "QComboBox:disabled{color:#586069;border-color:#20262e;}"
        "QComboBox QAbstractItemView{background:#161b22;color:#c9d1d9;border:1px solid #30363d;"
        "selection-background-color:#1f6feb;selection-color:#ffffff;outline:none;}"
    )

    def _chip(self, text: str, tip: str, slot, *, checkable: bool = False) -> QPushButton:
        b = QPushButton(text)
        b.setToolTip(tip)
        b.setCheckable(checkable)
        b.setCursor(Qt.PointingHandCursor)
        b.setStyleSheet(self._CHIP_QSS)
        font = QFont(b.font())
        font.setPixelSize(CHIP_FONT_PX)
        b.setFont(font)
        b.setFixedHeight(CHIP_PX)
        b.clicked.connect(lambda _=False: slot())
        return b

    def _build_view_controls(self) -> QWidget:
        """The view chip block: the 2D/3D essentials plus a summon grip, everything else folded.

        `_build` docks it at the TOP of napari's left column, above the layer controls
        (`MosaicPane.dock_view_controls`), so the canvas gains the height the old horizontal
        top dock spent (UI feedback 2026-08-19: "Should be on the left column, where the
        controls are"). Chip attributes (`_btn_3d`, `_btn_roi`, `_btn_focus`, `_btn_record`,
        `_btn_fovs`) are pinned by tests and GATE 3; only the parenting and row wrapping moved.

        HERO DECLUTTER (team feedback 2026-08-25: "minimize most of the... tools. They just
        take up too much room"): only the 2D/3D essentials stay visible; the rest of the
        chips fold behind the "controls" summon grip, collapsed by default, per view.
        """
        view_box = QFrame(self)
        view_box.setStyleSheet(self._BOX_QSS)
        vv = QVBoxLayout(view_box)
        vv.setContentsMargins(COLUMN_PX, COLUMN_PX, COLUMN_PX, COLUMN_PX)
        vv.setSpacing(COLUMN_PX)
        # NO 2D button (Julio, 2026-08-25: "There should not be 2D button since we make
        # separate tabs for the 3d view."): a 2D tab IS 2D and a 3D tab IS 3D; nothing
        # switches modes in place. The chip and `_view_roi_2d` are deleted whole.
        self._btn_3d = self._chip("3D", "Open this view in 3D as a new tab.", self._open_3d)
        self._btn_focus = self._chip("⌖ focus", "Jump the z-slider to the sharpest plane.",
                                     self._focus_reference_plane)
        # The "▣ plate" chip is GONE (UI feedback 2026-08-17): the working layout keeps the plate
        # BESIDE the views, so "bring the plate forward" stopped being a job. The ⚙ controls chip
        # is built in `operator_panel()` now — the whole per-window operator surface lives in the
        # views window's collapsible dock (2026-08-19).
        self._btn_record = self._chip(
            "⏺ movie", "Export this view as an .mp4 over its time or z axis.",
            self._record_movie)
        # The PNG export renders the DATA, never the canvas: a screenshot is screen resolution,
        # this is the visible layer's own pixels at native pitch (Julio: "a high-resolution
        # (i.e., zoom-able, high DPI) PNG for powerpoints").
        self._btn_png = self._chip(
            "⎙ png", "Save this view as a full-resolution PNG.", self._save_png)
        # FOVs. The ROI chips beside it are for a box the user draws; this is for the boxes the
        # ACQUISITION already drew. On a sparse run — the AF sweep sets are 16 fields at 7x the
        # field pitch, so 3% of the mosaic is data — checking focus means visiting each field, and
        # doing that by wheel-zoom is the complaint this answers.
        self._btn_fovs = self._chip("⊞ FOVs", self._FOVS_TIP, self._open_fovs)
        # ONE grid of every chip, all visible, nothing folded (Julio, 2026-08-25: "the GUI
        # buttons such as 'FOVs' shouldn't collapse"). The ROI chip is two-state (draw / go).
        from qtpy.QtWidgets import QGridLayout

        self._btn_roi = self._chip(self._ROI_DRAW[0], self._ROI_DRAW[1], self._roi_chip_clicked)
        grid = QGridLayout()
        grid.setSpacing(COLUMN_PX)
        chips = [
            self._btn_3d, self._btn_roi, self._btn_fovs,
            self._chip("⊙ select", "Click an ROI to select it; Delete removes it.",
                       self._select_rois),
            self._chip("✕ clear", "Remove all ROIs in this window.", self._clear_rois),
            self._chip("→ window", "Open the drawn ROIs as child views.", self._open_roi_children),
            self._btn_focus, self._btn_record, self._btn_png,
        ]
        # 3D camera snaps, IN CAMERA (no panels: the grid is the whole UI): three axis
        # planes plus a refit. Hidden on a 2D tab; `note_volume_tab` shows them.
        self._snap_chips = [
            self._chip("XY", "Snap the 3D camera to the XY plane (top view).",
                       lambda: _volume_view.snap_camera(self, "xy")),
            self._chip("XZ", "Snap the 3D camera to the XZ plane.",
                       lambda: _volume_view.snap_camera(self, "xz")),
            self._chip("YZ", "Snap the 3D camera to the YZ plane.",
                       lambda: _volume_view.snap_camera(self, "yz")),
            self._chip("fit", "Refit the volume to the canvas; the angles stay.",
                       lambda: _volume_view.snap_camera(self, "fit")),
        ]
        for chip in self._snap_chips:
            chip.setVisible(self._is_volume_tab)
        chips += self._snap_chips
        for k, chip in enumerate(chips):
            chip.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            grid.addWidget(chip, k // 3, k % 3)
        for col in range(3):
            grid.setColumnStretch(col, 1)
        vv.addLayout(grid)
        self._refresh_record_chip()
        self._refresh_fovs_chip()
        # THE PER-WINDOW OPERATOR SURFACE IS NOT IN THIS BLOCK. "Operators for this window" (the
        # dropdown, ⚙ controls, Run, save) and the Detect row live in `operator_panel()`, docked
        # DIRECTLY BELOW this block in the same left column by `_build`. The Defaults group (auto
        # focus / make default / diverged / reset) is SHELVED outright — the settings STORE stays
        # (`ViewSettings` / `ViewDefaults` still drive autofocus-on-open and child-window LUT
        # inheritance), only its control surface is gone.
        self._view_controls = view_box
        return view_box

    #: Whether this view IS a 3D tab (spawned by another view's 3D chip).
    _is_volume_tab = False

    def note_volume_tab(self) -> None:
        """This view IS the 3D tab: there is no 2D/3D mode switch (a 2D tab is 2D, a 3D
        tab is 3D - Julio, 2026-08-25), so its own 3D chip is disabled with the way back
        stated: closing the tab."""
        self._is_volume_tab = True
        btn = getattr(self, "_btn_3d", None)
        if btn is not None and _alive(btn):
            btn.setEnabled(False)
            btn.setToolTip("This tab is the 3D view; close it to go back to 2D.")
        for chip in getattr(self, "_snap_chips", ()):
            if _alive(chip):
                chip.setVisible(True)

    _AT_DEFAULTS_QSS = "color:#8b949e;font-size:10px;border:none;"
    _PROGRESS_QSS = (
        "QProgressBar{background:#161b22;border:1px solid #30363d;border-radius:3px;"
        "color:#c9d1d9;font-size:10px;text-align:center;}"
        "QProgressBar::chunk{background:#e3b341;border-radius:3px;}"
    )

    #: This window's operator surface, or None until a dock asks for it. A class default so a
    #: half-built window answers rather than raising out of Qt's attribute machinery.
    _op_panel = None

    def operator_panel(self) -> QWidget:
        """THIS WINDOW's operator surface, LIVING IN THIS WINDOW'S OWN LEFT COLUMN.

        The old "Operators for this window" toolbar (dropdown, ⚙ controls, Run, save) plus the
        pane's Detect row. It sat in the right-edge dock for one day; Julio (2026-08-19): "The
        operators for this window row should also be on the left vertical dock. The bulk
        processing is what is solutioned on the right vertical column." `_build` docks it under
        the 2D/3D·ROI chips, one panel per view, and `dispose` deletes it with the view.
        """
        panel = self._op_panel
        if panel is not None and _alive(panel):
            return panel
        panel = QWidget()
        panel.setStyleSheet("background:#0b0e14;")
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(4)

        # A plain box: the fold's own "operators" summon bar is the title (quiet by
        # default, fewer words), so a second heading here would just repeat it.
        op_box = QFrame(self)
        op_box.setStyleSheet(self._BOX_QSS)
        ov = QVBoxLayout(op_box)
        ov.setContentsMargins(8, 5, 8, 6)
        ov.setSpacing(4)
        opr = QHBoxLayout(); opr.setSpacing(4)
        self._op_combo = QComboBox()
        self._op_combo.setStyleSheet(self._COMBO_CHIP_QSS)
        for spec in self._operator_specs:
            self._op_combo.addItem(str(spec[1]), spec[0])
        if self._op_combo.count() == 0:
            self._op_combo.addItem("no operators", None)
            self._op_combo.setEnabled(False)
        opr.addWidget(self._op_combo, 1)
        # ONE FLOW, fewer buttons (Julio, 2026-08-25): "They can preview on the window...
        # After they preview, they can say run on plate, and then it will save to disk.
        # No body runs on whole plate to preview." Preview is window-scoped and writes
        # nothing; Run on plate is THE save path (and the bulk path - the cards died with
        # the right-edge dock). The save checkbox is gone.
        self._btn_preview = self._chip(
            "Preview", "Previews this view, writing nothing: the same computation Run on "
            "plate saves, shown at the z in view (a 3D tab shows the volume). Draw an ROI "
            "for a faster preview.", self._preview_view_operator)
        opr.addWidget(self._btn_preview)
        self._btn_run_plate = self._chip(
            "Run on plate", self._RUN_PLATE_TIP, self._run_plate_operator)
        opr.addWidget(self._btn_run_plate)
        ov.addLayout(opr)
        self._op_combo.currentIndexChanged.connect(lambda _i: self._on_operator_changed())
        # NO "Match layers to raw" and no LUT chrome here: match-to-raw is shelved whole
        # (Julio, 2026-08-19) and the two-button LUT clipboard lives in the 2D/3D·ROI block.
        pv.addWidget(op_box)
        # THE PARAM SLOT: ⚙ controls INSERTS the operator's panel here, directly under the
        # operators row (Julio, 2026-08-25: "see this as an insertion to a list"). One slot,
        # one panel at a time; the widget stays the plate's (_op_tabs), merely re-hosted.
        self._param_slot = QVBoxLayout()
        self._param_slot.setContentsMargins(0, 0, 0, 0)
        pv.addLayout(self._param_slot)

        self._op_panel = panel
        self._on_operator_changed()
        return panel

    def _on_operator_changed(self) -> None:
        """ONE parameter surface (Julio, 2026-08-25, ruling w): the selected operator's panel
        is always under the row; no params, nothing there. `operator_kwargs_for` reads it."""
        combo = getattr(self, "_op_combo", None)
        key = combo.currentData() if combo is not None else None
        self._refresh_save_tooltip(key)
        self._remove_param_slot()
        if not key or not self._declared_params(str(key)):
            return
        release = getattr(self._plate(), "release_operator_panel", None)
        if not callable(release):
            return
        try:
            panel = release(str(key))
        except Exception as exc:                         # noqa: BLE001 - named, never silent
            self._say(f"controls: could not open {key}: {exc}")
            return
        if panel is not None:
            self._insert_param_slot(str(key), panel)

    #: The operator panel currently inserted in this view's param slot, and its key.
    _inserted_panel = None
    _inserted_key = None

    #: The inserted slot's height ceiling: a FIXED slot in the column (the chart's rule -
    #: inserting a slot shrinks the flexible neighbours, never the whole window). The panel
    #: scrolls inside itself.
    _PARAM_SLOT_MAX_PX = 360

    def _insert_param_slot(self, key: str, panel) -> None:
        """Insert *panel* under the operators row, releasing any previously inserted one."""
        self._remove_param_slot()
        panel.setMaximumHeight(self._PARAM_SLOT_MAX_PX)
        self._param_slot.addWidget(panel)
        panel.setVisible(True)
        self._inserted_panel = panel
        self._inserted_key = str(key)

    def _remove_param_slot(self) -> None:
        """Take the inserted panel OUT without disposing it - it is the plate's live widget
        (the run's single source of truth), so it must survive this view. The plate ADOPTS
        it back: a parentless orphan awaiting deleteLater measured a segfault in the next
        window's teardown."""
        panel, key = self._inserted_panel, self._inserted_key
        self._inserted_panel = self._inserted_key = None
        if panel is None or not _alive(panel):
            return
        self._param_slot.removeWidget(panel)
        panel.hide()
        panel.setParent(None)
        adopt = getattr(self._plate(), "adopt_operator_panel", None)
        if callable(adopt) and key:
            try:
                adopt(key)
            except Exception as exc:             # noqa: BLE001 - never let a re-home block
                log.warning("view %s could not return %s's panel to the plate: %s: %s",
                            self.window_id, key, type(exc).__name__, exc)

    def _plate(self):
        """The plate window, or None. The plate owns every operator panel; this window borrows."""
        return None if self._manager is None else self._manager.parent()

    # -- ONE WINDOW: this view can host the plate view + log as slots (2026-08-25) --------------

    #: Whether this view currently hosts the plate view + log slots.
    _hosts_plate_slots = False

    def adopt_plate_slots(self, plate_box, log_panel) -> bool:
        """Host the plate's two slot widgets in this view's left column. Qt reparents them
        out of any previous holder; returns whether the column exists to host in."""
        slot = getattr(self, "_plate_log_slot", None)
        if slot is None:
            return False
        slot.addWidget(plate_box)
        slot.addWidget(log_panel)
        plate_box.setVisible(True)
        log_panel.setVisible(True)
        self._hosts_plate_slots = True
        return True

    def release_plate_slots(self) -> None:
        """Stop hosting: the plate ADOPTS its widgets home (they are never orphans)."""
        if not self._hosts_plate_slots:
            return
        self._hosts_plate_slots = False
        adopt = getattr(self._plate(), "adopt_plate_slots_home", None)
        if callable(adopt):
            try:
                adopt()
            except Exception as exc:             # noqa: BLE001 - a re-home must never block
                log.warning("view %s could not return the plate slots: %s: %s",
                            self.window_id, type(exc).__name__, exc)

    def _run_scope(self):
        """WHERE a run from this window goes: ``(regions, windows)``: its regions narrowed to
        the ROI's own FOVs, and (ruling z) the ROI as a window in each of those frames, so
        the engine reads and solves the box plus a halo rather than whole fields."""
        regions = list(self._regions)
        if self._roi_bbox is None or not regions:
            return regions, None
        from squidxplorer._mosaic_source import fov_windows_px, fovs_overlapping_bbox

        scoped: "dict[str, list[int]]" = {}
        windows: dict = {}
        for region in regions:
            fovs = fovs_overlapping_bbox(self._meta or {}, region, self._roi_bbox)
            if not fovs:
                return regions, None
            scoped[region] = fovs
            for fov, window in fov_windows_px(self._meta or {}, region, self._roi_bbox).items():
                if fov in fovs:
                    windows[(region, fov)] = window
        total = sum(len((self._meta or {}).get("fovs_per_region", {}).get(r) or []) for r in regions)
        picked = sum(len(v) for v in scoped.values())
        if any((r, f) not in windows for r, fs in scoped.items() for f in fs):
            windows = {}                          # a field the box touches but cannot window: whole
        if picked >= total and not windows:
            return regions, None
        self._say(f"ROI: running on {picked} of {total} field(s) - the ones your box touches.")
        return scoped, (windows or None)

    def _plate_operator_kwargs(self, key: str) -> dict:
        """What *key*'s panel on the plate is currently set to. ``{}`` = its declared defaults."""
        plate = self._plate()
        reader = getattr(plate, "operator_kwargs_for", None)
        if not callable(reader):
            return {}
        try:
            return dict(reader(str(key)) or {})
        except ValueError:
            raise                                # a REFUSED setting: the launch says it and stops
        except Exception as exc:                 # noqa: BLE001 - named, never a silent default
            log.warning("view %s could not read %s's parameters from the plate: %s: %s",
                        self.window_id, key, type(exc).__name__, exc)
            return {}

    @staticmethod
    def _declared_params(key: str) -> tuple:
        """The Params *key* declares (headline + advanced), or () when it declares none."""
        from squidxplorer import operator_params
        from squidxplorer._operations import operator_name

        try:
            return tuple(operator_params(operator_name(str(key))))
        except Exception:                        # noqa: BLE001 - an unknown key has none
            return ()

    def _z_kwargs_for_mode(self, key: str, current: dict) -> dict:
        """What THIS WINDOW'S 2D/3D choice means for the operator's z handling."""
        from squidxplorer import is_region_operator
        from squidxplorer._operations import operator_name

        if self._render_mode != "3d" or int((self._meta or {}).get("n_z") or 1) <= 1:
            return {}
        try:
            if not is_region_operator(operator_name(str(key))):
                return {}
        except Exception:                        # noqa: BLE001 - an unknown key: leave it alone
            return {}
        if "z_operator" in current and current["z_operator"] is None:
            return {}                            # already keeping every plane, on purpose
        chosen = str(current.get("z_operator") or "")
        from squidxplorer._engine import Z_REDUCER, operator_consumes

        try:
            reduces = bool(operator_consumes(chosen) & Z_REDUCER) if chosen else True
        except Exception:                        # noqa: BLE001 - an unknown z operator: leave it
            return {}
        if not reduces:
            return {}
        self._say(f"3D: stitching all {int((self._meta or {}).get('n_z') or 1)} z-planes "
                  f"(z_operator=None, every plane kept) - one pose graph, every plane fused "
                  f"from it.")
        return {"z_operator": None}

    def set_render_mode(self, mode: str) -> None:
        """Record whether this window is a PLANE or a VOLUME, and repaint what says so."""
        mode = "3d" if str(mode).lower() == "3d" else "2d"
        if mode == getattr(self, "_render_mode", "2d"):
            return
        self._render_mode = mode

    _RUN_PLATE_TIP = "Run on the plate selection and save to disk."

    def _refresh_save_tooltip(self, key) -> None:
        """The Run-on-plate chip says WHAT a save writes: a copy-saving operator's artifact
        is stitched_<folder>, not an OME-Zarr — implying the wrong artifact is how a register
        preview read as "doesn't do anything"."""
        btn = getattr(self, "_btn_run_plate", None)
        if btn is None:
            return
        tip = self._RUN_PLATE_TIP
        try:
            from squidxplorer._engine import operator_saves_copy
            from squidxplorer._operations import operator_name

            if key and operator_saves_copy(operator_name(str(key))):
                src = getattr(self._reader, "source_id", None)
                acq = Path(str(src)).name if src else "<folder>"
                tip = (f"Run over the plate selection and write stitched_{acq} beside the "
                       "acquisition (hardlinked copy with registered coordinates).")
        except Exception:                            # noqa: BLE001 - a tooltip, never a crash
            pass
        btn.setToolTip(tip)

    def _window_operators(self) -> list:
        """THE OPERATORS FOR THIS WINDOW: every processing layer it holds, raw excluded."""
        mosaic = getattr(self._pane, "mosaic", None) if self._pane is not None else None
        if mosaic is None:
            return []
        try:
            return [op for op in mosaic.ops() if op and op != _RAW_OP]
        except Exception as exc:                         # noqa: BLE001 - "none yet", but SAID
            log.warning("view %s could not read its layers' operators: %s: %s",
                        self.window_id, type(exc).__name__, exc)
            return []

    def show_operator_controls_for(self, key: str) -> None:
        """Select *key* in this view's dropdown; its controls follow (one surface)."""
        combo = getattr(self, "_op_combo", None)
        if combo is None:
            self._say(f"this view has no operator row to host {key!r}.")
            return
        i = next((k for k in range(combo.count()) if combo.itemData(k) == str(key)), None)
        if i is None:
            self._say(f"{key!r} is not in this view's operator dropdown.")
            return
        combo.setCurrentIndex(i)                 # the panel follows the selection

    def _refresh_record_chip(self) -> None:
        """Enable the record chip only when there is a movie to make, and SAY WHY when there is not."""
        from squidxplorer._video import axis_length, can_record, default_axis, encoder_problem

        btn = getattr(self, "_btn_record", None)
        if btn is None:
            return
        meta = self._meta or {}
        if not can_record(meta):
            btn.setEnabled(False)
            btn.setToolTip("This acquisition has a single timepoint and a single z plane, so "
                           "there is no axis to sweep into a movie.")
            return
        problem = encoder_problem()
        if problem:
            btn.setEnabled(False)
            btn.setToolTip(f"No mp4 encoder on this machine - {problem}")
            return
        axis = default_axis(meta)
        n = axis_length(meta, axis)
        btn.setEnabled(True)
        btn.setToolTip(
            f"Export what this window is showing as an .mp4: {n} frames along the "
            f"{'time' if axis == 't' else 'z'} axis of the region on screen, with the channels "
            f"that are visible and the contrast that is set. Runs off the UI thread; click again "
            f"to cancel.")

    def _visible_channels(self) -> "list[str]":
        """The channels the user can actually SEE, in acquisition order."""
        pane = self._pane
        mosaic = getattr(pane, "mosaic", None) if pane is not None else None
        names = [c["name"] for c in (self._meta or {}).get("channels", [])]
        if mosaic is None:
            return names
        visible = []
        for name in names:
            layer = mosaic.find(_RAW_OP, name)
            if layer is None or bool(getattr(layer, "visible", True)):
                visible.append(name)
        return visible or names

    def _record_movie(self) -> None:
        """Export this view's sweep to an .mp4. Second click cancels the run in flight."""
        from squidxplorer._video import DEFAULT_FPS, axis_length, can_record, default_axis

        worker = self._video_worker
        if worker is not None and worker.isRunning():
            worker.stop()
            self._say("cancelling the movie export…")
            return
        if self._reader is None or self._meta is None:
            self._say("show a region in this view first, then export a movie.")
            return
        if not can_record(self._meta):
            self._say("this acquisition has a single timepoint and a single z plane, so there is "
                      "no axis to sweep into a movie.")
            return
        region = self.current_region()
        axis = default_axis(self._meta)
        channels = self._visible_channels()
        luts = self._per_channel_luts()
        windows = [luts[ch]["clim"] for ch in channels
                   if luts.get(ch, {}).get("clim") is not None]
        if len(windows) != len(channels):
            windows = None
        rgb = {ch: luts[ch]["rgb"] for ch in channels if luts.get(ch, {}).get("rgb") is not None}

        default_name = f"{region}_{axis}.mp4"
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save the {axis.upper()}-axis movie of {region}", default_name, "Movie (*.mp4)")
        if not path:
            return
        if not str(path).lower().endswith(".mp4"):
            path = f"{path}.mp4"

        from squidxplorer._workers import _VideoWorker

        n = axis_length(self._meta, axis)
        w = _VideoWorker(self._reader, self._meta, region, path, axis=axis, fps=DEFAULT_FPS,
                         channels=channels, windows=windows, rgb_by_channel=rgb,
                         z_level=self._z_slider_index(), time_point=self.time_point, parent=self)
        self._show_progress(0, f"movie: 0 of {n} frames")
        self._say(f"exporting {n} {axis}-axis frames of {region} to {path}…")
        _launch_worker(
            self, w, slot="_video_worker",
            on_done=self._on_movie_done,
            on_problem=self._on_movie_failed,
            on_progress=lambda d, total: self._show_progress(
                int(100 * d / max(1, total)), f"movie: frame {d} of {total}"),
            on_finished=lambda: self._forget_video_worker(w),
            signals={"cancelled": self._on_movie_cancelled})

    def _z_slider_index(self) -> int:
        """Which z plane this window is showing, or 0 when it has no z slider."""
        v = self._napari_viewer()
        try:
            dims = v.dims
            nsteps = tuple(int(x) for x in (getattr(dims, "nsteps", ()) or ()))
            if len(nsteps) < 3 or nsteps[0] < 2:
                return 0
            return int(dims.current_step[0])
        except Exception:                            # noqa: BLE001 - no slider is z=0, not a crash
            return 0

    def _forget_video_worker(self, worker) -> None:
        if self._video_worker is worker:
            self._video_worker = None
        try:
            worker.deleteLater()
        except RuntimeError:
            pass

    def _on_movie_done(self, path: str, frames: int, seconds: float) -> None:
        self._hide_progress()
        size_mb = 0.0
        try:
            size_mb = Path(path).stat().st_size / 1e6
        except OSError:
            pass
        self._say(f"movie: {frames} frames -> {path} ({size_mb:.1f} MB) in {seconds:.1f}s.")

    def _on_movie_failed(self, reason: str) -> None:
        self._hide_progress()
        self._say(f"movie export failed: {reason}")

    def _on_movie_cancelled(self) -> None:
        self._hide_progress()
        self._say("movie export cancelled.")

    def _save_png(self) -> None:
        """Save what this view is SHOWING as a high-resolution PNG — data pixels, never a
        canvas screenshot. Every refusal is a named sentence, never a silent no-op."""
        from squidxplorer._png import PNG_MAX_PX, PngChannel, png_problem

        worker = self._png_worker
        if worker is not None and worker.isRunning():
            self._say("png: an export is already running - it will say when it lands.")
            return
        if self._reader is None or self._meta is None:
            self._say("png: show a region in this view first, then save a PNG.")
            return
        if self._render_mode == "3d":
            self._say("png: this view is showing a 3D volume - volume export is out of scope. "
                      "Switch the view to 2D to save a PNG of a plane.")
            return
        mosaic = getattr(self._pane, "mosaic", None) if self._pane is not None else None
        op = mosaic.visible_op() if mosaic is not None else None
        if op is None:
            self._say("png: no visible layer to export - every layer of this view is hidden.")
            return
        problem = png_problem()
        if problem:
            self._say(f"png: {problem}")
            return

        from squidxplorer._napari_view import colormap_hue_rgb, colormap_mid_rgb, full_res_level

        z_now = self._z_slider_index()
        channels = []
        for c in (self._meta or {}).get("channels", []):
            # Per CHANNEL, the topmost VISIBLE layer: the one-lit-op rule is per channel, so
            # the screen legitimately composites raw beside a result. One op's walk here
            # silently dropped the raw channels of a mixed scene from the export.
            layer = mosaic.top_visible_layer(c["name"])
            if layer is None:
                continue
            clim = getattr(layer, "contrast_limits", None)
            if clim is None:                      # a labels layer has no contrast window
                continue
            rgb = colormap_hue_rgb(layer) or colormap_mid_rgb(layer) or (255, 255, 255)
            channels.append(PngChannel(c["name"], layer.data, tuple(clim), tuple(rgb),
                                       z_index=z_now))
        if not channels:
            self._say("png: no visible intensity channel to export.")
            return

        region = self.current_region()
        # A FOVs view exports the FIELD on screen, not the whole well its camera sits over.
        fov = self._fov_slider.fov if (self._fov_mode and self._fov_slider is not None) else None
        if fov is not None:
            channels, fov = self._crop_channels_to_fov(channels, region, int(fov))
        what = f"{region} · {op}" if fov is None else f"{region} fov {fov} · {op}"
        title = f"Save a PNG of {what}"
        try:
            shape = tuple(full_res_level(channels[0].data).shape)   # metadata, no decode
            if max(int(shape[-2]), int(shape[-1])) > PNG_MAX_PX:
                title += f" (long side capped at {PNG_MAX_PX} px)"
        except Exception:                        # noqa: BLE001 - the cap note is best-effort
            pass
        src = getattr(self._reader, "source_id", None)
        acq = Path(str(src)).name if src else region
        stem = f"{acq}_{op}" if fov is None else f"{acq}_{region}_fov{fov}_{op}"
        path, _ = QFileDialog.getSaveFileName(self, title, f"{stem}.png",
                                              "PNG image (*.png)")
        if not path:
            return
        if not str(path).lower().endswith(".png"):
            path = f"{path}.png"

        from squidxplorer._workers import _PngWorker

        w = _PngWorker(channels, path, parent=self)
        self._say(f"png: rendering {what} at full resolution to {path}…")
        _launch_worker(
            self, w, slot="_png_worker",
            on_done=self._on_png_done,
            on_problem=self._on_png_failed,
            on_finished=lambda: self._forget_png_worker(w))

    def _crop_channels_to_fov(self, channels: list, region: str, fov: int) -> tuple:
        """Crop each channel's data to one field's box, for a FOVs view's export.

        Returns ``(channels, fov)``; ``fov`` comes back None when the geometry cannot
        answer, and the caller exports the whole region under the plain name instead.
        """
        from squidxplorer._mosaic_source import mosaic_bbox_um
        from squidxplorer._napari_view import pyramid_levels

        box = ((getattr(self, "_fov_boxes_cache", None) or {}).get(int(fov))
               or self._fov_boxes().get(int(fov)))
        region_bbox = mosaic_bbox_um(self._meta or {}, region)
        if box is None or region_bbox is None:
            return channels, None
        out = []
        for c in channels:
            levels = pyramid_levels(c.data) or [c.data]
            cut = _crop_levels_to_bbox(levels, region_bbox, box)
            if cut is None:
                return channels, None
            out.append(c._replace(data=cut[0]))
        return out, int(fov)

    def _forget_png_worker(self, worker) -> None:
        if self._png_worker is worker:
            self._png_worker = None
        try:
            worker.deleteLater()
        except RuntimeError:
            pass

    def _on_png_done(self, path: str, width: int, height: int, step: int, seconds: float) -> None:
        from squidxplorer._png import PNG_MAX_PX

        size_mb = 0.0
        try:
            size_mb = Path(path).stat().st_size / 1e6
        except OSError:
            pass
        note = ("" if int(step) <= 1 else
                f" Long side capped at {PNG_MAX_PX} px: decimated {int(step)}x from native.")
        self._say(f"png: {width}x{height} px -> {path} ({size_mb:.1f} MB) in {seconds:.1f}s.{note}")

    def _on_png_failed(self, reason: str) -> None:
        self._say(f"png export failed: {reason}")

    def _on_cursor_position(self, _event=None) -> None:
        """Name the FOV under the cursor, on the canvas, as it crosses a seam."""
        pane = self._pane
        model = getattr(getattr(pane, "mosaic", None), "_model", None) if pane is not None else None
        if model is None:
            return
        try:
            pos = tuple(model.cursor.position or ())
            if len(pos) < 2:
                return
            y_um, x_um = float(pos[-2]), float(pos[-1])
            region = self.current_region()
            if not region:
                return
            from squidxplorer._mosaic_source import fov_at_point

            fov = fov_at_point(self._meta or {}, region, x_um, y_um)
            model.text_overlay.text = (
                f"{region} · FOV {fov}" if fov is not None else f"{region} · off-mosaic")
        except Exception:                                 # noqa: BLE001 - a readout, never fatal
            return

    def _preview_view_operator(self) -> None:
        """PREVIEW: the selected operator on THIS view's regions; nothing is written."""
        regions, windows = self._run_scope()
        self._launch_operator(save=False, regions=regions, windows=windows)

    def _run_plate_operator(self) -> None:
        """RUN ON PLATE: the one SAVE path - the plate selection (or the whole plate), to disk.

        This is also the bulk path (the right-edge dock's cards are retired): with an
        acquisition SET loaded and the plate's bulk box ticked, the save goes over every
        member, which needs the run launched without a requester (`run_over_set`'s contract).
        """
        plate = self._plate()
        bulk = bool(getattr(plate, "_acq_set", None)) and bool(
            getattr(getattr(plate, "_bulk_all_box", None), "isChecked", lambda: False)())
        self._launch_operator(save=True, regions=None, requester=None if bulk else self)

    def _volume_preview_refusal(self, key: str, regions, windows) -> Optional[str]:
        """Ruling aa: the sentence refusing a 3D tab's preview whose FULL-DEPTH result would
        not fit this machine's display budget, decided from the geometry alone BEFORE a
        plane is read; None when it fits, reduces z, or is a region operator's own fusion."""
        from squidxplorer import is_region_operator
        from squidxplorer._engine import operator_reduces_depth
        from squidxplorer._operations import operator_name

        name = operator_name(str(key))
        try:
            if is_region_operator(name) or operator_reduces_depth(name):
                return None
        except Exception:                        # noqa: BLE001 - an unknown key: the run says it
            return None
        meta = self._meta or {}
        nz = int(meta.get("n_z") or 1)
        if nz <= 1:
            return None
        fh, fw = (int(v) for v in meta.get("frame_shape") or (0, 0))
        n_ch = len(meta.get("channels") or [])
        itemsize = int(np.dtype(meta.get("dtype") or "uint16").itemsize)
        per_region = meta.get("fovs_per_region") or {}
        if isinstance(regions, dict):
            scope = {str(r): list(f) for r, f in regions.items()}
        else:
            here = self.current_region()
            scope = {str(here): list(per_region.get(here) or [])} if here else {}
        n_fovs, px_total = 0, 0
        for r, fovs in scope.items():
            for f in fovs:
                n_fovs += 1
                w = (windows or {}).get((r, f))
                px_total += (w[1] - w[0]) * (w[3] - w[2]) if w else fh * fw
        need = px_total * nz * n_ch * itemsize
        budget = _volume_view._brick_budget_bytes()
        if need <= budget:
            return None
        return (f"3D preview over {n_fovs} FOV(s) x {nz} planes x {n_ch} channel(s) needs "
                f"~{need / 1e9:.1f} GB, over this machine's {budget / 1e9:.1f} GB display "
                "budget; draw an ROI (the box plus a halo is what gets solved).")

    def _launch_operator(self, *, save: bool, regions, requester="self", windows=None) -> None:
        """The one launch: Preview and Run on plate differ only in scope and `save`."""
        if self._run_operator is None:
            self._say("the operator engine isn't connected to this window.")
            return
        key = self._op_combo.currentData() if getattr(self, "_op_combo", None) is not None else None
        if not key:
            self._say("no operator selected.")
            return
        # A VOLUME view's preview delivers the whole depth (ruling aa), refused by name when
        # the result would not fit the display budget, BEFORE a plane is read.
        deliver_depth = bool(not save and self._render_mode == "3d")
        if deliver_depth:
            refusal = self._volume_preview_refusal(key, regions, windows)
            if refusal:
                self._say(refusal)
                return
        try:
            kwargs = dict(self._plate_operator_kwargs(key))
        except ValueError as exc:                # a refused panel setting: SAY it, run nothing
            self._say(str(exc))
            return
        kwargs.update(self._z_kwargs_for_mode(key, kwargs))
        # Whether THIS run leaves a disk artifact: the save box, or a copy-saving operator's own
        # `copy` kwarg riding through from its panel. Read by `operator_done`, which must say
        # how to GET the artifact when a copy-saving preview run left none (Julio, 2026-08-19:
        # "Registering the wells doesn't do anything").
        self._op_run_wrote = bool(save or (kwargs or {}).get("copy"))
        if requester == "self":
            requester = self
        try:
            log.info("view %s running %s on %s with %s", self.window_id, key,
                     ("the plate scope" if regions is None else
                      (regions if isinstance(regions, dict) else list(regions))), kwargs)
            # EVERY preview is the full solve, the same computation a save writes (Julio,
            # 2026-08-26: "Make the 2D preview show plane of the 3D solve"); the tab decides
            # what is DISPLAYED: a 2D tab the plane it is on (z_level, clamped by the
            # worker), a 3D tab the volume (deliver_depth).
            self._run_operator(key, regions=regions, save=save, requester=requester,
                               operator_kwargs=kwargs, z_level=self._z_slider_index(),
                               windows=windows, deliver_depth=deliver_depth)
            # No echo: the log.info line above is the structured twin, and the banner
            # strip that showed the echo is retired (2026-08-25).
        except Exception as exc:                          # noqa: BLE001 - named to the window
            self._say(f"could not start {self._op_combo.currentText()}: {exc}")

    def operator_started(self, action: str) -> None:
        """A run THIS window asked for has begun. Opens the console's started/done pair."""
        self._op_action = str(action)
        self._op_address = self.address()
        self.log.started(self._op_action, address=self._op_address)
        self._show_progress(None)

    def operator_progress(self, report) -> None:
        """The run advanced. ``report`` is a :class:`~squidxplorer._progress.ProgressReport`."""
        bar = getattr(self, "_op_progress", None)
        if bar is None or self._op_action is None:
            return                               # no run open: a late report shows no bar
        percent = report.percent
        if percent is None:
            self._show_progress(None, report.sentence())
            return
        bar.setRange(0, 100)
        bar.setValue(int(percent))
        bar.setFormat(report.sentence())
        bar.show()

    #: Whether the run this window last asked for leaves a disk artifact; see _launch_operator.
    _op_run_wrote = False

    def operator_done(self, action: str, seconds: float) -> None:
        """Emit the console's ``done`` line, closing the started/done pair."""
        self.log.done(str(action), float(seconds), address=self._closing_address())
        # The done line's console twin is log.done above; the artifact hint had ONLY the
        # retired banner, so it goes to the log itself (nothing said may be lost).
        hint = self._preview_artifact_hint(str(action))
        if hint:
            self._say(hint)
        self._op_action = self._op_address = None
        self._hide_progress()

    def _preview_artifact_hint(self, action: str) -> str:
        """One line naming the artifact a COPY-SAVING operator's PREVIEW run did not write.

        Measured complaint (Julio, 2026-08-19): register ran with save unchecked, the preview
        layer landed, no stitched_ copy appeared, and nothing said how to get one. Empty for a
        run that wrote (save box or a ``copy=True`` kwarg) and for every other operator.
        """
        if self._op_run_wrote:
            return ""
        try:
            from squidxplorer._engine import operator_saves_copy

            if not operator_saves_copy(action):
                return ""
            src = getattr(self._reader, "source_id", None)
            acq = Path(str(src)).name if src else "<folder>"
            return (f"{action}: preview only - tick save to write stitched_{acq} "
                    "(hardlinked copy with registered coordinates).")
        except Exception:                            # noqa: BLE001 - a hint, never a crash
            return ""

    def operator_failed(self, action: str, reason: str) -> None:
        """The failure outcome: an action that starts and then says nothing looks like one still running."""
        self.log.failed(str(action), str(reason), address=self._closing_address())
        self._op_action = self._op_address = None
        self._hide_progress()

    def _show_progress(self, percent, text: str = "working…") -> None:
        """Put the bar up. ``percent=None`` = INDETERMINATE (Qt's own busy sweep, range 0..0)."""
        bar = getattr(self, "_op_progress", None)
        if bar is None:
            return
        if percent is None:
            bar.setRange(0, 0)
        else:
            bar.setRange(0, 100)
            bar.setValue(int(percent))
        bar.setFormat(text)
        bar.show()

    def _hide_progress(self) -> None:
        """Take the bar down. Called from both terminal callbacks, so it cannot be left running."""
        bar = getattr(self, "_op_progress", None)
        if bar is None:
            return
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.hide()

    def _closing_address(self):
        """The address the OPEN half of the pair was written with, so the two lines agree."""
        return self._op_address if self._op_address is not None else self.address()

    def deliver_result(self, op: str, result, *, visible: bool) -> int:
        """Add one operator's :class:`~squidxplorer._result.Result` to THIS window's layer stack."""
        pane = self._pane
        mosaic = getattr(pane, "mosaic", None) if (pane is not None
                                                   and getattr(pane, "ok", False)) else None
        if mosaic is None:
            return 0
        region = str(result.region_id)
        if region != self.current_region():
            log.debug("[%s] %s result for %s not shown here: this window is on %s",
                      self.window_id, op, region, self.current_region())
            return 0
        from squidxplorer._mosaic_source import mosaic_bbox_um
        from squidxplorer._napari_pane import _colormap_for

        preview_bbox = mosaic_bbox_um(self._meta, region)
        dz = (self._meta or {}).get("dz_um")
        added = 0
        for channel in result.channels:
            plane = result.plane(channel)
            if int(result.z_depth) <= 1 and getattr(plane, "ndim", 0) == 3 and plane.shape[0] == 1:
                plane = plane[0]
            placement = getattr(plane, "placement", None)
            # THE RESULT'S OWN FOOTPRINT, never the whole region's: a scoped run's pixels
            # cover only its FOVs (2026-08-25), so cropping them against the region's mosaic
            # bbox took the wrong window and the ROI child gained a full-field layer (Julio:
            # "decon layer is != raw view"). The whole-region bbox is the fallback only for
            # a result that declares none.
            result_bbox = getattr(getattr(result, "extent", None), "bbox_um", None)
            bbox = region_bbox = (placement.bbox_um if placement is not None
                                  else (result_bbox or preview_bbox))
            # A copy-saving operator's look is a PASTE at solved positions, so it can be served
            # like raw: the on-demand pyramid with the registered positions substituted — full
            # native resolution under zoom. A FUSED result (stitch) is never substituted: its
            # pixels are the result.
            data, multiscale = plane, None
            if placement is not None and getattr(placement, "fovs", None):
                pyr = self._registered_pyramid(str(op), placement, region, channel)
                if pyr is not None:
                    data, multiscale = pyr, True
            if self._roi_bbox is not None and region_bbox is not None:
                cropped = _crop_levels_to_bbox(
                    data if multiscale else [data], region_bbox, self._roi_bbox)
                if cropped is None:
                    continue
                levels, bbox = cropped
                data = levels if multiscale else levels[0]
            try:
                mosaic.add_result(
                    result.kind, str(op), channel, data,
                    colormap=_colormap_for(channel, (self._meta or {}).get("channels")),
                    bbox_um=bbox,
                    z_scale_um=(dz if int(result.z_depth) > 1 else None),
                    visible=bool(visible),
                    multiscale=multiscale,
                )
            except Exception as exc:             # noqa: BLE001 - named, never a missing layer
                self._say(f"{op}: the {channel} layer could not be added: {exc}")
                continue
            added += 1
        if added:
            self._result_region = region
            seen = getattr(self, "_delivered_ops", None)
            if seen is None:
                seen = self._delivered_ops = set()
            if str(op) not in seen:
                seen.add(str(op))
                fit = getattr(mosaic, "reset_view", None)
                if callable(fit):
                    fit()
            if bool(visible) and int(result.z_depth) > 1 and self._render_mode == "3d":
                self._show_result_volume(str(op))
        return added

    def _show_result_volume(self, op: str) -> None:
        """Ruling aa: a full-depth result that landed in a VOLUME view is rendered as this
        view's bricked volume, in place of whatever volume was up, under the result layers'
        own LUTs (`_volume_view.volume_source` reads the bricks off the *op* layers)."""
        mosaic = getattr(self._pane, "mosaic", None) if self._pane is not None else None
        if mosaic is None:
            return
        try:
            _volume_view.close_native3d(self)        # give the layers their identities back
            mosaic.show_op(op)                       # so the volume source is THIS result
            _volume_view.open_3d(self)
        except Exception as exc:                     # noqa: BLE001 - named, never a flat layer
            self._say(f"{op}: the result landed as layers but could not be rendered as a "
                      f"volume: {exc}")

    def _registered_pyramid(self, op: str, placement, region, channel):
        """Fine-to-native pyramid at *placement*'s solved positions, or None to keep the paste.

        Only for a copy-saving operator (operator_saves_copy — its look IS a paste of raw
        frames, so re-serving them at the registered positions is the same pixels at full
        depth of zoom) over a single-z acquisition; anything else keeps the delivered array.
        """
        from squidxplorer._engine import operator_saves_copy
        from squidxplorer._mosaic_source import fuse_region_pyramid

        if (self._reader is None or self._meta is None
                or int(self._meta.get("n_z") or 1) != 1
                or not operator_saves_copy(str(op))):
            return None
        try:
            meta2 = dict(self._meta)
            pos = dict(meta2.get("fov_positions_um") or {})
            pitch = float(placement.pixel_size_um)
            for f, (oy, ox) in zip(placement.fovs, placement.origins_px):
                pos[(str(region), int(f))] = (
                    placement.origin_um[1] + float(ox) * pitch,
                    placement.origin_um[0] + float(oy) * pitch,
                )
            meta2["fov_positions_um"] = pos
            res = fuse_region_pyramid(self._reader, meta2, str(region), str(channel),
                                      time_point=int(placement.reg_t or 0))
        except Exception as exc:                 # noqa: BLE001 - the paste is a fine fallback
            log.warning("[%s] %s: registered pyramid not built (%s) - keeping the paste.",
                        self.window_id, op, exc)
            return None
        return res[0] if res else None

    def _drop_result_layers(self, why: str) -> None:
        """Drop every operator layer in this window, keeping ``raw``."""
        pane = self._pane
        mosaic = getattr(pane, "mosaic", None) if (pane is not None
                                                   and getattr(pane, "ok", False)) else None
        if mosaic is None:
            self._result_region = None
            return
        gone = [op for op in list(mosaic.ops()) if op != _RAW_OP]
        for op in gone:
            try:
                mosaic.remove_op(op)
            except Exception:                    # noqa: BLE001 - best effort per group
                pass
        self._result_region = None
        if gone:
            self._say(f"dropped the {', '.join(gone)} layer(s): {why}")

    # (The Detect-nuclei surface — _spot_source, _detect_nuclei, _on_nuclei_ready and
    # the pane's Detect row — was shelved 2026-08-24 with the spot/cellpose operators.)

    def _napari_viewer(self):
        """The live napari viewer (or headless ``ViewerModel``) behind this window's pane.

        One branch: the pane's ``_viewer`` IS ``mosaic.model`` on every real pane (the mosaic is
        constructed over that very viewer), so the old model-then-``_viewer`` two-step answered
        the same object twice — the second step existed only for a deleted test stub whose
        ``model`` was None.
        """
        pane = self._pane
        if pane is None or not getattr(pane, "ok", False):
            return None
        return getattr(pane, "_viewer", None)

    def _focus_reference_plane(self) -> None:
        """Jump the z-slider to the sharpest plane (Tenengrad) of the current region's centre FOV."""
        v = self._napari_viewer()
        region = self._cursor.region if self._cursor is not None else (
            self._regions[0] if self._regions else None)
        if v is None or region is None or self._reader is None or self._meta is None:
            self._say("show a region in this view first, then focus the reference plane.")
            return
        z_levels = list((self._meta.get("z_levels") or []))
        if len(z_levels) <= 1:
            self._say(f"{region}: this acquisition has a single z plane, so there is no "
                      "reference plane to find.")
            return
        if self._focus_worker is not None and self._focus_worker.isRunning():
            self._say("already finding the reference plane…")
            return
        from squidxplorer._napari3d import _center_fov
        fov = _center_fov(self._meta, region)
        if fov is None:
            fovs = (self._meta.get("fovs_per_region") or {}).get(region) or [0]
            fov = int(fovs[0])
        chan = self._meta["channels"][0]["name"]
        from squidxplorer._workers import _FocusWorker

        w = _FocusWorker(self._reader, self._meta, region, int(fov), chan, parent=self)
        self._say("finding the sharpest z (Tenengrad autofocus)…")
        _launch_worker(
            self, w, slot="_focus_worker",
            on_problem=self._say,
            signals={"ready": lambda z_i, note: self._on_reference_plane(int(z_i), note)})

    def _on_reference_plane(self, z_index: int, note: str) -> None:
        """The sharpest plane is known. MOVE THIS WINDOW'S z SLIDER to it, or say why not."""
        v = self._napari_viewer()
        if v is None:
            return
        try:
            dims = v.dims
            nsteps = tuple(int(n) for n in (getattr(dims, "nsteps", ()) or ()))
            if len(nsteps) < 3 or nsteps[0] < 2:
                self._say(f"sharpest plane is z={z_index}, but no z slider could be moved - "
                          "this view is showing a single plane.")
                return
            step = list(dims.current_step)
            step[0] = max(0, min(int(z_index), nsteps[0] - 1))
            dims.current_step = tuple(step)
            self._say(f"reference plane: z={z_index}. {note}".strip())
        except Exception as exc:                         # noqa: BLE001 - named, never silent
            self._say(f"could not move the z-slider: {exc}")

    # -- the ROI cluster lives in `_roi_tools` (clamp-at-draw, acquisition-pixel costing and the
    # -- child windows move intact); thin delegates because tests and the ROI chips actuate these
    # -- by name on the window. -------------------------------------------------------------------
    @staticmethod
    def _sync_roi_width(viewer, layer, screen_px: float = 3.0) -> None:
        _roi_tools.sync_roi_width(viewer, layer, screen_px)

    def _roi_shapes_layer(self, create: bool = False):
        return _roi_tools.roi_shapes_layer(self, create)

    #: The FOVs chip's own description, held apart from the per-region count appended to it. The
    #: count changes whenever the plate navigates this window somewhere else, so the two are
    #: joined at refresh time rather than the chip's tooltip being re-derived from its own text.
    _FOVS_TIP = "Step through this region's FOVs one at a time."

    def _refresh_fovs_chip(self) -> None:
        """Enable the FOVs chip only when there are fields to walk, and SAY WHY when there are not.

        Three separate refusals, kept separate because they have different fixes -- the pattern
        ``_refresh_record_chip`` sets. Unlike that one this must be re-run on a REGION change:
        ``n_t`` and ``n_z`` are properties of the acquisition, but how many FOVs there are is a
        property of the region, and the plate can navigate this window to another one.
        """
        btn = getattr(self, "_btn_fovs", None)
        if btn is None:
            return
        if self._fov_mode:
            btn.setEnabled(False)
            btn.setToolTip("This view already steps through FOVs - use the slider at the bottom.")
            return
        if self._manager is None:
            btn.setEnabled(False)
            btn.setToolTip("This window has no manager, so it cannot open a child view.")
            return
        # A STITCHED mosaic has no honest FOV boxes to draw (Julio, 2026-08-18: "stitched mosaic
        # should not show fov boundaries"): registration moved the fields off the preview
        # placement `mosaic_fov_bboxes_um` describes, so the boxes would sit beside the pixels
        # they claim to name — the plausible-wrong-geometry failure. Declaration-derived
        # (`is_region_operator`), never an operator-name comparison.
        try:
            from squidxplorer import is_region_operator
            from squidxplorer._operations import operator_name

            pane = self._pane
            mosaic = getattr(pane, "mosaic", None) if pane is not None else None
            shown = mosaic.visible_op() if mosaic is not None else None
            if shown and is_region_operator(operator_name(str(shown))):
                btn.setEnabled(False)
                btn.setToolTip(
                    "This window is showing a stitched mosaic: its fields are registered, so the "
                    "preview FOV boxes would not sit on the pixels they name. Show the raw layer "
                    "to walk FOVs.")
                return
        except Exception:                            # noqa: BLE001 - a chip verdict, never fatal
            pass
        from squidxplorer._mosaic_source import mosaic_fov_bboxes_um

        region = self.current_region()
        try:
            n = len(mosaic_fov_bboxes_um(self._meta or {}, region)) if region else 0
        except (KeyError, ValueError, TypeError) as exc:
            btn.setEnabled(False)
            btn.setToolTip(str(exc))
            return
        if not n:
            btn.setEnabled(False)
            btn.setToolTip(f"{region} has no locatable FOVs.")
            return
        btn.setEnabled(True)
        btn.setToolTip(f"{self._FOVS_TIP}\n\n{n} FOV(s) in {region}.")

    def _open_fovs(self) -> None:
        """Open a child view that walks THIS region's FOVs. One region, one child, one tab."""
        if self._manager is None:
            self._say("this window isn't attached to the view registry, so it cannot open a child.")
            return
        region = self.current_region()
        if not region:
            self._say("no region is showing, so there are no FOVs to walk.")
            return
        boxes = self._fov_boxes()          # says its own reason if it cannot answer
        if not boxes:
            return
        child = self._manager.open_child([region], parent_id=self.window_id, fovs=True,
                                         luts=self._per_channel_luts())
        if child is None:
            self._say("the FOV view could not be opened.")
            return
        self._say(f"walking {len(boxes)} FOV(s) of {region} in view {child.window_id}.")

    # -- the FOV walk: every field of this region, drawn, and the camera stepped across them ----
    #: Edge colours for the FOV rectangles. Two, because there are exactly two states a field can
    #: be in on this surface, and naming them here is what stops "what does current look like"
    #: being answered in two places.
    _FOV_EDGE_IDLE = "#8b949e"
    _FOV_EDGE_CURRENT = "#f0883e"

    def _fov_boxes(self) -> "dict":
        """``{fov: (x0, y0, x1, y1)}`` for the current region, or ``{}`` having SAID why not.

        One geometry, and it is the one that placed the pixels:
        :func:`squidxplorer._mosaic_source.mosaic_fov_bboxes_um`. It raises rather than returning
        a short answer precisely so this can say the reason out loud -- fifteen rectangles out of
        sixteen look exactly as convincing as sixteen, so a silent gap here is a picture of a
        region with a hole in it.
        """
        from squidxplorer._mosaic_source import mosaic_fov_bboxes_um

        region = self.current_region()
        if not region:
            return {}
        try:
            # The crossing is NAMED: the walk's geometry is TopLeftBoxUm, the drawing code
            # takes plain corner tuples.
            return {f: b.bbox()
                    for f, b in mosaic_fov_bboxes_um(self._meta or {}, region).items()}
        except (KeyError, ValueError, TypeError) as exc:
            self._say(f"cannot locate this region's FOVs: {exc}")
            return {}

    def _fov_shapes_layer(self, boxes: "dict"):
        """This window's FOV Shapes layer, rebuilt from *boxes* in ONE write.

        DELIBERATELY NOT ``_roi_shapes_layer``. Sharing it looks tidier and would draw a rectangle
        that lies, in four separate ways, all of them via ``_on_roi_data``:

        * ``_clamp_last_roi`` holds the LAST shape to ``GL_MAX_3D_TEXTURE_SIZE``. That ceiling is
          2048 px on the Apple floor and a 40x frame here is 4168 px, so the last field's
          rectangle would be silently shrunk and the user told "ROI held to the 3D ceiling" about
          a box they never drew. That clamp is a promise about a box the USER dragged.
        * it renames the next shape ``R{n+1}``, so opening a FOVs view would renumber the user's
          next ROI to R17.
        * it prints ``_roi_cost_line`` -- a brick estimate for a field nobody boxed.
        * ``_clear_rois`` ("Remove all ROIs in this window") would wipe the fields, and
          ``_open_roi_children``'s "the most recently drawn one" fallback would open FOV 15.

        Separate layer, and every one of those sentences stays literally true of the ROI layer.

        Colours are set DIRECTLY rather than through a property cycle (which is what the ROI layer
        uses): the current-field highlight writes ``edge_color`` itself, and two colour rules on
        one layer is how a highlight comes back wrong after an unrelated property write.
        """
        v = self._napari_viewer()
        if v is None or not boxes:
            return None, None
        layer = self._fov_layer
        if layer is None or layer not in list(v.layers):
            try:
                layer = v.add_shapes(name="FOVs", face_color="transparent",
                                     edge_color=self._FOV_EDGE_IDLE)
                layer.editable = False
                from squidxplorer._napari_view import MosaicLayers
                MosaicLayers._label_units(layer)     # stage um: napari >= 0.7 nulls mixed units
            except Exception as exc:                     # noqa: BLE001 - named, never fatal
                self._say(f"could not draw the FOV boxes ({type(exc).__name__}: {exc}).")
                return v, None
            self._fov_layer = layer
            try:                                         # border reacts to zoom, like the ROI one
                v.camera.events.zoom.connect(
                    lambda e=None, vv=v, ly=layer: self._sync_roi_width(vv, ly))
            except Exception:                            # noqa: BLE001
                pass
        # ONE write for the whole region. Sixteen appends would be sixteen events, sixteen
        # re-triangulations and (on a shared layer) sixteen clamp passes.
        rects = [np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)
                 for (x0, y0, x1, y1) in (boxes[f] for f in boxes)]
        try:
            layer.data = rects
            layer.text = {"string": [f"fov {f}" for f in boxes],
                          "color": self._FOV_EDGE_IDLE, "size": 8, "anchor": "upper_left"}
        except Exception as exc:                         # noqa: BLE001 - named, never fatal
            self._say(f"could not draw the FOV boxes ({type(exc).__name__}: {exc}).")
            return v, None
        self._sync_roi_width(v, layer)
        return v, layer

    def _highlight_fov(self, index: int) -> None:
        """Make exactly one FOV rectangle read as current, by COLOUR.

        Deliberately not ``edge_width``: napari's setter broadcasts a scalar across every shape,
        and ``_sync_roi_width`` assigns exactly such a scalar on every ``camera.events.zoom``. A
        width-based highlight would therefore survive until the first wheel click and then vanish
        — intermittently, which is worse than never having worked.
        """
        layer = self._fov_layer
        if layer is None:
            return
        try:
            n = len(getattr(layer, "data", []) or [])
            if n == 0:
                return
            colors = [self._FOV_EDGE_IDLE] * n
            if 0 <= int(index) < n:
                colors[int(index)] = self._FOV_EDGE_CURRENT
            layer.edge_color = colors
        except Exception:                                # noqa: BLE001 - the highlight is cosmetic
            pass

    def _draw_fov_boxes(self) -> None:
        """Draw every FOV of the current region and size the slider to them. Once per region."""
        slider = self._fov_slider
        if slider is None:
            return
        boxes = self._fov_boxes()
        # Held so a STEP costs no geometry at all. Rebuilt whenever the region is redrawn, which
        # is the only thing that can invalidate it -- the boxes are a property of the region, and
        # the region is the one thing a redraw is triggered by.
        self._fov_boxes_cache = boxes
        slider.set_fovs(list(boxes))
        if not boxes:
            return
        self._fov_shapes_layer(boxes)
        self._highlight_fov(slider.index)

    def _on_fov_changed(self, index: int, fov: int) -> None:
        """Point the camera at one field. NO LOAD HAPPENS HERE, and that is the whole design.

        The region's mosaic is already resident and lazy, so framing a field costs a camera write
        and napari materialises only the tiles that field covers.
        """
        boxes = getattr(self, "_fov_boxes_cache", None) or {}
        box = boxes.get(int(fov))
        if box is None:
            box = self._fov_boxes().get(int(fov))
        if box is None:
            self._frame_fov_done()
            return
        pane = self._pane
        if pane is not None and getattr(pane, "ok", False):
            said = pane.mosaic.frame_bbox_um(box)
            if said:
                self._say(said)
        self._highlight_fov(int(index))
        self._frame_fov_done()

    def _frame_fov_done(self) -> None:
        """Open the FOV axis's playback gate.

        THE ONE PLACE it can honestly be opened for this axis, and deliberately NOT inside
        ``_frame_done``. That method exists because a region step and a timepoint step both wait
        on a mosaic load, so one arrival opens both of their gates. A FOV step waits on nothing --
        the frame IS the camera write that just happened -- and routing it through ``_frame_done``
        would let a timepoint reload, or a RETIRED load, advance a FOV animation.

        Without this the axis advances exactly one frame and then sits until the 180 s stall
        watchdog fires: napari's ``QtDims._set_frame`` closes the gate on every step and only a
        canvas draw reopens it, and ``AxisPlayback`` drives a ``Dims`` with no canvas behind it.
        """
        slider = self._fov_slider
        if slider is not None:
            slider.frame_done()

    _auto_worker = None

    def _reset_contrast_off_thread(self, channel: str, sample) -> None:
        """napari's once button for one channel: the window rule over the displayed slice,
        computed on a worker (9.3 ms measured on a 2050^2 frame: not free on the Qt thread),
        landed through set_contrast so every surface of the identity agrees."""
        from squidxplorer._qthread_life import detach
        from squidxplorer._workers import _AutoContrastWorker

        old = self._auto_worker
        if old is not None and old.isRunning():
            detach(old)
        worker = _AutoContrastWorker({str(channel): sample}, parent=self)
        worker.done.connect(self._apply_auto_contrast)
        worker.problem.connect(lambda why: self._say(f"auto-contrast: {why}"))
        self._auto_worker = worker
        worker.start()

    def _apply_auto_contrast(self, windows) -> None:
        pane = self._pane
        mosaic = getattr(pane, "mosaic", None) if pane is not None else None
        if mosaic is None or not windows:
            return
        with mosaic.programmatic():
            for channel, (lo, hi) in dict(windows).items():
                try:
                    mosaic.set_contrast(channel, lo, hi)
                except KeyError:
                    continue
        log.debug("view %s auto-contrast landed for %s", self.window_id, sorted(windows))

    #: The ROI chip's two faces: draw a box, or go to the box just drawn.
    _ROI_DRAW = ("▭ ROI", "Draw an ROI rectangle inside the mosaic.")
    _ROI_GO = ("→ ROI", "Open the drawn ROI as a child view.")
    _roi_count = 0
    _roi_used = False

    def _on_roi_data(self, layer) -> None:
        _roi_tools.on_roi_data(self, layer)
        n = len(list(getattr(layer, "data", []) or []))
        if n > self._roi_count:
            self._roi_used = False               # a NEW box: the arrow is offered again
        self._roi_count = n
        self._refresh_roi_chip()

    def _refresh_roi_chip(self) -> None:
        """The ROI chip reads as the go-to arrow while an unused box exists, else as draw."""
        btn = getattr(self, "_btn_roi", None)
        if btn is None or not _alive(btn):
            return
        text, tip = self._ROI_GO if (self._roi_count and not self._roi_used) else self._ROI_DRAW
        btn.setText(text)
        btn.setToolTip(tip)

    def _roi_chip_clicked(self) -> None:
        if self._roi_count and not self._roi_used:
            self._open_roi_children()
            return
        self._new_roi()

    def _live_texture_limit(self) -> int:
        return _roi_tools.live_texture_limit(self)

    def _clamp_last_roi(self, layer) -> None:
        """Hold the just-drawn ROI to the live 3D ceiling. See `_roi_tools.clamp_last_roi`."""
        _roi_tools.clamp_last_roi(self, layer)

    def _roi_cost_line(self, layer) -> str:
        return _roi_tools.roi_cost_line(self, layer)

    def _new_roi(self) -> None:
        _roi_tools.new_roi(self)

    def _select_rois(self) -> None:
        _roi_tools.select_rois(self)

    def _clear_rois(self) -> None:
        _roi_tools.clear_rois(self)

    def _region_for_roi(self, bbox) -> Optional[str]:
        return _roi_tools.region_for_roi(self, bbox)

    def _open_roi_children(self) -> None:
        _roi_tools.open_roi_children(self)
        self._roi_used = True                    # used: the chip hands back to drawing
        self._refresh_roi_chip()

    def _install_canvas_loupe(self, pane) -> None:
        """Give this window's canvas a shift-left-click magnifier.

        Refuses by name rather than half-installing: a loupe with no GL widget to sit on would be
        a gesture that swallows shift-clicks and shows nothing.
        """
        from squidxplorer._napari_loupe import CanvasLoupe

        viewer = self._napari_viewer()
        canvas = getattr(pane, "canvas_widget", None)
        if viewer is None or canvas is None:
            log.debug("view %s: no canvas widget, so no loupe", self.window_id)
            return
        source_for = getattr(self._manager, "loupe_source_for", None) if self._manager else None
        if source_for is None:
            log.debug("view %s: no loupe source registry, so no loupe", self.window_id)
            return
        self._loupe = CanvasLoupe(
            viewer=viewer, canvas_widget=canvas, meta=self._meta or {},
            source_for=source_for, mosaic=pane.mosaic,
            region_of=self.current_region,
            time_point_of=lambda: self.time_point,
            look_of=self._screen_look,
            say=self._say, parent=self)

    def _screen_look(self) -> "tuple[list, Any, list, list]":
        """WHAT IS ON SCREEN IN THIS WINDOW, as ``(names, colors, windows, mask)``.

        One harvest of the three facts anything rendering "the same picture as the canvas" needs:
        the contrast window per channel, the colour each channel is tinted with RIGHT NOW, and
        which channels are actually visible. All three come from the methods that already own
        them -- :meth:`_per_channel_luts`, :meth:`_visible_channels` and
        ``_video._channel_colors`` -- so this is a harvest, not a fourth opinion.

        In ``meta["channels"]`` ORDER, and covering every channel including hidden ones, because
        that is the axis order the loupe sources return their ``(C, y, x)`` crops on. Re-indexing
        at the consumer is how a crop ends up composited with another channel's colour.

        A ``window`` of ``None`` means "this window has no opinion about that channel"; the caller
        falls back to the source's own. That fallback direction matters: a loupe is a magnifier OF
        THE SURFACE IT SITS ON, so the canvas outranks the source wherever the canvas has an
        answer -- which is the rule ``_plate_overview._loupe_lut`` states for the plate.
        """
        from squidxplorer._video import _channel_colors

        names = [c["name"] for c in (self._meta or {}).get("channels", [])]
        luts = self._per_channel_luts()
        visible = set(self._visible_channels())
        rgb = {n: luts[n]["rgb"] for n in names if (luts.get(n) or {}).get("rgb") is not None}
        colors = _channel_colors(self._meta or {}, names, rgb)
        windows = [(luts.get(n) or {}).get("clim") for n in names]
        mask = [n in visible for n in names]
        return names, colors, windows, mask

    def _per_channel_luts(self) -> "dict[str, dict]":
        return _lut_clipboard.per_channel_luts(self)

    def current_settings(self) -> "dict[str, Any]":
        """This window's global-default settings AS THEY ARE ON SCREEN right now."""
        out = self.settings.snapshot()
        live = self._per_channel_luts()
        if live:
            out["luts"] = live
        return out

    def _apply_luts(self, luts: "Optional[dict]") -> Optional[int]:
        """Put per-channel contrast + colormap on this window's layers. ``None`` = no mosaic here."""
        return _lut_clipboard.apply_luts(self, luts)

    def _apply_channel_visibility(self, visibility: "Optional[dict]") -> None:
        """Show/hide channels per the setting; an empty setting means no opinion and touches nothing."""
        pane = self._pane
        mosaic = getattr(pane, "mosaic", None) if pane is not None else None
        if mosaic is None:
            return
        with mosaic.programmatic():
            for name, on in (visibility or {}).items():
                try:
                    mosaic.set_channel_visible(str(name), bool(on))
                except Exception:                        # noqa: BLE001 - a missing channel is skipped
                    pass

    def _apply_settings_once(self) -> None:
        """Put this window's settings on screen, ONCE, now that its layers exist."""
        if self._settings_applied:
            return
        pane = self._pane
        mosaic = getattr(pane, "mosaic", None) if pane is not None else None
        if mosaic is None:
            return
        self._settings_applied = True
        self._apply_luts(self.settings.get("luts"))
        self._apply_channel_visibility(self.settings.get("channel_visibility"))
        self._watch_user_visibility(mosaic)
        hook = getattr(mosaic, "on_reset_contrast", None)
        if callable(hook):
            hook(self._reset_contrast_off_thread)
        if self.settings.get("tenengrad_focus"):
            try:
                self._focus_reference_plane()
            except Exception as exc:                     # noqa: BLE001 - named, never silent
                self._say(f"could not autofocus this window: {exc}")

    def _watch_user_visibility(self, mosaic) -> None:
        """Record channel visibility the user changed via napari's eye icons, marking divergence."""
        hook = getattr(mosaic, "on_user_visibility", None)
        if not callable(hook):
            return
        try:
            hook(self._on_user_visibility)
        except Exception:                                # noqa: BLE001 - the seam stays optional
            pass

    def _on_user_visibility(self, channel: str, visible: bool) -> None:
        vis = dict(self.settings.get("channel_visibility") or {})
        vis[str(channel)] = bool(visible)
        self.settings.set("channel_visibility", vis)

    def _on_depth_changed(self, channel: str, lo: float, hi: float) -> None:
        """The dataset proved it holds bigger numbers: open every slider here to reach them.

        Arrives on the GUI thread (``ViewerManager.depthChanged`` is emitted from the fuse worker
        and queued), so touching layers from here is safe.

        This moves the slider's TRAVEL and never its VALUE -- widening a range cannot clip, so
        nothing on screen changes colour. The visible effect is the handle re-scaling, which is
        the honest report that the data turned out to be bigger than the first region suggested.
        """
        pane = self._pane
        mosaic = getattr(pane, "mosaic", None) if pane is not None else None
        if mosaic is not None:
            try:
                mosaic.widen_contrast_range(str(channel), float(lo), float(hi))
            except Exception:                    # noqa: BLE001 - a slider bound is never fatal
                log.exception("could not widen this window's contrast range to (%s, %s).", lo, hi)
        # `open_native_3d*` build their layers in a napari.Viewer of their OWN, outside
        # `MosaicLayers`, so the walk above cannot reach them. A `BrickedVolume` popout needs
        # nothing here: its `_viewer` IS `mosaic.model`, so its bricks were in that same walk.
        popout = getattr(self, "_native3d", None)
        if popout is not None and hasattr(popout, "layers"):
            try:
                from squidxplorer._napari3d import widen_contrast_range

                widen_contrast_range(popout, float(lo), float(hi))
            except Exception:                    # noqa: BLE001 - ditto, and the popout may be gone
                pass

    # "Match layers to raw" is SHELVED WHOLE (Julio, 2026-08-19): the button, this window's
    # handler, `_lut_clipboard.match_raw_contrast` and `MosaicLayers.match_contrast_to` are gone.

    @property
    def time_point(self) -> int:
        """Which timepoint THIS window is showing."""
        bar = getattr(self, "_time_point_bar", None)
        return bar.time_point if bar is not None else 0

    def _on_time_point_changed(self, time_point: int) -> None:
        """A user moved THIS window's timepoint, or playback advanced it. Reload the mosaic."""
        self._say(f"time_point {time_point + 1} of {self._time_point_bar.count}")
        # THE LOUPE STAYS UP AND RE-READS at the new frame — the opposite of a region change, and
        # deliberately so: the anchor is a point in THIS region, which still exists, so dismissing
        # would throw away a valid position. The plate's `set_time_point` re-requests for exactly
        # this reason, and the bug it fixed was an inset magnifying frame 0 forever under a moving
        # label. The crop's LRU key carries the timepoint, so this is a real re-read.
        if self._loupe is not None:
            self._loupe.retarget()
        if self._time_point_bar.is_playing:
            self._load_mosaic(region=self.current_region())
            return
        if getattr(self, "_time_load_timer", None) is None:
            self._time_load_timer = QTimer(self)
            self._time_load_timer.setSingleShot(True)
            self._time_load_timer.timeout.connect(
                lambda: self._load_mosaic(self.current_region()))
        self._time_load_timer.start(_REGION_LOAD_DEBOUNCE_MS)

    def _on_region_changed(self, index: int, region: str) -> None:
        """Current region moved. Debounce the fuse; the slider label already moved instantly."""
        if getattr(self, "_load_timer", None) is None:
            self._load_timer = QTimer(self)
            self._load_timer.setSingleShot(True)
            self._load_timer.timeout.connect(
                lambda: self._load_mosaic(self._pending_region))
        self._pending_region = region
        self._load_timer.start(_REGION_LOAD_DEBOUNCE_MS)
        # How many FOVs a region has is a property of the REGION, so the chip's verdict goes stale
        # the moment the plate navigates this window somewhere else. The boxes themselves are
        # redrawn by `_on_done`, which is where the new mosaic actually lands.
        self._refresh_fovs_chip()
        # THE LOUPE GOES DOWN on a region change, and is NOT re-read. Its anchor is a world point,
        # and the next region sits at its own stage coordinates, so the same point is somewhere
        # else entirely -- or nowhere. Re-reading silently under a new region is the wrong-image
        # failure `docs/plate-contract.md` is written against. (A TIMEPOINT change is the opposite
        # case and is handled as such: see `_on_time_point_changed`.)
        if self._loupe is not None:
            self._loupe.dismiss()

    # -- the mosaic load/playback pipeline lives in `_mosaic_playback` (generation dropping and
    # -- the frame gate move intact — docs/rendering-contract.md); thin delegates because tests
    # -- drive _load_mosaic / _on_plane / _on_done by name, and _on_plane unbound over a duck. ----
    def _load_mosaic(self, region: Optional[str]) -> None:
        """Fuse one region's FOVs into this pane. See `_mosaic_playback.load_mosaic`."""
        _mosaic_playback.load_mosaic(self, region)

    def _worker_ended(self, worker) -> None:
        _mosaic_playback.worker_ended(self, worker)

    def _retire_worker(self, worker) -> None:
        _mosaic_playback.retire_worker(self, worker)

    def _forget_worker(self, worker) -> None:
        _mosaic_playback.forget_worker(self, worker)

    def _is_current_load(self, gen: int) -> bool:
        return _mosaic_playback.is_current_load(self, gen)

    def _on_plane(self, region: str, channel: str, levels, bbox_um, window=None,
                  gen: Optional[int] = None) -> None:
        _mosaic_playback.on_plane(self, region, channel, levels, bbox_um, window, gen=gen)

    def _on_done(self, region: str, n: int, gen: Optional[int] = None) -> None:
        _mosaic_playback.on_done(self, region, n, gen=gen)

    def _frame_done(self) -> None:
        _mosaic_playback.frame_done(self)

    def _selected_roi(self) -> "tuple":
        """(bbox, region) of the selected ROI, else (None, None). See `_roi_tools.selected_roi`."""
        return _roi_tools.selected_roi(self)

    def _roi_center_fov(self, region: str, bbox: Optional[tuple] = None) -> Optional[int]:
        return _roi_tools.roi_center_fov(self, region, bbox)

    # -- the 3D/volume cluster lives in `_volume_view`. The close-before-read invariant is
    # -- STRUCTURAL there: open_3d closes the old volume and dispatches into the scene-reading
    # -- paths through module-internal calls. Thin delegates because tests borrow these unbound
    # -- onto duck shells (test_stitch_in_3d) and call them by name. ------------------------------
    def _replace_native3d(self, open_it) -> None:
        """ONE 3D popout per window. See `_volume_view.replace_native3d`."""
        _volume_view.replace_native3d(self, open_it)

    def _close_native3d(self) -> None:
        """Take this window's 3D view down; idempotent. See `_volume_view.close_native3d`."""
        _volume_view.close_native3d(self)

    def _open_3d(self) -> None:
        """3D opens IN A NEW TAB (2026-08-19 mock): a sibling view over the same region carries
        the volume, so the 2D view stays exactly as it is.

        The volume machinery is untouched (`_volume_view.open_3d` runs in the CHILD): an ROI —
        this window's own crop or a drawn rectangle — travels as the child's `roi_bbox`, so the
        bricked render lands in the child's canvas; the scene facts 3D harvests (which layer,
        its LUTs) are read from THIS window, whose scene is the one on screen. With no manager
        (a library caller, tests borrowing the method) it renders in-window as before.
        """
        # getattr, not a bare read: tests borrow this method unbound onto a duck shell (the
        # `_volume_view` module docstring records the convention).
        if getattr(self, "_manager", None) is None:
            _volume_view.open_3d(self)
            return
        region = self.current_region()
        if not region:
            self._say("no region to render in 3D.")
            return
        roi_bbox = self._roi_bbox
        if roi_bbox is None:
            sel_bbox, sel_region = self._selected_roi()
            if sel_bbox is not None and sel_region is not None:
                roi_bbox, region = sel_bbox, sel_region
        child = self._manager.open_child([region], roi_bbox=roi_bbox, parent_id=self.window_id)
        if child is None:
            self._say("3D: could not open a new view for the volume; rendering here instead.")
            _volume_view.open_3d(self)
            return
        child.set_display_name(f"3D · {region}")
        child.note_volume_tab()
        _volume_view.open_3d(child, scene_from=self)

    def _on_screen_luts(self, op: str) -> "tuple[dict, dict]":
        return _volume_view.on_screen_luts(self, op)

    def _open_roi_3d(self, region: str, roi_bbox: tuple) -> None:
        _volume_view.open_roi_3d(self, region, roi_bbox)

    def _volume_source(self, window: tuple):
        """WHICH volume 3D renders. See `_volume_view.volume_source`."""
        return _volume_view.volume_source(self, window)

    def _refresh_bricks(self) -> None:
        _volume_view.refresh_bricks(self)

    def _displayed_pitch_um(self, layer, *, what: str):
        return _volume_view.displayed_pitch_um(self, layer, what=what)

    def _render_roi_volume(self, mosaic, contrast_by: dict, colormap_by: dict) -> None:
        _volume_view.render_roi_volume(self, mosaic, contrast_by, colormap_by)

    def show_region(self, region: str) -> bool:
        """Point this window at *region*, ADOPTING it when this window did not already hold it.

        The plate is a free navigator: clicking a well outside this window's original selection is
        a request to look there, not a mistake. Adopting is a re-scope of the ONE cursor, and
        ``RegionCursor.set_order`` is built for exactly this — it keeps you on the region you are
        already looking at, announcing the new order without announcing a position, so the slider
        resizes and NO second mosaic load is triggered. The single load then comes from
        ``set_region`` below, through the ordinary ``_on_region_changed`` path.

        The rendering contract is honoured with no new code: ``_load_mosaic`` already drops the
        previous region's operator layers and removes the raw layers before adding any, because
        ``_shown_region`` differs. That is the "different region -> remove first" rule, and
        navigation is now simply a new way of entering it.

        Returns False, having SAID why in this window's log, when the acquisition has no such
        region — a navigator that silently does nothing is indistinguishable from one that is
        broken.
        """
        region = str(region)
        if self._cursor is None:
            return False
        if self._cursor.region == region:
            return True                      # already here: no reload, no re-announce
        if self._cursor.position_of(region) is None:
            known = [str(r) for r in ((self._meta or {}).get("regions") or [])]
            if known and region not in known:
                self._say(f"{region} is not in this acquisition.")
                return False
            # Grow the order IN ACQUISITION ORDER rather than by appending, so the slider still
            # reads left-to-right across the plate however the user wandered there.
            want = set(self._cursor.regions) | {region}
            order = [r for r in known if r in want] or (self._cursor.regions + [region])
            self._cursor.set_order(order)
            self.regionsChanged.emit(self)
        self._cursor.set_region(region)
        return True

    @property
    def _regions(self) -> "list[str]":
        """Every region this window can REACH — the cursor's order, never a field.

        A property rather than the field it replaces, because the set is no longer fixed at
        construction: the plate can point an open window at a region it was not opened over, which
        re-scopes the cursor. Anything that cached the answer in a second field would go stale on
        that re-scope, silently, and the things that read this are the ones that would show it
        wrong — ``ViewerManager.views()`` and the ``viewFocused`` payload that paints the plate.
        Reading THROUGH the cursor means there is nothing to keep in step.

        Falls back to the seed for the window between ``__init__`` starting and the cursor being
        built (``_build`` runs several statements later, and ``_derived_name`` is computed in
        between), and ``getattr`` rather than a bare read for the documented reason the class
        defaults above exist: on a QObject whose ``__init__`` has not finished, a missing attribute
        raises out of Qt's own machinery instead of answering.
        """
        cursor = getattr(self, "_cursor", None)
        if cursor is not None and cursor.regions:
            return cursor.regions
        return list(getattr(self, "_seed_regions", []))

    def current_region(self) -> str:
        """The region this window is showing right now (it can hold several and step with its slider)."""
        region = self._cursor.region if self._cursor is not None else None
        if not region:
            region = self._regions[0] if self._regions else ""
        return str(region)

    def address(self):
        """WHERE this window is, as :class:`~squidxplorer._address.Address` or ``Extent``."""
        region = self.current_region()
        if self._roi_bbox is not None:
            return Extent(region_id=region, bbox_um=self._roi_bbox)
        return Address(region_id=region)

    def view_log(self) -> ViewLog:
        """This window's logger, addressed to wherever it is pointing at this instant."""
        return self.log.at(self.address())

    def _say(self, text: str) -> None:
        """Tell the user via the LOGGER, refusal-shaped text at WARNING (the in-window
        banner strip is retired - Julio, 2026-08-25: "That should appear in the logger");
        the collapsed log band shows the latest line. The pane's ``said`` list stays fed:
        it is the recording seam tests and gates assert on, never a pixel."""
        if not text:
            return
        from squidxplorer._logpane import status_level

        self.view_log().log(status_level(text), "%s", text)
        said = getattr(self._pane, "said", None)
        if said is not None:
            said.append(str(text))
            del said[:-500]

    def set_active(self, active: bool) -> None:
        """Halt draw/refresh on windows the user is not touching."""
        if active:
            return
        # A background tab's loupe goes down and drops its crops. The THREAD stays: it is idle and
        # cheap, and stopping/restarting it on every tab click would pay a teardown for nothing.
        # The crops are the part the deck's per-view memory line is about.
        try:
            if self._loupe is not None:
                self._loupe.dismiss()
                self._loupe.clear_cache()
        except Exception:                            # noqa: BLE001 - best effort
            pass
        for control in (self._slider, getattr(self, "_time_point_bar", None)):
            if control is None:
                continue
            try:
                if control.is_playing:
                    control.stop()
            except Exception:                        # noqa: BLE001 - best effort
                pass

    @property
    def host(self):
        """The deck holding this view as a tab, or None when it is its own window."""
        return self._host

    def reveal(self) -> None:
        """Bring this view to the front and make it the one on screen.

        A view is either a top-level window or a tab, and "show me this one" means something
        different in each. Answering it HERE is what lets ``ViewerManager.focus`` keep one call
        site: a manager that had to know about decks would need the same branch at four more."""
        target = self._host or self
        # `showNormal()` ONLY when actually minimised. Called unconditionally it also
        # DE-MAXIMISES a maximised deck — so every programmatic reveal (a plate navigate, a
        # cached-result replay) yanked a full-screen views window back to its normal size, which
        # is one shape of Julio's "tabs do not collapse sometimes" (2026-08-19; the other shape
        # was `collapse()` racing the navigator's raise, and the navigator is gone).
        if target.isMinimized():
            target.showNormal()
        else:
            target.show()
        target.raise_()
        target.activateWindow()
        if self._host is not None:
            self._host.set_current(self)

    def collapse(self) -> None:
        """Minimise. Minimising a TAB is meaningless, so a hosted view minimises its deck — which
        is also what the user means by "collapse all" when their views are tabs."""
        (self._host or self).showMinimized()

    def request_close(self) -> None:
        """Close this view, whichever kind of thing it currently is.

        A tab page never receives a close event, so ``close()`` on a hosted view would run
        ``closeEvent`` and delete the widget while the deck still held it. The deck untabs first
        and then disposes, which is why closing goes through the host when there is one."""
        if self._host is None:
            self.close()
        else:
            self._host.close_page(self)

    def _is_active_view(self) -> bool:
        """Is this the view the user is working in?

        For a window that is `isActiveWindow()`. For a tab it is BOTH the deck being active and
        this page being the current one — a background tab in a focused deck is no more "the view
        the user is in" than a window behind another."""
        host = self._host
        if host is None:
            return self.isActiveWindow()
        return bool(host.isActiveWindow()) and host.current_page() is self

    def changeEvent(self, event):                    # noqa: N802 - Qt naming
        from qtpy.QtCore import QEvent

        if event.type() == QEvent.ActivationChange:
            active = self._is_active_view()
            self.set_active(active)                  # halt playback in a window nobody is watching
            # ...AND TELL THE REGISTRY WHO IS IN FRONT. Only on the way IN: deactivation is not
            # "no view is focused", it is usually the user clicking the plate, and the plate is
            # how you drive the focused view — clearing the focus here would mean the plate lost
            # its target the instant you reached for it.
            if active and self._manager is not None:
                self._manager.note_focus(self.window_id)
        super().changeEvent(event)

    def resizeEvent(self, e):                        # noqa: N802 - Qt naming
        """Grow the type with the window, exactly as the root does."""
        super().resizeEvent(e)
        rescale_fonts(self)

    def dispose(self) -> None:
        """Everything that must happen before this view stops existing, WHEREVER it lives.

        Extracted from ``closeEvent`` because that conflates two jobs — *join my threads* and
        *tell the registry I am gone* — and only a top-level window ever gets a close event. A
        view that stops existing some other way (removed from a container, dropped by a caller
        that never showed it) skipped every join and every deregistration silently.

        IDEMPOTENT, and that is not decoration: a window can be disposed by its owner and then
        still receive a ``closeEvent``, and joining a QThread twice or closing a napari Viewer
        twice is exactly the shape that aborts the interpreter rather than raising.

        Each step is wrapped separately, on purpose. These are independent teardowns and one that
        throws must not strand the ones after it — a QThread left running past destruction aborts
        the process ("QThread: Destroyed while thread is still running"), which is a far worse
        outcome than the error being swallowed.
        """
        if self._disposed:
            return
        self._disposed = True
        # DISARM THE DEBOUNCE TIMERS FIRST: a pending single-shot fires into a torn-down
        # window during the deleteLater drain (the plate's closeEvent records the identical
        # measured segfault; these two were never stopped here).
        for name in ("_load_timer", "_time_load_timer"):
            timer = getattr(self, name, None)
            if timer is not None:
                try:
                    timer.stop()
                    timer.timeout.disconnect()
                except (TypeError, RuntimeError):
                    pass
        # RELEASE, never dispose: the inserted parameter panel and the hosted plate/log slots
        # are the PLATE's live widgets; dying with this view would lose them for good.
        try:
            self._remove_param_slot()
        except Exception:                            # noqa: BLE001 - teardown must continue
            pass
        try:
            self.release_plate_slots()
        except Exception:                            # noqa: BLE001 - teardown must continue
            pass
        panel = self._op_panel
        self._op_panel = None
        if panel is not None and _alive(panel):
            # The panel can live inside the pane's napari window (the docked left column), not
            # directly in this window: delete it explicitly or it outlives the view as an orphan.
            try:
                panel.setParent(None)
                panel.deleteLater()
            except Exception:                        # noqa: BLE001 - a dead panel is already gone
                pass
        try:
            if self._worker is not None and self._worker.isRunning():
                self._worker.stop()
                self._worker.wait(2000)
        except Exception:                            # noqa: BLE001
            pass
        try:
            if self._focus_worker is not None and self._focus_worker.isRunning():
                self._focus_worker.stop()
                self._focus_worker.wait(2000)
        except Exception:                            # noqa: BLE001
            pass
        try:
            if self._video_worker is not None and self._video_worker.isRunning():
                self._video_worker.stop()
                self._video_worker.wait(2000)
        except Exception:                            # noqa: BLE001
            pass
        try:
            if self._png_worker is not None and self._png_worker.isRunning():
                self._png_worker.stop()
                self._png_worker.wait(2000)
        except Exception:                            # noqa: BLE001
            pass
        try:
            if self._slider is not None:
                self._slider.shutdown()
        except Exception:                            # noqa: BLE001
            pass
        try:
            # The FOV axis owns a napari AnimationThread exactly as the region slider does, and Qt
            # aborts the process on a QThread destroyed while running. Closing a FOVs tab mid-walk
            # is the ordinary way to meet that.
            if self._fov_slider is not None:
                self._fov_slider.shutdown()
        except Exception:                            # noqa: BLE001
            pass
        try:
            # BEFORE the pane goes: the inset is parented to the canvas the pane is about to
            # destroy, and the loupe worker is a QThread like every other one joined above.
            if self._loupe is not None:
                self._loupe.shutdown()
        except Exception:                            # noqa: BLE001
            pass
        try:
            vol, self._native3d = self._native3d, None
            close = getattr(vol, "close", None)
            if callable(close):
                close()
        except Exception:                            # noqa: BLE001
            pass
        try:
            bar = getattr(self, "_time_point_bar", None)
            if bar is not None:
                bar.shutdown()
        except Exception:                            # noqa: BLE001
            pass
        for worker in list(getattr(self, "_retired_workers", []) or []):
            try:
                worker.stop()
                worker.wait(2000)
            except Exception:                        # noqa: BLE001
                pass
        # THE NAPARI VIEWER. Nothing called this before: `MosaicPane.shutdown` had zero callers
        # in squidxplorer/, tests/ or tools/, while its own docstring said every owner calls it
        # before deleteLater(). deleteLater() on the Qt wrapper does NOT close the Viewer —
        # napari keeps every Viewer in its own instance registry — so one GL context and tens of
        # MB leaked per window CLOSED. It goes last because it tears down the surface the workers
        # above draw into, so they are joined first.
        try:
            if self._pane is not None:
                self._pane.shutdown()
        except Exception:                            # noqa: BLE001
            pass
        self.closed.emit(self)

    def closeEvent(self, event):                     # noqa: N802 - Qt naming
        self.dispose()
        super().closeEvent(event)


class ViewerManager(QObject):
    """Registry of open :class:`RegionViewer` windows, keyed by a monotonic ID."""

    windowsChanged = Signal()
    runProgressChanged = Signal(object)
    viewFocused = Signal(object)
    windowOpened = Signal(object)
    # The dataset's contrast ceiling rose: every open window must widen its sliders to (lo, hi).
    # A SIGNAL and not a direct call because `_bitdepth` observes on the fuse WORKER thread, and a
    # queued signal to this GUI-thread object is what marshals it. The subscriber on the depth
    # object must therefore do nothing but emit this -- see `_on_depth_rose`.
    depthChanged = Signal(str, float, float)

    def __init__(self, reader: Any = None, meta: Optional[dict] = None,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._reader = reader
        self._meta = meta
        self._windows: "dict[int, RegionViewer]" = {}
        self._next_id = 1
        self._focused_id: Optional[int] = None
        self._selected_ids: "list[int]" = []
        #: The tab decks holding views. A list rather than one, because a detached view can be
        #: re-docked into a second deck later and phase 2 wants to drag between them; today the
        #: app makes exactly one, lazily, through `deck()`.
        self._decks: "list" = []
        #: Do spawned views become TABS? One policy point, consulted only by `_spawn`, so this can
        #: be flipped without touching a single caller.
        #:
        #: OFF BY DEFAULT, and that is a finding rather than caution. Turned on for every spawn it
        #: reparents a view on every open, and the PR's suite did not survive that: measured
        #: 2026-08-10, `test_plate_navigates_views`, `test_rename` and `test_raise_plate` went from
        #: passing to 0xC0000005, and `test_nav_close_selected` to 0xC0000409. Those are aborts,
        #: not failures — the same class `tests/test_window_lifetime` exists for. So the deck ships
        #: as something a caller asks for, exercised by its own tests, and the default flips only
        #: once those aborts are understood rather than routed around.
        self.tabbed_views = False
        self.operator_specs: "list" = []
        self.run_operator: Optional[Any] = None
        #: ``op -> loupe source``, set by the root ``PlateWindow``. THE PIXEL SOURCE A CANVAS
        #: LOUPE READS, handed down the same way ``run_operator`` is rather than each window
        #: building its own: a source caches a whole field's planes (tens of MB) and a
        #: written-plate source is MUTATED as wells land, so a per-window copy would be both
        #: expensive and, during a run, wrong about which wells exist. ``None`` means no loupe
        #: (a manager built by a test).
        #:
        #: It is a CALLABLE and not the dict, because ``PlateWindow._release_loupe_sources``
        #: rebinds ``_loupe_sources`` wholesale — a bound ``.get`` would keep answering from the
        #: dead dict for the rest of the session.
        self.loupe_source_for = None
        #: Installs the collapsible Operators dock on a new deck / free window, or None (a
        #: manager built by a test, or a library caller). Set by `PlateWindow`; consulted only
        #: by `deck()` and `_spawn`, so the dock is wired ONCE per host, never per region.
        self.defaults = ViewDefaults()

        self._run_progress = None


        # A manager can be handed a reader at construction and never see `set_dataset` (the view
        # settings suite builds one exactly that way), so the depth has to be armed from BOTH or
        # those windows measure against whatever the previous dataset left behind.
        self._arm_depth(meta)

    def _arm_depth(self, meta: Optional[dict]) -> None:
        """Start measuring a new acquisition's contrast ceiling and republish every rise."""
        depth = _bitdepth.new_dataset((meta or {}).get("dtype"))
        depth.on_change(self._on_depth_rose)

    def _on_depth_rose(self, channel: str, lo: float, hi: float) -> None:
        """Called ON THE FUSE WORKER THREAD. Emit and return -- do nothing else here.

        The emit is what hops to the GUI thread (Qt queues a signal across threads); touching a
        napari layer from here would be a cross-thread write into the render path.
        """
        self.depthChanged.emit(str(channel), float(lo), float(hi))

    def set_dataset(self, reader: Any, meta: dict) -> None:
        """Point every FUTURE window at a new acquisition, and forget the last one's look.

        A NEW DATASET IS A NEW LOOK. A contrast window is a statement in the previous
        acquisition's counts: carry (11, 111) from a 12-bit set onto one that is 12-bit shifted
        into 16 and every channel renders black; carry it the other way and everything saturates.
        `channel_visibility` goes for the same reason -- it is keyed by channel NAME, and the
        names differ between acquisitions, so what survives is either dead weight or an
        accidental match that opens a channel dark.

        What deliberately STAYS: everything in `defaults` that describes HOW you look rather than
        at WHAT -- focus mode, and the rest of `_SETTING_BASELINE`.
        """
        self._reader, self._meta = reader, meta
        self._arm_depth(meta)
        self.defaults.set("luts", {})
        self.defaults.set("channel_visibility", {})

    @property
    def run_progress(self):
        """The in-flight work's latest ``ProgressReport``, or None when nothing is running."""
        return self._run_progress

    def set_run_progress(self, report) -> None:
        """Publish (or clear, with None) what is running, for the navigator's bar."""
        self._run_progress = report
        self.runProgressChanged.emit(report)

    @property
    def windows(self) -> "list[RegionViewer]":
        return list(self._windows.values())

    def views(self) -> "list[View]":
        """Every open window as a :class:`View` (a named region-set) — the unit an operator targets."""
        out: "list[View]" = []
        for win in self.windows:
            roi = getattr(win, "_roi_bbox", None)
            # An elif, not a precedence rule: `RegionViewer.__init__` refuses to build a window
            # that is both, so at most one of these can be true.
            if getattr(win, "_fov_mode", False):
                kind = "fovs"
            elif roi is not None:
                kind = "roi"
            else:
                kind = "window"
            out.append(View(
                id=f"w{win.window_id}", name=win.display_name,
                regions=tuple(win._regions),
                kind=kind,
                window_id=win.window_id, roi_bbox=roi))
        return out

    def window(self, window_id: Optional[int]) -> "Optional[RegionViewer]":
        """The open window with this id, or None — including for None itself, so a caller holding
        "the id I opened, if any" can ask without checking twice. An id that outlives its window
        is exactly the kind of thing that should answer None in one place rather than four."""
        if window_id is None:
            return None
        return self._windows.get(int(window_id))

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
                   parent_id: Optional[int] = None, luts: Optional[dict] = None,
                   fovs: bool = False) -> Optional[RegionViewer]:
        """Open a CHILD window from a parent window (the next level of the tree).

        ``fovs=True`` opens a FOV WALK instead: the same regions, uncropped, with a slider that
        steps the camera across the region's fields. It is mutually exclusive with ``roi_bbox``
        and ``RegionViewer.__init__`` refuses the combination by name rather than picking one."""
        regions = [str(r) for r in regions if r]
        if not regions:
            return None
        base = RegionViewer._view_label(regions)
        title = f"{base}  ◂ view {parent_id}" if parent_id is not None else base
        return self._spawn(regions, title=title, roi_bbox=roi_bbox, parent_id=parent_id, luts=luts,
                           fovs=fovs)

    def _baseline_for(self, parent_id: Optional[int]) -> "dict[str, Any]":
        """The settings a NEW window opens with: the global defaults, with ``_INHERIT`` reading the opener."""
        parent = self._windows.get(int(parent_id)) if parent_id is not None else None
        live: "dict[str, Any]" = {}
        if parent is not None:
            try:
                live = parent.current_settings()
            except Exception:                            # noqa: BLE001 - fall back to the defaults
                log.warning("could not read view %s's settings; the new window takes the defaults.",
                            parent_id)
        out: "dict[str, Any]" = {}
        for name, mode in _SETTING_BASELINE.items():
            value = self.defaults.get(name)
            if mode == _INHERIT and name in live:
                value = live[name]
            out[name] = value
        return out

    def _spawn(self, regions: "list[str]", *, title: Optional[str] = None,
               roi_bbox: Optional[tuple] = None,
               parent_id: Optional[int] = None, luts: Optional[dict] = None,
               fovs: bool = False) -> Optional[RegionViewer]:
        if self._reader is None or self._meta is None:
            log.warning("open() called before a dataset was loaded; ignoring.")
            return None
        wid = self._next_id
        self._next_id += 1
        n = len(regions)
        if fovs:
            what = "FOVs in "
        elif roi_bbox is not None:
            what = "ROI in "
        else:
            what = ""
        clock = _measure.WindowOpen(
            f"{what}{n} region{'' if n == 1 else 's'}: "
            f"{RegionViewer._view_label(regions)}",
            n_targets=n)
        baseline = self._baseline_for(parent_id)
        if luts is not None:
            baseline["luts"] = luts
        win = RegionViewer(
            self._reader, self._meta, regions, window_id=wid, title=title,
            manager=self, roi_bbox=roi_bbox,
            operator_specs=self.operator_specs, run_operator=self.run_operator,
            parent_id=parent_id, settings=ViewSettings(baseline), fovs=fovs,
        )
        win.open_clock = clock
        win.closed.connect(self._on_window_closed)
        win.regionsChanged.connect(self._on_window_regions_changed)
        self._windows[wid] = win
        self._focused_id = wid
        self._selected_ids = [wid]
        # THE ONE POLICY POINT for "views are tabs". Every opener — the plate's Open view, a
        # marquee, a double-click, an ROI child, the default layout — arrives here, so none of them
        # needs to know, and turning `tabbed_views` off gives independent windows back with no
        # other edit.
        deck = self.deck() if self.tabbed_views else None
        if deck is not None:
            deck.dock_page(win)
            # A new tab must be SEEN: `show()` alone does not un-minimise, so a view opened while
            # the deck sat minimised landed in a window that stayed in the dock/taskbar — the
            # other measured shape of "tabs do not collapse / do not come back" (2026-08-19).
            if deck.isMinimized():
                deck.showNormal()
            deck.show()
            deck.raise_()
            deck.activateWindow()
        else:
            win.show()
            win.raise_()
            win.activateWindow()
        self._replay_cached_results(win)
        self.windowOpened.emit(win)
        self.windowsChanged.emit()
        self.viewFocused.emit(list(win._regions))
        self._sync_plate_slots()
        return win

    def _sync_plate_slots(self, *_args) -> None:
        """Host the plate view + log in the DECK's current view (one window, 2026-08-25).

        The plate keeps its books; only where the two widgets render moves. With no view to
        host (deck empty, or free-standing windows), the plate ADOPTS them home and shows
        itself again - the app always has a surface.
        """
        plate = self.parent()
        take = getattr(plate, "plate_slot_widgets", None)
        if not callable(take):
            return
        deck = self.deck(create=False) if self.tabbed_views else None
        view = deck.current_page() if deck is not None else None
        if view is None or not hasattr(view, "adopt_plate_slots"):
            adopt = getattr(plate, "adopt_plate_slots_home", None)
            if callable(adopt):
                adopt()
            return
        if getattr(view, "_hosts_plate_slots", False):
            return                               # already the host; nothing to move
        widgets = take()
        if widgets is None:
            return                               # nothing to host before an ingest
        for w in self._windows.values():
            w._hosts_plate_slots = False
        if not view.adopt_plate_slots(*widgets):
            plate.adopt_plate_slots_home()
            return
        hide = getattr(plate, "maybe_hide_for_one_window", None)
        if callable(hide):
            hide(deck)

    def _replay_cached_results(self, win: RegionViewer) -> int:
        """Give a NEWLY OPENED window every operator result already computed for its region."""
        from squidxplorer._recipe import acquisition_version, cached_operator_results

        region = win.current_region()
        if not region or self._reader is None:
            return 0
        added = 0
        for op, result in cached_operator_results(region, acquisition_version(self._reader)):
            try:
                added += int(win.deliver_result(op, result, visible=False) or 0)
            except Exception as exc:            # noqa: BLE001 - a replay must not fail an open
                log.warning("view %s could not take the cached %s result for %s: %s",
                            win.window_id, op, region, exc)
        if added:
            win._say(f"reused {added} already-computed layer(s) for {region} "
                     "(no recompute) - toggle them in the layers panel.")
        return added

    @property
    def focused_id(self) -> Optional[int]:
        """The window id of the active view (its plate hue reads brighter), or None."""
        return self._focused_id

    @property
    def selected_ids(self) -> "list[int]":
        """Window ids selected in the navigator; the plate washes each in its own hue."""
        return [i for i in getattr(self, "_selected_ids", []) if i in self._windows]

    def set_selected(self, ids: "Sequence[int]") -> None:
        """The navigator selection changed (possibly many rows). Store it and re-tint the plate."""
        self._selected_ids = [int(i) for i in ids]
        self._focused_id = self._selected_ids[0] if self._selected_ids else None
        self.viewFocused.emit([])

    def refresh_deck_titles(self) -> None:
        """Tab text follows a rename. The navigator gets this free by rebuilding on
        ``windowsChanged``; a deck holds its labels in the tab bar, so it has to be told."""
        for deck in self.decks():
            try:
                deck.refresh_titles()
            except Exception:                        # noqa: BLE001 - a label is never worth a crash
                pass

    def decks(self) -> "list":
        """Every tab deck this manager has made, live ones only."""
        self._forget_dead_decks()
        return list(self._decks)

    def deck(self, create: bool = True):
        """THE deck, made on first use. One policy point, so "views are tabs" is one decision.

        Lazy because a session that only ever opens the plate should not build a window nobody
        asked for, and `tests/test_no_orphan_windows` is entitled to say so.
        """
        live = [d for d in self._decks if _alive(d)]
        self._decks = live
        if live:
            return live[0]
        if not create:
            return None
        from squidxplorer._view_deck import ViewDeck

        deck = ViewDeck(index=len(self._decks) + 1)
        deck.pageActivated.connect(self.note_focus)
        # ONE WINDOW: the current tab hosts the plate view + log; a tab switch re-homes them.
        deck.pageActivated.connect(self._sync_plate_slots)
        # ...and the deck is the app surface while the plate window hides: drop-to-open and
        # the essential menu actions forward to the plate.
        bind = getattr(deck, "bind_plate", None)
        if callable(bind):
            bind(self.parent())
        # NO right-edge operator dock (retired 2026-08-25): a view's Run on plate is the bulk
        # path, and each view's operator panel lives in that view's own LEFT column.
        # A BOUND METHOD, NEVER A SELF-CAPTURING LAMBDA. PyQt keeps a lambda alive in a slot proxy
        # parented to the SENDER, so `destroyed` -- which fires while the deck is being torn down --
        # would call into this manager whether or not the manager still exists. Connected as a
        # bound method, PyQt weak-references the receiver and drops the connection when it goes.
        # Measured: the lambda aborted the process (0xC0000409) during fixture teardown in
        # test_raise_plate, where the manager is parented to a fake plate that dies first. It is
        # the same rule test_window_lifetime states for timers, and it applies to every deferred
        # call, not only to QTimer.
        deck.destroyed.connect(self._on_deck_destroyed)
        self._decks.append(deck)
        return deck

    def _on_deck_destroyed(self, *_args) -> None:
        self._forget_dead_decks()

    def _forget_dead_decks(self) -> None:
        self._decks = [d for d in self._decks if _alive(d)]

    def _on_window_regions_changed(self, win: "RegionViewer") -> None:
        """A window adopted a region. Re-publish it.

        Nothing here recomputes anything: `_regions` reads through the cursor, so `views()` and the
        `viewFocused` payload are already correct by the time this runs. What they are not is
        ANNOUNCED — the navigator rebuilds on `windowsChanged` and the plate re-tints on
        `viewFocused`, and neither has any way to notice a cursor moved inside a window.
        """
        self.windowsChanged.emit()
        self.viewFocused.emit(list(win._regions))

    def note_focus(self, window_id: int) -> None:
        """The user activated this window THEMSELVES — its title bar, alt-tab, a click in its canvas.

        The passive half of :meth:`focus`. Until this existed, ``_focused_id`` was written only by
        the app moving focus (``_spawn``, ``focus``, ``set_selected``, ``clear_focus``), never by
        the user moving it, so clicking a view's title bar changed nothing: the plate kept washing
        the previously focused window, and anything reading "the window the user is looking at"
        off ``focused_id`` read the wrong window. It was a lie about the user, told by a registry
        that only watched itself.

        RECORDS AND RE-PUBLISHES. It must never raise or activate: ``focus`` calls
        ``activateWindow()``, which fires ``changeEvent``, which lands here — and a raise from here
        would be an infinite ping-pong with the window manager. The unchanged-id early return is
        the second guard on that, and it is why ``focus`` sets ``_focused_id`` BEFORE it activates.
        """
        wid = int(window_id)
        win = self._windows.get(wid)
        if win is None or self._focused_id == wid:
            return
        self._focused_id = wid
        if wid not in self._selected_ids:
            # Mirrors _spawn: the plate washes the SELECTED views, so a window the user brought
            # forward has to be in that set or focusing it would move no hue at all.
            self._selected_ids = [wid]
        self.viewFocused.emit(list(win._regions))

    def active_view(self) -> "Optional[RegionViewer]":
        """The window a plate click drives: the focused one, or None when no view is open.

        ONE answer to "which window is active", so the plate, the LUT export and the navigator
        cannot disagree about it. Anything asking that question must come here rather than ask a
        window ``isActiveWindow()`` — a window can be the active TOP-LEVEL without being the view
        the user is working in (the plate itself takes activation on every click), and once views
        can be tabbed there is no longer one window per view at all.
        """
        if self._focused_id is None:
            return None
        return self._windows.get(self._focused_id)

    def focus(self, window_id: int) -> None:
        win = self._windows.get(int(window_id))
        if win is not None:
            self._focused_id = int(window_id)       # BEFORE activateWindow(); see note_focus
            win.reveal()                            # a window raises; a tab also becomes current
            self.viewFocused.emit(list(win._regions))

    def raise_plate(self) -> bool:
        """Bring the ROOT plate window to the front. Returns whether there was one to raise."""
        plate = self.parent()
        if plate is None or not hasattr(plate, "raise_"):
            return False
        try:
            if plate.isMinimized():
                plate.showNormal()
            plate.raise_()
            plate.activateWindow()
        except Exception:                            # noqa: BLE001 - a raise is never worth a crash
            return False
        return True

    def close(self, window_id: int) -> None:
        win = self._windows.get(int(window_id))
        if win is not None:
            win.request_close()      # a window closes; a tab is untabbed and then disposed

    def close_all(self) -> None:
        for win in list(self._windows.values()):
            win.request_close()

    def _on_window_closed(self, win: "RegionViewer") -> None:
        clock = getattr(win, "open_clock", None)
        if clock is not None:
            clock.finish(_measure.STOPPED, "closed before its mosaic landed")
        wid = getattr(win, "window_id", -1)
        self._windows.pop(wid, None)
        if self._focused_id == wid:
            self._focused_id = None
        self.windowsChanged.emit()
        # ONE WINDOW: the closed view may have hosted the plate view + log. Re-home them into
        # the deck's current page, or back into the plate window when no view is left.
        try:
            self._sync_plate_slots()
        except Exception as exc:                 # noqa: BLE001 - a re-home must not block a close
            log.warning("could not re-home the plate slots: %s: %s", type(exc).__name__, exc)


class StatusRow(QObject):
    """THE ONE bar in the app: the run-progress bar, shown only while a run is live, labelled
    with the run's own words (Julio, 2026-08-25: "The memory usage bar is confusing, it looks
    as if it was the deconvolution progressing"; the memory bar is deleted, memory stays a
    DEBUG footprint line). Built here, adopted by the log slot."""

    def __init__(self, manager: ViewerManager, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._work_label = QLabel("")
        self._work_label.setStyleSheet("color:#8b949e;font-size:11px;border:none;")
        self._work_label.setWordWrap(True)
        self._work_label.hide()
        self._work_bar = QProgressBar()
        self._work_bar.setTextVisible(False)
        self._work_bar.setFixedHeight(14)
        self._work_bar.setStyleSheet(
            "QProgressBar{background:#161b22;border:1px solid #30363d;border-radius:3px;}"
            "QProgressBar::chunk{background:#1f6feb;border-radius:3px;}"
        )
        self._work_bar.hide()
        manager.runProgressChanged.connect(self._on_run_progress)
        self._on_run_progress(manager.run_progress)

    def widgets(self) -> tuple:
        """The two widgets, in `LogPanel.adopt_status_row`'s order."""
        return (self._work_label, self._work_bar)

    def _on_run_progress(self, report) -> None:
        """Draw (or take down) the work bar. ``report`` is a ``ProgressReport``, or None for idle."""
        try:
            if report is None:
                self._work_label.hide()
                self._work_bar.hide()
                return
            try:
                sentence, percent = report.sentence(), report.percent
            except Exception:                        # noqa: BLE001 - a bad report is not a crash
                self._work_label.hide()
                self._work_bar.hide()
                return
            self._work_label.setText(sentence)
            if percent is None:
                self._work_bar.setRange(0, 0)
            else:
                self._work_bar.setRange(0, 100)
                self._work_bar.setValue(int(percent))
            self._work_label.show()
            self._work_bar.show()
        except RuntimeError:
            # Adopted by the log slot, which can die inside a hosting view (one window,
            # 2026-08-25): a dead bar unhooks this slot for good.
            sender = self.sender()
            try:
                if sender is not None:
                    sender.runProgressChanged.disconnect(self._on_run_progress)
            except (AttributeError, TypeError, RuntimeError):
                pass
