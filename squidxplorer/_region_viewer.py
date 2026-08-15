"""Decentralized viewer windows: one INDEPENDENT napari window per selection."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from qtpy.QtCore import QObject, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

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

# THE LUT clipboard lives in `_lut_clipboard` now; this name is the SAME dict object, kept
# because tests (and history) reach it as `_region_viewer._LUT_CLIPBOARD`.
from squidxplorer._lut_clipboard import CLIPBOARD as _LUT_CLIPBOARD  # noqa: E402
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


class RegionViewer(QMainWindow):
    """ONE independent napari window over a subset of regions."""

    closed = Signal(object)

    _op_action: Optional[str] = None
    _op_address: Any = None
    _result_region: Optional[str] = None
    _op_progress: Any = None
    open_clock: Any = None

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
    ) -> None:
        super().__init__(parent)
        self._reader = reader
        self._meta = meta
        self._regions = [str(r) for r in regions]
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
        self._spot_worker = None
        self._focus_worker = None
        self._video_worker = None
        self._manager = manager
        self._operator_specs = list(operator_specs or [])
        self._render_mode = "2d"
        self._run_operator = run_operator
        self.parent_id = parent_id
        self.settings = settings if settings is not None else ViewSettings()
        self._settings_applied = False
        self._roi_bbox = roi_bbox
        self._roi_layer = None
        self._op_action: Optional[str] = None
        self._op_address: Any = None
        self._result_region: Optional[str] = None

        self._derived_name = title or self._view_label(self._regions)
        if self._roi_bbox is not None:
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
            msg = QLabel(f"napari viewer unavailable — {message}")
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

        try:
            ch_combo = getattr(pane, "detect_channel", None)
            if ch_combo is not None and ch_combo.count() == 0:
                for c in (self._meta or {}).get("channels", []):
                    ch_combo.addItem(str(c["name"]))
            btn = getattr(pane, "detect_button", None)
            if btn is not None:
                btn.setEnabled(True)
                btn.clicked.connect(self._detect_nuclei)
        except Exception:                                # noqa: BLE001 - detection stays optional
            pass

        lay.addWidget(self._build_top_row(), 0)
        lay.addWidget(pane, 1)

        self._cursor = RegionCursor()
        self._cursor.on_problem(self._say)
        self._cursor.subscribe(self._on_region_changed)
        self._slider = RegionSlider()
        self._slider.on_problem(self._say)
        self._slider.bind(self._cursor)
        lay.addWidget(self._slider)

        self._time_point_bar = TimePointBar(on_change=self._on_time_point_changed, playback=True)
        self._time_point_bar.on_problem(self._say)
        self._time_point_bar.set_count(int((self._meta or {}).get("n_t", 1) or 1))
        lay.addWidget(self._time_point_bar)

        self.setCentralWidget(central)

        self._cursor.set_order(self._regions)
        if self._cursor.index is None and self._regions:
            self._cursor.set_index(0)

    _BOX_QSS = "QFrame{background:#0d1117;border:1px solid #232b3a;border-radius:5px;}"
    _TITLE_QSS = "color:#8b949e;font-size:10px;font-weight:700;border:none;"
    _CHIP_QSS = (
        "QPushButton{background:#161b22;color:#c9d1d9;border:1px solid #30363d;"
        "border-radius:4px;padding:3px 9px;font-size:11px;}"
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

        view_box, vv = self._titled_box("2D / 3D · ROI")
        r1 = QHBoxLayout(); r1.setSpacing(4)
        self._btn_2d = self._chip("2D", "View the SELECTED ROI in 2D (opens it as a child window); "
                                  "with no ROI picked, just shows the mosaic in 2D.", self._view_roi_2d)
        self._btn_3d = self._chip("3D", "Open this view in 3D at NATIVE resolution (the region if it "
                                  "fits the GPU texture, else draw an ROI to pick the spot). "
                                  "Replaces this window's previous 3D view rather than adding "
                                  "another window.",
                                  self._open_3d)
        self._btn_focus = self._chip("⌖ focus", "Jump the z-slider to the sharpest plane "
                                     "(Tenengrad autofocus) of this region's centre FOV.",
                                     self._focus_reference_plane)
        self._btn_plate = self._chip("▣ plate", "Bring the plate window to the front — it ends up "
                                     "buried under the views opened from it.", self._raise_plate)
        self._btn_controls = self._chip(
            "⚙ controls", "Bring the plate window forward AND open the controls for the operator "
            "this window is showing, so its parameters (iterations, thresholds) are one click "
            "away. Says so when the window is showing raw pixels, which have none.",
            self._show_operator_controls)
        self._btn_record = self._chip(
            "⏺ movie", "Export what this window is showing as an .mp4, sweeping the acquisition's "
            "time axis (or its z axis when there is no time series). Runs off the UI thread; "
            "click again to cancel.", self._record_movie)
        r1.addWidget(self._btn_2d); r1.addWidget(self._btn_3d); r1.addWidget(self._btn_focus)
        r1.addWidget(self._btn_record)
        r1.addWidget(self._btn_plate)
        r1.addStretch(1)
        self._refresh_record_chip()
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

        op_box, ov = self._titled_box("Operators for this window")
        opr = QHBoxLayout(); opr.setSpacing(4)
        self._op_combo = QComboBox()
        self._op_combo.setStyleSheet(self._COMBO_CHIP_QSS)
        for spec in self._operator_specs:
            self._op_combo.addItem(str(spec[1]), spec[0])
        if self._op_combo.count() == 0:
            self._op_combo.addItem("no operators", None)
            self._op_combo.setEnabled(False)
        opr.addWidget(self._op_combo, 1)
        opr.addWidget(self._btn_controls)
        self._controls_note = QLabel("")
        self._controls_note.setStyleSheet("color:#8b949e;font-size:11px;border:none;")
        self._controls_note.setWordWrap(False)
        opr.addWidget(self._controls_note)
        self._op_combo.currentIndexChanged.connect(lambda _i: self._refresh_controls_note())
        opr.addWidget(self._chip("Run", "Run the selected operator on THIS view's regions.",
                                 self._run_view_operator))
        self._save_chk = QCheckBox("save")
        self._save_chk.setToolTip("Off = preview only (nothing written to disk). On = persist the "
                                  "operator result as an OME-Zarr.")
        self._save_chk.setStyleSheet("QCheckBox{color:#c9d1d9;font-size:11px;}")
        opr.addWidget(self._save_chk)
        ov.addLayout(opr)
        self._op_progress = QProgressBar()
        self._op_progress.setTextVisible(True)
        self._op_progress.setFixedHeight(16)
        self._op_progress.setStyleSheet(self._PROGRESS_QSS)
        self._op_progress.hide()
        ov.addWidget(self._op_progress)
        sync = QHBoxLayout(); sync.setSpacing(4)
        _across = QLabel("between windows:")
        _across.setStyleSheet(self._AT_DEFAULTS_QSS)
        sync.addWidget(_across)
        sync.addWidget(self._chip("⧉ Copy LUTs",
                                  "THIS WINDOW → clipboard: its per-channel contrast + colormap. "
                                  "The only way to move contrast to a window that is ALREADY OPEN "
                                  "(a new window inherits, an open one does not), and the only "
                                  "one that carries the colormap. Shared with the plate.",
                                  self._copy_luts))
        sync.addWidget(self._chip("⤓ Paste LUTs",
                                  "clipboard → THIS WINDOW: apply the copied contrast + colormap "
                                  "to this window's channels. Counts as you changing contrast "
                                  "here, so this window will report itself diverged.",
                                  self._paste_luts))
        _within = QLabel("│  in this window:")
        _within.setStyleSheet(self._AT_DEFAULTS_QSS)
        sync.addWidget(_within)
        sync.addWidget(self._chip("≡ Match layers to raw",
                                  "THIS WINDOW's operator layers ← THIS WINDOW's raw: put raw's "
                                  "contrast window on every operator layer of the same channel, "
                                  "so flipping between raw and a result compares the same window. "
                                  "Results open on their own auto window so they are legible "
                                  "alone; this is the deliberate opt-in to raw's. Touches no "
                                  "other window and does not move raw.",
                                  self._match_raw_contrast))
        sync.addStretch(1)
        ov.addLayout(sync)
        h.addWidget(op_box, 1)

        def_box, dv = self._titled_box("Defaults")
        d1 = QHBoxLayout(); d1.setSpacing(4)
        self._focus_default_chk = QCheckBox("auto focus")
        self._focus_default_chk.setToolTip(
            "Jump to the sharpest plane (Tenengrad) once, when this window first shows a region. "
            "Later regions keep the z you are on. A global default; ticking it HERE changes this "
            "window only and marks it diverged.")
        self._focus_default_chk.setStyleSheet("QCheckBox{color:#c9d1d9;font-size:11px;}")
        self._focus_default_chk.setChecked(bool(self.settings.get("tenengrad_focus")))
        self._focus_default_chk.toggled.connect(self._on_focus_default_toggled)
        d1.addWidget(self._focus_default_chk)
        d1.addStretch(1)
        self._make_default_btn = self._chip(
            "make default", "Make THIS window's settings the default for windows opened from now "
            "on. Windows already open are left exactly as they are.", self._make_default)
        d1.addWidget(self._make_default_btn)
        dv.addLayout(d1)
        d2 = QHBoxLayout(); d2.setSpacing(4)
        self._diverged_label = QLabel("")
        self._diverged_label.setStyleSheet(self._AT_DEFAULTS_QSS)
        d2.addWidget(self._diverged_label, 1)
        self._reset_btn = self._chip(
            "↺ reset", "Put every setting back to what this window opened with.",
            self._reset_settings)
        d2.addWidget(self._reset_btn)
        dv.addLayout(d2)
        def_box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        h.addWidget(def_box, 0)
        self._refresh_divergence()

        row.setMaximumHeight(108)
        return row

    _AT_DEFAULTS_QSS = "color:#8b949e;font-size:10px;border:none;"
    _DIVERGED_QSS = "color:#e3b341;font-size:10px;font-weight:700;border:none;"
    _PROGRESS_QSS = (
        "QProgressBar{background:#161b22;border:1px solid #30363d;border-radius:3px;"
        "color:#c9d1d9;font-size:10px;text-align:center;}"
        "QProgressBar::chunk{background:#e3b341;border-radius:3px;}"
    )

    _SETTING_LABELS = {
        "tenengrad_focus": "auto focus",
        "channel_visibility": "channels",
        "luts": "contrast",
    }

    def _refresh_divergence(self) -> None:
        """Say which settings this window has overridden, and enable reset only when something is."""
        lab = getattr(self, "_diverged_label", None)
        if lab is None:
            return
        names = self.settings.diverged
        if names:
            pretty = ", ".join(self._SETTING_LABELS.get(n, n) for n in names)
            lab.setText(f"⚠ diverged: {pretty}")
            lab.setStyleSheet(self._DIVERGED_QSS)
        else:
            lab.setText("at the defaults")
            lab.setStyleSheet(self._AT_DEFAULTS_QSS)
        lab.setToolTip(
            "This window's settings differ from what it opened with. '↺ reset' puts them back; "
            "'make default' pushes them to windows opened from now on."
            if names else "This window's settings are the ones it opened with.")
        btn = getattr(self, "_reset_btn", None)
        if btn is not None:
            btn.setEnabled(bool(names))

    def _on_focus_default_toggled(self, on: bool) -> None:
        """The autofocus default, changed IN THIS WINDOW. Never propagated to the others."""
        self.settings.set("tenengrad_focus", bool(on))
        self._refresh_divergence()
        self._say(f"auto focus {'on' if on else 'off'} for this window.")

    def _sync_settings_widgets(self) -> None:
        """Put the controls back in step with the settings after a programmatic change."""
        chk = getattr(self, "_focus_default_chk", None)
        if chk is None:
            return
        chk.blockSignals(True)
        try:
            chk.setChecked(bool(self.settings.get("tenengrad_focus")))
        finally:
            chk.blockSignals(False)

    def _reset_settings(self) -> None:
        """Back to what this window opened with: the global defaults, or, for contrast, its parent's."""
        names = self.settings.reset()
        if not names:
            self._say("this window is already showing the settings it opened with.")
            return
        self._apply_luts(self.settings.get("luts"))
        self._apply_channel_visibility(self.settings.get("channel_visibility"))
        self._sync_settings_widgets()
        self._refresh_divergence()
        pretty = ", ".join(self._SETTING_LABELS.get(n, n) for n in names)
        self._say(f"reset {pretty} to what this window opened with.")

    def _raise_plate(self) -> None:
        """Bring the plate window back to the front. See ``ViewerManager.raise_plate``."""
        if self._manager is None or not self._manager.raise_plate():
            self._say("there is no plate window to raise from here.")

    def _plate(self):
        """The plate window, or None. The plate owns every operator panel; this window borrows."""
        return None if self._manager is None else self._manager.parent()

    def _run_scope(self):
        """WHERE a run from this window goes: its regions, narrowed to the ROI's own FOVs."""
        regions = list(self._regions)
        if self._roi_bbox is None or not regions:
            return regions
        from squidxplorer._mosaic_source import fovs_overlapping_bbox

        scoped: "dict[str, list[int]]" = {}
        for region in regions:
            fovs = fovs_overlapping_bbox(self._meta or {}, region, self._roi_bbox)
            if not fovs:
                return regions
            scoped[region] = fovs
        total = sum(len((self._meta or {}).get("fovs_per_region", {}).get(r) or []) for r in regions)
        picked = sum(len(v) for v in scoped.values())
        if picked >= total:
            return regions
        self._say(f"ROI: running on {picked} of {total} field(s) — the ones your box touches.")
        return scoped

    def _plate_operator_kwargs(self, key: str) -> dict:
        """What *key*'s panel on the plate is currently set to. ``{}`` = its declared defaults."""
        plate = self._plate()
        reader = getattr(plate, "operator_kwargs_for", None)
        if not callable(reader):
            return {}
        try:
            return dict(reader(str(key)) or {})
        except Exception as exc:                 # noqa: BLE001 - named, never a silent default
            log.warning("view %s could not read %s's parameters from the plate: %s: %s",
                        self.window_id, key, type(exc).__name__, exc)
            return {}

    def _params_summary(self, key: str) -> str:
        """One line of what *key* is set to, for the chip's side text and the run echo."""
        plate = self._plate()
        reader = getattr(plate, "operator_params_text", None)
        if not callable(reader):
            return "defaults"
        try:
            return str(reader(str(key)))
        except Exception:                        # noqa: BLE001 - a label must never raise
            return "defaults"

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
        chosen = str(current.get("z_operator") or "")
        from squidxplorer._engine import Z_REDUCER, operator_consumes

        try:
            reduces = bool(operator_consumes(chosen) & Z_REDUCER) if chosen else True
        except Exception:                        # noqa: BLE001 - an unknown z operator: leave it
            return {}
        if not reduces:
            return {}
        self._say(f"3D: stitching all {int((self._meta or {}).get('n_z') or 1)} z-planes "
                  f"(z operator 'keepz') — one pose graph, every plane fused from it.")
        return {"z_operator": "keepz"}

    def set_render_mode(self, mode: str) -> None:
        """Record whether this window is a PLANE or a VOLUME, and repaint what says so."""
        mode = "3d" if str(mode).lower() == "3d" else "2d"
        if mode == getattr(self, "_render_mode", "2d"):
            return
        self._render_mode = mode
        self._refresh_controls_note()

    def _refresh_controls_note(self) -> None:
        """Print what the chip would run with, beside the chip. Derived on every call."""
        note = getattr(self, "_controls_note", None)
        if note is None:
            return
        combo = getattr(self, "_op_combo", None)
        key = combo.currentData() if combo is not None else None
        if not key:
            note.setText("")
            return
        note.setText(f"{self._render_mode.upper()} · {self._params_summary(str(key))}")

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

    def _show_operator_controls(self) -> None:
        """Raise the plate AND open the controls for the operators THIS WINDOW has."""
        raised = self._manager is not None and self._manager.raise_plate()
        if not raised:
            self._say("controls: there is no plate window to open operator controls in.")
            return

        combo = getattr(self, "_op_combo", None)
        key = combo.currentData() if combo is not None else None
        if not key:
            self._say("controls: no operator is selected in this window's dropdown, so there is "
                      "nothing to tune. The plate is in front.")
            return

        plate = self._manager.parent()
        activate = getattr(plate, "_activate_operator", None)
        if activate is None:
            self._say(f"controls: this plate cannot open operator tabs, so {key} cannot be tuned "
                      "from here.")
            return
        try:
            activate(str(key))
        except Exception as exc:                         # noqa: BLE001 - named, never a dead click
            self._say(f"controls: could not open {key}: {exc}")
            return
        self._refresh_controls_note()
        self._say(f"controls: {combo.currentText()} — open on the plate window. This window will "
                  f"run it {self._render_mode.upper()} with {self._params_summary(str(key))}.")

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
            btn.setToolTip(f"No mp4 encoder on this machine — {problem}")
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

        from squidxplorer._viewer import _VideoWorker

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

    def _make_default(self) -> None:
        """Make THIS window's settings the default for windows opened FROM NOW ON."""
        if self._manager is None:
            self._say("this window has no manager, so it cannot set the global default.")
            return
        if not self._manager.make_default(self.window_id):
            self._say("this window is not in the registry, so it cannot set the global default.")
            return
        self._say("these are now the defaults for windows opened from now on; windows already "
                  "open are unchanged.")

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

    def _run_view_operator(self) -> None:
        """Run the operator picked in this window's dropdown on THIS view's regions, via the real engine."""
        if self._run_operator is None:
            self._say("the operator engine isn't connected to this window.")
            return
        key = self._op_combo.currentData() if getattr(self, "_op_combo", None) is not None else None
        if not key:
            self._say("no operator selected.")
            return
        regions = self._run_scope()
        save = bool(self._save_chk.isChecked()) if getattr(self, "_save_chk", None) is not None else False
        kwargs = dict(self._plate_operator_kwargs(key))
        kwargs.update(self._z_kwargs_for_mode(key, kwargs))
        try:
            log.info("view %s running %s on %s with %s", self.window_id, key,
                     (regions if isinstance(regions, dict) else list(regions)), kwargs)
            self._run_operator(key, regions=regions, save=save, requester=self,
                               operator_kwargs=kwargs)
            mode = "saving" if save else "previewing"
            self._echo(f"{mode} {self._op_combo.currentText()} on {self._view_label(regions)} "
                       f"[{self._render_mode.upper()}] · {self._params_summary(key)}.")
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
        if bar is None:
            return
        percent = report.percent
        if percent is None:
            self._show_progress(None, report.sentence())
            return
        bar.setRange(0, 100)
        bar.setValue(int(percent))
        bar.setFormat(report.sentence())
        bar.show()

    def operator_done(self, action: str, seconds: float) -> None:
        """Emit the console's ``done`` line, closing the started/done pair."""
        self.log.done(str(action), float(seconds), address=self._closing_address())
        self._echo(f"{action} finished in {float(seconds):.1f} s.")
        self._op_action = self._op_address = None
        self._hide_progress()

    def operator_failed(self, action: str, reason: str) -> None:
        """The failure outcome: an action that starts and then says nothing looks like one still running."""
        self.log.failed(str(action), str(reason), address=self._closing_address())
        self._echo(f"{action} failed: {reason}")
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
            bbox = region_bbox = (placement.bbox_um if placement is not None else preview_bbox)
            if self._roi_bbox is not None and region_bbox is not None:
                cropped = _crop_levels_to_bbox([plane], region_bbox, self._roi_bbox)
                if cropped is None:
                    continue
                levels, bbox = cropped
                plane = levels[0]
            try:
                mosaic.add_result(
                    result.kind, str(op), channel, plane,
                    colormap=_colormap_for(channel),
                    bbox_um=bbox,
                    z_scale_um=(dz if int(result.z_depth) > 1 else None),
                    visible=bool(visible),
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
        return added

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

    def _spot_source(self):
        """The (channel, raw layer) to detect nuclei on, or (None, None) with nothing to segment."""
        pane = self._pane
        mosaic = getattr(pane, "mosaic", None) if pane is not None else None
        if mosaic is None:
            return None, None
        v = self._napari_viewer()
        active = getattr(getattr(v, "layers", None), "selection", None)
        active_layer = getattr(active, "active", None) if active is not None else None
        for c in (self._meta or {}).get("channels", []):
            name = c["name"]
            layer = mosaic.find(_RAW_OP, name)
            if layer is None:
                continue
            if active_layer is not None and layer is active_layer:
                return name, layer
        for c in (self._meta or {}).get("channels", []):
            name = c["name"]
            layer = mosaic.find(_RAW_OP, name)
            if layer is not None:
                return name, layer
        return None, None

    def _detect_nuclei(self):
        """Detect nuclei (Cellpose) on THIS view's MIP, off the GUI thread, and overlay the mask."""
        if self._spot_worker is not None and self._spot_worker.isRunning():
            self._say("nuclei detection is already running in this window.")
            return
        channel, layer = None, None
        pane = self._pane
        picker = getattr(pane, "detect_channel", None) if pane is not None else None
        mosaic = getattr(pane, "mosaic", None) if pane is not None else None
        if picker is not None and mosaic is not None and picker.currentText():
            channel = picker.currentText()
            layer = mosaic.find(_RAW_OP, channel)
        if layer is None:
            channel, layer = self._spot_source()
        if layer is None:
            self._say("show a region in this view first, then detect nuclei.")
            return
        try:
            from squidxplorer._viewer import _SpotWorker
            from squidxplorer._spots import SpotParams
            from squidxplorer._workers import nuclei_operator
        except Exception as exc:                          # noqa: BLE001
            self._say(f"nuclei detection unavailable: {exc}")
            return
        region = self._cursor.region if self._cursor is not None else (
            self._regions[0] if self._regions else "")
        # The registry picks the algorithm (cellpose when its requires are importable, else the
        # watershed), and the SAME name labels the action, sources the panel's values and runs —
        # the label used to say cellpose whichever ran, and the params were hardcoded defaults.
        algo_name, segment = nuclei_operator()
        from squidxplorer._engine import operator_params

        declared = {p.name for p in operator_params(algo_name)}
        panel = self._plate_operator_kwargs(algo_name)
        params = SpotParams(**{k: v for k, v in panel.items() if k in declared})
        action = f"nuclei({algo_name}, {channel})"
        where = self.address()
        began = time.monotonic()
        data = layer.data
        if self._roi_bbox is not None:
            from squidxplorer._mosaic_source import mosaic_bbox_um
            from squidxplorer._napari_view import pyramid_levels

            region_bbox = mosaic_bbox_um(self._meta or {}, region)
            if region_bbox is not None:
                cropped = _crop_levels_to_bbox(list(pyramid_levels(data)), region_bbox,
                                               self._roi_bbox)
                if cropped is not None:
                    data = cropped[0]
                    self._say("detecting on the ROI only — the box you drew, not the whole well.")
        w = _SpotWorker(region, channel, data, None, None, params, parent=self,
                        algorithm=(algo_name, segment))
        self.log.started(action, address=where)
        weights = " — first run downloads weights…" if algo_name == "cellpose" else "…"
        self._echo(f"detecting nuclei ({algo_name}) on the {channel} MIP{weights}")
        _launch_worker(
            self, w, slot="_spot_worker",
            # `problem` keeps its subscriber ORDER: the console's failed line, then the echo.
            on_problem=[lambda m, a=action, d=where: self.log.failed(a, str(m), address=d),
                        self._echo],
            signals={
                "ready": self._on_nuclei_ready,
                "finished_count": lambda r, c, n, a=action, d=where, t0=began: (
                    self.log.done(f"{a}: {n} nuclei", time.monotonic() - t0, address=d),
                    self._echo(f"{n} nuclei detected on {c}."),
                ),
            })

    def _on_nuclei_ready(self, region, channel, labels, centroids, bbox_um, count):
        """Lay the label mask over the mosaic as a napari Labels layer, aligned to the raw channel."""
        v = self._napari_viewer()
        pane = self._pane
        mosaic = getattr(pane, "mosaic", None) if pane is not None else None
        if v is None or mosaic is None or labels is None:
            return
        raw = mosaic.find(_RAW_OP, channel)
        name = f"nuclei: {channel}"
        try:
            for lyr in list(v.layers):
                if getattr(lyr, "name", "") == name:
                    v.layers.remove(lyr)
        except Exception:                                 # noqa: BLE001
            pass
        kw = {"name": name}
        try:
            if raw is not None and getattr(raw, "scale", None) is not None:
                kw["scale"] = tuple(raw.scale[-2:])
            if raw is not None and getattr(raw, "translate", None) is not None:
                kw["translate"] = tuple(raw.translate[-2:])
        except Exception:                                 # noqa: BLE001 - overlay still lands at origin
            pass
        try:
            v.add_labels(np.asarray(labels).astype("uint32"), **kw)
        except Exception as exc:                          # noqa: BLE001 - named, never silent
            self._say(f"could not lay the nuclei mask: {exc}")

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

    def _set_ndisplay(self, n: int) -> None:
        v = self._napari_viewer()
        if v is None:
            self._say(f"cannot switch to {n}D — the napari viewer isn't available here.")
            return
        try:
            v.dims.ndisplay = int(n)
        except Exception as exc:                         # noqa: BLE001 - named, never silent
            self._say(f"could not switch to {n}D: {exc}")

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
        from squidxplorer._viewer import _FocusWorker

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
                self._say(f"sharpest plane is z={z_index}, but no z slider could be moved — "
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
    def _view_roi_2d(self) -> None:
        _roi_tools.view_roi_2d(self)

    @staticmethod
    def _sync_roi_width(viewer, layer, screen_px: float = 3.0) -> None:
        _roi_tools.sync_roi_width(viewer, layer, screen_px)

    def _roi_shapes_layer(self, create: bool = False):
        return _roi_tools.roi_shapes_layer(self, create)

    def _on_roi_data(self, layer) -> None:
        _roi_tools.on_roi_data(self, layer)

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

    # -- the LUT gestures live in `_lut_clipboard` (one clipboard, one home); thin delegates
    # -- because tests and the chips actuate these by name on the window. -------------------------
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
        for name, on in (visibility or {}).items():
            try:
                mosaic.set_channel_visible(str(name), bool(on))
            except Exception:                            # noqa: BLE001 - a missing channel is skipped
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
        self._refresh_divergence()

    def _copy_luts(self) -> None:
        _lut_clipboard.copy_luts(self)

    def _match_raw_contrast(self) -> None:
        """Raw's contrast onto every operator layer. See `_lut_clipboard.match_raw_contrast`."""
        _lut_clipboard.match_raw_contrast(self)

    def _paste_luts(self) -> None:
        _lut_clipboard.paste_luts(self)

    @property
    def time_point(self) -> int:
        """Which timepoint THIS window is showing."""
        bar = getattr(self, "_time_point_bar", None)
        return bar.time_point if bar is not None else 0

    def _on_time_point_changed(self, time_point: int) -> None:
        """A user moved THIS window's timepoint, or playback advanced it. Reload the mosaic."""
        self._say(f"time_point {time_point + 1} of {self._time_point_bar.count}")
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
        """3D of THIS view, closed-then-read structurally. See `_volume_view.open_3d`."""
        _volume_view.open_3d(self)

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
        if text:
            self.view_log().info("%s", text)
        self._echo(text)

    def _echo(self, text: str) -> None:
        """The in-window status line only, for events the console already carries a structured line for."""
        if self._pane is not None and getattr(self._pane, "ok", False):
            self._pane.say(text)

    def set_active(self, active: bool) -> None:
        """Halt draw/refresh on windows the user is not touching."""
        if active:
            return
        for control in (self._slider, getattr(self, "_time_point_bar", None)):
            if control is None:
                continue
            try:
                if control.is_playing:
                    control.stop()
            except Exception:                        # noqa: BLE001 - best effort
                pass

    def changeEvent(self, event):                    # noqa: N802 - Qt naming
        from qtpy.QtCore import QEvent

        if event.type() == QEvent.ActivationChange:
            self.set_active(self.isActiveWindow())
        super().changeEvent(event)

    def resizeEvent(self, e):                        # noqa: N802 - Qt naming
        """Grow the type with the window, exactly as the root does."""
        super().resizeEvent(e)
        rescale_fonts(self)

    def closeEvent(self, event):                     # noqa: N802 - Qt naming
        try:
            if self._worker is not None and self._worker.isRunning():
                self._worker.stop()
                self._worker.wait(2000)
        except Exception:                            # noqa: BLE001
            pass
        try:
            if self._spot_worker is not None and self._spot_worker.isRunning():
                self._spot_worker.stop()
                self._spot_worker.wait(2000)
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
            if self._slider is not None:
                self._slider.shutdown()
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
        self.closed.emit(self)
        super().closeEvent(event)


class ViewerManager(QObject):
    """Registry of open :class:`RegionViewer` windows, keyed by a monotonic ID."""

    windowsChanged = Signal()
    memoryChanged = Signal(float)
    runProgressChanged = Signal(object)
    viewFocused = Signal(object)
    windowOpened = Signal(object)

    def __init__(self, reader: Any = None, meta: Optional[dict] = None,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._reader = reader
        self._meta = meta
        self._windows: "dict[int, RegionViewer]" = {}
        self._next_id = 1
        self._focused_id: Optional[int] = None
        self._selected_ids: "list[int]" = []
        self.operator_specs: "list" = []
        self.run_operator: Optional[Any] = None
        self.defaults = ViewDefaults()

        self._run_progress = None

        self._mem_timer = QTimer(self)
        self._mem_timer.setInterval(2000)
        self._mem_timer.timeout.connect(self._poll_memory)
        self._mem_timer.start()

    def set_dataset(self, reader: Any, meta: dict) -> None:
        self._reader, self._meta = reader, meta

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
            out.append(View(
                id=f"w{win.window_id}", name=win.display_name,
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
                   parent_id: Optional[int] = None, luts: Optional[dict] = None) -> Optional[RegionViewer]:
        """Open a CHILD window from an ROI drawn in a parent window (the next level of the tree)."""
        regions = [str(r) for r in regions if r]
        if not regions:
            return None
        base = RegionViewer._view_label(regions)
        title = f"{base}  ◂ view {parent_id}" if parent_id is not None else base
        return self._spawn(regions, title=title, roi_bbox=roi_bbox, parent_id=parent_id, luts=luts)

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

    def rename(self, window_id: int, name: "Optional[str]") -> bool:
        """Give window *window_id* a new display label; False when the id is unknown or the name blank."""
        win = self._windows.get(int(window_id))
        if win is None:
            return False
        if not win.set_display_name(name):
            return False
        self.windowsChanged.emit()
        return True

    def make_default(self, window_id: int) -> bool:
        """Adopt one window's settings as the global defaults, for windows opened FROM NOW ON."""
        win = self._windows.get(int(window_id))
        if win is None:
            return False
        for name, value in win.current_settings().items():
            self.defaults.set(name, value)
        win.settings.adopt()
        win._refresh_divergence()
        return True

    def _spawn(self, regions: "list[str]", *, title: Optional[str] = None,
               roi_bbox: Optional[tuple] = None,
               parent_id: Optional[int] = None, luts: Optional[dict] = None) -> Optional[RegionViewer]:
        if self._reader is None or self._meta is None:
            log.warning("open() called before a dataset was loaded; ignoring.")
            return None
        wid = self._next_id
        self._next_id += 1
        n = len(regions)
        clock = _measure.WindowOpen(
            f"{'ROI in ' if roi_bbox is not None else ''}{n} region{'' if n == 1 else 's'}: "
            f"{RegionViewer._view_label(regions)}",
            n_targets=n)
        baseline = self._baseline_for(parent_id)
        if luts is not None:
            baseline["luts"] = luts
        win = RegionViewer(
            self._reader, self._meta, regions, window_id=wid, title=title,
            manager=self, roi_bbox=roi_bbox,
            operator_specs=self.operator_specs, run_operator=self.run_operator,
            parent_id=parent_id, settings=ViewSettings(baseline),
        )
        win.open_clock = clock
        win.closed.connect(self._on_window_closed)
        self._windows[wid] = win
        self._focused_id = wid
        self._selected_ids = [wid]
        win.show()
        win.raise_()
        win.activateWindow()
        self._replay_cached_results(win)
        self.windowOpened.emit(win)
        self.windowsChanged.emit()
        self.viewFocused.emit(list(win._regions))
        return win

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
                     "(no recompute) — toggle them in the layers panel.")
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

    def focus(self, window_id: int) -> None:
        win = self._windows.get(int(window_id))
        if win is not None:
            self._focused_id = int(window_id)
            win.showNormal()
            win.raise_()
            win.activateWindow()
            self.viewFocused.emit(list(win._regions))

    def raise_views(self, ids: "Sequence[int]") -> None:
        """Bring the selected windows to the front, un-minimising each; activate the last for focus."""
        wins = [self._windows.get(int(i)) for i in ids]
        wins = [w for w in wins if w is not None]
        for w in wins:
            try:
                w.showNormal()
                w.raise_()
            except Exception:                            # noqa: BLE001 - best effort per window
                pass
        if wins:
            try:
                wins[-1].activateWindow()
            except Exception:                            # noqa: BLE001
                pass

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
            win.close()

    def close_all(self) -> None:
        for win in list(self._windows.values()):
            win.close()

    def collapse_all(self) -> None:
        """Minimise every open window at once; they stay in the navigator and a row click restores one."""
        for win in list(self._windows.values()):
            try:
                win.showMinimized()
            except Exception:                            # noqa: BLE001 - best effort per window
                pass
        self._focused_id = None
        self.viewFocused.emit([])

    def _on_window_closed(self, win: "RegionViewer") -> None:
        clock = getattr(win, "open_clock", None)
        if clock is not None:
            clock.finish(_measure.STOPPED, "closed before its mosaic landed")
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
    """The "Open View list": every open window by ID, plus a live memory bar."""

    def __init__(self, manager: ViewerManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._manager = manager

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

        header = QLabel("Window navigator")
        header.setStyleSheet("color:#c9d1d9;font-size:13px;font-weight:600;border:none;")
        lay.addWidget(header)

        self._tree = QTreeWidget(self)
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._tree.setFocusPolicy(Qt.StrongFocus)
        self._tree.itemActivated.connect(self._on_activated)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self._syncing = False
        lay.addWidget(self._tree, 1)

        row = QHBoxLayout()
        row.setSpacing(6)
        self._close_btn = QPushButton("Close selected views")
        self._close_btn.clicked.connect(self._close_selected)
        self._collapse_btn = QPushButton("Collapse all")
        self._collapse_btn.clicked.connect(self._manager.collapse_all)
        self._close_all_btn = QPushButton("Close all")
        self._close_all_btn.clicked.connect(self._close_all)
        row.addWidget(self._close_btn)
        row.addWidget(self._close_all_btn)
        row.addWidget(self._collapse_btn)
        row.addStretch(1)
        lay.addLayout(row)
        self._refresh_nav_buttons()

        self._mem_label = QLabel("Memory")
        self._mem_label.setStyleSheet("color:#8b949e;font-size:11px;border:none;")
        lay.addWidget(self._mem_label)
        self._mem_bar = QProgressBar(self)
        self._mem_bar.setRange(0, 100)
        self._mem_bar.setTextVisible(True)
        self._mem_bar.setFixedHeight(14)
        lay.addWidget(self._mem_bar)

        self._work_label = QLabel("")
        self._work_label.setStyleSheet("color:#8b949e;font-size:11px;border:none;")
        self._work_label.setWordWrap(True)
        self._work_label.hide()
        lay.addWidget(self._work_label)
        self._work_bar = QProgressBar(self)
        self._work_bar.setTextVisible(False)
        self._work_bar.setFixedHeight(14)
        self._work_bar.setStyleSheet(
            "QProgressBar{background:#161b22;border:1px solid #30363d;border-radius:3px;}"
            "QProgressBar::chunk{background:#1f6feb;border-radius:3px;}"
        )
        self._work_bar.hide()
        lay.addWidget(self._work_bar)

        manager.windowsChanged.connect(self.refresh)
        manager.memoryChanged.connect(self._on_memory)
        manager.runProgressChanged.connect(self._on_run_progress)
        self._on_run_progress(manager.run_progress)
        self.refresh()

    def take_status_row(self) -> tuple:
        """Hand the memory bar and the run-progress bar to whoever is going to show them."""
        lay = self.layout()
        widgets = (self._mem_label, self._mem_bar, self._work_label, self._work_bar)
        for w in widgets:
            lay.removeWidget(w)
        return widgets

    def showEvent(self, e):
        """Hand the tree keyboard focus, and give the arrows a row to start from."""
        super().showEvent(e)
        self._tree.setFocus()
        if self._tree.currentItem() is None and self._tree.topLevelItemCount():
            self._syncing = True
            try:
                self._tree.setCurrentItem(self._tree.topLevelItem(0))
            finally:
                self._syncing = False

    def refresh(self) -> None:
        if _sip is not None and _sip.isdeleted(self):
            return
        self._syncing = True
        try:
            self._tree.clear()
            items: "dict[int, QTreeWidgetItem]" = {}
            windows = self._manager.windows
            by_id = {int(w.window_id): w for w in windows}
            for win in sorted(windows, key=lambda w: int(w.window_id)):
                wid = int(win.window_id)
                item = QTreeWidgetItem([win.windowTitle()])
                item.setData(0, Qt.UserRole, wid)
                pid = getattr(win, "parent_id", None)
                parent_item = items.get(int(pid)) if pid is not None and int(pid) in by_id else None
                if parent_item is not None:
                    parent_item.addChild(item)
                else:
                    self._tree.addTopLevelItem(item)
                items[wid] = item
            self._tree.expandAll()
            selected = set(self._manager.selected_ids)
            for wid, item in items.items():
                if wid in selected:
                    item.setSelected(True)
        finally:
            self._syncing = False
        self._refresh_nav_buttons()

    def _on_selection_changed(self) -> None:
        """Row selection is the wash and the operator target set; empty selection clears the wash."""
        if self._syncing:
            return
        ids = [int(i) for i in (it.data(0, Qt.UserRole) for it in self._tree.selectedItems())
               if i is not None]
        self._refresh_nav_buttons()
        self._manager.set_selected(ids)
        self._manager.raise_views(ids)

    def _on_activated(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        wid = item.data(0, Qt.UserRole)
        if wid is not None:
            self._manager.focus(int(wid))

    def _on_context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if item is None:
            return
        wid = item.data(0, Qt.UserRole)
        if wid is None:
            return
        menu = QMenu(self)
        act = menu.addAction("Rename…")
        if menu.exec(self._tree.viewport().mapToGlobal(pos)) is act:
            self.rename_window(int(wid))

    def rename_window(self, window_id: int) -> bool:
        """Ask for a new label for *window_id* and apply it. Returns whether anything changed."""
        win = self._manager._windows.get(int(window_id))
        if win is None:
            return False
        text, ok = QInputDialog.getText(
            self, f"Rename view [{window_id}]",
            f"Label for view [{window_id}] (the [{window_id}] prefix stays, so log lines still "
            f"point here):",
            text=win.display_name)
        if not ok:
            return False
        return self._manager.rename(int(window_id), text)

    def _refresh_nav_buttons(self) -> None:
        """Enable each navigator button only when it has something to act on, and SAY WHY not."""
        n = len(self._manager.windows)
        selected = len(self._tree.selectedItems())
        self._close_btn.setEnabled(bool(selected))
        self._close_btn.setToolTip(
            "Close every view selected here (shift/ctrl-click to select several)." if selected
            else ("No view is selected, so there is nothing to close. Click a row above first."
                  if n else "No view is open."))
        self._close_all_btn.setEnabled(bool(n))
        self._close_all_btn.setToolTip(
            f"Close all {n} open view(s), selected or not." if n
            else "No view is open, so there is nothing to close.")
        self._collapse_btn.setEnabled(bool(n))
        self._collapse_btn.setToolTip(
            "Minimise every open window (click a row to bring one back)." if n
            else "No view is open, so there is nothing to minimise.")

    def _close_all(self) -> None:
        """Close every open view, whatever is selected."""
        for wid in [int(w.window_id) for w in self._manager.windows]:
            self._manager.close(wid)

    def _close_selected(self) -> None:
        """Close EVERY selected row, not just the current one."""
        ids = [int(i) for i in (it.data(0, Qt.UserRole) for it in self._tree.selectedItems())
               if i is not None]
        for wid in ids:
            self._manager.close(wid)

    def _on_run_progress(self, report) -> None:
        """Draw (or take down) the work bar. ``report`` is a ``ProgressReport``, or None for idle."""
        if report is None:
            self._work_label.hide()
            self._work_bar.hide()
            return
        try:
            sentence, percent = report.sentence(), report.percent
        except Exception:                            # noqa: BLE001 - a bad report is not a crash
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

    def _on_memory(self, frac: float) -> None:
        pct = max(0, min(100, int(round(frac * 100))))
        self._mem_bar.setValue(pct)
        warn = pct >= 85
        self._mem_label.setText("Memory — HIGH, close a view" if warn else "Memory")
        color = "#f85149" if warn else "#3fb950"
        self._mem_bar.setStyleSheet(
            "QProgressBar{background:#161b22;border:1px solid #30363d;border-radius:3px;}"
            f"QProgressBar::chunk{{background:{color};border-radius:3px;}}"
        )


def _process_memory_fraction() -> Optional[float]:
    """This process's RSS as a fraction of total system RAM, or None if it can't be measured."""
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        return float(proc.memory_info().rss) / float(psutil.virtual_memory().total)
    except Exception:                                # noqa: BLE001 - psutil optional
        pass
    try:
        import resource

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys

        rss = float(rss_kb) if sys.platform == "darwin" else float(rss_kb) * 1024.0
        import os

        total = float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        return rss / total if total > 0 else None
    except Exception:                                # noqa: BLE001
        return None
