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

from squidmip import _measure
from squidmip._time_point import TimePointBar
from squidmip._address import Address, Extent
from squidmip._logpane import ViewLog, get_logger
from squidmip._fontscale import rescale_fonts, window_screen

log = get_logger("regionviewer")

#: PyQt's C++-object-liveness oracle. The ONLY way to ask "has Qt already destroyed this widget",
#: which a slot connected to a longer-lived object has to ask before it touches its own children.
#: Optional so a binding without it degrades to today's behaviour rather than failing to import.
#: sip lives under the BINDING, not under qtpy: PyQt5.sip, PyQt6.sip, and nothing at all under
#: PySide (which uses shiboken instead). qtpy tells us which binding is live, so ask it rather than
#: guessing, and degrade to today's behaviour when there is no sip to ask.
_sip = None
try:                                                     # pragma: no cover - binding detail
    import importlib

    import qtpy as _qtpy

    if _qtpy.API_NAME.startswith("PyQt"):
        _sip = importlib.import_module(f"{_qtpy.API_NAME}.sip")
except Exception:                                        # pragma: no cover
    try:
        import sip as _sip                               # older PyQt5 packagings, top-level sip
    except ImportError:
        _sip = None

#: Cross-window LUT clipboard for Julio's "sync windows = copy/paste LUTs": one window's per-channel
#: (contrast_limits, colormap) is stashed here by "Copy LUTs" and applied by "Paste LUTs" in any
#: other window (or the plate). A parameter file on the desktop is the same idea; this is the
#: in-session GUI form of it. Keyed by channel name -> {"clim": (lo, hi), "cmap": <name>,
#: "rgb": (r, g, b) | None}. ``cmap`` is what a napari layer is SET to; ``rgb`` is what that
#: colormap LOOKS like reduced to one 8-bit colour, for consumers that store a colour and not a
#: ramp (the Minerva export). ``None`` there means "this colormap is not one colour" - see
#: :func:`squidmip._napari_view.colormap_hue_rgb`.
_LUT_CLIPBOARD: "dict[str, dict]" = {}

#: Distinct edge colours cycled per ROI so each annotation box is told apart (Julio: "roi boxes
#: should have different colors"). A qualitative set, high-contrast on tissue.
_ROI_COLORS: "tuple[str, ...]" = (
    "#58a6ff", "#f778ba", "#3fb950", "#f0883e", "#a371f7", "#e3b341", "#39c5cf", "#ff7b72",
)


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
    model + the engine hook (``run_on_view``) is the plumbing under it.

    ``name`` carries the window's LABEL WITHOUT the ``[wid]`` bracket (2026-08-03). It used to carry
    ``windowTitle()``, bracket included, and that was safe only while nothing read it. Now that
    ``_run_scope.describe_view_target`` prints it, an id living inside a string field is an id that
    can drift from ``window_id`` sitting beside it, so the printer composes ``[{window_id}] {name}``
    from the two fields and neither one spells the other."""
    id: str
    name: str
    regions: tuple
    kind: str = "window"
    window_id: Optional[int] = None
    roi_bbox: Optional[tuple] = None
    parent_id: Optional[int] = None


#: Baseline mode: the manager's global default, whoever opened the window.
_GLOBAL = "global"
#: Baseline mode: the OPENER's current value, falling through to the global default when the plate
#: opened the window, because the plate has no window of its own to inherit from.
_INHERIT = "inherit"

#: The settings that are GLOBAL DEFAULTS, and how each one picks its baseline in a new window.
#:
#: Julio's rule, 2026-07-29: **settings that describe HOW you look are global defaults; settings
#: that describe WHAT you are looking at are per-window.** Only the first class appears here. 2D/3D,
#: the region cursor and the z / time_point position are the second class, and neither
#: :class:`ViewDefaults` nor :class:`ViewSettings` has a word for them. That silence is the design:
#: a per-window setting is not "a default that is always overridden", it is not a default at all,
#: and giving it a slot here would invite somebody to seed it.
#:
#: ``luts`` is ``_INHERIT``, and that ONE line is the whole contrast rule. The decision was written
#: as two rules -- the global default for a window opened from the plate, the parent's LUTs for an
#: ROI child -- but they are one rule stated twice, because a plate-opened window's opener IS the
#: default. Collapsing them costs nothing and still gives Julio the behaviour he asked for: an ROI
#: child looks like its parent, which is what makes a crop comparable to the source it was cut from.
_SETTING_BASELINE = {
    "tenengrad_focus": _GLOBAL,
    "channel_visibility": _GLOBAL,
    "luts": _INHERIT,
}


def _copy_setting(value: Any) -> Any:
    """A private copy of one setting's value, so two windows can never share a mutable one.

    Two levels deep, which is exactly what these settings are: ``channel_visibility`` is
    ``{channel: bool}`` and ``luts`` is ``{channel: {"clim": ..., "cmap": ...}}``. Deliberately not
    ``deepcopy``: a ``cmap`` can be a live napari colormap object when it has no name, and copying
    that is neither wanted nor safe.
    """
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
    """The global defaults a NEW window opens with. Owned by :class:`ViewerManager`.

    On the manager and not on the plate window because windows come and go while the manager is the
    registry: the registry is the one object whose lifetime the defaults can safely share.

    Every field's default is "no opinion", so this object existing changes nothing until somebody
    sets something. ``tenengrad_focus`` off means a window does not autofocus unless asked;
    ``channel_visibility`` and ``luts`` empty mean the channels come up exactly as napari made them,
    rather than a default fighting a colormap nobody asked it to change.
    """

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
    """ONE window's global-default settings: the BASELINE it opened with, plus the overrides the
    user made in that window.

    Divergence is tracked EXPLICITLY, as the set of names overridden here, and not computed as
    "differs from today's default". The difference shows up twice and both cases are wrong the
    other way round:

    * Changing the default later must not light a marker on a window nobody touched, and must not
      silently move one either. A window reads the defaults once, at construction; after that, the
      default is a fact about the NEXT window.
    * An ROI child that inherited a diverged parent's contrast has not itself diverged. It shows
      what it was opened with, and that is the honest thing to offer a reset back to.

    So :meth:`reset` restores what the window opened with, not today's default. Pushing the other
    way is :meth:`ViewerManager.make_default`, an explicit act; nothing here ever reaches outward,
    because propagating one window's change to the others fights the independence the
    decentralization bought.
    """

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
        """Change *name* FOR THIS WINDOW, and report whether it is now diverged on it.

        Setting a value back to the baseline clears the override rather than leaving a sticky
        marker: the marker has to mean "you are not looking at what this window opened with", so it
        cannot survive the value going back.
        """
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
        """Take the current values as the new baseline, so nothing reads as diverged any more.

        Called after "make this the default": the window's settings ARE the default now, and a
        marker still claiming otherwise would be the silent disagreement this whole affordance
        exists to prevent.
        """
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

#: The Apple floor for GL_MAX_3D_TEXTURE_SIZE, used only until the live canvas can be asked. Not a
#: second literal: it is the one ``_napari_view`` owns.
from squidmip._napari_view import (                            # noqa: E402  (kept beside its use)
    _DEFAULT_MAX_3D_TEXTURE,
    full_res_level,
)


def _brick_budget_bytes() -> int:
    """How much a bricked 3D view may hold resident.

    ``_budget.cache_budget`` is the repo's one answer to "how much of this machine may a cache
    take", measured off FREE memory rather than total, so this does not invent a fourth memory
    mechanism. It is deliberately a share of the same budget the 2D pyramid cache uses: a 3D view is
    up instead of heavy 2D navigation, not as well as it.
    """
    try:
        from squidmip._budget import cache_budget

        return int(cache_budget())
    except Exception:                                    # noqa: BLE001 - a floor beats no render
        return 512 << 20


def _started(vol):
    """``open()`` the volume and hand it back, so ``_replace_native3d`` still takes one callable."""
    vol.open()
    return vol


class RegionViewer(QMainWindow):
    """ONE independent napari window over a subset of regions.

    Owns its own napari pane, its own region cursor + slider, and its own mosaic-load pipeline.
    Shares the app's single ``reader``/``meta`` (stateless reads). Closing it stops its worker and
    joins its slider's animation thread so a close during playback cannot abort the process.
    """

    closed = Signal(object)   # emits self, so the registry can drop it

    #: The open half of this window's console pair, and the region its operator layers describe.
    #: CLASS defaults as well as ``__init__`` assignments, for the same reason
    #: ``PlateWindow._result_acc`` is one: a bare ``getattr`` on a QObject whose ``__init__`` has
    #: not run raises out of Qt's own attribute machinery instead of returning the default, and
    #: these are read from slots that fire on windows built by tests and by Qt alike.
    _op_action: Optional[str] = None
    _op_address: Any = None
    _result_region: Optional[str] = None
    #: This window's operator-run bar, built in the control row. Same class-default rule: the four
    #: ``operator_*`` callbacks are called by the plate on whatever window asked, including one a
    #: test built without the row.
    _op_progress: Any = None
    #: The :class:`squidmip._measure.WindowOpen` clock for THIS window's open, set by
    #: ``ViewerManager._spawn`` because the clock starts before the window exists. Same class-default
    #: rule: ``_on_plane`` and ``_on_done`` are slots and fire on windows a test built directly,
    #: which never went through ``_spawn`` and so are honestly unmeasured rather than broken.
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
        #: Which mosaic load this window is waiting for. Bumped by every `_load_mosaic`, carried
        #: by every result, and checked on arrival: that is how a superseded read is DROPPED
        #: rather than waited for on the GUI thread. See `_load_mosaic`.
        self._load_gen = 0
        self._retired_workers: list = []       # superseded loads, held until Qt reaps them
        #: WHICH REGION'S MOSAIC IS IN THE PANE. One fact, two consequences, and they are the same
        #: question asked twice: a reload of the SAME region must not re-frame the camera
        #: (`_on_done`) and must not destroy the layers to rebuild them (`_load_mosaic`). A reload
        #: of a DIFFERENT region must do both — its mosaic is a different shape, and napari does
        #: not survive being handed one through a reused layer.
        self._shown_region: Optional[str] = None
        self._pending_region: Optional[str] = None
        self._load_timer: Optional[QTimer] = None
        self._time_load_timer: Optional[QTimer] = None
        self._pane = None
        self._slider = None
        self._cursor = None
        self._native3d = None      # THE 3D popout of this window; see _replace_native3d
        self._spot_worker = None   # nuclei detection (Cellpose) on this view's MIP, off-thread
        self._focus_worker = None  # Tenengrad reference-plane autofocus, off-thread
        self._video_worker = None  # .mp4 export of this view's T (or Z) sweep, off-thread
        # OPERATOR CONTROLS AT EACH LEVEL (the deck: "Operators for this window"; Julio, 2026-07-23:
        # "I don't see operator controls like the powerpoint specified at each level"). This is not a
        # contradiction of "operators work on Views" -- it IS that: the window's operator control runs
        # the SAME registry on THIS view's regions. Selecting where to run stitching = pick the view,
        # run it here. The manager also lets an ROI open a CHILD window (the view tree).
        self._manager = manager
        self._operator_specs = list(operator_specs or [])
        self._run_operator = run_operator
        self.parent_id = parent_id      # the view this was spawned from (ROI child) -> tree nesting
        # THE HOW-YOU-LOOK SETTINGS FOR THIS WINDOW (Task 6, 2026-07-29). Read ONCE, at
        # construction, from whatever the manager handed down: the global defaults, except that
        # contrast comes from the opener, so an ROI child looks like its parent. From here on the
        # window owns them, and changing one changes THIS window and marks it diverged. A window
        # built without a manager gets the stock defaults so it is never settings-less.
        self.settings = settings if settings is not None else ViewSettings()
        self._settings_applied = False
        # An ROI child carries the parent's ROI box (deck: "ROI -> child window"). Cropping the load
        # to it lands with the loader work; today it scopes the title + is recorded for that step.
        self._roi_bbox = roi_bbox
        self._roi_layer = None     # the napari Shapes layer this window draws ROI rectangles on
        # THIS WINDOW'S OPERATOR RUN (2026-07-29). ``_op_action`` / ``_op_address`` are the open
        # half of the console's started/done pair, captured when the run STARTS and carried into
        # whichever line closes it -- see operator_started for why the address is never re-read.
        # ``_result_region`` is the region the operator layers in this window describe, so a region
        # change can drop a layer that would otherwise keep claiming to describe the region it was
        # computed on while sitting over a different one.
        self._op_action: Optional[str] = None
        self._op_address: Any = None
        self._result_region: Optional[str] = None

        # Name the window by the regions it holds (the deck shows the slider as "<> A1, B6, C3"),
        # not "N regions" — Julio: "'2 regions' is a bad name". Truncate a long list so the title
        # bar stays readable, keeping the count only as an overflow tail.
        #
        # THE ID IS NOT THE NAME (Julio, 2026-08-03: "we should be able to rename our windows ...
        # this might break our logging and data model"). It does not, and the split below is why:
        # ``window_id`` is the identity and it is immutable, ``_display_name`` is a LABEL and it is
        # not. Everything functional — the registry, the log prefix, navigator row identity, tree
        # nesting, contrast inheritance, ``make_default``, the plate's followed-windows set — keys
        # on the int and never on this string, and nothing anywhere parses a window title. So a
        # rename moves the label and moves nothing else. The ``[wid]`` prefix is rendered here and
        # is NOT user-editable: it is the only visible join between a log line ("[3] A1 fov 2 ...",
        # `_logpane._address_prefix`) and a window on the desktop, and `_address.py`'s naming law
        # makes that join the point.
        self._derived_name = title or self._region_label(self._regions)
        if self._roi_bbox is not None:
            self._derived_name = f"ROI · {self._derived_name}"
        self._display_name = self._derived_name
        self._refresh_title()
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        # THE LOGGER FOR THIS WINDOW (Task 1, 2026-07-29). Every line it emits carries two things:
        # the VIEW id, which is this window's ordinal and ours, and the ADDRESS, which is where in
        # the acquisition the action happened and is spelled in Squid's words. So a line reads
        # "[3] A1 fov 2  decon(sigma=2.0)  started" and the one global console can print from every
        # open window without any of them relying on the user knowing which window they meant.
        #
        # This replaces `view_tag`, a string of the form "V3:ROI@A1" that packed the view id and
        # the place into one token. Deleted rather than kept: a second spelling of the same two
        # facts is exactly the drift `_address.py`'s naming law exists to stop.
        self.log = ViewLog(log, self.window_id)

        # A modest, cascaded window — the deck's windows are small tiles, not full-screen slabs.
        # Cascade by ID so several opened in a row do not land exactly on top of one another.
        #
        # 860x720 is a DESIGN size in logical pixels, and it is only the right size if the display
        # scale is being honoured. Where it is not (Qt5 rounding a 150% display down to 100%, see
        # `_viewer.enable_hidpi`) this window came up at 860 PHYSICAL pixels on a 4K panel, which
        # is about three inches wide: Spencer's report. The rounding fix is the actual repair, so
        # this floor is a SECOND line of defence rather than the fix -- a view window is never
        # worth opening at under a third of the screen, whatever the scale factor turns out to be.
        #
        # The cascade offset is measured from the HOME SCREEN's work area, not from the desktop
        # origin. `move(120, 90)` is a GLOBAL coordinate, so on a laptop + external monitor it put
        # every view window on whichever display owns (0, 0) -- the plate could be on the external
        # and its views would open on the laptop, at a size computed from a third display's
        # geometry. Same numbers, now relative to the screen the plate is on.
        self.resize(*self._default_view_size())
        off = 28 * ((self.window_id - 1) % 8)
        home = self._home_screen()
        origin = home.availableGeometry().topLeft() if home is not None else None
        ox, oy = (origin.x(), origin.y()) if origin is not None else (0, 0)
        self.move(ox + 120 + off, oy + 90 + off)

        self._build()

    #: The shape a view window OPENS at, in logical pixels. Deliberately not named `_DESIGN_W`:
    #: that name already means "the width the TYPE was authored for" in `_fontscale`, and this
    #: window scales its type against that module's 1100, not against this. Two different facts,
    #: two different names.
    _OPEN_W, _OPEN_H = 860, 720

    def _home_screen(self):
        """The display this window belongs on: the PLATE's, else this window's, else the primary.

        A view window is parentless and unshown while it is being sized, so asking Qt which screen
        it is on answers "the primary one" no matter where the user is working. The plate IS
        reachable, though, by exactly the route ``ViewerManager.raise_plate`` already uses: the
        manager's Qt parent is the ``PlateWindow``. A view opens from the plate, so the plate's
        display is the right one to measure and place against.
        """
        opener = self._manager.parent() if self._manager is not None else None
        return window_screen(opener if opener is not None else self)

    def _default_view_size(self) -> tuple:
        """The design size, floored at a third of the screen and capped to fit on it.

        Measured against the screen the window will OPEN on (see :meth:`_home_screen`), not
        ``primaryScreen()``: on a laptop + external monitor those are different displays with
        different work areas, and "a third of the screen" computed from the wrong one is the
        cross-monitor size inconsistency, not a rounding artefact.
        """
        w, h = self._OPEN_W, self._OPEN_H
        screen = self._home_screen()
        if screen is None:
            return w, h
        avail = screen.availableGeometry()
        w = min(max(w, avail.width() // 3), max(1, avail.width() - 40))
        h = min(max(h, avail.height() // 3), max(1, avail.height() - 80))
        return int(w), int(h)

    @staticmethod
    def _region_label(regions: "list[str]", limit: int = 3) -> str:
        if not regions:
            return "(empty)"
        if len(regions) <= limit:
            return ", ".join(regions)
        return ", ".join(regions[:limit]) + f", +{len(regions) - limit}"

    # -- the name, which is not the identity ---------------------------------------------
    @property
    def display_name(self) -> str:
        """The window's LABEL, without the ``[wid]`` bracket. Mutable; the id is not."""
        return self._display_name

    def _refresh_title(self) -> None:
        """Render identity + label into the title bar. The ONE place the two are joined."""
        self.setWindowTitle(f"[{self.window_id}] {self._display_name}")

    def set_display_name(self, name: "Optional[str]") -> bool:
        """Rename this window. Returns False for a blank name, which is a refusal, not a reset.

        An empty box in the rename dialog means "I changed my mind", so it must not silently wipe
        the region-derived name a user relies on to tell two windows apart. Passing ``None``
        explicitly RESTORES the derived name, which is the deliberate undo.
        """
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

        # Wire the pane's OWN "Detect on: [channel] Detect nuclei" strip (the channel-aware Cellpose
        # picker). It was only connected for the old central pane, so in a window it was a dead
        # button -- Julio's "I can't detect nuclei on my ROI". Populate the channel list, enable it,
        # and run detection on THIS view (the ROI crop for an ROI child).
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

        # Each window navigates time INDEPENDENTLY: that is the point of the decentralization, and
        # a shared position would mean comparing two wells at the same timepoint was impossible.
        # Same widget CLASS as the plate's, deliberately, so the two can never disagree about what
        # a timepoint control is. Hidden at n_t == 1, so this call site stays unconditional.
        #
        # WITH PLAYBACK, and only here. A window can honestly animate the time axis because its
        # picture comes from `_MosaicWorker`, which takes a `t` and fuses that timepoint. The
        # plate's bar cannot and does not: its preview cells are cached per (token, region) with
        # no timepoint, so a plate play button would animate timepoint 0's pixels under a moving
        # label. Same class, playback where the read path can serve it. See `_time_point`.
        self._time_point_bar = TimePointBar(on_change=self._on_time_point_changed, playback=True)
        self._time_point_bar.on_problem(self._say)
        self._time_point_bar.set_count(int((self._meta or {}).get("n_t", 1) or 1))
        lay.addWidget(self._time_point_bar)

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
    #: The operator dropdown's OWN chrome. `_CHIP_QSS` above is `QPushButton`-only, so a combo
    #: styled with it alone declared no `color` at all -- and `_build_top_row` sets a SELECTOR-LESS
    #: `background:#0b0e14` on the row, which Qt parses as `*{...}` and applies to every descendant.
    #: The background therefore came from the row's sheet and the FOREGROUND from the OS palette:
    #: white in macOS dark mode, BLACK in light mode, on a near-black box (Julio, light mode).
    #: Foreground and background are stated together here, for the closed combo AND for the popup
    #: `QAbstractItemView` (a separate top-level window that the `QComboBox` selector never reaches),
    #: so neither half is ever left to the platform palette to supply.
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

        # LEFT: the "2D 3D'" toggle + native-3D popout + ROI tools. The toggle drives the embedded
        # pane's ndisplay (which already renders 3D at max texture res, contrast preserved); the
        # popout is the single-FOV native volume for when the fused mosaic exceeds the GPU texture.
        view_box, vv = self._titled_box("2D / 3D · ROI")
        r1 = QHBoxLayout(); r1.setSpacing(4)
        self._btn_2d = self._chip("2D", "View the SELECTED ROI in 2D (opens it as a child window); "
                                  "with no ROI picked, just shows the mosaic in 2D.", self._view_roi_2d)
        # 3D is ONE thing: a NATIVE-resolution popout of this view (never the whole fused mosaic,
        # which exceeds the GPU texture and renders blocky). 2D just keeps the mosaic. No embedded
        # 3D toggle, no separate "native" button -- one behaviour, so the cases don't explode.
        self._btn_3d = self._chip("3D", "Open this view in 3D at NATIVE resolution (the region if it "
                                  "fits the GPU texture, else draw an ROI to pick the spot). "
                                  "Replaces this window's previous 3D view rather than adding "
                                  "another window.",
                                  self._open_3d)
        # Tenengrad autofocus, back on the slider (Julio): jump this window's z-slider to the
        # sharpest plane of the current region's centre FOV. The worker lived under the removed
        # central viewer; here it drives the window's own napari z dims.
        self._btn_focus = self._chip("⌖ focus", "Jump the z-slider to the sharpest plane "
                                     "(Tenengrad autofocus) of this region's centre FOV.",
                                     self._focus_reference_plane)
        # The way BACK. Every view is bigger than the plate it was opened from, so the plate ends up
        # under the pile it spawned. This sits in r1 with 2D/3D/focus rather than in r2 with the ROI
        # tools: it is a WINDOW action, not something you do to the mosaic.
        self._btn_plate = self._chip("▣ plate", "Bring the plate window to the front — it ends up "
                                     "buried under the views opened from it.", self._raise_plate)
        # ▣ plate's twin, and deliberately next to it: same journey back to the plate, but it also
        # opens the tab for the operator THIS window is showing, so the thing you came back to
        # change is already in front of you. Julio: "so that we can tweak, say the iterations".
        self._btn_controls = self._chip(
            "⚙ controls", "Bring the plate window forward AND open the controls for the operator "
            "this window is showing, so its parameters (iterations, thresholds) are one click "
            "away. Says so when the window is showing raw pixels, which have none.",
            self._show_operator_controls)
        # RECORD. A window shows ONE index of T and ONE of Z at a time, so the only way to look at
        # a time series or a focus sweep today is to drag a slider and remember. This exports the
        # sweep as a file — the axis the acquisition actually has (T if it is a time series, else
        # Z), the region on screen, the channels that are visible, the contrast that is set.
        #
        # ENABLED WHEN `n_t > 1 or n_z > 1`, which is `_video.can_record`, and the disabled tooltip
        # says which. Gating on n_t alone would hide the button on every acquisition on this
        # machine (all n_t=1) — see the rationale in `squidmip/_video.py`'s docstring.
        self._btn_record = self._chip(
            "⏺ movie", "Export what this window is showing as an .mp4, sweeping the acquisition's "
            "time axis (or its z axis when there is no time series). Runs off the UI thread; "
            "click again to cancel.", self._record_movie)
        r1.addWidget(self._btn_2d); r1.addWidget(self._btn_3d); r1.addWidget(self._btn_focus)
        r1.addWidget(self._btn_record)
        r1.addWidget(self._btn_plate)
        r1.addWidget(self._btn_controls)     # beside ▣ plate: both are the way BACK to the plate
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
        self._op_combo.setStyleSheet(self._COMBO_CHIP_QSS)
        for spec in self._operator_specs:
            self._op_combo.addItem(str(spec[1]), spec[0])   # label shown, key as data
        if self._op_combo.count() == 0:
            self._op_combo.addItem("no operators", None)
            self._op_combo.setEnabled(False)
        opr.addWidget(self._op_combo, 1)
        opr.addWidget(self._chip("Run", "Run the selected operator on THIS view's regions.",
                                 self._run_view_operator))
        # SAVE-TO-DISK toggle, OFF by default: a window run is normally a PREVIEW ("see how the
        # results would look"); only tick this to persist an OME-Zarr (Julio + the Spencer huddle).
        self._save_chk = QCheckBox("save")
        self._save_chk.setToolTip("Off = preview only (nothing written to disk). On = persist the "
                                  "operator result as an OME-Zarr.")
        self._save_chk.setStyleSheet("QCheckBox{color:#c9d1d9;font-size:11px;}")
        opr.addWidget(self._save_chk)
        ov.addLayout(opr)
        # HOW FAR THE RUN HAS GOT, in the box the Run button is in. Julio, 2026-08-03, on a decon
        # that took 433 s over one region: "there's nothing on the child window that tells me how
        # much is left what's the progress, or that it is working. It only tells me that it worked
        # after layers populated, but how long is that?" For those seven minutes the only moving
        # thing was the footprint line in the log, which is a memory printer being read as a
        # progress bar. This is the affordance that replaces reading the log.
        #
        # Hidden when idle, on purpose: a bar sitting at 0% over an idle window is indistinguishable
        # from a wedged run, which is the failure this is meant to cure rather than reproduce.
        self._op_progress = QProgressBar()
        self._op_progress.setTextVisible(True)
        self._op_progress.setFixedHeight(16)
        self._op_progress.setStyleSheet(self._PROGRESS_QSS)
        self._op_progress.hide()
        ov.addWidget(self._op_progress)
        # Nuclei detection lives on the pane's own "Detect on: [channel] Detect nuclei" strip (the
        # channel-aware Cellpose picker Julio asked for) -- wired to _detect_nuclei in _build. No
        # duplicate control here.
        # Row 2: contrast sync. THREE controls, TWO scopes, and the scopes are captioned because
        # that is the only thing that made them look like duplicates (Julio: "if contrast are
        # synched, the LUT copy paste should be removed", and "'Match raw contrast' seems like a
        # strange button to have"). They are not duplicates. What is actually synced automatically
        # is ONE scope: napari's `link_layers` keeps the operator layers of one channel together
        # INSIDE THIS WINDOW, from the next write on. Nothing at all crosses windows -- each
        # window builds its own napari ViewerModel (`MosaicPane.__init__` -> `build_pane`), and
        # napari cannot link layers across viewers. So:
        #
        #   between windows  -> copy/paste, the ONLY path. It also carries the COLORMAP, which no
        #                       link and no match ever touches (`link_layers(..., ("contrast_
        #                       limits",))`). The clipboard is shared with the plate's own
        #                       'LUTs' buttons, so plate <-> window works both ways.
        #   in this window   -> match, the one-shot equalise. `link_layers` connects EVENTS and
        #                       does NOT equalise at link time, so a fresh operator layer sits on
        #                       its own auto window until somebody writes one. This writes them,
        #                       without moving raw's window to do it.
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
        # MATCH. It sits HERE, in the row the operator that made those layers was run from, and
        # not in a menu: an operator result is seeded from its OWN pixels so it arrives legible on
        # its own terms, which means raw and the result are on two different stretches and the
        # before->after flip compares two stretches rather than two images. This is the one click
        # that turns it into a real comparison, and it is next to the operator Run button because
        # that is where the user is standing when they want it. Renamed from "Match raw contrast",
        # which named the SOURCE and not the scope and so read as a third clipboard verb.
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

        # DEFAULTS: which global defaults THIS window has overridden, and the two ways out of it.
        # A global default that silently disagrees with what is on screen is worse than no default
        # at all, so divergence is stated IN THE WINDOW and not only in the model. Same principle
        # as the compact placement mode labelling itself: never let the user not know which state
        # they are in. "make default" is the only outward push, and it is a deliberate click.
        def_box, dv = self._titled_box("Defaults")
        d1 = QHBoxLayout(); d1.setSpacing(4)
        self._focus_default_chk = QCheckBox("auto focus")
        # ONCE, not per region: _apply_settings_once returns early after the first mosaic, so the
        # jump happens when this window first paints and never again. The tooltip said "whenever
        # this window loads a region", which promised a per-region refocus the code does not do.
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

    # -- global default vs per-window override, made visible -----------------------------
    _AT_DEFAULTS_QSS = "color:#8b949e;font-size:10px;border:none;"
    _DIVERGED_QSS = "color:#e3b341;font-size:10px;font-weight:700;border:none;"
    #: The operator-run bar. Amber chunk to match the "working" colour the rest of the app uses for
    #: an in-flight run, so the bar and the dot the user already reads are the same state.
    _PROGRESS_QSS = (
        "QProgressBar{background:#161b22;border:1px solid #30363d;border-radius:3px;"
        "color:#c9d1d9;font-size:10px;text-align:center;}"
        "QProgressBar::chunk{background:#e3b341;border-radius:3px;}"
    )

    #: What each setting is CALLED in the window, so a marker reads as English rather than as a
    #: field name. The keys are the field names, which stay the vocabulary everywhere else.
    _SETTING_LABELS = {
        "tenengrad_focus": "auto focus",
        "channel_visibility": "channels",
        "luts": "contrast",
    }

    def _refresh_divergence(self) -> None:
        """Say which settings this window has overridden, and enable reset only when there is
        something to reset."""
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
        # _say, not _echo: this is a settings change with no structured console line behind it, and
        # its NEIGHBOURS in the same box ("make default", "reset") both _say. A quiet control next
        # to a loud one reads as a control that did nothing, which is why it was clicked four times.
        self._say(f"auto focus {'on' if on else 'off'} for this window.")

    def _sync_settings_widgets(self) -> None:
        """Put the controls back in step with the settings after a programmatic change.

        ``blockSignals`` because ``setChecked`` emits ``toggled``, which would write the value we
        just read straight back through :meth:`ViewSettings.set` and re-diverge the window we are
        in the middle of resetting.
        """
        chk = getattr(self, "_focus_default_chk", None)
        if chk is None:
            return
        chk.blockSignals(True)
        try:
            chk.setChecked(bool(self.settings.get("tenengrad_focus")))
        finally:
            chk.blockSignals(False)

    def _reset_settings(self) -> None:
        """Back to what this window opened with: the global defaults, or, for contrast, its
        parent's. NOT today's default, which the window may never have seen."""
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
        """Bring the plate window back to the front. See ``ViewerManager.raise_plate``.

        Says so out loud when it cannot, rather than making a button that silently does nothing --
        a window with no manager is the standalone case, and a dead button that looks alive is the
        failure this project keeps naming.
        """
        if self._manager is None or not self._manager.raise_plate():
            self._say("there is no plate window to raise from here.")

    def _window_operators(self) -> list:
        """THE OPERATORS FOR THIS WINDOW: every processing layer it holds, raw excluded.

        `mosaic.ops()` is the whole answer and it is derived, not tracked -- it walks the layers
        this pane owns and reads each one's declared `(op, channel)` identity. So a window that has
        had decon and bgsub run on it lists both, in the order they arrived, and a window that has
        had nothing run lists none. No bookkeeping to fall out of step with what is on screen.
        """
        mosaic = getattr(self._pane, "mosaic", None) if self._pane is not None else None
        if mosaic is None:
            return []
        try:
            return [op for op in mosaic.ops() if op and op != _RAW_OP]
        except Exception:                                # noqa: BLE001 - treat as "none yet"
            return []

    def _show_operator_controls(self) -> None:
        """Raise the plate AND open the controls for the operators THIS WINDOW has.

        Julio, 2026-08-05, correcting the first cut of this: *"the controls is actually for the
        'operators for this window'. And it is not bringing up the plateview window."*

        Both corrections are the same mistake. The first version asked `visible_op()` -- ONE
        operator, the one whose layer is lit -- and, worse, made **raising the plate conditional on
        finding it**. So on a window showing raw, or with the operator layer toggled off, the chip
        did nothing at all: no tab, and not even the trip back to the plate that `▣ plate` gives
        you unconditionally. A gate that swallows the half that always works is worse than no gate.

        Now:

        * **The plate comes forward FIRST, always.** It is the same journey `▣ plate` makes and it
          cannot depend on what is loaded. Whatever else this chip does or declines to do, the
          window you are going back to is in front of you.
        * **Then every operator this window holds gets its tab** (`_window_operators`, which reads
          the layers' own declared identity). Plural, because that is what the window has: a window
          with decon and bgsub run on it opens both, last one focused. `_activate_operator` chooses
          a hand-written panel or one generated from the declared `params`, so an iteration count,
          a threshold, or a plugin operator's parameters all arrive with no edit here and no name
          comparison (`tests/test_operator_declaration.py` fails the build on one).
        * **A window with no operator results still raises**, and says what is missing rather than
          looking broken.
        """
        raised = self._manager is not None and self._manager.raise_plate()
        if not raised:
            self._say("controls: there is no plate window to open operator controls in.")
            return

        ops = self._window_operators()
        if not ops:
            self._say("controls: nothing has been run on this window yet, so it has no operator "
                      "controls. The plate is in front; run an operator from there.")
            return

        plate = self._manager.parent()
        activate = getattr(plate, "_activate_operator", None)
        if activate is None:
            self._say(f"controls: this plate cannot open operator tabs, so {', '.join(ops)} "
                      "cannot be tuned from here.")
            return

        opened, failed = [], []
        for op in ops:
            try:
                activate(op)
                opened.append(op)
            except Exception as exc:                     # noqa: BLE001 - named, never a dead click
                failed.append(f"{op} ({exc})")
        if opened:
            self._say(f"controls: {', '.join(opened)} — open on the plate window."
                      + (f" Could not open {'; '.join(failed)}." if failed else ""))
        else:
            self._say(f"controls: could not open {'; '.join(failed)}.")

    # -- movie export: this view's T (or Z) sweep, as a file ------------------------------
    def _refresh_record_chip(self) -> None:
        """Enable the record chip only when there is a movie to make, and SAY WHY when there is not.

        Two separate refusals, kept separate because they have different fixes: an acquisition with
        one timepoint and one z plane has no axis to sweep, and a machine with no ffmpeg cannot
        encode anything. A single greyed-out button with one tooltip would collapse them.
        """
        from squidmip._video import axis_length, can_record, default_axis, encoder_problem

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
        """The channels the user can actually SEE, in acquisition order.

        A movie of a hidden channel is a movie of something not on screen. Falls back to every
        channel when the pane has no mosaic to ask (a window that has not painted yet), because
        "all of them" is the state napari starts in.
        """
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
        """Export this view's sweep to an .mp4. Second click cancels the run in flight.

        THE UI THREAD DOES A DIALOG AND NOTHING ELSE. No plane is read and no frame is encoded
        here: both go to :class:`_VideoWorker`. Measured, this handler costs well under a
        millisecond outside the modal dialog, against ~4 s of work on the real 10x region.
        """
        from squidmip._video import DEFAULT_FPS, axis_length, can_record, default_axis

        worker = self._video_worker
        if worker is not None and worker.isRunning():
            worker.stop()
            self._say("cancelling the movie export…")
            return
        if self._reader is None or self._meta is None:
            self._say("open a region first, then export a movie.")
            return
        if not can_record(self._meta):
            self._say("this acquisition has a single timepoint and a single z plane, so there is "
                      "no axis to sweep into a movie.")
            return
        region = self.current_region()
        axis = default_axis(self._meta)
        channels = self._visible_channels()
        # The CONTRAST ON SCREEN, latched for the whole movie. Read off the layers rather than
        # recomputed, for the same reason `current_settings` does: the user drags contrast in
        # napari's own controls and the layers are the only thing that knows where it ended up.
        luts = self._per_channel_luts()
        windows = [luts[ch]["clim"] for ch in channels
                   if luts.get(ch, {}).get("clim") is not None]
        if len(windows) != len(channels):
            windows = None          # partial is not a window set; let the recorder derive one
        rgb = {ch: luts[ch]["rgb"] for ch in channels if luts.get(ch, {}).get("rgb") is not None}

        default_name = f"{region}_{axis}.mp4"
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save the {axis.upper()}-axis movie of {region}", default_name, "Movie (*.mp4)")
        if not path:
            return
        if not str(path).lower().endswith(".mp4"):
            path = f"{path}.mp4"

        from squidmip._viewer import _VideoWorker

        n = axis_length(self._meta, axis)
        w = _VideoWorker(self._reader, self._meta, region, path, axis=axis, fps=DEFAULT_FPS,
                         channels=channels, windows=windows, rgb_by_channel=rgb,
                         z=self._z_slider_index(), t=self.time_point, parent=self)
        w.progress.connect(lambda d, total: self._show_progress(
            int(100 * d / max(1, total)), f"movie: frame {d} of {total}"))
        w.done.connect(self._on_movie_done)
        w.problem.connect(self._on_movie_failed)
        w.cancelled.connect(self._on_movie_cancelled)
        w.finished.connect(lambda: self._forget_video_worker(w))
        self._video_worker = w
        self._show_progress(0, f"movie: 0 of {n} frames")
        self._say(f"exporting {n} {axis}-axis frames of {region} to {path}…")
        w.start()

    def _z_slider_index(self) -> int:
        """Which z plane this window is showing, or 0 when it has no z slider.

        Only the T path uses it (a time-lapse is recorded AT a focus); the Z path sweeps every
        plane and ignores it.
        """
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
            pass            # already gone

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
        """Make THIS window's settings the default for windows opened FROM NOW ON.

        Deliberately leaves every open window alone, this one included except that it stops
        reporting diverged (because it no longer is). Propagating a change across open windows was
        ruled out: it fights the independence the decentralization bought, and the user asked for a
        default, not a broadcast.
        """
        if self._manager is None:
            self._say("this window has no manager, so it cannot set the global default.")
            return
        if not self._manager.make_default(self.window_id):
            self._say("this window is not in the registry, so it cannot set the global default.")
            return
        self._say("these are now the defaults for windows opened from now on; windows already "
                  "open are unchanged.")

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
        # SAVE OFF by default = preview (see how it looks); ON persists an OME-Zarr (Spencer huddle).
        save = bool(self._save_chk.isChecked()) if getattr(self, "_save_chk", None) is not None else False
        try:
            # ``requester=self`` IS the completion path. Without it run_operator started a QThread
            # and returned, so this window called in and never heard back -- which is why the
            # result rendered on the plate and 'raw' / 'flatfield' / 'stitched' were not
            # selectable in the window that asked for them (Julio, 2026-07-29). The plate calls
            # operator_started / operator_done / operator_failed and deliver_result on us.
            self._run_operator(key, regions=regions, save=save, requester=self)
            mode = "saving" if save else "previewing"
            self._echo(f"{mode} {self._op_combo.currentText()} on {self._region_label(regions)}.")
        except Exception as exc:                          # noqa: BLE001 - named to the window
            self._say(f"could not start {self._op_combo.currentText()}: {exc}")

    # -- the completion path: what the plate calls back on THIS window --------------------
    #
    # Julio, 2026-07-29, the longest-standing report in the backlog: "the layers such as 'raw',
    # 'flatfield', 'stitched', in the window that I decided to compute, are simply not available
    # when I run an operator on the window. Also, even if we have a cache of operations, when it
    # propagates to other windows, it adds a layer, but it doesn't toggle it."
    #
    # THE ROOT CAUSE was that nothing came back. ``PlateWindow.run_operator`` started a QThread and
    # returned; ``_run_view_operator`` called into it and never heard another word, so the result
    # could only render where the run was orchestrated -- the plate -- and the window that asked
    # showed nothing new. A window cannot render a result it is never told about, so these four
    # methods are the callback, and everything else follows from them.
    #
    # THE LAYER STACK IS NOT NEW, AND THAT IS THE POINT. ``MosaicLayers`` (op, channel) already IS
    # a per-window stack -- ``raw`` plus one group per operator -- and ``squidmip._layer_tree``
    # already mounts the grouped, checkbox-toggleable tree in EVERY ``MosaicPane``, so every window
    # has had the presentation since it was written. What was missing on this side of the app was a
    # PRODUCER, exactly as ``squidmip._op_result`` says it was missing on the plate's side.

    def operator_started(self, action: str) -> None:
        """A run THIS window asked for has begun. Opens the console's started/done pair.

        The address is captured HERE and carried into whichever line closes the pair, never read
        again when that line fires. The user is free to move this window's region slider while the
        operator runs, and a "done" line naming where the window is NOW rather than what was
        actually worked on is a lie the log would tell confidently. Same rule, and the same reason,
        as :meth:`_detect_nuclei`.
        """
        self._op_action = str(action)
        self._op_address = self.address()
        self.log.started(self._op_action, address=self._op_address)
        # The bar comes up INDETERMINATE and immediately, before the worker has said anything.
        # There is a real gap between the click and the first report (the reader's metadata warm,
        # the pool priming, the disk guard), and a window that shows nothing across it is the state
        # being complained about. It goes determinate on the first report, which arrives at 0 of N.
        self._show_progress(None)

    def operator_progress(self, report) -> None:
        """The run advanced. ``report`` is a :class:`~squidmip._progress.ProgressReport`.

        Determinate when the report has a total, INDETERMINATE when it does not — never a
        fabricated percentage. The report's own ``sentence()`` is the text, so the region window
        and the plate's status line say the same thing about the same run rather than two
        hand-built strings that can drift.
        """
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
        """``[3] A1  mip  done in 1.4 s`` -- the half of the pair that was never emitted."""
        self.log.done(str(action), float(seconds), address=self._closing_address())
        self._echo(f"{action} finished in {float(seconds):.1f} s.")
        self._op_action = self._op_address = None
        self._hide_progress()

    def operator_failed(self, action: str, reason: str) -> None:
        """The third outcome, and it must exist: an action that starts and then says nothing is
        indistinguishable from one still running."""
        self.log.failed(str(action), str(reason), address=self._closing_address())
        self._echo(f"{action} failed: {reason}")
        self._op_action = self._op_address = None
        self._hide_progress()

    def _show_progress(self, percent, text: str = "working…") -> None:
        """Put the bar up. ``percent=None`` = INDETERMINATE (Qt's own busy sweep, range 0..0).

        Qt draws an animated sweep for a 0..0 range and no percentage, which is exactly the right
        picture for "running, denominator unknown" — see ``squidmip._activity``: a progress bar
        that invents a denominator is a lie that gets believed.
        """
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
        """Take the bar down. Called from BOTH terminal callbacks, and the plate calls exactly one
        of them on success, failure and a stopped run alike — so the bar cannot be left running
        over a run that is over."""
        bar = getattr(self, "_op_progress", None)
        if bar is None:
            return
        bar.setRange(0, 100)                     # leave no indeterminate sweep behind
        bar.setValue(0)
        bar.hide()

    def _closing_address(self):
        """The address the OPEN half of the pair was written with, so the two lines agree.

        Falls back to this window's address only when no pair was opened here (a plate-wide run
        the user never started from this window), because a line with no address at all is the one
        thing the global console cannot afford.
        """
        return self._op_address if self._op_address is not None else self.address()

    def deliver_result(self, op: str, result, *, visible: bool) -> int:
        """Add one operator's :class:`~squidmip._result.Result` to THIS window's layer stack.

        Returns the number of layers added, so a delivery that reached nothing is reported by the
        caller rather than assumed to have landed.

        ``visible`` is the answer to Julio's second sentence. A result reaching a window that did
        NOT ask for it arrives with ``visible=False``: the window gains the layer, so it is there
        to toggle, and what the user is looking at does not change under them. Adding a visible
        layer to a window somebody is not watching is a change they did not ask for and cannot see
        happening, and it is worse than not adding it at all, because they cannot even tell it
        happened. The window that ASKED gets ``visible=True``, because asking is the consent.

        WHAT IS RENDERED IS WHAT THE RESULT DECLARES. The channel set comes from
        ``result.channels`` and the z scale is applied only when ``result.z_depth`` says there is a
        z axis to scale. Nothing here infers a channel list from this window's metadata or a depth
        from an array's ``ndim``: that is the whole point of a self-describing result, and this is
        its first consumer that renders one.

        PLACEMENT COMES FROM THE PIXELS, and only falls back to ``mosaic_bbox_um``. A per-FOV
        operator's planes are fused by ``_op_result._fuse``, which IS the raw preview's placement
        code, so the preview's footprint is exactly right for them and flipping between raw and the
        result is a comparison rather than two differently-framed pictures. A REGION operator's
        planes are not: ``stitch_region`` fuses onto its own canvas and hands back a
        :class:`~squidmip._placement.PlacedArray` carrying the :class:`~squidmip._placement.Placement`
        that produced them, so the plane is placed by that.

        The two disagree, measured. The preview rounds every tile origin to a whole pixel and the
        stitch keeps it fractional: on the real 10x set ``manual0`` is 11462 x 9587 px as a preview
        and 11463 x 9587 as a stitch, so the stitched layer used to be squeezed into a box one row
        shorter than its own pixels -- Julio: "the stitched view is not exactly the same as that of
        raw". And ``stitch_plate`` accepts a FOV subset of a region, whose mosaic spans only those
        fields while ``mosaic_bbox_um`` always spans the whole well: on the synthetic 1536 plate,
        A1 fovs [0, 1] fuse to 2084 x 3157 px and were stretched 1.515x vertically across all four
        fields' worth of stage.

        ``Extent.bbox_um`` is still not the carrier -- it means "the ROI a request was narrowed to",
        and a second meaning for it is how the address model starts to drift. The geometry rides on
        the ARRAY, which is what ``PlacedArray`` exists for.
        """
        pane = self._pane
        mosaic = getattr(pane, "mosaic", None) if (pane is not None
                                                   and getattr(pane, "ok", False)) else None
        if mosaic is None:
            return 0
        region = str(result.region_id)
        if region != self.current_region():
            # A layer must not claim to describe a region this window is not showing: it would sit
            # at that region's stage coordinates on top of a different region's raw, and the user
            # would read the offset as something the operator did. The same rule the raw path
            # follows in _on_plane, for the same reason.
            log.debug("[%s] %s result for %s not shown here: this window is on %s",
                      self.window_id, op, region, self.current_region())
            return 0
        from squidmip._mosaic_source import mosaic_bbox_um
        from squidmip._napari_pane import _colormap_for

        preview_bbox = mosaic_bbox_um(self._meta, region)
        dz = (self._meta or {}).get("dz_um")
        added = 0
        for channel in result.channels:
            plane = result.plane(channel)
            placement = getattr(plane, "placement", None)
            bbox = region_bbox = (placement.bbox_um if placement is not None else preview_bbox)
            # AN ROI CHILD CROPS, through the same helper that crops its raw pyramid, so the
            # operator layer covers exactly the boxed tissue the raw layer under it covers.
            if self._roi_bbox is not None and region_bbox is not None:
                cropped = _crop_levels_to_bbox([plane], region_bbox, self._roi_bbox)
                if cropped is None:
                    continue                     # the box does not overlap: nothing of it to draw
                levels, bbox = cropped
                plane = levels[0]
            try:
                # `add_result`, not `add_mosaic`: the RESULT's own declaration picks the layer
                # type. `Result.kind` carries the operator registry's `produces`, so a segmentation
                # arrives as a napari Labels layer here and in the plate pane by the same rule,
                # rather than as an Image auto-windowed as if its label ids were photons.
                mosaic.add_result(
                    result.kind, str(op), channel, plane,
                    colormap=_colormap_for(channel),
                    bbox_um=bbox,
                    # Only a result that DECLARES depth gets a z scale. _place ignores it for a
                    # 2-D layer anyway; reading it off the declaration rather than off the array
                    # is what keeps the renderer honest when a z-preserving operator lands.
                    z_scale_um=(dz if int(result.z_depth) > 1 else None),
                    visible=bool(visible),
                )
            except Exception as exc:             # noqa: BLE001 - named, never a missing layer
                self._say(f"{op}: the {channel} layer could not be added: {exc}")
                continue
            added += 1
        if added:
            self._result_region = region
        return added

    def _drop_result_layers(self, why: str) -> None:
        """Drop every operator layer in this window, keeping ``raw``.

        Called on a region change. An operator layer describes ONE region; left standing while the
        window moves to another one it sits at the old region's stage coordinates over the new
        region's raw, which renders as a plausible picture and is the failure mode this codebase
        refuses everywhere else. Said out loud, because a layer disappearing without explanation is
        its own small mystery.
        """
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

    # -- nuclei detection (Cellpose) on THIS view -------------------------------------
    def _spot_source(self):
        """The (channel, raw layer) to detect nuclei on: the active raw channel if one is selected,
        else the first raw channel present. Returns (None, None) if there is nothing to segment."""
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
        for c in (self._meta or {}).get("channels", []):        # else the first raw channel
            name = c["name"]
            layer = mosaic.find(_RAW_OP, name)
            if layer is not None:
                return name, layer
        return None, None

    def _detect_nuclei(self):
        """Detect nuclei (Cellpose) on THIS view's MIP, off the GUI thread, and lay the mask over the
        mosaic. Reuses the app's one _SpotWorker + segmenter table -- no reimplementation, and it
        respects the MIP change (_full_res_mip). On an ROI child the layer data is the ROI crop, so
        detection runs on exactly the boxed tissue -- which is what "detect nuclei on my ROI" needs."""
        if self._spot_worker is not None and self._spot_worker.isRunning():
            self._say("nuclei detection is already running in this window.")
            return
        # Honour the pane's "Detect on:" channel picker; fall back to the active/first raw channel.
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
            self._say("open a region first, then detect nuclei.")
            return
        try:
            from squidmip._viewer import _SpotWorker
            from squidmip._spots import SpotParams
        except Exception as exc:                          # noqa: BLE001
            self._say(f"nuclei detection unavailable: {exc}")
            return
        region = self._cursor.region if self._cursor is not None else (
            self._regions[0] if self._regions else "")
        # THE STARTED / DONE PAIR, in the shape Task 1 specifies:
        #     [3] A1  nuclei(cellpose, DAPI)  started
        #     [3] A1  nuclei(cellpose, DAPI): 412 nuclei  done in 1.4 s
        # The address is captured HERE and carried into the callbacks, NOT read again when they
        # fire. The user is free to move this window's region slider while Cellpose runs, and a
        # "done" line naming where the window is NOW rather than what was actually worked on is a
        # lie the log would tell confidently. An address is only worth having if it is the address
        # of the work.
        action = f"nuclei(cellpose, {channel})"
        where = self.address()
        began = time.monotonic()
        w = _SpotWorker(region, channel, layer.data, None, None, SpotParams(), parent=self)
        w.ready.connect(self._on_nuclei_ready)
        w.problem.connect(lambda m, a=action, d=where: self.log.failed(a, str(m), address=d))
        w.problem.connect(self._echo)
        w.finished_count.connect(
            lambda r, c, n, a=action, d=where, t0=began: (
                self.log.done(f"{a}: {n} nuclei", time.monotonic() - t0, address=d),
                self._echo(f"{n} nuclei detected on {c}."),
            ))
        self._spot_worker = w
        self.log.started(action, address=where)
        self._echo(f"detecting nuclei (Cellpose) on the {channel} MIP — first run downloads weights…")
        w.start()

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
            for lyr in list(v.layers):                    # replace a prior mask for this channel
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

    def _focus_reference_plane(self) -> None:
        """Jump this window's z-slider to the sharpest plane (Tenengrad) of the current region's
        centre FOV. Reuses the app's _FocusWorker; the result moves napari's own z dims."""
        v = self._napari_viewer()
        region = self._cursor.region if self._cursor is not None else (
            self._regions[0] if self._regions else None)
        if v is None or region is None or self._reader is None or self._meta is None:
            self._say("open a region first, then focus the reference plane.")
            return
        # A stack of one plane has no reference plane to find, and ranking it would report a
        # "sharpest plane" for the only plane there is. A refusal has to be a SENTENCE: this guard
        # came off `PlateWindow` with the orphan button (2026-07-29) rather than being dropped with
        # the rest of that dead chain, because it is the message the user needs on a 2D acquisition.
        z_levels = list((self._meta.get("z_levels") or []))
        if len(z_levels) <= 1:
            self._say(f"{region}: this acquisition has a single z plane, so there is no "
                      "reference plane to find.")
            return
        if self._focus_worker is not None and self._focus_worker.isRunning():
            self._say("already finding the reference plane…")
            return
        from squidmip._napari3d import _center_fov
        fov = _center_fov(self._meta, region)
        if fov is None:
            fovs = (self._meta.get("fovs_per_region") or {}).get(region) or [0]
            fov = int(fovs[0])
        chan = self._meta["channels"][0]["name"]
        from squidmip._viewer import _FocusWorker

        w = _FocusWorker(self._reader, self._meta, region, int(fov), chan, parent=self)
        w.ready.connect(lambda z_i, note: self._on_reference_plane(int(z_i), note))
        if hasattr(w, "problem"):
            w.problem.connect(self._say)
        self._focus_worker = w
        self._say("finding the sharpest z (Tenengrad autofocus)…")
        w.start()

    def _on_reference_plane(self, z_index: int, note: str) -> None:
        """The sharpest plane is known. MOVE THIS WINDOW'S z SLIDER to it, or say why not.

        Announcing a reference plane over a slider that never moved is the silent failure the
        control exists to avoid, so the axis is checked before it is written rather than after.
        ``if step:`` was not that check: on a 2D ``(y, x)`` layer it is true, and ``step[0] = z``
        then drove Y and reported a reference plane anyway. This guard came off ``PlateWindow`` with
        the orphan focus button (2026-07-29) instead of being deleted along with it.
        """
        v = self._napari_viewer()
        if v is None:
            return
        try:
            dims = v.dims
            nsteps = tuple(int(n) for n in (getattr(dims, "nsteps", ()) or ()))
            # z is the LEADING axis of a (z, y, x) layer. Two ways there is no z slider to move:
            # fewer than three axes (a plain 2D image), or a leading axis with a single step.
            if len(nsteps) < 3 or nsteps[0] < 2:
                self._say(f"sharpest plane is z={z_index}, but no z slider could be moved — "
                          "this view is showing a single plane.")
                return
            step = list(dims.current_step)
            step[0] = max(0, min(int(z_index), nsteps[0] - 1))   # clamp: never index past the stack
            dims.current_step = tuple(step)
            self._say(f"reference plane: z={z_index}. {note}".strip())
        except Exception as exc:                         # noqa: BLE001 - named, never silent
            self._say(f"could not move the z-slider: {exc}")

    def _view_roi_2d(self) -> None:
        """2D view of the SELECTED ROI: open it as a child window (same annotation the 3D button
        renders in 3D). With no ROI picked, just show the mosaic in 2D."""
        bbox, _region = self._selected_roi()
        if bbox is None:
            self._set_ndisplay(2)
            return
        self._open_roi_children()

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
            # Per-ROI COLOURS + a hovering NAME label (Julio: "roi boxes should have different
            # colors" + "an roi name hovering over the bounding box of each annotation", QuPath
            # style). Each shape gets a name property (R1, R2, ...) that drives both the edge-colour
            # cycle and the text label. Wrapped so a napari text/property hiccup still yields a
            # usable ROI layer rather than breaking ROI drawing.
            try:
                layer = v.add_shapes(
                    name="ROIs", face_color="transparent",
                    properties={"name": np.array([], dtype=object)},
                    text={"string": "{name}", "color": "white", "size": 9,
                          "anchor": "upper_left"},
                    edge_color="name", edge_color_cycle=list(_ROI_COLORS),
                )
                layer.current_properties = {"name": np.array(["R1"], dtype=object)}
                layer.events.data.connect(
                    lambda e=None, ly=layer: self._on_roi_data(ly))
            except Exception:                            # noqa: BLE001 - fall back to a plain layer
                layer = v.add_shapes(name="ROIs", edge_color="#58a6ff",
                                     face_color="transparent")
            self._roi_layer = layer
            self._sync_roi_width(v, layer)
            try:                                         # border reacts to zoom from here on
                v.camera.events.zoom.connect(
                    lambda e=None, vv=v, ly=layer: self._sync_roi_width(vv, ly))
            except Exception:                            # noqa: BLE001
                pass
        return v, layer

    def _on_roi_data(self, layer) -> None:
        """After a shape is added/removed: name the NEXT ROI R{n+1}, and SAY WHAT THE LAST ONE COSTS.

        Julio's standing complaint is that you can draw an ROI and only discover afterwards that it
        will not render. The size is knowable the instant the box exists, so it is reported the
        instant the box exists -- how many pixels, and how many GL textures that needs on THIS GPU.
        That is the drawing-time feedback the refusal used to stand in for; the refusal itself is
        gone, because bricking renders the box either way.
        """
        try:
            n = len(getattr(layer, "data", []) or [])
            layer.current_properties = {"name": np.array([f"R{n + 1}"], dtype=object)}
        except Exception:                                # noqa: BLE001 - labelling is cosmetic
            pass
        try:
            self._clamp_last_roi(layer)
        except Exception:                                # noqa: BLE001 - never break ROI drawing
            pass
        try:
            self._say(self._roi_cost_line(layer))
        except Exception:                                # noqa: BLE001 - the readout is advisory
            pass

    def _live_texture_limit(self) -> int:
        """The GPU's real GL_MAX_3D_TEXTURE_SIZE, or the documented Apple floor. Never a literal
        here: ``_napari_pane`` owns the query and ``_napari_view`` owns the fallback."""
        try:
            return int(self._pane._live_max_3d_texture())
        except Exception:                                # noqa: BLE001
            return int(_DEFAULT_MAX_3D_TEXTURE)

    def _clamp_last_roi(self, layer) -> None:
        """Hold the just-drawn ROI to what one GL texture can render, in place.

        THE GUARANTEE: anything you can draw, you can render at full native resolution. Julio's
        standing complaint is "I can select ROIs that can't be seen" -- so the box is corrected the
        moment it exists, at the size the GPU can actually hold, instead of being accepted and then
        refused. The correction is anchored at the drag's starting corner so the rectangle stops
        growing rather than jumping somewhere else.
        """
        from squidmip import _bricks

        # RE-ENTRANCY. Writing `layer.data` re-emits the Shapes layer's own data event, which lands
        # back here -- measured as an immediate RecursionError the first time this ran against the
        # real window. The correction is one edit, so the guard is a plain flag rather than a
        # disconnect/reconnect dance that could leave the ROI layer unwired if anything raised.
        if getattr(self, "_clamping", False):
            return
        rects = list(getattr(layer, "data", []) or [])
        px = float((self._meta or {}).get("pixel_size_um") or 0.0)
        if not rects or px <= 0:
            return
        arr = np.asarray(rects[-1])
        if arr.ndim != 2 or arr.shape[0] < 4:
            return                                       # not a rectangle; leave it alone
        ys, xs = arr[:, -2].astype(float), arr[:, -1].astype(float)
        limit = self._live_texture_limit()
        (nx0, ny0, nx1, ny1), clamped = _bricks.clamp_bbox_um(
            (xs.min(), ys.min(), xs.max(), ys.max()), px, limit)
        if not clamped:
            return
        new = np.array(arr, dtype=float)
        new[:, -1] = np.where(xs > xs.min(), nx1, nx0)
        new[:, -2] = np.where(ys > ys.min(), ny1, ny0)
        rects[-1] = new
        self._clamping = True
        try:
            layer.data = rects
        finally:
            self._clamping = False
        span = limit * px
        self._say(f"ROI held to the 3D ceiling: {limit} x {limit} px ({span:.0f} x {span:.0f} um) "
                  f"— the largest volume this GPU renders from one texture at full resolution.")

    def _roi_cost_line(self, layer) -> str:
        """"R3: 4096 x 3072 px (3080 x 2310 um) — 12 bricks on this GPU." Empty when unknowable."""
        from squidmip import _bricks

        rects = list(getattr(layer, "data", []) or [])
        if not rects:
            return ""
        arr = np.asarray(rects[-1])
        ys, xs = arr[:, -2], arr[:, -1]
        px = float((self._meta or {}).get("pixel_size_um") or 0.0)
        if px <= 0:
            return ""
        h_um, w_um = float(ys.max() - ys.min()), float(xs.max() - xs.min())
        h, w = int(round(h_um / px)), int(round(w_um / px))
        if h <= 0 or w <= 0:
            return ""
        limit = _DEFAULT_MAX_3D_TEXTURE
        try:
            limit = int(self._pane._live_max_3d_texture())
        except Exception:                                # noqa: BLE001 - the Apple value is the floor
            pass
        nz = len(list((self._meta or {}).get("z_levels") or [0]))
        single = _bricks.fits_single_texture(h, w, nz, limit)
        edge = limit if single else _bricks.DEFAULT_BRICK_EDGE
        n = len(_bricks.plan(h, w, limit=limit, edge=edge))
        how = ("fits ONE texture" if single
               else f"{n} bricks (over the {limit} px texture limit — bricked, not refused)")
        return f"R{len(rects)}: {h} x {w} px ({h_um:.0f} x {w_um:.0f} um), {nz} z — 3D: {how}."

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

    def _region_for_roi(self, bbox) -> Optional[str]:
        """Which of THIS window's regions the ROI box sits in (by its centroid, in stage um), so an
        ROI child opens on the ONE region it actually covers -- not all the parent's regions, which
        is why a box drawn on B7 'did not overlap' A7/A8 and fell back to the whole region."""
        cur = self._cursor.region if self._cursor is not None else (
            self._regions[0] if self._regions else None)
        if bbox is None:
            return cur
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        try:
            from squidmip._mosaic_source import mosaic_bbox_um
            for r in self._regions:
                rb = mosaic_bbox_um(self._meta, r)
                if rb is not None and rb[0] <= cx <= rb[2] and rb[1] <= cy <= rb[3]:
                    return r
        except Exception:                                # noqa: BLE001 - fall back to current region
            pass
        return cur

    def _open_roi_children(self) -> None:
        """Open the SELECTED ROI(s) as child window(s), each scoped to the single region it sits in.

        Julio: "I don't have a dropdown of which ROI to open — it opens them all. It should open the
        one that I'm currently selected." So we open ``layer.selected_data`` (the ROI(s) selected in
        napari's Shapes layer); with nothing selected we open the last one drawn, not the whole set."""
        v = self._napari_viewer()
        layer = self._roi_layer
        rects = list(getattr(layer, "data", []) or []) if layer is not None else []
        if v is None or layer is None or layer not in list(v.layers) or not rects:
            self._say("no ROI to open — draw one with '▭ new' first.")
            return
        if self._manager is None:
            self._say(f"{len(rects)} ROI(s) drawn, but this window has no manager to open children.")
            return
        # The SELECTED ROI(s); if none are selected, the most recently drawn one.
        sel = sorted(int(i) for i in (getattr(layer, "selected_data", None) or set()))
        idxs = sel if sel else [len(rects) - 1]
        opened = 0
        for i in idxs:
            if i < 0 or i >= len(rects):
                continue
            bbox = None
            try:
                arr = np.asarray(rects[i])
                ys, xs = arr[:, -2], arr[:, -1]        # world coords are (..., y, x)
                bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            except Exception:                            # noqa: BLE001 - a shapeless ROI still opens
                pass
            region = self._region_for_roi(bbox)
            if region is None:
                continue
            # The child inherits THIS window's contrast/colormap so it looks like its parent
            # (Julio: "the contrast for the ROIs is not the same as the parent window's"). The
            # manager derives that from parent_id now -- one place decides what a new window opens
            # with, rather than each opening site posting its own LUTs.
            child = self._manager.open_child(
                [region], roi_bbox=bbox, parent_id=self.window_id)
            if child is not None:
                opened += 1
        self._say(f"opened {opened} ROI child window(s) on the selected ROI"
                  + ("s" if opened != 1 else "") + ".")

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
            # ``cmap`` is a NAME, and a name is not a colour: "squid-Fluorescence_488_nm_Ex" tells
            # an exporter nothing. ``rgb`` is the same colormap reduced to the one 8-bit triple it
            # tints with, or None when it does not reduce to one (a multi-stop map). Recorded here
            # and not derived later because the napari layer is the only thing that has the
            # lookup table; by the time a LUT dict reaches _minerva the layer is gone.
            try:
                from squidmip._napari_view import colormap_hue_rgb
                lut["rgb"] = colormap_hue_rgb(layer)
            except Exception:                            # noqa: BLE001
                lut["rgb"] = None
            out[name] = lut
        return out

    def current_settings(self) -> "dict[str, Any]":
        """This window's global-default settings AS THEY ARE ON SCREEN right now.

        ``luts`` is read back off the napari layers rather than out of the record, because the user
        drags contrast in napari's own controls and the layers are the only thing that knows where
        it ended up. Everything else is exactly what the window recorded.
        """
        out = self.settings.snapshot()
        live = self._per_channel_luts()
        if live:
            out["luts"] = live
        return out

    def _apply_luts(self, luts: "Optional[dict]") -> Optional[int]:
        """Put per-channel contrast + colormap on this window's layers. ``None`` = no mosaic here.

        The one place LUTs land on layers: the settings baseline, a paste, and a reset all go
        through it, so there is a single answer to "what does applying a LUT do".
        """
        pane = self._pane
        mosaic = getattr(pane, "mosaic", None) if pane is not None else None
        if mosaic is None:
            return None
        applied = 0
        for ch, lut in (luts or {}).items():
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
        return applied

    def _apply_channel_visibility(self, visibility: "Optional[dict]") -> None:
        """Show/hide channels per the setting. An EMPTY setting means no opinion, so the channels
        stay exactly as napari made them rather than being forced on by a default."""
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
        """Put this window's settings on screen, ONCE, now that its layers exist.

        Once and not per region change: after the first application the window owns its own
        contrast, so re-seeding on the next region would throw away an adjustment the user made in
        between -- which is the whole reason the baseline is a starting point and not a rule.
        """
        if self._settings_applied:
            return
        pane = self._pane
        mosaic = getattr(pane, "mosaic", None) if pane is not None else None
        if mosaic is None:
            return
        self._settings_applied = True
        self._apply_luts(self.settings.get("luts"))
        self._apply_channel_visibility(self.settings.get("channel_visibility"))
        # Subscribe AFTER applying, so seeding the baseline is not mistaken for the user diverging.
        self._watch_user_visibility(mosaic)
        if self.settings.get("tenengrad_focus"):
            try:
                self._focus_reference_plane()
            except Exception as exc:                     # noqa: BLE001 - named, never silent
                self._say(f"could not autofocus this window: {exc}")

    def _watch_user_visibility(self, mosaic) -> None:
        """Record channel visibility the USER changed with napari's own eye icons, so the window
        reports diverged. Guarded: a mosaic without the seam simply never diverges on visibility,
        which is a smaller loss than a window that will not build."""
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
        caught = self._per_channel_luts()
        if not caught:
            self._say("no channels on screen to copy LUTs from.")
            return
        _LUT_CLIPBOARD.clear()
        _LUT_CLIPBOARD.update(caught)
        self._say(f"copied LUTs for {len(caught)} channel(s) — paste them into another window.")

    def _match_raw_contrast(self) -> None:
        """Put the RAW layer's contrast window on every operator layer of the same channel.

        The one action that makes the raw->operator flip a comparison. Operator layers are seeded
        from their own pixels on arrival (deliberately: a decon result has to be legible alone
        before you can judge its iteration count), and napari's per-channel link connects events
        without equalising values, so the two sides sit on two different stretches until somebody
        writes one. This writes them.

        Deliberately NOT recorded as a setting change: it moves operator layers only, never raw,
        so ``_per_channel_luts`` -- which reads the raw layers -- reports exactly what it did
        before, and the window has not diverged from its defaults. Unlike a paste, which does move
        raw and therefore is recorded.
        """
        pane = self._pane
        mosaic = getattr(pane, "mosaic", None) if pane is not None else None
        if mosaic is None:
            self._say("no mosaic here to match contrast on.")
            return
        matched = mosaic.match_contrast_to(_RAW_OP)
        if not matched:
            self._say("nothing to match — this window has no operator layers over the raw mosaic "
                      "yet. Run an operator on this view first.")
            return
        self._say(f"matched {matched} operator layer(s) to the raw contrast window.")

    def _paste_luts(self) -> None:
        if not _LUT_CLIPBOARD:
            self._say("no copied LUTs yet — use '⧉ Copy LUTs' in another window first.")
            return
        applied = self._apply_luts(_LUT_CLIPBOARD)
        if applied is None:
            self._say("no mosaic here to paste LUTs onto.")
            return
        # A paste IS the user changing contrast in this window, so it is RECORDED: the window now
        # differs from what it opened with, and the Defaults box has to say so.
        self.settings.set("luts", self._per_channel_luts())
        self._refresh_divergence()
        self._say(f"pasted LUTs onto {applied} channel(s).")

    # -- navigation ---------------------------------------------------------------------
    @property
    def time_point(self) -> int:
        """Which timepoint THIS window is showing."""
        bar = getattr(self, "_time_point_bar", None)
        return bar.time_point if bar is not None else 0

    def _on_time_point_changed(self, time_point: int) -> None:
        """A user moved THIS window's timepoint, or playback advanced it. Reload the mosaic.

        Only a user gesture arrives here; TimePointBar does not echo its own programmatic moves, so
        sizing the bar on open cannot trigger a load.

        A PLAYBACK step is loaded IMMEDIATELY and a drag is DEBOUNCED, which is one rule stated
        twice rather than two policies: never have more than one load in flight. Playback already
        guarantees that with the frame gate — napari does not request the next timepoint until
        `frame_done()` says this one is on screen — so a debounce there would only add its own
        interval to every frame and cap the achievable rate. A DRAG is not gated by anything: the
        scrollbar emits a step per pixel of travel, so without the debounce a 40-step drag would
        start 40 loads and cancel 39 of them. Same debounce, same value, as the region axis.
        """
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

    def _load_mosaic(self, region: Optional[str]) -> None:
        """Fuse one region's FOVs into this window's napari pane, one layer per channel."""
        pane = self._pane
        if pane is None or not getattr(pane, "ok", False):
            return
        if self._reader is None or self._meta is None or not region:
            return
        from squidmip._viewer import _MosaicWorker

        # SUPERSEDE THE PRIOR LOAD WITHOUT BLOCKING THE UI THREAD. This used to stop the prior
        # worker and then BLOCKING-JOIN it for up to 2 s, on the GUI thread. `stop()` only sets an
        # Event that `_MosaicWorker.run` polls BETWEEN channels, and one channel is a full
        # `fuse_region_pyramid` plus a contrast seed that materialises the coarsest level, so a
        # scrub froze the window for as long as the current channel took, up to that cap.
        # Under playback that is a freeze per frame.
        #
        # What the wait was really buying was "no stale pixels", and a GENERATION does that
        # properly: every result carries the load it belongs to, and anything from an earlier one
        # is dropped on arrival. So the old worker is asked to stop, kept referenced until Qt
        # reaps it (a QThread destroyed while running aborts the process), and simply ignored.
        self._load_gen = int(getattr(self, "_load_gen", 0)) + 1
        gen = self._load_gen
        prior = self._worker
        if prior is not None and prior.isRunning():
            prior.stop()
            self._retire_worker(prior)

        # An operator layer belongs to the region it was computed on. Moving to another region
        # would leave it placed at the old region's stage coordinates over the new region's raw.
        if self._result_region is not None and self._result_region != str(region):
            self._drop_result_layers(f"this window moved from {self._result_region} to {region}")
        # REMOVE THE RAW LAYERS FOR A REGION CHANGE; KEEP THEM FOR A TIMEPOINT CHANGE. That
        # distinction is the whole rule, and getting either half wrong has a measured cost.
        #
        # KEEPING them when the region is the SAME is what makes playback possible. `add_mosaic`
        # reuses a layer it can FIND (6465069, "reuse layers across region changes instead of
        # destroying them", Julio: "I can't cycle rapidly through these mosaics"), so an
        # unconditional removal here guaranteed the slow path on every reload — and since the
        # decentralization this window is the only viewing path, so that fix had no live caller
        # left. Measured on sim_5d_2x2_t3 with a real napari canvas: the read is 10-13 ms in the
        # worker, while rebuilding costs 165-265 ms of GUI thread PER CHANNEL. That was ~1.3 s a
        # frame and ~0.8 s of frozen window; reusing makes it ~210 ms and ~0.4 s.
        #
        # REMOVING them when the region CHANGES is not caution, it is a crash fix. A different
        # region is a different mosaic with a different shape, and `_reuse_layer` does not refuse
        # a shape it cannot survive — it assigns `layer.data` and napari raises downstream.
        # Driven against a real ViewerModel: deeper->shallower pyramid and 2D->3D both raise
        # IndexError, 3D->2D raises ValueError, and the half-assigned layer then aborts the
        # process on teardown. Ragged regions are ordinary: manual0 is 27 FOVs and manual1 is 28
        # on the 10x tissue set, so their mosaics differ. `tests/test_time_point_playback.py`
        # walks those transitions against a real `MosaicLayers`, because the pane STUB the rest
        # of the suite uses records `add_mosaic` and returns, so no stubbed test can ever see it.
        #
        # `_shown_region` is the one fact both this and the framing in `_on_done` ask about:
        # which region's mosaic is currently in the pane. A stale answer here can only cost an
        # unnecessary rebuild, never a crash, which is the right way round.
        if self._shown_region != str(region):
            pane.mosaic.remove_op(_RAW_OP)
        channels = [c["name"] for c in self._meta["channels"]]
        # t=THIS WINDOW'S TIMEPOINT. Without it the worker fused timepoint 0 whatever the
        # timepoint bar said, and the reload this method performs on every slider move repainted
        # the same pixels.
        w = _MosaicWorker(self._reader, self._meta, region, channels, parent=self,
                          t=self.time_point)
        w.ready.connect(lambda r, ch, levels, bbox, win:
                        self._on_plane(r, ch, levels, bbox, win, gen=gen))
        w.problem.connect(self._say)
        w.finished_count.connect(lambda n: self._on_done(region, n, gen=gen))
        # EVERY WORKER IS DELETED WHEN IT ENDS. `parent=self` makes Qt's C++ object graph own it,
        # so a finished worker stays alive for as long as the window does — measured: 78 live
        # `_MosaicWorker` objects after 78 playback frames, with `gc.collect()` freeing none of
        # them, which is exactly the accumulation `tools/run_suite_chunked.py` diagnosed as the
        # reason the suite cannot run in one process. Before playback a window created a worker
        # per navigation; now it creates one per FRAME, so an unbounded pile is no longer
        # something that merely offends.
        w.finished.connect(lambda: self._worker_ended(w))
        self._worker = w
        w.start()

    def _worker_ended(self, worker) -> None:
        """A load's thread has ended. Drop every reference to it, ours and Qt's."""
        if self._worker is worker:
            self._worker = None
        self._forget_worker(worker)
        try:
            worker.deleteLater()
        except RuntimeError:
            pass            # already gone

    def _retire_worker(self, worker) -> None:
        """Let a superseded worker die on its own time, without dropping it on the floor.

        Two failures are being avoided at once. Dropping the only reference to a RUNNING QThread
        lets Python free it and Qt aborts the process ("QThread: Destroyed while thread is still
        running") — the same hazard `RegionSlider.shutdown` exists for. Waiting for it instead
        blocks the GUI thread, which is what this replaces. So it is parked here and removed when
        Qt says it has finished.
        """
        retired = getattr(self, "_retired_workers", None)
        if retired is None:
            retired = self._retired_workers = []
        retired.append(worker)
        # No second `finished` connection: `_load_mosaic` already wired one to `_worker_ended`,
        # which forgets it here AND deletes it. Two hooks doing half the cleanup each is how one
        # of them ends up being the only one anybody remembers to update.

    def _forget_worker(self, worker) -> None:
        retired = getattr(self, "_retired_workers", None)
        if retired is not None and worker in retired:
            retired.remove(worker)

    def _is_current_load(self, gen: int) -> bool:
        """Whether *gen* is the load this window is still waiting for."""
        return int(gen) == int(getattr(self, "_load_gen", 0))

    def _on_plane(self, region: str, channel: str, levels, bbox_um, window=None,
                  gen: Optional[int] = None) -> None:
        pane = self._pane
        if pane is None or not getattr(pane, "ok", False):
            return
        # A SUPERSEDED LOAD'S PIXELS ARE DROPPED. The region check below cannot see this one: a
        # timepoint change keeps the region, so the loser and the winner agree about `region` and
        # differ only in `t`. Without the generation, the retired worker's timepoint 0 lands after
        # the new worker's timepoint 2 and the window shows the older frame under the newer label.
        if gen is not None and not self._is_current_load(gen):
            return
        if self._cursor is not None and self._cursor.region != region:
            return                                  # a later region won the race; drop this one
        from squidmip._napari_pane import _colormap_for

        # ROI CHILD: crop the lazy pyramid to the ROI box before adding, so napari materialises only
        # the ROI corner (read a corner, not the whole region). A window with no ROI box adds the
        # full region unchanged. The crop also adjusts bbox_um so placement lands on the ROI.
        add_levels, add_bbox = levels, bbox_um
        # The worker's window describes the WHOLE region, which is what this window is adding —
        # unless the crop below replaces the pixels, in which case it describes something else.
        add_window = window
        if self._roi_bbox is not None and bbox_um is not None:
            cropped = _crop_levels_to_bbox(levels, bbox_um, self._roi_bbox)
            if cropped is not None:
                add_levels, add_bbox = cropped
                # An ROI child is seeded from ITS OWN corner, exactly as before this seed moved
                # off the UI thread. Deriving it here still costs nothing measurable: the worker
                # has already decoded and cached the level the crop slices.
                add_window = None
            else:
                self._say("ROI does not overlap this region — showing the whole region.")

        pane.mosaic.add_mosaic(
            _RAW_OP, channel, add_levels,
            contrast_limits=add_window,
            colormap=_colormap_for(channel),
            multiscale=True,
            bbox_um=add_bbox,
            z_scale_um=(self._meta or {}).get("dz_um"),
        )
        # FIRST PAINT stops here, one line after the layer is actually in the pane, for the same
        # reason the operator metric stops in _on_tile rather than where the worker emits: what is
        # being measured is what the user saw, and a queue delay between those two is the thing
        # being looked for. The clock keeps the FIRST report and drops the rest, so the second
        # channel of this region -- and every later region this window navigates to -- needs no
        # "have I done this already" flag here.
        if self.open_clock is not None:
            self.open_clock.first_layer()

    def _on_done(self, region: str, n: int, gen: Optional[int] = None) -> None:
        pane = self._pane
        if pane is None or not getattr(pane, "ok", False):
            return
        # A retired worker finishing must NOT open the playback gate: the frame the user is
        # waiting for is still being read, and letting the next one be requested here is exactly
        # the backlog the gate exists to prevent.
        if gen is not None and not self._is_current_load(gen):
            return
        if n == 0:
            pane.say(f"{region}: no mosaic could be built (see the message above).")
            # NOW the raw layers go, and only now. `_load_mosaic` deliberately leaves them alone
            # so a reload can reuse them; the one case where that would lie is a load that
            # produced nothing, where the previous frame's pixels would sit under this region's
            # name. This is where that is known, so this is where they are dropped.
            try:
                pane.mosaic.remove_op(_RAW_OP)
            except Exception:                        # noqa: BLE001 - already gone is fine
                pass
            self._shown_region = None
            if self.open_clock is not None:
                self.open_clock.finish(_measure.FAILED, f"{region}: no mosaic could be built")
            self._frame_done()
            return
        pane.say("")
        # ASKED BEFORE THE try, RECORDED AFTER IT, and deliberately not inside: framing is
        # cosmetic and its failure is swallowed, but `_shown_region` also decides whether the NEXT
        # load may reuse these layers. Updating it inside the try would tie a correctness fact to
        # whether a camera move happened to succeed — and against a pane whose model is absent it
        # never does, so the flag would never advance and every reload would rebuild.
        first_look = self._shown_region != str(region)
        try:
            # show_op makes EXACTLY one group visible, so calling it unconditionally would hide an
            # operator layer this window is legitimately still showing for this same region -- the
            # user's toggle, undone by a reload they did not ask anything of.
            if self._result_region is None:
                pane.mosaic.show_op(_RAW_OP)
            # RE-FRAME ONLY WHEN THE PICTURE IS SOMEWHERE ELSE. A timepoint step reloads the SAME
            # region at the same stage coordinates, so resetting the camera each time does two
            # unwanted things: it costs 85-130 ms of GUI thread per frame (measured), and it drags
            # the user's zoom back to fit-the-region on every frame of playback -- you cannot
            # watch a blob move at 1:1 if the camera keeps pulling out. Framing follows the
            # REGION, which is the thing whose extent actually changed.
            if first_look:
                pane.mosaic.model.reset_view()
        except Exception:                            # noqa: BLE001 - view framing is cosmetic
            pass
        self._shown_region = str(region)
        # Seed this window's settings ONCE, now that the layers exist. For an ROI child that is the
        # parent's contrast, so the child looks like the window it was cut out of.
        self._apply_settings_once()
        # The open is OVER: every channel of the first region is in the pane. Idempotent, so the
        # second region this window loads does not record a second open -- what is being measured is
        # opening a window, not changing region inside one.
        if self.open_clock is not None:
            self.open_clock.finish()
        self._frame_done()

    def _frame_done(self) -> None:
        """Open the playback gate: this mosaic is on screen, the next frame may be requested.

        BOTH axes, unconditionally. One mosaic load is what a region step and a timepoint step
        both wait on, so the completion is one event and it opens whichever gate is closed. Asking
        which axis is playing first would be a second copy of "who is animating", and the bar and
        the slider each already know: `frame_done` on a control that is not playing is a no-op.
        """
        if self._slider is not None:
            self._slider.frame_done()
        bar = getattr(self, "_time_point_bar", None)
        if bar is not None:
            bar.frame_done()

    # -- 2D -> 3D, per window -----------------------------------------------------------
    def _selected_roi(self) -> "tuple":
        """(bbox, region) of the ROI currently SELECTED in this window's Shapes layer, else
        (None, None). Lets 2D/3D act on the picked ROI so one annotation serves both — Julio: "select
        the ROI and click 2d or 3d, so I don't have to do a 2d and a 3d annotation in the same place"."""
        layer = self._roi_layer
        v = self._napari_viewer()
        if layer is None or v is None or layer not in list(v.layers):
            return None, None
        rects = list(getattr(layer, "data", []) or [])
        sel = sorted(int(i) for i in (getattr(layer, "selected_data", None) or set()))
        if not sel or sel[0] >= len(rects):
            return None, None
        try:
            arr = np.asarray(rects[sel[0]])
            ys, xs = arr[:, -2], arr[:, -1]
            bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        except Exception:                                # noqa: BLE001
            return None, None
        return bbox, self._region_for_roi(bbox)

    def _roi_center_fov(self, region: str, bbox: Optional[tuple] = None) -> Optional[int]:
        """The FOV nearest the ROI box's centre (stage um), so an ROI's 3D lands on the tissue you
        boxed. ``bbox`` defaults to this window's own ROI box; None everywhere => region centre."""
        bbox = bbox if bbox is not None else self._roi_bbox
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
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

    def _replace_native3d(self, open_it) -> None:
        """ONE 3D popout per window: close the one this window already has, then open the new one.

        Julio: "consider changing the 3D interaction so clicking 3D reuses the current window
        instead of opening an extra one". Each 3D click used to construct a fresh ``napari.Viewer``
        (``_napari3d.py:268`` and ``:366``) and only the LATEST was remembered here, so five clicks
        left five top-level windows on screen -- and napari holds every Viewer in a global set
        (see ``_napari3d._wire_close_to_release_memory``), so dropping our reference never closed
        anything. This is the reason that pile grows.

        This is REPLACEMENT, not in-place reuse. The stronger reading -- click 3D and flip THIS
        window's embedded pane to ``ndisplay=3`` -- is not available, and deliberately so: the
        pane's layers are the FUSED PYRAMID, whose level 0 is already capped to ``_MAX_FUSED_PX``
        (``_mosaic_source.py:123``), while 3D reads a NATIVE z-stack straight from the reader. That
        is the "still downsampled" bug ``_open_3d``'s own docstring names below. The in-place
        toggle does exist as its own control -- napari's ndisplay button, lifted into the pane at
        ``_napari_pane.py:223`` and wired to ``render_max_res_3d`` -- and shows the coarse volume,
        which is exactly why the 3D chip is a separate action.

        Closing FIRST rather than after also matters on a big volume: two native stacks resident at
        once is the memory spike ``_wire_close_to_release_memory`` was written to avoid.

        Failure to close is swallowed by design: a stale popout that will not go away must not stop
        the new one from opening, and *open_it* raising is the caller's to report by name.
        """
        old, self._native3d = self._native3d, None
        if old is not None:
            close = getattr(old, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:                    # noqa: BLE001 - already-closed / no Qt window
                    pass
        self._native3d = open_it()

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
        # 3D acts on the SELECTED ROI when one is picked (parent window), else this window's own ROI
        # box (an ROI child), else the region centre. One annotation -> 2D or 3D on demand.
        roi_bbox = self._roi_bbox
        if roi_bbox is None:
            sel_bbox, sel_region = self._selected_roi()
            if sel_bbox is not None and sel_region is not None:
                roi_bbox, region = sel_bbox, sel_region
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
        # ROI -> native CROSS-FOV fusion cropped to the box (exact subarray, full z, native res),
        # read straight from the reader. Else the whole region's centre FOV (gallery-view recipe).
        if roi_bbox is not None:
            self._open_roi_3d(region, roi_bbox, contrast_by, colormap_by)
            return

        fov = self._roi_center_fov(region, roi_bbox)
        from squidmip._napari3d import open_native_3d

        try:
            self._replace_native3d(lambda: open_native_3d(
                self._reader, self._meta, region, fov=fov,
                contrast_by_channel=contrast_by or None,
                colormap_by_channel=colormap_by or None,
            ))
        except Exception as exc:                     # noqa: BLE001 - named to the window, never silent
            self._say(f"3D could not open: {exc}")

    def _open_roi_3d(self, region: str, roi_bbox: tuple, contrast_by: dict, colormap_by: dict) -> None:
        """3D of an ROI, BRICKED and IN THIS WINDOW. Any ROI renders; none is refused.

        Julio: "the 3D rendering ROI design improvement, which now sucks because the user has no
        in-window computation and can select ROIs that can't be seen (I thought we had decided to do
        bricking)". Both halves are answered here.

        IN-WINDOW. This paints into the pane's own napari canvas (``_napari_viewer()``) instead of
        constructing a popout ``napari.Viewer``. The old objection to doing that -- ``_replace_native3d``'s
        docstring, "the pane's layers are the FUSED PYRAMID whose level 0 is already capped to
        _MAX_FUSED_PX, while 3D reads a NATIVE z-stack" -- was about reusing the pane's LAYERS. It
        does not apply to adding our own: the brick layers are read straight from the reader, exactly
        as the popout was, and the pyramid layers are hidden while they are up. Sharing a canvas was
        never the problem; sharing the pyramid was.

        BRICKED. An ROI over the GL texture limit is tiled rather than turned away. Nothing is read
        that the camera cannot see, so a box over a whole 11538 x 9645 region opens against a bounded
        budget instead of a 2.2 GB/channel fusion.
        """
        names = [c["name"] for c in (self._meta or {}).get("channels", [])]
        if not names:
            self._say("this acquisition declares no channels to render in 3D.")
            return
        # THE LAYER MODEL, not the bare viewer. A brick is an app layer like any other, so it is
        # created, adopted and dropped through the same `MosaicLayers` a flat mosaic is -- that is
        # what makes one checkbox, one contrast window and one group mean the same thing in 3D as
        # in 2D. See `MosaicLayers.adopt`.
        mosaic = getattr(self._pane, "mosaic", None) if self._pane is not None else None
        if mosaic is None or getattr(mosaic, "model", None) is None:
            self._say("3D needs this window's napari canvas, which isn't available here.")
            return
        from squidmip import _bricks
        from squidmip._brick_view import BrickedVolume
        from squidmip._napari3d import region_origin_um, roi_window_px, z_step_um

        window = roi_window_px(self._meta or {}, region, roi_bbox)
        origin = region_origin_um(self._meta or {}, region)
        if window is None or origin is None:
            self._say("ROI 3D: this ROI does not land on any FOV of this region.")
            return
        nz = len(list((self._meta or {}).get("z_levels") or [0]))
        if nz < 2:
            self._say("3D needs a z-stack; this acquisition has a single z plane.")
            return
        px = float((self._meta or {}).get("pixel_size_um") or 1.0)
        dz = z_step_um(self._meta or {}, px, where=f"3D ROI {region}")
        max_tex = _DEFAULT_MAX_3D_TEXTURE
        try:
            max_tex = int(self._pane._live_max_3d_texture())
        except Exception:                                # noqa: BLE001 - the Apple value is the floor
            pass
        read, source = self._volume_source(window)
        if source is None:
            return                                       # _volume_source already said why
        r0, r1, c0, c1 = window
        # The ROI's own world corner: the region origin plus the crop, in stage micrometres. This is
        # the same arithmetic `_crop_levels_to_bbox` does for 2D, so the volume lands exactly where
        # the box was drawn.
        roi_origin = (0.0, float(origin[1]) + r0 * px, float(origin[0]) + c0 * px)
        budget = _brick_budget_bytes()
        try:
            self._replace_native3d(lambda: _started(BrickedVolume(
                mosaic, self._reader, self._meta, region, window,
                channels=names, scale=(dz, px, px), origin_um=roi_origin,
                limit=max_tex, budget_bytes=budget,
                # `source` is the operator `_volume_source` chose off the DECLARATION, never off a
                # name. Handing it down is what lets each brick declare its identity, which is what
                # puts the volume inside the layer tree at all -- see `BrickedVolume._op`. It was
                # already computed here and spent only on a sentence in the log.
                op=source,
                contrast_by=contrast_by or None, colormap_by=colormap_by or None,
                say=self._say, parent=self, read=read,
            )))
        except Exception as exc:                         # noqa: BLE001 - named to the window
            self._say(f"ROI 3D could not open: {exc}")
            return
        # THE CAMERA IS THE INPUT to which bricks are resident and how finely they are sampled, so
        # the settle callback is what makes zooming converge to native. Debounced by the pane (120
        # ms quiet period) for the reason the 2D fetch is: a drag emits camera events far faster
        # than a brick can be read, and fetching per event grows a queue that never drains.
        try:
            self._pane.on_camera_settled(self._refresh_bricks)
        except Exception:                                # noqa: BLE001 - static bricks still render
            pass
        vol = self._native3d
        n = getattr(vol, "brick_count", 0)
        self._say(f"3D in-window: '{source}', {(r1 - r0)}x{(c1 - c0)} px ROI, {nz} z, "
                  f"{len(names)} channel(s), {n} texture{'' if n == 1 else 's'} at "
                  f"{px:.3f} um/px. {_bricks.ceiling_line(max_tex, px, measured=True)}")

    def _volume_source(self, window: tuple):
        """WHICH volume 3D renders: the operator layer this window is SHOWING, or raw.

        Julio: "make sure that the 2d/3d operator workflow is also end to end". 3D used to read
        ``mosaic.find(_RAW_OP, name)`` with ``_RAW_OP`` hardcoded, so a decon / bgsub / stitch
        result could be computed and displayed in 2D and then had no way to be seen as a volume --
        the data existed (per-plane fusion makes operators produce ``(T, C, Nz, Y, X)``), only the
        viewer refused it.

        The choice comes off the DECLARATION, never off the operator's name: ``visible_op()`` says
        which processing layer is lit and ``MosaicLayers._reduces_z`` asks the registry whether that
        operator consumes z. ``tests/test_operator_declaration.py`` fails the build on an
        ``x == "<operator name>"`` comparison precisely to stop the name test creeping back.

        Returns ``(read, label)`` -- ``read=None`` meaning "use the reader's raw z-stack" -- or
        ``(None, None)`` with a spoken reason when the visible layer has no volume to show.
        """
        mosaic = getattr(self._pane, "mosaic", None) if self._pane is not None else None
        if mosaic is None:
            return None, _RAW_OP
        try:
            op = mosaic.visible_op()
        except Exception:                                # noqa: BLE001 - fall back to raw
            return None, _RAW_OP
        if not op or op == _RAW_OP:
            return None, _RAW_OP
        # A z-REDUCER's result is (T, C, 1, Y, X): one plane. There is no volume, and rendering a
        # single slice as a "3D volume" would be a picture that lies about its own depth.
        try:
            if mosaic._reduces_z(op):
                self._say(f"3D: '{op}' reduces z to a single plane, so it has no volume to render. "
                          f"Show raw (or a z-preserving operator) and click 3D again.")
                return None, None
        except Exception:                                # noqa: BLE001 - undeclared: try to render
            pass
        px = float((self._meta or {}).get("pixel_size_um") or 1.0)
        origin = None
        try:
            from squidmip._napari3d import region_origin_um

            origin = region_origin_um(self._meta or {}, self.current_region())
        except Exception:                                # noqa: BLE001
            pass
        if origin is None:
            self._say(f"3D: '{op}' cannot be placed — this region has no stage positions.")
            return None, None
        # An operator layer carries its OWN grid: a parent window holds the fused pyramid (level 0
        # capped to _MAX_FUSED_PX), an ROI child holds a crop placed at the ROI. Indexing either
        # with mosaic pixels would be wrong in a different way each time. The layer's own
        # translate/scale is the one mapping that is true for both, so world micrometres are the
        # currency -- exactly as they are for placement everywhere else in this pane.
        srcs: dict = {}
        for ch in mosaic.channels(op):
            layer = mosaic.find(op, ch)
            # level 0 of a pyramid: the finest rung. THE one pyramid rule and not an isinstance
            # check -- napari's `MultiScaleData` is neither a list nor a tuple, so that branch was
            # False for every real pyramid and the whole thing travelled on as the brick source,
            # where indexing it walks the LEVELS instead of the z planes.
            data = full_res_level(getattr(layer, "data", None) if layer is not None else None)
            if data is None or getattr(data, "ndim", 0) < 3 or int(data.shape[0]) < 2:
                continue
            try:
                tr = tuple(float(v) for v in layer.translate[-2:])
                sc = tuple(float(v) for v in layer.scale[-2:])
            except Exception:                            # noqa: BLE001 - unplaceable layer
                continue
            if sc[0] <= 0 or sc[1] <= 0:
                continue
            srcs[ch] = (data, tr, sc)
        if not srcs:
            self._say(f"3D: '{op}' is on screen but carries no z depth here, so there is no volume "
                      f"to render.")
            return None, None
        ox_um, oy_um = float(origin[0]), float(origin[1])

        def _read(brick, channel, step, should_stop):
            """One brick out of the OPERATOR's on-screen volume, same contract as the raw reader:
            a mosaic-pixel window in, ``(z, y, x)`` out, None when superseded."""
            got = srcs.get(channel)
            if got is None or (should_stop is not None and should_stop()):
                return None
            src, (ty, tx), (sy, sx) = got
            # brick bounds are mosaic pixels -> stage micrometres -> this layer's own indices
            y0 = int(round((oy_um + brick.r0 * px - ty) / sy))
            y1 = int(round((oy_um + brick.r1 * px - ty) / sy))
            x0 = int(round((ox_um + brick.c0 * px - tx) / sx))
            x1 = int(round((ox_um + brick.c1 * px - tx) / sx))
            y0, x0 = max(0, y0), max(0, x0)
            y1, x1 = min(int(src.shape[-2]), y1), min(int(src.shape[-1]), x1)
            if y1 <= y0 or x1 <= x0:
                return None
            sub = np.asarray(src[:, y0:y1, x0:x1])
            s = max(1, int(step))
            return np.ascontiguousarray(sub[:, ::s, ::s]) if s > 1 else sub

        return _read, op

    def _refresh_bricks(self) -> None:
        """The camera stopped: re-decide stride and visible set. No-op unless 3D bricks are up."""
        vol = self._native3d
        refresh = getattr(vol, "refresh", None)
        if not callable(refresh):
            return
        try:
            refresh()
        except Exception as exc:                         # noqa: BLE001 - named, never silent
            self._say(f"3D: could not follow the camera ({exc}).")

    def _render_roi_volume(self, mosaic, contrast_by: dict, colormap_by: dict) -> None:
        """Render the EXACT ROI subarray in 3D: the cropped level-0 volume this window's 2D view
        shows (mosaic.find(RAW).data[0]), texture-bounded, carrying the on-screen contrast. This is
        the ROI you boxed, at the same extent as 2D -- not a whole FOV."""
        volumes: dict = {}
        for c in (self._meta or {}).get("channels", []):
            name = c["name"]
            layer = mosaic.find(_RAW_OP, name)
            if layer is None:
                continue
            # THE one pyramid rule (`_napari_view.full_res_level`), not an isinstance check:
            # napari's `MultiScaleData` is neither a list nor a tuple, so this used to hand the
            # whole pyramid on unchanged — and `np.asarray` on it below yields the COARSEST
            # level, i.e. the blocky volume this docstring promises it is not rendering.
            level0 = full_res_level(layer.data)                              # the ROI-cropped rung
            if getattr(level0, "ndim", 0) < 3 or int(level0.shape[0]) < 2:
                self._say("3D needs a z-stack; this ROI has a single z plane.")
                return
            volumes[name] = level0
        if not volumes:
            self._say("no channel on screen to render in 3D.")
            return
        px = float((self._meta or {}).get("pixel_size_um") or 1.0)
        max_tex = 2048
        try:
            max_tex = int(self._pane._live_max_3d_texture())
        except Exception:                                # noqa: BLE001 - Apple default is the floor
            pass
        from squidmip._napari3d import open_native_3d_volume, z_step_um

        dz = z_step_um(self._meta or {}, px, where="3D ROI volume")

        try:
            self._replace_native3d(lambda: open_native_3d_volume(
                {n: np.asarray(v) for n, v in volumes.items()},
                scale=(dz, px, px),
                title=f"3D ROI — {self._region_label(self._regions)}",
                contrast_by_channel=contrast_by or None,
                colormap_by_channel=colormap_by or None,
                max_texture=max_tex,
            ))
        except Exception as exc:                         # noqa: BLE001 - named to the window
            self._say(f"ROI 3D could not open: {exc}")

    # -- where this window is, in the acquisition ----------------------------------------
    def current_region(self) -> str:
        """The region this window is SHOWING right now. A window can hold several and steps
        through them with its slider, so "which region" is a question about the cursor, not about
        the list."""
        region = self._cursor.region if self._cursor is not None else None
        if not region:
            region = self._regions[0] if self._regions else ""
        return str(region)

    def address(self):
        """WHERE this window is, as :class:`~squidmip._address.Address` or ``Extent``.

        An ROI child carries a box, and a box is a slab rather than a point, so it answers with an
        :class:`~squidmip._address.Extent`. Everything else answers with an ``Address`` naming the
        region and leaving every other dimension None, which means "all of it".

        The window's ORDINAL is deliberately not in here. It is the view id, it belongs to the
        desktop rather than to the plate, and it travels beside the address on every line.
        """
        region = self.current_region()
        if self._roi_bbox is not None:
            return Extent(region_id=region, bbox_um=self._roi_bbox)
        return Address(region_id=region)

    def view_log(self) -> ViewLog:
        """This window's logger, addressed to wherever it is pointing at this instant."""
        return self.log.at(self.address())

    def _say(self, text: str) -> None:
        # ALWAYS log to the shared logger (the one global console is a sink of the root logger),
        # carrying this view's id AND its address -- Julio: "the logger isn't responding to what we
        # do in the windows... I'm blind to it." The pane status bar is the in-window echo; the
        # console is the record of what every open view did to what, which is what "the logger
        # deals with all open windows" needs in order to be readable with six windows open.
        if text:
            self.view_log().info("%s", text)
        self._echo(text)

    def _echo(self, text: str) -> None:
        """The in-window status line ONLY. Use it when the console already has a structured line
        for this event (a started/done pair), so the console does not carry a prose duplicate of
        something it just said in a form the user can scan."""
        if self._pane is not None and getattr(self._pane, "ok", False):
            self._pane.say(text)

    # -- render-halt: a window not being manipulated must not keep drawing ----------------
    def set_active(self, active: bool) -> None:
        """Halt draw/refresh on windows the user is not touching (Spencer's memory brief).

        A window that is not the active one stops its playback so it is not fusing regions in the
        background and competing for the GPU with the window the user is actually looking at.
        """
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
        """Grow the type with the window, exactly as the root does.

        Spencer, 2026-07-27: the root rescales its type on resize but "the `[N] <well>` view
        windows and the Log window do not... dragging one bigger leaves the type behind."

        This window was never reachable from the root's rescale: it is a separate TOP-LEVEL window,
        so `PlateWindow.findChildren(QWidget)` never saw it. The shared helper lives in
        `_fontscale` for that reason, and each window keeps its OWN scale, which is what the
        decentralised design wants: two views dragged to different sizes should not be forced to
        agree about type size.
        """
        super().resizeEvent(e)
        rescale_fonts(self)

    # -- teardown -----------------------------------------------------------------------
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
            # An export polls its stop flag before each frame, and a frame on the real 10x region
            # is ~0.4 s, so 2 s is the same generous cap every other worker here gets. A QThread
            # destroyed while running aborts the process; this is the join that prevents it.
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
        # The bricked 3D view owns a long-lived QThread. "QThread: Destroyed while thread is still
        # running" aborts the interpreter, so it is stopped and joined here like every other worker.
        try:
            vol, self._native3d = self._native3d, None
            close = getattr(vol, "close", None)
            if callable(close):
                close()
        except Exception:                            # noqa: BLE001
            pass
        try:
            # The TIME axis has an animation thread of its own, and Qt aborts the process on a
            # QThread destroyed while it is still running. Closing a window mid-playback is the
            # ordinary way to meet that, so it is joined here exactly as the region slider is.
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
    """Registry of open :class:`RegionViewer` windows, keyed by a monotonic ID.

    The root plate window owns one of these. It is the single source of "what windows are open",
    so the Open View list is a pure VIEW of it and can never drift from the real set of windows.
    Memory is polled here (not per window) so one warning speaks for the whole app. It also owns
    the :class:`ViewDefaults` every new window opens with, for the same reason: a default has to
    outlive the windows that read it.
    """

    windowsChanged = Signal()          # the set of open windows changed
    memoryChanged = Signal(float)      # process RSS as a fraction 0..1 of total RAM
    # WHATEVER WORK IS RUNNING, as one immutable ``squidmip._progress.ProgressReport``, or None when
    # nothing is. Julio: "Where the memory bar is, there should also be a loading bar for whichever
    # operator we're applying in bulk or in a specific window, even if it's preview."
    #
    # It lives on the MANAGER for the same reason memory does (see the class docstring): the bar is
    # ONE bar next to ONE memory bar, and the work it reports comes from several producers -- a
    # plate-wide operator run, a run started in a region window, and the raw preview. A per-window
    # signal would need the navigator to subscribe to windows that come and go, and to decide which
    # of them the single bar is currently about. The manager already outlives them all.
    runProgressChanged = Signal(object)   # ProgressReport | None
    viewFocused = Signal(object)       # a window was opened/raised -> its regions (list[str])
    # A window was just SPAWNED, carrying the window itself. ``windowsChanged`` says the SET
    # changed and is what a list view wants; a subscriber that has to reach into the new window's
    # napari pane (the plate, which follows its contrast and its eye icons) needs the window, and
    # deriving "which one is new" by differencing a set is a second answer to a question the
    # spawn already knows. Emitted after the window is registered, so the registry is coherent by
    # the time anyone reads it.
    windowOpened = Signal(object)      # -> the new RegionViewer

    def __init__(self, reader: Any = None, meta: Optional[dict] = None,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._reader = reader
        self._meta = meta
        self._windows: "dict[int, RegionViewer]" = {}
        self._next_id = 1
        self._focused_id: Optional[int] = None    # which view is active (its plate hue reads brighter)
        self._selected_ids: "list[int]" = []      # navigator multi-selection (Linux shift/ctrl)
        # Set by the root PlateWindow so every window's "Operators for this window" dropdown is the
        # SAME registry + the SAME run_operator (the CLI engine), scoped to that view.
        self.operator_specs: "list" = []
        self.run_operator: Optional[Any] = None
        # THE GLOBAL DEFAULTS (Task 6, 2026-07-29). Here and not on PlateWindow: windows come and
        # go, the manager is the registry, so the registry is the one lifetime the defaults can
        # share. A new window reads these at construction; an already-open window is never touched
        # by a later change, because a default is a fact about the NEXT window.
        self.defaults = ViewDefaults()

        #: The most recent report of whatever is running, or None. Held as well as emitted so a
        #: navigator built DURING a run shows the bar immediately instead of staying blank until the
        #: next unit lands -- which on decon, where one unit is minutes, is most of the run.
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
        """Publish (or clear, with None) what is running, for the navigator's bar.

        LAST WRITER WINS, on purpose. There is one bar, so there is one answer; producers do not
        overlap in practice (``_stop_preview`` runs before an operator run starts), and if they ever
        did, a bar that shows the most recent report is a true statement about SOMETHING running,
        where an aggregate over two different denominators would be a true statement about nothing.
        """
        self._run_progress = report
        self.runProgressChanged.emit(report)

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
        """Open a CHILD window from an ROI drawn in a parent window (the next level of the tree).

        Structurally the child is a window over the same regions carrying the ROI box; cropping the
        load to the box lands with the loader work. Titled so the Open View list shows the nesting.

        The child's contrast comes from ``parent_id`` via :meth:`_baseline_for`, so the caller does
        not have to remember to hand it over. ``luts``, if given, overrides that derivation; it is
        for a caller that means a specific LUT set rather than "whatever my parent has"."""
        regions = [str(r) for r in regions if r]
        if not regions:
            return None
        base = RegionViewer._region_label(regions)
        title = f"{base}  ◂ view {parent_id}" if parent_id is not None else base
        return self._spawn(regions, title=title, roi_bbox=roi_bbox, parent_id=parent_id, luts=luts)

    def _baseline_for(self, parent_id: Optional[int]) -> "dict[str, Any]":
        """The settings a NEW window opens with: the global default for each one, except that an
        ``_INHERIT`` setting takes the OPENER's current value when a window opened it.

        This is where the contrast rule lives, and it lives here exactly once. A window opened from
        the plate has no opener window, so ``_INHERIT`` falls through to the default; an ROI child
        has one, so it looks like its parent. The parent is read LIVE (``current_settings``) because
        contrast is dragged in napari and the layers are the only thing that knows the answer.
        """
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
        """Give window *window_id* a new display label. Returns False when the id is not in the
        registry or the name is blank, so a caller can say so rather than appear to succeed.

        Mirrors :meth:`make_default` deliberately: same lookup, same "False when absent" contract.
        The repaint is FREE — the navigator connects ``windowsChanged`` to ``refresh``, and
        ``refresh`` reads the title back off the window — so this adds no signal to the widget with
        the documented use-after-free (see ``OpenViewList.refresh``).

        What this does NOT touch, and the reason a rename is safe: the key of ``self._windows``,
        the ``view_id`` on every record this window logs, the ``Qt.UserRole`` int on its navigator
        row, ``parent_id`` nesting, ``_baseline_for``'s contrast inheritance, ``make_default``, and
        the plate's ``_followed_windows`` set. Every one of those is the integer.
        """
        win = self._windows.get(int(window_id))
        if win is None:
            return False
        if not win.set_display_name(name):
            return False
        self.windowsChanged.emit()
        return True

    def make_default(self, window_id: int) -> bool:
        """Adopt one window's settings as the global defaults, for windows opened FROM NOW ON.

        The explicit act that replaces propagation: no already-open window is touched, including
        this one, except that it stops reporting diverged because it no longer is. Returns False
        when the id is not in the registry, so a caller can say so rather than appear to succeed.
        """
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
        # THE WINDOW-OPEN CLOCK starts HERE, before the window is built, because building the napari
        # pane is time the user waits: a clock started after the constructor would measure
        # everything except the part Julio complained about ("If we can speed up window loading
        # time, that would be good"). It stops in _on_plane, the interface-thread handler that adds
        # the first mosaic layer -- not in the worker that produced it, because the gap between
        # those two is queue delay and queue delay is the suspect.
        n = len(regions)
        clock = _measure.WindowOpen(
            f"{'ROI in ' if roi_bbox is not None else ''}{n} region{'' if n == 1 else 's'}: "
            f"{RegionViewer._region_label(regions)}",
            n_targets=n)
        baseline = self._baseline_for(parent_id)
        if luts is not None:
            baseline["luts"] = luts          # an explicit LUT set beats the derived one
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
        self.windowOpened.emit(win)                     # ...so the plate can follow ITS napari
        self.windowsChanged.emit()
        self.viewFocused.emit(list(win._regions))       # highlight its regions on the plate
        return win

    def _replay_cached_results(self, win: RegionViewer) -> int:
        """Give a NEWLY OPENED window every operator result already computed for its region.

        Julio: "even if we have a cache of operations, when it propagates to other windows, it
        adds a layer, but it doesn't toggle it." ``PlateWindow._deliver_to_views`` settled that for
        the windows open at the moment a run finishes. This is the other half of the same
        sentence: a window opened AFTER the run got nothing at all, and the only way to see the
        result in it was to run the operator a second time over the same pixels.

        Reuse, not recompute -- and not IPC either. Both windows are in ONE interpreter over ONE
        reader object (``DEFAULT_MAX_GUI=1`` plus an flock refuses a second process), so the
        second window is handed the FIRST window's ``Result`` object itself: no re-read, no
        re-fuse, no copy.

        ``visible=False``, always. This window did not ask for the run, and the rule the
        propagation path already follows is that asking is the consent: the layer is there to
        toggle and nothing changes under someone who just opened a window. That also keeps
        ``_on_done``'s ``show_op(_RAW_OP)`` from firing, so raw stays the visible group.
        """
        from squidmip._recipe import acquisition_version, cached_operator_results

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

    def clear_focus(self) -> None:
        """No view is selected -> clear the plate wash. Emitting empty regions makes the plate's hue
        refresh find no focused view and paint nothing."""
        self._focused_id = None
        self._selected_ids = []
        self.viewFocused.emit([])

    @property
    def selected_ids(self) -> "list[int]":
        """Window ids selected in the navigator (Linux shift/ctrl multi-select). The plate washes
        each in its own hue."""
        return [i for i in getattr(self, "_selected_ids", []) if i in self._windows]

    def set_selected(self, ids: "Sequence[int]") -> None:
        """The navigator selection changed (possibly many rows). Store it and re-tint the plate."""
        self._selected_ids = [int(i) for i in ids]
        self._focused_id = self._selected_ids[0] if self._selected_ids else None
        self.viewFocused.emit([])                        # triggers PlateWindow._refresh_view_hues

    def focus(self, window_id: int) -> None:
        win = self._windows.get(int(window_id))
        if win is not None:
            self._focused_id = int(window_id)
            win.showNormal()
            win.raise_()
            win.activateWindow()
            self.viewFocused.emit(list(win._regions))   # move the plate wash onto this view

    def raise_views(self, ids: "Sequence[int]") -> None:
        """Bring the selected windows to the FRONT of the desktop (Julio: clicking a navigator row
        should raise its window). Un-minimise + raise each; activate the last for keyboard focus.
        Un-minimising also lifts a window collapsed by Collapse all."""
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
        """Bring the ROOT plate window to the front. Returns whether there was one to raise.

        Spencer, 2026-07-30: the plate "can get lost easily". It is the smallest window on the
        desktop and every view opened from it is larger, so by the third well the thing you
        navigate FROM is behind everything you navigated TO. Collapse all does not help -- it
        minimises the views AND leaves the plate wherever it was.

        The plate is reachable without new plumbing: ``PlateWindow`` constructs this registry as
        ``ViewerManager(parent=self)``, so the Qt parent IS the plate. Asking for it here rather
        than handing every ``RegionViewer`` a back-pointer keeps one object knowing the topology --
        the same reason the registry already owns "what windows are open".

        ``showNormal`` is conditional ON PURPOSE. ``focus()`` above calls it unconditionally, which
        is right there because a collapsed view must be restored; here it would UN-MAXIMISE a plate
        the user had maximised. Raising a window and resizing it are different requests, and this
        button only makes the first one.
        """
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
        """Minimise every open window at once (declutter when a bunch are open, Julio). They stay in
        the navigator; clicking a row restores that one (focus() does showNormal + raise)."""
        for win in list(self._windows.values()):
            try:
                win.showMinimized()
            except Exception:                            # noqa: BLE001 - best effort per window
                pass
        self._focused_id = None
        self.viewFocused.emit([])                        # nothing raised -> clear the plate wash

    def _on_window_closed(self, win: "RegionViewer") -> None:
        # A window closed before its mosaic ever landed is a wait somebody GAVE UP ON, which is the
        # most interesting open there is and the one that would otherwise leave no record at all.
        # No-op on a window that already loaded: WindowOpen.finish is idempotent and the first call
        # is the true one.
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
        # NESTED HIERARCHY with expand/collapse ARROWS (Julio: "arrows for the window object
        # hierarchy like Blender") — ROI children nest under their parent window. Linux-style
        # shift/ctrl MULTI-SELECT so operators can target several views at once.
        self._tree.setRootIsDecorated(True)
        self._tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # ARROW-KEY NAVIGATION (Spencer, 2026-07-27): up/down step through open windows, and
        # because selection already raises (`_on_selection_changed` -> `raise_views`), arrows
        # drive behaviour that exists rather than needing a second code path.
        #
        # A QTreeWidget moves its current row on arrow keys for free, so the missing piece was
        # never the key handling: it was FOCUS. Nothing gave this tree keyboard focus, so the
        # arrows went to whatever had it instead, and the feature looked absent while the
        # machinery underneath was complete.
        self._tree.setFocusPolicy(Qt.StrongFocus)
        # The plate wash STRICTLY follows the navigator selection: select rows -> wash those views;
        # deselect (nothing selected) -> no wash. itemActivated (double-click) also raises the window.
        self._tree.itemActivated.connect(self._on_activated)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        # RENAME (Julio, 2026-08-03: "we should be able to rename our windows"). A context menu and
        # a modal QInputDialog, NOT an in-place QTreeWidget editor: `refresh()` calls
        # `self._tree.clear()` on every `windowsChanged`, which destroys the item being edited, so
        # in-place editing needs refresh() to become incremental first — a bigger and riskier change
        # to this widget (see its use-after-free note) than the feature is worth.
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self._syncing = False   # guards refresh()'s programmatic selection from re-emitting
        lay.addWidget(self._tree, 1)

        row = QHBoxLayout()
        row.setSpacing(6)
        # "views", plural, because the tree is ExtendedSelection and the button closes all of it.
        # The old label said "Close view" while the handler read currentItem(), so the name and the
        # behaviour agreed with each other and BOTH disagreed with the surface they sat on.
        close_btn = QPushButton("Close selected views")
        close_btn.setToolTip("Close every view selected here (shift/ctrl-click to select several).")
        close_btn.clicked.connect(self._close_selected)
        collapse_btn = QPushButton("Collapse all")
        collapse_btn.setToolTip("Minimise every open window (click a row to bring one back).")
        collapse_btn.clicked.connect(self._manager.collapse_all)
        row.addWidget(close_btn)
        row.addWidget(collapse_btn)
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

        # THE WORK BAR, directly under the memory bar because that is where Julio asked for it:
        # "Where the memory bar is, there should also be a loading bar for whichever operator we're
        # applying in bulk or in a specific window, even if it's preview."
        #
        # HIDDEN WHEN IDLE, rather than parked empty. An always-present bar sitting at 0 % is
        # indistinguishable from a run that has started and produced nothing, which is precisely the
        # confusion this is meant to end. Absent means nothing is running; present means something
        # is, and it says what.
        self._work_label = QLabel("")
        self._work_label.setStyleSheet("color:#8b949e;font-size:11px;border:none;")
        self._work_label.setWordWrap(True)
        self._work_label.hide()
        lay.addWidget(self._work_label)
        self._work_bar = QProgressBar(self)
        self._work_bar.setTextVisible(False)
        self._work_bar.setFixedHeight(14)
        # BLUE, where memory is green/red. Two identically-coloured bars stacked on each other is
        # one bar with a mystery second value; the colour is what says these measure different things.
        self._work_bar.setStyleSheet(
            "QProgressBar{background:#161b22;border:1px solid #30363d;border-radius:3px;}"
            "QProgressBar::chunk{background:#1f6feb;border-radius:3px;}"
        )
        self._work_bar.hide()
        lay.addWidget(self._work_bar)

        manager.windowsChanged.connect(self.refresh)
        manager.memoryChanged.connect(self._on_memory)
        manager.runProgressChanged.connect(self._on_run_progress)
        # A navigator built mid-run must not wait for the next unit to find out (see
        # ViewerManager._run_progress). Ask once, now.
        self._on_run_progress(manager.run_progress)
        self.refresh()

    def take_status_row(self) -> tuple:
        """Hand the memory bar and the run-progress bar to whoever is going to show them.

        Julio, 2026-08-03: "the status bar and memory bar should be moved to inside the logger so
        that we save space." The v2 drawing deletes the status block that sat under this list.

        THE WIRING IS NOT TOUCHED, and that is the point. ``manager.memoryChanged`` and
        ``manager.runProgressChanged`` are connected to ``_on_memory`` / ``_on_run_progress`` on
        THIS object, and those write into these four widgets by attribute. Removing them from this
        layout and letting another layout adopt them reparents the pixels and leaves every one of
        those paths intact — including ``_on_run_progress``'s hide/show, which is what makes the
        progress bar absent while nothing runs wherever it lives. Rebuilding them in the log panel
        instead would have meant two memory bars and a choice about which one is real.

        Returns ``(memory_caption, memory_bar, work_caption, work_bar)`` in the order
        :meth:`squidmip._logpanel.LogPanel.adopt_status_row` takes them. Callable once; calling it
        twice is harmless (``removeWidget`` on an absent widget is a no-op) but pointless.
        """
        lay = self.layout()
        widgets = (self._mem_label, self._mem_bar, self._work_label, self._work_bar)
        for w in widgets:
            lay.removeWidget(w)
        return widgets

    def showEvent(self, e):
        """Hand the tree keyboard focus, and give the arrows a row to start from.

        Without a current item the first Down press selects nothing on some styles, so the feature
        reads as broken on exactly the keystroke a user tries first.
        """
        super().showEvent(e)
        self._tree.setFocus()
        if self._tree.currentItem() is None and self._tree.topLevelItemCount():
            # setCurrentItem WOULD select, and selecting raises a window. Guard it the same way
            # refresh() does, so merely opening the panel does not reorder the user's windows.
            self._syncing = True
            try:
                self._tree.setCurrentItem(self._tree.topLevelItem(0))
            finally:
                self._syncing = False

    def refresh(self) -> None:
        # A DESTROYED navigator must not rebuild itself. `manager.windowsChanged` is connected to
        # this bound method, and the ViewerManager outlives any one navigator, so a window closing
        # AFTER this widget's C++ side was destroyed re-enters here and touches `self._tree`. That
        # is a use-after-free, and it does not raise: it segfaults, with the crash landing in
        # `expandAll()` and blaming whichever test happened to run last. Measured that way during
        # the gap 1 work: every test passed, then the process died at 100% inside
        # `conftest.pytest_sessionfinish`'s close loop, so pytest never printed a summary and a
        # fully green suite could not be committed.
        #
        # sip.isdeleted is the only reliable question here. `try/except RuntimeError` does not
        # help, because PyQt raises that only when it KNOWS the object is gone; a widget torn down
        # by its parent's C++ destructor leaves a wrapper that still looks alive.
        if _sip is not None and _sip.isdeleted(self):
            return
        # Rebuild as a NESTED tree (ROI children under their parent window), then restore the multi-
        # selection from the manager (guarded so the programmatic selection does not re-fire
        # _on_selection_changed). No selection => no wash.
        self._syncing = True
        try:
            self._tree.clear()
            items: "dict[int, QTreeWidgetItem]" = {}
            windows = self._manager.windows
            by_id = {int(w.window_id): w for w in windows}
            # Place parents before children: a window whose parent isn't open yet lands at the root.
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
            self._tree.expandAll()                       # show the nested ROIs open by default
            selected = set(self._manager.selected_ids)
            for wid, item in items.items():
                if wid in selected:
                    item.setSelected(True)
        finally:
            self._syncing = False

    def _on_selection_changed(self) -> None:
        """Row selection IS the wash and the operator target set (Linux multi-select): the plate
        washes every selected view in its hue; empty selection clears the wash."""
        if self._syncing:
            return
        ids = [int(i) for i in (it.data(0, Qt.UserRole) for it in self._tree.selectedItems())
               if i is not None]
        self._manager.set_selected(ids)     # plate wash for every selected view
        self._manager.raise_views(ids)      # and bring the selected window(s) to the front

    def _on_activated(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        wid = item.data(0, Qt.UserRole)
        if wid is not None:
            self._manager.focus(int(wid))

    # -- rename -------------------------------------------------------------------------
    def _on_context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if item is None:
            return
        wid = item.data(0, Qt.UserRole)
        if wid is None:
            return
        menu = QMenu(self)
        act = menu.addAction("Rename…")
        # `exec`, not `exec_`: PyQt6 removed every trailing-underscore alias (see the note at the
        # bottom of `_viewer.py`), and this is a context menu, so an AttributeError here would only
        # ever fire in front of a user.
        if menu.exec(self._tree.viewport().mapToGlobal(pos)) is act:
            self.rename_window(int(wid))

    def rename_window(self, window_id: int) -> bool:
        """Ask for a new label for *window_id* and apply it. Returns whether anything changed.

        Public and separate from the menu handler so a keybinding, a test, or a future in-place
        editor drives the SAME path rather than a second one.

        The dialog seeds with the current label and asks for a label only: the ``[wid]`` bracket is
        never in the box, because a user-editable bracket would break the join between a log line
        and the window that emitted it.
        """
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

    def _close_selected(self) -> None:
        """Close EVERY selected row, not just the current one.

        The tree has been ExtendedSelection since 1073999 and the wash already follows the whole
        selection (see ``_on_selection_changed``, which reads ``selectedItems()``), so this was the
        one singular actor on a multi-select surface: select four views, press the button, three of
        them stay open and nothing says why.

        ``currentItem()`` was also not the same question as "what is selected". Current is the
        focus rectangle and it survives ctrl-clicking a row OFF, so the old button could close a
        view the user had just deselected.

        The ids are collected BEFORE the first close, and that ordering is load-bearing. Closing a
        window fires ``windowsChanged`` -> ``refresh()``, which calls ``self._tree.clear()`` and
        destroys every ``QTreeWidgetItem`` in it; reading item data across that loop is a
        use-after-free of exactly the kind the comment in ``refresh`` describes. ``close()`` is a
        no-op for an id already gone, so a parent window that takes its nested ROI children with
        it needs no special case here.
        """
        ids = [int(i) for i in (it.data(0, Qt.UserRole) for it in self._tree.selectedItems())
               if i is not None]
        for wid in ids:
            self._manager.close(wid)

    def _on_run_progress(self, report) -> None:
        """Draw (or take down) the work bar. ``report`` is a ``ProgressReport``, or None for idle.

        DETERMINATE ONLY WHEN THE REPORT IS. An indeterminate report gets Qt's busy animation
        (``setRange(0, 0)``) and its count without a percentage, never a fabricated one -- the same
        rule ``_progress.ProgressReport.percent`` and ``squidmip._activity`` already follow, and for
        the same reason: a progress bar that invents a denominator is a lie that gets believed.
        """
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
            self._work_bar.setRange(0, 0)            # Qt's busy sweep: working, total unknown
        else:
            self._work_bar.setRange(0, 100)
            self._work_bar.setValue(int(percent))
        self._work_label.show()
        self._work_bar.show()

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
