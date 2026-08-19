"""napari mosaic view: the processing-layer/channel hierarchy over a ViewerModel."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence as _SequenceABC
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from squidxplorer import _bitdepth
from squidxplorer._logpane import get_logger

log = get_logger("napari")

#: Fallback GPU 3D texture cap; the live value is read off the canvas at runtime.
_DEFAULT_MAX_3D_TEXTURE = 2048

#: Metadata key where a z-collapsed layer's stack is stashed (see ``_present_z_axis``).
_Z_STASH = "_zstack"

# napari is NOT imported at module scope: it pulls Qt, and the pure hierarchy logic
# must stay importable in a headless process.

VIEWER_ENV = "SQUIDXPLORER_VIEWER"
_NAPARI = "napari"
META_KEY = "squidxplorer"

#: Retired ndviewer_light spellings, recognised only to warn.
_RETIRED_NDV_NAMES = ("ndv", "ndviewer", "ndviewer_light")


def resolve_viewer(env: Optional[dict] = None) -> str:
    """Which viewer to build. There is exactly one: ``"napari"``."""
    src = os.environ if env is None else env
    want = str(src.get(VIEWER_ENV, "")).strip().lower()
    if want in _RETIRED_NDV_NAMES:
        log.warning(
            "%s=%s asks for the ndviewer_light fallback, which was deleted on 2026-07-30 "
            "(it imports PyQt5 at module scope and cannot share a process with a Qt6 napari). "
            "Building napari instead. Drop the variable to silence this.",
            VIEWER_ENV, want,
        )
    return _NAPARI


def napari_enabled(env: Optional[dict] = None) -> bool:
    """True when the napari view is the selected viewer, which is now always."""
    return resolve_viewer(env) == _NAPARI


# --------------------------------------------------------------------------------------
# Binding assertions: a napari upgrade that moves a symbol fails loudly at construction.
# --------------------------------------------------------------------------------------

REQUIRED_NAPARI_BINDINGS: tuple[tuple[str, str], ...] = (
    ("napari.components", "ViewerModel"),
    ("napari.components", "LayerList"),
    ("napari.qt", "QtViewer"),
)

#: Private napari symbols we depend on, checked separately (no ``__all__`` promise).
REQUIRED_PRIVATE_BINDINGS: tuple[tuple[str, str], ...] = (
    ("napari._qt.layer_controls", "QtLayerControlsContainer"),
)

REQUIRED_LAYER_ATTRS: tuple[str, ...] = ("metadata", "visible", "contrast_limits", "scale",
                                         "translate", "name", "events")
REQUIRED_LAYERLIST_ATTRS: tuple[str, ...] = ("link_layers", "unlink_layers")

#: PRIVATE attributes we read off a ``ViewerModel``. ``_canvas_size`` is how big napari believes
#: its canvas is, in ``(height, width)``, and it is what napari's own ``reset_view`` measures
#: against -- so :meth:`MosaicLayers.frame_bbox_um` reads the SAME number rather than asking Qt
#: for the widget size, which would be a second answer to "how big is the canvas" that agrees
#: with napari only until one of them is stale. It also exists on a Qt-free ``ViewerModel``,
#: which is what makes framing testable with no GL at all.
#:
#: Checked here rather than trusted: it carries no ``__all__`` promise, and a napari upgrade that
#: renames it must fail with a sentence, not silently frame every camera against ``(800, 600)``.
REQUIRED_MODEL_PRIVATE_ATTRS: tuple[str, ...] = ("_canvas_size",)


class NapariBindingError(RuntimeError):
    """A napari symbol this module depends on has moved, been renamed, or been removed."""


def verify_napari_bindings(modules: Optional[dict] = None) -> None:
    """Fail loudly if any napari API this module drives is missing."""
    import importlib

    missing: list[str] = []
    for dotted, attr in REQUIRED_NAPARI_BINDINGS:
        try:
            mod = modules[dotted] if modules and dotted in modules else importlib.import_module(dotted)
        except Exception as exc:  # pragma: no cover - import failure is reported, not swallowed
            missing.append(f"{dotted} (import failed: {exc!r})")
            continue
        if not hasattr(mod, attr):
            missing.append(f"{dotted}.{attr}")
        # A public name that exists but is no longer exported is a deprecation in progress.
        exported = getattr(mod, "__all__", None)
        if exported is not None and attr not in exported:
            missing.append(f"{dotted}.{attr} (present but no longer in __all__)")

    for dotted, attr in REQUIRED_PRIVATE_BINDINGS:
        try:
            mod = modules[dotted] if modules and dotted in modules else importlib.import_module(dotted)
        except Exception as exc:  # pragma: no cover
            missing.append(f"{dotted} (PRIVATE; import failed: {exc!r})")
            continue
        if not hasattr(mod, attr):
            missing.append(f"{dotted}.{attr} (PRIVATE)")

    # Private ATTRIBUTES of a ViewerModel, checked on a real instance because that is the only
    # place they exist. Constructing one is cheap and Qt-free -- and it is the same object
    # `MosaicLayers` is about to wrap, so if this passes, `frame_bbox_um` can read what it needs.
    for attr in REQUIRED_MODEL_PRIVATE_ATTRS:
        try:
            components = (modules["napari.components"] if modules and "napari.components" in modules
                          else importlib.import_module("napari.components"))
            probe = components.ViewerModel()
        except Exception as exc:  # pragma: no cover - reported, not swallowed
            missing.append(f"napari.components.ViewerModel (PRIVATE probe failed: {exc!r})")
            break
        if not hasattr(probe, attr):
            missing.append(f"napari.components.ViewerModel.{attr} (PRIVATE)")

    if missing:
        raise NapariBindingError(
            "napari's API has moved under us; the mosaic view cannot be trusted to render.\n"
            "Missing or de-exported: " + ", ".join(missing) + "\n"
            "This is a hard failure on purpose. The alternative — binding to whatever is there "
            "and hoping — is how `_voxel_scale` ran every time and did nothing for its whole life."
        )


# --------------------------------------------------------------------------------------
# The hierarchy — pure logic, no napari import
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MosaicKey:
    """Identity of one displayed mosaic: which processing layer, which channel."""

    op: str
    channel: str

    def label(self) -> str:
        """Human label for the napari layers list. Never parsed back."""
        return f"{self.op} · {self.channel}"

    def as_metadata(self) -> dict:
        return {META_KEY: {"op": self.op, "channel": self.channel}}


def key_of(layer: Any) -> Optional[MosaicKey]:
    """Recover a layer's identity from its metadata. Returns None for foreign layers."""
    meta = getattr(layer, "metadata", None) or {}
    ours = meta.get(META_KEY)
    if not isinstance(ours, dict):
        return None
    op, channel = ours.get("op"), ours.get("channel")
    if op is None or channel is None:
        return None
    return MosaicKey(str(op), str(channel))


def scale_translate_from_bbox_um(
    bbox_um: Sequence[float], shape: Sequence[int]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Map a stage-um ``(x0, y0, x1, y1)`` bbox onto napari's ``(y, x)`` scale/translate."""
    x0, y0, x1, y1 = (float(v) for v in bbox_um)
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"bbox_um must satisfy x1 > x0 and y1 > y0, got {tuple(bbox_um)!r}")
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"shape must be positive, got {tuple(shape)!r}")
    scale = ((y1 - y0) / h, (x1 - x0) / w)
    translate = (y0, x0)
    return scale, translate


#: Fraction of the canvas left blank around a framed box. This is napari's OWN default
#: (``ViewerModel.reset_view(margin=0.05)``), taken from there rather than chosen, so that a box
#: framed by :func:`camera_for_bbox_um` and a view reset by napari leave the same gap. A second
#: margin convention would make "reset" and "frame this" disagree by a few percent forever.
FRAME_MARGIN = 0.05


def camera_for_bbox_um(
    bbox_um: Sequence[float],
    canvas_size_hw: Sequence[float],
    *,
    margin: float = FRAME_MARGIN,
) -> tuple[tuple[float, float], float]:
    """``((centre_y, centre_x), zoom)`` that puts *bbox_um* on a canvas of *canvas_size_hw*.

    THE SAME ARITHMETIC NAPARI APPLIES TO THE WHOLE SCENE, applied to a box the scene does not
    describe. ``ViewerModel._get_2d_camera_zoom`` is ``(1 - margin) * min(canvas / extent)`` and
    ``_calculate_view_center`` is the midpoint; napari 0.8 exposes no public API that takes an
    extent, so the formula is written once HERE and every camera framing in this app calls it.
    Deriving it a second time somewhere else is how "fit the region" and "fit this field" end up
    disagreeing about what fitting means.

    ``bbox_um`` is ``(x0, y0, x1, y1)`` -- X FIRST, the spelling every world box in this app uses
    (see :func:`scale_translate_from_bbox_um`). ``canvas_size_hw`` is ``(height, width)``, which
    is napari's own order for both ``ViewerModel._canvas_size`` and ``VispyCanvas.size`` (whose
    docstring says so, and whose getter reverses vispy's ``(w, h)`` to produce it).

    THOSE TWO ORDERS DIFFER, AND READING ONE AS THE OTHER DOES NOT RAISE. It frames the box
    against the wrong canvas edge, which on a square canvas is invisible and on any other looks
    like somebody just preferred a different zoom. ``_brick_view._frame_camera`` did exactly that
    -- ``cw, ch = ...canvas.size`` on a property that returns ``(height, width)``, then
    ``min(cw / w_um, ch / h_um)`` -- until this function existed to be called instead.

    Raises rather than guessing on a degenerate box, a non-positive canvas, or a margin outside
    ``[0, 1)``: a camera pointed at a number nobody measured shows somewhere else with no error.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox_um)
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"bbox_um must satisfy x1 > x0 and y1 > y0, got {tuple(bbox_um)!r}")
    h_px, w_px = (float(v) for v in canvas_size_hw)
    if h_px <= 0 or w_px <= 0:
        raise ValueError(f"canvas_size_hw must be positive (height, width), "
                         f"got {tuple(canvas_size_hw)!r}")
    if not 0 <= float(margin) < 1:
        raise ValueError(f"margin must be in [0, 1), got {margin!r}")

    # Height against height, width against width. The `min` is what makes the WHOLE box fit: the
    # tighter axis decides, and the looser one gets the spare canvas.
    zoom = (1.0 - float(margin)) * min(h_px / (y1 - y0), w_px / (x1 - x0))
    return ((y0 + y1) / 2.0, (x0 + x1) / 2.0), zoom


def placement_for(ndim: int, bbox_um: Sequence[float], shape: Sequence[int],
                  z_scale_um: Optional[float] = None) -> tuple[tuple, tuple]:
    """``(scale, translate)`` for a layer of *ndim* axes whose trailing two are ``(y, x)``."""
    scale, translate = scale_translate_from_bbox_um(bbox_um, shape)
    extra = max(0, int(ndim) - 2)
    lead = (float(z_scale_um) if (extra and z_scale_um) else 1.0,) * extra
    return lead + tuple(scale), (0.0,) * extra + tuple(translate)


def _colormap_rgb(layer: Any) -> Optional[tuple]:
    """The RGB a napari layer tints with at full intensity, or None if unreadable."""
    cm = getattr(layer, "colormap", None)
    colors = getattr(cm, "colors", None)
    if colors is None:
        return None
    try:
        row = colors[-1]
        return (float(row[0]), float(row[1]), float(row[2]))
    except Exception:                       # noqa: BLE001 - unknown colormap shape; say nothing
        return None


#: Tolerance for a lookup-table row sitting off the black-to-hue line: one 8-bit step.
_HUE_TOL = 2.0 / 255.0


def colormap_hue_rgb(layer: Any) -> "Optional[tuple[int, int, int]]":
    """The single 8-bit RGB a black-to-hue colormap reduces to, or None if it does not reduce."""
    cm = getattr(layer, "colormap", None)
    colors = getattr(cm, "colors", None)
    if colors is None:
        return None
    try:
        table = np.asarray(colors, dtype=float)
        if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 3:
            return None
        rgb = table[:, :3]
        last = rgb[-1]
        denom = float(last @ last)
        if denom <= 0.0:            # the map ends at black: there is no hue to name
            return None
        # Every row must land on the black-to-hue line; a perceptual ramp fails here.
        t = np.clip((rgb @ last) / denom, 0.0, 1.0)[:, None]
        if float(np.max(np.abs(rgb - t * last))) > _HUE_TOL:
            return None
        return tuple(int(round(min(1.0, max(0.0, float(v))) * 255.0)) for v in last)
    except Exception:                       # noqa: BLE001 - unknown colormap shape; say nothing
        return None


def colormap_mid_rgb(layer: Any) -> "Optional[tuple[int, int, int]]":
    """The 8-bit RGB of a colormap's MIDDLE stop — the representative tint of a map that does
    not reduce to black-to-hue (a measured stain LUT is white-topped, so its hue lives mid-curve).
    """
    cm = getattr(layer, "colormap", None)
    colors = getattr(cm, "colors", None)
    if colors is None:
        return None
    try:
        table = np.asarray(colors, dtype=float)
        if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 3:
            return None
        mid = table[table.shape[0] // 2, :3]
        return tuple(int(round(min(1.0, max(0.0, float(v))) * 255.0)) for v in mid)
    except Exception:                       # noqa: BLE001 - unknown colormap shape; say nothing
        return None


class MosaicLayers:
    """The two-level hierarchy over a napari ``ViewerModel``."""

    def __init__(self, model: Any) -> None:
        self._model = model
        # channel -> the layers showing that channel, across every processing layer. Linked.
        self._by_channel: dict[str, list[Any]] = {}
        self._programmatic = 0
        self._user_contrast_cbs: list[Any] = []
        self._user_visibility_cbs: list[Any] = []
        self._last_visible: dict[str, bool] = {}
        self._user_op_cbs: list[Any] = []
        self._last_op_visible: dict[str, bool] = {}
        self._user_colormap_cbs: list[Any] = []
        self._last_colormap: dict[str, tuple] = {}
        # Last contrast seen per channel: link echoes carry an identical value and are
        # collapsed by value, including our own programmatic writes.
        self._last_seen: dict[str, tuple[float, float]] = {}
        self._max_3d_texture: int = _DEFAULT_MAX_3D_TEXTURE
        # (op, channel, property) triples currently being mirrored, keyed rather than a
        # single flag so a mirror running inside another identity's mirror is not swallowed.
        self._mirroring: set = set()
        # Re-entrancy guard for the selection-follows-visibility rule.
        self._selection_following = False
        try:
            model.dims.events.ndisplay.connect(self._reslice_hidden_layers)
        except Exception:                        # noqa: BLE001 - a stub model with no dims events
            pass

    def _reslice_hidden_layers(self, event=None) -> None:
        """Force-refresh hidden >2-D layers whose slice disagrees with their slice input after a 2D/3D flip."""
        for ly in self._all_ours():
            try:
                if bool(getattr(ly, "visible", False)) or int(getattr(ly, "ndim", 0)) <= 2:
                    continue
                ly.refresh(force=True)
            except Exception as exc:             # noqa: BLE001 - one odd layer is not the pane
                log.warning("could not re-slice %s after a 2D/3D flip: %s",
                            getattr(ly, "name", "layer"), exc)

    def _all_ours(self) -> list[Any]:
        """Every layer this pane made, including ones whose identity is surrendered while 3D is up."""
        out: list[Any] = list(self.ours())
        for peers in self._by_channel.values():
            for ly in peers:
                if ly not in out:
                    out.append(ly)
        return out

    @contextmanager
    def programmatic(self):
        """Mark contrast writes made BY US, so subscribers can ignore them."""
        self._programmatic += 1
        try:
            yield
        finally:
            self._programmatic -= 1

    @property
    def is_programmatic(self) -> bool:
        return self._programmatic > 0

    @property
    def model(self) -> Any:
        return self._model

    def ours(self) -> list[Any]:
        return [ly for ly in self._model.layers if key_of(ly) is not None]

    def ops(self) -> list[str]:
        """Processing layers currently present, in insertion order, de-duplicated."""
        seen: list[str] = []
        for ly in self.ours():
            k = key_of(ly)
            assert k is not None
            if k.op not in seen:
                seen.append(k.op)
        return seen

    def group(self, op: str) -> list[Any]:
        """Every channel layer belonging to one processing layer."""
        return [ly for ly in self.ours() if (k := key_of(ly)) is not None and k.op == op]

    def channels(self, op: str) -> list[str]:
        out: list[str] = []
        for ly in self.group(op):
            k = key_of(ly)
            assert k is not None
            if k.channel not in out:
                out.append(k.channel)
        return out

    def find(self, op: str, channel: str) -> Optional[Any]:
        """The representative layer of one identity: the one a control reads and writes."""
        for ly in self.ours():
            if key_of(ly) == MosaicKey(op, channel):
                return ly
        return None

    # An identity (op, channel) may be rendered by SEVERAL layers (one per brick in 3D).
    # `find` names the representative, `layers_for` names them all, and `_mirror_identity`
    # keeps IDENTITY_PROPS equal across them.

    #: Properties that belong to the identity rather than one layer object; ``blending``
    #: is deliberately absent (a rendering choice a volume legitimately makes differently).
    IDENTITY_PROPS: tuple = ("visible", "contrast_limits", "colormap", "gamma", "opacity")

    def layers_for(self, op: str, channel: str) -> list[Any]:
        """Every layer rendering one identity, representative first. Derived, never cached."""
        want = MosaicKey(str(op), str(channel))
        return [ly for ly in self.ours() if key_of(ly) == want]

    def adopt(self, op: str, channel: str, layer: Any) -> Any:
        """Bring a layer built elsewhere into the model under identity ``(op, channel)``."""
        key = MosaicKey(str(op), str(channel))
        siblings = self.layers_for(key.op, key.channel)
        meta = dict(getattr(layer, "metadata", None) or {})
        meta.update(key.as_metadata())
        layer.metadata = meta
        if siblings:
            with self.programmatic():
                for prop in self.IDENTITY_PROPS:
                    try:
                        self._set_identity_prop(layer, prop, getattr(siblings[0], prop))
                    except AttributeError:       # a layer type without this property
                        continue
        self._label_units(layer)
        self._register_channel(key.channel, layer)
        if not siblings and bool(getattr(layer, "visible", False)):
            # A FIRST surface arriving lit is the same gesture as one the user lights.
            self._darken_other_ops(key.channel, layer)
        return layer

    def drop_layer(self, layer: Any) -> None:
        """Remove one layer while leaving its identity alive; unlinks before removing."""
        for channel, peers in self._by_channel.items():
            if layer not in peers:
                continue
            try:
                self._model.layers.unlink_layers([layer], ("contrast_limits",))
            except Exception:                    # noqa: BLE001 - never linked, or already gone
                pass
            peers.remove(layer)
            linkable = self._link_set(channel)
            if len(linkable) > 1:
                self._model.layers.link_layers(linkable, ("contrast_limits",))
            break
        try:
            self._model.layers.remove(layer)
        except Exception:                        # noqa: BLE001 - already removed
            pass

    def _link_set(self, channel: str) -> list[Any]:
        """The layers of *channel* that ``link_layers`` connects: one per identity."""
        seen: set = set()
        out: list[Any] = []
        for ly in self._by_channel.get(channel) or []:
            k = key_of(ly)
            ident = (k.op, k.channel) if k is not None else id(ly)
            if ident in seen:
                continue
            seen.add(ident)
            out.append(ly)
        return out

    @staticmethod
    def _widen_range(layer: Any, lo: float, hi: float) -> bool:
        """Open ``layer``'s contrast slider to at least ``(lo, hi)``. NEVER narrows. Moved?

        ONE copy of the rule, because there are two callers with the same requirement and
        opposite reasons. ``_set_identity_prop`` widens so an inherited window is not clamped
        away by a range napari sized from one brick's dim corner; ``widen_contrast_range`` widens
        because a later region proved the dataset holds bigger numbers than the first one did.
        Both must be incapable of narrowing: a narrower range does not merely restyle the slider,
        napari clips ``contrast_limits`` into it and can reset the window outright.

        Returns False for anything with no range to widen -- Labels, Points, Shapes -- so callers
        can walk a whole layer list without sniffing types.
        """
        try:
            r0, r1 = (float(v) for v in layer.contrast_limits_range)
        except Exception:                        # noqa: BLE001 - not an intensity layer
            return False
        new = (min(r0, float(lo)), max(r1, float(hi)))
        if new == (r0, r1):
            return False
        try:
            layer.contrast_limits_range = new
        except Exception:                        # noqa: BLE001 - one odd surface is skipped
            return False
        return True

    def widen_contrast_range(self, lo: float, hi: float) -> int:
        """Open every image layer's contrast slider to at least ``(lo, hi)``. Returns how many moved.

        Called when `_bitdepth` raises the dataset ceiling -- the C3-then-E7 case, where the
        first region read looked 12-bit and a later one proved 14. Layers already on screen were
        built against the old ceiling and would otherwise keep a slider that cannot reach their
        own brightest pixels.

        WRAPPED IN ``programmatic()``. napari re-emits ``events.contrast_limits`` when the RANGE
        changes even though the window value does not move (it re-assigns the clipped tuple), and
        an echo that reaches the user-contrast subscriber reads as the user having dragged the
        slider -- which latches the plate to manual and kills per-region contrast.

        RGB layers are skipped: napari ignores contrast on them, and writing a range would be a
        no-op that looks like coverage.
        """
        moved = 0
        with self.programmatic():
            for layer in list(getattr(self._model, "layers", []) or []):
                if getattr(layer, "rgb", False):
                    continue
                if self._widen_range(layer, lo, hi):
                    moved += 1
        return moved

    @staticmethod
    def _set_identity_prop(layer: Any, prop: str, value: Any) -> bool:
        """Write one identity property onto one layer, widening the contrast range first. Returns whether it moved."""
        try:
            current = getattr(layer, prop)
        except Exception:                        # noqa: BLE001 - a layer type without this property
            return False
        try:
            if bool(current == value):
                return False
        except Exception:                        # noqa: BLE001 - unorderable value: write it
            pass
        if prop == "contrast_limits":
            # napari clamps contrast_limits to contrast_limits_range, so widen it first.
            try:
                MosaicLayers._widen_range(layer, float(value[0]), float(value[1]))
            except Exception:                    # noqa: BLE001 - no range to widen; write anyway
                pass
        try:
            setattr(layer, prop, value)
        except Exception:                        # noqa: BLE001 - one odd surface is skipped
            return False
        return True

    def _connect_identity_mirror(self, layer: Any) -> None:
        """Wire one layer into its identity's property mirror."""
        events = getattr(layer, "events", None)
        if events is None:
            return
        for prop in self.IDENTITY_PROPS:
            emitter = getattr(events, prop, None)
            if emitter is None:
                continue

            def _fire(event=None, _prop=prop, _src=layer) -> None:
                self._mirror_identity(_prop, _src)

            try:
                emitter.connect(_fire)
            except Exception:                    # noqa: BLE001 - a stub layer with no emitter
                continue

    def _mirror_identity(self, prop: str, src: Any) -> None:
        """Give every other layer of *src*'s identity the value *src* just took."""
        key = key_of(src)
        if key is None:
            return
        token = (key.op, key.channel, prop)
        if token in self._mirroring:
            return
        others = [ly for ly in self.layers_for(key.op, key.channel) if ly is not src]
        if not others:
            return                               # one layer IS the identity: every 2D case
        try:
            value = getattr(src, prop)
        except Exception:                        # noqa: BLE001 - nothing to mirror
            return
        self._mirroring.add(token)
        try:
            for ly in others:
                self._set_identity_prop(ly, prop, value)
        finally:
            self._mirroring.discard(token)

    def render_max_res_3d(self, on: bool) -> None:
        """Swap mosaics between the 2D multiscale pyramid and a full-res single-scale volume for 3D."""
        limit = int(self._max_3d_texture or _DEFAULT_MAX_3D_TEXTURE)
        with self.programmatic():
            for ly in self.ours():
                self._swap_layer_scale(ly, full_res=bool(on), limit=limit)

    @staticmethod
    def _fits_texture(level: Any, limit: int) -> bool:
        shp = getattr(level, "shape", None)
        if not shp:
            return False
        return max(int(s) for s in shp) <= int(limit)

    def _swap_layer_scale(self, ly: Any, *, full_res: bool, limit: int) -> None:
        meta = dict(getattr(ly, "metadata", None) or {})
        # A z-collapsed layer has no volume to render; restore its stack before either
        # direction of the swap, so one stash mechanism never stashes the other's placeholder.
        if _Z_STASH in meta:
            self._restore_layer_z(ly)
            meta = dict(getattr(ly, "metadata", None) or {})
        try:
            if full_res:
                data = pyramid_levels(ly.data)
                if data is None:
                    return                       # already single-scale, nothing to swap
                meta["_pyramid"] = data          # stash the pyramid so 2D can restore it
                ly.metadata = meta
                # napari renders 3D from one GL texture; target the finest pyramid level
                # that fits GL_MAX_3D_TEXTURE_SIZE, else the coarsest as a floor.
                chosen = data[-1]
                for lvl in data:
                    if self._fits_texture(lvl, limit):
                        chosen = lvl
                        break
                ly.multiscale = False
                ly.data = chosen
                log.info("napari 3D: rendering %s at %s (fills the %d px GPU texture budget; "
                         "full native res needs a crop: draw an ROI and open it)",
                         getattr(ly, "name", "layer"), tuple(getattr(chosen, "shape", ())), limit)
            else:
                pyr = meta.get("_pyramid")
                if pyr is None:
                    return                       # never swapped, or not one of ours
                ly.multiscale = True
                ly.data = list(pyr)
        except Exception as exc:                 # noqa: BLE001 - a render nicety, never fatal
            log.warning("napari 3D swap failed on %s: %s", getattr(ly, "name", "layer"), exc)

    def _restore_layer_z(self, ly: Any) -> None:
        """Give a z-collapsed layer its stack, its multiscale flag and its placement back."""
        meta = dict(getattr(ly, "metadata", None) or {})
        stash = meta.pop(_Z_STASH, None)
        if stash is None:
            return
        try:
            ly.metadata = meta
            ly.multiscale = bool(stash["multiscale"])
            ly.data = stash["data"]
            # AFTER the data: napari pads scale/translate to the new ndim with 1.0 / 0.0.
            ly.scale = stash["scale"]
            ly.translate = stash["translate"]
        except Exception as exc:                 # noqa: BLE001 - a presentation nicety, never fatal
            log.warning("z axis restore failed on %s: %s", getattr(ly, "name", "layer"), exc)

    def _collapse_layer_z(self, ly: Any, z_level: int) -> None:
        """Present a ``(z, y, x)`` layer as the single plane *z*, so it stops carrying a z axis."""
        meta = dict(getattr(ly, "metadata", None) or {})
        if _Z_STASH in meta:
            return                               # already collapsed; idempotent by design
        try:
            multiscale = bool(getattr(ly, "multiscale", False))
            data = ly.data
            # `list(...)`: a pyramid comes back as MultiScaleData, and indexing it as one
            # array walks the LEVELS instead of the z planes.
            levels = list(data) if multiscale else [data]
            if not levels or min(int(getattr(lv, "ndim", 2)) for lv in levels) < 3:
                return                           # already a plane: nothing to collapse
            scale, translate = tuple(ly.scale), tuple(ly.translate)
            meta[_Z_STASH] = {"data": list(levels) if multiscale else data,
                              "multiscale": multiscale,
                              "scale": scale, "translate": translate}
            coarsest = levels[-1]
            plane = coarsest[max(0, min(int(z_level), int(coarsest.shape[0]) - 1))]
            ly.metadata = meta
            ly.multiscale = False
            ly.data = plane
            # The coarsest level's own pixel size, not level 0's.
            fine_h, fine_w = int(levels[0].shape[-2]), int(levels[0].shape[-1])
            ly.scale = (scale[-2] * fine_h / float(plane.shape[-2]),
                        scale[-1] * fine_w / float(plane.shape[-1]))
            ly.translate = translate[-2:]
        except Exception as exc:                 # noqa: BLE001 - a presentation nicety, never fatal
            log.warning("z axis collapse failed on %s: %s", getattr(ly, "name", "layer"), exc)

    @staticmethod
    def _reduces_z(op: str) -> bool:
        """Does the operator behind this layer key declare ``consumes={"z"}``? Declaration, never name."""
        from squidxplorer._engine import Z_REDUCER, operator_consumes
        from squidxplorer._operations import operator_name

        try:
            return bool(operator_consumes(operator_name(str(op))) & Z_REDUCER)
        except Exception:                        # noqa: BLE001 - "raw", "computed", an unknown key
            return False

    def _present_z_axis(self) -> None:
        """Drop the pane's z axis while a z-reduced result is the layer on screen; else restore it."""
        model = self._model
        dims = getattr(model, "dims", None)
        if int(getattr(dims, "ndisplay", 2) or 2) == 3:
            return          # a 3D view is asking for the volume; taking z away would empty it
        op = self.visible_op()
        collapse = op is not None and self._reduces_z(op)
        # The plane the user was last looking at, so toggling raw back on does not jump to z=0.
        z = 0
        if dims is not None and int(getattr(dims, "ndim", 0) or 0) >= 3:
            try:
                z = int(dims.current_step[int(dims.ndim) - 3])
            except Exception:                    # noqa: BLE001 - a missing step is z=0
                z = 0
        with self.programmatic():
            for ly in self.ours():
                if collapse:
                    self._collapse_layer_z(ly, z)
                else:
                    self._restore_layer_z(ly)

    def add_mosaic(
        self,
        op: str,
        channel: str,
        data: Any,
        *,
        contrast_limits: Optional[tuple[float, float]] = None,
        colormap: Optional[Any] = None,
        multiscale: Optional[bool] = None,
        bbox_um: Optional[Sequence[float]] = None,
        visible: bool = True,
        blending: str = "additive",
        z_scale_um: Optional[float] = None,
    ) -> Any:
        """Add (or replace) the mosaic for one processing layer / channel pair.

        ``contrast_limits=None`` seeds a fluorescence auto-window; napari owns contrast
        from the moment the layer exists.
        """
        key = MosaicKey(str(op), str(channel))
        existing = self.find(key.op, key.channel)
        if existing is not None:
            # Reuse the layer: destroying it is slow and strands every subscriber bound to
            # it, and reuse keeps the user's contrast/colormap/visibility across regions.
            reused = self._reuse_layer(existing, data, bbox_um=bbox_um, z_scale_um=z_scale_um,
                                       multiscale=multiscale, visible=visible)
            self._present_z_axis()
            return reused

        kwargs: dict[str, Any] = {
            "name": key.label(),
            "metadata": key.as_metadata(),
            "visible": visible,
            # Additive, not the default 'translucent_no_depth': fluorescence channels are a
            # composite and must sum; with the default the last-added layer occludes the rest.
            "blending": blending,
        }
        window = contrast_limits
        if window is None:
            window = _auto_window_for(data, bool(multiscale))
        if window is not None:
            lo, hi = float(window[0]), float(window[1])
            # A degenerate window is passed through, NOT widened: widening it to (lo, lo+1)
            # renders a blank channel as full white, i.e. as signal.
            if hi > lo:
                kwargs["contrast_limits"] = (lo, hi)
        if colormap is not None:
            kwargs["colormap"] = colormap
        if multiscale is not None:
            kwargs["multiscale"] = multiscale

        # Placed at construction: assigning scale/translate to a live layer moves the dims
        # range under it and costs a second whole-region fuse.
        placed = False
        if bbox_um is not None:
            try:
                shape = tuple(_first_level_shape(data, bool(multiscale)))
                kwargs["scale"], kwargs["translate"] = placement_for(
                    len(shape), bbox_um, shape[-2:], z_scale_um)
                placed = True
            except (ValueError, TypeError, IndexError):
                placed = False

        ndim_before = int(getattr(getattr(self._model, "dims", None), "ndim", 0) or 0)
        with self.programmatic():
            layer = self._model.add_image(data, **kwargs)
            self._park_new_axes(ndim_before)

            # The slider must span the DATA's range, not the window we seeded, or the user cannot
            # open the window back up past our choice.
            #
            # `range_for`, not `dtype_range`: MONO12 and MONO16 share the uint16 container, so
            # the dtype answers 65535 for both and a 12-bit acquisition gets a slider whose
            # useful travel is the bottom sixteenth. `_bitdepth` measures which it actually is.
            #
            # UNCONDITIONAL. This used to require `"contrast_limits" in kwargs`, which meant a
            # layer whose seed was degenerate (a blank channel) or absent got NO range at all and
            # was left pinned to whatever extent napari inferred from its own sample -- exactly
            # the clipped slider this change exists to remove.
            try:
                dt = getattr(_first_level(data, bool(multiscale)), "dtype", None)
                lo_r, hi_r = _bitdepth.range_for(dt)
                lo_w, hi_w = kwargs.get("contrast_limits", (lo_r, hi_r))
                # Never narrower than what is displayed, or napari clamps the window itself.
                layer.contrast_limits_range = (min(lo_r, float(lo_w)), max(hi_r, float(hi_w)))
            except Exception:               # noqa: BLE001 - cosmetic; the layer is already good
                pass

            if bbox_um is not None and not placed:
                shape = tuple(_first_level_shape(data, bool(multiscale)))[-2:]
                self._place(layer, bbox_um, shape, z_scale_um)
            elif placed:
                self._label_units(layer)

            self._register_channel(key.channel, layer)
            # add_image does NOT move the camera; reset only while this is the FIRST layer,
            # so a later channel does not yank the view back while the user is panning.
            try:
                if len(self.ours()) <= 1:
                    self._model.reset_view()
            except Exception:                    # noqa: BLE001 - view convenience, never fatal
                pass
        # A layer that ARRIVES lit is the same gesture as one the user lights. Outside
        # programmatic() deliberately: the peer really did go dark and the plate must be told.
        if visible:
            self._darken_other_ops(key.channel, layer)
        self._present_z_axis()
        return layer

    def _park_new_axes(self, ndim_before: int) -> None:
        """Park an axis this pane did not have until now on its opening plane, not on index 0."""
        from squidxplorer._contrast import opening_z

        dims = getattr(self._model, "dims", None)
        if dims is None:
            return
        try:
            ndim = int(getattr(dims, "ndim", 0) or 0)
            if ndim <= int(ndim_before):
                return                           # nothing was prepended: nothing to park
            step = list(dims.current_step)
            moved = False
            for axis in range(ndim - int(ndim_before)):   # napari prepends, so the new ones lead
                lo, hi, pitch = (float(v) for v in tuple(dims.range[axis])[:3])
                if pitch <= 0:
                    continue
                want = opening_z(int(round((hi - lo) / pitch)) + 1)
                if int(step[axis]) != want:
                    step[axis] = want
                    moved = True
            if moved:
                dims.current_step = tuple(step)
        except Exception as exc:                 # noqa: BLE001 - presentation, never fatal
            log.warning("could not park the new display axis: %s: %s", type(exc).__name__, exc)

    def reset_view(self) -> None:
        """Point the camera at everything on screen. OUR write, never a user gesture."""
        try:
            self._model.reset_view()
        except Exception as exc:                 # noqa: BLE001 - a camera move, never fatal
            log.warning("could not fit the view to the new layers: %s: %s",
                        type(exc).__name__, exc)

    def frame_bbox_um(self, bbox_um: Sequence[float], *,
                      margin: float = FRAME_MARGIN) -> Optional[str]:
        """Point the camera at ONE stage-micrometre box. OUR write, never a user gesture.

        :meth:`reset_view` frames everything on screen; this frames a box the layers do not
        describe -- one FOV of a mosaic that is entirely resident. That is what makes stepping
        through a region's fields free: no read, no layer, only the camera.

        Returns ``None`` when the camera moved, or a SENTENCE naming why it did not -- never a
        silent no-op. A control whose entire job is to move the picture, and which does nothing
        without saying so, is the dead-control failure ``AxisPlayback.play`` is written against.

        Inside :meth:`programmatic` for the same reason ``reset_view`` and ``add_mosaic``'s first
        reset are: the plate is a SINK of this window's napari, and a camera move of ours must not
        be read back as the user having done something.

        2-D ONLY, by refusal. In 3-D the visible extent depends on ``camera.angles``, and napari
        solves that with ``_calculate_bounding_box``; copying that here would be a second rule for
        a mode this serves no purpose in. ``camera.angles`` is likewise never touched --
        ``reset_view`` resets them because it IS a reset, but a per-step framing that spun a
        user's rotation back would be taking something that belongs to them.
        """
        model = self._model
        try:
            if int(getattr(model.dims, "ndisplay", 2)) != 2:
                return ("the camera can only be framed on a box in 2D; switch back from 3D to "
                        "step through fields.")
            canvas = getattr(model, "_canvas_size", None)
            if canvas is None:
                # A named failure, not a guessed (800, 600): framing every camera against a canvas
                # napari no longer reports would be wrong by the aspect ratio, silently.
                raise NapariBindingError(
                    "napari.components.ViewerModel._canvas_size is gone, so this app cannot tell "
                    "how big the canvas is and cannot frame a box on it.")
            centre, zoom = camera_for_bbox_um(bbox_um, canvas, margin=margin)
            with self.programmatic():
                model.camera.center = centre     # napari front-fills a 2-tuple to (0, y, x)
                model.camera.zoom = zoom
            return None
        except NapariBindingError:
            raise
        except Exception as exc:                 # noqa: BLE001 - named to the caller, never fatal
            return f"could not frame the view ({type(exc).__name__}: {exc})."

    def _reuse_layer(self, layer: Any, data: Any, *, bbox_um, z_scale_um, multiscale, visible):
        """Point an existing layer at new pixels, keeping everything the user owns."""
        with self.programmatic():
            # Restore any stashed z stack FIRST; `_present_z_axis` re-decides after the new
            # data lands.
            self._restore_layer_z(layer)
            layer.data = data
            # The new region may be brighter than the one this layer was built for (C3 at 3437
            # replaced by E7 at 16380 in the 14-bit set), and a slider still bounded by the old
            # region's ceiling cannot reach the new pixels. Widening only ever opens it further,
            # and the layer's own current window is the floor, so the VALUE cannot move.
            try:
                dt = getattr(_first_level(data, bool(multiscale)), "dtype", None)
                lo_r, hi_r = _bitdepth.range_for(dt)
                lo_w, hi_w = (float(v) for v in layer.contrast_limits)
                self._widen_range(layer, min(lo_r, lo_w), max(hi_r, hi_w))
            except Exception:                    # noqa: BLE001 - cosmetic; the data already landed
                pass
            if visible is not None:
                layer.visible = bool(visible)
            if bbox_um is not None:
                shape = tuple(_first_level_shape(data, bool(multiscale)))[-2:]
                self._place(layer, bbox_um, shape, z_scale_um)
        return layer

    def _place(self, layer: Any, bbox_um: Sequence[float], shape: Sequence[int],
               z_scale_um: Optional[float] = None) -> None:
        """Put *layer* at its stage-micrometre footprint. THE one placement rule, shared."""
        scale, translate = placement_for(
            int(getattr(layer, "ndim", len(shape))), bbox_um, shape, z_scale_um)
        layer.scale = scale
        layer.translate = translate
        self._label_units(layer)

    @staticmethod
    def _label_units(layer: Any) -> None:
        """Micrometres on every axis, so napari's scale bar reads the layer's units."""
        try:
            layer.units = ("um",) * int(getattr(layer, "ndim", 2))
        except Exception:                # noqa: BLE001 - cosmetic; the scale is already right
            pass

    # Labels/Points results deliberately skip _register_channel: contrast is linked per
    # channel and these layer types have no `contrast_limits` at all.

    def _add_result(self, adder: str, op: str, channel: str, data: Any,
                    kwargs: dict, bbox_um: Optional[Sequence[float]],
                    shape: Optional[Sequence[int]]) -> Any:
        """Shared body of :meth:`add_labels` / :meth:`add_points`."""
        key = MosaicKey(str(op), str(channel))
        if self.find(key.op, key.channel) is not None:
            self.remove_op_channel(key.op, key.channel)   # a re-run REPLACES, never stacks up

        kwargs = dict(kwargs)
        kwargs["name"] = key.label()
        kwargs["metadata"] = key.as_metadata()

        with self.programmatic():
            layer = getattr(self._model, adder)(data, **kwargs)
            if bbox_um is not None:
                if shape is None:
                    raise ValueError(
                        f"{adder} for {key.label()!r} was given bbox_um but no shape. A Points "
                        "layer carries no array shape, so the micrometres-per-pixel scale cannot "
                        "be derived from the data; pass shape=<the mask's (h, w)>. Leaving it "
                        "unplaced would silently park every centroid at the world origin."
                    )
                self._place(layer, bbox_um, tuple(shape)[-2:])
        return layer

    def add_labels(self, op: str, channel: str, data: Any, *,
                   bbox_um: Optional[Sequence[float]] = None, visible: bool = True,
                   opacity: float = 0.5, blending: str = "translucent") -> Any:
        """Add (or replace) a segmentation mask as a napari ``Labels`` layer."""
        return self._add_result(
            "add_labels", op, channel, data,
            {"visible": visible, "opacity": float(opacity), "blending": blending},
            bbox_um, getattr(data, "shape", None),
        )

    def add_points(self, op: str, channel: str, data: Any, *,
                   bbox_um: Optional[Sequence[float]] = None,
                   shape: Optional[Sequence[int]] = None, visible: bool = True,
                   size: float = 12.0, symbol: str = "ring",
                   face_color: str = "transparent", border_color: str = "yellow",
                   features: Optional[Any] = None) -> Any:
        """Add (or replace) detection centroids as a napari ``Points`` layer; (N, 2) in (row, col)."""
        kwargs: dict[str, Any] = {
            "visible": visible, "size": float(size), "symbol": symbol,
            "face_color": face_color, "border_color": border_color,
        }
        if features is not None:
            kwargs["features"] = features
        return self._add_result("add_points", op, channel, data, kwargs, bbox_um, shape)

    # The delivery seam: one operator result -> the layer type its declaration names.
    def _add_intensity(self, op, channel, data, *, colormap, bbox_um, visible, z_scale_um,
                       multiscale, contrast_limits):
        """The pixels measure light: an Image layer, windowed, colormapped, blended additively."""
        return self.add_mosaic(op, channel, data, colormap=colormap, bbox_um=bbox_um,
                               visible=visible, z_scale_um=z_scale_um, multiscale=multiscale,
                               contrast_limits=contrast_limits)

    def _add_label_result(self, op, channel, data, *, colormap, bbox_um, visible, z_scale_um,
                          multiscale, contrast_limits):
        """The pixels are object ids: a Labels layer. The dropped arguments are meaningless for labels."""
        import numpy as _np

        arr = _np.asarray(data)
        if not (_np.issubdtype(arr.dtype, _np.integer) or arr.dtype == bool):
            raise ValueError(
                f"operator {op!r} declares produces='labels' but its {channel} pixels are "
                f"{arr.dtype}. A label image is integer object ids; napari's Labels layer rejects "
                "floats, and a float 'label' cannot be picked or counted.")
        return self.add_labels(op, channel, arr, bbox_um=bbox_um, visible=visible)

    #: result kind -> the adapter that turns it into a layer. Extended by adding a row.
    _RESULT_ADDERS: dict = {
        "intensity": _add_intensity,
        "labels": _add_label_result,
    }

    def add_result(self, kind: str, op: str, channel: str, data: Any, *,
                   colormap: Optional[Any] = None, bbox_um: Optional[Sequence[float]] = None,
                   visible: bool = True, z_scale_um: Optional[float] = None,
                   multiscale: Optional[bool] = None,
                   contrast_limits: Optional[tuple[float, float]] = None) -> Any:
        """Add one operator result as the layer type its *kind* (the ``produces`` declaration) names."""
        adder = self._RESULT_ADDERS.get(str(kind))
        if adder is None:
            raise ValueError(
                f"operator {op!r} declares result kind {kind!r}, which this viewer cannot draw; "
                f"it knows {sorted(self._RESULT_ADDERS)}.")
        return adder(self, op, channel, data, colormap=colormap, bbox_um=bbox_um, visible=visible,
                     z_scale_um=z_scale_um, multiscale=multiscale,
                     contrast_limits=contrast_limits)

    def _register_channel(self, channel: str, layer: Any) -> None:
        peers = self._by_channel.setdefault(channel, [])
        peers.append(layer)
        # Mirror FIRST: the identity must agree with itself before any tap is told about it.
        self._connect_identity_mirror(layer)
        # Connections are made HERE, per layer, because layer objects are destroyed and
        # recreated on a region change and a subscription made elsewhere goes deaf after one.
        self._connect_user_contrast(channel, layer)
        self._connect_user_visibility(channel, layer)
        self._connect_user_colormap(channel, layer)
        key = key_of(layer)
        if key is not None:
            self._connect_user_op(key.op, layer)
        # LAST, and the order is load-bearing: a subscriber hears "mip came on" before
        # "raw went off", which is the order the gesture actually happened in.
        self._connect_exclusive_op(channel, layer)
        # After the mirror on purpose: when a representative goes dark, its siblings are
        # already dark by the time this decides where the selection can honestly go.
        self._connect_selection_follow(layer)
        # Link contrast across processing layers of this channel. link_layers connects
        # events only; it does not equalise values at link time (deliberate).
        linkable = self._link_set(channel)
        if len(linkable) > 1:
            self._model.layers.link_layers(linkable, ("contrast_limits",))

    # `match_contrast_to` (raw -> operator layers) was shelved whole with its button
    # (Julio, 2026-08-19: "Shelf the match layers to raw").

    def remove_op_channel(self, op: str, channel: str) -> bool:
        """Remove an identity: every layer rendering ``(op, channel)``, not just the first."""
        holders = self.layers_for(op, channel)
        if not holders:
            return False
        for layer in holders:
            self.drop_layer(layer)
        return True

    def remove_op(self, op: str) -> list[str]:
        gone = []
        for channel in list(self.channels(op)):
            if self.remove_op_channel(op, channel):
                gone.append(channel)
        return gone

    def show_op(self, op: str) -> list[str]:
        """Make exactly one processing layer visible. Returns the channels now showing."""
        if op not in self.ops():
            raise KeyError(f"no processing layer named {op!r}; have {self.ops()!r}")
        for ly in self.ours():
            k = key_of(ly)
            assert k is not None
            ly.visible = k.op == op
        self._present_z_axis()
        return self.channels(op)

    def visible_op(self) -> Optional[str]:
        for ly in self.ours():
            if ly.visible:
                k = key_of(ly)
                assert k is not None
                return k.op
        return None

    def set_channel_visible(self, channel: str, visible: bool) -> None:
        """Show/hide one channel across the visible processing layer only."""
        current = self.visible_op()
        if current is None:
            return
        for ly in self.group(current):
            k = key_of(ly)
            assert k is not None
            if k.channel == channel:
                ly.visible = bool(visible)

    def contrast(self, channel: str) -> Optional[tuple[float, float]]:
        peers = self._by_channel.get(channel) or []
        if not peers:
            return None
        lo, hi = peers[0].contrast_limits
        return float(lo), float(hi)

    def set_contrast(self, channel: str, lo: float, hi: float) -> None:
        peers = self._by_channel.get(channel) or []
        if not peers:
            raise KeyError(f"no layer for channel {channel!r}")
        # Linked, so writing one writes them all; write the first and let napari propagate.
        peers[0].contrast_limits = (float(lo), float(hi))

    def _connect_user_contrast(self, channel: str, layer: Any) -> None:
        """Wire one layer into *channel*'s user-contrast fan-out; link echoes are collapsed by value."""
        def _fire(event=None, _ch=channel):
            peers = self._by_channel.get(_ch) or []
            if not peers:
                return
            lo, hi = float(peers[0].contrast_limits[0]), float(peers[0].contrast_limits[1])
            if self._last_seen.get(_ch) == (lo, hi):
                return                      # a link echo of a value already accounted for
            self._last_seen[_ch] = (lo, hi)
            if self.is_programmatic:
                return                      # OUR write: recorded, never reported as a gesture
            for cb in list(self._user_contrast_cbs):
                cb(_ch, lo, hi)

        layer.events.contrast_limits.connect(_fire)

    def _connect_user_colormap(self, channel: str, layer: Any) -> None:
        """Wire one layer's colormap into *channel*'s colour fan-out; an RGB triple travels, never a Colormap."""
        def _fire(event=None, _ch=channel):
            peers = self._by_channel.get(_ch) or []
            if not peers:
                return
            rgb = _colormap_rgb(peers[0])
            if rgb is None or self._last_colormap.get(_ch) == rgb:
                return
            self._last_colormap[_ch] = rgb
            if self.is_programmatic:
                return                      # OUR write: recorded, never reported as a gesture
            for cb in list(self._user_colormap_cbs):
                cb(_ch, rgb)

        layer.events.colormap.connect(_fire)

    def on_user_colormap(self, callback) -> None:
        """Subscribe to colormap changes the USER made. ``callback(channel, (r, g, b))``."""
        self._user_colormap_cbs.append(callback)

    def channel_rgb(self, channel: str) -> Optional[tuple]:
        """The RGB the canvas is tinting *channel* with right now, or None if it has no layers."""
        peers = self._by_channel.get(channel) or []
        return _colormap_rgb(peers[0]) if peers else None

    def _connect_user_visibility(self, channel: str, layer: Any) -> None:
        """Wire one layer's eye icon into *channel*'s visibility fan-out (visible = ANY layer showing it)."""
        def _fire(event=None, _ch=channel):
            peers = self._by_channel.get(_ch) or []
            if not peers:
                return
            on = any(bool(getattr(p, "visible", False)) for p in peers)
            if self._last_visible.get(_ch) == on:
                return                      # an echo, or a peer flip that did not change the answer
            self._last_visible[_ch] = on
            if self.is_programmatic:
                return                      # OUR write: recorded, never reported as a gesture
            for cb in list(self._user_visibility_cbs):
                cb(_ch, on)

        # Seeded at registration: an unseeded latch always delivers its first event even
        # when the answer did not move.
        self._last_visible[channel] = any(
            bool(getattr(p, "visible", False)) for p in (self._by_channel.get(channel) or []))
        layer.events.visible.connect(_fire)

    def on_user_visibility(self, callback) -> None:
        """Subscribe to channel visibility the USER changed. ``callback(channel, visible)``."""
        self._user_visibility_cbs.append(callback)

    def _connect_user_op(self, op: str, layer: Any) -> None:
        """Wire one layer's visibility into the per-op fan-out: "is this processing layer on screen"."""
        def _fire(event=None, _op=str(op)):
            on = any(bool(getattr(ly, "visible", False)) for ly in self.group(_op))
            if self._last_op_visible.get(_op) == on:
                return                      # a sibling channel of the same group; already told
            self._last_op_visible[_op] = on
            if self.is_programmatic:
                return                      # OUR write: recorded, never reported as a gesture
            for cb in list(self._user_op_cbs):
                cb(_op, on)

        # Seed the latch so the op's first event is collapsed like any other echo.
        self._last_op_visible[str(op)] = any(
            bool(getattr(ly, "visible", False)) for ly in self.group(str(op)))
        layer.events.visible.connect(_fire)

    def _darken_other_ops(self, channel: str, keep: Any) -> list[str]:
        """Make *keep* the only lit mosaic for *channel*. Returns the ops darkened."""
        key = key_of(keep)
        if key is None:
            return []
        gone: list[str] = []
        for peer in list(self._by_channel.get(channel) or []):
            if peer is keep or not bool(getattr(peer, "visible", False)):
                continue
            pk = key_of(peer)
            if pk is None or pk.op == key.op:
                continue
            peer.visible = False
            gone.append(pk.op)
        return gone

    def _connect_exclusive_op(self, channel: str, layer: Any) -> None:
        """At most ONE operator's mosaic per channel is lit at a time (mosaics are additive)."""
        def _fire(event=None, _ch=channel, _ly=layer):
            if self.is_programmatic:
                return
            if bool(getattr(_ly, "visible", False)):
                self._darken_other_ops(_ch, _ly)
            # Both directions re-decide the z axis: a layer going dark changes what is on screen.
            self._present_z_axis()

        layer.events.visible.connect(_fire)

    def _connect_selection_follow(self, layer: Any) -> None:
        """Wire one layer into the selection-follows-visibility rule (ticket #8).

        napari's layer-controls panel shows the SELECTED layer and napari never moves the
        selection on a visibility flip, so unticking a layer left its controls up over a
        picture it is no longer part of.
        """
        def _fire(event=None, _ly=layer) -> None:
            self._follow_selection_off(_ly)

        try:
            layer.events.visible.connect(_fire)
        except Exception:                        # noqa: BLE001 - a stub layer with no emitter
            pass

    def _follow_selection_off(self, layer: Any) -> None:
        """A hidden layer may not keep the selection: move it to the topmost VISIBLE layer of
        the same op, else any visible layer. With nothing visible the selection stays put —
        there is nowhere honest to move it."""
        if self._selection_following:
            return                               # our own selection write echoing back
        if bool(getattr(layer, "visible", False)):
            return                               # only a flip to OFF vacates the selection
        try:
            selection = self._model.layers.selection
        except Exception:                        # noqa: BLE001 - a model without a selection
            return
        if layer not in selection:
            return
        candidates = [ly for ly in reversed(list(self._model.layers))
                      if bool(getattr(ly, "visible", False))]
        if not candidates:
            return
        key = key_of(layer)
        target = None
        if key is not None:
            target = next((ly for ly in candidates
                           if (k := key_of(ly)) is not None and k.op == key.op), None)
        if target is None:
            target = next((ly for ly in candidates if key_of(ly) is not None), candidates[0])
        self._selection_following = True
        try:
            selection.active = target
        except Exception as exc:                 # noqa: BLE001 - selection is a convenience
            log.warning("could not move the layer selection off a hidden layer: %s", exc)
        finally:
            self._selection_following = False

    def on_user_op(self, callback) -> None:
        """Subscribe to the processing layer the user showed or hid. ``callback(op, visible)``."""
        self._user_op_cbs.append(callback)

    def channel_visible(self, channel: str) -> Optional[bool]:
        """Is this channel on screen anywhere? None when the channel has no layers."""
        peers = self._by_channel.get(channel) or []
        if not peers:
            return None
        return any(bool(getattr(p, "visible", False)) for p in peers)

    def on_user_contrast(self, callback) -> None:
        """Subscribe to contrast changes the USER made. ``callback(channel, lo, hi)``."""
        self._user_contrast_cbs.append(callback)

    def on_contrast_changed(self, callback) -> None:
        """Subscribe to contrast changes via napari's public event."""
        for peers in self._by_channel.values():
            if peers:
                peers[0].events.contrast_limits.connect(callback)


def _auto_window_for(data: Any, multiscale: bool) -> Optional[tuple[float, float]]:
    """The seed contrast window for *data*, or None to let napari autoscale."""
    from squidxplorer._contrast import auto_contrast, sample_plane

    try:
        levels = data if multiscale else [data]
        plane = sample_plane(levels)
        return None if plane is None else auto_contrast(plane)
    except Exception:                       # noqa: BLE001 - seeding is cosmetic, never fatal
        return None


def _first_level(data: Any, multiscale: bool) -> Any:
    """The full-resolution array, whether or not ``data`` is a pyramid."""
    return data[0] if multiscale else data


def _first_level_shape(data: Any, multiscale: bool) -> Sequence[int]:
    """Shape of the full-resolution plane, whether or not ``data`` is a pyramid."""
    return _first_level(data, multiscale).shape


# `layer.data` for a multiscale layer is napari's `MultiScaleData`: a Sequence that is
# neither list nor tuple, reports level 0's shape/ndim as its own, and whose __array__ is
# the COARSEST level. Every reader of a layer's data goes through these two helpers.


def pyramid_levels(data: Any) -> Optional[list]:
    """The levels of *data* when it is a multiscale pyramid, else None. Highest resolution first."""
    if isinstance(data, (str, bytes)) or not isinstance(data, _SequenceABC):
        return None
    if len(data) == 0:
        raise ValueError("the layer holds an EMPTY multiscale pyramid — nothing to read.")
    return list(data) if int(getattr(data[0], "ndim", 0)) >= 2 else None


def full_res_level(data: Any) -> Any:
    """The full-resolution array behind a napari layer's ``data``, pyramid or not. Never materialises."""
    levels = pyramid_levels(data)
    return data if levels is None else levels[0]


# --------------------------------------------------------------------------------------
# The embedded pane
# --------------------------------------------------------------------------------------


def build_pane(parent: Any = None) -> tuple[Any, MosaicLayers, Any]:
    """Build a real napari Viewer and hand back ``(qt_viewer, MosaicLayers, viewer)``."""
    verify_napari_bindings()

    import napari

    # show=False: no top-level window and no second event loop; the host QApplication drives it.
    viewer = napari.Viewer(show=False)
    enable_scale_bar(viewer)
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    if parent is not None and qt_viewer is not None:
        qt_viewer.setParent(parent)
    return qt_viewer, MosaicLayers(viewer), viewer


def enable_scale_bar(viewer: Any, unit: str = "um") -> None:
    """Turn on napari's built-in scale bar, in micrometres, for the mosaic view."""
    sb = getattr(viewer, "scale_bar", None)
    if sb is None:
        return
    sb.visible = True
    sb.colored = False          # follow the theme foreground, like the rest of napari's chrome
    try:
        sb.position = "bottom_right"
    except Exception:           # noqa: BLE001 - position is cosmetic
        pass
    try:
        sb.unit = unit          # deprecated in napari >=0.7 (layer.units wins there); harmless now
    except Exception:           # noqa: BLE001 - unit label is cosmetic; the bar still shows
        pass
