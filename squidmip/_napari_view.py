"""napari mosaic view — the processing-layer/channel hierarchy, behind a flag.

WHY THIS EXISTS
---------------
ndviewer_light renders one plane at a time. Our mosaics are multiscale pyramids, and the
reason to move to napari is that it renders pyramids natively. Two measurements gated this
module (see ``docs/napari-gate.md``): napari is *faster* than ndv per warm tile
(16.7 ms vs 26.5 ms on identical 512² tiles, identical checksum), and clipped pan over a
16384² lazy pyramid costs 22.6 ms median / 29.8 ms p90 while RSS grows 52 MB against a
537 MB level — i.e. issue #1942's "multiscale zarrs go slow when clipped" does NOT
reproduce, because napari fetches the clipped region rather than materialising the level.

THE EMBEDDING PATH IS PUBLIC
----------------------------
Earlier spikes drove napari through ``viewer.window._qt_viewer``. That is private, and
``Window.qt_viewer`` is public but raises a FutureWarning describing itself as an
"implementation detail" to be removed in >= 0.9.0. Neither is a foundation, and this project
has already lost a day to a private binding that bound cleanly and did nothing
(``_voxel_scale``, swallowed by ``except AttributeError: pass`` because vispy had frozen the
Visual).

The supported path is:

    ViewerModel()            # napari.components.ViewerModel, in components.__all__
    QtViewer(model)          # napari.qt.QtViewer, in napari.qt.__all__

``QtViewer.__init__`` is annotated ``viewer: ViewerModel``, so this is the intended
construction, not a lucky accident. Verified present and identical on napari 0.6.6 (the
version installed here) AND 0.8.0.

Building the canvas this way means there is no napari ``Window`` at all, so the menu bar,
the dock widgets and the plugin surface are never constructed — measured: 0 menu items,
0 dock widgets, no layer-controls container. That is a structural answer to "watch out for
feature bloat", not chrome hidden after the fact.

THE LAYER HIERARCHY
-------------------
Julio's model is two levels deep::

    PROCESSING LAYER   (raw | stitched | deconvolved | background-subtracted | ...)
      -> CHANNELS      (405, 488, 561, 638 ...)
         -> CONTRAST   per channel

**napari has no layer groups.** ``LayerGroup``/``GroupLayer`` appear nowhere in the package
and ``LayerList`` is flat. The hierarchy is therefore built here, out of three public pieces:

* **Group identity lives in ``layer.metadata``**, never parsed back out of ``layer.name``.
  ian-stitcher recovers the wavelength with ``extractWavelength(layer.name)``, and that class
  of bug has already bitten this codebase twice: petakit's OME-TIFF reader emits channel names
  its own ``wavelength_from_channel`` regex cannot parse, and 3f1bf3f fixed Squid's
  ``Fluorescence_488_nm_Ex`` failing a parser that wanted ``\\s*nm``. The name is a human
  label; the metadata is the truth.
* **A processing-layer toggle is a visibility flip over one group** — the before/after
  stitching toggle.
* **Per-channel contrast is shared across processing layers via ``LayerList.link_layers``**,
  keyed on CHANNEL. This is what makes contrast survive the before->after toggle, and it means
  there is exactly ONE contrast value per channel in the whole application. That is a
  structural answer to "make sure there's no knowledge duplication in the GUI — I can still see
  the duplicated sliders": a second slider for the same channel cannot disagree with the first,
  because they are the same linked property.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
It does not compute contrast windows. ``_viewer._pct_window`` already owns that rule,
including the deliberate choice NOT to widen a degenerate window to ``(lo, lo + 1)`` — which
would clip a blank channel to full white so it reads as signal. Re-deriving it here is exactly
the duplication we are trying to delete, so callers pass ``contrast_limits`` in.

It does not own channel colours either; ``_channels.CHANNEL_COLORS_MAP`` is Squid's
authoritative palette and is resolved through ``_channels`` rather than restated.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence as _SequenceABC
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from squidmip._logpane import get_logger

log = get_logger("napari")

#: Fallback GPU 3D texture cap (Apple GPUs report GL_MAX_3D_TEXTURE_SIZE = 2048). The live value
#: is read off the canvas at runtime; this is only used until that is known.
_DEFAULT_MAX_3D_TEXTURE = 2048

#: Key on a layer's own ``metadata`` where its z stack waits while a z-REDUCED result is the layer
#: on screen. See ``MosaicLayers._present_z_axis``; the same stash-on-the-layer mechanism
#: ``_swap_layer_scale`` uses for the 2D/3D swap, and for the same reason -- a layer that is never
#: destroyed cannot strand the contrast, visibility and colormap subscribers bound to it.
_Z_STASH = "_zstack"

# NOTE: napari is NOT imported at module scope. It costs ~88 ms and pulls Qt, and the pure
# hierarchy logic below must stay importable (and testable) in a headless process with no
# napari installed at all. Every napari touch is inside a function.

VIEWER_ENV = "SQUIDMIP_VIEWER"
_NAPARI = "napari"
META_KEY = "squidmip"

#: Spellings that used to select the ndviewer_light fallback. Kept ONLY to recognise them and
#: say what happened, so an old launcher, script or shell profile that still exports one gets a
#: sentence instead of a viewer that quietly is not the one it asked for.
_RETIRED_NDV_NAMES = ("ndv", "ndviewer", "ndviewer_light")


def resolve_viewer(env: Optional[dict] = None) -> str:
    """Which viewer to build. There is exactly one: ``"napari"``.

    THE single place this is decided. ``_napari_pane.make_pane`` asks this rather than parsing
    the variable itself — two readers of one environment variable is exactly the knowledge
    duplication that produces controls disagreeing about what is on screen.

    **The ndviewer_light fallback is deleted** (Julio, 2026-07-30; foreshadowed in
    ``_mosaic_source.py``: "I don't want to have an ndviewer light fall back", and in
    ``docs/SCOPE.md``: "a half-alive second viewer stack. Delete, do not maintain."). Two
    reasons, and the second is the one that forced it:

    * napari is the renderer and won a written gate (``docs/napari-gate.md``), so the second
      stack was carrying no load.
    * ``ndviewer_light.core`` imports PyQt5 at module scope. Under Qt6 that drags PyQt5 into the
      process, vispy then refuses to load PyQt6 beside it, and the process aborts. A fallback
      that cannot coexist with the renderer it is a fallback FOR is not a safety net.

    This function is kept, rather than deleted along with the branch, because it is still the one
    place the viewer choice is named, and because callers must keep asking exactly one thing.
    """
    src = os.environ if env is None else env
    want = str(src.get(VIEWER_ENV, "")).strip().lower()
    if want in _RETIRED_NDV_NAMES:
        # Said out loud, never silently ignored: the user asked for a specific viewer and is not
        # getting it. NO FALLBACKS means the substitution is announced, not hidden.
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
# Binding assertions
# --------------------------------------------------------------------------------------
# Everything this module uses is public, but "public" is not "permanent" — napari renamed
# and deprecated the Qt access path twice between 0.5 and 0.8. These assertions turn a napari
# upgrade that moves one of them into a loud, named failure at construction time instead of a
# viewer that silently renders nothing. They are mutation-tested (test_napari_view.py proves
# the check bites when a symbol is renamed); an assertion nobody has watched fail is only a
# comment.

REQUIRED_NAPARI_BINDINGS: tuple[tuple[str, str], ...] = (
    ("napari.components", "ViewerModel"),
    ("napari.components", "LayerList"),
    ("napari.qt", "QtViewer"),
)

#: PRIVATE napari symbols we depend on, checked separately because they carry no ``__all__``
#: promise at all. There is exactly one, and it is deliberate: ``QtLayerControlsContainer`` is
#: napari's REAL per-channel contrast surface (range slider, auto-scale buttons, colormap combo).
#: Julio's instruction is to use napari's own controls rather than rebuild them, and rebuilding
#: them is what produced the duplicated sliders in the first place. napari does not export this
#: widget publicly, so the choice is: use the private symbol behind a guard that fails loudly on
#: upgrade, or reimplement the control surface and reintroduce the duplication. The guard is the
#: lesser evil, and it is mutation-tested.
REQUIRED_PRIVATE_BINDINGS: tuple[tuple[str, str], ...] = (
    ("napari._qt.layer_controls", "QtLayerControlsContainer"),
)

# Attributes we drive on a layer / model. Same reasoning.
REQUIRED_LAYER_ATTRS: tuple[str, ...] = ("metadata", "visible", "contrast_limits", "scale",
                                         "translate", "name", "events")
REQUIRED_LAYERLIST_ATTRS: tuple[str, ...] = ("link_layers", "unlink_layers")


class NapariBindingError(RuntimeError):
    """A napari symbol this module depends on has moved, been renamed, or been removed."""


def verify_napari_bindings(modules: Optional[dict] = None) -> None:
    """Fail loudly if any napari API this module drives is missing.

    ``modules`` is an injection seam for the mutation test: it maps a dotted module name to an
    object to inspect instead of importing. Production passes nothing.
    """
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

    # Private symbols: existence only. There is no __all__ to check, which is precisely why
    # these are listed separately rather than quietly mixed in with the supported ones.
    for dotted, attr in REQUIRED_PRIVATE_BINDINGS:
        try:
            mod = modules[dotted] if modules and dotted in modules else importlib.import_module(dotted)
        except Exception as exc:  # pragma: no cover
            missing.append(f"{dotted} (PRIVATE; import failed: {exc!r})")
            continue
        if not hasattr(mod, attr):
            missing.append(f"{dotted}.{attr} (PRIVATE)")

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
    """Identity of one displayed mosaic: which processing layer, which channel.

    The unit displayed is always an assembled MOSAIC, never a single FOV.
    """

    op: str
    channel: str

    def label(self) -> str:
        """Human label for the napari layers list. NOT parsed back — see module docstring."""
        return f"{self.op} · {self.channel}"

    def as_metadata(self) -> dict:
        return {META_KEY: {"op": self.op, "channel": self.channel}}


def key_of(layer: Any) -> Optional[MosaicKey]:
    """Recover a layer's identity from its METADATA. Returns None for foreign layers.

    Foreign layers (a user-added points layer, a plugin's output) are deliberately tolerated
    and ignored rather than crashing the group logic.
    """
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
    """Map ``_tiling``'s world box onto napari's per-layer placement.

    ``bbox_um`` is ``(x0, y0, x1, y1)`` in stage micrometres — X FIRST. napari's world axes for
    a 2D image are ``(row, col)`` = ``(y, x)`` — Y FIRST. The axis order flips, which is exactly
    the sort of silent transpose that produces a mosaic that looks plausible and is wrong, so it
    is done once, here, and pinned by a test.

    Both sides already speak stage micrometres, so there is no unit conversion — only the flip.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox_um)
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"bbox_um must satisfy x1 > x0 and y1 > y0, got {tuple(bbox_um)!r}")
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"shape must be positive, got {tuple(shape)!r}")
    scale = ((y1 - y0) / h, (x1 - x0) / w)
    translate = (y0, x0)
    return scale, translate


def placement_for(ndim: int, bbox_um: Sequence[float], shape: Sequence[int],
                  z_scale_um: Optional[float] = None) -> tuple[tuple, tuple]:
    """``(scale, translate)`` for a layer of *ndim* axes whose trailing two are ``(y, x)``.

    The arithmetic of :meth:`MosaicLayers._place`, lifted out so it can also be handed to
    ``add_image`` as keyword arguments. Two consumers, one rule: a layer that is placed by a
    SECOND copy of this is one edit away from sitting next to the mosaic it claims to describe.

    Why ``add_image`` needs it at all: assigning ``layer.scale`` AFTER the layer exists changes the
    world extent, which moves napari's dims range, which moves the z slider, which makes napari
    re-slice — a second whole-region fuse of a mosaic that was already on screen. Placing at
    construction is the same picture with one fetch instead of two.
    """
    scale, translate = scale_translate_from_bbox_um(bbox_um, shape)
    extra = max(0, int(ndim) - 2)
    lead = (float(z_scale_um) if (extra and z_scale_um) else 1.0,) * extra
    return lead + tuple(scale), (0.0,) * extra + tuple(translate)


def _colormap_rgb(layer: Any) -> Optional[tuple]:
    """The RGB a napari layer tints with, as three floats in 0..1.

    Read at FULL INTENSITY (the last stop of the colormap's lookup table), because that is the
    colour the canvas shows for a saturated pixel and therefore the tint the plate has to match.
    A colormap with no usable table (a napari version that reshapes it, a custom object) returns
    None and the caller leaves the plate's colour alone -- guessing a tint would silently
    recolour the plate to something that is on no screen.
    """
    cm = getattr(layer, "colormap", None)
    colors = getattr(cm, "colors", None)
    if colors is None:
        return None
    try:
        row = colors[-1]
        return (float(row[0]), float(row[1]), float(row[2]))
    except Exception:                       # noqa: BLE001 - unknown colormap shape; say nothing
        return None


#: How far a lookup-table row may sit off the black-to-hue line and still count as that hue, in
#: 0..1 colour units. 2/255 is one 8-bit step per component: below that, no screen and no exported
#: hex can tell the difference, so insisting on exactness would reject colormaps for rounding.
_HUE_TOL = 2.0 / 255.0


def colormap_hue_rgb(layer: Any) -> "Optional[tuple[int, int, int]]":
    """The ONE 8-bit RGB a layer's colormap reduces to, or ``None`` if it does not reduce.

    Some consumers store one colour per channel, not a ramp - Minerva's story ``groups`` carry a
    single six-digit ``"color"`` per channel (``_minerva.auto_groups``) and there is no second
    field to put a gradient in. A colormap is representable there only when it is a black-to-
    single-hue ramp: every row of its lookup table a non-negative multiple of the last row.

    That covers everything this app puts on a layer by itself - ``_napari_pane._colormap_for``
    builds exactly ``[[0,0,0,1], [r,g,b,1]]`` from Squid's palette - and napari's own ``red``,
    ``green``, ``blue``, ``cyan``, ``magenta``, ``yellow`` and ``gray``, all of which are two-stop
    black-to-hue maps.

    It does NOT cover a multi-stop map the user picked in napari's colormap combo (``viridis``,
    ``turbo``, ``inferno``, ...). There is no correct single colour for those: the last stop is
    the top of a ramp, not the map. **Returning that stop would emit a colour the user is not
    looking at**, so this returns ``None`` and lets the caller keep whatever it had - which for
    the Minerva export means the acquisition's ``display_color``, a colour that is at least
    honestly labelled as coming from the acquisition rather than from the screen.

    NOT A DUPLICATE OF :func:`_colormap_rgb` / :meth:`MosaicLayers.channel_rgb`, which sit right
    beside it and look like the same question. Those answer "what tint does the plate paint with",
    and for that the last stop is the right and sufficient answer: the plate composites a live
    canvas, and a slightly-off tint on a thumbnail is a cosmetic mismatch the user can see next to
    the real thing and ignore. This answers "what colour do we WRITE INTO A FILE that outlives the
    session", where the same approximation is a lie with no context beside it to correct it. Same
    lookup table, different tolerance for being wrong, so they are deliberately two functions.
    """
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
        # Project every row onto the last row and require it to land there. A ramp that dips or
        # swings off the line (any perceptual map) fails here and is reported as unrepresentable.
        t = np.clip((rgb @ last) / denom, 0.0, 1.0)[:, None]
        if float(np.max(np.abs(rgb - t * last))) > _HUE_TOL:
            return None
        return tuple(int(round(min(1.0, max(0.0, float(v))) * 255.0)) for v in last)
    except Exception:                       # noqa: BLE001 - unknown colormap shape; say nothing
        return None


class MosaicLayers:
    """The two-level hierarchy over a napari ``ViewerModel``.

    Wraps a ViewerModel rather than subclassing it: napari owns that model's lifecycle, and
    inheriting from a pydantic-evented model to add two dicts is how you acquire a base class
    you cannot upgrade.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        # channel -> the layers showing that channel, across every processing layer. Linked.
        self._by_channel: dict[str, list[Any]] = {}
        # Depth of "this write came from US, not the user". See `programmatic()`.
        self._programmatic = 0
        self._user_contrast_cbs: list[Any] = []
        #: Subscribers to the eye icons, so the plate can drop its own channel checkboxes.
        self._user_visibility_cbs: list[Any] = []
        #: channel -> last visibility REPORTED, so a peer flip that does not change the answer
        #: ("is this channel on screen at all") is not delivered as a user gesture.
        self._last_visible: dict[str, bool] = {}
        #: Subscribers to the PROCESSING LAYER the user is showing, so the plate can follow which
        #: operator this window is looking at. Separate from the visibility fan-out above on
        #: purpose -- see ``_connect_user_op``.
        self._user_op_cbs: list[Any] = []
        #: op -> last "is any layer of this processing layer visible" REPORTED, collapsing the one
        #: gesture that arrives once per channel of the group into one delivery.
        self._last_op_visible: dict[str, bool] = {}
        #: Subscribers to the LUT, so the plate tints a channel the way the canvas does.
        self._user_colormap_cbs: list[Any] = []
        #: channel -> last RGB reported, collapsing link/peer echoes the same way.
        self._last_colormap: dict[str, tuple] = {}
        # Last contrast value SEEN per channel, updated on every event including our own
        # programmatic writes. Linked layers propagate a write to their peers and each peer
        # then emits its own event, so one user drag arrives here once per layer showing the
        # channel. Those echoes carry an identical value, which is what distinguishes them
        # from a real gesture. Tracking programmatic writes here too matters: without it, a
        # user dragging BACK to a previously delivered value would be suppressed as an echo.
        self._last_seen: dict[str, tuple[float, float]] = {}
        #: GPU 3D texture cap, read off the live canvas by the pane; napari refuses to render a
        #: single 3D texture larger than this per axis, so the 3D swap targets the level that fills
        #: it rather than a bigger volume napari would silently downsample.
        self._max_3d_texture: int = _DEFAULT_MAX_3D_TEXTURE

    # -- who moved the contrast: us, or the user? ---------------------------------------
    @contextmanager
    def programmatic(self):
        """Mark contrast writes made BY US, so subscribers can ignore them.

        This distinction is the whole safety property of the contrast design, and it is not
        theoretical: the plate is a SINK, and when it wrote a viewer-originated autoscale back
        into its own policy state it latched all four channels to MANUAL on open. That killed
        per-region contrast dead from frame one while the plate still drew an amber "wells NOT
        comparable" badge — a badge that was therefore lying. A sink must never write back to
        the owner, and it can only obey that rule if it can tell who moved the value.

        Everything this class sets itself (the initial percentile window from ``_pct_window``,
        a re-add, a link propagation) happens inside this block. Only a genuine user drag on
        napari's slider arrives outside it.
        """
        self._programmatic += 1
        try:
            yield
        finally:
            self._programmatic -= 1

    @property
    def is_programmatic(self) -> bool:
        return self._programmatic > 0

    # -- introspection ------------------------------------------------------------------
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
        """ONE layer answering to ``(op, channel)`` — the first. See :meth:`find_all`."""
        for ly in self.ours():
            if key_of(ly) == MosaicKey(op, channel):
                return ly
        return None

    def find_all(self, op: str, channel: str) -> list[Any]:
        """EVERY layer answering to ``(op, channel)``: one for a mosaic, N for a 3-D volume.

        A key is not unique and is not meant to be. ``_brick_view.BrickedVolume`` stamps the SAME
        ``(op, channel)`` on every brick on purpose, so the tree groups a hundred textures into one
        row and one toggle drives the whole volume — and while 3D is up the pane's own 2-D layers
        surrender that identity, so the bricks are the only holders.

        :meth:`find` answers "a layer for this pair", which is what a thumbnail or a napari-controls
        selection wants. Anything that DRIVES the pair — a visibility checkbox, a contrast write —
        must reach all of them. Julio, 2026-08-06: *"Turning off one layer doesn't turn the other
        like in the 2D view."* The layer tree wrote ``find(...).visible``, so unchecking a channel
        hid ONE brick of a hundred and left the volume lit under a cleared checkbox.
        """
        key = MosaicKey(str(op), str(channel))
        return [ly for ly in self.ours() if key_of(ly) == key]

    # -- 2D pyramid <-> 3D full resolution ----------------------------------------------
    def render_max_res_3d(self, on: bool) -> None:
        """Swap our image mosaics between the fast multiscale pyramid (2D) and their FULL-RES
        single-scale volume (3D).

        napari does not support multiscale in 3D: the instant ``ndisplay`` flips to 3 it drops a
        multiscale layer to the COARSEST level unconditionally (``_scalar_field/_slice.py``), which
        is the blocky volume Julio screenshotted. We do not want that. When 3D is on we hand each
        layer its level-0 ``(Z, Y, X)`` array so napari renders the volume at the highest resolution it
        can hold; when it goes back to 2D we restore the pyramid so
        navigation stays fast. napari 0.6.6 allows the in-place swap (verified). Idempotent: a
        layer already in the requested form is skipped, so re-applying on a region change is safe.
        """
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
        # A layer parked in the z-COLLAPSED presentation (see `_present_z_axis`) has no volume to
        # render, so give it its stack back before either direction of the 3D swap. Two mechanisms
        # stash on the same layer; whichever runs second must not stash the other's placeholder.
        if _Z_STASH in meta:
            self._restore_layer_z(ly)
            meta = dict(getattr(ly, "metadata", None) or {})
        try:
            if full_res:
                # `pyramid_levels` and not `isinstance(ly.data, (list, tuple))`: napari hands a
                # multiscale layer's data back as its own `MultiScaleData`, so that check was
                # False for EVERY pyramid in the app and this swap returned at its first line —
                # a silent no-op, leaving napari to drop the layer to its coarsest level in 3D,
                # which is exactly the blocky volume this method exists to prevent.
                data = pyramid_levels(ly.data)
                if data is None:
                    return                       # already single-scale, nothing to swap
                meta["_pyramid"] = data          # stash the pyramid so 2D can restore it
                ly.metadata = meta
                # napari renders 3D from ONE GL texture and refuses any axis over
                # GL_MAX_3D_TEXTURE_SIZE (~2048 on Apple GPUs); handed a bigger volume it does its
                # OWN crude stride-downsample. So target the FINEST pyramid level that still fits
                # the texture: that fills the GPU budget = the max resolution napari can physically
                # show for the whole region. Native full res needs a CROP (one texture cannot
                # hold 5731 px), which is what an ROI child window is for. AGAVE is cancelled, see
                # docs/VERSIONS.md. Levels are finest-first; take the first that fits, else the
                # coarsest as a floor.
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

    # -- a z-REDUCED result is presented WITHOUT a z axis --------------------------------
    #
    # Julio, from the running GUI: "the MIP layer doesn't collapse the z-level axis inside napari".
    # docs/DESIGN.md:188 already states the rule -- "z slider hidden when nz is 1 (a z reduced
    # result)" -- and docs/rendering-contract.md:13 states the producer half of it: a fused level
    # is `(y, x)` rather than `(1, y, x)` when nz == 1, "so no singleton z-slider appears".
    #
    # WHAT WAS ACTUALLY WRONG, measured rather than assumed. The MIP layer is ALREADY 2-D by the
    # time it gets here: `project_well` returns (T, C, 1, Y, X), `_workers._on_well` takes
    # `image[0, :, 0]`, and the fused result plane that reaches `add_result` is (Y, X). Nothing
    # downstream keeps a singleton z. The slider belongs to the RAW layer sharing the pane --
    # `_mosaic_source.fuse_region_pyramid` hands napari `(nz, y, x)` levels -- and napari derives
    # `dims.ndim` from the MAXIMUM over EVERY layer, visible or not. So delivering the MIP darkens
    # raw (see `_darken_other_ops`) and leaves raw's z slider standing over a picture it cannot
    # move: a control that lies about what is on screen.
    #
    # So the rule is applied where "what is on screen" is decided, and it is read off the
    # OPERATOR'S OWN DECLARATION (`consumes`), never off its name: a z-reducer showing means no z
    # axis, anything else showing (raw, or a plane-op result that genuinely kept its depth) means
    # the axis comes straight back.
    #
    # The swap is `_swap_layer_scale`'s mechanism exactly -- stash on the layer's own metadata,
    # reassign `data` -- so no layer is ever destroyed and no subscriber is ever stranded, which is
    # the property `add_mosaic`'s reuse path exists to protect. It is also why the collapsed
    # presentation uses the COARSEST pyramid level: the layer it is applied to is by construction
    # the hidden one, and a full-resolution plane would be a whole-region fuse nobody looks at.

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
            # AFTER the data: napari pads scale/translate to the new ndim with 1.0 / 0.0, which
            # would silently render an anisotropic stack isotropically (the defect `_place`'s z
            # step exists to prevent). The saved values are the ones `_place` computed.
            ly.scale = stash["scale"]
            ly.translate = stash["translate"]
        except Exception as exc:                 # noqa: BLE001 - a presentation nicety, never fatal
            log.warning("z axis restore failed on %s: %s", getattr(ly, "name", "layer"), exc)

    def _collapse_layer_z(self, ly: Any, z: int) -> None:
        """Present a ``(z, y, x)`` layer as the single plane *z*, so it stops carrying a z axis."""
        meta = dict(getattr(ly, "metadata", None) or {})
        if _Z_STASH in meta:
            return                               # already collapsed; idempotent by design
        try:
            multiscale = bool(getattr(ly, "multiscale", False))
            data = ly.data
            # `list(...)` and not `isinstance(data, list)`: napari wraps a pyramid in its own
            # ``MultiScaleData``, which is a Sequence of levels but is neither a list nor a tuple,
            # and indexing it as if it were one array walks the LEVELS instead of the z planes.
            levels = list(data) if multiscale else [data]
            if not levels or min(int(getattr(lv, "ndim", 2)) for lv in levels) < 3:
                return                           # already a plane: nothing to collapse
            scale, translate = tuple(ly.scale), tuple(ly.translate)
            meta[_Z_STASH] = {"data": list(levels) if multiscale else data,
                              "multiscale": multiscale,
                              "scale": scale, "translate": translate}
            coarsest = levels[-1]
            plane = coarsest[max(0, min(int(z), int(coarsest.shape[0]) - 1))]
            ly.metadata = meta
            ly.multiscale = False
            ly.data = plane
            # The COARSEST level's own pixel size, not level 0's: `scale` placed the pyramid, whose
            # scale is level 0's, and this presentation is a different-sized array in the same box.
            fine_h, fine_w = int(levels[0].shape[-2]), int(levels[0].shape[-1])
            ly.scale = (scale[-2] * fine_h / float(plane.shape[-2]),
                        scale[-1] * fine_w / float(plane.shape[-1]))
            ly.translate = translate[-2:]
        except Exception as exc:                 # noqa: BLE001 - a presentation nicety, never fatal
            log.warning("z axis collapse failed on %s: %s", getattr(ly, "name", "layer"), exc)

    @staticmethod
    def _reduces_z(op: str) -> bool:
        """Does the operator behind this layer key DECLARE ``consumes={"z"}``?

        The DECLARATION, never the name. ``tests/test_operator_declaration.py`` fails the build on
        an ``x == "<operator name>"`` comparison for exactly this reason: a name test answers for
        one operator and is wrong for the next one registered with ``add_projector``. The layer key
        may be scoped to an exploration tab (``"mip@tab2"``), which is in no registry, so it goes
        through ``operator_name`` first -- the one place that ``@`` rule is spelled.
        """
        from squidmip._engine import Z_REDUCER, operator_consumes
        from squidmip._operations import operator_name

        try:
            return bool(operator_consumes(operator_name(str(op))) & Z_REDUCER)
        except Exception:                        # noqa: BLE001 - "raw", "computed", an unknown key
            return False

    def _present_z_axis(self) -> None:
        """Drop the pane's z axis while a z-REDUCED result is the layer on screen; else restore it.

        Called wherever "which processing layer is showing" settles: a result arriving lit, a user
        toggling an eye icon or a layer-tree checkbox, and ``show_op``. One rule, every surface,
        because two surfaces enforcing one rule is this project's most-repeated defect.
        """
        model = self._model
        dims = getattr(model, "dims", None)
        if int(getattr(dims, "ndisplay", 2) or 2) == 3:
            return          # a 3D view is asking for the volume; taking z away would empty it
        op = self.visible_op()
        collapse = op is not None and self._reduces_z(op)
        # The plane the user was last looking at, so toggling raw back on does not jump the stack
        # to z=0 -- and so the collapsed presentation is the plane they had, not an arbitrary one.
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

    # -- construction -------------------------------------------------------------------
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

        ``contrast_limits=None`` means "derive one", and what it derives is
        ``_contrast.auto_contrast`` — the FLUORESCENCE rule ported from maragall/stitcher:
        background peak to black, 99.9th percentile on top.

        This is a SEED, not a second owner. napari still owns contrast from the moment the layer
        exists: the user drags napari's slider, the plate follows napari, and nothing recomputes
        this behind them. What it replaces is napari's own autoscale, which for fluorescence puts
        the low end inside the background distribution — so the background lifts off black, the
        tissue saturates, and four additive channels sum to white. Julio, on screen: "the channels
        are not well contrast-adjusted (background looks colored)."

        Derived from the COARSEST pyramid level (~36x fewer pixels on the 10x set, and the level
        napari already fetches for the thumbnail), so seeding costs nothing.
        """
        key = MosaicKey(str(op), str(channel))
        existing = self.find(key.op, key.channel)
        if existing is not None:
            # REUSE the layer: assign new pixels instead of destroying and rebuilding it.
            #
            # A region change used to remove four layers and add four back. Measured on a bare
            # ViewerModel that is 18 ms against 2 ms for an in-place update, and on a LIVE viewer
            # the gap is far worse -- `remove_op` was measured growing 176 ms -> 960 ms over six
            # region changes, because each removal tears down vispy nodes and layer-control
            # widgets that then have to be rebuilt. Julio: "I can't cycle rapidly through these
            # mosaics."
            #
            # Reuse also deletes a whole BUG CLASS rather than just time. Every subscription in
            # this app binds to layer objects -- contrast, visibility, colormap -- and each one
            # has already broken once because `_load_mosaic` destroyed the object underneath it
            # ("the sink went deaf after a rebuild"). A layer that is never destroyed cannot
            # strand its subscribers. It also keeps the user's contrast, colormap and visibility
            # across a region change, which is what "one value per channel" is supposed to mean.
            reused = self._reuse_layer(existing, data, bbox_um=bbox_um, z_scale_um=z_scale_um,
                                       multiscale=multiscale, visible=visible)
            self._present_z_axis()
            return reused

        kwargs: dict[str, Any] = {
            "name": key.label(),
            "metadata": key.as_metadata(),
            "visible": visible,
            # ADDITIVE, not napari's default 'translucent_no_depth'. Fluorescence channels are
            # a COMPOSITE: each carries independent signal and they must sum, exactly as
            # _montage.py already does in the browser ("screen blending, which is the same
            # additive composite"). With the default, the last-added layer simply OCCLUDES the
            # rest — four layers exist, all four visible, each with its own correct colormap, and
            # the user still sees one channel. On the 10x tissue set the order ends 638 nm
            # (#FF0000), so the mosaic rendered flat RED and read as a single-channel bug.
            # Reported twice from the live GUI: "mosaic showing red, so like single collor" and
            # "why is the mosaic only displaying a channel?".
            #
            # Additive is right ACROSS CHANNELS and wrong across OPERATORS of one channel, and a
            # blending mode cannot tell those two apart -- see `_connect_exclusive_op`, which is
            # where that distinction is actually enforced.
            "blending": blending,
        }
        window = contrast_limits
        if window is None:
            window = _auto_window_for(data, bool(multiscale))
        if window is not None:
            lo, hi = float(window[0]), float(window[1])
            # A degenerate window is passed through, NOT widened. _pct_window returns hi <= lo
            # for a blank channel deliberately, because widening it to (lo, lo+1) renders a
            # blank channel as full white, i.e. as signal.
            if hi > lo:
                kwargs["contrast_limits"] = (lo, hi)
        if colormap is not None:
            kwargs["colormap"] = colormap
        if multiscale is not None:
            kwargs["multiscale"] = multiscale

        # PLACED AT CONSTRUCTION, not afterwards. See `placement_for`: assigning scale/translate to
        # a live layer moves the dims range under it and costs a second whole-region fuse.
        placed = False
        if bbox_um is not None:
            try:
                shape = tuple(_first_level_shape(data, bool(multiscale)))
                kwargs["scale"], kwargs["translate"] = placement_for(
                    len(shape), bbox_um, shape[-2:], z_scale_um)
                placed = True
            except (ValueError, TypeError, IndexError):
                # A bbox this rule refuses is a placement problem, not a reason to have no layer.
                # `_place` below raises the same way and is where that has always been reported.
                placed = False

        # Everything here is OUR write, not the user's. Subscribers must be able to tell the
        # difference or the plate latches manual on open and kills per-region contrast.
        with self.programmatic():
            layer = self._model.add_image(data, **kwargs)

            # The slider must span the DTYPE, not the window we seeded. napari sizes
            # contrast_limits_range from the data it sampled, so a tight seed leaves the user
            # unable to open the window back up past it -- the control silently bounds them to
            # our choice. The stitcher sets this immediately after every add for the same reason.
            try:
                from squidmip._contrast import dtype_range

                dt = getattr(_first_level(data, bool(multiscale)), "dtype", None)
                if dt is not None and "contrast_limits" in kwargs:
                    lo_r, hi_r = dtype_range(dt)
                    lo_w, hi_w = kwargs["contrast_limits"]
                    # Never narrower than what is displayed, or napari clamps the window itself.
                    layer.contrast_limits_range = (min(lo_r, lo_w), max(hi_r, hi_w))
            except Exception:               # noqa: BLE001 - cosmetic; the layer is already good
                pass

            if bbox_um is not None and not placed:
                shape = tuple(_first_level_shape(data, bool(multiscale)))[-2:]
                self._place(layer, bbox_um, shape, z_scale_um)
            elif placed:
                # `_place` also labels the axes; the placement itself already went in above.
                self._label_units(layer)

            self._register_channel(key.channel, layer)
            # Point the camera at the data. add_image does NOT move the camera, so the first
            # mosaic landed outside the view and the canvas stayed black while all four layers
            # sat correctly in the layer list -- Julio: "all I see are the controls... it just
            # looks like an empty gray canvas". Reset only while this is the FIRST layer, so a
            # later channel does not yank the view back while the user is panning. Inside the
            # programmatic() block: reset_view is OUR camera move, not a user gesture.
            try:
                if len(self.ours()) <= 1:
                    self._model.reset_view()
            except Exception:                    # noqa: BLE001 - view convenience, never fatal
                pass
        # A layer that ARRIVES lit is the same gesture as one the user lights, so it obeys the
        # same one-operator-per-channel rule. Otherwise an operator result delivered over a
        # visible raw is a sum from the moment it lands, and the user never touched anything.
        # OUTSIDE `programmatic()` deliberately: the peer really did go dark, and the plate is a
        # sink that has to be told, exactly as it is told about a user's own toggle.
        if visible:
            self._darken_other_ops(key.channel, layer)
        # ...and the pane presents a z axis only while something on screen HAS one. A z-reduced
        # result arriving lit is exactly the moment raw's slider stops describing the picture.
        self._present_z_axis()
        return layer

    def _reuse_layer(self, layer: Any, data: Any, *, bbox_um, z_scale_um, multiscale, visible):
        """Point an EXISTING layer at new pixels, keeping everything the user owns.

        What is deliberately NOT touched: contrast_limits, colormap, gamma, opacity, blending.
        Those are the user's, and a region change is not a reason to reset them -- that is the
        whole point of one contrast value per channel. What IS updated: the data, the placement
        (each region sits at its own stage coordinates) and the z scale.

        Placement goes through `_place`, the ONE placement rule -- not a second copy of the
        arithmetic, which is exactly the two-owner defect `_place` exists to prevent.

        Everything happens inside `programmatic()` so the plate's sinks do not read our write as
        a user gesture. napari re-renders on the data assignment; nothing else has to be told.
        """
        with self.programmatic():
            # A region change replaces the pixels, so any z stack stashed for the previous region
            # is now a stale mosaic wearing this layer's name. Restore FIRST (which puts the layer
            # back at full ndim) and let `_present_z_axis` re-decide after the new data lands.
            self._restore_layer_z(layer)
            layer.data = data
            if visible is not None:
                layer.visible = bool(visible)
            if bbox_um is not None:
                shape = tuple(_first_level_shape(data, bool(multiscale)))[-2:]
                self._place(layer, bbox_um, shape, z_scale_um)
        return layer

    def _place(self, layer: Any, bbox_um: Sequence[float], shape: Sequence[int],
               z_scale_um: Optional[float] = None) -> None:
        """Put *layer* at its stage-micrometre footprint. THE one placement rule, shared.

        Trailing two axes are (y, x); a z-stack's leading axis is not placed by bbox_um, which
        describes the XY footprint only, so scale/translate are padded to line up with the
        trailing spatial axes.

        The z axis carries the STEP in micrometres, not 1.0. With a unit z scale the 2-D slider
        still steps correctly but the 3-D toggle renders an isotropic block out of anisotropic
        data — IMA-255 exists precisely because dz/pixel has to reach the renderer. Same world
        units as x/y, so the ratio comes out right.

        Shared by ``add_mosaic`` / ``add_labels`` / ``add_points`` on purpose: an analysis result
        that is placed by a SECOND copy of this arithmetic is one edit away from sitting next to
        the mosaic it claims to describe rather than on top of it.
        """
        scale, translate = placement_for(
            int(getattr(layer, "ndim", len(shape))), bbox_um, shape, z_scale_um)
        layer.scale = scale
        layer.translate = translate
        self._label_units(layer)

    @staticmethod
    def _label_units(layer: Any) -> None:
        """Micrometres on every axis, so napari's scale bar reads the LAYER's units.

        The >=0.7 path, now that ``viewer.scale_bar.unit`` is deprecated (IMA-265). Guarded: an
        older napari has no ``.units``, and mislabelling is never worth crashing a mosaic over.

        Split out of :meth:`_place` so that a layer placed at CONSTRUCTION (``add_mosaic`` hands
        scale/translate to ``add_image``) is still labelled exactly once. Re-running the whole of
        ``_place`` to get the label would re-assign the scale, which is the extra fuse that change
        exists to remove.
        """
        try:
            layer.units = ("um",) * int(getattr(layer, "ndim", 2))
        except Exception:                # noqa: BLE001 - cosmetic; the scale is already right
            pass

    # -- analysis results: the NON-image layer types -------------------------------------
    # add_mosaic makes Image layers. A segmentation operator's output is not an image, and
    # rendering it as one is not a cosmetic mistake: a label image through add_image is a
    # near-black gradient under an intensity colormap, with no label picking and no transparent
    # background. Every napari segmentation surface returns Labels (napari-segment-blobs-
    # and-things-with-membranes annotates `-> "napari.types.LabelsData"`; cellpose-napari calls
    # viewer.add_labels(masks)), so this is the layer type the operators after this one need.
    #
    # These deliberately do NOT go through _register_channel. Contrast is linked per channel and
    # a Labels/Points layer has no `contrast_limits` at all — registering one as a peer makes the
    # next link_layers call raise. napari OWNS contrast; an analysis overlay has no part in it.

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
        """Add (or replace) a segmentation MASK as a napari ``Labels`` layer.

        *data* must be an integer (or bool) array — napari's Labels layer rejects floats. Label
        ``0`` is background and renders transparent, so the mosaic underneath stays visible.
        """
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
        """Add (or replace) detection CENTROIDS as a napari ``Points`` layer.

        *data* is ``(N, 2)`` in ``(row, col)`` — napari's own 2D world axis order, which is also
        what ``skimage`` centroids and ``blob_log`` return, so no transpose happens anywhere.
        An EMPTY ``(0, 2)`` array is legitimate and still produces a layer: "zero found" is an
        answer, and skipping the layer would make it indistinguishable from "nothing ran".

        *features* rides along as the per-object record (Fractal's feature-table contract: one
        row per object, keyed by label value).
        """
        kwargs: dict[str, Any] = {
            "visible": visible, "size": float(size), "symbol": symbol,
            "face_color": face_color, "border_color": border_color,
        }
        if features is not None:
            kwargs["features"] = features
        return self._add_result("add_points", op, channel, data, kwargs, bbox_um, shape)

    # -- THE delivery seam: one operator result -> the layer type its DECLARATION names ---------
    #
    # Before this, every operator result reached napari through `add_mosaic`, i.e. through
    # `add_image`, whatever the pixels meant. `spot` was already shipping a LABEL IMAGE down that
    # path: integer object ids handed to the fluorescence auto-window (`_auto_window_for`), which
    # stretched "label 1 .. label 400" as if it were photons and drew an opaque near-black gradient
    # over the mosaic it was supposed to annotate. `add_labels` existed the whole time, twelve
    # lines further up this file, and no delivery path called it.
    #
    # The cure is a TABLE keyed by the result's declared kind, not a branch. Nothing here knows
    # what a spot or a cellpose is; it knows "labels" and "intensity", and an operator says which
    # one it is at registration (`add_projector(..., produces=...)`). A third kind -- a Points
    # result, a Shapes result -- is one more entry plus one adapter, and no edit to either caller.
    def _add_intensity(self, op, channel, data, *, colormap, bbox_um, visible, z_scale_um,
                       multiscale, contrast_limits):
        """The pixels measure light: an Image layer, windowed, colormapped, blended additively."""
        return self.add_mosaic(op, channel, data, colormap=colormap, bbox_um=bbox_um,
                               visible=visible, z_scale_um=z_scale_um, multiscale=multiscale,
                               contrast_limits=contrast_limits)

    def _add_label_result(self, op, channel, data, *, colormap, bbox_um, visible, z_scale_um,
                          multiscale, contrast_limits):
        """The pixels are OBJECT IDS: a Labels layer.

        Four of this adapter's arguments are dropped on purpose, and dropping them is the whole
        point rather than an omission:

        * ``colormap`` / ``contrast_limits`` — a label VALUE is a name, not a quantity. napari's
          Labels layer has no ``contrast_limits`` at all, and its colouring is a per-label lookup;
          handing it a fluorescence window would be meaningless if it were even accepted.
        * ``multiscale`` — a coarser level of a label image is only correct under nearest-neighbour
          decimation, and nothing on this path guarantees that. A result arrives as one array here.
        * ``z_scale_um`` — the operator that produces labels today is a plane-op delivered per
          plane; there is no z axis on this layer to scale.

        Integer dtype is REQUIRED by napari and is checked here rather than left to raise from
        inside napari, so the message names the operator whose declaration was wrong.
        """
        import numpy as _np

        arr = _np.asarray(data)
        if not (_np.issubdtype(arr.dtype, _np.integer) or arr.dtype == bool):
            raise ValueError(
                f"operator {op!r} declares produces='labels' but its {channel} pixels are "
                f"{arr.dtype}. A label image is integer object ids; napari's Labels layer rejects "
                "floats, and a float 'label' cannot be picked or counted.")
        return self.add_labels(op, channel, arr, bbox_um=bbox_um, visible=visible)

    #: result kind -> the adapter that turns it into a layer. THE dispatch. Extended by adding a
    #: row, never by adding a branch in a caller.
    _RESULT_ADDERS: dict = {
        "intensity": _add_intensity,
        "labels": _add_label_result,
    }

    def add_result(self, kind: str, op: str, channel: str, data: Any, *,
                   colormap: Optional[Any] = None, bbox_um: Optional[Sequence[float]] = None,
                   visible: bool = True, z_scale_um: Optional[float] = None,
                   multiscale: Optional[bool] = None,
                   contrast_limits: Optional[tuple[float, float]] = None) -> Any:
        """Add one operator result as the layer type its *kind* names.

        *kind* is the operator registry's ``produces`` declaration, carried here on the
        :class:`squidmip._result.Result`. Every sink calls THIS, so a new result kind lands in the
        plate pane and in every region window at once instead of in whichever one was edited.

        Raises ``ValueError``, naming the operator, on a kind this viewer cannot draw — an operator
        that declares something the display does not implement must not fall back to drawing it as
        an image, which is exactly the failure being fixed.
        """
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
        # Subscribe THIS layer to the channel's user-contrast fan-out.
        #
        # This is the Defect 5 fix. on_user_contrast used to walk _by_channel once, at
        # subscribe time, and connect to the layer objects it found. _load_mosaic destroys and
        # recreates every layer on a region change, so the recreated layers had no connection
        # and the sync silently stopped after exactly one region change. Connecting HERE means
        # the subscription is keyed on the CHANNEL and any layer that ever shows it is wired
        # up, whenever it is created.
        self._connect_user_contrast(channel, layer)
        self._connect_user_visibility(channel, layer)
        self._connect_user_colormap(channel, layer)
        key = key_of(layer)
        if key is not None:
            # Keyed on the OP, and connected here for the same reason the three above are: layer
            # objects are destroyed and recreated on every region change, so a subscription made
            # anywhere else goes deaf after one.
            self._connect_user_op(key.op, layer)
        # LAST, and the order is load-bearing. This tap darkens the channel's other operators, and
        # each of those emits its own op event; connecting it after `_connect_user_op` means a
        # subscriber hears "mip came on" before it hears "raw went off", which is the order the
        # gesture actually happened in. Connected before, the consequence was reported ahead of
        # the cause.
        self._connect_exclusive_op(channel, layer)
        # Link contrast across every processing layer showing this channel, so that FROM THE
        # NEXT WRITE ON there is only ever one value: a drag on any peer moves them all.
        #
        # What linking does NOT do, measured on a bare ViewerModel: it does not equalise the
        # values at link time. napari's link_layers connects events, nothing more. Each layer
        # keeps the window `add_mosaic` seeded from ITS OWN pixels until somebody writes one,
        # and only then do they converge. That is deliberate and it is not a bug: a fresh decon
        # result has to be individually legible on arrival or you cannot judge whether it used
        # the right iteration count. When the user wants the flip to be a real comparison
        # instead, `match_contrast_to` equalises them on demand ("Match raw contrast").
        if len(peers) > 1:
            self._model.layers.link_layers(peers, ("contrast_limits",))

    def match_contrast_to(self, op: str) -> int:
        """Copy *op*'s contrast window onto every OTHER processing layer of the same channel.

        The explicit half of the contrast story. Layers are seeded per layer (see
        :meth:`add_mosaic`) so an operator result arrives legible on its own terms, and linked
        per channel (see :meth:`_register_channel`) so they never diverge again once written.
        Between those two lies the case this method serves: raw and its operator layers are on
        DIFFERENT windows, so flipping between them compares two stretches rather than two
        images. Calling this makes the flip a real comparison.

        Writes each peer explicitly rather than nudging *op*'s own layer and relying on the
        link to propagate: assigning a layer the value it already holds is not a state change
        anyone should have to reason about, and the point here is to be sure, not to be clever.

        Returns the number of peer layers written (channels with no *op* layer are skipped, and
        *op*'s own layers are not counted).
        """
        matched = 0
        for channel, peers in self._by_channel.items():
            source = self.find(op, channel)
            if source is None:
                continue                     # this channel has no `op` layer to match against
            window = (float(source.contrast_limits[0]), float(source.contrast_limits[1]))
            for ly in peers:
                if ly is source:
                    continue
                try:
                    ly.contrast_limits = window
                except Exception:            # noqa: BLE001 - one odd layer is skipped, not fatal
                    continue
                matched += 1
        return matched

    def remove_op_channel(self, op: str, channel: str) -> bool:
        layer = self.find(op, channel)
        if layer is None:
            return False
        peers = self._by_channel.get(channel, [])
        if layer in peers:
            # Unlink BEFORE removal: a linked layer that is destroyed while still linked leaves
            # napari holding a callback onto a dead layer.
            if len(peers) > 1:
                self._model.layers.unlink_layers(peers, ("contrast_limits",))
            peers.remove(layer)
            if len(peers) > 1:
                self._model.layers.link_layers(peers, ("contrast_limits",))
            # No re-tap needed: the tap lives on EVERY layer of the channel, not on a lead, so
            # removing one cannot leave the channel untapped.
        self._model.layers.remove(layer)
        return True

    def remove_op(self, op: str) -> list[str]:
        gone = []
        for channel in list(self.channels(op)):
            if self.remove_op_channel(op, channel):
                gone.append(channel)
        return gone

    # -- the before/after toggle --------------------------------------------------------
    def show_op(self, op: str) -> list[str]:
        """Make exactly one processing layer visible. Returns the channels now showing.

        This is the stitching before->after toggle. Channel contrast is preserved across the
        switch because contrast is linked per channel, not stored per processing layer.
        """
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

    # -- contrast, one value per channel -------------------------------------------------
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
        """Wire one layer into *channel*'s user-contrast fan-out.

        Every layer of the channel is connected, not just the first, because "the first" is a
        layer object and layer objects do not survive a region change.

        Linked layers propagate a write to their peers, and each peer then emits its own
        event, so one user drag arrives here once per layer showing the channel. The echoes
        are collapsed by VALUE, not by a re-entrancy flag: napari emits the peers' events
        after this handler has already returned, so a flag set and cleared around the delivery
        catches none of them (measured -- three linked layers delivered three callbacks).
        """
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
        """Wire one layer's COLORMAP into *channel*'s colour fan-out.

        Julio: "I change channel colormap in napari and plate view doesn't react." The plate
        composites with its own ``(C, 3)`` RGB table, resolved once from the acquisition's
        ``display_color``. That table was a SECOND answer to "what colour is this channel",
        settled at open and never revised -- so recolouring a layer in napari left the two panes
        tinting the same channel differently, which is the same defect shape as the contrast that
        would not follow.

        What travels is an RGB TRIPLE, not napari's colormap object: the plate composites with
        floats and must not learn what a napari ``Colormap`` is. The triple is the colormap's
        value at full intensity, which is exactly the tint the canvas shows.
        """
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
        """Subscribe to colormap changes the USER made. ``callback(channel, (r, g, b))``, floats
        in 0..1 -- the colour the canvas is actually tinting that channel."""
        self._user_colormap_cbs.append(callback)

    def channel_rgb(self, channel: str) -> Optional[tuple]:
        """The RGB the canvas is tinting *channel* with right now, or None if it has no layers."""
        peers = self._by_channel.get(channel) or []
        return _colormap_rgb(peers[0]) if peers else None

    def _connect_user_visibility(self, channel: str, layer: Any) -> None:
        """Wire one layer's eye icon into *channel*'s visibility fan-out.

        Exactly the same shape as the contrast tap, and for the same reason: Julio, on the plate
        view -- "there shouldn't be any controls for the plate view. It just reacts to toggles
        and contrast adjustments in napari." A plate that owns its own checkboxes is a second
        control over one quantity, which is this project's most-repeated defect; a plate that
        SUBSCRIBES cannot disagree with what is on the canvas.

        Visibility is NOT linked across processing layers the way contrast is -- hiding the
        stitched 488 must not hide the raw 488, because the before/after toggle IS a visibility
        flip over a processing-layer group. So the channel is reported visible when ANY layer
        showing it is visible, which is what "is this channel on screen" actually means.
        """
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

        # SEEDED at registration, for the same reason the op latch below is: an unseeded latch
        # always delivers its first event even when the answer did not move. Switching operators
        # leaves "is 488 on screen" at True throughout, and an unseeded latch announced that
        # non-change to the plate as though the user had touched the channel.
        self._last_visible[channel] = any(
            bool(getattr(p, "visible", False)) for p in (self._by_channel.get(channel) or []))
        layer.events.visible.connect(_fire)

    def on_user_visibility(self, callback) -> None:
        """Subscribe to channel visibility the USER changed, via napari's own eye icons.

        ``callback(channel, visible)``. The seam that lets the plate drop its checkboxes.
        """
        self._user_visibility_cbs.append(callback)

    def _connect_user_op(self, op: str, layer: Any) -> None:
        """Wire one layer's visibility into the PROCESSING-LAYER fan-out.

        This is a second tap on the same ``visible`` event as ``_connect_user_visibility``, and
        it has to be, because that one answers a different question and its echo filter destroys
        this one's answer. It reports "is this CHANNEL on screen anywhere", deliberately ORed
        across processing layers, so showing the ``mip`` group while ``raw`` is still up leaves
        every channel's answer unchanged and it returns at its first branch. That is correct for
        the eye icons and it is exactly why nothing reached the plate when the user picked a
        processing layer in a window's layer tree: the state moved (``visible_op`` would have
        said so) and no signal was ever emitted. Julio: "after I click an operator layer in our
        window, the thumbnails don't update."

        So the question here is per OP -- "is this processing layer on screen" -- and the echo
        collapse is per op too, which folds the one group toggle that fires once per channel into
        a single delivery.
        """
        def _fire(event=None, _op=str(op)):
            on = any(bool(getattr(ly, "visible", False)) for ly in self.group(_op))
            if self._last_op_visible.get(_op) == on:
                return                      # a sibling channel of the same group; already told
            self._last_op_visible[_op] = on
            if self.is_programmatic:
                return                      # OUR write: recorded, never reported as a gesture
            for cb in list(self._user_op_cbs):
                cb(_op, on)

        # SEED the latch with the truth as the layer is registered, so the op's first event is
        # collapsed like any other echo when it reports a state that never changed. An unseeded
        # latch always fires once: `_darken_other_ops` hiding raw · 405 while raw · 488 is still
        # up leaves "is raw on screen" at True, and an unseeded latch announced that non-change as
        # a gesture, ahead of the real one.
        self._last_op_visible[str(op)] = any(
            bool(getattr(ly, "visible", False)) for ly in self.group(str(op)))
        layer.events.visible.connect(_fire)

    # -- one operator per channel on screen ----------------------------------------------
    def _darken_other_ops(self, channel: str, keep: Any) -> list[str]:
        """Make *keep* the ONLY lit mosaic for *channel*. Returns the ops darkened.

        Operates over ``_by_channel``, which holds image mosaics and nothing else: ``add_labels``
        and ``add_points`` deliberately skip ``_register_channel`` (see ``_add_result``), so a
        segmentation mask or a spot overlay is never darkened by a mosaic switch. That is the
        behaviour ``SPOTS_OP`` already relies on -- an analysis overlay is drawn OVER whichever
        mosaic is showing, not instead of it -- and it is preserved here for free rather than
        special-cased.
        """
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
        """At most ONE operator's mosaic per channel is lit at a time.

        Julio: "Intensity grows with the amount of layers that are toggled on in my window."

        Exactly right, and it is arithmetic, not a rendering glitch. Every mosaic is added
        ``additive`` (see ``add_mosaic``), so raw · 488 + mip · 488 + decon · 488 all lit is
        three copies of one channel's signal summed into one green. Turn on a fourth and it gets
        brighter again. The channels were never the problem -- summing 405/488/561/638 is the
        whole point of a composite and is what ``additive`` is for -- the problem is that the
        SAME channel is on screen more than once.

        WHY NOT `translucent`. Because a blending mode cannot express this requirement. napari
        blends a layer against everything beneath it in the one flat stack; there is no "sum with
        those siblings, occlude these". Setting the operator layers ``translucent`` at full
        opacity makes each one OCCLUDE the channels under it -- which is precisely the defect the
        additive choice was made to fix ("mosaic showing red, so like single collor"), just moved
        one level up. At partial opacity it is worse: a weighted average of raw and decon is not a
        comparison of them, it is a third picture that is neither, and the contrast slider then
        describes none of what is on screen.

        WHY EXCLUSIVITY. It is already this application's model, stated in three places and
        enforced in none of them: ``show_op`` is documented as "make exactly one processing layer
        visible", ``visible_op`` returns ONE op, and ``on_user_op``'s subscriber is the plate
        asking "which operator am I looking at", a question with one answer. The layer tree simply
        let the user check two groups at once and nothing said no.

        Scoped per CHANNEL rather than per group, which is strictly more permissive than
        ``show_op``: raw · 405 beside mip · 488 is two different channels and is a legitimate
        composite, so it is left alone. Only a second copy of the same channel is darkened.

        Enforced HERE, on the layer's own ``visible`` event, rather than in the layer tree,
        because the tree is not the only surface that writes it -- napari's own eye icons do too,
        and a rule that lives in one of two controls over one quantity is this project's
        most-repeated defect. Both surfaces write ``layer.visible``, so both obey.

        Programmatic writes are skipped: ``_reuse_layer`` restores a region's saved visibility and
        ``render_max_res_3d`` swaps data under lit layers, and neither is a user choosing an
        operator. The ARRIVAL of a new lit layer is handled in ``add_mosaic`` instead, where the
        layer is known to be new.

        Recursion is bounded by construction: darkening a peer fires this handler for that peer,
        which returns at its first branch because the peer is no longer visible.
        """
        def _fire(event=None, _ch=channel, _ly=layer):
            if self.is_programmatic:
                return
            if bool(getattr(_ly, "visible", False)):
                self._darken_other_ops(_ch, _ly)
            # ...and BOTH directions re-decide the z axis: a layer going dark never forces anything
            # else on, but it does change what is on screen, which is what the axis describes.
            self._present_z_axis()

        layer.events.visible.connect(_fire)

    def on_user_op(self, callback) -> None:
        """Subscribe to the PROCESSING LAYER the user showed or hid in this window.

        ``callback(op, visible)``. The seam that lets the plate follow a window's layer tree the
        same way it already follows its contrast, its eye icons and its colormaps: the window is
        the owner of "which operator am I looking at", and the plate is a sink.
        """
        self._user_op_cbs.append(callback)

    def channel_visible(self, channel: str) -> Optional[bool]:
        """Is this channel on screen anywhere? None when the channel has no layers."""
        peers = self._by_channel.get(channel) or []
        if not peers:
            return None
        return any(bool(getattr(p, "visible", False)) for p in peers)

    def on_user_contrast(self, callback) -> None:
        """Subscribe to contrast changes the USER made. Programmatic writes never arrive here.

        ``callback(channel, lo, hi)``. This is the seam that lets the plate be a pure sink: it
        is told what the owner resolved, and it never writes back.

        The subscription is per CHANNEL and outlives the layers: channels added after this
        call are covered too, because the connection is made in ``_register_channel`` rather
        than swept up here. Only the callback is recorded here.
        """
        self._user_contrast_cbs.append(callback)

    def on_contrast_changed(self, callback) -> None:
        """Subscribe to contrast changes via napari's PUBLIC event.

        This replaces the ndv contrast tap, which subclassed ``ndv.views.bases.LutView`` and
        reached into the private ``_lut_controllers`` dict — the most ndv-entangled design in
        the codebase and the one thing that could not have been ported.
        """
        for peers in self._by_channel.values():
            if peers:
                peers[0].events.contrast_limits.connect(callback)


def _auto_window_for(data: Any, multiscale: bool) -> Optional[tuple[float, float]]:
    """The seed contrast window for *data*, or None to let napari autoscale.

    None is returned for a blank or unreadable plane rather than a guess, so a channel with no
    signal is not handed a window that renders its noise as tissue. Any failure here is
    cosmetic -- a wrong window is a bad picture, a raised exception inside add_image is no
    picture at all -- so it degrades to napari's own autoscale instead of propagating.
    """
    from squidmip._contrast import auto_contrast, sample_plane

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


# -- what comes BACK off a layer ------------------------------------------------------------
#
# The two helpers above describe data on its way INTO napari, where the caller already knows
# whether it built a pyramid. These two describe the return trip, where it does not — and that
# is a different question with a trap in it.
#
# `layer.data` for a multiscale layer is NOT the list that was handed in. napari wraps it in
# `napari.layers._multiscale_data.MultiScaleData`, a `Sequence` of levels that is neither a list
# nor a tuple, and that presents level 0's `ndim`, `shape`, `dtype` and `size` as its own. So an
# `isinstance(data, (list, tuple))` pyramid check misses it and every line after that check runs
# on the WRONG object:
#
#   * `data[z]` walks the LEVELS, not the z planes (IndexError, or a coarser level);
#   * `data.max(axis=0)` raises `AttributeError: 'MultiScaleData' object has no attribute 'max'`
#     — the crash Julio hit running cellpose from the running GUI, at `_workers._full_res_mip`;
#   * `np.asarray(data)` silently returns `MultiScaleData.__array__`, which is the COARSEST
#     level. That one is the dangerous half: a nuclei count on a 4x-downsampled level merges
#     touching nuclei and under-reports, and nothing on screen says so.
#
# `MosaicLayers._collapse_layer_z` learned this in 2026-08 and wrote it in a comment; the four
# other sites that read `layer.data` did not. So the knowledge lives here now, once, and every
# reader of a layer's data goes through it.


def pyramid_levels(data: Any) -> Optional[list]:
    """The levels of *data* when it is a MULTISCALE PYRAMID, else None. Highest resolution first.

    Recognises every container napari's multiscale contract can produce — the ``list``/``tuple``
    handed to ``add_image(..., multiscale=True)``, and the ``MultiScaleData`` that comes back out
    of ``layer.data``. The discriminator for "is this a pyramid at all" is that it is a
    ``Sequence`` (``numpy``/``dask``/``zarr`` arrays are not) whose FIRST ELEMENT is itself an
    array. A plain nested Python list encodes ONE array and is not a pyramid, which is why the
    element's ``ndim`` is checked rather than the container's type alone.

    Raises on an EMPTY sequence rather than returning None: "the layer holds no levels" is a
    broken layer, not a single-scale one, and returning None would hand the caller a container it
    would then index.
    """
    if isinstance(data, (str, bytes)) or not isinstance(data, _SequenceABC):
        return None
    if len(data) == 0:
        raise ValueError("the layer holds an EMPTY multiscale pyramid — nothing to read.")
    return list(data) if int(getattr(data[0], "ndim", 0)) >= 2 else None


def full_res_level(data: Any) -> Any:
    """The FULL-RESOLUTION array behind a napari layer's ``data``, pyramid or not.

    Level 0 for a pyramid, the array itself otherwise. Never materialises anything: a lazy
    dask/zarr level is returned lazy, so the caller decides what to compute.
    """
    levels = pyramid_levels(data)
    return data if levels is None else levels[0]


# --------------------------------------------------------------------------------------
# The embedded pane
# --------------------------------------------------------------------------------------


def build_pane(parent: Any = None) -> tuple[Any, MosaicLayers, Any]:
    """Build a REAL napari Viewer and hand back its window, canvas and layer facade.

    Returns ``(qt_viewer, MosaicLayers, viewer)``.

    This used to construct a bare ``QtViewer(ViewerModel())`` with no napari Window, to keep
    napari's chrome out. That was the wrong trade. The Window is where napari's layer controls,
    dims sliders (the z control), ndisplay 2D/3D button, contrast behaviour and stylesheet all
    live -- strip it and you must rebuild all of that by hand, badly. Julio, looking at the
    result: "You're not showing me a napari window... I don't understand why you're inventing the
    wheel when napari literally has an API."

    ``show=False`` so no top-level window appears and no second event loop starts; the host
    QApplication drives it, and the caller reparents ``viewer.window._qt_window`` into our pane.
    """
    verify_napari_bindings()

    import napari

    viewer = napari.Viewer(show=False)
    enable_scale_bar(viewer)
    qt_viewer = getattr(viewer.window, "_qt_viewer", None)
    if parent is not None and qt_viewer is not None:
        qt_viewer.setParent(parent)
    return qt_viewer, MosaicLayers(viewer), viewer


def enable_scale_bar(viewer: Any, unit: str = "um") -> None:
    """Turn on napari's built-in scale bar, in micrometres, for the mosaic view (IMA-265).

    "Part of zooming into our mosaic is having a scale bar." napari already HAS one, and our layers
    already carry ``layer.scale`` in micrometres (set from ``pixel_size_um`` / ``dz_um`` in
    :meth:`MosaicLayers.add_mosaic`), so the bar's length is derived from real world coordinates by
    napari itself -- there is nothing to construct and nothing to keep in sync. This only makes it
    visible and names the unit; a bar that lied would be worse than none, and the number is napari's
    own, computed from the same scale the pixels are placed by.

    The unit is set two ways on purpose: ``layer.units`` (each mosaic, in add_mosaic) is the source
    napari >=0.7 reads, and ``viewer.scale_bar.unit`` is what <0.7 reads. Both are guarded so a
    binding that lacks either still yields a visible bar rather than a crash.
    """
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
