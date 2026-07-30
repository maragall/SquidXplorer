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
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from squidmip._logpane import get_logger

log = get_logger("napari")

#: Fallback GPU 3D texture cap (Apple GPUs report GL_MAX_3D_TEXTURE_SIZE = 2048). The live value
#: is read off the canvas at runtime; this is only used until that is known.
_DEFAULT_MAX_3D_TEXTURE = 2048

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
        for ly in self.ours():
            if key_of(ly) == MosaicKey(op, channel):
                return ly
        return None

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
        try:
            if full_res:
                data = ly.data
                if not isinstance(data, (list, tuple)):
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
            return self._reuse_layer(existing, data, bbox_um=bbox_um, z_scale_um=z_scale_um,
                                     multiscale=multiscale, visible=visible)

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

            if bbox_um is not None:
                shape = tuple(_first_level_shape(data, bool(multiscale)))[-2:]
                self._place(layer, bbox_um, shape, z_scale_um)

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
        scale, translate = scale_translate_from_bbox_um(bbox_um, shape)
        extra = max(0, int(getattr(layer, "ndim", len(shape))) - 2)
        lead = (float(z_scale_um) if (extra and z_scale_um) else 1.0,) * extra
        layer.scale = lead + tuple(scale)
        layer.translate = (0.0,) * extra + tuple(translate)
        # Micrometres on every axis (x/y from pixel size, z from the µm step above), so napari's
        # scale bar reads the LAYER's units -- the >=0.7 path, now that viewer.scale_bar.unit is
        # deprecated (IMA-265). Guarded: an older napari has no .units, and mislabelling is never
        # worth crashing a mosaic over. Here in _place so EVERY placed layer is labelled once.
        try:
            layer.units = ("um",) * int(getattr(layer, "ndim", len(scale)))
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
        # Link contrast across every processing layer showing this channel, so the
        # before->after toggle preserves the window and there is only ever one value.
        if len(peers) > 1:
            self._model.layers.link_layers(peers, ("contrast_limits",))

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

        layer.events.visible.connect(_fire)

    def on_user_visibility(self, callback) -> None:
        """Subscribe to channel visibility the USER changed, via napari's own eye icons.

        ``callback(channel, visible)``. The seam that lets the plate drop its checkboxes.
        """
        self._user_visibility_cbs.append(callback)

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
