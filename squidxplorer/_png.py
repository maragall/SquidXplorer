"""High-resolution PNG export of what a view window is SHOWING.

Never a canvas screenshot (that is screen resolution): the visible operator layer's own
full-resolution pixels, one plane per visible channel, composited through the ONE
window-multiply-sum loop (``_montage.composite``) under the contrast that is on screen —
the same latched-look rule the .mp4 export (``_video``) follows. Materialising a plane can
decode FOVs, so :func:`render_view_png` belongs on a worker thread (``_workers._PngWorker``),
never the Qt thread.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import numpy as np

from squidxplorer._mosaic_source import _MAX_FUSED_PX

#: Long-side cap, THE fuser's own (8192 px): a PNG is a display artifact, and past this a
#: paste into a slide deck stops opening. A clip is reported through the returned ``step``.
PNG_MAX_PX = int(_MAX_FUSED_PX)


class PngChannel(NamedTuple):
    """One visible channel of the scene being exported, as the window sees it."""

    name: str
    data: object            #: the napari layer's ``data`` — pyramid, stack or plane; may be lazy
    clim: tuple             #: the on-screen ``(lo, hi)`` contrast window
    rgb: tuple              #: the layer's 8-bit tint, ``(r, g, b)`` in 0–255
    #: THIS channel's z plane (a raw stack's on-screen z; a single-plane result clamps to its
    #: own plane). None falls back to the opening z. Per channel, because one scene mixes
    #: depth-keeping and z-reduced layers.
    z_index: Optional[int] = None


def png_problem() -> Optional[str]:
    """``None`` when this machine can write a PNG, else a sentence naming what is missing."""
    try:
        from PIL import Image  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - reported by name, never swallowed
        return f"Pillow is not installed ({type(exc).__name__}: {exc})."
    return None


def render_view_png(
    channels: "Sequence[PngChannel]",
    *,
    max_px: int = PNG_MAX_PX,
) -> tuple[np.ndarray, int]:
    """Composite *channels*, each at its OWN z plane, into one ``(H, W, 3)`` uint8 RGB.

    Returns ``(rgb, step)`` where ``step`` is the decimation that kept the long side within
    *max_px* — 1 means native resolution, anything else is a clip the caller must SAY.
    Materialises each channel's full-resolution plane, so call this off the Qt thread.
    """
    if not channels:
        raise ValueError("no channels to export - every channel is hidden.")
    from squidxplorer._workers import _full_res_plane  # the one layer-data plane rule

    planes = [np.asarray(_full_res_plane(_level_within(c.data, max_px), c.z_index))
              for c in channels]
    shapes = {p.shape for p in planes}
    if len(shapes) > 1:
        raise ValueError(
            f"the visible channels disagree on shape ({sorted(shapes)}) - one image cannot "
            "hold them.")
    h, w = planes[0].shape
    # The level pick is a decimation too: ``step`` reports the TOTAL against native.
    native_long = max(_native_long_side(c.data, max(h, w)) for c in channels)
    level_step = max(1, int(round(native_long / float(max(h, w)))))
    step = max(1, -(-max(h, w) // max(1, int(max_px))))    # ceil
    if step > 1:
        planes = [p[::step, ::step] for p in planes]
    step *= level_step
    store = np.stack(planes)
    colors = np.stack([np.asarray(c.rgb, dtype=np.float32) / 255.0 for c in channels])
    windows = [(float(c.clim[0]), float(c.clim[1])) for c in channels]
    from squidxplorer._montage import composite

    return composite(store, colors, windows), step


def _native_long_side(data, fallback: int) -> int:
    """The long side of *data*'s NATIVE (finest) level, or *fallback* for a non-pyramid."""
    from squidxplorer._napari_view import pyramid_levels

    levels = pyramid_levels(data)
    shp = getattr(levels[0], "shape", None) if levels else None
    return max(int(v) for v in shp[-2:]) if shp else int(fallback)


def _level_within(data, max_px: int):
    """The finest pyramid level of *data* whose long side fits *max_px*, else the coarsest;
    a non-pyramid passes through. A raw pyramid's native rung is windowed and lazy, so
    materialising it whole only to stride it (the old path) decoded every FOV at stride 1
    and held the whole native plane; the level AT the export's decimation is the same
    pixels, already strided (every rung is bit-exact to the fuser at its stride)."""
    from squidxplorer._napari_view import pyramid_levels

    levels = pyramid_levels(data)
    if not levels:
        return data
    for level in levels:
        shp = getattr(level, "shape", None) or ()
        if shp and max(int(v) for v in shp[-2:]) <= int(max_px):
            return level
    return levels[-1]


def write_png(rgb: np.ndarray, out_path) -> str:
    """Write ``(H, W, 3)`` uint8 RGB to *out_path* losslessly. Returns the path written."""
    problem = png_problem()
    if problem:
        raise RuntimeError(f"cannot write {out_path}: {problem}")
    from PIL import Image

    out_path = str(out_path)
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), mode="RGB").save(
        out_path, format="PNG")
    return out_path
