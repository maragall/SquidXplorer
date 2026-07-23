"""Native-resolution napari 3D, adopting hongquanli/gallery-view's recipe (not reinventing it).

WHY THIS EXISTS. napari renders 3D from ONE GL texture and refuses any axis over
GL_MAX_3D_TEXTURE_SIZE (~2048 on Apple GPUs), so handing it a fused REGION mosaic (5731 px) forces
napari's own crude downsample and the volume looks blocky. gallery-view sidesteps this the only way
that works: it feeds napari a SINGLE NATIVE ZYX STACK (one FOV / acquisition, ~2084 px) that fits
the texture, single-scale, with a micrometre voxel scale and the LUT carried over. That is native
resolution because the volume never exceeds the texture. We cannot import gallery-view (it pins
napari <0.6, we run 0.6.6), so this replicates its recipe: the exact add_image call, the
(dz, px, px) scale, additive blending, carried-over contrast, and a micrometre scale bar.

A single FOV, not the whole region, is deliberate: the region is a mosaic and cannot fit one
texture at native resolution. This is the "max res preview" of one field; AGAVE remains the path
for a path-traced, whole-region volume.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

log = logging.getLogger("squidmip.napari3d")


def _center_fov(meta: dict, region: str) -> Optional[int]:
    """The FOV nearest the region's stage centroid, so the 3D preview lands on representative
    tissue rather than a corner. Falls back to the first FOV when positions are unavailable."""
    fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
    if not fovs:
        return None
    positions = meta.get("fov_positions_um") or {}
    pts = [(f, positions.get((region, f))) for f in fovs]
    pts = [(f, p) for f, p in pts if p is not None]
    if not pts:
        return int(fovs[0])
    cx = float(np.mean([p[0] for _f, p in pts]))
    cy = float(np.mean([p[1] for _f, p in pts]))
    return int(min(pts, key=lambda fp: (fp[1][0] - cx) ** 2 + (fp[1][1] - cy) ** 2)[0])


def _native_stack(reader: Any, meta: dict, region: str, fov: int, channel: str) -> np.ndarray:
    """One FOV's native (z, y, x) stack for a channel. Reads only this field's planes."""
    z_levels = list(meta.get("z_levels") or [0])
    planes = []
    for z in z_levels:
        plane = np.asarray(reader.read(region, fov, channel, int(z)))
        if plane.ndim != 2:
            plane = plane.reshape(plane.shape[-2:])
        planes.append(plane)
    return np.stack(planes, axis=0) if len(planes) > 1 else planes[0][None, ...]


def _auto_clim(stack: np.ndarray) -> Optional[tuple]:
    """Contrast for a channel whose on-screen LUT was NOT carried in.

    Without this, napari autoscales an un-supplied channel to its full data range and a
    fluorescence volume renders as washed-out noise -- Julio's "contrast messes up". We use the
    app's own fluorescence rule (mode + 2 sigma to black, 99.9th pct on top) so the 3D view matches
    what 2D shows, and fall back to gallery-view's plain (1, 99.9) percentile pair only if that
    helper is somehow unavailable. Never returns a raw full-range window."""
    try:
        from squidmip._contrast import auto_contrast

        win = auto_contrast(stack)
        if win is not None:
            return (float(win[0]), float(win[1]))
    except Exception:                                   # noqa: BLE001 - fall through to percentile
        pass
    try:
        return (float(np.percentile(stack, 1)), float(np.percentile(stack, 99.9)))
    except Exception:                                   # noqa: BLE001 - let napari autoscale
        return None


def _add_bounding_box(viewer: Any, scale: tuple, shape_zyx: tuple) -> None:
    """gallery-view's micrometre bounding box with 100 um ticks, so the volume reads at scale."""
    nz, ny, nx = shape_zyx
    z_max, y_max, x_max = nz * scale[0], ny * scale[1], nx * scale[2]
    edges = [
        [[0, 0, 0], [0, 0, x_max]], [[0, 0, x_max], [0, y_max, x_max]],
        [[0, y_max, x_max], [0, y_max, 0]], [[0, y_max, 0], [0, 0, 0]],
        [[z_max, 0, 0], [z_max, 0, x_max]], [[z_max, 0, x_max], [z_max, y_max, x_max]],
        [[z_max, y_max, x_max], [z_max, y_max, 0]], [[z_max, y_max, 0], [z_max, 0, 0]],
        [[0, 0, 0], [z_max, 0, 0]], [[0, 0, x_max], [z_max, 0, x_max]],
        [[0, y_max, x_max], [z_max, y_max, x_max]], [[0, y_max, 0], [z_max, y_max, 0]],
    ]
    tick = min(z_max, y_max, x_max) * 0.02
    ticks: list = []
    for x in np.arange(100, x_max, 100):
        ticks += [[[0, 0, x], [0, tick, x]], [[0, 0, x], [tick, 0, x]]]
    for y in np.arange(100, y_max, 100):
        ticks += [[[0, y, 0], [0, y, tick]], [[0, y, 0], [tick, y, 0]]]
    for z in np.arange(100, z_max, 100):
        ticks += [[[z, 0, 0], [z, tick, 0]], [[z, 0, 0], [z, 0, tick]]]
    viewer.add_shapes(
        [np.array(line) for line in edges + ticks],
        shape_type="line", edge_color="white", edge_width=2,
        name="Bounding Box (100um ticks)",
    )


def _wire_close_to_release_memory(viewer: Any) -> None:
    """Drop the multi-GB stacks when the 3D popout is closed (gallery-view's memory patch).

    napari keeps every Viewer in a global set and vispy holds a C-level ref to each layer's array
    via the volume it uploaded to the GPU, so X-ing the window does NOT free the stacks -- they
    linger in RSS for the life of the app, which fights the memory bar Spencer asked for. Swapping
    each layer's data for a 1x1x1 stub forces vispy to release the buffer, then clear + gc reclaims
    it."""
    import gc

    try:
        qt_window = viewer.window._qt_window
    except Exception:                                   # noqa: BLE001 - no Qt window, nothing to wire
        return
    original = qt_window.closeEvent

    def _close_and_release(event) -> None:
        try:
            for layer in list(viewer.layers):
                data = getattr(layer, "data", None)
                if isinstance(data, np.ndarray):
                    layer.data = np.zeros((1, 1, 1), dtype=data.dtype)
            viewer.layers.clear()
        except Exception:                               # noqa: BLE001 - best-effort reclaim
            pass
        original(event)
        gc.collect()

    qt_window.closeEvent = _close_and_release


def open_native_3d(
    reader: Any,
    meta: dict,
    region: str,
    *,
    fov: Optional[int] = None,
    channels: Optional[Sequence[str]] = None,
    contrast_by_channel: Optional[dict] = None,
    colormap_by_channel: Optional[dict] = None,
) -> Any:
    """Open a fresh napari 3D viewer on ONE FOV's native z-stack (gallery-view's recipe).

    Returns the napari ``Viewer`` (a popout window). Raises with a named reason if the stack cannot
    be built, so the caller can route it to the log rather than a silent no-op.
    """
    import napari  # lazy: heavy import, and a machine without napari still runs the 2D app

    fov = _center_fov(meta, region) if fov is None else int(fov)
    if fov is None:
        raise ValueError(f"region {region!r} has no FOVs to render in 3D.")
    names = list(channels) if channels else [c["name"] for c in meta.get("channels", [])]
    if not names:
        raise ValueError("this acquisition declares no channels to render.")

    px = float(meta.get("pixel_size_um") or 1.0)
    dz = float(meta.get("dz_um") or px)                 # z step in um; fall back to xy if absent
    contrast_by_channel = contrast_by_channel or {}
    colormap_by_channel = colormap_by_channel or {}

    title = f"3D native (napari) — {region} / fov {fov}"
    viewer = napari.Viewer(ndisplay=3, title=title)
    n_z = 1
    first_shape: Optional[tuple] = None
    for ch in names:
        try:
            stack = _native_stack(reader, meta, region, fov, ch)
        except Exception as exc:                        # noqa: BLE001 - named, then continue
            log.error("3D native: could not read %s/%s/fov %s: %s", region, ch, fov, exc)
            continue
        n_z = max(n_z, int(stack.shape[0]))
        kwargs = {
            "name": ch,
            "scale": (dz, px, px),                      # (z, y, x) micrometres, gallery-view style
            "blending": "additive",
            "rendering": "mip",
        }
        cmap = colormap_by_channel.get(ch)
        if cmap is not None:
            kwargs["colormap"] = cmap
        # Carry the on-screen LUT; if this channel had none, derive one (never raw full-range,
        # which is how a fluorescence volume "messes up" -- washed out. See _auto_clim.)
        clim = contrast_by_channel.get(ch)
        if clim is None:
            clim = _auto_clim(stack)
        if clim is not None:
            kwargs["contrast_limits"] = tuple(clim)
        viewer.add_image(stack, **kwargs)
        if first_shape is None:
            first_shape = tuple(stack.shape)

    if not viewer.layers:
        viewer.close()
        raise ValueError(f"{region}/fov {fov}: no channel could be read, so there is no 3D volume.")

    # Micrometre scale bar + a 100 um-tick bounding box + a title overlay, exactly like gallery-view.
    if first_shape is not None:
        try:
            _add_bounding_box(viewer, (dz, px, px), first_shape)
        except Exception:                               # noqa: BLE001 - overlay is cosmetic
            pass
    try:
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "um"
        viewer.text_overlay.visible = True
        viewer.text_overlay.text = title
        viewer.text_overlay.font_size = 12
        viewer.text_overlay.color = "white"
        viewer.text_overlay.position = "top_center"
    except Exception:                                   # noqa: BLE001 - cosmetic
        pass
    # Free the native stacks the instant the popout is closed, so it does not hold RSS for the app's
    # life (Spencer's memory brief). gallery-view's own patch.
    _wire_close_to_release_memory(viewer)
    log.info("3D native: opened %s / fov %s, %d channel(s), %d z at native %.3f um/px, dz %.2f um",
             region, fov, len(viewer.layers), n_z, px, dz)
    return viewer
