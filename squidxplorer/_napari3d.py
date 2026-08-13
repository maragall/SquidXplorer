"""Native-resolution napari 3D: single-FOV z-stacks and bricked ROI volumes that stay under
the GL 3D texture limit, following gallery-view's recipe."""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Sequence

import numpy as np

from squidxplorer._logpane import get_logger

log = get_logger("napari3d")


def _center_fov(meta: dict, region: str) -> Optional[int]:
    """The FOV nearest the region's stage centroid; the first FOV when positions are unavailable."""
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


def z_step_um(meta: dict, px: float, where: str = "3D") -> float:
    """The acquisition's z step in micrometres, or the xy pixel size with a LOUD warning."""
    dz = meta.get("dz_um")
    if dz:
        return float(dz)
    log.warning(
        "%s: this acquisition has dz_um=%r, so the z step is UNKNOWN and the volume is being "
        "drawn with z=%.4f um (the xy pixel size) as a stand-in. Z proportions are NOT to scale.",
        where, dz, float(px),
    )
    return float(px)


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
    """Contrast for a channel whose on-screen LUT was NOT carried in; never a raw full-range window."""
    try:
        from squidxplorer._contrast import auto_contrast

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

    vispy holds a C-level ref to each layer's array, so closing alone does not free them;
    swapping each layer's data for a 1x1x1 stub forces the release.
    """
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


def pin_max_compositing(viewer: Any, layer: Any) -> bool:
    """Composite this layer into the canvas with the GL **max** equation, and KEEP it there.

    MIP is a maximum, so GL max compositing makes bricked and unbricked the same picture, where
    additive sums brick MIPs at the joins. napari's canvas re-applies blending on every layer
    insert/reorder/visibility change, so the equation is re-pinned by wrapping the exact method
    napari calls.
    """
    try:
        visual = viewer.window._qt_viewer.canvas.layer_to_visual[layer]
    except Exception:                                   # noqa: BLE001 - no canvas (headless/tests)
        return False
    if getattr(visual, "_squid_max_pinned", False):
        return True
    original = visual._on_blending_change

    def _keep_max(event=None) -> None:
        original(event)
        visual.node.set_gl_state(depth_test=False, cull_face=False, blend=True,
                                 blend_equation="max")
        visual.node.update()

    try:
        visual._on_blending_change = _keep_max
        visual._squid_max_pinned = True
        _keep_max()
    except Exception as exc:                            # noqa: BLE001 - named; seams, not a crash
        log.warning("brick compositing: could not pin GL max on %s (%s); joins may show as "
                    "bright lines under rotation.", getattr(layer, "name", "layer"), exc)
        return False
    return True


def region_origin_um(meta: dict, region: str) -> Optional[tuple]:
    """The region mosaic's top-left in stage micrometres — the same origin ``mosaic_bbox_um`` uses."""
    positions = meta.get("fov_positions_um") or {}
    fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
    if not fovs or not positions:
        return None
    try:
        xs = [float(positions[(region, f)][0]) for f in fovs]
        ys = [float(positions[(region, f)][1]) for f in fovs]
    except (KeyError, TypeError, IndexError):
        return None
    return (min(xs), min(ys))


def roi_window_px(meta: dict, region: str, roi_bbox_um: Sequence[float]) -> Optional[tuple]:
    """An ROI box in stage um -> ``(r0, r1, c0, c1)`` LEVEL-0 mosaic pixels, clipped to the region."""
    from squidxplorer._placement import fov_offsets_px, mosaic_extent_px

    origin = region_origin_um(meta, region)
    px = float(meta.get("pixel_size_um") or 0.0)
    if origin is None or px <= 0:
        return None
    x0, y0 = origin
    try:
        rx0, ry0, rx1, ry1 = (float(v) for v in roi_bbox_um)
    except (TypeError, ValueError):
        return None
    try:
        positions = meta.get("fov_positions_um") or {}
        fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
        offsets = fov_offsets_px(positions, region, fovs, px)
        h_px, w_px = mosaic_extent_px(offsets, tuple(int(v) for v in meta["frame_shape"]))
    except Exception:                                   # noqa: BLE001 - fall back to the box itself
        w_px = h_px = None
    c0, c1 = int(round((min(rx0, rx1) - x0) / px)), int(round((max(rx0, rx1) - x0) / px))
    r0, r1 = int(round((min(ry0, ry1) - y0) / px)), int(round((max(ry0, ry1) - y0) / px))
    c0, r0 = max(0, c0), max(0, r0)
    if w_px:
        c1 = min(int(w_px), c1)
    if h_px:
        r1 = min(int(h_px), r1)
    if c1 <= c0 or r1 <= r0:
        return None
    return (r0, r1, c0, c1)


def _plane_cache():
    """The bounded LRU that stops adjacent bricks re-decoding the same FOV plane.

    Built under a lock: every caller is a ``_BrickLoader`` QThread, and two racing builders
    would silently double the byte bound.
    """
    global _PLANES
    with _PLANES_LOCK:
        if _PLANES is None:
            from squidxplorer._budget import cache_budget
            from squidxplorer._mosaic_source import MemoryBoundedLRUCache

            _PLANES = MemoryBoundedLRUCache(max(64 << 20, int(cache_budget()) // 2))
        return _PLANES


#: Built on first use so importing this module costs no memory measurement.
_PLANES: Any = None
_PLANES_LOCK = threading.Lock()


def _read_plane(reader: Any, region: str, fov: int, channel: str, z_level: int) -> np.ndarray:
    """One decoded FOV plane, from the shared bounded cache when it is already there."""
    cache = _plane_cache()
    try:
        from squidxplorer._mosaic_source import _source_token

        key = (_source_token(reader), region, int(fov), channel, int(z_level))
    except Exception:                                   # noqa: BLE001 - uncacheable reader: read it
        key = None
    if key is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit
    frame = np.asarray(reader.read(region, int(fov), channel, int(z_level)))
    if frame.ndim != 2:
        frame = frame.reshape(frame.shape[-2:])
    if key is not None:
        try:
            cache.put(key, frame)
        except ValueError:                              # noqa: PERF203 - a plane over the whole
            pass                                        # budget: read it every time, never crash
    return frame


def read_brick(reader: Any, meta: dict, region: str, window: Sequence[int], channel: str, *,
               step: int = 1, should_stop: Optional[Any] = None) -> Optional[np.ndarray]:
    """ONE brick's ``(z, y, x)`` voxels, fused across the FOVs it overlaps, strided by *step*.

    *window* is ``(r0, r1, c0, c1)`` in level-0 mosaic pixels (what :func:`roi_window_px` returns).
    Returns None when stopped, so a cancelled brick cannot be cached as the answer.
    """
    from squidxplorer._placement import fov_offsets_px

    positions = meta.get("fov_positions_um") or {}
    px = float(meta.get("pixel_size_um") or 0.0)
    fovs = list((meta.get("fovs_per_region") or {}).get(region) or [])
    if not fovs or not positions or px <= 0:
        return None
    r0, r1, c0, c1 = (int(v) for v in window)
    fh, fw = (int(v) for v in meta["frame_shape"])
    offsets = fov_offsets_px(positions, region, fovs, px)
    z_levels = list(meta.get("z_levels") or [0])
    nz = len(z_levels)
    vol: Optional[np.ndarray] = None
    for f in fovs:
        if should_stop is not None and should_stop():
            return None
        fr, fc = offsets[f]
        ir0, ir1 = max(r0, fr), min(r1, fr + fh)        # FOV window ∩ brick window
        ic0, ic1 = max(c0, fc), min(c1, fc + fw)
        if ir1 <= ir0 or ic1 <= ic0:
            continue
        for zi, z in enumerate(z_levels):
            if should_stop is not None and should_stop():
                return None
            frame = _read_plane(reader, region, int(f), channel, int(z))
            if vol is None:
                vol = np.zeros((nz, r1 - r0, c1 - c0), dtype=frame.dtype)
            vol[zi, ir0 - r0:ir1 - r0, ic0 - c0:ic1 - c0] = frame[ir0 - fr:ir1 - fr,
                                                                   ic0 - fc:ic1 - fc]
    if vol is None:
        return None
    s = max(1, int(step))
    return np.ascontiguousarray(vol[:, ::s, ::s]) if s > 1 else vol


def open_native_3d(
    reader: Any,
    meta: dict,
    region: str,
    *,
    fov: Optional[int] = None,
    channels: Optional[Sequence[str]] = None,
    contrast_by_channel: Optional[dict] = None,
    colormap_by_channel: Optional[dict] = None,
    roi_bbox_um: Optional[Sequence[float]] = None,
) -> Any:
    """Open a fresh napari 3D viewer on ONE FOV's native z-stack (gallery-view's recipe).

    ``roi_bbox_um`` crops each channel's stack to the ROI's window within the FOV. Raises with a
    named reason if the stack cannot be built.
    """
    import napari  # lazy: heavy import, and a machine without napari still runs the 2D app

    fov = _center_fov(meta, region) if fov is None else int(fov)
    if fov is None:
        raise ValueError(f"region {region!r} has no FOVs to render in 3D.")
    names = list(channels) if channels else [c["name"] for c in meta.get("channels", [])]
    if not names:
        raise ValueError("this acquisition declares no channels to render.")

    px = float(meta.get("pixel_size_um") or 1.0)
    dz = z_step_um(meta, px, where=f"3D native {region}/fov {fov}")
    contrast_by_channel = contrast_by_channel or {}
    colormap_by_channel = colormap_by_channel or {}

    # ROI window in this FOV's own pixels (its stage position is the FOV's top-left um).
    crop = None
    if roi_bbox_um is not None:
        p = (meta.get("fov_positions_um") or {}).get((region, fov))
        if p is not None:
            try:
                fx, fy = float(p[0]), float(p[1])
                rx0, ry0, rx1, ry1 = (float(v) for v in roi_bbox_um)
                fh, fw = (int(v) for v in meta["frame_shape"])
                c0 = max(0, int(round((rx0 - fx) / px)))
                c1 = min(fw, int(round((rx1 - fx) / px)))
                r0 = max(0, int(round((ry0 - fy) / px)))
                r1 = min(fh, int(round((ry1 - fy) / px)))
                if c1 > c0 and r1 > r0:
                    crop = (r0, r1, c0, c1)
            except Exception:                           # noqa: BLE001 - fall back to the whole FOV
                crop = None

    title = f"3D native (napari) — {region} / fov {fov}" + (" / ROI" if crop else "")
    viewer = napari.Viewer(ndisplay=3, title=title)
    n_z = 1
    first_shape: Optional[tuple] = None
    for ch in names:
        try:
            stack = _native_stack(reader, meta, region, fov, ch)
        except Exception as exc:                        # noqa: BLE001 - named, then continue
            log.error("3D native: could not read %s/%s/fov %s: %s", region, ch, fov, exc)
            continue
        if crop is not None:
            r0, r1, c0, c1 = crop
            stack = stack[..., r0:r1, c0:c1]            # exact ROI subarray, full z
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
        # Carry the on-screen LUT; if this channel had none, derive one (never raw full-range).
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
    # Free the native stacks the instant the popout is closed.
    _wire_close_to_release_memory(viewer)
    log.info("3D native: opened %s / fov %s, %d channel(s), %d z at native %.3f um/px, dz %.2f um",
             region, fov, len(viewer.layers), n_z, px, dz)
    return viewer


def open_native_3d_volume(
    volumes_by_channel: dict,
    *,
    scale: tuple,
    title: str,
    contrast_by_channel: Optional[dict] = None,
    colormap_by_channel: Optional[dict] = None,
    max_texture: int = 2048,
) -> Any:
    """Render READY per-channel ``(z, y, x)`` volumes in a napari 3D popout at NATIVE resolution.

    A volume over the GL texture limit is tiled into bricks composed with GL max, never
    downsampled and never refused. Takes volumes already in memory; the in-window path
    (``_brick_view.BrickedVolume``) is what a whole region must use.
    """
    import napari

    from squidxplorer import _bricks

    contrast_by_channel = contrast_by_channel or {}
    colormap_by_channel = colormap_by_channel or {}
    vols = {name: np.asarray(v) for name, v in volumes_by_channel.items() if v is not None}
    if not vols:
        raise ValueError("no channel volume to render in 3D.")
    first = next(iter(vols.values()))
    if first.ndim != 3:
        raise ValueError(f"a 3D volume must be (z, y, x); got shape {first.shape}.")
    nz, y, x = (int(v) for v in first.shape)
    limit = int(max_texture)
    single = _bricks.fits_single_texture(y, x, nz, limit)
    bricks = _bricks.plan(y, x, limit=limit,
                          edge=(limit if single else _bricks.DEFAULT_BRICK_EDGE))

    viewer = napari.Viewer(ndisplay=3, title=title)
    for name, vol in vols.items():
        clim = contrast_by_channel.get(name)
        if clim is None:
            clim = _auto_clim(vol)                      # ONE window for the whole channel, so the
        cmap = colormap_by_channel.get(name)            # bricks cannot step in brightness
        for b in bricks:
            kwargs = {
                "name": name if single else f"{name} ▪ {b.iy},{b.ix}",
                "scale": scale,
                "translate": (0.0, b.r0 * float(scale[1]), b.c0 * float(scale[2])),
                "blending": "additive",
                "rendering": "mip",
            }
            if cmap is not None:
                kwargs["colormap"] = cmap
            if clim is not None:
                kwargs["contrast_limits"] = tuple(clim)
            layer = viewer.add_image(vol[:, b.r0:b.r1, b.c0:b.c1], **kwargs)
            if not single:
                pin_max_compositing(viewer, layer)
    if len(bricks) > 1:
        log.info("3D volume: %dx%d px is over the %d px texture limit — rendered as %d brick(s) "
                 "per channel at NATIVE resolution (GL max compositing, no seam).",
                 y, x, limit, len(bricks))
    if not viewer.layers:
        viewer.close()
        raise ValueError("no channel could be rendered, so there is no 3D volume.")
    try:
        _add_bounding_box(viewer, scale, first.shape)
    except Exception:                                   # noqa: BLE001 - overlay is cosmetic
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
    _wire_close_to_release_memory(viewer)
    log.info("3D native volume: %d channel(s), shape %s, scale %s", len(vols), first.shape, scale)
    return viewer
