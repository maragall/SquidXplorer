"""The 3D/volume view of a RegionViewer: in-window bricked volumes, native popouts, one at a time.

Extracted from ``_region_viewer`` (2026-08-14). The rules here are the measured ones CLAUDE.md
defends and they move VERBATIM — 3D paints into the window's own napari canvas; which layer 3D
shows comes off the DECLARATION (``MosaicLayers.visible_op`` / ``_reduces_z``, never an operator
name); LUTs are harvested from the layer being RENDERED (:func:`on_screen_luts` on the source),
never from raw; a path that renders the LAYER's pixels takes the layer's pitch
(:func:`displayed_pitch_um`, refusing by name) while a drawn ROI's read path counts in
acquisition pixels; ``_bricks.py`` is untouched.

**The close-before-read invariant is STRUCTURAL here.** While a volume is up, ``MosaicLayers``
describes BRICKS, so reading the scene before closing the old volume harvests contrast off one
brick (the measured 1-of-9 defect). :func:`open_3d` closes first and then dispatches into the
scene-reading paths (:func:`on_screen_luts`, :func:`volume_source`, :func:`open_roi_3d`) through
MODULE-INTERNAL calls — there is no public entry to those reads that does not pass the close.

Functions over the window (the ``_ingest`` precedent), with ``_native3d`` and ``_roi_bbox``
staying ON the window: ``tests/test_stitch_in_3d`` borrows these functions unbound onto a duck
shell that holds both as plain attributes, and ``closeEvent`` clears ``_native3d`` directly.
``RegionViewer`` keeps thin delegates for every name tests reach.
"""

from __future__ import annotations

import numpy as np

from squidxplorer._napari_view import full_res_level

_RAW_OP = "raw"


def _brick_budget_bytes() -> int:
    """How much a bricked 3D view may hold resident."""
    try:
        from squidxplorer._budget import cache_budget

        return int(cache_budget())
    except Exception:                                    # noqa: BLE001 - a floor beats no render
        return 512 << 20


def _started(vol):
    """``open()`` the volume and hand it back, so ``replace_native3d`` still takes one callable."""
    vol.open()
    return vol


def replace_native3d(win, open_it) -> None:
    """ONE 3D popout per window: close the one this window already has, then open the new one."""
    close_native3d(win)
    win._native3d = open_it()


def close_native3d(win) -> None:
    """Take this window's 3D view down, and put the 2D scene back. Idempotent."""
    old, win._native3d = win._native3d, None
    if old is None:
        return
    close = getattr(old, "close", None)
    if callable(close):
        try:
            close()
        except Exception:                        # noqa: BLE001 - already-closed / no Qt window
            pass


def open_3d(win) -> None:
    """3D = THIS view at NATIVE resolution, read STRAIGHT FROM THE READER (gallery-view recipe).

    Closes any volume already up BEFORE the scene is read — while one is up, ``MosaicLayers``
    describes bricks, so a read before the close harvests contrast and sources off one brick
    (measured: 1 of 9 bricks yielded voxels, contrast ``(0.0, 1.0)`` against ``(120, 900)`` on
    screen). The reading paths below are module-internal on purpose: they cannot be reached
    without passing this close.
    """
    win.set_render_mode("3d")
    region = win._cursor.region if win._cursor is not None else (
        win._regions[0] if win._regions else None)
    if region is None or win._reader is None or win._meta is None:
        win._say("no region to render in 3D.")
        return
    roi_bbox = win._roi_bbox
    if roi_bbox is None:
        sel_bbox, sel_region = win._selected_roi()
        if sel_bbox is not None and sel_region is not None:
            roi_bbox, region = sel_bbox, sel_region
    close_native3d(win)
    if roi_bbox is not None:
        open_roi_3d(win, region, roi_bbox)
        return

    fov = win._roi_center_fov(region, roi_bbox)
    from squidxplorer._napari3d import open_native_3d

    contrast_by, colormap_by = on_screen_luts(win, _RAW_OP)
    try:
        replace_native3d(win, lambda: open_native_3d(
            win._reader, win._meta, region, fov=fov,
            contrast_by_channel=contrast_by or None,
            colormap_by_channel=colormap_by or None,
        ))
    except Exception as exc:                     # noqa: BLE001 - named to the window, never silent
        win._say(f"3D could not open: {exc}")


def on_screen_luts(win, op: str) -> "tuple[dict, dict]":
    """``(contrast_by_channel, colormap_by_channel)`` as *op*'s layers are showing them."""
    mosaic = getattr(win._pane, "mosaic", None) if win._pane is not None else None
    contrast_by: dict = {}
    colormap_by: dict = {}
    if mosaic is None:
        return contrast_by, colormap_by
    for c in (win._meta or {}).get("channels", []):
        name = c["name"]
        layer = mosaic.find(str(op), name)
        if layer is None:
            continue
        try:
            contrast_by[name] = tuple(layer.contrast_limits)
        except Exception:                        # noqa: BLE001
            pass
        try:
            cmap = layer.colormap
            colormap_by[name] = getattr(cmap, "name", cmap)
        except Exception:                        # noqa: BLE001
            pass
    return contrast_by, colormap_by


def open_roi_3d(win, region: str, roi_bbox: tuple) -> None:
    """3D of an ROI, BRICKED and IN THIS WINDOW. Any ROI renders; none is refused."""
    names = [c["name"] for c in (win._meta or {}).get("channels", [])]
    if not names:
        win._say("this acquisition declares no channels to render in 3D.")
        return
    mosaic = getattr(win._pane, "mosaic", None) if win._pane is not None else None
    if mosaic is None or getattr(mosaic, "model", None) is None:
        win._say("3D needs this window's napari canvas, which isn't available here.")
        return
    from squidxplorer import _bricks
    from squidxplorer._brick_view import BrickedVolume
    from squidxplorer._napari3d import region_origin_um, roi_window_px, z_step_um
    from squidxplorer._napari_view import _DEFAULT_MAX_3D_TEXTURE

    window = roi_window_px(win._meta or {}, region, roi_bbox)
    origin = region_origin_um(win._meta or {}, region)
    if window is None or origin is None:
        win._say("ROI 3D: this ROI does not land on any FOV of this region.")
        return
    nz = len(list((win._meta or {}).get("z_levels") or [0]))
    if nz < 2:
        win._say("3D needs a z-stack; this acquisition has a single z plane.")
        return
    px = float((win._meta or {}).get("pixel_size_um") or 1.0)
    dz = z_step_um(win._meta or {}, px, where=f"3D ROI {region}")
    max_tex = _DEFAULT_MAX_3D_TEXTURE
    try:
        max_tex = int(win._pane._live_max_3d_texture())
    except Exception:                                # noqa: BLE001 - the Apple value is the floor
        pass
    read, source, src_pitch = volume_source(win, window)
    if source is None:
        return
    contrast_by, colormap_by = on_screen_luts(win, source)
    r0, r1, c0, c1 = window
    roi_origin = (0.0, float(origin[1]) + r0 * px, float(origin[0]) + c0 * px)
    budget = _brick_budget_bytes()
    try:
        replace_native3d(win, lambda: _started(BrickedVolume(
            mosaic, win._reader, win._meta, region, window,
            channels=names, scale=(dz, px, px), origin_um=roi_origin,
            limit=max_tex, budget_bytes=budget,
            op=source,
            contrast_by=contrast_by or None, colormap_by=colormap_by or None,
            say=win._say, parent=win, read=read,
        )))
    except Exception as exc:                         # noqa: BLE001 - named to the window
        win._say(f"ROI 3D could not open: {exc}")
        return
    try:
        win._pane.on_camera_settled(win._refresh_bricks)
    except Exception:                                # noqa: BLE001 - static bricks still render
        pass
    vol = win._native3d
    n = getattr(vol, "brick_count", 0)
    if src_pitch is None:
        voxels = (f"Voxels at {px:.3f} um/px, the acquisition's own — read straight from the "
                  f"reader.")
    else:
        spy, spx = src_pitch
        coarser = ("" if max(spy, spx) <= px * 1.001 else
                   f" — COARSER than the acquisition's {px:.3f} um/px, because a displayed "
                   f"operator layer is the fused preview at its own decimation")
        voxels = (f"Voxels at {spy:.3f} x {spx:.3f} um/px, read off the '{source}' "
                  f"layer{coarser}.")
    win._say(f"3D in-window: '{source}', {(r1 - r0)}x{(c1 - c0)} px ROI, {nz} z, "
             f"{len(names)} channel(s), {n} texture{'' if n == 1 else 's'}. {voxels} "
             f"{_bricks.ceiling_line(max_tex, px, measured=True)}")


def volume_source(win, window: tuple):
    """WHICH volume 3D renders: the operator layer this window is SHOWING, or raw."""
    mosaic = getattr(win._pane, "mosaic", None) if win._pane is not None else None
    if mosaic is None:
        return None, _RAW_OP, None
    try:
        op = mosaic.visible_op()
    except Exception:                                # noqa: BLE001 - fall back to raw
        return None, _RAW_OP, None
    if not op or op == _RAW_OP:
        return None, _RAW_OP, None
    try:
        if mosaic._reduces_z(op):
            win._say(f"3D: '{op}' reduces z to a single plane, so it has no volume to render. "
                     f"Show raw (or a z-preserving operator) and click 3D again.")
            return None, None, None
    except Exception:                                # noqa: BLE001 - undeclared: try to render
        pass
    px = float((win._meta or {}).get("pixel_size_um") or 1.0)
    origin = None
    try:
        from squidxplorer._napari3d import region_origin_um

        origin = region_origin_um(win._meta or {}, win.current_region())
    except Exception:                                # noqa: BLE001
        pass
    if origin is None:
        win._say(f"3D: '{op}' cannot be placed — this region has no stage positions.")
        return None, None, None
    srcs: dict = {}
    for ch in mosaic.channels(op):
        layer = mosaic.find(op, ch)
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
        win._say(f"3D: '{op}' is on screen but carries no z depth here, so there is no volume "
                 f"to render.")
        return None, None, None
    ox_um, oy_um = float(origin[0]), float(origin[1])
    pitch = tuple(next(iter(srcs.values()))[2])

    def _read(brick, channel, step, should_stop):
        """One brick out of the OPERATOR's on-screen volume, same contract as the raw reader."""
        got = srcs.get(channel)
        if got is None or (should_stop is not None and should_stop()):
            return None
        src, (ty, tx), (sy, sx) = got
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

    return _read, op, pitch


def refresh_bricks(win) -> None:
    """The camera stopped: re-decide stride and visible set. No-op unless 3D bricks are up."""
    vol = win._native3d
    refresh = getattr(vol, "refresh", None)
    if not callable(refresh):
        return
    try:
        refresh()
    except Exception as exc:                         # noqa: BLE001 - named, never silent
        win._say(f"3D: could not follow the camera ({exc}).")


def displayed_pitch_um(win, layer, *, what: str):
    """MICROMETRES PER PIXEL of the pixels a layer is actually showing, as ``(y, x)``."""
    scale = getattr(layer, "scale", None)
    try:
        py, px = (float(v) for v in tuple(scale)[-2:])
    except (TypeError, ValueError, IndexError):
        win._say(f"3D refused: {what} carries no napari scale, so the micrometres per "
                 f"displayed pixel are unknown. The acquisition's pixel_size_um is NOT that "
                 f"number — the mosaic on screen is fused at its own decimation — so "
                 f"there is nothing honest to render this volume at.")
        return None
    if not (py > 0 and px > 0):
        win._say(f"3D refused: {what} is placed at a scale of ({py}, {px}) um/px, which is "
                 f"not a pitch. A volume cannot be given a physical size from it.")
        return None
    return (py, px)


def render_roi_volume(win, mosaic, contrast_by: dict, colormap_by: dict) -> None:
    """Render the ROI subarray in 3D from the 2D layer's own pixels; only tests reach this path."""
    volumes: dict = {}
    pitch = None
    for c in (win._meta or {}).get("channels", []):
        name = c["name"]
        layer = mosaic.find(_RAW_OP, name)
        if layer is None:
            continue
        if pitch is None:
            pitch = displayed_pitch_um(win, layer, what=f"the raw '{name}' mosaic layer")
            if pitch is None:
                return
        level0 = full_res_level(layer.data)
        if getattr(level0, "ndim", 0) < 3 or int(level0.shape[0]) < 2:
            win._say("3D needs a z-stack; this ROI has a single z plane.")
            return
        volumes[name] = level0
    if not volumes:
        win._say("no channel on screen to render in 3D.")
        return
    py_um, px_um = pitch
    max_tex = 2048
    try:
        max_tex = int(win._pane._live_max_3d_texture())
    except Exception:                                # noqa: BLE001 - Apple default is the floor
        pass
    from squidxplorer._napari3d import open_native_3d_volume, z_step_um

    dz = z_step_um(win._meta or {}, py_um, where="3D ROI volume")

    try:
        replace_native3d(win, lambda: open_native_3d_volume(
            {n: np.asarray(v) for n, v in volumes.items()},
            scale=(dz, py_um, px_um),
            title=f"3D ROI — {win._view_label(win._regions)}",
            contrast_by_channel=contrast_by or None,
            colormap_by_channel=colormap_by or None,
            max_texture=max_tex,
        ))
    except Exception as exc:                         # noqa: BLE001 - named to the window
        win._say(f"ROI 3D could not open: {exc}")
