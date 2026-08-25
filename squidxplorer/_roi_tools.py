"""ROI drawing, clamping and the ROI child windows of a RegionViewer.

Extracted from ``_region_viewer`` (2026-08-14). The rules this cluster owns are the ones
CLAUDE.md defends and they move INTACT:

* **The clamp happens at DRAWING time** (:func:`clamp_last_roi` <- ``_bricks.clamp_bbox_um``),
  against the live ``GL_MAX_3D_TEXTURE_SIZE`` — queried, never hardcoded — so anything drawable
  is renderable from one texture and the limit is felt while drawing, not refused afterwards.
* **The clamp and the cost line count in ACQUISITION pixels** (``meta["pixel_size_um"]``),
  because a drawn ROI's 3D reads whole FOV planes off the reader, not the decimated mosaic on
  screen (``tests/test_roi_pitch.py``). ``_bricks.py`` itself is untouched.

Functions over the window (the ``_ingest`` precedent), with the state staying ON the window
(``_roi_bbox`` — also read by the playback crop and 3D entry — plus ``_roi_layer`` and the
``_clamping`` reentrancy latch): test duck-shells set ``_roi_bbox`` directly on their stand-ins.
``RegionViewer`` keeps thin delegates because tests and the ROI chips actuate them by name.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

#: Edge colours the named ROIs cycle through (THE definition; `_region_viewer._ROI_COLORS`
#: aliases it).
ROI_COLORS: "tuple[str, ...]" = (
    "#58a6ff", "#f778ba", "#3fb950", "#f0883e", "#a371f7", "#e3b341", "#39c5cf", "#ff7b72",
)


def view_roi_2d(win) -> None:
    """Open the SELECTED ROI as a child 2D window; with no ROI picked, just show the mosaic in 2D."""
    win.set_render_mode("2d")
    bbox, _region = selected_roi(win)
    if bbox is None:
        win._set_ndisplay(2)
        return
    open_roi_children(win)


def sync_roi_width(viewer, layer, screen_px: float = 3.0) -> None:
    """Keep the ROI border a ~constant on-screen thickness as the camera zooms."""
    try:
        zoom = float(getattr(viewer.camera, "zoom", 1.0)) or 1.0
        w = max(1e-6, float(screen_px) / zoom)
        layer.edge_width = w
        layer.current_edge_width = w
    except Exception:                                # noqa: BLE001 - width is cosmetic
        pass


def roi_shapes_layer(win, create: bool = False):
    """This window's ROI Shapes layer (creating it, zoom-reactive, on first use if asked)."""
    v = win._napari_viewer()
    if v is None:
        return None, None
    layer = win._roi_layer
    if layer is None or layer not in list(v.layers):
        if not create:
            return v, None
        try:
            layer = v.add_shapes(
                name="ROIs", face_color="transparent",
                properties={"name": np.array([], dtype=object)},
                text={"string": "{name}", "color": "white", "size": 9,
                      "anchor": "upper_left"},
                edge_color="name", edge_color_cycle=list(ROI_COLORS),
            )
            layer.current_properties = {"name": np.array(["R1"], dtype=object)}
            layer.events.data.connect(
                lambda e=None, ly=layer: on_roi_data(win, ly))
        except Exception:                            # noqa: BLE001 - fall back to a plain layer
            layer = v.add_shapes(name="ROIs", edge_color="#58a6ff",
                                 face_color="transparent")
        win._roi_layer = layer
        sync_roi_width(v, layer)
        try:
            v.camera.events.zoom.connect(
                lambda e=None, vv=v, ly=layer: sync_roi_width(vv, ly))
        except Exception:                            # noqa: BLE001
            pass
    return v, layer


def on_roi_data(win, layer) -> None:
    """After a shape is added/removed: name the NEXT ROI R{n+1}, and SAY WHAT THE LAST ONE COSTS."""
    try:
        n = len(getattr(layer, "data", []) or [])
        layer.current_properties = {"name": np.array([f"R{n + 1}"], dtype=object)}
    except Exception:                                # noqa: BLE001 - labelling is cosmetic
        pass
    try:
        clamp_last_roi(win, layer)
    except Exception:                                # noqa: BLE001 - never break ROI drawing
        pass
    try:
        win._say(roi_cost_line(win, layer))
    except Exception:                                # noqa: BLE001 - the readout is advisory
        pass


def live_texture_limit(win) -> int:
    """The GPU's real GL_MAX_3D_TEXTURE_SIZE, or the documented Apple floor; never a literal here."""
    from squidxplorer._napari_view import _DEFAULT_MAX_3D_TEXTURE
    try:
        return int(win._pane._live_max_3d_texture())
    except Exception:                                # noqa: BLE001
        return int(_DEFAULT_MAX_3D_TEXTURE)


def clamp_last_roi(win, layer) -> None:
    """Hold the just-drawn ROI to what one GL texture can render, in place."""
    from squidxplorer import _bricks

    if getattr(win, "_clamping", False):
        return
    rects = list(getattr(layer, "data", []) or [])
    px = float((win._meta or {}).get("pixel_size_um") or 0.0)
    if not rects or px <= 0:
        return
    arr = np.asarray(rects[-1])
    if arr.ndim != 2 or arr.shape[0] < 4:
        return
    ys, xs = arr[:, -2].astype(float), arr[:, -1].astype(float)
    limit = live_texture_limit(win)
    from squidxplorer._conventions import AcqPitchUm

    (nx0, ny0, nx1, ny1), clamped = _bricks.clamp_bbox_um(
        (xs.min(), ys.min(), xs.max(), ys.max()), AcqPitchUm(px), limit)
    if not clamped:
        return
    new = np.array(arr, dtype=float)
    new[:, -1] = np.where(xs > xs.min(), nx1, nx0)
    new[:, -2] = np.where(ys > ys.min(), ny1, ny0)
    rects[-1] = new
    win._clamping = True
    try:
        layer.data = rects
    finally:
        win._clamping = False
    span = limit * px
    win._say(f"ROI held to the 3D ceiling: {limit} x {limit} px ({span:.0f} x {span:.0f} um) "
             f"at the acquisition's own {px:g} um/px - the largest volume this GPU renders "
             f"from one texture, and 3D reads those voxels from the reader, not from the "
             f"decimated mosaic you are drawing on.")


def roi_cost_line(win, layer) -> str:
    """"R3: 4096 x 3072 px (3080 x 2310 um) — 12 bricks on this GPU." Empty when unknowable."""
    from squidxplorer import _bricks
    from squidxplorer._napari_view import _DEFAULT_MAX_3D_TEXTURE

    rects = list(getattr(layer, "data", []) or [])
    if not rects:
        return ""
    arr = np.asarray(rects[-1])
    ys, xs = arr[:, -2], arr[:, -1]
    px = float((win._meta or {}).get("pixel_size_um") or 0.0)
    if px <= 0:
        return ""
    h_um, w_um = float(ys.max() - ys.min()), float(xs.max() - xs.min())
    h, w = int(round(h_um / px)), int(round(w_um / px))
    if h <= 0 or w <= 0:
        return ""
    limit = _DEFAULT_MAX_3D_TEXTURE
    try:
        limit = int(win._pane._live_max_3d_texture())
    except Exception:                                # noqa: BLE001 - the Apple value is the floor
        pass
    nz = len(list((win._meta or {}).get("z_levels") or [0]))
    single = _bricks.fits_single_texture(h, w, nz, limit)
    edge = limit if single else _bricks.DEFAULT_BRICK_EDGE
    n = len(_bricks.plan(h, w, limit=limit, edge=edge))
    how = ("fits ONE texture" if single
           else f"{n} bricks (over the {limit} px texture limit - bricked, not refused)")
    return f"R{len(rects)}: {h} x {w} px ({h_um:.0f} x {w_um:.0f} um), {nz} z - 3D: {how}."


def new_roi(win) -> None:
    """Start drawing an ROI rectangle inside the mosaic (deck: boxes inside the well view)."""
    v, layer = roi_shapes_layer(win, create=True)
    if v is None or layer is None:
        win._say("ROI needs the napari viewer, which isn't available here.")
        return
    try:
        v.layers.selection.active = layer
        layer.mode = "add_rectangle"
        win._say("Draw an ROI rectangle, then '→ window' to open it as a child window.")
    except Exception as exc:                         # noqa: BLE001
        win._say(f"could not start an ROI: {exc}")


def select_rois(win) -> None:
    """Enter select mode so an ROI can be clicked and deleted."""
    v, layer = roi_shapes_layer(win, create=False)
    if v is None or layer is None:
        win._say("draw an ROI first with '▭ new'.")
        return
    try:
        v.layers.selection.active = layer
        layer.mode = "select"
        win._say("Select mode: click an ROI, then press Delete/Backspace to remove it.")
    except Exception as exc:                         # noqa: BLE001
        win._say(f"could not enter select mode: {exc}")


def clear_rois(win) -> None:
    """Remove every ROI in this window."""
    v, layer = roi_shapes_layer(win, create=False)
    if v is None or layer is None or not list(getattr(layer, "data", []) or []):
        win._say("no ROIs to clear.")
        return
    try:
        layer.data = []
        win._say("cleared all ROIs.")
    except Exception as exc:                         # noqa: BLE001
        win._say(f"could not clear ROIs: {exc}")


def region_for_roi(win, bbox) -> Optional[str]:
    """The region the ROI box's centroid sits in (stage um), so an ROI child opens on that region."""
    cur = win._cursor.region if win._cursor is not None else (
        win._regions[0] if win._regions else None)
    if bbox is None:
        return cur
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    try:
        from squidxplorer._mosaic_source import mosaic_bbox_um
        for r in win._regions:
            rb = mosaic_bbox_um(win._meta, r)
            if rb is not None and rb[0] <= cx <= rb[2] and rb[1] <= cy <= rb[3]:
                return r
    except Exception:                                # noqa: BLE001 - fall back to current region
        pass
    return cur


def open_roi_children(win) -> None:
    """Open the SELECTED ROI(s) as child window(s), each scoped to the single region it sits in."""
    v = win._napari_viewer()
    layer = win._roi_layer
    rects = list(getattr(layer, "data", []) or []) if layer is not None else []
    if v is None or layer is None or layer not in list(v.layers) or not rects:
        win._say("no ROI to open - draw one with '▭ new' first.")
        return
    if win._manager is None:
        win._say(f"{len(rects)} ROI(s) drawn, but this window has no manager to open children.")
        return
    sel = sorted(int(i) for i in (getattr(layer, "selected_data", None) or set()))
    idxs = sel if sel else [len(rects) - 1]
    opened = 0
    for i in idxs:
        if i < 0 or i >= len(rects):
            continue
        bbox = None
        try:
            arr = np.asarray(rects[i])
            ys, xs = arr[:, -2], arr[:, -1]
            bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        except Exception:                            # noqa: BLE001 - a shapeless ROI still opens
            pass
        region = region_for_roi(win, bbox)
        if region is None:
            continue
        child = win._manager.open_child(
            [region], roi_bbox=bbox, parent_id=win.window_id)
        if child is not None:
            opened += 1
    win._say(f"opened {opened} ROI child window(s) on the selected ROI"
             + ("s" if opened != 1 else "") + ".")


def selected_roi(win) -> "tuple":
    """(bbox, region) of the ROI selected in this window's Shapes layer, else (None, None)."""
    layer = win._roi_layer
    v = win._napari_viewer()
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
    return bbox, region_for_roi(win, bbox)


def roi_center_fov(win, region: str, bbox: Optional[tuple] = None) -> Optional[int]:
    """The FOV nearest the ROI box's centre (stage um); None everywhere means the region centre."""
    bbox = bbox if bbox is not None else win._roi_bbox
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    positions = (win._meta or {}).get("fov_positions_um") or {}
    fovs = ((win._meta or {}).get("fovs_per_region") or {}).get(region) or []
    best, best_d = None, None
    for f in fovs:
        p = positions.get((region, int(f)))
        if p is None:
            continue
        d = (p[0] - cx) ** 2 + (p[1] - cy) ** 2
        if best_d is None or d < best_d:
            best, best_d = int(f), d
    return best
