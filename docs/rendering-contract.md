# Rendering contract: the mosaic pyramid, 2D/3D, and max-resolution

This is the contract that keeps rendering correct. The recurring failure ("the AI messes up the
rendering because it can't understand the nature of the pyramid") is always the same root cause:
handing napari data at the wrong abstraction. Read this before touching any render path.

## The data structure: a lazy multiscale pyramid

A fused region mosaic is NOT a single array. `squidmip._mosaic_source.fuse_region_pyramid` returns
`(levels, step, nz)`:

- `levels` is a list, **highest resolution first**, each entry a **lazy** (dask) array shaped
  `(z, y, x)` — or `(y, x)` when `nz == 1` (so no singleton z-slider appears).
- Levels **downsample Y and X only. Z is NEVER coarsened** — napari puts its dimension slider on z,
  and `layer.scale = (dz_um, py, px)` so anisotropic data renders anisotropically.
- Each level is fused DIRECTLY from the FOV tiles at its own decimation (`frame[::step, ::step]`),
  not by coarsening level 0 — a coarse level never materialises a full-res intermediate.
- It is lazy: **building the pyramid reads nothing.** napari materialises only the clipped visible
  tile of the level matching the current zoom. Slicing a level (e.g. an ROI crop) is free; the read
  happens only when that slice is materialised.
- Placement: `mosaic_bbox_um(meta, region)` returns `(x0, y0, x1, y1)` in stage micrometres.
  `add_mosaic(bbox_um=...)` sets `layer.scale`/`translate` so pyramid pixels map to µm world.
  Level 0 is native resolution.

## Two abstractions

**2D — the whole pyramid.** Render with napari `multiscale=True`, passing the full `levels` list.
Max resolution is level 0, fetched only where the user is zoomed in. This is always correct because
napari picks the level per zoom. Never flatten the pyramid to one array "to keep it simple" — that
throws away the whole reason 2D is fast on a 5731x4793 mosaic.

**3D — ONE GPU texture.** napari renders a 3D volume from a single GL 3D texture, capped at
`GL_MAX_3D_TEXTURE_SIZE` (~2048 on Apple GPUs; read the live value from the canvas, don't assume).
**If any Y/X axis of the volume exceeds the cap, napari SILENTLY downsamples to a blocky coarse
level.** That silent downsample is the bug. So in 3D you must NEVER hand napari the whole fused
mosaic. You hand it a **texture-bounded native volume**: `(z, y, x)` with `max(y, x) <= texture`,
full z. Two ways to get one:

1. `open_native_3d(...)` — one native FOV (~2084 px, fits the texture). The "3D of this region"
   quick look. gallery-view's original recipe.
2. `open_native_3d_volume(volumes_by_channel, ...)` — a READY native volume, e.g. an **ROI's
   level-0 crop** fused across the FOVs the ROI spans. This is the **organoid path**: box the
   organoid, render exactly that at native resolution. It enforces the texture cap by **raising**
   (NO silent downsample, NO fallback — Julio's rule) when the ROI is too big; the user draws a
   smaller ROI.

Recipe for both (from hongquanli/gallery-view, adapted to napari 0.6.6): `add_image(vol,
scale=(dz, py, px), blending="additive", rendering="mip", contrast_limits=<carried LUT>)`, a 100µm
bounding box, a µm scale bar, and a close-handler that releases the GPU buffers.

## What a window open COSTS, and the two rules that keep it to one fetch

A fused level is ONE dask block per z (`_mosaic_source.fuse_region_pyramid`), so materialising any
part of a level materialises that whole (level, z): a decode of every FOV in the region. On the
real 10x set (manual0, 27 FOVs) that is ~107 ms cold, per channel, per z. **The number of DISTINCT
z a window open touches is therefore the window's load time**, and it is decided entirely by the
order of the calls in `MosaicLayers.add_mosaic`. Two rules keep it at one:

1. **The plane the window opens on is the plane the contrast comes from.** `_contrast.opening_z`
   names that index once, `sample_plane` seeds from it, and it is `(n - 1) // 2` because that is
   **napari's own centring**, not a second opinion. It was `n // 2`, so on an even stack the seed
   described plane 5 while the canvas showed plane 4 — invisible on screen, one extra whole-region
   decode per channel.
2. **A layer is placed at CONSTRUCTION**, via `placement_for` handed to `add_image` as
   `scale`/`translate`, never by assigning `layer.scale` afterwards. Assigning it later moves the
   world extent, which moves napari's dims range, which moves the slider off the plane already
   fetched — another whole-region decode per channel.

Measured on that region, four channels: 432 whole-frame decodes before, 216 after. One decode pass
per channel remains and is NOT removable from outside napari — `Image.__init__` slices itself at
point 0 before the viewer's dims can be consulted.

## Contrast

Carry the on-screen LUT (per channel `contrast_limits` + colormap) into 3D so it matches 2D. If a
channel has no carried LUT, derive one with the maragall fluorescence rule (`_contrast.auto_contrast`
— background mode + 2σ to black, 99.9th pct on top), never napari's raw full-range autoscale (which
renders fluorescence washed out). See `_napari3d._auto_clim`.

## Operators over the abstractions

An operator targets a **View** (`_region_viewer.View`: a named region-set — a window, an ROI, the
selection, or the whole plate). It runs on the View's regions via the CLI engine (`_command` /
`_engine`), writes an OME-Zarr layer, and that layer is itself a pyramid (`open_pyramid`) — so a
processed result renders under the exact same 2D/3D contract as raw. Operators do not live in the
windows; they are picked centrally and aimed at a View (Spencer, 2026-07-23).

## The gallery-view bridge (organoids at max res)

To render an organoid at max resolution in 3D with the gallery-view recipe:
1. Box the organoid as an ROI in its region window -> ROI child window.
2. The child crops the lazy pyramid to the box (`_crop_levels_to_bbox`) — level 0 is the native
   `(z, y_roi, x_roi)` volume, fused across only the FOVs the ROI spans, read on demand.
3. "3D" on the child materialises that level-0 crop and calls `open_native_3d_volume`, which renders
   it natively IF `max(y_roi, x_roi) <= GL_MAX_3D_TEXTURE_SIZE`, else refuses and says so.

That is the contract: 2D is the pyramid, 3D is a texture-bounded native crop of it, operators run on
Views of either, and nothing silently downsamples.
