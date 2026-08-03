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

### Not every result is an image: the operator declares its RESULT KIND

The paragraph above is true of an operator whose pixels measure LIGHT, which was every operator
until segmentation arrived. A segmenter returns a **label image**: integer object ids, where the
value is a name and not a quantity. Handing one to `add_image` is not a cosmetic mistake. The
fluorescence auto-window (`_napari_view._auto_window_for`, background peak to black) stretches
"label 1 … label 400" as if it were photons, so the mask renders as a near-black gradient, with an
opaque background covering the mosaic it is supposed to annotate, no label colours and no
click-to-pick. That is what `spot` did from the day it shipped until 2026-08-03.

The cure is a declaration, not a branch. An operator's registry entry carries `produces`
(`_engine.Operator`, `projection.INTENSITY` / `projection.LABELS`) exactly as it carries `consumes`:

| `produces` | the pixels are | the layer |
| --- | --- | --- |
| `"intensity"` (default) | a measurement of light | napari `Image`, windowed, colormapped, additive |
| `"labels"` | integer object ids, 0 = background | napari `Labels`, transparent background, no window |

The kind is read ONCE on the display side (`_viewer.PlateWindow._as_result`), rides on the
`Result`'s `Substance`, and both sinks — the plate's own pane and every open region window — call
`MosaicLayers.add_result(result.kind, …)`, which dispatches off a TABLE. Nothing in either sink
knows what a segmentation is.

**The rules this adds to the contract:**

- A result kind is declared by the operator and read by the delivery path. **Never test an
  operator's name.** `tests/test_operator_declaration.py::test_no_module_branches_on_an_operator_name`
  checks that over the AST, with a written-down allowlist of the two comparisons that predate it.
- A kind the viewer cannot draw **raises**, naming the operator. No fallback to `add_image`: a
  segmentation quietly rendered as fluorescence is the defect, and a fallback is how it comes back.
- Labels are **never** windowed, never interpolated and never multiscale on this path. A coarser
  level of a label image is only meaningful under nearest-neighbour decimation, which nothing here
  guarantees.
- A label mosaic's ids are **per FOV**, so two nuclei in different FOVs of one region can share an
  id and therefore a colour. `_op_result` fuses by paste-and-stride (`_mosaic_source
  .fuse_region_mosaic`), which is correct — it never blends or averages, so no invented label
  values appear at a seam — but region-wide unique ids need inter-FOV work, which is the
  `consumes={"fov"}` seam a plane-op structurally cannot reach.
- A labels result **cannot be written to a plate.** It is a plane-op, so z survives at full depth,
  and `_output._validate_image` accepts `Z == 1` only. The refusal is loud and is the same one
  every plane-op has had since IMA-223; nothing about the result kind changes it.

## The gallery-view bridge (organoids at max res)

To render an organoid at max resolution in 3D with the gallery-view recipe:
1. Box the organoid as an ROI in its region window -> ROI child window.
2. The child crops the lazy pyramid to the box (`_crop_levels_to_bbox`) — level 0 is the native
   `(z, y_roi, x_roi)` volume, fused across only the FOVs the ROI spans, read on demand.
3. "3D" on the child materialises that level-0 crop and calls `open_native_3d_volume`, which renders
   it natively IF `max(y_roi, x_roi) <= GL_MAX_3D_TEXTURE_SIZE`, else refuses and says so.

That is the contract: 2D is the pyramid, 3D is a texture-bounded native crop of it, operators run on
Views of either, and nothing silently downsamples.
