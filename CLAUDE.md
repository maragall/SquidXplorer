# SquidXplorer

Post-acquisition HCS plate viewer. Reads finished Squid well-plate data (T, C, Z, FOV already on
disk); no live capture, no stage motion.

## The operator contract

`templates/operator/README.md` is the contract, and it is the public one: a complete, installable
example package a contributor copies. Read it before adding an operator anywhere.

Four declarations on the registry record, and nothing generic branches on an operator's NAME
(`tests/test_operator_declaration.py` fails the build if anything does):

| declaration | decides |
|---|---|
| `consumes` | the engine's loop and the output shape — `{"z"}` collapses z, `frozenset()` keeps it, `{"fov"}` is a whole-well region operator |
| `produces` | what the pixels MEAN — the napari layer type, **and how the OME-Zarr writer coarsens a pyramid level** |
| `params` | what one entry can be RUN with (`params=` makes the registered object a factory) |
| `requires` | the modules it needs — **listed either way, run refused BY NAME when missing** |

**ONE table** (`_engine._OPERATORS`, 2026-08-05). `add_projector` and `add_region_operator` are two
registrars over one record, sharing one validator (`_engine._declare`); `add_region_operator`
stamps `consumes=REGION_OP` (`{"fov"}`) and that declaration is what `stitch_plate` selects on,
what `project_plate` refuses on, and what `_compose` refuses inside a chain. `runnable_operators()`
is `sorted(_OPERATORS)` and is defined once; `available_projectors()` / `available_region_operators()`
are its two complementary filters; `is_region_operator(name)` is the ONE spelling of "which loop
runs this", replacing `name in available_region_operators()` at ten call sites. Deleted with the
second table: `_stitch._REGION_OPERATORS`, `_stitch._REGION_REQUIRES` (a sidecar of a sidecar),
`region_operator_available`, `region_operator_requires`, and two duplicate `runnable_operators`
bodies. The queries are named `operator_consumes` / `operator_produces` / `operator_params` /
`operator_requires` / `bind_operator` — they answer for every entry, so they are no longer spelled
`projector_*`.

`requires=` (2026-08-05) is the same word on all three registrars — `add_projector`,
`add_region_operator`, `add_segmenter`. It closed a measured silent success: `decon`, `decon3d` and
`flatfield` import packages absent from `[project.dependencies]`, raised ImportError one call deep,
and `project_plate(on_error=...)` filed that as a per-well skip — a green run that wrote nothing.
Per-well fault isolation now refuses to absorb `ImportError` / `MissingDependency`
(`_engine._NOT_A_WELL_FAULT`): a missing package is not a corrupt well.

**`produces` reaches the DISK, not just the layer type** (2026-08-06). `_output._REDUCERS` maps a
result kind to how a pyramid level is derived from the one below it, and `_write_field` /
`_multiscales` take `produces` from `operator_produces(inner)` in `write_plate`. `intensity` is a
2x2 block **mean**; `labels` is 2x2 **nearest**, because the mean of two object ids is a third
object's id. Measured before the fix, on a real `spot` run over the 10x set (`manual0`, fov 0,
z 5): **1004 of 1 085 764 level-1 pixels carried an id present in none of their four level-0
source pixels** — blocks holding only background and object 4 were written as object 2, a real
object elsewhere in the field, and the store still declared `"type": "mean"`. Level 0 was the only
trustworthy rung and nothing said so. `_reducer_for` **refuses an unknown kind by name** rather
than defaulting to the mean; that default is exactly how this happened. It is the same argument
`_compose` already made ("a `produces="labels"` step must be last") and `_stitch` already made
(labels are never feathered) — the writer was doing the arithmetic both of them refuse.

**A segmenter declares which knobs it READS: `honours`** (2026-08-06). `add_segmenter(honours=…)`
names the `SpotParams` fields the algorithm actually uses; `_spots.segmenter_honours` is the ONE
reader, and both the operator's `params=` and the operator callable's own `__name__` derive from
it. `cellpose` was registered with the whole of `SPOT_PARAMS`, so `_param_panel` drew four spin
boxes and the console line and recipe named four numbers — while `cellpose_nuclei` reads
`min_distance_px` and nothing else. Measured on `synthetic_1536_wellplate` A1 / 405 nm, 1024 px:
`min_area_px` 30 and 4000 returned the **same 42 masks, byte for byte**, where that parameter's
own meaning leaves 2. Declaring the honest subset makes `--param min_area_px=80` a **named refusal**
from `Operator.bind` instead of a number the run drops.
`tests/test_operator_declaration.py::test_every_parameter_a_segmentation_operator_DECLARES_changes_its_pixels`
is the guard: a declared parameter that cannot change the label image fails the build.

**Discovery**: `squidmip/_plugins.py` scans the `squidmip.operators` entry-point group on
`import squidmip`, AFTER the built-ins. An operator in someone else's package needs no edit here.
A broken plugin aborts the import, NAMED; `SQUIDMIP_NO_PLUGINS=1` is the escape hatch. The
hardcoded built-in imports in `squidmip/__init__.py` stay — discovery is additive.

**Composition** (2026-08-05): a chain is written wherever a name is —
`projector="flatfield + decon + mip"`, accepted by `project_plate`, `write_plate`, `stitch_region`,
the CLI's `--projector` and the `run_operator` command with no new argument on any of them. The
expression IS `RecipeChain.label()`, and `RecipeChain.parse()` is its inverse, so the words a
console prints are the words that run. `squidmip/_compose.py` derives the composed operator's four
declarations from its parts (`consumes` union, `produces` last, `params` namespaced
`<step>.<param>`, `requires` union) and carries `corrects_illumination` / `for_channel` through.

Refused by declaration, never by name, never reordered: a **z-reducer that is not last** (no stack
left), a **`produces="labels"` step that is not last** (arithmetic on object ids), a **z-SELECTING
step inside a chain** (`reference` — its z is solved on raw planes outside the operator), and a
**repeated step** (namespaced params would be ambiguous). A bare name still resolves to the exact
registry object, so nothing existing routes through composition.

**GUI panels ARE generated from `params`** (2026-08-05). `squidmip/_param_panel.py` builds one
widget per declared `Param`, choosing it from the TYPE OF THE DEFAULT — `bool` a check box, `int`
a spin, `float` a decimal spin, `str` a text field, the `blurb` its tooltip. Any other type is
**refused by name**; a guessed widget is how a value the user typed becomes a value the run did
not receive. It is the FALLBACK for an operator with no hand-written panel, reached from
**Process well-plates -> From their declaration** (built off `runnable_operators()`, so a plugin
appears with no edit here). A chain's params arrive namespaced (`spot.min_area_px`) and are drawn
as one group per step. `_viewer._activate_operator` opens that panel or states a refusal; it used
to be a silent no-op for any key the card table did not know.

The bespoke panels stay: `StitcherPanel` and `DeconQCPanel` do things a parameter form cannot.
`STITCH_DEFAULTS` is still read off `stitch_region`'s own signature (`_op_panels._stitch_default`)
rather than off a declaration. `add_region_operator` now ACCEPTS `params=` — it is the same record
as every other operator — and `stitch` declares none, because its ~10 knobs reach `stitch_region`
as `**kwargs` and moving them to `Param` records is a separate change with its own evidence.

Measured while building it: `_workers._OperatorWorker`'s PREVIEW branch called `project_plate`
without `operator_kwargs` while the save branch passed them, so a panel value reached the console
line and not the pixels (57 labels vs 57 at `min_area_px` 30/400; 57 vs 44 once fixed). **The same
omission was still live in `_command.EngineExecutor`** and was fixed on 2026-08-06: its preview
branch called `project_plate` without `operator_kwargs` while its save branch passed them, so the
headless and GUI command surface previewed at the defaults. On `sim_5d_2x2_t3` / A1 / `spot`,
preview at `min_area_px=4000` found 55 objects — the same 55 the defaults find — where the save of
that identical command found 30. Two branches of one command must pass the same arguments; the
test that pins it asserts the preview's answer CHANGES with the parameter.

## Flat-field is PER CHANNEL, through `for_channel`

An illumination profile is a per-channel measurement and the stored `.npy` is `(C, Y, X)`. Until
2026-08-06 `_flatfield` held ONE global `FlatfieldProfile` and applied it to every channel of a
run, because the module docstring recorded per-channel dispatch as impossible: *"a plane-op's
callable shape is `Iterable[plane] -> plane`: it never sees which CHANNEL the plane came from."*
That stopped being true when `for_channel` was added for decon (`projection.bind_channel`,
`_decon.optics_for_channel`) to fix the identical defect — all four channels deconvolved with one
PSF. Flat-field was simply left behind.

Measured on the 10x set, whose own stored profile carries four genuinely different gain fields
(405: 0.974–1.020, 488: 0.645–1.102, 638: 0.840–1.096), correcting through the registered operator
after the GUI's own "Load illumination profile":

| channel | one profile (was) | its own (right) | pixels differing | mean abs | max |
|---|---|---|---|---|---|
| 405 | 799.76 | 799.76 | 0.000% | 0.00 | 0 |
| 488 | 3128.63 | 3120.88 | **99.792%** | 155.68 | 1799 |
| 561 | 792.53 | 792.54 | 88.684% | 2.89 | 20 |
| 638 | 2118.68 | 2129.37 | **99.578%** | 67.32 | 1307 |

Silent because 405 is channel 0 — the channel anyone checks first was bit-identical.

The rules now: `_flatfield._active` is a `{channel: profile}` map; **`set_profile(profile, *,
channel=…)` requires the channel** (a gain field without the channel that measured it is not a
thing this package will hold); `_profile_for` **refuses by name**, distinguishing "nothing
installed" from "installed for these other channels"; and
`FlatfieldProfile.per_channel_from_npy(path, names)` is the ONE place a channel NAME becomes a
plane INDEX of the stored file — it was open-coded in `_stitch` and the GUI took plane 0 for
everything. `_stitch._selected_profiles` uses a GUI selection only when it covers every channel of
the run, warns by name when it does not, and never broadcasts.

## What a write REPORTS is read off the result, not the intent

`write_from_stream`'s manifest carries `complete` and `stopped` as **two facts** (2026-08-06).
`complete` used to be `not stopped`, i.e. purely "nobody pressed cancel", so a well lost to
`on_error` — an unreadable TIFF, a corrupt field, exactly what per-well fault isolation exists to
survive — still deleted the `.squidmip-incomplete` marker. Measured on a copy of `sim_5d_2x2_t3`
with one well's TIFFs corrupted: `is_incomplete(store)` was **False with 4 of 16 fields missing**,
while the well group went on advertising images that are not on disk. The CLI warned; the STORE,
which is what `_check_output`, Odon's samplesheet walk and every external reader consult, did not.
`complete` now means *every field this run owed is on disk*, the marker records the shortfall, and
`_command` reads `stopped` for its STOPPED verdict so a skipped-well run is still reported as
PARTIAL rather than relabelled a cancellation.

Counts are named with their own unit for the same reason: `n_fields_written` over `n_wells`
printed **"16/4 wells written"** on a healthy plate and **"12/4"** on one that lost a quarter of
itself, and `_command`'s cancel line printed "stopped after 12 of 4 target(s)".

## 3D is capped at DRAWING time, and renders in-window

`docs/rendering-contract.md` is the contract; this is the rule that binds every render path.

**An ROI rectangle is clamped to the live `GL_MAX_3D_TEXTURE_SIZE` as it is drawn**
(`_bricks.clamp_bbox_um` <- `RegionViewer._clamp_last_roi`). So *anything drawable is renderable,
at full native resolution, from one texture* — the limit is felt while drawing instead of being a
refusal afterwards. **Query the ceiling, never hardcode it**: 2048 px (1540 um) on Apple, commonly
16384 px (12321 um) on desktop NVIDIA, which is 512x the volume. `_bricks.ceiling_line` states it
in the window so better hardware visibly lifts it.

**3D paints into the window's own napari canvas** (`_brick_view.BrickedVolume`), never a fresh
`napari.Viewer`. Adding our own layers to the pane is fine; what was never allowed is rendering the
pane's fused PYRAMID in 3D, whose level 0 is capped to `_MAX_FUSED_PX`.

**Which layer 3D shows comes off the declaration** — `MosaicLayers.visible_op()` picks it and
`_reduces_z` (i.e. the registry's `consumes`) refuses a Z_REDUCER's single plane with a reason.
Never compare an operator name; `tests/test_operator_declaration.py` fails the build on it.

Three numbers decide whether a volume "looks downsampled", and all three are enforced in
`_bricks`, not asserted: voxels per screen pixel must stay **>= 1** (`uniform_step` floors the
ratio), zooming in must **monotonically refine to stride 1**, and **z is never strided** — bricks
tile Y and X only, so a volume keeps every acquired plane and cannot read as flattened.

**TWO pitches exist and they are not interchangeable** (2026-08-06, `tests/test_roi_pitch.py`):

* `meta["pixel_size_um"]` is the **acquisition's**, and it is the unit of LEVEL-0 MOSAIC PIXELS —
  what `_placement.fov_offsets_px` lays FOVs out in, what `_napari3d.roi_window_px` converts an
  ROI box into, what `_bricks.plan` tiles, and what `read_brick` hands back reading off the reader;
* the **displayed** pitch is what the mosaic on screen has. `fuse_region_pyramid` decimates
  (`step = ceil(mosaic_px / _MAX_FUSED_PX)` — 2 on the 10x set), and the layer records the truth in
  its own `scale`, because `add_mosaic` places it from `bbox_um / shape`: 1.504 against 0.752.

**A path that renders the LAYER's pixels takes the layer's pitch; a path that reads the READER's
takes the acquisition's.** `RegionViewer._displayed_pitch_um` reads the former off `layer.scale`
and refuses by name rather than falling back. `_render_roi_volume` was pushing `scale=(dz, px, px)`
for decimated pixels — a **1.99 z:xy aspect where it should be 1.00** — and is fixed.

The clamp is the other way round and was **already right**: `_clamp_last_roi`, `_roi_cost_line` and
`_bricks.ceiling_line` count in acquisition pixels because a drawn ROI's 3D goes
`_open_roi_3d` -> `BrickedVolume` -> `read_brick`, which reads whole FOV planes off the reader.
Measured: the shipped 1540 um clamp is a 2048 px level-0 window, `fits_single_texture` True, and
`read_brick` returns 0.7520 um/voxel. Clamping at the displayed pitch instead gives 4096 px, 16
bricks, 4x the read — under a sentence promising one texture. `_bricks.py` is untouched; do not
"fix" it.

Bricking (many textures + GL `max` compositing + a 1-voxel halo) is the mechanism underneath and is
pixel-exact, but it is NOT what a drawn ROI takes any more. Do not route a whole region through it
expecting interaction — see the measured cost in the contract.

## The plate window owns no viewer, and contrast is not implemented twice

**The plate has no napari surface of its own.** `PlateWindow._mosaic_pane` was pinned to `None` on
2026-07-23 when viewing decentralized into independent `RegionViewer` windows, and it was deleted
on 2026-08-06 along with everything that guarded on it: `_load_mosaic`, `_on_mosaic_plane`,
`_on_mosaic_done`, `_region_frame_done`, the four napari-dims helpers, `_adopt_centre_view`,
`_add_result_layers`, `_bind_napari_contrast`, `_stop_mosaic_worker`, `_populate_detect_channels`,
`_stop_spots`, the plate's whole spot-detection chain, `_make_mosaic_pane` and `_ChannelBar`
(unbuilt since 2026-07-22). Measured before the cut, on the real fixture: 2 of 2 `_load_mosaic`
calls returned at the guard, 0 `_MosaicWorker` objects were built, and a 140 ms debounce `QTimer`
armed 3 times into a slot that could only return.
`tests/test_plate_follows_windows.py` now pins `not hasattr(win, "_mosaic_pane")` — the ABSENCE of
the attribute, not `is None`, because `None` is what let twenty dead methods read as a feature.

**Contrast is ONE job on each side of one seam, not two implementations of one job.** This was
raised as duplication twice and audited on 2026-08-06; the answer is no, and it is recorded here so
it is not raised a third time.

| side | methods | what it does |
|---|---|---|
| `_viewer.py` (plate) | `_bind_window_contrast`, `_adopt_window_view`, `_on_detail_contrast`, `_plate_copy_luts`, `_plate_paste_luts`, `on_screen_luts` | makes the plate a **sink** of a window's napari. Never reads or writes a napari layer; every write goes through `PlateOverview.follow_channel_window` / `set_channel_window`. |
| `_region_viewer.py` (window) | `_per_channel_luts`, `_apply_luts`, `_copy_luts`, `_paste_luts`, `_match_raw_contrast` | reads and writes **this window's own napari layers**. `_match_raw_contrast` is raw -> operator layers *within* one window and delegates to `MosaicLayers.match_contrast_to`. |

They share a word and no code path. `on_screen_luts` **delegates** to the focused window's
`_per_channel_luts` (one hop, not a second reader), and `_LUT_CLIPBOARD` is one dict both sides
name. The one genuine duplicate was `_adopt_centre_view` (plate pane) against `_adopt_window_view`
(per window) — same job, and the pane-flavoured one was the dead one. Task 1 removed it; there is
nothing left to collapse.

## Two producers of a region's pixels, and they are not interchangeable

`stitch_plate()` is the mosaic **of record**: registration, fusion, native resolution, ~0.9 GB per
27-FOV well. `_mosaic_source` / `_gallery` produce **preview placement**: FOVs pasted at their stage
coordinates, later-overwrites-earlier, decimated on read. Both go through the SAME `_placement`
helpers, so they are one geometry at two resolutions — never two implementations. Anything that is a
LOOK (a window's mosaic, the plate cells, Gallery View) takes the preview path; anything that is a
RESULT takes `stitch_plate`. Adding a third placement rule is the defect shape this repo has the
most of, because the error renders as a plausible image.

A **FOV subset of a region** is expressed one way everywhere: the mapping `{region: [fov, ...]}`,
which is `stitch_plate(regions=…)`'s own parameter and `GalleryScope.fovs`. It is produced by
`PlateWindow.selected_region_fovs()` from the plate selection (marquee, shift-click, Cmd/Ctrl-A).
Do not build a second selection mechanism; `_placement.fov_offsets_px` normalises whatever FOV set
it is handed, so the crop needs no cropping code.

The **.mp4 export** (`_video.region_movie_frames`) is a LOOK and takes the preview path, one
`fuse_region_mosaic` per channel per axis index, capped at `MOVIE_MAX_PX` (1080 on the long side —
a movie is a display artifact, and the fuser's own 8192 cap is a 200 MB RGB frame no player opens).

## Recording is post-acquisition, and that is why it exists here

`squidmip/_video.py` assembles an **already-acquired axis** into an .mp4 — T when `n_t > 1`, else
Z. It is not camera capture, which belongs to Squid. That distinction is load-bearing: the module
was deleted on 2026-07-31 (c25d84d, "users record at acquisition time, not here") and
`scripts/smoke_import.py` recorded the removal as final; both statements are true of camera capture
and were applied to a module whose first paragraph disclaims it.

The button is offered when **`n_t > 1 or n_z > 1`** (`_video.can_record`), never on `n_t` alone:
every acquisition on this machine is `n_t = 1`, so a T-only gate makes the feature invisible on all
real data and drops the Z focus sweep.

**Contrast is latched once for the whole movie**, taken from the layers on screen. Per-frame
percentiles — what the deleted module did — normalise away the very change being recorded: a blob
moving through a field shifts the percentiles, so the movie comes out looking static. This is the
same "one contrast rule per quantity" discipline as `_montage.composite`, which the frame
compositor routes through rather than carrying a second windowing loop.

**Encoder**: `imageio` + `imageio-ffmpeg` (its wheel carries an ffmpeg binary for all three
platforms, so no system ffmpeg is needed), declared as the `[video]` extra and pulled in by `[gui]`
and `[test]`. It is probed by `_video.encoder_problem()` before anything is written, and the chip
greys out naming what is missing. Both packages are in `scripts/hcs-viewer.spec`'s `collect_all`
list, not its excludes — the same package-data blind spot napari has there.

## A layer's `data` is not the list that went in

`layer.data` for a `multiscale=True` layer is `napari.layers._multiscale_data.MultiScaleData`, a
`Sequence` of levels that is **neither a list nor a tuple** and that reports level 0's `ndim`,
`shape`, `dtype` and `size` as its own — while `np.asarray()` of it returns the **coarsest** level
(`__array__` is `_data[-1]`). So `isinstance(data, (list, tuple))` as a pyramid check is False for
every real pyramid in this app, and each site that used it failed differently:

| site | how it failed |
|---|---|
| `_workers._full_res_mip` | `AttributeError: 'MultiScaleData' object has no attribute 'max'` — Julio, running cellpose on the 10x set. Loud. |
| `_workers._full_res_plane` | `data[z]` walked the LEVELS, not the z planes |
| `MosaicLayers._swap_layer_scale` | returned at its first line: the 3D full-res swap was a **silent no-op**, so napari kept dropping the layer to its coarsest level in 3D |
| `RegionViewer._render_roi_volume` | passed the pyramid on, and `np.asarray` picked the coarsest rung: a blocky volume with no message |

**`_napari_view.pyramid_levels()` / `full_res_level()` are the one rule**, and every reader of a
layer's data goes through them. The discriminator is `Sequence` (numpy/dask/zarr arrays are not)
whose FIRST ELEMENT is an array, so a plain nested list still reads as one array. Do not add a
fifth `isinstance` check; `MosaicLayers._collapse_layer_z` already knew this and wrote it in a
comment, and a comment is not a mechanism.

`tests/conftest.py`'s `StubLayer` now wraps a multiscale add in a real `MultiScaleData`. It used to
hand the plain list back, and that lie is why four sites drifted with the suite green.

## Nothing decodes on the Qt thread — including the contrast seed

The contrast seed is the one that keeps getting missed, because it does not look like a read.
`_contrast.sample_plane` already picks the COARSEST pyramid level, so it looks free — but every
level of a raw-preview pyramid is fused from the FOV TIFFs at its own decimation, so materialising
even the smallest rung decodes every FOV of the region. Measured twice, on two different paths:
128 ms of frozen UI per region on the mosaic path (493–604 ms on the reporting machine), and the
same shape of cost per cell for a gallery, which is N regions and would have been N freezes.

Both are fixed the same way and it is now the rule: whatever computes the pixels computes the
window, on the worker thread, and the UI receives `(lo, hi)` as data. See `_MosaicWorker`
(`_workers.py`, `_auto_window_for` on the worker) and `_gallery_window.GalleryWorker`. Results
arrive per unit so the first one paints while the rest are still being read.
`tests/test_gallery.py::test_the_gallery_never_reads_a_plane_on_the_qt_thread` pins it by recording
the thread ident of every `reader.read`, so a later refactor cannot quietly reintroduce it.
`tests/test_video_window.py::test_the_export_never_reads_a_plane_on_the_qt_thread` is the same
instrument on the .mp4 export (`_workers._VideoWorker`), where the click handler measures 0.91 ms
against a 4.20 s export on the real 10x acquisition.

Caches are shared, not per-feature: `_mosaic_source.plane_cache()`, `_platecache.PlateCellCache`,
both bounded by `_budget.cache_budget()`. A feature-private cache spends the same budget twice and
the two evict against each other.

## Agent skills

### Issue tracker

Markdown docs in the separate `Cephla-Lab/AI-docs` repo under `SquidXplorer/`, with `to-do` /
`in-progress` / `done` as the status semaphore. Not GitHub Issues, not Linear; `IMA-###` ids are
legacy. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles, unchanged, recorded as a `Status:` line in each ticket file. See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` at the root, ADRs in `docs/adr/`. Neither exists yet;
`/domain-modeling` creates them lazily. `docs/rendering-contract.md` and `docs/plate-contract.md`
already act as undeclared ADRs — read them before touching the render or read paths. See
`docs/agents/domain.md`.
