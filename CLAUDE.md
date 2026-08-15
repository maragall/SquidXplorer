# SquidXplorer

Post-acquisition HCS plate viewer. Reads finished Squid well-plate data (T, C, Z, FOV already on
disk); no live capture, no stage motion.

## Domain-model v2 renames (2026-08-12)

"Projector" was fossil naming for the engine loop, and the axis words now use Squid's spelling.
Dated docs under docs/ may still say "projector"; this file and the code do not.

| old | new |
|---|---|
| `project_plate` / `stitch_plate` | `run_plate` (one entry, dispatching off `consumes`; the loops are private) |
| `add_projector` | `add_operator` (`add_region_operator` stays) |
| `available_projectors()` | `available_plane_operators()` |
| `projector=` / `--projector` | `operator=` / `--operator` |
| stitch's inner z-handling `projector=` | `z_operator=` (it rides in `operator_kwargs`, so `operator=` would collide) |
| reader `read(..., z, t=0)` | `read(..., z_level, time_point=0)` (`plane_ref` likewise; `project_well`, `TileDescriptor`, tile sources, caches, workers follow) |
| `_acquisition.Channel` | `DisplayChannel` |

On-disk / Acquisition contract keys are untouched: `n_t`, `n_z`, `z_levels`, `dz_um`.

## Architecture v2: kill list, OperatorRun, ingest extraction (2026-08-13)

- **Kill list**: `_benchmark`, `_bench_stitchers`, `_oracle`, `_odon_bench`, `_prefs` and
  `_terminal` are deleted with their tests and the tools harnesses that imported them
  (`tools/benchmark.py`, `tools/odon_benchmark.py`, `tools/stitch_demo.py`), plus
  `_viewer._build_cli_tab`, which had no caller left. The close-all "don't show me this again"
  checkbox stays but is a session flag (`PlateWindow._warn_close_all`) — no prefs file.
- **`_run.OperatorRun`** (Qt-free) owns a run's identity and books: key/layer key, requester,
  first-paint clock, address, per-region accumulators, error, `settle_stranded` and the closing
  verdict in `_measure`'s outcome words. Created by `run_operator`, closed by the drain slot;
  `PlateWindow` keeps the signal wiring and the `_runs_settled` counter. `_run_label`,
  `_run_units` and `_resolved_target` were write-only fields and are gone.
- **`_ingest.py`** owns the acquisition-open pipeline — `ingest`, the raw-preview lifecycle and
  the loupe-source bookkeeping — as functions over the window's own state (one bookkeeping).
  `PlateWindow` forwards; its `_start_preview` forwarder stays the ONE place the timepoint bar
  reaches the pixels. `_viewer.py`: 4676 -> 4344 lines.
- **One identity module, one writer of the plate's shown layer**. `_operations.py` owns the
  operator identity model — the cards, the layer-key vocabulary (`operator_layer_key` /
  `operator_name` / `result_kind`) and `OperationStack` (`_layers.py` is deleted into it, along
  with `remove` / `remove_suffix`, product-dead since the exploration tabs went).
  `PlateWindow._apply_layers` is the ONLY caller of `PlateOverview.set_active_layer`: every path
  that changes what the plate shows writes the stack and calls it. `_return_to_raw` was the live
  drift — it put the overview on raw while the stack still claimed the operator layer, so the
  Layers tab showed a transform ON over a raw plate; it now disables the transforms through the
  stack. Deliberately NOT unified into one plate+window registry: a window's identity store is
  napari's own Layers list read through `MosaicLayers` (derived, never cached — see the layer-model
  section), a window legitimately carries identities the plate never ran (an ROI child re-uses its
  parent's groups), and the plate's refusal of a region-operator layer is already declaration-
  derived (`is_region_operator`) at the one seam (`_follow_window_layer`).
- **The reader contract is importable on its own**: `squidxplorer/contract/reader.py` holds
  `SquidAcquisitionReader` (typing-only module; `reader.py` re-exports it, `squidxplorer.__init__`
  unchanged), so Squid can `from squidxplorer.contract.reader import SquidAcquisitionReader`
  across the repo boundary without touching the readers. The rest of the Squid seam waits on named
  triggers: when core-service (#578) merges, the `job_completed` SSE listener and a live reader
  land behind this same contract; when schema v2 (#593) lands, declarative wells replace the
  `parse_well_id` heuristic.

## The operator contract

`templates/operator/README.md` is the contract, and it is the public one: a complete, installable
example package a contributor copies. Read it before adding an operator anywhere.

Five declarations on the registry record, and nothing generic branches on an operator's NAME
(`tests/test_operator_declaration.py` fails the build if anything does):

| declaration | decides |
|---|---|
| `consumes` | the engine's loop and the output shape — `{"z"}` collapses z, `frozenset()` keeps it, `{"fov"}` is a whole-well region operator |
| `produces` | what the pixels MEAN — the napari layer type, **and how the OME-Zarr writer coarsens a pyramid level** |
| `params` | what one entry can be RUN with (`params=` makes the registered object a factory) |
| `requires` | the modules it needs — **listed either way, run refused BY NAME when missing** |
| `extra` | the install payload — the `[project.optional-dependencies]` group that makes it runnable; `None` means core |

**`extra=` and the installer** (2026-08-13). `operator_extra(name)` sits beside the other
queries, `do_list_operators` rows carry it, and the declaration test enforces both directions: an
operator requiring a non-core package must declare an extra, and a declared extra must name a
real group (so `flatfield`, which requires tilefusion, is `extra="stitch"` like
`stitch`/`coordinate`; `decon`/`decon3d` are `"decon"`, `cellpose` is `"segment"`).
`scripts/installer/` (stdlib, Qt-free, tests in `tests/test_installer_menu.py`) is built on it:
`menu.py` generates the install checkbox menu FROM the registry — defaults core/stitch/decon
checked, segment unchecked, and a failed `cuda12_available()` probe shades decon with the probe's
own reason (today: every Mac) — and `bootstrap.py` drives uv into the private env (`--dry-run`
prints the exact commands; a missing uv is a refusal carrying the install hint). tilefusion and
petakit are pinned to commit SHAs in pyproject so two installs resolve the same bits; the Windows
exe is built elsewhere (`scripts/installer/README.md`).

**ONE table** (`_engine._OPERATORS`, 2026-08-05). `add_operator` and `add_region_operator` are two
registrars over one record, sharing one validator (`_engine._declare`); `add_region_operator`
stamps `consumes=REGION_OP` (`{"fov"}`) and that declaration is what the region loop selects on
and the per-FOV loop refuses on. `runnable_operators()`
is `sorted(_OPERATORS)` and is defined once; `available_plane_operators()` / `available_region_operators()`
are its two complementary filters; `is_region_operator(name)` is the ONE spelling of "which loop
runs this", replacing `name in available_region_operators()` at ten call sites. Deleted with the
second table: `_stitch._REGION_OPERATORS`, `_stitch._REGION_REQUIRES` (a sidecar of a sidecar),
`region_operator_available`, `region_operator_requires`, and two duplicate `runnable_operators`
bodies. The queries are named `operator_consumes` / `operator_produces` / `operator_params` /
`operator_requires` / `bind_operator` — they answer for every entry, so they are no longer spelled
`projector_*`.

`requires=` (2026-08-05) is the same word on every registrar — `add_operator`,
`add_region_operator`, `_spots.add_segmentation_operator`. It closed a measured silent success: `decon`, `decon3d` and
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
`_stitch` already made (labels are never feathered) — the writer was doing arithmetic on object
ids, which nothing that knows what labels mean will do.

**The segmenter registry is folded into the operator table** (2026-08-13). One operator per
algorithm — `spot` IS the Otsu-watershed, `cellpose` declares `min_distance_px` and nothing
else — registered through `_spots.add_segmentation_operator`, which filters `SPOT_PARAMS` so
`SpotParams` stays the one place the knobs and defaults are written. Deleted with the second
registry: `add_segmenter`, `resolve_segmenter`, `preferred_segmenter`, `segmenter_honours`,
`available_segmenters`, `segmenter_available`, `Segmenter`, `MissingSegmenterDependency`, and
`honours` as a concept — the entry's own `params=` is the honest subset now, so an undeclared
`--param` is a named refusal from `Operator.bind`.
`tests/test_operator_declaration.py::test_every_parameter_an_operator_DECLARES_changes_its_pixels`
is the guard, parametrized over EVERY registered operator with `params` (skipping by `requires`
when a dependency is absent): a declared parameter that cannot change the output fails the build.

**Discovery**: `squidxplorer/_plugins.py` scans the `squidxplorer.operators` entry-point group on
`import squidxplorer`, AFTER the built-ins. An operator in someone else's package needs no edit here.
A broken plugin aborts the import, NAMED; `SQUIDXPLORER_NO_PLUGINS=1` is the escape hatch. The
hardcoded built-in imports in `squidxplorer/__init__.py` stay — discovery is additive.

**Composition was cut** (2026-08-13, added 2026-08-05). An `operator=` string is ONE registered
name: a chain expression (`"flatfield + decon + mip"`) is a named refusal from
`_engine._resolve_operator` explaining that chaining was removed and that composing happens in
Python (`plane_op` around the steps + `add_operator`, a few lines). `_declare` still refuses
`+()` in a registered name — `RecipeChain.parse` must round-trip a recipe label — and
`_recipe.py` is untouched: the result cache and LUT clipboard ride on `RecipeChain`.

**One result type** (2026-08-13). `OperatorResult` is deleted:
`_op_result.RegionResultAccumulator.result()` builds the self-describing `_result.Result` itself —
`Extent` carries region + bbox_um, `Substance` carries channels/z_depth/dtype/pixel_size_um/kind
(`result_kind` is read there, once), `data` is the per-channel plane list — and `op` travels as a
parameter of the delivery calls, where it already was. `_viewer._as_result` went with it; the
accumulator's refusals (channel mismatch, unknown FOV, incomplete region, and now the missing
pixel size) are unchanged in meaning.

**One verdict** (2026-08-13). `_measure.verdict(landed, owed, skipped, stopped)` is the single
OK/PARTIAL/STOPPED computation. `landed` counts FIELDS and only zero is read;
`owed`/`skipped` count target wells; `stopped` comes from the manifest or the stop
event, never from `complete`; a stopped run's detail stays the caller's own sentence.

**One dispatch** (2026-08-14). `_dispatch.run_operator_once` is the single save-vs-preview
control flow, and the only caller of `verdict` for a run: it owns the `write_plate` /
`run_plate` branch, builds `operator_kwargs` ONCE from one `parameters` argument used by BOTH
branches (the twice-fixed preview-forgot-`operator_kwargs` drift is now unwritable), counts
landed fields and skipped regions, and reads `stopped` off the manifest or a stop poll —
including one final poll after either branch, so a stop requested in the run's tail is a
stopped run on every surface. `_workers._OperatorWorker._run_body` (Qt signals, "stopped by
the window") and `_command.do_run_operator` (console sentences, the result dict) are thin
adapters over it. Not in `_measure` (the cost ledger owns no engine knowledge) and not in
`_run` (GUI books the headless executor must not import).

**GUI panels ARE generated from `params`** (2026-08-05). `squidxplorer/_param_panel.py` builds one
widget per declared `Param`, choosing it from the TYPE OF THE DEFAULT — `bool` a check box, `int`
a spin, `float` a decimal spin, `str` a text field, the `blurb` its tooltip. Any other type is
**refused by name**; a guessed widget is how a value the user typed becomes a value the run did
not receive. It is the FALLBACK for an operator with no hand-written panel, reached from
**Process well-plates -> From their declaration** (built off `runnable_operators()`, so a plugin
appears with no edit here). `_viewer._activate_operator` opens that panel or states a refusal; it
used to be a silent no-op for any key the card table did not know.

The bespoke panels stay: `StitcherPanel` and `DeconQCPanel` do things a parameter form cannot.

**Stitch joined the declaration system** (2026-08-13). The registration is a factory declaring
`z_operator`, `register`, `registration_channel`, `registration_t` and `correct_illumination` as
`Param` records, so `--param`, recipes and the probe test describe stitch like every other
operator; `STITCH_DEFAULTS` and the `StitcherPanel` read the declaration, not
`inspect.signature`, and None-defaulted signature knobs state their fixed meaning concretely
(`registration_channel` None = index 0, `correct_illumination` None = on). Still kwargs, each for
a reason: `blend_px` / `registration_z` / `correct_distortion` (their None is measured from the
data), `block_px` / `max_workers` (cannot change the pixels), and `rel_thresh` / `abs_thresh` —
tilefusion clamps `rel_thresh <= 1.0` to its own factor 3.0 and floors the rejection cutoff at
150 px (`_BLUNDER_FLOOR_PX`), so neither knob can change a solve short of a >150 px blunder, and
the probe test below could never vouch for declaring them. The probe covers stitch through a
synthetic 2x2 region (`tests/test_operator_declaration.py::_StitchProbeReader`) whose content
errors registration measurably solves, skipping by `requires` where tilefusion is absent.

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
survive — still deleted the `.squidxplorer-incomplete` marker. Measured on a copy of `sim_5d_2x2_t3`
with one well's TIFFs corrupted: `is_incomplete(store)` was **False with 4 of 16 fields missing**,
while the well group went on advertising images that are not on disk. The CLI warned; the STORE,
which is what `_check_output`, Odon's samplesheet walk and every external reader consult, did not.
`complete` now means *every field this run owed is on disk*, the marker records the shortfall, and
`_command` reads `stopped` for its STOPPED verdict so a skipped-well run is still reported as
PARTIAL rather than relabelled a cancellation.

Counts are named with their own unit for the same reason: `n_fields_written` over `n_wells`
printed **"16/4 wells written"** on a healthy plate and **"12/4"** on one that lost a quarter of
itself, and `_command`'s cancel line printed "stopped after 12 of 4 target(s)".

**ONE marker, and the openers read it** (2026-08-06). `.squidxplorer-incomplete`, written inside
`plate.ome.zarr` by `write_from_stream` and by nothing else, is THE record that a store is not
whole. `_output.incomplete_reason(dir)` is the one reader that turns it into a sentence (quoting
the marker's own `fields` / `fields_written` / `stopped`, never re-deriving them) and takes either
the store or the `.hcs` folder holding it, so no caller has to resolve that itself.

The writer half was already right; the READER half was not. `_viewer._open_computed` tested for a
file called `INCOMPLETE` — a second name for the same fact, whose only writer
(`_viewer._note_partial_output`) had **zero callers** — so the guard was dead and a save the user
stopped opened as a finished acquisition. Measured on a store written by `write_from_stream` with
`stop()` after 3 of 8 fields: `is_incomplete(store)` True, `(base/"INCOMPLETE").exists()` False,
and the window printed "loading computed plate · 4 wells" over 2 wells on disk. The dead writer,
`PlateWindow._run_out_dir` and the `finished_ok` hook that cleared it are gone — a GUI flag cannot
see a well the engine skipped and cannot be set at all by a process that was killed, which is
exactly why the store settles this itself. `_cli.run` refuses an INCOMPLETE plate as INPUT through
the same function; `contract/validate.py` already warned on it. External readers were never the
silent ones: measured, `ngio.open_ome_zarr_plate` raises `NgioValidationError` on the absent wells
and `validate_plate` reports 5 hard errors.

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

**While a volume is up, `MosaicLayers` is NOT describing the mosaic** (2026-08-06). `open()` moves
the `(op, channel)` identity off the pane's 2-D layers onto the bricks on purpose, so from that
moment `find` / `channels` / `visible_op` answer about BRICKS. Therefore `_open_3d` calls
`_close_native3d()` **before it reads the scene**, not merely before it builds the next view.
Measured, second 3D click over a `bgsub` layer: the source was one 512 px brick, **1 of 9** bricks
of the ROI yielded voxels, and the contrast came back `(0.0, 1.0)` against `(120, 900)` on screen.
Same reason, same commit: a key is **one-to-many**, so anything that DRIVES a pair uses
`MosaicLayers.layers_for` (the tree's checkbox drove 1 brick of N) and `find` is only for anything
needing *a* layer. 3D's LUTs are harvested from the layer being RENDERED (`_on_screen_luts(source)`),
never from raw.

**napari will not re-slice a HIDDEN layer, but it will still update its slice INPUT** — so a 2D/3D
flip leaves every hidden layer claiming a dimensionality its own slice does not have.
`Layer._slice_dims` assigns `_slice_input` unconditionally and then calls `_refresh_sync`, which
returns at `if not (self.visible or force)`. `Image._update_thumbnail` — which EVERY
`contrast_limits` write calls, with no visibility guard — then does `np.max(image, axis=0)` on a
stale 2-D thumbnail and hands `scipy.ndimage.zoom` a 2-element zoom for a rank-1 array. Measured on
the real 10x set against a multiscale raw `(10, 5731, 4794)`: open a 3-D volume, drag the contrast,
`RuntimeError: sequence argument must have length equal to input rank` — through
`MosaicLayers.set_contrast` and through a bare layer write alike, and in BOTH flip directions.
Reachable only since bricks became real model layers (`adopt` registers them, so `_link_set` links
a brick to the hidden flat mosaics of its channel). `MosaicLayers._reslice_hidden_layers` is
subscribed to `dims.events.ndisplay` — one place, so napari's own 2D/3D button is covered too — and
calls `refresh(force=True)`, napari's own public opt-out of that guard. 4.5 ms on the real pyramid.
`ndim > 2` is the exact precondition of napari's branch, not an optimisation.

**A REGION operator's result reaches the layer at full depth** (2026-08-06).
`_workers._OperatorWorker._result_pixels` emits `image[0]` — `(C, Nz, Y, X)` — for an operator
declaring `consumes={"fov"}`, and `RegionResultAccumulator.add` accepts it. The slot used to emit
`image[0, :, 0]` for everything: on the real 10x set `stitch_plate` yielded `(1, 1, 10, 2084, 7711)`
and the display got `(1, 2084, 7711)`, so a stitched mosaic declared `z_depth 1` and 3D correctly
refused a volume that no longer existed. The whole display side was already written for depth
(the accumulated `Result`'s `z_depth`, `deliver_result`'s `z_scale_um`, `_volume_source`'s
`ndim >= 3`); one index contradicted all three. The per-FOV path still delivers ONE plane — keeping its depth means
`Nz` re-fusions over ~9.4 GB of accumulated tiles for one 27-FOV well — and `_z_dropped_note` says
which plane of how many rather than letting a limitation read as a result.

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

## ONE layer model, 2D and 3D: an identity may be rendered by several layers

Julio: *"why is the layering of 2d and 3d different in the same place?"*

Because 2D assumed **an identity IS a layer**. A flat mosaic is one `add_image` per
`(op, channel)`, so `MosaicLayers.find` was the whole model and every rule was written against it.
A VOLUME is not: `_brick_view.BrickedVolume` tiles it into one Image layer per BRICK, all of them
the same operator's same channel. 3D therefore either LOST each rule or grew a partial private
copy of it, and that one divergence produced four user-visible defects in one evening (a volume
with no checkbox, a coarse 2D mosaic drawable over it, a channel checkbox reaching one brick, a
second contrast model).

**The fact is now declared in the shared model, not branched on in a caller**
(`squidxplorer/_napari_view.py`):

| | |
|---|---|
| `find(op, channel)` | the **representative** — the layer a control reads and writes |
| `layers_for(op, channel)` | **every** layer rendering that identity, derived, never cached |
| `IDENTITY_PROPS` | `visible`, `contrast_limits`, `colormap`, `gamma`, `opacity` — the properties that belong to the IDENTITY, i.e. exactly what napari's layer controls expose. `blending` is deliberately absent: it is a rendering choice a volume legitimately makes differently |
| `_mirror_identity` | the ONE mechanism keeping those equal across surfaces. A no-op when an identity has one layer, so every 2D path is untouched by construction |
| `adopt(op, channel, layer)` | THE way a layer built elsewhere becomes an app layer: stamps identity, **takes the identity's current values**, labels micrometres, registers the channel |
| `drop_layer(layer)` | one surface leaves; the identity survives while another holds it |

`BrickedVolume` now takes **`MosaicLayers`, not a bare viewer**, and every brick is added through
`adopt` and removed through `drop_layer`. Deleted with that: `BrickedVolume._link_contrast` and
`_propagating`, the second contrast propagator. `_contrast_by` survives as the SEED for the first
brick of a channel only. `remove_op_channel` removes the identity — every holder — because
removing the representative alone left the rest on screen as foreign layers.

The napari contrast **link set stays one layer per identity** (`_link_set`): `link_layers` connects
every ordered pair, so linking bricks would be quadratic in the brick count and leave a link record
behind every eviction. Surfaces follow the representative through the mirror instead, and the
answer is unchanged: one contrast value per channel in the whole application.

Pinned by tests parametrized over the two scenes (`tests/conftest.py::build_flat_scene` /
`build_volume_scene`, used by `test_viewer_3d.py` and `test_layer_tree.py`), against a real
`ViewerModel` with several bricks — a rule that only holds for a one-brick volume is the bug
wearing a disguise.

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

## A tile's identity includes its TIMEPOINT

`docs/plate-contract.md` is the contract; this is the rule. `_tiling.TileDescriptor` is
`(level, key, channel, bbox_um, t)` and **`t` has no default**, because every source now reads the
frame off the REQUEST — `ReaderTileSource` and `ZarrPyramidSource` take no `t` at construction at
all. The plate cell cache made the same choice for the same reason (`(token, t, region)`, `t` in
the KEY and not the token), and it buys the same thing: `TileCache` holds both frames, so stepping
back to a timepoint already seen is a hit.

It was measured, not reasoned: on `sim_5d_2x2_t3`, whose blob moves with `t`, one FOV tile came
back **byte-identical** after `PlateOverview.set_time_point(2)` — sha `24d0d02d…` where t=2 is
`a265917c…` — while the plate reported it was showing timepoint 2. The descriptor carried no
timepoint, so nothing on that path could ask the question, while both caches under it (`_platecache`
and `ReaderTileSource._planes`) keyed on one. `_workers.py` **raises** on exactly this mismatch one
module over; the tile path avoided the reconciliation by never asking.

A tile read at one frame and drawn under another is worse than one that does not move, so two
places refuse rather than reconcile: `TileCache._nearest_ancestor` matches `t` as strictly as
`channel` (the blur fallback is the one site licensed to draw something other than what was asked
for), and `InMemoryMultiscale` — one frame's seeded cells — raises on a request for another, with
`CompositePlateSource` refusing a mismatched `PlateCellCache` and routing a coarse tile at another
`t` to the reader. `set_time_point` rebuilds nothing; it repaints, and the new descriptors do it.

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

`squidxplorer/_video.py` assembles an **already-acquired axis** into an .mp4 — T when `n_t > 1`, else
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

**A cached `TiffFile` is a FILE OBJECT, so reading one is not re-entrant** (2026-08-06).
`reader._TiffHandles` is the one handle cache and the only accessor is `page(path, index)` /
`read(path)` — a context manager that holds a **per-file lock** for the decode. `SquidOMEReader`
and `SquidMultiPageTiffReader` both went through a `_tif(path)` that handed the shared object out
unguarded, and `pages[p].asarray()` seeks: two threads decoding two pages of one file moved one
seek position under each other. Measured on the 10x acquisition, `manual0` FOV 17: **0 errors in 8
serial reads, 10 of 40 threaded** (`TiffFileError: suspicious number of tags`, `ValueError: failed
to read 8686112 bytes, got 0`); 0 of 40 once locked. One lock per FILE, not per reader — the
parallelism is across FOVs and each FOV is its own file.

That fault was invisible because the layer above absorbed it, which is the more important rule:

**A run may not end holding a result it never delivered.** `_viewer._on_result` refuses to draw a
region until every FOV is in (half a mosaic reads as something the operator did), and it did that
by returning with the accumulator still in the run's books — where nothing ever looked again.
`OperatorRun.settle_stranded`, forwarded by `PlateWindow._settle_stranded_results` from
`_on_run_drained`, now resolves every leftover: it delivers a complete one and NAMES an incomplete
one in the accumulator's own words ("23 of 27 FOV(s) have results"), sets the run's `error` so
`_close_requester_pair` reports `operator_failed`, and logs it. Before that, the two
corrupt-looking reads above cost the whole region its layer while the
plate printed "✓ Maximum Intensity Projection · 1 well" and the window that asked was told
"finished in 4.6 s" — and `⚙ controls` then opened no tab, because `_window_operators()` was
honestly empty. `_workers._OperatorWorker._on_error` also logs the skipped field and its cause; it
was a bare `except`-shaped swallow whose only trace was a red dot on a mosaic cell.

**GATE 3 (`tools/gates.py --inventory`)** is the sweep that finds this class: every control of a
real `PlateWindow` AND a real `RegionViewer`, actuated, with a verdict per control — reaches /
neutralised / hidden / disabled / raised / no outcome. Its region window uses a pane whose
`mosaic` is a real `MosaicLayers` over a Qt-free `napari.components.ViewerModel`, so everything but
vispy's painting is production code. `--self-test` bolts a dead chip onto each window and requires
the gate to name it.

## Running the test suite: one full graphical run at a time

The suite drives real Qt widgets and real napari `ViewerModel`s, so a full run is RAM-heavy.
Several agents (or an agent and Julio) often work this repo concurrently in worktrees, and N
simultaneous full-suite runs can exhaust the host's memory — which then fails tests for reasons
that are the MACHINE's, not the code's (the same trap as ADR-0001's wall-clock gates).

The rules:

- **One full `python -m pytest tests/ -q` on this machine at a time.** While working, run the
  focused test files for the code being touched; save the full suite for one run at the end.
- A run that showed memory pressure (MemoryError, killed workers, wildly slow collection) is not
  an authoritative result. Rerun it solo and say that happened; never report it as a plain pass
  or failure.
- **The suite renders offscreen by default** (`tests/conftest.py` sets
  `QT_QPA_PLATFORM=offscreen` via setdefault): on the native platform every real widget test
  opens actual windows and macOS yanks focus to each one, stealing the keyboard from whoever is
  typing. Verified identical counts offscreen vs cocoa (2867 passed both ways, 2026-08-15).
  `QT_QPA_PLATFORM=cocoa pytest …` still gets real windows for a visual debug.

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
