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

## Run-contract and read-path deepening (2026-08-15, branch arch-cards)

The second review pass's remaining cards (plan:
`AI-docs/SquidXplorer/in-progress/2026-08-15-run-contract-and-read-path-deepening-plan.md`).

**One kwargs contract on BOTH engine arms.** The record grew `accepts=` (the explicit
passthrough: kwargs the callable takes that cannot be Params — `_STITCH_PARAMS`' own comment
records the measured reasons) and `inner_param=` (the declared param naming the INNER operator
that shapes the output; stitch's and coordinate's is `z_operator`).
`_engine.split_operator_kwargs` is THE validator both loops call — an unknown key is refused BY
NAME before any directory exists (`write_plate` used to skip validation entirely on the region
arm) — and `_engine.operator_output(name, kwargs)` answers the writer's depth and pyramid kind
off the record (the literal `"z_operator"`/`"mip"` reconstruction in `write_plate` is gone;
minerva forwards `z_operator` only when the record takes it). `coordinate` declares
`z_operator`/`correct_illumination` and REFUSES the registration family. `run_plate`'s `n_fovs`
default is `N_FOVS_LOOP_DEFAULT` (each loop's own default) and the region arm refuses an
explicit crop, naming `regions={region: [fov, ...]}` — it was silently discarded.
`write_from_stream` resolves its scope through `scope_wells` like both engine loops (it was the
one left out: an ROI save owed every FOV of a mapped region and marked its own store
incomplete); a region operator owes ONE anchor field per region.

**The reader's identity is DECLARED: `source_id`** joined `contract.reader` — what every cache
key and staleness token builds from, promoted from the private `_path` five modules reached
into. The Zarr reader's is the acquisition ROOT (its `_path` is the STORE, so `plate_token`
statted sidecars that never exist there). `reader.parse_coordinates_csv` is the PURE half of
the coordinates parse; `_assemble_metadata` is the one 13-key Acquisition assembly (it was
copied four times); `_warn_recorded_mismatch` the one recorded-vs-observed cross-check.

**ONE walk of an NGFF store: `contract.store`** (`ome_attrs` v0.4+v0.5, `resolve_plate_dir`,
`level_paths`) — reader/_tilesource/_montage/_mosaic_source fold their four private copies onto
it (only reader's could read `.zattrs`), and `_tilesource.plate_layout_from_store` now EXISTS
under the exact name docs/plate-contract.md promises. Stitch's z picks stay `n//2` ON PURPOSE:
registration and the shared flatfield .npy are parity-bound to maragall/stitcher
(tests/test_integration.py pins the geometry); only display-side picks use `opening_z`.

**The pane seam has ONE headless adapter: `_napari_pane.model_pane_class()`** (real Qt-free
`ViewerModel` + real `MosaicLayers`) — conftest, GATE 3 and the walkthrough all take it from
there. `StubMosaic`/`StubLayer` are DELETED; tests cross the interface production crosses
(napari's list-typed contrast_limits, Colormap objects, layer identity for reuse, events for
history, configured — not replaced — dims). `RegionViewer._napari_viewer` is one branch:
`pane._viewer` IS `mosaic.model` on every real pane.

**`_pixels.read_pixels` is the world-µm pixel address**: `PixelRequest(bbox_um, out_px
ceiling, channel, time_point)` with the ladder pick INSIDE, composed by `_paste_field` over any
`TileSource` — no new placement rule, and the two adapters are pinned pixel-identical
(`tests/test_pixels.py`). `PixelRequest` names its frame (the ladder's corner convention).
Consumer migrations (loupe, gallery, fusers) are staged in the AI-docs plan, one measured pass
each. The montage's bare `ts.open` — the last raw tensorstore opener on the read path — rides
`_tsctx.HANDLES` now.

**The nuclei button reads the registry**: `_workers.nuclei_operator()` picks
cellpose-vs-watershed off `operator_available`, and the caller resolves it ONCE so the label,
the console line, the panel whose values reach the run and the run name the same algorithm
(`SpotParams()` was hardcoded — the panel's values never reached the pixels). The LUT record
travels WHOLE on paste (clim through `MosaicLayers.set_contrast`, `on` through
`set_channel_visible`; the plate copies rgb+on out through its own `channel_rgb`/
`channel_visible` readers).

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
petakit are pinned to commit SHAs in pyproject so two installs resolve the same bits.

**ONE file per platform** (2026-08-16). The installer artifact is a single file: the wheel and a
uv binary ride INSIDE the frozen bootstrapper (PyInstaller `--add-data`, read back through
`bootstrap.payload_dirs()` — `sys._MEIPASS/payload` first, beside-the-program and PATH as
fallbacks). `build-installer.yml` freezes it on windows-latest (`SquidXplorer-Setup.exe`),
macos-latest arm64 (one executable in an Info-ZIP that preserves the exec bit GitHub's artifact
zip drops) and ubuntu-22.04 (oldest supported glibc on purpose), then packs the Linux one as
`SquidXplorer-Setup-x86_64.AppImage` (AppDir + appimagetool; icon
`scripts/installer/squidxplorer.png`). Every job does a REAL core install into a scratch env and
imports the result — a dry-run cannot catch a bootstrapper that prints the right commands and
installs nothing. A finished install leaves a double-clickable launcher via
`bootstrap.create_launcher`, one per platform: a Windows desktop shortcut, a
`~/Applications/SquidXplorer.app` bundle (built locally, so unlike the downloaded setup binary it
carries no Gatekeeper quarantine flag), an XDG `squidxplorer.desktop` menu entry. A launcher
failure is said and never fatal. `bootstrap._interactive` gates the hold-the-window-open Enter
prompt on a real tty, so a piped or CI run never blocks. What is NOT done: the binaries are
unsigned — macOS first launch is right-click → Open, and Linux needs one `chmod +x`
(`scripts/installer/README.md` is the user-facing story).

**decon installs EVERYWHERE, and the GPU probe is a fact, not a shade** (2026-08-25, branch
gpu-setup; Julio: "It looks like it couldn't detect my GPU for Mac. What about
Linux/Windows?"). Measured: `torch` was declared nowhere (the MPS backend worked in the dev env
by accident, a customer Mac always ran CPU) and the installer shaded decon on every Mac and
every non-NVIDIA box because its probe was `nvidia-smi` alone. Now `bootstrap.gpu_backend()`
answers one of three, shown as the decon row's note and printed by the install: `GPU: CUDA
(petakit)` (an NVIDIA driver speaking CUDA 12; the installer adds the `decon-cuda` extra,
cupy-cuda12x, petakit's own path), `GPU: Apple (torch MPS)` (Apple Silicon; `torch` rides the
`decon` extra by `sys_platform == 'darwin' and platform_machine == 'arm64'` marker), or `CPU
only: <why>` (everything else, Intel Macs included; petakit's numpy path). decon defaults
checked on every machine. torch is Apple-only ON PURPOSE: `_decon_gpu` never runs on a CPU
torch device, the PyPI Linux wheel drags the CUDA runtime (gigabytes) and the Windows one is
CPU-only, so elsewhere it would be weight that changes no pixel. What is NOT done: CUDA torch
wheels (index-specific, download.pytorch.org) are installed by nothing here, NVIDIA is CuPy's;
and the petakit pin (64de19b) still hard-requires cupy-cuda12x, so `.[decon]` still fails to
RESOLVE on a Mac until petakit `cupy-optional` (97b06b0, `petakit[cuda12]`) is pushed and the
SHA bumped (an unpushed SHA is a 404 tarball). `build-installer.yml` still skips the decon
install on macOS for the same reason.

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

## Architecture-improvement runs: worktrees, AI-docs, live workspaces

The review → cards → apply flow ran 2026-08-14 (seven candidates, five applied same-day); these
are the rules it proved:

- **Implementation agents work in git worktrees**, one branch per independent card, so the tree
  Julio is typing in stays free. Cards that share files go to ONE agent sequentially — parallel
  worktrees over `_viewer.py`/`_region_viewer.py` would only have manufactured merge conflicts.
- **The review and its design docs go to AI-docs, committed AND pushed**, filed by semaphore
  state (`to-do/` for unimplemented designs, `done/` once approved and merged). The dashboard
  reads that repo every ~15 min; an unpushed doc is invisible to the team.
- **Other sessions work this repo concurrently, and a worktree may branch from THEIR lineage,
  not from main.** Measured on that run: the agent worktrees branched from `merge-installer`'s
  head, 17 commits ahead of main, so merging the refactor branches necessarily published another
  session's in-flight work with them. Before merging: find where the base lineage lives
  (`git branch --contains`, `git merge-base --is-ancestor`), never merge into or commit onto a
  branch checked out in another live worktree, and when a merge would carry someone else's
  unfinished work along, ask rather than proceed — that day it was wanted; the day it is not,
  it overwrites a live workspace's story.
- **Applied architecture work merges to main automatically** (Julio, 2026-08-15) — no separate
  go-ahead per branch. The gate is verification, not permission: each branch's full suite green
  solo, then ONE combined full-suite run on the merged result before calling it done (the
  combination is what no branch tested). The live-workspace rule above still bounds this: an
  automatic merge never gets to publish another session's unfinished lineage unasked.

## The plate is a NAVIGATOR, views are TABS (2026-08-16, ported from PR #12)

The plate-navigator UI, re-applied commit by commit from jsschwrz's `plate-navigator` branch
(PR #12, written against pre-rename `squidmip`; a textual merge could not cross the renames).
The PR's own `PLAN-plate-navigation.md` is the source design; the port decisions live in
`AI-docs/SquidXplorer/.../2026-08-16-plate-navigator-port-plan.md`.

- **A plain left-click on a well NAVIGATES the active view** while one is open — the selection is
  untouched, empty-space click still clears (the one click-driven deselect), double-click still
  opens, and the navigate defers one `doubleClickInterval()` so the two gestures cannot collide.
  The mode is a fact the plate is TOLD (`set_click_navigates`, one writer:
  `PlateWindow._refresh_plate_navigation`); `RegionViewer.show_region` adopts a foreign region in
  acquisition order through the ONE cursor. `_regions` is a read-only property over the cursor
  (`_seed_regions` is what the window was opened over and names it); `ViewerManager.note_focus` /
  `active_view()` make "which window is the user in" true and singular.
- **The working layout**: `PlateWindow(default_layout=True, tabbed_views=True)`, set ONLY by
  `main()` — the launcher asks for a layout, a library caller gets a bare window (and ~120 test
  `ingest()` calls stay cheap). Root ~1/5 of the work area (`_fontscale.default_root_width`),
  one lazy view over EVERY well beside it (`beside_rect`, placed by FRAME with client margins
  subtracted — the client-rect version put the title bar off-screen). Never `select_all()`.
- **`_view_deck.ViewDeck`**: views as tabs, one current, detach by drag; the whole `RegionViewer`
  reparents as one object and the napari canvas is never touched. `RegionViewer.dispose()` is the
  teardown wherever a view lives (a tab never gets a closeEvent; `request_close` goes through the
  deck), and it finally calls `MosaicPane.shutdown()` — the leak was one GL context + tens of MB
  per CLOSED window. `manager.tabbed_views` is the one policy point in `_spawn`. Deck `destroyed`
  connects a BOUND METHOD, never a self-capturing lambda (measured 0xC0000409).
- **The loupe engine lives in `_loupe.py`**, shared by the plate's press-and-hold and the canvas
  loupe (`_napari_loupe`, shift-left-click, wheel ladder with a session-scoped factor). The
  extraction preserved MAIN's fixes: the n//2 z pick, the memory-bounded LRU caches.
  `_pct_window` moved to `_montage` (one percentile rule below the GUI boundary);
  `_qthread_life.detach` is the QThread ownership rule on its own. The plate keeps only its
  gesture half and paints through the shared `paint_loupe_inset`.
- **The FOV walk**: `_mosaic_source.mosaic_fov_bboxes_um` is THE box geometry (top-left
  convention, NOT `_tilesource.fov_bboxes_um`'s centre — half a frame apart, measured 195.9 um);
  `_napari_view.camera_for_bbox_um` is napari's fit rule written once (canvas order is
  `(height, width)`; `_brick_view`'s crossed axes were the measured bug); `frame_bbox_um` frames
  one field, 2D-only, inside `programmatic()`. `RegionViewer(fovs=True)` refuses `roi_bbox` by
  name and draws its own Shapes layer, never the ROI layer.
- **Contrast sliders span the DATA's bit depth** (`_bitdepth`): measured, never declared (Squid
  writes MONO12 into uint16 and stamps no depth), monotone rising (C3 reads 3437, E7 16380 in one
  14-bit set), observed on FULL-RESOLUTION frames in the fusers (decimated planes under-state the
  max), crossing threads as `ViewerManager.depthChanged`. `MosaicLayers._widen_range` is the one
  never-narrows rule; `set_dataset` forgets the last acquisition's look (luts, visibility, LUT
  clipboard, ceiling).
- **Icon + drag-open, installer-only** (Julio: "I want one install story"). `scripts/installer/
  make-icon.py` draws the wellplate art once (.ico + .png); the .exe freezes with `--icon`;
  launchers carry it via `_installed_icon` (copied beside the env — the one-file extraction dir
  dies). Drag a folder onto ANY launcher to open it: the Windows shortcut and the Linux `%f`
  entry pass the path as an argument; the macOS `.app` declares `public.folder` and
  `_viewer._FileOpenFilter` catches the resulting `QFileOpenEvent` — installed BEFORE the window
  and buffering, because a launch-with-document's event lands during the splash's event
  processing (measured with a real bundle; a post-window filter missed it every time).
  `Setup-Windows.ps1`/`mip-tool.bat` are DELETED: rigs update by
  re-running the installer. What is NOT done: GL-in-tabs was hand-verified on the PR's Windows
  machine (4 canvases, 20 dock/undock cycles, ~88 MB/view) and the offscreen suite cannot
  re-check it; a hand check on this Mac is owed. Commit H (drag tabs BETWEEN windows) stays
  deferred, hooks in place (`dock_page`/`undock_page`, `_host`, `ViewerManager._decks`).

## Viewport slicing is ASYNC (2026-08-24, branch async-slicing)

Julio, live on the 900-FOV 20x set: "when I zoom in rapidly, it's not responsive." napari's
own async slicing (NAP-4) is on process-wide; the canvas keeps the previous rung while the
fine one computes on the slicer's pool. The rules:

- **Enabled ONLY via env**: `squidxplorer/__init__` sets `NAPARI_ASYNC=1` before the first
  `get_settings()` (`_async_slicing.configure`). NEVER assign the napari setting: any
  settings assignment autosaves the user's global settings.yaml (measured), while
  env-sourced values are excluded from every save. `SQUIDXPLORER_SYNC_SLICING=1` opts out;
  a user's own `NAPARI_ASYNC` wins.
- **napari's only ready-event consumer is QtViewer**, so production panes apply responses
  already; `_napari_pane.attach_async_slice_apply` is the same apply half for the headless
  ModelPane and test scenes — ALWAYS main-thread marshalled (an inline apply from the pool
  reaches Qt-connected listeners and aborts the process; measured SIGABRT).
- **`_reslice_hidden_layers` stays synchronous** through `_refresh_sync` with refresh's full
  flag set (`force` alone refreshes nothing) — the 2026-08-06 thumbnail race must not reopen.
- Tests whose pins are inherently sync-semantic (`test_time_point_playback`,
  `test_roi_pitch`) force each viewer's slicer sync via the per-viewer `_force_sync` knob;
  the suite otherwise runs the production (async) configuration.

## Operator shelf (Julio, 2026-08-21 + 2026-08-24, branch shelf-operators)

Julio's rulings, verbatim: "spot, shelf"; "cellpose, shelf"; Detect row -> "Shelve it all";
"Shelve keepz too"; "Shelf the background substitution logic"; and the 2D/3D decon merge
("3D decon would still use a 2D PSF, since there is no more to draw from"). The Minerva rule
applies: deleted whole, grep-proven, reinstating starts from git history, absences pinned in
tests (`test_operator_declaration.py::test_the_shelved_operators_are_gone_whole` and per-file
pins). **The surviving registry is exactly `mip`, `decon`, `stitch`, `register`.**

- **Shelved whole**: `_background.py` (bgsub + its exports + scikit-image core dep, whose one
  stated consumer was rolling_ball — napari declares scikit-image itself, measured, so the
  frozen build's `collect_all("skimage")` still resolves); `_spots.py` + `_cellpose.py` +
  `add_segmentation_operator` + the entire Detect surface (`_SpotWorker`, `nuclei_operator`,
  the pane's Detect row, `RegionViewer._detect_nuclei`, `_full_res_mip`) + the `segment`
  optional-dependency group; `keepz`; `coordinate` (register=False on `stitch` is the same
  run); `reference` with `project_reference`/`select_reference_z`, project_well's whole
  `select_index` arm (`reference_channel`/`picked_z` params included) and `_acq_output`'s
  hardlink writer arm (`_link_selected_planes` + `_per_plane_files`), whose last producer it
  was — `projection._tenengrad` stays, the GUI's z-slider autofocus reads it; and the
  `flatfield` OPERATOR (`_ACTIVE_OP`, `_correct_with_active`, `_profile_for`, `flatfield_op`,
  its card, and `run_operator`'s auto-estimate name-branch — KNOWN_NAME_BRANCHES is empty
  again).
- **decon IS the volume solve.** The 2-D per-plane `decon` was deleted and `decon3d` renamed
  onto the name: ONE code path (`deconvolve_stack`), PSF depth follows the stack depth, and the
  n_z=1 case was PINNED before `deconvolve_plane` was deleted — the volume solve over a 1-plane
  stack equals the 2-D in-focus solve (float32 max abs diff 0.00195, uint16 max 1 count, and
  the depth-1 3-D PSF's central plane renormalises exactly to the old in-focus slice).
  `deconvolve_plane`, `make_psf_2d`, `deconvolve`, `decon3d_op` are gone; `decon_op` builds the
  volume solve; `"decon3d"` is refused BY NAME with a pointer to `decon`
  (`_engine._resolve_operator`). The decon card (QC panel) is unchanged — it always ran the
  3-D solve. Backends (2026-08-25): petakit CuPy on NVIDIA, `_decon_gpu` torch/MPS on Apple
  Silicon (z-tiled over `min(recommended_max_memory, free RAM)` with petakit's own tile plan;
  the tiled solve equals petakit's tiled solve to 1 count and is NOT the whole solve, up to
  25% of peak at seams with a 95-plane PSF, the log line says so), petakit numpy elsewhere.
- **`z_operator=None` means KEEP EVERY PLANE** — the shelved `keepz`'s one load-bearing job.
  `stitch_region` resolves None to a module-local identity record (`_stitch._KEEP_EVERY_PLANE`,
  deliberately NOT in the registry), `operator_output` answers `(False, "intensity")` for a
  None inner operator, `RegionViewer._z_kwargs_for_mode` passes `{"z_operator": None}` for 3D
  full-z stitching, and the StitcherPanel's combo offers it as `_op_panels.KEEP_EVERY_PLANE`
  ("keep every z plane"), mapped to None before anything asks the registry. A stale recipe
  label saying "keepz" still PARSES (RecipeChain is textual); binding it is the registry's
  named unknown-operator refusal.
- **Deliberately kept**: the labels VOCABULARY (`labels_op`, `produces="labels"`, `LABELS`, the
  nearest-only labels pyramid reducer in `_output._REDUCERS`) — plugin/template surface,
  pinned by `test_the_labels_vocabulary_survives_the_shelf`; and stitch's whole flat-field
  MACHINERY (`FlatfieldProfile`, `estimate_profile`, `correct_flatfield`,
  `per_channel_from_npy`, the `set_profile(s)`/`active_profiles` store, `resolve_flatfield`,
  `_FlatfieldWorker`). The GUI loader survived the card cut as the NON-operator
  `illumination` card (`_viewer._build_illumination_tab`): load a stored .npy or estimate
  per channel from plate tiles; `_stitch._selected_profiles` reads what it installs.
- Tests grew two conftest fixtures replacing the shelved exemplars: `blob_operator` (a
  cardless, core, params-declaring labels plane-op — what `spot` was to the panel/CLI/runner
  machinery tests) and `identity_operator` (what `keepz` was to the acquisition-format writer
  tests). `CLI_ONLY_OPERATORS` is empty: every survivor has a card.

## Squid's downsampled well mosaics are read, and written back when absent (2026-08-19)

`squidxplorer/_wellimage.py` reads what Squid's SAVE_DOWNSAMPLED_WELL_IMAGES writes:
`<t>/mosaic_view/wells/<well_id>_<N>um.tiff`, a (C, Y, X) TIFF per well at the integer plate
factor `max(1, round(target_um / pixel_size_um))` (target 2 µm on every rig: 10x -> 3, 20x -> 6,
40x -> 12), origin = the well's min tile top-left, the SAME origin `fov_offsets_px` uses, so
the file lands on the fused mosaic with no conversion (positions are centres there and top-lefts
here; the half-frame shift cancels in the differences).

- **The factor is derived from the FILE'S OWN SIZE** against `mosaic_extent_px`, never from the
  objective table; a size fitting no integer factor, a corrupt file, or a missing channel reads
  as ABSENT with a named log line and the ordinary FOV fusion takes over. Measured parity on a
  smooth set: bit-identical to the area-mean of the fused native mosaic; a uniform 4-count
  sampling offset (mean vs stride) against the strided rung, no spatial misplacement.
- **`fuse_region_pyramid`'s rungs at step >= factor come from the well image** (nearest map on
  the shared origin, `_wellimage.resample_plane`), so first paint is one small file read: 24-well
  1024 px set, coarse-rung materialisation 2.3 ms vs 8.5 ms warm. Finer rungs fuse as ever.
- **`_PreviewWorker` seeds plate cells from the well images** after the cell cache and before
  the FOV walk (same 24-well set: 0.13 s vs 0.56 s cold, 0.056 s vs 0.45 s warm), publishing the
  cells to the cache; and after a finished preview it BACKFILLS an absent `mosaic_view/wells`
  through the vendored downsampler (`downsample_plane`, from Squid `mosaic_utils.downsample_tile`)
  so the acquisition afterwards looks microscope-produced. The backfill is best-effort (a
  read-only mount is logged, never fatal), atomic per file, and the ONE exception to "SquidXplorer
  never writes into your data": Julio's design doc asked for exactly this write.
- **A well image is ONE z** (the widget keeps the last plane blitted), so `n_z > 1` never serves
  one; the backfill writes z = n_z - 1, what Squid's canvas would hold.
- **A pad-tainted backfill is stamped, and completion deletes it** (2026-08-19). A backfill
  through a padded reader bakes the padded FOVs in as black; the file carries a `padded_fovs`
  stamp (`reader.padded_slots`, the `PaddedSlots` record of what `_pad_to_plan` invented). Once
  any stamped FOV has data on disk, `load_well_stack` DELETES the file — the one file this
  package deletes, provably ours (Squid never writes the stamp) and provably stale — and the
  next backfill rewrites it clean. A fully padded well or a padded z/t plane writes no file;
  the backfill skips per WELL (any-resolution existing file wins), not per timepoint, so the
  rewrite lands while Squid's own files stay untouched. Unstamped black files predating the
  stamp cannot be told from Squid's and are never deleted.
- `SQUIDXPLORER_WELL_IMAGES=0` turns the feature off; tests/conftest.py sets that default for
  the suite so preview tests keep their pinned read counts, and tests/test_wellimage.py is the
  coverage that turns it on.
## Every raw-preview rung is WINDOWED (2026-08-19, branch windowed-rungs)

Julio, live, the 452-FOV single-z set: "Zooming-in causes my viewer to stop responding." The
coarse rungs of `fuse_region_pyramid` were one whole-region `delayed` fuse each, and napari's
draw blocks synchronously on the slice it asks for — every zoom notch decoded all 452 frames to
show a viewport covering a dozen (0.35–3.7 s per rung per channel, measured).

- **Every rung rides `_WindowedLevel`** (coarse and fine, one paste rule): a viewport slice
  decodes only the FOVs under it — 25/64/187 reads at a 1000 px window on that set, bit-exact
  to `_fuse_levels` at every stride (pinned on the real data and in
  `tests/test_mosaic_source.py`). `_fuse_levels` survives as the reference the parity tests pin
  windows against; the nested `_plane` and its one-pass multi-level fill are gone.
- **Kept, each verified**: the well-image short-circuit (now `_WindowedLevel._whole_plane`,
  whole plane cached under the old plane key), the all-FOVs-unreadable refusal (per WINDOW
  now), one `_bitdepth` observation per decoded frame, z stacked per plane. A full-window
  compute caches the whole plane, so napari's coarsest thumbnail re-pull stays a ~1 ms hit.
- **The cache holds only what FITS as a set** (`_region_fits`): 452 × 3.6 MB frames against
  the 465 MB budget flooded out every smaller entry, so a warm slice re-decoded its FOVs.
  `_sub` caches the rung's own step-s subframes (s² smaller — the whole region fits at coarse
  steps); full frames are cached only when the region's worth fits. Warm coarse slices: 0
  reads. The trade: stride-1 pans on an over-budget set re-decode the viewport's few FOVs.
- The z concatenate blocks dask's exact-window fusion for nz > 1 (true of the fine rungs since
  they shipped): the honest grain there is the 2048 px CHUNK, one z — pinned, not fixed.
- No whole-plane rungs were kept: the coarsest is a single chunk (one Python call), and the
  off-thread contrast seed measured 0.41–0.69 s against 0.36 s before — same order, while every
  on-thread coarse slice dropped from 452 decodes to the viewport's own.

## Plate/views simplification (2026-08-19, branch plate-simplify, Julio's annotated mocks)

- **The Window navigator is DELETED** (`OpenViewList`): the deck's tabs superseded its list.
  What survived, where: close-all in View > Close All Views (`PlateWindow._close_all_views`),
  the memory/run bars as `_region_viewer.StatusRow` (still adopted by the log panel),
  `ViewerManager.rename` (repaints the deck tab text). `raise_views`/`collapse_all`/
  `make_default` died with their only callers; the GUI has no rename affordance left (owed).
- **The readout strip is a LOG LINE**: `PlateWindow._readout` is `_LogReadout`, a shim whose
  `setText` logs (INFO status, WARNING for refusal-shaped text, consecutive duplicates dropped)
  and whose `text()` still answers for tools/gates and `_gui_commands`. The per-FOV run tick no
  longer writes it — the same sentence is live on the log panel's work bar. The status bar
  carries the Selection caption instead.
- **The Operators cards live in the VIEWS window**: `_operator_dock.OperatorDock`, a right-edge
  dock collapsed by default to a vertical grip, installed ONCE per deck / free window by
  `ViewerManager` through `operator_dock_installer` (set by the plate). It holds the plate-built
  card launcher (`_build_operator_cards`; cards still open tabs in the PLATE's `_left_tabs`,
  which hides itself while empty so the LOG owns the band) AND the current view's
  `RegionViewer.operator_panel()` — the old "Operators for this window" toolbar plus the pane's
  Detect row — swapped on `pageActivated`. Julio earlier rejected a centre-top operators chip;
  the right-edge dock is the explicitly requested different thing.
- **LUT copy/paste is SHELVED completely** (Julio: "adding complexity for no reason"): the plate
  pair, the window chips and `_lut_clipboard.CLIPBOARD` are gone. What survives is not the
  clipboard: `per_channel_luts`/`apply_luts`/`match_raw_contrast`, the automatic window → plate
  contrast tap, and the stain `display_lut` path. The Defaults group (auto focus / make default /
  diverged / reset) is shelved too; the `ViewSettings`/`ViewDefaults` STORE stays (autofocus-on-
  open, child-window LUT inheritance).
- **3D opens in a NEW deck tab** (`RegionViewer._open_3d` spawns a child via `open_child`,
  `_volume_view.open_3d(child, scene_from=parent)` harvests LUTs from the 2D scene); the 2D tab
  is untouched. `reveal()` only `showNormal()`s a MINIMISED target (it used to de-maximise the
  deck), and `_spawn` un-minimises a deck before docking a new tab into it.
- Kept on purpose (zarr inventory): the stitch writer, `_open_computed`, `_ZarrLoupeSource`,
  `resolve_plate_root`, `incomplete_reason`. Deleted: the write-only `_processed_plate`.

## Minerva shelved (Julio, 2026-08-19, clean-architecture pass)

The Minerva Author export/render product is DELETED whole: `_minerva.py`, `_minerva_panel.py`,
`_MinervaWorker`/`_MinervaRenderWorker`, the non-runnable "minerva" card, the `PlateWindow`
delegates, `PlateWindow.on_screen_luts` (its only consumer was the Minerva tab's checkbox) and
`_run_scope.subset_selection` (its only purpose was the export's expansion). Kept because they
have non-Minerva callers: `_lut_clipboard.per_channel_luts` (loupe, movie, settings snapshots),
`colormap_hue_rgb` and `selected_region_fovs`. Dated docs under docs/ (IMA-228 review, SCOPE,
DESIGN) still tell its story on purpose; re-instating starts from git history, not a stub.

## Chroma reconstruction and the pad-partial format matrix (2026-08-19, branch stain-chroma)

- **A color-recorded-gray channel with a usable overview displays through VIRTUAL CHROMA
  CHANNELS**, not the stain LUT: the reader expands it into the same (R)/(G)/(B) components a
  real color file gets; (G) is the file's own plane, (R)/(B) are that plane times the overview
  PNG's local R/G and B/G ratio windows (`_stain.ChromaSource`, bilinear upsample, ratios
  clipped to `_RATIO_MAX`). The LUT stays the fallback when the PNG/geometry is absent. The
  geometry convention is MEASURED, not assumed: the mosaic yaml's `top_left_mm` is (y, x) and a
  FOV's stage position is its CENTER (verified corr 0.88 at zero offset; the corner convention
  measured 0.01). An uncovered FOV reads its own plane with ratio 1 (neutral), counted in one
  log line.
- **pad_partial reaches all three formats** through `reader._pad_to_plan`/`_PadPartialMixin`:
  individual images (the original), OME-TIFF (plan = root coordinates.csv + declared Nz/Nt),
  Zarr HCS (plan = the plate metadata's declared wells + field_count). No plan record = a named
  no-op, never a guess. The padded open's warning says it places by the PLAN on purpose; the
  multipage-TIFF reader still ignores the flag (not one of the design's named formats).

## Second dead-code pass (Julio, 2026-08-19: "Less is more")

Deleted whole, each grep-proven caller-free: `_odon.py` + the `--odon` CLI arm ("we never use
that"), `_pixels.py` (the staged pixel-address abstraction; its consumer migrations never
happened — reinstate from git history if they do), `build_montage` + the static HTML montage
half of `_montage.py` (the compositor/percentile/downsample half is LIVE — the plate renders
through it), `ViewerManager.rename` and `_TO_BE_ADDED`. Kept on purpose: the
`ViewSettings.reset/adopt/diverged` trio — it is the settings-inheritance tests' own
infrastructure, not a corpse.

## Multi-acquisition sets (2026-08-19, branch acq-folder)

`_acqset.py` (Qt-free): a SET is a folder that is not itself an acquisition but has >= 2
immediate child acquisitions (acceptance = `open_reader` accepts it; written plates excluded via
`resolve_plate_root`). Ingest opens the first member and records the set; cycling (title-bar
prev/next, Cmd/Ctrl-Left/Right) is a RE-INGEST of the neighbour — one data model. Bulk saves:
the "save runs: all N acquisitions" checkbox routes an unscoped SAVE through
`_acqset.run_over_set` (`_acqset_gui.SetRunWorker`): sequential, one `run_operator_once` per
member with the SAME parameters, per-member fault isolation, a measured tally line. ONE operator
per run; composition stays refused. Owed: per-set disk estimate; a set-run cancel button.

## Views-window space reclaim (2026-08-19, branch views-space, Julio's live-GUI feedback)

- **Everything window-scoped lives in napari's LEFT column, above the layer controls**: the
  2D/3D·ROI chip block (`RegionViewer._build_view_controls`, chip attribute names unchanged)
  and, directly under it, `operator_panel()` (dropdown, ⚙ controls, Run, save, Detect row).
  `MosaicPane.dock_view_controls` is the seam (napari `add_dock_widget` + a remove/re-add hoist
  of the other left docks); a pane that cannot dock (headless ModelPane) keeps the column in
  the window body so tests and GATE 3 still actuate every control. `_build_top_row` is deleted
  — the full-width top toolbar is gone and the canvas gets its height. The run/movie progress
  bar stays in the window body (visible whatever the columns do).
- **The right-edge `OperatorDock` holds ONLY the bulk-processing cards** (Julio: "The operators
  for this window row should also be on the left vertical dock. The bulk processing is what is
  solutioned on the right vertical column."): `_panels`/`show_window_panel` and the
  `pageActivated` panel swap are deleted. Collapsed, the dock is a FULL-HEIGHT dark grip (a
  QStackedWidget page under a zero-height title bar) — the old grip-as-title-bar left the
  dock's empty content area painted platform-white for the window's whole height.
- **"Match layers to raw" is shelved whole**: the button, `RegionViewer._match_raw_contrast`,
  `_lut_clipboard.match_raw_contrast` and `MosaicLayers.match_contrast_to` (the button chain
  was its last caller). Absences pinned in tests.
- **The LUT clipboard is back as exactly two buttons** (Julio: "ultra simple, minimal, two
  button logic"): `_lut_clipboard.CLIPBOARD` (one plain dict) + `copy_luts`/`paste_luts` over
  the surviving `per_channel_luts`/`apply_luts`, driven by two chips in the view-controls
  block. **The plate follows a PASTE and only a paste**: `RegionViewer.lutsPasted` →
  `PlateWindow._follow_window_luts` → `follow_channel_window` (contrast only, never the manual
  latch, colours untouched so a stain-LUT plate look survives). A drag still leaves the plate
  alone — both 2026-08-06 ("the plate image shouldn't change unless we paste a LUT") and the
  paste-parity requirement hold; `test_a_lut_paste_reaches_the_plate_and_the_two_agree` pins
  the parity.
- **A copy-saving operator's preview run names its artifact**: `RegisterPanel.copy_check`
  defaults CHECKED (the copy is the operator's purpose; default-off was "Registering the wells
  doesn't do anything"), `operator_done` appends "preview only — tick save to write
  stitched_<acq>" for an `operator_saves_copy` run that wrote nothing (`_op_run_wrote`,
  recorded at launch), and the window save box's tooltip names stitched_<acq> instead of
  implying an OME-Zarr when a copy-saving operator is selected.

## No em or en dashes in GUI strings (Julio, 2026-08-24)

Every user-facing string — widget text, tooltips, captions, banners, refusals, and anything
that reaches the log panel — uses commas, colons, periods, or hyphens, never `—` or `–`.
Enforced by `tests/test_console_readable.py::test_no_gui_string_carries_an_em_or_en_dash`,
which sweeps every non-docstring string literal in `squidxplorer/` (a dash in a Python string
is prose bound for a human surface). Docstrings and comments are out of scope.

## ONE WINDOW + one flow + the knob principle (2026-08-25, branch one-window-dock)

Julio's declutter directive (chart: `~/Downloads/napari dock all states.png`; plan:
`AI-docs/SquidXplorer/in-progress/2026-08-25-one-window-slot-dock-plan.md`). The rules:

- **ONE WINDOW**: the plate window keeps every book (runs, ingest, selection) but its VIEW
  (`PlateOverview`) and the LOG render as SLOTS in the current view's left column.
  `_viewer._PlateSlotBox` is the plate slot: FIXED 240 px, collapsible to a grip; the log
  opens at 3/4 of it. `ViewerManager._sync_plate_slots` hosts on spawn and tab activation
  and re-homes on close; `PlateWindow.adopt_plate_slots_home` is the never-an-orphan
  guarantee (with no views left the plate window shows again - the app always has a
  surface). In the working layout a hosted plate HIDES and the deck takes the work area;
  `ViewDeck.bind_plate` makes the deck an app surface (drop-to-open, File/View menu
  forwards, a "Plate Window" action to bring it back). Ingest mounts the rebuilt overview
  through ONE mount point (`_mount_overview`), so a re-ingest lands in the live slot.
  Widgets that can now die inside a hosting view tolerate it at teardown (the overview's
  and log panel's timer stops, StatusRow's self-unhooking slots - each was a measured
  abort out of a closeEvent).
- **One flow, fewer buttons**: the operators row is Preview (this view's regions,
  save=False) and Run on plate (save=True, plate-selection scope - also the set-bulk path).
  "No body runs on whole plate to preview." Deleted with absence pins: the row's save
  checkbox, every panel-level run/save button (StitcherPanel, GenericOperatorPanel,
  RegisterPanel), the plate's destination-picker run tab, the right-edge `_operator_dock`
  and its bulk cards, `_qtstyle.operator_card`. ⚙ controls INSERTS the operator's live
  panel (the plate's `_op_tabs` widget, one source of truth) into a param slot under the
  operators row; a second click removes it and the plate re-adopts the panel (never a
  parentless orphan - one measured a segfault). A panel's refused setting PROPAGATES out
  of `operator_kwargs_for` as the launch refusal (it was swallowed into running defaults);
  the keep-every-plane combo label maps to `z_operator=None` through
  `_op_panels.z_operator_choice`.
- **The knob principle**: "the user should only tweak what can't be deduced from
  acquisition filenames." `Param.advanced` (declaration-driven, never a name match) sends
  a knob to a collapsed "advanced" section; headline is the exception. Decon's headline is
  iterations + NI (the one PSF input no Squid file records); the NA edit is CUT - NA, mag,
  dxy, dz, nz display read-only in the on-demand PSF slot, and a wrong rig profile is
  handled by GUARDRAILS (`_decon.rig_profile_notes`: pixel size vs sensor x binning /
  magnification, folder-name objective vs record, dz vs axial PSF extent - one advisory
  log line each, the run proceeds on the record). Stitch declares registration_channel /
  registration_t / correct_illumination advanced; its panel headline is z-handling +
  register on/off.
- **The turbo preview is REMOVED**: the QC sweep's picture is the in-view data layer under
  the channel's OWN colormap; `DeconQCResultView` keeps the stepper, halo/core caption,
  mean-delta metric and "use k iterations". `qc_composite`/`composite_centre_at`/
  `turbo_rgb`/`GAP_RGB` are deleted from `_decon_qc` (the CLI montage stays).
- **A depth-keeping preview shows the z the asking view is on**: the run carries
  `z_level` (view `_z_slider_index` -> `run_operator` -> `_OperatorWorker`), the plane pick
  clamps to the result's depth, dims stay put (pinned in `tests/test_decon_z_in_view.py`).
  An ROI preview scopes to the box's FOVs (`{region: [fov, ...]}`), every channel.
- **`RegionViewer.dispose` disarms the region/timepoint debounce timers** - a pending
  single-shot fired into a torn-down window during the deleteLater drain (deterministic
  segfault at PYTHONHASHSEED=0, gone with the disarm).

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
