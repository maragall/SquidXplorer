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
| `consumes` | the engine's loop and the output shape — `{"z"}` collapses z, `frozenset()` keeps it |
| `produces` | what the pixels MEAN, and therefore the napari layer type |
| `params` | what one entry can be RUN with (`params=` makes the registered object a factory) |
| `requires` | the modules it needs — **listed either way, run refused BY NAME when missing** |

`requires=` (2026-08-05) is the same word on all three registrars — `add_projector`,
`add_region_operator`, `add_segmenter`. It closed a measured silent success: `decon`, `decon3d` and
`flatfield` import packages absent from `[project.dependencies]`, raised ImportError one call deep,
and `project_plate(on_error=...)` filed that as a per-well skip — a green run that wrote nothing.
Per-well fault isolation now refuses to absorb `ImportError` / `MissingDependency`
(`_engine._NOT_A_WELL_FAULT`): a missing package is not a corrupt well.

**Discovery**: `squidmip/_plugins.py` scans the `squidmip.operators` entry-point group on
`import squidmip`, AFTER the built-ins. An operator in someone else's package needs no edit here.
A broken plugin aborts the import, NAMED; `SQUIDMIP_NO_PLUGINS=1` is the escape hatch. The
hardcoded built-in imports in `squidmip/__init__.py` stay — discovery is additive.

**Not supported, do not build against it**: composition (`_recipe.RecipeChain` documents chaining
and nothing executes it), and GUI panels generated from `params` (`_op_panels.py` is hand-written
per operator). Both are named in the template README so a contributor is not misled.

## Two producers of a region's pixels, and they are not interchangeable

`stitch_plate()` is the mosaic **of record**: registration, fusion, native resolution, ~0.9 GB per
27-FOV well. `_mosaic_source` / `_gallery` produce **preview placement**: FOVs pasted at their stage
coordinates, later-overwrites-earlier, decimated on read. Both go through the SAME `_placement`
helpers, so they are one geometry at two resolutions — never two implementations. Anything that is a
LOOK (pane 2's mosaic, the plate cells, Gallery View) takes the preview path; anything that is a
RESULT takes `stitch_plate`. Adding a third placement rule is the defect shape this repo has the
most of, because the error renders as a plausible image.

A **FOV subset of a region** is expressed one way everywhere: the mapping `{region: [fov, ...]}`,
which is `stitch_plate(regions=…)`'s own parameter and `GalleryScope.fovs`. It is produced by
`PlateWindow.selected_region_fovs()` from the plate selection (marquee, shift-click, Cmd/Ctrl-A).
Do not build a second selection mechanism; `_placement.fov_offsets_px` normalises whatever FOV set
it is handed, so the crop needs no cropping code.

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
