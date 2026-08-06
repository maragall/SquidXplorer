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
`STITCH_DEFAULTS` is **not** derived from a declaration and cannot be — `add_region_operator`
carries no `params=` at all — but it no longer mirrors private `_stitch` constants either: it reads
`stitch_region`'s own signature (`_op_panels._stitch_default`).

Measured while building it: `_workers._OperatorWorker`'s PREVIEW branch called `project_plate`
without `operator_kwargs` while the save branch passed them, so a panel value reached the console
line and not the pixels (57 labels vs 57 at `min_area_px` 30/400; 57 vs 44 once fixed).

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

Bricking (many textures + GL `max` compositing + a 1-voxel halo) is the mechanism underneath and is
pixel-exact, but it is NOT what a drawn ROI takes any more. Do not route a whole region through it
expecting interaction — see the measured cost in the contract.

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
