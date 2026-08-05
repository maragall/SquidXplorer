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
