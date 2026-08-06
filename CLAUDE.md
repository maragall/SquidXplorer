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

**Not supported, do not build against it**: GUI panels generated from `params` (`_op_panels.py` is
hand-written per operator). Named in the template README so a contributor is not misled.

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
