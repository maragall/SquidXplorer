# The SquidXplorer operator template

**Copy this directory. Rename the package. Replace the algorithm. Keep the shape.**

This is a complete, installable example of a SquidXplorer *operator* living in its own package.
After `pip install -e .` into the environment that runs SquidXplorer, the operator it declares
appears in `available_projectors()`, in the CLI's `--projector`, in the viewer's runnable-operator
list and in the `list_operators` command — with **no edit to SquidXplorer and no fork**. (It does
not get a *card* in the viewer's operator panel; see §3.)

```
templates/operator/
├── pyproject.toml                        the entry point that makes this a plugin
├── README.md                             this file: THE CONTRACT
├── squidmip_operator_template/
│   ├── __init__.py                       register(): the four declarations
│   └── _stdev.py                         the algorithm
└── tests/test_template_operator.py       the four groups of tests every operator needs
```

---

## 1. Install it

```bash
pip install -e templates/operator          # into the environment that runs SquidXplorer
python -c "import squidmip; print(squidmip.available_projectors())"
# ['bgsub', 'cellpose', 'decon', 'decon3d', 'flatfield', 'mip', 'reference', 'spot', 'stdev']
#                                                                                    ^^^^^^^
# it runs, headless, over a whole plate:
squidmip /path/to/acquisition --projector stdev --output-folder /tmp/out

pip uninstall squidmip-operator-template                        # and it leaves cleanly
```

Or from Python, which is what the template's own tests do:

```python
import squidmip
reader = squidmip.open_reader("/path/to/acquisition")
for region, fov, image in squidmip.project_plate(reader, projector="stdev",
                                                 operator_kwargs={"smooth_sigma": 0.0}):
    ...                                    # (T, C, 1, Y, X), the acquisition's native dtype
squidmip.write_plate(reader, "/tmp/out.hcs", projector="stdev")   # navigable OME-Zarr plate
```

---

## 2. THE CONTRACT

### 2.1 What an operator IS

One callable, one shape, for every operator in the system:

```python
operator(planes: Iterable[np.ndarray]) -> np.ndarray
```

`planes` is an iterable of 2-D `(Y, X)` arrays — all the same shape, all the same dtype, the
acquisition's native dtype (`uint16` on Squid hardware). You return **one** 2-D `(Y, X)` array.

That is the whole interface. There is no base class, no `Operator` subclass, no lifecycle, no
`setup`/`teardown`. What varies between operators is not the callable's shape — it is **what the
engine puts in the iterable**, and that is decided by one declaration you make.

### 2.2 `consumes` — what arrives, and what shape comes out

| `consumes` | what one call receives | what you return | result shape | examples |
|---|---|---|---|---|
| `frozenset({"z"})` (default) | **every z-plane of one `(t, c)`** — the whole stack | one plane | `(T, C, 1, Y, X)` — z collapses to 1 | `mip`, `reference`, `stdev` |
| `frozenset()` | **exactly one plane** | one plane | `(T, C, Nz, Y, X)` — z survives at full depth | `decon`, `bgsub`, `flatfield`, `spot`, `cellpose` |

The engine has ONE loop. It groups the FOV's planes over the axis you declared and calls you per
group. Nothing branches on your operator's name — a test in SquidXplorer
(`test_no_module_branches_on_an_operator_name`) fails the build if any module compares against a
registered operator's name without a written justification.

For a plane-op, write a natural `plane -> plane` function and lift it:

```python
from squidmip import plane_op, add_projector
add_projector("mything", plane_op(my_plane_function))     # consumes inferred = frozenset()
```

**`consumes={"fov"}` is refused by name.** An operator is `Iterable[plane] -> plane` and never sees
a tile's stage geometry, so anything inter-FOV (stitching, fitting an illumination field across a
well) cannot be expressed here. That work belongs to the *region operator* table
(`squidmip.add_region_operator`), whose entries take `(reader, region, fovs) -> (T, C, Nz, Y, X)`.

**Stream. Do not call `list(planes)`.** The engine runs several wells concurrently and its peak
memory is (wells in flight) × (one well's footprint). Materialising a stack multiplies that by the
stack depth. `mip` is a running maximum and `stdev` is Welford's algorithm for exactly this reason;
both hold two planes regardless of depth. An operator that genuinely must materialise (an EDF, say)
is allowed and owns its own documented memory profile — say so in its docstring.

### 2.3 `produces` — what your pixels MEAN, and how they are drawn

| `produces` | meaning | rendering |
|---|---|---|
| `"intensity"` (default) | the pixels measure light | napari **Image** layer: windowed, colormapped, blended additively across channels |
| `"labels"` | the pixels are integer **object ids** | napari **Labels** layer: 0 is transparent, never windowed, never interpolated, click-to-pick |

This is not cosmetic. Before this declaration existed, `spot` emitted a label image down the
intensity path and a segmentation arrived auto-windowed by the fluorescence contrast rule, as if
label 37 were 37 photons. Declaring `"labels"` also requires an integer dtype; a float array with
`produces="labels"` is refused by name, naming your operator.

A kind SquidXplorer cannot draw is refused, not silently drawn as intensity. If you need a third
kind (points, a mesh), that is a change to SquidXplorer's delivery path, not something you can
declare your way into.

### 2.4 `params` — how your parameters are declared and how they reach the GUI

```python
Param(name: str, default: Any, blurb: str = "")
```

Declaring `params=` changes what `add_projector` does with your callable: **it is read as a
factory.** SquidXplorer calls it once with your declared defaults to build the entry's default
binding, and calls it again with a caller's `operator_kwargs` for a run that names different
values.

```python
add_projector("stdev", stdev_op, params=(Param("smooth_sigma", 1.0, "..."),))

squidmip.bind_projector("stdev", {"smooth_sigma": 0.0})            # a different binding
squidmip.project_plate(reader, projector="stdev",
                       operator_kwargs={"smooth_sigma": 0.0})      # a run at that binding
```

So ONE entry covers every setting. Do not register `stdev_tight` and `stdev_loose`.

A parameter name you did not declare is **refused by name**, listing what you do accept. Accepting
and dropping it would run at defaults while the console line and the recipe both said otherwise —
a wrong result that looks right.

A `Param` has a name, a default and one line of prose. Deliberately **no type, no range, no widget
hint**: the moment it carries a widget hint it has become the GUI's schema and two places own the
same fact.

**How parameters reach the GUI, stated honestly.** `squidmip.projector_params("<name>")` is the
public accessor, and `list_operators` reports every parameter with its default, so an **agent or
script driving the app can set them today**. The desktop GUI, as of 2026-08-05, **does not yet
build widgets from this declaration** — it hand-writes a panel class per operator in
`squidmip/_op_panels.py`, and an operator with no hand-written panel gets no widgets and runs at
its declared defaults. Your operator therefore *runs* from the GUI at its defaults, and is *tuned*
from the CLI, the command surface, or a script. A generic panel driven by `projector_params()` is a
known, named gap in SquidXplorer, not something your package can fix.

**What a `Param` cannot be: a channel.** The engine calls you per `(t, c, z-group)` and hands you
planes only, so a value bound at registration never learns which channel it is looking at. If your
operator must be specialised per channel (a PSF at the right emission wavelength, say), stamp a
`for_channel(acquisition_path, channel) -> operator` attribute on your callable; the engine calls
it once per channel and runs what it returns. See `squidmip._decon.optics_for_channel`.

### 2.5 `requires` — how you declare dependencies, and what happens when they are missing

```python
add_projector("stdev", stdev_op, params=(...), requires=("scipy",))
```

Importable **module** names — what you `import`, not what pip installs, when the two differ
(`scikit-image` installs the module `skimage`; declare `skimage`).

What happens when one is missing:

| | |
|---|---|
| **Listed?** | **Yes, always.** `available_projectors()` still contains your operator. Dropping it would make "scipy is not installed" and "nobody wrote this operator" look identical to the user. |
| **`operator_available("stdev")`** | `(False, "operator 'stdev' needs scipy, which is not installed (pip install scipy)")` |
| **`list_operators`** | the row carries `"available": false` and `"unavailable_reason"`, and the summary line names every unavailable operator |
| **A run** | **refused by name, before any well is read.** `bind_projector` raises `MissingOperatorDependency` with that sentence. The headless command surface returns the refusal code `unavailable_operator`; the GUI puts the sentence in the readout and never starts a worker. |

**Declare a module even if you import it lazily, several calls deep.** This is the direction that
bites, and it is measured rather than hypothetical: three of SquidXplorer's own operators (`decon`,
`decon3d`, `flatfield`) imported packages that were not in its dependency list at all. On a stock
install they were advertised, they raised `ImportError` one call deep, per-well fault isolation
(`project_plate(on_error=...)`) recorded that as one skip per well — and a whole-plate run finished
green having written nothing. `requires=` is what turns that into a refusal.

Declare what the entry needs **at its declared defaults**. `stdev` declares `scipy` even though
`smooth_sigma=0` would never reach the import, because the default is `1.0`. Under-declaring is the
dangerous direction; over-declaring only greys out a row.

### 2.6 What SquidXplorer guarantees you in return

* Your operator is dispatched by the same loop, on the same threads, with the same bounded-memory
  window as the built-ins. You get whole-plate parallelism for free.
* It is delivered to napari by the same path, with the layer type your `produces` chose.
* It appears in the CLI, the viewer, `list_operators` and the recipe/console line automatically.
* It is validated by SquidXplorer's own test suite: `tests/test_operator_declaration.py` is
  parametrised over `available_projectors()`, so tests written before your operator existed run
  against it the moment it is installed.
* Its results are written by `write_plate` as a navigable OME-Zarr plate — z-reducers at `Nz=1`,
  plane-ops at the acquisition's full depth.

---

## 2.6 Composition: your operator chains with the shipped ones, for free

Your operator is composable the moment it is registered. A chain is written wherever a name is,
separated by `+`, with parameters in parentheses:

```python
project_plate(reader, projector="flatfield + my_operator(smooth_sigma=2.0) + mip")
write_plate(reader, out_dir, projector="my_operator+mip")
stitch_region(reader, "B2", fovs, projector="my_operator+mip")
```

That string is exactly what `RecipeChain.label()` prints, so a console line or a pasted recipe
script is a runnable expression. The composed operator's four declarations are **derived from its
parts** — `consumes` is the union, `produces` is the last step's, `requires` is the union, and
`params` are namespaced `my_operator.smooth_sigma` — so you declare nothing extra.

What your `consumes` buys you, and what it costs:

| chain | composes? |
|---|---|
| plane-op → plane-op | yes, `Nz` survives both |
| plane-op → z-reducer | yes, the reducer consumes what the plane-ops produced (lazily — the stack is never resident) |
| **z-reducer → anything** | **refused by name.** After a reducer there is one plane and no stack. A z-reducer is the LAST step or the only one |
| **`produces="labels"` → anything** | **refused by name.** The next step would do arithmetic on object ids |
| a z-SELECTING operator (`select_index`) inside a chain | **refused by name.** Its z is solved on raw planes outside the operator, so a chain around it would never touch the planes it picks |

Refusals name both operators and say what to do instead; nothing is ever silently reordered. See
`squidmip/_compose.py`.

---

## 3. What this template deliberately does NOT support

Stated so you do not build against something that is not there.

* **A GUI panel from your declaration.** See §2.4. Parameters are reachable from code and from the
  command surface today; the desktop panel is hand-written per operator.
* **Inter-FOV work through this table.** See §2.2 — `add_region_operator` instead. Be warned that
  it is a WEAKER contract than the one this README documents: a region operator can declare
  `requires=`, and nothing else. `consumes` is implicitly `{"fov"}`, `produces` is hardcoded to
  `"intensity"` by every reader of that table, and its parameters are undeclared `**kwargs` that
  nothing validates and no UI can enumerate (`_op_panels.STITCH_DEFAULTS` mirrors `stitch`'s by
  hand, from private constants). Adding an inter-FOV operator is possible and it is not yet this
  template's contract.
* **Registering into the viewer's card table.** The card list (label, blurb, ordering) lives in
  SquidXplorer and a plugin cannot add to it. Your operator is *runnable* and *listed* without a
  card; `operator_label()` falls back to the registry name. A card is presentation, the engine is
  capability, and the two are asked separately.

---

## 4. Failure modes, and what each one looks like

| what you did | what happens |
|---|---|
| `from squidmip import ...` at **module scope** in your plugin | `OperatorPluginError: ... AttributeError: partially initialized module ... (most likely due to a circular import)`. SquidXplorer loads you from inside `import squidmip`, so importing it back at module scope is re-entrant. **Import squidmip inside `register()`** — the template does, with the reason written next to it. This is the first mistake everyone makes. |
| Your operator's name is already taken | `add_projector` raises at import; `import squidmip` fails naming your plugin, its distribution, and the collision |
| Your module raises on import | `OperatorPluginError` out of `import squidmip`, naming your plugin and the original error |
| Your `register()` raises | same, with "raised while registering its operators" |
| Your entry point points at nothing | same, at `ep.load()` |
| A `requires=` module is missing | operator LISTED, run REFUSED by name — see §2.5 |
| You declared `consumes={"fov"}` | refused at registration, pointing at `add_region_operator` |
| You declared `produces="mesh"` | refused at registration, listing the kinds that exist |
| You declared a parameter twice | refused at registration — a duplicate makes `operator_kwargs` ambiguous |
| Your factory returned something not callable | refused at registration, explaining that `params=` makes the registered object a factory |

A broken plugin **stops `import squidmip`**. That is deliberate: the alternative is an application
that silently does not have the operator you installed, which is the same defect `requires=` exists
to end, one layer up. The escape hatch, for a user whose app will not start, is
`SQUIDMIP_NO_PLUGINS=1`, and it is named in every one of those error messages.

---

## 5. Checklist before you publish

- [ ] The package name and the entry-point key identify **you**, not this template.
- [ ] The operator name identifies the **algorithm**, and is unlikely to collide.
- [ ] `consumes` matches what the code actually does — a test asserts it (group 3 in the tests).
- [ ] `produces` matches what the pixels mean, and the dtype suits it.
- [ ] Every parameter is a `Param` with a default and a blurb. No module-level globals: the engine
      calls you from several threads at once.
- [ ] Every module you import — including lazily — is in `requires=` **and** in `dependencies`.
- [ ] The operator streams; it does not call `list(planes)`.
- [ ] Errors are raised, never swallowed. A partial result that looks whole is the failure this
      project bans.
- [ ] `pytest tests/` passes, including group 4 (the plugin actually loads).
