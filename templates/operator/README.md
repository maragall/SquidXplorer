# The SquidXplorer operator template

**Copy this directory. Rename the package. Replace the algorithm. Keep the shape.**

This is a complete, installable example of a SquidXplorer *operator* living in its own package.
After `pip install -e .` into the environment that runs SquidXplorer, the operator it declares
appears in `available_plane_operators()`, in the CLI's `--operator`, in the viewer's runnable-operator
list and in the `list_operators` command — with **no edit to SquidXplorer and no fork**. (It does
not get a *card* in the viewer's operator panel; see §3.)

```
templates/operator/
├── pyproject.toml                        the entry point that makes this a plugin
├── README.md                             this file: THE CONTRACT
├── squidxplorer_operator_template/
│   ├── __init__.py                       register(): the four declarations
│   └── _stdev.py                         the algorithm
└── tests/test_template_operator.py       the four groups of tests every operator needs
```

---

## 1. Install it

```bash
pip install -e templates/operator          # into the environment that runs SquidXplorer
python -c "import squidxplorer; print(squidxplorer.available_plane_operators())"
# ['bgsub', 'cellpose', 'decon', 'decon3d', 'flatfield', 'mip', 'reference', 'spot', 'stdev']
#                                                                                    ^^^^^^^
# it runs, headless, over a whole plate:
squidxplorer /path/to/acquisition --operator stdev --output-folder /tmp/out

pip uninstall squidxplorer-operator-template                        # and it leaves cleanly
```

Or from Python, which is what the template's own tests do:

```python
import squidxplorer
reader = squidxplorer.open_reader("/path/to/acquisition")
for region, fov, image in squidxplorer.run_plate(reader, operator="stdev",
                                             operator_kwargs={"smooth_sigma": 0.0}):
    ...                                    # (T, C, 1, Y, X), the acquisition's native dtype
squidxplorer.write_plate(reader, "/tmp/out.hcs", operator="stdev")   # navigable OME-Zarr plate
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
from squidxplorer import plane_op, add_operator
add_operator("mything", plane_op(my_plane_function))      # consumes inferred = frozenset()
```

**`consumes={"fov"}` is refused by `add_operator`.** An `add_operator` operator is
`Iterable[plane] -> plane` and never sees a tile's stage geometry, so anything inter-FOV (stitching,
fitting an illumination field across a well) cannot be expressed with that callable. Register it
with `squidxplorer.add_region_operator` instead, whose entries take
`(reader, region, fovs) -> (T, C, Nz, Y, X)`. It is the **same registry table** and the same four
declarations — `add_region_operator` stamps `consumes={"fov"}` on the record, and `run_plate`
finds your operator by reading it.

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

Declaring `params=` changes what `add_operator` does with your callable: **it is read as a
factory.** SquidXplorer calls it once with your declared defaults to build the entry's default
binding, and calls it again with a caller's `operator_kwargs` for a run that names different
values.

```python
add_operator("stdev", stdev_op, params=(Param("smooth_sigma", 1.0, "..."),))

squidxplorer.bind_operator("stdev", {"smooth_sigma": 0.0})            # a different binding
squidxplorer.run_plate(reader, operator="stdev",
                   operator_kwargs={"smooth_sigma": 0.0})          # a run at that binding
```

So ONE entry covers every setting. Do not register `stdev_tight` and `stdev_loose`.

A parameter name you did not declare is **refused by name**, listing what you do accept. Accepting
and dropping it would run at defaults while the console line and the recipe both said otherwise —
a wrong result that looks right.

A `Param` has a name, a default and one line of prose. Deliberately **no type, no range, no widget
hint**: the moment it carries a widget hint it has become the GUI's schema and two places own the
same fact.

**How parameters reach the GUI.** `squidxplorer.operator_params("<name>")` is the public accessor,
and `list_operators` reports every parameter with its default, so an agent or script driving the
app can set them. The desktop GUI reads the same accessor: **your declared parameters become
widgets, with no code in SquidXplorer that knows your operator's name.**
`squidxplorer/_param_panel.py` builds the panel, and the widget is chosen from the TYPE OF YOUR
DEFAULT — which is why `Param` needs no widget hint:

| your default | the widget you get |
|---|---|
| `bool` | a check box |
| `int` | an integer spin |
| `float` | a decimal spin |
| `str` | a text field |

Your `blurb` becomes the widget's tooltip. A default of any other type (`None`, a tuple, an array)
is **refused by name**: the panel says which parameter and which type rather than guessing a
widget, because a guessed widget is how a value the user typed becomes a value the run did not
receive. Give such a parameter a default of a type in the table above, or keep it CLI-only.

The panel is the FALLBACK: an operator SquidXplorer ships a hand-written panel for (`stitch`,
`decon`) keeps it, because those do things a parameter form cannot. Yours will not have one, so
yours gets the generic panel. It reaches the plate exactly the way a hand-written panel's values
do — through `operator_kwargs` — and your operator appears under **Process well-plates -> From
their declaration** without an edit anywhere in SquidXplorer.

Before 2026-08-05 this paragraph documented the opposite, and it was the weakest link in this
whole contract: this README told you to declare `params` into a GUI that ignored them, so
`spot` and `cellpose` declared four parameters each and not one was settable from any panel.

**What a `Param` cannot be: a channel.** The engine calls you per `(t, c, z-group)` and hands you
planes only, so a value bound at registration never learns which channel it is looking at. If your
operator must be specialised per channel (a PSF at the right emission wavelength, say), stamp a
`for_channel(acquisition_path, channel) -> operator` attribute on your callable; the engine calls
it once per channel and runs what it returns. See `squidxplorer._decon.optics_for_channel`.

### 2.5 `requires` — how you declare dependencies, and what happens when they are missing

```python
add_operator("stdev", stdev_op, params=(...), requires=("scipy",))
```

Importable **module** names — what you `import`, not what pip installs, when the two differ
(`scikit-image` installs the module `skimage`; declare `skimage`).

What happens when one is missing:

| | |
|---|---|
| **Listed?** | **Yes, always.** `available_plane_operators()` still contains your operator. Dropping it would make "scipy is not installed" and "nobody wrote this operator" look identical to the user. |
| **`operator_available("stdev")`** | `(False, "operator 'stdev' needs scipy, which is not installed (pip install scipy)")` |
| **`list_operators`** | the row carries `"available": false` and `"unavailable_reason"`, and the summary line names every unavailable operator |
| **A run** | **refused by name, before any well is read.** `bind_operator` raises `MissingOperatorDependency` with that sentence. The headless command surface returns the refusal code `unavailable_operator`; the GUI puts the sentence in the readout and never starts a worker. |

**Declare a module even if you import it lazily, several calls deep.** This is the direction that
bites, and it is measured rather than hypothetical: three of SquidXplorer's own operators (`decon`,
`decon3d`, `flatfield`) imported packages that were not in its dependency list at all. On a stock
install they were advertised, they raised `ImportError` one call deep, per-well fault isolation
(`run_plate(on_error=...)`) recorded that as one skip per well — and a whole-plate run finished
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
  parametrised over `available_plane_operators()`, so tests written before your operator existed run
  against it the moment it is installed.
* Its results are written by `write_plate` as a navigable OME-Zarr plate — z-reducers at `Nz=1`,
  plane-ops at the acquisition's full depth.

---

## 2.6 Composition happens in Python

There is no chain syntax: to combine steps, wrap them in one callable and register the result as
its own operator (`plane_op` around a function that applies your steps in order, then
`add_operator`) — a few lines, declared like any other entry.

---

## 3. What this template deliberately does NOT support

Stated so you do not build against something that is not there.

* **A HAND-WRITTEN GUI panel.** You get the generic one built from your `params` (§2.4), which is
  a form. A panel with behaviour of its own — unit conversions, controls that grey each other out,
  an iterative QC loop that publishes a picture — lives in SquidXplorer and a plugin cannot add one.
* **Inter-FOV work through `add_operator`.** See §2.2 — `add_region_operator` instead. It is the
  SAME record and the same four declarations as of 2026-08-05 (it used to be a second table with
  only `requires=`, and `produces` hardcoded to `"intensity"` by every reader of it). What still
  differs is the callable shape, and it is not what this template's example code is written
  against. `stitch`'s controls stay hand-written (`_op_panels.StitcherPanel`) because it converts
  units and greys out knobs — not because there is no declaration to read.
* **Registering into the viewer's card table.** The card list (label, blurb, ordering) lives in
  SquidXplorer and a plugin cannot add to it. Your operator is *runnable* and *listed* without a
  card; `operator_label()` falls back to the registry name. A card is presentation, the engine is
  capability, and the two are asked separately.

---

## 4. Failure modes, and what each one looks like

| what you did | what happens |
|---|---|
| `from squidxplorer import ...` at **module scope** in your plugin | `OperatorPluginError: ... AttributeError: partially initialized module ... (most likely due to a circular import)`. SquidXplorer loads you from inside `import squidxplorer`, so importing it back at module scope is re-entrant. **Import squidxplorer inside `register()`** — the template does, with the reason written next to it. This is the first mistake everyone makes. |
| Your operator's name is already taken | `add_operator` raises at import; `import squidxplorer` fails naming your plugin, its distribution, and the collision |
| Your module raises on import | `OperatorPluginError` out of `import squidxplorer`, naming your plugin and the original error |
| Your `register()` raises | same, with "raised while registering its operators" |
| Your entry point points at nothing | same, at `ep.load()` |
| A `requires=` module is missing | operator LISTED, run REFUSED by name — see §2.5 |
| You declared `consumes={"fov"}` on `add_operator` | refused at registration, pointing at `add_region_operator` |
| You declared `produces="mesh"` | refused at registration, listing the kinds that exist |
| You declared a parameter twice | refused at registration — a duplicate makes `operator_kwargs` ambiguous |
| Your factory returned something not callable | refused at registration, explaining that `params=` makes the registered object a factory |

A broken plugin **stops `import squidxplorer`**. That is deliberate: the alternative is an application
that silently does not have the operator you installed, which is the same defect `requires=` exists
to end, one layer up. The escape hatch, for a user whose app will not start, is
`SQUIDXPLORER_NO_PLUGINS=1`, and it is named in every one of those error messages.

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
