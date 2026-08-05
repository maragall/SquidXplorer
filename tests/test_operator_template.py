"""The STANDARDISED OPERATOR TEMPLATE: the declared-dependency seam and the discovery seam.

Julio's Friday deliverable, phase 3: *"Standardized operator layer: public template defining what
the viewer expects before/after processing"*, so that new operators are cheap to add and
*"community contributors can adapt their own repos to work with Squid Explorer without custom
integration."*

Two mechanisms make that possible, and this file pins both.

1. ``requires=`` ON THE OPERATOR RECORD (§1-§3). ``_spots.Segmenter`` has declared its optional
   packages since Cellpose landed; ``Operator`` did not. The consequence was measured on a stock
   ``pip install .[gui]``: ``decon``, ``decon3d`` and ``flatfield`` import packages that are not in
   this project's dependency list at all, so they were advertised by ``available_projectors()``,
   raised ImportError from a lazy import one call deep, and ``project_plate(on_error=...)`` filed
   that as a per-well skip — a whole-plate run that finished GREEN having written nothing. These
   tests pin the refusal that replaces it, and pin that per-well fault isolation can no longer
   absorb it.

2. ENTRY-POINT DISCOVERY (§4-§5). ``squidmip/__init__.py`` ends in a hardcoded side-effect import
   list, and that list was the only thing that made a registration run — so an operator in a
   package we do not ship appeared nowhere. The ``squidmip.operators`` entry-point group is the
   seam that fixes it, and it must fail LOUD and NAMED on a broken plugin, never skip it silently:
   a plugin that quietly does not load is indistinguishable from an operator nobody wrote, which
   is the same defect ``requires=`` exists to end, one layer up.

§6 pins the template package itself, because a template that does not install is not a template.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import squidmip as s
from squidmip._engine import MissingOperatorDependency, Operator
from squidmip._plugins import GROUP, DISABLE_ENV, OperatorPluginError, load_operator_plugins
from squidmip.projection import MissingDependency

_REPO = Path(__file__).resolve().parents[1]
_TEMPLATE = _REPO / "templates" / "operator"

#: A module name that cannot possibly be importable. Spelled once.
ABSENT = "a_package_that_is_definitely_not_installed_xyzzy"


def _passthrough(planes):
    """A trivial z-reducer, for tests that care about the DECLARATION rather than the pixels."""
    return np.maximum.reduce([np.asarray(p) for p in planes])


# ==============================================================================================
# 1. THE DECLARATION
# ==============================================================================================

def test_operator_carries_requires_with_the_same_spelling_as_segmenter():
    """One word, three registries. A second spelling (``needs=``, ``deps=``) would make the
    template teach two contracts for one idea."""
    from squidmip._spots import Segmenter

    assert "requires" in Operator.__dataclass_fields__
    assert "requires" in Segmenter.__dataclass_fields__
    assert Operator.__dataclass_fields__["requires"].default == ()


def test_every_registrar_takes_requires_by_the_same_keyword():
    """``add_projector``, ``add_region_operator`` and ``add_segmenter`` — the three seams a
    contributor can plug into. Checked by signature so a fourth registrar cannot quietly ship
    without it."""
    import inspect

    from squidmip._spots import add_segmenter
    from squidmip._stitch import add_region_operator

    for fn in (s.add_projector, add_region_operator, add_segmenter):
        assert "requires" in inspect.signature(fn).parameters, fn.__name__


def test_a_bare_string_requires_is_read_as_one_module_not_eight_letters():
    """The tuple-comma trap. ``requires="cellpose"`` must mean one module."""
    s.add_projector("tpl_bare_string", _passthrough, requires="cellpose")

    assert s.projector_requires("tpl_bare_string") == ("cellpose",)


def test_a_requirement_that_is_not_a_module_name_is_refused_at_registration():
    with pytest.raises(ValueError, match="MODULE names"):
        s.add_projector("tpl_bad_requires", _passthrough, requires=(None,))


def test_declaring_nothing_keeps_the_pre_existing_contract_exactly():
    """Every registration written before this feature keeps its exact meaning."""
    s.add_projector("tpl_no_requires", _passthrough)

    assert s.projector_requires("tpl_no_requires") == ()
    assert s.operator_available("tpl_no_requires") == (True, "")


# ==============================================================================================
# 2. WHAT AN UNAVAILABLE OPERATOR DOES — LISTED, AND REFUSED BY NAME
# ==============================================================================================

@pytest.fixture
def small_acquisition(squid_dataset):
    """The tiny synthetic acquisition, as a path. Two regions, so a "skipped every well" bug is
    visible as an empty result rather than as one missing field."""
    root, _arrays = squid_dataset
    return root


@pytest.fixture
def small_reader(small_acquisition):
    return s.open_reader(str(small_acquisition))


@pytest.fixture
def unavailable():
    """A registered operator whose declared package does not exist."""
    s.add_projector("tpl_unavailable", _passthrough, requires=(ABSENT,))
    return "tpl_unavailable"


def test_an_unavailable_operator_is_still_listed(unavailable):
    """ABSENT IS NOT UNWRITTEN. Filtering it out of the list would make "the package is missing"
    and "nobody wrote this operator" identical to every caller — the exact rule
    ``available_segmenters`` has always applied, now applied to the operator table."""
    assert unavailable in s.available_projectors()


def test_availability_is_reported_with_a_reason_and_the_install_command(unavailable):
    ok, why = s.operator_available(unavailable)

    assert not ok
    assert unavailable in why
    assert ABSENT in why
    assert f"pip install {ABSENT}" in why


def test_binding_refuses_by_name_before_any_work(unavailable):
    with pytest.raises(MissingOperatorDependency, match=ABSENT):
        s.bind_projector(unavailable)


def test_the_refusal_is_a_missing_dependency_so_a_runner_can_tell_it_from_a_data_fault(unavailable):
    """One base class across the three registries: ``projection.MissingDependency``. That is what
    lets ``project_plate`` say "environment fault, not a corrupt well" without importing each
    registry to name its own exception."""
    from squidmip._spots import MissingSegmenterDependency

    assert issubclass(MissingOperatorDependency, MissingDependency)
    assert issubclass(MissingSegmenterDependency, MissingDependency)


def test_reading_an_unavailable_operators_declaration_still_works(unavailable):
    """A UI must be able to describe a row it is greying out. Only RUNNING is refused."""
    assert s.projector_consumes(unavailable) == frozenset({"z"})
    assert s.projector_produces(unavailable) == "intensity"
    assert s.projector_params(unavailable) == ()


def test_an_unknown_name_and_an_unavailable_one_are_different_answers():
    """An agent branches on these differently: one means "pick another name", the other means
    "install the package". Collapsing them makes both un-actionable."""
    ok_unknown, why_unknown = s.operator_available("no_such_operator_at_all")

    assert not ok_unknown
    assert "unknown projector" in why_unknown


# ==============================================================================================
# 3. THE SILENT SUCCESS THIS CLOSES
# ==============================================================================================

def test_project_plate_refuses_before_reading_a_well_and_on_error_cannot_swallow_it(
        small_reader, unavailable):
    """THE DEFECT, pinned. Previously: the operator raised ImportError for every well, ``on_error``
    recorded each as a skip, the stream ended normally and the run reported success with nothing
    produced. Now the refusal happens at bind time, which is BEFORE the per-well loop exists."""
    skipped = []

    with pytest.raises(MissingOperatorDependency):
        list(s.project_plate(small_reader, n_fovs=1, projector=unavailable,
                             on_error=lambda region, fov, exc: skipped.append((region, fov))))

    assert skipped == [], "a missing package was filed as a per-well skip"


def test_an_undeclared_lazy_import_error_is_also_not_a_per_well_skip(small_reader):
    """The backstop for an operator whose author forgot to declare. An ImportError raised from
    inside the operator will raise identically for EVERY well, so isolating it skips all of them
    and the run finishes green — which is not fault isolation, it is a hidden total failure."""
    def _needs_the_absent(planes):
        __import__(ABSENT)
        return next(iter(planes))

    s.add_projector("tpl_undeclared", _needs_the_absent)
    skipped = []

    with pytest.raises(ImportError):
        list(s.project_plate(small_reader, n_fovs=1, projector="tpl_undeclared",
                             on_error=lambda region, fov, exc: skipped.append((region, fov))))

    assert skipped == []


def test_a_genuine_per_well_data_fault_is_still_isolated(small_reader):
    """The contract ``on_error`` was built for is UNCHANGED. One corrupt well must not abort a
    plate; this is the line between the two behaviours."""
    def _explodes(planes):
        raise ValueError("this well's pixels are corrupt")

    s.add_projector("tpl_data_fault", _explodes)
    skipped = []

    produced = list(s.project_plate(small_reader, n_fovs=1, projector="tpl_data_fault",
                                    on_error=lambda region, fov, exc: skipped.append(region)))

    assert produced == []
    assert skipped, "a per-well data fault must still be isolated, not raised"


def test_the_command_surface_refuses_with_its_own_code(small_acquisition, unavailable):
    """``unavailable_operator`` is a DISTINCT refusal code from ``unknown_operator``, because the
    caller's next move differs: pick another name, versus install a package."""
    from squidmip._command import (UNAVAILABLE_OPERATOR, CommandBus, EngineExecutor,
                                   OpenAcquisition, RunOperator)

    bus = CommandBus(EngineExecutor())
    bus.execute(OpenAcquisition(path=str(small_acquisition)))
    result = bus.execute(RunOperator(operator=unavailable, regions=["B2"]))

    assert not result.ok
    assert result.refusal == UNAVAILABLE_OPERATOR
    assert ABSENT in result.message


def test_list_operators_reports_availability_without_filtering_the_list(unavailable):
    from squidmip._command import CommandBus, EngineExecutor, ListOperators

    result = CommandBus(EngineExecutor()).execute(ListOperators())
    rows = {row["name"]: row for row in result.data["operators"]}

    assert unavailable in rows, "an unavailable operator was dropped from the answer"
    assert rows[unavailable]["available"] is False
    assert ABSENT in rows[unavailable]["unavailable_reason"]
    assert rows[unavailable]["requires"] == [ABSENT]
    assert rows["mip"]["available"] is True
    assert unavailable in result.data["unavailable"]


def test_the_built_in_operators_with_heavyweight_lazy_imports_declare_them():
    """The three that were measured advertising themselves and producing nothing. Pinned by name
    HERE, in a test, rather than left to be rediscovered on the next clean install."""
    assert s.projector_requires("decon") == ("petakit",)
    assert s.projector_requires("decon3d") == ("petakit",)
    assert s.projector_requires("flatfield") == ("tilefusion",)
    assert s.projector_requires("cellpose") == ("cellpose",)


def test_every_region_operator_declares_its_requirements():
    """The region table keeps ``requires`` in a SIDECAR dict (``_REGION_REQUIRES``) rather than on
    a record, because ``_REGION_OPERATORS`` maps a name straight to a callable and promoting it to
    an ``Operator``-style record would touch every consumer of ``_resolve_region_operator``. A
    sidecar is only safe while the two are written together — this is the guard that says so, and
    the reason the table wants unifying."""
    from squidmip._stitch import _REGION_OPERATORS, _REGION_REQUIRES

    assert set(_REGION_REQUIRES) == set(_REGION_OPERATORS)


def test_region_operators_declare_and_refuse_the_same_way(small_reader):
    """The parallel table gets the same word and the same behaviour, so a contributor learns the
    contract once."""
    from squidmip._stitch import add_region_operator, region_operator_available

    add_region_operator("tpl_region_unavailable", lambda reader, region, fovs, **kw: None,
                        requires=(ABSENT,))

    assert "tpl_region_unavailable" in s.available_region_operators()
    ok, why = region_operator_available("tpl_region_unavailable")
    assert not ok and ABSENT in why
    with pytest.raises(MissingOperatorDependency, match=ABSENT):
        list(s.stitch_plate(small_reader, operator="tpl_region_unavailable"))


# ==============================================================================================
# 4. DISCOVERY: AN OPERATOR THAT LIVES IN ANOTHER PACKAGE
# ==============================================================================================

class _FakeEntryPoint:
    """Enough of an ``importlib.metadata.EntryPoint`` for the loader. ``load()`` is the seam."""

    dist = None

    def __init__(self, name, value, loader):
        self.name = name
        self.value = value
        self._loader = loader

    def load(self):
        return self._loader()


def _with_entry_points(monkeypatch, entry_points):
    """Patch the METADATA source, not our own helper — so the loader's sorting, its group filter
    and its error handling are all still under test."""
    monkeypatch.setattr("importlib.metadata.entry_points",
                        lambda group=None: list(entry_points))


def test_the_group_name_is_the_documented_one():
    """The template's ``pyproject.toml`` writes this string. If it changes here and not there,
    every plugin ever published becomes invisible — so it is pinned in both places."""
    assert GROUP == "squidmip.operators"
    assert '[project.entry-points."squidmip.operators"]' in (_TEMPLATE / "pyproject.toml").read_text()


def test_a_plugin_registers_its_operator_into_the_same_table_as_the_built_ins(monkeypatch):
    def _register():
        s.add_projector("tpl_from_a_plugin", _passthrough, produces="intensity")

    _with_entry_points(monkeypatch, [_FakeEntryPoint("demo", "demo:register", lambda: _register)])

    assert load_operator_plugins() == ["demo"]
    assert "tpl_from_a_plugin" in s.available_projectors()


def test_a_plugin_that_fails_to_import_aborts_by_name_never_silently(monkeypatch):
    """LOUD AND NAMED. A skipped plugin is an application that silently does not have the operator
    the user installed — the same defect as an operator that runs and produces nothing."""
    def _explode():
        raise ModuleNotFoundError("No module named 'torch'")

    _with_entry_points(monkeypatch,
                       [_FakeEntryPoint("brokenplug", "brokenplug:register", _explode)])

    with pytest.raises(OperatorPluginError) as exc:
        load_operator_plugins()

    assert "brokenplug" in str(exc.value)
    assert "brokenplug:register" in str(exc.value)
    assert "torch" in str(exc.value)
    assert DISABLE_ENV in str(exc.value), "the message must carry the escape hatch"
    assert isinstance(exc.value.__cause__, ModuleNotFoundError)


def test_a_plugin_whose_register_raises_aborts_by_name(monkeypatch):
    def _register():
        raise RuntimeError("my registration is wrong")

    _with_entry_points(monkeypatch,
                       [_FakeEntryPoint("badreg", "badreg:register", lambda: _register)])

    with pytest.raises(OperatorPluginError, match="badreg"):
        load_operator_plugins()


def test_a_plugin_that_collides_with_a_built_in_name_is_refused_not_allowed_to_clobber(monkeypatch):
    """Discovery runs AFTER the built-ins precisely so this is the outcome. A plugin that replaced
    ``mip`` would change what every existing recipe means."""
    def _register():
        s.add_projector("mip", _passthrough)

    _with_entry_points(monkeypatch,
                       [_FakeEntryPoint("collider", "collider:register", lambda: _register)])

    with pytest.raises(OperatorPluginError, match="already"):
        load_operator_plugins()
    assert s.bind_projector("mip") is not _passthrough


def test_a_module_entry_point_registers_by_import_side_effect(monkeypatch):
    """``my_package`` (no ``:callable``) is supported too — it is how this package's own operator
    modules register. Not callable, so nothing is called."""
    import types

    module = types.ModuleType("tpl_side_effect_module")

    def _load():
        s.add_projector("tpl_side_effect", _passthrough)
        return module

    _with_entry_points(monkeypatch, [_FakeEntryPoint("sideeffect", "mod", _load)])

    load_operator_plugins()

    assert "tpl_side_effect" in s.available_projectors()


def test_plugins_load_in_a_deterministic_order(monkeypatch):
    """Two plugins can collide on an operator name. A collision whose winner depends on filesystem
    iteration order is reproducible on one machine and not another."""
    seen = []
    eps = [_FakeEntryPoint(n, f"{n}:register", lambda n=n: (lambda: seen.append(n)))
           for n in ("zulu", "alpha", "mike")]
    _with_entry_points(monkeypatch, eps)

    load_operator_plugins()

    assert seen == ["alpha", "mike", "zulu"]


def test_the_escape_hatch_skips_discovery_entirely(monkeypatch):
    """For a user whose app will not start because of somebody else's plugin."""
    _with_entry_points(monkeypatch, [_FakeEntryPoint(
        "boom", "boom:register", lambda: (_ for _ in ()).throw(RuntimeError("boom")))])
    monkeypatch.setenv(DISABLE_ENV, "1")

    assert load_operator_plugins() == []


def test_discovery_is_additive_the_built_ins_do_not_go_through_it():
    """The hardcoded imports keep working. Routing ``mip`` through installed metadata would make
    the shipped operators depend on that metadata being intact, for no benefit."""
    declared = {name for name, _target, _dist in s.declared_operator_plugins()}

    for built_in in ("mip", "reference", "decon", "bgsub", "flatfield", "spot", "cellpose"):
        assert built_in in s.available_projectors()
        assert built_in not in declared


# ==============================================================================================
# 5. THE TEMPLATE PACKAGE ITSELF
# ==============================================================================================

def test_the_template_ships_every_file_a_contributor_needs():
    for relative in ("pyproject.toml", "README.md",
                     "squidmip_operator_template/__init__.py",
                     "squidmip_operator_template/_stdev.py",
                     "tests/test_template_operator.py"):
        assert (_TEMPLATE / relative).exists(), f"templates/operator/{relative} is missing"


def test_the_templates_operator_declares_all_four_things():
    """The template must demonstrate the WHOLE record, or it teaches half a contract."""
    source = (_TEMPLATE / "squidmip_operator_template" / "__init__.py").read_text()

    for declaration in ("consumes=", "produces=", "params=", "requires="):
        assert declaration in source, f"the template's registration omits {declaration}"


def test_the_template_does_not_import_squidmip_at_module_scope():
    """The circular-import trap, pinned. SquidXplorer loads a plugin from INSIDE ``import
    squidmip``, so a module-scope ``from squidmip import ...`` in the plugin is re-entrant and
    fails with a partially-initialised module. The template imports inside ``register()`` and says
    why; this test stops that from being edited back."""
    import ast

    tree = ast.parse((_TEMPLATE / "squidmip_operator_template" / "__init__.py").read_text())
    for node in tree.body:                    # module scope only, deliberately
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = getattr(node, "module", None) or node.names[0].name
            assert not name.startswith("squidmip."), name
            assert name != "squidmip", (
                "the template imports squidmip at module scope; that is re-entrant and is the "
                "first mistake every plugin author makes")


def test_the_template_readme_states_the_contract_the_viewer_depends_on():
    """The deliverable is *a public template defining what the viewer expects before/after
    processing*. These are the facts a contributor cannot guess."""
    readme = (_TEMPLATE / "README.md").read_text()

    for fact in (
        "Iterable[np.ndarray]) -> np.ndarray",   # the callable shape
        "(T, C, 1, Y, X)",                       # a z-reducer's output shape
        "(T, C, Nz, Y, X)",                      # a plane-op's output shape
        'frozenset({"z"})',                      # the z-reducer declaration
        "napari **Image** layer",                # what produces=intensity renders as
        "napari **Labels** layer",               # what produces=labels renders as
        "Param(name: str, default: Any",         # how a parameter is declared
        "requires=",                             # how a dependency is declared
        "unavailable_operator",                  # what happens when it is missing
        "SQUIDMIP_NO_PLUGINS=1",                 # the escape hatch
    ):
        assert fact in readme, f"templates/operator/README.md does not state: {fact}"


def test_the_template_names_what_it_does_not_support():
    """Composition does NOT exist — ``_recipe.RecipeChain`` documents it and nothing executes it.
    A template that implies otherwise sends a contributor to build against a hole."""
    readme = (_TEMPLATE / "README.md").read_text()

    assert "does NOT support" in readme
    assert "nothing executes it" in readme


def test_the_template_package_imports_and_registers_in_a_clean_interpreter():
    """The template's own ``register()`` runs, in a subprocess so this suite's registry is
    untouched. Proves the file is not merely well-formed prose: the four declarations reach the
    engine and describe a real operator."""
    script = (
        "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
        "import squidmip, squidmip_operator_template as t\n"
        "t.register()\n"
        "assert t.OPERATOR_NAME in squidmip.available_projectors()\n"
        "assert squidmip.projector_consumes(t.OPERATOR_NAME) == frozenset({'z'})\n"
        "assert squidmip.projector_produces(t.OPERATOR_NAME) == 'intensity'\n"
        "assert squidmip.projector_requires(t.OPERATOR_NAME) == ('scipy',)\n"
        "assert {p.name for p in squidmip.projector_params(t.OPERATOR_NAME)} == "
        "{'smooth_sigma', 'ddof'}\n"
        "import numpy as np\n"
        "out = squidmip.bind_projector(t.OPERATOR_NAME, {'smooth_sigma': 0.0})("
        "[np.full((4, 4), 10, np.uint16), np.full((4, 4), 20, np.uint16)])\n"
        "assert out.shape == (4, 4) and out.dtype == np.uint16 and out.max() == 5, out\n"
        "print('OK')\n"
    ) % (str(_REPO), str(_TEMPLATE))

    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", DISABLE_ENV: "1",
                               "HOME": str(Path.home())})

    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
