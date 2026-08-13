"""The standardised operator template: the declared-dependency seam and the discovery seam."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import squidxplorer as s
from squidxplorer._engine import MissingOperatorDependency, Operator
from squidxplorer._plugins import GROUP, DISABLE_ENV, OperatorPluginError, load_operator_plugins
from squidxplorer.projection import MissingDependency

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

def test_operator_carries_requires_with_an_empty_default():
    assert "requires" in Operator.__dataclass_fields__
    assert Operator.__dataclass_fields__["requires"].default == ()


def test_every_registrar_takes_requires_by_the_same_keyword():
    """Checked by signature so a new registrar cannot quietly ship without it."""
    import inspect

    from squidxplorer._spots import add_segmentation_operator
    from squidxplorer._stitch import add_region_operator

    for fn in (s.add_operator, add_region_operator, add_segmentation_operator):
        assert "requires" in inspect.signature(fn).parameters, fn.__name__


def test_a_bare_string_requires_is_read_as_one_module_not_eight_letters():
    """The tuple-comma trap. ``requires="cellpose"`` must mean one module."""
    s.add_operator("tpl_bare_string", _passthrough, requires="cellpose")

    assert s.operator_requires("tpl_bare_string") == ("cellpose",)


def test_a_requirement_that_is_not_a_module_name_is_refused_at_registration():
    with pytest.raises(ValueError, match="MODULE names"):
        s.add_operator("tpl_bad_requires", _passthrough, requires=(None,))


def test_declaring_nothing_keeps_the_pre_existing_contract_exactly():
    """Every registration written before this feature keeps its exact meaning."""
    s.add_operator("tpl_no_requires", _passthrough)

    assert s.operator_requires("tpl_no_requires") == ()
    assert s.operator_available("tpl_no_requires") == (True, "")


# ==============================================================================================
# 2. WHAT AN UNAVAILABLE OPERATOR DOES — LISTED, AND REFUSED BY NAME
# ==============================================================================================

@pytest.fixture
def small_acquisition(squid_dataset):
    """The tiny synthetic acquisition, as a path."""
    root, _arrays = squid_dataset
    return root


@pytest.fixture
def small_reader(small_acquisition):
    return s.open_reader(str(small_acquisition))


@pytest.fixture
def unavailable():
    """A registered operator whose declared package does not exist."""
    s.add_operator("tpl_unavailable", _passthrough, requires=(ABSENT,))
    return "tpl_unavailable"


def test_an_unavailable_operator_is_still_listed(unavailable):
    """Filtering it out would make "package missing" and "nobody wrote it" identical."""
    assert unavailable in s.available_plane_operators()


def test_availability_is_reported_with_a_reason_and_the_install_command(unavailable):
    ok, why = s.operator_available(unavailable)

    assert not ok
    assert unavailable in why
    assert ABSENT in why
    assert f"pip install {ABSENT}" in why


def test_binding_refuses_by_name_before_any_work(unavailable):
    with pytest.raises(MissingOperatorDependency, match=ABSENT):
        s.bind_operator(unavailable)


def test_the_refusal_is_a_missing_dependency_so_a_runner_can_tell_it_from_a_data_fault(unavailable):
    assert issubclass(MissingOperatorDependency, MissingDependency)


def test_reading_an_unavailable_operators_declaration_still_works(unavailable):
    """A UI must be able to describe a row it is greying out. Only RUNNING is refused."""
    assert s.operator_consumes(unavailable) == frozenset({"z"})
    assert s.operator_produces(unavailable) == "intensity"
    assert s.operator_params(unavailable) == ()


def test_an_unknown_name_and_an_unavailable_one_are_different_answers():
    """One means "pick another name", the other "install the package"."""
    ok_unknown, why_unknown = s.operator_available("no_such_operator_at_all")

    assert not ok_unknown
    assert "unknown operator" in why_unknown


# ==============================================================================================
# 3. THE SILENT SUCCESS THIS CLOSES
# ==============================================================================================

def test_run_plate_refuses_before_reading_a_well_and_on_error_cannot_swallow_it(
        small_reader, unavailable):
    """The refusal happens at bind time, before the per-well loop exists."""
    skipped = []

    with pytest.raises(MissingOperatorDependency):
        list(s.run_plate(small_reader, n_fovs=1, operator=unavailable,
                             on_error=lambda region, fov, exc: skipped.append((region, fov))))

    assert skipped == [], "a missing package was filed as a per-well skip"


def test_an_undeclared_lazy_import_error_is_also_not_a_per_well_skip(small_reader):
    """The backstop for an operator whose author forgot to declare."""
    def _needs_the_absent(planes):
        __import__(ABSENT)
        return next(iter(planes))

    s.add_operator("tpl_undeclared", _needs_the_absent)
    skipped = []

    with pytest.raises(ImportError):
        list(s.run_plate(small_reader, n_fovs=1, operator="tpl_undeclared",
                             on_error=lambda region, fov, exc: skipped.append((region, fov))))

    assert skipped == []


def test_a_genuine_per_well_data_fault_is_still_isolated(small_reader):
    """One corrupt well must not abort a plate; this is the line between the two behaviours."""
    def _explodes(planes):
        raise ValueError("this well's pixels are corrupt")

    s.add_operator("tpl_data_fault", _explodes)
    skipped = []

    produced = list(s.run_plate(small_reader, n_fovs=1, operator="tpl_data_fault",
                                    on_error=lambda region, fov, exc: skipped.append(region)))

    assert produced == []
    assert skipped, "a per-well data fault must still be isolated, not raised"


def test_the_command_surface_refuses_with_its_own_code(small_acquisition, unavailable):
    """``unavailable_operator`` is a distinct refusal code from ``unknown_operator``."""
    from squidxplorer._command import (UNAVAILABLE_OPERATOR, CommandBus, EngineExecutor,
                                   OpenAcquisition, RunOperator)

    bus = CommandBus(EngineExecutor())
    bus.execute(OpenAcquisition(path=str(small_acquisition)))
    result = bus.execute(RunOperator(operator=unavailable, regions=["B2"]))

    assert not result.ok
    assert result.refusal == UNAVAILABLE_OPERATOR
    assert ABSENT in result.message


def test_list_operators_reports_availability_without_filtering_the_list(unavailable):
    from squidxplorer._command import CommandBus, EngineExecutor, ListOperators

    result = CommandBus(EngineExecutor()).execute(ListOperators())
    rows = {row["name"]: row for row in result.data["operators"]}

    assert unavailable in rows, "an unavailable operator was dropped from the answer"
    assert rows[unavailable]["available"] is False
    assert ABSENT in rows[unavailable]["unavailable_reason"]
    assert rows[unavailable]["requires"] == [ABSENT]
    assert rows["mip"]["available"] is True
    assert unavailable in result.data["unavailable"]


def test_the_built_in_operators_with_heavyweight_lazy_imports_declare_them():
    """Pinned by name so it is not rediscovered on the next clean install."""
    assert s.operator_requires("decon") == ("petakit",)
    assert s.operator_requires("decon3d") == ("petakit",)
    assert s.operator_requires("flatfield") == ("tilefusion",)
    assert s.operator_requires("cellpose") == ("cellpose",)


def test_every_region_operator_declares_its_requirements():
    """The declaration is readable through the same function every other operator's is."""
    names = s.available_region_operators()
    assert names, "no region operator is registered at all"
    for name in names:
        assert isinstance(s.operator_requires(name), tuple)
    # `stitch` reaches tilefusion one call deep and must say so, by name.
    assert "tilefusion" in s.operator_requires("stitch"), s.operator_requires("stitch")


def test_region_operators_declare_and_refuse_the_same_way(small_reader):
    """The same word, the same registrar family and the same behaviour for region operators."""
    s.add_region_operator("tpl_region_unavailable", lambda reader, region, fovs, **kw: None,
                          requires=(ABSENT,))

    assert "tpl_region_unavailable" in s.available_region_operators()
    assert "tpl_region_unavailable" in s.runnable_operators()
    ok, why = s.operator_available("tpl_region_unavailable")
    assert not ok and ABSENT in why
    with pytest.raises(MissingOperatorDependency, match=ABSENT):
        list(s.run_plate(small_reader, operator="tpl_region_unavailable"))


# ==============================================================================================
# 4. DISCOVERY: AN OPERATOR THAT LIVES IN ANOTHER PACKAGE
# ==============================================================================================

class _FakeEntryPoint:
    """Enough of an ``importlib.metadata.EntryPoint`` for the loader; carries a ``group`` because the loader filters on it."""

    dist = None

    def __init__(self, name, value, loader, group=GROUP):
        self.name = name
        self.value = value
        self.group = group
        self._loader = loader

    def load(self):
        return self._loader()


#: An entry point in somebody else's group; a loader that stopped filtering is caught by name.
_OFF_GROUP = _FakeEntryPoint(
    "not_ours", "somebody_else:register",
    lambda: (_ for _ in ()).throw(AssertionError(
        "the loader loaded an entry point from another package's group")),
    group="some.other.group",
)


def _with_entry_points(monkeypatch, entry_points):
    """Patch the metadata source, not our own helper, and filter on group for real."""
    planted = list(entry_points) + [_OFF_GROUP]

    def _entry_points(group=None):
        return [ep for ep in planted if group is None or ep.group == group]

    monkeypatch.setattr("importlib.metadata.entry_points", _entry_points)


def test_the_group_name_is_the_documented_one():
    """Pinned in both places: the group string and the template's pyproject.toml."""
    assert GROUP == "squidxplorer.operators"
    assert '[project.entry-points."squidxplorer.operators"]' in (_TEMPLATE / "pyproject.toml").read_text()


def test_a_plugin_registers_its_operator_into_the_same_table_as_the_built_ins(monkeypatch):
    def _register():
        s.add_operator("tpl_from_a_plugin", _passthrough, produces="intensity")

    _with_entry_points(monkeypatch, [_FakeEntryPoint("demo", "demo:register", lambda: _register)])

    assert load_operator_plugins() == ["demo"]
    assert "tpl_from_a_plugin" in s.available_plane_operators()


def test_a_plugin_that_fails_to_import_aborts_by_name_never_silently(monkeypatch):
    """A silently skipped plugin is an app that quietly lacks the operator the user installed."""
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
    """A plugin that replaced ``mip`` would change what every existing recipe means."""
    def _register():
        s.add_operator("mip", _passthrough)

    _with_entry_points(monkeypatch,
                       [_FakeEntryPoint("collider", "collider:register", lambda: _register)])

    with pytest.raises(OperatorPluginError, match="already"):
        load_operator_plugins()
    assert s.bind_operator("mip") is not _passthrough


def test_a_module_entry_point_registers_by_import_side_effect(monkeypatch):
    """``my_package`` (no ``:callable``) is supported too; nothing is called."""
    import types

    module = types.ModuleType("tpl_side_effect_module")

    def _load():
        s.add_operator("tpl_side_effect", _passthrough)
        return module

    _with_entry_points(monkeypatch, [_FakeEntryPoint("sideeffect", "mod", _load)])

    load_operator_plugins()

    assert "tpl_side_effect" in s.available_plane_operators()


def test_plugins_load_in_a_deterministic_order(monkeypatch):
    """A collision whose winner depends on filesystem iteration order is not reproducible."""
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
    """The hardcoded imports keep working; discovery is additive."""
    declared = {name for name, _target, _dist in s.declared_operator_plugins()}

    for built_in in ("mip", "reference", "decon", "bgsub", "flatfield", "spot", "cellpose"):
        assert built_in in s.available_plane_operators()
        assert built_in not in declared


# ==============================================================================================
# 5. THE TEMPLATE PACKAGE ITSELF
# ==============================================================================================

def test_the_template_ships_every_file_a_contributor_needs():
    for relative in ("pyproject.toml", "README.md",
                     "squidxplorer_operator_template/__init__.py",
                     "squidxplorer_operator_template/_stdev.py",
                     "tests/test_template_operator.py"):
        assert (_TEMPLATE / relative).exists(), f"templates/operator/{relative} is missing"


def test_the_templates_operator_declares_all_four_things():
    """The template must demonstrate the WHOLE record, or it teaches half a contract."""
    source = (_TEMPLATE / "squidxplorer_operator_template" / "__init__.py").read_text()

    for declaration in ("consumes=", "produces=", "params=", "requires="):
        assert declaration in source, f"the template's registration omits {declaration}"


def test_the_template_does_not_import_squidxplorer_at_module_scope():
    """The circular-import trap: plugins load from inside ``import squidxplorer``."""
    import ast

    tree = ast.parse((_TEMPLATE / "squidxplorer_operator_template" / "__init__.py").read_text())
    for node in tree.body:                    # module scope only, deliberately
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = getattr(node, "module", None) or node.names[0].name
            assert not name.startswith("squidxplorer."), name
            assert name != "squidxplorer", (
                "the template imports squidxplorer at module scope; that is re-entrant and is the "
                "first mistake every plugin author makes")


def test_the_template_readme_states_the_contract_the_viewer_depends_on():
    """These are the facts a contributor cannot guess."""
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
        "SQUIDXPLORER_NO_PLUGINS=1",                 # the escape hatch
    ):
        assert fact in readme, f"templates/operator/README.md does not state: {fact}"


def test_the_template_names_what_it_does_not_support():
    """A template that implies a feature exists sends a contributor to build against a hole."""
    readme = (_TEMPLATE / "README.md").read_text()

    assert "does NOT support" in readme
    assert "A HAND-WRITTEN GUI panel" in readme
    assert "a plugin cannot add one" in readme


def test_the_template_states_how_a_declared_param_becomes_a_widget():
    """The default-type-to-widget mapping rule has to be in the public contract."""
    readme = (_TEMPLATE / "README.md").read_text()

    for fact in (
        "squidxplorer/_param_panel.py",          # where the rule lives
        "TYPE OF YOUR\nDEFAULT",             # what the widget is chosen from
        "a check box",                       # bool
        "an integer spin",                   # int
        "a decimal spin",                    # float
        "a text field",                      # str
        "refused by name",                   # anything else
        "From\ntheir declaration",           # where it shows up in the app
    ):
        assert fact in readme, f"templates/operator/README.md does not state: {fact!r}"


def test_the_template_states_that_composition_happens_in_python():
    """No chain syntax exists; the README says where composing DOES happen."""
    readme = (_TEMPLATE / "README.md").read_text()

    assert "Composition happens in Python" in readme
    assert "no chain syntax" in readme


def test_the_template_package_imports_and_registers_in_a_clean_interpreter():
    """The template's own ``register()`` runs, in a subprocess so this suite's registry is untouched."""
    script = (
        "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
        "import squidxplorer, squidxplorer_operator_template as t\n"
        "t.register()\n"
        "assert t.OPERATOR_NAME in squidxplorer.available_plane_operators()\n"
        "assert squidxplorer.operator_consumes(t.OPERATOR_NAME) == frozenset({'z'})\n"
        "assert squidxplorer.operator_produces(t.OPERATOR_NAME) == 'intensity'\n"
        "assert squidxplorer.operator_requires(t.OPERATOR_NAME) == ('scipy',)\n"
        "assert {p.name for p in squidxplorer.operator_params(t.OPERATOR_NAME)} == "
        "{'smooth_sigma', 'ddof'}\n"
        "import numpy as np\n"
        "out = squidxplorer.bind_operator(t.OPERATOR_NAME, {'smooth_sigma': 0.0})("
        "[np.full((4, 4), 10, np.uint16), np.full((4, 4), 20, np.uint16)])\n"
        "assert out.shape == (4, 4) and out.dtype == np.uint16 and out.max() == 5, out\n"
        "print('OK')\n"
    ) % (str(_REPO), str(_TEMPLATE))

    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", DISABLE_ENV: "1",
                               "HOME": str(Path.home())})

    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
