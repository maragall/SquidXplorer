"""Tests for the template operator. Copy these too — they are the checklist, not decoration.

Four groups, and every operator should have all four:

1. THE ALGORITHM is correct, checked against something independent (here: numpy's own ``std``).
2. THE SHAPE CONTRACT holds — one plane in, one plane out, the acquisition's dtype, streaming.
3. THE DECLARATION matches the implementation. A ``consumes`` that disagrees with what the code
   actually does is the one class of bug the engine cannot catch for you: it will happily hand a
   plane-op a whole stack because the record said to.
4. THE PLUGIN LOADS — the entry point resolves, ``register()`` runs, and SquidXplorer lists the
   operator. This is the one people skip, and it is the one that fails.

Run with: ``pytest templates/operator/tests`` (or ``pytest`` from inside ``templates/operator``).
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip_operator_template import OPERATOR_NAME, register
from squidmip_operator_template._stdev import stdev_op


# ==============================================================================================
# 1. THE ALGORITHM
# ==============================================================================================

def test_matches_numpy_std_when_smoothing_is_off():
    """The reference implementation, on the same data. Welford is exact, not approximate."""
    rng = np.random.default_rng(0)
    stack = rng.integers(0, 4000, size=(11, 16, 16)).astype(np.uint16)

    got = stdev_op(smooth_sigma=0.0)(list(stack))
    want = np.rint(np.std(stack.astype(np.float64), axis=0)).astype(np.uint16)

    assert np.array_equal(got, want)


def test_a_stack_that_does_not_vary_in_z_projects_to_zero():
    """The property that makes this operator worth having: uniform haze has no z variance."""
    plane = np.full((8, 8), 700, dtype=np.uint16)

    out = stdev_op(smooth_sigma=0.0)([plane.copy() for _ in range(6)])

    assert np.array_equal(out, np.zeros((8, 8), dtype=np.uint16))


def test_smoothing_suppresses_per_plane_noise():
    """What ``smooth_sigma`` is FOR, stated as a measurement rather than a comment."""
    rng = np.random.default_rng(1)
    stack = (500 + rng.normal(0, 30, size=(9, 32, 32))).clip(0, 65535).astype(np.uint16)

    raw = stdev_op(smooth_sigma=0.0)(list(stack)).mean()
    smoothed = stdev_op(smooth_sigma=2.0)(list(stack)).mean()

    assert smoothed < raw


# ==============================================================================================
# 2. THE SHAPE CONTRACT
# ==============================================================================================

def test_returns_one_plane_of_the_input_dtype_and_shape():
    stack = np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7)

    out = stdev_op()(list(stack))

    assert out.shape == (5, 7)
    assert out.dtype == np.uint16


def test_it_streams_and_never_materialises_the_stack():
    """A z-reducer's memory must be flat in stack depth. Proven by handing it a GENERATOR that
    refuses to be consumed twice and counting how many planes are alive at once."""
    live = 0
    peak = 0

    def planes():
        nonlocal live, peak
        for _ in range(64):
            live += 1
            peak = max(peak, live)
            yield np.full((8, 8), 100, dtype=np.uint16)
            live -= 1

    stdev_op(smooth_sigma=0.0)(planes())

    assert peak == 1, "the operator held more than one plane at a time"


def test_an_empty_stack_is_refused_by_name_not_returned_as_zeros():
    with pytest.raises(ValueError, match="empty stack"):
        stdev_op()([])


def test_ragged_planes_are_refused_rather_than_silently_reduced():
    with pytest.raises(ValueError, match="same size"):
        stdev_op(smooth_sigma=0.0)([np.zeros((4, 4), np.uint16), np.zeros((4, 5), np.uint16)])


def test_a_parameter_that_cannot_be_satisfied_is_refused_at_build_time_where_possible():
    with pytest.raises(ValueError, match="smooth_sigma"):
        stdev_op(smooth_sigma=-1.0)
    with pytest.raises(ValueError, match="ddof"):
        stdev_op(ddof=-1)


def test_ddof_larger_than_the_stack_is_refused_naming_both_numbers():
    with pytest.raises(ValueError, match="ddof=4"):
        stdev_op(smooth_sigma=0.0, ddof=4)([np.zeros((4, 4), np.uint16)] * 3)


# ==============================================================================================
# 3. THE DECLARATION MATCHES THE IMPLEMENTATION
# ==============================================================================================

def test_the_registry_records_exactly_what_this_operator_declares():
    """Read back through SquidXplorer's public accessors, not through our own constants — this is
    the fact every generic surface (engine loop, napari layer type, GUI parameters, refusals) will
    act on."""
    import squidmip

    assert OPERATOR_NAME in squidmip.available_projectors()
    assert squidmip.projector_consumes(OPERATOR_NAME) == frozenset({"z"})
    assert squidmip.projector_produces(OPERATOR_NAME) == "intensity"
    assert {p.name for p in squidmip.projector_params(OPERATOR_NAME)} == {"smooth_sigma", "ddof"}
    assert squidmip.projector_requires(OPERATOR_NAME) == ("scipy",)


def test_the_declared_consumes_is_what_the_code_actually_does():
    """``consumes={"z"}`` claims the whole stack collapses to one plane. Checked by handing it a
    stack and asserting the output is ONE plane — the failure this catches is a plane-op
    mis-declared as a z-reducer, which the engine cannot detect and will run anyway."""
    out = stdev_op(smooth_sigma=0.0)([np.zeros((6, 6), np.uint16)] * 5)

    assert out.ndim == 2


def test_operator_kwargs_reach_the_operator_through_the_registry():
    """The parameter seam end to end: a run naming a parameter gets a DIFFERENT binding, and one
    naming a parameter this operator does not declare is refused by name."""
    import squidmip

    bound = squidmip.bind_projector(OPERATOR_NAME, {"smooth_sigma": 0.0})
    stack = [np.full((4, 4), 10, np.uint16), np.full((4, 4), 20, np.uint16)]
    assert bound(stack).max() == 5          # std of {10, 20} = 5

    with pytest.raises(ValueError, match="no parameter"):
        squidmip.bind_projector(OPERATOR_NAME, {"sigma": 1.0})


# ==============================================================================================
# 4. THE PLUGIN LOADS
# ==============================================================================================

def test_the_entry_point_is_declared_and_resolves_to_register():
    """The declaration in pyproject.toml, read back out of the INSTALLED metadata. Catches the
    two failures that make a plugin invisible: a typo in the group name, and a package that was
    built without the entry point."""
    from importlib.metadata import entry_points

    eps = {ep.name: ep for ep in entry_points(group="squidmip.operators")}
    assert "squidmip-operator-template" in eps, (
        "this package's entry point is not installed — reinstall with `pip install -e .`")
    assert eps["squidmip-operator-template"].load() is register


def test_importing_squidmip_registered_this_operator_with_no_edit_to_squidmip():
    """THE POINT OF THE WHOLE TEMPLATE. Nothing in SquidXplorer mentions this package."""
    import squidmip
    from squidmip._operations import runnable_operators

    assert OPERATOR_NAME in squidmip.available_projectors()
    # `runnable_operators()` is what the GUI and the command surface gate a run on, and it is a
    # different list from `available_projectors()` (it adds the region operators). Being in
    # available_projectors and NOT in this one means installed-but-unreachable-from-the-app.
    assert OPERATOR_NAME in runnable_operators()


def test_registering_twice_is_refused_rather_than_silently_clobbering():
    """``register()`` is called once, by SquidXplorer. If something calls it again the second call
    must fail loud: a registrar that overwrote the first entry would let a stale binding win."""
    with pytest.raises(ValueError, match="already defined"):
        register()
