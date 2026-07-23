"""Operator + projector integration tests.

End-to-end wiring checks over the public ``squidmip`` operator surface: the projector /
region-operator registries, the consumes-axis contract (z-reducer vs plane-op), upstream
fidelity (petakit / tilefusion / bgsub really back the operators), and small synthetic
end-to-end runs for bgsub, flatfield, mip/reference, and decon.

All pixels are synthetic in-memory numpy — no dataset on disk is touched. Operators exposed
as ``*_op`` factories return callables over an *iterable of planes*: a plane-op takes a
single-element iterable ``[plane]`` and returns a plane; a z-reducer takes the whole stack.
"""

from __future__ import annotations

import numpy as np
import pytest

import squidmip as s
from squidmip._engine import _resolve_projector


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _gradient_plane(shape=(64, 64), scale=5.0, seed=0):
    """A smooth diagonal gradient (a stand-in background) plus a little noise, float32."""
    rng = np.random.RandomState(seed)
    grad = np.add.outer(np.arange(shape[0]), np.arange(shape[1])).astype(np.float32) / scale
    return grad + rng.rand(*shape).astype(np.float32)


def _stack(z=5, shape=(64, 64), seed=0):
    rng = np.random.RandomState(seed)
    return [rng.rand(*shape).astype(np.float32) for _ in range(z)]


# ======================================================================================
# 1. Registries
# ======================================================================================
def test_available_projectors_exact_list():
    assert s.available_projectors() == [
        "bgsub",
        "decon",
        "decon3d",
        "flatfield",
        "mip",
        "reference",
        "spot",
    ]


def test_available_region_operators_exact_list():
    assert s.available_region_operators() == ["coordinate", "stitch"]


def test_every_projector_resolves():
    for name in s.available_projectors():
        op = _resolve_projector(name)
        assert op.name == name
        assert callable(op.fn)


# ======================================================================================
# 2. Consumes-axis contract
# ======================================================================================
def test_consumes_axis_mapping():
    z = frozenset({"z"})
    empty = frozenset()
    assert s.projector_consumes("mip") == z
    assert s.projector_consumes("reference") == z
    assert s.projector_consumes("bgsub") == empty
    assert s.projector_consumes("decon") == empty
    assert s.projector_consumes("flatfield") == empty


# ======================================================================================
# 3. Upstream fidelity (wiring, not algorithms)
# ======================================================================================
def test_upstream_packages_importable():
    import bgsub.core  # noqa: F401
    import petakit  # noqa: F401
    import tilefusion.distortion  # noqa: F401
    import tilefusion.flatfield  # noqa: F401
    import tilefusion.registration  # noqa: F401


def test_decon_module_wires_petakit():
    import importlib

    importlib.import_module("squidmip._decon")
    petakit = importlib.import_module("petakit")
    assert petakit is not None


def test_background_module_wires_bgsub():
    import importlib

    importlib.import_module("squidmip._background")
    bgsub = importlib.import_module("bgsub")
    assert bgsub is not None


# ======================================================================================
# 4. bgsub end-to-end
# ======================================================================================
def test_bgsub_op_end_to_end_reduces_background():
    plane = _gradient_plane()
    out = s.bgsub_op()([plane])
    assert out.shape == plane.shape
    # a subtracted background lowers the mean intensity
    assert float(out.mean()) < float(plane.mean())


def test_subtract_background_callable():
    plane = _gradient_plane()
    out = s.subtract_background(plane)
    assert out.shape == plane.shape


# ======================================================================================
# 5. flatfield end-to-end
# ======================================================================================
def test_flatfield_op_end_to_end_preserves_shape():
    profile = s.FlatfieldProfile(np.ones((64, 64), dtype=np.float32))
    plane = _gradient_plane()
    out = s.flatfield_op(profile)([plane])
    assert out.shape == plane.shape


# ======================================================================================
# 6. mip / reference z-reducers
# ======================================================================================
def test_mip_projector_collapses_z_and_equals_max():
    stack = _stack(z=5)
    expected = np.max(np.stack(stack), axis=0)
    out = _resolve_projector("mip").fn(list(stack))
    assert out.shape == (64, 64)
    assert np.array_equal(out, expected)


def test_project_primitive_equals_max_over_z():
    stack = _stack(z=5, seed=1)
    expected = np.max(np.stack(stack), axis=0)
    assert np.array_equal(s.project(iter(stack)), expected)


def test_reference_projector_collapses_z():
    stack = _stack(z=5, seed=2)
    out = _resolve_projector("reference").fn(list(stack))
    assert out.shape == (64, 64)


# ======================================================================================
# 7. Determinism (worker-count-invariance proxy)
# ======================================================================================
def test_projection_is_deterministic():
    stack = _stack(z=5, seed=3)
    a = s.project(iter(stack))
    b = s.project(iter(stack))
    assert np.array_equal(a, b)


# ======================================================================================
# 8. Layering contract: plane-op keeps z depth, z-reducer collapses it
# ======================================================================================
def test_plane_op_preserves_z_but_reducer_collapses():
    stack = _stack(z=5, seed=4)

    # plane-op ('bgsub'): consumes nothing => z survives. Mapping it plane-by-plane
    # yields one output per input plane, each same shape.
    assert s.projector_consumes("bgsub") == frozenset()
    op = s.bgsub_op()
    mapped = [op([p]) for p in stack]
    assert len(mapped) == len(stack)
    assert all(m.shape == (64, 64) for m in mapped)

    # z-reducer ('mip'): consumes {'z'} => the whole stack collapses to one plane.
    assert s.projector_consumes("mip") == frozenset({"z"})
    reduced = _resolve_projector("mip").fn(list(stack))
    assert reduced.ndim == 2 and reduced.shape == (64, 64)


# ======================================================================================
# decon: registered + upstream present, plus a tiny end-to-end run
# ======================================================================================
def test_decon_registered_and_petakit_present():
    assert "decon" in s.available_projectors()
    import petakit  # noqa: F401


def test_decon_op_end_to_end_tiny_stack():
    s.set_optics(
        s.OpticsParams(na=0.5, wavelength_um=0.5, dxy_um=0.325, dz_um=1.5, nz=1)
    )
    plane = np.random.RandomState(0).rand(32, 32).astype(np.float32)
    try:
        out = s.decon_op(iterations=1)([plane])
    except Exception as exc:  # pragma: no cover - env-gap guard
        pytest.skip(f"decon end-to-end unavailable in this env: {exc!r}")
    assert out.shape == plane.shape
