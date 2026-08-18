"""Operator integration tests over the public ``squidxplorer`` surface.

All pixels are synthetic in-memory numpy; no dataset on disk is touched.
"""

from __future__ import annotations

import numpy as np
import pytest

# These exercise standalone repos the clean-room CI install does not have; skip when absent.
pytest.importorskip("petakit")
pytest.importorskip("tilefusion")
pytest.importorskip("bgsub")

import squidxplorer as s
from squidxplorer._engine import _resolve_operator


def _gradient_plane(shape=(64, 64), scale=5.0, seed=0):
    """A smooth diagonal gradient (a stand-in background) plus a little noise, float32."""
    rng = np.random.RandomState(seed)
    grad = np.add.outer(np.arange(shape[0]), np.arange(shape[1])).astype(np.float32) / scale
    return grad + rng.rand(*shape).astype(np.float32)


def _stack(z=5, shape=(64, 64), seed=0):
    rng = np.random.RandomState(seed)
    return [rng.rand(*shape).astype(np.float32) for _ in range(z)]


def test_available_plane_operators_exact_list():
    assert s.available_plane_operators() == [
        "bgsub",
        "cellpose",
        "decon",
        "decon3d",
        "flatfield",
        "keepz",
        "mip",
        "reference",
        "spot",
    ]


def test_available_region_operators_exact_list():
    assert s.available_region_operators() == ["coordinate", "register", "stitch"]


def test_every_operator_resolves():
    for name in s.available_plane_operators():
        op = _resolve_operator(name)
        assert op.name == name
        assert callable(op.fn)


def test_consumes_axis_mapping():
    z = frozenset({"z"})
    empty = frozenset()
    assert s.operator_consumes("mip") == z
    assert s.operator_consumes("reference") == z
    assert s.operator_consumes("bgsub") == empty
    assert s.operator_consumes("decon") == empty
    assert s.operator_consumes("flatfield") == empty


def test_upstream_packages_importable():
    import bgsub.core  # noqa: F401
    import petakit  # noqa: F401
    import tilefusion.distortion  # noqa: F401
    import tilefusion.flatfield  # noqa: F401
    import tilefusion.registration  # noqa: F401


def test_decon_module_wires_petakit():
    """"Wired" is the ``requires=`` declaration, not the import."""
    import importlib

    importlib.import_module("squidxplorer._decon")
    for name in ("decon", "decon3d"):
        assert "petakit" in s.operator_requires(name), (name, s.operator_requires(name))


def test_background_module_wires_bgsub():
    """`estimate_background(method="sep")` must actually reach ``bgsub.core._run_sep``."""
    import importlib

    bg = importlib.import_module("squidxplorer._background")
    import bgsub.core

    calls = []
    real = bgsub.core._run_sep

    def spy(img, radius):
        calls.append((img.shape, radius))
        return real(img, radius)

    bgsub.core._run_sep = spy
    try:
        out = bg.estimate_background(
            np.random.RandomState(0).rand(32, 32).astype(np.float32),
            bg.BackgroundParams(method="sep", radius_px=8),
        )
    finally:
        bgsub.core._run_sep = real
    assert calls == [((32, 32), 8)], calls
    assert out.shape == (32, 32) and out.dtype == np.float32


def test_bgsub_op_end_to_end_reduces_background():
    plane = _gradient_plane()
    out = s.bgsub_op()([plane])
    assert out.shape == plane.shape
    assert float(out.mean()) < float(plane.mean())


def test_subtract_background_callable():
    plane = _gradient_plane()
    out = s.subtract_background(plane)
    assert out.shape == plane.shape


def test_flatfield_op_end_to_end_preserves_shape():
    profile = s.FlatfieldProfile(np.ones((64, 64), dtype=np.float32))
    plane = _gradient_plane()
    out = s.flatfield_op(profile)([plane])
    assert out.shape == plane.shape


def test_mip_collapses_z_and_equals_max():
    stack = _stack(z=5)
    expected = np.max(np.stack(stack), axis=0)
    out = _resolve_operator("mip").fn(list(stack))
    assert out.shape == (64, 64)
    assert np.array_equal(out, expected)


def test_project_primitive_equals_max_over_z():
    stack = _stack(z=5, seed=1)
    expected = np.max(np.stack(stack), axis=0)
    assert np.array_equal(s.project(iter(stack)), expected)


def test_reference_collapses_z():
    stack = _stack(z=5, seed=2)
    out = _resolve_operator("reference").fn(list(stack))
    assert out.shape == (64, 64)


def test_projection_is_deterministic():
    stack = _stack(z=5, seed=3)
    a = s.project(iter(stack))
    b = s.project(iter(stack))
    assert np.array_equal(a, b)


def test_plane_op_preserves_z_but_reducer_collapses():
    stack = _stack(z=5, seed=4)

    # plane-op: consumes nothing => z survives, one output per input plane
    assert s.operator_consumes("bgsub") == frozenset()
    op = s.bgsub_op()
    mapped = [op([p]) for p in stack]
    assert len(mapped) == len(stack)
    assert all(m.shape == (64, 64) for m in mapped)

    # z-reducer: consumes {'z'} => the whole stack collapses to one plane
    assert s.operator_consumes("mip") == frozenset({"z"})
    reduced = _resolve_operator("mip").fn(list(stack))
    assert reduced.ndim == 2 and reduced.shape == (64, 64)


def test_decon_registered_and_petakit_present():
    assert "decon" in s.available_plane_operators()
    import petakit  # noqa: F401


def test_decon_op_end_to_end_tiny_stack():
    s.set_optics(
        s.OpticsParams(na=0.5, wavelength_um=0.5, dxy_um=0.325, dz_um=1.5, nz=1)
    )
    plane = np.random.RandomState(0).rand(32, 32).astype(np.float32)
    # Only a missing package skips; any other fault must fail the test.
    try:
        out = s.decon_op(iterations=1)([plane])
    except ImportError as exc:
        pytest.skip(f"decon needs a package that is not installed here: {exc!r}")
    assert out.shape == plane.shape
    assert np.isfinite(out).all(), "decon returned non-finite pixels"
