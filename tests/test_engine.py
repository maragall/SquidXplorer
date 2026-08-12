"""Parallel/streaming plate engine + operator table, over an in-memory fake reader."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import squidxplorer._engine as engine
from squidxplorer import (
    add_operator,
    available_plane_operators,
    plane_op,
    project_well,
    run_plate,
)


class FakeReader:
    """In-memory stand-in for a ``SquidReader``, instrumented for the engine tests."""

    def __init__(
        self,
        n_wells: int = 4,
        *,
        n_fovs: int = 1,
        channels: tuple[str, ...] = ("c0", "c1"),
        z_levels: tuple[int, ...] = (0, 1, 2),
        n_t: int = 1,
        shape: tuple[int, int] = (4, 4),
        dtype=np.uint16,
        read_sleep: float = 0.0,
        fail_on: tuple[str, int] | None = None,
    ) -> None:
        self._regions = [f"W{i:04d}" for i in range(n_wells)]
        self._fovs = list(range(n_fovs))
        self._channels = list(channels)
        self._z_levels = list(z_levels)
        self._n_t = n_t
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._read_sleep = read_sleep
        self._fail_on = fail_on

        # instrumentation (thread-safe)
        self._lock = threading.Lock()
        self.events: list[str] = []          # "meta" / "read" in first-touch order
        self.wells_started: set[tuple[str, int]] = set()
        self.read_count = 0

    @property
    def metadata(self) -> dict:
        with self._lock:
            self.events.append("meta")
        return {
            "regions": self._regions,
            "fovs_per_region": {r: list(self._fovs) for r in self._regions},
            "channels": [{"name": c} for c in self._channels],
            "z_levels": self._z_levels,
            "n_z": len(self._z_levels),
            "n_t": self._n_t,
            "frame_shape": self._shape,
            "dtype": self._dtype,
        }

    def read(self, region, fov, channel, z_level, time_point=0):
        with self._lock:
            self.events.append("read")
            self.wells_started.add((region, fov))
            self.read_count += 1
        if self._fail_on is not None and (region, fov) == self._fail_on:
            raise ValueError(f"synthetic read failure at region={region!r} fov={fov} z={z_level}")
        if self._read_sleep:
            time.sleep(self._read_sleep)
        # Value grows with z so max-over-z is well-defined.
        base = (hash((region, fov, channel, time_point)) % 100) * 10
        return np.full(self._shape, base + int(z_level), dtype=self._dtype)


@pytest.fixture(autouse=True)
def _restore_operator_table():
    """Snapshot/restore the module-global operator table so tests that add don't leak."""
    saved = dict(engine._OPERATORS)
    try:
        yield
    finally:
        engine._OPERATORS.clear()
        engine._OPERATORS.update(saved)


def _collect(reader, **kw) -> dict[tuple[str, int], np.ndarray]:
    """Drain run_plate into a {(region, fov): image} dict (order-independent compare)."""
    return {(r, f): img for r, f, img in run_plate(reader, **kw)}


def test_mip_is_available_by_default():
    assert "mip" in available_plane_operators()


def test_available_plane_operators_is_sorted_and_reflects_registration():
    add_operator("zzz_custom", lambda planes: next(iter(planes)))
    names = available_plane_operators()
    assert names == sorted(names)
    assert "zzz_custom" in names


def test_add_duplicate_name_raises():
    with pytest.raises(ValueError, match="already defined"):
        add_operator("mip", lambda planes: next(iter(planes)))


def test_add_rejects_empty_name_and_non_callable():
    with pytest.raises(ValueError, match="non-empty"):
        add_operator("", lambda planes: next(iter(planes)))
    with pytest.raises(ValueError, match="not callable"):
        add_operator("bad", object())  # type: ignore[arg-type]


def test_run_plate_unknown_operator_raises_named():
    reader = FakeReader(n_wells=2)
    with pytest.raises(KeyError, match="unknown operator 'nope'"):
        next(run_plate(reader, operator="nope"))


def test_the_per_fov_loop_refuses_a_region_operator_by_its_declaration():
    """`stitch` resolves here; `consumes` is what says it is the wrong loop."""
    reader = FakeReader(n_wells=2)
    with pytest.raises(ValueError, match="consumes fov.*run_plate"):
        next(engine._project_plate(reader, operator="stitch"))


def test_yields_every_well_with_correct_shape_and_dtype():
    reader = FakeReader(n_wells=7)
    out = _collect(reader, workers=3)
    assert set(out) == {(f"W{i:04d}", 0) for i in range(7)}
    for img in out.values():
        assert img.shape == (reader._n_t, len(reader._channels), 1, *reader._shape)
        assert img.dtype == reader._dtype


def test_parallel_output_is_pixel_identical_to_single_thread():
    reader = FakeReader(n_wells=5, channels=("c0", "c1", "c2"))
    parallel = _collect(reader, workers=4)
    for (region, fov), img in parallel.items():
        expected = project_well(reader, region, fov)  # single-thread reference
        np.testing.assert_array_equal(img, expected)


def test_result_is_deterministic_across_worker_counts():
    reader = FakeReader(n_wells=9)
    one = _collect(reader, workers=1)
    many = _collect(reader, workers=4)
    assert set(one) == set(many)
    for key in one:
        np.testing.assert_array_equal(one[key], many[key])


def test_respects_n_fovs():
    reader = FakeReader(n_wells=3, n_fovs=2)
    out = _collect(reader, workers=2, n_fovs=2)
    assert len(out) == 6  # 3 wells x 2 fovs
    assert {f for _, f in out} == {0, 1}


def test_operator_swap_runs_through_the_same_engine():
    # A non-MIP operator selected purely by name; the engine code is untouched.
    add_operator("first_z", lambda planes: next(iter(planes)))
    reader = FakeReader(n_wells=3, z_levels=(0, 1, 2, 3))
    out = _collect(reader, workers=2, operator="first_z")
    # Guard against an empty dict: a loop over nothing asserts nothing.
    assert set(out) == {(f"W{i:04d}", 0) for i in range(3)}, sorted(out)
    for (region, fov), img in out.items():
        for c_i, ch in enumerate(reader._channels):
            first_plane = reader.read(region, fov, ch, reader._z_levels[0])
            np.testing.assert_array_equal(img[0, c_i, 0], first_plane)
            # and it is genuinely NOT the MIP
            assert not np.array_equal(img[0, c_i, 0], project_well(reader, region, fov)[0, c_i, 0])


def test_failure_in_one_well_propagates_and_aborts_the_stream():
    reader = FakeReader(n_wells=6, fail_on=("W0003", 0))
    with pytest.raises(ValueError, match="synthetic read failure at region='W0003'"):
        _collect(reader, workers=3)


def test_bounded_window_does_not_prefetch_the_whole_plate():
    # Consuming one result must have started at most `workers + 1` wells, not all N.
    n_wells, workers = 40, 3
    reader = FakeReader(n_wells=n_wells, read_sleep=0.01)
    gen = run_plate(reader, workers=workers)
    try:
        next(gen)  # consume exactly one well
        with reader._lock:
            started = len(reader.wells_started)
        assert started <= workers + 1, f"prefetched {started} wells with only {workers} workers"
        assert started < n_wells  # emphatically not the whole plate
    finally:
        gen.close()  # GeneratorExit -> ThreadPoolExecutor shuts down


def test_metadata_is_warmed_before_any_read():
    # metadata must be touched single-threaded before reads fan out.
    reader = FakeReader(n_wells=4)
    list(run_plate(reader, workers=2))
    assert reader.events, "engine never touched the reader"
    assert reader.events[0] == "meta"
    assert reader.events.index("meta") < reader.events.index("read")


def test_invalid_workers_raises():
    reader = FakeReader(n_wells=2)
    with pytest.raises(ValueError, match="workers must be >= 1"):
        next(run_plate(reader, workers=0))


# The consumes-axis registry: consumes=frozenset() is a plane-op (z survives),
# consumes={"z"} is a z-reducer (z collapses to 1). One group-by-then-reduce loop serves both.

def _first(planes):
    return next(iter(planes))


def _plus_one(plane):
    return plane + 1


def test_shipped_operators_declare_the_z_axis():
    # Both mip and reference consume z; z-selecting is a way of picking within z.
    assert engine.operator_consumes("mip") == frozenset({"z"})
    assert engine.operator_consumes("reference") == frozenset({"z"})


def test_consumes_is_orthogonal_to_select_index():
    from squidxplorer.projection import project as mip, project_reference
    assert getattr(mip, "select_index", None) is None
    assert getattr(project_reference, "select_index", None) is not None
    # ...yet they declare the same consumed axis.
    assert engine.operator_consumes("mip") == engine.operator_consumes("reference")


def test_add_operator_defaults_to_z_reducer():
    add_operator("legacy_style", _first)                 # no consumes=
    assert engine.operator_consumes("legacy_style") == frozenset({"z"})


def test_add_operator_records_a_plane_op():
    add_operator("planeop", plane_op(_plus_one), consumes=frozenset())
    assert engine.operator_consumes("planeop") == frozenset()


def test_consumes_accepts_any_iterable_of_axis_names():
    add_operator("as_set", _first, consumes={"z"})
    add_operator("as_str", _first, consumes="z")
    add_operator("as_tuple", _first, consumes=())
    assert engine.operator_consumes("as_set") == frozenset({"z"})
    assert engine.operator_consumes("as_str") == frozenset({"z"})
    assert engine.operator_consumes("as_tuple") == frozenset()


def test_operator_consumes_unknown_name_is_loud():
    with pytest.raises(KeyError, match="unknown operator 'nope'"):
        engine.operator_consumes("nope")


def test_fov_is_refused_by_name_and_points_at_the_region_seam():
    # An add_operator callable never sees a tile's x/y stage geometry; {"fov"} is
    # add_region_operator's to stamp.
    with pytest.raises(ValueError, match="fov"):
        add_operator("would_be_stitch", _first, consumes=frozenset({"fov"}))


def test_unknown_axis_is_refused_named():
    with pytest.raises(ValueError, match="unsupported.*'t'|'t'.*unsupported"):
        add_operator("timelapse", _first, consumes=frozenset({"t"}))


def test_plane_op_preserves_z_and_maps_each_plane():
    add_operator("plus_one", plane_op(_plus_one), consumes=frozenset())
    reader = FakeReader(n_wells=2, z_levels=(0, 1, 3))
    out = _collect(reader, workers=2, operator="plus_one")
    assert set(out) == {(f"W{i:04d}", 0) for i in range(2)}, sorted(out)
    for (region, fov), img in out.items():
        # z survives a plane-op: one output plane per input plane, in z_levels order.
        assert img.shape == (reader._n_t, len(reader._channels), 3, *reader._shape)
        for c_i, ch in enumerate(reader._channels):
            for k, z in enumerate(reader._z_levels):
                np.testing.assert_array_equal(img[0, c_i, k], reader.read(region, fov, ch, z) + 1)


def test_plane_op_is_never_routed_through_the_z_reduction():
    # A plane-op must see exactly one plane per call, never the stack.
    seen = []

    def spy(planes):
        planes = list(planes)
        seen.append(len(planes))
        return planes[0]

    add_operator("spy", spy, consumes=frozenset())
    reader = FakeReader(n_wells=1, z_levels=(0, 1, 2, 3))
    _collect(reader, workers=1, operator="spy")
    assert seen and set(seen) == {1}, f"plane-op was handed stacks of {sorted(set(seen))} planes"


def test_z_reducer_still_sees_the_whole_stack():
    seen = []

    def spy(planes):
        planes = list(planes)
        seen.append(len(planes))
        return planes[0]

    add_operator("spy_z", spy, consumes=frozenset({"z"}))
    reader = FakeReader(n_wells=1, z_levels=(0, 1, 2, 3))
    _collect(reader, workers=1, operator="spy_z")
    assert seen and set(seen) == {4}


def test_adding_a_plane_op_needs_zero_engine_edits():
    add_operator("bgsub_like", plane_op(lambda p: (p // 2)), consumes=frozenset())
    assert "bgsub_like" in available_plane_operators()
    reader = FakeReader(n_wells=1, z_levels=(0, 1))
    ((_, img),) = list(_collect(reader, workers=1, operator="bgsub_like").items())
    np.testing.assert_array_equal(img[0, 0, 0], reader.read("W0000", 0, "c0", 0) // 2)


def test_mip_shape_is_still_z_collapsed_to_one():
    reader = FakeReader(n_wells=3, z_levels=(0, 1, 2))
    out = _collect(reader, workers=2)
    assert set(out) == {(f"W{i:04d}", 0) for i in range(3)}, sorted(out)
    for img in out.values():
        assert img.shape[2] == 1


def test_n_equals_1_mip_is_byte_identical_to_the_single_plane():
    # With one z, a MIP must return that plane's bytes, unchanged.
    reader = FakeReader(n_wells=2, z_levels=(7,))
    out = _collect(reader, workers=2)
    assert set(out) == {(f"W{i:04d}", 0) for i in range(2)}, sorted(out)
    for (region, fov), img in out.items():
        for c_i, ch in enumerate(reader._channels):
            plane = reader.read(region, fov, ch, 7)
            np.testing.assert_array_equal(img[0, c_i, 0], plane)
            assert img.dtype == plane.dtype


def test_mip_pixels_unchanged_by_the_registry_rewrite():
    reader = FakeReader(n_wells=4, channels=("c0", "c1", "c2"), z_levels=(0, 2, 5))
    for (region, fov), img in _collect(reader, workers=3).items():
        for c_i, ch in enumerate(reader._channels):
            stack = [reader.read(region, fov, ch, z) for z in reader._z_levels]
            np.testing.assert_array_equal(img[0, c_i, 0], np.max(np.stack(stack), axis=0))


def test_plane_op_adapter_makes_the_declaration_inferable():
    # plane_op() stamps `consumes` on the callable, so the registration site need not repeat it.
    add_operator("inferred", plane_op(_plus_one))
    assert engine.operator_consumes("inferred") == frozenset()
