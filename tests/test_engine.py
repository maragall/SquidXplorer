"""IMA-188 unit tests — parallel/streaming plate engine + projector table.

Clean-room (no ``integration`` mark, no data on disk): the reader is a controllable in-memory
fake so we can exercise the engine's own logic — projector table, projector swap (AC4), completion
streaming, bounded in-flight window, fail-loud propagation, and metadata warm-up ordering —
without the real 1536wp fixture. The real seam (189 reader → 188 engine on real pixels) is
proven separately by the 188↔183 cross commit in ``tests/test_integration.py``.

The fake mirrors exactly the slice of the IMA-189 reader contract that ``project_well`` and
``project_plate`` touch: a ``metadata`` dict and ``read(region, fov, channel, z, t)`` → 2D plane.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import squidmip._engine as engine
from squidmip import (
    add_projector,
    available_projectors,
    plane_op,
    project_plate,
    project_well,
)


class FakeReader:
    """In-memory stand-in for an IMA-189 ``SquidReader`` — only what the engine consumes.

    Instrumented for the engine tests: it records the order of ``metadata`` vs ``read`` access
    (warm-up ordering), the set of wells that have begun reading (bounded-window / laziness),
    and can be told to sleep per read (to keep wells in flight) or raise on a chosen well
    (fail-loud). ``read`` returns a constant plane whose value grows with ``z`` so a max
    projection has a known, non-degenerate answer.
    """

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

    def read(self, region, fov, channel, z, t=0):
        with self._lock:
            self.events.append("read")
            self.wells_started.add((region, fov))
            self.read_count += 1
        if self._fail_on is not None and (region, fov) == self._fail_on:
            raise ValueError(f"synthetic read failure at region={region!r} fov={fov} z={z}")
        if self._read_sleep:
            time.sleep(self._read_sleep)
        # value grows with z so max-over-z is well-defined and != any lower slice
        base = (hash((region, fov, channel, t)) % 100) * 10
        return np.full(self._shape, base + int(z), dtype=self._dtype)


@pytest.fixture(autouse=True)
def _restore_projector_table():
    """Snapshot/restore the module-global projector table so tests that add don't leak."""
    saved = dict(engine._PROJECTORS)
    try:
        yield
    finally:
        engine._PROJECTORS.clear()
        engine._PROJECTORS.update(saved)


def _collect(reader, **kw) -> dict[tuple[str, int], np.ndarray]:
    """Drain project_plate into a {(region, fov): image} dict (order-independent compare)."""
    return {(r, f): img for r, f, img in project_plate(reader, **kw)}


# ── projector table ─────────────────────────────────────────────────────────────────────

def test_mip_is_available_by_default():
    assert "mip" in available_projectors()


def test_available_projectors_is_sorted_and_reflects_registration():
    add_projector("zzz_custom", lambda planes: next(iter(planes)))
    names = available_projectors()
    assert names == sorted(names)
    assert "zzz_custom" in names


def test_add_duplicate_name_raises():
    with pytest.raises(ValueError, match="already defined"):
        add_projector("mip", lambda planes: next(iter(planes)))


def test_add_rejects_empty_name_and_non_callable():
    with pytest.raises(ValueError, match="non-empty"):
        add_projector("", lambda planes: next(iter(planes)))
    with pytest.raises(ValueError, match="not callable"):
        add_projector("bad", object())  # type: ignore[arg-type]


def test_project_plate_unknown_projector_raises_named():
    reader = FakeReader(n_wells=2)
    with pytest.raises(KeyError, match="unknown projector 'nope'"):
        next(project_plate(reader, projector="nope"))


# ── correctness: concurrency changes no pixel ───────────────────────────────────────────

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
    assert len(out) == 6  # 3 wells × 2 fovs
    assert {f for _, f in out} == {0, 1}


# ── AC4: pluggable projector swaps with zero engine edits ────────────────────────────────

def test_projector_swap_runs_through_the_same_engine():
    # A non-MIP projector (returns the FIRST z-plane) selected purely by name — the engine
    # code is untouched. Proves project_plate(..., projector=<name>) is the pluggable seam.
    add_projector("first_z", lambda planes: next(iter(planes)))
    reader = FakeReader(n_wells=3, z_levels=(0, 1, 2, 3))
    out = _collect(reader, workers=2, projector="first_z")
    for (region, fov), img in out.items():
        for c_i, ch in enumerate(reader._channels):
            first_plane = reader.read(region, fov, ch, reader._z_levels[0])
            np.testing.assert_array_equal(img[0, c_i, 0], first_plane)
            # and it is genuinely NOT the MIP (which would pick the largest z)
            assert not np.array_equal(img[0, c_i, 0], project_well(reader, region, fov)[0, c_i, 0])


# ── fail loud (per-well resilience is IMA-186's, not the engine's) ───────────────────────

def test_failure_in_one_well_propagates_and_aborts_the_stream():
    reader = FakeReader(n_wells=6, fail_on=("W0003", 0))
    with pytest.raises(ValueError, match="synthetic read failure at region='W0003'"):
        _collect(reader, workers=3)


# ── bounded in-flight window / laziness (peak RSS flat vs plate size) ────────────────────

def test_bounded_window_does_not_prefetch_the_whole_plate():
    # With N wells >> workers, consuming ONE result must have started at most `workers + 1`
    # wells (prime `workers`, one refill after the single completion) — NOT all N. This is the
    # invariant that keeps ~139 MB per-well results from piling up → peak RSS flat in plate size.
    n_wells, workers = 40, 3
    reader = FakeReader(n_wells=n_wells, read_sleep=0.01)
    gen = project_plate(reader, workers=workers)
    try:
        next(gen)  # consume exactly one well
        with reader._lock:
            started = len(reader.wells_started)
        assert started <= workers + 1, f"prefetched {started} wells with only {workers} workers"
        assert started < n_wells  # emphatically not the whole plate
    finally:
        gen.close()  # GeneratorExit → ThreadPoolExecutor shuts down


def test_metadata_is_warmed_before_any_read():
    # The engine must touch reader.metadata (single-threaded) before fanning out reads, so the
    # IMA-189 reader's lazy index/time-folders are populated before concurrent read() calls.
    reader = FakeReader(n_wells=4)
    list(project_plate(reader, workers=2))
    assert reader.events, "engine never touched the reader"
    assert reader.events[0] == "meta"
    assert reader.events.index("meta") < reader.events.index("read")


# ── argument validation ──────────────────────────────────────────────────────────────────

def test_invalid_workers_raises():
    reader = FakeReader(n_wells=2)
    with pytest.raises(ValueError, match="workers must be >= 1"):
        next(project_plate(reader, workers=0))


# ══════════════════════════════════════════════════════════════════════════════════════════
# IMA-210 — the consumes-axis registry
#
# An operator declares WHICH AXIS it eats, and the engine derives the loop from that:
#   consumes = frozenset()      plane-op   plane -> plane, z SURVIVES (decon/bgsub/flatfield)
#   consumes = frozenset({"z"}) z-reducer  all z -> one plane, z collapses to 1 (mip/reference)
# One group-by-then-reduce loop serves both. `consumes` is orthogonal to `select_index`
# (which is HOW a z-reducer picks, not WHICH axis it eats) — both mip and reference are {"z"}.
# ══════════════════════════════════════════════════════════════════════════════════════════

def _first(planes):
    return next(iter(planes))


def _plus_one(plane):
    """A plane-op written the natural way: plane -> plane."""
    return plane + 1


# ── the declaration ───────────────────────────────────────────────────────────────────────

def test_shipped_projectors_declare_the_z_axis():
    # BOTH mip and reference consume z. z-SELECTING (reference) is not a different axis: it is
    # a different way of picking within z. Splitting them here is what broke channel alignment.
    assert engine.projector_consumes("mip") == frozenset({"z"})
    assert engine.projector_consumes("reference") == frozenset({"z"})


def test_consumes_is_orthogonal_to_select_index():
    from squidmip.projection import project as mip, project_reference
    assert getattr(mip, "select_index", None) is None
    assert getattr(project_reference, "select_index", None) is not None
    # ...yet they declare the SAME consumed axis.
    assert engine.projector_consumes("mip") == engine.projector_consumes("reference")


def test_add_projector_defaults_to_z_reducer():
    add_projector("legacy_style", _first)                 # no consumes= → the old contract
    assert engine.projector_consumes("legacy_style") == frozenset({"z"})


def test_add_projector_records_a_plane_op():
    add_projector("planeop", plane_op(_plus_one), consumes=frozenset())
    assert engine.projector_consumes("planeop") == frozenset()


def test_consumes_accepts_any_iterable_of_axis_names():
    add_projector("as_set", _first, consumes={"z"})
    add_projector("as_str", _first, consumes="z")
    add_projector("as_tuple", _first, consumes=())
    assert engine.projector_consumes("as_set") == frozenset({"z"})
    assert engine.projector_consumes("as_str") == frozenset({"z"})
    assert engine.projector_consumes("as_tuple") == frozenset()


def test_projector_consumes_unknown_name_is_loud():
    with pytest.raises(KeyError, match="unknown projector 'nope'"):
        engine.projector_consumes("nope")


# ── the axes this seam refuses (IMA-222 owns inter-FOV) ───────────────────────────────────

def test_fov_is_refused_by_name_and_points_at_the_region_seam():
    # A stitcher consumes fov, but a _PROJECTORS callable is Iterable[plane] -> plane and never
    # sees a tile's x/y stage geometry. Declaring {"fov"} here would be a promise we cannot keep.
    with pytest.raises(ValueError, match="fov"):
        add_projector("stitch", _first, consumes=frozenset({"fov"}))


def test_unknown_axis_is_refused_named():
    with pytest.raises(ValueError, match="unsupported.*'t'|'t'.*unsupported"):
        add_projector("timelapse", _first, consumes=frozenset({"t"}))


# ── the engine: one group-by-then-reduce loop, two shapes ─────────────────────────────────

def test_plane_op_preserves_z_and_maps_each_plane():
    add_projector("plus_one", plane_op(_plus_one), consumes=frozenset())
    reader = FakeReader(n_wells=2, z_levels=(0, 1, 3))
    out = _collect(reader, workers=2, projector="plus_one")
    for (region, fov), img in out.items():
        # z SURVIVES a plane-op — one output plane per input plane, in z_levels order.
        assert img.shape == (reader._n_t, len(reader._channels), 3, *reader._shape)
        for c_i, ch in enumerate(reader._channels):
            for k, z in enumerate(reader._z_levels):
                np.testing.assert_array_equal(img[0, c_i, k], reader.read(region, fov, ch, z) + 1)


def test_plane_op_is_never_routed_through_the_z_reduction():
    # THE point of `consumes`: a plane-op must see exactly ONE plane per call, never the stack.
    seen = []

    def spy(planes):
        planes = list(planes)
        seen.append(len(planes))
        return planes[0]

    add_projector("spy", spy, consumes=frozenset())
    reader = FakeReader(n_wells=1, z_levels=(0, 1, 2, 3))
    _collect(reader, workers=1, projector="spy")
    assert seen and set(seen) == {1}, f"plane-op was handed stacks of {sorted(set(seen))} planes"


def test_z_reducer_still_sees_the_whole_stack():
    seen = []

    def spy(planes):
        planes = list(planes)
        seen.append(len(planes))
        return planes[0]

    add_projector("spy_z", spy, consumes=frozenset({"z"}))
    reader = FakeReader(n_wells=1, z_levels=(0, 1, 2, 3))
    _collect(reader, workers=1, projector="spy_z")
    assert seen and set(seen) == {4}


def test_adding_a_plane_op_needs_zero_engine_edits():
    # The whole abstraction test: a new operator is ONE add_projector call + a name at the
    # call site. If this ever needs an engine branch, the registry is the wrong shape.
    add_projector("bgsub_like", plane_op(lambda p: (p // 2)), consumes=frozenset())
    assert "bgsub_like" in available_projectors()
    reader = FakeReader(n_wells=1, z_levels=(0, 1))
    ((_, img),) = list(_collect(reader, workers=1, projector="bgsub_like").items())
    np.testing.assert_array_equal(img[0, 0, 0], reader.read("W0000", 0, "c0", 0) // 2)


# ── regression guards: MIP behaviour is untouched ─────────────────────────────────────────

def test_mip_shape_is_still_z_collapsed_to_one():
    reader = FakeReader(n_wells=3, z_levels=(0, 1, 2))
    for img in _collect(reader, workers=2).values():
        assert img.shape[2] == 1


def test_n_equals_1_mip_is_byte_identical_to_the_single_plane():
    # THE regression guard: with one z, a MIP must return that plane's bytes, unchanged.
    reader = FakeReader(n_wells=2, z_levels=(7,))
    for (region, fov), img in _collect(reader, workers=2).items():
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
    # plane_op() stamps `consumes` on the callable, so the registration site does not have to
    # repeat it — the same idiom project_reference already uses for `select_index`.
    add_projector("inferred", plane_op(_plus_one))
    assert engine.projector_consumes("inferred") == frozenset()
