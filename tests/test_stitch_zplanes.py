"""Per-plane 3-D fusion: every z plane fused, with ONE solved geometry.

The fixture is tests/test_stitch.py's synthetic mosaic with a Z axis bolted on: each z plane is
the master texture plus a per-plane constant, so a plane is identifiable by its pixel values
alone and a swapped/duplicated/missing plane is a numeric failure, not a shape one.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tilefusion", reason="tilefusion (maragall/stitcher) not installed: the stitch "
                                         "adapter is UNTESTED here, which is not the same as passing")

from squidxplorer._engine import _OPERATORS, add_operator
from squidxplorer._flatfield import (
    FlatfieldProfile,
    clear_profile,
    correct_flatfield,
    set_profiles,
)
from squidxplorer._stitch import stitch_region
from squidxplorer.projection import PLANE_OP, plane_op

from tests.test_stitch import CHANNELS, GRID, TILE, _FakeReader, _master

N_Z = 4
_PLANE_OFFSET = 300      # plane k = master + k * _PLANE_OFFSET, so planes are distinguishable


class _ZReader(_FakeReader):
    """The stitch fixture with a real z axis: plane k is the master lifted by k*_PLANE_OFFSET."""

    def __init__(self, *args, n_z: int = N_Z, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata["n_z"] = n_z
        self.metadata["z_levels"] = list(range(n_z))
        self.read_z: list[int] = []

    def read(self, region, fov, channel, z_level=0, time_point=0):
        self.read_z.append(int(z_level))
        base = super().read(region, fov, channel, 0, time_point).astype(np.int32)
        return np.clip(base + int(z_level) * _PLANE_OFFSET, 0, 65535).astype(np.uint16)


@pytest.fixture(scope="module")
def master():
    return _master()


def _passthrough(p):
    return p


_PASSTHROUGH = "zplanes_passthrough"


@pytest.fixture(autouse=True, scope="module")
def _register_passthrough():
    """Registered for this module only: `add_operator` writes to a process-global table that
    tests/test_operator_integration.py asserts the exact contents of."""
    add_operator(_PASSTHROUGH, plane_op(_passthrough))
    try:
        yield
    finally:
        _OPERATORS.pop(_PASSTHROUGH, None)


def _identity_profile(shape=(TILE, TILE)) -> FlatfieldProfile:
    return FlatfieldProfile(np.ones(shape, dtype=np.float32), None)


def _vignette_profile(shape=(TILE, TILE)) -> FlatfieldProfile:
    """A real-ish ~10%-deep radial vignette, deep enough that a double-apply is visible."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    r2 = ((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2
    field = 1.0 - 0.10 * r2 / 2.0
    field /= field.mean()          # FlatfieldProfile requires mean 1.0 (a gain, not a brightness)
    return FlatfieldProfile(field.astype(np.float32), None)


def test_a_plane_op_fuses_every_z_plane(master):
    reader = _ZReader(master)
    out = stitch_region(reader, "A1", list(range(GRID * GRID)), z_operator=_PASSTHROUGH,
                        register=False, correct_illumination=False)

    assert out.ndim == 5
    assert out.shape[2] == N_Z, f"expected {N_Z} fused z planes, got {out.shape[2]}"
    # every plane distinct: rules out "plane 0 written N_Z times"
    signatures = {out[0, 0, z].tobytes() for z in range(N_Z)}
    assert len(signatures) == N_Z, "fused planes are not distinct — z was broadcast, not fused"
    # and in the right order: a shuffled z axis passes the distinctness check but not this
    means = [float(out[0, 0, z].mean()) for z in range(N_Z)]
    assert means == sorted(means), f"z planes came out in the wrong order: {means}"


def test_a_z_reducer_is_unchanged_and_still_collapses_z(master):
    reader = _ZReader(master)
    out = stitch_region(reader, "A1", list(range(GRID * GRID)), z_operator="mip",
                        register=False, correct_illumination=False)
    assert out.shape[2] == 1
    assert not isinstance(out.base, np.memmap), "a single fused plane must not spill to disk"


def test_the_z_loop_reads_one_plane_at_a_time(master):
    """Streaming, not stacking (~9.4 GB on the real set otherwise). Asserted on read ORDER, not
    RSS: all of plane k's reads must happen before any of plane k+1's."""
    reader = _ZReader(master)
    stitch_region(reader, "A1", list(range(GRID * GRID)), z_operator=None,
                  register=False, correct_illumination=False)

    runs = [z for i, z in enumerate(reader.read_z) if i == 0 or z != reader.read_z[i - 1]]
    assert runs == list(range(N_Z)), (
        f"z was not read outermost — read-order z runs were {runs}, expected one contiguous run "
        f"per plane. A z-inner loop holds every plane's tiles at once (~9.4 GB on the real set)."
    )


def test_every_plane_is_fused_with_the_same_solved_offsets(master):
    """Registration is solved once, on one raw plane; a per-plane re-solve would shear the stack
    with depth. Proven two ways: the solver is called once, and each fused plane equals a
    single-plane fuse of the same data to within 1 count (float32 blend, uint16 round-trip)."""
    error = {1: (0.0, 5.0), 2: (4.0, 0.0)}      # a real, registrable stage error to solve away
    reader = _ZReader(_master(), error_px=error)

    calls = []
    import squidxplorer._stitch as stitch_mod
    real_solve = stitch_mod.solve_offsets_px

    def _spy(*a, **kw):
        calls.append(1)
        return real_solve(*a, **kw)

    stitch_mod.solve_offsets_px = _spy
    try:
        out = stitch_region(reader, "A1", list(range(GRID * GRID)), z_operator=_PASSTHROUGH,
                            correct_illumination=False, correct_distortion=False)
    finally:
        stitch_mod.solve_offsets_px = real_solve

    assert len(calls) == 1, (
        f"registration ran {len(calls)} times for {N_Z} z planes; it must run ONCE — a per-plane "
        "solve makes the planes disagree by their residuals and the stack shears with depth")

    p0 = out[0, 0, 0].astype(np.int32)
    for z in range(1, N_Z):
        delta = out[0, 0, z].astype(np.int32) - p0
        inside = np.asarray(delta)[np.asarray(out[0, 0, 0]) > 0]   # skip the unwritten border
        worst = int(np.abs(inside - z * _PLANE_OFFSET).max())
        assert worst <= 1, (
            f"plane {z} is not registered identically to plane 0: the plane-to-plane difference is "
            f"off by up to {worst} counts from the constant {z * _PLANE_OFFSET} the fixture put "
            "there, which means the seams moved between planes")


def test_the_placement_reports_one_solve_for_the_whole_stack(master):
    reader = _ZReader(master)
    out = stitch_region(reader, "A1", list(range(GRID * GRID)), z_operator=_PASSTHROUGH,
                        correct_illumination=False, correct_distortion=False)
    placement = out.placement
    assert placement.reg_z == N_Z // 2, "geometry must be solved on the acquisition's middle plane"
    assert placement.reg_t == 0
    assert len(placement.offsets_px) == GRID * GRID
    assert len(placement.origins_px) == GRID * GRID
    assert out.shape[-2:] == placement.shape


# Before z-plane fusion, `stitch` wrapped the reader in `_FlatfieldReader` (correction ON by
# default) and only the plane-op refusal stopped the `flatfield` operator correcting twice.
# Lifting that refusal makes the combination reachable, hence this guard.


def _flatfield_test_op(profiles: dict):
    """What the shelved `flatfield` operator declared, test-owned: a plane-op stamped
    corrects_illumination=True with per-channel binding via for_channel."""
    from squidxplorer.projection import plane_op

    def _op_for(profile):
        def _f(plane):
            return correct_flatfield(plane, profile)
        op = plane_op(_f)
        op.corrects_illumination = True
        return op

    base = _op_for(next(iter(profiles.values())))
    base.for_channel = lambda path, channel: _op_for(profiles[str(channel)])
    return base


def test_the_flatfield_correction_is_not_idempotent(master):
    """The premise of the guard, measured rather than assumed: correcting twice divides the dim
    corners by the gain field's square, it is not a no-op."""
    profile = _vignette_profile()
    plane = master[:TILE, :TILE]
    once = correct_flatfield(plane, profile)
    twice = correct_flatfield(once, profile)
    differing = float((once != twice).mean())
    assert differing > 0.5, (
        f"only {differing:.1%} of pixels changed on a second correction — if this is ever ~0 the "
        "guard below is unnecessary, but that would be a change in correct_flatfield, not a fact "
        "about flat-fielding")
    assert int(np.abs(twice.astype(np.int32) - once.astype(np.int32)).max()) > 0


def test_stitching_a_correcting_operator_with_read_path_correction_refuses(master):
    """Keyed on the corrects_illumination DECLARATION, not any operator name: exactly one of
    the read path and the operator may flat-field per pass."""
    profile = _vignette_profile()
    name = "ff_test_guard"
    add_operator(name, _flatfield_test_op({c: profile for c in CHANNELS}))
    try:
        reader = _ZReader(master)
        with pytest.raises(ValueError, match="not idempotent"):
            stitch_region(reader, "A1", list(range(GRID * GRID)), z_operator=name,
                          register=False)           # correct_illumination defaults to ON
        with pytest.raises(ValueError, match="not idempotent"):
            stitch_region(reader, "A1", list(range(GRID * GRID)), z_operator=name,
                          register=False, correct_illumination=True)
    finally:
        _OPERATORS.pop(name, None)


def test_each_single_correction_path_stays_available_and_corrects_exactly_once(master):
    """The guard must refuse the double, never the single. Fusing raw data then correcting the
    result is NOT equivalent to correcting per-tile before the blend, so what's asserted is that
    the two single-correction spellings agree with each other and both differ from uncorrected."""
    profile = _vignette_profile()
    fovs = list(range(GRID * GRID))

    name = "ff_test_single"
    add_operator(name, _flatfield_test_op({c: profile for c in CHANNELS}))
    try:
        by_operator = np.asarray(stitch_region(_ZReader(master), "A1", fovs, z_operator=name,
                                               register=False, correct_illumination=False))
        by_read_path = np.asarray(stitch_region(
            _ZReader(master), "A1", fovs, z_operator=_PASSTHROUGH, register=False,
            correct_illumination=True, flatfield={c: profile for c in CHANNELS}))
    finally:
        _OPERATORS.pop(name, None)

    uncorrected = np.asarray(stitch_region(_ZReader(master), "A1", fovs, z_operator=_PASSTHROUGH,
                                           register=False, correct_illumination=False))

    assert by_operator.shape == by_read_path.shape == uncorrected.shape
    assert np.array_equal(by_operator, by_read_path), (
        "the two single-correction spellings disagree — correcting in the read path and correcting "
        "in the operator must apply the SAME correction to the same tiles")
    assert not np.array_equal(by_operator, uncorrected), (
        "correction had no effect at all; the test proves nothing about double-applying")


def test_stitching_a_label_operator_refuses_rather_than_averaging_object_ids(master):
    """Feathered blending of integer object ids produces objects that do not exist. The guard
    reads produces=="labels" off the record — a test-registered labels op, since the built-in
    segmenters were shelved 2026-08-24 (the labels vocabulary itself stays plugin surface)."""
    from squidxplorer.projection import labels_op, plane_op

    name = "labels_zp_test"
    add_operator(name, labels_op(plane_op(lambda p: (p > p.mean()).astype(p.dtype))))
    try:
        reader = _ZReader(master)
        with pytest.raises(ValueError, match="label"):
            stitch_region(reader, "A1", list(range(GRID * GRID)), z_operator=name,
                          register=False, correct_illumination=False)
    finally:
        _OPERATORS.pop(name, None)


def test_project_well_z_selects_exactly_one_acquisition_plane(master):
    from squidxplorer.projection import project_well

    reader = _ZReader(master)
    whole = project_well(reader, "A1", 0, reduce=plane_op(_passthrough), consumes=PLANE_OP)
    assert whole.shape[2] == N_Z
    for z in range(N_Z):
        one = project_well(reader, "A1", 0, reduce=plane_op(_passthrough), consumes=PLANE_OP, z_level=z)
        assert one.shape[2] == 1
        assert np.array_equal(one[:, :, 0], whole[:, :, z]), (
            f"z={z} did not select acquisition plane {z}")


def test_project_well_refuses_a_single_plane_for_a_z_reducer(master):
    """"The MIP of one plane" would be a different result wearing the same operator name."""
    from squidxplorer.projection import project, project_well

    reader = _ZReader(master)
    with pytest.raises(ValueError, match="only meaningful for a plane-op"):
        project_well(reader, "A1", 0, reduce=project, z_level=1)
