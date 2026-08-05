"""IMA-277 per-plane 3-D fusion: every z plane fused, with ONE solved geometry.

What this file is for, stated as the failures it would catch:

  * "only plane 0 came out"          — the old behaviour, which was refused rather than shipped;
  * "plane 0 was broadcast over z"   — a shape-only assertion passes this;
  * "each plane solved its own registration" — the stack shears with depth and nothing says so;
  * "the flat-field got applied twice" — 88.6% of pixels wrong, silently (measured);
  * "the z loop held every plane's tiles" — 9.4 GB on the real set, i.e. it does not run.

The fixture is tests/test_stitch.py's synthetic mosaic with a Z AXIS bolted on: each z plane is
the master texture plus a per-plane constant, so a plane is identifiable by its pixel values
alone and a swapped/duplicated/missing plane is a numeric failure, not a shape one.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tilefusion", reason="tilefusion (maragall/stitcher) not installed: the stitch "
                                         "adapter is UNTESTED here, which is not the same as passing")

from squidmip._engine import _PROJECTORS, add_projector
from squidmip._flatfield import (
    FlatfieldProfile,
    clear_profile,
    correct_flatfield,
    set_profile,
)
from squidmip._stitch import stitch_region
from squidmip.projection import PLANE_OP, plane_op

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

    def read(self, region, fov, channel, z=0, t=0):
        self.read_z.append(int(z))
        base = super().read(region, fov, channel, 0, t).astype(np.int32)
        return np.clip(base + int(z) * _PLANE_OFFSET, 0, 65535).astype(np.uint16)


@pytest.fixture(scope="module")
def master():
    return _master()


# A do-nothing plane-op. Two jobs: it exercises the per-plane path while PRESERVING the pixels
# (bgsub, the obvious real plane-op, subtracts this fixture's texture away to zeros, and a test
# comparing all-zero planes proves nothing), and it lets a stitch run with the read-path
# correction as the only correction.
def _passthrough(p):
    return p


_PASSTHROUGH = "zplanes_passthrough"


@pytest.fixture(autouse=True, scope="module")
def _register_passthrough():
    """Registered for this module only. `add_projector` writes to a PROCESS-GLOBAL table, and
    tests/test_operator_integration.py asserts that table's exact contents — a module-import-time
    registration here silently fails that test from another file, which is the kind of coupling
    a fixture with a teardown exists to prevent."""
    add_projector(_PASSTHROUGH, plane_op(_passthrough))
    try:
        yield
    finally:
        _PROJECTORS.pop(_PASSTHROUGH, None)


def _identity_profile(shape=(TILE, TILE)) -> FlatfieldProfile:
    return FlatfieldProfile(np.ones(shape, dtype=np.float32), None)


def _vignette_profile(shape=(TILE, TILE)) -> FlatfieldProfile:
    """A real-ish ~10%-deep radial vignette. Deep enough that a double-apply is visible."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    r2 = ((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2
    field = 1.0 - 0.10 * r2 / 2.0
    field /= field.mean()          # FlatfieldProfile requires mean 1.0 (a gain, not a brightness)
    return FlatfieldProfile(field.astype(np.float32), None)


# ---------------------------------------------------------------------------------------
# 1. every plane lands, and each is its own plane
# ---------------------------------------------------------------------------------------


def test_a_plane_op_fuses_every_z_plane(master):
    """The headline. A plane-op used to raise NotImplementedError here."""
    reader = _ZReader(master)
    out = stitch_region(reader, "A1", list(range(GRID * GRID)), projector=_PASSTHROUGH,
                        register=False, correct_illumination=False)

    assert out.ndim == 5
    assert out.shape[2] == N_Z, f"expected {N_Z} fused z planes, got {out.shape[2]}"
    # Every plane DISTINCT: rules out "plane 0 written N_Z times".
    signatures = {out[0, 0, z].tobytes() for z in range(N_Z)}
    assert len(signatures) == N_Z, "fused planes are not distinct — z was broadcast, not fused"
    # ...and in the right ORDER: the fixture lifts each plane by a known constant, so the plane
    # means must be monotonically increasing. A shuffled z axis passes the distinctness check.
    means = [float(out[0, 0, z].mean()) for z in range(N_Z)]
    assert means == sorted(means), f"z planes came out in the wrong order: {means}"


def test_a_z_reducer_is_unchanged_and_still_collapses_z(master):
    """mip must be byte-for-byte what it was: one plane out, whatever the stack depth."""
    reader = _ZReader(master)
    out = stitch_region(reader, "A1", list(range(GRID * GRID)), projector="mip",
                        register=False, correct_illumination=False)
    assert out.shape[2] == 1
    assert not isinstance(out.base, np.memmap), "a single fused plane must not spill to disk"


def test_the_z_loop_reads_one_plane_at_a_time(master):
    """Streaming, not stacking. The reason this was never done is memory (~9.4 GB on the real set).

    Asserted on the READ ORDER rather than on RSS, because RSS is noisy and the property that
    actually bounds memory is structural: all of plane k's reads happen before any of plane k+1's.
    A z-INNER loop (project the whole stack per FOV, then fuse) interleaves them and fails this.
    """
    reader = _ZReader(master)
    stitch_region(reader, "A1", list(range(GRID * GRID)), projector="bgsub",
                  register=False, correct_illumination=False)

    # Drop to the sequence of DISTINCT z values in read order: a z-outer loop visits each z once.
    runs = [z for i, z in enumerate(reader.read_z) if i == 0 or z != reader.read_z[i - 1]]
    assert runs == list(range(N_Z)), (
        f"z was not read outermost — read-order z runs were {runs}, expected one contiguous run "
        f"per plane. A z-inner loop holds every plane's tiles at once (~9.4 GB on the real set)."
    )


# ---------------------------------------------------------------------------------------
# 2. ONE geometry for every plane — "pixel identical in all planes"
# ---------------------------------------------------------------------------------------


def test_every_plane_is_fused_with_the_same_solved_offsets(master):
    """The correctness requirement, not an optimisation.

    Registration is geometry: it is solved once, on one raw plane at one timepoint. If each z
    re-solved, the planes would disagree by their residuals and the stack would shear with depth.

    Proven two ways, because either alone is weak: (a) the solve runs ONCE (spy on the solver),
    and (b) each fused plane equals — to the pixel — a single-plane fuse of the same data, which
    is what "pixel identical in all planes" cashes out to.
    """
    error = {1: (0.0, 5.0), 2: (4.0, 0.0)}      # a real, registrable stage error to solve away
    reader = _ZReader(_master(), error_px=error)

    calls = []
    import squidmip._stitch as stitch_mod
    real_solve = stitch_mod.solve_offsets_px

    def _spy(*a, **kw):
        calls.append(1)
        return real_solve(*a, **kw)

    stitch_mod.solve_offsets_px = _spy
    try:
        out = stitch_region(reader, "A1", list(range(GRID * GRID)), projector=_PASSTHROUGH,
                            correct_illumination=False, correct_distortion=False)
    finally:
        stitch_mod.solve_offsets_px = real_solve

    assert len(calls) == 1, (
        f"registration ran {len(calls)} times for {N_Z} z planes; it must run ONCE — a per-plane "
        "solve makes the planes disagree by their residuals and the stack shears with depth")

    # (b) The fixture's planes differ ONLY by the constant _PLANE_OFFSET. Fuse them with one
    # geometry and the fused planes must differ only by that same constant, everywhere — including
    # across the seams, where the feather weights are what a re-solve would move. A per-plane solve
    # shifts the tiles relative to each other, so the difference stops being constant exactly at
    # the seams, which is where a stack shears.
    #
    # Tolerance is 1 count and it is arithmetic, not slack: fuse_plane blends in float32 and
    # _cast_like rounds back to uint16, so a value landing on .5 can round either way. A moved seam
    # is worth hundreds of counts on this texture (its dynamic range is ~40000), so 1 does not hide
    # one.
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
    """The provenance attached to the pixels describes ONE solve, not N_Z of them."""
    reader = _ZReader(master)
    out = stitch_region(reader, "A1", list(range(GRID * GRID)), projector=_PASSTHROUGH,
                        correct_illumination=False, correct_distortion=False)
    placement = out.placement
    assert placement.reg_z == N_Z // 2, "geometry must be solved on the acquisition's middle plane"
    assert placement.reg_t == 0
    assert len(placement.offsets_px) == GRID * GRID
    assert len(placement.origins_px) == GRID * GRID
    assert out.shape[-2:] == placement.shape


# ---------------------------------------------------------------------------------------
# 3. THE FLAT-FIELD DOUBLE-APPLY GUARD
# ---------------------------------------------------------------------------------------
#
# This is the one that had to be built in the same change as the refusal it unblocks. Before
# IMA-277, `stitch` wrapped the reader in `_FlatfieldReader` (correction ON by default) and the
# ONLY thing stopping the `flatfield` projector from correcting the same pixels a second time was
# the plane-op refusal. Lifting that refusal makes the combination reachable.


def test_the_flatfield_correction_is_not_idempotent(master):
    """The premise of the guard, measured rather than assumed.

    If correcting twice were a no-op, no guard would be needed. It is not: dividing by the gain
    field twice divides the dim corners by its square.
    """
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


def test_stitching_the_flatfield_operator_with_read_path_correction_refuses(master):
    """The guard. Both corrections on = refuse, by name of both escapes."""
    reader = _ZReader(master)
    set_profile(_vignette_profile())
    try:
        with pytest.raises(ValueError, match="not idempotent"):
            stitch_region(reader, "A1", list(range(GRID * GRID)), projector="flatfield",
                          register=False)           # correct_illumination defaults to ON
        with pytest.raises(ValueError, match="not idempotent"):
            stitch_region(reader, "A1", list(range(GRID * GRID)), projector="flatfield",
                          register=False, correct_illumination=True)
    finally:
        clear_profile()


def test_the_guard_catches_an_operator_registered_under_another_name(master):
    """Keyed on the DECLARATION, not on the string "flatfield".

    `flatfield_op(profile)` hands out correcting operators under any name the caller picks. A
    `== "flatfield"` guard — which this package does have, in two other places — would miss every
    one of them, and the double-apply would be back with no test failing.
    """
    from squidmip._flatfield import flatfield_op

    name = "flatfield_under_another_name"
    add_projector(name, flatfield_op(_vignette_profile()))
    try:
        reader = _ZReader(master)
        with pytest.raises(ValueError, match="not idempotent"):
            stitch_region(reader, "A1", list(range(GRID * GRID)), projector=name, register=False)
    finally:
        _PROJECTORS.pop(name, None)


def test_each_single_correction_path_stays_available_and_corrects_exactly_once(master):
    """The guard must refuse the DOUBLE, never the single. Both spellings, checked on pixels.

    Reference: fuse the raw data with no correction anywhere, and separately correct the fused
    result is NOT the same thing (correction is per-tile, before the blend) — so the assertion is
    the one that matters operationally: the two single-correction spellings agree with each other,
    and both differ from the uncorrected fuse.
    """
    profile = _vignette_profile()
    fovs = list(range(GRID * GRID))

    set_profile(profile)
    try:
        # (a) correction in the OPERATOR, read path off.
        by_operator = np.asarray(stitch_region(_ZReader(master), "A1", fovs, projector="flatfield",
                                               register=False, correct_illumination=False))
        # (b) correction in the READ PATH, operator is a passthrough plane-op.
        by_read_path = np.asarray(stitch_region(
            _ZReader(master), "A1", fovs, projector=_PASSTHROUGH, register=False,
            correct_illumination=True, flatfield={c: profile for c in CHANNELS}))
    finally:
        clear_profile()

    uncorrected = np.asarray(stitch_region(_ZReader(master), "A1", fovs, projector=_PASSTHROUGH,
                                           register=False, correct_illumination=False))

    assert by_operator.shape == by_read_path.shape == uncorrected.shape
    assert np.array_equal(by_operator, by_read_path), (
        "the two single-correction spellings disagree — correcting in the read path and correcting "
        "in the operator must apply the SAME correction to the same tiles")
    assert not np.array_equal(by_operator, uncorrected), (
        "correction had no effect at all; the test proves nothing about double-applying")


# ---------------------------------------------------------------------------------------
# 4. what still cannot be stitched, and says so
# ---------------------------------------------------------------------------------------


def test_stitching_a_label_operator_refuses_rather_than_averaging_object_ids(master):
    """Feathered blending of integer object ids produces objects that do not exist."""
    labels = [n for n, op in _PROJECTORS.items() if op.produces == "labels"]
    assert labels, "expected at least one labels operator (spot/cellpose) to be registered"
    reader = _ZReader(master)
    for name in labels:
        with pytest.raises(ValueError, match="label"):
            stitch_region(reader, "A1", list(range(GRID * GRID)), projector=name,
                          register=False, correct_illumination=False)


# ---------------------------------------------------------------------------------------
# 5. project_well's single-plane selector, the primitive the z loop is built on
# ---------------------------------------------------------------------------------------


def test_project_well_z_selects_exactly_one_acquisition_plane(master):
    from squidmip.projection import project_well

    reader = _ZReader(master)
    whole = project_well(reader, "A1", 0, reduce=plane_op(_passthrough), consumes=PLANE_OP)
    assert whole.shape[2] == N_Z
    for z in range(N_Z):
        one = project_well(reader, "A1", 0, reduce=plane_op(_passthrough), consumes=PLANE_OP, z=z)
        assert one.shape[2] == 1
        assert np.array_equal(one[:, :, 0], whole[:, :, z]), (
            f"z={z} did not select acquisition plane {z}")


def test_project_well_refuses_a_single_plane_for_a_z_reducer(master):
    """"The MIP of one plane" would be a different result wearing the same operator name."""
    from squidmip.projection import project, project_well

    reader = _ZReader(master)
    with pytest.raises(ValueError, match="only meaningful for a plane-op"):
        project_well(reader, "A1", 0, reduce=project, z=1)
