"""IMA-222 stitch operator: registration recovers a KNOWN error, and the plate generator
mirrors ``project_plate``'s contract.

The fixture is a synthetic mosaic cut from one master image, so there is **ground truth**:
the correct fused result is a crop of the master. That turns every claim into a number —
"the solve recovered the 6 px I injected", "the stitched mosaic is closer to truth than the
coordinate-placed one" — instead of "a picture appeared", which is what a shape-only test
asserts and is exactly how a stitcher ships broken.

Geometry mirrors the real 10x tissue acquisition in the one way that matters: tiles overlap
by a real, registrable fraction (64 of 256 px here; ~208 of 2084 px there).
"""

from __future__ import annotations

import numpy as np
import pytest

# tilefusion (maragall/stitcher) is the library this module ADAPTS, and it is deliberately not a
# dependency: pyproject says so in as many words ("No tilefusion dependency ... importing
# tilefusion runs its heavy __init__ (numba/GPU/basicpy)"), and only the ~40-line store-config
# wrapper is vendored. `squidmip._stitch` therefore imports it lazily, so this file COLLECTS
# without it and only fails mid-test with ModuleNotFoundError -- which is what it did on every CI
# runner, as hard failures for a package CI was never asked to install.
#
# Skip rather than install it in CI: adding a git dependency to the build is exactly what was just
# removed for ndviewer_light, and numba/basicpy would dominate the job. The cost is real and is
# stated here rather than hidden: THE STITCH ADAPTER IS NOT COVERED IN CI, only on a machine that
# has tilefusion. A skip is not a pass. To gate this seam, install tilefusion deliberately.
pytest.importorskip("tilefusion", reason="tilefusion (maragall/stitcher) not installed: the stitch "
                                         "adapter is UNTESTED here, which is not the same as passing")

from squidmip._stitch import (
    _REGION_OPERATORS,
    _mosaic_geometry,
    _positions_yx_um,
    add_region_operator,
    available_region_operators,
    solve_offsets_px,
    stitch_plate,
    stitch_region,
)

TILE = 256
STEP = 192            # -> 64 px overlap, ~25%
GRID = 2              # 2x2 = 4 FOVs
PIXEL_UM = 1.0        # 1 um/px keeps micrometres and pixels numerically identical, so a
#                       sign/scale slip shows up as a wrong number rather than hiding in a
#                       unit conversion.
CHANNELS = ["Fluorescence_405_nm_Ex", "Fluorescence_488_nm_Ex"]


def _master(seed: int = 0) -> np.ndarray:
    """A smooth, high-contrast random texture — registrable, unlike white noise or a ramp.

    Phase correlation needs broadband structure that is *locally unique*. White noise
    aliases under any sub-pixel shift; a gradient has no unique peak. A low-pass filtered
    random field has both, which is why it is the standard synthetic registration fixture.

    "Lightly" filtered is load-bearing, and was found by measurement rather than taste:
    tilefusion correlates with ``normalization="phase"``, which whitens the spectrum, so an
    over-smoothed field leaves only high-frequency noise to correlate on and the lock
    collapses. Measured on this fixture: a 3x 7-px blur scores NCC 0.44 and recovers 1.3 px
    of a known 6 px shift; a single 5-px blur scores 0.9999 and recovers it exactly.
    """
    rng = np.random.default_rng(seed)
    n = (GRID - 1) * STEP + TILE
    field = rng.normal(size=(n, n))
    # Cheap separable box blur; avoids a scipy dependency in the test.
    field = np.apply_along_axis(np.convolve, 0, field, np.ones(5) / 5, mode="same")
    field = np.apply_along_axis(np.convolve, 1, field, np.ones(5) / 5, mode="same")
    field -= field.min()
    field /= field.max()
    return (field * 40000 + 2000).astype(np.uint16)


class _FakeReader:
    """Minimal ``SquidReader`` duck-type: ``.metadata`` + ``.read``.

    Deliberately not the real reader — this test is about the stitch math, and a synthetic
    on-disk acquisition would only add TIFF I/O between the assertion and the thing asserted.
    """

    def __init__(self, master: np.ndarray, error_px: dict[int, tuple[float, float]] | None = None,
                 regions=("A1",), step: int = STEP, n_t: int = 1, good_t: int = 0):
        self._master = master
        self._step = step
        # Only ONE timepoint carries registrable structure; the others are noise. That makes
        # "the solve actually read the timepoint it was told to" an assertion about NUMBERS
        # rather than about provenance -- a mutation that hardcodes t=0 survives any test that
        # only checks what the Placement claims.
        self._good_t = good_t
        self._true = [
            ((i // GRID) * step, (i % GRID) * step) for i in range(GRID * GRID)
        ]  # (y, x) top-left of each tile in the master, in pixels
        err = error_px or {}
        # Reported stage positions carry the injected error; (x_um, y_um) as the reader emits.
        positions = {}
        for region in regions:
            for i, (y, x) in enumerate(self._true):
                dy, dx = err.get(i, (0.0, 0.0))
                positions[(region, i)] = ((x + dx) * PIXEL_UM, (y + dy) * PIXEL_UM)
        self.metadata = {
            "regions": list(regions),
            "fovs_per_region": {r: list(range(GRID * GRID)) for r in regions},
            "fov_positions_um": positions,
            "channels": [{"name": c} for c in CHANNELS],
            "z_levels": [0],
            "n_z": 1,
            "n_t": n_t,
            "frame_shape": (TILE, TILE),
            "dtype": np.dtype(np.uint16),
            "pixel_size_um": PIXEL_UM,
        }
        self.reads = 0

    def read(self, region, fov, channel, z=0, t=0):
        self.reads += 1
        y, x = self._true[fov]
        if t != self._good_t:
            # Unregistrable by construction: white noise aliases under any sub-pixel shift.
            return np.random.default_rng(1000 + fov).integers(
                0, 65535, size=(TILE, TILE), dtype=np.uint16)
        tile = self._master[y : y + TILE, x : x + TILE]
        # Second channel is a scaled copy: distinct data, same geometry, so a channel mix-up
        # in the fuse is visible while registration stays well-posed on channel 0.
        return tile if channel == CHANNELS[0] else (tile // 2 + 500).astype(np.uint16)


@pytest.fixture(scope="module")
def master():
    return _master()


# ---------------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------------


def test_positions_are_swapped_to_yx(master):
    """The reader stores (x_um, y_um); tilefusion is (y, x). The swap must happen once, here."""
    reader = _FakeReader(master)
    pos = _positions_yx_um(reader.metadata, "A1", [0, 1, 2])
    # fov 1 is one step to the RIGHT (+x, same y); fov 2 is one step DOWN (+y, same x).
    assert pos[0] == (0.0, 0.0)
    assert pos[1] == (0.0, float(STEP))
    assert pos[2] == (float(STEP), 0.0)


def test_missing_position_refuses_rather_than_stacking(master):
    reader = _FakeReader(master)
    del reader.metadata["fov_positions_um"][("A1", 2)]
    with pytest.raises(KeyError, match="no stage position"):
        _positions_yx_um(reader.metadata, "A1", [0, 1, 2, 3])


def test_mosaic_geometry_accounts_for_overlap(master):
    """Extent is the bounding box of placed tiles, NOT n_tiles x tile (which ignores overlap)."""
    (h, w), origins = _mosaic_geometry(
        [(0.0, 0.0), (0.0, STEP), (STEP, 0.0), (STEP, STEP)], (1.0, 1.0), (TILE, TILE)
    )
    assert (h, w) == (STEP + TILE, STEP + TILE)
    assert origins[0] == (0.0, 0.0) and origins[3] == (float(STEP), float(STEP))


def test_mosaic_origins_stay_fractional():
    """Sub-pixel origins must survive: truncating them re-introduces the misalignment the
    registration just removed."""
    (_h, _w), origins = _mosaic_geometry([(0.0, 0.0), (0.0, 10.4)], (1.0, 1.0), (8, 8))
    assert origins[1][1] == pytest.approx(10.4)


# ---------------------------------------------------------------------------------------
# registration: the number, not the picture
# ---------------------------------------------------------------------------------------


def test_solve_recovers_injected_stage_error(master):
    """Inject a known stage error on one tile; the solve must cancel it to sub-pixel."""
    err = {3: (6.0, -4.0)}   # tile 3's reported position is wrong by (dy, dx)
    reader = _FakeReader(master, error_px=err)
    fovs = list(range(GRID * GRID))
    tiles = np.stack(
        [np.stack([reader.read("A1", f, CHANNELS[0])]) for f in fovs]
    )  # (n, C=1, Y, X)
    positions = _positions_yx_um(reader.metadata, "A1", fovs)

    offsets = solve_offsets_px(tiles, positions, (1.0, 1.0), (TILE, TILE), max_workers=2)

    corrected = np.asarray(positions) + offsets
    # The corrected layout must reproduce the TRUE grid up to a global translation (the solve
    # is gauge-free: it anchors tile 0, so only relative geometry is meaningful).
    truth = np.array([[(i // GRID) * STEP, (i % GRID) * STEP] for i in range(4)], float)
    residual = (corrected - corrected[0]) - (truth - truth[0])
    assert np.abs(residual).max() < 0.5, f"residual {residual}"


def test_no_overlap_degrades_to_stage_positions(master):
    """A sparse acquisition (no registrable overlap) is not an error — it falls back."""
    reader = _FakeReader(master, step=TILE * 4)
    fovs = list(range(GRID * GRID))
    tiles = np.zeros((len(fovs), 1, TILE, TILE), np.uint16)
    positions = _positions_yx_um(reader.metadata, "A1", fovs)
    offsets = solve_offsets_px(tiles, positions, (1.0, 1.0), (TILE, TILE), max_workers=2)
    assert offsets.shape == (4, 2)
    assert np.array_equal(offsets, np.zeros((4, 2)))


# ---------------------------------------------------------------------------------------
# fusion: stitched must be measurably closer to ground truth than coordinate placement
# ---------------------------------------------------------------------------------------


def _rmse_vs_truth(fused: np.ndarray, master: np.ndarray) -> float:
    """RMSE of the fused channel-0 mosaic against the master crop it should reproduce.

    Compared on the INTERIOR only: the mosaic border is a single-tile feather ramp whose
    normalized weight is fine but whose edge pixels are dominated by one tile, so including
    them measures the ramp rather than the seam.
    """
    a = fused[0, 0, 0].astype(np.float64)
    # Crop both to their common extent: an UNregistered mosaic is larger than the truth
    # (the stage error inflates the bounding box), which is itself a symptom, not a reason
    # to skip the comparison. Both share the top-left origin (tile 0 anchors there).
    h, w = min(a.shape[0], master.shape[0]), min(a.shape[1], master.shape[1])
    a = a[:h, :w]
    b = master[:h, :w].astype(np.float64)
    m = 40
    return float(np.sqrt(np.mean((a[m:-m, m:-m] - b[m:-m, m:-m]) ** 2)))


@pytest.fixture(scope="module")
def fused_pair(master):
    """Both operators over the same errored acquisition — one fuse per mode, reused."""
    reader = _FakeReader(master, error_px={3: (6.0, -4.0)})
    fovs = list(range(GRID * GRID))
    kw = dict(channels=[0], blend_px=24, block_px=512, max_workers=2)
    stitched = stitch_region(reader, "A1", fovs, register=True, **kw)
    placed = stitch_region(reader, "A1", fovs, register=False, **kw)
    return stitched, placed


def test_stitch_region_shape_and_dtype(fused_pair, master):
    stitched, _ = fused_pair
    assert stitched.ndim == 5 and stitched.shape[:3] == (1, 1, 1)   # (T, C, 1, Y, X)
    assert stitched.dtype == np.uint16
    assert stitched.shape[3] >= STEP + TILE - 8                     # mosaic, not one tile


def test_the_fused_write_rounds_to_the_dtype_instead_of_truncating(master, monkeypatch):
    """The blend accumulates in float32, ``out`` is the acquisition dtype (uint16 here).

    A plain slice assignment truncates toward zero, which is a half-count systematic dimming of
    every pixel of the mosaic. Every other operator in this codebase routes its write through
    ``_cast_like`` for exactly that reason (``_background``/``_decon``/``_flatfield`` all say so
    in as many words); stitch was the one that did not, and now shares the same helper.

    Pinned on NUMBERS, not on which function was called: the fuse kernel is replaced by one that
    hands ``write_block`` a block of known fractional values, and the assertion is on what lands
    in the returned array. 11.5 and 12.7 are the discriminating ones -- truncation gives 11 and
    12, rounding gives 12 and 13. 10.5 -> 10 is asserted too, and is not a truncation: it is
    ``np.rint``'s half-to-EVEN, which is what the shared helper does and therefore what stitch
    must do as well.
    """
    import tilefusion.fusion as tf_fusion

    fractions = np.array([10.5, 11.5, 12.7, 13.2], dtype=np.float32)

    def _fake_fuse_plane(*, write_block, padded_shape, channels, **_kw):
        h, w = padded_shape
        block = np.zeros((channels, h, w), dtype=np.float32)
        block[:, 0, : fractions.size] = fractions
        write_block(0, h, 0, w, block)

    monkeypatch.setattr(tf_fusion, "fuse_plane", _fake_fuse_plane)

    out = stitch_region(_FakeReader(master), "A1", [0, 1], channels=[0], register=False,
                        blend_px=24, block_px=512, max_workers=1)

    assert out.dtype == np.uint16
    assert list(out[0, 0, 0, 0, : fractions.size]) == [10, 12, 13, 13]


def test_stitching_beats_coordinate_placement(fused_pair, master):
    """The load-bearing assertion: registration must measurably reduce error vs ground truth."""
    stitched, placed = fused_pair
    e_stitched = _rmse_vs_truth(stitched, master)
    e_placed = _rmse_vs_truth(placed, master)
    assert e_stitched < e_placed * 0.5, f"stitched {e_stitched:.1f} vs placed {e_placed:.1f}"


def test_all_channels_share_one_geometry(master):
    """Channels must be placed by the SAME solve — independent per-channel geometry would
    make the channels of one well stop overlaying."""
    reader = _FakeReader(master, error_px={3: (6.0, -4.0)})
    out = stitch_region(reader, "A1", list(range(4)), blend_px=24, block_px=512, max_workers=2)
    assert out.shape[1] == len(CHANNELS)
    c0, c1 = out[0, 0, 0].astype(np.float64), out[0, 1, 0].astype(np.float64)
    # Channel 1 is channel 0 // 2 + 500 by construction, so if geometry matched, the two
    # mosaics correlate near-perfectly. A per-channel geometry slip destroys that.
    inner = (slice(40, -40), slice(40, -40))
    assert np.corrcoef(c0[inner].ravel(), c1[inner].ravel())[0, 1] > 0.99


def test_channel_selection_is_honoured(master):
    reader = _FakeReader(master)
    out = stitch_region(reader, "A1", [0, 1], channels=[1], blend_px=24, block_px=512,
                        register=False, max_workers=2)
    assert out.shape[1] == 1


def test_bad_channel_index_is_named(master):
    reader = _FakeReader(master)
    with pytest.raises(ValueError, match="out of range"):
        stitch_region(reader, "A1", [0], channels=[7])


def test_empty_fovs_refused(master):
    with pytest.raises(ValueError, match="no FOVs"):
        stitch_region(_FakeReader(master), "A1", [])


def test_missing_pixel_size_refused(master):
    reader = _FakeReader(master)
    reader.metadata["pixel_size_um"] = None
    with pytest.raises(ValueError, match="pixel_size_um is required"):
        stitch_region(reader, "A1", [0, 1])


# ---------------------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------------------


def test_default_operators_present():
    assert available_region_operators() == ["coordinate", "stitch"]


def test_add_and_resolve_region_operator(master):
    name = "_test_op"
    _REGION_OPERATORS.pop(name, None)
    try:
        add_region_operator(name, lambda r, reg, fovs, **kw: np.zeros((1, 1, 1, 2, 2), np.uint16))
        assert name in available_region_operators()
        out = list(stitch_plate(_FakeReader(master), operator=name))
        assert [r for r, _f, _i in out] == ["A1"]
    finally:
        _REGION_OPERATORS.pop(name, None)


def test_duplicate_operator_refused():
    with pytest.raises(ValueError, match="already defined"):
        add_region_operator("stitch", lambda *a, **k: None)


@pytest.mark.parametrize("bad", ["", None])
def test_invalid_operator_registration(bad):
    with pytest.raises(ValueError):
        add_region_operator(bad or "", bad)


def test_unknown_operator_names_the_alternatives(master):
    with pytest.raises(KeyError, match="unknown region operator"):
        list(stitch_plate(_FakeReader(master), operator="nope"))


# ---------------------------------------------------------------------------------------
# stitch_plate: the project_plate contract, mirrored
# ---------------------------------------------------------------------------------------


def _fast_plate(reader, **kw):
    """stitch_plate with the cheap operator settings the contract tests need."""
    kw.setdefault("channels", [0])
    kw.setdefault("blend_px", 24)
    kw.setdefault("block_px", 512)
    kw.setdefault("max_workers", 2)
    kw.setdefault("register", False)
    return stitch_plate(reader, **kw)


def test_one_result_per_region_anchored_at_first_fov(master):
    """A stitched well yields ONE array, not one per FOV — the contract difference, asserted."""
    reader = _FakeReader(master, regions=("A1", "A2"))
    out = list(_fast_plate(reader))
    assert sorted(r for r, _f, _i in out) == ["A1", "A2"]
    assert {f for _r, f, _i in out} == {0}                    # anchor fov = fovs[0]
    assert all(img.ndim == 5 for _r, _f, img in out)


def test_regions_subset_is_honoured(master):
    reader = _FakeReader(master, regions=("A1", "A2"))
    out = list(_fast_plate(reader, regions=["A2"]))
    assert [r for r, _f, _i in out] == ["A2"]


def test_regions_subset_ignores_unknown_and_dedups(master):
    reader = _FakeReader(master, regions=("A1", "A2"))
    out = list(_fast_plate(reader, regions=["A2", "A2", "ZZ"]))
    assert [r for r, _f, _i in out] == ["A2"]


def test_workers_must_be_positive(master):
    with pytest.raises(ValueError, match="workers must be >= 1"):
        list(_fast_plate(_FakeReader(master), workers=0))


def test_failure_is_loud_by_default(master):
    def boom(reader, region, fovs, **kw):
        raise RuntimeError("corrupt plane")

    name = "_test_boom"
    _REGION_OPERATORS.pop(name, None)
    add_region_operator(name, boom)
    try:
        with pytest.raises(RuntimeError, match="corrupt plane"):
            list(stitch_plate(_FakeReader(master, regions=("A1", "A2")), operator=name))
    finally:
        _REGION_OPERATORS.pop(name, None)


def test_on_error_skips_the_well_and_keeps_going(master):
    """One corrupt well must not abort a plate when the caller opts in — project_plate's
    IMA-186 contract, same keyword, same signature."""
    def flaky(reader, region, fovs, **kw):
        if region == "A1":
            raise RuntimeError("corrupt plane")
        return np.zeros((1, 1, 1, 2, 2), np.uint16)

    name = "_test_flaky"
    _REGION_OPERATORS.pop(name, None)
    add_region_operator(name, flaky)
    seen = []
    try:
        out = list(
            stitch_plate(
                _FakeReader(master, regions=("A1", "A2")),
                operator=name,
                on_error=lambda r, f, e: seen.append((r, f, type(e).__name__)),
            )
        )
    finally:
        _REGION_OPERATORS.pop(name, None)
    assert [r for r, _f, _i in out] == ["A2"]
    assert seen == [("A1", 0, "RuntimeError")]


def test_window_is_bounded_by_workers(master):
    """Peak memory is workers x one mosaic, so at most `workers` operators may run at once."""
    import threading

    live = 0
    peak = 0
    lock = threading.Lock()
    gate = threading.Event()

    def counted(reader, region, fovs, **kw):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        gate.wait(0.05)
        with lock:
            live -= 1
        return np.zeros((1, 1, 1, 2, 2), np.uint16)

    name = "_test_counted"
    _REGION_OPERATORS.pop(name, None)
    add_region_operator(name, counted)
    try:
        reader = _FakeReader(master, regions=tuple(f"A{i}" for i in range(8)))
        assert len(list(stitch_plate(reader, operator=name, workers=2))) == 8
    finally:
        _REGION_OPERATORS.pop(name, None)
    assert peak <= 2, f"in-flight window ran to {peak}, expected <= 2"


def test_stitching_a_plane_op_refuses_instead_of_keeping_only_z0():
    """A plane-op must not be stitched until per-plane fusion exists (IMA-277).

    `stitch_region` fuses with z=1 by construction: `out` is allocated with a z extent of 1,
    write_block writes [t, :, 0, ...] and fuse_plane gets z_level=0. That is right for a
    z-reducer, whose project_well output is (T, C, 1, Y, X). For a plane-op the output is
    (T, C, Nz, Y, X), so the old `[:, channels, 0]` silently kept plane 0 and discarded the
    rest — on exported science data, on three of the six registered projectors.
    """
    import pytest
    from squidmip._stitch import _resolve_projector, stitch_region

    plane_ops = [n for n in ("bgsub", "decon", "flatfield")
                 if not _resolve_projector(n).consumes]
    assert plane_ops, "expected bgsub/decon/flatfield to be plane-ops (consumes == frozenset())"

    for name in plane_ops:
        with pytest.raises(NotImplementedError, match="plane-op"):
            stitch_region(_DummyReader(), "A1", [0, 1], projector=name, register=False)


class _DummyReader:
    """Refusal must happen before any pixel is read, so the reader is never touched."""

    metadata = {"regions": ["A1"], "channels": ["c0"], "fov_positions_um": {}}

    def __getattr__(self, name):  # pragma: no cover - must not be reached
        raise AssertionError(f"reader touched ({name}) before the plane-op guard refused")


# ---------------------------------------------------------------------------------------
# blunder rejection: the operator's two knobs (ported from maragall/stitcher's GUI)
# ---------------------------------------------------------------------------------------
#
# maragall/stitcher exposes "Outlier rel: N%" and "abs: N px" as the two controls over
# two_round_optimization's blunder rejection. They were module constants here, so the
# stitcher panel had nothing to bind to. These tests pin that the values actually REACH
# the solver -- a parameter that is accepted and then ignored is the exact defect shape
# this repo has shipped before (a test that read green while the function it called had
# grown a third return value).


def _spy_two_round(monkeypatch):
    """Record the args tilefusion's two_round_optimization is called with, and short-circuit
    it. Patched on the tilefusion module because _solve imports it at CALL time."""
    import tilefusion.optimization as opt

    seen = {}

    def _fake(edges, n_tiles, anchors, rel, abs_, flag):
        seen.update(rel=rel, abs=abs_, n_tiles=n_tiles, anchors=anchors)
        return np.zeros((n_tiles, 2), dtype=np.float64)

    monkeypatch.setattr(opt, "two_round_optimization", _fake)
    return seen


def _tiles_and_positions(master):
    reader = _FakeReader(master, error_px={3: (6.0, -4.0)})
    fovs = list(range(GRID * GRID))
    tiles = np.stack([np.stack([reader.read("A1", f, CHANNELS[0])]) for f in fovs])
    return tiles, _positions_yx_um(reader.metadata, "A1", fovs)


def test_solve_defaults_are_tilefusion_run_s_own_thresholds(master, monkeypatch):
    """Unset, the solve must behave EXACTLY as it did: TileFusion.run()'s 0.5 / 2.0."""
    from squidmip._stitch import _ABS_THRESH, _REL_THRESH

    seen = _spy_two_round(monkeypatch)
    tiles, positions = _tiles_and_positions(master)
    solve_offsets_px(tiles, positions, (1.0, 1.0), (TILE, TILE), max_workers=2)
    assert (seen["rel"], seen["abs"]) == (_REL_THRESH, _ABS_THRESH)


def test_solve_forwards_the_operator_s_thresholds(master, monkeypatch):
    seen = _spy_two_round(monkeypatch)
    tiles, positions = _tiles_and_positions(master)
    solve_offsets_px(tiles, positions, (1.0, 1.0), (TILE, TILE), max_workers=2,
                     rel_thresh=0.25, abs_thresh=7.5)
    assert (seen["rel"], seen["abs"]) == (0.25, 7.5)


def test_stitch_region_forwards_the_thresholds_all_the_way_down(master, monkeypatch):
    """The one that matters for the panel: the kwargs a user sets in the LEFT pane travel
    stitch_plate -> stitch_region -> solve_offsets_px -> two_round_optimization. Accepting
    them at the top and dropping them one layer down would leave the controls inert while
    looking like they worked."""
    seen = _spy_two_round(monkeypatch)
    reader = _FakeReader(master, error_px={3: (6.0, -4.0)})
    stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0], blend_px=24,
                  block_px=512, max_workers=2, rel_thresh=0.33, abs_thresh=9.0)
    assert (seen["rel"], seen["abs"]) == (0.33, 9.0)


def test_thresholds_must_be_positive(master):
    """A zero/negative threshold rejects every link or none; refuse by name rather than
    silently solving on an empty edge set and returning zeros that look like 'no error'."""
    tiles, positions = _tiles_and_positions(master)
    with pytest.raises(ValueError, match="rel_thresh"):
        solve_offsets_px(tiles, positions, (1.0, 1.0), (TILE, TILE), rel_thresh=0.0)
    with pytest.raises(ValueError, match="abs_thresh"):
        solve_offsets_px(tiles, positions, (1.0, 1.0), (TILE, TILE), abs_thresh=-1.0)


# ---------------------------------------------------------------------------------------
# write_plate: the SAVE path has to carry the operator's settings too
# ---------------------------------------------------------------------------------------
#
# The panel's controls reach the preview through stitch_plate(**operator_kwargs). Without
# the same seam on write_plate, "Run on the whole plate" would quietly use the pipeline
# defaults while the panel showed the user's settings -- a tuned registration thrown away
# at exactly the moment it is written to disk, with nothing said.


class _MetaOnlyReader:
    metadata = {"regions": ["A1"], "fovs_per_region": {"A1": [0]},
                "channels": [{"name": "c0"}], "pixel_size_um": 1.0,
                "frame_shape": (8, 8), "dtype": "uint16", "n_t": 1}


def test_write_plate_forwards_operator_kwargs_to_stitch_plate(monkeypatch):
    import squidmip._output as out_mod
    import squidmip._stitch as st

    seen = {}

    def _fake_stitch_plate(reader, **kw):
        seen.update(kw)
        return iter(())

    monkeypatch.setattr(st, "stitch_plate", _fake_stitch_plate)
    monkeypatch.setattr(out_mod, "write_from_stream", lambda *a, **k: {"written": 0})
    out_mod.write_plate(_MetaOnlyReader(), "/tmp/does-not-matter", projector="stitch",
                        operator_kwargs={"blend_px": 64, "rel_thresh": 0.25, "register": False})
    assert seen["blend_px"] == 64
    assert seen["rel_thresh"] == 0.25
    assert seen["register"] is False


def test_write_plate_refuses_operator_kwargs_an_operator_does_not_declare():
    """Accepting a parameter and dropping it is the silent failure this seam exists to avoid.

    The REASON for the refusal moved on 2026-08-03 and the test says so. It used to be a rule
    about WHICH TABLE the operator is in: "project_plate has no such seam -- a projector's
    parameters are baked in at registration". A projector can now declare its own ``params``
    (``Operator.bind``), so the refusal comes from ``mip``'s own entry, which declares none. That
    is a strictly better answer to the same question: ``mip`` genuinely takes no ``blend_px``,
    whereas the old message would have refused a parameter the operator really did have.
    """
    import squidmip._output as out_mod

    with pytest.raises(ValueError, match="declares no parameters"):
        out_mod.write_plate(_MetaOnlyReader(), "/tmp/does-not-matter", projector="mip",
                            operator_kwargs={"blend_px": 64})
# ═══════════════════════════════════════════════════════════════════════════════════════
# Defect 2: registration must run on the registration channel, ALWAYS.
#
# `reg_c = channels.index(reg_c_global) if reg_c_global in channels else 0` silently
# registered on whichever channel happened to be FIRST in the selected subset, while the
# docstring promised "Registration always runs on registration_channel, whatever this
# selects." The same region stitched with different channel selections therefore got
# DIFFERENT SOLVED OFFSETS, with no warning — non-reproducible scientific output.
#
# The module's existing _FakeReader cannot catch this: its channel 1 is `tile // 2 + 500`,
# i.e. the same geometry, so registering on either channel gives the same answer and any
# comparison of offsets passes whether or not the bug is present. A test needs a channel
# that registers to a DIFFERENT answer, which is what _SplitChannelReader provides.
# ═══════════════════════════════════════════════════════════════════════════════════════


class _SplitChannelReader(_FakeReader):
    """Channel 0 is registrable texture; channel 1 is FLAT and carries no alignment signal.

    That asymmetry is the whole point. Registering on channel 0 recovers the injected stage
    error; registering on channel 1 cannot recover anything and degrades to the stage
    positions (all-zero offsets). So "which channel solved this" becomes a NUMBER, and the
    silent substitution stops being invisible.
    """

    def read(self, region, fov, channel, z=0, t=0):
        self.reads += 1
        if channel == CHANNELS[1]:
            return np.full((TILE, TILE), 1000, dtype=np.uint16)   # flat: unregistrable
        y, x = self._true[fov]
        return self._master[y : y + TILE, x : x + TILE]


_ERR = {3: (6.0, -4.0)}       # a known stage error only the textured channel can recover


def _offsets(reader, **kw):
    g: dict = {}
    stitch_region(reader, "A1", list(range(4)), blend_px=24, block_px=512, max_workers=2,
                  geometry=g, **kw)
    return g["offsets_px"]


def test_the_fixture_can_actually_tell_the_two_channels_apart(master):
    """Guard the guard: if both channels solved the same, every test below would be vacuous.

    This repo has already shipped a test that was dead its whole life. A fixture that cannot
    distinguish the two outcomes is the same failure mode, so prove the distinction FIRST.
    """
    textured = _offsets(_SplitChannelReader(master, error_px=_ERR),
                        registration_channel=CHANNELS[0])
    flat = _offsets(_SplitChannelReader(master, error_px=_ERR),
                    registration_channel=CHANNELS[1])
    assert np.abs(textured[3]).max() > 2.0, "channel 0 should recover the 6px injected error"
    assert np.abs(flat).max() < 0.5, "flat channel 1 should recover nothing"


def test_registration_channel_outside_the_selection_still_drives_the_solve(master):
    """THE BUG. Selecting only channel 1 must not move registration onto channel 1.

    Before the fix `reg_c` fell back to 0 — index 0 OF THE SUBSET, i.e. global channel 1, the
    flat one — so the solve silently returned zeros and the mosaic was placed on raw stage
    coordinates while the caller believed it had registered on channel 0.
    """
    got = _offsets(_SplitChannelReader(master, error_px=_ERR),
                   registration_channel=CHANNELS[0], channels=[1])
    assert np.abs(got[3]).max() > 2.0, (
        f"registration did not run on {CHANNELS[0]!r}: offsets {got[3]} look like the flat "
        "channel's (all-zero) solve, i.e. the channel was silently substituted."
    )


def test_the_solved_geometry_does_not_depend_on_which_channels_were_selected(master):
    """The reproducibility property, stated directly: same region + same registration channel
    => same offsets, whatever subset is being fused. This is the promise at _stitch.py:332."""
    kw = dict(registration_channel=CHANNELS[0])
    both = _offsets(_SplitChannelReader(master, error_px=_ERR), **kw)
    only_0 = _offsets(_SplitChannelReader(master, error_px=_ERR), channels=[0], **kw)
    only_1 = _offsets(_SplitChannelReader(master, error_px=_ERR), channels=[1], **kw)
    np.testing.assert_allclose(both, only_0, atol=1e-9)
    np.testing.assert_allclose(both, only_1, atol=1e-9)


def test_the_registration_only_channel_is_not_leaked_into_the_output(master):
    """Reading the registration channel to solve on it must not add it to the fused result —
    the caller asked for one channel and must get exactly one, in the order requested."""
    reader = _SplitChannelReader(master, error_px=_ERR)
    out = stitch_region(reader, "A1", list(range(4)), channels=[1], blend_px=24, block_px=512,
                        max_workers=2, registration_channel=CHANNELS[0])
    assert out.shape[1] == 1
    # channel 1 is the FLAT one; if the textured registration channel leaked into the output
    # this plane would have structure instead of being (feathered) constant.
    plane = out[0, 0, 0].astype(np.float64)
    interior = plane[60:-60, 60:-60]
    assert interior.std() < 1.0, f"output plane is not the flat channel (std={interior.std():.1f})"


def test_registration_costs_exactly_one_extra_plane_read_per_fov(master):
    """Registering on the RAW plane costs a read, and the price is pinned at ONE plane per FOV.

    This replaces a test that asserted registration was free. It was free because registration
    consumed the projector's output — which is precisely the defect: on a z-stack the pose graph
    was then solved on a MIP, an image maragall/stitcher never registers, costing 2 of 43 pairs
    and up to 6.62 px on the 10x tissue set. Reading the acquisition's own middle z-plane is what
    buys back the agreement, so the honest thing to pin is the SIZE of that cost, not its absence.

    One plane per FOV, one channel, one z. Not a second z-stack pass: if this ever regresses into
    re-reading depth, a 27-FOV 10-deep well pays 10x for it on every stitch.
    """
    def reads(**kw):
        r = _SplitChannelReader(master, error_px=_ERR)
        stitch_region(r, "A1", list(range(4)), blend_px=24, block_px=512, max_workers=2, **kw)
        return r.reads

    n_fovs = 4
    baseline = reads(channels=[1], register=False, registration_channel=CHANNELS[0])
    registered = reads(channels=[1], registration_channel=CHANNELS[0])
    assert registered == baseline + n_fovs, (
        f"registration should read exactly one extra plane per FOV: "
        f"register=False {baseline}, register=True {registered}, FOVs {n_fovs}"
    )

    # And the cost does not depend on the channel SELECTION — registration reads its own plane
    # either way, which is why the solved geometry is selection-independent by construction.
    assert reads(registration_channel=CHANNELS[0]) - reads(
        register=False, registration_channel=CHANNELS[0]) == n_fovs


def test_an_unknown_registration_channel_is_still_refused_by_name(master):
    # Unchanged behaviour, pinned so the fix does not accidentally make this permissive.
    with pytest.raises(ValueError, match="not a channel of this acquisition"):
        stitch_region(_SplitChannelReader(master), "A1", [0, 1],
                      registration_channel="Fluorescence_638_nm_Ex")


# ═══════════════════════════════════════════════════════════════════════════════════════
# Defect 3: the placement travels WITH the array, unconditionally.
# ═══════════════════════════════════════════════════════════════════════════════════════


def test_the_mosaic_carries_its_placement_without_being_asked(master):
    """No `geometry=` out-dict passed. The geometry must exist anyway.

    This is the defect in one line: the solved transform used to be computed at t=0 and then
    discarded unless the caller opted in to receiving it.
    """
    out = stitch_region(_SplitChannelReader(master, error_px=_ERR), "A1", list(range(4)),
                        blend_px=24, block_px=512, max_workers=2,
                        registration_channel=CHANNELS[0])
    p = out.placement
    assert p.shape == out.shape[-2:], "placement disagrees with the array it came back on"
    assert p.fovs == (0, 1, 2, 3)
    assert p.pixel_size_um == PIXEL_UM


def test_the_placement_names_the_channel_that_actually_solved_it(master):
    """Provenance, and the other half of the Defect 2 fix: the data says which channel
    solved its transform, instead of leaving it inferred from the caller's arguments."""
    out = stitch_region(_SplitChannelReader(master, error_px=_ERR), "A1", list(range(4)),
                        channels=[1], blend_px=24, block_px=512, max_workers=2,
                        registration_channel=CHANNELS[0])
    assert out.placement.reg_channel == CHANNELS[0]      # NOT the selected channel 1
    assert out.placement.reg_t == 0
    assert out.placement.registered


def test_coordinate_placement_does_not_claim_a_registration_channel(master):
    out = stitch_region(_SplitChannelReader(master), "A1", list(range(4)), register=False,
                        blend_px=24, block_px=512, max_workers=2,
                        registration_channel=CHANNELS[0])
    assert out.placement.reg_channel is None
    assert not out.placement.registered
    assert not any(any(o) for o in out.placement.offsets_px)


def test_the_placement_offsets_are_the_solved_ones(master):
    """One source of truth: what the placement reports must BE the solve, not a re-derivation."""
    g: dict = {}
    out = stitch_region(_SplitChannelReader(master, error_px=_ERR), "A1", list(range(4)),
                        blend_px=24, block_px=512, max_workers=2, geometry=g,
                        registration_channel=CHANNELS[0])
    np.testing.assert_allclose(np.asarray(out.placement.offsets_px), np.asarray(g["offsets_px"]))
    assert np.abs(np.asarray(out.placement.offsets_px)[3]).max() > 2.0   # the injected error


def test_the_mosaic_is_still_an_ordinary_array_for_every_existing_consumer(master):
    """stitch_plate yields these into the viewer's worker and the OME-Zarr writer unchanged."""
    out = stitch_region(_SplitChannelReader(master), "A1", [0, 1], blend_px=24, block_px=512,
                        max_workers=2, register=False)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 5 and out.dtype == np.uint16
    np.testing.assert_array_equal(np.asarray(out) * 0, np.zeros_like(np.asarray(out)))


# ---------------------------------------------------------------------------------------
# Defect 1: controls ported from maragall/stitcher, each pinned to its ENGINE CALL
# ---------------------------------------------------------------------------------------
#
# "A parameter that is accepted and then ignored" is a defect shape this repo has shipped
# before -- and it is shipped in the REFERENCE tool too: app.py:1472 builds a "Correct lens
# distortion" checkbox that nothing ever reads, so unchecking it does nothing and distortion
# correction always runs. So every control below is pinned to the tilefusion call it drives,
# in both directions (on -> called, off -> NOT called).

def test_lens_distortion_correction_reaches_tilefusion(master, monkeypatch):
    """The control the owner asked about BY NAME. It must drive
    tilefusion.distortion.build_seam_corrections, not merely be accepted."""
    import tilefusion.distortion as dist

    seen = {}

    def _fake(tf, **kw):
        seen["tf"] = tf
        return {}

    monkeypatch.setattr(dist, "build_seam_corrections", _fake)
    reader = _FakeReader(master, error_px={3: (6.0, -4.0)})
    stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0],
                  correct_distortion=True, max_workers=2)
    assert "tf" in seen


def test_distortion_off_does_not_call_the_engine_at_all(master, monkeypatch):
    """The reference tool's own checkbox fails exactly this."""
    import tilefusion.distortion as dist

    called = []
    monkeypatch.setattr(dist, "build_seam_corrections", lambda tf, **kw: called.append(1) or {})
    reader = _FakeReader(master, error_px={3: (6.0, -4.0)})
    stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0],
                  correct_distortion=False, max_workers=2)
    assert called == []


def test_the_distortion_adapter_exposes_everything_tilefusion_reads_off_a_TileFusion(
        master, monkeypatch):
    """squidmip runs tilefusion's pieces on IN-MEMORY arrays, so there is no TileFusion
    instance to hand build_seam_corrections. It is duck-typed on exactly six members; this
    pins the adapter against that surface, so a tilefusion upgrade that reaches for a seventh
    fails HERE rather than at a user's plate."""
    import tilefusion.distortion as dist

    seen = {}
    monkeypatch.setattr(dist, "build_seam_corrections",
                        lambda tf, **kw: seen.update(tf=tf) or {})
    reader = _FakeReader(master, error_px={3: (6.0, -4.0)})
    stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0],
                  correct_distortion=True, max_workers=2)
    tf = seen["tf"]
    for attr in ("_pixel_size", "_tile_positions", "Y", "X", "pairwise_metrics",
                 "_read_tile_region"):
        assert hasattr(tf, attr), attr
    assert (tf.Y, tf.X) == (TILE, TILE)
    assert len(tf._tile_positions) == GRID * GRID
    # The overlap strip of a real pair, read back through the adapter.
    assert np.asarray(tf._read_tile_region(0, slice(0, 8), slice(0, 8))).shape == (8, 8)


def test_distortion_is_fit_on_the_REGISTERED_positions_not_the_stage_ones(master, monkeypatch):
    """build_seam_corrections' own docstring: it corrects the residual AFTER the global solve.
    Fitting it on raw stage positions would re-measure the error registration just removed."""
    import tilefusion.distortion as dist

    seen = {}
    monkeypatch.setattr(dist, "build_seam_corrections",
                        lambda tf, **kw: seen.update(pos=np.array(tf._tile_positions)) or {})
    reader = _FakeReader(master, error_px={3: (6.0, -4.0)})
    stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0],
                  correct_distortion=True, max_workers=2)
    stage = np.array(_positions_yx_um(reader.metadata, "A1", list(range(GRID * GRID))))
    assert not np.allclose(seen["pos"], stage)


def test_the_warp_field_reaches_the_fuser(master, monkeypatch):
    """fuse_plane takes get_field=. Building the corrections and then not passing them would
    compute the whole elastic fit and throw it away -- the accepted-and-ignored shape again."""
    import tilefusion.fusion as fusion

    real = fusion.fuse_plane
    seen = {}

    def _spy(**kw):
        seen["get_field"] = kw.get("get_field")
        return real(**kw)

    monkeypatch.setattr(fusion, "fuse_plane", _spy)
    reader = _FakeReader(master, error_px={3: (6.0, -4.0)})
    stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0],
                  correct_distortion=True, max_workers=2)
    assert callable(seen["get_field"])


def test_without_distortion_the_fuser_gets_no_field(master, monkeypatch):
    """``correct_distortion=False`` is now EXPLICIT here. It used to be left off the call and
    rely on the signature default, which was ``False`` and became "on wherever it can run" on
    2026-08-03. The subject of the test is unchanged: off -> the fuser gets no warp field."""
    import tilefusion.fusion as fusion

    real = fusion.fuse_plane
    seen = {}
    monkeypatch.setattr(fusion, "fuse_plane",
                        lambda **kw: seen.update(get_field=kw.get("get_field")) or real(**kw))
    reader = _FakeReader(master)
    stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0], max_workers=2,
                  correct_distortion=False)
    assert seen["get_field"] is None


def test_distortion_correction_is_ON_by_default(master, monkeypatch):
    """Julio, 2026-08-03: "Correct lens distort should be defaulted to on". Asked of the ENGINE
    call and not of the checkbox, because the checkbox is one of four places the default lived
    and the only one that matters to a CLI or a script is this one.

    MUTATION: put the signature default back to ``False`` -> get_field is None -> red."""
    import tilefusion.fusion as fusion

    real = fusion.fuse_plane
    seen = {}
    monkeypatch.setattr(fusion, "fuse_plane",
                        lambda **kw: seen.update(get_field=kw.get("get_field")) or real(**kw))
    reader = _FakeReader(master, error_px={3: (6.0, -4.0)})
    stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0], max_workers=2)
    assert callable(seen["get_field"])


def test_the_on_by_default_does_not_break_coordinate_placement(master, monkeypatch):
    """The default is "on wherever it can run", not a plain True, and this is the reason.

    Distortion correction is registration-only and stitch_region RAISES on the combination. The
    `coordinate` control operator, Minerva's fusion and the A/B benchmarks all pass
    ``register=False`` and say nothing about distortion, so a plain ``True`` default would have
    made every one of them raise on a combination none of them asked for.

    MUTATION: change the default to a plain ``True`` -> ValueError -> red."""
    import tilefusion.fusion as fusion

    real = fusion.fuse_plane
    seen = {}
    monkeypatch.setattr(fusion, "fuse_plane",
                        lambda **kw: seen.update(get_field=kw.get("get_field")) or real(**kw))
    reader = _FakeReader(master)
    stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0], max_workers=2,
                  register=False)
    assert seen["get_field"] is None


def test_an_explicit_yes_with_no_registration_still_raises(master):
    """The loud refusal is for a USER asking for something impossible, and it survives the
    default change untouched."""
    reader = _FakeReader(master)
    with pytest.raises(ValueError, match="needs registration"):
        stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0], max_workers=2,
                      register=False, correct_distortion=True)


# -- registration timepoint -------------------------------------------------------------

def test_the_registration_timepoint_actually_drives_which_PIXELS_are_solved_on(master):
    """_REG_T was a hardcoded 0: on a multi-timepoint acquisition the geometry silently solved
    at t=0 with no way to see or set it. Same defect CLASS as the registration-channel
    substitution bug -- a solve running somewhere the user cannot name.

    This asserts on the SOLVED OFFSETS, not on what the Placement claims. Only t=2 carries
    registrable structure in this fixture, so a stitch_region that accepted registration_t and
    then read t=0 anyway recovers nothing -- which is exactly the accepted-and-ignored shape,
    and it survives any test that only checks provenance."""
    reader = _FakeReader(master, error_px={3: (6.0, -4.0)}, n_t=3, good_t=2)
    right, wrong = {}, {}
    stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0],
                  registration_t=2, max_workers=2, geometry=right)
    stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0],
                  registration_t=0, max_workers=2, geometry=wrong)
    # FOV 3 was displaced by a known (6, -4) px. Solving on the real structure recovers it.
    assert np.allclose(right["offsets_px"][3], (-6.0, 4.0), atol=1.0), right["offsets_px"][3]
    assert not np.allclose(wrong["offsets_px"][3], (-6.0, 4.0), atol=1.0)


def test_a_registration_timepoint_outside_the_acquisition_is_refused(master):
    reader = _FakeReader(master)
    reader.metadata["n_t"] = 2
    with pytest.raises(ValueError, match="registration_t"):
        stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0],
                      registration_t=5, max_workers=2)


def test_the_placement_reports_the_timepoint_that_actually_solved(master):
    """Provenance that lies is worse than none: a Placement claiming reg_t=0 while the solve
    moved elsewhere is exactly what the _REG_T comment warned about."""
    reader = _FakeReader(master)
    reader.metadata["n_t"] = 4
    out = stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0],
                        registration_t=3, max_workers=2)
    assert out.placement.reg_t == 3


# -- automatic blend width --------------------------------------------------------------

def test_auto_blend_width_is_measured_from_the_real_overlap(master):
    """maragall/stitcher's Auto checkbox. The panel's own tooltip explains why a fixed default
    is dangerous -- a ramp wider than the overlap never reaches full weight and dims the seam
    -- and this is the control that removes the guess. Formula is the reference tool's:
    median seam overlap x 2, floored at 10."""
    from squidmip._stitch import _BLEND_PX, auto_blend_px

    # Spacing chosen so the ANSWER IS NOT THE DEFAULT: a 100 px overlap -> 200 px ramp, while
    # _BLEND_PX is 128. With the module fixture's 64 px overlap the measurement happens to
    # equal the default, and a mutant that ignores the measurement entirely stays green.
    step = TILE - 100
    positions = [((i // GRID) * float(step), (i % GRID) * float(step))
                 for i in range(GRID * GRID)]
    b = auto_blend_px(positions, (PIXEL_UM, PIXEL_UM), (TILE, TILE))
    assert b == 200
    assert b != _BLEND_PX


def test_auto_blend_reaches_the_feather_profile(master, monkeypatch):
    import tilefusion.utils as utils

    real = utils.make_1d_profile
    seen = []
    monkeypatch.setattr(utils, "make_1d_profile",
                        lambda n, b, *a, **kw: seen.append(b) or real(n, b, *a, **kw))
    from squidmip._stitch import auto_blend_px

    reader = _FakeReader(master)
    positions = _positions_yx_um(reader.metadata, "A1", list(range(GRID * GRID)))
    expected = auto_blend_px(positions, (PIXEL_UM, PIXEL_UM), (TILE, TILE))
    stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0],
                  blend_px=None, max_workers=2)          # None = Auto
    assert seen and seen[0] == expected


def test_an_explicit_blend_width_still_wins_over_auto(master, monkeypatch):
    import tilefusion.utils as utils

    real = utils.make_1d_profile
    seen = []
    monkeypatch.setattr(utils, "make_1d_profile",
                        lambda n, b, *a, **kw: seen.append(b) or real(n, b, *a, **kw))
    reader = _FakeReader(master)
    stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0],
                  blend_px=37, max_workers=2)
    assert seen and seen[0] == 37


def test_auto_blend_falls_back_when_nothing_overlaps(master):
    """A sparse/freeform acquisition legitimately has isolated FOVs -- the same case
    solve_offsets_px degrades to stage placement for. Auto must not divide by an empty set."""
    from squidmip._stitch import auto_blend_px

    far = [(0.0, 0.0), (10000.0, 0.0), (0.0, 10000.0), (10000.0, 10000.0)]
    assert auto_blend_px(far, (PIXEL_UM, PIXEL_UM), (TILE, TILE)) > 0


# ---------------------------------------------------------------------------------------
# registration reads the RAW plane, not the projector's output
# ---------------------------------------------------------------------------------------


class _ZStackReader(_FakeReader):
    """A 3-plane stack whose MIP is unregistrable and whose MIDDLE plane is not.

    z=1 carries the master texture with the injected stage error. z=0 and z=2 are a CONSTANT
    field brighter than any master pixel, identical for every FOV. So:

        max over z  -> the constant, everywhere: no structure, no alignment signal at all
        plane z=1   -> the texture: the 6 px error is recoverable

    That asymmetry turns "which image did registration actually solve on" into a NUMBER. It is
    the same trick ``_SplitChannelReader`` plays for channels, and it exists because the MIP
    substitution was invisible for exactly as long as nothing measured it: on the real 10x
    tissue set it cost 2 of 43 pairs and 6.62 px, silently.
    """

    _FLOOR = 60000  # above the master's ~42000 ceiling, so it wins every max()

    def __init__(self, *args, n_z: int = 3, **kw):
        super().__init__(*args, **kw)
        self.metadata["n_z"] = n_z
        self.metadata["z_levels"] = list(range(n_z))
        self._mid_z = n_z // 2

    def read(self, region, fov, channel, z=0, t=0):
        if z != self._mid_z:
            self.reads += 1
            return np.full((TILE, TILE), self._FLOOR, dtype=np.uint16)
        return super().read(region, fov, channel, z=z, t=t)


def test_registration_solves_on_the_raw_middle_plane_not_the_projected_one(master):
    """The defect this module carried: the pose graph was solved on the projector's output.

    ``stitch_region`` runs a z-reducer, so on a z-stack the thing registration used to see was a
    MIP -- an image ``TileFusion`` never registers. Here the MIP is constant by construction, so
    a regression to it recovers NOTHING while the correct read recovers the injected 6 px.
    """
    reader = _ZStackReader(master, error_px=_ERR)
    fovs = list(range(4))
    positions = _positions_yx_um(reader.metadata, "A1", fovs)

    # Guard the guard: prove the two registration inputs give DIFFERENT answers on this fixture,
    # or the assertion below is vacuous. This repo has already shipped a test that was dead its
    # whole life.
    mip = np.stack([
        np.max(np.stack([reader.read("A1", f, CHANNELS[0], z=z) for z in range(3)]), axis=0)[None]
        for f in fovs
    ])
    mip_offsets = solve_offsets_px(mip, positions, (PIXEL_UM, PIXEL_UM), (TILE, TILE))
    assert np.abs(mip_offsets).max() < 0.5, (
        "the fixture's MIP is supposed to be unregistrable; if it registers, this test cannot "
        "tell the MIP path from the raw-plane path"
    )

    g: dict = {}
    stitch_region(_ZStackReader(master, error_px=_ERR), "A1", fovs,
                  blend_px=24, block_px=512, max_workers=2,
                  registration_channel=CHANNELS[0], geometry=g)
    offsets = np.asarray(g["offsets_px"])
    assert np.abs(offsets[3]).max() > 2.0, (
        "registration recovered nothing, i.e. it solved on the MIP (constant here) rather than "
        f"the raw middle plane; offsets={offsets.tolist()}"
    )
    assert g["placement"].reg_z == 1, "the middle plane of a 3-deep stack is z=1"


def test_the_registration_plane_is_the_callers_and_is_recorded(master):
    """``registration_z`` selects the plane, and the Placement says which one it was.

    Provenance for the same reason as ``reg_channel``/``reg_t``: a solve running on a plane the
    user cannot name is how this class of defect stays invisible.
    """
    g: dict = {}
    stitch_region(_ZStackReader(master, error_px=_ERR), "A1", list(range(4)),
                  blend_px=24, block_px=512, max_workers=2, registration_z=0,
                  registration_channel=CHANNELS[0], geometry=g)
    assert g["placement"].reg_z == 0
    # z=0 is the constant floor: nothing to register against, so it degrades to stage placement.
    assert np.abs(np.asarray(g["offsets_px"])).max() < 0.5

    with pytest.raises(ValueError, match="registration_z=7 is outside"):
        stitch_region(_ZStackReader(master), "A1", list(range(4)), registration_z=7,
                      blend_px=24, block_px=512, max_workers=2)


def test_a_tile_that_registered_against_nothing_is_placed_by_the_affine_not_left_at_stage():
    """``two_round_optimization`` does NOT place a tile with no registered edge — it warns.

    Its docstring is explicit that such a tile "is left for the caller's affine/stage-model
    fallback", and this module used to have no such fallback, so the tile kept a ZERO offset and
    fused at its raw, miscalibrated stage position. Zero is also what a perfectly-placed tile
    gets, so nothing downstream could tell the two apart. Measured on the real 10x tissue set,
    region manual1: FOV 3 registers against nothing and TileFusion places it 34.4 px away.
    """
    from squidmip._stitch import _place_unconstrained_tiles

    # 3x3 grid at a 100 um pitch, 1 um/px. Tile 8 (the corner) is the isolated one.
    step, n = 100.0, 3
    positions = [(float(r * step), float(c * step)) for r in range(n) for c in range(n)]
    # Every neighbouring pair EXCEPT the ones touching tile 8: 12 - 2 = 10 edges, over the
    # _MIN_PAIRS_FOR_AFFINE floor of 8.
    metrics, edges = {}, []
    for i in range(n * n):
        for j in (i + 1, i + n):
            if j >= n * n or (j == i + 1 and (i + 1) % n == 0):
                continue
            if 8 in (i, j):
                continue
            dy = positions[j][0] - positions[i][0]
            dx = positions[j][1] - positions[i][1]
            metrics[(i, j)] = (dy, dx, 0.99)
            edges.append({"i": i, "j": j, "t": np.array([dy, dx]), "w": 0.99})
    assert len(metrics) >= 8

    solved = np.zeros((n * n, 2), dtype=np.float64)
    placed = _place_unconstrained_tiles(solved, edges, metrics, positions, (1.0, 1.0))

    assert np.allclose(placed[:8], 0.0), "tiles the solve constrained must keep their offsets"
    assert not np.allclose(placed[8], 0.0), (
        "the unconstrained corner tile was left at its raw stage position; the affine fallback "
        "did not run"
    )


# ---------------------------------------------------------------------------------------
# flat-field correction, applied in the READ path
# ---------------------------------------------------------------------------------------


def _nonunit_profile(gain: float = 2.0):
    """A profile with a KNOWN, non-unit gain: the top half is dimmer, mean forced to 1.0."""
    from squidmip._flatfield import FlatfieldProfile

    ff = np.ones((TILE, TILE), dtype=np.float32)
    ff[: TILE // 2] = gain
    ff /= ff.mean()
    return FlatfieldProfile(ff)


def test_the_flatfield_wrapper_corrects_every_plane_it_hands_back(master):
    """The wrapper IS the mechanism, and its position is the feature.

    maragall/stitcher corrects inside _read_tile_region -- the reader feeding the registration
    strips -- so phase correlation runs on corrected pixels. Wrapping the READER puts squidmip in
    the same place, and unlike correcting the projector's OUTPUT it holds for a non-monotone
    z-reducer too, because the reducer never sees uncorrected data.
    """
    from squidmip._flatfield import correct_flatfield
    from squidmip._stitch import _FlatfieldReader

    inner = _FakeReader(master)
    profile = _nonunit_profile()
    wrapped = _FlatfieldReader(inner, {CHANNELS[0]: profile})

    raw = inner.read("A1", 0, CHANNELS[0], 0, 0)
    got = wrapped.read("A1", 0, CHANNELS[0], 0, 0)
    np.testing.assert_array_equal(got, correct_flatfield(raw, profile))
    assert not np.array_equal(got, raw), "the fixture profile must actually change the pixels"

    # A channel with no profile passes through untouched rather than raising -- the same
    # identity fallback TileWarper gives an unfittable seam.
    np.testing.assert_array_equal(
        wrapped.read("A1", 0, CHANNELS[1], 0, 0), inner.read("A1", 0, CHANNELS[1], 0, 0))
    assert wrapped.metadata is inner.metadata, "metadata must be delegated, not snapshotted"


def test_illumination_correction_is_on_by_default_and_recorded(master):
    """Default ON is a step PAST the standalone (its checkbox is on, but nothing presses the
    Calculate button), so both the default and its provenance are pinned."""
    g: dict = {}
    stitch_region(_FakeReader(master), "A1", list(range(4)), blend_px=24, block_px=512,
                  max_workers=2, geometry=g)
    assert g["placement"].illumination_corrected is True

    g_off: dict = {}
    stitch_region(_FakeReader(master), "A1", list(range(4)), blend_px=24, block_px=512,
                  max_workers=2, correct_illumination=False, geometry=g_off)
    assert g_off["placement"].illumination_corrected is False


def test_a_supplied_profile_is_used_instead_of_estimating(master):
    """A caller-supplied profile must be honoured verbatim — that is how stitch_plate keeps every
    well on ONE plate-wide gain field instead of a different one per well."""
    kw = dict(channels=[0], blend_px=24, block_px=512, max_workers=2, register=False,
              correct_distortion=False)
    supplied = stitch_region(_FakeReader(master), "A1", list(range(4)),
                             flatfield={CHANNELS[0]: _nonunit_profile()}, **kw)
    plain = stitch_region(_FakeReader(master), "A1", list(range(4)),
                          correct_illumination=False, **kw)
    assert not np.array_equal(np.asarray(supplied), np.asarray(plain)), (
        "the supplied non-unit profile did not reach the fused pixels")


def test_the_gui_selected_profile_reaches_stitching(master):
    """A profile chosen in the GUI's flat-field tab must correct the STITCH too.

    It did not. ``_flatfield.set_profile`` was read by exactly one consumer -- the registered
    ``flatfield`` plane-op -- so ``resolve_flatfield`` walked straight past it to the ``.npy``
    lookup and estimated its own field. Two owners of "which gain field is this plate corrected
    by", and the user's choice was the one with no effect.
    """
    from squidmip._flatfield import clear_profile, set_profile
    from squidmip._stitch import resolve_flatfield

    reader = _FakeReader(master)
    chosen = _nonunit_profile()
    set_profile(chosen)
    try:
        resolved = resolve_flatfield(reader, "A1", list(range(4)))
        assert all(p is chosen for p in resolved.values()), (
            "the GUI-selected profile did not reach the stitch's profile resolution")
        assert set(resolved) == set(CHANNELS), "every channel of the run must be covered"

        # ...and an explicit per-call profile still outranks it: one total precedence, no tie.
        kw = dict(channels=[0], blend_px=24, block_px=512, max_workers=2, register=False,
                  correct_distortion=False)
        explicit = stitch_region(_FakeReader(master), "A1", list(range(4)),
                                 flatfield={CHANNELS[0]: _nonunit_profile(gain=4.0)}, **kw)
        from_global = stitch_region(_FakeReader(master), "A1", list(range(4)), **kw)
        assert not np.array_equal(np.asarray(explicit), np.asarray(from_global)), (
            "the explicit flatfield= argument was overridden by the GUI-selected global")
    finally:
        clear_profile()


def test_the_flatfield_stage_says_what_it_is_doing_before_it_does_it(master, monkeypatch, caplog):
    """Announce the stage at its START, not in the past tense once it is over.

    Every flat-field line used to be emitted after the work finished, so the log panel -- which is
    also where the run's progress bar lives, and the only surface that can move during a stage that
    runs BEFORE the first region -- was silent for the whole estimate and then reported it as done.
    """
    import logging

    import squidmip._flatfield as ff_mod

    log_at_entry = []

    def _fake_estimate(planes, *, use_darkfield=False):
        log_at_entry.append([r.getMessage() for r in caplog.records])
        return _nonunit_profile()

    monkeypatch.setattr(ff_mod, "estimate_profile", _fake_estimate)
    caplog.set_level(logging.INFO)
    from squidmip._stitch import estimate_region_flatfield

    estimate_region_flatfield(_FakeReader(master), "A1", list(range(4)), channels=[0, 1], z=0)

    assert len(log_at_entry) == 2, "the estimator should have run once per channel"
    said_first = log_at_entry[0]
    assert any("estimating" in m for m in said_first), (
        "nothing was logged before the first channel's estimate started")
    assert any("channel 1 of 2" in m for m in said_first), "no per-channel progress line"
    assert any("channel 2 of 2" in m for m in log_at_entry[1]), (
        "the second channel's line arrived after its estimate, not before it")


def test_the_affine_fallback_refuses_to_guess_from_too_few_pairs():
    """Below ``_MIN_PAIRS_FOR_AFFINE`` the fit is not trustworthy, so the tiles stay put.

    Degrading to the stage position is the honest outcome here — it is what the caller had
    before — and it must not be dressed up as a solve.
    """
    from squidmip._stitch import _place_unconstrained_tiles

    positions = [(0.0, 0.0), (0.0, 100.0), (100.0, 0.0), (500.0, 500.0)]
    metrics = {(0, 1): (0.0, 100.0, 0.9), (0, 2): (100.0, 0.0, 0.9)}
    edges = [{"i": 0, "j": 1, "t": np.array([0.0, 100.0]), "w": 0.9},
             {"i": 0, "j": 2, "t": np.array([100.0, 0.0]), "w": 0.9}]
    solved = np.zeros((4, 2), dtype=np.float64)
    placed = _place_unconstrained_tiles(solved, edges, metrics, positions, (1.0, 1.0))
    assert np.allclose(placed, 0.0)
