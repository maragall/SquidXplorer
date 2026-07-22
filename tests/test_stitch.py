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
                 regions=("A1",), step: int = STEP):
        self._master = master
        self._step = step
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
            "n_t": 1,
            "frame_shape": (TILE, TILE),
            "dtype": np.dtype(np.uint16),
            "pixel_size_um": PIXEL_UM,
        }
        self.reads = 0

    def read(self, region, fov, channel, z=0, t=0):
        self.reads += 1
        y, x = self._true[fov]
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


def test_write_plate_refuses_operator_kwargs_for_a_non_region_operator():
    """project_plate has no such seam -- a projector's parameters are baked in at
    registration. Accepting them here and dropping them is the silent failure this whole
    change exists to avoid, so refuse BY NAME instead."""
    import squidmip._output as out_mod

    with pytest.raises(ValueError, match="operator_kwargs"):
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


def test_honouring_the_registration_channel_costs_no_extra_reads(master):
    """The fix reads a channel the caller did not select — but not one extra byte off disk.

    project_well already decodes EVERY channel of the FOV; the old code just discarded the
    unselected ones at `[:, channels, 0]`. So including the registration channel is free in
    I/O, and that is worth pinning: if this ever regresses into a second read pass, a 27-FOV
    well pays for it on every stitch.
    """
    def reads(**kw):
        r = _SplitChannelReader(master, error_px=_ERR)
        stitch_region(r, "A1", list(range(4)), blend_px=24, block_px=512, max_workers=2, **kw)
        return r.reads

    all_channels = reads(registration_channel=CHANNELS[0])
    reg_outside_selection = reads(channels=[1], registration_channel=CHANNELS[0])
    no_registration = reads(channels=[1], register=False, registration_channel=CHANNELS[0])
    assert reg_outside_selection == all_channels == no_registration, (
        f"read counts diverged: all={all_channels}, reg-outside-selection="
        f"{reg_outside_selection}, register=False {no_registration}"
    )


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
