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
