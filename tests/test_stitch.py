"""Stitch operator tests: the synthetic mosaic is cut from one master image, so every
claim is a number against ground truth."""

from __future__ import annotations

import numpy as np
import pytest

# tilefusion is deliberately not a dependency; without it this seam is untested, not passing.
pytest.importorskip("tilefusion", reason="tilefusion (maragall/stitcher) not installed: the stitch "
                                         "adapter is UNTESTED here, which is not the same as passing")

from squidxplorer._engine import (
    _OPERATORS,
    add_region_operator,
    available_region_operators,
)
from squidxplorer._stitch import (
    _mosaic_geometry,
    _positions_yx_um,
    solve_offsets_px,
    _stitch_plate,
    stitch_region,
)

TILE = 256
STEP = 192            # -> 64 px overlap, ~25%
GRID = 2              # 2x2 = 4 FOVs
PIXEL_UM = 1.0        # 1 um/px keeps micrometres and pixels numerically identical
CHANNELS = ["Fluorescence_405_nm_Ex", "Fluorescence_488_nm_Ex"]


def _master(seed: int = 0) -> np.ndarray:
    """A lightly low-pass filtered random texture: registrable, unlike white noise or a ramp."""
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
    """Minimal ``SquidReader`` duck-type: ``.metadata`` + ``.read``."""

    def __init__(self, master: np.ndarray, error_px: dict[int, tuple[float, float]] | None = None,
                 regions=("A1",), step: int = STEP, n_t: int = 1, good_t: int = 0):
        self._master = master
        self._step = step
        # Only one timepoint carries registrable structure; the others are noise.
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

    def read(self, region, fov, channel, z_level=0, time_point=0):
        self.reads += 1
        y, x = self._true[fov]
        if time_point != self._good_t:
            # Unregistrable by construction: white noise aliases under any sub-pixel shift.
            return np.random.default_rng(1000 + fov).integers(
                0, 65535, size=(TILE, TILE), dtype=np.uint16)
        tile = self._master[y : y + TILE, x : x + TILE]
        # Second channel is a scaled copy: distinct data, same geometry.
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
    """Truncating sub-pixel origins re-introduces the misalignment registration just removed."""
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
    # The solve is gauge-free (anchors tile 0), so compare relative geometry only.
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
    """RMSE of the fused channel-0 mosaic against the master crop, on the interior only."""
    a = fused[0, 0, 0].astype(np.float64)
    # Crop both to their common extent; both share the top-left origin (tile 0 anchors there).
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
    """Truncation gives 11/12 for 11.5/12.7, rounding gives 12/13; 10.5 -> 10 is half-to-even."""
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
    """Registration must measurably reduce error vs ground truth."""
    stitched, placed = fused_pair
    e_stitched = _rmse_vs_truth(stitched, master)
    e_placed = _rmse_vs_truth(placed, master)
    assert e_stitched < e_placed * 0.5, f"stitched {e_stitched:.1f} vs placed {e_placed:.1f}"


def test_all_channels_share_one_geometry(master):
    """Channels must be placed by the same solve or they stop overlaying."""
    reader = _FakeReader(master, error_px={3: (6.0, -4.0)})
    out = stitch_region(reader, "A1", list(range(4)), blend_px=24, block_px=512, max_workers=2)
    assert out.shape[1] == len(CHANNELS)
    c0, c1 = out[0, 0, 0].astype(np.float64), out[0, 1, 0].astype(np.float64)
    # Channel 1 is channel 0 // 2 + 500 by construction, so matched geometry correlates near 1.
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
    _OPERATORS.pop(name, None)
    try:
        add_region_operator(name, lambda r, reg, fovs, **kw: np.zeros((1, 1, 1, 2, 2), np.uint16))
        assert name in available_region_operators()
        out = list(_stitch_plate(_FakeReader(master), operator=name))
        assert [r for r, _f, _i in out] == ["A1"]
    finally:
        _OPERATORS.pop(name, None)


def test_duplicate_operator_refused():
    with pytest.raises(ValueError, match="already defined"):
        add_region_operator("stitch", lambda *a, **k: None)


@pytest.mark.parametrize("bad", ["", None])
def test_invalid_operator_registration(bad):
    with pytest.raises(ValueError):
        add_region_operator(bad or "", bad)


def test_unknown_operator_names_the_alternatives(master):
    with pytest.raises(KeyError, match="unknown operator 'nope'"):
        list(_stitch_plate(_FakeReader(master), operator="nope"))


def test_a_plane_operator_handed_to_the_region_loop_is_refused_as_the_wrong_kind(master):
    """`mip` is a real operator and not a region one; the refusal has to say which, not 'unknown'."""
    with pytest.raises(KeyError, match="registered operator but not a REGION operator"):
        list(_stitch_plate(_FakeReader(master), operator="mip"))


# ---------------------------------------------------------------------------------------
# _stitch_plate: the per-FOV loop contract, mirrored
# ---------------------------------------------------------------------------------------


def _fast_plate(reader, **kw):
    """_stitch_plate with the cheap operator settings the contract tests need."""
    kw.setdefault("channels", [0])
    kw.setdefault("blend_px", 24)
    kw.setdefault("block_px", 512)
    kw.setdefault("max_workers", 2)
    kw.setdefault("register", False)
    return _stitch_plate(reader, **kw)


def test_one_result_per_region_anchored_at_first_fov(master):
    """A stitched well yields one array, not one per FOV."""
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
    _OPERATORS.pop(name, None)
    add_region_operator(name, boom)
    try:
        with pytest.raises(RuntimeError, match="corrupt plane"):
            list(_stitch_plate(_FakeReader(master, regions=("A1", "A2")), operator=name))
    finally:
        _OPERATORS.pop(name, None)


def test_on_error_skips_the_well_and_keeps_going(master):
    """One corrupt well must not abort a plate when the caller opts in."""
    def flaky(reader, region, fovs, **kw):
        if region == "A1":
            raise RuntimeError("corrupt plane")
        return np.zeros((1, 1, 1, 2, 2), np.uint16)

    name = "_test_flaky"
    _OPERATORS.pop(name, None)
    add_region_operator(name, flaky)
    seen = []
    try:
        out = list(
            _stitch_plate(
                _FakeReader(master, regions=("A1", "A2")),
                operator=name,
                on_error=lambda r, f, e: seen.append((r, f, type(e).__name__)),
            )
        )
    finally:
        _OPERATORS.pop(name, None)
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
    _OPERATORS.pop(name, None)
    add_region_operator(name, counted)
    try:
        reader = _FakeReader(master, regions=tuple(f"A{i}" for i in range(8)))
        assert len(list(_stitch_plate(reader, operator=name, workers=2))) == 8
    finally:
        _OPERATORS.pop(name, None)
    assert peak <= 2, f"in-flight window ran to {peak}, expected <= 2"


def test_stitching_a_plane_op_fuses_every_plane_instead_of_keeping_only_z0(master):
    """A plane-op is stitched per z plane; no plane may go missing."""
    from squidxplorer._stitch import _resolve_operator, stitch_region

    plane_ops = [n for n in ("bgsub", "decon", "flatfield")
                 if not _resolve_operator(n).consumes]
    assert plane_ops, "expected bgsub/decon/flatfield to be plane-ops (consumes == frozenset())"

    n_z = 3
    reader = _FakeReader(master)
    reader.metadata["n_z"] = n_z
    reader.metadata["z_levels"] = list(range(n_z))
    out = stitch_region(reader, "A1", [0, 1, 2, 3], z_operator="bgsub", register=False,
                        correct_illumination=False)
    assert out.shape[2] == n_z, (
        f"a plane-op fused {out.shape[2]} of {n_z} z planes; keeping only plane 0 is the silent "
        "truncation the old NotImplementedError existed to prevent")


# ---------------------------------------------------------------------------------------
# blunder rejection: the two outlier knobs must actually reach the solver
# ---------------------------------------------------------------------------------------


def _spy_two_round(monkeypatch):
    """Record the args two_round_optimization is called with, and short-circuit it.

    Patched on the tilefusion module because _solve imports it at call time.
    """
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
    """Unset, the solve must use TileFusion.run()'s own 0.5 / 2.0."""
    from squidxplorer._stitch import _ABS_THRESH, _REL_THRESH

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
    """Panel kwargs travel the region loop -> stitch_region -> solve_offsets_px -> two_round_optimization."""
    seen = _spy_two_round(monkeypatch)
    reader = _FakeReader(master, error_px={3: (6.0, -4.0)})
    stitch_region(reader, "A1", list(range(GRID * GRID)), channels=[0], blend_px=24,
                  block_px=512, max_workers=2, rel_thresh=0.33, abs_thresh=9.0)
    assert (seen["rel"], seen["abs"]) == (0.33, 9.0)


def test_thresholds_must_be_positive(master):
    """A zero/negative threshold rejects every link or none; refuse by name."""
    tiles, positions = _tiles_and_positions(master)
    with pytest.raises(ValueError, match="rel_thresh"):
        solve_offsets_px(tiles, positions, (1.0, 1.0), (TILE, TILE), rel_thresh=0.0)
    with pytest.raises(ValueError, match="abs_thresh"):
        solve_offsets_px(tiles, positions, (1.0, 1.0), (TILE, TILE), abs_thresh=-1.0)


# ---------------------------------------------------------------------------------------
# write_plate: the save path has to carry the operator's settings too
# ---------------------------------------------------------------------------------------


class _MetaOnlyReader:
    metadata = {"regions": ["A1"], "fovs_per_region": {"A1": [0]},
                "channels": [{"name": "c0"}], "pixel_size_um": 1.0,
                "frame_shape": (8, 8), "dtype": "uint16", "n_t": 1}


def test_write_plate_forwards_operator_kwargs_to_the_region_loop(monkeypatch):
    import squidxplorer._output as out_mod
    import squidxplorer._stitch as st

    seen = {}

    def _fake_region_loop(reader, **kw):
        seen.update(kw)
        return iter(())

    monkeypatch.setattr(st, "_stitch_plate", _fake_region_loop)

    def _fake_write_from_stream(meta, stream, out, **kw):
        fields = sum(1 for _ in stream)          # drain it: the real one consumes the stream
        return {"plate": str(out), "tiff": None, "n_wells": 0, "n_fields": fields,
                "n_fields_written": fields, "levels": 1, "complete": True, "stopped": False}

    monkeypatch.setattr(out_mod, "write_from_stream", _fake_write_from_stream)
    out_mod.write_plate(_MetaOnlyReader(), "/tmp/does-not-matter", operator="stitch",
                        operator_kwargs={"blend_px": 64, "rel_thresh": 0.25, "register": False})
    assert seen["blend_px"] == 64
    assert seen["rel_thresh"] == 0.25
    assert seen["register"] is False


def test_write_plate_refuses_operator_kwargs_an_operator_does_not_declare():
    """Accepting a parameter and dropping it is the silent failure this seam exists to avoid."""
    import squidxplorer._output as out_mod

    with pytest.raises(ValueError, match="declares no parameters"):
        out_mod.write_plate(_MetaOnlyReader(), "/tmp/does-not-matter", operator="mip",
                            operator_kwargs={"blend_px": 64})


# ---------------------------------------------------------------------------------------
# registration must run on the registration channel, always
# ---------------------------------------------------------------------------------------


class _SplitChannelReader(_FakeReader):
    """Channel 0 is registrable texture; channel 1 is flat and carries no alignment signal."""

    def read(self, region, fov, channel, z_level=0, time_point=0):
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
    """Guard the guard: if both channels solved the same, every test below would be vacuous."""
    textured = _offsets(_SplitChannelReader(master, error_px=_ERR),
                        registration_channel=CHANNELS[0])
    flat = _offsets(_SplitChannelReader(master, error_px=_ERR),
                    registration_channel=CHANNELS[1])
    assert np.abs(textured[3]).max() > 2.0, "channel 0 should recover the 6px injected error"
    assert np.abs(flat).max() < 0.5, "flat channel 1 should recover nothing"


def test_registration_channel_outside_the_selection_still_drives_the_solve(master):
    """Selecting only channel 1 must not move registration onto channel 1."""
    got = _offsets(_SplitChannelReader(master, error_px=_ERR),
                   registration_channel=CHANNELS[0], channels=[1])
    assert np.abs(got[3]).max() > 2.0, (
        f"registration did not run on {CHANNELS[0]!r}: offsets {got[3]} look like the flat "
        "channel's (all-zero) solve, i.e. the channel was silently substituted."
    )


def test_the_solved_geometry_does_not_depend_on_which_channels_were_selected(master):
    """Same region + same registration channel => same offsets, whatever subset is fused."""
    kw = dict(registration_channel=CHANNELS[0])
    both = _offsets(_SplitChannelReader(master, error_px=_ERR), **kw)
    only_0 = _offsets(_SplitChannelReader(master, error_px=_ERR), channels=[0], **kw)
    only_1 = _offsets(_SplitChannelReader(master, error_px=_ERR), channels=[1], **kw)
    np.testing.assert_allclose(both, only_0, atol=1e-9)
    np.testing.assert_allclose(both, only_1, atol=1e-9)


def test_the_registration_only_channel_is_not_leaked_into_the_output(master):
    """Reading the registration channel to solve on it must not add it to the fused result."""
    reader = _SplitChannelReader(master, error_px=_ERR)
    out = stitch_region(reader, "A1", list(range(4)), channels=[1], blend_px=24, block_px=512,
                        max_workers=2, registration_channel=CHANNELS[0])
    assert out.shape[1] == 1
    # channel 1 is the flat one; a leak would give this plane structure.
    plane = out[0, 0, 0].astype(np.float64)
    interior = plane[60:-60, 60:-60]
    assert interior.std() < 1.0, f"output plane is not the flat channel (std={interior.std():.1f})"


def test_registration_costs_exactly_one_extra_plane_read_per_fov(master):
    """Registering on the raw plane costs exactly one extra plane read per FOV."""
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

    # The cost does not depend on the channel selection.
    assert reads(registration_channel=CHANNELS[0]) - reads(
        register=False, registration_channel=CHANNELS[0]) == n_fovs


def test_an_unknown_registration_channel_is_still_refused_by_name(master):
    with pytest.raises(ValueError, match="not a channel of this acquisition"):
        stitch_region(_SplitChannelReader(master), "A1", [0, 1],
                      registration_channel="Fluorescence_638_nm_Ex")


# ---------------------------------------------------------------------------------------
# the placement travels with the array, unconditionally
# ---------------------------------------------------------------------------------------


def test_the_mosaic_carries_its_placement_without_being_asked(master):
    """No `geometry=` out-dict passed; the geometry must exist anyway."""
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
    """the region loop yields these into the viewer's worker and the OME-Zarr writer unchanged."""
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
    """squidxplorer runs tilefusion's pieces on IN-MEMORY arrays, so there is no TileFusion
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
    from squidxplorer._stitch import _BLEND_PX, auto_blend_px

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
    from squidxplorer._stitch import auto_blend_px

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
    from squidxplorer._stitch import auto_blend_px

    far = [(0.0, 0.0), (10000.0, 0.0), (0.0, 10000.0), (10000.0, 10000.0)]
    assert auto_blend_px(far, (PIXEL_UM, PIXEL_UM), (TILE, TILE)) > 0


# ---------------------------------------------------------------------------------------
# registration reads the RAW plane, not the z operator's output
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

    def read(self, region, fov, channel, z_level=0, time_point=0):
        if z_level != self._mid_z:
            self.reads += 1
            return np.full((TILE, TILE), self._FLOOR, dtype=np.uint16)
        return super().read(region, fov, channel, z_level=z_level, time_point=time_point)


def test_registration_solves_on_the_raw_middle_plane_not_the_projected_one(master):
    """The defect this module carried: the pose graph was solved on the z operator's output.

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
        np.max(np.stack([reader.read("A1", f, CHANNELS[0], z_level=z) for z in range(3)]), axis=0)[None]
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
    from squidxplorer._stitch import _place_unconstrained_tiles

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
    from squidxplorer._flatfield import FlatfieldProfile

    ff = np.ones((TILE, TILE), dtype=np.float32)
    ff[: TILE // 2] = gain
    ff /= ff.mean()
    return FlatfieldProfile(ff)


def test_the_flatfield_wrapper_corrects_every_plane_it_hands_back(master):
    """The wrapper IS the mechanism, and its position is the feature.

    maragall/stitcher corrects inside _read_tile_region -- the reader feeding the registration
    strips -- so phase correlation runs on corrected pixels. Wrapping the READER puts squidxplorer in
    the same place, and unlike correcting the z operator's OUTPUT it holds for a non-monotone
    z-reducer too, because the reducer never sees uncorrected data.
    """
    from squidxplorer._flatfield import correct_flatfield
    from squidxplorer._stitch import _FlatfieldReader

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
    """A caller-supplied profile must be honoured verbatim — that is how the region loop keeps every
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
    from squidxplorer._flatfield import clear_profile, set_profiles
    from squidxplorer._stitch import resolve_flatfield

    reader = _FakeReader(master)
    chosen = _nonunit_profile()
    # ONE PER CHANNEL. The global was a single profile and this function broadcast it, which is
    # what made the GUI and the stored .npy disagree; the selection now has to cover the run.
    set_profiles({c: chosen for c in CHANNELS})
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


def test_loading_the_stored_profile_in_the_gui_does_not_change_the_stitch(master, tmp_path):
    """ONE FILE, ONE ANSWER. ``resolve_flatfield`` loaded the acquisition's ``(C, Y, X)`` ``.npy``
    per channel, while ``_selected_profiles`` broadcast the GUI's single global over every
    channel — so loading THAT SAME FILE through "Load illumination profile" changed the mosaic:
    identical for channel 0 and different for every other one (measured on the 10x set: max|d|
    0.3335 / 0.0237 / 0.1411 for 488 / 561 / 638). Same data, two answers, decided by whether the
    user had clicked a button.
    """
    pytest.importorskip("tilefusion.flatfield")
    from tilefusion.flatfield import save_flatfield

    from squidxplorer._flatfield import FlatfieldProfile, clear_profile, set_profiles
    from squidxplorer._stitch import _flatfield_npy_path, resolve_flatfield

    reader = _FakeReader(master)
    reader._path = tmp_path                      # give the acquisition a home for its profile
    path = _flatfield_npy_path(reader)
    fields = []
    for i in range(len(CHANNELS)):               # one GENUINELY different field per channel
        f = np.ones((TILE, TILE), dtype=np.float32)
        f[: TILE // 2] = 1.5 + 0.75 * i
        fields.append((f / f.mean()).astype(np.float32))
    save_flatfield(path, np.stack(fields), None)

    clear_profile()
    from_file = resolve_flatfield(reader, "A1", list(range(4)))
    set_profiles(FlatfieldProfile.per_channel_from_npy(path, CHANNELS))
    try:
        with_selection = resolve_flatfield(reader, "A1", list(range(4)))
    finally:
        clear_profile()

    assert set(from_file) == set(with_selection) == set(CHANNELS)
    for name in CHANNELS:
        a = from_file[name].flatfield
        b = with_selection[name].flatfield
        assert np.array_equal(a, b), (
            f"{name}: the stitch corrects by a different gain field once the SAME file is loaded "
            f"in the GUI — max|d| {float(np.abs(a - b).max()):.4f} (file mean {a.mean():.4f} "
            f"range {a.min():.3f}-{a.max():.3f} vs selection range {b.min():.3f}-{b.max():.3f})")


def test_a_partial_gui_selection_is_named_in_the_log_and_never_broadcast(master, caplog):
    """The GUI's auto-estimate installs the ONE channel it estimated. That is not a licence to
    correct the others with it — and it must not be dropped in silence either."""
    import logging

    from squidxplorer._flatfield import FlatfieldProfile, clear_profile, set_profile
    from squidxplorer._stitch import _selected_profiles

    clear_profile()
    ff = np.ones((TILE, TILE), dtype=np.float32)
    set_profile(FlatfieldProfile(ff), channel=CHANNELS[0])
    try:
        with caplog.at_level(logging.WARNING, logger="squidxplorer"):
            picked = _selected_profiles(CHANNELS)
    finally:
        clear_profile()
    assert picked is None, (
        f"a selection covering {1} of {len(CHANNELS)} channels was used for all of them: {picked}")
    text = caplog.text
    assert CHANNELS[0] in text and CHANNELS[1] in text, (
        f"the partial selection was dropped without naming the channels: {text!r}")


def test_the_flatfield_stage_says_what_it_is_doing_before_it_does_it(master, monkeypatch, caplog):
    """Announce the stage at its START, not in the past tense once it is over.

    Every flat-field line used to be emitted after the work finished, so the log panel -- which is
    also where the run's progress bar lives, and the only surface that can move during a stage that
    runs BEFORE the first region -- was silent for the whole estimate and then reported it as done.
    """
    import logging

    import squidxplorer._flatfield as ff_mod

    log_at_entry = []

    def _fake_estimate(planes, *, use_darkfield=False):
        log_at_entry.append([r.getMessage() for r in caplog.records])
        return _nonunit_profile()

    monkeypatch.setattr(ff_mod, "estimate_profile", _fake_estimate)
    caplog.set_level(logging.INFO)
    from squidxplorer._stitch import estimate_region_flatfield

    estimate_region_flatfield(_FakeReader(master), "A1", list(range(4)), channels=[0, 1],
                              z_level=0)

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
    from squidxplorer._stitch import _place_unconstrained_tiles

    positions = [(0.0, 0.0), (0.0, 100.0), (100.0, 0.0), (500.0, 500.0)]
    metrics = {(0, 1): (0.0, 100.0, 0.9), (0, 2): (100.0, 0.0, 0.9)}
    edges = [{"i": 0, "j": 1, "t": np.array([0.0, 100.0]), "w": 0.9},
             {"i": 0, "j": 2, "t": np.array([100.0, 0.0]), "w": 0.9}]
    solved = np.zeros((4, 2), dtype=np.float64)
    placed = _place_unconstrained_tiles(solved, edges, metrics, positions, (1.0, 1.0))
    assert np.allclose(placed, 0.0)


# ================================================================================ the one kwargs
# contract on the region arm (2026-08-15): declared params bind, `accepts` passes through, and
# anything else is a NAMED refusal — the same invariant Operator.with_params always held on the
# plane arm.


def test_the_region_arm_refuses_an_unknown_parameter_BY_NAME():
    """`**operator_kwargs` used to splat past Operator.bind, so a typo ran at the defaults."""
    from squidxplorer import run_plate

    stream = run_plate(None, operator="stitch", operator_kwargs={"bogus": 1})
    with pytest.raises(ValueError, match="has no parameter 'bogus'"):
        next(iter(stream))


def test_a_region_operator_refuses_an_n_fovs_crop_and_names_the_mapping():
    """`n_fovs` used to be silently DISCARDED on the region arm (`n_fovs=None` hardcoded)."""
    from squidxplorer import run_plate

    with pytest.raises(ValueError, match=r"regions=\{region: \[fov, \.\.\.\]\}"):
        run_plate(None, operator="stitch", n_fovs=2)


def test_write_plate_refuses_a_typoed_stitch_knob_BEFORE_any_directory_exists(tmp_path):
    """The region arm used to skip pre-flight validation, so the refusal came after the plate
    skeleton and the incomplete marker were already on disk."""
    from squidxplorer import write_plate

    class _Reader:
        metadata = {"regions": [], "fovs_per_region": {}, "channels": [], "n_z": 1}

    out = tmp_path / "out"
    with pytest.raises(ValueError, match="has no parameter 'bogus'"):
        write_plate(_Reader(), out, operator="stitch", operator_kwargs={"bogus": 1})
    assert not out.exists(), "the refusal came after the writer had already made directories"


def test_coordinate_refuses_the_registration_family_by_name():
    """`coordinate` IS registration-off; a registration knob is a contradiction, not a no-op.
    It used to be swallowed: `kwargs["register"] = False` silently overrode a caller's True."""
    from squidxplorer._engine import split_operator_kwargs

    for knob in ("register", "registration_channel", "registration_t"):
        with pytest.raises(ValueError, match=f"has no parameter '{knob}'"):
            split_operator_kwargs("coordinate", {knob: 1})


def test_the_writer_asks_the_RECORD_for_output_depth_and_kind():
    """`write_plate` used to reconstruct stitch's declaration from the literal string
    "z_operator" and a hardcoded "mip" — a third-party region operator got a mip-shaped
    pyramid. The record answers now, off `inner_param` and the declared default."""
    from squidxplorer._engine import operator_output

    assert operator_output("mip") == (True, "intensity")
    assert operator_output("stitch") == (True, "intensity")          # declared default: mip
    assert operator_output("stitch", {"z_operator": "keepz"}) == (False, "intensity")
    assert operator_output("coordinate") == (True, "intensity")
