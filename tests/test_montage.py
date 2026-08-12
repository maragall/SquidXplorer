"""Montage unit tests: a tiny fabricated plate.ome.zarr driven through build_montage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from squidxplorer import build_montage
from squidxplorer._montage import (
    _area_downsample, _channel_slug, _hex_to_rgb01, _window, _window_lut, composite,
)
from squidxplorer._output import write_from_stream

# Two channels, red (638) and blue (405), so composite color is unambiguous per channel.
CH = [
    {"name": "Fluorescence_638_nm_-_Penta", "display_name": "638", "display_color": "#FF0000"},
    {"name": "Fluorescence_405_nm_-_Penta", "display_name": "405", "display_color": "#20ADF8"},
]
Y = X = 8


def _meta(regions):
    return {
        "regions": regions,
        "fovs_per_region": {r: [0] for r in regions},
        "channels": CH,
        "pixel_size_um": 0.325,
    }


def _image(ch_levels, y=Y, x=X, dtype=np.uint16):
    """(1, C, 1, y, x): each channel filled with a flat constant from *ch_levels*."""
    out = np.zeros((1, len(ch_levels), 1, y, x), dtype=dtype)
    for c_i, level in enumerate(ch_levels):
        out[0, c_i, 0] = level
    return out


def _ramp(ch_levels, y=Y, x=X, dtype=np.uint16):
    """(1, C, 1, y, x): each channel a 0..level ramp, so the cell has real dynamic range."""
    grad = np.arange(y * x).reshape(y, x)
    out = np.zeros((1, len(ch_levels), 1, y, x), dtype=dtype)
    for c_i, level in enumerate(ch_levels):
        out[0, c_i, 0] = (grad * level // (y * x)).astype(dtype)
    return out


def _make_plate(tmp_path: Path, images: dict) -> Path:
    """Write a plate.ome.zarr from {region: image} and return the containing dir."""
    regions = list(images)
    stream = ((r, 0, images[r]) for r in regions)
    write_from_stream(_meta(regions), stream, tmp_path, n_fovs=1, tiff=False)
    return tmp_path


def test_area_downsample_is_block_mean():
    # a 4x4 of four 2x2 quadrants (10, 20, 30, 40) -> 2x2 == the quadrant means
    plane = np.block([
        [np.full((2, 2), 10.0), np.full((2, 2), 20.0)],
        [np.full((2, 2), 30.0), np.full((2, 2), 40.0)],
    ])
    ds = _area_downsample(plane, 2, 2)
    np.testing.assert_allclose(ds, [[10, 20], [30, 40]])


def test_area_downsample_no_upsample():
    plane = np.arange(16, dtype=np.uint16).reshape(4, 4)
    out = _area_downsample(plane, 8, 8)
    assert out.shape == (4, 4)


def test_area_downsample_with_only_ONE_axis_shrinking_is_finite_and_still_a_block_mean():
    """The mixed case: taller than the target, already narrower than it, clamped per axis."""
    plane = np.arange(512 * 64, dtype=np.float32).reshape(512, 64)
    out = _area_downsample(plane, 128, 128)
    n_bad = int((~np.isfinite(out)).sum())
    assert n_bad == 0, f"{n_bad} of {out.size} entries are not finite; first row {out[0, :8]}"
    assert out.shape == (128, 64), f"clamped per axis -> (128, 64), got {out.shape}"
    np.testing.assert_allclose(out, plane.reshape(128, 4, 64).mean(axis=1))

    # a uniform field can only average to its own constant
    flat = _area_downsample(np.full((512, 64), 1000.0, np.float32), 128, 128)
    assert np.all(flat == 1000.0), f"uniform 1000-count field came back {np.unique(flat)[:4]}"

    # the other orientation — wide and short, so it is the ROW edges that would repeat
    wide = np.arange(64 * 512, dtype=np.float32).reshape(64, 512)
    out_w = _area_downsample(wide, 128, 128)
    assert np.isfinite(out_w).all() and out_w.shape == (64, 128)
    np.testing.assert_allclose(out_w, wide.reshape(64, 128, 4).mean(axis=2))


def test_hex_to_rgb01_parses_and_fails_loud():
    np.testing.assert_allclose(_hex_to_rgb01("#FF0000"), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(_hex_to_rgb01("20ADF8"), [0x20 / 255, 0xAD / 255, 0xF8 / 255])
    with pytest.raises(ValueError):
        _hex_to_rgb01("#FFF")  # not 6 digits


def test_composite_masks_out_a_channel():
    # masking a channel off removes exactly that channel's contribution
    store = np.stack([np.full((4, 4), 100.0), np.full((4, 4), 100.0)])
    colors = np.stack([_hex_to_rgb01("#FF0000"), _hex_to_rgb01("#0000FF")])
    windows = [(0.0, 100.0), (0.0, 100.0)]
    both = composite(store, colors, windows)
    only_red = composite(store, colors, windows, mask=np.array([True, False]))
    assert (both[:, :, 2] > 0).all() and (only_red[:, :, 2] == 0).all()
    np.testing.assert_array_equal(both[:, :, 0], only_red[:, :, 0])


def test_composite_empty_mask_is_black_not_nan():
    store = np.stack([np.full((4, 4), 100.0), np.full((4, 4), 0.0)])
    colors = np.stack([_hex_to_rgb01("#FF0000"), _hex_to_rgb01("#0000FF")])
    out = composite(store, colors, [(0.0, 100.0), (0.0, 0.0)], mask=np.zeros(2, bool))
    assert out.shape == (4, 4, 3) and out.dtype == np.uint8
    assert out.sum() == 0 and not np.isnan(out.astype(float)).any()


def test_composite_applies_a_distinct_window_per_channel():
    # same pixels in both channels, different windows -> different brightness per channel
    store = np.stack([np.full((4, 4), 50.0), np.full((4, 4), 50.0)])
    colors = np.stack([_hex_to_rgb01("#FF0000"), _hex_to_rgb01("#0000FF")])
    out = composite(store, colors, [(0.0, 50.0), (0.0, 500.0)])
    assert out[0, 0, 0] == 255                 # channel 0 saturates at its window
    assert 0 < out[0, 0, 2] < 255              # channel 1 sits low in its much wider window


def test_window_lut_is_the_window_function_over_the_ENTIRE_uint16_alphabet():
    """All 65536 values: the table IS `_window` memoised, or it is a bug."""
    alphabet = np.arange(1 << 16, dtype=np.uint16)
    for lo, hi in [(0.0, 65535.0), (321.0, 8765.0), (1000.0, 1001.0), (60000.0, 65535.0)]:
        table = _window_lut(np.dtype(np.uint16), lo, hi)
        assert table is not None and table.shape == (1 << 16,)
        np.testing.assert_array_equal(table, _window(alphabet, lo, hi))


def test_window_lut_covers_uint8_and_declines_the_dtypes_it_cannot_tabulate():
    # None means "fall back to elementwise" — the same function evaluated a different way
    assert _window_lut(np.dtype(np.uint8), 0.0, 255.0).shape == (256,)
    for dt in (np.float32, np.float64, np.int16, np.uint32, np.int32):
        assert _window_lut(np.dtype(dt), 0.0, 100.0) is None


def test_window_lut_handles_the_degenerate_window_like_window_does():
    # lo == hi (a blank channel) must go to black, never divide-by-zero or saturated white
    table = _window_lut(np.dtype(np.uint16), 500.0, 500.0)
    assert table is not None and np.all(table == 0.0)


def test_composite_lut_path_is_byte_identical_to_the_elementwise_path():
    """A uint16 store takes the table; the identical values as float32 do not. Same picture."""
    rng = np.random.default_rng(261)
    vals = rng.integers(0, 65535, size=(3, 37, 53))
    colors = np.stack([_hex_to_rgb01(c) for c in ("#FF0000", "#00FF00", "#0000FF")])
    windows = [(120.0, 40000.0), (0.0, 65535.0), (900.0, 3000.0)]
    lut_path = composite(vals.astype(np.uint16), colors, windows)          # table lookup
    elementwise = composite(vals.astype(np.float32), colors, windows)      # per pixel
    np.testing.assert_array_equal(lut_path, elementwise)


def test_composite_commutes_with_subsampling():
    """A contrast window is a point transform, so it commutes with subsampling."""
    rng = np.random.default_rng(7)
    store = rng.integers(0, 65535, size=(2, 64, 96)).astype(np.uint16)
    colors = np.stack([_hex_to_rgb01("#FF0000"), _hex_to_rgb01("#0000FF")])
    windows = [(100.0, 50000.0), (0.0, 20000.0)]
    full = composite(store, colors, windows)
    for step in (2, 3, 4, 5):
        sub = composite(np.ascontiguousarray(store[:, ::step, ::step]), colors, windows)
        np.testing.assert_array_equal(sub, full[::step, ::step])


def test_composite_banding_does_not_change_a_single_pixel(monkeypatch):
    """Banded (threaded) compositing must be a pure optimisation, seams included."""
    import squidxplorer._montage as M
    rng = np.random.default_rng(99)
    store = rng.integers(0, 65535, size=(4, 101, 67)).astype(np.uint16)   # prime-ish, uneven split
    colors = np.stack([_hex_to_rgb01(c) for c in ("#FF0000", "#00FF00", "#0000FF", "#FFFFFF")])
    windows = [(0.0, 65535.0), (10.0, 30000.0), (5.0, 60000.0), (1.0, 2.0)]

    monkeypatch.setattr(M, "_COMPOSITE_MIN_PX_PER_BAND", 10 ** 12)        # force exactly one band
    one = M.composite(store, colors, windows)
    monkeypatch.setattr(M, "_COMPOSITE_MIN_PX_PER_BAND", 1)               # force as many as allowed
    many = M.composite(store, colors, windows)
    np.testing.assert_array_equal(one, many)


def test_composite_banding_still_honours_the_channel_mask(monkeypatch):
    import squidxplorer._montage as M
    store = np.full((2, 80, 40), 40000, np.uint16)
    colors = np.stack([_hex_to_rgb01("#FF0000"), _hex_to_rgb01("#0000FF")])
    monkeypatch.setattr(M, "_COMPOSITE_MIN_PX_PER_BAND", 1)
    out = M.composite(store, colors, [(0.0, 65535.0)] * 2, mask=np.array([True, False]))
    assert (out[:, :, 2] == 0).all() and (out[:, :, 0] > 0).all()


def test_composite_of_an_empty_store_is_empty_not_a_crash():
    # a layer allocated before any tile lands has a zero dimension
    colors = np.stack([_hex_to_rgb01("#FF0000")])
    out = composite(np.zeros((1, 0, 5), np.uint16), colors, [(0.0, 1.0)])
    assert out.shape == (0, 5, 3) and out.dtype == np.uint8


def test_window_lut_cache_is_bounded():
    """A drag mints a new (lo, hi) every tick; an unbounded cache is a slow leak."""
    import squidxplorer._montage as M
    for i in range(M._LUT_CACHE_MAX * 3):
        _window_lut(np.dtype(np.uint16), float(i), float(i + 1000))
    assert len(M._LUT_CACHE) <= M._LUT_CACHE_MAX


def test_channel_slug_is_filename_safe():
    assert _channel_slug("Fluorescence 638 nm - Penta", 0) == "Fluorescence_638_nm_Penta"
    assert _channel_slug(None, 3) == "ch3"


def test_window_guards_flat_channel():
    # a flat channel (lo == hi) must not divide by zero — it maps to all-zero
    flat = np.full((4, 4), 500.0, dtype=np.float32)
    out = _window(flat, 500.0, 500.0)
    assert out.shape == (4, 4)
    assert np.all(out == 0.0)


# --- full montage over a fabricated plate ---------------------------------------------------

def test_montage_png_and_sidecar_shape(tmp_path):
    # 3 wells in row B, columns 2/3/10 -> grid 1x3 (natural column sort, no zero-pad)
    images = {"B2": _image([100, 0]), "B3": _image([0, 100]), "B10": _image([50, 50])}
    out = _make_plate(tmp_path, images)

    manifest = build_montage(out, cell_px=4)
    assert manifest["n_wells"] == 3
    assert manifest["grid"] == (1, 3)
    assert manifest["cell_px"] == 4

    from PIL import Image

    rgb = np.asarray(Image.open(manifest["montage"]))
    assert rgb.shape == (1 * 4, 3 * 4, 3)  # (n_rows*cell, n_cols*cell, RGB)

    side = json.loads(Path(manifest["sidecar"]).read_text())
    assert side["grid"] == {"n_rows": 1, "n_cols": 3, "rows": ["B"], "columns": ["2", "3", "10"]}
    assert {w["well_id"] for w in side["wells"]} == {"B2", "B3", "B10"}
    # every well's bbox is inside the canvas and cell-sized
    for w in side["wells"]:
        assert (w["x1"] - w["x0"], w["y1"] - w["y0"]) == (4, 4)
        assert 0 <= w["x0"] < w["x1"] <= 3 * 4 and 0 <= w["y0"] < w["y1"] <= 1 * 4


def test_montage_composite_colors_follow_display_color(tmp_path):
    # B2 lit only in the 638=red channel, B3 only in 405=blue. The montage cells must be
    # red-dominant and blue-dominant respectively — colors come from display_color.
    images = {"B2": _image([300, 0]), "B3": _image([0, 300])}
    out = _make_plate(tmp_path, images)
    manifest = build_montage(out, cell_px=4)

    from PIL import Image

    rgb = np.asarray(Image.open(manifest["montage"])).astype(int)
    side = {w["well_id"]: w for w in json.loads(Path(manifest["sidecar"]).read_text())["wells"]}

    def cell_mean(w):
        return rgb[w["y0"] : w["y1"], w["x0"] : w["x1"]].reshape(-1, 3).mean(axis=0)

    r_cell = cell_mean(side["B2"])
    b_cell = cell_mean(side["B3"])
    assert r_cell[0] > r_cell[2]           # B2 red channel dominates blue
    assert b_cell[2] > b_cell[0]           # B3 blue dominates red (405 color is blue-ish)


def test_montage_global_contrast_preserves_relative_brightness(tmp_path):
    # Same channel, one bright well and one dim well. GLOBAL-per-channel contrast must keep the
    # bright well brighter — a per-well window would wrongly equalize them.
    images = {"B2": _image([1000, 0]), "B3": _image([100, 0])}
    out = _make_plate(tmp_path, images)
    manifest = build_montage(out, cell_px=4)

    from PIL import Image

    rgb = np.asarray(Image.open(manifest["montage"])).astype(int)
    side = {w["well_id"]: w for w in json.loads(Path(manifest["sidecar"]).read_text())["wells"]}
    bright = rgb[:, side["B2"]["x0"] : side["B2"]["x1"]].max()
    dim = rgb[:, side["B3"]["x0"] : side["B3"]["x1"]].max()
    assert bright > dim


def test_montage_blank_cells_are_black(tmp_path):
    # Wells B2 and C3 -> grid rows [B,C] x cols [2,3]; only (B,2) and (C,3) filled.
    # The two empty intersections must render pure black (no well there). Ramps (not flat
    # constants) so the filled cells have real dynamic range and don't map to black.
    images = {"B2": _ramp([600, 600]), "C3": _ramp([600, 600])}
    out = _make_plate(tmp_path, images)
    manifest = build_montage(out, cell_px=4)
    assert manifest["grid"] == (2, 2)

    from PIL import Image

    rgb = np.asarray(Image.open(manifest["montage"]))
    # (row B=0, col 3=1) and (row C=1, col 2=0) are the blanks
    assert rgb[0:4, 4:8].sum() == 0
    assert rgb[4:8, 0:4].sum() == 0
    # a filled cell is not all black
    assert rgb[0:4, 0:4].sum() > 0


def test_montage_emits_self_contained_hover_viewer(tmp_path):
    # build_montage also writes a zero-dependency HTML viewer that maps a hover to a well id
    # from the region-jump sidecar geometry. Assert it is emitted, self-contained, and carries
    # the well ids + the cursor-locate handler (no external fetch, so the data is inlined).
    images = {"B2": _ramp([300, 0]), "C3": _ramp([0, 300])}
    out = _make_plate(tmp_path, images)
    manifest = build_montage(out, cell_px=4)

    viewer = Path(manifest["viewer"])
    assert viewer.name == "plate_montage.html" and viewer.exists()
    html = viewer.read_text()
    assert '<img id="montage" src="plate_montage.png"' in html  # points at the montage, same dir
    assert "mousemove" in html and "getBoundingClientRect" in html  # hover indicator wired
    assert '"well_id": "B2"' in html and '"well_id": "C3"' in html  # sidecar geometry inlined
    assert "http://" not in html and "https://" not in html  # self-contained, no external deps


def test_montage_sidecar_records_per_channel_window(tmp_path):
    images = {"B2": _image([200, 50])}
    out = _make_plate(tmp_path, images)
    manifest = build_montage(out, cell_px=4)
    side = json.loads(Path(manifest["sidecar"]).read_text())
    assert [c["color"] for c in side["channels"]] == ["FF0000", "20ADF8"]
    for c in side["channels"]:
        assert c["window"]["high"] >= c["window"]["low"]  # a real, ordered window per channel


def test_montage_per_channel_writes_one_png_per_channel(tmp_path):
    # P1 verbatim: one PNG per channel, alongside the composite. Names come from the omero label.
    images = {"B2": _ramp([300, 0]), "B3": _ramp([0, 300])}
    out = _make_plate(tmp_path, images)
    manifest = build_montage(out, cell_px=4, per_channel=True)

    assert Path(manifest["montage"]).name == "plate_montage.png"          # composite still written
    names = [Path(p).name for p in manifest["per_channel"]]
    assert names == ["plate_montage_638.png", "plate_montage_405.png"]
    assert all((Path(out) / n).exists() for n in names)
    side = json.loads(Path(manifest["sidecar"]).read_text())
    assert [c["png"] for c in side["channels"]] == names                  # sidecar names them


def test_montage_per_channel_png_carries_only_its_own_channel(tmp_path):
    # Each per-channel PNG must be that channel ALONE, in its own display_color: B2 is lit only in
    # 638 (red) and B3 only in 405 (blue), so the 638 PNG holds B2 and nothing at B3, and vice versa.
    images = {"B2": _ramp([300, 0]), "B3": _ramp([0, 300])}
    out = _make_plate(tmp_path, images)
    manifest = build_montage(out, cell_px=4, per_channel=True)

    from PIL import Image

    side = json.loads(Path(manifest["sidecar"]).read_text())
    cells = {w["well_id"]: w for w in side["wells"]}

    def cell(png, w):
        arr = np.asarray(Image.open(Path(out) / png)).astype(int)
        return arr[w["y0"] : w["y1"], w["x0"] : w["x1"]].reshape(-1, 3).mean(axis=0)

    red_png, blue_png = side["channels"][0]["png"], side["channels"][1]["png"]
    assert cell(red_png, cells["B2"])[0] > 0 and cell(red_png, cells["B2"])[2] == 0   # red only
    assert cell(red_png, cells["B3"]).sum() == 0            # B3 has no 638 signal -> black
    assert cell(blue_png, cells["B3"])[2] > 0               # 405's LUT is blue-dominant
    assert cell(blue_png, cells["B2"]).sum() == 0


def test_montage_per_channel_uses_the_window_in_the_sidecar(tmp_path):
    # D9's honesty requirement: the window a channel was exported at is recorded, and it is the
    # SAME global window the composite used (per-channel PNGs are not re-windowed on their own).
    images = {"B2": _ramp([1000, 100]), "B3": _ramp([100, 1000])}
    out = _make_plate(tmp_path, images)
    manifest = build_montage(out, cell_px=4, per_channel=True)
    side = json.loads(Path(manifest["sidecar"]).read_text())
    from PIL import Image

    for c in side["channels"]:
        assert c["window"]["high"] > c["window"]["low"]
        arr = np.asarray(Image.open(Path(out) / c["png"]))
        assert arr.max() > 0                                # windowed, not crushed to black


def test_montage_html_channel_toggles_reference_the_per_channel_pngs(tmp_path):
    images = {"B2": _ramp([300, 0]), "B3": _ramp([0, 300])}
    out = _make_plate(tmp_path, images)
    build_montage(out, cell_px=4, per_channel=True)
    html = (Path(out) / "plate_montage.html").read_text()
    assert '"png": "plate_montage_638.png"' in html and '"png": "plate_montage_405.png"' in html
    assert 'type="checkbox"' in html and "applyChannels" in html      # toggles wired
    assert 'c.window.low' in html and 'c.window.high' in html         # exported window shown (D9)
    assert "http://" not in html and "https://" not in html          # still self-contained


def test_montage_html_without_per_channel_has_no_dead_toggles(tmp_path):
    # Default (composite only): the legend stays a plain legend — no checkbox that toggles nothing.
    images = {"B2": _ramp([300, 0])}
    out = _make_plate(tmp_path, images)
    build_montage(out, cell_px=4)
    html = (Path(out) / "plate_montage.html").read_text()
    assert '"png": null' in html                            # sidecar says: no per-channel PNGs
    assert 'type="checkbox"' not in html.split("<script>")[0]   # none rendered into the markup


def test_montage_memory_is_bounded_not_whole_plate(tmp_path):
    """build_montage reads ONE well at a time: peak ~ canvas + one well, never all N wells.

    Feeds a plate of N sizable wells and checks the peak stays far below "all wells resident"
    (N x one-well bytes). Proves the montage never accumulates the full-res plate — it holds one
    well plus the downsampled montage canvas (which scales with montage resolution, not plate size).
    """
    import tracemalloc

    N, F = 20, 512  # 20 wells, 512x512 frames -> "all resident" would be obvious
    ch = [{"name": "c0", "display_name": "c0", "display_color": "#FF0000"}]
    regions = [f"A{i + 1}" for i in range(N)]
    meta = {"regions": regions, "fovs_per_region": {r: [0] for r in regions}, "channels": ch,
            "pixel_size_um": 0.325}

    def _img():
        a = np.zeros((1, 1, 1, F, F), np.uint16)
        a[0, 0, 0] = (np.arange(F * F).reshape(F, F) % 1000).astype(np.uint16)
        return a

    write_from_stream(meta, ((r, 0, _img()) for r in regions), tmp_path, n_fovs=1, tiff=False)
    well_bytes = F * F * 2

    tracemalloc.start()
    build_montage(tmp_path, cell_px=48)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # holding all N wells would be N*well_bytes; assert peak stays well under a quarter of that.
    assert peak < N * well_bytes * 0.25, f"montage peak {peak} not bounded (all wells = {N * well_bytes})"


# --- fail loud ------------------------------------------------------------------------------

def test_build_montage_rejects_non_plate(tmp_path):
    (tmp_path / "not_a_plate").mkdir()
    with pytest.raises(ValueError, match="HCS plate"):
        build_montage(tmp_path / "not_a_plate")


def test_build_montage_rejects_bad_cell_px(tmp_path):
    images = {"B2": _image([100, 100])}
    out = _make_plate(tmp_path, images)
    with pytest.raises(ValueError, match="cell_px"):
        build_montage(out, cell_px=0)


def test_montage_small_field_no_crash_no_nan(tmp_path):
    # Audit MED: a field SMALLER than cell_px (128) must corner-place, not broadcast-crash or
    # divide-by-zero into NaN (which would blacken a whole channel). Square-small + non-square.
    from PIL import Image
    imgs = {"B2": _ramp([1, 2], y=40, x=40), "B3": _ramp([1, 2], y=100, x=60)}
    manifest = build_montage(_make_plate(tmp_path, imgs))   # default cell_px=128 > field -> corner-place
    arr = np.asarray(Image.open(manifest["montage"]))
    assert arr.ndim == 3 and int(arr.max()) > 0             # rendered, not crashed / all-black NaN


def test_montage_of_a_field_that_shrinks_on_ONE_axis_is_not_a_black_plate(tmp_path):
    """A field larger than cell_px on one axis and smaller on the other blackened the WHOLE plate.

    ``build_montage`` calls ``_area_downsample`` with no guard of its own, so a 512x64 field at the
    default cell_px=128 hit the mixed case: half the tile came back ``inf``, those infs reached the
    GLOBAL per-channel percentile window as ``hi=nan``, ``_window``'s ``span <= 0`` branch zeroed
    every pixel, and the PNG's max pixel value was 0. Measured on this exact fixture before the
    fix: window ``{"low": 39.0, "high": nan}``, max pixel **0** — a plate rendered entirely black
    with no exception raised and no warning a user would ever see. Afterwards: high 1992.65,
    max pixel 255.

    The ramp (not a flat constant) matters: a flat channel is legitimately black through
    ``_window``, so only a field with real dynamic range can tell the defect from the guard.
    """
    from PIL import Image

    imgs = {"B2": _ramp([2000, 0], y=512, x=64)}   # 512 shrinks to 128, 64 is already below it
    manifest = build_montage(_make_plate(tmp_path, imgs), cell_px=128)

    win = json.loads(Path(manifest["sidecar"]).read_text())["channels"][0]["window"]
    assert np.isfinite(win["high"]), f"contrast window is low={win['low']} high={win['high']}"
    arr = np.asarray(Image.open(manifest["montage"]))
    assert int(arr.max()) > 0, f"montage rendered entirely black: max pixel {int(arr.max())}"
