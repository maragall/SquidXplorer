"""Color-camera acquisitions: uint8 .bmp/.png planes, including (Y, X, 3) RGB."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from squidxplorer import open_reader

_ACQ_YAML = """\
objective:
  pixel_size_um: 0.418
z_stack:
  nz: 1
time_series:
  nt: 1
"""

_CH_YAML = """\
channels:
- name: BF LED matrix full
  display_color: '#FFFFFF'
"""


def _write(root, name, arr):
    root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(root / name)


def _sidecars(root):
    (root / "acquisition.yaml").write_text(_ACQ_YAML)
    (root / "acquisition_channels.yaml").write_text(_CH_YAML)


def _gray(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (16, 16), dtype=np.uint8)


def test_a_grayscale_bmp_acquisition_opens(tmp_path):
    """The real failing case: uint8 grayscale .bmp planes (RGB2GRAY color-camera output)."""
    root = tmp_path / "acq"
    for fov in (0, 1):
        _write(root / "0", f"manual_{fov}_0_BF_LED_matrix_full.bmp", _gray(fov))
    _sidecars(root)
    r = open_reader(root)
    m = r.metadata
    assert [c["name"] for c in m["channels"]] == ["BF_LED_matrix_full"]
    assert m["frame_shape"] == (16, 16) and m["dtype"] == np.uint8
    assert np.array_equal(r.read("manual", 1, "BF_LED_matrix_full", 0), _gray(1))


def test_an_rgb_bmp_becomes_three_primary_tinted_channels(tmp_path):
    """(Y, X, 3) color splits into (R)/(G)/(B) channels whose display colors are the pure primaries — additively blended they ARE the original color, and"""
    rgb = np.stack([_gray(1), _gray(2), _gray(3)], axis=-1)
    root = tmp_path / "acq"
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", rgb)
    _sidecars(root)
    r = open_reader(root)
    m = r.metadata
    names = [c["name"] for c in m["channels"]]
    assert names == ["BF_LED_matrix_full (R)", "BF_LED_matrix_full (G)", "BF_LED_matrix_full (B)"]
    assert [c["display_color"] for c in m["channels"]] == ["#FF0000", "#00FF00", "#0000FF"]
    assert m["frame_shape"] == (16, 16), "frame_shape is spatial only, never the color axis"
    for i, name in enumerate(names):
        plane = r.read("manual", 0, name, 0)
        assert plane.ndim == 2 and np.array_equal(plane, rgb[..., i]), name
    assert r.plane_path("manual", 0, names[0], 0).name == "manual_0_0_BF_LED_matrix_full.bmp"
    assert [c.get("color_source") for c in m["channels"]] == ["file", "file", "file"]
    with pytest.raises(ValueError, match="color plane"):
        r.read("manual", 0, "BF_LED_matrix_full", 0)


def test_mixed_color_and_mono_channels_coexist(tmp_path):
    """Squid writes color BF beside mono fluorescence in one acquisition — same dtype, two plane shapes; the color one expands, the mono one is untouched."""
    root = tmp_path / "acq"
    rgb = np.stack([_gray(1), _gray(2), _gray(3)], axis=-1)
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", rgb)
    _write(root / "0", "manual_0_0_Fluorescence_405_nm_Ex.bmp", _gray(9))
    _sidecars(root)
    m = open_reader(root).metadata
    names = [c["name"] for c in m["channels"]]
    assert "Fluorescence_405_nm_Ex" in names and "BF_LED_matrix_full (R)" in names
    assert len(names) == 4


def test_colormap_prefers_the_acquisitions_display_color_else_the_name_palette():
    """The RGB component channels carry pure primaries in their resolved display_color; the window's colormap must read THAT, not the wavelength palette"""
    pytest.importorskip("napari")
    from squidxplorer._acquisition import DisplayChannel
    from squidxplorer._channels import fallback_color
    from squidxplorer._napari_pane import _colormap_for

    channels = [DisplayChannel(name="BF_LED_matrix_full (R)",
                               display_name="BF LED matrix full (R)",
                               display_color="#FF0000")]
    cm = _colormap_for("BF_LED_matrix_full (R)", channels)
    assert tuple(np.asarray(cm.colors)[-1][:3]) == (1.0, 0.0, 0.0)
    cm2 = _colormap_for("BF LED matrix full (R)", channels)
    assert tuple(np.asarray(cm2.colors)[-1][:3]) == (1.0, 0.0, 0.0)

    cm = _colormap_for("Fluorescence_488_nm_Ex", None)
    h = fallback_color("Fluorescence_488_nm_Ex").lstrip("#")
    want = tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    got = tuple(float(v) for v in np.asarray(cm.colors)[-1][:3])
    assert got == pytest.approx(want)
    assert _colormap_for("BF_LED_matrix_full (R)", None) == "gray"


def test_rgb_components_seed_the_files_full_range(tmp_path):
    """All three primaries share the FILE's own range: per-channel percentiles would tint the additive reconstruction and read as 'completely dark' on brightfield."""
    root = tmp_path / "acq"
    rgb = np.stack([_gray(1), _gray(2), _gray(3)], axis=-1)
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", rgb)
    _sidecars(root)
    r = open_reader(root)
    _ = r.metadata
    assert r.is_rgb_component("BF_LED_matrix_full (R)")
    assert not r.is_rgb_component("BF_LED_matrix_full")     # the on-disk base is not virtual

    from squidxplorer._workers import _MosaicWorker

    w = _MosaicWorker.__new__(_MosaicWorker)                # _seed_window needs no QThread state
    w._reader, w._meta = r, r.metadata
    assert w._seed_window("BF_LED_matrix_full (G)", None, lambda *a: (9.0, 10.0)) == (0.0, 255.0)
    assert w._seed_window("BF_LED_matrix_full", None, lambda *a: (9.0, 10.0)) == (9.0, 10.0)


def test_the_plate_windows_rgb_components_at_the_files_full_range(tmp_path, qapp):
    """The plate composites additively; per-channel percentile windows distort the hue the triplet exists to reconstruct (measured live: dark red/teal plate"""
    import squidxplorer._viewer as V

    root = tmp_path / "acq"
    rgb = np.stack([_gray(1), _gray(2), _gray(3)], axis=-1)
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", rgb)
    (root / "coordinates.csv").write_text("region,fov,z_level,x (mm),y (mm)\nmanual,0,0,1.0,2.0\n")
    _sidecars(root)
    win = V.PlateWindow(None)
    try:
        win.ingest(str(root))
        wins = win._overview.channel_windows()
        assert wins == [(0.0, 255.0)] * 3, wins
        win._overview._contrast.add(0, np.full((8, 8), 40, dtype=np.uint8))
        assert win._overview.channel_windows()[0] == (0.0, 255.0)
    finally:
        win._stop_worker()
        win.close()


def _mosaic_sidecar(root, k_r=0.3, k_b=0.6):
    """Squid's colored overview: a PNG with a known stain + the yaml calling the channel RGB."""
    mv = root / "0" / "mosaic_view"
    mv.mkdir(parents=True, exist_ok=True)
    t = np.tile(np.linspace(0.2, 1.0, 200), (120, 1))
    png = np.stack([255 * t ** k_r, 255 * t, 255 * t ** k_b], axis=-1).astype(np.uint8)
    Image.fromarray(png).save(mv / "mosaic_2um_x.png")
    (mv / "mosaic_2um.yaml").write_text(
        "rgb_channel_names:\n- 20x BF LED matrix full\nrgb_view_files:\n- mosaic_2um_x.png\n")


def test_a_color_channel_recorded_gray_is_detected_and_gets_the_stain_lut(tmp_path):
    """The mosaic yaml says the channel was RGB live; its files are 2-D: the display gets the stain colormap measured from the overview PNG."""
    root = tmp_path / "acq"
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", _gray(7))
    _sidecars(root)
    _mosaic_sidecar(root)
    from squidxplorer._napari_pane import _colormap_for

    chs = open_reader(root).metadata["channels"]
    lut = chs[0].get("display_lut")
    assert lut is not None and len(lut) == 256
    assert lut[-1] == (1.0, 1.0, 1.0)                       # background stays white
    assert abs(lut[128][0] - 0.5 ** 0.3) < 0.05             # the measured k_R reaches the LUT
    assert abs(lut[128][2] - 0.5 ** 0.6) < 0.05
    assert chs[0].get("color_source") == "estimated"
    pytest.importorskip("napari")
    cm = _colormap_for(chs[0]["name"], chs)
    assert len(np.asarray(cm.colors)) == 256


def test_without_an_overview_the_gray_channel_stays_gray_with_no_color_source(tmp_path):
    root = tmp_path / "acq"
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", _gray(7))
    _sidecars(root)
    chs = open_reader(root).metadata["channels"]
    assert chs[0].get("color_source") is None and chs[0].get("display_lut") is None


def test_a_real_rgb_acquisition_expands_and_takes_no_stain_lut(tmp_path):
    root = tmp_path / "acq"
    rgb = np.stack([_gray(1), _gray(2), _gray(3)], axis=-1)
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", rgb)
    _sidecars(root)
    _mosaic_sidecar(root)
    chs = open_reader(root).metadata["channels"]
    assert [c["name"] for c in chs] == [f"BF_LED_matrix_full ({x})" for x in "RGB"]
    assert all(c.get("display_lut") is None for c in chs)   # true primaries need no model


def test_reconstruction_is_switched_off_by_the_env_flag_or_the_live_override(tmp_path, monkeypatch):
    """SQUIDXPLORER_NO_RECONSTRUCTED_COLOR=1, or set_reconstruction(False): no stain LUT, the mono channel with its yaml color; file color is untouched."""
    from squidxplorer import _stain

    root = tmp_path / "acq"
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", _gray(7))
    _sidecars(root)
    _mosaic_sidecar(root)
    try:
        _stain.set_reconstruction(False)
        chs = open_reader(root).metadata["channels"]
        assert chs[0].get("display_lut") is None and chs[0].get("color_source") is None
        _stain.set_reconstruction(True)
        chs = open_reader(root).metadata["channels"]
        assert chs[0].get("display_lut") is not None
        assert chs[0].get("color_source") == "estimated"
    finally:
        _stain.set_reconstruction(None)

    monkeypatch.setenv("SQUIDXPLORER_NO_RECONSTRUCTED_COLOR", "1")
    chs = open_reader(root).metadata["channels"]
    assert len(chs) == 1
    assert chs[0].get("display_lut") is None and chs[0].get("color_source") is None
    assert chs[0]["display_color"] == "#FFFFFF"

    rgb_root = tmp_path / "acq_rgb"
    _write(rgb_root / "0", "manual_0_0_BF_LED_matrix_full.bmp",
           np.stack([_gray(1), _gray(2), _gray(3)], axis=-1))
    _sidecars(rgb_root)
    rgb_chs = open_reader(rgb_root).metadata["channels"]
    assert [c.get("color_source") for c in rgb_chs] == ["file", "file", "file"]


def test_color_note_names_each_provenance_once():
    """The shared sentence the window says and the tree pins; silent for plain channels."""
    from squidxplorer._acquisition import DisplayChannel
    from squidxplorer._channels import color_note, color_sources

    def ch(source):
        return DisplayChannel(name="x", display_name="x", display_color="#FFFFFF",
                              color_source=source)

    assert color_note([ch(None)]) is None and color_sources([ch(None)]) == []
    note = color_note([ch("estimated"), ch("file"), ch("file")])
    assert note.startswith("color: ") and "estimated colormap" in note and "file color" in note
    assert color_sources([ch("file"), ch("estimated")]) == ["estimated", "file"]


# --- Virtual chroma components (color-recorded-gray + overview geometry) -----------------------

_ACQ_YAML_2UM = """\
objective:
  pixel_size_um: 2.0
z_stack:
  nz: 1
time_series:
  nt: 1
"""


def _chroma_sidecar(root, png, top_left_mm_yx, resolution_um=2.0):
    """A mosaic_view overview WITH geometry: what makes chroma expansion possible."""
    mv = root / "0" / "mosaic_view"
    mv.mkdir(parents=True, exist_ok=True)
    Image.fromarray(png).save(mv / "mosaic_2um_x.png")
    h, w = png.shape[:2]
    y0, x0 = top_left_mm_yx
    res_mm = resolution_um / 1000.0
    (mv / "mosaic_2um.yaml").write_text(
        f"resolution_um: {resolution_um}\n"
        "full:\n"
        f"  top_left_mm:\n  - {y0}\n  - {x0}\n"
        f"  extents_mm:\n  - {y0}\n  - {y0 + h * res_mm}\n  - {x0}\n  - {x0 + w * res_mm}\n"
        "rgb_channel_names:\n- 20x BF LED matrix full\n"
        "rgb_view_files:\n- mosaic_2um_x.png\n")


def _chroma_acq(tmp_path, fov_centers_mm, planes=None):
    """A gray acquisition at 2 um/px (1:1 with the overview) with stage positions."""
    root = tmp_path / "acq"
    lines = ["region,x (mm),y (mm),z (mm)"]
    for fov, (x_mm, y_mm) in enumerate(fov_centers_mm):
        plane = _gray(fov) if planes is None else planes[fov]
        _write(root / "0", f"manual_{fov}_0_BF_LED_matrix_full.bmp", plane)
        lines.append(f"manual,{x_mm},{y_mm},")
    (root / "coordinates.csv").write_text("\n".join(lines) + "\n")
    (root / "0" / "coordinates.csv").write_text("\n".join(lines) + "\n")
    (root / "acquisition.yaml").write_text(_ACQ_YAML_2UM)
    (root / "acquisition_channels.yaml").write_text(_CH_YAML)
    return root


def _flat_png(shape, r, g, b):
    png = np.zeros(shape + (3,), dtype=np.uint8)
    png[..., 0], png[..., 1], png[..., 2] = r, g, b
    return png


def test_chroma_components_reconstruct_the_overviews_known_ratios(tmp_path):
    """With geometry + positions, a color-recorded-gray channel expands into (R)/(G)/(B): (G) is the file's own plane, (R)/(B) scale it by the PNG's local"""
    plane = np.full((16, 16), 80, dtype=np.uint8)
    png = _flat_png((40, 60), r=150, g=100, b=50)                # ratios 1.5 and 0.5
    root = _chroma_acq(tmp_path, [(0.056, 0.036)], planes=[plane])
    _chroma_sidecar(root, png, top_left_mm_yx=(0.0, 0.0))
    r = open_reader(root)
    names = [c["name"] for c in r.metadata["channels"]]
    assert names == [f"BF_LED_matrix_full ({x})" for x in "RGB"]
    assert np.array_equal(r.read("manual", 0, names[1], 0), plane)          # G untouched
    assert np.array_equal(r.read("manual", 0, names[0], 0),
                          np.full((16, 16), 120, dtype=np.uint8))           # 80 * 1.5
    assert np.array_equal(r.read("manual", 0, names[2], 0),
                          np.full((16, 16), 40, dtype=np.uint8))            # 80 * 0.5
    assert r.is_rgb_component(names[0]) and not r.is_rgb_component("BF_LED_matrix_full")


def test_chroma_is_neutral_outside_coverage_and_logged_once(tmp_path, caplog):
    """A FOV the PNG does not cover reads neutral gray (ratio 1.0) — R equals the file's own plane — and the shortfall is said in ONE log line naming how"""
    import logging

    plane = np.full((16, 16), 80, dtype=np.uint8)
    png = _flat_png((40, 60), r=150, g=100, b=50)
    root = _chroma_acq(tmp_path, [(0.056, 0.036), (9.0, 9.0)], planes=[plane, plane])
    _chroma_sidecar(root, png, top_left_mm_yx=(0.0, 0.0))
    r = open_reader(root)
    with caplog.at_level(logging.INFO, logger="squidxplorer.reader"):
        _ = r.metadata
    assert np.array_equal(r.read("manual", 1, "BF_LED_matrix_full (R)", 0), plane)
    assert np.array_equal(r.read("manual", 0, "BF_LED_matrix_full (R)", 0),
                          np.full((16, 16), 120, dtype=np.uint8))
    lines = [m for m in caplog.messages if "lack coverage" in m]
    assert len(lines) == 1 and "1 of 2 FOV(s)" in lines[0]


def test_chroma_partial_coverage_is_neutral_only_where_uncovered(tmp_path):
    """A window half off the PNG keeps real chroma on the covered half, neutral on the rest."""
    plane = np.full((16, 16), 80, dtype=np.uint8)
    png = _flat_png((40, 60), r=150, g=100, b=50)
    root = _chroma_acq(tmp_path, [(0.008, 0.036)], planes=[plane])
    _chroma_sidecar(root, png, top_left_mm_yx=(0.0, 0.0))
    r = open_reader(root)
    out = r.read("manual", 0, "BF_LED_matrix_full (R)", 0)
    assert np.array_equal(out[:, -4:], np.full((16, 4), 120, dtype=np.uint8))   # covered edge
    assert np.array_equal(out[:, :2], plane[:, :2])                             # off the PNG


def test_a_chroma_active_channel_takes_no_stain_lut(tmp_path):
    """Where chroma expansion is active the components carry real color; a display LUT on top would double-tint, so none is attached."""
    t = np.tile(np.linspace(0.2, 1.0, 60), (40, 1))
    png = np.stack([255 * t ** 0.3, 255 * t, 255 * t ** 0.6], axis=-1).astype(np.uint8)
    root = _chroma_acq(tmp_path, [(0.056, 0.036)])
    _chroma_sidecar(root, png, top_left_mm_yx=(0.0, 0.0))
    chs = open_reader(root).metadata["channels"]
    assert [c["name"] for c in chs] == [f"BF_LED_matrix_full ({x})" for x in "RGB"]
    assert all(c.get("display_lut") is None for c in chs)


def test_chroma_needs_geometry_else_the_lut_fallback_stands(tmp_path):
    """The same overview WITHOUT resolution_um/top_left_mm cannot place a FOV: no expansion, and the stain LUT is attached exactly as before."""
    root = _chroma_acq(tmp_path, [(0.056, 0.036)])
    _mosaic_sidecar(root)                        # geometry-less yaml, fittable stain PNG
    chs = open_reader(root).metadata["channels"]
    assert [c["name"] for c in chs] == ["BF_LED_matrix_full"]
    assert chs[0].get("display_lut") is not None


def test_a_stain_lut_channel_seeds_the_zero_to_white_window(tmp_path):
    """Fix 2: a channel displayed through the stain LUT windows [0, white] — white being the LUT fit's own percentile — because the LUT's t is"""
    root = tmp_path / "acq"
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", _gray(7))
    _sidecars(root)
    _mosaic_sidecar(root)
    r = open_reader(root)
    meta = r.metadata
    assert meta["channels"][0].get("display_lut") is not None

    from squidxplorer._stain import STAIN_WHITE_PERCENTILE
    from squidxplorer._workers import _MosaicWorker

    w = _MosaicWorker.__new__(_MosaicWorker)
    w._reader, w._meta = r, meta
    plane = _gray(7).astype(np.uint16) * 3
    lo, hi = w._seed_window("BF_LED_matrix_full", [plane], lambda *a: (9.0, 10.0))
    assert lo == 0.0 and hi == pytest.approx(float(np.percentile(plane, STAIN_WHITE_PERCENTILE)))
    root2 = tmp_path / "acq2"
    _write(root2 / "0", "manual_0_0_BF_LED_matrix_full.bmp", _gray(7))
    _sidecars(root2)
    r2 = open_reader(root2)
    w2 = _MosaicWorker.__new__(_MosaicWorker)
    w2._reader, w2._meta = r2, r2.metadata
    assert w2._seed_window("BF_LED_matrix_full", [plane], lambda *a: (9.0, 10.0)) == (9.0, 10.0)


def test_chroma_neutrality_and_ratios_survive_fractional_resampling(tmp_path):
    """The ratio window is upsampled with fractional bilinear weights (real data: 2 um chroma over 0.42 um pixels)."""
    plane = np.full((16, 16), 80, dtype=np.uint8)
    png = _flat_png((40, 60), r=150, g=100, b=50)
    root = _chroma_acq(tmp_path, [(0.056, 0.036), (9.0, 9.0)], planes=[plane, plane])
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.5\nz_stack:\n  nz: 1\ntime_series:\n  nt: 1\n")
    _chroma_sidecar(root, png, top_left_mm_yx=(0.0, 0.0))
    r = open_reader(root)
    assert np.array_equal(r.read("manual", 1, "BF_LED_matrix_full (R)", 0), plane)  # neutral EXACT
    assert np.array_equal(r.read("manual", 0, "BF_LED_matrix_full (R)", 0),
                          np.full((16, 16), 120, dtype=np.uint8))                   # 80 * 1.5
