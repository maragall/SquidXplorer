"""Color-camera acquisitions: uint8 .bmp/.png planes, including (Y, X, 3) RGB.

Squid picks the extension by DTYPE (utils_acquisition.get_image_filepath): uint16 -> .tiff,
everything else -> IMAGE_FORMAT (.bmp default) — the color-camera path. Measured on the real set
`_2026-08-13_18-07-54.442667` (22 uint8 grayscale BMPs), which the reader previously refused as
"contains no ...tiff".

A COLOR plane is served as three channels tinted pure R/G/B: additive blending reconstructs the
file's exact color on screen, and every operator, fuser and cache keeps receiving the 2-D
grayscale planes the whole pipeline is written for. No second render path.
"""

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
    """(Y, X, 3) color splits into (R)/(G)/(B) channels whose display colors are the pure
    primaries — additively blended they ARE the original color, and read() hands each consumer
    the 2-D plane the pipeline contract promises."""
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


def test_reading_a_color_channel_by_its_base_name_fails_by_name(tmp_path):
    """The base name is not a channel once expanded — half-supporting it would hand a 3-D array
    to consumers whose contract is 2-D."""
    rgb = np.stack([_gray(1), _gray(2), _gray(3)], axis=-1)
    root = tmp_path / "acq"
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", rgb)
    _sidecars(root)
    r = open_reader(root)
    _ = r.metadata
    with pytest.raises(ValueError, match="color plane"):
        r.read("manual", 0, "BF_LED_matrix_full", 0)


def test_mixed_color_and_mono_channels_coexist(tmp_path):
    """Squid writes color BF beside mono fluorescence in one acquisition — same dtype, two
    plane shapes; the color one expands, the mono one is untouched."""
    root = tmp_path / "acq"
    rgb = np.stack([_gray(1), _gray(2), _gray(3)], axis=-1)
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", rgb)
    _write(root / "0", "manual_0_0_Fluorescence_405_nm_Ex.bmp", _gray(9))
    _sidecars(root)
    m = open_reader(root).metadata
    names = [c["name"] for c in m["channels"]]
    assert "Fluorescence_405_nm_Ex" in names and "BF_LED_matrix_full (R)" in names
    assert len(names) == 4


def test_colormap_prefers_the_acquisitions_display_color():
    """The RGB component channels carry pure primaries in their resolved display_color; the
    window's colormap must read THAT, not the wavelength palette (which cannot know '(R)')."""
    pytest.importorskip("napari")
    from squidxplorer._napari_pane import _colormap_for

    from squidxplorer._acquisition import DisplayChannel

    # the REAL metadata type: DisplayChannel records, which duck-type .get but are not dicts
    channels = [DisplayChannel(name="BF_LED_matrix_full (R)",
                               display_name="BF LED matrix full (R)",
                               display_color="#FF0000")]
    cm = _colormap_for("BF_LED_matrix_full (R)", channels)
    assert tuple(np.asarray(cm.colors)[-1][:3]) == (1.0, 0.0, 0.0)
    # matched by display_name too: results deliver whichever spelling the layer carries
    cm2 = _colormap_for("BF LED matrix full (R)", channels)
    assert tuple(np.asarray(cm2.colors)[-1][:3]) == (1.0, 0.0, 0.0)


def test_rgb_components_seed_the_files_full_range(tmp_path):
    """All three primaries share the FILE's own range: per-channel percentiles would tint the
    additive reconstruction and read as 'completely dark' on brightfield."""
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
    """The mosaic yaml says the channel was RGB live; its files are 2-D: the display gets the
    stain colormap measured from the overview PNG. Detection is automatic, pixels untouched."""
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
    pytest.importorskip("napari")
    cm = _colormap_for(chs[0]["name"], chs)
    assert len(np.asarray(cm.colors)) == 256


def test_without_an_overview_the_gray_channel_stays_gray(tmp_path):
    root = tmp_path / "acq"
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", _gray(7))
    _sidecars(root)
    chs = open_reader(root).metadata["channels"]
    assert chs[0].get("display_lut") is None


def test_a_real_rgb_acquisition_expands_and_takes_no_stain_lut(tmp_path):
    root = tmp_path / "acq"
    rgb = np.stack([_gray(1), _gray(2), _gray(3)], axis=-1)
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", rgb)
    _sidecars(root)
    _mosaic_sidecar(root)
    chs = open_reader(root).metadata["channels"]
    assert [c["name"] for c in chs] == [f"BF_LED_matrix_full ({x})" for x in "RGB"]
    assert all(c.get("display_lut") is None for c in chs)   # true primaries need no model


def test_rgb_components_declare_file_color_provenance(tmp_path):
    """Case 1 of the color cascade: real (Y, X, 3) components are the file's own color."""
    root = tmp_path / "acq"
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp",
           np.stack([_gray(1), _gray(2), _gray(3)], axis=-1))
    _sidecars(root)
    chs = open_reader(root).metadata["channels"]
    assert [c.get("color_source") for c in chs] == ["file", "file", "file"]


def test_the_estimated_lut_declares_estimated_provenance(tmp_path):
    """Case 3: the density-fit LUT is derived, and the entry says so."""
    root = tmp_path / "acq"
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", _gray(7))
    _sidecars(root)
    _mosaic_sidecar(root)
    chs = open_reader(root).metadata["channels"]
    assert chs[0].get("display_lut") is not None
    assert chs[0].get("color_source") == "estimated"


def test_a_plain_channel_carries_no_color_source(tmp_path):
    """Case 4: an ordinary channel shows its own yaml color, which needs no label."""
    root = tmp_path / "acq"
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", _gray(7))
    _sidecars(root)
    chs = open_reader(root).metadata["channels"]
    assert chs[0].get("color_source") is None and chs[0].get("display_lut") is None


def test_the_env_flag_kills_reconstruction_to_honest_gray(tmp_path, monkeypatch):
    """SQUIDXPLORER_NO_RECONSTRUCTED_COLOR=1: no stain LUT, the mono channel with its yaml
    color. Real RGB still expands, because the file's own color is not a reconstruction."""
    monkeypatch.setenv("SQUIDXPLORER_NO_RECONSTRUCTED_COLOR", "1")
    root = tmp_path / "acq"
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", _gray(7))
    _sidecars(root)
    _mosaic_sidecar(root)
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


def test_set_reconstruction_round_trips_the_flag(tmp_path):
    """The live override the View menu flips; None returns to the environment's answer."""
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


def test_colormap_without_a_resolved_color_still_uses_the_name_palette():
    pytest.importorskip("napari")
    from squidxplorer._channels import fallback_color
    from squidxplorer._napari_pane import _colormap_for

    cm = _colormap_for("Fluorescence_488_nm_Ex", None)
    h = fallback_color("Fluorescence_488_nm_Ex").lstrip("#")
    want = tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    got = tuple(float(v) for v in np.asarray(cm.colors)[-1][:3])
    assert got == pytest.approx(want)
    # an unrecognised channel with no resolved color stays gray, never a guess
    assert _colormap_for("BF_LED_matrix_full (R)", None) == "gray"
