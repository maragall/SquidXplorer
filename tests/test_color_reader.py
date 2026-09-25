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


def _overview_sidecar(root):
    """Squid's colored overview WITH geometry: the strongest trigger the shelved
    reconstruction had (PNG + yaml calling the channel RGB + placement)."""
    import numpy as np
    from PIL import Image

    mv = root / "0" / "mosaic_view"
    mv.mkdir(parents=True, exist_ok=True)
    t = np.tile(np.linspace(0.2, 1.0, 60), (40, 1))
    png = np.stack([255 * t ** 0.3, 255 * t, 255 * t ** 0.6], axis=-1).astype(np.uint8)
    Image.fromarray(png).save(mv / "mosaic_2um_x.png")
    (mv / "mosaic_2um.yaml").write_text(
        "resolution_um: 2.0\n"
        "full:\n"
        "  top_left_mm:\n  - 0.0\n  - 0.0\n"
        "  extents_mm:\n  - 0.0\n  - 0.08\n  - 0.0\n  - 0.12\n"
        "rgb_channel_names:\n- 20x BF LED matrix full\n"
        "rgb_view_files:\n- mosaic_2um_x.png\n")


def test_the_color_reconstruction_is_shelved_whole(tmp_path):
    """Julio, 2026-09-25: "Shelf the color reconstruction logic. I meant RGB images looking
    in RGB, there was an error with configuration file. No need to reconstruct." A color
    channel recorded gray was a CONFIGURATION error: even with the overview PNG, its geometry
    and stage positions all present, the channel opens as ONE plain gray channel — no virtual
    (R)/(G)/(B) expansion, no stain LUT, no derived color provenance — and the _stain module
    is gone. Real (Y, X, 3) files still expand into their own components ("file")."""
    root = tmp_path / "acq"
    _write(root / "0", "manual_0_0_BF_LED_matrix_full.bmp", _gray(7))
    lines = ["region,x (mm),y (mm),z (mm)", "manual,0.056,0.036,"]
    (root / "coordinates.csv").write_text("\n".join(lines) + "\n")
    (root / "0" / "coordinates.csv").write_text("\n".join(lines) + "\n")
    _sidecars(root)
    _overview_sidecar(root)
    chs = open_reader(root).metadata["channels"]
    assert [c["name"] for c in chs] == ["BF_LED_matrix_full"]
    assert chs[0].get("display_lut") is None
    assert chs[0].get("color_source") is None
    with pytest.raises(ImportError):
        import squidxplorer._stain  # noqa: F401


def test_color_note_names_the_file_provenance_once():
    """The shared sentence the window says; silent for plain channels. "file" is the ONE
    derived-color word left since the reconstruction shelf."""
    from squidxplorer._acquisition import DisplayChannel
    from squidxplorer._channels import color_note, color_sources

    def ch(source):
        return DisplayChannel(name="x", display_name="x", display_color="#FFFFFF",
                              color_source=source)

    assert color_note([ch(None)]) is None and color_sources([ch(None)]) == []
    note = color_note([ch("file"), ch(None)])
    assert note == "color: file color (the plane's own RGB components)"
    assert color_sources([ch("file")]) == ["file"]
