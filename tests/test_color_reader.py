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
