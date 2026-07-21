"""Shared test fixtures.

`squid_dataset` builds a tiny, real-shaped Squid individual-TIFF acquisition on disk
(2 regions x 2 fov x 2 z x 2 channels, 4x4 uint16 frames) with a legacy-schema
coordinates.csv and a pre-v1.0 (camera_settings-nested color) acquisition_channels.yaml,
plus the acquisition parameters.json scalars. Returns (root_path, {(region,fov,z,ch): array}).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

REGIONS = ["B2", "B3"]
FOVS = [0, 1]
NZ = 2
# One channel present in the YAML (color via nested camera_settings), one ABSENT from the
# YAML (exercises the CHANNEL_COLORS_MAP wavelength fallback). Both contain '_' and '-'.
CH_IN_YAML = "Fluorescence_638_nm_-_Penta"
CH_NOT_IN_YAML = "Fluorescence_561_nm_-_Penta"
CHANNELS = [CH_IN_YAML, CH_NOT_IN_YAML]

_YAML = """\
version: 1
objective: 20x
channels:
- name: Fluorescence 638 nm - Penta
  camera_settings:
    '1':
      display_color: '#FF0000'
      exposure_time_ms: 50.0
"""

# Legacy flat sidecar (fallback source). Note magnification/sensor -> recomputed px 0.188,
# deliberately DIFFERENT from acquisition.yaml's stored 0.325 so tests prove which is used.
_PARAMS = {
    "Nz": NZ,
    "Nt": 1,
    "dz(um)": 1.5,
    "objective": {"magnification": 20.0},
    "sensor_pixel_size_um": 3.76,
}

# Authoritative rich metadata. pixel_size_um is stored (binning-aware), not recomputed.
_ACQ_YAML = """\
objective:
  pixel_size_um: 0.325
  magnification: 20.0
  sensor_pixel_size_um: 3.76
sample:
  wellplate_format: 1536 well plate
z_stack:
  nz: 2
  delta_z_mm: 0.0015
time_series:
  nt: 1
"""


def _pixel_value(r_i, fov, z, c_i):
    # deterministic, unique per plane so exact-read comparisons are meaningful
    return r_i * 1000 + fov * 100 + z * 10 + c_i


def _write_timepoint(folder: Path, arrays: dict, tag: int = 0):
    folder.mkdir(parents=True, exist_ok=True)
    for r_i, region in enumerate(REGIONS):
        for fov in FOVS:
            for z in range(NZ):
                for c_i, ch in enumerate(CHANNELS):
                    base = _pixel_value(r_i, fov, z, c_i) + tag * 5000
                    arr = (np.arange(16, dtype=np.uint16).reshape(4, 4) + base).astype(np.uint16)
                    tifffile.imwrite(folder / f"{region}_{fov}_{z}_{ch}.tiff", arr)
                    arrays[(region, fov, z, ch)] = arr


@pytest.fixture
def squid_dataset(tmp_path):
    root = tmp_path / "acq"
    arrays: dict = {}
    _write_timepoint(root / "0", arrays, tag=0)
    (root / "acquisition_channels.yaml").write_text(_YAML)
    (root / "acquisition.yaml").write_text(_ACQ_YAML)
    (root / "acquisition parameters.json").write_text(json.dumps(_PARAMS))
    return root, arrays


@pytest.fixture
def pyramid_dataset(tmp_path):
    """A Squid acquisition whose fields are big enough to produce a REAL pyramid.

    `squid_dataset` uses 4x4 frames, and `_output._PYRAMID_MIN_YX` (256) collapses anything
    that small to level 0 alone — so it cannot exercise pyramid level selection at all. This
    one writes 640px fields (two levels: 640 -> 320) and stays deliberately minimal otherwise
    (1 well, 1 fov, 1 channel, 2 z) so the extra pixels don't cost real time.

    Returns (root, region, frame_size).
    """
    root = tmp_path / "acq_pyr"
    size, region, ch = 640, "B2", CH_IN_YAML
    folder = root / "0"
    folder.mkdir(parents=True, exist_ok=True)
    for z in range(2):
        # A gradient plus a per-z offset: downsampled levels stay distinguishable, and a crop
        # from a known position has a predictable value (so a loupe read can be checked).
        yy, xx = np.mgrid[0:size, 0:size]
        arr = ((yy + xx).astype(np.uint16) + z * 7)
        tifffile.imwrite(folder / f"{region}_0_{z}_{ch}.tiff", arr)
    (root / "acquisition_channels.yaml").write_text(_YAML)
    (root / "acquisition.yaml").write_text(_ACQ_YAML)
    (root / "acquisition parameters.json").write_text(json.dumps(_PARAMS))
    return root, region, size


@pytest.fixture
def real_dataset():
    """The real hongquan dataset if present locally; else skip (used by integration tests)."""
    path = Path.home() / "Downloads" / "z_stack_2026-05-15_18-39-28.532906 hongquan"
    if not path.is_dir():
        pytest.skip("real hongquan dataset not present")
    return path
