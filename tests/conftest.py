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


# Legacy-schema coordinates.csv (Monkey layout: an explicit fov column). The two FOVs of each
# region sit side by side in X at a 1-frame pitch (4 px * 0.325 um/px = 0.0013 mm), so a mosaic
# of them is 2 wide x 1 tall — small, but a REAL relative offset the placement math must honour.
_FOV_PITCH_MM = 4 * 0.325 / 1000.0
_COORDS_HEADER = "region,fov,z_level,x (mm),y (mm),z (um),time"


def _coords_csv() -> str:
    lines = [_COORDS_HEADER]
    for r_i, region in enumerate(REGIONS):
        for fov in FOVS:
            x = 10.0 + r_i * 5.0 + fov * _FOV_PITCH_MM     # fov 1 is one frame-width right of fov 0
            y = 20.0 + r_i * 5.0                            # same row -> a 2x1 mosaic
            lines.append(f"{region},{fov},0,{x:.6f},{y:.6f},0.0,0.0")
    return "\n".join(lines) + "\n"


@pytest.fixture
def squid_dataset(tmp_path):
    root = tmp_path / "acq"
    arrays: dict = {}
    _write_timepoint(root / "0", arrays, tag=0)
    (root / "acquisition_channels.yaml").write_text(_YAML)
    (root / "acquisition.yaml").write_text(_ACQ_YAML)
    (root / "acquisition parameters.json").write_text(json.dumps(_PARAMS))
    (root / "coordinates.csv").write_text(_coords_csv())
    return root, arrays


@pytest.fixture
def real_dataset():
    """The real hongquan dataset if present locally; else skip (used by integration tests)."""
    path = Path.home() / "Downloads" / "z_stack_2026-05-15_18-39-28.532906 hongquan"
    if not path.is_dir():
        pytest.skip("real hongquan dataset not present")
    return path
