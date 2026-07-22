"""Shared test fixtures.

`squid_dataset` builds a tiny, real-shaped Squid individual-TIFF acquisition on disk
(2 regions x 2 fov x 2 z x 2 channels, 4x4 uint16 frames) with a legacy-schema
coordinates.csv and a pre-v1.0 (camera_settings-nested color) acquisition_channels.yaml,
plus the acquisition parameters.json scalars. Returns (root_path, {(region,fov,z,ch): array}).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Pin the Qt binding before any test module imports qtpy (or napari, which imports it for you).
# PyQt5, PyQt6 and PySide6 are all installed here and qtpy's default preference order still starts
# at PyQt5; loading two Qt majors in one process aborts the interpreter. conftest is imported
# before every test module, so this is the earliest hook that covers the whole session.
os.environ.setdefault("QT_API", "pyqt6")

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


# Real Squid coordinates.csv schema (verified against synthetic_2x2_wellplate): region + x/y in
# mm, a z column that is present but EMPTY, and NO fov column. FOV identity is row order within
# a region. Rows are repeated once per z-level here (NZ=2) because that is what a multi-z
# acquisition writes — the reader must de-duplicate on (region, x, y) before counting, or every
# real z-stack would trip the row-count cross-check.
_FOV_MM = {0: (10.0, 20.0), 1: (10.5, 20.0)}   # fov 1 is +0.5 mm in x => same row, next column


def _coordinates_csv() -> str:
    lines = ["region,x (mm),y (mm),z (mm)"]
    for region in REGIONS:
        for _z in range(NZ):                    # one row per z-level, same stage position
            for fov in FOVS:
                x, y = _FOV_MM[fov]
                lines.append(f"{region},{x},{y},")
    return "\n".join(lines) + "\n"


@pytest.fixture
def squid_dataset(tmp_path):
    root = tmp_path / "acq"
    arrays: dict = {}
    _write_timepoint(root / "0", arrays, tag=0)
    (root / "acquisition_channels.yaml").write_text(_YAML)
    (root / "acquisition.yaml").write_text(_ACQ_YAML)
    (root / "acquisition parameters.json").write_text(json.dumps(_PARAMS))
    (root / "coordinates.csv").write_text(_coordinates_csv())
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


# --- IMA-254: one fixture per Squid output writer --------------------------------------------
#
# The builders live in tests/writer_fixtures.py and are imported INSIDE each fixture, not at
# module scope: writer_fixtures imports this module's shape constants, so a top-level import here
# would be circular. Deferring it also keeps collection cheap for the many tests that need none
# of them.
#
# Every one of these produces the SAME logical acquisition (2 regions x 2 FOVs x 2 z x 2 channels
# of 4x4 uint16) through a DIFFERENT writer, which is what lets the coverage suite assert
# identical metadata and identical pixels across all six with no per-writer special-casing.

def _writer_fixture(tmp_path, builder, name):
    from tests import writer_fixtures

    root = builder(tmp_path / name)
    return root, writer_fixtures.expected_arrays()


@pytest.fixture
def multipage_dataset(tmp_path):
    """MULTI_PAGE_TIFF: ``0/{region}_{fov:04}_stack.tiff``, positions inline, no coordinates.csv."""
    from tests import writer_fixtures

    return _writer_fixture(tmp_path, writer_fixtures.build_multi_page_tiff, "acq_multipage")


@pytest.fixture
def ome_tiff_dataset(tmp_path):
    """SaveOMETiffJob: ``ome_tiff/{region}_{fov:04}.ome.tiff``, 5-D TZCYX."""
    from tests import writer_fixtures

    return _writer_fixture(tmp_path, writer_fixtures.build_ome_tiff, "acq_ome")


@pytest.fixture
def zarr_hcs_dataset(tmp_path):
    """SaveZarrJob HCS: ``plate.ome.zarr/{row}/{col}/{fov}/0``, 5-D TCZYX."""
    from tests import writer_fixtures

    return _writer_fixture(tmp_path, writer_fixtures.build_zarr_hcs, "acq_zarr_hcs")


@pytest.fixture
def zarr_per_fov_dataset(tmp_path):
    """SaveZarrJob non-HCS default: ``zarr/{region}/fov_{n}.ome.zarr/0``, 5-D TCZYX."""
    from tests import writer_fixtures

    return _writer_fixture(tmp_path, writer_fixtures.build_zarr_per_fov, "acq_zarr_fov")


@pytest.fixture
def zarr_6d_dataset(tmp_path):
    """SaveZarrJob non-HCS 6D: ``zarr/{region}/acquisition.zarr``, 6-D FTCZYX (non-standard)."""
    from tests import writer_fixtures

    return _writer_fixture(tmp_path, writer_fixtures.build_zarr_6d, "acq_zarr_6d")


@pytest.fixture
def real_dataset():
    """The real 10x laser-AF tissue acquisition; else skip (used by integration tests).

    Repointed from the old `hongquan` z-stack, which was deleted. This is the acquisition the
    product is actually demoed on, and it is the harder case: a GLASS SLIDE with freeform regions
    (manual0 27 FOVs / manual1 28), Nz=10, 4 channels, 0.752 um/px, OME-TIFF on disk. Real pixels,
    real overlap (~209 px, ~10%), real per-channel focus disagreement.
    """
    path = Path("/Users/julioamaragall/Downloads/"
                "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy")
    if not path.is_dir():
        pytest.skip(f"real tissue acquisition not present at {path}")
    return path
