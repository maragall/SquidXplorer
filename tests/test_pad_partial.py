"""Pad partial acquisitions: a stopped run opens at its PLANNED final state, unwritten = black.

Squid can be stopped mid-acquisition and this is a post-acquisition tool that assumes final-state
input. The plan is on disk already: the root coordinates.csv (written up front, every position
the run intended) plus the declared Nz/Nt. Metadata declares that grid; ``read`` serves a zeros
plane (max_intensity = 0) for a planned-but-unwritten slot; anything outside the grid refuses
exactly as before.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from squidxplorer import open_reader

_ACQ_YAML = """\
objective:
  pixel_size_um: 0.5
z_stack:
  nz: 3
time_series:
  nt: 2
"""

_CSV = """\
region,x (mm),y (mm)
B2,1.0,1.0
B2,2.0,1.0
B3,3.0,1.0
"""


def _partial(tmp_path):
    """Plan: B2 x 2 FOVs + B3 x 1 FOV, nz=3, nt=2. On disk: ONE plane (B2 fov0 z0 t0)."""
    root = tmp_path / "acq"
    (root / "0").mkdir(parents=True)
    tifffile.imwrite(root / "0" / "B2_0_0_Fluorescence_405_nm_Ex.tiff",
                     np.full((8, 8), 7, np.uint16))
    (root / "acquisition.yaml").write_text(_ACQ_YAML)
    (root / "coordinates.csv").write_text(_CSV)
    return root


def test_a_stopped_run_declares_its_planned_final_state(tmp_path):
    with pytest.warns(UserWarning, match="partial acquisition.*BLACK"):
        m = open_reader(_partial(tmp_path), pad_partial=True).metadata
    assert m["regions"] == ["B2", "B3"], "the planned-but-empty region must exist"
    assert m["fovs_per_region"] == {"B2": [0, 1], "B3": [0]}
    assert m["n_z"] == 3 and m["z_levels"] == [0, 1, 2]
    assert m["n_t"] == 2
    assert len(m["fov_positions_um"]) == 3, "every planned FOV must be placeable"


def test_an_unwritten_slot_reads_as_zeros_and_a_real_one_as_itself(tmp_path):
    r = open_reader(_partial(tmp_path), pad_partial=True)
    with pytest.warns(UserWarning):
        ch = r.metadata["channels"][0]["name"]
    assert r.read("B2", 0, ch, 0).max() == 7                      # the real plane
    for args in (("B2", 1, ch, 0), ("B3", 0, ch, 0),              # padded FOV / region
                 ("B2", 0, ch, 2), ("B2", 0, ch, 0, 1)):          # padded z / timepoint
        plane = r.read(*args)
        assert plane.shape == (8, 8) and plane.dtype == np.uint16 and plane.max() == 0, args


def test_outside_the_plan_still_refuses(tmp_path):
    r = open_reader(_partial(tmp_path), pad_partial=True)
    with pytest.warns(UserWarning):
        ch = r.metadata["channels"][0]["name"]
    with pytest.raises(KeyError):
        r.read("Z9", 0, ch, 0)
    with pytest.raises(KeyError):
        r.read("B2", 7, ch, 0)
    with pytest.raises((KeyError, IndexError)):
        r.read("B2", 0, ch, 9)


def test_a_complete_acquisition_is_untouched(squid_dataset, recwarn):
    """Padding must never engage on a finished run (and is opt-in: the CLI never opts) — same grid, no partial warning."""
    root, _ = squid_dataset
    m = open_reader(root).metadata
    assert not [w for w in recwarn if "partial acquisition" in str(w.message)]
    assert m["regions"] == ["B2", "B3"]


# --- pad_partial reaches the OME-TIFF reader ----------------------------------------------------

_OME_CH_YAML = """\
channels:
- name: Fluorescence 405 nm Ex
  display_color: '#8000FF'
"""


def _partial_ome(tmp_path):
    """Same plan as ``_partial`` (B2 x 2 + B3 x 1, nz=3, nt=2); on disk ONE 1x1x1 OME stack."""
    root = tmp_path / "acq"
    (root / "ome_tiff").mkdir(parents=True)
    tifffile.imwrite(root / "ome_tiff" / "B2_0.ome.tiff",
                     np.full((1, 1, 1, 8, 8), 7, np.uint16), metadata={"axes": "TZCYX"})
    (root / "acquisition.yaml").write_text(_ACQ_YAML)
    (root / "acquisition_channels.yaml").write_text(_OME_CH_YAML)
    (root / "coordinates.csv").write_text(_CSV)
    return root


def test_a_stopped_ome_acquisition_declares_its_planned_final_state(tmp_path):
    with pytest.warns(UserWarning, match="partial acquisition.*BLACK"):
        m = open_reader(_partial_ome(tmp_path), pad_partial=True).metadata
    assert m["regions"] == ["B2", "B3"]
    assert m["fovs_per_region"] == {"B2": [0, 1], "B3": [0]}
    assert m["n_z"] == 3 and m["n_t"] == 2
    assert len(m["fov_positions_um"]) == 3, "every planned FOV must be placeable"


def test_an_unwritten_ome_slot_reads_as_zeros_and_a_real_one_as_itself(tmp_path):
    r = open_reader(_partial_ome(tmp_path), pad_partial=True)
    with pytest.warns(UserWarning):
        ch = r.metadata["channels"][0]["name"]
    assert r.read("B2", 0, ch, 0).max() == 7                      # the real plane
    for args in (("B2", 1, ch, 0), ("B3", 0, ch, 0),              # padded FOV / region
                 ("B2", 0, ch, 2), ("B2", 0, ch, 0, 1)):          # z / t beyond the file's own
        plane = r.read(*args)
        assert plane.shape == (8, 8) and plane.dtype == np.uint16 and plane.max() == 0, args
    with pytest.raises(KeyError):
        r.read("Z9", 0, ch, 0)                                    # outside the plan: refused


def test_an_unpadded_ome_open_keeps_refusing_missing_slots(tmp_path):
    root = _partial_ome(tmp_path)
    r = open_reader(root)
    with pytest.warns(UserWarning):                               # recorded-vs-observed Nz/Nt
        ch = r.metadata["channels"][0]["name"]
    with pytest.raises(KeyError):
        r.read("B3", 0, ch, 0)


# --- the padded open SAYS it prefers the plan; it never claims the executed file is absent ------


def test_the_padded_open_names_the_plan_preference_not_an_absent_file(tmp_path):
    """The executed 0/coordinates.csv EXISTS on a stopped run; it just cannot place a padded
    grid. The warning must say the preference is deliberate, not call the file absent."""
    root = _partial(tmp_path)
    (root / "0" / "coordinates.csv").write_text("region,x (mm),y (mm)\nB2,1.0,1.0\n")
    with pytest.warns(UserWarning) as record:
        open_reader(root, pad_partial=True).metadata
    about_positions = [str(w.message) for w in record
                       if "coordinates.csv" in str(w.message) and "position" in str(w.message)]
    assert about_positions, "the padded open must say where its positions come from"
    assert all("PLAN's positions" in t for t in about_positions)
    assert not any("is absent" in t for t in about_positions)
