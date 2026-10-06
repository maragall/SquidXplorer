"""Pad partial acquisitions: a stopped run opens at its PLANNED final state, unwritten = black."""

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


def _assert_padded_open(r):
    """The plan's grid, black unwritten slots, the real plane itself, refusal outside the plan."""
    m = r.metadata
    assert m["regions"] == ["B2", "B3"], "the planned-but-empty region must exist"
    assert m["fovs_per_region"] == {"B2": [0, 1], "B3": [0]}
    assert m["n_z"] == 3 and m["z_levels"] == [0, 1, 2]
    assert m["n_t"] == 2
    assert len(m["fov_positions_um"]) == 3, "every planned FOV must be placeable"
    ch = m["channels"][0]["name"]
    assert r.read("B2", 0, ch, 0).max() == 7
    for args in (("B2", 1, ch, 0), ("B3", 0, ch, 0),
                 ("B2", 0, ch, 2), ("B2", 0, ch, 0, 1)):
        plane = r.read(*args)
        assert plane.shape == (8, 8) and plane.dtype == np.uint16 and plane.max() == 0, args
    with pytest.raises(KeyError):
        r.read("Z9", 0, ch, 0)
    return ch


def test_a_stopped_run_opens_at_its_planned_final_state_with_black_slots(tmp_path):
    with pytest.warns(UserWarning, match="partial acquisition.*BLACK"):
        r = open_reader(_partial(tmp_path), pad_partial=True)
        r.metadata
    ch = _assert_padded_open(r)
    with pytest.raises(KeyError):
        r.read("B2", 7, ch, 0)
    with pytest.raises((KeyError, IndexError)):
        r.read("B2", 0, ch, 9)


def test_the_reader_declares_what_padding_invented(tmp_path):
    """``padded_slots`` names the invented slots — what the well-image backfill stamps."""
    root = _partial(tmp_path)
    r = open_reader(root, pad_partial=True)
    with pytest.warns(UserWarning):
        slots = r.padded_slots
    assert dict(slots.fovs) == {"B2": frozenset({1}), "B3": frozenset({0})}
    assert slots.z_levels == frozenset({1, 2})
    assert slots.time_points == frozenset({1})
    assert bool(slots), "a padded open must read as padded"

    unpadded = open_reader(root)
    assert not unpadded.padded_slots, "an unpadded open invents nothing"


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


def test_a_stopped_ome_acquisition_opens_at_its_planned_final_state_with_black_slots(tmp_path):
    with pytest.warns(UserWarning, match="partial acquisition.*BLACK"):
        r = open_reader(_partial_ome(tmp_path), pad_partial=True)
        r.metadata
    _assert_padded_open(r)


def test_an_unpadded_ome_open_keeps_refusing_missing_slots(tmp_path):
    root = _partial_ome(tmp_path)
    r = open_reader(root)
    with pytest.warns(UserWarning):
        ch = r.metadata["channels"][0]["name"]
    with pytest.raises(KeyError):
        r.read("B3", 0, ch, 0)


def test_the_padded_open_names_the_plan_preference_not_an_absent_file(tmp_path):
    """The executed 0/coordinates.csv EXISTS on a stopped run; it just cannot place a padded grid."""
    root = _partial(tmp_path)
    (root / "0" / "coordinates.csv").write_text("region,x (mm),y (mm)\nB2,1.0,1.0\n")
    with pytest.warns(UserWarning) as record:
        open_reader(root, pad_partial=True).metadata
    about_positions = [str(w.message) for w in record
                       if "coordinates.csv" in str(w.message) and "position" in str(w.message)]
    assert about_positions, "the padded open must say where its positions come from"
    assert all("PLAN's positions" in t for t in about_positions)
    assert not any("is absent" in t for t in about_positions)


def _skipped_timepoints(tmp_path):
    """THE 40x live test (2026-10-05): Nt 3 at a 2 s interval, one timepoint whose frames span
    10 s, the run ENDED (.done). Squid skipped the rest; one folder is all there will be."""
    root = _partial(tmp_path)
    (root / "acquisition.yaml").write_text(_ACQ_YAML.replace("nt: 2", "nt: 3\n  delta_t_s: 2.0"))
    (root / ".done").write_text("")
    (root / "0" / "coordinates.csv").write_text(
        "region,fov,z_level,x (mm),y (mm),z (um),time\n"
        "B2,0,0,1.0,1.0,100.0,2026-09-25_18-10-21.000000\n"
        "B2,0,1,1.0,1.0,100.5,2026-09-25_18-10-31.000000\n")
    return root


def test_a_skipped_timepoint_series_names_its_cause_in_the_open_warning(tmp_path):
    """The padded frames are black and the warning said only "stopped run"; the customer saw
    frames that "don't change". The cause is on disk: the end marker, the declared interval,
    the executed csv's own timestamps."""
    root = _skipped_timepoints(tmp_path)
    with pytest.warns(UserWarning) as caught:
        r = open_reader(root, pad_partial=True)
        assert r.metadata["n_t"] == 3
    texts = [str(w.message) for w in caught if "padded to the planned final state (" in str(w.message)]
    assert texts, [str(w.message) for w in caught]
    text = texts[0]
    assert "1 of the declared 3 timepoint(s) are on disk" in text, text
    assert "ENDED" in text, text
    assert "took 10 s against the 2 s interval" in text, text
    assert "set the interval above one timepoint's duration" in text, text
