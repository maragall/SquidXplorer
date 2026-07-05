"""Cross-slot integration tests — the "cross commit" surface for SquidMIP.

This is the SHARED integration file: each slot appends ONE section here as it lands, testing
the real seam between it and the slots it depends on — no mocks, on real data. The file grows
one section per ticket; keep sections ordered by slot and self-contained.

Datasets:
  * ``sim_1536wp``        — synthetic plate scale (1536 wells); the ``sim_1536wp`` fixture.
  * real Squid acquisition — a real dataset on disk (the ``real_dataset`` fixture, a folder
    under ~/Downloads), different shape/Nz from the synthetic one.

acquisition.yaml is the single required metadata format (JSON support removed).

Everything here is marked ``integration`` (needs real data on disk) and is deselected in
clean-room CI via ``pytest -m "not integration"``.

Sections
--------
  IMA-183 ↔ IMA-189 : open_reader -> select_fovs -> project_well  (below)
  IMA-188 ↔ IMA-183 : parallel/streaming engine over project()   (added by the IMA-188 slot)
  ...
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from squidmip import open_reader, project_well, select_fovs

SIM_1536WP = Path("/Users/julioamaragall/CEPHLA/Data/sim_1536wp")


@pytest.fixture
def sim_1536wp():
    if not SIM_1536WP.is_dir():
        pytest.skip(f"sim_1536wp not present at {SIM_1536WP}")
    return SIM_1536WP


def _assert_well_matches_np_max(reader, region, fov):
    """project_well(region, fov) == np.max over z_levels of the reader's own exact reads."""
    meta = reader.metadata
    out = project_well(reader, region, fov)
    assert out.shape == (meta["n_t"], len(meta["channels"]), 1, *meta["frame_shape"])
    assert out.dtype == meta["dtype"]
    for t in range(meta["n_t"]):
        for c_i, ch in enumerate(c["name"] for c in meta["channels"]):
            ref = np.max(
                np.stack(
                    [reader.read(region, fov, ch, z, t) for z in meta["z_levels"]]
                ),
                axis=0,
            )
            np.testing.assert_array_equal(out[t, c_i, 0], ref)


# ══════════════════════════════════════════════════════════════════════════════════════
# SECTION: IMA-183 ↔ IMA-189  —  open_reader -> select_fovs -> project_well
# (next slot: append a "SECTION: IMA-188 ↔ IMA-183" block below, don't edit this one)
# ══════════════════════════════════════════════════════════════════════════════════════

# --- sim_1536wp (synthetic plate scale) ---
@pytest.mark.integration
def test_sim1536_metadata_sanity(sim_1536wp):
    # sim_1536wp's acquisition.yaml declares nz=3 but 20 z-planes exist on disk. The reader
    # must WARN and trust the filenames (IMA-189 "filenames are ground truth"). Asserting the
    # warning here turns incidental noise into a documented check + covers the Nz-mismatch path.
    with pytest.warns(UserWarning, match="Recorded Nz"):
        meta = open_reader(sim_1536wp).metadata
    assert len(meta["regions"]) == 1536
    assert all(
        fovs == [0] for fovs in meta["fovs_per_region"].values()
    )  # one FOV per well
    assert meta["n_z"] == 20
    assert meta["z_levels"] == list(range(20))
    assert len(meta["channels"]) == 4
    assert meta["dtype"] == np.uint16
    assert meta["frame_shape"] == (4168, 4168)


@pytest.mark.filterwarnings("ignore:Recorded Nz")  # asserted in test_sim1536_metadata_sanity
@pytest.mark.integration
def test_sim1536_select_one_fov_per_well(sim_1536wp):
    meta = open_reader(sim_1536wp).metadata
    wells = select_fovs(meta, n_fovs=1)
    assert len(wells) == 1536
    assert all(fovs == [0] for fovs in wells.values())


@pytest.mark.filterwarnings("ignore:Recorded Nz")  # asserted in test_sim1536_metadata_sanity
@pytest.mark.integration
def test_sim1536_project_sampled_wells_pixel_exact(sim_1536wp):
    reader = open_reader(sim_1536wp)
    regions = reader.metadata["regions"]
    for region in (
        regions[0],
        regions[len(regions) // 2],
        regions[-1],
    ):  # first / mid / last
        _assert_well_matches_np_max(reader, region, 0)


# (single-well memory-footprint + speed baselines live in tests/test_performance.py)


# --- real Squid acquisition on disk (different shape/Nz) ---
@pytest.mark.integration
def test_real_acquisition_pipeline_end_to_end(real_dataset):
    reader = open_reader(real_dataset)
    meta = reader.metadata
    # Everything the projection needs is present and complete -> the pipeline runs pixel-exact
    # on a real acquisition whose shape/Nz differ from the sim fixture.
    assert meta["regions"]
    assert meta["z_levels"]
    assert meta["channels"]
    wells = select_fovs(meta, n_fovs=1)
    assert set(wells) == set(meta["regions"])
    region = meta["regions"][0]
    _assert_well_matches_np_max(reader, region, wells[region][0])


@pytest.mark.integration
def test_real_acquisition_mip_actually_combines_z(real_dataset):
    # Efficacy on the real z-stack: the MIP must (a) dominate every single z-slice pixel-wise
    # (the max-projection property) and (b) genuinely COMBINE planes — with >1 z it must not
    # equal any single slice, i.e. it is not silently passing one plane through.
    reader = open_reader(real_dataset)
    meta = reader.metadata
    region = meta["regions"][0]
    fov = meta["fovs_per_region"][region][0]
    z_levels = meta["z_levels"]
    out = project_well(reader, region, fov)              # (T, C, 1, Y, X) — computed once
    # validate EVERY timepoint and EVERY channel, not just t0/c0
    for t in range(meta["n_t"]):
        for c_i, ch in enumerate(c["name"] for c in meta["channels"]):
            mip = out[t, c_i, 0]
            slices = [reader.read(region, fov, ch, z, t) for z in z_levels]
            for s in slices:
                assert (mip >= s).all()                  # (a) dominates every slice
            assert np.array_equal(mip, np.max(np.stack(slices), axis=0))
            if len(z_levels) > 1:                        # (b) combines, not a pass-through
                assert all(not np.array_equal(mip, s) for s in slices)
                assert (mip > np.stack(slices).min(axis=0)).any()
