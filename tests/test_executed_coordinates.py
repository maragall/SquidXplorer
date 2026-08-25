"""FOV placement uses where the stage ACTUALLY went, not where it was told to go."""
from __future__ import annotations

import warnings

import pytest

from squidxplorer.reader import COORDS_EXECUTED, COORDS_PLANNED, _coords_path, load_fov_positions_um

_PLANNED = "region,x (mm),y (mm),z (mm)\nA1,1.0,1.0,0.0\nA1,2.0,1.0,0.0\n"

#: An explicit fov column, and positions that DIFFER from the plan — a fixture where the two agree
#: cannot tell a correct reader from a broken one.
_EXECUTED = (
    "region,fov,z_level,x (mm),y (mm),z (um),time\n"
    "A1,0,0,1.5,1.25,0.0,0.0\n"
    "A1,1,0,2.5,1.25,0.0,0.0\n"
)


def _acq(tmp_path, planned=True, executed=True):
    root = tmp_path / "acq"
    (root / "0").mkdir(parents=True)
    if planned:
        (root / "coordinates.csv").write_text(_PLANNED)
    if executed:
        (root / "0" / "coordinates.csv").write_text(_EXECUTED)
    return root


def test_the_resolver_prefers_the_executed_file_then_the_plan_then_reports_absence(tmp_path):
    """Not every acquisition has an executed file; refusing those would reject the installed base."""
    root = _acq(tmp_path)
    assert _coords_path(root) == (root / "0" / "coordinates.csv", COORDS_EXECUTED)
    root = _acq(tmp_path / "planned_only", executed=False)
    assert _coords_path(root) == (root / "coordinates.csv", COORDS_PLANNED)
    root = _acq(tmp_path / "neither", planned=False, executed=False)
    assert _coords_path(root) == (None, None)


def test_the_positions_come_from_the_executed_file_without_a_warning(tmp_path):
    """The good path must stay quiet, or the warning trains people to ignore it."""
    root = _acq(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = load_fov_positions_um(root, {"A1": [0, 1]})
    xs = sorted(round(v[0]) for v in got.values())
    assert xs == [1500, 2500], f"placed from the PLAN, not the record: {got}"


def test_falling_back_to_planned_ANNOUNCES_itself(tmp_path):
    """A fallback that quietly swaps one accuracy for another is what this repo forbids, so the degraded read is usable AND loud, and the warning names the"""
    root = _acq(tmp_path, executed=False)
    with pytest.warns(UserWarning) as caught:
        got = load_fov_positions_um(root, {"A1": [0, 1]})
    xs = sorted(round(v[0]) for v in got.values())
    assert xs == [1000, 2000], "the planned positions were not used"

    msg = str(caught[0].message)
    assert "PLANNED" in msg, "the warning does not say which source was used"
    assert "seams" in msg, "the warning does not say what the degraded source COSTS"
    assert "hand-built" in msg, "the warning does not say what a missing executed file implies"
