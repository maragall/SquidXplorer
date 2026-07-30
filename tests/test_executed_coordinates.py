"""FOV placement uses where the stage ACTUALLY went, not where it was told to go.

2026-07-29. Squid writes two different files named ``coordinates.csv`` and until today we only
read the weaker one.

* ``{root}/coordinates.csv`` is ``region, x, y, z``, written BEFORE the run by
  ``multi_point_controller.py:735-744``. It is the PLAN.
* ``{root}/{time_point}/coordinates.csv`` is ``region, fov, z_level, x, y, z, time``, written by
  the worker as it goes (``multi_point_worker.py:757``, columns at ``:802-805``). It is what
  HAPPENED.

Autofocus, backlash and a stage that did not quite arrive all live in the difference. Placing FOVs
from the plan draws the mosaic where the run was supposed to happen, which produces a picture that
looks right and is not, and that is worse than a crash because nothing tells you.

Two agents found this independently on 2026-07-29, one reading Squid's writer and one reading
Squid's design docs, which is why it is treated as fact rather than a hunch. Neither of the real
acquisitions on this machine exercised it, because both are single-timepoint and their planned and
executed positions agree closely enough that nothing looked wrong.

The executed file also carries an explicit ``fov`` column, so reading it needs no row-order
inference. Row-order inference is the fragile half of reading the planned file, the half that has
to raise rather than guess when the count disagrees (``reader.py`` refuses to "place FOVs at
positions that would look plausible but be wrong"). Preferring the executed file removes that risk
rather than mitigating it.

Direction of travel, so nobody reads this as the final answer: Squid has APPROVED making the
per-FOV OME ``translation`` authoritative and demoting ``coordinates.csv`` to a derived export,
with "the OME translation is authoritative if the two ever disagree". Our zarr path already
prefers ``translation``. This is the raw-TIFF path catching up by one step, and the timeline for
the rest is filed with Squid as an action item.
"""
from __future__ import annotations

from pathlib import Path

from squidmip.reader import COORDS_EXECUTED, COORDS_PLANNED, _coords_path, load_fov_positions_um

_PLANNED = "region,x (mm),y (mm),z (mm)\nA1,1.0,1.0,0.0\nA1,2.0,1.0,0.0\n"

#: The executed file: an explicit fov column, and positions that DIFFER from the plan, which is
#: the whole point. A fixture where the two agree cannot tell a correct reader from a broken one.
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


def test_the_executed_file_wins_when_both_exist(tmp_path):
    """The bug this fixes: we were reading the plan while the record sat one directory down."""
    root = _acq(tmp_path)
    assert _coords_path(root) == (root / "0" / "coordinates.csv", COORDS_EXECUTED)


def test_the_planned_file_is_still_read_when_there_is_no_executed_one(tmp_path):
    """Not every acquisition has one, and refusing those would reject the installed base."""
    root = _acq(tmp_path, executed=False)
    assert _coords_path(root) == (root / "coordinates.csv", COORDS_PLANNED)


def test_neither_present_is_not_an_error_here(tmp_path):
    """Placement raises later with a clear message; this resolver just reports absence."""
    root = _acq(tmp_path, planned=False, executed=False)
    assert _coords_path(root) == (None, None)


def test_the_positions_actually_come_from_the_executed_file(tmp_path):
    """The resolver choosing a path is not enough: the numbers have to change."""
    root = _acq(tmp_path)
    got = load_fov_positions_um(root, {"A1": [0, 1]})
    # Executed x is 1.5 and 2.5 mm, planned is 1.0 and 2.0. Micrometres in our world space.
    xs = sorted(round(v[0]) for v in got.values())
    assert xs == [1500, 2500], f"placed from the PLAN, not the record: {got}"


def test_falling_back_to_planned_ANNOUNCES_itself(tmp_path):
    """Julio, 2026-07-29: "looks like the fallback creates confusing behavior". It did.

    A fallback that quietly swaps one accuracy for another is the thing this repo's own rule
    forbids: no fallbacks, fail to the logger. So the degraded read is usable AND loud, and the
    warning has to name the CONSEQUENCE, not just the fact. A line that says "used the planned
    file" without saying what that costs reads as noise and gets scrolled past.
    """
    import pytest as _pytest

    root = _acq(tmp_path, executed=False)
    with _pytest.warns(UserWarning) as caught:
        got = load_fov_positions_um(root, {"A1": [0, 1]})
    xs = sorted(round(v[0]) for v in got.values())
    assert xs == [1000, 2000], "the planned positions were not used"

    msg = str(caught[0].message)
    assert "PLANNED" in msg, "the warning does not say which source was used"
    assert "seams" in msg, "the warning does not say what the degraded source COSTS"
    assert "hand-built" in msg, "the warning does not say what a missing executed file implies"


def test_the_executed_path_does_NOT_warn(tmp_path):
    """The good path must stay quiet, or the warning trains people to ignore it."""
    import warnings as _warnings

    root = _acq(tmp_path)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        load_fov_positions_um(root, {"A1": [0, 1]})
