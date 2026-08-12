"""What a run is AIMED AT, with no Qt and no napari in the process.

Every decision ``squidxplorer._run_scope`` makes is here: whether a run may start at all, what the
scope selector's value resolves to against live state, the sentence that names the resolved
target before the compute is spent, and the ``(region, fov)`` expansion a Minerva export is
built from. The Qt widgets in ``_viewer`` only render these answers.

Each test in this file was watched fail before the implementation existed, and the mutation
notes on the sharper ones name the edit that turns them red.

Was ``tests/test_explore.py``. The exploration pane's own rules were tested here too — tab
identity, the subset cursor, the preview progress sentence, the per-region layer name — and
they went with the pane on 2026-08-05.
"""

from __future__ import annotations

import pytest

from squidxplorer import _run_scope as E


# --- "am I allowed to start an operator run?" ---------------------------------------------------

class _Thread:
    def __init__(self, running, preview=False):
        self._running = running
        if preview:
            self.IS_PREVIEW = True

    def isRunning(self):
        return self._running


def test_an_operator_run_is_blocked_by_another_operator_run():
    assert E.operator_busy(_Thread(True), []) is True
    assert E.operator_busy(None, [_Thread(True)]) is True


def test_nothing_running_is_not_busy():
    assert E.operator_busy(None, []) is False
    assert E.operator_busy(_Thread(False), [_Thread(False)]) is False


def test_a_draining_raw_preview_does_not_block_an_operator_run():
    """The bug this exists for: opening a side-pane tab RE-SCOPES and restarts the raw preview,
    and the retired one keeps running for a moment. Counting it as "already processing" made the
    very next operator run refuse itself with "let the current run finish first" — silently, from
    the user's point of view, because they never started a run to finish.

    MUTATION: drop the ``IS_PREVIEW`` filter and this goes red.
    """
    assert E.operator_busy(None, [_Thread(True, preview=True)]) is False
    # ...but a real run draining alongside it still blocks.
    assert E.operator_busy(None, [_Thread(True, preview=True), _Thread(True)]) is True


# --- ONE control panel, scope instead of a second set of buttons -------------------------------

def test_whole_dataset_scope_is_the_historical_plate_wide_run():
    regions, problem = E.resolve_run_scope(E.SCOPE_PLATE, selection=["B2"])
    assert regions is None and problem is None      # None == the whole plate, unchanged


def test_selection_scope_reads_the_plate_selection():
    regions, problem = E.resolve_run_scope(E.SCOPE_SELECTION, selection=["B3", "B2", "B3"])
    assert regions == ["B3", "B2"] and problem is None


def test_selection_scope_with_nothing_selected_is_the_whole_dataset():
    """The default scope must behave exactly as the plate did before a selector existed."""
    assert E.resolve_run_scope(E.SCOPE_SELECTION, selection=[]) == (None, None)


def test_current_region_scope():
    assert E.resolve_run_scope(E.SCOPE_REGION, current_region="B4") == (["B4"], None)


def test_current_region_scope_with_no_region_open_refuses_out_loud():
    """MUTATION: return ``(None, None)`` here and this goes red — and in the GUI, choosing
    'current region' with nothing open would silently run the WHOLE PLATE."""
    regions, problem = E.resolve_run_scope(E.SCOPE_REGION, current_region=None)
    assert regions is None
    assert problem and "current region" in problem


def test_an_unknown_scope_is_named_not_guessed():
    regions, problem = E.resolve_run_scope("everything, obviously")
    assert regions is None
    assert problem and "is not a run scope" in problem


def test_the_scope_list_is_the_only_catalogue_and_starts_at_the_default():
    assert E.RUN_SCOPES[0] == E.SCOPE_SELECTION
    assert set(E.RUN_SCOPES) == {E.SCOPE_SELECTION, E.SCOPE_PLATE, E.SCOPE_REGION}


# --- the Minerva subset ------------------------------------------------------------------------

def test_subset_selection_expands_every_region_to_all_its_fovs():
    """A REGION IS A MOSAIC OF FOVs — the export unit is the region, and it is fused from every
    FOV of it, never from FOV 0 standing in for the well."""
    fovs = {"B2": [0, 1, 2], "B3": [0, 1]}
    assert E.subset_selection(["B2", "B3"], fovs) == [
        ("B2", 0), ("B2", 1), ("B2", 2), ("B3", 0), ("B3", 1)]


def test_subset_selection_keeps_the_tabs_region_order():
    fovs = {"B2": [0], "B3": [0]}
    assert E.subset_selection(["B3", "B2"], fovs) == [("B3", 0), ("B2", 0)]


def test_subset_selection_names_a_region_it_cannot_expand():
    """NO SILENT FAILURE: exporting fewer regions than the tab shows, quietly, is the defect."""
    with pytest.raises(ValueError) as exc:
        E.subset_selection(["B2", "B9"], {"B2": [0]})
    assert "B9" in str(exc.value)


def test_subset_selection_rejects_a_region_with_no_fovs():
    with pytest.raises(ValueError) as exc:
        E.subset_selection(["B2"], {"B2": []})
    assert "B2" in str(exc.value)


def test_subset_selection_rejects_an_empty_subset():
    with pytest.raises(ValueError):
        E.subset_selection([], {"B2": [0]})


def test_subset_selection_carries_a_plate_fov_box_through():
    """The caller owns WHICH regions; only the plate can say which FIELDS inside one.

    This was the third of three places that expanded a region to all its FOVs before the export
    could see the user's box (with ``PlateWindow.selected_region_fovs`` and
    ``minerva_selection``), so an export could pick wells and never fields. A region absent from
    the boxes still expands whole — that is the ONLY meaning of absence, and it is what keeps
    every existing caller byte-for-byte.
    """
    fovs = {"B2": [0, 1, 2], "B3": [0, 1]}
    assert E.subset_selection(["B2", "B3"], fovs, {"B2": [1, 2]}) == [
        ("B2", 1), ("B2", 2), ("B3", 0), ("B3", 1)]
    # omitted entirely == today's behaviour, unchanged
    assert E.subset_selection(["B2", "B3"], fovs) == \
        E.subset_selection(["B2", "B3"], fovs, {})


def test_subset_selection_refuses_a_boxed_fov_the_acquisition_does_not_have():
    """A field the plate offers that the metadata does not is a DISAGREEMENT between two views
    of the same acquisition. Exporting whatever survives the intersection would hide it."""
    with pytest.raises(ValueError) as exc:
        E.subset_selection(["B2"], {"B2": [0, 1]}, {"B2": [1, 9]})
    assert "9" in str(exc.value) and "B2" in str(exc.value)


# ---------------------------------------------------------------------------------------
# Defect 2: the resolved target set is CONFIRMED before the run starts
# ---------------------------------------------------------------------------------------
#
# Prior art the owner has already accepted: Fractal names the resolved target set before a
# job starts. "Run" on a 1536-well plate is otherwise indistinguishable from "Run" on the
# one well the user thought was selected, until the compute is already spent.

def test_the_target_set_is_named_not_just_counted():
    from squidxplorer._run_scope import describe_run_target

    s = describe_run_target(["A1", "A2", "A3"], total=96)
    assert "3 regions" in s
    assert "A1, A2, A3" in s


def test_one_region_is_not_pluralised():
    from squidxplorer._run_scope import describe_run_target

    assert "1 region:" in describe_run_target(["B7"], total=96)


def test_a_long_target_list_is_elided_but_the_count_stays_exact():
    """The point is the MAGNITUDE. 400 well ids in a status line is the same as none."""
    from squidxplorer._run_scope import describe_run_target

    s = describe_run_target([f"A{i}" for i in range(400)], total=1536)
    assert "400 regions" in s
    assert len(s) < 200
    assert "A0, A1" in s          # the head is still shown
    assert "..." in s or "…" in s


def test_the_whole_plate_says_the_whole_plate_and_its_size():
    """`regions is None` is the plate-wide path. The user must see the number that implies."""
    from squidxplorer._run_scope import describe_run_target

    s = describe_run_target(None, total=1536)
    assert "1536" in s


def test_an_empty_target_refuses_rather_than_describing_a_run():
    from squidxplorer._run_scope import describe_run_target

    assert describe_run_target([], total=96) is None
