"""What a run is AIMED AT, with no Qt and no napari in the process.

Every decision squidxplorer._run_scope makes is here: whether a run may start at all, what the
scope selector's value resolves to against live state, and the sentence that names the
resolved target before compute is spent.
The Qt widgets in _viewer only render these answers.
"""

from __future__ import annotations

import pytest

from squidxplorer import _run_scope as E


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
    """Opening a side-pane tab re-scopes and restarts the raw preview, and the retired one keeps
    running for a moment; counting it as "already processing" silently refused the very next
    operator run. MUTATION: drop the IS_PREVIEW filter and this goes red."""
    assert E.operator_busy(None, [_Thread(True, preview=True)]) is False
    # a real run draining alongside it still blocks
    assert E.operator_busy(None, [_Thread(True, preview=True), _Thread(True)]) is True


def test_whole_dataset_scope_is_the_historical_plate_wide_run():
    regions, problem = E.resolve_run_scope(E.SCOPE_PLATE, selection=["B2"])
    assert regions is None and problem is None      # None == the whole plate, unchanged


def test_selection_scope_reads_the_plate_selection():
    regions, problem = E.resolve_run_scope(E.SCOPE_SELECTION, selection=["B3", "B2", "B3"])
    assert regions == ["B3", "B2"] and problem is None


def test_selection_scope_with_nothing_selected_is_the_whole_dataset():
    assert E.resolve_run_scope(E.SCOPE_SELECTION, selection=[]) == (None, None)


def test_current_region_scope():
    assert E.resolve_run_scope(E.SCOPE_REGION, current_region="B4") == (["B4"], None)


def test_current_region_scope_with_no_region_open_refuses_out_loud():
    """MUTATION: return (None, None) here and this goes red — in the GUI, 'current region' with
    nothing open would silently run the WHOLE PLATE."""
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


# the resolved target set is confirmed before the run starts

def test_the_target_set_is_named_not_just_counted():
    from squidxplorer._run_scope import describe_run_target

    s = describe_run_target(["A1", "A2", "A3"], total=96)
    assert "3 regions" in s
    assert "A1, A2, A3" in s


def test_one_region_is_not_pluralised():
    from squidxplorer._run_scope import describe_run_target

    assert "1 region:" in describe_run_target(["B7"], total=96)


def test_a_long_target_list_is_elided_but_the_count_stays_exact():
    from squidxplorer._run_scope import describe_run_target

    s = describe_run_target([f"A{i}" for i in range(400)], total=1536)
    assert "400 regions" in s
    assert len(s) < 200
    assert "A0, A1" in s          # the head is still shown
    assert "..." in s or "…" in s


def test_the_whole_plate_says_the_whole_plate_and_its_size():
    from squidxplorer._run_scope import describe_run_target

    s = describe_run_target(None, total=1536)
    assert "1536" in s


def test_an_empty_target_refuses_rather_than_describing_a_run():
    from squidxplorer._run_scope import describe_run_target

    assert describe_run_target([], total=96) is None
