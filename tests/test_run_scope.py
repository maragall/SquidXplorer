"""What a run is AIMED AT, with no Qt and no napari in the process."""

from __future__ import annotations

from squidxplorer import _run_scope as E
from squidxplorer._run_scope import describe_run_target


class _Thread:
    def __init__(self, running, preview=False):
        self._running = running
        if preview:
            self.IS_PREVIEW = True

    def isRunning(self):
        return self._running


def test_an_operator_run_is_blocked_by_another_operator_run_but_not_by_a_draining_preview():
    """A retired raw preview keeps running for a moment; counting it as "already processing" blocks a legal run."""
    assert E.operator_busy(_Thread(True), []) is True
    assert E.operator_busy(None, [_Thread(True)]) is True
    assert E.operator_busy(None, []) is False
    assert E.operator_busy(_Thread(False), [_Thread(False)]) is False
    assert E.operator_busy(None, [_Thread(True, preview=True)]) is False
    assert E.operator_busy(None, [_Thread(True, preview=True), _Thread(True)]) is True


def test_each_scope_resolves_off_the_window_state_it_names():
    assert E.resolve_run_scope(E.SCOPE_PLATE, selection=["B2"]) == (None, None)   # None == whole plate
    assert E.resolve_run_scope(E.SCOPE_SELECTION, selection=["B3", "B2", "B3"]) == (["B3", "B2"], None)
    assert E.resolve_run_scope(E.SCOPE_SELECTION, selection=[]) == (None, None)
    assert E.resolve_run_scope(E.SCOPE_REGION, current_region="B4") == (["B4"], None)
    regions, problem = E.resolve_run_scope("everything, obviously")
    assert regions is None and problem and "is not a run scope" in problem
    assert E.RUN_SCOPES[0] == E.SCOPE_SELECTION
    assert set(E.RUN_SCOPES) == {E.SCOPE_SELECTION, E.SCOPE_PLATE, E.SCOPE_REGION}


def test_current_region_scope_with_no_region_open_refuses_out_loud():
    """MUTATION: return (None, None) here and this goes red; in the GUI that would silently run the WHOLE PLATE."""
    regions, problem = E.resolve_run_scope(E.SCOPE_REGION, current_region=None)
    assert regions is None
    assert problem and "current region" in problem


def test_the_target_set_is_named_counted_exactly_and_elided_when_long():
    s = describe_run_target(["A1", "A2", "A3"], total=96)
    assert "3 regions" in s and "A1, A2, A3" in s
    assert "1 region:" in describe_run_target(["B7"], total=96)
    s = describe_run_target([f"A{i}" for i in range(400)], total=1536)
    assert "400 regions" in s and len(s) < 200
    assert "A0, A1" in s and ("..." in s or "…" in s)
    assert "1536" in describe_run_target(None, total=1536)
    assert describe_run_target([], total=96) is None, "an empty target is a refusal, not a run"
