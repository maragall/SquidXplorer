"""Pane 3's rules, with no Qt and no napari in the process.

Everything the exploration pane DECIDES lives in ``squidmip._explore`` precisely so it can be
tested here: tab identity, the subset slider's cursor, the progress sentence a preview run
writes while it computes, and the (region, fov) list a Minerva export of a subset is built
from. The Qt widgets in ``_viewer`` only render these answers.

Each test in this file was watched fail before the implementation existed, and the mutation
notes on the sharper ones name the edit that turns them red.
"""

from __future__ import annotations

import pytest

from squidmip import _explore as E


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


def test_subset_scope_reads_the_subset_parked_in_the_side_pane():
    """Pane 3 OWNS the subset; the left panel READS it. One owner, one reader."""
    regions, problem = E.resolve_run_scope(E.SCOPE_SUBSET, parked_subset=["B2", "B3"])
    assert regions == ["B2", "B3"] and problem is None


def test_subset_scope_with_an_empty_side_pane_refuses_out_loud():
    regions, problem = E.resolve_run_scope(E.SCOPE_SUBSET, parked_subset=[])
    assert regions is None
    assert problem and "side pane" in problem


def test_an_unknown_scope_is_named_not_guessed():
    regions, problem = E.resolve_run_scope("everything, obviously")
    assert regions is None
    assert problem and "is not a run scope" in problem


def test_the_scope_list_is_the_only_catalogue_and_starts_at_the_default():
    assert E.RUN_SCOPES[0] == E.SCOPE_SELECTION
    assert set(E.RUN_SCOPES) == {E.SCOPE_SELECTION, E.SCOPE_PLATE, E.SCOPE_REGION, E.SCOPE_SUBSET}


# --- tab identity ----------------------------------------------------------------------------

def test_preview_tab_key_is_set_based_and_order_independent():
    k = E.preview_tab_key("acq", "mip", ["B3", "B2"])
    assert k == E.preview_tab_key("acq", "mip", ["B2", "B3"])
    assert k == E.preview_tab_key("acq", "mip", ["B2", "B3", "B2"])   # duplicates collapse
    assert k != E.preview_tab_key("acq", "mip", ["B2"])               # a different set differs


def test_preview_tab_key_separates_operators_and_acquisitions():
    """Two preview runs must be able to COEXIST as two tabs — that is the whole feature.

    MUTATION: drop ``op_key`` from the digest and this goes red, because running two operators
    on one selection would then focus the first tab instead of opening a second, and the user
    could never compare them side by side.
    """
    assert E.preview_tab_key("acq", "mip", ["B2"]) != E.preview_tab_key("acq", "stitch", ["B2"])
    assert E.preview_tab_key("plate_a", "mip", ["B2"]) != E.preview_tab_key("plate_b", "mip", ["B2"])


def test_preview_tab_key_never_collides_with_a_selection_tab():
    """A preview tab and a hand-picked exploration tab over the same wells are DIFFERENT tabs."""
    assert E.preview_tab_key("acq", "mip", ["B2"]) != E.exploration_tab_key("acq", ["B2"])
    assert E.preview_tab_key("acq", "mip", ["B2"]).startswith("preview:")


def test_preview_tab_key_rejects_empty():
    with pytest.raises(ValueError):
        E.preview_tab_key("acq", "mip", [])


def test_preview_tab_label_names_the_operator_and_the_wells():
    assert E.preview_tab_label("MIP", ["B2"]) == "MIP · B2"
    assert E.preview_tab_label("MIP", ["B5", "B2", "B3"]) == "MIP · B2–B5 (3)"
    assert "preview:" not in E.preview_tab_label("MIP", ["B2"])   # never show the hash as a title


def test_region_range_label_is_the_one_implementation():
    """``_viewer.exploration_tab_label`` is this function — two spellings of one label is exactly
    the duplication that has bitten this project."""
    from squidmip import _viewer

    assert _viewer.exploration_tab_label is E.exploration_tab_label
    assert _viewer.exploration_tab_key is E.exploration_tab_key


# --- the subset slider -----------------------------------------------------------------------

def test_cursor_starts_on_the_first_region():
    c = E.SubsetCursor(["B2", "B3", "B4"])
    assert c.regions == ["B2", "B3", "B4"]
    assert c.index == 0
    assert c.region == "B2"
    assert len(c) == 3


def test_cursor_dedupes_and_keeps_selection_order():
    c = E.SubsetCursor(["B4", "B2", "B4"])
    assert c.regions == ["B4", "B2"]


def test_cursor_needs_at_least_one_region():
    with pytest.raises(ValueError):
        E.SubsetCursor([])


def test_cursor_clamps_instead_of_raising():
    c = E.SubsetCursor(["B2", "B3"])
    assert c.set_index(99) is True
    assert c.index == 1 and c.region == "B3"
    assert c.set_index(-5) is True
    assert c.index == 0


def test_cursor_reports_no_move_when_the_index_is_unchanged():
    """MUTATION: return True unconditionally from ``set_index`` and this goes red. It matters
    because every True re-aims the tab's viewer, and a slider that re-loads a mosaic on every
    stray event is the stutter the centre pane's settle-coalescer exists to prevent."""
    c = E.SubsetCursor(["B2", "B3"])
    assert c.set_index(0) is False
    assert c.set_index(1) is True
    assert c.set_index(1) is False


def test_cursor_tracks_which_regions_are_already_loaded():
    c = E.SubsetCursor(["B2", "B3", "B4"])
    assert c.pending() == ["B2", "B3", "B4"]
    assert c.is_loaded("B2") is False
    c.mark_loaded("B2")
    assert c.is_loaded("B2") is True
    assert c.pending() == ["B3", "B4"]


def test_cursor_refuses_to_mark_a_region_it_is_not_scoped_to():
    """The tab claims to show a subset; marking a foreign region loaded would make that claim
    false without anyone being told."""
    c = E.SubsetCursor(["B2"])
    with pytest.raises(ValueError):
        c.mark_loaded("B9")


# --- what a preview run says while it computes ------------------------------------------------

def test_progress_sentence_counts_regions_not_fovs():
    assert E.progress_sentence("MIP", 0, 3) == "MIP · 0 of 3 regions computed"
    assert E.progress_sentence("MIP", 1, 1) == "MIP · 1 of 1 region computed · complete"


def test_progress_sentence_says_complete_only_when_it_is():
    assert "complete" not in E.progress_sentence("MIP", 2, 3)
    assert E.progress_sentence("MIP", 3, 3).endswith("complete")


def test_progress_sentence_surfaces_failures():
    """A region that could not be computed must be SAID, not dropped from the count."""
    s = E.progress_sentence("MIP", 2, 3, failed=1)
    assert "1 failed" in s


def test_progress_sentence_refuses_a_nonsense_total():
    with pytest.raises(ValueError):
        E.progress_sentence("MIP", 0, 0)


def test_progress_sentence_never_says_live():
    """Post-acquisition tool: 'live' is the wrong word and is retired from user-facing copy."""
    assert "live" not in E.progress_sentence("MIP", 1, 2).lower()


# --- layer naming ------------------------------------------------------------------------------

def test_subset_layer_op_is_unique_per_region():
    """Each computed region becomes its OWN napari layer group, so results accumulate on screen
    as they compute instead of the last one replacing the previous.

    MUTATION: drop ``region`` from the op string and this goes red — and in the GUI every
    region's result would land on the same napari layer, which is the "layers don't update in
    the napari mosaic" complaint reproduced exactly.
    """
    assert E.subset_layer_op("MIP", "B2") != E.subset_layer_op("MIP", "B3")
    assert E.subset_layer_op("MIP", "B2") == "MIP · B2"
    assert E.subset_layer_op("raw", "B2") == "raw · B2"


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
