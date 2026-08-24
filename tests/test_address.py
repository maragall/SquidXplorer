"""The address model: an Extent identifies a region by its coordinates, never a draw order,
and every logged line carries one."""

from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError

import pytest

from squidxplorer._address import Address, Extent
from squidxplorer._logpane import (
    ADDRESS_FIELD,
    LOG_DATEFORMAT,
    LOG_FORMAT,
    VIEW_FIELD,
    LogBus,
    ViewLog,
    XPLORER_ROOT,
    format_record,
    get_logger,
    thread_id_filter,
)


def test_an_address_is_frozen_and_hashable():
    a = Address("A1", fov=2)
    with pytest.raises(FrozenInstanceError):
        a.fov = 3                              # type: ignore[misc]
    assert {a: "result"}[Address("A1", fov=2)] == "result"
    assert len({Address("A1", fov=2), Address("A1", fov=2)}) == 1


def test_an_extent_is_frozen_and_hashable_including_its_ranges_and_its_box():
    """Pins that no field smuggled in a list, which would make the whole object unhashable."""
    e = Extent("A1", fovs=(1, 2), z_levels=range(0, 5), time_points=range(0, 3),
               channels=("DAPI", "GFP"), bbox_um=(0.0, 0.0, 100.0, 50.0))
    with pytest.raises(FrozenInstanceError):
        e.region_id = "B2"                     # type: ignore[misc]
    assert hash(e) == hash(Extent("A1", fovs=(1, 2), z_levels=range(0, 5),
                                  time_points=range(0, 3), channels=("DAPI", "GFP"),
                                  bbox_um=(0.0, 0.0, 100.0, 50.0)))


def test_a_list_of_fovs_is_accepted_and_becomes_a_hashable_tuple():
    """Lists are converted rather than refused, so a caller that forgot tuple(...) fails at
    hash — far from the mistake — instead of every call site needing to convert first."""
    assert Extent("A1", fovs=[3, 1, 1]).fovs == (1, 3)
    assert hash(Extent("A1", fovs=[3, 1]))


def test_the_same_box_drawn_twice_is_ONE_key():
    """THE point of the model: two draws of the same box are one extent, one key, one
    computation — under a packed ordinal id they were two."""
    first_draw = Extent("A1", bbox_um=(120.0, 340.0, 620.0, 840.0))
    second_draw = Extent("A1", bbox_um=(120.0, 340.0, 620.0, 840.0))

    assert first_draw == second_draw
    assert first_draw.key() == second_draw.key()
    assert len({first_draw, second_draw}) == 1


def test_the_same_box_dragged_from_the_opposite_corner_is_the_same_box():
    """Orientation is normalised (drag direction doesn't matter); magnitudes are not."""
    dragged_down = Extent("A1", bbox_um=(120.0, 340.0, 620.0, 840.0))
    dragged_up = Extent("A1", bbox_um=(620.0, 840.0, 120.0, 340.0))
    assert dragged_down == dragged_up
    assert dragged_down.bbox_um == (120.0, 340.0, 620.0, 840.0)


def test_deleting_an_ROI_does_not_change_another_ROIs_key_but_the_packed_id_SHIFTS():
    """The contrast is the test: the packed id renumbers on delete, an Extent key does not."""
    from squidxplorer._plate import roi_code

    keep = Extent("A1", bbox_um=(0.0, 0.0, 10.0, 10.0))
    before = keep.key()

    packed_before = roi_code("A1", 1)
    packed_after = roi_code("A1", 0)
    assert packed_before != packed_after, "the packed id was expected to shift; the premise moved"

    drawn = [Extent("A1", bbox_um=(50.0, 50.0, 60.0, 60.0)), keep]
    del drawn[0]
    assert drawn[0].key() == before
    assert drawn[0] == keep


def test_None_means_all_of_it_and_a_bare_region_is_the_whole_thing():
    e = Extent("A1")
    assert (e.fovs, e.z_levels, e.time_points, e.channels, e.bbox_um) == (None,) * 5
    assert "fovs=*" in e.key() and "time_points=*" in e.key() and "bbox_um=*" in e.key()

    a = Address("A1")
    assert (a.fov, a.z_level, a.time_point, a.channel) == (None,) * 4
    assert a.label() == "A1"


def test_an_empty_restriction_and_no_restriction_are_the_SAME_key():
    """"every channel" and "the empty channel set" must key identically, or two spellings of
    one slab become two cache entries."""
    assert Extent("A1", channels=()).key() == Extent("A1", channels=None).key()
    assert Extent("A1", fovs=[]) == Extent("A1")


def test_an_extent_round_trips_through_a_dict_with_its_Nones_intact():
    """Must survive JSON; a lost None turns "all of it" into "nothing"."""
    for e in (Extent("A1"),
              Extent("manual0", fovs=(0, 4), z_levels=range(2, 9), time_points=range(0, 3),
                     channels=("DAPI", "GFP"), bbox_um=(1.5, 2.5, 3.5, 4.5))):
        assert Extent.from_dict(e.to_dict()) == e
        assert Extent.from_dict(e.to_dict()).key() == e.key()

    import json
    e = Extent("A1", z_levels=range(0, 4))
    assert Extent.from_dict(json.loads(json.dumps(e.to_dict()))) == e


def test_an_address_round_trips_and_an_address_becomes_the_slab_it_names():
    a = Address("A1", fov=2, z_level=5, time_point=1, channel="DAPI")
    assert Address.from_dict(a.to_dict()) == a

    e = Extent.over(a)
    assert e.region_id == "A1" and e.fovs == (2,)
    assert e.z_levels == range(5, 6) and e.time_points == range(1, 2)
    assert e.channels == ("DAPI",)
    assert Extent.over(Address("A1")) == Extent("A1"), "an unrestricted address is the whole slab"


def test_the_key_is_stable_prose_and_not_a_salted_hash():
    """``hash()`` of a str is salted per process, so a key derived from it would silently miss
    every cache entry written by a previous run."""
    assert Address("A1", fov=2).key() == "A1|fov=2|z_level=*|time_point=*|channel=*"
    assert Extent("A1", bbox_um=(0.0, 0.0, 1.0, 2.0)).key().endswith("bbox_um=0.0,0.0,1.0,2.0")


def test_the_label_spells_squids_words_out_rather_than_shortening_them():
    """Abbreviating here ("z 5" instead of "z_level 5") is how a second vocabulary for one
    thing starts."""
    label = Address("A1", fov=2, z_level=5, time_point=1, channel="DAPI").label()
    assert label == "A1 fov 2 z_level 5 time_point 1 DAPI"
    assert Address("A1", fov=2).label() == "A1 fov 2", "None must not print"


def _records(caplog):
    return [r for r in caplog.records if hasattr(r, VIEW_FIELD)]


def test_every_line_a_window_logs_carries_its_view_id_AND_its_address(caplog):
    """Asserted through the record's structured fields, not the formatted string, or this
    would test the formatter instead of the seam."""
    view = ViewLog(get_logger("test_address"), 3, Address("A1", fov=2))
    with caplog.at_level(logging.INFO):
        view.info("mosaic loaded")

    rec = _records(caplog)[-1]
    assert getattr(rec, VIEW_FIELD) == 3
    assert getattr(rec, ADDRESS_FIELD) == Address("A1", fov=2)
    assert isinstance(getattr(rec, ADDRESS_FIELD), Address)


def test_the_bracket_is_the_VIEW_and_the_rest_is_the_ADDRESS(caplog):
    """The view ordinal (desktop) and the address (plate) are deliberately kept apart in the
    line: ``[3] A1 fov 2  decon(sigma=2.0)  started``."""
    view = ViewLog(get_logger("test_address"), 3)
    with caplog.at_level(logging.INFO):
        view.started("decon(sigma=2.0)", address=Address("A1", fov=2))
        view.done("decon(sigma=2.0)", 1.42, address=Address("A1", fov=2))

    started, done = _records(caplog)[-2:]
    assert started.getMessage() == "[3] A1 fov 2  decon(sigma=2.0)  started"
    assert done.getMessage() == "[3] A1 fov 2  decon(sigma=2.0)  done in 1.4 s"


def test_a_per_call_address_beats_the_windows_standing_one(caplog):
    view = ViewLog(get_logger("test_address"), 5, Address("A1"))
    roi = Extent("A1", bbox_um=(0.0, 0.0, 10.0, 10.0))
    with caplog.at_level(logging.INFO):
        view.started("mip", address=roi)

    rec = _records(caplog)[-1]
    assert getattr(rec, ADDRESS_FIELD) is roi
    assert "roi [0.0,0.0 10.0,10.0] um" in rec.getMessage()


def test_an_action_that_fails_says_so_rather_than_going_quiet(caplog):
    view = ViewLog(get_logger("test_address"), 1, Address("B2"))
    with caplog.at_level(logging.INFO):
        view.started("decon")
        view.failed("decon", "no weights")

    rec = _records(caplog)[-1]
    assert rec.levelno == logging.WARNING
    assert rec.getMessage() == "[1] B2  decon  failed: no weights"


def test_a_window_logs_its_address_without_the_window_having_to_remember_to(qtbot=None):
    """Checked structurally (source inspection) so a bare log.info re-added to a window fails
    this without needing Qt."""
    import inspect

    from squidxplorer import _region_viewer

    src = inspect.getsource(_region_viewer.RegionViewer._say)
    assert "self.view_log()" in src, "a window's status line stopped carrying its address"
    assert "log.info(" not in src.replace("self.view_log().info(", ""), \
        "a window logged without a view id and an address"


def test_our_logger_names_live_inside_squids_hierarchy():
    """squid.xplorer.* nests under Squid's own root so squid.logging.set_stdout_log_level's
    handler walk still reaches us."""
    assert XPLORER_ROOT == "squid.xplorer"
    assert get_logger("viewer").name == "squid.xplorer.viewer"
    assert get_logger().name == "squid.xplorer"


def test_our_console_renders_a_line_exactly_the_way_squid_would():
    """LOG_FORMAT/LOG_DATEFORMAT are copied verbatim from squid/logging.py (Squid is not a
    dependency), so this test is what keeps the copy in step."""
    record = logging.LogRecord(name="squid.xplorer.viewer", level=logging.INFO,
                               pathname="/x/_viewer.py", lineno=4413,
                               msg="[3] A1 fov 2  decon(sigma=2.0)  started", args=(),
                               exc_info=None)
    thread_id_filter(record)

    theirs = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFORMAT).format(record)
    assert format_record(record) == theirs


def test_a_record_of_ours_never_breaks_a_formatter_of_theirs():
    """thread_id is Squid's custom field; a record without it raises on %(thread_id)d, so our
    handler injects one via the same filter."""
    bus = LogBus()
    try:
        handler = bus.install()
        assert any(f is thread_id_filter for f in handler.filters), \
            "our handler does not inject Squid's thread_id"
        record = logging.LogRecord(name="squid.xplorer.test", level=logging.INFO,
                                   pathname=__file__, lineno=1, msg="hi", args=(), exc_info=None)
        handler.handle(record)
        logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFORMAT).format(record)   # must not raise
    finally:
        bus.uninstall()


def test_two_consoles_on_one_root_share_ONE_handler_and_both_still_receive():
    """Two LogBus instances (one per window) must share one handler on the root logger, or
    every line duplicates; a bus leaving must not remove the handler while another still holds
    it."""
    root = logging.getLogger("")
    first, second = LogBus(), LogBus()
    seen_first, seen_second = [], []
    first.subscribe(lambda lvl, line: seen_first.append(line))
    second.subscribe(lambda lvl, line: seen_second.append(line))
    try:
        h1 = first.install()
        h2 = second.install()
        assert h1 is h2, "a second bus attached a second handler: every line now arrives twice"
        assert sum(1 for h in root.handlers if h is h1) == 1

        get_logger("test_address").info("said once")
        assert len([ln for ln in seen_first if "said once" in ln]) == 1
        assert len([ln for ln in seen_second if "said once" in ln]) == 1

        first.uninstall()
        assert h1 in root.handlers, "the last bus standing lost its handler"
        get_logger("test_address").info("still delivered")
        assert any("still delivered" in ln for ln in seen_second)
    finally:
        first.uninstall()
        second.uninstall()
    assert not any(h is h1 for h in root.handlers), "the shared handler outlived every bus"
