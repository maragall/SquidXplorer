"""The address model, and the one global console that proves it.

WHAT WAS WRONG
--------------
The only identifier this application had was ``scope = row * 1_000_000 + col * 10_000 + roi``
(:mod:`squidmip._plate`). Three defects, and they compound:

1. **It does three jobs.** It is the flat cache key, the logger's id for a thing, and the
   navigator's row. A data address and a view id are different questions and it answers both
   badly.
2. **One of its three fields is not a coordinate.** ``row`` and ``col`` are where the microscope
   was. ``roi`` is THE ORDER SOMEBODY DREW BOXES. Consequences, both real: draw the same box twice
   and the second draw gets slot 1, so identical work is computed and cached twice; delete ROI 2
   and every later id shifts under whatever was pointing at it.
3. **It omits every other dimension.** No z level, no timepoint, no channel. The timepoint bug in
   Task 4 (every consumer reads ``t=0`` and presents it as the whole dataset) is not expressible
   in it, let alone fixable.

And it breaks the naming law :mod:`squidmip._address` is written under: our ROI is a user-drawn
box, a software concept, but the packed id puts it in the structural slot where Squid puts a FIELD
OF VIEW, a physical one. Squid separately uses "ROI" for a manually drawn SCAN SHAPE, an
acquisition INPUT. One word, two ontologies, pointing opposite ways.

WHAT IS PINNED HERE
-------------------
* ``Address`` and ``Extent`` are frozen and hashable, so they can be keys with no defensive copy.
* The same box drawn twice is ONE key, which is the whole reason an ROI stops being an ordinal.
* Deleting an ROI does not move another one's key. Contrasted directly against the packed id,
  which does move: that contrast is the test, not a comment.
* ``None`` means "all of it", and survives a round trip.
* Every line a window logs carries an address, asserted through the emitted RECORD. A test that
  greps the formatted string is testing the formatter.
* Our console renders in Squid's layout, asserted against ``logging.Formatter(LOG_FORMAT)``, so a
  drift from ``squid/logging.py`` is a red test rather than a discovery at the merge.
"""

from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError

import pytest

from squidmip._address import Address, Extent
from squidmip._logpane import (
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


# --- frozen and hashable ------------------------------------------------------------------------

def test_an_address_is_frozen_and_hashable():
    """A key that can be mutated after it is used as a key is not a key: the dict entry it landed
    in becomes unreachable and the same object then misses itself."""
    a = Address("A1", fov=2)
    with pytest.raises(FrozenInstanceError):
        a.fov = 3                              # type: ignore[misc]
    assert {a: "result"}[Address("A1", fov=2)] == "result"
    assert len({Address("A1", fov=2), Address("A1", fov=2)}) == 1


def test_an_extent_is_frozen_and_hashable_including_its_ranges_and_its_box():
    """``range`` and a tuple of floats are both hashable; this pins that no field smuggled in a
    list, which would make the whole object unhashable at the first ROI."""
    e = Extent("A1", fovs=(1, 2), z_levels=range(0, 5), time_points=range(0, 3),
               channels=("DAPI", "GFP"), bbox_um=(0.0, 0.0, 100.0, 50.0))
    with pytest.raises(FrozenInstanceError):
        e.region_id = "B2"                     # type: ignore[misc]
    assert hash(e) == hash(Extent("A1", fovs=(1, 2), z_levels=range(0, 5),
                                  time_points=range(0, 3), channels=("DAPI", "GFP"),
                                  bbox_um=(0.0, 0.0, 100.0, 50.0)))


def test_a_list_of_fovs_is_accepted_and_becomes_a_hashable_tuple():
    """Callers hold lists. Refusing them would push a ``tuple(...)`` onto every call site, and the
    one that forgot would fail at ``hash``, i.e. at the cache, i.e. far from the mistake."""
    assert Extent("A1", fovs=[3, 1, 1]).fovs == (1, 3)
    assert hash(Extent("A1", fovs=[3, 1]))


# --- an ROI is a BOX, not an ordinal ------------------------------------------------------------

def test_the_same_box_drawn_twice_is_ONE_key():
    """THE point of the model. Under the packed id the second draw got ROI slot 1, so identical
    work was computed and cached twice. As a box, two draws of the same box are one extent, one
    key, one computation.

    MUTATION: put a draw ordinal back into Extent and this goes red."""
    first_draw = Extent("A1", bbox_um=(120.0, 340.0, 620.0, 840.0))
    second_draw = Extent("A1", bbox_um=(120.0, 340.0, 620.0, 840.0))

    assert first_draw == second_draw
    assert first_draw.key() == second_draw.key()
    assert len({first_draw, second_draw}) == 1


def test_the_same_box_dragged_from_the_opposite_corner_is_the_same_box():
    """Orientation is normalised: dragging bottom-right to top-left describes the same rectangle
    and must not buy a second cache entry. Magnitudes are NOT normalised; see the module docstring
    of ``_address.py`` for why an approximate equality would not be a key at all."""
    dragged_down = Extent("A1", bbox_um=(120.0, 340.0, 620.0, 840.0))
    dragged_up = Extent("A1", bbox_um=(620.0, 840.0, 120.0, 340.0))
    assert dragged_down == dragged_up
    assert dragged_down.bbox_um == (120.0, 340.0, 620.0, 840.0)


def test_deleting_an_ROI_does_not_change_another_ROIs_key_but_the_packed_id_SHIFTS():
    """The contrast IS the test. Two boxes drawn in A1; the first is deleted.

    Under the packed id the survivor was ROI slot 1 and becomes ROI slot 0, so its cache entry,
    its logger id and its navigator row all move under whatever was pointing at them. Under an
    extent, nothing was ever numbered, so nothing moves.
    """
    from squidmip._plate import roi_code

    keep = Extent("A1", bbox_um=(0.0, 0.0, 10.0, 10.0))
    before = keep.key()

    # the packed id, for the same pair: the survivor was slot 1 and is renumbered to slot 0
    packed_before = roi_code("A1", 1)
    packed_after = roi_code("A1", 0)
    assert packed_before != packed_after, "the packed id was expected to shift; the premise moved"

    drawn = [Extent("A1", bbox_um=(50.0, 50.0, 60.0, 60.0)), keep]
    del drawn[0]
    assert drawn[0].key() == before
    assert drawn[0] == keep


# --- None means all of it, and it round-trips ---------------------------------------------------

def test_None_means_all_of_it_and_a_bare_region_is_the_whole_thing():
    e = Extent("A1")
    assert (e.fovs, e.z_levels, e.time_points, e.channels, e.bbox_um) == (None,) * 5
    assert "fovs=*" in e.key() and "time_points=*" in e.key() and "bbox_um=*" in e.key()

    a = Address("A1")
    assert (a.fov, a.z_level, a.time_point, a.channel) == (None,) * 4
    assert a.label() == "A1"


def test_an_empty_restriction_and_no_restriction_are_the_SAME_key():
    """"every channel" and "the empty channel set" describe one slab. Two spellings of one slab is
    two cache entries for one computation, which is the defect this module exists to remove."""
    assert Extent("A1", channels=()).key() == Extent("A1", channels=None).key()
    assert Extent("A1", fovs=[]) == Extent("A1")


def test_an_extent_round_trips_through_a_dict_with_its_Nones_intact():
    """Task 2 makes a cached result carry its own extent, so this has to survive JSON. The Nones
    are the part that is easy to lose, and losing one turns "all of it" into "nothing"."""
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
    """``hash()`` of a str is salted per process, so a key derived from it cannot survive a restart
    and would silently miss every cache entry written by the previous run."""
    assert Address("A1", fov=2).key() == "A1|fov=2|z_level=*|time_point=*|channel=*"
    assert Extent("A1", bbox_um=(0.0, 0.0, 1.0, 2.0)).key().endswith("bbox_um=0.0,0.0,1.0,2.0")


def test_the_label_spells_squids_words_out_rather_than_shortening_them():
    """A console is where a vocabulary is learned. ``z 5`` printed a thousand times a run is how a
    second name for ``z_level`` starts, and two names for one physical thing is the drift the
    naming law exists to prevent."""
    label = Address("A1", fov=2, z_level=5, time_point=1, channel="DAPI").label()
    assert label == "A1 fov 2 z_level 5 time_point 1 DAPI"
    assert Address("A1", fov=2).label() == "A1 fov 2", "None must not print"


# --- the logger: every line a window emits carries an address -----------------------------------

def _records(caplog):
    return [r for r in caplog.records if hasattr(r, VIEW_FIELD)]


def test_every_line_a_window_logs_carries_its_view_id_AND_its_address(caplog):
    """Asserted through the RECORD, not the string. One global console printing from every open
    window cannot lean on the user knowing which window they meant, so "what happened to what" has
    to be structured data on the record and not prose that happens to read well.

    MUTATION: drop the ``extra`` from ViewLog.process and this goes red while every string
    assertion in the suite stays green."""
    view = ViewLog(get_logger("test_address"), 3, Address("A1", fov=2))
    with caplog.at_level(logging.INFO):
        view.info("mosaic loaded")

    rec = _records(caplog)[-1]
    assert getattr(rec, VIEW_FIELD) == 3
    assert getattr(rec, ADDRESS_FIELD) == Address("A1", fov=2)
    assert isinstance(getattr(rec, ADDRESS_FIELD), Address)


def test_the_bracket_is_the_VIEW_and_the_rest_is_the_ADDRESS(caplog):
    """``[3] A1 fov 2  decon(sigma=2.0)  started``. The ordinal belongs to the desktop and the
    address belongs to the plate, and the line keeps them apart on purpose."""
    view = ViewLog(get_logger("test_address"), 3)
    with caplog.at_level(logging.INFO):
        view.started("decon(sigma=2.0)", address=Address("A1", fov=2))
        view.done("decon(sigma=2.0)", 1.42, address=Address("A1", fov=2))

    started, done = _records(caplog)[-2:]
    assert started.getMessage() == "[3] A1 fov 2  decon(sigma=2.0)  started"
    assert done.getMessage() == "[3] A1 fov 2  decon(sigma=2.0)  done in 1.4 s"


def test_a_per_call_address_beats_the_windows_standing_one(caplog):
    """A window pointing at A1 that runs work on a boxed ROI must log the ROI, not the window."""
    view = ViewLog(get_logger("test_address"), 5, Address("A1"))
    roi = Extent("A1", bbox_um=(0.0, 0.0, 10.0, 10.0))
    with caplog.at_level(logging.INFO):
        view.started("mip", address=roi)

    rec = _records(caplog)[-1]
    assert getattr(rec, ADDRESS_FIELD) is roi
    assert "roi [0.0,0.0 10.0,10.0] um" in rec.getMessage()


def test_an_action_that_fails_says_so_rather_than_going_quiet(caplog):
    """An action that starts and then says nothing is indistinguishable from one still running."""
    view = ViewLog(get_logger("test_address"), 1, Address("B2"))
    with caplog.at_level(logging.INFO):
        view.started("cellpose")
        view.failed("cellpose", "no weights")

    rec = _records(caplog)[-1]
    assert rec.levelno == logging.WARNING
    assert rec.getMessage() == "[1] B2  cellpose  failed: no weights"


def test_a_window_logs_its_address_without_the_window_having_to_remember_to(qtbot=None):
    """The window-level end of the same property, without Qt: ``RegionViewer.address`` is the one
    place a window answers "where am I", and ``_say`` routes through it. Checked structurally so
    this fails if someone re-adds a bare ``log.info`` to a window."""
    import inspect

    from squidmip import _region_viewer

    src = inspect.getsource(_region_viewer.RegionViewer._say)
    assert "self.view_log()" in src, "a window's status line stopped carrying its address"
    assert "log.info(" not in src.replace("self.view_log().info(", ""), \
        "a window logged without a view id and an address"


# --- one console, in Squid's language -----------------------------------------------------------

def test_our_logger_names_live_inside_squids_hierarchy():
    """Squid's root is ``squid`` and every tool of theirs is a child of it. We are a Squid tool, so
    ``squid.xplorer.<module>`` is the truthful name and merges for free. It also puts us under
    ``squid.logging.set_stdout_log_level``, which walks the ``squid`` root's handlers and cannot
    reach a logger named outside the hierarchy."""
    assert XPLORER_ROOT == "squid.xplorer"
    assert get_logger("viewer").name == "squid.xplorer.viewer"
    assert get_logger().name == "squid.xplorer"


def test_our_console_renders_a_line_exactly_the_way_squid_would():
    """LOG_FORMAT / LOG_DATEFORMAT are copied verbatim from ``squid/logging.py``, whose comment
    exports them "for use by other modules". Copied and not imported, because Squid is not a
    dependency of v1 -- which is exactly why a test has to hold the copy in step. When the viewer
    eventually runs inside Squid, one console shows both streams in one format and interleaves
    correctly instead of one of them looking like debris.

    MUTATION: change one separator in LOG_FORMAT or in format_record and this goes red."""
    record = logging.LogRecord(name="squid.xplorer.viewer", level=logging.INFO,
                               pathname="/x/_viewer.py", lineno=4413,
                               msg="[3] A1 fov 2  decon(sigma=2.0)  started", args=(),
                               exc_info=None)
    thread_id_filter(record)

    theirs = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFORMAT).format(record)
    assert format_record(record) == theirs


def test_a_record_of_ours_never_breaks_a_formatter_of_theirs():
    """``thread_id`` is a CUSTOM field of Squid's, injected by a filter on the handler. A record
    without it raises on ``%(thread_id)d``, so ours are given one, on the emitting thread, from
    the same call they use."""
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
    """The double-handler hazard. Two handlers of ours on one logger means every line arrives
    twice, and a console that duplicates every event is worse than none: the user cannot tell one
    event from two. One :class:`LogBus` is built per root window, so this is not hypothetical.

    Refusing the second install would trade a duplicated line for a silent one, so the handler is
    SHARED instead. And a bus leaving must not take the handler with it while another still holds
    it, or closing the second window silences the first."""
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
