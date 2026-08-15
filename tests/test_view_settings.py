"""Settings scope: which settings are global defaults, which are per-window, and who owns each."""

from __future__ import annotations

import gc
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

# Environment gate, not a skipped assertion: PyQt5 is an optional extra (`.[gui]`) imported at
# module scope by `squidxplorer._region_viewer`.
pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from squidxplorer._region_viewer import (  # noqa: E402
    _GLOBAL,
    _INHERIT,
    _SETTING_BASELINE,
    ViewDefaults,
    ViewerManager,
    ViewSettings,
)

from .conftest import CH_IN_YAML, CH_NOT_IN_YAML, REGIONS  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixture)

# Settings describing WHAT you're looking at; none may have a global default.
PER_WINDOW = ("ndisplay", "region_id", "z_level", "time_point", "roi_bbox")


def test_only_how_you_look_settings_have_global_defaults():
    """The table, as an assertion. Adding a per-window setting to the defaults breaks this."""
    assert set(_SETTING_BASELINE) == {"tenengrad_focus", "channel_visibility", "luts"}, (
        "the set of global defaults changed; if that was deliberate, the classification in "
        "2026-07-29-gui-backlog-plan.md Task 6 has to change with it")
    for name in PER_WINDOW:
        assert name not in _SETTING_BASELINE, (
            f"{name!r} describes WHAT a window is looking at, so it cannot have a global default")


def test_the_contrast_rule_is_one_rule():
    """Contrast inherits from whoever opened the window. That single line is both written rules."""
    assert _SETTING_BASELINE["luts"] == _INHERIT
    assert _SETTING_BASELINE["tenengrad_focus"] == _GLOBAL
    assert _SETTING_BASELINE["channel_visibility"] == _GLOBAL


def test_asking_the_defaults_for_a_per_window_setting_raises_and_says_why():
    d = ViewDefaults()
    for name in ("ndisplay", "z_level", "time_point"):
        with pytest.raises(KeyError) as caught:
            d.get(name)
        assert "per-window" in str(caught.value), (
            "the refusal has to say WHY, or the next person adds the field")
        with pytest.raises(KeyError):
            d.set(name, 1)
        with pytest.raises(KeyError):
            ViewSettings().get(name)


def test_defaults_start_with_no_opinion():
    """Stock defaults must change nothing, so this object existing is not itself a behaviour."""
    d = ViewDefaults()
    assert d.tenengrad_focus is False
    assert d.channel_visibility == {}
    assert d.luts == {}


def test_a_windows_settings_are_a_private_copy_of_the_defaults():
    d = ViewDefaults()
    d.set("luts", {CH_IN_YAML: {"clim": (5.0, 50.0), "cmap": "red"}})
    s = ViewSettings(d.snapshot())

    s.get("luts")[CH_IN_YAML]["clim"] = (0.0, 0.0)      # a caller mutating what it was handed
    assert s.get("luts") == {CH_IN_YAML: {"clim": (5.0, 50.0), "cmap": "red"}}

    s.set("luts", {CH_IN_YAML: {"clim": (1.0, 2.0), "cmap": "red"}})
    assert d.luts == {CH_IN_YAML: {"clim": (5.0, 50.0), "cmap": "red"}}, (
        "a window wrote through to the global default; they are sharing one dict")


def test_setting_a_value_back_to_the_baseline_clears_the_override():
    """A sticky divergence marker after the value returns to baseline would be its own lie."""
    s = ViewSettings(ViewDefaults().snapshot())
    assert s.diverged == ()
    assert s.set("tenengrad_focus", True) is True
    assert s.diverged == ("tenengrad_focus",)
    assert s.set("tenengrad_focus", False) is False
    assert s.diverged == ()


def test_reset_goes_back_to_what_the_window_opened_with_not_to_a_later_default():
    """An ROI child's baseline is its parent's contrast, not the global default."""
    parent_luts = {CH_IN_YAML: {"clim": (7.0, 99.0), "cmap": "red"}}
    s = ViewSettings({"tenengrad_focus": False, "channel_visibility": {}, "luts": parent_luts})

    s.set("luts", {CH_IN_YAML: {"clim": (0.0, 1.0), "cmap": "red"}})
    assert s.is_diverged("luts")
    assert s.reset() == ("luts",)
    assert s.get("luts") == parent_luts
    assert s.diverged == ()
    assert s.reset() == (), "a second reset claimed to move something"


def test_adopt_makes_the_current_values_the_new_baseline():
    """After "make this the default" the window IS the default, so it must report no divergence."""
    s = ViewSettings(ViewDefaults().snapshot())
    s.set("tenengrad_focus", True)
    assert s.is_diverged()
    s.adopt()
    assert s.diverged == ()
    assert s.get("tenengrad_focus") is True
    assert s.baseline("tenengrad_focus") is True


# `napari_pane_stub` replaces only the one seam needing a GL context (`make_pane`); everything
# below is otherwise production code against a real ViewerManager and RegionViewer.

_LUT_A = {CH_IN_YAML: {"clim": (11.0, 111.0), "cmap": "red"},
          CH_NOT_IN_YAML: {"clim": (22.0, 222.0), "cmap": "green"}}
_LUT_B = {CH_IN_YAML: {"clim": (33.0, 333.0), "cmap": "red"},
          CH_NOT_IN_YAML: {"clim": (44.0, 444.0), "cmap": "green"}}


@pytest.fixture
def manager(qapp, napari_pane_stub, squid_dataset):
    """A real ViewerManager over the real reader, with no PlateWindow in the way.

    Windows use WA_DeleteOnClose, so close() only schedules deletion; drain and collect happen
    here, with the app alive, rather than in the middle of an unrelated later test.
    """
    from squidxplorer import open_reader

    root, _arrays = squid_dataset
    reader = open_reader(str(root))
    mgr = ViewerManager(reader, reader.metadata)
    try:
        yield mgr
    finally:
        mgr._mem_timer.stop()
        mgr.close_all()
        for _ in range(20):
            qapp.processEvents()
        gc.collect()
        qapp.processEvents()


def _loaded(qapp, win):
    """Wait until this window's first mosaic is on screen AND its settings have been applied."""
    assert _drain_until(qapp, lambda: win._settings_applied, timeout=30), (
        f"view {win.window_id} never finished loading, so its settings never landed")
    return win


def _layer_clims(win):
    """(channel -> contrast_limits) as they actually sit on this window's layers."""
    mosaic = win._pane.mosaic
    out = {}
    for ch in (CH_IN_YAML, CH_NOT_IN_YAML):
        layer = mosaic.find("raw", ch)
        if layer is not None:
            # napari reports contrast_limits as a LIST; the value is the assertion, not the type
            out[ch] = tuple(layer.contrast_limits) if layer.contrast_limits is not None else None
    return out


def test_a_new_window_inherits_the_current_defaults(qapp, manager):
    """The defaults are read at construction, so a window opened after a change carries it."""
    manager.defaults.set("tenengrad_focus", True)
    manager.defaults.set("channel_visibility", {CH_NOT_IN_YAML: False})
    manager.defaults.set("luts", _LUT_A)

    win = manager.open([REGIONS[0]])
    assert win is not None

    assert win.settings.get("tenengrad_focus") is True
    assert win.settings.get("channel_visibility") == {CH_NOT_IN_YAML: False}
    assert win.settings.get("luts") == _LUT_A
    assert win.settings.diverged == (), (
        "a window that only read the defaults is reporting divergence")

    _loaded(qapp, win)
    assert _layer_clims(win) == {CH_IN_YAML: (11.0, 111.0), CH_NOT_IN_YAML: (22.0, 222.0)}, (
        "the default contrast never reached the layers, so the default is a dead setting")


def test_an_roi_child_inherits_its_parents_contrast_not_the_global_default(qapp, manager):
    """The default and the parent are deliberately different, so inheriting is distinguishable
    from defaulting."""
    manager.defaults.set("luts", _LUT_A)
    parent = manager.open([REGIONS[0]])
    _loaded(qapp, parent)

    # Read live off the layers, not the record: napari writes contrast to the layer, not the record.
    for ch, lut in _LUT_B.items():
        parent._pane.mosaic.find("raw", ch).contrast_limits = lut["clim"]

    child = manager.open_child([REGIONS[0]], roi_bbox=(0.0, 0.0, 5.0, 5.0),
                               parent_id=parent.window_id)
    assert child is not None

    # Field by field, not by dict equality: a live LUT record also carries derived fields (e.g.
    # `rgb`) unrelated to inheritance, so pinning the whole dict would false-alarm on those.
    inherited = child.settings.get("luts")
    assert set(inherited) == {CH_IN_YAML, CH_NOT_IN_YAML}
    for ch, expected in _LUT_B.items():
        assert inherited[ch]["clim"] == expected["clim"], (
            "the ROI child took the global default instead of its parent's contrast")
        assert inherited[ch]["cmap"] == expected["cmap"]
    for ch, defaulted in _LUT_A.items():
        assert inherited[ch]["clim"] != defaulted["clim"]

    assert child.settings.diverged == (), (
        "a child that inherited a parent's contrast is reporting itself diverged; it is showing "
        "exactly what it opened with")

    _loaded(qapp, child)
    assert _layer_clims(child) == {CH_IN_YAML: (33.0, 333.0), CH_NOT_IN_YAML: (44.0, 444.0)}, (
        "the inherited contrast never reached the child's layers")

    # The other half of the same rule: a plate-opened window has no opener, so "inherit" hands it
    # the global default.
    sibling = manager.open([REGIONS[1]])
    assert sibling.settings.get("luts") == _LUT_A, (
        "a plate-opened window inherited from somewhere; its opener is the default")


def test_an_roi_child_takes_the_global_default_for_the_global_settings(qapp, manager):
    """Only contrast inherits; autofocus and channel visibility are global defaults for every
    window, ROI children included."""
    manager.defaults.set("tenengrad_focus", False)
    manager.defaults.set("channel_visibility", {CH_NOT_IN_YAML: True})
    parent = manager.open([REGIONS[0]])
    _loaded(qapp, parent)
    parent.settings.set("tenengrad_focus", True)
    parent.settings.set("channel_visibility", {CH_NOT_IN_YAML: False})

    child = manager.open_child([REGIONS[0]], roi_bbox=(0.0, 0.0, 5.0, 5.0),
                               parent_id=parent.window_id)
    assert child.settings.get("tenengrad_focus") is False, (
        "autofocus inherited from the parent; it is a global default")
    assert child.settings.get("channel_visibility") == {CH_NOT_IN_YAML: True}


def test_changing_a_setting_in_one_window_does_not_change_another(qapp, manager):
    """Through the real control the user clicks, not through the model behind it."""
    manager.defaults.set("luts", _LUT_A)
    one = manager.open([REGIONS[0]])
    two = manager.open([REGIONS[1]])
    _loaded(qapp, one)
    _loaded(qapp, two)

    one._focus_default_chk.setChecked(True)
    for ch, lut in _LUT_B.items():
        one._pane.mosaic.find("raw", ch).contrast_limits = lut["clim"]
    one.settings.set("luts", one._per_channel_luts())

    assert two.settings.get("tenengrad_focus") is False
    assert two.settings.get("luts") == _LUT_A
    assert two.settings.diverged == (), "window two diverged because window one was touched"
    assert _layer_clims(two) == {CH_IN_YAML: (11.0, 111.0), CH_NOT_IN_YAML: (22.0, 222.0)}
    assert manager.defaults.tenengrad_focus is False, (
        "a per-window change wrote through to the global default")
    assert manager.defaults.luts == _LUT_A


def test_a_diverged_window_says_so_in_the_window_and_reset_clears_it(qapp, manager):
    one = manager.open([REGIONS[0]])
    _loaded(qapp, one)

    assert one.settings.diverged == ()
    assert one._diverged_label.text() == "at the defaults"
    assert one._reset_btn.isEnabled() is False, (
        "reset offers itself with nothing to reset")

    one._focus_default_chk.setChecked(True)

    assert one.settings.diverged == ("tenengrad_focus",)
    assert one.settings.is_diverged("tenengrad_focus") is True
    assert "diverged" in one._diverged_label.text()
    assert "auto focus" in one._diverged_label.text(), (
        "the marker has to name WHICH setting diverged, or it cannot be acted on")
    assert one._reset_btn.isEnabled() is True

    one._reset_settings()

    assert one.settings.diverged == ()
    assert one.settings.get("tenengrad_focus") is False
    assert one._focus_default_chk.isChecked() is False, (
        "the control still shows the overridden value after a reset")
    assert one._diverged_label.text() == "at the defaults"
    assert one._reset_btn.isEnabled() is False


def test_every_control_in_the_defaults_box_reaches_the_shared_console(qapp, manager, caplog):
    """A quiet control next to a loud one reads as a control that did nothing, so all three of
    the settings-box controls must speak in the console."""
    import logging

    one = manager.open([REGIONS[0]])
    _loaded(qapp, one)

    def _console_lines(fn) -> list[str]:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=one.log.logger.name):
            fn()
        return [r.getMessage() for r in caplog.records]

    ticked = _console_lines(lambda: one._focus_default_chk.setChecked(True))
    assert any("auto focus" in m for m in ticked), (
        "ticking auto focus said nothing in the console; the other two controls in its box do")

    reset = _console_lines(one._reset_settings)
    assert any("auto focus" in m for m in reset), (
        "reset stopped naming what it put back")

    one._focus_default_chk.setChecked(True)
    made = _console_lines(one._make_default)
    assert made, "make default said nothing in the console"


def test_the_auto_focus_tooltip_describes_the_once_per_window_behaviour(qapp, manager):
    """The tooltip used to promise a refocus on every region change, which the code does not do."""
    one = manager.open([REGIONS[0]])
    tip = one._focus_default_chk.toolTip()

    assert "once" in tip.lower(), "the tooltip does not say the jump happens once"
    assert "whenever this window loads a region" not in tip, (
        "the tooltip still promises a per-region refocus")

    _loaded(qapp, one)
    assert one._settings_applied is True
    called = []
    one._focus_reference_plane = lambda: called.append(1)
    one.settings.set("tenengrad_focus", True)
    one._apply_settings_once()
    assert called == [], (
        "_apply_settings_once ran a second time; the tooltip's 'once' is now the wrong description")


def test_resetting_contrast_puts_the_layers_back(qapp, manager):
    """Reset is not just a flag: the pixels on screen have to go back too."""
    manager.defaults.set("luts", _LUT_A)
    one = manager.open([REGIONS[0]])
    _loaded(qapp, one)

    for ch, lut in _LUT_B.items():
        one._pane.mosaic.find("raw", ch).contrast_limits = lut["clim"]
    one.settings.set("luts", one._per_channel_luts())
    assert one.settings.is_diverged("luts")

    one._reset_settings()

    assert _layer_clims(one) == {CH_IN_YAML: (11.0, 111.0), CH_NOT_IN_YAML: (22.0, 222.0)}
    assert one.settings.diverged == ()


def test_changing_the_default_does_not_retroactively_change_a_diverged_window(qapp, manager):
    """The case where a retroactive write would destroy work the user did by hand."""
    manager.defaults.set("luts", _LUT_A)
    one = manager.open([REGIONS[0]])
    _loaded(qapp, one)

    for ch, lut in _LUT_B.items():
        one._pane.mosaic.find("raw", ch).contrast_limits = lut["clim"]
    one.settings.set("luts", one._per_channel_luts())
    one._focus_default_chk.setChecked(True)
    assert set(one.settings.diverged) == {"luts", "tenengrad_focus"}

    later = {CH_IN_YAML: {"clim": (900.0, 999.0), "cmap": "red"}}
    manager.defaults.set("luts", later)
    manager.defaults.set("tenengrad_focus", True)
    qapp.processEvents()

    assert one.settings.get("luts")[CH_IN_YAML]["clim"] == (33.0, 333.0), (
        "the new default overwrote a window the user had already adjusted")
    assert _layer_clims(one)[CH_IN_YAML] == (33.0, 333.0)
    assert set(one.settings.diverged) == {"luts", "tenengrad_focus"}, (
        "the window stopped reporting diverged because the default moved under it")
    assert one.settings.baseline("luts") == _LUT_A, (
        "the baseline moved; reset would now go somewhere the window never was")

    nxt = manager.open([REGIONS[1]])
    assert nxt.settings.get("luts") == later
    assert nxt.settings.get("tenengrad_focus") is True


def test_make_default_is_the_only_outward_push_and_leaves_open_windows_alone(qapp, manager):
    """The affordance that replaces propagation: an explicit act, aimed at future windows."""
    manager.defaults.set("luts", _LUT_A)
    one = manager.open([REGIONS[0]])
    two = manager.open([REGIONS[1]])
    _loaded(qapp, one)
    _loaded(qapp, two)

    for ch, lut in _LUT_B.items():
        one._pane.mosaic.find("raw", ch).contrast_limits = lut["clim"]
    one.settings.set("luts", one._per_channel_luts())
    one._focus_default_chk.setChecked(True)

    one._make_default()

    assert manager.defaults.tenengrad_focus is True
    assert manager.defaults.luts[CH_IN_YAML]["clim"] == (33.0, 333.0)
    assert one.settings.diverged == (), (
        "the window that IS the default still claims to diverge from it")

    # Window two, already open, is deliberately untouched.
    assert two.settings.get("tenengrad_focus") is False
    assert two.settings.get("luts") == _LUT_A
    assert _layer_clims(two) == {CH_IN_YAML: (11.0, 111.0), CH_NOT_IN_YAML: (22.0, 222.0)}

    three = manager.open([REGIONS[0]])
    assert three.settings.get("tenengrad_focus") is True
    assert three.settings.get("luts")[CH_IN_YAML]["clim"] == (33.0, 333.0)


def test_make_default_on_a_closed_window_refuses_rather_than_appearing_to_work(qapp, manager):
    one = manager.open([REGIONS[0]])
    wid = one.window_id
    one.close()
    for _ in range(10):
        qapp.processEvents()
    assert manager.make_default(wid) is False


def test_pasting_luts_marks_the_window_diverged(qapp, manager):
    """A paste IS the user changing contrast here, so the window has to admit it moved."""
    from squidxplorer import _region_viewer as RV

    manager.defaults.set("luts", _LUT_A)
    one = manager.open([REGIONS[0]])
    two = manager.open([REGIONS[1]])
    _loaded(qapp, one)
    _loaded(qapp, two)

    for ch, lut in _LUT_B.items():
        one._pane.mosaic.find("raw", ch).contrast_limits = lut["clim"]
    one._copy_luts()
    two._paste_luts()

    assert two.settings.is_diverged("luts") is True
    assert "contrast" in two._diverged_label.text()
    assert _layer_clims(two) == {CH_IN_YAML: (33.0, 333.0), CH_NOT_IN_YAML: (44.0, 444.0)}
    assert one.settings.diverged == (), "the copy diverged the window it copied FROM"
    RV._LUT_CLIPBOARD.clear()


def test_copy_paste_luts_is_the_only_contrast_path_between_two_open_windows(qapp, manager):
    """Pins that copy/paste is the only contrast path between two already-open windows: napari's
    `link_layers` cannot cross windows, `_baseline_for` only fires at open, and `make_default`
    only affects future windows. The copy is also the only carrier of the colormap."""
    from squidxplorer import _region_viewer as RV

    RV._LUT_CLIPBOARD.clear()
    one = manager.open([REGIONS[0]])
    two = manager.open([REGIONS[1]])
    _loaded(qapp, one)
    _loaded(qapp, two)

    before_clims = dict(_layer_clims(two))
    before_cmaps = {ch: two._pane.mosaic.find("raw", ch).colormap
                    for ch in (CH_IN_YAML, CH_NOT_IN_YAML)}

    for ch, lut in _LUT_B.items():
        layer = one._pane.mosaic.find("raw", ch)
        layer.contrast_limits = lut["clim"]
        layer.colormap = "magenta"
    assert before_cmaps[CH_IN_YAML] != "magenta", "the fixture already used the colour under test"

    assert _layer_clims(two) == before_clims, (
        "window one's contrast reached an already-open window two on its own; if that is now real "
        "then this test, not the chips, is what is wrong")
    assert manager.make_default(one.window_id) is True
    assert _layer_clims(two) == before_clims, "make_default reached back into an open window"

    one._copy_luts()
    two._paste_luts()

    assert _layer_clims(two) == {CH_IN_YAML: (33.0, 333.0), CH_NOT_IN_YAML: (44.0, 444.0)}
    for ch in (CH_IN_YAML, CH_NOT_IN_YAML):
        cmap = two._pane.mosaic.find("raw", ch).colormap
        assert getattr(cmap, "name", cmap) == "magenta", (
            f"{ch}: the colormap did not travel, and no other mechanism carries it at all")
    RV._LUT_CLIPBOARD.clear()


def test_match_raw_contrast_is_wired_to_this_window_s_mosaic_and_leaves_it_at_defaults(
        qapp, manager):
    """The chip's handler must reach this window's own layers, and writing operator layers only
    (never raw) must not mark the window diverged."""
    one = manager.open([REGIONS[0]])
    _loaded(qapp, one)
    mosaic = one._pane.mosaic

    for ch in (CH_IN_YAML, CH_NOT_IN_YAML):
        raw = mosaic.find("raw", ch)
        if raw is None:
            continue
        raw.contrast_limits = (100.0, 900.0)
        peer = mosaic.add_mosaic("decon", ch, np.full((16, 16), 9000, dtype=np.uint16))
        peer.contrast_limits = (1.0, 2.0)
    diverged_before = one.settings.diverged

    one._match_raw_contrast()

    matched = 0
    for ch in (CH_IN_YAML, CH_NOT_IN_YAML):
        raw, peer = mosaic.find("raw", ch), mosaic.find("decon", ch)
        if raw is None or peer is None:
            continue
        assert list(peer.contrast_limits) == list(raw.contrast_limits), ch
        matched += 1
    assert matched, "the window had no raw/operator pair to match, so nothing was proven"
    assert one.settings.diverged == diverged_before, (
        "matching operator layers to raw moved the window's recorded settings; it writes "
        "operator layers only and raw is untouched"
    )


def test_the_autofocus_default_is_actually_read_when_a_window_loads(qapp, manager):
    """Off by default, so the stock default leaves the old behaviour exactly as it was."""
    off = manager.open([REGIONS[0]])
    _loaded(qapp, off)
    assert off._focus_worker is None, (
        "a window autofocused with the setting off")

    manager.defaults.set("tenengrad_focus", True)
    on = manager.open([REGIONS[1]])
    _loaded(qapp, on)
    assert on._focus_worker is not None, (
        "the autofocus default was never read, so it is a dead setting")


# A second window reuses the first window's operator result via `squidxplorer._recipe.RESULTS`.

def _fake_result(region, channels):
    """A finished operator result, shaped like what `PlateWindow._as_result` builds."""
    from squidxplorer._address import Extent
    from squidxplorer._result import Result

    planes = [np.full((16, 16), 700 + i, dtype=np.uint16) for i, _ in enumerate(channels)]
    return Result.of(Extent(region_id=region), planes, channels=tuple(channels),
                     z_depth=1, pixel_size_um=0.325, dtype="uint16")


def test_a_second_window_reuses_the_first_windows_result_without_recomputing(qapp, manager):
    """Open A, compute once, open B on the same region: B gains A's very `Result` object, with
    no re-fuse and no copy."""
    from squidxplorer._recipe import acquisition_version, cache_operator_result

    region = REGIONS[0]
    channels = [CH_IN_YAML, CH_NOT_IN_YAML]

    first = manager.open([region])
    _loaded(qapp, first)

    result = _fake_result(region, channels)
    assert first.deliver_result("mip", result, visible=True) == len(channels)
    cache_operator_result("mip", result, acquisition_version(manager._reader))

    second = manager.open([region])
    _loaded(qapp, second)

    layers = {ch: second._pane.mosaic.find("mip", ch) for ch in channels}
    assert all(ly is not None for ly in layers.values()), (
        f"the second window did not gain the already-computed layers: {layers}")
    for ch, ly in layers.items():
        assert ly.visible is False, (
            f"{ch}: this window did not ask for the run, so its layer must arrive dark")
    # Identity, not equality, is the proof of reuse.
    for i, ch in enumerate(channels):
        assert layers[ch].data is result.plane(ch), (
            f"{ch}: the second window was handed different pixels, so something recomputed")
        assert int(np.asarray(layers[ch].data)[0, 0]) == 700 + i


def test_a_window_opened_on_a_DIFFERENT_region_gains_nothing(qapp, manager):
    """A cached region's result must not land in a window showing a different region."""
    from squidxplorer._recipe import acquisition_version, cache_operator_result

    channels = [CH_IN_YAML, CH_NOT_IN_YAML]
    cache_operator_result("mip", _fake_result(REGIONS[0], channels),
                          acquisition_version(manager._reader))

    other = manager.open([REGIONS[1]])
    _loaded(qapp, other)
    assert all(other._pane.mosaic.find("mip", ch) is None for ch in channels)
    assert other._result_region is None


def test_a_pasted_lut_carries_CHANNEL_VISIBILITY_not_just_contrast(qapp, manager):
    """The record has four keys and the paste used to apply two: a window with a channel switched
    OFF pasted its LUTs and the target kept the channel lit — a silent partial paste."""
    from squidxplorer import _region_viewer as RV

    RV._LUT_CLIPBOARD.clear()
    one = manager.open([REGIONS[0]])
    two = manager.open([REGIONS[1]])
    _loaded(qapp, one)
    _loaded(qapp, two)

    one._pane.mosaic.set_channel_visible(CH_IN_YAML, False)
    assert two._pane.mosaic.channel_visible(CH_IN_YAML) is not False

    one._copy_luts()
    two._paste_luts()

    assert two._pane.mosaic.channel_visible(CH_IN_YAML) is False, (
        "the copied window had this channel switched OFF; the paste dropped that")
    RV._LUT_CLIPBOARD.clear()
