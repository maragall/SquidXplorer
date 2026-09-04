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
    _RAW_OP,
    _SETTING_BASELINE,
    ViewDefaults,
    ViewerManager,
    ViewSettings,
)

from .conftest import CH_IN_YAML, CH_NOT_IN_YAML, REGIONS  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixture)

# Settings describing WHAT you're looking at; none may have a global default.
PER_WINDOW = ("ndisplay", "region_id", "z_level", "time_point", "roi_bbox")


def test_only_how_you_look_settings_have_global_defaults_and_contrast_inherits():
    """The table, as an assertion. Adding a per-window setting to the defaults breaks this."""
    assert set(_SETTING_BASELINE) == {"tenengrad_focus", "channel_visibility", "luts"}, (
        "the set of global defaults changed; if that was deliberate, the classification in "
        "2026-07-29-gui-backlog-plan.md Task 6 has to change with it")
    for name in PER_WINDOW:
        assert name not in _SETTING_BASELINE, (
            f"{name!r} describes WHAT a window is looking at, so it cannot have a global default")
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
    """A real ViewerManager over the real reader, with no PlateWindow in the way."""
    from squidxplorer import open_reader

    root, _arrays = squid_dataset
    reader = open_reader(str(root))
    mgr = ViewerManager(reader, reader.metadata)
    try:
        yield mgr
    finally:
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
    """The default and the parent are deliberately different, so inheriting is distinguishable from defaulting."""
    manager.defaults.set("luts", _LUT_A)
    parent = manager.open([REGIONS[0]])
    _loaded(qapp, parent)

    for ch, lut in _LUT_B.items():
        parent._pane.mosaic.find("raw", ch).contrast_limits = lut["clim"]

    child = manager.open_child([REGIONS[0]], roi_bbox=(0.0, 0.0, 5.0, 5.0),
                               parent_id=parent.window_id)
    assert child is not None

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

    sibling = manager.open([REGIONS[1]])
    assert sibling.settings.get("luts") == _LUT_A, (
        "a plate-opened window inherited from somewhere; its opener is the default")


def test_an_roi_child_takes_the_global_default_for_the_global_settings(qapp, manager):
    """Only contrast inherits; autofocus and channel visibility are global defaults for every window, ROI children included."""
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

    one.settings.set("tenengrad_focus", True)   # the Defaults box is shelved; the STORE remains
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


def test_auto_focus_applies_once(qapp, manager):
    one = manager.open([REGIONS[0]])
    _loaded(qapp, one)
    assert one._settings_applied is True
    called = []
    one._focus_reference_plane = lambda: called.append(1)
    one.settings.set("tenengrad_focus", True)
    one._apply_settings_once()
    assert called == [], (
        "_apply_settings_once ran a second time; the tooltip's 'once' is now the wrong description")


def test_changing_the_default_does_not_retroactively_change_a_diverged_window(qapp, manager):
    """The case where a retroactive write would destroy work the user did by hand."""
    manager.defaults.set("luts", _LUT_A)
    one = manager.open([REGIONS[0]])
    _loaded(qapp, one)

    for ch, lut in _LUT_B.items():
        one._pane.mosaic.find("raw", ch).contrast_limits = lut["clim"]
    one.settings.set("luts", one._per_channel_luts())
    one.settings.set("tenengrad_focus", True)
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
    """Open A, compute once, open B on the same region: B gains A's very `Result` object, with no re-fuse and no copy."""
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
    for i, ch in enumerate(channels):
        assert layers[ch].data is result.plane(ch), (
            f"{ch}: the second window was handed different pixels, so something recomputed")
        assert int(np.asarray(layers[ch].data)[0, 0]) == 700 + i


def test_loading_a_NEW_dataset_forgets_the_last_ones_look_and_contrast_ceiling(manager):
    """A contrast window is a statement in the PREVIOUS acquisition's counts."""
    from squidxplorer import _bitdepth

    manager.defaults.set("luts", _LUT_A)
    manager.defaults.set("channel_visibility", {CH_IN_YAML: False})
    manager.defaults.set("tenengrad_focus", True)
    _bitdepth.depth().observe(3437.0)
    assert _bitdepth.range_for(np.uint16) == (0.0, 4095.0)

    manager.set_dataset(manager._reader, manager._meta)

    assert manager.defaults.get("luts") == {}
    assert manager.defaults.get("channel_visibility") == {}
    assert manager.defaults.get("tenengrad_focus") is True
    assert _bitdepth.depth().observed is None
    assert _bitdepth.range_for(np.uint16) == (0.0, 65535.0)


def test_a_ceiling_rise_reaches_a_window_that_is_ALREADY_open(qapp, manager):
    """THE test."""
    from squidxplorer import _bitdepth

    import numpy as np

    _bitdepth.depth().observe_array(np.array([[3437]], np.uint16), CH_IN_YAML)
    win = manager.open([REGIONS[0]])
    _loaded(qapp, win)

    layer = win._pane.mosaic.find(_RAW_OP, CH_IN_YAML)
    assert layer is not None
    assert tuple(layer.contrast_limits_range) == (0.0, _bitdepth.channel_ceiling(3437.0))
    layer.contrast_limits = (100.0, 3000.0)     # the window the user is looking through

    _bitdepth.depth().observe_array(np.array([[16380]], np.uint16), CH_IN_YAML)   # E7 is fused
    for _ in range(5):
        qapp.processEvents()                    # the rise is queued to the GUI thread

    assert tuple(layer.contrast_limits_range) == (0.0, 16383.0)
    assert tuple(layer.contrast_limits) == (100.0, 3000.0)


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


def test_applying_a_lut_record_carries_CHANNEL_VISIBILITY_not_just_contrast(qapp, manager):
    """The record has four keys and `apply_luts` puts three on the layers (`rgb` is the plate's spelling)."""
    one = manager.open([REGIONS[0]])
    two = manager.open([REGIONS[1]])
    _loaded(qapp, one)
    _loaded(qapp, two)

    one._pane.mosaic.set_channel_visible(CH_IN_YAML, False)
    assert two._pane.mosaic.channel_visible(CH_IN_YAML) is not False

    two._apply_luts(one._per_channel_luts())

    assert two._pane.mosaic.channel_visible(CH_IN_YAML) is False, (
        "the source window had this channel switched OFF; applying its record dropped that")


def test_the_users_contrast_survives_a_region_round_trip(qapp, manager):
    """A region move rebuilds the raw layers; the look on screen must ride the move.

    Measured before the fix: the user's (500, 1234) came back as a fresh auto seed
    (2969.0, 2994.5) after B2 -> B3 -> B2, because `load_mosaic` removed the raw layers
    without reading them first."""
    win = manager.open(list(REGIONS))
    _loaded(qapp, win)
    win._pane.mosaic.set_contrast(CH_IN_YAML, 500.0, 1234.0)

    assert win.show_region(REGIONS[1])
    assert _drain_until(qapp, lambda: win._shown_region == REGIONS[1], timeout=30)
    assert win.show_region(REGIONS[0])
    assert _drain_until(qapp, lambda: win._shown_region == REGIONS[0], timeout=30)

    assert _layer_clims(win)[CH_IN_YAML] == (500.0, 1234.0), (
        "the round trip handed the user's window back to the auto seed")
