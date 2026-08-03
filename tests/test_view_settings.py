"""Settings scope: which settings are global defaults, which are per-window, and who says so.

WHAT WAS WRONG
--------------
Every display setting was per-window by accident rather than by decision, and the one exception
was hand-carried. ``RegionViewer`` took an ``initial_luts`` dict, and the ONLY caller that filled
it was ``_open_roi_children``, which read the parent's layers itself and posted them down. Three
consequences, and they compound:

1. **The rule lived at the call sites.** Whether a new window inherited anything depended on which
   line opened it. A second opening site (the plate's "Open view", a future gallery double-click, a
   restored session) would have to remember to do the same read, and any that forgot produced a
   window whose contrast disagreed with everything else for no stated reason.
2. **There were no defaults at all.** Turning autofocus on, or hiding a channel, was a per-window
   act repeated in every window forever. Julio's call, 2026-07-29: global default with a per-window
   override, under the rule **settings that describe HOW you look are global defaults; settings
   that describe WHAT you are looking at are per-window.**
3. **A window could not say which state it was in.** With a default in the picture that is the
   dangerous part: a global default that silently disagrees with what is on screen is worse than no
   default at all. The same objection killed a silent fallback between coordinate sources and forced
   the compact placement mode to label itself.

WHAT IS PINNED HERE
-------------------
* ``ViewDefaults`` is owned by ``ViewerManager``, not by a window: windows come and go, the registry
  is what a default can outlive.
* A new window reads the defaults ONCE, at construction. A later change to the default reaches the
  NEXT window and never an already-open one -- asserted for a diverged window specifically, which
  is the case where a retroactive write would destroy work the user did by hand.
* **ONE contrast rule, not two.** The decision was written as two ("the global default for a window
  opened from the plate, the parent's LUTs for an ROI child"); ``_SETTING_BASELINE["luts"] ==
  "inherit"`` is both of them, because a plate-opened window's opener IS the default. Both halves
  are asserted through the real ``ViewerManager`` path, including that an ROI child gets its
  parent's contrast and NOT the global default when the two differ.
* Divergence is EXPLICIT (the set of settings overridden in this window), not computed against
  today's default. So changing a default cannot light a marker on a window nobody touched, and an
  ROI child that inherited a diverged parent's contrast is not itself diverged.
* Divergence is VISIBLE, in the window, with reset enabled only when there is something to reset.
* One window's change never reaches another. "make default" is the only outward push, it is a
  deliberate act, and it still leaves every open window alone.
* The per-window settings (2D/3D, the region, z, time_point) have NO slot in the defaults object,
  and asking for one is a ``KeyError`` that says why. A silence is easy to fill in by accident; a
  raise is not.
"""

from __future__ import annotations

import gc
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

# The ONE guard in this file, and it is an ENVIRONMENT gate rather than a skipped assertion: PyQt5
# is an optional extra (`.[gui]`), `squidmip._region_viewer` imports it at module scope, so without
# it there is nothing here to test rather than something being waved past. Every GUI test file in
# this suite gates the same way. Nothing below is conditional, marked, or xfailed.
pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from squidmip._region_viewer import (  # noqa: E402
    _GLOBAL,
    _INHERIT,
    _SETTING_BASELINE,
    ViewDefaults,
    ViewerManager,
    ViewSettings,
)

from .conftest import CH_IN_YAML, CH_NOT_IN_YAML, REGIONS  # noqa: E402
from .test_viewer import _drain_until, qapp  # noqa: E402,F401  (fixture)

# --------------------------------------------------------------------------------------
# The rule itself, as data. No Qt: this half is a model and is tested as one.
# --------------------------------------------------------------------------------------

#: The settings Julio classified as PER-WINDOW. None of them may acquire a global default, because
#: they describe WHAT you are looking at, and a default for that is a default for someone else's
#: subject. Spelled in the codebase's own vocabulary (Squid's words for the physical dimensions).
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
    """The marker means "not what this window opened with", so it cannot survive the value going
    back. A sticky marker is a lie in the same direction as no marker at all."""
    s = ViewSettings(ViewDefaults().snapshot())
    assert s.diverged == ()
    assert s.set("tenengrad_focus", True) is True
    assert s.diverged == ("tenengrad_focus",)
    assert s.set("tenengrad_focus", False) is False
    assert s.diverged == ()


def test_reset_goes_back_to_what_the_window_opened_with_not_to_a_later_default():
    """An ROI child's baseline is its parent's contrast, which the global default never held. Reset
    has to mean that baseline, or reset would silently retarget the child at the plate."""
    parent_luts = {CH_IN_YAML: {"clim": (7.0, 99.0), "cmap": "red"}}
    s = ViewSettings({"tenengrad_focus": False, "channel_visibility": {}, "luts": parent_luts})

    s.set("luts", {CH_IN_YAML: {"clim": (0.0, 1.0), "cmap": "red"}})
    assert s.is_diverged("luts")
    assert s.reset() == ("luts",)
    assert s.get("luts") == parent_luts
    assert s.diverged == ()
    assert s.reset() == (), "a second reset claimed to move something"


def test_adopt_makes_the_current_values_the_new_baseline():
    """After "make this the default" the window IS the default, so a marker still claiming
    divergence would be the silent disagreement this affordance exists to stop."""
    s = ViewSettings(ViewDefaults().snapshot())
    s.set("tenengrad_focus", True)
    assert s.is_diverged()
    s.adopt()
    assert s.diverged == ()
    assert s.get("tenengrad_focus") is True
    assert s.baseline("tenengrad_focus") is True


# --------------------------------------------------------------------------------------
# The same rule through the REAL registry and REAL windows
# --------------------------------------------------------------------------------------
#
# `napari_pane_stub` replaces the one seam that needs a GL context (`make_pane`, which is false
# under QT_QPA_PLATFORM=offscreen and would send `_build` down its "napari unavailable" branch).
# Everything below it is production code: the real ViewerManager, the real RegionViewer, the real
# `_MosaicWorker`. napari's own rendering is not exercised here and never was under offscreen.

_LUT_A = {CH_IN_YAML: {"clim": (11.0, 111.0), "cmap": "red"},
          CH_NOT_IN_YAML: {"clim": (22.0, 222.0), "cmap": "green"}}
_LUT_B = {CH_IN_YAML: {"clim": (33.0, 333.0), "cmap": "red"},
          CH_NOT_IN_YAML: {"clim": (44.0, 444.0), "cmap": "green"}}


@pytest.fixture
def manager(qapp, napari_pane_stub, squid_dataset):
    """A real ViewerManager over the real reader, with no PlateWindow in the way.

    The windows this hands out set WA_DeleteOnClose, so close() only SCHEDULES deletion; the drain
    and the collect happen HERE, with the app alive, rather than letting a Qt wrapper whose C++ half
    is gone be collected in the middle of an unrelated later test.
    """
    from squidmip import open_reader

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
            out[ch] = layer.contrast_limits
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
    """THE contrast rule. The default and the parent are deliberately different, so inheriting is
    distinguishable from defaulting -- which is the whole assertion."""
    manager.defaults.set("luts", _LUT_A)
    parent = manager.open([REGIONS[0]])
    _loaded(qapp, parent)

    # The user drags contrast in napari, which writes the LAYERS. That is why the parent is read
    # live rather than out of its record: the layers are the only thing that knows the answer.
    for ch, lut in _LUT_B.items():
        parent._pane.mosaic.find("raw", ch).contrast_limits = lut["clim"]

    child = manager.open_child([REGIONS[0]], roi_bbox=(0.0, 0.0, 5.0, 5.0),
                               parent_id=parent.window_id)
    assert child is not None

    assert child.settings.get("luts") == {
        CH_IN_YAML: {"clim": (33.0, 333.0), "cmap": "red"},
        CH_NOT_IN_YAML: {"clim": (44.0, 444.0), "cmap": "green"},
    }, "the ROI child took the global default instead of its parent's contrast"
    assert child.settings.get("luts") != _LUT_A

    assert child.settings.diverged == (), (
        "a child that inherited a parent's contrast is reporting itself diverged; it is showing "
        "exactly what it opened with")

    _loaded(qapp, child)
    assert _layer_clims(child) == {CH_IN_YAML: (33.0, 333.0), CH_NOT_IN_YAML: (44.0, 444.0)}, (
        "the inherited contrast never reached the child's layers")

    # The OTHER half of the same rule: a window opened from the PLATE has no opener window, so the
    # same "inherit" line hands it the global default. One rule, both origins.
    sibling = manager.open([REGIONS[1]])
    assert sibling.settings.get("luts") == _LUT_A, (
        "a plate-opened window inherited from somewhere; its opener is the default")


def test_an_roi_child_takes_the_global_default_for_the_global_settings(qapp, manager):
    """Only contrast inherits. Autofocus and channel visibility are global defaults for EVERY
    window, ROI children included, which is what the classification says."""
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

    one._focus_default_chk.setChecked(True)              # the user ticks it in window one
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

    # The default is a fact about the NEXT window, and that half must still work.
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

    # Window two, already open, is untouched. That is the decentralization, not an oversight.
    assert two.settings.get("tenengrad_focus") is False
    assert two.settings.get("luts") == _LUT_A
    assert _layer_clims(two) == {CH_IN_YAML: (11.0, 111.0), CH_NOT_IN_YAML: (22.0, 222.0)}

    # And it reaches the next window opened.
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
    from squidmip import _region_viewer as RV

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


def test_match_raw_contrast_is_wired_to_this_window_s_mosaic_and_leaves_it_at_defaults(
        qapp, manager):
    """The window-level half of "Match raw contrast": the chip's handler reaches THIS window's
    layers, and the action does not pretend the window's settings changed.

    Two claims, both of which a unit test on MosaicLayers cannot make:

    * the handler is bound to the right pane (a typo'd attribute would find no mosaic and say
      "no mosaic here" while looking like it worked);
    * it does NOT mark the window diverged. It writes operator layers only, never raw, so the
      window's recorded LUTs are byte-for-byte what they were -- unlike a paste, which moves raw
      and IS recorded (see the paste test above).
    """
    one = manager.open([REGIONS[0]])
    _loaded(qapp, one)
    mosaic = one._pane.mosaic

    # A fake operator result over the raw mosaic, on a deliberately DIFFERENT window from raw's.
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
    """A setting nobody reads is dead wiring. Off by default, so this also pins that the stock
    default leaves the old behaviour exactly as it was."""
    off = manager.open([REGIONS[0]])
    _loaded(qapp, off)
    assert off._focus_worker is None, (
        "a window autofocused with the setting off")

    manager.defaults.set("tenengrad_focus", True)
    on = manager.open([REGIONS[1]])
    _loaded(qapp, on)
    assert on._focus_worker is not None, (
        "the autofocus default was never read, so it is a dead setting")
