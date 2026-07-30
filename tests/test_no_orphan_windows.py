"""Building the root window must not put a window on the user's desktop that nobody asked for.

2026-07-29, Task 9. Julio: a stray untitled window, about 129x59, floating with no home, on both
launches of a session. This is it, and the mechanism is a one-liner with a big consequence.

    self._focus_btn = QPushButton("Focus reference plane")     # no parent
    self._focus_btn.hide()

In Qt, a widget constructed with NO PARENT and never added to a layout is a TOP-LEVEL WINDOW. The
``hide()`` on the next line is why this was thought to be harmless. It is not, because
``_sync_focus_button`` then ran on every ingest and did::

    btn.setVisible(len((self._meta or {}).get("z_levels", [])) > 1)

So the button un-hid itself, as a bare 178x30 frameless window with no title, for any acquisition
with more than one z level -- which is most of them. Measured here before the fix, on the standard
z-stack fixture: ``QPushButton "Focus reference plane", top level, visible True, 178x30``. Nothing
else in the process was visible, because a headless test never calls ``show()``.

The backlog entry guessed at this and the plan recorded ``_update_focus_button`` as having "zero
call sites". Both were half right. The button really was an orphan and the reference-plane control
really did move onto each window's own z-slider in ``d07db43``; but the sync method was NOT
uncalled, and it is precisely what made the orphan visible. A hidden orphan is untidy. An orphan
that un-hides itself is a bug report.

WHAT IS PINNED HERE

1. Constructing and ingesting into a root window makes NOTHING visible. This is the general rule
   rather than the specific button, and it is exactly as strong as it should be: the caller owns
   ``show()``. Any future widget built without a parent and then made visible fails this, which is
   the whole class of defect rather than the one instance of it.
2. The dead reference-plane chain on ``PlateWindow`` is gone, and stays gone. The button, the sync
   method, the handler and the z-slider helper it fed all had exactly one entry point between them
   and it was the orphan's ``clicked``.

NOT pinned, deliberately: the several other parentless widgets ``PlateWindow`` keeps as hidden
orphans so that old call sites still resolve (a ``QStackedWidget``, a ``QComboBox``, the "3D native"
and "Return to raw view" buttons). They are documented as such in the source, they are never made
visible, and rule 1 above is what holds them to that.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless Qt; must precede PyQt import

import sys  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("PyQt5")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip(
        "PySide already loaded (napari/pytest-qt) — Qt binding conflict; run with "
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 to run the PyQt5 GUI tests.",
        allow_module_level=True,
    )

from PyQt5.QtWidgets import QApplication  # noqa: E402

from squidmip import _viewer as V  # noqa: E402

from .conftest import shutdown_plate_window  # noqa: E402
from .test_viewer import qapp, stub_detail  # noqa: E402,F401  (fixtures)


def _visible_top_levels():
    """Every top-level widget Qt currently considers visible, as readable descriptions."""
    out = []
    for w in QApplication.topLevelWidgets():
        try:
            if not w.isVisible():
                continue
        except RuntimeError:                      # a wrapper whose C++ half is already gone
            continue
        label = getattr(w, "text", None)
        out.append(f"{type(w).__name__}(title={w.windowTitle()!r}, "
                   f"text={label() if callable(label) else ''!r}, "
                   f"{w.width()}x{w.height()})")
    return out


def test_building_and_ingesting_shows_no_window_the_caller_did_not_open(
        qapp, stub_detail, squid_dataset):
    """A headless test never calls show(), so anything visible here showed ITSELF.

    MUTATION: put back ``self._focus_btn = QPushButton(...)`` with no parent plus a
    ``setVisible(True)`` on a z-stack -> the button appears in this list -> red.
    """
    root, _ = squid_dataset
    before = set(_visible_top_levels())

    win = V.PlateWindow(None)
    win.ingest(str(root))

    assert len(win._meta["z_levels"]) > 1, (
        "fixture invalid: the orphan only un-hid itself on a MULTI-z acquisition, so a "
        "single-plane fixture would pass this test without testing anything")
    strays = sorted(set(_visible_top_levels()) - before)
    assert strays == [], f"the root put {len(strays)} window(s) on screen by itself: {strays}"
    shutdown_plate_window(qapp, win)


def test_the_dead_reference_plane_chain_is_not_on_the_plate_window(qapp, stub_detail,
                                                                  squid_dataset):
    """Focus is per-window now (each window's own z-slider, ``d07db43``). The plate's copy was
    reachable only through the orphan button, so every link in it was dead code with a live
    ``setEnabled`` habit. Named one by one so a re-introduction is a failing test, not a review
    comment."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))

    for gone in ("_focus_btn", "_sync_focus_button", "_focus_reference_plane",
                 "_on_focus_problem", "_on_reference_plane", "_set_z_index"):
        assert not hasattr(win, gone), (
            f"PlateWindow.{gone} is back. The reference plane lives on each window's z-slider; "
            "a second copy on the plate is what produced the orphan window.")
    shutdown_plate_window(qapp, win)


def test_the_focus_worker_itself_survives_because_the_windows_use_it(qapp, stub_detail,
                                                                    squid_dataset):
    """Deleting the chain must not take the Tenengrad worker with it: ``RegionViewer`` imports
    ``_viewer._FocusWorker`` by name, so removing it would break the control that REPLACED the
    orphan."""
    assert hasattr(V, "_FocusWorker")
    from squidmip._region_viewer import RegionViewer

    assert hasattr(RegionViewer, "_focus_reference_plane")
    assert hasattr(RegionViewer, "_on_reference_plane")
