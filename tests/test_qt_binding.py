"""Which Qt binding this app actually runs on, and the one line only a real launch executes.

THE BUG THIS FILE EXISTS FOR. The Qt6 migration landed in three commits (10b8348 every import
through qtpy, f7f9b28 the Qt5-only APIs, ce5605c ndviewer_light removed) and the suite was green
under ``QT_API=pyqt6``. None of that made the app RUN on Qt6. qtpy picks a binding by preference
order, and its order starts at PyQt5, so on any machine with both installed the tree looked
migrated and quietly kept running Qt5 -- confirmed 2026-07-31, ``qtpy.API_NAME`` reporting
PyQt5 5.15.14 with PyQt6 6.11 installed alongside. "The suite passes under Qt6" and "the app uses
Qt6" are different claims and only the first one was ever tested.

That matters beyond tidiness: Qt5 rounds fractional display scale factors, so a 150% Windows
display renders every window two-thirds size, which is the "about 3x2 inches on a 4K monitor"
report from the demo machine. Qt6 passes the fraction through by default.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import subprocess
import sys

import pytest


def test_importing_squidmip_pins_the_qt_binding_to_qt6():
    """Importing the package must decide the binding, and must decide it as Qt6 where Qt6 exists.

    Asserted through a CHILD PROCESS with a clean environment, because this one has long since
    imported qtpy: by the time any test runs, the binding is settled and re-importing proves
    nothing. The child imports squidmip FIRST, which is the real entry point's order
    (`python -m squidmip._viewer` runs the package __init__ before anything else).
    """
    if importlib.util.find_spec("PyQt6") is None:
        pytest.skip("PyQt6 is not installed here; the pin is conditional on it by design")

    env = {k: v for k, v in os.environ.items() if k != "QT_API"}
    env["QT_QPA_PLATFORM"] = "offscreen"
    out = subprocess.run(
        [sys.executable, "-c", "import squidmip, qtpy; print(qtpy.API_NAME)"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "PyQt6", (
        f"expected the package to pin PyQt6, got {out.stdout.strip()!r}")


def test_an_explicit_binding_in_the_environment_still_wins():
    """The pin is a default, not a lock. Anyone whose machine misbehaves under Qt6 needs
    ``QT_API=pyqt5 squidmip-view`` to keep working, so the demo is never one bad GPU driver away
    from having no viewer at all."""
    if importlib.util.find_spec("PyQt5") is None:
        pytest.skip("PyQt5 is not installed here, so there is no second binding to fall back to")

    env = dict(os.environ, QT_API="pyqt5", QT_QPA_PLATFORM="offscreen")
    out = subprocess.run(
        [sys.executable, "-c", "import squidmip, qtpy; print(qtpy.API_NAME)"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "PyQt5"


def test_the_event_loop_is_entered_with_a_spelling_that_exists_on_qt6():
    """``QApplication.exec_()`` is PyQt5-only: PyQt6 dropped every trailing-underscore alias, so
    that call is an AttributeError there and the app dies immediately after painting its window.

    This is asserted by SOURCE, not by running it, and that is the point. ``main()`` returns the
    window without entering the loop whenever ``_squidmip_test`` is set, which every GUI test
    sets -- so this statement is the one line in the app that no test can ever execute. A source
    assertion is the only coverage available for it, and something had to cover it: the migration
    left this call behind and 1073 passing tests did not notice.
    """
    from squidmip import _viewer

    # Comment lines are stripped first: the fix is DOCUMENTED in a comment that names the broken
    # spelling, and a naive substring search finds the explanation and calls it the defect.
    src = inspect.getsource(_viewer.main)
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "exec_()" not in code, "exec_() is Qt5-only and breaks the Qt6 launch path"
    assert "app.exec()" in code


def test_fractional_display_scaling_is_passed_through_not_rounded():
    """Qt5's default rounding policy snaps 125/150/175% displays to 100% or 200%. On the 4K
    Windows demo machine that is the difference between a usable window and a three-inch one, and
    it is invisible on macOS because Retina is exactly 2x and rounds to itself. Source-asserted
    for the same reason as above: ``enable_hidpi`` must run BEFORE a QApplication exists, so a
    test that has already built one cannot observe its effect."""
    from squidmip import _viewer

    src = inspect.getsource(_viewer.enable_hidpi)
    assert "HighDpiScaleFactorRoundingPolicy" in src
    assert "PassThrough" in src
