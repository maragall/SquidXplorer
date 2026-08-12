"""Which Qt binding this app actually runs on, and the one line only a real launch executes.

qtpy picks a binding by preference order starting at PyQt5, so on a machine with both PyQt5 and
PyQt6 installed the tree can look migrated to Qt6 while quietly still running Qt5 under a green
Qt6 suite. That matters beyond tidiness: Qt5 rounds fractional display scale factors, so a 150%
Windows display renders every window two-thirds size; Qt6 passes the fraction through by default.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import subprocess
import sys

import pytest


def test_importing_squidxplorer_pins_the_qt_binding_to_qt6():
    """Asserted through a CHILD PROCESS with a clean environment, because this process has long
    since imported qtpy and re-importing here proves nothing. The child imports squidxplorer
    first, matching the real entry point's order."""
    if importlib.util.find_spec("PyQt6") is None:
        pytest.skip("PyQt6 is not installed here; the pin is conditional on it by design")

    env = {k: v for k, v in os.environ.items() if k != "QT_API"}
    env["QT_QPA_PLATFORM"] = "offscreen"
    out = subprocess.run(
        [sys.executable, "-c", "import squidxplorer, qtpy; print(qtpy.API_NAME)"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "PyQt6", (
        f"expected the package to pin PyQt6, got {out.stdout.strip()!r}")


def test_an_explicit_binding_in_the_environment_still_wins():
    """The pin is a default, not a lock: `QT_API=pyqt5 squidxplorer-view` must still work."""
    if importlib.util.find_spec("PyQt5") is None:
        pytest.skip("PyQt5 is not installed here, so there is no second binding to fall back to")

    env = dict(os.environ, QT_API="pyqt5", QT_QPA_PLATFORM="offscreen")
    out = subprocess.run(
        [sys.executable, "-c", "import squidxplorer, qtpy; print(qtpy.API_NAME)"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "PyQt5"


def test_the_event_loop_is_entered_with_a_spelling_that_exists_on_qt6():
    """`QApplication.exec_()` is PyQt5-only — PyQt6 dropped every trailing-underscore alias.

    Asserted by SOURCE, not by running it: `main()` returns the window without entering the loop
    whenever `_squidxplorer_test` is set, which every GUI test sets, so this line can never
    execute under the suite.
    """
    from squidxplorer import _viewer

    src = inspect.getsource(_viewer.main)
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "exec_()" not in code, "exec_() is Qt5-only and breaks the Qt6 launch path"
    assert "app.exec()" in code


def test_fractional_display_scaling_is_passed_through_not_rounded():
    """Qt5's default rounding policy snaps 125/150/175% displays to 100% or 200%. Source-asserted
    because `enable_hidpi` must run BEFORE a QApplication exists, so a test that has already built
    one cannot observe its effect."""
    from squidxplorer import _viewer

    src = inspect.getsource(_viewer.enable_hidpi)
    assert "HighDpiScaleFactorRoundingPolicy" in src
    assert "PassThrough" in src
