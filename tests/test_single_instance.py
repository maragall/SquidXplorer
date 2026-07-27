"""The GUI refuses to open a second window while one is already open (cross-process).

Julio: "there's a bunch of windows open ... you open another instance of the GUI without
closing previous". Every agent proof run left its window on his screen, and nothing in the
app stopped the next one. A launcher-script fix is not a cap: the cap has to live in the
GUI, so it holds no matter who starts the process.

The primitive is ``flock`` on a slot file, NOT a pidfile. A pidfile has to be cleaned up,
and a killed or crashed GUI never cleans up -- which is exactly the state these runs end
in, so a pidfile would have wedged the app shut. An flock is released by the kernel when
the process dies, however it dies, so the cap is self-healing.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

from squidmip._viewer import (
    GuiAlreadyOpen,
    acquire_gui_slot,
    gui_slot_limit,
    release_gui_slot,
)

#: The cap IS flock. Where flock does not exist (Windows) there is no cross-process cap to test,
#: so the tests below that assert a refusal have nothing to assert -- they are not "broken on
#: Windows", they are inapplicable. What Windows must still guarantee is that the app opens at
#: all, and ``test_a_platform_without_flock_still_opens_the_gui`` pins that on EVERY platform,
#: unskipped. Keep it that way: skipping the launch test is what would reopen the crash.
requires_flock = pytest.mark.skipif(
    importlib.util.find_spec("fcntl") is None,
    reason="no fcntl on this platform: the flock-based cap does not apply (the launch path is "
           "covered unconditionally by test_a_platform_without_flock_still_opens_the_gui)",
)


@pytest.fixture
def slots(tmp_path, monkeypatch):
    """Point the guard at a private lock dir so a real GUI on this machine is untouched."""
    monkeypatch.setenv("SQUIDMIP_GUI_LOCK_DIR", str(tmp_path))
    monkeypatch.delenv("SQUIDMIP_MAX_GUI", raising=False)
    return tmp_path


def test_the_first_gui_gets_a_slot(slots):
    handle = acquire_gui_slot()
    assert handle is not None
    release_gui_slot(handle)


@requires_flock
def test_a_second_gui_is_refused_while_the_first_holds_its_slot(slots):
    first = acquire_gui_slot()
    try:
        with pytest.raises(GuiAlreadyOpen):
            acquire_gui_slot()
    finally:
        release_gui_slot(first)


def test_releasing_frees_the_slot_for_the_next_window(slots):
    first = acquire_gui_slot()
    release_gui_slot(first)

    second = acquire_gui_slot()          # the previous window closed: this must be allowed
    release_gui_slot(second)


@requires_flock
def test_the_cap_is_configurable(slots, monkeypatch):
    monkeypatch.setenv("SQUIDMIP_MAX_GUI", "2")
    assert gui_slot_limit() == 2

    a = acquire_gui_slot()
    b = acquire_gui_slot()               # two are allowed now
    try:
        with pytest.raises(GuiAlreadyOpen):
            acquire_gui_slot()           # the third is not
    finally:
        release_gui_slot(a)
        release_gui_slot(b)


@requires_flock
def test_the_refusal_names_the_limit_and_how_to_override(slots):
    first = acquire_gui_slot()
    try:
        with pytest.raises(GuiAlreadyOpen) as exc:
            acquire_gui_slot()
    finally:
        release_gui_slot(first)

    msg = str(exc.value)
    assert "SQUIDMIP_MAX_GUI" in msg, "the refusal must say how to raise the cap"
    assert "1" in msg, "the refusal must say what the cap is"


@requires_flock
def test_a_window_built_directly_still_takes_a_slot(slots, monkeypatch):
    """main() is not the only door. Proof scripts and debug launchers construct a PlateWindow
    themselves, which is exactly how the screen filled up, so the slot is taken on show()."""
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    import squidmip._viewer as V

    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidmip_test", True)
    monkeypatch.setattr(V, "_gui_cap_applies", lambda: True)   # offscreen is exempt; force the cap

    first = V.PlateWindow(None)
    first.show()
    try:
        assert getattr(first, "_gui_slot", None) is not None, "a visible window took no slot"
        with pytest.raises(GuiAlreadyOpen):
            acquire_gui_slot()                                 # the cap is really held
    finally:
        first.close()

    freed = acquire_gui_slot()          # closing the window gave the slot back
    release_gui_slot(freed)


def test_a_platform_without_flock_still_opens_the_gui(slots, monkeypatch):
    """Windows has no ``fcntl``, and the cap must never be the thing that stops the app booting.

    Both doors into ``acquire_gui_slot`` -- ``main()`` and ``PlateWindow.showEvent`` -- catch
    only :class:`GuiAlreadyOpen`, so an ImportError out of here killed the process before any
    window appeared. ``scripts/Setup-Windows.ps1`` installs a Desktop shortcut pointing straight
    down that path, so on Windows the shipped launcher was the crash.

    Simulated rather than skipped-on-win32 on purpose. Skipping would make the guard unprovable
    on the Linux and macOS runs, and the obvious way to quiet a red Windows job -- marking this
    file skipif(win32) -- would then reopen the hole with no red test anywhere.
    """
    # setitem, not a bare assignment: it restores the real module on Unix and removes the key
    # again on Windows, where there was none to begin with.
    monkeypatch.setitem(sys.modules, "fcntl", None)   # `import fcntl` -> ModuleNotFoundError

    handle = acquire_gui_slot()
    assert handle is not None, "the GUI must still open where flock does not exist"
    assert handle.fd == -1, "no lock was taken, and the handle has to say so"

    # The documented trade-off: no flock means no cross-process cap. Losing the cap is a
    # nicety; losing the app is not. Pinned so the degradation stays deliberate.
    second = acquire_gui_slot()
    release_gui_slot(second)

    release_gui_slot(handle)             # must tolerate the sentinel fd, not raise


@requires_flock
def test_a_crashed_gui_does_not_wedge_the_app_shut(slots):
    """The self-healing property. A slot whose holder died is reusable with no cleanup.

    Simulated by closing the fd without the tidy release path -- which is what the kernel
    does for a killed process. A pidfile design fails this test, and that failure mode
    (app permanently refusing to start after a crash) is worse than the bug being fixed.
    """
    handle = acquire_gui_slot()
    os.close(handle.fd)                  # the holder dies; no release_gui_slot()

    survivor = acquire_gui_slot()
    release_gui_slot(survivor)
