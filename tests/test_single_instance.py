"""The GUI refuses a second window while one is open, via flock on a slot file."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

from squidxplorer._viewer import (
    GuiAlreadyOpen,
    acquire_gui_slot,
    gui_slot_limit,
    release_gui_slot,
)

# Where flock does not exist (Windows) the cross-process cap does not apply.
requires_flock = pytest.mark.skipif(
    importlib.util.find_spec("fcntl") is None,
    reason="no fcntl on this platform: the flock-based cap does not apply (the launch path is "
           "covered unconditionally by test_a_platform_without_flock_still_opens_the_gui)",
)


@pytest.fixture
def slots(tmp_path, monkeypatch):
    """Point the guard at a private lock dir so a real GUI on this machine is untouched."""
    monkeypatch.setenv("SQUIDXPLORER_GUI_LOCK_DIR", str(tmp_path))
    monkeypatch.delenv("SQUIDXPLORER_MAX_GUI", raising=False)
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
    monkeypatch.setenv("SQUIDXPLORER_MAX_GUI", "2")
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
    assert "SQUIDXPLORER_MAX_GUI" in msg, "the refusal must say how to raise the cap"
    assert "1" in msg, "the refusal must say what the cap is"


@requires_flock
def test_a_window_built_directly_still_takes_a_slot(slots, monkeypatch):
    """main() is not the only door: the slot is taken on show()."""
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    import squidxplorer._viewer as V

    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)
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
    """The cap must never be the thing that stops the app booting where fcntl is absent."""
    monkeypatch.setitem(sys.modules, "fcntl", None)   # `import fcntl` -> ModuleNotFoundError

    handle = acquire_gui_slot()
    assert handle is not None, "the GUI must still open where flock does not exist"
    assert handle.fd == -1, "no lock was taken, and the handle has to say so"

    second = acquire_gui_slot()
    release_gui_slot(second)

    release_gui_slot(handle)             # must tolerate the sentinel fd, not raise


@requires_flock
def test_a_crashed_gui_does_not_wedge_the_app_shut(slots):
    """A slot whose holder died is reusable with no cleanup."""
    handle = acquire_gui_slot()
    os.close(handle.fd)                  # the holder dies; no release_gui_slot()

    survivor = acquire_gui_slot()
    release_gui_slot(survivor)
