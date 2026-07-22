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

import os

import pytest

from squidmip._viewer import (
    GuiAlreadyOpen,
    acquire_gui_slot,
    gui_slot_limit,
    release_gui_slot,
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
