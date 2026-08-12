"""The plate carries NO contrast control surface. There is no channel bar to carry one.

`_ChannelBar` was a per-channel strip under the plate with a colour dot, a name and a contrast
readout. It has been deleted as a TYPE, not merely unbuilt — an unbuilt class is what let a gate's
mutation self-test target something no window contains.

Contrast has ONE owner — each window's napari — and the plate is a SINK: it follows through
`PlateWindow._on_detail_contrast` -> `PlateOverview.follow_channel_window`.

Deliberately imports NO napari: constructing a napari canvas in the same process loads a second Qt
binding on top of `_viewer`'s, which aborts the interpreter rather than failing a test.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded - Qt binding conflict", allow_module_level=True)

from qtpy.QtWidgets import (  # noqa: E402
    QAbstractButton, QApplication, QCheckBox, QSlider,
)

import squidxplorer._viewer as V  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)
    return app


@pytest.fixture
def plate(qapp, squid_dataset):
    """A real ingested plate window — what a user is actually handed."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    qapp.processEvents()
    yield win
    win.close()
    qapp.processEvents()


def test_there_is_no_channel_bar_class_to_reintroduce():
    assert not hasattr(V, "_ChannelBar"), (
        "_ChannelBar is back; if the plate is to have a channel strip again it needs a call site "
        "in PlateWindow, or this file is testing an object no user can see")


def test_the_plate_view_carries_no_contrast_control_at_all(plate):
    """MUTATION: mount a QSlider on `PlateOverview` wired to `set_channel_window` and this goes red."""
    ov = plate._overview
    assert ov.findChildren(QSlider) == [], "the plate view grew a contrast slider again"
    assert ov.findChildren(QCheckBox) == [], "the plate view grew a channel checkbox again"
    assert ov.findChildren(QAbstractButton) == [], "the plate view grew a control again"


def test_the_only_slider_on_the_window_is_the_TIMEPOINT_slider(plate):
    """A slider is allowed when it moves a different QUANTITY — the timepoint bar's slider
    selects WHICH timepoint is shown, not a second owner of a value napari owns."""
    from squidxplorer._time_point import TimePointBar

    for slider in plate.findChildren(QSlider):
        owner = slider.parent()
        while owner is not None and not isinstance(owner, TimePointBar):
            owner = owner.parent()
        assert owner is not None, (
            f"a slider on the plate window that is not the timepoint bar's: {slider!r}")


def test_the_plate_follows_contrast_and_cannot_set_it(plate):
    """The plate is a SINK: it has the entry point a window's napari pushes into, and no widget
    the user can drag to push into it themselves."""
    ov = plate._overview
    assert callable(getattr(ov, "follow_channel_window", None))
    assert callable(getattr(ov, "set_channel_visible", None))
    assert callable(getattr(plate, "_on_detail_contrast", None))
    assert ov.findChildren(QAbstractButton) == []
