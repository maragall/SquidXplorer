"""The plate carries NO contrast control surface."""

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


def test_the_plate_follows_contrast_and_carries_no_control_to_set_it(plate):
    """The plate is a SINK: a window's napari pushes into it, and no widget lets the user push into it themselves."""
    ov = plate._overview
    assert callable(getattr(ov, "follow_channel_window", None))
    assert callable(getattr(ov, "set_channel_visible", None))
    assert ov.findChildren(QSlider) == [], "the plate view grew a contrast slider again"
    assert ov.findChildren(QCheckBox) == [], "the plate view grew a channel checkbox again"
    assert ov.findChildren(QAbstractButton) == [], "the plate view grew a control again"


def test_the_only_slider_on_the_window_is_the_TIMEPOINT_slider(plate):
    """A slider is allowed when it moves a different QUANTITY — the timepoint bar's slider selects WHICH timepoint is shown, not a second owner of a value napari owns."""
    from squidxplorer._time_point import TimePointBar

    for slider in plate.findChildren(QSlider):
        owner = slider.parent()
        while owner is not None and not isinstance(owner, TimePointBar):
            owner = owner.parent()
        assert owner is not None, (
            f"a slider on the plate window that is not the timepoint bar's: {slider!r}")
