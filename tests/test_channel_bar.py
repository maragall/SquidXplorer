"""The plate carries NO contrast control surface. There is no channel bar to carry one.

RETARGETED 2026-08-06, from ``_viewer._ChannelBar`` to the real window.

WHAT CHANGED. ``_ChannelBar`` was a per-channel strip under the plate: a colour dot, a name, and a
contrast READOUT. Every test in this file built one directly and asserted it had no sliders, no
checkboxes and no buttons. The window stopped constructing it in ``8b0cbfc`` (2026-07-22) and never
constructed it again, so for two weeks this file proved a property of an object no user could see,
and the same interval is exactly when ``tools/gates.py``'s mutation self-test was patching
``_ChannelBar.__init__`` to inject a duplicate control -- a mutation that ran zero times and let
the gate print PASS against a codebase it had not mutated. The class was deleted on 2026-08-06.

WHAT DID NOT CHANGE. The requirement, which is Julio's, three rounds running:

    "Make sure there's no knowledge duplication in the GUI. I can still see the duplicated
    sliders."
    "there shouldn't be any controls for the plate view. It just reacts to toggles and contrast
    adjustments in napari."

Contrast has ONE owner -- each window's napari -- and the plate is a SINK: it follows through
``PlateWindow._on_detail_contrast`` -> ``PlateOverview.follow_channel_window``. Two widgets that
can move one value is what this file exists to fail the build for, so the assertions moved onto the
surface a user actually gets rather than being deleted with the widget.

Deliberately imports NO napari. Constructing a napari canvas in the same process loads a second Qt
binding on top of ``_viewer``'s, and the resulting clash ABORTS the interpreter rather than failing
a test.
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
    """A REAL ingested plate window. The point of this file is what a user is handed."""
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    qapp.processEvents()
    yield win
    win.close()
    qapp.processEvents()


def test_there_is_no_channel_bar_class_to_reintroduce():
    """The strip is gone as a TYPE, not merely unbuilt.

    An unbuilt class is what this whole file was testing, and what let a gate's mutation target
    something no window contains. If it comes back it has to come back with a call site.
    """
    assert not hasattr(V, "_ChannelBar"), (
        "_ChannelBar is back; if the plate is to have a channel strip again it needs a call site "
        "in PlateWindow, or this file is testing an object no user can see")


def test_the_plate_view_carries_no_contrast_control_at_all(plate):
    """MUTATION: mount a QSlider on ``PlateOverview`` wired to ``set_channel_window`` and this
    goes red. That is exactly the duplicate ``tools/gates.py`` re-injects for its self-test.

    Measured with that gate on a real window: origin/main carried 8 sliders + 4 auto buttons in
    the plate view; here it reports 0 and 0.
    """
    ov = plate._overview
    assert ov.findChildren(QSlider) == [], "the plate view grew a contrast slider again"
    assert ov.findChildren(QCheckBox) == [], "the plate view grew a channel checkbox again"
    assert ov.findChildren(QAbstractButton) == [], "the plate view grew a control again"


def test_the_only_slider_on_the_window_is_the_TIMEPOINT_slider(plate):
    """A slider is allowed on the window when it moves a different QUANTITY.

    The timepoint bar's slider selects WHICH timepoint is shown; it is not a second owner of any
    value napari owns. Named rather than merely tolerated, so a contrast slider cannot arrive here
    and pass as "the window has always had a slider".
    """
    from squidxplorer._time_point import TimePointBar

    for slider in plate.findChildren(QSlider):
        owner = slider.parent()
        while owner is not None and not isinstance(owner, TimePointBar):
            owner = owner.parent()
        assert owner is not None, (
            f"a slider on the plate window that is not the timepoint bar's: {slider!r}")


def test_the_plate_follows_contrast_and_cannot_set_it(plate):
    """The plate is a SINK. It has the entry point a window's napari pushes into, and no widget
    the user can drag to push into it themselves."""
    ov = plate._overview
    # the sink exists...
    assert callable(getattr(ov, "follow_channel_window", None))
    assert callable(getattr(ov, "set_channel_visible", None))
    # ...and the plate reaches it only from a window's gesture, never from its own control.
    assert callable(getattr(plate, "_on_detail_contrast", None))
    assert ov.findChildren(QAbstractButton) == []
