"""The plate's channel bar carries NO contrast control surface.

Deliberately imports NO napari. Constructing a napari canvas in the same process loads a second
Qt binding on top of _viewer's PyQt5, and the resulting clash ABORTS the interpreter rather than
failing a test.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
if "PySide6" in sys.modules or "PySide2" in sys.modules:
    pytest.skip("PySide already loaded - Qt binding conflict", allow_module_level=True)


class _StubOverview:
    """Just enough PlateOverview for _ChannelBar: it is the BAR under test, not the plate."""

    def __init__(self, n=3):
        self._contrast = type("C", (), {"dmax": 65535.0})()
        self._labels = [f"ch{i}" for i in range(n)]

    def channel_windows(self):
        return [(0.0, 1000.0)] * len(self._labels)

    def set_channel_visible(self, i, on):
        pass


_APP = None


def _bar():
    import numpy as np
    from PyQt5.QtWidgets import QApplication

    from squidmip._viewer import _ChannelBar

    # Keep a REFERENCE. `QApplication.instance() or QApplication([])` as an expression binds
    # nothing, so the app is garbage-collected immediately and the next QWidget aborts the
    # interpreter with no Python-level error.
    global _APP
    _APP = QApplication.instance() or QApplication([])
    ov = _StubOverview()
    colors = np.tile(np.array([[1.0, 1.0, 1.0]]), (len(ov._labels), 1))
    return _ChannelBar(ov._labels, colors, ov)


def test_the_plate_carries_no_contrast_slider_or_auto_button():
    """Julio, three rounds running: 'Make sure there\'s no knowledge duplication in the GUI. I
    can still see the duplicated sliders.' Contrast has ONE owner — the central array viewer —
    and two widgets that can move one value is what the IMA-268 gate fails the build for.

    Measured with that gate on a real window: origin/main carried 8 sliders + 4 auto buttons in
    the plate view; here it reports 'contrast: 0 sliders, 0 auto buttons in the plate view'.
    """
    from PyQt5.QtWidgets import QPushButton, QSlider

    bar = _bar()
    assert bar.findChildren(QSlider) == []
    assert [b for b in bar.findChildren(QPushButton) if b.text() == "auto"] == []


def test_the_bar_still_REPORTS_the_window_the_owner_resolved():
    """A readout is not a control. The plate must still show the window so the two panes can be
    seen to agree — it just must not be a second way to SET it."""
    bar = _bar()
    bar.set_window(1, 12.0, 3400.0)
    assert "12" in bar._rows[1][1].text()
    assert "3400" in bar._rows[1][1].text()


def test_the_channel_visibility_checkbox_survives():
    """Visibility masks a channel out of the PLATE composite — a different value from napari's
    per-layer visibility, which hides the layer in pane 2. Two controls, two values."""
    from PyQt5.QtWidgets import QCheckBox

    assert len(_bar().findChildren(QCheckBox)) == 3


def test_a_window_for_a_channel_the_plate_does_not_have_is_ignored():
    """The owner can broadcast a channel index the plate has no row for (RGB mode, a re-ingest).
    Indexing blindly would raise inside a signal handler, where it is easy to lose."""
    bar = _bar()
    bar.set_window(99, 0.0, 1.0)
