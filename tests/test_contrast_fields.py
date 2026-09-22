"""The numeric contrast fields: visible numbers that follow the active layer and commit
through the app's contrast seam (Nick, ValidTX: "need numbers here, not just blank sliders")."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("napari")


@pytest.fixture
def scene():
    from napari.components import ViewerModel
    from qtpy.QtWidgets import QApplication

    from squidxplorer._contrast_fields import ContrastFields
    from squidxplorer._napari_view import MosaicLayers

    QApplication.instance() or QApplication([])
    ml = MosaicLayers(ViewerModel())
    rng = np.random.default_rng(0)
    for i, ch in enumerate(("488", "561")):
        ml.add_mosaic("raw", ch, rng.integers(0, 4000, (16, 16), dtype=np.uint16),
                      bbox_um=(0.0, 0.0, 16.0, 16.0))
    fields = ContrastFields(ml)
    fields.bind()
    return ml, fields


def test_the_fields_follow_the_active_layer_and_its_drags_live(scene):
    ml, fields = scene
    layer_488 = ml.find("raw", "488")
    layer_561 = ml.find("raw", "561")

    ml.model.layers.selection.active = layer_488
    layer_488.contrast_limits = (100.0, 900.0)          # what a slider drag writes
    assert (fields.lo_edit.text(), fields.hi_edit.text()) == ("100", "900")
    assert fields.isEnabled()

    ml.model.layers.selection.active = layer_561        # the controls page retargets
    layer_561.contrast_limits = (7.5, 250.0)
    assert (fields.lo_edit.text(), fields.hi_edit.text()) == ("7.5", "250")


def test_a_typed_value_commits_through_the_seam_and_never_narrows_the_range(scene):
    ml, fields = scene
    layer = ml.find("raw", "488")
    ml.model.layers.selection.active = layer
    heard = []
    ml.on_user_contrast(lambda ch, lo, hi: heard.append((ch, lo, hi)))

    fields.lo_edit.setText("50")
    fields.hi_edit.setText("70000")                     # beyond napari's uint16-seeded range
    fields.hi_edit.editingFinished.emit()               # Enter

    assert tuple(layer.contrast_limits) == (50.0, 70000.0)
    assert float(layer.contrast_limits_range[1]) >= 70000.0   # widened, not clipped
    # The plate's follow tap hears a typed value exactly like a slider drag.
    assert ("488", 50.0, 70000.0) in heard
