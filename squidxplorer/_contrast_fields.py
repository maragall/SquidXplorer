"""Numeric lo/hi contrast fields for the ACTIVE layer (Nick, ValidTX: "need numbers here,
not just blank sliders").

Measured on napari 0.6.6: the always-visible contrast control is a bare range slider
(``_QDoubleRangeSlider``); its numbers exist only inside the right-click
``QContrastLimitsPopup``, a hidden affordance. This row makes them always visible and
editable, one row under napari's own layer-controls page.

The fields FOLLOW the active layer live (slider drags included, via the layer's own
``contrast_limits`` event) and a typed value commits on Enter THROUGH the app's seam:
``MosaicLayers.widen_contrast_range`` then ``MosaicLayers.set_contrast``, so the identity
mirror carries it to every holder and the never-narrows range rule holds. A foreign layer
(no app identity) commits to the layer directly. Not an intensity layer: fields blank,
disabled, never hidden (nothing collapses).
"""

from __future__ import annotations

from typing import Any, Optional

from qtpy.QtGui import QDoubleValidator, QFont
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget

_FIELD_PX = 58          # each edit's width: 5 digits of uint16 plus a sign fit
_FONT_PX = 11           # the chip font size, matched by value (no import cycle)


def _fmt(value: float) -> str:
    """A number the user can read back: integers plain, fractions to two decimals."""
    s = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return s or "0"


class ContrastFields(QWidget):
    """Two editable numeric fields mirroring the active layer's contrast window."""

    def __init__(self, mosaic: Any, parent=None) -> None:
        super().__init__(parent)
        self._mosaic = mosaic
        self._layer: Optional[Any] = None
        font = QFont()
        font.setPixelSize(_FONT_PX)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        label = QLabel("contrast")
        label.setFont(font)
        lay.addWidget(label, 0)
        self.lo_edit = self._field(font, "Lower contrast limit. Type a value and press "
                                         "Enter to apply it to this channel.")
        self.hi_edit = self._field(font, "Upper contrast limit. Type a value and press "
                                         "Enter to apply it to this channel.")
        lay.addWidget(self.lo_edit, 1)
        lay.addWidget(self.hi_edit, 1)
        self.setEnabled(False)

    def _field(self, font, tip: str) -> QLineEdit:
        edit = QLineEdit(self)
        edit.setFont(font)
        edit.setMinimumWidth(_FIELD_PX)
        edit.setToolTip(tip)
        validator = QDoubleValidator(self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        edit.setValidator(validator)
        edit.editingFinished.connect(self._commit)
        return edit

    # -- wiring ------------------------------------------------------------
    def bind(self) -> None:
        """Start following the viewer's active layer. Separate from __init__ so a scene
        that never materialises (pane build failure) costs no connections."""
        model = getattr(self._mosaic, "model", None)
        if model is None:
            return
        model.layers.selection.events.active.connect(self._retarget)
        self._retarget()

    def _retarget(self, event=None) -> None:
        """Point the fields at the layer napari's controls page is showing."""
        old = self._layer
        if old is not None:
            try:
                old.events.contrast_limits.disconnect(self._refresh)
            except Exception:                    # noqa: BLE001 - a dead layer disconnects itself
                pass
        model = getattr(self._mosaic, "model", None)
        layer = getattr(getattr(model, "layers", None), "selection", None)
        layer = getattr(layer, "active", None)
        if layer is None or getattr(layer, "contrast_limits", None) is None:
            self._layer = None
            self.lo_edit.clear()
            self.hi_edit.clear()
            self.setEnabled(False)
            return
        self._layer = layer
        layer.events.contrast_limits.connect(self._refresh)
        self.setEnabled(True)
        self._refresh()

    def _refresh(self, event=None) -> None:
        """Mirror the layer's window into the fields; a field being typed in is left alone."""
        layer = self._layer
        if layer is None:
            return
        try:
            lo, hi = (float(v) for v in layer.contrast_limits)
        except Exception:                        # noqa: BLE001 - layer torn down mid-event
            return
        if not self.lo_edit.hasFocus():
            self.lo_edit.setText(_fmt(lo))
        if not self.hi_edit.hasFocus():
            self.hi_edit.setText(_fmt(hi))

    # -- the write ---------------------------------------------------------
    def _commit(self) -> None:
        """Apply the typed pair through the app's contrast seam."""
        layer = self._layer
        if layer is None:
            return
        try:
            lo = float(self.lo_edit.text())
            hi = float(self.hi_edit.text())
        except ValueError:
            self._refresh()                      # unreadable: show the real values again
            return
        if hi < lo:
            lo, hi = hi, lo
        from squidxplorer._napari_view import key_of

        key = key_of(layer)
        mosaic = self._mosaic
        if key is not None and mosaic is not None:
            # Widen FIRST (programmatic, echo-suppressed), then write: a typed value beyond
            # napari's range would otherwise be clipped away before it lands.
            mosaic.widen_contrast_range(key.channel, lo, hi)
            try:
                mosaic.set_contrast(key.channel, lo, hi)
                return
            except KeyError:
                pass                             # identity gone mid-edit: fall through
        try:
            from squidxplorer._napari_view import MosaicLayers

            MosaicLayers._widen_range(layer, lo, hi)
            layer.contrast_limits = (lo, hi)
        except Exception:                        # noqa: BLE001 - a foreign layer may refuse
            self._refresh()
