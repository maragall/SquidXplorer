"""A GUI panel generated from an operator's ``params`` declaration.

The widget is chosen from the TYPE of each Param's default; anything else is refused by
name. The fallback panel — the hand-written panels in ``_op_panels`` stay.
"""

from __future__ import annotations

import re

from typing import Any, Optional, Sequence

from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QSpinBox,
)

from squidxplorer._op_panels import (
    KEEP_EVERY_PLANE, _SUB, _Panel, _apply_qss, _row, _wrapped, z_operator_choice,
)


def _one_sentence(text) -> str:
    """The first sentence of *text*: a tooltip is at most one short sentence (Julio,
    2026-08-25: "too many description and tooltips and stuff")."""
    text = str(text or "").strip()
    if not text:
        return ""
    # The first sentence AS WRITTEN: the split keeps the sentence's own terminator and
    # never invents one (a tooltip is the declaration's words, verbatim).
    return re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0].strip()

#: Widget kind by the EXACT type of a Param's default — never isinstance: bool is a
#: subclass of int, so an isinstance ladder would draw every check box as a 0/1 spinner.
WIDGET_KINDS = {
    bool: "check",
    int: "spin",
    float: "decimal",
    str: "text",
}

# Deliberately wide spin ranges: a range that clipped a legal value would silently edit it.
_INT_RANGE = (-1_000_000_000, 1_000_000_000)
_FLOAT_RANGE = (-1.0e12, 1.0e12)
_FLOAT_DECIMALS = 4



# ---------------------------------------------------------------------------------------
# policy (no Qt) — the decisions, separately from the pixels
# ---------------------------------------------------------------------------------------

def widget_kind(default: Any) -> Optional[str]:
    """The widget kind for a default value, or ``None`` when nothing here can edit it."""
    return WIDGET_KINDS.get(type(default))


def unsupported_params(params: Sequence) -> list[tuple[str, str]]:
    """``[(param_name, type_name), ...]`` for every declared parameter this panel cannot draw."""
    return [(p.name, type(p.default).__name__)
            for p in params if widget_kind(p.default) is None]


def panel_refusal(key: str) -> Optional[str]:
    """Why a generic panel cannot be built for *key*, or ``None`` when it can."""
    from squidxplorer import is_region_operator, operator_available
    from squidxplorer._engine import operator_params

    if is_region_operator(key) and not operator_params(key):
        return (f"'{key}' fuses a whole well and declares no params=, so there is nothing for a "
                "generic form to show. Declare params= on it and this panel draws them.")
    ok, why = operator_available(key)
    if not ok:
        return why
    try:
        params = operator_params(key)
    except Exception as exc:                       # resolvable-but-unreadable: say it, don't crash
        return f"'{key}' declares no parameters this panel can read: {exc}"
    bad = unsupported_params(params)
    if bad:
        spelled = ", ".join(f"{name} (a {kind})" for name, kind in bad)
        return (f"'{key}' declares a parameter this panel cannot draw a widget for: {spelled}. "
                f"The widget is chosen from the DEFAULT's type, and it knows "
                f"{', '.join(sorted(t.__name__ for t in WIDGET_KINDS))}. Run it from the CLI with "
                f"--param, or give the parameter a default of a type this table knows.")
    return None


# ---------------------------------------------------------------------------------------
# the panel
# ---------------------------------------------------------------------------------------

class GenericOperatorPanel(_Panel):
    """One operator's declared parameters, as widgets. NO run buttons (one flow, Julio
    2026-08-25): the view's operators row launches every run — Preview and Run on plate —
    and reads this panel through ``kwargs()``."""

    def __init__(self, host, key: str):
        from squidxplorer._engine import operator_inner_param, operator_params
        from squidxplorer._operations import operator_label

        self.key = str(key)
        params = operator_params(self.key)
        try:
            self._inner_param = operator_inner_param(self.key)
        except Exception:                        # noqa: BLE001 - an unknown key draws plainly
            self._inner_param = None
        super().__init__(host, operator_label(self.key),
                         "Parameters from the operator's own declaration.")

        self.widgets: dict = {}
        self._defaults = {p.name: p.default for p in params}
        # No sentence for a parameter-less operator (Julio, 2026-08-25: the slot "just has
        # like BS AI text"): its ⚙ chip is disabled and nothing inserts.
        # THE ADVANCED SPLIT is DECLARATION-DRIVEN (Param.advanced), never a name match
        # (Julio, 2026-08-25: "all the other knobs ... should be hidden in a 'advanced'
        # slot"). Headline knobs stay visible; the rest collapse behind one toggle.
        headline = [p for p in params if not getattr(p, "advanced", False)]
        hidden = [p for p in params if getattr(p, "advanced", False)]
        for param in headline:
            self._add_param(param)
        self.adv_btn = None
        self._advanced = None
        if hidden:
            from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

            # Exactly this label (Julio, 2026-08-25: "'more...' button should be
            # 'advanced parameters'"), collapsed by default.
            self.adv_btn = QPushButton("advanced parameters")
            self.adv_btn.setCheckable(True)
            self.adv_btn.setToolTip("Knobs most runs leave at their defaults.")
            self.v.addWidget(self.adv_btn)
            self._advanced = QWidget()
            av = QVBoxLayout(self._advanced)
            av.setContentsMargins(0, 0, 0, 0)
            av.setSpacing(6)
            for param in hidden:
                self._add_param(param, into=av)
            self._advanced.setVisible(False)
            self.adv_btn.toggled.connect(self._advanced.setVisible)
            self.v.addWidget(self._advanced)

        self.v.addWidget(self.status)
        self.v.addStretch(1)
        _apply_qss(self)

    # -- building ------------------------------------------------------------------------
    def _add_param(self, param, into=None) -> None:
        """One :class:`Param` -> one labelled widget, with its ``blurb``'s FIRST sentence
        as the tooltip (verbosity strip, Julio 2026-08-25). *into* is the advanced
        section's layout; None means the headline."""
        target = into if into is not None else self.v
        leaf = param.name
        if self._inner_param is not None and param.name == self._inner_param:
            # The INNER operator is a choice, not free text: the plane operators plus
            # the keep-every-plane label (mapped to None in kwargs()).
            from qtpy.QtWidgets import QComboBox

            from squidxplorer import available_plane_operators

            widget = QComboBox()
            for name in available_plane_operators():
                widget.addItem(name)
            widget.addItem(KEEP_EVERY_PLANE)
            i = widget.findText(str(param.default))
            if i >= 0:
                widget.setCurrentIndex(i)
        else:
            kind = widget_kind(param.default)
            widget = _build_widget(kind, param.default)
        if widget is None:                 # unreachable: panel_refusal catches it first
            return
        tip = _one_sentence(param.blurb) or f"declared by {self.key!r}"
        widget.setToolTip(tip)
        self.widgets[param.name] = widget
        label = QLabel(f"{leaf}:")
        label.setToolTip(tip)
        # The blurb is the TOOLTIP only (text diet, Julio 2026-08-25: "There is so much text
        # in the operator UI. As if it was a book"); no wrapped paragraph under each widget.
        if isinstance(widget, QCheckBox):
            widget.setText(leaf)
            target.addWidget(widget)
        else:
            target.addLayout(_row(label, widget))

    # -- reading back --------------------------------------------------------------------
    def kwargs(self) -> dict:
        """The panel's widget values as ``operator_kwargs``, keyed by the DECLARED name.

        Every declared parameter is sent, untouched ones included, so the console line and
        the recipe describe the same numbers as the run.
        """
        return {name: _read_widget(widget) for name, widget in self.widgets.items()}

    def set_param(self, name: str, value) -> "Optional[str]":
        """Write ONE declared parameter into its widget, so an outside control (a view's
        inline iterations spin) edits the same number the run reads. Refusal by name."""
        from qtpy.QtWidgets import QComboBox

        widget = self.widgets.get(str(name))
        if widget is None:
            return (f"{self.key!r} declares no parameter {name!r}; "
                    f"it has {sorted(self.widgets)}.")
        if isinstance(widget, QComboBox):
            label = KEEP_EVERY_PLANE if value is None else str(value)
            i = widget.findText(label)
            if i < 0:
                return f"{self.key}: {label!r} is not one of the offered choices."
            widget.setCurrentIndex(i)
            return None
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))
        else:
            return f"no way to write a {type(widget).__name__}."
        return None

class DeconPanel(GenericOperatorPanel):
    """``decon``'s declared params plus the one control that is not a Param: NI.

    The immersion refractive index is the one PSF input no Squid file records (design
    principles: the user tweaks only what acquisition files cannot express). It is a
    SESSION setting (`_decon.set_session_ni`), not an operator kwarg, so it cannot be a
    Param; it survives the QC page's shelving as this one row. The iteration choice is
    made by hand now: draw an ROI, set iterations, Preview, repeat (the sweep is shelved,
    Julio 2026-08-25)."""

    def __init__(self, host):
        super().__init__(host, "decon")
        from qtpy.QtWidgets import QDoubleSpinBox, QLabel

        from squidxplorer._decon import session_ni, set_session_ni

        # NI as a DROPDOWN, value with the medium beside it (Julio, 2026-08-25, ruling w):
        # 1.000 (air) default, water, silicone, glycerol, oil, or custom.
        from qtpy.QtWidgets import QComboBox

        from squidxplorer._decon import IMMERSION_MEDIA

        self.ni_combo = QComboBox()
        for value, medium in IMMERSION_MEDIA:
            self.ni_combo.addItem(f"{value:.3f} ({medium})", float(value))
        self.ni_combo.addItem("custom", None)
        self.ni_spin = QDoubleSpinBox()
        self.ni_spin.setDecimals(3)
        self.ni_spin.setRange(1.0, 2.0)
        self.ni_spin.setSingleStep(0.01)
        self.ni_spin.setVisible(False)
        current = float(session_ni() or 1.0)
        k = next((i for i in range(self.ni_combo.count())
                  if self.ni_combo.itemData(i) == current), None)
        if k is None:
            k = self.ni_combo.count() - 1
            self.ni_spin.setValue(current)
        self.ni_combo.setCurrentIndex(k)
        self.ni_spin.setVisible(self.ni_combo.itemData(k) is None)

        def _ni() -> float:
            v = self.ni_combo.currentData()
            return float(self.ni_spin.value()) if v is None else float(v)

        def _on_ni(*_):
            self.ni_spin.setVisible(self.ni_combo.currentData() is None)
            set_session_ni(_ni())

        self.ni_combo.currentIndexChanged.connect(_on_ni)
        self.ni_spin.valueChanged.connect(_on_ni)
        set_session_ni(_ni())
        lab = QLabel("ni:")
        lab.setToolTip("Immersion refractive index; the one PSF input no Squid file records.")
        self.ni_combo.setToolTip(lab.toolTip())
        at = self.v.indexOf(self.status)
        self.v.insertLayout(at, _row(lab, self.ni_combo))
        self.v.insertWidget(at + 1, self.ni_spin)
        _apply_qss(self)


class RegisterPanel(GenericOperatorPanel):
    """``register``'s declared params plus the one control that is not a Param: the copy switch.

    The copy switch cannot be a Param (it cannot change the preview's pixels, so the declaration
    probe would rightly fail it); it rides the record's ``accepts`` passthrough instead.
    """

    def __init__(self, host):
        super().__init__(host, "register")

        self.copy_check = QCheckBox("write registered copy (stitched_<folder>)")
        # CHECKED by default: the copy IS this operator's purpose and costs hardlinks. Unchecked
        # was how "Registering the wells doesn't do anything" happened (Julio, 2026-08-19) — a
        # green run whose only product was a preview layer.
        self.copy_check.setChecked(True)
        self.copy_check.setToolTip(
            "Write stitched_<folder> beside the acquisition with the solved positions; "
            "the source is never written.")
        at = self.v.indexOf(self.status)
        self.v.insertWidget(at, self.copy_check)
        _apply_qss(self)                # the base styled its own widgets before these existed

    def kwargs(self) -> dict:
        kw = super().kwargs()
        if self.copy_check.isChecked():
            kw["copy"] = True
        return kw


def _build_widget(kind: Optional[str], default: Any):
    """The one place a widget kind becomes a widget. Returns ``None`` for an unknown kind."""
    if kind == "check":
        widget = QCheckBox()
        widget.setChecked(bool(default))
        return widget
    if kind == "spin":
        widget = QSpinBox()
        widget.setRange(*_INT_RANGE)
        widget.setValue(int(default))
        return widget
    if kind == "decimal":
        widget = QDoubleSpinBox()
        widget.setDecimals(_FLOAT_DECIMALS)
        widget.setRange(*_FLOAT_RANGE)
        widget.setSingleStep(0.1)
        widget.setValue(float(default))
        return widget
    if kind == "text":
        widget = QLineEdit()
        widget.setText(str(default))
        return widget
    return None


def _read_widget(widget) -> Any:
    """A widget's value, in the TYPE the declaration's default had."""
    from qtpy.QtWidgets import QComboBox

    if isinstance(widget, QComboBox):
        return z_operator_choice(widget.currentText())   # keep-every-plane spells None
    if isinstance(widget, QCheckBox):
        return bool(widget.isChecked())
    if isinstance(widget, QSpinBox):
        return int(widget.value())
    if isinstance(widget, QDoubleSpinBox):
        return float(widget.value())
    if isinstance(widget, QLineEdit):
        return str(widget.text())
    raise ValueError(f"no way to read {type(widget).__name__} back - this panel builds only the "
                     f"widgets in WIDGET_KINDS")
