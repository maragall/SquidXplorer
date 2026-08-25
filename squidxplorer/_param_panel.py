"""A GUI panel generated from an operator's ``params`` declaration.

The widget is chosen from the TYPE of each Param's default; anything else is refused by
name. The fallback panel — the hand-written panels in ``_op_panels`` stay.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QSpinBox,
)

from squidxplorer._op_panels import _SUB, _Panel, _apply_qss, _row, _wrapped

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
                "generic form to show. Its controls are hand-written in squidxplorer._op_panels "
                "(StitcherPanel). Declare params= on it and this panel draws them.")
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
        from squidxplorer._engine import operator_params
        from squidxplorer._operations import operator_label

        self.key = str(key)
        params = operator_params(self.key)
        super().__init__(host, operator_label(self.key),
                         "Parameters from the operator's own declaration.")

        self.widgets: dict = {}
        self._defaults = {p.name: p.default for p in params}
        if not params:
            self.v.addWidget(_wrapped(
                f"{self.key!r} declares no parameters; it runs as it ships.", _SUB))
        for param in params:
            self._add_param(param)

        self.v.addWidget(self.status)
        self.v.addStretch(1)
        _apply_qss(self)

    # -- building ------------------------------------------------------------------------
    def _add_param(self, param) -> None:
        """One :class:`Param` -> one labelled widget, with its ``blurb`` as the tooltip."""
        kind = widget_kind(param.default)
        leaf = param.name
        widget = _build_widget(kind, param.default)
        if widget is None:                 # unreachable: panel_refusal catches it first
            return
        tip = param.blurb or f"{param.name}, declared by {self.key!r}. Default: {param.default!r}."
        widget.setToolTip(tip)
        self.widgets[param.name] = widget
        label = QLabel(f"{leaf}:")
        label.setToolTip(tip)
        # The blurb is the TOOLTIP only (text diet, Julio 2026-08-25: "There is so much text
        # in the operator UI. As if it was a book"); no wrapped paragraph under each widget.
        if isinstance(widget, QCheckBox):
            widget.setText(leaf)
            self.v.addWidget(widget)
        else:
            self.v.addLayout(_row(label, widget))

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
        widget = self.widgets.get(str(name))
        if widget is None:
            return (f"{self.key!r} declares no parameter {name!r}; "
                    f"it has {sorted(self.widgets)}.")
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
            "Write stitched_<folder> beside the acquisition: image files hardlinked, sidecars "
            "copied, coordinates.csv rewritten with the solved positions. The source "
            "acquisition is never written.")
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
