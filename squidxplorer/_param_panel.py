"""A GUI panel generated from an operator's ``params`` declaration.

The widget is chosen from the TYPE of each Param's default; anything else is refused by
name. The fallback panel — the hand-written panels in ``_op_panels`` stay.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
)

from squidxplorer._op_panels import _SUB, _Panel, _apply_qss, _head, _row, _wrapped

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

#: How many wells a preview runs over unless the user moves the spinner.
DEFAULT_PREVIEW_WELLS = 4


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
    """One operator's declared parameters, as widgets, plus the run that carries them."""

    def __init__(self, host, key: str):
        from squidxplorer._engine import operator_consumes, operator_params
        from squidxplorer._operations import operator_label

        self.key = str(key)
        params = operator_params(self.key)
        # Saveable is read off `consumes`, never off the name: a plane-op keeps z at full depth.
        self._can_save = bool(operator_consumes(self.key))
        super().__init__(
            host, operator_label(self.key),
            "Every control below is a parameter this operator DECLARES "
            "(squidxplorer._engine.Param), drawn from the type of its default. There is no panel "
            "written by hand for it, so this is the declaration itself, on screen.")

        self.widgets: dict = {}
        self._defaults = {p.name: p.default for p in params}
        if not params:
            self.v.addWidget(_wrapped(
                f"{self.key!r} declares no parameters: its behaviour is fixed at registration, so "
                "there is nothing to set. The run below is the operator at what it ships with.",
                _SUB))
        for param in params:
            self._add_param(param)

        # -- run ---------------------------------------------------------------------------
        self.v.addWidget(_head("PREVIEW"))
        n_wells = max(1, len(getattr(host, "_order", []) or []))
        self.wells_spin = QSpinBox()
        self.wells_spin.setRange(1, n_wells)
        self.wells_spin.setValue(min(DEFAULT_PREVIEW_WELLS, n_wells))
        self.wells_spin.setToolTip(
            "How many wells the preview runs over. Parameters are judged by eye on a few wells "
            "before a plate-wide run is worth its minutes.")
        self.v.addLayout(_row(QLabel("First"), self.wells_spin, QLabel("wells")))

        self.run_btn = QPushButton("Preview with these parameters")
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.setToolTip(
            "Run the operator over the wells above with exactly the values set here, and stream "
            "the result into the plate. Nothing is written to disk.")
        self.run_btn.clicked.connect(self._preview)
        self.v.addWidget(self.run_btn)

        # Not built at all when it does not apply — an unparented button pops up as a floating window.
        self.save_btn = None
        if self._can_save:
            self.save_btn = QPushButton("Run on the whole plate and save…")
            self.save_btn.setCursor(Qt.PointingHandCursor)
            self.save_btn.setToolTip(
                "Run over the whole plate with these parameters and write a navigable OME-Zarr. "
                "You are asked where.")
            self.save_btn.clicked.connect(self._save)
            self.v.addWidget(self.save_btn)
        else:
            self.v.addWidget(_wrapped(
                f"{self.key!r} keeps the z-stack at full depth (it consumes no axis), so there is "
                "no plate to save — preview only. The raw acquisition is never modified.", _SUB))

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumHeight(6)
        self.progress.setVisible(False)
        self.v.addWidget(self.progress)
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
        if isinstance(widget, QCheckBox):
            widget.setText(leaf)
            self.v.addWidget(widget)
        else:
            self.v.addLayout(_row(label, widget))
        if param.blurb:
            self.v.addWidget(_wrapped(param.blurb, _SUB))

    # -- reading back --------------------------------------------------------------------
    def kwargs(self) -> dict:
        """The panel's widget values as ``operator_kwargs``, keyed by the DECLARED name.

        Every declared parameter is sent, untouched ones included, so the console line and
        the recipe describe the same numbers as the run.
        """
        return {name: _read_widget(widget) for name, widget in self.widgets.items()}

    def _launch(self, *, regions, save: bool) -> None:
        try:
            kwargs = self.kwargs()
        except ValueError as exc:               # a refused value -> say it, run nothing
            self.say(str(exc))
            return
        changed = {k: v for k, v in kwargs.items() if v != self._defaults.get(k)}
        self.say(f"{self.key}: running with "
                 + (", ".join(f"{k}={v}" for k, v in sorted(changed.items())) if changed
                    else "its declared defaults"))
        self.host.run_operator(self.key, regions=regions, save=save, operator_kwargs=kwargs)

    def _preview(self) -> None:
        order = list(getattr(self.host, "_order", []) or [])
        if not order:
            self.say("no acquisition is open — there are no wells to preview on.")
            return
        self._launch(regions=order[:self.wells_spin.value()], save=False)

    def _save(self) -> None:
        # regions=None is UNSCOPED: run_operator resolves it against the run selector.
        self._launch(regions=None, save=True)


class RegisterPanel(GenericOperatorPanel):
    """``register``'s declared params plus the one control that is not a Param: the copy switch.

    The copy switch cannot be a Param (it cannot change the preview's pixels, so the declaration
    probe would rightly fail it); it rides the record's ``accepts`` passthrough instead. The
    OME-Zarr save is hidden because this operator's disk artifact is the registered copy.
    """

    def __init__(self, host):
        super().__init__(host, "register")
        if self.save_btn is not None:
            self.save_btn.setVisible(False)

        self.copy_check = QCheckBox("write registered copy (stitched_<folder>)")
        self.copy_check.setToolTip(
            "Write stitched_<folder> beside the acquisition: image files hardlinked (a second "
            "name for the same bytes — no duplication; copied in full where the filesystem "
            "refuses links), sidecars copied, and coordinates.csv rewritten with the solved "
            "positions. The source acquisition is never written.")
        self.run_all_btn = QPushButton("Register the selected wells")
        self.run_all_btn.setCursor(Qt.PointingHandCursor)
        self.run_all_btn.setToolTip(
            "Solve every selected well (the run selector's scope) with the parameters above. "
            "With the copy box checked, each well's rows land in stitched_<folder> as it solves.")
        self.run_all_btn.clicked.connect(self._run_selected)
        at = self.v.indexOf(self.progress)
        self.v.insertWidget(at, self.copy_check)
        self.v.insertWidget(at + 1, self.run_all_btn)

    def kwargs(self) -> dict:
        kw = super().kwargs()
        if self.copy_check.isChecked():
            kw["copy"] = True
        return kw

    def _run_selected(self) -> None:
        # regions=None is UNSCOPED: run_operator resolves it against the run selector.
        self._launch(regions=None, save=False)


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
    raise ValueError(f"no way to read {type(widget).__name__} back — this panel builds only the "
                     f"widgets in WIDGET_KINDS")
