"""A GUI panel built FROM an operator's ``params`` declaration. The fifth reader of the record.

THE GAP THIS CLOSES, MEASURED
-----------------------------
``_engine.Operator`` has declared ``params=(Param(name, default, blurb), ...)`` since Cellpose was
made a real operator, and four readers grew up around it: :meth:`Operator.bind` (applies them),
:func:`squidmip._cli._parse_param` (``--param min_area_px=80``), :mod:`squidmip._recipe` (writes
them into a chain expression) and :mod:`squidmip._compose` (namespaces them).

**The GUI was not one of them.** ``squidmip/_op_panels.py`` hand-writes a panel CLASS per operator,
so an operator that declared four parameters got ZERO widgets and ran silently at its defaults:
``spot`` and ``cellpose`` declare ``sigma_px``, ``min_area_px``, ``min_distance_px`` and
``split_touching`` between them and not one was reachable from any panel. That is not a cosmetic
gap — ``templates/operator/README.md`` is a PUBLIC contract telling a contributor to declare
``params``, and §2.4 of it had to document that the GUI ignores them.

This module is the fifth reader, and it is generic: it asks the registry for an operator's
``params`` and builds one widget per :class:`~squidmip._engine.Param`. Nothing here knows an
operator's name (``tests/test_operator_declaration.py`` fails the build if anything does), so
``spot``, ``cellpose`` and an operator in somebody else's package discovered through
``squidmip._plugins`` all get the same real controls from the same code.

THE MAPPING RULE, AND WHY IT IS THE DEFAULT'S TYPE
-------------------------------------------------
:class:`Param` deliberately declares **no type, no range and no widget hint** — its own docstring
says "the moment it grows a widget hint it has become the UI's schema and two places own the same
fact". So the widget is chosen from the one thing the declaration does carry: the TYPE OF THE
DEFAULT. One dict, :data:`WIDGET_KINDS`, is the whole rule::

    bool   -> a check box            split_touching=True
    int    -> an integer spin        min_area_px=30
    float  -> a decimal spin         sigma_px=2.0
    str    -> a text field           (nothing declares one today, see below)

``bool`` is looked up by EXACT type rather than ``isinstance``, because ``bool`` is a subclass of
``int`` in Python and ``isinstance(True, int)`` is ``True`` — an isinstance ladder in the wrong
order silently renders every check box as a 0/1 spinner.

A default whose type is not in that table (``None``, a tuple, a numpy array) is REFUSED BY NAME:
the panel says which parameter and which type, and offers the operator at its defaults rather than
inventing a widget. A guessed widget is how a value the user typed becomes a value the run did not
receive, which is the exact failure this module exists to end.

**Why ``str`` is a text field and not a combo.** A combo needs a set of legal values, and a
``Param`` declares none — there is no ``choices=`` and, per the docstring quoted above, there
deliberably will not be one. A combo built from a lone default would be a one-item combo, i.e. a
control that cannot be changed. Free text is the honest widget for "a string, and the declaration
does not say which". Nothing in this build declares a ``str`` parameter, so this branch ships
untested against a real operator and is pinned by a synthetic one in the tests.

CHAINS
------
``projector_params("bgsub + spot")`` returns the chain's parts' parameters NAMESPACED
``<step>.<param>`` (``squidmip._compose``), so a chain arrives here as a flat tuple whose names
carry the structure. :func:`group_params` splits on the first ``.`` and the panel draws one
HEADED GROUP per step, in chain order. Nothing else changes: the values go back through
``operator_kwargs`` under their namespaced names, which is exactly what
:meth:`Operator.bind` -> ``_compose._rebinder`` expects. A chain is not a special case here
because composition already made it not one.

WHAT THIS PANEL IS NOT
----------------------
It is the FALLBACK, not a replacement for :class:`~squidmip._op_panels.StitcherPanel` or
:class:`~squidmip._op_panels.DeconQCPanel`. Those two do things a parameter form cannot: the
stitcher converts units, drops registration-only knobs when registration is off and refuses a
plane-op with a sentence; the decon panel runs an iterative QC loop and publishes a picture into
pane 3. Generating a form over them would delete real behaviour to gain uniformity.
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

from squidmip._op_panels import _SUB, _Panel, _apply_qss, _head, _row, _wrapped

#: THE MAPPING RULE. The type of a ``Param``'s default -> the kind of widget that edits it.
#: Looked up by EXACT type (``type(default)``), never by ``isinstance``: ``bool`` is a subclass of
#: ``int``, so an isinstance ladder in the wrong order draws every check box as a 0/1 spinner.
WIDGET_KINDS = {
    bool: "check",
    int: "spin",
    float: "decimal",
    str: "text",
}

#: Spin ranges. A ``Param`` declares no range (by design — see the module docstring), so these are
#: the panel's own, and they are deliberately wide: a range that clipped a legal value would be a
#: silent edit of the number the user typed, which is the failure this module exists to end. The
#: floor is negative because nothing forbids a negative parameter.
_INT_RANGE = (-1_000_000_000, 1_000_000_000)
_FLOAT_RANGE = (-1.0e12, 1.0e12)
_FLOAT_DECIMALS = 4

#: How many wells a preview runs over unless the user moves the spinner. The same "first N wells"
#: shape ``_viewer._build_plane_op_tab`` uses, and for the same reason: an operator's parameters are
#: judged by eye on a few wells before a plate-wide run is worth its minutes.
DEFAULT_PREVIEW_WELLS = 4


# ---------------------------------------------------------------------------------------
# policy (no Qt) — the decisions, separately from the pixels
# ---------------------------------------------------------------------------------------

def widget_kind(default: Any) -> Optional[str]:
    """The widget kind for a default value, or ``None`` when nothing here can edit it.

    ``None`` is not a shrug: every caller turns it into a NAMED refusal that says which parameter
    and which type. See :func:`unsupported_params`.
    """
    return WIDGET_KINDS.get(type(default))


def param_step(name: str) -> tuple[Optional[str], str]:
    """Split a possibly namespaced parameter name: ``"spot.min_area_px"`` -> ``("spot", …)``.

    The inverse of what :mod:`squidmip._compose` does when it derives a chain's ``params``. Split on
    the FIRST ``.`` only, matching ``_compose._rebinder``'s ``partition(".")`` exactly — the two have
    to agree or a value would be routed to a step that never asked for it.
    """
    step, dot, parameter = str(name).partition(".")
    return (step, parameter) if dot else (None, str(name))


def group_params(params: Sequence) -> list[tuple[Optional[str], list]]:
    """``[(step_or_None, [Param, ...]), ...]`` — a chain's parameters grouped by their step.

    In first-appearance order, which for a chain is CHAIN ORDER: ``_compose`` builds the namespaced
    tuple by walking the parts in order, so the groups come out reading left to right the way the
    expression is written. A bare operator's parameters are one group keyed ``None``.
    """
    groups: dict[Optional[str], list] = {}
    for param in params:
        step, _ = param_step(param.name)
        groups.setdefault(step, []).append(param)
    return list(groups.items())


def unsupported_params(params: Sequence) -> list[tuple[str, str]]:
    """``[(param_name, type_name), ...]`` for every declared parameter this panel cannot draw."""
    return [(p.name, type(p.default).__name__)
            for p in params if widget_kind(p.default) is None]


def panel_refusal(key: str) -> Optional[str]:
    """Why a generic panel cannot be built for *key*, or ``None`` when it can.

    THE NAMED REFUSAL. ``_viewer._activate_operator`` used to be a silent no-op for any key its
    card table did not know — the click landed, nothing opened, and nothing said why. Silence is
    the bug; this is the sentence that replaces it.

    Three refusals, and each one is read off a declaration rather than off the name:

    * a REGION operator (``stitch``, ``coordinate``) — ``add_region_operator`` carries no
      ``params`` declaration AT ALL, so there is nothing here to read. That asymmetry between the
      two tables is exactly why ``_op_panels.STITCH_DEFAULTS`` still exists;
    * the key does not resolve to an operator, or this machine cannot run it because a declared
      ``requires=`` package is missing — both in the registry's own words, via
      :func:`squidmip.operator_available`, which resolves a bare NAME and a CHAIN alike;
    * it declares a parameter whose default has a type nothing here can edit.

    A CHAIN (``"bgsub + spot"``) is NOT refused: ``_compose`` derives its ``params`` from its
    parts, namespaced, and :func:`group_params` draws one group per step. Whether the HOST can
    launch that chain is a separate question the panel asks separately — see
    :meth:`GenericOperatorPanel._launch_refusal`.
    """
    from squidmip import available_region_operators, operator_available
    from squidmip._engine import projector_params

    if key in available_region_operators():
        return (f"'{key}' is a REGION operator, and that table declares no params= at all — there "
                "is nothing for a generic panel to read. Its controls are hand-written in "
                "squidmip._op_panels (StitcherPanel).")
    ok, why = operator_available(key)
    if not ok:
        return why
    try:
        params = projector_params(key)
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
    """One operator's declared parameters, as widgets, plus the run that carries them.

    Built from :func:`squidmip._engine.projector_params` and nothing else. The values leave through
    ``host.run_operator(key, ..., operator_kwargs=...)`` — the SAME argument
    :class:`~squidmip._op_panels.StitcherPanel` uses, which is what makes the value's journey to the
    pixels one already-tested path rather than a second one.
    """

    def __init__(self, host, key: str):
        from squidmip._engine import projector_consumes, projector_params
        from squidmip._operations import operator_label

        self.key = str(key)
        params = projector_params(self.key)
        self._reduces_z = bool(projector_consumes(self.key))
        super().__init__(
            host, operator_label(self.key),
            "Every control below is a parameter this operator DECLARES "
            "(squidmip._engine.Param), drawn from the type of its default. There is no panel "
            "written by hand for it, so this is the declaration itself, on screen.")

        self.widgets: dict = {}
        self._defaults = {p.name: p.default for p in params}
        if not params:
            self.v.addWidget(_wrapped(
                f"{self.key!r} declares no parameters: its behaviour is fixed at registration, so "
                "there is nothing to set. The run below is the operator at what it ships with.",
                _SUB))
        for step, group in group_params(params):
            if step is not None:
                # A CHAIN. `_compose` namespaces a chain's parameters `<step>.<param>`, so the
                # groups below read left to right in the order the expression is written.
                self.v.addWidget(_head(step.upper()))
            for param in group:
                self._add_param(param, step)

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

        # SAVE IS OFFERED OFF `consumes`, NEVER OFF THE NAME. A plane-op keeps z at full depth and
        # the OME-Zarr writer's per-field contract is the reason `_build_plane_op_tab` offers no
        # save either; a z-reducer collapses z to 1 and can be persisted. One declaration decides.
        #
        # NOT BUILT AT ALL when it does not apply, rather than built and left out of the layout:
        # `_viewer._raw_btn` is the precedent -- an orphan QPushButton nobody parented POPPED UP AS
        # A FLOATING WINDOW ("a 'return to raw view' window pops up. That I don't get.").
        self.save_btn = None
        if self._reduces_z:
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

        # BUILDABLE AND LAUNCHABLE ARE TWO QUESTIONS. A chain's parameters are perfectly readable
        # (that is the whole of `group_params`), while `run_operator` gates on
        # `runnable_operators()`, which lists TABLE KEYS and has never contained an expression. So
        # the form is shown and the buttons are greyed with the reason, rather than offering a
        # click that would be refused in a status line somewhere else.
        why = self._launch_refusal()
        if why:
            self.run_btn.setEnabled(False)
            if self.save_btn is not None:
                self.save_btn.setEnabled(False)
            self.say(why)

    def _launch_refusal(self) -> Optional[str]:
        """Why this window cannot RUN what this panel shows, or ``None``."""
        from squidmip._operations import runnable_operators

        if self.key in runnable_operators():
            return None
        return (f"the parameters of '{self.key}' are shown above, but this window runs operators "
                f"by registry key and '{self.key}' is not one — a chain expression is run from the "
                f"CLI (--projector '{self.key}' --param …). Runnable here: "
                f"{', '.join(runnable_operators())}.")

    # -- building ------------------------------------------------------------------------
    def _add_param(self, param, step: Optional[str]) -> None:
        """One :class:`Param` -> one labelled widget, with its ``blurb`` as the tooltip."""
        kind = widget_kind(param.default)
        _step, leaf = param_step(param.name)
        widget = _build_widget(kind, param.default)
        if widget is None:                 # unreachable through _activate_operator (panel_refusal
            return                         # catches it first), and silent here would be the bug
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

        Every declared parameter is sent, including the untouched ones. Sending only the changed
        ones would be smaller and would mean the same thing today, but it would make the console
        line (``_action_label``) and the recipe describe a DIFFERENT set of numbers from the run,
        and "the log says what ran" is the property this project spends most of its comments on.
        A namespaced name is passed through as declared — ``Operator.bind`` validates it against
        the same tuple this panel was built from.
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
        # regions=None is UNSCOPED, not "the whole plate": run_operator resolves it against the run
        # selector. Same spelling StitcherPanel uses, for the same reason.
        self._launch(regions=None, save=True)


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
    """A widget's value, in the TYPE the declaration's default had.

    The type matters as much as the number: ``spots_op`` builds a ``SpotParams`` dataclass out of
    these, and a ``min_area_px`` arriving as ``30.0`` where ``30`` was declared is the kind of
    thing that survives all the way to a comparison against an integer pixel count.
    """
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
