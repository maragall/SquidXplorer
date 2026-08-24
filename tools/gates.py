#!/usr/bin/env python3
"""Structural gates over the REAL widget tree of a shown window.

    QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python tools/gates.py

GATE 2 checks that no more than one widget can write any given piece of state (no duplicated
controllers). GATE 3 checks that every reachable control produces an observable outcome when
actuated (no dead controls). Both gates open a real, shown window and drive the real widget tree
rather than reading source or calling handlers directly; both are mutation-checked via --self-test.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SQUIDXPLORER_FIXTURE_PLATE overrides the default fixture path; regenerate it with
# tools/make_5d_fixture.py if missing (see the SKIP message in main()).
PLATE = os.environ.get("SQUIDXPLORER_FIXTURE_PLATE") or \
    "/Users/julioamaragall/Downloads/sim_2x2_36fov_96wp"

_APP = None
_MISSING = object()      # sentinel: attribute was inherited, not the class's own (see monkey())


def _app():
    """The QApplication, on the binding squidxplorer actually ships (import it first so QT_API is set)."""
    global _APP
    import squidxplorer  # noqa: F401  -- sets QT_API before qtpy resolves a binding
    from qtpy.QtWidgets import QApplication
    _APP = QApplication.instance() or QApplication([])
    return _APP


# A probe returns a hashable {slot key: value} snapshot of ONE piece of application state, read
# from the model, never from a widget. The KEY, not the concern, is the unit of duplication: four
# per-channel visibility checkboxes are four controls over four different truths.

def _probe_contrast(w):
    ov = w._overview
    out = {}
    for i, (lo, hi) in enumerate(ov.channel_windows() or []):
        out[f"contrast[{i}]"] = (round(float(lo), 6), round(float(hi), 6))
    return out


def _probe_visibility(w):
    ov = w._overview
    if ov._mask is None:
        return {}
    return {f"visibility[{i}]": bool(b) for i, b in enumerate(ov._mask)}


def _probe_active_layer(w):
    return {"active layer": w._overview._active}


def _probe_selection(w):
    return {"plate selection": (repr(w._overview._sel), tuple(w._selected_regions))}


def _probe_current_well(w):
    return {"current well": w._current_well}


def _probe_zoom(w):
    ov = w._overview
    return {"zoom / viewport": (round(ov._cd, 4), round(ov._ox, 4), round(ov._oy, 4))}


def _probe_colormap(w):
    cols = getattr(w._overview, "_colors", None)
    if cols is None:
        return {}
    return {f"channel colour / LUT[{i}]": tuple(row) for i, row in enumerate(cols.tolist())}


PROBES = (_probe_contrast, _probe_visibility, _probe_active_layer,
          _probe_selection, _probe_current_well, _probe_zoom, _probe_colormap)


def _concern_of(key: str) -> str:
    """'visibility[2]' -> 'visibility'. The table is written per concern, the check is per slot."""
    return key.split("[", 1)[0].strip()


#: Probes that raised while snapshotting: {probe name: "TypeName: message"}.
BROKEN_PROBES: dict[str, str] = {}


def _snapshot(w):
    """Every probe's reading, and a record of any probe that could not take one (never swallowed)."""
    out = {}
    for p in PROBES:
        try:
            out.update(p(w))
        except Exception as exc:                       # noqa: BLE001 - recorded, then reported
            BROKEN_PROBES[p.__name__] = f"{type(exc).__name__}: {exc}"
    return out


#: Probes whose reading did not move when the model underneath them was moved by hand.
FROZEN_PROBES: dict[str, str] = {}

#: Controls whose actuation raised during GATE 2's sweep.
RAISING_CONTROLS: dict[str, str] = {}


def _drive_the_model(w):
    """One model-level mutation per probe: {probe name: (what, apply)}.

    Writes the MODEL, never a widget — this asks "can this reading move at all", not "can a user
    move it" (GATE 2's question). Every driver restores what it moved; ``apply()`` returns an undo
    callable, or None when the model has nothing here to move.
    """
    ov = w._overview

    def _flip_visibility():
        if ov._mask is None or not len(ov._mask):
            return None
        was = bool(ov._mask[0])
        ov.set_channel_visible(0, not was)
        return lambda: ov.set_channel_visible(0, was)

    def _move_contrast():
        windows = list(ov.channel_windows() or [])
        if not windows:
            return None
        lo, hi = windows[0]
        ov.set_channel_window(0, 12345.0, 23456.0)
        return lambda: ov.set_channel_window(0, float(lo), float(hi))

    def _move_colour():
        if ov._colors is None or not len(ov._colors):
            return None
        was = tuple(ov._colors[0].tolist())
        ov.set_channel_color(0, (0.125, 0.25, 0.375))
        return lambda: ov.set_channel_color(0, was)

    def _move_active():
        was = ov._active
        ov.set_active_layer("a layer no operator is called")
        return lambda: ov.set_active_layer(was)

    def _move_selection():
        was = set(ov._selection)
        if was:
            ov.clear_selection()
        else:
            ov.select_all()

        def undo():
            ov._selection = set(was)
            ov.selectionChanged.emit(ov.selected_wells())
            ov.update()
        return undo

    def _move_current_well():
        # `_current_well` is a property over `self._cursor`, not an overview attribute.
        regions = list(getattr(w, "_order", None) or (w._meta or {}).get("regions") or [])
        if not regions:
            return None
        was = w._current_well
        w._current_well = regions[-1] if was == regions[0] else regions[0]
        return lambda: setattr(w, "_current_well", was)

    def _move_zoom():
        was = float(ov._cd)
        ov._cd = was + 7.0
        return lambda: setattr(ov, "_cd", was)

    return {
        "_probe_visibility":    ("PlateOverview.set_channel_visible(0, ...)", _flip_visibility),
        "_probe_contrast":      ("PlateOverview.set_channel_window(0, ...)", _move_contrast),
        "_probe_colormap":      ("PlateOverview.set_channel_color(0, ...)", _move_colour),
        "_probe_active_layer":  ("PlateOverview.set_active_layer(...)", _move_active),
        "_probe_selection":     ("PlateOverview.select_all() / clear_selection()", _move_selection),
        "_probe_current_well":  ("PlateWindow._current_well = <another region>", _move_current_well),
        "_probe_zoom":          ("PlateOverview._cd += 7", _move_zoom),
    }


def dead_probes(w):
    """Every probe whose reading cannot move, checked by moving the model underneath it.

    ``_snapshot`` catches a probe that raises; this catches one that returns a constant instead
    (silently unable to detect any duplicate, since nothing ever moves). Returns {probe name: why}.
    """
    app = _app()
    dead: dict[str, str] = {}
    drivers = _drive_the_model(w)
    for p in PROBES:
        name = p.__name__
        if name in BROKEN_PROBES:
            continue                       # already reported, and louder
        what, apply = drivers.get(name, (None, None))
        if apply is None:
            dead[name] = "no liveness driver in _drive_the_model: this probe is UNCHECKED"
            continue
        try:
            keys = list(p(w))
        except Exception:                   # noqa: BLE001 - already in BROKEN_PROBES
            continue
        if not keys:
            dead[name] = "the probe read NO slots at all on this acquisition"
            continue
        before = _snapshot(w)
        try:
            undo = apply()
        except Exception as exc:            # noqa: BLE001 - the driver itself is a finding
            dead[name] = f"driving {what} raised {type(exc).__name__}: {exc}"
            continue
        if undo is None:
            dead[name] = (f"the model has nothing for it to read here ({what} found no channel / "
                          f"no value), so this probe measured nothing on this acquisition")
            continue
        app.processEvents()
        after = _snapshot(w)
        try:
            undo()
            app.processEvents()
        except Exception as exc:            # noqa: BLE001 - a driver that cannot undo is a finding
            dead[name] = f"undoing {what} raised {type(exc).__name__}: {exc}"
            continue
        if not [k for k in keys if k in before and k in after and after[k] != before[k]]:
            dead[name] = (f"{what} changed nothing this probe reads — it cannot vary, so no "
                          f"widget could ever be reported as moving it")
    return dead


# How many control surfaces each concern is allowed to have. Fails on a count too HIGH (a
# duplicate appeared) as well as too LOW (the control was lost). 0 means the plate must not own
# this at all (napari owns it instead); every EXPECTED entry below is currently 0.
EXPECTED = {
    "contrast":              0,
    "visibility":            0,
    "channel colour / LUT":  0,
    "active layer":          0,
    "plate selection":       0,   # "Select all" is a widget and legitimately moves this; see EXEMPT
    "current well":          0,   # double-click: a gesture, no widget
    "zoom / viewport":       0,   # the wheel: a gesture, no widget
}

# Controls that legitimately move a probe as a side effect of something else, not a second owner
# of it. Each entry needs a reason.
EXEMPT = {
    ("plate selection", "select all"): "a bulk gesture over the one selection model, not a "
                                       "second representation of it",
}


def _label(wdg) -> str:
    """A stable, human-usable name for a widget: its text, else its tooltip, else its class."""
    for attr in ("text", "toolTip", "objectName"):
        try:
            v = getattr(wdg, attr)()
        except Exception:
            continue
        if v:
            return str(v).strip().splitlines()[0][:60]
    return type(wdg).__name__


def _where(wdg) -> str:
    """Which pane the widget lives in — the answer a human needs to go and look at it."""
    chain = []
    p = wdg
    for _ in range(12):
        p = p.parent()
        if p is None:
            break
        chain.append(type(p).__name__)
    # `_ChannelBar` / `LightweightViewer` are gone but kept here: a stale name costs nothing, a
    # missing one makes a re-introduced widget report its owner as "?".
    for interesting in ("_ChannelBar", "PlateOverview", "LightweightViewer", "PlateWindow"):
        if interesting in chain:
            return interesting
    return chain[-1] if chain else "?"


def interactive_widgets(root):
    """Every widget in the tree a user can act on, in a stable order."""
    from qtpy.QtWidgets import (
        QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox,
    )
    kinds = (QAbstractSlider, QAbstractSpinBox, QComboBox, QAbstractButton)
    out = []
    for k in kinds:
        for wdg in root.findChildren(k):
            if wdg not in out:
                out.append(wdg)
    # A composite control (superqt's QLabeledSlider, a spinbox's own buttons) contains an
    # interactive widget too; only the outermost of a nest is a control surface.
    outer = [w for w in out if not any(o is not w and o.isAncestorOf(w) for o in out)]
    return outer


def _actuate(wdg):
    """Move *wdg* the way a user would. Returns a callable that puts it back, or None.

    Value widgets are moved to a genuinely different value and then restored; buttons are clicked
    and cannot be un-clicked, which is why the baseline is re-read before every widget rather than
    once at the start.
    """
    from qtpy.QtWidgets import (
        QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox,
    )
    if not (wdg.isEnabled() and wdg.isVisible()):
        return None
    if isinstance(wdg, QAbstractSlider):
        old = wdg.value()
        lo, hi = wdg.minimum(), wdg.maximum()
        new = hi if old < (lo + hi) / 2 else lo
        if new == old:
            return None
        wdg.setValue(new)
        return lambda: wdg.setValue(old)
    if isinstance(wdg, QAbstractSpinBox):
        old = wdg.value()
        wdg.stepUp()
        if wdg.value() == old:
            wdg.stepDown()
        return lambda: wdg.setValue(old)
    if isinstance(wdg, QComboBox):
        old = wdg.currentIndex()
        if wdg.count() < 2:
            return None
        wdg.setCurrentIndex((old + 1) % wdg.count())
        return lambda: wdg.setCurrentIndex(old)
    if isinstance(wdg, QAbstractButton):
        if wdg.isCheckable():
            old = wdg.isChecked()
            wdg.setChecked(not old)
            return lambda: wdg.setChecked(old)
        wdg.click()
        return None
    return None


def _recorder(called, detail=None):
    """``rec(name)`` -> a callable that records and returns *ret*, appending into *called*.

    *detail*, when given, also receives ``(name, args, kwargs)`` — lets a caller check that a
    value a user set actually arrived at the call, not just that the widget changed.
    """
    def rec(name, ret=None):
        def f(*a, **k):
            called.append(name)
            if detail is not None:
                detail.append((name, a, k))
            return ret
        return f
    return rec


#: Entry points that must be stubbed BEFORE ``PlateWindow`` is constructed: ``ViewerManager``
#: captures ``win.run_operator`` as a bound method at construction, so patching the class
#: afterwards leaves already-built region windows calling the real handler.
_EARLY_STUBS = ("run_operator",)


def _neutralise_early(monkey, called, detail=None):
    """The stubs that have to be in place before the plate window exists. See :data:`_EARLY_STUBS`."""
    import squidxplorer._viewer as V

    rec = _recorder(called, detail)
    for m in _EARLY_STUBS:
        if hasattr(V.PlateWindow, m):
            monkey(V.PlateWindow, m, rec(f"PlateWindow.{m}"))


def _neutralise(win, monkey, called=None):
    """Stop a click from doing something a gate has no business doing.

    Anything that opens a modal dialog, launches a multi-minute operator run, re-ingests, or
    closes the app is turned into a recorded no-op — observed, but not run.
    """
    from qtpy.QtWidgets import QFileDialog, QMessageBox
    import squidxplorer._viewer as V

    called = [] if called is None else called
    rec = _recorder(called)

    monkey(QFileDialog, "getExistingDirectory", staticmethod(rec("getExistingDirectory", "")))
    monkey(QFileDialog, "getOpenFileName", staticmethod(rec("getOpenFileName", ("", ""))))
    monkey(QFileDialog, "getSaveFileName", staticmethod(rec("getSaveFileName", ("", ""))))
    monkey(QFileDialog, "exec_", rec("QFileDialog.exec_", 0))
    for m in ("warning", "information", "critical", "question", "about"):
        monkey(QMessageBox, m, staticmethod(rec(f"QMessageBox.{m}", 0)))
    monkey(QMessageBox, "exec_", rec("QMessageBox.exec_", 0))
    for m in (*_EARLY_STUBS, "ingest", "close", "_open_acquisition_dialog"):
        if hasattr(V.PlateWindow, m):
            monkey(V.PlateWindow, m, rec(f"PlateWindow.{m}"))
    return called


def find_duplicate_controls(win, verbose=False):
    """{probe name: [widget descriptions that can move it]} — the gate's raw evidence."""
    app = _app()
    owners: dict[str, list[str]] = {}
    for wdg in interactive_widgets(win):
        before = _snapshot(win)
        try:
            undo = _actuate(wdg)
        except Exception as exc:                       # noqa: BLE001 - recorded, then reported
            RAISING_CONTROLS[f"{_label(wdg)} [{type(wdg).__name__} in {_where(wdg)}]"] = \
                f"{type(exc).__name__}: {exc}"
            continue
        if undo is None and not _is_button(wdg):
            continue
        app.processEvents()
        after = _snapshot(win)
        desc = f"{_label(wdg)} [{type(wdg).__name__} in {_where(wdg)}]"
        for key, old in before.items():
            if key in after and after[key] != old:
                if (_concern_of(key), _label(wdg).lower()) in EXEMPT:
                    continue
                owners.setdefault(key, [])
                if desc not in owners[key]:
                    owners[key].append(desc)
                if verbose:
                    print(f"    {key:26s} <- {desc}")
        if undo is not None:
            try:
                undo()
                app.processEvents()
            except Exception:
                pass
    return owners


def _is_button(wdg):
    from qtpy.QtWidgets import QAbstractButton
    return isinstance(wdg, QAbstractButton)


def contrast_surfaces(win):
    """Contrast controls on the PLATE side, which must be zero.

    Kept alongside the empirical sweep because it must hold even for a slider that is currently
    disabled or hidden, which the sweep would not actuate.
    """
    from qtpy.QtWidgets import QAbstractSlider, QPushButton
    plate = getattr(win, "_overview", None)
    if plate is None:
        return [], []
    sliders = [f"{_label(s)} [{type(s).__name__}]" for s in plate.findChildren(QAbstractSlider)]
    autos = [f"{_label(b)} [auto button]" for b in plate.findChildren(QPushButton)
             if "auto" in b.text().lower()]
    return sliders, autos


def _drain_preview(win, timeout_s=180):
    """Block until the plate's background preview stream has finished.

    Quiescence is part of the measurement: the plate's contrast is a running percentile, and
    sweeping while it is still live attributes an unrelated update to whatever widget was under
    the cursor at that moment.
    """
    import time
    app = _app()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        p = getattr(win, "_preview", None)
        if p is None or not p.isRunning():
            break
        app.processEvents()
        time.sleep(0.02)
    for _ in range(20):            # let the queued tileReady slots actually run
        app.processEvents()
        time.sleep(0.01)


def gate_no_duplicated_controllers(dataset=PLATE, verbose=False, mutate=None):
    """Returns (ok, list of human-readable findings).

    *mutate*, when given, is called with the shown, ingested window before the sweep — how
    --self-test mounts a duplicate control.
    """
    import squidxplorer._viewer as V
    app = _app()
    findings, ok = [], True
    BROKEN_PROBES.clear()

    win = V.PlateWindow(None)
    win.resize(1600, 900)
    win.show()                     # SHOWN: an unshown splitter reports defaults, not the product
    app.processEvents()
    win.ingest(dataset)
    app.processEvents()
    if win._reader is None:
        return False, [f"FAIL  could not open {dataset}: {win._readout.text()!r}"]
    _drain_preview(win)
    if mutate is not None:
        mutate(win)
        app.processEvents()

    # 1. structural: the plate must own no contrast control at all.
    sliders, autos = contrast_surfaces(win)
    if sliders or autos:
        ok = False
        findings.append(f"FAIL  contrast: the plate view still carries {len(sliders)} slider(s) "
                        f"and {len(autos)} auto button(s) — {sliders + autos}")
    else:
        findings.append("PASS  contrast: 0 sliders, 0 auto buttons in the plate view")

    # 2. the probes themselves, before anything is actuated (run first because the sweep clicks
    #    "Select all", which would make a later liveness check see a no-op).
    FROZEN_PROBES.clear()
    FROZEN_PROBES.update(dead_probes(win))

    # 3. the empirical half: actuate everything, group by the state it moved.
    patches = []

    def monkey(obj, name, value):
        # An inherited C++ slot (QWidget.close) re-set on the subclass becomes an unbound sip
        # method, so restoring it must DELETE rather than re-assign; track own-vs-inherited here.
        patches.append((obj, name, obj.__dict__.get(name, _MISSING)))
        setattr(obj, name, value)

    try:
        _neutralise(win, monkey)
        owners = find_duplicate_controls(win, verbose=verbose)
    finally:
        for obj, name, old in reversed(patches):
            if old is _MISSING:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass
            else:
                setattr(obj, name, old)

    by_concern: dict[str, dict[str, list[str]]] = {}
    for key, got in owners.items():
        by_concern.setdefault(_concern_of(key), {})[key] = got

    for concern, expected in sorted(EXPECTED.items()):
        slots = by_concern.get(concern, {})
        worst = max((len(v) for v in slots.values()), default=0)
        if worst > expected:
            ok = False
            detail = "".join(
                f"          {k}:\n" + "".join(f"            - {g}\n" for g in v)
                for k, v in sorted(slots.items()) if len(v) > expected)
            findings.append(f"FAIL  {concern}: {worst} control surfaces over one value, "
                            f"expected {expected} —\n{detail.rstrip()}")
        else:
            findings.append(f"PASS  {concern}: at most {worst} control surface(s) over any one "
                            f"value (expected at most {expected})")

    # 4. an undeclared concern: state two+ widgets can move that nobody has assigned an owner to.
    for concern, slots in sorted(by_concern.items()):
        if concern in EXPECTED:
            continue
        for key, got in sorted(slots.items()):
            if len(got) > 1:
                ok = False
                findings.append(f"FAIL  {key}: UNDECLARED concern with {len(got)} control "
                                f"surfaces — add it to EXPECTED and pick an owner: {got}")

    # 5. a probe that could not read is a check that did not run, and fails rather than vanishing.
    for name, why in sorted(BROKEN_PROBES.items()):
        ok = False
        findings.append(f"FAIL  {name}: the probe itself raised, so its concern was NOT "
                        f"checked — {why}")

    # 6. a control whose actuation raised never reached step 3, so was never checked as an owner.
    for desc, why in sorted(RAISING_CONTROLS.items()):
        ok = False
        findings.append(f"FAIL  {desc}: actuating it raised, so it was NOT measured against any "
                        f"concern — {why}")

    # 7. a probe that cannot vary is a decoration: it would pass every concern by never measuring.
    for name, why in sorted(FROZEN_PROBES.items()):
        ok = False
        findings.append(f"FAIL  {name}: the probe cannot VARY, so its concern was NOT "
                        f"checked — {why}")
    if not FROZEN_PROBES and not BROKEN_PROBES:
        findings.append(f"PASS  every probe ({len(PROBES)}) reads a value that moves when the "
                        f"model under it moves")

    win.close()
    app.processEvents()
    return ok, findings


# --- GATE 3: no dead controls, clicked not called -----------------------------------------------

#: Widget classes defined in these packages are somebody else's controls, not ours to declare
#: alive or dead.
_THIRD_PARTY = ("napari", "superqt", "vispy", "qtpy", "PyQt", "PySide", "qtconsole")


def _third_party(wdg) -> bool:
    """True when *wdg* — or any widget it sits inside — belongs to a library rather than to us."""
    node = wdg
    for _ in range(20):
        if node is None:
            return False
        mod = type(node).__module__ or ""
        if mod.startswith(_THIRD_PARTY):
            # A plain QPushButton is `PyQt6.QtWidgets`; only a library SUBCLASS/CONTAINER disqualifies.
            if not mod.startswith(("PyQt", "PySide", "qtpy")):
                return True
        node = node.parent()
    return False


def our_controls(root):
    """Every control in *root* that is ours, with hidden/disabled ones kept (reported, not dropped)."""
    return [w for w in interactive_widgets(root) if not _third_party(w)]


class _LogSpy:
    """Every log record OURS emitted while a control is being actuated.

    Scoped to ``squid.xplorer`` — offscreen Qt logs its own platform warnings on every click, so
    an unscoped spy would credit every control with "something happened" including a dead one.
    """

    def __init__(self) -> None:
        import logging

        from squidxplorer._logpane import XPLORER_ROOT

        self.records: list[str] = []
        outer = self

        class _H(logging.Handler):
            def emit(self, record):
                if not record.name.startswith(XPLORER_ROOT):
                    return
                try:
                    outer.records.append(record.getMessage()[:200])
                except Exception:                                  # noqa: BLE001
                    outer.records.append(record.msg if isinstance(record.msg, str) else "?")

        self._h = _H()
        # The level check happens on the emitting logger, not the stdlib root.
        self._root = logging.getLogger(XPLORER_ROOT)

    def __enter__(self):
        self._root.addHandler(self._h)
        self._level, self._root.level = self._root.level, 10
        return self

    def __exit__(self, *exc):
        self._root.removeHandler(self._h)
        self._root.level = self._level
        return False


def _texts(root) -> dict:
    """Every piece of text a user can read in *root*, keyed by (class, ordinal) rather than
    identity so a click that REPLACES a label still reads as a change."""
    from qtpy.QtWidgets import QLabel, QLineEdit, QPlainTextEdit, QTextEdit

    out, n = {}, {}
    for cls in (QLabel, QLineEdit, QPlainTextEdit, QTextEdit):
        for wdg in root.findChildren(cls):
            i = n[cls.__name__] = n.get(cls.__name__, -1) + 1
            try:
                txt = wdg.text() if hasattr(wdg, "text") else wdg.toPlainText()
            except Exception:                                      # noqa: BLE001 - deleted C++ half
                continue
            out[f"{cls.__name__}[{i}]"] = str(txt)[:400]
    return out


def _widget_states(root) -> dict:
    """Every control's OWN state. A control that enables, disables, checks or re-fills another
    control has reached something, and this is the cheapest true reading of that."""
    from qtpy.QtWidgets import QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox

    out, n = {}, {}
    for wdg in interactive_widgets(root):
        cls = type(wdg).__name__
        i = n[cls] = n.get(cls, -1) + 1
        key = f"{cls}[{i}]"
        try:
            state = [wdg.isEnabled(), wdg.isVisible()]
            if isinstance(wdg, QAbstractButton):
                state += [wdg.isChecked(), wdg.text()]
            elif isinstance(wdg, (QAbstractSlider, QAbstractSpinBox)):
                state += [wdg.value(), wdg.minimum(), wdg.maximum()]
            elif isinstance(wdg, QComboBox):
                state += [wdg.currentIndex(), wdg.count(), wdg.currentText()]
        except Exception:                                          # noqa: BLE001
            continue
        out[key] = tuple(state)
    return out


def _layer_state(root) -> dict:
    """The napari model under a region window: what the pixels are, via MosaicLayers."""
    pane = getattr(root, "_pane", None)
    mosaic = getattr(pane, "mosaic", None) if pane is not None else None
    if mosaic is None:
        return {}
    out = {}
    try:
        out["mosaic ops"] = tuple(mosaic.ops())
        for i, ly in enumerate(mosaic.ours()):
            cl = getattr(ly, "contrast_limits", None)
            cm = getattr(ly, "colormap", None)
            out[f"layer[{i}]"] = (
                getattr(ly, "name", "?"), bool(getattr(ly, "visible", True)),
                tuple(cl) if cl is not None else None,
                str(getattr(cm, "name", cm))[:40],
                tuple(getattr(ly, "scale", ()) or ()),
            )
        viewer = getattr(pane, "_viewer", None)
        dims = getattr(viewer, "dims", None)
        if dims is not None:
            out["napari dims step"] = tuple(dims.current_step)
            out["napari ndisplay"] = int(dims.ndisplay)
    except Exception as exc:                                       # noqa: BLE001 - recorded, not hidden
        out["mosaic UNREADABLE"] = f"{type(exc).__name__}: {exc}"
    return out


def _windows_open() -> tuple:
    from qtpy.QtWidgets import QApplication

    return tuple(sorted(f"{type(w).__name__}:{w.windowTitle()}"
                        for w in QApplication.topLevelWidgets() if w.isVisible()))


def _tab_state(root) -> dict:
    from qtpy.QtWidgets import QTabWidget

    out = {}
    for i, tabs in enumerate(root.findChildren(QTabWidget)):
        try:
            out[f"tabs[{i}]"] = (tuple(tabs.tabText(j) for j in range(tabs.count())),
                                 tabs.currentIndex())
        except Exception:                                          # noqa: BLE001
            continue
    return out


def _pixels(root, blank=None):
    """A hash of what the window draws, with *blank* (the widget being actuated) painted out —
    its own click repaint (focus ring, hover, pressed state) is the actuation, not an outcome.
    4px pad because a focus ring is drawn outside the widget's own geometry.
    """
    import hashlib

    from qtpy.QtCore import QPoint, QRect
    from qtpy.QtGui import QPainter

    try:
        img = root.grab().toImage()
        if blank is not None and blank is not root:
            dpr = img.devicePixelRatioF() or 1.0
            tl = blank.mapTo(root, QPoint(0, 0))
            pad = 4
            rect = QRect(int((tl.x() - pad) * dpr), int((tl.y() - pad) * dpr),
                         int((blank.width() + 2 * pad) * dpr),
                         int((blank.height() + 2 * pad) * dpr))
            painter = QPainter(img)
            painter.fillRect(rect, painter.background())
            painter.end()
        ptr = img.bits()
        ptr.setsize(img.sizeInBytes())
        return hashlib.blake2b(bytes(ptr), digest_size=8).hexdigest()
    except Exception:                                              # noqa: BLE001
        return None


def _quieten_timers(root):
    """Stop every repeating QTimer under *root* (memory poll, log panel), and say how many.

    A periodic repaint would otherwise make the pixel hash flicker independent of any click.
    """
    from qtpy.QtCore import QTimer

    stopped = 0
    for timer in root.findChildren(QTimer):
        if timer.isActive() and not timer.isSingleShot():
            timer.stop()
            stopped += 1
    return stopped


def _pixel_noise(root, app, n=6, seconds=1.5) -> bool:
    """Do idle grabs of *root* differ across real time? If so, pixels are not evidence.

    Sampled over a second rather than back-to-back, so a slow (e.g. 2s) timer tick isn't missed.
    """
    import time

    seen = set()
    for _ in range(n):
        app.processEvents()
        seen.add(_pixels(root))
        time.sleep(seconds / n)
    return len(seen) > 1


def wait_still(roots, app, tries=20, step=0.05) -> bool:
    """Pump events until two consecutive grabs of every root agree; returns whether they ever did.

    `_pixel_noise` proves no periodic repaint; this catches a repaint queued by the PREVIOUS
    control landing inside the next one's measurement and being misattributed to it.
    """
    import time

    for _ in range(tries):
        first = [_pixels(r) for r in roots]
        app.processEvents()
        time.sleep(step)
        app.processEvents()
        if [_pixels(r) for r in roots] == first:
            return True
    return False


def _fingerprint(root, use_pixels: bool, blank=None) -> dict:
    out = {}
    out.update({f"text {k}": v for k, v in _texts(root).items()})
    out.update({f"control {k}": v for k, v in _widget_states(root).items()})
    out.update({f"tab {k}": v for k, v in _tab_state(root).items()})
    out.update({f"napari {k}": v for k, v in _layer_state(root).items()})
    out["windows on screen"] = _windows_open()
    if use_pixels:
        out["pixels"] = _pixels(root, blank)
    return out


#: Why each neutralised entry point is neutralised — printed alongside the "reached but not run"
#: verdict, so that claim is never confused with "did what it promises".
NEUTRALISED_WHY = {
    "RegionViewer._open_3d": "vispy needs a live GL context; the volume path cannot be driven "
                             "offscreen (docs/rendering-contract.md)",
    "RegionViewer._view_roi_2d": "opens a child window and a second preview stream",
    "RegionViewer._record_movie": "writes an .mp4 and runs a multi-second encode",
    "RegionViewer._open_roi_children": "opens one child window per ROI",
    "PlateWindow.run_operator": "a multi-minute operator run over the plate",
    "PlateWindow.ingest": "re-opens the acquisition under the sweep",
    "PlateWindow.close": "ends the window the sweep is walking",
    "PlateWindow._open_acquisition_dialog": "a modal file dialog",
}


#: Input controls: moving one changes nothing on screen, so the sweep would call them dead.
#: Excluded from the sweep and instead PROVEN by prove_inputs_reach_the_run, which checks the
#: value actually arrives at the call.
DEFERRED_INPUTS = {
    ("view", "QComboBox"): "the operator picker — read by 'Run'",
    ("view", "save"): "preview vs persist — read by 'Run'",
}


def prove_inputs_reach_the_run(view, detail, app):
    """Set the operator picker and ``save`` box, click Run, and read the arguments that actually
    arrived at ``PlateWindow.run_operator`` — "the widget changed" is not proof the value did.

    Returns rows for both controls; a run that never reaches the call gives them "no outcome".
    """
    from qtpy.QtWidgets import QCheckBox, QComboBox, QPushButton

    combo = next((c for c in view.findChildren(QComboBox) if not _third_party(c)), None)
    save = next((c for c in view.findChildren(QCheckBox) if c.text() == "save"), None)
    run = next((b for b in view.findChildren(QPushButton) if b.text() == "Run"), None)
    if combo is None or save is None or run is None:
        return [("view", "Run inputs", "-", "no outcome",
                 "the operator picker, the save box or Run is missing from this window")]

    # Set to a value different from the default, so a handler that ignores it and passes a
    # default cannot accidentally agree with what was set.
    want_index = 1 if combo.count() > 1 else 0
    combo.setCurrentIndex(want_index)
    want_key = combo.currentData()
    save.setChecked(not save.isChecked())
    want_save = save.isChecked()

    n = len(detail)
    run.click()
    app.processEvents()
    calls = [d for d in detail[n:] if d[0] == "PlateWindow.run_operator"]
    if not calls:
        return [("view", combo.currentText()[:30], "QComboBox", "no outcome",
                 "clicking Run did not reach PlateWindow.run_operator at all"),
                ("view", "save", "QCheckBox", "no outcome",
                 "clicking Run did not reach PlateWindow.run_operator at all")]
    _name, args, kwargs = calls[-1]
    # The stub replaced a CLASS attribute, so it's called unbound: args[0] is the PlateWindow.
    pos = args[1:]
    got_key = kwargs.get("key", pos[0] if pos else None)
    got_save = kwargs.get("save")
    rows = []
    rows.append(("view", combo.currentText()[:30], "QComboBox",
                 "reaches" if got_key == want_key else "no outcome",
                 f"the picked operator arrived as run_operator(key={got_key!r})"
                 if got_key == want_key else
                 f"picked {want_key!r}, the run was asked for {got_key!r}"))
    rows.append(("view", "save", "QCheckBox",
                 "reaches" if got_save == want_save else "no outcome",
                 f"arrived as run_operator(save={got_save!r})" if got_save == want_save else
                 f"ticked save={want_save!r}, the run was asked for save={got_save!r}"))
    return rows


def _home_tab(win):
    """Put the plate back on its Operators home tab between controls — the first card clicked
    opens its own tab, hiding the cards behind it, so the sweep must restore the surface."""
    tabs = getattr(win, "_left_tabs", None)
    if tabs is not None and tabs.count():
        tabs.setCurrentIndex(0)


def sweep_controls(root, kind: str, app, watched=None, recorder=None, settle=None, observed=None):
    """Actuate every control of *root* and return one row per control: (label, class, verdict, evidence).

    Verdicts: ``reaches`` (something observable changed), ``raised`` (the click raised),
    ``neutralised`` (reached a stubbed entry point; its outcome is not observed here),
    ``no outcome`` (nothing changed/logged/called), ``hidden``/``disabled`` (not reachable now).

    *watched* is extra roots to fingerprint (a region window's chips can change the plate too).
    """
    recorder = [] if recorder is None else recorder
    observed = [] if observed is None else observed
    watched = [root, *(watched or [])]
    for r in watched:
        _quieten_timers(r)
    use_pixels = [not _pixel_noise(r, app) for r in watched]
    for r, p in zip(watched, use_pixels):
        if not p:
            print(f"...   {kind:5} {'(pixels are NOT evidence here: ' + type(r).__name__ + ' repaints on its own)':<40}",
                  file=sys.__stdout__, flush=True)
    rows = []

    def emit(row):
        # Streamed as produced: a click that hangs or crashes the process still leaves the last
        # printed line naming the control that was in the chair.
        rows.append(row)
        print(f"...   {kind:5} {row[0][:40]:<40} {row[2]}", file=sys.__stdout__, flush=True)

    for wdg in our_controls(root):
        if settle is not None:
            settle(root)
            app.processEvents()
        label, cls = _label(wdg), type(wdg).__name__
        if (kind, label) in DEFERRED_INPUTS or (kind, cls) in DEFERRED_INPUTS:
            continue                    # proven by `prove_inputs_reach_the_run`, not by a sweep
        if not wdg.isVisible():
            emit((label, cls, "hidden", "not on screen in this state"))
            continue
        if not wdg.isEnabled():
            tip = (wdg.toolTip() or "").strip().splitlines()
            emit((label, cls, "disabled", tip[0][:120] if tip else "no tooltip says why"))
            continue
        still = wait_still(watched, app)
        # `blank=wdg` only on the window it lives in — other watched windows have nothing to paint out.
        before = [_fingerprint(r, p and still, wdg if i == 0 else None)
                  for i, (r, p) in enumerate(zip(watched, use_pixels))]
        n_calls, n_obs = len(recorder), len(observed)
        with _LogSpy() as spy:
            try:
                undo = _actuate(wdg)
            except Exception as exc:                               # noqa: BLE001 - THE finding
                emit((label, cls, "raised", f"{type(exc).__name__}: {exc}"))
                continue
            app.processEvents()
            # A still frame again before reading the outcome, so a LATE repaint (tab building, a
            # worker's first tile) is credited to this control rather than the next one.
            wait_still(watched, app, tries=10)
        after = [_fingerprint(r, p and still, wdg if i == 0 else None)
                 for i, (r, p) in enumerate(zip(watched, use_pixels))]

        reached = recorder[n_calls:]
        if reached:
            name = reached[0]
            emit((label, cls, "neutralised",
                  f"reached {name} — not run here: {NEUTRALISED_WHY.get(name, 'see _neutralise')}"))
            if undo is not None:
                try:
                    undo()
                    app.processEvents()
                except Exception:                                  # noqa: BLE001
                    pass
            continue

        changed = []
        for i, (b, a) in enumerate(zip(before, after)):
            where = "" if i == 0 else f" (on the {['', 'plate'][min(i, 1)]})"
            for key in sorted(b):
                if key in a and a[key] != b[key]:
                    # The control's own state moving is the actuation, not an outcome.
                    if i == 0 and key.startswith(f"control {cls}["):
                        continue
                    changed.append(f"{key}{where}")
        for call in dict.fromkeys(observed[n_obs:]):
            changed.append(f"called {call}")
        if spy.records:
            changed.append(f"logged: {spy.records[0][:110]!r}")
        emit((label, cls, "reaches" if changed else "no outcome",
                     "; ".join(changed[:4]) if changed else
                     "nothing in the app changed, nothing was logged"))
        if undo is not None:
            try:
                undo()
                app.processEvents()
            except Exception:                                      # noqa: BLE001
                pass
    return rows


def _model_pane_class():
    """THE shared headless pane adapter, from its production home (``_napari_pane``): a real
    Qt-free ``ViewerModel`` + real ``MosaicLayers``. Does not prove a layer was PAINTED, only
    that it exists in the model with the right scale/contrast/visibility."""
    from squidxplorer._napari_pane import model_pane_class

    return model_pane_class()


def _watch_window_stacking(monkey, seen):
    """Wrap (never stub) the window-stacking calls, so "bring the plate forward" is observable.

    ``raise_()`` / ``activateWindow()`` are no-ops on the offscreen platform, so a click that
    only re-stacks the window would otherwise read as dead; wrapping records the call while the
    real (no-op) implementation still runs.
    """
    from qtpy.QtWidgets import QWidget

    import squidxplorer._viewer as V

    def wrap(name):
        real = getattr(QWidget, name)

        def f(self, *a, **k):
            seen.append(f"{type(self).__name__}.{name}()")
            return real(self, *a, **k)
        return f

    for m in ("raise_", "activateWindow", "showNormal"):
        monkey(V.PlateWindow, m, wrap(m))


def _neutralise_view(monkey, called):
    """Stop a region window's chips from doing what this harness has no business doing.

    Same rule as :func:`_neutralise`: the call is recorded and not run. Appends into the
    caller's *called* list so one recorder covers both the plate's entry points and this window's.
    """
    from squidxplorer import _region_viewer as RV

    def rec(name):
        def f(*a, **k):
            called.append(name)
            return None
        return f

    for m in ("_open_3d", "_record_movie", "_open_roi_children", "_view_roi_2d"):
        if hasattr(RV.RegionViewer, m):
            monkey(RV.RegionViewer, m, rec(f"RegionViewer.{m}"))
    return called


def gate_no_dead_controls(dataset=PLATE, mutate_plate=None, mutate_view=None, verbose=False):
    """Open a real plate and a real region window, sweep both, and report every dead control.

    Returns ``(ok, findings, rows)``; *rows* is the inventory, which ``--inventory`` prints.
    """
    import squidxplorer._napari_pane as napari_pane
    import squidxplorer._viewer as V

    app = _app()
    patches = []

    def monkey(obj, name, value):
        patches.append((obj, name, obj.__dict__.get(name, _MISSING)))
        setattr(obj, name, value)

    pane_cls = _model_pane_class()
    monkey(napari_pane, "make_pane", staticmethod(lambda *a, **k: (pane_cls(), "napari", "")))
    # BEFORE the window: see `_EARLY_STUBS`. `ingest` cannot be stubbed here, so the rest of
    # `_neutralise` still runs after the acquisition is open.
    recorded: list[str] = []
    detail: list[tuple] = []            # (name, args, kwargs) — see `prove_inputs_reach_the_run`
    _neutralise_early(monkey, recorded, detail)

    win = V.PlateWindow(None)
    win.resize(1600, 900)
    win.show()
    app.processEvents()
    win.ingest(dataset)
    app.processEvents()
    if win._reader is None:
        for obj, name, old in reversed(patches):
            setattr(obj, name, old) if old is not _MISSING else delattr(obj, name)
        return False, [f"FAIL  could not open {dataset}: {win._readout.text()!r}"], []
    _drain_preview(win)

    rows: list[tuple] = []
    try:
        if mutate_plate is not None:
            mutate_plate(win)
            app.processEvents()
        _neutralise(win, monkey, recorded)
        # BEFORE the window is opened: `RegionViewer._chip` connects a bound method captured at
        # construction, so patching the class afterwards leaves already-built chips calling the
        # real handler (measured: SIGSEGV in vispy when an unstubbed 3D chip was clicked).
        _neutralise_view(monkey, recorded)
        seen: list[str] = []
        _watch_window_stacking(monkey, seen)
        rows += [("plate", *r) for r in sweep_controls(win, "plate", app, recorder=recorded,
                                                       settle=_home_tab, observed=seen)]

        regions = list((win._reader.metadata or {}).get("regions") or [])
        view = win._viewer_manager.open(regions[:1]) if regions else None
        app.processEvents()
        if view is None:
            rows.append(("view", "(no region window)", "-", "no window",
                         "the acquisition declares no regions, so no view could be opened"))
        else:
            if mutate_view is not None:
                mutate_view(view)
                app.processEvents()
            # ONE recorder for both windows: a region chip that calls back into the plate (Run ->
            # PlateWindow.run_operator) has to be seen as reaching a neutralised entry point too.
            rows += [("view", *r) for r in sweep_controls(view, "view", app, watched=[win],
                                                          recorder=recorded, observed=seen)]
            # The input controls the sweep skipped: proven by the arguments that arrive at the run.
            rows += prove_inputs_reach_the_run(view, detail, app)
    finally:
        for obj, name, old in reversed(patches):
            if old is _MISSING:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass
            else:
                setattr(obj, name, old)
        try:
            win._viewer_manager.close_all()
        except Exception:                                          # noqa: BLE001
            pass
        win.close()
        app.processEvents()

    dead = [r for r in rows if r[3] == "no outcome"]
    raised = [r for r in rows if r[3] == "raised"]
    findings = []
    for where, label, cls, _v, why in raised:
        findings.append(f"FAIL  {where}: {label!r} [{cls}] RAISED when clicked — {why}")
    for where, label, cls, _v, why in dead:
        findings.append(f"FAIL  {where}: {label!r} [{cls}] has no observable outcome — {why}")
    def n(verdict):
        return sum(1 for r in rows if r[3] == verdict)

    findings.append(f"PASS  {n('reaches')} control(s) reached something observable; "
                    f"{n('neutralised')} reached an entry point this harness stubs (outcome not "
                    f"observed here); {n('hidden')} hidden, {n('disabled')} disabled")
    return (not dead and not raised), findings, rows


def print_inventory(rows) -> None:
    """The deliverable a human reads: one line per control, with its verdict and the evidence."""
    width = max((len(r[1]) for r in rows), default=10)
    where = None
    for row in rows:
        if row[0] != where:
            where = row[0]
            title = ("PLATE WINDOW" if where == "plate" else "REGION VIEWER")
            print("\n" + "-" * 100)
            print(f"{title} — every control a user can act on")
            print("-" * 100)
        _w, label, cls, verdict, why = row
        print(f"  {label:<{width}}  {cls:<16} {verdict:<13} {why}")


# --- mutation check: prove the gate can fail -----------------------------------------------------

def _mount_contrast_duplicate(win):
    """Bolt a second, independently draggable owner of the contrast window onto the plate —
    mounted on the shown, ingested window itself, so the mutation is guaranteed live."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QSlider
    ov = win._overview
    for c_i in range(len(ov.channel_windows() or [])):
        s = QSlider(Qt.Orientation.Horizontal, win)
        s.setRange(0, 65535)
        s.setValue(30000)
        s.valueChanged.connect(lambda v, i=c_i: ov.set_channel_window(i, 0.0, float(v)))
        win.statusBar().addWidget(s)
        s.show()


def _mount_visibility_duplicate(win):
    """A second owner of ``visibility[0]``, in a concern the gate was never specifically taught."""
    from qtpy.QtWidgets import QCheckBox
    ov = win._overview
    box = QCheckBox("show ch0", win)
    box.setChecked(True)
    box.toggled.connect(lambda on: ov.set_channel_visible(0, on))
    win.statusBar().addWidget(box)
    box.show()


def _freeze_a_probe(win):
    """Make ONE probe's value a constant, so it cannot vary and no widget could ever be reported
    as moving it. Patched on the class (``_current_well`` is a property); undone by
    :func:`_unfreeze_a_probe`."""
    import squidxplorer._viewer as V

    _FROZEN_PROBE_ORIGINAL.append(V.PlateWindow._current_well)
    V.PlateWindow._current_well = property(lambda self: None, lambda self, value: None)


#: The real ``PlateWindow._current_well`` descriptor while ``_freeze_a_probe`` is in effect.
_FROZEN_PROBE_ORIGINAL: list = []


def _unfreeze_a_probe():
    import squidxplorer._viewer as V

    while _FROZEN_PROBE_ORIGINAL:
        V.PlateWindow._current_well = _FROZEN_PROBE_ORIGINAL.pop()


def _mount_dead_button(win):
    """Bolt a chip that looks alive and is wired to nothing onto the plate."""
    from qtpy.QtWidgets import QPushButton
    b = QPushButton("⚙ tune", win)
    b.setToolTip("Tune the operator on screen.")      # a promise; nothing keeps it
    win.statusBar().addWidget(b)
    b.show()


def _mount_dead_view_button(view):
    """The same mutation on a region window, wired to a handler that returns at its first guard
    rather than being connected to nothing — the shape GATE 3 actually has to catch."""
    from qtpy.QtWidgets import QPushButton

    def _guarded():
        if getattr(view, "_this_attribute_does_not_exist", None) is None:
            return                                    # ...every time
        view._say("unreachable")

    b = QPushButton("↯ enhance", view)
    b.setToolTip("Enhance this window.")
    b.clicked.connect(lambda _=False: _guarded())
    view.layout().addWidget(b)
    b.show()


def self_test_dead_controls(dataset=PLATE):
    """Prove GATE 3 can fail: mount a dead chip on each window and require it to be named."""
    print("=" * 100)
    print("SELF-TEST 1/3: GATE 3 must PASS on the tree as it stands")
    ok, findings, _rows = gate_no_dead_controls(dataset)
    for f in findings:
        print("   ", f)
    if not ok:
        print("\nSELF-TEST FAILED: GATE 3 is red before the mutation was applied.")
        return 1

    print("=" * 100)
    print("SELF-TEST 2/3: bolting a chip wired to NOTHING onto the plate — must fail")
    ok_mut, findings_mut, _ = gate_no_dead_controls(dataset, mutate_plate=_mount_dead_button)
    for f in findings_mut:
        if f.startswith("FAIL"):
            print("   ", f)
    if ok_mut or not any("tune" in f for f in findings_mut):
        print("\nSELF-TEST FAILED: a chip connected to nothing was added and GATE 3 did not name "
              "it. The gate does not work.")
        return 1
    print("\n    the gate bit on a plate chip.")

    print("=" * 100)
    print("SELF-TEST 3/3: a REGION-window chip whose handler returns at its first guard —")
    print("               the shape of the plate's timepoint bar.")
    ok_v, findings_v, _ = gate_no_dead_controls(dataset, mutate_view=_mount_dead_view_button)
    for f in findings_v:
        if f.startswith("FAIL"):
            print("   ", f)
    if ok_v or not any("enhance" in f for f in findings_v):
        print("\nSELF-TEST FAILED: a region chip with a dead handler was added and GATE 3 stayed "
              "green — it only sees the plate, or it only sees a missing connection.")
        return 1
    print("\n    the gate bit on a region window, on an EARLY RETURN rather than a missing "
          "connection.")

    print("=" * 100)
    print("SELF-TEST: mutations removed; confirming GATE 3 is green again")
    ok_back, _f, _r = gate_no_dead_controls(dataset)
    if not ok_back:
        print("SELF-TEST FAILED: GATE 3 did not recover after the mutations were removed.")
        return 1
    print("\nSELF-TEST PASSED (GATE 3): passes clean, names a dead plate chip, names a dead "
          "region chip, and recovers.")
    return 0


def self_test(dataset=PLATE):
    """Reintroduce the duplicate, require the gate to bite, remove it, require the gate to pass."""
    print("=" * 100)
    print("SELF-TEST 1/4: the gate must PASS on the tree as it stands")
    ok, findings = gate_no_duplicated_controllers(dataset)
    for f in findings:
        print("   ", f)
    if not ok:
        print("\nSELF-TEST FAILED: the gate is red before the mutation was even applied.")
        return 1

    print("=" * 100)
    print("SELF-TEST 2/4: reintroducing a per-channel contrast slider on the plate —")
    print("               the gate MUST now fail, or it is decorative.")
    ok_mut, findings_mut = gate_no_duplicated_controllers(
        dataset, mutate=_mount_contrast_duplicate)
    for f in findings_mut:
        print("   ", f)
    if ok_mut:
        print("\nSELF-TEST FAILED: a duplicate contrast slider was added and the gate stayed "
              "GREEN. The gate does not work.")
        return 1
    if not any(f.startswith("FAIL") and "contrast" in f for f in findings_mut):
        print("\nSELF-TEST FAILED: the gate failed, but not on contrast.")
        return 1
    print("\n    the gate bit, as it must.")

    # A gate that only knows about contrast is a hard-coded assertion; duplicate a DIFFERENT
    # control to prove it generalises instead.
    print("=" * 100)
    print("SELF-TEST 3/4: duplicating a control in a DIFFERENT concern (channel visibility) —")
    print("               the gate must generalise, not just know about contrast.")
    ok_vis, findings_vis = gate_no_duplicated_controllers(
        dataset, mutate=_mount_visibility_duplicate)
    for f in findings_vis:
        if f.startswith("FAIL"):
            print("   ", f)
    if ok_vis:
        print("\nSELF-TEST FAILED: a duplicate VISIBILITY control was added and the gate stayed "
              "green — the gate only knows about contrast, so it is a hard-coded assertion, not "
              "a gate.")
        return 1
    if not any("visibility" in f and f.startswith("FAIL") for f in findings_vis):
        print("\nSELF-TEST FAILED: the gate failed, but not on visibility — it did not actually "
              "detect the duplicate it was given.")
        return 1
    print("\n    the gate bit on a concern it was never specifically taught. It generalises.")

    # A probe that cannot vary is invisible to everything above (its concern reports "at most 0
    # surfaces" while measuring nothing), so the mechanism that catches it is mutation-checked too.
    print("=" * 100)
    print("SELF-TEST 4/4: freezing a probe's value to a CONSTANT (the _probe_fov shape) —")
    print("               the gate must NAME the probe, not report its concern as clean.")
    try:
        ok_frozen, findings_frozen = gate_no_duplicated_controllers(
            dataset, mutate=_freeze_a_probe)
    finally:
        _unfreeze_a_probe()
    for f in findings_frozen:
        if f.startswith("FAIL"):
            print("   ", f)
    if ok_frozen:
        print("\nSELF-TEST FAILED: a probe was frozen to a constant and the gate stayed green — "
              "its concern reported 'at most 0 surfaces' over a value that cannot move, which is "
              "exactly what _probe_fov did for six weeks.")
        return 1
    if not any("_probe_current_well" in f and f.startswith("FAIL") for f in findings_frozen):
        print("\nSELF-TEST FAILED: the gate failed, but did not name the frozen probe.")
        return 1
    print("\n    the gate named the dead probe instead of passing its concern.")

    print("=" * 100)
    print("SELF-TEST: mutation removed; confirming the gate is green again")
    ok_back, _ = gate_no_duplicated_controllers(dataset)
    if not ok_back:
        print("SELF-TEST FAILED: the gate did not recover after the mutations were removed.")
        return 1

    print("\nSELF-TEST PASSED: the gate passes clean, fails on a reintroduced contrast duplicate, "
          "fails on an unrelated duplicate, and recovers.")
    return self_test_dead_controls(dataset)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=PLATE)
    ap.add_argument("--self-test", action="store_true",
                    help="mutation-check the gate: reintroduce a duplicate and require a failure")
    ap.add_argument("--inventory", action="store_true",
                    help="print GATE 3's control inventory: every control of a real plate and a "
                         "real region window, with a verdict and the evidence for each")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.dataset):
        print(f"SKIP  dataset absent on this machine, cannot run: {args.dataset}")
        if args.dataset == PLATE:
            print('      rebuild it with: python tools/make_5d_fixture.py '
                  f'"{PLATE}" --fovs 36 --nz 1 --nt 1 --well-pitch-mm 9.0 '
                  '--declared-format "384 well plate"')
        return 0
    if args.self_test:
        return self_test(args.dataset)
    if args.inventory:
        try:
            ok3, findings3, rows = gate_no_dead_controls(args.dataset, verbose=args.verbose)
        except Exception:
            traceback.print_exc()
            return 1
        print_inventory(rows)
        print("\n" + "=" * 100)
        for f in findings3:
            print(f)
        return 0 if ok3 else 1

    print("=" * 100)
    print("GATE 2 (IMA-268): exactly one control surface per concern")
    print("=" * 100)
    try:
        ok, findings = gate_no_duplicated_controllers(args.dataset, verbose=args.verbose)
    except Exception:
        traceback.print_exc()
        return 1
    for f in findings:
        print(f)
    print("=" * 100)
    print("GATE 2: PASS" if ok else "GATE 2: FAIL")

    print("=" * 100)
    print("GATE 3: no dead controls — clicked, not called")
    print("=" * 100)
    try:
        ok3, findings3, rows = gate_no_dead_controls(args.dataset, verbose=args.verbose)
    except Exception:
        traceback.print_exc()
        return 1
    if args.verbose:
        print_inventory(rows)
    for f in findings3:
        print(f)
    print("=" * 100)
    print("GATE 3: PASS" if ok3 else "GATE 3: FAIL")
    return 0 if (ok and ok3) else 1


if __name__ == "__main__":
    rc = main()
    # os._exit, not sys.exit: Qt can SIGSEGV unwinding at interpreter shutdown, which would
    # otherwise decide the exit code instead of the gate's own verdict.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
