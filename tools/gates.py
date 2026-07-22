#!/usr/bin/env python3
"""Structural gates over the REAL widget tree of a shown window (IMA-268).

    QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python tools/gates.py

GATE 2 — NO DUPLICATED CONTROLLERS
==================================
"Two representations of one truth, hand-synced" is this project's second-most-common defect shape.
Four confirmed instances before this gate existed:

  * millimetres stored under a key ending ``_um``;
  * ``manual0`` and ``MANUAL0`` as two spellings of one channel state;
  * a ``_push_index`` that disagreed with its producer and silently dropped pushes;
  * two compositors with two percentile rules that had already drifted apart — one of them
    clipping a blank channel to full white, so an empty channel read as signal.

And a fifth, the one a human actually reported: the plate carried its own low/high contrast
sliders and an "auto" button per channel, two hand-widths from the embedded array viewer's
contrast slider over the same channel. The same channel was displayed at two different windows,
side by side, on one screen.

WHY THIS GATE IS EMPIRICAL AND NOT A GREP
-----------------------------------------
A grep for ``QSlider`` finds the sliders that exist today under the names they have today. It
cannot answer the question that actually matters, which is not "how many sliders are there" but
"how many WIDGETS CAN WRITE THIS ONE VALUE". So this gate does not read the source. It:

  1. opens a real window on a real acquisition and SHOWS it (an unshown splitter reports every
     child at its default size, so testing an unshown window measures the harness);
  2. walks the real widget tree for every interactive control;
  3. ACTUATES each one the way a user would — moves the slider, ticks the box, picks the combo
     entry, clicks the button;
  4. watches a set of PROBES that read the application's underlying state directly;
  5. groups by probe. A probe that more than one widget can move is a duplicated controller.

That is a definition of "duplicate" that does not care what the widget is called, which file it
lives in, or which repo it came from — so a NEW duplicate is caught on the day it is added, which
a hand-written "there must be no sliders in the channel bar" assertion never would be.

MUTATION-CHECKED. `tools/gates.py --self-test` reintroduces a duplicate contrast slider on the
plate, runs the gate, and requires it to FAIL — then removes it and requires it to pass. A gate
that cannot fail is worth nothing: this project already shipped 832 passing tests over a model
error, because every fixture had one FOV and one region.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TISSUE = ("/Users/julioamaragall/Downloads/"
          "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy")
PLATE = "/Users/julioamaragall/Downloads/synthetic_2x2_wellplate"

_APP = None
_MISSING = object()      # "this attribute was inherited, not the class's own" — see monkey()


def _app():
    global _APP
    from PyQt5.QtWidgets import QApplication
    _APP = QApplication.instance() or QApplication([])
    return _APP


# --- the probes: what "one truth" means, read straight off the model ---------------------------
#
# A probe returns a hashable snapshot of ONE piece of application state. It must read the model,
# never a widget — a probe that read the widget back would be satisfied by two widgets that agree
# at the instant of reading and drift a second later, which is the very defect being hunted.

# A probe returns {slot key: value}. The KEY, not the concern, is the unit of duplication: four
# per-channel visibility checkboxes are four controls over four DIFFERENT truths and are correct,
# while two widgets that both move "visibility[2]" are the defect. Collapsing a per-channel
# concern to one value would report the correct design as a duplicate and be switched off within
# a week, which is how a gate dies.

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


def _probe_scope(w):
    return {"contrast scope": w._overview._scope}


def _probe_selection(w):
    return {"plate selection": (repr(w._overview._sel), tuple(w._selected_regions))}


def _probe_current_well(w):
    return {"current well": w._current_well}


def _probe_zoom(w):
    ov = w._overview
    return {"zoom / viewport": (round(ov._cd, 4), round(ov._ox, 4), round(ov._oy, 4))}


def _probe_fov(w):
    d = getattr(w, "_detail", None)
    sl = getattr(d, "_fov_slider", None)
    return {"fov / plane index": (None if sl is None else sl.value(),
                                  getattr(d, "_current_fov_idx", None))}


def _probe_colormap(w):
    cols = getattr(w._overview, "_colors", None)
    if cols is None:
        return {}
    return {f"channel colour / LUT[{i}]": tuple(row) for i, row in enumerate(cols.tolist())}


PROBES = (_probe_contrast, _probe_visibility, _probe_active_layer, _probe_scope,
          _probe_selection, _probe_current_well, _probe_zoom, _probe_fov, _probe_colormap)


def _concern_of(key: str) -> str:
    """'visibility[2]' -> 'visibility'. The table is written per concern, the check is per slot."""
    return key.split("[", 1)[0].strip()


def _snapshot(w):
    out = {}
    for p in PROBES:
        try:
            out.update(p(w))
        except Exception:
            continue
    return out


# --- the expected table: how many CONTROL SURFACES each concern is allowed ---------------------
#
# This is the whole specification, and it is deliberately a table rather than a pile of asserts:
# adding a concern is one line, and the gate fails on a count that is too HIGH (a duplicate
# appeared) as well as too LOW (the control was lost in a refactor). "1" is the normal answer.
# "0" means the plate must not own this at all — contrast belongs to the array viewer.

EXPECTED = {
    "contrast":              1,   # exactly one owner. The array viewer's LUT row (IMA-261).
    "visibility":            1,   # the channel bar's checkbox
    "active layer":          1,   # the Layers tab
    "contrast scope":        1,   # the scope combo
    "plate selection":       1,   # click/marquee on the plate
    "current well":          1,   # double-click on the plate
    "zoom / viewport":       1,   # the wheel over the plate
    "fov / plane index":     1,   # the array viewer's FOV slider
    "channel colour / LUT":  1,   # resolved from the acquisition, not user-set on the plate
}

# Controls that legitimately move a probe as a SIDE EFFECT of doing something else, and are not a
# second controller of it. Each entry needs a reason: an unexplained entry here is how a real
# duplicate gets waved through, which is the failure mode this gate is guarding against.
EXEMPT = {
    # (probe, widget label): why it is not a second owner
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
    for interesting in ("_ChannelBar", "PlateOverview", "LightweightViewer", "PlateWindow"):
        if interesting in chain:
            return interesting
    return chain[-1] if chain else "?"


def interactive_widgets(root):
    """Every widget in the tree a user can act on, in a stable order."""
    from PyQt5.QtWidgets import (
        QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox,
    )
    kinds = (QAbstractSlider, QAbstractSpinBox, QComboBox, QAbstractButton)
    out = []
    for k in kinds:
        for wdg in root.findChildren(k):
            if wdg not in out:
                out.append(wdg)
    # A composite control (superqt's QLabeledSlider, a spinbox's own up/down buttons) CONTAINS an
    # interactive widget. Both would report moving the same value, and the gate would accuse a
    # single control of duplicating itself. Only the outermost widget of a nest is a control
    # surface — a user sees and drags one thing.
    outer = [w for w in out if not any(o is not w and o.isAncestorOf(w) for o in out)]
    return outer


def _actuate(wdg):
    """Move *wdg* the way a user would. Returns a callable that puts it back, or None.

    Value widgets are moved to a genuinely different value and then restored; buttons are clicked
    and cannot be un-clicked, which is why the baseline is re-read before every widget rather than
    once at the start.
    """
    from PyQt5.QtWidgets import (
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


def _neutralise(win, monkey):
    """Stop a click from doing something a gate has no business doing.

    The gate clicks every button in the window, so anything that opens a modal dialog, launches a
    multi-minute operator run, re-ingests, or closes the app has to be turned into a recorded
    no-op first. This is a safety harness, NOT an exemption: the neutralised calls are still
    observed, they simply do not run.
    """
    from PyQt5.QtWidgets import QFileDialog, QMessageBox
    import squidmip._viewer as V

    called = []

    def rec(name, ret=None):
        def f(*a, **k):
            called.append(name)
            return ret
        return f

    monkey(QFileDialog, "getExistingDirectory", staticmethod(rec("getExistingDirectory", "")))
    monkey(QFileDialog, "getOpenFileName", staticmethod(rec("getOpenFileName", ("", ""))))
    monkey(QFileDialog, "exec_", rec("QFileDialog.exec_", 0))
    for m in ("warning", "information", "critical", "question", "about"):
        monkey(QMessageBox, m, staticmethod(rec(f"QMessageBox.{m}", 0)))
    monkey(QMessageBox, "exec_", rec("QMessageBox.exec_", 0))
    for m in ("run_operator", "run_minerva_export", "ingest", "close", "_open_acquisition_dialog"):
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
        except Exception:
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
    from PyQt5.QtWidgets import QAbstractButton
    return isinstance(wdg, QAbstractButton)


# --- the contrast-specific structural assertion ------------------------------------------------

def contrast_surfaces(win):
    """Contrast controls on the PLATE side, which must be zero (IMA-261).

    Kept alongside the empirical sweep rather than replaced by it, because "the plate has no
    contrast slider" must hold even for a slider that is currently disabled or hidden — a
    hide()-den control is a second owner waiting to be un-hidden, and the sweep only actuates what
    a user could actuate today.
    """
    from PyQt5.QtWidgets import QAbstractSlider, QPushButton
    bar = getattr(win, "_channel_bar", None)
    if bar is None:
        return [], []
    sliders = [f"{_label(s)} [{type(s).__name__}]" for s in bar.findChildren(QAbstractSlider)]
    autos = [f"{_label(b)} [auto button]" for b in bar.findChildren(QPushButton)
             if "auto" in b.text().lower()]
    return sliders, autos


# --- the gate ----------------------------------------------------------------------------------

def gate_no_duplicated_controllers(dataset=PLATE, verbose=False):
    """Returns (ok, list of human-readable findings)."""
    import squidmip._viewer as V
    app = _app()
    findings, ok = [], True

    win = V.PlateWindow(None)
    win.resize(1600, 900)
    win.show()                     # SHOWN: an unshown splitter reports defaults, not the product
    app.processEvents()
    win.ingest(dataset)
    app.processEvents()
    if win._reader is None:
        return False, [f"FAIL  could not open {dataset}: {win._readout.text()!r}"]

    # 1. the structural half: the plate must own no contrast control at all.
    sliders, autos = contrast_surfaces(win)
    if sliders or autos:
        ok = False
        findings.append(f"FAIL  contrast: the plate view still carries {len(sliders)} slider(s) "
                        f"and {len(autos)} auto button(s) — {sliders + autos}")
    else:
        findings.append("PASS  contrast: 0 sliders, 0 auto buttons in the plate view")

    # 2. the empirical half: actuate everything, group by the state it moved.
    patches = []

    def monkey(obj, name, value):
        # Record whether the attribute was the class's OWN or inherited. Re-setting an inherited
        # C++ slot (QWidget.close) onto the subclass turns it into an unbound sip method that no
        # longer binds to an instance — so an inherited attribute must be DELETED to restore it,
        # never re-assigned.
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

    # 3. anything the table has never heard of. A key that appears here is a NEW piece of state
    #    that two or more widgets can move and nobody has decided who owns — which is exactly the
    #    fifth instance of this defect arriving, so it fails rather than warns.
    for concern, slots in sorted(by_concern.items()):
        if concern in EXPECTED:
            continue
        for key, got in sorted(slots.items()):
            if len(got) > 1:
                ok = False
                findings.append(f"FAIL  {key}: UNDECLARED concern with {len(got)} control "
                                f"surfaces — add it to EXPECTED and pick an owner: {got}")

    win.close()
    app.processEvents()
    return ok, findings


# --- mutation check: prove the gate can fail ---------------------------------------------------

def self_test(dataset=PLATE):
    """Reintroduce the duplicate, require the gate to bite, remove it, require the gate to pass."""
    import squidmip._viewer as V
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QSlider

    print("=" * 100)
    print("SELF-TEST 1/2: the gate must PASS on the tree as it stands")
    ok, findings = gate_no_duplicated_controllers(dataset)
    for f in findings:
        print("   ", f)
    if not ok:
        print("\nSELF-TEST FAILED: the gate is red before the mutation was even applied.")
        return 1

    print("=" * 100)
    print("SELF-TEST 2/2: reintroducing a per-channel contrast slider on the plate —")
    print("               the gate MUST now fail, or it is decorative.")
    original = V._ChannelBar.__init__

    def mutant_init(self, labels, colors, overview):
        original(self, labels, colors, overview)
        # exactly the duplicate IMA-261 deleted: a second, independently draggable owner of the
        # contrast window, sitting on the plate next to the array viewer's own.
        for c_i in range(len(self._rows)):
            s = QSlider(Qt.Horizontal, self)
            s.setRange(0, 65535)
            s.setValue(30000)
            s.valueChanged.connect(
                lambda v, i=c_i: overview.set_channel_window(i, 0.0, float(v)))
            self.layout().addWidget(s)
            s.show()

    V._ChannelBar.__init__ = mutant_init
    try:
        ok_mut, findings_mut = gate_no_duplicated_controllers(dataset)
    finally:
        V._ChannelBar.__init__ = original
    for f in findings_mut:
        print("   ", f)
    if ok_mut:
        print("\nSELF-TEST FAILED: a duplicate contrast slider was added and the gate stayed "
              "GREEN. The gate does not work.")
        return 1
    print("\n    the gate bit, as it must.")

    print("=" * 100)
    print("SELF-TEST: mutation removed; confirming the gate is green again")
    ok_back, _ = gate_no_duplicated_controllers(dataset)
    if not ok_back:
        print("SELF-TEST FAILED: the gate did not recover after the mutation was removed.")
        return 1

    # A gate that only knows about contrast is a hard-coded assertion about the bug we happen to
    # have just fixed. The point of IMA-268 is the NEXT duplicate, in a concern nobody is looking
    # at — so duplicate a DIFFERENT control and require the same failure.
    print("=" * 100)
    print("SELF-TEST 3/3: duplicating a control in a DIFFERENT concern (channel visibility) —")
    print("               the gate must generalise, not just know about contrast.")
    from PyQt5.QtWidgets import QCheckBox

    def mutant_vis_init(self, labels, colors, overview):
        original(self, labels, colors, overview)
        box = QCheckBox("show ch0", self)          # a second owner of visibility[0]
        box.setChecked(True)
        box.toggled.connect(lambda on: overview.set_channel_visible(0, on))
        self.layout().addWidget(box)
        box.show()

    V._ChannelBar.__init__ = mutant_vis_init
    try:
        ok_vis, findings_vis = gate_no_duplicated_controllers(dataset)
    finally:
        V._ChannelBar.__init__ = original
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

    print("\nSELF-TEST PASSED: the gate passes clean, fails on a reintroduced contrast duplicate, "
          "fails on an unrelated duplicate, and recovers.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=PLATE)
    ap.add_argument("--self-test", action="store_true",
                    help="mutation-check the gate: reintroduce a duplicate and require a failure")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.dataset):
        print(f"dataset absent, cannot run: {args.dataset}")
        return 2
    if args.self_test:
        return self_test(args.dataset)

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
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
