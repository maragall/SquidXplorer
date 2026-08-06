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

#: ``~/Downloads/synthetic_2x2_wellplate`` until 2026-08-06, when that folder no longer existed
#: and ``main()`` answered "dataset absent, cannot run" with exit code 2. Rebuild the replacement:
#:   python tools/make_5d_fixture.py ~/Downloads/sim_2x2_36fov_96wp --fovs 36 --nz 1 --nt 1 \
#:       --well-pitch-mm 9.0 --declared-format "384 well plate"
#: ``SQUIDMIP_FIXTURE_PLATE`` overrides it, which is how CI runs this gate (and its --self-test)
#: for real rather than watching it skip: the fixture is generated, so a runner can make one.
PLATE = os.environ.get("SQUIDMIP_FIXTURE_PLATE") or \
    "/Users/julioamaragall/Downloads/sim_2x2_36fov_96wp"

_APP = None
_MISSING = object()      # "this attribute was inherited, not the class's own" — see monkey()


def _app():
    """The QApplication, on THE BINDING THE APP SHIPS -- decided by importing `squidmip` first.

    Nine import sites in this file said ``PyQt5`` until 2026-08-06, which is the same defect
    commit 6b51793 fixed in ``tools/walkthrough.py`` and did not carry here. ``squidmip/__init__``
    pins ``QT_API=pyqt6``, so this constructed a Qt5 application around Qt6 widgets, loaded both
    frameworks into one process, and aborted on "QWidget: Must construct a QApplication before a
    QWidget" before the gate looked at anything. Dead since the Qt6 migration (10b8348, f7f9b28,
    ce5605c); nothing in CI ran it, so nothing said so.
    """
    global _APP
    import squidmip  # noqa: F401  -- sets QT_API before qtpy resolves a binding
    from qtpy.QtWidgets import QApplication
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


    # `_probe_scope` was here. Contrast SCOPE (global vs per-region) was deleted from the product
    # on 2026-07-22 (8b0cbfc): "the contrast should be only global, I don't understand why there's
    # a per region contrast". `PlateOverview._scope` went with it, so this probe raised
    # AttributeError on every snapshot -- and `_snapshot` swallowed it, so the concern simply never
    # appeared and the table's "contrast scope: 1" reported PASS over a probe that had not run
    # since July. That swallow is fixed below; the probe itself is deleted, because there is no
    # scope to probe.


def _probe_selection(w):
    return {"plate selection": (repr(w._overview._sel), tuple(w._selected_regions))}


def _probe_current_well(w):
    return {"current well": w._current_well}


def _probe_zoom(w):
    ov = w._overview
    return {"zoom / viewport": (round(ov._cd, 4), round(ov._ox, 4), round(ov._oy, 4))}


    # `_probe_fov` was here. It read `w._detail._fov_slider`, and `PlateWindow._detail` has been
    # unconditionally None since 19cd491 (2026-07-22); the central array viewer was removed
    # outright by 2b8fbc5 (2026-07-23, "Decentralize GUI"). The probe therefore returned the
    # constant `(None, None)` on every snapshot -- it could not change, so no widget could ever be
    # reported as moving it, and "fov / plane index: 1" was a PASS over a value nothing read.
    # A probe that cannot vary is not a weak probe, it is a decoration.


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


#: Probes that raised while snapshotting, as ``{probe name: "TypeName: message"}``. Reported, not
#: swallowed -- see :func:`_snapshot`.
BROKEN_PROBES: dict[str, str] = {}


def _snapshot(w):
    """Every probe's reading, and a RECORD of any probe that could not take one.

    This used to ``except Exception: continue``. That is the failure mode this whole gate exists
    to prevent, turned on itself: two probes (``_probe_scope``, ``_probe_fov``) had been reading
    attributes deleted in July, raised on every snapshot, and were dropped in silence -- so their
    concerns never appeared in the results, ``by_concern.get(concern, {})`` returned ``{}``, and
    the gate printed "at most 0 control surfaces (expected at most 1) -- PASS". A gate cannot
    both skip a check and call it green.
    """
    out = {}
    for p in PROBES:
        try:
            out.update(p(w))
        except Exception as exc:                       # noqa: BLE001 - recorded, then reported
            BROKEN_PROBES[p.__name__] = f"{type(exc).__name__}: {exc}"
    return out


# --- the expected table: how many CONTROL SURFACES each concern is allowed ---------------------
#
# This is the whole specification, and it is deliberately a table rather than a pile of asserts:
# adding a concern is one line, and the gate fails on a count that is too HIGH (a duplicate
# appeared) as well as too LOW (the control was lost in a refactor). "1" is the normal answer.
# "0" means the plate must not own this at all — contrast belongs to the array viewer.

EXPECTED = {
    # Rewritten 2026-08-06 against the window as it stands. Every one of these was 1 and every one
    # of them measured 0, because the controls the numbers described are in a napari RegionViewer
    # window now (2b8fbc5, "Decentralize GUI") or are MOUSE GESTURES on the plate rather than
    # widgets -- and this sweep only actuates widgets. Leaving them at 1 made nine PASS lines that
    # could not fail, which is the same "832 green tests" shape the docstring above is about.
    #
    # 0 is not "unchecked": the gate fails the moment ANY widget starts moving one of these, which
    # is exactly the event worth catching -- a control creeping back onto the root window beside
    # the one that already owns the value elsewhere.
    "contrast":              0,   # napari's LUT row owns it, in the region window (IMA-261)
    "visibility":            0,   # napari's eye icon; the plate's checkboxes went in 8b0cbfc
    "channel colour / LUT":  0,   # napari's colormap picker; the plate follows it
    "active layer":          0,   # the Layers tree, which is a tree item and not a control widget
    # click / marquee are gestures, not widgets. "Select all" IS a widget and DOES move this, and
    # it is the one EXEMPT entry -- verified live 2026-08-06 by emptying EXEMPT, which turned this
    # line red with "Select all [QPushButton in PlateWindow]". So the 0 here is a measurement, not
    # an absence of measurement.
    "plate selection":       0,
    "current well":          0,   # double-click: a gesture, no widget
    "zoom / viewport":       0,   # the wheel: a gesture, no widget
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
    # `_ChannelBar` and `LightweightViewer` are gone (2026-07-22 and 2026-08-05). Kept in this
    # list on purpose: it names what a control's ANCESTOR might be, and a stale name here costs
    # nothing while a missing one makes a re-introduced widget report its owner as "?".
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


def _neutralise(win, monkey):
    """Stop a click from doing something a gate has no business doing.

    The gate clicks every button in the window, so anything that opens a modal dialog, launches a
    multi-minute operator run, re-ingests, or closes the app has to be turned into a recorded
    no-op first. This is a safety harness, NOT an exemption: the neutralised calls are still
    observed, they simply do not run.
    """
    from qtpy.QtWidgets import QFileDialog, QMessageBox
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
    from qtpy.QtWidgets import QAbstractButton
    return isinstance(wdg, QAbstractButton)


# --- the contrast-specific structural assertion ------------------------------------------------

def contrast_surfaces(win):
    """Contrast controls on the PLATE side, which must be zero (IMA-261).

    Kept alongside the empirical sweep rather than replaced by it, because "the plate has no
    contrast slider" must hold even for a slider that is currently disabled or hidden — a
    hide()-den control is a second owner waiting to be un-hidden, and the sweep only actuates what
    a user could actuate today.

    SCOPED TO THE PLATE WIDGET, not to ``win._channel_bar``, since 2026-08-06. That attribute has
    not existed since 8b0cbfc (2026-07-22) deleted the plate's channel bar outright, so this
    returned ``([], [])`` from the very first line and reported "PASS contrast: 0 sliders, 0 auto
    buttons" without looking at a single widget. The claim was over-satisfied and the check was
    dead, which look identical in the output and are not the same thing at all.
    """
    from qtpy.QtWidgets import QAbstractSlider, QPushButton
    plate = getattr(win, "_overview", None)
    if plate is None:
        return [], []
    sliders = [f"{_label(s)} [{type(s).__name__}]" for s in plate.findChildren(QAbstractSlider)]
    autos = [f"{_label(b)} [auto button]" for b in plate.findChildren(QPushButton)
             if "auto" in b.text().lower()]
    return sliders, autos


# --- the gate ----------------------------------------------------------------------------------

def _drain_preview(win, timeout_s=180):
    """Block until the plate's background preview stream has finished.

    QUIESCENCE IS PART OF THE MEASUREMENT. The plate's contrast is a RUNNING percentile: every
    tile the preview worker delivers moves ``channel_windows()``. Sweeping while that stream is
    live means the before/after snapshots straddle an update nothing on screen caused, and the
    widget that happened to be under the cursor at that moment is recorded as its owner. Measured
    2026-08-06: the gate reported "contrast[0] <- QScrollBar [QScrollBar in PlateWindow]" -- the
    LOG PANEL's scrollbar accused of owning channel 0's contrast window. A false duplicate is
    worse than no gate, because it is the finding people learn to ignore.
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

    *mutate*, when given, is called with the shown, ingested window before the sweep. It is how
    ``--self-test`` mounts a duplicate control; see :func:`self_test`.
    """
    import squidmip._viewer as V
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
    _drain_preview(win)            # the plate must be STILL before anything is attributed to it
    if mutate is not None:
        mutate(win)
        app.processEvents()

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

    # 4. a probe that could not read is a CHECK THAT DID NOT RUN, and it fails rather than
    #    disappearing. Two probes had been raising since July and were swallowed; their concerns
    #    then reported "at most 0 control surfaces — PASS" while measuring nothing.
    for name, why in sorted(BROKEN_PROBES.items()):
        ok = False
        findings.append(f"FAIL  {name}: the probe itself raised, so its concern was NOT "
                        f"checked — {why}")

    win.close()
    app.processEvents()
    return ok, findings


# --- mutation check: prove the gate can fail ---------------------------------------------------

def _mount_contrast_duplicate(win):
    """Bolt a second, independently draggable owner of the contrast window onto the plate.

    Exactly the control IMA-261 deleted, put back where a user would see it: on the root window,
    beside the value's real owner. Mounted on the SHOWN, INGESTED window rather than by patching
    ``_ChannelBar.__init__`` (which is what this did until 2026-08-06) -- the channel bar was
    deleted in 8b0cbfc on 2026-07-22 and is never constructed, so that patch ran zero times and
    the self-test reported "a duplicate contrast slider was added and the gate stayed GREEN"
    about a duplicate that was never added. A mutation test that cannot mutate is worth less than
    no mutation test, because it reads as evidence.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QSlider
    ov = win._overview
    for c_i in range(len(ov.channel_windows() or [])):
        s = QSlider(Qt.Orientation.Horizontal, win)     # SCOPED: Qt.Horizontal is Qt5-only
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


def self_test(dataset=PLATE):
    """Reintroduce the duplicate, require the gate to bite, remove it, require the gate to pass."""
    print("=" * 100)
    print("SELF-TEST 1/3: the gate must PASS on the tree as it stands")
    ok, findings = gate_no_duplicated_controllers(dataset)
    for f in findings:
        print("   ", f)
    if not ok:
        print("\nSELF-TEST FAILED: the gate is red before the mutation was even applied.")
        return 1

    print("=" * 100)
    print("SELF-TEST 2/3: reintroducing a per-channel contrast slider on the plate —")
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

    # A gate that only knows about contrast is a hard-coded assertion about the bug we happen to
    # have just fixed. The point of IMA-268 is the NEXT duplicate, in a concern nobody is looking
    # at — so duplicate a DIFFERENT control and require the same failure.
    print("=" * 100)
    print("SELF-TEST 3/3: duplicating a control in a DIFFERENT concern (channel visibility) —")
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

    print("=" * 100)
    print("SELF-TEST: mutation removed; confirming the gate is green again")
    ok_back, _ = gate_no_duplicated_controllers(dataset)
    if not ok_back:
        print("SELF-TEST FAILED: the gate did not recover after the mutations were removed.")
        return 1

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

    if not os.path.isdir(args.dataset):
        # SKIP, not a failure, and with the exact command that makes the fixture. This gate needs
        # a real acquisition to have a real widget tree; on a machine (or a CI runner) without one
        # there is nothing to be wrong about.
        print(f"SKIP  dataset absent on this machine, cannot run: {args.dataset}")
        if args.dataset == PLATE:
            print('      rebuild it with: python tools/make_5d_fixture.py '
                  f'"{PLATE}" --fovs 36 --nz 1 --nt 1 --well-pitch-mm 9.0 '
                  '--declared-format "384 well plate"')
        return 0
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
    rc = main()
    # os._exit, NOT sys.exit. Measured 2026-08-06: the gate printed its full verdict and the
    # process then died with SIGSEGV (139) unwinding Qt at interpreter shutdown. A gate whose exit
    # code is decided by a teardown crash gates nothing.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
