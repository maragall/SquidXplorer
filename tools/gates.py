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

GATE 3 — NO DEAD CONTROLS
=========================
GATE 2 asks "does more than one widget move this value". GATE 3 asks the complementary and, on the
evidence, more expensive question: **does this widget move ANYTHING at all.** Every defect in the
table below shipped past a green unit suite because every test called the handler instead of
clicking the thing:

  * ``_on_detect_nuclei`` — the "run Cellpose" handler's only entry point was a button on a pane
    deleted in July, so it was never called at all;
  * ``run_operator`` opened its preview tab into a pane kept hidden — the decon QC picture was
    published to nobody for six weeks;
  * the plate's timepoint bar called ``self._say``, which does not exist on ``PlateWindow``;
  * this gate's own ``--self-test`` mutated a ``_ChannelBar`` the window had stopped constructing,
    so it ran zero mutations and printed PASS.

So: open a real plate AND a real region window, take the full inventory of the controls a user can
actually reach, actuate each one, and require SOME observable outcome — a changed status line, a
changed widget anywhere in the app, a tab, a window, a napari layer, a log record, a call into a
neutralised entry point, or changed pixels. A control with none of those is reported by name.

``--inventory`` prints the whole table with a verdict per control and is the deliverable a human
reads; the gate is the same sweep with a pass/fail on the end. Mutation-checked the same way GATE
2 is: ``--self-test`` bolts a button wired to nothing onto each window and requires the gate to
name it.
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


def _recorder(called, detail=None):
    """``rec(name)`` -> a callable that records and returns *ret*, appending into *called*.

    *detail*, when given, also receives ``(name, args, kwargs)`` — which is what lets GATE 3 ask
    the question that matters for an input control: did the value the user typed ARRIVE at the
    call. That is the shape of the ``operator_kwargs`` defect (57 labels vs 44).
    """
    def rec(name, ret=None):
        def f(*a, **k):
            called.append(name)
            if detail is not None:
                detail.append((name, a, k))
            return ret
        return f
    return rec


#: Entry points that must be stubbed BEFORE ``PlateWindow`` is constructed, not after it is shown.
#: ``ViewerManager`` is built inside ``PlateWindow.__init__`` and captures ``win.run_operator`` as a
#: BOUND METHOD, which every region window then holds as ``self._run_operator``. Patching the class
#: afterwards leaves that bound method pointing at the real one — measured 2026-08-06: the region
#: window's ``Run`` chip started a genuine operator run in the middle of the sweep, and its result
#: landed while the NEXT control ("save") was being measured, so a checkbox was credited with
#: adding a napari layer.
_EARLY_STUBS = ("run_operator", "run_minerva_export")


def _neutralise_early(monkey, called, detail=None):
    """The stubs that have to be in place before the plate window exists. See :data:`_EARLY_STUBS`."""
    import squidmip._viewer as V

    rec = _recorder(called, detail)
    for m in _EARLY_STUBS:
        if hasattr(V.PlateWindow, m):
            monkey(V.PlateWindow, m, rec(f"PlateWindow.{m}"))


def _neutralise(win, monkey, called=None):
    """Stop a click from doing something a gate has no business doing.

    The gate clicks every button in the window, so anything that opens a modal dialog, launches a
    multi-minute operator run, re-ingests, or closes the app has to be turned into a recorded
    no-op first. This is a safety harness, NOT an exemption: the neutralised calls are still
    observed, they simply do not run.
    """
    from qtpy.QtWidgets import QFileDialog, QMessageBox
    import squidmip._viewer as V

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


# ==================================================================================================
# GATE 3 — NO DEAD CONTROLS.  Clicked, not called.
# ==================================================================================================

#: Widget classes defined in these packages are somebody else's controls. napari's dims play
#: buttons, superqt's labelled sliders and Qt's own scroll bars are all inside our windows and none
#: of them is ours to declare alive or dead — a napari button that does nothing in an offscreen
#: canvas is a napari fact, and reporting it here would bury the ones that are ours under noise.
_THIRD_PARTY = ("napari", "superqt", "vispy", "qtpy", "PyQt", "PySide", "qtconsole")


def _third_party(wdg) -> bool:
    """True when *wdg* — or any widget it sits inside — belongs to a library rather than to us."""
    node = wdg
    for _ in range(20):
        if node is None:
            return False
        mod = type(node).__module__ or ""
        if mod.startswith(_THIRD_PARTY):
            # A plain QPushButton is `PyQt6.QtWidgets`, and refusing those would empty the sweep.
            # Only a SUBCLASS defined in a library, or a library CONTAINER, disqualifies.
            if not mod.startswith(("PyQt", "PySide", "qtpy")):
                return True
        node = node.parent()
    return False


def our_controls(root):
    """The inventory: every control in *root* that is OURS, with hidden/disabled ones kept.

    Hidden and disabled are kept deliberately and reported as their own verdicts. "The control is
    not on screen in this state" is an answer a human needs — it is how ``_on_detect_nuclei``'s
    button vanished — and dropping those rows would turn an absence into a silence.
    """
    return [w for w in interactive_widgets(root) if not _third_party(w)]


class _LogSpy:
    """Every log record OURS emitted while a control is being actuated.

    A handler is not a probe of state, it is a probe of ACTIVITY — and half this app's controls
    report by logging (``RegionViewer._say`` goes to the shared logger before it goes to the pane).

    SCOPED TO ``squid.xplorer``, and that scope is the difference between evidence and noise.
    Offscreen Qt routes its own platform warnings through logging ("This plugin does not support
    raise()", "Cannot open file theme_dark:/…"), one or more per click, so an unscoped spy would
    report EVERY control as having done something — including a chip wired to nothing, which is
    the exact thing this gate exists to catch. Measured: ``▣ plate`` and ``⚙ controls`` both listed
    a Qt warning as their only evidence before this was scoped.
    """

    def __init__(self) -> None:
        import logging

        from squidmip._logpane import XPLORER_ROOT

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
        # OUR logger, not the stdlib root: the level check happens on the logger that EMITS, so
        # lowering the root's level would not have let a single INFO from `squid.xplorer.*` through.
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
    """Every piece of text a user can read in *root*, keyed stably.

    Keyed by (class, ordinal) rather than by object identity: a click that REPLACES a label still
    has to be visible as a change, and identity keys would make the old and new label two
    different, both-unchanged entries.
    """
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
    """The napari model under a region window: what the pixels ARE.

    This is the reading that makes the gate about PIXELS rather than about widgets. It goes
    through ``MosaicLayers``, so a control that adds a layer group, hides one, moves a contrast
    window or swaps a colormap is visible here whatever widget it used to do it.
    """
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
    """A hash of what the window draws, with the CONTROL ITSELF painted out.

    *blank* is the widget being actuated. Its own rectangle is excluded, and that exclusion is
    load-bearing rather than tidy: a button repaints itself when it is clicked (focus ring, hover,
    the native style's pressed state), and that repaint is the ACTUATION, not an outcome. Measured
    2026-08-06 — the gate's own mutation, a ``QPushButton`` connected to nothing, was reported as
    "reaches: pixels" and the self-test failed to catch its own dead chip. Exactly the same rule
    already applied to the control's own entry in :func:`_widget_states`, one level down.

    A 4-pixel pad, because a focus ring is drawn outside the widget's geometry.
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
    """Stop every REPEATING QTimer under *root*, and say how many.

    A window with a periodic repaint has no stable pixel state, and this app has two: the
    navigator's memory poll (2 s) and the log panel's. Measured 2026-08-06: the gate's own
    mutation — a chip wired to nothing — was credited with "reaches: pixels" INTERMITTENTLY,
    because a memory bar happened to tick inside its ``processEvents``. The self-test passed on
    one run and failed on the next over the same code, which is worse than a gate that never
    works, because it teaches people to re-run it.

    Stopped at the source rather than tolerated by a weaker probe: a periodic repaint is not
    something any control does, so nothing under test is lost.
    """
    from qtpy.QtCore import QTimer

    stopped = 0
    for timer in root.findChildren(QTimer):
        if timer.isActive() and not timer.isSingleShot():
            timer.stop()
            stopped += 1
    return stopped


def _pixel_noise(root, app, n=6, seconds=1.5) -> bool:
    """Do IDLE grabs of *root* differ, ACROSS REAL TIME? If so, pixels are not evidence.

    A caret, a spinner or a queued repaint would make every control look alive, and a probe that
    always fires turns a gate green over anything. Measuring the noise instead of assuming it is
    absent is the difference between this and a decoration.

    The sampling spans over a second on purpose. Six back-to-back grabs take about 40 ms and would
    sit entirely inside the gap between two ticks of a 2 s timer, so the old version measured that
    the window was quiet for one fortieth of the interval it had to be quiet over.
    """
    import time

    seen = set()
    for _ in range(n):
        app.processEvents()
        seen.add(_pixels(root))
        time.sleep(seconds / n)
    return len(seen) > 1


def wait_still(roots, app, tries=20, step=0.05) -> bool:
    """Pump events until two consecutive grabs of every root agree. Returns whether they ever did.

    THE BASELINE MUST BE A STILL FRAME. `_pixel_noise` proves the window has no PERIODIC repaint;
    this is the other half, and it is the half that actually bit. The sweep clicks ~30 controls in
    a row and several of them start something asynchronous (a preview stream, a tab build, a
    focus worker). A repaint queued by control N lands inside control N+1's ``processEvents`` and
    is attributed to it — measured 2026-08-06 on the gate's OWN mutation, a chip wired to nothing,
    reported as "reaches: pixels" on one run and correctly as dead on the next. An intermittent
    gate is worse than no gate: it teaches people to re-run it.
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


#: Why each neutralised entry point is neutralised. Printed next to the control that reached it,
#: because "the click reached its handler and the harness stopped it there" and "the click did
#: what it promises" are DIFFERENT claims and this file must never print the second for the first.
NEUTRALISED_WHY = {
    "RegionViewer._open_3d": "vispy needs a live GL context; the volume path cannot be driven "
                             "offscreen (docs/rendering-contract.md)",
    "RegionViewer._view_roi_2d": "opens a child window and a second preview stream",
    "RegionViewer._record_movie": "writes an .mp4 and runs a multi-second encode",
    "RegionViewer._open_roi_children": "opens one child window per ROI",
    "PlateWindow.run_operator": "a multi-minute operator run over the plate",
    "PlateWindow.run_minerva_export": "writes an export tree",
    "PlateWindow.ingest": "re-opens the acquisition under the sweep",
    "PlateWindow.close": "ends the window the sweep is walking",
    "PlateWindow._open_acquisition_dialog": "a modal file dialog",
}


#: INPUT controls: their whole job is to hold a value another control reads, so moving one changes
#: nothing on screen and the sweep would call them dead. They are not swept — they are PROVEN
#: instead, by :func:`prove_inputs_reach_the_run`, which sets each one and then reads the arguments
#: that arrived at the call. An entry here without a proof is how a real dead input gets waved
#: through, so the proof emits their rows and the gate fails if it cannot.
DEFERRED_INPUTS = {
    ("view", "QComboBox"): "the operator picker — read by 'Run'",
    ("view", "save"): "preview vs persist — read by 'Run'",
}


def prove_inputs_reach_the_run(view, detail, app):
    """Set the region window's operator picker and its ``save`` box, click **Run**, and read the
    arguments that actually arrived at ``PlateWindow.run_operator``.

    This is the one question worth asking about an input control, and it is the question this
    project has already got wrong once with money on it: ``_workers._OperatorWorker``'s preview
    branch called ``project_plate`` WITHOUT ``operator_kwargs`` while the save branch passed them,
    so a value the user typed reached the console line and not the pixels — 57 labels against 44.
    "The widget changed" would have been green for that. "The value arrived at the call" is not.

    Returns rows for both controls; a run that never reaches the call gives them ``no outcome``,
    which fails the gate exactly as a dead chip does.
    """
    from qtpy.QtWidgets import QCheckBox, QComboBox, QPushButton

    combo = next((c for c in view.findChildren(QComboBox) if not _third_party(c)), None)
    save = next((c for c in view.findChildren(QCheckBox) if c.text() == "save"), None)
    run = next((b for b in view.findChildren(QPushButton) if b.text() == "Run"), None)
    if combo is None or save is None or run is None:
        return [("view", "Run inputs", "-", "no outcome",
                 "the operator picker, the save box or Run is missing from this window")]

    # A DIFFERENT entry from the one it opens on, and `save` flipped, so a handler that ignores
    # them and passes a default cannot accidentally agree with what was set.
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
    # The stub replaced an attribute on the CLASS, so it is called unbound: args[0] is the
    # PlateWindow. Reading `args[0]` as the operator key reported "the run was asked for
    # <PlateWindow object ...>" — a harness bug that reads exactly like a product one.
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
    """Put the plate back on its Operators home tab between controls.

    Not tidiness. The first card clicked opens its own tab and takes the focus with it, so the six
    operator cards BEHIND it become ``isVisible() == False`` and the sweep would report them as
    "hidden" — six controls silently unmeasured because of the order this file happened to walk
    the tree in. A sweep that changes what it can still reach has to put the surface back.
    """
    tabs = getattr(win, "_left_tabs", None)
    if tabs is not None and tabs.count():
        tabs.setCurrentIndex(0)


def sweep_controls(root, kind: str, app, watched=None, recorder=None, settle=None, observed=None):
    """Actuate every control of *root* and return one row per control.

    A row is ``(label, class, verdict, evidence)``. The verdicts:

    ``reaches``      something observable changed, and the evidence names WHAT;
    ``raised``       the click raised — the loudest possible finding, and a failure;
    ``neutralised``  the click reached an entry point this harness deliberately stubs. The handler
                     was REACHED (which is the half that keeps dying); its outcome is NOT observed
                     here and the row says which one and why;
    ``no outcome``   nothing anywhere in the app changed, nothing was logged, nothing was called;
    ``hidden`` / ``disabled``  the user cannot reach it in this state, reported not skipped.

    *watched* is an extra list of roots to fingerprint besides *root* — a region window's chips
    reach the PLATE, and a change there is the outcome, so the plate has to be watched too.
    *recorder* is the list :func:`_neutralise` / :func:`_neutralise_view` append to.
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
        # STREAMED as it is produced, the same rule `tools/walkthrough.py::check` follows: this
        # sweep clicks every button in a Qt app, and a click that hangs or aborts the process
        # would otherwise take the whole table with it and name nothing. The last line printed IS
        # the control that was in the chair.
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
        # A STILL FRAME first, then the baseline. See `wait_still`: the previous control's queued
        # repaint would otherwise land inside this one's measurement.
        still = wait_still(watched, app)
        # `blank=wdg` only on the window the control lives in; on any OTHER watched window the
        # control is not a descendant and nothing needs painting out.
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
            # ...and a still frame again before reading the outcome, so a control whose effect is
            # a LATE repaint (a tab building, a worker's first tile) is credited with it here
            # rather than blamed on whatever is clicked next. INSIDE the spy: a line logged while
            # this settles is this control's line.
            wait_still(watched, app, tries=10)
        after = [_fingerprint(r, p and still, wdg if i == 0 else None)
                 for i, (r, p) in enumerate(zip(watched, use_pixels))]

        reached = recorder[n_calls:]
        if reached:
            # The handler WAS reached — the half that keeps dying in this codebase — and the
            # harness stopped it there on purpose. Reported as its own verdict rather than folded
            # into `reaches`, because what happens after the entry point is not measured here.
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
                    # The control's OWN state moving is the actuation, not an outcome. Excluding it
                    # is what stops every checkbox in the app from reporting itself alive.
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


class _ModelPane:
    """Built lazily: a region-window pane whose napari canvas is absent but whose MODEL is real.

    ``napari.components.ViewerModel`` is Qt-free, so ``MosaicLayers`` — the class every operator
    layer, every contrast write and every visibility toggle in a region window goes through — runs
    headless in full. This is strictly more honest than the recording stub in ``tests/conftest.py``
    (which has no ``ops()`` at all, so ``RegionViewer._window_operators()`` returned ``[]`` against
    it and every ⚙ controls test had to replace the mosaic wholesale to see anything).

    What is NOT covered, and is not claimed: the vispy canvas. Nothing here proves a layer was
    PAINTED, only that it exists in the model with the scale, contrast and visibility it should.
    """


def _model_pane_class():
    from qtpy.QtWidgets import QWidget

    from napari.components import ViewerModel

    from squidmip._napari_view import MosaicLayers

    class ModelPane(QWidget):
        __doc__ = _ModelPane.__doc__
        ok = True

        def __init__(self):
            super().__init__()
            self._viewer = ViewerModel()
            self.mosaic = MosaicLayers(self._viewer)
            self.detect_channel = None
            self.detect_button = None
            self.said = []

        def say(self, text):
            self.said.append(text)

    return ModelPane


def _watch_window_stacking(monkey, seen):
    """WRAP (never stub) the window-stacking calls, so "bring the plate forward" is observable.

    ``raise_()`` and ``activateWindow()`` are no-ops on the offscreen platform — Qt says so itself,
    once per call: "This plugin does not support raise()". So ``▣ plate``, whose entire job is to
    bring the plate forward and which says nothing when it succeeds, changed NOTHING this harness
    could read and was reported as a dead control. It is not dead;
    ``tests/test_raise_plate.py::test_clicking_it_raises_the_plate`` proves the wiring against a
    counting fake.

    Wrapping rather than neutralising is the point: the real call still runs, and the record is
    evidence of the outcome rather than of the harness having intercepted it.
    """
    from qtpy.QtWidgets import QWidget

    import squidmip._viewer as V

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

    Same rule as :func:`_neutralise`: the call is RECORDED and not run, so "the click reached the
    handler" is still measured — it is the handler's blast radius that is contained. Appends into
    the caller's *called* list so one recorder covers the plate's entry points and this window's.
    """
    from squidmip import _region_viewer as RV

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
    import squidmip._napari_pane as napari_pane
    import squidmip._viewer as V

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
        # BEFORE the window is opened, not after. `RegionViewer._chip` connects a BOUND METHOD
        # captured at construction (``b.clicked.connect(lambda _=False: slot())``), so patching the
        # class afterwards leaves every already-built chip calling the real handler. Measured: the
        # sweep reported `2D` as "no outcome" (its neutralised stub never ran, so nothing was
        # recorded) and then took the process down with SIGSEGV on `3D`, which reached vispy.
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
            # The INPUT controls the sweep skipped: proven by the arguments that arrive at the run.
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


def _mount_dead_button(win):
    """Bolt a chip that LOOKS alive and is wired to nothing onto the plate.

    This is the shape of every defect in GATE 3's docstring: a button with a face, a tooltip and
    a cursor, connected either to no slot at all or to a handler that returns at its first guard.
    """
    from qtpy.QtWidgets import QPushButton
    b = QPushButton("⚙ tune", win)
    b.setToolTip("Tune the operator on screen.")      # a promise; nothing keeps it
    win.statusBar().addWidget(b)
    b.show()


def _mount_dead_view_button(view):
    """The same mutation on a REGION window, wired to a handler that returns at its first guard.

    Deliberately not "connected to nothing": ``_on_detect_nuclei`` and the plate's timepoint bar
    both HAD a handler, and it is the early return that made them dead. A gate that only catches
    a missing connection would have caught neither.
    """
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
    print("               the shape of _on_detect_nuclei and of the plate's timepoint bar.")
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
    # os._exit, NOT sys.exit. Measured 2026-08-06: the gate printed its full verdict and the
    # process then died with SIGSEGV (139) unwinding Qt at interpreter shutdown. A gate whose exit
    # code is decided by a teardown crash gates nothing.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
