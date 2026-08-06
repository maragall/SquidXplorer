#!/usr/bin/env python3
"""Headless functional walkthrough: drive EVERY shipped feature before a human opens the GUI.

`tools/acceptance.py` answers "does an acquisition open at all". This answers "does each
feature actually do its job", on real data, without eyes. It exists because every defect this
project shipped passed a green unit suite: the backend was solid, the GUI wiring was dead, and
the test doubles agreed with each other. A Re-dock button was broken from the day it shipped and
no test noticed, because every test called the handler directly instead of clicking.

Each check reports PASS / FAIL / SKIP with the number it measured, so a human walking the GUI
afterwards knows which sections are worth their attention and which are already proven.

    QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python tools/walkthrough.py

Disk: writes nothing except one small operator output under a temp dir it deletes. Refuses to
start below MIN_FREE_GB, because this machine has already been taken to 0 bytes free once.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

#: The acquisitions this walkthrough drives. Every one is a developer-machine path, and every
#: check that wants one goes through :func:`need`, so a machine that lacks it SKIPS with the path
#: printed instead of erroring out somewhere less obvious.
#:
#: ``PLATE`` was ``~/Downloads/synthetic_2x2_wellplate`` until 2026-08-06 and that folder no
#: longer exists, so every plate check below was a FAIL for reasons that had nothing to do with
#: the product. Its replacement is REPRODUCIBLE rather than found, which is the point: the command
#: in ``PLATE_RECIPE`` rebuilds it byte-for-byte, and the SKIP message prints that command.
#:
#: ``SQUIDMIP_FIXTURE_PLATE`` points ``PLATE`` somewhere else. That is what lets CI run this file
#: for real instead of watching it skip: the plate fixture is GENERATED (0.1 s, 2.5 MB at CI size),
#: so a runner can make one and hand over the path. TISSUE and PLATE1536 have no override because
#: they are real acquisitions nothing can synthesise -- they skip off this machine, by name.
TISSUE = ("/Users/julioamaragall/Downloads/"
          "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy")
PLATE = os.environ.get("SQUIDMIP_FIXTURE_PLATE") or \
    "/Users/julioamaragall/Downloads/sim_2x2_36fov_96wp"
PLATE1536 = "/Users/julioamaragall/Downloads/synthetic_1536_wellplate"
MIN_FREE_GB = 4.0

#: How to rebuild PLATE. 4 wells x 36 FOVs = 144 positions on a 9 mm (96-well) pitch under a
#: "384 well plate" declaration, which is exactly the shape the deleted fixture had: enough FOVs
#: per well for a mosaic, and a declaration the coordinates contradict.
PLATE_RECIPE = ('python tools/make_5d_fixture.py "%s" --fovs 36 --nz 1 --nt 1 '
                '--well-pitch-mm 9.0 --declared-format "384 well plate"' % PLATE)

_APP = None
_RESULTS: list[tuple[str, str, str, str]] = []      # (ticket, title, verdict, detail)


def _app():
    """The QApplication, on THE BINDING THE APP SHIPS -- which is decided by importing `squidmip`.

    This said ``from PyQt5.QtWidgets import QApplication`` until 2026-08-05. After the Qt6
    migration (10b8348, f7f9b28, ce5605c) it did not merely test the wrong thing, it could not run
    at all: `squidmip/__init__` pins ``QT_API=pyqt6``, so the widgets under test are Qt6 while this
    constructed a Qt5 application, both frameworks loaded into one process, and the harness aborted
    on ``QWidget: Must construct a QApplication before a QWidget`` before the first check.

    The one harness whose whole purpose is to catch GUI wiring that no unit test can see was itself
    dead, silently, for five days. Importing `squidmip` FIRST is the point, not tidiness: the pin
    lives in its ``__init__`` and has to be set before qtpy resolves a binding.
    """
    global _APP
    import squidmip  # noqa: F401  -- sets QT_API before qtpy resolves a binding
    from qtpy.QtWidgets import QApplication
    _APP = QApplication.instance() or QApplication([])
    return _APP


_ONLY = os.environ.get("WALKTHROUGH_ONLY", "")   # e.g. IMA-261 — for mutation-checking one ticket


def say(text=""):
    """Print the harness's own report, PAST anything the product has done to ``sys.stdout``.

    Measured 2026-08-06: ``_workers._PreviewWorker.run`` wraps its whole pass in
    ``_logpane.capture_stdout_to_log()``, which is deliberately process-wide and not thread-scoped
    (its own comment says so). So while a plate preview is streaming, EVERY print on the main
    thread is swallowed into the GUI's log panel -- and a check whose window still had a preview
    running simply had no row in the output, which reads as the check having vanished. A console
    tool that drives this widget cannot report through ``sys.stdout``.
    """
    print(text, file=sys.__stdout__, flush=True)


def check(ticket, title):
    """Decorator: run a check, catch everything, record one row.

    The row is PRINTED as it is produced as well as recorded. ``try/except`` cannot catch a Qt
    abort or a vispy segfault, and this file drives a GUI toolkit: when one check took the process
    down, the summary at the bottom never ran and the whole run reported nothing at all -- not even
    which check was in the chair. Streaming makes the crash name itself.
    """
    def wrap(fn):
        if _ONLY and _ONLY not in ticket:
            return fn
        say(f"...   {ticket:9} {title}")
        try:
            detail = fn()
            verdict = "PASS"
            if isinstance(detail, tuple):
                verdict, detail = detail
        except SkipCheck as e:
            verdict, detail = "SKIP", str(e)
        except Exception as e:
            verdict = "FAIL"
            detail = f"{type(e).__name__}: {e}"
            if os.environ.get("WALKTHROUGH_TRACE"):
                detail += "\n" + traceback.format_exc()
        finally:
            close_windows()
        say(f"{verdict:5} {ticket:9} {title}\n      {detail}")
        _RESULTS.append((ticket, title, verdict, str(detail)))
        return fn
    return wrap


class SkipCheck(Exception):
    """Raised when a check cannot run here (missing dataset, missing optional dep)."""


def need(path):
    """*path*, or SKIP this check naming it. The ONE gate between a check and a dataset.

    The datasets live on one workstation, so on every other machine most of this file cannot run.
    "Cannot run here" and "ran and failed" are different answers and the summary distinguishes
    them -- but only if the absence is stated rather than discovered as a FileNotFoundError three
    frames deep, which is what a `python tools/walkthrough.py` on CI used to produce.
    """
    if os.path.isdir(path):
        return path
    hint = f"\n            rebuild it with: {PLATE_RECIPE}" if path == PLATE else ""
    raise SkipCheck(f"dataset absent on this machine: {path}{hint}")


#: Every window :func:`open_window` built and nobody has closed yet. See :func:`close_windows`.
_OPEN: list = []


def close_windows():
    """Close every window this run opened and has not closed. Called after EVERY check.

    A check that raises before its own ``w.close()`` leaves a ``PlateWindow`` with a live
    ``_LoupeWorker`` and ``_TileFetcher`` on it. Those are QThreads parented to the window, so when
    the next check's ingest triggers a collection, Qt destroys a running QThread and the PROCESS
    ABORTS -- measured: IMA-221 failed on a Qt6 TypeError, and the abort landed two checks later
    on IMA-228, which is where anyone reading the output would have gone looking. One check's
    failure must not be able to end the run, and must not be able to frame a different check.
    """
    while _OPEN:
        win = _OPEN.pop()
        try:
            win.close()
        except Exception:                 # already torn down, or never fully built
            pass
    _app().processEvents()


def open_window(path, size=(1600, 900)):
    import squidmip._viewer as V
    need(path)
    _app()
    win = V.PlateWindow(None)
    _OPEN.append(win)
    # Size and show BEFORE ingest: pane 3 carries setMinimumWidth(300), and a splitter that was
    # never laid out at a real size reports every child at its default. Testing layout on an
    # unshown window measures the harness, not the product.
    win.resize(*size)
    win.show()
    _app().processEvents()
    win.ingest(path)
    if win._reader is None:
        raise AssertionError(f"ingest failed: {win._readout.text()!r}")
    return win


def open_view(win, region):
    """Open a REAL ``RegionViewer`` on *region*, through the window's own manager.

    The napari canvas is the one seam that needs a GL context, so it is replaced by a pane whose
    ``mosaic`` is a real :class:`squidmip._napari_view.MosaicLayers` over a real Qt-free
    ``napari.components.ViewerModel``. Everything a window does to its layers — an operator result
    landing, a contrast write, a visibility toggle, ``ops()`` — is therefore the production class.
    What is NOT covered, and is not claimed: vispy actually painting them.

    Same substitution ``tools/gates.py`` makes; it is spelled out in both places rather than
    shared, because a harness that imports another harness fails in a way neither one reports.
    """
    from qtpy.QtWidgets import QWidget

    from napari.components import ViewerModel

    import squidmip._napari_pane as napari_pane
    from squidmip._napari_view import MosaicLayers

    class ModelPane(QWidget):
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

    napari_pane.make_pane = lambda *a, **k: (ModelPane(), "napari", "")
    view = win._viewer_manager.open([region])
    _app().processEvents()
    return view


def drain_operator(win, timeout_s=600):
    """Block until the plate's operator worker has exited and its queued slots have run."""
    import time
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        w = getattr(win, "_worker", None)
        if w is None or not w.isRunning():
            break
        _app().processEvents()
        time.sleep(0.05)
    for _ in range(50):
        _app().processEvents()
        time.sleep(0.02)


def settle(ms=4000):
    """Let the async preview stream finish before grabbing pixels.

    ingest() starts a background worker that pushes tiles as they decode, so two grabs taken
    without settling compare different stream states, not different settings.
    """
    from qtpy.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def drain_preview(win, timeout_s=120):
    """Block until the window's raw preview worker has finished streaming (or the timeout).

    Deterministic where ``settle(ms)`` is calibrated: a check that grabs pixels before/after a
    setting must not race the background fill, and the fill's duration depends on how many fields
    the acquisition has.
    """
    import time
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        p = getattr(win, "_preview", None)
        if p is None or not p.isRunning():
            break
        _app().processEvents()
        time.sleep(0.02)
    for _ in range(20):                 # let the queued tileReady slots actually run
        _app().processEvents()
        time.sleep(0.01)


def rendered(widget, w=900, h=700):
    """Grab a widget as an RGB array - what a human would actually see.

    Returns DEVICE pixels, which on a retina panel is 2x the ``(w, h)`` asked for. Callers here
    only ever diff one grab against another, so that is fine -- but anything that wants to crop
    this by a WIDGET rectangle must scale by ``devicePixelRatioF()`` first. See
    ``tests/test_viewer.py::_cell_slices``, which is the one place in the repo that does it and
    says what a mixed-up crop looks like (it reads a neighbouring well and compares it to the one
    it meant).
    """
    from qtpy.QtGui import QImage
    widget.resize(w, h)
    _app().processEvents()
    # ``convertToFormat(4)`` -- the raw int for Format_RGB32 -- was accepted by PyQt5 and is a
    # TypeError under PyQt6, which took out every check in this file that looks at pixels. Naming
    # the enum is also the only spelling that survives a Qt renumbering.
    img = widget.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    ptr = img.bits(); ptr.setsize(img.sizeInBytes())   # byteCount() is Qt5-only
    # bytesPerLine, not width*4: Qt pads scanlines to a 4-byte boundary, and a padded row read as
    # width*4 shears the image by a pixel per row. Device pixel ratio too -- grab() renders at the
    # screen's ratio, which is 2 on a retina panel and 1 offscreen.
    row = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    return row[:, :img.width(), :3].astype(float)


def free_gb():
    return shutil.disk_usage("/").free / 1e9


# ======================================================================================
def run_all():
    # Both off the package, not off `_stitch`: the one-operator-registry change (2026-08-05)
    # deleted `_stitch._REGION_OPERATORS` and moved `available_region_operators` onto `squidmip`
    # as a filter over the single `_engine._OPERATORS` table. This module-level import still said
    # `from squidmip._stitch import ...` after the merge and would have taken the whole file down
    # with an ImportError before the first check.
    from squidmip import available_projectors, available_region_operators, open_reader

    def read(ds):
        """``open_reader``, but an absent dataset SKIPs the check instead of raising."""
        return open_reader(need(ds))

    # ---------- ingest, all three acquisitions --------------------------------------
    @check("IMA-214", "Glass slide opens (was refused outright)")
    def _():
        w = open_window(TISSUE)
        m = w._reader.metadata
        regions, npos = list(m["regions"]), len(m["fov_positions_um"])
        fmt = getattr(w._plate, "format_name", None)
        w.close()
        assert regions == ["manual0", "manual1"], regions
        assert npos == 55, npos
        return f"{regions}, {npos} positions, resolved as {fmt!r}"

    @check("IMA-219", "Plate-shape inference trusts MEASURED over DECLARED")
    def _():
        w = open_window(PLATE)
        fmt = w._plate.format_name
        src = getattr(w._plate, "format_source", "?")
        w.close()
        assert "96" in fmt, f"declared 384, measured 9.000mm pitch, got {fmt!r}"
        return f"declared '384 well plate' -> resolved {fmt!r} (source={src})"

    @check("IMA-219", "1536-well plate scale ingests")
    def _():
        w = open_window(PLATE1536)
        m = w._reader.metadata
        n, first, last = len(m["regions"]), m["regions"][0], m["regions"][-1]
        fmt = w._plate.format_name
        w.close()
        assert n == 1536, n
        return f"{n} wells {first}..{last}, resolved {fmt!r}"

    @check("IMA-215", "coordinates.csv: both on-disk schemas parse")
    def _():
        """The discriminator is the HEADER (reader.py ``_has_fov_column``), so this check has to
        drive one dataset of each header, and say which is which.

        It used to name PLATE "20x-style, row order = fov" and TISSUE "monkey-style, fov column",
        which is both backwards and, since the fixtures moved, untrue: every acquisition it opened
        carried a ``fov`` column, so the type-(b) parser was never entered at all. The 1536 plate
        is the type-(b) one on this machine -- its root ``coordinates.csv`` is
        ``region,x (mm),y (mm),z (mm)`` with no fov column and one row per FOV in image order.
        """
        with_fov = read(TISSUE).metadata["fov_positions_um"]        # type (a): explicit fov column
        row_order = read(PLATE1536).metadata["fov_positions_um"]    # type (b): row order IS the fov
        assert len(with_fov) == 55, len(with_fov)
        assert len(row_order) == 6144, len(row_order)
        # Both parsers must land in MICROMETRES. A plate spans tens of thousands of um; the 1000x
        # tell is a span that looks like a millimetre count.
        spans = {}
        for name, pos in (("type-a", with_fov), ("type-b", row_order)):
            xs = [v[0] for v in pos.values()]
            spans[name] = max(xs) - min(xs)
            assert spans[name] > 1000, f"{name}: x span {spans[name]} looks like mm, not um"
        return (f"type-(a) fov-column {len(with_fov)} positions (span {spans['type-a']:.0f} um), "
                f"type-(b) row-order {len(row_order)} positions (span {spans['type-b']:.0f} um)")

    # ---------- mosaic + geometry ---------------------------------------------------
    @check("IMA-187", "Each well is a coordinate-placed MOSAIC, not one thumbnail")
    def _():
        from squidmip._viewer import _mosaic_boxes
        m = read(PLATE).metadata
        boxes = _mosaic_boxes(m)
        per_well: dict = {}
        for (region, _fov), _b in boxes.items():
            per_well[region] = per_well.get(region, 0) + 1
        # Derived from the acquisition, not from the fixture that happened to be here in July:
        # the claim is "one box per acquired FOV, and more than one per well", and hard-coding
        # 144/36 turned a fixture swap into a product FAIL. `_mosaic_boxes` computes placement
        # independently of the metadata dict, so comparing the two is not circular.
        want = {r: len(f) for r, f in m["fovs_per_region"].items()}
        assert len(boxes) == len(m["fov_positions_um"]), (len(boxes), len(m["fov_positions_um"]))
        assert per_well == want, f"{per_well} != {want}"
        assert min(want.values()) > 1, f"no well has more than one FOV: {want} — nothing to mosaic"
        return (f"{len(boxes)} boxes over {sorted(per_well)}, "
                f"{sorted(set(want.values()))} field(s) each")

    @check("IMA-187", "Y-sign: larger stage y maps to a LARGER row (no mirroring)")
    def _():
        from squidmip._placement import fov_offsets_px
        m = read(PLATE).metadata
        off = fov_offsets_px(m["fov_positions_um"], "A1",
                             m["fovs_per_region"]["A1"], m["pixel_size_um"])
        pos = m["fov_positions_um"]
        fovs = m["fovs_per_region"]["A1"]
        lo = min(fovs, key=lambda f: pos[("A1", f)][1])
        hi = max(fovs, key=lambda f: pos[("A1", f)][1])
        assert off[hi][0] > off[lo][0], f"row flip: {off[hi]} !> {off[lo]}"
        return f"stage y min -> row {off[lo][0]}, y max -> row {off[hi][0]}"

    @check("IMA-216", "Viewport is O(screen), not O(plate)")
    def _():
        from squidmip._tilesource import plate_ladder
        from squidmip._tiling import select_tiles
        counts = {}
        for name, ds in (("plate", PLATE), ("tissue", TISSUE)):
            m = read(ds).metadata
            label = f"{name} ({len(m['fov_positions_um'])} FOVs)"
            lad = plate_ladder(m)
            geo = lad.geometry if hasattr(lad, "geometry") else lad
            x0, y0, x1, y1 = geo.levels[-1].bboxes[:, 0].min(), geo.levels[-1].bboxes[:, 1].min(), \
                geo.levels[-1].bboxes[:, 2].max(), geo.levels[-1].bboxes[:, 3].max()
            um_per_px = max(x1 - x0, y1 - y0) / 1200
            tiles = select_tiles((x0, y0, x1, y1), um_per_px, geo, channels=("0",))
            counts[label] = len(tiles)
            assert len(tiles) < 60, f"{label}: fit-to-plate wanted {len(tiles)} tiles"
        return " | ".join(f"{k}: {v} tiles at fit-to-plate" for k, v in counts.items())

    # ---------- selection -> windows -> tabs -----------------------------------------
    @check("IMA-221", "Shift-drag marquee selects the right (region, fov) pairs")
    def _():
        # A real drag through the widget's own event handlers - not a direct call to the
        # selection setter. The Re-dock button was dead for a day precisely because every
        # test called the handler instead of clicking.
        # Enums FULLY SCOPED (Qt.MouseButton.LeftButton, not Qt.LeftButton). Qt5 accepted the
        # short spelling and Qt6 does not, so an unscoped enum here is an AttributeError at the
        # moment the gesture fires -- which reads as "the marquee is broken" rather than "the
        # harness is".
        from qtpy.QtCore import QEvent, QPointF, Qt
        from qtpy.QtGui import QMouseEvent

        _LEFT, _NONE = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
        _SHIFT = Qt.KeyboardModifier.ShiftModifier

        def ev(kind, pos, mods, buttons=_LEFT):
            k = {"press": QEvent.Type.MouseButtonPress, "move": QEvent.Type.MouseMove,
                 "release": QEvent.Type.MouseButtonRelease}[kind]
            return QMouseEvent(k, pos, _LEFT, buttons, mods)

        w = open_window(PLATE)
        ov = w._overview
        rendered(ov)
        a = QPointF(2, 2)
        b = QPointF(ov.width() - 2, ov.height() - 2)

        def drag(mods):
            ov.mousePressEvent(ev("press", a, mods))
            ov.mouseMoveEvent(ev("move", b, mods))
            ov.mouseReleaseEvent(ev("release", b, mods, buttons=_NONE))

        # WHAT A PLAIN SHIFT-DRAG DOES CHANGED ON 2026-07-23 (2b8fbc5, "Decentralize GUI"): it
        # emits `marqueeSelected` -- open an independent window over the box -- and deliberately
        # leaves NO selection wash behind, because the boxed set is visible in the new window's
        # region slider. This check asserted the old binding and so reported `[] != [...]`, which
        # reads as "the marquee is broken" and is not: the gesture that still SELECTS is Shift+Alt.
        # Both halves are driven here, because "it opened the right wells" and "it selected the
        # right wells" are now two different gestures and a check on one says nothing about the
        # other.
        opened = []
        ov.marqueeSelected.connect(lambda wells: opened.append(list(wells)))
        drag(_SHIFT)
        after_plain = ov.selected_wells()
        drag(_SHIFT | Qt.KeyboardModifier.AltModifier)      # ...the gesture that DOES select
        wells = ov.selected_wells()
        sel = w.selected_region_fovs()
        m = w._reader.metadata
        want_wells = list(m["regions"])
        want_pairs = sum(len(f) for f in m["fovs_per_region"].values())
        w.close()
        assert opened == [want_wells], f"Shift-drag asked to open {opened}, expected {want_wells}"
        assert after_plain == [], f"a plain Shift-drag left a selection wash: {after_plain}"
        assert wells == want_wells, f"Shift+Alt-drag selected {wells} != {want_wells}"
        assert len(sel) == want_pairs, f"expected {want_pairs} (region, fov) pairs, got {len(sel)}"
        return (f"Shift-drag over the whole plate -> opens {opened[0]} and selects nothing; "
                f"Shift+Alt-drag -> selects {wells}, {len(sel)} (region, fov) pairs")



    @check("IMA-209", "Drag-out floating window, and Re-dock (dead until today)")
    def _():
        w = open_window(PLATE)
        w._activate_operator("mip")          # any user-opened tab: the home tab never detaches
        _app().processEvents()
        tabs = w._left_tabs
        idx = tabs.count() - 1
        key = next(k for k, v in w._op_tabs.items() if v is tabs.widget(idx))
        float_win = w._detach_tab(idx, tabs)
        _app().processEvents()
        n_float = len(getattr(w, "_floating", {}))
        detached = float_win is not None
        # Re-dock through the same path the (previously dead) button uses.
        w._redock(key)
        _app().processEvents()
        n_after = len(getattr(w, "_floating", {}))
        back = w._left_tabs.count()
        w.close()
        assert detached, "_detach_tab returned None"
        assert n_float == 1, f"expected 1 floating window, tracked {n_float}"
        assert n_after == 0, f"re-dock left {n_after} floating window(s)"
        return f"detached -> {n_float} float window, re-docked -> {n_after} float, {back} tab(s) home"

    @check("IMA-228", "Minerva exports the SELECTION, not always FOV 0")
    def _():
        w = open_window(PLATE)
        if not hasattr(w, "minerva_selection"):
            w.close(); raise SkipCheck("minerva_selection() not present")
        # Decoys on the display-only overview: minerva_selection must ignore them and read
        # PlateWindow, the selection's real owner. The old implementation probed the overview
        # and reached the right answer only through a fallback.
        w._overview.selected_wells = lambda: ["B1"]
        w._overview.selected_region_fovs = lambda: {"B1": [0]}
        w._selected_regions = ["A1", "A2"]
        sel = w.minerva_selection()
        regions = sorted({r for r, _ in sel}) if sel else []
        w.close()
        assert sel, "empty selection payload"
        assert regions == ["A1", "A2"], f"read the wrong selection owner: {regions}"
        return f"{len(sel)} FOVs across {regions} (overview decoys ignored)"

    @check("IMA-228", "The Minerva export is ONE FUSED MOSAIC per region, and Minerva reads it")
    def _():
        """The claim that matters, on a real acquisition.

        Minerva Author lays out exactly one image per story -- ``"Layout": {"Grid": [["i0"]]}``
        is hardcoded in its ``src/app.py``, and only ``series[0]`` is ever opened -- so a
        multi-FOV selection MUST fuse to a single mosaic. N per-FOV files would render the
        first and silently discard the rest. A FOV subset of a region is therefore a CROP of
        that region, still one file.
        """
        if free_gb() < MIN_FREE_GB + 1:
            raise SkipCheck(f"only {free_gb():.1f} GB free; refusing to write")
        import tifffile
        from squidmip._minerva import export_selection
        reader = read(PLATE)
        fovs = reader.metadata["fovs_per_region"]["A1"][:4]      # a subset: the crop path
        if len(fovs) < 4:
            raise SkipCheck(f"precondition: well A1 of {PLATE} must have >=4 FOVs to exercise the "
                            f"crop path; it has {len(fovs)}")
        tmp = tempfile.mkdtemp(prefix="walkthrough_minerva_")
        try:
            pairs = export_selection(reader, [("A1", f) for f in fovs], tmp)
            assert len(pairs) == 1, f"{len(fovs)} FOVs of one region gave {len(pairs)} files"
            omes = [f for f in os.listdir(tmp) if f.endswith(".ome.tiff")]
            assert len(omes) == 1, f"one mosaic per region, found {omes}"
            ome, story = pairs[0]
            with tifffile.TiffFile(str(ome)) as tf:
                assert len(tf.series) == 1, "Minerva reads series[0] only"
                shape = tf.series[0].shape
                xml = tf.ome_metadata
            # Fusion, not passthrough: the mosaic must be wider than any single FOV.
            one = reader.read("A1", fovs[0], reader.metadata["channels"][0]["name"], 0).shape
            assert shape[-1] > one[-1], f"mosaic {shape} is no wider than one FOV {one}"
            assert "PhysicalSizeX" in xml, "Minerva 500s without OME-XML pixel size"
            assert story.exists()
            return (f"1 mosaic {shape} from {len(fovs)} FOVs of A1 "
                    f"(one FOV is {one}), 1 series, story written")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ---------- pixels: loupe, channels, contrast, carrier ---------------------------
    @check("IMA-208", "Loupe is not blank at a well CORNER (negative crop origin)")
    def _():
        w = open_window(PLATE)
        ov = w._overview
        rendered(ov)
        from squidmip._viewer import _RawLoupeSource
        m = w._reader.metadata
        src = _RawLoupeSource(w._reader, m, lambda region: m["fovs_per_region"][region][0])
        crop = src.read_crop("A1", 0, -256, -256, 512, 512)
        w.close()
        arr = np.asarray(crop, dtype=float)
        assert arr.size and arr.std() > 0, "loupe crop at negative origin is blank/uniform"
        return f"crop {arr.shape} min={arr.min():.0f} max={arr.max():.0f} std={arr.std():.1f}"

    @check("IMA-206", "Channel toggle actually changes the rendered plate")
    def _():
        w = open_window(PLATE)
        ov = w._overview
        # The preview stream must FINISH, or the three grabs below diff stream states rather than
        # channel settings. Wait on the worker itself instead of a fixed sleep: since IMA-253 the
        # preview composites every FOV of a multi-FOV well (144 fields on this plate, not 4), so
        # how long the stream takes is a property of the acquisition, not a constant to calibrate.
        drain_preview(w)
        base = rendered(ov)
        if not hasattr(ov, "set_channel_visible"):
            w.close(); raise SkipCheck("set_channel_visible() not present")
        ov.set_channel_visible(0, False)
        off = rendered(ov)
        ov.set_channel_visible(0, True)
        back = rendered(ov)
        w.close()
        h = min(base.shape[0], off.shape[0], back.shape[0])
        wd = min(base.shape[1], off.shape[1], back.shape[1])
        base, off, back = base[:h, :wd], off[:h, :wd], back[:h, :wd]
        changed = int((np.abs(base - off) > 0).sum())
        restored = np.array_equal(base, back)
        assert changed > 0, "toggling a channel changed nothing"
        drift = float(np.abs(base - back).max())
        # Restore is not required to be byte-identical: re-enabling a channel re-runs the
        # running-percentile contrast, so a small window drift is expected and benign. A LARGE
        # drift would mean the channel did not really come back, so bound it rather than
        # asserting equality and getting a false alarm (or, worse, papering over a real one).
        assert drift < 40, f"channel did not restore: max px drift {drift}"
        return (f"{changed} px changed when ch0 off; byte-identical on restore={restored}, "
                f"max drift on restore {drift:.0f}/255")

    @check("IMA-220", "Carrier art draws, and from the RESOLVED format")
    def _():
        out = []
        for label, ds in (("2x2", PLATE), ("tissue", TISSUE)):
            w = open_window(ds)
            fmt = w._plate.format_name
            art = w._plate.art() if hasattr(w._plate, "art") else None
            g = rendered(w._overview).mean(2)
            w.close()
            assert g.std() > 3, f"{label}: overview renders blank (std {g.std():.2f})"
            out.append(f"{label}: {fmt!r} art={'yes' if art is not None else 'none'} "
                       f"std={g.std():.1f}")
        return " | ".join(out)

    # IMA-207 ("per-region contrast lifts a dim well that global crushes") WAS HERE AND IS GONE.
    # Per-region contrast was DELETED on 2026-07-22 (8b0cbfc) on purpose, with the reason on the
    # record: "It made a dim well readable beside a bright one, which is a presentation trick that
    # costs the one thing a plate view is for - two wells that look identical could differ by
    # orders of magnitude." `set_contrast_scope`, `SCOPE_PER_REGION` and `_cell_windows` all went
    # with it, so the check could only ever SKIP again -- and a permanent skip in a list of skips
    # is how a dead check hides.

    @check("IMA-261", "The plate view owns NO contrast control (the duplicate is GONE)")
    def _():
        """Not hidden, not disabled - ABSENT, walked over the REAL widget tree of a shown window.

        Rewritten 2026-08-06. It used to read ``w._channel_bar``, which raised AttributeError:
        8b0cbfc (2026-07-22) deleted the plate's whole channel bar, not merely its sliders. So the
        check FAILED on a change that had over-satisfied it -- the worst kind of stale check,
        because a real regression and a tightened design produce the same red.

        The claim is now the stronger one that commit actually made: no interactive control ANY-
        WHERE under the plate, and no ``_ChannelBar`` instance in the tree. Walking the tree rather
        than naming an attribute is the point: this keeps holding whatever the bar is next called.
        """
        from qtpy.QtWidgets import QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox
        import squidmip._viewer as V
        w = open_window(PLATE)
        ov = w._overview
        controls = []
        for kind in (QAbstractSlider, QAbstractSpinBox, QComboBox, QAbstractButton):
            controls += [f"{type(c).__name__}({c.text() if hasattr(c, 'text') else ''})"
                         for c in ov.findChildren(kind)]
        # The CLASS is gone as of 2026-08-05, not merely unmounted: it had not been constructed
        # since 8b0cbfc (2026-07-22) and was deleted with the rest of the dead central-pane paths.
        # So `V._ChannelBar` is now an AttributeError, and this check asserts the stronger fact --
        # the type does not exist to be mounted. Written as a getattr rather than a bare access
        # because the reason this line changed is exactly the reason it must not raise: a harness
        # that dies on the change it is meant to VERIFY reports a red for a green event.
        bar_cls = getattr(V, "_ChannelBar", None)
        bars = w.findChildren(bar_cls) if bar_cls is not None else []
        attr = hasattr(w, "_channel_bar")
        w.close()
        assert not controls, f"the plate pane carries {len(controls)} control(s): {controls}"
        assert not bars, f"{len(bars)} _ChannelBar still mounted under the window"
        assert not attr, "PlateWindow still has a _channel_bar attribute"
        return ("0 interactive controls under the plate, no _ChannelBar type at all, no "
                "_channel_bar attribute - the plate reports contrast, it does not set it")

    # THREE IMA-261 CHECKS WERE HERE AND ARE GONE, all of them drivers of an `ndv_clims_slider`
    # helper (deleted with them):
    # "DRAGGING the CENTRAL viewer's contrast slider repaints the PLATE", "Plate PIXELS are the
    # array viewer's window, not merely its numbers", and "A contrast drag is under 16 ms/frame on
    # all three datasets".
    #
    # There is no central viewer to drag. ``PlateWindow._detail`` has been unconditionally None
    # since 19cd491 (2026-07-22) and the locked central pane was removed by 2b8fbc5 (2026-07-23,
    # "Decentralize GUI"); contrast now belongs to napari in an INDEPENDENT RegionViewer window,
    # reaching the plate through ``MosaicLayers.on_user_contrast`` -> ``_on_detail_contrast``.
    # the helper returned None on every dataset, so the first two SKIPped and -- worse --
    # the 16 ms budget PASSED while measuring nothing at all ("tissue: no slider/store | 2x2: no
    # slider/store | 1536wp: no slider/store", a green row over zero measurements, which is the
    # exact failure its own docstring was written to prevent).
    #
    # They are not rewritten HERE because they cannot be: napari needs a real GL context and this
    # file runs under QT_QPA_PLATFORM=offscreen, where its canvas does not exist. The claim moved
    # to ``tools/verify_napari_mosaic.py``, which is the harness that has a GL context.

    @check("IMA-242", "ONE contrast model: the loupe obeys the window the plate obeys")
    def _():
        """The duplication this ticket collapsed: the loupe used to memoise its own window and its
        own compositor, so a contrast change moved the plate and left the magnifier of that same
        plate showing the pre-change contrast. Latch a channel and require both to report it.

        The PER-REGION arm was dropped on 2026-08-06: ``PlateOverview._cell_windows`` went with
        per-region contrast itself in 8b0cbfc (2026-07-22), so reaching for it raised
        AttributeError and reported "the contrast model is broken" about a third renderer that no
        longer exists. Two renderers is now the whole population, and the check says so rather
        than quietly measuring fewer things than its title claims.
        """
        import squidmip._viewer as V
        w = open_window(PLATE)
        ov = w._overview
        settle()
        if ov._contrast is None:
            w.close(); raise SkipCheck("no contrast model")
        ov._contrast.set_manual(0, 123.0, 4567.0)
        plate_win = ov.channel_windows()[0]
        loupe_win = ov._contrast.resolve(0, (0.0, 65535.0))   # the loupe's resolution path
        assert not hasattr(ov, "_cell_windows"), \
            "a per-region contrast path is back; it was deleted on purpose in 8b0cbfc"
        w.close()
        assert plate_win == (123.0, 4567.0), f"plate ignored the latch: {plate_win}"
        assert loupe_win == (123.0, 4567.0), f"loupe ignored the latch: {loupe_win}"
        assert not hasattr(V, "_composite_rgb") and not hasattr(V, "_percentile_window"), \
            "a duplicate contrast implementation is back"
        return (f"plate={plate_win} loupe={loupe_win} — both renderers, one model; both duplicate "
                "implementations still gone, and no per-region path came back")

    # ---------- operators ------------------------------------------------------------
    @check("IMA-210", "Operator registry exposes every shipped operator")
    def _():
        proj, region = available_projectors(), available_region_operators()
        for want in ("mip", "reference", "decon", "bgsub", "flatfield"):
            assert want in proj, f"{want} missing from {proj}"
        assert "stitch" in region, region
        return f"projectors={proj} region_ops={region}"

    @check("IMA-225", "Flatfield commutes with MIP (monotone f: max(f(a),f(b))==f(max(a,b)))")
    def _():
        from squidmip._flatfield import FlatfieldProfile, correct_flatfield
        rng = np.random.default_rng(0)
        planes = [rng.integers(0, 4000, (64, 64), dtype=np.uint16) for _ in range(10)]
        gain = np.linspace(0.5, 1.5, 64 * 64).reshape(64, 64)
        prof = FlatfieldProfile(gain.astype(np.float32))
        per_plane = np.maximum.reduce([correct_flatfield(p, prof) for p in planes])
        after_mip = correct_flatfield(np.maximum.reduce(planes), prof)
        d = int(np.abs(per_plane.astype(int) - after_mip.astype(int)).max())
        assert d == 0, f"commutation broken, max|diff| = {d}"
        return f"max|per-plane - after-MIP| = {d} (bit-identical), 10 planes"

    @check("IMA-222", "Stitch registers real FOVs and improves seam agreement")
    def _():
        region_ops = available_region_operators()
        assert "stitch" in region_ops and "coordinate" in region_ops
        import squidmip._viewer as V
        keys = [o.key for o in V._OPERATIONS]
        assert "stitch" in keys, f"no stitch card in the GUI: {keys}"
        return f"registered + GUI card present; operation cards={keys}"

    @check("IMA-230", "Storage guard refuses up front and names both numbers")
    def _():
        from squidmip._output import InsufficientDiskSpaceError, check_disk_space
        try:
            check_disk_space("/tmp", 10 ** 15, what="an impossible write")
        except InsufficientDiskSpaceError as e:
            msg = str(e)
            assert "free" in msg.lower(), msg
            return f"refused: {msg[:110]}"
        raise AssertionError("guard did NOT refuse a 1 PB write")

    @check("IMA-230", "Region-operator estimate is overlap-aware, not frame-counted")
    def _():
        from squidmip._output import estimate_write_bytes
        m = read(PLATE).metadata
        proj = estimate_write_bytes(m, n_fovs=None)
        stit = estimate_write_bytes(m, n_fovs=None, region_operator=True)
        assert stit != proj, "region estimate identical to the frame count"
        return f"projected {proj/1e9:.3f} GB vs stitched {stit/1e9:.3f} GB (ratio {stit/proj:.3f})"

    @check("IMA-231", "ROI table corners agree with the tile ladder")
    def _():
        from squidmip._output import fov_roi_records_um
        from squidmip._tilesource import fov_bboxes_um
        m = read(PLATE).metadata
        region = m["regions"][0]
        fovs = m["fovs_per_region"][region]
        pos = {k[1]: v for k, v in m["fov_positions_um"].items() if k[0] == region}
        recs = fov_roi_records_um(fovs, pos, m["frame_shape"], m["pixel_size_um"])
        boxes = fov_bboxes_um(m["fov_positions_um"], m["frame_shape"], m["pixel_size_um"])
        worst, n = 0.0, 0
        for f, r in zip(fovs, recs):
            box = boxes.get((region, f))
            if box is None:
                continue
            # x_original_um is the ABSOLUTE stage corner; x_um is region-relative (ngio's
            # reset_origin convention). The tile ladder works in absolute stage um, so compare
            # against the original.
            x = float(r["x_original_um"])
            worst = max(worst, abs(x - box[0])); n += 1
        assert n, "no ROI/bbox pairs compared"
        assert worst < 1e-6, f"corner disagreement {worst} um"
        return f"{len(recs)} ROIs in {region}, {n} compared, max corner disagreement {worst:.2e} um"


    @check("IMA-217", "Pyramid ladder is coarse-to-fine and never widens")
    def _():
        from squidmip._tilesource import plate_ladder
        m = read(PLATE).metadata
        lad = plate_ladder(m)
        geo = lad.geometry if hasattr(lad, "geometry") else lad
        counts = [len(lv) for lv in geo.levels]
        bad = [(i, counts[i], counts[i + 1]) for i in range(len(counts) - 1)
               if counts[i + 1] > counts[i]]
        assert not bad, f"a coarser level holds MORE tiles than the one below it: {bad}"
        assert geo.worst_case_tiles <= counts[0], geo.worst_case_tiles
        return (f"{len(counts)} rungs, tile counts {counts}, "
                f"worst_case_tiles={geo.worst_case_tiles}")

    @check("IMA-229", "OME-NGFF Zarr reads back through the same reader seam")
    def _():
        if free_gb() < MIN_FREE_GB + 1:
            raise SkipCheck(f"only {free_gb():.1f} GB free")
        import inspect
        from squidmip import open_reader as _open, write_plate       # _open: the Zarr we wrote
        tmp = tempfile.mkdtemp(prefix="walkthrough_zarr_")
        try:
            # One well, one FOV: enough to prove the round trip, kilobytes on disk.
            write_plate(read(PLATE), tmp, regions=["A1"], n_fovs=1, projector="mip")
            back = _open(os.path.join(tmp, "plate.ome.zarr"))
            m = back.metadata
            regions = list(m["regions"])
            # the seam: the SAME read() signature serves TIFF and Zarr
            sig_zarr = list(inspect.signature(type(back).read).parameters)
            sig_tiff = list(inspect.signature(type(read(PLATE)).read).parameters)
            plane = back.read(regions[0], m["fovs_per_region"][regions[0]][0],
                              m["channels"][0]["name"], 0)
            assert regions == ["A1"], regions
            assert sig_zarr == sig_tiff, f"{sig_zarr} != {sig_tiff}"
            assert plane.ndim == 2 and plane.size, plane.shape
            return (f"wrote + reread {regions}, read() signature identical to the TIFF "
                    f"reader {sig_tiff}, plane {plane.shape} {plane.dtype}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # IMA-255 ("A z-stack switches to 3D volume rendering with the right geometry") WAS HERE AND
    # IS GONE. It drove `ndviewer_light`'s volume monkey-patch. That package was deleted from this
    # product on 2026-07-30 (58d342d, "Delete the ndviewer_light fallback: one renderer, and a
    # named failure") precisely BECAUSE it imports PyQt5 at module scope.
    #
    # The check anticipated the deletion and raised SkipCheck on ImportError -- but the package is
    # still installed on this workstation as a sibling editable checkout, so the import SUCCEEDED,
    # dragged Qt5 into a Qt6 process ("Class QT_ROOT_LEVEL_POOL... is implemented in both PyQt6 and
    # PyQt5"), and killed the run with "QWidget: Must construct a QApplication before a QWidget".
    # Measured 2026-08-06: every check after it never ran. A skip guarded by ImportError cannot
    # protect against an import that works and is still wrong.
    #
    # Its own docstring already named the replacement, and that replacement exists:
    # `tests/test_viewer_3d.py` asserts napari's voxel scale.

    # ---------- the two features that landed on 2026-08-05 ----------------------------
    #
    # Added the same day they merged. This file's contract is "drive EVERY shipped feature", and a
    # feature that ships without a check here is exactly how the 25 above came to exist.

    @check("IMA-ii.5", "Gallery View fuses one comparable cell per (region, channel)")
    def _():
        """A gallery is a LOOK, not an operator run: one fused cell per region per channel, at a
        common decimation, with contrast SHARED per channel so a dim well and a bright well do not
        both come out looking mid-grey. Both halves are asserted -- the pixels and the sharing."""
        from squidmip._gallery import _channel_names, fuse_gallery_cell, shared_windows

        r = read(PLATE)
        m = r.metadata
        regions = sorted(m["regions"])[:2]
        ch = _channel_names(m, None)[0]
        cells = [fuse_gallery_cell(r, m, rg, sorted(m["fovs_per_region"][rg]), ch, target_px=256)
                 for rg in regions]
        assert all(c is not None for c in cells), f"a region produced no cell: {regions}"
        for c in cells:
            assert c.image.size and float(c.image.std()) > 0, \
                f"{c.region}/{c.channel} fused to a flat cell"
            assert c.n_fovs > 1, f"{c.region} is one field, not a mosaic ({c.n_fovs})"
            assert c.covered is not None and bool(c.covered.any()), "no covered pixels"
        # comparable: same channel, same decimation, so two wells are read against each other
        assert len({c.step for c in cells}) == 1, \
            f"cells decimated differently ({[c.step for c in cells]}) -- not comparable"
        win = shared_windows(cells)
        assert ch in win and win[ch][1] > win[ch][0], f"no shared window for {ch}: {win}"
        return (f"{len(cells)} cells over {regions} ch={ch}, "
                f"{cells[0].n_fovs} FOVs each at 1/{cells[0].step}, "
                f"shared window {win[ch][0]:.0f}-{win[ch][1]:.0f}")

    @check("IMA-video", "The mp4 recorder writes a movie whose frames actually DIFFER")
    def _():
        """The defect this feature can have is a movie every frame of which is identical -- which
        looks like a working feature until someone plays it. So the mp4 is DECODED BACK and the
        frames compared, on the axis `default_axis` picks for this acquisition."""
        from squidmip import _video as V

        why = V.encoder_problem()
        if why:
            raise SkipCheck(f"no encoder here: {why}")
        if free_gb() < MIN_FREE_GB:
            raise SkipCheck(f"only {free_gb():.1f} GB free; refusing to write")

        src = TISSUE if os.path.isdir(TISSUE) else PLATE
        r = read(src)
        m = r.metadata
        if not V.can_record(m):
            raise SkipCheck(f"{os.path.basename(src)} has no t or z axis to sweep")
        axis = V.default_axis(m)
        region = sorted(m["regions"])[0]

        tmp = tempfile.mkdtemp(prefix="walkthrough_movie_")
        try:
            frames = list(V.region_movie_frames(r, m, region, axis=axis))
            path, n = V.write_mp4(frames, os.path.join(tmp, "m.mp4"))
            size = os.path.getsize(path)
            assert n > 1 and size > 0, f"{n} frame(s), {size} B"

            import imageio.v3 as iio
            back = list(iio.imiter(path))
            assert len(back) == n, f"encoded {n} frames, decoded {len(back)}"
            diffs = [float(np.abs(back[i].astype(int) - back[i + 1].astype(int)).mean())
                     for i in range(len(back) - 1)]
            # NOT `> 0`. Feeding the encoder the SAME frame ten times does not decode back to ten
            # byte-identical frames -- H.264 leaves compression noise, measured at 0.0000007 to
            # 0.0046 mean abs difference per pair, and only one pair of nine came back exactly 0.
            # So `> 0` would have passed a completely static movie whenever the encoder happened
            # not to produce an exact tie. The real signal is two orders of magnitude away: 1.61
            # minimum on this z-stack, 13.93 on the 3-timepoint fixture. 0.1 sits 16x below the
            # smallest real motion and 20x above the largest artifact.
            assert diffs and min(diffs) > 0.1, (
                f"consecutive frames are effectively IDENTICAL on the {axis} axis "
                f"(min {min(diffs):.4f}, floor 0.1): {[round(d, 4) for d in diffs]}")
            # the standing rule: an operator never writes into the acquisition
            assert not os.path.realpath(path).startswith(os.path.realpath(src)), \
                "the recorder wrote inside the acquisition folder"
            return (f"{n} frames on '{axis}' from {region}, {size/1e6:.2f} MB, "
                    f"decoded back {len(back)}, min consecutive diff {min(diffs):.2f}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @check("IMA-controls", "⚙ controls: run an operator, and the chip opens ITS tab on the plate")
    def _():
        """Julio, 2026-08-06: "The controls now brings plate view, but doesn't open the operator
        tab." The whole journey, clicked: open a region window, run an operator on it from the
        plate, then press the chip and look at the plate's tab bar.

        Every step of this was already covered by a unit test and the journey was still broken,
        because the break was between the steps: the run reported success and delivered no layer,
        so ``_window_operators()`` was honestly empty and the chip had nothing to open. That is
        why this check asserts the LAYER and the TAB, not the click.
        """
        from qtpy.QtWidgets import QPushButton

        w = open_window(need(TISSUE))
        region = list(w._reader.metadata["regions"])[0]
        view = open_view(w, region)
        assert view is not None, "no region window opened"
        drain_preview(w)

        w.run_operator("mip", regions=[region], save=False, requester=view)
        drain_operator(w)
        held = view._window_operators()

        before = w._left_tabs.count()
        chip = [b for b in view.findChildren(QPushButton) if "controls" in b.text()][0]
        chip.click()
        _app().processEvents()
        titles = [w._left_tabs.tabText(i) for i in range(w._left_tabs.count())]
        opened = list(w._op_tabs)
        current = w._left_tabs.tabText(w._left_tabs.currentIndex())
        said = " ".join(view._pane.said[-3:])
        w._viewer_manager.close_all()
        w.close()

        assert held == ["mip"], (
            f"the run reported success and the window holds {held} — the layer never landed, so "
            f"the chip has nothing to open. Window said: {said!r}")
        assert opened == ["mip"], f"the chip opened {opened}, not the operator the window holds"
        assert current in titles and current != "Operators", (
            f"the tab was added but not brought to the front: current={current!r} of {titles}")
        return (f"ran mip on {region} -> window holds {held}; ⚙ controls -> "
                f"{before} tab(s) becomes {len(titles)}, focused {current!r}")

    @check("IMA-controls", "One reader, many threads: a plane read is not corrupted by a sibling")
    def _():
        """The engine reads FOVs from a thread pool through ONE reader object, and the reader
        cached one ``tifffile.TiffFile`` per file and handed it out unguarded. Two threads decoding
        two pages of the same file moved one seek position under each other.

        It surfaced as an intermittent per-field skip -- ``TiffFileError('suspicious number of
        tags')`` -- which the engine's per-well fault isolation absorbed, so a run could quietly
        lose fields and then fail to draw the region's layer at all.
        """
        import concurrent.futures as cf

        r = read(need(TISSUE))
        m = r.metadata
        region = list(m["regions"])[0]
        # SEVERAL FOVs and SEVERAL passes. The race is a seek moved between one thread's header
        # read and its strip read, so it is a probability, not a certainty: the first FOV alone,
        # read once per channel, came out clean against the unlocked reader. Mutation-checked at
        # this size — removing the lock in `_TiffHandles.read` turns this red.
        fovs = sorted(m["fovs_per_region"][region])[:3]
        channels = [c["name"] for c in m["channels"]]
        jobs = [(f, c, z) for f in fovs for c in channels
                for z in range(int(m.get("nz") or 1))] * 4

        def one(job):
            f, c, z = job
            try:
                r.read(region, f, c, z, 0)
                return None
            except Exception as exc:                 # noqa: BLE001 - the measurement
                return f"{type(exc).__name__}: {exc}"

        with cf.ThreadPoolExecutor(8) as ex:
            errs = [e for e in ex.map(one, jobs) if e]
        assert not errs, (f"{len(errs)} of {len(jobs)} concurrent reads of one file failed — the "
                          f"reader is not thread-safe: {errs[:2]}")
        return (f"{len(jobs)} concurrent reads of {region} FOVs {fovs} on 8 threads, "
                f"0 corrupted")

    # ---------- the one real write, disk-guarded and cleaned up -----------------------
    @check("IMA-222", "A stitched well SAVES end to end (then is deleted)")
    def _():
        if free_gb() < MIN_FREE_GB + 2:
            raise SkipCheck(f"only {free_gb():.1f} GB free; refusing to write")
        from squidmip import write_plate
        from squidmip._output import estimate_write_bytes
        m = read(PLATE).metadata
        est = estimate_write_bytes(m, n_fovs=None, regions=["A1"], region_operator=True)
        tmp = tempfile.mkdtemp(prefix="walkthrough_stitch_")
        try:
            write_plate(read(PLATE), tmp, projector="stitch", regions=["A1"], n_fovs=None)
            size = sum(os.path.getsize(os.path.join(d, f))
                       for d, _, fs in os.walk(tmp) for f in fs)
            return (f"wrote {size/1e9:.3f} GB (estimate {est/1e9:.3f} GB, "
                    f"guard over-predicts by {est/max(size,1):.2f}x)")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def main():
    if free_gb() < MIN_FREE_GB:
        say(f"REFUSING TO RUN: only {free_gb():.1f} GB free, need {MIN_FREE_GB} GB.")
        return 2
    say(f"disk before: {free_gb():.1f} GB free\n")
    run_all()

    width = max(len(t) for _, t, _, _ in _RESULTS)
    n_pass = n_fail = n_skip = 0
    say("=" * (width + 30))
    for ticket, title, verdict, detail in _RESULTS:
        n_pass += verdict == "PASS"; n_fail += verdict == "FAIL"; n_skip += verdict == "SKIP"
        say(f"{verdict:5} {ticket:9} {title}")
        say(f"      {detail}")
    say("=" * (width + 30))
    say(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    say(f"disk after: {free_gb():.1f} GB free")
    return 1 if n_fail else 0


if __name__ == "__main__":
    rc = main()
    # os._exit, NOT sys.exit. Measured: with every check reported and the summary printed, this
    # process still died with SIGSEGV (139) on the way out -- Qt/vispy teardown after a GUI run,
    # and nothing to do with any check's verdict. A harness whose exit code is decided by a
    # teardown crash cannot gate anything, so the verdict is committed here, after the report and
    # before the interpreter tries to unwind a widget tree it no longer needs.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
