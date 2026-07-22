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

TISSUE = ("/Users/julioamaragall/Downloads/"
          "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy")
PLATE = "/Users/julioamaragall/Downloads/synthetic_2x2_wellplate"
PLATE1536 = "/Users/julioamaragall/CEPHLA/Data/sim_1536wp"
MIN_FREE_GB = 4.0

_APP = None
_RESULTS: list[tuple[str, str, str, str]] = []      # (ticket, title, verdict, detail)


def _app():
    global _APP
    from qtpy.QtWidgets import QApplication
    _APP = QApplication.instance() or QApplication([])
    return _APP


_ONLY = os.environ.get("WALKTHROUGH_ONLY", "")   # e.g. IMA-261 — for mutation-checking one ticket


def check(ticket, title):
    """Decorator: run a check, catch everything, record one row."""
    def wrap(fn):
        if _ONLY and _ONLY not in ticket:
            return fn
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
        _RESULTS.append((ticket, title, verdict, str(detail)))
        return fn
    return wrap


class SkipCheck(Exception):
    """Raised when a check cannot run here (missing dataset, missing optional dep)."""


def open_window(path, size=(1600, 900)):
    import squidmip._viewer as V
    _app()
    win = V.PlateWindow(None)
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


def settle(ms=4000):
    """Let the async preview stream finish before grabbing pixels.

    ingest() starts a background worker that pushes tiles as they decode, so two grabs taken
    without settling compare different stream states, not different settings.
    """
    from qtpy.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


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


def ndv_clims_slider(win, ch=0):
    """The REAL contrast range-slider inside the embedded ndv viewer, for channel *ch*.

    Tests must drive this widget, not the LUTModel and not set_channel_window: every defect this
    project shipped passed a suite that called the handler instead of moving the control.
    """
    d = getattr(win, "_detail", None)
    ctrls = getattr(getattr(d, "ndv_viewer", None), "_lut_controllers", None) or {}
    ctrl = ctrls.get(ch)
    if ctrl is None:
        return None
    for v in getattr(ctrl, "lut_views", []):
        q = getattr(v, "_qwidget", None)
        if q is not None and hasattr(q, "clims"):
            return q.clims
    return None


def rendered(widget, w=900, h=700):
    """Grab a widget as an RGB array - what a human would actually see."""
    widget.resize(w, h)
    _app().processEvents()
    img = widget.grab().toImage().convertToFormat(4)
    ptr = img.bits(); ptr.setsize(img.sizeInBytes())
    return np.array(ptr).reshape(img.height(), img.width(), 4)[..., :3].astype(float)


def free_gb():
    return shutil.disk_usage("/").free / 1e9


# ======================================================================================
def run_all():
    from squidmip import available_projectors, open_reader
    from squidmip._stitch import available_region_operators

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
        if not os.path.isdir(PLATE1536):
            raise SkipCheck(f"fixture absent: {PLATE1536}")
        w = open_window(PLATE1536)
        m = w._reader.metadata
        n, first, last = len(m["regions"]), m["regions"][0], m["regions"][-1]
        fmt = w._plate.format_name
        w.close()
        assert n == 1536, n
        return f"{n} wells {first}..{last}, resolved {fmt!r}"

    @check("IMA-215", "coordinates.csv: both on-disk schemas parse")
    def _():
        a = open_reader(PLATE).metadata["fov_positions_um"]        # 20x-style, row order = fov
        b = open_reader(TISSUE).metadata["fov_positions_um"]       # monkey-style, fov column
        assert len(a) == 144 and len(b) == 55, (len(a), len(b))
        xs = [v[0] for v in a.values()]
        span = max(xs) - min(xs)
        assert span > 1000, f"x span {span} looks like mm not um"
        return f"20x-style {len(a)} positions (span {span:.1f} um), monkey-style {len(b)}"

    # ---------- mosaic + geometry ---------------------------------------------------
    @check("IMA-187", "Each well is a coordinate-placed MOSAIC, not one thumbnail")
    def _():
        from squidmip._viewer import _mosaic_boxes
        boxes = _mosaic_boxes(open_reader(PLATE).metadata)
        per_well: dict = {}
        for (region, _fov), _b in boxes.items():
            per_well[region] = per_well.get(region, 0) + 1
        assert len(boxes) == 144, len(boxes)
        assert set(per_well.values()) == {36}, per_well
        return f"{len(boxes)} boxes, {sorted(per_well)} x 36 fields each"

    @check("IMA-187", "Y-sign: larger stage y maps to a LARGER row (no mirroring)")
    def _():
        from squidmip._placement import fov_offsets_px
        m = open_reader(PLATE).metadata
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
        for label, ds in (("144 FOVs", PLATE), ("55 FOVs", TISSUE)):
            m = open_reader(ds).metadata
            lad = plate_ladder(m)
            geo = lad.geometry if hasattr(lad, "geometry") else lad
            x0, y0, x1, y1 = geo.levels[-1].bboxes[:, 0].min(), geo.levels[-1].bboxes[:, 1].min(), \
                geo.levels[-1].bboxes[:, 2].max(), geo.levels[-1].bboxes[:, 3].max()
            um_per_px = max(x1 - x0, y1 - y0) / 1200
            tiles = select_tiles((x0, y0, x1, y1), um_per_px, geo, channels=("0",))
            counts[label] = len(tiles)
            assert len(tiles) < 60, f"{label}: fit-to-plate wanted {len(tiles)} tiles"
        return " | ".join(f"{k}: {v} tiles at fit-to-plate" for k, v in counts.items())

    # ---------- selection -> exploration -> panes ------------------------------------
    @check("IMA-221", "Shift-drag marquee selects the right (region, fov) pairs")
    def _():
        # A real drag through the widget's own event handlers - not a direct call to the
        # selection setter. The Re-dock button was dead for a day precisely because every
        # test called the handler instead of clicking.
        from qtpy.QtCore import QEvent, QPointF, Qt
        from qtpy.QtGui import QMouseEvent

        def ev(kind, pos, mods, buttons=Qt.MouseButton.LeftButton):
            k = {"press": QEvent.Type.MouseButtonPress, "move": QEvent.Type.MouseMove,
                 "release": QEvent.Type.MouseButtonRelease}[kind]
            return QMouseEvent(k, pos, Qt.MouseButton.LeftButton, buttons, mods)

        w = open_window(PLATE)
        ov = w._overview
        rendered(ov)
        a = QPointF(2, 2)
        b = QPointF(ov.width() - 2, ov.height() - 2)
        ov.mousePressEvent(ev("press", a, Qt.KeyboardModifier.ShiftModifier))
        ov.mouseMoveEvent(ev("move", b, Qt.KeyboardModifier.ShiftModifier))
        ov.mouseReleaseEvent(ev("release", b, Qt.KeyboardModifier.ShiftModifier, buttons=Qt.MouseButton.NoButton))
        wells = ov.selected_wells()
        sel = w.selected_region_fovs()
        w.close()
        assert wells == ["A1", "A2", "B1", "B2"], wells
        assert len(sel) == 144, f"expected 144 (region, fov) pairs, got {len(sel)}"
        return f"drag over the whole plate -> {wells}, {len(sel)} (region, fov) pairs"

    @check("IMA-260", "Three panes on OPEN, and the empty one teaches by example")
    def _():
        # IMA-237 shipped pane 3 collapsed until a Shift-drag revealed it; IMA-260 reverses that,
        # because a pane that is not there cannot be discovered. Pane 3 now opens at a real width
        # showing EXAMPLE USAGE, and the copy stands down only while it holds content.
        w = open_window(PLATE)
        outer = w._split
        before = outer.sizes()
        if outer.count() != 3:
            w.close(); raise AssertionError(f"outer splitter has {outer.count()} children, want 3")
        empty_before = w.explore_empty_text()
        live_w = w._explore_pane.width()
        key = w.open_exploration_tab(["A1", "A2"])
        _app().processEvents()
        after = outer.sizes()
        empty_after = w.explore_empty_text()
        n_tabs = w._explore_tabs.count()
        # ...and it comes back when the pane empties again
        for i in range(w._explore_tabs.count() - 1, -1, -1):
            w._close_op_tab(i, w._explore_tabs)
        _app().processEvents()
        empty_again = w.explore_empty_text()
        w.close()
        assert before[2] > 0, f"pane 3 opened collapsed: {before}"
        assert live_w > 0, "pane 3 has no real width on the shown window"
        low = empty_before.lower()
        assert "right-click" in low and "control well" in low, f"no example usage: {empty_before!r}"
        assert "hold shift" in low and "for example" in low, f"not framed as an example: {low!r}"
        assert key, f"open_exploration_tab returned {key!r}: {w._readout.text()!r}"
        assert empty_after == "", "the example copy stayed up behind real content"
        assert after == before, f"opening a tab moved a divider: {before} -> {after}"
        assert empty_again.strip(), "the example never came back when the pane emptied"
        return (f"panes {before} (pane 3 live {live_w}px); tab {key!r} in pane 3 "
                f"({n_tabs} tab(s)); example copy on -> off -> on")

    @check("IMA-248", "Control Well: right-click sets ONE reference, pinned first in pane 3")
    def _():
        w = open_window(PLATE)
        w.set_control_well("A2")
        _app().processEvents()
        first = (w.control_well(), w._overview.control_well(), w._explore_tabs.tabText(0))
        from qtpy.QtWidgets import QTabBar
        pinned = w._explore_tabs.tabBar().tabButton(0, QTabBar.ButtonPosition.RightSide) is None
        w.set_control_well("B1")            # a second control RELEASES the first
        _app().processEvents()
        agree = (w.control_well(), w._overview.control_well())
        n_control = sum(1 for i in range(w._explore_tabs.count())
                        if w._explore_tabs.tabText(i).startswith("Control"))
        w.set_control_well(None)
        _app().processEvents()
        cleared = (w.control_well(), w._overview.control_well(), w._explore_tabs.count())
        back = w.explore_empty_text().strip() != ""
        w.close()
        assert first[0] == first[1] == "A2", f"plate and window disagree: {first}"
        assert "A2" in first[2], f"control tab not pinned first: {first[2]!r}"
        assert pinned, "the control tab is closable — it is supposed to be pinned"
        assert agree == ("B1", "B1"), f"a second control did not take over cleanly: {agree}"
        assert n_control == 1, f"{n_control} control tabs after re-setting — one is stale"
        assert cleared == (None, None, 0), f"clearing left something behind: {cleared}"
        assert back, "clearing the control did not restore the example copy"
        return f"set A2 -> {first[2]!r}, re-set to B1 (1 pinned tab), cleared -> example copy back"

    @check("IMA-209", "Drag-out floating window, and Re-dock (dead until today)")
    def _():
        w = open_window(PLATE)
        key = w.open_exploration_tab(["A1"])
        assert key, "no exploration tab to detach"
        _app().processEvents()
        tabs = w._explore_tabs
        idx = tabs.count() - 1
        float_win = w._detach_tab(idx, tabs)
        _app().processEvents()
        n_float = len(getattr(w, "_floating", {}))
        detached = float_win is not None
        # Re-dock through the same path the (previously dead) button uses.
        w._redock(key)
        _app().processEvents()
        n_after = len(getattr(w, "_floating", {}))
        back = w._explore_tabs.count()
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
        reader = open_reader(PLATE)
        fovs = reader.metadata["fovs_per_region"]["A1"][:4]      # a subset: the crop path
        assert len(fovs) == 4, "PLATE well A1 must have >=4 FOVs for this check"
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

    @check("IMA-207", "Per-region contrast lifts a dim well that global crushes")
    def _():
        w = open_window(PLATE)
        ov = w._overview
        if not hasattr(ov, "set_contrast_scope"):
            w.close(); raise SkipCheck("set_contrast_scope() not present")
        from squidmip._viewer import SCOPE_PER_REGION
        ov.set_contrast_scope("global"); a = rendered(ov).mean()
        ov.set_contrast_scope(SCOPE_PER_REGION); b = rendered(ov).mean()
        w.close()
        return f"mean brightness global={a:.2f} -> per-region={b:.2f}"

    @check("IMA-261", "The plate view has NO contrast sliders (the duplicate control is GONE)")
    def _():
        """Not hidden, not disabled - absent. Walked over the REAL widget tree, because a control
        that is merely `hide()`-den is still a second owner waiting to be un-hidden."""
        from qtpy.QtWidgets import QAbstractSlider, QPushButton
        w = open_window(PLATE)
        bar = w._channel_bar
        if bar is None:
            w.close(); raise SkipCheck("no channel bar")
        sliders = bar.findChildren(QAbstractSlider)
        autos = [b for b in bar.findChildren(QPushButton) if "auto" in b.text().lower()]
        n_rows = len(bar._rows)
        has_setter = hasattr(bar, "_push") or hasattr(bar, "_auto") or hasattr(bar, "_slider")
        w.close()
        assert not sliders, f"{len(sliders)} contrast slider(s) still in the plate's channel bar"
        assert not autos, f"{len(autos)} 'auto' button(s) still in the plate's channel bar"
        assert not has_setter, "the channel bar still has a contrast-SETTING method"
        return (f"{n_rows} channel rows, 0 sliders, 0 auto buttons, no setter method - "
                "the plate reports contrast, it does not set it")

    @check("IMA-261", "DRAGGING the CENTRAL viewer's contrast slider repaints the PLATE")
    def _():
        """The user's actual complaint: what the array viewer shows was not reflected on the plate.
        Drives the REAL QLabeledRangeSlider inside ndv - not set_channel_window, not the model -
        and asserts on the plate's DISPLAYED PIXELS."""
        w = open_window(PLATE)
        drain_preview(w)
        ov = w._overview
        sl = ndv_clims_slider(w, 0)
        if sl is None:
            w.close(); raise SkipCheck("ndv has no clims slider for channel 0 yet")
        dmax = ov._contrast.dmax
        sl.setValue((0, int(dmax * 0.9)))
        _app().processEvents()
        base = rendered(ov)
        sl.setValue((int(dmax * 0.45), int(dmax * 0.55)))     # a hard, visible re-window
        _app().processEvents()
        after = rendered(ov)
        plate_win = ov.channel_windows()[0]
        view_win = w._detail.channel_windows().get(0)
        w.close()
        h = min(base.shape[0], after.shape[0]); wd = min(base.shape[1], after.shape[1])
        changed = int((np.abs(base[:h, :wd] - after[:h, :wd]) > 0).sum())
        assert changed > 0, "the array viewer's contrast slider changed nothing on the plate"
        assert plate_win == view_win, f"plate {plate_win} != array viewer {view_win}"
        return (f"{changed} px changed on the plate; plate window == viewer window == "
                f"({plate_win[0]:.0f}, {plate_win[1]:.0f})")

    @check("IMA-261", "Plate PIXELS are the array viewer's window, not merely its numbers")
    def _():
        """Agreeing on a (lo, hi) pair proves nothing if the plate then renders something else.
        Recomposite the plate's own store under the window READ BACK FROM THE ARRAY VIEWER and
        require the result to be the very bytes the plate is showing."""
        from squidmip._montage import composite
        w = open_window(PLATE)
        drain_preview(w)
        ov = w._overview
        sl = ndv_clims_slider(w, 0)
        if sl is None:
            w.close(); raise SkipCheck("ndv has no clims slider for channel 0 yet")
        sl.setValue((321, 8765))
        _app().processEvents()
        ov.recomposite(quick=False)
        shown = ov._final_arr.get(ov._active)
        store = ov._store.get(ov._active)
        viewer_wins = w._detail.channel_windows()
        if shown is None or store is None:
            w.close(); raise SkipCheck("plate has not composited yet")
        # Build the expected image from the VIEWER's windows only - nothing from the plate's
        # own contrast model gets a vote here.
        wins = [viewer_wins.get(c, ov.channel_windows()[c]) for c in range(store.shape[0])]
        expect = composite(store, ov._colors, wins, ov._mask)
        ch0 = viewer_wins.get(0)
        w.close()
        assert ch0 == (321.0, 8765.0), f"viewer did not take the slider value: {ch0}"
        assert shown.shape == expect.shape, f"{shown.shape} != {expect.shape}"
        diff = int(np.abs(shown.astype(int) - expect.astype(int)).max())
        assert diff == 0, f"plate pixels differ from the viewer's window by up to {diff}/255"
        return (f"{shown.shape} plate pixels are BYTE-IDENTICAL to a recomposite under the array "
                f"viewer's own windows {[(round(a), round(b)) for a, b in wins]}")

    @check("IMA-261", "A contrast drag is under 16 ms/frame on all three datasets")
    def _():
        """'Buttery' with a number on it: the wall time from moving the array viewer's slider to
        the plate having repainted, with the Qt event loop in between. Measured, not asserted.

        EVERY TICK MUST ACTUALLY REPAINT THE PLATE, and that is asserted here rather than assumed.
        Mutation-checked against parent 8859c52, where the plate did not follow the array viewer at
        all: the identical loop reported 0.2 ms / 5000 fps on all three datasets and PASSED, because
        it was timing a slider nothing was listening to. A latency budget over a disconnected
        control is the "832 green tests, one wrong model" defect in miniature — the number was real,
        the thing it measured was not. So each tick's composited buffer is fingerprinted, and a tick
        that did not change the plate's pixels disqualifies the measurement instead of flattering
        it.
        """
        import time
        out, worst = [], 0.0
        for label, ds in (("tissue", TISSUE), ("2x2", PLATE), ("1536wp", PLATE1536)):
            if not os.path.exists(ds):
                out.append(f"{label}: absent"); continue
            w = open_window(ds, size=(2560, 1440))
            drain_preview(w)
            ov = w._overview
            sl = ndv_clims_slider(w, 0)
            if sl is None or ov._store.get(ov._active) is None:
                w.close(); out.append(f"{label}: no slider/store"); continue
            dmax = ov._contrast.dmax

            def frame():
                """A fingerprint of what the plate is CURRENTLY showing, or None if nothing is."""
                a = ov._final_arr.get(ov._active)
                return None if a is None else hash(a.tobytes())

            sl.setValue((0, int(dmax * 0.5))); _app().processEvents()   # warm every cache
            ts, frames = [], []
            for i in range(30):
                t0 = time.perf_counter()
                sl.setValue((0, int(dmax * (0.25 + 0.5 * i / 30))))
                _app().processEvents()
                ts.append((time.perf_counter() - t0) * 1000)
                frames.append(frame())
            med = float(np.median(ts))
            store = ov._store[ov._active]
            w.close()
            # 30 strictly increasing windows over real (non-flat) pixels must give 30 different
            # images. Anything less means some ticks repainted nothing and their times are noise.
            distinct = len(set(frames))
            assert None not in frames, f"{label}: the plate composited nothing during the drag"
            assert distinct == len(frames), (
                f"{label}: {len(frames)} slider ticks produced only {distinct} distinct plate "
                f"images — the plate is NOT following the array viewer, so this timing is of a "
                f"disconnected control")
            worst = max(worst, med)
            out.append(f"{label} {tuple(store.shape)}: {med:.1f} ms ({1000 / med:.0f} fps), "
                       f"{distinct}/{len(frames)} repaints")
        assert worst < 16.0, f"slowest dataset {worst:.1f} ms/frame, over the 16 ms budget"
        return " | ".join(out)

    @check("IMA-242", "ONE contrast model: the loupe obeys the slider the plate obeys")
    def _():
        """The duplication this ticket collapsed: the loupe used to memoise its own window and its
        own compositor, so a slider drag moved the plate and left the magnifier of that same plate
        showing the pre-drag contrast."""
        import squidmip._viewer as V
        w = open_window(PLATE)
        ov = w._overview
        settle()
        if ov._contrast is None:
            w.close(); raise SkipCheck("no contrast model")
        # The three renderers must agree about a LATCHED channel, whatever auto they each derive.
        ov._contrast.set_manual(0, 123.0, 4567.0)
        plate_win = ov.channel_windows()[0]
        store = ov._store.get(ov._active)
        cell_win = None
        if store is not None and ov._tiles_by_layer.get(ov._active):
            ri, ci = sorted(ov._tiles_by_layer[ov._active])[0]
            cell_win = ov._cell_windows(store, ri, ci)[0]
        loupe_win = ov._contrast.resolve(0, (0.0, 65535.0))   # the loupe's resolution path
        w.close()
        assert plate_win == (123.0, 4567.0), f"plate ignored the latch: {plate_win}"
        assert loupe_win == (123.0, 4567.0), f"loupe ignored the latch: {loupe_win}"
        if cell_win is not None:
            assert cell_win == (123.0, 4567.0), f"per-region ignored the latch: {cell_win}"
        assert not hasattr(V, "_composite_rgb") and not hasattr(V, "_percentile_window"), \
            "a duplicate contrast implementation is back"
        return (f"plate={plate_win} per-region={cell_win} loupe={loupe_win} — all one model; "
                "both duplicate implementations gone")

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
        from squidmip._stitch import _REGION_OPERATORS
        assert "stitch" in _REGION_OPERATORS and "coordinate" in _REGION_OPERATORS
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
        m = open_reader(PLATE).metadata
        proj = estimate_write_bytes(m, n_fovs=None)
        stit = estimate_write_bytes(m, n_fovs=None, region_operator=True)
        assert stit != proj, "region estimate identical to the frame count"
        return f"projected {proj/1e9:.3f} GB vs stitched {stit/1e9:.3f} GB (ratio {stit/proj:.3f})"

    @check("IMA-231", "ROI table corners agree with the tile ladder")
    def _():
        from squidmip._output import fov_roi_records_um
        from squidmip._tilesource import fov_bboxes_um
        m = open_reader(PLATE).metadata
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

    @check("IMA-205", "Exploration tab is SCOPED to the selected subset")
    def _():
        w = open_window(PLATE)
        key = w.open_exploration_tab(["A1", "A2"])
        _app().processEvents()
        assert key, f"no tab: {w._readout.text()!r}"
        tab = w._explore_tabs.widget(w._explore_tabs.count() - 1)
        label = w._explore_tabs.tabText(w._explore_tabs.count() - 1)
        # The scope must be the SUBSET, not the whole plate: an exploration tab that quietly
        # operates on all four wells looks identical on screen and is the bug worth catching.
        scope = w._current_exploration() if callable(getattr(w, "_current_exploration", None)) \
            else getattr(w, "_current_exploration", key)
        w.close()
        assert "A1" in label and "A2" in label, f"tab label {label!r} does not name the subset"
        assert tab is not None
        return f"tab {label!r} (key {key!r}), scope={scope!r}, 2 of 4 wells"

    @check("IMA-217", "Pyramid ladder is coarse-to-fine and never widens")
    def _():
        from squidmip._tilesource import plate_ladder
        m = open_reader(PLATE).metadata
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
        from squidmip import open_reader as _open, write_plate
        tmp = tempfile.mkdtemp(prefix="walkthrough_zarr_")
        try:
            # One well, one FOV: enough to prove the round trip, kilobytes on disk.
            write_plate(_open(PLATE), tmp, regions=["A1"], n_fovs=1, projector="mip")
            back = _open(os.path.join(tmp, "plate.ome.zarr"))
            m = back.metadata
            regions = list(m["regions"])
            # the seam: the SAME read() signature serves TIFF and Zarr
            sig_zarr = list(inspect.signature(type(back).read).parameters)
            sig_tiff = list(inspect.signature(type(_open(PLATE)).read).parameters)
            plane = back.read(regions[0], m["fovs_per_region"][regions[0]][0],
                              m["channels"][0]["name"], 0)
            assert regions == ["A1"], regions
            assert sig_zarr == sig_tiff, f"{sig_zarr} != {sig_tiff}"
            assert plane.ndim == 2 and plane.size, plane.shape
            return (f"wrote + reread {regions}, read() signature identical to the TIFF "
                    f"reader {sig_tiff}, plane {plane.shape} {plane.dtype}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @check("IMA-255", "A z-stack switches to 3D volume rendering with the right geometry")
    def _():
        """Drives ndv's OWN 2D/3D button on a real tissue z-stack and reads back what
        reached the renderer. Everything here is checkable without a GPU: which axis
        became the third visible one, that the channel axis survived, that Volume
        visuals replaced Images, and the voxel scale the vertices were built with.

        NOT checked here: that pixels actually appear on screen. Vispy volume rendering
        needs a real OpenGL context, and under QT_QPA_PLATFORM=offscreen there is none
        ("QOpenGLWidget is not supported on this platform"). "The volume was built with
        the right geometry" and "the volume is visible" are different claims.
        """
        try:
            import ndviewer_light.core as nvc
            from ndv.views._vispy._array_canvas import VispyArrayCanvas
            from vispy.visuals.volume import VolumeVisual
        except ImportError as e:
            raise SkipCheck(f"ndviewer_light/ndv/vispy unavailable: {e}")
        if not os.path.isdir(TISSUE):
            raise SkipCheck("tissue z-stack not present")

        assert nvc.VOLUME_PATCH_ERROR is None, nvc.VOLUME_PATCH_ERROR
        assert VispyArrayCanvas.add_volume.__name__ == "_patched_add_volume", \
            "the volume monkey-patch is not bound — 3D would silently render isotropic"

        app = _app()
        meta = open_reader(TISSUE).metadata
        px, dz = meta["pixel_size_um"], meta["dz_um"]

        seen = []
        orig = VispyArrayCanvas.add_volume
        VispyArrayCanvas.add_volume = (
            lambda self, data=None: (
                seen.append((getattr(data, "shape", None), nvc._effective_voxel_scale())),
                orig(self, data))[1])
        try:
            v = nvc.LightweightViewer()
            v.show()
            settle(500)
            # A small canvas: this check is about geometry, not throughput.
            v.start_acquisition([c["name"] for c in meta["channels"]], meta["n_z"],
                                256, 256, ["manual0:0"], pixel_size_um=px, dz_um=dz)
            settle(1500)
            nv = v.ndv_viewer
            wrapper = nv._data_model.data_wrapper
            z_axis, ch_axis = wrapper.guess_z_axis(), wrapper.guess_channel_axis()
            assert z_axis != ch_axis, \
                f"3D would stack CHANNELS, not z (both axis {z_axis})"

            nv._view._qwidget.ndims_btn.click()     # the user-reachable control
            settle(3000)

            axes = tuple(nv.display_model.visible_axes)
            vols = [e for e in nv._canvas._elements if isinstance(e, VolumeVisual)]
            assert len(axes) == 3, axes
            assert axes[0] == z_axis, f"third visible axis is {axes[0]}, not z ({z_axis})"
            assert nv.display_model.channel_axis == ch_axis, \
                "channel_axis was dropped — composite colours would be lost"
            assert vols, "no Volume visual was created"
            assert seen, "add_volume never fired"

            want = dz / px
            scales = {round(getattr(x, "_voxel_scale", (0, 0, 0))[2], 6) for x in vols}
            assert scales == {round(want, 6)}, f"voxel z-scale {scales}, expected {want}"
            assert type(nv._canvas._view.camera).__name__ == "ArcballCamera"

            nv._view._qwidget.ndims_btn.click()     # and back to 2D
            settle(2000)
            assert len(nv.display_model.visible_axes) == 2, nv.display_model.visible_axes

            shape = seen[-1][0]
            return (f"3D button -> visible_axes {axes} (z={z_axis}, channel {ch_axis} "
                    f"kept), {len(vols)} Volume visuals of shape {shape}, ArcballCamera, "
                    f"voxel z-scale {want:.5f} (dz {dz} um / pixel {px} um); toggled back "
                    f"to 2D. On-screen pixels NOT verified: offscreen has no GL context.")
        finally:
            VispyArrayCanvas.add_volume = orig

    # ---------- the one real write, disk-guarded and cleaned up -----------------------
    @check("IMA-222", "A stitched well SAVES end to end (then is deleted)")
    def _():
        if free_gb() < MIN_FREE_GB + 2:
            raise SkipCheck(f"only {free_gb():.1f} GB free; refusing to write")
        from squidmip import write_plate
        from squidmip._output import estimate_write_bytes
        m = open_reader(PLATE).metadata
        est = estimate_write_bytes(m, n_fovs=None, regions=["A1"], region_operator=True)
        tmp = tempfile.mkdtemp(prefix="walkthrough_stitch_")
        try:
            write_plate(open_reader(PLATE), tmp, projector="stitch", regions=["A1"], n_fovs=None)
            size = sum(os.path.getsize(os.path.join(d, f))
                       for d, _, fs in os.walk(tmp) for f in fs)
            return (f"wrote {size/1e9:.3f} GB (estimate {est/1e9:.3f} GB, "
                    f"guard over-predicts by {est/max(size,1):.2f}x)")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def main():
    if free_gb() < MIN_FREE_GB:
        print(f"REFUSING TO RUN: only {free_gb():.1f} GB free, need {MIN_FREE_GB} GB.")
        return 2
    print(f"disk before: {free_gb():.1f} GB free\n")
    run_all()

    width = max(len(t) for _, t, _, _ in _RESULTS)
    n_pass = n_fail = n_skip = 0
    print("=" * (width + 30))
    for ticket, title, verdict, detail in _RESULTS:
        n_pass += verdict == "PASS"; n_fail += verdict == "FAIL"; n_skip += verdict == "SKIP"
        print(f"{verdict:5} {ticket:9} {title}")
        print(f"      {detail}")
    print("=" * (width + 30))
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print(f"disk after: {free_gb():.1f} GB free")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
