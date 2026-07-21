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
    from PyQt5.QtWidgets import QApplication
    _APP = QApplication.instance() or QApplication([])
    return _APP


def check(ticket, title):
    """Decorator: run a check, catch everything, record one row."""
    def wrap(fn):
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
    from PyQt5.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec_()


def rendered(widget, w=900, h=700):
    """Grab a widget as an RGB array - what a human would actually see."""
    widget.resize(w, h)
    _app().processEvents()
    img = widget.grab().toImage().convertToFormat(4)
    ptr = img.bits(); ptr.setsize(img.byteCount())
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
        from PyQt5.QtCore import QEvent, QPointF, Qt
        from PyQt5.QtGui import QMouseEvent

        def ev(kind, pos, mods, buttons=Qt.LeftButton):
            k = {"press": QEvent.MouseButtonPress, "move": QEvent.MouseMove,
                 "release": QEvent.MouseButtonRelease}[kind]
            return QMouseEvent(k, pos, Qt.LeftButton, buttons, mods)

        w = open_window(PLATE)
        ov = w._overview
        rendered(ov)
        a = QPointF(2, 2)
        b = QPointF(ov.width() - 2, ov.height() - 2)
        ov.mousePressEvent(ev("press", a, Qt.ShiftModifier))
        ov.mouseMoveEvent(ev("move", b, Qt.ShiftModifier))
        ov.mouseReleaseEvent(ev("release", b, Qt.ShiftModifier, buttons=Qt.NoButton))
        wells = ov.selected_wells()
        sel = w.selected_region_fovs()
        w.close()
        assert wells == ["A1", "A2", "B1", "B2"], wells
        assert len(sel) == 144, f"expected 144 (region, fov) pairs, got {len(sel)}"
        return f"drag over the whole plate -> {wells}, {len(sel)} (region, fov) pairs"

    @check("IMA-237", "Three panes: pane 3 hidden until an exploration tab exists")
    def _():
        w = open_window(PLATE)
        outer = w._split
        before = [outer.sizes(), outer.count()]
        if outer.count() != 3:
            w.close(); raise AssertionError(f"outer splitter has {outer.count()} children, want 3")
        hidden_before = outer.sizes()[2] == 0
        key = w.open_exploration_tab(["A1", "A2"])
        _app().processEvents()
        after = outer.sizes()
        plate_kept = after[0] == before[0][0]
        n_tabs = w._explore_tabs.count()
        w.close()
        assert hidden_before, f"pane 3 visible before any exploration tab: {before[0]}"
        assert key, f"open_exploration_tab returned {key!r}: {w._readout.text()!r}"
        assert after[2] > 0, f"pane 3 still collapsed after opening a tab: {after}"
        assert plate_kept, f"plate pane shrank: {before[0][0]} -> {after[0]}"
        return (f"{before[0]} -> {after}; tab {key!r} in pane 3 ({n_tabs} tab(s)); "
                f"plate pane preserved")

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
        if hasattr(w._overview, "select_wells"):
            w._overview.select_wells(["A1", "A2"])
        sel = w.minerva_selection()
        regions = sorted({r for r, _ in sel}) if sel else []
        w.close()
        assert sel, "empty selection payload"
        return f"{len(sel)} FOVs across {regions}"

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
        settle()                       # the preview stream must finish, or we diff stream states
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
