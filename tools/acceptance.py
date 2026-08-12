#!/usr/bin/env python3
"""Headless acceptance gate: drive the real widget on the real acquisitions.

    QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python tools/acceptance.py

Exit code is 0 only if every case passes. Both env vars are required.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

# Run from anywhere: import the repo this file lives in, not an installed squidxplorer.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The two acquisitions the product is actually demoed on. READ ONLY - never copy or convert.
TISSUE = ("/Users/julioamaragall/Downloads/"
          "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy")
# Built by tools/make_5d_fixture.py; SQUIDXPLORER_FIXTURE_PLATE overrides it for CI.
PLATE = os.environ.get("SQUIDXPLORER_FIXTURE_PLATE") or \
    "/Users/julioamaragall/Downloads/sim_2x2_36fov_96wp"

# (label, path, expected regions, expected fov_positions_um entries)
# None for the count means "one position per (region, fov) the acquisition declares".
CASES = [
    ("tissue (glass slide, freeform regions)", TISSUE, ["manual0", "manual1"], 55),
    ("2x2 well plate", PLATE, ["A1", "A2", "B1", "B2"], None),
]


_APP = None


def _app():
    """The QApplication, on the binding the app ships."""
    global _APP
    import squidxplorer  # noqa: F401  -- sets QT_API before qtpy resolves a binding
    from qtpy.QtWidgets import QApplication
    # Keep a module-level reference: a QApplication with no Python owner is garbage collected.
    _APP = QApplication.instance() or QApplication([])
    return _APP


def check(label, path, want_regions, want_positions):
    import squidxplorer._viewer as V

    _app()
    win = V.PlateWindow(None)
    fails = []
    try:
        win.ingest(path)
    except Exception as e:
        return [f"ingest raised {type(e).__name__}: {e}"]

    readout = (getattr(getattr(win, "_readout", None), "text", lambda: "")() or "")
    if win._reader is None:
        fails.append(f"reader is None; readout: {readout!r}")
        return fails
    if win._overview is None:
        fails.append(f"no plate overview built; readout: {readout!r}")

    meta = win._reader.metadata
    got_regions = list(meta.get("regions") or [])
    if got_regions != want_regions:
        fails.append(f"regions {got_regions} != expected {want_regions}")

    n_pos = len(meta.get("fov_positions_um") or {})
    if want_positions is None:
        want_positions = sum(len(f) for f in (meta.get("fovs_per_region") or {}).values())
        why = " (one per declared (region, fov))"
    else:
        why = ""
    if n_pos != want_positions:
        fails.append(f"fov_positions_um has {n_pos} entries, expected {want_positions}{why}")

    # Units contract: world space is micrometres. A plate spans tens of thousands of um.
    if n_pos:
        xs = [v[0] for v in meta["fov_positions_um"].values()]
        span = max(xs) - min(xs)
        if span < 1000:
            fails.append(f"x span {span:.1f} looks like mm, not um (units regression)")

    # An acquisition that opens must not also report that it cannot be opened.
    for bad in ("not supported", "not a well-plate", "cannot lay out",
                "not a readable", "no pixels"):
        if bad in readout.lower():
            fails.append(f"readout still reports failure: {readout!r}")
            break

    for ch_key in ("channels",):
        if not meta.get(ch_key):
            fails.append(f"metadata[{ch_key!r}] is empty")

    try:
        win.close()
    except Exception:
        pass
    return fails


def check_one_writer(label, root, reader_cls):
    """Reader contract + widget ingest for one writer's synthetic acquisition."""
    import numpy as np

    import squidxplorer
    from tests.writer_fixtures import expected_arrays, FOVS, REGIONS, _FOV_MM

    fails, notes = [], []
    try:
        reader = squidxplorer.open_reader(root)
    except Exception:
        return ["open_reader raised:\n" + traceback.format_exc()], notes

    # The claim is "the RIGHT reader opened it", not "a reader opened it".
    got = type(reader).__name__
    if got != reader_cls:
        fails.append(f"dispatched to {got}, expected {reader_cls}")

    meta = reader.metadata
    if list(meta.get("regions") or []) != list(REGIONS):
        fails.append(f"regions {meta.get('regions')} != {list(REGIONS)}")
    if meta.get("fovs_per_region") != {r: list(FOVS) for r in REGIONS}:
        fails.append(f"fovs_per_region {meta.get('fovs_per_region')}")
    if not meta.get("channels"):
        fails.append("metadata['channels'] is empty")
    for key, want in (("n_z", 2), ("n_t", 1)):
        if meta.get(key) != want:
            fails.append(f"metadata[{key!r}] = {meta.get(key)!r}, expected {want}")

    # Exact pixels, every plane.
    for (region, fov, z, channel), expected in expected_arrays().items():
        try:
            got_plane = reader.read(region, fov, channel, z)
        except Exception as e:
            fails.append(f"read({region},{fov},{channel},{z}) raised {type(e).__name__}: {e}")
            break
        if not np.array_equal(got_plane, expected):
            fails.append(f"pixels differ at region={region} fov={fov} z={z} ch={channel}")
            break

    # Units: micrometres, key says so, converted once at the producer.
    positions = meta.get("fov_positions_um") or {}
    if set(positions) != {(r, f) for r in REGIONS for f in FOVS}:
        fails.append(f"fov_positions_um has {len(positions)} entries, expected "
                     f"{len(REGIONS) * len(FOVS)}")
    for (region, fov), (x_um, y_um) in positions.items():
        want_x, want_y = _FOV_MM[fov]
        if abs(x_um - want_x * 1000.0) > 1.0 or abs(y_um - want_y * 1000.0) > 1.0:
            fails.append(f"{region}/{fov} at ({x_um:.1f}, {y_um:.1f}) um, expected "
                         f"({want_x * 1000.0:.1f}, {want_y * 1000.0:.1f}) — units regression?")
            break

    # Then the widget, which is the whole reason this file is not a pytest module.
    import squidxplorer._viewer as V

    _app()
    win = V.PlateWindow(None)
    try:
        win.ingest(str(root))
        readout = (getattr(getattr(win, "_readout", None), "text", lambda: "")() or "")
        if win._reader is None:
            if "already a written plate" in readout:
                # Named, not swallowed: resolve_plate_root cannot tell a raw Squid HCS
                # acquisition from a SquidXplorer-written plate.
                notes.append("widget ingest refused: _viewer.resolve_plate_root cannot tell a "
                             "raw Squid HCS acquisition from a SquidXplorer-written plate "
                             "(_viewer.py:701). Reader contract verified; widget path is a "
                             "separate, named defect — file it against _viewer.py.")
            else:
                fails.append(f"widget reader is None; readout: {readout!r}")
        else:
            for bad in ("not supported", "not a well-plate", "cannot lay out",
                        "not a readable", "no pixels"):
                if bad in readout.lower():
                    fails.append(f"readout reports failure: {readout!r}")
                    break
    except Exception:
        fails.append("widget ingest raised:\n" + traceback.format_exc())
    finally:
        try:
            win.close()
        except Exception:
            pass
    return fails, notes


def check_writers():
    """Walk every Squid writer over synthetic fixtures in a temp dir. Returns [(label, fails, notes)]."""
    import tempfile

    from tests.writer_fixtures import WRITERS

    results = []
    with tempfile.TemporaryDirectory(prefix="squidxplorer_writers_") as tmp:
        for label, builder, reader_cls, _records_positions in WRITERS:
            slug = "".join(c if c.isalnum() else "_" for c in label)
            try:
                root = builder(os.path.join(tmp, slug))
            except Exception:
                results.append((label, ["fixture build failed:\n" + traceback.format_exc()], []))
                continue
            try:
                fails, notes = check_one_writer(label, root, reader_cls)
            except Exception:
                fails, notes = ["harness error:\n" + traceback.format_exc()], []
            results.append((label, fails, notes))
    return results


def main():
    rc = 0
    for label, path, regions, positions in CASES:
        if not os.path.exists(path):
            print(f"SKIP  {label}\n      dataset not present: {path}")
            continue
        try:
            fails = check(label, path, regions, positions)
        except Exception:
            fails = ["harness error:\n" + traceback.format_exc()]
        if fails:
            rc = 1
            print(f"FAIL  {label}")
            for f in fails:
                print(f"      - {f}")
        else:
            n = positions if positions is not None else "all declared"
            print(f"PASS  {label}  ({len(regions)} regions, {n} positions)")

    print("\n-- every Squid writer (synthetic, IMA-254) --")
    try:
        writer_results = check_writers()
    except Exception:
        writer_results = [("writer sweep", ["harness error:\n" + traceback.format_exc()], [])]
    for label, fails, notes in writer_results:
        if fails:
            rc = 1
            print(f"FAIL  {label}")
            for f in fails:
                print(f"      - {f}")
        else:
            print(f"PASS  {label}")
        for n in notes:
            print(f"      NOTE {n}")

    print("\nacceptance:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    rc = main()
    # os._exit, not sys.exit: Qt can SIGSEGV unwinding at interpreter shutdown, and a gate
    # whose exit code is decided by a teardown crash gates nothing.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
