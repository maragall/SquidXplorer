#!/usr/bin/env python3
"""Behavioural diff against the last known-good commit, so a regression cannot hide behind
a green suite.

The FOV slider broke and 808 tests stayed green, because every test called the handler
directly instead of driving the widget, and because the suite only ever knew about HEAD.
This runs the SAME probe against two checkouts and prints a three-column table.

    python tools/regression_vs_baseline.py            # probe both, diff
    python tools/regression_vs_baseline.py --probe    # probe THIS checkout, emit JSON

The hard part is not measuring, it is classification. Most differences here are INTENDED:
the baseline refused glass slides, had no operators beyond mip/reference, and had no
exploration pane. So a difference is not a verdict. Each key carries an `expect`:

    same      - must not change. A difference is a REGRESSION.
    improved  - HEAD should be >= baseline. A decrease is a REGRESSION.
    intended  - known to differ. Recorded, never failed on.

Anything not listed is reported as UNCLASSIFIED, which is a prompt to think, not a pass.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
BASELINE = os.path.expanduser("~/Cephla/worktrees/SquidMIP/baseline-1504c05")

TISSUE = ("/Users/julioamaragall/Downloads/"
          "test_10x_laser_af_z_stack_2025-10-28_13-40-43.939945 yy")
PLATE = "/Users/julioamaragall/Downloads/synthetic_2x2_wellplate"

# key -> (expectation, why)
EXPECT = {
    "plate.regions":        ("same", "the acquisition did not change"),
    "plate.n_positions":    ("intended", "IMA-187/215: the baseline never read coordinates.csv, so it reported 0"),
    "plate.overview":       ("same", "the well plate opened before and must still open"),
    "plate.slider_wells":   ("same", "the FOV slider must list every well - this is the reported regression"),
    # These two probe a feature the baseline does not have, so they read "n/a" there. They are
    # still SAME-expectations against HEAD-over-time: once the exploration pane exists, a dropped
    # push or a disagreeing producer is the regression that started this harness (b6a6e01).
    "plate.push_dropped":   ("n/a-or-same", "0 dropped pushes once an exploration pane exists"),
    "plate.preview_agrees": ("n/a-or-same", "producer and consumer must describe one well list"),
    "tissue.overview":      ("intended", "IMA-214: glass slides were REFUSED at baseline, now open"),
    "tissue.regions":       ("intended", "same reason - baseline returned nothing"),
    "operators":            ("intended", "IMA-210/222/223/224/225 added operators"),
    "region_operators":     ("intended", "IMA-222 added the stitch seam, absent at baseline"),
    "panes":                ("intended", "IMA-237 added the third pane"),
    "units_key":            ("intended", "the mm key was renamed to fov_positions_um and converted"),
}


def probe() -> dict:
    """Everything measurable in ONE process, tolerant of a checkout that lacks a feature."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out: dict = {}

    def attempt(key, fn):
        try:
            out[key] = fn()
        except Exception as e:
            out[key] = f"<{type(e).__name__}: {e}>"

    from PyQt5.QtCore import QEventLoop, QTimer
    from PyQt5.QtWidgets import QApplication
    import squidmip._viewer as V

    app = QApplication.instance() or QApplication([])

    def settle(ms=6000):
        loop = QEventLoop(); QTimer.singleShot(ms, loop.quit); loop.exec_()

    def open_win(path):
        w = V.PlateWindow(None); w.resize(1600, 900); w.show()
        w.ingest(path); settle()
        return w

    # --- the well plate: this path worked at baseline and must still work ---------
    w = open_win(PLATE)
    meta = w._reader.metadata if w._reader is not None else {}
    attempt("plate.overview", lambda: w._overview is not None)
    attempt("plate.regions", lambda: list(meta.get("regions") or []))
    attempt("plate.n_positions", lambda: len(meta.get("fov_positions_um")
                                             or meta.get("fov_positions") or {}))
    attempt("units_key", lambda: "fov_positions_um" if "fov_positions_um" in meta else "fov_positions")
    attempt("plate.slider_wells", lambda: len(getattr(w, "_order", []) or []))
    attempt("panes", lambda: w._split.count())
    attempt("operators", lambda: sorted(o.key for o in V._OPERATIONS))

    # The reported regression: open an exploration tab, then check the slider is still
    # fed. At baseline there are no exploration tabs, so this reads as "n/a" rather than
    # failing - the absence of a feature is not a regression in it.
    def _slider_after_subset():
        if not hasattr(w, "open_exploration_tab"):
            return "n/a (no exploration pane at baseline)"
        w.open_exploration_tab([(meta.get("regions") or ["A1"])[0]]); settle(4000)
        return int(getattr(w, "_dropped_pushes", 0))
    attempt("plate.push_dropped", _slider_after_subset)
    attempt("plate.preview_agrees",
            lambda: (getattr(w, "_preview_order", None) == getattr(w, "_push_order", None))
            if hasattr(w, "_push_order") else "n/a (no subset scoping at baseline)")
    try:
        w.close()
    except Exception:
        pass

    # --- the glass slide: REFUSED at baseline, must open now ---------------------
    w2 = open_win(TISSUE)
    m2 = w2._reader.metadata if w2._reader is not None else {}
    attempt("tissue.overview", lambda: w2._overview is not None)
    attempt("tissue.regions", lambda: list(m2.get("regions") or []))
    try:
        w2.close()
    except Exception:
        pass

    def _region_ops():
        from squidmip._stitch import available_region_operators
        return sorted(available_region_operators())
    attempt("region_operators", _region_ops)
    return out


def run_probe_in(checkout: str) -> dict:
    env = dict(os.environ, PYTHONPATH=checkout,
               QT_QPA_PLATFORM="offscreen", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    r = subprocess.run([sys.executable, os.path.join(checkout, "tools", "regression_vs_baseline.py"),
                        "--probe"], capture_output=True, text=True, env=env, cwd=checkout, timeout=900)
    try:
        return json.loads(r.stdout[r.stdout.index("{"):r.stdout.rindex("}") + 1])
    except Exception:
        return {"<probe failed>": (r.stdout[-300:] + r.stderr[-300:])}


def main() -> int:
    if "--probe" in sys.argv:
        print(json.dumps(probe(), default=str))
        return 0

    if not os.path.isdir(BASELINE):
        print(f"baseline checkout missing: {BASELINE}\n"
              f"create it with:  git worktree add {BASELINE} --detach 1504c05")
        return 2

    # The baseline predates this script, so run THIS copy of it against that checkout.
    probe_at_baseline = os.path.join(BASELINE, "tools", "regression_vs_baseline.py")
    os.makedirs(os.path.dirname(probe_at_baseline), exist_ok=True)
    if not os.path.exists(probe_at_baseline):
        import shutil
        shutil.copy(os.path.abspath(__file__), probe_at_baseline)

    print("probing baseline 1504c05 ...", flush=True)
    base = run_probe_in(BASELINE)
    print("probing HEAD ...", flush=True)
    head = run_probe_in(REPO)

    keys = sorted(set(base) | set(head))
    width = max(len(k) for k in keys)
    regressions = []
    print("\n" + "=" * 100)
    for k in keys:
        b, h = base.get(k, "<absent>"), head.get(k, "<absent>")
        expect, why = EXPECT.get(k, ("unclassified", "not classified - decide what this should be"))
        if b == h:
            verdict = "same"
        elif expect == "n/a-or-same" and isinstance(b, str) and b.startswith("n/a"):
            # The baseline lacks the feature entirely. Absence is not a regression; but pin the
            # HEAD value so a LATER run catches it changing.
            verdict = "new" if h in (0, True) else "REGRESSION?"
            if verdict == "REGRESSION?":
                regressions.append((k, b, h, why))
        elif expect == "intended":
            verdict = "INTENDED"
        elif expect == "improved" and isinstance(b, int) and isinstance(h, int) and h >= b:
            verdict = "improved"
        else:
            verdict = "REGRESSION?" if expect != "unclassified" else "UNCLASSIFIED"
            regressions.append((k, b, h, why))
        print(f"{verdict:12} {k:{width}}  baseline={b!r}")
        if b != h:
            print(f"{'':12} {'':{width}}  HEAD    ={h!r}")
    print("=" * 100)
    if regressions:
        print(f"\n{len(regressions)} difference(s) that are NOT known-intended:\n")
        for k, b, h, why in regressions:
            print(f"  {k}\n    baseline {b!r}\n    HEAD     {h!r}\n    expected: {why}\n")
    else:
        print("\nno unexplained differences against 1504c05")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
