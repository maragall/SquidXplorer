"""PyInstaller entry point for the frozen HCS viewer.

Normal launch is exactly ``squidmip._viewer.main`` — the frozen bundle and the
``squidmip-view`` console script run the same code.

``--selftest DATASET`` (IMA-232) is the headless proof that the *bundle* works, not just
the source tree: it builds the real ``PlateWindow`` against a real acquisition folder
offscreen, then prints what was actually ingested and exits. A .app that starts and then
dies on the first dataset is worth less than no .app, and "a window appeared" is not
evidence against that; the region/FOV/channel counts printed here are.
"""

import json
import os
import sys

if not getattr(sys, "frozen", False):
    # Running from a checkout: put THIS tree's repo root ahead of site-packages. An
    # editable install elsewhere on the machine can point `squidmip` at a different
    # checkout (it did — at a worktree with no _viewer.py), and running a script file
    # does not put the cwd on sys.path to save you. Frozen builds skip this: the bundle
    # carries its own copy and the repo root does not exist on the demoer's machine.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _selftest(dataset: str) -> int:
    """Launch offscreen, ingest *dataset*, print a JSON summary. Exit 0 iff it ingested."""
    # Must precede any QApplication: there is no display in CI or over ssh, and a frozen
    # windowed bundle would otherwise abort before reaching a single line of our code.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from qtpy.QtWidgets import QApplication

    from squidmip._viewer import PlateWindow

    app = QApplication.instance() or QApplication([])
    # The viewer's own test escape hatch (_viewer.main checks it) — set on the app rather
    # than faked here, so the check exercises the shipped code path.
    app.setProperty("_squidmip_test", True)

    win = PlateWindow(dataset)
    app.processEvents()

    meta = getattr(win, "_meta", None)
    reader = getattr(win, "_reader", None)
    ok = reader is not None and meta is not None
    summary = {
        "ingested": bool(ok),
        "frozen": bool(getattr(sys, "frozen", False)),
        "dataset": dataset,
        "readout": win._readout.text() if hasattr(win, "_readout") else "",
    }
    if ok:
        fovs_per_region = meta.get("fovs_per_region") or {}
        summary.update(
            regions=len(fovs_per_region),
            fovs=sum(len(v) for v in fovs_per_region.values()),
            channels=[c["name"] for c in (meta.get("channels") or [])],
            z_levels=len(meta.get("z_levels") or []),
            frame_shape=list(meta.get("frame_shape") or ()),
            pixel_size_um=meta.get("pixel_size_um"),
        )
    if ok:
        summary["compute"] = _compute_check(win)
        ok = summary["compute"].get("ok", False)
    summary["ingested"] = bool(summary["ingested"]) and ok

    print("SELFTEST " + json.dumps(summary), flush=True)
    win.close()
    return 0 if ok else 1


def _compute_check(win) -> dict:
    """MIP one REAL FOV, then run each plane operator on a crop of it.

    Ingest alone would not catch a frozen bundle that ships a broken scipy/scikit-image:
    the reader path touches neither. The bundle's ``excludes`` list is aggressive (it
    drops ~150 MB of skimage's optional imageio/OpenCV back ends), so the operators that
    depend on what is LEFT have to be executed, not assumed. A crop, not a whole frame,
    because rolling-ball on 2084x2084 takes minutes and this is a smoke check.
    """
    import numpy as np

    from squidmip import project_well, richardson_lucy_gaussian, subtract_background

    out = {}
    try:
        meta = win._meta
        region = next(iter(meta["fovs_per_region"]))
        fov = meta["fovs_per_region"][region][0]
        mip = project_well(win._reader, region, fov)     # (T, C, 1, Y, X)
        out["mip_shape"] = list(mip.shape)
        out["mip_dtype"] = str(mip.dtype)
        crop = np.ascontiguousarray(mip[0, 0, 0, :256, :256])
        bg = subtract_background(crop)                   # scikit-image rolling_ball
        dec = richardson_lucy_gaussian(crop)             # scipy.ndimage gaussian_filter
        out["bgsub_mean_delta"] = float(crop.mean() - bg.mean())
        out["decon_shape"] = list(dec.shape)
        out["ok"] = mip.shape[2] == 1 and bg.shape == crop.shape and dec.shape == crop.shape
    except Exception as exc:  # report, never crash: the point is a legible verdict
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        if i + 1 >= len(sys.argv):
            print("--selftest needs an acquisition folder", file=sys.stderr)
            return 2
        return _selftest(sys.argv[i + 1])

    from squidmip._viewer import main as viewer_main

    viewer_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
