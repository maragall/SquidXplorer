"""PyInstaller entry point for the frozen HCS viewer.

Normal launch is exactly squidxplorer._viewer.main. --selftest DATASET builds the real
PlateWindow against a real acquisition offscreen and prints what was actually ingested, so the
frozen bundle (not just the source tree) is proven to work.
"""

import json
import os
import sys

if not getattr(sys, "frozen", False):
    # Editable installs elsewhere can point `squidxplorer` at a different checkout, and running
    # a script file does not add the cwd to sys.path.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _selftest(dataset: str) -> int:
    """Launch offscreen, ingest *dataset*, print a JSON summary. Exit 0 iff it ingested."""
    # Must precede any QApplication: no display in CI or over ssh.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # squidxplorer first, then qtpy: importing the package pins QT_API, and qtpy resolves the
    # binding at its own import.
    from squidxplorer._viewer import PlateWindow

    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    app.setProperty("_squidxplorer_test", True)

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
    """MIP one real FOV, then run each plane operator on a crop of it.

    A crop rather than a whole frame: rolling-ball on 2084x2084 takes minutes.
    """
    import numpy as np

    from squidxplorer import deconvolve_plane, project_well, subtract_background

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
        dec = deconvolve_plane(crop)                     # petakit vectorial PSF, one plane
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

    from squidxplorer._viewer import main as viewer_main

    viewer_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
