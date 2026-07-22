"""Dependency-import smoke test (CI). Fails loudly if any runtime dep won't import on this OS —
the cheap guard the maintainer asked for before freezing artifacts."""
import importlib, sys

MODULES = [
    "squidmip", "squidmip.reader", "squidmip._engine", "squidmip.projection",
    "squidmip._output", "squidmip._zarr_store", "squidmip._montage", "squidmip._cli",
    "squidmip._video", "squidmip._viewer",           # _viewer needs a Qt binding (gui extra)
    "numpy", "tifffile", "tensorstore", "pydantic_settings", "imageio", "imageio_ffmpeg",
    "qtpy.QtWidgets", "ndviewer_light.core",
]
failed = []
for m in MODULES:
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        failed.append((m, f"{type(e).__name__}: {e}"))
        print(f"FAIL  {m}: {type(e).__name__}: {e}")
    else:
        print(f"ok    {m}")
if failed:
    print(f"\n{len(failed)} import(s) failed", file=sys.stderr)
    sys.exit(1)
print("\nall imports OK")
