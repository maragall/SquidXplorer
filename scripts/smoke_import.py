"""Dependency-import smoke test (CI): fails loudly if any runtime dep won't import on this OS."""
import importlib, sys

MODULES = [
    "squidxplorer", "squidxplorer.reader", "squidxplorer._engine", "squidxplorer.projection",
    "squidxplorer._output", "squidxplorer._zarr_store", "squidxplorer._montage", "squidxplorer._cli",
    "squidxplorer._viewer",                              # _viewer needs the gui extra
    "numpy", "tifffile", "tensorstore", "pydantic_settings",
    "squidxplorer._video", "imageio", "imageio_ffmpeg",
    # qtpy is the import the app actually performs; it fails loudly if the pinned binding is missing.
    "qtpy.QtWidgets", "napari",
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
