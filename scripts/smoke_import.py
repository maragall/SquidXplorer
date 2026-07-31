"""Dependency-import smoke test (CI). Fails loudly if any runtime dep won't import on this OS —
the cheap guard the maintainer asked for before freezing artifacts."""
import importlib, sys

MODULES = [
    "squidmip", "squidmip.reader", "squidmip._engine", "squidmip.projection",
    "squidmip._output", "squidmip._zarr_store", "squidmip._montage", "squidmip._cli",
    "squidmip._video", "squidmip._viewer",           # _viewer needs the gui extra
    "numpy", "tifffile", "tensorstore", "pydantic_settings", "imageio", "imageio_ffmpeg",
    # The Qt binding is imported through qtpy, never by name, and squidmip/__init__ pins it to
    # pyqt6 before qtpy loads -- so importing `PyQt5.QtWidgets` here proved that a binding the
    # app no longer uses is installed. `qtpy.QtWidgets` is the import the app actually performs,
    # and it fails loudly if the pinned binding is missing.
    #
    # `ndviewer_light.core` was dropped from this list on 2026-07-31: the package was removed as
    # a dependency in ce5605c (it imported PyQt5 at module scope and could not share a process
    # with a Qt6 napari), so this line was asserting the presence of something the project had
    # deliberately deleted.
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
