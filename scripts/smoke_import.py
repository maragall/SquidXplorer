"""Dependency-import smoke test (CI). Fails loudly if any runtime dep won't import on this OS —
the cheap guard the maintainer asked for before freezing artifacts."""
import importlib, sys

MODULES = [
    "squidmip", "squidmip.reader", "squidmip._engine", "squidmip.projection",
    "squidmip._output", "squidmip._zarr_store", "squidmip._montage", "squidmip._cli",
    "squidmip._viewer",                              # _viewer needs the gui extra
    "numpy", "tifffile", "tensorstore", "pydantic_settings",
    # `squidmip._video`, `imageio` and `imageio_ffmpeg` are BACK, 2026-08-05, and so is the
    # feature. They were in this list from its first commit (03c5246), were dropped on 2026-07-31
    # when `squidmip/_video.py` was deleted by the other half of IMA-185 (c25d84d, "remove
    # recording from viewer"), and the line that stood here said flatly: "Recording lives on in
    # maragall/SimpleXplorer ... it is not coming back here."
    #
    # THAT CLAIM WAS ABOUT THE WRONG THING, and it is corrected rather than quietly dropped.
    # c25d84d's rationale -- "users record at acquisition time, not here" -- is about CAMERA
    # CAPTURE, which does belong to Squid. `_video.py`'s own first paragraph explicitly disclaims
    # camera capture: it assembles an ALREADY-ACQUIRED axis (T, or Z) of data already on disk into
    # an .mp4, which is a post-acquisition export and is exactly what this viewer is for. The
    # deletion answered a question the module was not asking, and it took the only way to see a
    # time series or a focus sweep as motion with it.
    #
    # The 2026-07-31 removal was still right at the time and its cost is worth keeping written
    # down: this gate asserted a module the project had deleted, so `smoke` failed on ALL THREE
    # OSes and `freeze` (which has `needs: smoke`) was SKIPPED on every run rather than failing
    # visibly. Both packages are now declared -- `squidmip[video]`, pulled in by `[gui]` and
    # `[test]` -- so this gate is asserting something the project installs on purpose.
    "squidmip._video", "imageio", "imageio_ffmpeg",
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
