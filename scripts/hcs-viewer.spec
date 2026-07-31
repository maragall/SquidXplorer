# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the MIP tool desktop app (IMA-232).

    pyinstaller --noconfirm --distpath dist --workpath build scripts/hcs-viewer.spec

Produces ``dist/hcs-viewer.app`` — a single arm64 macOS bundle a demoer downloads and
points at THEIR OWN acquisition folder.

Why a spec file and not a `pyinstaller ...` command line
--------------------------------------------------------
The CI one-liner in .github/workflows/build.yml passed ``--windowed`` AND ``-c``; the
later flag wins, so on macOS it produced a console executable and **no .app bundle at
all**. A spec makes the bundle explicit, makes ``target_arch`` explicit, and gives the
excludes a home — none of which fit on one line.

NO DATA IS BUNDLED. That is the business case, not an oversight: the tool reads a
terabyte-scale acquisition **in place** from the demoer's own disk, so the download is a
binary and the hosting bill is zero. Every ``datas`` entry below is a library resource
(Qt plugins, vispy glyph atlases, tensorstore's compiled extension) — grep this file for
a dataset path and you will not find one.

Size, honestly
--------------
This bundle is large because the dependency set is large: Qt5 (the whole widget +
OpenGL stack), NumPy, SciPy, scikit-image, pandas, zarr/numcodecs, tensorstore, and
vispy (ndv's GPU canvas). The ``excludes`` list below removes what is genuinely never
imported at runtime — measured with ``scripts/build_app.py --verify``, which launches
the frozen bundle against a real acquisition, so an exclude that breaks the app is
caught rather than shipped. See the ticket report for the measured number.
"""

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Freeze THIS checkout, not whatever an editable install elsewhere on the machine points
# `squidmip` at. On the build machine it pointed at a different worktree entirely, one
# with no _viewer.py — which would have frozen a bundle with no GUI in it.
_REPO_ROOT = os.path.dirname(os.path.abspath(SPECPATH))  # noqa: F821
if _REPO_ROOT not in sys.path:                            # noqa: F821
    sys.path.insert(0, _REPO_ROOT)                        # noqa: F821

_HIDDEN = []
_DATAS = []
_BINARIES = []

# Packages whose submodules are resolved at RUNTIME (registries, entry points, lazy
# loaders), which static analysis therefore cannot see. Each one is here because it is
# a known PyInstaller blind spot, not defensively:
#   ndviewer_light / ndv / vispy - the detail viewer picks its canvas backend by string
#   tensorstore                  - compiled extension + driver registry (the zarr reader)
#   zarr / numcodecs             - codec registry keyed by the codec name in zarr.json
#   skimage                      - lazy submodule loader (restoration.rolling_ball)
for _pkg in ("ndviewer_light", "ndv", "vispy", "tensorstore", "zarr", "numcodecs", "skimage"):
    _d, _b, _h = collect_all(_pkg)
    _DATAS += _d
    _BINARIES += _b
    _HIDDEN += _h

# tensorstore's compiled extension imports ml_dtypes at init and nothing imports it in
# Python, so static analysis misses it entirely. Measured, not guessed: without this the
# frozen bundle died on `from ._tensorstore import *` -> ModuleNotFoundError: ml_dtypes.
_d, _b, _h = collect_all("ml_dtypes")
_DATAS += _d
_BINARIES += _b
_HIDDEN += _h

# squidmip's own operator registries (_PROJECTORS / _REGION_OPERATORS) are populated by
# import side effect, so every module must be present even if nothing imports it by name.
_HIDDEN += collect_submodules("squidmip")

# Never imported by this app. Each arrives as a transitive dependency of scikit-image /
# pandas / vispy, and together they were 190 MB of the first (517 MB) build — measured
# per-directory with `du -sk` on the bundle, not estimated. The image I/O back ends are
# the big ones: squidmip reads TIFF through tifffile and Zarr through tensorstore, so
# skimage.io's imageio/OpenCV plug-ins are pure dead weight. scripts/build_app.py
# --verify runs the real operators inside the bundle, so a wrong exclude here fails
# loudly instead of shipping.
_EXCLUDES = [
    "cv2",                 # 110 MB of OpenCV, pulled by skimage's optional io plug-in
    "imageio", "imageio_ffmpeg",   # 48 MB (a bundled ffmpeg); skimage.io only
    "mypy",                # a type checker, in a shipped GUI
    "lxml", "cryptography",
    "matplotlib",          # skimage.io plugins + pandas.plotting reference it; the app plots nothing
    "tkinter",             # Tk is a second, unused GUI toolkit
    "IPython", "jupyter_core", "notebook", "ipykernel", "ipywidgets",
    "pytest", "_pytest", "pytest_qt",
    "sphinx", "docutils",
    "numba", "llvmlite",   # tilefusion pulls these; squidmip deliberately does not import tilefusion
    "torch", "tensorflow",
    # BINDING FLIPPED 2026-07-31. This list excluded PyQt6 and trimmed PyQt5's unused modules,
    # which was correct while the app was Qt5 and is now exactly backwards: squidmip/__init__
    # pins QT_API=pyqt6, so a bundle built from the old list ships the one binding the app
    # refuses to use and excludes the one it requires.
    #
    # UNVERIFIED, and said plainly rather than implied: this spec is built by CI on Windows
    # (.github/workflows/build.yml), and no PyInstaller build was run for this change. The
    # module-by-module trim below is carried over by NAME from the Qt5 list, so a Qt6 submodule
    # that PyInstaller needs and this excludes would surface as an ImportError in the frozen
    # app, not here. Whoever runs the next build should read its warn-*.txt before trusting it.
    "PyQt6.QtWebEngineWidgets", "PyQt6.QtWebEngineCore", "PyQt6.QtWebEngine",
    "PyQt6.QtBluetooth", "PyQt6.QtNfc", "PyQt6.QtQuick3D", "PyQt6.QtLocation",
    "PyQt6.QtDesigner", "PyQt6.QtHelp", "PyQt6.QtMultimedia", "PyQt6.QtMultimediaWidgets",
    "PyQt5", "PySide2", "PySide6",
]

a = Analysis(
    [os.path.join(SPECPATH, "hcs_viewer_entry.py")],  # noqa: F821
    pathex=[_REPO_ROOT],
    binaries=_BINARIES,
    datas=_DATAS,
    hiddenimports=_HIDDEN,
    hookspath=[],
    runtime_hooks=[],
    excludes=_EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="hcs-viewer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # windowed: the .app is what a demoer double-clicks
    # M-series; the released macOS artifact is Apple Silicon, like odon's. Only macOS
    # understands target_arch, and CI freezes this same spec on Linux and Windows.
    target_arch="arm64" if sys.platform == "darwin" else None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="hcs-viewer",
)

app = BUNDLE(
    coll,
    name="hcs-viewer.app",
    icon=None,
    bundle_identifier="com.cephla.squidmip.hcsviewer",
    info_plist={
        "CFBundleName": "MIP tool",
        "CFBundleDisplayName": "MIP tool",
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": True,
        # The app opens a FOLDER (a Squid acquisition), so it must be droppable onto the
        # dock icon and openable from Finder's "Open With".
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "Squid acquisition folder",
                "CFBundleTypeRole": "Viewer",
                "LSItemContentTypes": ["public.folder"],
                "CFBundleTypeOSTypes": ["fold"],
            }
        ],
    },
)
