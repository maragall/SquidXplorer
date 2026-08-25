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

import importlib.util
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Freeze THIS checkout, not whatever an editable install elsewhere on the machine points
# `squidxplorer` at. On the build machine it pointed at a different worktree entirely, one
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
#   napari + npe2/app_model/... - THE RENDERER. See the block below; this was the bug.
#   vispy                        - the GPU canvas picks its backend by string
#   tensorstore                  - compiled extension + driver registry (the zarr reader)
#   zarr / numcodecs             - codec registry keyed by the codec name in zarr.json
#   skimage                      - lazy submodule loader (restoration.rolling_ball)
#
# NAPARI WAS MISSING FROM THIS LIST UNTIL 2026-08-05, and it is the single reason the first
# frozen bundle was not shippable. MEASURED, not theorised: the built .app launched, painted the
# plate shell and read the acquisition ("2 wells loaded, 2 multi-FOV region(s)"), and then every
# view window rendered a dark-red panel reading
#
#     napari viewer unavailable - NapariBindingError: napari's API has moved under us ...
#     Missing or de-exported: napari.components (import failed: FileNotFoundError(2, 'No such
#     file or directory')), napari.qt, napari._qt.layer_controls ... There is no mosaic.
#
# `find hcs-viewer.app -name napari.yaml` returned NOTHING and there was no `napari/` directory
# anywhere in the bundle, while `vispy/` was present -- because vispy was on this list and napari
# never had been. napari is not a submodule blind spot like the others; it was simply ABSENT.
#
# This is exactly the failure mode this spec's own excludes note predicted: invisible to the test
# suite (which imports the real napari from site-packages) and visible ONLY in the frozen app.
# `verify_napari_bindings()` caught it and refused loudly rather than rendering garbage, which is
# why it reads as a clean error panel instead of a crash -- the guard worked, the packaging did not.
#
# The ecosystem packages are here for the same reason as napari itself: npe2/app_model read YAML
# and JSON manifests off disk at import, and a manifest that is not collected is a FileNotFoundError
# at runtime, not a missing-module warning at build time.
#
# ndviewer_light and ndv were REMOVED from this list on the same day: both were dropped as
# dependencies (ce5605c) because ndviewer_light imports PyQt5 at module scope, so `collect_all`
# was being asked for packages that are not installed and silently contributed nothing.
#
# napari_console is deliberately NOT in this list. It is the in-viewer IPython console, and
# IPython/jupyter_core/ipykernel are excluded below; collecting the console while excluding its
# interpreter would trade one runtime ImportError for another. This app is a plate viewer, not a
# notebook, and nothing on the demo path opens a console.
#
# imageio / imageio_ffmpeg: the .mp4 export's encoder (squidxplorer/_video.py). imageio resolves
# its FORMAT plug-ins by name at runtime, and imageio_ffmpeg ships an ffmpeg BINARY as package
# data that no import can reveal -- so `collect_all` is what brings both, and it is the same
# manifest-off-disk failure napari has above. Without it `_video.encoder_problem()` would
# correctly report "no ffmpeg binary" inside the frozen app and the movie button would be
# permanently greyed out: the guard would work and the packaging would not, exactly as with
# napari. Both were in _EXCLUDES below until 2026-08-05 on the grounds that only skimage.io
# wanted them, which stopped being true when this package started encoding movies of its own.
# Costs ~48 MB of bundle, which is the whole price of the feature.
for _pkg in ("napari", "napari_svg", "npe2", "app_model", "magicgui",
             "superqt", "psygnal", "in_n_out", "vispy", "tensorstore", "zarr", "numcodecs",
             "skimage", "imageio", "imageio_ffmpeg"):
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

# squidxplorer's own operator registries (_PROJECTORS / _REGION_OPERATORS) are populated by
# import side effect, so every module must be present even if nothing imports it by name.
_HIDDEN += collect_submodules("squidxplorer")

# Never imported by this app. Each arrives as a transitive dependency of scikit-image /
# pandas / vispy, and together they were 190 MB of the first (517 MB) build — measured
# per-directory with `du -sk` on the bundle, not estimated. The image I/O back ends are
# the big ones: squidxplorer reads TIFF through tifffile and Zarr through tensorstore, so
# skimage.io's imageio/OpenCV plug-ins are pure dead weight. scripts/build_app.py
# --verify runs the real operators inside the bundle, so a wrong exclude here fails
# loudly instead of shipping.
_EXCLUDES = [
    "cv2",                 # 110 MB of OpenCV, pulled by skimage's optional io plug-in
    # `imageio` and `imageio_ffmpeg` USED TO BE HERE, on the grounds that only skimage.io wanted
    # them. squidxplorer/_video.py wants them now, so they moved to the collect_all list above.
    "mypy",                # a type checker, in a shipped GUI
    "lxml", "cryptography",
    # matplotlib is excluded unconditionally again (2026-08-25): its one consumer was the
    # decon QC sweep (_decon_qc.py), which is shelved whole - the app plots nothing.
    "matplotlib",
] + [
    "tkinter",             # Tk is a second, unused GUI toolkit
    "IPython", "jupyter_core", "notebook", "ipykernel", "ipywidgets",
    "pytest", "_pytest", "pytest_qt",
    "sphinx", "docutils",
    "numba", "llvmlite",   # tilefusion pulls these; squidxplorer deliberately does not import tilefusion
    "torch", "tensorflow",
    # BINDING FLIPPED 2026-07-31. This list excluded PyQt6 and trimmed PyQt5's unused modules,
    # which was correct while the app was Qt5 and is now exactly backwards: squidxplorer/__init__
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
    # SIGNING (2026-08-05). Both of these were hardcoded None, and the note that belongs here is
    # what was MEASURED on the resulting bundle rather than what is usually assumed:
    #
    #   codesign -dv   ->  Signature=adhoc, TeamIdentifier=not set
    #   codesign --verify --deep --strict  ->  exit 0 (the bundle is internally consistent)
    #   spctl -a -vv   ->  REJECTED
    #
    # So the bundle was ALREADY ad-hoc signed with codesign_identity=None: PyInstaller ad-hoc
    # signs arm64 binaries itself, because macOS will not execute an unsigned arm64 Mach-O at all.
    # Ad-hoc signing is therefore not the missing piece and never was, and setting it explicitly
    # here would change nothing. `spctl` rejects an ad-hoc bundle no matter how it was produced:
    # Gatekeeper wants a Developer ID signature plus notarisation, and NEITHER can be produced
    # without the owner's Apple Developer credentials. See SIGNING.md.
    #
    # What is new is that both are now OVERRIDABLE from the environment, so the owner signs by
    # exporting two variables and re-running the same build command, with no edit to this file:
    #
    #   export SQUIDXPLORER_CODESIGN_IDENTITY="Developer ID Application: <Name> (TEAMID)"
    #   python scripts/build_app.py --dataset /path/to/acquisition
    #
    # Unset (the default, and what CI does) reproduces today's behaviour exactly: ad-hoc, and
    # honest about it. NOTHING here disables or weakens a security control -- the fallback is the
    # same ad-hoc signature macOS already required, not an unsigned binary.
    codesign_identity=os.environ.get("SQUIDXPLORER_CODESIGN_IDENTITY") or None,
    # Only meaningful together with a real identity + --options runtime (Hardened Runtime), which
    # notarisation requires. scripts/entitlements.plist is the starting point; SIGNING.md explains
    # why it is a starting point and not a final answer.
    entitlements_file=(os.environ.get("SQUIDXPLORER_ENTITLEMENTS")
                       or (os.path.join(SPECPATH, "entitlements.plist")  # noqa: F821
                           if os.environ.get("SQUIDXPLORER_CODESIGN_IDENTITY") else None)),
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
    bundle_identifier="com.cephla.squidxplorer.hcsviewer",
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
