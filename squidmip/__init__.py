"""SquidMIP — format-aware ingest for Squid well-plate acquisitions.

The public surface is intentionally tiny::

    from squidmip import open_reader, select_fovs, project_well
    reader = open_reader("/path/to/acquisition")
    meta = reader.metadata
    plane = reader.read("B3", 15, "Fluorescence_638_nm_-_Penta", z=0)   # (Y, X), native dtype

    wells = select_fovs(meta, n_fovs=1)                  # {region: [fov, ...]}
    img = project_well(reader, "B3", 0)                  # (T, C, 1, Y, X), native dtype

    for region, fov, image in project_plate(reader, workers=8):   # whole plate, parallel + streamed
        ...                                              # (T, C, 1, Y, X) per well, bounded memory

    write_plate(reader, "/path/out")          # canonical multiscale OME-zarr plate (tiff=True adds TIFFs)
    build_montage("/path/out")                # static plate montage PNG + region-jump sidecar + hover viewer

    ladder = plate_ladder(meta)                          # the µm tile ladder, from metadata alone
    src = ZarrPyramidSource("/path/out/plate.ome.zarr")   # TileSource over the written plate
    preview = InMemoryMultiscale(ladder, channels)        # TileSource for a live run (byte-budgeted)
    tiles = select_tiles(bbox_um, um_per_px, src.ladder.geometry)   # O(viewport), see _tiling.py

    export_selection(reader, [("B3", 0)])     # Minerva Author: OME-TIFF + .story.json per FOV
    launch_minerva()                          # best-effort; returns False if it isn't installed
"""

import importlib.util as _importlib_util
import os as _os

# ---------------------------------------------------------------------------------------------
# WHICH QT BINDING THIS PROCESS USES. Decided here, once, before anything can import qtpy.
#
# The GUI was migrated off PyQt5 (10b8348, f7f9b28, ce5605c: every import through qtpy, every enum
# fully scoped, every Qt5-only API replaced, ndviewer_light deleted because it imported PyQt5 at
# module scope). What never landed was the PIN, and without it qtpy's own preference order starts
# at PyQt5 -- so on a machine with both installed, the tree LOOKED migrated and quietly kept
# running on Qt5. Verified on 2026-07-31: `qtpy.API_NAME` reported PyQt5 5.15.14 on the developer
# machine with PyQt6 6.11 sitting right next to it. Qt6 is also what makes the app honour
# fractional display scaling by default (see `_viewer.enable_hidpi`), which is the difference
# Spencer saw between his Qt6 build and the Qt5 one.
#
# Conditional on PyQt6 actually being importable, and that condition is load-bearing, not
# defensive noise: an existing venv installed from an older `.[gui]` has PyQt5 ONLY, and this
# package is installed editable there, so a plain `git pull` applies immediately with no reinstall.
# An unconditional pin would turn that pull into an ImportError at launch on the demo machine.
# `find_spec` rather than a try/import: it answers the same question without paying for the import.
#
# An explicit QT_API in the environment always wins, so `QT_API=pyqt5 squidmip-view` remains the
# escape hatch if Qt6 ever misbehaves on a given machine.
if "QT_API" not in _os.environ and _importlib_util.find_spec("PyQt6") is not None:
    _os.environ["QT_API"] = "pyqt6"

from squidmip._engine import (
    Operator,
    add_projector,
    available_projectors,
    project_plate,
    projector_consumes,
)
from squidmip._minerva import export_selection, launch_minerva
from squidmip._montage import build_montage
from squidmip._output import write_plate
from squidmip._stitch import (
    add_region_operator,
    available_region_operators,
    solve_offsets_px,
    stitch_plate,
    stitch_region,
)
from squidmip._tiling import Geometry, TileCache, TileDescriptor, select_tiles
from squidmip._platecache import PlateCellCache
from squidmip._tilesource import (
    CompositePlateSource,
    InMemoryMultiscale,
    PlateLadder,
    ZarrPyramidSource,
    plate_ladder,
)
from squidmip.projection import (
    PLANE_OP,
    Z_REDUCER,
    plane_op,
    project,
    project_well,
    select_fovs,
)
from squidmip.reader import (
    SquidAcquisitionReader,
    SquidMultiPageTiffReader,
    SquidOMEReader,
    SquidReader,
    SquidZarrReader,
    open_reader,
)

# --- the IMA-223/224/225 plane-ops ------------------------------------------------------------
#
# Imported for their SIDE EFFECT: each module ends in one ``add_projector`` call, which is the
# entire cost of adding an operator under IMA-210 (no engine edit, no dispatch table anywhere
# else). Importing them here — rather than leaving each module to be found by whoever needs it —
# is what makes the names appear in ``available_projectors()``, and therefore automatically in
# the CLI's ``--projector`` validation and the viewer's projector selector, which both read that
# list rather than a hardcoded one.
#
# KNOWN LIMIT (documented in _engine's docstring): these are plane-ops, so their output keeps z
# at FULL depth, and ``_output._validate_image`` accepts only Z == 1. A plane-op therefore
# streams correctly out of ``project_plate`` but fails LOUD at ``write_plate``. That is by
# design for now — loud, not silently wrong — and it lifts the moment the writer learns Z > 1.
from squidmip import _background, _decon, _flatfield, _spots  # noqa: E402,F401  (registration side effect)
from squidmip._background import BackgroundParams, bgsub_op, subtract_background
from squidmip._decon import (
    OpticsParams,
    decon3d_op,
    decon_op,
    deconvolve,
    deconvolve_plane,
    deconvolve_stack,
    set_optics,
)
from squidmip._flatfield import FlatfieldProfile, correct_flatfield, estimate_profile, flatfield_op
from squidmip._spots import SpotParams, SpotResult, detect_spots, spots_op

__all__ = [
    "open_reader",
    # The NAME of that interface, for a consumer in another repo (Squid) to depend on.
    "SquidAcquisitionReader",
    # One reader per Squid writer, all behind the same interface (IMA-254).
    "SquidReader",
    "SquidMultiPageTiffReader",
    "SquidOMEReader",
    "SquidZarrReader",
    "select_fovs",
    "project",
    "project_well",
    "project_plate",
    "add_projector",
    "available_projectors",
    # IMA-210 consumes-axis registry
    "projector_consumes",
    "Operator",
    "plane_op",
    "PLANE_OP",
    "Z_REDUCER",
    "write_plate",
    "build_montage",
    # IMA-222 region operators (inter-FOV; the parallel table to the projectors)
    "stitch_region",
    "stitch_plate",
    "solve_offsets_px",
    "add_region_operator",
    "available_region_operators",
    # IMA-216 tiler + IMA-217 sources
    "select_tiles",
    "Geometry",
    "TileCache",
    "TileDescriptor",
    "plate_ladder",
    "PlateLadder",
    "ZarrPyramidSource",
    "InMemoryMultiscale",
    # Gap 1: the persistent plate cells, and the composite source they feed
    "CompositePlateSource",
    "PlateCellCache",
    # IMA-228 Minerva export
    "export_selection",
    "launch_minerva",
    # IMA-223/224/225 plane-ops (registered as "decon" / "bgsub" / "flatfield")
    # IMA-247: decon runs on Julio's petakit (real vectorial PSF), not a Gaussian guess.
    "deconvolve",
    "deconvolve_plane",
    "deconvolve_stack",
    "decon_op",
    "decon3d_op",
    "OpticsParams",
    "set_optics",
    "subtract_background",
    "BackgroundParams",
    "bgsub_op",
    "correct_flatfield",
    "estimate_profile",
    "FlatfieldProfile",
    "flatfield_op",
    # spot detection / nuclei counting (registered as "spots")
    "SpotParams",
    "SpotResult",
    "detect_spots",
    "spots_op",
]
__version__ = "0.1.0"
