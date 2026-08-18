"""SquidXplorer — format-aware ingest for Squid well-plate acquisitions."""

import importlib.util as _importlib_util
import os as _os

# Pin the Qt binding to PyQt6 (when importable) before anything imports qtpy;
# an explicit QT_API in the environment always wins.
if "QT_API" not in _os.environ and _importlib_util.find_spec("PyQt6") is not None:
    _os.environ["QT_API"] = "pyqt6"

from squidxplorer._engine import (
    MissingOperatorDependency,
    Operator,
    Param,
    add_operator,
    add_region_operator,
    available_plane_operators,
    available_region_operators,
    bind_operator,
    is_region_operator,
    operator_available,
    operator_consumes,
    operator_extra,
    operator_params,
    operator_produces,
    operator_requires,
    run_plate,
    runnable_operators,
)
from squidxplorer._minerva import export_selection, launch_minerva
from squidxplorer._montage import build_montage
from squidxplorer._output import write_plate
from squidxplorer._plugins import (
    GROUP as OPERATOR_PLUGIN_GROUP,
    OperatorPluginError,
    declared_operator_plugins,
    load_operator_plugins,
)
from squidxplorer._stitch import (
    solve_offsets_px,
    stitch_region,
)
from squidxplorer._tiling import Geometry, TileCache, TileDescriptor, select_tiles
from squidxplorer._platecache import PlateCellCache
from squidxplorer._tilesource import (
    CompositePlateSource,
    InMemoryMultiscale,
    PlateLadder,
    ZarrPyramidSource,
    plate_ladder,
)
from squidxplorer.projection import (
    INTENSITY,
    LABELS,
    PLANE_OP,
    Z_REDUCER,
    labels_op,
    plane_op,
    project,
    project_well,
    select_fovs,
)
from squidxplorer.reader import (
    SquidAcquisitionReader,
    SquidMultiPageTiffReader,
    SquidOMEReader,
    SquidReader,
    SquidZarrReader,
    open_reader,
)

# Imported for their side effect: each module registers its operator(s).
from squidxplorer import _background, _decon, _flatfield, _register, _spots  # noqa: E402,F401  (registration side effect)
from squidxplorer._background import BackgroundParams, bgsub_op, subtract_background
from squidxplorer._decon import (
    OpticsParams,
    decon3d_op,
    decon_op,
    deconvolve,
    deconvolve_plane,
    deconvolve_stack,
    optics_for_channel,
    set_optics,
)
from squidxplorer._flatfield import FlatfieldProfile, correct_flatfield, estimate_profile, flatfield_op
from squidxplorer._spots import SpotParams, SpotResult, detect_spots, spots_op

# Entry-point plugins register after the built-ins, so a name collision is refused.
load_operator_plugins()

__all__ = [
    "open_reader",
    "SquidAcquisitionReader",
    "SquidReader",
    "SquidMultiPageTiffReader",
    "SquidOMEReader",
    "SquidZarrReader",
    "select_fovs",
    "project",
    "project_well",
    "run_plate",
    "add_operator",
    "available_plane_operators",
    "operator_consumes",
    "Operator",
    "plane_op",
    "PLANE_OP",
    "Z_REDUCER",
    "operator_produces",
    "operator_params",
    "bind_operator",
    "labels_op",
    "INTENSITY",
    "LABELS",
    "Param",
    "operator_requires",
    "operator_extra",
    "operator_available",
    "MissingOperatorDependency",
    "load_operator_plugins",
    "declared_operator_plugins",
    "OperatorPluginError",
    "OPERATOR_PLUGIN_GROUP",
    "write_plate",
    "build_montage",
    # region operators (inter-FOV)
    "stitch_region",
    "solve_offsets_px",
    "add_region_operator",
    "available_region_operators",
    "is_region_operator",
    "runnable_operators",
    # tiler + tile sources
    "select_tiles",
    "Geometry",
    "TileCache",
    "TileDescriptor",
    "plate_ladder",
    "PlateLadder",
    "ZarrPyramidSource",
    "InMemoryMultiscale",
    "CompositePlateSource",
    "PlateCellCache",
    # Minerva export
    "export_selection",
    "launch_minerva",
    # plane-ops (registered as "decon" / "bgsub" / "flatfield")
    "deconvolve",
    "deconvolve_plane",
    "deconvolve_stack",
    "decon_op",
    "decon3d_op",
    "OpticsParams",
    "optics_for_channel",
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
