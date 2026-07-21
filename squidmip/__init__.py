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
"""

from squidmip._engine import (
    Operator,
    add_projector,
    available_projectors,
    project_plate,
    projector_consumes,
)
from squidmip._montage import build_montage
from squidmip._output import write_plate
from squidmip._tiling import Geometry, TileCache, TileDescriptor, select_tiles
from squidmip._tilesource import InMemoryMultiscale, PlateLadder, ZarrPyramidSource, plate_ladder
from squidmip.projection import (
    PLANE_OP,
    Z_REDUCER,
    plane_op,
    project,
    project_well,
    select_fovs,
)
from squidmip.reader import SquidReader, open_reader

__all__ = [
    "open_reader",
    "SquidReader",
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
    # IMA-216 tiler + IMA-217 sources
    "select_tiles",
    "Geometry",
    "TileCache",
    "TileDescriptor",
    "plate_ladder",
    "PlateLadder",
    "ZarrPyramidSource",
    "InMemoryMultiscale",
]
__version__ = "0.1.0"
