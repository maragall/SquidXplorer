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
"""

from squidmip._correction import (estimate_background, subtract_background, with_correction)
from squidmip._engine import add_projector, available_projectors, project_plate
from squidmip._montage import build_montage
from squidmip._output import write_plate
from squidmip.projection import project, project_well, select_fovs
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
    "write_plate",
    "build_montage",
    "subtract_background",
    "with_correction",
    "estimate_background",
]
__version__ = "0.1.0"
