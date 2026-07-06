"""Validate a written plate against the official OME-NGFF v0.5 pydantic models.

Lays on the tried-and-true ``ome-zarr-models`` (OME's reference pydantic schema) for the one
part worth borrowing spec code for: is our metadata JSON actually valid? The writer stays our
lean, streaming, ndviewer-verified tensorstore code; this checks every group it writes against
the published schema, failing loud (pydantic ValidationError) on any drift.

Test-time only — ``ome-zarr-models`` is a ``[test]`` extra, not a runtime dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

from ome_zarr_models.v05 import image as I
from ome_zarr_models.v05 import plate as P
from ome_zarr_models.v05 import well as W


def _ome(group_dir: Path) -> dict:
    return json.loads((group_dir / "zarr.json").read_text())["attributes"]["ome"]


def assert_valid_ngff_plate(plate_dir) -> None:
    """Validate the plate group + every well group + every field image group (v0.5 schema).

    Raises ``pydantic.ValidationError`` if any group's ``attributes.ome`` violates the spec.
    """
    plate_dir = Path(plate_dir)
    plate_ome = _ome(plate_dir)
    P.PlateBase.model_validate(plate_ome["plate"])  # rows/columns/wells/field_count

    for well in plate_ome["plate"]["wells"]:
        well_ome = _ome(plate_dir / well["path"])
        W.WellMeta.model_validate(well_ome["well"])  # images -> field paths
        for image in well_ome["well"]["images"]:
            field_ome = _ome(plate_dir / well["path"] / image["path"])
            I.ImageAttrs.model_validate(field_ome)  # multiscales (+ omero as allowed extra)
