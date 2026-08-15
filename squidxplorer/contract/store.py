"""THE walk of an OME-NGFF store: group attributes, plate-dir resolution, level paths.

docs/plate-contract.md elevates one store walk as the canonical seam. Before this module the
attrs reader existed four times (`reader._group_attrs`, `_tilesource._read_ome`,
`_montage._read_group_ome`, inline in `_mosaic_source.level_paths`) and only the first handled
a v0.4 ``.zattrs`` store — three consumers silently could not read what the fourth could.
``_resolve_plate_dir`` likewise existed twice, verbatim. One copy each now, and every consumer
gains the v0.4/v0.5 normalisation for free.
"""

from __future__ import annotations

import json
from pathlib import Path

_ZARR_V2_ATTRS = ".zattrs"
_ZARR_V3_META = "zarr.json"


def ome_attrs(group_dir) -> dict:
    """The OME metadata payload of a zarr group, normalising the v0.4 / v0.5 difference.

    v0.4 keeps it in ``.zattrs`` at top level; v0.5 nests it under ``zarr.json``'s
    ``attributes.ome``. Returns ``{}`` for a directory that is neither.
    """
    path = Path(group_dir)
    v2 = path / _ZARR_V2_ATTRS
    if v2.exists():
        return json.loads(v2.read_text() or "{}")
    v3 = path / _ZARR_V3_META
    if v3.exists():
        attrs = json.loads(v3.read_text() or "{}").get("attributes") or {}
        ome = attrs.get("ome")
        return ome if isinstance(ome, dict) else attrs
    return {}


def resolve_plate_dir(plate_path) -> Path:
    """Accept either ``plate.ome.zarr`` itself or the directory ``write_plate`` wrote it into."""
    p = Path(plate_path)
    if "plate" in ome_attrs(p):
        return p
    if (p / "plate.ome.zarr").is_dir():
        return p / "plate.ome.zarr"
    raise ValueError(
        f"{plate_path!s} is not an OME-NGFF HCS plate (no plate.ome.zarr / plate group "
        "metadata). Point this at write_plate's output directory or its plate.ome.zarr."
    )


def level_paths(group) -> "list[Path]":
    """Every resolution level of an OME-NGFF image group, highest resolution first."""
    group = Path(group)
    ome = ome_attrs(group)
    multiscales = ome.get("multiscales") or []
    if not multiscales:
        raise ValueError(f"{group}: no 'multiscales' metadata; not an OME-NGFF image group.")
    datasets = multiscales[0].get("datasets") or []
    if not datasets:
        raise ValueError(f"{group}: multiscales carries no 'datasets' (no resolution levels).")
    return [group / str(d["path"]) for d in datasets]
