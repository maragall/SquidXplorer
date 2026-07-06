"""Low-level zarr v3 store + NGFF group primitives (vendored from tilefusion io/zarr.py).

Vendored, NOT imported: importing ``tilefusion`` runs its heavy ``__init__`` (numba's
threading-layer pin, GPU/``cupy`` probes, ``basicpy``), which would make SquidMIP fail to
install/run on a machine without those. ``create_array`` here is a thin tensorstore-config
wrapper (the substantive reuse); the group writers are plain ``zarr.json`` JSON.

Two node kinds in an OME-NGFF v0.5 / zarr-v3 store:
  * arrays  — created by ``create_array`` (tensorstore writes the array ``zarr.json`` + chunks)
  * groups  — plain ``zarr.json`` with ``node_type: group`` + optional ``attributes.ome``,
              written by ``write_group`` (plate / well / row / field-image groups).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import tensorstore as ts

# Full-res arrays are chunked to this per-plane tile so a viewer reads a region without
# pulling the whole (Y, X) plane; downsample levels are smaller so they clamp to their shape.
_CHUNK_YX = 1024


def create_array(
    path,
    shape: Sequence[int],
    dtype,
    *,
    chunk: Optional[Sequence[int]] = None,
    max_workers: int = 4,
) -> ts.TensorStore:
    """Create a zarr v3 array at *path* (blosc-zstd) and return an open tensorstore handle.

    Shape is 5-D ``(t, c, z, y, x)`` (Squid canonical order). ``chunk`` defaults to one
    ``(1, 1, 1, <=1024, <=1024)`` tile; every chunk dim is clamped into ``[1, shape_i]`` so
    tiny arrays (e.g. 4x4 test frames) and odd shapes are always valid. ``delete_existing``
    makes a rewrite idempotent (a rerun overwrites cleanly).
    """
    shape = tuple(int(s) for s in shape)
    if chunk is None:
        y, x = shape[-2], shape[-1]
        chunk = (1, 1, 1, min(y, _CHUNK_YX), min(x, _CHUNK_YX))
    chunk = tuple(max(1, min(int(c), int(s))) for c, s in zip(chunk, shape))
    dt = np.dtype(dtype)

    config: dict[str, Any] = {
        "context": {
            "file_io_concurrency": {"limit": max_workers},
            "data_copy_concurrency": {"limit": max_workers},
        },
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(path)},
        "metadata": {
            "shape": list(shape),
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunk)}},
            "chunk_key_encoding": {"name": "default"},
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {
                    "name": "blosc",
                    "configuration": {"cname": "zstd", "clevel": 5, "shuffle": "bitshuffle"},
                },
            ],
            "data_type": dt.name,
            "dimension_names": ["t", "c", "z", "y", "x"],
        },
    }
    # create (overwriting any prior store so a rerun is idempotent). delete_existing may not be
    # combined with open=True, and create already returns an open, writable handle.
    return ts.open(config, create=True, delete_existing=True).result()


def write_array(store: ts.TensorStore, data: np.ndarray) -> None:
    """Write a whole array into an open store (contiguous copy so tensorstore is happy)."""
    store[...].write(np.ascontiguousarray(data)).result()


def write_group(path, ome: Optional[dict] = None) -> None:
    """Write a zarr v3 group ``zarr.json`` at *path*, with optional ``attributes.ome``.

    A bare group (``ome=None``) is a structural node (plate row); an ``ome`` payload carries
    the plate / well / multiscales+omero metadata that ndviewer and ome-zarr readers consume.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"zarr_format": 3, "node_type": "group", "attributes": {}}
    if ome is not None:
        doc["attributes"]["ome"] = ome
    (path / "zarr.json").write_text(json.dumps(doc, indent=2))
