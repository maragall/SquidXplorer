"""IMA-184 output: canonical multiscale OME-zarr HCS plate + individual-TIFF export.

Consumes IMA-188's ``project_plate`` stream (single-thread — the engine parallelises the
projection internally and hands results back one at a time, so the writer needs no locking)
and writes each well as it arrives. Two outputs from one pass:

  1. ``<out>/plate.ome.zarr``  — OME-NGFF v0.5 HCS *plate* (zarr v3):
        plate.ome.zarr/                     zarr.json  = plate group (rows/columns/wells)
          {row}/                            zarr.json  = row group (bare)
            {col}/                          zarr.json  = well group (images -> field indices)
              {field}/                      zarr.json  = image group (multiscales + omero)
                0/  1/  2/                  arrays: 0 = full-res (T,C,1,Y,X), 1/2 = pyramid
     Opens in ndviewer_light (directory-walk -> array ``0`` + ``omero`` colors) AND validates
     as a spec plate (plate/well group metadata + a >=2-level pyramid ndviewer_light ignores).

  2. ``<out>/tiff/{t}/{region}_{fov}_0_{channel}.tiff`` — individual per-plane TIFFs in Squid's
     filename convention, z collapsed to ``0`` (the projection), native dtype. You Yan's
     "individual tiff output": channel identity lives in the filename, no OME-XML, so it drops
     straight into Nick's existing Squid-reading workflow.

Flow::

    reader.metadata ─► select_fovs ─► plate/row/well GROUP metadata written UP FRONT
                                       (full layout known from metadata, so the stream's
                                        completion-order arrival needs no ordering logic)
    project_plate(reader, ...) ─► (region, fov, (T,C,1,Y,X))
                                       │  per well, as it arrives:
                                       ├─► field group: arrays 0/1/2 (pyramid) + multiscales+omero
                                       └─► individual TIFFs (one per channel, per timepoint)

Colors come from ``metadata.channels[].display_color`` (IMA-189 already resolves them, mapped
by name, raising on an unrecognised channel) — the writer never re-parses the acquisition YAML.
Channel order in ``omero`` and in the TIFF filenames follows ``metadata.channels`` order, which
is exactly the array's C-axis order (IMA-183 builds the C axis from that list).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import tifffile

from squidmip._engine import project_plate
from squidmip._zarr_store import create_array, write_array, write_group
from squidmip.projection import select_fovs

_NGFF_VERSION = "0.5"
_PYRAMID_FACTORS = (1, 2, 4)  # level 0 (full), 1 (/2), 2 (/4); clamped to array size
_WELL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")  # e.g. B2 -> ("B", "2"); AA12 -> ("AA", "12")


# --- well id <-> row/col --------------------------------------------------------------------

def split_well(region: str) -> tuple[str, str]:
    """Split a well id into (row, col) so ``row + col == region`` round-trips.

    ndviewer_light rebuilds ``well_id = row_dir + col_dir`` by string concatenation, so the
    column is NOT zero-padded (``B2`` -> ``B``/``2``, not ``B``/``02``) — padding would break
    discovery. A region that is not ``<letters><digits>`` (e.g. a manual/no-plate acquisition)
    is refused loud rather than written to a mislabelled directory.
    """
    m = _WELL_RE.match(region)
    if not m:
        raise ValueError(
            f"region {region!r} is not a <letters><digits> well id (e.g. 'B2'); the HCS plate "
            "layout needs a row/column split. Manual/no-plate acquisitions are out of scope."
        )
    return m.group(1), m.group(2)


def _row_sort_key(row: str):
    # Plate row order: A..Z then AA..AF (shorter labels first, then lexicographic).
    return (len(row), row)


# --- NGFF metadata builders -----------------------------------------------------------------

def plate_metadata(regions: Iterable[str], field_count: int, name: str = "plate") -> dict:
    """OME-NGFF v0.5 ``plate`` group metadata from the well ids (rows/columns/wells)."""
    splits = [(r, *split_well(r)) for r in regions]
    rows = sorted({row for _, row, _ in splits}, key=_row_sort_key)
    cols = sorted({col for _, _, col in splits}, key=int)
    wells = [
        {"path": f"{row}/{col}", "rowIndex": rows.index(row), "columnIndex": cols.index(col)}
        for _, row, col in splits
    ]
    return {
        "version": _NGFF_VERSION,
        "plate": {
            "name": name,
            "rows": [{"name": r} for r in rows],
            "columns": [{"name": c} for c in cols],
            "wells": wells,
            "field_count": int(field_count),
        },
    }


def _multiscales(pixel_size_um: Optional[float], n_levels: int) -> dict:
    """multiscales metadata: datasets named 0..n-1, y/x scale doubling per level."""
    p = float(pixel_size_um) if pixel_size_um else 1.0
    datasets = [
        {
            "path": str(i),
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, 1.0, p * (2 ** i), p * (2 ** i)]}
            ],
        }
        for i in range(n_levels)
    ]
    return {
        "version": _NGFF_VERSION,
        "name": "0",
        "axes": [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": datasets,
    }


def _omero(channels: list[dict], dtype) -> dict:
    """omero rendering metadata: per-channel label + hex color (no '#') + a full-range window."""
    dmax = float(np.iinfo(np.dtype(dtype)).max)
    return {
        "channels": [
            {
                "label": ch.get("display_name") or ch["name"],
                "color": str(ch["display_color"]).lstrip("#"),
                "active": True,
                "window": {"min": 0.0, "max": dmax, "start": 0.0, "end": dmax},
            }
            for ch in channels
        ]
    }


# --- pyramid ---------------------------------------------------------------------------------

def _downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean reduce Y and X by *factor* (crop the remainder), preserving dtype."""
    t, c, z, y, x = arr.shape
    ny, nx = y // factor, x // factor
    cropped = arr[..., : ny * factor, : nx * factor]
    reduced = cropped.reshape(t, c, z, ny, factor, nx, factor).mean(axis=(4, 6))
    return reduced.astype(arr.dtype)


def pyramid_levels(arr: np.ndarray, factors: tuple[int, ...] = _PYRAMID_FACTORS) -> list[np.ndarray]:
    """Full-res + downsampled levels; a level is emitted only while both Y//f and X//f >= 1."""
    levels: list[np.ndarray] = []
    for f in factors:
        if arr.shape[-2] // f < 1 or arr.shape[-1] // f < 1:
            break
        levels.append(arr if f == 1 else _downsample(arr, f))
    return levels


# --- field + tiff writers --------------------------------------------------------------------

def _write_field(field_dir: Path, image: np.ndarray, channels: list[dict], pixel_size_um) -> int:
    """Write one field's pyramid arrays (0/1/2) + image-group metadata. Returns level count."""
    levels = pyramid_levels(image)
    for i, level in enumerate(levels):
        store = create_array(field_dir / str(i), level.shape, level.dtype)
        write_array(store, level)
    write_group(
        field_dir,
        {
            "version": _NGFF_VERSION,
            "multiscales": [_multiscales(pixel_size_um, len(levels))],
            "omero": _omero(channels, image.dtype),
        },
    )
    return len(levels)


def _write_tiffs(tiff_root: Path, region: str, fov: int, image: np.ndarray, channel_names: list[str]) -> None:
    """Individual per-plane TIFFs: tiff/{t}/{region}_{fov}_0_{channel}.tiff, native dtype."""
    n_t = image.shape[0]
    for t in range(n_t):
        tdir = tiff_root / str(t)
        tdir.mkdir(parents=True, exist_ok=True)
        for c_i, channel in enumerate(channel_names):
            plane = image[t, c_i, 0]  # (Y, X), native dtype, z collapsed
            tifffile.imwrite(tdir / f"{region}_{fov}_0_{channel}.tiff", plane)


# --- orchestration ---------------------------------------------------------------------------

def write_from_stream(
    metadata: dict,
    stream: Iterator[tuple[str, int, np.ndarray]],
    out_dir,
    *,
    n_fovs: int = 1,
    tiff: bool = True,
) -> dict:
    """Write the plate + TIFFs from a ``(region, fov, image)`` stream and *metadata*.

    The core of :func:`write_plate`, split out so it can be driven clean-room in tests with a
    fabricated metadata dict + a hand-built stream (no reader, no data on disk).
    """
    out_dir = Path(out_dir)
    plate_dir = out_dir / "plate.ome.zarr"
    tiff_root = out_dir / "tiff"

    wells = select_fovs(metadata, n_fovs=n_fovs)  # {region: [fov, ...]}, deterministic
    field_index = {  # (region, fov) -> 0-based field index within the well
        (region, fov): i for region, fovs in wells.items() for i, fov in enumerate(fovs)
    }

    # Full plate/row/well group metadata written UP FRONT (layout is fully known from metadata).
    write_group(plate_dir, plate_metadata(wells.keys(), field_count=n_fovs))
    for region, fovs in wells.items():
        row, col = split_well(region)
        write_group(plate_dir / row)  # bare row group
        write_group(
            plate_dir / row / col,
            {"version": _NGFF_VERSION, "well": {"images": [{"path": str(i)} for i in range(len(fovs))]}},
        )

    channels = metadata["channels"]
    channel_names = [c["name"] for c in channels]
    pixel_size_um = metadata.get("pixel_size_um")

    n_written = 0
    levels_seen = 0
    for region, fov, image in stream:
        row, col = split_well(region)
        field = field_index[(region, fov)]
        levels_seen = _write_field(plate_dir / row / col / str(field), image, channels, pixel_size_um)
        if tiff:
            _write_tiffs(tiff_root, region, fov, image, channel_names)
        n_written += 1

    return {
        "plate": str(plate_dir),
        "tiff": str(tiff_root) if tiff else None,
        "n_wells": len(wells),
        "n_fields_written": n_written,
        "pyramid_levels": levels_seen,
    }


def write_plate(
    reader,
    out_dir,
    *,
    n_fovs: int = 1,
    workers: Optional[int] = None,
    projector: str = "mip",
    tiff: bool = True,
) -> dict:
    """Project a plate (IMA-188) and write the canonical OME-zarr + individual TIFFs.

    Consumes :func:`squidmip.project_plate` lazily — each projected well is written as it
    arrives, so peak memory stays at the engine's bounded window, never the whole plate.

    Parameters
    ----------
    reader:
        An IMA-189 ``SquidReader`` (from ``open_reader``).
    out_dir:
        Destination directory; receives ``plate.ome.zarr/`` and (if *tiff*) ``tiff/``.
    n_fovs, workers, projector:
        Passed straight to :func:`squidmip.project_plate`.
    tiff:
        Also write the individual per-plane TIFF export (default True).

    Returns
    -------
    dict
        Manifest: output paths, well/field counts, pyramid level count.
    """
    metadata = reader.metadata
    stream = project_plate(reader, n_fovs=n_fovs, workers=workers, projector=projector)
    return write_from_stream(metadata, stream, out_dir, n_fovs=n_fovs, tiff=tiff)
