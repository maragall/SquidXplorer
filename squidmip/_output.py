"""IMA-184 output: canonical multiscale OME-zarr HCS plate + individual-TIFF export.

Consumes IMA-188's ``project_plate`` stream (single-thread — the engine parallelises the
projection internally and hands results back one at a time, so the writer needs no locking)
and writes each well as it arrives. Two outputs from one pass:

  1. ``<out>/plate.ome.zarr``  — OME-NGFF v0.5 HCS *plate* (zarr v3), matching Squid's canonical
     ``control/core/zarr_writer.py`` (single resolution level, no pyramid — a per-FOV pyramid is
     pointless for one small field; plate-view thumbnails are IMA-193's, not this writer's):
        plate.ome.zarr/                     zarr.json  = plate group (rows/columns/wells)
          {row}/                            zarr.json  = row group (bare)
            {col}/                          zarr.json  = well group (images -> raw fov ids)
              {fov}/                        zarr.json  = image group (multiscales + omero)
                0/                          array: full-res (T, C, 1, Y, X), native dtype
     Opens in ndviewer_light (directory-walk -> array ``0`` + ``omero`` colors) AND validates as
     a spec plate (plate/well group metadata) under an independent reader (zarr-python).

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
                                       ├─► field group: array 0 (full-res) + multiscales + omero
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
_WAVELENGTH_RE = re.compile(r"(?<!\d)(\d{3,4})(?!\d)")  # a standalone 3-4 digit nm in a channel name


# --- well id <-> row/col --------------------------------------------------------------------

def parse_well_id(region: str) -> tuple[str, str]:
    """Split a well id into (row_letters, col_digits) — vendored from Squid ``utils.parse_well_id``.

    Squid's canonical parser upper-cases then partitions alphabetic vs numeric characters
    (``"aa3" -> ("AA", "3")``); the HCS layout is ``plate.ome.zarr/{row}/{col}/{fov}/0`` and
    ndviewer_light rebuilds ``well_id = row_dir + col_dir`` by concatenation. So the column is
    NOT zero-padded — ``B2 -> B/2`` (``B/02`` would still be discovered, ``"02".isdigit()`` is
    True, but report the well as ``B02`` != the real id ``B2``, breaking well-id fidelity).

    We match Squid's accepted inputs exactly (uppercase, multi-letter rows, no padding) but,
    for a scientific tool, additionally ASSERT the canonical ``<letters><digits>`` shape and
    fail loud: a manual/no-plate region (Squid would silently accumulate stray chars into the
    column) must not be written to a mislabelled directory.
    """
    s = str(region).upper()
    letters = "".join(c for c in s if c.isalpha())
    digits = "".join(c for c in s if not c.isalpha())
    if not letters or not digits.isdigit() or letters + digits != s:
        raise ValueError(
            f"region {region!r} is not a canonical <letters><digits> well id (e.g. 'B2', 'AA3'); "
            "the HCS plate layout needs a row/column split. Manual/no-plate acquisitions are out "
            "of scope (IMA-189: well-plate layout only)."
        )
    return letters, digits


# Back-compat alias for the earlier name used in this module's history.
split_well = parse_well_id


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


def _multiscales(pixel_size_um: Optional[float], dz_um: Optional[float] = None) -> dict:
    """Single-level multiscales metadata (Squid canonical: one dataset ``0``, no pyramid).

    A per-FOV pyramid is pointless for one small field; plate-view thumbnails are a navigator
    concern (IMA-193), not this writer's. Scale/axes mirror Squid's zarr_writer exactly.
    """
    p = float(pixel_size_um) if pixel_size_um else 1.0
    dz = float(dz_um) if dz_um else 1.0
    return {
        "version": _NGFF_VERSION,
        "name": "0",
        "axes": [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": [
            {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, dz, p, p]}]}
        ],
    }


def _wavelength_nm(channel: dict) -> Optional[int]:
    """Best-effort emission wavelength (nm) parsed from the channel name, else None."""
    m = _WAVELENGTH_RE.search(channel.get("name", ""))
    return int(m.group(1)) if m else None


def _omero(channels: list[dict], dtype) -> dict:
    """omero rendering metadata (Squid shape): label, hex color (no '#'), window, wavelength."""
    dmax = float(np.iinfo(np.dtype(dtype)).max)
    out = []
    for ch in channels:
        entry = {
            "label": ch.get("display_name") or ch["name"],
            "color": str(ch["display_color"]).lstrip("#"),
            "active": True,
            "window": {"min": 0.0, "max": dmax, "start": 0.0, "end": dmax},
        }
        wl = _wavelength_nm(ch)
        if wl is not None:
            entry["emission_wavelength"] = {"value": wl, "unit": "nanometer"}
        out.append(entry)
    return {"channels": out}


# --- field + tiff writers --------------------------------------------------------------------

def _validate_image(image: np.ndarray, channels: list[dict]) -> None:
    """Fail loud on anything that isn't a projected ``(T, C, 1, Y, X)`` frame for these channels."""
    if image.ndim != 5 or image.shape[2] != 1:
        raise ValueError(
            f"expected a projected (T, C, 1, Y, X) array (z collapsed to 1), got shape {image.shape}. "
            "IMA-184 writes the projection output of IMA-188; a non-5D or Z>1 array is a seam bug."
        )
    if image.shape[1] != len(channels):
        raise ValueError(
            f"image has C={image.shape[1]} channels but metadata lists {len(channels)} "
            f"({[c['name'] for c in channels]}); channel/axis mismatch — refusing to mislabel omero."
        )


def _write_field(field_dir: Path, image: np.ndarray, channels: list[dict], pixel_size_um, dz_um=None) -> None:
    """Write one field: the single full-res array ``0`` + image-group multiscales + omero."""
    _validate_image(image, channels)
    store = create_array(field_dir / "0", image.shape, image.dtype)
    write_array(store, image)
    write_group(
        field_dir,
        {
            "version": _NGFF_VERSION,
            "multiscales": [_multiscales(pixel_size_um, dz_um)],
            "omero": _omero(channels, image.dtype),
        },
    )


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

    # Full plate/row/well group metadata written UP FRONT (layout is fully known from metadata).
    write_group(plate_dir, plate_metadata(wells.keys(), field_count=n_fovs))
    for region, fovs in wells.items():
        row, col = parse_well_id(region)
        write_group(plate_dir / row)  # bare row group
        # well.images paths are the RAW fov ids (Squid uses {fov} as the field dir + image path),
        # not a re-indexed 0-based field index — so a non-contiguous fov set stays faithful.
        write_group(
            plate_dir / row / col,
            {"version": _NGFF_VERSION, "well": {"images": [{"path": str(f)} for f in fovs]}},
        )

    channels = metadata["channels"]
    channel_names = [c["name"] for c in channels]
    pixel_size_um = metadata.get("pixel_size_um")
    dz_um = metadata.get("dz_um")

    n_written = 0
    for region, fov, image in stream:
        row, col = parse_well_id(region)
        # field directory is the RAW fov id (Squid convention), digit-named for ndviewer.
        _write_field(plate_dir / row / col / str(fov), image, channels, pixel_size_um, dz_um)
        if tiff:
            _write_tiffs(tiff_root, region, fov, image, channel_names)
        n_written += 1

    return {
        "plate": str(plate_dir),
        "tiff": str(tiff_root) if tiff else None,
        "n_wells": len(wells),
        "n_fields_written": n_written,
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
