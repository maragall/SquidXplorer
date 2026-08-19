"""Save a region operator's fused mosaics the way maragall/stitcher exports them.

The stitcher's export format IS Squid's OME-TIFF convention (its
``tilefusion.ome_tiff_export.write_ome_tiff``: a tiled BigTIFF carrying OME-XML with axes
TZCYX, channel names and PhysicalSizeX/Y/Z in micrometres, matching Squid's own
utils_ome_tiff_writer). This writer calls that exact function — parity by construction, never a
copy — and lays the files out as a Squid OME-TIFF ACQUISITION beside the source:
``<operator>_<folder>/ome_tiff/{region}_0.ome.tiff``, one fused mosaic per region as that
region's single FOV, with the acquisition's sidecars copied and ``coordinates.csv`` rewritten
to the mosaic origins. ``open_reader`` therefore re-opens the folder (SquidOMEReader), so a
saved stitch drops straight back into the GUI.

Native resolution only: a region operator that streams a decimated LOOK is refused by name —
the acquisition sidecars declare the source's pixel size and a decimated mosaic under them
would be placed wrong. (One caveat, recorded not hidden: SquidOMEReader reads ``frame_shape``
from ONE sample file, so regions whose registered extents differ by a few pixels share one
declared frame shape; every plane still reads back at its own true shape.)
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from squidxplorer._engine import (MissingOperatorDependency, is_region_operator, operator_extra,
                                  operator_output, operator_produces, operator_saves_copy,
                                  split_operator_kwargs)
from squidxplorer._placement import Placement

_log = logging.getLogger(__name__)

_MM_PER_UM = 1e-3

#: Shortest round-trip repr for the rewritten stage coordinates, `_register._rewrite_csv`'s rule.
_COORDS_HEADER = "region,fov,x (mm),y (mm)"


def fused_format_dst(reader, operator: str) -> Optional[Path]:
    """``<operator>_<folder>`` beside the source when this save writes the fused format, else None.

    Declaration-driven: a region operator (``consumes={"fov"}``) producing intensity, not one
    that saves a registered copy of the acquisition (register), over a reader whose
    ``source_id`` is an on-disk directory. Anything else keeps the OME-Zarr path.
    """
    try:
        if (not is_region_operator(operator) or operator_saves_copy(operator)
                or operator_produces(operator) != "intensity"):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    src = Path(str(getattr(reader, "source_id", "") or ""))
    if not str(src) or not src.is_dir():
        return None
    return src.parent / f"{operator}_{src.name}"


def _refuse_by_declaration(operator: str, operator_kwargs: Optional[dict]) -> None:
    """Refuse by name any run whose output is not one fused intensity mosaic per region."""
    if not is_region_operator(operator):
        raise ValueError(
            f"operator {operator!r} runs per FOV; the fused format stores one mosaic per "
            "region. Save it in the acquisition's own format instead "
            "(_acq_output.write_acquisition_planes).")
    if operator_saves_copy(operator):
        raise ValueError(
            f"operator {operator!r} saves a registered COPY of the acquisition (copy=True); "
            "there is no fused mosaic to write. Its save already routes through the engine.")
    _collapses, produces = operator_output(operator, operator_kwargs)
    if produces != "intensity":
        raise ValueError(
            f"a run of {operator!r} with these parameters produces {produces!r} pixels; the "
            "fused OME-TIFF stores intensity mosaics only, so this result must go to the "
            "OME-Zarr writer (write_plate).")


def _ome_tiff_writer(operator: str):
    """The stitcher's own exporter — parity by construction; a missing install is a NAMED refusal."""
    try:
        from tilefusion.ome_tiff_export import write_ome_tiff
    except ImportError:
        extra = operator_extra(operator) or "stitch"
        raise MissingOperatorDependency(
            f"saving {operator!r} in the stitcher's format needs tilefusion "
            f"(pip install squidxplorer[{extra}]); it writes the OME-TIFF itself so the two "
            "outputs cannot drift.") from None
    return write_ome_tiff


def _placement_of(operator: str, region, image) -> Placement:
    """The streamed mosaic's Placement, refused by name when the stream carries none."""
    placement = getattr(image, "placement", None)
    if not isinstance(placement, Placement):
        raise ValueError(
            f"operator {operator!r} streamed a bare array for region {region!r}; the fused "
            "format needs the mosaic's Placement (origin and pixel size) to write a re-openable "
            "acquisition. Return a PlacedArray, or save as an OME-Zarr plate (write_plate).")
    return placement


def _refuse_decimated(operator: str, region, placement: Placement, native_px: float) -> None:
    """The acquisition sidecars declare *native_px*; a decimated LOOK under them places wrong."""
    if abs(float(placement.pixel_size_um) - float(native_px)) > 1e-6 * float(native_px):
        raise ValueError(
            f"operator {operator!r} streamed region {region!r} at "
            f"{placement.pixel_size_um} um/px where the acquisition's native pitch is "
            f"{native_px} um/px — a decimated LOOK, not the mosaic of record. The fused format "
            "stores native resolution only; save as an OME-Zarr plate (write_plate) instead.")


def _write_coordinates_csv(root: Path, origins_mm: dict) -> None:
    """One row per written region: the fused mosaic's top-left, the mosaic path's convention."""
    lines = [_COORDS_HEADER]
    for region in origins_mm:
        x_mm, y_mm = origins_mm[region]
        lines.append(f"{region},0,{x_mm!r},{y_mm!r}")
    (root / "coordinates.csv").write_text("\n".join(lines) + "\n")


def write_fused_acquisition(reader, operator: str, dst, *, regions=None,
                            operator_kwargs: Optional[dict] = None, workers=None,
                            on_well=None, on_error=None, stop=None) -> dict:
    """Fuse every selected region and write the mosaics as a Squid OME-TIFF acquisition at *dst*.

    One ``ome_tiff/{region}_0.ome.tiff`` per region — the stitcher's own export, at native
    resolution, the region's whole depth (the inner ``z_operator``'s declaration decides it) —
    plus the source's sidecars, with ``coordinates.csv`` carrying each mosaic's origin as its
    single FOV's position and the declared z count rewritten when the run collapses z. Built
    under a ``.partial`` name and renamed whole; returns a summary dict sharing
    ``write_from_stream``'s counting keys (a region operator owes ONE anchor field per region).
    """
    from squidxplorer import run_plate
    from squidxplorer._acq_output import _copy_sidecars, _rewrite_z_count
    from squidxplorer.projection import scope_wells

    _refuse_by_declaration(operator, operator_kwargs)
    split_operator_kwargs(operator, operator_kwargs)   # refuse an unknown key before any directory
    write_ome_tiff = _ome_tiff_writer(operator)        # ...and a missing writer dependency too
    src = Path(str(getattr(reader, "source_id", "") or ""))
    if not str(src) or not src.is_dir():
        raise ValueError(
            f"reader source {src!s} is not an on-disk acquisition folder; there is nothing to "
            "copy sidecars from. Save as OME-Zarr instead (write_plate).")

    meta = reader.metadata
    wells = scope_wells(meta, None, regions)
    owed = sum(1 for fovs in wells.values() if fovs)   # ONE anchor field per region
    channel_names = [c["name"] for c in meta["channels"]]
    if meta.get("pixel_size_um") is None:
        raise ValueError(
            "the acquisition metadata has no pixel_size_um; the fused format's sidecars must "
            "declare the pitch the mosaics are placed at. Add objective.pixel_size_um to "
            "acquisition.yaml, or save as an OME-Zarr plate (write_plate).")
    native_px = float(meta["pixel_size_um"])
    collapses, _produces = operator_output(operator, operator_kwargs)

    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    _copy_sidecars(src, tmp, [])                       # root sidecars only; no per-time folders
    if collapses:
        _rewrite_z_count(tmp)
    (tmp / "ome_tiff").mkdir()

    skipped_regions: set = set()

    def _on_error(region, fov, exc):
        skipped_regions.add(str(region))
        if on_error is not None:
            on_error(region, fov, exc)

    origins_mm: dict = {}
    n_written = 0
    stopped = False
    stream = run_plate(reader, operator=operator, regions=regions, n_fovs=None, workers=workers,
                       on_error=_on_error, operator_kwargs=operator_kwargs)
    try:
        for region, fov, image in stream:
            if stop is not None and stop():
                stopped = True
                break
            placement = _placement_of(operator, region, image)
            _refuse_decimated(operator, region, placement, native_px)
            arr = np.asarray(image)                    # (n_t, C, Z, H, W) — TCZYX, native
            write_ome_tiff(
                arr, tmp / "ome_tiff" / f"{region}_0.ome.tiff",
                pixel_size_um=(placement.pixel_size_um, placement.pixel_size_um),
                z_step_um=placement.z_step_um if arr.shape[2] > 1 else None,
                channel_names=channel_names, creator="squidxplorer")
            y_um, x_um = placement.origin_um
            origins_mm[str(region)] = (float(x_um) * _MM_PER_UM, float(y_um) * _MM_PER_UM)
            n_written += 1
            if on_well is not None:
                on_well(region, fov, image)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    if not stopped and stop is not None and stop():
        stopped = True

    _write_coordinates_csv(tmp, origins_mm)

    complete = (not stopped) and n_written >= owed
    if complete:
        if dst.exists():
            shutil.rmtree(dst)
        os.rename(tmp, dst)
        path = dst
    else:
        path = tmp   # left under the .partial name on purpose: it must not read as finished
    _log.info("fused save: %s — %d of %d region mosaic(s) written as Squid OME-TIFF.",
              path.name, n_written, owed)
    return {
        "path": str(path),
        "format": "fused-ome-tiff",
        "n_wells": len(wells),
        "n_fields": owed,
        "n_fields_written": n_written,
        "skipped_regions": sorted(skipped_regions),
        "complete": complete,
        "stopped": bool(stopped),
    }
