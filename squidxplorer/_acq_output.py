"""Save a z-collapsing per-FOV run in the acquisition's OWN format, full resolution.

One projected plane per FOV, bit-exact, written as ``{region}_{fov}_0_{channel}.{ext}`` under
``<operator>_<acquisition-folder>`` beside the source: the folder a user finds without knowing
NGFF. Sidecars are copied (never image files) and the copied z count is rewritten to 1 so the
output round-trips through ``open_reader``. Everything else keeps the OME-Zarr writer.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
import yaml

from squidxplorer._acquisition import _LEGACY_PARAMS
from squidxplorer._engine import (bind_operator, is_region_operator, operator_consumes,
                                  operator_produces)

_TIFF_SUFFIXES = (".tiff", ".tif")

#: Squid's own extension rule (utils_acquisition.get_image_filepath): uint16 -> .tiff, else .bmp.
_DTYPE_SUFFIX = {np.dtype(np.uint16): ".tiff", np.dtype(np.uint8): ".bmp"}


def acquisition_format_dst(reader, operator: str) -> Optional[Path]:
    """``<operator>_<folder>`` beside the source when this save writes acquisition format, else None.

    Declaration-driven: a per-FOV, z-collapsing, intensity-producing operator over a reader whose
    ``source_id`` is an on-disk directory. Anything else keeps the OME-Zarr path.
    """
    try:
        if (is_region_operator(operator) or "z" not in operator_consumes(operator)
                or operator_produces(operator) != "intensity"):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    src = Path(str(getattr(reader, "source_id", "") or ""))
    if not str(src) or not src.is_dir():
        return None
    return src.parent / f"{operator}_{src.name}"


def _refuse_by_declaration(operator: str) -> None:
    """Refuse by name any operator whose output is not one projected intensity plane per FOV."""
    if is_region_operator(operator):
        raise ValueError(
            f"operator {operator!r} consumes a whole well's FOVs; acquisition-format output is "
            "one projected plane per FOV and has no place for a fused region. Save it as an "
            "OME-Zarr plate instead (write_plate).")
    if "z" not in operator_consumes(operator):
        raise ValueError(
            f"operator {operator!r} keeps z (consumes="
            f"{sorted(operator_consumes(operator))}); this writer only knows one projected "
            "z-plane per FOV, so a full-depth result must go to the OME-Zarr writer "
            "(write_plate).")
    if operator_produces(operator) != "intensity":
        raise ValueError(
            f"operator {operator!r} produces {operator_produces(operator)!r} pixels; the "
            "acquisition format stores intensity planes only, so this result must go to the "
            "OME-Zarr writer (write_plate).")


def _copy_sidecars(src: Path, dst: Path, time_names: "list[str]") -> int:
    """Copy root and per-time-folder sidecar files into *dst*, preserving layout; never images."""
    # Lazy: importing _register registers its operator, a side effect a save may take on
    # but a module import must not.
    from squidxplorer._register import _SIDECAR_SUFFIXES

    copied = 0
    for f in sorted(src.iterdir()):
        if f.is_file() and f.suffix.lower() in _SIDECAR_SUFFIXES:
            shutil.copy2(f, dst / f.name)
            copied += 1
    for name in time_names:
        folder = src / name
        if not folder.is_dir():
            continue
        (dst / name).mkdir(parents=True, exist_ok=True)
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in _SIDECAR_SUFFIXES:
                shutil.copy2(f, dst / name / f.name)
                copied += 1
    return copied


def _rewrite_nz_line(path: Path) -> bool:
    """Rewrite ``z_stack: nz:`` to 1 in place, byte-identical elsewhere; False when not found."""
    lines = path.read_bytes().decode("utf-8").splitlines(keepends=True)
    in_z_stack = False
    for i, line in enumerate(lines):
        if line[:1] and not line[:1].isspace():
            in_z_stack = line.startswith("z_stack:")
            continue
        if in_z_stack and line.lstrip().startswith("nz:"):
            indent = line[: len(line) - len(line.lstrip())]
            ending = "\r\n" if line.endswith("\r\n") else "\n"
            lines[i] = f"{indent}nz: 1{ending}"
            path.write_bytes("".join(lines).encode("utf-8"))
            return True
    return False


def _rewrite_z_count(root: Path) -> None:
    """Declare 1 z level in the copied metadata so the output opens without a mismatch warning."""
    acq_yaml = root / "acquisition.yaml"
    if acq_yaml.exists():
        if _rewrite_nz_line(acq_yaml):
            return
        data = yaml.safe_load(acq_yaml.read_text()) or {}
        if isinstance(data.get("z_stack"), dict) and "nz" in data["z_stack"]:
            data["z_stack"]["nz"] = 1
            acq_yaml.write_text(yaml.safe_dump(data, sort_keys=False))
        return
    legacy = root / _LEGACY_PARAMS
    if legacy.exists():
        try:
            params = json.loads(legacy.read_text())
        except ValueError:
            return
        if isinstance(params, dict) and "Nz" in params:
            params["Nz"] = 1
            legacy.write_text(json.dumps(params, indent=2))


def _channel_suffix(reader, region, fov, channel, z_levels, plane) -> str:
    """The source file's own extension for *channel* when discoverable, else Squid's dtype rule."""
    plane_path = getattr(reader, "plane_path", None)
    if callable(plane_path):
        try:
            suffix = Path(plane_path(region, fov, channel, z_levels[0], 0)).suffix
            if suffix:
                return suffix
        except (KeyError, IndexError):
            pass
    return _DTYPE_SUFFIX.get(plane.dtype, ".tiff")


def _write_plane(path: Path, plane: np.ndarray) -> None:
    """One plane to disk by its extension: tifffile for TIFF, Pillow for everything else."""
    if path.suffix.lower() in _TIFF_SUFFIXES:
        tifffile.imwrite(path, plane)
        return
    from PIL import Image

    Image.fromarray(plane).save(path)


def write_acquisition_planes(reader, operator: str, dst, *, regions=None,
                             operator_kwargs: Optional[dict] = None, workers=None,
                             on_well=None, on_error=None, stop=None) -> dict:
    """Run *operator* over every FOV and write the results as a Squid acquisition at *dst*.

    Built under a ``.partial`` name and renamed whole, so a killed or stopped run never reads as
    a finished acquisition. Returns a summary dict sharing ``write_from_stream``'s counting keys
    (``n_fields`` owed, ``n_fields_written``, ``complete``, ``stopped``).
    """
    from squidxplorer import run_plate
    from squidxplorer.projection import scope_wells

    _refuse_by_declaration(operator)
    bind_operator(operator, operator_kwargs)   # refuse an unknown parameter before any directory
    src = Path(reader.source_id)
    if not src.is_dir():
        raise ValueError(
            f"reader source {src!s} is not an on-disk acquisition folder; there is nothing to "
            "copy sidecars from. Save as OME-Zarr instead (write_plate).")

    meta = reader.metadata
    wells = scope_wells(meta, None, regions)
    fields_owed = sum(len(fovs) for fovs in wells.values())
    channels = [c["name"] for c in meta["channels"]]
    z_levels = list(meta["z_levels"])
    n_t = int(meta["n_t"])
    src_time_dirs = sorted((d.name for d in src.iterdir() if d.is_dir() and d.name.isdigit()),
                           key=int)
    time_names = src_time_dirs if len(src_time_dirs) == n_t else [str(i) for i in range(n_t)]

    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    _copy_sidecars(src, tmp, src_time_dirs)
    _rewrite_z_count(tmp)
    for name in time_names:
        (tmp / name).mkdir(exist_ok=True)

    skipped_regions: set = set()

    def _on_error(region, fov, exc):
        skipped_regions.add(str(region))
        if on_error is not None:
            on_error(region, fov, exc)

    suffixes: dict = {}
    n_written = 0
    stopped = False
    stream = run_plate(reader, operator=operator, regions=regions, n_fovs=None, workers=workers,
                       on_error=_on_error, operator_kwargs=operator_kwargs)
    try:
        for region, fov, image in stream:
            if stop is not None and stop():
                stopped = True
                break
            arr = np.asarray(image)
            if arr.ndim != 5 or arr.shape[2] != 1:
                raise ValueError(
                    f"operator {operator!r} streamed shape {arr.shape}; a z-collapsing per-FOV "
                    "result is (n_t, n_channels, 1, y, x) and anything else cannot be written "
                    "as one plane per (fov, channel, t).")
            if arr.shape[0] != n_t or arr.shape[1] != len(channels):
                raise ValueError(
                    f"operator {operator!r} streamed {arr.shape[0]} timepoint(s) x "
                    f"{arr.shape[1]} channel(s) where the acquisition has {n_t} x "
                    f"{len(channels)}; refusing to guess which is which.")
            for t, tname in enumerate(time_names):
                for c_i, channel in enumerate(channels):
                    plane = np.ascontiguousarray(arr[t, c_i, 0])
                    if channel not in suffixes:
                        suffixes[channel] = _channel_suffix(reader, region, fov, channel,
                                                            z_levels, plane)
                    _write_plane(tmp / tname / f"{region}_{fov}_0_{channel}{suffixes[channel]}",
                                 plane)
            n_written += 1
            if on_well is not None:
                on_well(region, fov, image)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    if not stopped and stop is not None and stop():
        stopped = True

    complete = (not stopped) and n_written >= fields_owed
    if complete:
        if dst.exists():
            shutil.rmtree(dst)
        os.rename(tmp, dst)
        path = dst
    else:
        path = tmp   # left under the .partial name on purpose: it must not read as finished
    return {
        "path": str(path),
        "format": "acquisition",
        "n_wells": len(wells),
        "n_fields": fields_owed,
        "n_fields_written": n_written,
        "skipped_regions": sorted(skipped_regions),
        "complete": complete,
        "stopped": bool(stopped),
    }
