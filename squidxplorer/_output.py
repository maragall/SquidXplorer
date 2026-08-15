"""Canonical multiscale OME-Zarr HCS plate output + individual-TIFF export."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import tifffile

from squidxplorer._engine import N_FOVS_LOOP_DEFAULT as _N_FOVS_LOOP_DEFAULT
from squidxplorer._engine import _default_workers, run_plate
from squidxplorer._volume import release as release_pages
from squidxplorer._zarr_store import create_array, write_group
from squidxplorer.contract import contract_stamp
from squidxplorer.projection import (
    INTENSITY,
    LABELS,
    RESULT_KINDS,
    cast_like,
    scope_wells,
)

_NGFF_VERSION = "0.5"

_PYRAMID_MIN_YX = 256
_PYRAMID_MAX_LEVELS = 6
_WRITE_WORKERS = min(4, _default_workers())


_DISK_SAFETY_FACTOR = 1.03
_DISK_NON_IMAGE_BYTES = 100 * 1024
_DISK_HEADROOM = 0.10               # keep this FRACTION of free space free
_DISK_MIN_FREE_BYTES = 256 * 1024 ** 2   # ...and never less than this absolute floor
#: Marker for "this store did not finish writing"; :func:`is_incomplete` is the one reader.
INCOMPLETE_MARKER = ".squidxplorer-incomplete"
# The pre-rename spelling (package was ``squidmip``): read and cleared, never written.
_MARKER_NAMES = (INCOMPLETE_MARKER, ".squidmip-incomplete")
_PARTIAL_PREFIX = "."               # dot prefix keeps partials out of digit-named field discovery
_PARTIAL_SUFFIX = ".partial"
_GB = 1024 ** 3


class InsufficientDiskSpaceError(OSError):
    """Refusing to start a write that would not fit (raised BEFORE anything is created)."""


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def plate_pyramid_factor(frame_shape, **kw) -> float:
    """Total written pixels per level-0 pixel, i.e. the exact pyramid overhead of one field."""
    shapes = pyramid_shapes(frame_shape, **kw)
    y0, x0 = shapes[0]
    return float(sum(y * x for y, x in shapes) / (y0 * x0))


def estimate_write_bytes(metadata: dict, *, n_fovs: Optional[int] = 1, regions=None,
                         tiff: bool = False, n_z: int = 1,
                         region_operator: bool = False, wells=None) -> int:
    """Bytes :func:`write_from_stream` will need for this acquisition (uncompressed estimate).

    ``wells`` is the ALREADY-RESOLVED ``{region: [fov, ...]}`` scope when the caller has one
    (``write_from_stream`` always does) — the estimate then counts exactly the scoped fields
    instead of re-deriving them from ``n_fovs``/``regions``.
    """
    frame_shape = metadata.get("frame_shape")
    channels = metadata.get("channels") or []
    if not frame_shape or not channels:
        return 0
    ny, nx = int(frame_shape[0]), int(frame_shape[1])
    itemsize = np.dtype(metadata.get("dtype", "uint16")).itemsize
    fovs_per_region = metadata.get("fovs_per_region") or {}

    if wells is not None:
        scoped = [r for r in wells if r in fovs_per_region]
    else:
        scoped = (list(fovs_per_region) if regions is None
                  else [r for r in regions if r in fovs_per_region])
    if region_operator:
        # A region operator emits one fused mosaic per region, not one frame per FOV.
        px_per_field = _region_mosaic_pixels(metadata, scoped, (ny, nx))
    else:
        if wells is not None:
            n_fields = sum(len(f) for f in wells.values())
        elif n_fovs is None:
            n_fields = sum(len(fovs_per_region[r]) for r in scoped)
        else:                               # select_fovs takes at most n_fovs per region
            n_fields = sum(min(int(n_fovs), len(fovs_per_region[r])) for r in scoped)
        px_per_field = n_fields * ny * nx

    frame_bytes = px_per_field * int(metadata.get("n_t", 1) or 1) * len(channels) * int(n_z) * itemsize
    total = frame_bytes * plate_pyramid_factor((ny, nx))
    if tiff:
        total += frame_bytes                # a second, uncompressed, pyramid-free copy
    return int(total * _DISK_SAFETY_FACTOR) + _DISK_NON_IMAGE_BYTES


def _region_mosaic_pixels(metadata: dict, scoped, frame_shape) -> int:
    """Total pixels a region operator writes: the summed area of each region's fused mosaic."""
    ny, nx = int(frame_shape[0]), int(frame_shape[1])
    positions_um = metadata.get("fov_positions_um") or {}
    px_um = metadata.get("pixel_size_um")
    fovs_per_region = metadata.get("fovs_per_region") or {}
    if not positions_um or not px_um:
        return sum(len(fovs_per_region.get(r, ())) for r in scoped) * ny * nx

    from squidxplorer._placement import fov_offsets_px, mosaic_extent_px

    total = 0
    for region in scoped:
        fovs = list(fovs_per_region.get(region, ()))
        if not fovs:
            continue
        try:
            offsets = fov_offsets_px(positions_um, region, fovs, float(px_um))
            h, w = mosaic_extent_px(offsets, (ny, nx))
        except (KeyError, ValueError):
            h, w = ny * len(fovs), nx        # positions unusable for this region: over-estimate
        total += int(h) * int(w)
    return total


def free_bytes(path) -> int:
    """Free bytes on the filesystem that will hold *path*, or -1 when it cannot be stat-ed."""
    p = Path(path).absolute()
    for candidate in (p, *p.parents):
        if candidate.is_dir():
            try:
                return int(shutil.disk_usage(candidate).free)
            except OSError:
                return -1
    return -1


def check_disk_space(out_dir, required_bytes: int, *, headroom: Optional[float] = None,
                     min_free_bytes: Optional[int] = None, what: str = "this write") -> None:
    """Raise :class:`InsufficientDiskSpaceError` unless *required_bytes* fits with headroom."""
    if required_bytes <= 0:
        return
    frac = _env_float("SQUIDXPLORER_DISK_HEADROOM", _DISK_HEADROOM) if headroom is None else float(headroom)
    floor = (int(_env_float("SQUIDXPLORER_MIN_FREE_BYTES", _DISK_MIN_FREE_BYTES))
             if min_free_bytes is None else int(min_free_bytes))
    free = free_bytes(out_dir)
    if free < 0:
        return                                    # can't stat the disk -> don't block the run
    reserve = max(int(free * frac), floor)
    budget = free - reserve
    if required_bytes > budget:
        raise InsufficientDiskSpaceError(
            f"refusing to start: {what} needs ~{required_bytes / _GB:.2f} GB but "
            f"{Path(out_dir).absolute()} has {free / _GB:.2f} GB free "
            f"(keeping {reserve / _GB:.2f} GB headroom, so {max(budget, 0) / _GB:.2f} GB usable). "
            "Free space, pick another disk, or lower the headroom "
            "(disk_headroom= / SQUIDXPLORER_DISK_HEADROOM)."
        )


def is_incomplete(plate_dir) -> bool:
    """True while a plate store is mid-write, or if the write that made it never finished."""
    return any((Path(plate_dir) / name).exists() for name in _MARKER_NAMES)


def incomplete_reason(plate_dir) -> Optional[str]:
    """One sentence naming this store's shortfall, or ``None`` when it is whole."""
    root = Path(plate_dir)
    marker = next((p for d in (root, root / "plate.ome.zarr")
                   for name in _MARKER_NAMES
                   if (p := d / name).exists()), None)
    if marker is None:
        return None
    try:
        info = json.loads(marker.read_text())
    except (OSError, ValueError):
        info = {}
    owed, wrote = info.get("fields"), info.get("fields_written")
    how = "was stopped mid-write" if info.get("stopped") else "did not finish"
    if isinstance(owed, int) and isinstance(wrote, int):
        return (f"the write that produced this plate {how}: {wrote} of {owed} field(s) landed, "
                f"so wells its own metadata promises are not on disk")
    # No counts means the marker was never replaced, i.e. the process died mid-run.
    return (f"the write that produced this plate {how} — it was still running when it ended, so "
            f"an unknown number of the wells its metadata promises are not on disk")


def _mark_incomplete(plate_dir: Path, info: dict) -> None:
    (plate_dir / INCOMPLETE_MARKER).write_text(json.dumps(info, indent=2))


def _clear_incomplete(plate_dir: Path) -> None:
    for name in _MARKER_NAMES:
        try:
            (plate_dir / name).unlink()
        except FileNotFoundError:
            pass


def _partial_dir(well_dir: Path, fov) -> Path:
    return well_dir / f"{_PARTIAL_PREFIX}{fov}{_PARTIAL_SUFFIX}"


def _publish(tmp: Path, final: Path) -> None:
    """Atomically make *tmp* visible as *final* — a field appears whole or not at all."""
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)


def _cleanup_partials(directory: Path) -> int:
    """Remove any leftover ``.{fov}.partial`` intermediates under *directory*. Returns the count."""
    n = 0
    if not directory.is_dir():
        return 0
    for child in directory.iterdir():
        if child.is_dir() and child.name.startswith(_PARTIAL_PREFIX) and child.name.endswith(_PARTIAL_SUFFIX):
            shutil.rmtree(child, ignore_errors=True)
            n += 1
    return n


def parse_well_id(region: str) -> tuple[str, str]:
    """Split a well id into (row_letters, col_digits); case preserved, canonical shape enforced."""
    s = str(region)
    letters = "".join(c for c in s if c.isalpha())
    digits = "".join(c for c in s if not c.isalpha())
    if not letters or not digits.isdigit() or letters + digits != s:
        raise ValueError(
            f"region {region!r} is not a canonical <letters><digits> well id (e.g. 'B2', 'AA3'); "
            "the HCS plate layout needs a row/column split. Manual/no-plate acquisitions are out "
            "of scope (IMA-189: well-plate layout only)."
        )
    return letters, digits


split_well = parse_well_id


def _row_sort_key(row: str):
    # Plate row order: A..Z then AA..AF (shorter labels first, then lexicographic).
    return (len(row), row)


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


def _downsample_yx(image: np.ndarray) -> np.ndarray:
    """Halve a ``(T, C, Z, Y, X)`` field in Y and X by 2x2 block-mean, native dtype kept."""
    fy = 2 if image.shape[-2] >= 2 else 1
    fx = 2 if image.shape[-1] >= 2 else 1
    y = (image.shape[-2] // fy) * fy                       # crop to a multiple of the axis factor
    x = (image.shape[-1] // fx) * fx
    cropped = image[..., :y, :x]
    ds = cropped.reshape(*cropped.shape[:-2], y // fy, fy, x // fx, fx).mean(axis=(-3, -1), dtype=np.float32)
    return cast_like(ds, image.dtype, copy=False)


def _subsample_yx(image: np.ndarray) -> np.ndarray:
    """Halve a ``(T, C, Z, Y, X)`` field in Y and X by taking one pixel per 2x2 block (labels)."""
    fy = 2 if image.shape[-2] >= 2 else 1
    fx = 2 if image.shape[-1] >= 2 else 1
    y = (image.shape[-2] // fy) * fy                       # crop to a multiple of the axis factor
    x = (image.shape[-1] // fx) * fx
    return np.ascontiguousarray(image[..., :y:fy, :x:fx])


#: How a pyramid level is derived from the one below it, per ``produces`` declaration.
_REDUCERS = {
    INTENSITY: (_downsample_yx, "mean", "2x2 block mean"),
    LABELS: (_subsample_yx, "nearest", "2x2 nearest (object ids are never averaged)"),
}


def _reducer_for(produces: str):
    """``(reducer, ngff_type, description)`` for a result kind, or raise naming the kind."""
    try:
        return _REDUCERS[str(produces)]
    except KeyError:
        raise ValueError(
            f"cannot build a pyramid for result kind {produces!r}: this writer knows how to "
            f"coarsen {sorted(_REDUCERS)} and refuses to guess. Averaging pixels whose meaning it "
            f"does not know is how object ids became other object ids. Known result kinds are "
            f"{sorted(RESULT_KINDS)}; add an entry to squidxplorer._output._REDUCERS for a new one."
        ) from None


def _pyramid(image: np.ndarray, *, min_yx: int = _PYRAMID_MIN_YX,
             max_levels: int = _PYRAMID_MAX_LEVELS,
             produces: str = INTENSITY) -> list[np.ndarray]:
    """Level list ``[full-res, /2, /4, ...]``, halving until the coarsest fits *min_yx*."""
    reduce_yx, _type, _desc = _reducer_for(produces)
    levels = [image]
    while (max(levels[-1].shape[-2:]) > int(min_yx) and len(levels) < int(max_levels)):
        levels.append(reduce_yx(levels[-1]))
    return levels


def pyramid_shapes(frame_shape, *, min_yx: int = _PYRAMID_MIN_YX,
                   max_levels: int = _PYRAMID_MAX_LEVELS) -> list[tuple[int, int]]:
    """The ``(Y, X)`` of every pyramid level :func:`_pyramid` would write, from the shape alone."""
    y, x = int(frame_shape[0]), int(frame_shape[1])
    if y < 1 or x < 1:
        raise ValueError(f"frame_shape must be positive, got {frame_shape!r}")
    shapes = [(y, x)]
    while max(shapes[-1]) > int(min_yx) and len(shapes) < int(max_levels):
        y, x = shapes[-1]
        fy, fx = (2 if y >= 2 else 1), (2 if x >= 2 else 1)
        shapes.append((y // fy, x // fx))
    return shapes


def _multiscales(level_shapes: list[tuple], pixel_size_um: Optional[float], dz_um: Optional[float] = None,
                 position_um: Optional[tuple] = None, produces: str = INTENSITY) -> dict:
    """multiscales metadata for a per-FOV pyramid: one ``datasets`` entry per level."""
    _reduce, ngff_type, ngff_desc = _reducer_for(produces)
    p = float(pixel_size_um) if pixel_size_um else 1.0
    dz = float(dz_um) if dz_um else 1.0
    y0, x0 = level_shapes[0]
    datasets = []
    for i, (y, x) in enumerate(level_shapes):
        sy, sx = p * (y0 / y), p * (x0 / x)   # coarse levels have a larger physical pixel
        xforms: list[dict] = [{"type": "scale", "scale": [1.0, 1.0, dz, sy, sx]}]
        if position_um is not None:
            xforms.append({"type": "translation",
                           "translation": [0.0, 0.0, 0.0, float(position_um[1]), float(position_um[0])]})
        datasets.append({"path": str(i), "coordinateTransformations": xforms})
    doc = {
        "version": _NGFF_VERSION,
        "name": "0",
        "type": ngff_type,
        "metadata": {"method": f"squidxplorer._output.{_reduce.__name__}", "description": ngff_desc},
        "axes": [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": datasets,
    }
    return doc


def _excitation_nm(channel: dict) -> Optional[float]:
    """The channel's EXCITATION wavelength in nm, or None when its name states none."""
    from squidxplorer._channels import excitation_nm

    return excitation_nm(channel.get("name", ""))


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
        wl = _excitation_nm(ch)
        if wl is not None:
            entry["excitation_wavelength"] = {"value": wl, "unit": "nanometer"}
        out.append(entry)
    return {"channels": out}


def _validate_image(image: np.ndarray, channels: list[dict]) -> None:
    """Fail loud on anything that isn't a ``(T, C, Nz, Y, X)`` operator result for these channels."""
    if image.ndim != 5:
        raise ValueError(
            f"expected a 5-D (T, C, Z, Y, X) operator result, got shape {image.shape} "
            f"({image.ndim}-D). Every operator output is TCZYX — z-reducers give Z=1, plane-ops "
            "give the acquisition's full depth; anything else is a seam bug."
        )
    if any(int(s) < 1 for s in image.shape):
        raise ValueError(
            f"(T, C, Z, Y, X) array has an empty axis: shape {image.shape}. Every axis must have "
            "at least one element — a zero-sized field is not a field."
        )
    if image.shape[1] != len(channels):
        raise ValueError(
            f"image has C={image.shape[1]} channels but metadata lists {len(channels)} "
            f"({[c['name'] for c in channels]}); channel/axis mismatch — refusing to mislabel omero."
        )


def field_origin_um(centre_um, frame_shape, pixel_size_um) -> Optional[tuple[float, float]]:
    """Stage-µm ``(x, y)`` of a field's TOP-LEFT pixel, from its recorded CENTRE position."""
    if centre_um is None or not pixel_size_um or frame_shape is None:
        return None
    p = float(pixel_size_um)
    if not p > 0:
        return None
    h, w = int(frame_shape[0]), int(frame_shape[1])
    return (float(centre_um[0]) - w * p / 2.0, float(centre_um[1]) - h * p / 2.0)


def _write_field(field_dir: Path, image: np.ndarray, channels: list[dict], pixel_size_um, dz_um=None,
                 position_um: Optional[tuple] = None, produces: str = INTENSITY) -> int:
    """Write one field: pyramid levels ``0..L`` + multiscales + omero; returns the level count.

    Z is written one plane at a time to keep the writer's transient at one plane's pyramid.
    """
    _validate_image(image, channels)
    level_shapes = pyramid_shapes(image.shape[-2:])
    stores = [create_array(field_dir / str(i), (*image.shape[:3], *shape), image.dtype)
              for i, shape in enumerate(level_shapes)]
    for z in range(image.shape[2]):
        plane = np.asarray(image[:, :, z:z + 1])   # asarray: a spilled mosaic is read back here
        levels = _pyramid(plane, produces=produces)
        # zip() would silently truncate if the two ladders ever disagreed.
        if len(levels) != len(stores):
            raise AssertionError(
                f"pyramid ladder disagrees with pyramid_shapes: {len(levels)} levels built for "
                f"{len(stores)} arrays created (frame {image.shape[-2:]})")
        for store, level in zip(stores, levels):
            store[:, :, z:z + 1].write(np.ascontiguousarray(level)).result()
        del plane, levels
        release_pages(image)   # drop the pages just read, if the producer spilled to a scratch file
    write_group(
        field_dir,
        {
            "version": _NGFF_VERSION,
            "multiscales": [_multiscales(level_shapes, pixel_size_um, dz_um, position_um,
                                         produces=produces)],
            "omero": _omero(channels, image.dtype),
        },
    )
    return len(stores)


def _write_tiffs(tiff_root: Path, region: str, fov: int, image: np.ndarray, channel_names: list[str]) -> None:
    """Individual per-plane TIFFs: tiff/{t}/{region}_{fov}_{z}_{channel}.tiff, native dtype."""
    n_t, _, n_z = image.shape[:3]
    for t in range(n_t):
        tdir = tiff_root / str(t)
        tdir.mkdir(parents=True, exist_ok=True)
        for z in range(n_z):
            for c_i, channel in enumerate(channel_names):
                plane = np.asarray(image[t, c_i, z])   # (Y, X), native dtype
                final = tdir / f"{region}_{fov}_{z}_{channel}.tiff"
                tmp = final.with_name(final.name + _PARTIAL_SUFFIX)
                try:
                    tifffile.imwrite(tmp, plane)
                    os.replace(tmp, final)
                except BaseException:
                    tmp.unlink(missing_ok=True)
                    raise


_ROI_TABLE_NAME = "FOV_ROI_table"
_ROI_INDEX_KEY = "FieldIndex"

# SquidXplorer key (µm) -> ngio/fractal column (µm). A RENAME, never a conversion.
_NGIO_COLUMN = {
    "x_um": "x_micrometer",
    "y_um": "y_micrometer",
    "z_um": "z_micrometer",
    "len_x_um": "len_x_micrometer",
    "len_y_um": "len_y_micrometer",
    "len_z_um": "len_z_micrometer",
    "x_original_um": "x_micrometer_original",
    "y_original_um": "y_micrometer_original",
    "z_original_um": "z_micrometer_original",
}
_ROI_NUMERIC_UM = list(_NGIO_COLUMN)          # X columns, in written order
_ROI_OBS_COLUMNS = ("path_in_well",)          # string columns -> AnnData obs (ngio PLATE_COLUMNS)


def fov_roi_records_um(fovs, positions_um, frame_shape, pixel_size_um, *,
                       dz_um: Optional[float] = None, n_z: int = 1) -> list[dict]:
    """One ROI record per FOV of one region — all lengths and positions in MICROMETRES (``_um``)."""
    p = float(pixel_size_um or 0.0)
    if not p > 0:
        return []
    h, w = int(frame_shape[0]), int(frame_shape[1])
    len_x_um, len_y_um = w * p, h * p
    # The ROI describes the physical volume the plane came from: z-spacing x n planes.
    len_z_um = float(dz_um) * max(1, int(n_z)) if dz_um else 1.0

    raw = []
    for fov in fovs:
        corner = field_origin_um(positions_um.get(fov), (h, w), p)
        if corner is None:
            continue
        raw.append((fov, float(corner[0]), float(corner[1])))
    if not raw:
        return []
    x0 = min(x for _, x, _ in raw)      # the region's own origin: its top-left corner
    y0 = min(y for _, _, y in raw)
    return [
        {
            "FieldIndex": f"FOV_{fov}",   # ngio index convention; the RAW fov id, so the name
            "path_in_well": str(fov),     # and path_in_well point at the field dir on disk
            "x_um": x - x0, "y_um": y - y0, "z_um": 0.0,
            "len_x_um": len_x_um, "len_y_um": len_y_um, "len_z_um": len_z_um,
            "x_original_um": x, "y_original_um": y, "z_original_um": 0.0,
        }
        for fov, x, y in raw
    ]


def _check_roi_micrometres(records: list[dict], frame_extent_um: float) -> None:
    """Fail loud if the FOV pitch says the positions were millimetres wearing a ``_um`` key."""
    xs = sorted({round(r["x_um"], 6) for r in records})
    ys = sorted({round(r["y_um"], 6) for r in records})
    gaps = [b - a for v in (xs, ys) for a, b in zip(v, v[1:]) if b > a]
    if gaps and min(gaps) < frame_extent_um / 100.0:
        raise ValueError(
            f"FOV pitch is {min(gaps):.4g} µm for a {frame_extent_um:.4g} µm frame — that is "
            "millimetres in a `_um` key (1000x). Refusing to write an FOV_ROI_table that would "
            "put every downstream tool 1000x off; positions come from metadata['fov_positions_um']."
        )


def _zarr_write_anndata_roi_table(table_dir: Path, records: list[dict]) -> None:
    """Write *records* as an AnnData-encoded zarr v3 group at *table_dir* (no anndata dependency)."""
    import zarr

    root = zarr.open_group(str(table_dir), mode="w", zarr_format=3)
    root.attrs.update({
        "encoding-type": "anndata", "encoding-version": "0.1.0",
        "type": "roi_table",
        "fractal_table_version": "1",     # fractal-tasks-core <= 1.6 refuses a table without it
        "table_version": "1",             # ngio's spelling (extra="allow" keeps both)
        "backend": "anndata_v1",
        "index_key": _ROI_INDEX_KEY, "index_type": "str",
    })

    x = np.array([[float(r[k]) for k in _ROI_NUMERIC_UM] for r in records], dtype=np.float64)
    arr = root.create_array("X", shape=x.shape, dtype="float64")
    arr[...] = x
    arr.attrs.update({"encoding-type": "array", "encoding-version": "0.2.0"})

    def _string_column(group, name: str, values: list[str]) -> None:
        a = group.create_array(name, shape=(len(values),), dtype=str)
        a[...] = np.array(values, dtype=object)
        a.attrs.update({"encoding-type": "string-array", "encoding-version": "0.2.0"})

    obs = root.create_group("obs")
    obs.attrs.update({"encoding-type": "dataframe", "encoding-version": "0.2.0",
                      "_index": _ROI_INDEX_KEY, "column-order": list(_ROI_OBS_COLUMNS)})
    _string_column(obs, _ROI_INDEX_KEY, [str(r[_ROI_INDEX_KEY]) for r in records])
    for col in _ROI_OBS_COLUMNS:
        _string_column(obs, col, [str(r[col]) for r in records])

    var = root.create_group("var")
    var.attrs.update({"encoding-type": "dataframe", "encoding-version": "0.2.0",
                      "_index": "_index", "column-order": []})
    _string_column(var, "_index", [_NGIO_COLUMN[k] for k in _ROI_NUMERIC_UM])

    for empty in ("layers", "obsm", "varm", "obsp", "varp", "uns"):
        g = root.create_group(empty)
        g.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})


def write_fov_roi_table(image_dir, records: list[dict], *, table_name: str = _ROI_TABLE_NAME) -> Optional[Path]:
    """Write ``<image_dir>/tables/<table_name>`` from :func:`fov_roi_records_um` records."""
    if not records:
        return None
    _check_roi_micrometres(records, float(records[0]["len_x_um"]))
    image_dir = Path(image_dir)
    tables_dir = image_dir / "tables"
    table_dir = tables_dir / table_name
    tmp = tables_dir / f"{_PARTIAL_PREFIX}{table_name}{_PARTIAL_SUFFIX}"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        _zarr_write_anndata_roi_table(tmp, records)
        _publish(tmp, table_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    # The tables group is a plain zarr group whose attrs index its members.
    existing = []
    zj = tables_dir / "zarr.json"
    if zj.exists():
        try:
            existing = json.loads(zj.read_text()).get("attributes", {}).get("tables", [])
        except (OSError, ValueError):
            existing = []
    names = list(dict.fromkeys([*existing, table_name]))
    zj.write_text(json.dumps({"zarr_format": 3, "node_type": "group",
                              "attributes": {"tables": names}}, indent=2))
    return table_dir


def write_from_stream(
    metadata: dict,
    stream: Iterator[tuple[str, int, np.ndarray]],
    out_dir,
    *,
    n_fovs: Optional[int] = 1,
    tiff: bool = False,
    on_well=None,
    write_workers: int = _WRITE_WORKERS,
    stop=None,
    regions=None,
    check_disk: bool = True,
    disk_headroom: Optional[float] = None,
    min_free_bytes: Optional[int] = None,
    roi_table: bool = True,
    region_operator: bool = False,
    n_z: int = 1,
    produces: str = INTENSITY,
) -> dict:
    """Write the plate + (optionally) TIFFs from a ``(region, fov, image)`` stream and *metadata*.

    ``on_well`` runs on a writer thread and must be thread-safe; ``stop()`` is polled before each
    submit and drains in-flight writes when it returns True.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from threading import Lock

    out_dir = Path(out_dir)
    plate_dir = out_dir / "plate.ome.zarr"
    tiff_root = out_dir / "tiff"

    # THE resolver both engine loops use — never a second derivation of the run's scope. A
    # {region: [fov, ...]} mapping (an ROI run) therefore counts exactly the fields the stream
    # will yield: the writer used to owe EVERY FOV of a mapped region and mark its own store
    # incomplete after a run that did exactly what was asked.
    wells = scope_wells(metadata, n_fovs, regions)
    if region_operator:
        # A region operator owes ONE fused result per region, published under the ANCHOR fov
        # (fovs[0]) the region loop yields it as.
        wells = {r: f[:1] for r, f in wells.items() if f}
    # field_count is a single plate-level scalar: the most fields any scoped well carries.
    field_count = max((len(f) for f in wells.values()), default=0)

    if check_disk:
        need = estimate_write_bytes(metadata, n_fovs=n_fovs, regions=regions, tiff=tiff,
                                    region_operator=region_operator, n_z=n_z, wells=wells)
        scope = "this plate write" if regions is None else f"this {len(wells)}-well write"
        check_disk_space(out_dir, need, headroom=disk_headroom, min_free_bytes=min_free_bytes,
                         what=scope)

    # One fused region at a time when the result has real depth: a deep fused mosaic is GBs,
    # so the writer pool must not re-widen the memory window behind the region stream.
    if region_operator and int(n_z) > 1:
        write_workers = 1

    # Full plate/row/well group metadata is written up front; the contract stamp rides on the
    # plate group once.
    write_group(plate_dir, plate_metadata(wells.keys(), field_count=field_count),
                attributes=contract_stamp())
    # The store declares itself unfinished from its first byte until its last.
    fields_owed = sum(len(f) for f in wells.values())
    _mark_incomplete(plate_dir, {"wells": list(wells), "fields": fields_owed})
    for region, fovs in wells.items():
        row, col = parse_well_id(region)
        write_group(plate_dir / row)  # bare row group
        # well.images paths are the RAW fov ids, so a non-contiguous fov set stays faithful.
        write_group(
            plate_dir / row / col,
            {"version": _NGFF_VERSION, "well": {"images": [{"path": str(f)} for f in fovs]}},
        )

    channels = metadata["channels"]
    channel_names = [c["name"] for c in channels]
    pixel_size_um = metadata.get("pixel_size_um")
    dz_um = metadata.get("dz_um")
    positions_um = metadata.get("fov_positions_um") or {}   # {} when there is no coordinates.csv

    remaining = {r: len(f) for r, f in wells.items()}   # fields still owed per region
    remaining_lock = Lock()

    def _write_one(region, fov, image):
        row, col = parse_well_id(region)
        well_dir = plate_dir / row / col
        # Frame shape comes from the IMAGE, so a cropped/binned field is placed at its true extent.
        origin_um = field_origin_um(positions_um.get((region, fov)), image.shape[-2:], pixel_size_um)
        # Build in a ".{fov}.partial" directory and rename into place, so the field only ever
        # appears complete.
        tmp = _partial_dir(well_dir, fov)
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            levels = _write_field(tmp, image, channels, pixel_size_um, dz_um,
                                  position_um=origin_um, produces=produces)
            if tiff:
                _write_tiffs(tiff_root, region, fov, image, channel_names)
            _publish(tmp, well_dir / str(fov))
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)     # never leave this field's intermediate behind
            raise
        with remaining_lock:                            # per-REGION cleanup as the region finishes
            remaining[region] -= 1
            done_region = remaining[region] <= 0
        if done_region:
            _cleanup_partials(well_dir)
            if roi_table:
                # Persist path only — the live viewer has coordinates.csv already.
                fov_pos = {f: positions_um[(region, f)] for f in wells[region]
                           if (region, f) in positions_um}
                write_fov_roi_table(well_dir, fov_roi_records_um(
                    wells[region], fov_pos, image.shape[-2:], pixel_size_um,
                    dz_um=dz_um, n_z=int(metadata.get("n_z", 1) or 1)))
        if on_well is not None:  # live consumer (plate viewer): render tile + push to ndviewer
            on_well(region, fov, image)
        return levels

    n_written = 0
    n_levels = 1
    stopped = False
    n_writers = max(1, int(write_workers))
    try:
        with ThreadPoolExecutor(max_workers=n_writers, thread_name_prefix="squidxplorer-write") as ex:
            pending: set = set()
            for region, fov, image in stream:
                if stop is not None and stop():
                    stopped = True
                    break
                pending.add(ex.submit(_write_one, region, fov, image))
                if len(pending) >= n_writers:    # keep <= n_writers wells in flight (bounded memory)
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        n_levels = f.result()    # re-raises a writer-thread exception here
                        n_written += 1
            for f in pending:                     # drain the tail (and any in-flight after a stop)
                n_levels = f.result()
                n_written += 1
    finally:
        # Close the producer promptly on a stop/exception; a plain iterator has no close().
        close = getattr(stream, "close", None)
        if callable(close):
            close()
        # Sweep every well's leftovers, however this run ended.
        for region in wells:
            row, col = parse_well_id(region)
            _cleanup_partials(plate_dir / row / col)

    # Complete means every field this run owed is on disk, not merely "nobody pressed stop".
    complete = (not stopped) and n_written >= fields_owed
    if complete:
        _clear_incomplete(plate_dir)   # last act of a finished write: the store is now trustworthy
    else:
        _mark_incomplete(plate_dir, {"wells": list(wells), "fields": fields_owed,
                                     "fields_written": n_written,
                                     "stopped": bool(stopped)})

    return {
        "plate": str(plate_dir),
        "tiff": str(tiff_root) if tiff else None,
        "n_wells": len(wells),
        "n_fields": fields_owed,          # what this run OWED, beside what it wrote
        "n_fields_written": n_written,
        "levels": n_levels,
        "complete": complete,
        "stopped": bool(stopped),
    }


def write_plate(
    reader,
    out_dir,
    *,
    n_fovs=_N_FOVS_LOOP_DEFAULT,
    workers: Optional[int] = None,
    operator: str = "mip",
    tiff: bool = False,
    on_well=None,
    write_workers: int = _WRITE_WORKERS,
    stop=None,
    on_error=None,
    regions=None,
    check_disk: bool = True,
    disk_headroom: Optional[float] = None,
    min_free_bytes: Optional[int] = None,
    roi_table: bool = True,
    operator_kwargs: Optional[dict] = None,
) -> dict:
    """Project a plate and write the canonical OME-zarr + individual TIFFs; returns the manifest."""
    from squidxplorer._engine import (bind_operator, is_region_operator,
                                      operator_output, split_operator_kwargs)

    metadata = reader.metadata
    region_operator = is_region_operator(operator)
    if operator_kwargs:
        # Refuse an unknown parameter BEFORE any directory is made — on BOTH arms. The region
        # arm used to skip this, so a typo'd stitch knob was caught only after the plate
        # skeleton and the incomplete marker were on disk.
        if region_operator:
            split_operator_kwargs(operator, operator_kwargs)
        else:
            bind_operator(operator, operator_kwargs)
    # Result depth and pixel meaning come off the record's OWN output query (inner_param for a
    # region operator, its own declarations otherwise), so the disk estimate and the pyramid
    # reducer cannot disagree with what is then written — and the writer never reconstructs a
    # declaration from a parameter name.
    collapses_z, produces_out = operator_output(operator, operator_kwargs)
    n_z_out = 1 if collapses_z else int(metadata.get("n_z", 1) or 1)

    stream = run_plate(reader, operator=operator, n_fovs=n_fovs, workers=workers,
                       on_error=on_error, regions=regions, operator_kwargs=operator_kwargs)
    n_fovs_concrete = ((None if region_operator else 1)
                       if n_fovs is _N_FOVS_LOOP_DEFAULT else n_fovs)
    return write_from_stream(metadata, stream, out_dir, n_fovs=n_fovs_concrete, tiff=tiff,
                             on_well=on_well,
                             write_workers=write_workers, stop=stop, regions=regions,
                             check_disk=check_disk, disk_headroom=disk_headroom,
                             min_free_bytes=min_free_bytes, roi_table=roi_table,
                             region_operator=region_operator, n_z=n_z_out,
                             produces=produces_out)
