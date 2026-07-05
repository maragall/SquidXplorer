"""SquidMIP reader: format-aware ingest for Squid individual-TIFF acquisitions.

``open_reader(path)`` dispatches on the on-disk format and returns a reader. Only the
individual-TIFFs layout is implemented (IMA-189); the other Squid output formats
(multi-page TIFF, OME-TIFF, Zarr) are detected and rejected with a clear message, marking
the seam where future readers plug in.

Individual-TIFFs layout (one channel per file), verified against real data::

    <acq>/
    ├── acquisition parameters.json
    ├── acquisition_channels.yaml
    ├── coordinates.csv
    └── 0/                                    # timepoint folder (1/, 2/, … if Nt>1)
        └── {region}_{fov}_{z}_{channel}.tiff

Discovery flow::

    open_reader ──► detect format ──► SquidReader
                                          │
        glob timepoint folders (0/,1/…) ──┤─► n_t
        glob *.tiff in t0, parse stems ───┤─► regions, fovs_per_region, channels, z-levels
        read ONE frame ──────────────────┤─► frame_shape, dtype   (NOT hardcoded)
        acquisition.yaml (or JSON) ───────┴─► dz_um, pixel_size_um, wellplate_format, Nz/Nt cross-check

The (region, fov, z, channel) index is parsed from FILENAMES — the ground truth. Scalar
metadata comes from acquisition.yaml (authoritative pixel size etc.), the flat JSON as a
legacy fallback. coordinates.csv is not read: for one-FOV-per-well the plate layout comes
from the well ID + wellplate_format; per-FOV stage positions are a deferred stitching concern.
read() constructs the path directly and returns exactly what tifffile decodes (native dtype),
refusing non-2D planes and dtypes outside {uint8, uint16}.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from squidmip._acquisition import load_acquisition_metadata
from squidmip._channels import load_channel_yaml, resolve_channels

# region has no underscore; fov and z are ints; channel is the remainder (may contain _ and -).
_STEM_RE = re.compile(r"^(?P<region>[^_]+)_(?P<fov>\d+)_(?P<z>\d+)_(?P<channel>.+)$")
_TIFF_SUFFIXES = (".tiff", ".tif")

# Squid grayscale planes are MONO8 (uint8) or MONO12/MONO16 (uint16); see
# software/squid/camera/utils.py get_available_pixel_formats. It never writes uint32/float
# grayscale (RGB formats are color -> ndim>2, rejected separately). We preserve the native
# dtype but refuse anything outside this set so a non-raw stack can't be silently projected.
_SUPPORTED_DTYPES = (np.dtype("uint8"), np.dtype("uint16"))


def _validate_plane(arr, path: Path):
    """Guard a decoded plane: 2D grayscale, dtype uint8/uint16. Returns arr unchanged."""
    if arr.ndim != 2:
        raise ValueError(
            f"{path.name} is not a 2D grayscale plane (shape {arr.shape}); "
            "color/RGB (brightfield) channels are not supported (deferred)."
        )
    if arr.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"{path.name} has dtype {arr.dtype}; Squid writes uint8 (MONO8) or uint16 "
            "(MONO12/MONO16). An unexpected dtype (e.g. uint32/float) usually means the input "
            "is not a raw Squid capture; refused rather than silently projected."
        )
    return arr


def _natural_key(s: str):
    """Natural sort key so well IDs order as B2 < B3 < B10 (not lexicographic B10 < B2)."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", s)]


def open_reader(path) -> "SquidReader":
    """Detect the acquisition format at *path* and return a reader.

    Raises NotImplementedError for formats other than individual TIFFs (the dispatch seam).
    """
    path = Path(path)
    if not path.is_dir():
        raise NotImplementedError(
            f"{path!s} is not a directory. Point open_reader at a Squid acquisition folder."
        )
    if (path / "ome_tiff").is_dir():
        raise NotImplementedError(
            "OME-TIFF tiles layout detected (ome_tiff/). Not implemented in IMA-189 "
            "(individual TIFFs only); this is the format-dispatch seam for a future reader."
        )
    if (path / "zarr.json").exists() or any(path.glob("*.zarr")):
        raise NotImplementedError(
            "Zarr layout detected. Not implemented in IMA-189; format-dispatch seam."
        )
    return SquidReader(path)


class SquidReader:
    """Lazy reader over a Squid individual-TIFF acquisition folder."""

    def __init__(self, path) -> None:
        self._path = Path(path)
        self._time_folders: Optional[list[Path]] = None
        self._index: Optional[dict] = None
        self._meta: Optional[dict] = None

    # -- timepoints -------------------------------------------------------
    def _discover_time_folders(self) -> list[Path]:
        if self._time_folders is None:
            numeric = [d for d in self._path.iterdir() if d.is_dir() and d.name.isdigit()]
            self._time_folders = (
                sorted(numeric, key=lambda d: int(d.name)) if numeric else [self._path]
            )
        return self._time_folders

    # -- index ------------------------------------------------------------
    def _build_index(self) -> dict:
        """Map {(region, fov, z, channel): file_suffix} from the first timepoint folder."""
        if self._index is not None:
            return self._index
        folder = self._discover_time_folders()[0]
        index: dict = {}
        for f in folder.iterdir():
            if f.suffix.lower() not in _TIFF_SUFFIXES:
                continue
            m = _STEM_RE.match(f.stem)
            if not m:
                continue  # e.g. {region}_{fov}_stack.tiff (multi-page) — not this reader's format
            key = (m["region"], int(m["fov"]), int(m["z"]), m["channel"])
            index[key] = f.suffix
        if not index:
            raise ValueError(
                "No Squid individual-TIFF files "
                "({region}_{fov}_{z}_{channel}.tiff) found in "
                f"{folder!s}"
            )
        self._index = index
        return index

    # -- metadata ---------------------------------------------------------
    @property
    def metadata(self) -> dict:
        if self._meta is not None:
            return self._meta
        index = self._build_index()
        time_folders = self._discover_time_folders()

        fovs: dict[str, set] = {}
        channels: set = set()
        z_levels: set = set()
        for (region, fov, z, channel) in index:
            fovs.setdefault(region, set()).add(fov)
            channels.add(channel)
            z_levels.add(z)
        # Deterministic, natural-sorted order (filesystem iteration order is not stable).
        regions = sorted(fovs, key=_natural_key)

        z_sorted = sorted(z_levels)
        n_z = len(z_sorted)
        n_t = len(time_folders)

        # Filenames + timepoint folders are ground truth; the recorded Nz/Nt are cross-checks.
        acq = load_acquisition_metadata(self._path)
        if acq["n_z_declared"] is not None and acq["n_z_declared"] != n_z:
            warnings.warn(
                f"Recorded Nz ({acq['n_z_declared']}) != distinct z levels in filenames "
                f"({n_z}); using the filename-derived value."
            )
        if acq["n_t_declared"] is not None and acq["n_t_declared"] != n_t:
            warnings.warn(
                f"Recorded Nt ({acq['n_t_declared']}) != timepoint folders found ({n_t}); "
                "using the folder-derived value."
            )

        # frame shape + dtype come from a real frame — they vary with binning / pixel format.
        sample_key = next(iter(index))
        sample_path = self._resolve_file(time_folders[0], sample_key, index[sample_key])
        sample = _validate_plane(tifffile.imread(sample_path), sample_path)

        self._meta = {
            "regions": regions,
            "fovs_per_region": {r: sorted(fovs[r]) for r in regions},
            "channels": resolve_channels(sorted(channels), load_channel_yaml(self._path)),
            "n_z": n_z,
            "z_levels": z_sorted,
            "dz_um": acq["dz_um"],
            "pixel_size_um": acq["pixel_size_um"],  # authoritative (acquisition.yaml), not recomputed
            "wellplate_format": acq["wellplate_format"],
            "frame_shape": tuple(sample.shape),
            "dtype": sample.dtype,
            "n_t": n_t,
        }
        return self._meta

    # -- read -------------------------------------------------------------
    def read(self, region, fov, channel, z, t=0):
        """Return one plane as a 2D array in its native dtype. Lazy: reads exactly one file."""
        index = self._build_index()
        time_folders = self._discover_time_folders()
        key = (str(region), int(fov), int(z), str(channel))
        if key not in index:
            raise KeyError(
                f"No such plane region={region!r} fov={fov} channel={channel!r} z={z}. "
                f"Known regions={sorted({k[0] for k in index})}, "
                f"channels={sorted({k[3] for k in index})}."
            )
        t = int(t)
        if not 0 <= t < len(time_folders):
            raise IndexError(f"t={t} out of range (n_t={len(time_folders)}).")
        path = self._resolve_file(time_folders[t], key, index[key])
        return _validate_plane(tifffile.imread(path), path)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _resolve_file(folder: Path, key, suffix: str) -> Path:
        """Build the plane's path, tolerating .tiff/.tif suffix drift across timepoints."""
        region, fov, z, channel = key
        candidate = folder / f"{region}_{fov}_{z}_{channel}{suffix}"
        if candidate.exists():
            return candidate
        for alt in _TIFF_SUFFIXES:
            other = folder / f"{region}_{fov}_{z}_{channel}{alt}"
            if other.exists():
                return other
        return candidate  # let tifffile raise a clear FileNotFoundError
