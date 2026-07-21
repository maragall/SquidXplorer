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
legacy fallback. coordinates.csv IS read (IMA-215) into ``metadata['fov_positions']`` —
``{(region, fov): (x_mm, y_mm, z_um)}`` — which the plate view uses to place multiple FOVs
per well as a mosaic. It is advisory, never authoritative: the (region, fov, z, channel)
index still comes from filenames, and an absent/ambiguous coordinates.csv yields ``{}`` (the
plate falls back to the well ID + wellplate_format layout) rather than a guessed position.
read() constructs the path directly and returns exactly what tifffile decodes (native dtype),
refusing non-2D planes and dtypes outside {uint8, uint16}.
"""

from __future__ import annotations

import csv
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


def _plate_key(region: str):
    """Sort well ids in true plate ROW-MAJOR order: A,B,...,Z,AA,AB,... with the column by integer
    (so B2 < B3 < B10, and B < AA — single-letter rows before double-letter, not lexicographic
    where "AA" < "B"). Downstream consumers (projection engine, plate viewer) then process wells
    top-to-bottom, left-to-right. Non-well-plate region names fall back after the plate wells.

    Changed from a plain natural sort in IMA-189: the old key ordered "AA" before "B", so a 1536wp
    plate processed row A, then the AA-AF rows, then B..Z — filling the plate view out of visual
    order. Row-major here fixes fill/scrub order for every slot. (Owner: IMA-185; see eng review.)
    """
    m = re.match(r"^([A-Za-z]+)(\d+)$", region)
    if not m:
        return (1, len(region), region, 0)          # non-plate ids: stable, after the wells
    return (0, len(m.group(1)), m.group(1).upper(), int(m.group(2)))


def read_fov_positions(path, fovs_per_region: dict) -> dict:
    """``{(region, fov): (x_mm, y_mm, z_um)}`` parsed from ``coordinates.csv`` (IMA-215).

    Two on-disk layouts exist and both are handled::

        Monkey-style   region,fov,z_level,x (mm),y (mm),z (um),time   -> fov read directly
        20x-style      region,x (mm),y (mm),z (mm)                    -> NO fov column

    For the 20x layout the FOV identity is the per-region ROW ORDER, which is an assumption, not
    a fact: it is only sound when the row count for a region matches the FOV count parsed from
    that region's filenames (filenames are ground truth — see the module docstring). When the
    counts disagree the region is OMITTED rather than mis-assigned, so a caller placing FOVs by
    coordinate fails that well loudly instead of silently putting images in the wrong physical
    position. A missing/unreadable coordinates.csv yields ``{}`` — absence is normal for older
    acquisitions, not an error.
    """
    csv_path = Path(path) / "coordinates.csv"
    if not csv_path.is_file():
        return {}
    try:
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error) as exc:
        warnings.warn(f"coordinates.csv present but unreadable ({exc}); ignoring it.")
        return {}
    if not rows:
        return {}

    def _col(fieldnames, *wanted):
        """Match a column by its normalised name ('x (mm)' -> 'x', 'z_level' -> 'zlevel')."""
        norm = {re.sub(r"\(.*?\)|[\s_]", "", (f or "")).lower(): f for f in fieldnames}
        for w in wanted:
            if w in norm:
                return norm[w]
        return None

    fields = list(rows[0].keys())
    c_region = _col(fields, "region")
    c_x, c_y, c_z = _col(fields, "x"), _col(fields, "y"), _col(fields, "z")
    c_fov = _col(fields, "fov")
    if not (c_region and c_x and c_y):
        warnings.warn(f"coordinates.csv has no region/x/y columns ({fields}); ignoring it.")
        return {}
    z_is_mm = bool(c_z) and "mm" in c_z.lower()   # z is (um) in Monkey, (mm) in 20x -> normalise to um

    positions: dict[tuple, tuple] = {}
    by_region: dict[str, list] = {}
    for row in rows:
        region = (row.get(c_region) or "").strip()
        if not region:
            continue
        try:
            x, y = float(row[c_x]), float(row[c_y])
            z = float(row[c_z]) if c_z and row.get(c_z) not in (None, "") else 0.0
        except (TypeError, ValueError):
            continue                          # a malformed row is skipped, never guessed at
        if z_is_mm:
            z *= 1000.0
        if c_fov is not None:
            try:
                positions[(region, int(float(row[c_fov])))] = (x, y, z)
            except (TypeError, ValueError, KeyError):
                continue
        else:
            by_region.setdefault(region, []).append((x, y, z))

    for region, coords in by_region.items():      # 20x layout: per-region row order IS the fov order
        known = fovs_per_region.get(region)
        if not known or len(known) != len(coords):
            warnings.warn(
                f"coordinates.csv has {len(coords)} row(s) for region {region!r} but its filenames "
                f"declare {len(known or [])} FOV(s); omitting the region's coordinates rather than "
                "mis-assigning them."
            )
            continue
        for fov, xyz in zip(known, coords):
            positions[(region, fov)] = xyz
    return positions


def open_reader(path) -> "SquidReader":
    """Detect the acquisition format at *path* and return a reader.

    Raises NotImplementedError for formats other than individual TIFFs (the dispatch seam).
    """
    path = Path(path)
    if not path.is_dir():
        raise NotImplementedError(
            f"{path!s} is not a directory. Point open_reader at a Squid acquisition folder."
        )
    ome = path / "ome_tiff"
    # OME-TIFF only if ome_tiff/ actually CONTAINS .ome.tiff files. Squid often leaves an EMPTY
    # ome_tiff/ placeholder next to an individual-TIFF acquisition — that empty folder must NOT
    # shadow the individual-TIFF reader.
    if ome.is_dir() and any(ome.rglob("*.ome.tif*")):
        return SquidOMEReader(path)
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

    # -- coordinates ------------------------------------------------------
    def _fov_positions(self, fovs_per_region: dict) -> dict:
        return read_fov_positions(self._path, fovs_per_region)

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
        regions = sorted(fovs, key=_plate_key)   # true plate row-major (A,B,...,Z,AA,...)

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

        fovs_per_region = {r: sorted(fovs[r]) for r in regions}
        self._meta = {
            "regions": regions,
            "fovs_per_region": fovs_per_region,
            "channels": resolve_channels(sorted(channels), load_channel_yaml(self._path)),
            "n_z": n_z,
            "z_levels": z_sorted,
            "dz_um": acq["dz_um"],
            "pixel_size_um": acq["pixel_size_um"],  # authoritative (acquisition.yaml), not recomputed
            "wellplate_format": acq["wellplate_format"],
            "frame_shape": tuple(sample.shape),
            "dtype": sample.dtype,
            "n_t": n_t,
            # IMA-215: per-FOV stage positions, {} when coordinates.csv is absent/unusable.
            "fov_positions": self._fov_positions(fovs_per_region),
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

    def plane_path(self, region, fov, channel, z, t=0) -> Path:
        """Path to one raw plane's TIFF on disk (no decode). The HCS viewer points the embedded
        ndviewer at these raw files directly (register_image), so the detail view is the true
        z-stack with zero extra bytes copied — read-only, never written."""
        index = self._build_index()
        time_folders = self._discover_time_folders()
        key = (str(region), int(fov), int(z), str(channel))
        if key not in index:
            raise KeyError(f"No such plane region={region!r} fov={fov} channel={channel!r} z={z}.")
        t = int(t)
        if not 0 <= t < len(time_folders):
            raise IndexError(f"t={t} out of range (n_t={len(time_folders)}).")
        return self._resolve_file(time_folders[t], key, index[key])

    def plane_ref(self, region, fov, channel, z, t=0) -> tuple:
        """(filepath, page_index) for one plane — the viewer registers this into ndviewer. Individual
        TIFFs hold one plane per file, so the page index is always 0."""
        return str(self.plane_path(region, fov, channel, z, t)), 0

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


# {region}_{fov} stem (region = well id, no trailing _<digits>; fov = trailing integer).
_OME_STEM_RE = re.compile(r"^(?P<region>.+)_(?P<fov>\d+)$")
_OME_SUFFIXES = (".ome.tiff", ".ome.tif", ".OME.TIFF", ".OME.TIF")


class SquidOMEReader:
    """Lazy reader over a Squid OME-TIFF acquisition.

    Layout (from Squid's utils_ome_tiff_writer): ``<acq>/ome_tiff/{region}_{fov}.ome.tiff`` — ONE
    file per well-FOV, each a 5-D ``TZCYX`` stack (dimension order written as TZCYX). Presents the
    SAME interface as :class:`SquidReader` (``metadata`` + ``read`` + ``plane_ref``), so the engine,
    CLI and viewer consume it unchanged. Reads one plane at a time (``TiffFile.pages[p]``) so memory
    stays bounded; the TiffFile handles are cached per file.
    """

    def __init__(self, path) -> None:
        self._path = Path(path)
        self._ome = self._path / "ome_tiff"
        self._files: Optional[dict] = None      # {(region, fov): Path}
        self._meta: Optional[dict] = None
        self._axes: Optional[str] = None        # non-spatial axes order, e.g. "TZC"
        self._handles: dict = {}                # Path -> tifffile.TiffFile (cached)

    def _discover(self) -> dict:
        if self._files is not None:
            return self._files
        files: dict = {}
        for f in sorted(self._ome.iterdir() if self._ome.is_dir() else []):
            name = f.name
            stem = next((name[: -len(s)] for s in _OME_SUFFIXES if name.endswith(s)), None)
            if stem is None:
                continue
            m = _OME_STEM_RE.match(stem)
            if m:
                files[(m["region"], int(m["fov"]))] = f
        if not files:
            raise ValueError(f"No {{region}}_{{fov}}.ome.tiff files found in {self._ome!s}")
        self._files = files
        return files

    def _tif(self, path: Path):
        tif = self._handles.get(path)
        if tif is None:
            tif = tifffile.TiffFile(path)
            self._handles[path] = tif
        return tif

    @property
    def metadata(self) -> dict:
        if self._meta is not None:
            return self._meta
        files = self._discover()
        sample = self._tif(next(iter(files.values()))).series[0]
        dims = dict(zip(sample.axes, sample.shape))     # e.g. {'T':2,'Z':3,'C':2,'Y':64,'X':80}
        n_t, n_z, n_c = dims.get("T", 1), dims.get("Z", 1), dims.get("C", 1)
        self._axes = "".join(a for a in sample.axes if a in "TZC")   # non-spatial order for paging

        fovs: dict[str, set] = {}
        for (region, fov) in files:
            fovs.setdefault(region, set()).add(fov)
        regions = sorted(fovs, key=_plate_key)

        # Channels come from acquisition_channels.yaml, in file order (== the writer's C-axis order).
        yaml_map = load_channel_yaml(self._path)
        names = list(yaml_map.keys())
        if len(names) != n_c:
            # yaml disagrees with the file — fall back to the OME channel names, else generic labels.
            ome_names = _ome_channel_names(self._tif(next(iter(files.values()))))
            names = [_normalize_local(n) for n in ome_names] if len(ome_names) == n_c \
                else [f"C{i}" for i in range(n_c)]
        channels = resolve_channels(names, yaml_map)

        acq = load_acquisition_metadata(self._path)
        if acq["n_z_declared"] is not None and acq["n_z_declared"] != n_z:
            warnings.warn(f"Recorded Nz ({acq['n_z_declared']}) != OME Z ({n_z}); using {n_z}.")
        fovs_per_region = {r: sorted(fovs[r]) for r in regions}
        self._meta = {
            "regions": regions,
            "fovs_per_region": fovs_per_region,
            "channels": channels,
            "n_z": n_z,
            "z_levels": list(range(n_z)),
            "dz_um": acq["dz_um"],
            "pixel_size_um": acq["pixel_size_um"],
            "wellplate_format": acq["wellplate_format"],
            "frame_shape": (int(dims.get("Y", sample.shape[-2])), int(dims.get("X", sample.shape[-1]))),
            "dtype": np.dtype(sample.dtype),
            "n_t": n_t,
            # IMA-215: same coordinate source as the individual-TIFF reader (sibling file).
            "fov_positions": read_fov_positions(self._path, fovs_per_region),
        }
        return self._meta

    def _page_index(self, t: int, z: int, c: int) -> int:
        """Flat IFD page index for (t, z, c), honouring the file's non-spatial axis order."""
        meta = self.metadata
        sizes = {"T": meta["n_t"], "Z": meta["n_z"], "C": len(meta["channels"])}
        pos = {"T": t, "Z": z, "C": c}
        order = self._axes or "TZC"
        return int(np.ravel_multi_index([pos[a] for a in order], [sizes[a] for a in order]))

    def _channel_index(self, channel) -> int:
        names = [c["name"] for c in self.metadata["channels"]]
        return names.index(str(channel))

    def read(self, region, fov, channel, z, t=0):
        """Return one plane as a 2D native-dtype array (reads exactly one IFD page)."""
        files = self._discover()
        key = (str(region), int(fov))
        if key not in files:
            raise KeyError(f"No such well/FOV region={region!r} fov={fov}. Known: {sorted(files)[:8]}")
        p = self._page_index(int(t), int(z), self._channel_index(channel))
        tif = self._tif(files[key])
        return _validate_plane(np.asarray(tif.pages[p].asarray()), files[key])

    def plane_ref(self, region, fov, channel, z, t=0) -> tuple:
        """(filepath, page_index) for one plane — the viewer registers this (with the page) into
        ndviewer, so the raw z-stack displays straight from the .ome.tiff, zero bytes copied."""
        p = self._page_index(int(t), int(z), self._channel_index(channel))
        return str(self._discover()[(str(region), int(fov))]), p


def _normalize_local(name: str) -> str:
    from squidmip._channels import normalize
    return normalize(name)


def _ome_channel_names(tif) -> list:
    """Best-effort channel names from the OME-XML (Channel Name=...), else []."""
    try:
        xml = tif.ome_metadata or ""
        return re.findall(r'<Channel[^>]*\bName="([^"]*)"', xml)
    except Exception:
        return []
