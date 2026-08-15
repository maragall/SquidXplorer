"""Format-aware readers for Squid acquisitions.

``open_reader(path)`` dispatches on the on-disk format and returns one of four readers
(individual TIFFs, multi-page TIFF, OME-TIFF, OME-NGFF Zarr) behind one interface:
``metadata`` (micrometres, ``_um``-suffixed) plus ``read(region, fov, channel, z, t)``.
"""

from __future__ import annotations

import bisect
import contextlib
import json
import re
import threading
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from squidxplorer._acquisition import Acquisition, load_acquisition_metadata
from squidxplorer._channels import excitation_nm, fallback_color, load_channel_yaml, resolve_channels
from squidxplorer.contract import check_plate_contract
from squidxplorer.contract.reader import SquidAcquisitionReader  # noqa: F401 (re-export)

# region has no underscore; fov and z are ints; channel is the remainder (may contain _ and -).
_STEM_RE = re.compile(r"^(?P<region>[^_]+)_(?P<fov>\d+)_(?P<z>\d+)_(?P<channel>.+)$")
_TIFF_SUFFIXES = (".tiff", ".tif")

# MULTI_PAGE_TIFF stems; fov zero-padding is a deployment setting, so the width is parsed.
_STACK_STEM_RE = re.compile(r"^(?P<region>[^_]+)_(?P<fov>\d+)_stack$")

# TIFF tags Squid's multi-page writer populates per page.
_TAG_IMAGE_DESCRIPTION = 270
_TAG_PAGE_NAME = 285

# Squid grayscale planes are MONO8 (uint8) or MONO12/MONO16 (uint16).
_SUPPORTED_DTYPES = (np.dtype("uint8"), np.dtype("uint16"))


class _TiffHandles:
    """Cached ``tifffile.TiffFile`` handles with a per-file lock: decoding seeks, so it is not re-entrant."""

    def __init__(self) -> None:
        self._handles: dict = {}                # Path -> tifffile.TiffFile
        self._locks: dict = {}                  # Path -> threading.Lock
        self._guard = threading.Lock()          # guards the two dicts, never held during a read

    def _entry(self, path: Path):
        with self._guard:
            tif = self._handles.get(path)
            if tif is None:
                tif = self._handles[path] = tifffile.TiffFile(path)
                self._locks[path] = threading.Lock()
            return tif, self._locks[path]

    @contextlib.contextmanager
    def read(self, path: Path):
        """Yield the ``TiffFile`` for *path* with its lock held. Decode inside the block."""
        tif, lock = self._entry(path)
        with lock:
            yield tif

    def page(self, path: Path, index: int):
        """One decoded IFD page, validated."""
        with self.read(path) as tif:
            return _validate_plane(np.asarray(tif.pages[int(index)].asarray()), path)


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


_COORDS_NAME = "coordinates.csv"
# Header spellings drift across Squid generations; match the axis letter of a column mentioning mm.
_X_COL_RE = re.compile(r"^\s*x\b.*\(\s*mm\s*\)", re.I)
_Y_COL_RE = re.compile(r"^\s*y\b.*\(\s*mm\s*\)", re.I)

#: Which coordinates.csv a set of FOV positions came from.
COORDS_EXECUTED = "executed"
COORDS_PLANNED = "planned"


def _coords_path(root):
    """The best available ``coordinates.csv`` as ``(path, source)``: executed (``0/``) preferred over planned."""
    root = Path(root)
    executed = root / "0" / _COORDS_NAME
    if executed.exists():
        return executed, COORDS_EXECUTED
    planned = root / _COORDS_NAME
    if planned.exists():
        return planned, COORDS_PLANNED
    return None, None


def _coord_columns(fieldnames) -> tuple[str, str]:
    """Locate the x/y millimetre columns in a coordinates.csv header, failing loud if absent."""
    names = list(fieldnames or [])
    x = next((n for n in names if n and _X_COL_RE.match(n)), None)
    y = next((n for n in names if n and _Y_COL_RE.match(n)), None)
    if x is None or y is None:
        raise ValueError(
            f"{_COORDS_NAME} has no recognisable x/y millimetre columns (header: {names}). "
            "Expected something like 'x (mm)' and 'y (mm)'; without them FOVs cannot be placed."
        )
    return x, y


_FOV_COL_RE = re.compile(r"^\s*fov\s*$", re.I)


def _fov_column(fieldnames):
    """The explicit ``fov`` column if this coordinates.csv has one, else ``None``.

    The header, never row counts, discriminates the two real schemas: with a ``fov``
    column the id is stated; without one it is row order within the region.
    """
    return next((n for n in list(fieldnames or []) if n and _FOV_COL_RE.match(n)), None)


_MM_TO_UM = 1000.0


def _parse_mm_pair(raw_x: str, raw_y: str, region: str, line_no: int):
    """``(x_mm, y_mm)`` floats, or ``None`` for a blank (position-less) row. Raises on garbage."""
    if not raw_x or not raw_y:
        return None
    try:
        return float(raw_x), float(raw_y)
    except ValueError:
        raise ValueError(
            f"{_COORDS_NAME} line {line_no}: region {region!r} has non-numeric "
            f"coordinates ({raw_x!r}, {raw_y!r}); refusing to guess a stage position."
        ) from None


def _positions_from_fov_column(reader, fovs_per_region: dict, fov_col, x_col, y_col) -> dict:
    """Parse the schema whose rows state their FOV id; repeats collapse, conflicts and set mismatches raise."""
    by_region: dict[str, dict[int, tuple]] = {}
    for line_no, row in enumerate(reader, start=2):
        region = (row.get("region") or "").strip()
        if not region or region not in fovs_per_region:
            continue
        pair = _parse_mm_pair(
            (row.get(x_col) or "").strip(), (row.get(y_col) or "").strip(), region, line_no
        )
        if pair is None:
            continue
        raw_fov = (row.get(fov_col) or "").strip()
        try:
            fov = int(raw_fov)
        except ValueError:
            raise ValueError(
                f"{_COORDS_NAME} line {line_no}: region {region!r} has a non-integer fov id "
                f"({raw_fov!r}); the fov column is the row -> image mapping and cannot be guessed."
            ) from None
        x, y = pair
        key = (round(x, 6), round(y, 6))    # tolerate float-repr drift
        seen = by_region.setdefault(region, {})
        if fov in seen:
            if seen[fov][0] != key:
                raise ValueError(
                    f"{_COORDS_NAME} line {line_no}: region {region!r} fov {fov} appears at two "
                    f"conflicting stage positions ({seen[fov][0]} and {key} mm). A repeated fov id "
                    "is normal (one row per z-level) only when the position is identical; differing "
                    "positions mean the file is corrupt or concatenated — refusing to pick one."
                )
            continue                        # same position repeated (one row per z / per t)
        seen[fov] = (key, (x * _MM_TO_UM, y * _MM_TO_UM))

    positions: dict = {}
    for region, seen in by_region.items():
        expected = set(fovs_per_region[region])
        if set(seen) != expected:
            missing = sorted(expected - set(seen))
            extra = sorted(set(seen) - expected)
            raise ValueError(
                f"{_COORDS_NAME}: region {region!r} lists {len(seen)} distinct stage position(s) "
                f"for fov ids that do not match the {len(expected)} FOV(s) found in the filenames "
                f"(missing from the CSV: {missing}; in the CSV but not on disk: {extra}). "
                "Refusing to place a partially-known plate at positions that would look plausible "
                "but be wrong."
            )
        for fov, (_key, xy) in seen.items():
            positions[(region, fov)] = xy
    return positions


def _warn_recorded_mismatch(acq: dict, *, n_z: int, z_source: str,
                            n_t=None) -> None:
    """The recorded-vs-observed cross-check, ONCE: what is on disk is ground truth.

    Three adapters carried private copies of this warning; the wording drifted while the rule
    did not. ``n_t=None`` skips the timepoint check for stores that have no folder axis.
    """
    if acq.get("n_z_declared") is not None and acq["n_z_declared"] != n_z:
        warnings.warn(
            f"Recorded Nz ({acq['n_z_declared']}) != {z_source} ({n_z}); "
            "using the observed value."
        )
    if n_t is not None and acq.get("n_t_declared") is not None and acq["n_t_declared"] != n_t:
        warnings.warn(
            f"Recorded Nt ({acq['n_t_declared']}) != timepoint folders found ({n_t}); "
            "using the folder-derived value."
        )


def _assemble_metadata(*, regions, fovs_per_region, fov_positions_um, channels, n_z, z_levels,
                       dz_um, pixel_size_um, wellplate_format, frame_shape, dtype, n_t):
    """THE one ``Acquisition`` assembly all four adapters share.

    The 13-key block used to be copied four times; a key added to three of them and forgotten
    in the fourth would have shipped as a per-format metadata hole. Keyword-only so every
    adapter states every key.
    """
    return Acquisition(**{
        "regions": regions,
        "fovs_per_region": fovs_per_region,
        "fov_positions_um": fov_positions_um,
        "channels": channels,
        "n_z": n_z,
        "z_levels": z_levels,
        "dz_um": dz_um,
        "pixel_size_um": pixel_size_um,
        "wellplate_format": wellplate_format,
        "frame_shape": frame_shape,
        "dtype": dtype,
        "n_t": n_t,
    })


def _parse_fov_positions_um(root, fovs_per_region: dict) -> tuple:
    """Parse ``coordinates.csv`` into ``({(region, fov): (x_um, y_um)}, mismatched)``, in micrometres.

    Positions are de-duplicated per region, then cross-checked against the filename-derived
    FOV count; regions that fail land in ``mismatched`` instead of the positions dict.
    """
    path, source = _coords_path(root)
    if path is None:
        return {}, {}
    if source == COORDS_PLANNED:
        warnings.warn(
            f"{_COORDS_NAME}: using PLANNED positions. The per-timepoint EXECUTED file "
            f"(0/{_COORDS_NAME}) is absent, so FOVs are placed where the run intended to go, not "
            "where the stage actually went: seams will be off by whatever correction autofocus and "
            "backlash applied. Every real Squid acquisition writes the executed file, so a dataset "
            "without one is usually hand-built."
        )

    return parse_coordinates_csv(path.read_text(), fovs_per_region)


def parse_coordinates_csv(text: str, fovs_per_region: dict) -> tuple:
    """PURE half of the coordinates parse: CSV text in, positions out. No filesystem.

    ``({(region, fov): (x_um, y_um)}, mismatched)`` — exactly
    :func:`_parse_fov_positions_um`'s contract, minus finding the file. Extracted so the
    string→dict transform is testable as strings (the suite used to write 35 real
    ``coordinates.csv`` files to disk to test it).
    """
    import csv
    import io

    reader = csv.DictReader(io.StringIO(text))
    x_col, y_col = _coord_columns(reader.fieldnames)
    fov_col = _fov_column(reader.fieldnames)
    if fov_col is not None:
        return _positions_from_fov_column(reader, fovs_per_region, fov_col, x_col, y_col), {}
    ordered: dict[str, list] = {}
    seen: dict[str, set] = {}
    for line_no, row in enumerate(reader, start=2):
        region = (row.get("region") or "").strip()
        if not region or region not in fovs_per_region:
            continue
        pair = _parse_mm_pair(
            (row.get(x_col) or "").strip(), (row.get(y_col) or "").strip(), region, line_no
        )
        if pair is None:
            continue
        x, y = pair
        key = (round(x, 6), round(y, 6))   # tolerate float-repr drift when de-duplicating
        if key in seen.setdefault(region, set()):
            continue                    # same position repeated (one row per z / per t)
        seen[region].add(key)
        ordered.setdefault(region, []).append((x * _MM_TO_UM, y * _MM_TO_UM))

    positions: dict = {}
    mismatched: dict = {}
    for region, coords in ordered.items():
        fovs = list(fovs_per_region[region])
        if len(coords) != len(fovs):
            # Per-region cross-check: record and continue, callers decide strictness.
            mismatched[region] = (len(coords), len(fovs))
            continue
        for fov, xy in zip(fovs, coords):
            positions[(region, fov)] = xy
    return positions, mismatched


def _mismatch_message(mismatched: dict) -> str:
    """The refusal text for regions whose position count disagrees with their FOV count."""
    parts = ", ".join(
        f"region {region!r} lists {n_pos} distinct stage position(s) but {n_fov} FOV(s) "
        "were found in the filenames"
        for region, (n_pos, n_fov) in sorted(mismatched.items())
    )
    return (
        f"{_COORDS_NAME}: {parts}. "
        "Without a 'fov' column the Nth position must be the Nth FOV, so a count "
        "mismatch means the mapping is unknowable — refusing to place FOVs at "
        "positions that would look plausible but be wrong."
    )


def load_fov_positions_um(root, fovs_per_region: dict) -> dict:
    """Strict parse: raises ValueError naming every region that fails the cross-check."""
    positions, mismatched = _parse_fov_positions_um(root, fovs_per_region)
    if mismatched:
        raise ValueError(_mismatch_message(mismatched))
    return positions


def _fov_positions_um_or_empty(root, fovs_per_region: dict) -> dict:
    """``load_fov_positions_um`` degraded per region: failed regions warn and lose placement only."""
    try:
        positions, mismatched = _parse_fov_positions_um(root, fovs_per_region)
    except ValueError as e:
        warnings.warn(
            f"{_COORDS_NAME} is unusable ({e}) — continuing WITHOUT stage positions: the "
            "acquisition still opens, but multi-FOV wells render as a single tile instead of "
            "a coordinate-placed mosaic."
        )
        return {}

    if mismatched:
        kept = sorted({region for region, _ in positions})
        warnings.warn(
            f"{_COORDS_NAME} is unusable for {len(mismatched)} of "
            f"{len(mismatched) + len(kept)} region(s) ({_mismatch_message(mismatched)}) — "
            f"those regions render as a single tile instead of a coordinate-placed mosaic. "
            f"Kept stage positions for: {', '.join(kept) if kept else '(none)'}."
        )
    return positions


def _plate_key(region: str):
    """Sort key for true plate row-major order (A..Z before AA..; columns by integer)."""
    m = re.match(r"^([A-Za-z]+)(\d+)$", region)
    if not m:
        return (1, len(region), region, 0)          # non-plate ids: stable, after the wells
    return (0, len(m.group(1)), m.group(1).upper(), int(m.group(2)))


def _time_folders(path: Path) -> list[Path]:
    """Squid's timepoint folders (``0/``, ``1/``, …) under *path*, else *path* itself; sorted numerically."""
    numeric = [d for d in path.iterdir() if d.is_dir() and d.name.isdigit()]
    return sorted(numeric, key=lambda d: int(d.name)) if numeric else [path]


#: How many unrecognised filenames an error message may name.
_NAMES_IN_ERRORS = 8


def _note_odd_name(names: list, name: str) -> None:
    """Keep the ``_NAMES_IN_ERRORS`` lexicographically smallest *name*s seen, in order."""
    if len(names) >= _NAMES_IN_ERRORS:
        if name >= names[-1]:
            return
        names.pop()
    bisect.insort(names, name)


def _classify_tiff_folder(folder: Path, entries: "Optional[list]" = None
                          ) -> tuple[int, int, list, list]:
    """``(n_individual, n_stack, other_names, entries)`` for one timepoint folder.

    The listing is unsorted and materialised once, then handed back so the chosen
    reader does not scan the same directory again.
    """
    if entries is None:
        entries = list(folder.iterdir())
    individual = stacks = 0
    other: list = []
    for f in entries:
        if not f.is_file() or f.suffix.lower() not in _TIFF_SUFFIXES:
            continue
        if _STEM_RE.match(f.stem):
            individual += 1
        elif _STACK_STEM_RE.match(f.stem):
            stacks += 1
        else:
            _note_odd_name(other, f.name)
    return individual, stacks, other, entries


def open_reader(path) -> SquidAcquisitionReader:
    """Detect the acquisition format at *path* and return a reader; unrecognised layouts raise, naming both sides."""
    path = Path(path)
    if not path.is_dir():
        raise NotImplementedError(
            f"{path!s} is not a directory. Point open_reader at a Squid acquisition folder."
        )
    ome = path / "ome_tiff"
    # Squid often leaves an empty ome_tiff/ placeholder; it must not shadow the TIFF readers.
    if ome.is_dir() and any(ome.rglob("*.ome.tif*")):
        return SquidOMEReader(path)
    store = _find_zarr_store(path)
    if store is not None:
        return SquidZarrReader(store, acquisition_root=path)

    folder = _time_folders(path)[0]
    individual, stacks, other, entries = _classify_tiff_folder(folder)
    if individual:
        if stacks:
            warnings.warn(
                f"{folder!s} contains BOTH {individual} individual-TIFF plane(s) "
                f"({{region}}_{{fov}}_{{z}}_{{channel}}.tiff) and {stacks} multi-page stack(s) "
                "({region}_{fov}_stack.tiff). Squid writes one or the other per acquisition, so "
                "this folder holds two runs. Reading the individual TIFFs and IGNORING the "
                "stacks — split them into separate folders to read the stacks."
            )
        # Seed the reader with the listing already paid for above.
        return SquidReader(path, _scanned=(folder, entries))
    if stacks:
        return SquidMultiPageTiffReader(path)
    raise ValueError(
        f"{path!s} is not a readable Squid acquisition: {folder!s} contains no "
        "{region}_{fov}_{z}_{channel}.tiff (individual TIFF writer) and no "
        "{region}_{fov}_stack.tiff (MULTI_PAGE_TIFF writer), there is no ome_tiff/ folder with "
        ".ome.tiff files (SaveOMETiffJob) and no plate.ome.zarr/ or zarr/ store (SaveZarrJob). "
        + (f"Non-matching TIFF files present: {other}. " if other else "")
        + "Point open_reader at the acquisition folder itself, not a parent or a subfolder."
    )


# Squid non-HCS per-FOV layout: zarr/{region}/fov_{n}.ome.zarr.
_PER_FOV_ZARR_RE = re.compile(r"^fov_(?P<fov>\d+)\.ome\.zarr$")

# Non-standard 6D layout: a zarr ARRAY (not group), so it is recognised by name.
_SIXD_ZARR_NAME = "acquisition.zarr"


def _nonhcs_region_children(region_dir: Path) -> tuple[list, Optional[Path]]:
    """``(per_fov_groups, sixd_array_or_None)`` inside one ``zarr/{region_id}/`` directory."""
    if not region_dir.is_dir():
        return [], None
    per_fov = sorted(
        (c for c in region_dir.iterdir() if c.is_dir() and _PER_FOV_ZARR_RE.match(c.name)),
        key=lambda c: int(_PER_FOV_ZARR_RE.match(c.name)["fov"]),
    )
    sixd = region_dir / _SIXD_ZARR_NAME
    return per_fov, (sixd if sixd.is_dir() else None)


def _find_zarr_store(path: Path):
    """The Zarr root to read at/under *path*, or ``None`` if this is not a Zarr acquisition.

    An unrecognisable ``zarr/`` folder raises rather than falling through to the TIFF readers.
    """
    if _is_zarr_group(path):
        return path
    plate = path / "plate.ome.zarr"
    if _is_zarr_group(plate):
        return plate
    for candidate in sorted(path.glob("*.zarr")):
        if _is_zarr_group(candidate):
            return candidate
    bare = path / "zarr"
    if not bare.is_dir():
        return None
    subdirs = [d for d in sorted(bare.iterdir()) if d.is_dir()]
    if not subdirs:
        return None
    for d in subdirs:
        per_fov, sixd = _nonhcs_region_children(d)
        if _is_zarr_group(d) or per_fov or sixd is not None:
            return bare
    raise ValueError(
        f"{bare!s} looks like Squid's non-HCS Zarr output but no readable store was found in it. "
        f"Region folders present: {[d.name for d in subdirs[:8]]}. Expected one of: "
        "zarr/{region}/fov_{n}.ome.zarr (SaveZarrJob non-HCS default), "
        "zarr/{region}/acquisition.zarr (SaveZarrJob non-HCS 6D), or zarr/{region}/ itself being "
        "an OME-NGFF image group. Refusing rather than reporting an empty acquisition."
    )


class SquidReader:
    """Lazy reader over a Squid individual-TIFF acquisition folder."""

    def __init__(self, path, *, _scanned: "Optional[tuple[Path, list]]" = None) -> None:
        self._path = Path(path)
        self._time_folders: Optional[list[Path]] = None
        self._index: Optional[dict] = None
        self._meta: Optional[dict] = None
        #: ``(folder, entries)`` from a listing the caller already paid for; consumed once.
        self._scanned = _scanned

    # -- timepoints -------------------------------------------------------

    @property
    def source_id(self) -> str:
        """The acquisition folder this reader reads (the contract's identity member)."""
        return str(self._path)

    def _discover_time_folders(self) -> list[Path]:
        if self._time_folders is None:
            self._time_folders = _time_folders(self._path)
        return self._time_folders

    # -- index ------------------------------------------------------------
    def _build_index(self) -> dict:
        """Map {(region, fov, z, channel): file_suffix} from the first timepoint folder."""
        if self._index is not None:
            return self._index
        folder = self._discover_time_folders()[0]
        entries = None
        if self._scanned is not None:
            scanned_folder, scanned_entries = self._scanned
            self._scanned = None                 # one use only; see __init__
            if scanned_folder == folder:
                entries = scanned_entries
        if entries is None:
            entries = folder.iterdir()
        index: dict = {}
        skipped: list = []
        for f in entries:
            if f.suffix.lower() not in _TIFF_SUFFIXES:
                continue
            m = _STEM_RE.match(f.stem)
            if m:
                key = (m["region"], int(m["fov"]), int(m["z"]), m["channel"])
                index[key] = f.suffix
                continue
            # A stem matching a known Squid pattern raises; anything else is remembered for the error.
            if _STACK_STEM_RE.match(f.stem):
                raise ValueError(
                    f"{f.name} is Squid's MULTI_PAGE_TIFF layout "
                    "({region}_{fov:0FILE_ID_PADDING}_stack.tiff, written by SaveImageJob when "
                    "_def.FILE_SAVING_OPTION == FileSavingOption.MULTI_PAGE_TIFF), not the "
                    "individual-TIFF layout ({region}_{fov}_{z}_{channel}.tiff) this reader "
                    "serves. Use squidxplorer.open_reader(), which dispatches to "
                    "SquidMultiPageTiffReader for this format."
                )
            _note_odd_name(skipped, f.name)
        if not index:
            raise ValueError(
                "No Squid individual-TIFF files "
                "({region}_{fov}_{z}_{channel}.tiff) found in "
                f"{folder!s}" + (f"; TIFF files present but unrecognised: {skipped}" if skipped
                                 else "")
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
        regions = sorted(fovs, key=_plate_key)   # true plate row-major (A,B,...,Z,AA,...)

        z_sorted = sorted(z_levels)
        n_z = len(z_sorted)
        n_t = len(time_folders)

        # Filenames + timepoint folders are ground truth; the recorded Nz/Nt are cross-checks.
        acq = load_acquisition_metadata(self._path)
        _warn_recorded_mismatch(acq, n_z=n_z, z_source="distinct z levels in filenames", n_t=n_t)

        # Frame shape/dtype come from a real frame; ``min`` keeps the sampled plane reproducible.
        sample_key = min(index)
        sample_path = self._resolve_file(time_folders[0], sample_key, index[sample_key])
        sample = _validate_plane(tifffile.imread(sample_path), sample_path)

        fovs_per_region = {r: sorted(fovs[r]) for r in regions}
        self._meta = _assemble_metadata(
            regions=regions,
            fovs_per_region=fovs_per_region,
            fov_positions_um=_fov_positions_um_or_empty(self._path, fovs_per_region),
            channels=resolve_channels(sorted(channels), load_channel_yaml(self._path)),
            n_z=n_z,
            z_levels=z_sorted,
            dz_um=acq["dz_um"],
            pixel_size_um=acq["pixel_size_um"],
            wellplate_format=acq["wellplate_format"],
            frame_shape=tuple(sample.shape),
            dtype=sample.dtype,
            n_t=n_t,
        )
        return self._meta

    # -- read -------------------------------------------------------------
    def read(self, region, fov, channel, z_level, time_point=0):
        """Return one plane as a 2D array in its native dtype. Lazy: reads exactly one file."""
        index = self._build_index()
        time_folders = self._discover_time_folders()
        key = (str(region), int(fov), int(z_level), str(channel))
        if key not in index:
            raise KeyError(
                f"No such plane region={region!r} fov={fov} channel={channel!r} z={z_level}. "
                f"Known regions={sorted({k[0] for k in index})}, "
                f"channels={sorted({k[3] for k in index})}."
            )
        time_point = int(time_point)
        if not 0 <= time_point < len(time_folders):
            raise IndexError(f"t={time_point} out of range (n_t={len(time_folders)}).")
        path = self._resolve_file(time_folders[time_point], key, index[key])
        return _validate_plane(tifffile.imread(path), path)

    def plane_path(self, region, fov, channel, z_level, time_point=0) -> Path:
        """Path to one raw plane's TIFF on disk (no decode)."""
        index = self._build_index()
        time_folders = self._discover_time_folders()
        key = (str(region), int(fov), int(z_level), str(channel))
        if key not in index:
            raise KeyError(f"No such plane region={region!r} fov={fov} channel={channel!r} z={z_level}.")
        time_point = int(time_point)
        if not 0 <= time_point < len(time_folders):
            raise IndexError(f"t={time_point} out of range (n_t={len(time_folders)}).")
        return self._resolve_file(time_folders[time_point], key, index[key])

    def plane_ref(self, region, fov, channel, z_level, time_point=0) -> tuple:
        """(filepath, page_index) for one plane; individual TIFFs are one plane per file, so page 0."""
        return str(self.plane_path(region, fov, channel, z_level, time_point)), 0

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


def _page_json(page, path: Path, page_index: int) -> dict:
    """The per-page metadata dict Squid embeds in ImageDescription, or a loud failure.

    A page can carry two 270 tags; the first that parses as JSON with ``z_level`` wins.
    """
    for tag in page.tags:
        if tag.code != _TAG_IMAGE_DESCRIPTION:
            continue
        try:
            payload = json.loads(tag.value)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and "z_level" in payload:
            return payload
    raise ValueError(
        f"{path.name} page {page_index} carries no Squid metadata: no ImageDescription (TIFF tag "
        f"{_TAG_IMAGE_DESCRIPTION}) holding JSON with a 'z_level' key. Squid's MULTI_PAGE_TIFF "
        "writer records z_level/channel/region_id/fov/x_mm/y_mm/z_mm/time on every page; without "
        "it the page's place in the (z, channel) grid is unknowable. Refusing to guess from page "
        "order — a guessed order silently mis-assigns channels."
    )


def _page_channel(page, payload: dict, path: Path, page_index: int) -> str:
    """The page's channel name: PageName (tag 285) first, the JSON ``channel`` key as fallback; disagreement raises."""
    tag = page.tags.get(_TAG_PAGE_NAME)
    from_tag = str(tag.value).strip() if tag is not None and tag.value else ""
    from_json = str(payload.get("channel") or "").strip()
    if from_tag and from_json and from_tag != from_json:
        raise ValueError(
            f"{path.name} page {page_index} disagrees with itself about the channel: PageName "
            f"(tag {_TAG_PAGE_NAME}) says {from_tag!r} but ImageDescription JSON says "
            f"{from_json!r}. Refusing to pick one — a mislabelled channel is invisible downstream."
        )
    name = from_tag or from_json
    if not name:
        raise ValueError(
            f"{path.name} page {page_index} names no channel: neither PageName (tag "
            f"{_TAG_PAGE_NAME}) nor a 'channel' key in the ImageDescription JSON. Squid's "
            "MULTI_PAGE_TIFF writer sets both."
        )
    return name


class SquidMultiPageTiffReader:
    """Lazy reader over Squid's MULTI_PAGE_TIFF acquisitions (one stack file per field).

    Page order is not a usable index; the (z, channel) grid is read from each page's own metadata.
    """

    def __init__(self, path) -> None:
        self._path = Path(path)
        self._time_folders_cache: Optional[list[Path]] = None
        self._indexes: dict[int, dict] = {}     # t -> {(region, fov, z, channel): (Path, page)}
        self._positions_mm: dict = {}           # {(region, fov): (x_mm, y_mm)} from t=0
        self._meta: Optional[dict] = None
        self._handles = _TiffHandles()

    # -- discovery ---------------------------------------------------------

    @property
    def source_id(self) -> str:
        """The acquisition folder this reader reads (the contract's identity member)."""
        return str(self._path)

    def _discover_time_folders(self) -> list[Path]:
        if self._time_folders_cache is None:
            self._time_folders_cache = _time_folders(self._path)
        return self._time_folders_cache


    def _index_for(self, time_point: int) -> dict:
        """``{(region, fov, z, channel): (path, page_index)}`` for one timepoint folder."""
        if time_point in self._indexes:
            return self._indexes[time_point]
        folder = self._discover_time_folders()[time_point]
        index: dict = {}
        stacks = [f for f in sorted(folder.iterdir())
                  if f.suffix.lower() in _TIFF_SUFFIXES and _STACK_STEM_RE.match(f.stem)]
        for f in stacks:
            m = _STACK_STEM_RE.match(f.stem)
            region, fov = m["region"], int(m["fov"])
            with self._handles.read(f) as tif:
                pages = list(enumerate(tif.pages))
            for page_index, page in pages:
                payload = _page_json(page, f, page_index)
                channel = _page_channel(page, payload, f, page_index)
                z = int(payload["z_level"])
                key = (region, fov, z, channel)
                if key in index:
                    raise ValueError(
                        f"{f.name} has two pages claiming z={z} channel={channel!r} (pages "
                        f"{index[key][1]} and {page_index}). One of them would be unreachable; "
                        "refusing rather than serving whichever happened to be indexed last."
                    )
                index[key] = (f, page_index)
                if time_point == 0:
                    self._record_position(region, fov, payload)
        if not index:
            raise ValueError(
                "No Squid MULTI_PAGE_TIFF stacks ({region}_{fov}_stack.tiff) found in "
                f"{folder!s}"
            )
        self._indexes[time_point] = index
        return index

    def _record_position(self, region: str, fov: int, payload: dict) -> None:
        """First page wins for a field's stage position; per-z stage jitter is not a conflict."""
        key = (str(region), int(fov))
        if key in self._positions_mm:
            return
        try:
            x_mm, y_mm = float(payload["x_mm"]), float(payload["y_mm"])
        except (KeyError, TypeError, ValueError):
            return
        self._positions_mm[key] = (x_mm, y_mm)

    # -- metadata ----------------------------------------------------------
    @property
    def metadata(self) -> dict:
        if self._meta is not None:
            return self._meta
        index = self._index_for(0)
        time_folders = self._discover_time_folders()

        fovs: dict[str, set] = {}
        channels: set = set()
        z_levels: set = set()
        for (region, fov, z, channel) in index:
            fovs.setdefault(region, set()).add(fov)
            channels.add(channel)
            z_levels.add(z)
        regions = sorted(fovs, key=_plate_key)
        fovs_per_region = {r: sorted(fovs[r]) for r in regions}
        z_sorted = sorted(z_levels)
        n_z, n_t = len(z_sorted), len(time_folders)

        acq = load_acquisition_metadata(self._path)
        _warn_recorded_mismatch(acq, n_z=n_z, z_source="distinct z levels in the stack pages",
                                n_t=n_t)

        s_region, s_fov, s_z, s_channel = next(iter(index))
        sample = self.read(s_region, s_fov, s_channel, s_z)
        self._meta = _assemble_metadata(
            regions=regions,
            fovs_per_region=fovs_per_region,
            fov_positions_um=self._positions_um(fovs_per_region),
            channels=resolve_channels(sorted(channels), load_channel_yaml(self._path)),
            n_z=n_z,
            z_levels=z_sorted,
            dz_um=acq["dz_um"],
            pixel_size_um=acq["pixel_size_um"],
            wellplate_format=acq["wellplate_format"],
            frame_shape=tuple(sample.shape),
            dtype=sample.dtype,
            n_t=n_t,
        )
        return self._meta

    def _positions_um(self, fovs_per_region: dict) -> dict:
        """``{(region, fov): (x_um, y_um)}``: per-page inline positions first, coordinates.csv as fallback."""
        self._index_for(0)                       # populates _positions_mm
        if self._positions_mm:
            return {k: (x * _MM_TO_UM, y * _MM_TO_UM)
                    for k, (x, y) in self._positions_mm.items()}
        return _fov_positions_um_or_empty(self._path, fovs_per_region)

    # -- read --------------------------------------------------------------
    def _locate(self, region, fov, channel, z_level, time_point) -> tuple:
        time_folders = self._discover_time_folders()
        time_point = int(time_point)
        if not 0 <= time_point < len(time_folders):
            raise IndexError(f"t={time_point} out of range (n_t={len(time_folders)}).")
        index = self._index_for(time_point)
        key = (str(region), int(fov), int(z_level), str(channel))
        if key not in index:
            raise KeyError(
                f"No such plane region={region!r} fov={fov} channel={channel!r} z={z_level}. "
                f"Known regions={sorted({k[0] for k in index})}, "
                f"channels={sorted({k[3] for k in index})}."
            )
        return index[key]

    def read(self, region, fov, channel, z_level, time_point=0):
        """Return one plane as a 2D array in its native dtype (decodes exactly one IFD page)."""
        path, page_index = self._locate(region, fov, channel, z_level, time_point)
        return self._handles.page(path, page_index)

    def plane_path(self, region, fov, channel, z_level, time_point=0) -> Path:
        """The stack file holding this plane (the whole field's pages)."""
        return self._locate(region, fov, channel, z_level, time_point)[0]

    def plane_ref(self, region, fov, channel, z_level, time_point=0) -> tuple:
        """``(filepath, page_index)`` for one plane."""
        path, page_index = self._locate(region, fov, channel, z_level, time_point)
        return str(path), page_index


# {region}_{fov} stem (region = well id, no trailing _<digits>; fov = trailing integer).
_OME_STEM_RE = re.compile(r"^(?P<region>.+)_(?P<fov>\d+)$")
_OME_SUFFIXES = (".ome.tiff", ".ome.tif", ".OME.TIFF", ".OME.TIF")


class SquidOMEReader:
    """Lazy reader over a Squid OME-TIFF acquisition: one 5-D TZCYX stack per well-FOV."""

    def __init__(self, path) -> None:
        self._path = Path(path)
        self._ome = self._path / "ome_tiff"
        self._files: Optional[dict] = None      # {(region, fov): Path}
        self._meta: Optional[dict] = None
        self._axes: Optional[str] = None        # non-spatial axes order, e.g. "TZC"
        self._handles = _TiffHandles()


    @property
    def source_id(self) -> str:
        """The acquisition folder this reader reads (the contract's identity member)."""
        return str(self._path)

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


    @property
    def metadata(self) -> dict:
        if self._meta is not None:
            return self._meta
        files = self._discover()
        with self._handles.read(next(iter(files.values()))) as _tif:
            sample = _tif.series[0]
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
            # yaml disagrees with the file: fall back to OME channel names, else generic labels.
            with self._handles.read(next(iter(files.values()))) as _tif:
                ome_names = _ome_channel_names(_tif)
            names = [_normalize_local(n) for n in ome_names] if len(ome_names) == n_c \
                else [f"C{i}" for i in range(n_c)]
        channels = resolve_channels(names, yaml_map)

        acq = load_acquisition_metadata(self._path)
        _warn_recorded_mismatch(acq, n_z=n_z, z_source="OME Z")
        fovs_per_region = {r: sorted(fovs[r]) for r in regions}
        self._meta = _assemble_metadata(
            regions=regions,
            fovs_per_region=fovs_per_region,
            fov_positions_um=_fov_positions_um_or_empty(self._path, fovs_per_region),
            channels=channels,
            n_z=n_z,
            z_levels=list(range(n_z)),
            dz_um=acq["dz_um"],
            pixel_size_um=acq["pixel_size_um"],
            wellplate_format=acq["wellplate_format"],
            frame_shape=(int(dims.get("Y", sample.shape[-2])), int(dims.get("X", sample.shape[-1]))),
            dtype=np.dtype(sample.dtype),
            n_t=n_t,
        )
        return self._meta

    def _page_index(self, time_point: int, z_level: int, c: int) -> int:
        """Flat IFD page index for (t, z, c), honouring the file's non-spatial axis order."""
        meta = self.metadata
        sizes = {"T": meta["n_t"], "Z": meta["n_z"], "C": len(meta["channels"])}
        pos = {"T": time_point, "Z": z_level, "C": c}
        order = self._axes or "TZC"
        return int(np.ravel_multi_index([pos[a] for a in order], [sizes[a] for a in order]))

    def _channel_index(self, channel) -> int:
        names = [c["name"] for c in self.metadata["channels"]]
        return names.index(str(channel))

    def read(self, region, fov, channel, z_level, time_point=0):
        """Return one plane as a 2D native-dtype array (reads exactly one IFD page)."""
        files = self._discover()
        key = (str(region), int(fov))
        if key not in files:
            raise KeyError(f"No such well/FOV region={region!r} fov={fov}. Known: {sorted(files)[:8]}")
        p = self._page_index(int(time_point), int(z_level), self._channel_index(channel))
        return self._handles.page(files[key], p)

    def plane_ref(self, region, fov, channel, z_level, time_point=0) -> tuple:
        """(filepath, page_index) for one plane."""
        p = self._page_index(int(time_point), int(z_level), self._channel_index(channel))
        return str(self._discover()[(str(region), int(fov))]), p


def _normalize_local(name: str) -> str:
    from squidxplorer._channels import normalize
    return normalize(name)


def _ome_channel_names(tif) -> list:
    """Best-effort channel names from the OME-XML (Channel Name=...), else []."""
    try:
        xml = tif.ome_metadata or ""
        return re.findall(r'<Channel[^>]*\bName="([^"]*)"', xml)
    except Exception:
        return []


# OME-NGFF Zarr input. The layout contract lives in docs/plate-contract.md.

_ZARR_V3_META = "zarr.json"
_ZARR_V2_GROUP = ".zgroup"
_ZARR_V2_ATTRS = ".zattrs"
_ZARR_V2_ARRAY = ".zarray"

# UDUNITS-2 length units -> micrometres; absent/unknown units are treated as micrometres.
_UNIT_TO_UM = {
    "angstrom": 1e-4, "nanometer": 1e-3, "micrometer": 1.0, "micron": 1.0,
    "millimeter": 1e3, "centimeter": 1e4, "meter": 1e6,
}


def _is_zarr_group(path: Path) -> bool:
    """True if *path* is a zarr GROUP node — v3 (``zarr.json``) or v2 (``.zgroup``)."""
    path = Path(path)
    if not path.is_dir():
        return False
    if (path / _ZARR_V2_GROUP).exists():
        return True
    meta = path / _ZARR_V3_META
    if not meta.exists():
        return False
    try:
        return json.loads(meta.read_text()).get("node_type") == "group"
    except (ValueError, OSError):
        return False


def _group_attrs(path: Path) -> dict:
    """The OME metadata payload of a zarr group — ``contract.store.ome_attrs``, THE one walk."""
    from squidxplorer.contract.store import ome_attrs

    return ome_attrs(path)


def _open_zarr_array(path: Path):
    """Open one zarr array (v2 or v3) as a lazy tensorstore handle via the process-wide pool."""
    from squidxplorer._tsctx import HANDLES

    path = Path(path)
    driver = "zarr" if (path / _ZARR_V2_ARRAY).exists() else "zarr3"
    return HANDLES.get(path, driver=driver, open_only=True)


def _unit_to_um(unit) -> float:
    """Scale factor from an axis's declared unit to micrometres (1.0 when unit is absent)."""
    if not unit:
        return 1.0
    factor = _UNIT_TO_UM.get(str(unit).strip().lower().rstrip("s"))
    if factor is None:
        warnings.warn(
            f"OME-NGFF space axis unit {unit!r} is not a length this reader converts; treating "
            "the value as micrometres. Physical placement may be wrong — check the store."
        )
        return 1.0
    return factor


class _Multiscale:
    """The parsed ``multiscales[0]`` of one image group; everything physical is in micrometres."""

    def __init__(self, group: Path) -> None:
        attrs = _group_attrs(group)
        multiscales = attrs.get("multiscales") or []
        if not multiscales:
            raise ValueError(
                f"{group!s} is not an OME-NGFF image group: no 'multiscales' metadata. "
                "Expected a field/image group written by Squid or any NGFF writer."
            )
        ms = multiscales[0]
        self.group = group
        self.omero = attrs.get("omero") or {}
        axes = ms.get("axes") or []
        # The spec fixes axis order but not which axes are present; 2-D/4-D stores are legal.
        self.axis_names = [str(a.get("name", "")).lower() for a in axes]
        self.units = {n: a.get("unit") for n, a in zip(self.axis_names, axes)}

        datasets = ms.get("datasets") or []
        if not datasets:
            raise ValueError(f"{group!s}: multiscales has no 'datasets' (no resolution levels).")
        level0 = datasets[0]        # datasets are ordered highest -> lowest resolution
        self.array_path = group / str(level0["path"])

        transforms = level0.get("coordinateTransformations") or []
        scale = next((t.get("scale") for t in transforms if t.get("type") == "scale"), None)
        translation = next(
            (t.get("translation") for t in transforms if t.get("type") == "translation"), None
        )
        self._scale = list(scale) if scale else [1.0] * len(self.axis_names)
        self._translation = list(translation) if translation else None

    def _axis(self, name: str) -> Optional[int]:
        return self.axis_names.index(name) if name in self.axis_names else None

    def _physical(self, values, name: str) -> Optional[float]:
        i = self._axis(name)
        if i is None or values is None or i >= len(values):
            return None
        return float(values[i]) * _unit_to_um(self.units.get(name))

    @property
    def pixel_size_um(self) -> Optional[float]:
        return self._physical(self._scale, "x")

    @property
    def dz_um(self) -> Optional[float]:
        return self._physical(self._scale, "z")

    @property
    def position_um(self) -> Optional[tuple]:
        """``(x_um, y_um)`` from the dataset ``translation``, or ``None`` when it carries none."""
        if self._translation is None:
            return None
        x, y = self._physical(self._translation, "x"), self._physical(self._translation, "y")
        return None if x is None or y is None else (x, y)

    @property
    def is_6d_fov(self) -> bool:
        """True for Squid's non-standard 6D ``acquisition.zarr`` — a leading ``fov`` axis."""
        return self.axis_names[:1] == ["fov"]

    def index(self, shape, time_point: int, c: int, z_level: int, fov: int = 0) -> tuple:
        """The tensorstore index tuple selecting the single ``(y, x)`` plane at (t, c, z[, fov])."""
        picks = {"t": time_point, "c": c, "z": z_level, "fov": fov}
        return tuple(
            slice(None) if n in ("y", "x") else picks.get(n, 0)
            for n in self.axis_names[: len(shape)]
        )

    def size(self, shape, name: str, default: int = 1) -> int:
        i = self._axis(name)
        return int(shape[i]) if i is not None and i < len(shape) else default


class SquidZarrReader:
    """Lazy reader over an OME-NGFF Zarr acquisition — HCS plate or bare per-region image groups.

    Only resolution level 0 is served; a projection from a downsampled level would be silently wrong.
    """

    def __init__(self, path, acquisition_root=None) -> None:
        self._path = Path(path)
        # Sidecars (acquisition.yaml, coordinates.csv) live beside the store, not inside it.
        self._root = Path(acquisition_root) if acquisition_root is not None else self._path.parent
        self._fields: Optional[dict] = None      # {(region, fov): Path to the image group}
        self._ms: dict = {}                      # image group Path -> _Multiscale (parsed metadata only)
        self._meta: Optional[dict] = None
        self._contract_version = None            # set by _discover: what the store declares, or None


    @property
    def source_id(self) -> str:
        """The acquisition ROOT (where acquisition.yaml / coordinates.csv live), NOT the store.

        The staleness token and every cache key build from this; keyed on the store dir they
        statted sidecars that never exist there and silently degraded.
        """
        return str(self._root)

    # -- discovery ---------------------------------------------------------
    def _discover(self) -> dict:
        if self._fields is not None:
            return self._fields
        # Contract stamp is compared before any path is reconstructed from it; unstamped stores proceed.
        self._contract_version = check_plate_contract(self._path)
        attrs = _group_attrs(self._path)
        fields = (
            self._discover_hcs(attrs["plate"]) if isinstance(attrs.get("plate"), dict)
            else self._discover_flat()
        )
        if not fields:
            raise ValueError(
                f"{self._path!s} contains no readable OME-NGFF images: the plate lists no wells "
                "and no per-region image groups were found."
            )
        self._fields = fields
        return fields

    def _discover_hcs(self, plate: dict) -> dict:
        """``plate.wells[].path`` -> well group -> ``well.images[].path`` -> field image groups."""
        fields: dict = {}
        for well in plate.get("wells") or []:
            rel = str(well.get("path", "")).strip("/")
            if not rel:
                continue
            # Region id is row name + column name, the inverse of _output.parse_well_id.
            region = "".join(rel.split("/"))
            well_dir = self._path / rel
            images = (_group_attrs(well_dir).get("well") or {}).get("images") or []
            for i, image in enumerate(images):
                name = str(image.get("path", ""))
                if not name:
                    continue
                # Non-numeric field paths fall back to their list position for the int FOV.
                fields[(region, int(name) if name.isdigit() else i)] = well_dir / name
        return fields

    def _discover_flat(self) -> dict:
        """Non-HCS: map every region folder under ``zarr/`` to its field image group(s).

        Serves all three real shapes: the region folder as image group, per-FOV
        ``fov_{n}.ome.zarr``, and the 6D ``acquisition.zarr``; anything else raises.
        """
        fields: dict = {}
        for child in sorted(self._path.iterdir()):
            if not child.is_dir():
                continue
            if _is_zarr_group(child) and "multiscales" in _group_attrs(child):
                fields[(child.name, 0)] = child
                continue
            per_fov, sixd = _nonhcs_region_children(child)
            for group in per_fov:
                fov = int(_PER_FOV_ZARR_RE.match(group.name)["fov"])
                fields[(child.name, fov)] = group
            if sixd is not None:
                for fov in range(self._sixd_fov_count(sixd)):
                    fields[(child.name, fov)] = sixd
            if not per_fov and sixd is None:
                raise ValueError(
                    f"{child!s} is under a Squid non-HCS zarr/ folder but is not a readable "
                    "store: it is not an OME-NGFF image group and contains no "
                    "fov_{n}.ome.zarr (non-HCS default) or acquisition.zarr (non-HCS 6D). "
                    f"Contents: {[c.name for c in sorted(child.iterdir())][:8]}."
                )
        # A store that is a single image group (handed in directly) is that one region.
        if not fields and "multiscales" in _group_attrs(self._path):
            fields[(self._path.name.replace(".ome.zarr", "").replace(".zarr", ""), 0)] = self._path
        return fields

    def _sixd_fov_count(self, sixd: Path) -> int:
        """How many FOVs the 6D array's leading axis holds, read from the array's own shape."""
        ms = self._multiscale(sixd)
        if ms.axis_names[:1] != ["fov"]:
            raise ValueError(
                f"{sixd!s} is Squid's non-standard 6D layout (build_6d_zarr_path) but its "
                f"multiscales axes are {ms.axis_names}, not the expected FTCZYX with 'fov' "
                "leading. Refusing to guess which axis is the FOV — guessing draws every field "
                "at the wrong index without erroring."
            )
        return int(self._array(sixd).shape[0])

    def _multiscale(self, group: Path) -> _Multiscale:
        ms = self._ms.get(group)
        if ms is None:
            ms = self._ms[group] = _Multiscale(group)
        return ms

    def _array(self, group: Path):
        """The open store for a group; not memoised here — ``_tsctx.HANDLES`` is the cache."""
        return _open_zarr_array(self._multiscale(group).array_path)

    # -- metadata ----------------------------------------------------------
    @property
    def metadata(self) -> dict:
        if self._meta is not None:
            return self._meta
        fields = self._discover()

        fovs: dict[str, set] = {}
        for (region, fov) in fields:
            fovs.setdefault(region, set()).add(fov)
        regions = sorted(fovs, key=_plate_key)
        fovs_per_region = {r: sorted(fovs[r]) for r in regions}

        sample_group = fields[(regions[0], fovs_per_region[regions[0]][0])]
        ms = self._multiscale(sample_group)
        arr = self._array(sample_group)
        shape, dtype = arr.shape, np.dtype(arr.dtype.numpy_dtype)
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"{ms.array_path!s} has dtype {dtype}; Squid writes uint8 (MONO8) or uint16 "
                "(MONO12/MONO16). An unexpected dtype usually means the store is not a raw Squid "
                "capture; refused rather than silently projected."
            )
        n_z = ms.size(shape, "z")
        n_t = ms.size(shape, "t")

        self._meta = _assemble_metadata(
            regions=regions,
            fovs_per_region=fovs_per_region,
            fov_positions_um=self._positions_um(fields, fovs_per_region),
            channels=self._channels(ms, ms.size(shape, "c")),
            n_z=n_z,
            z_levels=list(range(n_z)),
            dz_um=ms.dz_um,
            pixel_size_um=ms.pixel_size_um,
            wellplate_format=self._wellplate_format(regions),
            frame_shape=(ms.size(shape, "y", shape[-2]), ms.size(shape, "x", shape[-1])),
            dtype=dtype,
            n_t=n_t,
        )
        return self._meta

    def _positions_um(self, fields: dict, fovs_per_region: dict) -> dict:
        """Stage positions in micrometres: dataset ``translation`` first, coordinates.csv second."""
        from_store = {}
        for key, group in fields.items():
            ms = self._multiscale(group)
            if ms.is_6d_fov:
                # One 6D array has one translation for all FOVs, so it cannot place them; use the CSV.
                continue
            position = ms.position_um
            if position is not None:
                from_store[key] = position
        if from_store:
            return from_store
        return _fov_positions_um_or_empty(self._root, fovs_per_region)

    def _channels(self, ms: _Multiscale, n_c: int) -> list:
        """Channel list from ``omero.channels``, falling back to acquisition_channels.yaml, then generic labels."""
        yaml_map = load_channel_yaml(self._root)
        omero_channels = (ms.omero.get("channels") or [])[:n_c]
        if len(omero_channels) == n_c and n_c:
            out = []
            for entry in omero_channels:
                label = str(entry.get("label") or "")
                name = _normalize_local(label) if label else ""
                colour = str(entry.get("color") or "").strip()
                info = yaml_map.get(name)
                out.append({
                    "name": name,
                    "display_name": (info["display_name"] if info else None) or label or name,
                    "display_color": ("#" + colour.lstrip("#")) if colour
                                     else (info["display_color"] if info else None)
                                     or fallback_color(name) or "#FFFFFF",
                    "exposure_time_ms": info["exposure_time_ms"] if info else None,
                    "excitation_nm": excitation_nm(name),
                })
            return out
        names = list(yaml_map.keys())
        if len(names) != n_c:
            warnings.warn(
                f"Zarr store declares C={n_c} but carries no usable omero channel metadata and no "
                "matching acquisition_channels.yaml; falling back to generic channel labels."
            )
            names = [f"C{i}" for i in range(n_c)]
        try:
            return resolve_channels(names, yaml_map)
        except ValueError:
            return [{"name": n, "display_name": n, "display_color": "#FFFFFF",
                     "exposure_time_ms": None, "excitation_nm": excitation_nm(n)}
                    for n in names]

    def _wellplate_format(self, regions: list):
        """Declared (sibling acquisition.yaml) beats inferred."""
        try:
            declared = load_acquisition_metadata(self._root)["wellplate_format"]
        except (FileNotFoundError, ValueError):
            declared = None                     # a Zarr store need not ship acquisition.yaml
        if declared:
            return declared
        from squidxplorer._plate_shape import infer_plate_format

        try:
            return infer_plate_format(regions)
        except Exception:
            return None

    # -- read --------------------------------------------------------------
    def _field(self, region, fov) -> Path:
        fields = self._discover()
        key = (str(region), int(fov))
        if key not in fields:
            raise KeyError(
                f"No such well/FOV region={region!r} fov={fov}. "
                f"Known regions={sorted({k[0] for k in fields})}."
            )
        return fields[key]

    def _channel_index(self, channel) -> int:
        names = [c["name"] for c in self.metadata["channels"]]
        if str(channel) not in names:
            raise KeyError(f"No such channel {channel!r}. Known channels={names}.")
        return names.index(str(channel))

    def read(self, region, fov, channel, z_level, time_point=0):
        """Return one plane as a 2D array in its native dtype; only the covering chunks are read."""
        group = self._field(region, fov)
        meta = self.metadata
        z_level, time_point = int(z_level), int(time_point)
        if not 0 <= z_level < meta["n_z"]:
            raise IndexError(f"z={z_level} out of range (n_z={meta['n_z']}).")
        if not 0 <= time_point < meta["n_t"]:
            raise IndexError(f"t={time_point} out of range (n_t={meta['n_t']}).")
        arr = self._array(group)
        idx = self._multiscale(group).index(
            arr.shape, time_point, self._channel_index(channel), z_level, fov=int(fov)
        )
        plane = np.asarray(arr[idx].read().result())
        return _validate_plane(plane, self._multiscale(group).array_path)

    def plane_ref(self, region, fov, channel, z_level, time_point=0) -> tuple:
        """``(path, 0)`` for one plane, where *path* is the field's NGFF image group."""
        self._channel_index(channel)            # validate like the TIFF readers do
        return str(self._field(region, fov)), 0
