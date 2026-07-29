"""SquidMIP reader: format-aware ingest for EVERY Squid output writer.

``open_reader(path)`` dispatches on the on-disk format and returns a reader. All four readers sit
behind ONE interface — ``metadata`` (identical key set, micrometres, ``_um``-suffixed) plus
``read(region, fov, channel, z, t)`` returning a 2-D native-dtype plane:

    individual TIFFs   :class:`SquidReader`      (IMA-189)   ``<acq>/{t}/{region}_{fov}_{z}_{ch}.tiff``
    multi-page TIFF    :class:`SquidMultiPageTiffReader` (IMA-254) ``<acq>/{t}/{region}_{fov}_stack.tiff``
    OME-TIFF           :class:`SquidOMEReader`               ``<acq>/ome_tiff/{region}_{fov}.ome.tiff``
    OME-NGFF Zarr      :class:`SquidZarrReader`  (IMA-229)   ``<acq>/plate.ome.zarr/…`` or ``<acq>/zarr/…``

The interface IS the seam: engine, CLI and viewer consume any of them with no ``isinstance``
check and no parallel API.

COVERAGE FOLLOWS THE SPEC, NOT THE LOCAL DISK (IMA-254). The writer list above is read out of
``control/core/job_processing.py`` (``SaveImageJob``, ``SaveOMETiffJob``, ``SaveZarrJob``) and
``control/utils.py`` (the three zarr path builders), and every one of them has a synthetic
fixture in ``tests/conftest.py`` sized in kilobytes. Before this, two writers were unserved
because only two acquisitions had ever been tested against and both came from the same writer;
one of the two — MULTI_PAGE_TIFF — was not refused but SILENTLY skipped file-by-file, so a full
acquisition reported as empty. Every unsupported or malformed layout now fails loudly, naming
the format it found and the formats it expected. ``tests/test_writer_coverage.py`` asserts there
is no ``continue`` past a file matching a known Squid naming pattern.

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
        coordinates.csv (dedup by x,y) ───┤─► fov_positions_um {(region,fov): (x_um, y_um)}
        acquisition.yaml (or JSON) ───────┴─► dz_um, pixel_size_um, wellplate_format, Nz/Nt cross-check

The (region, fov, z, channel) index is parsed from FILENAMES — the ground truth. Scalar
metadata comes from acquisition.yaml (authoritative pixel size etc.), the flat JSON as a
legacy fallback. read() constructs the path directly and returns exactly what tifffile
decodes (native dtype), refusing non-2D planes and dtypes outside {uint8, uint16}.

``coordinates.csv`` IS read (IMA-187), into ``metadata["fov_positions_um"]``, so multiple FOVs
per region can be placed at their true stage offsets. TWO schemas ship in real Squid output and
both parse (IMA-215), discriminated by the HEADER and never by row count::

    (a) region,fov,z_level,x (mm),y (mm),z (um),time     # fov id STATED
    (b) region,x (mm),y (mm),z (mm)                      # fov id is ROW ORDER

In (a) the ``fov`` column is the row -> image mapping, so row order is irrelevant; repeats (one
row per z-level) collapse, and a repeat that disagrees about x/y is a hard error.

In (b) there is NO ``fov`` column — the only link from a row to an image is position within the
region's rows, so the Nth row of a region maps to that region's Nth sorted FOV. Rows are
de-duplicated on (region, x, y) FIRST, because a multi-z / multi-timepoint acquisition can
repeat a stage position once per z-level; comparing raw row counts against FOV counts would
then fail on every genuine z-stack. The de-duplicated count IS cross-checked and fails loud
on a mismatch: a wrong mapping does not crash, it silently draws a scrambled mosaic. That
failure is CONTAINED to the coordinate half of the metadata (``_fov_positions_um_or_empty``):
a truncated CSV costs placement, not the whole acquisition.

Units: coordinates.csv records MILLIMETRES, but this package's world space is MICROMETRES and
every world-space key ends in ``_um`` (see ``squidmip/_tiling.py``). The conversion happens
here, at the producer, and the metadata key is ``fov_positions_um``.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from squidmip._acquisition import Acquisition, load_acquisition_metadata
from squidmip._channels import fallback_color, load_channel_yaml, resolve_channels

# region has no underscore; fov and z are ints; channel is the remainder (may contain _ and -).
_STEM_RE = re.compile(r"^(?P<region>[^_]+)_(?P<fov>\d+)_(?P<z>\d+)_(?P<channel>.+)$")
_TIFF_SUFFIXES = (".tiff", ".tif")

# IMA-254. Squid's SaveImageJob has TWO on-disk shapes, selected by _def.FILE_SAVING_OPTION:
#
#   FileSavingOption.<default>         {region}_{fov}_{z}_{channel}.tiff    -> _STEM_RE
#   FileSavingOption.MULTI_PAGE_TIFF   {region}_{fov:0{FILE_ID_PADDING}}_stack.tiff
#
# The fov field is ZERO-PADDED to control._def.FILE_ID_PADDING, which is a deployment setting
# (0 on the reference config, but sites set it to 3, 4, ...). The width is therefore parsed, never
# assumed: ``\d+`` then ``int()``. Writing the padding into the pattern would make this reader
# silently blind on any rig configured differently — the exact class of failure IMA-254 is about.
_STACK_STEM_RE = re.compile(r"^(?P<region>[^_]+)_(?P<fov>\d+)_stack$")

# TIFF tags Squid's multi-page writer populates per page (job_processing.SaveImageJob.save_image).
_TAG_IMAGE_DESCRIPTION = 270
_TAG_PAGE_NAME = 285

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


_COORDS_NAME = "coordinates.csv"
# Header tolerance: Squid writes "x (mm)"/"y (mm)", but whitespace and case drift across
# generations. Match on the leading axis letter of a column that mentions mm.
_X_COL_RE = re.compile(r"^\s*x\b.*\(\s*mm\s*\)", re.I)
_Y_COL_RE = re.compile(r"^\s*y\b.*\(\s*mm\s*\)", re.I)


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


# IMA-215: the second real on-disk schema carries an explicit ``fov`` column. Its presence in the
# HEADER is the whole format signal — see _fov_column.
_FOV_COL_RE = re.compile(r"^\s*fov\s*$", re.I)


def _fov_column(fieldnames):
    """The explicit ``fov`` column if this coordinates.csv has one, else ``None``.

    This single header lookup IS the format discriminator (IMA-215). Two schemas ship in real
    Squid output:

        (a) ``region,fov,z_level,x (mm),y (mm),z (um),time``   — the fov id is STATED
        (b) ``region,x (mm),y (mm),z (mm)``                    — the fov id is row ORDER

    Detection must be on the header and never on row counts. Row counts are data: a type-(a) file
    can happen to have exactly one row per FOV (a single-z acquisition), and a type-(b) file can
    happen to have a count that looks like a z-repeat pattern. Guessing from them would silently
    swap the two mappings, and a swapped mapping does not crash — it draws every tile in the wrong
    place. The header is the schema; the schema decides.
    """
    return next((n for n in list(fieldnames or []) if n and _FOV_COL_RE.match(n)), None)


_MM_TO_UM = 1000.0


def _parse_mm_pair(raw_x: str, raw_y: str, region: str, line_no: int):
    """``(x_mm, y_mm)`` floats, or ``None`` for a blank (position-less) row. Raises on garbage."""
    if not raw_x or not raw_y:
        return None                     # a blank position row carries no placement info
    try:
        return float(raw_x), float(raw_y)
    except ValueError:
        raise ValueError(
            f"{_COORDS_NAME} line {line_no}: region {region!r} has non-numeric "
            f"coordinates ({raw_x!r}, {raw_y!r}); refusing to guess a stage position."
        ) from None


def _positions_from_fov_column(reader, fovs_per_region: dict, fov_col, x_col, y_col) -> dict:
    """Parse the type-(a) ("monkey") schema, where each row STATES its FOV id (IMA-215).

    ``region,fov,z_level,x (mm),y (mm),z (um),time``. This is the strictly better of the two
    schemas: the row -> FOV mapping is declared, so row order is irrelevant and a shuffled or
    interleaved file still places correctly. The positional fallback in
    :func:`load_fov_positions_um` exists only because the type-(b) schema has nothing else to go on.

    Three properties are enforced, all for the same reason the positional path enforces its count:
    a wrong mapping renders a plausible-looking, wrong mosaic and nothing downstream can catch it.

    1. **Repeats of the same FOV are collapsed, not counted.** A z-stack writes one row per
       z-level per FOV (the real 10x dataset: 550 rows, 55 FOVs, Nz=10). The first row for a FOV
       wins.
    2. **A repeat that DISAGREES about x/y is a hard error**, not a silent first-wins. Two
       different stage positions filed under one FOV id means the file is corrupt or was
       concatenated from two runs; picking either one would be a guess.
    3. **The FOV id set must match the filename-derived set exactly.** An id in the CSV with no
       image (or an image with no row) leaves part of the plate unplaceable.

    Units: x/y are millimetres here exactly as in the type-(b) schema, so the same single
    ``_MM_TO_UM`` conversion applies. The ``z (um)`` column is ALREADY micrometres and is not
    read at all — there is no third unit crossing this boundary and no second scale factor.
    """
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
        key = (round(x, 6), round(y, 6))    # tolerate float-repr drift, as the positional path does
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


def _parse_fov_positions_um(root, fovs_per_region: dict) -> tuple:
    """Parse ``coordinates.csv`` into ``({(region, fov): (x_um, y_um)}, mismatched)`` — MICROMETRES.

    Returns BOTH halves of the cross-check: the positions of every region that passed, and
    ``mismatched`` mapping ``region -> (n_positions, n_fovs)`` for every region that did not.
    Splitting it this way is what lets one truncated well cost only its own mosaic:
    :func:`load_fov_positions_um` raises if ``mismatched`` is non-empty (the strict contract),
    while :func:`_fov_positions_um_or_empty` keeps the regions that passed and warns about the
    rest. Neither can place a FOV at an unverified position, which is the invariant that matters.

    The file records millimetres; world space in this package is micrometres (``_tiling.py``),
    and the units invariant is that every world-space value is µm and every key carrying one
    ends in ``_um``. The mm -> µm conversion therefore happens HERE, at the single producer,
    rather than in each consumer — an unsuffixed mm value crossing into µm code is a silent
    1000x error that draws a plausible picture.

    Returns ``{}`` (present but empty — never a missing key) when the file is absent, so a
    consumer can degrade to single-FOV rendering instead of hitting a KeyError.

    The mapping is **row order within a region**: coordinates.csv carries no ``fov`` column, so
    the Nth distinct position recorded for a region is that region's Nth sorted FOV. Two
    safeguards keep that honest:

    1. **De-duplicate on (region, x, y) before counting.** A multi-z or multi-timepoint
       acquisition can write one row per z-level at the same stage position; raw row counts
       would then be a multiple of the FOV count and every z-stack would fail the check below.
       De-duplicating first makes the count mean "distinct stage positions", which is the thing
       that should equal the FOV count. First occurrence wins, preserving file order.
    2. **Cross-check the de-duplicated count against the filename-derived FOV count** and raise
       on a mismatch. This is the load-bearing guard: an off-by-one or a truncated CSV produces
       a mosaic that looks entirely reasonable while every tile sits in the wrong place, and
       nothing downstream can detect it.

    Rows whose region is not in *fovs_per_region* are ignored (a CSV may describe regions whose
    images were never written); a region with images but no rows is simply absent from the
    result, and placement for it raises later with a clear message.
    """
    import csv

    path = Path(root) / _COORDS_NAME
    if not path.exists():
        return {}, {}

    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        x_col, y_col = _coord_columns(reader.fieldnames)
        fov_col = _fov_column(reader.fieldnames)
        if fov_col is not None:
            # An explicit 'fov' column names each position outright, so there is no row-order
            # inference to cross-check and no mismatch to report.
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
            # De-duplication compares the raw mm values; only the stored value is converted.
            ordered.setdefault(region, []).append((x * _MM_TO_UM, y * _MM_TO_UM))

    positions: dict = {}
    mismatched: dict = {}
    for region, coords in ordered.items():
        fovs = list(fovs_per_region[region])
        if len(coords) != len(fovs):
            # Record and skip rather than abort the loop. The cross-check is per region --
            # one truncated well says nothing about the others -- and a caller that wants the
            # strict all-or-nothing contract raises on this dict (see load_fov_positions_um).
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
    """Strict parse: every region cross-checks, or nothing is returned.

    Raises :class:`ValueError` naming every region whose de-duplicated position count
    disagrees with its FOV count. See :func:`_parse_fov_positions_um` for the parsing rules
    and :func:`_fov_positions_um_or_empty` for the degrading variant the viewer uses.
    """
    positions, mismatched = _parse_fov_positions_um(root, fovs_per_region)
    if mismatched:
        raise ValueError(_mismatch_message(mismatched))
    return positions


def _fov_positions_um_or_empty(root, fovs_per_region: dict) -> dict:
    """``load_fov_positions_um`` degraded PER REGION on an unusable coordinates.csv.

    ``metadata`` is the acquisition's whole identity: regions, channels, dtype, frame shape.
    Those come from the FILENAMES and one decoded frame and are readable whatever the CSV says.
    Before this, a truncated or malformed coordinates.csv raised out of the middle of the
    metadata dict literal, so every one of those fields became unreachable and the viewer
    reported "not a readable Squid acquisition" for an acquisition it could render perfectly
    well minus the multi-FOV mosaic (IMA-187).

    Degrading is safe precisely because a region absent from the mapping already means "no
    stage positions for that region" — consumers fall back to single-tile rendering, and
    ``_placement.fov_offsets_px`` raises a KeyError naming the missing FOVs rather than
    guessing. It does NOT weaken the cross-check: an ambiguous region still never produces a
    scrambled mosaic, it produces no mosaic, loudly (``UserWarning``).

    The degradation is per REGION, not per file. A plate where one well was cut short mid-run
    (its CSV rows outnumber its written FOVs) used to cost every OTHER well its mosaic too,
    because the first mismatch raised and the whole mapping collapsed to ``{}``. One truncated
    well says nothing about the wells that cross-check perfectly, so only the well that failed
    loses its positions now.

    A malformed FILE — unparseable coordinates, no recognisable x/y columns, conflicting
    duplicate rows — is still all-or-nothing: those :class:`ValueError`\\ s come from the parse
    itself, before any region can be judged, and there is nothing partial to salvage. Anything
    that is not a :class:`ValueError` still propagates.
    """
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
        # Name both halves. The old message said only that something was unusable, which read
        # as a whole-plate failure even when a single well was at fault.
        warnings.warn(
            f"{_COORDS_NAME} is unusable for {len(mismatched)} of "
            f"{len(mismatched) + len(kept)} region(s) ({_mismatch_message(mismatched)}) — "
            f"those regions render as a single tile instead of a coordinate-placed mosaic. "
            f"Kept stage positions for: {', '.join(kept) if kept else '(none)'}."
        )
    return positions


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


def _time_folders(path: Path) -> list[Path]:
    """Squid's timepoint folders (``0/``, ``1/``, …) under *path*, else *path* itself.

    Shared by every TIFF-shaped reader: ``multi_point_worker`` writes to
    ``{experiment_path}/{time_point:0{FILE_ID_PADDING}}``, so the folder NAME is a zero-padded
    integer of unknown width and the sort must be numeric, not lexicographic.
    """
    numeric = [d for d in path.iterdir() if d.is_dir() and d.name.isdigit()]
    return sorted(numeric, key=lambda d: int(d.name)) if numeric else [path]


def _classify_tiff_folder(folder: Path) -> tuple[int, int, list]:
    """``(n_individual, n_stack, other_names)`` for one timepoint folder.

    The dispatch evidence, gathered ONCE so the error message can state what was actually on
    disk. ``other_names`` is capped — an error listing 20 000 filenames is not a message.
    """
    individual = stacks = 0
    other: list = []
    for f in sorted(folder.iterdir()):
        if not f.is_file() or f.suffix.lower() not in _TIFF_SUFFIXES:
            continue
        if _STEM_RE.match(f.stem):
            individual += 1
        elif _STACK_STEM_RE.match(f.stem):
            stacks += 1
        elif len(other) < 8:
            other.append(f.name)
    return individual, stacks, other


def open_reader(path) -> "SquidReader":
    """Detect the acquisition format at *path* and return a reader.

    Dispatch covers every writer in ``control/core/job_processing.py`` (IMA-254). The mapping,
    verified against that module rather than against whichever acquisitions happen to be on a
    given machine — the reason two writers went unserved is that only two acquisitions were ever
    tested, and both came from the same writer:

    ===========================================  =======================================  =========
    Squid writer                                 on-disk shape                            reader
    ===========================================  =======================================  =========
    ``SaveImageJob`` (default)                   ``{t}/{region}_{fov}_{z}_{ch}.tiff``     SquidReader
    ``SaveImageJob`` MULTI_PAGE_TIFF             ``{t}/{region}_{fov}_stack.tiff``        SquidMultiPageTiffReader
    ``SaveOMETiffJob``                           ``ome_tiff/{region}_{fov}.ome.tiff``     SquidOMEReader
    ``SaveZarrJob`` HCS                          ``plate.ome.zarr/{row}/{col}/{fov}/0``   SquidZarrReader
    ``SaveZarrJob`` non-HCS per-FOV              ``zarr/{region}/fov_{n}.ome.zarr/0``     SquidZarrReader
    ``SaveZarrJob`` non-HCS 6D (non-standard)    ``zarr/{region}/acquisition.zarr``       SquidZarrReader
    ===========================================  =======================================  =========

    Anything else raises, NAMING what was found and what was expected. There is no path on which
    an unrecognised layout yields an empty-looking acquisition: a reader that reports "0 images"
    for a directory full of images is worse than one that refuses, because the operator believes
    it.
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
    store = _find_zarr_store(path)
    if store is not None:
        return SquidZarrReader(store, acquisition_root=path)

    folder = _time_folders(path)[0]
    individual, stacks, other = _classify_tiff_folder(folder)
    if individual:
        if stacks:
            # Both writers' output in one folder. Serve the richer one, but say so: this is
            # either a re-run over an existing folder or a config change mid-acquisition, and
            # silently ignoring half the files is how IMA-254 happened in the first place.
            warnings.warn(
                f"{folder!s} contains BOTH {individual} individual-TIFF plane(s) "
                f"({{region}}_{{fov}}_{{z}}_{{channel}}.tiff) and {stacks} multi-page stack(s) "
                "({region}_{fov}_stack.tiff). Squid writes one or the other per acquisition, so "
                "this folder holds two runs. Reading the individual TIFFs and IGNORING the "
                "stacks — split them into separate folders to read the stacks."
            )
        return SquidReader(path)
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


# IMA-254. Squid's non-HCS Zarr output nests one MORE level than IMA-229 assumed:
# ``zarr/{region_id}/fov_{n}.ome.zarr`` (``control/utils.build_per_fov_zarr_path``), one NGFF image
# group per FOV, with the array at ``…/fov_{n}.ome.zarr/0``. ``n`` is the raw FOV index, unpadded.
_PER_FOV_ZARR_RE = re.compile(r"^fov_(?P<fov>\d+)\.ome\.zarr$")

# ``control/utils.build_6d_zarr_path``: the non-standard 6D layout, one (FOV, T, C, Z, Y, X) array
# per region. Note this node is a zarr ARRAY, not a group — Squid's ZarrWriter writes the OME
# metadata into the array's own ``zarr.json`` attributes with ``datasets[0].path == "."`` — so
# ``_is_zarr_group`` is False for it and it must be recognised by NAME.
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

    Recognised (IMA-229, extended in IMA-254), in priority order:

    * *path* IS a zarr group — an ``…​.ome.zarr`` plate or image group handed in directly;
    * ``<path>/plate.ome.zarr`` — Squid's (and SquidMIP's own writer's) canonical HCS output;
    * ``<path>/*.zarr`` — any other single ``.zarr`` sibling, e.g. a renamed plate;
    * ``<path>/zarr/`` — the non-HCS layouts. THREE shapes live under here and all three are
      served: a bare image group per region (``zarr/{region}/`` itself, what IMA-229 assumed and
      what SquidMIP's own round-trip writes), Squid's real per-FOV
      ``zarr/{region}/fov_{n}.ome.zarr``, and the non-standard 6D
      ``zarr/{region}/acquisition.zarr``.

    A ``zarr/`` folder that holds region subdirectories but nothing recognisable RAISES rather
    than returning ``None``. Returning ``None`` would fall through to the individual-TIFF reader,
    whose complaint ("no {region}_{fov}_{z}_{channel}.tiff") names the wrong format entirely and
    sends the reader hunting for missing TIFFs in a Zarr acquisition.
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

    def __init__(self, path) -> None:
        self._path = Path(path)
        self._time_folders: Optional[list[Path]] = None
        self._index: Optional[dict] = None
        self._meta: Optional[dict] = None

    # -- timepoints -------------------------------------------------------
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
        index: dict = {}
        skipped: list = []
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in _TIFF_SUFFIXES:
                continue
            m = _STEM_RE.match(f.stem)
            if m:
                key = (m["region"], int(m["fov"]), int(m["z"]), m["channel"])
                index[key] = f.suffix
                continue
            # IMA-254. This branch used to be a bare ``continue`` with a comment naming the very
            # format it was discarding. On a MULTI_PAGE_TIFF acquisition EVERY file took it, the
            # index came out empty, and the acquisition reported as unreadable-because-empty
            # rather than as the wrong reader for a known Squid format. The code knew and said
            # nothing. A stem that matches a known Squid pattern now raises here; anything else
            # is remembered so the empty-index error can name it.
            if _STACK_STEM_RE.match(f.stem):
                raise ValueError(
                    f"{f.name} is Squid's MULTI_PAGE_TIFF layout "
                    "({region}_{fov:0FILE_ID_PADDING}_stack.tiff, written by SaveImageJob when "
                    "_def.FILE_SAVING_OPTION == FileSavingOption.MULTI_PAGE_TIFF), not the "
                    "individual-TIFF layout ({region}_{fov}_{z}_{channel}.tiff) this reader "
                    "serves. Use squidmip.open_reader(), which dispatches to "
                    "SquidMultiPageTiffReader for this format."
                )
            if len(skipped) < 8:
                skipped.append(f.name)
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
        self._meta = Acquisition(**{
            "regions": regions,
            "fovs_per_region": fovs_per_region,
            # {(region, fov): (x_um, y_um)} — MICROMETRES, per the package units invariant.
            # {} when coordinates.csv is absent OR unusable (never raises out of metadata).
            # Present on BOTH reader classes so consumers never have to ask which reader they
            # hold (IMA-187).
            "fov_positions_um": _fov_positions_um_or_empty(self._path, fovs_per_region),
            "channels": resolve_channels(sorted(channels), load_channel_yaml(self._path)),
            "n_z": n_z,
            "z_levels": z_sorted,
            "dz_um": acq["dz_um"],
            "pixel_size_um": acq["pixel_size_um"],  # authoritative (acquisition.yaml), not recomputed
            "wellplate_format": acq["wellplate_format"],
            "frame_shape": tuple(sample.shape),
            "dtype": sample.dtype,
            "n_t": n_t,
        })
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


def _page_json(page, path: Path, page_index: int) -> dict:
    """The per-page metadata dict Squid embeds in ImageDescription, or a loud failure.

    ``SaveImageJob.save_image`` calls ``TiffWriter.write(metadata=..., description=json.dumps(...),
    extratags=[(285, 's', 0, channel, False)])``. tifffile honours BOTH arguments, so each page
    carries TWO ImageDescription (270) tags: Squid's own JSON first, then tifffile's "shaped" JSON
    (the same keys plus ``shape``). Either answers the question, so every 270 tag is tried and the
    first one that parses as an object carrying ``z_level`` wins — that tolerates a tifffile
    version that emits only one of them, in either order.
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
    """The page's channel name: PageName (tag 285) first, the JSON ``channel`` key as fallback.

    Tag 285 is the primary source because it is the one field a generic TIFF viewer also shows,
    so it is what an operator sees; the JSON copy exists for readers that drop unknown tags. When
    both are present and DISAGREE the file is refused rather than resolved by precedence — two
    different channel names for one plane means the writer or a post-processing step is broken,
    and picking either one mislabels pixels in a way nothing downstream can detect.
    """
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
    """Lazy reader over Squid's MULTI_PAGE_TIFF acquisitions (IMA-254).

    Layout, from ``control/core/job_processing.py`` ``SaveImageJob.save_image`` on the
    ``_def.FILE_SAVING_OPTION == FileSavingOption.MULTI_PAGE_TIFF`` branch::

        <acq>/{t}/{region}_{fov:0{FILE_ID_PADDING}}_stack.tiff

    ONE file per (timepoint, region, FOV), APPENDED to one page at a time — every (z, channel) of
    that field is a separate IFD page in whatever order the acquisition happened to run. The file
    carries no series axes (tifffile reports four independent ``YX`` series for four pages), so
    page ORDER is not a usable index.

    The index therefore comes from each page's own metadata, never from its position:

    * ``PageName`` (tag 285) -> channel name;
    * ``ImageDescription`` (tag 270) -> JSON with ``z_level``, ``channel``, ``channel_index``,
      ``region_id``, ``fov``, ``x_mm``, ``y_mm``, ``z_mm``, ``time`` and, when the piezo is in
      use, ``z_piezo (um)``.

    That is also why an interrupted or re-ordered acquisition still reads correctly, and why a
    page missing its metadata is refused instead of being slotted in by position.

    Presents the SAME interface as :class:`SquidReader`, :class:`SquidOMEReader` and
    :class:`SquidZarrReader` — the identical ``metadata`` key set (micrometres, ``_um``-suffixed)
    plus ``read``/``plane_ref``. The interface is the seam; no consumer branches on reader type.

    Positions: this writer records the stage position INLINE, per page, in MILLIMETRES
    (``x_mm``/``y_mm``). The mm -> um conversion happens here, at the producer, into
    ``fov_positions_um``, exactly as ``coordinates.csv`` is converted in
    :func:`load_fov_positions_um`. There is no second scale factor anywhere downstream.
    """

    def __init__(self, path) -> None:
        self._path = Path(path)
        self._time_folders_cache: Optional[list[Path]] = None
        self._indexes: dict[int, dict] = {}     # t -> {(region, fov, z, channel): (Path, page)}
        self._positions_mm: dict = {}           # {(region, fov): (x_mm, y_mm)} from t=0
        self._meta: Optional[dict] = None
        self._handles: dict = {}                # Path -> tifffile.TiffFile (cached)

    # -- discovery ---------------------------------------------------------
    def _discover_time_folders(self) -> list[Path]:
        if self._time_folders_cache is None:
            self._time_folders_cache = _time_folders(self._path)
        return self._time_folders_cache

    def _tif(self, path: Path):
        tif = self._handles.get(path)
        if tif is None:
            tif = tifffile.TiffFile(path)
            self._handles[path] = tif
        return tif

    def _index_for(self, t: int) -> dict:
        """``{(region, fov, z, channel): (path, page_index)}`` for one timepoint folder."""
        if t in self._indexes:
            return self._indexes[t]
        folder = self._discover_time_folders()[t]
        index: dict = {}
        stacks = [f for f in sorted(folder.iterdir())
                  if f.suffix.lower() in _TIFF_SUFFIXES and _STACK_STEM_RE.match(f.stem)]
        for f in stacks:
            m = _STACK_STEM_RE.match(f.stem)
            # FILE_ID_PADDING is a deployment setting; int() reads whatever width was written.
            region, fov = m["region"], int(m["fov"])
            tif = self._tif(f)
            for page_index, page in enumerate(tif.pages):
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
                if t == 0:
                    self._record_position(region, fov, payload)
        if not index:
            raise ValueError(
                "No Squid MULTI_PAGE_TIFF stacks ({region}_{fov}_stack.tiff) found in "
                f"{folder!s}"
            )
        self._indexes[t] = index
        return index

    def _record_position(self, region: str, fov: int, payload: dict) -> None:
        """First page wins for a field's stage position — deliberately, and without a cross-check.

        ``multi_point_worker`` re-reads ``stage.get_pos()`` for EVERY capture, so the x/y recorded
        on a field's z=1 page differs from its z=0 page by the stage's own repeatability. Treating
        that jitter as a conflict (the way ``coordinates.csv`` treats two genuinely different
        positions filed under one fov id) would refuse every real z-stack. The first page of a
        field is the position at which that field was first imaged, which is the tile origin.
        """
        key = (str(region), int(fov))
        if key in self._positions_mm:
            return
        try:
            x_mm, y_mm = float(payload["x_mm"]), float(payload["y_mm"])
        except (KeyError, TypeError, ValueError):
            return          # a page without a position simply contributes none
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
        if acq["n_z_declared"] is not None and acq["n_z_declared"] != n_z:
            warnings.warn(
                f"Recorded Nz ({acq['n_z_declared']}) != distinct z levels in the stack pages "
                f"({n_z}); using the page-derived value."
            )
        if acq["n_t_declared"] is not None and acq["n_t_declared"] != n_t:
            warnings.warn(
                f"Recorded Nt ({acq['n_t_declared']}) != timepoint folders found ({n_t}); "
                "using the folder-derived value."
            )

        # frame shape + dtype come from a real decoded page — they vary with binning / pixel format.
        s_region, s_fov, s_z, s_channel = next(iter(index))
        sample = self.read(s_region, s_fov, s_channel, s_z)
        self._meta = Acquisition(**{
            "regions": regions,
            "fovs_per_region": fovs_per_region,
            "fov_positions_um": self._positions_um(fovs_per_region),
            "channels": resolve_channels(sorted(channels), load_channel_yaml(self._path)),
            "n_z": n_z,
            "z_levels": z_sorted,
            "dz_um": acq["dz_um"],
            "pixel_size_um": acq["pixel_size_um"],
            "wellplate_format": acq["wellplate_format"],
            "frame_shape": tuple(sample.shape),
            "dtype": sample.dtype,
            "n_t": n_t,
        })
        return self._meta

    def _positions_um(self, fovs_per_region: dict) -> dict:
        """``{(region, fov): (x_um, y_um)}`` — MICROMETRES, converted HERE at the single producer.

        The pages' inline ``x_mm``/``y_mm`` are authoritative for this writer: they are the stage
        position of the capture itself, not a plan of where it was supposed to go. A sibling
        ``coordinates.csv`` is the fallback only when no page carried a position at all.
        """
        self._index_for(0)                       # populates _positions_mm
        if self._positions_mm:
            return {k: (x * _MM_TO_UM, y * _MM_TO_UM)
                    for k, (x, y) in self._positions_mm.items()}
        return _fov_positions_um_or_empty(self._path, fovs_per_region)

    # -- read --------------------------------------------------------------
    def _locate(self, region, fov, channel, z, t) -> tuple:
        time_folders = self._discover_time_folders()
        t = int(t)
        if not 0 <= t < len(time_folders):
            raise IndexError(f"t={t} out of range (n_t={len(time_folders)}).")
        index = self._index_for(t)
        key = (str(region), int(fov), int(z), str(channel))
        if key not in index:
            raise KeyError(
                f"No such plane region={region!r} fov={fov} channel={channel!r} z={z}. "
                f"Known regions={sorted({k[0] for k in index})}, "
                f"channels={sorted({k[3] for k in index})}."
            )
        return index[key]

    def read(self, region, fov, channel, z, t=0):
        """Return one plane as a 2D array in its native dtype (decodes exactly one IFD page)."""
        path, page_index = self._locate(region, fov, channel, z, t)
        page = self._tif(path).pages[page_index]
        return _validate_plane(np.asarray(page.asarray()), path)

    def plane_path(self, region, fov, channel, z, t=0) -> Path:
        """The stack file holding this plane. Unlike the individual-TIFF reader's one-file-per-
        plane, this path holds the whole field — see :meth:`plane_ref` for the addressable form."""
        return self._locate(region, fov, channel, z, t)[0]

    def plane_ref(self, region, fov, channel, z, t=0) -> tuple:
        """``(filepath, page_index)`` — the viewer registers this straight into ndviewer, so the
        raw stack displays from the .tiff on disk with zero bytes copied."""
        path, page_index = self._locate(region, fov, channel, z, t)
        return str(path), page_index


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
        self._meta = Acquisition(**{
            "regions": regions,
            "fovs_per_region": fovs_per_region,
            # Same key, same meaning as SquidReader — the shared-interface promise in this
            # class's docstring. An OME acquisition with a sibling coordinates.csv gets real
            # placement; without one this is {} and the mosaic degrades to a single field.
            # (Positions inside the OME-XML are not read: different parsing path, no dataset
            # on hand to validate it against. See .spec NOT-in-scope.)
            "fov_positions_um": _fov_positions_um_or_empty(self._path, fovs_per_region),
            "channels": channels,
            "n_z": n_z,
            "z_levels": list(range(n_z)),
            "dz_um": acq["dz_um"],
            "pixel_size_um": acq["pixel_size_um"],
            "wellplate_format": acq["wellplate_format"],
            "frame_shape": (int(dims.get("Y", sample.shape[-2])), int(dims.get("X", sample.shape[-1]))),
            "dtype": np.dtype(sample.dtype),
            "n_t": n_t,
        })
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


# ==================================================================================================
# IMA-229: Zarr input (OME-NGFF)
# ==================================================================================================
#
# PRIOR ART, and what was adopted. The layout below is not invented; it is read straight from the
# OME-NGFF specification sources (github.com/ome/ngff-spec, branches 0.4 and 0.5 — index.bs, the
# JSON schemas and the published examples) and cross-checked against what SquidMIP's OWN writer
# (``squidmip/_output.py``, already validated against the official ``ome-zarr-models`` pydantic
# schema in ``tests/ngff_check.py``) emits. Anything a real NGFF reader (ome-zarr-py, ngio, napari)
# cannot open would be a bug here.
#
#   HCS plate      ``plate.ome.zarr/{row}/{col}/{fov}/{level}``
#     plate group   ``plate`` -> ``rows``/``columns``/``wells``; each well entry has ``path``
#                   ("A/1" — the row NAME then the column NAME, regex ^[A-Za-z0-9]+/[A-Za-z0-9]+$),
#                   ``rowIndex``, ``columnIndex``. The region id is the concatenation ``row+col``
#                   ("B" + "2" = "B2"), the exact inverse of ``_output.parse_well_id``.
#     well group    ``well`` -> ``images``: a list of ``{"path": ...}``. The spec allows ANY
#                   alphanumeric field path, so the listed paths are used verbatim rather than
#                   assuming 0..n-1 — Squid writes the raw (possibly non-contiguous) FOV id there.
#     image group   ``multiscales`` (+ optional ``omero``) — see below.
#
#   non-HCS        ``zarr/{region_id}/{level}`` — a bare image group per region, no plate node.
#                  Each region gets the single FOV 0; there is no well/field level to read.
#
#   multiscales    ``axes`` (2-5 entries, ordered time, then channel, then space z,y,x) and
#                  ``datasets`` (ordered highest -> lowest resolution; each has ``path`` and
#                  ``coordinateTransformations`` = exactly one ``scale``, optionally followed by
#                  one ``translation``). Level 0 (``datasets[0]``) is the full-resolution array and
#                  is the only one this reader serves — a MIP must never be computed from a
#                  downsampled level. Array SHAPES come from the arrays, never from a scale factor:
#                  the writer's 2x2 block-mean crops odd axes, so level shapes are floor(prev/2).
#
#   VERSIONS       v0.4 is zarr v2 — group metadata is a ``.zattrs`` file with ``plate`` / ``well``
#                  / ``multiscales`` at the TOP level. v0.5 is zarr v3 — a single ``zarr.json`` with
#                  the same payload namespaced under ``attributes.ome``. Both are read; refusing
#                  either would lock out half the real stores. ``_group_attrs`` is the one place
#                  that difference lives.
#
#   POSITIONS      The dataset-level ``translation`` (applied AFTER ``scale``, so it is already in
#                  physical units) is the ONLY position mechanism the spec defines — there is no
#                  well-level or plate-level stage metadata in either version. SquidMIP's writer
#                  emits no translation, so a sibling ``coordinates.csv`` is the documented
#                  fallback, and either way the result lands in ``fov_positions_um``.
#
#   UNITS          ``axes[].unit`` is a UDUNITS-2 string and is only a SHOULD, so it can be absent.
#                  Every physical value taken out of a store (pixel size, dz, translation) is
#                  converted to MICROMETRES HERE, at this single producer, by the axis's own unit —
#                  a store written in millimetres must not reach a ``_um`` key as millimetres. That
#                  is the same failure that was fixed on main; there is exactly one conversion and
#                  no consumer compensates for it.

_ZARR_V3_META = "zarr.json"
_ZARR_V2_GROUP = ".zgroup"
_ZARR_V2_ATTRS = ".zattrs"
_ZARR_V2_ARRAY = ".zarray"

# UDUNITS-2 length units -> micrometres. Absent/unknown units are treated as micrometres (the
# de-facto microscopy default) rather than guessed at, and never silently rescaled.
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
    """The OME metadata payload of a zarr group, normalising the v0.4 / v0.5 difference.

    v0.5 (zarr v3) nests it as ``zarr.json -> attributes -> ome``; v0.4 (zarr v2) puts the same
    keys at the top level of ``.zattrs``. Callers then read ``plate`` / ``well`` / ``multiscales``
    without caring which version wrote the store. A v3 group whose attributes carry no ``ome``
    namespace (e.g. a bioformats2raw-style store) falls back to the raw attributes.
    """
    path = Path(path)
    v2 = path / _ZARR_V2_ATTRS
    if v2.exists():
        return json.loads(v2.read_text() or "{}")
    v3 = path / _ZARR_V3_META
    if v3.exists():
        attrs = json.loads(v3.read_text() or "{}").get("attributes") or {}
        ome = attrs.get("ome")
        return ome if isinstance(ome, dict) else attrs
    return {}


def _open_zarr_array(path: Path):
    """Open one zarr array (v2 or v3) as a lazy tensorstore handle: no data is read here.

    Goes through the process-wide pool (``_tsctx``) rather than opening directly, so every reader
    binds to ONE bounded ``cache_pool`` and the number of live handles is capped. This used to
    fill ``SquidZarrReader._arrays``, an unbounded per-reader dict with no eviction.
    """
    from squidmip._tsctx import HANDLES

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
    """The parsed ``multiscales[0]`` of one image group: axis order, level-0 path, scale, offset.

    Everything physical it exposes is already in MICROMETRES (see the UNITS note above).
    """

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
        # Axis NAMES are the dimension order; the spec fixes the ORDER (t, c, then z, y, x) but
        # not which axes are present, so a 2-D or 4-D store is legal and must still map cleanly.
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

    def index(self, shape, t: int, c: int, z: int, fov: int = 0) -> tuple:
        """The tensorstore index tuple selecting the single ``(y, x)`` plane at (t, c, z[, fov]).

        ``fov`` matters only for the 6D layout; a 5D store has no ``fov`` axis and the value is
        simply never consulted. It is threaded through rather than defaulted inside because the
        alternative — letting the unknown axis fall to ``picks.get(n, 0)`` — served FOV 0's pixels
        for every FOV of the region, which is a silently wrong image, not an error.
        """
        picks = {"t": t, "c": c, "z": z, "fov": fov}
        return tuple(
            slice(None) if n in ("y", "x") else picks.get(n, 0)
            for n in self.axis_names[: len(shape)]
        )

    def size(self, shape, name: str, default: int = 1) -> int:
        i = self._axis(name)
        return int(shape[i]) if i is not None and i < len(shape) else default


class SquidZarrReader:
    """Lazy reader over an OME-NGFF Zarr acquisition — HCS plate or bare per-region image groups.

    Presents the SAME interface as :class:`SquidReader` and :class:`SquidOMEReader`: the identical
    ``metadata`` key set with the identical meanings (micrometres, ``_um``-suffixed), and
    ``read(region, fov, channel, z, t)`` returning one 2-D native-dtype plane. The reader interface
    IS the seam — the engine, CLI and viewer take a Zarr acquisition with no change and no
    ``isinstance`` check. See the module-level IMA-229 block for the layout and its spec citations.

    Only resolution level 0 is served. The pyramid exists for navigation; a projection computed
    from a downsampled level would be silently wrong, so the coarse levels are never read.
    """

    def __init__(self, path, acquisition_root=None) -> None:
        self._path = Path(path)
        # Sidecars (acquisition.yaml, coordinates.csv) live beside the store, not inside it:
        # ``<acq>/plate.ome.zarr`` and ``<acq>/zarr/`` are both children of the acquisition folder.
        self._root = Path(acquisition_root) if acquisition_root is not None else self._path.parent
        self._fields: Optional[dict] = None      # {(region, fov): Path to the image group}
        self._ms: dict = {}                      # image group Path -> _Multiscale (cached)
        self._arrays: dict = {}                  # image group Path -> open tensorstore (cached)
        self._meta: Optional[dict] = None

    # -- discovery ---------------------------------------------------------
    def _discover(self) -> dict:
        if self._fields is not None:
            return self._fields
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
            # The region id is the row NAME + column NAME, the inverse of _output.parse_well_id
            # (which writes B2 -> B/2, never B/02, so concatenation restores the true well id).
            region = "".join(rel.split("/"))
            well_dir = self._path / rel
            images = (_group_attrs(well_dir).get("well") or {}).get("images") or []
            for i, image in enumerate(images):
                name = str(image.get("path", ""))
                if not name:
                    continue
                # Field paths are arbitrary alphanumerics per spec; Squid writes the raw FOV id.
                # A non-numeric path still needs an int FOV for the shared interface, so it falls
                # back to its position in the list.
                fields[(region, int(name) if name.isdigit() else i)] = well_dir / name
        return fields

    def _discover_flat(self) -> dict:
        """Non-HCS: map every region folder under ``zarr/`` to its field image group(s).

        Three shapes, all real (IMA-254 — the first was the only one IMA-229 handled, and it is
        the one Squid does NOT write; it exists because SquidMIP's own round-trip produces it):

        ``zarr/{region}/``                     the region folder IS the image group; one FOV, id 0.
        ``zarr/{region}/fov_{n}.ome.zarr``     ``build_per_fov_zarr_path`` — 5D TCZYX per FOV.
        ``zarr/{region}/acquisition.zarr``     ``build_6d_zarr_path`` — ONE 6D FTCZYX array whose
                                               leading axis is the FOV. Every FOV of the region
                                               maps to the same node; ``_Multiscale.index`` picks
                                               the FOV out of the leading axis at read time.

        A region folder that holds none of these RAISES rather than being skipped: a region that
        vanishes from ``regions`` looks exactly like a region that was never acquired.
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
        # A store that IS a single image group (handed in directly) is that one region.
        if not fields and "multiscales" in _group_attrs(self._path):
            fields[(self._path.name.replace(".ome.zarr", "").replace(".zarr", ""), 0)] = self._path
        return fields

    def _sixd_fov_count(self, sixd: Path) -> int:
        """How many FOVs the 6D array's leading axis holds.

        Read from the ARRAY's shape, never from ``region_fov_counts`` or any sidecar: the shape is
        what was actually allocated, and a count taken from elsewhere that disagreed would index
        past the end (loud) or leave real FOVs unreachable (silent).
        """
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
        arr = self._arrays.get(group)
        if arr is None:
            arr = self._arrays[group] = _open_zarr_array(self._multiscale(group).array_path)
        return arr

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

        self._meta = Acquisition(**{
            "regions": regions,
            "fovs_per_region": fovs_per_region,
            "fov_positions_um": self._positions_um(fields, fovs_per_region),
            "channels": self._channels(ms, ms.size(shape, "c")),
            "n_z": n_z,
            "z_levels": list(range(n_z)),
            "dz_um": ms.dz_um,
            "pixel_size_um": ms.pixel_size_um,
            "wellplate_format": self._wellplate_format(regions),
            "frame_shape": (ms.size(shape, "y", shape[-2]), ms.size(shape, "x", shape[-1])),
            "dtype": dtype,
            "n_t": n_t,
        })
        return self._meta

    def _positions_um(self, fields: dict, fovs_per_region: dict) -> dict:
        """Stage positions in MICROMETRES: dataset ``translation`` first, coordinates.csv second.

        The NGFF spec defines no other position mechanism, and SquidMIP's own writer emits no
        translation — so a store round-tripped through this package legitimately has none, and the
        sibling ``coordinates.csv`` (both schemas, IMA-215) is the documented fallback. When
        neither exists the value is ``{}``: present but empty, exactly as on the TIFF readers, so
        consumers degrade to single-tile rendering instead of hitting a KeyError.
        """
        from_store = {}
        for key, group in fields.items():
            ms = self._multiscale(group)
            if ms.is_6d_fov:
                # One 6D array holds every FOV of a region behind ONE translation, so that
                # translation cannot be a per-FOV stage position. Copying it onto each FOV would
                # stack the whole region on one tile — a plausible-looking, wrong mosaic. Fall
                # through to coordinates.csv, which does record per-FOV positions.
                continue
            position = ms.position_um
            if position is not None:
                from_store[key] = position
        if from_store:
            return from_store
        return _fov_positions_um_or_empty(self._root, fovs_per_region)

    def _channels(self, ms: _Multiscale, n_c: int) -> list:
        """Channel list from ``omero.channels`` (labels + colours), the NGFF rendering metadata.

        Falls back to a sibling ``acquisition_channels.yaml``, then to generic ``C{i}`` labels with
        the shared wavelength/brightfield colour resolution. A store with neither is still opened —
        a legal NGFF image need not carry ``omero`` — but the colours are then a best effort, so the
        loss is announced rather than passed off as acquisition truth.
        """
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
                    "ex": info["ex"] if info else None,
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
            return [{"name": n, "display_name": n, "display_color": "#FFFFFF", "ex": None}
                    for n in names]

    def _wellplate_format(self, regions: list):
        """Declared (sibling acquisition.yaml) beats inferred — the IMA-219 D1 precedence."""
        try:
            declared = load_acquisition_metadata(self._root)["wellplate_format"]
        except (FileNotFoundError, ValueError):
            declared = None                     # a Zarr store need not ship acquisition.yaml
        if declared:
            return declared
        from squidmip._plate_shape import infer_plate_format

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

    def read(self, region, fov, channel, z, t=0):
        """Return one plane as a 2D array in its native dtype.

        Lazy in the sense that matters for a plate-scale store: tensorstore reads only the chunks
        covering this ``(t, c, z)`` plane, never the whole ``(T, C, Z, Y, X)`` field.
        """
        group = self._field(region, fov)
        meta = self.metadata
        z, t = int(z), int(t)
        if not 0 <= z < meta["n_z"]:
            raise IndexError(f"z={z} out of range (n_z={meta['n_z']}).")
        if not 0 <= t < meta["n_t"]:
            raise IndexError(f"t={t} out of range (n_t={meta['n_t']}).")
        arr = self._array(group)
        idx = self._multiscale(group).index(
            arr.shape, t, self._channel_index(channel), z, fov=int(fov)
        )
        plane = np.asarray(arr[idx].read().result())
        return _validate_plane(plane, self._multiscale(group).array_path)

    def plane_ref(self, region, fov, channel, z, t=0) -> tuple:
        """``(path, 0)`` for one plane, where *path* is the field's NGFF **image group**.

        The TIFF readers return a file plus an IFD page because a TIFF plane is addressable that
        way. A Zarr plane is not a file — it is a slice spread over chunk files — so the honest
        referent is the image group itself: a valid NGFF node that ndviewer and every ome-zarr
        reader can open directly, with no bytes copied. The page index is 0 to keep the tuple
        shape identical for callers.
        """
        self._channel_index(channel)            # validate like the TIFF readers do
        return str(self._field(region, fov)), 0
