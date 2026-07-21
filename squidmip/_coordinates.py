"""Per-FOV stage positions from ``coordinates.csv`` (IMA-215).

A Squid acquisition can carry the coordinate table in up to THREE places, and the two
column schemas are NOT two microscope generations — they are two files at different paths
in the SAME acquisition::

    <acq>/
    ├── coordinates.csv                    region,x (mm),y (mm),z (mm)
    │                                      ← the PLANNED grid. z column is EMPTY in 3 of 4
    │                                        real datasets. No fov column.
    ├── original_coordinates/
    │   └── original_coordinates_0.csv     ← third copy; differs from 0/ only in float repr
    └── 0/                                 # timepoint folder
        └── coordinates.csv                region,fov,z_level,x (mm),y (mm),z (um),time
                                           ← the ACTUAL executed positions, post-autofocus.

``20x_scan_2025-09-05`` carries BOTH schemas; ``synthetic_2x2_wellplate`` carries the 4-column
schema at BOTH paths. So dispatch is on the header's COLUMN SET (presence of ``fov``/``z_level``),
never on dataset provenance — and never on header spacing, since both spell it ``x (mm)``.

Load flow::

    load_fov_positions_um(root, time_folder, fovs_per_region)
             │
      locate: {t}/coordinates.csv  ─►  <acq>/coordinates.csv  ─►  none ─► {}
             │
      normalize header (BOM / CRLF / case / whitespace)
             │
      ┌──────┴───────┐
      │ has `fov`?   │
      └──┬────────┬──┘
       yes│        │no
         ▼         ▼
    group by     per-region row count
   (region,fov)  ├── exactly 1 row ─► fov = 0        (unambiguous)
         │       └── >1 row ────────► OMIT + warn    (row order is unverifiable)
    take lowest
     z_level;
    XY constant?
    tolerance+warn
         └────┬────┘
              ▼
      normalize to µm:  x,y mm×1000 │ z (um) as-is │ z (mm)×1000 │ empty → None
              ▼
      {(region, fov): (x_um, y_um, z_um | None)}

Two decisions worth keeping in view (see ``docs/ima-215-eng-review.md``):

* **Never guess row order.** The unlabelled schema has no ``fov`` column, so a fov index can
  only come from the per-region row position. That is trustworthy only when a region has
  exactly ONE row. Comparing a fabricated fov SET against the filename-derived set does not
  rescue it — set equality is permutation-invariant, so a reordered file passes with every
  position wrong. Multi-row regions are therefore omitted, not guessed.
* **Nothing here raises.** A missing, malformed or unreadable table degrades to ``{}`` plus a
  warning. Positions have no consumer yet; a bad sidecar must never brick the MIP pipeline.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Optional

FILENAME = "coordinates.csv"

# XY is recorded once per FOV and repeats across z levels. Real drift is 0; what shows up is
# float round-tripping (observed: 3930.75 vs 3930.7499999999995). 0.5 µm is sub-pixel for every
# Squid objective (20x ≈ 0.325 µm/px), so it separates formatting noise from a real disagreement.
XY_TOLERANCE_UM = 0.5

# Column spelling -> multiplier into micrometres. Squid writes "x (mm)"; the paren-space variants
# and a µm spelling are accepted so a hand-edited or re-exported table still parses.
_MM = 1000.0
_X_COLUMNS = {"x (mm)": _MM, "x(mm)": _MM, "x_mm": _MM, "x (um)": 1.0, "x(um)": 1.0, "x_um": 1.0}
_Y_COLUMNS = {"y (mm)": _MM, "y(mm)": _MM, "y_mm": _MM, "y (um)": 1.0, "y(um)": 1.0, "y_um": 1.0}
_Z_COLUMNS = {"z (mm)": _MM, "z(mm)": _MM, "z_mm": _MM, "z (um)": 1.0, "z(um)": 1.0, "z_um": 1.0}


def _normalize(name: str) -> str:
    """Header cell -> comparable key: strip the UTF-8 BOM, surrounding whitespace, and case."""
    return (name or "").replace("﻿", "").strip().lower()


def _find_column(fields: list, table: dict):
    """First header field present in *table*; returns (raw_field_name, scale_to_um) or (None, None)."""
    for raw in fields:
        scale = table.get(_normalize(raw))
        if scale is not None:
            return raw, scale
    return None, None


def _to_float(cell) -> Optional[float]:
    """Parse one numeric cell. Blank/absent/unparseable -> None (the root schema leaves z empty)."""
    if cell is None:
        return None
    text = str(cell).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _candidate_files(root: Path, time_folder: Optional[Path]) -> list:
    """coordinates.csv paths in precedence order: timepoint folder first, acquisition root second.

    The timepoint copy wins because it is the only one carrying explicit fov labels and a real
    z; the root copy is the planned grid and leaves z empty in 3 of 4 real datasets. (The XY
    difference between them is only ~1 pixel, so accuracy is not the reason.)
    """
    candidates = []
    if time_folder is not None:
        candidates.append(Path(time_folder) / FILENAME)
    if time_folder is None:
        default_t0 = root / "0"
        if default_t0.is_dir():
            candidates.append(default_t0 / FILENAME)
    candidates.append(root / FILENAME)

    seen, ordered = set(), []
    for path in candidates:
        resolved = str(path)
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(path)
    return [p for p in ordered if p.is_file()]


def _read_rows(path: Path):
    """(rows, normalized_fieldnames, raw_fieldnames). Tolerates BOM and CRLF."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        raw_fields = list(reader.fieldnames or [])
        rows = list(reader)
    return rows, [_normalize(f) for f in raw_fields], raw_fields


def _region_of(row: dict, region_field: str) -> Optional[str]:
    value = row.get(region_field)
    text = "" if value is None else str(value).strip()
    return text or None


def _parse_labelled(rows, raw_fields, fov_field, z_level_field, x_col, y_col, z_col, region_field):
    """7-column schema: one row per (fov, z_level). Collapse to one entry per FOV.

    XY repeats identically across z levels (verified: 550 rows / 55 FOVs), so XY is taken from
    any row and cross-checked with a sub-pixel tolerance. z genuinely varies per plane, so the
    LOWEST z_level is used as the FOV's reference height — an arbitrary row pick would make the
    value's meaning undefined.
    """
    x_field, x_scale = x_col
    y_field, y_scale = y_col
    z_field, z_scale = z_col

    best: dict = {}          # (region, fov) -> (z_level, x_um, y_um, z_um)
    xy_conflicts: set = set()
    skipped = 0

    for row in rows:
        region = _region_of(row, region_field)
        fov_raw = _to_float(row.get(fov_field))
        x_um = _to_float(row.get(x_field))
        y_um = _to_float(row.get(y_field))
        if region is None or fov_raw is None or x_um is None or y_um is None:
            skipped += 1
            continue
        x_um *= x_scale
        y_um *= y_scale

        z_level = _to_float(row.get(z_level_field)) if z_level_field else None
        z_level = 0.0 if z_level is None else z_level

        z_um = None
        if z_field is not None:
            raw_z = _to_float(row.get(z_field))
            z_um = None if raw_z is None else raw_z * z_scale

        key = (region, int(fov_raw))
        previous = best.get(key)
        if previous is None:
            best[key] = (z_level, x_um, y_um, z_um)
            continue

        prev_level, prev_x, prev_y, prev_z = previous
        if abs(prev_x - x_um) > XY_TOLERANCE_UM or abs(prev_y - y_um) > XY_TOLERANCE_UM:
            xy_conflicts.add(key)
        if z_level < prev_level:
            # keep the already-validated XY; only the reference height moves down
            best[key] = (z_level, prev_x, prev_y, z_um)

    if xy_conflicts:
        warnings.warn(
            f"{len(xy_conflicts)} FOV(s) in {FILENAME} have XY varying by more than "
            f"{XY_TOLERANCE_UM} um across z levels (e.g. {sorted(xy_conflicts)[:3]}); XY is "
            "expected constant within a z-stack. Using the first row's XY for each."
        )
    if skipped:
        warnings.warn(f"Skipped {skipped} unparseable row(s) in {FILENAME}.")

    return {key: (x, y, z) for key, (_level, x, y, z) in best.items()}


def _parse_unlabelled(rows, x_col, y_col, z_col, region_field):
    """4-column schema: no fov column, so a fov index can only come from row position.

    Row position is only trustworthy when a region has exactly ONE row (then it is fov 0 and
    there is nothing to order). With several rows per region the file's order may or may not
    match the filename fov tokens, and nothing in the file can tell us which — so those regions
    are omitted with a warning rather than silently mis-assigned.
    """
    x_field, x_scale = x_col
    y_field, y_scale = y_col
    z_field, z_scale = z_col

    by_region: dict = {}
    skipped = 0
    for row in rows:
        region = _region_of(row, region_field)
        x_um = _to_float(row.get(x_field))
        y_um = _to_float(row.get(y_field))
        if region is None or x_um is None or y_um is None:
            skipped += 1
            continue
        z_um = None
        if z_field is not None:
            raw_z = _to_float(row.get(z_field))
            z_um = None if raw_z is None else raw_z * z_scale
        by_region.setdefault(region, []).append((x_um * x_scale, y_um * y_scale, z_um))

    positions: dict = {}
    ambiguous = []
    for region, entries in by_region.items():
        if len(entries) == 1:
            positions[(region, 0)] = entries[0]
        else:
            ambiguous.append(region)

    if ambiguous:
        warnings.warn(
            f"{FILENAME} has no 'fov' column and {len(ambiguous)} region(s) with multiple rows "
            f"({sorted(ambiguous)[:5]}{'...' if len(ambiguous) > 5 else ''}). A fov index can "
            "only be inferred from row order, which nothing in the file verifies, so these "
            "regions are omitted rather than guessed. Use the timepoint coordinates.csv "
            "(region,fov,z_level,...) for per-FOV positions in multi-FOV regions."
        )
    if skipped:
        warnings.warn(f"Skipped {skipped} unparseable row(s) in {FILENAME}.")

    return positions


def load_fov_positions_um(root, time_folder=None, fovs_per_region=None) -> dict:
    """Return ``{(region, fov): (x_um, y_um, z_um | None)}`` for one acquisition.

    All values are micrometres, matching the ``dz_um`` / ``pixel_size_um`` the reader already
    publishes. ``z`` is ``None`` when the table leaves it blank (the root schema does, in 3 of
    4 real datasets).

    Positions describe timepoint 0 only. Each timepoint folder carries its own table with its
    own post-autofocus values; multi-timepoint positions are deferred.

    Never raises: a missing, malformed or unreadable table yields ``{}`` and a warning.

    Parameters
    ----------
    root : path
        Acquisition folder.
    time_folder : path, optional
        Timepoint folder to prefer. Defaults to ``<root>/0`` when it exists.
    fovs_per_region : dict, optional
        ``{region: [fov, ...]}`` derived from FILENAMES (the ground truth established by
        IMA-189). Used only to warn about entries the filenames do not corroborate — it can
        never authorise a fabricated ordering, so it does not gate parsing.
    """
    root = Path(root)
    files = _candidate_files(root, time_folder)
    if not files:
        return {}

    path = files[0]
    try:
        rows, fields, raw_fields = _read_rows(path)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        warnings.warn(f"Could not read {path}: {exc}. Continuing without stage positions.")
        return {}

    if not raw_fields:
        warnings.warn(f"{path} has no header row. Continuing without stage positions.")
        return {}

    region_field = next((r for r in raw_fields if _normalize(r) == "region"), None)
    x_col = _find_column(raw_fields, _X_COLUMNS)
    y_col = _find_column(raw_fields, _Y_COLUMNS)
    z_col = _find_column(raw_fields, _Z_COLUMNS)
    if region_field is None or x_col[0] is None or y_col[0] is None:
        warnings.warn(
            f"{path} header {raw_fields} is not a recognised Squid coordinates table "
            "(needs 'region' plus x/y columns). Continuing without stage positions."
        )
        return {}

    try:
        if "fov" in fields:
            fov_field = next(r for r in raw_fields if _normalize(r) == "fov")
            z_level_field = next((r for r in raw_fields if _normalize(r) == "z_level"), None)
            positions = _parse_labelled(
                rows, raw_fields, fov_field, z_level_field, x_col, y_col, z_col, region_field
            )
        else:
            positions = _parse_unlabelled(rows, x_col, y_col, z_col, region_field)
    except Exception as exc:                                  # never brick the reader
        warnings.warn(f"Failed to parse {path}: {exc!r}. Continuing without stage positions.")
        return {}

    if fovs_per_region:
        unknown = [key for key in positions if key[1] not in set(fovs_per_region.get(key[0], ()))]
        if unknown:
            warnings.warn(
                f"{len(unknown)} entry(ies) in {path.name} name a (region, fov) with no matching "
                f"image files (e.g. {sorted(unknown)[:3]}). Filenames are the ground truth; these "
                "positions are kept but may describe planned-but-not-acquired FOVs."
            )

    return positions
