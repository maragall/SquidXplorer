#!/usr/bin/env python3
"""Measure the real per-well FOV grid and tile overlap of Squid acquisitions (IMA-211 T2).

Why this exists: every stitching decision downstream depends on a number nobody had measured.
If an acquisition runs 0% overlap, phase correlation has nothing to correlate on and **only
nominal placement can work** — so this gates whether registration code is worth writing at
all. It is cheap to run and it answers the question with data instead of assumption.

It also demonstrates the IMA-211 geometry correction in the most direct way available:
it reads BOTH coordinate files and reports where they disagree.

    coordinates.csv                                   region,x (mm),y (mm),z (mm)
        the PLANNED grid. No `fov` column, so a FOV index can only be inferred from
        row order — the "silent wrong tile" hazard docs/ima-189-eng-review.md refused.

    original_coordinates/original_coordinates_{t}.csv region,fov,z_level,x (mm),y (mm),z (um),time
        the ACTUAL stage positions, WITH an explicit `fov` key and an explicit `z_level`.
        Cell assignment becomes arithmetic instead of inference.

Usage::

    python scripts/measure_overlap.py PATH [PATH ...]
    python scripts/measure_overlap.py ~/Downloads/20x_scan_* --json

Reports per acquisition: which coordinate files exist, the FOV grid per region, the step in
mm, the frame size in px, the derived overlap fraction, and any disagreement between the two
coordinate sources. Never writes anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# Imported lazily inside the frame-size probe so the script still reports geometry on a
# machine without tifffile.


def _read_csv(path: Path) -> list:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _col(row: dict, *names):
    """Fetch the first present column, tolerating the ' (mm)' / '(mm)' spacing drift."""
    for n in names:
        if n in row:
            return row[n]
    norm = {k.replace(" ", "").lower(): v for k, v in row.items()}
    for n in names:
        k = n.replace(" ", "").lower()
        if k in norm:
            return norm[k]
    raise KeyError(f"none of {names} in {list(row)}")


def _modal_step(values) -> float:
    """Smallest consistent positive spacing among sorted unique coordinates.

    The modal positive delta, not the mean: a partial or non-rectangular scan leaves gaps,
    and a mean would silently blend a real step with a skipped one.
    """
    uniq = sorted({round(float(v), 4) for v in values})
    deltas = [round(b - a, 4) for a, b in zip(uniq, uniq[1:]) if b - a > 1e-6]
    if not deltas:
        return 0.0
    counts = defaultdict(int)
    for d in deltas:
        counts[d] += 1
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _frame_shape(root: Path):
    """Shape of one real plane, so overlap is derived from pixels rather than assumed."""
    try:
        import tifffile
    except ImportError:
        return None
    for tp in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        for f in sorted(tp.iterdir()):
            if f.suffix.lower() in (".tif", ".tiff"):
                try:
                    return tuple(tifffile.imread(f).shape[:2])
                except Exception:
                    return None
    return None


def _pixel_size_um(root: Path):
    """Object-space pixel size in µm, and where it came from.

    The two metadata generations are NOT interchangeable and conflating them is a 20x error:

      acquisition.yaml       objective.pixel_size_um    ALREADY object-space and binning-aware
                                                        (per squidmip/_acquisition.py:3-5)
      acquisition params.json sensor_pixel_size_um      SENSOR pitch — must be divided by
                                                        objective.magnification

    Reading the JSON's sensor pitch as if it were object-space reports a ~95% overlap on a
    scan that actually overlaps ~9%, which would make every downstream stitching decision
    wrong in the same direction.
    """
    y = root / "acquisition.yaml"
    if y.exists():
        try:
            import yaml

            d = yaml.safe_load(y.read_text()) or {}
            v = (d.get("objective") or {}).get("pixel_size_um")
            if v:
                return float(v), "acquisition.yaml (object-space)"
        except Exception:
            pass
    j = root / "acquisition parameters.json"
    if j.exists():
        try:
            d = json.loads(j.read_text())
            if d.get("pixel_size_um"):
                return float(d["pixel_size_um"]), "json pixel_size_um"
            sensor = d.get("sensor_pixel_size_um")
            mag = (d.get("objective") or {}).get("magnification")
            if sensor and mag:
                return float(sensor) / float(mag), f"json sensor/{mag:g}x"
            if sensor:
                return None, "json has sensor pitch but no magnification — cannot convert"
        except Exception:
            pass
    return None, "not found"


def survey(root: Path) -> dict:
    """Measure one acquisition. Returns a plain dict so --json is a straight dump."""
    out = {"path": str(root), "regions": {}, "notes": []}

    coords = root / "coordinates.csv"
    orig_dir = root / "original_coordinates"
    orig = sorted(orig_dir.glob("original_coordinates_*.csv")) if orig_dir.is_dir() else []

    out["has_coordinates_csv"] = coords.exists()
    out["has_original_coordinates"] = bool(orig)

    if not orig:
        out["notes"].append(
            "NO original_coordinates/ — per-FOV geometry would have to fall back to row-order "
            "inference on coordinates.csv (the silent-wrong-tile hazard). This is the fallback "
            "case IMA-211 flagged as undecided."
        )

    frame = _frame_shape(root)
    px_um, px_src = _pixel_size_um(root)
    out["frame_shape"] = frame
    out["pixel_size_um"] = px_um
    out["pixel_size_source"] = px_src

    rows = _read_csv(orig[0]) if orig else (_read_csv(coords) if coords.exists() else [])
    if not rows:
        out["notes"].append("no readable coordinate rows")
        return out

    source = "original_coordinates" if orig else "coordinates.csv"
    out["geometry_source"] = source
    has_fov = "fov" in rows[0]
    out["has_explicit_fov_column"] = has_fov

    # z_level is explicit in original_coordinates, so multi-z rows are FILTERED rather than
    # guessed at — this is exactly the open worry IMA-187 recorded about row-count checks.
    if "z_level" in rows[0]:
        z0 = {r["z_level"] for r in rows}
        if len(z0) > 1:
            out["notes"].append(
                f"multi-z coordinate file ({len(z0)} z_levels) — filtered to z_level=0; "
                "a row-count cross-check against fovs_per_region would have been wrong here"
            )
        rows = [r for r in rows if r["z_level"] == sorted(z0)[0]]

    by_region = defaultdict(list)
    for r in rows:
        by_region[r["region"]].append(r)

    for region, rs in sorted(by_region.items()):
        xs = [float(_col(r, "x (mm)", "x(mm)")) for r in rs]
        ys = [float(_col(r, "y (mm)", "y(mm)")) for r in rs]
        nx = len({round(v, 4) for v in xs})
        ny = len({round(v, 4) for v in ys})
        step_x, step_y = _modal_step(xs), _modal_step(ys)

        entry = {
            "n_fov": len(rs),
            "grid": [ny, nx],
            "step_mm": [step_y, step_x],
            "rectangular": ny * nx == len(rs),
        }
        if frame and px_um:
            span_x_mm = frame[1] * px_um / 1000.0
            span_y_mm = frame[0] * px_um / 1000.0
            entry["overlap_frac"] = [
                round(1.0 - step_y / span_y_mm, 4) if span_y_mm and step_y else None,
                round(1.0 - step_x / span_x_mm, 4) if span_x_mm and step_x else None,
            ]
        else:
            entry["overlap_frac"] = None
            entry["overlap_note"] = "needs both frame_shape and pixel_size_um"
        if not entry["rectangular"]:
            out["notes"].append(
                f"region {region}: {len(rs)} FOVs do not fill a {ny}x{nx} grid — "
                "non-rectangular or partial scan; a strict lattice gate would refuse it"
            )
        out["regions"][region] = entry

    # The correction, demonstrated: does the planned grid agree with where the stage went?
    if orig and coords.exists():
        planned = _read_csv(coords)
        p_by_region = defaultdict(int)
        for r in planned:
            p_by_region[r["region"]] += 1
        for region, entry in out["regions"].items():
            if p_by_region.get(region, 0) != entry["n_fov"]:
                out["notes"].append(
                    f"region {region}: coordinates.csv has {p_by_region.get(region, 0)} rows "
                    f"but original_coordinates has {entry['n_fov']} FOVs — row-order mapping "
                    "between them would MISALIGN tiles"
                )

    # The JSON's declared grid, if any, versus what actually happened.
    j = root / "acquisition parameters.json"
    if j.exists():
        try:
            d = json.loads(j.read_text())
            decl = (d.get("Nx"), d.get("Ny"))
            if decl != (None, None):
                out["declared_grid_json"] = list(decl)
                real = next(iter(out["regions"].values()))["grid"] if out["regions"] else None
                if real and [decl[1], decl[0]] != real:
                    out["notes"].append(
                        f"acquisition parameters.json declares Nx={decl[0]},Ny={decl[1]} but the "
                        f"real grid is {real[0]}x{real[1]} — DO NOT derive grid geometry from "
                        "that file; derive spacing from the coordinates themselves"
                    )
        except Exception:
            pass

    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", type=Path, help="acquisition folder(s)")
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    args = ap.parse_args(argv)

    results = []
    for p in args.paths:
        if not p.is_dir():
            print(f"skip (not a directory): {p}", file=sys.stderr)
            continue
        try:
            results.append(survey(p))
        except Exception as exc:  # a bad dataset must not hide the good ones
            print(f"ERROR surveying {p}: {exc}", file=sys.stderr)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    for r in results:
        print(f"\n=== {r['path']}")
        print(
            f"  coordinate files : coordinates.csv={r['has_coordinates_csv']}  "
            f"original_coordinates={r['has_original_coordinates']}"
            + (f"  (explicit fov column: {r.get('has_explicit_fov_column')})" if r.get("geometry_source") else "")
        )
        print(f"  frame            : {r['frame_shape']}   pixel_size_um={r['pixel_size_um']} [{r.get('pixel_size_source')}]")
        for region, e in list(r["regions"].items())[:6]:
            ov = e.get("overlap_frac")
            ov_s = (
                f"overlap y={ov[0]:.1%} x={ov[1]:.1%}"
                if ov and ov[0] is not None and ov[1] is not None
                else f"overlap unknown ({e.get('overlap_note', '')})"
            )
            print(
                f"  region {region:<8} {e['n_fov']:>4} FOV  grid {e['grid'][0]}x{e['grid'][1]}  "
                f"step {e['step_mm'][0]:.4f}/{e['step_mm'][1]:.4f} mm  {ov_s}"
            )
        if len(r["regions"]) > 6:
            print(f"  ... and {len(r['regions']) - 6} more regions")
        for n in r["notes"]:
            print(f"  ! {n}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
