"""Check a plate against ``docs/plate-contract.md``, and say which half it broke.

Gap 5 of the three-viewers review. The shape is record-zstack-viewer's, and the shape IS the
point:

    ERRORS   are violations of the STABLE section. The store is not the thing it claims to be, and
             a reader that proceeds renders something plausible and wrong.
    WARNINGS are absences in the OPTIONAL section. Every one of them has a named fallback that
             already exists in code, so the store is READABLE and merely reads for less. A missing
             ``translation`` is a warning; a missing level 0 is an error. Collapsing those two into
             one severity is how a validator becomes noise that nobody runs.

This module was ``tests/ngff_check.py``, which validated our own output against OME's official
``ome-zarr-models`` pydantic schema and could therefore only ever be pointed at a plate we had
just written. It is promoted out of ``tests/`` so that a USER can point it at a plate they were
handed:

    python -m squidmip.contract.validate /path/to/plate.ome.zarr

``ome-zarr-models`` stays a ``[test]`` extra rather than becoming a runtime dependency. When it is
absent the schema pass is skipped and the structural checks below still run, which is the whole
reason they are written out longhand here instead of delegating: the structural half is the half
that encodes OUR contract, and OME's schema does not know about it. The skip is reported as a
warning rather than swallowed, because "validated" and "half-validated" must not look alike.

Everything checked here is stated in ``docs/plate-contract.md``. If the two ever disagree, the
document is the contract and this file is the bug.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from squidmip.contract.paths import field_path
from squidmip.contract.version import (
    PLATE_CONTRACT_VERSION,
    PlateContractError,
    compare_contract_version,
    read_contract_version,
)

#: The stable axis order. Squid's canonical order and the one every array in this package is
#: written and read in. A store that reorders these is not readable by this build at all.
STABLE_AXES = ["t", "c", "z", "y", "x"]

#: Written by ``_output`` from the first byte of a plate write to the last. Its presence means the
#: run did not finish, so the store may be missing wells that the plate metadata promises. IMPORTED,
#: never re-spelled: two copies of a filename are two different files the moment one is edited.
from squidmip._output import INCOMPLETE_MARKER as _INCOMPLETE_MARKER

#: UDUNITS-2 length units the reader can convert to micrometres (``reader._UNIT_TO_UM``). A unit
#: outside this set cannot be converted, so a physical value taken from the store would reach a
#: ``_um`` key in the wrong unit, which is the 1000x class of bug.
_KNOWN_UNITS = {
    "angstrom", "nanometer", "micrometer", "micron", "millimeter", "centimeter", "meter",
}


@dataclass
class ValidationReport:
    """What a plate got wrong, split by whether it stops a reader or only narrows one."""

    path: str
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    contract_version: Optional[str] = None

    @property
    def ok(self) -> bool:
        """True when nothing in the STABLE section is broken. Warnings do not make it False."""
        return not self.errors

    def summary(self) -> str:
        head = (f"{self.path}: {'OK' if self.ok else 'INVALID'} "
                f"({len(self.errors)} error(s), {len(self.warnings)} warning(s)), "
                f"declared contract {self.contract_version or 'none'}")
        lines = [head]
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  warning {w}" for w in self.warnings]
        return "\n".join(lines)


def _ome_attrs(group_dir: Path) -> dict:
    """The OME payload of a group, via the reader's own normaliser. Imported late on purpose.

    ``reader`` imports ``contract.version`` (it compares the stamp), so importing it at module
    scope here would close a cycle. It is also the heavier module of the two, and the writer path
    pulls ``contract`` on every plate write while never needing this function.
    """
    from squidmip.reader import _group_attrs

    return _group_attrs(group_dir)


def _array_shape(array_dir: Path) -> Optional[list]:
    """Shape of a zarr array from its own metadata, v3 (``zarr.json``) or v2 (``.zarray``)."""
    v3 = array_dir / "zarr.json"
    if v3.exists():
        try:
            return list(json.loads(v3.read_text())["shape"])
        except (ValueError, OSError, KeyError):
            return None
    v2 = array_dir / ".zarray"
    if v2.exists():
        try:
            return list(json.loads(v2.read_text())["shape"])
        except (ValueError, OSError, KeyError):
            return None
    return None


def _check_field(base: Path, wellpath: str, fov: str, report: ValidationReport, seen: dict) -> None:
    """One field image group: stable layout to ``errors``, optional content counted in ``seen``."""
    group = Path(field_path(base, wellpath, fov))
    where = f"{wellpath}/{fov}"
    if not group.is_dir():
        report.errors.append(f"{where}: the well lists this field but the directory is missing")
        return
    attrs = _ome_attrs(group)
    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        report.errors.append(f"{where}: no 'multiscales' on the image group")
        return
    ms = multiscales[0]

    axes = [str(a.get("name", "")).lower() for a in (ms.get("axes") or [])]
    if axes != STABLE_AXES:
        report.errors.append(
            f"{where}: axes are {axes or 'absent'}, not the stable TCZYX order {STABLE_AXES}")

    datasets = ms.get("datasets") or []
    if not datasets:
        report.errors.append(f"{where}: 'multiscales[0].datasets' is empty, so there is no level 0")
        return

    shapes = []
    for i, dataset in enumerate(datasets):
        level = dataset.get("path")
        if level is None:
            report.errors.append(f"{where}: datasets[{i}] has no 'path'")
            continue
        array_dir = Path(field_path(base, wellpath, fov, level))
        shape = _array_shape(array_dir)
        if shape is None:
            report.errors.append(
                f"{where}: datasets[{i}].path = {level!r} is declared but is not a zarr array "
                "on disk")
            continue
        shapes.append((i, shape))

    if shapes:
        (_, first) = shapes[0]
        # The t axis. The format carries it and the writer writes every timepoint; most of the
        # READ side collapses to t=0. Announcing that is the whole reason this is checked here:
        # a multi-timepoint plate currently renders as its first frame with no error anywhere,
        # and every fixture in the suite is Nt=1, so nothing else catches it.
        if len(first) == 5 and first[0] > 1:
            seen["n_t"] = max(seen["n_t"], int(first[0]))
        for i, shape in shapes[1:]:
            if shape[-2] > first[-2] or shape[-1] > first[-1]:
                report.errors.append(
                    f"{where}: level {i} is LARGER than level 0 in Y/X ({shape[-2:]} vs "
                    f"{first[-2:]}), so datasets are not ordered highest resolution first. A MIP "
                    "computed from this store would come from a coarser level.")
                break

    for axis in (ms.get("axes") or []):
        if str(axis.get("name", "")).lower() not in ("z", "y", "x"):
            continue
        unit = axis.get("unit")
        if unit is None:
            seen["no_unit"] += 1
        elif str(unit).strip().lower().rstrip("s") not in _KNOWN_UNITS:
            report.errors.append(
                f"{where}: axis {axis.get('name')!r} declares unit {unit!r}, which cannot be "
                "converted to micrometres. Every physical value in this contract is micrometres.")

    if len(datasets) == 1:
        seen["single_level"] += 1
    if not any(t.get("type") == "translation"
               for t in (datasets[0].get("coordinateTransformations") or [])):
        seen["no_translation"] += 1
    if not attrs.get("omero"):
        seen["no_omero"] += 1


def _schema_errors(plate_dir: Path, plate: dict) -> Optional[list]:
    """OME's own pydantic verdict, or ``None`` when ``ome-zarr-models`` is not installed.

    This is the part that is worth borrowing spec code for: is our metadata JSON actually valid
    against the published v0.5 schema? The writer stays our lean tensorstore code; the schema
    stays OME's.
    """
    try:
        from ome_zarr_models.v05 import image as I
        from ome_zarr_models.v05 import plate as P
        from ome_zarr_models.v05 import well as W
    except ImportError:
        return None

    errors = []

    def _ome(group_dir: Path) -> dict:
        return json.loads((group_dir / "zarr.json").read_text())["attributes"]["ome"]

    try:
        P.PlateBase.model_validate(plate)
    except Exception as e:                       # pydantic ValidationError, or a malformed payload
        errors.append(f"plate group fails the OME v0.5 schema: {e}")
    for well in plate.get("wells") or []:
        rel = str(well.get("path", "")).strip("/")
        well_dir = plate_dir / rel
        if not (well_dir / "zarr.json").exists():
            continue                             # already reported as a structural error
        try:
            well_ome = _ome(well_dir)
            W.WellMeta.model_validate(well_ome["well"])
        except Exception as e:
            errors.append(f"{rel}: well group fails the OME v0.5 schema: {e}")
            continue
        for image in well_ome["well"]["images"]:
            image_dir = well_dir / str(image.get("path", ""))
            if not (image_dir / "zarr.json").exists():
                continue
            try:
                I.ImageAttrs.model_validate(_ome(image_dir))
            except Exception as e:
                errors.append(f"{rel}/{image.get('path')}: image group fails the OME v0.5 "
                              f"schema: {e}")
    return errors


def validate_plate(path) -> ValidationReport:
    """Validate a written plate. Stable violations become errors, optional absences warnings."""
    plate_dir = Path(path)
    report = ValidationReport(path=str(plate_dir))

    if not plate_dir.is_dir():
        report.errors.append("no such directory")
        return report

    declared = read_contract_version(plate_dir)
    report.contract_version = declared
    try:
        verdict = compare_contract_version(declared)
    except PlateContractError as e:
        # The validator REPORTS what the reader RAISES. Same policy, one place (version.py), two
        # deliveries: a reader must stop, a validator must finish and list everything it found.
        report.errors.append(str(e))
        verdict = "refused"
    if verdict == "absent":
        report.warnings.append(
            "this store declares no plate contract version, so nothing here was checked against a "
            f"declared one. Plates written before {PLATE_CONTRACT_VERSION} landed, and every "
            "third-party NGFF store, are unstamped; this is expected for them.")
    elif verdict == "minor-ahead":
        report.warnings.append(
            f"declares contract {declared}, newer than this build's {PLATE_CONTRACT_VERSION}. The "
            "stable layout is unchanged, but optional content added since is not checked here.")

    if (plate_dir / _INCOMPLETE_MARKER).exists():
        report.warnings.append(
            f"{_INCOMPLETE_MARKER} is present: the write that produced this store did not finish, "
            "so wells the plate metadata promises may be absent.")

    attrs = _ome_attrs(plate_dir)
    plate = attrs.get("plate")
    if not isinstance(plate, dict):
        report.errors.append(
            "not an HCS plate: the group carries no 'plate' metadata (rows / columns / wells)")
        return report

    wells = plate.get("wells") or []
    if not wells:
        report.errors.append("the plate lists no wells")

    seen = {"no_translation": 0, "no_omero": 0, "single_level": 0, "no_unit": 0, "n_t": 1}
    n_fields = 0
    for well in wells:
        rel = str(well.get("path", "")).strip("/")
        if not rel:
            report.errors.append(f"a plate well entry has no 'path': {well!r}")
            continue
        well_dir = plate_dir / rel
        if not well_dir.is_dir():
            report.errors.append(f"{rel}: the plate lists this well but the directory is missing")
            continue
        images = (_ome_attrs(well_dir).get("well") or {}).get("images") or []
        if not images:
            report.errors.append(f"{rel}: the well group lists no images (fields)")
            continue
        for image in images:
            name = str(image.get("path", ""))
            if not name:
                report.errors.append(f"{rel}: a well image entry has no 'path'")
                continue
            n_fields += 1
            _check_field(plate_dir, rel, name, report, seen)

    if seen["no_translation"]:
        has_csv = (plate_dir.parent / "coordinates.csv").exists()
        report.warnings.append(
            f"{seen['no_translation']}/{n_fields} field(s) carry no dataset 'translation'. "
            + ("The sibling coordinates.csv is the documented fallback and it is present."
               if has_csv else
               "The documented fallback is a sibling coordinates.csv and there is none, so FOV "
               "positions are unavailable and consumers degrade to single-tile rendering."))
    if seen["no_omero"]:
        report.warnings.append(
            f"{seen['no_omero']}/{n_fields} field(s) carry no 'omero' block, so channel colours "
            "and windows fall back to auto-contrast.")
    if seen["single_level"]:
        report.warnings.append(
            f"{seen['single_level']}/{n_fields} field(s) have only level '0' and no pyramid. That "
            "is legal (small fields are written single-level on purpose); navigation falls back "
            "to level '0' and pays full resolution for a thumbnail.")
    if seen["n_t"] > 1:
        report.warnings.append(
            f"this plate has {seen['n_t']} timepoints. The store is correct and carries all of "
            "them, but the plate overview and the loupe read t=0 unconditionally "
            "(_viewer._ComputedPlateWorker._read, _ZarrLoupeSource.coarse), so they show the "
            "FIRST timepoint and say nothing about the rest. The montage and the tile source do "
            "take a timepoint. See the 'Time' section of docs/plate-contract.md.")
    if seen["no_unit"]:
        report.warnings.append(
            f"{seen['no_unit']} spatial axis/axes declare no unit. NGFF makes the unit a SHOULD, "
            "so this is legal; the reader assumes micrometres for them.")

    schema = _schema_errors(plate_dir, plate)
    if schema is None:
        report.warnings.append(
            "ome-zarr-models is not installed, so the official OME v0.5 schema pass was SKIPPED "
            "and only the structural checks above ran. Install the [test] extra for the full "
            "check.")
    else:
        report.errors.extend(schema)

    return report


def assert_valid_ngff_plate(plate_dir) -> None:
    """Raise unless *plate_dir* satisfies the stable contract and the OME v0.5 schema.

    The assertion form ``tests/`` has always used. It is deliberately strict about ERRORS only:
    a test plate legitimately has no coordinates.csv and legitimately writes single-level fields,
    and a validator that failed on those would be a validator every test had to suppress.
    """
    report = validate_plate(plate_dir)
    if not report.ok:
        raise AssertionError(report.summary())


def main(argv=None) -> int:
    """``python -m squidmip.contract.validate <plate.ome.zarr> [...]``. Exit 1 if any is invalid."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__.strip().splitlines()[0])
        print("usage: python -m squidmip.contract.validate <plate.ome.zarr> [more...]")
        return 2
    bad = 0
    for arg in args:
        report = validate_plate(arg)
        print(report.summary())
        bad += 0 if report.ok else 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
