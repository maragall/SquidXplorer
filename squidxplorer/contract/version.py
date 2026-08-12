"""The plate contract's version: stamped by the writer, compared by the reader.

Policy: absent -> proceed; same major -> proceed; minor ahead -> warn; major mismatch or an
unparseable stamp -> refuse. The prose contract is ``docs/plate-contract.md``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

#: The contract this build writes and understands. MAJOR bumps when the stable section of
#: ``docs/plate-contract.md`` changes; MINOR when the optional section gains an entry.
PLATE_CONTRACT_VERSION = "1.0"

#: The attributes key the stamp lives under, deliberately outside OME's namespace.
CONTRACT_NAMESPACE = "squidxplorer"
CONTRACT_KEY = "plate_contract_version"

_ZARR_V3_META = "zarr.json"
_ZARR_V2_ATTRS = ".zattrs"


class PlateContractError(Exception):
    """A store declares a plate contract this build cannot read correctly."""


def contract_stamp() -> dict:
    """The ``attributes``-level payload the writer stamps onto the plate group."""
    return {CONTRACT_NAMESPACE: {CONTRACT_KEY: PLATE_CONTRACT_VERSION}}


def read_contract_version(plate_dir) -> Optional[str]:
    """The contract version a store declares, or ``None`` when it declares none."""
    plate_dir = Path(plate_dir)
    for name in (_ZARR_V2_ATTRS, _ZARR_V3_META):
        meta = plate_dir / name
        if not meta.exists():
            continue
        try:
            doc = json.loads(meta.read_text() or "{}")
        except (ValueError, OSError):
            return None
        attrs = doc.get("attributes", doc) if name == _ZARR_V3_META else doc
        block = attrs.get(CONTRACT_NAMESPACE) if isinstance(attrs, dict) else None
        if isinstance(block, dict) and block.get(CONTRACT_KEY) is not None:
            return str(block[CONTRACT_KEY])
    return None


def _split(version: str) -> tuple:
    """``"1.2"`` -> ``(1, 2)``. Raises ``ValueError`` on anything that is not MAJOR.MINOR."""
    parts = str(version).strip().split(".")
    if len(parts) != 2:
        raise ValueError(f"{version!r} is not MAJOR.MINOR")
    return (int(parts[0]), int(parts[1]))


def compare_contract_version(declared: Optional[str], supported: str = PLATE_CONTRACT_VERSION):
    """Return ``"absent"``, ``"ok"`` or ``"minor-ahead"``; raise on a major mismatch or bad stamp."""
    if declared is None:
        return "absent"
    want_major, want_minor = _split(supported)
    try:
        got_major, got_minor = _split(declared)
    except ValueError:
        raise PlateContractError(
            f"this store declares plate contract version {declared!r}, which is not a MAJOR.MINOR "
            f"version this build can compare against {supported!r}. Refusing to read it: a stamp "
            "that cannot be compared is worse than no stamp, because something deliberately made "
            "a promise and we cannot tell which one. See docs/plate-contract.md."
        ) from None
    if got_major != want_major:
        raise PlateContractError(
            f"this store declares plate contract version {declared}, but this build of SquidXplorer "
            f"reads {supported}. A MAJOR difference means a stable guarantee moved (the "
            f"{{row}}/{{col}}/{{fov}}/{{level}} hierarchy, TCZYX axis order, level 0 being full "
            "resolution, or micrometre units). Reading it with these assumptions would produce a "
            "plate that looks plausible and is wrong, so it is refused instead. Upgrade or "
            "downgrade SquidXplorer to a build whose major matches, or re-run the projection. See "
            "docs/plate-contract.md."
        )
    if got_minor > want_minor:
        return "minor-ahead"
    return "ok"


def check_plate_contract(plate_dir, *, warn: bool = True) -> Optional[str]:
    """Read a store's stamp and apply the policy; returns the declared version or ``None``."""
    declared = read_contract_version(plate_dir)
    verdict = compare_contract_version(declared)
    if verdict == "minor-ahead" and warn:
        warnings.warn(
            f"{plate_dir!s} declares plate contract {declared}, newer than this build's "
            f"{PLATE_CONTRACT_VERSION}. The stable layout is unchanged (that is what a minor bump "
            "promises), so it is read normally, but any optional content added since is ignored.",
            stacklevel=2,
        )
    return declared
