"""The plate contract's version: stamped by the writer, COMPARED by the reader.

Gap 5 of the three-viewers review (Hongquan, 2026-07-28), pulled to full v1 scope by Julio on
2026-07-28. The written form of the contract is ``docs/plate-contract.md``; this module is the
machine-checkable half of it.

WHY THIS EXISTS, and it is not a hypothetical. ``_output._NGFF_VERSION`` is written at four sites
and read at ZERO. Version discrimination in the reader is structural sniffing (``_group_attrs``
tests whether a ``.zattrs`` or a ``zarr.json`` exists), which works today and degrades to a SILENT
MISPARSE the first time the layout moves, because nothing ever compares a declared version to a
supported one. Meanwhile the prose contract in ``reader.py`` drifted into stating the opposite of
what the writer does about ``translation``, and stayed wrong for months. An unversioned contract
does not announce that it has gone stale; a versioned one does.

TWO DIFFERENT VERSIONS LIVE IN THIS STORE, and conflating them is the mistake to avoid:

* ``_output._NGFF_VERSION`` ("0.5") is the OME-NGFF SPEC version. It says which published schema
  the ``attributes.ome`` payload conforms to. It belongs to OME and we do not get to bump it.
* ``PLATE_CONTRACT_VERSION`` below is SQUIDXPLORER's promise about the store: the group hierarchy, the
  axis order, what level 0 means, which units physical values are in, and which parts are optional
  with a named fallback. Two stores can both be valid NGFF v0.5 and disagree completely on all of
  that, which is precisely why the spec version cannot stand in for this one.

WHERE THE STAMP LIVES. ``zarr.json -> attributes -> squidxplorer -> plate_contract_version`` on the
PLATE group, once per store. Deliberately OUTSIDE the ``ome`` namespace: ``attributes.ome`` is
OME's, it is what ``ome-zarr-models`` validates, and putting a private key in someone else's
namespace is how you fail a schema check you did not write. Deliberately ONCE, not per well and
not per field: a store has one layout, and the failure mode this module exists to prevent is a
value written in many places and read in none.

MISMATCH POLICY, and the whole design rests on it:

    absent          PROCEED. Every plate written before 2026-07-29 is unstamped, and so is every
                    legitimate third-party NGFF store, which this reader explicitly supports (four
                    zarr layouts, two spec versions). Refusing the unstamped case would reject the
                    installed base to enforce a rule invented after it was written. An absent stamp
                    is "I make no promise", which the structural checks then have to earn.
    same major      PROCEED. A minor bump may only ADD an optional guarantee, so the stable section
                    is unchanged by construction and this reader's assumptions still hold.
    minor ahead     WARN, then proceed. The store declares optional content we were not built to
                    use. That is a lossy read, not a wrong one, and the loss is announced.
    major mismatch  REFUSE, loudly, naming both versions. A major bump means a stable guarantee
                    moved: the hierarchy, the axis order, the meaning of level 0 or the units.
                    Reading it with these assumptions produces a plate that LOOKS right and is not.
    unparseable     REFUSE. A stamp we cannot compare is worse than no stamp, because something
                    deliberately declared a promise and we cannot tell which one.

That is the same judgement ``reader._parse_fov_positions_um`` already makes when it refuses "to
place FOVs at positions that would look plausible but be wrong", applied one level up. This repo's
character is to fail loud rather than render something subtly wrong, and a silently misparsed
plate is the subtly-wrong outcome par excellence.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

#: The version of the SquidXplorer plate contract that this build writes and understands.
#:
#: MAJOR bumps when anything in the STABLE section of ``docs/plate-contract.md`` changes: the
#: ``{row}/{col}/{fov}/{level}`` hierarchy, TCZYX axis order, level 0 being full resolution, the
#: rule that a MIP never comes from a coarser level, or micrometres as the unit of every ``_um``
#: value. A reader built for a different major cannot read the store correctly, so it refuses.
#:
#: MINOR bumps when the OPTIONAL section gains an entry. Optional content has a named fallback by
#: definition, so an older reader stays CORRECT and merely reads less.
PLATE_CONTRACT_VERSION = "1.0"

#: The attributes key the stamp lives under, outside OME's namespace. See the module docstring.
CONTRACT_NAMESPACE = "squidxplorer"
CONTRACT_KEY = "plate_contract_version"

_ZARR_V3_META = "zarr.json"
_ZARR_V2_ATTRS = ".zattrs"


class PlateContractError(Exception):
    """A store declares a plate contract this build cannot read correctly."""


def contract_stamp() -> dict:
    """The ``attributes``-level payload the writer stamps onto the plate group.

    Returned as a whole attributes fragment rather than a bare string so that the writer never has
    to know the key names, and so adding a second contract-level field later touches this file
    only.
    """
    return {CONTRACT_NAMESPACE: {CONTRACT_KEY: PLATE_CONTRACT_VERSION}}


def read_contract_version(plate_dir) -> Optional[str]:
    """The contract version a store declares, or ``None`` when it declares none.

    Reads the raw zarr attributes, both layouts: v3 keeps them in ``zarr.json -> attributes``, v2
    in ``.zattrs`` at the top level. Unlike ``reader._group_attrs`` this does NOT descend into the
    ``ome`` namespace, because the stamp deliberately sits beside it rather than inside it.

    A store that is not a zarr group at all returns ``None`` rather than raising: this function
    answers "what does it claim", and "nothing" is a legitimate answer that the caller's policy,
    not this function, decides what to do about.
    """
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
    """Compare a declared version against what this build supports. The policy, in one place.

    Returns one of ``"absent"``, ``"ok"``, ``"minor-ahead"``. Raises :class:`PlateContractError` on
    a major mismatch or an unparseable stamp. The module docstring argues each branch; this
    function is deliberately pure so the policy is testable without writing a store.
    """
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
    """Read a store's stamp and apply the policy. Returns the declared version, or ``None``.

    Raises :class:`PlateContractError` on a major mismatch. Emits a ``UserWarning`` on a
    minor-ahead store, because the read is lossy rather than wrong and the loss must be announced
    rather than absorbed. ``warn=False`` is for the validator, which collects the same condition
    into its report instead of raising it through the warnings machinery.
    """
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
