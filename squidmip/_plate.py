"""Plate geometry — the single source of truth for well-plate shape and well-id rules.

Before this module, plate geometry lived in FOUR places that had to agree and were never
checked against each other::

    _viewer.py  _PLATE_DIMS      well count -> (rows, cols)      (hand-copy of Squid's CSV)
    _viewer.py  _row_letter      index -> A..Z, AA..AF
    reader.py   _plate_key       well id -> row-major sort key
    _output.py  _row_sort_key    row order
                parse_well_id    well id -> (row, col)

All of them now delegate here.

Why a plain frozen dataclass and not an ABC + WellPlate/SlideCarrier hierarchy: Squid's
own ``sample_formats.csv`` already models a glass slide as a plate row with
``rows=1, cols=1`` (``glass slide,0,0,0,0,0,0,0,1,1``), and pushes both through one
settings signal of one shape (``widgets.py:12965``). Upstream agrees --
``gui_hcs.py:2397``: *"Not sure why glass slide is so special here? It seems like it's
just a '1 well plate'."* The only thing that actually differs between the two cases is
the label vocabulary, and that is data, not behaviour.

Geometry source: the ``rows``/``cols`` columns of Squid's ``sample_formats.csv`` are
physical constants of the plate hardware -- no user edits them -- so they are embedded
here rather than vendored as a data file. The mm/pixel columns of that CSV (which a user
CAN customise via ``cache/sample_formats.csv``) are deliberately NOT modelled: nothing in
squidmip consumes them, and carrying them would import a drift risk for no delivered
feature.

Well-id rules enforced here (each one closes a verified silent-corruption path)::

    "B2"      -> ("B", 2)     canonical
    "aa3"     -> ("AA", 3)    upper-cased, never zero-padded
    "R0"      -> refused      column 0 exists on NO Squid plate (columns are 1-based).
                              Squid's flexible mode really does emit R0/R1
                              (widgets.py:6584), and the old parser accepted it as
                              row "R" column "0", writing plate.ome.zarr/R/0/.
    "R2C3"    -> refused      not <letters><digits>
    "region"  -> refused      no column at all
    "0"       -> refused      no row at all
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Canonical well id: letters then digits, nothing else. Anchored on purpose -- a partial
# match is what let "R2C3" and "region_1" through in earlier revisions.
_WELL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")

# Squid sample formats -> (rows, cols). These are the `rows`/`cols` columns of
# objective_and_sample_formats/sample_formats.csv, keyed by Squid's own convention
# (control/_def.py:1151): f"{n} well plate" for numeric formats, else the literal string.
#
#   "0" is NOT a CSV row -- it is a hardcoded no-plate special case in Squid
#   (control/_def.py:1196) with rows=1, cols=1. Missing it would strand real acquisitions.
#
#   "4 well plate" is likewise absent from the CSV but was present in the viewer's old
#   _PLATE_DIMS table; kept so no currently-working input regresses.
_FORMATS: dict[str, tuple[int, int]] = {
    "glass slide": (1, 1),
    "0": (1, 1),
    "4 well plate": (2, 2),
    "6 well plate": (2, 3),
    "12 well plate": (3, 4),
    "24 well plate": (4, 6),
    "96 well plate": (8, 12),
    "384 well plate": (16, 24),
    "1536 well plate": (32, 48),
}

# Well counts -> the same dims, for the tolerant "find a number in the string" fallback the
# viewer relied on (_plate_grid used re.search(r"(\d+)")). Preserves that behaviour.
_BY_COUNT: dict[int, tuple[int, int]] = {
    4: (2, 2), 6: (2, 3), 12: (3, 4), 24: (4, 6),
    96: (8, 12), 384: (16, 24), 1536: (32, 48),
}


class NotAWellPlateError(ValueError):
    """A region id is not a usable well-plate position.

    Its own type, rather than a bare ValueError, because the viewer must tell "this file
    is unreadable" apart from "this is readable but isn't a well plate" -- two different
    messages for the user. Previously it distinguished them by WHICH LINE raised, so
    moving a check between functions silently changed what users were told.

    Subclasses ValueError so existing ``except ValueError`` handlers keep working.
    """


def row_letter(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, ... (plate row labels)."""
    s, i = "", index + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def row_index(letters: str) -> int:
    """Inverse of :func:`row_letter`. "A" -> 0, "AA" -> 26."""
    n = 0
    for c in letters.upper():
        n = n * 26 + (ord(c) - 64)
    return n - 1


def parse_well_id(region: str) -> tuple[str, str]:
    """Split a well id into (ROW_LETTERS, COL_DIGITS), or raise :class:`NotAWellPlateError`.

    Matches Squid's accepted inputs (upper-cased, multi-letter rows, no zero-padding --
    ``B2`` stays ``B``/``2``, never ``B``/``02``, because ndviewer_light rebuilds the well
    id by concatenating the directory names) but additionally ASSERTS the canonical shape.
    A manual/no-plate region must not be written to a mislabelled directory.

    Column 0 is refused: no Squid plate format has a column 0 (they are 1-based). This is
    what stops ``R0`` -- Squid's real flexible-region naming (``widgets.py:6584``) -- from
    being silently filed under row ``R``, column ``0``.
    """
    s = str(region).upper()
    m = _WELL_RE.match(s)
    if not m:
        raise NotAWellPlateError(
            f"region {region!r} is not a canonical <letters><digits> well id (e.g. 'B2', "
            "'AA3'); the HCS plate layout needs a row/column split. Manual / non-well-plate "
            "acquisitions are out of scope."
        )
    letters, digits = m.group(1), m.group(2)
    if int(digits) < 1:
        raise NotAWellPlateError(
            f"region {region!r} has column {int(digits)}, but plate columns are 1-based on "
            "every Squid format — this is a flexible-region / manual id (Squid names those "
            "R0, R1, …), not a well id. Refused rather than filed under a mislabelled well."
        )
    return letters, digits


def is_well_id(region: str) -> bool:
    """True when *region* is a canonical, in-range well id."""
    try:
        parse_well_id(region)
        return True
    except NotAWellPlateError:
        return False


def sort_key(region: str):
    """Sort key putting well ids in true plate ROW-MAJOR order.

    A, B, ... Z, AA, AB, ... -- and the column by integer, so B2 < B3 < B10, and B < AA
    (single-letter rows before double-letter, NOT lexicographic where "AA" < "B"). A
    1536-well plate must fill row A, then B..Z, then AA..AF; the naive natural sort put the
    AA rows second and filled the plate view out of visual order.

    Non-well ids sort AFTER the plate wells, stably.
    """
    m = _WELL_RE.match(str(region))
    if not m:
        return (1, len(str(region)), str(region), 0)
    return (0, len(m.group(1)), m.group(1).upper(), int(m.group(2)))


@dataclass(frozen=True)
class Plate:
    """A plate's grid shape and label vocabulary. Immutable; cheap to pass around.

    ``row_labels``/``col_labels`` and the reverse index maps are built ONCE in
    ``__post_init__``, never per call: ``sort_key`` and ``index_of`` run once per region
    inside ``sorted()``, so rebuilding a label list per call would make a 1536-well sort
    quadratic.
    """

    format: str
    rows: int
    cols: int

    row_labels: tuple = field(default=(), repr=False)
    col_labels: tuple = field(default=(), repr=False)
    _row_of: dict = field(default_factory=dict, repr=False, compare=False)
    _col_of: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        rl = tuple(row_letter(i) for i in range(self.rows))
        cl = tuple(str(c) for c in range(1, self.cols + 1))
        object.__setattr__(self, "row_labels", rl)
        object.__setattr__(self, "col_labels", cl)
        object.__setattr__(self, "_row_of", {r: i for i, r in enumerate(rl)})
        object.__setattr__(self, "_col_of", {c: i for i, c in enumerate(cl)})

    # -- construction -----------------------------------------------------
    @classmethod
    def from_format(cls, wellplate_format) -> Optional["Plate"]:
        """Build from a Squid ``wellplate_format`` string, or None if unrecognised.

        Returns None rather than raising: the viewer's contract is to fall back to a
        present-only grid for an unknown format, not to refuse the acquisition.
        """
        if wellplate_format is None:
            return None
        key = str(wellplate_format).strip()
        dims = _FORMATS.get(key) or _FORMATS.get(key.lower())
        if dims is None:
            # Tolerant fallback: pull a well count out of the string ("384 well plate",
            # "384wp", "384"). Mirrors the viewer's old re.search(r"(\d+)") behaviour.
            m = re.search(r"(\d+)", key)
            dims = _BY_COUNT.get(int(m.group(1))) if m else None
        if dims is None:
            return None
        return cls(format=key, rows=dims[0], cols=dims[1])

    # -- positions --------------------------------------------------------
    def index_of(self, region: str) -> tuple[int, int]:
        """(row_index, col_index) for a well id, or raise :class:`NotAWellPlateError`.

        Bounds-checked against THIS plate, so a well id that is canonical but off the
        plate (``B99`` on a 96-well) is refused rather than silently placed out of grid.
        """
        letters, digits = parse_well_id(region)
        ri, ci = row_index(letters), int(digits) - 1
        if ri >= self.rows or ci >= self.cols:
            raise NotAWellPlateError(
                f"well {region!r} is outside a {self.format} plate "
                f"({self.rows} rows x {self.cols} cols)."
            )
        return ri, ci

    def contains(self, region: str) -> bool:
        """True when *region* is a valid well id that fits on this plate."""
        try:
            self.index_of(region)
            return True
        except NotAWellPlateError:
            return False

    def well_id(self, row_index_: int, col_index: int) -> str:
        """(row_index, col_index) -> well id, e.g. (1, 1) -> "B2"."""
        return f"{self.row_labels[row_index_]}{self.col_labels[col_index]}"

    # Re-exported so callers need only the Plate, not the module functions too.
    sort_key = staticmethod(sort_key)
    parse_well_id = staticmethod(parse_well_id)
    is_well_id = staticmethod(is_well_id)
