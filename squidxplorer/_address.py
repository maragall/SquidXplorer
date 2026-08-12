"""WHERE in the acquisition something is, and WHICH SLAB of it a result covers.

Task 1 of the GUI backlog plan (2026-07-29). The one identifier this application had was
``scope = row * 1_000_000 + col * 10_000 + roi`` (:mod:`squidxplorer._plate`, still there, see the
scope note at the end of this docstring). Two of its three fields are real acquisition coordinates
and the third is THE ORDER SOMEBODY DREW BOXES. It names no z level, no timepoint and no channel,
and it is simultaneously

* the flat cache key,
* the logger's id for a thing, and
* the navigator's row,

so it is asked to be a data address and a view id at once and is a poor version of both. Two
consequences, both observed rather than theorised: draw the same box twice and identical work is
computed and cached twice, because the second draw gets ROI slot 1; delete ROI 2 and every later id
shifts under whatever was pointing at it.

THE LAW THIS MODULE IS WRITTEN UNDER (Julio, 2026-07-29)
--------------------------------------------------------
    **Squid models the physical world. We model the processing of what it recorded. The language
    must agree.**

The test for any identifier, in order:

1. **Does it name something that exists in the microscope?** A region on a plate, a field of view
   of a camera, a z level of a stage, a timepoint of a clock, a channel of illumination. Then use
   SQUID'S WORD EXACTLY. Not a synonym, not an abbreviation, not a shortening: ``region_id`` not
   ``region_name``, ``z_level`` not ``z``, ``time_point`` not ``t``, ``fov`` not ``field``. Same
   reality, same word. This is not deference to Squid. Two names for one physical thing is how
   drift starts, and the drift is only discovered at the merge.
2. **Does it exist only in our software?** A recipe, a pyramid level, a window, a cache, an
   extent, a view id. Then pick a word SQUID WILL NEVER USE, so the two vocabularies cannot
   collide.
3. **Never reuse a physical word for a software concept.**

:class:`Address` is category 1 throughout, which is why every one of its fields is spelled the way
``Squid`` spells it. :class:`Extent`, ``bbox_um`` and the view id are category 2: Squid has no
extent, no bounding box and no windows, so those names are ours and stay ours.

RULE 3 IS THE ONE THE CURRENT CODE BREAKS, AND ``roi`` BREAKS IT IN TWO DIRECTIONS AT ONCE
------------------------------------------------------------------------------------------
Our ROI is a user-drawn box: a software concept, category 2. But in ``row*1e6 + col*1e4 + roi`` it
occupies the structural slot where Squid puts a FIELD OF VIEW, which is a physical thing. And Squid
separately uses "ROI" in its own GUI for a manually drawn SCAN SHAPE, which is an acquisition
INPUT: the opposite direction of travel from ours, which is a selection made after the fact over
data already on disk. One word, two ontologies, pointing opposite ways, sitting in a slot that
belongs to a third concept. Searching Squid for an ROI concept at the level our packed id puts it
finds nothing, because there is nothing there: ``fov`` is what lives at that position.

A known second instance, diagnosed and deliberately NOT fixed here (it is not Task 1 work): in
:mod:`squidxplorer._region_viewer`, ``region`` sometimes means a place on a plate and sometimes means a
window on a desktop. Same defect, same rule, and the fix is the same shape: the desktop thing is
category 2 and needs a word of its own.

WHAT AN ADDRESS BUYS
--------------------
* **An ROI stops being an ordinal.** It is a ``bbox_um``. The same box drawn twice is the same
  extent, therefore the same key, therefore computed once. Delete an ROI and nothing shifts,
  because nothing was ever numbered.
* **A view id stays a plain integer per window**, and it is NOT in this module. That is where an
  ordinal legitimately belongs: windows are things on a desktop, they are ours, they come and go,
  and ``RegionViewer.window_id`` already is one. A log line is ``[3] A1 fov 2 ...``: the bracket is
  the VIEW and the rest is the ADDRESS, and keeping them in two objects is the whole point.
* **``time_point`` sits in an extent exactly as ``z_level`` does.** No special case, and nothing
  about the cache needs deciding when the timepoint slider lands (Task 4).

WHY THERE IS NO TOLERANCE ON ``bbox_um``
----------------------------------------
It is tempting to call two boxes "the same" when they agree to within a nanometre, since no stage
repeats better than that. Refused: an approximate equality is not transitive (a is near b, b is
near c, a is not near c), and a key whose equality is not transitive is not a key. Two boxes are
the same box when their four numbers are the same numbers. What IS normalised is orientation:
``(x0, y0, x1, y1)`` is stored with ``x0 <= x1`` and ``y0 <= y1``, so dragging a box from its
bottom-right corner and dragging the same box from its top-left corner produce one extent. The
``(x0, y0, x1, y1)`` layout, X FIRST, in stage micrometres, is this repo's existing convention;
see :func:`squidxplorer._napari_view.scale_translate_from_bbox_um`, which raises unless ``x1 > x0``.

``label()`` SPELLS THE WORDS OUT
--------------------------------
``A1 fov 2 z_level 5`` and never ``A1 fov 2 z 5``. A console is where a vocabulary is learned, and
a shortening printed a thousand times a run is how a second name for one thing starts. The common
case is a region alone, so the lines stay short anyway.

KNOWN GAP, FOUND BY THE FIRST CONSUMER, AND IT IS TASK 2's TO CLOSE
--------------------------------------------------------------------
``region_id`` is ONE region, on both types, deliberately. So a run over the whole plate has no
single extent: it is a SET of them, one per cell. The plate window's console lines show it — a run
over one region logs its extent, a run over many logs its count and no address. Resisted here: a
sentinel ``region_id`` (``"*"``, ``None``, ``"<plate>"``) would make the type able to say
"everywhere", and every consumer would then have to branch on whether the region field is a region.
The right answer is the one Task 2 already reaches for, where a cached RESULT carries its own
extent: the run's answer is one extent per cell, not one extent for the run, and Task 3's plate
census is already specified as ``{chain: [address]}`` — a list.

SCOPE (deliberate, 2026-07-29)
------------------------------
This module is ADDED ALONGSIDE the packed integer id, which keeps every one of its callers.
Migrating the cache key onto :class:`Extent` is Task 2/3 work; doing it here would have made the
commit unlandable against a suite that pins the packed id's behaviour. ``_plate.well_code``,
``_plate.roi_code`` and ``_plate.cache_scope`` carry a comment naming this module as their
successor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

#: What a canonical key prints where a field is None. None means "all of it", which is the common
#: case; a literal token keeps every key the same shape so two keys can be compared by eye.
ALL = "*"


def _opt_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _opt_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _range_key(r: "Optional[range]") -> str:
    return ALL if r is None else f"{r.start}:{r.stop}:{r.step}"


def _range_to_list(r: "Optional[range]") -> "Optional[list]":
    return None if r is None else [r.start, r.stop, r.step]


def _range_from_list(v: Any) -> "Optional[range]":
    if v is None:
        return None
    if isinstance(v, range):
        return v
    start, stop, step = (int(x) for x in v)
    return range(start, stop, step)


def _float4(value: Any) -> "Optional[Tuple[float, float, float, float]]":
    """A bbox as four floats with ``x0 <= x1`` and ``y0 <= y1``. See the docstring: orientation is
    normalised (the same box dragged from either corner is one box), magnitudes are not."""
    if value is None:
        return None
    x0, y0, x1, y1 = (float(v) for v in value)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


@dataclass(frozen=True)
class Address:
    """WHERE in the acquisition. Real dimensions only, in Squid's own words.

    ``None`` means "all of it" on that dimension. An address is therefore also how you say "well
    A1, everything in it": ``Address("A1")``.

    Frozen and hashable so it can be a dict key and a cache key without a defensive copy anywhere.
    """

    region_id: str                   # Squid's word. "A1", "manual0"
    fov: Optional[int] = None        # flat index, row-major, fov = row * nx + col (schema v2)
    z_level: Optional[int] = None    # Squid's word. An INDEX, not micrometres
    time_point: Optional[int] = None  # Squid's word
    channel: Optional[str] = None    # by NAME, as Squid does

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", str(self.region_id))
        object.__setattr__(self, "fov", _opt_int(self.fov))
        object.__setattr__(self, "z_level", _opt_int(self.z_level))
        object.__setattr__(self, "time_point", _opt_int(self.time_point))
        object.__setattr__(self, "channel", _opt_str(self.channel))

    # -- identity ----------------------------------------------------------------------
    def key(self) -> str:
        """The canonical string form. Stable across processes and across runs.

        Deliberately NOT ``hash()``: Python's string hash is salted per process, so a key derived
        from it would not survive a restart, and this string is meant to end up in the on-disk
        cache when Task 2 migrates it.
        """
        return "|".join((
            self.region_id,
            f"fov={ALL if self.fov is None else self.fov}",
            f"z_level={ALL if self.z_level is None else self.z_level}",
            f"time_point={ALL if self.time_point is None else self.time_point}",
            f"channel={ALL if self.channel is None else self.channel}",
        ))

    def label(self) -> str:
        """What a human reads in the console: ``A1 fov 2 z_level 5``. Omits every ``None``."""
        parts = [self.region_id]
        if self.fov is not None:
            parts.append(f"fov {self.fov}")
        if self.z_level is not None:
            parts.append(f"z_level {self.z_level}")
        if self.time_point is not None:
            parts.append(f"time_point {self.time_point}")
        if self.channel is not None:
            parts.append(str(self.channel))
        return " ".join(parts)

    def __str__(self) -> str:
        return self.label()

    # -- round trip --------------------------------------------------------------------
    def to_dict(self) -> dict:
        """JSON-safe, and it keeps every ``None`` so "all of it" survives the round trip."""
        return {
            "region_id": self.region_id,
            "fov": self.fov,
            "z_level": self.z_level,
            "time_point": self.time_point,
            "channel": self.channel,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Address":
        return cls(
            region_id=data["region_id"],
            fov=data.get("fov"),
            z_level=data.get("z_level"),
            time_point=data.get("time_point"),
            channel=data.get("channel"),
        )


@dataclass(frozen=True)
class Extent:
    """WHICH SLAB a result covers. Ranges over an :class:`Address`.

    ``None`` means "all of it" on that dimension, so ``Extent("A1")`` is the whole well and is the
    common case. ``bbox_um`` is the ROI, and it is what makes an ROI stop being an ordinal.

    ``fovs`` and ``channels`` are normalised to sorted, duplicate-free tuples. A slab is a SET of
    fields and a SET of channels: the order they were listed in is not part of what the slab
    covers, and leaving it in would let one slab have two keys, which is the exact bug this module
    exists to remove. Ordering for DISPLAY is a window's business, not an extent's.
    """

    region_id: str
    fovs: "Optional[Tuple[int, ...]]" = None
    z_levels: "Optional[range]" = None
    time_points: "Optional[range]" = None
    channels: "Optional[Tuple[str, ...]]" = None
    bbox_um: "Optional[Tuple[float, float, float, float]]" = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", str(self.region_id))
        object.__setattr__(self, "fovs", _norm_ints(self.fovs))
        object.__setattr__(self, "channels", _norm_strs(self.channels))
        object.__setattr__(self, "bbox_um", _float4(self.bbox_um))
        if self.z_levels is not None and not isinstance(self.z_levels, range):
            object.__setattr__(self, "z_levels", _range_from_list(self.z_levels))
        if self.time_points is not None and not isinstance(self.time_points, range):
            object.__setattr__(self, "time_points", _range_from_list(self.time_points))

    # -- identity ----------------------------------------------------------------------
    def key(self) -> str:
        """The canonical string form: the cache key an ROI deserves.

        Two draws of the same box give the same four numbers, so they give ONE key, so the work
        behind it is done once. Nothing here encodes the order anything was created in, which is
        why deleting one extent cannot move another.
        """
        return "|".join((
            self.region_id,
            f"fovs={ALL if self.fovs is None else ','.join(str(f) for f in self.fovs)}",
            f"z_levels={_range_key(self.z_levels)}",
            f"time_points={_range_key(self.time_points)}",
            f"channels={ALL if self.channels is None else ','.join(self.channels)}",
            f"bbox_um={ALL if self.bbox_um is None else ','.join(repr(v) for v in self.bbox_um)}",
        ))

    def label(self) -> str:
        """What a human reads in the console. An ROI reads as its box, rounded for the eye only:
        the KEY is :meth:`key` and is never rounded."""
        parts = [self.region_id]
        if self.fovs is not None:
            parts.append("fov " + ",".join(str(f) for f in self.fovs))
        if self.z_levels is not None:
            parts.append(f"z_level {self.z_levels.start}..{self.z_levels.stop - 1}")
        if self.time_points is not None:
            parts.append(f"time_point {self.time_points.start}..{self.time_points.stop - 1}")
        if self.channels is not None:
            parts.append(",".join(self.channels))
        if self.bbox_um is not None:
            x0, y0, x1, y1 = self.bbox_um
            parts.append(f"roi [{x0:.1f},{y0:.1f} {x1:.1f},{y1:.1f}] um")
        return " ".join(parts)

    def __str__(self) -> str:
        return self.label()

    # -- round trip --------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "fovs": None if self.fovs is None else list(self.fovs),
            "z_levels": _range_to_list(self.z_levels),
            "time_points": _range_to_list(self.time_points),
            "channels": None if self.channels is None else list(self.channels),
            "bbox_um": None if self.bbox_um is None else list(self.bbox_um),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Extent":
        return cls(
            region_id=data["region_id"],
            fovs=data.get("fovs"),
            z_levels=_range_from_list(data.get("z_levels")),
            time_points=_range_from_list(data.get("time_points")),
            channels=data.get("channels"),
            bbox_um=data.get("bbox_um"),
        )

    # -- construction ------------------------------------------------------------------
    @classmethod
    def over(cls, address: Address, *, bbox_um: Optional[Sequence[float]] = None) -> "Extent":
        """The one-point slab an :class:`Address` describes, optionally boxed by an ROI.

        The bridge between the two types, so a caller holding an address never has to remember
        which field became which plural.
        """
        return cls(
            region_id=address.region_id,
            fovs=None if address.fov is None else (address.fov,),
            z_levels=None if address.z_level is None else range(address.z_level,
                                                                address.z_level + 1),
            time_points=None if address.time_point is None else range(address.time_point,
                                                                      address.time_point + 1),
            channels=None if address.channel is None else (address.channel,),
            bbox_um=bbox_um,
        )


def _norm_ints(value: Any) -> "Optional[Tuple[int, ...]]":
    """Sorted, duplicate-free, or None. An EMPTY sequence collapses to None on purpose: "no
    restriction" and "the empty restriction" are the same slab, and one meaning must have one key."""
    if value is None:
        return None
    out = tuple(sorted({int(v) for v in value}))
    return out or None


def _norm_strs(value: Any) -> "Optional[Tuple[str, ...]]":
    if value is None:
        return None
    out = tuple(sorted({str(v) for v in value}))
    return out or None
