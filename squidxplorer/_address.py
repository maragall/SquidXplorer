"""Where in the acquisition something is (Address), and which slab a result covers (Extent).

Physical dimensions use Squid's exact words (region_id, fov, z_level, time_point, channel);
software-only concepts (Extent, bbox_um) use words Squid will never use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

#: What a canonical key prints where a field is None ("all of it").
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
    """A bbox as four floats with ``x0 <= x1`` and ``y0 <= y1`` (orientation normalised)."""
    if value is None:
        return None
    x0, y0, x1, y1 = (float(v) for v in value)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


@dataclass(frozen=True)
class Address:
    """Where in the acquisition. Real dimensions only, in Squid's own words.

    ``None`` means "all of it" on that dimension. Frozen and hashable so it can be a cache key.
    """

    region_id: str                   # "A1", "manual0"
    fov: Optional[int] = None        # flat index, row-major
    z_level: Optional[int] = None    # an index, not micrometres
    time_point: Optional[int] = None
    channel: Optional[str] = None    # by name

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", str(self.region_id))
        object.__setattr__(self, "fov", _opt_int(self.fov))
        object.__setattr__(self, "z_level", _opt_int(self.z_level))
        object.__setattr__(self, "time_point", _opt_int(self.time_point))
        object.__setattr__(self, "channel", _opt_str(self.channel))

    def key(self) -> str:
        """The canonical string form. Stable across processes and across runs."""
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

    def to_dict(self) -> dict:
        """JSON-safe; keeps every ``None`` so "all of it" survives the round trip."""
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
    """Which slab a result covers: ranges over an :class:`Address`.

    ``None`` means "all of it" on that dimension; ``bbox_um`` is the ROI. ``fovs`` and
    ``channels`` are normalised to sorted, duplicate-free tuples so one slab has one key.
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

    def key(self) -> str:
        """The canonical string form: two draws of the same box give one key."""
        return "|".join((
            self.region_id,
            f"fovs={ALL if self.fovs is None else ','.join(str(f) for f in self.fovs)}",
            f"z_levels={_range_key(self.z_levels)}",
            f"time_points={_range_key(self.time_points)}",
            f"channels={ALL if self.channels is None else ','.join(self.channels)}",
            f"bbox_um={ALL if self.bbox_um is None else ','.join(repr(v) for v in self.bbox_um)}",
        ))

    def label(self) -> str:
        """Console form; an ROI reads as its box, rounded for the eye only."""
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

    @classmethod
    def over(cls, address: Address, *, bbox_um: Optional[Sequence[float]] = None) -> "Extent":
        """The one-point slab an :class:`Address` describes, optionally boxed by an ROI."""
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
    """Sorted, duplicate-free, or None; an empty sequence collapses to None."""
    if value is None:
        return None
    out = tuple(sorted({int(v) for v in value}))
    return out or None


def _norm_strs(value: Any) -> "Optional[Tuple[str, ...]]":
    if value is None:
        return None
    out = tuple(sorted({str(v) for v in value}))
    return out or None
