"""A cached result that says what it is, so the plate composites each cell from its own
declaration instead of comparing two results against each other.

There is no code path here that compares two results — ``tests/test_result.py`` pins that over the
AST. ``Result`` is declared ``eq=False`` for the same reason: two results are not comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from squidxplorer._address import Extent

__all__ = ["Substance", "Result", "composite_channels", "composite_plate"]


def _dtype_name(value: Any) -> str:
    """A dtype as a stable string (``"uint16"``), accepted as a ``numpy.dtype``, a scalar type, or
    a string already."""
    for attr in ("name", "__name__"):
        got = getattr(value, attr, None)
        if isinstance(got, str) and got:
            return got
    text = str(value).strip()
    if not text:
        raise ValueError("a result must declare its dtype; got an empty one")
    return text


@dataclass(frozen=True)
class Substance:
    """WHAT a result is made of: its channel set, z depth, dtype and pixel size.

    Every field is concrete — unlike an :class:`~squidxplorer._address.Extent`, there is no ``None``
    meaning "all of it" here. Frozen and hashable, unlike :class:`Result`, comparing two
    descriptions is fine; it is comparing two results that this module refuses.

    ``kind`` is what the pixels MEAN (the operator registry's ``produces`` declaration), carried on
    the result so a sink doesn't have to go back to the registry. It picks the napari layer type:
    ``"intensity"`` is windowed and colormapped, ``"labels"`` is integer ids with no window.
    """

    channels: "tuple[str, ...]"
    z_depth: int
    dtype: str
    pixel_size_um: float
    kind: str = "intensity"

    def __post_init__(self) -> None:
        chans = tuple(str(c) for c in self.channels)
        if not chans:
            raise ValueError(
                "a result must declare at least one channel; an empty channel set cannot be "
                "drawn and cannot be told apart from 'all of them'")
        if len(set(chans)) != len(chans):
            raise ValueError(f"a result declared a channel twice: {list(chans)}; "
                             "a duplicate name makes plane lookup by name ambiguous")
        object.__setattr__(self, "channels", chans)

        depth = int(self.z_depth)
        if depth < 1:
            raise ValueError(f"z_depth must be at least 1, got {depth}; a result with no z planes "
                             "has no pixels, and a MIP has depth 1 rather than depth 0")
        object.__setattr__(self, "z_depth", depth)

        object.__setattr__(self, "dtype", _dtype_name(self.dtype))

        px = float(self.pixel_size_um)
        if not (px > 0):
            raise ValueError(
                f"pixel_size_um must be positive, got {px}. This codebase refuses to guess a "
                "scale rather than placing pixels at a size that would look plausible and be "
                "wrong; see reader.py's refusal to place FOVs without stage positions")
        object.__setattr__(self, "pixel_size_um", px)

        kind = str(self.kind).strip()
        if not kind:
            raise ValueError(
                "a result must declare what its pixels MEAN; got an empty kind. An operator's "
                "registry entry answers this (produces='intensity' / 'labels')")
        object.__setattr__(self, "kind", kind)

    def label(self) -> str:
        """``DAPI,GFP  z_depth 1  uint16  0.325 um/px  labels``. ``kind`` is appended only when it
        is not ``"intensity"``, the default every other label in this app has always meant."""
        text = (f"{','.join(self.channels)}  z_depth {self.z_depth}  {self.dtype}  "
                f"{self.pixel_size_um:g} um/px")
        return text if self.kind == "intensity" else f"{text}  {self.kind}"

    def __str__(self) -> str:
        return self.label()

    def to_dict(self) -> dict:
        return {
            "channels": list(self.channels),
            "z_depth": self.z_depth,
            "dtype": self.dtype,
            "pixel_size_um": self.pixel_size_um,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "Substance":
        # `kind` falls back to "intensity": caches written before this field existed have no such
        # key, and everything the app could produce down that path was an intensity.
        return cls(
            channels=tuple(data["channels"]),
            z_depth=data["z_depth"],
            dtype=data["dtype"],
            pixel_size_um=data["pixel_size_um"],
            kind=data.get("kind", "intensity"),
        )


@dataclass(frozen=True, eq=False)
class Result:
    """A cached result: WHERE it is (:class:`~squidxplorer._address.Extent`), WHAT it is made of
    (:class:`Substance`), and the pixels.

    ``eq=False`` is deliberate: the plate never asks whether two cells agree, only what each one
    is. (It also sidesteps a generated ``__eq__`` raising on a numpy array field.)

    ``data`` is channel-major on axis 0 (``(C, ...)``) -- a sequence of per-channel planes
    qualifies, which is how :class:`squidxplorer._op_result.RegionResultAccumulator` hands its fused
    planes over without restacking them -- and may be ``None`` for a declaration read back before
    its pixels are paged in.
    """

    extent: Extent
    substance: Substance
    data: Any = field(default=None, repr=False)

    @property
    def region_id(self) -> str:
        return self.extent.region_id

    @property
    def channels(self) -> "tuple[str, ...]":
        return self.substance.channels

    @property
    def z_depth(self) -> int:
        return self.substance.z_depth

    @property
    def dtype(self) -> str:
        return self.substance.dtype

    @property
    def pixel_size_um(self) -> float:
        return self.substance.pixel_size_um

    @property
    def kind(self) -> str:
        return self.substance.kind

    def declares(self, channel: str) -> bool:
        return str(channel) in self.substance.channels

    def label(self) -> str:
        return f"{self.extent.label()}  {self.substance.label()}"

    def __str__(self) -> str:
        return self.label()

    def plane(self, channel: str) -> Any:
        """This result's pixels for *channel*, by name (not index — a producer's channel order is
        not guaranteed to match the display's)."""
        name = str(channel)
        if self.data is None:
            raise ValueError(
                f"result for {self.extent.label()} carries no pixels (data is None); it is a "
                "declaration only")
        try:
            index = self.substance.channels.index(name)
        except ValueError:
            raise KeyError(
                f"result for {self.extent.label()} has no channel {name!r}; "
                f"it carries {list(self.substance.channels)}") from None
        return self.data[index]

    @classmethod
    def of(cls, extent: Extent, data: Any, *, channels: Sequence[str], z_depth: int,
           pixel_size_um: float, dtype: Any = None, kind: str = "intensity") -> "Result":
        """Build a result. ``dtype`` is taken from the pixels unless given; ``z_depth`` is never
        derived, since a ``(C, Z, Y, X)`` stack and a ``(C, Y, X)`` plane set can't be told apart
        from ``ndim`` alone."""
        names = tuple(str(c) for c in channels)
        if dtype is None:
            if data is None:
                raise ValueError("a result with no pixels must declare its dtype explicitly")
            dtype = getattr(data, "dtype", None)
            if dtype is None:
                raise ValueError(
                    "cannot take a dtype from this data; pass dtype= explicitly")
        shape = getattr(data, "shape", None)
        if shape is not None and len(shape) and int(shape[0]) != len(names):
            raise ValueError(
                f"result for {extent.label()}: pixels are {shape[0]}-deep on the channel axis but "
                f"{len(names)} channel(s) were declared ({list(names)}); refusing to guess which "
                "plane is which")
        return cls(
            extent=extent,
            substance=Substance(channels=names, z_depth=z_depth, dtype=dtype,
                                pixel_size_um=pixel_size_um, kind=kind),
            data=data,
        )

    def to_dict(self) -> dict:
        """The declaration, JSON-safe; the pixels are not in here (see ``_platecache``)."""
        return {"extent": self.extent.to_dict(), "substance": self.substance.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping, pixels: Any = None) -> "Result":
        return cls(
            extent=Extent.from_dict(data["extent"]),
            substance=Substance.from_dict(data["substance"]),
            data=pixels,
        )


def composite_channels(cells: "Iterable[Result]") -> "tuple[str, ...]":
    """Every channel the plate can show: the union of what its cells declare, in first-seen order.

    A union, not an intersection or "the first cell's set" — a run over half the plate with
    ``{DAPI, GFP}`` and the other half with ``{DAPI}`` should show DAPI everywhere and GFP where it
    exists.
    """
    seen: "dict[str, None]" = {}
    for cell in cells:
        for name in cell.channels:
            seen.setdefault(name, None)
    return tuple(seen)


def composite_plate(cells: "Mapping[Any, Result]", channel: str) -> dict:
    """The plate's cells for one *channel*, each drawn from its own declaration. A cell that does
    not declare *channel* is absent from the result, not drawn as black."""
    return {key: cell.plane(channel) for key, cell in cells.items() if cell.declares(channel)}
