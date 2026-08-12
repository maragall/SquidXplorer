"""A cached result that says what it is, so nothing ever has to compare two of them.

Task 2 of the GUI backlog plan (2026-07-29), and it exists to REPLACE something that was banned.

THE BANNED THING, AND WHY
-------------------------
An earlier draft of Task 3 (combine two runs in one plate view) proposed detecting mixed-recipe
plates and warning about them: walk the cells, notice that one run used ``{DAPI, GFP}`` and another
used ``{DAPI}``, and put up a warning. Julio removed it. That is disclosure bolted onto a painter,
and the bolt does not generalise: the next divergence is z depth, then pixel size, then dtype, and
each one needs its own comparison, its own warning and its own test.

The replacement is this module. A result carries its own :class:`~squidxplorer._address.Extent` (WHERE
it is) and its own :class:`Substance` (WHAT it is made of), and the plate composites **what each
cell declares**. Two runs with different channel sets are then not a mismatch to detect. They are
two results that each say what they are, and each cell is drawn from its own declaration. A later
divergence in z depth or pixel size needs no new code, because nothing was ever comparing.

THE TEST OF WHETHER THIS IS BUILT RIGHT
---------------------------------------
**There is no code path here that compares two results.** If a line of this module ever reads
``if a.channels != b.channels``, the banned thing has been rebuilt under a new name.
``tests/test_result.py`` asserts the absence structurally, over the AST rather than over prose, so
a comment mentioning the word cannot make it pass and a real comparison cannot hide in a helper.

Two consequences of taking that seriously, both visible in the code below:

* :class:`Result` is declared ``eq=False``. Two results are not comparable, by construction, and
  not merely by convention. The equality a caller reaches for when they want to "check the plate is
  consistent" simply does not exist on the type.
* A cell that does not declare a channel is **absent** from that channel's composite. It is not
  drawn as black, not drawn from a neighbour's idea of what that channel is, and not reported as an
  error. It did not produce that channel, so there is nothing of it to draw.

WHY THE CHANNEL SET LIVES IN THE SUBSTANCE AND NOT IN THE EXTENT
----------------------------------------------------------------
:class:`~squidxplorer._address.Extent` already has a ``channels`` field, and it looks like a duplicate.
It is not, and the difference is the whole reason this type is needed:

* An extent describes the slab that was **asked for**, where ``None`` means "all of it". That is a
  request, and "all of it" is only resolvable by going back and asking the acquisition.
* A substance describes what the result **is**, and it is never ``None`` and never empty. That is
  what makes a result self-describing: a consumer holding one needs nothing else to draw it.

So there is exactly one way to ask a result what channels it has, ``result.channels``, and it
answers concretely. Nothing reconciles the two, because a request and a product are different
questions.

ORDER: SORTED IN AN EXTENT, PRESERVED IN A SUBSTANCE
----------------------------------------------------
``Extent.channels`` is sorted, because an extent is a KEY and one slab must not have two spellings.
``Substance.channels`` is NOT sorted, because it is DATA: it is the axis order of the pixels, and
sorting it would silently rename every plane. :meth:`Result.plane` therefore looks channels up BY
NAME, never by index, the same rule and for the same reason as
:meth:`squidxplorer._op_result.OperatorResult.plane`.

ON THE WORD "SHAPE"
-------------------
The plan calls this second half "its shape". The word is spoken for three times over in this
codebase: ``ndarray.shape`` is an axis tuple, :mod:`squidxplorer._plate_shape` is plate geometry, and
the writer's validators use it in the numpy sense. Rule 2 of the naming law (see
:mod:`squidxplorer._address`) says a software concept takes a word that cannot collide, so this one is
``Substance``: what a result is made of. ``result.substance.channels`` cannot be misread as an axis
tuple, and ``result.data.shape`` keeps its numpy meaning untouched.

SCOPE (deliberate, 2026-07-29)
------------------------------
The cache KEY is untouched: :class:`squidxplorer._recipe.ResultCache` still keys
``(scope, version, chain)`` with ``scope`` the packed ``row*1e6 + col*1e4 + roi`` string from
``_plate.cache_scope``. Migrating the key onto :class:`~squidxplorer._address.Extent` is Task 3 work,
and doing both at once would have made neither landable. What changes here is the VALUE: the cache
stored bare arrays, and now stores results that know themselves.

This module is pure Python: no Qt, no numpy, like :mod:`squidxplorer._address` and
:mod:`squidxplorer._recipe`. It is the model, testable in isolation, and importing it must not cost a
consumer anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from squidxplorer._address import Extent

__all__ = ["Substance", "Result", "composite_channels", "composite_plate"]


def _dtype_name(value: Any) -> str:
    """A dtype as a stable STRING: ``"uint16"``.

    A string and not a ``numpy.dtype``, because a result's declaration has to survive JSON on its
    way to the on-disk cache, and because this module stays numpy-free. Accepts what producers
    actually hold: a ``numpy.dtype`` (has ``.name``), a scalar type such as ``numpy.uint16`` (has
    ``.__name__``), or a string already.
    """
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
    """WHAT a result is made of: its channel set, its z depth, its dtype and its pixel size.

    Every field is concrete. There is no ``None`` meaning "all of it" anywhere in here, which is
    the difference between this and an :class:`~squidxplorer._address.Extent`: an extent may describe a
    request, a substance describes a product, and a product that cannot say what it is is not
    self-describing.

    Frozen and hashable, so Task 3's legend can group by it without a defensive copy. Comparable,
    unlike :class:`Result`: comparing two DESCRIPTIONS is how a legend lists what is present, and
    it is the comparison of two RESULTS that this module refuses.

    ``kind`` is what the pixels MEAN — the operator registry's ``produces`` declaration, carried on
    the result so a sink does not have to go back to the registry with a layer key that may be
    scoped (``"spot@tab2"``) and split it apart to ask. It is what picks the napari layer type:
    ``"intensity"`` is windowed and colormapped, ``"labels"`` is integer object ids with a
    transparent background and no window at all. Not validated against a list here: the vocabulary
    belongs to the engine and is enforced at ``add_projector``, which is the boundary where the
    declaration is actually made; a second copy of the list in this module is the two-owner defect
    this codebase keeps meeting.
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
        """What a human reads in a legend: ``DAPI,GFP  z_depth 1  uint16  0.325 um/px  labels``.

        Squid's words spelled out, as :meth:`squidxplorer._address.Address.label` does: ``z_depth``
        and never ``nz``, because a shortening printed a thousand times a run is how a second name
        for one thing starts.

        The kind is appended only when it is NOT ``"intensity"``. Intensity is what every label in
        this app has always meant, so spelling it out on every line would be noise; a result whose
        pixels are object ids is the exceptional thing and says so.
        """
        text = (f"{','.join(self.channels)}  z_depth {self.z_depth}  {self.dtype}  "
                f"{self.pixel_size_um:g} um/px")
        return text if self.kind == "intensity" else f"{text}  {self.kind}"

    def __str__(self) -> str:
        return self.label()

    # -- round trip --------------------------------------------------------------------
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
        # `kind` is READ WITH A FALLBACK, not required. Every declaration written to the on-disk
        # cache before 2026-08-03 has no such key, and those results are intensities -- that was
        # the only thing the app could produce down this path. Refusing them would reject the
        # installed base to enforce a field invented after it, which is the same judgement the
        # plate contract makes about an absent `plate_contract_version`.
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

    ``eq=False`` IS THE DESIGN, NOT AN OVERSIGHT. Two results are not comparable. The plate does
    not ask whether two cells agree; it asks each cell what it is and draws that. Removing the
    equality removes the shape of the banned feature from the type itself, so rebuilding it takes
    a deliberate act rather than an ``==``. (It also removes a real hazard: a generated ``__eq__``
    over a field holding a numpy array raises "truth value of an array is ambiguous" at the first
    call, which is a confusing way to learn you were doing something you should not.)

    ``data`` is channel-major on axis 0 -- ``(C, ...)`` -- matching
    :class:`squidxplorer._op_result.OperatorResult`. It may be ``None``: a result read back from a
    declaration on disk knows what it is before its pixels are paged in, and Task 3's plate census
    is a question about declarations only.
    """

    extent: Extent
    substance: Substance
    data: Any = field(default=None, repr=False)

    # -- what it declares --------------------------------------------------------------
    @property
    def region_id(self) -> str:
        return self.extent.region_id

    @property
    def channels(self) -> "tuple[str, ...]":
        """The concrete channel set. ONE way to ask, and it never answers "all of them"."""
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
        """What these pixels MEAN — ``"intensity"`` or ``"labels"``. ONE way to ask."""
        return self.substance.kind

    def declares(self, channel: str) -> bool:
        """Does THIS result carry *channel*?

        A question a result answers about itself, which is what makes the composite work without
        comparing anything. Membership, not equality: ``in`` asks one object what it holds, where
        ``==`` would need a second object to hold it against.
        """
        return str(channel) in self.substance.channels

    def label(self) -> str:
        """``A1  DAPI,GFP  z_depth 1  uint16  0.325 um/px`` -- where it is, then what it is."""
        return f"{self.extent.label()}  {self.substance.label()}"

    def __str__(self) -> str:
        return self.label()

    # -- reading it --------------------------------------------------------------------
    def plane(self, channel: str) -> Any:
        """This result's pixels for *channel*, BY NAME.

        By name and not by index, because the channel order at the producer is not guaranteed to
        be the channel order at the display, and an index would resolve silently to the wrong
        colour rather than raising. The error names what this result DOES carry, since the whole
        point of a self-describing result is that it can say so.
        """
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

    # -- construction ------------------------------------------------------------------
    @classmethod
    def of(cls, extent: Extent, data: Any, *, channels: Sequence[str], z_depth: int,
           pixel_size_um: float, dtype: Any = None, kind: str = "intensity") -> "Result":
        """Build a result, taking the dtype FROM THE PIXELS unless one is given.

        Derived rather than declared because a producer cannot then mislabel its own output, and
        because dtype is the one field that is unambiguous in the array. ``z_depth`` is NOT
        derived: a ``(C, Z, Y, X)`` stack and a ``(C, Y, X)`` plane set are told apart only by
        knowing which the operator produced, and guessing from ``ndim`` is exactly the kind of
        plausible-and-wrong inference this codebase refuses.
        """
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

    # -- round trip --------------------------------------------------------------------
    def to_dict(self) -> dict:
        """The DECLARATION, JSON-safe. The pixels are not in here.

        Pixels go to disk as pixels (see :mod:`squidxplorer._platecache`); what has to travel as JSON
        is the part that says what those pixels are. Splitting them is what lets Task 3's census
        answer "what is on this plate" without paging a single array in.
        """
        return {"extent": self.extent.to_dict(), "substance": self.substance.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping, pixels: Any = None) -> "Result":
        return cls(
            extent=Extent.from_dict(data["extent"]),
            substance=Substance.from_dict(data["substance"]),
            data=pixels,
        )


# --- compositing: each cell drawn from ITS OWN declaration ---------------------------------------
#
# This is the half that replaces the banned warning. Read both functions together: neither of them
# holds two results at once, and that is not an accident of how they were written, it is the
# requirement. Task 3 adds a legend on top of these; if a comparison appears, it appears here
# first, so tests/test_result.py scans this module's AST.

def composite_channels(cells: "Iterable[Result]") -> "tuple[str, ...]":
    """Every channel the plate can show: the UNION of what its cells declare.

    A union and not an intersection, and not "the first cell's set applied to all of them". A run
    over half the plate with ``{DAPI, GFP}`` and a run over the other half with ``{DAPI}`` gives a
    plate that shows DAPI everywhere and GFP where GFP exists, which is the truth. An intersection
    would silently hide a channel that was computed; a first-cell rule would silently invent one
    that was not.

    First-seen order, so each result's own channel order (which is its pixel axis order) is
    preserved rather than alphabetised.
    """
    seen: "dict[str, None]" = {}
    for cell in cells:
        for name in cell.channels:
            seen.setdefault(name, None)
    return tuple(seen)


def composite_plate(cells: "Mapping[Any, Result]", channel: str) -> dict:
    """The plate's cells for one *channel*, each drawn from its own declaration.

    Keyed the same way *cells* was keyed, so the caller's own notion of a cell (an
    :class:`~squidxplorer._address.Address`, a ``(row, col)``, a scope string) survives. A cell that
    does not declare *channel* is ABSENT from the returned map: it did not produce that channel, so
    there is nothing of it to draw. Absent rather than black, because black is a measurement.
    """
    return {key: cell.plane(channel) for key, cell in cells.items() if cell.declares(channel)}
