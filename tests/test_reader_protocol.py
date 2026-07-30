"""The reader interface, given a name Squid can import.

WHAT WAS WRONG
--------------
``open_reader`` dispatches to four duck-typed classes -- ``SquidReader``,
``SquidMultiPageTiffReader``, ``SquidOMEReader``, ``SquidZarrReader`` -- and there was no base
class and no Protocol anywhere. ``reader.py``'s own docstring called that a virtue: "The interface
IS the seam: engine, CLI and viewer consume any of them with no isinstance check and no parallel
API." Inside this repo that is true, and duck typing across four classes with one test suite is
fine.

Across a REPO BOUNDARY it is not a contract. SquidXplorer is a child node of Squid software: v1
opens from a button in Squid, v2 replaces its mosaic and multi-channel views. For that merge to be
a no-op rather than a translation layer, Squid needs a NAME to depend on, and there was none. The
only importable name was ``SquidReader``, which is one of the four and the wrong one for three of
``open_reader``'s six dispatch targets, and ``open_reader`` itself was annotated ``-> SquidReader``,
so the repo's own type annotation asserted the thing that is not true.

WHAT IS PINNED HERE
-------------------
* All four reader classes satisfy :class:`squidmip.reader.SquidAcquisitionReader`, exercised
  through every one of the six writer fixtures rather than through whichever acquisition happens
  to be on this machine.
* The Protocol describes what the four ALREADY do. Its member set is the intersection, asserted:
  ``plane_path`` is excluded and the assertion says why (two of four have one, because only a TIFF
  plane is a file), so nobody "completes" the Protocol later by adding a member half the
  implementations lack.
* The SIGNATURES agree, checked with ``inspect.signature``. ``runtime_checkable`` only looks for
  attribute presence, so an ``isinstance`` pass alone would let ``read(self, region, fov)`` count
  as an implementation. That gap is exactly what would bite at the merge, where the caller is in
  another repo and cannot be fixed by grepping this one.
* It is STRUCTURAL: an object written elsewhere that implements the three members satisfies it
  without importing or inheriting from anything here. That is the property an ABC would not have,
  and it is why Squid's own live-acquisition reader (fed by ``signal_zarr_frame_written``) can be
  a ``SquidAcquisitionReader`` without a dependency pointing the wrong way.
* The name is exported from the package root and listed in ``__all__``, because "Squid can import
  it" is the entire point and a name reachable only as ``squidmip.reader.X`` is a weaker promise.

NOT pinned, deliberately: that ``isinstance`` is a real check. It is a smoke test. The docstring on
the Protocol says so, and the signature test above is what actually holds the four to the contract.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest

import squidmip
from squidmip import open_reader
from squidmip.reader import (
    SquidAcquisitionReader,
    SquidMultiPageTiffReader,
    SquidOMEReader,
    SquidReader,
    SquidZarrReader,
)
from tests.writer_fixtures import WRITERS

#: The four classes ``open_reader`` can return. Named here so a fifth reader added without a
#: Protocol check fails on this list rather than passing unnoticed.
READER_CLASSES = (SquidReader, SquidMultiPageTiffReader, SquidOMEReader, SquidZarrReader)

#: The Protocol's members: what all four implement. Not a wish list -- see the intersection test.
PROTOCOL_MEMBERS = ("metadata", "read", "plane_ref")


def declared_members() -> set:
    """The Protocol's declared members, read off the class body.

    Not ``__protocol_attrs__``: that is Python 3.12+, and this package supports 3.10 (see
    ``requires-python``). The class body is the declaration on every version.
    """
    return {name for name in vars(SquidAcquisitionReader) if not name.startswith("_")}


@pytest.fixture(params=WRITERS, ids=[w[0] for w in WRITERS])
def writer_acquisition(request, tmp_path):
    """``(label, root, reader_class_name)`` for one Squid writer's tiny synthetic acquisition."""
    label, builder, reader_cls, _records_positions = request.param
    return label, builder(tmp_path / re.sub(r"\W+", "_", label)), reader_cls


# --- the four satisfy it ------------------------------------------------------------------------

def test_every_reader_satisfies_the_protocol(writer_acquisition):
    """Every writer's reader IS a ``SquidAcquisitionReader``, on a real opened acquisition."""
    label, root, reader_cls = writer_acquisition
    reader = open_reader(root)
    assert type(reader).__name__ == reader_cls, label
    assert isinstance(reader, SquidAcquisitionReader), (
        f"{label}: {type(reader).__name__} does not satisfy SquidAcquisitionReader"
    )


def test_the_six_fixtures_cover_all_four_reader_classes():
    """The parametrisation above is only worth trusting if it reaches every class."""
    covered = {cls for _label, _builder, cls, _pos in WRITERS}
    assert covered == {c.__name__ for c in READER_CLASSES}


def test_the_protocol_is_usable_as_the_annotation_open_reader_now_carries(writer_acquisition):
    """A consumer annotating against the interface gets pixels from all four, no branch anywhere.

    This is the merge case in miniature: one function, no ``isinstance``, four on-disk formats.
    """
    _label, root, _cls = writer_acquisition

    def first_plane(reader: SquidAcquisitionReader) -> np.ndarray:
        meta = reader.metadata
        region = meta["regions"][0]
        fov = meta["fovs_per_region"][region][0]
        channel = meta["channels"][0]["name"]
        return reader.read(region, fov, channel, meta["z_levels"][0])

    plane = first_plane(open_reader(root))
    assert plane.ndim == 2 and plane.dtype == np.uint16


# --- it describes what exists, not what we wish existed ----------------------------------------

def test_the_protocol_members_are_exactly_the_intersection_of_the_four():
    """Every member is on all four classes, and nothing all four share is silently omitted.

    The second half is the one with teeth: a Protocol that under-describes is a seam a consumer
    has to guess past.
    """
    for member in PROTOCOL_MEMBERS:
        assert member in declared_members(), member
        for cls in READER_CLASSES:
            assert hasattr(cls, member), f"{cls.__name__} has no {member}"

    assert declared_members() == set(PROTOCOL_MEMBERS)

    # Nothing all four share is left undeclared. ``plane_path`` is the only public name two of
    # them add, and the next test states why it stays out.
    shared = set.intersection(*({n for n in dir(cls) if not n.startswith("_")}
                                for cls in READER_CLASSES))
    assert shared == set(PROTOCOL_MEMBERS)


def test_plane_path_is_left_out_because_only_a_tiff_plane_is_a_file():
    """Two of four have ``plane_path``. A member two implementations lack would type-check a lie."""
    have = {cls.__name__ for cls in READER_CLASSES if hasattr(cls, "plane_path")}
    assert have == {"SquidReader", "SquidMultiPageTiffReader"}
    assert "plane_path" not in declared_members()


def test_the_signatures_agree_with_the_four_implementations():
    """``isinstance`` checks names, not shapes. This checks shapes.

    ``read(region, fov, channel, z, t=0)`` and ``plane_ref(region, fov, channel, z, t=0)``: the
    same parameter names in the same order with the same default, on all four. A reader that
    renamed ``t`` to ``time_point`` (which the naming law will eventually want, everywhere at
    once) must not be able to do it alone.
    """
    for name in ("read", "plane_ref"):
        want = inspect.signature(getattr(SquidAcquisitionReader, name))
        for cls in READER_CLASSES:
            got = inspect.signature(getattr(cls, name))
            assert got == want, f"{cls.__name__}.{name}{got} != the protocol's {name}{want}"

    # metadata is a read-only property on all four: computed lazily, cached, never assigned to.
    assert isinstance(SquidAcquisitionReader.metadata, property)
    for cls in READER_CLASSES:
        attr = inspect.getattr_static(cls, "metadata")
        assert isinstance(attr, property), f"{cls.__name__}.metadata is not a property"
        assert attr.fset is None, f"{cls.__name__}.metadata is settable"


# --- structural, which is why it is a Protocol and not a base class ----------------------------

def test_a_foreign_class_satisfies_it_without_inheriting_anything():
    """The Squid-side case: an in-process reader over a live acquisition, no import of ours.

    Written here as a class that inherits from ``object`` and knows nothing about this package.
    If this needed a base class or a ``register()`` call, the dependency would point from Squid
    into SquidXplorer for a type Squid is meant to consume.
    """
    class LiveAcquisitionReader:                     # what Squid could hand us, mid-run
        @property
        def metadata(self) -> dict:
            return {"regions": ["A1"], "fovs_per_region": {"A1": [0]}}

        def read(self, region, fov, channel, z, t=0):
            return np.zeros((2, 2), dtype=np.uint16)

        def plane_ref(self, region, fov, channel, z, t=0) -> tuple:
            return ("A1_0_0_ch.tiff", 0)

    assert isinstance(LiveAcquisitionReader(), SquidAcquisitionReader)
    for cls in READER_CLASSES:
        assert not issubclass(LiveAcquisitionReader, cls)   # nothing was inherited


def test_a_class_missing_a_member_does_not_satisfy_it():
    """The check is worth something only if something fails it."""
    class HalfAReader:
        @property
        def metadata(self) -> dict:
            return {}

        def read(self, region, fov, channel, z, t=0):
            return np.zeros((2, 2), dtype=np.uint16)
        # no plane_ref: a viewer could not register a plane out of this

    assert not isinstance(HalfAReader(), SquidAcquisitionReader)


# --- reachable from another repo ----------------------------------------------------------------

def test_the_name_is_exported_from_the_package_root():
    """``from squidmip import SquidAcquisitionReader``. That sentence is the deliverable."""
    assert squidmip.SquidAcquisitionReader is SquidAcquisitionReader
    assert "SquidAcquisitionReader" in squidmip.__all__
