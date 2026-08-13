"""The reader interface, given a name Squid can import.

open_reader dispatches to four duck-typed classes with no base class and no Protocol, which is
fine inside this repo but not across a repo boundary: SquidXplorer is a child node of Squid
software, and for the merge to be a no-op Squid needs a name to depend on.

Pinned: all four reader classes satisfy squidxplorer.reader.SquidAcquisitionReader; the Protocol
describes exactly what all four already do (the intersection, not a wish list — plane_path is
excluded because only two of four have one); signatures agree via inspect.signature, since
runtime_checkable only checks attribute presence; it is structural (no inheritance needed, so
Squid's own reader can satisfy it without a dependency pointing the wrong way); and the name is
exported from the package root.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest

import squidxplorer
from squidxplorer import open_reader
from squidxplorer.reader import (
    SquidAcquisitionReader,
    SquidMultiPageTiffReader,
    SquidOMEReader,
    SquidReader,
    SquidZarrReader,
)
from tests.writer_fixtures import WRITERS

READER_CLASSES = (SquidReader, SquidMultiPageTiffReader, SquidOMEReader, SquidZarrReader)

PROTOCOL_MEMBERS = ("metadata", "read", "plane_ref")


def declared_members() -> set:
    """The Protocol's declared members, read off the class body (not __protocol_attrs__, which
    is 3.12+ and this package supports 3.10)."""
    return {name for name in vars(SquidAcquisitionReader) if not name.startswith("_")}


@pytest.fixture(params=WRITERS, ids=[w[0] for w in WRITERS])
def writer_acquisition(request, tmp_path):
    """(label, root, reader_class_name) for one Squid writer's tiny synthetic acquisition."""
    label, builder, reader_cls, _records_positions = request.param
    return label, builder(tmp_path / re.sub(r"\W+", "_", label)), reader_cls


def test_every_reader_satisfies_the_protocol(writer_acquisition):
    label, root, reader_cls = writer_acquisition
    reader = open_reader(root)
    assert type(reader).__name__ == reader_cls, label
    assert isinstance(reader, SquidAcquisitionReader), (
        f"{label}: {type(reader).__name__} does not satisfy SquidAcquisitionReader"
    )


def test_the_six_fixtures_cover_all_four_reader_classes():
    covered = {cls for _label, _builder, cls, _pos in WRITERS}
    assert covered == {c.__name__ for c in READER_CLASSES}


def test_the_protocol_is_usable_as_the_annotation_open_reader_now_carries(writer_acquisition):
    """The merge case in miniature: one function, no isinstance, four on-disk formats."""
    _label, root, _cls = writer_acquisition

    def first_plane(reader: SquidAcquisitionReader) -> np.ndarray:
        meta = reader.metadata
        region = meta["regions"][0]
        fov = meta["fovs_per_region"][region][0]
        channel = meta["channels"][0]["name"]
        return reader.read(region, fov, channel, meta["z_levels"][0])

    plane = first_plane(open_reader(root))
    assert plane.ndim == 2 and plane.dtype == np.uint16


def test_the_protocol_members_are_exactly_the_intersection_of_the_four():
    """Every member is on all four classes, and nothing all four share is silently omitted."""
    for member in PROTOCOL_MEMBERS:
        assert member in declared_members(), member
        for cls in READER_CLASSES:
            assert hasattr(cls, member), f"{cls.__name__} has no {member}"

    assert declared_members() == set(PROTOCOL_MEMBERS)

    shared = set.intersection(*({n for n in dir(cls) if not n.startswith("_")}
                                for cls in READER_CLASSES))
    assert shared == set(PROTOCOL_MEMBERS)


def test_plane_path_is_left_out_because_only_a_tiff_plane_is_a_file():
    have = {cls.__name__ for cls in READER_CLASSES if hasattr(cls, "plane_path")}
    assert have == {"SquidReader", "SquidMultiPageTiffReader"}
    assert "plane_path" not in declared_members()


def test_the_signatures_agree_with_the_four_implementations():
    """isinstance checks names, not shapes; this checks shapes, so a reader that renamed t to
    time_point could not do it alone."""
    for name in ("read", "plane_ref"):
        want = inspect.signature(getattr(SquidAcquisitionReader, name))
        for cls in READER_CLASSES:
            got = inspect.signature(getattr(cls, name))
            assert got == want, f"{cls.__name__}.{name}{got} != the protocol's {name}{want}"

    assert isinstance(SquidAcquisitionReader.metadata, property)
    for cls in READER_CLASSES:
        attr = inspect.getattr_static(cls, "metadata")
        assert isinstance(attr, property), f"{cls.__name__}.metadata is not a property"
        assert attr.fset is None, f"{cls.__name__}.metadata is settable"


def test_a_foreign_class_satisfies_it_without_inheriting_anything():
    """The Squid-side case: an in-process reader over a live acquisition, no import of ours."""
    class LiveAcquisitionReader:                     # what Squid could hand us, mid-run
        @property
        def metadata(self) -> dict:
            return {"regions": ["A1"], "fovs_per_region": {"A1": [0]}}

        def read(self, region, fov, channel, z_level, time_point=0):
            return np.zeros((2, 2), dtype=np.uint16)

        def plane_ref(self, region, fov, channel, z_level, time_point=0) -> tuple:
            return ("A1_0_0_ch.tiff", 0)

    assert isinstance(LiveAcquisitionReader(), SquidAcquisitionReader)
    for cls in READER_CLASSES:
        assert not issubclass(LiveAcquisitionReader, cls)   # nothing was inherited


def test_a_class_missing_a_member_does_not_satisfy_it():
    class HalfAReader:
        @property
        def metadata(self) -> dict:
            return {}

        def read(self, region, fov, channel, z_level, time_point=0):
            return np.zeros((2, 2), dtype=np.uint16)
        # no plane_ref: a viewer could not register a plane out of this

    assert not isinstance(HalfAReader(), SquidAcquisitionReader)


def test_the_name_is_exported_from_the_package_root():
    """`from squidxplorer import SquidAcquisitionReader` is the deliverable."""
    assert squidxplorer.SquidAcquisitionReader is SquidAcquisitionReader
    assert "SquidAcquisitionReader" in squidxplorer.__all__
