"""The reader interface, given a name Squid can import."""

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

# source_id joined 2026-08-15: the identity caches key on, promoted from private `_path`.
PROTOCOL_MEMBERS = ("metadata", "read", "plane_ref", "source_id")


def declared_members() -> set:
    """The Protocol's declared members, read off the class body (not __protocol_attrs__, which is 3.12+ and this package supports 3.10)."""
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


def test_the_signatures_agree_with_the_four_implementations():
    """isinstance checks names, not shapes; this checks shapes, so a reader that renamed t to time_point could not do it alone."""
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

        @property
        def source_id(self) -> str:
            return "/live/acq"

    assert isinstance(LiveAcquisitionReader(), SquidAcquisitionReader)
    for cls in READER_CLASSES:
        assert not issubclass(LiveAcquisitionReader, cls)   # nothing was inherited

    class HalfAReader:
        @property
        def metadata(self) -> dict:
            return {}

        def read(self, region, fov, channel, z_level, time_point=0):
            return np.zeros((2, 2), dtype=np.uint16)

    assert not isinstance(HalfAReader(), SquidAcquisitionReader)


def test_the_contract_module_is_the_definition_exported_at_the_root_and_stays_import_light():
    """Squid imports squidxplorer.contract.reader across the repo boundary; the module must define the one Protocol object and pull in nothing beyond typing"""
    import ast

    from squidxplorer.contract import reader as contract

    assert contract.SquidAcquisitionReader is SquidAcquisitionReader
    assert squidxplorer.SquidAcquisitionReader is SquidAcquisitionReader
    assert "SquidAcquisitionReader" in squidxplorer.__all__
    tree = ast.parse(inspect.getsource(contract))
    mods = {alias.name for node in ast.walk(tree)
            if isinstance(node, ast.Import) for alias in node.names}
    mods |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert mods <= {"__future__", "typing", "numpy"}, mods
