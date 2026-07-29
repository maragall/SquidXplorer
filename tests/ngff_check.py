"""Back-compat shim. The real validator is ``squidmip.contract.validate``.

This module used to hold the ``ome-zarr-models`` pydantic pass on its own. It was promoted into
the package (gap 5, 2026-07-29) for one reason: living in ``tests/`` it could only ever be pointed
at a plate we had just written, and the thing a user actually needs is to validate a plate they
were HANDED:

    python -m squidmip.contract.validate /path/to/plate.ome.zarr

The promoted version still runs OME's official schema, and adds the half OME's schema knows
nothing about: SquidMIP's own contract (``docs/plate-contract.md``), with stable-contract
violations as ERRORS and missing optional sidecars as WARNINGS. ``ome-zarr-models`` stays a
``[test]`` extra; when it is absent the structural checks still run and the skip is reported.

The name below is kept because five test modules import it.
"""

from squidmip.contract.validate import assert_valid_ngff_plate  # noqa: F401
