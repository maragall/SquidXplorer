"""SquidMIP CLI (IMA-186) — run a post-processing operator over an HCS acquisition, headless.

This is the same engine the GUI drives, exposed for high-throughput/batch use: point it at a Squid
well-plate acquisition and it iterates the chosen operator (a z-reduction, e.g. MIP) over every well
of the grid-like plate and writes a navigable multiscale OME-Zarr plate. No GUI, no FIJI.

Structured after Cephla's stitcher CLI: a declarative pydantic ``ProcessParameters`` model (its field
docstrings become ``--help`` text) + ``CliApp.run`` + a thin ``run()`` that opens the reader and calls
``write_plate``. Keeping the "what to run" as data (parameters) and "how to run" as the shared engine
means a new operator is a new ``--projector`` value, not new CLI plumbing.

    squidmip <acquisition>                      # MIP every well -> <acquisition>.hcs/plate.ome.zarr
    squidmip <acquisition> --projector mip --workers 8 --output-folder /mnt/big
    squidmip <acquisition> --tiff               # also write the uncompressed per-plane TIFF export
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, field_validator
from pydantic_settings import CliApp, CliPositionalArg

logger = logging.getLogger("squidmip")


class ProcessParameters(BaseModel, use_attribute_docstrings=True):
    """Run a post-processing operator over an HCS well-plate acquisition (high-throughput, headless)."""

    input_folder: CliPositionalArg[str]
    """A Squid HCS acquisition folder on this machine (the latest Cephla acquisition format)."""

    projector: str = "mip"
    """Operator to run over every well — a z-reduction. 'mip' = maximum intensity projection.
    (Register more with squidmip.add_projector; the CLI needs no change to gain one.)"""

    output_folder: Optional[str] = None
    """Directory to receive ``<acquisition-name>.hcs/`` (plate.ome.zarr). Defaults to a sibling of
    the input acquisition. The output can be hundreds of GB on a large plate — aim it at a disk with
    room."""

    workers: Optional[int] = None
    """Projection worker threads. Default: all usable cores (the engine is memory-bandwidth-bound, so
    more workers mainly helps on cold/network storage)."""

    tiff: bool = False
    """Also write the individual per-plane TIFF export (Squid filename convention). This is a SECOND,
    uncompressed copy — roughly doubles on-disk size — so it's off by default."""

    limit: Optional[int] = None
    """Process only the first N wells — a quick SLICE of the plate (subset preview) so you can test
    the operator without committing the whole plate's compute + disk. Default: every well."""

    verbose: bool = False
    """Show debug-level logging."""

    @field_validator("limit")
    @classmethod
    def _positive_limit(cls, v):
        if v is not None and v < 1:
            raise ValueError(f"limit must be >= 1, got {v}")
        return v

    @field_validator("input_folder")
    @classmethod
    def _exists(cls, v: str) -> str:
        p = Path(v).expanduser()
        if not p.is_dir():
            raise ValueError(f"input_folder {v!r} is not an existing directory")
        return str(p.resolve())

    @field_validator("projector")
    @classmethod
    def _known_projector(cls, v: str) -> str:
        # Validate UP FRONT: otherwise the name is only resolved lazily inside project_plate, after
        # write_plate has already written an empty plate skeleton to disk, then crashes with a raw
        # traceback. A clean CLI error before any output is the safe behavior.
        from squidmip import available_projectors

        avail = available_projectors()
        if v not in avail:
            raise ValueError(f"unknown projector {v!r}; available: {sorted(avail)}")
        return v


def run(params: ProcessParameters) -> dict:
    """Open the acquisition and write the operator's OME-Zarr plate; return write_plate's manifest."""
    from squidmip import open_reader, write_plate

    reader = open_reader(params.input_folder)
    # Multi-FOV policy (IMA-191): current scope is one FOV per well; sample the first and warn.
    fpr = reader.metadata["fovs_per_region"]
    multi = sum(1 for r in fpr if len(fpr[r]) > 1)
    if multi:
        logger.warning("%d well(s) have >1 FOV — sampling one FOV per well (high-throughput "
                       "stitching not yet implemented).", multi)
    name = Path(params.input_folder).name
    out_parent = Path(params.output_folder) if params.output_folder else Path(params.input_folder).parent
    out_dir = out_parent / f"{name}.hcs"
    # Optional plate slice: process only the first N wells (a subset preview that won't cost the
    # whole plate). Order = the reader's region order (deterministic).
    regions = None
    if params.limit is not None:
        all_regions = list(reader.metadata["regions"])
        regions = all_regions[: params.limit]
        logger.info("SLICE: first %d of %d wells (%s%s)", len(regions), len(all_regions),
                    ", ".join(regions[:8]), " …" if len(regions) > 8 else "")
    logger.info("running '%s' over %s -> %s", params.projector, name, out_dir)

    # Resilient batch: a single corrupt/missing plane should NOT abort a multi-hour plate run. Log
    # and SKIP the offending well (fault isolation, opt-in via on_error), then report the count.
    skipped: list[str] = []

    def on_error(region, fov, exc):
        skipped.append(region)
        logger.warning("SKIP well %s (fov %s): %s: %s", region, fov, type(exc).__name__, exc)

    manifest = write_plate(
        reader, out_dir, projector=params.projector, workers=params.workers,
        tiff=params.tiff, on_error=on_error, regions=regions,
    )
    logger.info(
        "done: %s (%d/%d wells written, %d pyramid level(s))%s",
        manifest["plate"], manifest["n_fields_written"], manifest["n_wells"], manifest["levels"],
        f"  + TIFFs at {manifest['tiff']}" if manifest["tiff"] else "",
    )
    if skipped:
        logger.warning("%d well(s) SKIPPED due to read errors: %s",
                       len(skipped), ", ".join(skipped[:15]) + (" …" if len(skipped) > 15 else ""))
    manifest["skipped"] = skipped
    return manifest


def main(args: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if args is None else args)
    params = CliApp.run(ProcessParameters, cli_args=argv)
    logging.basicConfig(level=logging.DEBUG if params.verbose else logging.INFO)
    run(params)


if __name__ == "__main__":
    main()
