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

    n_fovs: Optional[int] = 1
    """FOVs to project per well. 1 (default) keeps the historical one-FOV-per-well behaviour.
    Pass 0 for EVERY FOV in every well (the multi-FOV mosaic, IMA-187) — note this multiplies
    both compute and output size by the FOV count, so a 36-FOV plate is ~36x the work."""

    limit: Optional[int] = None
    """Process only the first N wells — a quick SLICE of the plate (subset preview) so you can test
    the operator without committing the whole plate's compute + disk. Default: every well."""

    odon: bool = False
    """Also write an Odon samplesheet next to the plate and launch Odon on it (IMA-212).

    Odon is a separately-installed GPL-3 desktop viewer — SquidMIP never bundles it. Install it
    from https://github.com/alexcoulton/odon/releases, or set $ODON_BIN. Note Odon has no
    well-plate model, so it shows the fields as a flat mosaic, and it ignores our channel colors."""

    verbose: bool = False
    """Show debug-level logging."""

    @field_validator("n_fovs")
    @classmethod
    def _n_fovs(cls, v):
        # 0 is the CLI spelling of "all". pydantic-settings maps flags to scalars, so a
        # sentinel int is cleaner here than accepting the literal string "all" or None.
        if v is not None and v < 0:
            raise ValueError(f"n_fovs must be >= 0 (0 = every FOV), got {v}")
        return v

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
    """Open the acquisition and write the operator's OME-Zarr plate; return write_plate's manifest.

    The operator run itself goes through the SHARED command layer (``squidmip._command``) — the
    exact surface the GUI drives — so "it works from the CLI" and "the button works" stop being
    two separate questions. This function keeps only what is genuinely CLI-shaped around that one
    command: the wellplate-format scope guard, the multi-FOV warning, the ``--limit`` plate slice
    (expressed as the command's explicit ``regions`` list), and the Odon hand-off, which is a
    post-write launch of a separate program and not an operator at all.
    """
    from squidmip import open_reader
    from squidmip._command import CommandBus, EngineExecutor, RunOperator

    reader = open_reader(params.input_folder)
    # Scope guard: 1536-well plates only for now (not a general product yet). Fail loud, before any write.
    fmt = str(reader.metadata.get("wellplate_format", ""))
    if not any(s in fmt for s in ("384", "1536")):
        raise SystemExit(f"squidmip currently supports 384- and 1536-well plates (got {fmt or 'unknown'!r}).")
    # Multi-FOV policy (IMA-187): n_fovs=0 on the CLI means "every FOV"; anything else is an
    # explicit count. Only warn about discarding FOVs when we are actually discarding them.
    fpr = reader.metadata["fovs_per_region"]
    n_fovs = None if params.n_fovs == 0 else params.n_fovs
    multi = sum(1 for r in fpr if len(fpr[r]) > 1)
    if multi and n_fovs is not None:
        logger.warning("%d well(s) have >1 FOV — projecting %d per well; pass --n-fovs 0 to "
                       "project every FOV.", multi, n_fovs)
    elif n_fovs is None:
        total = sum(len(v) for v in fpr.values())
        logger.info("projecting ALL %d FOV(s) across %d well(s)", total, len(fpr))
    name = Path(params.input_folder).name
    out_parent = (Path(params.output_folder).expanduser() if params.output_folder
                  else Path(params.input_folder).parent)
    out_dir = out_parent / f"{name}.hcs"
    # Optional plate slice: process only the first N wells (a subset preview that won't cost the
    # whole plate). Order = the reader's region order (deterministic). This is the command layer's
    # explicit-``regions`` path — the ONE way a subset is expressed on either surface.
    regions = None
    if params.limit is not None:
        all_regions = list(reader.metadata["regions"])
        regions = all_regions[: params.limit]
        logger.info("SLICE: first %d of %d wells (%s%s)", len(regions), len(all_regions),
                    ", ".join(regions[:8]), " …" if len(regions) > 8 else "")
    logger.info("running '%s' over %s -> %s", params.projector, name, out_dir)

    # Drive the SHARED command surface. The reader is already open, so hand it to the executor
    # rather than making it re-open the folder. A refusal comes back as a value with a code — the
    # CLI turns it into a clean SystemExit instead of a traceback, the same failure the GUI shows
    # as a status-line sentence.
    bus = CommandBus(EngineExecutor(params.input_folder, reader=reader))
    result = bus.execute(RunOperator(
        operator=params.projector, regions=regions, save=True,
        output_folder=str(out_parent), n_fovs=n_fovs, workers=params.workers, tiff=params.tiff,
    ))
    if not result.ok:
        raise SystemExit(f"{params.projector}: {result.message}")
    manifest = dict(result.data["manifest"])
    skipped = list(result.data.get("skipped") or [])
    logger.info(
        "done: %s (%d/%d wells written, %d pyramid level(s))%s",
        manifest["plate"], manifest["n_fields_written"], manifest["n_wells"], manifest["levels"],
        f"  + TIFFs at {manifest['tiff']}" if manifest["tiff"] else "",
    )
    if skipped:
        logger.warning("%d well(s) SKIPPED due to read errors: %s",
                       len(skipped), ", ".join(skipped[:15]) + (" …" if len(skipped) > 15 else ""))
    manifest["skipped"] = skipped

    # IMA-212: hand the finished plate to Odon. Deliberately AFTER the plate is fully written
    # and recorded in the manifest, so a missing binary costs the user nothing — the output is
    # already on disk and complete. The samplesheet is derived by walking that output, not from
    # `regions`/`n_fovs`, so it cannot disagree with what was actually written.
    if params.odon:
        from squidmip._odon import launch_odon, write_samplesheet

        samplesheet = write_samplesheet(out_dir)
        manifest["odon_samplesheet"] = str(samplesheet)
        try:
            launch_odon(samplesheet)
        except FileNotFoundError as exc:
            raise SystemExit(f"{exc}\n\nThe plate itself is written: {manifest['plate']}") from exc

    return manifest


def main(args: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if args is None else args)
    params = CliApp.run(ProcessParameters, cli_args=argv)
    logging.basicConfig(level=logging.DEBUG if params.verbose else logging.INFO)
    run(params)


if __name__ == "__main__":
    main()
