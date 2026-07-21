"""Third-party stitchers, wrapped as SquidMIP **region operators** so they land in the
existing operator benchmark (``tools/benchmark.py`` / :mod:`squidmip._benchmark`) rather
than in a second, worse harness.

Why this shape. The benchmark already measures a region operator's speed (Julio's
``StageTimer``), footprint (``RSSSampler``), allocation attribution (``AllocationSampler``)
and seam quality (``overlap_ncc`` on a pair chosen from the STAGE COORDINATES before any
operator runs). An external stitcher that satisfies the region-operator contract —
``operator(reader, region, fovs, **kwargs) -> (T, C, 1, Y, X)`` plus a filled ``geometry``
dict — is therefore measured by exactly the same code, on exactly the same seam, as
``stitch`` and ``coordinate``. Nothing in the comparison is bespoke to the challenger.

The one thing that must match for the comparison to mean anything is the INPUT. Each
challenger receives the same per-FOV z-projection (``project_well``, ``mip``) of the same
single channel at the same stage positions that :func:`squidmip._stitch.stitch_region`
receives. What differs is only the registration solve and the fusion — which is the whole
question.

Registered here:

``ashlar``
    labsyspharm/ashlar (Muhlich et al., *Bioinformatics* 2022,
    https://doi.org/10.1093/bioinformatics/btac544; code
    https://github.com/labsyspharm/ashlar). Its own ``EdgeAligner`` (phase-correlation on
    neighbour pairs, permutation-derived error threshold, maximum spanning tree, then a
    linear model for the tiles it could not register) and its own ``Mosaic`` fusion
    (``utils.pastefunc_blend``). We supply only an in-memory ``Reader``/``Metadata`` pair so
    it never touches BioFormats or the disk.

Deliberately NOT registered, with the reason on record: see :data:`UNAVAILABLE`.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------------------
# What we could not run, and what it would take. No estimated numbers appear anywhere.
# --------------------------------------------------------------------------------------

UNAVAILABLE = {
    "mcmicro": {
        "what": "MCmicro (Schapiro et al., Nature Methods 2022, "
                "https://doi.org/10.1038/s41592-021-01308-y; https://mcmicro.org) is a "
                "Nextflow PIPELINE whose stitching/registration module IS ashlar. It adds "
                "no stitching algorithm of its own.",
        "needs": "nextflow (JVM) + docker or singularity for the containerised modules",
        "why_not": "neither `nextflow` nor `docker` is on PATH on this machine",
        "cost": "Nextflow is a ~40 MB JVM install (java 11 is present). Docker Desktop is "
                "multi-GB and is ruled out by the disk budget. Even installed, the "
                "stitching number it produced would be the ashlar number below plus "
                "container and workflow overhead, measuring Nextflow, not a stitcher.",
    },
    "bigstitcher": {
        "what": "BigStitcher (Horl et al., Nature Methods 2019, "
                "https://doi.org/10.1038/s41592-019-0501-0) — Fiji/ImageJ2 plugin over "
                "BigDataViewer; phase-correlation pairwise shifts plus global optimisation.",
        "needs": "a Fiji installation (ImageJ2 + BigStitcher update site), driven headless "
                "via an ImageJ macro or pyimagej/scyjava",
        "why_not": "Fiji is not installed (java 11 IS present, so the JVM half is met)",
        "cost": "Fiji is ~1.5 GB installed and pyimagej pulls a Maven-resolved ImageJ2 "
                "dependency tree of similar size. With 6 GB free and a hard 3 GB floor "
                "that does not fit alongside the fused mosaics. Additionally its native "
                "input is a BDV/HDF5 or N5 dataset, so the acquisition would have to be "
                "CONVERTED first — which the read-only-data rule forbids.",
    },
    "petakit5d": {
        "what": "PetaKit5D (Ruan et al., Nature Methods 2024, "
                "https://doi.org/10.1038/s41592-024-02475-4; "
                "https://github.com/abcucberkeley/PetaKit5D) — the Betzig-lab LLSM "
                "processing toolkit. It DOES include a stitching module "
                "(XR_matlab_stitching_wrapper).",
        "needs": "MATLAB (or the compiled PetaKit5D-standalone binaries) — the toolkit is "
                "MATLAB, not Python",
        "why_not": "no MATLAB on this machine, and the `petakit` PYTHON package that IS "
                   "importable here is a DIFFERENT piece of software: it is Julio's own "
                   "repo github.com/maragall/deconvolution, which re-implements two "
                   "PetaKit5D DECONVOLUTION algorithms (Richardson-Lucy, OTF-masked "
                   "Wiener) and contains no stitching code at all.",
        "cost": "a MATLAB licence + toolboxes, or the multi-GB PetaKit5D-standalone MCR "
                "bundle. Both are out of budget; neither can be estimated from here.",
    },
}


# --------------------------------------------------------------------------------------
# ashlar
# --------------------------------------------------------------------------------------

class _ArrayMetadata:
    """ashlar ``Metadata`` over tiles we already hold in RAM.

    Positions are in PIXELS and zero-based, which is what ashlar's own
    ``BioformatsMetadata.tile_position`` returns (it divides microns by ``pixel_size``
    before handing them over), so ashlar's ``EdgeAligner`` sees exactly the units it
    expects.
    """

    def __init__(self, tiles: np.ndarray, positions_px: np.ndarray, pixel_size_um: float):
        # tiles: (n_tiles, n_channels, Y, X)
        self._tiles = tiles
        self._positions = np.asarray(positions_px, dtype=np.float64)
        self._positions = self._positions - self._positions.min(axis=0)
        self._pixel_size = float(pixel_size_um)

    @property
    def _num_images(self) -> int:
        return int(self._tiles.shape[0])

    @property
    def num_images(self) -> int:
        return self._num_images

    @property
    def num_channels(self) -> int:
        return int(self._tiles.shape[1])

    @property
    def pixel_size(self) -> float:
        return self._pixel_size

    @property
    def pixel_dtype(self):
        return self._tiles.dtype

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def size(self) -> np.ndarray:
        return np.array(self._tiles.shape[2:], dtype=np.int64)

    def tile_position(self, i: int) -> np.ndarray:
        return self._positions[i]

    def tile_size(self, i: int) -> np.ndarray:
        return self.size

    @property
    def centers(self) -> np.ndarray:
        return self.positions + self.size / 2

    @property
    def origin(self) -> np.ndarray:
        return self.positions.min(axis=0)


class _ArrayReader:
    """ashlar ``Reader`` over the same in-RAM tiles. ``read(series, c) -> 2-D plane``."""

    def __init__(self, metadata: _ArrayMetadata):
        self.metadata = metadata
        self.path = "<memory>"

    def read(self, series: int, c: int) -> np.ndarray:
        return self.metadata._tiles[int(series), int(c)]


def ashlar_region(
    reader,
    region: str,
    fovs: Sequence[int],
    *,
    projector: str = "mip",
    registration_channel=None,
    channels: Optional[Sequence[int]] = None,
    max_shift_um: float = 30.0,
    filter_sigma: float = 0.0,
    geometry: Optional[dict] = None,
    timer=None,
    **_ignored,
) -> np.ndarray:
    """Stitch one well with **ashlar's** aligner and ashlar's fusion.

    Stage names are deliberately the same three that :func:`squidmip._stitch.stitch_region`
    reports — ``project`` / ``register`` / ``fuse`` — so the per-stage table compares like
    with like:

    ``project``
        identical code to ``stitch``: ``project_well`` per FOV. Shared, not re-implemented,
        so the comparison isolates the stitcher rather than the reader.
    ``register``
        ``ashlar.reg.EdgeAligner.run()`` — ashlar's whole solve (thresholding, pairwise
        phase correlation, spanning tree, linear model). ashlar has no separate optimise
        step, so ``stitch``'s ``register`` + ``optimize`` are the comparable pair.
    ``fuse``
        ``ashlar.reg.Mosaic.assemble_channel`` per channel per timepoint, ashlar's own
        alpha-blended paste.

    Deviations from a stock ``ashlar`` command line, all of which are recorded because a
    silently retuned competitor is not a competitor:

    * ``do_make_thumbnail=False`` — the thumbnail exists for cross-CYCLE ``LayerAligner``
      registration, which a single-cycle mosaic never uses. Leaving it on would charge
      ashlar for work its result does not depend on.
    * ``max_shift_um=30`` rather than ashlar's default 15. The measured stage step on this
      acquisition is 1.4104 mm against a 1.567 mm tile, so a real correction can be tens
      of microns; 15 um would reject good matches and make ashlar look worse than it is
      for a reason that is ours, not ashlar's. Raise/lower via the operator kwarg.
    """
    from squidmip._placement import fov_offsets_px
    from squidmip._stitch import _NullTimer, _pixel_size, _resolve_projector, _positions_yx_um
    from squidmip.projection import project_well

    timer = timer or _NullTimer()
    fovs = list(fovs)
    if not fovs:
        raise ValueError(f"region {region!r}: no FOVs to stitch.")

    meta = reader.metadata
    all_channels = [c["name"] for c in meta["channels"]]
    if channels is None:
        channels = list(range(len(all_channels)))
    channels = [int(c) for c in channels]

    pixel_size = _pixel_size(meta)
    tile_shape = tuple(int(v) for v in meta["frame_shape"])
    dtype = np.dtype(meta["dtype"])
    n_t = int(meta["n_t"])
    _op = _resolve_projector(projector)

    with timer.stage("project"):
        tiles = np.empty((len(fovs), n_t, len(channels), *tile_shape), dtype=dtype)
        for i, fov in enumerate(fovs):
            tiles[i] = project_well(reader, region, fov, reduce=_op.fn,
                                    consumes=_op.consumes)[:, channels, 0]

    stage_px = fov_offsets_px(meta["fov_positions_um"], region, fovs,
                              float(meta["pixel_size_um"]))
    positions_px = np.array([stage_px[f] for f in fovs], dtype=np.float64)

    import ashlar.reg as areg

    md = _ArrayMetadata(tiles[:, 0], positions_px, pixel_size[0])
    ash_reader = _ArrayReader(md)

    with timer.stage("register"):
        with warnings.catch_warnings():
            # ashlar warns loudly per rejected pair; the count is reported via
            # `geometry["ashlar_discarded"]` instead of 400 lines of stderr.
            warnings.simplefilter("ignore")
            aligner = areg.EdgeAligner(
                ash_reader, channel=0, max_shift=max_shift_um,
                filter_sigma=filter_sigma, do_make_thumbnail=False, verbose=False,
            )
            aligner.run()

    solved = np.asarray(aligner.positions, dtype=np.float64)
    solved = solved - solved.min(axis=0)
    h = int(np.ceil((solved[:, 0] + tile_shape[0]).max()))
    w = int(np.ceil((solved[:, 1] + tile_shape[1]).max()))
    origins = [(float(y), float(x)) for y, x in solved]

    if geometry is not None:
        geometry.update(
            fovs=list(fovs),
            offsets_px=solved - md.positions,
            origins_px=origins,
            shape=(h, w),
            pixel_size_um=pixel_size[0],
            tile_shape=tile_shape,
            stitcher="ashlar",
            ashlar_version=getattr(areg, "__version__", None) or _ashlar_version(),
            ashlar_max_shift_um=max_shift_um,
        )

    with timer.stage("fuse"), warnings.catch_warnings():
        # ashlar 1.20 calls scikit-image APIs deprecated in 0.26 (remove_small_holes'
        # area_threshold, binary_dilation); one FutureWarning per pasted tile buries the
        # table. The calls still work — this silences the noise, not an error.
        warnings.simplefilter("ignore", FutureWarning)
        out = np.zeros((n_t, len(channels), 1, h, w), dtype=dtype)
        for t in range(n_t):
            # ashlar's Mosaic reads through aligner.reader, so point that reader's tile
            # store at this timepoint's planes before assembling it.
            md._tiles = tiles[:, t]
            mosaic = areg.Mosaic(aligner, (h, w), channels=range(len(channels)),
                                 verbose=False)
            # Mosaic pastes at aligner.positions; use the zero-based copy so nothing
            # falls outside the canvas we allocated.
            saved, aligner.positions = aligner.positions, solved
            try:
                for ci in range(len(channels)):
                    out[t, ci, 0] = mosaic.assemble_channel(ci)
            finally:
                aligner.positions = saved
        md._tiles = tiles[:, 0]

    return out


def _ashlar_version() -> Optional[str]:
    try:
        from importlib.metadata import version
        return version("ashlar")
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Registration into the operator table
# --------------------------------------------------------------------------------------

def ashlar_filtered_region(reader, region, fovs, **kwargs):
    """ashlar with ``--filter-sigma 1``, its documented knob for noisy/low-contrast tiles.

    Registered as a SECOND row rather than folded into the ``ashlar`` default, because
    which one is "ashlar's number" is a real question and hiding either answer would be
    picking the flattering one. Measured on manual0's 0|1 seam, registration only:

    ===================  =======  =========  ========
    setting              dy px    dx px      seam NCC
    ===================  =======  =========  ========
    stage coordinates      0.00   1875.60      0.9430
    filter_sigma = 0       -2.11   1885.73      0.9560
    filter_sigma = 1       -6.30   1881.60      0.9716
    tilefusion (stitch)    -6.40   1882.05      0.9720
    ===================  =======  =========  ========

    ``max_shift`` is NOT the limiting knob: 15 / 30 / 100 um give bit-identical offsets at
    either sigma, so the 0.956 result is a whitening/contrast effect, not a clamp. With
    the filter on, ashlar lands within half a pixel of tilefusion on both axes.
    """
    kwargs.setdefault("filter_sigma", 1.0)
    return ashlar_region(reader, region, fovs, **kwargs)


CHALLENGERS = {"ashlar": ashlar_region, "ashlar-filtered": ashlar_filtered_region}


def register_challengers() -> list[str]:
    """Add every importable third-party stitcher to the region-operator table.

    Idempotent, and silent about the ones that are not installed: a stitcher that cannot
    be imported must show up as a MISSING ROW in the report, never as a fabricated one.
    Returns the names that were made available.
    """
    from squidmip._stitch import add_region_operator, available_region_operators

    added = []
    have = set(available_region_operators())
    for name, fn in CHALLENGERS.items():
        if name in have:
            added.append(name)
            continue
        try:
            _probe(name)
        except Exception:
            continue
        add_region_operator(name, fn)
        added.append(name)
    return added


def _probe(name: str) -> None:
    if name.startswith("ashlar"):
        import ashlar.reg  # noqa: F401
    else:
        raise KeyError(name)


def availability_report() -> str:
    """One block per stitcher we could NOT run: what it needs, why not, what it would cost."""
    lines = ["not run (no numbers reported for these, by design):"]
    for name, info in UNAVAILABLE.items():
        lines.append(f"\n  {name}")
        lines.append(f"    what   : {info['what']}")
        lines.append(f"    needs  : {info['needs']}")
        lines.append(f"    why not: {info['why_not']}")
        lines.append(f"    cost   : {info['cost']}")
    return "\n".join(lines)
