"""Third-party stitchers, wrapped as SquidXplorer region operators so they land in the
existing operator benchmark rather than a second harness.

Registered here: ``ashlar`` (labsyspharm/ashlar), via an in-memory ``Reader``/``Metadata``
pair so it never touches BioFormats or the disk. Deliberately not registered, with the
reason on record: see :data:`UNAVAILABLE`.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np


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


class _ArrayMetadata:
    """ashlar ``Metadata`` over tiles already held in RAM. Positions are in pixels,
    zero-based, matching what ashlar's own ``BioformatsMetadata.tile_position`` returns."""

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
    z_operator: str = "mip",
    registration_channel=None,
    channels: Optional[Sequence[int]] = None,
    max_shift_um: float = 30.0,
    filter_sigma: float = 0.0,
    geometry: Optional[dict] = None,
    timer=None,
    **_ignored,
) -> np.ndarray:
    """Stitch one well with ashlar's aligner and ashlar's fusion. Stage names match
    ``squidxplorer._stitch.stitch_region``'s (``project`` / ``register`` / ``fuse``) so the
    per-stage table compares like with like. ``max_shift_um`` defaults to 30, not ashlar's
    15, because the measured stage step here exceeds 15 um and would reject good matches."""
    from squidxplorer._placement import fov_offsets_px
    from squidxplorer._stitch import _NullTimer, _pixel_size, _resolve_operator, _positions_yx_um
    from squidxplorer.projection import project_well

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
    _op = _resolve_operator(z_operator)

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
            # ashlar warns loudly per rejected pair; silence it rather than bury the log.
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
        # ashlar 1.20 hits scikit-image deprecation warnings per pasted tile; noise, not an error.
        warnings.simplefilter("ignore", FutureWarning)
        out = np.zeros((n_t, len(channels), 1, h, w), dtype=dtype)
        for t in range(n_t):
            # ashlar's Mosaic reads tiles through aligner.reader; point it at this timepoint.
            md._tiles = tiles[:, t]
            mosaic = areg.Mosaic(aligner, (h, w), channels=range(len(channels)),
                                 verbose=False)
            # Mosaic pastes at aligner.positions; swap in the zero-based copy for the canvas.
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


def ashlar_filtered_region(reader, region, fovs, **kwargs):
    """ashlar with ``--filter-sigma 1``, registered as a second row rather than folded into
    the default, since which one is "ashlar's number" is a real question."""
    kwargs.setdefault("filter_sigma", 1.0)
    return ashlar_region(reader, region, fovs, **kwargs)


CHALLENGERS = {"ashlar": ashlar_region, "ashlar-filtered": ashlar_filtered_region}


def register_challengers() -> list[str]:
    """Add every importable third-party stitcher to the region-operator table. Idempotent;
    silent about ones that fail to import. Returns the names that were made available."""
    from squidxplorer._engine import add_region_operator, available_region_operators

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
