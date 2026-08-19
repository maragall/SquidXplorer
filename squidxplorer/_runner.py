"""The runner seam: how a dispatched run reaches the engine, as a DECLARED interface.

`_dispatch.run_operator_once` used to reach the engine through lazy ``import squidxplorer``
attribute lookups — an accidental monkeypatch seam — and owned both arms' bodies itself. The
seam is now a :class:`Runner` protocol shaped by what the two arms actually need: ``run_save``
returns the writer's manifest, ``run_preview`` returns ``(landed, stopped)``. The run's identity
travels as a :class:`~squidxplorer._runspec.RunSpec` — the name and kwargs, never a bound
callable — so a future runner can restore-then-run in another process (ian-stitcher's pattern).

:class:`InProcessRunner` is the ONE implementation, on purpose: the alignment doc gates any
process-pool or remote runner on a measured spawn-vs-decode cost. It resolves ``write_plate`` /
``run_plate`` lazily through the package exactly as the old dispatch code did, so the parity
tests' ``monkeypatch.setattr(squidxplorer, ...)`` still intercepts.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from squidxplorer._engine import N_FOVS_LOOP_DEFAULT
from squidxplorer._runspec import RunSpec


@runtime_checkable
class Runner(Protocol):
    """What a run executor owes the dispatch: two arms, spec in, facts out."""

    def run_save(self, reader, spec: RunSpec, *, out_dir=None, tiff: bool = False,
                 workers=None, on_well=None, on_error=None, stop=None) -> dict:
        """Persist the run; returns the writer's manifest (counting keys included)."""
        ...  # pragma: no cover - protocol declaration

    def run_preview(self, reader, spec: RunSpec, *, workers=None, on_well=None,
                    on_error=None, stop=None) -> "tuple[int, bool]":
        """Stream the run to *on_well*, writing nothing; returns ``(landed, stopped)``."""
        ...  # pragma: no cover - protocol declaration


class InProcessRunner:
    """The engine in this process — the two arm bodies `_dispatch` used to hold, verbatim."""

    def run_save(self, reader, spec: RunSpec, *, out_dir=None, tiff: bool = False,
                 workers=None, on_well=None, on_error=None, stop=None) -> dict:
        # Lazy, and through the package: the parity tests monkeypatch these on `squidxplorer`.
        import squidxplorer
        from squidxplorer import _acq_output, _fused_output

        operator = spec.operator
        operator_kwargs = spec.operator_kwargs
        regions, n_fovs = spec.regions, spec.n_fovs

        # Declaration-driven writer choice, and an explicit out_dir (the GUI's chosen folder,
        # the CLI's --out) is THE destination for every writer; beside-the-source is only the
        # default. A per-FOV intensity operator over an on-disk acquisition saves in the
        # acquisition's own format, full resolution (z-collapsing or z-keeping alike; only a run
        # owing every FOV, n_fovs=None, qualifies). A region operator's fused mosaic saves in
        # the stitcher's format (a Squid-style OME-TIFF per region, re-openable by open_reader).
        # Everything else keeps the OME-Zarr plate.
        acq_dst = _acq_output.acquisition_format_dst(reader, operator) if n_fovs is None else None
        fused_dst = (_fused_output.fused_format_dst(reader, operator)
                     if n_fovs is None or n_fovs is N_FOVS_LOOP_DEFAULT else None)
        if acq_dst is not None:
            return _acq_output.write_acquisition_planes(
                reader, operator, out_dir or acq_dst, regions=regions,
                operator_kwargs=operator_kwargs,
                workers=workers, on_well=on_well, on_error=on_error, stop=stop)
        if fused_dst is not None:
            return _fused_output.write_fused_acquisition(
                reader, operator, out_dir or fused_dst, regions=regions,
                operator_kwargs=operator_kwargs,
                workers=workers, on_well=on_well, on_error=on_error, stop=stop)
        return squidxplorer.write_plate(
            reader, out_dir, operator=operator, n_fovs=n_fovs, workers=workers, tiff=tiff,
            on_well=on_well, stop=stop, on_error=on_error, regions=regions,
            operator_kwargs=operator_kwargs)

    def run_preview(self, reader, spec: RunSpec, *, workers=None, on_well=None,
                    on_error=None, stop=None) -> "tuple[int, bool]":
        # PREVIEW: the same engine over the same arguments, writing nothing to disk.
        import squidxplorer

        stream = squidxplorer.run_plate(
            reader, operator=spec.operator, workers=workers, n_fovs=spec.n_fovs,
            on_error=on_error, regions=spec.regions, operator_kwargs=spec.operator_kwargs)
        landed, stopped = 0, False
        try:
            for region, fov, image in stream:
                if stop is not None and stop():
                    stopped = True      # deliver nothing computed after the request to stop
                    break
                landed += 1
                if on_well is not None:
                    on_well(region, fov, image)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()                 # shut the engine's pool down NOW, not at GC
        return landed, stopped
