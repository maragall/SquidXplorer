"""Collect one operator's per-FOV output into the region's self-describing result."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from squidxplorer._address import Extent

__all__ = ["RegionResultAccumulator"]


class RegionResultAccumulator:
    """Collect one operator's per-FOV output and hand back the region's
    :class:`~squidxplorer._result.Result`.

    ``op`` names the accumulator, not the result: it travels as a parameter of the delivery
    calls, where it already is.
    """

    def __init__(self, op: str, region: str, meta: Mapping, channels: Sequence[str],
                 *, region_operator: bool = False,
                 fovs: "Sequence[int] | None" = None) -> None:
        self.op = str(op)
        self.region = str(region)
        self.channels = tuple(str(c) for c in channels)
        self._region_operator = bool(region_operator)
        self._planes: dict[int, np.ndarray] = {}
        # THE RUN'S OWN SCOPE (Julio, 2026-08-25, "Can't run decon sub FOV?"): an ROI preview
        # runs `{region: [fov, ...]}` and owes exactly those fields. The books used to owe every
        # FOV of the region off the metadata, so a scoped run was refused as "1 of 9 FOV(s)"
        # and reported as fields that could not be read, which was false. The scoped list
        # replaces the region's in the meta this accumulator places and measures with, so the
        # fuser and the bbox are the raw mosaic's own code over the run's own fields.
        if fovs is not None and not region_operator:
            per_region = dict(meta.get("fovs_per_region") or {})
            per_region[str(region)] = [int(f) for f in fovs]
            meta = {**dict(meta), "fovs_per_region": per_region}
        self._meta = meta
        if region_operator:
            self._expected: list[int] = []
        else:
            self._expected = [int(f) for f in
                              ((meta.get("fovs_per_region") or {}).get(region) or [])]

    def add(self, fov: int, planes: Any) -> None:
        """Record one unit's output: ``(C, Y, X)``, or ``(C, Nz, Y, X)`` from a region operator."""
        arr = np.asanyarray(planes)
        allowed = (3, 4) if self._region_operator else (3,)
        if arr.ndim not in allowed:
            expected = ("a (C, Y, X) or (C, Nz, Y, X) stack" if self._region_operator
                        else "a (C, Y, X) stack")
            raise ValueError(
                f"{self.op!r} region {self.region!r} FOV {fov}: expected {expected}, "
                f"got shape {arr.shape}")
        if arr.shape[0] != len(self.channels):
            raise ValueError(
                f"{self.op!r} region {self.region!r} FOV {fov}: result has {arr.shape[0]} "
                f"channel(s) but the acquisition has {len(self.channels)} "
                f"({list(self.channels)}); refusing to guess which is which")
        if not self._region_operator and int(fov) not in self._expected:
            raise ValueError(
                f"{self.op!r}: FOV {fov} is not in this run's scope of region {self.region!r} "
                f"({len(self._expected)} FOV(s)); refusing to place it at the origin")
        self._planes[int(fov)] = arr

    def complete(self) -> bool:
        """Is the whole region in? A region operator is complete at its first result."""
        if self._region_operator:
            return bool(self._planes)
        return bool(self._expected) and len(self._planes) >= len(self._expected)

    def result(self):
        """The region's :class:`~squidxplorer._result.Result`. Raises unless the region is
        COMPLETE and the acquisition declares a pixel size — a result that cannot say its own
        scale is not self-describing, and inventing one is exactly the plausible-and-wrong
        guess this codebase refuses."""
        from squidxplorer._operations import result_kind
        from squidxplorer._result import Result

        if not self.complete():
            raise ValueError(
                f"{self.op!r} region {self.region!r} is incomplete: "
                f"{len(self._planes)} of {len(self._expected)} FOV(s) have results; "
                f"refusing to draw a mosaic with holes in it")
        pixel_size_um = (self._meta or {}).get("pixel_size_um")
        if not pixel_size_um:
            raise ValueError(
                f"{self.op!r} region {self.region!r}: this acquisition declares no pixel size, "
                f"so the result cannot declare its scale and will not be drawn as a layer")
        if self._region_operator:
            stack = next(iter(self._planes.values()))
            planes = [np.asanyarray(stack[i]) for i in range(len(self.channels))]
        else:
            planes = [self._fuse(i) for i in range(len(self.channels))]
        if not planes:
            raise ValueError(
                f"{self.op!r} region {self.region!r}: the result carries no planes to show")
        first = planes[0]
        # z_depth from the pixels, which is unambiguous HERE and only here: the channel axis is
        # already split off, so a 3-D plane's leading axis can only be z. The general
        # (C, Z, Y, X) / (C, Y, X) ambiguity _result.Result.of refuses to guess at does not
        # arise once the channel axis is gone.
        z_depth = int(first.shape[0]) if int(getattr(first, "ndim", 2)) >= 3 else 1
        return Result.of(
            Extent(region_id=self.region, bbox_um=self._bbox()), planes,
            channels=self.channels, z_depth=z_depth,
            pixel_size_um=float(pixel_size_um), dtype=first.dtype,
            kind=result_kind(self.op),
        )

    def _fuse(self, c_idx: int) -> np.ndarray:
        """Place this channel's FOVs with the RAW mosaic's own placement code."""
        from squidxplorer._mosaic_source import fuse_region_mosaic

        planes = self._planes

        class _PlaneReader:
            @staticmethod
            def read(region, fov, channel, z_level=0, time_point=0):
                stack = planes.get(int(fov))
                return None if stack is None else stack[c_idx]

        fused = fuse_region_mosaic(_PlaneReader(), self._meta, self.region,
                                   self.channels[c_idx])
        if fused is None:
            raise ValueError(
                f"{self.op!r} region {self.region!r}: the acquisition carries no stage "
                f"positions / pixel size, so its FOVs cannot be placed into a mosaic")
        return fused[0]

    def _bbox(self):
        """The stage-µm footprint of THESE pixels."""
        from squidxplorer._mosaic_source import mosaic_bbox_um

        if self._region_operator and self._planes:
            placement = getattr(next(iter(self._planes.values())), "placement", None)
            if placement is not None:
                return placement.bbox_um
        return mosaic_bbox_um(self._meta, self.region)
