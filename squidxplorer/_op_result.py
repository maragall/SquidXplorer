"""An operator's output as a RESULT TYPE, ready to become a napari layer group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = ["OperatorResult", "RegionResultAccumulator"]


@dataclass(frozen=True)
class OperatorResult:
    """One operator's pixels for one region: the thing a layer GROUP is made of."""

    op: str
    region: str
    channels: tuple[str, ...]
    planes: tuple[np.ndarray, ...]
    bbox_um: Optional[tuple[float, float, float, float]] = None

    def plane(self, channel: str) -> np.ndarray:
        """This result's pixels for *channel*, by name: ``(Y, X)``, or ``(Nz, Y, X)`` at full depth."""
        try:
            return self.planes[self.channels.index(channel)]
        except ValueError:
            raise KeyError(
                f"{self.op!r} result for region {self.region!r} has no channel {channel!r}; "
                f"it carries {list(self.channels)}"
            ) from None


class RegionResultAccumulator:
    """Collect one operator's per-FOV output and hand back the region's :class:`OperatorResult`."""

    def __init__(self, op: str, region: str, meta: Mapping, channels: Sequence[str],
                 *, region_operator: bool = False) -> None:
        self.op = str(op)
        self.region = str(region)
        self._meta = meta
        self.channels = tuple(str(c) for c in channels)
        self._region_operator = bool(region_operator)
        self._planes: dict[int, np.ndarray] = {}
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
                f"{self.op!r}: FOV {fov} is not in region {self.region!r} "
                f"(it has {len(self._expected)} FOV(s)); refusing to place it at the origin")
        self._planes[int(fov)] = arr

    def complete(self) -> bool:
        """Is the whole region in? A region operator is complete at its first result."""
        if self._region_operator:
            return bool(self._planes)
        return bool(self._expected) and len(self._planes) >= len(self._expected)

    def result(self) -> OperatorResult:
        """The region's result. Raises unless the region is COMPLETE."""
        if not self.complete():
            raise ValueError(
                f"{self.op!r} region {self.region!r} is incomplete: "
                f"{len(self._planes)} of {len(self._expected)} FOV(s) have results; "
                f"refusing to draw a mosaic with holes in it")
        if self._region_operator:
            stack = next(iter(self._planes.values()))
            planes = tuple(np.asanyarray(stack[i]) for i in range(len(self.channels)))
        else:
            planes = tuple(self._fuse(i) for i in range(len(self.channels)))
        return OperatorResult(
            op=self.op, region=self.region, channels=self.channels, planes=planes,
            bbox_um=self._bbox(),
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
