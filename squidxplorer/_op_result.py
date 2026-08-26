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
                 fovs: "Sequence[int] | None" = None,
                 windows: "Mapping[int, tuple] | None" = None) -> None:
        self.op = str(op)
        self.region = str(region)
        self.channels = tuple(str(c) for c in channels)
        self._region_operator = bool(region_operator)
        self._planes: dict[int, np.ndarray] = {}
        # Ruling z (sub-FOV decon): a windowed run's field is ``(C[, Nz], h, w)`` for its own
        # ``(r0, r1, c0, c1)`` frame window, placed at the FOV's offset PLUS the window's corner
        # (the same ``fov_offsets_px`` the whole-frame fuser uses, one placement rule), and the
        # result's footprint is the windows' union, not the region's.
        self._windows: dict[int, tuple] = {
            int(f): tuple(int(v) for v in w) for f, w in (windows or {}).items()}
        self._window_origin: tuple = (0, 0)      # (row, col) of the union in the region mosaic
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
        """Record one unit's output: ``(C, Y, X)``, or ``(C, Nz, Y, X)`` at full depth (a
        region operator's fused stack, or a per-FOV result a 3D tab asked for whole)."""
        arr = np.asanyarray(planes)
        if arr.ndim not in (3, 4):
            raise ValueError(
                f"{self.op!r} region {self.region!r} FOV {fov}: expected a (C, Y, X) or "
                f"(C, Nz, Y, X) stack, got shape {arr.shape}")
        window = self._windows.get(int(fov))
        if window is not None:
            r0, r1, c0, c1 = window
            if tuple(arr.shape[-2:]) != (r1 - r0, c1 - c0):
                raise ValueError(
                    f"{self.op!r} region {self.region!r} FOV {fov}: the run's window is "
                    f"{r1 - r0}x{c1 - c0} px but the result is {arr.shape[-2]}x{arr.shape[-1]}; "
                    "refusing to place pixels whose footprint is not the window's")
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
        elif self._windows:
            planes = [self._fuse_windows(i) for i in range(len(self.channels))]
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
        """Place this channel's FOVs with the RAW mosaic's own placement code; a stack with
        depth is fused one z plane at a time and stacked back ``(Nz, H, W)``."""
        from squidxplorer._mosaic_source import fuse_region_mosaic

        planes = self._planes
        depth = max((int(a.shape[1]) if a.ndim == 4 else 1) for a in planes.values())

        class _PlaneReader:
            @staticmethod
            def read(region, fov, channel, z_level=0, time_point=0):
                stack = planes.get(int(fov))
                if stack is None:
                    return None
                return stack[c_idx, z_level] if stack.ndim == 4 else stack[c_idx]

        fused = [fuse_region_mosaic(_PlaneReader(), self._meta, self.region,
                                    self.channels[c_idx], z_level=z) for z in range(depth)]
        if any(f is None for f in fused):
            raise ValueError(
                f"{self.op!r} region {self.region!r}: the acquisition carries no stage "
                f"positions / pixel size, so its FOVs cannot be placed into a mosaic")
        return fused[0][0] if depth == 1 else np.stack([f[0] for f in fused])

    def _fuse_windows(self, c_idx: int) -> np.ndarray:
        """Place this channel's WINDOWS: each at its FOV's offset plus the window's corner,
        later-overwrites-earlier like the preview fuser, over the windows' union only."""
        from squidxplorer._placement import fov_offsets_px

        meta = self._meta
        fovs = [f for f in self._expected if f in self._planes]
        offsets = fov_offsets_px(meta.get("fov_positions_um") or {}, self.region, fovs,
                                 meta.get("pixel_size_um"))
        tops = {f: (offsets[f][0] + self._windows[f][0], offsets[f][1] + self._windows[f][2])
                for f in fovs}
        r_min = min(t[0] for t in tops.values())
        c_min = min(t[1] for t in tops.values())
        r_max = max(tops[f][0] + (self._windows[f][1] - self._windows[f][0]) for f in fovs)
        c_max = max(tops[f][1] + (self._windows[f][3] - self._windows[f][2]) for f in fovs)
        self._window_origin = (int(r_min), int(c_min))
        depth = max((int(a.shape[1]) if a.ndim == 4 else 1) for a in self._planes.values())
        first = next(iter(self._planes.values()))
        shape = (depth, r_max - r_min, c_max - c_min) if depth > 1 else (r_max - r_min, c_max - c_min)
        out = np.zeros(shape, dtype=first.dtype)
        for f in fovs:
            plane = self._planes[f][c_idx]
            if depth > 1 and plane.ndim == 2:
                plane = plane[None]
            top, left = tops[f][0] - r_min, tops[f][1] - c_min
            h, w = plane.shape[-2:]
            out[..., top:top + h, left:left + w] = plane
        return out

    def _bbox(self):
        """The stage-µm footprint of THESE pixels."""
        from squidxplorer._mosaic_source import mosaic_bbox_um

        if self._region_operator and self._planes:
            placement = getattr(next(iter(self._planes.values())), "placement", None)
            if placement is not None:
                return placement.bbox_um
        box = mosaic_bbox_um(self._meta, self.region)
        if not self._windows or box is None:
            return box
        # The windows' union inside the scoped fields' mosaic (fused already, so the origin
        # is known); the same origin `mosaic_bbox_um` and `fov_offsets_px` share.
        px = float(self._meta["pixel_size_um"])
        r0, c0 = self._window_origin
        fovs = [f for f in self._expected if f in self._planes]
        h = max(self._fuse_span(f, 0) for f in fovs) - r0
        w = max(self._fuse_span(f, 1) for f in fovs) - c0
        x0, y0 = box[0] + c0 * px, box[1] + r0 * px
        return (x0, y0, x0 + w * px, y0 + h * px)

    def _fuse_span(self, fov: int, axis: int) -> int:
        """The far edge (row or col, region-mosaic px) of *fov*'s window."""
        from squidxplorer._placement import fov_offsets_px

        meta = self._meta
        fovs = [f for f in self._expected if f in self._planes]
        off = fov_offsets_px(meta.get("fov_positions_um") or {}, self.region, fovs,
                             meta.get("pixel_size_um"))[fov]
        r0, r1, c0, c1 = self._windows[fov]
        return off[0] + r1 if axis == 0 else off[1] + c1
