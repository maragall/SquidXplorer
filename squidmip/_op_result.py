"""An operator's output as a RESULT TYPE, ready to become a napari layer group.

Julio: "what if we want to see stitched AND deconvolved AND background subed. That's why we
need the toggles." And: "when I run the MIP or background sub or flatfield correction or the
stitcher or decon, like these are also reflected in the plate view and in my central viewer
and that's why I turn layers on and off."

Why this module exists
----------------------
There was no result type. Every operator emitted a bare ``(region, fov, ndarray)`` triple and
the display side had exactly one sink -- ``PlateWindow._on_push`` -> ``register_array``, the
ndviewer push path. So **no operator's pixels ever reached pane 2's napari**
:class:`~squidmip._napari_view.MosaicLayers`, and there was no before/after toggle for any of
them. That is not a bgsub bug; bgsub is just where it was noticed. The group toggle UI
(:mod:`squidmip._layer_tree`) was already built, already tested and already mounted -- it had
nothing to show because the PRODUCER side did not exist.

So this is deliberately a result type rather than a special case: the same envelope carries a
plane-op's output (accumulated per FOV), a z-reducer's, and a region operator's fused mosaic.
The same missing type is what blocks a gallery view, which wants exactly this -- "the pixels
this operator produced, for this region, per channel, with a frame".

The frame is the load-bearing part
----------------------------------
The mosaic is built with :func:`squidmip._mosaic_source.fuse_region_mosaic` and
:func:`squidmip._placement.fov_offsets_px` -- **the raw mosaic's own placement helpers**, not
a second implementation. A processed layer with its own extent or its own placement rule
would still look like a picture, and flipping the toggle would MOVE it: every difference the
user saw would be misregistration rather than the operator's effect. Two representations of
one geometry is this project's dominant defect shape, and a viewer is the worst place to
have it because the error renders as a plausible image.

``fuse_region_mosaic`` is duck-typed on ``reader.read(region, fov, channel, z, t)``, so
feeding it the operator's planes instead of the file's is all it takes to get the operator's
mosaic in the raw mosaic's exact frame. That is orchestration of the code that already
exists, which is the standing instruction for this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = ["OperatorResult", "RegionResultAccumulator"]


@dataclass(frozen=True)
class OperatorResult:
    """One operator's pixels for one region: the thing a layer GROUP is made of.

    ``op`` is the group key -- the operator, never the region. napari groups by
    ``layer.metadata["squidmip"]["op"]`` (see :class:`squidmip._napari_view.MosaicKey`), and
    "stitched AND deconvolved AND background subed" is three groups with three toggles. Keying
    by region instead would make a second region overwrite the first operator's group.
    """

    op: str
    region: str
    channels: tuple[str, ...]
    planes: tuple[np.ndarray, ...]
    bbox_um: Optional[tuple[float, float, float, float]] = None

    def plane(self, channel: str) -> np.ndarray:
        """This result's pixels for *channel*, by NAME -- ``(Y, X)``, or ``(Nz, Y, X)`` when the
        producer kept its depth (a region operator's per-plane fusion; see
        :meth:`RegionResultAccumulator.add`).

        By name and not by index on purpose: the channel ORDER at the operator is not
        guaranteed to be the channel order at the layer, and an index would resolve silently
        to the wrong colour rather than raising.
        """
        try:
            return self.planes[self.channels.index(channel)]
        except ValueError:
            raise KeyError(
                f"{self.op!r} result for region {self.region!r} has no channel {channel!r}; "
                f"it carries {list(self.channels)}"
            ) from None


class RegionResultAccumulator:
    """Collect one operator's per-FOV output and hand back the region's :class:`OperatorResult`.

    A plane-op / z-reducer emits ONE result per FOV, so a region's layer cannot be drawn until
    its FOVs are in; a region operator (stitch, coordinate) emits the fused region already and
    must not be re-placed. Both are the same type to the caller -- ``region_operator`` is the
    only switch, and it is set from the operator registry rather than sniffed from a shape.
    """

    def __init__(self, op: str, region: str, meta: Mapping, channels: Sequence[str],
                 *, region_operator: bool = False) -> None:
        self.op = str(op)
        self.region = str(region)
        self._meta = meta
        self.channels = tuple(str(c) for c in channels)
        self._region_operator = bool(region_operator)
        self._planes: dict[int, np.ndarray] = {}
        if region_operator:
            # One whole result. The FOV id it arrives under is the anchor FOV and carries no
            # placement meaning, so the accumulator does not validate it against the region.
            self._expected: list[int] = []
        else:
            self._expected = [int(f) for f in
                              ((meta.get("fovs_per_region") or {}).get(region) or [])]

    # -- collecting --------------------------------------------------------------------
    def add(self, fov: int, planes: Any) -> None:
        """Record one unit's output: ``(C, Y, X)``, or ``(C, Nz, Y, X)`` from a region operator.

        Refuses rather than reshaping. A channel-count mismatch broadcast into place, or an
        unrecognised FOV placed at the origin, produces a layer that looks like the operator's
        output and is not -- and a wrong picture in a scientific viewer is unrecoverable in a
        way a loud refusal is not.

        THE DEPTH IS ALLOWED ONLY WHERE IT CAN BE KEPT. A region operator arrives already fused,
        so its ``(C, Nz, Y, X)`` volume passes straight through :meth:`result` and reaches the
        layer with every acquired plane -- which is what makes a stitched mosaic renderable in 3D
        (IMA-277 fuses per plane; ``_workers._OperatorWorker._result_pixels`` is the producer). The
        per-FOV path re-fuses through :meth:`_fuse`, one plane per call, so a depth there would
        have to be fused ``Nz`` times over tiles this object is holding; it is refused by shape
        rather than silently indexed, and the producer sends one plane.
        """
        arr = np.asarray(planes)
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

    # -- producing ---------------------------------------------------------------------
    def result(self) -> OperatorResult:
        """The region's result. Raises unless the region is COMPLETE.

        Half a region is not a result: drawn as a layer it is a mosaic with holes, and the
        user reads holes as something the operator did. NO SILENT FAILURES -- this project has
        six confirmed instances of the opposite.
        """
        if not self.complete():
            raise ValueError(
                f"{self.op!r} region {self.region!r} is incomplete: "
                f"{len(self._planes)} of {len(self._expected)} FOV(s) have results; "
                f"refusing to draw a mosaic with holes in it")
        if self._region_operator:
            stack = next(iter(self._planes.values()))
            planes = tuple(np.asarray(stack[i]) for i in range(len(self.channels)))
        else:
            planes = tuple(self._fuse(i) for i in range(len(self.channels)))
        return OperatorResult(
            op=self.op, region=self.region, channels=self.channels, planes=planes,
            bbox_um=self._bbox(),
        )

    # -- internals ---------------------------------------------------------------------
    def _fuse(self, c_idx: int) -> np.ndarray:
        """Place this channel's FOVs with the RAW mosaic's own placement code.

        ``fuse_region_mosaic`` reads through ``reader.read(region, fov, channel, z, t)``, so
        the adapter below simply serves the operator's planes in place of the file's. Identical
        offsets, identical extent, identical decimation -- which is what makes the before/after
        toggle a comparison rather than two differently-framed pictures.
        """
        from squidmip._mosaic_source import fuse_region_mosaic

        planes = self._planes

        class _PlaneReader:
            @staticmethod
            def read(region, fov, channel, z=0, t=0):
                stack = planes.get(int(fov))
                return None if stack is None else stack[c_idx]

        fused = fuse_region_mosaic(_PlaneReader(), self._meta, self.region,
                                   self.channels[c_idx])
        if fused is None:
            # Same "not derivable, do not guess" signal the raw path gives: no stage positions
            # or no pixel size means a mosaic would be a WRONG picture, not a rough one.
            raise ValueError(
                f"{self.op!r} region {self.region!r}: the acquisition carries no stage "
                f"positions / pixel size, so its FOVs cannot be placed into a mosaic")
        return fused[0]

    def _bbox(self):
        from squidmip._mosaic_source import mosaic_bbox_um

        return mosaic_bbox_um(self._meta, self.region)
