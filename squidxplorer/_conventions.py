"""Named coordinate conventions: the two FOV-box frames and the two pixel pitches.

Both pairs are doc-guarded facts elsewhere (CLAUDE.md "The FOV walk", "TWO pitches exist") and
each past mix-up rendered as a plausible image: the box conventions are half a frame apart,
measured 195.9 um on the 40x AF-sweep set, and the pitch mix-up drew a 1.99 z:xy aspect where it
should be 1.00. These types make crossing either seam a NAMED act instead of an accident.

Qt-free, numpy-free: importable from any module including ``_bricks``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TopLeftBoxUm:
    """A stage-um FOV box under the TOP-LEFT convention: the recorded stage position is the
    frame's top-left corner. The mosaic path's convention (``mosaic_fov_bboxes_um``); reading a
    CENTRE-convention box here is the measured 195.9 um half-frame shear this type prevents.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def bbox(self) -> tuple[float, float, float, float]:
        """The plain ``(x0, y0, x1, y1)`` tuple for the drawing code."""
        return (self.x0, self.y0, self.x1, self.y1)

    def to_center(self) -> "CenterBoxUm":
        """The SAME physical box, re-encoded centre-style. A named crossing, never an accident."""
        w, h = self.x1 - self.x0, self.y1 - self.y0
        return CenterBoxUm(cx=self.x0 + w / 2.0, cy=self.y0 + h / 2.0, w=w, h=h)

    # Unpacking a top-left box yields exactly its own bbox: no convention can be crossed by it.
    def __iter__(self):
        return iter(self.bbox())

    def __len__(self) -> int:
        return 4

    def __getitem__(self, i):
        return self.bbox()[i]


@dataclass(frozen=True)
class CenterBoxUm:
    """A stage-um FOV box under the CENTRE convention: the recorded stage position is the
    frame's centre. The plate-ladder / NGFF-translation convention (``_tilesource.fov_bboxes_um``,
    ``_output.field_origin_um``); it sits half a frame, measured 195.9 um, from the top-left one.
    """

    cx: float
    cy: float
    w: float
    h: float

    def bbox(self) -> tuple[float, float, float, float]:
        """The plain ``(x0, y0, x1, y1)`` tuple for the drawing code."""
        return (self.cx - self.w / 2.0, self.cy - self.h / 2.0,
                self.cx + self.w / 2.0, self.cy + self.h / 2.0)

    def to_top_left(self) -> TopLeftBoxUm:
        """The SAME physical box, re-encoded corner-style. A named crossing, never an accident."""
        return TopLeftBoxUm(*self.bbox())

    def _refuse(self) -> TypeError:
        return TypeError(
            "CenterBoxUm does not unpack: its fields are (cx, cy, w, h), and reading them as "
            "(x0, y0, x1, y1) is the half-frame convention mix-up, measured 195.9 um, this type "
            "exists to prevent. Cross by name: .bbox() or .to_top_left().")

    def __iter__(self):
        raise self._refuse()

    def __getitem__(self, i):
        raise self._refuse()


@dataclass(frozen=True)
class _PitchUm:
    """A micrometres-per-pixel value that refuses to be a bare number: ``.um`` is the one door."""

    um: float

    def __post_init__(self):
        u = float(self.um)
        if not (math.isfinite(u) and u > 0):
            raise ValueError(
                f"{type(self).__name__} needs a finite positive micrometres-per-pixel, "
                f"got {self.um!r}.")

    def px_from_um(self, um: float) -> float:
        """How many of THIS pitch's pixels a micrometre span covers."""
        return float(um) / float(self.um)

    def um_from_px(self, px: float) -> float:
        """The micrometre span of a pixel count at THIS pitch."""
        return float(px) * float(self.um)

    def _bare(self) -> TypeError:
        return TypeError(
            f"{type(self).__name__} is not a bare number: read .um to use its value. The "
            f"acquisition's pitch and a displayed layer's pitch differ by the fuse decimation "
            f"(measured 2x on the 10x set), and silently mixing them drew a 1.99 z:xy aspect "
            f"where it should be 1.00.")

    def __float__(self):
        raise self._bare()

    def __add__(self, other):
        raise self._bare()

    __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __add__
    __truediv__ = __rtruediv__ = __add__

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.um!r} um/px; read .um)"


@dataclass(frozen=True)
class AcqPitchUm(_PitchUm):
    """The ACQUISITION's pitch: the unit of level-0 mosaic pixels, what the reader's planes have.
    NOT what a fused layer displays; that is :class:`DisplayPitchUm` (CLAUDE.md, "TWO pitches").
    """


@dataclass(frozen=True)
class DisplayPitchUm(_PitchUm):
    """A DISPLAYED layer's pitch: what its own ``layer.scale`` records after fuse decimation.
    NOT the acquisition's ``pixel_size_um``; that is :class:`AcqPitchUm` (CLAUDE.md, "TWO pitches").
    """


def acq_um(pitch) -> float:
    """The acquisition pitch as a bare float, for a contract that counts LEVEL-0 pixels.

    Accepts :class:`AcqPitchUm` or a bare number (legacy callers, and ``meta['pixel_size_um']``
    stays a float). A :class:`DisplayPitchUm` is REFUSED BY NAME: counted at the displayed pitch,
    the shipped 2048 px one-texture clamp becomes a 4096 px, 16-brick, 4x read.
    """
    if isinstance(pitch, AcqPitchUm):
        return float(pitch.um)
    if isinstance(pitch, DisplayPitchUm):
        raise TypeError(
            "DisplayPitchUm where the ACQUISITION pitch is required: this contract counts "
            "level-0 acquisition pixels (what the reader's planes and fov_offsets_px are in). "
            "Pass the acquisition's own pixel_size_um as AcqPitchUm, never layer.scale.")
    return float(pitch)


def display_um(pitch) -> float:
    """The displayed pitch as a bare float, for a contract that renders a LAYER's own pixels.

    Accepts :class:`DisplayPitchUm` or a bare number (legacy callers). An :class:`AcqPitchUm` is
    REFUSED BY NAME: pushed for decimated pixels it renders the volume 2x too small in y and x.
    """
    if isinstance(pitch, DisplayPitchUm):
        return float(pitch.um)
    if isinstance(pitch, AcqPitchUm):
        raise TypeError(
            "AcqPitchUm where the DISPLAYED pitch is required: this contract renders a layer's "
            "own pixels, whose pitch is what that layer's scale records after fuse decimation. "
            "Read it off layer.scale as DisplayPitchUm, never off meta['pixel_size_um'].")
    return float(pitch)
