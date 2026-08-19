"""The operator identity model: which post-processing operators exist, what they are called,
where their results are filed, and the plate's ordered layer stack over those identities.
A card is presentation, the engine is capability. No Qt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from squidxplorer._engine import runnable_operators as _runnable_operators


@dataclass(frozen=True)
class Operation:
    """One post-processing operator's card: key, label, blurb, and its tab-builder method name."""
    key: str
    label: str
    blurb: str
    build_tab: str        # name of the PlateWindow method that builds this operator's UI tab

    @property
    def runnable(self) -> bool:
        """Can the ENGINE run this key, as opposed to the card merely existing?"""
        return self.key in runnable_operators()

_OPERATIONS = (
    Operation("mip", "Maximum Intensity Projection",
              "Collapse each well's z-stack to one max-intensity image; save a navigable OME-Zarr plate.",
              "_build_mip_tab"),
    Operation("reference", "Reference plane (best focus)",
              "Keep each well's sharpest z-plane (Tenengrad) instead of combining them; save a "
              "navigable OME-Zarr plate.",
              "_build_reference_tab"),
    Operation("stitch", "Stitch (register + fuse)",
              "Register every FOV of a well against its neighbours and fuse one seamless mosaic "
              "per well, instead of trusting the stage coordinates alone.",
              "_build_stitch_tab"),
    Operation("register", "Register FOVs (no fusion)",
              "Solve each well's per-FOV offsets from the overlaps without fusing or touching a "
              "pixel, and optionally write stitched_<folder>: a hardlinked copy of the "
              "acquisition whose coordinates.csv carries the registered positions.",
              "_build_register_tab"),
    # Plane-ops keep z at full depth, so they get _build_plane_op_tab (preview only).
    Operation("decon", "Deconvolution (Richardson-Lucy)",
              "Sharpen against a vectorial PSF computed from this acquisition's own optics (NA, "
              "emission wavelength, pixel size, z-step) -- not an assumed Gaussian. Richardson-Lucy "
              "is semi-convergent, so the iteration count is chosen by eye against a turbo x-z / "
              "y-z view rather than defaulted.",
              "_build_decon_tab"),
    Operation("bgsub", "Background subtraction",
              "Remove the smooth out-of-focus haze from every plane with a rolling ball (ImageJ's "
              "algorithm). A LAYER: the raw is untouched on disk and one toggle away.",
              "_build_bgsub_tab"),
    Operation("flatfield", "Flat-field correction",
              "Divide out the objective's illumination profile so the corners match the centre. "
              "Needs an illumination profile (.npy) from the stitcher or estimated from the plate.",
              "_build_flatfield_tab"),
)
_OPERATIONS_BY_KEY = {op.key: op for op in _OPERATIONS}

# The operator a "save this to disk" button runs. Named, never spelled positionally.
_SAVE_OPERATOR = "mip"


def operator_layer_key(op_key: str, tab_key: Optional[str]) -> str:
    """Layer id an operator's results are filed under: bare key, or "<op>@<tab_key>" when scoped."""
    return f"{op_key}@{tab_key}" if tab_key else op_key


def operator_name(layer_key: str) -> str:
    """The REGISTRY name behind a layer key: ``"spot@tab2"`` -> ``"spot"``."""
    return str(layer_key).split("@", 1)[0]


def result_kind(layer_key: str) -> str:
    """What an operator's pixels MEAN: "intensity"/"labels"; unregistered keys answer "intensity"."""
    from squidxplorer._engine import _OPERATORS

    op = _OPERATORS.get(operator_name(layer_key))
    return op.produces if op is not None else "intensity"


#: Every runnable operator, re-exported from the ENGINE — never derived from ``_OPERATIONS``.
runnable_operators = _runnable_operators


def operator_label(key: str) -> str:
    """Human label for an operator: its card's if it has one, else the registry name itself."""
    op = _OPERATIONS_BY_KEY.get(key)
    return op.label if op is not None else key


def _action_label(key: str, operator_kwargs: Optional[dict] = None) -> str:
    """What the console calls one action: ``decon(sigma=2.0)``. Delegates to ``Recipe.label``."""
    from squidxplorer._recipe import Recipe

    return Recipe.operator(str(key), **(operator_kwargs or {})).label()


@dataclass
class Layer:
    key: str          # stable id ("raw", "mip", "reference", ...)
    label: str
    enabled: bool = True


class OperationStack:
    """The plate's ordered, toggleable layer stack: the topmost ENABLED layer is what the plate
    renders. Base 'raw' is the floor — never disabled, never moved, never removed."""

    def __init__(self) -> None:
        self._layers: list[Layer] = [Layer("raw", "raw", True)]

    def add(self, key: str, label: str) -> None:
        """Add (or re-add) an operation layer on top, enabled. Re-adding moves it to the top."""
        self._layers = [ly for ly in self._layers if ly.key != key]
        self._layers.append(Layer(key, label, True))

    def toggle(self, key: str, enabled: bool) -> bool:
        """Enable/disable a layer; the base ('raw') can never be disabled."""
        if key == "raw":
            return True
        for ly in self._layers:
            if ly.key == key:
                ly.enabled = enabled
                return ly.enabled
        return False

    def move(self, key: str, delta: int) -> None:
        """Reorder a layer by +/- steps; the base ('raw') never moves off the bottom."""
        if key == "raw":
            return
        idx = next((i for i, ly in enumerate(self._layers) if ly.key == key), None)
        if idx is None:
            return
        floor = 1 if self._layers and self._layers[0].key == "raw" else 0
        new = max(floor, min(len(self._layers) - 1, idx + delta))
        if new != idx:
            self._layers.insert(new, self._layers.pop(idx))

    def top_enabled(self) -> Optional[Layer]:
        """The topmost enabled layer (what the plate renders), or None if all are off."""
        for ly in reversed(self._layers):
            if ly.enabled:
                return ly
        return None

    def layers(self) -> list[Layer]:
        return list(self._layers)

    def reset(self) -> None:
        self._layers = [Layer("raw", "raw", True)]
