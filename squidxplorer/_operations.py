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

# (Shelved 2026-08-24 with their operators: the reference, bgsub and flatfield cards. The
# survivors are exactly the runnable set — mip, decon, stitch, register — plus one NON-operator
# card, `illumination`: the profile loader/estimator STITCH rides, which outlived the shelved
# flatfield operator on purpose.)
_OPERATIONS = (
    Operation("mip", "Maximum Intensity Projection",
              "Collapse each well's z-stack to one max-intensity image; save a navigable OME-Zarr plate.",
              "_build_mip_tab"),
    Operation("stitch", "Stitch (register + fuse)",
              "Register every FOV of a well against its neighbours and fuse one seamless mosaic "
              "per well, instead of trusting the stage coordinates alone.",
              "_build_stitch_tab"),
    Operation("register", "Register FOVs (no fusion)",
              "Solve each well's per-FOV offsets from the overlaps without fusing or touching a "
              "pixel, and optionally write stitched_<folder>: a hardlinked copy of the "
              "acquisition whose coordinates.csv carries the registered positions.",
              "_build_register_tab"),
    Operation("decon", "Deconvolution (Richardson-Lucy)",
              "3-D Richardson-Lucy per FOV, vectorial PSF from this acquisition's own optics. "
              "Pick the iteration count by eye with the sweep.",
              "_build_decon_tab"),
    Operation("illumination", "Illumination profile (for stitching)",
              "Load a stored per-channel illumination profile (.npy) or estimate one live from "
              "plate tiles (the stitcher's BaSiC estimator). Stitch's read path corrects tiles "
              "with what is installed here when it covers every channel of the run.",
              "_build_illumination_tab"),
)
_OPERATIONS_BY_KEY = {op.key: op for op in _OPERATIONS}

# The operator a "save this to disk" button runs. Named, never spelled positionally.
_SAVE_OPERATOR = "mip"


def operator_layer_key(op_key: str, tab_key: Optional[str]) -> str:
    """Layer id an operator's results are filed under: bare key, or "<op>@<tab_key>" when scoped."""
    return f"{op_key}@{tab_key}" if tab_key else op_key


def operator_name(layer_key: str) -> str:
    """The REGISTRY name behind a layer key: ``"decon@tab2"`` -> ``"decon"``."""
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
