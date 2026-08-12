"""The operator card registry: which post-processing operators exist, what they are called,
and where their results are filed. A card is presentation, the engine is capability. No Qt."""

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
    # NOT an operator: an export hand-off. Non-runnable because nobody registered "minerva".
    Operation("minerva", "Open in Minerva Author",
              "Export the selected FOVs to Minerva-ingestable OME-TIFFs and open Minerva Author on them.",
              "_build_minerva_tab"),
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

# Roadmap cards shown under "TO BE ADDED", as (label, blurb).
_TO_BE_ADDED: list = []


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
