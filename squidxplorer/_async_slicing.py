"""napari's async slicing (NAP-4), enabled through the ONE non-persisting channel.

Rapid zoom blocks the canvas under sync slicing: napari materialises the viewport's FOV
decodes inside the draw, on the Qt thread (Julio, live on the 900-FOV 20x set: "when I zoom
in rapidly, it's not responsive"). With ``NAPARI_ASYNC=1`` the slice computes on the layer
slicer's pool and the canvas keeps the previous rung until the response lands — coarse-first
progressive draw, upstream napari code.

Why the ENV and never ``get_settings()``: assigning any napari setting autosaves the WHOLE
model to the user's global settings.yaml (measured on this machine), and this app already
assigns ``gui_notification_level`` — while env-sourced values are excluded from every save
(measured: the yaml carries no async key after such a save). The env must be in place before
the process's first ``get_settings()`` call, which is why ``squidxplorer/__init__`` calls
:func:`configure` at import. The per-viewer ``_LayerSlicer._force_sync`` knob alone cannot
cover the zoom path: ``Layer.refresh`` consults the SETTING live and slices synchronously on
the calling thread when it is off.

``SQUIDXPLORER_SYNC_SLICING=1`` opts out; a user's own ``NAPARI_ASYNC`` wins either way.
"""

from __future__ import annotations

from typing import MutableMapping

_TRUTHY = ("1", "true", "yes", "on")


def configure(env: MutableMapping[str, str]) -> bool:
    """Set ``NAPARI_ASYNC=1`` in *env* unless opted out; return whether async is on."""
    if str(env.get("SQUIDXPLORER_SYNC_SLICING", "")).strip().lower() in _TRUTHY:
        return False
    if "NAPARI_ASYNC" in env:                    # the user's own answer wins either way
        return str(env["NAPARI_ASYNC"]).strip().lower() in _TRUTHY
    env["NAPARI_ASYNC"] = "1"
    return True
