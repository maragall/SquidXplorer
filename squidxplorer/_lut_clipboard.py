"""The window-side LUT helpers that SURVIVED the clipboard's shelving (2026-08-19).

The copy/paste clipboard (``CLIPBOARD``, ``copy_luts``, ``paste_luts``, the plate-side pair and
every button that drove them) was deleted on Julio's instruction: "Shelf the LUT logic
completely. That's just adding complexity to the code for no reason." What stays here is
everything that is NOT the clipboard:

* :func:`per_channel_luts` — read a window's per-channel look off its own napari layers (the
  loupe, the movie export and settings snapshots all read it);
* :func:`apply_luts` — put a stored look on a window's layers (child-window LUT inheritance);
* :func:`match_raw_contrast` — raw -> operator layers WITHIN one window.

The contrast SEAM is unchanged and stays audited as one job per side (see CLAUDE.md): these
functions read and write A WINDOW'S OWN napari layers through its ``MosaicLayers``.
"""

from __future__ import annotations

from typing import Optional

_RAW_OP = "raw"



def per_channel_luts(win) -> "dict[str, dict]":
    """Read this window's per-channel LUTs off its OWN napari layers (the raw mosaic's)."""
    out: "dict[str, dict]" = {}
    pane = win._pane
    mosaic = getattr(pane, "mosaic", None) if pane is not None else None
    if mosaic is None:
        return out
    for c in (win._meta or {}).get("channels", []):
        name = c["name"]
        layer = mosaic.find(_RAW_OP, name)
        if layer is None:
            continue
        lut: dict = {}
        try:
            lut["clim"] = tuple(layer.contrast_limits)
        except Exception:                            # noqa: BLE001
            lut["clim"] = None
        try:
            cmap = layer.colormap
            lut["cmap"] = getattr(cmap, "name", cmap)
        except Exception:                            # noqa: BLE001
            lut["cmap"] = None
        try:
            from squidxplorer._napari_view import colormap_hue_rgb, colormap_mid_rgb

            # A stain LUT is white-topped and does not reduce to a hue; its mid stop is the
            # tint the plate can carry, so a paste still updates the plate's color.
            lut["rgb"] = colormap_hue_rgb(layer) or colormap_mid_rgb(layer)
        except Exception:                            # noqa: BLE001
            lut["rgb"] = None
        try:
            on = mosaic.channel_visible(name)
            lut["on"] = None if on is None else bool(on)
        except Exception:                            # noqa: BLE001
            lut["on"] = None
        out[name] = lut
    return out


def apply_luts(win, luts: "Optional[dict]") -> Optional[int]:
    """Put the FULL record — contrast, colormap AND channel on/off — on this window's layers.

    ``None`` = no mosaic here. The record's four keys all travel: ``clim`` through
    ``MosaicLayers.set_contrast`` (the identity model's own linked write, so peers and the
    mirror follow), ``cmap`` on the layer (the identity mirror fans it out by event), ``on``
    through ``set_channel_visible``. It used to apply clim+cmap only, so a window→window paste
    silently dropped which channels were lit. ``rgb`` is the PLATE's colour spelling and has no
    window-side inverse; a colour travels to a window as ``cmap`` or not at all.
    """
    pane = win._pane
    mosaic = getattr(pane, "mosaic", None) if pane is not None else None
    if mosaic is None:
        return None
    applied = 0
    for ch, lut in (luts or {}).items():
        layer = mosaic.find(_RAW_OP, ch)
        if layer is None:
            continue
        try:
            if lut.get("clim") is not None:
                lo, hi = lut["clim"]
                mosaic.set_contrast(ch, float(lo), float(hi))
            if lut.get("cmap") is not None:
                layer.colormap = lut["cmap"]
            applied += 1
        except Exception:                            # noqa: BLE001 - a missing channel is skipped
            continue
        if lut.get("on") is not None:
            try:
                mosaic.set_channel_visible(ch, bool(lut["on"]))
            except Exception:                        # noqa: BLE001 - visibility is best-effort
                pass
    return applied


def match_raw_contrast(win) -> None:
    """Put the RAW layer's contrast window on every operator layer of the same channel.

    Delegates to ``MosaicLayers.match_contrast_to`` — raw -> operator layers WITHIN one window,
    which is the window side of the audited one-seam contrast model.
    """
    pane = win._pane
    mosaic = getattr(pane, "mosaic", None) if pane is not None else None
    if mosaic is None:
        win._say("no mosaic here to match contrast on.")
        return
    matched = mosaic.match_contrast_to(_RAW_OP)
    if not matched:
        win._say("nothing to match — this window has no operator layers over the raw mosaic "
                 "yet. Run an operator on this view first.")
        return
    win._say(f"matched {matched} operator layer(s) to the raw contrast window.")
