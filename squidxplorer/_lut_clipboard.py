"""The window-side LUT helpers, and the ULTRA-MINIMAL two-button clipboard (2026-08-19).

The old clipboard chrome (the plate-side pair, per-channel pickers) was shelved on Julio's
instruction and stays shelved; the same day he asked for the minimum back: "I do want the copy
paste LUT. But ultra simple, minimal, two button logic." That minimum is :data:`CLIPBOARD` plus
:func:`copy_luts` / :func:`paste_luts`, thin wrappers over the two readers/writers that never
left:

* :func:`per_channel_luts` — read a window's per-channel look off its own napari layers (the
  loupe, the movie export, Minerva's on-screen-LUTs hop and settings snapshots all read it);
* :func:`apply_luts` — put a stored look on a window's layers (child-window LUT inheritance,
  and the paste).

"Match layers to raw" (``match_raw_contrast`` / ``MosaicLayers.match_contrast_to``) was shelved
whole the same day.

The contrast SEAM is unchanged and stays audited as one job per side (see CLAUDE.md): these
functions read and write A WINDOW'S OWN napari layers through its ``MosaicLayers``; the plate
hears about a paste through its own follow tap, never from here.
"""

from __future__ import annotations

from typing import Optional

_RAW_OP = "raw"

#: THE clipboard — one per process, shared by every window's copy/paste pair. A plain dict on
#: purpose (Julio, 2026-08-19: "ultra simple, minimal, two button logic").
CLIPBOARD: "dict[str, dict]" = {}


def copy_luts(win) -> int:
    """Copy this window's per-channel look into :data:`CLIPBOARD`. Returns channels copied."""
    luts = per_channel_luts(win)
    CLIPBOARD.clear()
    CLIPBOARD.update(luts)
    return len(luts)


def paste_luts(win) -> Optional[int]:
    """Paste :data:`CLIPBOARD` onto this window's layers. ``None`` = no mosaic here.

    PLATE PARITY is the caller's half: ``RegionViewer._paste_luts`` emits ``lutsPasted`` after
    this lands, and the plate (``PlateWindow._follow_window_luts`` ->
    ``PlateOverview.follow_channel_window``) reads the pasted window's own layers and follows
    each channel's window. The paste is the ONE event that moves the plate's contrast — a drag
    still does not — so the drift Julio named ("plate contrast is different from the window
    contrast") cannot re-open without breaking the pinned parity test.
    """
    return apply_luts(win, dict(CLIPBOARD))



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
