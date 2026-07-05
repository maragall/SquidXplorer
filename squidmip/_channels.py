"""Channel identity + display-color resolution for Squid acquisitions.

Two naming worlds must be reconciled:

    YAML  (acquisition_channels.yaml):  "Fluorescence 638 nm - Penta"   (spaces, dashes)
    FILE  ({region}_{fov}_{z}_{channel}.tiff):  "Fluorescence_638_nm_-_Penta"  (underscores)

Squid builds the filename token by replacing whitespace + filesystem-unsafe chars with "_"
(dash is safe, so it is preserved). `normalize()` reproduces that, giving one canonical key
(the FILE form) that `read()` accepts and that metadata is keyed on.

Color resolution is a LAYERED FALLBACK — never hard-fail, because legacy / pre-YAML
acquisitions exist:

    1. acquisition_channels.yaml, channel-level `display_color`     (Squid v1.0+)
    2. acquisition_channels.yaml, nested camera_settings.<cam>.display_color   (pre-v1.0)
    3. CHANNEL_COLORS_MAP, matched by wavelength / brightfield letter substring
    4. DEFAULT_COLOR + a warning
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import yaml

# Authoritative Squid palette — software/control/_def.py:CHANNEL_COLORS_MAP.
# Keys are wavelengths (nm) and single-letter brightfield channels.
CHANNEL_COLORS_MAP = {
    "405": "#20ADF8",
    "488": "#1FFF00",
    "561": "#FFCF00",
    "638": "#FF0000",
    "730": "#770000",
    "R": "#FF0000",
    "G": "#1FFF00",
    "B": "#3300FF",
}
DEFAULT_COLOR = "#FFFFFF"

_WAVELENGTHS = ("405", "488", "561", "638", "730")
# Mirrors Squid's filename sanitization: whitespace + \ / : * ? " < > | -> "_".
_UNSAFE_CHARS = re.compile(r'[\s/\\:*?"<>|]')


def normalize(name: str) -> str:
    """Convert a YAML channel name to its filename (canonical) form."""
    return _UNSAFE_CHARS.sub("_", str(name).strip())


def fallback_color(filename_channel: str) -> str | None:
    """Best-effort color from the wavelength or brightfield letter in the channel name."""
    for wl in _WAVELENGTHS:
        # match the wavelength as a standalone number (not part of a longer digit run)
        if re.search(rf"(?<!\d){wl}(?!\d)", filename_channel):
            return CHANNEL_COLORS_MAP[wl]
    m = re.search(r"(?:^|_)([RGB])(?:_|$)", filename_channel)
    if m:
        return CHANNEL_COLORS_MAP.get(m.group(1))
    return None


def _extract_color(channel: dict) -> str | None:
    """v1.0+ top-level display_color, else pre-v1.0 nested camera_settings.<first cam>."""
    if channel.get("display_color"):
        return channel["display_color"]
    cameras = channel.get("camera_settings") or {}
    for cam_key in sorted(cameras):  # first camera key, not a hardcoded '1'
        color = (cameras[cam_key] or {}).get("display_color")
        if color:
            return color
    return None


def _extract_exposure(channel: dict):
    cameras = channel.get("camera_settings") or {}
    for cam_key in sorted(cameras):
        exposure = (cameras[cam_key] or {}).get("exposure_time_ms")
        if exposure is not None:
            return exposure
    return channel.get("exposure_time_ms")


def load_channel_yaml(root) -> dict:
    """Parse acquisition_channels.yaml into {filename_form_name: {display_name, display_color, ex}}.

    Returns {} when the file is absent (legacy acquisitions) so resolution falls back cleanly.
    """
    path = Path(root) / "acquisition_channels.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    out: dict = {}
    for channel in data.get("channels") or []:
        name = channel.get("name")
        if not name:
            continue
        key = normalize(name)
        if key in out:
            warnings.warn(
                f"Channel name collision after normalization: {name!r} maps to {key!r}, "
                "which is already present; keeping the first entry."
            )
            continue
        out[key] = {
            "display_name": name,
            "display_color": _extract_color(channel),
            "ex": _extract_exposure(channel),
        }
    return out


def resolve_channels(filename_channels, yaml_map: dict) -> list[dict]:
    """Produce the metadata `channels` list, keyed on the canonical (filename) name.

    Each entry: {name, display_name, display_color, ex}. Color uses the layered fallback.
    """
    resolved = []
    for name in filename_channels:
        info = yaml_map.get(name)
        if info is not None:
            color = info["display_color"] or fallback_color(name) or DEFAULT_COLOR
            resolved.append(
                {
                    "name": name,
                    "display_name": info["display_name"],
                    "display_color": color,
                    "ex": info["ex"],
                }
            )
        else:
            color = fallback_color(name)
            if color is None:
                warnings.warn(f"No display color found for channel {name!r}; using {DEFAULT_COLOR}.")
                color = DEFAULT_COLOR
            resolved.append(
                {"name": name, "display_name": name, "display_color": color, "ex": None}
            )
    return resolved
