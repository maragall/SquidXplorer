"""Channel identity + display-color resolution for Squid acquisitions.

Reconciles YAML channel names ("Fluorescence 638 nm - Penta") with filename tokens
("Fluorescence_638_nm_-_Penta"); the FILE form is the canonical key.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import yaml

# Squid's palette (software/control/_def.py): wavelengths (nm) and brightfield letters.
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

_WAVELENGTHS = ("405", "488", "561", "638", "730")
# Mirrors Squid's filename sanitization: whitespace + \ / : * ? " < > | -> "_".
_UNSAFE_CHARS = re.compile(r'[\s/\\:*?"<>|]')

# A 3-digit wavelength immediately before "nm", in any of Squid's spellings. `nm(?![A-Za-z])`
# rather than `\bnm\b`: "_" is a word character, so `\b` does not fire before "_-_Penta".
_WAVELENGTH_NM = re.compile(r"(?<!\d)(\d{3})(?!\d)[\s_-]*nm(?![A-Za-z])", re.IGNORECASE)


def normalize(name: str) -> str:
    """Convert a YAML channel name to its filename (canonical) form."""
    return _UNSAFE_CHARS.sub("_", str(name).strip())


def excitation_nm(channel_name) -> float | None:
    """The channel's excitation wavelength in nm, or ``None`` for a broadband channel."""
    text = str(channel_name).strip()
    if text.isdigit():
        return float(text)
    match = _WAVELENGTH_NM.search(text)
    return float(match.group(1)) if match else None


def fallback_color(filename_channel: str) -> str | None:
    """Best-effort color from the wavelength or brightfield letter in the channel name."""
    for wl in _WAVELENGTHS:
        if re.search(rf"(?<!\d){wl}(?!\d)", filename_channel):
            return CHANNEL_COLORS_MAP[wl]
    m = re.search(r"(?:^|_)([RGB])(?:_|$)", filename_channel)
    if m:
        return CHANNEL_COLORS_MAP.get(m.group(1))
    return None


def _extract_color(channel: dict) -> str | None:
    """v1.0+ top-level display_color, else pre-v1.0 nested camera_settings.<first cam>."""
    if not isinstance(channel, dict):
        return None
    if channel.get("display_color"):
        return channel["display_color"]
    cameras = channel.get("camera_settings")
    if isinstance(cameras, dict):
        for cam_key in sorted(cameras, key=str):   # first camera key, not a hardcoded '1'
            cam = cameras.get(cam_key)
            color = cam.get("display_color") if isinstance(cam, dict) else None
            if color:
                return color
    return None


def _extract_exposure(channel: dict):
    if not isinstance(channel, dict):
        return None
    cameras = channel.get("camera_settings")
    if isinstance(cameras, dict):
        for cam_key in sorted(cameras, key=str):
            cam = cameras.get(cam_key)
            exposure = cam.get("exposure_time_ms") if isinstance(cam, dict) else None
            if exposure is not None:
                return exposure
    return channel.get("exposure_time_ms")


def load_channel_yaml(root) -> dict:
    """Parse the channel YAML into ``{filename_name: {display_name, display_color,
    exposure_time_ms}}``; ``{}`` when neither YAML exists."""
    root = Path(root)
    path = root / "acquisition_channels.yaml"
    if not path.exists():
        path = root / "acquisition.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    out: dict = {}
    channels = data.get("channels") if isinstance(data, dict) else None
    for channel in channels or []:
        if not isinstance(channel, dict):
            continue
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
            "exposure_time_ms": _extract_exposure(channel),
        }
    return out


def resolve_channels(filename_channels, yaml_map: dict) -> list[dict]:
    """Produce the metadata `channels` list; color is YAML -> CHANNEL_COLORS_MAP -> raise."""
    resolved = []
    for name in filename_channels:
        info = yaml_map.get(name)
        color = (info["display_color"] if info else None) or fallback_color(name)
        if color is None:
            raise ValueError(
                f"Could not resolve a display color for channel {name!r}: not in the channel "
                "YAML and no wavelength / brightfield match in CHANNEL_COLORS_MAP. An "
                "unrecognized channel is refused rather than given a placeholder color."
            )
        resolved.append(
            {
                "name": name,
                "display_name": info["display_name"] if info else name,
                "display_color": color,
                "exposure_time_ms": info["exposure_time_ms"] if info else None,
                "excitation_nm": excitation_nm(name),
            }
        )
    return resolved


# -- color provenance (DisplayChannel.color_source) ----------------------------------------------
# The vocabulary is written where it is stamped: reader._expand_rgb_channels ("file"),
# _stain.attach_stain_luts ("estimated"); "reconstructed" is reserved for the overview-chroma
# expansion. Most derived first, so the least trustworthy source leads the sentence.
_COLOR_SOURCE_ORDER = ("estimated", "reconstructed", "file")
_COLOR_SOURCE_NOTES = {
    "estimated": "estimated colormap (density fit from the acquisition's overview)",
    "reconstructed": "reconstructed from the acquisition's own overview",
    "file": "file color (the plane's own RGB components)",
}


def color_sources(channels) -> list[str]:
    """The distinct ``color_source`` words of *channels*, most derived first; ``[]`` when all
    channels show their own yaml color. An unknown word passes through rather than vanishing."""
    found = {c.get("color_source") for c in (channels or [])} - {None}
    ordered = [s for s in _COLOR_SOURCE_ORDER if s in found]
    return ordered + sorted(found - set(_COLOR_SOURCE_ORDER))


def color_note(channels) -> str | None:
    """One sentence naming where the display color came from, or ``None`` for plain channels."""
    sources = color_sources(channels)
    if not sources:
        return None
    return "color: " + "; ".join(_COLOR_SOURCE_NOTES.get(s, s) for s in sources)
