"""Channel identity + display-color resolution for Squid acquisitions.

Two naming worlds must be reconciled:

    YAML  (acquisition_channels.yaml):  "Fluorescence 638 nm - Penta"   (spaces, dashes)
    FILE  ({region}_{fov}_{z}_{channel}.tiff):  "Fluorescence_638_nm_-_Penta"  (underscores)

Squid builds the filename token by replacing whitespace + filesystem-unsafe chars with "_"
(dash is safe, so it is preserved). `normalize()` reproduces that, giving one canonical key
(the FILE form) that `read()` accepts and that metadata is keyed on.

Color resolution is a LAYERED FALLBACK, but an UNRECOGNIZED channel is a hard error — a
silent white default would let a mislabeled channel through, which for a scientific tool is
worse than stopping:

    1. channel YAML, channel-level `display_color`                (Squid v1.0+)
    2. channel YAML, nested camera_settings.<cam>.display_color   (pre-v1.0)
    3. CHANNEL_COLORS_MAP, matched by wavelength / brightfield letter substring
    4. raise ValueError, naming the channel (no white default)

The channel YAML is read from ``acquisition_channels.yaml`` (dedicated snapshot) or, if that
is absent, from the ``channels:`` block of ``acquisition.yaml`` (which carries the same data).

THE EXCITATION WAVELENGTH IS PART OF THE CHANNEL RECORD (2026-08-05)
--------------------------------------------------------------------
A channel's wavelength is a property OF THE CHANNEL, and this module is where channels are
parsed — so :func:`excitation_nm` reads it here and :func:`resolve_channels` puts it on the
record, beside the colour that is resolved from the very same number.

It is read from the NAME because that is the only place an acquisition writes it. Checked on
this machine's three acquisitions (the 10x tissue set, the 20x scan, the 1536 plate): each
ships a ``configurations.xml``, and every ``<mode>`` in it carries ``Name``, ``ExposureTime``,
``AnalogGain``, ``IlluminationSource``, ``IlluminationIntensity``, ``ZOffset`` and
``EmissionFilterPosition`` — a filter-WHEEL SLOT (``1`` for every mode in all three files),
with no table anywhere in the acquisition mapping a slot to a passband. So there is no stated
excitation and no stated emission; ``Name`` is the record. Squid's own ``acquisition.yaml``
writer (``control/acquisition_yaml_loader.py``) confirms it from the other side: the
``objective:`` block is name / magnification / pixel_size_um / camera_binning, and the
``channels:`` block is names.

``_decon`` consumes this to build a PSF. It is the ONLY wavelength parse in the package: the
optics layer no longer re-opens the acquisition through a second reader to ask the same
question (that second reader globbed ``*_nm_Ex.tiff`` and so could not see a multi-band
``Fluorescence 638 nm - Penta`` channel at all).
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

_WAVELENGTHS = ("405", "488", "561", "638", "730")
# Mirrors Squid's filename sanitization: whitespace + \ / : * ? " < > | -> "_".
_UNSAFE_CHARS = re.compile(r'[\s/\\:*?"<>|]')

# A 3-digit wavelength IMMEDIATELY BEFORE the "nm" token, in any of Squid's spellings:
#
#     Fluorescence 638 nm Ex        Fluorescence_638_nm_Ex        Fluorescence_638_nm_-_Penta
#
# Anchored on "nm" rather than on the five palette wavelengths, because the palette is a
# rendering choice and this is a physical measurement: a scope fitted with a 445 line must
# report 445, not None. Anchored on "nm" rather than on any 3-digit run, because channel names
# also carry numbers that are not wavelengths, and answering with one of THOSE would put a
# confidently wrong kernel on a real image. `nm(?![A-Za-z])` and not `\bnm\b`: "_" is a word
# character, so `\b` does not fire between "nm" and "_-_Penta".
_WAVELENGTH_NM = re.compile(r"(?<!\d)(\d{3})(?!\d)[\s_-]*nm(?![A-Za-z])", re.IGNORECASE)


def normalize(name: str) -> str:
    """Convert a YAML channel name to its filename (canonical) form."""
    return _UNSAFE_CHARS.sub("_", str(name).strip())


def excitation_nm(channel_name) -> float | None:
    """The channel's EXCITATION wavelength in nm, or ``None`` when the name states none.

    ``None`` is a real answer, not a failure to try: ``BF_LED_matrix_full`` and ``DF_LED_matrix``
    are broadband and have no line. Callers that need a wavelength must refuse on ``None`` rather
    than substitute one — see ``_decon.optics_for_channel``.

    Accepts the FILE form (``Fluorescence_638_nm_-_Penta``), the YAML form
    (``Fluorescence 638 nm Ex``) and a bare number (``"488"``), which is how an OME-TIFF that
    kept only the digits spells the same channel.
    """
    text = str(channel_name).strip()
    if text.isdigit():
        return float(text)
    match = _WAVELENGTH_NM.search(text)
    return float(match.group(1)) if match else None


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
    """v1.0+ top-level display_color, else pre-v1.0 nested camera_settings.<first cam>. Tolerant of
    a real acquisition yaml where a section is a scalar (never call .get on a non-dict)."""
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
    exposure_time_ms}}``.

    Reads ``acquisition_channels.yaml`` (dedicated snapshot); if absent, reads the ``channels:``
    block of ``acquisition.yaml`` (same structure). Returns {} when neither exists.
    """
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
    """Produce the metadata `channels` list, keyed on the canonical (filename) name.

    Each entry: {name, display_name, display_color, exposure_time_ms, excitation_nm}. Color:
    YAML -> CHANNEL_COLORS_MAP -> raise. An unrecognized channel is refused, never given a
    placeholder color. ``excitation_nm`` is ``None`` for a broadband channel, which is an
    answer; see :func:`excitation_nm`.
    """
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
