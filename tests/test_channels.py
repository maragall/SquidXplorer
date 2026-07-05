"""Tests for channel identity + layered color resolution (AC2, AC3)."""

import warnings

import pytest

from squidmip._channels import (
    CHANNEL_COLORS_MAP,
    fallback_color,
    load_channel_yaml,
    normalize,
    resolve_channels,
)


# --- AC3: filename normalization (spaces -> underscore, dash preserved) ------
@pytest.mark.parametrize(
    "yaml_name, expected",
    [
        ("Fluorescence 638 nm - Penta", "Fluorescence_638_nm_-_Penta"),
        ("Fluorescence 405 nm - Penta", "Fluorescence_405_nm_-_Penta"),
        ("BF LED matrix full", "BF_LED_matrix_full"),
    ],
)
def test_normalize_round_trip(yaml_name, expected):
    assert normalize(yaml_name) == expected


def test_normalize_collision_guard_warns_and_keeps_first(tmp_path):
    # "A B" and "A_B" both normalize to "A_B" -> collision must warn, not silently overwrite.
    (tmp_path / "acquisition_channels.yaml").write_text(
        "channels:\n"
        "- name: A B\n"
        "  display_color: '#111111'\n"
        "- name: A_B\n"
        "  display_color: '#222222'\n"
    )
    with pytest.warns(UserWarning, match="collision"):
        out = load_channel_yaml(tmp_path)
    assert out["A_B"]["display_color"] == "#111111"  # first wins


# --- AC2: colors from YAML ---------------------------------------------------
def test_load_channel_yaml_nested_camera_color(tmp_path):
    (tmp_path / "acquisition_channels.yaml").write_text(
        "channels:\n"
        "- name: Fluorescence 638 nm - Penta\n"
        "  camera_settings:\n"
        "    '1':\n"
        "      display_color: '#FF0000'\n"
        "      exposure_time_ms: 50.0\n"
    )
    out = load_channel_yaml(tmp_path)
    entry = out["Fluorescence_638_nm_-_Penta"]
    assert entry["display_color"] == "#FF0000"
    assert entry["ex"] == 50.0
    assert entry["display_name"] == "Fluorescence 638 nm - Penta"


def test_load_channel_yaml_top_level_color_preferred(tmp_path):
    # v1.0+ layout: top-level display_color wins over camera_settings.
    (tmp_path / "acquisition_channels.yaml").write_text(
        "channels:\n"
        "- name: Fluorescence 488 nm - Penta\n"
        "  display_color: '#1FFF00'\n"
        "  camera_settings:\n"
        "    '1':\n"
        "      display_color: '#000000'\n"
    )
    out = load_channel_yaml(tmp_path)
    assert out["Fluorescence_488_nm_-_Penta"]["display_color"] == "#1FFF00"


def test_load_channel_yaml_absent_returns_empty(tmp_path):
    assert load_channel_yaml(tmp_path) == {}


def test_load_channel_yaml_falls_back_to_acquisition_yaml(tmp_path):
    # no dedicated acquisition_channels.yaml -> read the channels: block of acquisition.yaml
    (tmp_path / "acquisition.yaml").write_text(
        "channels:\n- name: Fluorescence 638 nm - Penta\n  display_color: '#FF0000'\n"
    )
    out = load_channel_yaml(tmp_path)
    assert out["Fluorescence_638_nm_-_Penta"]["display_color"] == "#FF0000"


# --- fallback palette --------------------------------------------------------
@pytest.mark.parametrize(
    "channel, expected",
    [
        ("Fluorescence_638_nm_-_Penta", "#FF0000"),
        ("Fluorescence_405_nm_-_Penta", "#20ADF8"),
        ("Fluorescence_561_nm_-_Penta", "#FFCF00"),
        ("Fluorescence_730_nm", "#770000"),
        ("BF_LED_matrix_R", "#FF0000"),
    ],
)
def test_fallback_color_by_wavelength_or_letter(channel, expected):
    assert fallback_color(channel) == expected


def test_fallback_color_unknown_returns_none():
    assert fallback_color("SomeWeird_Channel") is None


# --- resolve_channels: the layered fallback in one place ---------------------
def test_resolve_channels_uses_yaml_then_falls_back(tmp_path):
    yaml_map = {
        "Fluorescence_638_nm_-_Penta": {
            "display_name": "Fluorescence 638 nm - Penta",
            "display_color": "#FF0000",
            "ex": 50.0,
        }
    }
    # 638 is in YAML; 561 is not -> wavelength fallback.
    resolved = resolve_channels(
        ["Fluorescence_638_nm_-_Penta", "Fluorescence_561_nm_-_Penta"], yaml_map
    )
    by_name = {c["name"]: c for c in resolved}
    assert by_name["Fluorescence_638_nm_-_Penta"]["display_color"] == "#FF0000"
    assert by_name["Fluorescence_638_nm_-_Penta"]["ex"] == 50.0
    assert by_name["Fluorescence_561_nm_-_Penta"]["display_color"] == "#FFCF00"
    assert by_name["Fluorescence_561_nm_-_Penta"]["display_name"] == "Fluorescence_561_nm_-_Penta"


def test_resolve_channels_unknown_channel_raises():
    # no YAML entry and no wavelength/BF match -> explicit failure, never a placeholder color
    with pytest.raises(ValueError, match="Could not resolve a display color"):
        resolve_channels(["Totally_Unknown"], {})


def test_palette_matches_hongquan_yaml():
    # guard against silently drifting from the authoritative Squid map
    assert CHANNEL_COLORS_MAP["405"] == "#20ADF8"
    assert CHANNEL_COLORS_MAP["638"] == "#FF0000"
