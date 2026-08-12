"""Tests for channel identity + layered color resolution."""

import warnings

import pytest

from squidxplorer._channels import (
    CHANNEL_COLORS_MAP,
    excitation_nm,
    fallback_color,
    load_channel_yaml,
    normalize,
    resolve_channels,
)


# --- filename normalization (spaces -> underscore, dash preserved) -----------
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
    # "A B" and "A_B" both normalize to "A_B"
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


# --- colors from YAML ----------------------------------------------------------
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
    assert entry["exposure_time_ms"] == 50.0
    assert entry["display_name"] == "Fluorescence 638 nm - Penta"


def test_load_channel_yaml_top_level_color_preferred(tmp_path):
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
            "exposure_time_ms": 50.0,
        }
    }
    # 638 is in YAML; 561 falls back to the wavelength palette
    resolved = resolve_channels(
        ["Fluorescence_638_nm_-_Penta", "Fluorescence_561_nm_-_Penta"], yaml_map
    )
    by_name = {c["name"]: c for c in resolved}
    assert by_name["Fluorescence_638_nm_-_Penta"]["display_color"] == "#FF0000"
    assert by_name["Fluorescence_638_nm_-_Penta"]["exposure_time_ms"] == 50.0
    assert by_name["Fluorescence_561_nm_-_Penta"]["display_color"] == "#FFCF00"
    assert by_name["Fluorescence_561_nm_-_Penta"]["display_name"] == "Fluorescence_561_nm_-_Penta"


def test_resolve_channels_unknown_channel_raises():
    with pytest.raises(ValueError, match="Could not resolve a display color"):
        resolve_channels(["Totally_Unknown"], {})


# --- excitation wavelength: this package's one channel-wavelength parse -------
@pytest.mark.parametrize(
    "channel, expected",
    [
        ("Fluorescence_638_nm_-_Penta", 638.0),
        ("Fluorescence 638 nm - Penta", 638.0),
        ("Fluorescence_405_nm_Ex", 405.0),
        ("Fluorescence 488 nm Ex", 488.0),
        ("Fluorescence_730_nm", 730.0),
        ("488", 488.0),
        ("Fluorescence_445_nm_Ex", 445.0),
    ],
)
def test_excitation_nm_reads_every_spelling_squid_writes(channel, expected):
    assert excitation_nm(channel) == expected


@pytest.mark.parametrize(
    "channel",
    ["BF_LED_matrix_full", "DF_LED_matrix", "BF_LED_matrix_full_R", "SomeWeird_Channel"],
)
def test_excitation_nm_is_none_for_a_broadband_channel(channel):
    """None is the answer for a channel with no line, not a failure to parse."""
    assert excitation_nm(channel) is None


def test_excitation_nm_ignores_a_number_that_is_not_a_wavelength():
    """Anchored on the 'nm' token, not any 3-digit run."""
    assert excitation_nm("Cube_405X_slot_512") is None
    assert excitation_nm("Fluorescence_512_slot_405_nm_Ex") == 405.0


def test_resolve_channels_carries_the_excitation_wavelength():
    resolved = resolve_channels(
        ["Fluorescence_638_nm_-_Penta", "BF_LED_matrix_full_R"], {})
    by_name = {c["name"]: c for c in resolved}
    assert by_name["Fluorescence_638_nm_-_Penta"]["excitation_nm"] == 638.0
    assert by_name["BF_LED_matrix_full_R"]["excitation_nm"] is None


def test_palette_matches_hongquan_yaml():
    assert CHANNEL_COLORS_MAP["405"] == "#20ADF8"
    assert CHANNEL_COLORS_MAP["638"] == "#FF0000"
