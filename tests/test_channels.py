"""Tests for channel identity + layered color resolution."""

import pytest

from squidxplorer._channels import (
    excitation_nm,
    fallback_color,
    load_channel_yaml,
    normalize,
    resolve_channels,
)


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


def test_load_channel_yaml_sources_and_color_precedence(tmp_path):
    """Absent -> {}; acquisition.yaml is the fallback source; a top-level color beats the camera one."""
    assert load_channel_yaml(tmp_path) == {}
    (tmp_path / "acquisition.yaml").write_text(
        "channels:\n- name: Fluorescence 638 nm - Penta\n  display_color: '#FF0000'\n"
    )
    assert load_channel_yaml(tmp_path)["Fluorescence_638_nm_-_Penta"]["display_color"] == "#FF0000"
    (tmp_path / "acquisition_channels.yaml").write_text(
        "channels:\n"
        "- name: Fluorescence 638 nm - Penta\n"
        "  camera_settings:\n"
        "    '1':\n"
        "      display_color: '#FF0000'\n"
        "      exposure_time_ms: 50.0\n"
        "- name: Fluorescence 488 nm - Penta\n"
        "  display_color: '#1FFF00'\n"
        "  camera_settings:\n"
        "    '1':\n"
        "      display_color: '#000000'\n"
    )
    out = load_channel_yaml(tmp_path)
    entry = out["Fluorescence_638_nm_-_Penta"]
    assert entry["display_color"] == "#FF0000"
    assert entry["exposure_time_ms"] == 50.0
    assert entry["display_name"] == "Fluorescence 638 nm - Penta"
    assert out["Fluorescence_488_nm_-_Penta"]["display_color"] == "#1FFF00"


@pytest.mark.parametrize(
    "channel, expected",
    [
        ("Fluorescence_638_nm_-_Penta", "#FF0000"),
        ("Fluorescence_405_nm_-_Penta", "#20ADF8"),
        ("Fluorescence_561_nm_-_Penta", "#FFCF00"),
        ("Fluorescence_730_nm", "#770000"),
        ("BF_LED_matrix_R", "#FF0000"),
        ("SomeWeird_Channel", None),
    ],
)
def test_fallback_color_by_wavelength_or_letter(channel, expected):
    assert fallback_color(channel) == expected


def test_resolve_channels_uses_yaml_then_falls_back_carrying_the_excitation_line():
    yaml_map = {
        "Fluorescence_638_nm_-_Penta": {
            "display_name": "Fluorescence 638 nm - Penta",
            "display_color": "#FF0000",
            "exposure_time_ms": 50.0,
        }
    }
    resolved = resolve_channels(
        ["Fluorescence_638_nm_-_Penta", "Fluorescence_561_nm_-_Penta", "BF_LED_matrix_full_R"],
        yaml_map,
    )
    by_name = {c["name"]: c for c in resolved}
    assert by_name["Fluorescence_638_nm_-_Penta"]["display_color"] == "#FF0000"
    assert by_name["Fluorescence_638_nm_-_Penta"]["exposure_time_ms"] == 50.0
    assert by_name["Fluorescence_638_nm_-_Penta"]["excitation_nm"] == 638.0
    assert by_name["Fluorescence_561_nm_-_Penta"]["display_color"] == "#FFCF00"
    assert by_name["Fluorescence_561_nm_-_Penta"]["display_name"] == "Fluorescence_561_nm_-_Penta"
    assert by_name["BF_LED_matrix_full_R"]["excitation_nm"] is None
    with pytest.raises(ValueError, match="Could not resolve a display color"):
        resolve_channels(["Totally_Unknown"], {})


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
        ("Fluorescence_512_slot_405_nm_Ex", 405.0),
        ("Cube_405X_slot_512", None),
        ("BF_LED_matrix_full", None),
        ("DF_LED_matrix", None),
        ("BF_LED_matrix_full_R", None),
        ("SomeWeird_Channel", None),
    ],
)
def test_excitation_nm_reads_every_spelling_squid_writes_and_is_none_for_broadband(channel, expected):
    """Anchored on the 'nm' token, not any 3-digit run; None is the answer for a channel with no line."""
    assert excitation_nm(channel) == expected
