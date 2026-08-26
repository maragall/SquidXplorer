"""The typed acquisition model: validate the ACQUISITION, not just the command line."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from squidxplorer._acquisition import Acquisition, DisplayChannel


def _kw(**over):
    base = dict(
        regions=["A1", "A2"],
        fovs_per_region={"A1": [0, 1], "A2": [0]},
        fov_positions_um={("A1", 0): (0.0, 0.0), ("A1", 1): (100.0, 0.0)},
        channels=[{"name": "Fluorescence_488_nm_Ex", "display_name": "488",
                   "display_color": "#1FFF00", "excitation_nm": 488.0}],
        n_z=3,
        z_levels=[0, 1, 2],
        dz_um=1.5,
        pixel_size_um=0.325,
        wellplate_format="24 well plate",
        frame_shape=(2084, 3000),
        dtype=np.dtype("uint16"),
        n_t=2,
    )
    base.update(over)
    return base


def test_builds_from_the_reader_dict():
    a = Acquisition(**_kw())
    assert a.pixel_size_um == 0.325
    assert a.frame_shape == (2084, 3000)
    assert a.dtype == np.dtype("uint16")
    assert a.channels[0].name == "Fluorescence_488_nm_Ex"
    assert isinstance(a.channels[0], DisplayChannel)
    assert a.channel_names == ["Fluorescence_488_nm_Ex"]


def test_a_missing_required_field_is_refused_naming_it():
    kw = _kw()
    del kw["n_t"]
    with pytest.raises(ValidationError, match="n_t"):
        Acquisition(**kw)


def test_an_unknown_field_is_refused_so_a_typo_is_not_silently_stored():
    with pytest.raises(ValidationError, match="pixel_size"):
        Acquisition(**_kw(pixel_size=0.325))


@pytest.mark.parametrize("bad", [
    dict(n_z="three"), dict(n_t=0), dict(n_z=0), dict(pixel_size_um=0.0),
    dict(pixel_size_um=-1.0), dict(dz_um=-1.5),
])
def test_a_wrong_or_nonpositive_value_is_refused_at_construction(bad):
    with pytest.raises(ValidationError):
        Acquisition(**_kw(**bad))


def test_fovs_per_region_must_cover_every_region():
    with pytest.raises(ValidationError, match="A2"):
        Acquisition(**_kw(fovs_per_region={"A1": [0, 1]}))


def test_z_levels_must_agree_with_n_z():
    with pytest.raises(ValidationError, match="z_levels"):
        Acquisition(**_kw(z_levels=[0, 1]))


def test_pixel_size_may_be_absent_but_asking_for_it_raises_naming_the_field():
    a = Acquisition(**_kw(pixel_size_um=None))
    assert a.pixel_size_um is None
    with pytest.raises(ValueError, match="pixel_size_um"):
        a.require_pixel_size_um()
    assert Acquisition(**_kw()).require_pixel_size_um() == 0.325


def test_dz_may_be_absent_or_zero_but_asking_for_it_as_a_scale_raises():
    with pytest.raises(ValueError, match="dz_um"):
        Acquisition(**_kw(dz_um=None)).require_dz_um()
    assert Acquisition(**_kw()).require_dz_um() == 1.5
    a = Acquisition(**_kw(dz_um=0.0, n_z=1, z_levels=[0]))
    assert a.dz_um == 0.0 and a["dz_um"] == 0.0
    with pytest.raises(ValueError, match="dz_um"):
        a.require_dz_um()


def test_channel_index_refuses_an_unknown_channel_rather_than_returning_zero():
    a = Acquisition(**_kw())
    assert a.channel_index("Fluorescence_488_nm_Ex") == 0
    with pytest.raises(KeyError, match="Fluorescence_638_nm_Ex"):
        a.channel_index("Fluorescence_638_nm_Ex")


def test_the_mapping_shim_behaves_like_the_old_dict():
    a = Acquisition(**_kw())
    assert a["pixel_size_um"] == 0.325
    assert a["frame_shape"] == (2084, 3000)
    assert a["channels"][0]["name"] == "Fluorescence_488_nm_Ex"
    assert a.get("dz_um") == 1.5
    assert a.get("nope", "fallback") == "fallback"
    assert "pixel_size_um" in a
    assert "nope" not in a
    assert set(a.keys()) >= {"regions", "channels", "n_z", "n_t", "dtype", "frame_shape"}
    assert dict(a)["n_t"] == 2
    assert all(isinstance(k, str) for k in a), f"iterating yielded non-keys: {list(a)[:2]}"
    assert list(a) == list(a.keys())
    with pytest.raises(KeyError):
        a["pixel_size"]


def test_every_reader_returns_a_validated_acquisition(squid_dataset, multipage_dataset,
                                                      ome_tiff_dataset, zarr_hcs_dataset):
    """All four reader classes must hand back the same validated type."""
    from squidxplorer.reader import open_reader

    for root, _ in (squid_dataset, multipage_dataset, ome_tiff_dataset, zarr_hcs_dataset):
        meta = open_reader(root).metadata
        assert isinstance(meta, Acquisition), f"{root} reader returned {type(meta).__name__}"
        assert meta.channel_names, "no channels"
        assert meta.frame_shape[0] > 0 and meta.frame_shape[1] > 0


def test_the_reader_boundary_refuses_a_malformed_acquisition(squid_dataset, monkeypatch):
    """A bad acquisition must die at the reader, naming the field."""
    import squidxplorer.reader as reader_mod
    from squidxplorer.reader import open_reader

    real = reader_mod.load_acquisition_metadata

    def broken(root):
        m = dict(real(root))
        m["pixel_size_um"] = -1.0
        return m

    monkeypatch.setattr(reader_mod, "load_acquisition_metadata", broken)
    with pytest.raises(ValidationError, match="pixel_size_um"):
        open_reader(squid_dataset[0]).metadata
