"""The typed acquisition model: validate the ACQUISITION, not just the command line.

Pins three things: the schema is validated once, at the reader boundary, naming the offending
field; the model is still a Mapping, so existing `meta["..."]` call sites keep working; and
genuinely-optional fields have loud accessors (asking for a missing pixel_size_um raises naming
the field, never a substituted default).
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from squidxplorer._acquisition import Acquisition, Channel


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
    assert isinstance(a.channels[0], Channel)


def test_channel_names_are_a_convenience_not_a_reimplementation():
    assert Acquisition(**_kw()).channel_names == ["Fluorescence_488_nm_Ex"]


def test_a_missing_required_field_is_refused_naming_it():
    kw = _kw()
    del kw["n_t"]
    with pytest.raises(ValidationError, match="n_t"):
        Acquisition(**kw)


def test_an_unknown_field_is_refused_so_a_typo_is_not_silently_stored():
    with pytest.raises(ValidationError, match="pixel_size"):
        Acquisition(**_kw(pixel_size=0.325))


def test_a_wrong_typed_field_is_refused_at_construction():
    with pytest.raises(ValidationError):
        Acquisition(**_kw(n_z="three"))


def test_negative_dimensions_are_refused():
    with pytest.raises(ValidationError):
        Acquisition(**_kw(n_t=0))
    with pytest.raises(ValidationError):
        Acquisition(**_kw(n_z=0))


def test_a_nonpositive_pixel_size_is_refused_at_the_boundary_not_at_use():
    with pytest.raises(ValidationError):
        Acquisition(**_kw(pixel_size_um=0.0))
    with pytest.raises(ValidationError):
        Acquisition(**_kw(pixel_size_um=-1.0))


def test_fovs_per_region_must_cover_every_region():
    with pytest.raises(ValidationError, match="A2"):
        Acquisition(**_kw(fovs_per_region={"A1": [0, 1]}))


def test_z_levels_must_agree_with_n_z():
    with pytest.raises(ValidationError, match="z_levels"):
        Acquisition(**_kw(z_levels=[0, 1]))


def test_pixel_size_may_be_absent_but_asking_for_it_raises_naming_the_field():
    a = Acquisition(**_kw(pixel_size_um=None))
    assert a.pixel_size_um is None                      # modelled Optional, honestly
    with pytest.raises(ValueError, match="pixel_size_um"):
        a.require_pixel_size_um()                        # but LOUD at the point of use


def test_require_pixel_size_returns_the_value_when_present():
    assert Acquisition(**_kw()).require_pixel_size_um() == 0.325


def test_dz_may_be_absent_but_asking_for_it_raises_naming_the_field():
    a = Acquisition(**_kw(dz_um=None))
    with pytest.raises(ValueError, match="dz_um"):
        a.require_dz_um()
    assert Acquisition(**_kw()).require_dz_um() == 1.5


def test_a_zero_dz_is_STORED_but_refused_as_a_scale():
    # a single-plane acquisition really does record delta_z_mm: 0 in several fixtures; it is
    # only meaningless once used as a z scale
    a = Acquisition(**_kw(dz_um=0.0, n_z=1, z_levels=[0]))
    assert a.dz_um == 0.0
    assert a["dz_um"] == 0.0
    with pytest.raises(ValueError, match="dz_um"):
        a.require_dz_um()


def test_a_negative_dz_is_refused_outright():
    with pytest.raises(ValidationError, match="dz_um"):
        Acquisition(**_kw(dz_um=-1.5))


def test_channel_index_refuses_an_unknown_channel_rather_than_returning_zero():
    a = Acquisition(**_kw())
    assert a.channel_index("Fluorescence_488_nm_Ex") == 0
    with pytest.raises(KeyError, match="Fluorescence_638_nm_Ex"):
        a.channel_index("Fluorescence_638_nm_Ex")


def test_subscript_access_still_works_for_unmigrated_call_sites():
    a = Acquisition(**_kw())
    assert a["pixel_size_um"] == 0.325
    assert a["frame_shape"] == (2084, 3000)
    assert a["channels"][0]["name"] == "Fluorescence_488_nm_Ex"   # channels stay dict-like too


def test_get_and_in_and_keys_behave_like_the_old_dict():
    a = Acquisition(**_kw())
    assert a.get("dz_um") == 1.5
    assert a.get("nope", "fallback") == "fallback"
    assert "pixel_size_um" in a
    assert "nope" not in a
    assert set(a.keys()) >= {"regions", "channels", "n_z", "n_t", "dtype", "frame_shape"}
    assert dict(a)["n_t"] == 2
    # `for k in meta` must yield KEYS, not BaseModel.__iter__'s (key, value) pairs
    assert all(isinstance(k, str) for k in a), f"iterating yielded non-keys: {list(a)[:2]}"
    assert list(a) == list(a.keys())


def test_an_unknown_key_raises_keyerror_not_none():
    with pytest.raises(KeyError):
        Acquisition(**_kw())["pixel_size"]


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
        m["pixel_size_um"] = -1.0       # physically impossible; would invert every offset
        return m

    monkeypatch.setattr(reader_mod, "load_acquisition_metadata", broken)
    with pytest.raises(ValidationError, match="pixel_size_um"):
        open_reader(squid_dataset[0]).metadata


def test_the_mapping_shim_is_not_dramatically_slower_than_the_dict_it_replaced():
    """Regression gate: consulting model_fields (a pydantic classproperty) per lookup made the
    shim ~18x slower than a dict, which showed up as apparent flakiness in Qt event-loop tests.
    Bound is relative to a dict measured in the same process/moment, so it isn't load-sensitive.
    """
    import timeit

    a = Acquisition(**_kw())
    d = dict(_kw())
    n = 20000
    dict_s = min(timeit.repeat(lambda: d["channels"], number=n, repeat=7))
    model_s = min(timeit.repeat(lambda: a["channels"], number=n, repeat=7))
    assert model_s < dict_s * 6, (
        f"Acquisition subscript is {model_s / dict_s:.1f}x a dict lookup "
        f"({model_s:.4f}s vs {dict_s:.4f}s for {n} lookups). The Mapping shim must not consult "
        "pydantic's model_fields classproperty at lookup time — see _ACQ_KEYS."
    )
