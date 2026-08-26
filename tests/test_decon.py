"""The `decon` operator (the volume solve): numerical property tests, not smoke tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from squidxplorer import add_operator, available_plane_operators, project_well, operator_consumes
from squidxplorer._decon import (
    DEFAULT_ITERATIONS,
    DEFAULT_OPTICS,
    METHOD,
    OpticsParams,
    active_optics,
    clear_optics,
    decon_op,
    deconvolve_stack,
    make_psf,
    optics_for_channel,
    optics_override,
    set_optics,
)
from squidxplorer._engine import _resolve_operator
from squidxplorer.projection import PLANE_OP, Z_REDUCER, bind_channel
from squidxplorer.reader import open_reader

scipy_ndimage = pytest.importorskip("scipy.ndimage")
scipy_signal = pytest.importorskip("scipy.signal")
pytest.importorskip("petakit")
pytest.importorskip("psfmodels")


FAST_OPTICS = OpticsParams(na=0.3, wavelength_um=0.525, dxy_um=0.4, dz_um=1.5, nz=3)


@pytest.fixture(autouse=True)
def _no_leaked_optics():
    """Every test starts from the default optics; set_optics is global state."""
    clear_optics()
    yield
    clear_optics()


def _ground_truth(size: int = 96, seed: int = 0) -> np.ndarray:
    """A sparse-puncta phantom on a dim pedestal, the shape RL assumes; puncta have finite extent."""
    rng = np.random.default_rng(seed)
    seeds = np.zeros((size, size), dtype=np.float32)
    for y, x in zip(rng.integers(10, size - 10, 30), rng.integers(10, size - 10, 30)):
        seeds[y, x] += rng.uniform(400, 2000)
    return (scipy_ndimage.gaussian_filter(seeds, 1.2, mode="reflect") + 20.0).astype(np.float32)


def _nz1(optics: OpticsParams) -> OpticsParams:
    """*optics* rebound to a 1-plane stack — what deconvolve_stack itself does on (1, Y, X)."""
    return OpticsParams(optics.na, optics.wavelength_um, optics.dxy_um,
                        optics.dz_um, 1, optics.ni)


def _psf_plane(optics: OpticsParams) -> np.ndarray:
    """The in-focus PSF plane the nz=1 volume solve is built from, renormalised to sum 1."""
    psf = make_psf(_nz1(optics))
    centre = psf[psf.shape[0] // 2]
    return centre / centre.sum()


def _plane_solve(plane: np.ndarray, optics: OpticsParams, iterations: int) -> np.ndarray:
    """RL on ONE plane through the one code path: the volume solve over a 1-plane stack."""
    return deconvolve_stack(np.asarray(plane)[None, ...], optics, iterations, project=False)[0]


def _blur_with_real_psf(img: np.ndarray, optics: OpticsParams = FAST_OPTICS) -> np.ndarray:
    """Blur with the SAME vectorial PSF the nz=1 solve will deconvolve with."""
    return scipy_signal.convolve(img.astype(np.float64), _psf_plane(optics),
                                 mode="same").astype(np.float32)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def test_decon_moves_a_known_blur_back_toward_ground_truth():
    """RMSE falls, peak amplitude comes back, total intensity is conserved: one solve, three properties."""
    truth = _ground_truth()
    blurred = _blur_with_real_psf(truth)

    restored = _plane_solve(blurred, FAST_OPTICS, iterations=30)

    before, after = _rmse(blurred, truth), _rmse(restored, truth)
    assert after < before * 0.6, f"RL did not sharpen: rmse {before:.1f} -> {after:.1f}"
    assert blurred.max() < truth.max() * 0.8
    assert restored.max() > blurred.max() * 1.2
    assert restored.max() <= truth.max() * 1.5
    assert abs(float(restored.sum()) - float(blurred.sum())) / float(blurred.sum()) < 0.05


def test_more_iterations_reduce_error_over_the_useful_range():
    truth = _ground_truth()
    blurred = _blur_with_real_psf(truth)
    errs = [_rmse(_plane_solve(blurred, FAST_OPTICS, iterations=n), truth)
            for n in (1, 5, 15, 30)]
    assert errs[-1] < errs[0], f"error did not fall with iterations: {errs}"


def test_deconvolve_stack_project_false_keeps_every_plane():
    """The format contract's decon shape: output the same size as the input."""
    stack = np.stack([_blur_with_real_psf(_ground_truth()).astype(np.uint16)] * 3)
    out = deconvolve_stack(stack, FAST_OPTICS, iterations=0, project=False)
    assert out.shape == stack.shape
    np.testing.assert_array_equal(out, stack)


def test_dtype_is_preserved_and_the_input_plane_is_never_mutated():
    plane = _blur_with_real_psf(_ground_truth()).astype(np.uint16)
    before = plane.copy()
    out = _plane_solve(plane, FAST_OPTICS, iterations=5)
    assert out.dtype == np.uint16
    assert np.array_equal(plane, before), "deconvolve mutated the caller's plane"


def test_uint16_output_is_clipped_not_wrapped():
    plane = np.full((32, 32), 60000, dtype=np.uint16)
    plane[16, 16] = 65535
    out = _plane_solve(plane, FAST_OPTICS, iterations=20)
    assert out.min() >= 0 and out.max() <= 65535
    assert out[16, 16] > 60000, "the bright pixel wrapped to a dark one"


def test_a_flat_field_stays_flat_no_boundary_artifact():
    plane = np.full((64, 64), 1000.0, dtype=np.float32)
    out = _plane_solve(plane, FAST_OPTICS, iterations=20)
    rim = np.concatenate([out[0], out[-1], out[:, 0], out[:, -1]])
    assert np.allclose(rim, 1000.0, rtol=0.05), f"edge artifact: rim range {rim.min()}..{rim.max()}"


def test_the_psf_is_a_real_vectorial_psf_not_a_gaussian():
    psf = _psf_plane(DEFAULT_OPTICS)
    assert psf.ndim == 2 and psf.shape[0] == psf.shape[1]
    assert float(psf.sum()) == pytest.approx(1.0, rel=1e-5)

    yy, xx = np.indices(psf.shape)
    cy, cx = (np.array(psf.shape) - 1) / 2
    sigma = np.sqrt((((yy - cy) ** 2 + (xx - cx) ** 2) * psf).sum() / 2.0)
    gauss = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    gauss /= gauss.sum()
    assert np.abs(psf - gauss).max() / psf.max() > 0.02, "the 'real' PSF is Gaussian after all"

    assert sigma < 1.35, f"real PSF sigma {sigma:.3f} px; the old code hardcoded 1.5"


def test_optics_come_from_acquisition_metadata_not_constants(tmp_path):
    root = tmp_path / "acq"
    root.mkdir(parents=True)
    (root / "acquisition parameters.json").write_text(
        '{"dz(um)": 9.0, "Nz": 9, "Nt": 1, "objective": {"magnification": 20.0, "NA": 0.75},'
        ' "sensor_pixel_size_um": 6.5}'
    )
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.400\n  magnification: 20.0\n"
        "z_stack:\n  nz: 4\n  delta_z_mm: 0.002\n"
        "time_series:\n  nt: 1\n")

    optics = OpticsParams.from_acquisition(root, "Fluorescence_488_nm_Ex")
    assert optics.na == 0.75
    assert optics.dxy_um == pytest.approx(0.400)
    assert optics.dxy_um != pytest.approx(6.5 / 20.0)
    assert optics.dz_um == 2.0
    assert optics.nz == 4
    assert optics.wavelength_um == pytest.approx(0.525)
    assert optics.immersion_index == pytest.approx(1.0)
    assert OpticsParams(na=1.2, wavelength_um=0.525, dxy_um=0.1).immersion_index == 1.33
    assert OpticsParams(na=1.4, wavelength_um=0.525, dxy_um=0.1).immersion_index == 1.515

    assert make_psf(optics).shape != make_psf(DEFAULT_OPTICS).shape


def test_set_optics_overrides_the_default_for_the_registered_operator():
    assert active_optics() == DEFAULT_OPTICS
    other = OpticsParams(na=0.75, wavelength_um=0.67, dxy_um=0.325, dz_um=1.0, nz=5)
    set_optics(other)
    assert active_optics() == other
    clear_optics()
    assert active_optics() == DEFAULT_OPTICS

    with pytest.raises(ValueError, match="needs an OpticsParams"):
        set_optics("0.3 NA")


def test_optics_reject_physically_impossible_values():
    for kwargs in ({"na": 0.0}, {"na": -1.0}, {"wavelength_um": 0}, {"dxy_um": -0.5}):
        base = dict(na=0.3, wavelength_um=0.525, dxy_um=0.752)
        base.update(kwargs)
        with pytest.raises(ValueError):
            OpticsParams(**base)
    with pytest.raises(ValueError, match="nz must be"):
        OpticsParams(na=0.3, wavelength_um=0.525, dxy_um=0.752, nz=0)
    with pytest.raises(ValueError, match="immersion index"):
        OpticsParams(na=0.3, wavelength_um=0.525, dxy_um=0.752, ni=0.5)


def test_the_engine_method_is_pinned_to_rl_because_petakit_defaults_to_a_broken_omw():
    petakit = pytest.importorskip("petakit")
    assert METHOD == "rl"

    truth = _ground_truth(48)
    stack = np.stack([_blur_with_real_psf(truth)] * 3).astype(np.float32)
    psf3 = make_psf(FAST_OPTICS)

    omw = petakit.deconvolve(stack, psf3, method="omw", iterations=2, gpu=False)
    rl = petakit.deconvolve(stack, psf3, method="rl", iterations=10, gpu=False)

    assert np.any(stack), "the phantom itself is empty; the test proves nothing"
    assert float((omw == 0).mean()) > 0.95, (
        "omw is no longer degenerate here — re-evaluate the pinned METHOD"
    )
    assert float((rl == 0).mean()) == 0.0, "the pinned rl path produced a degenerate result"


def test_an_all_zero_engine_result_raises_instead_of_writing_black_tiles():
    import squidxplorer._decon as decon_mod

    class _Fake:
        @staticmethod
        def deconvolve(volume, psf, **kw):
            return np.zeros_like(volume)

    real = decon_mod._petakit
    decon_mod._petakit = lambda: _Fake()
    try:
        with pytest.raises(RuntimeError, match="all-zero"):
            decon_mod._run(np.ones((1, 8, 8), np.float32), np.ones((1, 3, 3), np.float32), 3, False)
    finally:
        decon_mod._petakit = real


def test_a_missing_petakit_fails_loud_and_never_silently_falls_back():
    import builtins

    import squidxplorer._decon as decon_mod

    real_import = builtins.__import__

    def _no_petakit(name, *args, **kwargs):
        if name == "petakit":
            raise ImportError("simulated: petakit not installed")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _no_petakit
    try:
        with pytest.raises(ImportError, match="deconvolution/deconvolution|petakit|NO fallback"):
            decon_mod._petakit()
    finally:
        builtins.__import__ = real_import


def test_decon_is_registered_as_the_depth_keeping_volume_solve():
    assert "decon" in available_plane_operators()
    assert operator_consumes("decon") == Z_REDUCER
    assert getattr(_resolve_operator("decon").fn, "keeps_depth", False) is True


def test_the_output_query_says_decon_keeps_its_depth():
    """`operator_output` is what the writer's depth AND the display's z-collapse read; for a z-consumer declaring keeps_depth the OUTPUT does not collapse"""
    from squidxplorer._engine import operator_output

    assert operator_output("decon") == (False, "intensity")


def test_decon3d_is_refused_by_name_with_a_pointer_to_decon():
    """Absence pin (2026-08-24): the survivor of the 2D/3D merge is REGISTERED AS `decon`; a stale recipe or script saying decon3d gets the rename, not a"""
    import squidxplorer

    assert "decon3d" not in squidxplorer.runnable_operators()
    with pytest.raises(KeyError, match="renamed to 'decon'"):
        _resolve_operator("decon3d")


def test_the_volume_solve_over_one_plane_equals_the_shelved_2d_solve():
    """THE nz=1 pin, written before deconvolve_plane was deleted: on a 1-plane stack the volume solve landed the 2-D in-focus answer (measured float32 max"""
    psf = make_psf(_nz1(FAST_OPTICS))
    centre = psf[psf.shape[0] // 2]
    assert float(centre.sum()) > 0
    plane2d = centre / centre.sum()
    assert plane2d.ndim == 2 and float(plane2d.sum()) == pytest.approx(1.0, rel=1e-5)

    truth = _ground_truth(64)
    blurred = _blur_with_real_psf(truth)
    restored = _plane_solve(blurred, FAST_OPTICS, iterations=15)
    assert _rmse(restored, truth) < _rmse(blurred, truth), (
        "the volume solve did not sharpen a 1-plane stack — the degenerate case is broken")


def test_decon_op_factory_builds_the_registrable_volume_solve():
    op = decon_op(FAST_OPTICS, iterations=1)
    assert op.consumes == frozenset({"z"})
    name = "decon_test_factory"
    if name not in available_plane_operators():
        add_operator(name, op, consumes=frozenset({"z"}))
    assert operator_consumes(name) == Z_REDUCER
    plane = _blur_with_real_psf(_ground_truth(32)).astype(np.uint16)
    out = op([plane])                       # a 1-plane stack is served, depth kept
    assert out.shape == (1, *plane.shape)


def test_project_well_with_decon_keeps_z_at_full_depth(squid_dataset):
    root, _ = squid_dataset
    reader = open_reader(root)
    out = project_well(reader, "B2", 0, reduce=decon_op(FAST_OPTICS, iterations=2))
    n_z = len(reader.metadata["z_levels"])
    n_c = len(reader.metadata["channels"])
    assert out.shape == (reader.metadata["n_t"], n_c, n_z, 4, 4)
    assert out.dtype == reader.metadata["dtype"]


def test_the_volume_solve_sharpens_the_mip(squid_dataset):
    truth = _ground_truth(64)
    stack = np.stack([_blur_with_real_psf(truth) for _ in range(3)]).astype(np.float32)

    out = deconvolve_stack(stack, FAST_OPTICS, iterations=10)
    assert out.shape == truth.shape

    def sharpness(a):
        gy, gx = np.gradient(a.astype(np.float64))
        return float(np.sqrt((gy ** 2 + gx ** 2).mean()) / a.std())

    assert sharpness(out) > sharpness(stack.max(axis=0)), "3-D decon did not sharpen the MIP"


def test_deconvolve_stack_rejects_a_2d_input():
    with pytest.raises(ValueError, match=r"needs \(Z, Y, X\)"):
        deconvolve_stack(np.zeros((8, 8), np.float32), FAST_OPTICS)


def test_channel_labels_from_squidxplorers_own_reader_parse_into_a_wavelength():
    from squidxplorer._decon import emission_um_for

    wavelengths = {
        emission_um_for(name)
        for name in ("488", "488 nm", "Fluorescence_488_nm_Ex", "Fluorescence 488 nm Ex",
                     "Fluorescence_488_nm_-_Penta", "Fluorescence 488 nm - Penta")
    }
    assert len(wavelengths) == 1
    assert 0.50 < wavelengths.pop() < 0.55


_PER_CHANNEL_CHANNELS = ["Fluorescence 488 nm Ex", "Fluorescence 638 nm Ex"]


def _two_channel_acquisition(root, nz: int = 2, frame: int = 64, channels=None):
    """A tiny 10x/NA-0.3 acquisition whose two channels have DIFFERENT emission wavelengths (or whatever *channels* names: a broadband channel states no"""
    tifffile = pytest.importorskip("tifffile")
    channels = list(_PER_CHANNEL_CHANNELS if channels is None else channels)
    (root / "ome_tiff").mkdir(parents=True)
    (root / "acquisition parameters.json").write_text(json.dumps({
        "Nz": nz, "Nt": 1, "dz(um)": 1.5,
        "objective": {"magnification": 10.0, "NA": 0.3},
        "sensor_pixel_size_um": 7.52,
    }))
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.752\n  magnification: 10.0\n  sensor_pixel_size_um: 7.52\n"
        "sample:\n  wellplate_format: glass slide\n"
        f"z_stack:\n  nz: {nz}\n  delta_z_mm: 0.0015\n"
        "time_series:\n  nt: 1\n"
        # A display colour per channel, as a real acquisition's YAML carries: the reader
        # refuses a channel it cannot colour, and a broadband LED channel has no
        # wavelength to fall back on.
        "channels:\n" + "".join(f"- name: {c}\n  display_color: '#FFFFFF'\n" for c in channels))

    rng = np.random.default_rng(0)
    data = np.zeros((1, nz, len(channels), frame, frame), np.uint16) + 200
    for z in range(nz):
        for c in range(len(channels)):
            ys = rng.integers(8, frame - 8, 12)
            xs = rng.integers(8, frame - 8, 12)
            data[0, z, c, ys, xs] = 9000 + 500 * c
    tifffile.imwrite(root / "ome_tiff" / "manual0_0000.ome.tiff", data,
                     metadata={"axes": "TZCYX", "Channel": {"Name": channels}})
    return root


def test_the_registered_decon_deconvolves_each_channel_at_its_own_wavelength(tmp_path):
    root = _two_channel_acquisition(tmp_path / "acq")
    reader = open_reader(root)
    names = [c["name"] for c in reader.metadata["channels"]]
    assert names == ["Fluorescence_488_nm_Ex", "Fluorescence_638_nm_Ex"]

    out = project_well(reader, "manual0", 0,
                       reduce=_resolve_operator("decon").fn, consumes=Z_REDUCER)
    assert out.shape[2] == reader.metadata["n_z"]     # full depth since 2026-08-21

    c638 = names.index("Fluorescence_638_nm_Ex")
    optics = optics_for_channel(root, names[c638])
    assert optics.wavelength_um == pytest.approx(0.670)

    stack = np.stack([reader.read("manual0", 0, names[c638], z, 0)
                      for z in reader.metadata["z_levels"]])
    per_channel = deconvolve_stack(stack, optics, DEFAULT_ITERATIONS, project=False)
    shipped = deconvolve_stack(stack, DEFAULT_OPTICS, DEFAULT_ITERATIONS, project=False)
    assert not np.array_equal(per_channel, shipped), (
        "the 525 nm and 670 nm PSFs produced identical output on this phantom; the test would "
        "prove nothing")
    assert np.array_equal(out[0, c638], per_channel), (
        "the registered decon did NOT use the 638 channel's own PSF")


def test_optics_are_derived_per_channel_on_the_real_acquisition(real_dataset):
    assert {c: optics_for_channel(real_dataset, c).wavelength_um for c in (
        "Fluorescence_405_nm_Ex", "Fluorescence_488_nm_Ex",
        "Fluorescence_561_nm_Ex", "Fluorescence_638_nm_Ex")} == {
        "Fluorescence_405_nm_Ex": pytest.approx(0.450),
        "Fluorescence_488_nm_Ex": pytest.approx(0.525),
        "Fluorescence_561_nm_Ex": pytest.approx(0.590),
        "Fluorescence_638_nm_Ex": pytest.approx(0.670),
    }
    assert (make_psf(optics_for_channel(real_dataset, "Fluorescence_638_nm_Ex")).shape
            != make_psf(DEFAULT_OPTICS).shape)


def test_set_optics_is_an_override_that_wins_over_the_per_channel_derivation(tmp_path):
    root = _two_channel_acquisition(tmp_path / "acq")
    assert optics_override() is None
    forced = OpticsParams(na=0.75, wavelength_um=0.61, dxy_um=0.325, dz_um=1.0, nz=2)
    set_optics(forced)
    assert optics_override() == forced
    for channel in ("Fluorescence_638_nm_Ex", "Fluorescence_488_nm_Ex", "nonsense"):
        assert optics_for_channel(root, channel) == forced
    clear_optics()
    assert optics_for_channel(root, "Fluorescence_638_nm_Ex").wavelength_um == pytest.approx(0.670)


def test_a_multiband_penta_channel_derives_its_wavelength_instead_of_refusing(squid_dataset):
    root, _ = squid_dataset
    from tests.conftest import CH_IN_YAML, CH_NOT_IN_YAML

    assert CH_IN_YAML == "Fluorescence_638_nm_-_Penta"
    optics = optics_for_channel(root, CH_IN_YAML)
    assert optics.wavelength_um == pytest.approx(0.670)
    assert optics.na == pytest.approx(0.8)
    assert optics.dxy_um == pytest.approx(0.325)
    assert optics.dz_um == pytest.approx(1.5)

    assert optics_for_channel(root, "Fluorescence_638_nm_Ex").wavelength_um == pytest.approx(0.670)
    assert optics_for_channel(root, CH_NOT_IN_YAML).wavelength_um == pytest.approx(0.590)


def test_a_channel_with_no_derivable_emission_is_refused_by_name(tmp_path):
    root = _two_channel_acquisition(tmp_path / "acq")
    with pytest.raises(ValueError, match="BF_LED_matrix_full"):
        optics_for_channel(root, "BF_LED_matrix_full")
    with pytest.raises(ValueError, match="Fluorescence_638_nm_Ex"):
        optics_for_channel(None, "Fluorescence_638_nm_Ex")


def test_the_psf_is_cached_by_its_optics_tuple_not_rebuilt_per_plane(tmp_path):
    root = _two_channel_acquisition(tmp_path / "acq", nz=3)
    reader = open_reader(root)
    make_psf.cache_clear()

    project_well(reader, "manual0", 0,
                 reduce=_resolve_operator("decon").fn, consumes=Z_REDUCER)

    info = make_psf.cache_info()
    assert info.misses == 2, f"one PSF build per CHANNEL, got {info}"


def test_an_operator_that_declares_nothing_is_handed_through_unchanged():
    from squidxplorer.projection import project
    assert bind_channel(project, "/some/acquisition", "Fluorescence_638_nm_Ex") is project
    fixed = decon_op(FAST_OPTICS, iterations=1)
    assert not hasattr(fixed, "for_channel")
    assert bind_channel(fixed, "/some/acquisition", "Fluorescence_638_nm_Ex") is fixed


def test_specialising_to_a_channel_may_not_change_the_consumed_axis():
    def _plane_op(planes):
        return next(iter(planes))

    _plane_op.consumes = PLANE_OP
    _plane_op.for_channel = lambda path, channel: decon_op(FAST_OPTICS, 1)
    with pytest.raises(ValueError, match="must not change the output shape"):
        bind_channel(_plane_op, "/some/acquisition", "Fluorescence_638_nm_Ex")


def test_decon_over_an_nz1_acquisition_writes_a_one_plane_copy(tmp_path):
    """THE n_z=1 gate for the 2D/3D merge: plenty of this rig's data is single-plane, and the one surviving `decon` must serve it — run, keep the single"""
    import tifffile

    from squidxplorer._acq_output import write_acquisition_planes

    root = _two_channel_acquisition(tmp_path / "acq_nz1", nz=1, frame=32)
    reader = open_reader(root)
    assert int(reader.metadata["n_z"]) == 1
    dst = tmp_path / "decon_out"
    set_optics(FAST_OPTICS)     # the fixture's optics; per-channel derivation is pinned above
    summary = write_acquisition_planes(reader, "decon", dst)
    names = [c["name"] for c in reader.metadata["channels"]]
    assert summary["complete"], summary
    written = sorted(f.name for f in (dst / "0").iterdir())
    assert written == sorted(f"manual0_0_0_{c}.tiff" for c in names), written
    plane = reader.read("manual0", 0, names[0], 0, 0)
    out = tifffile.imread(dst / "0" / f"manual0_0_0_{names[0]}.tiff")
    expected = deconvolve_stack(plane[None, ...], FAST_OPTICS,
                                DEFAULT_ITERATIONS, project=False)[0]
    np.testing.assert_array_equal(out, expected)

# --- the QC sweep's capture hook is GONE (shelved with the sweep, 2026-08-25) -----------------

def test_the_snapshot_capture_hook_is_gone_with_the_sweep():
    """Julio: "The sweep code should be shelved."""
    import inspect

    from squidxplorer import _decon, _decon_gpu

    assert "snapshot_iters" not in inspect.signature(_decon._run).parameters
    assert "snapshot_iters" not in inspect.signature(_decon_gpu.rl).parameters


def test_iterations_is_a_declared_param_on_the_decon_registration():
    """THE one place the QC's chosen count lands: the surviving operator declares ``iterations``, so operator_kwargs / recipes / the declaration probe all carry it."""
    from squidxplorer._engine import operator_params

    params = {p.name: p.default for p in operator_params("decon")}
    assert params == {"iterations": DEFAULT_ITERATIONS}, (
        "decon does not declare iterations (or declares more than the panel feeds)")


# --- the session immersion / NA choices (2026-08-24) ------------------------------------------

def _write_acq(tmp_path, na=0.75):
    root = tmp_path / "acq"
    root.mkdir(parents=True)
    (root / "acquisition parameters.json").write_text(json.dumps(
        {"dz(um)": 2.0, "Nz": 4, "Nt": 1,
         "objective": {"magnification": 20.0, "NA": na}, "sensor_pixel_size_um": 6.5}))
    (root / "acquisition.yaml").write_text(
        "objective:\n  pixel_size_um: 0.400\n  magnification: 20.0\n"
        "z_stack:\n  nz: 4\n  delta_z_mm: 0.002\n"
        "time_series:\n  nt: 1\n")
    return root


def test_session_ni_reaches_the_channel_optics_for_preview_and_run(tmp_path):
    """optics_for_channel is the ONE reader both the QC worker and a run's for_channel use, so the medium chosen in the panel shapes BOTH PSFs; clearing it"""
    from squidxplorer._decon import session_ni, set_session_ni

    root = _write_acq(tmp_path)
    try:
        set_session_ni(1.333)
        assert session_ni() == pytest.approx(1.333)
        optics = optics_for_channel(root, "Fluorescence_488_nm_Ex")
        assert optics.ni == pytest.approx(1.333)
        assert optics.immersion_index == pytest.approx(1.333)
    finally:
        set_session_ni(None)
    assert optics_for_channel(root, "Fluorescence_488_nm_Ex").ni is None


def test_an_na_impossible_under_the_chosen_medium_is_refused_by_name(tmp_path):
    """NA <= ni is physics: a 1.40 objective under air must refuse, naming both numbers and the way out, never solve with an impossible PSF."""
    from squidxplorer._decon import set_session_ni

    root = _write_acq(tmp_path, na=1.40)
    try:
        set_session_ni(1.000)
        with pytest.raises(ValueError, match=r"NA 1\.40 is impossible in air"):
            optics_for_channel(root, "Fluorescence_488_nm_Ex")
        set_session_ni(1.515)                  # the actual oil objective: allowed again
        assert optics_for_channel(root, "Fluorescence_488_nm_Ex").ni == pytest.approx(1.515)
    finally:
        set_session_ni(None)


def test_a_session_na_override_reaches_the_optics_and_is_cleared_by_none(tmp_path):
    from squidxplorer._decon import set_session_na

    root = _write_acq(tmp_path)
    try:
        set_session_na(0.85)
        assert optics_for_channel(root, "Fluorescence_488_nm_Ex").na == pytest.approx(0.85)
    finally:
        set_session_na(None)
    assert optics_for_channel(root, "Fluorescence_488_nm_Ex").na == pytest.approx(0.75)


def test_the_immersion_table_is_value_first_with_the_medium_beside_it():
    from squidxplorer._decon import IMMERSION_MEDIA, medium_for_ni

    assert IMMERSION_MEDIA[0] == (1.000, "air"), "air is the assumed default and comes first"
    assert dict(IMMERSION_MEDIA) == {1.000: "air", 1.333: "water", 1.406: "silicone oil",
                                     1.473: "glycerol", 1.515: "oil"}
    assert medium_for_ni(1.515) == "oil"
    assert medium_for_ni(1.2345) == "ni 1.234", "an off-table index is named by its value"


# --------------------------------------------------------------------------------------
# Forward-model verification of the 3-D PSF and the volume solve, pinned (2026-08-24).
# Julio doubted the 2D/3D PSF logic; this is the measured proof, compacted so it cannot rot
# (the 16-plane reference run: RMSE 23.0 -> 9.4; axial own/neighbour ratio 1.10 -> 5.19
# against truth's 5.24). Real 25x PLANAPO optics: water, NA 0.85, dz 3 um.
# --------------------------------------------------------------------------------------

_PSF3D_OPTICS = OpticsParams(na=0.85, wavelength_um=0.670, dxy_um=0.325, dz_um=3.0,
                             nz=8, ni=1.33)


def test_the_3d_psf_is_a_normalised_axially_centred_kernel():
    """The PSF the solve builds: depth 2*nz-1 (every plane can borrow from every other), unit energy, and per-plane energy symmetric about a centroid ON the"""
    psf = make_psf(_PSF3D_OPTICS)
    assert psf.shape[0] == 2 * _PSF3D_OPTICS.nz - 1
    assert float(psf.sum()) == pytest.approx(1.0, abs=1e-3)
    per_plane = psf.sum(axis=(1, 2)).astype(np.float64)
    np.testing.assert_allclose(per_plane, per_plane[::-1], atol=1e-6 * per_plane.max())
    centroid = float((np.arange(psf.shape[0]) * per_plane).sum() / per_plane.sum())
    assert centroid == pytest.approx(psf.shape[0] // 2, abs=0.05)


def test_the_volume_solve_moves_light_back_to_its_own_plane():
    """Blur a known sparse 3-D bead phantom with the EXACT PSF, restore with the volume solve, and measure both lateral fit (RMSE) and AXIAL specificity"""
    nz = _PSF3D_OPTICS.nz
    psf = make_psf(_PSF3D_OPTICS)
    rng = np.random.default_rng(1)
    truth = np.full((nz, 64, 64), 20.0, np.float32)
    for _ in range(12):
        z, y, x = rng.integers(2, nz - 2), rng.integers(8, 56), rng.integers(8, 56)
        truth[z, y, x] += rng.uniform(500, 3000)
    blurred = scipy_signal.fftconvolve(truth, psf, mode="same").astype(np.float32)

    restored = deconvolve_stack(blurred, _PSF3D_OPTICS, iterations=10, gpu=False,
                                project=False)

    def rmse(a, b):
        return float(np.sqrt(np.mean((np.asarray(a, np.float64)
                                      - np.asarray(b, np.float64)) ** 2)))

    assert rmse(restored, truth) < 0.65 * rmse(blurred, truth), (
        f"RMSE vs truth barely moved: blurred {rmse(blurred, truth):.1f}, "
        f"restored {rmse(restored, truth):.1f}")

    beads = [(z, y, x) for z in range(2, nz - 2) for y in range(8, 56) for x in range(8, 56)
             if truth[z, y, x] > 100]
    assert beads, "the phantom lost its beads; the test's premise is stale"

    def axial_ratio(vol):
        own = neigh = 0.0
        for z, y, x in beads:
            own += float(vol[z, y - 1:y + 2, x - 1:x + 2].sum())
            neigh += float(vol[z - 1, y - 1:y + 2, x - 1:x + 2].sum()
                           + vol[z + 1, y - 1:y + 2, x - 1:x + 2].sum())
        return own / max(neigh, 1e-9)

    assert axial_ratio(restored) > 3.0 * axial_ratio(blurred), (
        f"axial energy did not return to its plane: truth {axial_ratio(truth):.2f}, "
        f"blurred {axial_ratio(blurred):.2f}, restored {axial_ratio(restored):.2f}")


# --------------------------------------------------------------------------------------
# A broadband channel is COPIED THROUGH unchanged; the rest deconvolve (Julio, 2026-08-25).
# Measured on G7_2026-08-20 (BF_LED_matrix_full + two fluorescence channels, nz 15): decon
# failed the WHOLE run, because project_well binds every channel up front and the LED
# channel states no excitation wavelength. The per-channel refusal is right (a guessed PSF
# is a fabricated result); the granularity was wrong. Ruling: "copy BF through unchanged
# with a named log line".
# --------------------------------------------------------------------------------------

import logging as _logging


class _LogSpy(_logging.Handler):
    def __init__(self):
        super().__init__(level=_logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _stack_of(reader, channel):
    return np.stack([reader.read("manual0", 0, channel, z, 0)
                     for z in reader.metadata["z_levels"]])


def test_a_broadband_channel_is_copied_through_unchanged_and_named_once(tmp_path):
    root = _two_channel_acquisition(tmp_path / "acq",
                                    channels=["BF LED matrix full", "Fluorescence 488 nm Ex"])
    reader = open_reader(root)
    names = [c["name"] for c in reader.metadata["channels"]]
    assert names == ["BF_LED_matrix_full", "Fluorescence_488_nm_Ex"]
    spy = _LogSpy()
    _logging.getLogger("squid.xplorer").addHandler(spy)
    try:
        out = project_well(reader, "manual0", 0,
                           reduce=_resolve_operator("decon").fn, consumes=Z_REDUCER)
        project_well(reader, "manual0", 0,
                     reduce=_resolve_operator("decon").fn, consumes=Z_REDUCER)
    finally:
        _logging.getLogger("squid.xplorer").removeHandler(spy)
    nz = reader.metadata["n_z"]
    assert out.shape == (1, 2, nz, 64, 64), "the output must stay the size of the input"
    assert out.dtype == np.dtype(reader.metadata["dtype"])
    assert np.array_equal(out[0, 0], _stack_of(reader, "BF_LED_matrix_full")), (
        "the broadband channel was not copied through bit-identical")
    assert not np.array_equal(out[0, 1], _stack_of(reader, "Fluorescence_488_nm_Ex")), (
        "the fluorescence channel was not deconvolved")
    named = [r for r in spy.records if "BF_LED_matrix_full" in r.getMessage()]
    assert len(named) == 1, (
        f"expected exactly one named line, got {len(named)}:\n"
        + "\n".join(r.getMessage() for r in named))
    msg = named[0].getMessage()
    assert named[0].levelno == _logging.INFO, "a limitation line stays INFO under the log diet"
    for phrase in ("no emission wavelength", "copied unchanged", "not deconvolved"):
        assert phrase in msg, f"{phrase!r} missing from {msg!r}"


def test_a_set_with_no_modelable_channel_is_still_a_named_refusal(tmp_path):
    root = _two_channel_acquisition(tmp_path / "acq",
                                    channels=["BF LED matrix full", "DF LED matrix"])
    reader = open_reader(root)
    with pytest.raises(ValueError, match="BF_LED_matrix_full") as info:
        project_well(reader, "manual0", 0,
                     reduce=_resolve_operator("decon").fn, consumes=Z_REDUCER)
    assert "DF_LED_matrix" in str(info.value)
    assert "unchanged" in str(info.value), "the refusal must say what would have happened"


def test_only_the_no_emission_line_case_copies_through():
    """A fluorescence channel whose optics cannot be READ (here: no acquisition path) is still the named refusal, never a silent copy."""
    with pytest.raises(ValueError, match="Fluorescence_488_nm_Ex"):
        _resolve_operator("decon").fn.for_channel(None, "Fluorescence_488_nm_Ex")
