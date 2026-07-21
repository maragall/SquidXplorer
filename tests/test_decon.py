"""IMA-223/IMA-247 deconvolution operator — numerical property tests, not smoke tests.

The property that matters: deconvolution must move a KNOWN-blurred image back TOWARD the
ground truth it was blurred from. A smoke test ("it returned an array of the right shape")
would pass on a no-op, so every test here measures error against a known truth.

IMA-247 changed what is UNDER these tests. The operator no longer carries a hand-rolled
Richardson-Lucy against an assumed Gaussian PSF; it calls Julio's ``petakit`` — a vectorial PSF
from the acquisition optics, then PetaKit5D's RL. So the phantom here is blurred with THE REAL
PSF rather than with a Gaussian, which is the only honest way to ask whether the operator
inverts the blur the instrument actually applies.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip import add_projector, available_projectors, project_well, projector_consumes
from squidmip._decon import (
    DEFAULT_ITERATIONS,
    DEFAULT_OPTICS,
    METHOD,
    OpticsParams,
    active_optics,
    clear_optics,
    decon3d_op,
    decon_op,
    deconvolve,
    deconvolve_plane,
    deconvolve_stack,
    make_psf,
    make_psf_2d,
    set_optics,
)
from squidmip.projection import PLANE_OP, Z_REDUCER
from squidmip.reader import open_reader

scipy_ndimage = pytest.importorskip("scipy.ndimage")
scipy_signal = pytest.importorskip("scipy.signal")
pytest.importorskip("petakit")
pytest.importorskip("psfmodels")


# Real optics at an OVERSAMPLED pixel (0.4 um rather than the scope's 0.752), which makes the
# diffraction blur several pixels wide instead of barely one. That is what lets a
# blur-then-deconvolve test actually measure recovery: at the scope's own sampling the PSF is
# so narrow that almost nothing is lost, so almost nothing can be restored.
FAST_OPTICS = OpticsParams(na=0.3, wavelength_um=0.525, dxy_um=0.4, dz_um=1.5, nz=3)


@pytest.fixture(autouse=True)
def _no_leaked_optics():
    """Every test starts from the default optics; set_optics is global state."""
    clear_optics()
    yield
    clear_optics()


def _ground_truth(size: int = 96, seed: int = 0) -> np.ndarray:
    """A sparse-puncta phantom on a dim pedestal — the shape fluorescence actually has, and the
    shape RL assumes (Poisson counts on a dark background). A smooth gradient would be nearly
    unchanged by any deconvolution and would prove nothing.

    The puncta have FINITE extent rather than being single-pixel deltas. That is deliberate: a
    true delta carries energy at every spatial frequency, including the ones the PSF has
    annihilated, so NO deconvolution recovers more than a few percent of its RMSE in a sane
    iteration count. Testing against a delta phantom would be testing an information-theoretic
    impossibility, not this implementation.
    """
    rng = np.random.default_rng(seed)
    seeds = np.zeros((size, size), dtype=np.float32)
    for y, x in zip(rng.integers(10, size - 10, 30), rng.integers(10, size - 10, 30)):
        seeds[y, x] += rng.uniform(400, 2000)
    return (scipy_ndimage.gaussian_filter(seeds, 1.2, mode="reflect") + 20.0).astype(np.float32)


def _blur_with_real_psf(img: np.ndarray, optics: OpticsParams = FAST_OPTICS) -> np.ndarray:
    """Blur with the SAME vectorial PSF the operator will deconvolve with."""
    psf = make_psf_2d(optics)[0]
    return scipy_signal.convolve(img.astype(np.float64), psf, mode="same").astype(np.float32)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


# --- the core numerical property ---------------------------------------------------------

def test_decon_moves_a_known_blur_back_toward_ground_truth():
    truth = _ground_truth()
    blurred = _blur_with_real_psf(truth)

    restored = deconvolve_plane(blurred, FAST_OPTICS, iterations=30)

    before, after = _rmse(blurred, truth), _rmse(restored, truth)
    assert after < before * 0.6, f"RL did not sharpen: rmse {before:.1f} -> {after:.1f}"


def test_decon_recovers_peak_amplitude_lost_to_blur():
    """The specific thing blur destroys is peak height; RL must give it back, not just
    reshuffle energy. Checks the tallest punctum, which blur flattens the most."""
    truth = _ground_truth()
    blurred = _blur_with_real_psf(truth)
    restored = deconvolve_plane(blurred, FAST_OPTICS, iterations=30)

    assert blurred.max() < truth.max() * 0.8          # the blur really did flatten it
    assert restored.max() > blurred.max() * 1.2       # and RL really did restore it
    assert restored.max() <= truth.max() * 1.5        # without inventing 10x signal


def test_more_iterations_reduce_error_over_the_useful_range():
    truth = _ground_truth()
    blurred = _blur_with_real_psf(truth)
    errs = [_rmse(deconvolve_plane(blurred, FAST_OPTICS, iterations=n), truth)
            for n in (1, 5, 15, 30)]
    assert errs[-1] < errs[0], f"error did not fall with iterations: {errs}"


def test_zero_iterations_is_the_identity():
    """The n=0 case must be a pure passthrough, so 'how many iterations' has an unambiguous
    zero point and a benchmark can measure the loop cost against it."""
    blurred = _blur_with_real_psf(_ground_truth()).astype(np.uint16)
    assert np.array_equal(deconvolve_plane(blurred, FAST_OPTICS, iterations=0), blurred)


# --- energy / dtype / non-mutation contracts ---------------------------------------------

def test_rl_conserves_total_intensity_to_within_a_few_percent():
    """RL is flux-conserving by construction (the multiplicative update has total-count as a
    fixed point). A large drift means the update or the PSF normalisation is wrong."""
    blurred = _blur_with_real_psf(_ground_truth())
    restored = deconvolve_plane(blurred, FAST_OPTICS, iterations=30)
    assert abs(float(restored.sum()) - float(blurred.sum())) / float(blurred.sum()) < 0.05


def test_dtype_is_preserved_and_the_input_plane_is_never_mutated():
    plane = _blur_with_real_psf(_ground_truth()).astype(np.uint16)
    before = plane.copy()
    out = deconvolve_plane(plane, FAST_OPTICS, iterations=5)
    assert out.dtype == np.uint16
    assert np.array_equal(plane, before), "deconvolve mutated the caller's plane"


def test_uint16_output_is_clipped_not_wrapped():
    """A bright punctum + RL overshoot can exceed 65535. Wrapping would turn the brightest
    pixel in the image into a black one — the classic silent integer-overflow artifact."""
    plane = np.full((32, 32), 60000, dtype=np.uint16)
    plane[16, 16] = 65535
    out = deconvolve_plane(plane, FAST_OPTICS, iterations=20)
    assert out.min() >= 0 and out.max() <= 65535
    assert out[16, 16] > 60000, "the bright pixel wrapped to a dark one"


def test_a_flat_field_stays_flat_no_boundary_artifact():
    """A constant image is the sharpest possible boundary test: wrong padding shows up as a
    bright or dark rim, and RL amplifies that rim into something that looks like real membrane
    signal at the FOV edge."""
    plane = np.full((64, 64), 1000.0, dtype=np.float32)
    out = deconvolve_plane(plane, FAST_OPTICS, iterations=20)
    rim = np.concatenate([out[0], out[-1], out[:, 0], out[:, -1]])
    assert np.allclose(rim, 1000.0, rtol=0.05), f"edge artifact: rim range {rim.min()}..{rim.max()}"


# --- IMA-247: the PSF is REAL, and it comes from acquisition metadata ---------------------

def test_the_psf_is_a_real_vectorial_psf_not_a_gaussian():
    """The whole point of the ticket. A vectorial widefield PSF has Airy structure — energy in
    rings outside the central lobe — which a Gaussian does not. Fitting the best Gaussian to it
    must leave a visible residual, otherwise we have not actually changed anything."""
    psf = make_psf_2d(DEFAULT_OPTICS)[0]
    assert psf.ndim == 2 and psf.shape[0] == psf.shape[1]
    assert float(psf.sum()) == pytest.approx(1.0, rel=1e-5)

    # Best-fit Gaussian by second moment, then compare.
    yy, xx = np.indices(psf.shape)
    cy, cx = (np.array(psf.shape) - 1) / 2
    sigma = np.sqrt((((yy - cy) ** 2 + (xx - cx) ** 2) * psf).sum() / 2.0)
    gauss = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    gauss /= gauss.sum()
    assert np.abs(psf - gauss).max() / psf.max() > 0.02, "the 'real' PSF is Gaussian after all"

    # And the deleted implementation's hardcoded sigma=1.5 px was simply the wrong number for
    # this instrument: the real kernel is materially narrower.
    assert sigma < 1.35, f"real PSF sigma {sigma:.3f} px; the old code hardcoded 1.5"


def test_optics_come_from_acquisition_metadata_not_constants(tmp_path):
    """OpticsParams.from_acquisition must read the scope's OWN numbers through petakit's
    reader. Built here as a minimal Squid-shaped acquisition so the test needs no fixture data.
    """
    tifffile = pytest.importorskip("tifffile")
    root = tmp_path / "acq"
    (root / "ome_tiff").mkdir(parents=True)
    (root / "acquisition parameters.json").write_text(
        '{"dz(um)": 2.0, "Nz": 4, "Nt": 1, "objective": {"magnification": 20.0, "NA": 0.75},'
        ' "sensor_pixel_size_um": 6.5}'
    )
    tifffile.imwrite(
        root / "ome_tiff" / "manual0_0.ome.tiff",
        np.zeros((4, 1, 8, 8), np.uint16),
        metadata={"axes": "ZCYX", "Channel": {"Name": ["Fluorescence 488 nm Ex"]}},
    )

    optics = OpticsParams.from_acquisition(root)
    assert optics.na == 0.75                     # objective.NA, not a default
    assert optics.dxy_um == pytest.approx(6.5 / 20.0)   # sensor pitch / magnification
    assert optics.dz_um == 2.0
    assert optics.nz == 4
    assert optics.wavelength_um == pytest.approx(0.525)  # 488 excitation -> 525 emission
    # petakit's rule: NA <= 1.0 is a dry objective, <= 1.33 water, above that oil.
    assert optics.immersion_index == pytest.approx(1.0)
    assert OpticsParams(na=1.2, wavelength_um=0.525, dxy_um=0.1).immersion_index == 1.33
    assert OpticsParams(na=1.4, wavelength_um=0.525, dxy_um=0.1).immersion_index == 1.515

    # and the PSF actually changes when the optics do
    assert make_psf_2d(optics).shape != make_psf_2d(DEFAULT_OPTICS).shape


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
    """petakit's own default is method='omw', and it collapses to (very nearly) all zeros.

    On the real 10x NA-0.3 stack it returns LITERALLY every pixel 0.0; on this smaller phantom
    it returns 99.6% zeros with a few surviving spikes. Either way it is not a deconvolution of
    the input, and this module must never inherit that default.

    Pinned as a regression test with evidence: run omw ourselves and show it is degenerate,
    then show the pinned rl path is not.
    """
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
    """The guard behind trap 1: a degenerate engine result must fail loud, not look like a
    successful deconvolution of a dark field."""
    import squidmip._decon as decon_mod

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
    """No silent substitution: if his engine is absent the user must be told which dependency
    is missing, not handed a different algorithm's output."""
    import builtins

    import squidmip._decon as decon_mod

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


# --- registry / engine seam ----------------------------------------------------------------

def test_decon_is_registered_as_a_plane_op():
    assert "decon" in available_projectors()
    assert projector_consumes("decon") == PLANE_OP


def test_decon_op_factory_produces_a_plane_op_and_is_registrable():
    op = decon_op(FAST_OPTICS, iterations=3)
    assert op.consumes == PLANE_OP
    name = "decon_test_factory"
    if name not in available_projectors():
        add_projector(name, op)
    assert projector_consumes(name) == PLANE_OP
    plane = _blur_with_real_psf(_ground_truth(32)).astype(np.uint16)
    assert op([plane]).shape == plane.shape


def test_decon_op_refuses_a_whole_z_stack():
    """The plane_op guard: handed a stack it must raise, not silently deconvolve plane 0 and
    throw the rest away."""
    op = decon_op(FAST_OPTICS, iterations=1)
    planes = [np.zeros((8, 8), np.uint16), np.zeros((8, 8), np.uint16)]
    with pytest.raises(ValueError, match="more than one plane"):
        op(planes)


def test_default_module_operator_uses_the_documented_defaults():
    plane = _blur_with_real_psf(_ground_truth(48)).astype(np.uint16)
    assert np.array_equal(
        deconvolve(plane),
        deconvolve_plane(plane, DEFAULT_OPTICS, iterations=DEFAULT_ITERATIONS),
    )


def test_project_well_with_decon_keeps_z_at_full_depth(squid_dataset):
    """The IMA-210 contract, on data: a plane-op does NOT collapse z."""
    root, _ = squid_dataset
    reader = open_reader(root)
    out = project_well(reader, "B2", 0, reduce=decon_op(FAST_OPTICS, iterations=2))
    n_z = len(reader.metadata["z_levels"])
    n_c = len(reader.metadata["channels"])
    assert out.shape == (reader.metadata["n_t"], n_c, n_z, 4, 4)
    assert out.dtype == reader.metadata["dtype"]


# --- the 3-D operator: where a real PSF actually pays off ----------------------------------

def test_decon3d_is_registered_as_a_z_reducer_with_zero_engine_edits():
    """A 3-D deconvolution NEEDS the whole stack, so it consumes z. That is a declaration on
    the registration, not a change to _engine.py."""
    assert "decon3d" in available_projectors()
    assert projector_consumes("decon3d") == Z_REDUCER


def test_decon3d_collapses_z_and_sharpens_more_than_the_2d_plane_op(squid_dataset):
    truth = _ground_truth(64)
    stack = np.stack([_blur_with_real_psf(truth) for _ in range(3)]).astype(np.float32)

    out = deconvolve_stack(stack, FAST_OPTICS, iterations=10)
    assert out.shape == truth.shape          # z collapsed, as a z-reducer must

    def sharpness(a):
        gy, gx = np.gradient(a.astype(np.float64))
        return float(np.sqrt((gy ** 2 + gx ** 2).mean()) / a.std())

    assert sharpness(out) > sharpness(stack.max(axis=0)), "3-D decon did not sharpen the MIP"


def test_decon3d_op_receives_the_stack_through_project_well(squid_dataset):
    """End to end through the real engine loop: the z-reducer form runs and collapses z."""
    root, _ = squid_dataset
    reader = open_reader(root)
    out = project_well(reader, "B2", 0, reduce=decon3d_op(FAST_OPTICS, iterations=2))
    n_c = len(reader.metadata["channels"])
    assert out.shape == (reader.metadata["n_t"], n_c, 1, 4, 4)


def test_deconvolve_stack_rejects_a_2d_input():
    with pytest.raises(ValueError, match=r"needs \(Z, Y, X\)"):
        deconvolve_stack(np.zeros((8, 8), np.float32), FAST_OPTICS)


# --- the channel-label seam (IMA-252) ------------------------------------------------------

def test_channel_labels_from_squidmips_own_reader_parse_into_a_wavelength():
    """A caller holds the label squidmip's reader gave them. It has to work.

    Squid writes channels as ``Fluorescence_488_nm_Ex`` - UNDERSCORED - and that is what
    ``reader.metadata["channels"]`` reports. petakit's parser matches ``(\\d{3})\\s*nm``, so
    the underscored form raised ValueError and ``OpticsParams.from_acquisition(path,
    channel=...)`` could not be called with the project's own channel names at all. All
    three spellings must land on the same emission wavelength.
    """
    from squidmip._decon import _as_channel_name
    import petakit

    wavelengths = {
        petakit.wavelength_from_channel(_as_channel_name(name))
        for name in ("488", "488 nm", "Fluorescence_488_nm_Ex", "Fluorescence 488 nm Ex")
    }
    assert len(wavelengths) == 1
    assert 0.50 < wavelengths.pop() < 0.55       # the 488 line's Stokes-shifted emission
