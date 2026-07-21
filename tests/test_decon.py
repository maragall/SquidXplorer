"""IMA-223 deconvolution operator — numerical property tests, not smoke tests.

The property that matters: deconvolution must move a KNOWN-blurred image back TOWARD the
ground truth it was blurred from. A smoke test ("it returned an array of the right shape")
would pass on a no-op, so every test here measures error against a known truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from squidmip import add_projector, available_projectors, project_well, projector_consumes
from squidmip._decon import (
    DEFAULT_ITERATIONS,
    DEFAULT_SIGMA_PX,
    decon_op,
    deconvolve,
    richardson_lucy_gaussian,
)
from squidmip.projection import PLANE_OP
from squidmip.reader import open_reader

scipy_ndimage = pytest.importorskip("scipy.ndimage")


def _ground_truth(size: int = 96, seed: int = 0) -> np.ndarray:
    """A sparse-puncta phantom — the shape fluorescence actually has, and the shape RL assumes
    (Poisson counts on a dark background). A smooth gradient would be nearly unchanged by any
    deconvolution and would prove nothing."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 20.0, dtype=np.float32)
    ys = rng.integers(8, size - 8, 40)
    xs = rng.integers(8, size - 8, 40)
    for y, x in zip(ys, xs):
        img[y, x] += rng.uniform(400, 2000)
    return img


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return scipy_ndimage.gaussian_filter(img, sigma, mode="reflect")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


# --- the core numerical property ---------------------------------------------------------

def test_decon_moves_a_known_blur_back_toward_ground_truth():
    truth = _ground_truth()
    blurred = _blur(truth, 2.0)

    restored = richardson_lucy_gaussian(blurred, sigma=2.0, iterations=30)

    before, after = _rmse(blurred, truth), _rmse(restored, truth)
    assert after < before * 0.6, f"RL did not sharpen: rmse {before:.1f} -> {after:.1f}"


def test_decon_recovers_peak_amplitude_lost_to_blur():
    """The specific thing blur destroys is peak height; RL must give it back, not just
    reshuffle energy. Checks the tallest punctum, which blur flattens the most."""
    truth = _ground_truth()
    blurred = _blur(truth, 2.0)
    restored = richardson_lucy_gaussian(blurred, sigma=2.0, iterations=30)

    assert blurred.max() < truth.max() * 0.5          # the blur really did flatten it
    assert restored.max() > blurred.max() * 1.5       # and RL really did restore it
    assert restored.max() <= truth.max() * 1.5        # without inventing 10x signal


def test_more_iterations_monotonically_reduce_error_over_the_useful_range():
    truth = _ground_truth()
    blurred = _blur(truth, 2.0)
    errs = [_rmse(richardson_lucy_gaussian(blurred, sigma=2.0, iterations=n), truth)
            for n in (1, 5, 15, 30)]
    assert errs == sorted(errs, reverse=True), f"error not monotone in iterations: {errs}"


def test_zero_iterations_is_the_identity():
    """The n=0 case must be a pure passthrough, so 'how many iterations' has an unambiguous
    zero point and a benchmark can measure the loop cost against it."""
    blurred = _blur(_ground_truth(), 2.0).astype(np.uint16)
    assert np.array_equal(richardson_lucy_gaussian(blurred, sigma=2.0, iterations=0), blurred)


# --- energy / dtype / non-mutation contracts ---------------------------------------------

def test_rl_conserves_total_intensity_to_within_a_percent():
    """RL is flux-conserving by construction (the multiplicative update has total-count as a
    fixed point). A drift here means the update or the boundary handling is wrong."""
    blurred = _blur(_ground_truth(), 2.0)
    restored = richardson_lucy_gaussian(blurred, sigma=2.0, iterations=30)
    assert abs(restored.sum() - blurred.sum()) / blurred.sum() < 0.01


def test_dtype_is_preserved_and_the_input_plane_is_never_mutated():
    plane = _blur(_ground_truth(), 2.0).astype(np.uint16)
    before = plane.copy()
    out = richardson_lucy_gaussian(plane, sigma=2.0, iterations=5)
    assert out.dtype == np.uint16
    assert np.array_equal(plane, before), "deconvolve mutated the caller's plane"


def test_uint16_output_is_clipped_not_wrapped():
    """A bright punctum + RL overshoot can exceed 65535. Wrapping would turn the brightest
    pixel in the image into a black one — the classic silent integer-overflow artifact."""
    plane = np.full((32, 32), 60000, dtype=np.uint16)
    plane[16, 16] = 65535
    out = richardson_lucy_gaussian(plane, sigma=1.0, iterations=20)
    assert out.min() >= 0 and out.max() <= 65535
    assert out[16, 16] > 60000, "the bright pixel wrapped to a dark one"


# --- the boundary-mode trap (numpy 'reflect' != scipy 'reflect') --------------------------

def test_boundary_mode_default_matches_numpy_symmetric_not_numpy_reflect():
    """scipy's 'reflect' is (d c b a | a b c d) — numpy calls that 'symmetric'. numpy's
    'reflect' (d c b | a b c d) is scipy's 'mirror'. Pinning the equivalence on data so a
    future edit to mode= cannot silently introduce edge artifacts that look like signal."""
    row = np.arange(1, 9, dtype=np.float64)[None, :]
    k_sigma = 1.0
    scipy_reflect = scipy_ndimage.gaussian_filter(row, k_sigma, mode="reflect")

    pad = 8
    np_symmetric = scipy_ndimage.gaussian_filter(
        np.pad(row, ((0, 0), (pad, pad)), mode="symmetric"), k_sigma, mode="nearest"
    )[:, pad:-pad]
    np_reflect = scipy_ndimage.gaussian_filter(
        np.pad(row, ((0, 0), (pad, pad)), mode="reflect"), k_sigma, mode="nearest"
    )[:, pad:-pad]

    assert np.allclose(scipy_reflect, np_symmetric, atol=1e-9)
    assert not np.allclose(scipy_reflect, np_reflect, atol=1e-6)


def test_flat_field_stays_flat_at_the_edges_no_boundary_artifact():
    """A constant image is the sharpest possible boundary test: any wrong padding shows up as
    a bright or dark rim. The rim must be indistinguishable from the interior."""
    plane = np.full((64, 64), 1000.0, dtype=np.float32)
    out = richardson_lucy_gaussian(plane, sigma=3.0, iterations=20)
    rim = np.concatenate([out[0], out[-1], out[:, 0], out[:, -1]])
    assert np.allclose(rim, 1000.0, rtol=2e-3), f"edge artifact: rim range {rim.min()}..{rim.max()}"


def test_wrap_boundary_would_bleed_across_the_edge_and_reflect_does_not():
    """Evidence that the mode choice is load-bearing: a bright edge column bleeds to the
    OPPOSITE edge under 'wrap' (what an FFT-based RL does by default) and does not under
    the default reflect."""
    plane = np.full((48, 48), 10.0, dtype=np.float32)
    plane[:, 0] = 5000.0
    reflected = richardson_lucy_gaussian(plane, sigma=3.0, iterations=10, mode="reflect")
    wrapped = richardson_lucy_gaussian(plane, sigma=3.0, iterations=10, mode="wrap")
    assert wrapped[:, -1].mean() > reflected[:, -1].mean() * 5


# --- prior art cross-check ----------------------------------------------------------------

def test_agrees_with_scikit_image_richardson_lucy_on_the_interior():
    """Cross-check against the reference implementation (skimage.restoration.richardson_lucy)
    with an EXPLICIT Gaussian PSF. They differ at the border by construction — skimage
    convolves via FFT — so the comparison is on the interior, where the algorithms must be the
    same algorithm. This is what justifies the kernel-free two-blur shortcut."""
    restoration = pytest.importorskip("skimage.restoration")
    sigma, n_iter, size = 2.0, 10, 96

    truth = _ground_truth(size)
    blurred = _blur(truth, sigma)
    scale = blurred.max()

    # explicit PSF for the reference: a normalised Gaussian on an odd grid
    r = int(4 * sigma) | 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    psf = np.exp(-(yy ** 2 + xx ** 2) / (2 * sigma ** 2)).astype(np.float64)
    psf /= psf.sum()

    # skimage seeds its estimate with a constant 0.5 (on its normalised input), so the seed is
    # matched explicitly here — otherwise the two would differ only because they started from
    # different places, which would prove nothing about the update rule.
    ref = restoration.richardson_lucy(blurred / scale, psf, num_iter=n_iter, clip=False) * scale
    ours = richardson_lucy_gaussian(
        blurred, sigma=sigma, iterations=n_iter,
        init=np.full(blurred.shape, 0.5 * scale, np.float32),
    )

    m = slice(3 * r, size - 3 * r)
    rel = np.abs(ours[m, m] - ref[m, m]).max() / ref[m, m].max()
    assert rel < 0.05, f"interior disagrees with skimage by {rel:.3f} of peak"


# --- registry / engine seam ----------------------------------------------------------------

def test_decon_is_registered_as_a_plane_op():
    assert "decon" in available_projectors()
    assert projector_consumes("decon") == PLANE_OP


def test_decon_op_factory_produces_a_plane_op_and_is_registrable():
    op = decon_op(sigma=1.0, iterations=3)
    assert op.consumes == PLANE_OP
    name = "decon_test_factory"
    if name not in available_projectors():
        add_projector(name, op)
    assert projector_consumes(name) == PLANE_OP
    plane = _blur(_ground_truth(32), 1.0).astype(np.uint16)
    assert op([plane]).shape == plane.shape


def test_decon_op_refuses_a_whole_z_stack():
    """The plane_op guard: handed a stack it must raise, not silently deconvolve plane 0 and
    throw the rest away."""
    op = decon_op(sigma=1.0, iterations=1)
    planes = [np.zeros((8, 8), np.uint16), np.zeros((8, 8), np.uint16)]
    with pytest.raises(ValueError, match="more than one plane"):
        op(planes)


def test_default_module_operator_uses_the_documented_defaults():
    plane = _blur(_ground_truth(48), DEFAULT_SIGMA_PX).astype(np.uint16)
    assert np.array_equal(
        deconvolve(plane),
        richardson_lucy_gaussian(plane, sigma=DEFAULT_SIGMA_PX, iterations=DEFAULT_ITERATIONS),
    )


def test_project_well_with_decon_keeps_z_at_full_depth(squid_dataset):
    """The IMA-210 contract, on data: a plane-op does NOT collapse z."""
    root, _ = squid_dataset
    reader = open_reader(root)
    out = project_well(reader, "B2", 0, reduce=decon_op(sigma=0.5, iterations=2))
    n_z = len(reader.metadata["z_levels"])
    n_c = len(reader.metadata["channels"])
    assert out.shape == (reader.metadata["n_t"], n_c, n_z, 4, 4)
    assert out.dtype == reader.metadata["dtype"]
