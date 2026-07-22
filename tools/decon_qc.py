#!/usr/bin/env python3
"""IMA-252 command line: run the RL semi-convergence sweep and write the QC outputs.

This file is now ONLY the command line. The measurement (:func:`halo_core_ratio`,
:func:`recommend`), the structure picking and the turbo rendering live in
``squidmip._decon_qc``, because the GUI's deconvolution panel needs exactly the same
functions and a package cannot import a script directory. One implementation, two front
ends - see that module's docstring.

Usage
-----
    python tools/decon_qc.py                       # defaults: tissue set, manual0/fov 0, 488
    python tools/decon_qc.py --iterations 12 --out /tmp/qc
    python tools/decon_qc.py --region manual1 --fov 3 --channel Fluorescence_405_nm_Ex

Outputs (into --out): ``decon_qc_montage.png``, ``decon_qc_curve.png``, ``decon_qc.csv``.
The datasets are opened READ ONLY and nothing is ever written back next to them.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

# Run from anywhere: import the repo this file lives in, not whatever `squidmip` happens
# to be installed. The mac filesystem is case-insensitive, so an invoker sitting in
# .../CEPHLA/ instead of .../Cephla/ otherwise resolves a different tree.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from squidmip._decon_qc import (            # noqa: E402  (path pin must come first)
    DEFAULT_CROP_HALF,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_VIEW_HALF,
    TISSUE,
    _lateral_sigma_px,
    brightest_structure,
    crop_around,
    halo_core_ratio,
    load_stack,
    qc_window_um,
    recommend,
    write_curve,
    write_montage,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", default=TISSUE)
    p.add_argument("--region", default=None, help="default: the first region")
    p.add_argument("--fov", type=int, default=0)
    p.add_argument("--channel", default=None, help="default: the first channel")
    p.add_argument("--iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
                   help="run RL for k = 1..N (default 8)")
    p.add_argument("--crop-half", type=int, default=DEFAULT_CROP_HALF,
                   help="half-width in px of the region RL is run on")
    p.add_argument("--view-half", type=int, default=DEFAULT_VIEW_HALF,
                   help="half-width in px of the montage panels (the RL crop is wider on "
                        "purpose; this is what gets looked at)")
    p.add_argument("--out", default=".", help="directory for the montage, curve and csv")
    p.add_argument("--no-gpu", action="store_true",
                   help="force the CPU backend (same RL update, different backend)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from squidmip._decon import DEFAULT_ITERATIONS, METHOD, OpticsParams, _run, make_psf

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stack, region, channel, meta = load_stack(
        args.dataset, args.region, args.fov, args.channel)
    optics = OpticsParams.from_acquisition(args.dataset, channel=channel)
    optics = OpticsParams(optics.na, optics.wavelength_um, optics.dxy_um, optics.dz_um,
                          int(stack.shape[0]), optics.ni)
    dxy_um, dz_um = optics.dxy_um, optics.dz_um

    # The Airy radius of THIS instrument: the smallest core the optics could form. It is
    # what separates "core" from "halo" below, and it comes from NA and wavelength, not
    # from a tuning constant.
    core_um = 0.61 * optics.wavelength_um / optics.na
    window_um = qc_window_um(core_um, stack.shape[0], dz_um)

    psf = make_psf(optics)
    print(f"dataset   : {args.dataset}")
    print(f"selection : region={region} fov={args.fov} channel={channel} "
          f"z={stack.shape[0]} planes, frame {stack.shape[1:]}")
    print(f"optics    : NA={optics.na} lambda_em={optics.wavelength_um} um "
          f"dxy={dxy_um} um dz={dz_um} um  (ni={optics.immersion_index})")
    print(f"psf       : vectorial, shape {psf.shape}, lateral sigma "
          f"{_lateral_sigma_px(psf):.3f} px  <- NA {optics.na}, not the old "
          f"hardcoded 1.5 px Gaussian")
    print(f"metric    : mean brightness of the halo ({core_um:.3f}..{window_um:.3f} um "
          f"shell) / mean brightness of the {core_um:.3f} um core")

    z_margin = int(np.ceil(window_um / dz_um))
    centre_full = brightest_structure(stack, dxy_um, dz_um, core_um,
                                      z_margin=z_margin, xy_margin=args.crop_half)
    crop, centre = crop_around(stack, centre_full, args.crop_half)
    print(f"structure : brightest at (z,y,x)={centre_full} in the frame; RL runs on a "
          f"{crop.shape} crop, structure at {centre}")
    print(f"engine    : petakit method={METHOD!r}, gpu={not args.no_gpu}\n")

    rows = [("raw", crop.astype(np.float32))]
    for k in range(1, args.iterations + 1):
        rows.append((f"{k}", _run(crop, psf, k, gpu=not args.no_gpu)))
        print(f"  ran RL k={k}", flush=True)

    values = [halo_core_ratio(v, centre, dxy_um, dz_um, core_um, window_um)
              for _, v in rows]
    raw_value, curve = values[0], values[1:]
    ks = list(range(1, args.iterations + 1))

    print("\niter  halo/core   delta vs previous")
    print(f" raw  {raw_value:.6f}")
    for k, value, prev in zip(ks, curve, [raw_value] + curve[:-1]):
        print(f"{k:>4}  {value:.6f}   {value - prev:+.6f}")

    best, kind, verdict = recommend(ks, curve)
    print()
    print(verdict)
    print(f"Measured on ONE FOV ({region}/{args.fov}/{channel}) - a recommendation for THIS "
          "sample at THIS exposure, never a global default; SNR and structure decide the "
          f"answer. The shipped default is DEFAULT_ITERATIONS={DEFAULT_ITERATIONS} and it "
          "stays until a human has looked at the montage and changed it deliberately.")

    montage = out / "decon_qc_montage.png"
    write_montage(montage, rows, centre, dxy_um, dz_um,
                  f"RL semi-convergence - {region}/{args.fov}/{channel} - turbo, "
                  f"per-row normalised", view_half=args.view_half)
    curve_png = out / "decon_qc_curve.png"
    write_curve(curve_png, ks, curve, best if kind == "turn" else None)
    csv_path = out / "decon_qc.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iterations", "halo_core_ratio"])
        w.writerow([0, f"{raw_value:.6f}"])
        for k, value in zip(ks, curve):
            w.writerow([k, f"{value:.6f}"])
    print(f"\nwrote {montage}\nwrote {curve_png}\nwrote {csv_path}")
    return 0


def _lateral_sigma_px(psf):
    """Second-moment-equivalent lateral sigma of the in-focus PSF plane, in pixels.

    Printed on every run so "the PSF really is the NA-0.3 one" is a measurement in the
    log rather than a claim in a docstring.
    """
    plane = np.asarray(psf[psf.shape[0] // 2], dtype=np.float64)
    total = plane.sum()
    if total <= 0:
        return float("nan")
    yy, xx = np.ogrid[:plane.shape[0], :plane.shape[1]]
    cy = float((plane * yy).sum() / total)
    cx = float((plane * xx).sum() / total)
    var = float((plane * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / total) / 2.0
    return float(np.sqrt(var))


if __name__ == "__main__":
    raise SystemExit(main())
