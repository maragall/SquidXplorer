# Decon demo SOP

Raw-vs-decon comparison media for the 25x_C4 acquisition: solve, still figures, seeded-orbit
video. One command:

```
python scripts/decon_demo_sop.py --iterations 2 --seed 4242 --scheme shared
```

## Steps

1. **Solve**: `decon_whole.py` deconvolves the whole acquisition (full z per solve, lateral
   tiling with per-channel halos, MPS cache released per window) into a sibling
   `decon46_iN_<acquisition>` set. Skipped when that set already exists.
2. **Figures**: `compare_figures.py` renders full-FOV, 561-only, real-GL stills at the QC'd
   poses (hero, XZ) for raw and each decon set, then composes eight raw | decon PNGs: four
   at the floor-95 scheme, four peak-matched.
3. **Video**: `record_volume_comparison.py` records the SAME `random_orbit_steps(seed)`
   list over raw and the decon set (sync is structural: one list, two passes) and composes
   them side by side into `~/Desktop/raw_vs_decon_561_seed<seed>.mp4`.

## Parameters

| flag | default | meaning |
|---|---|---|
| `acquisition` | the 25x_C4 set | pinned today; another input is a named refusal |
| `--channel` | Fluorescence_561_nm_Ex | pinned today (the shared window is 561's) |
| `--iterations` | 2 | decon iterations; names the output set `decon46_iN_...` |
| `--seed` | 4242 | orbit seed; one seed, one storyboard, baked into the video name |
| `--scheme` | matched | `matched` (default, Julio-approved on the stills): each side's ceiling is its OWN 561 z-MIP p99.999 so peaks render equal (the CEO's intensity matching - on this set raw (264, 3681), decon-i2 (272, 7123)); each side's floor is its OWN auto-contrast rule floor (`_contrast.auto_contrast`: mode + 2 sigma vs the background population's percentile), which drops raw's out-of-focus veil; gamma 0.7 shared. Absolute cross-side brightness is intentionally NOT comparable under this scheme - that is the spec. `shared` (135, 1109) and `floor` (95, 1109) keep one window on both sides at gamma 1. |
| `--out-dir` | ~/Desktop/decon_demo_figures | where the stills and figures land |

## Outputs

- `decon46_iN_<acquisition>/` beside the source acquisition.
- Eight figures in `--out-dir`: `compare_iN_{hero,xz}.png` (floor-95) and
  `compare_iN_{hero,xz}_pm.png` (peak-matched), plus the per-side stills.
- `~/Desktop/raw_vs_decon_561_seed<seed>.mp4`.

## Standing caveats

- **Biggs-Andrews at 3+ iterations**: the acceleration's lambda is global per solved
  volume, so tile interiors differ slightly from an un-runnable whole solve; within ~1
  count at 2 or fewer iterations. The solve log states it per run.
- **Tolerated teardown segfault**: each GL pass may exit -11 in Qt teardown AFTER its
  recording or stills are whole; the drivers verify the artifact and continue with a
  WARNING. Root cause unfound; the artifacts are unaffected.
