# IMA-211 — Engineering Review (longform)

**Ticket:** IMA-211 — Stitcher operators: ASHLAR / MCmicro / BigStitcher / PetaKit5D
**Branch:** `juliomaragall/ima-211-stitcher-operators`
**Review:** `/plan-eng-review` + outside-voice challenge, completed 2026-07-20.
**Outcome:** **Scope collapsed, not merely reduced.** Most of IMA-211 is duplicated by three
tickets that landed on origin during this review. What survives is the acceptance oracle.

> Working plan: `.spec/open/ima-211.md`. This doc is the tracked narrative — including the part
> where this review's own first pass was wrong.

---

## 1. The headline: IMA-211 was overtaken mid-review

| Ticket | Branch on origin | Owns | Overlaps IMA-211's |
|---|---|---|---|
| **IMA-222** | `…/ima-222-op-stitch` | The Stitch operator | the entire stitch operator |
| **IMA-226** | `…/ima-226-live-any-operator` | IMA-210's registry half (`consumes` on `add_projector`) **and surfacing `reference`** | the axis seam and the registry fix |
| **IMA-187** | `…/ima-187-multi-fov-mosaic` | Multi-FOV mosaic, `fov_positions`, `select_fovs(None)`=all | FOV geometry and selection |

The first version of this review planned a registry refactor, an axis seam, FOV geometry
plumbing and a stitch operator. Every one of those is now someone else's ticket, already
reviewed and pushed. Building them here would be duplicated work and a merge conflict.

**IMA-211 re-scopes to the one thing nobody owns: evaluation.**

## 2. Where this review's first pass was wrong

The outside voice found four factual errors. All were then verified by hand against
`~/Downloads/20x_scan_2025-09-05_17-57-50`. Recording them because the surviving reasoning
depends on knowing which parts did not survive.

**Wrong coordinate file.** The first pass proposed joining `coordinates.csv` to filename-derived
FOV indices — which has no join key, and *is* the row-order fabrication that
`docs/ima-189-eng-review.md` already refused as a "silent wrong tile" hazard. Squid writes **two**
coordinate files, and the right one was never found: `original_coordinates/original_coordinates_{t}.csv`
= `region,fov,z_level,x (mm),y (mm),z (um),time`, with an explicit FOV key, recording the *actual*
stage position where `coordinates.csv` records the *planned* grid.

**`acquisition parameters.json` is not a geometry cross-check.** It reports `Nx=1, Ny=1, dx=0.9`
for a region that actually holds **36 FOVs in a 6×6 grid at 0.7056 mm**. The proposed
"consistent Nx×Ny lattice at expected spacing" gate would have refused every real dataset.

**scipy is not a dependency.** `pyproject.toml:13-22` has no scipy. The first pass argued for a
lean dependency set and proposed "phase cross-correlation via scipy FFT" in the same breath.

**"4 FOVs per well" was the smallest case.** Real data is 16 FOV/well (hongquan) and 36 (20x
scan) — which inverts the memory sizing and weakens the "tilefusion's optimizer is oversized"
argument that had justified reimplementing it.

The root cause is worth keeping: the first pass searched for real acquisitions, did not find any,
and designed an on-disk data contract from a docstring plus a prior learning. **Not finding the
data should have blocked designing the parser, not licensed inferring it.**

## 3. What survives

### The acceptance oracle

The ticket said *prototype/evaluate* four stitchers. The first pass answered "none is shippable
inside our Windows app" — true, but a different question. Evaluation does not require shipping,
and **IMA-222 has no acceptance criterion for stitch quality**. Build your own stitcher, never run
ASHLAR once, and you have no way to know yours is good.

1. **Primary, runs today:** cut one real 2084² plane into a grid at the true overlap with injected
   ±20 px offsets; require recovered offsets within 1 px and the composite bit-exact outside
   feather zones.
2. **Seam metric:** mean absolute difference across the seam ≤ 1.5× the same statistic inside a tile.
3. **Fallback accounting:** per-well `registered` vs `nominal-fallback` in the manifest; above a
   threshold, a loud warning, never a silent pass.
4. **External cross-check:** ASHLAR out-of-process for 3-5 wells; max displacement disagreement ≤ 2 px.
5. **Negative test:** blank / uniform-noise overlap must fall back to nominal, never shift.

Only ASHLAR is worth running. **MCmicro is not a stitcher** — it is a Nextflow orchestrator that
shells out to ASHLAR, so evaluating it measures ASHLAR twice; strike it from the ticket title.
BigStitcher needs a JVM and a mandatory BigDataViewer resave and has no plate/well concept.
PyPetaKit5D needs the MATLAB Compiler Runtime, is Linux-only, and assumes 3D light-sheet
throughout. ASHLAR itself boots a JVM via pyjnius **at import time**, which is why it is a dev-box
oracle and never a shipped dependency.

### A correction that contradicts IMA-187's lock ⚠ (time-sensitive)

IMA-187 locked its FOV keying as *"row-order-per-region with a loud cross-check"* from
`coordinates.csv`. That works from the incomplete picture above. With
`original_coordinates_{t}.csv` the mapping becomes arithmetic on an explicit key, and IMA-187's
own recorded worry — *"whether coordinates.csv carries one row per z-level on multi-z
acquisitions, which would break the row-count cross-check"* — dissolves, because
`original_coordinates` carries `z_level` explicitly and can be filtered rather than guessed.

This only helps if raised **before IMA-187 merges**.

### Scale facts that corroborate IMA-187

```
frame                2084 × 2084 uint16        =   8.7 MB
region C5            36 FOVs, 6×6, 0.7056 mm   ≈   9% overlap
composite side       5 × 1892 + 2084           ≈ 11,544 px
per well (T=1, C=4)                            ≈ 1.07 GB
```

This independently reproduces IMA-187's locked "11541×11541 px = 1.07 GB at 4ch uint16" from a
different dataset and a different direction — good corroboration for its **no-pixel-composition**
decision. It also means that if IMA-222 ever does compose pixels, `_engine.py:187-193`'s
*"peak RSS ≈ workers × one well's footprint, flat in plate size"* degrades ~40× while `workers`
still defaults to CPU count. A workers cap for any fov-consuming operator is not an optimization.

### The blockers are not what the ticket says

The ticket says BLOCKED on the laser-AF fix. More precisely: **the overlap fraction must be
measured before anyone writes registration code** — it is ~9% here, but if Nick's acquisition runs
0%, correlation is dead on arrival and only nominal placement works. And **focus quality is what
the laser-AF block actually protects**: out-of-focus tiles wreck phase correlation, so a
stitch-quality run against a broken-AF acquisition measures autofocus, not stitching.

Neither blocks the synthetic harness, which runs today.

## 4. What was built, and what it measured

Two things landed on this branch. Both run today; neither waits on the laser-AF fix.

**`squidmip/_oracle.py` + 26 tests — the acceptance gate.** It grades a stitcher without
containing one: cut a known image into overlapping tiles at known positions, hand the tiles to a
stitcher, check whether it puts them back. Deliberately no registration inside it — a grader that
reimplements the thing it grades is circular. numpy only, so it costs the shipped package nothing.
The load-bearing tests are the negative ones: a stitcher that ignores the pixels and returns the
nominal grid must FAIL, and the seam metric must spike when a tile is misplaced. A gate that
passes everything is worse than no gate.

Two bugs surfaced while building it, both worth recording. Random per-tile offsets leave parts of
the canvas covered by no tile, so "bit-exact" and the seam metric were both scoring zero-padding;
every metric now masks by actual coverage. And perturbing tiles by more than half the overlap
**tears** the mosaic — a real property of stitching, not a quirk of the harness — so the fixture
refuses to build one rather than silently under-testing.

**`scripts/measure_overlap.py` + 14 tests — the survey.** Run over all 5 acquisitions on disk:

| Acquisition | FOV/well | Grid | Overlap | `original_coordinates/` |
|---|---|---|---|---|
| `20x_scan_2025-09-05_17-57-50` | 36 | 6×6 | **9.2%** | yes |
| `synthetic_2x2_wellplate` | 36 ×4 wells | 6×6 | **9.2%** | no |
| `test_10x_laser_af_z_stack…yy` | 27 / 28 | 6×5 partial | n/a | yes |
| `sim_1536wp` (×2) | 1 | 1×1 | n/a | no |

Six results that change decisions:

1. **Overlap is ~9.2% and consistent, so registration is viable.** The 0%-overlap case that would
   have killed correlation outright does not occur anywhere on disk.
2. **`original_coordinates/` exists in only 2 of 5 acquisitions — a fallback is mandatory**, not a
   nice-to-have. Its necessity is now settled; its design is not.
3. **`manual0`/`manual1` regions are real**, so the `parse_well_id` blocker is a live defect.
4. **Non-rectangular scans are real** — 27 and 28 FOVs that do not fill their 6×5 grid. A strict
   lattice gate would have refused valid data, exactly as the outside voice predicted.
5. **Multi-z coordinate files are real** (10 z_levels). IMA-187's recorded worry is genuine, and
   `original_coordinates`' explicit `z_level` column resolves it by filtering rather than guessing
   — which strengthens the case for the correction in §3.
6. **A 20× units trap.** `acquisition parameters.json` stores the raw *sensor* pitch plus
   `objective.magnification`; `acquisition.yaml` stores object-space directly. Reading the JSON
   value as object-space reports **95% overlap on a 9% scan**. Caught during this work and pinned
   with a regression test.

## 5. Recommendation

Re-scope IMA-211 to the oracle; close the rest as duplicated. Highest value first: the synthetic
cut-and-restitch harness (runs today, becomes IMA-222's acceptance gate), then measuring the real
overlap fraction, then raising the geometry correction against IMA-187 before it merges.

## 6. Open questions for merge review

1. Does IMA-187 accept the geometry correction? If yes its FOV-keying task shrinks and its multi-z
   worry closes; if no, record why `original_coordinates` was rejected. The survey found a real
   10-z_level dataset, so this is not theoretical.
1b. What is the fallback when `original_coordinates/` is absent (3 of 5 acquisitions)? Row-order
   inference with a loud cross-check, or refuse outright? Necessity is settled; design is not.
2. IMA-222's B1 — naive placement vs. re-opening the no-tilefusion rule vs. shipping mislabeled —
   remains open. The scale and overlap findings are inputs to it; IMA-211 does not decide it.
3. Does IMA-211 stay a separate ticket once re-scoped to the oracle, or fold into IMA-222?
   Recommendation: keep it separate — an acceptance oracle owned by the ticket it grades is a
   weaker gate.
