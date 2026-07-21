# IMA-218 — Engineering Review (longform)

**Ticket:** IMA-218 — B1 · PlateOverview mosaic (render integration), child of the IMA-187 B1 keystone
**Branch:** `juliomaragall/ima-218-plateoverview-mosaic`
**Review:** `/plan-eng-review`, completed 2026-07-20. Outside voice: Claude subagent (Codex not installed).
**Status:** NOT CLEARED — 6 decisions locked, 2 corrected after code verification, 6 unresolved, 2 critical gaps.

> This document is the reviewable record of the plan-eng-review. The working plan lives at
> `.spec/open/ima-218.md` (gitignored). Everything load-bearing is reproduced here so the
> diff on merge is self-contained. No implementation lands in this commit.

---

## 0. Why this is NOT cleared

Three blocking facts, established before any review section ran:

1. **Every upstream dependency is empty.** IMA-214, 215, 216, 217 are all `Backlog` in Linear,
   and every branch (`ima-214` … `ima-218`) sits at `1504c05` — identical to `main`, zero
   commits. IMA-218 cannot start today.
2. **Two locked decisions were not implementable as written** and required correction after
   reading the code (§3). A plan that passes review on its own prose but contradicts
   `_output.py` and `_engine.py` is worse than no plan.
3. **One strategic challenge to the core premise is unresolved** (G7, §6) and would change
   which tasks are needed at all.

---

## 1. Step 0 — scope challenge

**Spec-vs-reality contradictions found before reviewing anything:**

- **The spec omits its real blocker.** It lists IMA-216/217 as dependencies, but
  `reader.py:26-29` states plainly: *"coordinates.csv is not read … per-FOV stage positions
  are a deferred stitching concern."* You cannot place FOVs at stage coordinates without
  IMA-215 (coords-reader). That is the hard dependency and it was not listed.
- **`compose()` does not exist.** IMA-187 describes it as "currently a NotImplementedError
  stub"; grep finds no `compose` symbol anywhere in `squidmip/`. No B-slot ticket claims it.
- **The oracle contradicted the parent's.** IMA-187 requires *"N=1 result byte-identical to
  current single-FOV MIP"*; this spec weakened it to "N=1 unchanged", which is not a
  regression guard.
- **Blast radius undercounted.** The spec names one file. The tile contract has four
  producers, all assuming an exactly-88×88 tile: `_OperatorWorker._on_well:816`,
  `_PreviewWorker.run:920`, `_ComputedPlateWorker.run:977`, `_final_montage:884`.

**Prior learning applied:** `fov-axis-needs-geometry` (8/10, 2026-07-04) — per-FOV geometry in
metadata, not FOV count, is the load-bearing piece. Confirmed exactly right.

---

## 2. Decisions locked (D1–D6)

| # | Decision | Rejected |
|---|---|---|
| D1 | Coordinate placement into the existing canvas now; **viewport-tiler swap becomes its own ticket** | Blocking on 216/217 (zero commits); patching the canvas with nobody owning LOD |
| D2 | Coordinates come from **IMA-215**, added as a hard dependency | Inline parser (duplicates 215); synthetic grid (validates against fake geometry) |
| D3 | Workers compose a cell-sized mosaic; `add_tile` contract unchanged | Widget-side placement (touches all four producers, collides with the tiler ticket) |
| D4 | FOV count derived **per region** from `metadata['fovs_per_region']` | Global `n_fovs` setting (drops FOVs on ragged plates); leaving `n_fovs=1` |
| D5 | A FOV with no usable coordinate **fails loud per well** (red-x, run continues) | Silent skip (the silent partial-well bug); unlabelled synthetic fallback |
| D6 | Invalidated docs corrected in the same commit | Deferring to a cleanup pass |

**D1 rationale.** `_viewer.py:533` allocates `QImage(nc*88, nr*88)`; `add_tile:586` stamps an
exactly-88×88 thumbnail; `paintEvent:698` rescales that baked image. This is structurally
incapable of level-of-detail, so "depends on the LOD tiler **and** patch two methods" was
self-contradictory. At 88px/cell a 6×6 mosaic gives each FOV ~14px.

**D6 caught a bug that is stale today.** The module docstring (`_viewer.py:33-37`) claims the
run retains *"one downsampled 88x88xC **float32** tile PER WELL … ~190 MB for a 1536wp"*, but
`:822` allocates `np.empty((len(tiles), _CELL, _CELL), self._dtype)` — uint16. The documented
memory budget is roughly double the truth, before this ticket touches anything.

---

## 3. Post-review corrections (outside voice, verified against the code)

Four outside-voice claims were verified directly and folded in. Two reverse the *mechanism*
of a locked decision while leaving its intent intact.

**D3 CORRECTION — `_on_well` cannot compose a mosaic as described.**
`_output.py:332` calls `on_well(region, fov, image)` from inside `_write_one`, which is
submitted **once per FOV** to a writer `ThreadPoolExecutor`. There is no well-complete event
and no ordering guarantee. Composition therefore needs new `_OperatorWorker` state: a
per-region FOV accumulator, an arrival counter, and an explicit emit-when-complete rule, all
under the existing `self._lock` (`:808`). **Partial mosaics must never be emitted.**

**D5 CORRECTION — this is not a reuse of `_on_error`.**
`on_error` is invoked only from `_engine.py:238`, on a failed *projection* future. A
coordinate problem surfaces in `_on_well`, on a **writer** thread; raising there propagates
through `f.result()` and aborts the **entire run** — the opposite of the intent. A new
per-well error channel is required. Prefer checking coordinates *before* the write pass.

**Oracle #1 is unrunnable through the app.**
`_viewer.py:1516` rejects any format not in `_SUPPORTED_PLATES = ("384","1536")` with
*"only 384- and 1536-well plates are supported right now"*. A 2×2 plate never reaches the
renderer. The oracle is restated as a `PlateOverview` unit test; widening the scope guard is
a separate, undeclared change.

**The other producers cannot stay 1-FOV.**
`_PreviewWorker:927` reads `fovs_per_region[region][0]`; `_ComputedPlateWorker` is handed a
single `fov0` (`:1692`). Leaving them means: open → 1-FOV thumbnails, run MIP → 36-FOV
mosaics, reopen → back to 1-FOV. That reads as the view randomly changing zoom.

---

## 4. Test review

```
CODE PATHS                                             USER FLOWS
[+] reader.py — coords parse (IMA-215)                 [+] Open an acquisition with N>1 FOVs
  ├── [GAP] Monkey format region,fov,z_level,x,y,z       ├── [GAP] Plate shows a mosaic per well
  ├── [GAP] 20x format region,x,y,z (NO fov col)         ├── [GAP] [→E2E] MIP run writes N-FOV plate
  ├── [GAP] row-count mismatch handling                  └── [GAP] Reopen computed plate, mosaic intact
  └── [GAP] coordinates.csv absent entirely
[+] worker mosaic composition                          [+] Degraded acquisitions
  ├── [GAP] CRITICAL N=1 byte-identical (regression)     ├── [GAP] No coordinates.csv → red-x wells
  ├── [GAP] N=2 correct relative offsets                 └── [GAP] Partly-annotated plate → mixed
  ├── [GAP] 36 FOVs → 6x6, 144 placements
  ├── [GAP] overlapping FOVs (draw order defined)
  ├── [GAP] no partial mosaic ever emitted
  └── [GAP] ragged plate: wells with differing N
[+] _viewer.py:853 n_fovs derivation
  ├── [GAP] per-region N from fovs_per_region
  └── [★★ TESTED] single-FOV path — test_viewer.py:193
[+] Existing regressions to keep green
  ├── [★★ TESTED] tiles + hue status — :193 (asserts _raw==2, "one 88px tile per well")
  ├── [★★ TESTED] double-click raw z-stack — :211
  ├── [★★ TESTED] fov slider red box — :232
  └── [★★ TESTED] second ingest resets — :244

COVERAGE: 5/21 paths tested (24%)  |  Code paths: 5/14  |  User flows: 0/7
QUALITY: ★★★:0 ★★:5 ★:0  |  GAPS: 16 (1 E2E, 0 eval, 1 CRITICAL regression)
```

**Two fixture blockers.**
1. `tests/conftest.py:4-5` documents *"a legacy-schema coordinates.csv"*, but the
   `squid_dataset` fixture body (`:81-89`) never writes one. Doc-vs-code mismatch today.
2. `squid_dataset` is 2 regions × 2 FOV × 2 z × 2 ch. Oracle #1 needs 4 wells × 36 FOV — a
   new fixture (~576 TIFFs), not a CSV bolted onto the existing one.

**REGRESSION RULE (mandatory).** `test_viewer.py:193` asserts `len(win._worker._raw) == 2`
with the comment *"the worker keeps one 88px tile per well"*. D3/D4 change what `_raw` holds.
That test must be updated **and** a byte-identical N=1 guard added, or the parent's core
oracle is unverified.

---

## 5. Failure modes

| Failure | Test? | Handled? | User sees |
|---|---|---|---|
| coordinates.csv absent | GAP | D5 → red-x | Red wells; needs a status line or it reads as a tool bug |
| CSV row order ≠ FOV identity | GAP | **none** | **CRITICAL GAP** — images placed at wrong physical positions, silently |
| Coordinate math wrong (origin/Y-sign) | GAP | **none** | **CRITICAL GAP** — plausible-looking but mirrored/offset mosaic |
| FOV coord outside region bbox | GAP | none | Mosaic clipped or out-of-range canvas write |
| Ragged plate | GAP | D4 handles | Correct per-well mosaic |
| 1536wp × 36 FOV memory | GAP | none | See §7 |

**Critical gaps: 2.** Both are silent-wrong-data failures — the worst class for a scientific
tool, because nothing looks broken.

---

## 6. Open gaps (G1–G8) — no decision recorded

- **G1 — the coordinate math is entirely unspecified**, and it is the only genuinely hard part
  of the ticket: origin (per-region min or plate origin?), Y sign (stage up vs image rows
  down), mm→px via `pixel_size_um` and `frame_shape`, and out-of-bbox clipping.
- **G2 — overlap is immediate, not deferred.** Squid scans carry ~10% FOV overlap, so
  placement means FOVs overwrite. Deferring *blending* to IMA-222 is fine; deferring a
  documented **draw order** is not.
- **G3 — progress and detail-slider counting break at `n_fovs>1`.** `_total` is
  `len(regions)` (`:805`) but `_done` increments per `_on_well` (`:831`) → 144/4. And
  `pushReady.emit(info["idx"], …)` (`:839`) keys per-region, so 36 FOVs stomp one slot while
  paying 36 downsamples for 35 discarded results.
- **G4 — double-click becomes actively wrong.** `mouseDoubleClickEvent:682` emits FOV 0;
  on a mosaic the user clicks a visible FOV and silently gets a different one.
- **G5 — the `_CELL` cap counts one buffer of four.** `_raw` (~380 MB at `_CELL=176`),
  `_canvas` (~143 MB), one `_op_canvas` **per layer**, and the `_scaled` cache also scale.
- **G6 — T2 was mis-specified**: with no `fov` column there is no token to compare, only a
  per-region row count; failing the open on a mismatch contradicts D5.
- **G7 — STRATEGIC, the one real challenge to a locked decision.** At the capped `_CELL`,
  each FOV is ~15-29px, so sub-FOV coordinate accuracy is **invisible**. A fixed-pitch
  arrangement sorted from the coordinates would be pixel-indistinguishable for any
  raster/serpentine scan, needing no mm→px (G1), no origin/Y-sign (G1), no overlap policy
  (G2), and degrading gracefully when coords are missing (D5). Exact placement arguably only
  pays off once the tiler lets you zoom — the ticket D1 defers — so the sequencing may be
  inverted. **Not adopted**: it contradicts locked D1/D2 and IMA-187's own oracle. Julio's call.
- **G8 — IMA-217 is a phantom dependency.** No task consumes a pyramid source, and D1 renders
  into the baked canvas. Drop it or name the consumer.

---

## 7. Performance

`_final_montage:884` allocates `np.zeros((nr*_CELL, nc*_CELL, 3), np.float32)` and already
notes a *"~430 MB transient on a 1536wp"* at `_CELL=88`. Raising `_CELL` to fit a 6×6 mosaic
scales this **quadratically**: `_CELL=176` ≈ 4× (~1.7 GB), `_CELL=352` ≈ 16×. Per G5 this is
one of four buffers that scale together. A plate that would exceed the cap is a
**tiler-ticket case, not a bigger-canvas case** — which is the concrete argument for D1.

---

## 8. What already exists (reused, not rebuilt)

- `select_fovs` (`projection.py:166`) — already returns a list per well and already raises on
  a short slice rather than silently truncating. Only the `n_fovs=1` caller at
  `_viewer.py:853` changes.
- `wellFailed` / red-x rendering — the *display* side of D5 is reused; the *plumbing* is not (§3).
- `_area_downsample`, `_window`, `_hex_to_rgb01` (`_montage.py:122,141,149`) — compositing
  math, reused unchanged. Only placement is new.
- `well_at` (`_viewer.py:426`) — pure, unit-tested cell hit-testing, unchanged.
- `_fit_cell` (`:441`) — already guards the tiny-frame upscale case.

---

## 9. NOT in scope

- **Viewport tiler / true LOD zoom** — its own ticket per D1; the baked canvas cannot express LOD.
- **Stitching / blending overlapping FOVs** — placement only; blending is IMA-222 (op-stitch).
- **`compose()`** — referenced by IMA-187, exists nowhere; flagged, not built here.
- **Widening `_SUPPORTED_PLATES`** — oracle #1 becomes a unit test instead.
- **Distribution** — no new artifact; ships inside the existing `squidmip-view` entry point.

---

## 10. Worktree parallelization

| Step | Modules touched | Depends on |
|------|----------------|------------|
| S1 coords parse (IMA-215) | `squidmip/reader.py`, `tests/` | — |
| S2 fixtures (coords + 4×36) | `tests/` | — |
| S3 mosaic accumulator + placement | `squidmip/_viewer.py`, `tests/` | S1, S2 |
| S4 n_fovs derivation + counting | `squidmip/_viewer.py` | S1 |
| S5 doc/comment corrections | `squidmip/_viewer.py`, `squidmip/reader.py` | S3, S4 |

`Lane A: S1 → S3 → S5 (sequential, shared reader/viewer)` · `Lane B: S2 (independent, tests/ only)`

Launch A and B in parallel. **Conflict flag:** S3 and S4 both land in `_viewer.py` — sequence them.

---

## 11. Implementation tasks

P1 blocks ship. Full task records (with verify commands) are in the working plan and in
`~/.gstack/projects/maragall-SquidMIP/tasks-eng-review-*.jsonl`.

| ID | P | Component | Task |
|----|---|-----------|------|
| T1 | P1 | reader | Build IMA-215 coords parser, both CSV formats |
| T2 | P1 | reader | Define FOV identity when the CSV has no `fov` column (G6) |
| T3 | P1 | tests | Add a real `coordinates.csv` to `squid_dataset` |
| T3b | P1 | tests | New 4-well × 36-FOV fixture for the 144-placement oracle |
| T4 | P1 | viewer | Per-region FOV accumulator + emit-on-complete (D3 CORRECTION) |
| T4b | P1 | viewer | Resolve coordinate math: origin, Y sign, mm→px, clipping (G1) |
| T4c | P2 | viewer | Document and implement overlap draw order (G2) |
| T5 | P1 | viewer | **CRITICAL** N=1 byte-identical regression guard |
| T6 | P1 | viewer | Derive `n_fovs` per region at `_viewer.py:853` |
| T6b | P1 | viewer | Fix progress + `pushReady` counting for `n_fovs>1` (G3) |
| T7 | P1 | viewer | New per-well error channel for unusable coords (D5 CORRECTION) |
| T7b | P2 | viewer | Decide preview/computed-plate FOV consistency (§3) |
| T8 | P2 | viewer | Cap `_CELL` against **all four** buffers (G5) |
| T8b | P2 | viewer | Handle per-FOV double-click intent (G4) |
| T9 | P2 | docs | Correct the stale float32 memory claim + IMA-187 comments (D6) |
| T10 | P3 | spec | Linear dep line: add IMA-215, resolve IMA-217 (G8) |

---

## 12. Unresolved decisions that may bite you later

1. **G7 — coordinate placement vs fixed-pitch arrangement.** Settle before T4; it changes
   whether T4b, T4c, and much of T1 are needed at all.
2. **G1 — coordinate math**: origin, Y sign, mm→px, clipping. No decision recorded anywhere.
3. **G2 — overlap draw order** (first-wins vs last-wins). Determines day-one seam behavior.
4. **G8 — IMA-217**: no task consumes it. Drop it or name the consumer.
5. **Oracle #1 delivery**: confirm the unit-test reading rather than widening `_SUPPORTED_PLATES`.
6. **T7b**: must preview and computed-plate views match the mosaic, or is the inconsistency
   accepted and documented?

---

## 13. Review metadata

- Step 0 scope: **challenged** — 4 spec-vs-reality contradictions found; IMA-215 added as a hard dep.
- Architecture: 4 decisions locked. Code Quality: 2 (error policy, stale docs).
- Test review: coverage diagram produced, 16 gaps, 1 critical regression, 2 fixture blockers.
- Performance: 1 issue (quadratic `_CELL` scaling across 4 buffers).
- Outside voice: ran (Claude subagent). 14 findings — 4 verified against code and folded in as
  corrections, 2 reversed the mechanism of a locked decision, 1 (G7) left to the user.
- Failure modes: 6 mapped, **2 critical gaps**.
- Verdict: **NOT CLEARED** — dependencies unbuilt, 6 unresolved decisions.
