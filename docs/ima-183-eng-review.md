# IMA-183 — Engineering Review (longform)

**Ticket:** IMA-183 — One FOV per well for well-plate acquisitions
**Branch:** `juliomaragall/ima-183-one-fov-per-well-for-well-plate-acquisitions`
**Review:** `/plan-eng-review`, completed 2026-07-04. Outside voice: Claude subagent (Codex not installed).
**Status:** CLEARED for planning — but see §0. The reader assumptions were rebuilt after IMA-189 locked.

> This document is the reviewable record of the plan-eng-review. The working plan lives at
> `.spec/open/ima-183.md` (gitignored). Everything load-bearing is reproduced here so the
> merge diff is self-contained. The only behavior-bearing file this ticket has added to the
> tracked tree so far is `TODOS.md`; the implementation (T1–T6) lands after this doc is
> approved.

## Commits so far

| Commit | What | Tracked? |
|--------|------|----------|
| `b93b179` | Scaffold repo (shared base — README, gitignore, docs) | pre-existing |
| `479b54e` | `TODOS.md`: defer RGB/brightfield MIP read-path exactness | **yes, pushed** |
| _(pending)_ | This doc: `docs/ima-183-eng-review.md` | on approval |
| _(pending)_ | Implementation T1–T6 | after code review |

The enriched `.spec/open/ima-183.md` (15 KB) is **gitignored by project convention**
(`.gitignore:1: .spec/`), so it does not appear in the diff. It is mirrored in
`~/.gstack/projects/maragall-SquidMIP/`. This doc is the tracked, reviewable substitute.

---

## 0. Post-review reconciliation with the LOCKED IMA-189 (2026-07-05)

The review below was run when IMA-189 was still scoped as a *thin adapter over tilefusion*.
**IMA-189 has since been locked to a standalone reader** (`.spec/open/ima-189.md`,
`docs/ima-189-eng-review.md`). That changes the ground IMA-183 stands on. These corrections
**OVERRIDE** the corresponding statements later in this doc. This is the section to review
hardest.

**C1 — The reader API is different. Every IMA-183 signature was rewritten.**
My review assumed tilefusion's tile-indexed API: `reader.read_tile(tile_idx, z_level)` +
`tile_identifiers`/`unique_regions`. The locked IMA-189 reader is **key-indexed**:

```
open_reader(path) -> SquidReader
SquidReader.metadata -> { regions:[str] (natural-sorted),
                          fovs_per_region:{region:[int]},
                          channels:[{name,display_name,display_color,ex}],
                          n_z:int, z_levels:[int], dz_um, pixel_size_um,
                          wellplate_format, frame_shape:(Y,X), dtype, n_t }
SquidReader.read(region, fov, channel, z, t=0) -> (Y,X) native-dtype, exact
    # KeyError on unknown (region,fov,z,channel); IndexError on bad t;
    # raises on non-2D or dtype outside {uint8, uint16}
```
_(Verified against merged `squidmip/reader.py`. NOTE: `positions`/`coordinates.csv`
were **dropped** in the locked 189 — one FOV/well needs no per-FOV XY; plate layout
comes from the well ID + `wellplate_format`. Iterate `z_levels` (may be sparse), not
`range(n_z)`.)_

So `select_fovs` groups over `metadata.fovs_per_region` (not `tile_identifiers`), and
`project_well` loops `reader.read(region, fov, channel, z)` (not `read_tile`). Corrected
signatures in §4.

**C2 — "uint16" was wrong; preserve NATIVE dtype.** IMA-189's own §0 correction (verified
against `Cephla-Lab/Squid`): `4168×4168` is the *unbinned* crop; default binning is 2×2 →
~2084², and pixel format spans MONO8 (uint8) to MONO12/16 (uint16). The reader reads
`frame_shape`+`dtype` from the first real frame and preserves native dtype. **IMA-183 must
not hardcode uint16.** The MIP is `np.maximum` in native dtype; no cast. AC "bit-identical to
FIJI" stands, but the dtype is whatever the acquisition used.

**C3 — My D9 cross-model tension was moot (premise already corrected in IMA-189).** The whole
"read raw uint16 vs `read_tile` float32" debate rested on "float32 destroys exact uint16."
That is false: uint16 ⊂ float32 exactly (65535 < 2²⁴) and `_to_grayscale_2d` is a no-op for a
2D plane (`individual_tiffs.py:234`). The reversal I ran (D9) reached the right place — reuse
the reader — for a **wrong reason**. Under IMA-189 the point is simply: `reader.read()` already
returns exact native-dtype pixels, so IMA-183 does no dtype gymnastics at all. Cleaner than
either D9 option.

**C4 — Packaging is already done by IMA-189, and there is NO tilefusion dependency.** My
Issue 1 said "declare tilefusion in pyproject.toml." Locked IMA-189 is a standalone package
(deps: numpy, tifffile, pandas, pyyaml; **no tilefusion**) and adds `pyproject.toml` +
CI itself. IMA-183 adds nothing to packaging; it just imports `squidmip`. (tilefusion re-enters
only at IMA-184 for the OME-zarr writer.)

**C5 — coordinates.csv is not IMA-183's concern at all.** My Issue 5 (DRY: don't re-parse
coordinates.csv) is satisfied for free — IMA-189 owns all parsing (from *filenames*, the
version-robust ground truth; `fov` is the per-region enumeration index). IMA-183 touches only
`reader.metadata` and `reader.read()`.

**What survives unchanged:** the IMA-187 fold (§2), the pluggable-projector seam for IMA-188
(§2), streaming running-max (§4), fail-loud on missing planes (§4), the no-stitching guarantee,
and the fov-selection *semantics* (positional, §4) — these are all reader-agnostic.

---

## 1. Scope challenge (Step 0)

IMA-183 is genuinely small: given a reader, group FOVs by well, pick the first `n_fovs`, and
max each selected FOV's z-stack per channel. No new services, no complexity trigger. The real
risk was never size — it was **sequencing** and **hidden coupling**, which is where the review
spent its effort.

- **What already exists:** the SquidReader (IMA-189) does all ingest — well/FOV/channel/z
  discovery from filenames and exact per-plane reads. IMA-183 writes no parser.
- **Minimum change:** `select_fovs` + `project_well` + a `mip` reduce callable. ~2 small modules.
- **Not built here:** throughput (188), writer (184), CLI (186), viewer (185).

## 2. Build-order fold (user-supplied state machine)

IMA-183 is **slot #2** (189 → **183 (+187)** → 188 → 184 → 185 → 186 → 192). Consequences
folded into the plan:

- **Fold IMA-187 (FOV-count param).** The data model carries N FOVs/well from the start:
  `select_fovs(...) -> dict[well, list[fov]]`, parameter `n_fovs` (default 1). v1 uses length-1
  lists; IMA-187 multi-FOV needs no data-model change. Baking the list shape in now is the
  1-point constraint; retrofitting after 188/184 consume a `well→single` shape is a real
  refactor.
- **Pluggable projector for IMA-188.** The z-reduce is a named callable `mip(stack)->2D`, not
  `np.maximum` inlined into control flow. IMA-188 (slot #3) owns the projector *registry* (MIP
  now, EDF later) and swaps this callable — 183 must not make that swap a rewrite. 183 ships
  only `mip`.
- **No writer assumed.** IMA-184 (slot #4) does not exist yet; `project_well` returns an
  in-memory array.

## 3. Review findings (7 issues, all resolved)

| # | Area | Finding | Resolution |
|---|------|---------|------------|
| 1 | Arch | Depends on unbuilt reader + no packaging | Prereq stated. **Superseded by C4** — 189 owns packaging, no tilefusion. |
| 2 | Arch | MIP scope seam between 183/188 | 183 = FOV-select + single-well MIP; 188 = throughput. Kept. |
| 3 | Arch | Read path / pixel exactness | Raw uint16 read. **Reversed at D9, then mooted by C2/C3** — use `reader.read()`. |
| 4 | Arch | "correct FOV" undefined | Positional selection. Kept; sourced from `fovs_per_region` (C1). |
| 5 | Quality | DRY: don't re-parse coordinates.csv | Kept; **satisfied for free by C5** — 189 parses, 183 uses metadata. |
| 6 | Quality | Missing/corrupt z-plane | Fail loud, name well/channel/z. Kept; partly enforced by reader validation. |
| 7 | Perf | MIP memory | Streaming running-max, one plane in flight. Kept. |

## 4. Corrected component design (post-C1/C2)

```
select_fovs(metadata, n_fovs=1) -> dict[well, list[fov]]
  • group metadata.fovs_per_region by well (region)
  • take first n_fovs FOVs positionally (sorted); default 1 → one FOV/well
  • manual/no-region: region token is "manual" → one region, many fovs (see open Q1)
  • n_fovs > available for a well → clear error naming the well + its count

project_well(reader, well, fov, reduce=project) -> (T, C, 1, Y, X) native-dtype  # TCZYX, Z=1
  • for each t in range(n_t), each channel c in reader.metadata.channels:
      planes = (reader.read(well, fov, c.name, z, t) for z in reader.metadata.z_levels)
      acc = project(planes)                    # streaming running-max over z_levels
  • assemble → (T, C, 1, Y, X) TCZYX, Z kept size-1 (in-place reduction, not axis removal);
    matches Squid job_processing.py Zarr order → IMA-184 writer needs no special-casing
  • NO dtype cast (reader returns native uint8/uint16, exact)
  • iterate metadata.z_levels (may be sparse), NOT range(n_z)
  • missing/unreadable plane: reader.read raises located error; propagate loud
  • single z_level → project returns that plane unchanged
  • project is the pluggable primitive; 188 wraps + registers it

project(planes_iterable) -> 2D    # pure, dtype-preserving, bounded-memory reduction
  # running np.maximum fold; streams planes, never holds the whole z-stack.
  # the ONLY projector 183 ships (MIP); 188 adds EDF/etc. via its registry.
```

Data flow:

```
open_reader(path) ─► SquidReader (IMA-189)
        │ metadata.fovs_per_region, .channels, .n_z
        ▼
select_fovs(metadata, n_fovs=1) ─► {well: [fov, ...]}        (v1: one fov/well)
        │
        ▼  for each (well, fov):
project_well(reader, well, fov, reduce=mip)
        │  per channel: running-max over z via reader.read(well,fov,ch,z)
        ▼
(T, C, 1, Y, X) TCZYX native-dtype ─► [IMA-188 throughput / IMA-184 OME-zarr writer]
```

## 5. Test plan (greenfield; fixture = `~/CEPHLA/Data/sim_1536wp`)

Fixture: 1536 wells, fov=0 everywhere, Nz=20, 4 channels, uint16 4168×4168 (this sim is
unbinned uint16; real acquisitions may be binned uint8 — tests must not assume uint16, per C2).

- `select_fovs`: region grouping → 1536 length-1 lists; `n_fovs=2` on a multi-FOV fixture → 2
  FOVs/well; `n_fovs` over available → located error; manual token handling (open Q1).
- `project_well`: running-max equals `np.max(np.stack)` reference; **dtype == input dtype**
  (uint16 fixture → uint16; add a uint8 fixture); missing-plane → loud located error; `n_z==1`
  → single plane; C channels distinct.
- Integration: E2E over a well subset → one image/well (AC1); structural no-stitch test (AC3).

## 6. Failure modes

| Codepath | Failure | Test | Handling | Visible |
|----------|---------|------|----------|---------|
| read z loop | corrupt/missing plane | yes | reader raises + propagate | loud, located |
| select_fovs | `n_fovs` > available | yes | clear error | loud |
| manual branch | multi-position grab | yes | see open Q1 | — |
| streaming max | dtype/shape mismatch | assert | raise | loud |

No failure is silent AND untested AND unhandled → no critical gaps.

## 7. Implementation tasks

- [ ] **T1 (P1)** — verify `squidmip` imports (packaging from IMA-189); add nothing unless 189 didn't. [C4]
- [ ] **T2 (P1)** — `select_fovs(metadata, n_fovs=1) -> dict[well, list[fov]]`; positional; over-count error. [Issues 4,5; D10,D11; IMA-187 fold; C1,C5]
- [ ] **T3 (P1)** — `project_well` via `reader.read(...)`, streaming running-max, native dtype, `mip` isolated callable. [D9→C2,C3; Issue 7; IMA-188 seam]
- [ ] **T4 (P1)** — fail-loud on missing/unreadable plane, located. [Issue 6]
- [ ] **T5 (P2)** — structural no-stitching test. [AC3]
- [ ] **T6 (P2)** — E2E over sim_1536wp subset. [AC1]

## 8. Resolved decisions (from block-by-block feedback, 2026-07-05)

- **Manual/no-region → OUT OF SCOPE.** IMA-183 is a well-plate ticket; layout comes from well
  ID + `wellplate_format`. No special-casing; a `manual_...` region simply isn't handled here.
- **Native dtype, never upcast.** MIP preserves uint8/uint16 as the reader returns it; fail
  loud on anything else. So downstream (184) writes exactly what the camera produced.
- **Projection is `project(planes_iterable) -> plane` in IMA-183** (ships MIP only). IMA-188
  owns the pluggable projector *registry* + the parallel/streaming engine; 183 just keeps the
  primitive pure and bounded-memory so 188 can wrap and register it.
- **Dimensional model:** an FOV spans t, c, z. MIP reduces **z only**; t and c preserved.
  Per-FOV output = **(T, C, 1, Y, X)** TCZYX with **Z kept size-1** (Squid Zarr order, verified
  in `job_processing.py`) — in-place z-reduction, not axis removal.
- **IMA-187 fold** is the orthogonal FOV-count axis: `select_fovs(metadata, n_fovs=1) ->
  dict[well, list[fov]]`, driven off `metadata.fovs_per_region`; v1 = one FOV/well, the list
  shape carries up-to-4 future with no data-model change.

Intent asserted on the Notion "Squid MIP" page (IMA-183 section).

## 9. What shipped + test results

Public surface (added to `squidmip/__init__`): `select_fovs`, `project`, `project_well`.
- `squidmip/projection.py` — the three functions + module ASCII data-flow docstring.
- `tests/test_projection.py` — unit tests (project primitive, project_well, select_fovs;
  non-contiguous-z, multi-timepoint) + `tests/test_acquisition.py` dead-attribute guard.
- `tests/test_integration.py` — the shared cross-slot ("cross commit") file, one section per
  ticket. IMA-183 ↔ IMA-189 section covers, on real data:
  - **pixel-exact**: `project_well` == independent `np.max` over the reader's reads;
  - **efficacy**: MIP dominates every single z-slice AND (with >1 z) equals no single slice —
    proves it genuinely combines planes, not a pass-through;
  - **Nz-mismatch**: asserts the reader warns + trusts filenames on `sim_1536wp`;
  - **memory-bounded** single-well projection at 1536wp scale.

Results: **64 unit tests pass** (clean-room `pytest -m "not integration"`); **71 with
integration** on two datasets — synthetic `sim_1536wp` (1536 wells) and a real Squid
acquisition on disk (`real_dataset`, different shape/Nz). Two visual cross-checks saved to
`/tmp`: FIJI-equivalent bit-identical MIP (`|diff|=0`), and MIP-vs-single-slice showing the
projection recovers signal from other planes (68% of pixels brighter on the real z-stack).

**Single metadata format (JSON removed) — cross-slot decision carried in this branch.** Every
real acquisition we have (the real Squid dataset, sim_1536wp, current Squid) writes `acquisition.yaml`, so the legacy
flat `acquisition parameters.json` fallback had no real input — dead code with a permanent
two-format test burden. Removed it: `acquisition.yaml` is now the single required format
(`_acquisition.py` raises `FileNotFoundError` if absent — no silent recompute, no None-degrade).
This edits IMA-189's merged `_acquisition.py` + `test_acquisition.py` (a 189-contract change
carried in the 183 branch). Consequence for downstream: **IMA-184 can assume `pixel_size_um` /
`wellplate_format` are present** (no None-handling). IMA-183's projection reads no sidecar
scalars, so it is unaffected either way; the contract is just simpler now.

The `sim_1536wp` run also confirmed the design under stress: its recorded `Nz=3` disagreed with
the 20 z-planes on disk; the reader overrode it (filename-derived) and `project_well` iterated
`z_levels` correctly — the "filenames are ground truth" contract proving itself.

## 10. Performance baseline (single-thread) — the number IMA-188 must beat

Measured on `sim_1536wp`, cache-warm, steady state (`tests/test_performance.py`,
`benchmark_single_well`). Per well = 4 ch × 20 z = 80 planes, 2.78 GB:

| metric | value | note |
|--------|-------|------|
| `project_well` full | **~0.35 s/well** | read + streaming MIP |
| read (decode + I/O) | ~0.2 s | cache-warm; cold disk is storage-bound |
| MIP compute (`np.maximum`) | ~0.5 s | memory-bandwidth bound (~10 GB/s over 2.78 GB) |
| peak memory | **~278 MB** | (T,C,1,Y,X) result + ~2 in-flight planes; bounded, flat in Nz |
| **1536 wells, single-thread** | **~8–9 min (cache-warm)** | the baseline IMA-188 parallelizes |

Reading:
- **Algorithm is optimal.** Single-pass streaming `np.maximum`, O(planes), one pass,
  SIMD-vectorized. No faster MIP exists — you're bounded by touching the bytes.
- **Memory-bandwidth bound when warm** (compute > read); on COLD storage the read term
  dominates, bounded by real disk bandwidth — the epic's open "throughput vs real storage"
  question (needs Nick's storage to pin down). The ~8–9 min figure is best-case warm.
- **IMA-188 target:** parallel across wells → ~1.5 min @ 8 workers, ~45 s @ 16 (ideal I/O scaling).
- **Committed perf tests** (referenceable by 188): `tests/test_performance.py`
  `::test_single_well_speed_baseline` and `::test_single_well_memory_footprint`; 188 imports
  `benchmark_single_well` to compare its per-worker cost apples-to-apples.

## 11. IMA-188 handoff — preamble (short; prefix to the handoff)

> You are the **IMA-188 slot: throughput + pluggable projector.** You inherit a correct,
> memory-bounded, single-threaded per-well projector (IMA-183) on the standalone reader
> (IMA-189) — the projection is *done and optimal per well* (single-thread baseline
> **~0.35 s/well, ~8–9 min for 1536 warm**; `tests/test_performance.py`, §10). Your job is
> **throughput, not correctness**: run `project_well` across wells in parallel with bounded
> per-worker memory, and make the projector **pluggable** (MIP now via `project()`, EDF later)
> through the existing `project_well(..., reduce=)` seam — no 183 rewrite. **numba won't help**
> the memory-bandwidth-bound MIP; parallelize **across wells with a thread pool** (tifffile
> decode + `np.maximum` both release the GIL; a process pool would pay ~139 MB result pickling
> per well). You **own the 188↔183 cross commit** — append a `SECTION: IMA-188 ↔ IMA-183` block
> to `tests/test_integration.py`: parallel output pixel-identical to single-thread, beats the
> §10 baseline, per-worker memory bounded (via `benchmark_single_well`). Read the Notion
> "Squid MIP" page (state machine + "Cross commit" rule) first.

## 12. IMA-184 handoff (full — pasteable)

Assumes 188's output entry point (188 finalizes its exact public API; contract below is stable).

```
You are the IMA-184 slot in the SquidMIP build. SquidMIP is a high-throughput z-stack
maximum-intensity-projection tool for Squid well-plate acquisitions, built as a review-first
state machine — one git worktree per ticket, landed in dependency order. You are in the
ima-184 worktree, in your own independent context.

━━━ STEP 0 — READ THE SHARED STATE FIRST (before any code) ━━━
1. Notion "Squid MIP" (Blogs DB, id 3942dfbf-6ae4-81cf-bc41-fed456ddd398). Read the whole
   page: the state-machine model, the "Cross commit" rule, and the completed IMA-189 / 183 /
   188 sections. Your section goes under a new "IMA-184" header.
2. On main: docs/ima-189-eng-review.md, docs/ima-183-eng-review.md, docs/ima-188-eng-review.md,
   and the squidmip/ package. `git -C <this-worktree> merge origin/main` so you have the
   reader + projection + parallel engine.

━━━ WHAT IS ALREADY DONE (state you inherit — 189 + 183 + 188, on main) ━━━
    from squidmip import open_reader, select_fovs, project_well   # + 188's parallel engine
    - open_reader(path).metadata: regions, fovs_per_region, channels[{name, display_name,
      display_color, ex}], n_z, z_levels, dz_um, pixel_size_um, wellplate_format, frame_shape,
      dtype, n_t.  acquisition.yaml is REQUIRED (JSON removed) -> pixel_size_um and
      wellplate_format are GUARANTEED present (no None-handling).
    - Per (well, fov) the projection yields (T, C, 1, Y, X) native dtype in Squid's canonical
      Zarr order (TCZYX, Z=1) -> serialize WITHOUT transposition.
    - 188's parallel engine produces the projected plate (all wells), bounded memory.
    Confirm 188's exact public entry point from docs/ima-188-eng-review.md — 188 finalizes it.

━━━ YOUR TASK — IMA-184: output (OME-zarr canonical + per-well TIFF) ━━━
    - Write the projected plate as multiscale OME-zarr (canonical, navigable): axes TCZYX,
      pixel_size_um as the physical scale, channels[].display_color as omero rendering so it
      opens in ndviewer_light.
    - Per-well TIFF export (Nick's ask): one <well>.tif per well for external analysis software.
    - VENDOR the tilefusion OME-zarr writer (copy create_zarr_store / write_ome / colors.py into
      squidmip; do NOT import tilefusion — heavy __init__. See the IMA-184 writer notes in the
      project memory).
    - Standalone deps only (add zarr / tensorstore etc. to pyproject as needed).
    - Streaming / bounded memory: write each well as it is projected; never hold the whole plate.
    - Out of scope: parallel projection (188), UI/montage (185), CLI (186).

━━━ CROSS COMMIT (MANDATORY — you own 184 ↔ 188/183) ━━━
    Append a "SECTION: IMA-184 ↔ 188/183" block to tests/test_integration.py. No mocks, on
    /Users/julioamaragall/CEPHLA/Data/sim_1536wp: write the output, read the OME-zarr back, and
    assert it equals the in-memory projected plate (pixel-exact, dtype, TCZYX axes); assert a
    per-well TIFF round-trips; confirm it opens in ndviewer_light. @pytest.mark.integration,
    green before merge. A slot isn't done until its cross commit is green.

━━━ CLEAN-CODING CONVENTIONS (every slot — read as one hand) ━━━
    - Thin public surface in squidmip/__init__; logic in private modules; ASCII data-flow
      docstring on the main module. No cross-repo imports (vendor, don't import).
    - Preserve native dtype; fail LOUD; no dead/unused attributes. Bounded memory; lazy.
    - Tests: unit (mocked seam) AND the cross commit above; CI clean-room
      `pip install .[test]` + `pytest -m "not integration"`.
    - Encode intent a priori (design + tests before behavior); human reviews each block.

━━━ HOW YOU CLOSE OUT (handoff for the next slot = IMA-185) ━━━
    - Write docs/ima-184-eng-review.md (mirror 189/183/188).
    - Populate an "IMA-184" section on the Notion page: what you built, the human review points
      (in the user's voice), the output contract IMA-185 consumes, the Testing line. Add a
      "Placeholder for IMA-185" header.
    - VERIFICATION FIGURES (184 is visual — do this): render proof figures (e.g. an
      ndviewer_light screenshot of the written plate; an OME-zarr read-back-vs-in-memory diff
      panel). The Notion MCP CANNOT embed local PNGs, so save them to ~/Downloads and add
      labeled "[ drag PNG here ]" placeholders in your Notion section for the human to drag in
      (same pattern as IMA-183's verification figures).
    - After block-by-block review + test, merge --no-ff to main and push. Produce the IMA-185
      handoff prompt (same structure).

STOP after Step 0 and confirm your understanding of the inherited API + your task before
writing code. Encode intent a priori.
```

---

**Verdict:** IMA-183 CODE LOCKED — 189+183 cross-slot green on real data; single-thread perf
baseline recorded (§10). Ready for block-by-block review → merge. The reader-facing design in
§4 supersedes the tilefusion-based §3 (§0).
