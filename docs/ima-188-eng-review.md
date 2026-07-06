# IMA-188 — Engineering Review (longform)

**Ticket:** IMA-188 — High-throughput (1536wp) z-stack max projection
**Branch:** `juliomaragall/ima-188-high-throughput-1536wp-z-stack-max-projection`
**Review:** `/plan-eng-review`, then block-by-block code review, completed 2026-07-05.
**Status:** CODE LOCKED — merged `--no-ff` to main; 188↔183 cross commit green on both datasets.

> The working plan lives at `.spec/open/ima-188.md` (gitignored). This doc is the tracked,
> reviewable record. IMA-188 adds `squidmip/_engine.py`, three public names, and one
> cross-commit section — the whole diff is self-contained.

---

## 1. Intent (asserted a-priori, before code)

IMA-183 made per-well projection **correct and optimal** (single-thread ~0.42 s/well, bounded
memory via a streaming running-max). IMA-188 makes it **fast across the whole plate without
changing a single pixel**. Two walls the naive plate loop hits, and what IMA-188 does about them:

- **Time:** 1536 × 0.42 s ≈ **11 min** single-threaded → run wells concurrently.
- **Memory:** each projected well is ~139 MB; holding the plate = **213 GB** → stream, never materialise.

IMA-188 is **throughput, not correctness**. It does not touch the MIP math. It runs `project_well`
across wells, streams the results, and makes the z-reduction pluggable.

## 2. Inherited seam (verified against merged `squidmip/`)

```
project(planes: Iterable) -> ndarray        # bounded-memory running-max; the projector primitive
project_well(reader, region, fov, reduce=project) -> (T,C,1,Y,X) native dtype   # reduce= is the seam
select_fovs(meta, n_fovs=1) -> {region: [fov, ...]}
SquidReader.read(region, fov, channel, z, t=0) -> (Y,X)   # lazy, one file
```

`project_well(..., reduce=)` is the pluggable hook 183 built. IMA-188 sits entirely on top of it —
**zero changes to 183/189.**

## 3. Design decisions

| # | Decision | Why |
|---|----------|-----|
| 1 | **`ThreadPoolExecutor`, not processes** | Per-well cost is I/O + `tifffile` decode + one `np.maximum`; decode and the ufunc both **release the GIL**, so threads parallelise the real work. A process pool would pickle each ~139 MB result across the boundary for nothing. |
| 2 | **Bounded in-flight window (≤ workers)** | Prime `workers` tasks, submit one refill per completion (window slides). Completed 139 MB results cannot pile up → **peak RSS ≈ workers × one-well footprint, flat in plate size**. |
| 3 | **Warm `reader.metadata` once before fan-out** | The 189 reader's lazy index/time-folders use unlocked check-then-set. Touching `.metadata` single-threaded populates them, so concurrent `read()` hits only immutable state — thread-safe, no locks. |
| 4 | **Fail loud** | A bad well propagates and aborts the stream. IMA-188 is a clean producer; skip/manifest/resume resilience is IMA-186's (in `TODOS.md`). |
| 5 | **Pluggable projector table** | `add_projector` / `available_projectors`, seeded `{"mip": project}`. EDF/mean drop in by name with zero engine edits. Named `add_`, **not `register_`, to avoid collision with image *registration* (alignment)** in this imaging codebase. |
| 6 | **Adaptive `workers` default** | `_default_workers()`: `os.process_cpu_count()` → `sched_getaffinity` → `cpu_count` → 1. CPUs usable by *this process* (affinity/cgroup aware), never a hardcoded constant. |

## 4. The contract IMA-184 consumes (finalized — stable)

```python
project_plate(reader, *, n_fovs=1, workers=None, projector="mip")
    -> Iterator[tuple[str, int, np.ndarray]]      # (region, fov, image(T, C, 1, Y, X))
```

- **Streaming producer.** Yields one well at a time, in **completion order** (not plate order —
  key downstream by `(region, fov)`). Consume lazily and write each well as it arrives; the plate
  is never resident.
- `image` is `(T, C, 1, Y, X)` **native dtype**, Squid canonical Zarr order (Z kept size-1) —
  serialize WITHOUT transposition.
- `workers=None` → adaptive default. `projector` names a table entry (default `"mip"`).
- **Fail-loud:** a corrupt/missing plane raises through the iterator; 184 does not need to handle
  partial wells (that's 186/resume).
- Public surface: `from squidmip import project_plate, add_projector, available_projectors`.

## 5. Testing (unit → cross commit on both datasets)

**Unit — clean-room** (`tests/test_engine.py`, faked reader, `pytest -m "not integration"`): projector
table (default, add, duplicate-raises, empty/non-callable), unknown-projector raises, yields every
well with correct shape/dtype, **pixel-identical to single-thread**, **worker-count determinism**,
`n_fovs` passthrough, **AC4 projector swap**, **fail-loud propagation**, **bounded-window / no
prefetch**, **metadata-warmed-before-reads**, invalid-workers. **14 tests.**

**Cross commit — real seam, no mocks** (`tests/test_integration.py`, `SECTION: IMA-188 ↔ IMA-183`),
on **sim_1536wp AND real hongquan**, the mandated three:
1. **Pixel-identical** — parallel output byte-for-byte equal to single-thread (`max|diff|=0`).
2. **Scaling (measured) + non-regression** — see §6 for why this is measured, not hard-gated.
3. **Bounded memory** — peak ≈ workers × one-well footprint, flat in plate size.
Plus an AC4 registry-swap end-to-end on real data.

**Full suite: 91 passed** (78 unit clean-room + 13 integration on both datasets).

## 6. Performance (measured) — and an honest ceiling

Single-thread baseline (183 §10, this machine, cache-warm): **~0.42 s/well**, peak **278 MB**,
1536 wells ≈ **~11 min**.

- **Memory — flat in plate size (the headline win).** workers=6, peak stays ~**1.8 GB** whether you
  consume 4, 8, 12, or 20 wells (`[1669, 1808, 1842, 1808] MB`). Bounded by the window, not the
  plate. The full 1536-well plate materialised would be **213 GB**.
- **Byte-identical.** Parallel == single-thread across all sampled wells, `max|diff| = 0`.
- **Throughput — bandwidth-bound on warm cache.** Wall-clock for 16 wells: `[7.0, 5.0, 4.5, 5.0] s`
  at workers `[1, 2, 4, 8]` — best ~**1.55× at 4 workers, then saturates**. Per-well cost at 8
  workers ≈ the single-thread baseline (measured 465 ms vs 463 ms — dead even).

  **Why:** parallelism *overlaps* wells, it does not make one well cheaper. On the warm sim the
  work is memory-bandwidth-bound (183 §10: `np.maximum` compute + cache-served memcpy), so N
  threads contend on one memory bus. **The large speedup is I/O-bound and needs cold / real
  storage**, where reads dominate and parallel reads scale — which the cache-served symlink sim
  structurally **cannot** exercise. This is the epic's open **Decision C** (throughput target vs
  Nick's real storage). The cross-commit therefore *measures and prints* the scaling curve and
  hard-gates only **non-regression** (`t_N ≤ t_1 × 1.2`); it does not gate a flaky warm-cache
  speedup. Correctness and bounded memory — the unconditional guarantees — are hard-gated.

## 7. Verification figures (saved to `~/Downloads` — drag each PNG in to embed)

- **Byte-identical:** single-thread `project_well` vs `project_plate` on real pixels, `|diff| = 0`.
  `[ drag squidmip_ima188_parallel_byte_identical.png here ]`
- **Throughput vs workers** (warm sim), against the §10 single-thread line — shows the ~1.55×
  bandwidth-bound plateau. `[ drag squidmip_ima188_scaling_vs_workers.png here ]`
- **Peak memory flat in wells consumed** (workers=6) vs the "if materialized" line — proves the
  bounded window. `[ drag squidmip_ima188_memory_flat.png here ]`

## 8. Block-by-block review feedback applied

- **Adaptive workers** (was `os.cpu_count()`): `_default_workers()`, affinity/cgroup aware.
- **`register_projector` → `add_projector`** ("registry" → "projector table"): avoids the
  image-*registration* misread in an imaging codebase.
- **Honest scaling gate:** replaced a flaky per-well-beats-baseline assertion (it failed at 465 vs
  463 ms) with measure-and-print + non-regression, once the warm-cache bandwidth ceiling was
  confirmed empirically.

---

## 9. IMA-184 handoff (full — pasteable)

```
You are the IMA-184 slot in the SquidMIP build. SquidMIP is a high-throughput z-stack
maximum-intensity-projection tool for Squid well-plate acquisitions, built as a review-first
state machine — one git worktree per ticket, landed in dependency order. You are in the
ima-184 worktree, in your own independent context. You are slot #4 (189 → 183(+187) → 188 → 184).

━━━ STEP 0 — READ THE SHARED STATE FIRST (before any code) ━━━
1. Notion "Squid MIP" (Blogs DB, id 3942dfbf-6ae4-81cf-bc41-fed456ddd398). Read the WHOLE page:
   the state-machine model, the "Cross commit" rule, and the completed IMA-189 / 183 / 188
   sections. Your section goes under a new "IMA-184" header.
2. On main: docs/ima-189-eng-review.md, docs/ima-183-eng-review.md, docs/ima-188-eng-review.md,
   and the squidmip/ package. `git -C <this-worktree> merge origin/main` so you have the reader +
   projection + parallel engine.
3. READ THE SCRIPTS, not just the docs: squidmip/_engine.py (project_plate, the stream you
   consume), squidmip/projection.py (project_well output shape), squidmip/reader.py (metadata:
   pixel_size_um, wellplate_format, channels[].display_color), tests/test_integration.py (append
   your "SECTION: IMA-184 ↔ 188/183" block BELOW the 188 section, do not edit it).

━━━ WHAT IS ALREADY DONE (state you inherit — 189 + 183 + 188, on main) ━━━
    from squidmip import open_reader, select_fovs, project_well, project_plate, add_projector
    - open_reader(path).metadata: regions, fovs_per_region, channels[{name, display_name,
      display_color, ex}], n_z, z_levels, dz_um, pixel_size_um, wellplate_format, frame_shape,
      dtype, n_t. acquisition.yaml REQUIRED -> pixel_size_um and wellplate_format GUARANTEED present.
    - 188's PUBLIC ENTRY POINT you consume (confirmed, stable):
        project_plate(reader, *, n_fovs=1, workers=None, projector="mip")
            -> Iterator[(region, fov, ndarray(T, C, 1, Y, X))]     # native dtype, TCZYX, Z=1
      Streaming producer, completion order (key by (region, fov)); serialize WITHOUT transposition.
      Fail-loud (a bad well raises through the iterator — you do NOT handle partial wells).
      Bounded memory: consume it lazily and WRITE EACH WELL AS IT ARRIVES; never collect the plate.

━━━ YOUR TASK — IMA-184: output (OME-zarr canonical + per-well TIFF) ━━━
    - Write the projected plate as multiscale OME-zarr (canonical, navigable): axes TCZYX,
      pixel_size_um as physical scale, channels[].display_color as omero rendering (opens in
      ndviewer_light). Per-well TIFF export (Nick's ask): one <well>.tif per well.
    - VENDOR the tilefusion OME-zarr writer (copy create_zarr_store / write_ome / colors.py into
      squidmip; do NOT import tilefusion — heavy __init__). See the IMA-184 writer memory notes.
    - Consume project_plate lazily: stream each (region, fov, image) straight to disk, bounded memory.
    - Out of scope: parallel projection (188), UI/montage (185), CLI (186).

━━━ CROSS COMMIT (MANDATORY — you own 184 ↔ 188/183) ━━━
    Append "SECTION: IMA-184 ↔ 188/183" to tests/test_integration.py. TWO layers, as every slot:
    (1) UNIT, mocked at the seam (feed a fake project_plate stream) — clean-room, no data;
    (2) CROSS COMMIT, real seam no mocks, on BOTH datasets (sim_1536wp + real hongquan): write the
        output, read the OME-zarr back, assert it equals the in-memory projected plate (pixel-exact,
        dtype, TCZYX); a per-well TIFF round-trips; it opens in ndviewer_light.
    @pytest.mark.integration, green before merge. A slot isn't done until its cross commit is green.

━━━ CLEAN-CODING CONVENTIONS (every slot — read as one hand) ━━━
    Thin public surface in squidmip/__init__; logic in a private module with an ASCII data-flow
    docstring. No cross-repo imports (vendor, don't import). Preserve native dtype; fail LOUD; no
    dead attributes; bounded/streaming memory. Tests: unit (mocked seam) AND the cross commit;
    CI clean-room `pip install .[test]` + `pytest -m "not integration"`. Encode intent a-priori.

━━━ HOW YOU CLOSE OUT (handoff for the next slot = IMA-185) ━━━
    - Write docs/ima-184-eng-review.md (mirror 189/183/188).
    - Populate an "IMA-184" section on the Notion page (Intent a-priori → the output contract 185
      consumes → Testing (unit + cross commit on both datasets) → verification). End with a
      "Placeholder for IMA-185" header.
    - VERIFICATION FIGURES (184 is visual): ndviewer_light screenshot of the written plate; an
      OME-zarr read-back-vs-in-memory diff panel. Save to ~/Downloads, add "[ drag PNG here ]"
      placeholders in Notion (Notion MCP cannot embed local PNGs).
    - After block-by-block review + green cross commit, merge --no-ff to main and push. Produce the
      IMA-185 handoff prompt (same structure).

STOP after Step 0 and confirm your understanding of the inherited project_plate contract before
writing code. Encode intent a-priori.
```

---

**Verdict:** IMA-188 CODE LOCKED — 188↔183 cross-slot green on both datasets; correctness and
bounded-memory unconditional; throughput bandwidth-bound on the warm sim, real speedup deferred to
cold/real storage (Decision C).
