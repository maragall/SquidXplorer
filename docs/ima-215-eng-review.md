# IMA-215 — Engineering Review (longform)

**Ticket:** IMA-215 — Coordinates reader (parse `coordinates.csv` → per-FOV stage positions)
**Branch:** `juliomaragall/ima-215-coords-reader`
**Review:** `/plan-eng-review`, completed 2026-07-20. Outside voice: Claude subagent (Codex not installed).
**Status:** IMPLEMENTED — CLEARED WITH CONCERNS — 8 issues found, 12 decisions locked, 2 decisions reversed by the
outside voice, 0 unresolved, 1 strategic flag left open for Julio.

> This document is the reviewable record of the plan-eng-review. The working plan lives at
> `.spec/open/ima-215.md` (gitignored), so everything load-bearing is reproduced here and the
> merge diff is self-contained.
>
> **T1–T8 are implemented** (`squidmip/_coordinates.py`, wired into both readers,
> `tests/test_coordinates.py`). 31 new tests pass; the full unit suite is 142 passed / 2 skipped.
> Verified against real tables: `20x_scan` 36 entries, `sim_1536wp` 1536, the 10x dataset 55
> collapsed from 550 rows — zero warnings on all three.

---

## 1. Executive summary

The original three-line spec rested on a **factually wrong model of the input format**. It framed
the two CSV schemas as microscope generations ("Monkey-style" vs "20x-style"). They are not: they
are two files at different paths inside the *same* acquisition, and a single 20x dataset carries
both. Fixing that model is most of this review.

The outside voice then reversed two of my own decisions. The important one: my D5 cross-check was
**permutation-invariant** and therefore did not catch the bug it was written to catch. That is
corrected below (D5-R).

The ticket remains buildable and worth building, but its risk profile is: **zero consumers today,
no fully-openable real dataset to validate against, and it reverses a decision IMA-189 locked
deliberately.** Section 8 is the open flag.

---

## 2. Corrected format model (the load-bearing finding)

```
<acq>/
├── coordinates.csv                      region,x (mm),y (mm),z (mm)
│                                        ← PLANNED grid. z column EMPTY in 3 of 4 datasets.
├── acquisition parameters.json          (legacy — _acquisition.py REFUSES this)
├── original_coordinates/
│   └── original_coordinates_0.csv       ← third copy, labelled schema
└── 0/                                   # timepoint folder
    ├── coordinates.csv                  region,fov,z_level,x (mm),y (mm),z (um),time
    │                                    ← ACTUAL executed positions, post-autofocus.
    └── {region}_{fov}_{z}_{channel}.tiff
```

Verified on disk:

| Dataset | root CSV | `0/` CSV | rows (root / t0) | `acquisition.yaml` | openable? |
|---|---|---|---|---|---|
| `20x_scan_2025-09-05_17-57-50` | 4-col | 7-col | 36 / 36 | **NO** | **no** |
| `test_10x_laser_af_z_stack…yy` | 4-col | 7-col | 55 / 550 | **NO** | **no** (OME-TIFF, 55 files) |
| `synthetic_2x2_wellplate` | 4-col | **4-col** | — | **NO** | **no** |
| `sim_1536wp` | 4-col | — | 1536 / — | yes | **no** (122,880 broken symlinks) |

Three consequences:

1. **Dispatch on the header's column set, never on dataset provenance.** A 20x dataset carries both
   schemas. *(Correction to my own first pass: I initially noted "space before the paren" as a
   distinguishing tell. It is not — both headers use `x (mm)` with identical spacing. The only
   valid tell is the presence of the `fov` / `z_level` columns.)*
2. **Root and timepoint disagree numerically**: `manual0` root x=`98.2245316296875` vs t0
   x=`98.22418125`. Delta = 0.350 µm = **1.08 pixels** at the 20x pixel size of 0.325 µm.
3. **The labelled file is one row per `(fov, z_level)`** — 550 rows for 55 FOVs. XY repeats
   identically across z; z climbs (3930.75, 3932.25, 3933.75…).

Also verified: `fov` numbering **restarts per region** (28 distinct fov values across manual0 +
manual1 totalling 55 FOVs), confirming `(region, fov)` as the key.

`original_coordinates_0.csv` differs from `0/coordinates.csv` on 914 lines, but **every difference
is float repr noise** (`3930.75` vs `3930.7499999999995`) — not a semantic disagreement. It is
ignored, but that float noise is itself load-bearing for D3-R below.

---

## 3. Locked decisions

| # | Decision | Rationale |
|---|---|---|
| **D2** | Read `<acq>/{t}/coordinates.csv` by preference, fall back to `<acq>/coordinates.csv`. Dispatch on the presence of `fov`/`z_level` columns, tolerant of BOM, CRLF, trailing whitespace and the optional `time` column. | *Rationale restated after the outside voice.* The XY gain is only ~1 pixel, so accuracy is **not** the justification. The real one: the root file has **no fov labels and no z at all** (empty in 3 of 4 datasets). The timepoint file is the only source with both. |
| **D3-R** | Collapse to `(region,fov) → (x, y, z@ lowest z_level)`. Check XY constancy across z with a **sub-pixel tolerance and a warning**, not a hard assert. | *Revised by outside voice.* Float repr noise is real and observed (914 lines of it). A hard assert inside `metadata` would brick the whole reader on formatting drift. |
| **D4-R** | Normalize everything to micrometres, and name the key **`fov_positions_um`**. | `z (um)`≈3930 vs `z (mm)`≈3.9 is a silent 1000× error. *Revised:* every other physical key carries its unit (`dz_um`, `pixel_size_um`), so the bare name was inconsistent. |
| **D5-R** | **Fabricate a fov index only for regions with exactly one row.** For multi-row regions in the unlabelled schema, **omit the region and warn** — never guess ordering. | **Reversal.** My original D5 compared the fabricated fov set against `fovs_per_region` and raised on mismatch. That check is **permutation-invariant**: a shuffled CSV has identical counts and passes, so it did not catch the bug it existed for. Single-row regions are unambiguous (fov=0); multi-row ordering is unverifiable, so we decline rather than guess. |
| **D6** | Shared parser in new `squidmip/_coordinates.py`, wired into `SquidReader` **and** `SquidOMEReader`. | The two classes deliberately present the same metadata interface, and the only 10-z dataset is OME-TIFF. |
| **D7** | Fix four stale docstrings: `reader.py:28`, `reader.py:29` (already false today), `_acquisition.py:13`, `tests/conftest.py:5`. | `reader.py:29` promises "the flat JSON as a legacy fallback" while `_acquisition.py:47` raises `FileNotFoundError` and refuses JSON. Stale docs are worse than none. |
| **D8** | stdlib `csv`, not pandas. | `pandas>=2.0` at `pyproject.toml:17` has **zero** usage package-wide. A ≤7-column file does not justify the first import. Removal filed as a TODO rather than done here (packaging blast radius). |
| **D9-R** | Absent or malformed `coordinates.csv` → `{}` **and a warning**. Never raise out of `metadata`. | Regression guard plus risk posture: with no consumer today, a bad sidecar must never brick the MIP pipeline. |
| **D10** | Compute inside the memoized `metadata` property. | The engine loops over wells; an un-memoized parse re-reads a 1536-row file per access. Safe now that no path raises (D5-R, D9-R). |
| **D11** | Positions are **t=0 only**, documented explicitly. | Each timepoint folder has its own CSV with its own post-AF values. Folded into the existing multi-timepoint TODO rather than keyed by `t` now. |
| **D12** | OME-TIFF in-band `Plane PositionX/Y/Z` is **not** parsed. | The `yy` dataset routes to `SquidOMEReader` and OME-XML carries positions in-band — a fourth source. Deferred; the sidecar CSV is simpler and already handled by D6. |
| **D13** | `tests/test_reader.py:33`'s exact-key-set assertion **must** be updated. | **Mandatory regression fix**, no discretion: that test asserts `set(meta) == {…11 keys…}`, so adding a twelfth key fails it. |

---

## 4. Outside-voice challenge and disposition

Codex is not installed; an independent Claude subagent with fresh context ran the challenge. It
raised 14 points. Dispositions:

| # | Finding | Disposition |
|---|---|---|
| 1 | No evidence dataset is openable (missing `acquisition.yaml`) | **Accepted, and escalated** — I additionally found `sim_1536wp`'s 122,880 TIFFs are broken symlinks. Fixture work added as T7. |
| 2 | D5 refuses on a normal condition; could brick `metadata` | **Accepted** → D5-R omits+warns, D9-R never raises. |
| 3 | D5 is permutation-invariant, doesn't catch its target bug | **Accepted — the most important finding.** → D5-R. |
| 4 | Dual-format path buys ~1 pixel; parse labelled-only instead | **Partially accepted.** Delta confirmed at 1.08 px, so D2's rationale is restated. Rejected as a plan: labelled-only drops `sim_1536wp`, the 1536-well plate case, which is the tool's headline input. |
| 5 | `original_coordinates` differs from `0/` | **Rejected on the evidence.** All 914 differing lines are float repr noise, not semantic. Kept as a TODO. |
| 6 | OME datasets carry in-band positions | **Accepted as a deferral** → D12. |
| 7 | No `t` in the key | **Accepted** → D11 (documented, not keyed). |
| 8 | Unit not in the key name; z is `None` for root format | **Accepted** → D4-R renames to `fov_positions_um`; z declared `Optional` in the contract. |
| 9 | `test_reader.py:33` and conftest fixtures not in the plan | **Accepted** → D13 + T7. Verified: the assertion is an exact set comparison. |
| 10 | conftest:5 is a missing fixture, not just a stale docstring | **Accepted** → T7 builds the fixtures; D7 fixes the prose. |
| 11 | pandas decision never made | **Rejected as stated** — D8 did make it (stdlib `csv`). Removal stays a TODO, not this ticket. |
| 12 | Header-spacing tell distinguishes nothing | **Accepted** → D2 dispatches on the `fov`/`z_level` column set. |
| 13 | Hard XY assert will fire on float formatting | **Accepted** → D3-R uses tolerance + warn. |
| 14 | Strategic: zero consumers, reverses IMA-189, absent from build order | **Surfaced, not resolved** → Section 8. |

**Cross-model tension.** The outside voice argued for the minimal labelled-only parser (its #4); I
kept the dual-format path because `sim_1536wp` is the 1536-well plate the product is built around
and labelled-only silently yields nothing for it. D5-R absorbs the safety half of that argument —
we no longer *guess* in the ambiguous case, we decline — which is where the two positions actually
converge.

---

## 5. Data flow (final)

```
open_reader(path) ──► SquidReader │ SquidOMEReader
                            │
                      .metadata (memoized — D10)
                            │
        _coordinates.load_fov_positions_um(root, t0, fovs_per_region)
                            │
     ┌──────────── locate (D2) ─────────────┐
     │ {t}/coordinates.csv → <acq>/…  → none│
     └────┬──────────────┬─────────────┬────┘
          ▼              ▼             ▼
    sniff column set  sniff       return {}  (D9-R)
          │
   ┌──────┴───────┐
   │ has `fov`?   │
   └──┬────────┬──┘
    yes│       │no
      ▼        ▼
 group by   per-region row count
 (region,   ├── exactly 1 row → fov = 0        (unambiguous)
  fov)      └── >1 row → OMIT region + warn    (D5-R: never guess order)
      │        │
 XY const?     │
 tolerance     │
 + warn (D3-R) │
      └────┬───┘
           ▼
   normalize to µm (D4-R)
   x,y: mm×1000 │ z: 'z (um)' as-is │ 'z (mm)'×1000 │ empty → None
           │
           ▼
  {(region,fov): (x_um, y_um, z_um|None)}   → metadata['fov_positions_um']
```

---

## 6. Test coverage plan

All paths are new; coverage today is 0%.

```
_coordinates.load_fov_positions_um()
├── location (D2)      [GAP] t0 present → wins over root (assert values differ)
│                      [GAP] root only → parsed
│                      [GAP] neither → {}                       ★ REGRESSION (D9-R)
├── dispatch (D2)      [GAP] labelled 7-col → fov from column
│                      [GAP] unlabelled 4-col → single-row path
│                      [GAP] BOM / CRLF / trailing whitespace tolerated
│                      [GAP] unrecognised header → {} + warn
├── collapse (D3-R)    [GAP] 550 rows / 55 FOVs → 55 entries
│                      [GAP] XY drift within tolerance → warn, not raise  ★ float-noise
│                      [GAP] lowest z_level chosen when z_level 0 absent
├── fabrication (D5-R) [GAP] single-row region → fov 0                    ★ CRITICAL
│                      [GAP] multi-row unlabelled region → omitted + warn ★ CRITICAL
├── units (D4-R)       [GAP] z (um) unchanged │ z (mm) ×1000 │ x,y ×1000 │ empty z → None
└── keying             [GAP] fov restarts per region → no collision

reader.py integration
├── [GAP] SquidReader.metadata['fov_positions_um']
├── [GAP] SquidOMEReader.metadata['fov_positions_um']            (D6)
├── [GAP] test_reader.py:33 key-set assertion updated            ★ REGRESSION (D13)
└── [GAP] parsed once across repeated .metadata access           (D10)
```

`tests/test_coordinates.py`, matching `tests/test_reader.py` conventions (pytest, `tmp_path`
real-shaped fixtures, skip-if-missing for real data).

---

## 7. Failure modes

| Failure | Test | Handling | User sees |
|---|---|---|---|
| Unverifiable row order in unlabelled multi-row region | D5-R test | region omitted + warn | positions absent for that region, warned |
| µm/mm mixed | unit tests | normalized at parse | correct values |
| CSV absent | D9-R test | `{}` | positions absent, no crash |
| CSV malformed | header test | `{}` + warn | pipeline unaffected |
| XY float drift across z | D3-R test | tolerance + warn | correct values |
| Root file read instead of timepoint | precedence test | precedence rule | correct values |

**No critical gaps.** Every failure mode has a test and a non-fatal path; none fails silently, and
none can brick `metadata` for a field that currently has no consumer.

---

## 8. Open flag for Julio (not resolved by this review)

The outside voice's strongest point is strategic, and I could not settle it from the codebase:

- **No consumer exists.** `grep` for the field returns zero hits.
- **It reverses a deliberate IMA-189 decision**, whose rationale is written into three docstrings.
- **IMA-215 is absent from the recorded build order** (189→183→188→184→185→186→192).
- **Nothing validates it end to end**: no real dataset is currently openable.
- **The consumer's requirement is unknown** — whether IMA-193 wants the planned grid or the actual
  post-AF positions, stage-absolute or well-relative, is undecided. If it wants something other
  than what D2/D3-R produce, this parser gets rewritten.

I built the plan to minimise what gets thrown away if that happens: one self-contained module, no
guessing, non-fatal everywhere. **If you want the smaller bet, the alternative is the outside
voice's labelled-format-only parser (~40 lines, no fabrication, no `sim_1536wp` support)** — say so
and T1/T2 collapse into one task.

---

## 9. Implementation tasks

- [x] **T1 (P1, human ~3h / CC ~20min)** — `_coordinates.py`: header-set dispatch + µm normalization (D2, D4-R)
- [x] **T2 (P1, human ~2h / CC ~15min)** — `_coordinates.py`: single-row-only fabrication, omit+warn otherwise (D5-R)
- [x] **T3 (P1, human ~1h / CC ~10min)** — `_coordinates.py`: per-FOV collapse, tolerance-based XY check (D3-R)
- [x] **T4 (P1, human ~1h / CC ~10min)** — `reader.py`: expose `fov_positions_um` on both readers inside memoized `metadata` (D6, D10)
- [x] **T5 (P1, human ~30min / CC ~5min)** — tests: absent/malformed-CSV regression guard (D9-R)
- [x] **T6 (P2, human ~45min / CC ~10min)** — docs: fix four stale docstrings, extend the discovery diagram (D7)
- [x] **T7 (P1, human ~2h / CC ~15min)** — tests: add 7-col / 4-col / absent CSV fixtures to `conftest.py` **and update the `test_reader.py:33` key-set assertion** (D13)
- [x] **T8 (P3, human ~15min / CC ~5min)** — document the t=0 scope in the metadata contract (D11)

**Status:** all tasks landed.

**Parallelization:** sequential — every task converges on one new module and its wiring.

---

## 10. NOT in scope

- **Consuming the positions.** Deliberate groundwork; see Section 8.
- **`original_coordinates/`.** Differs only in float repr. TODO filed.
- **Removing unused `pandas>=2.0`.** Real, but a packaging change. TODO filed.
- **OME-XML in-band positions.** D12.
- **Per-timepoint positions (Nt>1).** D11; folded into the existing multi-timepoint TODO.
- **Per-plane z.** Collapsed by D3-R; geometry recoverable from `dz_um` + `z_levels`.
- **Stitching / mosaic placement.** IMA-193's.

## 11. TODOs captured (`TODOS.md`)

- Remove the unused `pandas>=2.0` runtime dependency.
- Reconcile `original_coordinates/original_coordinates_0.csv` if a dataset ever disagrees semantically.

## 12. Review metadata

- Issues found: 8 (5 architecture, 3 code quality) + 1 performance + 14 outside-voice points
- Decisions locked: 13 (2 reversed by the outside voice: D5→D5-R, D3→D3-R)
- Critical gaps: 0 · Unresolved decisions: 0 · Open strategic flag: 1 (Section 8)
- Prior learning applied: `squid-coordinates-no-fov-column` (9/10, 2026-07-04)
