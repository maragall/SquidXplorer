# IMA-189 — Engineering Review (longform)

**Ticket:** IMA-189 — Format-aware ingest (auto-recognize Squid filenames/format)
**Branch:** `juliomaragall/ima-189-auto-recognize-squid-format`
**Review:** `/plan-eng-review`, completed 2026-07-04. Outside voice: Claude subagent (Codex not installed).
**Status:** CLEARED — scope reduced, 7 decisions locked, 0 unresolved, 0 critical gaps.

> This document is the reviewable record of the plan-eng-review. The working plan lives at
> `.spec/open/ima-189.md` (gitignored). Everything load-bearing is reproduced here so the
> diff on merge is self-contained. The one behavior-bearing file this ticket adds to the
> tracked tree so far is `TODOS.md`; the implementation (T1–T6) lands after this review.

---

## 0. Post-review corrections (2026-07-05, after technical feedback + Squid-source verification)

Julio's feedback sent me to the `Cephla-Lab/Squid` source (`multi_point_worker.py`,
`job_processing.py`, `utils_acquisition.py`, `_def.py`, `models/acquisition_config.py`,
`config/repository.py`). Three earlier claims were **wrong** and two decisions change. The
corrections below OVERRIDE the corresponding statements later in this doc.

- **Frame size / dtype are NOT fixed.** `4168×4168` is the *unbinned* crop
  (`_def.py:752 CROP_WIDTH_UNBINNED=4168`); default binning is **2×2**
  (`BINNING_FACTOR_DEFAULT=2`) → ~2084². Pixel format spans MONO8→uint8, MONO12/MONO16→uint16
  (`squid/camera/utils.py:389`), plus ROI/crop. **The reader must read `frame_shape` + `dtype`
  from the first real frame** (the plan already does) and preserve **native dtype**, not
  "always uint16." (Fixes §3.1 and Decision 5.)
- **tilefusion's `float32` cast is NOT lossy for us — my "destroys exact uint16" claim was wrong.**
  uint16 ⊂ float32 exactly (max 65535 < 2²⁴), and `_to_grayscale_2d` is a no-op for a 2D plane
  (`individual_tiffs.py:234`). So tilefusion's read is lossless for 2D grayscale — which is why
  **your stitcher's MIP is fine using it.** SquidMIP reads raw only for **native dtype + half the
  memory (2B vs 4B/px) + zero dependency**, not for correctness. The only genuinely lossy path is
  grayscale-averaging a real RGB plane, which we exclude anyway. (Fixes §3.2 #1 and Decision-Build.)
- **`coordinates.csv` schema is version-dependent.** Current Squid writes
  `region, fov, z_level, x (mm), y (mm), z (um), time` (`multi_point_worker.py:801`) — it *does*
  have `fov`. The hongquan dataset I tested is an **older** schema (`region, x, y, z(mm)`, no
  `fov`). The filename token `{region}_{fov}_{z}_{channel}` is the **common denominator across
  versions**, and `fov` there is the per-region enumeration index (resets per region,
  `multi_point_worker.py:1063,1107`) — identical to the CSV `fov` when present. So Decision 4
  (parse filenames) is still correct, and now better justified: filenames are format-robust.
  (Refines §3.2 #2 and Decision 4.)

**Decision changes:**
- **Decision 2 (color join) — REVERSED from "strict raise" to LAYERED FALLBACK.** Legacy/pre-YAML
  acquisitions exist, so never hard-fail on a missing color. Order: (a) YAML channel-level
  `display_color` (Squid v1.0+, `acquisition_config.py:116`, always written, default `#FFFFFF`);
  (b) pre-v1.0 nested `camera_settings.*.display_color` (the hongquan layout — read the *first*
  camera key, not hardcoded `'1'`); (c) hardcoded `CHANNEL_COLORS_MAP` matched by wavelength / BF
  suffix; (d) neutral default + warning. See the authoritative map below.
- **Decision 5 (dtype) — WIDENED** to "exact **native** 2D plane (uint8 **or** uint16)"; still
  raise on `ndim > 2` (RGB/color, brightfield deferred — agreed out of scope for now).
- **`open_reader` is a FORMAT DISPATCHER, not just a filename parser.** Squid writes
  `INDIVIDUAL_IMAGES` (default), `MULTI_PAGE_TIFF` (`{region}_{fov}_stack.tiff`), `OME-TIFF`
  (`ome_tiff/`), and Zarr (`job_processing.py`). Implement the individual-tiffs reader now; leave
  the dispatch seam for other formats/customers. This mirrors tilefusion's `open_reader` *design*
  (pattern reuse, not code import) — reconciling your "reuse tilefusion" note with "no cross-repo
  import."

### Authoritative Squid `CHANNEL_COLORS_MAP` (`software/control/_def.py:1041`)
| Key | Hex | Name | | Key | Hex | Name |
|-----|-----|------|-|-----|-----|------|
| 405 | `#20ADF8` | bop blue | | 730 | `#770000` | dark red |
| 488 | `#1FFF00` | green | | R | `#FF0000` | red |
| 561 | `#FFCF00` | yellow | | G | `#1FFF00` | green |
| 638 | `#FF0000` | red | | B | `#3300FF` | blue |

This differs from the draft map pasted in review feedback (which had 405→`#0000FF`, 488→`#00FF00`,
and `_B/_G/_R` keys). The map above is verbatim from Squid master and **matches the hongquan
YAML** — use it. Match brightfield by wavelength/letter substring in the channel name.

---

## 1. Executive summary

IMA-189 is the **keystone** of SquidMIP: a reader that ingests a Squid `individual_tiffs`
acquisition directory, discovers everything from metadata + filenames (no manual per-file
input), and yields exact pixel planes on demand.

The review made one consequential change to the original spec and locked six further
decisions. The headline:

- **The spec said "depend on tilefusion." The review reversed that.** IMA-189 ships as a
  **standalone** package with no tilefusion dependency. The dependency was justified in the
  spec by "one dependency, both ends" — but that reasoning is really about IMA-184 (output /
  OME-zarr writer), not ingest. Once the reader parses filenames itself, the only tilefusion
  reuse left was ~15 trivial lines plus a display-color parser — not enough to justify a
  cross-repo pinned git dependency. At IMA-184 the OME-zarr writer is **vendored (copied in),
  not imported** — tilefusion's package `__init__` is heavy, so even the output ticket lifts
  `create_zarr_store` / `write_ome` / `colors.py` rather than depending on the whole library.
  **SquidMIP takes no cross-repo dependency at any slot.**

- **Net effect: the plan got smaller and safer.** One repo, PyPI-only deps, no cross-repo pin
  dance, and a read path that is correct-by-construction (raw `tifffile.imread`,
  filename-derived addressing).

The reversal was triggered by the independent outside-voice pass, which challenged the
hybrid/dependency direction I had converged on. That is the cross-model review working as
intended: a second model caught a strategic miscalibration the section-by-section review had
accepted.

---

## 2. What the ticket is (context)

From the spec:

> z-stack maximum image projection tool: Of course can do this in FIJI, but would be super to
> do in a more high-throughput way that your software can recognize file names and data format
> more easily. — Nick (Slack)

The projection itself is trivial (`np.max` over z, identical to FIJI). What makes SquidMIP
high-throughput is deleting the manual, format-blind workflow: parse the whole acquisition
from its metadata so no file is opened by hand. The same parsed metadata drives batch
projection **and** the plate-view navigation. **IMA-189 is that parser.** Projection (183/188),
output (184), and UI (193) are separate tickets.

Position in the epic (IMA-176): IMA-189 is issue #2, the ingest keystone that the projection,
output, and UI tickets all build on.

---

## 3. Investigation & verification (evidence, not assumption)

Everything below was verified against the real dataset
(`~/Downloads/z_stack_2026-05-15_18-39-28.532906 hongquan`) and the actual tilefusion source
(`~/CEPHLA/projects/stitcher/src/tilefusion/io/`), not inferred.

### 3.1 The real Squid `individual_tiffs` layout (verified)

```
<acq>/
├── acquisition parameters.json      # Nz=3, Nt=1, dz(um)=1.031, objective.magnification=20, sensor_pixel_size_um=3.76
├── acquisition_channels.yaml        # channels[].name + camera_settings.'1'.display_color (#hex) + exposure_time_ms
├── acquisition.yaml
├── coordinates.csv                  # columns: region, x (mm), y (mm), z (mm)   ← NO fov, NO z_level
└── 0/                               # time index 0 (1/, 2/, … if Nt>1)
    └── {region}_{fov}_{z}_{channel}.tiff     # e.g. B2_0_0_Fluorescence_638_nm_-_Penta.tiff
```

Verified facts:
- **576 tiffs** in `0/`; **48 coordinate rows**; 3 regions (B2, B3, B4) × 16 FOVs × 3 z × 4 ch.
- Every frame is **`(4168, 4168)` uint16, 2D single-plane** (confirmed via `tifffile.imread`).
- 4 fluorescence channels only (405/488/561/638) — **no brightfield/RGB in this dataset.**
- YAML colors: 405 `#20ADF8`, 488 `#1FFF00`, 561 `#FFCF00`, 638 `#FF0000`.
- YAML names use spaces + dashes (`Fluorescence 638 nm - Penta`), listed **descending**;
  filenames use underscores (`Fluorescence_638_nm_-_Penta`). `display_color` is **nested**
  under `channels[].camera_settings.'1'.display_color`; exposure is `exposure_time_ms`.
- `pixel_size_um = 3.76 / 20 = 0.188`.
- The `stitcher` repo (tilefusion's home) is **PUBLIC** (`gh repo view` → `isPrivate:false`).

### 3.2 What tilefusion already had

`tilefusion.io` (`base.py`, `individual_tiffs.py`, `_squid.py`) has a working `individual_tiffs`
reader: `open_reader()`, a `Reader` Protocol, `load_individual_tiffs_metadata()`,
`read_individual_tiffs_region/_tile()`, and `_squid.load_acquisition_params()`. It has a real
pytest suite (`tests/test_io_squid.py`, `test_io_readers.py`, `test_io_factory.py`, fixtures).

**Why we do NOT reuse it for IMA-189** (each verified in source):

1. **Lossy read path.** `read_individual_tiffs_region` ends with `.astype(np.float32)`
   (`individual_tiffs.py:366`) and runs every plane through `_to_grayscale_2d`, which averages
   RGB (`:288`, `:363`). Both destroy exact uint16 pixels. stitcher *wants* float32 (registration
   math), so this is correct for stitcher and wrong for an exact-pixel MIP consumer. We must not
   change it (blast radius on stitcher), and we must not route MIP reads through it.

2. **Fabricated FOV index.** `coordinates.csv` has no `fov` column, so
   `load_individual_tiffs_metadata` falls into the region-only branch (`individual_tiffs.py:116-124`)
   and **invents** `fov` as a per-region sequential row index. The spec's
   `read('B3', 15, …)` addresses by the *filename* fov token, which can disagree with a
   row-order index → silent wrong tile. (See decision 4.)

3. **The remaining reuse is trivial.** After the reader parses filenames itself, the only
   tilefusion pieces worth reusing were `_squid.load_acquisition_params` (~15 lines reading a
   JSON) and a new color parser. That does not justify a cross-repo pinned git dependency.
   (See the build decision.)

---

## 4. How the plan evolved (decision archaeology)

The review did not land on "standalone" immediately. The path matters, because it shows the
reversal was earned, not assumed:

```
Spec draft:           "depend on tilefusion" (Decision A), reader ~90% there, scope as thin adapter
      │
      ▼  Step 0 verification (read tilefusion source, dataset, packaging)
Interim direction:    HYBRID — push the 2 gaps (uint16 read, YAML colors) INTO tilefusion,
                      thin SquidMIP adapter, deliver via pinned git dependency
      │  (user confirmed: owns tilefusion, edit freely, git-dependency delivery)
      ▼  decision 4: adapter derives (region,fov,z,channel) from FILENAMES, not CSV
Refinement:           "uint16 raw read" is just tifffile.imread — nothing non-trivial to push
                      upstream. Only the color parser was left as a tilefusion change.
      │
      ▼  OUTSIDE VOICE challenge: a ~30-line color reader doesn't justify a cross-repo dep;
                      display color is a SquidMIP presentation concern, not stitcher's
      ▼
FINAL (locked):       STANDALONE ingest in SquidMIP. No tilefusion dep. tilefusion → IMA-184.
```

**Superseded decisions** (recorded so a future session doesn't re-litigate):
- "depend on tilefusion" (spec) → **superseded**: standalone.
- "hybrid: gaps in tilefusion" → **superseded**: color parser lives in SquidMIP.
- "git-dependency delivery" / "edit tilefusion freely" → **moot for 189**: no tilefusion change,
  no dep; both re-apply at IMA-184.

---

## 5. Locked decisions (with full rationale)

| # | Decision | Rationale | Your preference it serves |
|---|----------|-----------|---------------------------|
| **Build** | **Standalone SquidMIP package. No tilefusion dependency in 189.** tilefusion enters at IMA-184 (output). | The only reuse left was trivial; a cross-repo pin isn't justified for ingest. Removes cross-repo coupling from the keystone. | Right-sized diff; blast-radius/boring-by-default |
| **1** | Channel identity: **filename form is canonical** (`Fluorescence_638_nm_-_Penta`) — the key `read()` takes. Metadata also carries `display_name` (human, from YAML) + `display_color` + `ex`. | AC3/AC4 fix the underscored form as the addressing key; one unambiguous key, no second lookup. | Explicit over clever |
| **2** | YAML join is **strict, benign-absence degrades**: YAML present → every filename channel must have a YAML entry else **raise** naming the channel; YAML absent → names from filenames + `display_color=None`. | Colors must be read, not guessed (AC2). Silently painting a fluor channel the wrong color is a scientific error worse than stopping. Each mismatch shape has a defined outcome. | Handle more edge cases; explicit |
| **3** | **Add minimal CI now** — GitHub Actions: clean-room `pip install .` + `pytest`. | Proves the build is reproducible off the dev's conda env (the exact ambient-import trap this repo is in). stitcher public → no dep auth needed. | Well-tested; own it in production |
| **4** | The reader derives its `(region, fov, z, channel)` index by **parsing filenames** (ground truth), not `coordinates.csv` row order. | `coordinates.csv` has no `fov`/`z_level` column; row-order-fabricated fov could disagree with the filename token → silent wrong tile. Files are ground truth. | Handle edge cases; explicit > clever |
| **5** | **Scope to 2D uint16.** `read()` asserts a 2D plane; a 3D/RGB (brightfield) plane **raises** "non-2D channel not supported (brightfield deferred)". | MIP is a fluorescence-z operation. Loud refusal is honest scoping; silently returning `(Y,X,3)` corrupts the projection. | Explicit; fix the whole thing not the demo path |
| **6** | **`read(region, fov, channel, z, t=0)`**; `t` selects the `t`-th time folder (`0/`, `1/`…). `metadata.n_t` discovered from time folders. | Filenames carry no time token (time = separate dirs). Makes the contract honest + forward-compatible without building multi-t traversal. | Explicit; don't over-build |

---

## 6. Outside-voice challenge (independent second opinion) and disposition

An independent Claude subagent (fresh context) was asked to find what the section-by-section
review *missed*. It returned 13 findings. Disposition:

| # | Finding | Disposition |
|---|---------|-------------|
| 1 | `read()` has no `t` arg but metadata reports `n_t` — API inconsistent | **→ Decision 6** (add `t=0`, keep `n_t`) |
| 2 | "exact uint16" breaks on Squid RGB/brightfield planes (`(Y,X,3)`) | **→ Decision 5** (raise on non-2D; brightfield deferred) |
| 3 | `load_channel_colors` normalization unspecified / collision risk | **Folded in:** pin `spaces→underscore` (dash preserved), add round-trip + collision test |
| 4 | A ~30-line color reader doesn't justify a cross-repo pinned git dep; color is a presentation concern | **→ Build reversal** (standalone; color parser in SquidMIP) |
| 5 | Chicken-and-egg: can't pin `@commit` until tilefusion PR merges; local editable so only CI tests the pin | **Resolved by** build reversal (no dep, no pin) |
| 6 | CI clean-room needs private-repo auth | **Dismissed** — verified stitcher is public |
| 7 | `dz_um`/`n_z` two sources of truth, no reconciliation | **Folded in:** filenames = z-range ground truth; `dz_um` from params; warn if `Nz(params) ≠ max_z+1`; test |
| 8 | `fovs_per_region` scalar can't represent non-uniform FOV counts | **Folded in:** `fovs_per_region` is `{region: [fov ids]}`; fov = raw filename token (may be sparse) |
| 9 | coordinates.csv loaded but metadata has no per-FOV XY position | **Folded in:** include `positions {(region,fov):(x_mm,y_mm)}` now (free; prevents 193/188 rework) |
| 10 | Fixture generator is IMA-188 scope; breaks Windows CI; 120k inodes | **→ TODO** scoped to IMA-188; 189 keeps small real-shaped fixtures |
| minor | `camera_settings.'1'` hardcodes a camera index | **Folded in:** read the first camera key generically |
| minor | laziness test has no concrete assertion | **Folded in:** assert `tifffile.imread` call count via monkeypatch |
| minor | pixel_size provenance assumed | **Folded in:** compute from sensor/magnification, documented |

**Cross-model tension surfaced to the user** (User Sovereignty — not auto-applied): findings
#1, #2, and #4 each contradicted or extended a prior decision and were presented as explicit
decisions. #4 (the build reversal) was the strongest and flipped a previously "settled" call.

---

## 7. Final locked architecture

### 7.1 Interface

```
open_reader(path) -> SquidReader              # detects layout, cheap; no pixel I/O

SquidReader.metadata -> {
  regions:            [str],                    # ['B2','B3','B4']
  fovs_per_region:    {region: [int]},          # filename fov tokens, may be sparse
  channels:           [{name, display_name, display_color, ex}],   # name = filename form (canonical key)
  positions:          {(region, fov): (x_mm, y_mm)},   # from coordinates.csv
  n_z:  int, dz_um: float, pixel_size_um: float,
  frame_shape: (Y, X), dtype: np.dtype,         # from one real frame
  n_t:  int,
}

SquidReader.read(region, fov, channel, z, t=0) -> np.ndarray   # (Y,X) uint16, exact
```

### 7.2 Data flow

```
open_reader(path)
      │
      ▼  detect time folders (0/,1/…) ─────────────► n_t
      ▼  glob time-0 *.tiff  (one-time, O(files); cached)
  parse each stem: region _ fov _ z _ channel
      │           └────────────► regions, fovs_per_region, channels(filename form), n_z(range)
      ▼
  read acquisition parameters.json ─► dz_um, pixel_size_um (sensor/mag), Nz(cross-check)
  read coordinates.csv             ─► positions{(region,fov):(x,y)}
  read acquisition_channels.yaml   ─► strict join → display_name, display_color, ex   [decision 2]
      │                                   (absent → colors None; mismatch → raise)
      ▼
  metadata (all cached)

read(region,fov,channel,z,t=0)
      │  validate (region,fov,channel,z,t) against index → clear error if bad
      ▼  construct  <t>/{region}_{fov}_{z}_{channel}.tiff   (.tiff→.tif fallback)
   tifffile.imread → assert 2D uint16 (raise on RGB/non-2D)  [decision 5]
      ▼
   (Y,X) uint16   — one frame only (lazy; bounded memory)
```

### 7.3 Channel join pipeline

```
filenames ──► channel tokens: {'Fluorescence_638_nm_-_Penta', ...}   (canonical keys)
YAML ──► [{name:'Fluorescence 638 nm - Penta', color:'#FF0000', ex:50.0}, ...]
                    │ normalize(name): spaces → '_'   (dash preserved; verified collision-free on the 4 names)
                    ▼
        {'Fluorescence_638_nm_-_Penta': {display_name, color, ex}, ...}
                    │ strict join to filename tokens                       [decision 2]
        ├─ every filename token matched  → channels[] with colors
        ├─ token missing from YAML       → RAISE (names the token)
        └─ YAML file absent entirely     → channels[] with color=None
```

### 7.4 Module layout

```
squidmip/
  __init__.py        # exports open_reader, SquidReader
  reader.py          # open_reader(), SquidReader (index build, metadata, read)
  _channels.py       # normalize(), load_channel_colors(), strict join
  _acquisition.py    # read acquisition parameters.json + coordinates.csv
tests/
  test_reader.py     # AC1,4,5,6 + read() validation + non-2D raise + .tif fallback + t
  test_channels.py   # AC2,3 + normalization round-trip/collision + YAML absent/mismatch
  fixtures/          # tiny real-shaped fixture (few FOVs, real filenames + YAML + params)
pyproject.toml       # deps: numpy, tifffile, pandas, pyyaml  (NO tilefusion)
.github/workflows/ci.yml
```

---

## 8. Acceptance criteria → tests

| AC | Criterion | Test |
|----|-----------|------|
| 1 | Parses real dataset zero-config: 3 regions, 48 FOVs, 4 ch, Nz=3, (4168,4168) uint16 | `test_reader.py` against small real-shaped fixture |
| 2 | Channel names + display colors from YAML (not guessed), joined by normalized name | `test_channels.py`: 638→#FF0000, 561→#FFCF00, 488→#1FFF00, 405→#20ADF8 |
| 3 | Filename parse correct for channel with `_` and `-` (`Fluorescence_638_nm_-_Penta`) | `test_channels.py` normalization round-trip |
| 4 | `read('B3',15,'Fluorescence_638_nm_-_Penta',0)` == `tifffile.imread` of that file (dtype+values) | `test_reader.py` exact-equality |
| 5 | Channel count exactly 4 regardless of Nz (no z-as-channel) | `test_reader.py` |
| 6 | Lazy: reading one plane opens exactly one file | `test_reader.py` monkeypatch `imread` call count |

---

## 9. Test coverage plan

```
[+] _channels.py
  ├── [★★★] load_channel_colors happy: 4 colors by normalized name                        (AC2)
  ├── [★★★] normalize round-trip: 'Fluorescence 638 nm - Penta' → '..._638_nm_-_Penta'     (AC3)
  ├── [★★★] normalize collision guard: distinct YAML names never collapse to one key
  ├── [★★★] nested camera color read (first camera key, not hardcoded '1')
  ├── [★★★] YAML absent → colors None (benign degrade)                                     (decision 2)
  └── [★★★] YAML present, channel missing entry → raises naming the channel                (decision 2)
[+] reader.py
  ├── [★★★] index: 3 regions, fovs_per_region=48, 4 ch, n_z=3                               (AC1)
  ├── [★★★] channels=4 regardless of Nz                                                     (AC5)
  ├── [★★★] fov from filename token, not CSV row order                                      (decision 4)
  ├── [★★★] read exact uint16 == tifffile.imread(file)                                      (AC4)
  ├── [★★★] read raises on non-2D/RGB plane (synthetic (Y,X,3) fixture)                      (decision 5)
  ├── [★★  ] read(...,t=1) reads from 1/ folder (synthetic 2-timepoint fixture)              (decision 6)
  ├── [★★  ] lazy: read one plane → imread called exactly once                               (AC6)
  ├── [★★  ] invalid region/fov/channel/z/t → clear error
  ├── [★★  ] .tif vs .tiff suffix fallback
  └── [★★  ] Nz(params) ≠ max_z+1 → warn, use filename-derived range                         (#7)
[+] .github/workflows/ci.yml
  └── [→E2E] clean-room pip install . + pytest on small fixtures                             (decision 3)

Greenfield — all ship WITH the code (100% of the 6 ACs + every branch above).
```

---

## 10. Failure modes (production realism)

| Codepath | Realistic failure | Test? | Handled? | User sees |
|----------|-------------------|-------|----------|-----------|
| `read()` on RGB/brightfield | imread returns (Y,X,3) → corrupts MIP | yes (dec.5) | yes, raises | clear "non-2D not supported" |
| YAML channel mismatch | wrong/placeholder color on a fluor channel | yes | yes, raises naming channel | clear error, no silent mis-color |
| fov from CSV row order | wrong tile returned silently | yes (dec.4) | yes (parse filename) | correct tile |
| `Nz(params) ≠ files` | z-range disagreement (partial run) | yes | yes, warn + use files | warning, correct range |
| invalid read args | KeyError / cryptic FileNotFound | yes | yes, validate first | clear "no such region/fov/channel/z" |
| ambient tilefusion import | works local, breaks elsewhere | yes (CI) | yes (no tilefusion dep) | reproducible build |

**No failure mode is untested AND unhandled AND silent → 0 critical gaps.**

---

## 11. Implementation tasks (what actually gets built)

| Task | P | Effort (human / CC) | Component | Files |
|------|---|---------------------|-----------|-------|
| T1 | P1 | 2h / 20min | packaging | `pyproject.toml` (numpy, tifffile, pandas, pyyaml; no tilefusion) |
| T2 | P1 | 3h / 25min | channels | `squidmip/_channels.py`, `tests/test_channels.py` |
| T3 | P1 | 3h / 30min | acquisition | `squidmip/_acquisition.py` (params + coords→positions) |
| T4 | P1 | 1d / 40min | reader | `squidmip/reader.py`, `squidmip/__init__.py`, `tests/test_reader.py` |
| T5 | P1 | 2h / 15min | ci | `.github/workflows/ci.yml` |
| T6 | P2 | 1h / 10min | fixtures | `tests/fixtures/` (tiny real-shaped) |

### Parallelization
`Lane A: T2` · `Lane B: T3` · `Lane C: T1+T5` — all independent, run in parallel worktrees.
`T4` joins A+B. `T6` supports A/B/T4.

---

## 12. NOT in scope (deferred, with rationale)

- **OME-zarr output** → IMA-184, which **vendors** tilefusion's writer (`create_zarr_store` / `write_ome` / `colors.py`), not imports it. No cross-repo dependency at any slot.
- **Projection (`np.max` over z)** → IMA-183/188 (this ticket is ingest only).
- **Plate-view UI** → IMA-193 (consumes `positions`, `channels[].display_color`).
- **Scale-test fixture generator (48→1536-well symlink fan-out, ~4 TB logical)** → IMA-188.
- **Brightfield/RGB channel ingest** → deferred; `read()` raises on non-2D.
- **Multi-timepoint iteration/projection** → deferred; only the `read(...,t=0)` hook exists.

## 13. TODOs captured (`TODOS.md`)

1. **Scale-test fixture generator → IMA-188.** Symlink fan-out of 48 real FOVs across a
   1536-well plate (20 z × 4 ch, ~4 TB logical from ~19 GB on disk). Windows-symlink + 120k-inode
   caveats noted. 189 keeps small fixtures.
2. **Brightfield/RGB channel ingest → future ticket.** Linked to the `read()` non-2D raise so
   the limitation reads as deliberate, not a bug.
3. **Multi-timepoint iteration/projection → low-priority follow-up.** The `read(...,t=0)` hook
   already exists, so the extension is small.

---

## 14. Open flag for you

The plan pins **tilefusion enters at IMA-184**. If IMA-184's design assumed IMA-189 would
already carry the tilefusion dependency, that assumption is now void — IMA-184 owns introducing
it. Worth a note on the IMA-184 spec/issue so it isn't a surprise when you get there.

---

## 15. Review metadata

- Step 0 scope: **reduced** (standalone, reversed spec's tilefusion dependency).
- Architecture: 4 decisions locked. Code Quality: 1 correctness fork resolved (fov landmine).
- Test review: coverage diagram produced, all 6 ACs + edges mapped. Performance: 0 issues, 1 note.
- Outside voice: ran (Claude subagent). 13 findings — 1 reversed a decision, 2 became decisions,
  1 dismissed, rest folded in.
- Failure modes: 6 mapped, 0 critical gaps. Lake score: 7/7 complete option chosen.
