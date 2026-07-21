# TODOS

Deferred work captured during plan-eng-reviews. Each item records the reasoning
so a future session doesn't rediscover it from zero.

## Scale-test fixture generator → IMA-188
- **What:** A generator that fans the 48 real hongquan FOVs across a 1536-well plate via **symlinks** (Squid layout), synthesizing 20 z (cycling the real 3) × 4 channels. On-disk ≈ source (~19 GB); logical read ≈ 1536×20×4×33 MB ≈ **4 TB** (served from OS cache — proves scale/parse/decode/memory, NOT raw disk bandwidth; that needs Nick's real storage).
- **Why:** It's the harness for the IMA-188 high-throughput scale test, not ingest. Building it in 189 bloats the keystone and risks CI breakage.
- **Pros:** Proves the reader + projection hold at plate scale with bounded memory, cheaply (symlinks, not 4 TB of real bytes).
- **Cons:** Breaks on Windows CI runners (no symlink checkout); ~120k inodes; slow to materialize.
- **Context:** The IMA-189 `SquidReader` reads one plane per call, so a symlink fan-out exercises the exact read path at scale. Keep 189's own tests on small real-shaped fixtures.
- **Depends on / blocked by:** IMA-189 reader (landed); belongs to **IMA-188 (this slot)**.

## Resume / checkpoint for long plate runs → fast-follow after IMA-184
- **What:** Skip wells whose complete output already exists; clean partial output files on rerun.
- **Why:** A full 1536wp run takes minutes-to-hours; a crash mid-run currently restarts from 0, and partial outputs can silently corrupt the plate.
- **Pros:** Turns a full-run loss into an incremental retry; mitigates the threads segfault residual (a rerun skips finished wells).
- **Cons:** Needs a per-well "complete output" definition + atomic write/rename or cleanup logic.
- **Context:** IMA-188 engine uses ThreadPoolExecutor; failure policy = per-well manifest. A C-level segfault in decode can still abort the whole process. Surfaced by /plan-eng-review outside-voice #7 (2026-07-04).
- **Depends on:** IMA-184 output layout (what a "complete well output" looks like).

## Brightfield / RGB channel ingest → future ticket
- **What:** Support Squid brightfield channels saved as `(H,W,3)` RGB (and per-LED `_B/_G/_R`) planes, with a defined reduction-to-2D (or explicit color) policy.
- **Why:** IMA-189 `read()` deliberately **raises** on non-2D planes (decision 5). Without this note the raise reads like a bug.
- **Pros:** Broadens input coverage to brightfield acquisitions.
- **Cons:** Requires an RGB→2D policy the MIP tool may never need; better decided when brightfield is actually in scope.
- **Context:** Linked to the `read()` non-2D assertion in `squidmip/reader.py`. tilefusion's `_to_grayscale_2d` is a reference implementation if reduction is chosen.
- **Depends on / blocked by:** IMA-189 reader.

## Multi-timepoint iteration / projection → low priority follow-up
- **What:** Iterate or project across timepoints (Nt>1) beyond the single `read(...,t=0)` hook.
- **Why:** No current dataset has Nt>1; 189 makes the API honest (`read(...,t=0)` + `metadata.n_t`) without building traversal.
- **Pros:** Ready for time-lapse acquisitions when they appear.
- **Cons:** Ahead of demand; the MIP tool projects over z, not t.
- **Context:** The `t=0` param + time-folder discovery already exist, so the extension is small.
- **Depends on / blocked by:** A real Nt>1 acquisition.

## Confirm IMA-193 navigator reads the pyramid + plate/well metadata → IMA-193
- **What:** Before/during IMA-193, verify its plate-view navigator actually reads multi-level pyramids and OME-NGFF plate/well group metadata — not just full-res array `0` the way ndviewer_light does.
- **Why:** IMA-184 writes a ≥2-level pyramid + spec plate/well metadata. ndviewer_light (today's only reader) ignores both — it directory-walks and reads only `field/0` + `omero`. So the pyramid is currently invisible; IMA-193 is the consumer that justifies it. If IMA-193 also reads only level 0, that extra output delivered nothing.
- **Pros:** Validates the load-bearing assumption behind IMA-184's canonical/multiscale scope before more work rides on it.
- **Cons:** Can't be closed until IMA-193 is designed; until then the pyramid is written on faith.
- **Context:** ndviewer_light discovers plates by directory walk and reads array `0` + `omero` only (`ndviewer_light/core.py:1149`, `:1070`). IMA-184's cross commit already proves the plate opens under strict `ome-zarr-py`, so the metadata is spec-valid regardless.
- **Depends on / blocked by:** IMA-193 design.

## Fix upstream squid2minerva/colors.py display_color nesting → external repo
- **What:** `squid2minerva/colors.py:load_yaml_colors` reads `channel["display_color"]`, but real `acquisition_channels.yaml` nests it under `channel.camera_settings.<cam>.display_color`. Its Minerva OME-TIFF exports only get right colors via the wavelength-fallback map — a custom yaml color is silently ignored.
- **Why:** Confirmed against a real dataset yaml. It's correct-by-luck today because the fallback palette matches the standard 4 channels; any non-default color drops silently.
- **Pros:** Fixes silently-wrong colors in a sibling tool's exports.
- **Cons:** Different repo, different owner; not on any SquidMIP critical path.
- **Context:** SquidMIP does **not** carry this bug — IMA-189's `squidmip/_channels.py` already resolves `display_color` correctly (top-level v1.0+ *and* nested `camera_settings`, mapped by name, raises on unresolved), and IMA-184 consumes `metadata.channels[].display_color` rather than re-parsing the yaml. This TODO is purely a flag for whoever owns `~/CEPHLA/projects/explorer/squid2minerva`.
- **Depends on / blocked by:** squid2minerva maintainer.

## External stitchers as the ACCEPTANCE ORACLE (ASHLAR first) → IMA-211 (re-scoped to exactly this)
- **What:** Run ASHLAR out-of-process on a dev box over 3-5 real wells and compare its per-tile
  displacements against SquidMIP's own stitch output (IMA-222's operator). Require max disagreement ≤ 2 px. Optionally
  extend to BigStitcher/PetaKit5D later; ASHLAR alone carries most of the signal.
- **Why:** IMA-211 asked to *prototype/evaluate* four stitchers. The review's first answer —
  "none is shippable inside our Windows app" — is true but answers a different question.
  Evaluation does not require shipping, and this is the **acceptance oracle the ticket otherwise
  lacks**: build our own stitcher and never run ASHLAR once, and we have no way to know ours is
  any good.
- **Pros:** Turns a discarded scope item into the missing quality gate. Independent implementation
  disagreement is the strongest correctness signal available without ground truth.
- **Cons:** Needs a dev box with conda + a JVM (ASHLAR boots one via pyjnius at import). It is a
  lab harness, never packaged or shipped.
- **Context (verified 2026-07-20, do not re-research):**
  - **MCmicro is NOT a stitcher** — it is a Nextflow orchestrator that shells out to ASHLAR.
    Drop it from any bake-off; it would be measuring ASHLAR twice.
  - **ASHLAR** — the only importable Python option, and the only one with a native plate/well
    concept (`--plates`, `PlateReader`, `plate=`/`well=` kwargs on `FileSeriesReader`). Derives
    tile positions from a filename pattern + grid + overlap, no CSV/XML needed. **But
    `ashlar/reg.py` boots a JVM via pyjnius at import time regardless of which reader you use**,
    so it carries a Java prerequisite even for pure-TIFF input. Actively maintained (v1.20.0).
  - **BigStitcher** — Java/Fiji; no Python API (macro or BigStitcher-Spark subprocess only;
    PyImageJ is known-flaky for it). Requires a mandatory BigDataViewer XML+HDF5/N5 **resave**
    before it can read anything. No plate/well concept. Repo moved to JaneliaSciComp.
  - **PetaKit5D** — MATLAB; `PyPetaKit5D` wraps the MATLAB Compiler Runtime, is **Linux-only**,
    and defaults to SLURM. Needs a hand-built `ImageList_*.csv` with an `axisOrder` string. Its
    whole parameter surface assumes 3D light-sheet. Worst fit of the four for 2D plate wells.
- **Depends on / blocked by:** IMA-222's stitch operator (the baseline to compare against). NOT
  blocked on Nick's laser-AF fix — real multi-FOV acquisitions already exist on disk
  (`~/Downloads/20x_scan_2025-09-05_17-57-50` is 36 FOV/well at ~9% overlap).

## Reconcile IMA-187's FOV keying with `original_coordinates_{t}.csv` → raise before IMA-187 merges
- **What:** IMA-187 locked FOV keying as "row-order-per-region with a loud cross-check" against
  `coordinates.csv`. Squid also writes `original_coordinates/original_coordinates_{t}.csv` =
  `region,fov,z_level,x (mm),y (mm),z (um),time` — an explicit `fov` key, and the *actual* stage
  position rather than the *planned* grid. Switch the mapping to that file.
- **Why:** Row-order mapping is the "silent wrong tile" hazard `docs/ima-189-eng-review.md`
  already refused once. With an explicit key, cell assignment is arithmetic, not inference.
- **Pros:** Removes a whole risk class from a locked ticket. Also dissolves IMA-187's own recorded
  open worry ("whether coordinates.csv carries one row per z-level on multi-z acquisitions, which
  would break the row-count cross-check") — `original_coordinates` carries `z_level` explicitly.
- **Cons:** Needs a fallback for acquisitions lacking the `original_coordinates/` folder; nobody
  has surveyed how common that is.
- **Context:** Verified by hand 2026-07-20 on `~/Downloads/20x_scan_2025-09-05_17-57-50`. Same
  dataset shows `acquisition parameters.json` reporting `Nx=1,Ny=1,dx=0.9` for a region that
  actually holds 36 FOVs in a 6x6 grid at 0.7056 mm — so never derive grid geometry from that JSON;
  derive spacing from the coordinates themselves.
- **Depends on / blocked by:** TIME-SENSITIVE — only useful before `juliomaragall/ima-187-multi-fov-mosaic` merges.

## Measure the real overlap fraction across all acquisitions → gates IMA-222 registration
- **What:** Compute the actual tile overlap for every acquisition on disk, per objective/binning.
- **Why:** If any real acquisition runs 0% overlap, phase correlation is dead on arrival there and
  only nominal placement works. This is cheap and it gates a real design decision in IMA-222.
- **Pros:** Hours of work; decides whether registration code is worth writing at all.
- **Cons:** None material.
- **Context:** Measured ~9% on the 20x scan (2084 px frame, 0.7056 mm step). Sample size of one.
- **Depends on / blocked by:** nothing — runs today.
