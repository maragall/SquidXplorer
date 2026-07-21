# TODOS

Deferred work captured during plan-eng-reviews. Each item records the reasoning
so a future session doesn't rediscover it from zero.

## Deselect-all gesture (Esc) → fast-follow after IMA-221
- **What:** Esc (or click-on-empty-space) clears the whole plate selection and emits the cleared state.
- **Why:** After IMA-221, Shift+click toggle is the only removal gesture, so clearing a 200-well selection means 200 clicks or a throwaway marquee over an empty corner. There is no defined way back to nothing.
- **Pros:** Removes a real interaction dead end; one small handler on state that already exists.
- **Cons:** Adds a keyboard path to a currently mouse-only widget.
- **Context:** `PlateOverview` has **no `keyPressEvent` and never sets a focus policy** — this is the non-obvious part. Esc needs `setFocusPolicy(Qt.StrongFocus)` plus a handler, or the key silently does nothing and looks like a broken feature. Held out of IMA-221 so the AFK iteration stayed one concern.
- **Depends on / blocked by:** IMA-221's `_selection` state.

## Selection count readout → fast-follow after IMA-221
- **What:** Show "N wells / M FOVs selected" in the existing status readout while a selection is active.
- **Why:** IMA-221 emits on release, so the operator gets tinted cells but no number. On a 1536-well plate a selection is not countable by eye, which keeps Accept-gate verification qualitative.
- **Pros:** Turns visual impression into a checkable number; reuses the existing `_readout`.
- **Cons:** Needs a precedence rule or it fights the other writers.
- **Context:** `self._readout.setText(...)` already has **three writers** — hover, ingest status, and operator progress (`_viewer.py:1589`, `:1786`). A fourth writer without an ordering rule produces flicker. That contention question is the real work here, not the string formatting. Was option C on the IMA-221 Accept-gate decision; not chosen.
- **Depends on / blocked by:** IMA-221's `selectionChanged` signal.

## Per-FOV sub-cell selection → blocked on FOV geometry
- **What:** True per-FOV marquee: subdivide each well cell into its FOV sub-grid and hit-test within it.
- **Why:** This is the half of IMA-221's original acceptance ("marquee selects a set of FOVs") that was **de-scoped as unimplementable**. Recording it so the de-scope does not silently become permanent.
- **Pros:** Delivers the original spec intent. IMA-221's payload was deliberately shaped as `(region, fov)` pairs so **no consumer changes** when this lands.
- **Cons:** Genuinely blocked, and it invalidates `cells_in_rect`'s one-cell-per-well model once cells subdivide.
- **Context:** `reader.py:197` builds `fovs_per_region` from parsed filenames as **ids only — no x/y positions**. The learning `fov-axis-needs-geometry` (8/10, cross-model, 2026-07-04) already flagged per-FOV position + overlap as the load-bearing missing piece. Sibling of the FOV-composition/stitch TODO — they should land together.
- **Depends on / blocked by:** per-FOV geometry (position + overlap) reaching reader metadata, i.e. the deferred stitch work.

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

## Tile render loop + async fetch executor are unowned → IMA-218 scope check
- **What:** Before/during IMA-218, confirm its scope includes (a) rewriting `PlateOverview.paintEvent`/pan/zoom to consume tiles from `TileCache.resolve()` instead of the single full-plate QPixmap blit, and (b) the async fetch executor (thread/queue) that drives `mark_pending`/`insert`/`fetch_failed` on the cache.
- **Why:** IMA-216's eng review (outside voice #2, 2026-07-20) found the ticket graph has no owner for either: 216 is a pure library, 217 is a synchronous `read_tile`, and 218's spec says "place FOVs at stage coords" — not "rewrite the render loop". Without an owner, 216 lands as a library with no caller.
- **Pros:** Catches an unscoped rewrite before 218 is estimated; the executor design decides whether keep-parent-until-child-ready ever actually fires.
- **Cons:** Likely grows 218 or forces a new ticket; can't be closed until 218 is picked up.
- **Context:** `_tiling.py` deliberately owns NO concurrency (caller-driven lifecycle, eng review decision). Today's render path: `_viewer.py` paintEvent blits one cached scaled QPixmap; wheel/pan call `update()` per event. The tile path replaces that blit for the mosaic case; N=1 FOV fallback stays.
- **Depends on / blocked by:** IMA-216 (contract), IMA-217 (TileSource impl); belongs to IMA-218.

## Fix upstream squid2minerva/colors.py display_color nesting → external repo
- **What:** `squid2minerva/colors.py:load_yaml_colors` reads `channel["display_color"]`, but real `acquisition_channels.yaml` nests it under `channel.camera_settings.<cam>.display_color`. Its Minerva OME-TIFF exports only get right colors via the wavelength-fallback map — a custom yaml color is silently ignored.
- **Why:** Confirmed against a real dataset yaml. It's correct-by-luck today because the fallback palette matches the standard 4 channels; any non-default color drops silently.
- **Pros:** Fixes silently-wrong colors in a sibling tool's exports.
- **Cons:** Different repo, different owner; not on any SquidMIP critical path.
- **Context:** SquidMIP does **not** carry this bug — IMA-189's `squidmip/_channels.py` already resolves `display_color` correctly (top-level v1.0+ *and* nested `camera_settings`, mapped by name, raises on unresolved), and IMA-184 consumes `metadata.channels[].display_color` rather than re-parsing the yaml. This TODO is purely a flag for whoever owns `~/CEPHLA/projects/explorer/squid2minerva`.
- **Depends on / blocked by:** squid2minerva maintainer.

## Evaluate `original_coordinates/` as the geometry source → IMA-219 follow-up
- **What:** Both real acquisitions (`~/Downloads/20x_scan_2025-09-05_17-57-50`, `~/Downloads/synthetic_2x2_wellplate`) ship an `original_coordinates/` directory alongside `coordinates.csv` — planned vs. actual stage positions. Determine whether it is a cleaner input for plate-shape inference than the as-executed `coordinates.csv`.
- **Why:** IMA-219 infers well pitch from `coordinates.csv`, which records *actual* per-FOV stage positions and therefore carries autofocus drift, backlash, and any operator intervention. Planned coordinates would be exactly on the nominal grid, making pitch matching tighter and the tolerance smaller.
- **Pros:** Potentially removes the need for a ~5% pitch tolerance; a planned grid is noise-free by construction.
- **Cons:** Unknown format and unknown guarantee of presence — neither the reader nor any test touches it today. Adding a second geometry source doubles the "which file won" debugging surface.
- **Context:** Surfaced by the `/plan-eng-review` outside voice on 2026-07-20 and confirmed present in both datasets. IMA-219 measured `coordinates.csv` centroid pitch at exactly 9.000mm on `synthetic_2x2_wellplate`, so the as-executed file is already good enough — this is an optimization, not a fix. Start by diffing `original_coordinates/` against `coordinates.csv` on both datasets to see whether they differ at all.
- **Depends on / blocked by:** IMA-219 inference landing first, so there is something to compare against.

## Cross-check a declared wellplate_format against inferred geometry → future ticket
- **What:** When `acquisition.yaml` declares a format AND `coordinates.csv` is present, compute the inferred format anyway and warn on disagreement.
- **Why:** IMA-219 D1 trusts the declared field unconditionally when present. A stale or hand-edited yaml therefore renders the wrong plate silently, and the ticket's own inference machinery would have caught it. Precedent exists: `reader.py:180-188` already warns when declared Nz/Nt disagrees with the filenames.
- **Pros:** Reuses inference already built by IMA-219 for near-zero extra cost; catches a silent-wrong-plate class that nothing detects today.
- **Cons:** Disagreement is *expected* on sparse acquisitions (four wells of a 384-plate legitimately span 2x2), so a naive warning becomes noise. Needs a confidence threshold before it can be enabled.
- **Context:** Deliberately deferred during the 2026-07-20 eng review to keep IMA-219 to a fallback-only trigger. `sim_1536wp` in `~/CEPHLA/Data` is a live example of the hazard — its `coordinates.csv` spacing does not correspond to its declared "24 well plate".
- **Depends on / blocked by:** IMA-219 inference + its confidence score.
