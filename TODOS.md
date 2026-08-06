# TODOS

Deferred work captured during plan-eng-reviews. Each item records the reasoning
so a future session doesn't rediscover it from zero.

## Delete the incomplete-input tests → requested by Julio on `main-window-review`
- **What:** Remove SquidXplorer's tests that assert behaviour for malformed or incomplete
  acquisition input. Julio's call, 2026-07-28, on Spencer's `main-window-review` PR; Spencer
  asked for it to be written down here rather than only in Slack.
- **Why:** `Cephla-Lab/Squid` is the producer and already validates what it writes. Asserting the
  same guarantees on the consumer side duplicates a contract we do not own, and it means a
  change to Squid's validation silently makes our tests wrong rather than red.
- **Pros:** Drops a duplicated contract; the tests that remain are about *our* behaviour.
- **Cons:** If a malformed acquisition ever reaches us anyway (hand-edited folder, interrupted
  copy, a dangling symlink farm like the 1536wp case), nothing pins how we degrade. Decide
  deliberately whether "we refuse loudly" is itself a guarantee worth one test.
- **Context:** Scope is not yet enumerated. Identify which tests assert *producer* guarantees
  (missing `acquisition.yaml`, malformed `coordinates.csv`, absent channels) versus which assert
  *our* refusal behaviour, e.g. `reader.py:329-336`, which deliberately raises rather than
  placing FOVs at plausible-but-wrong positions. The second kind is ours and stays.
- **Depends on / blocked by:** Nothing.
- **SCOPED 2026-07-29, and the answer is that there is almost nothing to delete.** Enumerated
  every candidate: `test_the_reader_boundary_refuses_a_malformed_acquisition`
  (`tests/test_acquisition_model.py:200`), `test_a_malformed_file_is_still_all_or_nothing`
  (`tests/test_fov_positions.py:322`), `test_malformed_coordinates_csv_header_still_yields_metadata`,
  `test_monkey_malformed_csv_still_yields_metadata`, `test_an_incomplete_region_refuses_to_produce_a_result`,
  `test_a_failed_field_publishes_nothing_and_marks_the_plate_incomplete`,
  `test_open_computed_refuses_an_incomplete_plate`. Every one asserts OUR degradation or refusal
  behaviour, not a producer guarantee: that a bad field dies AT the reader naming the field
  rather than downstream "pointing at the victim rather than the cause", that salvage is
  per-region and all-or-nothing, that our own writer marks a partial plate. Those are the second
  kind named above, the kind that stays.
- **Evidence they earn their keep:** on 2026-07-28 `sim_1536wp` turned out to be 6144 dangling
  symlinks, and ten failing tests were classified in minutes precisely because `open_reader`
  refuses loudly instead of half-reading. That is the behaviour these tests pin.
- **Remaining question for Spencer:** if there is a specific test he had in mind that asserts a
  guarantee `Cephla-Lab/Squid` owns, name it and it goes. The blanket deletion would cost more
  than it saves.

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
## Gesture arbitration on PlateOverview (shift-select vs pan vs loupe) → IMA-221
- **What:** A single explicit gesture policy for `PlateOverview`'s mouse handlers, deciding between drag-to-pan (shipped), shift-drag marquee select (IMA-221), and press-and-hold loupe (IMA-208).
- **Why:** Three tickets independently add a gesture to the SAME handlers (`mousePressEvent:647`, `mouseMoveEvent:652`, `mouseReleaseEvent:670`). Whoever lands second inherits an undocumented conflict; the pan path already claims plain left-drag with a 3px threshold (:655).
- **Pros:** One place decides what a drag means; each later gesture ticket becomes additive instead of a rewrite of someone else's branch.
- **Cons:** Slightly more design up front than "add a modifier check"; needs agreement across three backlog tickets.
- **Context:** Today `mousePressEvent` unconditionally arms pan state on LeftButton. Shift-drag must branch BEFORE that. The loupe (IMA-208) wants press-and-hold, which competes with the same 3px pan threshold on the time axis rather than the modifier axis — so a modifier check alone won't settle it. `_sel` is currently a single `(ri,ci)` (:548) painted as one red box (:752-756); marquee select needs it to become a set with a multi-cell paint path.
- **Depends on / blocked by:** IMA-221 owns the selection gesture; coordinate with IMA-208 before either lands.
- **Status (IMA-205 rebase):** the MODIFIER axis is settled — Shift owns selection (`mousePressEvent` branches before pan arms), Shift+drag opens an independent window over the boxed regions (`marqueeSelected` -> `_on_marquee_selected`; it opened an exploration tab until that pane was removed), Shift+click stays a refine-one-well toggle that deliberately opens nothing. The TIME axis is still open: IMA-208's press-and-hold loupe competes with the plain-drag 3px pan threshold, not with a modifier, so it still needs a policy.

## Exploration-tab persistence across acquisitions → CLOSED (feature removed)
- **CLOSED 2026-08-05:** the exploration pane was removed. There are no exploration tabs to persist. A view is an independent window now, and
whether WINDOWS survive a re-ingest is a different question against `ViewerManager`.
- **What:** Decide whether exploration tabs survive re-ingesting a different acquisition, and if so how their region sets are revalidated.
- **Why:** `ingest()` (:1493-1505) resets reader/`_fov_index`/`_overview` but never `_op_tabs`. The eng-review fix closes exploration tabs on ingest (the safe default), but the richer behavior — reopen the same selection on a re-ingest of the SAME acquisition — is a real workflow for anyone iterating on one plate.
- **Pros:** Users re-open the same plate constantly while tuning operators; losing their exploration set every time is friction.
- **Cons:** Requires acquisition identity in the tab key plus revalidation that every region still exists in the new `_fov_index`.
- **Context:** Surfaced by the /plan-eng-review outside voice (2026-07-20) and confirmed in code. The eng-review decision includes acquisition id in the content-addressed tab key specifically so this extension stays cheap.
- **Depends on / blocked by:** IMA-205 landing its ingest-teardown fix first.

## Partial `.hcs` cleanup after a stopped save run → fast-follow after IMA-205
- **What:** Clean up (or mark as partial) the `.hcs` output directory when a `save=True` operator run is stopped mid-flight.
- **Why:** stopping is routine (a stopped run, a re-ingest, app close). A stopped save leaves a partial plate that `resolve_plate_root` will later happily recognize as a real plate, so the user can re-open a half-written result as if it were complete.
- **Pros:** Prevents silently trusting a truncated plate; complements the resume/checkpoint TODO already filed against IMA-184.
- **Cons:** Needs a "complete output" definition — the same definition the resume/checkpoint item needs, so the two should be designed together.
- **Context:** Before IMA-205, stopping only happened on app close (`closeEvent:1970`) or re-ingest, both of which end the session anyway. Making close-tab stop a run turns a rare path into a common one. Surfaced by the /plan-eng-review outside voice (2026-07-20).
- **Depends on / blocked by:** Overlaps the existing "Resume / checkpoint for long plate runs" TODO — resolve as one design.
## Writer cannot express "pixel size unknown" → future ticket
- **What:** `_output.py:173` does `p = float(pixel_size_um) if pixel_size_um else 1.0`, so a written plate records 1.0 for both "unknown" and "genuinely 1.0 µm/px". Give the writer a way to mark unknown (omit the scale, or record a sentinel/attribute), and teach readers to distinguish.
- **Why:** IMA-208's loupe draws a µm scale bar. On the computed-plate path the only pixel-size source is the multiscales `scale`, so the loupe cannot tell an unknown from a real 1.0 and must suppress the bar for BOTH. That is the honest behaviour but it silently degrades a legitimately-1.0µm acquisition.
- **Pros:** The loupe (and any future measurement UI) can show microns whenever they are actually known.
- **Cons:** Touches the writer, so already-written plates in the field stay ambiguous forever regardless; needs a migration story or a "legacy plates are ambiguous" acceptance.
- **Context:** Surfaced by both outside-voice passes during the IMA-208 eng review (2026-07-20) and verified. `_open_computed` (`_viewer.py:1683`) also builds `_meta` with no pixel size at all — IMA-208 adds the parse, but it can only read what the writer wrote.
- **Depends on / blocked by:** Nothing; independent of IMA-208, which works around it.

## `_open_computed` reuses well 0's FOV path for every well → future ticket
- **What:** `_viewer.py:1665` reads `fov0` from `wells_meta[0]` and `:1692` applies that same path to every well in `worker_wells`. A plate whose wells carry differing image ids silently renders the wrong image for those wells.
- **Why:** Latent silent-wrong-image bug in the computed-plate open path, independent of the loupe. IMA-208's D5 FOV helper covers the loupe's use of it, but the tile-loading path underneath stays wrong.
- **Pros:** Removes a whole class of silently-wrong renders; needed before multi-FOV plates are opened from disk.
- **Cons:** Requires per-well FOV resolution in `_open_computed` and a fixture with heterogeneous well image ids to test it.
- **Context:** Found by outside voice during the IMA-208 eng review (2026-07-20), verified by reading `_open_computed`. No current dataset triggers it — every well is written with the same fov id today — so it is latent rather than active.
- **Depends on / blocked by:** Overlaps viewer-side multi-FOV (IMA-187); worth doing together.

## Loupe neighbour prefetch on well crossing → fast-follow after IMA-208
- **What:** Prefetch the adjacent well's crop level while the loupe is held, so crossing a well boundary mid-hold is seamless.
- **Why:** IMA-208 deliberately scoped this out: a hold gesture usually stays within one well, and crop reads are already only a few MB, so the latency may never be noticeable.
- **Pros:** Removes the one visible hitch in the loupe's interaction.
- **Cons:** Speculative I/O and more cache surface for a case that may not matter; adds eviction pressure to a cache IMA-208 deliberately kept small.
- **Context:** IMA-208 D10 chose windowed crop reads (`_CHUNK_YX = 1024`, `_zarr_store.py:25`) over a whole-well cache. Revisit only if real use shows a hitch at well boundaries.
- **Depends on / blocked by:** IMA-208 landing and being used on a real plate.

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
- **Update (2026-07-27):** "the extension is small" is true of the READ path and false of the
  IDENTITY path. Split the item in three before estimating it.
  - **Navigation (cheap).** napari is natively ND and already puts z on its dims slider; handing
    it `(T, Z, Y, X)` plus axis labels gets a time slider with no new widget. `_mosaic_source`
    already threads `t` through `fuse_region_mosaic`/`_fuse_levels`, and its plane cache key is
    already `(token, region, channel, int(t), step, z)` — **`t` is in that key today**. `_agave.py`
    is further along still (`n_timepoints()`, `set_time(t)`, `volume_key(..., t=)`), and
    `_minerva` already labels exports `(t=N of M)` when `n_t > 1`.
  - **Reduction (free).** IMA-210's model generalises to time without a change: `consumes` is a
    frozenset, the output shape is already `(T, C, Z, Y, X)`, and a collapsed axis deliberately
    stays at size 1 rather than being dropped. So `consumes={"t"}` — max/mean over time — is
    expressible as written. Two caveats before trusting it: `add_projector` defaults to
    `consumes={"z"}`, and `_engine.py:219` documents the return as "`frozenset()` or `{"z"}`",
    so sweep for branches that compare `== {"z"}` instead of testing membership the way
    `_benchmark.py:524` does.
  - **Identity (the actual work, and a trap).** `ResultCache`'s key is
    `(scope, version, chain.key())`. `scope` comes from `_plate.cache_scope` and is the integer
    `RRCCOOOO` node id — row, column, ROI, **no `t`**. `version` is the ACQUISITION version,
    bumped as a live node's frames arrive. Its docstring calls that "the temporal dimension",
    which is a DIFFERENT temporal axis from experiment `t`: one is how much data has arrived,
    the other is which timepoint you are looking at. Folding `t` into `version` therefore looks
    natural and is a category error — `t=1` would read as "a newer version of `t=0`", so
    switching timepoints would silently evict and recompute forever, and two windows on
    different timepoints would contend for one entry. `t` belongs in `scope` (a fifth field in
    the node id) or as a fourth key component. Decide this BEFORE any Nt>1 work, because the
    node-id format is load-bearing for the cache, the logger ids and the window tree.
  - **Product question that gates the rest:** one window per timepoint (fits the window tree,
    but the navigator explodes at N timepoints x M wells), or one window with `t` on a slider
    (matches napari and the z precedent)? Reduction-over-time is orthogonal to both.
  - **Already time-safe, do not re-litigate:** the `coordinates.csv` de-duplication on
    `(region, x, y)` was written for exactly this — its docstring names "a multi-z or
    multi-timepoint acquisition" as the reason raw row counts cannot be trusted.

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

## 16-bit export + in-browser contrast windowing → follow-up after IMA-206
- **What:** Export per-channel data at 16 bits (PNG-16 or a small binary sidecar) and window it in
  the generated HTML on a `<canvas>`, so contrast means the same thing in the shared artifact as
  it does in the Qt viewer.
- **Why:** P1 for IMA-206 says "one PNG per channel; toggle **and contrast adjustment**". IMA-206
  ships toggles only in HTML (decision D9), because an 8-bit PNG has already crushed the shadows —
  re-stretching it in a browser cannot bring detail back, and a control that silently degrades the
  image is worse than no control in a scientific tool. So half of P1's sentence is deferred, not met.
- **Pros:** The shared artifact stops being a second-class citizen; a collaborator without the tool
  installed can do real contrast work. Keeps the exported HTML honest instead of limited.
- **Cons:** Much bigger export files (16-bit, and one per channel); real JS canvas work; the montage
  module is deliberately dependency-free and this pushes on that.
- **Context:** `_montage.py:340` already holds the per-channel `(C, H, W)` float canvas before it
  composites at `:371-382`, so the data exists at export time — what is missing is a 16-bit encoder
  and the browser-side windowing. The sidecar already records the per-channel window
  (`_montage.py:395-397`), which is what IMA-206's HTML displays instead. Surfaced by the
  /plan-eng-review outside voice (2026-07-20), which read D9 as possibly violating the P1 it quotes.
- **Depends on / blocked by:** IMA-206 landing the per-channel export first. Worth a stakeholder
  check with Nick before building — it may be that toggles are all anyone wanted.

## Grayscale rendering for a single visible channel → taste call, needs a user
- **What:** When exactly one channel is visible, render it grayscale instead of tinted with its LUT
  color; probably a user preference toggle rather than automatic behavior.
- **Why:** Many microscopists inspect single channels in grayscale — a red-tinted 638 image throws
  away perceived dynamic range compared to the same data in grey. IMA-206 keeps LUT color (decision
  OV9) because that matches the acceptance oracle and the existing compositing machinery.
- **Pros:** Matches how the target users actually look at single-channel data; better perceived
  contrast for exactly the inspection task the toggle exists to serve.
- **Cons:** Contradicts the "colors match the resolved display_color" oracle unless scoped as an
  explicit display mode; adds a second rendering rule to a path this ticket just unified.
- **Context:** With IMA-206's `composite(store, colors, windows, mask)` this is a small change —
  pass an identity/white color when `mask.sum() == 1` and a grayscale preference is set. Do NOT
  guess the default: ask Nick and You Yan which they expect, since it is a domain habit, not an
  engineering choice. Surfaced by the /plan-eng-review outside voice (2026-07-20).
- **Depends on / blocked by:** IMA-206's `composite()` seam; a user answer on the default.

## Fix upstream squid2minerva/colors.py display_color nesting → external repo
- **What:** `squid2minerva/colors.py:load_yaml_colors` reads `channel["display_color"]`, but real `acquisition_channels.yaml` nests it under `channel.camera_settings.<cam>.display_color`. Its Minerva OME-TIFF exports only get right colors via the wavelength-fallback map — a custom yaml color is silently ignored.
- **Why:** Confirmed against a real dataset yaml. It's correct-by-luck today because the fallback palette matches the standard 4 channels; any non-default color drops silently.
- **Pros:** Fixes silently-wrong colors in a sibling tool's exports.
- **Cons:** Different repo, different owner; not on any SquidMIP critical path.
- **Context:** SquidMIP does **not** carry this bug — IMA-189's `squidmip/_channels.py` already resolves `display_color` correctly (top-level v1.0+ *and* nested `camera_settings`, mapped by name, raises on unresolved), and IMA-184 consumes `metadata.channels[].display_color` rather than re-parsing the yaml. This TODO is purely a flag for whoever owns `~/CEPHLA/projects/explorer/squid2minerva`.
- **Depends on / blocked by:** squid2minerva maintainer.
- **Update (IMA-228 eng review, 2026-07-20):** this matters more than it looked. `squid2minerva/story.py`'s own docstring confirms Minerva Author ignores OME-TIFF channel colours and takes colour **only** from the `.story.json` groups — which are built from `colors.load_yaml_colors`. So this bug is the *entire* colour path for the salesperson tool's exports, not a secondary one. IMA-228 sidesteps it by feeding `metadata.channels[].display_color` straight into the story groups.

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

## ImageJ-style drag-BACK re-attach for floating tabs → follow-up after IMA-209
- **What:** Drag a floating tab window back onto the `_left_tabs` bar to re-dock it (the symmetric half of the IMA-209 detach gesture). IMA-209 ships a Re-dock button instead.
- **Why:** "ImageJ-style" most plausibly implies the symmetric gesture; the button covers the round trip but isn't the gesture Nick may picture.
- **Pros:** Full ImageJ parity; no chrome in the float.
- **Cons:** The expensive half of custom tab dragging — drop-target hit-testing, insertion-index calculation, cross-window drag state — none of it testable offscreen; roughly triples the untested gesture surface.
- **Context:** IMA-209 eng review D6 (2026-07-20). All re-dock LOGIC already exists as `_redock(key)`; only the gesture would be new. Wait for Nick to actually miss the drag before paying for it.
- **Depends on / blocked by:** IMA-209 landed; real user demand.

## IMA-205 exploration tab: verify it registers via _open_op_tab → CLOSED (feature removed)
- **CLOSED 2026-08-05:** the exploration pane was removed. It did register through `_open_op_tab` and inherited detach/float/re-dock; the
constraint is worth keeping for any NEW tab, and it is stated at `_open_op_tab` itself.
- **What:** When the exploration pane lands, open it through `_open_op_tab(key, title, builder)` so it inherits detach/float/re-dock for free.
- **Why:** IMA-209 made detach a property of the tab container (eng review D2); any tab bypassing the registry won't detach and won't be cleaned up by `_dispose_tab_widget`.
- **Pros:** Zero extra work in 205 to get the Nick float behavior.
- **Cons:** None — this is a one-line integration constraint.
- **Context:** Registry + cleanup contract in `squidmip/_viewer.py` (`_op_tabs`/`_floating`/`_dispose_tab_widget`).
- **Depends on / blocked by:** IMA-205 design.
## Viewport tiling / LOD for deep zoom → region-window ticket
- **What:** `viewport(bbox, zoom) -> tiles` with level-of-detail selection and frustum culling, so a single well can be zoomed into without fetching the whole mosaic.
- **Why:** Cut from IMA-187 during /plan-eng-review (2026-07-21). The plate overview does not need it: `PlateOverview` is a fixed-resolution bitmap montage (`_viewer.py:533` allocates `QImage(nc*88, nr*88)`, `:588` blits 88x88 blocks, `:697-701` smooth-scales the one bitmap), so a 36-FOV mosaic occupies 88 px at every zoom and there is nothing to cull.
- **Pros:** Unlocks true deep-zoom navigation into a well, which is what a region window is for.
- **Cons:** The IMA-187 draft specified it as "framework-neutral (pyqtgraph now, Viv later)" — but pyqtgraph appears nowhere in the repo or `pyproject.toml`, and Viv is an explicit non-goal. Designing it before a real consumer exists means guessing the interface.
- **Context:** IMA-187 composites at *thumbnail* scale into the existing 88 px cell instead. Deep zoom was consciously deferred here, not forgotten. The outside voice argued tiling is what makes coordinate placement pay off at all, which is a reason to sequence the region window's deep zoom sooner rather than to build tiling early.
- **Depends on / blocked by:** a real consumer, which today is the region window (`_region_viewer`), to supply the right interface shape.

## Freeform / manual-region layout (tissue, slide carrier) → own ticket
- **What:** A second layout mode so non-wellplate acquisitions (tissue, slide carrier, freeform ROIs) can be written and rendered by stage coordinate rather than row/column.
- **Why:** Split out of IMA-187 acceptance criterion 5 during /plan-eng-review (2026-07-21). It is not a small allowance inside the mosaic work — `_output.py:90-95` raises on any region that is not `<letters><digits>`, by design, to prevent mislabeled output directories; and `PlateOverview` indexes every cell by `(row_index, col_index)`.
- **Pros:** Extends the tool past wellplates to the tissue acquisitions the team actually wants next.
- **Cons:** Touches the well-id contract in the writer and the grid indexing in the viewer. Cannot be validated today — the tissue dataset is still TBD (the sample was removed from the machine).
- **Context:** IMA-187's coordinate placement is deliberately layout-agnostic (offsets derive purely from stage mm), so this ticket inherits the placement math for free and only owns the writer/viewer layout modes. Do NOT soften the `_output.py` raise to a warning as a shortcut — it exists to stop silent directory mislabeling.
- **Depends on / blocked by:** A real tissue acquisition to validate against; IMA-187's placement math (landing first).

## Per-FOV hit-testing on plate double-click → region-window ticket
- **What:** Make double-clicking a mosaic cell open the FOV actually under the cursor, instead of always FOV 0.
- **Why:** `_viewer.py:682` hardcodes `self.wellActivated.emit(c["well_id"], 0)`. Once a cell shows 36 FOVs, "which one did I click" becomes the obvious user expectation and is arguably the real payoff of the mosaic.
- **Pros:** Turns the mosaic from a picture into a navigation surface.
- **Cons:** Needs an inverse of the placement transform (pixel in cell -> FOV index), plus `_fov_index` widening — it is keyed by region only today, and slider labels are `f"{r}:0"`, one slot per well.
- **Context:** Deliberately excluded from IMA-187's acceptance criteria. `go_to_well_fov(well, fov)`, `plane_ref(region, fov, ...)` and the `wellActivated(str, int)` signal already carry a fov index, so the plumbing largely exists — the gap is `_fov_index` and the hit-test.
- **Depends on / blocked by:** IMA-187 placement math (supplies the transform to invert).
## Web / remote plate viewing → new ticket (IMA-212 could NOT answer this)
- **What:** An actual way to view a plate from a machine that is not the microscope — a browser-reachable renderer, or a viewer reading OME-Zarr over HTTP/S3 from a shared store.
- **Why:** IMA-212's spec opened with "evaluate Odon as the web/remote renderer." Reading Odon v0.1.5's source killed that premise: it is a native Rust desktop GUI (`eframe::run_native` on every path except `--check`), with no HTTP server, no WASM build, and no headless render-to-file. Its only listening socket is a localhost MCP control bridge on `127.0.0.1:17870` that its own docs say must not be network-exposed. "Remote" in Odon means it *reads* remote zarr, not that it *serves* pixels. So the original question is still completely open, and IMA-212 must not be read as having answered it.
- **Pros:** Unblocks anyone who needs to look at plate output without sitting at the instrument — the actual underlying need.
- **Cons:** Genuinely larger than a bridge ticket: needs a hosting story, an access-control story, and a decision between serving pixels (render server) vs serving bytes (object store + a zarr-reading web viewer).
- **Context:** SquidMIP's existing output is already a strong starting point — `plate.ome.zarr` is spec OME-NGFF v0.5, zarr v3, with a per-FOV pyramid, so a zarr-capable web viewer could read it over HTTP with no writer changes at all. The `_montage.py` hover viewer is a precedent for a static, server-free artifact. Worth scoping the "object store + web zarr viewer" option first, since it reuses everything already built.
- **Depends on / blocked by:** nothing technically; needs a product call on who the remote viewer is and where the data lives.

## Non-integer inter-level pyramid scale ratios → possible IMA-184 follow-up
- **What:** `_multiscales` (`squidmip/_output.py:165`) computes each level's scale as `p * (y0/y)`. Because `_downsample_yx` crops odd axes by one before halving, the ratio between levels is not exactly 2.0 — a 4167-px axis gives 2.0009…, not 2.
- **Why:** The metadata is *self-consistent* (the scale honestly describes each level's real size), so this is not obviously a bug. But readers that use `scale` for level-of-detail placement can accumulate misalignment, which shows up as image drift while zooming rather than as an error. Nobody has looked for it, and a length-check on the scale array — which is what our conformance tests do — cannot catch it.
- **Pros:** Rules out a class of subtle, silent geometric error in every downstream viewer.
- **Cons:** May well be a non-issue; fixing it (padding instead of cropping odd axes) changes pixel content at coarse levels, which is a real behaviour change to a shipped writer.
- **Context:** Surfaced by the IMA-212 outside-voice pass. Cheapest first step is observational and already on IMA-212's Phase 0 checklist: zoom hard in Odon (which uses `scale` for LOD) and in ndviewer_light, and watch for drift. Only open a real ticket if drift is visible.
- **Depends on / blocked by:** IMA-212 Phase 0 observation.
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

## Parallel export for large Minerva selections → deferred until IMA-205/187
- **What:** `squidmip/_minerva.py` exports FOVs in a plain sequential loop over `project_well`. When real multi-FOV selection lands, large selections should use `project_plate`'s existing worker pool.
- **Why:** `project_plate` (`_engine.py:131-139`) already parallelises and accepts `regions=`, so the machinery exists. A 50-FOV selection exported serially would be a visibly slow button.
- **Pros:** Reuses a tested parallel path instead of hand-rolling threads; large selections stop being a wait.
- **Cons:** `project_plate` picks FOVs through `select_fovs(metadata, n_fovs)`, which cannot express an arbitrary `(region, fov)` list — the semantics genuinely do not match a user selection today, so adopting it needs a new entry point, not a swap.
- **Context:** IMA-228 deliberately chose the sequential loop: selections are 1 FOV today (`n_fovs=1`, `_viewer.py:853`), each FOV is a few hundred ms, and explicit beat clever. Revisit when someone can measure an actual slow export on a real multi-FOV selection. Do not pre-optimise this on speculation.
- **Depends on / blocked by:** IMA-187 (multi-FOV).

## Pane 3 width is not remembered across a collapse → CLOSED (pane removed)
- **CLOSED 2026-08-05:** the exploration pane was removed. There is no pane 3 and no `_sync_explore_pane`.
- **What:** Persist the exploration pane's width so closing its last tab and re-opening one restores the width the user dragged the splitter to, instead of recomputing 22% of the window.
- **Why:** `_sync_explore_pane` hands pane 3's width back to the VIEWER pane on collapse and recomputes it on the next reveal. A user who deliberately widens pane 3, closes their last subset, then Shift-drags again silently loses that adjustment.
- **Pros:** Small — one remembered int, set from `_split.sizes()` at collapse time.
- **Cons:** Needs a rule for the case where the window was resized while pane 3 was collapsed, or the remembered width can exceed the window.
- **Context:** IMA-237 (2026-07-21). The reveal deliberately takes its width from the VIEWER pane, never the plate pane (requirement 5), and that rule must survive whatever remembering is added. `_sync_explore_pane` is the single place both directions live.
- **Depends on / blocked by:** nothing; IMA-237 landed.

## Drag a tab BETWEEN panes 1 and 3 → CLOSED (there is one bar)
- **CLOSED 2026-08-05:** the exploration pane was removed. There is only one tab bar left, so there is nowhere to drag a tab TO. The
generalised `_detach_tab(index, tabs)` seam survives if a second bar ever appears.
- **What:** Let a tab dragged out of pane 1 be dropped into pane 3's bar and vice versa, rather than only floating out and re-docking to its origin.
- **Why:** IMA-237 generalized `_detach_tab`/`_redock` to take a source tab-widget, so both bars already route through one implementation and a float remembers its home bar (`_home_tabs`). Cross-pane docking is now a gesture question, not an architecture one.
- **Pros:** The architecture is already paid for; this is the drop-target half.
- **Cons:** Same open question as the IMA-209 drag-back TODO — no user has asked for it yet.
- **Context:** IMA-237 (2026-07-21). Pairs with "ImageJ-style drag-BACK re-attach for floating tabs"; if either is built, build both against the same drop-target code.
- **Depends on / blocked by:** real user demand; do not pre-build.
