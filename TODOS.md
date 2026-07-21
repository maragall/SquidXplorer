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
## Gesture arbitration on PlateOverview (shift-select vs pan vs loupe) → IMA-221
- **What:** A single explicit gesture policy for `PlateOverview`'s mouse handlers, deciding between drag-to-pan (shipped), shift-drag marquee select (IMA-221), and press-and-hold loupe (IMA-208).
- **Why:** Three tickets independently add a gesture to the SAME handlers (`mousePressEvent:647`, `mouseMoveEvent:652`, `mouseReleaseEvent:670`). Whoever lands second inherits an undocumented conflict; the pan path already claims plain left-drag with a 3px threshold (:655).
- **Pros:** One place decides what a drag means; each later gesture ticket becomes additive instead of a rewrite of someone else's branch.
- **Cons:** Slightly more design up front than "add a modifier check"; needs agreement across three backlog tickets.
- **Context:** Today `mousePressEvent` unconditionally arms pan state on LeftButton. Shift-drag must branch BEFORE that. The loupe (IMA-208) wants press-and-hold, which competes with the same 3px pan threshold on the time axis rather than the modifier axis — so a modifier check alone won't settle it. `_sel` is currently a single `(ri,ci)` (:548) painted as one red box (:752-756); marquee select needs it to become a set with a multi-cell paint path.
- **Depends on / blocked by:** IMA-221 owns the selection gesture; coordinate with IMA-208 before either lands.
- **Status (IMA-205 rebase):** the MODIFIER axis is settled — Shift owns selection (`mousePressEvent` branches before pan arms), Shift+drag now also opens the exploration tab (`marqueeSelected` -> `_on_marquee_selected`), Shift+click stays a refine-one-well toggle that deliberately opens nothing. The TIME axis is still open: IMA-208's press-and-hold loupe competes with the plain-drag 3px pan threshold, not with a modifier, so it still needs a policy.

## Exploration-tab persistence across acquisitions → post-IMA-205
- **What:** Decide whether exploration tabs survive re-ingesting a different acquisition, and if so how their region sets are revalidated.
- **Why:** `ingest()` (:1493-1505) resets reader/`_fov_index`/`_overview` but never `_op_tabs`. The eng-review fix closes exploration tabs on ingest (the safe default), but the richer behavior — reopen the same selection on a re-ingest of the SAME acquisition — is a real workflow for anyone iterating on one plate.
- **Pros:** Users re-open the same plate constantly while tuning operators; losing their exploration set every time is friction.
- **Cons:** Requires acquisition identity in the tab key plus revalidation that every region still exists in the new `_fov_index`.
- **Context:** Surfaced by the /plan-eng-review outside voice (2026-07-20) and confirmed in code. The eng-review decision includes acquisition id in the content-addressed tab key specifically so this extension stays cheap.
- **Depends on / blocked by:** IMA-205 landing its ingest-teardown fix first.

## Partial `.hcs` cleanup after a stopped save run → fast-follow after IMA-205
- **What:** Clean up (or mark as partial) the `.hcs` output directory when a `save=True` operator run is stopped mid-flight.
- **Why:** IMA-205 makes stopping routine (closing an exploration tab stops its run). A stopped save leaves a partial plate that `resolve_plate_root` will later happily recognize as a real plate, so the user can re-open a half-written result as if it were complete.
- **Pros:** Prevents silently trusting a truncated plate; complements the resume/checkpoint TODO already filed against IMA-184.
- **Cons:** Needs a "complete output" definition — the same definition the resume/checkpoint item needs, so the two should be designed together.
- **Context:** Before IMA-205, stopping only happened on app close (`closeEvent:1970`) or re-ingest, both of which end the session anyway. Making close-tab stop a run turns a rare path into a common one. Surfaced by the /plan-eng-review outside voice (2026-07-20).
- **Depends on / blocked by:** Overlaps the existing "Resume / checkpoint for long plate runs" TODO — resolve as one design.

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

## IMA-205 exploration tab: verify it registers via _open_op_tab → IMA-205
- **What:** When the exploration pane lands, open it through `_open_op_tab(key, title, builder)` so it inherits detach/float/re-dock for free.
- **Why:** IMA-209 made detach a property of the tab container (eng review D2); any tab bypassing the registry won't detach and won't be cleaned up by `_dispose_tab_widget`.
- **Pros:** Zero extra work in 205 to get the Nick float behavior.
- **Cons:** None — this is a one-line integration constraint.
- **Context:** Registry + cleanup contract in `squidmip/_viewer.py` (`_op_tabs`/`_floating`/`_dispose_tab_widget`).
- **Depends on / blocked by:** IMA-205 design.
## Viewport tiling / LOD for deep zoom → exploration-pane ticket
- **What:** `viewport(bbox, zoom) -> tiles` with level-of-detail selection and frustum culling, so a single well can be zoomed into without fetching the whole mosaic.
- **Why:** Cut from IMA-187 during /plan-eng-review (2026-07-21). The plate overview does not need it: `PlateOverview` is a fixed-resolution bitmap montage (`_viewer.py:533` allocates `QImage(nc*88, nr*88)`, `:588` blits 88x88 blocks, `:697-701` smooth-scales the one bitmap), so a 36-FOV mosaic occupies 88 px at every zoom and there is nothing to cull.
- **Pros:** Unlocks true deep-zoom navigation into a well, which is what the exploration pane is for.
- **Cons:** The IMA-187 draft specified it as "framework-neutral (pyqtgraph now, Viv later)" — but pyqtgraph appears nowhere in the repo or `pyproject.toml`, and Viv is an explicit non-goal. Designing it before a real consumer exists means guessing the interface.
- **Context:** IMA-187 composites at *thumbnail* scale into the existing 88 px cell instead. Deep zoom was consciously deferred here, not forgotten. The outside voice argued tiling is what makes coordinate placement pay off at all, which is a reason to sequence the exploration pane sooner rather than to build tiling early.
- **Depends on / blocked by:** The exploration-pane ticket, which supplies the first real consumer and therefore the right interface shape.

## Freeform / manual-region layout (tissue, slide carrier) → own ticket
- **What:** A second layout mode so non-wellplate acquisitions (tissue, slide carrier, freeform ROIs) can be written and rendered by stage coordinate rather than row/column.
- **Why:** Split out of IMA-187 acceptance criterion 5 during /plan-eng-review (2026-07-21). It is not a small allowance inside the mosaic work — `_output.py:90-95` raises on any region that is not `<letters><digits>`, by design, to prevent mislabeled output directories; and `PlateOverview` indexes every cell by `(row_index, col_index)`.
- **Pros:** Extends the tool past wellplates to the tissue acquisitions the team actually wants next.
- **Cons:** Touches the well-id contract in the writer and the grid indexing in the viewer. Cannot be validated today — the tissue dataset is still TBD (the sample was removed from the machine).
- **Context:** IMA-187's coordinate placement is deliberately layout-agnostic (offsets derive purely from stage mm), so this ticket inherits the placement math for free and only owns the writer/viewer layout modes. Do NOT soften the `_output.py` raise to a warning as a shortcut — it exists to stop silent directory mislabeling.
- **Depends on / blocked by:** A real tissue acquisition to validate against; IMA-187's placement math (landing first).

## Per-FOV hit-testing on plate double-click → exploration-pane ticket
- **What:** Make double-clicking a mosaic cell open the FOV actually under the cursor, instead of always FOV 0.
- **Why:** `_viewer.py:682` hardcodes `self.wellActivated.emit(c["well_id"], 0)`. Once a cell shows 36 FOVs, "which one did I click" becomes the obvious user expectation and is arguably the real payoff of the mosaic.
- **Pros:** Turns the mosaic from a picture into a navigation surface.
- **Cons:** Needs an inverse of the placement transform (pixel in cell -> FOV index), plus `_fov_index` widening — it is keyed by region only today, and slider labels are `f"{r}:0"`, one slot per well.
- **Context:** Deliberately excluded from IMA-187's acceptance criteria. `go_to_well_fov(well, fov)`, `plane_ref(region, fov, ...)` and the `wellActivated(str, int)` signal already carry a fov index, so the plumbing largely exists — the gap is `_fov_index` and the hit-test.
- **Depends on / blocked by:** IMA-187 placement math (supplies the transform to invert).
