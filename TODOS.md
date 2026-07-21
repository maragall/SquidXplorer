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

## End-to-end non-wellplate ingest (glass slide / flexible regions) → new ticket
- **What:** Actually process acquisitions whose regions are not well ids — `glass slide`, `4 glass slide`, and Squid's flexible-region mode (`region_0`, `scan_area_1`). Requires: a zarr output layout for non-well regions, relaxing `_output.parse_well_id`, and lifting the `384|1536` gates at `_cli.py:97` and `_viewer.py:66`.
- **Why:** IMA-214 established that a glass slide is structurally just a 1x1 plate, but stopped at the model. Today these acquisitions are refused end to end, and before IMA-214 they were worse than refused — `region_N` filenames silently produced a max projection over a single z-plane.
- **Pros:** Unblocks glass-slide and multi-slide-carrier users, who are currently rejected outright. Upstream Squid agrees with the premise (`gui_hcs.py:2397`: `TODO(imo): ... It seems like it's just a "1 well plate"`).
- **Cons:** Touches reader, writer, CLI and viewer in one diff — hard to review and hard to bisect. The genuinely open design question is what the zarr layout should be when there is no `{row}/{col}` to write to; `plate.ome.zarr` and ndviewer_light both assume a well grid.
- **Context:** IMA-214 deliberately took scope option B (model + DRY adoption) and deferred this. `_output.parse_well_id` hard-raises on non-well regions **by design** (`_output.py:90-95`) — that is not a bug to delete casually, it exists so a manual acquisition can't be written to a mislabelled directory. Start by deciding the output layout, not by relaxing the parser.
- **Depends on / blocked by:** IMA-214 (Plate model + `NotAWellPlateError`).

## Detect drift between the vendored sample_formats.csv and the user's Squid → follow-up
- **What:** Compare SquidMIP's vendored `sample_formats.csv` against a located Squid install's `cache/sample_formats.csv` (and warn on mismatch), or checksum it at build time.
- **Why:** IMA-214 vendors the CSV so squidmip works standalone. But Squid prefers a user-editable `cache/sample_formats.csv` over its shipped copy (`_def.py:1174-1182`). A user who customized a plate definition gets geometry from squidmip that silently disagrees with what their microscope actually used.
- **Pros:** Closes the one failure mode in IMA-214 that has no test, no detection, and would be completely silent.
- **Cons:** Requires locating a Squid install, which IMA-214 explicitly rejected as a hard runtime dependency. Would have to be advisory-only and skip cleanly when Squid isn't present.
- **Context:** IMA-214 decision D4 chose "vendor + explicit override path". The override makes the problem *fixable* by a user who knows about it; this TODO is about making it *detectable* by one who doesn't. Flagged as the single critical gap in the IMA-214 failure-mode table.
- **Depends on / blocked by:** IMA-214 (vendored data + override path).

## 4-slide carrier has a PNG but no geometry row → blocked on upstream data
- **What:** Define geometry for the `4 glass slide` carrier so `Plate.carrier_image` and any overlay can position slides on it.
- **Why:** Squid's carrier-PNG registry lists `"4 glass slide": "images/4 slide carrier_1509x1010.png"` (`core.py:1532`), but `sample_formats.csv` has **no** `4 glass slide` row — only `glass slide,0,0,0,0,0,0,0,1,1`. So there is literally nothing to build a Plate from. IMA-214 makes `carrier_image` refuse it explicitly rather than half-support it.
- **Pros:** Completes the carrier registry; the 4-slide carrier is the Squid default background image (`core.py:1544` falls back to it), so it is not an exotic case.
- **Cons:** The data does not exist upstream. Either add a row to `sample_formats.csv` (an upstream Squid change, different repo/owner) or hand-measure the layout, which then drifts from Squid.
- **Context:** Also note the 4 slides may not be a uniform 2x2 grid with even spacing — if they aren't, a flat `Plate` dataclass can't express it and this is where IMA-214's decision D5 (collapse the class hierarchy) would need revisiting with a subclass.
- **Depends on / blocked by:** An upstream `sample_formats.csv` row, or a decision to hand-measure.

## number_of_skip is parsed but uninterpreted → low priority
- **What:** Interpret `sample_formats.csv`'s `number_of_skip` column (384-well plates have `1`, everything else `0`).
- **Why:** IMA-214's Plate carries the field because it carries the whole CSV row, but nothing reads it. Leaving it unmodelled is fine; leaving it *undocumented* would let a future reader assume it's already honoured.
- **Pros:** Removes an ambiguity about whether Plate's geometry is complete.
- **Cons:** Almost certainly acquisition-planning behavior (which wells the scope visits), not projection behavior — so the MIP tool may never need it.
- **Context:** Squid reads it in `_def.py:1159`. Check how Squid actually uses it before modelling anything.
- **Depends on / blocked by:** A real case where skipped outer wells matter to projection.
