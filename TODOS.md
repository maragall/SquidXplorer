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
