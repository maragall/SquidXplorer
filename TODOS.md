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

## Stitched layer in the Layers tab → blocked on a stitcher
- **What:** A `stitched` layer alongside `raw` and the operator layers, so a multi-FOV well renders as one composited image instead of a sampled single FOV.
- **Why:** IMA-227's ticket title promises "stitched / raw / MIP", but nothing in the repo stitches. `_OPERATIONS` (`_viewer.py:410`) holds exactly one entry (`mip`), and stitching is explicitly deferred in `_cli.py:104`, `_viewer.py:66`, and `README.md:67`. The layer machinery is ready; the pixels are not.
- **Pros:** The layer stack already supports N layers, so adding one is a registry entry plus a worker — the expensive part (toggle/reorder/render) is done.
- **Cons:** Needs per-FOV geometry (stage position + overlap) that the reader does not currently expose; without it there is nothing to composite.
- **Context:** IMA-227 was rescoped from "build the toggle" to "fix the toggle's defects" precisely because the toggle exists. The stitched layer is the one part of the original ticket that was genuinely unbuildable. Related learning: `fov-axis-needs-geometry` — a bare-index FOV list re-hardcodes the API the moment stitch is built, so model `metadata.fovs` as `list[{index,position,overlap}]` first.
- **Depends on / blocked by:** A stitcher (tilefusion `compose()` or equivalent) + per-FOV geometry in `metadata`. Related: IMA-187 (multi-FOV per well).

## Layers tab vs "Return to raw view" — two affordances, one action → product call
- **What:** Decide whether the Layers tab and the "Return to raw view" button (`_viewer.py:1208`) should both exist, or whether one should go.
- **Why:** With exactly one operator, the Layers tab reduces to a checkbox that hides MIP and shows raw — which is what the Return-to-raw button already does, via a different code path with different side effects (`_raw_btn` visibility, preview-worker restart, status reset).
- **Pros:** Removing one affordance removes a whole class of "these two controls disagree" bugs, and the eng review already found one such disagreement.
- **Cons:** They are not strictly equivalent — Return-to-raw also restarts the raw preview worker and resets well status, which unticking a layer does not. Collapsing them means deciding which of those side effects is the intended behaviour.
- **Context:** IMA-227 T2 unifies the *code path* (both route through `_apply_layers`), which fixes the correctness bug. Whether both should be *surfaced to users* is a UX/product question the eng review deliberately did not decide. Surfaced by /plan-eng-review outside-voice §7 (2026-07-20). Revisit when operator #2 lands, since the Layers tab earns its keep the moment there are two operators to stack.
- **Depends on / blocked by:** IMA-227 T2 landing first; ideally a second operator existing so the tab's value is testable.

## Fix upstream squid2minerva/colors.py display_color nesting → external repo
- **What:** `squid2minerva/colors.py:load_yaml_colors` reads `channel["display_color"]`, but real `acquisition_channels.yaml` nests it under `channel.camera_settings.<cam>.display_color`. Its Minerva OME-TIFF exports only get right colors via the wavelength-fallback map — a custom yaml color is silently ignored.
- **Why:** Confirmed against a real dataset yaml. It's correct-by-luck today because the fallback palette matches the standard 4 channels; any non-default color drops silently.
- **Pros:** Fixes silently-wrong colors in a sibling tool's exports.
- **Cons:** Different repo, different owner; not on any SquidMIP critical path.
- **Context:** SquidMIP does **not** carry this bug — IMA-189's `squidmip/_channels.py` already resolves `display_color` correctly (top-level v1.0+ *and* nested `camera_settings`, mapped by name, raises on unresolved), and IMA-184 consumes `metadata.channels[].display_color` rather than re-parsing the yaml. This TODO is purely a flag for whoever owns `~/CEPHLA/projects/explorer/squid2minerva`.
- **Depends on / blocked by:** squid2minerva maintainer.
