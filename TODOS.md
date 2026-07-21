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
