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

## `_scaled` allocates a full-plate pixmap at max zoom (~1.65 GB) → viewer perf
- **What:** `PlateOverview.paintEvent` rebuilds `self._scaled` at the full plate rect (`_viewer.py:696-699`), sized `nc*cd x nr*cd`. `wheelEvent` (`:639`) clamps zoom at `_fit_cd() * 40`, so on a 700x600 widget at 1536wp `cd` reaches ~518 and the pixmap becomes 24864x16576 ≈ **1.65 GB**. Crop the scale to the visible viewport instead.
- **Why:** A user who zooms all the way in on a dense plate can OOM or hard-freeze the app. It is latent today because nobody has zoomed to the clamp on a 1536wp with the montage populated.
- **Pros:** Removes a multi-GB allocation; makes deep zoom usable; the same crop IMA-220 already applies to the carrier layer (D11) generalizes to the montage.
- **Cons:** Touches the hot paint path and its `_scaled`/`_scaled_cd` cache invalidation, which is subtle (pan must not invalidate, zoom must).
- **Context:** Found by the IMA-220 outside voice while checking whether the new carrier cache could copy the `_scaled` pattern — it can't, for exactly this reason. IMA-220 therefore ships a viewport-cropped carrier cache (D11) while deliberately leaving the pre-existing montage bug alone. Both should end up sharing one crop helper.
- **Depends on / blocked by:** Nothing. Independent of IMA-220, but easiest right after it, since D11 lands the crop logic.

## Persist `wellplate_format` into written plate metadata → output writer
- **What:** The OME-Zarr plate written by the operator run carries no sample-format string, so reopening a computed plate loses it. `_open_computed` (`_viewer.py:1643`) builds `self._meta` at `:1683` with only `channels/z_levels/n_z/n_t/regions`. Write the format at plate-creation time and read it back here.
- **Why:** IMA-220 gives the ingest path a carrier background but the computed-plate path silently has none (decision D10), so the plate view changes appearance depending on how you opened it.
- **Pros:** Makes the two open paths visually consistent; the format is generally useful metadata for any future geometry-aware feature (IMA-214's Plate ABC, multi-FOV work).
- **Cons:** Needs a writer change plus a back-compat read (plates written before the change have no key), so the `None` path stays regardless.
- **Context:** Inferring the format from `(nr, nc)` was considered and rejected: a subset acquisition writes only acquired wells, so an 8x3 plate matches no standard format. Explicit metadata is the only reliable fix.
- **Depends on / blocked by:** The output writer (IMA-184 lineage). IMA-220 ships the `None` degradation in the meantime.

## Carrier backgrounds for 6/12/24/96 + slide carriers → follow-up to IMA-220
- **What:** Vendor the remaining four plate PNGs and add their rows to the `_plate.py` `CarrierSpec` table; separately, measure anchors for `glass slide` and `4 slide carrier`, which Squid hardcodes rather than calibrating.
- **Why:** IMA-220 deliberately ships only 384 and 1536 because `_viewer.py:66` `_SUPPORTED_PLATES = ("384", "1536")` rejects every other format at `:1516`. The other carriers would be unreachable assets and untestable code.
- **Pros:** Trivial once the scope guard lifts — the anchors are already verified in Squid's `sample_formats.csv`, and the outside voice edge-detected all six PNGs against them (max error ~1.5 px). Roughly a table edit plus two file copies.
- **Cons:** Slide carriers are genuinely harder: their CSV row is all zeros and Squid special-cases them with hardcoded origins and a different `mm_per_pixel` (0.1453), so they need real measurement, not vendoring.
- **Context:** IMA-220's capability check (D6) already returns `None` for every uncalibrated format, so adding these is additive and cannot regress current behavior.
- **Depends on / blocked by:** `_SUPPORTED_PLATES` widening, which is gated on stitching support for coarser plates.

## Fix upstream squid2minerva/colors.py display_color nesting → external repo
- **What:** `squid2minerva/colors.py:load_yaml_colors` reads `channel["display_color"]`, but real `acquisition_channels.yaml` nests it under `channel.camera_settings.<cam>.display_color`. Its Minerva OME-TIFF exports only get right colors via the wavelength-fallback map — a custom yaml color is silently ignored.
- **Why:** Confirmed against a real dataset yaml. It's correct-by-luck today because the fallback palette matches the standard 4 channels; any non-default color drops silently.
- **Pros:** Fixes silently-wrong colors in a sibling tool's exports.
- **Cons:** Different repo, different owner; not on any SquidMIP critical path.
- **Context:** SquidMIP does **not** carry this bug — IMA-189's `squidmip/_channels.py` already resolves `display_color` correctly (top-level v1.0+ *and* nested `camera_settings`, mapped by name, raises on unresolved), and IMA-184 consumes `metadata.channels[].display_color` rather than re-parsing the yaml. This TODO is purely a flag for whoever owns `~/CEPHLA/projects/explorer/squid2minerva`.
- **Depends on / blocked by:** squid2minerva maintainer.
