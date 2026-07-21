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
