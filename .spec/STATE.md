# STATE — IMA-212

- **Ticket:** IMA-212
- **Branch:** juliomaragall/ima-212-odon-zarr-bridge
- **Spec:** .spec/open/ima-212.md
- **Phase:** BUILT (T1–T7 implemented 2026-07-20; T0 gate deferred at user's explicit direction)
- **Mode:** attended → user delegated finalize+push, then directed implementation ahead of the gate

## Now
Code landed and pushed. `squidmip/_odon.py` + `--odon` CLI flag + `tests/test_odon.py`.
Verified: 27 passed / 1 skipped (the real-binary test — odon is not installed here); full
suite 138 passed / 2 skipped / 20 deselected, no regressions.

**The Phase 0 gate was NOT run.** The user chose to implement first and test in the master
merger workspace. So the open question the gate exists to answer is still open: Odon has no
plate model, so its mosaic is a linear sequence, not plate geometry — which may still turn
out to be disqualifying for plate work. The code is correct regardless; whether it is
*wanted* is what remains unanswered.

## Next
Manual: install Odon (Apple Silicon mac), run `squidmip <acq> --odon`, and answer the Phase 0
table in the spec — above all whether the flat mosaic is usable for plate work. Then either
adopt (consider the deferred GUI button) or close IMA-212 with the negative finding recorded.
T8 (rewrite `.sprint/oracle.md`) stays open; the watcher is still unarmed.

## Decisions
- **The spec's premise was false and was rewritten.** Odon v0.1.5 is a native Rust DESKTOP GUI
  (eframe/egui_glow; every path except `--check` calls `eframe::run_native`). No HTTP server,
  no WASM, no headless render; its only socket is a localhost MCP bridge on 127.0.0.1:17870.
  This ticket cannot evaluate web/remote rendering — that is now a TODO with its own ticket.
- Samplesheet points at the field dirs `write_plate` ALREADY writes; every one is a valid Odon
  image group (zarr v3, NGFF 0.5 `ome` nesting, ndim-length scale, y/x axes, uint16). Rejected:
  a second flat copy (doubles a hundreds-of-GB output), a symlink farm (redundant + Windows).
- **Enumerate by globbing `plate.ome.zarr/*/*/*/zarr.json`, NOT by re-calling `select_fovs`.**
  Reversal of the first draft, from the outside voice: re-calling it with independently passed
  `n_fovs`/`regions` can silently disagree with what was written, and a metadata-driven
  signature forecloses `--odon` against a previously written `.hcs` (the likeliest real use).
  Glob on `zarr.json`, never the directory — `_write_field` writes the group LAST, so a
  partially-written field dir exists without one.
- Oracle split (Decision 3A): automated CSV-schema + NGFF-conformance-over-the-rows + `odon
  --check` (skipif binary absent); the GUI half stays manual. Nothing automated can prove Odon
  renders — inherent to a GUI-only tool, which is why Phase 0 is a gate.
- Binary discovery: `$ODON_BIN` → PATH → platform default. macOS `.dmg` installs an `.app` and
  does NOT put odon on PATH, so the bundle fallback is the common case. **Never vendor or
  auto-download** — Odon is GPL-3.0-only, SquidMIP is BSD-3; mere aggregation only.
- Entry point: `--odon` on the existing CLI. No GUI button (0.1.x tool, ~15 downloads).
- Odon ignores `omero.color` (own 8-color cycle + `dapi`→blue hack, only channel 0 visible).
  Document it; do NOT rename channels to game the heuristic (clever over explicit).
- Detached-crash silence was NOT accepted: `poll()` after ~1s makes it visible (3 lines).

## Blockers
_(none technical — T0 needs a human with an Apple Silicon mac; Odon is not installed here.)_

## Learnings
- Odon: no HCS/plate support at all (zero `plate`/`well`/`hcs` hits in src/); pins any
  non-c/z/y/x axis (i.e. t) to 0; uint8/uint16 only. Releases are macOS .dmg (Apple Silicon
  only), Windows x86_64 .exe, Linux amd64 .deb — **no linux-arm64**.
- `_output.py:_write_field` writes arrays before `write_group`, so a field dir exists before
  its `zarr.json`. Any `dir.exists()` enumeration passes partial writes downstream.
- `_multiscales` inter-level ratios are not exactly 2.0 (odd axes cropped before halving) —
  TODO'd; Phase 0 checks whether the drift is visible when zooming.

## Iterations
- 0 — plan-eng-review: 6 issues (4 arch, 2 quality), 0 critical gaps, 0 unresolved. Outside
  voice (Claude subagent; Codex CLI not installed) returned 7 findings, all absorbed — two
  real defects the interactive review missed (exists-check predicate, false "cannot disagree"
  claim), one structural reversal (directory glob), one strategic (Phase 0 gate added).
- 1 — build T1–T7 (user directed implementation ahead of the T0 gate). `_odon.py`
  (write_samplesheet / iter_fields / find_odon / check_odon / launch_odon), `--odon` on the
  CLI, `tests/test_odon.py` (28 tests). Verify: `pytest tests/test_odon.py` 27 passed 1
  skipped; full suite 138 passed 2 skipped 20 deselected. End-to-end smoke confirmed natural
  plate ordering (B2, B3, B10, AA1), relative paths resolving from the CSV dir, and a
  half-written field (arrays present, zarr.json removed) correctly excluded.
