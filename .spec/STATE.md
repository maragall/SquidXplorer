# STATE — IMA-212

- **Ticket:** IMA-212
- **Branch:** juliomaragall/ima-212-odon-zarr-bridge
- **Spec:** .spec/open/ima-212.md
- **Phase:** PLAN LOCKED (eng review complete 2026-07-20) — **gated on a manual Phase 0 spike**
- **Mode:** attended → user delegated finalize+push (2026-07-20)

## Now
Plan locked and pushed. **Do NOT arm the ralph watcher** (`.sprint/oracle.md`) yet — T0 is a
human step and the AFK build has nothing to do until it passes.

## Next
T0 GATE (manual, ~30 min): install Odon, hand-write a ~20-row `id,path,well,fov` CSV over an
existing `.hcs`, run `odon --mosaic-samplesheet <csv>`, and answer the Phase 0 table in the
spec. **Key question: does the mosaic preserve plate geometry?** Odon has no plate model, so
it renders a linear sequence — plausibly disqualifying for plate work. If it is, close IMA-212
with the finding recorded; that is a successful evaluation, not a failure.

Only if T0 passes: T1 `write_samplesheet` (directory glob) → T2 `find_odon` → T3 `launch_odon`
→ T4 `check_odon` → T5 `--odon` CLI flag → T6 tests → T7 docs → T8 oracle rewrite.
Full task list with files/verify steps: spec "Implementation Tasks".

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
