# STATE — IMA-216

- **Ticket:** IMA-216
- **Branch:** juliomaragall/ima-216-viewport-tiler
- **Spec:** .spec/open/ima-216.md
- **Phase:** THINK
- **Mode:** attended → user delegated finish (commit+push, no further input until merge)

## Now
Spec locked. Next session starts Build at T1 (contract types in `_tiling.py`).

## Next
1. T1 — TileDescriptor + Geometry (validating ctor) + TileSource Protocol
2. T2 — pure `select_tiles` (LOD pick + vectorized cull)
3. T3 — TileCache (byte-budget LRU, pinning, resolve, invalidate)
4. T4 — 55k-box correctness + integration-marked timing
5. T5 — sibling spec edits (ima-217 implements contract; ima-218 ownership flags)

## Decisions
- Coarse LOD comes from a plate-level pyramid above per-FOV pyramids; construction owned by IMA-217 — per-FOV pyramids bottom out at 256px so O(viewport) fails at fit-to-plate — alt (descriptor cap fallback) rejected as never fixing the default view.
- Canonical world space = stage µm; zoom = µm/screen-px — matches `_multiscales` µm scales; framework-neutral — alt (screen px) rejected as welding core to Qt.
- 216 owns the contract (TileDescriptor + Geometry + TileSource Protocol); 217 implements it (edge 217→216) — consumer defines the interface — alt (defer to 218) rejected as unbudgeted adapter work.
- Cull = single vectorized op over per-level (N,4) µm arrays; correctness at 55k boxes in CI, timing only under `integration` marker — hard <2ms CI assert dropped (outside voice: timing asserts flake; repo precedent quarantines them).
- Pure/stateful split: `select_tiles` returns the IDEAL set; `TileCache.resolve` maps ideal→renderable (parent substitution) — resolves outside voice's "purity is illusory" objection.
- Parent retention: pin ancestors of pending tiles; pin cap = budget/2, overflow drops oldest pending; unpin on insert AND fetch_failed.
- Cache budget in bytes (`arr.nbytes`); oversized single tile admitted alone.
- Tiles cached pre-composite per channel; `invalidate(predicate)` for streaming acquisition.
- Cache lifecycle is synchronous + caller-driven (mark_pending/insert/fetch_failed); no threads in `_tiling.py`; executor + render-loop ownership flagged to IMA-218 (TODOS.md).
- Build-time deps: none (tests fabricate geometry); 214/215 feed real geometry at 218 integration. Spec's original "depends on 214, 215" was wrong (outside voice #6, accepted).
- Outside voice two-regime YAGNI challenge rejected — scope locked at Step 0; plate-pyramid path stands.

## Blockers
_(none)_

## Learnings
_(distilled in Reflect -> /learn)_

## Iterations
_(one line per Build iteration: n — what landed — verify result)_
