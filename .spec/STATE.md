# STATE — IMA-211

- **Ticket:** IMA-211
- **Branch:** juliomaragall/ima-211-stitcher-operators
- **Spec:** .spec/open/ima-211.md
- **Tracked record:** docs/ima-211-eng-review.md
- **Phase:** THINK
- **Mode:** attended → handed off; verified factual corrections applied without re-asking

## Now
_(nothing in flight)_

## Next — re-scoped to the acceptance oracle
1. **T4 (do first, TIME-SENSITIVE)** raise the `original_coordinates_{t}.csv` geometry correction
   against IMA-187 before that branch merges.
2. **T1** synthetic cut-and-restitch harness (oracle #1 + #5). Runs today. Becomes IMA-222's gate.
3. **T2** measure the real overlap fraction across all acquisitions on disk.
4. **T3** ASHLAR out-of-process as the external cross-check (oracle #4).
5. **T5** hand the workers-cap / 1.07 GB-per-well finding to IMA-222's open B1 decision.
6. **T6** strike MCmicro from the ticket title — it is not a stitcher.

## Decisions
- **SCOPE COLLAPSED.** IMA-211's stitch operator, axis seam, registry fix and FOV geometry are
  all duplicated by tickets that landed on origin during this review:
  **IMA-222** (stitch operator), **IMA-226** (consumes registry + surfacing `reference`),
  **IMA-187** (multi-FOV mosaic, `fov_positions`, `select_fovs(None)`). Do not rebuild them here.
- **What survives:** the acceptance oracle — the one thing no other ticket owns, and the thing
  IMA-222 lacks entirely.
- **MCmicro dropped permanently** — it is a Nextflow orchestrator that shells out to ASHLAR.
  ASHLAR is the only external worth running, and only out-of-process (it boots a JVM via pyjnius
  at import). BigStitcher needs a BigDataViewer resave; PyPetaKit5D needs the MATLAB Runtime and
  is Linux-only.
- **Geometry correction (contradicts IMA-187's lock):** Squid writes TWO coordinate files;
  `original_coordinates/original_coordinates_{t}.csv` has an explicit `fov` key and the actual
  stage position. Never derive grid geometry from `acquisition parameters.json` (it reports
  `Nx=1,Ny=1` for a real 6×6 grid).

## Blockers
- T1 is not blocked and is the highest-value build item.
- T4 is only useful before IMA-187 merges.
- Real blockers are narrower than the ticket claims: the **overlap fraction** must be measured
  before registration code is written (0% would make correlation impossible), and **focus
  quality** is what the laser-AF block actually protects — a stitch test against a broken-AF
  acquisition measures autofocus, not stitching.

## Unresolved (carry into merge review)
1. Does IMA-187 accept the geometry correction before merging?
2. IMA-222's B1 (naive placement vs. re-opening the no-tilefusion rule vs. shipping mislabeled)
   remains open; our scale/overlap findings are inputs, not a decision.
3. Does IMA-211 stay separate once re-scoped to the oracle, or fold into IMA-222? Recommend
   separate — an acceptance oracle owned by the ticket it grades is a weaker gate.

## Learnings
- **This review's first pass made four factual errors** (wrong coordinate file, trusted
  `acquisition parameters.json` for grid geometry, proposed scipy as if it were a dependency,
  sized on 4 FOV/well when real data is 16-36). All caught by the outside voice, all confirmed by
  hand. Root cause: it searched for real acquisitions, found none, and designed an on-disk data
  contract from a docstring plus a prior learning. **Not finding the data should block designing
  the parser, not license inferring it.**
- Second lesson: in a repo with many parallel worktrees, **check origin for sibling tickets before
  planning**, not after. Three of them had already claimed this ticket's scope.
- Prior learnings applied: `squidmip-z1-contract-and-cli-promise`, `ima-225-plan-locked`,
  `ndviewer-light-hcs-discovery`.

## Iterations
_(one line per Build iteration: n — what landed — verify result)_
