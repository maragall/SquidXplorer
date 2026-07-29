# NEXT STEPS

A running hit list of product requirements — things we want the tool to do, written down
the moment they occur to whoever is using it. Spencer or Julio picks them up.

Append with `/NS "..."`. Keep entries short: one line of what, one line of why it matters.
This file is for the *ask*, not the design.

**Not `TODOS.md`.** That file holds deferred *engineering* work captured during
plan-eng-reviews — What / Why / Pros / Cons / Context / Depends-on, written so a future
session does not rediscover the reasoning from zero. Entries here are earlier and lighter:
a user noticed something. An item that survives contact with a design discussion graduates
into an IMA ticket (and, if work gets deferred out of it, into `TODOS.md`). Nothing here is
committed to, scheduled, or estimated.

---

## Hit list

- [ ] **Font scaling for child windows** — the root window rescales its type on resize
  (`501f71e`), but the `[N] <well>` view windows and the Log window do not. They inherit the
  high-DPI fix, so they are the right size, but dragging one bigger leaves the type behind.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Confirm the fixed-size relaxation with Julio** — `501f71e` replaced
  `setFixedSize(596, 850)` with a resizable default. The old comment attributed the fixed size
  to Julio and to an explicit "identical on every monitor" invariant, which is now gone. Either
  confirm the change or gate the resize behind a setting.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **See a real mosaic render** — no run in this session has produced one: the first had no
  napari, the second had `C3` poisoning the whole plate. With `ece5d0b` placing 72 FOVs across
  8 wells, opening `A1` should finally draw a coordinate-placed mosaic. Until someone looks,
  "the mosaic works" is unverified.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Migrate the GUI to PyQt6** — napari ≥0.7 targets Qt6, and Cephla's Squid HCS GUI
  root-caused a glitchy mosaic to running napari 0.7 + vispy 0.16 under PyQt5
  (`jsschwrz/Squid` 757d571). AGAVE is built against Qt 6.9 and two Qt majors cannot share a
  process, so the volume renderer is unreachable from a Qt5 process. Do NOT rebase
  `origin/ima-qt6` — it is 88 commits behind and its payload is in `_viewer.py`, which main has
  moved by +2706/−1184 since. Redo on main. Needs `AA_ShareOpenGLContexts` (N windows each own
  a napari canvas) and the one-binding rule (PyQt5 + PyQt6 co-installed breaks GL rendering).
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Move `ndviewer_light` to qtpy** — prerequisite for the Qt6 migration: it imports PyQt5
  at module scope, so under Qt6 the `SQUIDMIP_VIEWER=ndv` fallback cannot load in-process.
  Scoped small — real imports are confined to `core.py`, none of the four known Qt6 breakers
  are present, 7 unscoped enums, and one genuine rename (`pyqtSignal` → `Signal`). The fiddly
  part is the PyInstaller spec and `environment.yml`, which hardcode `PyQt5.*` and
  `vispy.app.backends._pyqt5`.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Fix the Windows test-suite teardown crash** — `test_viewer`, `test_gui_commands`,
  `test_nav_wiring`, `test_viewer_3d` and `test_viewer_region_ids` die with `0xC0000005` at
  process exit, which eats pytest's summary line. Tests that passed look identical to tests
  that never ran, so Windows failures are effectively invisible. The suite currently has to be
  run per-file to get honest results.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Peak memory always reads 0 on Windows** — `_rss_mb()` takes its peak only from the
  POSIX-only `resource` module, so the footprint line reads `peak 0 MB` forever. `psutil` (now
  a declared dependency) exposes `PeakWorkingSetSize` on Windows. The low-memory warning the
  README promises cannot fire on Windows today.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **`test_plate.py`: 5 real failures** — including "two regions separated in y are two
  ROWS, not two columns". Pre-existing and unrelated to any recent change; not dependency
  gaps. Nobody has looked at whether they are Windows-specific.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Decide whether `.claude/skills/run-squidxplorer` ships** — it captures the verified
  Windows launch recipe (which venv, the undeclared deps, how to screenshot without stealing
  focus, why the suite must run per-file). Useful to the team, but it hardcodes paths specific
  to one workstation. Trim before committing.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Revisit the napari `<0.8` ceiling** — pinned to `>=0.7,<0.8` to match the range Cephla
  validated in the Squid HCS GUI. The bound is a Qt-binding hedge, not a known 0.8 defect;
  worth re-testing once the Qt6 migration lands.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Log window opens over the main window** — on start up the Log window lands on top of
  the main window. It should not. `_log_window` is sized (`resize(760, 240)`, `_viewer.py:3867`)
  but never positioned before `show()`, so Qt drops it wherever it likes — it covered the root
  on both launches this session.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Combine two runs in one plate view** — a way to merge the results of two runs into a
  single plate view. Use case: run `A1` alone to dial in the parameters, then run all the
  other wells with those parameters — and see the whole plate together afterwards.
  The transform-recipe + content-addressed result cache (`c304432`) already keys results per
  node, which is the machinery a merged view would read from.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Orphan "focus reference plane" window** — a stray window labelled *focus reference
  plane* floats with no home. Give it one. `d07db43` brought the Tenengrad reference plane back
  "onto the window's z-slider", so it likely escaped that placement; a small untitled 129x59
  top-level appeared alongside the root on both launches this session and may be it.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Pull in the original MIP Navigator** — bring the original MIP Navigator's functionality
  into SquidXplorer. **Reference: https://www.youtube.com/watch?v=c-TdJDUP734** — a demo of the
  tool. A MIP "map" of the wellplate on the left that updates as the cursor moves, with
  OpenSeadragon / Google-Maps controls: left-click-hold to pan, wheel to zoom.
  **Correction (2026-07-29, Julio):** both earlier notes here were wrong and this entry is now
  UNBLOCKED. The video is a recording of
  **https://github.com/maragall/ndviewer/tree/main/ndviewer_hcs** with everything preloaded, so the
  source is readable rather than only watchable. And the old MIP tool IS this repo, at commit
  **1504c05**, which is in our own history: `git show 1504c05:<path>` works. The previous note said
  the tool "is not on this workstation at all" and "do not go looking locally". Both false.

  **The feel is a technique, not a taste.** `ndviewer_hcs/plate_stack.py` holds a
  `PlateStackManager`: "Pre-computed Z x T plate assemblies stored as multi-page TIFF ... shape
  (t, z, channels, height, width), each page is a full assembled plate for one (t, z) combination.
  Memory-mapped for efficient random access." `get_page(t, z)` is a memmap index, so scrubbing z or
  t is a page read and not a fuse. The cache key is the DOWNSAMPLE FACTOR
  (`plate_stack_ds{f:.4f}.tiff` plus a metadata sidecar): one file per zoom rung for the whole
  plate, not per well and not per FOV. `PlateAssembler` derives the factor from a target pixel size
  and skips downsampling above 0.98.

  That is the direct answer to the 25 second fit-to-plate rung in the entry below: precompute the
  rung once and the slider is free. Handed to the agent building the persistent plate cache.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Show the app is loading, not crashed** — startup needs some indication that work is
  happening. Right now silence is indistinguishable from a crash. Corroborated this session:
  after launch the process reported an empty window title and a null window handle for several
  seconds while napari imported, and opening the 9-well plate took it from 91 MB to 419 MB with
  nothing on screen to say so.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Compact vs stage FOV placement** — a placement mode toggle: *stage* keeps FOVs where the
  stage actually put them, *compact* selectively removes the empty space between them. Today
  offsets come solely from stage micrometres through `_placement.fov_offsets_px`, which is the
  one seam an alternative placement would slot into.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Focus setting should persist between wells** — when reviewing data the *focus* setting
  should carry from one well to the next. Right now the user has to click *focus* again every
  time. The `✦ focus` button lives in each window's own `2D / 3D · ROI` strip, so the state is
  per-window and starts fresh with every window opened.
  **Proposed direction:** maybe `BEST FOCUSED PLANE` should be a global setting rather than a
  per-window one.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Timepoint support when Nt > 1** — decide how a dataset with time interacts with the
  viewer. Three separate things: `t` as a navigation axis (a slider, like z), `t` as a
  reduction axis (max-over-time as an operator), and `t` as part of a result's identity. The
  first two are close to free — napari is natively ND, and `Operator.consumes` is a frozenset
  so `{"t"}` needs no model change. The third is not: `ResultCache`'s `scope` is the
  `RRCCOOOO` node id with no `t` field, and its `version` means *acquisition* version, not
  timepoint. Detail and the failure mode are in `TODOS.md` under "Multi-timepoint iteration /
  projection". Open product question: one window per timepoint, or one window with a slider?
  <sub>added 2026-07-27 · Spencer</sub>

  **Measured while writing the plate contract (2026-07-29).** The store is not the problem: the
  writer writes every timepoint (`project_plate` calls `project_well` with `t=None`, so the array
  is `(n_t, C, 1, Y, X)`), and the individual-TIFF export writes `tiff/{t}/...`. The read side is
  where it collapses, and unevenly: `reader.read` and `_montage` and `_tilesource` all take a `t`,
  while `_viewer._ComputedPlateWorker._read`, `_ZarrLoupeSource.coarse` and `_on_well` hardcode
  `[0, :, 0]`. So a multi-timepoint plate renders as its FIRST frame with no error anywhere, and
  no test catches it because every fixture in the suite is `Nt = 1`. Written up honestly in the
  Time section of `docs/plate-contract.md`; `python -m squidmip.contract.validate` now warns when
  a plate carries more than one timepoint, and `tests/test_plate_contract.py` pins the three
  hardcoded sites so the doc cannot rot. Julio: users WILL drop multi-timepoint datasets on this.
  <sub>added 2026-07-29 · Julio</sub>

- [ ] **"Close view" should close selected views** — rename it *Close selected views* and have it
  accept a multi-select. `1073999` already gave the Window navigator multi-select (nested
  hierarchy, Blender arrows, Linux multi-select), so the selection model may already carry what
  the button needs — it is the action that is singular.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Arrow-key navigation in the Window navigator** — up/down arrows should move the
  selection so open windows can be stepped through by keystroke. `90e9894` already makes
  selecting a row raise its window to the front, so arrows would drive behaviour that exists.
  <sub>added 2026-07-27 · Spencer</sub>

- [ ] **Complete the MIP-on-plateview feature** — the deep-zoom work landed and is good, but it
  does not yet emulate the reference: **https://www.youtube.com/watch?v=c-TdJDUP734**. Handing
  over; picking this up needs the video watched first, because the gap is a feel question, not a
  bug list.

  *Shipped so far* (`8ee046c`, `7fb4eec`, `b8ca28d`, all on `main-window-review`):
  `ReaderTileSource` gives raw acquisitions MIP tiles with no written plate; `_TileFetcher`
  decodes off the GUI thread newest-first; `paintEvent` overlays cached tiles on the montage and
  sharpens in place. Gestures were already right — `wheelEvent` has zoomed about the cursor
  since IMA-221. `SQUIDMIP_DEEP_ZOOM=0` disables the whole overlay.

  *Known gaps to measure against the video:* tiles engage only above `cd > _CELL` (88 px/well),
  so everything below that zoom is still the flat 88 px montage — the video appears continuous
  from the whole plate inward. Coarse rungs cannot be served by `ReaderTileSource` as it stands:
  a fit-to-plate tile overlaps all 72 FOVs and measured **25 s** to build. The fix that was
  scoped but not built is a composite source — `InMemoryMultiscale` fed from the existing
  `_PreviewWorker` pass for plate rungs, `ReaderTileSource` for FOV rungs — which would also
  close the seam where the montage shows a mid-stack plane while tiles show a MIP.
  <sub>added 2026-07-28 · Spencer</sub>

  *Update, 2026-07-29 — the COST half of that is done.* The composite source is built
  (`_tilesource.CompositePlateSource`) and is what `set_tile_source` now arms, fed by the
  persisted plate cells (`_platecache`, gap 1 of the three-viewers review — the two turned out to
  be one piece of work). One fit-to-plate tile, measured: 2.387 s → 65 ms first / under 0.01 ms
  steady on the 55-FOV tissue set, and 8.448 s → 1.44 s first / 0.14 ms steady on the 1536-FOV
  `sim_1536wp`. Reopening the plate itself went 15.2 s → 0.075 s there, surviving a restart. The
  cache follows `ndviewer_hcs/plate_stack.py`, the tool in the reference video: per-well cells
  while the preview streams, compacted into one memory-mapped plate page when the pass finishes.

  **What is still open is the FEEL half, and it is a rendering question, not a cost one.** Two
  things remain: (a) a world-space enumerator, because `_visible_fov_tiles` is keyed by
  `(region, fov)` and a plate rung is keyed by a world grid cell — the montage's uniform cell
  grid and the ladder's stage micrometres agree only inside one cell, so placing plate-rung tiles
  on the montage needs a viewport in µm rather than in cells; and (b) the mid-stack-plane versus
  MIP seam, which is unchanged: the plate rungs now reproduce what the montage draws, so closing
  the seam means moving the montage to a projection, exactly as the note above says.
