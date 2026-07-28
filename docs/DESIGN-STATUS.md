# DESIGN.md to code, status map

> **This file is stale and is the documented source of two wrong conclusions in an
> external review (2026-07-28).** It was last refreshed at `9fc34b8`, 199 commits ago.
> Individual lines corrected since then are marked inline. Refresh it wholesale or
> retire it; a half-true status map is worse than none, because it is quoted.

- Each design object mapped to the module that implements it, with status.
- done: behavior + tests exist. gap: not built yet. naming: exists under another name.
- Tests green: STALE, do not trust this number. Last edited `9fc34b8` (2026-07-09), 199 commits ago. As of 2026-07-28 the suite could not even report: it segfaulted partway through and took pytest's summary line with it, so a run that crashed and a run that passed looked identical (see `tests/test_window_lifetime.py`). Re-measure before quoting.

## Domain

- Acquisition: squidmip/reader.py (SquidReader.metadata). done as behavior. gap: not a named `Acquisition` class (it is a reader + metadata dict). A thin wrapper class is a naming refactor, staged (churn over working code, low value).
- AcquisitionImage: the (T,C,1,Y,X) arrays the engine yields. done as behavior. gap: not a named class holding an OperationStack.
- OperationStack (LayerStack): gap. Not built. New capability: an ordered toggleable layer list + a Layers tab. Highest value new feature. GUI visual, needs your QA.
- Operation (Strategy): squidmip/_viewer.py `_OPERATIONS` (dataclass registry: key, label, blurb, build_tab) + squidmip/_engine.py `_PROJECTORS` (backend). done. naming: backend (projector) and UI (build_tab) are paired by key, not on one object. Formalizing one `Operation` object is a clean refactor, staged.
- MIP: squidmip/projection.py `project`. done (streaming, bounded, tested).
- ReferencePlane: squidmip/projection.py `project_reference` (Tenengrad). done backend. gap: GUI "use the plane I see" override (auto works; override is a GUI feature).
- Stitcher: stub. gap (deferred, multi FOV target). Interim single FOV + notice: done.
- VideoPlayer: REMOVED. `squidmip/_video.py` and the Record tab were deleted in `c25d84d` ("IMA-185: remove recording from viewer"); recording belongs to Squid, not to a post-acquisition viewer. Corrected 2026-07-28: this line said "done" and named a file that does not exist, which is how an external review concluded the status map had drifted.
- MinervaAuthor, NautilusAgent: roadmap cards only. done as stubs.

## Engine

- Iterator: squidmip/_engine.py `project_plate`. done. naming: called project_plate, not Iterator. Bounded window, parallel, on_error skip: done. No per operator subclass: done (Strategy).
- FOV assembly strategy: squidmip/projection.py `select_fovs` (single FOV) + the multi FOV notice. done as behavior. gap: not a named SingleFov/Stitching strategy object.

## IO

- ReadAcquisition: squidmip `open_reader`. done. GUI delegates (does not read directly): done.
- WriteAcquisition: squidmip/_output.py `write_plate` (multiscale OME-Zarr + optional TIFF + disk guard). done. The `.mp4 via _video` claim is void: see VideoPlayer above.
- CLI: squidmip/_cli.py (pydantic-settings CliApp, up front validation, resilient skip). done.

## GUI

- GUI: squidmip/_viewer.py `PlateWindow`. done. Delegates to engine + IO: done.
- Processing pane: `_left_tabs` QTabWidget (Home non-closable, black strip + thin white outline, per operator tabs via build_tab). done. Composite + Factory Method: done. LayersTab: gap (see OperationStack).
- HomeTab, ZProjectionUI (MIP + Reference), VideoPlayerUI, CliUI: done.
- PlateView: `PlateOverview`. done (hue dots capped, red box + red dot, zoom/pan, double click). gap: upsample on zoom from the pyramid (v2; pyramids are written, the read path is not wired).
- ArrayViewer: maragall/ndviewer_light. done: register_array push (LRU bounded, cleared on new acquisition), ArrayViewer reuse (fast scrub), z-slider collapses at nz=1. gap: the GUI always-push feed wiring (foundation built in ndviewer; wiring needs your visual QA on GL).
- Observer seam: Qt signals (worker -> window -> panes). done as behavior. naming: not a named bus; kept thin.

## Genuine remaining gaps (ranked, not naming churn)

- OperationStack + LayersTab: new capability. Build + your QA.
- always-push GUI feed: wire PlateView/ArrayViewer to the push foundation. Your QA (GL).
- ReferencePlane viewer-override + PlateView upsample-on-zoom: v2 GUI refinements.
- Stitcher (multi FOV) + Minerva/Nautilus: deferred stubs.
- Naming refactors (Acquisition/AcquisitionImage/Iterator/Operation classes): staged. High churn over a working tested system, low behavior value. Do only if you want the exact class names as the source of truth.
