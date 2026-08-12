# HCS Viewer, object design (v1)

- Informal object oriented spec in bullets. We formalize in code (pydantic) and a UML diagram after.
- Each class lists Purpose, Requirements (mined from the conversation), Collaborators.
- Operators are split into Implementation (backend) and Interface (UI). That split is the through line.
- Scope: post acquisition only. Reads finished Squid well plate data on disk (T, C, Z, FOV already saved). No live capture, no stage motion.

## Design principles

- Data intensiveness first: streaming, bounded memory, flat in plate size, never OOM, fail loud, skip a bad well and continue.
- One engine runs every operator. GUI and CLI share it.
- Adding an operator is one registry entry plus one UI builder. No subclass tree, no scattered text edits.
- No slop, no hallucinated constants. Derive or expose tuning.

## Patterns used (named, minimal, and why)

- Strategy: an Operation is the varying transform. The engine does not change to gain one.
- Registry: the operation table. One entry grows the menu, the cards, and the CLI.
- Factory Method: each Operation builds its own UI tab. Backend and UI stay paired on one object.
- Composite: the Processing pane is a set of tabs. The OperationStack is an ordered set of layers.
- Observer (Qt signals): operator UIs emit results. PlateView and ArrayViewer react. This is the "event handler that talks to the panes".
- Bounded producer and consumer: the Iterator streams wells with a fixed in flight window.
- Deliberately avoided: Iterator subclasses per operator (Strategy removes the need), deep GUI subclass trees, a double click that means many things (see Corrections).

## Tooling

- pydantic for parameter models and validation.
- pydantic-settings CliApp for the CLI, same shape as the stitcher.
- ruff and black for lint and format. Mirror the archived stitcher config (Cephla-Lab/image-stitcher/dev).
- imageio-ffmpeg for .mp4. tensorstore for zarr. PyQt5 for the GUI (optional extra). ndv via maragall/ndviewer_light.

## Corrections to the model (lingo and logic)

- Iterator needs no per operator subclass. The Operation is the Strategy. The only real iteration variance is FOV assembly (one FOV now, stitch later), so that is a separate strategy, not an Iterator subclass.
- OperationStorage is an ordered, toggleable list of layers (base first), not a strict LIFO stack. You enable, disable, and reorder any layer.
- VideoPlayer z is not an axis you record. You flatten Z first (MIP or ReferencePlane) or pick a single z index. Recording along Z is out of scope (see below).
- The GUI does not read or write files itself. It delegates to ReadAcquisition, WriteAcquisition, and the Iterator. Separation of concerns (you sensed this).
- Name: prefer PostProcessingOperation (or just Operation) over ImplementPostProcessingOperation. A class is a noun. The verb lives on the method (run / apply).

## Domain objects

### Acquisition

- Purpose: the read only input model. Lazy access to an on disk Squid well plate.
- Requirements:
  - Reads every Squid well plate scan format (adapt maragall/stitcher reader; ref Squid job_processing.py).
  - Exposes metadata: regions (wells), fovs_per_region, channels (name, color), z_levels, n_t, frame_shape, dtype, wellplate_format.
  - Grid like plate, thousands of FOVs. Lazy plane reads, never the whole plate in RAM.
  - One FOV per well is current scope (a well is a condition). Multi FOV handled by the FOV assembly strategy.
- Collaborators: ReadAcquisition builds it. Iterator and the panes consume it.

### AcquisitionImage

- Purpose: a processed result for one well (or the plate), carried through the layer stack.
- Requirements:
  - Native dtype, shape (T, C, 1, Y, X) for a z reduced result.
  - Holds or references its OperationStack.
- Collaborators: produced by an Operation via the Iterator. Written by WriteAcquisition. Shown by the panes.

### OperationStack (was OperationStorage)

- Purpose: an ordered, toggleable set of layers applied to the base image.
- Requirements:
  - Layer 0 is the base (raw preview). Later layers are applied operations.
  - Each layer: the Operation, its result reference, and an enabled flag.
  - Enable, disable, and reorder any layer. A Layers tab in the Processing pane drives this.
  - v1 usually holds base plus one operation. The structure supports more.
- Collaborators: owned by AcquisitionImage. Edited from the Processing pane Layers tab. Rendered by ArrayViewer and PlateView.

### Operation (abstract, the Strategy)

- Purpose: one post processing operation. Registered so the console, menu, and CLI grow automatically.
- Requirements:
  - Metadata: key, label, blurb.
  - Implementation (backend): the transform the Iterator runs. For z reductions this is a reduce over planes, streaming and bounded.
  - Interface (UI): build_tab(context) returns the operator tab for the Processing pane.
  - A new operator is one registry entry plus these two halves.
- Collaborators: run by the Iterator. UI hosted by the Processing pane. Emits results to PlateView and ArrayViewer.

#### Concrete operations

- MIP
  - Backend: max over z per channel. Streaming (running max), bounded (two planes).
  - UI: pick output folder, run over the whole plate.
- ReferencePlane
  - Backend: pick each well's sharpest z by Tenengrad focus. Streaming, bounded (best plane plus current).
  - UI: print the recommended plane (Tenengrad). User can override to the plane they see best in the viewer.
- Stitcher (stub)
  - Backend: multi FOV. Adapt maragall/stitcher into the high throughput path.
  - Scope: interim is randomly pick one FOV per well and say so in the GUI. Target is stitch then reduce then one composite per well.
- VideoPlayer (Record)
  - Backend: assemble an already acquired axis into an .mp4, or feed the slider. Streaming (one frame in RAM).
  - Parameters (pydantic model):
    - channel: composite, or a single channel index.
    - z: a single z index, or the result of flattening first (MIP or ReferencePlane). You must flatten z before recording.
    - fps: int. Sets the playback speed (mutates the time slider) and the encode rate. Independent of frame count.
    - export: bool to write .mp4, with filename (str) and output directory.
  - Axis: default T (time lapse), auto detected. Z fallback only when there is no time series.
  - UI: user clicks record a well. If z is not flattened, prompt to run MIP or ReferencePlane first. Opens its tab in the Processing pane.
- MinervaAuthor (stub, hidden from user)
  - Backend: hand the processed plate to Minerva Author. Locally hosted. Out of v1.
- NautilusAgent (stub, out of scope)
  - Backend: locally hosted agent that builds the operator you ask for. Out of v1.

## Engine

### Iterator (was iterator_engine)

- Purpose: run one Operation over each well (tile) of the acquisition. The single engine.
- Signature: Iterator(acquisition, operation) yields AcquisitionImage per well.
- Requirements:
  - Streaming with a bounded in flight window (workers times one well), flat in plate size.
  - Parallel workers. Same result as single thread.
  - Fault isolation: a corrupt or missing well is skipped and logged, never aborts the run.
  - No per operator subclass. The Operation is the Strategy.
- Collaborators: driven by the CLI and by the GUI Processing pane. Reads via Acquisition. Output to WriteAcquisition and the panes.
- FOV assembly strategy (the one real variance):
  - SingleFov: use one FOV per well (current).
  - Stitching: stitch FOVs, then reduce (target, via the Stitcher stub).

## IO

### ReadAcquisition

- Purpose: build an Acquisition from an input directory. The GUI does not read files directly.
- Signature: ReadAcquisition(input_dir) returns Acquisition.
- Requirements: read only, all Squid formats, fail loud on a non well plate.

### WriteAcquisition

- Purpose: persist an AcquisitionImage to an output directory the user picks.
- Signature: WriteAcquisition(acquisition_image, output_dir).
- Requirements:
  - Navigable multiscale OME-Zarr plate (re openable here or in any OME-Zarr tool).
  - Optional per plane TIFF export (off by default, doubles disk).
  - .mp4 for VideoPlayer.
  - Disk pre flight guard. Nothing written into the input folder.

### CLI

- Purpose: run the Iterator headless, high throughput.
- Signature: squidxplorer(input_dir, operation, output_dir, options) drives Iterator(acquisition, operation) then WriteAcquisition.
- Requirements:
  - pydantic-settings params (field docstrings become help). Validate up front (dir exists, known operator) before any disk write.
  - Resilient: skip a bad well and report the count.

## GUI

### GUI

- Purpose: present an Acquisition as a Qt window. Delegates work to the Iterator and IO.
- Signature: GUI(acquisition).
- Layout, three panes. Tabs live only inside the top left pane, not a global strip.
- Collaborators: the three panes below plus the Observer seam between them.

### Processing pane (top left)

- Purpose: pick and configure operators. This is where operator UIs live.
- Requirements:
  - Tabs, black on black, thin white outline. A Home tab that cannot be closed, printing options as a scrollable stack of text blocks.
  - Each operator opens its own tab. Composite of tabs, each built by its Operation (Factory Method).
  - An operator tab has an event handler that talks to PlateView and ArrayViewer (Observer).
  - A Layers tab toggles the OperationStack.
- Tabs:
  - HomeTab: the operator list plus roadmap (Minerva, Nautilus) and Open CLI.
  - ZProjectionUI: print the recommended reference plane (Tenengrad). Run MIP or ReferencePlane over the plate.
  - VideoPlayerUI: record a well. If z is not flattened, prompt to flatten first. Scope, fps, output folder, export .mp4.
  - CliUI: a preset terminal view. Presents the CLI commands.
  - LayersTab: toggle and reorder the OperationStack.

### PlateView (bottom left)

- Purpose: the low resolution plate navigator.
- Requirements:
  - One hue coded dot per well: grey not processed, amber processing, none when done, red x failed. Dot size capped (absolute).
  - Red box on the current well. Red dot where the cursor is.
  - Wheel zoom and drag pan. Double click navigates to a well.
  - Upsample on zoom: read a finer pyramid level as you zoom in. Memory frugal (streams the level that fits the view).
  - The plate title shows the acquisition name and the current mode (raw, or the operator).

### ArrayViewer (right)

- Purpose: the per well detail. A modded ndv (maragall/ndviewer_light).
- Requirements:
  - Push feed: each well's processed plane is pushed to a growing FOV slider (register_array, in memory, LRU bounded, sourced from the persisted pyramid at scale). The slider is the plate navigator.
  - Moving the FOV slider moves the red box on PlateView. Double click on the plate moves the slider.
  - Fast scrub: reuse the ndv ArrayViewer, do not rebuild it per position.
  - Collapsing sliders: z slider hidden when nz is 1 (a z reduced result). t slider hidden when nt is 1.
  - All controls live inside ndv (fps, play, subset, z, channel, fov). We upstream to ndv, no external controls.
  - Mutated by operator events (Observer).

## Out of scope (v1), with why

- Recording Z as a video axis: in HCS you reduce Z (MIP or reference plane), you do not scrub Z as a deliverable.
- Per frame stage position capture: we are post acquisition, positions are already on disk, there is no live stage to query.
- Live capture and stage motion: this tool never controls the microscope.
- MinervaAuthor and NautilusAgent: stubs, shown as roadmap only.

## What is easy to miss (flagged)

- Acquisition vs AcquisitionImage: input model vs processed result. Keep them distinct.
- The Observer seam: operator UIs, PlateView, and ArrayViewer coordinate through Qt signals. Name it, keep it thin (do not grow a Mediator god object).
- Contrast model: global per channel running histogram for the plate montage. Display only.
- Disk size guard: estimate before Write, conservative (over estimate only).
- Fault isolation: skip a bad well, report at the end.
- FOV assembly strategy: the one place multi FOV variance lives.
- Packaging: not an object, but a deliverable. Freeze to Linux AppImage, Windows, macOS with a dependency import smoke test.
