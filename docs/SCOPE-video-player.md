# Simple post-acquisition video player — scope

Turn an already-acquired **time-series** into (1) an in-app **scrubbable/playable** movie in the ndv
viewer, and (2) an exported **`.mp4`** per well. Post-acquisition: it **reads frames from disk**; it
never captures from a camera or moves a stage (contrast Squid's live "Simple Recording" tab).

## Two halves
- **Player (in-viewer):** ndv's ArrayViewer plays through **T** at a chosen fps; frames are pushed via
  our `register_array` (built) → a growing, scrubbable slider. The play button + fps control live
  **inside ndv** (we upstream fps = IMA-190) — no external controls.
- **Recorder/export:** the engine reads each well's T frames (one channel or a composite), encodes →
  `<well>.mp4`. Same path headless in the CLI.

## Axes & params
- **Records T only** (time-lapse). **Z is reduced beforehand** (MIP or Reference-plane), never a video
  axis. **C** = a chosen channel or composite. X/Y already on disk — nothing captured.
- **Params exposed** (mirror the Squid Recording *feel*, remapped for post-acq): output folder, name,
  **playback fps** (a display/encode rate — N frames at F fps = N/F s, independent of frame count),
  channel/composite, scope (per-well default; whole-plate montage movie if cheap).
- **Dropped vs Squid live:** capture-fps throttle, time-limit, stage/position provider — all
  meaningless post-acquisition (the exact antipattern to avoid).

## Run by
The one engine: `project_plate` streams per-(well, t) with bounded memory; Record encodes per well.
No new execution model.

## Scripts it leverages
**Ours (SquidXplorer):**
- `squidxplorer/reader.py` — read T frames per (well, fov, z, channel, t).
- `squidxplorer/_engine.py` + `squidxplorer/projection.py` — stream wells; reduce Z (MIP/reference) before T.
- `squidxplorer/_viewer.py` — Record tab UI + push frames to ndv for the in-app player.
- `squidxplorer/_cli.py` — headless export.
- `squidxplorer/_video.py` — **new**, thin mp4 encoder.
- `ndviewer_light/core.py::register_array` — **built**, the in-app growing/scrubbable player.

**New dependency:** `imageio-ffmpeg` (tried-and-true mp4 writer; bundled ffmpeg, no system install).

**Squid — REFERENCE ONLY (read for axes/params/UX; NOT imported):**
- `control/core/job_processing.py` — `WriteIndividualImages` / image-saver: on-disk frame layout +
  `frames.csv` manifest (what per-frame metadata is meaningful: frame_id, timestamp, channel, exposure).
- `control/core/stream_handler.py` — the save-fps throttle (informs our *playback*-fps semantics).
- `control/widgets.py::RecordingWidget` + `control/gui_hcs.py` + `control/_def.py::ENABLE_RECORDING`
  — the tab's UI shape (path / name / fps / record) whose *feel* we mirror.
- **ndv** — upstream target for the fps/play controls (IMA-190), so all controls stay inside ndv.

## Out of scope
Live capture, stage motion, recording Z as a video axis, per-frame stage-position capture.
