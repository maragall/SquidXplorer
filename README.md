# SquidXplorer

A local viewer for finished Squid HCS acquisitions. Open a plate, explore any well or region in its
own napari window, and run your processing operators on exactly the wells or ROIs you pick. Read
only: it never changes your acquisition and never runs the microscope.

## The idea

The **plate is the root**. Selecting wells opens an independent napari window over them; drawing an
ROI inside a window opens a **child window** over that region. Every window gets an integer id, and
they are all collected in the **Window navigator** on the left.

![SquidXplorer layout: the wellplate root with the Window navigator and bulk Operators, each selection opening a window with its own 2D/3D and Operators panel over a napari mosaic, and ROI child windows.](docs/viewer-layout.png)

**Layout (from the design deck):**

- **Root**: the Window navigator (a selectable list of open views) and the bulk Operators, above the
  Wellplate view with its Selection.
- **Each window**: a 2D / 3D control and Operators for that window, over the napari mosaic of the
  full well, with ROI boxes you can send to 2D or 3D, and a region slider `<> A1, B6, C3 ...`.
- **An ROI child window** is the same, with a slider over its ROIs `<> ROI1, ROI2, ROI3 ...`.

**Notes:**

- Each selection from the wellplate opens a new region view. Each window gets a positive id,
  collected in the Window navigator on the left.
- Windows not currently being manipulated halt their draw and refresh.
- A memory bar warns you before the system runs low.

## Operators

Processing runs **your own tested implementations, called directly**, never a reimplementation:

| Operator | Backend |
| --- | --- |
| Deconvolution (Richardson-Lucy, vectorial PSF) | `petakit` |
| Stitch and flat-field | `tilefusion` |
| Background subtraction | `bgsub` |
| Nuclei detection | Cellpose |
| Maximum intensity projection | built in |

Output is byte-identical to the standalone repos, pinned by `tests/test_operator_fidelity.py`.
Results are OME-Zarr layers you toggle on and off; the raw data on disk is never touched.

## Install (one time)

ONE installer file per platform, no Python needed: download `SquidXplorer-Setup.exe` (Windows),
`SquidXplorer-Setup.zip` (macOS arm64) or `SquidXplorer-Setup-x86_64.AppImage` (Linux) from the
latest **build-installer** Actions run and run it. It installs into its own private environment
and leaves a double-clickable **SquidXplorer** launcher (desktop shortcut / `~/Applications`
app / menu entry). Details, including what is unsigned: `scripts/installer/README.md`.

To update later: download and run the newer installer. It reuses the same environment.

Developers working from this checkout: `pip install -e .[gui]` into your own venv and run
`squidxplorer-view`.

## Open an acquisition

- Launch SquidXplorer. A small console opens beside it; that is normal and shows progress.
- **File, then Open acquisition folder**, and pick the acquisition (the folder holding the `0`
  folder and/or the `ome_tiff` folder).
- It reads both Squid formats (individual TIFFs and OME-TIFF), on 384 and 1536 plates.
- The first open downsamples every well to build the plate view; **opening the same acquisition
  again is a cache read** (measured: 15.2 s to 0.08 s on a 1536-well plate). Those thumbnails go
  in your own cache folder, never into the acquisition: `~/Library/Caches/squidxplorer` on macOS,
  `%LOCALAPPDATA%\cephla\squidxplorer\Cache` on Windows, `~/.cache/squidxplorer` on Linux. Delete that
  folder any time to reclaim the space (one 91 MB file per 1536-well plate); it rebuilds itself.
  `SQUIDXPLORER_CACHE_DIR` moves it, and `SQUIDXPLORER_PLATE_CACHE=0` turns it off.

## Explore

- **Click** a well to select it, **Shift-drag** a box or **Shift/Ctrl-click** to select several,
  then **Open view** to open them as one window.
- Inside a window: **2D / 3D**, draw an **ROI** and send it to its own child window, and run an
  **operator** on that view.
- The **Window navigator** lists every open view; select rows to highlight their wells on the
  plate, and **Collapse all** when the desktop gets busy.
