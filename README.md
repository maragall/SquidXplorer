# MIP tool

High throughput maximum intensity projection (MIP) for Squid well plate acquisitions. It opens a
finished acquisition, flattens each well's z stack into one image across the whole plate, and saves a
result you can reopen here, in napari, or in FIJI. Read only, it never changes your acquisition.

## What it does

- Opens a finished Squid well plate acquisition.
- Flattens each well's z stack into one max intensity projection (MIP), across the whole plate.
- Saves the result as a plate you can reopen here, in napari, or in FIJI.
- Read only. It never changes your acquisition and never runs the microscope.

## Setup (one time, Windows)

- You need Python 3.10, 3.11, or 3.12.
- If you do not have Python, install it from https://www.python.org/downloads/ . In the installer, tick "Add python.exe to PATH".
- Open PowerShell in the tool folder and run:
  - `powershell -ExecutionPolicy Bypass -File scripts\Setup-Windows.ps1`
- This puts a "MIP tool" shortcut on your Desktop.
- To update later: go into the folder and run `git pull`, then open the icon again.

## Open an acquisition

- Double click "MIP tool". A small black console opens next to it. That is normal, it shows progress. Closing it quits the app.
- Use the menu: File, then Open acquisition folder.
- Pick the acquisition folder (the one holding the 0 folder and/or the ome_tiff folder).
- It reads both Squid formats (individual TIFFs and OME-TIFF), on 384 and 1536 plates.

## The window

- Left: the buttons (run MIP, open CLI, layers).
- Bottom left: the plate. Grey dots are empty wells, so you always see the full plate shape. Scanned wells show their image.
- Right: the detail viewer for the well in view. It has its own controls: play and frames per second, a channel subset, and z (focus), t (time), and FOV sliders.
- Double click a well to open it on the right. The red box marks the well in view.

## Run MIP

- Click "Maximum Intensity Projection".
- Preview first (nothing saved): set "First N wells", click Preview. Good for a quick look before doing the whole plate.
- Whole plate: choose an output folder, click "Run on the whole plate".
- "Focus reference plane" jumps the z slider to the sharpest plane of the well in view.
- "Return to raw view" goes back to the unprocessed plate.

## The result

- MIP writes a plate folder named `<acquisition name>.hcs`.
- That folder is your result. To look at it again later, open the "MIP tool" and use File, then Open a computed MIP, and pick that .hcs folder.
- It also opens in napari or in FIJI.
- For plain TIFFs you can open directly in FIJI, run from the command line with `--tiff` (it adds a tiff folder next to the plate).

## Command line (optional)

- Click "Open CLI" for a terminal inside the app, or use your own PowerShell.
- Helpful commands:
  - MIP the whole plate and save FIJI TIFFs:
    - `python -m squidmip "C:\path\to\acquisition" --tiff`
  - Try the first 8 wells first (quick):
    - `python -m squidmip "C:\path\to\acquisition" --limit 8 --tiff`
  - Choose where to save:
    - `python -m squidmip "C:\path\to\acquisition" --tiff --output-folder C:\Users\you\Downloads`
  - See all options:
    - `python -m squidmip --help`

## Open in Minerva Author (optional)

- In "Process wells", click "Open in Minerva Author". It exports the well you have selected and
  starts Minerva Author on it.
- You get two files per well: a `.ome.tiff` (the image) and a `.story.json` (the colours and
  contrast). They go to a `minerva_export` folder in your home directory unless you choose another.
- Minerva Author cannot be pointed at a file automatically, so when it opens, click "Select File"
  and pick the `.story.json` the tab shows you. Use "Copy story path" or "Show in folder" if you
  need to find it. Your channel colours are already applied.
- Exporting works on its own. Opening Minerva Author needs Minerva installed: point
  `SQUIDMIP_MINERVA_HOME` at an `explorer` checkout that has already run its `setup.py`. Without
  it the export still succeeds and the tab tells you where the files are.
- One file per well. Combining several wells into one image needs the stitcher, which is coming
  later.

## Good to know

- Wells with more than one FOV: for now it uses the first FOV per well. Full multi FOV support (for example with the stitcher) is coming soon.
- It never writes into your acquisition folder. Results go only where you point them.
- Memory stays low: it holds at most one well at a time, so even a 1536 plate opens fine.
