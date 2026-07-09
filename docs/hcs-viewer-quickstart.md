# MIP tool, quick start

A desktop app for reviewing a **finished** Squid well-plate acquisition and processing it (MIP,
best-focus reference plane, or an .mp4 movie) without hand-tracking which files came from which well.
It reads your data **read-only** and writes results only to a folder you choose.

> Post-acquisition only. It opens data already on disk. It never controls the microscope.

---

## 1. Install (one time)

Python 3.10+.

```bash
conda activate ndviewer_light          # or your env with PyQt5 + ndviewer_light
cd /path/to/SquidMIP
pip install -e ".[gui]"                 # squidmip + the GUI extra (PyQt5, ndviewer_light, imageio)
```

Check it imports:

```bash
python -c "import squidmip, ndviewer_light; print('ok')"
```

---

## 2. Open the viewer

```bash
python -m squidmip._viewer                       # opens empty, then drag a folder in
python -m squidmip._viewer /path/to/acquisition  # or open one straight away
```

To open an acquisition once the window is up: **drag its folder onto the window**, or pass the path
as above. An acquisition folder is the one holding the numbered timepoint dirs (`0/`, `1/`, ...).

### Windows: one-command setup + Desktop shortcut

In Windows PowerShell, from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\Setup-Windows.ps1
```

This finds conda (even if it is not on PowerShell's PATH and you never ran `conda init`), creates the
`squidmip` environment if it is missing, and adds a **MIP tool** icon to the Desktop. The shortcut
launches the app with no console window and with the conda env activated (required on Windows, or Qt
fails to load). Double-click **MIP tool**, then drag an acquisition folder onto the window.

The window has three panes:

- **Top left, Process wells.** Pick an operator to run. Home tab lists the operators plus an
  Open CLI button and a Layers button. Each operator opens its own tab.
- **Bottom left, plate view.** One cell per well, laid out like the real plate.
- **Right, detail viewer.** The embedded ndviewer. Shows the z-stack of the well in view.

---

## 3. Navigate the plate

- **Double-click a well** to open it in the detail viewer on the right.
- **FOV slider** (inside the detail viewer) scrubs across every well of the plate. Each well loads
  its image on demand and is cached, so revisits are instant.
- **Play button** on the FOV slider auto-advances through wells. While playing it loads only the
  plane you are viewing, so it stays responsive.
- **z / t sliders** (inside the detail viewer) move through focus and time. The z slider disappears
  when there is nothing to scrub (a projected result has one z).
- Plate view: the **red box** marks the well in view, a **red dot** follows your cursor. Wheel to
  zoom, drag to pan.
- Dots appear on the plate **only while an operator is running**: amber = processing, red x = a well
  that failed and was skipped. A clean or finished plate shows no dots.

---

## 4. Run MIP or reference plane

Click **Maximum Intensity Projection** or **Reference plane** in Process wells. Its tab opens.

Two ways to run:

- **Preview (subset).** Set "First N wells" and click **Preview**. It computes those wells and
  streams the result into the plate and the detail viewer, writing nothing to disk. Leave
  "Save previews to disk" unchecked to just look. This is the cheap way to test an operator before
  committing the whole plate.
- **Whole plate.** Choose an output folder (it estimates the disk needed and refuses if it will not
  fit), then click **Run on the whole plate**. It writes a navigable OME-Zarr plate you can reopen
  here or in any OME-Zarr tool.

As each well finishes, its result appears in the detail viewer's FOV slider (z collapses to the
single projected plane) and its tile fills in on the plate.

- **MIP** takes the brightest value across z per pixel.
- **Reference plane** picks each well's sharpest z (Tenengrad focus) and shows that single plane.

---

## 5. Layers

Click **Layers** (Process wells) to toggle and reorder what the plate shows: the raw preview plus
each operator you have run. The topmost enabled layer is what the plate renders.

---

## 6. Record a movie

Click **Record video (.mp4)**. Pick scope (current well or every well), playback fps, and an output
folder, then **Record .mp4**. One movie per well. It runs in the background, so the window stays
responsive while it encodes. By default it records time (T); tick "Record Z focus sweep" to record
the z sweep instead.

---

## 7. Run it headless (the CLI)

Same engine, no window. Good for batch and for feeding FIJI. Open the **CLI** tab in Process wells
(Open CLI) for a live terminal inside the app, or use any terminal.

### See every option

```bash
squidmip --help
```

### Run MIP and save a slice you can open in FIJI

This is the common one. It runs MIP on the first 8 wells and writes them to your Downloads:

```bash
squidmip "/path/to/acquisition" --limit 8 --tiff --output-folder ~/Downloads
```

- It creates `~/Downloads/<acquisition-name>.hcs/` with two things:
  - `plate.ome.zarr/` a navigable multiscale plate.
  - `tiff/` plain TIFFs, one per well/channel/timepoint. **Open these in FIJI** (File, Open).
- Drop `--limit 8` to do the whole plate. Drop `--tiff` if you do not need the FIJI copy.
- `--workers 8` tunes throughput. A corrupt well is skipped and reported, never aborts the run.

### Running it in your own Terminal (independently)

The in-app CLI tab is the easy path. To run the same thing from your own Terminal (Applications ▸
Utilities ▸ Terminal on a Mac):

```bash
conda activate ndviewer_light
python -m squidmip "/path/to/acquisition" --limit 8 --tiff --output-folder ~/Downloads
```

`python -m squidmip` is the command; everything after it is the same as above.

### View the result

Open the written plate straight back in the viewer:

```bash
python -m squidmip._viewer "~/Downloads/<acquisition-name>.hcs"
```

or reopen the raw acquisition and use **Preview** in the MIP tab to see it without writing anything.

---

## 8. Watching memory

While the GUI runs it prints a line every few seconds:

```
[footprint] peak 1993 MB, current 1186 MB
```

When you close the window (or if it ever crashes) it prints the final peak:

```
[footprint] FINAL peak RSS: 1993 MB  (window closed)
```

Memory stays flat in plate size: only the well in view plus a bounded cache is ever resident, the
plate lives on disk.

---

## Notes and limits

- One FOV per well is the current scope. Wells with more than one FOV sample the first and say so.
- The tool never writes into your acquisition folder. Results go only where you point them.
- Reopening a written `<name>.hcs` plate: it is standard multiscale OME-Zarr.
