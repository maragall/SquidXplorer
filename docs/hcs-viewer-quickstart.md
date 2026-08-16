# SquidXplorer — quick start

A local viewer for finished Squid HCS acquisitions. Open a plate, explore any well in a napari
view, and run your processing operators on exactly the wells you pick.

**It is read only.** It never changes your acquisition and never runs the microscope.

---

## 1. Install (once)

Download the installer for your platform from the repository's latest **build-installer** run
(GitHub → Actions → build the installer) and run it. One file per platform, no Python needed:

| platform | file | run it |
|---|---|---|
| Windows | `SquidXplorer-Setup.exe` | double-click |
| macOS (Apple Silicon) | `SquidXplorer-Setup.zip` → one executable | first launch: right-click → Open (unsigned) |
| Linux (x86_64) | `SquidXplorer-Setup-x86_64.AppImage` | `chmod +x`, then double-click |

The installer shows a checkbox menu of operator packs, installs into its own private
environment, and leaves a double-clickable **SquidXplorer** launcher: a desktop shortcut on
Windows, `~/Applications/SquidXplorer.app` on macOS, an application-menu entry on Linux.
`scripts/installer/README.md` has the details, including what is and is not signed.

To update later: download and run the newer installer. It reuses the same environment.

---

## 2. Open an acquisition

An acquisition folder is the one holding the `0` folder and/or the `ome_tiff` folder.

Three ways in, fastest first:

1. **Drag the folder onto the launcher** — the Windows desktop shortcut, the macOS app (or its
   Dock icon), or the Linux menu entry.
2. Open the launcher, then **drop the folder onto the plate**.
3. Open the launcher, then **File → Open acquisition folder**.

A small console window opens alongside. That is normal — it shows progress, and it stays open if
something goes wrong so you can read the error.

> The first open builds a thumbnail for every well. Opening the same acquisition again is a cache
> read and is much faster. Thumbnails live in your own cache folder, never in the acquisition.

---

## 3. What you get: two windows

| | |
|---|---|
| **Left, narrow** | the **plate** — every well, plus the Window navigator and bulk operators |
| **Right, wide** | **SquidXplorer views** — a napari view of the wells, in tabs |

A view over every well opens by itself, so there is something to look at immediately. The two
windows are sized to sit side by side on your screen.

---

## 4. Move around by clicking the plate

**Left-click a well and the current view jumps to it.** That is the main gesture — the plate is
your navigator and the big window is where the pixels are.

You can click any well, including ones outside what the view was opened over; it will follow you
there.

Other gestures on the plate:

| gesture | what it does |
|---|---|
| **left-click a well** | show that well in the current view |
| **click empty space** | clear the selection |
| **Shift-drag** a box | open a new view over the wells you boxed |
| **Shift-click** / **Ctrl-click** | add or remove one well from the selection |
| **Select all**, then **Open view** | open one view over the whole plate |
| **double-click** a well | open a new view over just that well |
| **mouse wheel** | zoom the plate; drag to pan |

Selecting wells and **navigating** are separate on purpose: clicking a well to look at it does
**not** change the selection your operators will run on.

---

## 5. Tabs

Every view you open becomes a **tab** in the views window. You work in one at a time.

- **Click a tab** to switch. The plate highlights that view's wells.
- **× on a tab** closes that view.
- **Drag a tab out of the strip** to pull it into its own floating window.
- The **Window navigator** on the plate lists every view and says which window each one is in.

Each view holds its own napari viewer, so several open at once costs real memory. Past six views
the views window tells you roughly how much — close the ones you are done with.

---

## 6. Run an operator

Operators run your own tested implementations, never a reimplementation.

**On one view:** pick an operator in **"Operators for this window"**, set anything you need under
**controls**, and click **Run**. Results arrive as layers you can toggle on and off; the raw data
on disk is untouched.

**On a selection:** pick the wells on the plate, then use the operator panel under the plate to run
across all of them.

Available: maximum intensity projection (built in), deconvolution, stitch and flat-field,
background subtraction, and nuclei detection. The ones backed by separate packages need the
matching installer checkbox (or `pip install` extra).

Tick **save** to write results to disk as well as show them.

---

## 7. Where results go

Saved results are written as a plate folder named `<acquisition name>.hcs` next to your chosen
output location. Reopen one later with **File → Open a computed MIP**, or in napari or FIJI —
it is OME-Zarr, not a private format.

---

## If something looks wrong

- **The plate says "not a readable Squid acquisition"** — you probably picked a level too high or
  too low. Choose the folder that directly contains `0` and/or `ome_tiff`.
- **A view is blank** — check the region slider at the bottom of the view; you may be on a well
  that was not acquired. The status line names the region it is showing.
- **The plate is behind everything** — it is the small window; the views window is the big one.
- **Read the console.** It is the window that opened beside the app, and it keeps its last error.
