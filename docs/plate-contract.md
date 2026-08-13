# The plate contract: what SquidXplorer writes, and what it promises about it

`plate.ome.zarr` is written by `squidxplorer/_output.py` and read by `squidxplorer/reader.py`,
`squidxplorer/_tilesource.py` and `squidxplorer/_viewer.py`. We are both producer and consumer, which is
exactly why this document exists: an internal convention that nobody wrote down drifts, and the
drift is not visible until something renders wrong.

It drifted. From IMA-217 until 2026-07-29 the reader's own contract prose said, in two places,
that "SquidXplorer's writer emits no translation". The writer had been emitting one for months, and
`translation` is the PRIMARY mechanism that places every FOV on the plate. A maintainer reading
those lines would have concluded the live mechanism was dead. That is the defect that pulled this
document into v1 scope (Julio, 2026-07-28), rather than leaving it until a second implementation
reads our output.

The document is split in two, and the split is the useful part:

- **Stable.** Depend on it. It changes only with a MAJOR bump of `PLATE_CONTRACT_VERSION`, and a
  reader that meets a different major **refuses to open the store**.
- **Optional.** Each entry names its fallback INLINE. Every one of these fallbacks already existed
  in code before this document; none was written down as a guarantee, which is how one of them
  ended up implemented as a bare `except Exception` in `_viewer.py`.

Machine-checkable half: `squidxplorer/contract/`. Validate any plate with

    python -m squidxplorer.contract.validate /path/to/plate.ome.zarr

If this document and that code ever disagree, this document is the contract and the code is the
bug.

## The version

`squidxplorer/contract/version.py` holds `PLATE_CONTRACT_VERSION`. It is stamped ONCE, on the plate
group, at `zarr.json -> attributes -> squidxplorer -> plate_contract_version`, deliberately outside
OME's `attributes.ome` namespace (that namespace belongs to the spec and is what
`ome-zarr-models` validates).

It is not the same thing as `_output._NGFF_VERSION` ("0.5"), which is the OME-NGFF **spec**
version and belongs to OME. Two stores can both be valid NGFF v0.5 and disagree on everything
below.

| Bump | Means | An older reader |
| --- | --- | --- |
| MAJOR | something in **Stable** moved | **refuses**, naming both versions |
| MINOR | **Optional** gained an entry | warns, reads correctly, reads less |

What a reader does on mismatch, and the reasoning in one line each:

| Declared | Action |
| --- | --- |
| absent | **proceed.** Every plate written before 2026-07-29, and every third-party NGFF store, is unstamped. Refusing them would reject the installed base to enforce a rule invented after it. |
| same major, same or older minor | proceed |
| same major, newer minor | **warn**, then proceed. Optional content we were not built to use is ignored: a lossy read, not a wrong one, and the loss is announced. |
| different major | **refuse, loudly.** A stable guarantee moved, so the store would still open, still find its wells, and place them wrongly. |
| unparseable | **refuse.** Something deliberately made a promise and we cannot tell which one. |

This mirrors `reader._parse_fov_positions_um`, which already refuses "to place FOVs at positions
that would look plausible but be wrong" rather than guessing. Same judgement, one level up.

## Stable

### The group hierarchy

    plate.ome.zarr/                     plate group: rows / columns / wells
      {row}/                            row group (bare, structural)
        {col}/                          well group: well.images -> field paths
          {fov}/                        image group: multiscales (+ omero)
            0/                          array, full resolution
            1/ 2/ ...                   optional coarser levels

`{row}` and `{col}` are the row NAME and the column NAME, never zero-padded: `B2` is written
`B/2`, never `B/02`, so the region id is the plain concatenation of the two
(`_output.parse_well_id` and its exact inverse in `reader._discover_hcs`).

`{fov}` is the RAW acquisition FOV id, taken verbatim from `well.images[].path`. It is not a
re-indexed 0..n-1 field number, so a non-contiguous FOV set stays faithful. The spec permits any
alphanumeric field path; use the listed paths, do not assume a range.

**There are exactly two legitimate ways to get to a field, and inventing the path is neither.**

1. **Descend the metadata.** `plate.wells[].path` -> `well.images[].path` ->
   `multiscales[0].datasets[].path`. Every name is read, none is assumed. `_montage._PlateLayout`
   and `_tilesource.plate_layout_from_store` do this, and it is why they cope with FOV ids and
   level names they have never seen.
2. **Call `squidxplorer.contract.field_path(base, wellpath, fov, level)`** when you already hold the
   parts, which is the case on every read path in the viewer.

Before 2026-07-29 the path was reconstructed by f-string at four sites in `_viewer.py`, three of
which handed the result straight to a store open and never went near `reader.py`. Four copies of
one rule is four chances for it to change in three places, and folding them into one seam is what
makes the version gate above enforceable rather than decorative.

One shortcut survives and is worth knowing about: `_montage` and `_tilesource` open a level by
`field_dir / str(level_index)` rather than by the recorded `datasets[].path`. That is correct for
every store SquidXplorer writes (it names levels `"0"`, `"1"`, ...) and would break on a conforming
store that names them otherwise. It is a NAME per the spec, not an index.

### Axis order: TCZYX

Every array is 5-D `(t, c, z, y, x)`, and `multiscales[0].axes` says so in that order. Squid's
canonical order, `_zarr_store.create_array`'s `dimension_names`, and what every read path in this
package assumes.

The non-HCS Squid layout `zarr/{region}/acquisition.zarr` is 6-D `(fov, t, c, z, y, x)`. That is
Squid's shape, not ours: we read it, we never write it. See `reader._discover_flat`.

### Z: the axis is real, and `Nz > 1` is written (IMA-277, 2026-08-05)

`z` has always been a real axis of the store — `_output._multiscales` scales it by the
acquisition's `dz_um` — but until IMA-277 nothing could put more than one plane in it:
`_output._validate_image` refused any array with `shape[2] != 1`, and `_stitch.stitch_region`
refused any plane-op outright. Between them that meant **no path in this system wrote a Z>1
result**, which is why five of the eight registered operators could be run and displayed but
never persisted.

What is guaranteed now:

| Operator kind (`consumes`) | Written `Nz` | Operators |
| --- | --- | --- |
| z-reducer `{"z"}` | `1` | `mip`, `reference`, `decon3d` |
| plane-op `set()` | the acquisition's `n_z` | `bgsub`, `decon`, `flatfield`, `spot`, `cellpose` |

The depth comes from the operator's own declaration, in one place (`write_plate` reads
`operator_consumes`), so the disk pre-flight estimate and the bytes actually written cannot
disagree. A z-reduced write is byte-identical to what this writer has always produced.

Two consequences worth knowing:

* `_write_field` writes **one z plane at a time** and builds that plane's pyramid alone. The
  pyramid only halves Y and X, so this is pixel-identical to building the whole volume's — and it
  is the only version that fits: a 10-plane 4-channel fused mosaic of a 27-FOV 10x well is 8.79 GB.
* The individual-TIFF export's `{z}` filename field now carries the plane index it always named
  (`{region}_{fov}_{z}_{channel}.tiff`), instead of a hardcoded `0`.

**Where Z>1 is still flattened, and it is the RENDERER, not the store.**
`_tilesource.InMemoryMultiscale._planes` takes `arr[t, :, 0]` — plane 0 — when it folds a field
into the plate overview. That is pre-existing and unchanged by IMA-277 (plane-op results already
reached it through `project_plate`); it flattens the DISPLAY, never the written pixels. A viewer
that shows a z slider is separate work.

### Time: the format carries it, and the viewer now reads it

This section used to state a GAP. As of 2026-07-29 it states a guarantee, because the gap was
closed: it is kept in full because the shape of the old bug is worth remembering.

**What is guaranteed on disk.** `t` is the leading axis. The writer writes EVERY timepoint:
`project_well` is called with `t=None` (`_engine.project_plate`), which yields `(T, C, 1, Y, X)`
with `T = n_t`, and `_output._write_field` writes that array whole. The individual-TIFF export
writes one file per timepoint (`tiff/{t}/...`). Nothing on disk is lost, and
`reader.metadata["n_t"]` reports the true count.

**What reads it, at 2026-07-29.** Every consumer now takes a timepoint and clamps it to the store's
extent, so a stale slider position cannot index off the end of a shorter re-ingest:

| Consumer | Timepoint |
| --- | --- |
| `reader.read(region, fov, channel, z, t)` | takes `t`, defaults to 0 |
| `_montage.render` | takes `t`, clamped |
| `_tilesource` (every source) | reads `t` off the `TileDescriptor`, clamped — see "deep zoom" below |
| `_workers._ComputedPlateWorker._read` | takes `time_point`, clamped |
| `_plate_overview` loupe source (`read_crop`, `coarse`) | takes `time_point`, clamped |

`TimePointBar` (`squidxplorer/_time_point.py`) is the control, mounted on the plate and in every window.
One widget CLASS for both, so the two can never disagree about what a timepoint control is, and a
separate INSTANCE each, because a window navigates independently: a shared position would make
comparing two wells at different timepoints impossible.

**Playback, 2026-08-05, and only in a window.** `TimePointBar(playback=True)` walks the timepoints
with napari's own play button, fps popup, loop modes and off-thread `AnimationThread` — the same
`_region_nav.AxisPlayback` the region slider uses, with the axis passed in. Two properties are
load-bearing and both are inherited rather than written:

* **The frame gate.** napari's playback is debounced on the render (`QtDims._set_frame` drops a
  frame while `dims._play_ready` is False). A frame here costs a mosaic load, so the gate is closed
  by the step and opened by `RegionViewer._frame_done` when the picture is on screen. Playback
  therefore self-limits to the rate the data can be read at; measured on `sim_5d_2x2_t3`, 10 fps
  requested yields ~4.4 fps achieved with the rest DROPPED and none queued.
* **Superseded loads are dropped, not waited for.** Every load carries a generation
  (`RegionViewer._load_gen`); a result from an older one is ignored on arrival. The region check
  cannot do this job, because a timepoint change keeps the region and differs only in `t`.

**The plate's own pixels follow its bar, 2026-08-05.** This section used to record the plate as the
one consumer that did not, and the reason was two failures that had to be fixed together:

* `_PreviewWorker` never passed a `t` to `reader.read`, so the plate previewed frame 0 whatever
  its bar said — the same "a signature is not a call" shape as the loupe, one axis over.
* its persistent cell cache was keyed `(token, region)` with no timepoint, so threading `t`
  through the read ALONE would have replayed timepoint 0's cells under a label saying timepoint 1.
  Worse than the bug, which is why the fix is one change and not two.

The cell is now identified by `(token, t, region)` end to end — RAM key, file name
(`t<t>-<region>-<sha>.npz`) and packed page (`plate-cells-t<t>.npy`, whose sidecar records its own
`t` and is checked on read). The timepoint is in the KEY and deliberately not in the TOKEN: the
token directory is what `prune_stale` deletes, so a timepoint in it would delete t=0's cells the
moment you stepped to t=1, and stepping back would re-read the plate. `_start_preview` is the one
place a preview is built and it reads `PlateWindow.time_point`, so the first ingest, the
exploration-tab re-scope and `_return_to_raw` (which is what a timepoint change calls) cannot
disagree.

`FORMAT_VERSION` went 1 -> 2 with that change. A v1 cell carries no timepoint; it was written by a
producer that always read frame 0, but the record does not SAY so, and a cell whose timepoint is
unknown must never be served under one that is. The version is hashed into the token, so every v1
entry is unreachable immediately and deleted by `prune_stale` on the first publish after.

MEASURED on `sim_5d_2x2_t3` (4 regions x 4 FOVs x 2 channels, 256 px), stepping t=0 -> t=1 -> t=0
with `tools/measure_plate_t_steps.py`, median of 9, a cold cell cache per repetition, both columns
run back to back:

| step | before | after |
| --- | --- | --- |
| t=0, first visit | 13.9 ms, 4 wells read | 13.6 ms, 4 wells read |
| t=1 | **6.4 ms, 4 cache hits — and the WRONG pixels** | 13.9 ms, 4 wells read |
| t=0 again | 5.3 ms, 4 cache hits | 7.0 ms, 4 cache hits |
| t=0 and t=1 differ | **False** | True |

The before column is the whole bug in one row: the second step was the FASTEST of the three,
because it answered t=1 with frame 0's cells. "A full plate re-read per tick" turned out to be true
only of an uncached plate (`SQUIDXPLORER_PLATE_CACHE=0`); warm, every tick after the first was a replay
of the first frame. After, a NEW timepoint costs a plate read, which is honest work, and a
REVISITED one is a cache hit — the property a play button would need.

**The plate still does not play, and that is now a product decision rather than a blocker.** This
document used to say re-keying the cell cache "is the price of a plate play button". The price is
paid; nobody has asked for the button, so `TimePointBar(playback=False)` stands on the plate and
`play()` on it raises rather than no-ops. Adding it is a widget flag and a frame gate, not a
correctness problem.

**Deep zoom, 2026-08-06: the timepoint is part of the TILE'S IDENTITY.** This section used to name
the tile path as a known gap — `PlateOverview.set_tile_source` built its `CompositePlateSource`
with a `PlateCellCache` at the default `time_point=0`, passed no `t` to the source, and
`set_time_point` touched neither `_tile_src` nor `_tile_cache`. The gap was wider than "coarse
rungs": the FOV rungs were frozen too, because `_tiling.TileDescriptor` carried
`level, key, channel, bbox_um` and no timepoint at all, so *nothing on that path could ask the
question*. Measured on `sim_5d_2x2_t3`, whose blob moves with `t`:

| | sha256[:16] of one FOV tile, ('A1', 0), 405 nm |
| --- | --- |
| plate at t=0 | `24d0d02da9674b3e` |
| plate after `set_time_point(2)` | `24d0d02da9674b3e` — **byte-identical**, while the plate reported t=2 |
| what t=2 actually is | `a265917cd7c338cf` |

`TileDescriptor` now carries `t`, **with no default**, and every source reads the frame off the
REQUEST rather than off itself: `ReaderTileSource` and `ZarrPyramidSource` no longer take a `t` at
construction at all. That is the same argument as `(token, t, region)` above, in the same place —
the key — and it buys the same thing: `TileCache` holds both frames, so stepping back to one
already seen is a hit rather than a re-read. A default of `0` is refused precisely because
`PlateCellCache.for_reader(time_point=0)` had one and `_plate_overview` simply never passed it.

Two guards keep a tile read at one frame from being drawn under another, which is the failure that
is worse than not moving at all:

* `TileCache._nearest_ancestor` matches `t` as strictly as it matches `channel`. That fallback is
  the ONE site licensed to draw a tile other than the one requested, so a `t`-blind ancestor there
  would have reintroduced the whole defect at the one place nobody looks.
* `InMemoryMultiscale` holds ONE frame's seeded cells and raises on a request for another;
  `CompositePlateSource` refuses a `PlateCellCache` whose `time_point` disagrees with its own
  (the refusal `_workers._PreviewWorker` already makes at the other end of the same cells), and
  routes a coarse tile at another `t` to the reader — real pixels at real cost, counted in
  `coarse_from_reader` — instead of serving what happens to be resident.

**The shape of the bug that was here, worth remembering.** The plate overview and the loupe read
`arr[0, :, 0]` unconditionally, so a 40-timepoint plate looked exactly like a 1-timepoint plate.
No error, no warning, nothing wrong on screen. It survived because **every fixture in the suite was
`Nt = 1`**, so the bug was invisible by construction rather than by oversight: the tests could not
have caught it whatever they asserted. What ended it was building a multi-timepoint fixture whose
every plane is filled with a value derived from its timepoint, so a test can name which frame it is
holding from one pixel.

The loupe's coarse cache is now keyed by `(well, timepoint)` rather than by well alone. It had been
keyed by well, so once a well was read at one timepoint every later timepoint got that same picture
back. A cache that answers the wrong question quickly is worse than no cache.

**The second half of that, found 2026-08-03.** The table above is a claim about SIGNATURES, and
`test_every_documented_read_site_takes_a_timepoint` checks exactly that. Every loupe read site took
a `time_point` and NOTHING passed one: `_LoupeWorker.request` had no such parameter, so the widget
could not have supplied it, and every source fell to its `time_point=0` default. The plate moved,
the inset did not, and no test could see it because a parameter that exists and is never supplied
reads as correct from either end. A signature is not a call. The widget now holds the plate's
timepoint (`PlateOverview.set_time_point`), the worker carries it (including in its LRU key), and
`tests/test_viewer.py::test_the_loupe_reads_the_timepoint_the_plate_is_showing` drives it through
the widget rather than inspecting a signature.

`python -m squidxplorer.contract.validate` still warns when a plate carries more than one timepoint.
That warning was the only thing saying so out loud while the gap existed; it is now belt and braces.

**The next axis, found the same day: the FIELD.** The Time section closes by naming the part that
generalises — every fixture was `Nt = 1`, so the bug was invisible by construction. The identical
shape was live one axis over. A loupe read addresses `(well, fov, level, rect, t)`, and `fov` was
not an argument at all: `_fov_of_well` returned the region's FIRST field for every position, while
the widget handed the read a position measured across the whole MOSAIC. On a multi-FOV region the
loupe therefore stretched the entire mosaic onto field 0 — real pixels, from a field the cursor was
mostly not over, with nothing on screen to say so. `read_crop` now takes a `fov`, the widget
resolves it from the mosaic boxes it already draws by (`PlateOverview._fov_box_at`, shared with the
double-click hit test), and `_fov_of_well` is what a caller that has not resolved one falls back to.
The contrast WINDOW stays per WELL (`coarse` still reads the region's first field) so brightness
does not lurch as the cursor crosses a seam.

### Level 0 is full resolution, and a MIP never comes from a coarser level

`datasets` is ordered highest resolution first. `datasets[0]` is the full-resolution,
pixel-exact array. Coarser levels are 2x2 block means (`_output._downsample_yx`) and exist for
NAVIGATION only.

`SquidZarrReader` serves level 0 and only level 0. A projection computed from a downsampled level
would be silently wrong, so the coarse levels are never read on the compute path.

Array SHAPES come from the arrays themselves, never from a scale factor: the 2x2 block mean crops
odd axes, so level shapes are `floor(prev/2)` and a scale factor would disagree.

### Micrometres, and the `_um` suffix means it

Every physical value this package exposes is in micrometres, and every key carrying one is
suffixed `_um` (`pixel_size_um`, `dz_um`, `fov_positions_um`).

On disk, `axes[].unit` is a UDUNITS-2 string. Conversion to micrometres happens at exactly one
producer, `reader._unit_to_um`, using the axis's own declared unit. No consumer compensates. A
store written in millimetres must never reach a `_um` key as millimetres; that is the 1000x class
of bug, and it has been fixed here once already.

### Corners, not centres

A dataset `translation` places pixel (0, 0): the field's TOP-LEFT corner in stage micrometres.
`fov_positions_um` records where the stage was, i.e. the frame CENTRE, so the writer subtracts
half a frame (`_output.field_origin_um`). Half a frame is 388 um on a 2084 px 20x field, which is
half an FOV of mosaic shear if it is skipped.

Every level of a field carries the SAME corner. Area-averaged downsampling nudges the sample
centre by half a coarse pixel; carrying that here would make one field's levels disagree with each
other while breaking the "levels share an origin" assumption every mosaic compositor makes.

## Optional, each with its fallback

### Dataset `translation` falls back to a sibling `coordinates.csv`

`translation` is the only position mechanism the NGFF spec defines, and SquidXplorer's writer **does**
emit it (IMA-217). Three cases legitimately carry none: an acquisition with no recorded stage
positions, a store written before IMA-217, and Squid's 6-D layout, where one translation covers a
whole region and therefore cannot be a per-FOV position.

**Fallback:** `coordinates.csv` beside the store, both Squid schemas (IMA-215). Either way the
result lands in `reader.metadata["fov_positions_um"]`.

**When both are absent:** the value is `{}`, present but empty, exactly as on the TIFF readers, so
consumers degrade to single-tile rendering instead of hitting a `KeyError`. Note what does NOT
happen: positions are never inferred from a scan-order index map. That trades a measurement for an
assumption, cannot express stage drift or autofocus jitter, and breaks freeform tissue
acquisitions. `reader.py` raises instead.

**Placement-source precedence, per layout.** Zarr: the OME `translation` first, the sibling
`coordinates.csv` as the fallback above. Multipage TIFF: the per-page inline `x_mm`/`y_mm` tags
first, then the CSV. Individual TIFFs: the CSV is the only source, and the EXECUTED one
(`0/coordinates.csv`, positions as visited) wins over the planned one at the acquisition root.
Squid's demotion of `coordinates.csv` upstream changes nothing here: where a translation exists it
is already preferred, and the TIFF layouts carry no translation, so for them the CSV is their only
measurement, not a deprecated choice.

### `omero` falls back to auto-contrast

`omero.channels` carries the label, hex colour and display window per channel. A legal NGFF image
need not have it.

**Fallback:** a sibling `acquisition_channels.yaml`, then generic `C{i}` labels with the shared
wavelength/brightfield colour resolution, and display windows from auto-contrast. The store still
opens; the colours are then a best effort, and `reader._channels` announces that rather than
passing it off as acquisition truth.

### A multi-level pyramid falls back to level `"0"`

Fields at or below 256 px in Y and X are written single-level on purpose
(`_output._PYRAMID_MIN_YX`), and a foreign store may carry one level for any reason.

**Fallback:** level `"0"`, which a field always has. `squidxplorer.contract.field_levels` is the one
place that reads `multiscales[0].datasets[*].path` and applies this fallback. Consequence, not a
defect: navigation pays full resolution for a thumbnail, and the loupe's level selection cannot
bound the read, so it strides in TensorStore instead.

### `.squidxplorer-incomplete` marks a store mid-write

Written from a plate write's first byte to its last (IMA-230) and removed as the write's last act.
Its presence means the run did not finish and wells the plate metadata promises may be absent.
`_output.is_incomplete` reads it. A store without the marker was either finished or written by
something that is not us.

### `tables/FOV_ROI_table` is a convenience, not a source of truth

An AnnData-encoded Fractal/ngio ROI table giving every FOV's box in micrometres (IMA-231), written
per well on the persist path only. Its corners are `field_origin_um`, the same corner as the NGFF
`translation`. Nothing in this package reads it back; it exists so an external tool can recover
FOV boundaries after a region is fused.

**Fallback:** none needed. If it is missing, use `translation`, which is where its numbers came
from.

## What is deliberately NOT in this contract

`record-zstack-viewer`'s contract also covers `events.jsonl`, a `thumbnails/` directory and a
`recording/` section. Those describe a LIVE producer. v1 is post-acquisition only: SquidXplorer is
launched after the acquisition finishes, so there is no growing store, no event log to tail and no
partial-write state machine. See `docs/VERSIONS.md`, which states the version arc and the rule that
unbuilt things carry a TRIGGER, not a version number. The trigger for the live half of a contract
is the first time SquidXplorer is pointed at a folder that is still being written.
