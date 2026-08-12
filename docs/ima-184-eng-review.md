# IMA-184 — Engineering Review (longform)

Slot 4 of the SquidXplorer state machine (189 → 183(+187) → 188 → **184**). Output: a canonical
multiscale OME-zarr HCS plate + an individual-TIFF export, written by streaming IMA-188's
`project_plate`. Landed on `main` via `--no-ff` merge; cross commit green on both datasets.

Public surface: `from squidxplorer import write_plate`.

## 1. Intent (asserted a-priori, before code)

- **Two outputs, two consumers.** (1) A canonical **OME-NGFF v0.5 HCS plate** that opens in
  **ndviewer_light** (the navigable plate view). (2) **Individual per-plane TIFFs** —
  You Yan's "individual tiff output" — a drop-in for Nick's existing Squid-reading workflow.
- **Lay on the canonical Squid format**, don't reinvent it. Match Squid's own
  `control/core/{job_processing,zarr_writer}.py` + `utils.parse_well_id` exactly.
- **Single resolution level, no pyramid.** Squid's writer emits one level; a per-FOV pyramid
  is pointless for one small field (plate-view thumbnails are IMA-193's concern, not this
  writer's). Supersedes the earlier plan's pyramid.
- **Stream, bounded memory.** Write each well as it arrives; never hold the plate.
- **Fail loud.** Refuse a non-`(T,C,1,Y,X)` frame, a channel-count mismatch, or a
  non-`<letters><digits>` region — a mislabelled field must never be written silently.

## 2. Inherited seam (verified against merged `squidxplorer/`)

- **Contract consumed:** `project_plate(reader, *, n_fovs=1, workers=None, projector="mip")
  -> Iterator[(region, fov, ndarray(T, C, 1, Y, X))]` (`_engine.py`), native dtype, TCZYX,
  Z=1, streaming, completion order, fail-loud. Consumed single-thread — the engine
  parallelises internally, so the writer needs **no locking**.
- **Metadata surface (`reader.metadata`):** `regions`, `fovs_per_region`,
  `channels[{name, display_name, display_color, ex}]` (`display_color` guaranteed — IMA-189
  raises on an unresolved channel), `pixel_size_um`, `dz_um`, `wellplate_format`,
  `frame_shape`, `dtype`, `n_z`, `z_levels`, `n_t`. Channel order == the array C-axis order.

## 3. Design decisions

- **Vendor, don't import.** `squidxplorer/_zarr_store.py` vendors the ~40-line tensorstore
  store-config (blosc-zstd zarr v3) + NGFF group JSON. Importing `tilefusion` would run its
  heavy `__init__` (numba / GPU / basicpy). `tensorstore` is the only new runtime dep.
- **Layout = Squid canonical:** `plate.ome.zarr/{row}/{col}/{fov}/0`, arrays named `0`
  (not `scale{N}/image`). Region parse vendored from `utils.parse_well_id` (uppercase,
  multi-letter rows, **no zero-padding** — ndviewer rebuilds `well_id = row + col`). Field
  dir uses the **raw fov id** (not a re-indexed field index), so multi-FOV wells stay faithful.
- **Group metadata up front.** Plate/row/well groups are written from `reader.metadata`
  before consuming the stream, so completion-order arrival needs no ordering logic.
- **Colors from 189.** `omero` colors are `metadata.channels[].display_color`; the writer
  never re-parses the acquisition YAML.
- **TIFF path** is `tifffile.imwrite` per plane (no `write_ome`); channel identity in the
  filename, native dtype, z collapsed to `0`.

## 4. The contract IMA-185 consumes (finalized — stable)

```python
from squidxplorer import write_plate
manifest = write_plate(reader, out_dir, *, n_fovs=1, workers=None, projector="mip", tiff=True)
#   out_dir/plate.ome.zarr/{row}/{col}/{fov}/0   — OME-NGFF v0.5 HCS plate (zarr v3, TCZYX)
#   out_dir/tiff/{t}/{region}_{fov}_0_{channel}.tiff   — individual per-plane TIFFs
# manifest = {"plate": <path>, "tiff": <path|None>, "n_wells": int, "n_fields_written": int}
```

- The plate **opens in ndviewer_light** today (`discover_zarr_v3_fovs` → `hcs_plate`,
  `parse_zarr_v3_metadata` → per-channel colors) and **validates against ome-zarr-models**
  (official OME-NGFF v0.5 pydantic schema). So IMA-185's "opens in ndviewer_light" requirement
  is **already met** by this output — IMA-185 is the *navigable/montage* layer on top
  (montage artifact, region jump, or whatever the requirement lands on), not a re-write.
- Colors quantize to ndv's 7 named colormaps; the omero hex + wavelength-bearing channel names
  drive them (405→blue, 488→green, 561→yellow, 638→red).

## 5. Testing (unit → cross commit on both datasets)

- **Independent module** — `tests/test_output.py` (clean-room, faked stream, no data): layout,
  region parse (uppercase / no-pad / fail-loud), single-level, omero colors, individual TIFFs,
  uint8/uint16, fail-loud shape/channel guards, **bounded memory** (4× wells < 2× peak), and
  **spec validation** (`ome-zarr-models` v0.5).
- **Cross commit (184 ↔ 188/183)** — `tests/test_integration.py` `# SECTION: IMA-184 ↔ 188/183`,
  real seam no mocks on `sim_1536wp` + real hongquan: opens in ndviewer_light (`hcs_plate`,
  correct well ids) and zarr-python; OME-zarr array `0` and the individual TIFFs are
  **byte-identical** to `project_well` (`|diff| = 0`); omero colors == the reader's resolved
  colors; 1536-well layout scales; metadata validates against the v0.5 schema.
- **Full suite: 106 green** (88 clean-room + 18 integration).

## 6. Performance (measured, this machine)

- Write ≈ **0.26 s/well** (OME-zarr) + a small TIFF increment (4 ch, 4168² uint16).
- Peak memory **flat in plate size**: 67 → 109 → 109 → 109 → 109 MB for 1 → 16 wells — the
  writer holds ~one well, never the plate (vs a linear "if materialized" line).

## 7. Verification figures (saved to `~/Downloads` — drag each PNG in to embed)

- `squidxplorer_ima184_plate_colors.png` — OME-zarr plate, omero colors from the channel YAML.
- `squidxplorer_ima184_roundtrip_identity.png` — project_well vs OME-zarr vs TIFF, `|diff| = 0`.
- `squidxplorer_ima184_write_memory.png` — flat memory footprint vs "if materialized".
- `squidxplorer_ima184_write_speed.png` — per-well write time, both datasets.

## 8. Block-by-block review feedback applied

- Region parsing overstated ("padding breaks discovery") → corrected to well-id fidelity;
  vendored Squid `parse_well_id` (uppercase, no-pad) + fail-loud shape assert.
- "Why a pyramid if single-FOV?" → dropped the pyramid entirely; match Squid's single level.
- "Lay on tried-and-true public repos" → surveyed ome-zarr-py / ngff-zarr / iohub; kept the
  lean ndviewer-verified streaming writer and laid on `ome-zarr-models` (official v0.5 schema)
  for the real risk — metadata validity — as a test-time validator, no runtime dep.
- Individual-TIFF output (not OME-TIFF) per You Yan; dropped the `write_ome` vendor + Nick flag.

## 9. IMA-185 handoff (full — pasteable)

- **You inherit** a working canonical plate: `write_plate(reader, out) -> out/plate.ome.zarr`
  that opens in ndviewer_light + validates as OME-NGFF v0.5. The "opens in ndviewer_light"
  requirement is satisfied here — scope IMA-185 to the *navigable* delta (montage / plate-grid
  artifact / region jump), not a re-export.
- **Your cross commit (185 ↔ 184):** append `# SECTION: IMA-185 ↔ IMA-184` to
  `tests/test_integration.py` — drive `write_plate` on `sim_1536wp` + real hongquan, then run
  185's navigable output over that real plate (no mocks) and assert it renders/enumerates the
  wells correctly. A slot isn't done until its cross commit is green.
- **Don't** re-parse channel colors (use `metadata.channels[].display_color`) or add a pyramid
  (deferred to IMA-193; there's a TODO to confirm IMA-193's reader needs multiscale before
  building it). Keep bounded/streaming memory.
