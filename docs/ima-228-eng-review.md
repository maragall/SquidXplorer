# IMA-228 — Minerva export — engineering review

Reviewed: 2026-07-20 · Branch: `juliomaragall/ima-228-minerva-export` · Spec: `.spec/open/ima-228.md`

The spec as written is not buildable. Three of its load-bearing assumptions are false against the
code that exists today: `squid2minerva` is not an importable library, minerva-author cannot be
pointed at a dataset programmatically, and the viewer has no FOV selection model. This document
records what the review found, the decisions taken, and the implementation plan that follows.

---

## Decisions

| # | Decision | Why | Rejected |
|---|---|---|---|
| 1A | SquidMIP writes its own OME-TIFF via `tifffile`; **no `squid2minerva` dependency** | `tifffile` is already a hard dep (`pyproject.toml:15`); `_output._omero`/`_wavelength_nm` already produce correct channel colour + wavelength | Vendoring (drags the `display_color` bug from `TODOS.md:47`); SHA-pinned dep (drags a Flask stack + conflicting pins) |
| 2A | Launch minerva-author **out-of-process**, then show the exact `.story.json` path | minerva-author has no deep-link; the handoff is manual by design (`convert.py:145`) | Patching minerva-author upstream (cross-repo, lost on re-clone); export-only (drops the stated P1) |
| 3A | Export the **current well** now, behind a **list-shaped API** | No selection model exists; IMA-205/187 own that work and rewrite the same file | Blocking on IMA-205 (parks the independent 80%); building multi-select here (duplicates IMA-205's keystone) |
| 4A | **One OME-TIFF + one story.json per FOV**, return the list of paths | Minerva Author ingests one 2D image at a time (`story.py:53`); SquidMIP has no stitcher (`docs/SCOPE.md:45`) | Mosaic (needs a stitcher, out of scope); single-FOV cap (contradicts 3A) |
| 5A | Explicit `out_dir` argument, defaulting to `<acquisition>/minerva_export/` | Matches `write_plate`'s existing `out_dir` convention (`_output.py:377-390`) | Temp dir (OS sweep breaks a live Minerva session); always-prompt (modal friction on a one-click button) |
| 6A | Export runs on its own `QThread`, signals added to `_retire`'s tuple | `_retire` (`_viewer.py:1949`) disconnects a hardcoded signal-name list; a new worker not listed there leaks signals into a freshly-opened plate | Blocking the GUI thread |
| 7A | minerva-author located by explicit config; **launch degrades gracefully** | Export must never fail because a sibling checkout is missing | Hard-failing the export when minerva-author is absent |

Decisions 6A and 7A were auto-decided (spec-consistent defaults) after the user delegated the
remainder of the review.

### Corrections from the outside voice

An independent review (Claude subagent; Codex not installed) challenged the plan and found six
defects that verify against the code. All are folded in above and below.

| # | Correction | Verified at |
|---|---|---|
| OV1 | **Minerva ignores OME-TIFF channel colours.** Colour reaches Author *only* via story.json hex. Decision 1A's "correct colour" justification applies to the story groups, not the OME-XML. | `story.py:4-5` — "Minerva Author colors channels by index and ignores OME-TIFF channel colors" |
| OV2 | **Pixel size is a hard ingest gate**, not a nice-to-have. Missing → opaque HTTP 500. SquidMIP's `pixel_size_um` is nullable and defended elsewhere with a silent `1.0` fallback, which here would put a *wrong physical scale* into Minerva. | `app.py:1855-1860` `api_error(500, "Image is missing OME-XML pixel size")`; `_acquisition.py:62`; `_output.py:173` |
| OV3 | **Filename is a gate.** `check_ext` takes the last two extension components; anything not ending exactly `.ome.tif`/`.ome.tiff` sets `reader = None` and import dies as "Invalid tiff file". | `app.py:112`, `:333` |
| OV4 | **Do not deviate to `imwrite(ome=True)`.** The proven writer is `imwrite(path, img, photometric="minisblack", metadata=meta)` with OME inferred from the extension. Minerva's `_get_ome_version` returns 5 when SubIFDs tag 330 is absent and re-opens with `is_ome=False` — a different axis path that flat output survives *by accident*. Deviating for no reason is unforced risk. | `export.py:26`; `app.py:343` |
| OV5 | **Decision 1A removed the import, not the runtime dependency.** minerva-author has no venv of its own; `explorer/setup.py:50-57` installs its deps into `explorer/.venv`. T3 must locate `explorer/vendor/minerva-author/src/app.py` **and** `explorer/.venv/bin/python`. The plan no longer claims the launch stage is dependency-free. | `explorer/.venv/bin/python` and `vendor/minerva-author/src/app.py` both present; no `vendor/minerva-author/.venv` |
| OV6 | **Silent timepoint loss and an n_t× wasted read.** `project_well` loops `for t in range(n_t)` computing *every* timepoint; slicing `image[0]` discards the rest. On a timelapse that is unannounced data loss at n_t× the I/O cost. | `projection.py:155-162` |
| OV7 | **Re-export collides downstream.** `api_import` errors when `out_dat` exists and is not `samefile` with the loaded story — so re-exporting a well the user has already opened in Author fails on Minerva's side, not ours. | `app.py` `api_import` |

**Rejected — sequencing.** The outside voice argued the multi-FOV work has "no spec and no owner"
because `.spec/open/` holds only `ima-228.md`. That is an artifact of looking only at this
worktree: Linear tracks IMA-205 as Todo and IMA-187 as an existing issue, and IMA-187's own eng
review was locked on 2026-07-21. Decision 3A stands.

**Partially accepted — "just use `convert.py`".** The challenge that `convert.py` is already a
strict superset is fair on one point: it offers `--mip`, `--z N`, and best-focus, while this plan's
v1 silently hardcodes one projection. SquidMIP has a projector registry
(`available_projectors`, `_engine.py`), so exposing that choice is nearly free — folded into T4.
The conclusion ("ship nothing here, shell out to `convert.py`") is rejected: it reintroduces the
whole `explorer` checkout as a hard runtime dependency of the *export* path, which is exactly what
decision 1A ruled out. Under this plan `explorer` is optional and only the *launch* stage needs it.

---

## Why the spec's three assumptions are false

**"reuse squid2minerva" — it is not a library.** No `pyproject.toml` or `setup.cfg` exists in
`/Users/julioamaragall/CEPHLA/projects/explorer`; its `setup.py` is a venv bootstrapper that
git-clones minerva-author into `vendor/`. Imports resolve only because `run.py:33` does
`sys.path.insert(0, ROOT)`. `requirements.txt` hard-pins `tifffile==2025.5.10` and `zarr==2.18.7`,
both of which fight SquidMIP's `tifffile>=2023.1.0` (`pyproject.toml:15`). There are no git tags,
and `__init__.py` says `0.1.0` while the README says v0.3 — it can only be pinned by SHA.
`minerva.ensure_running()` (`minerva.py:33`) runs
`subprocess.Popen([sys.executable, vendor/minerva-author/src/app.py])`, so importing it would
require SquidMIP's own interpreter to carry `waitress`, `flask_cors`, `pydantic`,
`xsdata==24.3.1`, `ome-types==0.6.3`, `scikit-image` and `openslide-bin`.

The parts actually needed are ~60 lines of pure-array code: `export.write_ome` (`export.py:13`),
`story.auto_groups` (`story.py:35`), `story.write_story_file` (`story.py:53`). `open_reader` is
disk-only and useless here — SquidMIP already holds the pixels in memory.

**"launches minerva-author on it" — there is no deep link.** `ensure_running()` (`minerva.py:26`)
takes no arguments. `open_author()` (`minerva.py:42`) returns the bare string
`http://localhost:2020/`. The handoff is manual by design: `convert.py:145` prints "Click 'Select
File' and open the .story.json", repeated at `minerva.py:43-47`. The squid2minerva README's claim
of a hosted `http://localhost:2020/story/...` link is stale — the committed code neither builds
nor serves it.

**"Selected FOV(s)" — nothing selects FOVs.** `PlateOverview` exposes two signals
(`_viewer.py:520-521`): `hovered` (title readout only) and `wellActivated`, which fires solely from
`mouseDoubleClickEvent` (`_viewer.py:679-682`) and always passes `fov_index=0` — the line carries
the comment `# 1 FOV/well (IMA-183); IMA-187 will pick the FOV`. `self._sel` (`_viewer.py:548`) is
presentation-only: it draws a red box (`_viewer.py:752-756`) and is set by `select(ri, ci)`
(`_viewer.py:621-624`) driven *from* the ndviewer FOV slider (`_viewer.py:1902-1911`), never from a
click on the plate. `mousePressEvent` (`_viewer.py:647-650`) starts a drag-to-pan and nothing else.
The app runs one FOV per well today (`n_fovs=1`, `_viewer.py:853`).

---

## Architecture

```
                        SquidMIP process                    │  separate process
                                                            │
  PlateOverview                                             │
  self._sel (red box)                                       │
  _viewer.py:548                                            │
        │                                                   │
        │ _current_well  (_viewer.py:1039)                  │
        ▼                                                   │
  ┌──────────────────────────────┐                          │
  │ Minerva tab                  │  _open_op_tab(...)       │
  │ _build_minerva_tab           │  _viewer.py:1224         │
  │  out_dir picker              │  state={"dir":None}      │
  │  "Export + Open" button      │  _viewer.py:1279         │
  └──────────────┬───────────────┘                          │
                 │ [(region, fov), ...]                     │
                 ▼                                          │
  ┌──────────────────────────────┐                          │
  │ _MinervaWorker(QThread)      │  pattern: _OperatorWorker│
  │  signals: exported(str,str)  │  _viewer.py:773          │
  │           progress(int,int)  │  retire: _viewer.py:1949 │
  │           failed(str)        │                          │
  │           finished_ok()      │                          │
  └──────────────┬───────────────┘                          │
                 │  per (region, fov) — streamed, never accumulated
                 ▼                                          │
  ┌──────────────────────────────────────────────────┐      │
  │ squidmip/_minerva.py                             │      │
  │                                                  │      │
  │  require pixel_size_um  ── missing → raise ──────┼──────┼──▶ export refused, loud
  │    (OV2: Minerva 500s without PhysicalSizeX)     │      │
  │                 │                                │      │
  │  project_well(reader, region, fov, t=0)          │      │
  │    projection.py:119  →  (1,C,1,Y,X) native      │      │
  │                 │        (t param — OV6)         │      │
  │                 │ image[0, :, 0]  →  (C,Y,X)     │      │
  │                 ▼                                │      │
  │  write_ome_tiff(...)   path MUST end .ome.tiff   │      │
  │    imwrite(path, img, photometric="minisblack",  │      │
  │            metadata=meta)   ← proven shape, OV4  │      │
  │    channel NAMES from metadata["channels"]       │      │
  │                 │                                │      │
  │                 ▼                                │      │
  │  write_story(...)   →  <name>.story.json         │      │
  │    groups: 1st/99.9th percentile per channel     │      │
  │    COLOUR LIVES HERE (hex) ── OV1: Minerva       │      │
  │    ignores OME-TIFF channel colours entirely     │      │
  └──────────────┬───────────────────────────────────┘      │
                 │  returns [(ome_path, story_path), ...]   │
                 ▼                                          │
  ┌──────────────────────────────┐                          │
  │ launch_minerva()  best-effort│  QProcess                │  ┌────────────────┐
  │  locate app + interpreter    │  pattern: _ProcTerminal  │  │ minerva-author │
  │  poll :2020 for liveness ────┼──────────────────────────┼─▶│ Flask/waitress │
  │  webbrowser.open             │  _viewer.py:307-357      │  │ port 2020      │
  │  on failure: warn, keep files│                          │  └────────┬───────┘
  └──────────────┬───────────────┘                          │           │
                 ▼                                          │           │ user clicks
  tab shows the exact .story.json path + copy + reveal ──────┼───────────┘ "Select File"
                                                            │
```

**Failure isolation.** Export and launch are separate stages. A launch failure (minerva-author not
installed, port 2020 busy, server slow to boot) never invalidates the export — the files are on
disk and the tab shows their paths. This is decision 7A.

**Why one FOV at a time.** The worker streams: project → write → release, per FOV. Peak memory is
one `(T,C,1,Y,X)` array regardless of selection size. Accumulating a list of images before writing
would make peak memory scale with the selection and is the thing to avoid.

---

## What already exists (reuse, do not rebuild)

| Need | Existing code | Status |
|---|---|---|
| Pixels for one FOV | `projection.project_well(reader, region, fov)` → `(T,C,1,Y,X)` native dtype, channels in `metadata["channels"]` order (`projection.py:119-163`) | Reuse as-is |
| OME-TIFF writing | `tifffile` already a hard dep (`pyproject.toml:15`, `imagecodecs` at `:16`) | `imwrite(ome=True)` — zero new deps |
| Channel colour → story.json groups | `metadata["channels"][i]["display_color"]` (hex, resolved by `_channels.py`) | Convert hex → RGB tuple for `auto_groups`. **Not** via OME-XML — Minerva ignores that (OV1) |
| Channel label / wavelength | `_output._wavelength_nm` (`_output.py:223-226`) | Reuse for labels; do not re-parse the yaml |
| Correct `display_color` resolution | `_channels.py` (handles top-level v1.0+ *and* nested `camera_settings`) | Already correct — this is why 1A rejects vendoring |
| Card + menu item + tab | `_OPERATIONS` registry (`_viewer.py:410-414`), `_open_op_tab` (`_viewer.py:1224`), `_op_tab_shell` (`_viewer.py:1258`) | One registry entry gets all three |
| Per-tab destination state | `state = {"dir": None}` closure (`_viewer.py:1279`) | Same pattern |
| Off-thread work + teardown | `_OperatorWorker(QThread)` (`_viewer.py:773`), `_retire` (`_viewer.py:1942`) | Same pattern; **must** extend the signal tuple at `:1949` |
| Launching an external process | `_ProcTerminal` `QProcess` (`_viewer.py:307-357`) | Same pattern |
| Synthetic acquisition for tests | `squid_dataset` (`tests/conftest.py:79-87`) — regions `B2`/`B3`, fovs `0`/`1`, NZ=2, 2 channels, 4×4 uint16, deterministic values | Assert exported pixels byte-for-byte |
| Offscreen Qt test harness | `QT_QPA_PLATFORM=offscreen` (`tests/test_viewer.py:16`), `qapp` (`:66`), `stub_detail` (`:73`), `_drain_until` (`:79`) | Reuse for the worker test |

Nothing here is rebuilt. The genuinely new code is the OME-TIFF + story.json writer and the launch
shim.

---

## Test coverage

```
CODE PATHS                                              USER FLOWS
[+] squidmip/_minerva.py                                [+] Export the current well
  ├── export_selection(reader, sel, out_dir)              ├── [GAP] Button → files on disk → path shown
  │   ├── [GAP] N=1 happy path (pixels byte-exact)        ├── [GAP] Export with no well selected
  │   ├── [GAP] N=3 → 3 file pairs, list order stable     └── [GAP] Second export overwrites cleanly
  │   ├── [GAP] empty selection → ValueError
  │   ├── [GAP] out_dir does not exist → created
  │   ├── [GAP] unknown (region, fov) → clear error
  │   ├── [GAP] pixel_size_um is None → refuses loudly  [OV2]
  │   └── [GAP] n_t > 1 → only t is read, not all       [OV6]
  ├── write_ome_tiff(img_cyx, path, channels, px_um)
  │   ├── [GAP] OME-XML round-trips: C count, names
  │   ├── [GAP] pixel size written as PhysicalSizeX/Y   [OV2]
  │   ├── [GAP] non-.ome.tiff path → rejected           [OV3]
  │   └── [GAP] uint16 preserved (no float cast)      [+] Launch handoff
  ├── write_story(story_path, ome_path, groups)          ├── [GAP] minerva-author missing → files kept,
  │   ├── [GAP] in_file is absolute + points at the OME  │         clear message, no crash
  │   ├── [GAP] groups: 1st/99.9th percentile per chan   ├── [GAP] port 2020 already busy
  │   └── [GAP] group colour hex == display_color  [OV1] ├── [GAP] explorer/.venv missing      [OV5]
  └── launch_minerva(story_path)                         ├── [GAP] re-export of an open story  [OV7]
      ├── [GAP] app not found → returns False, no raise  └── [GAP] [→E2E] server boots → browser opens
      ├── [GAP] interpreter not found → False    [OV5]
      ├── [GAP] liveness poll times out → False       [+] Error states
      └── [GAP] already running → reuses the server      ├── [GAP] read error mid-export → which FOV failed
                                                         └── [GAP] out_dir read-only → actionable message
[+] squidmip/_viewer.py
  ├── _build_minerva_tab()
  │   ├── [GAP] renders with no acquisition loaded
  │   └── [GAP] out_dir picker updates state["dir"]
  ├── _run_minerva_export()
  │   ├── [GAP] refuses a second concurrent export
  │   └── [GAP] failed signal → readout shows the error
  └── _retire() signal tuple                          ← CRITICAL: regression risk
      └── [GAP] worker signals disconnect on close

COVERAGE: 0/31 paths tested (0%)   |   Code paths: 0/23   |   User flows: 0/8
QUALITY: —                          |   GAPS: 31 (1 E2E, 0 eval)
```

All 31 are gaps because none of this code exists yet. The plan is that every one is written
alongside the feature, not deferred.

**Regression rule.** `_retire` (`_viewer.py:1942-1959`) disconnects a hardcoded tuple of signal
names at `:1949`, and a worker whose signals are absent from it is never disconnected — a late
signal then paints onto a freshly-opened plate, the cross-plate corruption the comment at
`:1943-1946` records a prior review catching. The outside voice correctly noted this was
over-dramatized in the first draft: three of the four planned signals (`progress`, `failed`,
`finished_ok`) are *already* in the tuple, so the actual exposure is one missing name. The real
defect is the hardcoded list itself. T5 therefore replaces it with signal introspection, which
deletes the failure class for every future worker instead of paying it forward. The teardown test
(close the window mid-export, assert no signal fires afterward) still ships with the feature.

### Failure modes

| Codepath | Realistic production failure | Test? | Error handling? | User sees |
|---|---|---|---|---|
| `export_selection` | Acquisition on a network share drops mid-read | planned | per-FOV try/except, continue | which FOV failed, in the readout |
| `export_selection` | **Acquisition has no objective pixel size** → Minerva 500s on import (OV2) | planned | refuse before writing; never fall back to `1.0` | "cannot export: acquisition has no pixel size" |
| `export_selection` | **Timelapse: t>0 silently dropped** (OV6) | planned | explicit `t` argument, default 0, stated in the readout | "exported t=0 of N timepoints" |
| `write_ome_tiff` | `out_dir` on a full or read-only volume | planned | catch `OSError` | actionable message with the path |
| `write_ome_tiff` | **Path not ending `.ome.tiff`** → Minerva "Invalid tiff file" (OV3) | planned | enforce the suffix in the writer | n/a — prevented |
| `write_story` | Absolute path assumed but a relative one was passed | planned | resolve before write | n/a — prevented |
| `write_story` | **Colour written only to OME-XML** → Minerva shows index colours (OV1) | planned | colour flows through story groups | n/a — prevented by design |
| `launch_minerva` | minerva-author or `explorer/.venv` not found (OV5) | planned | returns `False` | "export succeeded, Minerva not found at ..." |
| `launch_minerva` | Port 2020 held by the salesperson tool | planned | liveness poll finds *a* server | reuses it — acceptable, documented |
| `launch_minerva` | **Re-export of a story already open in Author** (OV7) | planned | cannot prevent; detect and explain | "Minerva already has this dataset open — close it first" |
| `_MinervaWorker` | Window closed mid-export | planned | `_retire` + signal introspection | nothing — silent, correct teardown |
| `_run_minerva_export` | Double-click the button | planned | `isRunning()` guard | "export already running" |

No failure mode is left with no test AND no error handling AND a silent outcome. **Zero critical
gaps.** Two would have been silent before the outside voice: the `1.0` pixel-size fallback (wrong
physical scale in Minerva, no error anywhere) and the dropped timepoints.

---

## Performance

Small and bounded, with two things to get right.

**Streaming, not accumulating.** Peak memory must be one `(T,C,1,Y,X)` array. Write each FOV before
projecting the next. With a 3000×3000 uint16 4-channel FOV that is ~72 MB; accumulating a
12-FOV selection first would be ~864 MB for no reason.

**Sequential is correct here, and that is deliberate.** `project_plate` (`_engine.py:131-139`)
already parallelises with a worker pool and accepts `regions=`, so it is tempting. But its FOV
choice runs through `select_fovs(metadata, n_fovs)`, which does not express an arbitrary
`(region, fov)` list — the semantics do not match the selection this feature carries. Selections
are 1–12 FOVs and each is a few hundred milliseconds, so a plain loop over `project_well` is both
correct and fast enough. Explicit over clever. When IMA-187 lands real multi-FOV selection, revisit
using `project_plate` for large selections; that is captured as a TODO rather than guessed at now.

**`auto_groups` percentiles.** The contrast groups come from the 1st/99.9th percentile per channel,
which is a full pass over each channel plane. At 3000×3000×4 that is ~36M elements — tens of
milliseconds, fine. It runs on the worker thread regardless, so it never blocks the GUI.

**The n_t× read the outside voice caught (OV6).** `project_well` loops `for t in range(n_t)`
(`projection.py:155`) and projects *every* timepoint before we slice one out. On a 50-timepoint
timelapse that is a 50× wasted read of the whole z-stack — by far the largest performance defect in
the plan, and it was invisible because the waste happens inside a function the plan was "reusing
as-is." T1a adds an optional `t` parameter to `project_well`, backward compatible (default `None`
= all timepoints, preserving every existing caller).

No database access, no N+1, no caching opportunity worth taking.

---

## NOT in scope

| Deferred | Rationale |
|---|---|
| Shift-drag multi-FOV selection | IMA-205's keystone work, blocked on IMA-187; lands in the same file (decision 3A) |
| Stitching a selection into one mosaic | No stitcher exists; `docs/SCOPE.md:45` and `docs/DESIGN-STATUS.md:44` put it out of scope (decision 4A) |
| Deep-linking minerva-author to a story | Requires patching a third-party app in `explorer/vendor/`; must be upstreamed first (decision 2B rejected) |
| Packaging `squid2minerva` as a real library | Belongs to that repo's maintainer; `TODOS.md:46-52` already tracks its `display_color` bug |
| Bundling or installing minerva-author | SquidMIP locates it, it does not own its install (decision 7A) |
| Parallel export via `project_plate` | Selection sizes are small today; revisit when IMA-187 makes them large |
| Distribution/CI for a new artifact | None introduced — this adds a module to an existing package, no new binary or container |

---

## Implementation Tasks

Synthesized from this review's findings. Each task derives from a specific finding above.

- [ ] **T1a (P1, human: ~1h / CC: ~5min)** — `squidmip/projection.py` — add an optional `t` parameter to `project_well`
  - Surfaced by: Outside voice OV6 — `project_well` loops `for t in range(n_t)` (`projection.py:155`) and computes every timepoint, so exporting one means an n_t× wasted read plus silent loss of the rest
  - Files: `squidmip/projection.py`, `tests/test_projection.py`
  - Detail: `project_well(reader, region, fov, reduce=project, t=None)`. `None` keeps today's all-timepoints behaviour so every existing caller is untouched; an int reads only that timepoint and returns `(1,C,1,Y,X)`.
  - Verify: `pytest tests/test_projection.py -k timepoint`
- [ ] **T1 (P1, human: ~1 day / CC: ~30min)** — `squidmip/_minerva.py` — write the OME-TIFF + story.json writer
  - Surfaced by: Architecture issue 1 — `squid2minerva` is not an importable library (no `pyproject.toml`, `sys.path` hack at `run.py:33`, conflicting pins); corrected by OV1–OV4
  - Files: `squidmip/_minerva.py` (new)
  - Detail: `export_selection(reader, selection, out_dir=None, t=0, projector="mip") -> list[tuple[Path, Path]]`. Per FOV: `project_well(..., t=t)` → `image[0, :, 0]` → `(C,Y,X)`. Stream — never accumulate images.
    - **OV2:** refuse the whole export up front if `metadata["pixel_size_um"]` is falsy. Never inherit `_output.py:173`'s silent `1.0` fallback — that writes a wrong physical scale into Minerva rather than failing.
    - **OV3:** enforce that every output path ends `.ome.tiff`.
    - **OV4:** use the proven call shape `tifffile.imwrite(path, img, photometric="minisblack", metadata=meta)` with `axes="CYX"`, `PhysicalSizeX/Y` + `µm` units. Do **not** pass `ome=True`.
    - **OV1:** channel colour goes into the story groups as hex, converted from `metadata["channels"][i]["display_color"]`. OME-XML `Channel.Color` is optional garnish — Minerva ignores it.
  - Verify: `pytest tests/test_minerva.py -k export`
- [ ] **T2 (P1, human: ~4h / CC: ~15min)** — `squidmip/_minerva.py` — `out_dir` policy
  - Surfaced by: Code Quality issue 5 — `squid2minerva` hardcodes `<repo>/output` (`convert.py:22,55`), which would write into a sibling checkout
  - Files: `squidmip/_minerva.py`
  - Detail: explicit `out_dir` argument matching `write_plate`'s convention (`_output.py:377-390`); default `<acquisition>/minerva_export/`; `mkdir(parents=True, exist_ok=True)`; absolute paths in `story["in_file"]`.
  - Verify: `pytest tests/test_minerva.py -k out_dir`
- [ ] **T3 (P1, human: ~1 day / CC: ~30min)** — `squidmip/_minerva.py` — best-effort launch shim
  - Surfaced by: Architecture issue 2 — minerva-author has no deep-link and `ensure_running()` takes no dataset argument (`minerva.py:26`); scoped by OV5
  - Files: `squidmip/_minerva.py`
  - Detail: locate **both** `<explorer>/vendor/minerva-author/src/app.py` and `<explorer>/.venv/bin/python` from one env var (`SQUIDMIP_MINERVA_HOME`) — minerva-author has no venv of its own, its deps live in explorer's shared venv (`explorer/setup.py:50-57`). Start via `QProcess` following `_ProcTerminal` (`_viewer.py:307-357`), poll `http://localhost:2020/` for liveness, `webbrowser.open`. Return `False` on every failure — **never raise into the export path**.
  - Note: this is a real, undeclared runtime dependency for the *launch* stage only. The export stage stays dependency-free. Document it in the README rather than pretending it does not exist.
  - Verify: `pytest tests/test_minerva.py -k launch`
- [ ] **T4 (P1, human: ~1 day / CC: ~30min)** — `squidmip/_viewer.py` — Minerva tab + worker
  - Surfaced by: Architecture issue 3 — export the current well behind a list-shaped API (decision 3A)
  - Files: `squidmip/_viewer.py`
  - Detail: one `Operation("minerva", ...)` entry in `_OPERATIONS` (`:410`) for card + menu + tab; `_build_minerva_tab` using `_op_tab_shell` (`:1258`) and the `state={"dir":None}` closure (`:1279`); `_MinervaWorker(QThread)` following `_OperatorWorker` (`:773`); pass `[( _current_well, 0 )]` today so IMA-205 only widens the list.
  - Also expose the **projector choice** (`available_projectors`, `_engine.py`) in the tab. Surfaced by the outside voice: `convert.py` offers `--mip` / `--z N` / best-focus, and v1 silently hardcoding one projection is a real capability regression against the tool this replaces. SquidMIP already has the registry, so this is a combo box.
  - Verify: `pytest tests/test_viewer.py -k minerva`
- [ ] **T5 (P2, human: ~1h / CC: ~10min)** — `squidmip/_viewer.py` — make `_retire` introspect signals instead of hardcoding names
  - Surfaced by: Test review regression rule, **rescoped by the outside voice** — three of the four new worker signals are already in the tuple at `:1949`, so the exposure was one missing name, not a critical regression. The real defect is the hardcoded list.
  - Files: `squidmip/_viewer.py`, `tests/test_viewer.py`
  - Detail: replace the literal tuple with introspection over the worker's `pyqtSignal` class attributes, which deletes this failure class for every future worker. Keep the existing `TypeError` guard. Add the concurrent-export guard mirroring `run_operator`'s `isRunning()` check (`:1741-1743`). Test: close the window mid-export, assert no signal fires afterward.
  - Verify: `pytest tests/test_viewer.py -k retire`
- [ ] **T6 (P1, human: ~1 day / CC: ~30min)** — `tests/` — full coverage for all 31 paths
  - Surfaced by: Test review — 31/31 gaps, since none of this code exists yet
  - Files: `tests/test_minerva.py` (new), `tests/test_viewer.py`
  - Detail: use `squid_dataset` (`tests/conftest.py:79`) to assert exported pixels byte-for-byte against the fixture's deterministic planes; use `qapp`/`stub_detail`/`_drain_until` (`tests/test_viewer.py:66,73,79`) for the worker.
  - Verify: `pytest tests/ -q`
- [ ] **T7 (P2, human: ~15min / CC: ~2min)** — `squidmip/_viewer.py` — fix the stale roadmap comment
  - Surfaced by: Code Quality — the comment at `_viewer.py:417-421` describes disabled "hand-off to Minerva Author" roadmap cards that no longer exist, and `_TO_BE_ADDED` is empty; adding a real Minerva operation makes it doubly wrong
  - Files: `squidmip/_viewer.py`
  - Detail: delete the two stale lines; keep the accurate description of `_TO_BE_ADDED`.
  - Verify: read it
- [ ] **T8 (P2, human: ~30min / CC: ~5min)** — `.spec/`, `docs/` — correct the spec and oracle
  - Surfaced by: Architecture issues 2 and 3 — the acceptance oracle as written is not satisfiable
  - Files: `.spec/open/ima-228.md`, `.sprint/oracle.draft.md`
  - Detail: state current-well scope, one-file-per-FOV, and an oracle that is actually measurable.
  - Verify: read them
- [ ] **T9 (P2, human: ~30min / CC: ~5min)** — `README.md` — document the minerva-author runtime dependency
  - Surfaced by: Outside voice OV5 — the launch stage needs an `explorer` checkout with a populated `.venv`, which decision 1A's framing obscured
  - Files: `README.md`
  - Detail: state that export works standalone, that launch requires `SQUIDMIP_MINERVA_HOME` pointing at an `explorer` checkout that has run its `setup.py`, and that the handoff is a manual "Select File" in Author.
  - Verify: read it
- [ ] **T10 (P1, human: ~2h / CC: ~2h)** — manual — verify a real export actually ingests
  - Surfaced by: Outside voice OV1–OV4 — every ingest requirement here was found by reading minerva-author's source, and the gates (pixel size, filename, OME version fallback) are undocumented and unversioned (`--depth 1` clone of upstream `main`, no tag)
  - Files: —
  - Detail: export one real well, open it in a real Minerva Author, confirm the channels carry our colours and contrast. **No unit test can prove this** — the contract belongs to a third-party app nobody here controls. This is the one task that must not be skipped, and it gates the ticket.
  - Verify: manual, screenshot in the PR

---

## Parallelization

| Step | Modules touched | Depends on |
|---|---|---|
| T1a | `squidmip/` (`projection.py`) | — |
| T1, T2, T3 | `squidmip/` (new `_minerva.py`) | T1a (needs the `t` parameter) |
| T4, T5, T7 | `squidmip/` (`_viewer.py`) | T1 (needs the export signature) |
| T6 | `tests/` | T1a–T5 |
| T8, T9 | `.spec/`, `docs/`, `README.md` | — |
| T10 | — (manual) | T1–T4 |

`Lane A: T1a → T1 → T2 → T3 (sequential — T1a in projection.py, rest in _minerva.py)`
`Lane B: T8 → T9 (independent, docs only)`
`Lane C: T4 → T5 → T7 (sequential, all in _viewer.py) — waits on Lane A`
`Lane D: T6 — waits on A and C`
`Lane E: T10 — waits on A and C, gates the ticket`

Launch A and B in parallel. Then C. Then D and E. **Conflict flag:** Lane C touches `_viewer.py`,
which IMA-187 and IMA-205 will also restructure — keep the diff to one registry entry, one builder
method, one worker class, and the `_retire` change, so the eventual merge is mechanical. T1a
touches `projection.py`, which nothing else in flight modifies.

---

## Retrospective

Recent branch history (`dbade2e` plate grid, `6845136` viewer CLI terminal, `aa2695e`/`1504c05`
docs) is all viewer- and docs-facing, with no prior review-driven reverts in this area. The
`_retire` comment at `_viewer.py:1943-1946` records a previous review catching cross-plate tile
corruption from undisconnected worker signals — which is why the first draft over-weighted T5. The
outside voice corrected that: the pattern that caused the old bug is still there, so the right
response is to delete the pattern (signal introspection), not to add another entry to it.

---

## Completion summary

- Step 0: Scope Challenge — scope accepted as-is (3 files, 1 new class; complexity check did not trigger), but **all three of the spec's premises were corrected**
- Architecture Review: 3 issues found, 3 resolved
- Code Quality Review: 3 issues found, 3 resolved (2 asked, 1 folded as T7)
- Test Review: diagram produced, 31 gaps identified
- Performance Review: 1 issue found (the n_t× read, surfaced by the outside voice)
- NOT in scope: written
- What already exists: written
- TODOS.md updates: 2 items (squid2minerva colour path escalated; parallel export deferred)
- Failure modes: 0 critical gaps (2 would have been silent without the outside voice)
- Outside voice: ran (Claude subagent; Codex not installed) — 7 corrections accepted, 1 rejected, 1 partially accepted
- Parallelization: 5 lanes, 2 parallel / 3 sequential
- Lake Score: 5/5 decisions chose the complete option

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | — |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | — | — |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 12 issues, 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **CROSS-MODEL:** The outside voice (Claude subagent, Codex not installed) found 7 defects this review missed, all verified against source: colour is ignored in OME-XML (`story.py:4-5`), pixel size is a hard 500-level gate (`app.py:1855-1860`), the `.ome.tiff` filename is a gate (`app.py:112`), the proven `imwrite` shape should not be swapped for `ome=True`, the launch stage really does depend on `explorer/.venv`, `project_well` wastes an n_t× read and drops timepoints (`projection.py:155`), and re-export can collide inside Author. One finding rejected (sequencing — it read only `.spec/open/` and missed Linear's IMA-187/205). One partially accepted (`convert.py` superset — took the projector-choice gap, rejected "ship nothing").
- **VERDICT:** ENG CLEARED — ready to implement.

NO UNRESOLVED DECISIONS
