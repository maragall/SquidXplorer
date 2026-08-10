# Plate-driven navigation, a default working layout, and tabbed views

Plan of record, 2026-08-10. Written against `main` at `ab14a40`, to land on `easy-startup`.

## Context

Feedback says SquidXplorer is missing what users expect from comparable software: **the plate view
should be a navigator, not just a launcher.** Today the plate can only *open* windows — double-click
a well, or select wells and press "Open view". Once a child window exists the plate cannot steer it:
`RegionViewer._regions` is frozen at construction (`_region_viewer.py:443`) and
`RegionCursor.set_order` is called exactly once per window (`:711`). Nothing re-scopes an open window.

Three features, plus one prerequisite bug fix that surfaced during exploration.

### Decisions taken

| | |
|---|---|
| Click a well outside the child's selection | Navigates anyway — "plate = free navigator", region adopted on demand |
| Plate's blue operator selection | **Preserved** while a view is open; a plain click navigates only |
| Default child | All wells, lazy — only the current region renders |
| Screen | The one the app opens on (`availableGeometry()`) |
| Plate width | **420 px** (~1/5). Wells shrink; wheel-zoom recovers detail on demand |
| Computed `.hcs` plates | No default child (raw only) — documented gap |
| Second acquisition | Close the stale default child, open a fresh one |
| Quit dialog | Does not count the auto-opened default view |
| Tabs | Phased, and gated on a hand-check that GL-in-tabs works at all |

---

## Findings that shape the design

Measured against the tree and the installed venv, not reasoned.

**The napari pane leaks on every window close.** `MosaicPane.shutdown()` (`_napari_pane.py:582`)
has **zero callers** anywhere in `squidmip/`, `tests/` or `tools/`. `RegionViewer.closeEvent`
(`:3376-3434`) never calls it. Its own docstring names the consequence — "one leaked per pane built
— a GL context and tens of MB each, which killed a session after twenty of them." This is a real
pre-existing bug sitting exactly on the seam all three features touch.

**The plate is width-limited at full height, so a narrow root costs real detail.** Plate cells are
square (`_plate_overview._fit_cd:1649`). Measured px/well: 96wp 43.2 → 28.5, 384wp 21.6 → 14.2,
1536wp 10.8 → 7.1. 596x435 and 596x585 give the *identical* 43.2, proving the width limit. The
comment at `_viewer.py:960-964` ("past it the plate does not grow, only the gutters") is now stale —
it was written when the root was 850 tall. Accepted on the basis that the plate is a navigator and
`wheelEvent` (`_plate_overview.py:2267`) zooms with fit as the floor. **Say so in the readout on
first load** so the affordance is discoverable.

**The layout contract is NOT broken.** `tests/test_main_window_layout.py:48-54` forces its own sizes
(`w.resize(*size)`) and never calls `_default_root_size()`. Its assertions are a plate-vs-band
*height* share; this feature changes the root's *width*. Measured at 420x1000: plate 58.5% (floor
48.0), band exactly 365 px. Orthogonal.

**The two tests that DO break** were not the ones expected:
`tests/test_no_orphan_windows.py:100-118` ("anything visible here showed ITSELF") and
`tests/test_nav_wiring.py:160-165` (`len(windows) == 1`). Both stay green under the opt-in flag below
— which is the decisive argument for that design.

**Type at 420 px is 0.85x, not 0.70x.** `_fontscale.ui_scale` clamps at `max(0.85, ...)` (`:87`) and
`scale_qss_fonts` floors at 8 px (`:76`). Legibility is already bounded.

**"The active child window" does not currently exist.** `changeEvent` (`:3353`) calls
`set_active(self.isActiveWindow())` (`:3336`), which only halts playback — it never tells the
manager. Clicking a child's title bar does not move `_focused_id`. `on_screen_luts` (`:2298`)
already documents `focused_id` as "the window the user is looking at", so this is an existing lie.

**The plain-click branch is the only click-driven deselect path.** `_plate_overview.py:2426-2429`:
*"without it a batch selection could never be dropped by clicking."* Resolution — only the *hit a
well* case changes, and only while a view is open:

| gesture | today | after |
|---|---|---|
| click a well | selection := {that well} | **navigate active child; selection unchanged** |
| click empty space | selection := {} | unchanged — the escape hatch survives |
| Escape / Shift / Ctrl-click | edit selection | unchanged |
| double-click | open a new window | unchanged |

**`PlateWindow._highlight_view_regions` (`:4110`) is dead** — zero callers — and it calls
`highlight_regions`, which *replaces* `_selection`. Do not reuse it for any of this work; delete it,
following the "pin the absence, not the None" convention (`test_nav_wiring.py:100-105`).

---

## Work

### Commit A — fix the napari pane leak *(ships regardless of everything else)*

Extract `RegionViewer.dispose()` from `closeEvent` (`_region_viewer.py:3376-3434`), verbatim, plus
the missing `self._pane.shutdown()`, then `closeEvent` calls it. Idempotent via a `_disposed` bool
declared as a **class default as well as in `__init__`** — `:406-410` already documents why (a bare
attribute read on a half-built QObject raises out of Qt's machinery). Test: `shutdown` called
exactly once, and still once on a double dispose.

### Commit B — `_regions` becomes a property over the cursor

Behaviour-neutral, widest blast radius, so it lands alone and the full suite runs after it.
`self._regions = ...` at `:443` becomes `self._seed_regions` (immutable: "what this window was
opened over"); `_regions` becomes a read-only property returning `self._cursor.regions`. Fix the
reads at `:514` and `:711-713`. Payoff: `ViewerManager.views()` (`:3530`) and
`viewFocused.emit(list(win._regions))` (`:3670`, `:3740`) stay correct after a re-scope with no
refresh call. Verified safe: nothing outside the class assigns `win._regions`.

### Commit C — make `_focused_id` true

Add `ViewerManager.note_focus(wid)` — the passive half of `focus()`; it records and re-publishes and
must **never** raise or activate, or it ping-pongs with the window manager (`focus()` calls
`activateWindow()`, which fires `changeEvent`, which lands here). Early-return on unchanged id is the
second guard. Add `ViewerManager.active_view()` as the single answer to "which window is active", and
re-point the open-coded lookup at `_viewer.py:2325-2327` at it. `changeEvent` calls `note_focus` when
activated. The plate taking focus must **not** clear `_focused_id` — it means *last-focused child*;
this holds by construction since `PlateWindow` has no `changeEvent`. Comment it or someone will "fix" it.

### Commit D — the plate steers the active child

- `RegionViewer.show_region(region) -> bool`, beside `current_region()` (`:3290`). No-op if already
  there; if the region is not in the cursor's order, validate against `meta["regions"]`, grow the
  order **in acquisition order** (so the slider still reads left-to-right) and `set_order` — which
  keeps the current region, so the order subscribers fire and no second mosaic load is triggered.
  Then `set_region`. Returns `False` having *said why* when the acquisition has no such region.
- The rendering contract is honoured with **no new code**: `_load_mosaic` already drops operator
  layers (`:2586`) and calls `remove_op(_RAW_OP)` (`:2613`) because `_shown_region != region`.
- New `wellNavigated = Signal(str)` on `PlateOverview`, and a `_click_navigates` mode set by the one
  writer `PlateWindow._refresh_plate_navigation()` — the plate cannot see the window registry, so it
  is told. Hook the `:2430` branch per the table above. The marquee (`:2367`) and ctrl-click
  (`:2415`) branches are untouched.
- **Single-click vs double-click:** defer the navigate by one `QApplication.doubleClickInterval()`
  and cancel it in `mouseDoubleClickEvent` — exactly what `_hold` already does there, and for the
  same stated reason (`:2692-2696`). Without it a double-click navigates the active window *and*
  opens a new one, dropping the first window's operator layers. Timer connected to a **bound
  method**, stopped in `hideEvent`/`shutdown` (`test_window_lifetime.py:183-220`).
- `PlateWindow._on_well_navigated` moves the red frame with `_cursor.set_region`, **not**
  `activate` — `activate` sets `_activated`, which means "the user explicitly opened a region"
  (`test_nav_wiring.py:191-200`).
- Update the gesture state machine at `_plate_overview.py:2075-2110`, the stale `mousePressEvent`
  comment at `:2285-2293`, and the matrix at `tests/test_viewer.py:697-707`.

### Commit E — the default layout

- `_default_root_size()` (`:943`): `w = max(minimumWidth(), min(_DESIGN_W, avail.width() // 5))`.
  Height already grows. `_DESIGN_W` stays 596 — it is the *type denominator*. Keep
  `minimumWidth() == 420`: `tests/test_root_resize.py:93-95` pins it, and the 36 px to reach a
  literal 1/5 comes straight off the already-squeezed navigator.
- Put the arithmetic in pure functions in `_fontscale.py` (`default_root_width`, `beside_rect`)
  beside the existing `window_screen` — testable with no screen.
- **Opt-in flag:** `PlateWindow(path, *, default_layout=False)`, set to `True` only by `main()`
  (`:4831`). This states a product rule (the launcher asks for the layout; a library caller gets a
  bare window) rather than branching on "am I in a test", and it is what keeps ~120 existing
  `ingest()` calls from each building a real napari `QtViewer`.
- Hook `_on_plate_loaded()` at the tail of `ingest()` (`:2823-2827`) — provably the single common
  point for all four load paths, with all three refusals returning before it. Not called from
  `_open_computed`, which sets `reader = None` so `open()` refuses by contract; add a comment saying so.
- Defer past `show()`: with an `initial_path`, `ingest` runs inside `__init__` before `main()` shows
  the root, and `_spawn` shows the child. Set `_pending_default_view` and open from `showEvent`
  (`:4410`). Apply the root move/resize once per window lifetime so a re-ingest cannot snap a window
  the user has dragged.
- Open with `_viewer_manager.open(list(self._order))` — **verified selection-safe**; never
  `select_all()`, which writes `_selection` and emits `selectionChanged`.
- Place with `_place_beside(win)` *after* `open()` returns (geometry set before `show()` is discarded
  by some WMs). Align the child's top and height to the root's frame so it reads as one layout.
- Re-ingest: close the recorded `_default_view_id` and open a fresh one — every open `RegionViewer`
  holds its **own** reader captured at construction (`:441-442`), so reusing it would show the old
  acquisition's pixels under the new plate. Leave user-opened windows alone.
- `_confirm_close_all` (`:4441`) must not count the default view.
- Consider a one-line filter in `_refresh_view_hues` (`:4115`): a wash covering every region
  distinguishes nothing.

### Check F — prove GL-in-tabs by hand *(no code; gate on this)*

On this machine, hand-embed four `RegionViewer`s in a scratch `QTabWidget` and check (a) four GL
canvases in one window render at all, (b) the deck's minimum height fits the panel
(`_napari_pane.py:501-503` propagates a 560 px floor), (c) memory with four napari viewers. **The
whole suite runs under a platform with no GL, so this cannot be tested offscreen.** If (a) fails the
tab design is dead — for an afternoon, not a week.

### Commit G — tabbed views: container + detach *(only if F passes)*

New `squidmip/_view_deck.py` holding a `ViewDeck` top-level with a `_DetachTabs` from `_qt_tabs.py`
(`first_detachable=0` — its docstring already anticipates "every tab is a user-opened subset").
Not inside `PlateWindow`: the root's band is already three panels fighting over ~300 px, and the
4/5 window *is* the deck.

The only new reparent is the outermost one — the whole `RegionViewer` assembly moves as one object
and the napari canvas is never touched, which is what `_napari_pane.py:212-219` requires. Mirror
`_embed_native_window`'s three calls exactly; on detach, `setParent(None)` explicitly
(`removeTab` does not reparent).

Resolutions: `dispose()` from Commit A restores deregistration; `reveal()`/`collapse()`/
`request_close()` on `RegionViewer` answer correctly in both states so `ViewerManager` keeps one call
site each; "active" becomes *current page of its host* **and** *host is active window* — which makes
`QStackedWidget` hiding background tabs a memory **win**; and `PlateWindow.closeEvent` sweeps decks
beside the gallery (`:4550`) or a plateless deck survives holding the single-instance flock.

`ViewerManager.set_focused(wid)` becomes the one writer for "a view became current", emitting
`viewFocused` — so a tab switch repaints the plate hue through the path already wired at `:417`.
**Zero new signals.** Plate-as-navigator must read `manager.focused_id`, never `isActiveWindow()`,
or the two features will not compose.

`_qt_tabs.py` is **not edited**. `_FloatWindow` is deliberately not reused: it renders its own title
(`:91`), duplicating the `[wid] label` join that is the only link between a log line and a window.
Detach is just "stop being a tab" — less code than reusing it.

### Commit H — drag tabs between windows *(deferred)*

`TODOS.md` has refused this exact item twice, both times because the expensive half is untestable
offscreen and "roughly triples the untested gesture surface". Commit G leaves the hooks so H is
additive: public `dock_page(page, index)`/`undock_page(page)`, `_host` as the one answer to where a
page lives, `ViewerManager._decks`, and a drag payload of `window_id` (an int — a widget pointer
carried across a Qt drag is a crash). The 20-cycle dock/undock canary in G is also H's feasibility gate.

---

## Verification

- `tools/run_suite_chunked.py` — the suite cannot run in one process.
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is mandatory or the PyQt tests silently skip against PySide.
- New GUI tests: `QT_QPA_PLATFORM=offscreen` at module top **before** any Qt import; `qapp` and
  `_drain_until` from `.test_viewer`; `shutdown_plate_window` and `napari_pane_stub` from
  `.conftest` (napari has no GL offscreen); drain with `_drain_until`, never sleep.
- Geometry: construct, `resize`/`show`, `processEvents`, then read `.geometry()` —
  `isVisible()` is not "has pixels" (`test_main_window_layout.py:22-25`). The offscreen screen is
  **800x600**, so assert relative to `availableGeometry()` and the root's own frame, never literals.
  Split pure arithmetic (parametrised over six screen sizes) from the one real-pixel case.
- New files: `tests/test_plate_navigates_views.py`, `tests/test_default_layout.py`,
  `tests/test_view_deck.py`. Amend `tests/test_no_orphan_windows.py` rule 1 in prose — the honest
  statement becomes "a window the caller did not open is a stray, *unless* the caller asked for the
  default layout" — and add the companion test asserting the strays are exactly one `RegionViewer`.
- The assertion that makes "startup cost is independent of plate size" a fact: count `_MosaicWorker`
  constructions and assert exactly **1** for both a 4-region and a large region list.
- End-to-end on this machine via the Desktop shortcut with
  `~/Desktop/LaserAF_Test/MAINBRANCH_2026-06-25_15-57-10.748451` (21 regions, loads clean). Take a
  before/after `_measure.WindowOpen` number (`_region_viewer.py:3646`) — the auto-open now runs
  concurrently with `_start_preview`'s thumbnail fill over the same reader, which is disk contention
  that today only happens if the user double-clicks during preview.

### Order

A → B → C → D *(feature 1 complete)* → E *(feature 2 complete)* → F → G *(feature 3)* → H deferred.
