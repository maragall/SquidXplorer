# The SquidXplorer version arc

Julio, 2026-07-28. This file exists so that "post-acquisition only" reads as a **product
decision** and not as a limitation somebody should quietly fix.

If you are about to build something because it seems obviously missing, check here first. It may
be missing on purpose, and section 3 names the condition under which it stops being.

## v1: post-acquisition only. THIS IS THE CURRENT SCOPE

Squid's own software grows an **Open SquidXplorer** button that appears once an acquisition
finishes. Clicking it opens SquidXplorer on that folder. That is the whole integration.

What that buys, and why the boundary is worth defending:

- **There is no live folder to follow.** The acquisition is over before we are launched, so there
  is no tailer, no partial store, no growing-file race, and no "is it done yet" state machine.
  Live acquisition is where most of the complexity in a viewer of this kind lives.
- **The reader is stateless and the data is complete.** `reader.read(region, fov, channel, z)`
  answers from a finished folder. Anything on disk is final, so a cache key can be a forever-key.
- **The integration on Squid's side is a button**, not a shared process, not a protocol.

## v2: Squid integration. DIRECTION, NOT DESIGNED

SquidXplorer replaces Squid's **mosaic view** and **multi-channel view** inside the acquisition
GUI itself.

The thing to understand about v2 before designing it: those views are **live by definition**. A
mosaic view that cannot show the run in progress is not a replacement for the one it removes. So
v2 is the version that ends the v1 boundary above, and most of section 3 fires at once.

## v3: standalone cloud web app. DIRECTION, NOT DESIGNED

Cloud compute, Google Drive as the data store, a browser as the client.

The thing to understand about v3: producer and consumer stop sharing a process, a filesystem,
and a release. Data of unknown vintage is read by a client that was not built at the same time as
the writer. Everything that is currently an internal convention becomes a wire contract.

## 3. What is deliberately not built, and what would change that

Each of these is **correct** and **unbuilt on purpose**. The reason is never "it is a later
version". A version number is a promise about a plan; the entries below name a **trigger**, the
observable condition that makes the thing necessary.

The one standing reason to leave something unbuilt: **an API with no consumer is how this repo
grew four subsystems that read as dead.** Where that objection does not apply, build it now.

| Not built | Trigger that promotes it |
| --- | --- |
| Live acquisition following (an event-log tailer or file-watch engine) | The first time SquidXplorer is pointed at a folder that is still being written. |
| `filled_extent(axis)` and `is_extent_exact()` on the reader: how much of this store is really written, and do I know that exactly | The first read of a partially written store, **or** a decision to support damaged and partial acquisitions. |
| Revalidating store handles on open rather than per read | Same trigger. It exists for stores that grow. |
| A machine-checked on-disk contract with a compared `format_version` | Partly built in v1 because its absence already caused a defect. The full validator earns its keep when a second implementation reads our output. |
| A shared server-side or object-store thumbnail tier | Compute or storage moving off the local workstation. The **reasoning** ports (never write into the experiment folder), the **location** does not. |
| Qt6 migration | A napari upgrade past 0.7, or a rendering defect traced to the Qt5 binding. Note this is **not** blocked on and no longer motivated by AGAVE, which is cancelled. |

## 4. Cancelled, not deferred

**AGAVE will never be implemented** (Julio, 2026-07-28). It is not waiting on the Qt6 migration
and it is not waiting on anything else. Full-resolution 3D is served by the texture-bounded
native volume (`squidmip/_napari3d.py`), which is the **top tier**, not a preview below a better
renderer that is coming later. The real constraint is honest and explainable: napari renders 3D
from a single GL texture capped near 2048, so full native resolution needs a crop, and the ROI
child window is how you take one.

## 5. The rule this file exists to enforce

**Code with no caller carries a comment saying why, or it gets deleted.** Those are the only two
states.

"Unreferenced and unexplained" is not free. An external review of this repo read the tree
carefully and reached two wrong conclusions from it, because four subsystems looked identical
from the outside while meaning four different things: one cancelled, one waiting on a named
consumer, one being wired on a branch, one genuinely dead. AGAVE is the sharpest case. It read as
merely unreachable for weeks, so nobody deleted it, and meanwhile the UI kept telling users to
click a button the app no longer had.

## 6. A working note on Qt object lifetime, learned expensively

Three segfaults were root-caused on 2026-07-28 and every one had the same shape: **a Qt object
that Python owns, handed to Qt, and freed on someone else's schedule.** The per-window `QStyle`,
the `QApplication` held only by a pytest fixture cache, and a debounce `QTimer` left armed across
a window close with a `self`-capturing lambda as its slot.

They cost a day, and while they were live the entire test suite could not report a result: it
died partway through and took pytest's summary line with it, so a broken test and a test that
never ran looked identical. A test asserting a widget removed months earlier sat there failing,
unseen.

When you hand a Qt object to a widget, a view, a layout or a signal, ask who owns it afterwards.
If the answer is "a Python name in one place", that is the bug. `tests/test_window_lifetime.py`
records all three with their evidence tables.
