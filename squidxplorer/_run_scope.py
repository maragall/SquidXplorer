"""WHAT A RUN IS AIMED AT, and how that target is said out loud. No Qt, no napari.

Three questions, one module, because they are asked together at exactly one moment — the
instant before an operator run starts:

``operator_busy``
    May a run start at all, or is one already alive?
``resolve_run_scope``
    Turn the scope selector's value into the region list this run will iterate.
``describe_run_target`` / ``describe_view_target``
    Name the resolved target BEFORE the compute is spent. The first is the right shape for a
    flat region list (the plate and selection paths); the second names the WINDOWS a run is
    aimed at and the subset of each, because the flattener throws the windows away.

It lives outside ``_viewer`` for two reasons. It is testable with no display, no GL and no
napari (``tests/test_run_scope.py``), and ``_viewer`` is a 6000-line module several agents edit
at once.

It was called ``_explore`` until 2026-08-05, when the exploration pane it was named after was
removed. The pane's own rules (tab identity, the subset slider's cursor, the preview progress
sentence, the per-region layer-group name) went with it; what is left had never been about the
pane. ``subset_selection`` is the one exception and is kept deliberately — see its own note.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

__all__ = [
    "operator_busy",
    "RUN_SCOPES",
    "SCOPE_PLATE",
    "SCOPE_REGION",
    "SCOPE_SELECTION",
    "resolve_run_scope",
    "describe_run_target",
    "describe_view_target",
    "distinct_view_regions",
    "subset_selection",
]


def operator_busy(worker, retired) -> bool:
    """Is an OPERATOR RUN still alive? The question ``run_operator`` has to ask before starting.

    Deliberately NOT the same question as "is any producer thread still alive", which is what the
    window's ``_busy()`` answers. Conflating the two shipped a real defect: ``_retire`` parks the
    RAW PREVIEW in the same list, so a retired preview was still draining when the user launched
    the next operator, and the run refused itself with "already processing — let the current run
    finish first". The user had started nothing to finish; from their side the button simply did
    not work.

    A worker opts out by declaring ``IS_PREVIEW``. Only the raw preview does, and it says so on
    the class, so a new worker is counted as a run by default — the safe direction.
    """
    if worker is not None and worker.isRunning():
        return True
    return any(w.isRunning() for w in retired if not getattr(w, "IS_PREVIEW", False))


#: The scopes an operator run can be aimed at, in the order the selector lists them.
#:
#: Julio: "we have the controls for the whole dataset on the left, but those controls are
#: repeated for the subset on the right pane. Maybe it's not a good idea for there to be
#: repetition of knowledge in our user interface."
#:
#: He is right, and it had already cost this codebase: one pane launched operators off the
#: ``_OPERATIONS`` card table while another launched them off ``runnable_operators()``, with
#: different labels and different ``save`` defaults, and the two registries drifted in
#: production. So a target is a SCOPE on the one control panel, never a second set of buttons
#: somewhere else.
#:
#: There was a fourth, ``SCOPE_SUBSET`` ("side pane subset"), and it went with the exploration
#: pane on 2026-08-05. Nothing parks a subset any more, so the entry could only ever resolve to
#: its own refusal — a scope the user can pick and that can never run is worse than no scope.
SCOPE_SELECTION = "selected wells"
SCOPE_PLATE = "whole dataset"
SCOPE_REGION = "current region"
RUN_SCOPES = (SCOPE_SELECTION, SCOPE_PLATE, SCOPE_REGION)


def describe_run_target(regions, *, total: int, head: int = 6) -> "Optional[str]":
    """Name the resolved target set, in one sentence, BEFORE the run starts.

    Fractal's pre-run confirmation, which the owner has already accepted as prior art: a job
    tells you what it resolved to rather than making you infer it from what comes back. This
    codebase needs it more than most, because scope is resolved from live state (the plate
    selection, the open region) and the selector only names the RULE — "selected wells" —
    not the answer. "Run" on an accidental whole-plate
    selection and "Run" on the one well the user meant look identical until the compute is
    spent.

    ``regions is None`` is the plate-wide path, so the sentence carries *total* instead: the
    number is the whole point, and "the whole dataset" alone reads the same at 2 wells and at
    1536.

    Returns ``None`` for an empty target — there is no run to describe, and the caller must
    refuse with its own sentence rather than print a cheerful "0 regions".
    """
    if regions is None:
        return (f"this will run on the whole dataset — {int(total)} region"
                f"{'' if int(total) == 1 else 's'}")
    regions = [str(r) for r in regions]
    if not regions:
        return None
    n = len(regions)
    shown = ", ".join(regions[:head])
    if n > head:
        shown += f", ... (+{n - head} more)"
    return f"this will run on {n} region{'' if n == 1 else 's'}: {shown}"


#: How wide the label column may get before it is elided. Elided rather than wrapped so the block
#: stays ONE LINE PER WINDOW and remains scannable at a glance, which is the whole point of it.
VIEW_LABEL_WIDTH = 30


def _roi_spelling(bbox) -> str:
    """``"roi [120.0,340.0 636.0,856.0] um"``, in ``Extent.label()``'s exact words.

    Derived from ``Extent`` rather than copied from it. A second spelling of an ROI box is exactly
    the drift ``_address.py``'s naming law exists to stop, and the console already prints the first
    one on every addressed line.
    """
    from squidxplorer._address import Extent

    return Extent(region_id="", bbox_um=bbox).label().strip()


def distinct_view_regions(views) -> "list":
    """The union of the Views' regions, in first-seen order — the set a run actually iterates.

    THE flattener. ``PlateWindow._open_views_regions`` calls this and so does
    :func:`describe_view_target`, so the printed number and the executed number cannot disagree.
    Two flatteners is how a user comes to believe a region was processed twice.
    """
    seen: set = set()
    out: list = []
    for view in views or ():
        for region in (getattr(view, "regions", None) or ()):
            r = str(region)
            if r not in seen:
                seen.add(r)
                out.append(r)
    return out


def describe_view_target(views, *, action: str = "Run", head: int = 6,
                         label_width: int = VIEW_LABEL_WIDTH) -> "Optional[str]":
    """Print WHICH windows a run is aimed at, and which subset of each — as a block, before it runs.

    Julio, 2026-08-03: "we say smt like 'run on the {selected windows}'. But it has to print which
    windows and subsets thereof are selected. Make sure that it is printed in an organized manner."

    :func:`describe_run_target` is the sibling of this and is the right shape for a flat region
    list; it stays exactly as it is. This one exists because the flattener throws the windows away
    (``PlateWindow._open_views_regions``), so by the time a region list is printed there is nothing
    left that could name a window. Aimed at window-backed Views; the plate and selection paths keep
    ``describe_run_target``, because two spellings of "the whole dataset" is one too many.

    Returns ``None`` when nothing would run — no Views, or Views that hold no regions between them.
    The caller refuses with its own sentence rather than printing a cheerful zero, the same contract
    :func:`describe_run_target` already has.

    The shape::

        Run decon on 3 windows, 12 regions

          [2]  Deconvolution trial   4 regions   A1, A2, A3, A4
          [5]  ROI · B6  ◂ view 2    1 region    B6   roi [120.0,340.0 636.0,856.0] um
          [7]  C3, C4, C5, +5        8 regions   C3, C4, C5, C6, C7, C8, ... (+2 more)

          13 region slots across 3 windows, 12 distinct regions
          B6 is held by 2 windows and will be processed once

    Every column earns its place. **The bracket** is the identity, and it is the same token the log
    prefix prints (``_logpane._address_prefix``) and the title bar shows: a user reading a log line
    and reading this block must be reading the same name for the same thing. **The label** is
    ``View.name``, which is whatever the window has been renamed to — a rename is only worth having
    if something prints the name. **The names** truncate at *head* in the one overflow spelling this
    codebase already uses. **The ROI box** is the subset, in ``Extent.label()``'s words.

    **The last two lines are not decoration.** The target set is DEDUPLICATED, so the per-window
    counts do not sum to what runs. Printing only one of those two numbers is how a user comes to
    believe a region was processed twice, or that one was skipped. So both are printed, and the
    overlapping regions are named.
    """
    views = list(views or ())
    if not views:
        return None
    distinct = distinct_view_regions(views)
    if not distinct:
        return None

    rows = []
    slots = 0
    holders: "dict[str, int]" = {r: 0 for r in distinct}
    for view in views:
        # A window HOLDS a region once, however its own list happens to spell it, so dedupe within
        # the window before counting. Otherwise the per-window count and the reconciliation's
        # "region slots" would be two different numbers for the same window.
        regions = list(dict.fromkeys(str(r) for r in (getattr(view, "regions", None) or ())))
        slots += len(regions)
        for r in regions:
            holders[r] += 1
        wid = getattr(view, "window_id", None)
        ident = f"[{wid}]" if wid is not None else f"[{getattr(view, 'id', '?')}]"
        name = str(getattr(view, "name", "") or "")
        if len(name) > label_width:
            name = name[:label_width - 1] + "…"
        n = len(regions)
        count = f"{n} region{'' if n == 1 else 's'}"
        shown = ", ".join(regions[:head])
        if n > head:
            shown += f", ... (+{n - head} more)"
        bbox = getattr(view, "roi_bbox", None)
        if bbox is not None:
            shown = f"{shown}   {_roi_spelling(bbox)}" if shown else _roi_spelling(bbox)
        rows.append((ident, name, count, shown))

    id_w = max(len(r[0]) for r in rows)
    label_w = max(len(r[1]) for r in rows)
    count_w = max(len(r[2]) for r in rows)
    nw, nr = len(views), len(distinct)
    lines = [f"{action} on {nw} window{'' if nw == 1 else 's'}, "
             f"{nr} region{'' if nr == 1 else 's'}", ""]
    lines += [f"  {i:<{id_w}}  {lab:<{label_w}}  {c:<{count_w}}  {s}".rstrip()
              for i, lab, c, s in rows]
    lines.append("")

    if slots == nr:
        lines.append(f"  {nr} region{'' if nr == 1 else 's'} across "
                     f"{nw} window{'' if nw == 1 else 's'}")
    else:
        lines.append(f"  {slots} region slots across {nw} window{'' if nw == 1 else 's'}, "
                     f"{nr} distinct region{'' if nr == 1 else 's'}")
        dups = [(r, k) for r, k in holders.items() if k > 1]
        if len(dups) == 1:
            r, k = dups[0]
            lines.append(f"  {r} is held by {k} windows and will be processed once")
        elif dups:
            shown = ", ".join(f"{r} ×{k}" for r, k in dups[:head])
            if len(dups) > head:
                shown += f", ... (+{len(dups) - head} more)"
            lines.append(f"  {len(dups)} regions are held by more than one window and will each "
                         f"be processed once: {shown}")
    return "\n".join(lines)


def resolve_run_scope(scope: str, *, selection=None,
                      current_region=None) -> "tuple[Optional[list], Optional[str]]":
    """Turn the selector's value into the region list a run is aimed at.

    Returns ``(regions, problem)``. ``regions is None`` means the whole dataset — the historical
    plate-wide path, unchanged. ``problem`` is a SENTENCE for the status line and, when it is
    set, nothing should run: a scope the user chose but that has nothing behind it must be said
    out loud, never quietly widened to the whole plate. Silently running 1536 wells because the
    chosen scope happened to be empty is hours of compute nobody asked for.

    ``SCOPE_SELECTION`` is the default and is deliberately forgiving: with nothing selected it
    IS the whole dataset, which is exactly what the plate did before a selector existed.
    """
    if scope not in RUN_SCOPES:
        return None, (f"{scope!r} is not a run scope — this viewer can aim a run at: "
                      f"{', '.join(RUN_SCOPES)}")
    if scope == SCOPE_PLATE:
        return None, None
    if scope == SCOPE_SELECTION:
        picked = _uniq(selection or [])
        return (picked or None), None
    if not current_region:
        return None, ("no region is open in the viewer, so there is no 'current region' to "
                      "run on — double-click a well first, or pick another scope")
    return [str(current_region)], None


def _uniq(regions: Iterable) -> list:
    """De-duplicate keeping first-seen order. A target lists what the user picked, in their order."""
    return list(dict.fromkeys(str(r) for r in regions))


# --- the Minerva subset ------------------------------------------------------------------------

def subset_selection(regions: Sequence, fovs_per_region: Optional[dict],
                     fov_subsets: Optional[dict] = None) -> list:
    """``[(region, fov), ...]`` for every FOV of every region in a region subset.

    KEPT WITH NO GUI CALLER, deliberately, and this note is the reason. Its one caller was the
    exploration tab's "Open in Minerva" button, removed with the pane on 2026-08-05; the plate's
    own export reads :meth:`PlateWindow.minerva_selection` instead. It is retained because it is
    the region-set -> (region, fov) expansion the Minerva work is building FOV-level subsets on,
    it is pure and fully tested (``tests/test_run_scope.py``), and deleting a rule someone is
    mid-way through adopting costs more than the forty lines it saves. If the Minerva work lands
    without it, delete it then — but delete it knowingly.

    ``_minerva.export_selection`` groups the pairs back by region and fuses each into ONE mosaic
    — a FOV subset of a region yields the crop of that region's mosaic, still one file. That
    contract is not touched here; this only decides WHAT is exported.

    *fov_subsets* is ``{region: [fov, ...]}`` for the regions the user boxed only PART of on the
    plate (``PlateOverview.fov_subsets``). A region present there contributes only those fields;
    a region absent from it expands to all of its fields. Omitting the argument keeps the whole
    behaviour, which is what the CLI and the tests want.

    Refuses, by name, any region it cannot expand. Exporting three of the four regions asked for
    and saying nothing is precisely the silent failure this project has shipped six of. A boxed
    subset is validated against the acquisition too: a field the plate offers that the metadata
    does not is a disagreement, not something to quietly export.
    """
    regs = _uniq(regions)
    if not regs:
        raise ValueError("there are no regions to export")
    per = fovs_per_region or {}
    picked = fov_subsets or {}
    out: list = []
    for region in regs:
        fovs = per.get(region)
        if not fovs:
            raise ValueError(
                f"region {region!r} has no fields of view in this acquisition, so it cannot be "
                "fused into a mosaic for Minerva. Nothing was exported.")
        chosen = [int(f) for f in (picked.get(region) or fovs)]
        unknown = [f for f in chosen if f not in set(int(x) for x in fovs)]
        if unknown:
            raise ValueError(
                f"region {region!r} has no field(s) {unknown} in this acquisition. Nothing was "
                "exported.")
        out.extend((region, f) for f in chosen)
    return out
