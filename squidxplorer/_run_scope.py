"""What a run is aimed at, and how that target is said out loud. No Qt, no napari."""

from __future__ import annotations

from typing import Iterable, Optional

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
]


def operator_busy(worker, retired) -> bool:
    """Is an operator run still alive? Workers declaring ``IS_PREVIEW`` do not count."""
    if worker is not None and worker.isRunning():
        return True
    return any(w.isRunning() for w in retired if not getattr(w, "IS_PREVIEW", False))


#: The scopes an operator run can be aimed at, in the order the selector lists them.
SCOPE_SELECTION = "selected wells"
SCOPE_PLATE = "whole dataset"
SCOPE_REGION = "current region"
RUN_SCOPES = (SCOPE_SELECTION, SCOPE_PLATE, SCOPE_REGION)


def describe_run_target(regions, *, total: int, head: int = 6) -> "Optional[str]":
    """Name the resolved target set in one sentence before the run starts; ``None`` when empty.

    ``regions is None`` is the plate-wide path, so the sentence carries *total* instead.
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


#: How wide the label column may get before it is elided (one line per window).
VIEW_LABEL_WIDTH = 30


def _roi_spelling(bbox) -> str:
    """``"roi [120.0,340.0 636.0,856.0] um"``, in ``Extent.label()``'s exact words."""
    from squidxplorer._address import Extent

    return Extent(region_id="", bbox_um=bbox).label().strip()


def distinct_view_regions(views) -> "list":
    """The union of the Views' regions, in first-seen order — the set a run actually iterates."""
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
    """Print which windows a run is aimed at, and which subset of each; ``None`` when nothing runs."""
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
        # Dedupe within the window, so per-window counts match the reconciliation's slots.
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
    """Turn the selector's value into ``(regions, problem)``; ``regions is None`` = whole dataset.

    When ``problem`` is set nothing should run — an empty scope is said out loud, never
    quietly widened to the whole plate.
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
    """De-duplicate keeping first-seen order."""
    return list(dict.fromkeys(str(r) for r in regions))

