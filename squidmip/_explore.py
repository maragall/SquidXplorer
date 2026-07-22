"""Pane 3 — the EXPLORATION pane's rules. No Qt, no napari.

Julio's definition, which this module is shaped around:

    "The central pane and right pane are viewers. The right pane is essentially a COPY of the
    central pane, but it occurs on a SUBSET. The user can play with a central viewer and have
    things on the side that they may want to compute or run Minerva Author on."

So pane 3 is not a control strip that happens to sit next to a viewer. It is a viewer — built
by the same constructor as pane 2 (``_napari_pane.make_pane``) — aimed at a subset of the
plate, with a slider under it and controls that run operators on exactly that subset.

Everything in here is a DECISION rather than a widget, and it lives outside ``_viewer`` for two
reasons. It is testable with no display, no GL and no napari (``tests/test_explore.py``), and
``_viewer`` is a 6000-line module three agents are editing at once.

What is decided here
--------------------
``preview_tab_key`` / ``preview_tab_label``
    A preview run opens its own tab, keyed by acquisition + operator + region set, so two
    preview runs COEXIST as two tabs and can be compared. A hand-picked selection tab
    (``exploration_tab_key``) over the same wells is a different tab again.
``SubsetCursor``
    The model behind the slider under the tab's viewer: which region of the subset is in front,
    and which of them have been loaded into the viewer already.
``subset_layer_op``
    The napari layer-group name a region's result lands on. One group PER REGION, because
    progress must appear as real layers arriving, not as one layer whose data is overwritten
    when the run ends.
``subset_selection``
    The ``[(region, fov), ...]`` a Minerva export of this tab's subset is built from. A region
    is a mosaic of FOVs, so a region expands to ALL of its FOVs — and a region that cannot be
    expanded is named in an exception rather than silently dropped from the export.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Optional, Sequence

__all__ = [
    "operator_busy",
    "RUN_SCOPES",
    "SCOPE_PLATE",
    "SCOPE_REGION",
    "SCOPE_SELECTION",
    "SCOPE_SUBSET",
    "resolve_run_scope",
    "exploration_tab_key",
    "exploration_tab_label",
    "preview_tab_key",
    "preview_tab_label",
    "PREVIEW_PREFIX",
    "SubsetCursor",
    "progress_sentence",
    "subset_layer_op",
    "subset_selection",
]

#: Namespace for a tab opened BY a preview run, as opposed to one opened by a selection.
PREVIEW_PREFIX = "preview:"


def operator_busy(worker, retired) -> bool:
    """Is an OPERATOR RUN still alive? The question ``run_operator`` has to ask before starting.

    Deliberately NOT the same question as "is any producer thread still alive", which is what the
    window's ``_busy()`` answers and what a deferred tab re-sync has to wait for. Conflating the
    two shipped a real defect: ``_retire`` parks the RAW PREVIEW in the same list, and opening a
    side-pane tab re-scopes and restarts that preview — so the retired one was still draining when
    the user launched the next operator, and the run refused itself with "already processing — let
    the current run finish first". The user had started nothing to finish; from their side the
    button simply did not work.

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
#: He is right, and it had already cost this codebase: pane 1 launched operators off the
#: ``_OPERATIONS`` card table while pane 3 launched them off ``runnable_operators()``, with
#: different labels and different ``save`` defaults, and the two registries drifted in
#: production. So "run this on the subset" is a SCOPE on the one control panel, not a second
#: set of buttons in a second pane. Pane 3 owns the subset; pane 1 reads it.
SCOPE_SELECTION = "selected wells"
SCOPE_PLATE = "whole dataset"
SCOPE_REGION = "current region"
SCOPE_SUBSET = "side pane subset"
RUN_SCOPES = (SCOPE_SELECTION, SCOPE_PLATE, SCOPE_REGION, SCOPE_SUBSET)


def resolve_run_scope(scope: str, *, selection=None, current_region=None,
                      parked_subset=None) -> "tuple[Optional[list], Optional[str]]":
    """Turn the selector's value into the region list a run is aimed at.

    Returns ``(regions, problem)``. ``regions is None`` means the whole dataset — the historical
    plate-wide path, unchanged. ``problem`` is a SENTENCE for the status line and, when it is
    set, nothing should run: a scope the user chose but that has nothing behind it must be said
    out loud, never quietly widened to the whole plate. Silently running 1536 wells because the
    side pane happened to be empty is hours of compute nobody asked for.

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
    if scope == SCOPE_REGION:
        if not current_region:
            return None, ("no region is open in the viewer, so there is no 'current region' to "
                          "run on — double-click a well first, or pick another scope")
        return [str(current_region)], None
    parked = _uniq(parked_subset or [])
    if not parked:
        return None, ("the side pane has no subset parked in it, so there is nothing to run on "
                      "at that scope — Shift-drag a few wells on the plate first")
    return parked, None


def _uniq(regions: Iterable) -> list:
    """De-duplicate keeping first-seen order. The tab lists what the user picked, in their order."""
    return list(dict.fromkeys(str(r) for r in regions))


# --- tab identity ------------------------------------------------------------------------------

def exploration_tab_key(acq_id: str, regions) -> str:
    """Stable, CONTENT-ADDRESSED id for an exploration tab (IMA-205).

    Same acquisition + same SET of regions -> the same key, whichever order the user selected
    them in. That gives dedupe for free: re-selecting the same wells focuses the tab that is
    already open instead of piling up duplicates on every stray drag.

    ``acq_id`` is hashed in deliberately. Well ids repeat across plates ("B2" exists on every
    one), so a key built from regions alone would let a selection on a NEW acquisition dedupe
    onto a stale tab left over from the old one. ``ingest`` also closes exploration tabs, but
    the key has to be safe on its own rather than relying on that.
    """
    uniq = sorted(set(str(r) for r in regions))
    if not uniq:
        raise ValueError("an exploration tab needs at least one region")
    digest = hashlib.sha1("\x1f".join([acq_id, *uniq]).encode("utf-8")).hexdigest()[:10]
    return f"exp:{digest}"


def exploration_tab_label(regions) -> str:
    """Human-readable tab title — 'B2–B5 (4)'. The hash is the internal key, never the label."""
    uniq = sorted(set(str(r) for r in regions))
    if not uniq:
        return "exploration"
    if len(uniq) == 1:
        return uniq[0]
    return f"{uniq[0]}–{uniq[-1]} ({len(uniq)})"


def preview_tab_key(acq_id: str, op_key: str, regions) -> str:
    """Id of the tab a PREVIEW RUN of ``op_key`` over ``regions`` opens in pane 3.

    The operator is part of the identity. Julio asked for preview runs to open tabs "so that
    they look at how it is behaving", plural — running MIP and then stitch on one selection has
    to give two tabs to compare, not one tab that the second run steals. Keyed the same way as
    a selection tab otherwise (set-based, acquisition-scoped), and in its OWN namespace so a
    preview tab and a hand-picked tab over the same wells can both be open.
    """
    uniq = sorted(set(str(r) for r in regions))
    if not uniq:
        raise ValueError("a preview tab needs at least one region")
    payload = "\x1f".join([acq_id, str(op_key), *uniq])
    return f"{PREVIEW_PREFIX}{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:10]}"


def preview_tab_label(op_label: str, regions) -> str:
    """Tab title for a preview run — 'MIP · B2–B5 (3)'. Never the hash."""
    return f"{op_label} · {exploration_tab_label(regions)}"


# --- the slider under the tab's viewer ----------------------------------------------------------

class SubsetCursor:
    """Which region of a tab's subset is in front, and which are already in its viewer.

    The slider under the exploration viewer drives this. It is a MODEL rather than a widget so
    the clamping and the "did this actually move?" answer are testable without a display — and
    the second answer matters: every genuine move re-aims a viewer and may start a mosaic read,
    so a cursor that reports a move it did not make costs real I/O on every stray Qt event.

    ``loaded`` is deliberately here and not in the widget. A tab accumulates its regions'
    mosaics as layers, so "have I already got this one?" is part of the cursor's job, and a
    second copy of that set living in the widget is the two-places-hold-one-fact defect this
    project keeps shipping.
    """

    def __init__(self, regions: Sequence) -> None:
        regs = _uniq(regions)
        if not regs:
            raise ValueError("an exploration tab needs at least one region")
        self._regions = regs
        self._index = 0
        self._loaded: set = set()

    def __len__(self) -> int:
        return len(self._regions)

    @property
    def regions(self) -> list:
        return list(self._regions)

    @property
    def index(self) -> int:
        return self._index

    @property
    def region(self) -> str:
        return self._regions[self._index]

    def set_index(self, index: int) -> bool:
        """Move the cursor, CLAMPED to the subset. Returns whether it actually moved.

        Clamps rather than raising: a slider hands over whatever integer it has, and refusing an
        out-of-range one would turn a harmless UI event into an exception in a Qt slot — where
        it would be swallowed and the pane would simply stop responding.
        """
        index = max(0, min(int(index), len(self._regions) - 1))
        if index == self._index:
            return False
        self._index = index
        return True

    def is_loaded(self, region) -> bool:
        return str(region) in self._loaded

    def mark_loaded(self, region) -> None:
        region = str(region)
        if region not in self._regions:
            raise ValueError(
                f"{region!r} is not in this tab's subset ({self._regions}) — marking it loaded "
                "would make the tab's claim about what it is showing false")
        self._loaded.add(region)

    def pending(self) -> list:
        """Regions of the subset not yet in the viewer, in the tab's order."""
        return [r for r in self._regions if r not in self._loaded]


# --- what the user is told while a preview run computes ------------------------------------------

def progress_sentence(op_label: str, done: int, total: int, failed: int = 0) -> str:
    """One line saying how far a preview run over this subset has got.

    Counts REGIONS, because a region is the unit the pane shows (a mosaic of FOVs), and because
    the worker's own progress signal counts wells rather than fields for the same reason.

    The word "live" is deliberately absent: this is a post-acquisition tool, and the honest
    phrasing for what is happening is operator iteration, not a live feed.
    """
    total = int(total)
    if total <= 0:
        raise ValueError(f"a run over {total} regions has nothing to report")
    done = max(0, int(done))
    unit = "region" if total == 1 else "regions"
    out = f"{op_label} · {done} of {total} {unit} computed"
    if failed:
        out += f" · {int(failed)} failed"
    if done >= total:
        out += " · complete"
    return out


# --- napari layer naming --------------------------------------------------------------------------

def subset_layer_op(op_label: str, region: str) -> str:
    """The napari layer-GROUP name one region's result belongs to inside a tab's viewer.

    One group per (operator, region). Julio: "layers don't update in the napari mosaic... you
    instantiate an actual layer to be in the napari interface." So a run over four regions puts
    four groups on the canvas, each appearing the moment that region finishes, each at its own
    stage-coordinate box — instead of one group whose pixels are replaced at the end, which is
    the behaviour he was describing as broken.

    ``MosaicLayers`` never parses this back (identity is carried in ``layer.metadata``); it is a
    key and a human label, nothing more.
    """
    return f"{op_label} · {region}"


# --- the Minerva subset ------------------------------------------------------------------------

def subset_selection(regions: Sequence, fovs_per_region: Optional[dict]) -> list:
    """``[(region, fov), ...]`` for every FOV of every region in this tab's subset.

    This is what pane 3 hands to ``_minerva.export_selection``, which groups it back by region
    and fuses each into ONE mosaic — a FOV subset of a region yields the crop of that region's
    mosaic, still one file. That contract is not touched here; this only decides what the
    exploration tab's "Open in Minerva" button is scoped to, which is the tab's own subset
    rather than whatever happens to be selected on the plate.

    Refuses, by name, any region it cannot expand. Exporting three of the four regions a tab
    displays and saying nothing is precisely the silent failure this project has shipped six of.
    """
    regs = _uniq(regions)
    if not regs:
        raise ValueError("this tab has no regions to export")
    per = fovs_per_region or {}
    out: list = []
    for region in regs:
        fovs = per.get(region)
        if not fovs:
            raise ValueError(
                f"region {region!r} has no fields of view in this acquisition, so it cannot be "
                "fused into a mosaic for Minerva. Nothing was exported.")
        out.extend((region, int(f)) for f in fovs)
    return out
