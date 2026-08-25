"""One operator run's identity and books — CONTEXT's "Operator run", as an object.

Created by ``PlateWindow.run_operator`` once every refusal is behind it, closed by the drain
slot. Qt-free on purpose: the window owns the signals and forwards into this, so the facts of a
run live in one place instead of ~15 loose ``_run_*`` fields with settle logic across three slots.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from squidxplorer import _measure


class OperatorRun:
    """The in-flight (or latest finished) run's identity and books.

    ``label`` and ``requester`` are consumed by the drain's requester pair so a second drain
    cannot re-report; ``action`` is consumed the same way by the console's started/done pair.
    """

    def __init__(self, *, key: str, layer_key: str, label: Optional[str],
                 action: Optional[str], dest: str, address: Any, requester: Any,
                 is_partial: bool, t0: float, scope: Optional[dict] = None):
        self.key = key
        self.layer_key = layer_key    # the plate layer this run's tiles stream into
        self.label = label            # the bare action label reported to the requester
        self.action = action          # the console's started/done pair; consumed at the drain
        self.dest = dest              # status-line suffix: " → out.hcs" / " (preview — not saved)"
        self.address = address        # Extent for a one-region run, else None
        self.requester = requester    # the window that asked; told the operator_* callbacks
        self.is_partial = is_partial  # the run covers PART of each well ({region: [fov, ...]})
        self.t0 = t0                  # the user's gesture on the perf_counter clock (first paint)
        self.began = time.monotonic()
        self.error: Optional[str] = None
        self.accs: dict = {}          # region id -> RegionResultAccumulator, still filling
        #: The run's `{region: [fov, ...]}` when it is scoped to fields (an ROI preview), else
        #: None: what each region's accumulator OWES (Julio, 2026-08-25, sub-FOV decon).
        self.scope: Optional[dict] = dict(scope) if scope else None

    def settle_stranded(self, deliver: Callable[[str, Any], None]) -> list:
        """The run has ended: resolve every region whose result was still being accumulated.

        Returns one sentence per region that ended with NO layer, so the caller can report the
        run honestly. Always leaves ``accs`` empty.

        THE MEASURED DEFECT (2026-08-06), on Julio's own 10x acquisition and reported by him as
        "the controls now brings plate view, but doesn't open the operator tab". FOVs 17 and 19
        of ``manual0`` are corrupt (``TiffFileError: suspicious number of tags``). Per-field
        fault isolation skipped them, 25 of 27 FOVs landed, and ``_on_result``'s ``if not
        acc.complete(): return`` left the accumulator sitting in the books for the rest of the
        process. Nothing ever flushed it, so **no layer was ever built** -- while the plate
        printed "✓ Maximum Intensity Projection · 1 well" and the window that asked was told
        "finished in 4.6 s". A run that produced no pixels, announced twice as a success.

        A partial region is still REFUSED -- half a mosaic drawn as a layer reads as something
        the operator did, and :meth:`squidxplorer._op_result.RegionResultAccumulator.result` owns
        that rule. Its sentence is reused verbatim rather than re-worded here, so there is one
        wording of the refusal. A COMPLETE accumulator here would be one ``_on_result`` never got
        to flush; it is delivered rather than discarded.
        """
        accs, self.accs = dict(self.accs), {}
        stranded: list = []
        for region, acc in accs.items():
            try:
                result = acc.result()
            except ValueError as exc:            # incomplete: the accumulator's own sentence
                stranded.append(f"{acc.op} · {region}: {exc}")
                continue
            deliver(acc.op, result)
        if stranded and not self.error:
            # So the drain tells the window that ASKED that its run failed. Without this the
            # window's bar closed on ``operator_done`` over a run with nothing to show.
            self.error = "  ".join(stranded)
        return stranded

    def close(self, landed: Optional[int], stranded: int, elapsed: float) -> "tuple[str, str]":
        """``(outcome, detail)`` for the console's done/failed pair, in ``_measure``'s words.

        A run that landed nothing is a FAILURE however politely the engine returned, as is one
        that stranded a region. The sentences are the GUI's own; ``_measure.verdict``'s
        counts-based detail belongs to the surfaces that own the counts (the writer's manifest
        and the command layer).
        """
        if landed == 0:
            return _measure.FAILED, f"produced nothing after {elapsed:.1f} s"
        if stranded:
            return _measure.FAILED, (f"{stranded} region(s) landed no layer - some of "
                                     f"their fields could not be read")
        return _measure.OK, ""
