"""The Minerva Author hand-off, as its own product (IMA-228), out of ``PlateWindow``.

The whole cluster — the tab UI, the selection reading, the export run and the render run —
travels together because it IS one product: nothing else consumes ``minerva_selection`` and
nothing here is plate machinery. ``PlateWindow`` keeps thin delegates (tests, ``tools/gates.py``
and ``tools/walkthrough.py`` actuate ``run_minerva_export`` / ``minerva_selection`` /
``_build_minerva_tab`` by name on the window), and it stays the owner of two things on purpose:

* **the worker slots** ``_minerva`` / ``_minerva_render`` — thread LIFETIME belongs to the
  window, whose ``_retire`` / ``_join_retired`` / ``closeEvent`` seams are what keep a running
  QThread from being destroyed with the process; tests also read ``win._minerva`` directly;
* **the worker names** — this module resolves ``_MinervaWorker`` / ``_MinervaRenderWorker``
  through ``squidxplorer._workers`` AT CALL TIME, because that module attribute — the module
  that owns the classes — is the seam tests monkeypatch with spies. Importing the classes here
  would silently stop the spies intercepting while the tests kept passing.

Everything else the panel needs arrives as a named constructor dependency (live getters where
the window's state is mutable — the reader is replaced on every ingest), so what this product
consumes is written in one signature instead of discovered by reading four methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QWidget,
)

from squidxplorer._engine import available_plane_operators
from squidxplorer._minerva import MINERVA_HOME_ENV, MINERVA_URL, NEEDS_INTERNET_NOTE
from squidxplorer._operations import _OPERATIONS_BY_KEY
from squidxplorer._qtstyle import BTN_QSS, CHECK_QSS, COMBO_QSS, hline
from squidxplorer._worker_lifecycle import launch as _launch_worker


def _workers():
    """The Minerva worker classes, read off ``_workers`` at call time — the monkeypatch seam."""
    from squidxplorer import _workers as W
    return W._MinervaWorker, W._MinervaRenderWorker


class MinervaPanel:
    """The Minerva export/render product: one instance per :class:`PlateWindow`, built lazily."""

    def __init__(self, owner, *,
                 reader: Callable, meta: Callable, time_point: Callable, current_well: Callable,
                 say: Callable[[str], None],
                 selected_region_fovs: Callable,
                 on_screen_luts: Callable,
                 tab_shell: Callable):
        #: The QThread-lifetime owner and dialog parent. Worker slots live on IT (see module
        #: docstring); this panel never keeps a thread of its own.
        self._owner = owner
        self._reader = reader
        self._meta = meta
        self._time_point = time_point
        self._current_well = current_well
        self._say = say
        self._selected_region_fovs = selected_region_fovs
        self._on_screen_luts = on_screen_luts
        self._tab_shell = tab_shell

    # -- the selection ---------------------------------------------------------------------------
    def selection(self) -> list:
        """The ``[(region, fov), ...]`` the user actually selected — never a silent stand-in.

        The requirement is "open minerva-author with the selected region(s)", so this reads the
        selection instead of inventing one. Exactly two sources, in order:

        1. ``selected_region_fovs`` — **the window's** selection. ``PlateOverview`` is
           display-only: it maps grid cells to well ids and emits them, and ``PlateWindow`` is
           where they land (``_on_selection_changed`` -> ``_selected_regions``) because
           expanding a well to its FOVs needs ``fovs_per_region``, which only that side has.
           The previous version probed the overview too and fell back to
           ``PlateOverview.selected_wells()``; the overview never had a
           ``selected_region_fovs`` and the fallback was what made the export appear to work at
           all — a duck-typed chain standing in for reading the selection from its owner.
        2. The region open in the detail viewer (``current_well``): every FOV of it.

        Note the unit. The pairs are ``(region, fov)`` but the export groups them BY REGION and
        fuses each into one mosaic — a region is a mosaic containing an array of FOVs, never a
        FOV. Selecting a whole region yields all its FOVs here and one fused mosaic downstream.

        A region the user boxed only PART of yields only those FOVs, and downstream that is the
        same one mosaic CROPPED to them. Nothing here decides that: ``selected_region_fovs``
        reads the plate's own FOV subsets, and this method's job is only to fall back to the
        detail viewer's well when the plate has no selection at all.

        Nothing selected returns ``[]`` — the caller says so rather than exporting fov 0 of 36
        and calling it "the selected well".
        """
        fovs_per_region = (self._meta() or {}).get("fovs_per_region", {}) or {}

        def expand(regions) -> list:
            out = []
            for region in regions:
                out.extend((str(region), int(f)) for f in fovs_per_region.get(str(region), []))
            return out

        sel = [(str(r), int(f)) for r, f in self._selected_region_fovs()
               if int(f) in fovs_per_region.get(str(r), [])]
        if sel:
            return sel
        if self._current_well():
            return expand([self._current_well()])
        return []

    # -- the tab UI ------------------------------------------------------------------------------
    def build_tab(self) -> QWidget:
        """Minerva Author hand-off (IMA-228): export the SELECTION, then open Author on it.

        Scope comes from :meth:`selection` — the plate's selected wells (all of their
        FOVs, or the fields a Shift+Alt box picked inside a mosaic), else the well open in the
        detail viewer, which means every FOV of it.

        ONE FILE PAIR PER REGION, not per FOV. This docstring said "one file pair per FOV
        (Minerva opens one 2D image at a time and SquidXplorer has no stitcher)" long after both
        halves of that stopped being true: there IS a stitcher (the region-operator seam
        ``export_selection`` fuses through), and a region's FOVs become ONE mosaic because
        Minerva lays out exactly one image (``"Layout": {"Grid": [["i0"]]}``, hardcoded) and
        opens only ``series[0]``. A FOV subset is that mosaic CROPPED, still one file.

        The timepoint is the one the window is showing — :meth:`run_export` reads the window's
        ``time_point`` — so there is no control for it here and none is missing.
        """
        op = _OPERATIONS_BY_KEY["minerva"]
        w, v = self._tab_shell(
            op.label,
            "Writes an OME-TIFF plus a Minerva story for every selected region, at the timepoint "
            "the plate is showing, then starts Minerva Author. Zoom into a well and Shift+Alt-drag "
            "a box to export only the fields inside it - the mosaic is cropped to them. Author’s "
            "editor cannot be pointed at a file, so pick the .story.json below in its “Select "
            "File” dialog - the colours and contrast are already applied. To skip that step "
            "entirely, render a viewer instead (button below the paths).",
        )
        state = {"dir": None, "pairs": []}

        dir_lbl = QLabel("(defaults to a minerva_export folder in your home directory)")
        dir_lbl.setWordWrap(True)
        dir_lbl.setStyleSheet("color:#8b98ad;font-size:12px;")

        # Projection mode — the salesperson tool (squid2minerva convert.py) offers --mip/--z, so
        # hardcoding one here would be a capability regression. Driven by the operator registry.
        row = QHBoxLayout(); row.setSpacing(6)
        row.addWidget(QLabel("Projection"))
        proj = QComboBox(); proj.setStyleSheet(COMBO_QSS)
        proj.addItems(available_plane_operators())
        proj.setCurrentText("mip")
        row.addWidget(proj); row.addStretch(1)

        # "channels need to be set to specific colors" - the colours ON SCREEN, which the export's
        # own defaults (acquisition display_color + 1/99.9 percentiles) do not know about. Checked
        # by default because matching what you are looking at is the request; harmless with no view
        # open, because on_screen_luts() returns None there and the defaults apply unchanged.
        luts_cb = QCheckBox("Match the LUTs of the focused view window")
        luts_cb.setStyleSheet(CHECK_QSS)
        luts_cb.setChecked(True)
        luts_cb.setToolTip(
            "Use the contrast and colour you have on screen in the focused view window instead of "
            "the acquisition's channel colours and an automatic 1/99.9 percentile stretch.\n\n"
            "With no view window open there is nothing on screen to match and the automatic "
            "values are used. A channel that is not in that window keeps the automatic values "
            "too, and so does a channel showing a multi-stop colormap (viridis, turbo): Minerva "
            "stores one colour per channel and cannot hold a gradient.")

        launch_cb = QCheckBox("Open Minerva Author after exporting")
        launch_cb.setStyleSheet(CHECK_QSS)
        launch_cb.setChecked(True)

        path_lbl = QLabel("")
        path_lbl.setWordWrap(True)
        path_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path_lbl.setStyleSheet("color:#8b98ad;font-size:11px;")
        copy_btn = QPushButton("Copy story path"); copy_btn.setStyleSheet(BTN_QSS); copy_btn.hide()
        reveal_btn = QPushButton("Show in folder"); reveal_btn.setStyleSheet(BTN_QSS); reveal_btn.hide()
        # THE ZERO-CLICK DESTINATION. A separate button and not a replacement for the Author
        # launch: Author is the EDITOR (waypoints, story text, masks) and needs its Select File
        # click because its server has no route, flag or URL that opens a file; render.py is the
        # VIEWER and needs none. Julio said "viewer". Both are offered; neither is assumed.
        render_btn = QPushButton("Render a Minerva viewer (no file picking)")
        render_btn.setStyleSheet(BTN_QSS); render_btn.hide()
        render_btn.setToolTip(
            "Runs Minerva's own render.py on what you just exported and opens the finished "
            "exhibit. No Select File step.\n\n"
            "It is a viewer, not an editor: no waypoints, story text or masks.\n"
            "It writes a JPEG pyramid, so it is lossy; the OME-TIFF is untouched.\n"
            "Measured on this machine: about 2 s for a 2048x2048 4-channel crop and about "
            "132 s for a whole 11535x9635 4-channel region, plus about 13 s once per session "
            "while Minerva's renderer loads.\n"
            + NEEDS_INTERNET_NOTE)

        def pick():
            d = QFileDialog.getExistingDirectory(self._owner, "Save the Minerva export to folder")
            if not d:
                return
            state["dir"] = d
            dir_lbl.setText(d)

        def on_exported(pairs):
            state["pairs"] = pairs
            if not pairs:
                # An export that wrote NOTHING must not leave the previous one's paths on screen
                # with live Copy / Show in folder / Render buttons under them. `state["pairs"]` was
                # already emptied above, so the buttons had quietly become no-ops while still
                # naming files — a control that looks armed and does nothing.
                path_lbl.setText("")
                copy_btn.hide(); reveal_btn.hide(); render_btn.hide()
                return
            path_lbl.setText("\n".join(str(story) for _, story in pairs))
            copy_btn.show(); reveal_btn.show(); render_btn.show()

        def do_render():
            if state["pairs"]:
                self.run_render(state["pairs"])

        def do_copy():
            if state["pairs"]:
                QApplication.clipboard().setText("\n".join(str(s) for _, s in state["pairs"]))
                self._say("story path copied")

        def do_reveal():
            if state["pairs"]:
                from squidxplorer._minerva import reveal
                reveal(state["pairs"][0][1])

        pick_btn = QPushButton("Choose output folder…"); pick_btn.setStyleSheet(BTN_QSS)
        pick_btn.clicked.connect(pick)
        # Named for the UNIT that lands: one fused mosaic per selected region, cropped to the
        # fields you boxed. "Export the selected FOVs" promised N files and wrote one per region.
        run = QPushButton("Export the selection (one mosaic per region)")
        run.setStyleSheet(BTN_QSS)
        # Through the OWNER's delegate, not self.run_export: gates and tests stub
        # `run_minerva_export` on the window, and a click must hit whatever they installed.
        run.clicked.connect(lambda: self._owner.run_minerva_export(
            out_dir=state["dir"], z_operator=proj.currentText(),
            launch=launch_cb.isChecked(), on_exported=on_exported,
            luts=self._on_screen_luts() if luts_cb.isChecked() else None,
        ))
        copy_btn.clicked.connect(do_copy)
        reveal_btn.clicked.connect(do_reveal)
        render_btn.clicked.connect(do_render)

        net_lbl = QLabel(NEEDS_INTERNET_NOTE)
        net_lbl.setWordWrap(True)
        net_lbl.setStyleSheet("color:#8b98ad;font-size:11px;")

        v.addWidget(pick_btn); v.addWidget(dir_lbl)
        v.addLayout(row); v.addWidget(luts_cb); v.addWidget(launch_cb); v.addWidget(run)
        v.addWidget(hline()); v.addWidget(path_lbl); v.addWidget(copy_btn); v.addWidget(reveal_btn)
        v.addWidget(render_btn); v.addWidget(net_lbl)
        v.addStretch(1)
        run.setEnabled(self._reader() is not None)
        return w

    # -- the export run --------------------------------------------------------------------------
    def run_export(self, out_dir=None, z_operator: str = "mip", launch: bool = True,
                   on_exported=None, time_point=None, selection=None, luts=None):
        """Export the user's selection for Minerva Author and (optionally) open it.

        Runs off the GUI thread: projecting a well is real I/O plus compute, and starting
        Minerva Author polls a port for up to 90 s. Tests call this directly with launch=False.
        *selection* overrides :meth:`selection` (tests and future callers). *luts* is
        passed straight through to ``export_selection``: ``None`` means the percentile defaults,
        exactly as before this parameter existed. Deciding whether to match the screen belongs to
        the caller (the Minerva tab's checkbox calls the window's ``on_screen_luts``), not here -
        so this method has no opinion and stays trivially testable in both states.

        *time_point* is ``None`` by default, meaning THE TIMEPOINT THE WINDOW IS SHOWING. It used
        to default to the literal ``0`` and both GUI call sites took the default, so a
        multi-timepoint acquisition exported frame 0 whatever the timepoint bar said — the pixels
        on screen and the pixels in the OME-TIFF were different images, and nothing said so. An
        explicit *time_point* still wins, which is what keeps the CLI and the tests able to name
        one.
        """
        owner = self._owner
        meta = self._meta()
        if self._reader() is None or meta is None:
            self._say("open an acquisition first")
            return
        time_point = self._time_point() if time_point is None else int(time_point)
        if owner._minerva is not None and owner._minerva.isRunning():
            self._say("already exporting — let the current export finish first")
            return

        sel = list(selection) if selection is not None else self.selection()
        if not sel:
            self._say(
                "nothing selected — pick the well or FOVs to export "
                "(double-click a well on the plate), then export again")
            return

        # The export unit is a REGION (one fused mosaic each), so count regions, not FOVs.
        regions = list(dict.fromkeys(r for r, _ in sel))
        # ...but a region can now be CROPPED to some of its fields, and that changes the file that
        # lands. Say how many are cropped rather than letting "3 mosaics" mean either thing.
        per = (meta.get("fovs_per_region") or {})
        cropped = [r for r in regions
                   if 0 < len({f for rr, f in sel if rr == r}) < len(per.get(r) or [])]
        what = (f"{len(regions)} mosaic{'s' if len(regions) != 1 else ''} "
                f"({', '.join(regions)}, {len(sel)} FOVs"
                + (f", {len(cropped)} cropped" if cropped else "") + ")")
        n_t = meta.get("n_t", 1) or 1
        t_note = f" (t={time_point} of {n_t})" if n_t > 1 else ""
        minerva_worker_cls, _ = _workers()
        w = minerva_worker_cls(
            self._reader(), sel, out_dir, z_operator, time_point=time_point, launch=launch,
            luts=luts)

        def on_launched(ok):
            if ok:
                # The URL is named and not just implied. Exactly ONE tab is opened now (Minerva
                # Author opens its own on a cold start, so we no longer open a second), and the
                # one way that leaves the user with none is Author's webbrowser call failing to
                # find a browser - in which case it returns False, the server serves anyway, and
                # this line is the address to paste.
                self._say(
                    f"✓ Minerva Author open at {MINERVA_URL} - pick a .story.json "
                    f"({what}{t_note} exported)")
            else:
                self._say(
                    f"✓ exported {what}{t_note} — Minerva Author not found "
                    f"(set ${MINERVA_HOME_ENV} to an explorer checkout)")

        def on_exported_readout(pairs):
            # Report what LANDED, not what was asked for: a stop mid-export writes fewer.
            if not pairs:
                self._say("nothing exported")
                return
            done = regions[: len(pairs)]
            note = "" if len(pairs) == len(regions) else f" of {len(regions)} (stopped)"
            # The SUCCESS line says a mosaic was cropped, not just the in-flight one. This is the
            # line that stays on screen and the only one a user reads after the export, and a crop
            # that reads identically to a whole region is the same silent difference the filename
            # suffix (`_2fov`) exists to prevent on disk.
            crop = [r for r in cropped if r in done]
            crop_note = (f", {len(crop)} cropped to the FOVs you boxed" if crop else "")
            self._say(
                f"✓ exported {len(pairs)} mosaic{'s' if len(pairs) != 1 else ''}{note} from "
                f"{', '.join(done)}{t_note}{crop_note} → {Path(pairs[0][0]).parent}")

        self._say(f"● Minerva export · {what}{t_note} …")
        # The worker lands on the OWNER's `_minerva` slot: thread lifetime is the window's.
        # `exported` keeps its subscriber ORDER: an explicit on_exported first, the readout second.
        _launch_worker(
            owner, w, slot="_minerva",
            on_progress=lambda d, num: self._say(f"● Minerva export · {d}/{num} mosaics"),
            on_problem=lambda m: self._say(f"Minerva export failed: {m}"),
            signals={
                "exported": ([on_exported, on_exported_readout] if on_exported is not None
                             else on_exported_readout),
                "launched": on_launched,
            })

    # -- the render run --------------------------------------------------------------------------
    def run_render(self, pairs, threads=None, open_when_done: bool = True):
        """Render exported ``(ome, story)`` pairs into Minerva exhibits and open the first one.

        The zero-click destination. :meth:`run_export` hands the user to Minerva Author, which
        cannot be pointed at a file and so still needs its "Select File" click; this hands them a
        finished, already-coloured Minerva VIEWER instead. Both exist because they are different
        programs: Author edits, ``render.py`` renders. See
        :func:`squidxplorer._minerva.render_exhibit` for the costs, which are real and measured.

        Runs off the GUI thread. A render is minutes, not seconds.
        """
        owner = self._owner
        if not pairs:
            self._say("export something first - there is nothing to render")
            return
        if getattr(owner, "_minerva_render", None) is not None and owner._minerva_render.isRunning():
            self._say("already rendering - let the current render finish first")
            return
        n = len(pairs)
        _, render_worker_cls = _workers()
        w = render_worker_cls(pairs, threads=threads)

        def on_rendered(indexes):
            if not indexes:
                return                       # `failed` says why; an empty success is not a message
            note = "" if len(indexes) == n else f" of {n}"
            if open_when_done:
                from squidxplorer._minerva import open_exhibit
                open_exhibit(indexes[0])
            self._say(
                f"✓ rendered {len(indexes)} Minerva viewer{'s' if len(indexes) != 1 else ''}{note} "
                f"→ {Path(indexes[0]).parent}. {NEEDS_INTERNET_NOTE}")

        self._say(
            f"● Minerva render · {n} exhibit{'s' if n != 1 else ''} - this takes minutes …")
        # The failure is NAMED in the status line, because render.py runs as a script under a
        # FOREIGN venv: its failure is an exit code plus stderr, and if we do not print it
        # nothing does.
        _launch_worker(
            owner, w, slot="_minerva_render",
            on_progress=lambda d, tot: self._say(f"● Minerva render · {d}/{tot} exhibits"),
            on_problem=lambda m: self._say(f"Minerva render failed: {m}"),
            signals={"rendered": on_rendered})
