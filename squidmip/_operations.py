"""WHICH post-processing operators exist, what they are called, and where their results are filed.

Gap 6 of the GUI backlog plan (2026-07-29), step 3 of the split of ``squidmip/_viewer.py``.

WHY THIS WAS CUT, AND WHY HERE
------------------------------
This is the operator REGISTRY, and it has no Qt in it at all: one frozen dataclass, one tuple of
instances, one dict keyed off that tuple, and four pure functions over them. It sat in the middle of
an 8,000-line QMainWindow module purely by accident of history, which had two costs.

The first is that it could not be tested in isolation, so the fact that the CARD list and the
ENGINE's list are different sets was discovered in production rather than by a unit test.
:func:`runnable_operators` documents the two ways that bit: ``spot`` is a registered projector with
no card, and ``minerva`` is a card that is not an operator at all but an export hand-off.
(``reference`` used to be the first example and is now carded — see the entry below.)
Reading capability off ``_OPERATIONS`` therefore raised bare ``KeyError``s out of the Qt event loop.
The rule this module now states in one place is: **a card is presentation, the engine is capability,
and the two are asked separately.**

The second is that ``squidmip._gui_commands`` needs exactly two of these functions and has to reach
back into ``_viewer`` inside a function body to get them, importing PyQt5, napari and a QMainWindow
to answer "what is this operator called". That import can now point here instead.

WHAT IS IN HERE
---------------
* :class:`Operation`, the declaration of one operator: its key, its card copy, and the name of the
  ``_build_<x>_tab`` method that gathers its parameters.
* ``_OPERATIONS`` and ``_OPERATIONS_BY_KEY``, the declared set and its index.
* ``_SAVE_OPERATOR``, which operator a "save this to disk" button runs. Named rather than
  spelled ``_OPERATIONS[0].key``, so reordering the CARDS cannot silently change what RUNS.
* ``_TO_BE_ADDED``, the roadmap cards.
* :func:`operator_layer_key`, :func:`runnable_operators`, :func:`operator_label` and
  :func:`_action_label`: what a run is filed under, what can actually be run, what to call it on
  screen, and what to call it in the console.

WHAT IS DELIBERATELY NOT IN HERE
--------------------------------
``_BAND_DEFAULT_PX`` and ``_BAND_MAX_PX`` stayed in ``_viewer.py`` (they were
``_TOP_ROW_COMPACT_PX`` and ``_TOP_ROW_READING_PX`` until the band moved under the plate on
2026-08-03). They are the height of the navigator/operator/log band in pixels, which is window
layout, not an operator fact. They only looked like they
belonged because ``_SAVE_OPERATOR``'s comment had drifted three declarations away from
``_SAVE_OPERATOR`` and landed above them; that comment moved back to what it describes.

Nothing here imports Qt, and nothing here imports ``_viewer``. This removed 117 lines from
``_viewer.py``, which went from 4,994 lines to 4,881.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from squidmip._engine import runnable_operators as _runnable_operators


@dataclass(frozen=True)
class Operation:
    """One post-processing operator declared in ONE place — the 'operation template'. Adding a feature
    is a single entry here plus one ``_build_<x>_tab`` method; the console builds a card + menu item
    from ``label``/``blurb``, clicking it opens the tab that ``build_tab`` (a PlateWindow method name)
    returns, and every status text (progress, done) derives from ``label``. A flat record, not a
    subclass tree — no new operator ever edits scattered texts or the dispatch."""
    key: str
    label: str
    blurb: str
    build_tab: str        # name of the PlateWindow method that builds this operator's UI tab

    @property
    def runnable(self) -> bool:
        """Can the ENGINE run this key, as opposed to the card merely existing?

        DERIVED, never declared (2026-08-05). This was a hand-maintained ``bool`` field on the
        card and a test existed for the sole purpose of checking it still agreed with
        ``runnable_operators()`` — i.e. the card table carried a mirror of the engine's membership,
        and the mirror needed a guard. A card is presentation and the engine is capability; asking
        the engine is how you find out, and it cannot be stale.
        """
        return self.key in runnable_operators()

# The operator registry. MIP is operator #1; append an Operation + write its `_build_*_tab` and both
# the console cards and the Process-well-plates menu grow automatically.
_OPERATIONS = (
    Operation("mip", "Maximum Intensity Projection",
              "Collapse each well's z-stack to one max-intensity image; save a navigable OME-Zarr plate.",
              "_build_mip_tab"),
    # The OTHER z-reduction, and the one Julio asked for twice: the engine has had `reference` in
    # `_OPERATORS` since IMA-210 but no card, so it was CLI-only and in no dropdown and no menu.
    # Same builder as MIP (`_build_run_tab` is one builder for every z-reducer), same shape here.
    Operation("reference", "Reference plane (best focus)",
              "Keep each well's sharpest z-plane (Tenengrad) instead of combining them; save a "
              "navigable OME-Zarr plate.",
              "_build_reference_tab"),
    Operation("stitch", "Stitch (register + fuse)",
              "Register every FOV of a well against its neighbours and fuse one seamless mosaic "
              "per well, instead of trusting the stage coordinates alone.",
              "_build_stitch_tab"),
    # NOT an operator: an export hand-off. Handing "minerva" to the engine dies with a raw
    # KeyError: unknown operator 'minerva'. It used to say so with `runnable=False`, a field that
    # only ever restated what the registry already knows -- `Operation.runnable` asks it now, so
    # this card is non-runnable BECAUSE nobody registered "minerva", which is the actual reason.
    Operation("minerva", "Open in Minerva Author",
              "Export the selected FOVs to Minerva-ingestable OME-TIFFs and open Minerva Author on them.",
              "_build_minerva_tab"),
    # IMA-223/224/225 -- the PLANE-OPS. Unlike mip/stitch these keep z at full depth, so they get
    # _build_plane_op_tab (preview only) rather than _build_run_tab. The ORIGINAL reason was that
    # write_plate's _validate_image accepted Z == 1 only and would fail LOUD on save; IMA-277
    # lifted that, so the save path now exists and only this card has not been given it. Preview
    # only is therefore a GUI gap to close, not a contract -- do not cite Z == 1 to justify it.
    # The blurb said "the microscope's Gaussian PSF ... no explicit kernel". Both halves were
    # false and had been since IMA-247 deleted the reimplementation: the kernel is a VECTORIAL
    # PSF computed from the acquisition's own optics (NA 0.3 on this scope), and it is very much
    # explicit. A card that describes the wrong algorithm is how a user picks the wrong operator.
    Operation("decon", "Deconvolution (Richardson-Lucy)",
              "Sharpen against a vectorial PSF computed from this acquisition's own optics (NA, "
              "emission wavelength, pixel size, z-step) -- not an assumed Gaussian. Richardson-Lucy "
              "is semi-convergent, so the iteration count is chosen by eye against a turbo x-z / "
              "y-z view rather than defaulted.",
              "_build_decon_tab"),
    Operation("bgsub", "Background subtraction",
              "Remove the smooth out-of-focus haze from every plane with a rolling ball (ImageJ's "
              "algorithm). A LAYER: the raw is untouched on disk and one toggle away.",
              "_build_bgsub_tab"),
    Operation("flatfield", "Flat-field correction",
              "Divide out the objective's illumination profile so the corners match the centre. "
              "Needs an illumination profile (.npy) from the stitcher or estimated from the plate.",
              "_build_flatfield_tab"),
)
_OPERATIONS_BY_KEY = {op.key: op for op in _OPERATIONS}

# The operator a "save this to disk" button runs. This used to be spelled `_OPERATIONS[0].key`,
# which made a PRESENTATION edit (reordering the cards) silently change which operator the save
# button RUNS. Named, so the two cannot be confused.
#
# NO BUTTON READS IT TODAY: its one caller was the exploration tab's "Save this subset to disk…",
# removed with that pane on 2026-08-05. Kept as the named default for the next such button, and
# because the rule it records ("never spell it positionally") is what the test beside it pins.
_SAVE_OPERATOR = "mip"

# Roadmap cards shown under "TO BE ADDED", as (label, blurb). Empty: everything currently on the
# roadmap that we're willing to advertise has shipped as a real Operation above. Add an entry when
# a next operator (e.g. the Nautilus agent) is close enough to promise.
_TO_BE_ADDED: list = []


def operator_layer_key(op_key: str, tab_key: Optional[str]) -> str:
    """Layer id an operator's results are filed under.

    Plate-wide runs keep the bare operator key ("mip"). A run scoped to a CONTAINER gets
    "<op>@<tab_key>" — without that, two containers running the same operator both write into
    PlateOverview._op_canvas["mip"] and silently overwrite each other's tiles.

    NO CALLER NAMESPACES TODAY: the exploration tab was the only one and it was removed on
    2026-08-05, so `_viewer.run_operator` passes ``None``. The pair is kept rather than inlined
    because ``operator_name`` below (its inverse) is what every registry lookup goes through, and
    a layer key that MIGHT carry a namespace needs one spelling of the rule, not two."""
    return f"{op_key}@{tab_key}" if tab_key else op_key


def operator_name(layer_key: str) -> str:
    """The REGISTRY name behind a layer key: ``"spot@tab2"`` -> ``"spot"``, ``"mip"`` -> ``"mip"``.

    The exact inverse of :func:`operator_layer_key`, and it lives beside it so the ``@`` rule has
    ONE spelling. Needed because everything downstream of a run holds the LAYER key
    (``PlateWindow._active_op_key``), while anything that wants to read the operator's own
    declaration -- ``operator_produces``, ``operator_consumes``, ``available_region_operators``
    -- has to ask the registry, and the registry has never heard of ``"spot@tab2"``.

    That mismatch was already live: ``_on_result`` builds its accumulator with
    ``region_operator=(op in available_region_operators())`` off the layer key, so a stitch run
    scoped to an exploration tab looked like a per-FOV operator to the accumulator. Splitting here
    fixes both callers with one rule rather than two ``split("@")`` calls that can drift.
    """
    return str(layer_key).split("@", 1)[0]


def result_kind(layer_key: str) -> str:
    """What an operator's pixels MEAN, for a caller holding a LAYER key: ``"intensity"``/``"labels"``.

    :func:`operator_name` then the engine's own ``produces`` declaration. Three cases answer
    ``"intensity"`` without the registry having heard of the key, and each is a real one rather
    than a defensive shrug:

    * a REGION operator (``stitch``, ``coordinate``) — that table has no ``produces`` column
      because every entry in it fuses a mosaic out of the acquisition's own pixels;
    * ``"computed"``, the pseudo-key the reopened-plate path sets, whose pixels are a written
      plate's;
    * ``"raw"``, the preview.

    A key that is none of those and is not registered is still ``"intensity"``, which is what this
    app produced down every path before result kinds existed -- the same "absent means the old
    guarantee" rule the plate contract applies to an unstamped store.
    """
    from squidmip._engine import _OPERATORS

    op = _OPERATORS.get(operator_name(layer_key))
    return op.produces if op is not None else "intensity"


#: Every operator ``run_operator`` can stream live (IMA-226), re-exported from the ENGINE and never
#: derived from ``_OPERATIONS``. The two lists are not the same set and never were:
#:
#: * ``spot`` (and ``decon3d``, and the ``coordinate`` region operator) is a registered operator
#:   with no card, so ``_OPERATIONS_BY_KEY[key].label`` raised a bare ``KeyError`` out of the event
#:   loop the moment anything asked to run it. ``reference`` was the original example and now has a
#:   card; the SHAPE it demonstrated has not gone away, which is why
#:   ``test_every_runnable_operator_is_either_carded_or_declared_cli_only`` pins the other
#:   direction: an engine entry with no card must be listed as deliberately CLI-only.
#: * ``minerva`` is a card that is NOT an operator — it is an export hand-off. Handing its key to
#:   the engine dies with a raw ``KeyError: unknown operator 'minerva'`` in the status line.
#:
#: Both are cured by asking the engine what it can run. This used to be a function BODY here, a
#: second identical one in ``_cli`` and a third inlined in ``_command``, all spelling
#: ``sorted(set(available_projectors()) | set(available_region_operators()))`` because there were
#: two tables to union. There is one table, so there is one answer and one place it is computed.
runnable_operators = _runnable_operators


def operator_label(key: str) -> str:
    """Human label for an operator: its card's if it has one, else the registry name itself.

    A card is presentation, not capability (IMA-226) — an operator with no card must still be
    runnable and must still name itself in the status line and the layer stack."""
    op = _OPERATIONS_BY_KEY.get(key)
    return op.label if op is not None else key


def _action_label(key: str, operator_kwargs: Optional[dict] = None) -> str:
    """What the console calls one action: ``decon(sigma=2.0)``, or bare ``mip`` with no parameters.

    The REGISTRY key, not the card's human label, and the parameters spelled out. A console line
    has to be enough to reproduce the run from, and "Deconvolution" is not: two runs at different
    sigmas would print identically, which is precisely the mixed-recipe plate Task 3 is about.
    Sorted so the same call always renders the same string, i.e. so it can be compared by eye.
    """
    if not operator_kwargs:
        return str(key)
    args = ", ".join(f"{k}={operator_kwargs[k]}" for k in sorted(operator_kwargs))
    return f"{key}({args})"
