"""The grouped layer tree: PROCESSING LAYER -> CHANNELS, over napari's flat LayerList.

THE ARITHMETIC THAT FORCES THIS
-------------------------------
We key one napari layer per (processing layer, channel). Five operators -- raw, stitched,
deconvolved, background-subtracted, flatfield-corrected -- times four channels (405/488/561/638)
is 24 rows in a single flat list. A scientist cannot work in that.

Three facts bound the solution and are not up for re-litigation:

* **napari 0.6.6 has no layer groups.** There is not one ``LayerGroup`` symbol in the package.
  Upstream napari#2229 has been open since Feb 2021 and is still open; #5950 and #6345 sit
  behind it. Plan as if groups never arrive.
* **One layer per channel is idiomatic napari, not our mistake.** ``add_image(channel_axis=)``
  provably SPLITS into N layers (``split_channels``, ``viewer_model.py:1162``).
* **Nobody ships a layer cap.** A prior-art survey found no precedent for destroying layers on
  switch or for an arbitrary ceiling, and an earlier proposal here to cap at 8 was contradicted
  by that evidence. Not implemented, deliberately.

So the fix is the one BOTH shipped precedents chose: replace the layer-list UI, not the layers.

* brainglobe/**napari-experimental** builds its own grouped tree beside napari's list. Its own
  ethos: "the main layer list should only be used to add/remove layers".
* 4DNucleome/**PartSeg** is our exact architecture -- a host Qt app embedding napari -- and
  rebuilds its control surface out of napari's own widget classes.

WHAT IS PORTED, AND WHAT IS DELIBERATELY NOT
--------------------------------------------
Ported from napari-experimental: a ``QTreeView`` over a real ``QAbstractItemModel`` (they use
napari's ``QtNodeTreeView``/``QtNodeTreeModel``), ``CheckStateRole`` at both levels, a group
toggle that cascades to every leaf and then emits ``dataChanged`` for each child so the child
checkboxes repaint.

NOT ported, on purpose:

* **``GroupLayer._visible``.** They store the group's visibility as its own bool, and the
  consequence is written into their own code: nothing syncs it upward, so a group checkbox
  drifts out of step with the layers it claims to describe. This project's dominant defect
  shape is exactly that -- two representations of one truth, hand-synced (4+ confirmed, most
  recently the contrast sync silently killed by layer recreation). Here the group's check state
  is DERIVED on every read: all visible -> Checked, none -> Unchecked, otherwise
  PartiallyChecked. There is no group state, so there is nothing to drift.
* **Drag-and-drop reordering and the LayerList order sync.** Their tree owns ordering, which is
  most of the file (``flat_index_order``, ``_move_plan``, index reversal between tree and
  LayerList). We are ADDING a tree next to napari's real controls, not replacing them, so
  napari's own list keeps ordering and this stays a view. Less code that can disagree.
* **Deleting napari's docks (PartSeg).** Not done. dc0f288 embeds the real napari window on
  purpose, because hand-rebuilt controls were rejected as "not napari". Nothing here conflicts
  with napari's own list -- both write ``layer.visible`` -- so there is no reason to remove it.

IDENTITY IS (op, channel), NEVER A LAYER OBJECT AND NEVER A PARSED NAME
-----------------------------------------------------------------------
``_load_mosaic`` (``_viewer.py:5092``) destroys and recreates every layer on each region
change. A subscription bound to layer objects therefore dies silently on the next region --
that is precisely how the contrast sync was lost (``on_user_contrast``, ``_viewer.py:5174``).
Every row here resolves its layer through ``MosaicLayers.find(op, channel)`` at read time, and
identity comes from ``layer.metadata`` via ``key_of``. Names are labels; ian-stitcher's
``extractWavelength(layer.name)`` and petakit's unparseable channel names are why.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from PyQt5.QtCore import QAbstractItemModel, QEvent, QModelIndex, QSize, Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QFrame, QTreeView

from squidmip._napari_view import MosaicLayers, key_of

#: internalId marking a top-level (processing-layer) row. Qt hands the id back on parent()
#: lookups, so a child stores its OP ROW there and a top-level row stores this sentinel.
_TOP = 0xFFFFFFFF


#: napari's own delegate roles, resolved ONCE and by name. They live in
#: ``napari._qt.containers`` -- private, like ``QtLayerControlsContainer`` already is, and for the
#: same reason: the alternative is reimplementing napari's layer row and reintroducing exactly the
#: duplication this project keeps deleting. Resolved defensively so a napari upgrade that moves
#: them costs the PRETTY rendering and not the pane.
def _resolve_napari_roles() -> dict:
    try:
        from napari._qt.containers._base_item_model import ItemRole
        from napari._qt.containers.qt_layer_model import LoadedRole, ThumbnailRole
    except Exception:                       # noqa: BLE001 - cosmetic; the tree still works
        return {}
    return {"item": ItemRole, "thumbnail": ThumbnailRole, "loaded": LoadedRole}


_NAPARI_ROLES: dict = _resolve_napari_roles()


class _GroupItem:
    """What a PROCESSING-LAYER row reports itself to be.

    napari's delegate asks the item ``is_group()`` -- a branch it already carries for layer trees
    -- and paints a folder (open when expanded) instead of an image icon. So a group row gets
    napari's own folder treatment without us drawing anything.
    """

    def is_group(self) -> bool:
        return True


_GROUP_ITEM = _GroupItem()


def _empty_thumbnail() -> Any:
    """A fully transparent tile, for rows that have no pixels of their own."""
    img = QImage(32, 32, QImage.Format_RGBA8888)
    img.fill(0)
    return img


_EMPTY_THUMBNAIL = _empty_thumbnail()


def _thumbnail_image(layer) -> Optional[Any]:
    """The layer's own thumbnail as a QImage, or None.

    napari keeps ``layer.thumbnail`` as an RGBA array and repaints it as the data changes, so
    this is the SAME picture napari's own layer list shows -- read, never generated here.
    """
    if layer is None:
        return None
    thumb = getattr(layer, "thumbnail", None)
    if thumb is None:
        return None
    try:
        return QImage(thumb, thumb.shape[1], thumb.shape[0], QImage.Format_RGBA8888)
    except Exception:                       # noqa: BLE001 - an odd thumbnail shape is not fatal
        return None


class MosaicTreeModel(QAbstractItemModel):
    """Two-level model over ``MosaicLayers``. Owns structure, never owns visibility.

    ``_rows`` caches only the SHAPE of the hierarchy -- ``[(op, [channel, ...]), ...]`` -- because
    Qt requires ``rowCount``/``index`` to be stable between ``beginResetModel`` and
    ``endResetModel``. It is rebuilt wholesale from ``MosaicLayers`` whenever the LayerList
    changes; it is never edited in place, so it cannot drift into a second opinion about which
    layers exist. Visibility is not cached at all: every ``CheckStateRole`` read goes to the
    live layer.
    """

    def __init__(self, mosaic: MosaicLayers, parent=None) -> None:
        super().__init__(parent)
        self._mosaic = mosaic
        self._rows: list[tuple[str, list[str]]] = []
        self._watched: list[Any] = []
        self.refresh()

        # Layers appear and disappear underneath us: _load_mosaic wipes and rebuilds them on
        # every region change, and the user can delete one from napari's own list.
        layers = mosaic.model.layers
        layers.events.inserted.connect(self._on_layers_changed)
        layers.events.removed.connect(self._on_layers_changed)

    # -- structure ----------------------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild the hierarchy from the layers that exist RIGHT NOW."""
        self.beginResetModel()
        self._rows = [(op, list(self._mosaic.channels(op))) for op in self._mosaic.ops()]
        self._rewatch()
        self.endResetModel()

    def _on_layers_changed(self, event=None) -> None:
        self.refresh()

    def _rewatch(self) -> None:
        """Re-subscribe to ``layer.events.visible`` for the layers that exist now.

        Reading the truth is not enough: without this the checkbox is correct only until
        somebody toggles the layer from napari's own list, and then it is quietly stale. The
        subscriptions are rebuilt from scratch on every refresh precisely BECAUSE the layer
        objects are thrown away and remade -- binding once at construction is the mistake that
        killed the contrast sync.
        """
        for layer in self._watched:
            try:
                layer.events.visible.disconnect(self._on_layer_visible)
            except Exception:                    # noqa: BLE001 - layer already destroyed
                pass
        self._watched = []
        for layer in self._mosaic.ours():
            layer.events.visible.connect(self._on_layer_visible)
            self._watched.append(layer)

    def _on_layer_visible(self, event=None) -> None:
        layer = getattr(event, "source", None)
        key = key_of(layer) if layer is not None else None
        if key is None:
            return
        for op_row, (op, channels) in enumerate(self._rows):
            if op != key.op or key.channel not in channels:
                continue
            parent = self.index(op_row, 0)
            child = self.index(channels.index(key.channel), 0, parent)
            self.dataChanged.emit(child, child, [Qt.CheckStateRole])
            # The group's own check state is derived from this layer, so it changed too.
            self.dataChanged.emit(parent, parent, [Qt.CheckStateRole])

    # -- QAbstractItemModel -------------------------------------------------------------
    def index(self, row: int, column: int, parent=QModelIndex()) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        if not parent.isValid():
            return self.createIndex(row, column, _TOP)
        return self.createIndex(row, column, parent.row())

    def parent(self, index=QModelIndex()) -> QModelIndex:  # type: ignore[override]
        if not index.isValid() or index.internalId() == _TOP:
            return QModelIndex()
        return self.createIndex(int(index.internalId()), 0, _TOP)

    def rowCount(self, parent=QModelIndex()) -> int:
        if not parent.isValid():
            return len(self._rows)
        if parent.internalId() != _TOP or parent.column() != 0:
            return 0                             # channels are leaves
        if parent.row() >= len(self._rows):
            return 0
        return len(self._rows[parent.row()][1])

    def columnCount(self, parent=QModelIndex()) -> int:
        return 1

    def flags(self, index=QModelIndex()):
        if not index.isValid():
            return Qt.NoItemFlags
        # ItemIsUserCheckable is load-bearing: a model that answers CheckStateRole without it
        # renders a tree with no checkboxes -- readable, unclickable, and green under any test
        # that only calls setData directly.
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable

    def _key_at(self, index: QModelIndex) -> Optional[tuple[str, str]]:
        """``(op, channel)`` for a leaf, ``None`` for a processing-layer row."""
        if index.internalId() == _TOP:
            return None
        op_row = int(index.internalId())
        if op_row >= len(self._rows):
            return None
        op, channels = self._rows[op_row]
        if index.row() >= len(channels):
            return None
        return op, channels[index.row()]

    def data(self, index=QModelIndex(), role=Qt.DisplayRole):
        if not index.isValid():
            return None
        key = self._key_at(index)

        if role in (Qt.DisplayRole, Qt.ToolTipRole):
            if key is None:
                if index.row() >= len(self._rows):
                    return None
                return self._rows[index.row()][0]
            return key[1]

        if role == Qt.CheckStateRole:
            if key is None:
                if index.row() >= len(self._rows):
                    return None
                return self._group_state(self._rows[index.row()][0])
            layer = self._mosaic.find(*key)
            if layer is None:
                return None
            return Qt.Checked if layer.visible else Qt.Unchecked

        # --- the roles napari's own LayerDelegate paints from -------------------------------
        #
        # Julio: "I still don't like these napari layer UX. The original napari layer widgets
        # were way more beautiful." They were: napari does not draw a checkbox and a name, it
        # draws an EYE, a type icon and the layer's THUMBNAIL, through `LayerDelegate`. Serving
        # the delegate's roles here means we use napari's actual renderer instead of imitating
        # it -- the same choice as embedding napari's window rather than rebuilding its controls.
        if role == _NAPARI_ROLES.get("item"):
            # The delegate asks the item what it IS: a group gets a folder icon (it checks
            # `is_group()`, which exists for exactly this case), a channel gets the image icon
            # for its layer type.
            if key is None:
                return _GROUP_ITEM
            return self._mosaic.find(*key)

        if role == _NAPARI_ROLES.get("thumbnail"):
            # NEVER None. The delegate does QPixmap.fromImage(index.data(ThumbnailRole)) with no
            # guard, so a missing thumbnail is a TypeError on every repaint. A group has no
            # pixels of its own, so it gets a TRANSPARENT tile: nothing is drawn, and nothing is
            # invented either (borrowing a channel's thumbnail would label the group with one
            # arbitrary channel's picture).
            if key is None:
                return _EMPTY_THUMBNAIL
            layer = self._mosaic.find(*key)
            return _thumbnail_image(layer) or _EMPTY_THUMBNAIL

        if role == _NAPARI_ROLES.get("loaded"):
            # Always loaded. The alternative starts napari's loading GIF, which animates forever
            # unless something later says otherwise -- a spinner that outlives its cause is the
            # same lie as an indicator that cannot turn off.
            return True

        if role == Qt.SizeHintRole:
            return QSize(200, 34)                # napari's own row height; the thumbnail needs it

        return None

    # -- the contract napari's LayerDelegate expects of the model behind a view ------------
    #
    # `_paint_thumbnail` calls `index.model().sourceModel().all_loaded()`, because in napari the
    # view always sits behind a QSortFilterProxyModel. Ours does not, so painting raised
    # AttributeError on EVERY repaint -- 54 tracebacks in one launch, while the headless tests
    # stayed green because they read roles and never actually painted a row.
    #
    # Implementing the two methods is the smaller lie than inserting a proxy we do not otherwise
    # need. They are here, next to the delegate roles they serve, and named as what they are.

    def sourceModel(self):
        """This model IS the source; there is no proxy in front of it."""
        return self

    def all_loaded(self) -> bool:
        """Every row is loaded. See ``LoadedRole`` in ``data`` for why that is unconditional."""
        return True

    def _group_state(self, op: str):
        """DERIVED, never stored. See the module docstring on ``GroupLayer._visible``."""
        group = self._mosaic.group(op)
        if not group:
            return Qt.Unchecked
        visible = [bool(ly.visible) for ly in group]
        if all(visible):
            return Qt.Checked
        if not any(visible):
            return Qt.Unchecked
        return Qt.PartiallyChecked

    def setData(self, index=QModelIndex(), value=None, role=Qt.EditRole) -> bool:
        if not index.isValid() or role != Qt.CheckStateRole:
            return False
        want = Qt.CheckState(value) == Qt.Checked
        key = self._key_at(index)

        if key is not None:
            layer = self._mosaic.find(*key)
            if layer is None:
                return False
            layer.visible = want
            self.dataChanged.emit(index, index, [role])
            # The parent's state is derived from this leaf; repaint it too.
            parent = self.parent(index)
            if parent.isValid():
                self.dataChanged.emit(parent, parent, [role])
            return True

        if index.row() >= len(self._rows):
            return False
        op = self._rows[index.row()][0]
        for layer in self._mosaic.group(op):
            layer.visible = want
        self.dataChanged.emit(index, index, [role])
        # Toggling a group changes every child; emit for each so their checkboxes repaint.
        # (Straight out of napari-experimental's QtGroupLayerModel.setData.)
        for child_row in range(self.rowCount(index)):
            child = self.index(child_row, 0, index)
            self.dataChanged.emit(child, child, [role])
        return True


def _install_napari_delegate(view) -> bool:
    """Paint the rows with napari's OWN ``LayerDelegate``, not with Qt's default.

    Julio: "I still don't like these napari layer UX. The original napari layer widgets were way
    more beautiful."

    He is right, and the gap was never styling. A default Qt item view draws a native checkbox
    and a string. napari draws an EYE (its stylesheet paints the check indicator as an eye icon),
    a per-type icon, and the layer's live THUMBNAIL -- all in `LayerDelegate.paint`. Imitating
    that would be a second renderer to keep in step with napari's, which is the duplication this
    project keeps deleting; so the model serves the delegate's roles instead and napari paints.

    Returns whether it took, so the caller/tests can tell "napari painted this" from "we fell
    back". Failure is cosmetic and never fatal: an unstyled tree is ugly, an exception while
    building pane 2 costs the viewer.
    """
    if not _NAPARI_ROLES:
        return False
    try:
        from napari._qt.containers._layer_delegate import LayerDelegate

        view.setItemDelegate(LayerDelegate())
        return True
    except Exception:                       # noqa: BLE001 - cosmetic; keep Qt's default delegate
        return False


def _napari_stylesheet() -> str:
    """napari's OWN stylesheet, with its list rules extended to cover a tree.

    Julio: "I like the layer nesting, but the widgets look ugly, napari's original ones were way
    nicer." They were, and the reason is that this tree had no stylesheet at all: it rendered in
    the default Qt style inside a napari-themed dock, so it looked like a widget from another
    application -- which it was.

    The fix is NOT to hand-pick colours to match. That would be a second theme, drifting from
    napari's the moment the user switches theme. `napari.qt.get_current_stylesheet` is public and
    returns the real thing; the only gap is that napari styles ``QListView`` (its layer list) and
    never ``QTreeView``, so every list rule is DUPLICATED onto the tree selector. The values stay
    napari's -- nothing here invents a colour.

    Falls back to an empty stylesheet if napari changes the API: an unstyled tree is ugly, a
    crash on building the pane costs the whole viewer.
    """
    try:
        from napari.qt import get_current_stylesheet
        sheet = get_current_stylesheet()
    except Exception:                       # noqa: BLE001 - cosmetic; never fatal to the pane
        return ""
    extra = re.sub(r"QListView", "QTreeView",
                   "\n".join(m.group(0) for m in re.finditer(
                       r"[^{}]*QListView[^{}]*\{[^{}]*\}", sheet)))
    return sheet + "\n" + extra


class MosaicTree(QTreeView):
    """The grouped layer view: processing layers, each expanding into its channels.

    Lives ALONGSIDE napari's own layer list rather than replacing it (that is the divergence
    from napari-experimental's ethos, and it is deliberate -- dc0f288 embeds the real napari
    window because hand-rebuilt controls were rejected as "not napari"). Both surfaces write the
    same ``layer.visible``, so they cannot disagree: toggle one and the other repaints.
    """

    def __init__(self, mosaic: MosaicLayers, parent=None) -> None:
        super().__init__(parent)
        self.setModel(MosaicTreeModel(mosaic, self))
        self.setHeaderHidden(True)
        self.setUniformRowHeights(True)
        self.setExpandsOnDoubleClick(True)
        self.expandAll()
        self.model().modelReset.connect(self.expandAll)
        self.setIndentation(14)
        self.setRootIsDecorated(True)
        self.setAlternatingRowColors(False)
        self.setFrameShape(QFrame.NoFrame)      # napari's docks carry the frame, not the widget
        self._restyling = False                 # see changeEvent: setStyleSheet re-enters
        self.setStyleSheet(_napari_stylesheet())
        _install_napari_delegate(self)

    def changeEvent(self, e):
        """Follow a napari THEME switch. The stylesheet is a snapshot of the theme at build time;
        napari repaints its own widgets on a palette change and ours would otherwise stay the old
        theme's colours, which is the "two answers to one question" shape again.

        The reentrancy guard is load-bearing: ``setStyleSheet`` itself posts a StyleChange, so
        restyling from inside changeEvent without it recurses until the process dies -- which it
        did, immediately, the first time this was written.
        """
        super().changeEvent(e)
        if e.type() != QEvent.PaletteChange or self._restyling:
            return
        self._restyling = True
        try:
            self.setStyleSheet(_napari_stylesheet())
        finally:
            self._restyling = False
