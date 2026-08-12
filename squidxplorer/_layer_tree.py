"""The grouped layer tree: processing layer -> channels, over napari's flat LayerList.

Identity is (op, channel), resolved through ``MosaicLayers.layers_for`` at read time,
never a layer object and never a parsed name. Check state is derived, never stored.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from qtpy.QtCore import QAbstractItemModel, QEvent, QModelIndex, QSize, Qt
from qtpy.QtGui import QImage
from qtpy.QtWidgets import QFrame, QTreeView

from squidxplorer._napari_view import MosaicLayers, key_of

#: internalId marking a top-level (processing-layer) row; a child stores its op row there.
_TOP = 0xFFFFFFFF


#: napari's own delegate roles, resolved once and defensively: a napari upgrade that moves
#: them costs the pretty rendering and not the pane.
def _resolve_napari_roles() -> dict:
    try:
        from napari._qt.containers._base_item_model import ItemRole
        from napari._qt.containers.qt_layer_model import LoadedRole, ThumbnailRole
    except Exception:                       # noqa: BLE001 - cosmetic; the tree still works
        return {}
    return {"item": ItemRole, "thumbnail": ThumbnailRole, "loaded": LoadedRole}


_NAPARI_ROLES: dict = _resolve_napari_roles()


def _check_state(layers) -> Any:
    """A row's check state derived from the layers it stands for. Never stored."""
    visible = [bool(getattr(ly, "visible", False)) for ly in layers]
    if all(visible):
        return Qt.Checked
    if not any(visible):
        return Qt.Unchecked
    return Qt.PartiallyChecked


class _GroupItem:
    """What a processing-layer row reports itself to be: napari's delegate paints a folder."""

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
    """The layer's own thumbnail as a QImage, or None."""
    if layer is None:
        return None
    thumb = getattr(layer, "thumbnail", None)
    if thumb is None:
        return None
    try:
        # .copy() is load-bearing: QImage wraps the buffer without owning it, and napari
        # replaces `layer.thumbnail` with a new array, freeing the one Qt still paints through.
        img = QImage(thumb, thumb.shape[1], thumb.shape[0], QImage.Format_RGBA8888)
        return img.copy()                   # deep copy: the returned QImage owns its own pixels
    except Exception:                       # noqa: BLE001 - an odd thumbnail shape is not fatal
        return None


class MosaicTreeModel(QAbstractItemModel):
    """Two-level model over ``MosaicLayers``. Owns structure, never owns visibility."""

    def __init__(self, mosaic: MosaicLayers, parent=None) -> None:
        super().__init__(parent)
        self._mosaic = mosaic
        self._rows: list[tuple[str, list[str]]] = []
        self._watched: list[Any] = []
        self.refresh()

        # Layers appear and disappear underneath us: _load_mosaic rebuilds them on region change.
        layers = mosaic.model.layers
        layers.events.inserted.connect(self._on_layers_changed)
        layers.events.removed.connect(self._on_layers_changed)

    # -- structure ----------------------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild the hierarchy from the layers that exist right now."""
        self.beginResetModel()
        # Topmost first: napari renders its LayerList reversed, and there is ONE order — napari's.
        ops = list(reversed(self._mosaic.ops()))
        self._rows = [(op, list(reversed(self._mosaic.channels(op)))) for op in ops]
        self._rewatch()
        self.endResetModel()

    def _on_layers_changed(self, event=None) -> None:
        self.refresh()

    def _rewatch(self) -> None:
        """Re-subscribe to ``layer.events.visible``; layer objects are thrown away and remade."""
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
        # ItemIsUserCheckable is load-bearing: without it the tree renders no checkboxes.
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
            # Every layer of the pair: a channel is one mosaic layer or N bricks of a volume.
            layers = self._mosaic.layers_for(*key)
            if not layers:
                return None
            return _check_state(layers)

        # --- the roles napari's own LayerDelegate paints from -------------------------------
        if role == _NAPARI_ROLES.get("item"):
            if key is None:
                return _GROUP_ITEM
            return self._mosaic.find(*key)

        if role == _NAPARI_ROLES.get("thumbnail"):
            # Never None: the delegate does QPixmap.fromImage on this with no guard.
            if key is None:
                return _EMPTY_THUMBNAIL
            layer = self._mosaic.find(*key)
            return _thumbnail_image(layer) or _EMPTY_THUMBNAIL

        if role == _NAPARI_ROLES.get("loaded"):
            # Always loaded; the alternative starts napari's loading GIF forever.
            return True

        if role == Qt.SizeHintRole:
            return QSize(200, 34)                # napari's own row height; the thumbnail needs it

        return None

    # -- the contract napari's LayerDelegate expects of the model behind a view ------------
    # `_paint_thumbnail` calls `index.model().sourceModel().all_loaded()`; napari's view always
    # sits behind a QSortFilterProxyModel, ours does not, so both methods answer directly.

    def sourceModel(self):
        """This model IS the source; there is no proxy in front of it."""
        return self

    def all_loaded(self) -> bool:
        """Every row is loaded."""
        return True

    def _group_state(self, op: str):
        """Derived, never stored."""
        group = self._mosaic.group(op)
        if not group:
            return Qt.Unchecked
        return _check_state(group)

    def setData(self, index=QModelIndex(), value=None, role=Qt.EditRole) -> bool:
        if not index.isValid() or role != Qt.CheckStateRole:
            return False
        want = Qt.CheckState(value) == Qt.Checked
        key = self._key_at(index)

        if key is not None:
            # All of them: a checkbox that drives one brick of a hundred does not control.
            layers = self._mosaic.layers_for(*key)
            if not layers:
                return False
            for layer in layers:
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
        for child_row in range(self.rowCount(index)):
            child = self.index(child_row, 0, index)
            self.dataChanged.emit(child, child, [role])
        return True


def _install_napari_delegate(view) -> bool:
    """Paint the rows with napari's own ``LayerDelegate``; failure is cosmetic, never fatal."""
    if not _NAPARI_ROLES:
        return False
    try:
        from napari._qt.containers._layer_delegate import LayerDelegate

        view.setItemDelegate(LayerDelegate())
        return True
    except Exception:                       # noqa: BLE001 - cosmetic; keep Qt's default delegate
        return False


#: The napari selectors whose rules must also reach a ``QTreeView``. ``QtLayerList`` carries
#: the eye indicator and the 28 px item margin the thumbnail is painted clear of.
_TREE_SOURCE_SELECTORS = ("QListView", "QtLayerList")


def _napari_stylesheet(sheet: Optional[str] = None) -> str:
    """napari's own stylesheet, with its list rules duplicated onto the QTreeView selector.

    *sheet* is injectable for tests. Falls back to an empty stylesheet if napari changes the API.
    """
    if sheet is None:
        try:
            from napari.qt import get_current_stylesheet
            sheet = get_current_stylesheet()
        except Exception:                   # noqa: BLE001 - cosmetic; never fatal to the pane
            return ""
    extra = []
    for selector in _TREE_SOURCE_SELECTORS:
        blocks = "\n".join(m.group(0) for m in re.finditer(
            r"[^{}]*" + selector + r"[^{}]*\{[^{}]*\}", sheet))
        extra.append(re.sub(selector, "QTreeView", blocks))
    return sheet + "\n" + "\n".join(extra)


class MosaicTree(QTreeView):
    """The grouped layer view: processing layers, each expanding into its channels.

    Lives alongside napari's own layer list; both write the same ``layer.visible``.
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
        self.selectionModel().currentChanged.connect(self._select_in_napari)

    def _select_in_napari(self, current, _previous=None) -> None:
        """Selecting a row here selects the layer(s) in napari, so its controls follow."""
        model = self.model()
        if model is None or not current.isValid():
            return
        key = model._key_at(current)
        layers = (model._mosaic.layers_for(*key) if key is not None
                  else model._mosaic.group(model._rows[current.row()][0])
                  if current.row() < len(model._rows) else [])
        layers = [ly for ly in layers if ly is not None]
        if not layers:
            return
        try:
            selection = model._mosaic.model.layers.selection
            if len(layers) == 1:
                # `active` is the right seam for ONE layer: select_only() plus current.
                selection.active = layers[0]
            else:
                # ...and the wrong one for a group: that setter collapses a multi-select to one.
                selection.clear()
                selection.update(layers)
        except Exception:                      # noqa: BLE001 - selection is a convenience
            pass

    def changeEvent(self, e):
        """Follow a napari theme switch; the reentrancy guard stops setStyleSheet recursing."""
        super().changeEvent(e)
        if e.type() != QEvent.PaletteChange or self._restyling:
            return
        self._restyling = True
        try:
            self.setStyleSheet(_napari_stylesheet())
        finally:
            self._restyling = False
