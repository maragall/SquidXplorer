"""OperationStack: the ordered, toggleable layer stack behind the plate view (DESIGN.md).

- Layer 0 is the base ("raw" preview). Each applied operation adds a layer on top.
- Enable, disable, and reorder any layer. The topmost ENABLED layer is what the plate renders.
- v1 usually holds base plus one operation; the structure supports more.
- Pure data structure (no Qt), so it is unit-testable on its own. The Layers tab drives it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Layer:
    key: str          # stable id ("raw", "mip", "reference", ...)
    label: str        # shown in the Layers tab
    enabled: bool = True


class OperationStack:
    def __init__(self) -> None:
        self._layers: list[Layer] = [Layer("raw", "raw", True)]   # base layer, always present

    def add(self, key: str, label: str) -> None:
        """Add (or re-add) an operation layer on top, enabled. Re-adding moves it to the top."""
        self._layers = [ly for ly in self._layers if ly.key != key]
        self._layers.append(Layer(key, label, True))

    def remove(self, key: str) -> bool:
        """Drop an operation layer. Returns True if it was there. The base ('raw') is never
        removable — closing an exploration tab drops that tab's layers, and the plate must
        always keep a base to fall back to."""
        if key == "raw":
            return False
        before = len(self._layers)
        self._layers = [ly for ly in self._layers if ly.key != key]
        return len(self._layers) != before

    def remove_suffix(self, suffix: str) -> list[str]:
        """Drop every layer whose key ends with ``suffix`` (an exploration tab owns the layers
        keyed ``<op>@<tab_key>``, one per operator it ran). Returns the removed keys."""
        gone = [ly.key for ly in self._layers if ly.key != "raw" and ly.key.endswith(suffix)]
        if gone:
            self._layers = [ly for ly in self._layers if ly.key not in gone]
        return gone

    def toggle(self, key: str, enabled: bool) -> bool:
        """Enable/disable a layer. Returns the resulting state.

        The base ('raw') can never be disabled, for the same reason ``remove`` will not drop it:
        it is the layer everything else is recoverable TO. With raw disableable, unticking every
        box left ``top_enabled()`` returning None — and the window's ``_apply_layers`` silently
        does nothing on None, so the plate went on painting the last operator while every checkbox
        in the tab read OFF. A view that disagrees with its own controls is worse than no toggle.
        """
        if key == "raw":
            return True
        for ly in self._layers:
            if ly.key == key:
                ly.enabled = enabled
                return ly.enabled
        return False

    def move(self, key: str, delta: int) -> None:
        """Reorder a layer by +/- steps. The base ('raw') never moves off the bottom.

        That sentence was the docstring long before it was true: ``move('raw', +1)`` reordered the
        base like any other layer, and ``move('mip', -1)`` pushed it off index 0 from the other
        side. Either one puts raw ABOVE an operator, and since the plate renders the topmost
        ENABLED layer, every operator underneath becomes permanently invisible while its checkbox
        still reads ON — the layer stack lying about what is on screen. Raw is the floor: it does
        not move, and nothing moves below it.
        """
        if key == "raw":
            return
        idx = next((i for i, ly in enumerate(self._layers) if ly.key == key), None)
        if idx is None:
            return
        floor = 1 if self._layers and self._layers[0].key == "raw" else 0
        new = max(floor, min(len(self._layers) - 1, idx + delta))
        if new != idx:
            self._layers.insert(new, self._layers.pop(idx))

    def top_enabled(self) -> Layer | None:
        """The topmost enabled layer (what the plate renders), or None if all are off."""
        for ly in reversed(self._layers):
            if ly.enabled:
                return ly
        return None

    def layers(self) -> list[Layer]:
        return list(self._layers)

    def reset(self) -> None:
        self._layers = [Layer("raw", "raw", True)]
