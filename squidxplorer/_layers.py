"""OperationStack: the ordered, toggleable layer stack behind the plate view.

The topmost ENABLED layer is what the plate renders. Pure data structure (no Qt).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Layer:
    key: str          # stable id ("raw", "mip", "reference", ...)
    label: str
    enabled: bool = True


class OperationStack:
    def __init__(self) -> None:
        self._layers: list[Layer] = [Layer("raw", "raw", True)]

    def add(self, key: str, label: str) -> None:
        """Add (or re-add) an operation layer on top, enabled. Re-adding moves it to the top."""
        self._layers = [ly for ly in self._layers if ly.key != key]
        self._layers.append(Layer(key, label, True))

    def remove(self, key: str) -> bool:
        """Drop an operation layer; the base ('raw') is never removable."""
        if key == "raw":
            return False
        before = len(self._layers)
        self._layers = [ly for ly in self._layers if ly.key != key]
        return len(self._layers) != before

    def remove_suffix(self, suffix: str) -> list[str]:
        """Drop every layer whose key ends with ``suffix``; returns the removed keys."""
        gone = [ly.key for ly in self._layers if ly.key != "raw" and ly.key.endswith(suffix)]
        if gone:
            self._layers = [ly for ly in self._layers if ly.key not in gone]
        return gone

    def toggle(self, key: str, enabled: bool) -> bool:
        """Enable/disable a layer; the base ('raw') can never be disabled."""
        if key == "raw":
            return True
        for ly in self._layers:
            if ly.key == key:
                ly.enabled = enabled
                return ly.enabled
        return False

    def move(self, key: str, delta: int) -> None:
        """Reorder a layer by +/- steps; the base ('raw') never moves off the bottom."""
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
