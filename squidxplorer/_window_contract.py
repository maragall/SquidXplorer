"""The window interface each functions-over-the-window helper module consumes, DECLARED.

The pattern itself stays (``_ingest``'s and ``_mosaic_playback``'s docstrings defend it: state
lives ON the window, tests read the attributes by name, ``_on_plane`` is driven unbound over a
duck shell). What was implicit is which attributes each helper is allowed to reach: the lists
below are derived from the modules' own ``win.<attr>`` accesses, and
``tests/test_window_contract.py`` walks each module's AST and fails when an access is missing
from its Protocol here, or a declared name stops being accessed. The interface can grow, but
never silently.

Typing only, Qt-free: nothing imports these at runtime and the helpers' signatures are
unchanged (``win`` stays untyped there; a runtime annotation would import Qt types into
Qt-free callers). A name accessed as a call is declared as a method; a name read, written or
passed as a callback is declared as an attribute.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol


class IngestWindow(Protocol):
    """What ``_ingest`` reaches on the PlateWindow (the acquisition-open pipeline)."""

    # -- identity of the opened acquisition (written by ingest) ----------------------------------
    _acq_name: str
    _acq_path: Optional[Any]
    _reader: Any
    _meta: Optional[dict]
    _order: list
    _fov_index: dict
    _plate: Any
    _plate_format: Any
    _plate_format_override: Any

    # -- plate/GUI state the pipeline rebuilds per open ------------------------------------------
    _overview: Any
    _readout: Any
    _op_stack: Any
    _active_op_key: Any
    _cursor: Any
    _current_well: Any
    _current_fov: Any
    _selected_regions: Any
    _time_point_bar: Any
    _viewer_manager: Any
    time_point: int

    # -- the raw-preview lifecycle and the loupe-source books ------------------------------------
    _preview: Any
    _loupe_sources: dict

    # -- window methods the pipeline calls --------------------------------------------------------
    def _refresh_acq_cycle(self, *args: Any, **kwargs: Any) -> Any: ...

    def _apply_layers(self, *args: Any, **kwargs: Any) -> Any: ...
    def _declare_channel_axis(self, *args: Any, **kwargs: Any) -> Any: ...
    # ONE mount point for the rebuilt overview: the hosted plate slot (one window,
    # 2026-08-25) or the plate window's own column.
    def _mount_overview(self, *args: Any, **kwargs: Any) -> Any: ...
    def _set_empty_state(self, *args: Any, **kwargs: Any) -> Any: ...
    def _enable_operators(self, *args: Any, **kwargs: Any) -> Any: ...
    def _on_plate_loaded(self, *args: Any, **kwargs: Any) -> Any: ...
    def _recomposite(self, *args: Any, **kwargs: Any) -> Any: ...
    def _refresh_layers_tab(self, *args: Any, **kwargs: Any) -> Any: ...
    def _refresh_plate_navigation(self, *args: Any, **kwargs: Any) -> Any: ...
    def _release_loupe_sources(self, *args: Any, **kwargs: Any) -> Any: ...
    def _retire(self, *args: Any, **kwargs: Any) -> Any: ...
    def _start_preview(self, *args: Any, **kwargs: Any) -> Any: ...
    def _stop_preview(self, *args: Any, **kwargs: Any) -> Any: ...
    def _stop_worker(self, *args: Any, **kwargs: Any) -> Any: ...
    def _update_loupe_source(self, *args: Any, **kwargs: Any) -> Any: ...

    # -- bound methods passed on, not called here (signal slots and worker callbacks) ------------
    _clear_progress_if_idle: Any
    _on_hover: Any
    _on_marquee_selected: Any
    _on_preview_failed: Any
    _on_preview_tile: Any
    _on_selection_changed: Any
    _on_well_navigated: Any
    _publish_progress: Any
    activate_well: Any


class MosaicPlaybackWindow(Protocol):
    """What ``_mosaic_playback`` reaches on a RegionViewer (load pipeline + frame gate)."""

    # -- the load books (state stays ON the window; tests read _load_gen by name) ----------------
    _load_gen: int
    _worker: Any
    _retired_workers: Optional[list]
    _shown_region: Optional[str]
    _result_region: Optional[str]

    # -- what a load reads ------------------------------------------------------------------------
    _reader: Any
    _meta: Optional[dict]
    _pane: Any
    _cursor: Any
    _roi_bbox: Any
    _fov_slider: Any
    _slider: Any
    open_clock: Any
    time_point: int

    # -- window methods the pipeline calls --------------------------------------------------------
    def _say(self, *args: Any, **kwargs: Any) -> Any: ...

    def _apply_settings_once(self, *args: Any, **kwargs: Any) -> Any: ...
    def _draw_fov_boxes(self, *args: Any, **kwargs: Any) -> Any: ...
    def _drop_result_layers(self, *args: Any, **kwargs: Any) -> Any: ...
    def _on_done(self, *args: Any, **kwargs: Any) -> Any: ...
    def _on_fov_changed(self, *args: Any, **kwargs: Any) -> Any: ...
    def _on_plane(self, *args: Any, **kwargs: Any) -> Any: ...

class RoiToolsWindow(Protocol):
    """What ``_roi_tools`` reaches on a RegionViewer (the drawn-ROI toolset)."""

    _clamping: bool
    _cursor: Any
    _manager: Any
    _meta: Optional[dict]
    _pane: Any
    _regions: Any
    _roi_bbox: Any
    _roi_layer: Any
    window_id: Any

    def _napari_viewer(self, *args: Any, **kwargs: Any) -> Any: ...
    def _say(self, *args: Any, **kwargs: Any) -> Any: ...


class VolumeViewWindow(Protocol):
    """What ``_volume_view`` reaches on a RegionViewer (in-window 3D)."""

    _cursor: Any
    _meta: Optional[dict]
    _native3d: Any
    _pane: Any
    _reader: Any
    _refresh_bricks: Any            # passed as a callback, not called here
    _regions: Any
    _roi_bbox: Any

    def _roi_center_fov(self, *args: Any, **kwargs: Any) -> Any: ...
    def _say(self, *args: Any, **kwargs: Any) -> Any: ...
    def _selected_roi(self, *args: Any, **kwargs: Any) -> Any: ...
    def _view_label(self, *args: Any, **kwargs: Any) -> Any: ...
    def current_region(self, *args: Any, **kwargs: Any) -> Any: ...
    def set_render_mode(self, *args: Any, **kwargs: Any) -> Any: ...


class LutClipboardWindow(Protocol):
    """What ``_lut_clipboard`` reaches on a RegionViewer (the LUT copy/paste/match rules)."""

    _meta: Optional[dict]
    _pane: Any



#: helper module stem -> the Protocol naming every ``win.<attr>`` it may reach.
WINDOW_CONTRACTS: dict[str, type] = {
    "_ingest": IngestWindow,
    "_mosaic_playback": MosaicPlaybackWindow,
    "_roi_tools": RoiToolsWindow,
    "_volume_view": VolumeViewWindow,
    "_lut_clipboard": LutClipboardWindow,
}
