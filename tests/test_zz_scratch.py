import pytest
from tests.test_view_deck import _tabbed_plate
from tests.conftest import shutdown_plate_window

@pytest.mark.parametrize("step", ["insert", "insert_kwargs", "insert_toggle"])
def test_bisect(qapp, napari_pane_stub, squid_dataset, step):
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=1)
    v = views[0]
    try:
        v.operator_panel()
        combo = v._op_combo
        combo.setCurrentIndex(next(k for k in range(combo.count())
                                   if combo.itemData(k) == "stitch"))
        v._show_operator_controls()
        qapp.processEvents()
        panel = v._inserted_panel
        if step in ("insert_kwargs", "insert_toggle"):
            panel.blend_spin.setValue(2)
            assert win.operator_kwargs_for("stitch")["blend_px"] == 2
        if step == "insert_toggle":
            v._show_operator_controls()
            assert v._inserted_panel is None
    finally:
        shutdown_plate_window(qapp, win)

def test_after(qapp, napari_pane_stub, squid_dataset):
    root, _ = squid_dataset
    win, mgr, deck, views = _tabbed_plate(qapp, root, n_views=2)
    shutdown_plate_window(qapp, win)
