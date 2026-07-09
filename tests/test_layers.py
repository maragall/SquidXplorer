"""OperationStack: ordered, toggleable layer stack (pure model)."""
from squidmip._layers import OperationStack


def test_stack_add_toggle_top_and_reset():
    s = OperationStack()
    assert [l.key for l in s.layers()] == ["raw"]
    s.add("mip", "MIP")
    assert s.top_enabled().key == "mip"          # newest enabled layer shows
    s.toggle("mip", False)
    assert s.top_enabled().key == "raw"          # off -> falls back to base
    s.toggle("mip", True)
    s.add("reference", "Reference")
    assert [l.key for l in s.layers()] == ["raw", "mip", "reference"]
    assert s.top_enabled().key == "reference"
    s.reset()
    assert [l.key for l in s.layers()] == ["raw"]


def test_stack_reorder_and_readd_moves_to_top():
    s = OperationStack()
    s.add("mip", "MIP"); s.add("reference", "Reference")
    s.move("mip", +5)                            # clamp to top
    assert s.layers()[-1].key == "mip"
    s.add("reference", "Reference")              # re-add moves reference back to top
    assert s.layers()[-1].key == "reference"
