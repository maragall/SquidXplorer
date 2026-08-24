"""OperationStack: ordered, toggleable layer stack (pure model)."""
from squidxplorer._operations import OperationStack


def test_stack_add_toggle_top_and_reset():
    s = OperationStack()
    assert [l.key for l in s.layers()] == ["raw"]
    s.add("mip", "MIP")
    assert s.top_enabled().key == "mip"
    s.toggle("mip", False)
    assert s.top_enabled().key == "raw"
    s.toggle("mip", True)
    s.add("demo", "Demo")
    assert [l.key for l in s.layers()] == ["raw", "mip", "demo"]
    assert s.top_enabled().key == "demo"
    s.reset()
    assert [l.key for l in s.layers()] == ["raw"]


def test_stack_reorder_and_readd_moves_to_top():
    s = OperationStack()
    s.add("mip", "MIP"); s.add("demo", "Demo")
    s.move("mip", +5)
    assert s.layers()[-1].key == "mip"
    s.add("demo", "Demo")
    assert s.layers()[-1].key == "demo"
