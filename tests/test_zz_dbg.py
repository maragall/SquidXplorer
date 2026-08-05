import sys
import pytest

def test_dbg(qapp, squid_dataset):
    import squidmip._viewer as V
    root, _ = squid_dataset
    win = V.PlateWindow(None)
    win.ingest(str(root))
    ov = win._overview
    ov.resize(700, 560); ov.show(); qapp.processEvents()
    print("BOXES", ov._boxes, file=sys.stderr)
    print("meta fovs", win._meta["fovs_per_region"], file=sys.stderr)
    r, c = [rc for rc, w in ov._by_rc.items() if w == "B2"][0]
    ov._user_view = True
    ov._cd = ov._fit_cd() * 8
    print("cd", ov._cd, "fit", ov._fit_cd(), file=sys.stderr)
    print("cell_rect", ov._cell_rect(r, c), "cell_source", ov._cell_source(r, c), file=sys.stderr)
    for f in (0, 1):
        print(f, ov._boxes.get(("B2", f)), ov._block_rect(r, c, *ov._boxes[("B2", f)]), file=sys.stderr)
    win.close()
