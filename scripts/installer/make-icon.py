"""Generate the app icon: scripts/installer/squidxplorer.ico AND squidxplorer.png.

    python scripts/installer/make-icon.py

Both artifacts are committed, so this only needs re-running to CHANGE the art. It is kept
alongside the binaries so the icon stays editable instead of being an opaque blob nobody can
regenerate — and so the .ico (the Windows .exe and desktop shortcut) and the .png (the AppImage
and the XDG menu entry) can never drift apart: one drawing, two encodings.

The motif is a wellplate — the plate is the app's root object — with one well lit to stand for
the selection the whole UI is built around, and a chamfered A1 corner for orientation. Drawn once
at 4x and downsampled per size with LANCZOS: at 16px a literal 96-well grid is indistinguishable
from noise, which is why the grid here is 4x3 rather than true to a real plate.

Needs Pillow, which the dev venv already has (napari pulls it in).
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

SIZES = [16, 24, 32, 48, 64, 128, 256]
S = 1024
SCALE = S / 256.0

BG = (18, 42, 58, 255)          # deep slate-teal tile
PLATE = (232, 238, 240, 255)    # off-white plate body
WELL = (86, 122, 140, 255)      # an empty well
WELL_LIT = (255, 176, 59, 255)  # the selected well
EDGE = (140, 158, 168, 255)


def px(v):
    return int(round(v * SCALE))


def draw() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    d.rounded_rectangle([px(8), px(8), px(248), px(248)], radius=px(46), fill=BG)

    d.rounded_rectangle(
        [px(38), px(58), px(218), px(198)], radius=px(16), fill=PLATE, outline=EDGE, width=px(2)
    )
    # chamfer the top-left corner: the A1 orientation mark on a real plate
    d.polygon([(px(38), px(84)), (px(64), px(58)), (px(38), px(58))], fill=BG)

    cols, rows = 4, 3
    x0, y0, x1, y1 = px(58), px(80), px(200), px(180)
    cw, ch = (x1 - x0) / cols, (y1 - y0) / rows
    r = min(cw, ch) * 0.30
    for row in range(rows):
        for col in range(cols):
            cx, cy = x0 + cw * (col + 0.5), y0 + ch * (row + 0.5)
            lit = row == 0 and col == 0
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=WELL_LIT if lit else WELL)

    return img


def main() -> int:
    here = Path(__file__).resolve().parent
    ico = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "squidxplorer.ico"
    png = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "squidxplorer.png"
    img = draw()
    frames = [img.resize((n, n), Image.LANCZOS) for n in SIZES]
    frames[-1].save(
        ico, format="ICO", sizes=[(n, n) for n in SIZES], append_images=frames[:-1]
    )
    frames[-1].save(png, format="PNG")
    print(f"wrote {ico} ({', '.join(str(n) for n in SIZES)}) and {png} (256)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
