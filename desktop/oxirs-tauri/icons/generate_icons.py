#!/usr/bin/env python3
"""Generate the OxiRS Desktop app icons.

`tauri-build` requires `icons/icon.ico` to emit the Windows Resource file, and
`tauri.conf.json` points `bundle.icon` at `icons/icon.png`. Both are produced
here from one description so they cannot drift apart, and so the binaries in
this directory have provenance instead of being unexplained blobs.

The mark is an RDF triple — three nodes, three edges — in the palette the app
already uses (`ui/styles.css`). Each size is rendered independently at 8x and
downsampled, rather than resizing one large render, which keeps the 16px and
24px variants legible.

    python generate_icons.py        # rewrites icon.ico and icon.png

Requires Pillow.
"""

from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw

HERE = pathlib.Path(__file__).parent

# From ui/styles.css.
BG = (26, 26, 46, 255)  # --bg-primary  #1a1a2e
ACCENT = (233, 69, 96, 255)  # --accent      #e94560
TEXT = (224, 224, 224, 255)  # --text        #e0e0e0
# --border (#0f3460) is nearly invisible against --bg-primary, so the edges use a
# lightened form of it that still reads as the same blue.
EDGE = (45, 95, 158, 255)

SUPERSAMPLE = 8

# Unit-square geometry, scaled to whatever size is being rendered.
NODES = [
    ((0.50, 0.25), ACCENT),  # subject
    ((0.23, 0.73), TEXT),  # predicate
    ((0.77, 0.73), TEXT),  # object
]
EDGES = [(0, 1), (0, 2), (1, 2)]
NODE_RADIUS = 0.135
EDGE_WIDTH = 0.075
CORNER_RADIUS = 0.22

# 16-256 is what Windows Explorer, the taskbar, and Alt-Tab actually ask for.
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]
PNG_SIZE = 512


def render(size: int) -> Image.Image:
    """Render the icon at `size` px, supersampled then reduced."""
    s = size * SUPERSAMPLE
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        [(0, 0), (s - 1, s - 1)], radius=int(CORNER_RADIUS * s), fill=BG
    )

    def px(point: tuple[float, float]) -> tuple[float, float]:
        return (point[0] * s, point[1] * s)

    # Edges first so the nodes sit on top of the joins.
    for a, b in EDGES:
        draw.line(
            [px(NODES[a][0]), px(NODES[b][0])],
            fill=EDGE,
            width=max(1, int(EDGE_WIDTH * s)),
        )

    r = NODE_RADIUS * s
    for (cx, cy), color in NODES:
        cx, cy = cx * s, cy * s
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    # Pillow writes a genuine multi-image ICO when handed `sizes`; the largest
    # render is passed in so no upscaling happens.
    largest = render(max(ICO_SIZES))
    largest.save(
        HERE / "icon.ico",
        format="ICO",
        sizes=[(n, n) for n in ICO_SIZES],
    )
    render(PNG_SIZE).save(HERE / "icon.png", format="PNG")
    print(f"wrote icon.ico ({len(ICO_SIZES)} sizes) and icon.png ({PNG_SIZE}px)")


if __name__ == "__main__":
    main()
