#!/usr/bin/env python3
# Author: Paulo R. A. Gilz
"""Draw the application icon and write packaging/aoeviewer.ico.

    python tools/make_icon.py

Generated rather than committed as a binary. An .ico in a repository is a blob
nobody can review, nobody can adjust, and which quietly stops matching the
application it belongs to; this is a hundred lines that produce it on demand,
at every size Windows asks for, from a description you can read.

No Pillow. Everything here is NumPy plus the standard library, because the
machines this has to run on cannot always reach a package index, and an icon is
not worth a dependency. Shapes are drawn as signed distance fields and rendered
4x supersampled, which gives clean antialiased edges at 16 px -- where a naive
rasteriser turns a diagonal line into a staircase.

The design: a dark rounded square, a cyan flight profile climbing, levelling
and descending, an amber marker line crossing it. It reads as "plotted flight
data" at 256 px and still reads as "a chart" at 16 px, which is the only real
test of an icon.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from typing import List, Sequence, Tuple

import numpy as np

SUPERSAMPLE = 4
SIZES = (16, 24, 32, 48, 64, 128, 256)

BACKGROUND = (0x14, 0x1A, 0x24)
BORDER = (0x33, 0x44, 0x5C)
CURVE = (0x3F, 0xA9, 0xF5)
CURVE2 = (0x3D, 0xDC, 0x84)
MARKER = (0xFF, 0xB3, 0x00)
AXIS = (0x4A, 0x5B, 0x72)


def _grid(size: int):
    """Pixel-centre coordinates in a 0..1 box."""
    axis = (np.arange(size) + 0.5) / size
    return np.meshgrid(axis, axis, indexing="xy")


def _rounded_rect(size: int, margin: float, radius: float) -> np.ndarray:
    """Coverage of a rounded rectangle, 0..1."""
    x, y = _grid(size)
    half = 0.5 - margin
    dx = np.abs(x - 0.5) - (half - radius)
    dy = np.abs(y - 0.5) - (half - radius)
    outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    inside = np.minimum(np.maximum(dx, dy), 0.0)
    distance = outside + inside - radius
    # One supersampled pixel of feather; the downsample does the rest.
    edge = 1.0 / size
    return np.clip(0.5 - distance / edge, 0.0, 1.0)


def _polyline(size: int, points: Sequence[Tuple[float, float]],
              width: float) -> np.ndarray:
    """Coverage of a stroked polyline, 0..1.

    Distance to each segment, minimum over segments. Round joins fall out of
    this for free, which is what keeps the corners of the profile from
    developing spikes at small sizes.
    """
    x, y = _grid(size)
    distance = np.full((size, size), 1e9, dtype=np.float64)

    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dx, dy = x1 - x0, y1 - y0
        length_sq = dx * dx + dy * dy
        if length_sq <= 0:
            continue
        t = np.clip(((x - x0) * dx + (y - y0) * dy) / length_sq, 0.0, 1.0)
        distance = np.minimum(distance, np.hypot(x - (x0 + t * dx),
                                                 y - (y0 + t * dy)))

    edge = 1.0 / size
    return np.clip(0.5 - (distance - width * 0.5) / edge, 0.0, 1.0)


def _over(base: np.ndarray, color: Sequence[int], coverage: np.ndarray) -> None:
    """Composite a flat colour over the RGB buffer, in place."""
    alpha = coverage[..., None]
    base[..., :3] = base[..., :3] * (1.0 - alpha) + np.array(color, float) * alpha


def render(size: int) -> np.ndarray:
    """One RGBA image, uint8, size x size."""
    big = size * SUPERSAMPLE
    image = np.zeros((big, big, 4), dtype=np.float64)

    plate = _rounded_rect(big, 0.02, 0.16)
    _over(image, BACKGROUND, plate)
    # Coverage is 0..1; the channel is 0..255. Getting this wrong produces an
    # icon that is entirely, invisibly transparent, which looks like a blank
    # square rather than like a bug.
    image[..., 3] = plate * 255.0

    # A hairline border, drawn as the difference of two plates: it separates
    # the icon from a dark taskbar without needing a second shape.
    inner = _rounded_rect(big, 0.055, 0.13)
    _over(image, BORDER, np.clip(plate - inner, 0.0, 1.0) * 0.9)

    # Axes, faint. Present at 128 px and above, invisible below, which is the
    # correct amount of detail for each.
    if size >= 64:
        _over(image, AXIS, _polyline(big, [(0.17, 0.80), (0.86, 0.80)], 0.012))
        _over(image, AXIS, _polyline(big, [(0.17, 0.16), (0.17, 0.80)], 0.012))

    # The second trace first, so the primary one draws over it.
    _over(image, CURVE2, _polyline(big, [
        (0.19, 0.68), (0.34, 0.60), (0.50, 0.63), (0.66, 0.52), (0.84, 0.55),
    ], 0.055 if size >= 32 else 0.075))

    # Climb, level-off, cruise, descent -- the profile the generator flies.
    _over(image, CURVE, _polyline(big, [
        (0.19, 0.72), (0.33, 0.36), (0.44, 0.26), (0.62, 0.26), (0.84, 0.44),
    ], 0.075 if size >= 32 else 0.10))

    # The marker: a vertical line, which is the thing this viewer is for.
    _over(image, MARKER, _polyline(big, [(0.62, 0.14), (0.62, 0.84)], 0.030))

    # Box-filter down to the requested size. Averaging the supersampled buffer
    # is what produces the antialiasing; there is no filtering anywhere else.
    reduced = image.reshape(size, SUPERSAMPLE, size, SUPERSAMPLE, 4).mean(axis=(1, 3))
    return np.clip(np.rint(reduced), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ ICO


def _bmp_entry(rgba: np.ndarray) -> bytes:
    """One icon image as a 32-bit BMP DIB, the format every Windows reads.

    PNG-compressed entries are legal since Vista and would be smaller, but a
    handful of tools -- including some resource editors and installers -- still
    only understand the DIB form, and the whole file is under 400 KB either
    way. Correct everywhere beats small.
    """
    height, width, _ = rgba.shape

    # A DIB inside an .ico declares double the real height: the second half is
    # the AND mask, even for 32-bit images that do not need one.
    header = struct.pack(
        "<IiiHHIIiiII",
        40,                 # header size
        width, height * 2,
        1, 32,              # planes, bit count
        0,                  # BI_RGB
        width * height * 4,
        0, 0, 0, 0,
    )

    # Bottom-up rows, BGRA order.
    flipped = rgba[::-1]
    bgra = flipped[..., [2, 1, 0, 3]]
    pixels = bgra.tobytes()

    # The AND mask: one bit per pixel, rows padded to 4 bytes. Zero throughout,
    # because the alpha channel already carries the transparency.
    mask_row_bytes = ((width + 31) // 32) * 4
    mask = b"\x00" * (mask_row_bytes * height)

    return header + pixels + mask


def write_ico(path: str, images: List[np.ndarray]) -> None:
    entries = [_bmp_entry(image) for image in images]

    header = struct.pack("<HHH", 0, 1, len(entries))     # reserved, type, count
    directory = b""
    offset = 6 + 16 * len(entries)

    for image, payload in zip(images, entries):
        height, width, _ = image.shape
        directory += struct.pack(
            "<BBBBHHII",
            width if width < 256 else 0,     # 0 means 256
            height if height < 256 else 0,
            0, 0,                            # palette count, reserved
            1, 32,                           # planes, bit count
            len(payload), offset,
        )
        offset += len(payload)

    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(directory)
        for payload in entries:
            handle.write(payload)


def write_png(path: str, rgba: np.ndarray) -> None:
    """A plain PNG, for the Linux desktop entry and for eyeballing the design."""
    import zlib

    height, width, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height,
                                                8, 6, 0, 0, 0)))
        handle.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        handle.write(chunk(b"IEND", b""))


def main(argv=None) -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ico", default=os.path.join(here, "packaging",
                                                      "aoeviewer.ico"))
    parser.add_argument("--png", default=os.path.join(here, "packaging",
                                                      "aoeviewer.png"))
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(args.ico), exist_ok=True)
    images = [render(size) for size in SIZES]
    write_ico(args.ico, images)
    write_png(args.png, images[-1])

    print("wrote {0} ({1:.0f} KB, sizes {2})".format(
        args.ico, os.path.getsize(args.ico) / 1024,
        ", ".join(str(s) for s in SIZES)))
    print("wrote {0}".format(args.png))
    return 0


if __name__ == "__main__":
    sys.exit(main())
