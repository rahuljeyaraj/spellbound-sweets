"""Write the bowl markers as an A4 PDF to print at actual size.

    uv run python scripts/make_marker.py                  # ids 0 and 1, 5 mm band
    uv run python scripts/make_marker.py --ids 1 --band 2 # one marker, thin band
    uv run python scripts/make_marker.py --lid 150        # rim lid: id 0 centred in a 150 mm circle

Print at 100% or "actual size" (not "fit to page"), then measure each black square edge to edge.
If one is not the size printed under it, note the measured size. Cut along the grey line: the
white band round the black square must stay, the detector needs it. Glue it flat on the bed,
face up.
"""

from __future__ import annotations

import argparse

import cv2

from make_board import PT, write_pdf
from stereo import MARKER_DICT, MARKERS_MM

GAP_MM = 30.0  # space between the markers on the page


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=[0, 1], choices=sorted(MARKERS_MM))
    ap.add_argument("--band", type=float, default=5.0, help="white band round each black square, mm")
    ap.add_argument("--lid", type=float, metavar="DIAMETER",
                    help="instead: marker 0 in the centre of a circle this many mm across, to lay on the rim")
    a = ap.parse_args()
    page_w, page_h = 210.0, 297.0
    if a.lid:
        write_pdf(draw_lid(a.lid, page_w / 2, page_h / 2), page_w, page_h, "lid.pdf")
        return
    total = sum(MARKERS_MM[i] for i in a.ids) + GAP_MM * (len(a.ids) - 1)
    x0 = (page_w - total) / 2
    ops = ["0 g"]
    for marker_id in a.ids:
        size = MARKERS_MM[marker_id]
        ops += draw_marker(marker_id, size, a.band, x0, (page_h - size) / 2)
        x0 += size + GAP_MM
    write_pdf(ops, page_w, page_h, "marker.pdf")


def draw_marker(marker_id: int, size: float, band: float, x0: float, y0: float) -> list[str]:
    """PDF operators for one marker with its lower-left black corner at (x0, y0) mm."""
    cells = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(MARKER_DICT), marker_id, 6)
    cell = size / 6
    ops = ["0 g"]
    for r in range(6):
        for c in range(6):
            if cells[r, c] == 0:  # image rows run down, PDF y runs up
                ops.append(f"{(x0 + c * cell) * PT:.3f} {(y0 + (5 - r) * cell) * PT:.3f} "
                           f"{cell * PT:.3f} {cell * PT:.3f} re f")
    side = size + 2 * band
    ops.append(f"0.6 G 0.3 w {(x0 - band) * PT:.3f} {(y0 - band) * PT:.3f} "
               f"{side * PT:.3f} {side * PT:.3f} re S")
    ops.append(f"BT /F1 9 Tf {(x0 - band) * PT:.3f} {(y0 - band - 7) * PT:.3f} Td "
               f"(id {marker_id}, square {size:g} mm) Tj ET")
    return ops


def draw_lid(diameter: float, cx: float, cy: float) -> list[str]:
    """Marker 0 centred in a grey cut circle, with ticks at the circle's top and bottom."""
    size = MARKERS_MM[0]
    ops = draw_marker(0, size, 0.0, cx - size / 2, cy - size / 2)[:-2]  # no square cut line or label
    r, k = diameter / 2 * PT, diameter / 2 * PT * 0.5523  # four Bezier quarter arcs
    x, y = cx * PT, cy * PT
    ops.append(f"0.6 G 0.3 w {x + r:.3f} {y:.3f} m "
               f"{x + r:.3f} {y + k:.3f} {x + k:.3f} {y + r:.3f} {x:.3f} {y + r:.3f} c "
               f"{x - k:.3f} {y + r:.3f} {x - r:.3f} {y + k:.3f} {x - r:.3f} {y:.3f} c "
               f"{x - r:.3f} {y - k:.3f} {x - k:.3f} {y - r:.3f} {x:.3f} {y - r:.3f} c "
               f"{x + k:.3f} {y - r:.3f} {x + r:.3f} {y - k:.3f} {x + r:.3f} {y:.3f} c S")
    for s in (1, -1):
        ops.append(f"{x:.3f} {y + s * r:.3f} m {x:.3f} {y + s * (r - 8 * PT):.3f} l S")
    ops.append(f"0 g BT /F1 9 Tf {(cx - 45) * PT:.3f} {(cy - diameter / 2 - 8) * PT:.3f} Td "
               f"(lid: marker 0, square {size:g} mm, circle {diameter:g} mm across, cut on the grey line) Tj ET")
    return ops


if __name__ == "__main__":
    main()
