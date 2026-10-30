"""Write the bowl marker as an A4 PDF to print at actual size.

    uv run python scripts/make_marker.py         # writes calib/marker.pdf

Print at 100% or "actual size" (not "fit to page"), then measure the black square edge to edge.
If it is not 50 mm, note the measured size. Cut along the grey line: the white band round the
black square must stay, the detector needs it. Glue it flat on the bed, face up.
"""

from __future__ import annotations

import cv2

from make_board import PT, write_pdf
from stereo import MARKER_DICT, MARKER_ID, MARKER_MM

MARGIN_MM = 5.0  # white band kept round the black square; the cut piece is 60 mm


def main() -> None:
    cells = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(MARKER_DICT), MARKER_ID, 6)
    cell = MARKER_MM / 6
    page_w, page_h = 210.0, 297.0
    x0 = (page_w - MARKER_MM) / 2
    y0 = (page_h - MARKER_MM) / 2
    ops = ["0 g"]
    for r in range(6):
        for c in range(6):
            if cells[r, c] == 0:  # image rows run down, PDF y runs up
                ops.append(f"{(x0 + c * cell) * PT:.3f} {(y0 + (5 - r) * cell) * PT:.3f} "
                           f"{cell * PT:.3f} {cell * PT:.3f} re f")
    side = MARKER_MM + 2 * MARGIN_MM
    ops.append(f"0.6 G 0.3 w {(x0 - MARGIN_MM) * PT:.3f} {(y0 - MARGIN_MM) * PT:.3f} "
               f"{side * PT:.3f} {side * PT:.3f} re S")
    ops.append(f"BT /F1 10 Tf {(x0 - MARGIN_MM) * PT:.3f} {(y0 - MARGIN_MM - 8) * PT:.3f} Td "
               f"(marker {MARKER_ID}, black square {MARKER_MM:g} mm, cut on the grey line) Tj ET")
    write_pdf(ops, page_w, page_h, "marker.pdf")


if __name__ == "__main__":
    main()
