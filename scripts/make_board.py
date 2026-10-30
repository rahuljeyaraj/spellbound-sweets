"""Write the calibration checkerboard as an A4 landscape PDF to print at actual size.

    uv run python scripts/make_board.py          # writes calib/board.pdf

Print at 100% or "actual size" (not "fit to page"), then measure one square. If it is not
30 mm, pass the measured size to stereo_calibrate.py with --square. Glue it to stiff card so it
stays flat.
"""

from __future__ import annotations

from stereo import BOARD, CALIB_DIR, SQUARE_MM

PT = 72 / 25.4  # PDF points per mm


def main() -> None:
    cols, rows = BOARD[0] + 1, BOARD[1] + 1  # squares = inner corners + 1
    page_w, page_h = 297.0, 210.0
    x0 = (page_w - cols * SQUARE_MM) / 2
    y0 = (page_h - rows * SQUARE_MM) / 2 + 5  # PDF y runs upward; leave room for the ruler below
    ops = ["0 g"]
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                ops.append(f"{(x0 + c * SQUARE_MM) * PT:.3f} {(y0 + r * SQUARE_MM) * PT:.3f} "
                           f"{SQUARE_MM * PT:.3f} {SQUARE_MM * PT:.3f} re f")
    ruler_y = (y0 - 8) * PT
    ops.append(f"0.6 w {x0 * PT:.3f} {ruler_y:.3f} m {(x0 + 100) * PT:.3f} {ruler_y:.3f} l S")
    ops.append(f"BT /F1 10 Tf {(x0 + 104) * PT:.3f} {ruler_y - 3:.3f} Td "
               f"(this line is 100 mm, squares are {SQUARE_MM:g} mm) Tj ET")
    write_pdf(ops, page_w, page_h, "board.pdf")


def write_pdf(ops: list[str], page_w: float, page_h: float, name: str) -> None:
    """Write one page of PDF drawing operators (page size in mm) to calib/<name>."""
    stream = "\n".join(ops).encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w * PT:.3f} {page_h * PT:.3f}] "
        f"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".encode(),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(pdf))
        pdf += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(pdf)
    pdf += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    pdf += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    pdf += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()

    CALIB_DIR.mkdir(exist_ok=True)
    out = CALIB_DIR / name
    out.write_bytes(bytes(pdf))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
