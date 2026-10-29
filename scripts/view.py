"""Show both webcams side by side, to aim them before calibrating.

    uv run python scripts/view.py              # cameras 0 and 1
    uv run python scripts/view.py --cams 1 0   # swapped

Keys: s saves a snapshot of each camera to calib/views, q quits. Cross lines mark the middle
of each picture. If the second camera will not open alongside the first, the first is shown
alone with a note: move one camera to a separate USB port.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from stereo import CALIB_DIR, HEIGHT, WIDTH, Camera, wait_for_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, nargs=2, default=[0, 1], help="camera numbers (2 is OBS)")
    ap.add_argument("--exposure", type=float, help="lock exposure (e.g. -6 is 1/64 s); default auto")
    args = ap.parse_args()

    cams: list[Camera] = []
    problem = ""
    for i in args.cams:
        try:
            cams.append(Camera(i, args.exposure))
        except RuntimeError as e:
            problem = str(e)
            print(problem)
    if not cams:
        return
    try:
        wait_for_frames(cams)
        counts = [c.latest()[2] for c in cams]
        fps, t0 = [0.0] * len(cams), time.monotonic()
        while True:
            now = time.monotonic()
            if now - t0 >= 1.0:
                new = [c.latest()[2] for c in cams]
                fps = [(b - a) / (now - t0) for a, b in zip(counts, new)]
                counts, t0 = new, now
            tiles = []
            for c, f in zip(cams, fps):
                frame = c.latest()[0]
                tile = cv2.resize(frame, (WIDTH // 2, HEIGHT // 2))
                h, w = tile.shape[:2]
                cv2.line(tile, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
                cv2.line(tile, (0, h // 2), (w, h // 2), (255, 255, 0), 1)
                cv2.putText(tile, f"camera {c.index}  {f:.0f} fps  brightness {frame.mean():.0f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                tiles.append(tile)
            view = np.hstack(tiles)
            if problem:
                cv2.putText(view, "second camera did not open: use a separate USB port", (10, HEIGHT // 2 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("cameras", view)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                out = CALIB_DIR / "views"
                out.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%H%M%S")
                for c in cams:
                    cv2.imwrite(str(out / f"{stamp}_cam{c.index}.jpg"), c.latest()[0])
                print(f"saved to {out}")
    finally:
        for c in cams:
            c.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
