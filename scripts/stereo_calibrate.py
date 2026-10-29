"""Calibrate the two overhead webcams with the printed checkerboard.

    uv run python scripts/stereo_calibrate.py                  # cameras 0 and 1, 30 mm squares
    uv run python scripts/stereo_calibrate.py --square 29.6    # measured printed square size
    uv run python scripts/stereo_calibrate.py --from-saved     # recompute from saved photos only

The window shows both cameras live. Each camera's edge turns green and the board corners are
drawn when it sees the whole board. Hold the board still for about a second where both see it
and a pair is taken by itself (the picture flashes). Blue circles mark spots already taken; a
new pair needs the board moved or tilted.

Keys:
    t      take the table photo: board lying flat on the table, seen by camera 0
    c      calibrate from all saved photos and write calib/stereo.npz
    a      auto capture on or off
    space  take a pair now
    q      quit

Take 15 to 25 pairs: board held at different spots over the table, different heights, tilted
different ways, some near the edges of the picture. Then one table photo. Photos are kept in
calib/pairs so you can recompute later without the cameras.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from stereo import (
    BOARD, CALIB_DIR, CALIB_FILE, HEIGHT, SQUARE_MM, WIDTH, Calibration, board_points, find_board,
    open_cameras, wait_for_frames,
)

PAIRS = CALIB_DIR / "pairs"
TABLE_IMG = CALIB_DIR / "table_0.png"


def saved_pairs() -> list[tuple[str, str]]:
    return [(str(p), str(p).replace("_0.png", "_1.png")) for p in sorted(PAIRS.glob("*_0.png"))]


def calibrate(square: float) -> None:
    obj = board_points(square)
    objs, img0s, img1s = [], [], []
    for f0, f1 in saved_pairs():
        c0 = find_board(cv2.imread(f0, cv2.IMREAD_GRAYSCALE))
        c1 = find_board(cv2.imread(f1, cv2.IMREAD_GRAYSCALE))
        if c0 is None or c1 is None:
            print(f"skipping {f0}: board not found in both")
            continue
        objs.append(obj)
        img0s.append(c0)
        img1s.append(c1)
    if len(objs) < 8:
        print(f"only {len(objs)} usable pairs; take at least 15")
        return

    size = (WIDTH, HEIGHT)
    rms0, K0, D0, _, _ = cv2.calibrateCamera(objs, img0s, size, None, None)
    rms1, K1, D1, _, _ = cv2.calibrateCamera(objs, img1s, size, None, None)
    rms, K0, D0, K1, D1, R, T, _, _ = cv2.stereoCalibrate(
        objs, img0s, img1s, K0, D0, K1, D1, size, flags=cv2.CALIB_FIX_INTRINSIC)
    print(f"{len(objs)} pairs")
    print(f"camera 0 error {rms0:.2f} px, camera 1 error {rms1:.2f} px, stereo error {rms:.2f} px")
    print("  (under 0.5 px is good, under 1 px is usable)")
    print(f"cameras are {np.linalg.norm(T):.0f} mm apart")
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"they are turned {angle:.0f} degrees from each other")
    cal = Calibration(K0, D0, K1, D1, R, T)

    if TABLE_IMG.exists():
        corners = find_board(cv2.imread(str(TABLE_IMG), cv2.IMREAD_GRAYSCALE))
        if corners is None:
            print("table photo: board not found, take it again with t")
        else:
            _, rvec, tvec = cv2.solvePnP(obj, corners, K0, D0)
            Rt = cv2.Rodrigues(rvec)[0]
            # Make the table's Z point up, toward the cameras.
            if (-Rt.T @ tvec)[2] < 0:
                Rt = Rt @ np.diag([1.0, -1.0, -1.0])
            cal.table_R, cal.table_t = Rt, tvec.reshape(3)
            cam1_centre = (-R.T @ T).reshape(1, 3)
            h0 = cal.to_table(np.zeros((1, 3)))[0, 2]
            h1 = cal.to_table(cam1_centre)[0, 2]
            print(f"camera 0 is {h0:.0f} mm above the table, camera 1 is {h1:.0f} mm")
    else:
        print("no table photo yet: heights will be from camera 0, not the table. Press t to add it.")

    cal.save()
    print(f"wrote {CALIB_FILE}")


def quick_find(frame: np.ndarray) -> np.ndarray | None:
    """Fast board check for the live view; corners in half-size pixels to match the preview.

    Runs on the full picture: the board's squares are only about 16 px across from the
    cameras' height, too small to find at half size.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(
        gray, BOARD, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
    return corners.reshape(-1, 2) / 2 if ok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, nargs=2, default=[0, 1], help="camera numbers (2 is OBS)")
    ap.add_argument("--square", type=float, default=SQUARE_MM, help="printed square size in mm")
    ap.add_argument("--from-saved", action="store_true", help="recompute from saved photos only")
    ap.add_argument("--exposure", type=float, help="lock exposure (e.g. -6 is 1/64 s); default auto")
    ap.add_argument("--still", type=float, default=1.0, help="seconds the board must be held still")
    ap.add_argument("--target", type=int, default=20, help="pairs to aim for")
    args = ap.parse_args()

    if args.from_saved:
        calibrate(args.square)
        return

    PAIRS.mkdir(parents=True, exist_ok=True)
    cams = open_cameras(args.cams, args.exposure)
    try:
        wait_for_frames(cams)
        n = len(saved_pairs())
        auto = True
        note = "hold the board still where both cameras can see it"
        prev: np.ndarray | None = None  # camera 0 corners last frame
        still_since: float | None = None
        taken: list[list[np.ndarray]] = [[], []]  # board centres of saved pairs, per camera
        last_saved: np.ndarray | None = None
        flash = 0.0
        while True:
            now = time.monotonic()
            frames = [c.latest()[0] for c in cams]
            quick = [quick_find(f) for f in frames]
            both = all(q is not None for q in quick)

            # Auto capture: both see the board, it has stopped moving, and it is in a new spot.
            status = ""
            if both and prev is not None and np.abs(quick[0] - prev).mean() < 1.5:
                still_since = still_since or now
            else:
                still_since = None
            prev = quick[0] if quick[0] is not None else None
            if both and last_saved is not None and np.abs(quick[0] - last_saved).mean() < 25:
                status = "move the board to a new spot or tilt"
                still_since = None
            elif both and still_since is not None:
                left = args.still - (now - still_since)
                status = f"hold still {max(left, 0):.1f}"
                if auto and left <= 0:
                    full = [find_board(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
                    if all(c is not None for c in full):
                        cv2.imwrite(str(PAIRS / f"{n:03d}_0.png"), frames[0])
                        cv2.imwrite(str(PAIRS / f"{n:03d}_1.png"), frames[1])
                        n += 1
                        last_saved = quick[0]
                        for k in range(2):
                            taken[k].append(quick[k].mean(axis=0))
                        note = f"saved pair {n}"
                        print(note)
                        flash = now
                    still_since = None
            elif not both:
                seen = [f"camera {i}" for i, q in zip(args.cams, quick) if q is not None]
                status = f"board seen by {seen[0]} only" if seen else "board not seen"

            tiles = []
            for k, (f, q) in enumerate(zip(frames, quick)):
                tile = cv2.resize(f, (WIDTH // 2, HEIGHT // 2))
                for c in taken[k]:
                    cv2.circle(tile, tuple(int(v) for v in c), 6, (255, 200, 0), 2)
                if q is not None:
                    cv2.drawChessboardCorners(tile, BOARD, q.reshape(-1, 1, 2).astype(np.float32), True)
                cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1),
                              (0, 255, 0) if q is not None else (0, 0, 255), 4)
                tiles.append(tile)
            view = np.hstack(tiles)
            if now - flash < 0.3:
                view = cv2.addWeighted(view, 0.4, np.full_like(view, 255), 0.6, 0)
            bar = np.full((90, view.shape[1], 3), 30, np.uint8)
            cv2.putText(bar, f"{n} of {args.target} pairs   {status}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(bar, f"{note}   keys: t table photo, c calibrate, a auto {'on' if auto else 'off'}, "
                        f"space take now, q quit", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("stereo calibrate", np.vstack([view, bar]))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                auto = not auto
            if key == ord(" "):
                full = [find_board(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
                if all(c is not None for c in full):
                    cv2.imwrite(str(PAIRS / f"{n:03d}_0.png"), frames[0])
                    cv2.imwrite(str(PAIRS / f"{n:03d}_1.png"), frames[1])
                    n += 1
                    note, flash = f"saved pair {n}", now
                else:
                    note = "board not found in both, not saved"
                print(note)
            if key == ord("t"):
                if find_board(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)) is not None:
                    cv2.imwrite(str(TABLE_IMG), frames[0])
                    note, flash = "table photo saved", now
                else:
                    note = f"board not seen by camera {args.cams[0]}, table photo not saved"
                print(note)
            if key == ord("c"):
                calibrate(args.square)
                note = f"calibrated from {n} pairs, see the terminal"
    finally:
        for c in cams:
            c.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
