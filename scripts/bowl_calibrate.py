"""Find the Y axis in the cameras' 3D space from the marker on the bed. Run at every start-up.

    uv run python scripts/bowl_calibrate.py --port COM9
    uv run python scripts/bowl_calibrate.py --port COM9 --exposure -5 --stops 5 60 110 160 215

Needs the camera calibration from stereo_calibrate.py. The table and the printer can move
between runs; the cameras must not (if they did, recalibrate them).

Homes Y, checks the Y switch is open away from home (a stuck switch fakes a home), then moves
the bed slowly to each stop. At each stop both cameras find the marker and its four corners
are triangulated, using the median over several frames. From the marker centres it fits:

  - home: where the marker centre is at Y0
  - the rail direction: which way the marker moves per mm of Y
  - up: the marker's face normal (the bed's up)

and checks itself: the marker must move as far as the G-code said, the stops must lie on one
straight line, and the marker's printed size must come out right. If those are off, a camera
has been bumped. The result goes to calib/bowl.npz, and a picture of each stop to calib/bowl/.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from stereo import CALIB_DIR, CALIB_FILE, HEIGHT, MARKER_DICT, MARKERS_MM, WIDTH, Calibration, open_cameras, wait_for_frames
from y_speed import y_switch_triggered
from y_sweep import Marlin

BOWL_FILE = CALIB_DIR / "bowl.npz"
FEED = 3000  # mm/min (50 mm/s): slow, the bowl is on


def make_detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(MARKER_DICT), params)


def find_markers(det: cv2.aruco.ArucoDetector, frame: np.ndarray) -> dict[int, np.ndarray]:
    """Marker id -> its four corners in pixels (4, 2), for the ids we printed."""
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners) if int(i) in MARKERS_MM}


def measure(cal: Calibration, cams, det, frames: int) -> tuple[dict[int, np.ndarray], list[np.ndarray], dict[int, float]]:
    """Triangulate each marker's corners over several new frame pairs. Returns the median
    corners in camera 0 mm per id, the last frame of each camera, and the median error px."""
    per_id: dict[int, list[np.ndarray]] = {}
    errs: dict[int, list[float]] = {}
    last = [c.latest()[2] for c in cams]
    shots: list[np.ndarray] = []
    got = 0
    deadline = time.monotonic() + 10
    while got < frames and time.monotonic() < deadline:
        now = [c.latest() for c in cams]
        if any(n[2] == l for n, l in zip(now, last)):
            time.sleep(0.005)
            continue
        last = [n[2] for n in now]
        shots = [n[0] for n in now]
        got += 1
        m0, m1 = find_markers(det, shots[0]), find_markers(det, shots[1])
        for mid in m0.keys() & m1.keys():
            X = cal.triangulate(m0[mid], m1[mid])
            per_id.setdefault(mid, []).append(X)
            errs.setdefault(mid, []).append(cal.reprojection_error(X, m0[mid], m1[mid]))
    # Need the marker in at least half the frames to trust it.
    out = {mid: np.median(np.array(xs), axis=0) for mid, xs in per_id.items() if len(xs) >= frames / 2}
    return out, shots, {mid: float(np.median(errs[mid])) for mid in out}


def save_picture(shots: list[np.ndarray], det, name: str) -> None:
    views = []
    for shot in shots:
        v = shot.copy()
        for mid, c in find_markers(det, shot).items():
            cv2.polylines(v, [c.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.putText(v, f"id {mid}", tuple(int(x) for x in c[0] + [0, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
        views.append(cv2.resize(v, (WIDTH // 2, HEIGHT // 2)))
    (CALIB_DIR / "bowl").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(CALIB_DIR / "bowl" / name), np.hstack(views))


def fit_axis(ys: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit centre = home + y * step by least squares. Returns home, step (mm moved per mm of Y,
    camera 0 frame), and how far each stop is from the fitted point (mm)."""
    A = np.column_stack([np.ones_like(ys), ys])
    coef, *_ = np.linalg.lstsq(A, centres, rcond=None)
    home, step = coef[0], coef[1]
    return home, step, np.linalg.norm(centres - (home + np.outer(ys, step)), axis=1)


def face_normal(corners: np.ndarray, toward: np.ndarray) -> np.ndarray:
    """Unit normal of the marker's face, flipped to point toward the cameras."""
    n = np.cross(corners[1] - corners[0], corners[3] - corners[0])
    n /= np.linalg.norm(n)
    return n if np.dot(n, toward - corners.mean(axis=0)) > 0 else -n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--port", required=True, help="the SKR's serial port, e.g. COM9")
    ap.add_argument("--cams", type=int, nargs=2, default=[0, 1], help="camera numbers (2 is OBS)")
    ap.add_argument("--exposure", type=float, help="lock exposure (e.g. -5); default auto")
    ap.add_argument("--stops", type=float, nargs="+", default=[5, 60, 110, 160, 215], help="Y positions, mm")
    ap.add_argument("--frames", type=int, default=15, help="frame pairs per stop")
    ap.add_argument("--settle", type=float, default=0.7, help="seconds to wait after each stop")
    ap.add_argument("--marker", type=float, nargs=2, action="append", metavar=("ID", "MM"),
                    help="measured black square size of a printed marker, if not as designed")
    a = ap.parse_args()

    sizes = dict(MARKERS_MM)
    for mid, mm in a.marker or []:
        sizes[int(mid)] = mm
    cal = Calibration.load()
    if cal.size != (WIDTH, HEIGHT):
        raise SystemExit(f"{CALIB_FILE} was made at {cal.size}, cameras run at {(WIDTH, HEIGHT)}")
    det = make_detector()
    cam1_pos = -cal.R.T @ cal.T.ravel()
    between_cams = cam1_pos / 2  # "toward the cameras" for the face normal

    m = Marlin(a.port)
    cams = []
    found: dict[int, list[tuple[float, np.ndarray, float]]] = {}
    try:
        cams = open_cameras(a.cams, a.exposure)
        wait_for_frames(cams)
        time.sleep(1.0)  # let auto exposure settle
        m.send("G21")
        m.send("G90")
        m.send("M204 P500 T500")
        print("homing Y")
        m.send("G28 Y", timeout=60)
        m.send(f"G0 Y20 F{FEED}")
        m.send("M400", timeout=60)
        if y_switch_triggered(m):
            raise RuntimeError("the Y switch reads pressed at Y20, so the home was not real; check the switch")
        for y in a.stops:
            m.send(f"G0 Y{y:g} F{FEED}")
            m.send("M400", timeout=60)
            time.sleep(a.settle)
            corners, shots, errs = measure(cal, cams, det, a.frames)
            save_picture(shots, det, f"y{y:g}.jpg")
            seen = ", ".join(f"id {mid} ({errs[mid]:.2f} px)" for mid in sorted(corners)) or "no marker"
            print(f"Y{y:g}: {seen}")
            for mid, X in corners.items():
                found.setdefault(mid, []).append((y, X, errs[mid]))
        m.send(f"G0 Y5 F{FEED}")
        m.send("M400", timeout=60)
    except KeyboardInterrupt:
        m.ser.write(b"M410\n")
        print("stopped")
        return 1
    finally:
        for c in cams:
            c.close()
        m.close()

    if not found:
        print("no marker seen at any stop: look at the pictures in calib/bowl/")
        return 1
    # Use the marker seen at the most stops (both, if both are on the bed, are reported).
    results = {}
    for mid, rows in sorted(found.items()):
        ys = np.array([r[0] for r in rows])
        Xs = np.array([r[1] for r in rows])
        centres = Xs.mean(axis=1)
        side = float(np.mean([np.linalg.norm(np.roll(X, -1, axis=0) - X, axis=1).mean() for X in Xs]))
        print(f"\nmarker id {mid}: seen at {len(rows)} of {len(a.stops)} stops")
        print(f"  size: measured {side:.1f} mm, printed {sizes[mid]:g} mm ({100 * (side / sizes[mid] - 1):+.1f}%)")
        if len(rows) < 3:
            print("  fewer than 3 stops: cannot fit and check the axis")
            continue
        home, step, off = fit_axis(ys, centres)
        scale = float(np.linalg.norm(step))
        up = face_normal(Xs.mean(axis=0), between_cams)
        print(f"  travel: {scale:.3f} mm per commanded mm ({100 * (scale - 1):+.1f}%)")
        print(f"  straight line: worst stop {off.max():.1f} mm off, average {off.mean():.1f} mm")
        tilt = float(np.degrees(np.arcsin(abs(np.dot(step / scale, up)))))
        print(f"  rail vs bed face: {tilt:.1f} degrees out of flat")
        if cal.table_R is not None:
            h = cal.to_table(home.reshape(1, 3))[0]
            d = (step / scale) @ cal.table_R
            print(f"  home in table mm: X {h[0]:.0f}  Y {h[1]:.0f}  height {h[2]:.0f}")
            print(f"  rail direction in table axes: X {d[0]:+.3f}  Y {d[1]:+.3f}  Z {d[2]:+.3f}")
        ok = abs(scale - 1) < 0.02 and off.max() < 3 and abs(side / sizes[mid] - 1) < 0.03
        print("  check: " + ("GOOD" if ok else "OFF: a camera may have moved, or the marker is not flat"))
        results[mid] = dict(home=home, step=step, up=up, ys=ys, centres=centres, corners=Xs,
                            side=side, ok=ok, n=len(rows))
    if not results:
        return 1
    best = max(results, key=lambda mid: (results[mid]["n"], -abs(results[mid]["side"] / sizes[mid] - 1)))
    r = results[best]
    np.savez(BOWL_FILE, marker_id=best, home=r["home"], step=r["step"], up=r["up"], ys=r["ys"],
             centres=r["centres"], corners=r["corners"], stamp=time.time())
    print(f"\nsaved marker id {best} to {BOWL_FILE}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
