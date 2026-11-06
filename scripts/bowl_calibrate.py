"""Find the Y axis in the cameras' 3D space from the marker on the bed. Run at every start-up.

    uv run python scripts/bowl_calibrate.py --port COM9
    uv run python scripts/bowl_calibrate.py --port COM9 --exposure -5 --stops 5 60 110 160 215
    uv run python scripts/bowl_calibrate.py --port COM9 --rim 72.5 --wait-key   # rail and rim, a key press first

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
has been bumped. The result goes to calib/bowl.npz.

With --rim (the lid card lying on the bowl), it then finds the rim as rim_calibrate.py does, in
the same window: rail and rim in one go.

A live window shows both cameras the whole time, with each marker outlined, and under them its
3D position, the progress and the results. --wait-key holds everything until a key is pressed.
Press q to stop (mid-run it quick-stops the bed).
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time

import cv2
import numpy as np

from stereo import (CALIB_DIR, CALIB_FILE, HEIGHT, LID_ID, MARKER_DICT, MARKERS_MM, WIDTH, Calibration, open_cameras,
                    wait_for_frames)
from y_speed import y_switch_triggered
from y_sweep import Marlin

BOWL_FILE = CALIB_DIR / "bowl.npz"
FEED = 3000  # mm/min (50 mm/s): slow, the bowl is on


def make_detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(MARKER_DICT), params)


def find_markers(det: cv2.aruco.ArucoDetector, frame: np.ndarray) -> dict[int, np.ndarray]:
    """Marker id -> its four corners in pixels (4, 2), for the ids we printed. An id seen twice is
    left out: the two cameras could each keep a different copy and pair the wrong markers."""
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return {}
    ids = [int(i) for i in ids.ravel()]
    return {i: c.reshape(4, 2) for i, c in zip(ids, corners) if i in MARKERS_MM and ids.count(i) == 1}


def match_markers(cal: Calibration, m0: dict[int, np.ndarray], m1: dict[int, np.ndarray]
                  ) -> dict[int, tuple[np.ndarray, float]]:
    """Markers seen by both cameras: id -> (4 corners in table mm, error px)."""
    out = {}
    for mid in sorted(m0.keys() & m1.keys()):
        X = cal.triangulate(m0[mid], m1[mid])
        out[mid] = (cal.to_table(X), cal.reprojection_error(X, m0[mid], m1[mid]))
    return out


def draw_markers(img: np.ndarray, markers: dict[int, np.ndarray], both: dict[int, tuple[np.ndarray, float]]) -> None:
    """Outline each marker: magenta when both cameras see it (with its height), red for one only."""
    for mid, c in markers.items():
        colour = (255, 0, 255) if mid in both else (0, 0, 255)
        cv2.polylines(img, [c.astype(np.int32)], True, colour, 3)
        label = f"id {mid}" + (f"  h {both[mid][0][:, 2].mean():.0f}" if mid in both else "")
        cv2.putText(img, label, tuple(int(v) for v in c[0] + [0, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)


def marker_lines(m0: dict[int, np.ndarray], m1: dict[int, np.ndarray],
                 both: dict[int, tuple[np.ndarray, float]]) -> list[tuple[str, tuple[int, int, int]]]:
    """One line of text per marker: its 3D position, or which camera alone sees it."""
    lines = []
    for mid, (X, err) in both.items():
        c = X.mean(axis=0)
        side = np.linalg.norm(np.roll(X, -1, axis=0) - X, axis=1).mean()
        lines.append((f"marker {mid}: X {c[0]:.0f}  Y {c[1]:.0f}  height {c[2]:.0f} mm"
                      f"   size {side:.1f} mm   error {err:.1f} px", (255, 0, 255)))
    for mid in sorted((m0.keys() | m1.keys()) - both.keys()):
        lines.append((f"marker {mid}: seen by camera {0 if mid in m0 else 1} only", (0, 0, 255)))
    return lines


LINE_H = 22  # px per text line in the panel under the cameras
LOG_LINES = 18  # log lines the panel keeps: the whole rail and rim result fits


class Viewer:
    """Live window of both cameras with the markers outlined; under them a panel with the status,
    each marker's 3D position and the log, so a screen recording shows it all. wait_key holds
    everything until a key is pressed (time to start the recording)."""

    def __init__(self, cal: Calibration, cams, det, title: str = "bowl calibrate") -> None:
        self.cal, self.cams, self.det, self.title = cal, cams, det, title
        self.status = ""
        self.log: list[str] = []
        self.last = 0.0
        self.waiting = False  # wait_key: any key but q starts
        self.started = False

    def show(self, shots: list[np.ndarray] | None = None, found: list[dict[int, np.ndarray]] | None = None) -> None:
        if shots is None:
            if time.monotonic() - self.last < 0.05:
                return
            shots = [c.latest()[0] for c in self.cams]
        if found is None:
            found = [find_markers(self.det, s) for s in shots]
        self.last = time.monotonic()
        both = match_markers(self.cal, found[0], found[1])
        views = [s.copy() for s in shots]
        for v, f in zip(views, found):
            draw_markers(v, f, both)
        view = np.hstack([cv2.resize(v, (WIDTH // 2, HEIGHT // 2)) for v in views])

        lines = [(self.status, (0, 255, 255))] + marker_lines(found[0], found[1], both)
        lines += [(text, (235, 235, 235)) for text in self.log[-LOG_LINES:]]
        panel = np.full((LINE_H * (4 + LOG_LINES) + 12, view.shape[1], 3), 28, np.uint8)
        for k, (text, colour) in enumerate(lines):
            cv2.putText(panel, text, (12, 24 + LINE_H * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        cv2.imshow(self.title, np.vstack([view, panel]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise KeyboardInterrupt
        if self.waiting and key != 0xFF:
            self.started = True

    def say(self, text: str) -> None:
        print(text, flush=True)
        indent = " " * (len(text) - len(text.lstrip()) + 4)
        self.log += textwrap.wrap(text, 150, subsequent_indent=indent) or [""]  # fits the panel

    def wait_key(self, what: str) -> None:
        """Keep the window live until a key is pressed in it (q still quits)."""
        status, self.status = self.status, f"press any key to {what} (q quits)"
        self.waiting, self.started = True, False
        while not self.started:
            self.show()
            time.sleep(0.01)
        self.waiting, self.status = False, status


def measure(cal: Calibration, cams, det, frames: int, viewer: Viewer
            ) -> tuple[dict[int, np.ndarray], list[np.ndarray], dict[int, float]]:
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
        viewer.show(shots, [m0, m1])
        for mid in m0.keys() & m1.keys():
            X = cal.triangulate(m0[mid], m1[mid])
            per_id.setdefault(mid, []).append(X)
            errs.setdefault(mid, []).append(cal.reprojection_error(X, m0[mid], m1[mid]))
    # Need the marker in at least half the frames to trust it.
    out = {mid: np.median(np.array(xs), axis=0) for mid, xs in per_id.items() if len(xs) >= frames / 2}
    return out, shots, {mid: float(np.median(errs[mid])) for mid in out}


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
    ap.add_argument("--hold", type=float, default=0, help="close the window this many s after the end (0: on q)")
    ap.add_argument("--rim", type=float, metavar="RADIUS",
                    help="then find the rim too, as rim_calibrate.py does: lid card on the bowl, rim radius in mm")
    ap.add_argument("--wait-key", action="store_true",
                    help="wait for a key press in the window before anything moves: time to start a screen recording")
    a = ap.parse_args()

    sizes = dict(MARKERS_MM)
    for mid, mm in a.marker or []:
        sizes[int(mid)] = mm
    cal = Calibration.load()
    if cal.size != (WIDTH, HEIGHT):
        raise SystemExit(f"{CALIB_FILE} was made at {cal.size}, cameras run at {(WIDTH, HEIGHT)}")
    det = make_detector()

    m = Marlin(a.port)
    cams = []
    found: dict[int, list[tuple[float, np.ndarray, float]]] = {}
    ok = False
    try:
        cams = open_cameras(a.cams, a.exposure)
        wait_for_frames(cams)
        v = Viewer(cal, cams, det, "calibrate the rail and the rim" if a.rim else "bowl calibrate")
        v.status = "starting cameras"
        hold(v, 1.0)  # let auto exposure settle
        if a.wait_key:
            v.wait_key("calibrate the rail" + (" and the rim" if a.rim else ""))
        m.send("G21")
        m.send("G90")
        m.send("M204 P500 T500")
        v.status = "homing Y"
        m.send("G28 Y", timeout=60, idle=v.show)
        m.send(f"G0 Y20 F{FEED}")
        m.send("M400", timeout=60, idle=v.show)
        if y_switch_triggered(m):
            raise RuntimeError("the Y switch reads pressed at Y20, so the home was not real; check the switch")
        for y in a.stops:
            v.status = f"moving to Y{y:g}"
            m.send(f"G0 Y{y:g} F{FEED}")
            m.send("M400", timeout=60, idle=v.show)
            hold(v, a.settle)
            v.status = f"measuring at Y{y:g}"
            corners, shots, errs = measure(cal, cams, det, a.frames, v)
            seen = ", ".join(f"id {mid} ({errs[mid]:.2f} px)" for mid in sorted(corners)) or "no marker"
            v.say(f"Y{y:g}: {seen}")
            for mid, X in corners.items():
                found.setdefault(mid, []).append((y, X, errs[mid]))
        v.status = "back to Y5"
        m.send(f"G0 Y5 F{FEED}")
        m.send("M400", timeout=60, idle=v.show)

        ok = report(v, cal, found, sizes, len(a.stops))
        if a.rim is not None:
            if ok:
                from rim_calibrate import measure_rim  # here: rim_calibrate imports this file
                ok = measure_rim(v, cal, cams, det, a.rim)
            else:
                v.say("rail not GOOD: rim not measured")
        v.status = "done: press q to close"
        hold(v, a.hold or float("inf"))  # keep the window live until q (or --hold seconds)
    except KeyboardInterrupt:
        m.ser.write(b"M410\n")
        print("closed")
    finally:
        for c in cams:
            c.close()
        m.close()
        cv2.destroyAllWindows()
    return 0 if found and ok else 1


def hold(v: Viewer, seconds: float) -> None:
    """Wait while keeping the window live."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        v.show()
        time.sleep(0.01)


def fit_markers(cal: Calibration, found: dict[int, list[tuple[float, np.ndarray, float]]],
                sizes: dict[int, float], stops: int) -> tuple[dict[int, dict], list[str]]:
    """Fit and check the axis for each marker seen. Returns per marker id its fit (home, step,
    up, ok, ...) and the lines saying how it went. Shared with dodge.py, which calibrates by
    itself when it finds the printer has moved."""
    lines: list[str] = []
    if not found:
        return {}, ["no marker seen at any stop"]
    between_cams = -cal.R.T @ cal.T.ravel() / 2  # "toward the cameras" for the face normal
    results = {}
    for mid, rows in sorted(found.items()):
        ys = np.array([r[0] for r in rows])
        Xs = np.array([r[1] for r in rows])
        centres = Xs.mean(axis=1)
        side = float(np.mean([np.linalg.norm(np.roll(X, -1, axis=0) - X, axis=1).mean() for X in Xs]))
        lines.append(f"marker {mid}: seen at {len(rows)} of {stops} stops, size {side:.1f} mm "
                     f"(printed {sizes[mid]:g}, {100 * (side / sizes[mid] - 1):+.1f}%)")
        if len(rows) < 3:
            lines.append("  fewer than 3 stops: cannot fit and check the axis")
            continue
        home, step, off = fit_axis(ys, centres)
        scale = float(np.linalg.norm(step))
        up = face_normal(Xs.mean(axis=0), between_cams)
        tilt = float(np.degrees(np.arcsin(abs(np.dot(step / scale, up)))))
        lines.append(f"  travel {100 * (scale - 1):+.1f}%, off the line {off.max():.1f} mm worst, "
                     f"rail vs bed face {tilt:.1f} deg")
        if cal.table_R is not None:
            h = cal.to_table(home.reshape(1, 3))[0]
            d = (step / scale) @ cal.table_R
            lines.append(f"  Y0 at table X {h[0]:.0f}  Y {h[1]:.0f}  height {h[2]:.0f} mm; "
                         f"rail along X {d[0]:+.3f}  Y {d[1]:+.3f}  Z {d[2]:+.3f}")
        ok = abs(scale - 1) < 0.02 and off.max() < 3 and abs(side / sizes[mid] - 1) < 0.03
        lines.append("  check: " + ("GOOD" if ok else "OFF: a camera may have moved, the marker is not flat, "
                                    "or something held the bed back (a spider web round the motor once did)"))
        results[mid] = dict(home=home, step=step, up=up, ys=ys, centres=centres, corners=Xs,
                            side=side, ok=ok, n=len(rows))
    return results, lines


def best_marker(results: dict[int, dict], sizes: dict[int, float]) -> int:
    """The marker seen at the most stops, then the one whose size came out truest."""
    return max(results, key=lambda mid: (results[mid]["n"], -abs(results[mid]["side"] / sizes[mid] - 1)))


def save_bowl(mid: int, r: dict) -> None:
    np.savez(BOWL_FILE, marker_id=mid, home=r["home"], step=r["step"], up=r["up"], ys=r["ys"],
             centres=r["centres"], corners=r["corners"], stamp=time.time())


def report(v: Viewer, cal: Calibration, found: dict[int, list[tuple[float, np.ndarray, float]]],
           sizes: dict[int, float], stops: int) -> bool:
    """Fit and check the axis for each marker seen, save the best, and say how it went."""
    results, lines = fit_markers(cal, found, sizes, stops)
    for line in lines:
        v.say(line)
    # The lid card rides on the bowl while the rail is measured with --rim, but it comes off:
    # it is never the base marker.
    results = {mid: r for mid, r in results.items() if mid != LID_ID}
    if not results:
        v.say("the base marker was not seen: nothing saved")
        return False
    best = best_marker(results, sizes)
    save_bowl(best, results[best])
    v.say(f"saved marker {best} to calib/bowl.npz")
    return results[best]["ok"]


if __name__ == "__main__":
    sys.exit(main())
