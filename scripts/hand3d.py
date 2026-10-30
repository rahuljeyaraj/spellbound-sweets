"""Find hands in 3D with the two calibrated webcams and MediaPipe.

    uv run python scripts/hand3d.py                       # cameras 0 and 1
    uv run python scripts/hand3d.py --bowl 120 90 0       # also show distance to a point (table mm)

Run stereo_calibrate.py first. Each camera has its own thread that grabs frames and runs the
MediaPipe hand landmarker on them. The newest result from each camera is paired, the 21 hand
points are triangulated, and the hand is reported in table coordinates in mm: X and Y along the
calibration board as it lay on the table, Z is height above the table. The origin is the board
corner OpenCV numbered first, so to get the bowl's X and Y, rest a palm on the bowl centre and
read them off the screen.

The window shows both cameras with the hand drawn on (green when matched in both, red when
only one camera has it), and below them a live map: a top view of the table with the board,
the cameras, the hand and its trail, and a graph of hand height over the last few seconds.
Bed markers are outlined too (magenta when both cameras see one, with its 3D position listed;
red when only one does), and drawn on the map with the Y rail from bowl_calibrate.py.

Two hands in view: each hand in camera 0 is matched to the hand in camera 1 whose points
triangulate with the smallest reprojection error. A large error on every hand usually means
the cameras swapped numbers after a replug; try --cams 1 0.
"""

from __future__ import annotations

import argparse
import threading
from collections import deque
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from bowl_calibrate import BOWL_FILE, draw_markers, find_markers, make_detector, marker_lines, match_markers
from stereo import (
    BOARD, CALIB_FILE, HEIGHT, ROOT, SQUARE_MM, WIDTH, Calibration, Camera, open_cameras, wait_for_frames,
)

MODEL = ROOT / "models" / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

PALM = [0, 5, 9, 13, 17]  # wrist and the four knuckles
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def ensure_model() -> None:
    if not MODEL.exists():
        MODEL.parent.mkdir(exist_ok=True)
        print("downloading the MediaPipe hand model")
        urllib.request.urlretrieve(MODEL_URL, MODEL)


class Detector:
    """Runs the hand landmarker and the marker finder on one camera's newest frames in a thread
    of its own."""

    def __init__(self, cam: Camera, max_hands: int) -> None:
        self.cam = cam
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self.marker_det = make_detector()
        self.lock = threading.Lock()
        self.result: tuple[np.ndarray, float, list[np.ndarray], dict[int, np.ndarray]] | None = None
        self.seq = 0
        self.ms = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        last_count, last_ts = -1, -1
        while self.running:
            frame, stamp, count = self.cam.latest()
            ts = int(stamp * 1000)
            if frame is None or count == last_count or ts <= last_ts:
                time.sleep(0.002)
                continue
            last_count, last_ts = count, ts
            t0 = time.perf_counter()
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = self.landmarker.detect_for_video(image, ts)
            hands = [np.array([(p.x * WIDTH, p.y * HEIGHT) for p in h]) for h in res.hand_landmarks]
            markers = find_markers(self.marker_det, frame)
            with self.lock:
                self.result = (frame, stamp, hands, markers)
                self.seq += 1
                self.ms = (time.perf_counter() - t0) * 1000

    def latest(self) -> tuple[tuple[np.ndarray, float, list[np.ndarray], dict[int, np.ndarray]] | None, int, float]:
        with self.lock:
            return self.result, self.seq, self.ms

    def close(self) -> None:
        self.running = False
        self.thread.join(timeout=1)
        self.landmarker.close()


def match_hands(cal: Calibration, hands0: list[np.ndarray], hands1: list[np.ndarray],
                max_err: float) -> list[tuple[np.ndarray, float, int, int]]:
    """Pair hands across the cameras. Returns (21 points in table mm, error px, i0, i1) per hand."""
    options = []
    for i, h0 in enumerate(hands0):
        for j, h1 in enumerate(hands1):
            X = cal.triangulate(h0, h1)
            options.append((cal.reprojection_error(X, h0, h1), i, j, X))
    options.sort(key=lambda o: o[0])
    used0, used1, out = set(), set(), []
    for err, i, j, X in options:
        if err > max_err or i in used0 or j in used1:
            continue
        used0.add(i)
        used1.add(j)
        out.append((cal.to_table(X), err, i, j))
    return out


def draw_hand(img: np.ndarray, pts: np.ndarray, colour: tuple[int, int, int]) -> None:
    for a, b in CONNECTIONS:
        cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), colour, 2)
    for p in pts:
        cv2.circle(img, tuple(int(v) for v in p), 3, colour, -1)


# One colour per tracked hand, in BGR: yellow, blue, orange, light green.
COLOURS = [(0, 255, 255), (255, 128, 0), (0, 128, 255), (128, 255, 128)]


class Track:
    """One hand followed from frame to frame, so two hands keep their own number and colour."""

    def __init__(self, number: int) -> None:
        self.number = number
        self.colour = COLOURS[(number - 1) % len(COLOURS)]
        self.palm: np.ndarray | None = None
        self.pts: np.ndarray | None = None
        self.err = 0.0
        self.seen = 0.0
        self.trail: deque[tuple[float, np.ndarray]] = deque()
        self.history: deque[tuple[float, float | None, float | None]] = deque()


class Tracker:
    """Matches each frame's hands to the hands of the frame before by nearest palm."""

    def __init__(self, smooth: float, max_jump: float = 150.0, forget: float = 0.5) -> None:
        self.smooth, self.max_jump, self.forget = smooth, max_jump, forget
        self.tracks: list[Track] = []

    def update(self, now: float, found: list[tuple[np.ndarray, float, int, int]]
               ) -> list[tuple[Track, tuple[np.ndarray, float, int, int]]]:
        """Returns (track, match) for every hand seen this frame."""
        palms = [f[0][PALM].mean(axis=0) for f in found]
        pairs = sorted((float(np.linalg.norm(p - tr.palm)), i, k)
                       for i, p in enumerate(palms) for k, tr in enumerate(self.tracks))
        given: dict[int, Track] = {}
        used: set[int] = set()
        for d, i, k in pairs:
            if d <= self.max_jump and i not in given and k not in used:
                given[i] = self.tracks[k]
                used.add(k)
        for i in range(len(found)):
            if i not in given:
                # A new hand takes the lowest free number, so hands stay 1 and 2 as they come and go.
                taken = {tr.number for tr in self.tracks}
                track = Track(next(n for n in range(1, len(taken) + 2) if n not in taken))
                self.tracks.append(track)
                given[i] = track
        out = []
        for i, f in enumerate(found):
            tr = given[i]
            tr.palm = palms[i] if tr.palm is None else self.smooth * tr.palm + (1 - self.smooth) * palms[i]
            tr.pts, tr.err, tr.seen = f[0], f[1], now
            out.append((tr, f))
        self.tracks = [tr for tr in self.tracks if now - tr.seen <= self.forget]
        self.tracks.sort(key=lambda tr: tr.number)
        return out


class Map:
    """Live picture of the hands in table coordinates: a top-down map and a height-over-time graph.

    The map is centred on where the board lay for the table photo, and marks that board and the
    two cameras so you can tell which way round it is. Only meaningful with a table photo.
    """

    def __init__(self, cal: Calibration, span: float, bowl: list[float] | None, seconds: float = 6.0,
                 max_z: float = 600.0, size: tuple[int, int] = (WIDTH, 420)) -> None:
        self.w, self.h = size
        self.side = self.h  # the map is square, the graph takes the rest
        self.scale = self.side / (2 * span)
        self.centre = np.array([(BOARD[0] - 1) * SQUARE_MM / 2, (BOARD[1] - 1) * SQUARE_MM / 2])
        self.bowl = np.array(bowl) if bowl else None
        self.seconds, self.max_z = seconds, max_z
        self.cams = [cal.to_table(np.zeros((1, 3)))[0], cal.to_table((-cal.R.T @ cal.T).reshape(1, 3))[0]]
        # The Y axis from bowl_calibrate.py: where the marker centre runs from Y0 to Y220.
        self.rail = None
        if BOWL_FILE.exists():
            b = np.load(BOWL_FILE)
            self.rail = cal.to_table(np.array([b["home"] + y * b["step"] for y in (0.0, 220.0)]))
        self.background = self._background()

    def px(self, xy: np.ndarray) -> tuple[int, int]:
        """Table mm -> map pixel (table Y points up on screen)."""
        d = (np.asarray(xy[:2]) - self.centre) * self.scale
        return int(self.side / 2 + d[0]), int(self.side / 2 - d[1])

    def gy(self, z: float) -> int:
        """Height mm -> graph pixel row."""
        return int(self.h - 20 - (self.h - 40) * min(max(z, 0), self.max_z) / self.max_z)

    def _background(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), 30, np.uint8)
        span = self.side / 2 / self.scale
        for g in np.arange(-np.ceil(span / 100) * 100, span + 100, 100):
            a = self.px(self.centre + [g, -span]), self.px(self.centre + [g, span])
            b = self.px(self.centre + [-span, g]), self.px(self.centre + [span, g])
            cv2.line(img, *a, (55, 55, 55), 1)
            cv2.line(img, *b, (55, 55, 55), 1)
        cols, rows = BOARD
        corners = np.array([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]]) * SQUARE_MM
        cv2.polylines(img, [np.array([self.px(c) for c in corners])], True, (200, 200, 200), 1)
        cv2.circle(img, self.px([0, 0]), 4, (200, 200, 200), -1)
        for i, c in enumerate(self.cams):
            p = np.clip(self.px(c), 12, self.side - 12)
            cv2.drawMarker(img, tuple(int(v) for v in p), (255, 200, 0), cv2.MARKER_TRIANGLE_UP, 16, 2)
            cv2.putText(img, f"cam {i}", (int(p[0]) + 10, int(p[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 200, 0), 1)
        if self.bowl is not None:
            cv2.circle(img, self.px(self.bowl), int(60 * self.scale), (200, 0, 200), 2)
        if self.rail is not None:
            cv2.line(img, self.px(self.rail[0]), self.px(self.rail[1]), (120, 0, 120), 3)
            cv2.circle(img, self.px(self.rail[0]), 5, (120, 0, 120), -1)
            cv2.putText(img, "Y0", (self.px(self.rail[0])[0] - 30, self.px(self.rail[0])[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 0, 160), 1)
        cv2.putText(img, "top view, grid 100 mm", (8, self.h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        cv2.line(img, (self.side, 0), (self.side, self.h), (90, 90, 90), 1)
        for z in range(0, int(self.max_z) + 1, 100):
            y = self.gy(z)
            cv2.line(img, (self.side + 45, y), (self.w - 5, y), (55, 55, 55), 1)
            cv2.putText(img, f"{z}", (self.side + 8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        cv2.putText(img, f"palm height mm, last {self.seconds:g} s, one colour per hand"
                    + ("   thin line: distance to bowl" if self.bowl is not None else ""),
                    (self.side + 50, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        return img

    def update(self, now: float, tracker: Tracker, seen: set[int]) -> None:
        for tr in tracker.tracks:
            if tr.number in seen:
                tr.trail.append((now, tr.palm))
                dist = float(np.linalg.norm(tr.palm - self.bowl)) if self.bowl is not None else None
                tr.history.append((now, float(tr.palm[2]), dist))
            else:
                tr.history.append((now, None, None))
            while tr.trail and now - tr.trail[0][0] > 2.0:
                tr.trail.popleft()
            while tr.history and now - tr.history[0][0] > self.seconds:
                tr.history.popleft()

    def draw(self, now: float, tracker: Tracker, seen: set[int],
             marks: dict[int, tuple[np.ndarray, float]]) -> np.ndarray:
        img = self.background.copy()
        for mid, (X, _) in marks.items():
            cv2.fillPoly(img, [np.array([self.px(p) for p in X])], (255, 0, 255))
            x, y = self.px(X.mean(axis=0))
            cv2.putText(img, f"id {mid}", (x + 12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        x_of = lambda t: int(self.w - 5 - (now - t) / self.seconds * (self.w - self.side - 50))
        for tr in tracker.tracks:
            for t, p in tr.trail:
                fade = 1 - (now - t) / 2.0
                cv2.circle(img, self.px(p), 3, tuple(int(c * fade) for c in tr.colour), -1)
            if tr.number in seen:
                for a, b in CONNECTIONS:
                    cv2.line(img, self.px(tr.pts[a]), self.px(tr.pts[b]), tr.colour, 2)
                # Palm dot grows as the hand comes down to the table.
                near = 1 - min(max(tr.palm[2], 0), self.max_z) / self.max_z
                cv2.circle(img, self.px(tr.palm), int(5 + 10 * near), tr.colour, -1)
                x, y = self.px(tr.palm)
                cv2.putText(img, str(tr.number), (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tr.colour, 2)
            for k, thick in ((1, 2), (2, 1)):
                run: list[tuple[int, int]] = []
                for h in list(tr.history) + [(now, None, None)]:
                    if h[k] is None:
                        if len(run) > 1:
                            cv2.polylines(img, [np.array(run)], False, tr.colour, thick)
                        run = []
                    else:
                        run.append((x_of(h[0]), self.gy(h[k])))
        return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", type=int, nargs=2, default=[0, 1], help="camera numbers (2 is OBS)")
    ap.add_argument("--hands", type=int, default=2, help="most hands to look for")
    ap.add_argument("--max-err", type=float, default=20.0, help="reject pairs above this px error")
    ap.add_argument("--smooth", type=float, default=0.5, help="0 = none, nearer 1 = smoother, slower")
    ap.add_argument("--bowl", type=float, nargs=3, metavar=("X", "Y", "Z"), help="point in table mm")
    ap.add_argument("--exposure", type=float, help="lock exposure (e.g. -6 is 1/64 s); default auto")
    ap.add_argument("--span", type=float, default=600.0, help="map shows this many mm each side")
    args = ap.parse_args()

    cal = Calibration.load()
    if cal.size != (WIDTH, HEIGHT):
        raise SystemExit(f"{CALIB_FILE} was made at {cal.size}, cameras run at {(WIDTH, HEIGHT)}")
    if cal.table_R is None:
        print("no table photo in the calibration: Z is distance from camera 0, not height; no map")
    table_map = Map(cal, args.span, args.bowl) if cal.table_R is not None else None
    ensure_model()

    cams = open_cameras(args.cams, args.exposure)
    dets: list[Detector] = []
    try:
        wait_for_frames(cams)
        dets = [Detector(c, args.hands) for c in cams]
        tracker = Tracker(args.smooth)
        last_seq = [0, 0]
        last_print = 0.0
        frames_done, fps_t0, fps = 0, time.monotonic(), 0.0
        while True:
            got = [d.latest() for d in dets]
            if any(g[0] is None for g in got) or [g[1] for g in got] == last_seq:
                if cv2.waitKey(2) & 0xFF == ord("q"):
                    break
                continue
            last_seq = [g[1] for g in got]
            (f0, s0, hands0, mk0), (f1, s1, hands1, mk1) = got[0][0], got[1][0]
            skew_ms = abs(s0 - s1) * 1000

            now = time.monotonic()
            found = match_hands(cal, hands0, hands1, args.max_err)
            current = tracker.update(now, found)
            seen = {tr.number for tr, _ in current}
            marks = match_markers(cal, mk0, mk1)

            frames_done += 1
            if now - fps_t0 >= 1.0:
                fps, frames_done, fps_t0 = frames_done / (now - fps_t0), 0, now

            v0, v1 = f0.copy(), f1.copy()
            draw_markers(v0, mk0, marks)
            draw_markers(v1, mk1, marks)
            for h in hands0:
                draw_hand(v0, h, (0, 0, 255))
            for h in hands1:
                draw_hand(v1, h, (0, 0, 255))
            for tr, (_, _, i0, i1) in current:
                for v, h in ((v0, hands0[i0]), (v1, hands1[i1])):
                    draw_hand(v, h, tr.colour)
                    cv2.putText(v, str(tr.number), tuple(int(c) + 10 for c in h[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, tr.colour, 3)
            view = np.hstack([cv2.resize(v0, (WIDTH // 2, HEIGHT // 2)), cv2.resize(v1, (WIDTH // 2, HEIGHT // 2))])

            lines = [(f"{fps:.0f} fps  detect {got[0][2]:.0f}/{got[1][2]:.0f} ms  camera skew {skew_ms:.0f} ms",
                      (0, 255, 255))]
            for tr, _ in current:
                text = (f"hand {tr.number}: X {tr.palm[0]:.0f}  Y {tr.palm[1]:.0f}  height {tr.palm[2]:.0f} mm"
                        f"   lowest point {tr.pts[:, 2].min():.0f} mm   error {tr.err:.1f} px")
                if args.bowl:
                    text += f"   to bowl {np.linalg.norm(tr.palm - np.array(args.bowl)):.0f} mm"
                lines.append((text, tr.colour))
            lines += marker_lines(mk0, mk1, marks)
            if not current and (hands0 or hands1):
                lines.append((f"hand seen by {'both cameras but no match' if hands0 and hands1 else 'one camera only'}",
                              (0, 0, 255)))
            for k, (text, colour) in enumerate(lines):
                cv2.putText(view, text, (10, 25 + 25 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            if table_map is not None:
                table_map.update(now, tracker, seen)
                view = np.vstack([view, table_map.draw(now, tracker, seen, marks)])
            cv2.imshow("hand 3d", view)

            if current and now - last_print > 0.2:
                print("   ".join(text for text, _ in lines[1:]))
                last_print = now
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for d in dets:
            d.close()
        for c in cams:
            c.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
