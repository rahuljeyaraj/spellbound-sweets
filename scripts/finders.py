"""Many hand finders at once on both cameras, so a hand one finder misses another catches.

Why: MediaPipe's hand landmarker looks for palms in the whole frame shrunk to 192 px, so a hand
200 px across in a 1280x720 frame is only 30 px to it. Camera 1 still found such hands; camera 0,
from its angle, mostly did not (on 2026-11-05 it missed 69 hands where camera 1 missed 2,
including an open hand right over the bowl). Tested on the saved misses: brightening, sharpening,
denoising, colour or contrast fixes found almost none of them. A crop 2 to 2.5 times the hand's
size, centred on it, found the open hand every time.

So each frame pair runs, in parallel on a pool of threads, each with its own landmarker:
  - the whole frame, both cameras (as before: it keeps tracking from frame to frame);
  - follow crops: a square crop 2.5x the hand's size around every hand that only a crop found in
    the last frame, in the camera that found it, so a hand once found stays found (a hand the
    whole frame found needs no help);
  - guided crops: a hand only one camera found must lie on that camera's line of sight. The
    stretch of that line inside the play zone is drawn into the other camera and crops are placed
    along it, so the other camera looks hard exactly where the hand has to be;
  - sweep tiles: half- and third-size tiles over the whole frame, a few per frame in turn, to
    catch a hand both cameras missed. As many as fit: the sweep grows while the frame pair takes
    under the camera's frame time and shrinks when it takes longer.
Then the hands each camera found are merged: the same hand found by several finders is kept once,
from the finder that gave it the most pixels and did not cut it at a crop edge.

The CPU is the limit, not the threads: one landmarker call already spreads over several cores, and
the machine (Core Ultra 5 125H) tops out near 300 calls a second, threads or processes alike.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from bowl_calibrate import find_markers, make_detector
from hand3d import MODEL, PALM
from stereo import HEIGHT, WIDTH, Calibration, Camera

Rect = tuple[int, int, int, int]  # x, y, w, h
FOLLOW = 2.5  # follow and guided crops: this many times the hand's size
MIN_SIDE, MAX_SIDE = 240, HEIGHT  # crop side limits, px
MAX_FOLLOW = 4  # follow crops per camera
MAX_GUIDED = 3  # guided crops per lone hand
FRAME_S = 1 / 30  # camera frame time: the sweep grows while a frame pair takes less


def landmarker(mode: vision.RunningMode, hands: int) -> vision.HandLandmarker:
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=mode, num_hands=hands,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))


def tiles(frac: float, n: int) -> list[Rect]:
    """n x n tiles, each frac of the frame, spread evenly: neighbours overlap."""
    w, h = int(WIDTH * frac), int(HEIGHT * frac)
    return [(int(x), int(y), w, h) for y in np.linspace(0, HEIGHT - h, n) for x in np.linspace(0, WIDTH - w, n)]


def square(cx: float, cy: float, side: float) -> Rect:
    """A square crop centred near (cx, cy), pushed inside the frame."""
    s = int(np.clip(side, MIN_SIDE, MAX_SIDE))
    x = int(np.clip(cx - s / 2, 0, WIDTH - s))
    y = int(np.clip(cy - s / 2, 0, HEIGHT - s))
    return x, y, s, s


def hand_box(pts: np.ndarray) -> tuple[float, float, float]:
    """Centre x, y and size (the longer side of the box round the 21 points), px."""
    lo, hi = pts.min(0), pts.max(0)
    return (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, float((hi - lo).max())


class Finders:
    """Finds hands on both cameras' newest frames with many landmarkers in parallel. latest()
    gives per camera the same (frame, stamp, hands, markers) the one-landmarker Detector gave."""

    def __init__(self, cams: list[Camera], cal: Calibration, zone: Callable[[np.ndarray], np.ndarray],
                 reach: float, threads: int = 12, sweep: int = 8, hands: int = 2) -> None:
        """zone: table points (N, 3) -> each one's distance from where hands play, mm (0 over the
        bowl's path). Guided crops look along the part of a line of sight within reach of it,
        nearest first. sweep: the most sweep tiles per camera per frame pair."""
        self.cams, self.cal, self.zone, self.reach = cams, cal, zone, reach
        self.sweep_max, self.sweep = sweep, min(2, sweep)
        self.full = [landmarker(vision.RunningMode.VIDEO, hands) for _ in cams]
        self.made = list(self.full)  # every landmarker, to close them all at the end
        self.marker_dets = [make_detector() for _ in cams]
        self.local = threading.local()
        self.pool = ThreadPoolExecutor(threads)
        self.grid = tiles(1 / 2, 3) + tiles(1 / 3, 4)  # 9 + 16 tiles per camera
        self.turn = [0, 0]  # next sweep tile per camera
        self.c1 = (-cal.R.T @ cal.T.reshape(3)).reshape(3)  # camera 1's centre, camera 0 frame
        self.lock = threading.Lock()
        self.result: list | None = None
        self.seq = 0
        self.ms = 0.0
        self.stats = ""
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # --- one finder ---

    def _crop(self, frame: np.ndarray, rect: Rect) -> list[np.ndarray]:
        lm = getattr(self.local, "lm", None)
        if lm is None:
            lm = self.local.lm = landmarker(vision.RunningMode.IMAGE, 2)
            self.made.append(lm)
        x, y, w, h = rect
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame[y:y + h, x:x + w, ::-1]))
        return [np.array([(x + p.x * w, y + p.y * h) for p in hd]) for hd in lm.detect(img).hand_landmarks]

    def _full(self, k: int, frame: np.ndarray, ts: int) -> list[np.ndarray]:
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame[:, :, ::-1]))
        res = self.full[k].detect_for_video(img, ts)
        return [np.array([(p.x * WIDTH, p.y * HEIGHT) for p in hd]) for hd in res.hand_landmarks]

    # --- where to look ---

    def _guided(self, j: int, pts: np.ndarray) -> list[Rect]:
        """Crops in the other camera along the stretch of camera j's line of sight through this hand
        that lies in the play zone, nearest the zone's middle first."""
        cal, k = self.cal, 1 - j
        c = pts[PALM].mean(0).reshape(1, 1, 2).astype(np.float64)
        K, D = (cal.K0, cal.D0) if j == 0 else (cal.K1, cal.D1)
        u = cv2.undistortPoints(c, K, D).reshape(2)
        ray = np.array([u[0], u[1], 1.0])
        origin, d = (np.zeros(3), ray) if j == 0 else (self.c1, cal.R.T @ ray)
        d = d / np.linalg.norm(d)
        t = np.linspace(150, 4000, 400)
        X = origin + t[:, None] * d  # camera 0 frame
        dist = self.zone(cal.to_table(X))
        p0, p1 = cal.project(X)
        depth_k = X[:, 2] if k == 0 else (X @ cal.R.T + cal.T.reshape(1, 3))[:, 2]
        ok = (dist < self.reach) & (depth_k > 100)
        if not ok.any():
            return []
        pk = (p0 if k == 0 else p1)[ok]
        size = hand_box(pts)[2] * t[ok] / depth_k[ok]  # the hand's size in camera k, roughly
        rects, placed = [], []
        for i in np.argsort(dist[ok], kind="stable"):
            side = FOLLOW * size[i]
            if any(np.hypot(*(pk[i] - q)) < side * 0.5 for q in placed):
                continue
            if 0 <= pk[i][0] < WIDTH and 0 <= pk[i][1] < HEIGHT:
                placed.append(pk[i])
                rects.append(square(pk[i][0], pk[i][1], side))
            if len(rects) == MAX_GUIDED:
                break
        return rects

    def _plan(self, follow: list[list[np.ndarray]], lone: list[tuple[int, np.ndarray]]) -> list[list[Rect]]:
        """Crops per camera for this frame pair: follow, guided, then sweep tiles."""
        plan: list[list[Rect]] = [[], []]
        for k in (0, 1):
            for pts in sorted(follow[k], key=lambda p: -hand_box(p)[2])[:MAX_FOLLOW]:
                cx, cy, s = hand_box(pts)
                plan[k].append(square(cx, cy, FOLLOW * s))
        for j, pts in lone:
            plan[1 - j] += self._guided(j, pts)[:max(0, MAX_FOLLOW + MAX_GUIDED * 2 - len(plan[1 - j]))]
        for k in (0, 1):
            for _ in range(self.sweep):
                plan[k].append(self.grid[self.turn[k] % len(self.grid)])
                self.turn[k] += 1
        return plan

    # --- merging ---

    @staticmethod
    def _cut(pts: np.ndarray, rect: Rect) -> bool:
        """The hand touches an edge of its crop that is not the frame's edge: it may be cut off."""
        x, y, w, h = rect
        m = 3
        lo, hi = pts.min(0), pts.max(0)
        return ((x > 0 and lo[0] < x + m) or (y > 0 and lo[1] < y + m)
                or (x + w < WIDTH and hi[0] > x + w - m) or (y + h < HEIGHT and hi[1] > y + h - m))

    @staticmethod
    def _same(a: np.ndarray, b: np.ndarray) -> bool:
        """Two finds of one hand: their points sit on each other."""
        return np.linalg.norm(a - b, axis=1).mean() < 0.25 * max(hand_box(a)[2], hand_box(b)[2])

    @staticmethod
    def _merge(found: list[tuple[np.ndarray, Rect]]) -> list[np.ndarray]:
        """One copy of each hand: best first (whole in its crop, then the most pixels per hand),
        and any later find whose points sit on a kept hand's is the same hand."""
        def rank(f):
            pts, rect = f
            return (Finders._cut(pts, rect), max(rect[2], rect[3]) / max(hand_box(pts)[2], 1))
        # A hand cut at a crop edge ranks last but still counts if nothing else found it: it is
        # checked in 3D against the other camera like any other.
        kept: list[np.ndarray] = []
        for pts, rect in sorted(found, key=rank):
            if any(Finders._same(pts, q) for q in kept):
                continue
            kept.append(pts)
        return kept

    # --- the loop ---

    def _loop(self) -> None:
        from hand3d import match_hands  # here, so this module imports without a circular import
        last_counts, last_ts = (-1, -1), [-1, -1]
        follow: list[list[np.ndarray]] = [[], []]
        lone: list[tuple[int, np.ndarray]] = []
        while self.running:
            got = [c.latest() for c in self.cams]
            counts = tuple(g[2] for g in got)
            # Wait for a new frame from both cameras (both run at 30 fps).
            if any(g[0] is None for g in got) or counts[0] == last_counts[0] or counts[1] == last_counts[1]:
                time.sleep(0.002)
                continue
            last_counts = counts
            t0 = time.perf_counter()
            frames = [g[0] for g in got]
            ts = []
            for k, g in enumerate(got):
                ts.append(max(int(g[1] * 1000), last_ts[k] + 1))
                last_ts[k] = ts[k]
            plan = self._plan(follow, lone)
            jobs = [(k, None, self.pool.submit(self._full, k, frames[k], ts[k])) for k in (0, 1)]
            jobs += [(k, r, self.pool.submit(self._crop, frames[k], r)) for k in (0, 1) for r in plan[k]]
            markers = [self.pool.submit(find_markers, self.marker_dets[k], frames[k]) for k in (0, 1)]
            found: list[list[tuple[np.ndarray, Rect]]] = [[], []]
            whole: list[list[np.ndarray]] = [[], []]  # what the whole-frame finders found
            for k, r, fut in jobs:
                for pts in fut.result():
                    found[k].append((pts, r or (0, 0, WIDTH, HEIGHT)))
                    if r is None:
                        whole[k].append(pts)
            hands = [self._merge(found[k]) for k in (0, 1)]
            follow = [[h for h in hands[k] if not any(self._same(h, w) for w in whole[k])] for k in (0, 1)]
            mk = [m.result() for m in markers]
            # Hands only one camera has: the other camera searches along their line of sight next time.
            pairs = match_hands(self.cal, hands[0], hands[1], 20)
            lone = [(k, h) for k in (0, 1) for i, h in enumerate(hands[k])
                    if i not in {p[2 + k] for p in pairs}]
            secs = time.perf_counter() - t0
            # Use what is spare: more sweep tiles while a frame pair takes under a camera frame.
            if secs < FRAME_S * 0.85 and self.sweep < self.sweep_max:
                self.sweep += 1
            elif secs > FRAME_S * 1.1 and self.sweep > 0:
                self.sweep -= 1
            n = len(jobs)
            with self.lock:
                self.result = [(frames[k], got[k][1], hands[k], mk[k]) for k in (0, 1)]
                self.seq += 1
                self.ms = secs * 1000
                self.stats = (f"finders {n} ({len(plan[0])}+{len(plan[1])} crops, sweep {self.sweep}) "
                              f"{secs * 1000:.0f} ms")

    def latest(self) -> list[tuple[tuple | None, int, float]]:
        """Per camera: ((frame, stamp, hands, markers) or None, seq, ms), like Detector.latest()."""
        with self.lock:
            if self.result is None:
                return [(None, 0, 0.0), (None, 0, 0.0)]
            return [(self.result[k], self.seq, self.ms) for k in (0, 1)]

    def close(self) -> None:
        self.running = False
        self.thread.join(timeout=2)
        self.pool.shutdown(wait=True, cancel_futures=True)
        for lm in self.made:
            lm.close()
