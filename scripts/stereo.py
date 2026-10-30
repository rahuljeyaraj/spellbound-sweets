"""Shared parts for the two-camera hand tracker: cameras, the checkerboard, calibration file.

Used by stereo_calibrate.py and hand3d.py. Not run on its own.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CALIB_DIR = ROOT / "calib"
CALIB_FILE = CALIB_DIR / "stereo.npz"

# Both cameras run at this size for calibration and tracking. Change it and you must recalibrate.
WIDTH, HEIGHT = 1280, 720

# Checkerboard: inner corners (columns, rows) and square size in mm. 9 x 6 squares of 30 mm
# fits on A4. Measure a printed square and pass --square if the printer scaled it.
BOARD = (8, 5)
SQUARE_MM = 30.0

# Bowl marker: one ArUco 4x4 marker (6 cells wide with its black border) on the moving bed.
# Marker id -> black square size in mm. 50 mm gives 8.3 mm cells, about 4.5 px each from 1.1 m
# up; 40 mm about 3.7 px (works, 2026-10-30). Id 2 is the rim lid card; it must differ from the
# base marker's id.
MARKER_DICT = cv2.aruco.DICT_4X4_50
MARKERS_MM = {0: 50.0, 1: 40.0, 2: 50.0}
LID_ID = 2


def board_points(square_mm: float) -> np.ndarray:
    """The board's inner corners in board coordinates (mm), z = 0."""
    cols, rows = BOARD
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return pts


def find_board(gray: np.ndarray) -> np.ndarray | None:
    """Return the board's inner corners in pixels, or None if not fully seen."""
    ok, corners = cv2.findChessboardCornersSB(gray, BOARD, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    # Only the SB finder: the classic one sometimes numbers the corners from the other end,
    # which pairs the wrong corners across cameras (stereo error went from 0.9 to 15 px).
    return corners if ok else None


class Camera:
    """Reads one webcam in its own thread and keeps only the newest frame and its time."""

    def __init__(self, index: int, exposure: float | None = None, wb: float | None = None) -> None:
        self.index = index
        # Open straight into MJPG at the working size. Opening first in the default raw mode takes
        # USB bandwidth the second camera then cannot get.
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW, [
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"),
            cv2.CAP_PROP_FRAME_WIDTH, WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT,
        ])
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(
                f"camera {index} did not open. If the other camera is already running, they are "
                "probably sharing one USB port or hub: plug them into separate ports.")
        # The driver remembers these after the program closes, so always set them one way or the
        # other. DirectShow: 0.25 is manual exposure, 0.75 is auto; exposure is a power of two in
        # seconds (-6 is 1/64 s). Auto exposure lets a camera drop to 24 fps in room light, so
        # lock it for a steady 30 (-5 suits the room); it goes black if the room gets darker.
        # White balance does not affect frame rate. Leave it auto: one fixed temperature turned
        # one camera blue and the other red.
        if exposure is None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        if wb is None:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (w, h) != (WIDTH, HEIGHT):
            self.cap.release()
            raise RuntimeError(f"camera {index} gave {w}x{h}, wanted {WIDTH}x{HEIGHT}")
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.stamp = 0.0
        self.count = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            stamp = time.monotonic()
            if not ok:
                continue
            with self.lock:
                self.frame, self.stamp = frame, stamp
                self.count += 1

    def latest(self) -> tuple[np.ndarray | None, float, int]:
        with self.lock:
            return self.frame, self.stamp, self.count

    def close(self) -> None:
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


def open_cameras(indexes: list[int], exposure: float | None) -> list[Camera]:
    """Open each camera; if one fails, close the ones already open before raising."""
    cams: list[Camera] = []
    try:
        for i in indexes:
            cams.append(Camera(i, exposure))
    except Exception:
        for c in cams:
            c.close()
        raise
    return cams


def wait_for_frames(cams: list[Camera], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while any(c.latest()[0] is None for c in cams):
        if time.monotonic() > deadline:
            raise TimeoutError("no frames from the cameras")
        time.sleep(0.05)


@dataclass
class Calibration:
    K0: np.ndarray
    D0: np.ndarray
    K1: np.ndarray
    D1: np.ndarray
    R: np.ndarray  # camera 0 to camera 1 rotation
    T: np.ndarray  # camera 0 to camera 1 translation, mm
    table_R: np.ndarray | None = None  # table (board) to camera 0 rotation
    table_t: np.ndarray | None = None  # table (board) to camera 0 translation, mm
    size: tuple[int, int] = (WIDTH, HEIGHT)

    def save(self, path: Path = CALIB_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {} if self.table_R is None else {"table_R": self.table_R, "table_t": self.table_t}
        np.savez(path, K0=self.K0, D0=self.D0, K1=self.K1, D1=self.D1, R=self.R, T=self.T,
                 size=np.array(self.size), **extra)

    @classmethod
    def load(cls, path: Path = CALIB_FILE) -> "Calibration":
        d = np.load(path)
        return cls(d["K0"], d["D0"], d["K1"], d["D1"], d["R"], d["T"],
                   d["table_R"] if "table_R" in d else None,
                   d["table_t"] if "table_t" in d else None,
                   tuple(int(v) for v in d["size"]))

    def triangulate(self, px0: np.ndarray, px1: np.ndarray) -> np.ndarray:
        """Pixel points (N, 2) seen by both cameras -> 3D points (N, 3) in camera 0 frame, mm."""
        u0 = cv2.undistortPoints(px0.reshape(-1, 1, 2).astype(np.float64), self.K0, self.D0)
        u1 = cv2.undistortPoints(px1.reshape(-1, 1, 2).astype(np.float64), self.K1, self.D1)
        P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = np.hstack([self.R, self.T.reshape(3, 1)])
        X = cv2.triangulatePoints(P0, P1, u0.reshape(-1, 2).T, u1.reshape(-1, 2).T)
        return (X[:3] / X[3]).T

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """3D points (N, 3) in camera 0 frame, mm -> pixels (N, 2) in camera 0 and in camera 1."""
        p0, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), self.K0, self.D0)
        p1, _ = cv2.projectPoints(X, cv2.Rodrigues(self.R)[0], self.T, self.K1, self.D1)
        return p0.reshape(-1, 2), p1.reshape(-1, 2)

    def reprojection_error(self, X: np.ndarray, px0: np.ndarray, px1: np.ndarray) -> float:
        """Mean pixel distance between the 3D points projected back and what each camera saw."""
        p0, p1 = self.project(X)
        e0 = np.linalg.norm(p0 - px0, axis=1)
        e1 = np.linalg.norm(p1 - px1, axis=1)
        return float((e0.mean() + e1.mean()) / 2)

    def to_table(self, X: np.ndarray) -> np.ndarray:
        """Camera 0 frame -> table frame (mm). Z is height above the table. Needs the table pose."""
        if self.table_R is None:
            return X
        return (X - self.table_t.reshape(1, 3)) @ self.table_R

    def from_table(self, X: np.ndarray) -> np.ndarray:
        """Table frame (mm) -> camera 0 frame. The reverse of to_table."""
        if self.table_R is None:
            return X
        return X @ self.table_R.T + self.table_t.reshape(1, 3)
