"""Find the bowl's rim from the lid card, once. Run bowl_calibrate.py first.

    uv run python scripts/make_marker.py --lid 150      # print calib/lid.pdf, cut the circle
    uv run python scripts/rim_calibrate.py --radius 75  # lid card centred on the bowl

The bowl is fixed to the slider, so its rim is always in the same place relative to the base
marker. Lay the lid card (marker 2 in the middle) flat on the rim, centred, and run this. It
measures both markers over many frames and saves where the lid centre is from the base marker:
along the rail, across it and up (using the rail directions from bowl_calibrate.py). After that
the lid can go; hand3d.py draws the rim wherever the base marker is.

--radius is the rim's radius in mm (measure the opening); the lid card's own size does not
matter. A live window shows both cameras; press q to close it at the end.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from bowl_calibrate import BOWL_FILE, Viewer, hold, make_detector, measure
from stereo import CALIB_DIR, CALIB_FILE, HEIGHT, LID_ID, MARKERS_MM, WIDTH, Calibration, open_cameras, wait_for_frames

RIM_FILE = CALIB_DIR / "rim.npz"


def rail_frame(step: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rows: along the rail (+Y), across it, and up, as unit vectors in the same frame as the inputs."""
    u = step / np.linalg.norm(step)
    n = up - np.dot(up, u) * u
    n /= np.linalg.norm(n)
    return np.array([u, np.cross(n, u), n])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--radius", type=float, required=True, help="rim radius, mm")
    ap.add_argument("--cams", type=int, nargs=2, default=[0, 1], help="camera numbers (2 is OBS)")
    ap.add_argument("--exposure", type=float, help="lock exposure (e.g. -5); default auto")
    ap.add_argument("--frames", type=int, default=40, help="frame pairs to use")
    a = ap.parse_args()

    cal = Calibration.load()
    if cal.size != (WIDTH, HEIGHT):
        raise SystemExit(f"{CALIB_FILE} was made at {cal.size}, cameras run at {(WIDTH, HEIGHT)}")
    if not BOWL_FILE.exists():
        raise SystemExit("run bowl_calibrate.py first: the rail directions come from it")
    bowl = np.load(BOWL_FILE)
    base_id = int(bowl["marker_id"])
    frame = rail_frame(bowl["step"], bowl["up"])
    det = make_detector()

    cams = []
    try:
        cams = open_cameras(a.cams, a.exposure)
        wait_for_frames(cams)
        v = Viewer(cal, cams, det)
        v.status = "starting cameras"
        hold(v, 1.0)
        while True:
            v.status = f"measuring: marker {base_id} (base) and marker {LID_ID} (lid) must both be magenta"
            corners, _, errs = measure(cal, cams, det, a.frames, v)
            if base_id in corners and LID_ID in corners:
                break
            missing = [f"marker {i}" for i in (base_id, LID_ID) if i not in corners]
            v.say(f"not seen well enough by both cameras: {', '.join(missing)}; trying again")

        base, lid = corners[base_id], corners[LID_ID]
        along, across, up = frame @ (lid.mean(axis=0) - base.mean(axis=0))
        lid_n = np.cross(lid[1] - lid[0], lid[3] - lid[0])
        tilt = float(np.degrees(np.arccos(min(1.0, abs(np.dot(lid_n / np.linalg.norm(lid_n), frame[2]))))))
        side = float(np.linalg.norm(np.roll(lid, -1, axis=0) - lid, axis=1).mean())
        v.say(f"rim centre from marker {base_id}: along the rail {along:+.0f} mm, across {across:+.0f} mm, "
              f"up {up:+.0f} mm")
        v.say(f"lid tilt {tilt:.1f} deg from the bed (should be near 0); lid marker {side:.1f} mm (printed {MARKERS_MM[LID_ID]:g}); "
              f"errors {errs[base_id]:.1f} / {errs[LID_ID]:.1f} px")
        np.savez(RIM_FILE, base_id=base_id, offset=np.array([along, across, up]), radius=a.radius,
                 stamp=time.time())
        v.say(f"saved, rim radius {a.radius:g} mm. The lid can come off.")
        v.status = "done: press q to close"
        while True:
            v.show()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        for c in cams:
            c.close()
        cv2.destroyAllWindows()
    return 0 if RIM_FILE.exists() else 1


if __name__ == "__main__":
    sys.exit(main())
