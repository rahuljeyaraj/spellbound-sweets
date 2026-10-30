"""The bowl dodges a reaching hand.

    uv run python scripts/dodge.py --exposure -5                    # dry run: nothing moves
    uv run python scripts/dodge.py --port COM9 --exposure -5        # the real thing
    uv run python scripts/dodge.py --port COM9 --speed 300 --accel 3000 --trigger 140

Needs stereo_calibrate.py, bowl_calibrate.py and rim_calibrate.py done first. Run
bowl_calibrate.py again whenever the table or printer has moved.

Every frame, both cameras find the hands; all 21 points of each hand are put into "rail
coordinates" around the rim centre: how far along the rail, how far across it and how high
above the rim. Where the bowl is comes from the axis, not the camera: the Y we sent plus the
speed curve of the move, so it is known mid-move and when a hand covers the marker.

Dodge: when any hand point is within --trigger mm of the rim centre (seen from above) and no
more than --height mm above the rim, the bowl jumps to whichever end of the rail is furthest
from the hand (only ever the ends: short steps looked like the bowl following the hand). The
slow return to the middle goes in 12 mm hops, so a new hand never waits for it to finish. If it
is already there, it stays. A new dodge waits until the last move has finished, and until the
hand has come --hyst mm closer than it has been since (so a hand hovering between the ends does
not send the bowl back and forth); a hand over the opening always counts.

Feints: a random --feint share of dodges go only 40 to 80% of the way, then dart back to 0 to
40 mm from where the bowl started, so a hand that has learned to follow it is caught out. The
return is skipped if a hand below the rim is in its way. Hand points below the rim never trigger a dodge, and if one
is in the bowl or in its path, the bowl stays put: moving into a hand made it lose steps. After --rest-after seconds with no hand
near, the bowl drifts back to the middle slowly.

Lost steps: whenever the bowl is still, the base marker's position is compared with the axis.
If they differ by more than --slip mm, it re-homes by itself (about 3 s) and goes on. (Not
G92: Marlin shifts its soft limits with it, so a stall point became a wall.) 800 mm/s at 8000 mm/s2 with the bowl lost steps on long moves
(2026-10-30); 700 at 7000 was the last clean bench speed with weight on.

Keys: q quit, h re-home, d dodging on or off.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from bowl_calibrate import BOWL_FILE, draw_markers, match_markers
from hand3d import Detector, Map, Tracker, draw_hand, ensure_model, match_hands
from rim_calibrate import RIM_FILE, rail_frame
from stereo import CALIB_FILE, HEIGHT, WIDTH, Calibration, open_cameras, wait_for_frames
from y_speed import y_switch_triggered
from y_sweep import Marlin

YMIN, YMAX = 5.0, 215.0
RETURN_HOP = 12.0  # mm per hop when creeping back to rest (about 0.15 s each)
LAG = 0.08  # s from sending a move to the bed starting (bench-log: measured 0.08 to 0.1 s over)


class Rail:
    """The rail and the rim in table coordinates, from the bowl and rim calibration files."""

    def __init__(self, cal: Calibration) -> None:
        b, r = np.load(BOWL_FILE), np.load(RIM_FILE)
        if int(r["base_id"]) != int(b["marker_id"]):
            raise SystemExit("rim.npz was made with a different base marker: run rim_calibrate.py again")
        self.base_id = int(b["marker_id"])
        axes = rail_frame(b["step"], b["up"])  # camera 0 frame
        self.scale = float(np.linalg.norm(b["step"]))  # mm moved per mm of Y
        self.home = cal.to_table(b["home"].reshape(1, 3))[0]  # base marker centre at Y0
        self.axes = axes @ cal.table_R  # along, across, up in table coordinates
        self.rim_offset = r["offset"] @ self.axes  # rim centre from the base marker
        self.rim0 = self.home + self.rim_offset  # rim centre at Y0
        self.radius = float(r["radius"])

    def anchor(self, centre: np.ndarray, y: float) -> np.ndarray:
        """Re-place the rail so the base marker centre (table mm) is where it should be at Y=y,
        e.g. right after homing. Keeps the rail's direction. Returns the shift (along, across,
        up), mm: how far the printer has moved since bowl_calibrate.py."""
        new_home = centre - y * self.scale * self.axes[0]
        shift = (new_home - self.home) @ self.axes.T
        self.home = new_home
        self.rim0 = self.home + self.rim_offset
        return shift

    def rim_centre(self, y: float) -> np.ndarray:
        return self.rim0 + y * self.scale * self.axes[0]

    def to_rail(self, P: np.ndarray, y: float) -> np.ndarray:
        """Table points (N, 3) -> (N, 3): along the rail from the rim centre (mm), across, height above rim."""
        return (P - self.rim_centre(y)) @ self.axes.T

    def marker_y(self, centre: np.ndarray) -> float:
        """The Y the axis must be at for the base marker to be where it is seen."""
        return float((centre - self.home) @ self.axes[0]) / self.scale

    def ring(self, y: float, radius: float | None = None, n: int = 48) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = self.radius if radius is None else radius
        return self.rim_centre(y) + r * (np.outer(np.cos(t), self.axes[0]) + np.outer(np.sin(t), self.axes[1]))


def travelled(dist: float, speed: float, accel: float, t: float) -> float:
    """Distance covered t seconds into a rest-to-rest move with a trapezoid speed profile."""
    if t <= 0:
        return 0.0
    if speed * speed / accel >= dist:  # never reaches top speed: a triangle
        tp = (dist / accel) ** 0.5
        return 0.5 * accel * t * t if t < tp else dist - 0.5 * accel * max(2 * tp - t, 0) ** 2
    ta = speed / accel
    tc = (dist - speed * ta) / speed
    if t < ta:
        return 0.5 * accel * t * t
    if t < ta + tc:
        return 0.5 * speed * ta + speed * (t - ta)
    return dist - 0.5 * accel * max(2 * ta + tc - t, 0) ** 2


class Axis:
    """The Y axis: sends moves (or pretends to, in a dry run) and knows where the bed is mid-move."""

    def __init__(self, m: Marlin | None, y: float) -> None:
        self.m = m
        self.start = self.target = y
        self.t0 = 0.0
        self.dur = 0.0
        self.speed = self.accel = 1.0

    def move(self, target: float, speed: float, accel: float) -> None:
        target = min(max(target, YMIN), YMAX)
        now = time.monotonic()
        self.start, self.target = self.y(now), target
        self.speed, self.accel = speed, accel
        dist = abs(target - self.start)
        self.dur = (2 * (dist / accel) ** 0.5 if speed * speed / accel >= dist
                    else 2 * speed / accel + (dist - speed * speed / accel) / speed)
        if self.m:
            self.m.send(f"M204 P{accel:g} T{accel:g}")
            self.m.send(f"G0 Y{target:.1f} F{speed * 60:g}")
        self.t0 = time.monotonic() + LAG

    def y(self, now: float) -> float:
        dist = abs(self.target - self.start)
        if dist == 0:
            return self.target
        d = travelled(dist, self.speed, self.accel, min(now - self.t0, self.dur))
        return self.start + np.sign(self.target - self.start) * d

    def moving(self, now: float) -> bool:
        return now < self.t0 + self.dur


def dodge_target(y: float, hand_along: float) -> float:
    """Where to send the bowl: the end of the rail furthest from the hand. Always an end, never
    a short step, so it jumps away instead of seeming to follow the hand. hand_along is from the
    rim centre, in mm along the rail."""
    hand_y = y + hand_along
    return YMIN if hand_y - YMIN > YMAX - hand_y else YMAX


def path_blocked(low: np.ndarray, travel: float, radius: float, margin: float = 25.0) -> bool:
    """True if a hand point below the rim (rail coordinates, (N, 3)) is in the bowl now or in the
    strip it would sweep moving `travel` mm along the rail."""
    if len(low) == 0:
        return False
    lo, hi = min(0.0, travel) - radius - margin, max(0.0, travel) + radius + margin
    return bool(np.any((low[:, 0] > lo) & (low[:, 0] < hi) & (np.abs(low[:, 1]) < radius + margin)))


def home(m: Marlin, show) -> None:
    m.send("M204 P500 T500")
    m.send("G28 Y", timeout=60, idle=show)
    m.send("G0 Y20 F3000")
    m.send("M400", timeout=60, idle=show)
    if y_switch_triggered(m):
        raise RuntimeError("the Y switch reads pressed at Y20, so the home was not real; check the switch")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--port", help="the SKR's serial port, e.g. COM9; leave out for a dry run")
    ap.add_argument("--cams", type=int, nargs=2, default=[0, 1], help="camera numbers (2 is OBS)")
    ap.add_argument("--exposure", type=float, help="lock exposure (e.g. -5); default auto")
    ap.add_argument("--speed", type=float, default=700, help="dodge top speed, mm/s")
    ap.add_argument("--accel", type=float, default=7000, help="dodge acceleration, mm/s^2")
    ap.add_argument("--trigger", type=float, default=250, help="dodge when a hand point is this close, mm")
    ap.add_argument("--height", type=float, default=250, help="...and at most this high above the rim, mm")
    ap.add_argument("--feint", type=float, default=0.3, help="share of dodges that are feints, 0 to 1")
    ap.add_argument("--hyst", type=float, default=50, help="after a move, the hand must come this much closer, mm")
    ap.add_argument("--rest", type=float, default=110, help="where the bowl waits, Y mm")
    ap.add_argument("--rest-after", type=float, default=3.0, help="seconds with no hand near before going back")
    ap.add_argument("--slip", type=float, default=6.0, help="marker vs axis difference that means lost steps, mm")
    ap.add_argument("--max-err", type=float, default=20.0, help="reject hand pairs above this px error")
    ap.add_argument("--spread", action="store_true", help="Y driver in spreadCycle: louder, more torque at speed")
    a = ap.parse_args()

    cal = Calibration.load()
    if cal.size != (WIDTH, HEIGHT):
        raise SystemExit(f"{CALIB_FILE} was made at {cal.size}, cameras run at {(WIDTH, HEIGHT)}")
    if cal.table_R is None:
        raise SystemExit("the camera calibration has no table photo")
    for f, how in ((BOWL_FILE, "bowl_calibrate.py"), (RIM_FILE, "rim_calibrate.py")):
        if not f.exists():
            raise SystemExit(f"run {how} first")
    rail = Rail(cal)
    table_map = Map(cal, 600.0, None)
    ensure_model()

    m = Marlin(a.port) if a.port else None
    cams, dets = [], []
    cv2.namedWindow("dodge", cv2.WINDOW_NORMAL)  # resizable, to fit the screen
    cv2.resizeWindow("dodge", WIDTH * 3 // 4, (70 + HEIGHT // 2 + 420) * 3 // 4)
    try:
        cams = open_cameras(a.cams, a.exposure)
        wait_for_frames(cams)
        dets = [Detector(c, 2) for c in cams]
        tracker = Tracker(0.5)
        axis = Axis(m, a.rest)
        state = {"text": "starting", "banner": None, "enabled": True, "log": []}
        last_seq = [0, 0]
        last_near = time.monotonic()
        still_since = time.monotonic()
        dry_placed = False
        slips = 0
        ref = a.trigger + a.hyst
        last_hold = None
        pending: float | None = None  # where a feint goes back to once its first move ends
        rng = np.random.default_rng()

        def say(text: str) -> None:
            print(text)
            state["log"] = (state["log"] + [text])[-5:]

        def frame_view() -> tuple | None:
            """Newest results from both cameras, or None if nothing new."""
            got = [d.latest() for d in dets]
            if any(g[0] is None for g in got) or [g[1] for g in got] == last_seq:
                return None
            last_seq[:] = [g[1] for g in got]
            return got

        def draw(got, now: float, current, marks, near_pts) -> None:
            (f0, s0, hands0, mk0), (f1, s1, hands1, mk1) = got[0][0], got[1][0]
            y = axis.y(now)
            v0, v1 = f0.copy(), f1.copy()
            draw_markers(v0, mk0, marks)
            draw_markers(v1, mk1, marks)
            colour = (0, 0, 255) if axis.moving(now) else (0, 220, 0)
            for ring, thick in ((rail.ring(y), 3), (rail.ring(y, a.trigger), 1)):
                for v, p in zip((v0, v1), cal.project(cal.from_table(ring))):
                    cv2.polylines(v, [p.astype(np.int32)], True, colour, thick)
            for v, hs in ((v0, hands0), (v1, hands1)):
                for h in hs:
                    draw_hand(v, h, (0, 0, 255))
            for tr, (_, _, i0, i1) in current:
                draw_hand(v0, hands0[i0], tr.colour)
                draw_hand(v1, hands1[i1], tr.colour)
            view = np.hstack([cv2.resize(v0, (WIDTH // 2, HEIGHT // 2)), cv2.resize(v1, (WIDTH // 2, HEIGHT // 2))])
            # Status bar above the pictures, nothing written over them: the state, big, coloured
            # by what the bowl is doing, and one small line of numbers.
            text = state["banner"] or (("DRY RUN: " if m is None else "") + state["text"])
            upper = text.upper()
            bg = ((0, 0, 170) if state["banner"] else (0, 0, 150) if axis.moving(now)
                  else (0, 110, 160) if upper.startswith("HOLDING") or "HOM" in upper else (40, 40, 40))
            bar = np.full((70, WIDTH, 3), bg, np.uint8)
            cv2.putText(bar, text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            info = f"bowl Y {y:.0f}"
            if near_pts:
                tr, rp = min(near_pts, key=lambda n: np.hypot(n[1][0], n[1][1]))
                info += f"     hand {np.hypot(rp[0], rp[1]):.0f} mm from bowl, {rp[2]:+.0f} mm above rim"
            else:
                info += "     no hand"
            cv2.putText(bar, info, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            view = np.vstack([bar, view])
            seen = {tr.number for tr, _ in current}
            table_map.update(now, tracker, seen)
            mp = table_map.draw(now, tracker, seen, marks)
            for r, thick in ((rail.radius, 3), (a.trigger, 1)):
                cv2.circle(mp, table_map.px(rail.rim_centre(y)), int(r * table_map.scale), colour, thick)
            cv2.imshow("dodge", np.vstack([view, mp]))

        def anchor(y_now: float) -> None:
            """Bed still at a known Y (just homed): measure the base marker and re-place the rail."""
            seen = []
            end = time.monotonic() + 3.0
            while len(seen) < 8 and time.monotonic() < end:
                got = frame_view()
                if got is None:
                    cv2.waitKey(1)
                    continue
                marks = match_markers(cal, got[0][0][3], got[1][0][3])
                draw(got, time.monotonic(), [], marks, [])
                cv2.waitKey(1)
                if rail.base_id in marks:
                    seen.append(marks[rail.base_id][0].mean(axis=0))
            if len(seen) < 4:
                say(f"marker {rail.base_id} not seen after homing: rail not re-checked")
                return
            along, across, up = rail.anchor(np.median(seen, axis=0), y_now)
            say(f"anchored on marker: printer moved {along:+.0f} along, {across:+.0f} across, "
                f"{up:+.0f} up since calibration")
            if abs(across) > 10 or abs(up) > 10:
                state["banner"] = "printer moved sideways since calibration: run bowl_calibrate.py"
                say(state["banner"])

        def idle_show() -> None:
            got = frame_view()
            if got is not None:
                now = time.monotonic()
                marks = match_markers(cal, got[0][0][3], got[1][0][3])
                draw(got, now, [], marks, [])
            cv2.waitKey(1)

        if m:
            m.send("G21")
            m.send("G90")
            m.send(f"M569 S{0 if a.spread else 1} Y")  # stealthChop (quiet) unless --spread
            m.send(f"M203 Y{a.speed:g}")
            m.send(f"M201 Y{a.accel:g}")
            state["text"] = "homing Y"
            home(m, idle_show)
            anchor(20.0)
            axis = Axis(m, 20.0)
            axis.move(a.rest, 100, 1000)
        say(f"ready: dodge at {a.speed:g} mm/s, {a.accel:g} mm/s2, trigger {a.trigger:g} mm")
        state["text"] = "watching"

        while True:
            got = frame_view()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                state["enabled"] = not state["enabled"]
                say("dodging " + ("on" if state["enabled"] else "off"))
            if key == ord("h") and m and not axis.moving(time.monotonic()):
                state["text"] = "re-homing"
                home(m, idle_show)
                anchor(20.0)
                axis = Axis(m, 20.0)
                pending = None
                axis.move(a.rest, 100, 1000)
                state["banner"] = None
                say("re-homed")
            if got is None:
                continue
            now = time.monotonic()
            (f0, s0, hands0, mk0), (f1, s1, hands1, mk1) = got[0][0], got[1][0]
            found = match_hands(cal, hands0, hands1, a.max_err)
            current = tracker.update(now, found)
            marks = match_markers(cal, mk0, mk1)
            y = axis.y(now)

            # Every hand point in rail coordinates around the rim centre: along, across, above the rim.
            hand_rps = [(tr, rail.to_rail(pts, y)) for tr, (pts, _, _, _) in current]
            near_pts = [(tr, rp[np.argmin(np.hypot(rp[:, 0], rp[:, 1]))]) for tr, rp in hand_rps]
            if near_pts and min(np.hypot(rp[0], rp[1]) for _, rp in near_pts) < a.trigger + 150:
                last_near = now
            # Only points above the rim count as reaching. Below it a hand is beside or in the bowl,
            # and moving would hit it (that is what made it lose steps).
            reach = [p for _, rp in hand_rps for p in rp if 0 <= p[2] < a.height]
            dist = min((float(np.hypot(p[0], p[1])) for p in reach), default=np.inf)
            low = np.array([p for _, rp in hand_rps for p in rp if p[2] < 0]).reshape(-1, 3)

            if m is None and not dry_placed and rail.base_id in marks:
                # Dry run: start the pretend axis where the bowl really is.
                axis = Axis(None, rail.marker_y(marks[rail.base_id][0].mean(axis=0)))
                dry_placed = True
                y = axis.y(now)
            moving = axis.moving(now)
            if moving:
                still_since = now
            elif m and now - still_since > 0.3 and rail.base_id in marks and not state["banner"]:
                seen_y = rail.marker_y(marks[rail.base_id][0].mean(axis=0))
                if abs(seen_y - y) > a.slip:
                    # Lost steps: re-home. Not G92: Marlin moves the soft limits with it, so the
                    # spot where the bowl stalled became a wall it would not pass.
                    slips += 1
                    say(f"lost steps: axis said Y{y:.0f}, marker says Y{seen_y:.0f}; re-homing ({slips} so far)")
                    state["text"] = "re-homing"
                    home(m, idle_show)
                    anchor(20.0)
                    axis = Axis(m, 20.0)
                    pending = None
                    axis.move(a.rest, 100, 1000)
                    state["text"] = "watching"

            # Hysteresis: after a move, the hand must come --hyst mm closer than it has been since
            # (ref follows the hand's furthest distance) before the next dodge. A hand hovering
            # between the ends no longer sends the bowl back and forth.
            if moving or not np.isfinite(dist):
                ref = dist if moving and np.isfinite(dist) else a.trigger + a.hyst
            else:
                ref = min(a.trigger + a.hyst, max(ref, dist))
            # A hand over the opening always counts; otherwise a hand that followed the bowl to
            # its new spot could never come the extra --hyst mm closer.
            reaching = dist < a.trigger and dist < max(ref - a.hyst, rail.radius)

            # Why the bowl is not moving although a hand is near, logged when it changes.
            hold = None
            if not moving and not reaching:
                if dist < a.trigger:
                    hold = (f"holding: hand still at {dist:.0f} mm, must come within "
                            f"{max(ref - a.hyst, rail.radius):.0f} mm")
                elif path_blocked(low, 0.0, rail.radius):
                    hold = "holding: hand below the rim at the bowl"

            if pending is not None and not moving:
                # Second half of a feint: dart back, unless a low hand is now in the way.
                if path_blocked(low, (pending - y) * rail.scale, rail.radius) or state["banner"]:
                    say("feint: return cancelled, hand in the way")
                else:
                    axis.move(pending, a.speed, a.accel)
                    state["text"] = f"FEINT back to Y{pending:.0f}"
                    moving = True
                pending = None
            elif reaching and not moving and state["enabled"] and not state["banner"]:
                p = min(reach, key=lambda q: np.hypot(q[0], q[1]))
                target = dodge_target(y, p[0])
                if abs(target - y) <= 5:
                    hold = "holding: cornered at the end of the rail"
                elif path_blocked(low, (target - y) * rail.scale, rail.radius):
                    hold = "holding: hand below the rim in the bowl's path"
                else:
                    seen_at = max(s0, s1)  # when the newer of the two frames was taken
                    kind = "dodge"
                    if rng.random() < a.feint:
                        # Feint: only part of the way, then back near where it started.
                        back = min(max(y + np.sign(y - target) * rng.uniform(0, 40), YMIN), YMAX)
                        target = y + (target - y) * rng.uniform(0.4, 0.8)
                        pending = back
                        kind = "feint"
                    axis.move(target, a.speed, a.accel)
                    state["text"] = f"{kind.upper()} to Y{target:.0f}"
                    say(f"{kind}: hand {dist:.0f} mm away, {p[2]:+.0f} mm above rim; Y{y:.0f} to Y{target:.0f}"
                        + (f" then back to Y{pending:.0f}" if kind == "feint" else "")
                        + f"; frame to command {1000 * (time.monotonic() - seen_at):.0f} ms")
            elif not moving and now - last_near > a.rest_after and abs(y - a.rest) > 2 and not state["banner"]:
                # Creep back in short hops, so a hand arriving mid-way waits at most one hop.
                axis.move(y + float(np.clip(a.rest - y, -RETURN_HOP, RETURN_HOP)), 150, 2000)
                state["text"] = "creeping back to the middle"
            elif not moving and not reaching:
                state["text"] = "watching" if state["enabled"] else "dodging off"
            if hold:
                state["text"] = hold
            if (hold or "").split(" at ")[0] != (last_hold or "").split(" at ")[0]:
                if hold:
                    say(hold)
                last_hold = hold

            draw(got, now, current, marks, near_pts)
    except KeyboardInterrupt:
        if m:
            m.ser.write(b"M410\n")
        print("stopped")
    finally:
        for d in dets:
            d.close()
        for c in cams:
            c.close()
        if m:
            try:
                m.send("M400", timeout=10)
                m.send("M501")  # back to the saved speed settings
            finally:
                m.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
