"""The bowl dodges a reaching hand.

    uv run python scripts/dodge.py                                  # dry run: nothing moves
    uv run python scripts/dodge.py --port COM9                      # the real thing (cameras on auto)
    uv run python scripts/dodge.py --port COM9 --calibrate          # calibrate the rail first, in here
    uv run python scripts/dodge.py --port COM9 --exposure auto -5 --gain 0 64   # one locked, if ever needed
    uv run python scripts/dodge.py --port COM9 --speed 300 --accel 3000 --trigger 140

Needs stereo_calibrate.py, bowl_calibrate.py and rim_calibrate.py done first. The printer slides
on the table, so calibrate the rail before every start: bowl_calibrate.py, or --calibrate here,
which does the same measurement and check in this window. After every home the base marker is
measured; if the printer has moved sideways (over 10 mm) dodge calibrates the rail by itself
(about 30 s, the bed going slowly through five stops) and goes on. If that check fails, nothing is
saved and dodging stays off with the reason on the banner; h tries again.

Hands are found by finders.py: many hand landmarkers at once on both cameras (whole frames,
crops following each hand, crops along the other camera's line of sight, sweeping tiles), as one
landmarker on the whole frame missed most hands in camera 0. --plain goes back to that.
Cameras on auto exposure (the default). A locked exposure looked different from run to run: the
driver keeps the gain auto last chose, so the same number came out dark one time and washed out
the next (2026-11-05). On auto both cameras held 30 fps in room light.

Every frame, both cameras find the hands; all 21 points of each hand are put into "rail
coordinates" around the rim centre: how far along the rail, how far across it and how high
above the rim. Where the bowl is comes from the axis, not the camera: the Y we sent plus the
speed curve of the move, so it is known mid-move and when a hand covers the marker.

Dodge: when any hand point is within --trigger mm of the rim centre (seen from above) and no
more than --height mm above the rim, the bowl jumps to whichever end of the rail is furthest
from the hand (only ever the ends: short steps looked like the bowl following the hand). If it
is already there, it stays. Only the two ends are places to stay: after homing the bowl waits at
the home end, and if it is ever stopped anywhere else (a hand in its way, a feint return called
off) it goes on to an end as soon as it can. There is no drifting back to the middle (it delayed
dodges, and in short hops looked jittery). A new dodge waits until the last move has finished,
and until the hand has come --hyst mm closer than it has been since (so a hand hovering between
the ends does not send the bowl back and forth); a hand over the opening always counts. A
dropped frame or two does not reset that; only half a second with no hand does.

Feints: a random --feint share of dodges go only to the middle of the rail, then dart straight
back to the end they came from, so a hand that has learned to follow it is caught out. It goes
back even if a hand is waiting over that end: a hand over the opening always triggers, so the
bowl dodges away again at once. Only a hand below the rim in its way calls the return off; the
bowl then goes on to an end instead. (Calling it off for a hand over the end as well made every
feint stop in the middle and go on in two steps.)
Hand points below the rim never trigger a dodge, and if one is in the bowl or in its path, the
bowl stays put: moving into a hand made it lose steps.
Shake: staying put let one hand held low pin the bowl while the other took the lid. So whenever
the bowl is holding (a low hand in its way, cornered at the end, a hand that has not come
closer) and a hand whose palm (wrist and knuckles) is at least --shake-below mm below the rim
(2 cm: a hand down beside the bowl, not one over it or resting on top with its fingertips over
the edge) has a point within --shake-near mm of the rim centre, it rattles: a --shake mm hop
towards the middle and straight back, again and again until the hand goes or the way clears.
Each way takes about 0.15 s; if bumping a hand ever loses steps, the marker check re-homes as usual.
(12 mm hops did little, and counting hands over the bowl made it shake far too often.)
Lost steps: whenever the bowl is still, the base marker's position is compared with the axis.
If they differ by more than --slip mm, it re-homes by itself (about 3 s) and goes on. (Not
G92: Marlin shifts its soft limits with it, so a stall point became a wall.) 800 mm/s at 8000 mm/s2 with the bowl lost steps on long moves
(2026-10-30); 700 at 7000 was the last clean bench speed with weight on.

Horns: a hand showing the horns (index and little finger up, middle and ring folded) for
--horns-hold s, seen by either camera on its own, stops the game: any feint return is dropped and the bowl slides to the middle of
the rail at the gentle --slide speed and stays there, so a sweet can be taken. Only a hand below
the rim in its way holds the slide up. It stays under the spell as long as the horns are up.
Once they drop (gone for half a second), the kids have --resume s to take a sweet, counted
down on the screen; horns shown again in that time put it back under the spell, and the count
starts afresh when they drop. When the time is up, dodging starts again whatever the hands are
doing (or at once on d), and the bowl goes on to an end as usual, unless a hand is still in it.
To stop it again after that, the horns must be dropped and shown again.

Face: the LED face (leds/src/main.cpp) is found on USB by itself (--face). It sleeps when no
hand has been near for 2 s, watches while one is (a 3D hand within --wake mm of the rim centre;
hands further off, or seen by one camera only, are ignored by the eyes and the tone too), snarls from each dodge, feint or shake
and while a hand is in the bowl, and cries while the horns have stopped the game. Once snarling
it keeps on while a hand stays within --trigger mm, and for --snarl-hold s after (3 s: in case
it comes back), so it does not flick between snarling and watching. The
eyes follow the hand nearest the bowl along the rail, from the rail's middle; --face-flip if
they look the wrong way. If dodge.py stops sending, the face falls asleep after 2 s.

Sound (scripts/sound.py) plays on the USB speaker (--sound none for none): a tone that follows
the hand, the skull's voice for each mood, a whoosh fitted to every move, the spell sting, the
music box over the last seconds of the --resume time, and the night around it. --sound-record
writes what it plays to a WAV. --sound-only plays only some of it, by group or single sound:
--sound-only background spot hand --no-dodge is the night and the hand tone with the bowl still,
and --sound-only voice musicbox the creature and the music box.

Keys: q quit, h re-home, d dodging on or off (and ends a horns stop). While it is off, the
horns do nothing either.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import serial
from serial.tools import list_ports

from bowl_calibrate import BOWL_FILE, FEED, best_marker, draw_markers, fit_markers, match_markers, save_bowl
from finders import Finders
from hand3d import PALM, Detector, Map, Tracker, draw_hand, ensure_model, is_horns, match_hands
from rim_calibrate import RIM_FILE, rail_frame
from sound import GROUPS, SOUNDS, Sound
from stereo import CALIB_FILE, HEIGHT, MARKERS_MM, WIDTH, Calibration, open_cameras, wait_for_frames
from y_speed import y_switch_triggered
from y_sweep import Marlin

YMIN, YMAX = 5.0, 215.0
STOPS = (5.0, 60.0, 110.0, 160.0, 215.0)  # where the rail is measured when dodge calibrates it itself
YMID = (YMIN + YMAX) / 2
LAG = 0.08  # s from sending a move to the bed starting (bench-log: measured 0.08 to 0.1 s over)
FORGET = 0.5  # s with no hand near before the hysteresis starts afresh (the tracker's forget time)


class Rail:
    """The rail and the rim in table coordinates, from the bowl and rim calibration files."""

    def __init__(self, cal: Calibration) -> None:
        self.load(cal)

    def load(self, cal: Calibration) -> None:
        """Read the bowl and rim calibration files (again, after dodge has calibrated by itself)."""
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
        up), mm: how far the printer has moved since the rail was calibrated."""
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
        self.hop = 0.0  # a shake: how far out (signed, mm) before coming straight back; 0 for a move

    @staticmethod
    def duration(dist: float, speed: float, accel: float) -> float:
        return (2 * (dist / accel) ** 0.5 if speed * speed / accel >= dist
                else 2 * speed / accel + (dist - speed * speed / accel) / speed)

    def move(self, target: float, speed: float, accel: float) -> None:
        target = min(max(target, YMIN), YMAX)
        now = time.monotonic()
        self.start, self.target = self.y(now), target
        self.speed, self.accel = speed, accel
        self.hop = 0.0
        self.dur = self.duration(abs(target - self.start), speed, accel)
        if self.m:
            self.m.send(f"M204 P{accel:g} T{accel:g}")
            self.m.send(f"G0 Y{target:.1f} F{speed * 60:g}")
        self.t0 = time.monotonic() + LAG

    def shake(self, hop: float, speed: float, accel: float) -> None:
        """A hop of `hop` mm (signed) and straight back, sent as one pair so the bed always ends
        where it started. Marlin stops dead at the turn: the two moves point opposite ways."""
        now = time.monotonic()
        y0 = self.y(now)
        out = min(max(y0 + hop, YMIN), YMAX)
        self.start = self.target = y0
        self.hop = out - y0
        self.speed, self.accel = speed, accel
        self.dur = 2 * self.duration(abs(self.hop), speed, accel)
        if self.m:
            self.m.send(f"M204 P{accel:g} T{accel:g}")
            self.m.send(f"G0 Y{out:.1f} F{speed * 60:g}")
            self.m.send(f"G0 Y{y0:.1f} F{speed * 60:g}")
        self.t0 = time.monotonic() + LAG

    def y(self, now: float) -> float:
        if self.hop:
            dist, leg = abs(self.hop), self.dur / 2
            t = min(max(now - self.t0, 0.0), self.dur)
            d = (travelled(dist, self.speed, self.accel, t) if t < leg
                 else dist - travelled(dist, self.speed, self.accel, t - leg))
            return self.start + np.sign(self.hop) * d
        dist = abs(self.target - self.start)
        if dist == 0:
            return self.target
        d = travelled(dist, self.speed, self.accel, min(now - self.t0, self.dur))
        return self.start + np.sign(self.target - self.start) * d

    def moving(self, now: float) -> bool:
        return now < self.t0 + self.dur

    def shaking(self, now: float) -> bool:
        return bool(self.hop) and self.moving(now)


FACE_VID = 0x2886  # Seeed: the XIAO RP2350 that drives the face LEDs
MOODS = {"s": "sleeping", "w": "watching", "n": "snarl", "c": "crying"}


class Face:
    """The LED face (leds/src/main.cpp) on its own USB serial port. Gets a line with the mood
    letter and where the hand is, -100 (the face's far left) to 100, up to 20 times a second.
    If the lines stop, the face goes to sleep by itself after 2 s."""

    def __init__(self, port: str) -> None:
        self.ser = serial.Serial(port, 115200, timeout=0, write_timeout=0.05)
        self.mood = ""
        self.sent = 0.0

    def set(self, mood: str, hand_x: float | None, now: float) -> None:
        if mood == self.mood and now - self.sent < 0.05:
            return
        line = mood + ("" if hand_x is None else str(round(float(np.clip(hand_x, -1, 1)) * 100)))
        try:
            self.ser.write((line + "\n").encode())
            self.ser.reset_input_buffer()  # it prints its mood changes; nobody reads them
        except serial.SerialException as e:
            print(f"face: {e}", flush=True)
        self.mood, self.sent = mood, now

    def close(self) -> None:
        try:
            self.ser.write(b"s\n")
        except serial.SerialException:
            pass
        self.ser.close()


def find_face() -> str | None:
    return next((p.device for p in list_ports.comports() if p.vid == FACE_VID), None)


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
    """Home Y, check the switch has let go, then wait at the home end of the rail."""
    m.send("M204 P500 T500")
    m.send("G28 Y", timeout=60, idle=show)
    m.send("G0 Y20 F3000")
    m.send("M400", timeout=60, idle=show)
    if y_switch_triggered(m):
        raise RuntimeError("the Y switch reads pressed at Y20, so the home was not real; check the switch")
    m.send(f"G0 Y{YMIN:g} F3000")
    m.send("M400", timeout=60, idle=show)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--port", help="the SKR's serial port, e.g. COM9; leave out for a dry run")
    ap.add_argument("--cams", type=int, nargs=2, default=[0, 1], help="camera numbers (2 is OBS)")
    ap.add_argument("--exposure", nargs="+", type=lambda v: None if v == "auto" else float(v),
                    help="lock exposure (e.g. -5), or one per camera (e.g. auto -5); default auto. "
                         "Longer than -5 (e.g. -4) camera 1 drops to 16 fps; camera 0 tops out dim on a locked "
                         "exposure in evening light but is bright on auto at a full 30 fps")
    ap.add_argument("--gain", type=float, nargs=2, default=[0, 32],
                    help="gain per camera; 32 on camera 1 matches its brightness to camera 0")
    ap.add_argument("--speed", type=float, default=700, help="dodge top speed, mm/s")
    ap.add_argument("--accel", type=float, default=7000, help="dodge acceleration, mm/s^2")
    ap.add_argument("--trigger", type=float, default=250, help="dodge when a hand point is this close, mm")
    ap.add_argument("--height", type=float, default=250, help="...and at most this high above the rim, mm")
    ap.add_argument("--feint", type=float, default=0.3, help="share of dodges that are feints, 0 to 1")
    ap.add_argument("--hyst", type=float, default=50, help="after a move, the hand must come this much closer, mm")
    ap.add_argument("--shake", type=float, default=25, help="hop of the shake when a near hand pins the bowl, mm; 0 for none")
    ap.add_argument("--shake-near", type=float, default=150, help="shake when a hand point is this close to the rim centre, mm")
    ap.add_argument("--shake-below", type=float, default=20,
                    help="shake only for a hand whose palm is at least this far below the rim, mm")
    ap.add_argument("--shake-speed", type=float, default=300, help="shake top speed, mm/s")
    ap.add_argument("--shake-accel", type=float, default=5000, help="shake acceleration, mm/s^2")
    ap.add_argument("--horns-hold", type=float, default=0.3, help="horns held this long stop the game, s")
    ap.add_argument("--resume", type=float, default=10.0, help="after a horns stop, time to take a sweet once the horns drop, s")
    ap.add_argument("--slide", type=float, default=150, help="speed of the slide to the middle, mm/s")
    ap.add_argument("--slide-accel", type=float, default=1000, help="acceleration of that slide, mm/s^2")
    ap.add_argument("--face", default="auto", help="the LED face's serial port; auto finds it, none for no face")
    ap.add_argument("--wake", type=float, default=450,
                    help="the face wakes for a hand point this close to the rim centre (3D), mm")
    ap.add_argument("--snarl-hold", type=float, default=3.0,
                    help="the face snarls this long after a dodge, or after a hand leaves the bowl or goes, s")
    ap.add_argument("--face-flip", action="store_true", help="the eyes look the wrong way: mirror them")
    ap.add_argument("--sound", default="auto", help="auto plays on the USB speaker, none for no sound")
    ap.add_argument("--sound-device", help="the speaker: device number or part of its name (sound.py --devices)")
    ap.add_argument("--volume", type=float, default=1.0, help="sound volume, 0 to 1")
    ap.add_argument("--sound-record", metavar="WAV", help="also write everything the sound plays to this file")
    ap.add_argument("--sound-only", nargs="+", choices=list(GROUPS) + list(SOUNDS), metavar="NAME",
                    help=f"play only these groups ({' '.join(GROUPS)}; hand is the hand tone) or single "
                    "sounds (sound.py's SOUNDS), e.g. background spot hand: the night and the hand tone; "
                    "voice musicbox: the creature and the music box")
    ap.add_argument("--no-dodge", action="store_true",
                    help="start with dodging off, so the bowl stays put (d turns it on)")
    ap.add_argument("--slip", type=float, default=6.0, help="marker vs axis difference that means lost steps, mm")
    ap.add_argument("--max-err", type=float, default=20.0, help="reject hand pairs above this px error")
    ap.add_argument("--threads", type=int, default=12, help="hand finders running at once (finders.py)")
    ap.add_argument("--sweep", type=int, default=8, help="most sweep tiles per camera per frame pair (finders.py)")
    ap.add_argument("--calibrate", action="store_true",
                    help="calibrate the rail at start-up, as bowl_calibrate.py does (it also does so by itself "
                         "whenever it finds the printer moved sideways)")
    ap.add_argument("--plain", action="store_true",
                    help="one hand finder per camera on the whole frame, as before finders.py")
    ap.add_argument("--spread", action="store_true", help="Y driver in spreadCycle: louder, more torque at speed")
    ap.add_argument("--misses", action="store_true",
                    help="save both cameras' pictures to runs/misses when a hand is seen by one camera only")
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
    # Centred on the middle of the rail: the board's spot can be far off since the cameras moved.
    table_map = Map(cal, 400.0, None, centre=rail.rim_centre(YMID))
    ensure_model()

    m = Marlin(a.port) if a.port else None
    face_port = None if a.face == "none" else find_face() if a.face == "auto" else a.face
    face = Face(face_port) if face_port else None
    print(f"face on {face_port}" if face else "no face", flush=True)
    snd = Sound.open(a.sound, a.sound_device, a.volume, a.sound_record, a.sound_only)
    cams, dets, finders = [], [], None
    frames, run_t0 = 0, time.monotonic()
    seen_by = {k: 0 for k in ("both", "cam0 only", "cam1 only", "rejected", "none")}  # frames, for the end
    cv2.namedWindow("dodge", cv2.WINDOW_NORMAL)  # resizable, to fit the screen
    cv2.resizeWindow("dodge", WIDTH * 3 // 4, (70 + HEIGHT // 2 + 420) * 3 // 4)
    try:
        exposure = (a.exposure or [None]) * (2 if a.exposure is None or len(a.exposure) == 1 else 1)
        cams = open_cameras(a.cams, exposure, a.gain)
        wait_for_frames(cams)
        if a.plain:
            dets = [Detector(c, 2) for c in cams]
        else:
            # Hands are looked for along a line of sight only this close to the bowl's path.
            half = (YMAX - YMIN) / 2 * rail.scale

            def zone(P: np.ndarray) -> np.ndarray:
                rp = rail.to_rail(P, YMID)
                along = np.maximum(np.abs(rp[:, 0]) - half, 0)
                up = np.maximum(np.maximum(-100 - rp[:, 2], rp[:, 2] - a.height), 0)
                return np.sqrt(along ** 2 + rp[:, 1] ** 2 + up ** 2)

            finders = Finders(cams, cal, zone, a.trigger + 100, a.threads, a.sweep)
        tracker = Tracker(0.5)
        axis = Axis(m, YMIN)
        state = {"text": "starting", "banner": None, "enabled": not a.no_dodge, "stopped": False, "log": [],
                 "missed": [[0, 0.0], [0, 0.0]]}  # per camera: times and seconds it missed a hand the other had
        last_seq = [0, 0]
        still_since = time.monotonic()
        dry_placed = False
        slips = 0
        ref = a.trigger + a.hyst
        hand_seen = 0.0  # when a hand was last in reach height
        last_hold = None
        pending: float | None = None  # where a feint goes back to once its first move ends
        horns_from: float | None = None  # when the horns were first seen, while they are held
        horns_last = 0.0
        horns_spent = False  # these horns have stopped it once; they must be dropped first
        free_until: float | None = None  # when a horns stop ends, once the bowl is in the middle
        snarl_until = 0.0  # the face snarls until then, after a dodge
        hand_near = 0.0  # when a hand was last within --wake mm of the rim centre
        rng = np.random.default_rng()
        # A hand that gets no 3D position is shown on the status bar, and with --misses saved
        # when it lasts.
        miss_from: float | None = None
        last_kind, last_now = "none", 0.0
        miss_saved = 0.0
        miss_dir = Path(__file__).resolve().parent.parent / "runs" / "misses"

        def say(text: str) -> None:
            print(text, flush=True)
            state["log"] = (state["log"] + [text])[-5:]

        def frame_view() -> tuple | None:
            """Newest results from both cameras, or None if nothing new."""
            got = finders.latest() if finders else [d.latest() for d in dets]
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
            # HORNS on every hand that counts as horns: on that camera's own picture, or in 3D.
            # Shown in every phase, so you can find the directions that work.
            horns3d = [(i0, i1) for _, (pts, _, i0, i1) in current if is_horns(pts)]
            for k, (v, hs) in enumerate(((v0, hands0), (v1, hands1))):
                for i, h in enumerate(hs):
                    if is_horns(h) or any(pair[k] == i for pair in horns3d):
                        x, y = int(h[:, 0].min()), max(int(h[:, 1].min()) - 15, 40)
                        (tw, th), _ = cv2.getTextSize("HORNS", cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
                        cv2.rectangle(v, (x - 6, y - th - 10), (x + tw + 6, y + 10), (200, 0, 200), -1)
                        cv2.putText(v, "HORNS", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4,
                                    cv2.LINE_AA)
            # Debug counter: how often this camera missed a hand that the other one found.
            for v, (times, secs) in zip((v0, v1), state["missed"]):
                label = f"missed a hand {times}x, {secs:.1f} s"
                for colour, thick in (((0, 0, 0), 8), ((0, 255, 255) if times else (255, 255, 255), 3)):
                    cv2.putText(v, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, thick, cv2.LINE_AA)
            view =np.hstack([cv2.resize(v0, (WIDTH // 2, HEIGHT // 2)), cv2.resize(v1, (WIDTH // 2, HEIGHT // 2))])
            # Status bar above the pictures, nothing written over them: the state, big, coloured
            # by what the bowl is doing, and one small line of numbers.
            text = state["banner"] or (("DRY RUN: " if m is None else "") + state["text"])
            upper = text.upper()
            bg = ((0, 0, 170) if state["banner"] else (130, 0, 130) if state["stopped"]
                  else (0, 0, 150) if axis.moving(now)
                  else (0, 110, 160) if upper.startswith("HOLDING") or "HOM" in upper else (40, 40, 40))
            bar = np.full((70, WIDTH, 3), bg, np.uint8)
            cv2.putText(bar, text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            info = f"bowl Y {y:.0f}"
            if near_pts:
                tr, rp = min(near_pts, key=lambda n: np.hypot(n[1][0], n[1][1]))
                info += f"     hand {np.hypot(rp[0], rp[1]):.0f} mm from bowl, {rp[2]:+.0f} mm above rim"
            else:
                info += "     no hand"
            info += f"     {state.get('seen', '')}"
            if finders:
                info += f"     {finders.stats}"
            if horns_from is not None:
                info += f"     horns {now - horns_from:.1f} s"
            if face:
                info += f"     face {MOODS.get(face.mood, 'off')}"
            cv2.putText(bar, info, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
            view = np.vstack([bar, view])
            seen = {tr.number for tr, _ in current}
            table_map.update(now, tracker, seen)
            mp = table_map.draw(now, tracker, seen, marks)
            for r, thick in ((rail.radius, 3), (a.trigger, 1)):
                cv2.circle(mp, table_map.px(rail.rim_centre(y)), int(r * table_map.scale), colour, thick)
            show(np.vstack([view, mp]))

        def show(img: np.ndarray) -> None:
            """Fit the picture to the window here, keeping its shape. Left to itself, Windows
            stretches it to the window's shape and shrinks it without smoothing (jagged text)."""
            try:
                _, _, w, h = cv2.getWindowImageRect("dodge")
            except cv2.error:
                w = h = 0
            if w > 0 and h > 0:  # 0 when minimised
                s = min(w / img.shape[1], h / img.shape[0])
                fit = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                                 interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                canvas = np.zeros((h, w, 3), np.uint8)
                y0, x0 = (h - fit.shape[0]) // 2, (w - fit.shape[1]) // 2
                canvas[y0:y0 + fit.shape[0], x0:x0 + fit.shape[1]] = fit
                img = canvas
            cv2.imshow("dodge", img)

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
                if m:
                    say("printer moved sideways since calibration")
                    recalibrate()
                else:
                    state["banner"] = "printer moved sideways since calibration: run bowl_calibrate.py"
                    say(state["banner"])

        def settle(seconds: float) -> None:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                idle_show()

        def recalibrate() -> None:
            """The printer moved sideways: calibrate the rail as bowl_calibrate.py does, without
            leaving dodge. The bed goes slowly through the stops, the base marker is measured in
            3D at each, and the fitted rail is checked (travel, straightness, marker size), saved
            and used. If the check fails, nothing is saved and dodging stays off with the reason
            on the banner. Starts and ends at the home end, Y5."""
            say("calibrating the rail")
            found: dict[int, list[tuple[float, np.ndarray, float]]] = {}
            for i, stop in enumerate(STOPS):
                state["text"] = f"calibrating the rail: stop {i + 1} of {len(STOPS)}, Y{stop:g}"
                m.send(f"G0 Y{stop:g} F{FEED}")
                m.send("M400", timeout=60, idle=idle_show)
                settle(0.7)
                per_id: dict[int, list[np.ndarray]] = {}
                errs: dict[int, list[float]] = {}
                pairs, end = 0, time.monotonic() + 10
                while pairs < 15 and time.monotonic() < end:
                    got = frame_view()
                    if got is None:
                        cv2.waitKey(1)
                        continue
                    pairs += 1
                    mk0, mk1 = got[0][0][3], got[1][0][3]
                    draw(got, time.monotonic(), [], match_markers(cal, mk0, mk1), [])
                    cv2.waitKey(1)
                    for mid in mk0.keys() & mk1.keys():
                        X = cal.triangulate(mk0[mid], mk1[mid])  # camera 0 frame, as bowl.npz keeps it
                        per_id.setdefault(mid, []).append(X)
                        errs.setdefault(mid, []).append(cal.reprojection_error(X, mk0[mid], mk1[mid]))
                for mid, xs in per_id.items():
                    if len(xs) >= pairs / 2:
                        found.setdefault(mid, []).append((stop, np.median(np.array(xs), axis=0),
                                                          float(np.median(errs[mid]))))
                say(f"Y{stop:g}: " + (", ".join(f"marker {mid}" for mid in sorted(per_id)) or "no marker"))
            m.send(f"G0 Y{YMIN:g} F{FEED}")
            m.send("M400", timeout=60, idle=idle_show)
            results, lines = fit_markers(cal, found, MARKERS_MM, len(STOPS))
            for line in lines:
                say(line)
            best = best_marker(results, MARKERS_MM) if results else None
            if best is None or not results[best]["ok"]:
                state["banner"] = ("rail calibration OFF: check nothing holds the bed back, "
                                   "then press h to try again")
                say(state["banner"])
                return
            save_bowl(best, results[best])
            rail.load(cal)
            state["calibrated"] = True
            say(f"rail calibrated on marker {best} and saved to calib/bowl.npz")

        def idle_show() -> None:
            got = frame_view()
            if got is not None:
                now = time.monotonic()
                marks = match_markers(cal, got[0][0][3], got[1][0][3])
                draw(got, now, [], marks, [])
            cv2.waitKey(1)

        def rehome(text: str) -> Axis:
            """Home, re-anchor the rail on the marker and wait at the home end. The banner is
            cleared first, so anchor() puts it back if the printer is still out of place."""
            state["text"] = text
            state["banner"] = None
            if snd and text == "re-homing":  # not the first home at start-up
                snd.event("rehome")
            home(m, idle_show)
            anchor(YMIN)
            return Axis(m, YMIN)

        if m:
            m.send("G21")
            m.send("G90")
            m.send(f"M569 S{0 if a.spread else 1} Y")  # stealthChop (quiet) unless --spread
            m.send(f"M203 Y{a.speed:g}")
            m.send(f"M201 Y{a.accel:g}")
            axis = rehome("homing Y")
            if a.calibrate and not state["banner"] and not state.get("calibrated"):  # not twice
                recalibrate()
        say(f"ready: dodge at {a.speed:g} mm/s, {a.accel:g} mm/s2, trigger {a.trigger:g} mm")
        state["text"] = "watching"
        if snd:
            snd.ready(time.monotonic())
        run_t0 = time.monotonic()

        while True:
            got = frame_view()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                if state["stopped"]:
                    state["stopped"] = False
                    state["enabled"] = True
                    if snd:
                        snd.event("again")
                else:
                    state["enabled"] = not state["enabled"]
                say("dodging " + ("on" if state["enabled"] else "off"))
            if key == ord("h") and m and not axis.moving(time.monotonic()):
                axis = rehome("re-homing")
                pending = None
                say("re-homed")
                continue  # the frame in hand is from before the home
            if got is None:
                continue
            now = time.monotonic()
            frames += 1
            (f0, s0, hands0, mk0), (f1, s1, hands1, mk1) = got[0][0], got[1][0]
            found = match_hands(cal, hands0, hands1, a.max_err)
            current = tracker.update(now, found)

            # A hand left without a partner: the other camera did not find it, or both did and the
            # pair was rejected (its error is above --max-err).
            left0 = [h for i, h in enumerate(hands0) if i not in {f[2] for f in found}]
            left1 = [h for i, h in enumerate(hands1) if i not in {f[3] for f in found}]
            kind = ("rejected" if left0 and left1 else "cam0 only" if left0 else "cam1 only" if left1
                    else "both" if found else "none")
            seen_by[kind] += 1
            for k, other_only in ((0, "cam1 only"), (1, "cam0 only")):
                if kind == other_only:
                    state["missed"][k][0] += last_kind != kind
                    if last_kind == kind:
                        state["missed"][k][1] += now - last_now
            last_kind, last_now = kind, now
            state["seen"] = f"cams {len(hands0)}/{len(hands1)}"
            if kind == "rejected":
                err = min(cal.reprojection_error(cal.triangulate(h0, h1), h0, h1) for h0 in left0 for h1 in left1)
                state["seen"] += f" rejected {err:.0f} px"
            elif kind.endswith("only"):
                state["seen"] += f" {kind}"
            if kind in ("both", "none"):
                miss_from = None
            else:
                miss_from = now if miss_from is None else miss_from
                if a.misses and now - miss_from >= 0.3 and now - miss_saved >= 1.0:
                    miss_saved = now
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    miss_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(miss_dir / f"{stamp}_cam0.jpg"), f0)
                    cv2.imwrite(str(miss_dir / f"{stamp}_cam1.jpg"), f1)
                    size = lambda hs: " ".join(f"{np.ptp(h[:, 0]):.0f}x{np.ptp(h[:, 1]):.0f}" for h in hs) or "-"
                    print(f"miss {stamp}: {state['seen']}; hand boxes px cam0 {size(hands0)}, cam1 {size(hands1)}",
                          flush=True)
            marks = match_markers(cal, mk0, mk1)
            y = axis.y(now)

            # Every hand point in rail coordinates around the rim centre: along, across, above the rim.
            hand_rps = [(tr, rail.to_rail(pts, y)) for tr, (pts, _, _, _) in current]
            near_pts = [(tr, rp[np.argmin(np.hypot(rp[:, 0], rp[:, 1]))]) for tr, rp in hand_rps]
            # Only points above the rim count as reaching. Below it a hand is beside or in the bowl,
            # and moving would hit it (that is what made it lose steps).
            reach = [p for _, rp in hand_rps for p in rp if 0 <= p[2] < a.height]
            dist = min((float(np.hypot(p[0], p[1])) for p in reach), default=np.inf)
            low = np.array([p for _, rp in hand_rps for p in rp if p[2] < 0]).reshape(-1, 3)

            # Horns held for --horns-hold s stop the game. A dropped frame or two does not restart
            # the count. Once stopped, it lasts --resume s from the bowl reaching the middle. One
            # camera seeing them on its own picture is enough; so is the 3D hand.
            if (any(is_horns(h) for h in hands0 + hands1)
                    or any(is_horns(pts) for _, (pts, _, _, _) in current)):
                horns_last = now
                horns_from = now if horns_from is None else horns_from
            elif horns_from is not None and now - horns_last > 0.15:
                horns_from = None
                horns_spent = False
            # With dodging off (d, --no-dodge) the bowl stays put: the horns do nothing either.
            if (not state["stopped"] and state["enabled"] and not horns_spent and horns_from is not None
                    and now - horns_from >= a.horns_hold):
                state["stopped"] = True
                horns_spent = True
                free_until = None
                pending = None
                say("horns: stopped, sliding to the middle")
                if snd:
                    snd.event("horns")
            elif state["stopped"] and free_until is not None and now > free_until:
                state["stopped"] = False
                say(f"{a.resume:g} s up: dodging again")
                if snd:
                    snd.event("again")

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
                    axis = rehome("re-homing")
                    pending = None
                    state["text"] = "watching"
                    continue  # the hands and the bowl's place in this frame are from before the home

            # Hysteresis: after a move, the hand must come --hyst mm closer than it has been since
            # (ref follows the hand's furthest distance) before the next dodge. A hand hovering
            # between the ends no longer sends the bowl back and forth. A dropped frame keeps ref;
            # only a hand gone for FORGET s lets the next one count at once.
            if np.isfinite(dist):
                hand_seen = now
                # A shake is not a getaway: it leaves ref alone, or a hand creeping in during a
                # long rattle would never count as coming closer.
                ref = (dist if moving and not axis.shaking(now)
                       else min(a.trigger + a.hyst, max(ref, dist)))
            elif now - hand_seen > FORGET:
                ref = a.trigger + a.hyst
            # A hand over the opening always counts; otherwise a hand that followed the bowl to
            # its new spot could never come the extra --hyst mm closer.
            reaching = dist < a.trigger and dist < max(ref - a.hyst, rail.radius)
            # Only the two ends are places to stay. Stopped anywhere else (a feint whose return
            # was called off, a hand in the way), the bowl goes on to an end as soon as it can.
            between = min(y - YMIN, YMAX - y) > 5

            # Why the bowl is not moving although a hand is near, logged when it changes.
            hold = None
            if not moving and not reaching and not between and not state["stopped"]:
                if dist < a.trigger:
                    hold = (f"holding: hand still at {dist:.0f} mm, must come within "
                            f"{max(ref - a.hyst, rail.radius):.0f} mm")
                elif path_blocked(low, 0.0, rail.radius):
                    hold = "holding: hand below the rim at the bowl"

            if state["stopped"]:
                # Horns: slide to the middle and stay there, whatever the hands do, unless a low
                # hand is in the way.
                # It stays under the spell while the horns are up; the --resume s count only
                # starts once they have been gone for half a second, and horns held up again stop
                # the count until they drop again.
                if moving or abs(y - YMID) <= 2:
                    if not moving:
                        horns_up = horns_from is not None and now - horns_from >= a.horns_hold
                        if horns_up and free_until is not None:
                            free_until = None
                            say("horns again: under the spell until they drop")
                            if snd:
                                snd.post("cancel", tag="countdown", fade=0.6)
                        if free_until is None and now - horns_last > 0.5:
                            free_until = now + a.resume
                            say(f"horns down: {a.resume:g} s to take a sweet")
                            if snd:
                                snd.event("free", free_until)  # the music box ends with the time
                        state["text"] = ("STOPPED (horns): take a sweet! Dodging again in "
                                         f"{np.ceil(free_until - now):.0f} s" if free_until is not None
                                         else f"UNDER THE SPELL (horns): take a sweet! {a.resume:g} s more once the horns drop")
                elif path_blocked(low, (YMID - y) * rail.scale, rail.radius):
                    hold = "holding: hand below the rim in the way to the middle"
                elif not state["banner"]:
                    axis.move(YMID, a.slide, a.slide_accel)
                    state["text"] = f"HORNS: sliding to Y{YMID:.0f}"
                    if snd:
                        snd.move("slide", axis)
            elif pending is not None and not moving:
                # Second half of a feint: dart back, even onto a hand waiting there (the bowl dodges
                # it again at once), unless a low hand is now in the way. Called off, the bowl is
                # between the ends and goes on to one.
                if path_blocked(low, (pending - y) * rail.scale, rail.radius) or state["banner"]:
                    say("feint: return cancelled, hand in the way")
                else:
                    axis.move(pending, a.speed, a.accel)
                    snarl_until = now + a.snarl_hold
                    state["text"] = f"FEINT back to Y{pending:.0f}"
                    moving = True
                    if snd:
                        snd.move("back", axis)
                pending = None
            elif (reaching or between) and not moving and state["enabled"] and not state["banner"]:
                p = min(reach, key=lambda q: np.hypot(q[0], q[1])) if reach else None
                target = (dodge_target(y, p[0]) if p is not None
                          else YMIN if y - YMIN < YMAX - y else YMAX)
                if abs(target - y) <= 5:
                    hold = "holding: cornered at the end of the rail"
                    if snd and hold != last_hold:
                        snd.event("cornered")
                elif path_blocked(low, (target - y) * rail.scale, rail.radius):
                    hold = "holding: hand below the rim in the bowl's path"
                elif not reaching:
                    axis.move(target, a.speed, a.accel)
                    state["text"] = f"ON TO Y{target:.0f}"
                    say(f"on to an end: stopped at Y{y:.0f}, going to Y{target:.0f}")
                    if snd:
                        snd.move("on", axis)
                else:
                    seen_at = max(s0, s1)  # when the newer of the two frames was taken
                    kind = "dodge"
                    if not between and rng.random() < a.feint:
                        # Feint: to the middle of the rail, then back to the end it came from.
                        pending = YMIN if target > y else YMAX
                        target = YMID
                        kind = "feint"
                    axis.move(target, a.speed, a.accel)
                    snarl_until = now + a.snarl_hold
                    state["text"] = f"{kind.upper()} to Y{target:.0f}"
                    if snd:
                        snd.move(kind, axis)
                    say(f"{kind}: hand {dist:.0f} mm away, {p[2]:+.0f} mm above rim; Y{y:.0f} to Y{target:.0f}"
                        + (f" then back to Y{pending:.0f}" if kind == "feint" else "")
                        + f"; frame to command {1000 * (time.monotonic() - seen_at):.0f} ms")
            elif not moving and not reaching:
                state["text"] = "watching" if state["enabled"] else "dodging off"

            # Shake: the bowl is holding (a hand below the rim in its way, cornered at the end, a
            # hand at the bowl that has not come closer) but a hand is close. It rattles a short
            # hop in and straight back, again and again, so one hand held low cannot pin it while
            # the other takes the lid. Always towards the middle: at an end that is the only way.
            if axis.shaking(now):
                hold = last_hold  # mid-shake: still the same hold
            elif (hold and not moving and a.shake > 0 and state["enabled"] and not state["stopped"]
                    and not state["banner"]):
                # Only a hand down beside the bowl pins it; one over the opening is a reach. The
                # palm must be down, not just a point: a hand resting on top of the bowl has its
                # fingertips over the edge, below the rim, and that shook it (2026-11-06).
                low_hands = [(float(np.hypot(rp[:, 0], rp[:, 1]).min()), float(rp[PALM, 2].mean()))
                             for _, rp in hand_rps if rp[PALM, 2].mean() < -a.shake_below]
                near, palm_z = min(low_hands, default=(np.inf, 0.0))
                if near < a.shake_near:
                    axis.shake(a.shake if y < YMID else -a.shake, a.shake_speed, a.shake_accel)
                    moving = True
                    snarl_until = max(snarl_until, now + a.snarl_hold)
                    hold = ("shaking: " + hold.removeprefix("holding: ")
                            + f" (palm {palm_z:+.0f} mm, hand {near:.0f} mm from rim centre)")
                    if snd and not (last_hold or "").startswith("shaking"):
                        snd.event("cornered")
            if hold:
                state["text"] = hold
            if (hold or "").split(" at ")[0] != (last_hold or "").split(" at ")[0]:
                if hold:
                    say(hold)
                last_hold = hold

            # The face: crying while it has given up (horns), snarling during and for a while after
            # a dodge or while a hand is in the bowl, watching while a hand is near, else asleep.
            # The eyes follow the nearest hand along the rail, seen from the face at the rail's
            # middle. The sound gets the same mood and the palms.
            # Near: a 3D hand (seen by both cameras) with a point within --wake mm of the rim
            # centre. Any hand in either picture woke it before, even hands and feet across the
            # room (2026-11-06). Hands further off are ignored by the eyes and the tone too.
            near_hands = [tr for tr, rp in hand_rps if np.linalg.norm(rp, axis=1).min() < a.wake]
            if near_hands:
                hand_near = now
            in_bowl = path_blocked(low, 0.0, rail.radius)
            # Snarling lasts while a hand is in the bowl or still near, and --snarl-hold s after:
            # a hand pulled back for a moment often comes again, and the face flicking between
            # snarling and watching looked nervous, not fierce.
            if in_bowl or (now < snarl_until and dist < a.trigger):
                snarl_until = max(snarl_until, now + a.snarl_hold)
            mood = ("c" if state["stopped"] else "n" if now < snarl_until
                    else "w" if now - hand_near < 2.0 else "s")
            palms = rail.to_rail(np.array([tr.palm for tr in near_hands]), y) if near_hands else None
            if face:
                hand_x = None
                if palms is not None:
                    near = palms[np.argmin(np.hypot(palms[:, 0], palms[:, 1])), 0]  # along the rail from the bowl
                    hand_x = (near + (y - YMID) * rail.scale) / 200 * (-1 if a.face_flip else 1)
                face.set(mood, hand_x, now)
            if snd:
                snd.frame(now, mood, palms, in_bowl)

            draw(got, now, current, marks, near_pts)
    except KeyboardInterrupt:
        if m:
            m.ser.write(b"M410\n")
        print("stopped")
    finally:
        total = sum(seen_by.values()) - seen_by["none"]
        if total:
            print("frames with a hand: " + ", ".join(f"{k} {100 * n / total:.0f}%"
                                                     for k, n in seen_by.items() if k != "none"), flush=True)
        if frames:
            secs = time.monotonic() - run_t0
            print(f"{frames} frames in {secs:.0f} s: {frames / secs:.1f} a second", flush=True)
        if snd:
            snd.close()
        for d in dets:
            d.close()
        if finders:
            finders.close()
        for c in cams:
            c.close()
        if face:
            face.close()
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
