"""Find how fast the Y axis can sweep without losing steps.

    uv run python scripts/y_speed.py --port COM9
    uv run python scripts/y_speed.py --port COM9 --steps 100:1000 200:2000 300:3000

For each speed:accel step (mm/s : mm/s^2) it homes Y, then sweeps the bed end to end and
times each move. After every return to the near end the Y switch must be open; if it is
pressed, steps were lost and the run stops at once (re-homing first). After each step it
also measures smaller losses: the bed creeps back toward the Y switch 0.1 mm at a time until
the switch clicks (M119). The first check is the reference; if a later one differs by more
than --max-drift mm the run stops.

A slip during a move can still bump the bed into the end once; nothing here can stop a move
that is already running. Keep the jumps between steps small.

All settings are for this run only (nothing is saved with M500); M501 restores the saved ones
at the end. --spread switches the Y driver from stealthChop to spreadCycle (louder, more
torque at speed).
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from y_sweep import Marlin

CREEP_FEED = 300  # mm/min, for the slow moves near the switch


def y_switch_triggered(m: Marlin) -> bool:
    for line in m.send("M119"):
        if line.lower().startswith("y_min"):
            return "triggered" in line.lower()
    raise RuntimeError("M119 did not report y_min")


def switch_point(m: Marlin, start: float = 2.0, step: float = 0.1) -> float:
    """Creep from Y=start toward Y=0 (and a little past it) until the switch clicks."""
    m.send(f"G0 Y{start:g} F{CREEP_FEED}")
    m.send("M400")
    if y_switch_triggered(m):
        raise RuntimeError(f"the Y switch is already pressed at Y{start:g}: large drift")
    m.send("M211 S0")  # let the creep go a little below 0
    try:
        y = start
        while y > -3.0:
            y = round(y - step, 3)
            m.send(f"G0 Y{y:g} F{CREEP_FEED}")
            m.send("M400")
            if y_switch_triggered(m):
                return y
    finally:
        m.send(f"G0 Y{start:g} F{CREEP_FEED}")
        m.send("M400")
        m.send("M211 S1")
    raise RuntimeError("the Y switch never clicked")


def move_time(distance: float, speed: float, accel: float) -> float:
    """Time for one move from rest to rest with a trapezoid (or triangle) speed profile."""
    ramp = speed * speed / accel  # distance spent speeding up plus slowing down
    if ramp >= distance:
        return 2 * math.sqrt(distance / accel)
    return 2 * speed / accel + (distance - ramp) / speed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument(
        "--steps",
        nargs="+",
        default=["100:1000", "150:1500", "200:2000", "250:2500", "300:3000"],
        help="speed:accel pairs, mm/s:mm/s^2",
    )
    p.add_argument("--ymin", type=float, default=5)
    p.add_argument("--ymax", type=float, default=215)
    p.add_argument("--cycles", type=int, default=2, help="there-and-back sweeps per step")
    p.add_argument("--max-drift", type=float, default=0.3, help="mm (default 0.3)")
    p.add_argument("--spread", action="store_true", help="Y driver in spreadCycle")
    a = p.parse_args()

    steps = [tuple(float(v) for v in s.split(":")) for s in a.steps]
    distance = a.ymax - a.ymin

    m = Marlin(a.port)
    try:
        m.send("G21")
        m.send("G90")
        m.send(f"M569 S{0 if a.spread else 1} Y")
        top_speed = max(s for s, _ in steps)
        top_accel = max(acc for _, acc in steps)
        m.send(f"M203 Y{top_speed:g}")
        m.send(f"M201 Y{top_accel:g}")

        ref = None
        for speed, accel in steps:
            # Home before every step, so a step always starts from a known position.
            m.send("M204 P500 T500")
            m.send("G28 Y", timeout=60)
            if ref is None:
                ref = switch_point(m)
                print(f"reference: switch clicks at Y{ref:.1f}\n")
            m.send(f"G0 Y{a.ymin:g} F{CREEP_FEED * 10}")
            m.send("M400")
            m.send(f"M204 P{accel:g} T{accel:g}")
            expect = move_time(distance, speed, accel)
            times = []
            slipped = False
            for _ in range(a.cycles):
                for target in (a.ymax, a.ymin):
                    t0 = time.monotonic()
                    m.send(f"G0 Y{target:g} F{speed * 60:g}")
                    m.send("M400", timeout=60)
                    times.append(time.monotonic() - t0)
                # Back at the near end: the switch must be open. If it is pressed, steps
                # were lost and the bed is past where the board thinks; stop now.
                if y_switch_triggered(m):
                    slipped = True
                    break
            m.send("M204 P500 T500")
            if slipped:
                print(
                    f"{speed:g} mm/s, {accel:g} mm/s2: the Y switch was pressed at "
                    f"Y{a.ymin:g}: lost steps, stopping here"
                )
                m.send("G28 Y", timeout=60)
                break
            point = switch_point(m)
            drift = point - ref
            print(
                f"{speed:g} mm/s, {accel:g} mm/s2: {distance:g} mm in "
                f"{min(times):.2f} to {max(times):.2f} s (sums say {expect:.2f} s), "
                f"drift {drift:+.1f} mm"
            )
            if abs(drift) > a.max_drift:
                print("lost steps: stopping here")
                m.send("G28 Y", timeout=60)
                break

        m.send(f"G0 Y{a.ymin:g} F{CREEP_FEED * 10}")
        m.send("M400")
    except KeyboardInterrupt:
        m.ser.write(b"M410\n")
        print("stopped")
        return 1
    except RuntimeError as e:
        print(f"{e}: re-homing")
        m.send("M204 P500 T500")
        m.send("G28 Y", timeout=60)
        return 1
    finally:
        try:
            m.send("M501")  # back to the saved settings
        finally:
            m.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
