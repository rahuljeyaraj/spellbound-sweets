"""Home the Y axis, then sweep the bed end to end and time each move.

    uv run python scripts/y_sweep.py                      # list serial ports
    uv run python scripts/y_sweep.py --port COM9          # gentle sweep, 3 times
    uv run python scripts/y_sweep.py --port COM9 --speed 150 --accel 2000

Talks plain G-code to Marlin on the SKR Mini E3 V3 (native USB, 115200 baud). Only Y moves;
X and Z are never homed or moved. --speed is mm/s and --accel is mm/s^2; both are set on the
board with M203 / M201 / M204 for this run only (nothing is saved with M500).
"""

from __future__ import annotations

import argparse
import sys
import time

import serial
from serial.tools import list_ports


class Marlin:
    def __init__(self, port: str, baud: int = 115200) -> None:
        self.ser = serial.Serial(port, baud, timeout=0.1, write_timeout=2)
        self._handshake()

    def _handshake(self, timeout: float = 10.0) -> None:
        # The SKR's native USB does not reset on open, so ask until it answers.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.ser.reset_input_buffer()
            try:
                self.send("M110 N0", timeout=1.5)
                return
            except TimeoutError:
                print("no answer yet, retrying")
        raise TimeoutError("the board did not answer")

    def send(self, cmd: str, timeout: float = 30.0) -> list[str]:
        """Send one line and return what the board printed before its ok."""
        self.ser.write((cmd + "\n").encode("ascii"))
        lines: list[str] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            text = raw.decode("ascii", errors="replace").strip()
            if text.startswith("ok"):
                return lines
            if text.startswith("echo:busy"):
                deadline = time.monotonic() + timeout
                continue
            if text.lower().startswith("error"):
                raise RuntimeError(f"{cmd}: {text}")
            lines.append(text)
        raise TimeoutError(f"no ok for {cmd}")

    def close(self) -> None:
        self.ser.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--port", help="serial port, e.g. COM9 (leave out to list ports)")
    p.add_argument("--speed", type=float, default=50, help="top speed, mm/s (default 50)")
    p.add_argument("--accel", type=float, default=500, help="acceleration, mm/s^2 (default 500)")
    p.add_argument("--ymin", type=float, default=5, help="near end of the sweep, mm (default 5)")
    p.add_argument("--ymax", type=float, default=215, help="far end of the sweep, mm (default 215)")
    p.add_argument("--cycles", type=int, default=3, help="there-and-back sweeps (default 3)")
    a = p.parse_args()

    if not a.port:
        for port in list_ports.comports():
            print(f"{port.device}  {port.description}  {port.hwid}")
        return 0

    m = Marlin(a.port)
    try:
        for line in m.send("M115"):
            if line.startswith("FIRMWARE_NAME"):
                print(line)
        m.send("M211")  # soft endstops report
        m.send("G21")
        m.send("G90")
        m.send(f"M203 Y{a.speed:g}")
        m.send(f"M201 Y{a.accel:g}")
        m.send(f"M204 P{a.accel:g} T{a.accel:g}")

        print("homing Y")
        m.send("G28 Y", timeout=60)
        m.send(f"G0 Y{a.ymin:g} F{a.speed * 60:g}")
        m.send("M400", timeout=60)

        feed = a.speed * 60
        distance = a.ymax - a.ymin
        for i in range(a.cycles):
            for target in (a.ymax, a.ymin):
                t0 = time.monotonic()
                m.send(f"G0 Y{target:g} F{feed:g}")
                m.send("M400", timeout=60)
                dt = time.monotonic() - t0
                print(f"cycle {i + 1}: to Y{target:g}, {distance:g} mm in {dt:.2f} s")

        for line in m.send("M114"):
            print(line)
    except KeyboardInterrupt:
        m.ser.write(b"M410\n")
        print("stopped")
        return 1
    finally:
        m.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
