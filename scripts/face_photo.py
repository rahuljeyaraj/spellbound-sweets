"""Holds the LED face still for a photo: the snarl with the jaw wide open, yellow eyes under red
brows, no flicker, and the ring under the horns hand solid yellow.

    uv run python scripts/face_photo.py              # pupils down, at the bowl
    uv run python scripts/face_photo.py --look -60   # pupils toward the face's left (-100 to 100)
    uv run python scripts/face_photo.py --off        # let it go: it sleeps after 2 s

The face holds the pose until the next line it gets (leds/src/main.cpp, the letter p), so this
sends one line and exits. Starting dodge.py lets it go too.
"""

from __future__ import annotations

import argparse

import serial
from serial.tools import list_ports

FACE_VID = 0x2886  # Seeed: the XIAO RP2350, as in dodge.py


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--face", default="auto", help="the face's serial port; auto finds it")
    ap.add_argument("--look", type=int, help="where the pupils point, -100 (face's left) to 100; leave out for down")
    ap.add_argument("--off", action="store_true", help="end the pose")
    a = ap.parse_args()

    port = next((p.device for p in list_ports.comports() if p.vid == FACE_VID), None) if a.face == "auto" else a.face
    if not port:
        print("no face found; is the XIAO plugged in?")
        return 1
    line = "s" if a.off else "p" + ("" if a.look is None else str(max(-100, min(100, a.look))))
    with serial.Serial(port, 115200, timeout=0.5) as ser:
        ser.write((line + "\n").encode())
        ser.flush()
        print(f"face on {port}: {'pose off' if a.off else 'photo snarl, holding'} (sent {line!r})")
        reply = ser.readline().decode(errors="replace").strip()
        if reply:
            print(f"face says: {reply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
