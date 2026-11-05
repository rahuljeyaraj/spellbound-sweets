# Spellbound Sweets

A creature guards this bowl of sweets, and it doesn't want you to have one. Reach in and it
snatches the bowl away from your hand, snarling at you. Want a sweet? You'll have to defeat it
with a spell: show it the horns.

My entry for the element14 Community 2026 Halloween Hackathon. The full build story, with
video, is on element14: [link to the project post].

## How it works

- **The eyes.** Two webcams up on the wall watch the table. MediaPipe finds 21 points on each
  hand in each picture, and OpenCV turns each matching pair into a point in 3D, in millimetres.
- **The bowl.** It rides on the Y axis of an old Ender 3 printer (BTT SKR Mini E3 V3, TMC2209,
  Marlin), driven with plain G-code over USB. When a hand comes within 25 cm, the bowl jumps to
  the far end of the rail at 700 mm/s. Three dodges in ten are fakes: to the middle and back.
- **Keeping track of the bowl.** A printed marker on the slider's base lets the cameras check
  where the bowl really is. If the motor has lost steps, the axis re-homes by itself.
- **The face.** Two 16-LED rings for eyes and two 8 x 8 boards for a mouth, 168 WS2812 LEDs in
  all with the ring under the horns hand, driven by a Seeed XIAO RP2350. Four moods: sleeping,
  watching, snarling and crying.
- **The spell.** A finger rule spots the horns (index and little finger straight, middle and
  ring folded). Hold it for 0.3 s and the creature gives up. Drop the horns and you have 10
  seconds to take a sweet.
- **The sound.** A live mixer on a USB speaker: a theremin-like tone that follows your hand,
  the creature's snores, growls, laughs and sobs, a whoosh for every move, and a night of wind,
  wolves and crows behind it all.

Everything runs from one Python script, `scripts/dodge.py`, on a Windows mini PC.

```
two webcams ──► mini PC ──► SKR board (G-code) ──► Y axis and bowl
                        ──► XIAO RP2350 (mood) ──► 168 LEDs
                        ──► USB speaker
```

## What's in here

| Folder | What it is |
|---|---|
| `scripts/` | The Python side: the game, calibration, the hand tracker, the sound, axis speed tests |
| `leds/` | Firmware for the face (PlatformIO, Arduino-Pico core, Adafruit NeoPixel) |
| `face/face_preview.html` | A browser preview of the face, every LED in place, the mouse as the hand |
| `sounds/` | `sources.md` credits every recording; `preview.html` is a listening page |
| `print/` | The "TRY THIS!" sign as SVG, for a laser or a print |

Each script explains itself at the top of the file.

## Parts

| Part | Qty | Notes |
|---|---|---|
| Creality Ender 3 with BTT SKR Mini E3 V3 (TMC2209), Marlin 2.1.2.8 | 1 | only the Y axis is used |
| USB webcam, 1280 x 720 at 30 fps | 2 | each on its own USB port |
| Windows mini PC | 1 | runs everything |
| Seeed Studio XIAO RP2350 | 1 | drives the LEDs |
| WS2812 ring, 16 LEDs | 2 | the eyes |
| 8 x 8 WS2812 board | 2 | the mouth |
| WS2812 ring, 8 LEDs | 1 | under the horns hand |
| 74AHCT125 level shifter, 330 ohm resistor | 1 each | LED data line |
| 5 V power supply | 1 | for the LEDs |
| USB speaker | 1 | |
| Plastic flower pot, rim about 145 mm across | 1 | the bowl |

## Wiring the face

One WS2812 chain on pin D10 of the XIAO, through the 74AHCT125 and the 330 ohm resistor. The
chain order, seen from the front:

| LEDs | What |
|---|---|
| 0 to 15 | right eye ring, first LED at 9 o'clock, running clockwise |
| 16 to 31 | left eye ring, first LED at 3 o'clock, running clockwise |
| 32 to 95 | left mouth board, rows top to bottom, each row left to right |
| 96 to 159 | right mouth board, the same |
| 160 to 167 | the ring under the horns hand |

The firmware dims the whole face if it would draw more than 1 A. Until it hears from the PC,
it loops through its moods as a test.

## Build your own

Install [uv](https://github.com/astral-sh/uv) and clone this repo. Every script runs the same
way, and the first run installs what it needs:

```
uv run python scripts/<name>.py
```

1. **The axis.** Fix the bowl to the bed carriage and plug the SKR into the PC. Find your own
   safe speed: `y_speed.py` homes before every step and stops at the first slip.

   ```
   uv run python scripts/y_speed.py --port COM9
   ```

2. **The cameras.** Mount two webcams high up and well apart, both seeing the whole bowl and
   about 30 cm past the table edge. Aim them with `view.py`.
3. **Calibrate the cameras.** Print the chessboard and glue it to card. Take 15 to 25 pairs,
   then one photo of the board flat on the table. Aim for under 0.5 px of error. `hand3d.py`
   then shows your hands in 3D.

   ```
   uv run python scripts/make_board.py
   uv run python scripts/stereo_calibrate.py
   uv run python scripts/hand3d.py
   ```

4. **Find the bowl.** Print the markers and stick the 50 mm one on the slider's base. Print the
   lid card, lay it on the rim, and give your rim's radius in mm. The printer slides on the
   table, so the rail needs calibrating before each game: `bowl_calibrate.py`, or `--calibrate`
   on dodge. Dodge also does it by itself whenever it sees the printer has been pushed sideways.

   ```
   uv run python scripts/make_marker.py
   uv run python scripts/bowl_calibrate.py --port COM9
   uv run python scripts/make_marker.py --lid 145
   uv run python scripts/rim_calibrate.py --radius 72.5
   ```

5. **The face.** Wire the chain as above and flash the `leds` folder with PlatformIO.
6. **The sounds.** Download the recordings, cut them into takes, then try them with the pad,
   where the mouse is the hand.

   ```
   uv run python scripts/make_sounds.py --fetch
   uv run python scripts/make_sounds.py
   uv run python scripts/sound.py --pad
   ```

7. **Play.** Use your printer's port. Leave `--port` out for a dry run where nothing moves.
   Keys: q quits, h re-homes, d turns dodging on and off.

   ```
   uv run python scripts/dodge.py --port COM9 --calibrate
   ```

Calibration results go in `calib/`, which is not in git: they only fit my cameras on my wall.

## Things I learned the hard way

- Find the axis speed limit in small steps, homing before each one. At 800 mm/s it crashed into
  the end, and the end switch read "pressed" from then on, so every home after was a lie.
- Don't fix lost steps with G92. It moves the soft limits too, so the spot where the bowl hit
  a hand became an invisible wall.
- Two webcams would not run together on one USB hub. Give each its own port.
- Short dodges beside the hand look like the bowl is following it, like a puppy. Jump to the
  far end.

## Credits

- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
  by Google, [OpenCV](https://opencv.org), [Marlin](https://marlinfw.org),
  [PlatformIO](https://platformio.org) with
  [platform-raspberrypi](https://github.com/maxgerhardt/platform-raspberrypi) by Max Gerhardt,
  [Arduino-Pico](https://github.com/earlephilhower/arduino-pico) by Earle Philhower and
  [Adafruit NeoPixel](https://github.com/adafruit/Adafruit_NeoPixel).
- The sounds are cut from 81 recordings on [Freesound](https://freesound.org). Every one is
  listed with its author and licence in [sounds/sources.md](sounds/sources.md). Thank you to
  everyone who shared them.
- The horns hand is "hand gestures" by lukino_maker3d on
  [MakerWorld](https://makerworld.com/en/models/2328721-hand-gestures), CC BY-NC-SA.
- Much of the code was written with Claude, an AI helper from Anthropic.

## Licence

The code and the files in this repo are under the MIT licence: see [LICENSE](LICENSE). The
sound recordings, the MediaPipe model and other people's 3D models keep their own licences.
