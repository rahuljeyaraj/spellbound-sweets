"""The sounds of the haunted bowl: a live mixer on the USB speaker, driven by dodge.py.

    uv run python scripts/sound.py --pad                # try it: the mouse is the hand, keys fire reactions
    uv run python scripts/sound.py --demo               # render the demo round to sounds/demo_round.wav
    uv run python scripts/sound.py --devices            # list the audio outputs
    uv run python scripts/sound.py --check 120          # live stream for 120 s at volume 0: dropouts, delay

dodge.py makes a Sound (Sound.open) and tells it what happens: every frame the face's mood and
where the hands are (frame), every bed move with its Axis (move), and the moments of the game
(event: horns, free, again, rehome, cornered). Every call only does a few sums and puts a
message on a queue, so it returns at once; the sound card's own thread does the mixing.

What plays, from the SOUNDS table below (one line per sound: files, group, level, when):
- the hand tone, made live: a ghostly theremin (or glass harmonica, or choir) whose pitch follows
  the nearest hand's height above the rim, snapped to a dark scale with a glide between notes;
  closeness to the bowl makes it louder and brighter, a fast hand makes it shiver. Layers sound
  with the note: an octave under, a held root, a faint ghost above, and a third and a fifth that
  grow into a chord as the hand comes close, each trailing the note a moment. The wind rises as
  a hand comes near.
- the skull's voice by mood: snores in time with the mouth while sleeping, a sniff when it
  wakes, low breaths while watching, layered growls on every dodge, grumbles while a hand is in
  the bowl, a cackle now and then after a dodge that beat the hand, sobs and wails while crying.
- slides fitted to each move from the Axis: when the bed starts (t0) and how long it takes (dur).
- the spell: a sting when the horns stop the game, a music box over the last seconds of the
  10 s to take a sweet, a rising swell into the restart sting.
- background: a wind and night bed, and now and then a wolf, bats, crows, an owl, thunder, a door
  or chains, never on top of the skull's voice. The background ducks under the voice.

All of it goes through one reverb (one creature in one room), a cut of the deep bass the small
speaker can't play, and a limiter. The mix is mono-safe: everything is centred but the
background spot sounds (a little to the sides) and the reverb.
"""

from __future__ import annotations

import argparse
import heapq
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
SOUND_DIR = ROOT / "sounds"
SR = 48000
BLOCK = 512  # samples per mixing block, 10.7 ms
LATENCY = 0.06  # s of buffering asked of the sound card; dodge.py's moves start 0.08 s after the command


@dataclass
class S:
    """One sound: its files (a glob in sounds/), group, level in dB, how much each play varies
    (pitch +- semitones, level +- dB), whether it loops, and when it plays (for the listening page)."""
    files: str
    group: str
    db: float = 0.0
    pitch: float = 0.0
    vary: float = 0.0
    when: str = ""
    loop: bool = False
    chance: float = 1.0


# Which files, how loud, and when each sound plays. Swap a sound by changing its files.
SOUNDS = {
    # the skull's voice
    "snore": S("snore_*.wav", "voice", -3, 0.4, 1.0, "Sleeping: one breath in time with each snore of the mouth (every 3 s)."),
    "snort": S("snort_*.wav", "voice", -3, 0.5, 1.0, "Sleeping: about one breath in five is a snort instead."),
    "wake": S("wake_*.wav", "voice", -2, 0.5, 1.0, "The face wakes up: a hand has come (sleeping to watching)."),
    "settle": S("settle_*.wav", "voice", -5, 0.5, 1.0, "The face falls asleep again: a sigh or a yawn."),
    "breath": S("breath_*.wav", "voice", -13, 0.6, 2.0,
                "Watching: a low breath every few seconds, quicker and louder as a hand comes near."),
    "growl": S("growl_*.wav", "voice", 0, 0.8, 1.5, "Every dodge and feint: the main snarl."),
    "beast": S("beast_*.wav", "voice", -5, 0.8, 2.0, "Under most growls: an animal growl pitched down.", chance=0.85),
    "rumble": S("rumble_*.wav", "voice", -8, 0.5, 2.0, "Under most growls: a deep rumble.", chance=0.75),
    "hiss": S("hiss_*.wav", "voice", -6, 0.8, 2.0, "On top of some snarls: a hiss.", chance=0.35),
    "grumble": S("grumble_*.wav", "voice", -6, 0.8, 2.0,
                 "A hand in the bowl, the bowl cornered at the end, or re-homing: small grumbles."),
    "cackle": S("cackle_*.wav", "voice", -3, 0.6, 1.5,
                "After a dodge that beat the hand (1 in 4), more often after a feint (3 in 5), and at the restart."),
    "sob": S("sob_*.wav", "voice", -4, 0.4, 1.5, "Crying (the horns stopped it): sobs one after another."),
    "wail": S("wail_*.wav", "voice", -6, 0.6, 1.5, "Crying: now and then a wail instead of a sob.", chance=0.25),
    # the bowl sliding
    "whoosh": S("whoosh_*.wav", "slide", 0, 0.6, 1.0, "Every dodge, feint and return: stretched to the move."),
    "scrape": S("scrape_*.wav", "slide", -9, 0.5, 1.5, "Under the whoosh: the bowl dragged along."),
    "grind": S("grind_*.wav", "slide", -2, 0.3, 1.0, "Horns: the slow stone grind of the slide to the middle."),
    "creak": S("creak_*.wav", "slide", -8, 0.8, 2.0, "The end of that slide: a creak.", chance=0.6),
    # the spell
    "spell": S("spell_*.wav", "music", 0, 0.0, 0.5, "The horns stop the game: the magic sting."),
    "musicbox": S("musicbox_*.wav", "music", -3, 0.0, 0.0, "The last seconds of the 10 s to take a sweet: the warning."),
    "swell": S("swell_*.wav", "music", -6, 0.0, 0.5, "Rises into the restart, ending as the game starts again."),
    "restart": S("restart_*.wav", "music", 0, 0.0, 0.5, "Dodging again: the restart sting."),
    "awake": S("awake_*.wav", "music", -2, 0.0, 0.0, "dodge.py is ready: the bowl is awake."),
    # background
    "wind": S("wind_*.wav", "background", 0, 0.0, 0.0, "Always: the wind. It rises as a hand comes near.", loop=True),
    "night": S("night_*.wav", "background", -8, 0.0, 0.0, "Always, quietly: the night.", loop=True),
    "howl": S("howl_*.wav", "spot", 0, 0.5, 2.0, "Now and then: a wolf far off."),
    "bats": S("bats_*.wav", "spot", -2, 0.5, 2.0, "Now and then: bats flutter past."),
    "crow": S("crow_*.wav", "spot", -4, 0.5, 2.0, "Now and then: a crow."),
    "owl": S("owl_*.wav", "spot", -4, 0.3, 2.0, "Now and then: an owl."),
    "thunder": S("thunder_*.wav", "spot", -2, 0.3, 2.0, "Now and then: distant thunder."),
    "door": S("door_*.wav", "spot", -5, 0.5, 2.0, "Now and then: a creaking door."),
    "chains": S("chains_*.wav", "spot", -6, 0.5, 2.0, "Now and then: chains."),
}

# The groups: level dB and how much goes to the shared reverb (0 to 1).
GROUPS = {
    "background": (-9.0, 0.10),
    "spot": (-11.0, 0.45),
    "hand": (-8.0, 0.35),
    "voice": (-2.0, 0.22),
    "slide": (-6.0, 0.12),
    "music": (-4.0, 0.35),
}
MASTER_DB = 6.0  # into the limiter: the demo round comes out near -16 LUFS
# While a group is sounding, these groups go down by this many dB.
DUCK = {
    "voice": {"background": 6, "spot": 10, "hand": 3},
    "music": {"background": 8, "spot": 12, "hand": 6, "voice": 6},
}

# When things happen.
TIMING = {
    "snore_period": 3.0,  # s: the mouth's snore cycle in leds/src/main.cpp
    "snore_peak": 1.5,  # s into each cycle the mouth is open widest: the loud part goes there
    "snort_every": 5,  # about one breath in this many is a snort
    "breath_gap": (2.6, 4.4),  # s between breaths while watching
    "grumble_gap": (1.6, 3.0),  # s between grumbles while a hand stays in the bowl
    "grumble_rest": 4.0,  # s: at most one cornered or re-homing grumble this often
    "sob_gap": (0.1, 0.6),  # s between sobs
    "spot_gap": (9.0, 22.0),  # s between background spot sounds
    "spot_clear": 1.5,  # s the voice must have been quiet for a spot sound to start
    "xfade": 0.35,  # s: the old mood's sounds fade out over this when the mood changes
    "snarl_gap": 1.3,  # s: at most one full layered snarl this often; dodges in between get a hiss
    "cackle_after": 0.35,  # s after the move ends, the cackle (if the hand was beaten)
    "music_land": 0.35,  # s before the game restarts that the music box's last note lands
}

# The hand tone.
HAND = {
    "voice": "theremin",  # theremin, glass or choir
    "scale": (0, 2, 3, 5, 7, 8, 11),  # harmonic minor
    "low": 0.0, "high": 380.0,  # mm above the rim of the lowest and highest note
    "near": 70.0, "far": 650.0,  # mm from the bowl where the tone is loudest and quietest
    "quiet_db": -16.0,  # level at "far", against full at "near"
    "invert": False,  # True: a lower hand plays a higher note
    "crying_db": -10.0,  # softer while the game is stopped (the hand is allowed then)
    "wind_db": 7.0,  # the wind rises this much with a hand at the bowl
}
INSTRUMENTS = {
    # root (MIDI note of the lowest), notes across the height, partials (ratio, level), how much
    # brighter it gets close up, vibrato (Hz, semitones), voices (detune cents, level), attack and
    # release s, glide s, and the layers played with the hand's note (see LAYER below).
    "theremin": dict(root=57, notes=11, partials=((1, 1.0), (2, 0.10), (3, 0.13), (4, 0.03), (5, 0.04)),
                     bright=0.9, vib=(5.6, 0.22), voices=((0, 1.0),), attack=0.35, release=0.9, glide=0.09,
                     layers=(dict(oct=-1, level=0.45, glide=0.16),  # an octave under: body
                             dict(steps=2, level=0.32, glide=0.22, near=1.0),  # a third, as the hand comes
                             dict(steps=4, level=0.24, glide=0.30, near=1.0),  # a fifth, as the hand comes
                             dict(oct=1, level=0.10, cents=7, glide=0.06, near=0.6),  # a faint ghost above
                             dict(drone=True, level=0.20, glide=1.5))),  # the root held under it all
    "glass": dict(root=69, notes=10, partials=((1, 1.0), (2.0, 0.08), (3.0, 0.05), (4.2, 0.015)),
                  bright=0.5, vib=(4.0, 0.04), voices=((-3, 0.75), (4, 0.3)), attack=0.6, release=1.4, glide=0.12,
                  layers=(dict(oct=-1, level=0.35, glide=0.2),
                          dict(steps=2, level=0.30, glide=0.25, near=1.0),
                          dict(steps=4, level=0.22, glide=0.35, near=1.0))),
    "choir": dict(root=50, notes=11, partials=None, bright=1.0, vib=(5.0, 0.15),
                  voices=((-10, 0.6), (0, 0.7), (9, 0.6)), attack=0.7, release=1.2, glide=0.14,
                  layers=(dict(steps=2, level=0.45, glide=0.25, near=1.0),  # the choir sings a chord
                          dict(steps=4, level=0.35, glide=0.35, near=1.0))),
}
# A layer: another pitch played with the hand's note, through the same partials and vibrato.
# steps: scale steps above the note (2 is the third, 4 the fifth, in the scale); oct: octaves up
# or down; drone: the root instead of the note; level against the note's 1.0; near: how much it
# waits for the hand to come close (0 always there, 1 only close); cents: detune; glide: its own,
# slower than the note's, so the harmony trails a moment behind and shimmers as the hand moves.
# The tone is kept at the same loudness however many layers sound: they add colour, not volume.
LAYER = dict(steps=0, oct=0, drone=False, level=1.0, near=0.0, cents=0.0, glide=None)
# The choir's vowel goes from "oo" (far) to "ah" (close): formant Hz, level, width.
VOWELS = {"oo": ((300, 1.0, 70), (870, 0.30, 90), (2240, 0.10, 140)),
          "ah": ((730, 1.0, 90), (1090, 0.55, 110), (2440, 0.25, 140))}

AR = np.arange(BLOCK)


def db(x: float) -> float:
    return 10.0 ** (x / 20.0)


# ---- the sound files ----

@dataclass
class Take:
    data: np.ndarray  # mono float32 at SR, one guard sample on the end (for interpolation)
    peak: float  # s from the start to the loudest moment (snores are lined up on it)
    path: Path

    @property
    def length(self) -> float:
        return (len(self.data) - 1) / SR


def load(path: Path, loop: bool) -> Take:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(SR, sr)
        x = signal.resample_poly(x, SR // g, sr // g).astype(np.float32)
    w = 2400  # the loudest 50 ms, by a running sum (a minute-long bed loads in a moment)
    if len(x) > w:
        c = np.concatenate([[0.0], np.cumsum(np.abs(x), dtype=np.float64)])
        peak = (int(np.argmax(c[w:] - c[:-w])) + w // 2) / SR
    else:
        peak = float(np.argmax(np.abs(x))) / SR
    data = np.concatenate([x, x[:1] if loop else np.zeros(1, np.float32)]).astype(np.float32)
    return Take(data, peak, path)


class Bank:
    """Every sound in SOUNDS, loaded once at start-up: the takes of each, and a picker that never
    gives the same take twice in a row."""

    def __init__(self, folder: Path = SOUND_DIR, sounds: dict[str, S] = SOUNDS, keep=None) -> None:
        """keep: load only these sounds (the rest stay silent), for trying parts on their own."""
        self.sounds = sounds
        self.takes: dict[str, list[Take]] = {}
        for name, s in sounds.items():
            wanted = keep is None or name in keep
            self.takes[name] = [load(p, s.loop) for p in sorted(folder.glob(s.files))] if wanted else []
        self.last: dict[str, int] = {}
        self.missing = [n for n, t in self.takes.items() if not t and (keep is None or n in keep)]

    def pick(self, name: str, rng: np.random.Generator, near: float | None = None) -> Take | None:
        """A random take, not the one played last. With near (s), one of the two whose length is
        closest to it."""
        takes = self.takes.get(name) or []
        if not takes:
            return None
        choices = [i for i in range(len(takes)) if i != self.last.get(name)] or [0]
        if near is not None:
            choices = sorted(choices, key=lambda i: abs(takes[i].length - near))[:2]
        i = int(rng.choice(choices))
        self.last[name] = i
        return takes[i]


# ---- the mixer ----

class Voice:
    """One sound playing: a take at some rate (pitch), gain and pan, maybe looped, maybe with an
    envelope (slides follow the bed's speed), fading in or out."""

    def __init__(self, take: Take, group: str, gain: float, rate: float = 1.0, pan: float = 0.0,
                 loop: bool = False, tag: str = "", env: np.ndarray | None = None, fade_in: float = 0.0,
                 skip: float = 0.0) -> None:
        self.take, self.data, self.n = take, take.data, len(take.data) - 1
        self.group, self.rate, self.loop, self.tag, self.env = group, rate, loop, tag, env
        self.pos = skip * SR * rate  # skip: s of the take already gone (started late)
        self.t = 0  # output samples played
        self.gain = self.gain_now = gain
        self.lr = np.array([min(1.0, 1.0 - pan), min(1.0, 1.0 + pan)], np.float32)
        self.fade = 0.0 if fade_in > 0 else 1.0
        self.fade_to = 1.0
        self.fade_step = 1.0 / (fade_in * SR) if fade_in > 0 else 0.0
        self.start = 0  # where in the next block it begins
        self.name = ""  # which sound it is
        self.done = False

    def left(self) -> float:
        """s still to play (a loop: for ever)."""
        return math.inf if self.loop else (self.n - self.pos) / (self.rate * SR)

    def stop(self, seconds: float) -> None:
        self.fade_to = 0.0
        self.fade_step = 1.0 / max(seconds * SR, 1.0)

    def mix(self, out: np.ndarray, start: int) -> None:
        m = BLOCK - start
        x = self.pos + self.rate * AR[:m]
        if self.loop:
            x %= self.n
            m_ok = m
        else:
            m_ok = int(np.searchsorted(x, self.n))  # samples before the end of the take
            if m_ok < m:
                self.done = True
            x = x[:m_ok]
        i = x.astype(np.int64)
        f = (x - i).astype(np.float32)
        chunk = self.data[i] * (1 - f) + self.data[i + 1] * f
        self.pos += self.rate * m
        if self.loop:
            self.pos %= self.n
        g = self.gain_now + (self.gain - self.gain_now) * AR[:m_ok] / m
        self.gain_now = self.gain
        if self.fade != self.fade_to:
            up = self.fade_to > self.fade
            d = self.fade_step if up else -self.fade_step
            ramp = self.fade + d * AR[:m_ok]
            ramp = np.minimum(ramp, self.fade_to) if up else np.maximum(ramp, self.fade_to)
            self.fade = min(self.fade + d * m, self.fade_to) if up else max(self.fade + d * m, self.fade_to)
            if self.fade_to == 0 and self.fade == 0:
                self.done = True
            g = g * ramp
        if self.env is not None:
            e = self.env[self.t:self.t + m_ok]
            if len(e) < m_ok:
                e = np.concatenate([e, np.full(m_ok - len(e), self.env[-1] if len(self.env) else 0.0)])
                if not self.loop and self.env[-1] == 0:
                    self.done = True
            g = g * e
        self.t += m
        out[start:start + m_ok] += (chunk * g)[:, None] * self.lr


class Reverb:
    """One shared room for everything: a synthetic stereo impulse response (darker and shorter in
    the highs, like stone), convolved in blocks in the frequency domain."""

    def __init__(self, seconds: float = 2.0, rt60=(2.3, 1.8, 1.2, 0.6), seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        n = int(seconds * SR)
        t = np.arange(n) / SR
        edges = [None, 400, 2000, 6000, None]
        ir = np.zeros((2, n))
        for c in range(2):
            noise = rng.standard_normal(n)
            for k, rt in enumerate(rt60):
                lo, hi = edges[k], edges[k + 1]
                if lo is None:
                    sos = signal.butter(4, hi, "lowpass", fs=SR, output="sos")
                elif hi is None:
                    sos = signal.butter(4, lo, "highpass", fs=SR, output="sos")
                else:
                    sos = signal.butter(4, (lo, hi), "bandpass", fs=SR, output="sos")
                ir[c] += signal.sosfiltfilt(sos, noise) * 10 ** (-3 * t / rt)
            pre = int(0.018 * SR)
            ir[c, :pre] = 0
            ir[c, pre:pre + 480] *= np.linspace(0, 1, 480)  # soft onset
            for _ in range(10):  # a few early reflections off near walls
                d = int(rng.uniform(0.006, 0.055) * SR)
                ir[c, d] += rng.uniform(-0.5, 0.5) * (1 - d / (0.06 * SR))
            ir[c] /= np.sqrt(np.sum(ir[c] ** 2))
        self.P = -(-n // BLOCK)
        ir = np.pad(ir, ((0, 0), (0, self.P * BLOCK - n)))
        parts = ir.reshape(2, self.P, BLOCK)
        self.H = np.fft.rfft(np.concatenate([parts, np.zeros_like(parts)], axis=2), axis=2).astype(np.complex64)
        self.X = np.zeros((2 * self.P, BLOCK + 1), np.complex64)  # past input spectra, written twice
        self.w = 0
        self.prev = np.zeros(BLOCK, np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        """One block of mono input -> (BLOCK, 2)."""
        spec = np.fft.rfft(np.concatenate([self.prev, x])).astype(np.complex64)
        self.prev = x.astype(np.float32)
        self.w = (self.w - 1) % self.P
        self.X[self.w] = self.X[self.w + self.P] = spec
        Y = np.einsum("pk,cpk->ck", self.X[self.w:self.w + self.P], self.H)
        return np.fft.irfft(Y, axis=1)[:, BLOCK:].T.astype(np.float32)


class Limiter:
    """Look-ahead peak limiter: the gain is worked out per 64 samples and glides between them, so
    nothing passes the ceiling and nothing clicks."""

    def __init__(self, ceiling_db: float = -1.5, release: float = 0.15, look: int = 64) -> None:
        self.thr = db(ceiling_db)
        self.S = look
        self.delay = np.zeros((look, 2), np.float32)
        self.g = 1.0
        self.prev_req = 1.0
        self.rel = 1 - math.exp(-look / (SR * release))
        self.lowest = 1.0  # for the stats: the most it has had to turn down

    def process(self, x: np.ndarray) -> np.ndarray:
        S, nsub = self.S, len(x) // self.S
        peaks = np.abs(x).max(axis=1).reshape(nsub, S).max(axis=1)
        req = np.minimum(1.0, self.thr / np.maximum(peaks, 1e-9))
        knots = np.empty(nsub + 1)
        knots[0] = self.g
        prev = self.prev_req
        for j in range(nsub):
            # The knot where input sub-block j starts to play must suit it and the one before.
            knots[j + 1] = min(knots[j] + (1 - knots[j]) * self.rel, prev, req[j])
            prev = req[j]
        self.prev_req, self.g = prev, float(knots[-1])
        self.lowest = min(self.lowest, float(knots.min()))
        y = np.concatenate([self.delay, x[:-S]])
        self.delay = x[-S:].copy()
        gains = (knots[:-1, None] + (knots[1:] - knots[:-1])[:, None] * (np.arange(S) / S)[None, :]).ravel()
        return np.clip(y * gains[:, None].astype(np.float32), -self.thr, self.thr)


class HandTone:
    """The tone that follows the hand, synthesised a block at a time from a few partials. Every
    input is smoothed here as well, per block, so it glides and never steps."""

    def __init__(self, voice: str = HAND["voice"]) -> None:
        self.phases = np.zeros((1, 1))  # each layer's and singer's running phase
        self.set_voice(voice)
        self.present = False
        self.height = 0.5  # 0 bottom note, 1 top
        self.close_in = 0.0
        self.shiver_in = 0.0
        self.soft = False
        self.amp = 0.0
        self.close = 0.0
        self.shiver = 0.0
        self.softness = 1.0
        self.note = None  # scale step being played
        self.note_age = 0.0  # s since the note last changed
        self.pitch = None  # MIDI, gliding
        self.vib_phase = 0.0
        self.trem_phase = 0.0

    def set_voice(self, voice: str) -> None:
        self.voice, self.ins = voice, INSTRUMENTS[voice]
        # The note itself is layer 0.
        self.layers = [dict(LAYER, glide=self.ins["glide"])] + [{**LAYER, **la} for la in self.ins.get("layers", ())]
        shape = (len(self.layers), len(self.ins["voices"]))
        if self.phases.shape != shape:
            self.phases = np.zeros(shape)
        self.lpitch = [None] * len(self.layers)  # MIDI, gliding, per layer
        self.offsets = np.random.default_rng(1).uniform(0, 2 * np.pi, 48)
        self.offsets[0] = 0.0

    def step_pitch(self, step: int) -> float:
        scale = HAND["scale"]
        return self.ins["root"] + 12 * (step // len(scale)) + scale[step % len(scale)]

    def _layer(self, li: int, f0: np.ndarray, bright: float) -> np.ndarray:
        """One layer of the tone at the gliding pitch f0 (Hz, per sample), all its singers."""
        ins = self.ins
        mean_f = float(f0.mean())
        if ins["partials"] is None:  # the choir: harmonics shaped by a vowel
            k = np.arange(1, int(min(4500.0 / mean_f, 40)) + 1)
            amps = 0.12 / k
            for (f_oo, g_oo, w_oo), (f_ah, g_ah, w_ah) in zip(VOWELS["oo"], VOWELS["ah"]):
                fc, g, w = (f_oo + (f_ah - f_oo) * self.close, g_oo + (g_ah - g_oo) * self.close,
                            w_oo + (w_ah - w_oo) * self.close)
                amps = amps + g / (1 + ((k * mean_f - fc) / w) ** 2)
            ratios = k.astype(float)
        else:
            ratios = np.array([r for r, _ in ins["partials"]], float)
            amps = np.array([g for _, g in ins["partials"]], float) * bright ** np.arange(len(ratios))
        keep = ratios * mean_f < 9000
        ratios, amps = ratios[keep], amps[keep]
        # Same loudness for every voice: scale by the partials' power, not their sum (a choir's
        # many harmonics would otherwise come out much quieter than a near-sine). Fixed random
        # starting phases keep the peaks down.
        amps = amps / np.sqrt(np.sum(amps ** 2)) * 0.5
        offs = self.offsets[:len(ratios)]
        tone = np.zeros(BLOCK)
        for v, (cents, vlevel) in enumerate(ins["voices"]):
            ph = self.phases[li, v] + 2 * np.pi * np.cumsum(f0 * 2 ** (cents / 1200)) / SR
            self.phases[li, v] = ph[-1] % (2 * np.pi)
            tone += vlevel * (amps[:, None] * np.sin(ratios[:, None] * ph[None, :] + offs[:, None])).sum(axis=0)
        return tone / sum(vl for _, vl in ins["voices"])

    def render(self, out: np.ndarray) -> None:
        ins, dt = self.ins, BLOCK / SR
        a = lambda tau: 1 - math.exp(-dt / tau)
        # Loudness: fade in while a hand is here, out when it goes.
        tau = ins["attack"] if self.present else ins["release"]
        target = 1.0 if self.present else 0.0
        amp0 = self.amp
        self.amp += (target - self.amp) * a(tau / 3)
        if self.amp < 1e-4 and not self.present:
            self.amp = 0.0
            self.note = self.pitch = None
            self.lpitch = [None] * len(self.layers)
            return
        self.close += (self.close_in - self.close) * a(0.25)
        self.shiver += (self.shiver_in - self.shiver) * a(0.15)
        self.softness += ((db(HAND["crying_db"]) if self.soft else 1.0) - self.softness) * a(0.5)
        # The note: the nearest scale step, changed only once the hand is clearly nearer another
        # (well past the halfway point), and held at least a moment, so a hand held still does
        # not flicker between two notes.
        x = self.height * (ins["notes"] - 1)
        self.note_age += dt
        if self.note is None or (abs(x - self.note) > 0.8 and self.note_age > 0.12):
            self.note = int(round(x))
            self.note_age = 0.0
        # Vibrato blooms once a note has settled, as a singer's does, and grows with the shiver
        # (a fast hand), as does a tremolo.
        vrate, vdepth = ins["vib"]
        vrate *= 1 + 0.35 * self.shiver
        vdepth = vdepth * min(1.0, 0.35 + self.note_age / 0.4) + 0.45 * self.shiver
        vib = vdepth * np.sin(self.vib_phase + 2 * np.pi * vrate * AR / SR)
        self.vib_phase = (self.vib_phase + 2 * np.pi * vrate * dt) % (2 * np.pi)
        bright = 0.25 + 0.75 * self.close * ins["bright"]
        level = db(HAND["quiet_db"] * (1 - self.close))
        trem = 1 - 0.3 * self.shiver * (0.5 + 0.5 * np.sin(self.trem_phase + 2 * np.pi * 7.5 * AR / SR))
        self.trem_phase = (self.trem_phase + 2 * np.pi * 7.5 * dt) % (2 * np.pi)
        env = (amp0 + (self.amp - amp0) * AR / BLOCK) * level * self.softness * trem
        tone = np.zeros(BLOCK)
        power = 0.0
        for li, la in enumerate(self.layers):
            gate = la["level"] * (1 - la["near"] * (1 - self.close))
            p_target = (ins["root"] if la["drone"] else self.step_pitch(self.note + la["steps"])) + 12 * la["oct"]
            p0 = p_target if self.lpitch[li] is None else self.lpitch[li]
            p1 = p0 + (p_target - p0) * a(la["glide"])
            self.lpitch[li] = p1
            if li == 0:
                self.pitch = p1
            if gate < 1e-3:
                continue
            power += gate * gate
            f0 = 440.0 * 2 ** ((p0 + (p1 - p0) * AR / BLOCK + vib + la["cents"] / 100 - 69) / 12)
            tone += gate * self._layer(li, f0, bright)
        tone /= math.sqrt(power)  # the same loudness however many layers sound
        out += (tone * env).astype(np.float32)[:, None]


@dataclass(order=True)
class Cmd:
    at: int  # sample index it is due
    seq: int
    kind: str = field(compare=False)
    kw: dict = field(compare=False)
    posted: float = field(compare=False, default=0.0)  # time.monotonic() when posted
    when: float | None = field(compare=False, default=None)  # the time asked for, on the caller's clock


class Mixer:
    """Mixes everything, a block at a time, on the sound card's thread (or a fake clock, to
    render to a file). Commands come in through post(), with the time they are due on the
    caller's clock; render() is told when its first sample will be heard on that clock."""

    def __init__(self, bank: Bank, seed: int | None = None, hand_voice: str = HAND["voice"]) -> None:
        self.bank = bank
        self.rng = np.random.default_rng(seed)
        self.q: queue.SimpleQueue = queue.SimpleQueue()
        self.pending: list[Cmd] = []
        self.seq = 0
        self.voices: list[Voice] = []
        self.n = 0  # samples rendered
        self.t_first = 0.0
        self.hand = HandTone(hand_voice)
        self.hand_on = True  # False: the hand tone stays silent (Sound.open's groups)
        self.reverb = Reverb()
        self.limiter = Limiter()
        self.hpf = signal.butter(4, 90, "highpass", fs=SR, output="sos")
        self.hpf_zi = np.zeros((self.hpf.shape[0], 2, 2))
        self.act = {g: 0.0 for g in GROUPS}  # how much each group is sounding, 0 to 1
        self.gain_now = {g: db(GROUPS[g][0]) for g in GROUPS}
        self.mood, self.mood_at = "", 0
        self.next_at: int | None = None  # the mood's next breath, grumble or sob
        self.snore_k = 0
        self.next_snore: tuple[str, Take | None] | None = None
        self.snorted = False
        self.voice_heard = 0  # sample index the voice was last sounding
        self.spot_at = int(self.rng.uniform(4, 10) * SR)
        self.last_spot = ""
        self.wind: Voice | None = None
        self.started_beds = False
        self.fifo = np.zeros((0, 2), np.float32)
        self.tap = None  # called with each finished block (recording)
        self.log: list | None = None  # offline: (s, sound, take, rate) for every sound started
        # Live only: (s from posting to the first sample heard, s later than asked) per timed sound.
        self.late: list[tuple[float, float]] = []

    # -- from other threads --

    def post(self, kind: str, at: float | None = None, **kw) -> None:
        self.q.put((at, kind, kw, time.monotonic()))

    # -- on the mixing thread --

    def render(self, frames: int, t_first: float) -> np.ndarray:
        """frames samples of output; t_first is when the first of them will be heard, on the
        callers' clock (time.monotonic() live, the fake clock offline)."""
        parts, have = [self.fifo], len(self.fifo)
        while have < frames:
            parts.append(self._block(t_first + have / SR))
            have += BLOCK
        out = np.concatenate(parts)
        self.fifo = out[frames:]
        return out[:frames]

    def _at(self, at: float | None) -> int:
        if at is None:
            return self.n
        return max(self.n, self.n + int(round((at - self.t_first) * SR)))

    def _block(self, t_first: float) -> np.ndarray:
        self.t_first = t_first
        if not self.started_beds:
            self.started_beds = True
            self._beds()
        while True:
            try:
                at, kind, kw, posted = self.q.get_nowait()
            except queue.Empty:
                break
            self.seq += 1
            heapq.heappush(self.pending, Cmd(self._at(at), self.seq, kind, kw, posted, at))
        end = self.n + BLOCK
        while self.pending and self.pending[0].at < end:
            c = heapq.heappop(self.pending)
            self._do(c, c.at - self.n)
        self._patterns(end)

        buses = {g: np.zeros((BLOCK, 2), np.float32) for g in GROUPS}
        breaths = np.zeros((BLOCK, 2), np.float32)  # the mood's own snores, breaths and sobs
        for v in self.voices:
            start, v.start = v.start, 0
            v.mix(breaths if v.group == "voice" and v.tag == "mood" else buses[v.group], start)
        self.voices = [v for v in self.voices if not v.done]
        if self.hand_on:
            self.hand.render(buses["hand"])
        if self.wind:  # the wind rises with a hand near the bowl, and settles when it goes
            self.wind.gain = db(self.bank.sounds["wind"].db + HAND["wind_db"] * self.hand.close * min(self.hand.amp, 1))

        # How much each group sounds (for ducking): its level, -50 dBFS = 0 to -30 = 1, quick
        # to rise and slow to fall. For the voice only its reactions count (growls, laughs,
        # grunts): the background ducks under those, not under every snore, and the spot sounds
        # wait for them, not for the breathing.
        for g, b in buses.items():
            level = 10 * math.log10(float(np.mean(b * b)) + 1e-12)
            x = min(max((level + 50) / 20, 0.0), 1.0)
            self.act[g] += (x - self.act[g]) * (0.3 if x > self.act[g] else 0.018)
        if self.act["voice"] > 0.12:
            self.voice_heard = end
        buses["voice"] += breaths
        dry = np.zeros((BLOCK, 2), np.float32)
        send = np.zeros(BLOCK, np.float32)
        for g, b in buses.items():
            level, rev = GROUPS[g]
            duck = sum(d.get(g, 0) * self.act[src] for src, d in DUCK.items() if src != g)
            gain = db(level - duck + MASTER_DB)
            ramp = (self.gain_now[g] + (gain - self.gain_now[g]) * AR / BLOCK).astype(np.float32)[:, None]
            self.gain_now[g] = gain
            b *= ramp
            dry += b
            send += rev * b.mean(axis=1)
        mixed = dry + self.reverb.process(send)
        mixed, self.hpf_zi = signal.sosfilt(self.hpf, mixed, axis=0, zi=self.hpf_zi)
        block = self.limiter.process(mixed.astype(np.float32))
        self.n = end
        if self.tap:
            self.tap(block)
        return block

    def _start(self, name: str, offset: int = 0, gain_db: float = 0.0, rate: float = 1.0, pan: float = 0.0,
               tag: str = "", env: np.ndarray | None = None, fade_in: float = 0.0, take: Take | None = None,
               skip: float = 0.0) -> Voice | None:
        s = self.bank.sounds[name]
        take = take or self.bank.pick(name, self.rng)
        if take is None:
            return None
        rate *= 2 ** (self.rng.uniform(-s.pitch, s.pitch) / 12)
        gain = db(s.db + gain_db + self.rng.uniform(-s.vary, s.vary))
        if skip > 0:  # started part-way through: fade in so it does not click
            fade_in = max(fade_in, 0.005)
        v = Voice(take, s.group, gain, rate, pan, s.loop, tag, env, fade_in, skip)
        v.start = max(0, min(offset, BLOCK - 1))
        v.name = name
        self.voices.append(v)
        if self.log is not None:
            self.log.append(((self.n + v.start) / SR, name, take.path.stem, round(rate, 3)))
        return v

    def _beds(self) -> None:
        for name in ("wind", "night"):
            take = self.bank.pick(name, self.rng)
            if take:
                v = self._start(name, take=take, fade_in=3.0, skip=self.rng.uniform(0, take.length))
                if name == "wind":
                    self.wind = v

    def voice_busy(self) -> bool:
        """Is the skull saying something (not counting its mood's breaths and sobs, nor the
        last 0.4 s of a sound tailing off)?"""
        return any(v.group == "voice" and v.tag != "mood" and not v.done and v.left() > 0.4 for v in self.voices)

    def _do(self, c: Cmd, offset: int) -> None:
        kw = c.kw
        if c.kind == "play":
            if kw.get("wait_voice") and self.voice_busy():
                # Wait for the voice to finish, up to a limit, then give up.
                if c.at - kw.setdefault("first", c.at) < kw["wait_voice"] * SR:
                    self.seq += 1
                    heapq.heappush(self.pending, Cmd(c.at + BLOCK * 4, self.seq, c.kind, kw, c.posted))
                return
            s = self.bank.sounds[kw["name"]]
            if s.chance < 1 and not kw.get("sure") and self.rng.random() > s.chance:
                return
            laughing = any(v.name == "cackle" and not v.done and v.fade_to > 0 for v in self.voices)
            if laughing and kw["name"] in ("growl", "beast", "rumble", "cackle"):
                return  # it is laughing: no growl over the laugh, and not a second laugh
            for v in self.voices:  # sounds this one takes over from fade out quickly
                if v.name in kw.get("over", ()):
                    v.stop(0.25)
            skip = 0.0
            if kw.get("keep_end") and c.when is not None:
                # Asked for a time already past (a short --resume): start that far into it, so
                # it still ends when it should.
                skip = max(0.0, self.t_first + offset / SR - c.when)
            v = self._start(kw["name"], offset, kw.get("gain_db", 0.0), kw.get("rate", 1.0), kw.get("pan", 0.0),
                            kw.get("tag", ""), kw.get("env"), kw.get("fade_in", 0.0), take=kw.get("take"), skip=skip)
            if v and kw.get("timed"):
                heard = self.t_first + v.start / SR
                self.late.append((heard - c.posted, heard - (c.posted if c.when is None else c.when)))
        elif c.kind == "stop":
            for v in self.voices:
                if v.tag == kw["tag"]:
                    v.stop(kw.get("fade", 0.3))
        elif c.kind == "cancel":
            self.pending = [p for p in self.pending if p.kw.get("tag") != kw["tag"]]
            heapq.heapify(self.pending)
            for v in self.voices:
                if v.tag == kw["tag"]:
                    v.stop(kw.get("fade", 0.3))
        elif c.kind == "mood":
            self._mood(kw["mood"], c.at)
        elif c.kind == "hand":
            h = self.hand
            h.present, h.height, h.close_in, h.shiver_in = kw["present"], kw["height"], kw["close"], kw["shiver"]
        elif c.kind == "voice":
            self.hand.set_voice(kw["voice"])

    def _mood(self, mood: str, at: int) -> None:
        if mood == self.mood:
            return
        for v in self.voices:
            if v.tag == "mood":
                v.stop(TIMING["xfade"])
        self.mood, self.mood_at = mood, at
        self.hand.soft = mood == "c"
        r = self.rng.uniform
        self.snore_k = 1  # the first cycle leaves room for the sigh of falling asleep
        self.next_snore = None
        # A snarl from a dodge lasts dodge.py's --snarl-hold (3 s) and has its growl; grumbles are
        # for a snarl that stays (a hand in the bowl or still near), so the first waits longer.
        self.next_at = {"w": at + int(r(1.0, 2.0) * SR), "n": at + int(r(3.2, 4.2) * SR),
                        "c": at + int(0.9 * SR)}.get(mood)

    def _patterns(self, end: int) -> None:
        """The mood's own sounds (snores, breaths, grumbles, sobs) and the background spot
        sounds, started in the block they fall in."""
        r = self.rng.uniform
        if self.mood == "s":
            # Each snore is placed so its loudest moment falls when the mouth is open widest.
            period, peak = TIMING["snore_period"] * SR, TIMING["snore_peak"] * SR
            while True:
                if self.next_snore is None:
                    snort = (self.bank.takes["snort"] and not self.snorted
                             and self.rng.random() < 1 / TIMING["snort_every"])
                    self.snorted = bool(snort)  # never two snorts in a row
                    name = "snort" if snort else "snore"
                    self.next_snore = (name, self.bank.pick(name, self.rng))
                name, take = self.next_snore
                if take is None:
                    break
                start = int(self.mood_at + peak + self.snore_k * period - take.peak * SR)
                if start >= end:
                    break
                self._start(name, max(0, start - self.n), tag="mood", take=take, skip=max(0, self.n - start) / SR)
                self.next_snore = None
                self.snore_k += 1
        elif self.next_at is not None and self.next_at < end:
            offset = max(0, self.next_at - self.n)
            if self.mood == "w":  # breathing quickens and grows as a hand comes near the bowl
                near = self.hand.close * min(self.hand.amp, 1)
                self._start("breath", offset, gain_db=5 * near, tag="mood")
                self.next_at += int(r(*TIMING["breath_gap"]) * (1 - 0.45 * near) * SR)
            elif self.mood == "n":
                if self.voice_busy():
                    self.next_at = end + int(0.25 * SR)
                else:
                    self._start("grumble", offset, tag="mood")
                    self.next_at += int(r(*TIMING["grumble_gap"]) * SR)
            elif self.mood == "c":
                if any(v.tag == "mood" and not v.done and v.fade_to > 0 for v in self.voices):
                    self.next_at = end + int(r(*TIMING["sob_gap"]) * SR)
                else:
                    wail = self.bank.takes["wail"] and self.rng.random() < self.bank.sounds["wail"].chance
                    self._start("wail" if wail else "sob", offset, tag="mood")
                    self.next_at = end + int(r(*TIMING["sob_gap"]) * SR)
        if self.spot_at < end:
            quiet = (end - self.voice_heard) / SR >= TIMING["spot_clear"] and self.act["music"] < 0.1
            if not quiet or self.mood in ("n", "c"):
                self.spot_at = end + int(1.5 * SR)
            else:
                names = [n for n, s in self.bank.sounds.items() if s.group == "spot" and self.bank.takes[n]
                         and n != self.last_spot]
                if names:
                    self.last_spot = str(self.rng.choice(names))
                    self._start(self.last_spot, max(0, self.spot_at - self.n), pan=r(-0.5, 0.5))
                self.spot_at = end + int(r(*TIMING["spot_gap"]) * SR)


# ---- what dodge.py talks to ----

class OneEuro:
    """Smooths a noisy value: heavily while it is still, lightly while it moves (so a moving hand
    does not lag). Casiez et al., the 1 euro filter."""

    def __init__(self, min_cut: float = 0.8, beta: float = 0.004, d_cut: float = 1.0) -> None:
        self.min_cut, self.beta, self.d_cut = min_cut, beta, d_cut
        self.x = self.dx = self.t = None

    def __call__(self, x: float, t: float) -> float:
        if self.t is None or t - self.t > 0.5:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-3)
        alpha = lambda cut: 1 / (1 + 1 / (2 * math.pi * cut * dt))
        self.dx += (((x - self.x) / dt) - self.dx) * alpha(self.d_cut)
        self.x += (x - self.x) * alpha(self.min_cut + self.beta * abs(self.dx))
        self.t = t
        return self.x


def speed_curve(dur: float, speed: float, accel: float, dist: float, n: int) -> np.ndarray:
    """The bed's speed over a move (trapezoid, as dodge.py's travelled()), n samples over dur, 0 to 1."""
    t = np.arange(n) / SR
    if speed * speed / accel >= dist:
        tp = (dist / accel) ** 0.5
        v = np.minimum(accel * t, accel * np.maximum(2 * tp - t, 0))
    else:
        ta = speed / accel
        v = np.minimum.reduce([accel * t, np.full(n, speed), accel * np.maximum(dur - t, 0)])
    return np.clip(v / max(v.max(), 1e-9), 0, 1)


class Sound:
    """The game's side: turns what dodge.py sees into mixer commands. Nothing here waits."""

    def __init__(self, mixer: Mixer, live: "Live | None" = None) -> None:
        self.mixer, self.live = mixer, live
        self.rng = np.random.default_rng(int(mixer.rng.integers(1 << 32)))  # its own: it runs on dodge.py's thread
        self.mood = ""
        self.mood_since = 0.0
        self.f_height, self.f_dist = OneEuro(0.5, 0.006), OneEuro(0.6)
        self.last_p: np.ndarray | None = None
        self.last_t = 0.0
        self.speed = 0.0
        self.hand_last = -1.0
        self.in_bowl_since_move = False
        self.cackle_due: tuple[float, float] | None = None  # (when, chance) after a dodge
        self.grumbled = -1e9
        self.last_move = self.last_snarl = -1e9

    @classmethod
    def open(cls, how: str = "auto", device: str | None = None, volume: float = 1.0,
             record: str | None = None, groups: list[str] | None = None) -> "Sound | None":
        """For dodge.py: the speaker found and the stream running, or None (with the reason
        printed) for --sound none or no speaker. groups: play only these GROUPS ("hand" is the
        hand tone) and single SOUNDS by name, e.g. background, spot and hand to film the sound
        without the creature, or voice and musicbox for the creature."""
        if how == "none":
            return None
        try:
            keep = None if groups is None else {n for n, s in SOUNDS.items() if s.group in groups or n in groups}
            mixer = Mixer(Bank(keep=keep))
            mixer.hand_on = groups is None or "hand" in groups
            if mixer.bank.missing:
                print(f"sound: no files for {', '.join(mixer.bank.missing)} (run scripts/make_sounds.py)", flush=True)
            live = Live(mixer, device, volume, record)
        except Exception as e:  # no speaker, or it would not open: play the game without sound
            print(f"sound: off ({e})", flush=True)
            return None
        print(f"sound on {live.name}, volume {volume:g}" + (f", recording to {record}" if record else "")
              + (f", only {' '.join(groups)}" if groups else ""), flush=True)
        return cls(mixer, live)

    def post(self, kind: str, at: float | None = None, **kw) -> None:
        self.mixer.post(kind, at, **kw)

    def play(self, name, at: float | None = None, **kw) -> None:
        self.post("play", at, name=name, **kw)

    def ready(self, now: float) -> None:
        self.play("awake", now)

    def frame(self, now: float, mood: str, palms: np.ndarray | None, in_bowl: bool = False) -> None:
        """Every camera frame: the face's mood letter (s w n c) and the tracked palms in rail
        coordinates (along, across, up from the rim centre, mm), or None."""
        if mood != self.mood:
            old, self.mood = self.mood, mood
            self.post("mood", now, mood=mood)
            if old == "s" and mood in ("w", "n"):
                self.play("wake", now)
            elif old in ("w", "n") and mood == "s":
                self.play("settle", now)
            elif mood == "n" and in_bowl and now - self.last_move > 1.5 and now - self.grumbled >= TIMING["grumble_rest"]:
                self.grumbled = now
                self.play("grumble", now)  # a hand got into the bowl without a dodge
        if in_bowl:
            self.in_bowl_since_move = True
        if mood == "c":
            self.cackle_due = None  # it has lost: no laughing
        if self.cackle_due and now >= self.cackle_due[0]:
            if not self.in_bowl_since_move and self.rng.random() < self.cackle_due[1]:
                # It laughs at once; the growl still going gives way to it.
                self.play("cackle", now, over=("growl", "beast", "rumble", "grumble", "hiss"), tag="laugh")
            self.cackle_due = None

        if palms is not None and len(palms):
            p = palms[np.argmin(np.hypot(palms[:, 0], palms[:, 1]))]
            dist = self.f_dist(float(np.hypot(p[0], p[1])), now)
            height = self.f_height(float(p[2]), now)
            if self.last_p is not None and now - self.last_t < 0.3 and np.linalg.norm(p - self.last_p) < 150:
                v = float(np.linalg.norm(p - self.last_p)) / max(now - self.last_t, 1e-3)
                self.speed += (v - self.speed) * min(1.0, (now - self.last_t) / 0.15)
            self.last_p, self.last_t, self.hand_last = p, now, now
            h = (height - HAND["low"]) / (HAND["high"] - HAND["low"])
            h = float(np.clip(1 - h if HAND["invert"] else h, 0, 1))
            close = float(np.clip((HAND["far"] - dist) / (HAND["far"] - HAND["near"]), 0, 1))
            shiver = float(np.clip((self.speed - 200) / 700, 0, 1))
            self.post("hand", now, present=True, height=h, close=close, shiver=shiver)
        elif self.hand_last >= 0 and now - self.hand_last > 0.3:
            self.hand_last = -1.0
            self.last_p, self.speed = None, 0.0
            self.post("hand", now, present=False, height=0.5, close=0.0, shiver=0.0)

    def move(self, kind: str, axis) -> None:
        """The bed has been sent somewhere: dodge, feint, back (a feint's return), on (on to an
        end) or slide (to the middle after the horns). axis is dodge.py's Axis, just moved."""
        t0, dur = axis.t0, max(axis.dur, 0.05)
        dist = abs(axis.target - axis.start)
        if kind == "slide":
            env = np.concatenate([speed_curve(dur, axis.speed, axis.accel, dist, int(dur * SR)) ** 0.5,
                                  np.linspace(1, 0, int(0.25 * SR)) ** 2])
            take = self.mixer.bank.pick("grind", self.rng, near=dur + 0.25)
            rate = float(np.clip(take.length / (dur + 0.25), 0.8, 1.25)) if take else 1.0
            self.play("grind", t0, env=env.astype(np.float32), rate=rate, take=take, timed=True)
            self.play("creak", t0 + dur - 0.1)
            return
        tail = 0.12
        n = int(dur * SR)
        env = np.concatenate([0.15 + 0.85 * speed_curve(dur, axis.speed, axis.accel, dist, n) ** 0.7,
                              np.linspace(1, 0, int(tail * SR)) * 0.15]).astype(np.float32)
        take = self.mixer.bank.pick("whoosh", self.rng, near=dur + tail)
        rate = float(np.clip(take.length / (dur + tail), 0.75, 1.35)) if take else 1.0
        size = min(1.0, dist / 200)  # a short hop is a smaller sound
        self.play("whoosh", t0 - 0.02, env=env, rate=rate, take=take, gain_db=-6 * (1 - size), timed=True)
        self.play("scrape", t0, env=env, gain_db=-6 * (1 - size))
        if kind in ("dodge", "feint"):
            if t0 - self.last_snarl >= TIMING["snarl_gap"]:
                # The snarl, in layers a few ms apart so they do not sound like one file.
                rng = self.rng
                for layer, lead in (("growl", 0.03), ("beast", rng.uniform(0, 0.04)), ("rumble", 0.05),
                                    ("hiss", rng.uniform(-0.15, -0.05))):
                    self.play(layer, t0 - lead)
                self.last_snarl = t0
            else:  # still growling from the last one: just a quick hiss
                self.play("hiss", t0 - 0.03, sure=True, gain_db=-3)
        if kind in ("dodge", "feint"):
            self.in_bowl_since_move = False
            self.last_move = t0
        if kind in ("dodge", "feint", "back"):
            # The laugh comes once the hand stops chasing: another move first only puts it off,
            # keeping the better odds (a feint's return is the best).
            chance = SOUNDS["cackle"].chance * {"dodge": 0.25, "feint": 0.0, "back": 0.6}[kind]
            if self.cackle_due:
                chance = max(chance, self.cackle_due[1])
            if chance > 0:
                self.cackle_due = (t0 + dur + TIMING["cackle_after"], chance)

    def event(self, name: str, at: float | None = None) -> None:
        """horns (the horns stopped the game), free (at: when the 10 s to take a sweet end),
        again (dodging again), rehome, cornered."""
        now = at if name != "free" else None
        if name == "horns":
            self.post("cancel", tag="countdown")
            self.post("cancel", tag="laugh", fade=0.4)  # it has lost: no laughing
            self.play("spell")
            self.cackle_due = None
        elif name == "free" and at is not None:
            box = self.mixer.bank.pick("musicbox", self.rng)
            if box:
                land = at - TIMING["music_land"]
                # The music box's last note is 1.2 s before its end (make_sounds.py makes it so).
                self.play("musicbox", land - (box.length - 1.2), take=box, tag="countdown", keep_end=True)
            swell = self.mixer.bank.pick("swell", self.rng)
            if swell:
                self.play("swell", at - swell.peak, take=swell, tag="countdown", keep_end=True)
        elif name == "again":
            self.post("cancel", tag="countdown", fade=0.6)
            self.play("restart", now)
            self.play("cackle", now, gain_db=-2, sure=True, wait_voice=0.5)
        elif name in ("rehome", "cornered"):
            t = time.monotonic() if now is None else now
            if t - self.grumbled >= TIMING["grumble_rest"]:
                self.grumbled = t
                self.play("grumble", now, gain_db=-2)

    def close(self) -> None:
        if self.live:
            self.live.close()


# ---- the speaker ----

def find_speakers(want: str | None) -> list[int]:
    """The output devices to try, best first: a number, part of a name, or (None/auto) the USB
    speaker. Several entries share each name (one per Windows sound system); WASAPI comes first,
    as it runs at 48 kHz with little delay, then MME. The monitors' outputs are never picked by
    themselves."""
    import sounddevice as sd
    devs = sd.query_devices()
    apis = [h["name"] for h in sd.query_hostapis()]
    outs = [(i, d) for i, d in enumerate(devs) if d["max_output_channels"] > 0]
    rank = lambda i_d: ("WASAPI" not in apis[i_d[1]["hostapi"]], "MME" not in apis[i_d[1]["hostapi"]])
    if want and want.isdigit():
        return [int(want)]
    if want and want != "auto":
        hits = [o for o in outs if want.lower() in o[1]["name"].lower()]
        if not hits:
            raise RuntimeError(f"no output device named like {want!r}")
    else:
        hits = [o for o in outs if "usb" in o[1]["name"].lower()]
        if not hits:
            raise RuntimeError("no USB speaker found (use --sound-device)")
    return [i for i, _ in sorted(hits, key=rank)]


class Live:
    """The mixer on the real speaker: a sounddevice output stream whose callback mixes. Counts
    dropouts and how long each block takes; can record what it plays to a WAV."""

    def __init__(self, mixer: Mixer, device: str | None, volume: float, record: str | None) -> None:
        import sounddevice as sd
        self.mixer, self.volume = mixer, float(volume)
        self.underflows = 0
        self.blocks = 0
        self.failed = False
        self.busy: list[float] = []
        self.rec_q: queue.SimpleQueue | None = None
        if record:
            self.rec_q = queue.SimpleQueue()
            self.rec_thread = threading.Thread(target=self._write, args=(record,), daemon=True)
            self.rec_thread.start()
            mixer.tap = lambda block: self.rec_q.put(block.copy())
        failed = []
        for self.index in find_speakers(device):  # the best way to the speaker that opens
            info = sd.query_devices(self.index)
            self.name = f"{info['name']} ({sd.query_hostapis(info['hostapi'])['name']}, device {self.index})"
            self.channels = min(2, info["max_output_channels"])
            try:
                self.stream = sd.OutputStream(samplerate=SR, blocksize=BLOCK, channels=self.channels,
                                              dtype="float32", device=self.index, latency=LATENCY,
                                              callback=self._callback)
                self.stream.start()
                break
            except Exception as e:
                failed.append(f"{self.name}: {e}")
        else:
            raise RuntimeError("; ".join(failed))
        for f in failed:
            print(f"sound: could not open {f}", flush=True)
        self.latency = float(self.stream.latency)

    def _callback(self, outdata, frames, time_info, status) -> None:
        t0 = time.perf_counter()
        if status.output_underflow:
            self.underflows += 1
        ahead = time_info.outputBufferDacTime - time_info.currentTime
        if not 0 <= ahead < 1:
            ahead = self.latency
        try:
            block = self.mixer.render(frames, time.monotonic() + ahead) * self.volume
        except Exception as e:  # a bug must not stop the stream: say it once, play silence
            if not self.failed:
                print(f"sound: mixer failed: {e!r}", flush=True)
            self.failed = True
            block = np.zeros((frames, 2), np.float32)
        outdata[:] = block if self.channels == 2 else block.mean(axis=1, keepdims=True)
        self.blocks += 1
        self.busy.append(time.perf_counter() - t0)
        if len(self.busy) > 20000:
            del self.busy[:10000]

    def _write(self, path: str) -> None:
        with sf.SoundFile(path, "w", SR, 2, "PCM_16") as f:
            while True:
                block = self.rec_q.get()
                if block is None:
                    return
                f.write(block)

    def stats(self) -> str:
        b = np.array(self.busy) * 1000 if self.busy else np.zeros(1)
        text = (f"{self.blocks} blocks, {self.underflows} dropouts; mixing a {1000 * BLOCK / SR:.1f} ms block "
                f"took {np.median(b):.2f} ms typically, {np.percentile(b, 99):.1f} ms in the worst 1%, "
                f"{b.max():.1f} ms once; stream latency {1000 * self.latency:.0f} ms")
        if self.mixer.late:
            delay, late = (np.array(self.mixer.late) * 1000).T
            text += (f"; {len(delay)} timed sounds heard {np.median(delay):.0f} ms after the command "
                     f"({delay.min():.0f} to {delay.max():.0f}), {np.median(late):+.0f} ms against the time "
                     f"asked ({late.min():+.0f} to {late.max():+.0f})")
        return text

    def close(self) -> None:
        try:
            self.stream.stop()
            self.stream.close()
        except Exception as e:  # the speaker was unplugged: nothing to stop
            print(f"sound: closing the stream: {e}", flush=True)
        if self.rec_q:
            self.rec_q.put(None)
            self.rec_thread.join(timeout=5)
        print("sound: " + self.stats(), flush=True)


# ---- offline: the demo round ----

class FakeAxis:
    """Stands in for dodge.py's Axis on a fake clock, with the same sums for how long a move takes."""

    LAG = 0.08

    def __init__(self, y: float) -> None:
        self.start = self.target = y
        self.t0 = self.dur = 0.0
        self.speed = self.accel = 1.0

    def move(self, target: float, speed: float, accel: float, now: float) -> None:
        self.start, self.target, self.speed, self.accel = self.y(now), target, speed, accel
        dist = abs(target - self.start)
        self.dur = (2 * (dist / accel) ** 0.5 if speed * speed / accel >= dist
                    else 2 * speed / accel + (dist - speed * speed / accel) / speed)
        self.t0 = now + self.LAG

    def y(self, now: float) -> float:
        if now >= self.t0 + self.dur:
            return self.target
        f = max(0.0, now - self.t0) / max(self.dur, 1e-9)
        return self.start + (self.target - self.start) * (3 * f * f - 2 * f * f * f)

    def moving(self, now: float) -> bool:
        return now < self.t0 + self.dur


def render(script, seconds: float, seed: int = 3, hand_voice: str = HAND["voice"], bank: Bank | None = None
           ) -> np.ndarray:
    """Run a script on a fake clock and return what the mixer plays. script(sound, now) is
    called at 30 frames a second, like the camera loop."""
    mixer = Mixer(bank or Bank(), seed=seed, hand_voice=hand_voice)
    mixer.log = []
    snd = Sound(mixer)
    out = []
    frame_dt, next_frame = 1 / 30, 0.0
    while mixer.n < seconds * SR:
        now = mixer.n / SR
        while next_frame <= now + BLOCK / SR:
            script(snd, next_frame)
            next_frame += frame_dt
        out.append(mixer.render(BLOCK, now))
    render.log = mixer.log
    return np.concatenate(out)


def demo_round(t_offset: float = 0.0):
    """The scripted game: (script function, length s, timeline of (s, what happens)). The rules
    are dodge.py's: the face mood from the same tests, the bowl only ever at the ends, moves timed
    by the same Axis sums, the horns stop, 10 s in the middle, then dodging again. The hand is
    in rail coordinates and chases the bowl, so its distance jumps when the bowl dodges.
    t_offset puts the game on another clock (the live check runs it on time.monotonic())."""
    YMIN, YMAX, YMID = 5.0, 215.0, 110.0
    axis = FakeAxis(YMIN)
    st = {"stopped": False, "free_until": None, "snarl_until": -9.0, "hand_any": -9.0, "pending": None,
          "horns": False, "slid": False, "ready": False, "last_move": -9.0, "mood": "", "p": None,
          "feints": 0, "cornered": False, "ref": 300.0, "seen": []}
    timeline = []

    def note(t: float, text: str) -> None:
        timeline.append((round(t, 2), text))

    def hand(now: float, bowl: float, dt: float):
        """Where the hand is (along the rail from its middle, across, height above the rim), or None.
        It creeps in, chases the bowl, dips into it, shows the horns, takes a sweet and goes."""
        if now < 22 or now >= 66:
            st["p"] = None
            return None
        if st["p"] is None:
            st["p"] = np.array([440.0, 210.0, 340.0])
        st["seen"].append(bowl)  # the hand goes for where it saw the bowl half a second ago
        bowl = st["seen"][0] if len(st["seen"]) > 15 else bowl
        del st["seen"][:-16]
        stages = [(32, (bowl + 250, 90, 190), 40),  # creeps in, coming down
                  (35.4, (bowl, 30, 70 + 60 * math.sin(now * 1.7)), 330),  # reaches wherever the bowl is
                  (37.2, (bowl + 230, 120, 230), 260),  # pulls back, missed again
                  (41.6, (bowl, 30, 70 + 60 * math.sin(now * 1.7)), 330),  # and tries again
                  (44.6, (bowl, 10, -35), 300),  # into the bowl, below the rim
                  (46.6, (bowl + 120, 60, 260), 350),  # up high, showing the horns
                  (52, (bowl, 10, -40), 140),  # the game stopped: reaches in for a sweet
                  (60, (bowl + 270, 20, 320), 70),  # takes it and lifts away slowly
                  (99, (650, 160, 360), 130)]  # and goes
        goal, speed = next((np.array(g, float), v) for end, g, v in stages if now < end)
        d = goal - st["p"]
        st["p"] = st["p"] + d / max(np.linalg.norm(d), 1e-9) * min(np.linalg.norm(d), speed * dt)
        noise = np.random.default_rng(int(now * 30) + 1).normal(0, (5, 5, 12))  # stereo noise, worst in height
        return st["p"] + noise

    def script(snd: Sound, now: float) -> None:
        T = now + t_offset  # the time on the Sound's clock
        if not st["ready"]:
            st["ready"] = True
            snd.ready(T)
            note(now, "dodge.py ready: the bowl is awake")
        bowl = axis.y(T) - YMID  # where the rim centre is along the rail, from its middle
        p = hand(now, bowl, 1 / 30)
        rp = None if p is None else p - np.array([bowl, 0, 0])  # rail coordinates from the rim centre
        if p is not None:
            st["hand_any"] = now
        dist = math.hypot(rp[0], rp[1]) if rp is not None else math.inf
        in_bowl = rp is not None and rp[2] < 0 and dist < 100
        moving = axis.moving(T)
        # dodge.py's hysteresis: after a move the hand must come 50 mm closer than it has been since
        # (or be over the opening) before the next dodge.
        if rp is not None and rp[2] >= 0:
            st["ref"] = dist if moving else min(300.0, max(st["ref"], dist))
        reaching = dist < 250 and dist < max(st["ref"] - 50, 75)
        if not st["horns"] and now >= 46.3:  # the horns, shown from 46.0 s and held 0.3 s
            st["horns"] = st["stopped"] = True
            st["pending"] = None
            snd.event("horns", T)
            note(now, "the horns: the game stops (spell sting); the face cries")
        if st["stopped"]:
            if not moving and not st["slid"]:
                axis.move(YMID, 150, 1000, T)
                snd.move("slide", axis)
                st["slid"] = True
                note(now, f"slow stone slide to the middle ({axis.dur:.2f} s)")
            elif not moving and st["free_until"] is None:
                st["free_until"] = now + 10
                snd.event("free", st["free_until"] + t_offset)
                note(now, "in the middle: 10 s to take a sweet (the music box plays the last seconds)")
            elif st["free_until"] is not None and now > st["free_until"]:
                st["stopped"] = False
                snd.event("again", T)
                note(now, "10 s up: dodging again (restart sting, a cackle)")
                axis.move(YMIN if axis.y(T) - YMIN < YMAX - axis.y(T) else YMAX, 700, 7000, T)
                snd.move("on", axis)
                note(now, f"on to an end ({axis.dur:.2f} s)")
        elif st["pending"] is not None and not moving:
            axis.move(st["pending"], 700, 7000, T)
            snd.move("back", axis)
            note(now, f"the feint darts back ({axis.dur:.2f} s)")
            st["pending"] = None
            st["snarl_until"] = now + 1.5
        elif not moving and rp is not None and 0 <= rp[2] < 250 and reaching and not in_bowl:
            target = YMIN if axis.y(T) > YMID else YMAX
            kind = "dodge"
            if now > 34.6 and st["feints"] == 0:  # one feint, just before the hand pulls back: to the middle and straight back
                st["feints"] += 1
                kind, st["pending"], target = "feint", axis.y(T), YMID
            axis.move(target, 700, 7000, T)
            snd.move(kind, axis)
            st["last_move"] = now
            st["snarl_until"] = now + 1.5
            note(now, f"{kind}: hand {dist:.0f} mm from the bowl ({axis.dur:.2f} s move)")
        if not st["cornered"] and now >= 41.8:
            st["cornered"] = True
            snd.event("cornered", T)
            note(now, "cornered at the end of the rail: a grumble")
        mood = ("c" if st["stopped"] else "n" if now < st["snarl_until"] or in_bowl
                else "w" if now - st["hand_any"] < 2.0 else "s")
        if mood != st["mood"]:
            if st["mood"]:
                note(now, {"s": "the face falls asleep (a sigh, then snores)", "w": "the face watches",
                           "n": "the face snarls" + (": a hand in the bowl" if in_bowl else ""),
                           "c": "the face cries"}[mood])
            st["mood"] = mood
        snd.frame(T, mood, None if rp is None else rp[None, :], in_bowl)

    return script, 90.0, timeline


def write(path: Path, audio: np.ndarray) -> None:
    sf.write(path, audio, SR, subtype="PCM_16")
    try:
        sf.write(path.with_suffix(".mp3"), audio, SR, format="MP3")
    except Exception as e:  # MP3 is only for the listening page
        print(f"no mp3 for {path.name}: {e}")


# ---- the try-it pad ----

def pad(device: str | None, volume: float, seconds: float | None, test_keys: str = "") -> None:
    """A window where the mouse is the hand: across is along the rail (the bowl in the middle),
    up and down is height above the rim. Number keys fire the reactions. Real speaker.
    test_keys: press these keys by itself, one a second, with a pretend mouse going round in
    circles (to test it with nobody there)."""
    import cv2
    snd = Sound.open("auto", device, volume)
    if snd is None:
        return
    W, H = 900, 560
    state = {"mouse": None, "inside": False}
    axis = FakeAxis(110.0)
    log: list[str] = []
    st = {"stopped": False, "snarl_until": -9.0, "hand_any": -9.0, "free_until": None}

    def on_mouse(ev, x, y, flags, _):
        state["mouse"] = (x, y)
        state["inside"] = 0 < x < W - 1 and 60 < y < H - 1

    cv2.namedWindow("sound pad")
    cv2.setMouseCallback("sound pad", on_mouse)
    keys = {"1": "dodge", "2": "feint", "3": "horns", "4": "again", "5": "rehome", "6": "cornered",
            "7": "cackle", "8": "voice"}
    voices = list(INSTRUMENTS)
    t_start = time.monotonic()
    try:
        while seconds is None or time.monotonic() - t_start < seconds:
            now = time.monotonic()
            palms = None
            if test_keys:
                a = (now - t_start) * 1.3
                state["mouse"] = (int(W / 2 + 300 * math.cos(a)), int(H / 2 + 180 * math.sin(a)))
                state["inside"] = True
            if state["inside"] and state["mouse"]:
                x, y = state["mouse"]
                along = (x - W / 2) / (W / 2) * 450
                up = (H - y) / (H - 60) * 450 - 40
                palms = np.array([[along, 30.0, up]])
                st["hand_any"] = now
            key = cv2.waitKey(15) & 0xFF
            k = chr(key) if key < 255 else ""
            second = int(now - t_start) - 1
            if test_keys and 0 <= second < len(test_keys) and second not in state.setdefault("pressed", set()):
                state["pressed"].add(second)
                k = test_keys[second]
            if key in (ord("q"), 27):
                break
            what = keys.get(k)
            if what in ("dodge", "feint"):
                target = 5.0 if axis.y(now) > 110 else 215.0
                if what == "feint":
                    target = 110.0
                axis.move(target, 700, 7000, now)
                snd.move(what, axis)
                st["snarl_until"] = now + 1.5
            elif what == "horns":
                st["stopped"] = True
                snd.event("horns", now)
                axis.move(110.0, 150, 1000, now)
                axis.start = 5.0
                snd.move("slide", axis)
                st["free_until"] = now + axis.dur + 10
                snd.event("free", st["free_until"])
            elif what == "again" or (st["stopped"] and st["free_until"] and now > st["free_until"]):
                st["stopped"], st["free_until"] = False, None
                snd.event("again", now)
                what = "again"
            elif what in ("rehome", "cornered"):
                snd.event(what, now)
            elif what == "cackle":
                snd.play("cackle", now, sure=True)
            elif what == "voice":
                v = voices[(voices.index(snd.mixer.hand.voice) + 1) % len(voices)]
                snd.post("voice", voice=v)
                what = f"hand tone: {v}"
            if what:
                log.append(f"{time.strftime('%H:%M:%S')}  {what}")
            in_bowl = palms is not None and palms[0, 2] < 0 and abs(palms[0, 0]) < 120
            mood = ("c" if st["stopped"] else "n" if now < st["snarl_until"] or in_bowl
                    else "w" if now - st["hand_any"] < 2.0 else "s")
            snd.frame(now, mood, palms, in_bowl)

            img = np.full((H, W, 3), 25, np.uint8)
            cv2.rectangle(img, (0, 0), (W, 60), (60, 20, 60), -1)
            cv2.putText(img, f"face: {MOODS_LONG[mood]}   hand tone: {snd.mixer.hand.voice}"
                        f"   note {snd.mixer.hand.note if snd.mixer.hand.note is not None else '-'}",
                        (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            rim_y = int(H - 40 / 490 * (H - 60))
            cv2.line(img, (0, rim_y), (W, rim_y), (80, 80, 80), 1)
            cv2.ellipse(img, (W // 2, rim_y), (int(75 / 450 * W / 2), 14), 0, 0, 360, (0, 140, 255), 2)
            cv2.putText(img, "bowl", (W // 2 - 22, rim_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 1)
            if palms is not None:
                cv2.circle(img, state["mouse"], 14, (0, 255, 255), 2)
            help_text = ["mouse = the hand (across: along the rail, up: height)", "1 dodge   2 feint   3 horns",
                         "4 again   5 rehome   6 cornered", "7 cackle   8 next hand tone   q quit"]
            for i, line in enumerate(help_text + [""] + log[-6:]):
                cv2.putText(img, line, (12, 90 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
                            cv2.LINE_AA)
            for i, g in enumerate(GROUPS):  # level meters
                x0 = W - 200
                cv2.putText(img, g, (x0, 90 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                cv2.rectangle(img, (x0 + 85, 80 + 22 * i), (x0 + 85 + int(100 * snd.mixer.act[g]), 92 + 22 * i),
                              (0, 200, 0), -1)
            cv2.imshow("sound pad", img)
    finally:
        snd.close()
        cv2.destroyAllWindows()


MOODS_LONG = {"s": "sleeping", "w": "watching", "n": "snarl", "c": "crying"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pad", action="store_true", help="the try-it pad: the mouse is the hand")
    ap.add_argument("--demo", nargs="?", const=str(SOUND_DIR / "demo_round.wav"), help="render the demo round")
    ap.add_argument("--hand-voice", default=HAND["voice"], choices=list(INSTRUMENTS), help="the hand tone")
    ap.add_argument("--devices", action="store_true", help="list the audio outputs")
    ap.add_argument("--check", type=float, metavar="S", help="run the live stream this long and report")
    ap.add_argument("--sound-device", help="output device: number or part of its name (default: the USB speaker)")
    ap.add_argument("--volume", type=float, help="0 to 1 (default 1; 0 for --check)")
    ap.add_argument("--seconds", type=float, help="--pad: quit by itself after this long")
    ap.add_argument("--test-keys", default="", help="--pad: press these keys by itself, one a second")
    a = ap.parse_args()
    if a.devices:
        import sounddevice as sd
        print(sd.query_devices())
        print("picked, best first:", find_speakers(a.sound_device))
        return 0
    if a.demo:
        script, seconds, timeline = demo_round()
        audio = render(script, seconds, hand_voice=a.hand_voice)
        write(Path(a.demo), audio)
        events = [(t, "GAME  " + text) for t, text in timeline] + [
            (t, f"      {name:9} {take}" + (f" x{rate:g}" if abs(rate - 1) > 0.01 else ""))
            for t, name, take, rate in render.log]
        Path(a.demo).with_suffix(".txt").write_text("\n".join(f"{t:6.2f}  {text}" for t, text in sorted(events)) + "\n")
        print(f"{a.demo}: {len(audio) / SR:.0f} s, peak {20 * np.log10(np.abs(audio).max()):.1f} dBFS")
        return 0
    if a.check:
        vol = 0.0 if a.volume is None else a.volume
        snd = Sound.open("auto", a.sound_device, vol)
        if snd is None:
            return 1
        end = time.monotonic() + a.check
        while time.monotonic() < end:  # the demo round's calls, on the real clock, again and again
            start = time.monotonic()
            script, seconds, _ = demo_round(start)
            while time.monotonic() < min(start + seconds, end):
                script(snd, time.monotonic() - start)
                time.sleep(1 / 30)
        snd.close()
        return 0
    if a.pad:
        pad(a.sound_device, 1.0 if a.volume is None else a.volume, a.seconds, a.test_keys)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
