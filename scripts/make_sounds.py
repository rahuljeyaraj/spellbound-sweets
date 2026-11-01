"""Builds the bowl's sound set in sounds/: downloads the recordings, cuts and cleans them, and
makes the sounds that are synthesised (the music box, the spell stings).

    uv run python scripts/make_sounds.py --fetch      # download anything missing (slow, polite)
    uv run python scripts/make_sounds.py              # process everything into sounds/
    uv run python scripts/make_sounds.py --info       # numbers for each raw download, to pick from

Each download goes in a new folder of its own, sounds/raw/<name>/, with the page, author and
licence beside it in meta.json; sounds/sources.md is written from those. Nothing is ever run
from sounds/raw: the files there are only read as audio.

Processing (RECIPES below) turns raw files into finished takes, sounds/<cue>_<n>.wav, 48 kHz
mono: cut out of the recording, rumble filtered off, pitched if asked, silence trimmed off the
front (a late start ruins the sync) and the end, short fades so nothing clicks, loops made
seamless, and every take set to its group's loudness. sound.py's SOUNDS table says which files
each cue uses. sounds/ is not in git (too big); this script and sources.md are.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
SOUND_DIR = ROOT / "sounds"
RAW = SOUND_DIR / "raw"
SR = 48000

# Every recording we use or tried: our name -> Freesound sound id. --fetch puts each in its own
# folder under sounds/raw/. The high-quality preview (OGG, about 190 kbit/s) is public; the
# original file would need an account.
SOURCES = {
    # sleeping
    "snore_daxter": 114609, "snore_matty3": 432995, "snore_sirplus": 20545, "snore_jppi": 16755,
    "snore_matty4": 432994,
    # waking, watching
    "sniff_jrs": 67167, "sniff_inspectorj": 346024, "snort_andune": 61605, "yawn_nj": 564231,
    "breath_beast": 37822, "breath_zombie": 189281, "breath_death": 60571, "breath_arni": 140256,
    # snarl: monster, animal (pitched down), human, hiss, grumble
    "growl_cthulhu": 249686, "growl_seafury4": 48688, "growl_seafury5": 48709,
    "growl_aesqe1": 49127, "growl_aesqe2": 49128, "growl_quadra": 398908, "growl_sigsound": 869056,
    "growl_wolf": 338674, "growl_dogbeast": 404920, "growl_dogsnarl": 122183, "growl_beast": 170454,
    "growl_tiger": 439280, "growl_purr": 398254,
    "growl_man": 464993, "grunts_halgrimm3": 218895, "grunts_halgrimm2": 218894,
    "rumble_deep": 184313,
    "hiss_cat": 146960, "hiss_reit2": 343927, "hiss_reit3": 343928, "hiss_dino": 320345,
    # cackles and laughs
    "laugh_baron": 98382, "laugh_witch_mv": 223851, "laugh_witch_ad": 417826, "laugh_klank": 126113,
    "laugh_insane": 26729, "laugh_zyryt": 219110, "laugh_timtube": 169826, "laugh_ninja": 646307,
    "laugh_pepv": 212579,
    # crying
    "cry_man": 263776, "cry_qubodup": 200428, "cry_woman": 127005,
    "wail_ghosts": 556419, "wail_moan": 164285, "wail_dead": 413150, "wail_sadghost": 725112,
    # slides
    "whoosh_qubo": 60013, "whoosh_qubo22": 60009, "whoosh_qubo30": 60030, "whoosh_double": 19312,
    "whoosh_freqman": 25073, "whoosh_benboncan": 62260,
    "stone_sarcophagus": 457529, "stone_door": 578490, "stone_door2": 578491, "stone_slide": 424169,
    "stone_blocks": 25846, "stone_drag": 29987,
    "creak_wood": 175205, "creak_mafon": 377552, "creak_door_jg": 219499,
    # the spell
    "magic_glitter": 211624, "magic_sfx": 264981, "magic_gleam": 351408, "magic_smite": 249819,
    "magic_mallet": 349201, "choir_aah": 211494,
    "hit_dramatic": 222517, "hit_stab": 271945, "braam_unfa": 647712, "braam_whoosh2": 428575,
    "braam_whoosh1": 428576, "braam_timpani": 592041,
    "reverse_glassy": 9523, "reverse_rayson": 23127, "reverse_crash": 383903,
    "musicbox_lullaby": 412224,
    # background
    "wind_night": 17553, "wind_tonal1": 397091, "wind_tonal2": 397092, "wind_howl2": 117611,
    "wind_spooky2": 267891, "wind_howling": 157204, "wind_castle": 483479,
    "night_wildlife": 352514, "night_crickets": 333221,
    "wolf_howl": 398430, "wolf_ghost": 267179, "wolf_echo": 380156, "wolves": 158780,
    "bat_wings": 546823, "bat_fast": 349225, "bat_slow": 349224, "bats_flying": 437935,
    "crow_hooded": 404687, "crow_single": 418262, "crow_craw": 52450, "crows_murder": 214869,
    "owl_hoot": 465697, "owl_inchadney": 25945, "owl_tawny": 64544,
    "thunder_klang": 127140, "thunder3": 267550, "thunder4": 267549, "thunder_clap": 194364,
    "door_gate": 109710, "door_large": 70101, "door_creak": 168650, "door_squeak": 339184,
    "chain_big": 44076,
    # start-up
    "bell_toll": 73678, "bell_creepy": 180329, "bell_funeral": 329324, "bell_crows": 474266,
}

LICENCES = {
    "publicdomain/zero/1.0": "CC0", "licenses/by/4.0": "CC BY 4.0", "licenses/by/3.0": "CC BY 3.0",
    "licenses/by-nc/4.0": "CC BY-NC 4.0", "licenses/by-nc/3.0": "CC BY-NC 3.0",
    "licenses/sampling+/1.0": "Sampling+",
}


def get(url: str, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (spellbound-sweets sound set)"})
            return urllib.request.urlopen(req, timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            print(f"  busy (429), waiting {60 * (k + 1)} s", flush=True)
            time.sleep(60 * (k + 1))
    raise RuntimeError("unreachable")


def fetch(names: list[str]) -> None:
    """Download each missing source into a new folder of its own, with meta.json beside it."""
    for name in names:
        folder = RAW / name
        if (folder / "meta.json").exists():
            continue
        sid = SOURCES[name]
        page = get(f"https://freesound.org/s/{sid}/").decode("utf-8", "replace")
        meta = lambda prop: html.unescape(m.group(1)) if (m := re.search(
            rf'<meta property="{prop}" content="([^"]*)"', page)) else ""
        ogg = re.search(r'data-ogg="([^"]+)"', page)
        lic = re.search(r'href="https?://creativecommons\.org/([^"]+?)/?"[^>]*target', page)
        if not ogg:
            print(f"{name}: no preview found on the page, skipped", flush=True)
            continue
        url = ogg.group(1).replace("-lq.", "-hq.")
        data = get(url)
        folder.mkdir(parents=True, exist_ok=False)  # always a new, empty folder
        (folder / "preview.ogg").write_bytes(data)
        lic_text = lic.group(1) if lic else ""
        info = {"name": name, "freesound_id": sid, "title": meta("og:audio:title"),
                "author": meta("og:audio:artist"), "page": meta("og:url"),
                "licence": LICENCES.get(lic_text, lic_text or "unknown"), "file": url}
        (folder / "meta.json").write_text(json.dumps(info, indent=1))
        print(f"{name}: {info['title']} by {info['author']}, {info['licence']}, {len(data) // 1024} kB", flush=True)
        time.sleep(4)  # be polite: one sound every few seconds


# ---- measuring ----

def read(name: str) -> np.ndarray:
    """A raw download as mono float at SR. Stereo is averaged, unless the two sides cancel."""
    x, sr = sf.read(RAW / name / "preview.ogg", dtype="float64", always_2d=True)
    if x.shape[1] > 1:
        l, r = x[:, 0], x[:, 1]
        corr = float(np.dot(l, r) / (np.linalg.norm(l) * np.linalg.norm(r) + 1e-12))
        x = l if corr < -0.2 else x.mean(axis=1)
    else:
        x = x[:, 0]
    if sr != SR:
        g = math.gcd(SR, sr)
        x = signal.resample_poly(x, SR // g, sr // g)
    return x


K_WEIGHT = [signal.tf2sos([1.53512485958697, -2.69169618940638, 1.19839281085285],
                          [1.0, -1.69065929318241, 0.73248077421585]),
            signal.tf2sos([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])]


def loudness(x: np.ndarray) -> tuple[float, float]:
    """(integrated, loudest 400 ms) loudness in LUFS, ITU-R BS.1770 with its gates."""
    y = x
    for sos in K_WEIGHT:
        y = signal.sosfilt(sos, y)
    win, hop = int(0.4 * SR), int(0.1 * SR)
    if len(y) < win:
        ms = np.array([np.mean(y * y)])
    else:
        ms = np.array([np.mean(y[i:i + win] ** 2) for i in range(0, len(y) - win + 1, hop)])
    lk = -0.691 + 10 * np.log10(ms + 1e-15)
    gated = ms[lk > -70]
    if len(gated) == 0:
        return -70.0, float(lk.max())
    rel = -0.691 + 10 * np.log10(gated.mean()) - 10
    gated = ms[(lk > -70) & (lk > rel)]
    return float(-0.691 + 10 * np.log10(gated.mean())), float(lk.max())


def envelope(x: np.ndarray, win: float = 0.02, hop: float = 0.005) -> tuple[np.ndarray, int]:
    """Short-term level in dBFS, one value per hop."""
    w, h = int(win * SR), int(hop * SR)
    n = max(1, (len(x) - w) // h + 1)
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    seg = x[np.minimum(idx, len(x) - 1)]
    return 10 * np.log10(np.mean(seg * seg, axis=1) + 1e-12), h


def segments(x: np.ndarray, below: float = 30.0, gap: float = 0.15, min_len: float = 0.2,
             floor_margin: float = 8.0) -> list[tuple[float, float, float]]:
    """The separate sounds in a recording, split at its quiet gaps: (start s, end s, peak dB).
    Quiet is `below` dB under the loudest moment, or just above the noise floor if that is higher."""
    env, h = envelope(x)
    peak, floor = env.max(), np.percentile(env, 10)
    thr = max(peak - below, floor + floor_margin)
    on = env > thr
    out, i, n = [], 0, len(env)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < n and (on[j] or np.any(on[j:j + int(gap * SR / h)])):
            j += 1
        s, e = i * h / SR, (j * h + int(0.02 * SR)) / SR
        if e - s >= min_len:
            out.append((s, min(e, len(x) / SR), float(env[i:j].max())))
        i = j
    return out


def info(names: list[str]) -> None:
    """The numbers for each raw download: length, level, noise floor, clipping, how bright, and
    the sounds found in it."""
    for name in names:
        if not (RAW / name / "preview.ogg").exists():
            continue
        x = highpass(read(name), 40)
        env, _ = envelope(x)
        integ, mmax = loudness(x)
        clipped = int(np.sum(np.abs(x) > 0.995))
        spec = np.abs(np.fft.rfft(x[: SR * 20] * np.hanning(min(len(x), SR * 20))))
        f = np.fft.rfftfreq(min(len(x), SR * 20), 1 / SR)
        cent = float(np.sum(f * spec) / (np.sum(spec) + 1e-12))
        roll = float(f[np.searchsorted(np.cumsum(spec ** 2) / np.sum(spec ** 2), 0.99)])
        segs = segments(x)
        meta = json.loads((RAW / name / "meta.json").read_text())
        print(f"{name:18} {len(x) / SR:6.1f}s peak {20 * np.log10(np.abs(x).max() + 1e-12):5.1f} "
              f"loud {integ:5.1f}/{mmax:5.1f} LUFS floor {np.percentile(env, 10):5.1f} dB "
              f"clip {clipped:4d} centre {cent:5.0f} Hz 99% {roll:5.0f} Hz  {meta['licence']:10} "
              f"{len(segs)} parts: " + " ".join(f"{s:.2f}-{e:.2f}({p:.0f})" for s, e, p in segs[:14]))


# ---- processing ----

def highpass(x: np.ndarray, hz: float, order: int = 2) -> np.ndarray:
    return signal.sosfilt(signal.butter(order, hz, "highpass", fs=SR, output="sos"), x)


def lowpass(x: np.ndarray, hz: float, order: int = 2) -> np.ndarray:
    return signal.sosfilt(signal.butter(order, hz, "lowpass", fs=SR, output="sos"), x)


def repitch(x: np.ndarray, semitones: float) -> np.ndarray:
    """Pitch by resampling: down is also slower, like a bigger creature."""
    if not semitones:
        return x
    ratio = 2 ** (semitones / 12)
    up, down = 1000, int(round(1000 * ratio))
    g = math.gcd(up, down)
    return signal.resample_poly(x, up // g, down // g)


def trim(x: np.ndarray, front: float = 40.0, back: float = 45.0, pre: float = 0.004,
         fade_in: float = 0.003, fade_out: float = 0.04) -> np.ndarray:
    """Cut the silence off both ends (the front tight, so the sound starts when it is told to),
    then short fades so neither end clicks."""
    env, h = envelope(x, 0.01, 0.0025)
    peak = env.max()
    a = int(np.argmax(env > peak - front)) * h
    b = (len(env) - int(np.argmax(env[::-1] > peak - back))) * h + int(0.01 * SR)
    x = x[max(0, a - int(pre * SR)):min(len(x), b)].copy()
    n_in, n_out = int(fade_in * SR), min(int(fade_out * SR), len(x) // 2)
    x[:n_in] *= np.sin(np.linspace(0, np.pi / 2, n_in)) ** 2
    x[len(x) - n_out:] *= np.cos(np.linspace(0, np.pi / 2, n_out)) ** 2
    return x


def make_loop(x: np.ndarray, seconds: float, xfade: float = 3.0, start: float = 0.0) -> np.ndarray:
    """A seamless loop `seconds` long: the end is crossfaded (equal power) into the start."""
    n, X = int(seconds * SR), int(xfade * SR)
    a = int(start * SR)
    s = x[a:a + n + X]
    if len(s) < n + X:
        raise ValueError(f"needs {seconds + xfade:.1f} s from {start:.1f} s, the recording is {len(x) / SR:.1f} s")
    y = s[:n].copy()
    t = np.linspace(0, np.pi / 2, X)
    y[:X] = s[:X] * np.sin(t) + s[n:n + X] * np.cos(t)
    return y


def seam_jump(y: np.ndarray) -> float:
    """How big the step from a loop's last sample to its first is, against its usual step (1 is normal)."""
    return float(abs(y[0] - y[-1]) / (np.median(np.abs(np.diff(y))) + 1e-12))


def normalise(x: np.ndarray, target: float, how: str, ceiling: float = -1.0) -> tuple[np.ndarray, str]:
    """Set the loudness (LUFS; 'max' = loudest 400 ms, for one-shots; 'int' = integrated, for
    beds). If that would push a peak over the ceiling, settle for the ceiling and say so."""
    note, cut = "", 0.0
    for _ in range(3):  # limiting takes a little loudness off, so measure again
        integ, mmax = loudness(x)
        x = x * 10 ** ((target - (mmax if how == "max" else integ)) / 20)
        over = 20 * np.log10(np.abs(x).max() / 10 ** (ceiling / 20))
        if over <= 0.01:
            break
        x = limit(x, ceiling, max_cut=6.0)
        cut = max(cut, min(over, 6.0))
        note = f"peaks limited by up to {cut:.1f} dB"
    over = 20 * np.log10(np.abs(x).max() / 10 ** (ceiling / 20))
    if over > 0.01:  # more than the limiter may take: settle for a quieter take
        x = x / 10 ** (over / 20)
        note += f"; {over:.1f} dB under target"
    return x, note


def limit(x: np.ndarray, ceiling: float, max_cut: float, release: float = 0.06) -> np.ndarray:
    """Turn the peaks above the ceiling down (at most max_cut dB): 2 ms look-ahead, smooth release."""
    from scipy.ndimage import minimum_filter1d
    thr = 10 ** (ceiling / 20)
    need = np.maximum(np.minimum(1.0, thr / np.maximum(np.abs(x), 1e-12)), 10 ** (-max_cut / 20))
    need = minimum_filter1d(need, size=2 * int(0.002 * SR) + 1)  # the lowest gain within 2 ms
    a = math.exp(-1 / (release * SR))
    g = np.minimum(signal.lfilter([1 - a], [1, -a], need - 1, zi=[0.0])[0] + 1, need)  # smooth release
    ramp = int(0.001 * SR)
    return x * np.convolve(g, np.ones(ramp) / ramp, mode="same")  # no steps in the gain



# ---- synthesised sounds ----

def tone_env(n: int, attack: float, tau: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(t / max(attack, 1e-4), 1) * np.exp(-t / tau)


def tine(midi: float, length: float, strength: float, rng: np.random.Generator) -> np.ndarray:
    """One note of a music box: a plucked steel tine. Its overtones sit where a clamped bar's do
    (about 3 and 6.3 times the note) and die away faster than the note itself."""
    f = 440 * 2 ** ((midi - 69) / 12)
    n = int(length * SR)
    t = np.arange(n) / SR
    tau = 1.5 * (880 / f) ** 0.6
    x = np.zeros(n)
    for ratio, level, k in ((1.0, 1.0, 1.0), (3.01, 0.28, 0.3), (6.27, 0.12, 0.12), (11.2, 0.04, 0.05)):
        if f * ratio < 18000:
            x += level * np.sin(2 * np.pi * f * ratio * t + rng.uniform(0, 6.28)) * tone_env(n, 0.0008, tau * k)
    click = highpass(rng.standard_normal(int(0.004 * SR)), 2500) * np.linspace(1, 0, int(0.004 * SR)) * 0.08
    x[:len(click)] += click
    return x * strength


MELODIES = {
    # MIDI notes, A harmonic minor; the box winds down, each note a little later than the last.
    # Original tunes written for the bowl. The last one rings 1.2 s before the file ends, which
    # sound.py counts on to land it just before the game restarts.
    1: dict(notes=[88, 84, 81, 83, 80, 76, 77, 80, 83, 81], bass={0: [69, 64], 3: [64], 6: [62], 9: [57, 69]}),
    2: dict(notes=[81, 84, 88, 87, 88, 84, 83, 80, 81, 76], bass={0: [57, 64], 4: [64], 6: [62], 9: [64, 68]}),
}


def music_box(which: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed + which)
    m = MELODIES[which]
    gaps = [0.34 * 1.07 ** k for k in range(len(m["notes"]) - 1)]
    times = np.concatenate([[0], np.cumsum(gaps)])
    total = times[-1] + 1.2
    out = np.zeros(int((total + 0.01) * SR))
    for k, (t, note) in enumerate(zip(times, m["notes"])):
        sag = -0.25 * max(0, k - (len(times) - 3))  # the spring running down: the last notes sag
        notes = [(note + sag + rng.normal(0, 0.03), rng.uniform(0.75, 0.95) if k < len(times) - 1 else 1.0)]
        notes += [(b + sag, 0.35) for b in m["bass"].get(k, [])]
        for midi, strength in notes:
            x = tine(midi, total - t + 0.01, strength, rng)
            i = int(t * SR)
            out[i:i + len(x)] += x[:len(out) - i]
    # The clockwork inside: a faint whir that ticks with the governor, running down with the tune.
    n = len(out)
    t = np.arange(n) / SR
    rate = 11 * (1 - 0.25 * t / total)
    tick = (0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(rate) / SR)) ** 6
    whir = signal.sosfilt(signal.butter(2, (1800, 6000), "bandpass", fs=SR, output="sos"), rng.standard_normal(n))
    out += whir * tick * 0.012 * np.minimum(1, (total - t) / 0.8).clip(0, 1)
    # A slow wobble in speed, like an old box: done as a gentle delay modulation.
    wob = 0.0006 * SR * np.sin(2 * np.pi * 0.35 * np.arange(n) / SR)
    out = np.interp(np.arange(n) - wob - wob.max(), np.arange(n), out)
    fade = int(0.35 * SR)
    out[-fade:] *= np.linspace(1, 0, fade) ** 2
    return out


def choir_chord(notes: list[int], length: float, vowel: str = "ah", attack: float = 0.06, seed: int = 5
                ) -> np.ndarray:
    """A sung chord from additive voices shaped by vowel formants (as the hand tone's choir)."""
    rng = np.random.default_rng(seed)
    formants = {"ah": ((730, 1.0, 90), (1090, 0.55, 110), (2440, 0.25, 140)),
                "oo": ((300, 1.0, 70), (870, 0.3, 90), (2240, 0.1, 140))}[vowel]
    n = int(length * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for note in notes:
        for cents in (-9, 0, 8):  # three singers a note
            f0 = 440 * 2 ** ((note - 69) / 12 + cents / 1200)
            vib = 1 + 0.004 * np.sin(2 * np.pi * rng.uniform(4.5, 5.8) * t + rng.uniform(0, 6))
            phase = 2 * np.pi * np.cumsum(f0 * vib) / SR
            for k in range(1, int(5000 / f0) + 1):
                amp = 0.1 / k + sum(g / (1 + ((k * f0 - fc) / w) ** 2) for fc, g, w in formants)
                out += amp * np.sin(k * phase)
    env = np.minimum(t / attack, 1) * np.exp(-np.maximum(t - attack, 0) / (length / 2.5))
    return out * env / np.abs(out).max()


def shimmer(length: float, seed: int = 9) -> np.ndarray:
    """Sparkles: little high bell pings, more and higher as it goes, like dust lit up."""
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    out = np.zeros(n)
    pent = [0, 3, 5, 7, 10]
    for k in range(70):
        t0 = length * 0.85 * (k / 70) ** 0.8 + rng.uniform(0, 0.05)
        midi = 84 + 12 * (k / 70) + pent[rng.integers(5)] + 12 * rng.integers(0, 2)
        x = tine(midi, length - t0, rng.uniform(0.2, 0.5), rng)
        i = int(t0 * SR)
        out[i:i + len(x)] += x[:n - i]
    return out / np.abs(out).max()


def boom(length: float = 1.6, f0: float = 120, f1: float = 45, seed: int = 4) -> np.ndarray:
    """A low hit: a falling sine, saturated so a small speaker still shows it through the harmonics."""
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / 0.12)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.45)
    x = np.tanh(3 * x) / np.tanh(3)
    x += lowpass(rng.standard_normal(n), 900) * np.exp(-t / 0.05) * 0.3
    return x


def drone(length: float, notes=(45, 52, 57), seed: int = 6) -> np.ndarray:
    """A low swell of detuned saws, filtered dark: the room waking."""
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for note in notes:
        for cents in (-12, 0, 11):
            f = 440 * 2 ** ((note - 69) / 12 + cents / 1200)
            out += signal.sawtooth(2 * np.pi * f * t + rng.uniform(0, 6))
    out = lowpass(out, 700, 4)
    env = np.sin(np.pi * np.minimum(t / length, 1)) ** 1.5
    return out * env / np.abs(out).max()


def mix_layers(layers: list[tuple[np.ndarray, float, float]]) -> np.ndarray:
    """Layers (sound, start s, gain dB) mixed into one, each scaled to its loudest 400 ms first."""
    length = max(int(s * SR) + len(x) for x, s, _ in layers)
    out = np.zeros(length)
    for x, s, g in layers:
        _, mmax = loudness(x)
        i = int(s * SR)
        out[i:i + len(x)] += x * 10 ** ((g - 16 - mmax) / 20)
    return out


# ---- the recipes ----

# Loudness each group's takes are set to (LUFS): the loudest 400 ms of a one-shot, or the
# integrated loudness of a bed. sound.py's group levels then balance the groups.
TARGET = {"voice": (-16, "max"), "slide": (-16, "max"), "music": (-16, "max"), "spot": (-17, "max"),
          "background": (-24, "int")}
GROUP_OF = {}  # cue -> group, filled from sound.py's table below

# How each cue's takes are made, from which recording. Options:
#   parts: which of the recording's separate sounds to use: indexes into segments(), or
#          (start, end) in s; default all of them
#   seg: options for segments() (below, gap, min_len)
#   pitch: semitones (down is bigger and slower); max_len, min_len: keep parts this long (s,
#   after pitching); best: keep the loudest n; gain: dB; fade: fade-out s; lowpass, highpass: Hz
#   loop: make a seamless loop this many s long (start: where in the recording)
#   exact: cut (start, end) parts exactly there; otherwise each cut moves to the quietest
#          moment within 0.25 s, so it does not fall in the middle of a sound
#   group: the loudness group, when the cue is an alternative not in sound.py's table
RECIPES = [
    # sleeping: snores pitched down into a bigger creature
    ("snore", "snore_matty3", dict(pitch=-4, min_len=1.0, max_len=2.9)),
    ("snore", "snore_jppi", dict(pitch=-5, min_len=1.0, max_len=2.9, best=6)),
    ("snort", "snort_andune", dict(parts=[(1.20, 3.95)], pitch=-4)),
    ("snort", "snore_sirplus", dict(parts=[(2.40, 4.40)], pitch=-3)),
    ("wake", "grunts_halgrimm2", dict(pitch=-5, best=4)),
    ("wake", "sniff_jrs", dict(parts=[0], pitch=-3)),
    ("settle", "yawn_nj", dict(parts=[1], pitch=-5, fade=0.3)),
    ("settle", "breath_beast", dict(parts=[(0.38, 1.42)], pitch=-2, fade=0.3)),
    ("breath", "breath_beast", dict(parts=[(2.70, 4.40), (5.60, 7.20)], pitch=-3, fade=0.3)),
    ("breath", "breath_death", dict(parts=[(0.85, 4.2), (4.2, 7.73)], pitch=-2, fade=0.4)),
    # snarl
    ("growl", "growl_seafury4", dict()),
    ("growl", "growl_seafury5", dict()),
    ("growl", "growl_aesqe1", dict()),
    ("growl", "growl_aesqe2", dict()),
    ("growl", "growl_cthulhu", dict(parts=[(0.0, 2.6), (2.6, 5.0), (5.0, 7.6)], fade=0.25)),
    ("beast", "growl_tiger", dict(pitch=-5, min_len=1.0, max_len=2.6, best=5)),
    ("beast", "growl_dogbeast", dict(pitch=-5, min_len=1.2, max_len=3.0, best=4)),
    ("beast", "growl_wolf", dict(pitch=-4)),
    ("rumble", "growl_purr", dict(pitch=-3)),
    ("rumble", "rumble_deep", dict(min_len=0.5, max_len=1.6, best=4)),
    ("hiss", "hiss_reit2", dict(pitch=-3)),
    ("hiss", "hiss_reit3", dict(pitch=-3)),
    ("hiss", "hiss_cat", dict(pitch=-2)),
    ("grumble", "grunts_halgrimm3", dict(pitch=-6, best=6)),
    ("grumble", "growl_dogsnarl", dict(pitch=-5, min_len=0.6, max_len=1.8, best=5)),
    # laughs: the deep evil laughs are the skull's; the witch cackles are a runner-up
    ("cackle", "laugh_baron", dict(pitch=-2, fade=0.3)),
    ("cackle", "laugh_klank", dict(pitch=-2, fade=0.3)),
    ("cackle", "laugh_timtube", dict(parts=[0], pitch=-2, fade=0.3)),
    ("cackle", "laugh_pepv", dict(pitch=-2, fade=0.3)),
    ("alt_cackle_witch", "laugh_witch_mv", dict(parts=[0], fade=0.2, group="voice")),
    ("alt_cackle_witch", "laugh_witch_ad", dict(fade=0.15, group="voice")),
    ("alt_cackle_witch", "laugh_ninja", dict(fade=0.2, group="voice")),
    # crying
    ("sob", "cry_man", dict(pitch=-2, min_len=0.8, max_len=3.0, best=8)),
    ("sob", "cry_qubodup", dict(pitch=-2, min_len=0.5, max_len=3.0, best=6)),
    ("wail", "wail_dead", dict(parts=[(0.2, 3.9), (8.0, 11.6), (15.5, 19.2), (24.0, 27.6)], fade=0.6)),
    ("wail", "wail_sadghost", dict(parts=[0], max_len=8, fade=0.8)),
    ("wail", "wail_moan", dict(fade=0.6)),
    # the bowl sliding: whooshes at several lengths, so every move can be fitted closely
    ("whoosh", "whoosh_qubo", dict(fade=0.04)),
    ("whoosh", "whoosh_qubo22", dict(fade=0.04)),
    ("whoosh", "whoosh_qubo30", dict(fade=0.04)),
    ("whoosh", "whoosh_qubo22", dict(pitch=-4, fade=0.05)),
    ("whoosh", "whoosh_qubo30", dict(pitch=-5, fade=0.05)),
    ("whoosh", "whoosh_double", dict(fade=0.05)),
    ("whoosh", "whoosh_double", dict(pitch=-3, fade=0.05)),
    ("whoosh", "whoosh_freqman", dict(exact=True, parts=[(0.5, 1.25)], fade=0.15)),
    ("whoosh", "whoosh_benboncan", dict(exact=True, parts=[(1.3, 2.2)], fade=0.15)),
    ("scrape", "stone_slide", dict(exact=True, parts=[(15.0, 15.7), (17.2, 17.9), (20.1, 20.8), (23.0, 23.7)], fade=0.1)),
    ("scrape", "stone_drag", dict(parts=[2, 4], max_len=1.0, fade=0.1)),
    ("grind", "stone_sarcophagus", dict(parts=[0], pitch=-3, fade=0.2)),
    ("grind", "stone_door", dict(exact=True, parts=[(2.0, 3.3), (5.0, 6.3), (8.0, 9.3)], pitch=-2, fade=0.2)),
    ("grind", "stone_blocks", dict(exact=True, parts=[(3.0, 4.2), (6.5, 7.7)], pitch=-2, fade=0.2)),
    ("creak", "creak_wood", dict(best=4)),
    ("creak", "creak_mafon", dict(pitch=-3)),
    ("creak", "creak_door_jg", dict(pitch=-2)),
    # the spell: swells (a reversed cymbal peaks at its very end), restart hits
    ("swell", "reverse_glassy", dict(parts=[0], fade=0.01)),
    ("swell", "reverse_crash", dict(exact=True, parts=[(6.4, 8.63)], fade=0.01)),
    ("restart", "braam_timpani", dict(parts=[0], fade=0.3)),
    ("alt_restart_whoosh", "braam_whoosh1", dict(parts=[0], fade=0.4, group="music")),
    ("alt_restart_whoosh", "braam_whoosh2", dict(parts=[0], fade=0.4, group="music")),
    ("alt_musicbox_lullaby", "musicbox_lullaby", dict(exact=True, parts=[(26.5, 32.7)], fade=0.8, group="music")),
    # background beds: seamless loops of a minute or more
    ("wind", "wind_spooky2", dict(loop=62, start=6.0)),
    ("alt_wind_castle", "wind_castle", dict(loop=70, start=5.0, group="background")),
    ("alt_wind_night", "wind_night", dict(loop=50, start=3.5, group="background")),
    ("alt_wind_tonal", "wind_tonal1", dict(loop=44, start=2.0, group="background")),
    ("night", "night_wildlife", dict(loop=56, start=1.0)),
    ("alt_night_crickets", "night_crickets", dict(loop=70, start=10.0, group="background")),
    # now and then, far off
    ("howl", "wolf_howl", dict(parts=[0], fade=1.0)),
    ("howl", "wolf_ghost", dict(parts=[0], fade=1.0)),
    ("howl", "wolf_echo", dict(fade=1.0, max_len=9.5)),
    ("howl", "wolves", dict(parts=[(1.4, 9.0)], fade=1.5)),
    ("bats", "bat_fast", dict(parts=[0], fade=0.2)),
    ("bats", "bat_slow", dict(parts=[0], fade=0.4)),
    ("bats", "bats_flying", dict(parts=[(1.4, 6.6), (12.5, 17.7)], fade=0.5)),
    ("crow", "crow_single", dict(lowpass=6500, parts=[(0.4, 2.2)], fade=0.3)),
    ("crow", "crow_craw", dict(lowpass=6500, parts=[(0.0, 2.6)], fade=0.3)),
    ("crow", "crow_hooded", dict(lowpass=6500, parts=[(4.0, 7.3), (15.0, 17.5), (26.7, 29.2)], fade=0.4)),
    ("crow", "crows_murder", dict(lowpass=6500, parts=[(4.2, 10.6)], fade=1.0)),
    ("owl", "owl_hoot", dict(parts=[(0.1, 4.3)], fade=0.5)),
    ("owl", "owl_inchadney", dict(min_len=0.9, best=4, fade=0.4)),
    ("owl", "owl_tawny", dict(min_len=1.0, max_len=6, best=3, fade=0.5)),
    ("thunder", "thunder_klang", dict(min_len=3, max_len=12, best=3, fade=1.5)),
    ("thunder", "thunder3", dict(max_len=14, best=1, fade=2.0)),
    ("thunder", "thunder4", dict(max_len=14, best=1, fade=2.0)),
    ("door", "door_gate", dict(lowpass=6500, best=1, fade=0.5)),
    ("door", "door_large", dict(lowpass=6500, best=1, fade=0.5)),
    ("chains", "chain_big", dict(lowpass=6500, min_len=1.0, max_len=4.5, best=4, fade=0.4)),
]

# Takes made from several layers or synthesised (cue, name, function).
COMPOSITES = [
    ("musicbox", "tune1", lambda: music_box(1)),
    ("musicbox", "tune2", lambda: music_box(2)),
    ("spell", "glitter_choir", lambda: mix_layers([
        (raw_part("magic_glitter", 0), 0.0, -2), (choir_chord([57, 64, 69, 72, 76, 83], 2.6), 0.02, 0),
        (raw_part("magic_mallet", 0), 0.0, -6)])),
    ("spell", "shimmer_choir", lambda: mix_layers([
        (shimmer(2.4), 0.0, -3), (choir_chord([50, 57, 62, 65, 69, 76], 2.6, seed=8), 0.03, 0),
        (raw_part("magic_smite", 0), 0.0, -8)])),
    ("spell", "glitter_gleam", lambda: mix_layers([
        (raw_part("magic_glitter", 1), 0.0, -2), (raw_part("magic_gleam", 0), 0.05, -3),
        (choir_chord([57, 64, 72, 76], 2.4, seed=3), 0.02, -2)])),
    ("restart", "hit_boom", lambda: mix_layers([(raw_part("hit_dramatic", 0), 0.0, 0), (boom(), 0.0, -4)])),
    ("restart", "stab_boom", lambda: mix_layers([(raw_part("hit_stab", 0), 0.0, 0), (boom(f0=140), 0.0, -3)])),
    ("restart", "braam_boom", lambda: mix_layers([(raw_part("braam_unfa", 0), 0.0, 0), (boom(f0=110), 0.0, -5)])),
    ("swell", "noise", lambda: synth_swell(1.6)),
    ("awake", "bell_drone", lambda: mix_layers([(raw_part("bell_toll", 0), 0.6, 0), (drone(4.0), 0.0, -4)])),
    ("awake", "funeral_drone", lambda: mix_layers([(raw_part("bell_funeral", 0), 0.6, 0),
                                                   (drone(4.0, (43, 50, 55)), 0.0, -4)])),
    ("alt_awake_drone", "boom", lambda: mix_layers([(drone(3.5), 0.0, -2), (boom(2.0, 90, 40), 0.9, -6)])),
]


def synth_swell(length: float, seed: int = 2) -> np.ndarray:
    """A rush that rises and stops dead: noise, opening up and getting louder, cut at its peak."""
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    x = rng.standard_normal(n)
    out = np.zeros(n)
    for k in range(8):  # filter sweeps up, in steps crossfaded
        lo = 300 * 2 ** (k * 0.6)
        part = signal.sosfilt(signal.butter(2, (lo, min(lo * 3, 20000)), "bandpass", fs=SR, output="sos"), x)
        w = np.clip(1 - np.abs(t / length * 8 - k - 0.5) / 1.5, 0, 1)
        out += part * w
    out *= (t / length) ** 3
    out[-int(0.01 * SR):] *= np.linspace(1, 0, int(0.01 * SR))
    return out / np.abs(out).max()


def quietest(x: np.ndarray, t: float, reach: float = 0.25) -> float:
    """The quietest moment within reach s of t: a better place to cut."""
    env, h = envelope(x, 0.02, 0.005)
    a, b = max(0, int((t - reach) * SR / h)), min(len(env), int((t + reach) * SR / h) + 1)
    if t <= 0.01 or a >= b:
        return t
    return (a + int(np.argmin(env[a:b]))) * h / SR


def raw_part(name: str, k, seg: dict | None = None) -> np.ndarray:
    """Part k (an index into segments(), or (start, end) s) of a raw download, trimmed."""
    if not (RAW / name / "preview.ogg").exists():
        raise FileNotFoundError(name)
    x = highpass(read(name), 40)
    if isinstance(k, tuple):
        a, b = k
    else:
        a, b, _ = segments(x, **(seg or {}))[k]
    return trim(x[int(a * SR):int(b * SR)])


def build(only: list[str] | None = None) -> list[dict]:
    """Make every take in RECIPES and COMPOSITES; returns one record per file for the catalog."""
    import sound  # the cue table: groups and files
    for name, s in sound.SOUNDS.items():
        GROUP_OF[name] = s.group
    SOUND_DIR.mkdir(exist_ok=True)
    made = []
    for cue, src, opt in RECIPES:
        if only and cue not in only:
            continue
        if not (RAW / src / "preview.ogg").exists():
            print(f"{cue}: {src} not downloaded, skipped")
            continue
        x = highpass(read(src), opt.get("highpass", 40))
        if "lowpass" in opt:
            x = lowpass(x, opt["lowpass"])
        if "loop" in opt:
            takes = [make_loop(x, opt["loop"], opt.get("xfade", 3.0), opt.get("start", 0.0))]
        else:
            segs = segments(x, **opt.get("seg", {}))
            spans = []
            for p in opt.get("parts", range(len(segs))):
                if isinstance(p, tuple):
                    spans.append(p if opt.get("exact") else tuple(quietest(x, t) for t in p))
                elif p < len(segs):
                    spans.append(segs[p][:2])
                else:
                    print(f"{cue}: {src} has only {len(segs)} parts, no part {p}")
            takes = []
            for a, b in spans:
                y = repitch(x[int(a * SR):int(b * SR)], opt.get("pitch", 0))
                y = trim(y, fade_out=opt.get("fade", 0.04))
                if opt.get("min_len", 0) <= len(y) / SR <= opt.get("max_len", 99):
                    takes.append(y)
            if "best" in opt:
                takes = sorted(takes, key=lambda y: -loudness(y)[1])[:opt["best"]]
        for k, y in enumerate(takes):
            made.append(finish(cue, f"{src}{'_p%+d' % opt['pitch'] if opt.get('pitch') else ''}_{k + 1}", y,
                               opt.get("group"), opt.get("gain", 0.0), src, loop="loop" in opt))
    for cue, name, make in COMPOSITES:
        if only and cue not in only:
            continue
        try:
            y = make()
        except FileNotFoundError as e:
            print(f"{cue}: {name} needs a download that is missing ({e}), skipped")
            continue
        made.append(finish(cue, name, trim(y, fade_out=0.05) if cue != "swell" else y, None, 0.0, "made"))
    return made


def finish(cue: str, name: str, y: np.ndarray, group: str | None, gain: float, src: str, loop: bool = False
           ) -> dict:
    # An alternative (alt_<cue>_<what>) is made as loud as the cue it could replace.
    group = group or GROUP_OF.get(cue[4:].split("_")[0] if cue.startswith("alt_") else cue, "voice")
    target, how = TARGET[group]
    y, note = normalise(y, target + gain, how)
    path = SOUND_DIR / f"{cue}_{name}.wav"
    sf.write(path, y.astype(np.float32), SR, subtype="PCM_24")
    integ, mmax = loudness(y)
    rec = {"cue": cue, "file": path.name, "source": src, "seconds": round(len(y) / SR, 2),
           "lufs": round(mmax if how == "max" else integ, 1), "peak": round(20 * np.log10(np.abs(y).max()), 1),
           "note": note}
    if loop:
        rec["seam"] = round(seam_jump(y), 1)
    print(f"{path.name:44} {rec['seconds']:5.2f}s {rec['lufs']:6.1f} LUFS peak {rec['peak']:5.1f}"
          + (f" seam {rec['seam']}" if loop else "") + (f"  {note}" if note else ""))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--fetch", nargs="*", metavar="NAME", help="download missing sources (all, or these)")
    ap.add_argument("--info", nargs="*", metavar="NAME", help="numbers for the raw downloads (all, or these)")
    ap.add_argument("--only", nargs="+", metavar="CUE", help="remake only these cues")
    a = ap.parse_args()
    if a.fetch is not None:
        fetch(a.fetch or list(SOURCES))
        return 0
    if a.info is not None:
        info(a.info or sorted(SOURCES))
        return 0
    if a.only:  # remake some cues: delete just their old files
        for cue in a.only:
            for f in SOUND_DIR.glob(f"{cue}_*.wav"):
                f.unlink()
    else:
        for f in SOUND_DIR.glob("*.wav"):
            if not f.name.startswith("demo"):
                f.unlink()
    made = build(a.only)
    cat_file = SOUND_DIR / "catalog.json"
    old = json.loads(cat_file.read_text()) if a.only and cat_file.exists() else []
    cat = [r for r in old if r["cue"] not in a.only] + made if a.only else made
    cat_file.write_text(json.dumps(cat, indent=1))
    write_sources(cat)
    print(f"{len(made)} takes made")
    return 0


def write_sources(cat: list[dict]) -> None:
    """sounds/sources.md: every recording used, with its page, author and licence."""
    used: dict[str, list[str]] = {}
    for r in cat:
        if r["source"] != "made":
            used.setdefault(r["source"], []).append(r["cue"])
    lines = ["# Where the sounds come from", "",
             "Made by scripts/make_sounds.py from these Freesound recordings (the public high-quality",
             "previews), plus sounds synthesised in that script (the music box tunes, the choir chords,",
             "the shimmer, the booms, the drone and the swell). Thank you to everyone who shared them.",
             "", "| Recording | Author | Licence | Used for |", "|---|---|---|---|"]
    for src in sorted(used):
        m = json.loads((RAW / src / "meta.json").read_text())
        cues = sorted(set(used[src]))
        lines.append(f"| [{m['title']}]({m['page']}) | {m['author']} | {m['licence']} | {', '.join(cues)} |")
    lines += ["", "CC BY sounds need the author credited wherever the project is shown; the table above does",
              "that. CC BY-NC ones may not be used commercially.", ""]
    (SOUND_DIR / "sources.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
