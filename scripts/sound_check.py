"""Checks on the sound set and the mixer, with nobody listening.

    uv run python scripts/sound_check.py

Run after scripts/make_sounds.py. Prints PASS or FAIL for each:
- every take at its group's loudness (unless make_sounds.py had to settle for quieter)
- every take starts and ends at silence, and every loop's seam is no bigger a step than the
  loop's own ordinary steps (no clicks)
- the demo round, after the limiter, never above -1 dBFS, also between samples (true peak)
- every whoosh starts with its move, and every snore's loudest moment falls on the mouth's
- the skull never laughs while crying, and no growl starts over a laugh
- a hand held still on a noisy camera keeps its note
- mixing is fast enough: well over real time
"""

from __future__ import annotations

import json
import time

import numpy as np
import soundfile as sf
from scipy import signal

import make_sounds as ms
import sound
from sound import SOUND_DIR, SR, BLOCK

results: list[bool] = []


def check(ok: bool, text: str) -> None:
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + text)


def main() -> int:
    cat = json.loads((SOUND_DIR / "catalog.json").read_text())
    group = {**{n: s.group for n, s in sound.SOUNDS.items()}}

    # Loudness of every take against its group's target.
    off = []
    for r in cat:
        base = r["cue"][4:].split("_")[0] if r["cue"].startswith("alt_") else r["cue"]
        target = ms.TARGET[group.get(base, "voice")][0]
        if "under target" not in r["note"] and abs(r["lufs"] - target) > 0.6:
            off.append(f"{r['file']} {r['lufs']} vs {target}")
    check(not off, f"{len(cat)} takes at their group's loudness" + (": " + "; ".join(off[:5]) if off else ""))
    quiet = [r["file"] for r in cat if "under target" in r["note"]]
    print(f"      ({len(quiet)} settled quieter than the target to keep their peaks: {', '.join(quiet) or 'none'})")

    # Ends and seams.
    bad_ends, bad_seams = [], []
    for r in cat:
        x, _ = sf.read(SOUND_DIR / r["file"], dtype="float32")
        if "seam" in r:
            d = np.abs(np.diff(x))
            if abs(x[0] - x[-1]) > np.percentile(d, 99.9):
                bad_seams.append(r["file"])
            # and no jump in level across the seam: the 50 ms either side within 3 dB
            w = int(0.05 * SR)
            a, b = np.sqrt(np.mean(x[-w:] ** 2)), np.sqrt(np.mean(x[:w] ** 2))
            if abs(20 * np.log10((a + 1e-9) / (b + 1e-9))) > 3:
                bad_seams.append(r["file"] + " (level)")
        elif max(abs(x[0]), abs(x[-1])) > 2e-3:
            bad_ends.append(f"{r['file']} {x[0]:+.4f}/{x[-1]:+.4f}")
    check(not bad_ends, "every one-shot starts and ends at silence" + (": " + "; ".join(bad_ends[:5]) if bad_ends else ""))
    loops = [r for r in cat if "seam" in r]
    check(not bad_seams, f"{len(loops)} loops join without a click" + (": " + ", ".join(bad_seams) if bad_seams else ""))

    # The demo round through the real mixer.
    script, seconds, timeline = sound.demo_round()
    bank = sound.Bank()  # loaded first: the start-up load is not part of the mixing time
    t0 = time.perf_counter()
    audio = sound.render(script, seconds, bank=bank)
    took = time.perf_counter() - t0
    log = sound.render.log
    peak = 20 * np.log10(np.abs(audio).max())
    true_peak = 20 * np.log10(np.abs(signal.resample_poly(audio, 4, 1, axis=0)).max())
    check(peak <= -1.0, f"demo round sample peak {peak:.2f} dBFS (at most -1)")
    check(true_peak <= -1.0, f"demo round true peak {true_peak:.2f} dBFS (at most -1)")
    integ, mmax = ms.loudness(audio.mean(axis=1))
    print(f"      demo round loudness {integ:.1f} LUFS, loudest 400 ms {mmax:.1f} LUFS")

    moves = [t for t, text in timeline if text.split(":")[0] in ("dodge", "feint") or text.startswith(
        ("the feint darts back", "on to an end", "slow stone slide"))]
    starts = [t for t, name, _, _ in log if name in ("whoosh", "grind")]
    gaps = [min(abs(s - (m + sound.FakeAxis.LAG)) for s in starts) for m in moves]
    check(max(gaps) < 0.03, f"{len(moves)} moves each have a whoosh or grind starting with the bed "
                            f"(worst {1000 * max(gaps):.0f} ms off)")

    # The laugh: never while it cries (from the horns to the restart's own laugh), and no growl
    # starts on top of one.
    horns = next(t for t, text in timeline if text.startswith("the horns"))
    again = next(t for t, text in timeline if text.startswith("10 s up"))
    lengths = {t.path.stem: t.length for t in bank.takes["cackle"]}
    laughs = [(t, t + lengths[stem]) for t, name, stem, _ in log if name == "cackle"]
    crying = [round(a, 2) for a, _ in laughs if horns <= a < again - 0.01]
    over = [round(t, 2) for t, name, _, _ in log if name in ("growl", "beast", "rumble")
            and any(a <= t < b for a, b in laughs)]
    check(not crying and not over, f"{len(laughs)} laughs: none while crying{' (' + str(crying) + ')' if crying else ''}, "
                                   f"no growl over one{' (' + str(over) + ')' if over else ''}")

    # Snores: loudest moment on the mouth's widest, every 3 s from falling asleep.
    mixer = sound.Mixer(sound.Bank(keep={"snore", "snort"}), seed=2)
    mixer.log = []
    mixer.post("mood", 0.0, mood="s")
    while mixer.n < 40 * SR:
        mixer.render(BLOCK, mixer.n / SR)
    takes = {t.path.stem: t for name in ("snore", "snort") for t in mixer.bank.takes[name]}
    offs = []
    for t, name, stem, rate in mixer.log:
        peak_at = t + takes[stem].peak / rate
        k = round((peak_at - sound.TIMING["snore_peak"]) / sound.TIMING["snore_period"])
        offs.append(peak_at - (sound.TIMING["snore_peak"] + k * sound.TIMING["snore_period"]))
    check(max(abs(o) for o in offs) < 0.06, f"{len(offs)} snores land on the mouth (worst "
                                            f"{1000 * max(abs(o) for o in offs):.0f} ms off)")

    # A hand held still, its height noisy as the closer cameras make it.
    m = sound.Mixer(sound.Bank(keep=set()), seed=1)
    snd = sound.Sound(m)
    rng = np.random.default_rng(5)
    notes, nf = [], 0.0
    while m.n < 8 * SR:
        now = m.n / SR
        while nf <= now + BLOCK / SR:
            h = sound.HAND["low"] + (sound.HAND["high"] - sound.HAND["low"]) * 0.45  # right between two notes
            snd.frame(nf, "w", (np.array([200, 30, h]) + rng.normal(0, (5, 5, 12)))[None, :])
            nf += 1 / 30
        m.render(BLOCK, now)
        if now > 2:
            notes.append(m.hand.note)
    changes = sum(a != b for a, b in zip(notes, notes[1:]))
    check(changes <= 2, f"a hand held still between two notes, height noise 12 mm: {changes} note changes in 6 s")

    check(seconds / took > 4, f"mixing runs {seconds / took:.1f} times faster than real time (needs well over 1)")
    print(f"\n{sum(results)} of {len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
