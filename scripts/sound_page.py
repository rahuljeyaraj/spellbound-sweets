"""The listening page: sounds/preview.html, every sound with a play button, the demo round with
what happens when, and the big choices side by side.

    uv run python scripts/sound_page.py

Run scripts/make_sounds.py first. This renders the demo round and the comparison clips with the
real mixer (sound.py, on a fake clock), makes small MP3 copies of everything in sounds/web/ for
the browser, and writes the page from sound.py's SOUNDS table, so the page always says what the
game really does. Open sounds/preview.html in a browser.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

import sound
from sound import SOUNDS, SOUND_DIR, SR, Bank, FakeAxis, demo_round, render

WEB = SOUND_DIR / "web"

# What each sound is called on the page, and the face mood it belongs to (s w n c) for its colour.
LABELS = {
    "snore": ("Snore", "s"), "snort": ("Snort", "s"), "wake": ("Wakes up", "w"), "settle": ("Falls asleep", "s"),
    "breath": ("Breath", "w"), "growl": ("Growl: monster", "n"), "beast": ("Growl: animal, pitched down", "n"),
    "rumble": ("Growl: rumble underneath", "n"), "hiss": ("Hiss", "n"), "grumble": ("Grumble", "n"),
    "cackle": ("Cackle", "n"), "sob": ("Sob", "c"), "wail": ("Wail", "c"),
    "whoosh": ("Whoosh", None), "scrape": ("Scrape", None), "grind": ("Stone grind", "c"), "creak": ("Creak", "c"),
    "spell": ("Spell sting", "c"), "musicbox": ("Music box", "c"), "swell": ("Swell", "c"),
    "restart": ("Restart sting", None), "awake": ("The bowl is awake", None),
    "wind": ("Wind", None), "night": ("Night", None), "howl": ("Wolf", None), "bats": ("Bats", None),
    "crow": ("Crow", None), "owl": ("Owl", None), "thunder": ("Thunder", None), "door": ("Door", None),
    "chains": ("Chains", None),
}
GROUPS = [("voice", "The skull's voice"), ("slide", "The bowl sliding"), ("music", "The spell"),
          ("background", "The night around it"), ("spot", "Now and then, far off")]
MOOD_NAMES = {"s": "sleeping", "w": "watching", "n": "snarl", "c": "crying"}


def mp3(audio: np.ndarray, name: str, longest: float | None = None) -> str:
    """A small copy for the browser; long beds are cut, with a fade, to keep the page light."""
    if audio.ndim == 2 and audio.shape[1] == 2 and np.allclose(audio[:, 0], audio[:, 1]):
        audio = audio[:, 0]
    if longest and len(audio) > longest * SR:
        audio = audio[:int(longest * SR)].copy()
        fade = int(2 * SR)
        audio[-fade:] = (audio[-fade:].T * np.linspace(1, 0, fade)).T
    WEB.mkdir(parents=True, exist_ok=True)
    path = WEB / f"{name}.mp3"
    sf.write(path, audio, SR, format="MP3")
    return f"web/{path.name}"


def take_files(pattern: str) -> list[Path]:
    return sorted(SOUND_DIR.glob(pattern))


def clip_list(pattern: str) -> list[dict]:
    """Every file matching pattern, as page clips."""
    out = []
    for f in take_files(pattern):
        x, _ = sf.read(f, dtype="float32")
        out.append({"src": mp3(x, f.stem, longest=30), "seconds": round(len(x) / SR, 1), "name": f.stem})
    return out


# ---- the comparison clips ----

def hand_sweep(snd, now: float) -> None:
    """A hand comes in high and far, sweeps down to the bowl, rises close to it, shakes, and goes."""
    t = now - 0.6
    if 0 <= t < 9.9:
        if t < 4.5:
            f = t / 4.5
            p = (420 - 330 * f, 140 - 100 * f, 360 - 300 * f)
        elif t < 6.5:
            p = (90, 40, 60 + 260 * (t - 4.5) / 2)
        elif t < 8.0:
            p = (90 + 60 * math.sin(t * 25), 40, 320 + 30 * math.sin(t * 19))
        else:
            f = (t - 8.0) / 1.9
            p = (90 + 400 * f, 40 + 100 * f, 320)
        snd.frame(now, "w", np.array([p], float))
    else:
        snd.frame(now, "w", None)


def three_dodges():
    axis = FakeAxis(5.0)
    done = set()

    def script(snd, now: float) -> None:
        for k, t in enumerate((0.5, 2.6, 4.7)):
            if now >= t and k not in done:
                done.add(k)
                axis.move(215.0 if axis.y(now) < 110 else 5.0, 700, 7000, now)
                snd.move("dodge", axis)
        snd.frame(now, "n" if 0.5 <= now < 6.2 else "w", None)
    return script


def comparisons() -> list[dict]:
    choices = []
    beds = {"wind", "night"}
    hand = []
    for voice in sound.INSTRUMENTS:
        audio = render(hand_sweep, 12.0, hand_voice=voice, bank=Bank(keep=beds))
        hand.append({"label": voice.capitalize(), "src": mp3(audio, f"choice_hand_{voice}"),
                     "inuse": voice == sound.HAND["voice"],
                     "note": {"theremin": "A sine with a little edge and a singer's vibrato. Clear on a small speaker.",
                              "glass": "Two glassy tones beating slowly against each other. Softer, eerier.",
                              "choir": "A low choir whose 'oo' opens to 'ah' as the hand comes close."}[voice]})
    choices.append({"title": "The hand tone", "question": "Which voice should follow the hand?",
                    "how": "A hand comes in high and far away, sweeps down to the bowl, rises close to it, shakes, "
                           "and leaves. Listen to the wind rise as it comes near.",
                    "swap": "sound.py, HAND: voice", "options": hand})

    snarl_kinds = [("Layered", {"growl", "beast", "rumble", "hiss"}, SOUNDS,
                    "A monster growl, an animal growl pitched down and a rumble together, sometimes a hiss."),
                   ("Monster only", {"growl", "hiss"}, SOUNDS, "Just the monster growls."),
                   ("Animal only", {"beast", "rumble", "hiss"},
                    {**SOUNDS, "beast": dataclasses.replace(SOUNDS["beast"], db=0)},
                    "Just the animals (tiger, dog, wolf), pitched down to sound bigger.")]
    opts = []
    for label, keep, table, note in snarl_kinds:
        audio = render(three_dodges(), 7.5, bank=Bank(sounds=table, keep=keep | {"whoosh", "scrape"}))
        opts.append({"label": label, "src": mp3(audio, f"choice_snarl_{label.split()[0].lower()}"),
                     "inuse": label == "Layered", "note": note})
    choices.append({"title": "The snarl", "question": "Which growl on a dodge?",
                    "how": "Three dodges, two seconds apart, with their whooshes.",
                    "swap": "sound.py, SOUNDS: growl, beast, rumble", "options": opts})

    def files_choice(title, question, how, swap, groups):
        options = []
        for label, pattern, inuse, note in groups:
            clips = clip_list(pattern)
            if clips:
                options.append({"label": label, "clips": clips, "inuse": inuse, "note": note})
        choices.append({"title": title, "question": question, "how": how, "swap": swap, "options": options})

    files_choice("The cackle", "Whose laugh?", "Each button is one laugh.", "sound.py, SOUNDS: cackle files",
                 [("Deep evil laughs", "cackle_*.wav", True, "Men's evil laughs, pitched down a little."),
                  ("Witch cackles", "alt_cackle_witch_*.wav", False, "Higher, quicker witch cackles.")])
    files_choice("The wind", "Which wind under everything?", "The first 30 s of each loop.",
                 "sound.py, SOUNDS: wind files",
                 [("Spooky wind", "wind_*.wav", True, "Low, moaning gusts. 62 s loop."),
                  ("Castle wind", "alt_wind_castle_*.wav", False, "Wind gusting through an old building. 70 s loop."),
                  ("Night wind", "alt_wind_night_*.wav", False, "Howling night wind, a little thinner. 50 s loop."),
                  ("Tonal wind", "alt_wind_tonal_*.wav", False, "A soft whistling wind, very quiet. 44 s loop.")])
    files_choice("The night", "Which night sounds, quietly under the wind?", "The first 30 s of each loop.",
                 "sound.py, SOUNDS: night files",
                 [("Night wildlife", "night_*.wav", True, "Insects and far-off animals."),
                  ("Crickets", "alt_night_crickets_*.wav", False, "Crickets on a porch.")])
    files_choice("The music box", "Which warning tune before the game restarts?",
                 "It plays over the last 5 s of the 10 s to take a sweet, slowing down like it is winding down.",
                 "sound.py, SOUNDS: musicbox files",
                 [("Tune one and tune two", "musicbox_*.wav", True,
                   "Two short tunes written for the bowl, played by a synthesised music box. It takes turns."),
                  ("A recorded lullaby", "alt_musicbox_lullaby_*.wav", False,
                   "A real music box recording; its timing can't be lined up with the restart.")])
    files_choice("The restart", "What says the game is back on?", "The sting when dodging starts again. A cackle follows it.",
                 "sound.py, SOUNDS: restart files",
                 [("Hits with a boom", "restart_*.wav", True, "A short hit, braam or timpani with a low boom."),
                  ("Whoosh into a braam", "alt_restart_whoosh_*.wav", False, "Longer: a rush, then the hit.")])
    files_choice("The start-up", "What says the bowl is awake?", "Plays once, when dodge.py is ready.",
                 "sound.py, SOUNDS: awake files",
                 [("A bell and a drone", "awake_*.wav", True, "A tolling bell over a low swell."),
                  ("Drone and boom only", "alt_awake_*.wav", False, "No bell.")])
    return choices


# ---- the page ----

def build() -> None:
    moods: list[tuple[float, str]] = []
    frame = sound.Sound.frame

    def watch(self, now, mood, palms, in_bowl=False):
        if not moods or moods[-1][1] != mood:
            moods.append((round(now, 2), mood))
        return frame(self, now, mood, palms, in_bowl)

    sound.Sound.frame = watch
    script, seconds, timeline = demo_round()
    audio = render(script, seconds)
    sound.Sound.frame = frame
    sound.write(SOUND_DIR / "demo_round.wav", audio)
    demo = {"src": mp3(audio, "demo_round"), "seconds": seconds, "events": timeline,
            "moods": [(t, moods[i + 1][0] if i + 1 < len(moods) else seconds, m) for i, (t, m) in enumerate(moods)]}

    groups = []
    for g, title in GROUPS:
        cues = []
        for name, s in SOUNDS.items():
            if s.group != g:
                continue
            label, mood = LABELS.get(name, (name, None))
            cues.append({"name": name, "label": label, "mood": mood, "when": s.when, "clips": clip_list(s.files)})
        groups.append({"title": title, "cues": cues})
    data = {"demo": demo, "choices": comparisons(), "groups": groups,
            "hand": dict(sound.HAND), "timing": dict(sound.TIMING)}
    page = PAGE.replace("/*DATA*/", json.dumps(data))
    (SOUND_DIR / "preview.html").write_text(page, encoding="utf-8")
    files = sorted(WEB.glob("*.mp3"))
    print(f"sounds/preview.html: {sum(len(c['clips']) for g in groups for c in g['cues'])} takes, "
          f"{len(data['choices'])} choices, {len(files)} mp3s, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB")


PAGE = r"""<title>Spellbound Sweets Sounds</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Pirata+One&family=Chakra+Petch:wght@400;600&display=swap">
<style>
/* Layout: the face preview's dark bench, one column: the demo round with its timeline first,
   then the choices, then every sound by what it belongs to. Single dark look on purpose. */
:root {
  --bg: #0d0b0a;
  --wood: #2a1a10;
  --wood-hi: #3a2618;
  --ink: #efe3d4;
  --muted: #a8968a;
  --accent: #ff3a12;
  --line: #3a2a1e;
  --sleep: #6a8cff;
  --watch: #ffa02a;
  --snarl: #ff4a26;
  --cry: #2fd0f5;
  --display: "Pirata One", "Georgia", serif;
  --ui: "Chakra Petch", "Segoe UI", system-ui, sans-serif;
  color-scheme: dark;
}
body {
  background: var(--bg); color: var(--ink); font-family: var(--ui); font-size: 16px; line-height: 1.45;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 20px 40px;
}
.wrap { max-width: 820px; margin: 0 auto; display: grid; gap: 28px; }
h1 { font-family: var(--display); font-weight: 400; font-size: 2.6rem; margin: 0; color: var(--accent); line-height: 1; letter-spacing: .02em; }
h2 { font-family: var(--display); font-weight: 400; font-size: 1.8rem; margin: 0; color: var(--ink); text-wrap: balance; }
h3 { font-size: 1.05rem; margin: 0; font-weight: 600; }
p { margin: 0; }
.sub { color: var(--muted); max-width: 62ch; margin-top: 6px; }
section { display: grid; gap: 12px; }
.stage { background: var(--wood); border: 1px solid var(--line); border-radius: 6px; padding: 14px; display: grid; gap: 12px; }
button { font: inherit; color: inherit; cursor: pointer; }
button:focus-visible { outline: 2px solid var(--watch); outline-offset: 2px; }
.player { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.big { width: 58px; height: 58px; border-radius: 50%; border: 0; background: var(--accent); color: #1a0d07; font-size: 1.5rem; flex: none; }
.clock { font-variant-numeric: tabular-nums; color: var(--muted); }
.now { font-weight: 600; min-width: 0; flex: 1; }
.bar { position: relative; height: 34px; border-radius: 4px; overflow: hidden; background: #000; cursor: pointer; touch-action: none; }
.band { position: absolute; top: 0; bottom: 0; opacity: .55; }
.tick { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--ink); opacity: .7; }
.head { position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; box-shadow: 0 0 6px #fff; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--muted); font-size: .9rem; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; background: var(--c); vertical-align: -1px; }
.events { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; max-height: 340px; overflow-y: auto; }
.events button { width: 100%; text-align: left; background: none; border: 0; padding: 5px 8px; border-radius: 4px; display: grid; grid-template-columns: 4.2em 1fr; gap: 8px; }
.events button:hover { background: var(--wood-hi); }
.events .t { font-variant-numeric: tabular-nums; color: var(--muted); }
.events .on { background: var(--wood-hi); color: var(--watch); }
.cards { display: grid; gap: 14px; }
.card { background: var(--wood); border: 1px solid var(--line); border-radius: 6px; padding: 14px; display: grid; gap: 10px; }
.card .q { color: var(--muted); }
.opt { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; align-items: start; padding: 8px 0; border-top: 1px solid var(--line); }
.opt .name { font-weight: 600; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; min-width: 0; }
.opt .note { grid-column: 2; color: var(--muted); font-size: .92rem; min-width: 0; }
.inuse { font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; color: #1a0d07; background: var(--watch); border-radius: 3px; padding: 1px 6px; }
.swap { font-size: .85rem; color: var(--muted); }
.swap code { color: var(--ink); }
.play { width: 38px; height: 38px; border-radius: 50%; border: 1px solid var(--line); background: var(--wood-hi); flex: none; }
.takes { display: flex; flex-wrap: wrap; gap: 6px; grid-column: 2; }
.takes button { min-width: 36px; height: 32px; border-radius: 16px; border: 1px solid var(--line); background: var(--wood-hi); font-variant-numeric: tabular-nums; padding: 0 10px; }
.playing { background: var(--accent) !important; color: #1a0d07; border-color: var(--accent) !important; }
.cue { display: grid; grid-template-columns: 10px 1fr; gap: 4px 12px; padding: 10px 0; border-top: 1px solid var(--line); }
.cue .dot { width: 10px; height: 10px; border-radius: 50%; margin-top: 6px; background: var(--c, var(--line)); }
.cue .when { color: var(--muted); font-size: .92rem; grid-column: 2; }
.cue .takes { grid-column: 2; }
.none { color: var(--muted); font-style: italic; grid-column: 2; }
footer { color: var(--muted); font-size: .9rem; }
@media (prefers-reduced-motion: reduce) { .head { box-shadow: none; } }
</style>

<div class="wrap">
  <header>
    <h1>Spellbound Sweets Sounds</h1>
    <p class="sub">What the haunted bowl sounds like. Every sound here is what the game plays. Start with the demo round, then pick the ones you like below.</p>
  </header>

  <section>
    <h2>The demo round</h2>
    <p class="sub">90 seconds of a pretend game: the skull asleep, a hand creeping in, dodges and a feint, a hand in the bowl, the horns, the slide, crying, the music box, and the restart. Tap the bar or a line to jump there.</p>
    <div class="stage">
      <div class="player">
        <button class="big" id="demo-play" aria-label="Play the demo round">&#9654;</button>
        <span class="clock" id="demo-clock">0:00 / 1:30</span>
        <span class="now" id="demo-now"></span>
      </div>
      <div class="bar" id="demo-bar" role="slider" aria-label="Demo round position" tabindex="0"></div>
      <div class="legend" id="legend"></div>
      <ol class="events" id="events"></ol>
    </div>
  </section>

  <section>
    <h2>Choices for you</h2>
    <p class="sub">The ones marked in use are playing now. To swap, change the one line named under each.</p>
    <div class="cards" id="choices"></div>
  </section>

  <section>
    <h2>Every sound</h2>
    <p class="sub">Each number is one take. The game picks a different take each time, never the same one twice in a row, with a little change of pitch and level. The dot is the face's mood colour when it plays.</p>
    <div class="cards" id="groups"></div>
  </section>

  <footer>Sounds from Freesound and made in scripts/make_sounds.py. Authors and licences: sounds/sources.md.</footer>
</div>
<audio id="clip" preload="none"></audio>
<audio id="demo" preload="metadata"></audio>

<script>
const DATA = /*DATA*/;
const MOOD = { s: ["sleeping", "var(--sleep)"], w: ["watching", "var(--watch)"], n: ["snarl", "var(--snarl)"], c: ["crying", "var(--cry)"] };
const $ = id => document.getElementById(id);
const el = (tag, attrs = {}, kids = []) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") e.textContent = v; else if (k === "style") e.style.cssText = v; else e.setAttribute(k, v);
  }
  for (const k of [].concat(kids)) if (k) e.append(k);
  return e;
};
const mmss = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

// One clip at a time; the demo pauses while a clip plays and the other way round.
const clip = $("clip"), demo = $("demo");
let clipBtn = null;
function stopClip() { clip.pause(); if (clipBtn) clipBtn.classList.remove("playing"); clipBtn = null; }
function playClip(src, btn) {
  if (clipBtn === btn) { stopClip(); return; }
  stopClip(); demo.pause();
  clip.src = src; clip.currentTime = 0;
  clip.play().catch(() => {});
  clipBtn = btn; btn.classList.add("playing");
}
clip.addEventListener("ended", stopClip);

// The demo round.
const D = DATA.demo;
demo.src = D.src;
const bar = $("demo-bar");
for (const [t0, t1, m] of D.moods) bar.append(el("div", { class: "band", style: `left:${t0 / D.seconds * 100}%;width:${(t1 - t0) / D.seconds * 100}%;background:${MOOD[m][1]}` }));
const head = el("div", { class: "head", style: "left:0" });
bar.append(head);
for (const [k, [name, c]] of Object.entries(MOOD)) $("legend").append(el("span", { style: `--c:${c}`, text: `face ${name}` }));
const rows = D.events.map(([t, text]) => {
  bar.append(el("div", { class: "tick", style: `left:${t / D.seconds * 100}%` }));
  const b = el("button", {}, [el("span", { class: "t", text: mmss(t) }), el("span", { text: text })]);
  b.addEventListener("click", () => seek(t, true));
  const li = el("li", {}, b);
  $("events").append(li);
  return [t, b, text];
});
function seek(t, play) { demo.currentTime = Math.max(0, t - 0.5); if (play) start(); show(); }
function start() { stopClip(); demo.play().catch(() => {}); }
$("demo-play").addEventListener("click", () => demo.paused ? start() : demo.pause());
demo.addEventListener("play", () => { $("demo-play").innerHTML = "&#10074;&#10074;"; $("demo-play").setAttribute("aria-label", "Pause"); });
demo.addEventListener("pause", () => { $("demo-play").innerHTML = "&#9654;"; $("demo-play").setAttribute("aria-label", "Play the demo round"); });
function show() {
  const t = demo.currentTime || 0;
  head.style.left = `${t / D.seconds * 100}%`;
  $("demo-clock").textContent = `${mmss(t)} / ${mmss(D.seconds)}`;
  let cur = null;
  for (const r of rows) { r[1].classList.remove("on"); if (r[0] <= t + 0.05) cur = r; }
  if (cur) { cur[1].classList.add("on"); $("demo-now").textContent = cur[2]; }
  else $("demo-now").textContent = "the skull is asleep";
}
demo.addEventListener("timeupdate", show);
const at = e => { const r = bar.getBoundingClientRect(); return Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * D.seconds; };
bar.addEventListener("pointerdown", e => seek(at(e) + 0.5, true));
bar.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") seek(demo.currentTime + 5.5, false);
  if (e.key === "ArrowLeft") seek(demo.currentTime - 4.5, false);
});
show();

// The choices.
for (const c of DATA.choices) {
  const card = el("div", { class: "card" }, [el("h3", { text: c.title }), el("p", { class: "q", text: `${c.question} ${c.how}` })]);
  for (const o of c.options) {
    const name = el("div", { class: "name" }, [el("span", { text: o.label }), o.inuse ? el("span", { class: "inuse", text: "in use" }) : null]);
    let lead;
    if (o.src) {
      lead = el("button", { class: "play", "aria-label": `Play ${c.title}: ${o.label}`, text: "▶" });
      lead.addEventListener("click", () => playClip(o.src, lead));
    } else lead = el("span");
    const opt = el("div", { class: "opt" }, [lead, name, el("p", { class: "note", text: o.note })]);
    if (o.clips) {
      const takes = el("div", { class: "takes" });
      o.clips.forEach((k, i) => {
        const b = el("button", { "aria-label": `Play ${o.label} ${i + 1}`, title: `${k.name}, ${k.seconds} s`, text: String(i + 1) });
        b.addEventListener("click", () => playClip(k.src, b));
        takes.append(b);
      });
      opt.append(takes);
    }
    card.append(opt);
  }
  card.append(el("p", { class: "swap" }, [document.createTextNode("To swap: "), el("code", { text: c.swap })]));
  $("choices").append(card);
}

// Every sound.
for (const g of DATA.groups) {
  const card = el("div", { class: "card" }, [el("h3", { text: g.title })]);
  for (const c of g.cues) {
    const takes = el("div", { class: "takes" });
    c.clips.forEach((k, i) => {
      const b = el("button", { "aria-label": `Play ${c.label} ${i + 1}`, title: `${k.name}, ${k.seconds} s`, text: String(i + 1) });
      b.addEventListener("click", () => playClip(k.src, b));
      takes.append(b);
    });
    const colour = c.mood ? `--c:${MOOD[c.mood][1]}` : "";
    card.append(el("div", { class: "cue" }, [
      el("span", { class: "dot", style: colour }), el("h3", { text: c.label }),
      el("p", { class: "when", text: c.when }),
      c.clips.length ? takes : el("p", { class: "none", text: "No files yet." })]));
  }
  $("groups").append(card);
}
</script>
"""


if __name__ == "__main__":
    build()
