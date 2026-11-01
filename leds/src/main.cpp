// Face LEDs on one WS2812 chain on D10. The four moods from face/face_preview.html: sleeping
// (waiting), watching, snarl, crying (gives up).
//
// scripts/dodge.py sets the mood over USB serial, one line at a time: a mood letter, s w n c,
// then optionally where the hand is, -100 (the face's far left) to 100 (far right), e.g. "w-35".
// With a hand the eyes follow it; without one they wander. Until the first line arrives the
// face loops the moods, five seconds each, as a test; if the lines stop for STALE_MS it sleeps.
#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <math.h>

const int DATA_PIN = D10;    // GPIO3
const int N = 160;
const float BRIGHT = 1.0;    // full, as the preview now starts
const float LIMIT_MA = 1000; // scale the whole frame down if it would draw more than this
const float MOOD_S = 5;
const uint32_t STALE_MS = 2000;

// From the serial lines: the mood asked for (-1 before the first), when it came, and the hand.
int cmdMood = -1;
uint32_t cmdAt = 0;
bool hasHand = false;
float handX = 0;   // -1 far left to 1 far right, as the face sees it

// Chain order as wired, seen from the front (Rahul's drawing, 2026-10-31):
//   0-15    right-hand ring, starts at 9 o'clock, runs clockwise
//   16-31   left-hand ring, starts at 3 o'clock, runs clockwise
//   32-95   left mouth board, rows top to bottom, each row left to right
//   96-159  right mouth board, same
// Eye positions k count clockwise from 12 o'clock, as in face/face_preview.html.
const int RING_N = 16;
enum Eye { LEFT = 0, RIGHT = 1 };
const int RING_START[2] = { 16, 0 };    // chain index of each ring's first LED
const int RING_FIRST_K[2] = { 4, 12 };  // where that first LED sits: 3 o'clock, 9 o'clock
const int MOUTH_START = 32;

Adafruit_NeoPixel strip(N, DATA_PIN, NEO_GRB + NEO_KHZ800);

int eyeLed(Eye e, int k) {
  return RING_START[e] + ((k - RING_FIRST_K[e]) % RING_N + RING_N) % RING_N;
}

// x 0-15 left to right, y 0-7 top to bottom
int mouthLed(int x, int y) {
  return MOUTH_START + (x / 8) * 64 + y * 8 + (x % 8);
}

struct Rgb { float r, g, b; };
const Rgb RED = { 255, 20, 0 }, TOOTH = { 255, 215, 170 }, PURPLE = { 150, 0, 255 }, AMBER = { 255, 140, 0 };
const Rgb BLUE = { 20, 60, 255 }, CYAN = { 0, 210, 255 }, TEAR = { 160, 235, 255 };
const Rgb YELLOW = { 255, 230, 0 }, BLOOD = { 255, 0, 0 };

float rnd() { return random(10000) / 10000.0f; }
float wave(float t, float period) { return 0.5f + 0.5f * sinf(t * 2 * PI / period); }

// ---- mouth pictures, 16 wide by 8 tall ----

typedef char Pic[8][17];

// The snarl mouth. o = lip, T = tooth.
const char *OPEN[8] = {
  ".oooooooooooooo.",
  "oTTT.TTT.TTT.TTo",
  "o.T...T...T...To",
  "o..............o",
  "o..............o",
  "o..T...T...T...o",
  "o.TTT.TTT.TTT..o",
  ".oooooooooooooo." };
// Sad mouth, # = line
const char *FROWN[8] = {
  "................",
  "................",
  "................",
  ".....######.....",
  "...##......##...",
  "..#..........#..",
  ".#............#.",
  "................" };

// Place the upper jaw at row topY and the lower jaw at row botY. Where they overlap the teeth interlock.
void jaw(Pic g, int topY, int botY) {
  for (int r = 0; r < 8; r++) { memset(g[r], '.', 16); g[r][16] = 0; }
  auto put = [&](int r, const char *src) {
    if (r < 0 || r > 7) return;
    for (int x = 0; x < 16; x++) {
      char c = src[x];
      if (c == 'T' || (c == 'o' && g[r][x] != 'T')) g[r][x] = c;
    }
  };
  for (int i = 0; i < 3; i++) put(topY + i, OPEN[i]);
  for (int i = 0; i < 3; i++) put(botY + i, OPEN[5 + i]);
  for (int r = topY + 3; r < botY; r++) put(r, "o..............o");
}
Pic SHUT, AJAR, WIDE;

void setPx(Pic g, int x, int y, char ch) { if (y >= 0 && y < 8 && x >= 0 && x < 16) g[y][x] = ch; }

// ---- targets for this frame, and the colours shown now, eased toward them ----

float eyeT[2][RING_N][3], mouthT[128][3];
float eyeNow[2][RING_N][3], mouthNow[128][3];

void addRing(int e, int k, Rgb c, float v) {
  k = ((k % RING_N) + RING_N) % RING_N;
  eyeT[e][k][0] += c.r * v; eyeT[e][k][1] += c.g * v; eyeT[e][k][2] += c.b * v;
}

// keep the pupil off the brow at the top of the ring
float underBrow(float p) {
  if (p > 4.5f && p < 11.5f) return p;
  if (p >= 11.5f) return (p - 11.5f < 16 - p + 4.5f) ? 11.5f : 4.5f;
  return (4.5f - p < p + 16 - 11.5f) ? 4.5f : 11.5f;
}

enum Brow { FLAT, ANGRY, SAD };
enum Gaze { WANDER, DOWN };
struct EyeSpec { Rgb c; Brow brow; Rgb browC; float browV; Gaze gaze; };

const float PUPIL_SIDE = 0.3f;   // the pupil is three LEDs: the centre one bright, these two either side dimmer

// Only the brow and the pupil light; the rest of the ring stays off.
void openEye(int e, const EyeSpec &s, float t) {
  int side = e == LEFT ? -1 : 1;
  // angry brows slope down toward the nose, sad ones down toward the outside, flat ones sit level on top
  static const int UPPER_RIGHT[4] = { 0, 1, 2, 3 }, UPPER_LEFT[4] = { 13, 14, 15, 0 }, FLAT_B[5] = { 14, 15, 0, 1, 2 };
  bool isBrow[RING_N] = {};
  if (s.brow == FLAT) for (int k : FLAT_B) isBrow[k] = true;
  else {
    bool right = (s.brow == ANGRY) == (side < 0);   // the left eye's nose side is its upper right
    for (int i = 0; i < 4; i++) isBrow[right ? UPPER_RIGHT[i] : UPPER_LEFT[i]] = true;
  }
  for (int k = 0; k < RING_N; k++) if (isBrow[k]) addRing(e, k, s.browC, s.browV);

  float ang;                                   // screen angle the pupil points to
  if (s.gaze == DOWN) ang = PI / 2 + 0.2f * sinf(t * 0.8f);
  else if (hasHand) ang = PI / 2 - handX * PI / 2;   // 3 o'clock for far right, 9 for far left
  else ang = PI / 2 + 1.3f * sinf(t * 0.35f) + 0.4f * sinf(t * 0.9f);
  float p = underBrow(fmodf((ang + PI / 2) / (2 * PI) * RING_N + RING_N * 2, RING_N));
  // The pupil can sit between LEDs, so it glides: brightness falls from 1 at the pupil to
  // PUPIL_SIDE one LED away and to 0 two LEDs away. Right on an LED that is exactly three LEDs.
  for (int k = 0; k < RING_N; k++) {
    if (isBrow[k]) continue;
    float d = fabsf(k - p); d = fminf(d, RING_N - d);
    float v = d <= 1 ? 1 - (1 - PUPIL_SIDE) * d : d < 2 ? PUPIL_SIDE * (2 - d) : 0;
    if (v > 0) addRing(e, k, s.c, v);
  }
}

void finishEyes(float k) {
  for (int e = 0; e < 2; e++)
    for (int i = 0; i < RING_N; i++)
      for (int c = 0; c < 3; c++) eyeT[e][i][c] = fminf(255, eyeT[e][i][c] * k);
}

struct Ink { char ch; Rgb c; };
void paintMouth(Pic pic, const Ink *inks, int n, float k) {
  for (int y = 0; y < 8; y++)
    for (int x = 0; x < 16; x++)
      for (int i = 0; i < n; i++)
        if (pic[y][x] == inks[i].ch) {
          float *o = mouthT[y * 16 + x];
          o[0] = inks[i].c.r * k; o[1] = inks[i].c.g * k; o[2] = inks[i].c.b * k;
        }
}

// ---- the four moods ----

void sleeping(float t) {
  // snores with both eyes shut: a blue glow along the bottom of each ring
  float b = wave(t, 3);
  for (int e = 0; e < 2; e++)
    for (int k = 6; k <= 10; k++) addRing(e, k, BLUE, .35f + .5f * b);
  finishEyes(1);
  Ink inks[] = { { 'o', { 120, 0, 220 } }, { 'T', { 120, 150, 255 } } };
  paintMouth(b > .6f ? AJAR : SHUT, inks, 2, .3f + .4f * b);
}

void watching(float t) {
  EyeSpec s = { AMBER, FLAT, PURPLE, .6f, WANDER };
  openEye(LEFT, s, t); openEye(RIGHT, s, t);
  finishEyes(1);
  Ink inks[] = { { 'o', RED }, { 'T', TOOTH } };
  paintMouth(SHUT, inks, 2, .75f + .15f * sinf(t * 9));
}

void snarl(float t) {
  EyeSpec s = { YELLOW, ANGRY, BLOOD, .85f, WANDER };
  openEye(LEFT, s, t); openEye(RIGHT, s, t);
  finishEyes(.9f + .1f * rnd());
  Ink inks[] = { { 'o', RED }, { 'T', TOOTH } };
  paintMouth(sinf(t * 7) > -0.2f ? WIDE : SHUT, inks, 2, .85f + .15f * rnd());
}

void crying(float t) {
  // blue and sad: a one-LED tear rolls round each ring and tears stream down over the mouth, the lip trembles
  Pic g;
  bool tremble = fmodf(t, 2) < .7f && ((int)floorf(t * 8)) % 2;
  for (int y = 0; y < 8; y++) {
    const char *src = tremble ? (y > 0 ? FROWN[y - 1] : "................") : FROWN[y];
    memcpy(g[y], src, 17);
  }
  const float drops[2][2] = { { 2, 0 }, { 13, 4.5f } };
  for (auto &d : drops) {
    int y = (int)floorf(fmodf(t * 6 + d[1], 10)) - 1;
    setPx(g, (int)d[0], y, 'd'); setPx(g, (int)d[0], y - 1, 'e');
  }
  EyeSpec s = { BLUE, SAD, CYAN, .55f, DOWN };
  for (int e = 0; e < 2; e++) {
    int side = e == LEFT ? -1 : 1;
    openEye(e, s, t);
    float ph = fmodf(t + (side < 0 ? 0 : .8f), 1.6f) / 1.6f;
    addRing(e, lroundf(side < 0 ? 12 - 4 * ph : 4 + 4 * ph), TEAR, 1);
  }
  finishEyes(1);
  Ink inks[] = { { '#', BLUE }, { 'd', TEAR }, { 'e', { 40, 90, 160 } } };
  paintMouth(g, inks, 3, .9f);
}

// ---- main loop ----

void ease(float *now, const float *target, int n, float a) {
  for (int i = 0; i < n; i++) now[i] += (target[i] - now[i]) * a;
}

// Read whatever has come in; act on each complete line.
void readSerial() {
  static char line[16];
  static int n = 0;
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch != '\n' && ch != '\r') {
      if (n < (int)sizeof line - 1) line[n++] = ch;
      continue;
    }
    line[n] = 0;
    const char *m = n ? strchr("swnc", line[0]) : nullptr;
    if (m) {
      cmdMood = m - "swnc";
      cmdAt = millis();
      hasHand = n > 1;
      if (hasHand) handX = constrain(atoi(line + 1), -100, 100) / 100.0f;
    }
    n = 0;
  }
}

void setup() {
  Serial.begin(115200);
  rp2040.wdt_begin(1000);    // if the loop ever hangs, reboot instead of freezing on one face
  jaw(SHUT, 1, 4); jaw(AJAR, 1, 5); jaw(WIDE, 0, 5);
  strip.begin();
  strip.clear();
  strip.show();
}

void loop() {
  static const char *NAMES[4] = { "sleeping", "watching", "snarl", "crying" };
  static int lastMood = -1;
  uint32_t frameStart = millis();
  float t = frameStart / 1000.0f;
  rp2040.wdt_reset();
  readSerial();
  int mood;
  if (cmdMood < 0) mood = ((int)(t / MOOD_S)) % 4;   // nothing heard yet: loop them as a test
  else if (millis() - cmdAt > STALE_MS) { mood = 0; hasHand = false; }   // the PC went quiet: sleep
  else mood = cmdMood;
  if (mood != lastMood) { Serial.printf("%s, up %lu s\n", NAMES[mood], frameStart / 1000); lastMood = mood; }

  memset(eyeT, 0, sizeof eyeT);
  memset(mouthT, 0, sizeof mouthT);
  if (mood == 0) sleeping(t);
  else if (mood == 1) watching(t);
  else if (mood == 2) snarl(t);
  else crying(t);

  float a = mood == 2 ? 0.45f : 0.3f;
  ease(&eyeNow[0][0][0], &eyeT[0][0][0], 2 * RING_N * 3, a);
  ease(&mouthNow[0][0], &mouthT[0][0], 128 * 3, a);

  // about 20 mA per colour at full; scale down if the frame would draw more than the limit
  float sum = 0;
  for (int e = 0; e < 2; e++) for (int k = 0; k < RING_N; k++) for (int c = 0; c < 3; c++) sum += eyeNow[e][k][c];
  for (int i = 0; i < 128; i++) for (int c = 0; c < 3; c++) sum += mouthNow[i][c];
  float scale = BRIGHT;
  float ma = sum * scale / 255 * 20;
  if (ma > LIMIT_MA) scale *= LIMIT_MA / ma;

  auto out = [&](const float *c) { return strip.Color(c[0] * scale, c[1] * scale, c[2] * scale); };
  for (int e = 0; e < 2; e++)
    for (int k = 0; k < RING_N; k++) strip.setPixelColor(eyeLed((Eye)e, k), out(eyeNow[e][k]));
  for (int y = 0; y < 8; y++)
    for (int x = 0; x < 16; x++) strip.setPixelColor(mouthLed(x, y), out(mouthNow[y * 16 + x]));
  strip.show();

  int16_t wait = 16 - (int16_t)(millis() - frameStart);   // about 60 frames a second, like the preview
  if (wait > 0) delay(wait);
}
