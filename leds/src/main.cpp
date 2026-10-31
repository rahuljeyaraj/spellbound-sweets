// Face LEDs on one WS2812 chain on D10. Layout test: both eyes show red at the top and green
// at 3 o'clock; the mouth shows a red top row, a green left column and a white bottom-right
// corner. If every eye and both mouth boards show that, the mapping below is right.
#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

const int DATA_PIN = D10;    // GPIO3
const int N = 160;
const uint8_t LEVEL = 40;    // out of 255

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

void setup() {
  Serial.begin(115200);
  strip.begin();
  strip.clear();
  uint32_t red = strip.Color(LEVEL, 0, 0), green = strip.Color(0, LEVEL, 0), white = strip.Color(LEVEL, LEVEL, LEVEL);
  for (int i = 0; i < 2; i++) {
    Eye e = (Eye)i;
    strip.setPixelColor(eyeLed(e, 0), red);
    strip.setPixelColor(eyeLed(e, 4), green);
  }
  for (int x = 0; x < 16; x++) strip.setPixelColor(mouthLed(x, 0), red);
  for (int y = 1; y < 8; y++) strip.setPixelColor(mouthLed(0, y), green);
  strip.setPixelColor(mouthLed(15, 7), white);
  strip.show();
}

void loop() {
  Serial.println("layout test");
  delay(2000);
}
