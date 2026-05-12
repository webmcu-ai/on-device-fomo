// ======================================================
// XIAO ML KIT (OR XIAO ESP32S3 SENSE)
// FULL VISION ML WITH FOMO HEAD — v015
//
// Small Image collection, training, inference for education and proof of concept
//
// SD card stores: images in class folders
// SD card stores: headers in bin and .h text char array format
// Serial monitor and OLED output
// OLED inference: clusters high-confidence FOMO cells, draws one bounding box per cluster
// By Jeremy Ellis
// With free tier assistance from: Claude (code overview), ChatGPT (Critique), Gemini (Research) and Copilot (Alternate)
// Use at your own risk!
// MIT license
//
// Github Profile https://github.com/hpssjellis
// LinkedIn https://www.linkedin.com/in/jeremy-ellis-4237a9bb/
//
// For platformio you need the U8g2 library declared in the platformio.ini file and OPI PSRAM set
// lib_deps =  olikraus/U8g2 @ ^2.35.30
// ; Overriding defaults to enable OPI PSRAM
// build_flags =
//    -DBOARD_HAS_PSRAM
//    -DARDUINO_USB_CDC_ON_BOOT=1
// board_build.arduino.memory_type = qio_opi
// board_build.flash_mode = qio
// board_upload.flash_size = 8MB
//


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 0: CORE SYSTEM (ALWAYS INCLUDED)                                   ██
// ██  Headers, Defines, Pins, Globals, Memory, Weights, Setup, Loop           ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████


// optional Uncomment AFTER copying myWeights.h from SD to your sketch folder:
// Priority order: SD weights > baked-in weights > random He-init
//////////////////////////////////////IMPORTANT/////////////////////////////////////////////////
//#define USE_BAKED_WEIGHTS

#ifdef USE_BAKED_WEIGHTS
  #include "myWeights.h"
#endif

#include "esp_camera.h"
#include "img_converters.h"
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>
#include <algorithm>
#include <U8g2lib.h>
#include <Wire.h>

//U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);
U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);  // 180 degree re orientation so OLED is the correct way up

// ======================================================
// CONFIGURATION & ML HYPERPARAMETERS
// ======================================================

#define NUM_CLASSES 2

String myClassLabels[NUM_CLASSES] = {"0Blank", "1Cup"};

const int myTotalItems = NUM_CLASSES + 2;      // NUM_CLASSES + 2 for menu training and inference

float LEARNING_RATE = 0.0003;
int BATCH_SIZE = 12;
int TARGET_EPOCHS = 30;
int VALIDATION_IMAGES = 5;  // last N images per class held out for validation (0 = disabled)

// Detection threshold: only draw overlay when max cell confidence exceeds this
const float myFomoThreshold = 0.38f;    // fixed threshold used when myUseDynamicThreshold = false

// Dynamic threshold: set true to use (peakVal * myDynamicThresholdRatio), floored at myDynamicThresholdFloor
const bool  myUseDynamicThreshold    = true;   // true = adaptive per-frame, false = fixed myFomoThreshold
const float myDynamicThresholdRatio  = 0.90f;  // fraction of peak cell value to use as threshold
const float myDynamicThresholdFloor  = 0.38f;  // minimum threshold even when peak is low

const int myThresholdPress = 1100;
const int myThresholdRelease = 900;


// ======================================================
// UNIFIED TOUCH INPUT SYSTEM
// ======================================================
struct TouchState {
  bool isTouching = false;
  int tapCount = 0;
  unsigned long firstTapTime = 0;
  unsigned long lastReleaseTime = 0;
  unsigned long lastCheckTime = 0;
  const unsigned long tapWindow = 800;
  const int longPressTaps = 3;
  const unsigned long debounceDelay = 50;
};

TouchState myTouch;

// SYSTEM LOGIC VARIABLES
unsigned long myLastActivityTime = 0;
unsigned long myLastTapTime = 0;
const int myTapCooldown = 250;
int myMenuIndex = 1;
bool myIsSelected = false;
bool myWeightsTrained = false;

// XIAO ESP32-S3 Camera Pins
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

// ======================================================
// CONFIGURABLE INPUT RESOLUTION
// ======================================================
#define INPUT_SIZE 64

// ======================================================
// CNN + FOMO ARCHITECTURE CONSTANTS
// ======================================================

// Conv layer 1
#define CONV1_KERNEL_SIZE 3
#define CONV1_FILTERS 4
#define CONV1_WEIGHTS (CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE * 3 * CONV1_FILTERS)

// Conv layer 2
#define CONV2_KERNEL_SIZE 3
#define CONV2_FILTERS 8
#define CONV2_WEIGHTS (CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE * CONV1_FILTERS * CONV2_FILTERS)

// Feature map sizes
#define CONV1_OUTPUT_SIZE (INPUT_SIZE - 2)
#define POOL1_OUTPUT_SIZE (CONV1_OUTPUT_SIZE / 2)
#define CONV2_OUTPUT_SIZE (POOL1_OUTPUT_SIZE - 2)

// Flattened size — used for dense output weights (kept from v44 for weight save/load compat)
#define FLATTENED_SIZE (CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * CONV2_FILTERS)

// FOMO output grid: one cell per spatial position in the conv2 feature map
#define FOMO_GRID CONV2_OUTPUT_SIZE
#define FOMO_CELLS (FOMO_GRID * FOMO_GRID)

// Dense output weights: 1x1 conv over flattened conv2 -> NUM_CLASSES scores per cell
#define OUTPUT_WEIGHTS (FLATTENED_SIZE * NUM_CLASSES)

// ======================================================
// GLOBAL VARIABLE DEFINITIONS
// ======================================================

uint8_t* myRgbBuffer = nullptr;   // reusable 240x240x3 RGB buffer

bool mySDavailable = false;

// ML weight buffers (PSRAM)
float* myInputBuffer  = nullptr;
float* myConv1_w      = nullptr;
float* myConv1_b      = nullptr;
float* myConv2_w      = nullptr;
float* myConv2_b      = nullptr;
float* myOutput_w     = nullptr;   // 1x1 conv weights  [NUM_CLASSES][CONV2_FILTERS][1][1]
float* myOutput_b     = nullptr;   // 1x1 conv biases   [NUM_CLASSES]

// Gradient accumulator buffers
float* myConv1_w_grad  = nullptr;
float* myConv1_b_grad  = nullptr;
float* myConv2_w_grad  = nullptr;
float* myConv2_b_grad  = nullptr;
float* myOutput_w_grad = nullptr;
float* myOutput_b_grad = nullptr;

// Adam optimizer momentum buffers
float* myConv1_w_m  = nullptr;  float* myConv1_w_v  = nullptr;
float* myConv1_b_m  = nullptr;  float* myConv1_b_v  = nullptr;
float* myConv2_w_m  = nullptr;  float* myConv2_w_v  = nullptr;
float* myConv2_b_m  = nullptr;  float* myConv2_b_v  = nullptr;
float* myOutput_w_m = nullptr;  float* myOutput_w_v = nullptr;
float* myOutput_b_m = nullptr;  float* myOutput_b_v = nullptr;

// Intermediate activation buffers
float* myConv1_output = nullptr;
float* myPool1_output = nullptr;
float* myConv2_output = nullptr;

// FOMO map: [NUM_CLASSES][FOMO_CELLS] — sigmoid output for each class at each spatial cell
float* myFomoMap  = nullptr;

// Dense output — global classification score (GAP of FOMO map, or max-pool), used in validation
float* myDense_output = nullptr;

// Gradient propagation buffers (per-image signals, zeroed each image)
float* myFomoGrad  = nullptr;   // [FOMO_CELLS * CONV2_FILTERS] — grad w.r.t. conv2 output
float* myConv2_grad = nullptr;
float* myPool1_grad = nullptr;
float* myConv1_grad = nullptr;

// Bounding box as image fractions (0.0-1.0)
struct FomoBox {
  float x1, y1, x2, y2;
};

struct TrainingItem {
  String path;
  int label;
  std::vector<FomoBox> boxes;   // from annotations.json; empty = use default
};
std::vector<TrainingItem> myTrainingData;

// ======================================================
// UTILITY FUNCTIONS
// ======================================================
inline float clip_value(float v, float mn=-100, float mx=100) {
  if(isnan(v)||isinf(v)) return 0;
  return constrain(v,mn,mx);
}

inline float leaky_relu(float x)       { return x > 0 ? x : 0.1f * x; }
inline float leaky_relu_deriv(float x) { return x > 0 ? 1.0f : 0.1f; }

inline float my_sigmoid(float x)       { return 1.0f / (1.0f + expf(-x)); }
inline float my_sigmoid_deriv(float s) { return s * (1.0f - s); }  // s is already sigmoid(x)

// Flip myRgbBuffer (240x240 RGB888) horizontally in-place.
// Call after every fmt2rgb888() so collection, training, and inference all see the same orientation.
void myFlipImageHorizontal() {
  for (int y = 0; y < 240; y++) {
    uint8_t* row = myRgbBuffer + y * 240 * 3;
    for (int x = 0; x < 120; x++) {
      uint8_t* left  = row + x * 3;
      uint8_t* right = row + (239 - x) * 3;
      uint8_t tmp;
      tmp = left[0]; left[0] = right[0]; right[0] = tmp;  // R
      tmp = left[1]; left[1] = right[1]; right[1] = tmp;  // G
      tmp = left[2]; left[2] = right[2]; right[2] = tmp;  // B
    }
  }
}

// ======================================================
// UNIFIED TOUCH INPUT FUNCTIONS
// ======================================================
int myReadTouch() {
  int sum = 0;
  for (int i = 0; i < 3; i++) {
    sum += analogRead(A0);
    delayMicroseconds(100);
  }
  return sum / 3;
}

void myResetTouchState() {
  myTouch.isTouching = false;
  myTouch.tapCount = 0;
  myTouch.firstTapTime = 0;
  myTouch.lastReleaseTime = 0;
  myTouch.lastCheckTime = 0;
}

void myUpdateTouchState() {
  unsigned long now = millis();
  if (now - myTouch.lastCheckTime < 20) return;
  myTouch.lastCheckTime = now;

  int val = myReadTouch();
  bool touchActive = myTouch.isTouching
                      ? (val > myThresholdRelease)
                      : (val > myThresholdPress);

  if (touchActive && !myTouch.isTouching) {
    if (now - myTouch.lastReleaseTime < myTouch.debounceDelay) return;
    myTouch.isTouching = true;
    if (myTouch.tapCount == 0 || (now - myTouch.firstTapTime < myTouch.tapWindow)) {
      if (myTouch.tapCount == 0) myTouch.firstTapTime = now;
      myTouch.tapCount++;
      Serial.printf("Tap #%d\n", myTouch.tapCount);
    } else {
      myTouch.tapCount = 1;
      myTouch.firstTapTime = now;
      Serial.println("Tap #1 (new window)");
    }
  }

  if (!touchActive && myTouch.isTouching) {
    myTouch.isTouching = false;
    myTouch.lastReleaseTime = now;
  }
}

// Returns: 0=no action, 1=tap, 2=long press (3+ taps)
int myCheckTouchInput() {
  myUpdateTouchState();
  unsigned long now = millis();
  if (myTouch.tapCount > 0 && !myTouch.isTouching) {
    if (now - myTouch.firstTapTime > myTouch.tapWindow) {
      int result = (myTouch.tapCount >= myTouch.longPressTaps) ? 2 : 1;
      int count = myTouch.tapCount;
      myResetTouchState();
      if (result == 2) Serial.printf("LONG PRESS detected (%d taps)\n", count);
      else             Serial.printf("TAP detected (%d tap%s)\n", count, count > 1 ? "s" : "");
      return result;
    }
  }
  return 0;
}

void myCheckTouchBackground() { myUpdateTouchState(); }

int myPeekTouchAction() {
  myUpdateTouchState();
  unsigned long now = millis();
  if (myTouch.tapCount > 0 && !myTouch.isTouching) {
    if (now - myTouch.firstTapTime > myTouch.tapWindow) {
      return (myTouch.tapCount >= myTouch.longPressTaps) ? 2 : 1;
    }
  }
  return 0;
}


// ======================================================
// MEMORY ALLOCATION
// ======================================================
void myAllocateMemory() {
  if (myInputBuffer != nullptr) return;

  Serial.println("\n=== Allocating Memory ===");

  myInputBuffer  = (float*)ps_malloc(INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));
  myConv1_w      = (float*)ps_malloc(CONV1_WEIGHTS  * sizeof(float));
  myConv1_b      = (float*)ps_malloc(CONV1_FILTERS  * sizeof(float));
  myConv2_w      = (float*)ps_malloc(CONV2_WEIGHTS  * sizeof(float));
  myConv2_b      = (float*)ps_malloc(CONV2_FILTERS  * sizeof(float));
  myOutput_w     = (float*)ps_malloc(OUTPUT_WEIGHTS * sizeof(float));
  myOutput_b     = (float*)ps_malloc(NUM_CLASSES    * sizeof(float));

  myConv1_w_grad  = (float*)ps_malloc(CONV1_WEIGHTS  * sizeof(float));
  myConv1_b_grad  = (float*)ps_malloc(CONV1_FILTERS  * sizeof(float));
  myConv2_w_grad  = (float*)ps_malloc(CONV2_WEIGHTS  * sizeof(float));
  myConv2_b_grad  = (float*)ps_malloc(CONV2_FILTERS  * sizeof(float));
  myOutput_w_grad = (float*)ps_malloc(OUTPUT_WEIGHTS * sizeof(float));
  myOutput_b_grad = (float*)ps_malloc(NUM_CLASSES    * sizeof(float));

  myConv1_w_m  = (float*)ps_calloc(CONV1_WEIGHTS,  sizeof(float));
  myConv1_w_v  = (float*)ps_calloc(CONV1_WEIGHTS,  sizeof(float));
  myConv1_b_m  = (float*)ps_calloc(CONV1_FILTERS,  sizeof(float));
  myConv1_b_v  = (float*)ps_calloc(CONV1_FILTERS,  sizeof(float));
  myConv2_w_m  = (float*)ps_calloc(CONV2_WEIGHTS,  sizeof(float));
  myConv2_w_v  = (float*)ps_calloc(CONV2_WEIGHTS,  sizeof(float));
  myConv2_b_m  = (float*)ps_calloc(CONV2_FILTERS,  sizeof(float));
  myConv2_b_v  = (float*)ps_calloc(CONV2_FILTERS,  sizeof(float));
  myOutput_w_m = (float*)ps_calloc(OUTPUT_WEIGHTS,  sizeof(float));
  myOutput_w_v = (float*)ps_calloc(OUTPUT_WEIGHTS,  sizeof(float));
  myOutput_b_m = (float*)ps_calloc(NUM_CLASSES,     sizeof(float));
  myOutput_b_v = (float*)ps_calloc(NUM_CLASSES,     sizeof(float));

  // BUG FIX (v002): removed double allocation (new float[] blocks leaked before ps_malloc overwrote the pointer)
  myConv1_output = (float*)ps_malloc(CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE * CONV1_FILTERS * sizeof(float));
  myPool1_output = (float*)ps_malloc(POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE * CONV1_FILTERS * sizeof(float));
  myConv2_output = (float*)ps_malloc(CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * CONV2_FILTERS * sizeof(float));
  myFomoMap      = (float*)ps_malloc(FOMO_CELLS * NUM_CLASSES * sizeof(float));
  myDense_output = (float*)ps_malloc(NUM_CLASSES * sizeof(float));

  // Gradient propagation buffers
  myFomoGrad  = (float*)ps_malloc(FOMO_CELLS * CONV2_FILTERS * sizeof(float));  // grad w.r.t. conv2_output per cell
  myConv2_grad = (float*)ps_malloc(CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * CONV2_FILTERS * sizeof(float));
  myPool1_grad = (float*)ps_malloc(POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE * CONV1_FILTERS * sizeof(float));
  myConv1_grad = (float*)ps_malloc(CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE * CONV1_FILTERS * sizeof(float));

  if (!myInputBuffer || !myConv1_w || !myConv2_w || !myOutput_w ||
      !myConv1_output || !myPool1_output || !myConv2_output ||
      !myFomoMap || !myDense_output || !myFomoGrad) {
    Serial.println("FATAL: PSRAM allocation failed!");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "PSRAM ERROR!"); } while (u8g2.nextPage());
    while(1) { delay(1000); }
  }

  Serial.printf("Free PSRAM after allocation: %d bytes\n", ESP.getFreePsram());

  // He initialization
  float c1std = sqrt(2.0 / (9.0 * 3));
  for(int i=0; i<CONV1_WEIGHTS; i++) myConv1_w[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * c1std;
  for(int i=0; i<CONV1_FILTERS; i++) myConv1_b[i] = 0;

  float c2std = sqrt(2.0 / (9.0 * CONV1_FILTERS));
  for(int i=0; i<CONV2_WEIGHTS; i++) myConv2_w[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * c2std;
  for(int i=0; i<CONV2_FILTERS; i++) myConv2_b[i] = 0;

  // 1x1 conv output head — fan_in = CONV2_FILTERS
  float dstd = sqrt(2.0 / CONV2_FILTERS);
  for(int i=0; i<OUTPUT_WEIGHTS; i++) myOutput_w[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * dstd;
  for(int i=0; i<NUM_CLASSES; i++) myOutput_b[i] = 0;
  Serial.println("He-init random weights set");
}

// ======================================================
// WEIGHT SAVE / LOAD / EXPORT
// ======================================================
void myExportHeader() {
  if (!mySDavailable) { Serial.println("No SD card - cannot export header"); return; }
  if (!SD.exists("/header")) SD.mkdir("/header");
  File file = SD.open("/header/myWeights.h", FILE_WRITE);
  if (!file) return;
  file.println("#ifndef MY_MODEL_H\n#define MY_MODEL_H");
  file.println("// ======================================================");
  file.println("// IMPORTANT: After copying this file to your sketch folder,");
  file.println("// update BOTH of the following lines in your main sketch");
  file.println("// to match the number of classes and labels used during training:");
  file.println("//");
  file.printf( "//   #define NUM_CLASSES %d\n", NUM_CLASSES);
  file.print("//   String myClassLabels[NUM_CLASSES] = {");
  for (int i = 0; i < NUM_CLASSES; i++) {
    file.printf("\"%s\"", myClassLabels[i].c_str());
    if (i < NUM_CLASSES - 1) file.print(", ");
  }
  file.println("};");
  file.println("//");
  file.println("// Then uncomment:  #define USE_BAKED_WEIGHTS");
  file.println("// ======================================================");
  auto myDump = [&](const char* name, float* data, int size) {
    file.printf("const float %s[] = { ", name);
    for(int i=0; i<size; i++) {
      file.print(data[i], 6); file.print("f");
      if(i < size-1) file.print(", ");
      if((i+1)%8 == 0) file.println();
    }
    file.println(" };");
  };
  myDump("myModel_conv1_w",  myConv1_w,  CONV1_WEIGHTS);
  myDump("myModel_conv1_b",  myConv1_b,  CONV1_FILTERS);
  myDump("myModel_conv2_w",  myConv2_w,  CONV2_WEIGHTS);
  myDump("myModel_conv2_b",  myConv2_b,  CONV2_FILTERS);
  myDump("myModel_output_w", myOutput_w, OUTPUT_WEIGHTS);
  myDump("myModel_output_b", myOutput_b, NUM_CLASSES);
  file.println("#endif");
  file.close();
  Serial.println("You can copy /header/myWeights.h to the sketch folder, then uncomment #define USE_BAKED_WEIGHTS");
}

bool myLoadWeights() {
  if (!mySDavailable) { Serial.println("No SD card - skipping weight load"); return false; }
  if (!SD.exists("/header/myWeights.bin")) { Serial.println("No SD weights file found"); return false; }
  Serial.println("Loading weights from SD...");
  File f = SD.open("/header/myWeights.bin", FILE_READ);
  if (!f) return false;
  f.read((uint8_t*)myConv1_w,  CONV1_WEIGHTS  * 4);
  f.read((uint8_t*)myConv1_b,  CONV1_FILTERS  * 4);
  f.read((uint8_t*)myConv2_w,  CONV2_WEIGHTS  * 4);
  f.read((uint8_t*)myConv2_b,  CONV2_FILTERS  * 4);
  f.read((uint8_t*)myOutput_w, OUTPUT_WEIGHTS * 4);
  f.read((uint8_t*)myOutput_b, NUM_CLASSES    * 4);
  f.close();
  Serial.println("Weights loaded successfully");
  myWeightsTrained = true;
  return true;
}

void mySaveWeights() {
  if (!mySDavailable) { Serial.println("No SD card - cannot save weights"); return; }
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open("/header/myWeights.bin", FILE_WRITE);
  if (f) {
    f.write((uint8_t*)myConv1_w,  CONV1_WEIGHTS  * 4);
    f.write((uint8_t*)myConv1_b,  CONV1_FILTERS  * 4);
    f.write((uint8_t*)myConv2_w,  CONV2_WEIGHTS  * 4);
    f.write((uint8_t*)myConv2_b,  CONV2_FILTERS  * 4);
    f.write((uint8_t*)myOutput_w, OUTPUT_WEIGHTS * 4);
    f.write((uint8_t*)myOutput_b, NUM_CLASSES    * 4);
    f.close();
    Serial.println("Weights saved to SD");
  }
  myExportHeader();
}


// ======================================================
// IMAGE LOADING FROM SD (no augmentation — used for validation)
// ======================================================
bool myLoadImageFromFile(const char* path, float* buf) {
  File f = SD.open(path);
  if(!f) return false;
  size_t sz = f.size();
  uint8_t* jpg = (uint8_t*)ps_malloc(sz);
  if(!jpg) { f.close(); return false; }
  f.read(jpg, sz);
  f.close();
  if(!myRgbBuffer) { free(jpg); return false; }
  bool ok = fmt2rgb888(jpg, sz, PIXFORMAT_JPEG, myRgbBuffer);
  free(jpg);
  if(!ok) return false;
  myFlipImageHorizontal();
  for(int y=0; y<INPUT_SIZE; y++) {
    for(int x=0; x<INPUT_SIZE; x++) {
      int sy = (int)((y+0.5)*240.0/INPUT_SIZE); if(sy>239) sy=239;
      int sx = (int)((x+0.5)*240.0/INPUT_SIZE); if(sx>239) sx=239;
      int srcIdx = (sy*240 + sx)*3;
      int dstIdx = (y*INPUT_SIZE + x)*3;
      buf[dstIdx]   = myRgbBuffer[srcIdx]   / 255.0f;
      buf[dstIdx+1] = myRgbBuffer[srcIdx+1] / 255.0f;
      buf[dstIdx+2] = myRgbBuffer[srcIdx+2] / 255.0f;
    }
  }
  return true;
}




// ======================================================
// ANNOTATIONS.JSON LOADER
// Reads /images/<className>/annotations.json from SD.
// Returns a map of filename -> vector<FomoBox>.
// Format: { "img_001.jpg": [{x1,y1,x2,y2}, ...], ... }
// Uses a minimal hand-rolled parser — no heap-heavy JSON lib needed.
// ======================================================
std::vector<FomoBox> myParseBoxesForFile(const String& jsonText, const String& filename) {
  std::vector<FomoBox> result;
  // Find the key matching this filename
  int keyPos = jsonText.indexOf("\"" + filename + "\"");
  if (keyPos < 0) return result;
  // Find the opening '[' of this file's box array
  int arrOpen = jsonText.indexOf('[', keyPos);
  if (arrOpen < 0) return result;
  int arrClose = jsonText.indexOf(']', arrOpen);
  if (arrClose < 0) return result;
  String arrText = jsonText.substring(arrOpen, arrClose + 1);

  // Walk through each '{...}' object in the array
  int pos = 0;
  while (true) {
    int objOpen = arrText.indexOf('{', pos);
    if (objOpen < 0) break;
    int objClose = arrText.indexOf('}', objOpen);
    if (objClose < 0) break;
    String obj = arrText.substring(objOpen, objClose + 1);

    // Parse x1, y1, x2, y2 from the object string
    auto myGetVal = [&](const char* key) -> float {
      int k = obj.indexOf(key);
      if (k < 0) return 0.0f;
      int colon = obj.indexOf(':', k);
      if (colon < 0) return 0.0f;
      return obj.substring(colon + 1).toFloat();
    };

    FomoBox b;
    b.x1 = myGetVal("x1");
    b.y1 = myGetVal("y1");
    b.x2 = myGetVal("x2");
    b.y2 = myGetVal("y2");
    // Sanity check: box must have positive area
    if (b.x2 > b.x1 && b.y2 > b.y1) result.push_back(b);
    pos = objClose + 1;
  }
  return result;
}

// Load the full annotations.json text for a class folder.
// Returns "" if file absent (caller uses default box).
String myLoadAnnotationsJson(const String& classLabel) {
  String jsonPath = "/images/" + classLabel + "/annotations.json";
  if (!SD.exists(jsonPath)) return "";
  File f = SD.open(jsonPath, FILE_READ);
  if (!f) return "";
  String text = "";
  while (f.available()) text += (char)f.read();
  f.close();
  return text;
}

// ======================================================
// DATA AUGMENTATION CONFIG
// ======================================================
// When enabled, randomly crops a sub-window of the 240x240 image
// and shifts the Gaussian target to match the object's new apparent position.
// The object is assumed to be at the centre of the captured image.
// Enable by setting to 1; disable for validation/inference (uses full-frame).
#define USE_AUGMENTATION 1

// Crop size range: we pick a square crop between MIN and MAX pixels,
// then downsample to INPUT_SIZE. Larger crop = more zoom-out + more offset range.
// A 240x240 crop = no zoom, full frame. A 160x160 crop = 1.5x zoom-in.
const int MY_AUG_CROP_MIN = 160;   // tightest crop (most zoom-in)
const int MY_AUG_CROP_MAX = 220;   // loosest crop  (near full frame)

// ======================================================
// AUGMENTED IMAGE LOAD
// Returns false on failure.
// On success: fills buf with the cropped+resized image,
//             and sets outCx/outCy to the FOMO grid cell
//             where the object centroid appears after cropping.
// ======================================================
bool myLoadImageAugmented(const char* path, float* buf,
                             const std::vector<FomoBox>& srcBoxes,
                             std::vector<FomoBox>& outBoxes) {
  File f = SD.open(path);
  if (!f) return false;
  size_t sz = f.size();
  uint8_t* jpg = (uint8_t*)ps_malloc(sz);
  if (!jpg) { f.close(); return false; }
  f.read(jpg, sz);
  f.close();
  if (!myRgbBuffer) { free(jpg); return false; }
  bool ok = fmt2rgb888(jpg, sz, PIXFORMAT_JPEG, myRgbBuffer);
  free(jpg);
  if (!ok) return false;
  myFlipImageHorizontal();

#if USE_AUGMENTATION
  // Pick a random crop size
  int cropSize = MY_AUG_CROP_MIN + random(MY_AUG_CROP_MAX - MY_AUG_CROP_MIN + 1);

  // Max offset so the crop stays within 240x240
  int maxOffset = 240 - cropSize;

  // Random top-left corner of the crop
  int cropX = (maxOffset > 0) ? random(maxOffset + 1) : 0;
  int cropY = (maxOffset > 0) ? random(maxOffset + 1) : 0;

  // Transform each source box (image fractions 0-1, relative to full 240x240)
  // into crop-relative fractions, then store in outBoxes.
  // Boxes that land entirely outside the crop are dropped.
  outBoxes.clear();
  for (const FomoBox& b : srcBoxes) {
    // Source box in pixel coords (full 240x240 frame)
    float px1 = b.x1 * 240.0f,  py1 = b.y1 * 240.0f;
    float px2 = b.x2 * 240.0f,  py2 = b.y2 * 240.0f;
    // Clip to the crop window
    float cx1 = constrain(px1, (float)cropX, (float)(cropX + cropSize));
    float cy1 = constrain(py1, (float)cropY, (float)(cropY + cropSize));
    float cx2 = constrain(px2, (float)cropX, (float)(cropX + cropSize));
    float cy2 = constrain(py2, (float)cropY, (float)(cropY + cropSize));
    // If the box has no area within the crop, skip it
    if (cx2 - cx1 < 1.0f || cy2 - cy1 < 1.0f) continue;
    // Convert to crop-relative fractions
    FomoBox ob;
    ob.x1 = (cx1 - cropX) / cropSize;
    ob.y1 = (cy1 - cropY) / cropSize;
    ob.x2 = (cx2 - cropX) / cropSize;
    ob.y2 = (cy2 - cropY) / cropSize;
    outBoxes.push_back(ob);
  }

  // Resample: map INPUT_SIZE output pixels into the crop window
  for (int y = 0; y < INPUT_SIZE; y++) {
    for (int x = 0; x < INPUT_SIZE; x++) {
      int sy = cropY + (int)((y + 0.5f) * cropSize / INPUT_SIZE);
      int sx = cropX + (int)((x + 0.5f) * cropSize / INPUT_SIZE);
      sy = constrain(sy, 0, 239);
      sx = constrain(sx, 0, 239);
      int srcIdx = (sy * 240 + sx) * 3;
      int dstIdx = (y * INPUT_SIZE + x) * 3;
      buf[dstIdx]   = myRgbBuffer[srcIdx]   / 255.0f;
      buf[dstIdx+1] = myRgbBuffer[srcIdx+1] / 255.0f;
      buf[dstIdx+2] = myRgbBuffer[srcIdx+2] / 255.0f;
    }
  }
#else
  // No augmentation: original centred full-frame resize
  for (int y = 0; y < INPUT_SIZE; y++) {
    for (int x = 0; x < INPUT_SIZE; x++) {
      int sy = (int)((y + 0.5f) * 240.0f / INPUT_SIZE); if (sy > 239) sy = 239;
      int sx = (int)((x + 0.5f) * 240.0f / INPUT_SIZE); if (sx > 239) sx = 239;
      int srcIdx = (sy * 240 + sx) * 3;
      int dstIdx = (y * INPUT_SIZE + x) * 3;
      buf[dstIdx]   = myRgbBuffer[srcIdx]   / 255.0f;
      buf[dstIdx+1] = myRgbBuffer[srcIdx+1] / 255.0f;
      buf[dstIdx+2] = myRgbBuffer[srcIdx+2] / 255.0f;
    }
  }
  // No crop applied — boxes pass through unchanged
  outBoxes = srcBoxes;
#endif
  return true;
}
// ======================================================
// FORWARD DECLARATIONS
// ======================================================
void myActionCollect(int classIdx);
void myActionTrain();
void myActionInfer();
void myResetMenuState();
void myHandleMenuNavigation();
void myDrawMenu();
void myForwardPass(float* input);
void myMakeBoxTarget(float* map, const std::vector<FomoBox>& boxes);
void myMakeGaussianTarget(float* map, int cx, int cy);
void myBackwardFomoHead(int label, float* targetMap);
void myBackwardConv2();
void myBackwardPool1();
void myBackwardConv1();
void myUpdateWeights(int batchNum);

// ======================================================
// SETUP AND LOOP
// ======================================================
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  delay(1000);

  Serial.println("\n=== XIAO ESP32-S3 ML System Starting ===");
  Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());

  myRgbBuffer = (uint8_t*)ps_malloc(240 * 240 * 3);
  if (!myRgbBuffer) Serial.println("Failed to allocate RGB buffer!");

  pinMode(A0, INPUT);
  u8g2.begin();

  pinMode(21, OUTPUT);
  digitalWrite(21, HIGH);
  delay(100);

  Serial.println("Checking SD card...");
  SPI.begin();
  SPI.setFrequency(400000);

  mySDavailable = SD.begin(21, SPI, 400000, "/sd", 5, false);
  if (!mySDavailable) {
    SD.end();
    Serial.println("No SD card - continuing without it");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000);
  } else {
    Serial.println("SD card mounted successfully");
  }

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk  = XCLK_GPIO_NUM;  config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href  = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM; config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn  = PWDN_GPIO_NUM;  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;    config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_240X240; config.jpeg_quality = 12;
  config.fb_count = 1;
  esp_camera_init(&config);
  Serial.println("Camera initialized");



 // sensor_t* s = esp_camera_sensor_get();
 // if (s != NULL) { s->set_hmirror(s, 1); }

// to flip the camera image
sensor_t* mySensor = esp_camera_sensor_get();
if (mySensor != NULL) {
  mySensor->set_vflip(mySensor, 1);   // Flips vertically
  mySensor->set_hmirror(mySensor, 1); // Flips horizontally
}

  esp_log_level_set("*", ESP_LOG_WARN);
  esp_log_level_set("esp_camera", ESP_LOG_ERROR);

  myAllocateMemory();

#ifdef USE_BAKED_WEIGHTS
  memcpy(myConv1_w,  myModel_conv1_w,  CONV1_WEIGHTS  * sizeof(float));
  memcpy(myConv1_b,  myModel_conv1_b,  CONV1_FILTERS  * sizeof(float));
  memcpy(myConv2_w,  myModel_conv2_w,  CONV2_WEIGHTS  * sizeof(float));
  memcpy(myConv2_b,  myModel_conv2_b,  CONV2_FILTERS  * sizeof(float));
  memcpy(myOutput_w, myModel_output_w, OUTPUT_WEIGHTS * sizeof(float));
  memcpy(myOutput_b, myModel_output_b, NUM_CLASSES    * sizeof(float));
  Serial.println("Baked-in weights loaded from myWeights.h");
  myWeightsTrained = true;
#endif

  if (myLoadWeights()) {
    Serial.println("SD weights loaded - overriding baked-in weights");
  }

  myLastActivityTime = millis();
  myResetMenuState();
  delay(2000);

  Serial.println("System ready - Tap A0 to navigate, 3+ taps to select");
  myDrawMenu();
}

void loop() {
  myHandleMenuNavigation();
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 1: IMAGE COLLECTION FUNCTIONS                                      ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

// ======================================================
// SHARED OLED RENDER HELPERS
// ======================================================
void myRenderRgbToOLED(int imageCount) {
  int myOledWidth  = u8g2.getDisplayWidth();
  int myOledHeight = u8g2.getDisplayHeight();
  int myScaleX = 240 / myOledWidth;
  int myScaleY = 240 / myOledHeight;

  u8g2.firstPage();
  do {
    for (int myOledX = 0; myOledX < myOledWidth; myOledX++) {
      for (int myOledY = 0; myOledY < myOledHeight; myOledY++) {
        size_t myPixelIndex = ((myOledY * myScaleY) * 240 + (myOledX * myScaleX)) * 3;
        uint8_t myBrightness = (myRgbBuffer[myPixelIndex]     +
                                myRgbBuffer[myPixelIndex + 1] +
                                myRgbBuffer[myPixelIndex + 2]) / 3;
        if (myBrightness > 100) u8g2.drawPixel(myOledX, myOledY);
      }
    }
    if (imageCount >= 0) {
      u8g2.setFont(u8g2_font_ncenB10_tr);
      u8g2.setColorIndex(0); u8g2.drawBox(0, 0, 20, 15);
      u8g2.setColorIndex(1); u8g2.setCursor(3, 10); u8g2.print(String(imageCount));
    } else {
      u8g2.setFont(u8g2_font_5x7_tf);
      u8g2.setColorIndex(0); u8g2.drawBox(50, 0, 22, 8);
      u8g2.setColorIndex(1); u8g2.drawStr(52, 7, "LIVE");
    }
  } while (u8g2.nextPage());
}

void myDisplayImageOnOLED(camera_fb_t* fb, int imageCount) {
  if (!myRgbBuffer) { Serial.println("RGB buffer not allocated - skipping OLED preview"); return; }
  if (!fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) {
    Serial.println("Failed to convert JPEG to RGB888 for OLED"); return;
  }
  myFlipImageHorizontal();
  myRenderRgbToOLED(imageCount);
}

void myActionCollect(int classIdx) {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot collect images");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000); myResetMenuState(); return;
  }

  Serial.printf("\n>>> Collection mode: %s\n", myClassLabels[classIdx].c_str());
  Serial.println("  TAP (1-2 taps) = Capture image");
  Serial.println("  LONG PRESS (3+ taps) = Exit to menu");
  Serial.println("  Serial: 'T'=capture, 'L'=exit");

  myResetTouchState();

  String path = "/images/" + myClassLabels[classIdx];
  if (!SD.exists("/images")) SD.mkdir("/images");
  if (!SD.exists(path)) SD.mkdir(path);

  int counts[NUM_CLASSES] = {};
  File root = SD.open("/images/" + myClassLabels[classIdx]);
  if(root) {
    while(File file = root.openNextFile()) {
      if(!file.isDirectory() && (String(file.name()).endsWith(".jpg") ||
        String(file.name()).endsWith(".JPG"))) { counts[classIdx]++; }
      file.close();
    }
    root.close();
  }

  unsigned long lastCameraDrain = 0;
  unsigned long lastOLED = 0;
  bool oledNeedsUpdate = false;
  bool shouldCapture = false;

  while (true) {
    unsigned long now = millis();

    if (now - lastCameraDrain > 50) {
      lastCameraDrain = now;
      if (!shouldCapture) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) {
          if (now - lastOLED > 250 && myRgbBuffer) {
            if (fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) {
              myFlipImageHorizontal();
              oledNeedsUpdate = true; lastOLED = now;
            }
          }
          esp_camera_fb_return(fb);
        }
      }
    }

    if (oledNeedsUpdate) {
      oledNeedsUpdate = false;
      myRenderRgbToOLED(-1);
    }

    if (Serial.available()) {
      char c = Serial.read();
      if (c == 'l' || c == 'L') { myResetMenuState(); return; }
      else if (c == 't' || c == 'T') { shouldCapture = true; }
    }

    int touchAction = myCheckTouchInput();
    if (touchAction == 2) { Serial.println("Exiting collection mode"); myResetMenuState(); return; }
    else if (touchAction == 1) { shouldCapture = true; }

    if (shouldCapture) {
      shouldCapture = false;
      camera_fb_t* fb = esp_camera_fb_get();
      if (fb) {
        String fileName = path + "/img_" + String(millis()) + ".jpg";
        File file = SD.open(fileName, FILE_WRITE);
        if (file) {
          file.write(fb->buf, fb->len);
          file.close();
          counts[classIdx]++;
          Serial.printf("Saved: %s (Total: %d)\n", fileName.c_str(), counts[classIdx]);
          myDisplayImageOnOLED(fb, counts[classIdx]);
          delay(300);
          lastOLED = millis();
        }
        esp_camera_fb_return(fb);
      }
    }
    delay(5);
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 2: FORWARD PASS, FOMO HEAD, BACKWARD PASS, OPTIMIZER               ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

// ======================================================
// FORWARD PASS
// Fills: myConv1_output, myPool1_output, myConv2_output,
//        myFomoMap[NUM_CLASSES][FOMO_CELLS],
//        myDense_output[NUM_CLASSES]  (global average pool of myFomoMap)
// The `logits` output param is unused; callers can pass a dummy array.
// ======================================================
void myForwardPass(float* input) {
  // --- Conv1: INPUT_SIZE x INPUT_SIZE x 3 -> CONV1_OUTPUT_SIZE x CONV1_OUTPUT_SIZE x CONV1_FILTERS ---
  for(int f=0; f<CONV1_FILTERS; f++) {
    int ob = f * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE;
    for(int y=0; y<CONV1_OUTPUT_SIZE; y++) {
      for(int x=0; x<CONV1_OUTPUT_SIZE; x++) {
        float sum = myConv1_b[f];
        for(int ky=0; ky<3; ky++) {
          for(int kx=0; kx<3; kx++) {
            int inPos = ((y+ky)*INPUT_SIZE + (x+kx)) * 3;
            int wPos  = f*27 + ky*9 + kx*3;
            sum += input[inPos]   * myConv1_w[wPos]   +
                   input[inPos+1] * myConv1_w[wPos+1] +
                   input[inPos+2] * myConv1_w[wPos+2];
          }
        }
        myConv1_output[ob + y*CONV1_OUTPUT_SIZE + x] = leaky_relu(clip_value(sum));
      }
    }
  }

  // --- Pool1: 2x2 max-pool ---
  for(int f=0; f<CONV1_FILTERS; f++) {
    int ib = f * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE;
    int ob = f * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE;
    for(int y=0; y<POOL1_OUTPUT_SIZE; y++) {
      for(int x=0; x<POOL1_OUTPUT_SIZE; x++) {
        int iy=y*2, ix=x*2;
        float mv = myConv1_output[ib + iy*CONV1_OUTPUT_SIZE + ix];
        mv = max(mv, myConv1_output[ib + iy*CONV1_OUTPUT_SIZE + ix+1]);
        mv = max(mv, myConv1_output[ib + (iy+1)*CONV1_OUTPUT_SIZE + ix]);
        mv = max(mv, myConv1_output[ib + (iy+1)*CONV1_OUTPUT_SIZE + ix+1]);
        myPool1_output[ob + y*POOL1_OUTPUT_SIZE + x] = mv;
      }
    }
  }

  // --- Conv2: POOL1 -> CONV2_OUTPUT_SIZE x CONV2_OUTPUT_SIZE x CONV2_FILTERS ---
  // Weight layout: myConv2_w[f * CONV1_FILTERS*9 + c*9 + ky*3 + kx]  i.e. [f, c, ky, kx]
  for(int f=0; f<CONV2_FILTERS; f++) {
    int ob = f * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE;
    for(int y=0; y<CONV2_OUTPUT_SIZE; y++) {
      for(int x=0; x<CONV2_OUTPUT_SIZE; x++) {
        float sum = myConv2_b[f];
        for(int c=0; c<CONV1_FILTERS; c++) {
          int ib = c * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE;
          for(int ky=0; ky<3; ky++) {
            for(int kx=0; kx<3; kx++) {
              sum += myPool1_output[ib + (y+ky)*POOL1_OUTPUT_SIZE + (x+kx)] *
                     myConv2_w[f * CONV1_FILTERS*9 + c*9 + ky*3 + kx];
            }
          }
        }
        myConv2_output[ob + y*CONV2_OUTPUT_SIZE + x] = leaky_relu(clip_value(sum));
      }
    }
  }

  // --- FOMO head: 1x1 conv over conv2 spatial map -> myFomoMap[cls][cell] ---
  // For each spatial cell, compute a score per class as a dot product over CONV2_FILTERS.
  // Weight layout: myOutput_w[cls * CONV2_FILTERS + f]   (1x1 conv, no spatial kernel)
  // NOTE: OUTPUT_WEIGHTS is defined as FLATTENED_SIZE * NUM_CLASSES which equals
  //       CONV2_CELLS * CONV2_FILTERS * NUM_CLASSES — the 1x1 conv uses only the
  //       first CONV2_FILTERS weights per class (the spatial tiling).
  for(int cls=0; cls<NUM_CLASSES; cls++) {
    float* map = myFomoMap + cls * FOMO_CELLS;
    for(int cell=0; cell<FOMO_CELLS; cell++) {
      float sum = myOutput_b[cls];
      for(int f=0; f<CONV2_FILTERS; f++) {
        // conv2_output is stored [filter][cell]; access as [f][cell]
        sum += myConv2_output[f * FOMO_CELLS + cell] * myOutput_w[cls * CONV2_FILTERS + f];
      }
      map[cell] = my_sigmoid(clip_value(sum, -15, 15));
    }
  }

  // --- Global average pool of FOMO map -> myDense_output (used in validation) ---
  for(int cls=0; cls<NUM_CLASSES; cls++) {
    float* map = myFomoMap + cls * FOMO_CELLS;
    float s = 0;
    for(int cell=0; cell<FOMO_CELLS; cell++) s += map[cell];
    myDense_output[cls] = s / FOMO_CELLS;
  }
}

// ======================================================
// BOX TARGET MAP
// Fills every FOMO grid cell whose centre falls inside the
// bounding box with 1.0; all other cells are 0.0.
// Multiple boxes are OR-ed (any cell inside any box = 1.0).
// Background images (no boxes) → all-zero map.
//
// Default box used when annotations.json is absent:
//   x1=0.35 y1=0.35 x2=0.65 y2=0.65  (centred 30% region)
// ======================================================
const FomoBox myDefaultBox = { 0.35f, 0.35f, 0.65f, 0.65f };

void myMakeBoxTarget(float* map, const std::vector<FomoBox>& boxes) {
  memset(map, 0, FOMO_CELLS * sizeof(float));
  for (const FomoBox& b : boxes) {
    // Convert fraction to inclusive FOMO grid cell range
    int gx1 = (int)(b.x1 * FOMO_GRID);
    int gy1 = (int)(b.y1 * FOMO_GRID);
    int gx2 = (int)(b.x2 * FOMO_GRID);
    int gy2 = (int)(b.y2 * FOMO_GRID);
    gx1 = constrain(gx1, 0, FOMO_GRID - 1);
    gy1 = constrain(gy1, 0, FOMO_GRID - 1);
    gx2 = constrain(gx2, 0, FOMO_GRID - 1);
    gy2 = constrain(gy2, 0, FOMO_GRID - 1);
    for (int gy = gy1; gy <= gy2; gy++) {
      for (int gx = gx1; gx <= gx2; gx++) {
        map[gy * FOMO_GRID + gx] = 1.0f;
      }
    }
  }
}

// Legacy stub — kept so forward-declaration still resolves
void myMakeGaussianTarget(float* map, int cx, int cy) {
  // Wrapped into box target: treat single centroid as 1-cell box
  FomoBox b;
  b.x1 = (float)cx       / FOMO_GRID;
  b.y1 = (float)cy       / FOMO_GRID;
  b.x2 = (float)(cx + 1) / FOMO_GRID;
  b.y2 = (float)(cy + 1) / FOMO_GRID;
  std::vector<FomoBox> v = { b };
  myMakeBoxTarget(map, v);
}

// ======================================================
// BACKWARD PASS — FOMO HEAD (1x1 conv + sigmoid)
// Fills myConv2_grad (gradient w.r.t. myConv2_output).
// Accumulates into myOutput_w_grad and myOutput_b_grad.
// ======================================================
void myBackwardFomoHead(int label, float* targetMap) {
  // Zero myConv2_grad — it is a per-image propagation signal
  memset(myConv2_grad, 0, CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * CONV2_FILTERS * sizeof(float));

  for(int cls=0; cls<NUM_CLASSES; cls++) {
    float* map = myFomoMap + cls * FOMO_CELLS;
    for(int cell=0; cell<FOMO_CELLS; cell++) {
      float tgt  = (cls == label) ? targetMap[cell] : 0.0f;
      float diff = (map[cell] - tgt) * my_sigmoid_deriv(map[cell]);
      float dL   = 2.0f * diff / (float)(FOMO_CELLS * NUM_CLASSES);

      // Accumulate weight gradients
      myOutput_b_grad[cls] += dL;
      for(int f=0; f<CONV2_FILTERS; f++) {
        myOutput_w_grad[cls * CONV2_FILTERS + f] += dL * myConv2_output[f * FOMO_CELLS + cell];
        // Propagate into conv2_grad
        myConv2_grad[f * FOMO_CELLS + cell] += dL * myOutput_w[cls * CONV2_FILTERS + f];
      }
    }
  }
}

// ======================================================
// BACKWARD PASS — CONV2 (leaky ReLU + weight gradients)
// Reads myConv2_grad, fills myPool1_grad,
// accumulates myConv2_w_grad, myConv2_b_grad.
// ======================================================
void myBackwardConv2() {
  // Apply leaky-relu derivative to myConv2_grad in-place
  for(int i=0; i<CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * CONV2_FILTERS; i++) {
    myConv2_grad[i] *= leaky_relu_deriv(myConv2_output[i]);
  }

  // Zero Pool1 grad (per-image signal)
  memset(myPool1_grad, 0, POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE * CONV1_FILTERS * sizeof(float));

  for(int f=0; f<CONV2_FILTERS; f++) {
    int ob = f * CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE;
    for(int y=0; y<CONV2_OUTPUT_SIZE; y++) {
      for(int x=0; x<CONV2_OUTPUT_SIZE; x++) {
        float grad = myConv2_grad[ob + y*CONV2_OUTPUT_SIZE + x];
        myConv2_b_grad[f] += grad;
        for(int c=0; c<CONV1_FILTERS; c++) {
          int ib = c * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE;
          for(int ky=0; ky<3; ky++) {
            for(int kx=0; kx<3; kx++) {
              int pi = ib + (y+ky)*POOL1_OUTPUT_SIZE + (x+kx);
              int wi = f * CONV1_FILTERS*9 + c*9 + ky*3 + kx;
              myConv2_w_grad[wi] += grad * myPool1_output[pi];
              myPool1_grad[pi]   += grad * myConv2_w[wi];
            }
          }
        }
      }
    }
  }
}

// ======================================================
// BACKWARD PASS — POOL1 (max-pool, gradient routing)
// Reads myPool1_grad, fills myConv1_grad.
// ======================================================
void myBackwardPool1() {
  memset(myConv1_grad, 0, CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE * CONV1_FILTERS * sizeof(float));
  for(int f=0; f<CONV1_FILTERS; f++) {
    int ib = f * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE;
    int ob = f * POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE;
    for(int y=0; y<POOL1_OUTPUT_SIZE; y++) {
      for(int x=0; x<POOL1_OUTPUT_SIZE; x++) {
        int iy=y*2, ix=x*2;
        float poolVal = myPool1_output[ob + y*POOL1_OUTPUT_SIZE + x];
        float grad    = myPool1_grad[ob + y*POOL1_OUTPUT_SIZE + x];
        // Route gradient back to whichever input pixel was the max
        if(myConv1_output[ib + iy*CONV1_OUTPUT_SIZE + ix]     == poolVal) myConv1_grad[ib + iy*CONV1_OUTPUT_SIZE + ix]     += grad;
        if(myConv1_output[ib + iy*CONV1_OUTPUT_SIZE + ix+1]   == poolVal) myConv1_grad[ib + iy*CONV1_OUTPUT_SIZE + ix+1]   += grad;
        if(myConv1_output[ib + (iy+1)*CONV1_OUTPUT_SIZE + ix] == poolVal) myConv1_grad[ib + (iy+1)*CONV1_OUTPUT_SIZE + ix] += grad;
        if(myConv1_output[ib + (iy+1)*CONV1_OUTPUT_SIZE + ix+1] == poolVal) myConv1_grad[ib + (iy+1)*CONV1_OUTPUT_SIZE + ix+1] += grad;
      }
    }
  }
}

// ======================================================
// BACKWARD PASS — CONV1 (leaky ReLU + weight gradients)
// ======================================================
void myBackwardConv1() {
  for(int i=0; i<CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE * CONV1_FILTERS; i++) {
    myConv1_grad[i] *= leaky_relu_deriv(myConv1_output[i]);
  }
  for(int f=0; f<CONV1_FILTERS; f++) {
    int ob = f * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE;
    for(int y=0; y<CONV1_OUTPUT_SIZE; y++) {
      for(int x=0; x<CONV1_OUTPUT_SIZE; x++) {
        float grad = myConv1_grad[ob + y*CONV1_OUTPUT_SIZE + x];
        myConv1_b_grad[f] += grad;
        for(int ky=0; ky<3; ky++) {
          for(int kx=0; kx<3; kx++) {
            int inPos = ((y+ky)*INPUT_SIZE + (x+kx)) * 3;
            int wPos  = f*27 + ky*9 + kx*3;
            myConv1_w_grad[wPos]   += grad * myInputBuffer[inPos];
            myConv1_w_grad[wPos+1] += grad * myInputBuffer[inPos+1];
            myConv1_w_grad[wPos+2] += grad * myInputBuffer[inPos+2];
          }
        }
      }
    }
  }
}

// ======================================================
// OPTIMIZER — Adam
// ======================================================
void myAdamUpdate(float* w, float* g, float* m, float* v, int size, int step) {
  float b1=0.9f, b2=0.999f, eps=1e-6f;
  float lr_t = LEARNING_RATE * sqrtf(1 - powf(b2, step)) / (1 - powf(b1, step));
  for(int i=0; i<size; i++) {
    m[i] = b1*m[i] + (1-b1)*g[i];
    v[i] = b2*v[i] + (1-b2)*g[i]*g[i];
    w[i] -= lr_t * m[i] / (sqrtf(v[i]) + eps);
    w[i]  = clip_value(w[i], -10, 10);
  }
}

void myUpdateWeights(int step) {
  myAdamUpdate(myConv1_w,  myConv1_w_grad,  myConv1_w_m,  myConv1_w_v,  CONV1_WEIGHTS,  step);
  myAdamUpdate(myConv1_b,  myConv1_b_grad,  myConv1_b_m,  myConv1_b_v,  CONV1_FILTERS,  step);
  myAdamUpdate(myConv2_w,  myConv2_w_grad,  myConv2_w_m,  myConv2_w_v,  CONV2_WEIGHTS,  step);
  myAdamUpdate(myConv2_b,  myConv2_b_grad,  myConv2_b_m,  myConv2_b_v,  CONV2_FILTERS,  step);
  // 1x1 conv head: only CONV2_FILTERS weights per class are used
  myAdamUpdate(myOutput_w, myOutput_w_grad, myOutput_w_m, myOutput_w_v, NUM_CLASSES * CONV2_FILTERS, step);
  myAdamUpdate(myOutput_b, myOutput_b_grad, myOutput_b_m, myOutput_b_v, NUM_CLASSES,    step);
}

// ======================================================
// TRAINING FUNCTION
// ======================================================
void myActionTrain() {
  if (!mySDavailable) {
    Serial.println("No SD card - cannot train");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000); myResetMenuState(); return;
  }

  Serial.println("\n>>> Training mode (FOMO head)");
  Serial.println("  During training: 3+ taps = Save and exit");
  Serial.println("  After completion: TAP = Train again, 3+ taps = Exit");
  Serial.println("  Serial: 'T'=train again, 'L'=exit");

  myResetTouchState();

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 12, "TRAINING MODE");
    u8g2.drawStr(0, 24, "Loading...");
  } while (u8g2.nextPage());

  if (myLoadWeights()) Serial.println("Continuing from saved weights");
  else                  Serial.println("Starting fresh training");

  while (true) {
    myTrainingData.clear();
    for(int i=0; i<NUM_CLASSES; i++) {
      // Load annotations.json for this class (empty string if absent)
      String jsonText = (i == 0) ? "" : myLoadAnnotationsJson(myClassLabels[i]);
      bool hasJson = (jsonText.length() > 0);

      File root = SD.open("/images/" + myClassLabels[i]);
      if (root) {
        while(File file = root.openNextFile()) {
          if(!file.isDirectory()) {
            String fn = String(file.name());
            if(fn.endsWith(".jpg") || fn.endsWith(".JPG")) {
              TrainingItem item;
              item.path  = String(file.path());
              item.label = i;
              if (i == 0) {
                // Background class: no boxes — all-zero target map
                // boxes vector stays empty
              } else if (hasJson) {
                item.boxes = myParseBoxesForFile(jsonText, fn);
                // If this file wasn't in the JSON, fall back to default box
                if (item.boxes.empty()) item.boxes.push_back(myDefaultBox);
              } else {
                // No annotations.json: use centred default box for all images
                item.boxes.push_back(myDefaultBox);
              }
              myTrainingData.push_back(item);
            }
          }
          file.close();
        }
        root.close();
      }
      if (hasJson) Serial.printf("Class %s: annotations.json loaded\n", myClassLabels[i].c_str());
      else if (i != 0) Serial.printf("Class %s: no annotations.json, using default box\n", myClassLabels[i].c_str());
    }

    if(myTrainingData.empty()) {
      u8g2.firstPage();
      do { u8g2.drawStr(0, 20, "No Images!"); } while (u8g2.nextPage());
      delay(2000); myResetMenuState(); return;
    }

    // Sort for deterministic validation split
    std::sort(myTrainingData.begin(), myTrainingData.end(),
              [](const TrainingItem& a, const TrainingItem& b){ return a.path < b.path; });

    std::vector<TrainingItem> myValidationData;
    if (VALIDATION_IMAGES > 0) {
      int counts[NUM_CLASSES] = {};
      for (auto& item : myTrainingData) counts[item.label]++;
      int skip[NUM_CLASSES];
      for (int c = 0; c < NUM_CLASSES; c++) skip[c] = min(VALIDATION_IMAGES, counts[c]);

      std::vector<TrainingItem> trainOnly;
      int seen[NUM_CLASSES] = {};
      for (int i = (int)myTrainingData.size() - 1; i >= 0; i--) {
        int c = myTrainingData[i].label;
        if (seen[c] < skip[c]) { myValidationData.push_back(myTrainingData[i]); seen[c]++; }
        else                    { trainOnly.push_back(myTrainingData[i]); }
      }
      myTrainingData = trainOnly;
      Serial.printf("Val: %d images  Train: %d images\n",
                    (int)myValidationData.size(), (int)myTrainingData.size());
    }

    int total = myTrainingData.size();
    int batchesPerEpoch = (total + BATCH_SIZE - 1) / BATCH_SIZE;
    int totalBatches = TARGET_EPOCHS * batchesPerEpoch;
    Serial.printf("Training: %d images, %d batches\n", total, totalBatches);

    std::vector<int> indices;
    for(int i=0; i<total; i++) indices.push_back(i);

    float runningLoss = 0;
    int lossCount = 0;

    // BUG FIX (v002): static array for targetMap was wrong size (FOMO_CELLS, not NUM_CLASSES).
    // Now heap-allocated with the correct size.
    float* myTargetMap = (float*)ps_malloc(FOMO_CELLS * sizeof(float));
    if (!myTargetMap) { Serial.println("OOM: targetMap"); myResetMenuState(); return; }

    for(int batch=0; batch<totalBatches; batch++) {

      // Exit check via serial
      if (Serial.available()) {
        char c = Serial.read();
        if (c == 'x' || c == 'X' || c == 'l' || c == 'L') {
          Serial.println("Stopping training...");
          free(myTargetMap);
          mySaveWeights(); myWeightsTrained = true; myResetMenuState(); return;
        }
      }

      // Exit check via touch
      myCheckTouchBackground();
      if (myPeekTouchAction() == 2) {
        myCheckTouchInput();
        Serial.println("Long press - stopping training");
        free(myTargetMap);
        mySaveWeights(); myWeightsTrained = true; myResetMenuState(); return;
      }

      // Shuffle at epoch start
      if(batch % batchesPerEpoch == 0) {
        int epoch = batch/batchesPerEpoch + 1;
        Serial.printf("\n--- Epoch %d/%d ---\n", epoch, TARGET_EPOCHS);
        for(int i=total-1; i>0; i--) {
          int j = random(i+1); int tmp=indices[i]; indices[i]=indices[j]; indices[j]=tmp;
        }
      }

      int batchStart = (batch % batchesPerEpoch) * BATCH_SIZE;
      int batchEnd   = min(batchStart + BATCH_SIZE, total);

      float batchLoss = 0;
      int correctCount = 0;

      // Zero ALL weight gradient buffers once per batch
      memset(myConv1_w_grad,  0, CONV1_WEIGHTS         * sizeof(float));
      memset(myConv1_b_grad,  0, CONV1_FILTERS         * sizeof(float));
      memset(myConv2_w_grad,  0, CONV2_WEIGHTS         * sizeof(float));
      memset(myConv2_b_grad,  0, CONV2_FILTERS         * sizeof(float));
      memset(myOutput_w_grad, 0, NUM_CLASSES * CONV2_FILTERS * sizeof(float));  // only 1x1 weights
      memset(myOutput_b_grad, 0, NUM_CLASSES            * sizeof(float));

      for(int i=batchStart; i<batchEnd; i++) {
        int idx = indices[i];
        TrainingItem& img = myTrainingData[idx];

        // Load image; boxes are transformed through the random crop
        std::vector<FomoBox> croppedBoxes;
        if(!myLoadImageAugmented(img.path.c_str(), myInputBuffer, img.boxes, croppedBoxes)) continue;

        myForwardPass(myInputBuffer);  // fills myFomoMap and myDense_output

        // Build target heatmap:
        //   Background (label 0) or empty boxes -> all-zero map
        //   Object classes -> rectangular 1.0 regions per box
        if (img.label == 0 || croppedBoxes.empty()) {
          memset(myTargetMap, 0, FOMO_CELLS * sizeof(float));
        } else {
          myMakeBoxTarget(myTargetMap, croppedBoxes);
        }

        // Compute MSE loss over FOMO map
        float loss = 0;
        for(int cls=0; cls<NUM_CLASSES; cls++) {
          float* map = myFomoMap + cls * FOMO_CELLS;
          for(int cell=0; cell<FOMO_CELLS; cell++) {
            float tgt = (cls == img.label) ? myTargetMap[cell] : 0.0f;
            float diff = map[cell] - tgt;
            loss += diff * diff;
          }
        }
        loss /= (FOMO_CELLS * NUM_CLASSES);
        batchLoss += loss;

        // Predict class by max average cell activation
        int pred = 0;
        float predMax = myDense_output[0];
        for(int cls=1; cls<NUM_CLASSES; cls++) {
          if(myDense_output[cls] > predMax) { predMax = myDense_output[cls]; pred = cls; }
        }
        if(pred == img.label) correctCount++;

        myBackwardFomoHead(img.label, myTargetMap);
        myBackwardConv2();
        myBackwardPool1();
        myBackwardConv1();

        // Background touch / serial check every 3 images
        if (i % 3 == 0) {
          myCheckTouchBackground();
          if (myPeekTouchAction() == 2) {
            myCheckTouchInput();
            Serial.println("Long press - stopping training");
            free(myTargetMap);
            mySaveWeights(); myWeightsTrained = true; myResetMenuState(); return;
          }
          if (Serial.available()) {
            char c = Serial.read();
            if (c == 'x' || c == 'X' || c == 'l' || c == 'L') {
              Serial.println("Stopping training...");
              free(myTargetMap);
              mySaveWeights(); myWeightsTrained = true; myResetMenuState(); return;
            }
          }
        }
      }

      myUpdateWeights(batch + 1);

      float avgLoss  = batchLoss / (batchEnd - batchStart);
      float batchAcc = (float)correctCount / (batchEnd - batchStart);
      runningLoss += avgLoss;
      lossCount++;

      if((batch+1) % 5 == 0) {
        float displayLoss = runningLoss / lossCount;
        u8g2.firstPage();
        do {
          u8g2.setFont(u8g2_font_5x7_tf);
          u8g2.setCursor(0, 12); u8g2.print("Training...");
          u8g2.setCursor(0, 24);
          u8g2.print("B:"); u8g2.print(batch+1); u8g2.print("/"); u8g2.print(totalBatches);
          u8g2.setCursor(0, 36);
          u8g2.print("L:"); u8g2.print(displayLoss, 3);
          u8g2.print(" A:"); u8g2.print((int)(batchAcc*100)); u8g2.print("%");
        } while (u8g2.nextPage());
        runningLoss = 0; lossCount = 0;
      }
      if((batch+1) % 10 == 0) {
        Serial.printf("Batch %d/%d - Loss: %.4f - Acc: %.1f%%\n",
                      batch+1, totalBatches, avgLoss, batchAcc*100);
      }
    }

    free(myTargetMap);
    Serial.println("\n--- Training Complete ---");

    // Validation pass using myDense_output (GAP of FOMO map)
    if (!myValidationData.empty()) {
      int valCorrect = 0, valCount = 0;
      for (auto& vitem : myValidationData) {
        if (!myLoadImageFromFile(vitem.path.c_str(), myInputBuffer)) continue;
        myForwardPass(myInputBuffer);
        int pred = 0;
        for (int j=1; j<NUM_CLASSES; j++)
          if (myDense_output[j] > myDense_output[pred]) pred = j;
        if (pred == vitem.label) valCorrect++;
        valCount++;
      }
      if (valCount > 0) {
        Serial.printf("Validation Accuracy: %.1f%%  (%d/%d correct)\n",
                      100.0f * valCorrect / valCount, valCorrect, valCount);
      }
    }

    mySaveWeights();
    myWeightsTrained = true;

    u8g2.firstPage();
    do {
      u8g2.drawStr(0, 12, "DONE!");
      u8g2.drawStr(0, 24, "Tap:Again");
      u8g2.drawStr(0, 36, "3+Taps:Exit");
    } while (u8g2.nextPage());

    myResetTouchState();
    Serial.println("Waiting for input...");
    Serial.println("  Serial: T=train again  L=exit");
    Serial.println("  Touch:  1-2 taps=again  3+taps=exit");
    while (true) {
      if (Serial.available()) {
        char c = Serial.read();
        if (c == 'x' || c == 'X' || c == 'l' || c == 'L') { myResetMenuState(); return; }
        else if (c == 't' || c == 'T') { break; }
      }
      int touchAction = myCheckTouchInput();
      if (touchAction == 2) { myResetMenuState(); return; }
      else if (touchAction == 1) { Serial.println("Starting new training cycle"); break; }
      delay(10);
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 3: INFERENCE FUNCTION WITH FOMO OLED OVERLAY                       ██
// ██                                                                          ██
// ██  After every 10th frame, draws the live camera image on the OLED, then   ██
// ██  overlays a small rectangle centred on the highest-confidence FOMO cell   ██
// ██  for the predicted class (if confidence exceeds myFomoThreshold).        ██
// ██  A label bar at the bottom shows class name and detected cluster count.   ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████




void myActionInfer() {
  if (!myWeightsTrained) {
    Serial.println("ERROR: No trained weights! Please train first.");
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_6x10_tf);
      u8g2.drawStr(0, 12, "No weights!");
      u8g2.drawStr(0, 24, "Train first");
    } while (u8g2.nextPage());
    delay(3000); myResetMenuState(); return;
  }

  Serial.println("\n>>> Inference mode (FOMO)");
  Serial.println("  T or L = exit to menu");
  Serial.println("  FOMO grid: " + String(FOMO_GRID) + "x" + String(FOMO_GRID) +
                 " -> " + String(FOMO_CELLS) + " cells");

  myResetTouchState();

  if (!myInputBuffer || !myDense_output) {
    Serial.println("ERROR: Memory not allocated");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "NOT READY!"); } while (u8g2.nextPage());
    delay(2000); myResetMenuState(); return;
  }

  // Pre-compute resize lookup tables (once)
  // Maps each OUTPUT pixel index to the correct SOURCE pixel in the 240x240 frame
  static int sy_lookup[INPUT_SIZE];
  static int sx_lookup[INPUT_SIZE];
  static bool lookup_initialized = false;
  if (!lookup_initialized) {
    for(int i=0; i<INPUT_SIZE; i++) {
      sy_lookup[i] = min((int)((i+0.5)*240.0/INPUT_SIZE), 239);
      sx_lookup[i] = min((int)((i+0.5)*240.0/INPUT_SIZE), 239);
    }
    lookup_initialized = true;
    Serial.println("Resize lookup tables initialized");
  }

  int oW = u8g2.getDisplayWidth();   // 72 pixels wide
  int oH = u8g2.getDisplayHeight();  // 40 pixels tall
  int scX = 240 / oW;                // how many camera pixels per OLED pixel (x)
  int scY = 240 / oH;                // how many camera pixels per OLED pixel (y)

  unsigned long frameTimes[10];
  int frameIndex = 0;
  int pred = 0;
  float* predMap = nullptr;   // points into myFomoMap for the predicted class
  int peakCell = 0;
  float peakVal = 0.0f;
  int peakGridX = 0;
  int peakGridY = 0;
  float myActiveThreshold = myUseDynamicThreshold ? myDynamicThresholdFloor : myFomoThreshold;

  while (true) {
    unsigned long frameStart = millis();

    // Serial exit check
    if (Serial.available()) {
      char c = Serial.read();
      if (c == 't' || c == 'T' || c == 'l' || c == 'L') { myResetMenuState(); return; }
    }

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { delay(10); continue; }

    if (!myRgbBuffer) { esp_camera_fb_return(fb); delay(10); continue; }

    if (fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, myRgbBuffer)) {
      myFlipImageHorizontal();

      // Resize 240x240 -> INPUT_SIZE x INPUT_SIZE and normalise into myInputBuffer
      // Each pixel is scaled from 0-255 to 0.0-1.0 (multiply by 1/255)
      for(int y=0; y<INPUT_SIZE; y++) {
        int sy = sy_lookup[y];
        for(int x=0; x<INPUT_SIZE; x++) {
          int srcIdx = (sy * 240 + sx_lookup[x]) * 3;
          int dstIdx = (y * INPUT_SIZE + x) * 3;
          myInputBuffer[dstIdx]   = myRgbBuffer[srcIdx]   * 0.003921569f;
          myInputBuffer[dstIdx+1] = myRgbBuffer[srcIdx+1] * 0.003921569f;
          myInputBuffer[dstIdx+2] = myRgbBuffer[srcIdx+2] * 0.003921569f;
        }
      }

      myForwardPass(myInputBuffer);  // fills myFomoMap[NUM_CLASSES][FOMO_CELLS] and myDense_output

      // Find the best class by highest global average confidence (GAP of FOMO map)
      pred = 0;
      for(int i=1; i<NUM_CLASSES; i++) {
        if(myDense_output[i] > myDense_output[pred]) pred = i;
      }

      // Get the FOMO map for the predicted class
      // predMap[cell] = confidence (0..1) that the object is at that grid cell
      predMap = myFomoMap + pred * FOMO_CELLS;

      // Find the single peak cell (highest confidence) for OLED and serial summary
      peakCell = 0;
      peakVal = predMap[0];
      for(int cell=1; cell<FOMO_CELLS; cell++) {
        if(predMap[cell] > peakVal) { peakVal = predMap[cell]; peakCell = cell; }
      }
      peakGridX = peakCell % FOMO_GRID;   // column in FOMO grid
      peakGridY = peakCell / FOMO_GRID;   // row    in FOMO grid

      // Compute active threshold once per frame — shared by OLED and serial cluster loops.
      // Dynamic mode: 90% of the peak cell value, floored so blank scenes don't fire everything.
      // Fixed mode: use the constant myFomoThreshold.
      myActiveThreshold = myUseDynamicThreshold
        ? max(peakVal * myDynamicThresholdRatio, myDynamicThresholdFloor)
        : myFomoThreshold;

      // Draw OLED every 10 frames (myRgbBuffer still valid here, before fb_return)
      if (frameIndex == 9) {
        u8g2.firstPage();
        do {
          // --- Draw camera image ---
          // Convert each OLED pixel position back to the nearest camera pixel,
          // average the RGB channels to get brightness, threshold to black/white
          for (int ox = 0; ox < oW; ox++) {
            for (int oy = 0; oy < oH; oy++) {
              int pi = ((oy * scY) * 240 + (ox * scX)) * 3;
              uint8_t bright = (myRgbBuffer[pi] + myRgbBuffer[pi+1] + myRgbBuffer[pi+2]) / 3;
              if (bright > 100) u8g2.drawPixel(ox, oy);
            }
          }

          // --- Cluster high-confidence FOMO cells, draw one bounding box per cluster ---
          // Cells within myMergeRadius grid steps of an existing cluster centre merge in.
          // Up to 10 clusters supported; extras are silently dropped.
          struct MyCluster { int minX, minY, maxX, maxY; float maxVal; };
          MyCluster myClusters[10];
          int myClusterCount = 0;
          const int myMergeRadius = 6;  // tune: smaller = split more, larger = merge more

          for(int cell=0; cell<FOMO_CELLS; cell++) {
            if(predMap[cell] < myActiveThreshold) continue;
            int gx = cell % FOMO_GRID;
            int gy = cell / FOMO_GRID;

            // Find the nearest existing cluster centre
            int nearest = -1;
            for(int c=0; c<myClusterCount; c++) {
              int cx = (myClusters[c].minX + myClusters[c].maxX) / 2;
              int cy = (myClusters[c].minY + myClusters[c].maxY) / 2;
              if(abs(gx-cx) <= myMergeRadius && abs(gy-cy) <= myMergeRadius) {
                nearest = c; break;
              }
            }
            if(nearest >= 0) {
              // Expand existing cluster bounds
              myClusters[nearest].minX = min(myClusters[nearest].minX, gx);
              myClusters[nearest].minY = min(myClusters[nearest].minY, gy);
              myClusters[nearest].maxX = max(myClusters[nearest].maxX, gx);
              myClusters[nearest].maxY = max(myClusters[nearest].maxY, gy);
              myClusters[nearest].maxVal = max(myClusters[nearest].maxVal, predMap[cell]);
            } else if(myClusterCount < 10) {
              myClusters[myClusterCount++] = {gx, gy, gx, gy, predMap[cell]};
            }
          }

          // Draw one bounding rectangle per cluster
          for(int c=0; c<myClusterCount; c++) {
            int rx1 = (int)(myClusters[c].minX * oW / FOMO_GRID);
            int ry1 = (int)(myClusters[c].minY * oH / FOMO_GRID);
            int rx2 = (int)((myClusters[c].maxX + 1) * oW / FOMO_GRID);
            int ry2 = (int)((myClusters[c].maxY + 1) * oH / FOMO_GRID);
            u8g2.setColorIndex(0);   
            u8g2.drawFrame(rx1+1, ry1+1, rx2 - rx1-1, ry2 - ry1-1);   // smaller black rectangle         
            u8g2.setColorIndex(1);
            u8g2.drawFrame(rx1, ry1, rx2 - rx1, ry2 - ry1);           // bounding white rectangle
          }

          // --- Label bar at bottom ---
          // Show class name and cluster count only (confidence shown on serial, not redundant here)
          // removed for simplicity
          /*
          u8g2.setFont(u8g2_font_5x7_tf);
          u8g2.setColorIndex(0);
          u8g2.drawBox(0, oH - 9, oW, 9);
          u8g2.setColorIndex(1);
          char buf[24];
          snprintf(buf, sizeof(buf), "%s n=%d",
                   myClassLabels[pred].c_str(),
                   myClusterCount);
          u8g2.drawStr(1, oH - 1, buf);

          */

        } while (u8g2.nextPage());
      }
    }

    esp_camera_fb_return(fb);

    // --- Per-frame serial output ---
    frameTimes[frameIndex] = millis() - frameStart;
    float fps = 1000.0f / frameTimes[frameIndex];

    // Summary line: frame timing, predicted class, peak cell location
    Serial.printf("F%d: %lums (%.1fFPS) pred=%s(%.0f%%)",
                  frameIndex+1, frameTimes[frameIndex], fps,
                  myClassLabels[pred].c_str(), myDense_output[pred]*100);
    Serial.printf(" peak=(%d,%d)@%.2f, @%.2f", peakGridX, peakGridY, peakVal,myActiveThreshold);

    // All-class confidence summary
    Serial.print(" | ");
    for(int i=0; i<NUM_CLASSES; i++) {
      Serial.printf(" %s=%.0f%%", myClassLabels[i].c_str(), myDense_output[i]*100);
    }





// Cluster-based serial output — one line per frame showing each detected object
    // Total cluster count shown; each cluster reports its centroid and peak confidence.
    // myClusters/myClusterCount computed inside the OLED block every 10 frames;
    // on non-OLED frames recompute here for accurate per-frame serial output.
    struct MyClusterS { int minX, minY, maxX, maxY; float maxVal; };
    MyClusterS mySerClusters[10];
    int mySerClusterCount = 0;
    const int mySerMergeRadius = 6;

    for(int cell=0; cell<FOMO_CELLS; cell++) {
      if(predMap[cell] < myActiveThreshold) continue;
      int gx = cell % FOMO_GRID;
      int gy = cell / FOMO_GRID;
      int nearest = -1;
      for(int c=0; c<mySerClusterCount; c++) {
        int cx = (mySerClusters[c].minX + mySerClusters[c].maxX) / 2;
        int cy = (mySerClusters[c].minY + mySerClusters[c].maxY) / 2;
        if(abs(gx-cx) <= mySerMergeRadius && abs(gy-cy) <= mySerMergeRadius) {
          nearest = c; break;
        }
      }
      if(nearest >= 0) {
        mySerClusters[nearest].minX = min(mySerClusters[nearest].minX, gx);
        mySerClusters[nearest].minY = min(mySerClusters[nearest].minY, gy);
        mySerClusters[nearest].maxX = max(mySerClusters[nearest].maxX, gx);
        mySerClusters[nearest].maxY = max(mySerClusters[nearest].maxY, gy);
        mySerClusters[nearest].maxVal = max(mySerClusters[nearest].maxVal, predMap[cell]);
      } else if(mySerClusterCount < 10) {
        mySerClusters[mySerClusterCount++] = {gx, gy, gx, gy, predMap[cell]};
      }
    }

    Serial.printf(" | Clusters(%d):", mySerClusterCount);
    for(int c=0; c<mySerClusterCount; c++) {
      int cx = (mySerClusters[c].minX + mySerClusters[c].maxX) / 2;
      int cy = (mySerClusters[c].minY + mySerClusters[c].maxY) / 2;
      Serial.printf(" [%d,%d]@%.2f", cx, cy, mySerClusters[c].maxVal);
    }
    if(mySerClusterCount == 0) Serial.print(" none");
    Serial.println();






    frameIndex++;
    if (frameIndex >= 10) {
      // Check for touch input to exit inference after every 10 frames
      int touchVal = myReadTouch();
      if (touchVal > myThresholdPress) {
        Serial.println("Touch detected - exiting inference");
        delay(200); myResetMenuState(); return;
      }
      frameIndex = 0;
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 4: MENU SYSTEM                                                     ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

void myResetMenuState() {
  myIsSelected = false;
  myResetTouchState();
  myLastActivityTime = millis();
  myDrawMenu();
}

void myDrawMenu() {
  Serial.println("\n=== MENU ===");
  for (int i = 1; i <= myTotalItems; i++) {
    String label =
      (i <= NUM_CLASSES) ? myClassLabels[i - 1] :
      (i == NUM_CLASSES + 1) ? "Train" : "Infer";
    if (i == myMenuIndex) Serial.print(" > ");
    else                  Serial.print("   ");
    Serial.printf("%d. %s\n", i, label.c_str());
  }
  Serial.println("Commands: t=next (tap)  l=select (longpress)");

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 8, "TAP:Next HOLD:Ok");

    int myStartItem = (myMenuIndex <= NUM_CLASSES) ? 1 : myMenuIndex - 2;
    for (int i = 0; i < 3; i++) {
      int cur = myStartItem + i;
      if (cur > myTotalItems) break;
      String label =
        (cur <= NUM_CLASSES) ? myClassLabels[cur - 1] :
        (cur == NUM_CLASSES + 1) ? "Train" : "Infer";
      int y = 18 + i * 9;
      if (cur == myMenuIndex) u8g2.drawStr(0, y, ("> " + label).c_str());
      else                    u8g2.drawStr(0, y, ("  " + label).c_str());
    }
  } while (u8g2.nextPage());
}

void myExecuteMenuItem(int idx) {
  if (idx <= NUM_CLASSES)        myActionCollect(idx - 1);
  else if (idx == NUM_CLASSES+1) myActionTrain();
  else                           myActionInfer();
}

void myHandleMenuNavigation() {
  unsigned long myCurrentMillis = millis();

  if (!myIsSelected && Serial.available()) {
    char c = Serial.read();
    if (c >= '1' && c <= '9') {
      int newIndex = c - '0';
      if (newIndex <= myTotalItems) {
        myMenuIndex = newIndex;
        myIsSelected = true;
        myLastActivityTime = myCurrentMillis;
        myExecuteMenuItem(myMenuIndex);
      }
    }
    else if (c == 't' || c == 'T') {
      if (myCurrentMillis - myLastTapTime > myTapCooldown) {
        myMenuIndex++;
        if (myMenuIndex > myTotalItems) myMenuIndex = 1;
        myDrawMenu();
        myLastTapTime = myCurrentMillis;
        myLastActivityTime = myCurrentMillis;
      }
    }
    else if (c == 'l' || c == 'L') {
      myIsSelected = true;
      myLastActivityTime = myCurrentMillis;
      myExecuteMenuItem(myMenuIndex);
    }
  }

  if (!myIsSelected) {
    int touchAction = myCheckTouchInput();
    if (touchAction == 1) {
      if (myCurrentMillis - myLastTapTime > myTapCooldown) {
        myMenuIndex++;
        if (myMenuIndex > myTotalItems) myMenuIndex = 1;
        myDrawMenu();
        myLastTapTime = myCurrentMillis;
        myLastActivityTime = myCurrentMillis;
      }
    }
    else if (touchAction == 2) {
      myIsSelected = true;
      myLastActivityTime = myCurrentMillis;
      myExecuteMenuItem(myMenuIndex);
    }
  }
}
