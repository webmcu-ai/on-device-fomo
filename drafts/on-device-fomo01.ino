// ======================================================
// FOMO CONVERSION PATCH for firmware v44
// Jeremy Ellis — webmcu-ai
//
// What changes and why:
//
//   REMOVED:  Dense layer (Flatten → NUM_CLASSES)
//             OUTPUT_WEIGHTS = FLATTENED_SIZE * NUM_CLASSES  (was 20,184 floats)
//             myDense_grad, myOutput_w_m/v, myOutput_b_m/v
//
//   ADDED:    1×1 Conv head: CONV2_FILTERS → NUM_CLASSES per spatial cell
//             OUTPUT_WEIGHTS = CONV2_FILTERS * NUM_CLASSES  (now just 24 floats for 8×3)
//             myFomoMap[CONV2_OUTPUT_SIZE × CONV2_OUTPUT_SIZE × NUM_CLASSES]
//             Centred-Gaussian target map for training
//             Peak-find + count at inference time
//
//   SIZES with INPUT_SIZE=64, CONV2_OUTPUT_SIZE=29:
//     Output map:  29 × 29 × NUM_CLASSES  =  2523 floats
//     1×1 weights: CONV2_FILTERS × NUM_CLASSES  =  8 × 3  =  24 floats
//     (old dense weights were 6728 × 3 = 20,184 floats — gone)
//
// TRAINING STRATEGY:
//   Objects collected centred in frame.
//   Target map = Gaussian blob centred at grid cell (14,14) for true class,
//   all zeros for other classes.
//   Loss = mean squared error over all cells (simple, stable on device).
//   At inference: any cell > FOMO_THRESHOLD is a detection.
//   Multiple detections of the same class → count + centroid reported.
//
// HOW TO APPLY:
//   1. Replace the relevant #define block in Part 0 with SECTION A below.
//   2. Replace the global buffer declarations with SECTION B.
//   3. Replace myForwardPass() with SECTION C.
//   4. Replace myBackwardDense/Conv2/Pool1/Conv1 and myUpdateWeights with SECTION D.
//   5. Replace mySaveWeights/myLoadWeights/myExportHeader weight I/O with SECTION E.
//   6. Replace myActionInfer() with SECTION F.
//   7. In myActionTrain() replace the loss/backward block with SECTION G.
// ======================================================


// ██████████████████████████████████████████████████████████████████████████████
// SECTION A — Replace the CNN ARCHITECTURE CONSTANTS block in Part 0
// ██████████████████████████████████████████████████████████████████████████████

// ======================================================
// CNN ARCHITECTURE CONSTANTS  (FOMO version)
// ======================================================
#define CONV1_KERNEL_SIZE 3
#define CONV1_FILTERS 4
#define CONV1_WEIGHTS (CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE * 3 * CONV1_FILTERS)

#define CONV2_KERNEL_SIZE 3
#define CONV2_FILTERS 8
#define CONV2_WEIGHTS (CONV2_KERNEL_SIZE * CONV2_KERNEL_SIZE * CONV1_FILTERS * CONV2_FILTERS)

#define CONV1_OUTPUT_SIZE (INPUT_SIZE - 2)          // 62
#define POOL1_OUTPUT_SIZE (CONV1_OUTPUT_SIZE / 2)   // 31
#define CONV2_OUTPUT_SIZE (POOL1_OUTPUT_SIZE - 2)   // 29

// FOMO: no flatten, no dense — 1×1 conv replaces it
#define FOMO_GRID   CONV2_OUTPUT_SIZE               // 29  (spatial side)
#define FOMO_CELLS  (FOMO_GRID * FOMO_GRID)         // 841 cells

// 1×1 conv weights: one weight per (input_channel, output_class)
#define OUTPUT_WEIGHTS (CONV2_FILTERS * NUM_CLASSES)  // 8×3 = 24

// Gaussian target parameters (centred training)
#define FOMO_SIGMA   4.0f      // spread of target blob in grid cells
#define FOMO_THRESHOLD 0.25f   // activation threshold for a detection at inference


// ██████████████████████████████████████████████████████████████████████████████
// SECTION B — Replace the "Forward pass buffers" + "Backward pass buffers" +
//             gradient/Adam buffer declarations in Part 0 globals
// ██████████████████████████████████████████████████████████████████████████████

// Forward pass buffers
float* myConv1_output = nullptr;   // CONV1_OUTPUT_SIZE² × CONV1_FILTERS
float* myPool1_output = nullptr;   // POOL1_OUTPUT_SIZE² × CONV1_FILTERS
float* myConv2_output = nullptr;   // CONV2_OUTPUT_SIZE² × CONV2_FILTERS
float* myFomoMap      = nullptr;   // FOMO_CELLS × NUM_CLASSES  (spatial predictions)

// Backward pass buffers
float* myFomoGrad  = nullptr;   // gradient through the 1×1 head, shape = myFomoMap
float* myConv2_grad = nullptr;
float* myPool1_grad = nullptr;
float* myConv1_grad = nullptr;

// NOTE: myDense_output is kept as a tiny NUM_CLASSES array for compatibility
// with the OLED/serial display helpers — it will hold the mean activation per class.
float* myDense_output = nullptr;   // NUM_CLASSES  (mean across grid, for display)


// ██████████████████████████████████████████████████████████████████████████████
// SECTION C — Replace myForwardPass()
// ██████████████████████████████████████████████████████████████████████████████

void myForwardPass(float* input, float* logits) {
  // ── Conv1 ──────────────────────────────────────────────────────────────────
  // INPUT_SIZE × INPUT_SIZE × 3  →  CONV1_OUTPUT_SIZE × CONV1_OUTPUT_SIZE × CONV1_FILTERS
  for(int f=0; f<CONV1_FILTERS; f++) {
    int ob = f * CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE;
    for(int y=0; y<CONV1_OUTPUT_SIZE; y++) {
      for(int x=0; x<CONV1_OUTPUT_SIZE; x++) {
        float sum = 0;
        for(int ky=0; ky<3; ky++) {
          for(int kx=0; kx<3; kx++) {
            int inPos = ((y+ky)*INPUT_SIZE + (x+kx)) * 3;
            int wPos  = f*27 + ky*9 + kx*3;
            sum += input[inPos]  * myConv1_w[wPos]   +
                   input[inPos+1]* myConv1_w[wPos+1] +
                   input[inPos+2]* myConv1_w[wPos+2];
          }
        }
        myConv1_output[ob + y*CONV1_OUTPUT_SIZE + x] =
            leaky_relu(clip_value(sum + myConv1_b[f]));
      }
    }
  }

  // ── MaxPool1 ───────────────────────────────────────────────────────────────
  // CONV1_OUTPUT_SIZE² → POOL1_OUTPUT_SIZE²  (2×2 stride-2)
  for(int f=0; f<CONV1_FILTERS; f++) {
    int ib = f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE;
    int ob = f*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
    for(int y=0; y<POOL1_OUTPUT_SIZE; y++) {
      for(int x=0; x<POOL1_OUTPUT_SIZE; x++) {
        int iy=y*2, ix=x*2;
        float mx = myConv1_output[ib + iy*CONV1_OUTPUT_SIZE + ix];
        mx = max(mx, myConv1_output[ib + iy*CONV1_OUTPUT_SIZE + ix+1]);
        mx = max(mx, myConv1_output[ib + (iy+1)*CONV1_OUTPUT_SIZE + ix]);
        mx = max(mx, myConv1_output[ib + (iy+1)*CONV1_OUTPUT_SIZE + ix+1]);
        myPool1_output[ob + y*POOL1_OUTPUT_SIZE + x] = mx;
      }
    }
  }

  // ── Conv2 ──────────────────────────────────────────────────────────────────
  // POOL1_OUTPUT_SIZE² × CONV1_FILTERS → CONV2_OUTPUT_SIZE² × CONV2_FILTERS
  for(int f=0; f<CONV2_FILTERS; f++) {
    int ob = f*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE;
    for(int y=0; y<CONV2_OUTPUT_SIZE; y++) {
      for(int x=0; x<CONV2_OUTPUT_SIZE; x++) {
        float sum = 0;
        for(int c=0; c<CONV1_FILTERS; c++) {
          int ib = c*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
          for(int ky=0; ky<3; ky++) {
            for(int kx=0; kx<3; kx++) {
              sum += myPool1_output[ib + (y+ky)*POOL1_OUTPUT_SIZE + (x+kx)] *
                     myConv2_w[f*36 + c*9 + ky*3 + kx];
            }
          }
        }
        myConv2_output[ob + y*CONV2_OUTPUT_SIZE + x] =
            leaky_relu(clip_value(sum + myConv2_b[f]));
      }
    }
  }

  // ── FOMO head: 1×1 conv → sigmoid ─────────────────────────────────────────
  // For each spatial cell (gy, gx), dot the CONV2_FILTERS feature vector
  // with the class weights → NUM_CLASSES scores → sigmoid → myFomoMap
  //
  // myFomoMap layout: [class][cell]  i.e.  cls*FOMO_CELLS + gy*FOMO_GRID + gx
  // myOutput_w layout: [cls*CONV2_FILTERS + f]  (matches myOutput_b[cls])
  memset(myDense_output, 0, NUM_CLASSES * sizeof(float));

  for(int cls=0; cls<NUM_CLASSES; cls++) {
    float meanAct = 0;
    for(int gy=0; gy<FOMO_GRID; gy++) {
      for(int gx=0; gx<FOMO_GRID; gx++) {
        float score = myOutput_b[cls];
        for(int f=0; f<CONV2_FILTERS; f++) {
          // myConv2_output is [filter][cell], cell = gy*FOMO_GRID+gx
          score += myConv2_output[f*FOMO_CELLS + gy*FOMO_GRID + gx] *
                   myOutput_w[cls*CONV2_FILTERS + f];
        }
        // sigmoid activation — keeps output in (0,1) for MSE target matching
        float act = 1.0f / (1.0f + expf(-clip_value(score)));
        myFomoMap[cls*FOMO_CELLS + gy*FOMO_GRID + gx] = act;
        meanAct += act;
      }
    }
    // Mean activation across grid → displayed like the old softmax probability
    myDense_output[cls] = meanAct / FOMO_CELLS;

    // logits[] kept for API compatibility (not used for FOMO loss)
    logits[cls] = myDense_output[cls];
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// SECTION D — Replace all backward/optimizer functions
// ██████████████████████████████████████████████████████████████████████████████

// ── Gaussian target map helper ─────────────────────────────────────────────
// Fills targetMap[FOMO_CELLS] with a Gaussian blob centred at (cx,cy).
// Call once per training image. cx=cy=FOMO_GRID/2 for centred training.
void myMakeGaussianTarget(float* targetMap, int cx, int cy) {
  float s2 = 2.0f * FOMO_SIGMA * FOMO_SIGMA;
  for(int gy=0; gy<FOMO_GRID; gy++) {
    for(int gx=0; gx<FOMO_GRID; gx++) {
      float dy = gy - cy, dx = gx - cx;
      targetMap[gy*FOMO_GRID + gx] = expf(-(dx*dx + dy*dy) / s2);
    }
  }
}

// ── Backward through the 1×1 FOMO head ────────────────────────────────────
// loss = MSE over all cells and classes vs Gaussian targets
// dLoss/d(score_before_sigmoid) = (pred - target) * sigmoid_deriv
//                                = (pred - target) * pred * (1-pred)
//
// label    : true class index
// targetMap: pre-built Gaussian [FOMO_CELLS] for label class (others = 0 target)
void myBackwardFomoHead(int label, float* targetMap) {
  // myFomoGrad will hold dL/d(conv2_output_channel_f_at_cell)
  // — accumulated from all classes — used by myBackwardConv2 below
  memset(myFomoGrad, 0, FOMO_CELLS * CONV2_FILTERS * sizeof(float));

  for(int cls=0; cls<NUM_CLASSES; cls++) {
    float clsTarget = (cls == label) ? 1.0f : 0.0f;  // 1 if true class, else 0 map

    for(int cell=0; cell<FOMO_CELLS; cell++) {
      float pred = myFomoMap[cls*FOMO_CELLS + cell];
      float tgt  = (cls == label) ? targetMap[cell] : 0.0f;

      // MSE gradient × sigmoid derivative
      float dact = (pred - tgt) * pred * (1.0f - pred);

      // Accumulate weight gradient: dL/dW[cls,f] += dact * conv2_output[f,cell]
      myOutput_b_grad[cls] += dact;
      for(int f=0; f<CONV2_FILTERS; f++) {
        myOutput_w_grad[cls*CONV2_FILTERS + f] +=
            dact * myConv2_output[f*FOMO_CELLS + cell];
        // Propagate back into conv2 feature map
        myFomoGrad[f*FOMO_CELLS + cell] +=
            dact * myOutput_w[cls*CONV2_FILTERS + f];
      }
    }
  }
}

// ── Backward Conv2 ────────────────────────────────────────────────────────
// Almost identical to v44 except the incoming gradient is now myFomoGrad
// (which has the same shape as myConv2_output: [filter][cell])
void myBackwardConv2() {
  // myFomoGrad is the upstream gradient. Apply leaky_relu deriv.
  // We reuse myConv2_grad as the post-activation gradient buffer.
  for(int i=0; i<CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE*CONV2_FILTERS; i++) {
    myConv2_grad[i] = myFomoGrad[i] * leaky_relu_deriv(myConv2_output[i]);
  }

  memset(myPool1_grad, 0,
         POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));

  for(int f=0; f<CONV2_FILTERS; f++) {
    int ob = f*CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE;
    for(int y=0; y<CONV2_OUTPUT_SIZE; y++) {
      for(int x=0; x<CONV2_OUTPUT_SIZE; x++) {
        float grad = myConv2_grad[ob + y*CONV2_OUTPUT_SIZE + x];
        myConv2_b_grad[f] += grad;
        for(int c=0; c<CONV1_FILTERS; c++) {
          int ib = c*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
          for(int ky=0; ky<3; ky++) {
            for(int kx=0; kx<3; kx++) {
              int pi = ib + (y+ky)*POOL1_OUTPUT_SIZE + (x+kx);
              int wi = f*36 + c*9 + ky*3 + kx;
              myConv2_w_grad[wi] += grad * myPool1_output[pi];
              myPool1_grad[pi]   += grad * myConv2_w[wi];
            }
          }
        }
      }
    }
  }
}

// ── Backward Pool1 ────────────────────────────────────────────────────────
// Identical to v44 — max-pool gradient routing, no change needed.
void myBackwardPool1() {
  memset(myConv1_grad, 0,
         CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  for(int f=0; f<CONV1_FILTERS; f++) {
    int ib = f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE;
    int ob = f*POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE;
    for(int y=0; y<POOL1_OUTPUT_SIZE; y++) {
      for(int x=0; x<POOL1_OUTPUT_SIZE; x++) {
        int iy=y*2, ix=x*2;
        float poolVal = myPool1_output[ob + y*POOL1_OUTPUT_SIZE + x];
        float grad    = myPool1_grad[ob + y*POOL1_OUTPUT_SIZE + x];
        // Route gradient to whichever input was the max
        if(myConv1_output[ib+iy*CONV1_OUTPUT_SIZE+ix]     == poolVal) myConv1_grad[ib+iy*CONV1_OUTPUT_SIZE+ix]     += grad;
        if(myConv1_output[ib+iy*CONV1_OUTPUT_SIZE+ix+1]   == poolVal) myConv1_grad[ib+iy*CONV1_OUTPUT_SIZE+ix+1]   += grad;
        if(myConv1_output[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix] == poolVal) myConv1_grad[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix] += grad;
        if(myConv1_output[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix+1]==poolVal) myConv1_grad[ib+(iy+1)*CONV1_OUTPUT_SIZE+ix+1]+=grad;
      }
    }
  }
}

// ── Backward Conv1 ────────────────────────────────────────────────────────
// Identical to v44 — no change needed.
void myBackwardConv1() {
  for(int i=0; i<CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*CONV1_FILTERS; i++) {
    myConv1_grad[i] *= leaky_relu_deriv(myConv1_output[i]);
  }
  for(int f=0; f<CONV1_FILTERS; f++) {
    int ob = f*CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE;
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

// ── Optimizer — identical to v44 myAdamUpdate, kept as-is ─────────────────
// (myUpdateWeights just calls it with the new OUTPUT_WEIGHTS size)
void myUpdateWeights(int step) {
  myAdamUpdate(myConv1_w, myConv1_w_grad, myConv1_w_m, myConv1_w_v, CONV1_WEIGHTS, step);
  myAdamUpdate(myConv1_b, myConv1_b_grad, myConv1_b_m, myConv1_b_v, CONV1_FILTERS, step);
  myAdamUpdate(myConv2_w, myConv2_w_grad, myConv2_w_m, myConv2_w_v, CONV2_WEIGHTS, step);
  myAdamUpdate(myConv2_b, myConv2_b_grad, myConv2_b_m, myConv2_b_v, CONV2_FILTERS, step);
  myAdamUpdate(myOutput_w, myOutput_w_grad, myOutput_w_m, myOutput_w_v, OUTPUT_WEIGHTS, step);
  myAdamUpdate(myOutput_b, myOutput_b_grad, myOutput_b_m, myOutput_b_v, NUM_CLASSES, step);
}


// ██████████████████████████████████████████████████████████████████████████████
// SECTION E — Replace myAllocateMemory() forward/backward buffer block
//
// In myAllocateMemory() change these three lines:
//   myConv2_output = ... CONV2_OUTPUT_SIZE²×CONV2_FILTERS
//   myDense_output = ... NUM_CLASSES
//   myDense_grad   = ... FLATTENED_SIZE       ← DELETE
//   myConv2_grad   = ... CONV2_OUTPUT_SIZE²×CONV2_FILTERS
//   myPool1_grad   = ...
//   myConv1_grad   = ...
//
// To:
// ██████████████████████████████████████████████████████████████████████████████

  // (inside myAllocateMemory, replace the forward/backward buffer block)
  myConv1_output = (float*)ps_malloc(CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  myPool1_output = (float*)ps_malloc(POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  myConv2_output = (float*)ps_malloc(CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE*CONV2_FILTERS*sizeof(float));
  myFomoMap      = (float*)ps_malloc(FOMO_CELLS * NUM_CLASSES * sizeof(float));  // NEW
  myDense_output = (float*)ps_malloc(NUM_CLASSES * sizeof(float));               // kept for display

  // myFomoGrad: same shape as myConv2_output — gradient flowing back from head
  myFomoGrad  = (float*)ps_malloc(FOMO_CELLS * CONV2_FILTERS * sizeof(float));  // NEW
  myConv2_grad = (float*)ps_malloc(CONV2_OUTPUT_SIZE*CONV2_OUTPUT_SIZE*CONV2_FILTERS*sizeof(float));
  myPool1_grad = (float*)ps_malloc(POOL1_OUTPUT_SIZE*POOL1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  myConv1_grad = (float*)ps_malloc(CONV1_OUTPUT_SIZE*CONV1_OUTPUT_SIZE*CONV1_FILTERS*sizeof(float));
  // myDense_grad REMOVED — no dense layer

  // Also update the null-check to include myFomoMap:
  if (!myInputBuffer || !myConv1_w || !myConv2_w || !myOutput_w ||
      !myConv1_output || !myPool1_output || !myConv2_output || !myFomoMap) {
    // ... same FATAL error handling as before
  }

  // He-init for conv layers unchanged.
  // 1×1 head weights: small random init (fan_in = CONV2_FILTERS)
  float headStd = sqrt(2.0f / CONV2_FILTERS);
  for(int i=0; i<OUTPUT_WEIGHTS; i++)
    myOutput_w[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * headStd;
  for(int i=0; i<NUM_CLASSES; i++) myOutput_b[i] = -2.0f;
  // Note: bias init to -2 → sigmoid(-2)≈0.12 → network starts near "no detection"
  // This prevents the all-detection collapse that would happen with bias=0.


// ██████████████████████████████████████████████████████████████████████████████
// SECTION F — Replace myActionInfer()
// ██████████████████████████████████████████████████████████████████████████████

void myActionInfer() {
  if (!myWeightsTrained) {
    Serial.println("ERROR: No trained weights! Train first.");
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_6x10_tf);
      u8g2.drawStr(0, 12, "No weights!");
      u8g2.drawStr(0, 24, "Train first");
    } while (u8g2.nextPage());
    delay(3000);
    myResetMenuState();
    return;
  }

  Serial.println("\n>>> FOMO Inference mode");
  Serial.println("  T or L = exit to menu");
  Serial.printf("  Grid: %d×%d  Threshold: %.2f\n", FOMO_GRID, FOMO_GRID, FOMO_THRESHOLD);
  myResetTouchState();

  if (!myInputBuffer || !myFomoMap) {
    Serial.println("ERROR: Memory not allocated");
    myResetMenuState();
    return;
  }

  // Pre-compute resize lookup tables once
  static int sy_lookup[INPUT_SIZE];
  static int sx_lookup[INPUT_SIZE];
  static bool lookup_initialized = false;
  if (!lookup_initialized) {
    for(int i=0; i<INPUT_SIZE; i++) {
      sy_lookup[i] = min((int)((i+0.5)*240.0/INPUT_SIZE), 239);
      sx_lookup[i] = min((int)((i+0.5)*240.0/INPUT_SIZE), 239);
    }
    lookup_initialized = true;
  }

  unsigned long frameTimes[10];
  int frameIndex = 0;

  while (true) {
    unsigned long frameStart = millis();

    // Serial exit
    if (Serial.available()) {
      char c = Serial.read();
      if (c=='t'||c=='T'||c=='l'||c=='L') { myResetMenuState(); return; }
    }

    // Camera frame
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { delay(10); continue; }

    if (fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, myRgbBuffer)) {

      // Resize to INPUT_SIZE × INPUT_SIZE
      for(int y=0; y<INPUT_SIZE; y++) {
        int sy_off = sy_lookup[y] * 240;
        int dst_off = y * INPUT_SIZE;
        for(int x=0; x<INPUT_SIZE; x++) {
          int srcIdx = (sy_off + sx_lookup[x]) * 3;
          int dstIdx = (dst_off + x) * 3;
          myInputBuffer[dstIdx]   = myRgbBuffer[srcIdx]   * 0.003921569f;
          myInputBuffer[dstIdx+1] = myRgbBuffer[srcIdx+1] * 0.003921569f;
          myInputBuffer[dstIdx+2] = myRgbBuffer[srcIdx+2] * 0.003921569f;
        }
      }

      // Forward pass → fills myFomoMap[cls][cell]
      float logits[NUM_CLASSES];
      myForwardPass(myInputBuffer, logits);

      // ── Peak finding ──────────────────────────────────────────────────────
      // For each class, find all cells above FOMO_THRESHOLD that are a local
      // maximum in a 3×3 neighbourhood.  Report count + centroid (x,y).
      //
      // Grid coords are in [0, FOMO_GRID-1]. Map back to pixel coords:
      //   pixel_x = (gx + 0.5) * INPUT_SIZE / FOMO_GRID
      //   pixel_y = (gy + 0.5) * INPUT_SIZE / FOMO_GRID
      //
      // We store up to MAX_DETECTIONS per class for serial output.
      const int MAX_DETECTIONS = 8;

      struct Detection {
        int gx, gy;
        float conf;
      };

      for(int cls=0; cls<NUM_CLASSES; cls++) {
        float* map = myFomoMap + cls * FOMO_CELLS;

        Detection dets[MAX_DETECTIONS];
        int nDets = 0;

        for(int gy=0; gy<FOMO_GRID && nDets<MAX_DETECTIONS; gy++) {
          for(int gx=0; gx<FOMO_GRID && nDets<MAX_DETECTIONS; gx++) {
            float val = map[gy*FOMO_GRID + gx];
            if (val < FOMO_THRESHOLD) continue;

            // Local maximum check (3×3 neighbourhood)
            bool isMax = true;
            for(int dy=-1; dy<=1 && isMax; dy++) {
              for(int dx=-1; dx<=1 && isMax; dx++) {
                if(dx==0 && dy==0) continue;
                int ny=gy+dy, nx=gx+dx;
                if(ny<0||ny>=FOMO_GRID||nx<0||nx>=FOMO_GRID) continue;
                if(map[ny*FOMO_GRID + nx] >= val) isMax = false;
              }
            }
            if (isMax) {
              dets[nDets++] = {gx, gy, val};
            }
          }
        }

        // Serial report
        if (nDets > 0) {
          Serial.printf("[FOMO] %s: %d object(s)  ", myClassLabels[cls].c_str(), nDets);
          for(int d=0; d<nDets; d++) {
            // Centroid in pixel coords (relative to INPUT_SIZE image)
            float px = (dets[d].gx + 0.5f) * INPUT_SIZE / FOMO_GRID;
            float py = (dets[d].gy + 0.5f) * INPUT_SIZE / FOMO_GRID;
            Serial.printf("  #%d(x=%.0f,y=%.0f,c=%.2f)", d+1, px, py, dets[d].conf);
          }
          Serial.println();
        }
      }

      // ── OLED display (every 10th frame) ───────────────────────────────────
      // Shows the camera image + bounding dot overlay for detections,
      // plus a text line with the highest-confidence class and count.
      if (frameIndex == 9) {
        int oW = u8g2.getDisplayWidth();   // 72
        int oH = u8g2.getDisplayHeight();  // 40
        int scX = 240 / oW;
        int scY = 240 / oH;

        // Find best class (by mean activation) for text label
        int bestCls = 0;
        for(int i=1; i<NUM_CLASSES; i++)
          if(myDense_output[i] > myDense_output[bestCls]) bestCls = i;

        // Count detections for best class
        float* bestMap = myFomoMap + bestCls * FOMO_CELLS;
        int bestCount = 0;
        for(int cell=0; cell<FOMO_CELLS; cell++)
          if(bestMap[cell] >= FOMO_THRESHOLD) bestCount++;

        u8g2.firstPage();
        do {
          // Camera image
          for(int ox=0; ox<oW; ox++) {
            for(int oy=0; oy<oH; oy++) {
              int pi = ((oy*scY)*240 + (ox*scX))*3;
              uint8_t bright = (myRgbBuffer[pi]+myRgbBuffer[pi+1]+myRgbBuffer[pi+2])/3;
              if(bright > 100) u8g2.drawPixel(ox, oy);
            }
          }

          // Draw a dot on OLED for each detection above threshold
          // Map grid coords → OLED coords
          for(int cls=0; cls<NUM_CLASSES; cls++) {
            float* map = myFomoMap + cls * FOMO_CELLS;
            for(int gy=0; gy<FOMO_GRID; gy++) {
              for(int gx=0; gx<FOMO_GRID; gx++) {
                if(map[gy*FOMO_GRID+gx] >= FOMO_THRESHOLD) {
                  int ox2 = (int)((gx+0.5f)*oW/FOMO_GRID);
                  int oy2 = (int)((gy+0.5f)*oH/FOMO_GRID);
                  // 2×2 bright dot (inverted)
                  u8g2.setColorIndex(0);
                  u8g2.drawBox(ox2-1, oy2-1, 3, 3);
                  u8g2.setColorIndex(1);
                  u8g2.drawPixel(ox2, oy2);
                }
              }
            }
          }

          // Label bar at bottom
          u8g2.setFont(u8g2_font_5x7_tf);
          u8g2.setColorIndex(0);
          u8g2.drawBox(0, oH-9, oW, 9);
          u8g2.setColorIndex(1);
          char buf[24];
          snprintf(buf, sizeof(buf), "%s n=%d", myClassLabels[bestCls].c_str(), bestCount);
          u8g2.drawStr(1, oH-1, buf);
        } while (u8g2.nextPage());
      }
    }

    esp_camera_fb_return(fb);

    frameTimes[frameIndex] = millis() - frameStart;
    float fps2 = 1000.0f / frameTimes[frameIndex];
    Serial.printf("Frame %d: %lu ms (%.1f FPS)\n",
                  frameIndex+1, frameTimes[frameIndex], fps2);
    frameIndex++;

    if (frameIndex >= 10) {
      int touchVal = myReadTouch();
      if (touchVal > myThresholdPress) {
        delay(200);
        myResetMenuState();
        return;
      }
      frameIndex = 0;
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// SECTION G — In myActionTrain(), replace the per-image loss/backward block
//
// Find this block inside the batch loop:
//   float logits[NUM_CLASSES];
//   myForwardPass(myInputBuffer, logits);
//   float loss = -log(max(myDense_output[img.label], 1e-7f));
//   ...
//   myBackwardDense(img.label);
//   myBackwardConv2();
//   myBackwardPool1();
//   myBackwardConv1();
//
// Replace with:
// ██████████████████████████████████████████████████████████████████████████████

        // ── FOMO forward + loss + backward ──────────────────────────────────
        float logits[NUM_CLASSES];
        myForwardPass(myInputBuffer, logits);

        // Build centred Gaussian target for this image's class
        // (cx, cy) = centre of the FOMO_GRID
        static float myTargetMap[FOMO_CELLS];
        int cx = FOMO_GRID / 2;
        int cy = FOMO_GRID / 2;
        myMakeGaussianTarget(myTargetMap, cx, cy);

        // MSE loss: mean over all cells, both classes
        // (true class cell vs Gaussian, other class cells vs 0)
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

        // Pseudo-accuracy: does the peak cell belong to the right class?
        // Find class with highest max activation
        int pred = 0;
        float predMax = 0;
        for(int cls=0; cls<NUM_CLASSES; cls++) {
          float* map = myFomoMap + cls * FOMO_CELLS;
          float clsMax = 0;
          for(int cell=0; cell<FOMO_CELLS; cell++)
            if(map[cell] > clsMax) clsMax = map[cell];
          if(clsMax > predMax) { predMax = clsMax; pred = cls; }
        }
        if(pred == img.label) correctCount++;

        // Backward pass (FOMO head replaces Dense)
        myBackwardFomoHead(img.label, myTargetMap);
        myBackwardConv2();
        myBackwardPool1();
        myBackwardConv1();


// ██████████████████████████████████████████████████████████████████████████████
// SECTION E2 — mySaveWeights / myLoadWeights / myExportHeader
//
// The binary format changes because OUTPUT_WEIGHTS is now 24 not 20,184.
// The read/write calls are byte-count exact so they'll be wrong if you load
// old v44 classification weights into FOMO firmware or vice versa.
// The existing save/load code is structurally fine — the #defines do the
// right thing automatically — BUT add a magic header so mismatched files
// are caught rather than loading silently corrupted weights.
//
// In mySaveWeights(), before the f.write() calls, add:
//   uint32_t magic = 0x464F4D4F;  // "FOMO"
//   f.write((uint8_t*)&magic, 4);
//
// In myLoadWeights(), before the f.read() calls, add:
//   uint32_t magic = 0;
//   f.read((uint8_t*)&magic, 4);
//   if (magic != 0x464F4D4F) {
//     Serial.println("Weight file is not FOMO format — ignoring");
//     f.close(); return false;
//   }
//
// Everything else (the f.write/f.read calls) stays byte-for-byte identical;
// the compiler will compute the new smaller OUTPUT_WEIGHTS automatically.
// ██████████████████████████████████████████████████████████████████████████████


// ██████████████████████████████████████████████████████████████████████████████
// SECTION H — In myActionTrain(), zero the gradient buffers at batch start
//
// The existing memset block zeros OUTPUT_WEIGHTS entries of myOutput_w_grad.
// With FOMO, OUTPUT_WEIGHTS is now 24 (not 20,184) so the memset is correct
// automatically.  No change needed in the memset block.
//
// The myDense_grad memset is gone — that buffer no longer exists.
// Make sure you REMOVE this line if it exists:
//   memset(myOutput_w_grad, 0, OUTPUT_WEIGHTS * sizeof(float));  // ← keep, size changed
//   memset(myDense_grad, 0, ...);                                 // ← DELETE
// ██████████████████████████████████████████████████████████████████████████████
