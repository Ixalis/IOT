#include "anomaly_detector.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <math.h>

#include "ae_model_data.cc"  // the xxd output; contains g_ae_uint8_tflite and length

// Config - must match training
#define WINDOW 10
#define INP_DIM (WINDOW*2)
#define TENSOR_ARENA_SIZE (80*1024)  // start here; increase if AllocateTensors fails

static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// Threshold from your PC experiments (mean + k*std). Put the value you computed earlier.
static float ANOMALY_THRESHOLD = 0.05f; // <<-- replace with real threshold from pick_threshold.py

void anomaly_init() {
  const tflite::Model* model = tflite::GetModel(g_ae_uint8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema mismatch: %d vs %d\n", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<10> resolver;
  // add ops the dense autoencoder needs
  resolver.AddFullyConnected();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddMul();
  resolver.AddAdd();
  // add more ops if converter complains

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;

  TfLiteStatus alloc = interpreter->AllocateTensors();
  if (alloc != kTfLiteOk) {
    Serial.println("AllocateTensors() failed - increase arena size or reduce model size.");
    // optionally print error details or loop
  }
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.println("Anomaly model initialized.");
}

// Helper to quantize a float array into input->data.uint8 per tensor params
static void quantize_input(const float* in_floats, uint8_t* out_uint8, TfLiteTensor* tensor) {
  float scale = tensor->params.scale;
  int zp = tensor->params.zero_point;
  int len = tensor->bytes; // for uint8, bytes == elements
  for (int i=0;i<(INP_DIM);++i) {
    int q = (int)round(in_floats[i] / scale) + zp;
    if (q < 0) q = 0;
    if (q > 255) q = 255;
    out_uint8[i] = (uint8_t)q;
  }
}

// Helper to dequantize output uint8 into float array
static void dequantize_output(const uint8_t* in_uint8, float* out_floats, TfLiteTensor* tensor) {
  float scale = tensor->params.scale;
  int zp = tensor->params.zero_point;
  for (int i=0;i<INP_DIM;++i) {
    out_floats[i] = ( (int)in_uint8[i] - zp ) * scale;
  }
}

// Compute mean squared error between two float arrays (length INP_DIM)
static float compute_mse(const float* a, const float* b) {
  double s = 0.0;
  for (int i=0;i<INP_DIM;++i) {
    double d = a[i] - b[i];
    s += d*d;
  }
  return (float)(s / INP_DIM);
}

// Call this function with a flattened window of floats: [t0,h0,t1,h1,...] length INP_DIM
bool anomaly_check_window(float *window_input) {
  if (!interpreter || !input_tensor) return false;

  // Input is uint8 quantized
  if (input_tensor->type != kTfLiteUInt8) {
    Serial.println("Model input type not uint8; adjust code.");
    return false;
  }

  // quantize input into tensor buffer
  quantize_input(window_input, input_tensor->data.uint8, input_tensor);

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Interpreter Invoke failed");
    return false;
  }

  // Model output is uint8; dequantize to float
  static float recon_f[INP_DIM];
  dequantize_output(output_tensor->data.uint8, recon_f, output_tensor);

  // compute MSE
  float mse = compute_mse(window_input, recon_f);
  // Serial.printf("MSE=%.6f\n", mse); // debugging

  return (mse > ANOMALY_THRESHOLD);
}
