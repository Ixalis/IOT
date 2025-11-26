#include "anomaly_detect.h"
#include <math.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "ae_model_data.h"

// Config (match training)
#define WINDOW 10
#define INP_DIM (WINDOW * 2)
#define TENSOR_ARENA_SIZE (90 * 1024)

static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// Threshold computed from uint8 quantized model
#define ANOMALY_THRESHOLD 38.759529f
#define INPUT_SCALE 0.28633996844291687f
#define INPUT_ZERO_POINT 0
#define OUTPUT_SCALE 0.2398134469985962f
#define OUTPUT_ZERO_POINT 0


// ---------------------------------------------------------------------
// INITIALIZATION
// ---------------------------------------------------------------------
void anomaly_init() {
    // load model buffer
    const tflite::Model* model = tflite::GetModel(ae_model_data);
    
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Model schema mismatch: %d vs %d\n",
                      model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Resolver (ops used by autoencoder)
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddMul();
    resolver.AddAdd();

    // Allocator (YOLO UNO requirement)
    static tflite::MicroAllocator* allocator =
        tflite::MicroAllocator::Create(tensor_arena, TENSOR_ARENA_SIZE, nullptr);

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        allocator,
        nullptr,   // error reporter (optional)
        nullptr,   // resource vars
        nullptr    // profiler
    );

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors FAILED. Increase arena or reduce model.");
        return;
    }

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.println("Anomaly model initialized successfully.");
}


// ---------------------------------------------------------------------
// QUANTIZE INPUT
// ---------------------------------------------------------------------
static void quantize_input(const float* in_floats, uint8_t* out_uint8, TfLiteTensor* tensor) {
    float scale = tensor->params.scale;
    int zp = tensor->params.zero_point;

    for (int i = 0; i < INP_DIM; ++i) {
        int q = (int)round(in_floats[i] / scale) + zp;
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        out_uint8[i] = (uint8_t)q;
    }
}


// ---------------------------------------------------------------------
// DEQUANTIZE OUTPUT
// ---------------------------------------------------------------------
static void dequantize_output(const uint8_t* in_uint8, float* out_floats, TfLiteTensor* tensor) {
    float scale = tensor->params.scale;
    int zp = tensor->params.zero_point;

    for (int i = 0; i < INP_DIM; ++i) {
        out_floats[i] = ((int)in_uint8[i] - zp) * scale;
    }
}


// ---------------------------------------------------------------------
// MSE
// ---------------------------------------------------------------------
static float compute_mse(const float* a, const float* b) {
    double s = 0;
    for (int i = 0; i < INP_DIM; i++) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return (float)(s / INP_DIM);
}


// ---------------------------------------------------------------------
// MAIN ANOMALY CHECK
// input: window of 20 floats: {t0,h0,t1,h1,...}
// ---------------------------------------------------------------------
bool anomaly_check_window(float* window_input) {
    if (!interpreter || !input_tensor) return false;

    if (input_tensor->type != kTfLiteUInt8) {
        Serial.println("Input tensor not uint8!");
        return false;
    }

    Serial.print("WIN: ");
    for (int i = 0; i < INP_DIM; ++i) {
        Serial.print(window_input[i], 4);
        Serial.print(i == INP_DIM-1 ? "\n" : ",");
    }

    // quantize
    quantize_input(window_input, input_tensor->data.uint8, input_tensor);

    // infer
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke FAILED.");
        return false;
    }

    // dequantize output
    static float recon_f[INP_DIM];
    dequantize_output(output_tensor->data.uint8, recon_f, output_tensor);

    // compare with original
    float mse = compute_mse(window_input, recon_f);
    Serial.printf("MSE(runtime)=%.5f TH=%.5f\n", mse, ANOMALY_THRESHOLD);
    
    return (mse > ANOMALY_THRESHOLD);
}