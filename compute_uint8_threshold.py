# compute_uint8_threshold.py
# Calculate anomaly threshold using the uint8 quantized model (same as ESP32 will use)
#
# Usage: python compute_uint8_threshold.py
#
# Requires: tensorflow, numpy

import numpy as np
import tensorflow as tf

# ---------------- CONFIG ----------------
TFLITE_MODEL = "ae_uint8.tflite"
REP_WINDOWS_FILE = "rep_windows.npy"  # normal data used for calibration
K_FACTOR = 3.0  # threshold = mean + k * std
OUTPUT_FILE = "uint8_threshold.txt"
# ----------------------------------------

# 1) Load the quantized model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get quantization parameters
input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

print(f"Input:  scale={input_scale}, zero_point={input_zero_point}")
print(f"Output: scale={output_scale}, zero_point={output_zero_point}")

# 2) Load representative (normal) windows
rep_windows = np.load(REP_WINDOWS_FILE).astype(np.float32)
print(f"Loaded {len(rep_windows)} windows from {REP_WINDOWS_FILE}")

# 3) Run inference and compute MSE for each window
mse_list = []

for i, window in enumerate(rep_windows):
    # Quantize input: float -> uint8
    input_quantized = np.round(window / input_scale + input_zero_point).astype(np.uint8)
    input_quantized = input_quantized.reshape(1, -1)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_quantized)
    interpreter.invoke()
    output_quantized = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize output: uint8 -> float
    output_float = (output_quantized.astype(np.float32) - output_zero_point) * output_scale
    
    # Compute MSE in float domain (comparing original input to reconstructed output)
    mse = np.mean((window - output_float.flatten()) ** 2)
    mse_list.append(mse)
    
    if (i + 1) % 200 == 0:
        print(f"  Processed {i + 1}/{len(rep_windows)} windows...")

mse_array = np.array(mse_list)

# 4) Compute threshold
mean_mse = mse_array.mean()
std_mse = mse_array.std()
threshold = mean_mse + K_FACTOR * std_mse

print(f"\n{'='*50}")
print(f"Results (uint8 quantized model):")
print(f"  MSE mean:  {mean_mse:.6f}")
print(f"  MSE std:   {std_mse:.6f}")
print(f"  Threshold (mean + {K_FACTOR}*std): {threshold:.6f}")
print(f"{'='*50}")

# 5) Save threshold and stats
with open(OUTPUT_FILE, "w") as f:
    f.write(f"# uint8 model threshold stats\n")
    f.write(f"mean={mean_mse}\n")
    f.write(f"std={std_mse}\n")
    f.write(f"k={K_FACTOR}\n")
    f.write(f"threshold={threshold}\n")

print(f"\nSaved to {OUTPUT_FILE}")

# Also save as numpy for convenience
np.save("mse_uint8.npy", mse_array)
print(f"Saved MSE array to mse_uint8.npy")

# 6) Print C header snippet for ESP32
print(f"\n// Copy this to your ESP32 code:")
print(f"#define ANOMALY_THRESHOLD {threshold:.6f}f")
print(f"#define INPUT_SCALE {input_scale}f")
print(f"#define INPUT_ZERO_POINT {input_zero_point}")
print(f"#define OUTPUT_SCALE {output_scale}f")
print(f"#define OUTPUT_ZERO_POINT {output_zero_point}")
