# convert_quant.py
# Converts a Keras .h5 model to a fully integer-quantized uint8 TFLite model.
# Loads HDF5 with compile=False to avoid deserialization errors like
# "Could not deserialize 'keras.metrics.mse'".
#
# Usage:
# 1) Make sure ae_model.h5 exists (trained model)
# 2) Make sure rep_windows.npy exists (representative normal windows, shape (N, WINDOW*2))
#    If not, run the helper below (make_rep_windows) or create rep_windows.npy from your CSV.
# 3) python convert_quant.py

import tensorflow as tf
import numpy as np
import os
import sys

MODEL = "ae_model.h5"
WINDOW = 10
REP_WINDOWS = "rep_windows.npy"
OUT_TFLITE = "ae_uint8.tflite"

if not os.path.exists(MODEL):
    print(f"Error: model file not found: {MODEL}")
    sys.exit(1)

# Load model without trying to compile training config / metrics
print("Loading model (compile=False) ...")
model = tf.keras.models.load_model(MODEL, compile=False)
print("Model loaded.")

# Ensure representative windows exist
if not os.path.exists(REP_WINDOWS):
    print(f"Representative windows file '{REP_WINDOWS}' not found.")
    print("Attempting to generate rep_windows.npy from sensor_data.csv (if available).")
    if os.path.exists("sensor_data.csv"):
        import pandas as pd
        df = pd.read_csv("sensor_data.csv")
        if "temp" in df.columns and "hum" in df.columns:
            data = df[["temp","hum"]].values.astype("float32")
            # create windows
            n = len(data) - WINDOW + 1
            if n <= 0:
                print("Not enough rows in sensor_data.csv to create windows. Exiting.")
                sys.exit(1)
            X = np.zeros((n, WINDOW, 2), dtype=np.float32)
            for i in range(n):
                X[i] = data[i:i+WINDOW]
            Xf = X.reshape((X.shape[0], WINDOW*2))
            # take up to 1000 windows for representative dataset (prefer normal data)
            rep = Xf[:1000]
            np.save(REP_WINDOWS, rep)
            print(f"Saved representative windows to {REP_WINDOWS} with shape {rep.shape}")
        else:
            print("sensor_data.csv doesn't have required columns 'temp' and 'hum'.")
            sys.exit(1)
    else:
        print("No sensor_data.csv found. Create rep_windows.npy from your training/normal data and retry.")
        sys.exit(1)

# Load rep windows
rep_windows = np.load(REP_WINDOWS)
print(f"Loaded {REP_WINDOWS} with shape {rep_windows.shape}")

# Representative generator for quant conversion
def representative_dataset():
    # yield samples shaped (1, inp_dim) as float32
    for i in range(min(len(rep_windows), 200)):
        sample = rep_windows[i].astype(np.float32).reshape(1, rep_windows.shape[1])
        yield [sample]

print("Starting conversion with full integer quantization ...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

try:
    tflite_quant_model = converter.convert()
    with open(OUT_TFLITE, "wb") as f:
        f.write(tflite_quant_model)
    print(f"Wrote {OUT_TFLITE} (uint8, full integer quantized)")
except Exception as e:
    print("Quant conversion failed with exception:")
    print(e)
    print("\nCommon causes:")
    print("- Representative dataset shape mismatch (should be (N, WINDOW*2))")
    print("- Model uses unsupported ops for integer quantization")
    print("- TensorFlow version incompatibility")
    raise
