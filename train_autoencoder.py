# train_and_convert.py
# Single-script pipeline: train autoencoder on sensor windows, save model, compute MSE,
# produce representative windows, and convert to float & uint8 tflite.
#
# Usage: python train_and_convert.py
#
# Requires: tensorflow, numpy, pandas, scikit-learn

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
CSV = "sensor_data.csv"   # must contain columns: temp,hum
WINDOW = 10
BATCH = 128
EPOCHS = 30
LATENT = 8
REP_WINDOWS_N = 1000      # number of representative windows to save for quantization
MODEL_H5 = "ae_model.h5"
AE_FLOAT_TFLITE = "ae_float.tflite"
AE_UINT8_TFLITE = "ae_uint8.tflite"
# ----------------------------------------

if not os.path.exists(CSV):
    raise SystemExit(f"CSV file not found: {CSV}")

# 1) load data
df = pd.read_csv(CSV)
if "temp" not in df.columns or "hum" not in df.columns:
    raise SystemExit("CSV must have columns: temp,hum")

data = df[["temp","hum"]].values.astype("float32")

# 2) create sliding windows
def make_windows(arr, window):
    n = len(arr) - window + 1
    if n <= 0:
        raise ValueError("Not enough data for the chosen window size")
    X = np.zeros((n, window, arr.shape[1]), dtype=np.float32)
    for i in range(n):
        X[i] = arr[i:i+window]
    return X

X = make_windows(data, WINDOW)  # shape (samples, WINDOW, 2)
Xf = X.reshape((X.shape[0], WINDOW * 2))  # flattened windows

# 3) train/val split
X_train, X_val = train_test_split(Xf, test_size=0.2, shuffle=True, random_state=42)

# 4) build tiny dense autoencoder
from tensorflow.keras import layers, models
inp_dim = WINDOW * 2
inp = keras.Input(shape=(inp_dim,), name="input_flat")
x = layers.Dense(64, activation="relu")(inp)
x = layers.Dense(32, activation="relu")(x)
latent = layers.Dense(LATENT, activation="relu", name="latent")(x)
x = layers.Dense(32, activation="relu")(latent)
x = layers.Dense(64, activation="relu")(x)
out = layers.Dense(inp_dim, activation="linear", name="reconstruct")(x)

model = models.Model(inputs=inp, outputs=out)
model.compile(optimizer="adam", loss="mse")
model.summary()

# 5) train
history = model.fit(X_train, X_train,
                    epochs=EPOCHS,
                    batch_size=BATCH,
                    validation_data=(X_val, X_val),
                    verbose=2)

# 6) save model (HDF5) and float TFLite using the in-memory model (no re-load)
model.save(MODEL_H5)
print(f"Saved {MODEL_H5}")

# save float tflite (use model in memory to avoid load problems)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float = converter.convert()
with open(AE_FLOAT_TFLITE, "wb") as f:
    f.write(tflite_float)
print(f"Wrote {AE_FLOAT_TFLITE} (float)")

# 7) compute validation MSEs and save for threshold selection
recon_val = model.predict(X_val, batch_size=256)
mse_val = ((recon_val - X_val) ** 2).mean(axis=1)
np.save("mse_val.npy", mse_val)
print("Saved mse_val.npy (use this to compute threshold: mean + k*std)")

mean_mse = mse_val.mean()
std_mse = mse_val.std()
print(f"Validation MSE stats: mean={mean_mse:.6e}, std={std_mse:.6e}")
print(f"Suggested threshold (mean + 3*std) = {mean_mse + 3*std_mse:.6e}")

# 8) prepare representative windows for quantization: use subset of X_train (assumed normal)
rep_windows = X_train.copy()
if rep_windows.shape[0] > REP_WINDOWS_N:
    rep_windows = rep_windows[:REP_WINDOWS_N]
# Save as flattened windows for consumption by conversion script if needed
np.save("rep_windows.npy", rep_windows)
print(f"Saved rep_windows.npy with shape {rep_windows.shape}")

# 9) convert to uint8 TFLite (full-integer quantization)
# representative dataset generator that yields batches shaped [1, inp_dim] as float32
def representative_data_gen():
    for i in range(min(len(rep_windows), REP_WINDOWS_N)):
        sample = rep_windows[i].astype(np.float32).reshape(1, inp_dim)
        yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# force full integer quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

try:
    tflite_uint8 = converter.convert()
    with open(AE_UINT8_TFLITE, "wb") as f:
        f.write(tflite_uint8)
    print(f"Wrote {AE_UINT8_TFLITE} (uint8, full integer quantized)")
except Exception as e:
    print("Quant conversion failed:", e)
    print("You can try increasing REP_WINDOWS_N or ensure rep_windows.npy is valid normal data.")
import numpy as np
m = np.load("mse_val.npy")
print("mean =", m.mean())
print("std  =", m.std())
print("recommended threshold =", m.mean() + 3*m.std())
