#!/bin/bash
# ============================================================================
# Anomaly Detection Model Pipeline
# ============================================================================
# This script runs the complete pipeline:
#   1. Train autoencoder on sensor_data.csv
#   2. Convert model to uint8 TFLite
#   3. Compute threshold using quantized model
#   4. Generate C header file for ESP32
#
# Requirements:
#   - Python 3.8+
#   - tensorflow, numpy, pandas, scikit-learn
#   - sensor_data.csv with columns: temp, hum
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
# ============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  Anomaly Detection Model Pipeline"
echo "=============================================="
echo ""

# Check for sensor_data.csv
if [ ! -f "sensor_data.csv" ]; then
    echo "ERROR: sensor_data.csv not found!"
    echo "Please place your sensor data CSV in this directory."
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found!"
    exit 1
fi

echo "[1/4] Training autoencoder and converting to TFLite..."
echo "----------------------------------------------"
python3 train_and_convert.py
echo ""

# Check if conversion succeeded
if [ ! -f "ae_uint8.tflite" ]; then
    echo "ERROR: ae_uint8.tflite not created!"
    echo "TFLite conversion may have failed."
    exit 1
fi

echo "[2/4] Computing uint8 threshold..."
echo "----------------------------------------------"
python3 compute_uint8_threshold.py
echo ""

echo "[3/4] Converting model to C header..."
echo "----------------------------------------------"
python3 convert_model_to_header.py
echo ""

echo "[4/4] Done!"
echo "=============================================="
echo ""
echo "Generated files:"
echo "  - ae_model.keras        (saved Keras model)"
echo "  - ae_float.tflite       (float TFLite model)"
echo "  - ae_uint8.tflite       (quantized TFLite model)"
echo "  - ae_model_data.h       (C header for ESP32)"
echo "  - uint8_threshold.txt   (threshold values)"
echo "  - mse_val.npy           (validation MSE array)"
echo "  - mse_uint8.npy         (uint8 model MSE array)"
echo "  - rep_windows.npy       (representative data)"
echo ""
echo "Copy these to your ESP32 project:"
echo "  - ae_model_data.h"
echo "  - Update ANOMALY_THRESHOLD in anomaly_detect.cpp"
echo ""
echo "=============================================="