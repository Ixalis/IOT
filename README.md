# ESP32 Anomaly Detection with TinyML

Real-time temperature and humidity anomaly detection using an autoencoder neural network running on ESP32.

## Overview

This project trains a small autoencoder on "normal" sensor data, then deploys it to an ESP32 microcontroller. The model reconstructs input windows—if reconstruction error (MSE) exceeds a threshold, an anomaly is detected.

```
[DHT20 Sensor] → [Sliding Window] → [TFLite Autoencoder] → [MSE Check] → [Alert]
```

## Project Structure

```
project/
├── ml_pipeline/                    # Run on PC (training)
│   ├── sensor_data.csv             # Your training data
│   ├── train_and_convert.py        # Train model & convert to TFLite
│   ├── compute_uint8_threshold.py  # Calculate anomaly threshold
│   ├── convert_model_to_header.py  # Generate C header file
│   ├── run_pipeline.sh             # One-click script (Linux/macOS)
│   └── run_pipeline.bat            # One-click script (Windows)
│
├── esp32_firmware/                 # Upload to ESP32
│   ├── main.ino                    # Entry point, task creation
│   ├── anomaly_detect.h            # Anomaly detection header
│   ├── anomaly_detect.cpp          # TFLite inference code
│   ├── ae_model_data.h             # Generated model (from pipeline)
│   ├── temp_humi_monitor.h         # Sensor task header
│   ├── temp_humi_monitor.cpp       # Sensor reading + detection loop
│   └── global.h                    # Shared variables
```

## Requirements

### PC (Training)
- Python 3.8+
- TensorFlow 2.x
- NumPy, Pandas, scikit-learn

```bash
pip install tensorflow numpy pandas scikit-learn
```

### ESP32
- Arduino IDE or PlatformIO
- ESP32 board (tested on YOLO UNO)
- TensorFlow Lite Micro library
- DHT20 sensor
- 16x2 I2C LCD (optional)

## Quick Start

### 1. Prepare Training Data

Create `sensor_data.csv` with normal operating data:

```csv
temp,hum
25.3,65.2
25.4,65.1
25.2,65.3
...
```

> Collect at least 1000+ samples of **normal** behavior.

### 2. Run the Pipeline

**Linux/macOS:**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

**Windows:**
```batch
run_pipeline.bat
```

This will:
1. Train the autoencoder (30 epochs)
2. Convert to uint8 TFLite
3. Compute threshold from quantized model
4. Generate `ae_model_data.h`

### 3. Copy Files to ESP32 Project

```
ae_model_data.h → esp32_firmware/
```

Update threshold in `anomaly_detect.cpp` (printed by pipeline):
```cpp
#define ANOMALY_THRESHOLD 38.759529f
```

### 4. Upload to ESP32

Open in Arduino IDE, select your board, and upload.

## How It Works

### Training (PC)

```
Input: 10 samples × 2 features = 20 floats
       [t0,h0,t1,h1,...,t9,h9]
              ↓
       Dense(64) → Dense(32) → Dense(8) → Dense(32) → Dense(64)
              ↓
Output: Reconstructed 20 floats
```

The autoencoder learns to compress and reconstruct normal patterns. Anomalies produce high reconstruction error.

### Inference (ESP32)

```cpp
// Every 5 seconds:
1. Read DHT20 → (temp, humidity)
2. Push into sliding window buffer
3. Quantize: float → uint8
4. Run TFLite inference
5. Dequantize: uint8 → float
6. Compute MSE(input, reconstruction)
7. If MSE > threshold → ANOMALY!
```

## Configuration

### Model Parameters (`train_and_convert.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WINDOW` | 10 | Samples per window |
| `EPOCHS` | 30 | Training epochs |
| `LATENT` | 8 | Bottleneck size |
| `BATCH` | 128 | Batch size |

### Detection Parameters (`anomaly_detect.cpp`)

| Parameter | Description |
|-----------|-------------|
| `ANOMALY_THRESHOLD` | MSE threshold (from pipeline) |
| `TENSOR_ARENA_SIZE` | Memory for TFLite (90KB default) |

### Sampling (`temp_humi_monitor.cpp`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_INTERVAL_MS` | 5000 | Time between readings |
| `WINDOW` | 10 | Must match training |

## Troubleshooting

### Pipeline Issues

**TFLite conversion crashes (LLVM error on macOS):**
- The scripts use `concrete_functions` workaround
- If still failing, try: `pip install tensorflow==2.13.1`

**"CSV file not found":**
- Ensure `sensor_data.csv` is in same directory as scripts

### ESP32 Issues

**"AllocateTensors FAILED":**
- Increase `TENSOR_ARENA_SIZE` in `anomaly_detect.cpp`
- Try `100 * 1024` or `120 * 1024`

**Stack overflow / crash:**
- Increase task stack size in `main.ino`:
  ```cpp
  xTaskCreate(temp_humi_monitor, "...", 16384, NULL, 2, NULL);
  ```

**"Model schema mismatch":**
- Rebuild model with same TensorFlow version
- Check TFLite Micro library version

**Always detecting anomalies:**
- Threshold too low—retrain with more normal data
- Check sensor readings are in same range as training data

**Never detecting anomalies:**
- Threshold too high—try `K_FACTOR = 2.0` in threshold script
- Verify sensor is actually producing anomalous readings

## Serial Output Example

```
TempHumi Monitor task start
Anomaly model initialized successfully.
Warmup 1/10: T=25.30 H=65.20
Warmup 2/10: T=25.31 H=65.18
...
Warmup complete. Starting detection loop.
Normal T=25.32C H=65.15%
Normal T=25.30C H=65.20%
MSE(runtime)=2.34521 TH=38.75953
[ANOMALY] T=45.20C H=30.50%
MSE(runtime)=156.78234 TH=38.75953
```

## Memory Usage

| Component | Size |
|-----------|------|
| Model (uint8) | ~5-10 KB |
| Tensor Arena | 90 KB |
| Task Stack | 8 KB |
| Window Buffer | 80 bytes |

Total: ~100 KB RAM

## License

MIT License - Use freely for personal and commercial projects.

## Acknowledgments

- TensorFlow Lite Micro team
- ESP32 Arduino community
