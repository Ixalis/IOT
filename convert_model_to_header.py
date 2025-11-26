# convert_model_to_header.py
# Convert binary .tflite to C header file for ESP32
#
# Usage: python convert_model_to_header.py
#
# Output: ae_model_data.h

import os

# ---------------- CONFIG ----------------
TFLITE_MODEL = "ae_uint8.tflite"
OUTPUT_HEADER = "ae_model_data.h"
ARRAY_NAME = "ae_model_data"
# ----------------------------------------

if not os.path.exists(TFLITE_MODEL):
    raise SystemExit(f"Model not found: {TFLITE_MODEL}")

# Read binary model
with open(TFLITE_MODEL, "rb") as f:
    model_bytes = f.read()

model_size = len(model_bytes)
print(f"Read {TFLITE_MODEL}: {model_size} bytes")

# Generate C header
with open(OUTPUT_HEADER, "w") as f:
    f.write(f"// Auto-generated from {TFLITE_MODEL}\n")
    f.write(f"// Model size: {model_size} bytes\n\n")
    f.write("#ifndef AE_MODEL_DATA_H\n")
    f.write("#define AE_MODEL_DATA_H\n\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"const unsigned int {ARRAY_NAME}_len = {model_size};\n\n")
    f.write(f"alignas(8) const uint8_t {ARRAY_NAME}[] = {{\n")
    
    # Write bytes, 12 per line
    for i, byte in enumerate(model_bytes):
        if i % 12 == 0:
            f.write("    ")
        f.write(f"0x{byte:02x},")
        if i % 12 == 11:
            f.write("\n")
        else:
            f.write(" ")
    
    # Close array
    if model_size % 12 != 0:
        f.write("\n")
    f.write("};\n\n")
    f.write("#endif // AE_MODEL_DATA_H\n")

print(f"Wrote {OUTPUT_HEADER}")
print(f"\n// Usage in ESP32 code:")
print(f'#include "{OUTPUT_HEADER}"')
print(f"// Access model: {ARRAY_NAME}, {ARRAY_NAME}_len")