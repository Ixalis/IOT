#pragma once
#include <stdint.h>
#include <Arduino.h>

void anomaly_init();
bool anomaly_check_window(float *window_input /* length = WINDOW*2 */);
