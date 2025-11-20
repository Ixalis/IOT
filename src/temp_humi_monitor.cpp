// temp_humi_monitor.cpp
#include "temp_humi_monitor.h"
#include <Wire.h>
#include "DHT20.h"               // your DHT20 driver
#include "LiquidCrystal_I2C.h"
#include "anomaly_detector.h"
#include "global.h"              // for glob_temperature, glob_humidity etc.
#include <math.h>

#define DHT_SDA 11
#define DHT_SCL 12
#define LCD_COLS 16
#define LCD_ROWS 2

// matching training WINDOW
#define WINDOW 10
#define INP_DIM (WINDOW * 2)
#define SAMPLE_INTERVAL_MS 5000

// If you computed threshold on PC, keep it in anomaly_detector.cpp as single source.
// Here we just act on bool returned by anomaly_check_window().

static float window_buf[INP_DIM]; // flattened: t0,h0,t1,h1,...
static DHT20 dht20;
static LiquidCrystal_I2C lcd(0x33, LCD_COLS, LCD_ROWS); // your constructor; adjust addr if needed

// push new sample into flattened sliding window
static void push_sample(float t, float h) {
  // shift left by one sample (two floats)
  for (int i = 0; i < WINDOW - 1; ++i) {
    window_buf[i * 2]     = window_buf[(i + 1) * 2];
    window_buf[i * 2 + 1] = window_buf[(i + 1) * 2 + 1];
  }
  // append new sample at tail
  window_buf[(WINDOW - 1) * 2]     = t;
  window_buf[(WINDOW - 1) * 2 + 1] = h;
}

void temp_humi_monitor(void *pvParameters) {
  // Note: Serial.begin should be called once in setup(); avoid duplicating it here.
  Wire.begin(DHT_SDA, DHT_SCL); // SDA, SCL pins
  dht20.begin();
 lcd.begin();
lcd.backlight();


  Serial.println("TempHumi Monitor task start");

  // Ensure anomaly model initialized once. If you already call anomaly_init() in setup(),
  // this call will be harmless in most implementations, otherwise it's required.
  static bool anomaly_ready = false;
  if (!anomaly_ready) {
    anomaly_init();
    anomaly_ready = true;
  }

  // Warmup: collect WINDOW valid samples
  int warm = 0;
  while (warm < WINDOW) {
    dht20.read();
    float temperature = dht20.getTemperature();
    float humidity    = dht20.getHumidity();

    if (isnan(temperature) || isnan(humidity) || temperature < -40.0f || humidity < 0.0f) {
      Serial.println("Warmup: DHT read failed, retrying...");
      vTaskDelay(pdMS_TO_TICKS(1000));
      continue;
    }

    push_sample(temperature, humidity);
    warm++;
    Serial.printf("Warmup %d/%d: T=%.2f H=%.2f\n", warm, WINDOW, temperature, humidity);
    vTaskDelay(pdMS_TO_TICKS(500));
  }

  Serial.println("Warmup complete. Starting detection loop.");
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("ANOMALY: ----");

  for (;;) {
    dht20.read();
    float temperature = dht20.getTemperature();
    float humidity    = dht20.getHumidity();

    if (isnan(temperature) || isnan(humidity)) {
      Serial.println("DHT read failed - skipping sample");
      vTaskDelay(pdMS_TO_TICKS(SAMPLE_INTERVAL_MS));
      continue;
    }

    // update globals (if other tasks expect them)
    glob_temperature = temperature;
    glob_humidity = humidity;

    // push into window and run anomaly detection
    push_sample(temperature, humidity);

    bool is_anomaly = anomaly_check_window(window_buf);

    // Serial output
    if (is_anomaly) {
      Serial.printf("[ANOMALY] T=%.2fC H=%.2f%%\n", temperature, humidity);
      // visual feedback
      digitalWrite(LED_BUILTIN, HIGH);
      // LCD update
      lcd.setCursor(0,0);
      lcd.print("ANOMALY: YES ");
      lcd.setCursor(0,1);
      lcd.printf("T%.1f H%.0f   ", temperature, humidity); // pad to clear
      // TODO: call your CORE_IOT notify function here, e.g. CORE_IOT_report_anomaly(temperature, humidity);
    } else {
      Serial.printf("Normal T=%.2fC H=%.2f%%\n", temperature, humidity);
      digitalWrite(LED_BUILTIN, LOW);
      lcd.setCursor(0,0);
      lcd.print("ANOMALY: NO  ");
      lcd.setCursor(0,1);
      lcd.printf("T%.1f H%.0f   ", temperature, humidity);
    }

    // wait until next sample
    vTaskDelay(pdMS_TO_TICKS(SAMPLE_INTERVAL_MS));
  }

  // never reached
  vTaskDelete(NULL);
}
