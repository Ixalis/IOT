// temp_humi_monitor.cpp
#include "temp_humi_monitor.h"
#include <Wire.h>
#include "DHT20.h"
#include "LiquidCrystal_I2C.h"
#include "anomaly_detect.h"      // ✅ fixed
#include "global.h"
#include <math.h>

#define DHT_SDA 11
#define DHT_SCL 12
#define LCD_COLS 16
#define LCD_ROWS 2

#define WINDOW 10
#define INP_DIM (WINDOW * 2)
#define SAMPLE_INTERVAL_MS 5000

static float window_buf[INP_DIM];
static DHT20 dht20;
static LiquidCrystal_I2C lcd(0x33, LCD_COLS, LCD_ROWS);

static void push_sample(float t, float h) {
    for (int i = 0; i < WINDOW - 1; ++i) {
        window_buf[i * 2]     = window_buf[(i + 1) * 2];
        window_buf[i * 2 + 1] = window_buf[(i + 1) * 2 + 1];
    }
    window_buf[(WINDOW - 1) * 2]     = t;
    window_buf[(WINDOW - 1) * 2 + 1] = h;
}

void temp_humi_monitor(void *pvParameters) {
    Wire.begin(DHT_SDA, DHT_SCL);
    dht20.begin();
    lcd.begin();
    lcd.backlight();
    Serial.println("TempHumi Monitor task start");

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
    lcd.setCursor(0, 0);
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

        glob_temperature = temperature;
        glob_humidity = humidity;

        push_sample(temperature, humidity);
        bool is_anomaly = anomaly_check_window(window_buf);

        if (is_anomaly) {
            Serial.printf("[ANOMALY] T=%.2fC H=%.2f%%\n", temperature, humidity);
            digitalWrite(LED_BUILTIN, HIGH);
            lcd.setCursor(0, 0);
            lcd.print("ANOMALY: YES ");
        } else {
            Serial.printf("Normal T=%.2fC H=%.2f%%\n", temperature, humidity);
            digitalWrite(LED_BUILTIN, LOW);
            lcd.setCursor(0, 0);
            lcd.print("ANOMALY: NO  ");
        }
        
        lcd.setCursor(0, 1);
        lcd.printf("T%.1f H%.0f%%  ", temperature, humidity);

        vTaskDelay(pdMS_TO_TICKS(SAMPLE_INTERVAL_MS));
    }

    vTaskDelete(NULL);
}
