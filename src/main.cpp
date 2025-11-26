#include "global.h"
#include "anomaly_detect.h"
#include "led_blinky.h"
#include "neo_blinky.h"
#include "temp_humi_monitor.h"
#include "coreiot.h"
#include "task_check_info.h"
#include "task_toogle_boot.h"
#include "task_wifi.h"
#include "task_webserver.h"
#include "task_core_iot.h"

void setup()
{
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);  // add this if not elsewhere
    
    check_info_File(0);

    xTaskCreate(led_blinky, "Task LED Blink", 2048, NULL, 2, NULL);
    xTaskCreate(neo_blinky, "Task NEO Blink", 2048, NULL, 2, NULL);
    
    // ✅ 8192 minimum for TFLite inference
    xTaskCreate(temp_humi_monitor, "Task TEMP HUMI Monitor", 8192, NULL, 2, NULL);
    
    xTaskCreate(coreiot_task, "CoreIOT Task", 4096, NULL, 2, NULL);
}

void loop()
{
    if (check_info_File(1))
    {
        if (!Wifi_reconnect())
        {
            Webserver_stop();
        }
        else
        {
            //CORE_IOT_reconnect();
        }
    }
    Webserver_reconnect();
}