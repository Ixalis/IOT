#include "global.h"

#include "led_blinky.h"
#include "neo_blinky.h"
#include "temp_humi_monitor.h"
#include "mainserver.h"
#include "tinyml.h"



void setup() {
  Serial.begin(115200);

  xTaskCreate( led_blinky, "Task LED Blink" ,2048  ,NULL  ,2 , NULL);
  //xTaskCreate( neo_blinky, "Task NEO Blink" ,2048  ,NULL  ,2 , NULL);
  //xTaskCreate( temp_humi_monitor, "Task TEMP HUMI Monitor" ,2048  ,NULL  ,2 , NULL);
  //xTaskCreate( main_server_task, "Task Main Server" ,8192  ,NULL  ,2 , NULL);
  //xTaskCreate( tiny_ml_task, "Tiny ML Task" ,2048  ,NULL  ,2 , NULL);
}

void loop() {
  
}