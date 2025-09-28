#include "mainserver.h"
#include <WiFi.h>
#include <WebServer.h>

bool led1_state = false;
bool led2_state = false;
bool isAPMode = true;

WebServer server(80);

String ssid = "ESP32-AP123456789";
String password = "12345678";
String wifi_ssid = "";
String wifi_password = "";

unsigned long connect_start_ms = 0;
bool connecting = false;

String mainPage() {
  float temperature = 10;
  float humidity = 20;
  String led1 = led1_state ? "ON" : "OFF";
  String led2 = led2_state ? "ON" : "OFF";

  return R"rawliteral(
    <!DOCTYPE html><html><head>
      <meta name='viewport' content='width=device-width, initial-scale=1.0'>
      <title>ESP32 Dashboard</title>
      <style>
        body { font-family: Arial;text-align:center; margin:0;}
        .container { margin:20px auto; max-width:350px; background:#f9f9f9; border-radius:10px; box-shadow:0 2px 10px #ccc;padding:20px;}
        button { padding:10px 15px; margin:10px; font-size:18px;}
        #settings { float:right; background:#007bff;color:white; border-radius:4px;}
      </style>
    </head>
    <body>
      <div class='container'>
        <h2>ESP32 Dashboard</h2>
        <div>
          <b>Temperature:</b> <span id='temp'>)rawliteral" + String(temperature) + R"rawliteral(</span> &deg;C<br>
          <b>Humidity:</b> <span id='hum'>)rawliteral" + String(humidity) + R"rawliteral(</span> %<br>
        </div>
        <div>
            <button onclick='toggleLED(1)'>LED1: <span id="l1">)rawliteral" + led1 + R"rawliteral(</span></button>
            <button onclick='toggleLED(2)'>LED2: <span id="l2">)rawliteral" + led2 + R"rawliteral(</span></button>
        </div>
        <button id="settings" onclick="window.location='/settings'">&#9881; Settings</button>
      </div>
      <script>
        function toggleLED(id) {
          fetch('/toggle?led='+id)
          .then(response=>response.json())
          .then(json=>{
            document.getElementById('l1').innerText=json.led1;
            document.getElementById('l2').innerText=json.led2;
          });
        }
        setInterval(()=>{
          fetch('/sensors')
           .then(res=>res.json())
           .then(d=>{
             document.getElementById('temp').innerText=d.temp;
             document.getElementById('hum').innerText=d.hum;
           });
        },3000);
      </script>
    </body></html>
  )rawliteral";
}

String settingsPage() {
  return R"rawliteral(
    <!DOCTYPE html><html><head>
      <meta name='viewport' content='width=device-width, initial-scale=1.0'>
      <title>Settings</title>
      <style>
        body { font-family: Arial; text-align:center; margin:0;}
        .container { margin:20px auto; max-width:350px;background:#f9f9f9;border-radius:10px;box-shadow:0 2px 10px #ccc;padding:20px;}
        input[type=text], input[type=password]{width:90%;padding:10px;}
        button { padding:10px 15px; margin:10px; font-size:18px;}
      </style>
    </head>
    <body>
      <div class='container'>
        <h2>Wi-Fi Settings</h2>
        <form id="wifiForm">
          <input name="ssid" id="ssid" placeholder="SSID" required><br>
          <input name="password" id="pass" type="password" placeholder="Password" required><br><br>
          <button type="submit">Connect</button>
          <button type="button" onclick="window.location='/'">Back</button>
        </form>
        <div id="msg"></div>
      </div>
      <script>
        document.getElementById('wifiForm').onsubmit = function(e){
          e.preventDefault();
          let ssid = document.getElementById('ssid').value;
          let pass = document.getElementById('pass').value;
          fetch('/connect?ssid='+encodeURIComponent(ssid)+'&pass='+encodeURIComponent(pass))
            .then(r=>r.text())
            .then(msg=>{
              document.getElementById('msg').innerText=msg;
            });
        };
      </script>
    </body></html>
  )rawliteral";
}

// ========== Handlers ==========
void handleRoot() { server.send(200, "text/html", mainPage()); }

void handleToggle() {
  int led = server.arg("led").toInt();
  if (led == 1) led1_state = !led1_state;
  else if (led == 2) led2_state = !led2_state;
  server.send(200, "application/json",
    "{\"led1\":\"" + String(led1_state ? "ON":"OFF") +
    "\",\"led2\":\"" + String(led2_state ? "ON":"OFF") + "\"}");
}

void handleSensors() {
  float t = 10;
  float h = 20;
  String json = "{\"temp\":"+String(t)+",\"hum\":"+String(h)+"}";
  server.send(200, "application/json", json);
}

void handleSettings() { server.send(200, "text/html", settingsPage()); }

void handleConnect() {
  wifi_ssid = server.arg("ssid");
  wifi_password = server.arg("pass");
  server.send(200, "text/plain", "Connecting....");
  isAPMode = false;
  connecting = true;
  connect_start_ms = millis();
  connectToWiFi();
}

// ========== WiFi ==========
void setupServer() {
  server.on("/", HTTP_GET, handleRoot);
  server.on("/toggle", HTTP_GET, handleToggle);
  server.on("/sensors", HTTP_GET, handleSensors);
  server.on("/settings", HTTP_GET, handleSettings);
  server.on("/connect", HTTP_GET, handleConnect);
  server.begin();
}

void startAP() {
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid.c_str(), password.c_str());
  Serial.print("AP IP address: ");
  Serial.println(WiFi.softAPIP());
  isAPMode = true;
  connecting = false;
}

void connectToWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(wifi_ssid.c_str(), wifi_password.c_str());
  Serial.print("Connecting to ");
  Serial.print(wifi_ssid.c_str());
  Serial.print(wifi_password.c_str());
  
  Serial.println(wifi_ssid);
}

// ========== Main task ==========
void main_server_task(void *pvParameters){
  pinMode(BOOT_PIN, INPUT_PULLUP);

  startAP();
  setupServer();

  while(1){
    server.handleClient();

    // Nếu nhấn BOOT thì về AP mode
    if (digitalRead(BOOT_PIN) == LOW) {
      vTaskDelay(100);
      if (digitalRead(BOOT_PIN) == LOW) {
        if (!isAPMode) {
          startAP();
          setupServer();
        }
      }
    }

    // Nếu đang connect STA
    if (connecting) {
      if (WiFi.status() == WL_CONNECTED) {
        Serial.print("STA IP address: ");
        Serial.println(WiFi.localIP());
        isAPMode = false;
        connecting = false;
      } else if (millis() - connect_start_ms > 10000) { // timeout 10s
        Serial.println("WiFi connect failed! Back to AP.");
        startAP();
        setupServer();
        connecting = false;
      }
    }

    vTaskDelay(20); // avoid watchdog reset
  }
}