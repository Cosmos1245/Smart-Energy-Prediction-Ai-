

#define BLYNK_TEMPLATE_ID "TMPL3Uklz9HVa"

#define BLYNK_TEMPLATE_NAME "SEPS"

#define BLYNK_AUTH_TOKEN "YCxlfQUSDdy0F_pQNmKbMuQC4BJK_kUL"

#include <ESP8266WiFi.h>

#include <BlynkSimpleEsp8266.h>

#include <FirebaseESP8266.h>

#include <SoftwareSerial.h>

#include <WiFiUdp.h>

#include <NTPClient.h>

char ssid[] = "realme5g";

char pass[] = "12345678";

#define FIREBASE_HOST "https://seps-ai-default-rtdb.asiasoutheast1.firebasedatabase.app/"

#define FIREBASE_AUTH "2fsreZYcskecacz6jSAwuEkZKlewtvdzaMTbiV7E"

SoftwareSerial toArduino(D6, D7);

FirebaseData fbdo;

FirebaseAuth auth;

FirebaseConfig config;

// Blynk virtual pins

23

#define LIGHT_VIRTUAL_PIN V3

#define FAN_VIRTUAL_PIN V4

#define IRON_VIRTUAL_PIN V5

#define ANOMALY_VIRTUAL_PIN V6

#define VOLTAGE_VIRTUAL_PIN V7

#define POWER_LIGHT V11

#define POWER_FAN V12

#define POWER_IRON V13

#define TOTAL_POWER_VPIN V8

#define TOTAL_ENERGY_VPIN V9

#define ANOMALY_STATUS_VPIN V10

WiFiUDP ntpUDP;

NTPClient timeClient(ntpUDP, "pool.ntp.org", 19800);

String input = "";

// Energy tracking

float totalEnergyWh = 0;

unsigned long lastEnergyUpdate = 0;

void setup() {

Serial.begin(115200);

24

toArduino.begin(9600);

Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);

config.host = FIREBASE_HOST;

config.api_key = "";

config.signer.tokens.legacy_token = FIREBASE_AUTH;

Firebase.begin(&config, &auth);

Firebase.reconnectWiFi(true);

timeClient.begin();

timeClient.update();

}

void loop() {

Blynk.run();

static unsigned long lastNTPUpdate = 0;

if (millis() - lastNTPUpdate > 60000) {

timeClient.update();

lastNTPUpdate = millis();

}

while (toArduino.available()) {

25

char c = toArduino.read();

if (c == '\n') {

input.trim();

parseAndSendCurrent(input);

input = "";

} else {

input += c;

}

}

}

// Relay control commands from Blynk

BLYNK_WRITE(V0) { int state = param.asInt(); toArduino.println("CMD L:" + String(state));

}

BLYNK_WRITE(V1) { int state = param.asInt(); toArduino.println("CMD F:" + String(state));

}

BLYNK_WRITE(V2) { int state = param.asInt(); toArduino.println("CMD I:" + String(state));

}

// Anomaly logic

float checkAnomaly(float current) {

return (current < 0.1) ? 2.0 : 1.0;

}

26

// Main function to parse data from Arduino

void parseAndSendCurrent(String data) {

if (data.startsWith("CUR ")) {

data = data.substring(4);

float voltage = -1, currentL = -1, currentF = -1, currentI = -1;

float powerL = 0, powerF = 0, powerI = 0;

int idxV = data.indexOf("V:");

int idxL = data.indexOf("L:");

int idxF = data.indexOf("F:");

int idxI = data.indexOf("I:");

int idxPL = data.indexOf("PL:");

int idxPF = data.indexOf("PF:");

int idxPI = data.indexOf("PI:");

if (idxV != -1 && idxL != -1 && idxF != -1 && idxI != -1 && idxPL != -1 && idxPF != -1 && idxPI

!= -1) {

voltage = data.substring(idxV + 2, idxL - 1).toFloat();

currentL = data.substring(idxL + 2, idxF - 1).toFloat();

currentF = data.substring(idxF + 2, idxI - 1).toFloat();

currentI = data.substring(idxI + 2, idxPL - 1).toFloat();

powerL = data.substring(idxPL + 3, idxPF - 1).toFloat();

powerF = data.substring(idxPF + 3, idxPI - 1).toFloat();

powerI = data.substring(idxPI + 3).toFloat();

27

// Write to Blynk

Blynk.virtualWrite(VOLTAGE_VIRTUAL_PIN, voltage);

Blynk.virtualWrite(LIGHT_VIRTUAL_PIN, currentL);

Blynk.virtualWrite(FAN_VIRTUAL_PIN, currentF);

Blynk.virtualWrite(IRON_VIRTUAL_PIN, currentI);

Blynk.virtualWrite(POWER_LIGHT, powerL);

Blynk.virtualWrite(POWER_FAN, powerF);

Blynk.virtualWrite(POWER_IRON, powerI);

// Anomalies

float anomalyL = checkAnomaly(currentL);

float anomalyF = checkAnomaly(currentF);

float anomalyI = checkAnomaly(currentI);

float totalAnomaly = anomalyL + anomalyF + anomalyI;

Blynk.virtualWrite(ANOMALY_VIRTUAL_PIN, totalAnomaly);

if (anomalyL > 1.0 || anomalyF > 1.0 || anomalyI > 1.0) {

Blynk.virtualWrite(ANOMALY_STATUS_VPIN, 1); // Abnormal

} else {

Blynk.virtualWrite(ANOMALY_STATUS_VPIN, 0); // Normal

}

28

// Timestamp

unsigned long timestamp = timeClient.getEpochTime();

String readingPath = "/readings/" + String(timestamp);

String anomalyPath = "/anomalies/" + String(timestamp);

// Send to Firebase

Firebase.setFloat(fbdo, readingPath + "/voltage", voltage);

Firebase.setFloat(fbdo, readingPath + "/light", currentL);

Firebase.setFloat(fbdo, readingPath + "/fan", currentF);

Firebase.setFloat(fbdo, readingPath + "/iron", currentI);

Firebase.setFloat(fbdo, readingPath + "/light_power", powerL);

Firebase.setFloat(fbdo, readingPath + "/fan_power", powerF);

Firebase.setFloat(fbdo, readingPath + "/iron_power", powerI);

Firebase.setFloat(fbdo, anomalyPath + "/light_anomaly", anomalyL);

Firebase.setFloat(fbdo, anomalyPath + "/fan_anomaly", anomalyF);

Firebase.setFloat(fbdo, anomalyPath + "/iron_anomaly", anomalyI);

// ===== Total Power and Energy Consumption Calculation =====

float totalPower = powerL + powerF + powerI; // in Watts

Blynk.virtualWrite(TOTAL_POWER_VPIN, totalPower); // Show in Blynk

// Energy calculation every loop (approx once per second)

unsigned long now = millis();

29

if (lastEnergyUpdate == 0) lastEnergyUpdate = now;

float timeElapsedHours = (now - lastEnergyUpdate) / 3600000.0;

totalEnergyWh += totalPower * timeElapsedHours;

lastEnergyUpdate = now;

// Display and store total energy

Blynk.virtualWrite(TOTAL_ENERGY_VPIN, totalEnergyWh); // in Wh

Firebase.setFloat(fbdo, readingPath + "/total_power", totalPower);

Firebase.setFloat(fbdo, readingPath + "/total_energy_Wh", totalEnergyWh);

}

}

}
