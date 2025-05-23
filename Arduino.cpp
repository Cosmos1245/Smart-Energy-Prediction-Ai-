const int currentPin = A0;

const int currentfanPin = A1;

const int currentironPin = A2;

const int relayLight = 4;

const int relayfan = 5;

const int relayiron=6;

const int voltagePin = A4;

const int voltageSamples = 1000;

const float vcc = 5000.0;

const float adcResolution = 1023.0;

const float voltageCalibration = 311.0;

const float sensitivity = 100.0;

float offsetVoltageL = 2500.0;

float offsetVoltageF = 2500.0;

float offsetVoltageI = 2500.0;

int blynkCommandL = 0;

18

bool abnormalConditionL = false;

int blynkCommandF = 0;

bool abnormalConditionF = false;

int blynkCommandI = 0;

bool abnormalConditionI = false;

void setup() {

Serial.begin(9600);

pinMode(relayLight, OUTPUT);

digitalWrite(relayLight, HIGH);

pinMode(relayfan, OUTPUT);

digitalWrite(relayfan, HIGH);

pinMode(relayiron, OUTPUT);

digitalWrite(relayiron, HIGH);

offsetVoltageL = getSensorOffset(currentPin);

offsetVoltageF = getSensorOffset(currentfanPin);

offsetVoltageI = getSensorOffset(currentironPin);

}

void loop() {

float acVoltage = getACVoltage();

float currentL = getCurrent(currentPin, offsetVoltageL);

float currentF = getCurrent(currentfanPin, offsetVoltageF);

19

float currentI = getCurrent(currentironPin, offsetVoltageI);

float powerL = acVoltage * currentL;

float powerF = acVoltage * currentF;

float powerI = acVoltage * currentI;

Serial.print("CUR V:"); Serial.print(acVoltage, 2);

Serial.print(",L:"); Serial.print(currentL, 2);

Serial.print(",F:"); Serial.print(currentF, 2);

Serial.print(",I:"); Serial.print(currentI, 2);

Serial.print(",PL:"); Serial.print(powerL, 2);

Serial.print(",PF:"); Serial.print(powerF, 2);

Serial.print(",PI:"); Serial.println(powerI, 2);

if (abs(currentL) > 2.0) { abnormalConditionL = true; digitalWrite(relayLight, HIGH); }

else { abnormalConditionL = false; digitalWrite(relayLight, blynkCommandL == 1 ? LOW

: HIGH); }

if (abs(currentF) > 5.0) { abnormalConditionF = true; digitalWrite(relayfan, HIGH); }

else { abnormalConditionF = false; digitalWrite(relayfan, blynkCommandF == 1 ? LOW :

HIGH); }

if (abs(currentI) > 5.0) { abnormalConditionI = true; digitalWrite(relayiron, HIGH); }

20

else { abnormalConditionI = false; digitalWrite(relayiron, blynkCommandI == 1 ? LOW :

HIGH); }

if (Serial.available()) {

String cmd = Serial.readStringUntil('\n');

if (cmd.startsWith("CMD L:")) blynkCommandL = cmd.substring(6).toInt();

else if (cmd.startsWith("CMD F:")) blynkCommandF = cmd.substring(6).toInt();

else if (cmd.startsWith("CMD I:")) blynkCommandI = cmd.substring(6).toInt();

}

delay(2000);

}

float getSensorOffset(int pin) {

long sum = 0;

const int samples = 500;

for (int i = 0; i < samples; i++) {

sum += analogRead(pin);

delay(2);

}

float average = sum / (float)samples;

return (average / 1023.0) * vcc;

}

21

float getCurrent(int pin, float offset) {

int adcValue = analogRead(pin);

float adcVoltage = (adcValue / 1023.0) * vcc;

return (adcVoltage - offset) / sensitivity;

}

float getACVoltage() {

long sum = 0;

for (int i = 0; i < voltageSamples; i++) {

int raw = analogRead(voltagePin);

int centered = raw - 512;

sum += (long)centered * centered;

delayMicroseconds(500);

}

float mean = sum / (float)voltageSamples;

float rms = sqrt(mean);

float voltage = rms * voltageCalibration / 512.0;

return voltage;

}
