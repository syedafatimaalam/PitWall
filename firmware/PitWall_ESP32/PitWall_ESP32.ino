
#include <Arduino.h>

#define RF_PIN 23

void setup() {
  Serial.begin(115200);
  pinMode(RF_PIN, OUTPUT);
  digitalWrite(RF_PIN, LOW);

  Serial.println("PitWall RF HIGH/LOW Test");
}

void loop() {
  // Transmitter ON for 100 ms
  digitalWrite(RF_PIN, HIGH);
  delay(100);

  // Transmitter OFF for 900 ms
  digitalWrite(RF_PIN, LOW);
  delay(900);

  Serial.println("100 ms HIGH test completed");
}
