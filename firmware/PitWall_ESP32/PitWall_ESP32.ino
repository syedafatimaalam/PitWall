#define RF_DATA 23

void setup() {
  pinMode(RF_DATA, OUTPUT);
}

void loop() {
  digitalWrite(RF_DATA, HIGH);
  delay(500);

  digitalWrite(RF_DATA, LOW);
  delay(500);
}