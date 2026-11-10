
#include <Arduino.h>

// ==========================================
// PITWALL - MILESTONE 1
// Real Wheel RPM + 433 MHz RF Transmission
// ESP32 + 2 Encoders + FS1000A
// ==========================================

// ---------- PIN CONFIGURATION ----------

#define LEFT_ENCODER  34
#define RIGHT_ENCODER 35
#define RF_DATA       23

// ---------- RF CONFIGURATION ----------

// Manchester encoding:
// 0 = LOW, HIGH
// 1 = HIGH, LOW
//
// 500 us per chip = 1000 data bits/s

const uint32_t CHIP_US = 500;

// IMPORTANT:
// Set this to the actual number of RISING edges
// produced per complete WHEEL revolution.
// 20 is only a temporary placeholder.

const float PULSES_PER_REV = 20.0f;

// ---------- VARIABLES ----------

volatile uint32_t leftPulseCount = 0;
volatile uint32_t rightPulseCount = 0;

uint8_t sequenceNumber = 0;

uint32_t lastSampleTime = 0;

const uint32_t SAMPLE_INTERVAL_MS = 1000;

// ---------- ENCODER INTERRUPTS ----------

void IRAM_ATTR leftEncoderISR() {
  leftPulseCount++;
}

void IRAM_ATTR rightEncoderISR() {
  rightPulseCount++;
}

// ---------- CRC-16/CCITT-FALSE ----------

uint16_t calculateCRC(const uint8_t *data, size_t length) {

  uint16_t crc = 0xFFFF;

  for (size_t i = 0; i < length; i++) {

    crc ^= ((uint16_t)data[i] << 8);

    for (int bit = 0; bit < 8; bit++) {

      if (crc & 0x8000) {
        crc = (crc << 1) ^ 0x1021;
      } else {
        crc <<= 1;
      }
    }
  }

  return crc;
}

// ---------- LITTLE-ENDIAN HELPERS ----------

void writeUint16(uint8_t *destination, uint16_t value) {
  destination[0] = value & 0xFF;
  destination[1] = (value >> 8) & 0xFF;
}

void writeUint32(uint8_t *destination, uint32_t value) {

  destination[0] = value & 0xFF;
  destination[1] = (value >> 8) & 0xFF;
  destination[2] = (value >> 16) & 0xFF;
  destination[3] = (value >> 24) & 0xFF;
}

// ---------- MANCHESTER ENCODING ----------

void transmitChip(bool level) {

  digitalWrite(RF_DATA, level ? HIGH : LOW);

  delayMicroseconds(CHIP_US);
}

void transmitManchesterByte(uint8_t value) {

  for (int bit = 7; bit >= 0; bit--) {

    bool bitValue = (value >> bit) & 0x01;

    // 1 -> HIGH, LOW
    // 0 -> LOW, HIGH

    transmitChip(bitValue);
    transmitChip(!bitValue);
  }
}

// ---------- RF PACKET TRANSMISSION ----------

void transmitPacket(uint16_t leftRPM,
                    uint16_t rightRPM) {

  uint8_t packet[19] = {0};

  // Bytes 0-3: Preamble
  packet[0] = 0x55;
  packet[1] = 0x55;
  packet[2] = 0x55;
  packet[3] = 0x55;

  // Bytes 4-5: Sync word
  packet[4] = 0xD3;
  packet[5] = 0x91;

  // Byte 6: Sequence
  packet[6] = sequenceNumber++;

  // Bytes 7-10: Timestamp
  writeUint32(&packet[7], millis());

  // Bytes 11-12: Left wheel RPM
  writeUint16(&packet[11], leftRPM);

  // Bytes 13-14: Right wheel RPM
  writeUint16(&packet[13], rightRPM);

  // Byte 15: Throttle
  packet[15] = 0;

  // Byte 16: Flags
  // Bit 2 = Encoder-valid
  packet[16] = 0x04;

  // Bytes 17-18: CRC
  // CRC covers bytes 6-16 only

  uint16_t crc = calculateCRC(&packet[6], 11);

  writeUint16(&packet[17], crc);

  // Print complete packet

  Serial.print("TX HEX: ");

  for (int i = 0; i < 19; i++) {

    if (packet[i] < 0x10) Serial.print("0");

    Serial.print(packet[i], HEX);
    Serial.print(" ");
  }

  Serial.println();

  // Transmit Manchester encoded packet

  for (int i = 0; i < 19; i++) {
    transmitManchesterByte(packet[i]);
  }

  // RF idle state
  digitalWrite(RF_DATA, LOW);

  Serial.print("Packet sent | Sequence: ");
  Serial.print(packet[6]);

  Serial.print(" | Left RPM: ");
  Serial.print(leftRPM);

  Serial.print(" | Right RPM: ");
  Serial.println(rightRPM);

  Serial.println("--------------------------------");
}

// ---------- SETUP ----------

void setup() {

  Serial.begin(115200);

  pinMode(LEFT_ENCODER, INPUT);
  pinMode(RIGHT_ENCODER, INPUT);

  pinMode(RF_DATA, OUTPUT);

  digitalWrite(RF_DATA, LOW);

  attachInterrupt(
    digitalPinToInterrupt(LEFT_ENCODER),
    leftEncoderISR,
    RISING
  );

  attachInterrupt(
    digitalPinToInterrupt(RIGHT_ENCODER),
    rightEncoderISR,
    RISING
  );

  lastSampleTime = millis();

  Serial.println();
  Serial.println("================================");
  Serial.println("       PITWALL TELEMETRY");
  Serial.println("     MILESTONE 1 RF TEST");
  Serial.println("================================");
  Serial.println("Rotate wheels manually.");
  Serial.println("Sending encoder data via RF...");
  Serial.println();
}

// ---------- MAIN LOOP ----------

void loop() {

  uint32_t currentTime = millis();

  uint32_t elapsedTime =
    currentTime - lastSampleTime;

  if (elapsedTime >= SAMPLE_INTERVAL_MS) {

    // Read and reset encoder counters atomically

    noInterrupts();

    uint32_t leftPulses = leftPulseCount;
    uint32_t rightPulses = rightPulseCount;

    leftPulseCount = 0;
    rightPulseCount = 0;

    interrupts();

    // Calculate wheel speed

    float leftRPM =
      (leftPulses * 60000.0f) /
      (PULSES_PER_REV * elapsedTime);

    float rightRPM =
      (rightPulses * 60000.0f) /
      (PULSES_PER_REV * elapsedTime);

    // Convert RPM into unsigned 16-bit values

    uint16_t leftValue =
      (uint16_t)constrain(leftRPM, 0.0f, 65535.0f);

    uint16_t rightValue =
      (uint16_t)constrain(rightRPM, 0.0f, 65535.0f);

    // Display real encoder readings

    Serial.println("WHEEL SENSOR DATA");

    Serial.print("Left pulses  : ");
    Serial.println(leftPulses);

    Serial.print("Right pulses : ");
    Serial.println(rightPulses);

    Serial.print("Left RPM     : ");
    Serial.println(leftRPM, 2);

    Serial.print("Right RPM    : ");
    Serial.println(rightRPM, 2);

    // Send values over RF
    transmitPacket(leftValue, rightValue);

    lastSampleTime = millis();
  }
}
