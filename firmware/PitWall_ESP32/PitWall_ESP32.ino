
#include <Arduino.h>
#include <Wire.h>
#include <DabbleESP32.h>

// ==============================================
// PITWALL - COMPLETE ESP32 CAR FIRMWARE
// ESP32 Arduino Core 2.0.17
//
// Dabble Bluetooth GamePad
// L298N - 4 DC motors
// 2 MH wheel-speed sensors
// MPU6050 / GY-521
// TCRT5000 lap sensor
// FS1000A 433 MHz RF transmitter
// ==============================================

// ------------ MOTOR PINS ------------
// Channel A = RIGHT wheels
#define IN1 26
#define IN2 27

// Channel B = LEFT wheels
#define IN3 14
#define IN4 13

// ENA and ENB jumpers remain installed.
// Do not connect GPIOs to ENA/ENB.

// ------------ WHEEL SENSORS ------------
#define RIGHT_ENCODER 34
#define LEFT_ENCODER 35

// ------------ LAP SENSOR ------------
#define LAP_SENSOR 32

// ------------ MPU6050 ------------
#define MPU_SDA 21
#define MPU_SCL 22
#define MPU_ADDR 0x68

// ------------ RF TRANSMITTER ------------
#define RF_PIN 23

// ------------ SETTINGS ------------
#define PULSES_PER_REV 1

#define SENSOR_INTERVAL_MS 100
#define RF_INTERVAL_MS 500

#define BIT_US 1000
#define CHIP_US 500

// ------------ ENCODER VARIABLES ------------
volatile uint32_t rightPulses = 0;
volatile uint32_t leftPulses = 0;

portMUX_TYPE encoderMux = portMUX_INITIALIZER_UNLOCKED;

uint16_t rightRPM = 0;
uint16_t leftRPM = 0;

// ------------ TELEMETRY VARIABLES ------------
uint8_t sequenceNumber = 0;
uint8_t throttle = 0;
uint8_t flags = 0;

uint32_t lastSensorTime = 0;
uint32_t lastRFTime = 0;

// ------------ LAP VARIABLES ------------
uint32_t lapCount = 0;
uint32_t lastLapTime = 0;

bool previousLapState = false;

// ------------ MPU6050 VARIABLES ------------
bool mpuConnected = false;
int16_t accelXmg = 0;
int16_t accelYmg = 0;
int16_t accelZmg = 0;

int16_t gyroXdps10 = 0;
int16_t gyroYdps10 = 0;
int16_t gyroZdps10 = 0;

// ------------ CAR STATE ------------
enum CarDirection {
  STOPPED,
  FORWARD,
  BACKWARD,
  TURN_LEFT,
  TURN_RIGHT
};

CarDirection currentDirection = STOPPED;

// ==============================================
// WHEEL SENSOR INTERRUPTS
// ==============================================

void IRAM_ATTR rightEncoderISR() {
  portENTER_CRITICAL_ISR(&encoderMux);
  rightPulses++;
  portEXIT_CRITICAL_ISR(&encoderMux);
}

void IRAM_ATTR leftEncoderISR() {
  portENTER_CRITICAL_ISR(&encoderMux);
  leftPulses++;
  portEXIT_CRITICAL_ISR(&encoderMux);
}

// ==============================================
// MOTOR CONTROL
// ==============================================

void stopMotors() {

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);

  throttle = 0;
  currentDirection = STOPPED;
}

void moveForward() {

  // RIGHT wheels
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  // LEFT wheels
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);

  throttle = 100;
  currentDirection = FORWARD;
}

void moveBackward() {

  // RIGHT wheels
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);

  // LEFT wheels
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);

  throttle = 100;
  currentDirection = BACKWARD;
}

void turnLeft() {

  // RIGHT wheels forward
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  // LEFT wheels backward
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);

  throttle = 100;
  currentDirection = TURN_LEFT;
}

void turnRight() {

  // RIGHT wheels backward
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);

  // LEFT wheels forward
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);

  throttle = 100;
  currentDirection = TURN_RIGHT;
}

// ==============================================
// DABBLE BLUETOOTH CONTROL
// ==============================================

void updateBluetoothControl() {

  Dabble.processInput();

  if (GamePad.isUpPressed()) {

    moveForward();

  } else if (GamePad.isDownPressed()) {

    moveBackward();

  } else if (GamePad.isLeftPressed()) {

    turnLeft();

  } else if (GamePad.isRightPressed()) {

    turnRight();

  } else {

    stopMotors();
  }
}

// ==============================================
// MPU6050 INITIALIZATION
// ==============================================

bool initializeMPU() {

  // Wake up MPU6050
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);

  if (Wire.endTransmission() != 0) {
    return false;
  }

  delay(50);

  // Accelerometer range: +/- 2g
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);
  Wire.write(0x00);

  if (Wire.endTransmission() != 0) {
    return false;
  }

  // Gyroscope range: +/- 250 deg/s
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B);
  Wire.write(0x00);

  if (Wire.endTransmission() != 0) {
    return false;
  }

  return true;
}

// ==============================================
// READ MPU6050
// ==============================================

bool readMPU() {

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);

  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom(
        (uint8_t)MPU_ADDR,
        (uint8_t)14,
        (uint8_t)true
      ) != 14) {
    return false;
  }

  int16_t ax = (int16_t)(
    (Wire.read() << 8) | Wire.read()
  );

  int16_t ay = (int16_t)(
    (Wire.read() << 8) | Wire.read()
  );

  int16_t az = (int16_t)(
    (Wire.read() << 8) | Wire.read()
  );

  // Skip temperature
  Wire.read();
  Wire.read();

  int16_t gx = (int16_t)(
    (Wire.read() << 8) | Wire.read()
  );

  int16_t gy = (int16_t)(
    (Wire.read() << 8) | Wire.read()
  );

  int16_t gz = (int16_t)(
    (Wire.read() << 8) | Wire.read()
  );

  // Convert acceleration to milligravity
  accelXmg = (int32_t)ax * 1000 / 16384;
  accelYmg = (int32_t)ay * 1000 / 16384;
  accelZmg = (int32_t)az * 1000 / 16384;

  // Convert gyro to 0.1 degrees/sec
  gyroXdps10 = (int32_t)gx * 10 / 131;
  gyroYdps10 = (int32_t)gy * 10 / 131;
  gyroZdps10 = (int32_t)gz * 10 / 131;

  return true;
}

// ==============================================
// WHEEL RPM CALCULATION
// ==============================================

void updateWheelRPM(uint32_t elapsed) {

  uint32_t leftCount;
  uint32_t rightCount;

  portENTER_CRITICAL(&encoderMux);

  leftCount = leftPulses;
  rightCount = rightPulses;

  leftPulses = 0;
  rightPulses = 0;

  portEXIT_CRITICAL(&encoderMux);

  if (elapsed == 0) {
    elapsed = 1;
  }

  uint64_t denominator =
    (uint64_t)elapsed * PULSES_PER_REV;

  uint64_t leftValue =
    ((uint64_t)leftCount * 60000ULL)
    / denominator;

  uint64_t rightValue =
    ((uint64_t)rightCount * 60000ULL)
    / denominator;

  if (leftValue > 65535) {
    leftValue = 65535;
  }

  if (rightValue > 65535) {
    rightValue = 65535;
  }

  leftRPM = (uint16_t)leftValue;
  rightRPM = (uint16_t)rightValue;
}

// ==============================================
// LAP SENSOR
// ==============================================

void updateLapSensor(uint32_t now) {

  // Assumes TCRT5000 D0 is active LOW
  bool lapDetected =
    (digitalRead(LAP_SENSOR) == LOW);

  if (
    lapDetected &&
    !previousLapState &&
    (now - lastLapTime >= 500)
  ) {

    lapCount++;
    lastLapTime = now;

    // Set lap-event flag for RF packet
    flags |= 0x04;

    Serial.print("LAP DETECTED! Count: ");
    Serial.println(lapCount);
  }

  previousLapState = lapDetected;
}

// ==============================================
// SENSOR UPDATE
// ==============================================

void updateSensors(uint32_t now) {

  // Check lap sensor every loop
  updateLapSensor(now);

  uint32_t elapsed = now - lastSensorTime;

  if (elapsed < SENSOR_INTERVAL_MS) {
    return;
  }

  updateWheelRPM(elapsed);

  if (mpuConnected) {
    // mpuSampleValid = readMPU();

    //   if (!mpuSampleValid) {
    //     mpuConnected = false;
    //     Serial.println("MPU6050 read failed");
    //   }
    // } else {
    //   mpuSampleValid = false;
    // }

    //   lastSensorTime = now;
    readMPU();
    }

// ==============================================
// CRC-16 CCITT-FALSE
// ==============================================

uint16_t calculateCRC(
  const uint8_t *data,
  int length
) {

  uint16_t crc = 0xFFFF;

  for (int i = 0; i < length; i++) {

    crc ^= (uint16_t)data[i] << 8;

    for (int j = 0; j < 8; j++) {

      if (crc & 0x8000) {

        crc = (uint16_t)(
          (crc << 1) ^ 0x1021
        );

      } else {

        crc = (uint16_t)(crc << 1);
      }
    }
  }

  return crc;
}

// ==============================================
// MANCHESTER ENCODING
//
// Bit 0: HIGH -> LOW
// Bit 1: LOW -> HIGH
// ==============================================

void sendRFBit(bool bitValue) {

  digitalWrite(
    RF_PIN,
    bitValue ? LOW : HIGH
  );

  delayMicroseconds(CHIP_US);

  digitalWrite(
    RF_PIN,
    bitValue ? HIGH : LOW
  );

  delayMicroseconds(CHIP_US);
}

void sendRFByte(uint8_t value) {

  for (int i = 7; i >= 0; i--) {

    sendRFBit(
      (value >> i) & 0x01
    );
  }
}

// ==============================================
// SEND 19-BYTE RF TELEMETRY PACKET
//
// Bytes 0-3:   Preamble
// Bytes 4-5:   Sync
// Byte 6:      Sequence number
// Bytes 7-10:  Timestamp
// Bytes 11-12: Left RPM
// Bytes 13-14: Right RPM
// Byte 15:     Throttle
// Byte 16:     Flags
// Bytes 17-18: CRC-16
// ==============================================


void putU16LE(uint8_t *packet, int offset, uint16_t value) {
  packet[offset] = (uint8_t)(value & 0xFF);
  packet[offset + 1] = (uint8_t)(value >> 8);
}

void putI16LE(uint8_t *packet, int offset, int16_t value) {
  putU16LE(packet, offset, (uint16_t)value);
}

void sendTelemetry() {

  // PitWall protocol v2: 33 bytes.
  // Same preamble and sync as v1.
  // Flags bit 7 marks protocol v2.
  // Flags bit 6 marks a valid MPU sample.
  // Flags bit 2 remains the lap event.

  uint8_t packet[33] = {0};

  packet[0] = 0x55;
  packet[1] = 0x55;
  packet[2] = 0x55;
  packet[3] = 0x55;
  packet[4] = 0xD3;
  packet[5] = 0x91;

  packet[6] = sequenceNumber++;

  uint32_t timestamp = millis();

  for (int i = 0; i < 4; i++) {
    packet[7 + i] = (uint8_t)(timestamp >> (8 * i));
  }

  putU16LE(packet, 11, leftRPM);
  putU16LE(packet, 13, rightRPM);

  packet[15] = throttle;

  uint8_t packetFlags = flags | 0x80;

  if (mpuSampleValid) {
    packetFlags |= 0x40;
  }

  packet[16] = packetFlags;

  if (mpuSampleValid) {
    putI16LE(packet, 17, accelXmg);
    putI16LE(packet, 19, accelYmg);
    putI16LE(packet, 21, accelZmg);

    putI16LE(packet, 23, gyroXdps10);
    putI16LE(packet, 25, gyroYdps10);
    putI16LE(packet, 27, gyroZdps10);
  }

  // Lap count transmitted as uint16.
  // It wraps after 65535 laps.
  putU16LE(packet, 29, (uint16_t)lapCount);

  // CRC-16/CCITT-FALSE over bytes 6..30.
  uint16_t crc = calculateCRC(&packet[6], 25);

  putU16LE(packet, 31, crc);

  for (int i = 0; i < 33; i++) {
    sendRFByte(packet[i]);
  }

  digitalWrite(RF_PIN, LOW);

  // Clear the lap event after transmission.
  flags &= (uint8_t)~0x04;
}

// ==============================================
// SERIAL MONITOR OUTPUT
// ==============================================

void printTelemetry() {

  Serial.println();
  Serial.println("==========================");
  Serial.println("PITWALL LIVE TELEMETRY");
  Serial.println("==========================");

  Serial.print("Left RPM: ");
  Serial.println(leftRPM);

  Serial.print("Right RPM: ");
  Serial.println(rightRPM);

  Serial.print("Throttle: ");
  Serial.print(throttle);
  Serial.println("%");

  Serial.print("Lap Count: ");
  Serial.println(lapCount);

  Serial.print("Acceleration X (mg): ");
  Serial.println(accelXmg);

  Serial.print("Acceleration Y (mg): ");
  Serial.println(accelYmg);

  Serial.print("Acceleration Z (mg): ");
  Serial.println(accelZmg);

  Serial.print("Gyro X (0.1 dps): ");
  Serial.println(gyroXdps10);

  Serial.print("Gyro Y (0.1 dps): ");
  Serial.println(gyroYdps10);

  Serial.print("Gyro Z (0.1 dps): ");
  Serial.println(gyroZdps10);

  Serial.print("Motor State: ");

  switch (currentDirection) {

    case FORWARD:
      Serial.println("FORWARD");
      break;

    case BACKWARD:
      Serial.println("BACKWARD");
      break;

    case TURN_LEFT:
      Serial.println("LEFT");
      break;

    case TURN_RIGHT:
      Serial.println("RIGHT");
      break;

    default:
      Serial.println("STOPPED");
      break;
  }

  Serial.println("RF PACKET TRANSMITTED");
}

// ==============================================
// SETUP
// ==============================================

void setup() {

  Serial.begin(115200);
  delay(300);

  Serial.println();
  Serial.println("==========================");
  Serial.println("PITWALL SYSTEM STARTING");
  Serial.println("==========================");

  // Motor outputs start LOW
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  stopMotors();

  Serial.println("Motors initialized");

  // Wheel sensors
  // GPIO34/35 are input-only
  pinMode(RIGHT_ENCODER, INPUT);
  pinMode(LEFT_ENCODER, INPUT);

  attachInterrupt(
    digitalPinToInterrupt(RIGHT_ENCODER),
    rightEncoderISR,
    RISING
  );

  attachInterrupt(
    digitalPinToInterrupt(LEFT_ENCODER),
    leftEncoderISR,
    RISING
  );

  Serial.println("Wheel sensors initialized");

  // Lap sensor
  pinMode(LAP_SENSOR, INPUT_PULLUP);

  Serial.println("Lap sensor initialized");

  // RF transmitter
  digitalWrite(RF_PIN, LOW);
  pinMode(RF_PIN, OUTPUT);

  Serial.println("RF transmitter initialized");

  // MPU6050
  Wire.begin(MPU_SDA, MPU_SCL);

  Wire.setClock(100000);

  mpuConnected = initializeMPU();

  if (mpuConnected) {
    Serial.println("MPU6050 CONNECTED");
  } else {
    Serial.println("MPU6050 NOT FOUND");
  }

  // Bluetooth
  Dabble.begin("PitWall_Car");

  Serial.println("Bluetooth initialized");
  Serial.println("Device name: PitWall_Car");

  lastSensorTime = millis();
  lastRFTime = millis();

  stopMotors();

  Serial.println("==========================");
  Serial.println("PITWALL READY!");
  Serial.println("==========================");
}

// ==============================================
// MAIN LOOP
// ==============================================

void loop() {

  uint32_t now = millis();

  // Read Dabble GamePad
  updateBluetoothControl();

  // Update wheel RPM, MPU6050 and laps
  updateSensors(now);

  // Send RF telemetry every 500 ms
  if (now - lastRFTime >= RF_INTERVAL_MS) {

    sendTelemetry();

    printTelemetry();

    lastRFTime = millis();
  }
}
