# PitWall

RF-Based RC Car Telemetry and Machine Learning for Driver Performance Analysis

## Overview

PitWall is an Electrical Engineering project that combines
embedded systems, digital communications, RF telemetry,
and vehicle performance analysis.

## Planned Hardware

- ESP32 microcontroller
- Wheel speed encoders
- MPU6050 motion sensor
- TCRT5000 lap detection sensor
- 433 MHz OOK/ASK transmitter
- HackRF One (receive-only)

## Communication Protocol

- Modulation: OOK/ASK
- Line coding: Manchester
- Chip rate: 4,000 chips/s
- Packet length: 19 bytes
- Error detection: CRC-16/CCITT-FALSE

## Software Progress

- [x] Packet generation
- [x] Manchester encoding and decoding
- [x] OOK simulation
- [x] Noisy receiver simulation
- [x] CRC verification
- [x] Packet synchronization
- [x] Timing-phase recovery simulation
- [x] Packet Reception Ratio monitoring
- [ ] Dashboard integration
- [ ] Real HackRF reception
- [ ] Microcontroller firmware
- [ ] Machine learning analysis

## Running the Simulations

Install NumPy and Matplotlib in a Python virtual environment.

Run individual scripts from the receiver directory.

Example:

python receiver/timing_recovery.py

## Project Team

Syeda Fatima Alam  
Mohsin Mujtaba

Habib University