# ==========================================
# PITWALL - 100 PACKET TELEMETRY SESSION
# ==========================================
import time
import csv
from pathlib import Path

import numpy as np

from packet import make_packet
from manchester import manchester_encode, manchester_decode
from packet_decoder import decode_packet


# ==========================================
# 1. SIMULATION SETTINGS
# ==========================================

from itertools import count

NUM_PACKETS = None

CHIP_RATE = 4000
SAMPLE_RATE = 200000
SAMPLES_PER_CHIP = SAMPLE_RATE // CHIP_RATE

NOISE_STD = 0.15
THRESHOLD = 0.5

rng = np.random.default_rng(42)

csv_path = (
    Path(__file__).resolve().parent.parent
    / "pitwall_telemetry.csv"
)

decoded_rows = []
rejected_packets = 0
total_chip_errors = 0


# ==========================================
# 2. GENERATE AND TRANSMIT PACKETS
# ==========================================

for i in count():

    # Simulated vehicle movement
    timestamp = i * 200

    throttle = int(
        np.clip(
            55 + 30 * np.sin(i / 12),
            0,
            100
        )
    )

    left_rpm = int(
        150 + 65 * np.sin(i / 10)
    )

    right_rpm = int(
        145 + 60 * np.sin(i / 10 + 0.25)
    )

    # Create telemetry packet with CRC
    original_packet = make_packet(
        sequence=i%256,
        timestamp=timestamp,
        left_rpm=left_rpm,
        right_rpm=right_rpm,
        throttle=throttle,
        flags=0b00000100
    )

    # Manchester encoding
    chips = np.array(
        manchester_encode(original_packet),
        dtype=np.uint8
    )

    # OOK modulation
    transmitted_envelope = np.repeat(
        chips,
        SAMPLES_PER_CHIP
    )

    transmitted_iq = transmitted_envelope.astype(
        complex
    )

    # Add complex Gaussian noise
    noise = NOISE_STD * (
        rng.standard_normal(len(transmitted_iq))
        + 1j * rng.standard_normal(len(transmitted_iq))
    )

    received_iq = transmitted_iq + noise

    # Envelope detection
    received_envelope = np.abs(received_iq)

    # Recover Manchester chips
    chip_measurements = received_envelope.reshape(
        -1,
        SAMPLES_PER_CHIP
    ).mean(axis=1)

    recovered_chips = (
        chip_measurements > THRESHOLD
    ).astype(np.uint8)

    total_chip_errors += np.count_nonzero(
        recovered_chips != chips
    )

    # Manchester decoding + CRC verification
    try:
        decoded_packet = manchester_decode(
            recovered_chips.tolist()
        )

        telemetry = decode_packet(decoded_packet)

        decoded_rows.append([
            telemetry["timestamp_ms"],
            telemetry["sequence"],
            telemetry["left_rpm"],
            telemetry["right_rpm"],
            telemetry["throttle_percent"],
            True
        ])

    except ValueError:
        rejected_packets += 1
    
    # Update dashboard CSV after each packet
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "timestamp_ms",
            "seq",
            "left_rpm",
            "right_rpm",
            "throttle_pct",
            "crc_ok"
        ])

        writer.writerows(decoded_rows)
    time.sleep(0.2)
