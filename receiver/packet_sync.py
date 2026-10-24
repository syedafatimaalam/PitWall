# ==================================================
# PITWALL - AUTOMATIC PACKET DETECTION
# ==================================================

import numpy as np

from packet import make_packet
from manchester import manchester_encode, manchester_decode
from packet_decoder import decode_packet


# ==================================================
# 1. COMMUNICATION PARAMETERS
# ==================================================

CHIP_RATE = 4000
SAMPLE_RATE = 200000

SAMPLES_PER_CHIP = SAMPLE_RATE // CHIP_RATE

PACKET_LENGTH = 19
CHIPS_PER_PACKET = PACKET_LENGTH * 8 * 2

NOISE_STD = 0.15

rng = np.random.default_rng(42)


# ==================================================
# 2. GENERATE THREE TELEMETRY PACKETS
# ==================================================

packets = [
    make_packet(
        sequence=1,
        timestamp=1000,
        left_rpm=100,
        right_rpm=98,
        throttle=40,
        flags=0b00000100
    ),

    make_packet(
        sequence=2,
        timestamp=1200,
        left_rpm=130,
        right_rpm=128,
        throttle=60,
        flags=0b00000100
    ),

    make_packet(
        sequence=3,
        timestamp=1400,
        left_rpm=160,
        right_rpm=158,
        throttle=80,
        flags=0b00000110
    )
]


# ==================================================
# 3. CONSTRUCT CONTINUOUS CHIP STREAM
# ==================================================

# Represent silence with zero chips.
# This is a simplified model of the RF channel.

silence_before = [0] * 120
silence_between = [0] * 200
silence_after = [0] * 120

chip_stream = silence_before.copy()

for packet in packets:

    encoded = manchester_encode(packet)

    chip_stream.extend(encoded)

    chip_stream.extend(silence_between)

chip_stream.extend(silence_after)

chip_stream = np.array(chip_stream, dtype=np.uint8)

print("\n========== PITWALL PACKET SYNCHRONIZATION ==========")

print("\nTotal chips in stream:", len(chip_stream))


# ==================================================
# 4. SIMULATE NOISY IQ SIGNAL
# ==================================================

transmitted_envelope = np.repeat(
    chip_stream,
    SAMPLES_PER_CHIP
)

transmitted_iq = transmitted_envelope.astype(complex)

noise = NOISE_STD * (
    rng.standard_normal(len(transmitted_iq))
    + 1j * rng.standard_normal(len(transmitted_iq))
)

received_iq = transmitted_iq + noise


# ==================================================
# 5. OOK ENVELOPE DETECTION
# ==================================================

received_envelope = np.abs(received_iq)

chip_measurements = received_envelope.reshape(
    -1,
    SAMPLES_PER_CHIP
).mean(axis=1)

threshold = 0.5

recovered_chips = (
    chip_measurements > threshold
).astype(np.uint8)


# ==================================================
# 6. GENERATE PREAMBLE + SYNC SEARCH PATTERN
# ==================================================

PREAMBLE = bytes([0x55] * 4)
SYNC = bytes([0xD3, 0x91])

search_pattern = np.array(
    manchester_encode(PREAMBLE + SYNC),
    dtype=np.uint8
)

print("Search pattern length:", len(search_pattern), "chips")


# ==================================================
# 7. SEARCH FOR PACKETS
# ==================================================

detected_packets = []

i = 0

while i <= len(recovered_chips) - CHIPS_PER_PACKET:

    candidate_pattern = recovered_chips[
        i:i + len(search_pattern)
    ]

    # Search for exact Manchester preamble + sync
    if np.array_equal(candidate_pattern, search_pattern):

        print(f"\nPossible packet detected at chip {i}")

        packet_chips = recovered_chips[
            i:i + CHIPS_PER_PACKET
        ]

        try:
            decoded = manchester_decode(
                packet_chips.tolist()
            )

            telemetry = decode_packet(decoded)

            detected_packets.append(telemetry)

            print("CRC: VALID")
            print("Sequence:", telemetry["sequence"])
            print("Timestamp:", telemetry["timestamp_ms"], "ms")
            print("Left RPM:", telemetry["left_rpm"])
            print("Right RPM:", telemetry["right_rpm"])
            print("Throttle:", telemetry["throttle_percent"], "%")
            print("Lap marker:", telemetry["lap_marker"])

            # Skip past this packet
            i += CHIPS_PER_PACKET

            continue

        except ValueError as error:

            print("Packet rejected:", error)

    i += 1


# ==================================================
# 8. FINAL RESULTS
# ==================================================

print("\n========== RESULTS ==========")

print("Packets transmitted:", len(packets))
print("Valid packets detected:", len(detected_packets))

if detected_packets:

    sequences = [
        packet["sequence"]
        for packet in detected_packets
    ]

    print("Recovered sequences:", sequences)