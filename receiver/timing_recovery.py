# =====================================================
# PITWALL - CHIP TIMING RECOVERY
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

from packet import make_packet
from manchester import manchester_encode, manchester_decode
from packet_decoder import decode_packet


# =====================================================
# 1. COMMUNICATION PARAMETERS
# =====================================================

CHIP_RATE = 4000
SAMPLE_RATE = 200000

SAMPLES_PER_CHIP = SAMPLE_RATE // CHIP_RATE

PACKET_LENGTH = 19
CHIPS_PER_PACKET = PACKET_LENGTH * 8 * 2

NOISE_STD = 0.15

# Deliberately introduce an unknown timing offset
TIMING_OFFSET = 17

rng = np.random.default_rng(42)


# =====================================================
# 2. GENERATE THREE TELEMETRY PACKETS
# =====================================================

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


# =====================================================
# 3. BUILD CONTINUOUS CHIP STREAM
# =====================================================

chip_stream = [0] * 120

for packet in packets:
    chip_stream.extend(manchester_encode(packet))
    chip_stream.extend([0] * 200)

chip_stream.extend([0] * 120)

chip_stream = np.array(chip_stream, dtype=np.uint8)


# =====================================================
# 4. GENERATE OOK IQ SIGNAL
# =====================================================

transmitted_envelope = np.repeat(
    chip_stream,
    SAMPLES_PER_CHIP
)

transmitted_iq = transmitted_envelope.astype(complex)


# =====================================================
# 5. INTRODUCE UNKNOWN TIMING OFFSET
# =====================================================

# Add a partial-chip interval before the original signal.
# The receiver is NOT given TIMING_OFFSET.

received_signal = np.concatenate([
    np.zeros(TIMING_OFFSET, dtype=complex),
    transmitted_iq
])

noise = NOISE_STD * (
    rng.standard_normal(len(received_signal))
    + 1j * rng.standard_normal(len(received_signal))
)

received_iq = received_signal + noise

received_envelope = np.abs(received_iq)


# =====================================================
# 6. CREATE SYNCHRONIZATION PATTERN
# =====================================================

PREAMBLE = bytes([0x55] * 4)
SYNC = bytes([0xD3, 0x91])

sync_pattern = np.array(
    manchester_encode(PREAMBLE + SYNC),
    dtype=np.uint8
)

SYNC_LENGTH = len(sync_pattern)


# =====================================================
# 7. SEARCH ALL POSSIBLE TIMING PHASES
# =====================================================

best_score = -1
best_quality = -1

best_phase = None
best_chip_stream = None
best_sync_position = None

for phase in range(SAMPLES_PER_CHIP):

    # Ignore samples before the candidate phase
    shifted_envelope = received_envelope[phase:]

    # Keep only complete chip intervals
    usable_length = (
        len(shifted_envelope) // SAMPLES_PER_CHIP
    ) * SAMPLES_PER_CHIP

    shifted_envelope = shifted_envelope[:usable_length]

    # Average samples within each candidate chip
    chip_measurements = shifted_envelope.reshape(
        -1, SAMPLES_PER_CHIP
    ).mean(axis=1)

    # Convert measurements into binary chips
    candidate_chips = (
        chip_measurements > 0.5
    ).astype(np.uint8)

    # Search for synchronization pattern
    for i in range(
        len(candidate_chips) - SYNC_LENGTH + 1
    ):

        candidate = candidate_chips[i:i + SYNC_LENGTH]

        # Number of matching chips
        score = np.count_nonzero(
            candidate == sync_pattern
        )

        # Measure separation between expected ON and OFF chips
        # in the synchronization pattern.

        sync_measurements = chip_measurements[
            i:i + SYNC_LENGTH
        ]

        on_values = sync_measurements[sync_pattern == 1]
        off_values = sync_measurements[sync_pattern == 0]

        quality = on_values.mean() - off_values.mean()

        # Prioritize synchronization score.
        # If scores tie, choose the phase with better
        # ON/OFF amplitude separation.

        if (
            score > best_score
            or (
                score == best_score
                and quality > best_quality
            )
        ):

            best_score = score
            best_quality = quality

            best_phase = phase
            best_chip_stream = candidate_chips
            best_sync_position = i


print("\n========== PITWALL TIMING RECOVERY ==========")

print("Actual timing offset:", TIMING_OFFSET)
print("Recovered timing phase:", best_phase)

print("Synchronization score:", best_score, "/", SYNC_LENGTH)


# =====================================================
# 8. DETECT PACKETS USING RECOVERED PHASE
# =====================================================

detected_packets = []

i = 0

while i <= len(best_chip_stream) - CHIPS_PER_PACKET:

    candidate = best_chip_stream[
        i:i + SYNC_LENGTH
    ]

    score = np.count_nonzero(
        candidate == sync_pattern
    )

    # Require at least 95% synchronization match
    if score >= int(0.95 * SYNC_LENGTH):

        packet_chips = best_chip_stream[
            i:i + CHIPS_PER_PACKET
        ]

        try:
            decoded = manchester_decode(
                packet_chips.tolist()
            )

            telemetry = decode_packet(decoded)

            detected_packets.append(telemetry)

            print("\nPacket detected at chip:", i)
            print("Sequence:", telemetry["sequence"])
            print("Left RPM:", telemetry["left_rpm"])
            print("Right RPM:", telemetry["right_rpm"])
            print("Throttle:", telemetry["throttle_percent"], "%")
            print("CRC: VALID")

            i += CHIPS_PER_PACKET
            continue

        except ValueError:
            pass

    i += 1


# =====================================================
# 9. RESULTS
# =====================================================

print("\n========== RESULTS ==========")

print("Packets transmitted:", len(packets))
print("Packets recovered:", len(detected_packets))

print(
    "Recovered sequences:",
    [p["sequence"] for p in detected_packets]
)


# =====================================================
# 10. VISUALIZE TIMING RECOVERY
# =====================================================

# Plot the first 16 chips of Packet 1.

packet_start_sample = (
    TIMING_OFFSET
    + 120 * SAMPLES_PER_CHIP
)

samples_to_show = 16 * SAMPLES_PER_CHIP

start = packet_start_sample
end = start + samples_to_show

t_ms = (
    np.arange(start, end) / SAMPLE_RATE
) * 1000

plt.figure(figsize=(12, 7))

# Original transmitted signal, shifted by the
# actual timing offset for comparison.
plt.subplot(2, 1, 1)

plt.plot(
    t_ms,
    np.abs(received_signal[start:end]),
    label="Transmitted OOK envelope"
)

plt.title("PitWall - Original OOK Signal")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()


# Received signal after noise
plt.subplot(2, 1, 2)

plt.plot(
    t_ms,
    received_envelope[start:end],
    label="Received OOK envelope",
    color="orange"
)

plt.title("PitWall - Received OOK Signal with Noise")
plt.xlabel("Time (ms)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()