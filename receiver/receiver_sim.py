# ==========================================
# PITWALL - NOISY OOK RECEIVER SIMULATION
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

from packet import make_packet
from manchester import manchester_encode, manchester_decode
from packet_decoder import decode_packet

# ==========================================
# 1. GENERATE ORIGINAL TELEMETRY PACKET
# ==========================================

original_packet = make_packet(
    sequence=2,
    timestamp=2000,
    left_rpm=150,
    right_rpm=145,
    throttle=80,
    flags=0b00000100
)

print("\n========== PITWALL RF SIMULATION ==========")
print("\nOriginal packet:")
print(original_packet.hex(" ").upper())


# ==========================================
# 2. MANCHESTER ENCODING
# ==========================================

chips = np.array(
    manchester_encode(original_packet),
    dtype=np.uint8
)

chip_rate = 4000
sample_rate = 200000
samples_per_chip = sample_rate // chip_rate

print("\nManchester chips:", len(chips))


# ==========================================
# 3. GENERATE TRANSMITTED OOK SIGNAL
# ==========================================

# Expand each chip into 50 samples
transmitted_envelope = np.repeat(
    chips,
    samples_per_chip
)

# Simulate complex baseband IQ samples.
# Carrier is centered at 0 Hz in this representation.
transmitted_iq = transmitted_envelope.astype(complex)


# ==========================================
# 4. ADD CHANNEL NOISE
# ==========================================

rng = np.random.default_rng(42)

noise_std = 0.15

noise = noise_std * (
    rng.standard_normal(len(transmitted_iq))
    + 1j * rng.standard_normal(len(transmitted_iq))
)

received_iq = transmitted_iq + noise


# ==========================================
# 5. OOK ENVELOPE DETECTION
# ==========================================

# Magnitude of complex IQ samples
received_envelope = np.abs(received_iq)


# ==========================================
# 6. RECOVER MANCHESTER CHIPS
# ==========================================

# Average samples within each chip period.
# For now, we assume perfect chip timing.

chip_measurements = received_envelope.reshape(
    -1, samples_per_chip
).mean(axis=1)

threshold = 0.5

recovered_chips = (
    chip_measurements > threshold
).astype(np.uint8)

chip_errors = np.count_nonzero(
    recovered_chips != chips
)

print("\nChip errors:", chip_errors)


# ==========================================
# 7. MANCHESTER DECODING + CRC VERIFICATION
# ==========================================

try:
    # Recover bytes from Manchester chips
    decoded_packet = manchester_decode(
        recovered_chips.tolist()
    )

    print("\nManchester decoding: SUCCESS")

    print("\nRecovered packet:")
    print(decoded_packet.hex(" ").upper())

    # Verify packet structure and CRC
    telemetry = decode_packet(decoded_packet)

    import csv
    from pathlib import Path

    csv_path = Path(__file__).resolve().parent.parent / "pitwall_telemetry.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "timestamp_ms", "seq", "left_rpm",
            "right_rpm", "throttle_pct", "crc_ok"
        ])

        writer.writerow([
            telemetry["timestamp_ms"],
            telemetry["sequence"],
            telemetry["left_rpm"],
            telemetry["right_rpm"],
            telemetry["throttle_percent"],
            True
        ])

        print(f"Dashboard CSV saved: {csv_path}")
    print("\nCRC verification: VALID")

    # Display decoded telemetry
    print("\n========== PITWALL TELEMETRY ==========")

    print(f"Packet sequence: {telemetry['sequence']}")
    print(f"Timestamp: {telemetry['timestamp_ms']} ms")

    print(f"Left wheel RPM: {telemetry['left_rpm']}")
    print(f"Right wheel RPM: {telemetry['right_rpm']}")

    print(f"Throttle: {telemetry['throttle_percent']}%")

    print(f"Brake: {telemetry['brake']}")
    print(f"Lap detected: {telemetry['lap_marker']}")
    print(f"Encoder valid: {telemetry['encoder_valid']}")

    print("\nPACKET ACCEPTED!")

except ValueError as error:

    decoded_packet = None
    telemetry = None

    print("\nPACKET REJECTED!")
    print("Reason:", error)

# ==========================================
# 8. PLOT TRANSMITTED AND RECEIVED SIGNALS
# ==========================================

chips_to_show = 16
samples_to_show = chips_to_show * samples_per_chip

t_ms = (
    np.arange(samples_to_show) / sample_rate
) * 1000

plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
plt.step(
    t_ms,
    transmitted_envelope[:samples_to_show],
    where="post"
)
plt.title("PitWall - Transmitted OOK Envelope")
plt.ylabel("Amplitude")
plt.ylim(-0.2, 1.2)
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(
    t_ms,
    received_envelope[:samples_to_show]
)
plt.title("PitWall - Received OOK Envelope with Noise")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.step(
    np.arange(chips_to_show) / chip_rate * 1000,
    recovered_chips[:chips_to_show],
    where="post",
    label="Recovered chips"
)
plt.step(
    np.arange(chips_to_show) / chip_rate * 1000,
    chips[:chips_to_show],
    where="post",
    linestyle="--",
    label="Original chips"
)
plt.title("PitWall - Manchester Chip Recovery")
plt.xlabel("Time (ms)")
plt.ylabel("Chip value")
plt.ylim(-0.2, 1.2)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()