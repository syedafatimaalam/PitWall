# ============================================
# PITWALL - OOK MODULATION SIMULATION
# ============================================

import numpy as np
import matplotlib.pyplot as plt

from packet import make_packet
from manchester import manchester_encode


# ============================================
# 1. GENERATE PITWALL TELEMETRY PACKET
# ============================================

packet = make_packet(
    sequence=2,
    timestamp=2000,
    left_rpm=150,
    right_rpm=145,
    throttle=80,
    flags=0b00000100
)

print("Original packet:")
print(packet.hex(" ").upper())


# ============================================
# 2. MANCHESTER ENCODING
# ============================================

chips = manchester_encode(packet)

print("\nPacket length:", len(packet), "bytes")
print("Manchester chips:", len(chips))


# ============================================
# 3. DEFINE COMMUNICATION PARAMETERS
# ============================================

chip_rate = 4000       # Manchester chips per second
carrier_freq = 10000   # 10 kHz illustrative carrier
sample_rate = 200000   # 200 kHz simulation sampling rate

samples_per_chip = sample_rate // chip_rate

chip_duration = 1 / chip_rate

print("\nChip duration:", chip_duration * 1e6, "microseconds")


# ============================================
# 4. GENERATE DIGITAL WAVEFORMS
# ============================================

# Original packet bits (MSB first)
original_bits = np.unpackbits(
    np.frombuffer(packet, dtype=np.uint8)
)

# Expand each bit/chip into samples
original_waveform = np.repeat(
    original_bits,
    samples_per_chip * 2
)

manchester_waveform = np.repeat(
    chips,
    samples_per_chip
)

# Time axis for Manchester waveform
t = np.arange(len(manchester_waveform)) / sample_rate


# ============================================
# 5. OOK MODULATION
# ============================================

carrier = np.cos(2 * np.pi * carrier_freq * t)

ook_signal = manchester_waveform * carrier


# ============================================
# 6. PLOT THE SIGNALS
# ============================================

# Show first 8 original bits = 16 Manchester chips
bits_to_show = 8

samples_to_show = bits_to_show * 2 * samples_per_chip

time_ms = t[:samples_to_show] * 1000


plt.figure(figsize=(12, 9))


# Original binary data
plt.subplot(3, 1, 1)

plt.step(
    time_ms,
    original_waveform[:samples_to_show],
    where="post"
)

plt.title("PitWall - Original Binary Data")
plt.ylabel("Bit Value")
plt.ylim(-0.2, 1.2)
plt.grid(True)


# Manchester waveform
plt.subplot(3, 1, 2)

plt.step(
    time_ms,
    manchester_waveform[:samples_to_show],
    where="post"
)

plt.title("PitWall - Manchester Encoded Signal")
plt.ylabel("Chip Value")
plt.ylim(-0.2, 1.2)
plt.grid(True)


# OOK waveform
plt.subplot(3, 1, 3)

plt.plot(
    time_ms,
    ook_signal[:samples_to_show]
)

plt.title("PitWall - OOK Modulated Signal")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)


plt.tight_layout()
plt.show()