
import numpy as np
import matplotlib.pyplot as plt

# HackRF recording
FILE = r"C:\Users\ce\wheel_speed.iq"

SAMPLE_RATE = 2_000_000

# Read HackRF interleaved signed 8-bit IQ samples
raw = np.fromfile(FILE, dtype=np.int8)

if len(raw) % 2:
    raw = raw[:-1]

iq = (
    raw[0::2].astype(np.float32)
    + 1j * raw[1::2].astype(np.float32)
) / 128.0

print("Samples:", len(iq))
print("Duration:", len(iq) / SAMPLE_RATE, "seconds")

# Calculate average signal power in 1 ms windows
samples_per_window = 2000

n = len(iq) // samples_per_window

power = np.mean(
    np.abs(iq[:n * samples_per_window].reshape(
        n, samples_per_window
    )) ** 2,
    axis=1
)

power_db = 10 * np.log10(power + 1e-12)

time = (
    np.arange(n) * samples_per_window
    / SAMPLE_RATE
)

plt.figure(figsize=(12, 5))
plt.plot(time, power_db)

plt.xlabel("Time (seconds)")
plt.ylabel("Power (dBFS)")
plt.title("PitWall Wheel Speed RF Reception")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Minimum power:", np.min(power_db), "dBFS")
print("Maximum power:", np.max(power_db), "dBFS")
