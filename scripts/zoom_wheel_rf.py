
import numpy as np
import matplotlib.pyplot as plt

FILE = r"C:\Users\ce\wheel_speed.iq"
FS = 2_000_000

# First detected burst is around 0.6 seconds.
# Inspect 20 ms of that burst.
START_TIME = 0.62
DURATION = 0.020

start_sample = int(START_TIME * FS)
count = int(DURATION * FS)

# HackRF stores interleaved signed 8-bit IQ
with open(FILE, "rb") as f:
    f.seek(start_sample * 2)
    raw = np.fromfile(
        f, dtype=np.int8, count=count * 2
    )

if len(raw) < count * 2:
    raise RuntimeError("Not enough samples in file")

i = raw[0::2].astype(np.float32)
q = raw[1::2].astype(np.float32)

power = i**2 + q**2

# Average every 50 samples = 25 microseconds
WINDOW = 50
n = len(power) // WINDOW

smooth = power[:n * WINDOW].reshape(
    n, WINDOW
).mean(axis=1)

time_ms = (
    START_TIME
    + np.arange(n) * WINDOW / FS
) * 1000

power_db = 10 * np.log10(
    smooth / (128.0**2) + 1e-12
)

plt.figure(figsize=(13, 5))
plt.plot(time_ms, power_db)

plt.title("PitWall - Manchester RF Signal")
plt.xlabel("Time (ms)")
plt.ylabel("Received Power (dBFS)")
plt.grid(True)
plt.tight_layout()
plt.show()
