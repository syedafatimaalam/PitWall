
import numpy as np
from pathlib import Path
import os

FILE = Path(os.environ.get(
    "PITWALL_IQ_FILE",
    r"D:\PitWall Project\dataset\raw\pitwall_capture_new.iq"
))

SAMPLE_RATE = 2_000_000
BLOCK_SAMPLES = 2000  # 1 ms

if not FILE.exists():
    raise SystemExit(f"Recording not found: {FILE}")

print("Reading HackRF recording...")

raw = np.memmap(FILE, dtype=np.int8, mode="r")
total_samples = len(raw) // 2
num_blocks = total_samples // BLOCK_SAMPLES

levels = np.empty(num_blocks, dtype=np.float32)

for i in range(num_blocks):
    start = i * BLOCK_SAMPLES * 2
    end = start + BLOCK_SAMPLES * 2

    iq = np.asarray(
        raw[start:end], dtype=np.float32
    ).reshape(-1, 2)

    levels[i] = np.hypot(
        iq[:, 0], iq[:, 1]
    ).mean()

# Estimate the noise floor from quiet intervals.
noise_floor = np.percentile(levels, 20)

# Smooth the 1 ms signal levels over 10 ms.
window_ms = 10
kernel = np.ones(window_ms) / window_ms

smoothed = np.convolve(
    levels, kernel, mode="same"
)

# Detect sustained RF activity above noise.
threshold = noise_floor + 5.0
active = smoothed > threshold

# Locate activity intervals.
edges = np.diff(
    np.r_[False, active, False].astype(np.int8)
)

starts = np.where(edges == 1)[0]
ends = np.where(edges == -1)[0]

bursts = []

for start, end in zip(starts, ends):
    duration_ms = end - start

    # Expected PitWall packet duration ~152 ms.
    if 100 <= duration_ms <= 340:
        bursts.append((
            start / 1000.0,
            end / 1000.0
        ))

print("\n========== PITWALL RF BURSTS ==========")
print("Noise floor:", round(float(noise_floor), 2))
print("Threshold:", round(float(threshold), 2))
print("Bursts detected:", len(bursts))

for i, (start, end) in enumerate(bursts, 1):
    print(
        f"Burst {i:02d}: "
        f"{start:.3f}s - {end:.3f}s "
        f"({(end-start)*1000:.0f} ms)"
    )

print("\nAnalysis complete!")
