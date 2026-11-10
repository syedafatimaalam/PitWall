
import numpy as np

filename = "gqrx_20261102_123003_433858100_2000000_fc.raw"

fs = 2_000_000
center = 433_858_100
target = 433_916_277

# Read 3 seconds using memory mapping
iq = np.memmap(filename, dtype=np.complex64, mode="r")
iq = iq[:3 * fs]

# Process in 1 ms blocks to save memory
block = 2000
nblocks = len(iq) // block

power = np.empty(nblocks)

# Analyze energy near our transmitter frequency
for i in range(nblocks):
    samples = iq[i*block:(i+1)*block]
    t = np.arange(block) / fs

    shifted = samples * np.exp(
        -2j * np.pi * (target-center) * t
    )

    power[i] = np.abs(np.mean(shifted))**2

# Estimate signal ON/OFF
baseline = np.median(power)
threshold = max(baseline * 10, np.max(power) * 0.1)

active = power > threshold

# Identify contiguous active periods
edges = np.diff(
    np.r_[False, active, False].astype(int)
)

starts = np.where(edges == 1)[0]
ends = np.where(edges == -1)[0]

print("Noise baseline:", baseline)
print("Detection threshold:", threshold)

print("\nDetected RF bursts:")
for s, e in zip(starts, ends):
    duration = e - s
    if duration >= 5:
        print(
            f"Start: {s} ms, "
            f"Duration: {duration} ms"
        )
