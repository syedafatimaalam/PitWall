import numpy as np
import matplotlib.pyplot as plt

file = "gqrx_20261102_095117_433702700_2000000_fc.raw"
fs = 2_000_000
center = 433_702_700
target = 433_920_000

x = np.fromfile(file, dtype=np.complex64)

# Frequency-shift the target signal to baseband
n = np.arange(len(x))
x = x * np.exp(-2j * np.pi * (target-center)/fs * n)

# Smooth the amplitude to reveal OOK bursts
power = np.abs(x[:2_000_000])**2
smooth = np.convolve(power, np.ones(200)/200, mode="same")

plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(smooth))[::200]/fs, smooth[::200])
plt.xlabel("Time (seconds)")
plt.ylabel("Smoothed signal power")
plt.title("PitWall RF Recording - First Second")
plt.grid()
plt.tight_layout()
plt.savefig("pitwall_rf_signal.png", dpi=150)
