import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


filename = "gqrx_20261102_095117_433702700_2000000_fc.raw"

fs = 2_000_000
offset = 433_916_277.27 - 433_702_700

# Read first second
iq = np.fromfile(filename, dtype=np.complex64, count=300_000)

# Shift 433.92 MHz signal to 0 Hz
t = np.arange(len(iq), dtype=np.float32) / fs

phase = (-2 * np.pi * offset * t).astype(np.float32)
baseband = iq * np.exp(1j * phase).astype(np.complex64)


# Low-pass filter using NumPy FFT (no SciPy required)
spectrum = np.fft.fft(baseband)
frequencies = np.fft.fftfreq(len(baseband), d=1/fs)

spectrum[np.abs(frequencies) > 50000] = 0
filtered = np.fft.ifft(spectrum)

# Envelope detection and smoothing
envelope = np.abs(filtered)
smooth = np.convolve(envelope, np.ones(200)/200, mode="same")

# Zoom into 10 milliseconds of the RF packet
step = 10
start = int(0.100 * fs)
count = int(0.110 * fs)

plt.figure(figsize=(12,4))
plt.plot(t[start:count:step], smooth[start:count:step])
plt.xlabel("Time (seconds)")
plt.ylabel("Received amplitude")
plt.title("PitWall - Filtered OOK Envelope")
plt.grid()
plt.tight_layout()
plt.savefig("pitwall_ook_envelope.png", dpi=150)
plt.close()

print("Saved: pitwall_ook_envelope.png")


# Convert received amplitude into digital levels
threshold = 0.003
digital = (smooth > threshold).astype(int)

# Plot a 10 ms section
start = int(0.090 * fs)
end = int(0.120 * fs)
step = 20

plt.figure(figsize=(12, 4))
plt.step(
    t[start:end:step],
    digital[start:end:step],
    where="post"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Digital level")
plt.title("PitWall - Recovered OOK Digital Signal")
plt.yticks([0, 1])
plt.grid()
plt.tight_layout()
plt.savefig("pitwall_digital_bits.png", dpi=150)
plt.close()

print("Saved: pitwall_digital_bits.png")


# Measure HIGH pulse widths and spacing
edges = np.diff(digital.astype(np.int8))

rising = np.where(edges == 1)[0] + 1
falling = np.where(edges == -1)[0] + 1

if len(rising) > 0 and len(falling) > 0:
    widths = []
    for r in rising:
        next_fall = falling[falling > r]
        if len(next_fall):
            widths.append((next_fall[0] - r) / fs * 1000)

    print("First 10 HIGH pulse widths (ms):",
          np.round(widths[:10], 3))

    print("First 10 rising-edge intervals (ms):",
          np.round(np.diff(rising[:11]) / fs * 1000, 3))


# Compare raw and filtered amplitude over 10 ms
start = int(0.100 * fs)
end = int(0.110 * fs)
step = 20

plt.figure(figsize=(12, 4))

plt.plot(
    t[start:end:step],
    np.abs(iq[start:end:step]),
    label="Raw IQ amplitude",
    alpha=0.6
)

plt.plot(
    t[start:end:step],
    smooth[start:end:step],
    label="Filtered envelope",
    linewidth=2
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("PitWall - Raw vs Filtered RF Signal")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("pitwall_comparison.png", dpi=150)
plt.close()

print("Saved: pitwall_comparison.png")


# Examine envelope amplitudes inside and outside the packet
noise = smooth[int(0.02*fs):int(0.06*fs)]
signal = smooth[int(0.10*fs):int(0.12*fs)]

print("Noise median:", np.median(noise))
print("Signal minimum:", np.min(signal))
print("Signal median:", np.median(signal))
print("Signal maximum:", np.max(signal))
print("Signal percentiles:",
      np.percentile(signal, [10, 25, 50, 75, 90, 99]))
