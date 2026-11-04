
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

filename = "gqrx_20261102_095117_433702700_2000000_fc.raw"

fs = 2_000_000
center = 433_702_700

# Read 100 ms from inside the transmitted burst
with open(filename, "rb") as f:
    f.seek(int(0.11 * fs) * 8)
    iq = np.fromfile(f, dtype=np.complex64, count=200000)

window = np.hanning(len(iq))
spectrum = np.fft.fftshift(np.fft.fft(iq * window))
freq = np.fft.fftshift(np.fft.fftfreq(len(iq), 1/fs))

power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
frequency_mhz = (center + freq) / 1e6

plt.figure(figsize=(12, 4))
plt.plot(frequency_mhz, power_db)
plt.xlim(433.85, 433.98)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Relative magnitude (dB)")
plt.title("PitWall - Received RF Spectrum")
plt.grid()
plt.tight_layout()
plt.savefig("pitwall_spectrum.png", dpi=150)

print("Saved: pitwall_spectrum.png")
