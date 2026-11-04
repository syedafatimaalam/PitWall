import numpy as np

file = "gqrx_20261102_095117_433702700_2000000_fc.raw"
fs = 2_000_000
center = 433_702_700

# Analyze a section containing the transmission
with open(file, "rb") as f:
    f.seek(int(0.10 * fs) * 8)
    x = np.fromfile(f, dtype=np.complex64, count=262144)

# Find spectral peaks near the expected RF frequency
window = np.hanning(len(x))
spectrum = np.abs(np.fft.fftshift(np.fft.fft(x * window)))
freq = np.fft.fftshift(np.fft.fftfreq(len(x), 1/fs))

mask = (freq > 150000) & (freq < 300000)
indices = np.where(mask)[0]
peak = indices[np.argmax(spectrum[mask])]

print("Strongest frequency:", center + freq[peak], "Hz")
print("Offset from 433.920 MHz:",
      center + freq[peak] - 433_920_000, "Hz")
