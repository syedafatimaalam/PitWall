
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="PitWall | RF Signal Analysis",
    page_icon="📡",
    layout="wide",
)

# ----------------------------------------
# REAL HACKRF RECORDING CONFIGURATION
# ----------------------------------------

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from pitwall_decode import decode_burst

CAPTURE_DIR = PROJECT / "dataset" / "raw" / "live_captures"

SAMPLE_RATE = 2_000_000
CENTER_FREQ = 433_920_000
EXPECTED_BYTES = 12_000_000


# ----------------------------------------
# LOAD REAL IQ RECORDING
# ----------------------------------------

@st.cache_data(show_spinner=False, max_entries=5)
def detect_bursts(filename, file_size, modified_ns):
    raw = np.memmap(
        filename,
        dtype=np.int8,
        mode="r",
    )

    block_samples = 2000
    blocks = (len(raw) // 2) // block_samples

    if blocks == 0:
        return [], np.array([]), np.array([]), 0.0

    levels = np.empty(blocks, dtype=np.float32)

    for i in range(blocks):
        start = i * block_samples * 2

        iq = np.asarray(
            raw[start:start + block_samples * 2],
            dtype=np.float32,
        ).reshape(-1, 2)

        levels[i] = np.hypot(
            iq[:, 0],
            iq[:, 1],
        ).mean()

    noise_floor = np.percentile(levels, 20)

    smoothed = np.convolve(
        levels,
        np.ones(10) / 10,
        mode="same",
    )

    threshold = noise_floor + 5.0
    active = smoothed > threshold

    edges = np.diff(
        np.r_[False, active, False].astype(np.int8)
    )

    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    bursts = [
        (int(start * block_samples),
         int(end * block_samples))
        for start, end in zip(starts, ends)
        if 100 <= end - start <= 360
    ]

    times = np.arange(blocks) / 1000

    return bursts, times, levels, float(threshold)


# ----------------------------------------
# OOK ENVELOPE + MANCHESTER DECODING
# ----------------------------------------


def adaptive_ook_threshold(smooth, sample_rate=SAMPLE_RATE):
    """
    Estimate a local OOK threshold from real IQ amplitudes.

    Uses a rolling peak estimate so a burst whose signal
    strength changes over time can still be decoded.
    """
    # smooth is sampled at the original IQ rate.
    # A 20 ms window tracks slower amplitude changes,
    # rather than individual 0.5 ms Manchester chips.
    window = max(3, int(sample_rate * 0.020))
    window = min(window, len(smooth))

    # Efficient centered rolling maximum.
    # scipy is already commonly available with Radioconda.
    from scipy.ndimage import maximum_filter1d

    local_peak = maximum_filter1d(
        smooth,
        size=window,
        mode="nearest",
    )

    noise_floor = float(np.percentile(smooth, 10))

    threshold = (
        noise_floor
        + 0.42 * np.maximum(local_peak - noise_floor, 0)
    )

    return threshold


@st.cache_data(show_spinner=False, max_entries=10)
def analyze_burst(filename, file_size, modified_ns, start, end):
    raw = np.memmap(filename, dtype=np.int8, mode="r")
    total_samples = len(raw) // 2

    begin = max(0, start - int(0.012 * SAMPLE_RATE))
    finish = min(total_samples, end + int(0.012 * SAMPLE_RATE))

    iq = np.asarray(
        raw[begin * 2:finish * 2],
        dtype=np.float32,
    ).reshape(-1, 2)

    envelope = np.hypot(iq[:, 0], iq[:, 1])

    window = 100
    cumulative = np.concatenate(
        ([0.0], np.cumsum(envelope, dtype=np.float64))
    )
    smooth = (
        cumulative[window:] - cumulative[:-window]
    ) / window

    low = np.percentile(smooth, 10)
    high = np.percentile(smooth, 90)
    fixed_threshold = float((low + high) / 2)

    adaptive_threshold = adaptive_ook_threshold(smooth)

    decoded = None
    recovered = next(decode_burst(envelope), None)
    if recovered is not None:
        size, packet, row = recovered
        bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
        decoded = {**row, "size": size, "bits": bits}

    step = max(1, len(smooth) // 12000)

    times_ms = (
        begin
        + np.arange(0, len(smooth), step)
        + window / 2
    ) / SAMPLE_RATE * 1000

    return (
        times_ms,
        smooth[::step],
        fixed_threshold,
        decoded,
        adaptive_threshold[::step],
    )


# ----------------------------------------
# REAL RF SPECTRUM + WATERFALL
# ----------------------------------------

@st.cache_data(show_spinner=False, max_entries=5)
def calculate_spectrum(filename, file_size, modified_ns):
    raw = np.memmap(
        filename,
        dtype=np.int8,
        mode="r",
    )

    total_samples = len(raw) // 2

    fft_size = 2048

    frame_count = min(
        450,
        total_samples // fft_size,
    )

    if frame_count < 2:
        return None

    positions = np.linspace(
        0,
        total_samples - fft_size,
        frame_count,
        dtype=np.int64,
    )

    window = np.hanning(fft_size)

    power = np.empty(
        (frame_count, fft_size),
        dtype=np.float32,
    )

    for i, position in enumerate(positions):

        iq = np.asarray(
            raw[
                position * 2:
                (position + fft_size) * 2
            ],
            dtype=np.float32,
        ).reshape(-1, 2)

        complex_iq = (
            iq[:, 0] + 1j * iq[:, 1]
        )

        fft = np.fft.fftshift(
            np.fft.fft(complex_iq * window)
        )

        power[i] = np.abs(fft) ** 2

    spectrum = 10 * np.log10(
        np.maximum(power.mean(axis=0), 1e-12)
    )

    waterfall = 10 * np.log10(
        np.maximum(power, 1e-12)
    )

    frequencies = (
        CENTER_FREQ
        + np.fft.fftshift(
            np.fft.fftfreq(
                fft_size,
                d=1 / SAMPLE_RATE,
            )
        )
    ) / 1e6

    times = positions / SAMPLE_RATE

    return frequencies, spectrum, times, waterfall


# ----------------------------------------
# DASHBOARD
# ----------------------------------------

st.title("📡 PitWall — RF Signal Analysis")

st.caption(
    "Real HackRF One recordings | "
    "433.92 MHz | 2 MS/s | "
    "Manchester OOK telemetry"
)

if not CAPTURE_DIR.exists():
    st.error(
        f"Recording folder not found: {CAPTURE_DIR}"
    )
    st.stop()

recordings = sorted(
    CAPTURE_DIR.glob("*.iq"),
    key=lambda f: f.stat().st_mtime_ns,
    reverse=True,
)

if not recordings:
    st.warning(
        "No IQ recordings found. "
        "Start your HackRF receiver first."
    )
    st.stop()

selected = st.selectbox(
    "Select saved HackRF recording",
    recordings,
    format_func=lambda f: f.name,
)

file_size = selected.stat().st_size
modified_ns = selected.stat().st_mtime_ns

duration = file_size / 2 / SAMPLE_RATE

c1, c2, c3 = st.columns(3)

c1.metric(
    "Recording duration",
    f"{duration:.3f} s",
)

c2.metric(
    "IQ file size",
    f"{file_size / 1e6:.2f} MB",
)

c3.metric(
    "Capture status",
    "Complete" if file_size == EXPECTED_BYTES
    else "Partial / Other",
)

if file_size != EXPECTED_BYTES:
    st.info(
        "This is not a standard complete 3-second "
        "capture. Available samples will still be analyzed."
    )

# ----------------------------------------
# BURST DETECTION
# ----------------------------------------

with st.spinner("Detecting RF transmissions..."):
    bursts, energy_time, energy, energy_threshold = (
        detect_bursts(
            str(selected),
            file_size,
            modified_ns,
        )
    )

st.subheader("RF Activity Overview")

overview = go.Figure()

overview.add_trace(
    go.Scatter(
        x=energy_time,
        y=energy,
        mode="lines",
        name="Measured RF amplitude",
    )
)

overview.add_hline(
    y=energy_threshold,
    line_dash="dash",
    annotation_text="Burst threshold",
)

overview.update_layout(
    xaxis_title="Recording time (seconds)",
    yaxis_title="IQ magnitude",
    height=280,
)

st.plotly_chart(
    overview,
    use_container_width=True,
)

st.metric(
    "Detected RF bursts",
    len(bursts),
)

# ----------------------------------------
# GRAPH 1: ASK/OOK
# GRAPH 2: MANCHESTER
# ----------------------------------------

if bursts:

    burst_index = st.selectbox(
        "Select RF transmission",
        range(len(bursts)),
        format_func=lambda i: (
            f"Burst {i + 1} — "
            f"{bursts[i][0] / SAMPLE_RATE:.3f}s"
        ),
    )

    with st.spinner("Analyzing OOK and Manchester..."):
        (
            times_ms,
            envelope,
            threshold,
            decoded,
            adaptive_threshold,
        ) = analyze_burst(
            str(selected),
            file_size,
            modified_ns,
            *bursts[burst_index],
        )

    st.subheader("1. ASK/OOK Envelope")

    envelope_fig = go.Figure()

    envelope_fig.add_trace(
        go.Scatter(
            x=times_ms,
            y=envelope,
            mode="lines",
            name="Real RF envelope",
        )
    )

    envelope_fig.add_hline(
        y=threshold,
        line_dash="dash",
        annotation_text="OOK threshold",
    )


    envelope_fig.add_trace(
        go.Scatter(
            x=times_ms,
            y=adaptive_threshold,
            mode="lines",
            name="Adaptive OOK threshold",
            line=dict(color="orange", dash="dot"),
        )
    )


    envelope_fig.update_layout(
        xaxis_title="Time (ms)",
        yaxis_title="IQ magnitude",
        height=350,
    )

    st.plotly_chart(
        envelope_fig,
        use_container_width=True,
    )


    st.caption("The threshold lines show the OOK envelope. The decoder compares the two halves of each Manchester bit and verifies packet CRC.")

    st.subheader("2. Manchester Decoding")

    if decoded is None:

        st.warning(
            "No CRC-valid packet recovered from this burst. "
            "Try another transmission."
        )

    else:

        st.success(
            f"Manchester packet decoded · {decoded['size']}-byte frame · CRC PASSED"
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Sequence", decoded["seq"])
        m2.metric("Left RPM", decoded["left_rpm"])
        m3.metric("Right RPM", decoded["right_rpm"])
        m4.metric("Throttle", f'{decoded["throttle_pct"]}%')

        bits_to_show = st.slider(
            "Number of decoded bits to display",
            min_value=8,
            max_value=len(decoded["bits"]),
            value=48,
            step=8,
        )

        bits = decoded["bits"][:bits_to_show]
        chips = np.column_stack((bits ^ 1, bits)).ravel()

        chip_fig = go.Figure()

        chip_fig.add_trace(
            go.Scatter(
                x=np.arange(len(chips) + 1),
                y=np.r_[chips, chips[-1]],
                mode="lines",
                line_shape="hv",
                name="Manchester chips reconstructed from decoded bits",
            )
        )

        chip_fig.update_layout(
            xaxis_title="Chip index",
            yaxis_title="Chip level",
            yaxis=dict(
                tickvals=[0, 1],
                range=[-0.2, 1.2],
            ),
            height=320,
        )

        st.plotly_chart(
            chip_fig,
            use_container_width=True,
        )

        bit_string = " ".join(
            "".join(str(int(b)) for b in bits[i:i + 8])
            for i in range(0, len(bits), 8)
        )

        st.write("**Recovered binary data:**")
        st.code(bit_string)

        with st.expander("Decoded packet details"):
            st.write(
                "ESP32 timestamp:",
                decoded["timestamp_ms"],
                "ms",
            )
            st.write(
                "Lap event:",
                decoded["lap_event"],
            )
            st.write(
                "Flags:",
                decoded["flags"],
            )
            st.write("Packet HEX:")
            st.code(decoded["packet_hex"])

else:

    st.warning(
        "No RF bursts detected in this recording. "
        "The spectrum and waterfall are still available."
    )

# ----------------------------------------
# GRAPH 3: RF SPECTRUM
# GRAPH 4: WATERFALL
# ----------------------------------------

with st.spinner("Calculating real RF spectrum..."):
    spectral = calculate_spectrum(
        str(selected),
        file_size,
        modified_ns,
    )

if spectral is not None:

    frequencies, spectrum, times, waterfall = spectral

    st.subheader("3. RF Spectrum")

    spectrum_fig = go.Figure()

    spectrum_fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=spectrum,
            mode="lines",
            name="Measured spectral power",
        )
    )

    spectrum_fig.add_vline(
        x=433.92,
        line_dash="dash",
        annotation_text="433.92 MHz",
    )

    spectrum_fig.update_layout(
        xaxis_title="Frequency (MHz)",
        yaxis_title="Relative power (dB)",
        height=380,
    )

    st.plotly_chart(
        spectrum_fig,
        use_container_width=True,
    )

    st.subheader("4. RF Waterfall")

    waterfall_fig = go.Figure(
        data=go.Heatmap(
            x=frequencies,
            y=times,
            z=waterfall,
            colorscale="Viridis",
            colorbar=dict(
                title="Relative dB"
            ),
        )
    )

    waterfall_fig.update_layout(
        xaxis_title="Frequency (MHz)",
        yaxis_title="Recording time (seconds)",
        height=480,
    )

    st.plotly_chart(
        waterfall_fig,
        use_container_width=True,
    )

    st.caption(
        "Spectrum and waterfall are calculated from "
        "real IQ samples. Power is relative, not "
        "calibrated in dBm. Waterfall FFT frames "
        "are sampled across the recording."
    )

else:

    st.warning(
        "Not enough IQ samples for spectrum analysis."
    )

st.divider()

st.caption(
    f"Source: {selected.name} | "
    "Read-only IQ analysis | "
    "No synthetic data"
)
