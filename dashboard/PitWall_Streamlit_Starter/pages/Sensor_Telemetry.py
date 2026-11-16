
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="PitWall | Sensor Telemetry",
    page_icon="🏎️",
    layout="wide",
)

PROJECT = Path(__file__).resolve().parents[3]

CSV_FILE = (
    PROJECT / "processed" / "pitwall_telemetry_v2.csv"
)

st.title("🏎️ PitWall — Full Sensor Telemetry")
st.caption(
    "Real CRC-validated ESP32 RF packets | "
    "MPU6050 | Wheel RPM | Lap Counter"
)

if not CSV_FILE.exists():
    st.warning(
        "Waiting for protocol v2 telemetry. "
        "Upload the new ESP32 firmware and record "
        "fresh HackRF samples first."
    )
    st.stop()

try:
    df = pd.read_csv(CSV_FILE)
except (OSError, pd.errors.EmptyDataError) as exc:
    st.error(f"Cannot read telemetry CSV: {exc}")
    st.stop()

if df.empty:
    st.info("No decoded v2 telemetry packets yet.")
    st.stop()

required = [
    "timestamp_ms", "left_rpm", "right_rpm",
    "throttle_pct", "lap_count", "mpu_valid",
    "accel_x_mg", "accel_y_mg", "accel_z_mg",
    "gyro_x_dps10", "gyro_y_dps10", "gyro_z_dps10",
]

missing = [c for c in required if c not in df.columns]

if missing:
    st.error(f"Missing CSV columns: {missing}")
    st.stop()

numeric = [
    "timestamp_ms", "left_rpm", "right_rpm",
    "throttle_pct", "lap_count",
    "accel_x_mg", "accel_y_mg", "accel_z_mg",
    "gyro_x_dps10", "gyro_y_dps10", "gyro_z_dps10",
]

for column in numeric:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )

df["mpu_valid"] = (
    df["mpu_valid"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
)

df = df.dropna(subset=["timestamp_ms"])
df = df.drop_duplicates(
    subset=["timestamp_ms", "seq"],
    keep="last",
)

# Keep capture/append order rather than sorting by
# ESP32 millis(), which resets after a reboot.
df = df.reset_index(drop=True)

latest = df.iloc[-1]

def metric_value(value, unit=""):
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}{unit}"


st.subheader("Live Receiver — Latest Decoded Packet")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Left Wheel",
    metric_value(latest["left_rpm"], " RPM"),
)
c2.metric(
    "Right Wheel",
    metric_value(latest["right_rpm"], " RPM"),
)
c3.metric(
    "Throttle",
    metric_value(latest["throttle_pct"], "%"),
)
c4.metric(
    "Lap Count",
    int(latest["lap_count"])
    if pd.notna(latest["lap_count"])
    else "N/A",
)

st.write(
    "**MPU6050 status:**",
    "Valid measurement"
    if latest["mpu_valid"]
    else "Unavailable",
)

valid_df = df[df["mpu_valid"]].copy()

def sensor_chart(frame, fields, title, yaxis, scale=1.0):
    fig = go.Figure()

    # Use packet index to avoid discontinuities
    # when ESP32 timestamps reset.
    x = frame.index.to_numpy()

    for field, label in fields:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=frame[field] * scale,
                mode="lines+markers",
                name=label,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Decoded packet index",
        yaxis_title=yaxis,
        height=370,
        legend=dict(orientation="h"),
    )

    return fig


if valid_df.empty:
    st.warning(
        "No valid MPU6050 measurements have "
        "been received yet."
    )
else:
    st.subheader("Accelerometer — MPU6050")

    a1, a2, a3 = st.columns(3)

    last_valid = valid_df.iloc[-1]

    a1.metric(
        "Acceleration X",
        metric_value(last_valid["accel_x_mg"], " mg"),
    )
    a2.metric(
        "Acceleration Y",
        metric_value(last_valid["accel_y_mg"], " mg"),
    )
    a3.metric(
        "Acceleration Z",
        metric_value(last_valid["accel_z_mg"], " mg"),
    )

    st.plotly_chart(
        sensor_chart(
            valid_df,
            [
                ("accel_x_mg", "X"),
                ("accel_y_mg", "Y"),
                ("accel_z_mg", "Z"),
            ],
            "Three-Axis Acceleration",
            "Acceleration (g)",
            scale=0.001,
        ),
        use_container_width=True,
    )

    st.subheader("Gyroscope — MPU6050")

    g1, g2, g3 = st.columns(3)

    g1.metric(
        "Gyroscope X",
        metric_value(
            last_valid["gyro_x_dps10"] / 10,
            " °/s",
        ),
    )
    g2.metric(
        "Gyroscope Y",
        metric_value(
            last_valid["gyro_y_dps10"] / 10,
            " °/s",
        ),
    )
    g3.metric(
        "Gyroscope Z",
        metric_value(
            last_valid["gyro_z_dps10"] / 10,
            " °/s",
        ),
    )

    st.plotly_chart(
        sensor_chart(
            valid_df,
            [
                ("gyro_x_dps10", "X"),
                ("gyro_y_dps10", "Y"),
                ("gyro_z_dps10", "Z"),
            ],
            "Three-Axis Angular Velocity",
            "Angular velocity (°/s)",
            scale=0.1,
        ),
        use_container_width=True,
    )

st.subheader("Wheel Speed")

st.plotly_chart(
    sensor_chart(
        df,
        [
            ("left_rpm", "Left wheel"),
            ("right_rpm", "Right wheel"),
        ],
        "Left vs Right Wheel RPM",
        "RPM",
    ),
    use_container_width=True,
)

st.subheader("Decoded Telemetry Log")

st.dataframe(
    df.tail(100).iloc[::-1],
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Download v2 Telemetry CSV",
    data=df.to_csv(index=False),
    file_name="pitwall_telemetry_v2_export.csv",
    mime="text/csv",
)

st.caption(
    "All displayed values originate from CRC-valid "
    "RF packets. MPU6050 readings are displayed "
    "only when the transmitter marks them valid. "
    "Refresh the page to load newly decoded packets."
)
