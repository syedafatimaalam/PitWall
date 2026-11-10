"""Place this page in dashboard/pages/ under your Streamlit dashboard."""
from pathlib import Path
import pandas as pd
import streamlit as st

CSV_FILE = Path(__file__).resolve().parents[2] / "pitwall_live_rf.csv"

st.title("PitWall | Live RF Wheel Telemetry")
st.caption("Displays only CRC-valid packets decoded from HackRF. Near-live batch capture, not gapless reception.")

@st.fragment(run_every="2s")
def show_feed():
    if not CSV_FILE.exists():
        st.info("No live data yet. In another terminal, run python live_rf_bridge.py")
        st.code(str(CSV_FILE))
        return
    try:
        df = pd.read_csv(CSV_FILE)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        st.warning(f"Cannot read live feed yet: {exc}")
        return
    if df.empty:
        st.info("Waiting for CRC-valid packets...")
        return
    df = df.tail(200).copy()
    last = df.iloc[-1]
    a, b, c = st.columns(3)
    a.metric("Right wheel", f"{int(last['right_rpm'])} RPM")
    b.metric("Left wheel", f"{int(last['left_rpm'])} RPM")
    c.metric("CRC-valid packets logged", f"{len(pd.read_csv(CSV_FILE))}")
    st.line_chart(df.set_index("tx_timestamp_ms")[["left_rpm", "right_rpm"]],
                  x_label="ESP32 timestamp (ms)", y_label="RPM")
    st.dataframe(df.tail(12).iloc[::-1], hide_index=True, use_container_width=True)
    st.caption(f"Last saved record (UTC): {last['received_utc']} — verify freshness before calling this live")

show_feed()
