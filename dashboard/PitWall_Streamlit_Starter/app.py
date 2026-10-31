"""PitWall: Streamlit prototype with clearly labeled synthetic telemetry.

Run: streamlit run app.py
Replace load_live_csv() with the output of the validated SDR decoder later.
"""

from pathlib import Path
import math
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="PitWall | RC Telemetry", page_icon="🏁", layout="wide")
st.markdown("""
<style>
  .stApp {background:#f3f6fa;color:#172339}
  [data-testid="stSidebar"] {background:linear-gradient(160deg,#101d31,#182b43)}
  [data-testid="stSidebar"] * {color:#eaf1fc!important}
  .block-container {padding-top:1.4rem;max-width:1500px}
  h1,h2,h3 {color:#172339!important;letter-spacing:-.035em}
  [data-testid="stMetric"] {background:white;border:1px solid #e2e9f2;border-radius:12px;padding:15px 18px;box-shadow:0 2px 8px #17233908}
  [data-testid="stMetricLabel"] {color:#607089}
  [data-testid="stMetricValue"] {color:#1463d4}
  .eyebrow {font-size:.76rem;font-weight:800;letter-spacing:.16em;color:#50729e;text-transform:uppercase}
  .smallnote {color:#718096;font-size:.85rem}
  .tag {display:inline-block;border-radius:999px;padding:5px 11px;background:#e8f6ef;color:#10834b;font-size:.76rem;font-weight:800}
  .demo {display:inline-block;border-radius:999px;padding:5px 11px;background:#fff2df;color:#a45b08;font-size:.76rem;font-weight:800;margin-left:7px}
</style>
""", unsafe_allow_html=True)

CHART_COLORS = {"left_rpm":"#1477ed", "right_rpm":"#fa474b", "throttle":"#f59015",
                "ax":"#2979f1", "ay":"#ec3b57", "az":"#12ae77"}
REQUIRED = ["timestamp_ms","seq","left_rpm","right_rpm","throttle_pct", "crc_ok"]
OPTIONAL = ["ax","ay","az","yaw_dps","predicted_rpm","driver","lap"]


@st.cache_data
def demo_data(duration=90, rate=10):
    rng = np.random.default_rng(52)
    t = np.arange(0, duration, 1/rate)
    throttle = np.clip(46 + 32*np.sin(t*.31) + 18*np.sin(t*.73), 0, 95)
    base = np.maximum(0, 18 + throttle*1.52 + 11*np.sin(t*.95))
    left = np.clip(base + rng.normal(0, 3, len(t)) - 5*np.sin(t*.37), 0, None)
    right = np.clip(base + rng.normal(0, 3, len(t)) + 5*np.sin(t*.37), 0, None)
    recovered = rng.random(len(t)) > .045
    result = pd.DataFrame({
        "timestamp_ms":(t*1000).astype(int), "seq":np.arange(len(t)),
        "left_rpm":left, "right_rpm":right, "throttle_pct":throttle,
        "crc_ok":recovered, "ax":rng.normal(0,.08,len(t)) + np.gradient(base)/100,
        "ay":rng.normal(0,.07,len(t)) + .17*np.sin(t*.37),
        "az":rng.normal(0,.045,len(t)), "yaw_dps":10*np.sin(t*.37),
        "driver":np.where(t < duration/2,"Driver A","Driver B"),
        "lap":1+(t//18).astype(int),
    })
    # Demonstration prediction only: a lagged, noisy series, NOT a trained ML result.
    result["predicted_rpm"] = ((left+right)/2 + rng.normal(0, 5, len(t))).clip(0)
    result.loc[~result.crc_ok,["left_rpm","right_rpm","throttle_pct","predicted_rpm"]] = np.nan
    return result


def normalize_csv(upload):
    raw = pd.read_csv(upload)
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    data = raw.copy()
    for name in ["timestamp_ms","seq","left_rpm","right_rpm","throttle_pct"]:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data["crc_ok"] = data["crc_ok"].astype(str).str.lower().isin(["true","1","yes","valid"])
    if data[["timestamp_ms","seq"]].isna().any().any():
        raise ValueError("timestamp_ms and seq must contain valid numbers")
    data = data.sort_values("timestamp_ms").drop_duplicates(
        subset=["timestamp_ms", "seq"],
        keep="last"
    )
    return data


def line_plot(frame, series, title, y_label, colors=None):
    fig = go.Figure()
    x = frame["timestamp_ms"]/1000
    for name, label in series:
        if name in frame:
            fig.add_trace(go.Scatter(x=x, y=frame[name], mode="lines", name=label,
                line=dict(color=(colors or CHART_COLORS).get(name,"#5e6d8d"),width=2)))
    fig.update_layout(title=dict(text=title,font=dict(size=15,color="#1b2b44")),
        template="plotly_white", height=275, margin=dict(l=16,r=16,t=44,b=22),
        legend=dict(orientation="h",y=1.16,x=1,xanchor="right"),
        paper_bgcolor="white",plot_bgcolor="white",font=dict(color="#64748b",size=11))
    fig.update_xaxes(title="Time (s)",gridcolor="#edf1f7",zeroline=False)
    fig.update_yaxes(title=y_label,gridcolor="#edf1f7",zeroline=False)
    return fig


def recovery_rate(df):
    if df.empty:
        return 0.0

    valid = df[df["crc_ok"]].sort_values("timestamp_ms")

    if valid.empty:
        return 0.0

    # Expected packets based on our 200 ms simulation interval
    first_timestamp = df["timestamp_ms"].min()
    last_timestamp = df["timestamp_ms"].max()

    expected = int((last_timestamp - first_timestamp) / 200) + 1

    received = len(valid)

    return min(100.0, 100.0 * received / max(expected, 1))


with st.sidebar:
    st.markdown("## 🏁 PitWall")
    st.caption("RC CAR TELEMETRY · EE-322L")
    page = st.radio("Navigation",["Live Dashboard","RF Link Monitor","ML Prediction","Data Log","Driver Comparison","Settings"], label_visibility="collapsed")
    st.divider()
    source = st.selectbox(
    "Data source",
    ["Synthetic demo", "Upload decoded CSV", "Live Receiver"])
    upload = st.file_uploader("Decoded telemetry CSV",type="csv") if source == "Upload decoded CSV" else None
    if source == "Synthetic demo":
        auto = st.toggle("Live simulation",value=True)
        st.caption("Simulation data only — no RF connection")
    else:
        auto = False
        st.caption("Load validated decoded packets from your receiver")
    st.divider()
    st.caption("433.92 MHz  •  ASK/OOK  •  Manchester")

if source == "Upload decoded CSV" and upload is not None:
    try:
        df = normalize_csv(upload)
    except Exception as exc:
        st.error(f"CSV error: {exc}")
        st.stop()
elif source == "Upload decoded CSV":
    st.info("Upload a decoded CSV to view real recordings. The format is explained in README.md.")
    st.stop()
elif source == "Live Receiver":
    df = pd.DataFrame(columns=REQUIRED)
else:
    df = demo_data()

st.markdown('<div class="eyebrow">PitWall / Operations console</div>',unsafe_allow_html=True)
st.title(page)
st.markdown(f'<span class="tag">● {"CSV playback" if source != "Synthetic demo" else "Simulation ready"}</span><span class="demo">{"RECORDED DATA" if source != "Synthetic demo" else "SYNTHETIC DATA — NOT LIVE RF"}</span>',unsafe_allow_html=True)
st.write("")


@st.fragment(run_every="1s" if (auto or source == "Live Receiver") else None)
def display():
    global df

    if source == "Live Receiver":
        from pathlib import Path

        csv_path = (
            Path(__file__).resolve().parents[2]
            / "pitwall_telemetry.csv"
        )

        if not csv_path.exists():
            st.warning("Waiting for receiver telemetry...")
            return

        try:
            df = normalize_csv(csv_path)
        except Exception as error:
            st.warning(f"Waiting for valid telemetry: {error}")
            return
    if auto:
        idx = 80 + int(time.monotonic()*10)%max(1,len(df)-80)
        frame = df.iloc[max(0,idx-220):idx].copy()
    else:
        frame = df.tail(220).copy()
    valid = frame[frame.crc_ok]
    if valid.empty:
        st.warning("No valid packets in the selected interval")
        return
    last = valid.iloc[-1]
    prr = recovery_rate(frame)
    frame["average_rpm"] = frame[["left_rpm","right_rpm"]].mean(axis=1)

    if page == "Live Dashboard":
        a,b,c,d = st.columns(4)
        a.metric("◉ Left wheel",f"{last.left_rpm:.0f} RPM")
        b.metric("◉ Right wheel",f"{last.right_rpm:.0f} RPM")
        c.metric("▰ Motor command",f"{last.throttle_pct:.0f}%")
        d.metric("⌁ RF packet recovery",f"{prr:.1f}%")
        col1,col2 = st.columns(2)
        with col1:
            st.plotly_chart(line_plot(frame,[("left_rpm","Left"),("right_rpm","Right")],"Wheel speed", "RPM"),width="stretch")
        with col2:
            st.plotly_chart(line_plot(frame,[("throttle_pct","Throttle command")],"Motor command","PWM (%)"),width="stretch")
        col3,col4 = st.columns(2)
        with col3:
            if all(k in frame for k in ["ax","ay","az"]):
                st.plotly_chart(line_plot(frame,[("ax","X"),("ay","Y"),("az","Z")],"IMU acceleration (local log only)","g, demo-relative"),width="stretch")
            else:
                st.info("IMU data not present in RF-decoded CSV")
        with col4:
            if "predicted_rpm" in frame:
                st.plotly_chart(line_plot(frame,[("average_rpm","Measured"),("predicted_rpm","Prediction / demo")],"0.5 s wheel-speed forecast","RPM",{"average_rpm":"#1477ed","predicted_rpm":"#fa474b"}),width="stretch")
                st.caption("Synthetic forecast is illustrative, not a trained model.") if source == "Synthetic demo" else None
            else:
                st.info("ML predictions will appear after the prediction pipeline is connected.")
        st.markdown("#### Latest validated packet")
        st.dataframe(pd.DataFrame([{"Sequence":int(last.seq),"Timestamp":f"{last.timestamp_ms:.0f} ms","Left RPM":round(last.left_rpm,1),"Right RPM":round(last.right_rpm,1),"Command":f"{last.throttle_pct:.0f}%","CRC":"PASS"}]),hide_index=True,width="stretch")

    elif page == "RF Link Monitor":
        a,b,c = st.columns(3)
        a.metric("Packet recovery rate",f"{prr:.1f}%")
        b.metric("Valid CRC frames",f"{int(frame.crc_ok.sum())}")
        expected = int(
            round((frame["timestamp_ms"].max() - frame["timestamp_ms"].min()) / 200)
        ) + 1
        missing = max(0, expected - len(frame))
        c.metric("Estimated missing packets", f"{missing}")
        frame["recovered"] = frame.crc_ok.astype(int)
        st.plotly_chart(line_plot(frame,[("recovered","Packet accepted")],"Packet validity over time","Valid (1) / missing (0)"),width="stretch")
        st.info(
        "PRR and missing packets are estimated using the 200 ms "
        "simulation interval. Actual RF link measurements will "
        "require HackRF receiver data."
        )
    elif page == "ML Prediction":
        st.markdown("#### Predict average wheel RPM 0.5 seconds ahead")
        st.write("The proposed model is a Random Forest regressor evaluated against a persistence baseline. This screen will display the actual results when your model is connected.")
        st.plotly_chart(line_plot(frame,[("average_rpm","Measured")]+([("predicted_rpm","Illustrative estimate")] if "predicted_rpm" in frame else []),"Wheel-speed history and forecast","RPM",{"average_rpm":"#1477ed","predicted_rpm":"#fa474b"}),width="stretch")
        if source == "Synthetic demo":
            st.warning("Demo predictions are placeholders. MAE/RMSE must be computed on held-out real driving runs.")
        else:
            st.caption("Only show MAE/RMSE after joining predictions to their correct future measurement timestamps.")

    elif page == "Data Log":
        st.dataframe(frame.sort_values("timestamp_ms",ascending=False),hide_index=True,width="stretch",height=480)
        st.download_button("Download visible data",frame.to_csv(index=False),file_name="pitwall_telemetry.csv",mime="text/csv")

    elif page == "Driver Comparison":
        if "driver" in df and "lap" in df:
            summary = df.groupby("driver",dropna=False).agg(samples=("seq","size"),average_left_rpm=("left_rpm","mean"),average_right_rpm=("right_rpm","mean")).round(2)
            st.dataframe(summary,width="stretch")
            st.caption("Lap times require verified lap-start events. This view currently compares recorded wheel-speed data only.")
        else:
            st.info("Driver comparison requires driver and lap identifiers in the dataset.")

    elif page == "Settings":
        st.markdown("### Data contract")
        st.code("timestamp_ms,seq,left_rpm,right_rpm,throttle_pct,crc_ok",language="text")
        st.write("Optional fields: " + ", ".join(OPTIONAL))
        st.info("Integrate the SDR decoder by appending CRC-validated packets to a CSV or a thread-safe queue. Keep decoding separate from UI rendering.")


display()
