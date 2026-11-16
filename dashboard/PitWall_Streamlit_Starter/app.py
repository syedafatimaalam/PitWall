"""PitWall: Streamlit prototype with clearly labeled synthetic telemetry.

Run: streamlit run app.py
Replace load_live_csv() with the output of the validated SDR decoder later.
"""
import joblib
from pathlib import Path
import math
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from behavior_inference import predict_history

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
def demo_data(duration=60, rate=10):
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
    # Preserve the order in which real RF packets were saved.
# ESP32 uptime can reset, so do not sort by timestamp_ms.
    data = data.drop_duplicates(
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
        ["Live Receiver", "Upload decoded CSV"]
    )
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

        project_dir = next(
            (
                p for p in Path(__file__).resolve().parents
                if (p / "pitwall_decode.py").exists()
            ),
            None
        )

        if project_dir is None:
            st.error("Cannot locate PitWall project folder.")
            return

        csv_path = project_dir / "processed" / "pitwall_live_rf.csv"

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
        st.subheader("AI Driving Behavior Prediction")
        st.caption(
            "Real RF telemetry | Random Forest | "
            "Next consecutive packet prediction"
        )

        project_dir = next(
            (
                p for p in Path(__file__).resolve().parents
                if (p / "pitwall_decode.py").exists()
            ),
            None
        )

        if project_dir is None:
            st.error("Cannot locate PitWall project folder.")
            return

        model_path = (
            project_dir / "processed"
            / "pitwall_real_behavior_model.pkl"
        )

        csv_path = (
            project_dir / "processed"
            / "pitwall_live_rf.csv"
        )

        if source != "Live Receiver":
            st.warning("Select Live Receiver to use real RF telemetry.")
            return

        if not model_path.exists():
            st.warning("Real-data ML model not found.")
            return

        if not csv_path.exists():
            st.warning("Waiting for decoded RF telemetry.")
            return

        # Load only our own locally trained model.
        # Never load untrusted pickle/joblib files.
        bundle = joblib.load(model_path)
        model = bundle["model"]
        features = bundle["features"]

        # df is already loaded from the validated RF CSV.
        live = df[df["crc_ok"]].copy()

        if len(live) < 2:
            st.warning("Waiting for at least two valid RF packets.")
            return

        
        # Find the newest usable pair of real RF packets.
        # Do not invent or interpolate missing telemetry.
                # Use only the latest two CRC-valid RF packets.
        # Never display an older prediction as LIVE.

        live = live.reset_index(drop=True)

        if len(live) < 2:
            st.warning(
                "Waiting for two CRC-valid RF packets."
            )
            return

        current = live.iloc[-1]
        previous = live.iloc[-2]

        dt_ms = (
            float(current["timestamp_ms"])
            - float(previous["timestamp_ms"])
        )

        seq_gap = (
            int(current["seq"])
            - int(previous["seq"])
        ) % 256

        latest_received = pd.to_datetime(
            current["received_utc"],
            utc=True,
            errors="coerce"
        )

        if pd.isna(latest_received):
            st.warning(
                "Latest RF packet has no valid reception timestamp."
            )
            return

        age_seconds = (
            pd.Timestamp.now(tz="UTC") - latest_received
        ).total_seconds()

        st.caption(
            f"Latest RF packet: seq {int(current['seq'])} | "
            f"Age: {age_seconds:.1f}s | "
            f"Sequence gap: {seq_gap}"
        )

        if age_seconds > 15 or age_seconds < -5:
            st.warning(
                "RECORDED ML — No fresh CRC-valid RF packets. "
                "Waiting for new decoder output."
            )
            return

        if seq_gap != 1 or not (0 < dt_ms <= 2000):
            st.warning(
                "New RF packet received, but the latest two "
                "packets are not consecutive. "
                f"Sequence gap: {seq_gap}, "
                f"Time difference: {dt_ms:.0f} ms."
            )
            return

        is_recent = True
        st.success("🟢 LIVE ML — Fresh, consecutive RF packets")

        avg_rpm = (
            float(current["left_rpm"])
            + float(current["right_rpm"])
        ) / 2

        previous_avg_rpm = (
            float(previous["left_rpm"])
            + float(previous["right_rpm"])
        ) / 2

        input_values = {
            "left_rpm": float(current["left_rpm"]),
            "right_rpm": float(current["right_rpm"]),
            "throttle_pct": float(current["throttle_pct"]),
            "avg_rpm": avg_rpm,
            "rpm_difference": abs(
                float(current["left_rpm"])
                - float(current["right_rpm"])
            ),
            "previous_avg_rpm": previous_avg_rpm,
            "previous_throttle": float(
                previous["throttle_pct"]
            ),
            "previous_dt_ms": dt_ms,
        }

        input_frame = pd.DataFrame(
            [input_values],
            columns=features,
        )

        prediction = model.predict(input_frame)[0]
        probabilities = model.predict_proba(input_frame)[0]

        probability_map = {
            label: float(prob)
            for label, prob in zip(
                model.classes_,
                probabilities,
            )
        }

        confidence = max(probability_map.values()) * 100

        # File modification time indicates whether new decoded
        # data has recently reached the receiver CSV.
        
        a, b, c = st.columns(3)

        a.metric("Predicted behavior", prediction)
        b.metric("Model probability", f"{confidence:.1f}%")
        c.metric("Current average RPM", f"{avg_rpm:.0f}")

        st.caption(
            f"Input packet: sequence {int(current['seq'])} | "
            f"ESP32 timestamp: {int(current['timestamp_ms'])} ms"
        )

        st.markdown("#### Prediction probabilities")

        probability_df = pd.DataFrame(
            {
                "Behavior": list(probability_map.keys()),
                "Probability": [
                    round(p * 100, 1)
                    for p in probability_map.values()
                ],
            }
        )

        st.bar_chart(
            probability_df.set_index("Behavior")
        )


        st.markdown("#### Live ML prediction history")

        history_rows = []
        recent = live.tail(100).reset_index(drop=True)

        for i in range(1, len(recent)):
            now = recent.iloc[i]
            before = recent.iloc[i - 1]

            delta_ms = (
                float(now["timestamp_ms"])
                - float(before["timestamp_ms"])
            )

            sequence_gap = (
                int(now["seq"]) - int(before["seq"])
            ) % 256

            if (
                sequence_gap != 1
                or delta_ms <= 0
                or delta_ms > 2000
            ):
                continue

            average_rpm = (
                float(now["left_rpm"])
                + float(now["right_rpm"])
            ) / 2

            previous_average_rpm = (
                float(before["left_rpm"])
                + float(before["right_rpm"])
            ) / 2

            values = {
                "left_rpm": float(now["left_rpm"]),
                "right_rpm": float(now["right_rpm"]),
                "throttle_pct": float(now["throttle_pct"]),
                "avg_rpm": average_rpm,
                "rpm_difference": abs(
                    float(now["left_rpm"])
                    - float(now["right_rpm"])
                ),
                "previous_avg_rpm": previous_average_rpm,
                "previous_throttle": float(
                    before["throttle_pct"]
                ),
                "previous_dt_ms": delta_ms,
            }

            sample = pd.DataFrame([values], columns=features)
            probabilities = model.predict_proba(sample)[0]

            history_rows.append({
                "packet_index": i,
                "sequence": int(now["seq"]),
                **{
                    label: float(prob) * 100
                    for label, prob in zip(
                        model.classes_, probabilities
                    )
                },
            })

        if history_rows:
            history_df = pd.DataFrame(history_rows)

            probability_columns = list(model.classes_)

            st.line_chart(
                history_df.set_index("packet_index")[
                    probability_columns
                ],
                y_label="Predicted probability (%)",
                x_label="Packet position in recent history",
            )

            st.caption(
                "Predictions use real CRC-valid RF packets. "
                "The horizontal axis is packet position, "
                "not elapsed time."
            )

            st.dataframe(
                history_df.tail(10),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info(
                "Waiting for consecutive validated RF packets "
                "to build prediction history."
            )


        st.markdown("#### Current measured telemetry")

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Left RPM",
            f"{float(current['left_rpm']):.0f}"
        )
        m2.metric(
            "Right RPM",
            f"{float(current['right_rpm']):.0f}"
        )
        m3.metric(
            "Throttle",
            f"{float(current['throttle_pct']):.0f}%"
        )

        st.info(
            "Experimental real-data model. Predicts the "
            "next consecutive RF packet's RPM-change class, "
            "not a fixed 0.5-second horizon. Model "
            "probabilities are not calibrated confidence "
            "scores. Held-out accuracy: 66.7% on 30 "
            "examples; further independent driving tests "
            "are needed."
        )
#         elif page == "ML Prediction":
# st.markdown("#### Future Driving Behavior Prediction")
#                 st.caption("Random Forest model — predicts behavior 0.5 seconds ahead")                

        try:
            predictions = predict_history(frame)
            latest = predictions.iloc[-1]

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Behavior",
                    latest["predicted_behavior"]
                )

            with col2:
                st.metric(
                    "Prediction Confidence",
                    f"{latest['confidence']:.1%}"
                )

            st.markdown("### Prediction History")

            st.dataframe(
                predictions[
                    ["timestamp_ms", "predicted_behavior", "confidence"]
                ].tail(20).iloc[::-1],
                hide_index=True,
                use_container_width=True
            )

            if source == "Synthetic demo":
                st.warning(
                    "Predictions use simulated telemetry. "
                    "Real driving accuracy has not yet been evaluated."
                )

        except (ValueError, OSError, KeyError) as error:
            st.warning(str(error))
        
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
