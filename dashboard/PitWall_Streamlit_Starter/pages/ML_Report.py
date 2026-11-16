"""Post-recording ML report for the PitWall Streamlit dashboard."""

import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from pitwall_train_real_ml import CSV, MODEL_PATH, prepare_data

CAPTURE_DIR = PROJECT / "dataset" / "raw" / "live_captures"
IDLE_SECONDS = 12
LABELS = ["ACCELERATING", "DECELERATING", "STEADY"]

st.set_page_config(page_title="PitWall | ML Report", layout="wide")
st.title("ML Report")


def latest_capture_run():
    """Return the latest group of IQ writes separated by less than 12 seconds."""
    modified = []
    for capture in CAPTURE_DIR.glob("pitwall_*.iq"):
        try:
            modified.append(capture.stat().st_mtime)
        except FileNotFoundError:
            continue

    if not modified:
        return []

    modified.sort(reverse=True)
    run = modified[:1]
    for timestamp in modified[1:]:
        if run[-1] - timestamp >= IDLE_SECONDS:
            break
        run.append(timestamp)
    return run


@st.cache_data(show_spinner=False, max_entries=4)
def read_telemetry(csv_modified_ns):
    raw = pd.read_csv(CSV)
    if not {"received_utc", "crc_ok"}.issubset(raw.columns):
        raise ValueError("Expected received_utc and crc_ok columns in the telemetry CSV.")
    raw["received_utc"] = pd.to_datetime(raw["received_utc"], utc=True, format="mixed", errors="coerce")
    return raw


@st.cache_resource(show_spinner=False)
def load_model(model_modified_ns):
    bundle = joblib.load(MODEL_PATH)
    bundle["model"].set_params(n_jobs=1)
    return bundle


def score_run(rows, bundle):
    scored = prepare_data(rows)
    if not scored.empty:
        scored["prediction"] = bundle["model"].predict(scored[bundle["features"]])
        scored["match"] = scored["prediction"].eq(scored["target"])
    return scored


@st.fragment(run_every="2s")
def show_report():
    run = latest_capture_run()
    if not run:
        st.info("Start a PitWall recording to generate a report.")
        return

    last_write, first_write = run[0], run[-1]
    if time.time() - last_write < IDLE_SECONDS:
        st.info("Waiting for recording and decoding to finish. Leave this page open.")
        return

    try:
        raw = read_telemetry(CSV.stat().st_mtime_ns)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.info("No decoded telemetry CSV is available yet.")
        return
    except (OSError, ValueError) as error:
        st.warning(f"Could not read decoded telemetry. {error}")
        return

    start = pd.to_datetime(first_write - 1, unit="s", utc=True)
    end = pd.to_datetime(last_write + IDLE_SECONDS, unit="s", utc=True)
    crc_valid = raw["crc_ok"].astype(str).str.lower().isin(["true", "1", "yes", "valid"])
    rows = raw.loc[raw["received_utc"].between(start, end) & crc_valid].copy()

    local_start = datetime.fromtimestamp(first_write).astimezone()
    st.subheader("Latest recording run")
    st.caption(f"{local_start:%d %b %Y, %H:%M:%S %Z} · {len(run)} IQ files")

    if rows.empty:
        st.warning("No CRC-valid v1 packets were logged for this run.")
        return

    try:
        model_stat = MODEL_PATH.stat()
    except FileNotFoundError:
        st.info("Train and save the real-data model to score this run.")
        return

    try:
        scored = score_run(rows, load_model(model_stat.st_mtime_ns))
    except (ValueError, KeyError, OSError) as error:
        st.warning(f"Could not score this run. {error}")
        return

    valid, eligible, accuracy = st.columns(3)
    valid.metric("CRC-valid packets", len(rows))
    eligible.metric("Scored predictions", len(scored))

    if scored.empty:
        accuracy.metric("Run accuracy", "—")
        st.info("Scoring needs a usable previous packet and an observed consecutive next packet.")
        return

    matches = int(scored["match"].sum())
    accuracy.metric("Run accuracy", f"{matches / len(scored):.1%}")
    st.caption(
        f"{matches} of {len(scored)} scored predictions matched. "
        "The saved model replays packets from this run and predicts the next "
        "packet using only current and earlier readings."
    )
    saved_at = pd.to_datetime(model_stat.st_mtime, unit="s", utc=True)
    if rows["received_utc"].min() < saved_at:
        st.warning("Some packets predate the saved model and may have been used for training.")

    st.subheader("Predictions by actual wheel behavior")
    by_class = scored.groupby("target")["match"].agg(["size", "sum"])
    by_class = by_class.reindex(LABELS, fill_value=0).rename(
        columns={"size": "Scored", "sum": "Correct"}
    )
    by_class.index.name = "Actual"
    st.dataframe(by_class)

    st.subheader("Packet-by-packet results")
    details = pd.DataFrame(
        {
            "Packet": scored["seq"].astype(int),
            "Next packet": (scored["seq"].astype(int) + 1) % 256,
            "Average RPM": scored["avg_rpm"],
            "Next average RPM": scored["next_avg_rpm"],
            "Prediction": scored["prediction"],
            "Actual": scored["target"],
            "Matched": scored["match"],
        }
    )
    st.dataframe(details, hide_index=True)


show_report()
