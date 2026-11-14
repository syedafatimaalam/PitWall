"""Inference on 10 Hz PitWall telemetry using the trained synthetic classifier."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[2] / "ml" / "pitwall_behavior_model.pkl"


def predict_history(frame, model_path=MODEL_PATH):
    """Return predictions at valid 100 ms sample instants; never fabricate yaw."""
    if not model_path.exists():
        raise ValueError(f"Trained model missing: {model_path}. Run python ml/pitwall_future_behavior.py")
    if "yaw_dps" not in frame or frame["yaw_dps"].isna().any():
        raise ValueError("Real-time behavior prediction requires yaw_dps from an IMU. Current RF packet has no yaw field.")
    df = frame.loc[frame["crc_ok"]].copy().sort_values("timestamp_ms")
    if len(df) < 4:
        raise ValueError("At least four CRC-valid telemetry samples are required.")
    for col in ["left_rpm", "right_rpm", "throttle_pct", "yaw_dps", "timestamp_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["avg_rpm"] = (df["left_rpm"] + df["right_rpm"]) / 2
    df["rpm_1_ago"] = df["avg_rpm"].shift(1)
    df["rpm_3_ago"] = df["avg_rpm"].shift(3)
    df["throttle_1_ago"] = df["throttle_pct"].shift(1)
    df["yaw_1_ago"] = df["yaw_dps"].shift(1)
    # Training rows are uniformly spaced by 100 ms. Do not silently treat
    # dropped RF packets or a slower acquisition rate as equivalent samples.
    t = df["timestamp_ms"]
    regular = ((t - t.shift(1)).sub(100).abs() <= 25) & ((t - t.shift(3)).sub(300).abs() <= 75)
    model_data = joblib.load(model_path)
    features = model_data["features"]
    ready = df.loc[regular].dropna(subset=features).copy()
    if ready.empty:
        raise ValueError("Need four consecutive valid telemetry samples around 100 ms apart (10 Hz).")
    model = model_data["model"]
    probabilities = model.predict_proba(ready[features])
    ready["predicted_behavior"] = model.classes_[np.argmax(probabilities, axis=1)]
    ready["confidence"] = probabilities.max(axis=1)
    return ready
