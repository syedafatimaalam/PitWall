import argparse
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "processed" / "pitwall_live_rf.csv"
MODEL_PATH = ROOT / "processed" / "pitwall_real_behavior_model.pkl"

bundle = joblib.load(MODEL_PATH)
bundle["model"].set_params(n_jobs=1)

FEATURES = [
    "left_rpm",
    "right_rpm",
    "throttle_pct",
    "avg_rpm",
    "rpm_difference",
    "previous_avg_rpm",
    "previous_throttle",
    "previous_dt_ms",
]

MIN_CHANGE_RPM = 100.0

def prepare_features(raw):
    df = raw.copy()

    required = [
        "timestamp_ms", "seq", "left_rpm",
        "right_rpm", "throttle_pct", "crc_ok"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["crc_ok"] = (
        df["crc_ok"].astype(str).str.lower()
        .isin(["true", "1", "yes", "valid"])
    )

    df = df[df["crc_ok"]].copy()

    for col in required[:-1]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required[:-1])

    # Preserve receiver order, not global ESP32 uptime order.
    # A timestamp decrease indicates a new device session.
    df["session"] = (
        df["timestamp_ms"].diff().fillna(0).lt(0).cumsum()
    )

    # Sequence continuity helps avoid training across missing packets.
    df["seq_gap"] = (
        df.groupby("session")["seq"].diff().mod(256)
    )

    df["dt_ms"] = df.groupby("session")["timestamp_ms"].diff()

    # Break sequences when packet timing is implausible or packets
    # were missed. No interpolation or synthetic rows.
    new_segment = (
        df["seq_gap"].ne(1)
        | df["dt_ms"].le(0)
        | df["dt_ms"].gt(2000)
    )

    df["segment"] = new_segment.cumsum()

    df["avg_rpm"] = (
        df["left_rpm"] + df["right_rpm"]
    ) / 2

    df["rpm_difference"] = (
        df["left_rpm"] - df["right_rpm"]
    ).abs()

    group = df.groupby("segment", sort=False)

    df["previous_avg_rpm"] = group["avg_rpm"].shift(1)
    df["previous_throttle"] = group["throttle_pct"].shift(1)
    df["previous_dt_ms"] = group["timestamp_ms"].diff()

    return df


def prepare_data(raw):
    df = prepare_features(raw)
    group = df.groupby("segment", sort=False)

    # Label the change from this packet to the next real packet.
    df["next_avg_rpm"] = group["avg_rpm"].shift(-1)
    df["next_dt_ms"] = (
        group["timestamp_ms"].shift(-1) - df["timestamp_ms"]
    )

    change = df["next_avg_rpm"] - df["avg_rpm"]

    df["target"] = np.select(
        [
            change > MIN_CHANGE_RPM,
            change < -MIN_CHANGE_RPM,
        ],
        ["ACCELERATING", "DECELERATING"],
        default="STEADY",
    )

    df = df.dropna(
        subset=FEATURES + ["next_avg_rpm", "next_dt_ms"]
    )

    df = df[
        df["next_dt_ms"].between(200, 2000)
    ].copy()

    return df


def train_model():
    raw = pd.read_csv(CSV)
    data = prepare_data(raw)

    print("Real CRC-valid training examples:", len(data))
    print("Class counts:")
    print(data["target"].value_counts())

    if len(data) < 100:
        print(
            "\nNot enough consecutive real packets for "
            "reliable model training."
        )
        return

    if data["target"].nunique() < 2:
        print(
            "\nNeed driving sessions with more than one "
            "behavior class."
        )
        return

    # Never randomly split adjacent packets into train/test.
    # Keep complete contiguous segments together.
    if data["segment"].nunique() < 3:
        print(
            "\nNeed at least 3 independent continuous "
            "segments for a meaningful evaluation."
        )
        return

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            data[FEATURES],
            data["target"],
            groups=data["segment"],
        )
    )

    train = data.iloc[train_idx]
    test = data.iloc[test_idx]

    if train["target"].nunique() < 2:
        print("Training split lacks class diversity.")
        return

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train[FEATURES], train["target"])

    predicted = model.predict(test[FEATURES])

    print("\nHeld-out segment accuracy:",
          round(accuracy_score(test["target"], predicted), 3))

    print("\nClassification report:")
    print(classification_report(
        test["target"], predicted, zero_division=0
    ))

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "horizon": "next consecutive RF packet",
            "minimum_change_rpm": MIN_CHANGE_RPM,
            "training_rows": len(train),
            "test_rows": len(test),
        },
        MODEL_PATH,
    )

    print("\nSaved real-data model:", MODEL_PATH)


def predict_latest(raw, bundle, require_fresh=False):
    if raw.empty:
        return "Waiting for telemetry packets."

    # Select the latest received row. Feature preparation may discard it.
    latest_index = raw.index[-1]
    if require_fresh:
        received_at = pd.to_datetime(
            raw.iloc[-1].get("received_utc"), utc=True, errors="coerce"
        )
        if pd.isna(received_at) or pd.Timestamp.now(tz="UTC") - received_at > pd.Timedelta(seconds=8):
            return "Waiting for fresh RF packets."

    features = prepare_features(raw)
    if latest_index not in features.index:
        return "Waiting for a valid latest packet."
    row = features.loc[[latest_index]]
    model_input = row[bundle["features"]]
    if model_input.isna().any(axis=None):
        return "Waiting for two consecutive RF packets."

    prediction = bundle["model"].predict(model_input)[0]
    packet = row.iloc[0]
    return (
        f"seq={int(packet.seq)} | L={int(packet.left_rpm)} RPM | "
        f"R={int(packet.right_rpm)} RPM | throttle={int(packet.throttle_pct)}% | "
        f"next packet: {prediction}"
    )


def run_inference(watch):
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Train the real-data model first: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    last_message = None
    while True:
        try:
            # Mirror the decoder's append-only CSV without touching the file.
            raw = pd.read_csv(CSV)
            message = predict_latest(raw, bundle, require_fresh=watch)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            message = f"Waiting for RF telemetry: {CSV}"
        if message != last_message:
            print(message, flush=True)
            last_message = message
        if not watch:
            return
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Train or run the PitWall real-car model.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--predict-once", action="store_true", help="Predict from the latest saved packet.")
    mode.add_argument("--watch", action="store_true", help="Show predictions as live CSV rows arrive.")
    args = parser.parse_args()
    if args.predict_once or args.watch:
        try:
            run_inference(args.watch)
        except KeyboardInterrupt:
            print("\nLive inference stopped.")
    else:
        train_model()


if __name__ == "__main__":
    main()
