"""PitWall future driving-behavior demo (synthetic data, no hardware required).

Predict the car's movement state 0.5 seconds ahead from recent telemetry.
Run: python pitwall_future_behavior.py
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SAMPLE_PERIOD = 0.1  # 10 Hz telemetry
HORIZON = 5          # 5 samples = 0.5 seconds
FEATURES = [
    "left_rpm", "right_rpm", "throttle_pct", "yaw_dps",
    "avg_rpm", "rpm_1_ago", "rpm_3_ago",
    "throttle_1_ago", "yaw_1_ago",
]
LABELS = ["ACCELERATING", "BRAKING", "TURNING", "STEADY"]


def simulate_driving(runs=12, samples_per_run=600):
    """Create independent runs with piecewise driver commands and motor inertia."""
    rng = np.random.default_rng(204)
    rows = []
    for run_id in range(1, runs + 1):
        left = right = 0.0
        throttle = 0.0
        steer = 0.0
        hold = 0
        for i in range(samples_per_run):
            if hold <= 0:
                mode = rng.choice(["accelerate", "brake", "turn", "cruise"])
                if mode == "accelerate":
                    throttle = rng.uniform(65, 95)
                    steer = rng.uniform(-0.1, 0.1)
                elif mode == "brake":
                    throttle = rng.uniform(0, 12)
                    steer = rng.uniform(-0.1, 0.1)
                elif mode == "turn":
                    throttle = rng.uniform(35, 65)
                    steer = rng.choice([-1, 1]) * rng.uniform(0.6, 1)
                else:
                    throttle = rng.uniform(25, 55)
                    steer = rng.uniform(-0.1, 0.1)
                hold = int(rng.integers(15, 36))  # command lasts 1.5-3.5s
            hold -= 1
            left += 0.18 * (max(0, throttle * 1.7 - 12 * steer) - left)
            right += 0.18 * (max(0, throttle * 1.7 + 12 * steer) - right)
            rows.append({
                "run_id": run_id,
                "timestamp_ms": round(i * SAMPLE_PERIOD * 1000),
                "left_rpm": max(0, left + rng.normal(0, 1)),
                "right_rpm": max(0, right + rng.normal(0, 1)),
                "throttle_pct": throttle,
                "yaw_dps": steer * 55 + rng.normal(0, 2),
            })
    return pd.DataFrame(rows)


def prepare_examples(data):
    """Use only present/past features; future readings create training labels."""
    df = data.sort_values(["run_id", "timestamp_ms"]).copy()
    df["avg_rpm"] = (df.left_rpm + df.right_rpm) / 2
    group = df.groupby("run_id", sort=False)
    df["rpm_1_ago"] = group.avg_rpm.shift(1)
    df["rpm_3_ago"] = group.avg_rpm.shift(3)
    df["throttle_1_ago"] = group.throttle_pct.shift(1)
    df["yaw_1_ago"] = group.yaw_dps.shift(1)

    # Label what the vehicle will do at t+0.5s, not what it does now.
    future_rpm = group.avg_rpm.shift(-HORIZON)
    just_before_future = group.avg_rpm.shift(-(HORIZON - 1))
    future_yaw = group.yaw_dps.shift(-HORIZON)
    future_change = future_rpm - just_before_future
    # Categories are an operational definition for this demo.
    df["future_behavior"] = np.select(
        [future_yaw.abs() > 25, future_change > 2.0, future_change < -2.0],
        ["TURNING", "ACCELERATING", "BRAKING"],
        default="STEADY",
    )
    df["future_avg_rpm"] = future_rpm
    return df.dropna(subset=FEATURES + ["future_avg_rpm"]).copy()


def main():
    raw = simulate_driving()
    raw.to_csv("pitwall_synthetic_driving.csv", index=False)
    data = prepare_examples(raw)

    # Entire runs 1-9 for training; entire runs 10-12 for testing.
    train = data[data.run_id <= 9]
    test = data[data.run_id > 9]
    model = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    model.fit(train[FEATURES], train.future_behavior)
    predicted = model.predict(test[FEATURES])

    print("TRAINING LABEL COUNTS:\n", train.future_behavior.value_counts())
    print("\nTEST ACCURACY:", round(accuracy_score(test.future_behavior, predicted), 3))
    print("\nPER-CLASS RESULTS:\n", classification_report(
        test.future_behavior, predicted, labels=LABELS, zero_division=0))
    print("CONFUSION MATRIX (rows=actual, columns=predicted):")
    print(pd.DataFrame(confusion_matrix(test.future_behavior, predicted, labels=LABELS),
                       index=LABELS, columns=LABELS))

    # Sample inference using a held-out run's current/past readings only.
    sample = test.iloc[100]
    input_row = pd.DataFrame([{name: sample[name] for name in FEATURES}])
    probabilities = model.predict_proba(input_row)[0]
    print("\nEXAMPLE AT", round(sample.timestamp_ms / 1000, 1), "seconds:")
    print("Predicted behavior 0.5s ahead:", model.predict(input_row)[0])
    print("Actual future behavior:", sample.future_behavior)
    print("Class probabilities:", {name: round(float(prob), 3)
                                   for name, prob in zip(model.classes_, probabilities)})

    joblib.dump({"model": model, "features": FEATURES, "sample_period": SAMPLE_PERIOD,
                 "horizon_samples": HORIZON}, "pitwall_behavior_model.pkl")
    print("\nSaved pitwall_behavior_model.pkl and pitwall_synthetic_driving.csv")


if __name__ == "__main__":
    main()
