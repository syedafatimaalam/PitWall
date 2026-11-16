
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
RAW_DIR = PROJECT / "dataset" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = PROJECT / "processed"
SAMPLE_RATE = 2_000_000
CAPTURE_SECONDS = 60

def main():
    hackrf = shutil.which("hackrf_transfer")

    if not hackrf:
        raise SystemExit(
            "hackrf_transfer not found. Run this from "
            "the Anaconda Prompt where hackrf_info works."
        )

    filename = RAW_DIR / (
        "pitwall_" +
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") +
        ".iq"
    )

    print("\n========== PITWALL RECORDING ==========")
    print("Saving IQ to:", filename)

    command = [
        hackrf,
        "-r", str(filename),
        "-f", "433920000",
        "-s", str(SAMPLE_RATE),
        "-a", "0",
        "-l", "16",
        "-g", "20",
        "-n", str(SAMPLE_RATE * CAPTURE_SECONDS),
    ]

    subprocess.run(command, check=True)

    expected_size = SAMPLE_RATE * CAPTURE_SECONDS * 2

    if not filename.exists():
        raise RuntimeError("HackRF did not create a recording.")

    if filename.stat().st_size != expected_size:
        raise RuntimeError(
            "Recording size mismatch. "
            "Refusing to decode an incomplete capture."
        )

    print("\nRecording complete!")
    print("Starting automatic detection and decoding...")

    env = os.environ.copy()
    env["PITWALL_IQ_FILE"] = str(filename)

    subprocess.run(
        [sys.executable, str(PROJECT / "pitwall_decode.py")],
        cwd=str(PROJECT),
        env=env,
        check=True
    )

    print("\n========== PITWALL COMPLETE ==========")
    print("IQ recording:", filename)
    print("CSV:", PROJECT / "processed" / "pitwall_live_rf.csv")
    print("Dashboard data is ready.")

if __name__ == "__main__":
    main()
