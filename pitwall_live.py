
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
DECODER = PROJECT / "pitwall_decode.py"

CAPTURE_DIR = PROJECT / "dataset" / "raw" / "live_captures"
PROCESSED_DIR = PROJECT / "processed"
STOP_FILE = PROJECT / ".pitwall_stop_requested"

SAMPLE_RATE = 2_000_000
CAPTURE_SECONDS = 3
EXPECTED_BYTES = SAMPLE_RATE * CAPTURE_SECONDS * 2

# Only these files are cleared by the clear command.
CSV_FILES = [
    PROCESSED_DIR / "pitwall_live_rf.csv",
    PROCESSED_DIR / "pitwall_telemetry_v2.csv",
]


def recording_files():
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(CAPTURE_DIR.glob("*.iq"))


def show_status(recording=False):
    files = recording_files()
    complete = sum(
        f.stat().st_size == EXPECTED_BYTES for f in files
    )
    total_mb = sum(f.stat().st_size for f in files) / 1_000_000

    print("\n========== PITWALL STATUS ==========")
    print("Receiver:", "RECORDING" if recording else "STOPPED")
    print("IQ files:", len(files))
    print("Complete captures:", complete)
    print("IQ storage: %.2f MB" % total_mb)

    for csv_file in CSV_FILES:
        if csv_file.exists():
            print("Telemetry CSV:", csv_file.name)
    print("====================================\n")


def record_worker():
    hackrf = shutil.which("hackrf_transfer")

    if not hackrf:
        print("ERROR: hackrf_transfer not found.", flush=True)
        return 1

    if not DECODER.exists():
        print("ERROR: Decoder not found:", DECODER, flush=True)
        return 1

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    batch = 0

    print("HackRF recording started.", flush=True)

    while not STOP_FILE.exists():
        batch += 1

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        iq_file = (
            CAPTURE_DIR /
            f"pitwall_{timestamp}_{batch:06d}.iq"
        )

        # Reserve the filename exclusively.
        try:
            with iq_file.open("xb"):
                pass
        except FileExistsError:
            continue

        print("\nRecording:", iq_file.name, flush=True)

        command = [
            hackrf,
            "-r", str(iq_file),
            "-f", "433920000",
            "-s", str(SAMPLE_RATE),
            "-a", "0",
            "-l", "16",
            "-g", "20",
            "-n", str(SAMPLE_RATE * CAPTURE_SECONDS),
        ]

        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            print("Capture error:", exc, flush=True)
            break

        if result.returncode != 0:
            print("Capture failed; recording retained.", flush=True)
            if not STOP_FILE.exists():
                time.sleep(2)
            continue

        if iq_file.stat().st_size != EXPECTED_BYTES:
            print("Incomplete IQ file retained.", flush=True)
            continue

        print("IQ file saved successfully.", flush=True)

        # Finish the current capture and decoding before stopping.
        env = os.environ.copy()
        env["PITWALL_IQ_FILE"] = str(iq_file)

        result = subprocess.run(
            [sys.executable, str(DECODER)],
            cwd=str(PROJECT),
            env=env,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if result.returncode != 0:
            print("Decoder error; IQ file preserved.", flush=True)

    print("\nRecording stopped safely.", flush=True)
    return 0


def clear_session():
    print("\nThis will permanently delete:")
    print("  - All .iq files inside:", CAPTURE_DIR)
    print("  - PitWall live telemetry CSV files in:", PROCESSED_DIR)
    print("Other raw datasets and project files will NOT be touched.")

    confirmation = input(
        "\nType DELETE to confirm, or anything else to cancel: "
    ).strip()

    if confirmation != "DELETE":
        print("Clear cancelled.")
        return

    deleted = 0

    for file in recording_files():
        try:
            file.unlink()
            deleted += 1
        except OSError as exc:
            print("Could not delete:", file, exc)

    for file in CSV_FILES:
        if file.exists():
            try:
                file.unlink()
                print("Deleted:", file.name)
            except OSError as exc:
                print("Could not delete:", file, exc)

    print(f"Clear complete. Deleted {deleted} IQ recordings.")


def main():
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    worker = None

    print("\n" + "=" * 50)
    print("PITWALL - RF RECORDING CONTROL")
    print("REAL HACKRF DATA ONLY")
    print("=" * 50)
    print("Commands: start | stop | status | clear | exit")
    print("Recordings:", CAPTURE_DIR)

    while True:
        try:
            command = input("\nPitWall > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            command = "exit"
            print()

        recording = worker is not None and worker.poll() is None

        if command == "start":
            if recording:
                print("Already recording.")
                continue

            if not shutil.which("hackrf_transfer"):
                print("HackRF command not found in this terminal.")
                continue

            if not DECODER.exists():
                print("Decoder not found:", DECODER)
                continue

            STOP_FILE.unlink(missing_ok=True)

            worker = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker"],
                cwd=str(PROJECT),
            )
            print("Recording process launched.")

        elif command == "stop":
            if not recording:
                print("Receiver is not recording.")
                continue

            STOP_FILE.touch()
            print("Finishing current capture and decoding...")
            worker.wait()
            worker = None
            STOP_FILE.unlink(missing_ok=True)
            print("Receiver stopped. All IQ files preserved.")

        elif command == "status":
            show_status(recording)

        elif command == "clear":
            if recording:
                print("STOP recording before clearing data.")
            else:
                clear_session()

        elif command == "exit":
            if recording:
                STOP_FILE.touch()
                print("Stopping receiver safely...")
                worker.wait()
                STOP_FILE.unlink(missing_ok=True)

            print("PitWall closed. See you on track!")
            break

        elif command:
            print("Unknown command.")
            print("Available: start, stop, status, clear, exit")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.exit(record_worker())
    main()
