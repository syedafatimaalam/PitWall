
from pathlib import Path
import os
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parent
CAPTURES = PROJECT / "dataset" / "raw" / "live_captures"
DECODER = PROJECT / "pitwall_decode.py"

POLL_SECONDS = 1
STABLE_SECONDS = 2

# Files successfully processed during this watcher session.
processed = set()

print("PITWALL AUTOMATIC RF DECODER")
print("Watching:", CAPTURES)
print("Waiting for new completed IQ recordings...")
print("Press Ctrl+C to stop.\n")

try:
    while True:
        files = sorted(
            CAPTURES.glob("*.iq"),
            key=lambda p: (p.stat().st_mtime_ns, p.name)
        )

        # Start with the newest existing recording.
        # Older recordings are not replayed automatically.
        if not processed and files:
            processed.update(files[:-1])

        for iq_file in files:
            if iq_file in processed:
                continue

            try:
                stat1 = iq_file.stat()

                if stat1.st_size == 0:
                    continue

                if time.time() - stat1.st_mtime < STABLE_SECONDS:
                    continue

                time.sleep(0.3)
                stat2 = iq_file.stat()

                # Do not decode a file still being written.
                if stat1.st_size != stat2.st_size:
                    continue

                print(f"\nDecoding: {iq_file.name}", flush=True)

                env = os.environ.copy()
                env["PITWALL_IQ_FILE"] = str(iq_file)

                result = subprocess.run(
                    [sys.executable, str(DECODER)],
                    cwd=str(PROJECT),
                    env=env,
                    check=False,
                )

                if result.returncode == 0:
                    processed.add(iq_file)
                    print("Decode completed.", flush=True)
                else:
                    print(
                        f"Decoder exited with code {result.returncode}; "
                        "will retry.",
                        flush=True
                    )

            except (OSError, PermissionError) as exc:
                print("Waiting for file:", exc, flush=True)

        time.sleep(POLL_SECONDS)

except KeyboardInterrupt:
    print("\nAutomatic decoder stopped.")
