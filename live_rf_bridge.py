
"""
PitWall - Live RF Telemetry Bridge

ESP32 -> FS1000A -> HackRF -> Python -> CSV

Captures short RF batches, decodes Manchester packets,
validates CRC-16, and saves real wheel-speed readings.

Requires:
    numpy
    scipy
    hackrf_transfer
    decode_right_wheel_gqrx.py
"""

import csv
import subprocess
import tempfile
import time

from pathlib import Path
from datetime import datetime

import decode_right_wheel_gqrx as decoder


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

CSV_FILE = PROJECT_DIR / "pitwall_live_rf.csv"

CENTER_FREQ = 433_865_800
TRANSMITTER_FREQ = 433_920_000

SAMPLE_RATE = 2_000_000

CAPTURE_SECONDS = 6

LNA_GAIN = 16
VGA_GAIN = 20

RF_OFFSET = TRANSMITTER_FREQ - CENTER_FREQ

THRESHOLD = 0.22


# ==========================================
# CSV CONFIGURATION
# ==========================================

CSV_COLUMNS = [
    "received_at",
    "sequence",
    "timestamp_ms",
    "left_rpm",
    "right_rpm",
    "throttle",
    "flags",
    "crc_ok",
    "packet_hex"
]


def initialize_csv():

    if CSV_FILE.exists() and CSV_FILE.stat().st_size > 0:
        return

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )

        writer.writeheader()

    print("CSV initialized:", CSV_FILE, flush=True)


# ==========================================
# HACKRF CAPTURE
# ==========================================

def capture_rf(filename):

    total_samples = (
        SAMPLE_RATE * CAPTURE_SECONDS
    )

    command = [
    r"C:\Users\ce\radioconda\Library\bin\hackrf_transfer.exe",
    "-r", str(filename),
    "-f", str(CENTER_FREQ),
    "-s", str(SAMPLE_RATE),
    "-n", str(total_samples),
    "-l", str(LNA_GAIN),
    "-g", str(VGA_GAIN),
    "-a", "0"
    ]

    print(
        f"\nRecording {CAPTURE_SECONDS} seconds "
        "of RF...",
        flush=True
    )

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("HackRF capture failed:", flush=True)
        print(result.stderr, flush=True)
        return False

    if not filename.exists():
        print("No IQ file created.", flush=True)
        return False

    print("RF capture complete.", flush=True)

    return True


# ==========================================
# DECODE RF PACKETS
# ==========================================

def decode_capture(filename):

    print("Decoding RF packets...", flush=True)

    try:
        envelope = decoder.extract_envelope(
            filename,
            SAMPLE_RATE,
            RF_OFFSET
        )

        bursts, floor, high, threshold = (
            decoder.detect_bursts(
                envelope,
                THRESHOLD
            )
        )

        print(
            f"Detected {len(bursts)} candidate bursts",
            flush=True
        )

        valid_packets = []
        seen = set()

        for start, end in bursts:

            result = decoder.decode_burst(
                envelope,
                start
            )

            if result is None:
                continue

            packet, packet_time, inverted = result

            if len(packet) != 19:
                continue

            # Validate header
            if packet[:6] != decoder.HEADER:
                continue

            # Validate CRC again
            expected_crc = decoder.crc16(
                packet[6:17]
            )

            received_crc = int.from_bytes(
                packet[17:19],
                "little"
            )

            if expected_crc != received_crc:
                continue

            sequence = packet[6]

            timestamp_ms = int.from_bytes(
                packet[7:11],
                "little"
            )

            left_rpm = int.from_bytes(
                packet[11:13],
                "little"
            )

            right_rpm = int.from_bytes(
                packet[13:15],
                "little"
            )

            throttle = packet[15]
            flags = packet[16]

            packet_key = (
                sequence,
                timestamp_ms
            )

            if packet_key in seen:
                continue

            seen.add(packet_key)

            received_packet = {
                "sequence": sequence,
                "timestamp_ms": timestamp_ms,
                "left_rpm": left_rpm,
                "right_rpm": right_rpm,
                "throttle": throttle,
                "flags": flags,
                "crc_ok": True,
                "packet_hex": packet.hex(" ").upper()
            }

            valid_packets.append(
                received_packet
            )

            print(
                f"CRC PASS | "
                f"seq={sequence} | "
                f"LEFT={left_rpm} RPM | "
                f"RIGHT={right_rpm} RPM",
                flush=True
            )

        print(
            f"Valid packets: {len(valid_packets)}",
            flush=True
        )

        return valid_packets

    except Exception as error:

        print(
            "Decoder error:",
            error,
            flush=True
        )

        return []


# ==========================================
# SAVE PACKETS TO CSV
# ==========================================

def save_packets(packets):

    if not packets:
        return

    # Reuse one receive timestamp for this batch.
    # This is the decode time, not the exact RF arrival time.
    received_at = datetime.now().isoformat(
        timespec="milliseconds"
    )

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )

        for packet in packets:

            row = {
                "received_at": received_at,
                "sequence": packet["sequence"],
                "timestamp_ms": packet["timestamp_ms"],
                "left_rpm": packet["left_rpm"],
                "right_rpm": packet["right_rpm"],
                "throttle": packet["throttle"],
                "flags": packet["flags"],
                "crc_ok": packet["crc_ok"],
                "packet_hex": packet["packet_hex"]
            }

            writer.writerow(row)

    print(
        f"Saved {len(packets)} packets to CSV.",
        flush=True
    )


# ==========================================
# MAIN LOOP
# ==========================================

def main():

    print("=" * 50)
    print("       PITWALL LIVE RF RECEIVER")
    print("=" * 50)

    print("Center frequency:", CENTER_FREQ)
    print("Expected transmitter:", TRANSMITTER_FREQ)
    print("Sample rate:", SAMPLE_RATE)
    print("Capture duration:", CAPTURE_SECONDS)

    initialize_csv()

    print("\nWaiting for RF telemetry...")
    print("Press Ctrl+C to stop.\n")

    batch = 0

    try:

        with tempfile.TemporaryDirectory() as temp_dir:

            while True:

                batch += 1

                print(
                    f"\n========== BATCH {batch} ==========",
                    flush=True
                )

                iq_file = (
                    Path(temp_dir) / "capture.iq"
                )

                if not capture_rf(iq_file):

                    print(
                        "Retrying in 2 seconds...",
                        flush=True
                    )

                    time.sleep(2)
                    continue

                # hackrf_transfer saves signed 8-bit IQ,
                # while the working GQRX decoder expects
                # complex float32 IQ.
                #
                # Convert format before decoding.

                import numpy as np

                raw = np.fromfile(
                    iq_file,
                    dtype=np.int8
                )

                if len(raw) % 2:
                    raw = raw[:-1]

                samples = (
                    raw[0::2].astype(np.float32)
                    + 1j * raw[1::2].astype(np.float32)
                ) / 128.0

                converted_file = (
                    Path(temp_dir) / "capture_fc.raw"
                )

                samples.astype(
                    np.complex64
                ).tofile(converted_file)

                # Free raw arrays before decoding.
                del raw
                del samples

                packets = decode_capture(
                    converted_file
                )

                if packets:

                    save_packets(packets)

                else:

                    print(
                        "No CRC-valid packets in this batch.",
                        flush=True
                    )

                print(
                    "Batch finished. Starting next capture...",
                    flush=True
                )

    except KeyboardInterrupt:

        print("\nPitWall RF receiver stopped.")


if __name__ == "__main__":
    main()
