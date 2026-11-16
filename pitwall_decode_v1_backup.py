import csv
import os
from datetime import datetime, timezone
import numpy as np
from pathlib import Path

FILE = Path(os.environ.get(
    "PITWALL_IQ_FILE",
    r"D:\PitWall Project\dataset\raw\pitwall_capture_new.iq"
))
SAMPLE_RATE = 2_000_000
CHIP_SAMPLES = 1000
HEADER = bytes.fromhex("55 55 55 55 D3 91")
PACKET_SIZE = 19

def crc16(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF \
                if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc

def bits_to_bytes(bits):
    return np.packbits(bits).tobytes()

def decode_packet(packet):
    if len(packet) != PACKET_SIZE:
        return False

    if packet[:6] != HEADER:
        return False

    received_crc = int.from_bytes(packet[17:19], "little")
    expected_crc = crc16(packet[6:17])

    if received_crc != expected_crc:
        return False

    sequence = packet[6]
    timestamp = int.from_bytes(packet[7:11], "little")
    left_rpm = int.from_bytes(packet[11:13], "little")
    right_rpm = int.from_bytes(packet[13:15], "little")
    throttle = packet[15]
    flags = packet[16]

    print("\n" + "=" * 42)
    print("PITWALL PACKET DECODED!")
    print("=" * 42)
    print("Sequence:   ", sequence)
    print("Timestamp:  ", timestamp, "ms")
    print("Left RPM:   ", left_rpm)
    print("Right RPM:  ", right_rpm)
    print("Throttle:   ", str(throttle) + "%")
    print("Flags:      ", hex(flags))
    print("Lap event:  ", bool(flags & 0x04))
    print("CRC:         VALID")
    return True

if not FILE.exists():
    raise SystemExit("Capture file not found!")

raw = np.memmap(FILE, dtype=np.int8, mode="r")

# Search the previously detected burst intervals.
import runpy

bursts = runpy.run_path(
    str(Path(__file__).resolve().with_name("pitwall_find_bursts.py"))
)["bursts"]

print(f"Automatically detected {len(bursts)} bursts")

header_bits = np.unpackbits(
    np.frombuffer(HEADER, dtype=np.uint8)
)

found = set()
CSV_FILE = Path(__file__).resolve().parent / "processed" / "pitwall_live_rf.csv"
CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

CSV_COLUMNS = [
    "received_utc", "timestamp_ms", "seq",
    "left_rpm", "right_rpm", "throttle_pct",
    "flags", "lap_event", "crc_ok", "packet_hex"
]

existing = set()

if CSV_FILE.exists() and CSV_FILE.stat().st_size:
    with CSV_FILE.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            existing.add((row["timestamp_ms"], row["seq"]))

csv_file = CSV_FILE.open("a", newline="", encoding="utf-8")
writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)

if csv_file.tell() == 0:
    writer.writeheader()

def save_packet(packet):
    timestamp = int.from_bytes(packet[7:11], "little")
    seq = packet[6]
    key = (str(timestamp), str(seq))

    if key in existing:
        return

    flags = packet[16]

    writer.writerow({
        "received_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_ms": timestamp,
        "seq": seq,
        "left_rpm": int.from_bytes(packet[11:13], "little"),
        "right_rpm": int.from_bytes(packet[13:15], "little"),
        "throttle_pct": packet[15],
        "flags": flags,
        "lap_event": bool(flags & 0x04),
        "crc_ok": True,
        "packet_hex": packet.hex().upper()
    })

    csv_file.flush()
    existing.add(key)

print("PITWALL MANCHESTER DECODER")
print("Searching for valid CRC-checked packets...")

for burst_index, (start, end) in enumerate(bursts, 1):
    print(f"Checking burst {burst_index}...")

    # Include margins around each detected burst.
    start_sample = max(0, int((start - 0.012) * SAMPLE_RATE))
    end_sample = min(
        len(raw) // 2,
        int((end + 0.012) * SAMPLE_RATE)
    )

    iq = np.asarray(
        raw[2 * start_sample:2 * end_sample],
        dtype=np.float32
    ).reshape(-1, 2)

    # OOK envelope
    envelope = np.hypot(iq[:, 0], iq[:, 1])

    # Smooth short noise fluctuations.
    window = 100
    cumulative = np.concatenate((
        [0.0],
        np.cumsum(envelope, dtype=np.float64)
    ))

    smooth = (
        cumulative[window:] - cumulative[:-window]
    ) / window

    low = np.percentile(smooth, 10)
    high = np.percentile(smooth, 90)
    threshold = (low + high) / 2

    digital = (smooth > threshold).astype(np.uint8)

    # Try different Manchester chip alignments.
    for phase in range(0, CHIP_SAMPLES, 25):
        available = len(digital) - phase
        chips_count = available // CHIP_SAMPLES

        if chips_count < 304:
            continue

        section = digital[
            phase:phase + chips_count * CHIP_SAMPLES
        ]

        chips = (
            section.reshape(chips_count, CHIP_SAMPLES)
            .mean(axis=1) > 0.5
        ).astype(np.uint8)

        # Try both half-bit pair alignments.
        for shift in (0, 1):
            paired = chips[shift:]
            paired = paired[:len(paired) // 2 * 2]

            first = paired[0::2]
            second = paired[1::2]

            # Manchester transitions:
            # 0 = HIGH LOW, 1 = LOW HIGH
            valid = first != second
            bits = second.copy()

            # Try normal and inverted RF polarity.
            for invert in (False, True):
                candidate = bits ^ int(invert)

                for pos in range(
                    len(candidate) - PACKET_SIZE * 8 + 1
                ):
                    if not np.array_equal(
                        candidate[pos:pos + 48],
                        header_bits
                    ):
                        continue

                    if not np.all(
                        valid[pos:pos + PACKET_SIZE * 8]
                    ):
                        continue

                    packet = bits_to_bytes(
                        candidate[pos:pos + PACKET_SIZE * 8]
                    )

                    if packet in found:
                        continue

                    if decode_packet(packet):
                        found.add(packet)
                        save_packet(packet)

print("\n" + "=" * 42)
print("Unique valid packets:", len(found))

if not found:
    print("No CRC-valid packets found yet.")
    print("Next step: inspect timing, polarity,")
    print("frequency offset, and signal envelope.")

csv_file.close()
print("CSV saved:", CSV_FILE)
