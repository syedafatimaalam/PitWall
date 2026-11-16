
import csv
import os
import runpy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent

FILE = Path(os.environ.get(
    "PITWALL_IQ_FILE",
    str(PROJECT / "dataset" / "raw" / "pitwall_capture_new.iq"),
))

SAMPLE_RATE = 2_000_000
CHIP_PERIODS = (1002, 1000, 1001, 1003, 999, 1004, 998, 1005, 997, 1006)

HEADER = bytes.fromhex("55 55 55 55 D3 91")
HEADER_BITS = np.unpackbits(np.frombuffer(HEADER, dtype=np.uint8))

PROCESSED = PROJECT / "processed"
PROCESSED.mkdir(exist_ok=True)

COMMON_COLUMNS = [
    "received_utc", "timestamp_ms", "seq",
    "left_rpm", "right_rpm", "throttle_pct",
    "flags", "lap_event", "crc_ok", "packet_hex",
]

V2_COLUMNS = [
    "received_utc", "timestamp_ms", "seq",
    "left_rpm", "right_rpm", "throttle_pct",
    "flags", "lap_event", "mpu_valid",
    "accel_x_mg", "accel_y_mg", "accel_z_mg",
    "gyro_x_dps10", "gyro_y_dps10", "gyro_z_dps10",
    "lap_count", "crc_ok", "packet_hex",
]


def crc16(data):
    crc = 0xFFFF

    for byte in data:
        crc ^= byte << 8

        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF

    return crc


def u16(packet, offset):
    return int.from_bytes(
        packet[offset:offset + 2], "little"
    )


def i16(packet, offset):
    return int.from_bytes(
        packet[offset:offset + 2],
        "little",
        signed=True,
    )


def parse_packet(packet):
    size = len(packet)

    if size not in (19, 33):
        return None

    if packet[:6] != HEADER:
        return None

    flags = packet[16]

    # Protocol identification prevents a v2 packet
    # from being mistaken for a v1 packet.
    if size == 19 and (flags & 0x80):
        return None

    if size == 33 and not (flags & 0x80):
        return None

    received_crc = u16(packet, size - 2)
    calculated_crc = crc16(packet[6:size - 2])

    if received_crc != calculated_crc:
        return None

    row = {
        "received_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_ms": int.from_bytes(
            packet[7:11], "little"
        ),
        "seq": packet[6],
        "left_rpm": u16(packet, 11),
        "right_rpm": u16(packet, 13),
        "throttle_pct": packet[15],
        "flags": flags,
        "lap_event": bool(flags & 0x04),
        "crc_ok": True,
        "packet_hex": packet.hex().upper(),
    }

    if size == 33:
        valid = bool(flags & 0x40)

        row["mpu_valid"] = valid

        fields = {
            "accel_x_mg": 17,
            "accel_y_mg": 19,
            "accel_z_mg": 21,
            "gyro_x_dps10": 23,
            "gyro_y_dps10": 25,
            "gyro_z_dps10": 27,
        }

        for name, offset in fields.items():
            row[name] = (
                i16(packet, offset) if valid else ""
            )

        row["lap_count"] = u16(packet, 29)

    return row


def load_existing_keys(path):
    keys = set()

    if not path.exists() or path.stat().st_size == 0:
        return keys

    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys.add((
                row.get("timestamp_ms", ""),
                row.get("seq", ""),
            ))

    return keys


class CsvStore:
    def __init__(self, path, columns):
        self.path = path
        self.columns = columns
        self.keys = load_existing_keys(path)

        self.file = path.open(
            "a", newline="", encoding="utf-8"
        )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=columns,
        )

        if self.file.tell() == 0:
            self.writer.writeheader()
            self.file.flush()

    def save(self, row):
        key = (
            str(row["timestamp_ms"]),
            str(row["seq"]),
        )

        if key in self.keys:
            return False

        self.writer.writerow(row)
        self.file.flush()
        self.keys.add(key)
        return True

    def close(self):
        self.file.close()


def decode_burst(envelope):
    # Average a short window near the center of each sampled chip.
    window = 300
    cumulative = np.concatenate(([0.0], np.cumsum(envelope, dtype=np.float64)))
    smooth = (cumulative[window:] - cumulative[:-window]) / window
    for period in CHIP_PERIODS:
        for phase in range(0, period, 25):
            chips = smooth[phase::period]
            for shift in (0, 1):
                pairs = chips[shift:]
                pairs = pairs[:len(pairs) // 2 * 2].reshape(-1, 2)
                if len(pairs) < 19 * 8:
                    continue
                # Compare the two halves of each bit to tolerate envelope variation.
                bits = (pairs[:, 1] > pairs[:, 0]).astype(np.uint8)
                headers = np.lib.stride_tricks.sliding_window_view(bits, len(HEADER_BITS))
                for invert in (0, 1):
                    matches = np.flatnonzero(np.all(headers == (HEADER_BITS ^ invert), axis=1))
                    for pos in matches:
                        for size in (33, 19):
                            stop = pos + size * 8
                            if stop > len(bits):
                                continue
                            packet = np.packbits(bits[pos:stop] ^ invert).tobytes()
                            row = parse_packet(packet)
                            if row is not None:
                                yield size, packet, row


def main():
    if not FILE.exists():
        raise SystemExit(f"IQ file not found: {FILE}")

    print("DECODER IQ FILE:", FILE.resolve())
    raw = np.memmap(
        FILE, dtype=np.int8, mode="r"
    )

    detector = runpy.run_path(
        str(PROJECT / "pitwall_find_bursts.py")
    )

    bursts = detector["bursts"]

    print("Detected RF bursts:", len(bursts))

    stores = {
        19: CsvStore(
            PROCESSED / "pitwall_live_rf.csv",
            COMMON_COLUMNS,
        ),
        33: CsvStore(
            PROCESSED / "pitwall_telemetry_v2.csv",
            V2_COLUMNS,
        ),
    }

    found = set()
    counts = {19: 0, 33: 0}

    already_in_csv = 0

    try:
        for index, (start, end) in enumerate(bursts, 1):
            print(f"Checking burst {index}...")

            start_sample = max(
                0,
                int((start - 0.012) * SAMPLE_RATE),
            )

            end_sample = min(
                len(raw) // 2,
                int((end + 0.012) * SAMPLE_RATE),
            )

            iq = np.asarray(
                raw[
                    start_sample * 2:
                    end_sample * 2
                ],
                dtype=np.float32,
            ).reshape(-1, 2)

            if len(iq) < 100:
                continue

            envelope = np.hypot(
                iq[:, 0], iq[:, 1]
            )

            for size, packet, row in decode_burst(envelope):
                if packet in found:
                    continue
                found.add(packet)
                version = 2 if size == 33 else 1
                print(f"CRC PASS | v{version} | seq={row['seq']} "
                      f"| L={row['left_rpm']} | R={row['right_rpm']}")
                if stores[size].save(row):
                    counts[size] += 1
                else:
                    already_in_csv += 1
    finally:
        for store in stores.values():
            store.close()

    print("\nNew v1 packets:", counts[19])
    print("New v2 packets:", counts[33])
    print("v1 CSV:", PROCESSED / "pitwall_live_rf.csv")
    print("v2 CSV:", PROCESSED / "pitwall_telemetry_v2.csv")
    print("CRC-valid packets:", len(found))
    print("Already in CSV:", already_in_csv)


if __name__ == "__main__":
    main()
