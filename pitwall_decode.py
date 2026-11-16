
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
CHIP_SAMPLES = 1000

HEADER = bytes.fromhex("55 55 55 55 D3 91")
HEADER_BITS = np.unpackbits(
    np.frombuffer(HEADER, dtype=np.uint8)
)

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

    diagnostics = {
        "header_matches": 0,
        "manchester_pass": 0,
        "crc_pass": 0,
        "already_in_csv": 0,
    }

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

            window = 100

            cumulative = np.concatenate((
                [0.0],
                np.cumsum(
                    envelope, dtype=np.float64
                ),
            ))

            smooth = (
                cumulative[window:]
                - cumulative[:-window]
            ) / window

            low = np.percentile(smooth, 5)
            high = np.percentile(smooth, 95)

            threshold = low + 0.35 * (high - low)

            digital = (
                smooth > threshold
            ).astype(np.uint8)
            
            best_valid_pct = 0.0
            best_phase = None
            best_shift = None
            for phase in range(0, CHIP_SAMPLES, 25):
                count = (
                    len(digital) - phase
                ) // CHIP_SAMPLES

                if count < 19 * 16:
                    continue

                section = digital[
                    phase:
                    phase + count * CHIP_SAMPLES
                ]

                chips = (
                    section.reshape(
                        count, CHIP_SAMPLES
                    ).mean(axis=1) > 0.5
                ).astype(np.uint8)
                
                if phase == 0:
                    print(
                        f"  Burst {index}: "
                        f"{len(chips)} chips, "
                        f"first 32={''.join(map(str, chips[:32]))}"
                    )
                for shift in (0, 1):
                    paired = chips[shift:]
                    paired = paired[
                        :len(paired) // 2 * 2
                    ]

                    first = paired[0::2]
                    second = paired[1::2]

                    valid = first != second
                    valid_pct = 100.0 * np.mean(valid)

                    if valid_pct > best_valid_pct:
                        best_valid_pct = valid_pct
                        best_phase = phase
                        best_shift = shift                    
                    if phase == 0:
                        valid_pct = 100 * np.mean(valid)
                        print(
                            f"  Shift {shift}: "
                            f"Manchester-valid pairs = "
                            f"{valid_pct:.1f}%"
                        )
                    for invert in (False, True):
                        bits = (
                            second ^ int(invert)
                        )

                        # Check both packet lengths.
                        for packet_size in (33, 19):
                            bit_count = packet_size * 8

                            limit = (
                                len(bits) - bit_count + 1
                            )

                            if limit <= 0:
                                continue

                            for pos in range(limit):
                                if not np.array_equal(
                                    bits[pos:pos + 48],
                                    HEADER_BITS,
                                ):
                                    continue
                                diagnostics["header_matches"] += 1
                                stop = pos + bit_count

                                if not np.all(
                                    valid[pos:stop]
                                ):
                                    continue

                                diagnostics["manchester_pass"] += 1
                                packet = np.packbits(
                                    bits[pos:stop]
                                ).tobytes()

                                if packet in found:
                                    continue

                                row = parse_packet(packet)

                                if row is None:
                                    continue

                                diagnostics["crc_pass"] += 1
                                found.add(packet)

                                if stores[packet_size].save(row):
                                    counts[packet_size] += 1
                                else:
                                    diagnostics["already_in_csv"] += 1

                                    print(
                                        f"CRC PASS | v"
                                        f"{2 if packet_size == 33 else 1}"
                                        f" | seq={row['seq']}"
                                        f" | L={row['left_rpm']}"
                                        f" | R={row['right_rpm']}"
                                    )
                print(
                f"  BEST Manchester alignment: "
                f"{best_valid_pct:.1f}% "
                f"(phase={best_phase}, shift={best_shift})"
            )
    finally:
        for store in stores.values():
            store.close()

    print("\nNew v1 packets:", counts[19])
    print("New v2 packets:", counts[33])
    print("v2 CSV:", PROCESSED / "pitwall_telemetry_v2.csv")

    print("\n========== DECODER DIAGNOSTICS ==========")
    for name, value in diagnostics.items():
        print(f"{name}: {value}")

if __name__ == "__main__":
    main()
