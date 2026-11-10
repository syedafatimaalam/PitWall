
"""Decode PitWall Manchester/OOK telemetry from GQRX IQ recordings."""

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

OUTPUT_RATE = 20000
CHIP_US = 500
CHIP_SAMPLES = round(OUTPUT_RATE * CHIP_US / 1_000_000)

PACKET_SIZE = 19
PACKET_BITS = PACKET_SIZE * 8
PACKET_CHIPS = PACKET_BITS * 2

HEADER = bytes.fromhex("55 55 55 55 D3 91")


# -------- CRC-16/CCITT-FALSE --------

def crc16(data):
    crc = 0xFFFF

    for value in data:
        crc ^= value << 8

        for _ in range(8):
            crc = (
                (crc << 1) ^
                (0x1021 if crc & 0x8000 else 0)
            ) & 0xFFFF

    return crc


# -------- RECORDING INFORMATION --------

def recording_settings(path, rate_override, offset_override):
    match = re.search(
        r"_(\d{9,10})_(\d{6,9})_fc\.raw$",
        path.name,
        re.I
    )

    center = int(match.group(1)) if match else None

    rate = rate_override or (
        int(match.group(2)) if match else 2_000_000
    )

    offset = (
        offset_override
        if offset_override is not None
        else (433_920_000 - center if center is not None else 0.0)
    )

    return center, rate, offset


# -------- EXTRACT OOK ENVELOPE --------

def extract_envelope(path, sample_rate, offset,
                     chunk_size=400_000):

    if sample_rate % OUTPUT_RATE:
        raise ValueError(
            "Sample rate must be divisible by 20000"
        )

    raw = np.memmap(path, dtype="<f4", mode="r")

    if len(raw) % 2:
        raise ValueError(
            "Invalid I/Q file: odd number of float32 values"
        )

    total = len(raw) // 2
    decimation = sample_rate // OUTPUT_RATE
    usable = total // decimation * decimation

    result = np.empty(
        usable // decimation,
        dtype=np.float32
    )

    chunk_size = max(
        decimation,
        chunk_size // decimation * decimation
    )

    cursor = 0

    for start in range(0, usable, chunk_size):
        stop = min(start + chunk_size, usable)

        iq = np.asarray(
            raw[2 * start:2 * stop]
        ).view(np.complex64)

        indices = np.arange(
            stop - start,
            dtype=np.float64
        )

        rotation = np.exp(
            -2j * np.pi * offset *
            indices / sample_rate
        ).astype(np.complex64)

        mixed = iq * rotation

        # Average complex samples before envelope detection.
        envelope = np.abs(
            mixed.reshape(-1, decimation).mean(axis=1)
        )

        result[cursor:cursor + len(envelope)] = envelope
        cursor += len(envelope)

    return result


# -------- DETECT RF BURSTS --------

def detect_bursts(envelope, fraction):

    smooth = uniform_filter1d(
        envelope,
        size=120
    )

    floor = float(np.percentile(smooth, 25))
    high = float(np.percentile(smooth, 99.7))

    threshold = floor + fraction * (high - floor)

    active = smooth > threshold

    starts = np.flatnonzero(
        active & ~np.r_[False, active[:-1]]
    )

    ends = np.flatnonzero(
        active & ~np.r_[active[1:], False]
    )

    bursts = [
        (int(s), int(e))
        for s, e in zip(starts, ends)
        if 0.06 * OUTPUT_RATE <= (e - s)
        <= 0.35 * OUTPUT_RATE
    ]

    return bursts, floor, high, threshold


# -------- MANCHESTER PACKET DECODER --------

def decode_burst(envelope, burst_start):

    first = max(
        0,
        burst_start - int(0.020 * OUTPUT_RATE)
    )

    last = min(
        burst_start + int(0.020 * OUTPUT_RATE),
        len(envelope) -
        (PACKET_CHIPS + 12) * CHIP_SAMPLES - 1
    )

    if last < first:
        return None

    segment = envelope[
        first:last +
        (PACKET_CHIPS + 12) * CHIP_SAMPLES + 1
    ]

    cumulative = np.r_[
        0.0,
        np.cumsum(segment, dtype=np.float64)
    ]

    chip_offsets = (
        np.arange(PACKET_CHIPS + 12) *
        CHIP_SAMPLES
    )

    for start in range(first, last + 1):

        offsets = (start - first) + chip_offsets

        chips = (
            cumulative[offsets + CHIP_SAMPLES] -
            cumulative[offsets]
        ) / CHIP_SAMPLES

        raw_bits = (
            chips[0::2] > chips[1::2]
        ).astype(np.uint8)

        for inverted in (False, True):

            bits = (
                1 - raw_bits
                if inverted else raw_bits
            )

            for bit_start in range(
                len(bits) - PACKET_BITS + 1
            ):

                header = np.packbits(
                    bits[bit_start:bit_start + 48],
                    bitorder="big"
                ).tobytes()

                if header != HEADER:
                    continue

                packet = np.packbits(
                    bits[
                        bit_start:
                        bit_start + PACKET_BITS
                    ],
                    bitorder="big"
                ).tobytes()

                received_crc = int.from_bytes(
                    packet[17:19],
                    "little"
                )

                expected_crc = crc16(
                    packet[6:17]
                )

                if expected_crc == received_crc:

                    packet_start = (
                        start +
                        2 * bit_start * CHIP_SAMPLES
                    ) / OUTPUT_RATE

                    return (
                        packet,
                        packet_start,
                        inverted
                    )

    return None


# -------- MAIN PROGRAM --------

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "file",
        type=Path,
        help="GQRX *_fc.raw recording"
    )

    parser.add_argument(
        "--rate",
        type=int,
        default=None
    )

    parser.add_argument(
        "--offset",
        type=float,
        default=None,
        help="RF carrier minus GQRX center, in Hz"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.22
    )

    args = parser.parse_args()

    if not args.file.is_file():
        parser.error(
            f"Cannot find recording: {args.file}"
        )

    center, sample_rate, offset = recording_settings(
        args.file,
        args.rate,
        args.offset
    )

    print(f"File: {args.file}", flush=True)

    print(
        f"File size: {args.file.stat().st_size:,} bytes",
        flush=True
    )

    print(
        f"Sample rate: {sample_rate:,} samples/sec",
        flush=True
    )

    print(
        f"GQRX center: {center} Hz | "
        f"RF offset: {offset:,.0f} Hz",
        flush=True
    )

    print("Processing IQ samples...", flush=True)

    envelope = extract_envelope(
        args.file,
        sample_rate,
        offset
    )

    print(
        f"Duration: {len(envelope)/OUTPUT_RATE:.2f} seconds",
        flush=True
    )

    bursts, floor, high, threshold = detect_bursts(
        envelope,
        args.threshold
    )

    print(
        f"Envelope baseline={floor:.6g}, "
        f"high={high:.6g}, "
        f"threshold={threshold:.6g}",
        flush=True
    )

    print(
        f"Candidate bursts: {len(bursts)}",
        flush=True
    )

    good = 0

    for number, (start, end) in enumerate(bursts, 1):

        duration = (
            (end - start) * 1000 / OUTPUT_RATE
        )

        print(
            f"Burst {number}: "
            f"{start/OUTPUT_RATE:.3f}s, "
            f"{duration:.1f}ms",
            end="",
            flush=True
        )

        result = decode_burst(
            envelope,
            start
        )

        if result is None:
            print(
                " | no CRC-valid packet",
                flush=True
            )
            continue

        packet, packet_time, inverted = result

        sequence = packet[6]

        timestamp = int.from_bytes(
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

        good += 1

        print(
            f" | CRC PASS"
            f" | seq={sequence}"
            f" | LEFT={left_rpm} RPM"
            f" | RIGHT={right_rpm} RPM"
            f" | timestamp={timestamp}ms"
            f" | at={packet_time:.3f}s"
            f" | inverted={inverted}",
            flush=True
        )

        print(
            "  HEX:",
            packet.hex(" ").upper(),
            flush=True
        )

    print(
        f"\nRESULT: {good} CRC-valid packets "
        f"/ {len(bursts)} candidate bursts",
        flush=True
    )

    if good == 0:
        print(
            "Wheel-speed payload NOT verified yet.",
            flush=True
        )

        print(
            "Try --threshold 0.1 or investigate "
            "carrier offset and chip timing.",
            flush=True
        )


if __name__ == "__main__":
    main()
