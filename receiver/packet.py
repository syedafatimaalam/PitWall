"""PitWall 19-byte telemetry packet encoder and validator."""

PREAMBLE = bytes([0x55] * 4)
SYNC = bytes([0xD3, 0x91])


def crc16_ccitt_false(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def make_packet(sequence: int, timestamp: int, left_rpm: int,
                right_rpm: int, throttle: int, flags: int) -> bytes:
    """Construct 19-byte PitWall frame using the agreed field layout."""
    if not 0 <= sequence <= 255:
        raise ValueError("sequence must be 0..255")
    if not 0 <= timestamp <= 0xFFFFFFFF:
        raise ValueError("timestamp must be 0..4294967295")
    for name, value in (("left_rpm", left_rpm), ("right_rpm", right_rpm)):
        if not 0 <= value <= 65535:
            raise ValueError(f"{name} must be 0..65535")
    if not 0 <= throttle <= 100:
        raise ValueError("throttle must be 0..100")
    if not 0 <= flags <= 255:
        raise ValueError("flags must be 0..255")

    payload = (
        bytes([sequence])
        + timestamp.to_bytes(4, "little")
        + left_rpm.to_bytes(2, "little")
        + right_rpm.to_bytes(2, "little")
        + bytes([throttle, flags])
    )
    crc = crc16_ccitt_false(payload)
    packet = PREAMBLE + SYNC + payload + crc.to_bytes(2, "big")
    assert len(packet) == 19
    return packet


def check_packet(packet: bytes) -> bool:
    """Check frame length, preamble, sync, and payload CRC."""
    return (
        len(packet) == 19
        and packet[:4] == PREAMBLE
        and packet[4:6] == SYNC
        and crc16_ccitt_false(packet[6:17]) == int.from_bytes(packet[17:19], "big")
    )


if __name__ == "__main__":
    packet = make_packet(
        sequence=2,
        timestamp=2000,
        left_rpm=150,
        right_rpm=145,
        throttle=80,
        flags=0b00000100,
    )
    print("Packet (hex):", packet.hex(" ").upper())
    print("Packet length:", len(packet), "bytes")
    print("CRC (hex):", packet[-2:].hex().upper())
    print("CRC valid:", check_packet(packet))
    assert crc16_ccitt_false(b"123456789") == 0x29B1
