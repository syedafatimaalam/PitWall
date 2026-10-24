# ============================================
# PITWALL - PACKET DECODER
# ============================================

from packet import crc16_ccitt_false


# ============================================
# 1. PACKET CONSTANTS
# ============================================

PREAMBLE = bytes([0x55] * 4)
SYNC = bytes([0xD3, 0x91])

PACKET_LENGTH = 19


# ============================================
# 2. DECODE AND VALIDATE PACKET
# ============================================

def decode_packet(packet):

    # Check packet length
    if len(packet) != PACKET_LENGTH:
        raise ValueError(
            f"Invalid packet length: {len(packet)} bytes"
        )

    # Check preamble
    if packet[:4] != PREAMBLE:
        raise ValueError("Invalid preamble")

    # Check synchronization bytes
    if packet[4:6] != SYNC:
        raise ValueError("Invalid sync bytes")

    # Extract payload (bytes 6 to 16)
    payload = packet[6:17]

    # Extract received CRC (bytes 17 and 18)
    received_crc = int.from_bytes(
        packet[17:19],
        byteorder="big"
    )

    # Independently calculate CRC
    calculated_crc = crc16_ccitt_false(payload)

    # Verify CRC
    if received_crc != calculated_crc:
        raise ValueError(
            f"CRC mismatch! "
            f"Received: 0x{received_crc:04X}, "
            f"Calculated: 0x{calculated_crc:04X}"
        )

    # ========================================
    # 3. EXTRACT TELEMETRY VALUES
    # ========================================

    sequence = packet[6]

    timestamp = int.from_bytes(
        packet[7:11],
        byteorder="little"
    )

    left_rpm = int.from_bytes(
        packet[11:13],
        byteorder="little"
    )

    right_rpm = int.from_bytes(
        packet[13:15],
        byteorder="little"
    )

    throttle = packet[15]

    flags = packet[16]

    # Decode individual status flags
    brake = bool(flags & 0b00000001)

    lap_marker = bool(flags & 0b00000010)

    encoder_valid = bool(flags & 0b00000100)

    # Return decoded telemetry
    return {
        "sequence": sequence,
        "timestamp_ms": timestamp,
        "left_rpm": left_rpm,
        "right_rpm": right_rpm,
        "throttle_percent": throttle,
        "brake": brake,
        "lap_marker": lap_marker,
        "encoder_valid": encoder_valid,
        "crc_valid": True
    }


# ============================================
# 4. TEST THE PACKET DECODER
# ============================================

if __name__ == "__main__":

    from packet import make_packet

    test_packet = make_packet(
        sequence=2,
        timestamp=2000,
        left_rpm=150,
        right_rpm=145,
        throttle=80,
        flags=0b00000100
    )

    # # Simulate corruption during RF transmission
    # corrupted_packet = bytearray(test_packet)

    # # Flip one bit in the left RPM field
    # corrupted_packet[11] ^= 0x01

    # # Use the corrupted packet for testing
    # test_packet = bytes(corrupted_packet)

    print("\n========== PITWALL PACKET DECODER ==========")

    print("\nReceived packet:")
    print(test_packet.hex(" ").upper())

    try:
        telemetry = decode_packet(test_packet)

        print("\nCRC: VALID")

        print("\n========== TELEMETRY ==========")

        for key, value in telemetry.items():
            print(f"{key}: {value}")

    except ValueError as error:
        print("\nPACKET REJECTED!")
        print("Reason:", error)