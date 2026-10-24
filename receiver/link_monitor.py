# =====================================================
# PITWALL - RF LINK MONITOR
# Task 9: Packet Reception Ratio and CRC Failures
# =====================================================

from packet import make_packet
from packet_decoder import decode_packet


# =====================================================
# 1. EXPERIMENT SETTINGS
# =====================================================

TOTAL_PACKETS = 10

# Packets that never reach the receiver
LOST_SEQUENCES = {2,3,5,7,9}

# Packets that arrive but contain corrupted data
CORRUPTED_SEQUENCES = {4,8}


# =====================================================
# 2. GENERATE TELEMETRY PACKETS
# =====================================================

transmitted_packets = []

for sequence in range(1, TOTAL_PACKETS + 1):

    packet = make_packet(
        sequence=sequence,
        timestamp=1000 + (sequence - 1) * 200,
        left_rpm=100 + sequence * 5,
        right_rpm=98 + sequence * 5,
        throttle=min(100, 20 + sequence * 7),
        flags=0b00000100
    )

    transmitted_packets.append(packet)


print("\n========== PITWALL RF LINK MONITOR ==========")

print("\nPackets transmitted:", len(transmitted_packets))


# =====================================================
# 3. SIMULATE PACKET LOSS AND CORRUPTION
# =====================================================

received_packets = []

for packet in transmitted_packets:

    sequence = packet[6]

    # Simulate complete packet loss
    if sequence in LOST_SEQUENCES:

        print(f"\nPacket {sequence}: LOST")

        continue

    # Simulate a corrupted packet
    if sequence in CORRUPTED_SEQUENCES:

        corrupted_packet = bytearray(packet)

        # Flip one bit in the left RPM field
        # without updating the CRC.
        corrupted_packet[11] ^= 0x01

        packet = bytes(corrupted_packet)

    received_packets.append(packet)


# =====================================================
# 4. VERIFY RECEIVED PACKETS
# =====================================================

valid_packets = []

crc_failures = 0

for packet in received_packets:

    # Sequence byte is used here only for
    # simulation diagnostics.
    sequence = packet[6]

    try:

        telemetry = decode_packet(packet)

        valid_packets.append(telemetry)

        print(f"\nPacket {sequence}: ACCEPTED")
        print("Left RPM:", telemetry["left_rpm"])
        print("Right RPM:", telemetry["right_rpm"])
        print("CRC: VALID")

    except ValueError as error:

        crc_failures += 1

        print(f"\nPacket {sequence}: REJECTED")
        print("Reason:", error)


# =====================================================
# 5. CALCULATE PACKET RECEPTION RATIO
# =====================================================

transmitted_count = len(transmitted_packets)

valid_count = len(valid_packets)

received_count = len(received_packets)

prr = (
    valid_count / transmitted_count * 100
    if transmitted_count > 0
    else 0
)


# =====================================================
# 6. DETECT MISSING SEQUENCES
# =====================================================

expected_sequences = set(
    range(1, TOTAL_PACKETS + 1)
)

valid_sequences = {
    packet["sequence"]
    for packet in valid_packets
}

missing_sequences = sorted(
    expected_sequences - valid_sequences
)


# =====================================================
# 7. DISPLAY FINAL STATISTICS
# =====================================================

print("\n========== RF LINK STATISTICS ==========")

print("Packets transmitted:", transmitted_count)

print("Packets received (including corrupted):", received_count)

print("Valid packets received:", valid_count)

print("CRC failures:", crc_failures)

print(
    "Completely lost packets:",
    transmitted_count - received_count
)

print("Missing valid sequences:", missing_sequences)

print(f"\nPacket Reception Ratio: {prr:.2f}%")

print(
    "Packet Loss Ratio:",
    f"{100 - prr:.2f}%"
)

print("\n========== EXPERIMENT COMPLETE ==========")