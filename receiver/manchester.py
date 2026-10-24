# PitWall - Manchester Encoder

from packet import make_packet

def manchester_encode(data):

    encoded = []

    for byte in data:
        for i in range(7, -1, -1):

            bit = (byte >> i) & 1
            if bit == 0:
                encoded.extend([0, 1])
            else:
                encoded.extend([1, 0])
    return encoded


def manchester_decode(chips):

    decoded_bits = []

    # Every original bit is represented by two chips
    for i in range(0, len(chips), 2):

        pair = chips[i:i+2]

        if pair == [0, 1]:
            decoded_bits.append(0)

        elif pair == [1, 0]:
            decoded_bits.append(1)

        else:
            raise ValueError(f"Invalid Manchester pair: {pair}")

    # Convert decoded bits back into bytes
    decoded_bytes = bytearray()

    for i in range(0, len(decoded_bits), 8):

        byte = 0

        for bit in decoded_bits[i:i+8]:
            byte = (byte << 1) | bit

        decoded_bytes.append(byte)

    return bytes(decoded_bytes)

# ==============================
# TEST ENCODER AND DECODER
# ==============================

test_data = bytes([0xA5])

# Encode original data
encoded = manchester_encode(test_data)

# Decode Manchester chips
decoded = manchester_decode(encoded)

# ======================================
# TEST WITH ACTUAL PITWALL PACKET
# ======================================

# Generate telemetry packet
original_packet = make_packet(
    sequence=2,
    timestamp=2000,
    left_rpm=150,
    right_rpm=145,
    throttle=80,
    flags=0b00000100
)

# Manchester encoding
encoded_chips = manchester_encode(original_packet)

# Manchester decoding
decoded_packet = manchester_decode(encoded_chips)

# # Display results
# print("\n========== PITWALL MANCHESTER TEST ==========\n")

# print("Original packet:")
# print(original_packet.hex(" ").upper())

# print("\nOriginal packet length:", len(original_packet), "bytes")

# print("\nManchester encoded length:", len(encoded_chips), "chips")

# print("\nDecoded packet:")
# print(decoded_packet.hex(" ").upper())

# print("\nPacket match:", original_packet == decoded_packet)