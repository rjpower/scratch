"""Side-by-side comparison of JSON vs MessagePack encoding."""

import json
from datetime import datetime

import msgspec
import msgpack as standard_msgpack


def main() -> None:
    """Compare JSON and MessagePack for the same data."""
    # Sample data with various types
    data = {
        "id": 123456789012345,  # Large integer
        "name": "Alice Smith",
        "email": "alice@example.com",
        "active": True,
        "balance": 1234.56,
        "created_at": datetime.now().isoformat(),
        "tags": ["python", "golang", "rust", "javascript"],
        "metadata": {
            "team": "engineering",
            "level": "senior",
            "location": "San Francisco",
        },
        "scores": [95.5, 87.3, 92.1, 88.9, 91.2],
        # Binary data example (e.g., a small token or hash)
        "token": b"\x01\x02\x03\x04\x05\x06\x07\x08",
    }

    print("=" * 80)
    print("JSON vs MessagePack Comparison")
    print("=" * 80)

    # Standard library JSON
    json_bytes = json.dumps(data, default=lambda x: x.hex() if isinstance(x, bytes) else str(x)).encode(
        "utf-8"
    )
    print(f"\nStandard JSON:")
    print(f"  Size: {len(json_bytes)} bytes")
    print(f"  Human-readable: Yes")
    print(f"  Sample: {json_bytes[:100].decode('utf-8')}...")

    # msgspec JSON
    msgspec_json_encoder = msgspec.json.Encoder()
    msgspec_json_bytes = msgspec_json_encoder.encode(data)
    print(f"\nmsgspec JSON:")
    print(f"  Size: {len(msgspec_json_bytes)} bytes")
    print(f"  Savings vs standard: {len(json_bytes) - len(msgspec_json_bytes)} bytes")

    # msgspec MessagePack
    msgspec_msgpack_encoder = msgspec.msgpack.Encoder()
    msgspec_msgpack_bytes = msgspec_msgpack_encoder.encode(data)
    print(f"\nmsgspec MessagePack:")
    print(f"  Size: {len(msgspec_msgpack_bytes)} bytes")
    print(f"  Savings vs JSON: {len(json_bytes) - len(msgspec_msgpack_bytes)} bytes")
    print(f"  Size reduction: {(1 - len(msgspec_msgpack_bytes) / len(json_bytes)) * 100:.1f}%")
    print(f"  Human-readable: No (binary)")
    print(f"  Sample: {msgspec_msgpack_bytes[:50]}")

    # Standard msgpack library (for comparison)
    standard_msgpack_bytes = standard_msgpack.packb(data)
    print(f"\nStandard msgpack library:")
    print(f"  Size: {len(standard_msgpack_bytes)} bytes")

    print("\n" + "=" * 80)
    print("Type Support Comparison")
    print("=" * 80)

    # Test with types that JSON struggles with
    special_data = {
        "large_int": 2**63 - 1,  # Max int64
        "binary": bytes(range(256)),  # Binary data
        "datetime": datetime.now(),  # Native datetime (msgpack ext type)
    }

    print("\nData with types JSON doesn't handle natively:")

    # JSON requires encoding workarounds
    json_special = json.dumps(
        {
            "large_int": special_data["large_int"],
            "binary": special_data["binary"].hex(),  # Must encode as hex string
            "datetime": special_data["datetime"].isoformat(),  # Must encode as string
        }
    ).encode("utf-8")

    # MessagePack handles these natively
    msgpack_special = msgspec_msgpack_encoder.encode(special_data)

    print(f"  JSON size (with workarounds): {len(json_special)} bytes")
    print(f"  MessagePack size (native): {len(msgpack_special)} bytes")
    print(f"  MessagePack is {(1 - len(msgpack_special) / len(json_special)) * 100:.1f}% smaller")

    print("\n" + "=" * 80)
    print("Decoding Performance")
    print("=" * 80)

    # Decode to verify roundtrip
    msgspec_json_decoder = msgspec.json.Decoder()
    msgspec_msgpack_decoder = msgspec.msgpack.Decoder()

    decoded_json = msgspec_json_decoder.decode(msgspec_json_bytes)
    decoded_msgpack = msgspec_msgpack_decoder.decode(msgspec_msgpack_bytes)

    print(f"\nBoth formats successfully roundtrip!")
    print(f"JSON decoded: {list(decoded_json.keys())}")
    print(f"MessagePack decoded: {list(decoded_msgpack.keys())}")

    print("\n" + "=" * 80)
    print("Recommendations")
    print("=" * 80)
    print("""
Use MessagePack when:
  ✓ Performance is critical
  ✓ Bandwidth/storage efficiency matters
  ✓ Working with binary data
  ✓ Internal services communication
  ✓ Need native support for large integers, datetimes, binary

Use JSON when:
  ✓ Human readability required
  ✓ Browser/web API consumption
  ✓ Maximum compatibility needed
  ✓ Debugging ease is important
  ✓ Industry standard APIs (REST, etc.)
""")


if __name__ == "__main__":
    main()
