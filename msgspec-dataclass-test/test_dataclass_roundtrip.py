"""Test msgspec dataclass round-trip serialization with JSON and msgpack."""

from dataclasses import dataclass
from typing import List, Optional
import msgspec


@dataclass
class Person:
    """Example dataclass for testing."""
    name: str
    age: int
    email: Optional[str] = None
    hobbies: List[str] = None

    def __post_init__(self):
        if self.hobbies is None:
            self.hobbies = []


def test_json_roundtrip():
    """Test JSON serialization and deserialization."""
    print("=" * 60)
    print("JSON Round-trip Test")
    print("=" * 60)

    # Create original dataclass
    original = Person(
        name="Alice",
        age=30,
        email="alice@example.com",
        hobbies=["reading", "coding"]
    )
    print(f"\n1. Original object:")
    print(f"   Type: {type(original)}")
    print(f"   Value: {original}")
    print(f"   Is dataclass instance: {isinstance(original, Person)}")

    # Encode to JSON bytes
    json_encoder = msgspec.json.Encoder()
    json_bytes = json_encoder.encode(original)
    print(f"\n2. JSON encoded bytes:")
    print(f"   Type: {type(json_bytes)}")
    print(f"   Value: {json_bytes}")
    print(f"   Decoded string: {json_bytes.decode()}")

    # Decode back to Python object
    json_decoder = msgspec.json.Decoder(Person)
    decoded = json_decoder.decode(json_bytes)
    print(f"\n3. Decoded object:")
    print(f"   Type: {type(decoded)}")
    print(f"   Value: {decoded}")
    print(f"   Is dataclass instance: {isinstance(decoded, Person)}")

    # Verify equality
    print(f"\n4. Verification:")
    print(f"   decoded == original: {decoded == original}")
    print(f"   decoded.name: {decoded.name}")
    print(f"   decoded.age: {decoded.age}")
    print(f"   decoded.email: {decoded.email}")
    print(f"   decoded.hobbies: {decoded.hobbies}")

    # What happens without type hint?
    print(f"\n5. Decoding without type hint:")
    json_decoder_untyped = msgspec.json.Decoder()
    decoded_untyped = json_decoder_untyped.decode(json_bytes)
    print(f"   Type: {type(decoded_untyped)}")
    print(f"   Value: {decoded_untyped}")
    print(f"   Is dict: {isinstance(decoded_untyped, dict)}")

    assert decoded == original
    print("\n✓ JSON round-trip successful!")


def test_msgpack_roundtrip():
    """Test msgpack serialization and deserialization."""
    print("\n" + "=" * 60)
    print("MessagePack Round-trip Test")
    print("=" * 60)

    # Create original dataclass
    original = Person(
        name="Bob",
        age=25,
        email="bob@example.com",
        hobbies=["gaming", "music", "sports"]
    )
    print(f"\n1. Original object:")
    print(f"   Type: {type(original)}")
    print(f"   Value: {original}")
    print(f"   Is dataclass instance: {isinstance(original, Person)}")

    # Encode to msgpack bytes
    msgpack_encoder = msgspec.msgpack.Encoder()
    msgpack_bytes = msgpack_encoder.encode(original)
    print(f"\n2. MessagePack encoded bytes:")
    print(f"   Type: {type(msgpack_bytes)}")
    print(f"   Length: {len(msgpack_bytes)} bytes")
    print(f"   Value (hex): {msgpack_bytes.hex()}")

    # Decode back to Python object
    msgpack_decoder = msgspec.msgpack.Decoder(Person)
    decoded = msgpack_decoder.decode(msgpack_bytes)
    print(f"\n3. Decoded object:")
    print(f"   Type: {type(decoded)}")
    print(f"   Value: {decoded}")
    print(f"   Is dataclass instance: {isinstance(decoded, Person)}")

    # Verify equality
    print(f"\n4. Verification:")
    print(f"   decoded == original: {decoded == original}")
    print(f"   decoded.name: {decoded.name}")
    print(f"   decoded.age: {decoded.age}")
    print(f"   decoded.email: {decoded.email}")
    print(f"   decoded.hobbies: {decoded.hobbies}")

    # What happens without type hint?
    print(f"\n5. Decoding without type hint:")
    msgpack_decoder_untyped = msgspec.msgpack.Decoder()
    decoded_untyped = msgpack_decoder_untyped.decode(msgpack_bytes)
    print(f"   Type: {type(decoded_untyped)}")
    print(f"   Value: {decoded_untyped}")
    print(f"   Is dict: {isinstance(decoded_untyped, dict)}")

    assert decoded == original
    print("\n✓ MessagePack round-trip successful!")


def test_comparison():
    """Compare JSON vs msgpack encoding size."""
    print("\n" + "=" * 60)
    print("Size Comparison: JSON vs MessagePack")
    print("=" * 60)

    original = Person(
        name="Charlie" * 10,  # Longer name
        age=42,
        email="charlie@verylongdomainname.com",
        hobbies=["hobby1", "hobby2", "hobby3", "hobby4", "hobby5"]
    )

    json_bytes = msgspec.json.encode(original)
    msgpack_bytes = msgspec.msgpack.encode(original)

    print(f"\nOriginal object: {original}")
    print(f"\nJSON size: {len(json_bytes)} bytes")
    print(f"MessagePack size: {len(msgpack_bytes)} bytes")
    print(f"Size difference: {len(json_bytes) - len(msgpack_bytes)} bytes")
    print(f"MessagePack is {len(json_bytes) / len(msgpack_bytes):.2f}x smaller")

    print(f"\nJSON output:\n{json_bytes.decode()}")
    print(f"\nMessagePack output (hex):\n{msgpack_bytes.hex()}")


if __name__ == "__main__":
    test_json_roundtrip()
    test_msgpack_roundtrip()
    test_comparison()
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
