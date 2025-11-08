"""Example usage of msgspec with MessagePack."""

from datetime import datetime
from typing import Optional

import msgspec


# Define a structured type using msgspec.Struct (fastest option)
class User(msgspec.Struct):
    """User record with type validation."""

    id: int
    name: str
    email: str
    created_at: datetime
    metadata: dict[str, str]
    tags: list[str]
    score: Optional[float] = None


def main() -> None:
    """Demonstrate msgspec-msgpack usage."""
    # Create encoder/decoder
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(User)

    # Create sample data
    user = User(
        id=12345,
        name="Alice",
        email="alice@example.com",
        created_at=datetime.now(),
        metadata={"team": "engineering", "level": "senior"},
        tags=["python", "golang", "rust"],
        score=98.5,
    )

    # Encode to bytes
    encoded = encoder.encode(user)
    print(f"Encoded size: {len(encoded)} bytes")
    print(f"Encoded data: {encoded[:50]}...")  # Show first 50 bytes

    # Decode back to User
    decoded = decoder.decode(encoded)
    print(f"\nDecoded user: {decoded}")
    print(f"Type: {type(decoded)}")
    print(f"Created at: {decoded.created_at}")

    # Can also work with dicts/lists (no validation)
    untyped_encoder = msgspec.msgpack.Encoder()
    untyped_decoder = msgspec.msgpack.Decoder()

    data = {
        "users": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ],
        "count": 2,
    }

    encoded_dict = untyped_encoder.encode(data)
    decoded_dict = untyped_decoder.decode(encoded_dict)
    print(f"\nDecoded dict: {decoded_dict}")

    # Cross-language compatibility: save to file
    with open("/tmp/user_data.msgpack", "wb") as f:
        f.write(encoder.encode(user))
    print("\nData saved to /tmp/user_data.msgpack")
    print("Can be read by any MessagePack library in any language!")


if __name__ == "__main__":
    main()
