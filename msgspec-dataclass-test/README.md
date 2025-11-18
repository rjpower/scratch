# msgspec Dataclass Round-trip Test

This project tests msgspec's dataclass serialization behavior with both JSON and MessagePack formats.

## Key Findings

### 1. **Decoded Result Type**

When using msgspec to decode data:
- **With type hint**: `Decoder(Person)` → Returns a **dataclass instance**
- **Without type hint**: `Decoder()` → Returns a **dict**

This is true for both JSON and MessagePack formats.

### 2. **JSON Serialization**

```python
json_encoder = msgspec.json.Encoder()
json_decoder = msgspec.json.Decoder(Person)

# Encode
json_bytes = json_encoder.encode(person)  # Returns bytes

# Decode with type → dataclass instance
decoded = json_decoder.decode(json_bytes)  # Returns Person instance

# Decode without type → dict
decoded_dict = msgspec.json.Decoder().decode(json_bytes)  # Returns dict
```

### 3. **MessagePack Serialization**

```python
msgpack_encoder = msgspec.msgpack.Encoder()
msgpack_decoder = msgspec.msgpack.Decoder(Person)

# Encode
msgpack_bytes = msgpack_encoder.encode(person)  # Returns bytes

# Decode with type → dataclass instance
decoded = msgpack_decoder.decode(msgpack_bytes)  # Returns Person instance

# Decode without type → dict
decoded_dict = msgspec.msgpack.Decoder().decode(msgpack_bytes)  # Returns dict
```

### 4. **Size Comparison**

MessagePack typically produces smaller byte sizes than JSON:
- JSON: 188 bytes (human-readable)
- MessagePack: 164 bytes (binary format)
- MessagePack is ~1.15x smaller in this example

## Running the Tests

1. Install dependencies:
   ```bash
   pip install msgspec
   ```

2. Run the test:
   ```bash
   python test_dataclass_roundtrip.py
   ```

## Test Output Summary

The test demonstrates:
1. ✓ JSON round-trip preserves dataclass type (when decoder has type hint)
2. ✓ MessagePack round-trip preserves dataclass type (when decoder has type hint)
3. ✓ Without type hint, both formats decode to dict
4. ✓ Both formats correctly preserve all field values
5. ✓ MessagePack produces smaller byte sizes
