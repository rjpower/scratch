# json-bench

Benchmarking suite for Python JSON serialization libraries.

## Libraries Compared

- **json** - Python standard library
- **orjson** - Fast, correct JSON library
- **msgspec** - High-performance JSON/msgpack serialization
- **ujson** - Ultra fast JSON encoder/decoder
- **rapidjson** - Python wrapper for RapidJSON C++ library
- **pickle** - Python object serialization (baseline)
- **msgpack** - Binary serialization format

## Benchmark Scenarios

1. **Loop serialization**: Serialize 1000 documents individually in a loop
2. **Batch serialization**: Serialize all 1000 documents as a single top-level list

## Usage

```bash
uv run json-bench run
```

This will run benchmarks and generate comparison graphs in the current directory.
