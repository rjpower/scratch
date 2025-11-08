"""Benchmarking functions for JSON serialization libraries."""

import json
import pickle
import time
from typing import Any, Callable

import msgpack
import orjson
import msgspec
import rapidjson
import ujson
from tqdm import tqdm

from json_bench.data_generator import generate_documents


class BenchmarkResult:
    """Results from a benchmark run."""

    def __init__(
        self,
        library: str,
        mode: str,
        time_seconds: float,
        size_bytes: int,
        items_per_second: float,
        mb_per_second: float,
    ):
        self.library = library
        self.mode = mode
        self.time_seconds = time_seconds
        self.size_bytes = size_bytes
        self.items_per_second = items_per_second
        self.mb_per_second = mb_per_second

    def __repr__(self) -> str:
        return (
            f"{self.library:12s} ({self.mode:5s}): "
            f"{self.time_seconds:6.3f}s, "
            f"{self.items_per_second:8.1f} docs/s, "
            f"{self.mb_per_second:6.2f} MB/s, "
            f"{self.size_bytes / 1024 / 1024:6.2f} MB"
        )


def _benchmark_serializer(
    name: str,
    documents: list[dict],
    serialize_one: Callable[[Any], bytes],
    serialize_batch: Callable[[Any], bytes],
    num_iterations: int = 1,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """
    Benchmark a serialization library in both loop and batch modes.

    Args:
        name: Name of the library
        documents: List of documents to serialize
        serialize_one: Function to serialize a single document
        serialize_batch: Function to serialize a batch of documents
        num_iterations: Number of times to repeat the benchmark

    Returns:
        Tuple of (loop_result, batch_result)
    """
    num_docs = len(documents)

    # Loop mode: serialize each document individually
    start = time.perf_counter()
    total_size = 0
    for _ in range(num_iterations):
        for doc in documents:
            serialized = serialize_one(doc)
            total_size += len(serialized)
    loop_time = time.perf_counter() - start

    loop_size = total_size // num_iterations
    loop_result = BenchmarkResult(
        library=name,
        mode="loop",
        time_seconds=loop_time / num_iterations,
        size_bytes=loop_size,
        items_per_second=num_docs / (loop_time / num_iterations),
        mb_per_second=(loop_size / 1024 / 1024) / (loop_time / num_iterations),
    )

    # Batch mode: serialize all documents as a single list
    start = time.perf_counter()
    batch_size = 0
    for _ in range(num_iterations):
        serialized = serialize_batch(documents)
        batch_size = len(serialized)
    batch_time = time.perf_counter() - start

    batch_result = BenchmarkResult(
        library=name,
        mode="batch",
        time_seconds=batch_time / num_iterations,
        size_bytes=batch_size,
        items_per_second=num_docs / (batch_time / num_iterations),
        mb_per_second=(batch_size / 1024 / 1024) / (batch_time / num_iterations),
    )

    return loop_result, batch_result


def run_benchmarks(
    num_documents: int = 1000,
    num_iterations: int = 5,
) -> list[BenchmarkResult]:
    """
    Run benchmarks for all JSON serialization libraries.

    Args:
        num_documents: Number of documents to generate
        num_iterations: Number of times to repeat each benchmark

    Returns:
        List of BenchmarkResult objects
    """
    print(f"Generating {num_documents} test documents...")
    documents = generate_documents(num_documents)

    results = []

    # Standard library json
    print("\nBenchmarking json (standard library)...")
    loop, batch = _benchmark_serializer(
        "json",
        documents,
        lambda x: json.dumps(x).encode("utf-8"),
        lambda x: json.dumps(x).encode("utf-8"),
        num_iterations,
    )
    results.extend([loop, batch])
    print(f"  {loop}")
    print(f"  {batch}")

    # orjson
    print("\nBenchmarking orjson...")
    loop, batch = _benchmark_serializer(
        "orjson",
        documents,
        orjson.dumps,
        orjson.dumps,
        num_iterations,
    )
    results.extend([loop, batch])
    print(f"  {loop}")
    print(f"  {batch}")

    # msgspec (JSON mode)
    print("\nBenchmarking msgspec (JSON)...")
    encoder = msgspec.json.Encoder()
    loop, batch = _benchmark_serializer(
        "msgspec-json",
        documents,
        encoder.encode,
        encoder.encode,
        num_iterations,
    )
    results.extend([loop, batch])
    print(f"  {loop}")
    print(f"  {batch}")

    # msgspec (msgpack mode)
    print("\nBenchmarking msgspec (msgpack)...")
    mp_encoder = msgspec.msgpack.Encoder()
    loop, batch = _benchmark_serializer(
        "msgspec-msgpack",
        documents,
        mp_encoder.encode,
        mp_encoder.encode,
        num_iterations,
    )
    results.extend([loop, batch])
    print(f"  {loop}")
    print(f"  {batch}")

    # ujson
    print("\nBenchmarking ujson...")
    loop, batch = _benchmark_serializer(
        "ujson",
        documents,
        lambda x: ujson.dumps(x).encode("utf-8"),
        lambda x: ujson.dumps(x).encode("utf-8"),
        num_iterations,
    )
    results.extend([loop, batch])
    print(f"  {loop}")
    print(f"  {batch}")

    # rapidjson
    print("\nBenchmarking rapidjson...")
    loop, batch = _benchmark_serializer(
        "rapidjson",
        documents,
        lambda x: rapidjson.dumps(x).encode("utf-8"),
        lambda x: rapidjson.dumps(x).encode("utf-8"),
        num_iterations,
    )
    results.extend([loop, batch])
    print(f"  {loop}")
    print(f"  {batch}")

    # msgpack
    print("\nBenchmarking msgpack...")
    loop, batch = _benchmark_serializer(
        "msgpack",
        documents,
        msgpack.packb,
        msgpack.packb,
        num_iterations,
    )
    results.extend([loop, batch])
    print(f"  {loop}")
    print(f"  {batch}")

    # pickle (default protocol - usually 5)
    print("\nBenchmarking pickle (default protocol)...")
    loop, batch = _benchmark_serializer(
        "pickle",
        documents,
        pickle.dumps,
        pickle.dumps,
        num_iterations,
    )
    results.extend([loop, batch])
    print(f"  {loop}")
    print(f"  {batch}")

    # pickle protocol 4
    print("\nBenchmarking pickle (protocol 4)...")
    loop, batch = _benchmark_serializer(
        "pickle-p4",
        documents,
        lambda x: pickle.dumps(x, protocol=4),
        lambda x: pickle.dumps(x, protocol=4),
        num_iterations,
    )
    results.extend([loop, batch])
    print(f"  {loop}")
    print(f"  {batch}")

    # pickle protocol 5
    print("\nBenchmarking pickle (protocol 5)...")
    loop, batch = _benchmark_serializer(
        "pickle-p5",
        documents,
        lambda x: pickle.dumps(x, protocol=5),
        lambda x: pickle.dumps(x, protocol=5),
        num_iterations,
    )
    results.extend([loop, batch])
    print(f"  {loop}")
    print(f"  {batch}")

    # Hybrid: msgpack then pickle wrapper
    print("\nBenchmarking msgpack+pickle hybrid...")
    mp_enc = msgspec.msgpack.Encoder()

    def hybrid_loop(doc):
        msgpack_bytes = mp_enc.encode(doc)
        return pickle.dumps({"type": "msgpack", "blob": msgpack_bytes})

    def hybrid_batch(docs):
        msgpack_bytes = mp_enc.encode(docs)
        return pickle.dumps({"type": "msgpack", "blob": msgpack_bytes})

    loop, batch = _benchmark_serializer(
        "msgpack+pickle",
        documents,
        hybrid_loop,
        hybrid_batch,
        num_iterations,
    )
    results.extend([loop, batch])
    print(f"  {loop}")
    print(f"  {batch}")

    return results
