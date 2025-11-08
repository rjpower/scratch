#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "zstandard>=0.22.0",
#     "click>=8.1.0",
# ]
# ///
"""
Benchmark gzip and zstd compression effectiveness on JSON documents.

This script analyzes:
- Compression ratio vs compression level
- Execution time vs compression level
- Size reduction effectiveness
- Comparison with memcpy baseline
"""

import gzip
import json
import time
import random
from pathlib import Path
import zstandard as zstd
import click


def generate_sample_text(word_count: int, doc_id: int) -> str:
    """Generate sample text with realistic word patterns and randomization."""
    # Use a mix of common words to simulate realistic JSON data
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "data", "process", "system", "information", "user", "service", "application",
        "request", "response", "error", "status", "message", "result", "value",
        "database", "server", "client", "network", "performance", "security",
        "configuration", "parameter", "function", "method", "object", "property",
        "timestamp", "identifier", "resource", "endpoint", "payload", "metadata",
        "authorization", "authentication", "validation", "processing", "execution"
    ]

    words = []
    for i in range(word_count):
        # Add randomization to create unique documents
        rand_val = random.randint(0, 100)
        if rand_val < 10:
            # 10% - document-specific identifiers
            words.append(f"doc{doc_id}_item_{i}")
        elif rand_val < 20:
            # 10% - numbers
            words.append(f"{random.randint(0, 10000)}")
        elif rand_val < 25:
            # 5% - unique strings
            words.append(f"unique_{doc_id}_{random.randint(1000, 9999)}")
        else:
            # 75% - random common words (not sequential pattern)
            words.append(random.choice(common_words))

    return " ".join(words)


def create_json_document(doc_id: int) -> dict:
    """Create a JSON document with 1000 word body and unique metadata per document."""
    # Randomize document attributes to make each document unique
    authors = ["John Doe", "Jane Smith", "Bob Johnson", "Alice Williams", "Charlie Brown",
               "Diana Prince", "Eve Anderson", "Frank Miller", "Grace Lee", "Henry Taylor"]
    categories = ["technology", "science", "business", "health", "education",
                  "entertainment", "sports", "politics", "travel", "food"]
    doc_types = ["article", "blog", "report", "paper", "tutorial", "review", "guide"]
    statuses = ["published", "draft", "pending", "archived", "featured"]
    languages = ["en", "es", "fr", "de", "ja", "zh", "pt", "ru", "ar", "hi"]
    regions = ["us-west-2", "us-east-1", "eu-west-1", "ap-south-1", "eu-central-1"]

    all_tags = ["benchmark", "compression", "performance", "gzip", "json", "zstd",
                "optimization", "data", "processing", "analysis", "storage", "efficiency"]

    return {
        "id": f"doc_{doc_id:08d}_{random.randint(10000, 99999)}",
        "timestamp": f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:00Z",
        "version": f"{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,20)}",
        "type": random.choice(doc_types),
        "status": random.choice(statuses),
        "metadata": {
            "author": random.choice(authors),
            "category": random.choice(categories),
            "tags": random.sample(all_tags, k=random.randint(3, 8)),
            "views": random.randint(100, 50000),
            "likes": random.randint(10, 5000),
            "comments": random.randint(0, 500),
            "shares": random.randint(0, 1000),
            "language": random.choice(languages),
            "region": random.choice(regions),
        },
        "body": generate_sample_text(1000, doc_id),
        "summary": generate_sample_text(50, doc_id),
        "references": [
            {
                "id": f"ref_{doc_id}_{i}_{random.randint(1000, 9999)}",
                "url": f"https://example.com/doc/{doc_id}/{i}/{random.randint(1000, 9999)}",
                "title": f"Reference {i} for document {doc_id}"
            }
            for i in range(random.randint(5, 15))
        ],
        "analytics": {
            "page_load_time": round(random.uniform(0.5, 5.0), 3),
            "bounce_rate": round(random.uniform(0.1, 0.6), 2),
            "session_duration": round(random.uniform(30, 900), 2),
            "engagement_score": round(random.uniform(1.0, 10.0), 1),
        }
    }


def benchmark_memcpy(data: bytes, iterations: int) -> tuple[float, float, int]:
    """
    Benchmark memcpy (baseline - no compression).

    Returns:
        (avg_time, compression_ratio, size)
    """
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        # Force a copy using bytearray conversion (actual memory copy)
        copied_data = bytearray(data)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times) * 1000  # Convert to milliseconds
    return avg_time, 0.0, len(data)


def benchmark_gzip(data: bytes, compression_level: int, iterations: int) -> tuple[float, float, int]:
    """
    Benchmark gzip compression for a specific level.

    Returns:
        (avg_time, compression_ratio, compressed_size)
    """
    import io
    times = []
    compressed_data = None

    for _ in range(iterations):
        start = time.perf_counter()
        # Use BytesIO for in-memory compression
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=compression_level) as f:
            f.write(data)
        compressed_data = buf.getvalue()
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times) * 1000  # Convert to milliseconds
    original_size = len(data)
    compressed_size = len(compressed_data)
    compression_ratio = (1 - compressed_size / original_size) * 100  # Percentage reduction

    return avg_time, compression_ratio, compressed_size


def benchmark_zstd(data: bytes, compression_level: int, iterations: int) -> tuple[float, float, int]:
    """
    Benchmark zstd compression for a specific level.

    Returns:
        (avg_time, compression_ratio, compressed_size)
    """
    times = []
    compressed_data = None
    cctx = zstd.ZstdCompressor(level=compression_level)

    for _ in range(iterations):
        start = time.perf_counter()
        compressed_data = cctx.compress(data)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times) * 1000  # Convert to milliseconds
    original_size = len(data)
    compressed_size = len(compressed_data)
    compression_ratio = (1 - compressed_size / original_size) * 100  # Percentage reduction

    return avg_time, compression_ratio, compressed_size


def run_benchmarks(num_docs: int, iterations: int, gzip_levels: range, zstd_levels: range):
    """Run compression benchmarks across all compression levels."""
    print(f"Generating {num_docs:,} unique sample JSON documents...")

    # Generate multiple unique documents with different content
    docs = []
    for i in range(num_docs):
        doc = create_json_document(doc_id=i)
        docs.append(doc)

    # Convert all docs to a single JSON array
    json_str = json.dumps(docs, separators=(',', ':'))  # Compact format
    json_bytes = json_str.encode('utf-8')

    original_size = len(json_bytes)
    print(f"Total data size: {original_size:,} bytes ({original_size/1024/1024:.2f} MB)")
    print(f"Average document size: {original_size/num_docs:.0f} bytes")
    print(f"\nRunning benchmarks ({iterations} iterations each)...")

    results = {
        'memcpy': {},
        'gzip': {},
        'zstd': {},
    }

    # Benchmark memcpy (baseline)
    print("\n  Testing memcpy (baseline)...")
    avg_time, ratio, size = benchmark_memcpy(json_bytes, iterations * 10)
    results['memcpy'] = {
        'time': avg_time,
        'ratio': ratio,
        'size': size,
        'throughput': (original_size / 1024 / 1024) / (avg_time / 1000)  # MB/s
    }
    print(f"    Time: {avg_time:.3f}ms, Throughput: {results['memcpy']['throughput']:.1f} MB/s")

    # Benchmark gzip
    print(f"\n  Testing GZIP compression levels {gzip_levels.start} to {gzip_levels.stop - 1}...")
    for level in gzip_levels:
        print(f"    Level {level}...", end=" ", flush=True)
        avg_time, ratio, size = benchmark_gzip(json_bytes, level, iterations)
        results['gzip'][level] = {
            'time': avg_time,
            'ratio': ratio,
            'size': size,
            'throughput': (original_size / 1024 / 1024) / (avg_time / 1000)  # MB/s
        }
        print(f"Time: {avg_time:.2f}ms, Ratio: {ratio:.1f}%, Throughput: {results['gzip'][level]['throughput']:.1f} MB/s")

    # Benchmark zstd
    print(f"\n  Testing ZSTD compression levels {zstd_levels.start} to {zstd_levels.stop - 1}...")
    for level in zstd_levels:
        print(f"    Level {level}...", end=" ", flush=True)
        avg_time, ratio, size = benchmark_zstd(json_bytes, level, iterations)
        results['zstd'][level] = {
            'time': avg_time,
            'ratio': ratio,
            'size': size,
            'throughput': (original_size / 1024 / 1024) / (avg_time / 1000)  # MB/s
        }
        print(f"Time: {avg_time:.2f}ms, Ratio: {ratio:.1f}%, Throughput: {results['zstd'][level]['throughput']:.1f} MB/s")

    return results, original_size


def print_results(results: dict, original_size: int):
    """Print comprehensive benchmark results."""
    print("\n" + "="*100)
    print("COMPRESSION BENCHMARK RESULTS")
    print("="*100)

    print(f"\nOriginal size: {original_size:,} bytes ({original_size/1024/1024:.2f} MB)")

    # Print memcpy baseline
    print("\n" + "-"*100)
    print("MEMCPY BASELINE (no compression)")
    print("-"*100)
    memcpy = results['memcpy']
    print(f"Time: {memcpy['time']:.3f}ms | Throughput: {memcpy['throughput']:.1f} MB/s")

    # Print GZIP results
    print("\n" + "-"*100)
    print("GZIP COMPRESSION")
    print("-"*100)
    print(f"{'Level':<8} {'Time (ms)':<12} {'Ratio (%)':<12} {'Size (MB)':<12} {'Throughput':<15} {'vs memcpy':<15}")
    print("-"*100)

    for level in sorted(results['gzip'].keys()):
        data = results['gzip'][level]
        vs_memcpy = f"{data['time'] / memcpy['time']:.2f}x slower"
        size_mb = data['size'] / 1024 / 1024
        print(f"{level:<8} {data['time']:<12.2f} {data['ratio']:<12.1f} {size_mb:<12.2f} {data['throughput']:<15.1f} {vs_memcpy:<15}")

    # Find best GZIP levels
    best_gzip_ratio = max(results['gzip'].items(), key=lambda x: x[1]['ratio'])
    fastest_gzip = min(results['gzip'].items(), key=lambda x: x[1]['time'])
    print(f"\nBest compression: Level {best_gzip_ratio[0]} ({best_gzip_ratio[1]['ratio']:.1f}% reduction)")
    print(f"Fastest: Level {fastest_gzip[0]} ({fastest_gzip[1]['time']:.2f}ms)")

    # Print ZSTD results
    print("\n" + "-"*100)
    print("ZSTD COMPRESSION")
    print("-"*100)
    print(f"{'Level':<8} {'Time (ms)':<12} {'Ratio (%)':<12} {'Size (MB)':<12} {'Throughput':<15} {'vs memcpy':<15}")
    print("-"*100)

    for level in sorted(results['zstd'].keys()):
        data = results['zstd'][level]
        vs_memcpy = f"{data['time'] / memcpy['time']:.2f}x slower"
        size_mb = data['size'] / 1024 / 1024
        print(f"{level:<8} {data['time']:<12.2f} {data['ratio']:<12.1f} {size_mb:<12.2f} {data['throughput']:<15.1f} {vs_memcpy:<15}")

    # Find best ZSTD levels
    best_zstd_ratio = max(results['zstd'].items(), key=lambda x: x[1]['ratio'])
    fastest_zstd = min(results['zstd'].items(), key=lambda x: x[1]['time'])
    print(f"\nBest compression: Level {best_zstd_ratio[0]} ({best_zstd_ratio[1]['ratio']:.1f}% reduction)")
    print(f"Fastest: Level {fastest_zstd[0]} ({fastest_zstd[1]['time']:.2f}ms)")

    # Comparison summary
    print("\n" + "="*100)
    print("SUMMARY: GZIP vs ZSTD vs MEMCPY")
    print("="*100)

    # Get default levels if they exist, otherwise use best
    gzip_default_level = 6 if 6 in results['gzip'] else best_gzip_ratio[0]
    zstd_default_level = 3 if 3 in results['zstd'] else best_zstd_ratio[0]
    gzip_default = results['gzip'][gzip_default_level]
    zstd_default = results['zstd'][zstd_default_level]

    print(f"\n{'Method':<15} {'Level':<8} {'Time (ms)':<12} {'Ratio (%)':<12} {'Throughput':<15} {'vs memcpy':<15}")
    print("-"*100)
    memcpy_vs = "1.00x"
    gzip_def_vs = f"{gzip_default['time']/memcpy['time']:.2f}x"
    zstd_def_vs = f"{zstd_default['time']/memcpy['time']:.2f}x"
    gzip_best_vs = f"{best_gzip_ratio[1]['time']/memcpy['time']:.2f}x"
    zstd_best_vs = f"{best_zstd_ratio[1]['time']/memcpy['time']:.2f}x"
    gzip_fast_vs = f"{fastest_gzip[1]['time']/memcpy['time']:.2f}x"
    zstd_fast_vs = f"{fastest_zstd[1]['time']/memcpy['time']:.2f}x"

    print(f"{'memcpy':<15} {'-':<8} {memcpy['time']:<12.3f} {memcpy['ratio']:<12.1f} {memcpy['throughput']:<15.1f} {memcpy_vs:<15}")
    print(f"{'GZIP (default)':<15} {gzip_default_level:<8} {gzip_default['time']:<12.2f} {gzip_default['ratio']:<12.1f} {gzip_default['throughput']:<15.1f} {gzip_def_vs:<15}")
    print(f"{'ZSTD (default)':<15} {zstd_default_level:<8} {zstd_default['time']:<12.2f} {zstd_default['ratio']:<12.1f} {zstd_default['throughput']:<15.1f} {zstd_def_vs:<15}")
    print(f"{'GZIP (best)':<15} {best_gzip_ratio[0]:<8} {best_gzip_ratio[1]['time']:<12.2f} {best_gzip_ratio[1]['ratio']:<12.1f} {best_gzip_ratio[1]['throughput']:<15.1f} {gzip_best_vs:<15}")
    print(f"{'ZSTD (best)':<15} {best_zstd_ratio[0]:<8} {best_zstd_ratio[1]['time']:<12.2f} {best_zstd_ratio[1]['ratio']:<12.1f} {best_zstd_ratio[1]['throughput']:<15.1f} {zstd_best_vs:<15}")
    print(f"{'GZIP (fastest)':<15} {fastest_gzip[0]:<8} {fastest_gzip[1]['time']:<12.2f} {fastest_gzip[1]['ratio']:<12.1f} {fastest_gzip[1]['throughput']:<15.1f} {gzip_fast_vs:<15}")
    print(f"{'ZSTD (fastest)':<15} {fastest_zstd[0]:<8} {fastest_zstd[1]['time']:<12.2f} {fastest_zstd[1]['ratio']:<12.1f} {fastest_zstd[1]['throughput']:<15.1f} {zstd_fast_vs:<15}")

    print("\n" + "="*100)
    print("KEY FINDINGS:")
    print("="*100)
    if gzip_default['ratio'] > 0:
        print(f"- ZSTD is {zstd_default['ratio']/gzip_default['ratio']:.2f}x better compression ratio than GZIP at default levels")
    print(f"- ZSTD is {gzip_default['time']/zstd_default['time']:.2f}x faster than GZIP at default levels")
    print(f"- Best ZSTD compression ({best_zstd_ratio[1]['ratio']:.1f}%) vs Best GZIP ({best_gzip_ratio[1]['ratio']:.1f}%)")
    print(f"- All compression methods are slower than memcpy baseline, as expected")
    print("="*100)


@click.command()
@click.option('--docs', '-d', default=10000, help='Number of JSON documents to generate', show_default=True)
@click.option('--iterations', '-i', default=1, help='Number of iterations per compression level', show_default=True)
@click.option('--gzip-min', default=0, help='Minimum GZIP compression level (0 to 9)', show_default=True)
@click.option('--gzip-max', default=9, help='Maximum GZIP compression level (0 to 9)', show_default=True)
@click.option('--zstd-min', default=-10, help='Minimum ZSTD compression level (-100 to 22)', show_default=True)
@click.option('--zstd-max', default=10, help='Maximum ZSTD compression level (-100 to 22)', show_default=True)
def main(docs, iterations, gzip_min, gzip_max, zstd_min, zstd_max):
    """Benchmark GZIP and ZSTD compression on JSON documents."""
    print("="*100)
    print(f"GZIP vs ZSTD Compression Benchmark ({docs:,} documents, {iterations} iterations)")
    print("="*100)

    results, original_size = run_benchmarks(
        num_docs=docs,
        iterations=iterations,
        gzip_levels=range(gzip_min, gzip_max + 1),
        zstd_levels=range(zstd_min, zstd_max + 1)
    )
    print_results(results, original_size)


if __name__ == "__main__":
    main()
