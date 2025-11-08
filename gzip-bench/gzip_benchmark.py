#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib>=3.8.0",
#     "numpy>=1.26.0",
# ]
# ///
"""
Benchmark gzip compression effectiveness on JSON documents.

This script analyzes:
- Compression ratio vs compression level
- Execution time vs compression level
- Size reduction effectiveness
"""

import gzip
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def generate_sample_text(word_count: int = 1000) -> str:
    """Generate sample text with realistic word patterns."""
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
        # Add some variation to make it more realistic
        if i % 10 == 0:
            words.append(f"item_{i}")
        elif i % 7 == 0:
            words.append(f"{i}")
        else:
            words.append(common_words[i % len(common_words)])

    return " ".join(words)


def create_json_document() -> dict:
    """Create a JSON document with 1000 word body and metadata."""
    return {
        "id": "doc_12345678",
        "timestamp": "2025-11-08T10:30:00Z",
        "version": "1.0.0",
        "type": "article",
        "status": "published",
        "metadata": {
            "author": "John Doe",
            "category": "technology",
            "tags": ["benchmark", "compression", "performance", "gzip", "json"],
            "views": 12543,
            "likes": 892,
            "comments": 45,
            "shares": 123,
            "language": "en",
            "region": "us-west-2",
        },
        "body": generate_sample_text(1000),
        "summary": generate_sample_text(50),
        "references": [
            {"id": f"ref_{i}", "url": f"https://example.com/doc/{i}", "title": f"Reference {i}"}
            for i in range(10)
        ],
        "analytics": {
            "page_load_time": 1.234,
            "bounce_rate": 0.23,
            "session_duration": 456.78,
            "engagement_score": 8.5,
        }
    }


def benchmark_compression(data: bytes, compression_level: int, iterations: int = 100) -> tuple[float, float, int]:
    """
    Benchmark compression for a specific level.

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

    avg_time = np.mean(times) * 1000  # Convert to milliseconds
    original_size = len(data)
    compressed_size = len(compressed_data)
    compression_ratio = (1 - compressed_size / original_size) * 100  # Percentage reduction

    return avg_time, compression_ratio, compressed_size


def run_benchmarks():
    """Run compression benchmarks across all compression levels."""
    print("Generating sample JSON document...")
    doc = create_json_document()
    json_str = json.dumps(doc, indent=2)
    json_bytes = json_str.encode('utf-8')

    original_size = len(json_bytes)
    print(f"Original JSON size: {original_size:,} bytes ({original_size/1024:.2f} KB)")
    print(f"Word count in body: {len(doc['body'].split())}")
    print("\nRunning benchmarks...")

    results = {
        'levels': [],
        'times': [],
        'ratios': [],
        'sizes': [],
    }

    for level in range(10):  # Compression levels 0-9
        print(f"  Testing level {level}...", end=" ")
        avg_time, ratio, size = benchmark_compression(json_bytes, level)

        results['levels'].append(level)
        results['times'].append(avg_time)
        results['ratios'].append(ratio)
        results['sizes'].append(size)

        print(f"Time: {avg_time:.3f}ms, Reduction: {ratio:.1f}%, Size: {size:,} bytes")

    return results, original_size


def create_visualization(results: dict, original_size: int):
    """Create comprehensive visualization of benchmark results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GZIP Compression Analysis on JSON Documents\n(1000 word body + metadata)',
                 fontsize=14, fontweight='bold')

    levels = results['levels']
    times = results['times']
    ratios = results['ratios']
    sizes = results['sizes']

    # Plot 1: Compression Time vs Level
    ax1 = axes[0, 0]
    ax1.plot(levels, times, 'o-', linewidth=2, markersize=8, color='#2563eb')
    ax1.set_xlabel('Compression Level', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Compression Time (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('Execution Time vs Compression Level', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(levels)

    # Add min/max annotations
    min_idx = np.argmin(times)
    max_idx = np.argmax(times)
    ax1.annotate(f'Fastest\n{times[min_idx]:.3f}ms',
                xy=(levels[min_idx], times[min_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Plot 2: Compression Ratio vs Level
    ax2 = axes[0, 1]
    ax2.plot(levels, ratios, 's-', linewidth=2, markersize=8, color='#16a34a')
    ax2.set_xlabel('Compression Level', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Size Reduction (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Compression Effectiveness vs Level', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(levels)

    # Add max compression annotation
    best_idx = np.argmax(ratios)
    ax2.annotate(f'Best\n{ratios[best_idx]:.1f}%',
                xy=(levels[best_idx], ratios[best_idx]),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Plot 3: Compressed Size vs Level
    ax3 = axes[1, 0]
    bars = ax3.bar(levels, [s/1024 for s in sizes], color='#dc2626', alpha=0.7, edgecolor='black')
    ax3.axhline(y=original_size/1024, color='gray', linestyle='--', linewidth=2, label=f'Original: {original_size/1024:.1f} KB')
    ax3.set_xlabel('Compression Level', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Compressed Size (KB)', fontsize=11, fontweight='bold')
    ax3.set_title('Compressed Size vs Level', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(levels)
    ax3.legend()

    # Plot 4: Time vs Compression Ratio (scatter with level labels)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(times, ratios, s=200, c=levels, cmap='viridis',
                         edgecolors='black', linewidth=1.5, alpha=0.8)

    # Add level labels to points
    for i, level in enumerate(levels):
        ax4.annotate(f'{level}', (times[i], ratios[i]),
                    ha='center', va='center', fontweight='bold', fontsize=9)

    ax4.set_xlabel('Compression Time (ms)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Size Reduction (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Time vs Effectiveness Trade-off', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Compression Level', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save the figure
    output_path = Path(__file__).parent / 'gzip_compression_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to: {output_path}")

    # Also show it
    plt.show()


def print_summary(results: dict, original_size: int):
    """Print a summary of the benchmark results."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    times = results['times']
    ratios = results['ratios']
    sizes = results['sizes']

    fastest_idx = np.argmin(times)
    slowest_idx = np.argmax(times)
    best_compression_idx = np.argmax(ratios)
    worst_compression_idx = np.argmin(ratios)

    print(f"\nOriginal size: {original_size:,} bytes ({original_size/1024:.2f} KB)")
    print(f"\nFastest compression:")
    print(f"  Level {fastest_idx}: {times[fastest_idx]:.3f}ms ({ratios[fastest_idx]:.1f}% reduction)")

    print(f"\nBest compression:")
    print(f"  Level {best_compression_idx}: {ratios[best_compression_idx]:.1f}% reduction ({times[best_compression_idx]:.3f}ms)")
    print(f"  Compressed size: {sizes[best_compression_idx]:,} bytes ({sizes[best_compression_idx]/1024:.2f} KB)")

    print(f"\nRecommended (Level 6 - default balance):")
    print(f"  Time: {times[6]:.3f}ms")
    print(f"  Reduction: {ratios[6]:.1f}%")
    print(f"  Size: {sizes[6]:,} bytes ({sizes[6]/1024:.2f} KB)")

    # Calculate diminishing returns
    print(f"\nDiminishing returns analysis:")
    for i in range(1, 10):
        time_increase = ((times[i] - times[i-1]) / times[i-1]) * 100
        ratio_increase = ratios[i] - ratios[i-1]
        print(f"  {i-1} -> {i}: +{time_increase:+6.1f}% time, {ratio_increase:+.2f}% better compression")

    print("="*70)


def main():
    """Main entry point."""
    print("="*70)
    print("GZIP Compression Benchmark")
    print("="*70)

    results, original_size = run_benchmarks()
    print_summary(results, original_size)
    create_visualization(results, original_size)


if __name__ == "__main__":
    main()
