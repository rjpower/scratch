"""Command-line interface for json-bench."""

import sys
from pathlib import Path

from json_bench.benchmarks import run_benchmarks
from json_bench.visualize import plot_results


def main() -> int:
    """Main entry point for the CLI."""
    print("=" * 100)
    print("JSON SERIALIZATION BENCHMARK SUITE")
    print("=" * 100)
    print("\nComparing serialization libraries:")
    print("  - json (standard library)")
    print("  - orjson")
    print("  - msgspec (JSON + msgpack modes)")
    print("  - ujson")
    print("  - rapidjson")
    print("  - msgpack")
    print("  - pickle")
    print("\nRunning benchmarks with 1000 documents, 5 iterations each...")
    print("=" * 100)

    results = run_benchmarks(num_documents=1000, num_iterations=5)

    print("\n" + "=" * 100)
    print("GENERATING VISUALIZATIONS")
    print("=" * 100)

    output_dir = Path.cwd()
    plot_results(results, output_dir)

    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETE!")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
