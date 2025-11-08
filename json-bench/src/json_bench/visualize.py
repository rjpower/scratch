"""Visualization functions for benchmark results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from json_bench.benchmarks import BenchmarkResult


def plot_results(results: list[BenchmarkResult], output_dir: Path | str = ".") -> None:
    """
    Generate visualization plots for benchmark results.

    Creates multiple charts comparing:
    - Serialization time
    - Throughput (docs/second)
    - Bandwidth (MB/second)
    - Output size

    Args:
        results: List of BenchmarkResult objects
        output_dir: Directory to save plots (default: current directory)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to DataFrame
    df = pd.DataFrame([
        {
            "library": r.library,
            "mode": r.mode,
            "time_seconds": r.time_seconds,
            "items_per_second": r.items_per_second,
            "mb_per_second": r.mb_per_second,
            "size_mb": r.size_bytes / 1024 / 1024,
        }
        for r in results
    ])

    # Create pivot tables for easier plotting
    time_pivot = df.pivot(index="library", columns="mode", values="time_seconds")
    throughput_pivot = df.pivot(index="library", columns="mode", values="items_per_second")
    bandwidth_pivot = df.pivot(index="library", columns="mode", values="mb_per_second")
    size_pivot = df.pivot(index="library", columns="mode", values="size_mb")

    # Create a 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("JSON Serialization Library Benchmark Results", fontsize=16, fontweight="bold")

    # Plot 1: Serialization Time (lower is better)
    time_pivot.plot(kind="bar", ax=ax1, color=["#2E86AB", "#A23B72"])
    ax1.set_title("Serialization Time (lower is better)", fontweight="bold")
    ax1.set_xlabel("Library")
    ax1.set_ylabel("Time (seconds)")
    ax1.legend(title="Mode", labels=["Batch", "Loop"])
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

    # Plot 2: Throughput (higher is better)
    throughput_pivot.plot(kind="bar", ax=ax2, color=["#2E86AB", "#A23B72"])
    ax2.set_title("Throughput (higher is better)", fontweight="bold")
    ax2.set_xlabel("Library")
    ax2.set_ylabel("Documents/second")
    ax2.legend(title="Mode", labels=["Batch", "Loop"])
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")

    # Plot 3: Bandwidth (higher is better)
    bandwidth_pivot.plot(kind="bar", ax=ax3, color=["#2E86AB", "#A23B72"])
    ax3.set_title("Serialization Bandwidth (higher is better)", fontweight="bold")
    ax3.set_xlabel("Library")
    ax3.set_ylabel("MB/second")
    ax3.legend(title="Mode", labels=["Batch", "Loop"])
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")

    # Plot 4: Output Size (lower is better for compression)
    size_pivot.plot(kind="bar", ax=ax4, color=["#2E86AB", "#A23B72"])
    ax4.set_title("Serialized Output Size (lower is better)", fontweight="bold")
    ax4.set_xlabel("Library")
    ax4.set_ylabel("Size (MB)")
    ax4.legend(title="Mode", labels=["Batch", "Loop"])
    ax4.grid(axis="y", alpha=0.3)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    output_path = output_dir / "json_bench_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Create a comparison bar chart showing speedup relative to standard json
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle("Speedup Relative to Standard Library JSON", fontsize=16, fontweight="bold")

    # Calculate speedup for loop mode
    json_loop_time = time_pivot.loc["json", "loop"]
    speedup_loop = json_loop_time / time_pivot["loop"]
    speedup_loop = speedup_loop.drop("json")  # Remove json itself

    # Calculate speedup for batch mode
    json_batch_time = time_pivot.loc["json", "batch"]
    speedup_batch = json_batch_time / time_pivot["batch"]
    speedup_batch = speedup_batch.drop("json")  # Remove json itself

    # Plot loop speedup
    speedup_loop.plot(kind="barh", ax=ax5, color="#2E86AB")
    ax5.set_title("Loop Mode Speedup", fontweight="bold")
    ax5.set_xlabel("Speedup (x faster than json)")
    ax5.set_ylabel("Library")
    ax5.axvline(x=1.0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="json baseline")
    ax5.legend()
    ax5.grid(axis="x", alpha=0.3)

    # Plot batch speedup
    speedup_batch.plot(kind="barh", ax=ax6, color="#A23B72")
    ax6.set_title("Batch Mode Speedup", fontweight="bold")
    ax6.set_xlabel("Speedup (x faster than json)")
    ax6.set_ylabel("Library")
    ax6.axvline(x=1.0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="json baseline")
    ax6.legend()
    ax6.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    speedup_path = output_dir / "json_bench_speedup.png"
    plt.savefig(speedup_path, dpi=150, bbox_inches="tight")
    print(f"Speedup plot saved to: {speedup_path}")

    # Save results to CSV
    csv_path = output_dir / "json_bench_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to CSV: {csv_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
