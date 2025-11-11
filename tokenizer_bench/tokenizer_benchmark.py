#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.35.0",
#     "torch>=2.0.0",
#     "click>=8.1.0",
#     "numpy>=1.24.0",
# ]
# ///
"""
Benchmark HuggingFace AutoTokenizer with different batch sizes.

This script analyzes:
- Tokenization time vs batch size
- Throughput (tokens/sec and docs/sec) vs batch size
- Performance scaling with batch size
- Memory efficiency characteristics
"""

import time
import random
import click
import numpy as np
from transformers import AutoTokenizer


def generate_sample_text(word_count: int, doc_id: int) -> str:
    """Generate sample text with realistic word patterns."""
    # Use a mix of common words to simulate realistic text
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
        "authorization", "authentication", "validation", "processing", "execution",
        "machine", "learning", "model", "training", "inference", "neural", "network",
        "algorithm", "optimization", "gradient", "descent", "backpropagation", "layer"
    ]

    words = []
    for i in range(word_count):
        # Add some randomization
        rand_val = random.randint(0, 100)
        if rand_val < 5:
            # 5% - document-specific identifiers
            words.append(f"doc{doc_id}_item_{i}")
        elif rand_val < 10:
            # 5% - numbers
            words.append(f"{random.randint(0, 10000)}")
        else:
            # 90% - random common words
            words.append(random.choice(common_words))

    return " ".join(words)


def generate_documents(num_docs: int, words_per_doc: int) -> list[str]:
    """Generate a list of sample documents."""
    print(f"Generating {num_docs:,} documents with {words_per_doc:,} words each...")
    docs = []
    for i in range(num_docs):
        doc = generate_sample_text(words_per_doc, i)
        docs.append(doc)
    print(f"Generated {num_docs:,} documents (total ~{sum(len(d.split()) for d in docs):,} words)")
    return docs


def benchmark_tokenization(tokenizer, documents: list[str], batch_size: int, iterations: int = 3) -> dict:
    """
    Benchmark tokenization for a specific batch size.

    Returns:
        dict with timing and throughput metrics
    """
    num_docs = len(documents)
    times = []
    total_tokens = 0

    # Split documents into equal batches using numpy
    num_batches = num_docs // batch_size
    batches = np.array_split(documents, num_batches)

    for iteration in range(iterations):
        iteration_tokens = 0
        start = time.perf_counter()

        # Process each batch
        for batch in batches:
            # Tokenize the batch
            encoded = tokenizer(
                batch.tolist(),
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=2048
            )
            # Count tokens in this batch
            iteration_tokens += encoded['input_ids'].numel()

        end = time.perf_counter()
        times.append(end - start)
        total_tokens = iteration_tokens

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Calculate metrics
    docs_per_sec = num_docs / avg_time
    tokens_per_sec = total_tokens / avg_time
    avg_tokens_per_doc = total_tokens / num_docs

    return {
        'batch_size': batch_size,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'docs_per_sec': docs_per_sec,
        'tokens_per_sec': tokens_per_sec,
        'total_tokens': total_tokens,
        'avg_tokens_per_doc': avg_tokens_per_doc,
        'num_batches': num_batches
    }


def run_benchmarks(tokenizer_name: str, num_docs: int, words_per_doc: int,
                   batch_sizes: list[int], iterations: int) -> dict:
    """Run tokenization benchmarks across all batch sizes."""
    print("="*100)
    print(f"HuggingFace AutoTokenizer Benchmark")
    print(f"Model: {tokenizer_name}")
    print(f"Documents: {num_docs:,} x {words_per_doc:,} words")
    print(f"Iterations per batch size: {iterations}")
    print("="*100)
    print()

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded: vocab size = {tokenizer.vocab_size:,}")
    print()

    # Generate documents
    documents = generate_documents(num_docs, words_per_doc)
    print()

    # Run benchmarks for each batch size
    print(f"Running benchmarks for batch sizes: {batch_sizes[0]} to {batch_sizes[-1]}...")
    print()

    results = []
    for batch_size in batch_sizes:
        print(f"  Batch size {batch_size:3d}...", end=" ", flush=True)
        result = benchmark_tokenization(tokenizer, documents, batch_size, iterations)
        results.append(result)
        print(f"Time: {result['avg_time']:.3f}s, Throughput: {result['docs_per_sec']:.1f} docs/s, {result['tokens_per_sec']:.0f} tokens/s")

    return {
        'tokenizer_name': tokenizer_name,
        'num_docs': num_docs,
        'words_per_doc': words_per_doc,
        'results': results
    }


def print_results(benchmark_data: dict):
    """Print comprehensive benchmark results."""
    results = benchmark_data['results']
    tokenizer_name = benchmark_data['tokenizer_name']
    num_docs = benchmark_data['num_docs']
    words_per_doc = benchmark_data['words_per_doc']

    print("\n" + "="*120)
    print("TOKENIZER BENCHMARK RESULTS")
    print("="*120)
    print(f"Model: {tokenizer_name}")
    print(f"Dataset: {num_docs:,} documents x {words_per_doc:,} words")
    print(f"Average tokens per document: {results[0]['avg_tokens_per_doc']:.1f}")
    print(f"Total tokens: {results[0]['total_tokens']:,}")

    print("\n" + "-"*120)
    print("PERFORMANCE BY BATCH SIZE")
    print("-"*120)
    print(f"{'Batch':<8} {'Batches':<10} {'Avg Time':<12} {'Min Time':<12} {'Max Time':<12} {'Docs/sec':<12} {'Tokens/sec':<15}")
    print("-"*120)

    for result in results:
        print(f"{result['batch_size']:<8} "
              f"{result['num_batches']:<10} "
              f"{result['avg_time']:<12.4f} "
              f"{result['min_time']:<12.4f} "
              f"{result['max_time']:<12.4f} "
              f"{result['docs_per_sec']:<12.1f} "
              f"{result['tokens_per_sec']:<15.0f}")

    # Find best performers
    fastest = min(results, key=lambda x: x['avg_time'])
    highest_docs_throughput = max(results, key=lambda x: x['docs_per_sec'])
    highest_tokens_throughput = max(results, key=lambda x: x['tokens_per_sec'])

    print("\n" + "="*120)
    print("KEY FINDINGS")
    print("="*120)
    print(f"Fastest overall: Batch size {fastest['batch_size']} ({fastest['avg_time']:.3f}s)")
    print(f"Best docs/sec throughput: Batch size {highest_docs_throughput['batch_size']} ({highest_docs_throughput['docs_per_sec']:.1f} docs/sec)")
    print(f"Best tokens/sec throughput: Batch size {highest_tokens_throughput['batch_size']} ({highest_tokens_throughput['tokens_per_sec']:.0f} tokens/sec)")

    # Calculate speedup from smallest to largest batch (based on tokens/sec)
    baseline = results[0]
    max_batch = results[-1]
    throughput_speedup = highest_tokens_throughput['tokens_per_sec'] / baseline['tokens_per_sec']
    print(f"\nSpeedup from batch size {baseline['batch_size']} to {max_batch['batch_size']}: {throughput_speedup:.2f}x faster (tokens/sec)")

    # Show scaling efficiency
    print("\n" + "-"*120)
    print("BATCH SIZE SCALING (based on tokens/sec throughput)")
    print("-"*120)
    print(f"{'Batch':<8} {'Speedup':<12} {'Efficiency':<12} {'vs Batch 1':<15}")
    print("-"*120)

    baseline_tokens_per_sec = results[0]['tokens_per_sec']
    for result in results:
        speedup = result['tokens_per_sec'] / baseline_tokens_per_sec
        # Ideal speedup would be linear with batch size
        ideal_speedup = result['batch_size'] / results[0]['batch_size']
        efficiency = speedup / ideal_speedup * 100 if ideal_speedup > 0 else 0
        vs_baseline = f"{speedup:.2f}x"
        print(f"{result['batch_size']:<8} {speedup:<12.2f} {efficiency:<12.1f}% {vs_baseline:<15}")

    print("\n" + "="*120)
    print("OBSERVATIONS")
    print("="*120)

    # Calculate where diminishing returns start (based on tokens/sec improvement)
    improvements = []
    for i in range(1, len(results)):
        prev_tokens_per_sec = results[i-1]['tokens_per_sec']
        curr_tokens_per_sec = results[i]['tokens_per_sec']
        improvement = (curr_tokens_per_sec - prev_tokens_per_sec) / prev_tokens_per_sec * 100
        improvements.append((results[i]['batch_size'], improvement))

    # Find where improvement drops below 5%
    diminishing_point = None
    for batch_size, improvement in improvements:
        if improvement < 5.0:
            diminishing_point = batch_size
            break

    if diminishing_point:
        print(f"- Diminishing returns begin around batch size {diminishing_point}")
    print(f"- Batch processing provides up to {throughput_speedup:.2f}x speedup in tokens/sec throughput")
    print(f"- Optimal batch size for tokens/sec throughput: {highest_tokens_throughput['batch_size']}")

    # Memory consideration note
    print(f"- Larger batch sizes trade memory usage for throughput")
    print("="*120)


@click.command()
@click.option('--model', '-m', default='gpt2', help='HuggingFace model name', show_default=True)
@click.option('--docs', '-d', default=512, help='Number of documents to generate (default 64*8=512 for even division)', show_default=True)
@click.option('--words', '-w', default=1000, help='Words per document', show_default=True)
@click.option('--batch-min', default=1, help='Minimum batch size', show_default=True)
@click.option('--batch-max', default=64, help='Maximum batch size', show_default=True)
@click.option('--iterations', '-i', default=3, help='Number of iterations per batch size', show_default=True)
def main(model, docs, words, batch_min, batch_max, iterations):
    """Benchmark HuggingFace AutoTokenizer with different batch sizes."""
    # Generate batch sizes: 1, 2, 4, 8, 16, 32, 64
    batch_sizes = []
    size = batch_min
    while size <= batch_max:
        batch_sizes.append(size)
        if size < 2:
            size = 2
        else:
            size *= 2
    # Ensure max is included if not already
    if batch_sizes[-1] < batch_max:
        batch_sizes.append(batch_max)

    # Validate that docs is evenly divisible by all batch sizes
    for bs in batch_sizes:
        if docs % bs != 0:
            print(f"Warning: {docs} documents is not evenly divisible by batch size {bs}")
            print(f"Recommended: Use a number of docs that's a multiple of {batch_max} (e.g., {batch_max * 8})")
            break

    benchmark_data = run_benchmarks(
        tokenizer_name=model,
        num_docs=docs,
        words_per_doc=words,
        batch_sizes=batch_sizes,
        iterations=iterations
    )

    print_results(benchmark_data)


if __name__ == "__main__":
    main()
