#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click>=8.1.0",
#     "numpy>=1.24.0",
# ]
# ///
"""
Benchmark simple tokenization with different batch sizes.

This script analyzes:
- Tokenization time vs batch size
- Throughput (tokens/sec and docs/sec) vs batch size
- Performance scaling with batch size
- Batch processing efficiency
"""

import time
import random
import click
import numpy as np


class SimpleTokenizer:
    """Simple whitespace tokenizer that simulates HF tokenizer behavior."""

    def __init__(self):
        self.vocab = {}
        self.vocab_size = 50257  # GPT-2 vocab size

    def __call__(self, texts, padding=True, truncation=True, return_tensors=None, max_length=2048):
        """Tokenize a batch of texts."""
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize each text (simple split on whitespace)
        all_tokens = []
        for text in texts:
            tokens = text.split()
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]
            # Simulate token IDs
            token_ids = [hash(token) % self.vocab_size for token in tokens]
            all_tokens.append(token_ids)

        # Padding
        if padding:
            max_len = max(len(tokens) for tokens in all_tokens)
            all_tokens = [tokens + [0] * (max_len - len(tokens)) for tokens in all_tokens]

        # Return a dict-like object that mimics transformers output
        return {
            'input_ids': MockTensor(all_tokens)
        }


class MockTensor:
    """Mock tensor class to simulate PyTorch tensors."""

    def __init__(self, data):
        self.data = data

    def numel(self):
        """Return total number of elements."""
        if isinstance(self.data, list):
            return sum(len(row) for row in self.data)
        return len(self.data)


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
    total_words = sum(len(d.split()) for d in docs)
    print(f"Generated {num_docs:,} documents (total ~{total_words:,} words)")
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


def run_benchmarks(num_docs: int, words_per_doc: int,
                   batch_sizes: list[int], iterations: int) -> dict:
    """Run tokenization benchmarks across all batch sizes."""
    print("="*100)
    print(f"Tokenizer Batch Size Benchmark")
    print(f"Documents: {num_docs:,} x {words_per_doc:,} words")
    print(f"Iterations per batch size: {iterations}")
    print("="*100)
    print()

    # Create simple tokenizer
    print("Creating tokenizer...")
    tokenizer = SimpleTokenizer()
    print(f"Tokenizer created: vocab size = {tokenizer.vocab_size:,}")
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
        'num_docs': num_docs,
        'words_per_doc': words_per_doc,
        'results': results
    }


def print_results(benchmark_data: dict):
    """Print comprehensive benchmark results."""
    results = benchmark_data['results']
    num_docs = benchmark_data['num_docs']
    words_per_doc = benchmark_data['words_per_doc']

    print("\n" + "="*120)
    print("TOKENIZER BENCHMARK RESULTS")
    print("="*120)
    print(f"Dataset: {num_docs:,} documents x {words_per_doc:,} words")
    print(f"Average tokens per document: {results[0]['avg_tokens_per_doc']:.1f}")
    print(f"Total tokens: {results[0]['total_tokens']:,}")

    print("\n" + "-"*120)
    print("PERFORMANCE BY BATCH SIZE")
    print("-"*120)
    print(f"{'Batch':<8} {'Batches':<10} {'Avg Time (s)':<14} {'Min Time (s)':<14} {'Max Time (s)':<14} {'Docs/sec':<12} {'Tokens/sec':<15}")
    print("-"*120)

    for result in results:
        print(f"{result['batch_size']:<8} "
              f"{result['num_batches']:<10} "
              f"{result['avg_time']:<14.4f} "
              f"{result['min_time']:<14.4f} "
              f"{result['max_time']:<14.4f} "
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

    # Calculate speedup from smallest to largest batch
    baseline = results[0]
    fastest_speedup = baseline['avg_time'] / fastest['avg_time']
    print(f"\nSpeedup from batch size 1 to {fastest['batch_size']}: {fastest_speedup:.2f}x faster")

    # Show scaling efficiency
    print("\n" + "-"*120)
    print("BATCH SIZE SCALING")
    print("-"*120)
    print(f"{'Batch':<8} {'Speedup':<12} {'Efficiency':<12} {'vs Batch 1':<15}")
    print("-"*120)

    baseline_time = results[0]['avg_time']
    for result in results:
        speedup = baseline_time / result['avg_time']
        # Ideal speedup would be linear with batch size, but that's unrealistic
        # More realistic is comparing against log2 scaling
        ideal_speedup = result['batch_size'] / results[0]['batch_size']
        efficiency = speedup / ideal_speedup * 100 if ideal_speedup > 0 else 0
        vs_baseline = f"{speedup:.2f}x"
        print(f"{result['batch_size']:<8} {speedup:<12.2f} {efficiency:<12.1f}% {vs_baseline:<15}")

    print("\n" + "="*120)
    print("OBSERVATIONS")
    print("="*120)

    # Calculate where diminishing returns start
    improvements = []
    for i in range(1, len(results)):
        prev_time = results[i-1]['avg_time']
        curr_time = results[i]['avg_time']
        improvement = (prev_time - curr_time) / prev_time * 100
        improvements.append((results[i]['batch_size'], improvement))

    # Find where improvement drops below 5%
    diminishing_point = None
    for batch_size, improvement in improvements:
        if improvement < 5.0:
            diminishing_point = batch_size
            break

    if diminishing_point:
        print(f"- Diminishing returns begin around batch size {diminishing_point}")
    print(f"- Batch processing provides up to {fastest_speedup:.2f}x speedup over single-document processing")
    print(f"- Optimal batch size for this workload appears to be: {fastest['batch_size']}")

    # Show time per batch
    print(f"\nTime per batch:")
    for result in results:
        time_per_batch = result['avg_time'] / result['num_batches'] * 1000  # ms
        print(f"  Batch size {result['batch_size']:2d}: {time_per_batch:.2f}ms per batch ({result['num_batches']} batches total)")

    print("\n" + "="*120)


@click.command()
@click.option('--docs', '-d', default=512, help='Number of documents to generate (default 64*8=512 for even division)', show_default=True)
@click.option('--words', '-w', default=1000, help='Words per document', show_default=True)
@click.option('--batch-min', default=1, help='Minimum batch size', show_default=True)
@click.option('--batch-max', default=64, help='Maximum batch size', show_default=True)
@click.option('--iterations', '-i', default=3, help='Number of iterations per batch size', show_default=True)
def main(docs, words, batch_min, batch_max, iterations):
    """Benchmark tokenization with different batch sizes."""
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
        num_docs=docs,
        words_per_doc=words,
        batch_sizes=batch_sizes,
        iterations=iterations
    )

    print_results(benchmark_data)


if __name__ == "__main__":
    main()
