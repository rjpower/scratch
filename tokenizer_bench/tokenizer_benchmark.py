#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.35.0",
#     "torch>=2.0.0",
#     "tiktoken>=0.5.0",
#     "kitoken>=0.2.0",
#     "click>=8.1.0",
#     "numpy>=1.24.0",
# ]
# ///
"""
Unified tokenizer benchmark for HuggingFace, tiktoken, kitoken, and a simple baseline.

Benchmarks different tokenizers with varying batch sizes and outputs raw performance metrics.
"""

import time
import random
import click
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict


class TokenizerWrapper(ABC):
    """Abstract base class for tokenizer wrappers."""

    @abstractmethod
    def encode_batch(self, texts: List[str]) -> int:
        """Encode a batch of texts and return total token count."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get tokenizer name."""
        pass


class SimpleTokenizer(TokenizerWrapper):
    """Simple whitespace-based tokenizer for baseline comparison."""

    def __init__(self):
        self.vocab_size = 50257
        print(f"  Loaded simple tokenizer (vocab_size={self.vocab_size:,})")

    def encode_batch(self, texts: List[str]) -> int:
        total_tokens = 0
        for text in texts:
            tokens = text.split()
            total_tokens += len(tokens)
        return total_tokens

    def get_name(self) -> str:
        return "simple/whitespace"


class HuggingFaceTokenizer(TokenizerWrapper):
    """Wrapper for HuggingFace tokenizers."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer
        print(f"  Loading HuggingFace tokenizer: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name
        print(f"  Loaded (vocab_size={self.tokenizer.vocab_size:,})")

    def encode_batch(self, texts: List[str]) -> int:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=2048
        )
        return encoded['input_ids'].numel()

    def get_name(self) -> str:
        return f"huggingface/{self.model_name}"


class TiktokenTokenizer(TokenizerWrapper):
    """Wrapper for tiktoken tokenizers."""

    def __init__(self, encoding_name: str):
        import tiktoken
        print(f"  Loading tiktoken encoding: {encoding_name}...")
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name
        print(f"  Loaded")

    def encode_batch(self, texts: List[str]) -> int:
        total_tokens = 0
        for text in texts:
            tokens = self.tokenizer.encode(text)
            total_tokens += len(tokens)
        return total_tokens

    def get_name(self) -> str:
        return f"tiktoken/{self.encoding_name}"


class KitokenTokenizer(TokenizerWrapper):
    """Wrapper for kitoken tokenizers."""

    def __init__(self, tokenizer_json_path: str):
        from kitoken import Kitoken
        print(f"  Loading kitoken from: {tokenizer_json_path}...")
        self.tokenizer = Kitoken.from_file(tokenizer_json_path)
        self.tokenizer_json_path = tokenizer_json_path
        print(f"  Loaded")

    def encode_batch(self, texts: List[str]) -> int:
        total_tokens = 0
        for text in texts:
            tokens = self.tokenizer.encode(text, False)  # False = no special tokens
            total_tokens += len(list(tokens))
        return total_tokens

    def get_name(self) -> str:
        path = Path(self.tokenizer_json_path)
        # Get parent directory name if available
        if path.parent.name and path.parent.name != '.':
            return f"kitoken/{path.parent.name}"
        return f"kitoken/{path.stem}"


def generate_sample_text(word_count: int, doc_id: int) -> str:
    """Generate sample text with realistic word patterns."""
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
        rand_val = random.randint(0, 100)
        if rand_val < 5:
            words.append(f"doc{doc_id}_item_{i}")
        elif rand_val < 10:
            words.append(f"{random.randint(0, 10000)}")
        else:
            words.append(random.choice(common_words))

    return " ".join(words)


def generate_documents(num_docs: int, words_per_doc: int) -> List[str]:
    """Generate a list of sample documents."""
    print(f"Generating {num_docs:,} documents with {words_per_doc:,} words each...")
    docs = []
    for i in range(num_docs):
        doc = generate_sample_text(words_per_doc, i)
        docs.append(doc)
    print(f"Generated {num_docs:,} documents (total ~{sum(len(d.split()) for d in docs):,} words)")
    return docs


def benchmark_tokenizer(tokenizer: TokenizerWrapper, documents: List[str],
                        batch_size: int, iterations: int = 3) -> Dict:
    """
    Benchmark a tokenizer for a specific batch size.

    Returns:
        dict with timing and throughput metrics
    """
    num_docs = len(documents)
    times = []
    total_tokens = 0

    # Split documents into equal batches
    num_batches = num_docs // batch_size
    batches = np.array_split(documents, num_batches)

    for iteration in range(iterations):
        iteration_tokens = 0
        start = time.perf_counter()

        # Process each batch
        for batch in batches:
            batch_list = batch.tolist() if hasattr(batch, 'tolist') else list(batch)
            iteration_tokens += tokenizer.encode_batch(batch_list)

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
        'num_batches': num_batches,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'docs_per_sec': docs_per_sec,
        'tokens_per_sec': tokens_per_sec,
        'total_tokens': total_tokens,
        'avg_tokens_per_doc': avg_tokens_per_doc,
    }


def run_benchmark_suite(tokenizer: TokenizerWrapper, documents: List[str],
                        batch_sizes: List[int], iterations: int) -> Dict:
    """Run benchmark suite for a tokenizer across all batch sizes."""
    tokenizer_name = tokenizer.get_name()

    print(f"\n{'='*100}")
    print(f"BENCHMARKING: {tokenizer_name}")
    print(f"{'='*100}")

    results = []
    for batch_size in batch_sizes:
        print(f"  Batch size {batch_size:3d}...", end=" ", flush=True)
        result = benchmark_tokenizer(tokenizer, documents, batch_size, iterations)
        results.append(result)
        print(f"{result['avg_time']:.3f}s | {result['tokens_per_sec']:.0f} tokens/s")

    return {
        'tokenizer_name': tokenizer_name,
        'results': results
    }


def print_results_table(benchmark_data: Dict):
    """Print results table with raw numbers."""
    results = benchmark_data['results']
    tokenizer_name = benchmark_data['tokenizer_name']

    print(f"\n{'='*120}")
    print(f"{tokenizer_name}")
    print(f"{'='*120}")
    print(f"Avg tokens/doc: {results[0]['avg_tokens_per_doc']:.1f}")
    print(f"Total tokens: {results[0]['total_tokens']:,}")
    print()

    # Print header
    print(f"{'Batch':<8} {'Batches':<10} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12} {'Docs/s':<12} {'Tokens/s':<15}")
    print("-" * 120)

    # Print data rows
    for r in results:
        print(f"{r['batch_size']:<8} "
              f"{r['num_batches']:<10} "
              f"{r['avg_time']:<12.4f} "
              f"{r['min_time']:<12.4f} "
              f"{r['max_time']:<12.4f} "
              f"{r['docs_per_sec']:<12.1f} "
              f"{r['tokens_per_sec']:<15.0f}")

    print()


def find_hf_tokenizer_json(model_name: str) -> str:
    """Find tokenizer.json for a HuggingFace model in cache."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dirs = list(cache_dir.glob(f"models--{model_name.replace('/', '--')}*"))

    if not model_dirs:
        raise FileNotFoundError(f"Could not find cached model directory for {model_name}")

    model_dir = model_dirs[0]
    tokenizer_json_files = list(model_dir.glob("**/tokenizer.json"))

    if not tokenizer_json_files:
        raise FileNotFoundError(f"Could not find tokenizer.json for {model_name}")

    return str(tokenizer_json_files[0])


@click.command()
@click.option('--docs', '-d', default=512, help='Number of documents')
@click.option('--words', '-w', default=1000, help='Words per document')
@click.option('--batch-min', default=1, help='Minimum batch size')
@click.option('--batch-max', default=64, help='Maximum batch size')
@click.option('--iterations', '-i', default=3, help='Iterations per batch size')
@click.option('--simple', is_flag=True, help='Include simple baseline tokenizer')
@click.option('--hf-models', help='Comma-separated list of HuggingFace models to benchmark')
@click.option('--tiktoken-encodings', help='Comma-separated list of tiktoken encodings to benchmark')
@click.option('--kitoken-files', help='Comma-separated list of tokenizer.json paths for kitoken')
def main(docs, words, batch_min, batch_max, iterations, simple, hf_models, tiktoken_encodings, kitoken_files):
    """
    Unified tokenizer benchmark for HuggingFace, tiktoken, kitoken, and simple baseline.

    Examples:
        # Benchmark only simple tokenizer
        ./benchmark_all_tokenizers.py --simple

        # Benchmark HuggingFace models
        ./benchmark_all_tokenizers.py --hf-models gpt2,NousResearch/Nous-Hermes-Llama2-13b

        # Benchmark tiktoken
        ./benchmark_all_tokenizers.py --tiktoken-encodings cl100k_base,p50k_base

        # Benchmark kitoken with specific files
        ./benchmark_all_tokenizers.py --kitoken-files /path/to/tokenizer.json

        # Benchmark all
        ./benchmark_all_tokenizers.py --simple --hf-models gpt2 --tiktoken-encodings cl100k_base --kitoken-files /path/to/tokenizer.json
    """
    # Generate batch sizes
    batch_sizes = []
    size = batch_min
    while size <= batch_max:
        batch_sizes.append(size)
        size = size * 2 if size >= 2 else 2
    if batch_sizes[-1] < batch_max:
        batch_sizes.append(batch_max)

    # Validate docs divisibility
    for bs in batch_sizes:
        if docs % bs != 0:
            print(f"Warning: {docs} documents not evenly divisible by batch size {bs}")
            break

    # Generate documents once
    print("=" * 100)
    print("TOKENIZER BENCHMARK SUITE")
    print("=" * 100)
    print(f"Documents: {docs:,} x {words:,} words")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Iterations: {iterations}")
    print("=" * 100)
    documents = generate_documents(docs, words)

    all_results = []

    # Simple tokenizer
    if simple:
        try:
            tokenizer = SimpleTokenizer()
            result = run_benchmark_suite(tokenizer, documents, batch_sizes, iterations)
            all_results.append(result)
        except Exception as e:
            print(f"\nError benchmarking simple tokenizer: {e}")
            import traceback
            traceback.print_exc()

    # HuggingFace models
    if hf_models:
        for model_name in hf_models.split(','):
            model_name = model_name.strip()
            try:
                tokenizer = HuggingFaceTokenizer(model_name)
                result = run_benchmark_suite(tokenizer, documents, batch_sizes, iterations)
                all_results.append(result)
            except Exception as e:
                print(f"\nError benchmarking HuggingFace {model_name}: {e}")

    # tiktoken encodings
    if tiktoken_encodings:
        for encoding_name in tiktoken_encodings.split(','):
            encoding_name = encoding_name.strip()
            try:
                tokenizer = TiktokenTokenizer(encoding_name)
                result = run_benchmark_suite(tokenizer, documents, batch_sizes, iterations)
                all_results.append(result)
            except Exception as e:
                print(f"\nError benchmarking tiktoken {encoding_name}: {e}")

    # kitoken files
    if kitoken_files:
        for tokenizer_json in kitoken_files.split(','):
            tokenizer_json = tokenizer_json.strip()
            try:
                # If it's not a file path, try to find it in HF cache
                if not Path(tokenizer_json).exists() and not tokenizer_json.endswith('.json'):
                    print(f"\nAuto-detecting tokenizer.json for {tokenizer_json}...")
                    tokenizer_json = find_hf_tokenizer_json(tokenizer_json)
                    print(f"Found: {tokenizer_json}")

                tokenizer = KitokenTokenizer(tokenizer_json)
                result = run_benchmark_suite(tokenizer, documents, batch_sizes, iterations)
                all_results.append(result)
            except Exception as e:
                print(f"\nError benchmarking kitoken {tokenizer_json}: {e}")
                import traceback
                traceback.print_exc()

    # Print all results
    print(f"\n{'='*120}")
    print("RESULTS")
    print("=" * 120)

    for result in all_results:
        print_results_table(result)

    # Print summary comparison if multiple tokenizers were run
    if len(all_results) > 1:
        print(f"{'='*120}")
        print("SUMMARY (Best tokens/sec for each tokenizer)")
        print("=" * 120)
        print(f"{'Tokenizer':<45} {'Best Batch':<12} {'Tokens/s':<15} {'Time (s)':<12}")
        print("-" * 120)

        for result in all_results:
            best = max(result['results'], key=lambda x: x['tokens_per_sec'])
            print(f"{result['tokenizer_name']:<45} "
                  f"{best['batch_size']:<12} "
                  f"{best['tokens_per_sec']:<15.0f} "
                  f"{best['avg_time']:<12.4f}")

        print("=" * 120)


if __name__ == "__main__":
    main()
