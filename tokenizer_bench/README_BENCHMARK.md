# Tokenizer Benchmark Suite

Refactored and cleaned up tokenizer benchmark suite supporting HuggingFace, tiktoken, kitoken, and a simple baseline tokenizer.

## Features

- **Unified Interface**: Single benchmark script for all tokenizer types
- **Raw Performance Metrics**: Clean output with raw numbers (no excessive analysis)
- **Multiple Tokenizers**: Support for HuggingFace, tiktoken, kitoken, and simple baseline
- **Flexible Configuration**: Benchmark any combination of tokenizers
- **Batch Size Analysis**: Test performance across different batch sizes

## Quick Start

### Simple Baseline (No Network Required)

```bash
./tokenizer_benchmark.py --simple
```

### HuggingFace Tokenizers

```bash
# Single model
./tokenizer_benchmark.py --hf-models gpt2

# Multiple models
./tokenizer_benchmark.py --hf-models "gpt2,NousResearch/Nous-Hermes-Llama2-13b"
```

### tiktoken Encodings

```bash
# Single encoding
./tokenizer_benchmark.py --tiktoken-encodings cl100k_base

# Multiple encodings
./tokenizer_benchmark.py --tiktoken-encodings "cl100k_base,p50k_base"
```

### kitoken Tokenizers

```bash
# Using HF model name (auto-detects tokenizer.json from cache)
./tokenizer_benchmark.py --kitoken-files gpt2

# Using explicit path
./tokenizer_benchmark.py --kitoken-files /path/to/tokenizer.json

# Multiple tokenizers
./tokenizer_benchmark.py --kitoken-files "gpt2,NousResearch/Nous-Hermes-Llama2-13b"
```

### Comprehensive Benchmark

```bash
# Benchmark all tokenizers for GPT-2
./tokenizer_benchmark.py \
  --hf-models gpt2 \
  --tiktoken-encodings cl100k_base \
  --kitoken-files gpt2 \
  --simple

# Benchmark HuggingFace vs kitoken for Llama model
./tokenizer_benchmark.py \
  --hf-models "NousResearch/Nous-Hermes-Llama2-13b" \
  --kitoken-files "NousResearch/Nous-Hermes-Llama2-13b"
```

## Configuration Options

- `--docs, -d`: Number of documents (default: 512)
- `--words, -w`: Words per document (default: 1000)
- `--batch-min`: Minimum batch size (default: 1)
- `--batch-max`: Maximum batch size (default: 64)
- `--iterations, -i`: Iterations per batch size (default: 3)
- `--simple`: Include simple baseline tokenizer
- `--hf-models`: Comma-separated list of HuggingFace models
- `--tiktoken-encodings`: Comma-separated list of tiktoken encodings
- `--kitoken-files`: Comma-separated list of tokenizer.json paths or HF model names

## Output Format

The benchmark outputs clean tables with raw performance numbers:

```
========================================================================================================================
huggingface/gpt2
========================================================================================================================
Avg tokens/doc: 743.5
Total tokens: 380,672

Batch    Batches    Avg (s)      Min (s)      Max (s)      Docs/s       Tokens/s
------------------------------------------------------------------------------------------------------------------------
1        512        2.4567       2.4501       2.4631       208.4        154761
2        256        1.2890       1.2834       1.2945       397.2        295294
4        128        0.6987       0.6945       0.7029       732.8        544512
8        64         0.3901       0.3887       0.3916      1312.4        975584
16       32         0.2234       0.2223       0.2246      2291.7       1703421
32       16         0.1389       0.1382       0.1396      3686.4       2739851
64       8          0.1098       0.1091       0.1104      4663.2       3466982

```

## Architecture

The refactored benchmark uses a clean OOP design:

1. **TokenizerWrapper**: Abstract base class for all tokenizers
2. **Concrete Implementations**:
   - `SimpleTokenizer`: Whitespace-based baseline
   - `HuggingFaceTokenizer`: Wraps HuggingFace AutoTokenizer
   - `TiktokenTokenizer`: Wraps tiktoken encodings
   - `KitokenTokenizer`: Wraps kitoken encoders

3. **Unified Benchmark Function**: Single `benchmark_tokenizer()` function works with any tokenizer type
4. **Clean Output**: Minimal analysis, focus on raw numbers

## Comparison with Previous Version

### Before (Old)
- Separate functions for each tokenizer type
- Hardcoded HF and tiktoken support
- No kitoken support
- Excessive analysis in output
- Difficult to extend

### After (New)
- Unified interface via TokenizerWrapper
- Easy to add new tokenizer types
- Built-in kitoken support
- Clean, raw number output
- Modular and extensible

## Use Cases

### 1. Baseline Comparison
Compare your tokenizer against simple baseline:
```bash
./tokenizer_benchmark.py --simple --hf-models gpt2
```

### 2. Implementation Comparison
Compare different implementations of the same tokenizer:
```bash
./tokenizer_benchmark.py \
  --hf-models gpt2 \
  --tiktoken-encodings cl100k_base \
  --kitoken-files gpt2
```

### 3. Model Comparison
Compare different models:
```bash
./tokenizer_benchmark.py \
  --hf-models "gpt2,TinyLlama/TinyLlama-1.1B-Chat-v1.0,NousResearch/Nous-Hermes-Llama2-13b"
```

### 4. Batch Size Optimization
Find optimal batch size for your tokenizer:
```bash
./tokenizer_benchmark.py \
  --hf-models your-model \
  --batch-min 1 \
  --batch-max 128 \
  --iterations 5
```

## Notes

- First run downloads models to HuggingFace cache (~/.cache/huggingface/hub/)
- kitoken requires tokenizer.json file (auto-detected from HF cache if model name provided)
- Benchmark ensures documents are evenly divisible by batch sizes for fair comparison
- Simple tokenizer provides offline baseline without network requirements
