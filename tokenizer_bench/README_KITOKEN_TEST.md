# Kitoken Evaluation Test

Comprehensive test suite to evaluate [kitoken](https://github.com/Systemcluster/kitoken) against HuggingFace tokenizers, specifically for Llama-based tokenizers.

## Overview

This test validates that kitoken produces **identical tokenization results** to HuggingFace tokenizers across diverse and challenging inputs:

- **Long literary texts** (Shakespeare's Complete Works)
- **Randomly generated ASCII strings** (letters, numbers, punctuation)
- **Non-English text** (Chinese, Japanese, Korean, Arabic, Russian, Hebrew, Hindi)
- **Edge cases** (empty strings, special characters, combining marks)
- **Garbage bytes** (random byte sequences)
- **Mixed Unicode** (emojis, mathematical symbols, mixed scripts)

## Requirements

- Python 3.10 or higher
- Internet connection (for downloading Shakespeare and tokenizer files)
- The script uses `uv` for dependency management

### Dependencies (auto-installed by uv)

- `transformers>=4.35.0` - HuggingFace tokenizers
- `kitoken>=0.2.0` - The tokenizer library being tested
- `click>=8.1.0` - CLI interface
- `requests>=2.31.0` - For downloading test data

## Installation

The script is self-contained and will automatically install dependencies when run with `uv`:

```bash
chmod +x test_kitoken_evaluation.py
./test_kitoken_evaluation.py --help
```

## Usage

### Basic Usage

Run with default settings (TinyLlama tokenizer, full test suite):

```bash
./test_kitoken_evaluation.py
```

### Quick Mode

Run a reduced test suite for faster validation:

```bash
./test_kitoken_evaluation.py --quick
```

### Custom Model

Test with a different Llama-based model:

```bash
./test_kitoken_evaluation.py --model NousResearch/Llama-2-7b-hf
```

Note: Some models may require HuggingFace authentication. You can set this up with:

```bash
huggingface-cli login
```

### Using Local Tokenizer

If you have a tokenizer already downloaded, you can use a local path:

```bash
./test_kitoken_evaluation.py --model /path/to/local/tokenizer/directory
```

The directory should contain:
- `tokenizer.json` - The HuggingFace tokenizer configuration
- `tokenizer_config.json` - Tokenizer metadata
- Other tokenizer files (vocab, etc.)

## Test Cases

### 1. Shakespeare's Complete Works

Tests tokenization on long-form literary text (~5MB of text). This validates:
- Handling of long documents
- Consistent tokenization across large texts
- English language processing
- Classical literature vocabulary

The test downloads Shakespeare's complete works from Project Gutenberg and processes it in chunks.

### 2. Random ASCII Strings

Generates 100 random strings (20 in quick mode) with:
- Varying lengths (10-500 characters)
- Mixed content: letters, numbers, punctuation, whitespace
- Random patterns to catch edge cases

### 3. Non-English Text

Tests multilingual support with text in:
- **Chinese** (Simplified) - 你好世界
- **Japanese** (Hiragana/Katakana) - こんにちは
- **Korean** (Hangul) - 안녕하세요
- **Arabic** (RTL script) - مرحبا
- **Russian** (Cyrillic) - Привет
- **Hebrew** (RTL script) - שלום
- **Hindi** (Devanagari) - नमस्ते
- **Mixed scripts** - Combined multilingual text
- **Emojis** - 👋 🌍 🤖
- **Mathematical symbols** - ∀x∈ℝ, ∫∮, α β γ

### 4. Edge Cases

Special inputs designed to test boundary conditions:
- Empty strings
- Single characters (ASCII, Unicode, emojis)
- Repeated characters (1000x 'a', 500x '!')
- Very long strings (50,000+ characters)
- Control characters (`\x00`, `\x01`)
- Zero-width characters (`\u200B`, `\u200C`)
- Combining diacritical marks
- Mixed everything (all types combined)

### 5. Garbage Bytes

Random byte sequences including:
- Invalid UTF-8 sequences
- Null bytes
- Control characters (0x00-0x1F)
- High bytes (0x80-0xFF)

These are decoded with `errors='ignore'` to test robustness.

## Output

The test produces detailed output for each test category:

```
====================================================================================================
KITOKEN vs HUGGINGFACE TOKENIZER EVALUATION
====================================================================================================
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Quick mode: False
====================================================================================================

Loading tokenizers for: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Loading HuggingFace tokenizer...
HF tokenizer loaded: vocab_size=32000
Found tokenizer.json at: /path/to/tokenizer.json
Loading kitoken encoder...
Kitoken encoder loaded successfully

====================================================================================================
TEST 1: Shakespeare's Complete Works
====================================================================================================
Downloading Shakespeare's Complete Works...
Downloaded 5,458,199 characters of Shakespeare
Testing Shakespeare chunk 1/110... ✓ MATCH
Testing Shakespeare chunk 2/110... ✓ MATCH
...

====================================================================================================
TEST SUMMARY
====================================================================================================
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Tokenizer path: /path/to/tokenizer.json

Total tests: 285
Matches: 285 (100.0%)
Mismatches: 0 (0.0%)

====================================================================================================
✓ ALL TESTS PASSED - kitoken produces identical results to HuggingFace tokenizers!
====================================================================================================
```

If there are mismatches, detailed information is provided:

```
====================================================================================================
FAILURES
====================================================================================================

1. random_string_42
   Text preview: Hello world! This is a test...
   Text length: 234
   HF tokens: 45
   KT tokens: 46
   First diff at index 23: HF=1234 vs KT=5678
```

## Supported Models

The test works with any Llama-based tokenizer on HuggingFace, including:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (default, freely accessible)
- `NousResearch/Llama-2-7b-hf` (requires license agreement)
- `NousResearch/Llama-2-13b-hf`
- `meta-llama/Llama-2-7b-hf` (gated, requires authentication)
- `meta-llama/Meta-Llama-3-8B` (gated, requires authentication)
- Any other model using a Llama tokenizer

## Troubleshooting

### Network Access Issues

If you see 403 errors when downloading models:

```bash
# Authenticate with HuggingFace
huggingface-cli login

# Or use a local tokenizer
./test_kitoken_evaluation.py --model /path/to/local/tokenizer
```

### Model Not Found

Some models are gated and require accepting terms:
1. Visit the model page on HuggingFace
2. Click "Agree and access repository"
3. Run `huggingface-cli login` with your token

### Import Errors

If you see import errors, ensure you're using uv:

```bash
# Install uv if needed
pip install uv

# Run the script (dependencies auto-install)
./test_kitoken_evaluation.py
```

## Performance

Test execution times (approximate):

- **Quick mode**: ~30 seconds (reduced test suite)
- **Full mode**: ~2-5 minutes (complete test suite)

The majority of time is spent:
- Downloading Shakespeare (~5MB, one-time per run)
- Downloading tokenizer files (first run only, then cached)
- Processing large text chunks

## Implementation Details

### Token Comparison

The test compares token IDs directly:

```python
# HuggingFace
hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

# kitoken
kt_tokens = kitoken_encoder.encode(text, False)  # False = no special tokens

# Compare
assert hf_tokens == kt_tokens
```

### Special Token Handling

The test uses `add_special_tokens=False` to ensure fair comparison, as special token handling may differ between implementations.

### Error Handling

For garbage bytes, the test uses:
```python
text = data.decode('utf-8', errors='ignore')
```

This allows testing how tokenizers handle invalid UTF-8 sequences.

## Contributing

To add new test cases:

1. Add generation function (e.g., `generate_new_test_cases()`)
2. Call it in `run_tests()`
3. Process results with `compare_tokenization()`

Example:

```python
def generate_new_test_cases() -> List[str]:
    """Generate new test cases."""
    cases = [
        "test case 1",
        "test case 2",
    ]
    return cases

# In run_tests():
new_tests = generate_new_test_cases()
for i, text in enumerate(new_tests):
    result = compare_tokenization(hf_tokenizer, kitoken_encoder, text, f"new_test_{i+1}")
    all_results.append(result)
```

## License

This test script is provided as-is for evaluating kitoken compatibility with HuggingFace tokenizers.

## References

- [kitoken GitHub](https://github.com/Systemcluster/kitoken)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Project Gutenberg](https://www.gutenberg.org/) (Shakespeare source)
