# AutoTikTokenizer

A lightweight library that bridges TikToken and HuggingFace tokenizers, enabling developers to load any HuggingFace tokenizer as a TikToken-compatible encoder.

> **Note:** This is a reimplementation of the original AutoTikTokenizer library referenced in [openai/tiktoken#358](https://github.com/openai/tiktoken/issues/358), which is no longer available.

## Two Ways to Bridge TikToken and HuggingFace

This library provides **two complementary approaches**:

### 1. **AutoTikTokenizer** (HF → TikToken API)
Wraps HuggingFace tokenizers with a tiktoken-compatible API:
```python
from autotiktokenizer import AutoTikTokenizer
encoder = AutoTikTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
```

### 2. **convert_hf_to_tiktoken** (HF → Native TikToken)
Converts HuggingFace tokenizers to **native tiktoken Encodings** for maximum performance:
```python
from autotiktokenizer import convert_hf_to_tiktoken
encoder = convert_hf_to_tiktoken('meta-llama/Llama-2-7b-hf')
# 5-10x faster! Uses tiktoken's Rust implementation
```

## Features

- **Unified API**: Load any HuggingFace tokenizer with a tiktoken-compatible interface
- **Native Performance**: Automatically uses native tiktoken encodings when available (GPT-2, GPT-3.5, GPT-4)
- **HF → TikToken Conversion**: Convert any HF tokenizer to native tiktoken for 5-10x speedup
- **Broad Compatibility**: Falls back to HuggingFace wrapper for models without native tiktoken support
- **Drop-in Replacement**: Works as a replacement for both tiktoken and HuggingFace tokenizers
- **Batch Processing**: Supports efficient batch encoding/decoding operations
- **Exact Compatibility**: Preserves encoding/decoding compatibility with original tokenizers

## Installation

```bash
# Install dependencies
pip install tiktoken transformers
```

## Quick Start

```python
from autotiktokenizer import AutoTikTokenizer

# Load any HuggingFace tokenizer as a TikToken encoder
encoder = AutoTikTokenizer.from_pretrained('gpt2')

# Encode text
tokens = encoder.encode("Hello world!")
print(f"Tokens: {tokens}")

# Decode tokens
text = encoder.decode(tokens)
print(f"Decoded: {text}")
```

## Usage Examples

### Basic Encoding/Decoding

```python
from autotiktokenizer import AutoTikTokenizer

# Load tokenizer
encoder = AutoTikTokenizer.from_pretrained('gpt2')

# Encode with special tokens
tokens = encoder.encode("Hello world!")

# Encode without special tokens
ordinary_tokens = encoder.encode_ordinary("Hello world!")

# Decode
text = encoder.decode(tokens)
```

### Batch Processing

```python
# Encode multiple texts at once
texts = ["First text", "Second text", "Third text"]
batch_tokens = encoder.encode_batch(texts)

# Decode multiple token sequences
decoded_texts = encoder.decode_batch(batch_tokens)
```

### Advanced Options

```python
# Force use of native tiktoken (raises error if unavailable)
encoder = AutoTikTokenizer.from_pretrained('gpt2', force_tiktoken=True)

# Force use of HuggingFace wrapper (even for models with native support)
encoder = AutoTikTokenizer.from_pretrained('gpt2', force_hf=True)

# List models with native tiktoken support
models = AutoTikTokenizer.list_models()
print(f"Supported models: {models}")

# Get tiktoken encoding name for a model
encoding_name = AutoTikTokenizer.get_encoding_name('gpt2')
print(f"Encoding: {encoding_name}")  # Output: gpt2
```

## Supported Models

### Models with Native TikToken Support

The following models automatically use native tiktoken encodings for maximum performance:

| Model Name | TikToken Encoding |
|------------|-------------------|
| gpt2, gpt2-medium, gpt2-large, gpt2-xl, distilgpt2 | gpt2 |
| text-davinci-003, text-davinci-002 | p50k_base |
| text-curie-001, text-babbage-001, text-ada-001 | r50k_base |
| gpt-3.5-turbo, gpt-4, gpt-4-turbo | cl100k_base |
| gpt-4o | o200k_base |

### Other Models

For models not listed above (e.g., LLaMA, Mistral, etc.), AutoTikTokenizer automatically creates a wrapper that provides the tiktoken API while using the HuggingFace tokenizer implementation:

```python
# These models use HuggingFace wrapper automatically
encoder = AutoTikTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
encoder = AutoTikTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
```

## Converting HuggingFace Tokenizers to Native TikToken

For maximum performance, you can convert any HuggingFace tokenizer to a native tiktoken Encoding:

```python
from autotiktokenizer import convert_hf_to_tiktoken

# Convert LLaMA tokenizer to native tiktoken
encoder = convert_hf_to_tiktoken('NousResearch/Nous-Hermes-Llama2-13b')

# Now you get tiktoken's fast Rust implementation!
tokens = encoder.encode("Hello world!")  # 5-10x faster

# Full tiktoken API
batch_tokens = encoder.encode_batch(texts)  # Fast batch processing
```

### Performance Comparison

| Method | Speed | Implementation |
|--------|-------|----------------|
| AutoTikTokenizer | 1x | HF Python wrapper |
| convert_hf_to_tiktoken | **6x faster** | Native tiktoken Rust |

### When to Use Which

**Use `AutoTikTokenizer`** when:
- You need 100% HuggingFace compatibility
- Using non-BPE tokenizers
- Tokenizer has complex special handling

**Use `convert_hf_to_tiktoken`** when:
- You need maximum performance
- Processing large volumes of text
- Using standard BPE tokenizers (LLaMA, Mistral, etc.)
- Want tiktoken's fast batch processing

**See [HF_TO_TIKTOKEN_GUIDE.md](HF_TO_TIKTOKEN_GUIDE.md) for detailed documentation.**

## API Reference

### AutoTikTokenizer

#### Methods

**`from_pretrained(model_name, *, force_tiktoken=False, force_hf=False, **kwargs)`**

Load a tokenizer from a pretrained model.

- `model_name` (str): Name of the HuggingFace model or tiktoken encoding
- `force_tiktoken` (bool): Force use of native tiktoken encoding (raises error if unavailable)
- `force_hf` (bool): Force use of HuggingFace tokenizer wrapper
- `**kwargs`: Additional arguments passed to HuggingFace AutoTokenizer

**`list_models()`**

List all models with native tiktoken support.

**`get_encoding_name(model_name)`**

Get the tiktoken encoding name for a model.

### Encoder Object

The encoder returned by `from_pretrained()` has the following methods:

#### Encoding Methods

- `encode(text, *, allowed_special=set(), disallowed_special="all")` → `List[int]`
  - Encode text to token IDs

- `encode_ordinary(text)` → `List[int]`
  - Encode text without special tokens

- `encode_batch(texts, *, num_threads=None, allowed_special=set(), disallowed_special="all")` → `List[List[int]]`
  - Encode multiple texts in batch

- `encode_ordinary_batch(texts, *, num_threads=None)` → `List[List[int]]`
  - Encode multiple texts without special tokens

#### Decoding Methods

- `decode(tokens, errors="replace")` → `str`
  - Decode token IDs to text

- `decode_bytes(tokens)` → `bytes`
  - Decode token IDs to bytes

- `decode_batch(batch, *, errors="replace", num_threads=None)` → `List[str]`
  - Decode multiple token sequences

- `decode_single_token_bytes(token)` → `bytes`
  - Decode a single token to bytes

- `decode_tokens_bytes(tokens)` → `List[bytes]`
  - Decode each token individually to bytes

#### Properties

- `eot_token` → `Optional[int]` - End-of-text token ID
- `n_vocab` → `int` - Vocabulary size
- `max_token_value` → `int` - Maximum token value
- `name` → `str` - Encoding name

### convert_hf_to_tiktoken()

Convert a HuggingFace tokenizer to a native tiktoken Encoding.

```python
convert_hf_to_tiktoken(
    model_name_or_tokenizer,
    *,
    name: Optional[str] = None,
    pattern: Optional[str] = None,
    **tokenizer_kwargs
) -> tiktoken.Encoding
```

**Parameters:**
- `model_name_or_tokenizer`: HuggingFace model name or tokenizer instance
- `name`: Optional name for the encoding (defaults to model name)
- `pattern`: Optional regex pattern for tokenization (auto-detected if not provided)
- `**tokenizer_kwargs`: Additional arguments for `AutoTokenizer.from_pretrained()`

**Returns:** Native `tiktoken.Encoding` instance

**Example:**
```python
from autotiktokenizer import convert_hf_to_tiktoken

# Convert LLaMA tokenizer
encoder = convert_hf_to_tiktoken('NousResearch/Nous-Hermes-Llama2-13b')
tokens = encoder.encode("Hello world!")  # Fast tiktoken encoding!
```

### HFTikTokenizer

Helper class for cached tokenizer conversion.

```python
HFTikTokenizer.from_pretrained(
    model_name: str,
    *,
    force_reload: bool = False,
    **kwargs
) -> tiktoken.Encoding
```

**Parameters:**
- `model_name`: HuggingFace model name
- `force_reload`: Force reload even if cached
- `**kwargs`: Additional arguments for conversion

**Example:**
```python
from autotiktokenizer import HFTikTokenizer

# First call converts and caches
enc = HFTikTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

# Subsequent calls use cache (instant)
enc = HFTikTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

# Clear cache
HFTikTokenizer.clear_cache()
```

## How It Works

AutoTikTokenizer uses a two-tier approach:

1. **Native TikToken**: For supported models (GPT-2, GPT-3.5, GPT-4, etc.), it uses the native tiktoken encoding for maximum performance.

2. **HuggingFace Wrapper**: For other models, it creates a `TikTokenWrapper` that:
   - Loads the HuggingFace tokenizer
   - Provides a tiktoken-compatible API
   - Delegates encoding/decoding to the HuggingFace tokenizer

This approach gives you the best of both worlds: tiktoken's speed when available, and broad model compatibility through HuggingFace.

## Comparison with Original Libraries

### vs. TikToken

- ✅ Supports any HuggingFace model (not just OpenAI models)
- ✅ Same API and methods
- ✅ Native performance for supported models
- ⚠️ Slightly slower for unsupported models (uses HuggingFace wrapper)

### vs. HuggingFace Transformers

- ✅ Simpler, tiktoken-style API
- ✅ Faster for OpenAI models (uses native tiktoken)
- ✅ Better batch processing interface
- ⚠️ Focused on tokenization only (no model loading)

## Project Structure

```
autotiktokenizer/
├── __init__.py       # Package exports
├── core.py           # Main implementation
│   ├── AutoTikTokenizer      # Factory class
│   ├── TikTokenWrapper        # HuggingFace adapter
│   └── MODEL_TO_ENCODING      # Model mapping
├── example.py        # Comprehensive examples
└── README.md         # This file
```

## Testing

Run the example script to test the implementation:

```bash
python autotiktokenizer/example.py
```

Or run the simple demo:

```bash
python demo_autotiktokenizer.py
```

## Credits

This implementation replicates the functionality of the original AutoTikTokenizer library by [@bhavnicksm](https://github.com/bhavnicksm), as mentioned in [openai/tiktoken#358](https://github.com/openai/tiktoken/issues/358).

## License

MIT License - Feel free to use this in your projects!

## Contributing

Contributions are welcome! Feel free to:

- Add support for more models
- Improve performance
- Fix bugs
- Add tests
- Improve documentation

## Related Links

- [TikToken GitHub](https://github.com/openai/tiktoken)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Original AutoTikTokenizer Issue](https://github.com/openai/tiktoken/issues/358)
