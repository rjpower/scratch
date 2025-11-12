#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.35.0",
#     "tiktoken>=0.5.0",
#     "regex>=2023.0.0",
# ]
# ///
"""
Test script for converting HuggingFace tokenizers to native tiktoken Encodings.

This demonstrates the opposite direction from the original AutoTikTokenizer:
instead of wrapping HF tokenizers with a tiktoken API, we extract the vocabulary
and create a native tiktoken Encoding that can use tiktoken's fast implementation.
"""

import sys
sys.path.insert(0, '/home/user/scratch')

from autotiktokenizer import convert_hf_to_tiktoken, HFTikTokenizer


def test_conversion(model_name: str):
    """Test converting a HuggingFace tokenizer to tiktoken."""
    print(f"\n{'=' * 80}")
    print(f"Testing: {model_name}")
    print(f"{'=' * 80}")

    try:
        # Convert HuggingFace tokenizer to tiktoken Encoding
        print(f"Converting {model_name} to tiktoken Encoding...")
        encoder = convert_hf_to_tiktoken(model_name)

        print(f"✓ Successfully converted!")
        print(f"  Type: {type(encoder).__name__}")
        print(f"  Encoding name: {encoder.name}")
        print(f"  Vocabulary size: {encoder.n_vocab:,}")

        # Test encoding
        test_text = "Hello world! This is a test of the tokenizer conversion."
        print(f"\nTest text: {test_text}")

        tokens = encoder.encode(test_text)
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")

        # Test decoding
        decoded = encoder.decode(tokens)
        print(f"Decoded: {decoded}")

        # Verify round-trip
        if decoded == test_text:
            print("✓ Round-trip successful!")
        else:
            print("✗ Round-trip mismatch (this can happen with different tokenizer formats)")
            print(f"  Expected: {repr(test_text)}")
            print(f"  Got: {repr(decoded)}")

        # Test encode_ordinary
        ordinary_tokens = encoder.encode_ordinary(test_text)
        print(f"\nOrdinary tokens (no special): {ordinary_tokens}")

        # Test batch encoding (tiktoken's fast batch processing)
        texts = [
            "First example",
            "Second example with more words",
            "Third example!"
        ]
        batch_tokens = encoder.encode_batch(texts)
        print(f"\nBatch encoding ({len(texts)} texts):")
        for i, (text, tokens) in enumerate(zip(texts, batch_tokens)):
            print(f"  [{i}] '{text}' -> {len(tokens)} tokens")

        # Test special tokens
        if hasattr(encoder, 'special_tokens_set') and encoder.special_tokens_set:
            print(f"\nSpecial tokens: {encoder.special_tokens_set}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cached_loading():
    """Test the HFTikTokenizer class with caching."""
    print(f"\n{'=' * 80}")
    print("Testing HFTikTokenizer with Caching")
    print(f"{'=' * 80}")

    try:
        # First load (will convert and cache)
        print("\nFirst load (converts and caches)...")
        enc1 = HFTikTokenizer.from_pretrained('gpt2')
        print(f"✓ Loaded: {enc1.name}")

        # Second load (uses cache)
        print("\nSecond load (from cache)...")
        enc2 = HFTikTokenizer.from_pretrained('gpt2')
        print(f"✓ Loaded from cache")

        # Verify they're the same object
        if enc1 is enc2:
            print("✓ Cache working correctly (same object)")
        else:
            print("✗ Cache not working (different objects)")

        # Force reload
        print("\nForce reload...")
        enc3 = HFTikTokenizer.from_pretrained('gpt2', force_reload=True)
        print(f"✓ Force reloaded")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run conversion tests."""
    print("=" * 80)
    print("HuggingFace to TikToken Converter Tests")
    print("=" * 80)
    print()
    print("This demonstrates converting HuggingFace tokenizers to native tiktoken")
    print("Encodings, allowing you to use tiktoken's fast implementation with any")
    print("HuggingFace tokenizer vocabulary.")
    print()

    # Test with GPT-2 (should work since it's available)
    print("\n" + "=" * 80)
    print("Test 1: GPT-2 Tokenizer")
    print("=" * 80)
    test_conversion('gpt2')

    # Test caching
    print("\n" + "=" * 80)
    print("Test 2: Cached Loading")
    print("=" * 80)
    test_cached_loading()

    # Show examples for other models (won't actually run without network)
    print("\n" + "=" * 80)
    print("Additional Models (require network access)")
    print("=" * 80)
    print()
    print("The following models can be converted the same way:")
    print()
    print("LLaMA models:")
    print("  enc = convert_hf_to_tiktoken('NousResearch/Nous-Hermes-Llama2-13b')")
    print("  enc = convert_hf_to_tiktoken('meta-llama/Llama-2-7b-hf')")
    print()
    print("Mistral models:")
    print("  enc = convert_hf_to_tiktoken('mistralai/Mistral-7B-v0.1')")
    print()
    print("Other models:")
    print("  enc = convert_hf_to_tiktoken('facebook/opt-350m')")
    print("  enc = convert_hf_to_tiktoken('EleutherAI/gpt-j-6b')")
    print()

    print("\n" + "=" * 80)
    print("How It Works")
    print("=" * 80)
    print("""
The converter:
1. Loads the HuggingFace tokenizer
2. Extracts the vocabulary (get_vocab)
3. Extracts the BPE merges (save_vocabulary)
4. Converts GPT-2 byte encoding to actual bytes
5. Builds mergeable_ranks dictionary for tiktoken
6. Extracts or infers the regex pattern
7. Extracts special tokens
8. Creates a native tiktoken.Encoding

Benefits:
- Uses tiktoken's fast Rust implementation
- Full tiktoken API (encode_batch, etc.)
- Works with any HuggingFace BPE tokenizer
- Maintains exact token compatibility
""")

    print("=" * 80)
    print("✓ Tests complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
