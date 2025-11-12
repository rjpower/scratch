#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.35.0",
#     "tiktoken>=0.5.0",
# ]
# ///
"""
Example usage of AutoTikTokenizer.

This script demonstrates how to use AutoTikTokenizer to load various
HuggingFace models as tiktoken-compatible encoders.
"""

from autotiktokenizer import AutoTikTokenizer


def test_model(model_name: str, sample_text: str):
    """Test a model with sample text."""
    print(f"\n{'=' * 80}")
    print(f"Testing: {model_name}")
    print(f"{'=' * 80}")

    try:
        # Load the tokenizer
        encoder = AutoTikTokenizer.from_pretrained(model_name)

        # Display info
        print(f"Encoder type: {type(encoder).__name__}")
        if hasattr(encoder, 'name'):
            print(f"Encoding name: {encoder.name}")
        print(f"Vocabulary size: {encoder.n_vocab:,}")

        # Encode
        tokens = encoder.encode(sample_text)
        print(f"\nOriginal text: {sample_text}")
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")

        # Decode
        decoded = encoder.decode(tokens)
        print(f"Decoded text: {decoded}")

        # Verify round-trip
        if decoded == sample_text:
            print("✓ Round-trip encoding/decoding successful!")
        else:
            print("✗ Round-trip mismatch!")
            print(f"  Expected: {repr(sample_text)}")
            print(f"  Got: {repr(decoded)}")

        # Test encode_ordinary (without special tokens)
        ordinary_tokens = encoder.encode_ordinary(sample_text)
        print(f"\nOrdinary tokens (no special): {ordinary_tokens}")
        print(f"Ordinary token count: {len(ordinary_tokens)}")

        # Test batch encoding
        texts = [sample_text, "Another test.", "Final example!"]
        batch_tokens = encoder.encode_batch(texts)
        print(f"\nBatch encoding ({len(texts)} texts):")
        for i, (text, tokens) in enumerate(zip(texts, batch_tokens)):
            print(f"  [{i}] '{text}' -> {len(tokens)} tokens")

        # Test batch decoding
        batch_decoded = encoder.decode_batch(batch_tokens)
        print(f"\nBatch decoding:")
        for i, (original, decoded) in enumerate(zip(texts, batch_decoded)):
            match = "✓" if original == decoded else "✗"
            print(f"  [{i}] {match} '{decoded}'")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run examples with different models."""
    print("AutoTikTokenizer Examples")
    print("=" * 80)

    sample_text = "Hello world! This is a test of the AutoTikTokenizer library."

    # Test models with native tiktoken support
    print("\n" + "=" * 80)
    print("Models with Native TikToken Support")
    print("=" * 80)

    native_models = [
        "gpt2",
        "gpt2-medium",
        "distilgpt2",
    ]

    for model in native_models:
        test_model(model, sample_text)

    # List all supported models
    print("\n" + "=" * 80)
    print("All Models with Native TikToken Support")
    print("=" * 80)
    supported_models = AutoTikTokenizer.list_models()
    for model in supported_models:
        encoding = AutoTikTokenizer.get_encoding_name(model)
        print(f"  {model:<25} -> {encoding}")

    # Test model without native tiktoken support (uses HuggingFace wrapper)
    print("\n" + "=" * 80)
    print("Models Using HuggingFace Wrapper (No Native TikToken)")
    print("=" * 80)

    wrapper_models = [
        "meta-llama/Llama-2-7b-hf",  # LLaMA model
        "mistralai/Mistral-7B-v0.1",  # Mistral model
    ]

    for model in wrapper_models:
        print(f"\nAttempting to load: {model}")
        try:
            encoder = AutoTikTokenizer.from_pretrained(model)
            print(f"✓ Successfully loaded {model} with wrapper")
            print(f"  Encoder type: {type(encoder).__name__}")
            print(f"  Encoding name: {encoder.name}")
        except Exception as e:
            print(f"✗ Could not load {model}: {e}")
            print(f"  (This is expected if the model is not downloaded)")

    # Test force_hf option
    print("\n" + "=" * 80)
    print("Force HuggingFace Wrapper (even for models with native support)")
    print("=" * 80)

    print("\nLoading gpt2 with force_hf=True")
    try:
        encoder = AutoTikTokenizer.from_pretrained("gpt2", force_hf=True)
        print(f"✓ Loaded with wrapper")
        print(f"  Encoder type: {type(encoder).__name__}")
        tokens = encoder.encode("Test")
        print(f"  Test encoding works: {tokens}")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
