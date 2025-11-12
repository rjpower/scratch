#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.35.0",
#     "tiktoken>=0.5.0",
# ]
# ///
"""
Simple test script for AutoTikTokenizer to verify functionality.
"""

import sys
sys.path.insert(0, '/home/user/scratch')

from autotiktokenizer import AutoTikTokenizer


def main():
    print("=" * 80)
    print("AutoTikTokenizer - Simple Test")
    print("=" * 80)

    # Test 1: Load GPT-2 (has native tiktoken support)
    print("\nTest 1: Loading GPT-2 with native tiktoken support")
    print("-" * 80)

    encoder = AutoTikTokenizer.from_pretrained('gpt2')
    print(f"✓ Loaded encoder: {type(encoder).__name__}")
    print(f"  Encoding name: {encoder.name}")
    print(f"  Vocabulary size: {encoder.n_vocab:,}")

    # Test encoding/decoding
    sample_text = "Hello world! This is a test."
    tokens = encoder.encode(sample_text)
    decoded = encoder.decode(tokens)

    print(f"\n  Original: {sample_text}")
    print(f"  Tokens: {tokens}")
    print(f"  Token count: {len(tokens)}")
    print(f"  Decoded: {decoded}")
    print(f"  Round-trip successful: {decoded == sample_text}")

    # Test 2: Batch encoding
    print("\n\nTest 2: Batch encoding")
    print("-" * 80)

    texts = [
        "First example.",
        "Second example with more words.",
        "Third!"
    ]

    batch_tokens = encoder.encode_batch(texts)
    print(f"Encoded {len(texts)} texts:")
    for i, (text, tokens) in enumerate(zip(texts, batch_tokens)):
        print(f"  [{i}] '{text}' -> {len(tokens)} tokens: {tokens}")

    batch_decoded = encoder.decode_batch(batch_tokens)
    print(f"\nDecoded {len(batch_decoded)} texts:")
    for i, (original, decoded) in enumerate(zip(texts, batch_decoded)):
        match = "✓" if original == decoded else "✗"
        print(f"  [{i}] {match} '{decoded}'")

    # Test 3: encode_ordinary (no special tokens)
    print("\n\nTest 3: encode_ordinary (without special tokens)")
    print("-" * 80)

    text = "Test text"
    tokens_with_special = encoder.encode(text)
    tokens_ordinary = encoder.encode_ordinary(text)

    print(f"  Text: {text}")
    print(f"  With special tokens: {tokens_with_special} ({len(tokens_with_special)} tokens)")
    print(f"  Without special tokens: {tokens_ordinary} ({len(tokens_ordinary)} tokens)")

    # Test 4: List supported models
    print("\n\nTest 4: List models with native tiktoken support")
    print("-" * 80)

    models = AutoTikTokenizer.list_models()
    print(f"Found {len(models)} models with native tiktoken support:")
    for model in sorted(models):
        encoding = AutoTikTokenizer.get_encoding_name(model)
        print(f"  • {model:<25} -> {encoding}")

    # Test 5: Force HuggingFace wrapper
    print("\n\nTest 5: Force HuggingFace wrapper (even for native models)")
    print("-" * 80)

    encoder_hf = AutoTikTokenizer.from_pretrained('gpt2', force_hf=True)
    print(f"✓ Loaded encoder with force_hf=True: {type(encoder_hf).__name__}")
    print(f"  Encoding name: {encoder_hf.name}")

    tokens_hf = encoder_hf.encode(sample_text)
    decoded_hf = encoder_hf.decode(tokens_hf)
    print(f"\n  Original: {sample_text}")
    print(f"  Tokens: {tokens_hf}")
    print(f"  Decoded: {decoded_hf}")
    print(f"  Round-trip successful: {decoded_hf == sample_text}")

    # Compare native vs wrapper
    print(f"\n  Token count comparison:")
    print(f"    Native tiktoken: {len(tokens)} tokens")
    print(f"    HF wrapper: {len(tokens_hf)} tokens")
    print(f"    Tokens match: {tokens == tokens_hf}")

    print("\n" + "=" * 80)
    print("✓ All tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
