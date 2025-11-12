#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "tiktoken>=0.5.0",
# ]
# ///
"""
Test showing tiktoken encoding with chicken document.

This demonstrates using tiktoken directly (which works offline)
to show what the transparent tokenizer would do.
"""

import tiktoken


def main():
    """Test tiktoken with chicken document."""

    print("=" * 80)
    print("TikToken Encoding Test - Chicken Document")
    print("=" * 80)
    print()
    print("This demonstrates the fast tiktoken encoding that would be used")
    print("internally by our transparent HuggingFace-compatible tokenizer.")
    print()

    # Load GPT-2 encoding (works offline, built into tiktoken)
    print("Loading tiktoken GPT-2 encoding...")
    encoder = tiktoken.get_encoding("gpt2")
    print(f"✓ Loaded encoding: {encoder.name}")
    print(f"  Vocabulary size: {encoder.n_vocab:,}")
    print()

    # Sample document about chickens
    chicken_document = """Chickens are fascinating domesticated birds that have been bred for thousands of years.
These remarkable creatures are descendants of wild jungle fowl from Southeast Asia. A typical chicken
has a complex social structure and can recognize over 100 individual chickens in their flock. They
communicate using more than 30 different vocalizations, each with a specific meaning. Chickens have
excellent color vision and can see a wider range of colors than humans. Mother hens talk to their
chicks while they are still in the egg, and the chicks chirp back from inside the shell. Chickens
are omnivores and will eat insects, seeds, and even small mice if given the chance. A single hen can
lay over 300 eggs per year, though this varies by breed. Chickens take dust baths to keep their
feathers clean and free from parasites. They have been shown to possess mathematical abilities and
can perform basic arithmetic. Chickens dream while they sleep, just like humans and other mammals."""

    print("=" * 80)
    print("Sample Document (about chickens)")
    print("=" * 80)
    print(chicken_document)
    print()

    print("=" * 80)
    print("Document Statistics")
    print("=" * 80)
    print(f"Characters: {len(chicken_document)}")
    print(f"Words: {len(chicken_document.split())}")
    print(f"Lines: {chicken_document.count(chr(10)) + 1}")
    print()

    # Encode the document
    print("=" * 80)
    print("Encoding with TikToken...")
    print("=" * 80)
    tokens = encoder.encode(chicken_document)

    print(f"✓ Encoded successfully!")
    print(f"  Token count: {len(tokens)}")
    print(f"  Compression ratio: {len(chicken_document) / len(tokens):.2f} chars/token")
    print()

    # Show token details
    print("=" * 80)
    print("Token Sequence (first 30 tokens)")
    print("=" * 80)
    for i in range(min(30, len(tokens))):
        token_id = tokens[i]
        token_bytes = encoder.decode_single_token_bytes(token_id)
        token_str = token_bytes.decode('utf-8', errors='replace')
        # Show printable representation
        if '\n' in token_str:
            token_repr = '<newline>'
        else:
            token_repr = repr(token_str)[1:-1]  # Remove quotes
        print(f"  [{i:3d}] Token ID: {token_id:5d}  |  Text: {token_repr:20s}")
    print(f"  ...")
    print()

    # Show all tokens in compact format
    print("=" * 80)
    print("Full Token Sequence")
    print("=" * 80)
    print("Token IDs:")
    print("-" * 80)
    # Print in rows of 15
    for i in range(0, len(tokens), 15):
        chunk = tokens[i:min(i+15, len(tokens))]
        token_str = ", ".join(f"{t:5d}" for t in chunk)
        print(f"  [{i:3d}-{i+len(chunk)-1:3d}] {token_str}")
    print()

    # Decode individual tokens to show what each represents
    print("=" * 80)
    print("Token-by-Token Breakdown")
    print("=" * 80)
    print("Showing each token and its text:")
    print("-" * 80)

    current_pos = 0
    for i, token_id in enumerate(tokens[:50]):  # Show first 50 tokens
        token_bytes = encoder.decode_single_token_bytes(token_id)
        token_str = token_bytes.decode('utf-8', errors='replace')

        # Clean up display
        display_str = token_str.replace('\n', '\\n').replace('\r', '\\r')
        if not display_str.strip():
            display_str = '<space>' if token_str == ' ' else '<whitespace>'

        print(f"  {i:3d}. [{token_id:5d}] '{display_str}'")

    if len(tokens) > 50:
        print(f"  ... ({len(tokens) - 50} more tokens)")
    print()

    # Decode back to verify
    print("=" * 80)
    print("Decoding Verification")
    print("=" * 80)
    decoded = encoder.decode(tokens)

    print(f"✓ Decoded successfully!")
    print(f"  Decoded length: {len(decoded)} chars")
    print(f"  Original length: {len(chicken_document)} chars")
    print(f"  Perfect match: {decoded == chicken_document}")
    print()

    if decoded != chicken_document:
        print("Differences:")
        for i, (c1, c2) in enumerate(zip(chicken_document, decoded)):
            if c1 != c2:
                print(f"  Position {i}: '{c1}' != '{c2}'")

    # Show decoded text sample
    print("Decoded text (first 200 chars):")
    print("-" * 80)
    print(decoded[:200])
    print("..." if len(decoded) > 200 else "")
    print()

    # Test with individual chicken facts
    print("=" * 80)
    print("Batch Encoding Test - Individual Chicken Facts")
    print("=" * 80)

    chicken_facts = [
        "Chickens can run up to 9 miles per hour.",
        "A chicken's heart beats about 300 times per minute.",
        "Chickens have three eyelids.",
        "The longest recorded chicken flight lasted 13 seconds.",
        "Chickens can remember and recognize human faces.",
        "Baby chicks can learn from watching their mothers.",
        "Chickens experience REM sleep and dream.",
        "A mother hen turns her eggs about 50 times per day.",
    ]

    print(f"\nEncoding {len(chicken_facts)} chicken facts:")
    print("-" * 80)

    for i, fact in enumerate(chicken_facts, 1):
        tokens = encoder.encode(fact)
        print(f"  {i}. '{fact}'")
        print(f"     Tokens: {tokens}")
        print(f"     Count: {len(tokens)} tokens")
        print()

    # Batch encoding (faster!)
    print("Using batch encoding (tiktoken's fast API):")
    print("-" * 80)
    batch_tokens = encoder.encode_batch(chicken_facts)

    print(f"\n✓ Encoded {len(chicken_facts)} facts in batch:")
    for i, (fact, tokens) in enumerate(zip(chicken_facts, batch_tokens), 1):
        print(f"  {i}. {len(tokens):2d} tokens: '{fact[:50]}{'...' if len(fact) > 50 else ''}'")
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"""
Document: Chicken information text
  • Characters: {len(chicken_document)}
  • Words: {len(chicken_document.split())}
  • Tokens: {len(encoder.encode(chicken_document))}
  • Compression: {len(chicken_document) / len(encoder.encode(chicken_document)):.2f} chars/token

This same fast encoding would be used internally by:
  • convert_hf_to_tiktoken('meta-llama/Llama-2-7b-hf')
  • TransparentTikTokenizer('NousResearch/Nous-Hermes-Llama2-13b')

The user gets HuggingFace's familiar API with tiktoken's speed!
""")

    print("=" * 80)
    print("✓ Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
