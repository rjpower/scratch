#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.35.0",
#     "kitoken>=0.2.0",
#     "click>=8.1.0",
#     "requests>=2.31.0",
# ]
# ///
"""
Comprehensive test to evaluate kitoken against HuggingFace tokenizers.

This test validates that kitoken produces identical tokenization results to
HuggingFace tokenizers for the Nous/Llama tokenizer across diverse inputs:
- Long literary texts (Shakespeare)
- Randomly generated strings
- Non-English text (Chinese, Japanese, Arabic, Russian, etc.)
- Garbage bytes and edge cases
"""

import os
import random
import string
import tempfile
import click
import requests
from pathlib import Path
from typing import List, Tuple, Dict
from transformers import AutoTokenizer


def download_shakespeare() -> str:
    """Download Shakespeare's complete works for testing."""
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    print("Downloading Shakespeare's Complete Works...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    text = response.text
    # Remove Project Gutenberg header/footer (approximate)
    lines = text.split('\n')
    # Start after header (usually around line 100)
    start_idx = 0
    for i, line in enumerate(lines):
        if 'START OF THE PROJECT GUTENBERG' in line.upper():
            start_idx = i + 1
            break
    # End before footer
    end_idx = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if 'END OF THE PROJECT GUTENBERG' in lines[i].upper():
            end_idx = i
            break

    clean_text = '\n'.join(lines[start_idx:end_idx])
    print(f"Downloaded {len(clean_text):,} characters of Shakespeare")
    return clean_text


def generate_random_strings(count: int = 100) -> List[str]:
    """Generate random ASCII strings of varying lengths."""
    strings = []
    for _ in range(count):
        length = random.randint(10, 500)
        # Mix of letters, numbers, punctuation
        chars = string.ascii_letters + string.digits + string.punctuation + ' \n\t'
        random_str = ''.join(random.choice(chars) for _ in range(length))
        strings.append(random_str)
    print(f"Generated {count} random strings")
    return strings


def generate_non_english_text() -> List[str]:
    """Generate test strings in various non-English languages."""
    texts = [
        # Chinese
        "你好世界！这是一个测试字符串。人工智能正在改变世界。",
        "机器学习是计算机科学的一个分支，它使计算机能够在没有明确编程的情况下学习。",
        "深度学习神经网络自然语言处理计算机视觉",

        # Japanese
        "こんにちは世界！これはテスト文字列です。",
        "人工知能は世界を変えています。機械学習とディープラーニング。",
        "自然言語処理とコンピュータビジョンは重要な研究分野です。",

        # Korean
        "안녕하세요 세계! 이것은 테스트 문자열입니다.",
        "인공지능은 세상을 변화시키고 있습니다.",

        # Arabic
        "مرحبا بالعالم! هذا نص تجريبي.",
        "الذكاء الاصطناعي يغير العالم. التعلم الآلي والتعلم العميق.",

        # Russian
        "Привет, мир! Это тестовая строка.",
        "Искусственный интеллект меняет мир. Машинное обучение и глубокое обучение.",

        # Hebrew
        "שלום עולם! זהו מחרוזת בדיקה.",
        "בינה מלאכותית משנה את העולם.",

        # Hindi
        "नमस्ते दुनिया! यह एक परीक्षण स्ट्रिंग है।",
        "कृत्रिम बुद्धिमत्ता दुनिया को बदल रही है।",

        # Mixed scripts
        "Hello 世界 مرحبا Привет שלום नमस्ते 안녕하세요",
        "AI人工智能الذكاء الاصطناعيИскусственный интеллект",

        # Emojis and special Unicode
        "Hello! 👋 🌍 🤖 💻 ✨",
        "Testing emojis: 😀😃😄😁😆😅🤣😂🙂🙃😉😊",

        # Mathematical symbols
        "∀x∈ℝ: x²≥0 ∧ ∑ᵢ₌₁ⁿ i = n(n+1)/2",
        "α, β, γ, δ, ε, ζ, η, θ ∫∮∯∰ ∂∇∆",
    ]
    print(f"Generated {len(texts)} non-English text samples")
    return texts


def generate_garbage_bytes() -> List[bytes]:
    """Generate random garbage byte sequences."""
    garbage = []
    for _ in range(50):
        length = random.randint(10, 200)
        # Generate random bytes including null bytes and control characters
        random_bytes = bytes(random.randint(0, 255) for _ in range(length))
        garbage.append(random_bytes)
    print(f"Generated {len(garbage)} garbage byte sequences")
    return garbage


def generate_edge_cases() -> List[str]:
    """Generate edge case strings."""
    cases = [
        # Empty and whitespace
        "",
        " ",
        "\n",
        "\t",
        "   \n\t  ",

        # Single characters
        "a",
        "1",
        "!",
        "中",
        "🎉",

        # Repeated characters
        "a" * 1000,
        "!" * 500,
        "中" * 100,

        # Very long strings
        "The quick brown fox jumps over the lazy dog. " * 1000,

        # Special patterns
        "\x00\x01\x02\x03",  # Control characters
        "\u200B\u200C\u200D",  # Zero-width characters
        "A\u0301",  # Combining diacritical marks

        # Mixed everything
        "Test123!@# 中文 العربية Русский 🌟\n\tमिश्रित",
    ]
    print(f"Generated {len(cases)} edge case strings")
    return cases


def load_tokenizers(model_name_or_path: str) -> Tuple:
    """
    Load both HuggingFace and kitoken tokenizers.

    Args:
        model_name_or_path: Either a HuggingFace model name (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
                           or a local path to a directory containing tokenizer files

    Returns:
        Tuple of (hf_tokenizer, kitoken_encoder, tokenizer_json_path)
    """
    print(f"\nLoading tokenizers for: {model_name_or_path}")

    # Check if it's a local path
    local_path = Path(model_name_or_path)
    is_local = local_path.exists() and local_path.is_dir()

    # Load HuggingFace tokenizer
    print("Loading HuggingFace tokenizer...")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    print(f"HF tokenizer loaded: vocab_size={hf_tokenizer.vocab_size}")

    # Get the tokenizer.json file path
    if is_local:
        # Use the local path
        tokenizer_json_files = list(local_path.glob("tokenizer.json"))
        if not tokenizer_json_files:
            raise FileNotFoundError(f"Could not find tokenizer.json in {model_name_or_path}")
        tokenizer_json_path = str(tokenizer_json_files[0])
        print(f"Found tokenizer.json at: {tokenizer_json_path}")
    else:
        # HuggingFace caches models in ~/.cache/huggingface/hub/
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

        # Find the model directory
        model_dirs = list(cache_dir.glob(f"models--{model_name_or_path.replace('/', '--')}*"))
        if not model_dirs:
            raise FileNotFoundError(f"Could not find cached model directory for {model_name_or_path}")

        model_dir = model_dirs[0]
        # Look for tokenizer.json in snapshots
        tokenizer_json_files = list(model_dir.glob("**/tokenizer.json"))
        if not tokenizer_json_files:
            raise FileNotFoundError(f"Could not find tokenizer.json for {model_name_or_path}")

        tokenizer_json_path = str(tokenizer_json_files[0])
        print(f"Found tokenizer.json at: {tokenizer_json_path}")

    # Load kitoken encoder
    print("Loading kitoken encoder...")
    try:
        from kitoken import Kitoken
        kitoken_encoder = Kitoken.from_file(tokenizer_json_path)
        print("Kitoken encoder loaded successfully")
    except Exception as e:
        print(f"Error loading kitoken: {e}")
        raise

    return hf_tokenizer, kitoken_encoder, tokenizer_json_path


def compare_tokenization(hf_tokenizer, kitoken_encoder, text: str, test_name: str = "") -> Dict:
    """
    Compare tokenization results between HF and kitoken.

    Returns:
        Dict with comparison results including whether they match.
    """
    # HuggingFace tokenization
    hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

    # kitoken tokenization
    # Note: kitoken.encode() returns token IDs directly
    try:
        # Encode with kitoken (add_special_tokens equivalent might differ)
        kt_tokens = kitoken_encoder.encode(text, False)  # False = no special tokens
        kt_tokens = list(kt_tokens)  # Convert to list if needed
    except Exception as e:
        return {
            'test_name': test_name,
            'text_preview': text[:100] + ('...' if len(text) > 100 else ''),
            'text_length': len(text),
            'match': False,
            'error': str(e),
            'hf_tokens': hf_tokens,
            'kt_tokens': None,
        }

    # Compare
    match = hf_tokens == kt_tokens

    result = {
        'test_name': test_name,
        'text_preview': text[:100] + ('...' if len(text) > 100 else ''),
        'text_length': len(text),
        'match': match,
        'hf_tokens': hf_tokens,
        'kt_tokens': kt_tokens,
        'hf_token_count': len(hf_tokens),
        'kt_token_count': len(kt_tokens),
    }

    if not match:
        # Find first difference
        for i, (h, k) in enumerate(zip(hf_tokens, kt_tokens)):
            if h != k:
                result['first_diff_index'] = i
                result['first_diff_hf'] = h
                result['first_diff_kt'] = k
                break

        if len(hf_tokens) != len(kt_tokens):
            result['length_mismatch'] = True

    return result


def compare_tokenization_bytes(hf_tokenizer, kitoken_encoder, data: bytes, test_name: str = "") -> Dict:
    """
    Compare tokenization for byte sequences.

    Many tokenizers can handle bytes directly or after decoding with errors='ignore'.
    """
    try:
        # Try to decode with error handling
        text = data.decode('utf-8', errors='ignore')
        return compare_tokenization(hf_tokenizer, kitoken_encoder, text, test_name)
    except Exception as e:
        return {
            'test_name': test_name,
            'text_preview': str(data[:100]),
            'text_length': len(data),
            'match': False,
            'error': f"Byte decode error: {e}",
        }


def run_tests(model_name: str, quick: bool = False) -> Dict:
    """Run comprehensive tokenization tests."""
    print("="*100)
    print("KITOKEN vs HUGGINGFACE TOKENIZER EVALUATION")
    print("="*100)
    print(f"Model: {model_name}")
    print(f"Quick mode: {quick}")
    print("="*100)

    # Load tokenizers
    hf_tokenizer, kitoken_encoder, tokenizer_path = load_tokenizers(model_name)

    all_results = []

    # Test 1: Shakespeare (long literary text)
    print("\n" + "="*100)
    print("TEST 1: Shakespeare's Complete Works")
    print("="*100)
    shakespeare = download_shakespeare()

    # Test on chunks of Shakespeare
    chunk_size = 10000 if quick else 50000
    chunks = [shakespeare[i:i+chunk_size] for i in range(0, len(shakespeare), chunk_size)]
    chunks = chunks[:5 if quick else len(chunks)]  # Limit chunks in quick mode

    for i, chunk in enumerate(chunks):
        print(f"Testing Shakespeare chunk {i+1}/{len(chunks)}...", end=" ")
        result = compare_tokenization(hf_tokenizer, kitoken_encoder, chunk, f"shakespeare_chunk_{i+1}")
        all_results.append(result)
        print("✓ MATCH" if result['match'] else "✗ MISMATCH")
        if not result['match']:
            print(f"  ERROR: {result.get('error', 'Tokenization mismatch')}")

    # Test 2: Random strings
    print("\n" + "="*100)
    print("TEST 2: Random ASCII Strings")
    print("="*100)
    random_strings = generate_random_strings(20 if quick else 100)

    for i, text in enumerate(random_strings):
        print(f"Testing random string {i+1}/{len(random_strings)}...", end=" ")
        result = compare_tokenization(hf_tokenizer, kitoken_encoder, text, f"random_string_{i+1}")
        all_results.append(result)
        print("✓ MATCH" if result['match'] else "✗ MISMATCH")
        if not result['match']:
            print(f"  Preview: {text[:50]}...")
            print(f"  ERROR: {result.get('error', 'Tokenization mismatch')}")

    # Test 3: Non-English text
    print("\n" + "="*100)
    print("TEST 3: Non-English Text (Multilingual)")
    print("="*100)
    non_english = generate_non_english_text()

    for i, text in enumerate(non_english):
        print(f"Testing non-English text {i+1}/{len(non_english)}...", end=" ")
        result = compare_tokenization(hf_tokenizer, kitoken_encoder, text, f"non_english_{i+1}")
        all_results.append(result)
        print("✓ MATCH" if result['match'] else "✗ MISMATCH")
        if not result['match']:
            print(f"  Text: {text[:100]}...")
            print(f"  ERROR: {result.get('error', 'Tokenization mismatch')}")

    # Test 4: Edge cases
    print("\n" + "="*100)
    print("TEST 4: Edge Cases")
    print("="*100)
    edge_cases = generate_edge_cases()

    for i, text in enumerate(edge_cases):
        print(f"Testing edge case {i+1}/{len(edge_cases)}...", end=" ")
        result = compare_tokenization(hf_tokenizer, kitoken_encoder, text, f"edge_case_{i+1}")
        all_results.append(result)
        print("✓ MATCH" if result['match'] else "✗ MISMATCH")
        if not result['match']:
            print(f"  Text repr: {repr(text[:100])}")
            print(f"  ERROR: {result.get('error', 'Tokenization mismatch')}")

    # Test 5: Garbage bytes
    print("\n" + "="*100)
    print("TEST 5: Garbage Bytes")
    print("="*100)
    garbage = generate_garbage_bytes()
    garbage = garbage[:10 if quick else len(garbage)]

    for i, data in enumerate(garbage):
        print(f"Testing garbage bytes {i+1}/{len(garbage)}...", end=" ")
        result = compare_tokenization_bytes(hf_tokenizer, kitoken_encoder, data, f"garbage_bytes_{i+1}")
        all_results.append(result)
        print("✓ MATCH" if result['match'] else "✗ MISMATCH")
        if not result['match']:
            print(f"  ERROR: {result.get('error', 'Tokenization mismatch')}")

    return {
        'model_name': model_name,
        'tokenizer_path': tokenizer_path,
        'results': all_results,
    }


def print_summary(test_data: Dict):
    """Print test summary."""
    results = test_data['results']
    total = len(results)
    matches = sum(1 for r in results if r['match'])
    mismatches = total - matches

    print("\n" + "="*100)
    print("TEST SUMMARY")
    print("="*100)
    print(f"Model: {test_data['model_name']}")
    print(f"Tokenizer path: {test_data['tokenizer_path']}")
    print(f"\nTotal tests: {total}")
    print(f"Matches: {matches} ({matches/total*100:.1f}%)")
    print(f"Mismatches: {mismatches} ({mismatches/total*100:.1f}%)")

    if mismatches > 0:
        print("\n" + "="*100)
        print("FAILURES")
        print("="*100)
        for i, result in enumerate(results):
            if not result['match']:
                print(f"\n{i+1}. {result['test_name']}")
                print(f"   Text preview: {result['text_preview']}")
                print(f"   Text length: {result['text_length']}")
                if 'error' in result:
                    print(f"   Error: {result['error']}")
                else:
                    print(f"   HF tokens: {len(result.get('hf_tokens', []))}")
                    print(f"   KT tokens: {len(result.get('kt_tokens', []))}")
                    if 'first_diff_index' in result:
                        print(f"   First diff at index {result['first_diff_index']}: HF={result['first_diff_hf']} vs KT={result['first_diff_kt']}")

    print("\n" + "="*100)
    if mismatches == 0:
        print("✓ ALL TESTS PASSED - kitoken produces identical results to HuggingFace tokenizers!")
    else:
        print("✗ SOME TESTS FAILED - kitoken results differ from HuggingFace tokenizers")
    print("="*100)


@click.command()
@click.option('--model', '-m', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
              help='HuggingFace model name (must be a Llama-based model)', show_default=True)
@click.option('--quick', '-q', is_flag=True,
              help='Run in quick mode with fewer tests')
def main(model, quick):
    """
    Comprehensive test to evaluate kitoken against HuggingFace tokenizers.

    This test validates that kitoken produces identical tokenization results
    across diverse inputs including Shakespeare, random strings, non-English
    text, and garbage bytes.
    """
    try:
        test_data = run_tests(model, quick)
        print_summary(test_data)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
