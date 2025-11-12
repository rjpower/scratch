"""
Converter to create native tiktoken Encodings from HuggingFace tokenizers.

This module enables using tiktoken's fast encoder/decoder with any HuggingFace tokenizer
by extracting the vocabulary, merges, and special tokens, then creating a tiktoken.Encoding.
"""

import tiktoken
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import Dict, Optional, Set
import regex as re
import tempfile
import os


# GPT-2 byte encoder/decoder (used by many HuggingFace tokenizers)
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    This is used by GPT-2 and many other HuggingFace tokenizers.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def build_mergeable_ranks(vocab: Dict[str, int], merges_str: Optional[str] = None) -> Dict[bytes, int]:
    """
    Build mergeable_ranks dictionary from HuggingFace vocabulary and merges.

    The key insight: merges.txt contains BPE merge operations in order.
    Each line is a merge like "h e" meaning "merge 'h' + 'e' -> 'he'".
    The LINE NUMBER (rank) determines merge priority in BPE.

    Args:
        vocab: Dictionary mapping token strings to token IDs
        merges_str: String containing merge operations (from merges.txt)

    Returns:
        Dictionary mapping token bytes to merge ranks for tiktoken
    """
    # Create reverse byte encoder (from GPT-2 encoding to actual bytes)
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    def token_str_to_bytes(token_str: str) -> bytes:
        """Convert token string to bytes using GPT-2 byte decoder."""
        try:
            return bytes([byte_decoder[c] for c in token_str])
        except KeyError:
            # Fall back to UTF-8 for special tokens
            return token_str.encode('utf-8', errors='ignore')

    mergeable_ranks = {}

    # Parse merges.txt to understand merge operations
    merge_operations = []
    if merges_str:
        for line in merges_str.strip().split('\n'):
            line = line.strip()
            # Skip empty lines and header comments
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    # Each line is like "h e" meaning merge h+e -> he
                    merge_operations.append((parts[0], parts[1]))

    # Build set of tokens created by merges
    merged_token_strs = set()
    for part1, part2 in merge_operations:
        merged = part1 + part2
        merged_token_strs.add(merged)

    # Step 1: Add base tokens (those not created by merges)
    # These get low ranks (high priority)
    base_rank = 0
    base_tokens = []

    for token_str in vocab.keys():
        # Base tokens are either:
        # 1. Single character tokens
        # 2. Tokens not created by any merge operation
        if token_str not in merged_token_strs:
            base_tokens.append(token_str)

    # Sort base tokens by their token ID for deterministic ordering
    base_tokens.sort(key=lambda t: vocab[t])

    for token_str in base_tokens:
        token_bytes = token_str_to_bytes(token_str)
        if token_bytes:
            mergeable_ranks[token_bytes] = base_rank
            base_rank += 1

    # Step 2: Add merged tokens with ranks based on merge order
    # Each merge operation gets a rank = base_rank + line_number
    for merge_idx, (part1, part2) in enumerate(merge_operations):
        merged_str = part1 + part2

        # Only add if this token exists in the vocabulary
        if merged_str in vocab:
            token_bytes = token_str_to_bytes(merged_str)
            if token_bytes and token_bytes not in mergeable_ranks:
                # Rank = base_rank + merge index
                # This ensures merges happen in the order specified in merges.txt
                mergeable_ranks[token_bytes] = base_rank + merge_idx

    # Step 3: Add any remaining tokens from vocab that we haven't seen
    # (edge case handling)
    max_rank = max(mergeable_ranks.values()) if mergeable_ranks else 0
    for token_str in vocab.keys():
        token_bytes = token_str_to_bytes(token_str)
        if token_bytes and token_bytes not in mergeable_ranks:
            max_rank += 1
            mergeable_ranks[token_bytes] = max_rank

    return mergeable_ranks


def extract_pattern_from_tokenizer(tokenizer) -> str:
    """
    Extract or infer the regex pattern used by the tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        Regex pattern string
    """
    # Try to get pattern from tokenizer backend
    if hasattr(tokenizer, 'backend_tokenizer'):
        backend = tokenizer.backend_tokenizer
        if hasattr(backend, 'pre_tokenizer'):
            pre_tok = backend.pre_tokenizer
            # Some tokenizers store the pattern
            if hasattr(pre_tok, 'pattern'):
                return pre_tok.pattern

    # Default patterns for common tokenizer types
    if 'llama' in tokenizer.name_or_path.lower() or 'alpaca' in tokenizer.name_or_path.lower():
        # LLaMA uses a pattern similar to GPT-2
        return r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    elif 'gpt2' in tokenizer.name_or_path.lower():
        return r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    else:
        # Generic pattern that works for most BPE tokenizers
        return r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def convert_hf_to_tiktoken(
    model_name_or_tokenizer,
    *,
    name: Optional[str] = None,
    pattern: Optional[str] = None,
    **tokenizer_kwargs
) -> tiktoken.Encoding:
    """
    Convert a HuggingFace tokenizer to a native tiktoken Encoding.

    This allows you to use tiktoken's fast encoder/decoder with any HuggingFace tokenizer.

    Args:
        model_name_or_tokenizer: HuggingFace model name or tokenizer instance
        name: Optional name for the encoding (defaults to model name)
        pattern: Optional regex pattern for tokenization (auto-detected if not provided)
        **tokenizer_kwargs: Additional arguments passed to AutoTokenizer.from_pretrained

    Returns:
        tiktoken.Encoding instance

    Example:
        >>> from autotiktokenizer import convert_hf_to_tiktoken
        >>> enc = convert_hf_to_tiktoken('NousResearch/Nous-Hermes-Llama2-13b')
        >>> tokens = enc.encode("Hello world!")
        >>> text = enc.decode(tokens)
    """
    # Load tokenizer if needed
    if isinstance(model_name_or_tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_tokenizer, **tokenizer_kwargs)
        model_name = model_name_or_tokenizer
    else:
        tokenizer = model_name_or_tokenizer
        model_name = getattr(tokenizer, 'name_or_path', 'custom')

    # Set encoding name
    if name is None:
        # Clean up model name for encoding name
        name = model_name.replace('/', '_').replace('-', '_')

    # Extract vocabulary
    vocab = tokenizer.get_vocab()

    # Try to get merges if available
    merges_str = None
    if hasattr(tokenizer, 'save_vocabulary'):
        # Save to temp directory to get merges
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                files = tokenizer.save_vocabulary(tmpdir)
                # files is a tuple of (vocab_file, merges_file) or just (vocab_file,)
                if isinstance(files, tuple) and len(files) > 1:
                    merges_file = files[1]
                    if merges_file and os.path.exists(merges_file):
                        with open(merges_file, 'r', encoding='utf-8') as f:
                            merges_str = f.read()
            except Exception as e:
                print(f"Warning: Could not extract merges: {e}")

    # Build mergeable ranks
    mergeable_ranks = build_mergeable_ranks(vocab, merges_str)

    # Extract or infer pattern
    if pattern is None:
        pattern = extract_pattern_from_tokenizer(tokenizer)

    # Extract special tokens
    special_tokens = {}
    if hasattr(tokenizer, 'special_tokens_map'):
        for token_name, token_str in tokenizer.special_tokens_map.items():
            if isinstance(token_str, str) and token_str in vocab:
                special_tokens[token_str] = vocab[token_str]

    # Also add special tokens from special_tokens_map_extended if available
    if hasattr(tokenizer, 'added_tokens_encoder'):
        for token_str, token_id in tokenizer.added_tokens_encoder.items():
            if token_str not in special_tokens:
                special_tokens[token_str] = token_id

    # Create tiktoken Encoding
    try:
        encoding = tiktoken.Encoding(
            name=name,
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens
        )
        return encoding
    except Exception as e:
        raise ValueError(
            f"Failed to create tiktoken Encoding from tokenizer '{model_name}': {e}\n"
            f"This may happen if the tokenizer uses a format incompatible with tiktoken."
        )


def verify_conversion(
    hf_tokenizer,
    tiktoken_encoder: tiktoken.Encoding,
    test_texts: Optional[list] = None
) -> bool:
    """
    Verify that tiktoken conversion produces same tokens as HuggingFace.

    Args:
        hf_tokenizer: Original HuggingFace tokenizer
        tiktoken_encoder: Converted tiktoken Encoding
        test_texts: Optional list of test strings

    Returns:
        True if conversion is verified correct
    """
    if test_texts is None:
        test_texts = [
            "Hello world!",
            "This is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing tokenization with numbers: 123 456 789.",
        ]

    all_match = True
    mismatches = []

    for text in test_texts:
        # Encode with HuggingFace
        hf_tokens = hf_tokenizer.encode(text, add_special_tokens=False)

        # Encode with tiktoken
        try:
            tiktoken_tokens = tiktoken_encoder.encode_ordinary(text)
        except:
            tiktoken_tokens = tiktoken_encoder.encode(text)

        # Compare
        if hf_tokens != tiktoken_tokens:
            all_match = False
            mismatches.append({
                'text': text,
                'hf_tokens': hf_tokens,
                'tiktoken_tokens': tiktoken_tokens,
                'hf_count': len(hf_tokens),
                'tiktoken_count': len(tiktoken_tokens)
            })

    if not all_match:
        print(f"Warning: Token mismatch detected in {len(mismatches)}/{len(test_texts)} test cases")
        for i, mismatch in enumerate(mismatches[:3], 1):  # Show first 3 mismatches
            print(f"\n  Mismatch {i}: '{mismatch['text'][:50]}...'")
            print(f"    HF:      {mismatch['hf_tokens'][:10]}... ({mismatch['hf_count']} tokens)")
            print(f"    TikToken: {mismatch['tiktoken_tokens'][:10]}... ({mismatch['tiktoken_count']} tokens)")

    return all_match


class HFTikTokenizer:
    """
    Helper class that combines convert_hf_to_tiktoken with caching and validation.

    This class caches the converted tiktoken Encoding for repeated use.
    """

    _cache: Dict[str, tiktoken.Encoding] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        force_reload: bool = False,
        verify: bool = False,
        **kwargs
    ) -> tiktoken.Encoding:
        """
        Load a HuggingFace tokenizer as a tiktoken Encoding (with caching).

        Args:
            model_name: HuggingFace model name
            force_reload: Force reload even if cached
            verify: Verify conversion correctness (slower, for testing)
            **kwargs: Additional arguments for tokenizer loading

        Returns:
            tiktoken.Encoding instance
        """
        if model_name not in cls._cache or force_reload:
            # Convert tokenizer
            if verify:
                # Load HF tokenizer for verification
                hf_tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
                encoder = convert_hf_to_tiktoken(hf_tokenizer, **kwargs)

                # Verify conversion
                print(f"Verifying conversion for {model_name}...")
                is_correct = verify_conversion(hf_tokenizer, encoder)
                if is_correct:
                    print(f"✓ Conversion verified correct!")
                else:
                    print(f"⚠ Conversion may have issues - tokens don't match exactly")
            else:
                encoder = convert_hf_to_tiktoken(model_name, **kwargs)

            cls._cache[model_name] = encoder
        return cls._cache[model_name]

    @classmethod
    def clear_cache(cls):
        """Clear the tokenizer cache."""
        cls._cache.clear()
