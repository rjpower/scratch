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


def build_mergeable_ranks(vocab: Dict[str, int], merges: Optional[str] = None) -> Dict[bytes, int]:
    """
    Build mergeable_ranks dictionary from HuggingFace vocabulary and merges.

    Args:
        vocab: Dictionary mapping token strings to token IDs
        merges: Optional string containing merge operations (from merges.txt)

    Returns:
        Dictionary mapping token bytes to ranks for tiktoken
    """
    # Create reverse byte encoder (from GPT-2 encoding to actual bytes)
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    mergeable_ranks = {}

    # First, add all single-byte tokens and base tokens from vocab
    for token_str, token_id in vocab.items():
        # Try to decode the token from GPT-2 encoding to bytes
        try:
            # Convert token string to bytes using the GPT-2 byte decoder
            token_bytes = bytes([byte_decoder[c] for c in token_str])
            mergeable_ranks[token_bytes] = token_id
        except (KeyError, ValueError):
            # If decoding fails, encode as UTF-8 bytes directly
            # This handles special tokens and tokens not in the byte decoder
            try:
                token_bytes = token_str.encode('utf-8')
                mergeable_ranks[token_bytes] = token_id
            except:
                # Skip tokens that can't be encoded
                pass

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


class HFTikTokenizer:
    """
    Helper class that combines convert_hf_to_tiktoken with caching.

    This class caches the converted tiktoken Encoding for repeated use.
    """

    _cache: Dict[str, tiktoken.Encoding] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        force_reload: bool = False,
        **kwargs
    ) -> tiktoken.Encoding:
        """
        Load a HuggingFace tokenizer as a tiktoken Encoding (with caching).

        Args:
            model_name: HuggingFace model name
            force_reload: Force reload even if cached
            **kwargs: Additional arguments for tokenizer loading

        Returns:
            tiktoken.Encoding instance
        """
        if model_name not in cls._cache or force_reload:
            cls._cache[model_name] = convert_hf_to_tiktoken(model_name, **kwargs)
        return cls._cache[model_name]

    @classmethod
    def clear_cache(cls):
        """Clear the tokenizer cache."""
        cls._cache.clear()
