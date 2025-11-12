"""
Core implementation of AutoTikTokenizer.
"""

import tiktoken
from transformers import AutoTokenizer
from typing import Union, List, Optional, Set
import warnings


# Mapping from HuggingFace model names to tiktoken encodings
MODEL_TO_ENCODING = {
    # GPT-2 models
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2",
    "gpt2-large": "gpt2",
    "gpt2-xl": "gpt2",
    "distilgpt2": "gpt2",

    # GPT-3 models
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",

    # GPT-3.5 and GPT-4 models
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4o": "o200k_base",
}


class TikTokenWrapper:
    """
    Wrapper that provides a tiktoken-compatible API for HuggingFace tokenizers.

    This is used when a model doesn't have a native tiktoken encoding.
    """

    def __init__(self, hf_tokenizer, model_name: str):
        self.hf_tokenizer = hf_tokenizer
        self.model_name = model_name
        self.name = f"hf/{model_name}"

    def encode(
        self,
        text: str,
        *,
        allowed_special: Union[Set[str], str] = set(),
        disallowed_special: Union[Set[str], str] = "all"
    ) -> List[int]:
        """
        Encode text to tokens using HuggingFace tokenizer.

        Args:
            text: Text to encode
            allowed_special: Special tokens to allow (tiktoken compatibility)
            disallowed_special: Special tokens to disallow (tiktoken compatibility)

        Returns:
            List of token IDs
        """
        return self.hf_tokenizer.encode(text, add_special_tokens=True)

    def encode_ordinary(self, text: str) -> List[int]:
        """
        Encode text without special tokens.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        return self.hf_tokenizer.encode(text, add_special_tokens=False)

    def encode_batch(
        self,
        texts: List[str],
        *,
        num_threads: Optional[int] = None,
        allowed_special: Union[Set[str], str] = set(),
        disallowed_special: Union[Set[str], str] = "all"
    ) -> List[List[int]]:
        """
        Encode multiple texts in batch.

        Args:
            texts: List of texts to encode
            num_threads: Number of threads (unused, for API compatibility)
            allowed_special: Special tokens to allow
            disallowed_special: Special tokens to disallow

        Returns:
            List of token ID lists
        """
        return [self.encode(text, allowed_special=allowed_special,
                          disallowed_special=disallowed_special) for text in texts]

    def encode_ordinary_batch(
        self,
        texts: List[str],
        *,
        num_threads: Optional[int] = None
    ) -> List[List[int]]:
        """
        Encode multiple texts without special tokens.

        Args:
            texts: List of texts to encode
            num_threads: Number of threads (unused, for API compatibility)

        Returns:
            List of token ID lists
        """
        return [self.encode_ordinary(text) for text in texts]

    def decode(self, tokens: List[int], errors: str = "replace") -> str:
        """
        Decode tokens to text.

        Args:
            tokens: List of token IDs
            errors: How to handle decode errors (replace/ignore/strict)

        Returns:
            Decoded text
        """
        return self.hf_tokenizer.decode(tokens, skip_special_tokens=False)

    def decode_bytes(self, tokens: List[int]) -> bytes:
        """
        Decode tokens to bytes.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded bytes
        """
        text = self.decode(tokens)
        return text.encode('utf-8', errors='replace')

    def decode_batch(
        self,
        batch: List[List[int]],
        *,
        errors: str = "replace",
        num_threads: Optional[int] = None
    ) -> List[str]:
        """
        Decode multiple token sequences.

        Args:
            batch: List of token ID lists
            errors: How to handle decode errors
            num_threads: Number of threads (unused, for API compatibility)

        Returns:
            List of decoded texts
        """
        return [self.decode(tokens, errors=errors) for tokens in batch]

    def decode_single_token_bytes(self, token: int) -> bytes:
        """
        Decode a single token to bytes.

        Args:
            token: Token ID

        Returns:
            Token bytes
        """
        text = self.hf_tokenizer.decode([token])
        return text.encode('utf-8', errors='replace')

    def decode_tokens_bytes(self, tokens: List[int]) -> List[bytes]:
        """
        Decode each token individually to bytes.

        Args:
            tokens: List of token IDs

        Returns:
            List of token bytes
        """
        return [self.decode_single_token_bytes(token) for token in tokens]

    @property
    def eot_token(self) -> Optional[int]:
        """Get end-of-text token ID."""
        return self.hf_tokenizer.eos_token_id

    @property
    def n_vocab(self) -> int:
        """Get vocabulary size."""
        return self.hf_tokenizer.vocab_size

    @property
    def max_token_value(self) -> int:
        """Get maximum token value."""
        return self.hf_tokenizer.vocab_size - 1


class AutoTikTokenizer:
    """
    AutoTikTokenizer - Load any HuggingFace tokenizer as a tiktoken-compatible encoder.

    This class provides a unified interface for loading tokenizers, using native tiktoken
    encodings when available and falling back to HuggingFace tokenizer wrappers otherwise.

    Example:
        >>> from autotiktokenizer import AutoTikTokenizer
        >>> encoder = AutoTikTokenizer.from_pretrained('gpt2')
        >>> tokens = encoder.encode("Hello world!")
        >>> text = encoder.decode(tokens)
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        force_tiktoken: bool = False,
        force_hf: bool = False,
        **kwargs
    ):
        """
        Load a tokenizer from a pretrained model.

        Args:
            model_name: Name of the HuggingFace model or tiktoken encoding
            force_tiktoken: Force use of native tiktoken encoding (raises error if unavailable)
            force_hf: Force use of HuggingFace tokenizer wrapper
            **kwargs: Additional arguments passed to HuggingFace AutoTokenizer

        Returns:
            tiktoken.Encoding or TikTokenWrapper instance

        Raises:
            ValueError: If force_tiktoken is True but no native encoding exists
        """
        # Check if we should force HuggingFace wrapper
        if force_hf:
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            return TikTokenWrapper(hf_tokenizer, model_name)

        # Check if model has a native tiktoken encoding
        encoding_name = MODEL_TO_ENCODING.get(model_name)

        if encoding_name:
            # Use native tiktoken encoding
            try:
                return tiktoken.get_encoding(encoding_name)
            except Exception as e:
                if force_tiktoken:
                    raise ValueError(
                        f"Failed to load native tiktoken encoding '{encoding_name}' "
                        f"for model '{model_name}': {e}"
                    )
                warnings.warn(
                    f"Failed to load native tiktoken encoding '{encoding_name}', "
                    f"falling back to HuggingFace wrapper: {e}"
                )
        elif force_tiktoken:
            raise ValueError(
                f"No native tiktoken encoding available for model '{model_name}'. "
                f"Available models: {list(MODEL_TO_ENCODING.keys())}"
            )

        # Fall back to HuggingFace tokenizer wrapper
        try:
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            return TikTokenWrapper(hf_tokenizer, model_name)
        except Exception as e:
            raise ValueError(
                f"Failed to load tokenizer for model '{model_name}': {e}"
            )

    @classmethod
    def list_models(cls) -> List[str]:
        """
        List all models with native tiktoken support.

        Returns:
            List of model names
        """
        return list(MODEL_TO_ENCODING.keys())

    @classmethod
    def get_encoding_name(cls, model_name: str) -> Optional[str]:
        """
        Get the tiktoken encoding name for a model.

        Args:
            model_name: HuggingFace model name

        Returns:
            tiktoken encoding name or None if not available
        """
        return MODEL_TO_ENCODING.get(model_name)
