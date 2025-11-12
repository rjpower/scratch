"""
AutoTikTokenizer - A Bridge Between TikToken and HuggingFace Tokenizers

This library enables loading any HuggingFace tokenizer as a TikToken-compatible encoder,
combining TikToken's performance with HuggingFace's model compatibility.
"""

from .core import AutoTikTokenizer

__version__ = "0.1.0"
__all__ = ["AutoTikTokenizer"]
