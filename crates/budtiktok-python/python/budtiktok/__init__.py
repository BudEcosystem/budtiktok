# BudTikTok Python bindings
# Ultra-fast HuggingFace-compatible tokenizer with SIMD and multi-core support

from .budtiktok import (
    Encoding,
    Tokenizer,
    get_config,
    set_num_threads,
    get_num_threads,
    init_optimal,
)

# Alias for HuggingFace compatibility
PyTokenizer = Tokenizer

__all__ = [
    "Encoding",
    "Tokenizer",
    "PyTokenizer",
    "get_config",
    "set_num_threads",
    "get_num_threads",
    "init_optimal",
]

__version__ = "0.1.0"
