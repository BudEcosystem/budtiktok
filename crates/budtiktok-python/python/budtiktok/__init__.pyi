"""
Type stubs for BudTikTok Python bindings.

BudTikTok is an ultra-fast HuggingFace-compatible tokenizer with SIMD and multi-core support.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import numpy.typing as npt

class Encoding:
    """
    Encoding result from tokenization.

    Provides access to token IDs, attention masks, and other tokenization metadata.
    """

    @property
    def ids(self) -> List[int]:
        """Get token IDs as a list."""
        ...

    def get_ids_numpy(self) -> npt.NDArray[np.uint32]:
        """Get token IDs as a numpy array."""
        ...

    @property
    def type_ids(self) -> List[int]:
        """Get token type IDs."""
        ...

    @property
    def tokens(self) -> List[str]:
        """Get token strings."""
        ...

    @property
    def attention_mask(self) -> List[int]:
        """Get attention mask."""
        ...

    @property
    def special_tokens_mask(self) -> List[int]:
        """Get special tokens mask."""
        ...

    @property
    def offsets(self) -> List[Tuple[int, int]]:
        """Get byte offsets for each token."""
        ...

    @property
    def word_ids(self) -> List[Optional[int]]:
        """Get word IDs (which word each token belongs to)."""
        ...

    @property
    def sequence_ids(self) -> List[Optional[int]]:
        """Get sequence IDs."""
        ...

    def __len__(self) -> int:
        """Get the number of tokens."""
        ...

    def __getitem__(self, idx: int) -> int:
        """Get token ID at index."""
        ...


class Tokenizer:
    """
    HuggingFace-compatible tokenizer with 4-20x faster performance.

    Drop-in replacement for HuggingFace tokenizers with built-in
    SIMD acceleration and Rayon-based multi-core parallelism.
    """

    @staticmethod
    def from_file(path: str) -> "Tokenizer":
        """
        Load tokenizer from a tokenizer.json file.

        Args:
            path: Path to tokenizer.json file

        Returns:
            Tokenizer instance
        """
        ...

    @staticmethod
    def from_str(json: str) -> "Tokenizer":
        """
        Load tokenizer from a JSON string.

        Args:
            json: JSON string containing tokenizer configuration

        Returns:
            Tokenizer instance
        """
        ...

    @staticmethod
    def from_pretrained(model_name_or_path: str) -> "Tokenizer":
        """
        Load tokenizer from a pretrained model directory.

        Args:
            model_name_or_path: Path to model directory

        Returns:
            Tokenizer instance
        """
        ...

    @property
    def model_max_length(self) -> int:
        """Maximum sequence length."""
        ...

    @model_max_length.setter
    def model_max_length(self, value: int) -> None:
        ...

    @property
    def pad_token_id(self) -> int:
        """Pad token ID."""
        ...

    @pad_token_id.setter
    def pad_token_id(self, value: int) -> None:
        ...

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        ...

    def vocab_size_with_added(self, with_added_tokens: bool = True) -> int:
        """
        Get vocabulary size with option to include added tokens.

        Args:
            with_added_tokens: Whether to include added tokens in count

        Returns:
            Vocabulary size
        """
        ...

    def token_to_id(self, token: str) -> Optional[int]:
        """
        Convert a token to its ID.

        Args:
            token: Token string

        Returns:
            Token ID or None if not found
        """
        ...

    def id_to_token(self, id: int) -> Optional[str]:
        """
        Convert an ID to its token.

        Args:
            id: Token ID

        Returns:
            Token string or None if not found
        """
        ...

    def encode(self, text: str, add_special_tokens: bool = True) -> Encoding:
        """
        Encode a single text.

        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add special tokens (CLS, SEP, etc.)

        Returns:
            Encoding object
        """
        ...

    def encode_batch(
        self,
        texts: Sequence[str],
        add_special_tokens: bool = True
    ) -> List[Encoding]:
        """
        Encode a batch of texts in parallel.

        Uses Rayon's work-stealing thread pool for maximum parallelism.
        Releases the Python GIL during encoding.

        Args:
            texts: List of texts to tokenize
            add_special_tokens: Whether to add special tokens

        Returns:
            List of Encoding objects
        """
        ...

    def get_token_lengths(
        self,
        texts: Sequence[str],
        add_special_tokens: bool = True
    ) -> List[int]:
        """
        Get token lengths for a batch of texts.

        Optimized for token-budget batching systems.

        Args:
            texts: List of texts to get lengths for
            add_special_tokens: Whether to count special tokens

        Returns:
            List of token counts
        """
        ...

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        ...

    def __call__(
        self,
        text: Union[str, Sequence[str]],
        max_length: Optional[int] = None,
        padding: Optional[str] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = False,
    ) -> Dict[str, Union[npt.NDArray[np.uint32], List[List[int]]]]:
        """
        HuggingFace-compatible tokenization.

        Args:
            text: Single text or list of texts
            max_length: Maximum sequence length
            padding: Padding strategy ("longest", "max_length", or None)
            truncation: Whether to truncate sequences
            return_tensors: "np" for numpy arrays, "pt" for PyTorch, None for lists
            return_attention_mask: Whether to return attention mask
            return_token_type_ids: Whether to return token type IDs

        Returns:
            Dict with "input_ids", "attention_mask", optionally "token_type_ids"
        """
        ...

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Sequence[str],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: bool = False,
        return_attention_mask: bool = True,
    ) -> Dict[str, Union[npt.NDArray[np.uint32], List[List[int]], List[Encoding]]]:
        """
        Batch encode with HuggingFace-compatible interface.

        Args:
            batch_text_or_text_pairs: List of texts to encode
            add_special_tokens: Whether to add special tokens
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: "np" for numpy arrays, None for lists
            return_token_type_ids: Whether to return token type IDs
            return_attention_mask: Whether to return attention mask

        Returns:
            Dict with "input_ids", "attention_mask", and "encodings"
        """
        ...


# Alias for HF compatibility
PyTokenizer = Tokenizer


def get_config() -> Dict[str, Union[str, int, bool]]:
    """
    Get auto-configuration info.

    Returns a dict with detected hardware capabilities and optimal settings:
    - best_isa: Best available instruction set (e.g., "AVX2", "AVX512", "NEON")
    - physical_cores: Number of physical CPU cores
    - logical_cores: Number of logical CPU cores
    - use_simd_pretokenizer: Whether SIMD pretokenization is enabled
    - use_simd_normalizer: Whether SIMD normalization is enabled
    - cache_size: Optimal cache size
    - recommended_batch_size: Recommended batch size for parallelism
    """
    ...
