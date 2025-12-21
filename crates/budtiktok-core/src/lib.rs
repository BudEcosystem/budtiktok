//! BudTikTok Core - High-performance tokenization algorithms
//!
//! This crate provides the core tokenization algorithms used by BudTikTok:
//! - WordPiece (BERT-style)
//! - Unigram (SentencePiece-style)
//! - BPE (GPT-style, with O(n) linear algorithm)
//!
//! # Example
//!
//! ```rust,ignore
//! use budtiktok_core::{Tokenizer, WordPieceTokenizer};
//!
//! let tokenizer = WordPieceTokenizer::from_file("tokenizer.json")?;
//! let encoding = tokenizer.encode("Hello, world!", true)?;
//! println!("Token IDs: {:?}", encoding.ids);
//! ```

#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

// Core modules
pub mod error;
pub mod version;
pub mod runtime;
pub mod autoconfig;
pub mod allocator;
pub mod tables;
pub mod unicode;
pub mod cache;
pub mod double_array_trie;
pub mod trie;
pub mod vocab;
pub mod config;
pub mod encoding;
pub mod tokenizer;
pub mod normalizer;
pub mod pretokenizer;
pub mod pretokenizer_simd;
pub mod postprocessor;
pub mod decoder;
pub mod special_tokens;
pub mod memory;
pub mod swar;
pub mod arena;
pub mod simd_backends;
pub mod loader;

// HF-compatible pipeline (new architecture)
pub mod pipeline;

// SIMD utilities (required by wordpiece_hyper)
pub mod avx512;

// Tokenizer implementations
pub mod wordpiece;       // Scalar WordPiece
pub mod wordpiece_hyper; // SIMD WordPiece
pub mod wordlevel;       // High-performance WordLevel tokenizer
pub mod unigram;         // Unigram (SentencePiece)
pub mod unigram_fast;    // High-performance Unigram (10x faster)
pub mod bpe_fast;        // BPE utilities (Gpt2ByteEncoderFast)
pub mod bpe_linear;      // Production BPE (OptimizedBpeEncoder)

// Experimental implementations (archived - not faster or accurate enough)
// See: src/experiments/bpe_simd.rs

// Re-exports: Core types
pub use error::{Error, Result};
pub use version::{VERSION, GIT_HASH, BUILD_TIMESTAMP, BuildInfo, SimdFeatures, SimdTier, full_version};
pub use runtime::{
    RuntimeConfig, IsaSelection, ParallelismConfig, SystemInfo,
    system_info, apply_config, set_global_threads, get_global_threads,
};
pub use autoconfig::{AutoConfig, AutoConfigStats, get_auto_config, init_auto_config};
pub use allocator::{
    current_allocator, has_custom_allocator, AllocatorKind, AllocatorStats,
    AllocationHint, TokenArena,
};
pub use encoding::{
    Encoding,
    // Padding
    PaddingDirection, PaddingStrategy, PaddingParams, pad_encodings,
    // Truncation
    TruncationDirection, TruncationStrategy, TruncationParams, truncate_encoding,
};
pub use tokenizer::Tokenizer;
pub use config::TokenizerConfig;
pub use loader::{load_tokenizer, load_tokenizer_from_str};

// Re-exports: Data structures
pub use trie::{Trie, TrieBuilder, trie_from_vocab, CacheObliviousTrie, CacheObliviousTrieStats};
pub use double_array_trie::{DoubleArrayTrie, DoubleArrayTrieBuilder, DoubleArrayTrieStats, TrieLookup};
pub use arena::{Arena, ArenaVec, ArenaPool, TokenBuffer, TokenBufferPool};

// Re-exports: Text processing
pub use normalizer::{
    Normalizer, BertNormalizer, BertNormalizerConfig,
    NfcNormalizer, NfkcNormalizer, NfdNormalizer, NfkdNormalizer,
    LowercaseNormalizer, StripAccentsNormalizer,
    PrependNormalizer, ReplaceNormalizer, SequenceNormalizer,
    StreamingNormalizer, StreamingNormalizerState, SimdNormalizer,
    // New normalizers
    StripNormalizer, NmtNormalizer, PrecompiledNormalizer, NormalizerWrapper,
};
pub use pretokenizer::{
    PreTokenizer, PreToken, BertPreTokenizer, WhitespacePreTokenizer, MetaspacePreTokenizer,
    ByteLevelPreTokenizer, SplitPreTokenizer, SplitBehavior, PunctuationPreTokenizer,
    DigitsPreTokenizer, CharDelimiterSplit, UnicodeScriptsPreTokenizer, UnicodeScript,
    RegexSplitPreTokenizer, SequencePreTokenizer, PrependScheme as PreTokenizerPrependScheme,
};
pub use pretokenizer_simd::{SimdBertPreTokenizer, SimdWhitespacePreTokenizer};
pub use postprocessor::{
    PostProcessor, BertPostProcessor, RobertaPostProcessor, SpecialToken,
    SequencePostProcessor, ByteLevelPostProcessor, TemplatePostProcessor, TemplatePart,
    PostProcessorWrapper, TemplatePartSpec, SpecialTokenSpec, SequenceSpec,
};
pub use decoder::{
    Decoder, DecoderWrapper, ByteLevelDecoder, MetaspaceDecoder, WordPieceDecoder,
    BPEDecoder, ByteFallbackDecoder, FuseDecoder, StripDecoder, ReplaceDecoder,
    CTCDecoder, SequenceDecoder, PrependScheme,
    byte_to_gpt2_char, gpt2_char_to_byte,
    decode_batch_parallel, decode_chain_batch_parallel,
};
pub use special_tokens::{
    SpecialTokenMatcher, SpecialTokenMatcherBuilder, AddedToken, SpecialTokenMatch,
    ExtractedSegment, bert_special_tokens, gpt2_special_tokens, llama_special_tokens, t5_special_tokens,
};

// Re-exports: Pipeline (HF-compatible architecture)
pub use pipeline::{
    TokenizerPipeline, EncodeInput,
    TokenizerModel, Token,
    AddedVocabulary, ExtractedSegment as PipelineExtractedSegment,
};

// Re-exports: Memory utilities
pub use memory::{
    ScopedArena, with_arena, with_arena_reset,
    StringInterner, InternerStatsSnapshot, intern, global_interner_stats,
    VecPool, VecPoolStats, EncodingPool, EncodingVecs,
    InternedVocabulary, with_string_buffer, with_u32_buffer,
    // NUMA-aware memory management
    NumaTopology, NumaParallelConfig, set_thread_affinity, current_numa_node,
};

// Re-exports: SWAR utilities
pub use swar::{
    // Core SWAR operations
    has_zero_byte, has_byte, broadcast_byte, has_non_ascii,
    count_bytes_less_than, find_byte_position, find_whitespace_position,
    match_any_of_4,
    // Branchless classification
    is_ascii_whitespace_branchless, is_ascii_alphanumeric_branchless,
    is_ascii_punctuation_branchless, to_lowercase_branchless, to_uppercase_branchless,
    classify_byte_branchless,
    // Loop unrolling utilities
    find_first_whitespace_unrolled, count_whitespace_unrolled,
    is_all_ascii_unrolled, to_lowercase_ascii_unrolled, split_on_whitespace_ranges,
};

// Re-exports: SIMD utilities
pub use avx512::{
    is_avx512_available,
    classify_whitespace as avx512_classify_whitespace,
    classify_alphanumeric as avx512_classify_alphanumeric,
    classify_punctuation as avx512_classify_punctuation,
    classify_non_ascii as avx512_classify_non_ascii,
    pretokenize as avx512_pretokenize,
    find_word_boundaries as avx512_find_word_boundaries,
    count_whitespace as avx512_count_whitespace,
    is_all_ascii as avx512_is_all_ascii,
    find_first_non_ascii as avx512_find_first_non_ascii,
    to_lowercase_ascii as avx512_to_lowercase_ascii,
    to_uppercase_ascii as avx512_to_uppercase_ascii,
};
pub use simd_backends::{SimdCapabilities, get_capabilities};

// Re-exports: WordPiece tokenizers
pub use wordpiece::{WordPieceTokenizer, WordPieceConfig};
pub use wordpiece_hyper::{HyperWordPieceTokenizer, HyperConfig};

// Re-exports: BPE tokenizers (Production)
pub use bpe_fast::{
    Gpt2ByteEncoderFast, QuaternaryHeap, Merge, Pair, Symbol, Word, MergeMap,
    CharToId, build_merge_map, gpt2_pre_tokenize,
};
pub use bpe_linear::{
    // Production BPE encoder
    OptimizedBpeEncoder, Gpt2PreTokenizer, gpt2_pre_tokenize_fast,
    // BPE tokenizer with Tokenizer trait (for loader compatibility)
    BpeTokenizer, BpeConfig, MergeRule, parse_merges,
    // Legacy encoders (may be removed in future)
    TrieEncoder, FastLinearEncoder, DirectBpeEncoder, GreedyLinearEncoder,
};

// Re-exports: Unigram tokenizers
pub use unigram::{UnigramTokenizer, UnigramConfig, UnigramPiece, SPIECE_UNDERLINE};
pub use unigram_fast::{UnigramFast, UnigramFastConfig, UnigramTokenCache, normalize_for_hf, preprocess_text};

// Re-exports: WordLevel tokenizer
pub use wordlevel::{
    WordLevelTokenizer, WordLevelConfig, WordLevelCache,
    WordLevelPreToken, pretokenize_scalar, pretokenize_scalar_lowercase,
    pretokenize_simd, is_whitespace, is_punctuation, is_cjk_character,
    // Note: is_avx2_available and is_avx512_available exported from avx512 module
};

// Re-exports: Cache types
pub use cache::{
    CacheStats, ClockCache, ShardedCache, MultiLevelCache, MultiLevelCacheStats,
    TokenCache, ShardedTokenCache, MultiLevelTokenCache,
    CacheWarmer, WarmupConfig, WarmupResult,
};
