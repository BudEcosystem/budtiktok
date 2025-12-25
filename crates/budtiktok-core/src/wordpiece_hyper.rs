//! Hyper-Optimized WordPiece Tokenizer
//!
//! Maximum performance through:
//! - Direct hash lookup for common tokens (bypass trie entirely)
//! - Inline short-word optimization (< 8 bytes)
//! - Aggressive prefetching and cache optimization
//! - Cache-line aligned data structures (64-byte alignment)
//! - Zero-copy batch processing with Rayon parallelism
//! - SIMD-accelerated pre-tokenization (AVX2/AVX-512/NEON)
//! - simdjson-style 64-byte block processing with bitmask operations
//! - Vectorized hash bucket probing (8-way parallel comparison)
//! - AVX2/AVX-512 character classification
//!
//! Target: 5x+ faster than BlazeText
//!
//! Key optimizations based on research:
//! - simdjson: Process 64 bytes at once, use tzcnt for bitmask extraction
//! - StringZilla: SWAR for short strings, SIMD for longer sequences
//! - Google LinMaxMatch: O(n) greedy longest-match algorithm
//! - Swiss Tables: SIMD parallel probing in hash buckets

use serde::{Deserialize, Serialize};
use rayon::prelude::*;

use crate::vocab::Vocabulary;
use crate::unicode::{strip_accents, to_lowercase, normalize_bert_text, normalize, NormalizationForm};
use crate::avx512::{classify_whitespace, classify_punctuation, is_avx512_available};

// =============================================================================
// AVX2 Feature Detection (for systems without AVX-512)
// =============================================================================

/// Check if AVX2 is available at runtime
#[inline]
fn is_avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Software prefetch hint for read
#[inline(always)]
fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{std::arch::x86_64::_MM_HINT_T0}>(ptr as *const i8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = ptr;
    }
}

/// Perfect hash bucket for fast token lookup
/// Cache-line aligned (64 bytes) for optimal memory access
/// Layout: 8 hashes (64B) + 8 ids (32B) + 8 lengths (8B) + padding = 128 bytes
/// This allows 8-way parallel SIMD comparison
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct HashBucket {
    /// Hash values for collision detection (8 × 8 bytes = 64 bytes)
    hashes: [u64; 8],
    /// Token IDs (0 = empty) (8 × 4 bytes = 32 bytes)
    ids: [u32; 8],
    /// Token lengths for validation (8 × 1 byte = 8 bytes)
    lengths: [u8; 8],
}

/// Direct lookup table for single-byte tokens (256 entries)
/// Maps ASCII byte -> token_id + 1 (0 = not found)
/// This provides O(1) lookup with no hashing for punctuation and single chars
#[repr(C, align(64))]
struct SingleByteLookup {
    ids: [u32; 256],
}

impl SingleByteLookup {
    fn new() -> Self {
        Self { ids: [0; 256] }
    }

    #[inline(always)]
    fn get(&self, byte: u8) -> u32 {
        self.ids[byte as usize]
    }

    fn set(&mut self, byte: u8, id: u32) {
        self.ids[byte as usize] = id + 1; // +1 because 0 = not found
    }
}

impl HashBucket {
    fn new() -> Self {
        Self {
            hashes: [0; 8],
            ids: [0; 8],
            lengths: [0; 8],
        }
    }
}

/// Fast hash function (FNV-1a)
/// Note: Must use FNV-1a for consistency with incremental hash in speculative lookups
#[inline(always)]
fn fast_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Hyper-optimized tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperConfig {
    pub continuing_subword_prefix: String,
    pub max_input_chars_per_word: usize,
    pub unk_token: String,
    pub do_lower_case: bool,
    pub strip_accents: bool,
    pub tokenize_chinese_chars: bool,
    /// Hash table size (power of 2)
    pub hash_table_bits: u32,
}

impl Default for HyperConfig {
    fn default() -> Self {
        Self {
            continuing_subword_prefix: "##".to_string(),
            // Match HuggingFace's default (100, not 200)
            max_input_chars_per_word: 100,
            unk_token: "[UNK]".to_string(),
            do_lower_case: true,
            strip_accents: true,
            tokenize_chinese_chars: true,
            // 16K buckets (2MB) fits in L3 cache of most CPUs
            // BERT vocabulary is ~30K tokens, so 16K buckets with 8-way associativity
            // has ~2 tokens per bucket on average - excellent for cache efficiency
            hash_table_bits: 14,
        }
    }
}

/// Hyper-optimized WordPiece tokenizer
pub struct HyperWordPieceTokenizer {
    /// Direct lookup for single-byte tokens (punctuation, single letters)
    /// O(1) lookup with no hashing - fastest path
    single_byte_lookup: Box<SingleByteLookup>,
    /// Hash table for direct token lookup (bypasses trie for known tokens)
    hash_table: Vec<HashBucket>,
    /// Separate hash table for continuation tokens (##prefix stripped)
    /// Uses same SIMD-accelerated lookup as main table
    /// Smaller size (14-bit = 16K buckets) to reduce memory pressure
    continuation_hash_table: Vec<HashBucket>,
    /// Hash table mask for main table
    hash_mask: u64,
    /// Hash table mask for continuation table (smaller)
    continuation_mask: u64,
    /// Vocabulary for special tokens and fallback
    vocabulary: Vocabulary,
    /// Configuration
    config: HyperConfig,
    /// CLS token ID
    cls_id: Option<u32>,
    /// SEP token ID
    sep_id: Option<u32>,
    /// UNK token ID
    unk_id: u32,
    /// Continuation prefix bytes
    prefix_bytes: Vec<u8>,
    /// Cached AVX2 availability (avoid repeated detection in hot path)
    #[cfg(target_arch = "x86_64")]
    has_avx2: bool,
}

impl HyperWordPieceTokenizer {
    /// Create a new hyper-optimized tokenizer
    pub fn new(vocabulary: Vocabulary, config: HyperConfig) -> Self {
        let table_size = 1usize << config.hash_table_bits;
        let hash_mask = (table_size - 1) as u64;

        let mut single_byte_lookup = Box::new(SingleByteLookup::new());
        let hash_table_size = 1 << config.hash_table_bits;
        let mut hash_table = vec![HashBucket::new(); hash_table_size];

        // Continuation table is smaller (1/4 size)
        let continuation_bits = config.hash_table_bits.saturating_sub(2).max(10);
        let continuation_size = 1 << continuation_bits;
        let mut continuation_hash_table = vec![HashBucket::new(); continuation_size];

        let hash_mask = (hash_table_size - 1) as u64;
        let continuation_mask = (continuation_size - 1) as u64;

        let prefix_bytes = config.continuing_subword_prefix.as_bytes().to_vec();

        // Populate hash tables from vocabulary
        for (token, &id) in vocabulary.token_to_id_map() {
            if token.is_empty() {
                continue;
            }

            let bytes = token.as_bytes();

            // Single byte optimization
            if bytes.len() == 1 {
                single_byte_lookup.set(bytes[0], id);
            }

            // Check if it's a continuation token
            if bytes.starts_with(&prefix_bytes) && bytes.len() > prefix_bytes.len() {
                let sub_bytes = &bytes[prefix_bytes.len()..];
                let h = fast_hash(sub_bytes);
                let bucket_idx = (h & continuation_mask) as usize;

                // Try to insert into continuation table
                let mut inserted = false;
                for i in 0..16 {
                    let idx = (bucket_idx + i) & (continuation_mask as usize);
                    let bucket = &mut continuation_hash_table[idx];
                    for slot in 0..8 {
                        if bucket.ids[slot] == 0 {
                            bucket.hashes[slot] = h;
                            bucket.ids[slot] = id + 1;
                            bucket.lengths[slot] = sub_bytes.len() as u8;
                            inserted = true;
                            break;
                        }
                    }
                    if inserted {
                        break;
                    }
                }
            } else {
                // Main table
                let h = fast_hash(bytes);
                let bucket_idx = (h & hash_mask) as usize;

                let mut inserted = false;
                for i in 0..16 {
                    let idx = (bucket_idx + i) & (hash_mask as usize);
                    let bucket = &mut hash_table[idx];
                    for slot in 0..8 {
                        if bucket.ids[slot] == 0 {
                            bucket.hashes[slot] = h;
                            bucket.ids[slot] = id + 1;
                            bucket.lengths[slot] = bytes.len() as u8;
                            inserted = true;
                            break;
                        }
                    }
                    if inserted {
                        break;
                    }
                }
            }
        }

        let cls_id = vocabulary.cls_token().and_then(|t| vocabulary.token_to_id(t));
        let sep_id = vocabulary.sep_token().and_then(|t| vocabulary.token_to_id(t));
        let unk_id = vocabulary
            .unk_token()
            .and_then(|t| vocabulary.token_to_id(t))
            .unwrap_or(0);

        Self {
            single_byte_lookup,
            hash_table,
            continuation_hash_table,
            hash_mask,
            continuation_mask,
            vocabulary,
            config,
            cls_id,
            sep_id,
            unk_id,
            prefix_bytes,
            #[cfg(target_arch = "x86_64")]
            has_avx2: is_avx2_available(),
        }
    }

    /// Load a tokenizer from a JSON file
    pub fn from_pretrained(path: impl AsRef<std::path::Path>) -> crate::error::Result<Self> {
        use crate::config::TokenizerConfig;
        let config = TokenizerConfig::from_file(path)?;
        Self::from_config(config)
    }

    /// Create a tokenizer from a TokenizerConfig
    pub fn from_config(config: crate::config::TokenizerConfig) -> crate::error::Result<Self> {
        if !config.is_wordpiece() {
            return Err(crate::error::Error::InvalidConfig(
                "Not a WordPiece tokenizer config".to_string(),
            ));
        }

        let vocab = crate::vocab::Vocabulary::new(
            config.model.get_vocab(),
            crate::vocab::SpecialTokens {
                unk_token: config.model.unk_token.clone(),
                pad_token: config.get_special_token("pad").map(|t| t.content.clone()),
                cls_token: config.get_special_token("cls").map(|t| t.content.clone()),
                sep_token: config.get_special_token("sep").map(|t| t.content.clone()),
                mask_token: config.get_special_token("mask").map(|t| t.content.clone()),
                bos_token: config.get_special_token("bos").map(|t| t.content.clone()),
                eos_token: config.get_special_token("eos").map(|t| t.content.clone()),
            },
        );

        let hyper_config = HyperConfig {
            continuing_subword_prefix: config
                .model
                .continuing_subword_prefix
                .clone()
                .unwrap_or_else(|| "##".to_string()),
            max_input_chars_per_word: config.model.max_input_chars_per_word.unwrap_or(100),
            unk_token: config
                .model
                .unk_token
                .clone()
                .unwrap_or_else(|| "[UNK]".to_string()),
            do_lower_case: config
                .normalizer
                .as_ref()
                .and_then(|n| n.lowercase)
                .unwrap_or(true),
            strip_accents: config
                .normalizer
                .as_ref()
                .and_then(|n| n.strip_accents)
                .unwrap_or(true),
            tokenize_chinese_chars: config
                .normalizer
                .as_ref()
                .and_then(|n| n.handle_chinese_chars)
                .unwrap_or(true),
            hash_table_bits: 14, // Default
        };

        Ok(Self::new(vocab, hyper_config))
    }

    /// Direct hash lookup for token (returns token_id + 1, or 0 if not found)
    /// Optimized with AVX2 SIMD parallel comparison of 4 hash values at once
    /// Uses linear probing (up to 16 buckets) for overflow handling
    #[inline(always)]
    fn hash_lookup(&self, bytes: &[u8]) -> u32 {
        let h = fast_hash(bytes);
        let initial_bucket = (h & self.hash_mask) as usize;
        let len = bytes.len() as u8;
        let table_size = self.hash_table.len();

        // Linear probe up to 16 buckets
        // Key insight: stop probing when we find an empty slot, because if the token
        // existed, it would have been inserted there instead of probing further
        for probe in 0..16 {
            let bucket_idx = (initial_bucket + probe) % table_size;
            let bucket = &self.hash_table[bucket_idx];

            // Count occupied slots to determine if bucket is full
            let mut occupied = 0u8;

            // Use cached AVX2 flag (avoid repeated runtime detection)
            #[cfg(target_arch = "x86_64")]
            {
                if self.has_avx2 {
                    let result = unsafe { self.hash_lookup_avx2(bucket, h, len) };
                    if result != 0 {
                        return result;
                    }
                    // Count occupied slots
                    for i in 0..8 {
                        if bucket.ids[i] != 0 {
                            occupied += 1;
                        }
                    }
                    // If bucket is not completely full, token would be here if it existed
                    if occupied < 8 {
                        return 0;
                    }
                    continue;
                }
            }

            // Scalar fallback with unrolled comparisons
            let result = self.hash_lookup_scalar(bucket, h, len);
            if result != 0 {
                return result;
            }
            // Count occupied slots
            for i in 0..8 {
                if bucket.ids[i] != 0 {
                    occupied += 1;
                }
            }
            // If bucket is not completely full, token would be here if it existed
            if occupied < 8 {
                return 0;
            }
        }

        0
    }

    /// AVX2-accelerated 8-way hash lookup
    /// Compares 4 hashes at a time using 256-bit SIMD
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hash_lookup_avx2(&self, bucket: &HashBucket, h: u64, len: u8) -> u32 {
        use std::arch::x86_64::*;

        // Broadcast target hash to 4 lanes
        let target = _mm256_set1_epi64x(h as i64);

        // Load first 4 hashes (256 bits = 4 × 64-bit)
        let hashes_0_3 = _mm256_loadu_si256(bucket.hashes.as_ptr() as *const __m256i);
        // Load next 4 hashes
        let hashes_4_7 = _mm256_loadu_si256(bucket.hashes.as_ptr().add(4) as *const __m256i);

        // Compare: result lanes are all 1s if equal, all 0s if not
        let cmp_0_3 = _mm256_cmpeq_epi64(hashes_0_3, target);
        let cmp_4_7 = _mm256_cmpeq_epi64(hashes_4_7, target);

        // Get comparison masks (4 bits per comparison for 64-bit elements)
        // movemask_pd treats as doubles, gives us 1 bit per 64-bit element
        let mask_0_3 = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_0_3)) as u32;
        let mask_4_7 = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_4_7)) as u32;

        // Combine masks: bits 0-3 for hashes[0-3], bits 4-7 for hashes[4-7]
        let combined_mask = mask_0_3 | (mask_4_7 << 4);

        if combined_mask == 0 {
            return 0; // No hash match
        }

        // Found matching hash(es), verify length and return first match
        let mut mask = combined_mask;
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize;
            if bucket.lengths[idx] == len && bucket.ids[idx] != 0 {
                return bucket.ids[idx];
            }
            mask &= mask - 1; // Clear lowest set bit
        }

        0 // Hash matched but length didn't
    }

    /// Scalar 8-way hash lookup (fallback)
    #[inline(always)]
    fn hash_lookup_scalar(&self, bucket: &HashBucket, h: u64, len: u8) -> u32 {
        // Unrolled 8-way comparison
        if bucket.hashes[0] == h && bucket.lengths[0] == len && bucket.ids[0] != 0 {
            return bucket.ids[0];
        }
        if bucket.hashes[1] == h && bucket.lengths[1] == len && bucket.ids[1] != 0 {
            return bucket.ids[1];
        }
        if bucket.hashes[2] == h && bucket.lengths[2] == len && bucket.ids[2] != 0 {
            return bucket.ids[2];
        }
        if bucket.hashes[3] == h && bucket.lengths[3] == len && bucket.ids[3] != 0 {
            return bucket.ids[3];
        }
        if bucket.hashes[4] == h && bucket.lengths[4] == len && bucket.ids[4] != 0 {
            return bucket.ids[4];
        }
        if bucket.hashes[5] == h && bucket.lengths[5] == len && bucket.ids[5] != 0 {
            return bucket.ids[5];
        }
        if bucket.hashes[6] == h && bucket.lengths[6] == len && bucket.ids[6] != 0 {
            return bucket.ids[6];
        }
        if bucket.hashes[7] == h && bucket.lengths[7] == len && bucket.ids[7] != 0 {
            return bucket.ids[7];
        }
        0
    }

    /// Scalar lookup for only the second half of bucket (slots 4-7)
    /// Used when first 4 slots already checked inline
    #[inline(always)]
    fn hash_lookup_scalar_second_half(&self, bucket: &HashBucket, h: u64, len: u8) -> u32 {
        if bucket.hashes[4] == h && bucket.lengths[4] == len && bucket.ids[4] != 0 {
            return bucket.ids[4];
        }
        if bucket.hashes[5] == h && bucket.lengths[5] == len && bucket.ids[5] != 0 {
            return bucket.ids[5];
        }
        if bucket.hashes[6] == h && bucket.lengths[6] == len && bucket.ids[6] != 0 {
            return bucket.ids[6];
        }
        if bucket.hashes[7] == h && bucket.lengths[7] == len && bucket.ids[7] != 0 {
            return bucket.ids[7];
        }
        0
    }

    /// AVX2-accelerated lookup for only the second half of bucket (slots 4-7)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hash_lookup_avx2_second_half(&self, bucket: &HashBucket, h: u64, len: u8) -> u32 {
        use std::arch::x86_64::*;

        // Broadcast target hash to 4 lanes
        let target = _mm256_set1_epi64x(h as i64);

        // Load only the second 4 hashes (slots 4-7)
        let hashes_4_7 = _mm256_loadu_si256(bucket.hashes.as_ptr().add(4) as *const __m256i);

        // Compare
        let cmp_4_7 = _mm256_cmpeq_epi64(hashes_4_7, target);
        let mask_4_7 = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_4_7)) as u32;

        if mask_4_7 == 0 {
            return 0;
        }

        // Verify length for matching slots
        let mut mask = mask_4_7;
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize + 4; // Add 4 for second half offset
            if bucket.lengths[idx] == len && bucket.ids[idx] != 0 {
                return bucket.ids[idx];
            }
            mask &= mask - 1;
        }

        0
    }

    /// Lookup continuation token using dedicated SIMD-accelerated hash table
    /// Same performance as main hash_lookup but for ##-prefixed tokens
    #[inline(always)]
    fn continuation_lookup(&self, bytes: &[u8]) -> Option<u32> {
        let id = self.continuation_hash_lookup(bytes);
        if id != 0 {
            Some(id - 1)
        } else {
            None
        }
    }

    /// Direct hash lookup for continuation tokens (returns token_id + 1, or 0 if not found)
    /// Uses same SIMD-accelerated bucket probing as main hash lookup
    #[inline(always)]
    /// Continuation hash lookup with linear probing for overflow handling
    fn continuation_hash_lookup(&self, bytes: &[u8]) -> u32 {
        let h = fast_hash(bytes);
        let initial_bucket = (h & self.continuation_mask) as usize;
        let len = bytes.len() as u8;
        let table_size = self.continuation_hash_table.len();

        // Linear probe up to 16 buckets
        for probe in 0..16 {
            let bucket_idx = (initial_bucket + probe) % table_size;
            let bucket = &self.continuation_hash_table[bucket_idx];

            // Count occupied slots
            let mut occupied = 0u8;

            // Use cached AVX2 flag
            #[cfg(target_arch = "x86_64")]
            {
                if self.has_avx2 {
                    let result = unsafe { self.hash_lookup_avx2(bucket, h, len) };
                    if result != 0 {
                        return result;
                    }
                    for i in 0..8 {
                        if bucket.ids[i] != 0 {
                            occupied += 1;
                        }
                    }
                    if occupied < 8 {
                        return 0;
                    }
                    continue;
                }
            }

            // Scalar fallback
            let result = self.hash_lookup_scalar(bucket, h, len);
            if result != 0 {
                return result;
            }
            for i in 0..8 {
                if bucket.ids[i] != 0 {
                    occupied += 1;
                }
            }
            if occupied < 8 {
                return 0;
            }
        }

        0
    }

    /// Encode text to token IDs (hyper-fast path)
    #[inline]
    pub fn encode_fast(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Step 1: BERT text normalization (control chars, PUA, FFFD, etc.)
        // This matches HuggingFace's BertNormalizer exactly
        let bert_normalized = normalize_bert_text(text);

        // Step 2: Apply NFC normalization, lowercase, and accent stripping
        // Check if text needs Unicode normalization (contains non-ASCII)
        let needs_unicode_norm = bert_normalized.bytes().any(|b| b >= 0x80);

        let normalized: std::borrow::Cow<str> = if needs_unicode_norm {
            // Full Unicode NFC normalization (required for consistency with scalar tokenizer)
            let mut s = normalize(&bert_normalized, NormalizationForm::NFC);
            if self.config.do_lower_case {
                s = to_lowercase(&s);
            }
            if self.config.strip_accents {
                s = strip_accents(&s);
            }
            std::borrow::Cow::Owned(s)
        } else if self.config.do_lower_case && bert_normalized.bytes().any(|b| b >= b'A' && b <= b'Z') {
            // ASCII-only lowercase (fast path - NFC is identity for ASCII)
            std::borrow::Cow::Owned(bert_normalized.to_ascii_lowercase())
        } else {
            // ASCII-only, no transformation needed (NFC is identity for ASCII)
            std::borrow::Cow::Owned(bert_normalized)
        };

        let text_bytes = normalized.as_bytes();

        // Pre-tokenize and encode
        let mut ids = Vec::with_capacity(text_bytes.len() / 4 + 4);
        let word_bounds = self.pre_tokenize_inline(text_bytes);
        let num_words = word_bounds.len();

        // Process words with software prefetching for the next iteration
        // This hides memory latency by fetching the next bucket while processing the current one
        let mut i = 0;
        while i < num_words {
            let (start, end) = word_bounds[i];
            let word = &text_bytes[start..end];
            if word.is_empty() {
                i += 1;
                continue;
            }

            let word_len = word.len();

            // === PREFETCH NEXT BUCKET (hide memory latency) ===
            // Compute hash for next word and prefetch its bucket while we process current word
            if i + 1 < num_words {
                let (next_start, next_end) = word_bounds[i + 1];
                if next_end > next_start {
                    let next_word = &text_bytes[next_start..next_end];
                    if !next_word.is_empty() && next_word.len() > 1 {
                        let next_h = fast_hash(next_word);
                        let next_bucket_idx = (next_h & self.hash_mask) as usize;
                        // SAFETY: next_bucket_idx is bounded by hash_mask which is < hash_table.len()
                        unsafe { prefetch_read(self.hash_table.as_ptr().add(next_bucket_idx)); }
                    }
                }
            }

            // === ULTRA-FAST PATH: Single-byte direct lookup (O(1), no hashing) ===
            // Covers punctuation and single-letter tokens
            if word_len == 1 {
                let id = self.single_byte_lookup.get(word[0]);
                if id != 0 {
                    ids.push(id - 1);
                    i += 1;
                    continue;
                }
            }

            // === FAST PATH: Hash table lookup ===
            let h = fast_hash(word);
            let bucket_idx = (h & self.hash_mask) as usize;
            let bucket = &self.hash_table[bucket_idx];
            let len = word_len as u8;

            // Quick scalar check first (often succeeds for common words)
            // Unrolled check of first 4 slots - covers most cases with low latency
            let token_id = if bucket.hashes[0] == h && bucket.lengths[0] == len && bucket.ids[0] != 0 {
                bucket.ids[0]
            } else if bucket.hashes[1] == h && bucket.lengths[1] == len && bucket.ids[1] != 0 {
                bucket.ids[1]
            } else if bucket.hashes[2] == h && bucket.lengths[2] == len && bucket.ids[2] != 0 {
                bucket.ids[2]
            } else if bucket.hashes[3] == h && bucket.lengths[3] == len && bucket.ids[3] != 0 {
                bucket.ids[3]
            } else {
                // Check remaining slots with SIMD or scalar
                #[cfg(target_arch = "x86_64")]
                {
                    if self.has_avx2 {
                        // Only check the second half of the bucket with SIMD
                        // First 4 already checked above
                        let id = unsafe { self.hash_lookup_avx2_second_half(bucket, h, len) };
                        id
                    } else {
                        self.hash_lookup_scalar_second_half(bucket, h, len)
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                self.hash_lookup_scalar_second_half(bucket, h, len)
            };

            if token_id != 0 {
                ids.push(token_id - 1);
                i += 1;
                continue;
            }

            // Fallback to subword tokenization
            self.tokenize_subword(word, &mut ids);
            i += 1;
        }

        ids
    }

    /// Tokenize word into subwords using speculative parallel matching
    /// Returns [UNK] for the entire word if any part cannot be tokenized (standard WordPiece behavior)
    ///
    /// Optimization: Uses stack-allocated buffer to avoid heap allocation
    #[inline]
    fn tokenize_subword(&self, word: &[u8], ids: &mut Vec<u32>) {
        // Byte-length optimization: byte_len >= char_len in UTF-8
        // If byte_len <= max_chars, we're safe. Only count chars if byte_len > max.
        if word.len() > self.config.max_input_chars_per_word {
            // Byte length exceeds limit, need to count actual characters
            // Count UTF-8 characters by counting bytes that are NOT continuation bytes
            let char_count = word.iter().filter(|&&b| (b & 0xC0) != 0x80).count();
            if char_count > self.config.max_input_chars_per_word {
                ids.push(self.unk_id);
                return;
            }
        }

        // Stack-allocated buffer (max 128 subwords per word - handles long hex strings, base64, etc.)
        // A 100-char word with single-char tokens would produce ~100 tokens
        let mut temp_ids: [u32; 128] = [0; 128];
        let mut temp_count = 0usize;
        let mut start = 0;
        let mut is_first = true;

        while start < word.len() {
            let remaining = &word[start..];

            // Speculative matching with incremental hash computation
            let token_id = if is_first {
                self.speculative_hash_lookup(remaining)
            } else {
                self.speculative_continuation_lookup(remaining)
            };

            if let Some((id, matched_len)) = token_id {
                if temp_count < 128 {
                    temp_ids[temp_count] = id;
                    temp_count += 1;
                } else {
                    // Very rare: word has >128 subwords, fall back to UNK
                    ids.push(self.unk_id);
                    return;
                }
                start += matched_len;
                is_first = false;
            } else {
                // Fallback to sequential search for rare cases
                let remaining_len = remaining.len();
                let mut end = remaining_len;
                let mut found = false;

                while end > 0 {
                    let slice = &remaining[..end];

                    let tid = if is_first {
                        self.hash_lookup(slice)
                    } else {
                        self.continuation_lookup(slice)
                            .map(|id| id + 1)
                            .unwrap_or(0)
                    };

                    if tid != 0 {
                        if temp_count < 128 {
                            temp_ids[temp_count] = tid - 1;
                            temp_count += 1;
                        } else {
                            ids.push(self.unk_id);
                            return;
                        }
                        start += end;
                        is_first = false;
                        found = true;
                        break;
                    }

                    // Reduce end by one UTF-8 character
                    end -= 1;
                    while end > 0 && (remaining[end] & 0xC0) == 0x80 {
                        end -= 1;
                    }
                }

                if !found {
                    // Cannot tokenize entire word - return [UNK]
                    ids.push(self.unk_id);
                    return;
                }
            }
        }

        // Successfully tokenized - copy from stack buffer
        ids.extend_from_slice(&temp_ids[..temp_count]);
    }

    /// Speculative hash lookup with incremental hash computation
    /// Returns (token_id, matched_length) if found
    ///
    /// Strategy: WordPiece requires LONGEST match first.
    /// Optimization: Compute all prefix hashes in single O(n) pass, inline bucket checks
    /// Uses linear probing (up to 16 buckets) for overflow handling
    #[inline(always)]
    fn speculative_hash_lookup(&self, bytes: &[u8]) -> Option<(u32, usize)> {
        let len = bytes.len();
        if len == 0 {
            return None;
        }

        // Compute all prefix hashes incrementally - O(n) instead of O(n^2)
        // Use 24-char limit to cover most vocabulary tokens (longest common tokens are ~20 chars)
        let max_try = len.min(24);
        let mut prefix_hashes: [u64; 32] = [0; 32];
        let mut valid_lengths: [u8; 32] = [0; 32];
        let mut num_valid = 0usize;

        let mut h: u64 = 0xcbf29ce484222325;
        for i in 0..max_try {
            h ^= bytes[i] as u64;
            h = h.wrapping_mul(0x100000001b3);

            // Check if this is a valid UTF-8 boundary
            let next_pos = i + 1;
            let is_boundary = next_pos >= len || (bytes[next_pos] & 0xC0) != 0x80;

            if is_boundary {
                prefix_hashes[num_valid] = h;
                valid_lengths[num_valid] = next_pos as u8;
                num_valid += 1;
            }
        }

        if num_valid == 0 {
            return None;
        }

        let table_size = self.hash_table.len();

        // Check from longest to shortest with linear probing
        for i in (0..num_valid).rev() {
            let h = prefix_hashes[i];
            let end = valid_lengths[i] as usize;
            let initial_bucket = (h & self.hash_mask) as usize;

            // Prefetch initial bucket for next prefix length to hide latency
            if i > 0 {
                let next_h = prefix_hashes[i - 1];
                let next_bucket_idx = (next_h & self.hash_mask) as usize;
                // SAFETY: next_bucket_idx is bounded by hash_mask
                unsafe { prefetch_read(self.hash_table.as_ptr().add(next_bucket_idx)); }
            }

            // Linear probe up to 16 buckets
            for probe in 0..16 {
                let bucket_idx = (initial_bucket + probe) % table_size;
                let bucket = &self.hash_table[bucket_idx];

                // Prefetch next probe bucket
                if probe < 15 {
                    let next_bucket_idx = (initial_bucket + probe + 1) % table_size;
                    // SAFETY: next_bucket_idx is bounded by table_size
                    unsafe { prefetch_read(self.hash_table.as_ptr().add(next_bucket_idx)); }
                }

                // Count occupied slots
                let mut occupied = 0u8;

                // Inline check for first 4 slots (covers most cases)
                if bucket.hashes[0] == h && bucket.lengths[0] == end as u8 && bucket.ids[0] != 0 {
                    return Some((bucket.ids[0] - 1, end));
                }
                if bucket.ids[0] != 0 { occupied += 1; }

                if bucket.hashes[1] == h && bucket.lengths[1] == end as u8 && bucket.ids[1] != 0 {
                    return Some((bucket.ids[1] - 1, end));
                }
                if bucket.ids[1] != 0 { occupied += 1; }

                if bucket.hashes[2] == h && bucket.lengths[2] == end as u8 && bucket.ids[2] != 0 {
                    return Some((bucket.ids[2] - 1, end));
                }
                if bucket.ids[2] != 0 { occupied += 1; }

                if bucket.hashes[3] == h && bucket.lengths[3] == end as u8 && bucket.ids[3] != 0 {
                    return Some((bucket.ids[3] - 1, end));
                }
                if bucket.ids[3] != 0 { occupied += 1; }

                // Check remaining slots 4-7
                if bucket.hashes[4] == h && bucket.lengths[4] == end as u8 && bucket.ids[4] != 0 {
                    return Some((bucket.ids[4] - 1, end));
                }
                if bucket.ids[4] != 0 { occupied += 1; }

                if bucket.hashes[5] == h && bucket.lengths[5] == end as u8 && bucket.ids[5] != 0 {
                    return Some((bucket.ids[5] - 1, end));
                }
                if bucket.ids[5] != 0 { occupied += 1; }

                if bucket.hashes[6] == h && bucket.lengths[6] == end as u8 && bucket.ids[6] != 0 {
                    return Some((bucket.ids[6] - 1, end));
                }
                if bucket.ids[6] != 0 { occupied += 1; }

                if bucket.hashes[7] == h && bucket.lengths[7] == end as u8 && bucket.ids[7] != 0 {
                    return Some((bucket.ids[7] - 1, end));
                }
                if bucket.ids[7] != 0 { occupied += 1; }

                // If bucket is not full, token would be here if it existed
                if occupied < 8 {
                    break; // Move to next prefix length
                }
                // Otherwise continue probing
            }
        }

        // Try full length for very long tokens
        if len > max_try {
            let id = self.hash_lookup(bytes);
            if id != 0 {
                return Some((id - 1, len));
            }
        }

        None
    }

    /// Speculative continuation lookup with incremental hash computation
    /// Uses longest-match-first order as required by WordPiece
    /// Optimization: O(n) incremental hash, inline bucket checks
    /// Uses linear probing (up to 16 buckets) for overflow handling
    #[inline(always)]
    fn speculative_continuation_lookup(&self, bytes: &[u8]) -> Option<(u32, usize)> {
        let len = bytes.len();
        if len == 0 {
            return None;
        }

        // Compute all prefix hashes incrementally - O(n) instead of O(n^2)
        // Use 20-char limit for continuation tokens (most are ≤15 bytes but some can be longer)
        let max_try = len.min(20);
        let mut prefix_hashes: [u64; 32] = [0; 32];
        let mut valid_lengths: [u8; 32] = [0; 32];
        let mut num_valid = 0usize;

        let mut h: u64 = 0xcbf29ce484222325;
        for i in 0..max_try {
            h ^= bytes[i] as u64;
            h = h.wrapping_mul(0x100000001b3);

            let next_pos = i + 1;
            let is_boundary = next_pos >= len || (bytes[next_pos] & 0xC0) != 0x80;

            if is_boundary {
                prefix_hashes[num_valid] = h;
                valid_lengths[num_valid] = next_pos as u8;
                num_valid += 1;
            }
        }

        if num_valid == 0 {
            return None;
        }

        let table_size = self.continuation_hash_table.len();

        // Check from longest to shortest with linear probing
        for i in (0..num_valid).rev() {
            let h = prefix_hashes[i];
            let end = valid_lengths[i] as usize;
            let initial_bucket = (h & self.continuation_mask) as usize;

            // Prefetch initial bucket for next prefix length to hide latency
            if i > 0 {
                let next_h = prefix_hashes[i - 1];
                let next_bucket_idx = (next_h & self.continuation_mask) as usize;
                // SAFETY: next_bucket_idx is bounded by continuation_mask
                unsafe { prefetch_read(self.continuation_hash_table.as_ptr().add(next_bucket_idx)); }
            }

            // Linear probe up to 16 buckets
            for probe in 0..16 {
                let bucket_idx = (initial_bucket + probe) % table_size;
                let bucket = &self.continuation_hash_table[bucket_idx];

                // Prefetch next probe bucket
                if probe < 15 {
                    let next_bucket_idx = (initial_bucket + probe + 1) % table_size;
                    // SAFETY: next_bucket_idx is bounded by table_size
                    unsafe { prefetch_read(self.continuation_hash_table.as_ptr().add(next_bucket_idx)); }
                }

                // Count occupied slots
                let mut occupied = 0u8;

                // Inline scalar check for all 8 slots
                if bucket.hashes[0] == h && bucket.lengths[0] == end as u8 && bucket.ids[0] != 0 {
                    return Some((bucket.ids[0] - 1, end));
                }
                if bucket.ids[0] != 0 { occupied += 1; }

                if bucket.hashes[1] == h && bucket.lengths[1] == end as u8 && bucket.ids[1] != 0 {
                    return Some((bucket.ids[1] - 1, end));
                }
                if bucket.ids[1] != 0 { occupied += 1; }

                if bucket.hashes[2] == h && bucket.lengths[2] == end as u8 && bucket.ids[2] != 0 {
                    return Some((bucket.ids[2] - 1, end));
                }
                if bucket.ids[2] != 0 { occupied += 1; }

                if bucket.hashes[3] == h && bucket.lengths[3] == end as u8 && bucket.ids[3] != 0 {
                    return Some((bucket.ids[3] - 1, end));
                }
                if bucket.ids[3] != 0 { occupied += 1; }

                if bucket.hashes[4] == h && bucket.lengths[4] == end as u8 && bucket.ids[4] != 0 {
                    return Some((bucket.ids[4] - 1, end));
                }
                if bucket.ids[4] != 0 { occupied += 1; }

                if bucket.hashes[5] == h && bucket.lengths[5] == end as u8 && bucket.ids[5] != 0 {
                    return Some((bucket.ids[5] - 1, end));
                }
                if bucket.ids[5] != 0 { occupied += 1; }

                if bucket.hashes[6] == h && bucket.lengths[6] == end as u8 && bucket.ids[6] != 0 {
                    return Some((bucket.ids[6] - 1, end));
                }
                if bucket.ids[6] != 0 { occupied += 1; }

                if bucket.hashes[7] == h && bucket.lengths[7] == end as u8 && bucket.ids[7] != 0 {
                    return Some((bucket.ids[7] - 1, end));
                }
                if bucket.ids[7] != 0 { occupied += 1; }

                // If bucket is not full, token would be here if it existed
                if occupied < 8 {
                    break; // Move to next prefix length
                }
                // Otherwise continue probing
            }
        }

        // Try full length for very long tokens (uses linear probing via continuation_hash_lookup)
        if len > max_try {
            let id = self.continuation_hash_lookup(bytes);
            if id != 0 {
                return Some((id - 1, len));
            }
        }

        None
    }

    /// Inline pre-tokenization using SIMD block processing
    /// Uses AVX2 (32-byte) or AVX-512 (64-byte) depending on availability
    /// Falls back to scalar for inputs with non-ASCII characters to ensure
    /// correct handling of Unicode punctuation (e.g., fancy quotes like U+2019)
    #[inline]
    fn pre_tokenize_inline(&self, bytes: &[u8]) -> Vec<(usize, usize)> {
        let mut words = Vec::with_capacity(bytes.len() / 5 + 1);
        let len = bytes.len();

        // Check for non-ASCII characters - if found, use scalar path for correctness
        // SIMD paths only handle ASCII punctuation correctly
        let has_non_ascii = bytes.iter().any(|&b| b >= 0x80);
        if has_non_ascii {
            self.pre_tokenize_scalar(bytes, &mut words);
            return words;
        }

        // Try AVX-512 first (64-byte blocks) - ASCII only
        if len >= 64 && is_avx512_available() {
            return self.pre_tokenize_simd_64(bytes);
        }

        // Try AVX2 (32-byte blocks) - ASCII only
        if len >= 32 && is_avx2_available() {
            return self.pre_tokenize_avx2_32(bytes);
        }

        // Scalar path for shorter inputs or non-SIMD platforms
        self.pre_tokenize_scalar(bytes, &mut words);
        words
    }

    /// AVX2-accelerated pre-tokenization using 32-byte blocks
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn pre_tokenize_avx2_32_impl(&self, bytes: &[u8]) -> Vec<(usize, usize)> {
        use std::arch::x86_64::*;

        let mut words = Vec::with_capacity(bytes.len() / 5 + 1);
        let len = bytes.len();
        let mut offset = 0;
        let mut in_word = false;
        let mut word_start = 0usize;

        // Character class constants for AVX2
        let space = _mm256_set1_epi8(b' ' as i8);
        let tab = _mm256_set1_epi8(b'\t' as i8);
        let newline = _mm256_set1_epi8(b'\n' as i8);
        let carriage = _mm256_set1_epi8(b'\r' as i8);

        // Punctuation ranges
        let punct_lo1 = _mm256_set1_epi8(b'!' as i8 - 1);
        let punct_hi1 = _mm256_set1_epi8(b'/' as i8 + 1);
        let punct_lo2 = _mm256_set1_epi8(b':' as i8 - 1);
        let punct_hi2 = _mm256_set1_epi8(b'@' as i8 + 1);
        let punct_lo3 = _mm256_set1_epi8(b'[' as i8 - 1);
        let punct_hi3 = _mm256_set1_epi8(b'`' as i8 + 1);
        let punct_lo4 = _mm256_set1_epi8(b'{' as i8 - 1);
        let punct_hi4 = _mm256_set1_epi8(b'~' as i8 + 1);

        // Process 32-byte blocks
        while offset + 32 <= len {
            let chunk = _mm256_loadu_si256(bytes.as_ptr().add(offset) as *const __m256i);

            // Classify whitespace
            let ws_mask = _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_cmpeq_epi8(chunk, space),
                    _mm256_cmpeq_epi8(chunk, tab)
                ),
                _mm256_or_si256(
                    _mm256_cmpeq_epi8(chunk, newline),
                    _mm256_cmpeq_epi8(chunk, carriage)
                )
            );

            // Classify punctuation (4 ranges)
            let in_range1 = _mm256_and_si256(
                _mm256_cmpgt_epi8(chunk, punct_lo1),
                _mm256_cmpgt_epi8(punct_hi1, chunk)
            );
            let in_range2 = _mm256_and_si256(
                _mm256_cmpgt_epi8(chunk, punct_lo2),
                _mm256_cmpgt_epi8(punct_hi2, chunk)
            );
            let in_range3 = _mm256_and_si256(
                _mm256_cmpgt_epi8(chunk, punct_lo3),
                _mm256_cmpgt_epi8(punct_hi3, chunk)
            );
            let in_range4 = _mm256_and_si256(
                _mm256_cmpgt_epi8(chunk, punct_lo4),
                _mm256_cmpgt_epi8(punct_hi4, chunk)
            );

            let punct_mask = _mm256_or_si256(
                _mm256_or_si256(in_range1, in_range2),
                _mm256_or_si256(in_range3, in_range4)
            );

            // Combine into boundary mask
            let boundary = _mm256_or_si256(ws_mask, punct_mask);
            let boundary_bits = _mm256_movemask_epi8(boundary) as u32;
            let punct_bits = _mm256_movemask_epi8(punct_mask) as u32;

            if boundary_bits == 0 {
                // All 32 bytes are word characters
                if !in_word {
                    word_start = offset;
                    in_word = true;
                }
            } else if boundary_bits == 0xFFFFFFFF {
                // All 32 bytes are boundaries
                if in_word {
                    words.push((word_start, offset));
                    in_word = false;
                }
                // Extract punctuation tokens
                let mut p = punct_bits;
                while p != 0 {
                    let pos = p.trailing_zeros() as usize;
                    words.push((offset + pos, offset + pos + 1));
                    p &= p - 1;
                }
            } else {
                // Mixed: process bit by bit
                let mut pos = 0u32;
                while pos < 32 {
                    let abs_pos = offset + pos as usize;
                    let is_boundary = (boundary_bits >> pos) & 1 == 1;
                    let is_punct = (punct_bits >> pos) & 1 == 1;

                    if is_boundary {
                        if in_word {
                            words.push((word_start, abs_pos));
                            in_word = false;
                        }
                        if is_punct {
                            words.push((abs_pos, abs_pos + 1));
                        }
                        pos += 1;
                    } else {
                        if !in_word {
                            word_start = abs_pos;
                            in_word = true;
                        }
                        // Skip to next boundary
                        let remaining = boundary_bits >> pos;
                        if remaining != 0 {
                            pos += remaining.trailing_zeros();
                        } else {
                            pos = 32;
                        }
                    }
                }
            }

            offset += 32;
        }

        // Handle remaining bytes (including multi-byte UTF-8)
        while offset < len {
            let b = bytes[offset];

            // ASCII whitespace
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                if in_word {
                    words.push((word_start, offset));
                    in_word = false;
                }
                offset += 1;
                continue;
            }

            // ASCII punctuation
            if matches!(b, b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~') {
                if in_word {
                    words.push((word_start, offset));
                    in_word = false;
                }
                words.push((offset, offset + 1));
                offset += 1;
                continue;
            }

            // Multi-byte UTF-8 character
            if b >= 0x80 {
                let char_len = Self::utf8_len(b);
                let char_end = (offset + char_len).min(len);
                let char_bytes = &bytes[offset..char_end];

                // Check if it's CJK or Unicode punctuation
                let is_cjk = self.config.tokenize_chinese_chars && Self::is_cjk(char_bytes);
                let is_unicode_punct = Self::is_unicode_punctuation(char_bytes);

                if is_cjk || is_unicode_punct {
                    if in_word {
                        words.push((word_start, offset));
                        in_word = false;
                    }
                    words.push((offset, char_end));
                    offset = char_end;
                    continue;
                }

                // Regular multi-byte char, keep in word
                if !in_word {
                    word_start = offset;
                    in_word = true;
                }
                offset = char_end;
            } else {
                // Regular ASCII character
                if !in_word {
                    word_start = offset;
                    in_word = true;
                }
                offset += 1;
            }
        }

        if in_word {
            words.push((word_start, len));
        }

        words
    }

    /// AVX2 pre-tokenization wrapper (safe entry point)
    #[inline]
    fn pre_tokenize_avx2_32(&self, bytes: &[u8]) -> Vec<(usize, usize)> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.pre_tokenize_avx2_32_impl(bytes) };
            }
        }
        // Fallback
        let mut words = Vec::with_capacity(bytes.len() / 5 + 1);
        self.pre_tokenize_scalar(bytes, &mut words);
        words
    }

    /// SIMD-accelerated pre-tokenization using 64-byte blocks
    /// Based on simdjson's approach: classify 64 bytes at once, extract positions with tzcnt
    #[inline]
    fn pre_tokenize_simd_64(&self, bytes: &[u8]) -> Vec<(usize, usize)> {
        let mut words = Vec::with_capacity(bytes.len() / 5 + 1);
        let len = bytes.len();
        let mut offset = 0;
        let mut in_word = false;
        let mut word_start = 0usize;

        // Process 64-byte blocks
        while offset + 64 <= len {
            // Get bitmasks for whitespace and punctuation
            let ws_mask = classify_whitespace(&bytes[offset..]);
            let punct_mask = classify_punctuation(&bytes[offset..]);

            // Combine: word boundaries are whitespace OR punctuation
            let boundary_mask = ws_mask | punct_mask;

            // Process each position using tzcnt (trailing zero count)
            if boundary_mask == 0 {
                // All 64 bytes are word characters
                if !in_word {
                    word_start = offset;
                    in_word = true;
                }
            } else if boundary_mask == !0u64 {
                // All 64 bytes are boundaries
                if in_word {
                    words.push((word_start, offset));
                    in_word = false;
                }
                // Each punctuation is its own token (process punct_mask)
                Self::extract_punct_tokens(punct_mask, offset, &mut words);
            } else {
                // Mixed: process bit by bit using tzcnt
                self.process_mixed_block_64(
                    bytes, offset, ws_mask, punct_mask,
                    &mut words, &mut in_word, &mut word_start
                );
            }

            offset += 64;
        }

        // Handle remaining bytes with scalar code
        if offset < len {
            self.process_remaining_scalar(
                bytes, offset, &mut words, &mut in_word, &mut word_start
            );
        }

        // Close final word
        if in_word {
            words.push((word_start, len));
        }

        words
    }

    /// Process a 64-byte block with mixed word/boundary characters
    #[inline]
    fn process_mixed_block_64(
        &self,
        bytes: &[u8],
        offset: usize,
        ws_mask: u64,
        punct_mask: u64,
        words: &mut Vec<(usize, usize)>,
        in_word: &mut bool,
        word_start: &mut usize,
    ) {
        // Process transitions using bitmask arithmetic
        let boundary_mask = ws_mask | punct_mask;
        let _word_mask = !boundary_mask; // Keep for potential future optimization

        let mut pos = 0u32;
        while pos < 64 {
            let abs_pos = offset + pos as usize;

            if (boundary_mask >> pos) & 1 == 1 {
                // Boundary character
                if *in_word {
                    words.push((*word_start, abs_pos));
                    *in_word = false;
                }
                // Punctuation gets its own token
                if (punct_mask >> pos) & 1 == 1 {
                    // Check for multi-byte character
                    if bytes[abs_pos] >= 0x80 {
                        let char_len = Self::utf8_len(bytes[abs_pos]);
                        let char_end = (abs_pos + char_len).min(bytes.len());
                        words.push((abs_pos, char_end));
                        pos += char_len as u32;
                        continue;
                    } else {
                        words.push((abs_pos, abs_pos + 1));
                    }
                }
                pos += 1;
            } else {
                // Word character
                if !*in_word {
                    *word_start = abs_pos;
                    *in_word = true;
                }
                // Skip to next boundary using tzcnt
                let remaining_boundary = boundary_mask >> pos;
                if remaining_boundary != 0 {
                    let skip = remaining_boundary.trailing_zeros();
                    pos += skip;
                } else {
                    pos = 64;
                }
            }
        }
    }

    /// Extract punctuation tokens from bitmask
    #[inline]
    fn extract_punct_tokens(mut punct_mask: u64, offset: usize, words: &mut Vec<(usize, usize)>) {
        while punct_mask != 0 {
            let pos = punct_mask.trailing_zeros() as usize;
            words.push((offset + pos, offset + pos + 1));
            punct_mask &= punct_mask - 1;
        }
    }

    /// Process remaining bytes after SIMD blocks (scalar)
    #[inline]
    fn process_remaining_scalar(
        &self,
        bytes: &[u8],
        start: usize,
        words: &mut Vec<(usize, usize)>,
        in_word: &mut bool,
        word_start: &mut usize,
    ) {
        let mut i = start;
        while i < bytes.len() {
            let b = bytes[i];

            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                // Whitespace
                if *in_word {
                    words.push((*word_start, i));
                    *in_word = false;
                }
                i += 1;
            } else if matches!(b, b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~') {
                // ASCII punctuation
                if *in_word {
                    words.push((*word_start, i));
                    *in_word = false;
                }
                words.push((i, i + 1));
                i += 1;
            } else if b >= 0x80 {
                // Multi-byte character
                let char_len = Self::utf8_len(b);
                let char_end = (i + char_len).min(bytes.len());
                let char_bytes = &bytes[i..char_end];

                let is_cjk = self.config.tokenize_chinese_chars && Self::is_cjk(char_bytes);
                let is_punct = Self::is_unicode_punctuation(char_bytes);

                if is_cjk || is_punct {
                    if *in_word {
                        words.push((*word_start, i));
                        *in_word = false;
                    }
                    words.push((i, char_end));
                } else {
                    if !*in_word {
                        *word_start = i;
                        *in_word = true;
                    }
                }
                i = char_end;
            } else {
                // Regular word character
                if !*in_word {
                    *word_start = i;
                    *in_word = true;
                }
                i += 1;
            }
        }
    }

    /// Scalar pre-tokenization (fallback for short inputs)
    #[inline]
    fn pre_tokenize_scalar(&self, bytes: &[u8], words: &mut Vec<(usize, usize)>) {
        let mut i = 0;

        while i < bytes.len() {
            // Skip whitespace
            while i < bytes.len() && matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
                i += 1;
            }

            if i >= bytes.len() {
                break;
            }

            let word_start = i;
            let mut handled_special = false;

            // Find word end
            while i < bytes.len() {
                let b = bytes[i];

                if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                    break;
                }

                // ASCII punctuation - split as separate token
                if matches!(b, b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~') {
                    if i > word_start {
                        words.push((word_start, i));
                    }
                    words.push((i, i + 1));
                    i += 1;
                    handled_special = true;
                    break;
                }

                // Handle multi-byte characters (CJK, non-ASCII punctuation)
                if b >= 0x80 {
                    let char_len = Self::utf8_len(b);
                    let char_end = (i + char_len).min(bytes.len());
                    let char_bytes = &bytes[i..char_end];

                    // Check if this is CJK or non-ASCII punctuation
                    let is_cjk = self.config.tokenize_chinese_chars && Self::is_cjk(char_bytes);
                    let is_punct = Self::is_unicode_punctuation(char_bytes);

                    if is_cjk || is_punct {
                        if i > word_start {
                            words.push((word_start, i));
                        }
                        words.push((i, char_end));
                        i = char_end;
                        handled_special = true;
                        break;
                    }

                    // Regular multi-byte char, advance
                    i += char_len;
                } else {
                    i += 1;
                }
            }

            // Push remaining word (only if we didn't handle a special case)
            if !handled_special && i > word_start {
                words.push((word_start, i));
            }
        }
    }

    #[inline(always)]
    fn utf8_len(b: u8) -> usize {
        if b < 0x80 { 1 }
        else if b < 0xE0 { 2 }
        else if b < 0xF0 { 3 }
        else { 4 }
    }

    #[inline]
    fn is_cjk(bytes: &[u8]) -> bool {
        if bytes.len() < 3 {
            return false;
        }

        let b0 = bytes[0] as u32;
        let b1 = bytes.get(1).copied().unwrap_or(0) as u32;
        let b2 = bytes.get(2).copied().unwrap_or(0) as u32;

        let cp = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);

        matches!(cp, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF)
    }

    /// Check if UTF-8 bytes represent a Unicode punctuation character
    #[inline]
    fn is_unicode_punctuation(bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return false;
        }

        // Decode UTF-8 to codepoint
        let cp = match bytes.len() {
            1 => bytes[0] as u32,
            2 => {
                let b0 = bytes[0] as u32;
                let b1 = bytes.get(1).copied().unwrap_or(0) as u32;
                ((b0 & 0x1F) << 6) | (b1 & 0x3F)
            }
            3 => {
                let b0 = bytes[0] as u32;
                let b1 = bytes.get(1).copied().unwrap_or(0) as u32;
                let b2 = bytes.get(2).copied().unwrap_or(0) as u32;
                ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F)
            }
            _ => return false,
        };

        // Unicode punctuation - matches HuggingFace's _is_punctuation which checks if
        // unicodedata.category(char).startswith("P")
        // Categories: Pc, Pd, Pe, Pf, Pi, Po, Ps
        //
        // Note: This is a subset of common Unicode punctuation characters.
        // For complete accuracy, a full Unicode category table would be needed.
        matches!(cp,
            // Latin-1 Supplement punctuation (category Po mostly)
            0x00A1 |           // ¡ INVERTED EXCLAMATION MARK (Po)
            0x00A7 |           // § SECTION SIGN (Po)
            0x00AB |           // « LEFT-POINTING DOUBLE ANGLE QUOTATION MARK (Pi)
            0x00B6 |           // ¶ PILCROW SIGN (Po)
            0x00B7 |           // · MIDDLE DOT (Po)
            0x00BB |           // » RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK (Pf)
            0x00BF |           // ¿ INVERTED QUESTION MARK (Po)
            // General Punctuation (U+2000-U+206F) - excluding symbols (Sm)
            0x2010..=0x2027 |  // Hyphens (Pd), dashes, quotation marks (Pi/Pf), etc.
            0x2030..=0x203A |  // Per mille, prime, ‹ › quotation marks (Po/Pi/Pf)
            0x203B..=0x203E |  // Reference marks (Po)
            0x2041..=0x2043 |  // Caret, bullet, hyphen (Pc/Po/Pd)
            // Note: U+2044 (FRACTION SLASH) is Sm (Symbol, Math) - NOT punctuation!
            0x2045..=0x2051 |  // Brackets, bullets (Ps/Pe/Po)
            0x2053..=0x205E |  // Swung dash, flower, punctuation space (Po/Pd/Ps/Pe)
            // Dingbats - quotation marks
            0x275B..=0x275E |  // Heavy single/double quotation marks (Ps/Pe)
            0x276C..=0x2771 |  // Heavy angle brackets (Ps/Pe) - includes ❮❯
            // CJK Symbols and Punctuation (U+3000-U+303F)
            0x3001..=0x3003 |  // Ideographic comma, full stop, etc. (Po)
            0x3008..=0x3011 |  // CJK brackets (Ps/Pe)
            0x3014..=0x301F |  // More CJK brackets (Ps/Pe)
            // Katakana middle dots and other punctuation-like
            0x30FB |           // ・ KATAKANA MIDDLE DOT (Po)
            // Halfwidth and Fullwidth Forms (U+FF00-U+FFEF)
            0xFF01..=0xFF0F |  // Fullwidth ! to / (Po mostly)
            0xFF1A..=0xFF20 |  // Fullwidth : to @ (Po)
            0xFF3B..=0xFF40 |  // Fullwidth [ to ` (Ps/Pe/Pc)
            0xFF5B..=0xFF65 |  // Fullwidth { to halfwidth middle dot (Ps/Pe/Po)
            // Small Form Variants
            0xFE50..=0xFE6F    // Small forms of punctuation (Po/Pd)
        )
    }

    /// Check if character is a control character that should be stripped
    #[inline]
    fn is_control_char(c: char) -> bool {
        // Control characters (Cc category) except whitespace
        if c == '\t' || c == '\n' || c == '\r' {
            return false; // Keep these whitespace chars
        }
        // Unicode control characters
        matches!(c,
            '\u{0000}'..='\u{001F}' |  // C0 controls
            '\u{007F}'..='\u{009F}' |  // Delete and C1 controls
            '\u{FEFF}' |               // BOM / Zero-width no-break space
            '\u{200B}'..='\u{200F}' |  // Zero-width spaces and direction marks
            '\u{202A}'..='\u{202E}' |  // Bidi formatting
            '\u{2060}'..='\u{206F}'    // Word joiner and other format chars
        )
    }

    /// Encode with special tokens
    #[inline]
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 4 + 3);

        if let Some(cls) = self.cls_id {
            ids.push(cls);
        }

        ids.extend(self.encode_fast(text));

        if let Some(sep) = self.sep_id {
            ids.push(sep);
        }

        ids
    }

    /// Batch encode with parallel processing
    pub fn encode_batch_fast(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        if texts.is_empty() {
            return Vec::new();
        }

        if texts.len() == 1 {
            return vec![self.encode_fast(texts[0])];
        }

        texts.par_iter()
            .map(|text| self.encode_fast(text))
            .collect()
    }

    /// Batch encode with special tokens
    pub fn encode_batch_with_special(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        if texts.is_empty() {
            return Vec::new();
        }

        if texts.len() == 1 {
            return vec![self.encode_with_special(texts[0])];
        }

        texts.par_iter()
            .map(|text| self.encode_with_special(text))
            .collect()
    }

    /// Decode tokens
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut tokens = Vec::with_capacity(ids.len());

        for &id in ids {
            if let Some(token) = self.vocabulary.id_to_token(id) {
                if skip_special && token.starts_with('[') && token.ends_with(']') {
                    continue;
                }

                let t = if token.starts_with(&self.config.continuing_subword_prefix) {
                    &token[self.config.continuing_subword_prefix.len()..]
                } else {
                    token
                };
                tokens.push(t.to_string());
            }
        }

        let mut result = String::new();
        for (i, t) in tokens.iter().enumerate() {
            if i > 0 && !t.is_empty() {
                result.push(' ');
            }
            result.push_str(t);
        }
        result
    }

    /// Convert encoding to Encoding struct
    pub fn encode_to_encoding(&self, text: &str, add_special_tokens: bool) -> crate::encoding::Encoding {
        let ids = if add_special_tokens {
            self.encode_with_special(text)
        } else {
            self.encode_fast(text)
        };

        let mut encoding = crate::encoding::Encoding::with_capacity(ids.len());
        for &id in &ids {
            let token = self.vocabulary.id_to_token(id).unwrap_or("[UNK]").to_string();
            let is_special = token.starts_with('[') && token.ends_with(']');
            encoding.push(id, token, (0, 0), None, Some(0), is_special);
        }
        encoding
    }
}

// Implement Tokenizer trait for HyperWordPieceTokenizer
impl crate::tokenizer::Tokenizer for HyperWordPieceTokenizer {
    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> crate::error::Result<crate::encoding::Encoding> {
        Ok(self.encode_to_encoding(text, add_special_tokens))
    }

    fn encode_pair(
        &self,
        text: &str,
        text_pair: &str,
        add_special_tokens: bool,
    ) -> crate::error::Result<crate::encoding::Encoding> {
        let mut encoding = self.encode_to_encoding(text, false);
        let pair_encoding = self.encode_to_encoding(text_pair, false);

        // Update sequence IDs for second sequence
        let mut pair_with_seq_id = crate::encoding::Encoding::with_capacity(pair_encoding.len());
        for i in 0..pair_encoding.len() {
            pair_with_seq_id.push(
                pair_encoding.get_ids()[i],
                pair_encoding.get_tokens()[i].clone(),
                pair_encoding.get_offsets()[i],
                pair_encoding.get_word_ids()[i],
                Some(1),
                pair_encoding.get_special_tokens_mask()[i] == 1,
            );
        }

        encoding.merge(pair_with_seq_id, true);

        if add_special_tokens {
            // Add [CLS] at start and [SEP] after each sequence
            let mut with_special = crate::encoding::Encoding::with_capacity(encoding.len() + 3);

            if let Some(cls_id) = self.cls_id {
                with_special.push(cls_id, "[CLS]".to_string(), (0, 0), None, Some(0), true);
            }

            for i in 0..encoding.len() {
                with_special.push(
                    encoding.get_ids()[i],
                    encoding.get_tokens()[i].clone(),
                    encoding.get_offsets()[i],
                    encoding.get_word_ids()[i],
                    encoding.get_sequence_ids()[i],
                    encoding.get_special_tokens_mask()[i] == 1,
                );
            }

            if let Some(sep_id) = self.sep_id {
                with_special.push(sep_id, "[SEP]".to_string(), (0, 0), None, Some(0), true);
                with_special.push(sep_id, "[SEP]".to_string(), (0, 0), None, Some(1), true);
            }

            encoding = with_special;
        }

        Ok(encoding)
    }

    fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> crate::error::Result<Vec<crate::encoding::Encoding>> {
        use rayon::prelude::*;

        if texts.is_empty() {
            return Ok(Vec::new());
        }

        if texts.len() == 1 {
            return Ok(vec![self.encode_to_encoding(texts[0], add_special_tokens)]);
        }

        Ok(texts
            .par_iter()
            .map(|text| self.encode_to_encoding(text, add_special_tokens))
            .collect())
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> crate::error::Result<String> {
        Ok(HyperWordPieceTokenizer::decode(self, ids, skip_special_tokens))
    }

    fn save(&self, path: &std::path::Path) -> crate::error::Result<()> {
        use crate::config::{TokenizerConfig, ModelConfig, VocabFormat, AddedToken};

        let mut added_tokens = Vec::new();
        if let Some(token) = self.vocabulary().pad_token() {
            if let Some(id) = self.vocabulary().token_to_id(token) {
                added_tokens.push(AddedToken { id, content: token.to_string(), single_word: false, lstrip: false, rstrip: false, normalized: true, special: true });
            }
        }
        if let Some(token) = self.vocabulary().unk_token() {
            if let Some(id) = self.vocabulary().token_to_id(token) {
                added_tokens.push(AddedToken { id, content: token.to_string(), single_word: false, lstrip: false, rstrip: false, normalized: true, special: true });
            }
        }
        if let Some(token) = self.vocabulary().cls_token() {
            if let Some(id) = self.vocabulary().token_to_id(token) {
                added_tokens.push(AddedToken { id, content: token.to_string(), single_word: false, lstrip: false, rstrip: false, normalized: true, special: true });
            }
        }
        if let Some(token) = self.vocabulary().sep_token() {
            if let Some(id) = self.vocabulary().token_to_id(token) {
                added_tokens.push(AddedToken { id, content: token.to_string(), single_word: false, lstrip: false, rstrip: false, normalized: true, special: true });
            }
        }

        let config = TokenizerConfig {
            version: "1.0".to_string(),
            truncation: None,
            padding: None,
            added_tokens,
            normalizer: Some(crate::config::NormalizerConfig {
                normalizer_type: "BertNormalizer".to_string(),
                clean_text: Some(true),
                handle_chinese_chars: Some(self.config.tokenize_chinese_chars),
                strip_accents: Some(self.config.strip_accents),
                lowercase: Some(self.config.do_lower_case),
                normalizers: None,
                pattern: None,
                content: None,
            }),
            pre_tokenizer: Some(crate::config::PreTokenizerConfig {
                pretokenizer_type: "BertPreTokenizer".to_string(),
                add_prefix_space: None,
                replacement: None,
                use_regex: None,
                pretokenizers: None,
                pattern: None,
                behavior: None,
                invert: None,
            }),
            post_processor: Some(crate::config::PostProcessorConfig {
                processor_type: "BertProcessing".to_string(),
                single: None,
                pair: None,
                special_tokens: None,
                cls: self.vocabulary().cls_token().and_then(|t| self.vocabulary().token_to_id(t).map(|id| (t.to_string(), id))),
                sep: self.vocabulary().sep_token().and_then(|t| self.vocabulary().token_to_id(t).map(|id| (t.to_string(), id))),
            }),
            decoder: Some(crate::config::DecoderConfig {
                decoder_type: "WordPiece".to_string(),
                prefix: Some(self.config.continuing_subword_prefix.clone()),
                cleanup: Some(true),
                replacement: None,
                add_prefix_space: None,
                decoders: None,
            }),
            model: ModelConfig {
                model_type: "WordPiece".to_string(),
                unk_token: Some(self.config.unk_token.clone()),
                unk_id: None,
                continuing_subword_prefix: Some(self.config.continuing_subword_prefix.clone()),
                max_input_chars_per_word: Some(self.config.max_input_chars_per_word),
                vocab: VocabFormat::Dict(self.vocabulary().token_to_id_map().iter().map(|(k, &v)| (k.clone(), v)).collect()),
                merges: None,
                end_of_word_suffix: None,
                fuse_unk: None,
                byte_fallback: None,
                dropout: None,
            },
        };

        config.save(path, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::VocabularyBuilder;

    fn create_test_tokenizer() -> HyperWordPieceTokenizer {
        let vocab = VocabularyBuilder::new()
            .add_tokens([
                "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                "hello", "world", "##ing", "##ed", "un", "play", "##play", "##er", "the", "a",
            ])
            .unk_token("[UNK]")
            .pad_token("[PAD]")
            .cls_token("[CLS]")
            .sep_token("[SEP]")
            .build();

        HyperWordPieceTokenizer::new(vocab, HyperConfig::default())
    }

    #[test]
    fn test_encode_fast() {
        let tokenizer = create_test_tokenizer();
        let ids = tokenizer.encode_fast("hello world");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_encode_with_special() {
        let tokenizer = create_test_tokenizer();
        let ids = tokenizer.encode_with_special("hello");
        assert!(ids.len() >= 3);
        assert_eq!(ids[0], 2); // [CLS]
        assert_eq!(*ids.last().unwrap(), 3); // [SEP]
    }

    #[test]
    fn test_batch_encode() {
        let tokenizer = create_test_tokenizer();
        let texts = vec!["hello world", "the quick fox"];
        let results = tokenizer.encode_batch_fast(&texts);
        assert_eq!(results.len(), 2);
    }
}
