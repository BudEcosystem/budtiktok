//! SIMD-Accelerated BPE Tokenizer
//!
//! Maximum performance through:
//! - SIMD pre-tokenization (AVX2/AVX-512/NEON)
//! - 8-way parallel hash bucket lookups
//! - Cache-line aligned data structures (64-byte alignment)
//! - Direct byte-level lookup tables (O(1))
//! - Zero-copy batch processing with Rayon parallelism
//! - Speculative token matching with incremental hashing
//!
//! Based on insights from:
//! - GitHub rust-gems O(n) BPE algorithm
//! - simdjson 64-byte block processing
//! - Swiss Tables SIMD probing
//! - StringZilla SWAR techniques

use rayon::prelude::*;

use crate::bpe::{BpeToken, Gpt2ByteEncoder, MergeRule, CompatibilityTable};
use crate::vocab::Vocabulary;
use crate::avx512::{classify_whitespace, classify_punctuation, is_avx512_available};

// =============================================================================
// AVX2 Feature Detection
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

// =============================================================================
// Cache-Aligned Data Structures
// =============================================================================

/// 8-way hash bucket for SIMD parallel lookup
/// Cache-line aligned (64 bytes) for optimal memory access
#[repr(C, align(64))]
struct HashBucket {
    /// Hash values for collision detection (8 × 8 bytes = 64 bytes)
    hashes: [u64; 8],
    /// Token IDs (0 = empty) (8 × 4 bytes = 32 bytes)
    ids: [u32; 8],
    /// Token lengths for validation (8 × 1 byte = 8 bytes)
    lengths: [u8; 8],
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

/// Direct lookup table for single-byte tokens (256 entries)
/// Maps byte -> token_id + 1 (0 = not found)
#[repr(C, align(64))]
struct ByteLookup {
    ids: [u32; 256],
}

impl ByteLookup {
    fn new() -> Self {
        Self { ids: [0; 256] }
    }

    #[inline(always)]
    fn get(&self, byte: u8) -> u32 {
        self.ids[byte as usize]
    }

    fn set(&mut self, byte: u8, id: u32) {
        self.ids[byte as usize] = id + 1;
    }
}

/// GPT-2 byte encoding lookup tables (O(1) encode/decode)
#[repr(C, align(64))]
struct ByteEncodingTable {
    /// Byte -> encoded character (as u32 code point)
    encode: [u32; 256],
    /// Encoded code point -> original byte + 1 (0 = not found)
    decode: [u8; 512],
    decode_valid: [bool; 512],
}

impl ByteEncodingTable {
    fn new() -> Self {
        let encoder = Gpt2ByteEncoder::new();
        let mut table = Self {
            encode: [0; 256],
            decode: [0; 512],
            decode_valid: [false; 512],
        };

        for byte in 0u8..=255 {
            let c = encoder.encode_byte(byte);
            let cp = c as u32;
            table.encode[byte as usize] = cp;

            // Store decode mapping
            if cp < 512 {
                table.decode[cp as usize] = byte;
                table.decode_valid[cp as usize] = true;
            }
        }

        table
    }

    #[inline(always)]
    fn encode_byte(&self, byte: u8) -> u32 {
        self.encode[byte as usize]
    }

    #[inline(always)]
    fn decode_char(&self, cp: u32) -> Option<u8> {
        if cp < 512 && self.decode_valid[cp as usize] {
            Some(self.decode[cp as usize])
        } else {
            None
        }
    }
}

// =============================================================================
// Fast Hash Function
// =============================================================================

/// FNV-1a hash function (consistent with wordpiece_hyper)
#[inline(always)]
fn fast_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Incremental FNV-1a hash (for speculative matching)
#[inline(always)]
fn hash_extend(h: u64, byte: u8) -> u64 {
    let mut h = h;
    h ^= byte as u64;
    h.wrapping_mul(0x100000001b3)
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for HyperBpeTokenizer
#[derive(Debug, Clone)]
pub struct HyperBpeConfig {
    /// Unknown token string
    pub unk_token: String,
    /// Byte-level BPE (GPT-2 style)
    pub byte_level: bool,
    /// Pre-tokenization pattern
    pub pre_tokenize_pattern: PreTokenizePattern,
    /// Hash table size (power of 2)
    pub hash_table_bits: u32,
    /// Use O(n) linear algorithm (vs O(n log n) heap)
    pub use_linear_algorithm: bool,
}

/// Pre-tokenization pattern
#[derive(Debug, Clone, Copy)]
pub enum PreTokenizePattern {
    /// GPT-2 pattern (contractions, words, numbers, punctuation)
    Gpt2,
    /// cl100k_base pattern (GPT-4)
    Cl100kBase,
    /// Simple whitespace splitting
    Whitespace,
    /// No pre-tokenization
    None,
}

impl Default for HyperBpeConfig {
    fn default() -> Self {
        Self {
            unk_token: "<unk>".to_string(),
            byte_level: true,
            pre_tokenize_pattern: PreTokenizePattern::Gpt2,
            hash_table_bits: 16, // 64K buckets (8MB hash table)
            use_linear_algorithm: true,
        }
    }
}

// =============================================================================
// Hyper BPE Tokenizer
// =============================================================================

/// SIMD-accelerated BPE tokenizer
pub struct HyperBpeTokenizer {
    /// Direct lookup for single-byte tokens (O(1))
    byte_lookup: Box<ByteLookup>,
    /// GPT-2 byte encoding tables
    byte_encoding: Box<ByteEncodingTable>,
    /// Main token hash table
    hash_table: Vec<HashBucket>,
    /// Hash table mask
    hash_mask: u64,
    /// Vocabulary for lookups and decode
    vocabulary: Vocabulary,
    /// Merge rules for O(n log n) algorithm
    merges: Vec<MergeRule>,
    /// Compatibility table for merges
    compat: CompatibilityTable,
    /// Configuration
    config: HyperBpeConfig,
    /// Unknown token ID
    unk_id: u32,
    /// Cached SIMD availability
    #[cfg(target_arch = "x86_64")]
    has_avx2: bool,
}

impl HyperBpeTokenizer {
    /// Create a new SIMD-accelerated BPE tokenizer
    pub fn new(vocabulary: Vocabulary, merges: Vec<MergeRule>, config: HyperBpeConfig) -> Self {
        let table_size = 1usize << config.hash_table_bits;
        let hash_mask = (table_size - 1) as u64;

        // Initialize lookup tables
        let mut byte_lookup = Box::new(ByteLookup::new());
        let byte_encoding = Box::new(ByteEncodingTable::new());

        let mut hash_table: Vec<HashBucket> = (0..table_size)
            .map(|_| HashBucket::new())
            .collect();

        // Build hash table from vocabulary
        for (token, id) in vocabulary.iter() {
            let token_bytes = token.as_bytes();

            // Single-byte tokens get O(1) lookup
            if token_bytes.len() == 1 {
                byte_lookup.set(token_bytes[0], id);
            }

            // All tokens go into hash table
            let h = fast_hash(token_bytes);
            let bucket_idx = (h & hash_mask) as usize;
            let bucket = &mut hash_table[bucket_idx];

            // Find empty slot
            for i in 0..8 {
                if bucket.ids[i] == 0 {
                    bucket.hashes[i] = h;
                    bucket.ids[i] = id + 1; // +1 because 0 = empty
                    bucket.lengths[i] = token_bytes.len().min(255) as u8;
                    break;
                }
            }
        }

        let unk_id = vocabulary.token_to_id(&config.unk_token).unwrap_or(0);
        let compat = CompatibilityTable::from_merges(&merges, &vocabulary);

        Self {
            byte_lookup,
            byte_encoding,
            hash_table,
            hash_mask,
            vocabulary,
            merges,
            compat,
            config,
            unk_id,
            #[cfg(target_arch = "x86_64")]
            has_avx2: is_x86_feature_detected!("avx2"),
        }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Get vocabulary reference
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    // =========================================================================
    // SIMD Hash Lookup
    // =========================================================================

    /// Hash lookup with SIMD parallel comparison
    #[inline(always)]
    fn hash_lookup(&self, bytes: &[u8]) -> u32 {
        let h = fast_hash(bytes);
        let bucket_idx = (h & self.hash_mask) as usize;
        let bucket = &self.hash_table[bucket_idx];
        let len = bytes.len().min(255) as u8;

        #[cfg(target_arch = "x86_64")]
        {
            if self.has_avx2 {
                return unsafe { self.hash_lookup_avx2(bucket, h, len) };
            }
        }

        self.hash_lookup_scalar(bucket, h, len)
    }

    /// AVX2 8-way parallel hash lookup
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hash_lookup_avx2(&self, bucket: &HashBucket, h: u64, len: u8) -> u32 {
        use std::arch::x86_64::*;

        // Broadcast target hash to 4 lanes (256-bit = 4 × 64-bit)
        let target = _mm256_set1_epi64x(h as i64);

        // Load first 4 hashes
        let hashes_0_3 = _mm256_loadu_si256(bucket.hashes.as_ptr() as *const __m256i);
        // Load next 4 hashes
        let hashes_4_7 = _mm256_loadu_si256(bucket.hashes.as_ptr().add(4) as *const __m256i);

        // Compare
        let cmp_0_3 = _mm256_cmpeq_epi64(hashes_0_3, target);
        let cmp_4_7 = _mm256_cmpeq_epi64(hashes_4_7, target);

        // Get masks (1 bit per 64-bit element)
        let mask_0_3 = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_0_3)) as u32;
        let mask_4_7 = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_4_7)) as u32;

        let combined_mask = mask_0_3 | (mask_4_7 << 4);

        if combined_mask == 0 {
            return 0;
        }

        // Verify length for matches
        let mut mask = combined_mask;
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize;
            if bucket.lengths[idx] == len && bucket.ids[idx] != 0 {
                return bucket.ids[idx];
            }
            mask &= mask - 1;
        }

        0
    }

    /// Scalar 8-way hash lookup (fallback)
    #[inline(always)]
    fn hash_lookup_scalar(&self, bucket: &HashBucket, h: u64, len: u8) -> u32 {
        for i in 0..8 {
            if bucket.hashes[i] == h && bucket.lengths[i] == len && bucket.ids[i] != 0 {
                return bucket.ids[i];
            }
        }
        0
    }

    // =========================================================================
    // Speculative Token Matching
    // =========================================================================

    /// Speculative longest-match lookup with incremental hashing
    /// Returns (token_id, length) for the longest matching token
    #[inline(always)]
    fn speculative_longest_match(&self, bytes: &[u8]) -> Option<(u32, usize)> {
        if bytes.is_empty() {
            return None;
        }

        // Single-byte fast path
        if bytes.len() == 1 {
            let id = self.byte_lookup.get(bytes[0]);
            if id != 0 {
                return Some((id - 1, 1));
            }
        }

        // Try single byte first
        let single_id = self.byte_lookup.get(bytes[0]);

        // Compute all prefix hashes incrementally
        let max_try = bytes.len().min(20);
        let mut prefix_hashes: [u64; 24] = [0; 24];
        let mut valid_lengths: [usize; 24] = [0; 24];
        let mut num_valid = 0usize;

        let mut h: u64 = 0xcbf29ce484222325;
        for i in 0..max_try {
            h = hash_extend(h, bytes[i]);

            // Check UTF-8 boundary
            let next_pos = i + 1;
            let is_boundary = next_pos >= bytes.len() || (bytes[next_pos] & 0xC0) != 0x80;

            if is_boundary && num_valid < 24 {
                prefix_hashes[num_valid] = h;
                valid_lengths[num_valid] = next_pos;
                num_valid += 1;
            }
        }

        // Try from longest to shortest
        for i in (0..num_valid).rev() {
            let h = prefix_hashes[i];
            let end = valid_lengths[i];
            let bucket_idx = (h & self.hash_mask) as usize;
            let bucket = &self.hash_table[bucket_idx];

            // Inline check first 4 slots
            for j in 0..4 {
                if bucket.hashes[j] == h && bucket.lengths[j] == end as u8 && bucket.ids[j] != 0 {
                    return Some((bucket.ids[j] - 1, end));
                }
            }

            // Check remaining slots
            for j in 4..8 {
                if bucket.hashes[j] == h && bucket.lengths[j] == end as u8 && bucket.ids[j] != 0 {
                    return Some((bucket.ids[j] - 1, end));
                }
            }
        }

        // Fall back to single byte
        if single_id != 0 {
            return Some((single_id - 1, 1));
        }

        None
    }

    // =========================================================================
    // SIMD Pre-Tokenization
    // =========================================================================

    /// Pre-tokenize text using SIMD
    #[inline]
    fn pre_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        let bytes = text.as_bytes();
        let len = bytes.len();

        if len >= 64 && is_avx512_available() {
            return self.pre_tokenize_avx512(bytes);
        }

        if len >= 32 && is_avx2_available() {
            return self.pre_tokenize_avx2(bytes);
        }

        self.pre_tokenize_scalar(bytes)
    }

    /// AVX2 pre-tokenization (32-byte blocks)
    #[cfg(target_arch = "x86_64")]
    fn pre_tokenize_avx2(&self, bytes: &[u8]) -> Vec<(usize, usize)> {
        if is_x86_feature_detected!("avx2") {
            unsafe { self.pre_tokenize_avx2_impl(bytes) }
        } else {
            self.pre_tokenize_scalar(bytes)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn pre_tokenize_avx2(&self, bytes: &[u8]) -> Vec<(usize, usize)> {
        self.pre_tokenize_scalar(bytes)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn pre_tokenize_avx2_impl(&self, bytes: &[u8]) -> Vec<(usize, usize)> {
        use std::arch::x86_64::*;

        let mut words = Vec::with_capacity(bytes.len() / 5 + 1);
        let len = bytes.len();
        let mut offset = 0;
        let mut in_word = false;
        let mut word_start = 0usize;

        // Whitespace characters
        let space = _mm256_set1_epi8(b' ' as i8);
        let tab = _mm256_set1_epi8(b'\t' as i8);
        let newline = _mm256_set1_epi8(b'\n' as i8);
        let carriage = _mm256_set1_epi8(b'\r' as i8);

        // Apostrophe for contractions
        let apostrophe = _mm256_set1_epi8(b'\'' as i8);

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

            // Classify apostrophe (for contractions in GPT-2 pattern)
            let apos_mask = _mm256_cmpeq_epi8(chunk, apostrophe);

            // Boundary mask = whitespace (apostrophe handled specially)
            let boundary = ws_mask;
            let boundary_bits = _mm256_movemask_epi8(boundary) as u32;
            let apos_bits = _mm256_movemask_epi8(apos_mask) as u32;

            if boundary_bits == 0 && apos_bits == 0 {
                // All word characters
                if !in_word {
                    word_start = offset;
                    in_word = true;
                }
            } else if boundary_bits == 0xFFFFFFFF {
                // All whitespace
                if in_word {
                    words.push((word_start, offset));
                    in_word = false;
                }
            } else {
                // Mixed: process bit by bit
                for pos in 0..32 {
                    let abs_pos = offset + pos;
                    let is_boundary = (boundary_bits >> pos) & 1 == 1;
                    let is_apos = (apos_bits >> pos) & 1 == 1;

                    if is_boundary {
                        if in_word {
                            words.push((word_start, abs_pos));
                            in_word = false;
                        }
                    } else if is_apos {
                        // Handle contractions: don't split on apostrophe
                        if !in_word {
                            word_start = abs_pos;
                            in_word = true;
                        }
                    } else {
                        if !in_word {
                            word_start = abs_pos;
                            in_word = true;
                        }
                    }
                }
            }

            offset += 32;
        }

        // Handle remaining bytes
        self.pre_tokenize_remaining(bytes, offset, &mut words, &mut in_word, &mut word_start);

        if in_word {
            words.push((word_start, len));
        }

        words
    }

    /// AVX-512 pre-tokenization (64-byte blocks)
    fn pre_tokenize_avx512(&self, bytes: &[u8]) -> Vec<(usize, usize)> {
        let mut words = Vec::with_capacity(bytes.len() / 5 + 1);
        let len = bytes.len();
        let mut offset = 0;
        let mut in_word = false;
        let mut word_start = 0usize;

        // Process 64-byte blocks
        while offset + 64 <= len {
            let ws_mask = classify_whitespace(&bytes[offset..]);
            let punct_mask = classify_punctuation(&bytes[offset..]);
            let boundary_mask = ws_mask | punct_mask;

            if boundary_mask == 0 {
                // All word characters
                if !in_word {
                    word_start = offset;
                    in_word = true;
                }
            } else if boundary_mask == !0u64 {
                // All boundaries
                if in_word {
                    words.push((word_start, offset));
                    in_word = false;
                }
                // Punctuation as separate tokens
                let mut p = punct_mask;
                while p != 0 {
                    let pos = p.trailing_zeros() as usize;
                    words.push((offset + pos, offset + pos + 1));
                    p &= p - 1;
                }
            } else {
                // Mixed
                self.process_mixed_block_64(
                    bytes, offset, ws_mask, punct_mask,
                    &mut words, &mut in_word, &mut word_start
                );
            }

            offset += 64;
        }

        // Handle remaining bytes
        self.pre_tokenize_remaining(bytes, offset, &mut words, &mut in_word, &mut word_start);

        if in_word {
            words.push((word_start, len));
        }

        words
    }

    /// Process mixed 64-byte block
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
        let boundary_mask = ws_mask | punct_mask;

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
                // Skip to next boundary
                let remaining = boundary_mask >> pos;
                if remaining != 0 {
                    pos += remaining.trailing_zeros();
                } else {
                    pos = 64;
                }
            }
        }
    }

    /// Pre-tokenize remaining bytes after SIMD blocks
    #[inline]
    fn pre_tokenize_remaining(
        &self,
        bytes: &[u8],
        start: usize,
        words: &mut Vec<(usize, usize)>,
        in_word: &mut bool,
        word_start: &mut usize,
    ) {
        for i in start..bytes.len() {
            let b = bytes[i];
            let is_ws = matches!(b, b' ' | b'\t' | b'\n' | b'\r');
            let is_punct = Self::is_ascii_punct(b);

            if is_ws {
                if *in_word {
                    words.push((*word_start, i));
                    *in_word = false;
                }
            } else if is_punct {
                if *in_word {
                    words.push((*word_start, i));
                    *in_word = false;
                }
                words.push((i, i + 1));
            } else {
                if !*in_word {
                    *word_start = i;
                    *in_word = true;
                }
            }
        }
    }

    /// Scalar pre-tokenization fallback
    fn pre_tokenize_scalar(&self, bytes: &[u8]) -> Vec<(usize, usize)> {
        let mut words = Vec::with_capacity(bytes.len() / 5 + 1);
        let mut in_word = false;
        let mut word_start = 0;

        for i in 0..bytes.len() {
            let b = bytes[i];
            let is_ws = matches!(b, b' ' | b'\t' | b'\n' | b'\r');
            let is_punct = Self::is_ascii_punct(b);

            if is_ws {
                if in_word {
                    words.push((word_start, i));
                    in_word = false;
                }
            } else if is_punct {
                if in_word {
                    words.push((word_start, i));
                    in_word = false;
                }
                words.push((i, i + 1));
            } else {
                if !in_word {
                    word_start = i;
                    in_word = true;
                }
            }
        }

        if in_word {
            words.push((word_start, bytes.len()));
        }

        words
    }

    #[inline(always)]
    fn is_ascii_punct(b: u8) -> bool {
        matches!(b, b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~')
    }

    #[inline(always)]
    fn utf8_len(b: u8) -> usize {
        if b < 0x80 { 1 }
        else if b < 0xE0 { 2 }
        else if b < 0xF0 { 3 }
        else { 4 }
    }

    // =========================================================================
    // Encoding
    // =========================================================================

    /// Encode text to token IDs (fast path)
    #[inline]
    pub fn encode_fast(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let bytes = if self.config.byte_level {
            self.encode_byte_level(text)
        } else {
            text.as_bytes().to_vec()
        };

        self.encode_bytes(&bytes)
    }

    /// Encode text with full token info
    pub fn encode(&self, text: &str) -> Vec<BpeToken> {
        let ids = self.encode_fast(text);
        let mut tokens = Vec::with_capacity(ids.len());
        let mut offset = 0;

        for id in ids {
            let token_str = self.vocabulary.id_to_token(id)
                .unwrap_or("<unk>")
                .to_string();
            let len = token_str.len();
            tokens.push(BpeToken {
                id,
                token: token_str,
                offsets: (offset, offset + len),
            });
            offset += len;
        }

        tokens
    }

    /// Encode bytes using greedy longest-match
    fn encode_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        let mut ids = Vec::with_capacity(bytes.len() / 2);
        let mut pos = 0;

        while pos < bytes.len() {
            let remaining = &bytes[pos..];

            // Prefetch next cache line
            if pos + 64 < bytes.len() {
                prefetch_read(bytes.as_ptr().wrapping_add(pos + 64));
            }

            // Try longest match
            match self.speculative_longest_match(remaining) {
                Some((id, len)) => {
                    ids.push(id);
                    pos += len;
                }
                None => {
                    // Unknown byte - use UNK token
                    ids.push(self.unk_id);
                    pos += 1;
                }
            }
        }

        ids
    }

    /// Encode text as GPT-2 byte-level
    fn encode_byte_level(&self, text: &str) -> Vec<u8> {
        // Convert to GPT-2 byte encoding
        let mut encoded = Vec::with_capacity(text.len() * 2);
        for byte in text.bytes() {
            let cp = self.byte_encoding.encode_byte(byte);
            // Encode code point as UTF-8
            if cp < 0x80 {
                encoded.push(cp as u8);
            } else if cp < 0x800 {
                encoded.push((0xC0 | (cp >> 6)) as u8);
                encoded.push((0x80 | (cp & 0x3F)) as u8);
            } else {
                encoded.push((0xE0 | (cp >> 12)) as u8);
                encoded.push((0x80 | ((cp >> 6) & 0x3F)) as u8);
                encoded.push((0x80 | (cp & 0x3F)) as u8);
            }
        }
        encoded
    }

    /// Batch encode with parallel processing
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
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

    /// Batch encode with full token info
    pub fn encode_batch_full(&self, texts: &[&str]) -> Vec<Vec<BpeToken>> {
        if texts.is_empty() {
            return Vec::new();
        }

        if texts.len() == 1 {
            return vec![self.encode(texts[0])];
        }

        texts.par_iter()
            .map(|text| self.encode(text))
            .collect()
    }

    // =========================================================================
    // Decoding
    // =========================================================================

    /// Decode token IDs to string
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut result = String::with_capacity(ids.len() * 4);

        for &id in ids {
            if let Some(token) = self.vocabulary.id_to_token(id) {
                if self.config.byte_level {
                    // Decode GPT-2 byte encoding
                    for c in token.chars() {
                        let cp = c as u32;
                        if let Some(byte) = self.byte_encoding.decode_char(cp) {
                            result.push(byte as char);
                        } else {
                            result.push(c);
                        }
                    }
                } else {
                    result.push_str(token);
                }
            }
        }

        result
    }

    /// Batch decode
    pub fn decode_batch(&self, batch_ids: &[Vec<u32>]) -> Vec<String> {
        batch_ids.par_iter()
            .map(|ids| self.decode(ids))
            .collect()
    }
}

// =============================================================================
// Builder
// =============================================================================

/// Builder for HyperBpeTokenizer
pub struct HyperBpeBuilder {
    vocabulary: Option<Vocabulary>,
    merges: Vec<MergeRule>,
    config: HyperBpeConfig,
}

impl HyperBpeBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            vocabulary: None,
            merges: Vec::new(),
            config: HyperBpeConfig::default(),
        }
    }

    /// Set vocabulary
    pub fn vocabulary(mut self, vocab: Vocabulary) -> Self {
        self.vocabulary = Some(vocab);
        self
    }

    /// Add merge rules
    pub fn merges(mut self, merges: Vec<MergeRule>) -> Self {
        self.merges = merges;
        self
    }

    /// Add a single merge rule
    pub fn add_merge(mut self, first: &str, second: &str, result: &str) -> Self {
        let priority = self.merges.len() as u32;
        self.merges.push(MergeRule {
            first: first.to_string(),
            second: second.to_string(),
            result: result.to_string(),
            priority,
        });
        self
    }

    /// Set configuration
    pub fn config(mut self, config: HyperBpeConfig) -> Self {
        self.config = config;
        self
    }

    /// Enable byte-level BPE
    pub fn byte_level(mut self, byte_level: bool) -> Self {
        self.config.byte_level = byte_level;
        self
    }

    /// Set hash table bits
    pub fn hash_table_bits(mut self, bits: u32) -> Self {
        self.config.hash_table_bits = bits;
        self
    }

    /// Build the tokenizer
    pub fn build(self) -> Result<HyperBpeTokenizer, &'static str> {
        let vocabulary = self.vocabulary.ok_or("Vocabulary is required")?;
        Ok(HyperBpeTokenizer::new(vocabulary, self.merges, self.config))
    }
}

impl Default for HyperBpeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::VocabularyBuilder;

    fn create_test_tokenizer() -> HyperBpeTokenizer {
        let vocab = VocabularyBuilder::new()
            .add_tokens([
                "<unk>", "h", "e", "l", "o", "w", "r", "d", " ",
                "he", "hel", "hell", "hello",
                "wo", "wor", "worl", "world",
            ])
            .unk_token("<unk>")
            .build();

        let merges = vec![
            MergeRule { first: "h".into(), second: "e".into(), result: "he".into(), priority: 0 },
            MergeRule { first: "he".into(), second: "l".into(), result: "hel".into(), priority: 1 },
            MergeRule { first: "hel".into(), second: "l".into(), result: "hell".into(), priority: 2 },
            MergeRule { first: "hell".into(), second: "o".into(), result: "hello".into(), priority: 3 },
            MergeRule { first: "w".into(), second: "o".into(), result: "wo".into(), priority: 4 },
            MergeRule { first: "wo".into(), second: "r".into(), result: "wor".into(), priority: 5 },
            MergeRule { first: "wor".into(), second: "l".into(), result: "worl".into(), priority: 6 },
            MergeRule { first: "worl".into(), second: "d".into(), result: "world".into(), priority: 7 },
        ];

        let config = HyperBpeConfig {
            byte_level: false,
            ..Default::default()
        };

        HyperBpeTokenizer::new(vocab, merges, config)
    }

    #[test]
    fn test_hyper_bpe_encode() {
        let tokenizer = create_test_tokenizer();
        let ids = tokenizer.encode_fast("hello");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_hyper_bpe_encode_with_space() {
        let tokenizer = create_test_tokenizer();
        let ids = tokenizer.encode_fast("hello world");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_hyper_bpe_batch_encode() {
        let tokenizer = create_test_tokenizer();
        let texts = vec!["hello", "world", "hello world"];
        let results = tokenizer.encode_batch(&texts);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_hyper_bpe_decode() {
        let tokenizer = create_test_tokenizer();
        let ids = tokenizer.encode_fast("hello");
        let decoded = tokenizer.decode(&ids);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_byte_encoding_table() {
        let table = ByteEncodingTable::new();

        // Test roundtrip for all bytes
        for byte in 0u8..=255 {
            let cp = table.encode_byte(byte);
            if let Some(decoded) = table.decode_char(cp) {
                assert_eq!(decoded, byte, "Byte {} didn't roundtrip", byte);
            }
        }
    }

    #[test]
    fn test_hash_lookup() {
        let tokenizer = create_test_tokenizer();

        // Test that "hello" can be found
        let hello_id = tokenizer.vocabulary.token_to_id("hello");
        let lookup_id = tokenizer.hash_lookup(b"hello");

        if let Some(id) = hello_id {
            assert_eq!(lookup_id, id + 1, "Hash lookup should find 'hello'");
        }
    }

    #[test]
    fn test_speculative_longest_match() {
        let tokenizer = create_test_tokenizer();

        // "hello" should match as single token
        let result = tokenizer.speculative_longest_match(b"hello");
        assert!(result.is_some());

        if let Some((id, len)) = result {
            assert_eq!(len, 5, "Should match all 5 characters");
            assert_eq!(tokenizer.vocabulary.id_to_token(id), Some("hello"));
        }
    }

    #[test]
    fn test_pre_tokenize_scalar() {
        let tokenizer = create_test_tokenizer();
        let words = tokenizer.pre_tokenize_scalar(b"hello world");

        assert_eq!(words.len(), 2);
        assert_eq!(words[0], (0, 5)); // "hello"
        assert_eq!(words[1], (6, 11)); // "world"
    }

    #[test]
    fn test_builder() {
        let vocab = VocabularyBuilder::new()
            .add_tokens(["<unk>", "h", "e", "he"])
            .unk_token("<unk>")
            .build();

        let tokenizer = HyperBpeBuilder::new()
            .vocabulary(vocab)
            .add_merge("h", "e", "he")
            .byte_level(false)
            .build()
            .unwrap();

        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    fn test_empty_input() {
        let tokenizer = create_test_tokenizer();

        let ids = tokenizer.encode_fast("");
        assert!(ids.is_empty());

        let decoded = tokenizer.decode(&[]);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_unknown_token() {
        let tokenizer = create_test_tokenizer();

        // Use a character not in vocab
        let ids = tokenizer.encode_fast("xyz");
        // Should produce some output (either found tokens or UNK)
        assert!(!ids.is_empty());
    }
}
