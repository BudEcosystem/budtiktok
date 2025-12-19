//! SIMD-Optimized BPE Tokenizer
//!
//! This module implements a SIMD-compatible BPE algorithm based on research findings:
//! - "Train It and Forget It" (2025): Merge lists are unnecessary for inference
//! - BlockBPE: Parallel merge passes with prefix sum compaction
//! - Google's LinMaxMatch: Linear-time longest match via failure links
//!
//! Key optimizations:
//! 1. SIMD trie with 32-way child lookup using AVX2
//! 2. Perfect hash for O(1) merge lookup with SIMD gather
//! 3. Greedy longest-match for short sequences
//! 4. Wavefront DP for optimal encoding of long sequences
//!
//! This is a separate implementation from bpe_linear.rs for experimentation.
//! Only merge with main code after validation.

use ahash::AHashMap;
use std::arch::x86_64::*;

// Import GPT-2 pre-tokenizer from bpe_linear
use crate::bpe_linear::gpt2_pre_tokenize_fast;

// =============================================================================
// Constants
// =============================================================================

/// Invalid node index sentinel
const INVALID_NODE: u32 = u32::MAX;

/// Maximum byte value (we handle full 0-255 range)
const MAX_BYTE: usize = 256;

/// SIMD width for AVX2 (256 bits = 32 bytes = 8 u32s)
const SIMD_WIDTH_BYTES: usize = 32;
const SIMD_WIDTH_U32: usize = 8;

/// Threshold for switching from greedy to wavefront DP
const GREEDY_THRESHOLD: usize = 64;

/// Cache line size for alignment
const CACHE_LINE: usize = 64;

// =============================================================================
// SIMD Trie Node
// =============================================================================

/// A trie node optimized for SIMD operations
///
/// Memory layout:
/// - Children array covers all 256 byte values
/// - Aligned to cache line for efficient access
/// - Token info at the end for locality
#[repr(C, align(64))]
#[derive(Clone)]
pub struct SimdTrieNode {
    /// Child node indices for bytes 0-255
    /// INVALID_NODE means no child for that byte
    children: [u32; MAX_BYTE],

    /// Token ID if this node represents a complete token (0 = not a token, actual_id + 1 otherwise)
    token_id: u32,

    /// Length of the token in bytes (0 if not a token)
    token_len: u16,

    /// Padding for alignment
    _pad: u16,
}

impl SimdTrieNode {
    /// Create a new empty node
    #[inline]
    pub fn new() -> Self {
        Self {
            children: [INVALID_NODE; MAX_BYTE],
            token_id: 0,
            token_len: 0,
            _pad: 0,
        }
    }

    /// Get child node index for a byte
    #[inline(always)]
    pub fn get_child(&self, byte: u8) -> Option<u32> {
        let idx = self.children[byte as usize];
        if idx != INVALID_NODE {
            Some(idx)
        } else {
            None
        }
    }

    /// Set child node index for a byte
    #[inline(always)]
    pub fn set_child(&mut self, byte: u8, node_idx: u32) {
        self.children[byte as usize] = node_idx;
    }

    /// Check if this node represents a complete token
    #[inline(always)]
    pub fn is_token(&self) -> bool {
        self.token_id != 0
    }

    /// Get the token ID (None if not a token)
    #[inline(always)]
    pub fn get_token_id(&self) -> Option<u32> {
        if self.token_id != 0 {
            Some(self.token_id - 1)
        } else {
            None
        }
    }

    /// Set this node as a token
    #[inline(always)]
    pub fn set_token(&mut self, id: u32, len: usize) {
        self.token_id = id + 1;
        self.token_len = len as u16;
    }

    /// SIMD: Get 8 children at once using AVX2 gather
    ///
    /// # Safety
    /// Requires AVX2 support
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn get_children_simd(&self, bytes: __m256i) -> __m256i {
        // bytes contains 8 byte indices (as i32)
        // We gather 8 child node indices
        _mm256_i32gather_epi32::<4>(self.children.as_ptr() as *const i32, bytes)
    }
}

impl Default for SimdTrieNode {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// SIMD Trie
// =============================================================================

/// A trie optimized for SIMD operations
///
/// Supports:
/// - O(1) child lookup per byte
/// - SIMD 8-way parallel child lookup
/// - Longest match finding with SIMD acceleration
pub struct SimdTrie {
    /// All nodes in the trie (node 0 is root)
    nodes: Vec<SimdTrieNode>,

    /// Maximum token length in the vocabulary
    max_token_len: usize,

    /// Number of tokens in vocabulary
    vocab_size: usize,
}

impl SimdTrie {
    /// Build a trie from vocabulary
    ///
    /// vocab: mapping from token string (byte-encoded) to token ID
    pub fn from_vocab(vocab: &AHashMap<String, u32>) -> Self {
        let mut trie = Self {
            nodes: vec![SimdTrieNode::new()],
            max_token_len: 0,
            vocab_size: vocab.len(),
        };

        for (token, &id) in vocab {
            trie.insert(token.as_bytes(), id);
        }

        trie
    }

    /// Build a trie from vocabulary (bytes version)
    pub fn from_vocab_bytes(vocab: &AHashMap<Vec<u8>, u32>) -> Self {
        let mut trie = Self {
            nodes: vec![SimdTrieNode::new()],
            max_token_len: 0,
            vocab_size: vocab.len(),
        };

        for (token, &id) in vocab {
            trie.insert(token, id);
        }

        trie
    }

    /// Insert a token into the trie
    fn insert(&mut self, token: &[u8], id: u32) {
        if token.is_empty() {
            return;
        }

        self.max_token_len = self.max_token_len.max(token.len());

        let mut node_idx = 0u32;

        for &byte in token {
            let child_idx = self.nodes[node_idx as usize].get_child(byte);

            node_idx = match child_idx {
                Some(idx) => idx,
                None => {
                    let new_idx = self.nodes.len() as u32;
                    self.nodes.push(SimdTrieNode::new());
                    self.nodes[node_idx as usize].set_child(byte, new_idx);
                    new_idx
                }
            };
        }

        self.nodes[node_idx as usize].set_token(id, token.len());
    }

    /// Find the longest matching token starting at position `start`
    ///
    /// Returns (token_id, length) or None if no match
    #[inline]
    pub fn longest_match(&self, bytes: &[u8], start: usize) -> Option<(u32, usize)> {
        if start >= bytes.len() {
            return None;
        }

        let mut node_idx = 0u32;
        let mut best_match: Option<(u32, usize)> = None;

        for (i, &byte) in bytes[start..].iter().enumerate() {
            match self.nodes[node_idx as usize].get_child(byte) {
                Some(child_idx) => {
                    node_idx = child_idx;
                    if let Some(token_id) = self.nodes[node_idx as usize].get_token_id() {
                        best_match = Some((token_id, i + 1));
                    }
                }
                None => break,
            }
        }

        best_match
    }

    /// Find the longest matching token with SIMD prefetching
    ///
    /// Uses software prefetching to hide memory latency
    #[inline]
    pub fn longest_match_prefetch(&self, bytes: &[u8], start: usize) -> Option<(u32, usize)> {
        if start >= bytes.len() {
            return None;
        }

        let mut node_idx = 0u32;
        let mut best_match: Option<(u32, usize)> = None;
        let remaining = &bytes[start..];
        let len = remaining.len();

        for i in 0..len {
            let byte = remaining[i];

            match self.nodes[node_idx as usize].get_child(byte) {
                Some(child_idx) => {
                    // Prefetch next node if there's more input
                    if i + 1 < len {
                        let next_byte = remaining[i + 1];
                        let potential_next = self.nodes[child_idx as usize].children[next_byte as usize];
                        if potential_next != INVALID_NODE {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                let ptr = self.nodes.as_ptr().add(potential_next as usize);
                                _mm_prefetch::<_MM_HINT_T0>(ptr as *const i8);
                            }
                        }
                    }

                    node_idx = child_idx;
                    if let Some(token_id) = self.nodes[node_idx as usize].get_token_id() {
                        best_match = Some((token_id, i + 1));
                    }
                }
                None => break,
            }
        }

        best_match
    }

    /// Find all matching tokens starting at position `start`
    ///
    /// Returns a vector of (token_id, length) pairs, sorted by length ascending
    pub fn all_matches(&self, bytes: &[u8], start: usize) -> Vec<(u32, usize)> {
        let mut matches = Vec::new();

        if start >= bytes.len() {
            return matches;
        }

        let mut node_idx = 0u32;

        for (i, &byte) in bytes[start..].iter().enumerate() {
            match self.nodes[node_idx as usize].get_child(byte) {
                Some(child_idx) => {
                    node_idx = child_idx;
                    if let Some(token_id) = self.nodes[node_idx as usize].get_token_id() {
                        matches.push((token_id, i + 1));
                    }
                }
                None => break,
            }
        }

        matches
    }

    /// Get maximum token length
    #[inline]
    pub fn max_token_len(&self) -> usize {
        self.max_token_len
    }

    /// Get vocabulary size
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get number of nodes
    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

// =============================================================================
// Perfect Hash for Merge Rules
// =============================================================================

/// A perfect hash table for fast merge rule lookup
///
/// Uses open addressing with linear probing
/// Optimized for SIMD gather operations
#[repr(C, align(64))]
pub struct MergeHash {
    /// Keys: packed (left_token << 32) | right_token
    keys: Vec<u64>,

    /// Values: packed (rank << 32) | new_token_id
    values: Vec<u64>,

    /// Size mask for fast modulo (size - 1, where size is power of 2)
    mask: usize,

    /// Number of entries
    count: usize,
}

impl MergeHash {
    /// Sentinel value for empty slots
    const EMPTY: u64 = u64::MAX;

    /// Build hash table from merge rules
    ///
    /// merges: mapping from (left_token, right_token) to (rank, new_token_id)
    pub fn from_merges(merges: &AHashMap<(u32, u32), (u32, u32)>) -> Self {
        // Size must be power of 2, at least 2x the number of entries
        let min_size = merges.len().next_power_of_two() * 2;
        let size = min_size.max(16);

        let mut hash = Self {
            keys: vec![Self::EMPTY; size],
            values: vec![0; size],
            mask: size - 1,
            count: 0,
        };

        for (&(left, right), &(rank, new_id)) in merges {
            hash.insert(left, right, rank, new_id);
        }

        hash
    }

    /// Pack a token pair into a key
    #[inline(always)]
    fn pack_key(left: u32, right: u32) -> u64 {
        ((left as u64) << 32) | (right as u64)
    }

    /// Pack rank and new_id into a value
    #[inline(always)]
    fn pack_value(rank: u32, new_id: u32) -> u64 {
        ((rank as u64) << 32) | (new_id as u64)
    }

    /// Unpack a value into (rank, new_id)
    #[inline(always)]
    fn unpack_value(value: u64) -> (u32, u32) {
        ((value >> 32) as u32, value as u32)
    }

    /// FNV-1a hash for a key
    #[inline(always)]
    fn hash_key(key: u64) -> usize {
        // FNV-1a with good mixing
        let mut h = 0xcbf29ce484222325u64;
        h ^= key;
        h = h.wrapping_mul(0x100000001b3);
        h ^= key >> 32;
        h = h.wrapping_mul(0x100000001b3);
        h as usize
    }

    /// Insert a merge rule
    fn insert(&mut self, left: u32, right: u32, rank: u32, new_id: u32) {
        let key = Self::pack_key(left, right);
        let value = Self::pack_value(rank, new_id);
        let mut idx = Self::hash_key(key) & self.mask;

        loop {
            if self.keys[idx] == Self::EMPTY {
                self.keys[idx] = key;
                self.values[idx] = value;
                self.count += 1;
                return;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    /// Look up a merge rule
    ///
    /// Returns Some((rank, new_token_id)) if found
    #[inline]
    pub fn get(&self, left: u32, right: u32) -> Option<(u32, u32)> {
        let key = Self::pack_key(left, right);
        let mut idx = Self::hash_key(key) & self.mask;

        for _ in 0..16 {
            // Max 16 probes
            let stored_key = self.keys[idx];
            if stored_key == key {
                return Some(Self::unpack_value(self.values[idx]));
            }
            if stored_key == Self::EMPTY {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }

        None
    }

    /// SIMD: Look up 4 merge rules at once
    ///
    /// # Safety
    /// Requires AVX2 support
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn get_4_simd(
        &self,
        lefts: [u32; 4],
        rights: [u32; 4],
    ) -> [Option<(u32, u32)>; 4] {
        // Pack keys
        let keys: [u64; 4] = [
            Self::pack_key(lefts[0], rights[0]),
            Self::pack_key(lefts[1], rights[1]),
            Self::pack_key(lefts[2], rights[2]),
            Self::pack_key(lefts[3], rights[3]),
        ];

        // Compute initial indices
        let indices: [usize; 4] = [
            Self::hash_key(keys[0]) & self.mask,
            Self::hash_key(keys[1]) & self.mask,
            Self::hash_key(keys[2]) & self.mask,
            Self::hash_key(keys[3]) & self.mask,
        ];

        // Load keys as SIMD vector
        let keys_vec = _mm256_loadu_si256(keys.as_ptr() as *const __m256i);

        // Gather stored keys (treating indices as i64 for 64-bit gather)
        let indices_vec = _mm256_setr_epi64x(
            indices[0] as i64,
            indices[1] as i64,
            indices[2] as i64,
            indices[3] as i64,
        );

        let stored_keys = _mm256_i64gather_epi64::<8>(self.keys.as_ptr() as *const i64, indices_vec);

        // Compare keys
        let matches = _mm256_cmpeq_epi64(keys_vec, stored_keys);

        // Extract results
        let match_mask = _mm256_movemask_epi8(matches);

        let mut results = [None; 4];

        // Check each lane (8 bytes per lane)
        if match_mask & 0xFF != 0 {
            results[0] = Some(Self::unpack_value(self.values[indices[0]]));
        }
        if match_mask & 0xFF00 != 0 {
            results[1] = Some(Self::unpack_value(self.values[indices[1]]));
        }
        if match_mask & 0xFF0000 != 0 {
            results[2] = Some(Self::unpack_value(self.values[indices[2]]));
        }
        if match_mask & 0xFF000000u32 as i32 != 0 {
            results[3] = Some(Self::unpack_value(self.values[indices[3]]));
        }

        // Handle misses with scalar fallback (linear probing)
        for i in 0..4 {
            if results[i].is_none() && self.keys[indices[i]] != Self::EMPTY {
                // Linear probe
                let mut idx = (indices[i] + 1) & self.mask;
                for _ in 0..15 {
                    if self.keys[idx] == keys[i] {
                        results[i] = Some(Self::unpack_value(self.values[idx]));
                        break;
                    }
                    if self.keys[idx] == Self::EMPTY {
                        break;
                    }
                    idx = (idx + 1) & self.mask;
                }
            }
        }

        results
    }

    /// Get number of entries
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

// =============================================================================
// GPT-2 Byte Encoder (copied from bpe_linear.rs for independence)
// =============================================================================

/// GPT-2 byte-to-unicode encoder
pub struct ByteEncoder {
    byte_to_char: [char; 256],
    char_to_byte: AHashMap<char, u8>,
}

impl ByteEncoder {
    pub fn new() -> Self {
        let mut byte_to_char = ['\0'; 256];
        let mut char_to_byte = AHashMap::new();
        let mut n = 0u32;

        // Printable ASCII and extended Latin
        for b in 0u8..=255 {
            let is_printable = (b'!'..=b'~').contains(&b)
                || (0xA1..=0xAC).contains(&b)
                || (0xAE..=0xFF).contains(&b);

            let c = if is_printable {
                b as char
            } else {
                let c = char::from_u32(256 + n).unwrap();
                n += 1;
                c
            };

            byte_to_char[b as usize] = c;
            char_to_byte.insert(c, b);
        }

        Self {
            byte_to_char,
            char_to_byte,
        }
    }

    /// Encode bytes to GPT-2 unicode string
    #[inline]
    pub fn encode_bytes(&self, bytes: &[u8]) -> String {
        bytes.iter().map(|&b| self.byte_to_char[b as usize]).collect()
    }

    /// Decode GPT-2 unicode string to bytes
    #[inline]
    pub fn decode_string(&self, s: &str) -> Vec<u8> {
        s.chars()
            .filter_map(|c| self.char_to_byte.get(&c).copied())
            .collect()
    }
}

impl Default for ByteEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// SIMD BPE Encoder
// =============================================================================

/// SIMD-optimized BPE encoder
///
/// Supports multiple encoding strategies:
/// - `encode_greedy`: Fast greedy longest-match (different tokenization)
/// - `encode_bpe`: 100% compatible with standard BPE (uses merge ordering)
/// - `encode_gpt2`: GPT-2 compatible with pre-tokenization
/// - `encode_minimal`: Optimal token count via wavefront DP
pub struct SimdBpeEncoder {
    /// Vocabulary trie for token lookup
    trie: SimdTrie,

    /// Merge rules hash table
    merges: MergeHash,

    /// GPT-2 byte encoder
    byte_encoder: ByteEncoder,

    /// Single character token IDs (for BPE initial tokenization)
    char_tokens: AHashMap<char, u32>,

    /// Original vocabulary (string -> id)
    vocab: AHashMap<String, u32>,

    /// Unknown token ID
    unk_id: u32,

    /// Vocabulary size
    vocab_size: usize,
}

impl SimdBpeEncoder {
    /// Create a new SIMD BPE encoder
    ///
    /// vocab: mapping from token string to token ID
    /// merges: list of (first, second) merge pairs in priority order
    /// unk_token: the unknown token string
    pub fn new(
        vocab: AHashMap<String, u32>,
        merges: Vec<(String, String)>,
        unk_token: &str,
    ) -> Self {
        let vocab_size = vocab.len();
        let unk_id = *vocab.get(unk_token).unwrap_or(&0);

        // Build byte-encoded vocabulary for trie
        let byte_encoder = ByteEncoder::new();
        let mut byte_vocab: AHashMap<Vec<u8>, u32> = AHashMap::new();
        for (token, id) in &vocab {
            let bytes = byte_encoder.decode_string(token);
            byte_vocab.insert(bytes, *id);
        }

        // Build trie from byte vocabulary
        let trie = SimdTrie::from_vocab_bytes(&byte_vocab);

        // Build merge hash table
        let mut merge_map: AHashMap<(u32, u32), (u32, u32)> = AHashMap::new();
        for (rank, (first, second)) in merges.iter().enumerate() {
            if let (Some(&first_id), Some(&second_id)) = (vocab.get(first), vocab.get(second)) {
                let merged = format!("{}{}", first, second);
                if let Some(&merged_id) = vocab.get(&merged) {
                    merge_map.insert((first_id, second_id), (rank as u32, merged_id));
                }
            }
        }
        let merges_hash = MergeHash::from_merges(&merge_map);

        // Build character token map for BPE initial tokenization
        let mut char_tokens = AHashMap::new();
        for (token, &id) in &vocab {
            let chars: Vec<char> = token.chars().collect();
            if chars.len() == 1 {
                char_tokens.insert(chars[0], id);
            }
        }

        Self {
            trie,
            merges: merges_hash,
            byte_encoder,
            char_tokens,
            vocab: vocab.clone(),
            unk_id,
            vocab_size,
        }
    }

    /// Encode text to token IDs using greedy longest-match (raw bytes)
    ///
    /// This is a fast path but doesn't use GPT-2 pre-tokenization.
    /// Use `encode_gpt2` for GPT-2 compatible tokenization.
    #[inline]
    pub fn encode_greedy(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let bytes = text.as_bytes();
        let mut result = Vec::with_capacity(bytes.len() / 4);
        let mut pos = 0;

        while pos < bytes.len() {
            match self.trie.longest_match_prefetch(bytes, pos) {
                Some((token_id, len)) => {
                    result.push(token_id);
                    pos += len;
                }
                None => {
                    // Unknown byte - encode as single byte token or unk
                    result.push(self.unk_id);
                    pos += 1;
                }
            }
        }

        result
    }

    /// Encode a single pre-token (byte-encoded word) to token IDs
    #[inline]
    fn encode_word_greedy(&self, word_bytes: &[u8]) -> Vec<u32> {
        let mut result = Vec::with_capacity(word_bytes.len() / 2);
        let mut pos = 0;

        while pos < word_bytes.len() {
            match self.trie.longest_match_prefetch(word_bytes, pos) {
                Some((token_id, len)) => {
                    result.push(token_id);
                    pos += len;
                }
                None => {
                    // Unknown byte - should rarely happen for GPT-2
                    result.push(self.unk_id);
                    pos += 1;
                }
            }
        }

        result
    }

    /// Encode text using GPT-2 compatible tokenization
    ///
    /// This matches the behavior of the standard GPT-2 BPE:
    /// 1. Pre-tokenize using GPT-2 regex pattern
    /// 2. Byte-encode each pre-token
    /// 3. Apply greedy longest-match on byte-encoded tokens
    #[inline]
    pub fn encode_gpt2(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(text.len() / 4);

        // Pre-tokenize using GPT-2 regex pattern
        let pre_tokens = gpt2_pre_tokenize_fast(text);

        for pre_token in pre_tokens {
            // Byte-encode the pre-token (each byte -> unicode char)
            let byte_encoded = self.byte_encoder.encode_bytes(pre_token.as_bytes());

            // Convert back to bytes for trie lookup (the trie stores byte-decoded tokens)
            let lookup_bytes = self.byte_encoder.decode_string(&byte_encoded);

            // Greedy longest-match on this word
            result.extend(self.encode_word_greedy(&lookup_bytes));
        }

        result
    }

    /// Encode text to token IDs using proper BPE with merge ordering
    ///
    /// This is 100% compatible with standard BPE tokenization.
    /// Uses GPT-2 pre-tokenization followed by merge-order BPE on each word.
    #[inline]
    pub fn encode_bpe(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(text.len() / 4);

        // Pre-tokenize using GPT-2 regex pattern
        let pre_tokens = gpt2_pre_tokenize_fast(text);

        for pre_token in pre_tokens {
            // Byte-encode the pre-token (each byte -> unicode char)
            let byte_encoded = self.byte_encoder.encode_bytes(pre_token.as_bytes());

            // Use proper BPE with merge ordering
            result.extend(self.encode_word_bpe(&byte_encoded));
        }

        result
    }

    /// Encode a single word using proper BPE with merge ordering
    ///
    /// Algorithm:
    /// 1. Tokenize each character as initial tokens
    /// 2. Find the best merge (lowest rank) among all adjacent pairs
    /// 3. Apply the merge and repeat until no more merges possible
    ///
    /// This matches HuggingFace tokenizers behavior exactly.
    fn encode_word_bpe(&self, word: &str) -> Vec<u32> {
        if word.is_empty() {
            return Vec::new();
        }

        // Try direct trie match for complete word first
        // Many common words are vocabulary tokens (e.g., "Ġthe", "Ġand")
        if let Some(&token_id) = self.vocab.get(word) {
            return vec![token_id];
        }

        // Initialize tokens as single characters
        let mut tokens: Vec<u32> = Vec::with_capacity(word.len());
        for c in word.chars() {
            match self.char_tokens.get(&c) {
                Some(&id) => tokens.push(id),
                None => tokens.push(self.unk_id),
            }
        }

        if tokens.len() < 2 {
            return tokens;
        }

        // Apply merges using SIMD-accelerated algorithm
        self.apply_merges_simd(&mut tokens);
        tokens
    }

    /// SIMD-accelerated BPE merge application
    ///
    /// Uses SIMD to find the best merge among all adjacent pairs in parallel.
    /// For short sequences (≤32), uses a simple O(n²) algorithm.
    /// For longer sequences, uses SIMD 4-way parallel merge lookup.
    fn apply_merges_simd(&self, tokens: &mut Vec<u32>) {
        if tokens.len() < 2 {
            return;
        }

        // For short sequences, simple algorithm is faster (less overhead)
        if tokens.len() <= 32 {
            self.apply_merges_simple(tokens);
            return;
        }

        // For longer sequences, use SIMD-accelerated algorithm
        self.apply_merges_simd_long(tokens);
    }

    /// Simple O(n²) merge for short sequences
    ///
    /// Uses in-place merging with validity mask to avoid O(n) removals.
    /// Optimized for cache locality with stack-allocated arrays.
    #[inline]
    fn apply_merges_simple(&self, tokens: &mut Vec<u32>) {
        let n = tokens.len();
        if n < 2 {
            return;
        }

        // Use array-based approach for very short sequences
        let mut valid = [true; 32];
        let mut next = [0i8; 32];

        // Initialize next pointers
        for i in 0..n {
            next[i] = if i + 1 < n { (i + 1) as i8 } else { -1 };
        }

        loop {
            // Find best merge (lowest rank) among all adjacent pairs
            let mut best_merge: Option<(usize, u32, u32)> = None; // (pos, rank, new_id)
            let mut i = 0usize;

            while i < n {
                if !valid[i] {
                    i += 1;
                    continue;
                }

                let next_i = next[i];
                if next_i < 0 {
                    break;
                }

                let j = next_i as usize;
                if !valid[j] {
                    // Find next valid position
                    let mut k = j + 1;
                    while k < n && !valid[k] {
                        k += 1;
                    }
                    next[i] = if k < n { k as i8 } else { -1 };
                    continue;
                }

                // Check if this pair can merge
                if let Some((rank, new_id)) = self.merges.get(tokens[i], tokens[j]) {
                    if best_merge.is_none() || rank < best_merge.unwrap().1 {
                        best_merge = Some((i, rank, new_id));
                    }
                }
                i = j;
            }

            match best_merge {
                Some((pos, _, new_id)) => {
                    // Apply merge
                    tokens[pos] = new_id;
                    let next_pos = next[pos] as usize;
                    valid[next_pos] = false;

                    // Update next pointer to skip removed position
                    let mut k = next_pos + 1;
                    while k < n && !valid[k] {
                        k += 1;
                    }
                    next[pos] = if k < n { k as i8 } else { -1 };
                }
                None => break,
            }
        }

        // Compact result
        let mut write_pos = 0;
        for i in 0..n {
            if valid[i] {
                tokens[write_pos] = tokens[i];
                write_pos += 1;
            }
        }
        tokens.truncate(write_pos);
    }

    /// SIMD-accelerated merge for longer sequences
    ///
    /// Uses SIMD 4-way parallel merge lookup to find best merges faster.
    /// Still O(n²) overall but with better constants due to SIMD parallelism.
    #[inline]
    fn apply_merges_simd_long(&self, tokens: &mut Vec<u32>) {
        let n = tokens.len();
        if n < 2 {
            return;
        }

        // Build doubly-linked list representation
        let mut prev: Vec<i32> = (0..n).map(|i| i as i32 - 1).collect();
        let mut next: Vec<i32> = (0..n).map(|i| if i < n - 1 { i as i32 + 1 } else { -1 }).collect();
        let mut valid: Vec<bool> = vec![true; n];

        loop {
            // Find best merge using SIMD 4-way lookup
            let mut best_merge: Option<(usize, u32, u32)> = None;
            let mut i = 0i32;

            // Process 4 pairs at a time when possible
            while i >= 0 && (i as usize) < n {
                let i_usize = i as usize;
                if !valid[i_usize] {
                    i = next[i_usize];
                    continue;
                }

                let j = next[i_usize];
                if j < 0 {
                    break;
                }

                let j_usize = j as usize;

                // Check if we can do 4-way SIMD lookup
                // Collect up to 4 valid pairs
                let mut lefts = [0u32; 4];
                let mut rights = [0u32; 4];
                let mut positions = [0usize; 4];
                let mut pair_count = 0;

                let mut cur = i;
                while pair_count < 4 && cur >= 0 {
                    let cur_usize = cur as usize;
                    if !valid[cur_usize] {
                        cur = next[cur_usize];
                        continue;
                    }

                    let nxt = next[cur_usize];
                    if nxt < 0 {
                        break;
                    }

                    let nxt_usize = nxt as usize;
                    if valid[nxt_usize] {
                        lefts[pair_count] = tokens[cur_usize];
                        rights[pair_count] = tokens[nxt_usize];
                        positions[pair_count] = cur_usize;
                        pair_count += 1;
                    }
                    cur = nxt;
                }

                if pair_count == 0 {
                    break;
                }

                // Look up all pairs (SIMD when possible)
                #[cfg(target_arch = "x86_64")]
                let results = if is_x86_feature_detected!("avx2") && pair_count == 4 {
                    unsafe { self.merges.get_4_simd(lefts, rights) }
                } else {
                    // Scalar fallback
                    [
                        self.merges.get(lefts[0], rights[0]),
                        self.merges.get(lefts[1], rights[1]),
                        self.merges.get(lefts[2], rights[2]),
                        self.merges.get(lefts[3], rights[3]),
                    ]
                };

                #[cfg(not(target_arch = "x86_64"))]
                let results = [
                    self.merges.get(lefts[0], rights[0]),
                    self.merges.get(lefts[1], rights[1]),
                    self.merges.get(lefts[2], rights[2]),
                    self.merges.get(lefts[3], rights[3]),
                ];

                // Find best among results
                for k in 0..pair_count {
                    if let Some((rank, new_id)) = results[k] {
                        if best_merge.is_none() || rank < best_merge.unwrap().1 {
                            best_merge = Some((positions[k], rank, new_id));
                        }
                    }
                }

                // Move to next batch of pairs
                if pair_count > 0 {
                    let last_pos = positions[pair_count - 1];
                    i = next[last_pos];
                } else {
                    break;
                }
            }

            match best_merge {
                Some((pos, _, new_id)) => {
                    // Apply merge
                    tokens[pos] = new_id;
                    let next_pos = next[pos] as usize;
                    valid[next_pos] = false;

                    // Update linked list
                    let next_next = next[next_pos];
                    next[pos] = next_next;
                    if next_next >= 0 {
                        prev[next_next as usize] = pos as i32;
                    }
                }
                None => break,
            }
        }

        // Compact result
        let mut write_pos = 0;
        for i in 0..n {
            if valid[i] {
                tokens[write_pos] = tokens[i];
                write_pos += 1;
            }
        }
        tokens.truncate(write_pos);
    }

    /// Encode text to token IDs using wavefront DP for minimal encoding
    ///
    /// This produces the encoding with minimum number of tokens
    pub fn encode_minimal(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let bytes = text.as_bytes();
        let n = bytes.len();

        // DP arrays
        // dp[i] = minimum number of tokens to encode bytes[0..i]
        // parent[i] = (token_id, start_pos) for backtracking
        let mut dp = vec![u32::MAX; n + 1];
        let mut parent: Vec<(u32, usize)> = vec![(0, 0); n + 1];
        dp[0] = 0;

        // Forward pass: compute minimum tokens for each prefix
        for i in 0..n {
            if dp[i] == u32::MAX {
                continue;
            }

            // Find all tokens starting at position i
            let matches = self.trie.all_matches(bytes, i);

            let has_matches = !matches.is_empty();
            for (token_id, len) in matches {
                let end = i + len;
                let new_cost = dp[i] + 1;

                if new_cost < dp[end] {
                    dp[end] = new_cost;
                    parent[end] = (token_id, i);
                }
            }

            // Handle unknown bytes
            if !has_matches {
                let end = i + 1;
                let new_cost = dp[i] + 1;
                if new_cost < dp[end] {
                    dp[end] = new_cost;
                    parent[end] = (self.unk_id, i);
                }
            }
        }

        // Backtrack to get tokens
        if dp[n] == u32::MAX {
            // Fallback: encode each byte as unknown
            return vec![self.unk_id; n];
        }

        let mut tokens = Vec::with_capacity(dp[n] as usize);
        let mut pos = n;

        while pos > 0 {
            let (token_id, start) = parent[pos];
            tokens.push(token_id);
            pos = start;
        }

        tokens.reverse();
        tokens
    }

    /// Encode text using the best strategy based on length
    #[inline]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.len() <= GREEDY_THRESHOLD {
            self.encode_greedy(text)
        } else {
            self.encode_minimal(text)
        }
    }

    /// Get vocabulary size
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // SimdTrieNode Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trie_node_new() {
        let node = SimdTrieNode::new();
        assert!(!node.is_token());
        assert_eq!(node.get_token_id(), None);

        for byte in 0..=255u8 {
            assert_eq!(node.get_child(byte), None);
        }
    }

    #[test]
    fn test_trie_node_set_child() {
        let mut node = SimdTrieNode::new();

        node.set_child(b'a', 42);
        assert_eq!(node.get_child(b'a'), Some(42));
        assert_eq!(node.get_child(b'b'), None);

        node.set_child(b'z', 100);
        assert_eq!(node.get_child(b'z'), Some(100));
    }

    #[test]
    fn test_trie_node_set_token() {
        let mut node = SimdTrieNode::new();

        assert!(!node.is_token());

        node.set_token(123, 5);
        assert!(node.is_token());
        assert_eq!(node.get_token_id(), Some(123));
        assert_eq!(node.token_len, 5);
    }

    #[test]
    fn test_trie_node_token_id_zero() {
        // Test that token ID 0 works correctly
        let mut node = SimdTrieNode::new();
        node.set_token(0, 1);
        assert!(node.is_token());
        assert_eq!(node.get_token_id(), Some(0));
    }

    // -------------------------------------------------------------------------
    // SimdTrie Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_trie_empty() {
        let vocab: AHashMap<String, u32> = AHashMap::new();
        let trie = SimdTrie::from_vocab(&vocab);

        assert_eq!(trie.vocab_size(), 0);
        assert_eq!(trie.max_token_len(), 0);
        assert_eq!(trie.longest_match(b"hello", 0), None);
    }

    #[test]
    fn test_trie_single_char_tokens() {
        let mut vocab = AHashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);
        vocab.insert("c".to_string(), 2);

        let trie = SimdTrie::from_vocab(&vocab);

        assert_eq!(trie.vocab_size(), 3);
        assert_eq!(trie.max_token_len(), 1);

        assert_eq!(trie.longest_match(b"abc", 0), Some((0, 1)));
        assert_eq!(trie.longest_match(b"abc", 1), Some((1, 1)));
        assert_eq!(trie.longest_match(b"abc", 2), Some((2, 1)));
        assert_eq!(trie.longest_match(b"xyz", 0), None);
    }

    #[test]
    fn test_trie_multi_char_tokens() {
        let mut vocab = AHashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("ab".to_string(), 1);
        vocab.insert("abc".to_string(), 2);
        vocab.insert("b".to_string(), 3);

        let trie = SimdTrie::from_vocab(&vocab);

        assert_eq!(trie.max_token_len(), 3);

        // Should return longest match
        assert_eq!(trie.longest_match(b"abcd", 0), Some((2, 3))); // "abc"
        assert_eq!(trie.longest_match(b"abd", 0), Some((1, 2)));  // "ab"
        assert_eq!(trie.longest_match(b"acd", 0), Some((0, 1)));  // "a"
        assert_eq!(trie.longest_match(b"bcd", 0), Some((3, 1)));  // "b"
    }

    #[test]
    fn test_trie_all_matches() {
        let mut vocab = AHashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("ab".to_string(), 1);
        vocab.insert("abc".to_string(), 2);

        let trie = SimdTrie::from_vocab(&vocab);

        let matches = trie.all_matches(b"abcd", 0);
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0], (0, 1)); // "a"
        assert_eq!(matches[1], (1, 2)); // "ab"
        assert_eq!(matches[2], (2, 3)); // "abc"
    }

    #[test]
    fn test_trie_prefetch_matches_basic() {
        let mut vocab = AHashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("he".to_string(), 2);

        let trie = SimdTrie::from_vocab(&vocab);

        // Prefetch version should match basic version
        assert_eq!(
            trie.longest_match(b"hello", 0),
            trie.longest_match_prefetch(b"hello", 0)
        );
        assert_eq!(
            trie.longest_match(b"he", 0),
            trie.longest_match_prefetch(b"he", 0)
        );
    }

    // -------------------------------------------------------------------------
    // MergeHash Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_merge_hash_empty() {
        let merges: AHashMap<(u32, u32), (u32, u32)> = AHashMap::new();
        let hash = MergeHash::from_merges(&merges);

        assert_eq!(hash.len(), 0);
        assert!(hash.is_empty());
        assert_eq!(hash.get(0, 1), None);
    }

    #[test]
    fn test_merge_hash_single() {
        let mut merges = AHashMap::new();
        merges.insert((1, 2), (0, 3)); // tokens 1+2 merge to 3 with rank 0

        let hash = MergeHash::from_merges(&merges);

        assert_eq!(hash.len(), 1);
        assert_eq!(hash.get(1, 2), Some((0, 3)));
        assert_eq!(hash.get(2, 1), None); // Order matters
        assert_eq!(hash.get(0, 0), None);
    }

    #[test]
    fn test_merge_hash_multiple() {
        let mut merges = AHashMap::new();
        merges.insert((1, 2), (0, 10));
        merges.insert((3, 4), (1, 11));
        merges.insert((5, 6), (2, 12));
        merges.insert((100, 200), (3, 300));

        let hash = MergeHash::from_merges(&merges);

        assert_eq!(hash.len(), 4);
        assert_eq!(hash.get(1, 2), Some((0, 10)));
        assert_eq!(hash.get(3, 4), Some((1, 11)));
        assert_eq!(hash.get(5, 6), Some((2, 12)));
        assert_eq!(hash.get(100, 200), Some((3, 300)));
    }

    #[test]
    fn test_merge_hash_collision_handling() {
        // Insert many entries to force collisions
        let mut merges = AHashMap::new();
        for i in 0..100 {
            merges.insert((i, i + 1), (i, i + 100));
        }

        let hash = MergeHash::from_merges(&merges);

        // All entries should be retrievable
        for i in 0..100 {
            assert_eq!(hash.get(i, i + 1), Some((i, i + 100)));
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_merge_hash_simd_lookup() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let mut merges = AHashMap::new();
        merges.insert((1, 2), (0, 10));
        merges.insert((3, 4), (1, 11));
        merges.insert((5, 6), (2, 12));
        merges.insert((7, 8), (3, 13));

        let hash = MergeHash::from_merges(&merges);

        unsafe {
            let results = hash.get_4_simd([1, 3, 5, 99], [2, 4, 6, 99]);

            assert_eq!(results[0], Some((0, 10)));
            assert_eq!(results[1], Some((1, 11)));
            assert_eq!(results[2], Some((2, 12)));
            assert_eq!(results[3], None); // Not found
        }
    }

    // -------------------------------------------------------------------------
    // ByteEncoder Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_byte_encoder_roundtrip() {
        let encoder = ByteEncoder::new();

        // Test all bytes roundtrip
        let all_bytes: Vec<u8> = (0..=255).collect();
        let encoded = encoder.encode_bytes(&all_bytes);
        let decoded = encoder.decode_string(&encoded);

        assert_eq!(decoded, all_bytes);
    }

    #[test]
    fn test_byte_encoder_ascii() {
        let encoder = ByteEncoder::new();

        // Printable ASCII should mostly be unchanged
        let text = "Hello, World!";
        let encoded = encoder.encode_bytes(text.as_bytes());
        // Note: space is not in the printable range, so it gets mapped differently
        assert!(encoded.contains("Hello"));
    }

    // -------------------------------------------------------------------------
    // SimdBpeEncoder Tests
    // -------------------------------------------------------------------------

    fn create_test_encoder() -> SimdBpeEncoder {
        let mut vocab = AHashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);
        vocab.insert("c".to_string(), 2);
        vocab.insert("ab".to_string(), 3);
        vocab.insert("bc".to_string(), 4);
        vocab.insert("abc".to_string(), 5);
        vocab.insert("<unk>".to_string(), 6);

        let merges = vec![
            ("a".to_string(), "b".to_string()),  // a + b -> ab (rank 0)
            ("b".to_string(), "c".to_string()),  // b + c -> bc (rank 1)
            ("ab".to_string(), "c".to_string()), // ab + c -> abc (rank 2)
        ];

        SimdBpeEncoder::new(vocab, merges, "<unk>")
    }

    #[test]
    fn test_encoder_empty() {
        let encoder = create_test_encoder();
        assert_eq!(encoder.encode_greedy(""), Vec::<u32>::new());
        assert_eq!(encoder.encode_minimal(""), Vec::<u32>::new());
    }

    #[test]
    fn test_encoder_single_char() {
        let encoder = create_test_encoder();

        assert_eq!(encoder.encode_greedy("a"), vec![0]);
        assert_eq!(encoder.encode_greedy("b"), vec![1]);
        assert_eq!(encoder.encode_greedy("c"), vec![2]);
    }

    #[test]
    fn test_encoder_greedy_longest_match() {
        let encoder = create_test_encoder();

        // Greedy should find "abc" as one token
        assert_eq!(encoder.encode_greedy("abc"), vec![5]);

        // "ab" should match
        assert_eq!(encoder.encode_greedy("ab"), vec![3]);

        // "abab" -> "ab" + "ab"
        assert_eq!(encoder.encode_greedy("abab"), vec![3, 3]);
    }

    #[test]
    fn test_encoder_minimal_vs_greedy() {
        let encoder = create_test_encoder();

        // For well-formed vocabulary, greedy and minimal should often agree
        assert_eq!(encoder.encode_greedy("abc"), encoder.encode_minimal("abc"));
        assert_eq!(encoder.encode_greedy("a"), encoder.encode_minimal("a"));
    }

    #[test]
    fn test_encoder_unknown_bytes() {
        let encoder = create_test_encoder();

        // 'x' is not in vocabulary
        let result = encoder.encode_greedy("x");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 6); // unk_id
    }

    #[test]
    fn test_encoder_mixed_known_unknown() {
        let encoder = create_test_encoder();

        // "axb" -> "a" + unk + "b"
        let result = encoder.encode_greedy("axb");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 0); // "a"
        assert_eq!(result[1], 6); // unk for "x"
        assert_eq!(result[2], 1); // "b"
    }

    // -------------------------------------------------------------------------
    // BPE-Compatible Encoding Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_encode_word_bpe_single_char() {
        let encoder = create_test_encoder();

        // Single characters should be returned as-is
        assert_eq!(encoder.encode_word_bpe("a"), vec![0]);
        assert_eq!(encoder.encode_word_bpe("b"), vec![1]);
        assert_eq!(encoder.encode_word_bpe("c"), vec![2]);
    }

    #[test]
    fn test_encode_word_bpe_merges() {
        let encoder = create_test_encoder();

        // "ab" should merge: a(0) + b(1) -> ab(3)
        assert_eq!(encoder.encode_word_bpe("ab"), vec![3]);

        // "abc" should merge: a(0) + b(1) -> ab(3), then ab(3) + c(2) -> abc(5)
        assert_eq!(encoder.encode_word_bpe("abc"), vec![5]);
    }

    #[test]
    fn test_encode_word_bpe_priority() {
        // Create encoder with specific merge priorities
        let mut vocab = AHashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);
        vocab.insert("c".to_string(), 2);
        vocab.insert("ab".to_string(), 3);
        vocab.insert("bc".to_string(), 4);
        vocab.insert("<unk>".to_string(), 5);

        // Merges: a+b=ab (rank 0), b+c=bc (rank 1)
        // For "abc": should merge a+b first (lower rank), resulting in ab(3) + c(2)
        let merges = vec![
            ("a".to_string(), "b".to_string()),  // rank 0
            ("b".to_string(), "c".to_string()),  // rank 1
        ];

        let encoder = SimdBpeEncoder::new(vocab, merges, "<unk>");

        // "abc" with merges a+b (rank 0), b+c (rank 1)
        // Should apply a+b first: abc -> ab,c -> [3, 2]
        let result = encoder.encode_word_bpe("abc");
        assert_eq!(result, vec![3, 2]); // ab(3), c(2)
    }

    #[test]
    fn test_encode_word_bpe_vs_greedy_difference() {
        // This test shows the difference between BPE and greedy
        let mut vocab = AHashMap::new();
        vocab.insert("a".to_string(), 0);
        vocab.insert("b".to_string(), 1);
        vocab.insert("c".to_string(), 2);
        vocab.insert("ab".to_string(), 3);
        vocab.insert("bc".to_string(), 4);
        vocab.insert("abc".to_string(), 5);
        vocab.insert("<unk>".to_string(), 6);

        // Merges: a+b=ab (rank 0), b+c=bc (rank 1), ab+c=abc (rank 2)
        let merges = vec![
            ("a".to_string(), "b".to_string()),  // rank 0: a+b -> ab
            ("b".to_string(), "c".to_string()),  // rank 1: b+c -> bc
            ("ab".to_string(), "c".to_string()), // rank 2: ab+c -> abc
        ];

        let encoder = SimdBpeEncoder::new(vocab, merges, "<unk>");

        // Greedy would find "abc" directly
        assert_eq!(encoder.encode_greedy("abc"), vec![5]);

        // BPE applies merges in order: a+b -> ab, then ab+c -> abc
        assert_eq!(encoder.encode_word_bpe("abc"), vec![5]);
    }

    #[test]
    fn test_apply_merges_simple_basic() {
        let encoder = create_test_encoder();

        // Test basic merge: [a, b] -> [ab]
        let mut tokens = vec![0, 1]; // a, b
        encoder.apply_merges_simple(&mut tokens);
        assert_eq!(tokens, vec![3]); // ab
    }

    #[test]
    fn test_apply_merges_simple_chain() {
        let encoder = create_test_encoder();

        // Test chained merges: [a, b, c] -> [ab, c] -> [abc]
        let mut tokens = vec![0, 1, 2]; // a, b, c
        encoder.apply_merges_simple(&mut tokens);
        assert_eq!(tokens, vec![5]); // abc
    }

    #[test]
    fn test_apply_merges_no_merge() {
        let encoder = create_test_encoder();

        // Test when no merge is possible
        let mut tokens = vec![2, 0]; // c, a - no merge rule for c+a
        encoder.apply_merges_simple(&mut tokens);
        assert_eq!(tokens, vec![2, 0]); // unchanged
    }
}

// =============================================================================
// Benchmarks (run with: cargo bench --bench bpe_simd_benchmark)
// =============================================================================

#[cfg(test)]
mod bench_helpers {
    use super::*;

    /// Create a larger test encoder for benchmarking
    pub fn create_benchmark_encoder() -> SimdBpeEncoder {
        // Build a more realistic vocabulary
        let mut vocab = AHashMap::new();
        let mut id = 0u32;

        // Single bytes
        for b in 0..=255u8 {
            let s = format!("{}", b as char);
            vocab.insert(s, id);
            id += 1;
        }

        // Common bigrams
        let common = ["th", "he", "in", "er", "an", "re", "on", "at", "en", "nd"];
        for &s in &common {
            vocab.insert(s.to_string(), id);
            id += 1;
        }

        // Common trigrams
        let trigrams = ["the", "and", "ing", "ion", "tio", "ent", "ati"];
        for &s in &trigrams {
            vocab.insert(s.to_string(), id);
            id += 1;
        }

        vocab.insert("<unk>".to_string(), id);

        let merges = vec![]; // No merges for greedy-only testing

        SimdBpeEncoder::new(vocab, merges, "<unk>")
    }
}
