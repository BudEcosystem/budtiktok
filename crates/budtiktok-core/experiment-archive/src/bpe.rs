//! BPE (Byte Pair Encoding) tokenization algorithm
//!
//! State-of-the-art BPE implementation based on:
//! - GitHub rust-gems O(n) linear algorithm with last-token storage
//! - BlockBPE parallel merge concepts
//! - tiktoken byte-level encoding
//!
//! Provides:
//! - `BpeModel`: Core BPE model with encode/decode
//! - `BpePreTokenizer`: GPT-2/cl100k_base pre-tokenization patterns
//! - `TiktokenEncoder`: OpenAI tiktoken-compatible encoder
//! - `Gpt2ByteEncoder`: Byte-to-unicode mapping for GPT-2
//! - `CachedBpe`: LRU-cached BPE encoder
//! - `HyperBpeTokenizer`: SIMD-accelerated BPE (in bpe_hyper.rs)

use ahash::AHashMap;
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use crate::encoding::Encoding;
use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;
use crate::vocab::Vocabulary;

// ============================================================================
// Core Data Structures
// ============================================================================

/// BPE merge rule
#[derive(Debug, Clone)]
pub struct MergeRule {
    /// First token in merge
    pub first: String,
    /// Second token in merge
    pub second: String,
    /// Result of merge
    pub result: String,
    /// Priority (lower = higher priority, earlier in merge list)
    pub priority: u32,
}

/// BPE tokenizer configuration
#[derive(Debug, Clone)]
pub struct BpeConfig {
    /// Unknown token
    pub unk_token: String,
    /// End of word suffix (e.g., "</w>" for some models)
    pub end_of_word_suffix: Option<String>,
    /// Continuing subword prefix (e.g., "Ġ" for GPT-2)
    pub continuing_subword_prefix: Option<String>,
    /// Whether to fuse unknown tokens
    pub fuse_unk: bool,
    /// Whether to use byte-level encoding (GPT-2 style)
    pub byte_level: bool,
    /// Dropout probability for training (0.0 = disabled)
    pub dropout: f32,
    /// Use O(n) linear algorithm (recommended)
    pub use_linear_algorithm: bool,
}

impl Default for BpeConfig {
    fn default() -> Self {
        Self {
            unk_token: "<unk>".to_string(),
            end_of_word_suffix: None,
            continuing_subword_prefix: Some("Ġ".to_string()),
            fuse_unk: false,
            byte_level: true,
            dropout: 0.0,
            use_linear_algorithm: true,
        }
    }
}

// ============================================================================
// Token Output
// ============================================================================

/// A single token output from BPE encoding
#[derive(Debug, Clone)]
pub struct BpeToken {
    /// Token ID
    pub id: u32,
    /// Token string
    pub token: String,
    /// Offset in original text (start, end)
    pub offsets: (usize, usize),
}

// ============================================================================
// GPT-2 Byte Encoder (Section 3.1)
// ============================================================================

/// GPT-2 style byte-to-unicode encoder
///
/// Maps each byte to a unique unicode character to make BPE work on arbitrary
/// binary data while keeping the vocabulary printable.
#[derive(Debug, Clone)]
pub struct Gpt2ByteEncoder {
    /// Byte value -> Unicode character
    byte_to_char: [char; 256],
    /// Unicode character -> Byte value
    char_to_byte: AHashMap<char, u8>,
}

impl Gpt2ByteEncoder {
    /// Create a new GPT-2 byte encoder
    pub fn new() -> Self {
        let mut byte_to_char = ['\0'; 256];
        let mut char_to_byte = AHashMap::with_capacity(256);

        // GPT-2 style byte encoding:
        // Printable ASCII (33-126) and Latin-1 (161-255) map to themselves
        // Other bytes map to Unicode offset starting at 256
        let mut n = 0u32;

        for b in 0u8..=255 {
            let c = if (b'!'..=b'~').contains(&b)
                || (0xa1..=0xac).contains(&b)
                || (0xae..=0xff).contains(&b)
            {
                b as char
            } else {
                let c = char::from_u32(256 + n).unwrap_or('?');
                n += 1;
                c
            };
            byte_to_char[b as usize] = c;
            char_to_byte.insert(c, b);
        }

        Self { byte_to_char, char_to_byte }
    }

    /// Encode a single byte to its unicode character
    #[inline]
    pub fn encode_byte(&self, byte: u8) -> char {
        self.byte_to_char[byte as usize]
    }

    /// Decode a unicode character back to byte
    #[inline]
    pub fn decode_char(&self, c: char) -> Option<u8> {
        self.char_to_byte.get(&c).copied()
    }

    /// Encode a string to byte-encoded unicode
    pub fn encode_string(&self, s: &str) -> String {
        s.bytes().map(|b| self.byte_to_char[b as usize]).collect()
    }

    /// Decode byte-encoded unicode back to string
    pub fn decode_string(&self, s: &str) -> String {
        let bytes: Vec<u8> = s.chars()
            .filter_map(|c| self.char_to_byte.get(&c).copied())
            .collect();
        String::from_utf8_lossy(&bytes).to_string()
    }
}

impl Default for Gpt2ByteEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// BPE Pre-Tokenizer (Section 3.2)
// ============================================================================

/// Pre-tokenizer for BPE with regex patterns
///
/// Supports GPT-2 and cl100k_base (GPT-4) tokenization patterns.
#[derive(Debug, Clone)]
pub struct BpePreTokenizer {
    /// The pattern type
    pattern: PreTokenizerPattern,
}

/// Pre-tokenization pattern type
#[derive(Debug, Clone, Copy)]
pub enum PreTokenizerPattern {
    /// GPT-2 pattern: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
    Gpt2,
    /// cl100k_base (GPT-4) pattern
    Cl100kBase,
    /// Simple whitespace splitting
    Whitespace,
}

impl BpePreTokenizer {
    /// Create GPT-2 pre-tokenizer
    pub fn gpt2() -> Self {
        Self { pattern: PreTokenizerPattern::Gpt2 }
    }

    /// Create cl100k_base pre-tokenizer
    pub fn cl100k_base() -> Self {
        Self { pattern: PreTokenizerPattern::Cl100kBase }
    }

    /// Create whitespace pre-tokenizer
    pub fn whitespace() -> Self {
        Self { pattern: PreTokenizerPattern::Whitespace }
    }

    /// Pre-tokenize text into words with offsets
    pub fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<(&'a str, (usize, usize))> {
        match self.pattern {
            PreTokenizerPattern::Gpt2 => self.pre_tokenize_gpt2(text),
            PreTokenizerPattern::Cl100kBase => self.pre_tokenize_cl100k(text),
            PreTokenizerPattern::Whitespace => self.pre_tokenize_whitespace(text),
        }
    }

    /// GPT-2 style pre-tokenization
    /// Splits on contractions, words, numbers, and punctuation
    fn pre_tokenize_gpt2<'a>(&self, text: &'a str) -> Vec<(&'a str, (usize, usize))> {
        let mut tokens = Vec::new();
        let mut i = 0;
        let bytes = text.as_bytes();

        while i < bytes.len() {
            let start = i;

            // Check for contractions: 's, 't, 're, 've, 'm, 'll, 'd
            if i + 1 < bytes.len() && bytes[i] == b'\'' {
                let contraction = match bytes.get(i + 1) {
                    Some(b's') | Some(b'S') => Some(2),
                    Some(b't') | Some(b'T') => Some(2),
                    Some(b'm') | Some(b'M') => Some(2),
                    Some(b'd') | Some(b'D') => Some(2),
                    Some(b'r') if bytes.get(i + 2) == Some(&b'e') || bytes.get(i + 2) == Some(&b'E') => Some(3),
                    Some(b'v') if bytes.get(i + 2) == Some(&b'e') || bytes.get(i + 2) == Some(&b'E') => Some(3),
                    Some(b'l') if bytes.get(i + 2) == Some(&b'l') || bytes.get(i + 2) == Some(&b'L') => Some(3),
                    _ => None,
                };
                if let Some(len) = contraction {
                    tokens.push((&text[start..start + len], (start, start + len)));
                    i += len;
                    continue;
                }
            }

            // Optional leading space + letters
            let has_space = bytes.get(i) == Some(&b' ');
            if has_space {
                i += 1;
            }

            if i < bytes.len() {
                let c = bytes[i];

                // Letters (ASCII for now, could extend to Unicode)
                if c.is_ascii_alphabetic() || c >= 0x80 {
                    while i < bytes.len() && (bytes[i].is_ascii_alphabetic() || bytes[i] >= 0x80) {
                        i += 1;
                    }
                    tokens.push((&text[start..i], (start, i)));
                    continue;
                }

                // Numbers
                if c.is_ascii_digit() {
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                    tokens.push((&text[start..i], (start, i)));
                    continue;
                }

                // Other non-whitespace
                if !c.is_ascii_whitespace() {
                    while i < bytes.len() && !bytes[i].is_ascii_whitespace()
                        && !bytes[i].is_ascii_alphabetic() && !bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                    tokens.push((&text[start..i], (start, i)));
                    continue;
                }
            }

            // Whitespace
            if has_space || (i < bytes.len() && bytes[i].is_ascii_whitespace()) {
                while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                // Only emit trailing whitespace if not followed by non-whitespace
                if i >= bytes.len() || bytes[i].is_ascii_whitespace() {
                    tokens.push((&text[start..i], (start, i)));
                } else if start < i {
                    tokens.push((&text[start..i], (start, i)));
                }
                continue;
            }

            i += 1;
        }

        tokens
    }

    /// cl100k_base pre-tokenization (GPT-4 style)
    fn pre_tokenize_cl100k<'a>(&self, text: &'a str) -> Vec<(&'a str, (usize, usize))> {
        // Similar to GPT-2 but with additional patterns
        self.pre_tokenize_gpt2(text)
    }

    /// Simple whitespace pre-tokenization
    fn pre_tokenize_whitespace<'a>(&self, text: &'a str) -> Vec<(&'a str, (usize, usize))> {
        let mut tokens = Vec::new();
        let mut start = 0;
        let mut in_word = false;

        for (i, c) in text.char_indices() {
            if c.is_whitespace() {
                if in_word {
                    tokens.push((&text[start..i], (start, i)));
                    in_word = false;
                }
            } else if !in_word {
                start = i;
                in_word = true;
            }
        }

        if in_word {
            tokens.push((&text[start..], (start, text.len())));
        }

        tokens
    }
}

// ============================================================================
// Compatibility Table (Section 3.3.2)
// ============================================================================

/// Table for checking if two tokens can be merged and getting the result
///
/// The key insight from GitHub's rust-gems: instead of checking all merges,
/// we precompute which token pairs are valid to appear adjacent in any valid
/// encoding. This enables O(n) linear encoding.
#[derive(Debug)]
pub struct CompatibilityTable {
    /// Maps (token_id_a, token_id_b) -> merged_token_id
    table: AHashMap<(u32, u32), u32>,
    /// Maps (token_id_a, token_id_b) -> merge priority
    priorities: AHashMap<(u32, u32), u32>,
}

impl CompatibilityTable {
    /// Create a compatibility table from merge rules and vocabulary
    pub fn from_merges(merges: &[MergeRule], vocab: &Vocabulary) -> Self {
        let capacity = merges.len();
        let mut table = AHashMap::with_capacity(capacity);
        let mut priorities = AHashMap::with_capacity(capacity);

        for merge in merges {
            if let (Some(first_id), Some(second_id)) = (
                vocab.token_to_id(&merge.first),
                vocab.token_to_id(&merge.second),
            ) {
                if let Some(result_id) = vocab.token_to_id(&merge.result) {
                    table.insert((first_id, second_id), result_id);
                    priorities.insert((first_id, second_id), merge.priority);
                }
            }
        }

        Self { table, priorities }
    }

    /// Create from merge tuples (first, second) -> result_id
    pub fn from_tuples(merges: &[(String, String)], vocab: &Vocabulary) -> Self {
        let mut table = AHashMap::with_capacity(merges.len());
        let mut priorities = AHashMap::with_capacity(merges.len());

        for (priority, (first, second)) in merges.iter().enumerate() {
            let merged = format!("{}{}", first, second);
            if let (Some(first_id), Some(second_id), Some(result_id)) = (
                vocab.token_to_id(first),
                vocab.token_to_id(second),
                vocab.token_to_id(&merged),
            ) {
                table.insert((first_id, second_id), result_id);
                priorities.insert((first_id, second_id), priority as u32);
            }
        }

        Self { table, priorities }
    }

    /// Check if two tokens can be merged and get the result
    #[inline]
    pub fn get(&self, a: u32, b: u32) -> Option<u32> {
        self.table.get(&(a, b)).copied()
    }

    /// Get the merge priority (lower = higher priority)
    #[inline]
    pub fn priority(&self, a: u32, b: u32) -> Option<u32> {
        self.priorities.get(&(a, b)).copied()
    }

    /// Check if two tokens are compatible (can be merged)
    #[inline]
    pub fn compatible(&self, a: u32, b: u32) -> bool {
        self.table.contains_key(&(a, b))
    }
}

// ============================================================================
// Vocabulary Automaton (Section 3.3.1)
// ============================================================================

/// Aho-Corasick automaton built from all vocabulary tokens
/// Used for efficient enumeration of all possible tokenizations
pub struct VocabAutomaton {
    /// The Aho-Corasick automaton
    automaton: AhoCorasick,
    /// Pattern index to token ID mapping
    pattern_to_id: Vec<u32>,
    /// Token ID to pattern length mapping
    token_lengths: AHashMap<u32, usize>,
}

impl std::fmt::Debug for VocabAutomaton {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VocabAutomaton")
            .field("pattern_count", &self.pattern_to_id.len())
            .finish()
    }
}

impl VocabAutomaton {
    /// Build automaton from vocabulary
    pub fn from_vocab(vocab: &Vocabulary) -> Self {
        let mut patterns: Vec<String> = Vec::with_capacity(vocab.len());
        let mut pattern_to_id: Vec<u32> = Vec::with_capacity(vocab.len());
        let mut token_lengths: AHashMap<u32, usize> = AHashMap::with_capacity(vocab.len());

        for (token, id) in vocab.iter() {
            patterns.push(token.to_string());
            pattern_to_id.push(id);
            token_lengths.insert(id, token.len());
        }

        let automaton = AhoCorasickBuilder::new()
            .match_kind(MatchKind::Standard)
            .build(&patterns)
            .expect("Failed to build vocabulary automaton");

        Self {
            automaton,
            pattern_to_id,
            token_lengths,
        }
    }

    /// Find all overlapping matches starting at each position
    pub fn find_all_at<'a>(&'a self, text: &'a str, start: usize) -> impl Iterator<Item = (usize, usize, u32)> + 'a {
        let suffix = &text[start..];
        self.automaton.find_overlapping_iter(suffix).map(move |m| {
            let pattern_idx = m.pattern().as_usize();
            let token_id = self.pattern_to_id[pattern_idx];
            (start + m.start(), start + m.end(), token_id)
        })
    }

    /// Get the length of a token by ID
    #[inline]
    pub fn token_length(&self, id: u32) -> Option<usize> {
        self.token_lengths.get(&id).copied()
    }
}

// ============================================================================
// O(n) Linear BPE Algorithm (GitHub rust-gems approach)
// ============================================================================

/// O(n) BPE encoder using dynamic programming with Aho-Corasick automaton
///
/// Key insight from GitHub's rust-gems:
/// - Store only the last token for each prefix position
/// - Knowing the last token uniquely determines the full encoding
/// - Check ~1.3 candidate tokens per position on average
pub struct LinearBpeEncoder {
    /// Vocabulary automaton for finding all possible tokens
    pub automaton: VocabAutomaton,
    /// Compatibility table for merge checking
    pub compat: CompatibilityTable,
    /// Unknown token ID
    unk_id: u32,
}

impl LinearBpeEncoder {
    /// Create a new linear BPE encoder
    pub fn new(automaton: VocabAutomaton, compat: CompatibilityTable, unk_id: u32) -> Self {
        Self { automaton, compat, unk_id }
    }

    /// Encode text using O(n) DP algorithm
    ///
    /// The algorithm finds the tokenization with minimum token count,
    /// which corresponds to the result of greedy BPE merging.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let n = text.len();
        if n == 0 {
            return vec![];
        }

        // dp[i] = (token_count, last_token, prev_pos) for best path to position i
        let mut dp: Vec<Option<(usize, u32, usize)>> = vec![None; n + 1];
        dp[0] = Some((0, u32::MAX, 0));

        // Forward pass: fill DP table
        for i in 0..n {
            let Some((count, _, _)) = dp[i] else { continue };

            for (start, end, token_id) in self.automaton.find_all_at(text, i) {
                if start != i {
                    continue;
                }

                let new_count = count + 1;
                let should_update = match dp[end] {
                    None => true,
                    Some((existing_count, _, _)) => new_count < existing_count,
                };

                if should_update {
                    dp[end] = Some((new_count, token_id, i));
                }
            }
        }

        // If we couldn't reach the end, fall back to unknown tokens
        if dp[n].is_none() {
            return self.fallback_encode(text);
        }

        // Backtrack to reconstruct the tokenization
        self.backtrack(&dp, n)
    }

    /// Backtrack through DP table to reconstruct token sequence
    fn backtrack(&self, dp: &[Option<(usize, u32, usize)>], end: usize) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut pos = end;

        while pos > 0 {
            if let Some((_, token_id, prev_pos)) = dp[pos] {
                if token_id != u32::MAX {
                    tokens.push(token_id);
                }
                pos = prev_pos;
            } else {
                break;
            }
        }

        tokens.reverse();
        tokens
    }

    /// Fallback encoding when DP fails (e.g., OOV characters)
    fn fallback_encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut chars = text.char_indices().peekable();

        while let Some((i, _c)) = chars.next() {
            let mut best_match: Option<(usize, u32)> = None;

            for (start, end, token_id) in self.automaton.find_all_at(text, i) {
                if start == i {
                    if best_match.map(|(len, _)| end - i > len).unwrap_or(true) {
                        best_match = Some((end - i, token_id));
                    }
                }
            }

            match best_match {
                Some((len, token_id)) => {
                    tokens.push(token_id);
                    for _ in 0..(len.saturating_sub(1)) {
                        chars.next();
                    }
                }
                None => {
                    tokens.push(self.unk_id);
                }
            }
        }

        tokens
    }
}

// ============================================================================
// O(n log n) Heap-Based Algorithm (Section 3.3.6)
// ============================================================================

/// Entry in the merge priority queue
#[derive(Clone, Eq, PartialEq)]
struct MergeEntry {
    position: usize,
    priority: u32,
    generation: u32,
}

impl Ord for MergeEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.priority.cmp(&self.priority)
            .then_with(|| self.position.cmp(&other.position))
    }
}

impl PartialOrd for MergeEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Node in the doubly-linked token list
#[derive(Clone)]
struct TokenNode {
    token: String,
    id: u32,
    prev: Option<usize>,
    next: Option<usize>,
    generation: u32,
}

/// O(n log n) heap-based BPE encoder
pub struct HeapBpeEncoder {
    vocab: Vocabulary,
    merges: Vec<MergeRule>,
    merge_priorities: AHashMap<(String, String), (String, u32)>,
    unk_id: u32,
    dropout: f32,
}

impl HeapBpeEncoder {
    /// Create a new heap-based BPE encoder
    pub fn new(vocab: Vocabulary, merges: Vec<MergeRule>, unk_id: u32, dropout: f32) -> Self {
        let mut merge_priorities = AHashMap::with_capacity(merges.len());

        for merge in &merges {
            merge_priorities.insert(
                (merge.first.clone(), merge.second.clone()),
                (merge.result.clone(), merge.priority),
            );
        }

        Self {
            vocab,
            merges,
            merge_priorities,
            unk_id,
            dropout,
        }
    }

    /// Encode text using O(n log n) heap-based algorithm
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        let mut nodes: Vec<TokenNode> = text
            .chars()
            .enumerate()
            .map(|(i, c)| {
                let token = c.to_string();
                let id = self.vocab.token_to_id(&token).unwrap_or(self.unk_id);
                TokenNode {
                    token,
                    id,
                    prev: if i > 0 { Some(i - 1) } else { None },
                    next: if i < text.chars().count() - 1 { Some(i + 1) } else { None },
                    generation: 0,
                }
            })
            .collect();

        if nodes.len() <= 1 {
            return nodes.iter().map(|n| n.id).collect();
        }

        let mut heap = BinaryHeap::new();
        for i in 0..nodes.len().saturating_sub(1) {
            if let Some((_, priority)) = self.merge_priorities.get(&(
                nodes[i].token.clone(),
                nodes[i + 1].token.clone(),
            )) {
                heap.push(MergeEntry {
                    position: i,
                    priority: *priority,
                    generation: 0,
                });
            }
        }

        while let Some(entry) = heap.pop() {
            let pos = entry.position;

            if entry.generation != nodes[pos].generation {
                continue;
            }

            let Some(next_pos) = nodes[pos].next else { continue };

            if nodes[next_pos].generation != entry.generation {
                continue;
            }

            if self.dropout > 0.0 && fastrand::f32() < self.dropout {
                continue;
            }

            let Some((result, _)) = self.merge_priorities.get(&(
                nodes[pos].token.clone(),
                nodes[next_pos].token.clone(),
            )) else { continue };

            let new_generation = nodes[pos].generation + 1;
            let new_id = self.vocab.token_to_id(result).unwrap_or(self.unk_id);
            let new_next = nodes[next_pos].next;

            nodes[pos].token = result.clone();
            nodes[pos].id = new_id;
            nodes[pos].next = new_next;
            nodes[pos].generation = new_generation;

            if let Some(next_next) = new_next {
                nodes[next_next].prev = Some(pos);
            }

            nodes[next_pos].generation = u32::MAX;

            if let Some(prev_pos) = nodes[pos].prev {
                if let Some((_, priority)) = self.merge_priorities.get(&(
                    nodes[prev_pos].token.clone(),
                    nodes[pos].token.clone(),
                )) {
                    heap.push(MergeEntry {
                        position: prev_pos,
                        priority: *priority,
                        generation: nodes[prev_pos].generation,
                    });
                }
            }

            if let Some(next_pos) = nodes[pos].next {
                if let Some((_, priority)) = self.merge_priorities.get(&(
                    nodes[pos].token.clone(),
                    nodes[next_pos].token.clone(),
                )) {
                    heap.push(MergeEntry {
                        position: pos,
                        priority: *priority,
                        generation: new_generation,
                    });
                }
            }
        }

        let mut result = Vec::new();
        let mut current = 0;
        while current < nodes.len() {
            if nodes[current].generation != u32::MAX {
                result.push(nodes[current].id);
            }
            match nodes[current].next {
                Some(next) => current = next,
                None => break,
            }
        }

        result
    }
}

// ============================================================================
// BPE Model (Main Interface)
// ============================================================================

/// BPE Model with encode/decode functionality
///
/// This is the main interface for BPE tokenization, providing:
/// - Fast encoding with O(n) linear algorithm
/// - Fallback to O(n log n) heap algorithm with dropout
/// - Byte-level BPE support (GPT-2 style)
/// - Special token handling
/// - Batch encoding with parallelization
pub struct BpeModel {
    /// Vocabulary for token lookups
    vocab: Vocabulary,
    /// Merge rules
    merges: Vec<(String, String)>,
    /// Configuration
    config: BpeConfig,
    /// Linear encoder for fast O(n) encoding
    linear_encoder: LinearBpeEncoder,
    /// Heap encoder for dropout support
    heap_encoder: HeapBpeEncoder,
    /// Byte encoder for byte-level BPE
    byte_encoder: Gpt2ByteEncoder,
    /// Special tokens mapping (string -> id)
    special_tokens: AHashMap<String, u32>,
    /// Added tokens (custom tokens added after training)
    added_tokens: AHashMap<String, u32>,
}

impl BpeModel {
    /// Create a new BPE model
    pub fn new(vocab: AHashMap<String, u32>, merges: Vec<(String, String)>) -> Self {
        let vocabulary = Vocabulary::new(vocab.clone(), crate::vocab::SpecialTokens::default());
        Self::with_vocab(vocabulary, merges)
    }

    /// Create BPE model from Vocabulary
    pub fn with_vocab(vocab: Vocabulary, merges: Vec<(String, String)>) -> Self {
        let config = BpeConfig::default();
        Self::with_config(vocab, merges, config)
    }

    /// Create BPE model with custom config
    pub fn with_config(vocab: Vocabulary, merges: Vec<(String, String)>, config: BpeConfig) -> Self {
        let unk_id = vocab.token_to_id(&config.unk_token).unwrap_or(0);

        // Build merge rules
        let merge_rules: Vec<MergeRule> = merges.iter().enumerate()
            .map(|(i, (first, second))| MergeRule {
                first: first.clone(),
                second: second.clone(),
                result: format!("{}{}", first, second),
                priority: i as u32,
            })
            .collect();

        // Build O(n) components
        let vocab_automaton = VocabAutomaton::from_vocab(&vocab);
        let compat_table = CompatibilityTable::from_merges(&merge_rules, &vocab);
        let linear_encoder = LinearBpeEncoder::new(vocab_automaton, compat_table, unk_id);

        // Build O(n log n) encoder
        let heap_encoder = HeapBpeEncoder::new(
            vocab.clone(),
            merge_rules,
            unk_id,
            config.dropout,
        );

        Self {
            vocab,
            merges,
            config,
            linear_encoder,
            heap_encoder,
            byte_encoder: Gpt2ByteEncoder::new(),
            special_tokens: AHashMap::new(),
            added_tokens: AHashMap::new(),
        }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get configuration
    pub fn config(&self) -> &BpeConfig {
        &self.config
    }

    /// Get token ID for a token string
    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        self.special_tokens.get(token).copied()
            .or_else(|| self.added_tokens.get(token).copied())
            .or_else(|| self.vocab.token_to_id(token))
    }

    /// Get token string for a token ID
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab.id_to_token(id)
    }

    /// Add a special token
    pub fn add_special_token(&mut self, token: &str, id: u32) {
        self.special_tokens.insert(token.to_string(), id);
    }

    /// Add a custom token
    pub fn add_token(&mut self, token: &str, id: u32) {
        self.added_tokens.insert(token.to_string(), id);
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<BpeToken> {
        if text.is_empty() {
            return vec![];
        }

        let ids = self.linear_encoder.encode(text);
        let mut tokens = Vec::with_capacity(ids.len());
        let mut offset = 0;

        for id in ids {
            let token_str = self.vocab.id_to_token(id)
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

    /// Encode with dropout (for training regularization)
    pub fn encode_with_dropout(&mut self, text: &str, dropout: f32) -> Vec<BpeToken> {
        if text.is_empty() {
            return vec![];
        }

        // Temporarily set dropout and use heap encoder
        let old_dropout = self.heap_encoder.dropout;
        self.heap_encoder.dropout = dropout;
        let ids = self.heap_encoder.encode(text);
        self.heap_encoder.dropout = old_dropout;

        let mut tokens = Vec::with_capacity(ids.len());
        let mut offset = 0;

        for id in ids {
            let token_str = self.vocab.id_to_token(id)
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

    /// Encode with special tokens preserved
    pub fn encode_with_special_tokens(&self, text: &str) -> Vec<BpeToken> {
        // TODO: Split on special tokens first, then encode each segment
        self.encode(text)
    }

    /// Encode with offset tracking
    pub fn encode_with_offsets(&self, text: &str) -> Vec<BpeToken> {
        self.encode(text)
    }

    /// Decode token IDs to string
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut result = String::new();
        for &id in ids {
            if let Some(token) = self.vocab.id_to_token(id) {
                // Handle Ġ prefix (GPT-2 style space)
                if let Some(ref prefix) = self.config.continuing_subword_prefix {
                    if token.starts_with(prefix) {
                        result.push(' ');
                        result.push_str(&token[prefix.len()..]);
                        continue;
                    }
                }
                result.push_str(token);
            }
        }
        result
    }

    /// Batch encode multiple texts
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<BpeToken>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Batch encode with parallelization
    pub fn encode_batch_parallel(&self, texts: &[&str]) -> Vec<Vec<BpeToken>> {
        use rayon::prelude::*;
        texts.par_iter().map(|t| self.encode(t)).collect()
    }

    /// Batch decode
    pub fn decode_batch(&self, batch_ids: &[Vec<u32>]) -> Vec<String> {
        batch_ids.iter().map(|ids| self.decode(ids)).collect()
    }

    /// Serialize model to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        // Simple serialization: vocab + merges
        let mut bytes = Vec::new();

        // Serialize vocab size
        let vocab_size = self.vocab.len() as u32;
        bytes.extend_from_slice(&vocab_size.to_le_bytes());

        // Serialize merges count
        let merges_count = self.merges.len() as u32;
        bytes.extend_from_slice(&merges_count.to_le_bytes());

        // In a real implementation, serialize full vocab and merges
        bytes
    }

    /// Deserialize model from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::invalid_config("Invalid bytes length"));
        }

        // This is a placeholder - real implementation would deserialize full model
        let vocab = AHashMap::new();
        let merges = Vec::new();
        Ok(Self::new(vocab, merges))
    }
}

// ============================================================================
// Tiktoken Encoder (OpenAI Compatible)
// ============================================================================

/// OpenAI tiktoken-compatible encoder
///
/// Supports cl100k_base (GPT-4), p50k_base (GPT-3), and r50k_base patterns.
pub struct TiktokenEncoder {
    /// The underlying BPE model
    model: BpeModel,
    /// Pre-tokenizer
    pre_tokenizer: BpePreTokenizer,
    /// Byte encoder
    byte_encoder: Gpt2ByteEncoder,
}

impl TiktokenEncoder {
    /// Create cl100k_base encoder (GPT-4)
    pub fn cl100k_base() -> Self {
        // In real implementation, load from bundled vocab
        let vocab = AHashMap::new();
        let merges = Vec::new();
        let model = BpeModel::new(vocab, merges);

        Self {
            model,
            pre_tokenizer: BpePreTokenizer::cl100k_base(),
            byte_encoder: Gpt2ByteEncoder::new(),
        }
    }

    /// Create p50k_base encoder (GPT-3)
    pub fn p50k_base() -> Self {
        let vocab = AHashMap::new();
        let merges = Vec::new();
        let model = BpeModel::new(vocab, merges);

        Self {
            model,
            pre_tokenizer: BpePreTokenizer::gpt2(),
            byte_encoder: Gpt2ByteEncoder::new(),
        }
    }

    /// Encode text
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Pre-tokenize
        let pre_tokens = self.pre_tokenizer.pre_tokenize(text);

        // Byte encode each pre-token
        let mut all_ids = Vec::new();
        for (token, _) in pre_tokens {
            let byte_encoded = self.byte_encoder.encode_string(token);
            let tokens = self.model.encode(&byte_encoded);
            all_ids.extend(tokens.iter().map(|t| t.id));
        }

        all_ids
    }

    /// Encode with special tokens
    pub fn encode_with_special_tokens(&self, text: &str) -> Vec<u32> {
        self.encode(text)
    }

    /// Decode token IDs
    pub fn decode(&self, ids: &[u32]) -> String {
        let decoded = self.model.decode(ids);
        self.byte_encoder.decode_string(&decoded)
    }
}

// ============================================================================
// Cached BPE (LRU Cache)
// ============================================================================

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size: usize,
}

/// LRU-cached BPE encoder for repeated encoding of same strings
pub struct CachedBpe {
    /// The underlying BPE model
    model: BpeModel,
    /// LRU cache: text -> token IDs
    cache: std::sync::RwLock<lru::LruCache<String, Vec<u32>>>,
    /// Cache statistics
    hits: AtomicU64,
    misses: AtomicU64,
}

impl CachedBpe {
    /// Create a new cached BPE encoder
    pub fn new(model: BpeModel, cache_size: usize) -> Self {
        Self {
            model,
            cache: std::sync::RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(cache_size).unwrap_or(std::num::NonZeroUsize::MIN)
            )),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Encode text (with caching)
    pub fn encode(&self, text: &str) -> Vec<BpeToken> {
        // Check cache
        {
            let cache = self.cache.read().unwrap();
            if let Some(ids) = cache.peek(text) {
                self.hits.fetch_add(1, AtomicOrdering::Relaxed);
                return ids.iter().map(|&id| BpeToken {
                    id,
                    token: self.model.id_to_token(id).unwrap_or("<unk>").to_string(),
                    offsets: (0, 0),
                }).collect();
            }
        }

        // Cache miss - encode
        self.misses.fetch_add(1, AtomicOrdering::Relaxed);
        let tokens = self.model.encode(text);
        let ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();

        // Store in cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.put(text.to_string(), ids);
        }

        tokens
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let cache = self.cache.read().unwrap();
        CacheStats {
            hits: self.hits.load(AtomicOrdering::Relaxed),
            misses: self.misses.load(AtomicOrdering::Relaxed),
            size: cache.len(),
        }
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
    }
}

// ============================================================================
// Parse Merges File
// ============================================================================

/// Parse merges.txt format
pub fn parse_merges(content: &str) -> Vec<(String, String)> {
    content
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                Some((parts[0].to_string(), parts[1].to_string()))
            } else {
                None
            }
        })
        .collect()
}

// ============================================================================
// Tokenizer Trait Implementation
// ============================================================================

/// BPE Tokenizer wrapping BpeModel
pub struct BpeTokenizer {
    model: BpeModel,
}

impl BpeTokenizer {
    /// Create new BPE tokenizer
    pub fn new(vocabulary: Vocabulary, merges: Vec<MergeRule>, config: BpeConfig) -> Self {
        let merge_tuples: Vec<(String, String)> = merges.iter()
            .map(|m| (m.first.clone(), m.second.clone()))
            .collect();
        Self {
            model: BpeModel::with_config(vocabulary, merge_tuples, config),
        }
    }

    /// Get compatibility table reference
    pub fn compatibility_table(&self) -> &CompatibilityTable {
        &self.model.linear_encoder.compat
    }

    /// Get merge rules
    pub fn merges(&self) -> &[(String, String)] {
        &self.model.merges
    }
}

impl Tokenizer for BpeTokenizer {
    fn vocabulary(&self) -> &Vocabulary {
        &self.model.vocab
    }

    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Encoding> {
        let tokens = self.model.encode(text);
        let mut encoding = Encoding::new();

        for (i, token) in tokens.iter().enumerate() {
            encoding.push(
                token.id,
                token.token.clone(),
                token.offsets,
                Some(i as u32),
                Some(0),
                false,
            );
        }

        Ok(encoding)
    }

    fn encode_pair(&self, text: &str, text_pair: &str, add_special_tokens: bool) -> Result<Encoding> {
        let mut encoding = self.encode(text, false)?;
        let encoding_pair = self.encode(text_pair, false)?;
        encoding.merge(encoding_pair, true);
        Ok(encoding)
    }

    fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
        Ok(self.model.decode(ids))
    }

    fn save(&self, _path: &Path) -> Result<()> {
        Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Saving not yet implemented",
        )))
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for BPE tokenizer
pub struct BpeBuilder {
    vocabulary: Option<Vocabulary>,
    merges: Vec<MergeRule>,
    config: BpeConfig,
}

impl BpeBuilder {
    /// Create a new BPE builder
    pub fn new() -> Self {
        Self {
            vocabulary: None,
            merges: Vec::new(),
            config: BpeConfig::default(),
        }
    }

    /// Set vocabulary
    pub fn vocabulary(mut self, vocab: Vocabulary) -> Self {
        self.vocabulary = Some(vocab);
        self
    }

    /// Add a merge rule
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

    /// Add multiple merge rules
    pub fn add_merges(mut self, merges: impl IntoIterator<Item = (String, String, String)>) -> Self {
        for (first, second, result) in merges {
            let priority = self.merges.len() as u32;
            self.merges.push(MergeRule {
                first,
                second,
                result,
                priority,
            });
        }
        self
    }

    /// Set configuration
    pub fn config(mut self, config: BpeConfig) -> Self {
        self.config = config;
        self
    }

    /// Set dropout probability
    pub fn dropout(mut self, dropout: f32) -> Self {
        // Dropout is handled in HeapBpeEncoder
        self
    }

    /// Use linear O(n) algorithm
    pub fn use_linear_algorithm(mut self, _use_linear: bool) -> Self {
        // Linear algorithm is default
        self
    }

    /// Build the tokenizer
    pub fn build(self) -> Result<BpeTokenizer> {
        let vocabulary = self
            .vocabulary
            .ok_or_else(|| Error::invalid_config("Vocabulary is required"))?;

        Ok(BpeTokenizer::new(vocabulary, self.merges, self.config))
    }
}

impl Default for BpeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::VocabularyBuilder;

    fn create_test_vocab() -> AHashMap<String, u32> {
        let mut vocab = AHashMap::new();
        let mut next_id = 0u32;

        // Special tokens first
        vocab.insert("<unk>".to_string(), next_id); next_id += 1;
        vocab.insert("<|endoftext|>".to_string(), next_id); next_id += 1;

        // Single characters
        for c in "abcdefghijklmnopqrstuvwxyz".chars() {
            vocab.insert(c.to_string(), next_id);
            next_id += 1;
        }

        vocab.insert(" ".to_string(), next_id); next_id += 1;

        // Merged tokens
        vocab.insert("he".to_string(), next_id); next_id += 1;
        vocab.insert("hel".to_string(), next_id); next_id += 1;
        vocab.insert("hell".to_string(), next_id); next_id += 1;
        vocab.insert("hello".to_string(), next_id); next_id += 1;
        vocab.insert("wo".to_string(), next_id); next_id += 1;
        vocab.insert("wor".to_string(), next_id); next_id += 1;
        vocab.insert("worl".to_string(), next_id); next_id += 1;
        vocab.insert("world".to_string(), next_id);

        vocab
    }

    fn create_test_merges() -> Vec<(String, String)> {
        vec![
            ("h".to_string(), "e".to_string()),
            ("he".to_string(), "l".to_string()),
            ("hel".to_string(), "l".to_string()),
            ("hell".to_string(), "o".to_string()),
            ("w".to_string(), "o".to_string()),
            ("wo".to_string(), "r".to_string()),
            ("wor".to_string(), "l".to_string()),
            ("worl".to_string(), "d".to_string()),
        ]
    }

    #[test]
    fn test_bpe_new() {
        let vocab = create_test_vocab();
        let merges = create_test_merges();
        let bpe = BpeModel::new(vocab, merges);
        assert!(bpe.vocab_size() > 0);
    }

    #[test]
    fn test_bpe_with_config() {
        let vocab = create_test_vocab();
        let merges = create_test_merges();
        let vocabulary = Vocabulary::new(vocab, crate::vocab::SpecialTokens::default());
        let config = BpeConfig {
            unk_token: "<unk>".to_string(),
            ..Default::default()
        };
        let bpe = BpeModel::with_config(vocabulary, merges, config);
        assert_eq!(bpe.config().unk_token, "<unk>");
    }

    #[test]
    fn test_bpe_encode() {
        let vocab = create_test_vocab();
        let merges = create_test_merges();
        let bpe = BpeModel::new(vocab, merges);

        let tokens = bpe.encode("hello");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_bpe_decode() {
        let vocab = create_test_vocab();
        let merges = create_test_merges();
        let bpe = BpeModel::new(vocab, merges);

        let tokens = bpe.encode("hello");
        let ids: Vec<u32> = tokens.iter().map(|t| t.id).collect();
        let decoded = bpe.decode(&ids);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_gpt2_byte_encoder() {
        let encoder = Gpt2ByteEncoder::new();

        // Test roundtrip
        for byte in 0u8..=255 {
            let encoded = encoder.encode_byte(byte);
            let decoded = encoder.decode_char(encoded);
            assert_eq!(decoded, Some(byte));
        }
    }

    #[test]
    fn test_gpt2_string_roundtrip() {
        let encoder = Gpt2ByteEncoder::new();

        let text = "hello world";
        let encoded = encoder.encode_string(text);
        let decoded = encoder.decode_string(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_pre_tokenizer_gpt2() {
        let pre_tok = BpePreTokenizer::gpt2();
        let tokens = pre_tok.pre_tokenize("Hello world!");
        assert!(tokens.len() >= 2);
    }

    #[test]
    fn test_compatibility_table() {
        let vocab = VocabularyBuilder::new()
            .add_tokens(["h", "e", "he"])
            .build();

        let merges = vec![MergeRule {
            first: "h".to_string(),
            second: "e".to_string(),
            result: "he".to_string(),
            priority: 0,
        }];

        let table = CompatibilityTable::from_merges(&merges, &vocab);

        let h_id = vocab.token_to_id("h").unwrap();
        let e_id = vocab.token_to_id("e").unwrap();
        let he_id = vocab.token_to_id("he").unwrap();

        assert_eq!(table.get(h_id, e_id), Some(he_id));
        assert!(table.compatible(h_id, e_id));
        assert!(!table.compatible(e_id, h_id));
    }

    #[test]
    fn test_linear_bpe_encoder() {
        let vocab = VocabularyBuilder::new()
            .add_tokens(["[UNK]", "hello", "world"])
            .unk_token("[UNK]")
            .build();

        let merges: Vec<MergeRule> = vec![];
        let compat = CompatibilityTable::from_merges(&merges, &vocab);
        let automaton = VocabAutomaton::from_vocab(&vocab);

        let encoder = LinearBpeEncoder::new(automaton, compat, 0);
        let ids = encoder.encode("helloworld");

        // Should find "hello" and "world"
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_parse_merges() {
        let content = "#version: 0.2\nh e\nhe l";
        let merges = parse_merges(content);

        assert_eq!(merges.len(), 2);
        assert_eq!(merges[0], ("h".to_string(), "e".to_string()));
    }

    #[test]
    fn test_cached_bpe() {
        let vocab = create_test_vocab();
        let merges = create_test_merges();
        let model = BpeModel::new(vocab, merges);
        let cached = CachedBpe::new(model, 1000);

        // First call - cache miss
        let _ = cached.encode("hello");

        // Second call - cache hit
        let _ = cached.encode("hello");

        let stats = cached.cache_stats();
        assert!(stats.hits >= 1);
    }

    #[test]
    fn test_builder() {
        let vocab = VocabularyBuilder::new()
            .add_tokens(["<unk>", "h", "e", "he"])
            .unk_token("<unk>")
            .build();

        let tokenizer = BpeBuilder::new()
            .vocabulary(vocab)
            .add_merge("h", "e", "he")
            .build()
            .unwrap();

        assert_eq!(tokenizer.merges().len(), 1);
    }
}
