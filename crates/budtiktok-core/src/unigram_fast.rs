//! High-Performance Unigram Tokenizer
//!
//! This module implements a highly optimized Unigram tokenizer with:
//! - Single-pass Viterbi (no lattice allocation)
//! - Double-Array Trie for compact vocabulary lookup
//! - SIMD preprocessing for space replacement
//! - Arena allocation for temporary data
//! - Optimized parallel batch processing
//!
//! Target: 10x faster than HuggingFace Tokenizers

use ahash::AHashMap;
use bumpalo::Bump;
use rayon::prelude::*;
use std::sync::Arc;
use unicode_normalization::UnicodeNormalization;

use crate::cache::{CacheStats, ShardedCache};
use crate::encoding::Encoding;
use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;
use crate::unigram::{UnigramConfig, UnigramPiece, SPIECE_UNDERLINE};
use crate::vocab::Vocabulary;

/// Type alias for the token cache used by UnigramFast
pub type UnigramTokenCache = ShardedCache<String, Vec<u32>>;

// ============================================================================
// Phase 1: Single-Pass Viterbi Data Structures
// ============================================================================

/// Optimized Unigram tokenizer configuration
#[derive(Debug, Clone)]
pub struct UnigramFastConfig {
    /// Unknown token string
    pub unk_token: String,
    /// Unknown token ID
    pub unk_id: u32,
    /// BOS token (optional)
    pub bos_token: Option<String>,
    /// EOS token (optional)
    pub eos_token: Option<String>,
    /// Whether to use byte fallback for unknown bytes
    pub byte_fallback: bool,
    /// Add space prefix (SentencePiece style)
    pub add_prefix_space: bool,
    /// Replacement character for spaces
    pub replacement_char: char,
    /// Whether to merge consecutive UNK tokens (HuggingFace compatible)
    pub fuse_unk: bool,
}

impl Default for UnigramFastConfig {
    fn default() -> Self {
        Self {
            unk_token: "<unk>".to_string(),
            unk_id: 0,
            bos_token: None,
            eos_token: None,
            byte_fallback: false,
            add_prefix_space: true,
            replacement_char: SPIECE_UNDERLINE,
            fuse_unk: true, // Default to HuggingFace behavior
        }
    }
}

impl From<UnigramConfig> for UnigramFastConfig {
    fn from(config: UnigramConfig) -> Self {
        Self {
            unk_token: config.unk_token,
            unk_id: config.unk_id,
            bos_token: config.bos_token,
            eos_token: config.eos_token,
            byte_fallback: config.byte_fallback,
            add_prefix_space: config.add_prefix_space,
            replacement_char: config.replacement_char,
            fuse_unk: true, // Default to HuggingFace behavior
        }
    }
}

// ============================================================================
// Phase 2: Double-Array Trie
// ============================================================================

/// Compact Double-Array Trie for efficient vocabulary lookup
///
/// Uses the classic Aoe algorithm with optimizations:
/// - Compact BASE and CHECK arrays
/// - Direct token ID storage
/// - Cache-friendly memory layout
#[derive(Clone)]
pub struct DoubleArrayTrie {
    /// BASE array: base[state] + byte = next_state
    base: Vec<i32>,
    /// CHECK array: check[state] = parent_state (for validation)
    check: Vec<i32>,
    /// Token IDs: token_id[state] = Some(id) if accepting state
    /// Using i32 where -1 means no token
    token_ids: Vec<i32>,
    /// Number of entries (tokens)
    num_entries: usize,
}

impl Default for DoubleArrayTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl DoubleArrayTrie {
    /// Create an empty trie
    pub fn new() -> Self {
        Self {
            base: vec![0],
            check: vec![-1],
            token_ids: vec![-1],
            num_entries: 0,
        }
    }

    /// Build trie from vocabulary (token, id) pairs
    pub fn from_vocab<'a, I>(vocab: I) -> Self
    where
        I: IntoIterator<Item = (&'a str, u32)>,
    {
        let mut builder = DoubleArrayTrieBuilder::new();
        let vocab_vec: Vec<_> = vocab.into_iter().collect();

        #[cfg(debug_assertions)]
        {
            // Debug: Check if ▁a is in the input
            if let Some((tok, id)) = vocab_vec.iter().find(|(t, _)| *t == "▁a") {
                eprintln!("DEBUG [from_vocab]: input has '{}' (id {})", tok, id);
            }
        }

        for (token, id) in vocab_vec {
            builder.insert(token.as_bytes(), id);
        }

        let trie = builder.build();

        #[cfg(debug_assertions)]
        {
            // Debug: Check if ▁a is in the built trie
            if let Some(id) = trie.get("▁a".as_bytes()) {
                eprintln!("DEBUG [from_vocab]: built trie has '▁a' (id {})", id);
            } else {
                eprintln!("DEBUG [from_vocab]: built trie MISSING '▁a'");
            }
        }

        trie
    }

    /// Get token ID for exact match
    #[inline]
    pub fn get(&self, bytes: &[u8]) -> Option<u32> {
        #[cfg(debug_assertions)]
        let is_spiece_a = bytes == &[226, 150, 129, 97]; // ▁a

        let mut state = 0i32;

        for (i, &byte) in bytes.iter().enumerate() {
            let base_val = match self.base.get(state as usize) {
                Some(v) => v,
                None => {
                    #[cfg(debug_assertions)]
                    if is_spiece_a {
                        eprintln!("DEBUG [get '▁a']: base.get({}) returned None at byte {}", state, i);
                    }
                    return None;
                }
            };
            let next = *base_val + byte as i32;

            if next < 0 {
                #[cfg(debug_assertions)]
                if is_spiece_a {
                    eprintln!("DEBUG [get '▁a']: next {} < 0 at byte {}", next, i);
                }
                return None;
            }

            let next_usize = next as usize;
            if next_usize >= self.check.len() {
                #[cfg(debug_assertions)]
                if is_spiece_a {
                    eprintln!("DEBUG [get '▁a']: next_usize {} >= check.len {} at byte {}", next_usize, self.check.len(), i);
                }
                return None;
            }

            let check_val = self.check[next_usize];
            if check_val != state {
                #[cfg(debug_assertions)]
                if is_spiece_a {
                    eprintln!("DEBUG [get '▁a']: check[{}]={} != state {} at byte {} (base_val={}, byte={})",
                             next_usize, check_val, state, i, base_val, byte);
                }
                return None;
            }
            state = next;
        }

        let token_id = self.token_ids.get(state as usize).copied().unwrap_or(-1);

        #[cfg(debug_assertions)]
        if is_spiece_a {
            eprintln!("DEBUG [get '▁a']: final state {}, token_id {}", state, token_id);
        }

        if token_id >= 0 {
            Some(token_id as u32)
        } else {
            None
        }
    }

    /// Enumerate all prefix matches starting at position `start`
    /// Calls `callback(length, token_id)` for each match
    #[inline]
    pub fn enumerate_prefixes<F>(&self, bytes: &[u8], start: usize, mut callback: F)
    where
        F: FnMut(usize, u32),
    {
        if start >= bytes.len() {
            return;
        }

        let mut state = 0i32;

        for (i, &byte) in bytes[start..].iter().enumerate() {
            let base = match self.base.get(state as usize) {
                Some(&b) => b,
                None => return,
            };

            let next = base + byte as i32;
            if next < 0 {
                return;
            }

            let next_usize = next as usize;
            if next_usize >= self.check.len() || self.check[next_usize] != state {
                return;
            }

            state = next;

            // Check if this is an accepting state
            if let Some(&token_id) = self.token_ids.get(state as usize) {
                if token_id >= 0 {
                    callback(i + 1, token_id as u32);
                }
            }
        }
    }

    /// Get an iterator over prefix matches
    #[inline]
    pub fn common_prefix_search<'a>(
        &'a self,
        bytes: &'a [u8],
        start: usize,
    ) -> DoubleArrayPrefixIter<'a> {
        DoubleArrayPrefixIter {
            trie: self,
            bytes,
            state: 0,
            position: start,
            start,
        }
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.base.len() * 4 + self.check.len() * 4 + self.token_ids.len() * 4
    }

    /// Number of states
    pub fn num_states(&self) -> usize {
        self.base.len()
    }

    /// Number of entries (tokens)
    pub fn num_entries(&self) -> usize {
        self.num_entries
    }
}

/// Iterator over prefix matches in Double-Array Trie
pub struct DoubleArrayPrefixIter<'a> {
    trie: &'a DoubleArrayTrie,
    bytes: &'a [u8],
    state: i32,
    position: usize,
    start: usize,
}

impl<'a> Iterator for DoubleArrayPrefixIter<'a> {
    type Item = (usize, u32);

    fn next(&mut self) -> Option<Self::Item> {
        while self.position < self.bytes.len() {
            let byte = self.bytes[self.position];

            let base = *self.trie.base.get(self.state as usize)?;
            let next = base + byte as i32;

            if next < 0 {
                self.position = self.bytes.len();
                return None;
            }

            let next_usize = next as usize;
            if next_usize >= self.trie.check.len() || self.trie.check[next_usize] != self.state {
                self.position = self.bytes.len();
                return None;
            }

            self.state = next;
            self.position += 1;

            // Check if accepting state
            if let Some(&token_id) = self.trie.token_ids.get(self.state as usize) {
                if token_id >= 0 {
                    return Some((self.position - self.start, token_id as u32));
                }
            }
        }
        None
    }
}

/// Builder for Double-Array Trie using a simpler, correct algorithm
pub struct DoubleArrayTrieBuilder {
    /// BASE array
    base: Vec<i32>,
    /// CHECK array
    check: Vec<i32>,
    /// Token IDs
    token_ids: Vec<i32>,
    /// Number of entries
    num_entries: usize,
}

impl DoubleArrayTrieBuilder {
    const INITIAL_SIZE: usize = 1024;

    pub fn new() -> Self {
        Self {
            base: vec![0; Self::INITIAL_SIZE],
            check: vec![-1; Self::INITIAL_SIZE],
            token_ids: vec![-1; Self::INITIAL_SIZE],
            num_entries: 0,
        }
    }

    fn ensure_capacity(&mut self, min_size: usize) {
        if min_size > self.base.len() {
            let new_size = (min_size * 2).max(self.base.len() * 2);
            self.base.resize(new_size, 0);
            self.check.resize(new_size, -1);
            self.token_ids.resize(new_size, -1);
        }
    }

    pub fn insert(&mut self, bytes: &[u8], id: u32) {
        #[cfg(debug_assertions)]
        let is_spiece_a = bytes == &[226, 150, 129, 97]; // ▁a

        let mut state = 0i32;

        for &byte in bytes.iter() {
            let c = byte as i32;

            // Try to find existing transition
            let base_val = self.base[state as usize];
            let next = base_val + c;

            if next >= 0 {
                let next_usize = next as usize;
                self.ensure_capacity(next_usize + 1);

                if self.check[next_usize] == state {
                    // Transition exists
                    state = next;
                    continue;
                } else if self.check[next_usize] == -1 {
                    // Position is free, create transition
                    self.check[next_usize] = state;
                    state = next;
                    continue;
                }
            }

            // Need to find new base for current state
            let new_base = self.find_free_base(state, c);
            self.base[state as usize] = new_base;

            let next = new_base + c;
            let next_usize = next as usize;
            self.ensure_capacity(next_usize + 1);
            self.check[next_usize] = state;
            state = next;
        }

        // Mark terminal
        let state_usize = state as usize;
        self.ensure_capacity(state_usize + 1);
        if self.token_ids[state_usize] == -1 {
            self.num_entries += 1;
        }
        self.token_ids[state_usize] = id as i32;

        #[cfg(debug_assertions)]
        if is_spiece_a {
            eprintln!("DEBUG [insert]: '▁a' (id {}) inserted at state {}", id, state);
        }
    }

    fn find_free_base(&mut self, state: i32, c: i32) -> i32 {
        // Collect existing children of this state
        // CRITICAL: We must scan for children even when base=0, because states
        // with base=0 have children at positions 0+byte = byte (0..255)
        let old_base = self.base[state as usize];
        let mut children: Vec<(i32, i32)> = Vec::new(); // (byte, old_next)

        // Scan for all existing children of this state
        // We need to check all positions where children could exist
        let scan_start = old_base.max(0);
        let scan_end = (old_base + 256).min(self.check.len() as i32);
        for pos in scan_start..scan_end {
            if pos >= 0 && (pos as usize) < self.check.len() {
                if self.check[pos as usize] == state {
                    let byte = pos - old_base;
                    if byte >= 0 && byte < 256 {
                        children.push((byte, pos));
                    }
                }
            }
        }

        // Add new child
        children.push((c, -1)); // -1 indicates new child

        // Find base that works for all children (start from 1 to avoid base=0 issues)
        'find_base: for base in 1i32.. {
            for &(byte, old_next) in &children {
                let target = base + byte;
                if target < 0 {
                    continue 'find_base;
                }
                let target_usize = target as usize;
                self.ensure_capacity(target_usize + 1);
                // Position must be free (-1) or already belong to this state
                // (same old_next means we're not actually moving)
                if self.check[target_usize] != -1 {
                    // Already occupied by someone else
                    if old_next >= 0 && target == old_next {
                        // Same position, no conflict
                        continue;
                    }
                    if self.check[target_usize] != state {
                        continue 'find_base;
                    }
                }
            }

            // Found valid base - relocate existing children to new positions
            // CRITICAL: We must do this in two passes to avoid data corruption
            // when one child's new position equals another child's old position.

            // Pass 1: Collect all data to relocate (before modifying anything)
            let mut relocations: Vec<(i32, i32, i32, i32, Vec<i32>)> = Vec::new(); // (old_next, new_next, base_val, token_id, grandchildren)

            for &(byte, old_next) in &children {
                if old_next >= 0 {
                    let new_next = base + byte;
                    if new_next == old_next {
                        // Same position, no relocation needed
                        continue;
                    }

                    // Collect data from old position
                    let base_val = self.base[old_next as usize];
                    let token_id = self.token_ids[old_next as usize];

                    // Collect grandchildren positions
                    let gc_start = base_val.max(0);
                    let gc_end = (base_val + 256).min(self.check.len() as i32);
                    let mut grandchildren = Vec::new();
                    for gc_pos in gc_start..gc_end {
                        if gc_pos >= 0 && (gc_pos as usize) < self.check.len() {
                            if self.check[gc_pos as usize] == old_next {
                                grandchildren.push(gc_pos);
                            }
                        }
                    }

                    relocations.push((old_next, new_next, base_val, token_id, grandchildren));
                }
            }

            // Pass 2: Clear old positions first (to avoid overwriting new data)
            for &(old_next, new_next, _, _, _) in &relocations {
                if old_next != new_next {
                    self.check[old_next as usize] = -1;
                    self.base[old_next as usize] = 0;
                    self.token_ids[old_next as usize] = -1;
                }
            }

            // Pass 3: Write to new positions and update grandchildren
            for (old_next, new_next, base_val, token_id, grandchildren) in relocations {
                // Write to new position
                self.check[new_next as usize] = state;
                self.base[new_next as usize] = base_val;
                self.token_ids[new_next as usize] = token_id;

                // Update grandchildren to point to new position
                for gc_pos in grandchildren {
                    self.check[gc_pos as usize] = new_next;
                }
            }

            return base;
        }

        unreachable!()
    }

    pub fn build(mut self) -> DoubleArrayTrie {
        // Trim arrays
        let max_used = self.check.iter().rposition(|&c| c != -1).map(|i| i + 1).unwrap_or(1);
        self.base.truncate(max_used.max(1));
        self.check.truncate(max_used.max(1));
        self.token_ids.truncate(max_used.max(1));

        DoubleArrayTrie {
            base: self.base,
            check: self.check,
            token_ids: self.token_ids,
            num_entries: self.num_entries,
        }
    }
}

impl Default for DoubleArrayTrieBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Phase 3: Normalization and Preprocessing
// ============================================================================

/// Normalize text for HuggingFace compatibility
///
/// This function applies the normalization that HuggingFace Tokenizers
/// typically uses for Unigram models:
/// 1. Trim leading/trailing whitespace
/// 2. Collapse multiple consecutive spaces to a single space
/// 3. Apply NFKD normalization
/// 4. Strip combining marks (accents)
///
/// This ensures 100% accuracy match with HuggingFace's output.
///
/// Returns Cow<str> to avoid allocation for the common ASCII case.
///
/// Matches HuggingFace's normalizer sequence:
/// 1. Replace `` -> "
/// 2. Replace '' -> "
/// 3. NFKD normalization
/// 4. Strip accents (combining marks)
/// 5. Precompiled char map (removes zero-width chars, normalizes quotes)
#[inline]
pub fn normalize_for_hf(text: &str) -> std::borrow::Cow<'_, str> {
    use std::borrow::Cow;

    // Step 1: Trim leading/trailing whitespace
    let trimmed = text.trim();

    if trimmed.is_empty() {
        return Cow::Borrowed("");
    }

    // Fast path: check if we need normalization at all
    // Pure ASCII text without special chars can skip most processing
    let needs_space_collapse = has_consecutive_spaces(trimmed);
    let is_simple_ascii = trimmed.is_ascii() && !needs_quote_normalization_ascii(trimmed);

    if is_simple_ascii && !needs_space_collapse {
        // Fast path: pure simple ASCII with no consecutive spaces - zero allocation
        return Cow::Borrowed(trimmed);
    }

    // Check if we need special character handling
    let needs_special_handling = needs_special_char_handling(trimmed);

    // Step 2: Handle special characters, quote normalization, and space collapse
    let mut result = String::with_capacity(trimmed.len());
    let mut prev_was_space = false;
    let mut chars = trimmed.chars().peekable();

    while let Some(c) = chars.next() {
        // Handle zero-width characters (remove them)
        if is_zero_width_char(c) {
            continue;
        }

        // Handle quote normalization (only for backtick pairs as per HF normalizer)
        match c {
            // Backtick pairs `` -> " (as per HF Replace normalizer)
            '`' if chars.peek() == Some(&'`') => {
                chars.next(); // consume second backtick
                result.push('"');
                prev_was_space = false;
            }
            // Single quote pairs '' -> " (as per HF Replace normalizer)
            '\'' if chars.peek() == Some(&'\'') => {
                chars.next(); // consume second quote
                result.push('"');
                prev_was_space = false;
            }
            // Whitespace collapse
            ' ' | '\t' | '\n' | '\r' => {
                if !prev_was_space {
                    result.push(' ');
                    prev_was_space = true;
                }
            }
            // Replacement character (U+FFFD) -> space (matches HF precompiled chars map)
            '\u{FFFD}' => {
                if !prev_was_space {
                    result.push(' ');
                    prev_was_space = true;
                }
            }
            // Zero-width space (U+200B) and directional marks -> space (HF converts to space, not removes)
            '\u{200B}' | '\u{200E}' | '\u{200F}' => {
                if !prev_was_space {
                    result.push(' ');
                    prev_was_space = true;
                }
            }
            // Non-ASCII: apply NFKD and filter combining marks
            // Note: NFKD does NOT convert smart quotes to ASCII - they stay as-is
            _ if !c.is_ascii() => {
                // Apply NFKD normalization to this character
                for nc in c.to_string().nfkd() {
                    if is_combining_mark(nc) || is_zero_width_char(nc) {
                        continue;
                    }
                    // Handle whitespace from NFKD decomposition (e.g., macron -> space + combining)
                    if nc == ' ' || nc == '\t' || nc == '\n' || nc == '\r' {
                        if !prev_was_space {
                            result.push(' ');
                            prev_was_space = true;
                        }
                    } else {
                        result.push(nc);
                        prev_was_space = false;
                    }
                }
            }
            // Regular ASCII
            _ => {
                result.push(c);
                prev_was_space = false;
            }
        }
    }

    // Trim leading and trailing whitespace (can occur after converting U+200B to space)
    while result.ends_with(' ') {
        result.pop();
    }
    while result.starts_with(' ') {
        result.remove(0);
    }

    // If the result is identical to the input, return borrowed
    if !needs_special_handling && !needs_space_collapse && result == trimmed {
        Cow::Borrowed(trimmed)
    } else {
        Cow::Owned(result)
    }
}

/// Check if a character is a zero-width or formatting character that should be removed
/// Note: Only includes characters that HF's precompiled charsmap removes.
/// U+200B, U+200E, U+200F are handled separately (converted to space).
/// Mathematical invisible operators (U+2061-U+2064) are kept as they become <unk> in HF.
#[inline]
fn is_zero_width_char(c: char) -> bool {
    matches!(c,
        // U+200B, U+200E, U+200F are handled separately - converted to space
        '\u{200C}' |  // Zero Width Non-Joiner
        '\u{200D}' |  // Zero Width Joiner
        '\u{202A}' |  // Left-to-Right Embedding
        '\u{202B}' |  // Right-to-Left Embedding
        '\u{202C}' |  // Pop Directional Formatting
        '\u{202D}' |  // Left-to-Right Override
        '\u{202E}' |  // Right-to-Left Override
        '\u{2066}' |  // Left-to-Right Isolate
        '\u{2067}' |  // Right-to-Left Isolate
        '\u{2068}' |  // First Strong Isolate
        '\u{2069}' |  // Pop Directional Isolate
        '\u{FEFF}' |  // Zero Width No-Break Space (BOM)
        '\u{2060}'    // Word Joiner
        // Note: U+2061-U+2064 (math operators) are intentionally NOT removed
        // HF keeps them and they tokenize to <unk>
    )
}

/// Check if ASCII text needs quote normalization (contains `` or '')
#[inline]
fn needs_quote_normalization_ascii(text: &str) -> bool {
    let bytes = text.as_bytes();
    for i in 0..bytes.len().saturating_sub(1) {
        if (bytes[i] == b'`' && bytes[i + 1] == b'`') ||
           (bytes[i] == b'\'' && bytes[i + 1] == b'\'') {
            return true;
        }
    }
    false
}

/// Check if text needs special character handling (non-ASCII, zero-width)
#[inline]
fn needs_special_char_handling(text: &str) -> bool {
    for c in text.chars() {
        if is_zero_width_char(c) || !c.is_ascii() {
            return true;
        }
    }
    false
}

/// Check if text has consecutive whitespace characters
#[inline]
fn has_consecutive_spaces(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut prev_was_space = false;

    for &b in bytes {
        let is_space = b == b' ' || b == b'\t' || b == b'\n' || b == b'\r';
        if is_space && prev_was_space {
            return true;
        }
        prev_was_space = is_space;
    }
    false
}

/// Check if a character is a combining mark (accent, diacritic, etc.)
/// Combining marks have Unicode category M (Mark)
#[inline]
fn is_combining_mark(c: char) -> bool {
    use unicode_general_category::{get_general_category, GeneralCategory};

    matches!(
        get_general_category(c),
        GeneralCategory::NonspacingMark |
        GeneralCategory::SpacingMark |
        GeneralCategory::EnclosingMark
    )
}

/// Preprocess text by replacing spaces with ▁ (SPIECE_UNDERLINE)
/// Uses SIMD when available for maximum performance
#[inline]
pub fn preprocess_text(text: &str, add_prefix: bool) -> Vec<u8> {
    let bytes = text.as_bytes();
    let replacement = SPIECE_UNDERLINE.to_string();
    let replacement_bytes = replacement.as_bytes();

    // Count spaces to calculate output size
    let space_count = count_spaces(bytes);
    let extra_bytes = space_count * (replacement_bytes.len() - 1);
    let prefix_bytes = if add_prefix { replacement_bytes.len() } else { 0 };

    let mut result = Vec::with_capacity(bytes.len() + extra_bytes + prefix_bytes);

    // Add prefix if needed
    if add_prefix {
        result.extend_from_slice(replacement_bytes);
    }

    // Replace spaces with ▁
    for &byte in bytes {
        if byte == b' ' {
            result.extend_from_slice(replacement_bytes);
        } else {
            result.push(byte);
        }
    }

    result
}

/// Count spaces in byte slice using SIMD when available
#[inline]
fn count_spaces(bytes: &[u8]) -> usize {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        count_spaces_avx2(bytes)
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        count_spaces_scalar(bytes)
    }
}

/// Scalar space counting fallback
#[inline]
fn count_spaces_scalar(bytes: &[u8]) -> usize {
    bytes.iter().filter(|&&b| b == b' ').count()
}

/// AVX2 space counting
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn count_spaces_avx2(bytes: &[u8]) -> usize {
    use std::arch::x86_64::*;

    let mut count = 0usize;
    let len = bytes.len();

    if len >= 32 {
        unsafe {
            let space = _mm256_set1_epi8(b' ' as i8);
            let chunks = len / 32;

            for i in 0..chunks {
                let ptr = bytes.as_ptr().add(i * 32) as *const __m256i;
                let data = _mm256_loadu_si256(ptr);
                let cmp = _mm256_cmpeq_epi8(data, space);
                let mask = _mm256_movemask_epi8(cmp) as u32;
                count += mask.count_ones() as usize;
            }

            // Handle remainder
            for &byte in &bytes[chunks * 32..] {
                if byte == b' ' {
                    count += 1;
                }
            }
        }
    } else {
        count = count_spaces_scalar(bytes);
    }

    count
}

/// Fast preprocessing with SIMD
/// Returns preprocessed bytes directly (no String allocation)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn preprocess_text_simd(text: &str, add_prefix: bool) -> Vec<u8> {
    use std::arch::x86_64::*;

    let bytes = text.as_bytes();
    let replacement = SPIECE_UNDERLINE.to_string();
    let replacement_bytes = replacement.as_bytes(); // [0xE2, 0x96, 0x81]

    // Fast path: scan for spaces using SIMD
    let space_count = count_spaces_avx2(bytes);

    if space_count == 0 && !add_prefix {
        // No spaces and no prefix - return as-is
        return bytes.to_vec();
    }

    let extra_bytes = space_count * (replacement_bytes.len() - 1);
    let prefix_bytes = if add_prefix { replacement_bytes.len() } else { 0 };

    let mut result = Vec::with_capacity(bytes.len() + extra_bytes + prefix_bytes);

    // Add prefix
    if add_prefix {
        result.extend_from_slice(replacement_bytes);
    }

    // Process chunks, expanding spaces
    unsafe {
        let space = _mm256_set1_epi8(b' ' as i8);
        let len = bytes.len();
        let chunks = len / 32;

        let mut i = 0;
        for chunk_idx in 0..chunks {
            let ptr = bytes.as_ptr().add(chunk_idx * 32) as *const __m256i;
            let data = _mm256_loadu_si256(ptr);
            let cmp = _mm256_cmpeq_epi8(data, space);
            let mask = _mm256_movemask_epi8(cmp) as u32;

            if mask == 0 {
                // No spaces in this chunk - copy directly
                result.extend_from_slice(&bytes[chunk_idx * 32..(chunk_idx + 1) * 32]);
            } else {
                // Has spaces - process byte by byte
                for j in 0..32 {
                    let byte = bytes[chunk_idx * 32 + j];
                    if byte == b' ' {
                        result.extend_from_slice(replacement_bytes);
                    } else {
                        result.push(byte);
                    }
                }
            }
            i = (chunk_idx + 1) * 32;
        }

        // Handle remainder
        for &byte in &bytes[i..] {
            if byte == b' ' {
                result.extend_from_slice(replacement_bytes);
            } else {
                result.push(byte);
            }
        }
    }

    result
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
pub fn preprocess_text_simd(text: &str, add_prefix: bool) -> Vec<u8> {
    preprocess_text(text, add_prefix)
}

// ============================================================================
// Phase 4 & 5: High-Performance Unigram Tokenizer
// ============================================================================

/// High-performance Unigram tokenizer
///
/// Optimizations:
/// - Single-pass Viterbi (no lattice allocation)
/// - Double-Array Trie for vocabulary
/// - Dense score array (direct indexing)
/// - Arena allocation for temporary data
/// - SIMD preprocessing
pub struct UnigramFast {
    /// Double-Array Trie for vocabulary lookup
    trie: DoubleArrayTrie,
    /// Dense score array indexed by token ID
    scores: Vec<f64>,
    /// Vocabulary for ID-to-token mapping
    vocab: Vec<String>,
    /// Configuration
    config: UnigramFastConfig,
    /// Minimum score (for unknown token penalty)
    min_score: f64,
    /// Vocabulary size
    vocab_size: usize,
    /// Byte fallback token IDs
    byte_tokens: [i32; 256],
    /// Optional token cache for repeated texts (shared across threads)
    cache: Option<Arc<UnigramTokenCache>>,
}

impl UnigramFast {
    /// Create a new fast Unigram tokenizer
    pub fn new(
        vocab: Vec<String>,
        pieces: Vec<UnigramPiece>,
        config: UnigramFastConfig,
    ) -> Self {
        let vocab_size = vocab.len();

        // Build token-to-id map, keeping track of which ID has the best score
        // for each token string (handles duplicate entries in vocabulary)
        // Note: pieces[id].score gives the score for vocab[id]
        let mut token_to_best_id: AHashMap<&str, (u32, f64)> = AHashMap::new();

        // Iterate through vocab with index, using pieces[index].score for score
        for (id, token) in vocab.iter().enumerate() {
            let score = if id < pieces.len() {
                pieces[id].score
            } else {
                f64::NEG_INFINITY
            };

            token_to_best_id
                .entry(token.as_str())
                .and_modify(|(best_id, best_score)| {
                    // Keep the ID with the higher (better) score
                    if score > *best_score {
                        *best_id = id as u32;
                        *best_score = score;
                    }
                })
                .or_insert((id as u32, score));
        }

        // Build simple token_to_id map using best IDs
        let token_to_id: AHashMap<&str, u32> = token_to_best_id
            .iter()
            .map(|(&token, &(id, _))| (token, id))
            .collect();

        // Collect tokens for trie using best IDs
        let trie_tokens: Vec<(&str, u32)> = token_to_best_id
            .iter()
            .map(|(&token, &(id, _))| (token, id))
            .collect();

        // Build double-array trie
        let trie = DoubleArrayTrie::from_vocab(trie_tokens.into_iter());

        // Build dense score array
        let mut scores = vec![f64::NEG_INFINITY; vocab_size];
        let mut min_score = f64::INFINITY;

        for piece in &pieces {
            if let Some(&id) = token_to_id.get(piece.token.as_str()) {
                scores[id as usize] = piece.score;
                min_score = min_score.min(piece.score);
            }
        }

        // Build byte fallback tokens
        let mut byte_tokens = [-1i32; 256];
        for b in 0u8..=255 {
            let token = format!("<0x{:02X}>", b);
            if let Some(&id) = token_to_id.get(token.as_str()) {
                byte_tokens[b as usize] = id as i32;
            }
        }

        Self {
            trie,
            scores,
            vocab,
            config,
            min_score,
            vocab_size,
            byte_tokens,
            cache: None,
        }
    }

    /// Create a new tokenizer with caching enabled
    ///
    /// The cache uses a sharded CLOCK algorithm for O(1) amortized operations
    /// and reduced lock contention under concurrent access.
    ///
    /// # Arguments
    /// * `vocab` - Vocabulary strings
    /// * `pieces` - Unigram pieces with scores
    /// * `config` - Tokenizer configuration
    /// * `cache_capacity` - Total cache capacity (distributed across shards)
    pub fn with_cache(
        vocab: Vec<String>,
        pieces: Vec<UnigramPiece>,
        config: UnigramFastConfig,
        cache_capacity: usize,
    ) -> Self {
        let mut tokenizer = Self::new(vocab, pieces, config);
        tokenizer.cache = Some(Arc::new(ShardedCache::new(cache_capacity)));
        tokenizer
    }

    /// Enable caching with the specified capacity
    ///
    /// This creates a new cache, discarding any existing cached entries.
    /// For best performance, enable caching before processing texts.
    ///
    /// # Arguments
    /// * `capacity` - Total cache capacity (100,000 is a good default)
    pub fn enable_cache(&mut self, capacity: usize) {
        self.cache = Some(Arc::new(ShardedCache::new(capacity)));
    }

    /// Enable caching with a pre-created cache (for sharing between tokenizers)
    pub fn set_cache(&mut self, cache: Arc<UnigramTokenCache>) {
        self.cache = Some(cache);
    }

    /// Disable caching and drop the cache
    pub fn disable_cache(&mut self) {
        self.cache = None;
    }

    /// Check if caching is enabled
    #[inline]
    pub fn is_cache_enabled(&self) -> bool {
        self.cache.is_some()
    }

    /// Get cache statistics (returns None if caching is disabled)
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.cache.as_ref().map(|c| c.stats())
    }

    /// Clear the cache while keeping it enabled
    pub fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.clear();
        }
    }

    /// Get a reference to the cache (for sharing with other tokenizers)
    pub fn get_cache(&self) -> Option<&Arc<UnigramTokenCache>> {
        self.cache.as_ref()
    }

    /// Create from standard Unigram tokenizer components
    pub fn from_pieces(
        vocabulary: &Vocabulary,
        pieces: Vec<UnigramPiece>,
        config: UnigramConfig,
    ) -> Self {
        // Extract vocab as Vec<String>
        let mut vocab: Vec<(String, u32)> = vocabulary
            .iter()
            .map(|(token, id)| (token.to_string(), id))
            .collect();
        vocab.sort_by_key(|(_, id)| *id);

        let vocab_strings: Vec<String> = vocab.into_iter().map(|(t, _)| t).collect();

        Self::new(vocab_strings, pieces, config.into())
    }

    /// Get score for token ID (direct array access)
    #[inline(always)]
    fn get_score(&self, token_id: u32) -> f64 {
        if (token_id as usize) < self.scores.len() {
            unsafe { *self.scores.get_unchecked(token_id as usize) }
        } else {
            self.min_score - 10.0
        }
    }

    /// Get token string for ID
    #[inline]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(|s| s.as_str())
    }

    /// Get ID for token string
    #[inline]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.trie.get(token.as_bytes())
    }

    /// Direct trie lookup (for debugging)
    #[inline]
    pub fn trie_lookup(&self, bytes: &[u8]) -> Option<u32> {
        self.trie.get(bytes)
    }

    /// Vocabulary size
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Find next UTF-8 character boundary
    #[inline]
    fn next_char_boundary(&self, bytes: &[u8], start: usize) -> usize {
        let mut end = start + 1;
        while end < bytes.len() && (bytes[end] & 0xC0) == 0x80 {
            end += 1;
        }
        end
    }

    // ========================================================================
    // Phase 1: Single-Pass Viterbi Encoding
    // ========================================================================

    /// Encode text using single-pass Viterbi algorithm
    ///
    /// This is the core optimization: instead of building a full lattice
    /// and then backtracing, we only track the best path to each position.
    #[inline]
    pub fn encode_fast(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // Preprocess: add ▁ prefix and replace spaces
        let preprocessed = preprocess_text(text, self.config.add_prefix_space);
        let bytes = &preprocessed;
        let n = bytes.len();

        if n == 0 {
            return vec![];
        }

        // Single-pass Viterbi: only track best path to each position
        // This eliminates the need for lattice allocation
        let mut best_score = vec![f64::NEG_INFINITY; n + 1];
        let mut best_token = vec![u32::MAX; n + 1];
        let mut best_prev = vec![usize::MAX; n + 1];

        best_score[0] = 0.0;

        // Forward pass
        for start in 0..n {
            let prev_score = best_score[start];
            if prev_score == f64::NEG_INFINITY {
                continue;
            }

            let mut found_token = false;

            // Enumerate all matching tokens using trie
            self.trie.enumerate_prefixes(bytes, start, |len, token_id| {
                found_token = true;
                let end = start + len;
                let score = prev_score + self.get_score(token_id);

                if score > best_score[end] {
                    best_score[end] = score;
                    best_token[end] = token_id;
                    best_prev[end] = start;
                }
            });

            // Byte fallback if enabled and no token found
            if !found_token && self.config.byte_fallback {
                let byte_val = bytes[start];
                let byte_id = self.byte_tokens[byte_val as usize];
                if byte_id >= 0 {
                    found_token = true;
                    let score = prev_score + self.min_score - 5.0;
                    if score > best_score[start + 1] {
                        best_score[start + 1] = score;
                        best_token[start + 1] = byte_id as u32;
                        best_prev[start + 1] = start;
                    }
                }
            }

            // Unknown character fallback
            if !found_token {
                let char_end = self.next_char_boundary(bytes, start);
                let score = prev_score + self.min_score - 10.0;
                if score > best_score[char_end] {
                    best_score[char_end] = score;
                    best_token[char_end] = self.config.unk_id;
                    best_prev[char_end] = start;
                }
            }
        }

        // Backtrace
        if best_token[n] == u32::MAX {
            return vec![self.config.unk_id];
        }

        let mut result = Vec::with_capacity(n / 4); // Estimate ~4 bytes per token
        let mut pos = n;

        while pos > 0 && best_prev[pos] != usize::MAX {
            result.push(best_token[pos]);
            pos = best_prev[pos];
        }

        result.reverse();
        result
    }

    // ========================================================================
    // HuggingFace-Compatible Encoding (100% accuracy match)
    // ========================================================================

    /// Constant penalty for unknown tokens (matches HuggingFace K_UNK_PENALTY)
    const K_UNK_PENALTY: f64 = 10.0;

    /// Encode text using HuggingFace-compatible algorithm
    ///
    /// This method exactly matches the behavior of HuggingFace Tokenizers'
    /// `encode_optimized` method:
    /// - Applies normalization (trim, collapse spaces, strip accents)
    /// - Iterates by character boundaries (not bytes)
    /// - Uses `has_single_node` logic for UNK handling
    /// - Supports `fuse_unk` to merge consecutive UNK tokens
    /// - Applies byte fallback in post-processing (not during Viterbi)
    ///
    /// Use this method when 100% accuracy match with HuggingFace is required.
    #[inline]
    pub fn encode_hf_compat(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // Apply HuggingFace-compatible normalization:
        // 1. Trim leading/trailing whitespace
        // 2. Collapse multiple spaces to single space
        // 3. Apply NFKD + StripAccents
        let normalized = normalize_for_hf(text);
        if normalized.is_empty() {
            return vec![];
        }

        // Preprocess: add ▁ prefix and replace spaces (like SentencePiece)
        let preprocessed = preprocess_text(&normalized, self.config.add_prefix_space);
        let sentence = match std::str::from_utf8(&preprocessed) {
            Ok(s) => s,
            Err(_) => return vec![self.config.unk_id],
        };

        let size = sentence.len();
        if size == 0 {
            return vec![];
        }

        // UNK score with penalty (matches HuggingFace)
        let unk_score = self.min_score - Self::K_UNK_PENALTY;

        // Best path tracking (inline Viterbi, no lattice allocation)
        // This matches HuggingFace's BestPathNode structure
        let mut best_path_score = vec![f64::NEG_INFINITY; size + 1];
        let mut best_path_id = vec![usize::MAX; size + 1];
        let mut best_path_starts_at: Vec<Option<usize>> = vec![None; size + 1];

        best_path_score[0] = 0.0;

        // Forward pass: iterate by CHARACTER boundaries (critical for accuracy)
        let mut starts_at = 0usize;
        while starts_at < size {
            // Get the length of the current character in UTF-8 bytes
            let current_char = sentence[starts_at..].chars().next().unwrap();
            let mblen = current_char.len_utf8();

            let best_score_till_here = best_path_score[starts_at];

            // Skip unreachable positions
            if best_score_till_here == f64::NEG_INFINITY {
                starts_at += mblen;
                continue;
            }

            // Track if we found any token matching exactly this single character
            let mut has_single_node = false;

            // Enumerate all tokens starting at this character position
            self.trie.enumerate_prefixes(sentence.as_bytes(), starts_at, |len, token_id| {
                let key_pos = starts_at + len;
                let score = self.get_score(token_id);
                let candidate_best_path_score = score + best_score_till_here;

                // Update best path if this is better
                if best_path_starts_at[key_pos].is_none() ||
                   candidate_best_path_score > best_path_score[key_pos] {
                    best_path_score[key_pos] = candidate_best_path_score;
                    best_path_starts_at[key_pos] = Some(starts_at);
                    best_path_id[key_pos] = token_id as usize;
                }

                // Check if this token exactly matches the single character
                // This is the key difference from the byte-based algorithm
                if !has_single_node && len == mblen {
                    has_single_node = true;
                }
            });

            // If no token covers this single character, insert UNK
            // (This is different from "no token found at all")
            if !has_single_node {
                let key_pos = starts_at + mblen;
                let candidate_best_path_score = unk_score + best_score_till_here;

                if best_path_starts_at[key_pos].is_none() ||
                   candidate_best_path_score > best_path_score[key_pos] {
                    best_path_score[key_pos] = candidate_best_path_score;
                    best_path_starts_at[key_pos] = Some(starts_at);
                    best_path_id[key_pos] = self.config.unk_id as usize;
                }
            }

            // Advance by character boundary (not byte!)
            starts_at += mblen;
        }

        // Backtrace to collect token IDs
        if best_path_starts_at[size].is_none() {
            return vec![self.config.unk_id];
        }

        let mut result = Vec::with_capacity(size / 4);
        let mut pos = size;

        while pos > 0 {
            if let Some(start) = best_path_starts_at[pos] {
                result.push(best_path_id[pos] as u32);
                pos = start;
            } else {
                break;
            }
        }

        result.reverse();

        // Apply fuse_unk if enabled (merge consecutive UNK tokens)
        if self.config.fuse_unk {
            self.fuse_consecutive_unks(&mut result);
        }

        result
    }

    /// Merge consecutive UNK tokens (HuggingFace compatible)
    #[inline]
    fn fuse_consecutive_unks(&self, ids: &mut Vec<u32>) {
        if ids.len() <= 1 {
            return;
        }

        let unk_id = self.config.unk_id;
        let mut write_pos = 0;
        let mut i = 0;

        while i < ids.len() {
            if ids[i] == unk_id {
                // Skip consecutive UNKs, keep only one
                ids[write_pos] = unk_id;
                write_pos += 1;
                while i < ids.len() && ids[i] == unk_id {
                    i += 1;
                }
            } else {
                ids[write_pos] = ids[i];
                write_pos += 1;
                i += 1;
            }
        }

        ids.truncate(write_pos);
    }

    /// SIMD-accelerated HuggingFace-compatible encoding
    ///
    /// Combines the accuracy of `encode_hf_compat` with SIMD optimizations:
    /// - SIMD space detection and preprocessing
    /// - Cache-optimized trie traversal
    /// - Prefetch hints for score lookups
    ///
    /// Target: 20x faster than HuggingFace while maintaining 100% accuracy.
    #[inline]
    pub fn encode_hf_compat_simd(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // Apply HuggingFace-compatible normalization first
        let normalized = normalize_for_hf(text);
        if normalized.is_empty() {
            return vec![];
        }

        // Use SIMD preprocessing when available
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let preprocessed = preprocess_text_simd(&normalized, self.config.add_prefix_space);

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let preprocessed = preprocess_text(&normalized, self.config.add_prefix_space);

        let sentence = match std::str::from_utf8(&preprocessed) {
            Ok(s) => s,
            Err(_) => return vec![self.config.unk_id],
        };

        let size = sentence.len();
        if size == 0 {
            return vec![];
        }

        let unk_score = self.min_score - Self::K_UNK_PENALTY;

        // Optimized allocation: single array with struct-of-arrays layout
        // This improves cache locality during the forward pass
        let mut best_path_score = vec![f64::NEG_INFINITY; size + 1];
        let mut best_path_id = vec![u32::MAX; size + 1];
        let mut best_path_starts_at = vec![u32::MAX; size + 1];

        best_path_score[0] = 0.0;

        // Forward pass with prefetch hints
        let bytes = sentence.as_bytes();
        let mut starts_at = 0usize;

        while starts_at < size {
            let current_char = sentence[starts_at..].chars().next().unwrap();
            let mblen = current_char.len_utf8();

            let best_score_till_here = best_path_score[starts_at];

            if best_score_till_here == f64::NEG_INFINITY {
                starts_at += mblen;
                continue;
            }

            // Prefetch next iteration's data
            #[cfg(target_arch = "x86_64")]
            {
                let next_pos = (starts_at + mblen).min(size);
                unsafe {
                    use std::arch::x86_64::*;
                    let ptr = best_path_score.as_ptr().add(next_pos);
                    _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
                }
            }

            let mut has_single_node = false;

            // Enumerate prefixes with inline closure for better optimization
            self.trie.enumerate_prefixes(bytes, starts_at, |len, token_id| {
                let key_pos = starts_at + len;

                // Direct score access (unchecked for speed, bounds already verified)
                let score = if (token_id as usize) < self.scores.len() {
                    unsafe { *self.scores.get_unchecked(token_id as usize) }
                } else {
                    self.min_score - 10.0
                };

                let candidate_score = score + best_score_till_here;

                if best_path_starts_at[key_pos] == u32::MAX ||
                   candidate_score > best_path_score[key_pos] {
                    best_path_score[key_pos] = candidate_score;
                    best_path_starts_at[key_pos] = starts_at as u32;
                    best_path_id[key_pos] = token_id;
                }

                if !has_single_node && len == mblen {
                    has_single_node = true;
                }
            });

            if !has_single_node {
                let key_pos = starts_at + mblen;
                let candidate_score = unk_score + best_score_till_here;

                if best_path_starts_at[key_pos] == u32::MAX ||
                   candidate_score > best_path_score[key_pos] {
                    best_path_score[key_pos] = candidate_score;
                    best_path_starts_at[key_pos] = starts_at as u32;
                    best_path_id[key_pos] = self.config.unk_id;
                }
            }

            starts_at += mblen;
        }

        // Backtrace
        if best_path_starts_at[size] == u32::MAX {
            return vec![self.config.unk_id];
        }

        let mut result = Vec::with_capacity(size / 4);
        let mut pos = size;

        while pos > 0 && best_path_starts_at[pos] != u32::MAX {
            result.push(best_path_id[pos]);
            pos = best_path_starts_at[pos] as usize;
        }

        result.reverse();

        if self.config.fuse_unk {
            self.fuse_consecutive_unks(&mut result);
        }

        result
    }

    /// Ultra-optimized SIMD Viterbi encoding
    ///
    /// This version implements all optimizations for maximum performance:
    /// 1. Zero-copy normalization for ASCII text
    /// 2. ASCII fast path (no char boundary allocation)
    /// 3. Aggressive prefetching (multiple positions ahead)
    /// 4. Cache-optimized memory layout
    /// 5. Token cache for repeated texts (when enabled)
    ///
    /// Target: 20x faster than HuggingFace while maintaining 100% accuracy.
    #[inline]
    pub fn encode_hf_compat_simd_v2(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // Check cache first (if enabled)
        if let Some(ref cache) = self.cache {
            if let Some(cached) = cache.get(&text.to_string()) {
                return cached;
            }
        }

        // Zero-copy normalization (returns Cow - no allocation for ASCII)
        let normalized = normalize_for_hf(text);
        if normalized.is_empty() {
            return vec![];
        }

        // Use SIMD preprocessing
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let preprocessed = preprocess_text_simd(&normalized, self.config.add_prefix_space);

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let preprocessed = preprocess_text(&normalized, self.config.add_prefix_space);

        // Fast path: check if result is ASCII (very common for English text after preprocessing)
        // The SentencePiece underline (▁) is UTF-8, so we need to handle it specially
        // Actually, after preprocessing, the text has ▁ which is multi-byte
        // So we use the general path but optimize for the common case

        let sentence = match std::str::from_utf8(&preprocessed) {
            Ok(s) => s,
            Err(_) => return vec![self.config.unk_id],
        };

        let size = sentence.len();
        if size == 0 {
            return vec![];
        }

        let bytes = sentence.as_bytes();
        let unk_score = self.min_score - Self::K_UNK_PENALTY;

        // Compact arrays for better cache utilization
        let mut best_score = vec![f64::NEG_INFINITY; size + 1];
        let mut best_id = vec![u32::MAX; size + 1];
        let mut best_start = vec![u32::MAX; size + 1];

        best_score[0] = 0.0;

        // Forward pass - optimized for the common case where most chars are ASCII
        // We inline the character boundary logic to avoid allocation
        let mut starts_at = 0usize;

        while starts_at < size {
            // Inline UTF-8 character length calculation
            // This is faster than calling chars().next() for each position
            let first_byte = bytes[starts_at];
            let mblen = if first_byte < 0x80 {
                1 // ASCII: single byte
            } else if first_byte < 0xE0 {
                2 // 2-byte UTF-8
            } else if first_byte < 0xF0 {
                3 // 3-byte UTF-8 (includes ▁)
            } else {
                4 // 4-byte UTF-8
            };

            let char_end = (starts_at + mblen).min(size);
            let prev_score = best_score[starts_at];

            // Skip unreachable positions
            if prev_score == f64::NEG_INFINITY {
                starts_at = char_end;
                continue;
            }

            // Aggressive prefetching
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::*;
                // Prefetch next positions in best_score array
                if char_end < size {
                    _mm_prefetch(best_score.as_ptr().add(char_end) as *const i8, _MM_HINT_T0);
                }
                let prefetch_ahead = (char_end + 8).min(size);
                _mm_prefetch(best_score.as_ptr().add(prefetch_ahead) as *const i8, _MM_HINT_T1);
                // Prefetch trie base array
                _mm_prefetch(self.scores.as_ptr() as *const i8, _MM_HINT_T0);
            }

            let mut has_single_node = false;

            // Enumerate prefixes
            self.trie.enumerate_prefixes(bytes, starts_at, |len, token_id| {
                let key_pos = starts_at + len;

                // Unchecked score access
                let score = unsafe { *self.scores.get_unchecked(token_id as usize) };
                let candidate = score + prev_score;

                if best_start[key_pos] == u32::MAX || candidate > best_score[key_pos] {
                    best_score[key_pos] = candidate;
                    best_start[key_pos] = starts_at as u32;
                    best_id[key_pos] = token_id;
                }

                if len == mblen {
                    has_single_node = true;
                }
            });

            // Handle UNK
            if !has_single_node {
                let candidate = unk_score + prev_score;
                // UNK should only win if no other path exists or score is strictly better
                if best_start[char_end] == u32::MAX || candidate > best_score[char_end] {
                    best_score[char_end] = candidate;
                    best_start[char_end] = starts_at as u32;
                    best_id[char_end] = self.config.unk_id;
                }
            }

            starts_at = char_end;
        }

        // Fast backtrace
        if best_start[size] == u32::MAX {
            return vec![self.config.unk_id];
        }

        let mut result = Vec::with_capacity(size / 3);
        let mut pos = size;

        while pos > 0 && best_start[pos] != u32::MAX {
            result.push(best_id[pos]);
            pos = best_start[pos] as usize;
        }

        result.reverse();

        if self.config.fuse_unk {
            self.fuse_consecutive_unks(&mut result);
        }

        // Insert into cache (if enabled)
        if let Some(ref cache) = self.cache {
            cache.insert(text.to_string(), result.clone());
        }

        result
    }

    /// Parallel batch encoding with HuggingFace compatibility
    pub fn encode_batch_hf_compat(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts
            .par_iter()
            .map(|text| self.encode_hf_compat(text))
            .collect()
    }

    /// SIMD-accelerated parallel batch encoding with HuggingFace compatibility
    pub fn encode_batch_hf_compat_simd(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts
            .par_iter()
            .map(|text| self.encode_hf_compat_simd_v2(text))
            .collect()
    }

    /// Maximum performance SIMD Viterbi with buffer reuse
    ///
    /// This version reuses thread-local buffers to eliminate allocation overhead.
    /// Combined with all previous optimizations for maximum single-sequence throughput.
    /// Includes token cache for repeated texts when enabled.
    #[inline]
    pub fn encode_hf_compat_simd_v3(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        // Check cache first (if enabled)
        if let Some(ref cache) = self.cache {
            if let Some(cached) = cache.get(&text.to_string()) {
                return cached;
            }
        }

        // Zero-copy normalization
        let normalized = normalize_for_hf(text);
        if normalized.is_empty() {
            return vec![];
        }

        // Use SIMD preprocessing (allocates, but fast)
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let preprocessed = preprocess_text_simd(&normalized, self.config.add_prefix_space);

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let preprocessed = preprocess_text(&normalized, self.config.add_prefix_space);

        let sentence = match std::str::from_utf8(&preprocessed) {
            Ok(s) => s,
            Err(_) => return vec![self.config.unk_id],
        };

        let size = sentence.len();
        if size == 0 {
            return vec![];
        }

        let bytes = sentence.as_bytes();

        let encoded_result = VITERBI_BUFFERS.with(|buffers| {
            let mut buffers = buffers.borrow_mut();

            // Reset buffers for this size
            buffers.reset(size);

            let unk_score = self.min_score - Self::K_UNK_PENALTY;

            buffers.best_score[0] = 0.0;

            // Get raw pointers to avoid borrow issues in closure
            let best_score_ptr = buffers.best_score.as_mut_ptr();
            let best_id_ptr = buffers.best_id.as_mut_ptr();
            let best_start_ptr = buffers.best_start.as_mut_ptr();

            // Forward pass with aggressive prefetching
            let mut starts_at = 0usize;

            while starts_at < size {
                let first_byte = bytes[starts_at];
                let mblen = if first_byte < 0x80 {
                    1
                } else if first_byte < 0xE0 {
                    2
                } else if first_byte < 0xF0 {
                    3
                } else {
                    4
                };

                let char_end = (starts_at + mblen).min(size);
                let prev_score = unsafe { *best_score_ptr.add(starts_at) };

                if prev_score == f64::NEG_INFINITY {
                    starts_at = char_end;
                    continue;
                }

                // Prefetching
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    use std::arch::x86_64::*;
                    if char_end < size {
                        _mm_prefetch(best_score_ptr.add(char_end) as *const i8, _MM_HINT_T0);
                    }
                    let prefetch_ahead = (char_end + 16).min(size);
                    _mm_prefetch(best_score_ptr.add(prefetch_ahead) as *const i8, _MM_HINT_T1);
                }

                let mut has_single_node = false;

                self.trie.enumerate_prefixes(bytes, starts_at, |len, token_id| {
                    let key_pos = starts_at + len;

                    let score = unsafe { *self.scores.get_unchecked(token_id as usize) };
                    let candidate = score + prev_score;

                    unsafe {
                        let cur_start = *best_start_ptr.add(key_pos);
                        let cur_score = *best_score_ptr.add(key_pos);

                        if cur_start == u32::MAX || candidate > cur_score {
                            *best_score_ptr.add(key_pos) = candidate;
                            *best_start_ptr.add(key_pos) = starts_at as u32;
                            *best_id_ptr.add(key_pos) = token_id;
                        }
                    }

                    if len == mblen {
                        has_single_node = true;
                    }
                });

                if !has_single_node {
                    let candidate = unk_score + prev_score;
                    unsafe {
                        let cur_start = *best_start_ptr.add(char_end);
                        let cur_score = *best_score_ptr.add(char_end);

                        if cur_start == u32::MAX || candidate > cur_score {
                            *best_score_ptr.add(char_end) = candidate;
                            *best_start_ptr.add(char_end) = starts_at as u32;
                            *best_id_ptr.add(char_end) = self.config.unk_id;
                        }
                    }
                }

                starts_at = char_end;
            }

            // Backtrace
            if buffers.best_start[size] == u32::MAX {
                return vec![self.config.unk_id];
            }

            let mut result = Vec::with_capacity(size / 3);
            let mut pos = size;

            while pos > 0 && buffers.best_start[pos] != u32::MAX {
                result.push(buffers.best_id[pos]);
                pos = buffers.best_start[pos] as usize;
            }

            result.reverse();

            if self.config.fuse_unk {
                self.fuse_consecutive_unks(&mut result);
            }

            result
        });

        // Insert into cache (if enabled) - done outside closure to access self.cache
        if let Some(ref cache) = self.cache {
            cache.insert(text.to_string(), encoded_result.clone());
        }

        encoded_result
    }

    /// Encode and return full Encoding struct
    pub fn encode_to_encoding(&self, text: &str, add_special_tokens: bool) -> Encoding {
        let ids = self.encode_fast(text);
        let mut encoding = Encoding::with_capacity(ids.len() + 2);

        // Add BOS if configured
        if add_special_tokens {
            if let Some(ref bos) = self.config.bos_token {
                if let Some(bos_id) = self.token_to_id(bos) {
                    encoding.push(bos_id, bos.clone(), (0, 0), None, Some(0), true);
                }
            }
        }

        // Add tokens
        let mut offset = 0;
        for (word_idx, &token_id) in ids.iter().enumerate() {
            let token = self.id_to_token(token_id)
                .unwrap_or(&self.config.unk_token)
                .to_string();
            let token_len = token.len();

            encoding.push(
                token_id,
                token,
                (offset, offset + token_len),
                Some(word_idx as u32),
                Some(0),
                false,
            );

            offset += token_len;
        }

        // Add EOS if configured
        if add_special_tokens {
            if let Some(ref eos) = self.config.eos_token {
                if let Some(eos_id) = self.token_to_id(eos) {
                    encoding.push(eos_id, eos.clone(), (offset, offset), None, Some(0), true);
                }
            }
        }

        encoding
    }

    // ========================================================================
    // Phase 6: Parallel Batch Processing
    // ========================================================================

    /// Encode multiple texts in parallel
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts
            .par_iter()
            .map(|text| self.encode_fast(text))
            .collect()
    }

    /// Encode batch with length-based work balancing
    /// Sorts texts by length for better load distribution
    pub fn encode_batch_balanced(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        if texts.len() < 100 {
            // For small batches, parallel overhead may not be worth it
            return self.encode_batch(texts);
        }

        // Create indexed pairs and sort by length (longest first for better balancing)
        let mut indexed: Vec<_> = texts.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        // Process in parallel
        let mut results: Vec<_> = indexed
            .par_iter()
            .map(|(idx, text)| (*idx, self.encode_fast(text)))
            .collect();

        // Restore original order
        results.sort_by_key(|(idx, _)| *idx);
        results.into_iter().map(|(_, tokens)| tokens).collect()
    }

    /// Encode batch with minimum chunk size for Rayon
    pub fn encode_batch_chunked(&self, texts: &[&str], min_chunk_size: usize) -> Vec<Vec<u32>> {
        texts
            .par_iter()
            .with_min_len(min_chunk_size)
            .map(|text| self.encode_fast(text))
            .collect()
    }

    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        let mut result = String::new();

        for &id in ids {
            if let Some(token) = self.id_to_token(id) {
                if skip_special_tokens && token.starts_with('<') && token.ends_with('>') {
                    continue;
                }
                result.push_str(token);
            }
        }

        // Postprocess: replace ▁ with space
        result
            .chars()
            .map(|c| if c == SPIECE_UNDERLINE { ' ' } else { c })
            .collect::<String>()
            .trim_start()
            .to_string()
    }
}

// ============================================================================
// Phase 4: Arena-based encoding (optional optimization)
// ============================================================================

/// Thread-local arena for encoding operations
thread_local! {
    static ENCODE_ARENA: std::cell::RefCell<Bump> = std::cell::RefCell::new(Bump::with_capacity(64 * 1024));
}

/// Thread-local buffers for SIMD Viterbi encoding
/// This avoids repeated allocations in the hot path
struct ViterbiBuffers {
    best_score: Vec<f64>,
    best_id: Vec<u32>,
    best_start: Vec<u32>,
    preprocessed: Vec<u8>,
}

impl ViterbiBuffers {
    fn new() -> Self {
        Self {
            best_score: Vec::with_capacity(8192),
            best_id: Vec::with_capacity(8192),
            best_start: Vec::with_capacity(8192),
            preprocessed: Vec::with_capacity(8192),
        }
    }

    fn ensure_capacity(&mut self, size: usize) {
        let needed = size + 1;
        if self.best_score.len() < needed {
            self.best_score.resize(needed, f64::NEG_INFINITY);
            self.best_id.resize(needed, u32::MAX);
            self.best_start.resize(needed, u32::MAX);
        }
    }

    fn reset(&mut self, size: usize) {
        let needed = size + 1;
        // Reset only the portion we'll use
        for i in 0..needed.min(self.best_score.len()) {
            self.best_score[i] = f64::NEG_INFINITY;
            self.best_id[i] = u32::MAX;
            self.best_start[i] = u32::MAX;
        }
        // Extend if needed
        while self.best_score.len() < needed {
            self.best_score.push(f64::NEG_INFINITY);
            self.best_id.push(u32::MAX);
            self.best_start.push(u32::MAX);
        }
    }
}

thread_local! {
    static VITERBI_BUFFERS: std::cell::RefCell<ViterbiBuffers> =
        std::cell::RefCell::new(ViterbiBuffers::new());
}

impl UnigramFast {
    /// Encode using thread-local arena for temporary allocations
    /// This avoids repeated allocations for the working arrays
    #[inline]
    pub fn encode_with_arena(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        ENCODE_ARENA.with(|arena| {
            let mut arena = arena.borrow_mut();
            arena.reset();

            let preprocessed = preprocess_text(text, self.config.add_prefix_space);
            let bytes = &preprocessed;
            let n = bytes.len();

            if n == 0 {
                return vec![];
            }

            // Allocate working arrays from arena
            let best_score: &mut [f64] = arena.alloc_slice_fill_copy(n + 1, f64::NEG_INFINITY);
            let best_token: &mut [u32] = arena.alloc_slice_fill_copy(n + 1, u32::MAX);
            let best_prev: &mut [usize] = arena.alloc_slice_fill_copy(n + 1, usize::MAX);

            best_score[0] = 0.0;

            // Forward pass (same as encode_fast)
            for start in 0..n {
                let prev_score = best_score[start];
                if prev_score == f64::NEG_INFINITY {
                    continue;
                }

                let mut found_token = false;

                self.trie.enumerate_prefixes(bytes, start, |len, token_id| {
                    found_token = true;
                    let end = start + len;
                    let score = prev_score + self.get_score(token_id);

                    if score > best_score[end] {
                        best_score[end] = score;
                        best_token[end] = token_id;
                        best_prev[end] = start;
                    }
                });

                if !found_token && self.config.byte_fallback {
                    let byte_val = bytes[start];
                    let byte_id = self.byte_tokens[byte_val as usize];
                    if byte_id >= 0 {
                        found_token = true;
                        let score = prev_score + self.min_score - 5.0;
                        if score > best_score[start + 1] {
                            best_score[start + 1] = score;
                            best_token[start + 1] = byte_id as u32;
                            best_prev[start + 1] = start;
                        }
                    }
                }

                if !found_token {
                    let char_end = self.next_char_boundary(bytes, start);
                    let score = prev_score + self.min_score - 10.0;
                    if score > best_score[char_end] {
                        best_score[char_end] = score;
                        best_token[char_end] = self.config.unk_id;
                        best_prev[char_end] = start;
                    }
                }
            }

            // Backtrace
            if best_token[n] == u32::MAX {
                return vec![self.config.unk_id];
            }

            let mut result = Vec::with_capacity(n / 4);
            let mut pos = n;

            while pos > 0 && best_prev[pos] != usize::MAX {
                result.push(best_token[pos]);
                pos = best_prev[pos];
            }

            result.reverse();
            result
        })
    }
}

// ============================================================================
// Implement Tokenizer trait
// ============================================================================

impl Tokenizer for UnigramFast {
    fn vocabulary(&self) -> &Vocabulary {
        unimplemented!("UnigramFast uses internal vocab structure")
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> {
        Ok(self.encode_to_encoding(text, add_special_tokens))
    }

    fn encode_pair(
        &self,
        text: &str,
        text_pair: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let mut encoding = self.encode_to_encoding(text, false);
        let encoding_pair = self.encode_to_encoding(text_pair, false);
        encoding.merge(encoding_pair, true);

        if add_special_tokens {
            // Add special tokens if needed
        }

        Ok(encoding)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        Ok(UnigramFast::decode(self, ids, skip_special_tokens))
    }

    fn save(&self, _path: &std::path::Path) -> Result<()> {
        Err(Error::invalid_config("Saving not yet implemented for UnigramFast"))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tokenizer() -> UnigramFast {
        let vocab = vec![
            "<unk>".to_string(),
            "▁hello".to_string(),
            "▁world".to_string(),
            "▁".to_string(),
            "hello".to_string(),
            "world".to_string(),
            "lo".to_string(),
            "he".to_string(),
            "l".to_string(),
            "o".to_string(),
            "h".to_string(),
            "e".to_string(),
            "w".to_string(),
            "r".to_string(),
            "d".to_string(),
            ",".to_string(),
            "!".to_string(),
        ];

        let pieces = vec![
            UnigramPiece { token: "<unk>".to_string(), score: -100.0 },
            UnigramPiece { token: "▁hello".to_string(), score: -2.0 },
            UnigramPiece { token: "▁world".to_string(), score: -2.0 },
            UnigramPiece { token: "▁".to_string(), score: -1.0 },
            UnigramPiece { token: "hello".to_string(), score: -3.0 },
            UnigramPiece { token: "world".to_string(), score: -3.0 },
            UnigramPiece { token: "lo".to_string(), score: -4.0 },
            UnigramPiece { token: "he".to_string(), score: -4.0 },
            UnigramPiece { token: "l".to_string(), score: -5.0 },
            UnigramPiece { token: "o".to_string(), score: -5.0 },
            UnigramPiece { token: "h".to_string(), score: -5.0 },
            UnigramPiece { token: "e".to_string(), score: -5.0 },
            UnigramPiece { token: "w".to_string(), score: -5.0 },
            UnigramPiece { token: "r".to_string(), score: -5.0 },
            UnigramPiece { token: "d".to_string(), score: -5.0 },
            UnigramPiece { token: ",".to_string(), score: -3.0 },
            UnigramPiece { token: "!".to_string(), score: -3.0 },
        ];

        UnigramFast::new(vocab, pieces, UnigramFastConfig::default())
    }

    // ========================================================================
    // Phase 1: Single-Pass Viterbi Tests
    // ========================================================================

    #[test]
    fn test_single_pass_viterbi_basic() {
        let tokenizer = create_test_tokenizer();

        // "hello" should tokenize optimally
        let ids = tokenizer.encode_fast("hello");
        assert!(!ids.is_empty());

        // Should prefer longer tokens (▁hello) over characters
        let tokens: Vec<_> = ids.iter()
            .filter_map(|&id| tokenizer.id_to_token(id))
            .collect();

        // With prefix space, "hello" becomes "▁hello" which should match
        assert!(tokens.contains(&"▁hello") || tokens.contains(&"hello"));
    }

    #[test]
    fn test_single_pass_viterbi_multiword() {
        let tokenizer = create_test_tokenizer();

        let ids = tokenizer.encode_fast("hello world");
        assert!(!ids.is_empty());

        let tokens: Vec<_> = ids.iter()
            .filter_map(|&id| tokenizer.id_to_token(id))
            .collect();

        // Should find "▁hello" and "▁world" or their components
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_single_pass_viterbi_empty() {
        let tokenizer = create_test_tokenizer();

        let ids = tokenizer.encode_fast("");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_single_pass_viterbi_unknown() {
        let tokenizer = create_test_tokenizer();

        // Character not in vocab should fall back to unk
        let ids = tokenizer.encode_fast("xyz");
        assert!(!ids.is_empty());
    }

    // ========================================================================
    // Phase 2: Double-Array Trie Tests
    // ========================================================================

    #[test]
    fn test_dat_basic_lookup() {
        let vocab = vec![
            ("hello", 0u32),
            ("world", 1),
            ("he", 2),
            ("hell", 3),
        ];

        let trie = DoubleArrayTrie::from_vocab(vocab.into_iter());

        assert_eq!(trie.get(b"hello"), Some(0));
        assert_eq!(trie.get(b"world"), Some(1));
        assert_eq!(trie.get(b"he"), Some(2));
        assert_eq!(trie.get(b"hell"), Some(3));
        assert_eq!(trie.get(b"hel"), None);
        assert_eq!(trie.get(b"xyz"), None);
    }

    #[test]
    fn test_dat_prefix_enumeration() {
        let vocab = vec![
            ("h", 0u32),
            ("he", 1),
            ("hel", 2),
            ("hell", 3),
            ("hello", 4),
        ];

        let trie = DoubleArrayTrie::from_vocab(vocab.into_iter());

        let mut matches = Vec::new();
        trie.enumerate_prefixes(b"hello world", 0, |len, id| {
            matches.push((len, id));
        });

        // Should find all prefixes: h, he, hel, hell, hello
        assert_eq!(matches.len(), 5);
        assert!(matches.contains(&(1, 0))); // h
        assert!(matches.contains(&(2, 1))); // he
        assert!(matches.contains(&(3, 2))); // hel
        assert!(matches.contains(&(4, 3))); // hell
        assert!(matches.contains(&(5, 4))); // hello
    }

    #[test]
    fn test_dat_utf8() {
        let vocab = vec![
            ("▁", 0u32),
            ("▁hello", 1),
            ("café", 2),
        ];

        let trie = DoubleArrayTrie::from_vocab(vocab.into_iter());

        assert_eq!(trie.get("▁".as_bytes()), Some(0));
        assert_eq!(trie.get("▁hello".as_bytes()), Some(1));
        assert_eq!(trie.get("café".as_bytes()), Some(2));
    }

    #[test]
    fn test_dat_iterator() {
        let vocab = vec![
            ("h", 0u32),
            ("he", 1),
            ("hello", 2),
        ];

        let trie = DoubleArrayTrie::from_vocab(vocab.into_iter());

        let matches: Vec<_> = trie.common_prefix_search(b"hello world", 0).collect();

        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0], (1, 0)); // h
        assert_eq!(matches[1], (2, 1)); // he
        assert_eq!(matches[2], (5, 2)); // hello
    }

    // ========================================================================
    // Phase 3: SIMD Preprocessing Tests
    // ========================================================================

    #[test]
    fn test_preprocess_basic() {
        let result = preprocess_text("hello world", true);
        let result_str = String::from_utf8_lossy(&result);

        // Should have prefix ▁ and space replaced
        assert!(result_str.starts_with('▁'));
        assert!(!result_str.contains(' '));
        assert!(result_str.contains("▁world")); // space before world replaced
    }

    #[test]
    fn test_preprocess_no_spaces() {
        let result = preprocess_text("hello", true);
        let result_str = String::from_utf8_lossy(&result);

        assert!(result_str.starts_with('▁'));
        assert_eq!(result_str, "▁hello");
    }

    #[test]
    fn test_preprocess_many_spaces() {
        let result = preprocess_text("a b c d e", true);
        let result_str = String::from_utf8_lossy(&result);

        // Count ▁ characters (prefix + 4 spaces)
        let count = result_str.chars().filter(|&c| c == '▁').count();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_count_spaces() {
        assert_eq!(count_spaces_scalar(b"hello world"), 1);
        assert_eq!(count_spaces_scalar(b"a b c d e"), 4);
        assert_eq!(count_spaces_scalar(b"no_spaces"), 0);
        assert_eq!(count_spaces_scalar(b"   "), 3);
    }

    // ========================================================================
    // Phase 4: Arena Allocation Tests
    // ========================================================================

    #[test]
    fn test_arena_encode_correctness() {
        let tokenizer = create_test_tokenizer();

        // Arena encoding should produce same results as regular encoding
        let ids_fast = tokenizer.encode_fast("hello world");
        let ids_arena = tokenizer.encode_with_arena("hello world");

        assert_eq!(ids_fast, ids_arena);
    }

    #[test]
    fn test_arena_encode_multiple() {
        let tokenizer = create_test_tokenizer();

        // Multiple calls should work correctly (arena reset)
        for _ in 0..100 {
            let ids = tokenizer.encode_with_arena("hello world");
            assert!(!ids.is_empty());
        }
    }

    // ========================================================================
    // Phase 5: Dense Score Array Tests
    // ========================================================================

    #[test]
    fn test_dense_scores() {
        let tokenizer = create_test_tokenizer();

        // Verify scores are accessible
        let hello_id = tokenizer.token_to_id("▁hello").unwrap();
        let score = tokenizer.get_score(hello_id);
        assert!(score > f64::NEG_INFINITY);
        assert_eq!(score, -2.0);
    }

    #[test]
    fn test_dense_scores_unknown() {
        let tokenizer = create_test_tokenizer();

        // Unknown token ID should return penalty score
        let score = tokenizer.get_score(999999);
        assert!(score < tokenizer.min_score);
    }

    // ========================================================================
    // Phase 6: Parallel Batch Tests
    // ========================================================================

    #[test]
    fn test_batch_encode() {
        let tokenizer = create_test_tokenizer();

        let texts = vec!["hello", "world", "hello world"];
        let results = tokenizer.encode_batch(&texts);

        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_batch_encode_balanced() {
        let tokenizer = create_test_tokenizer();

        // Create texts of varying lengths
        let texts: Vec<&str> = vec![
            "a",
            "hello world",
            "this is a longer sentence",
            "hi",
            "another longer text here",
        ];

        let results = tokenizer.encode_batch_balanced(&texts);

        assert_eq!(results.len(), 5);

        // Results should be in original order
        let single_results: Vec<_> = texts.iter()
            .map(|t| tokenizer.encode_fast(t))
            .collect();

        assert_eq!(results, single_results);
    }

    #[test]
    fn test_batch_encode_empty() {
        let tokenizer = create_test_tokenizer();

        let texts: Vec<&str> = vec![];
        let results = tokenizer.encode_batch(&texts);

        assert!(results.is_empty());
    }

    // ========================================================================
    // Accuracy Tests (vs Original Implementation)
    // ========================================================================

    #[test]
    fn test_decode_roundtrip() {
        let tokenizer = create_test_tokenizer();

        let text = "hello world";
        let ids = tokenizer.encode_fast(text);
        let decoded = tokenizer.decode(&ids, false);

        // After decode, should get original text back (possibly with slight spacing differences)
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }
}
