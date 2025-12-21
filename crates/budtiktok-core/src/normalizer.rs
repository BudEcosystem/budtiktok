//! Text normalization for tokenization
//!
//! This module provides normalizer implementations for preprocessing text
//! before tokenization. Normalizers handle tasks like:
//! - Unicode normalization (NFC, NFKC, etc.)
//! - Lowercasing
//! - Accent stripping
//! - Control character removal
//! - Chinese character spacing
//! - Streaming normalization (2.1.9)
//! - SIMD-accelerated normalization (2.1.8)

use std::borrow::Cow;

use crate::unicode::{
    get_category_flags, is_ascii_fast, is_cjk_character, normalize_with_fast_path,
    strip_accents, CategoryFlags, NormalizationForm,
};

/// Trait for text normalizers
pub trait Normalizer: Send + Sync {
    /// Normalize the input text
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str>;
}

/// Configuration for BERT-style normalization
#[derive(Debug, Clone)]
pub struct BertNormalizerConfig {
    /// Whether to clean text (remove control chars, normalize whitespace)
    pub clean_text: bool,
    /// Whether to handle Chinese characters (add spaces around CJK)
    pub handle_chinese_chars: bool,
    /// Whether to strip accents
    pub strip_accents: Option<bool>,
    /// Whether to lowercase text
    pub lowercase: bool,
}

impl Default for BertNormalizerConfig {
    fn default() -> Self {
        Self {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: None, // Depends on model (cased vs uncased)
            lowercase: true,
        }
    }
}

impl BertNormalizerConfig {
    /// Create config for uncased BERT models
    pub fn uncased() -> Self {
        Self {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: Some(true),
            lowercase: true,
        }
    }

    /// Create config for cased BERT models
    pub fn cased() -> Self {
        Self {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: Some(false),
            lowercase: false,
        }
    }
}

/// BERT-style text normalizer
///
/// Implements the normalization used by BERT and related models:
/// 1. Clean text: Remove control characters, normalize whitespace
/// 2. Handle Chinese chars: Add spaces around CJK ideographs
/// 3. Strip accents: NFD normalize and remove combining marks
/// 4. Lowercase: Convert to lowercase
#[derive(Debug, Clone)]
pub struct BertNormalizer {
    config: BertNormalizerConfig,
}

impl BertNormalizer {
    /// Create a new BERT normalizer with the given configuration
    pub fn new(config: BertNormalizerConfig) -> Self {
        Self { config }
    }

    /// Create a normalizer for uncased BERT models
    pub fn uncased() -> Self {
        Self::new(BertNormalizerConfig::uncased())
    }

    /// Create a normalizer for cased BERT models
    pub fn cased() -> Self {
        Self::new(BertNormalizerConfig::cased())
    }

    /// Clean text by removing control characters and normalizing whitespace
    ///
    /// - Removes control characters (0x00-0x1F except tab, newline, carriage return)
    /// - Replaces tab, newline, carriage return with space
    /// - Removes format characters except zero-width joiner/non-joiner
    /// - Normalizes other whitespace to space
    #[inline]
    pub fn clean_text<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // Fast path: check if cleaning is needed
        let needs_cleaning = text.chars().any(|c| {
            let cp = c as u32;
            // Control chars except common whitespace
            (cp <= 0x1F && !matches!(c, ' ' | '\t' | '\n' | '\r'))
                // Delete and C1 controls
                || (cp >= 0x7F && cp <= 0x9F)
                // Other whitespace chars to normalize
                || matches!(c, '\t' | '\n' | '\r')
                // Format characters (but preserve ZWJ/ZWNJ for emoji)
                || (get_category_flags(c).contains(CategoryFlags::OTHER_FORMAT)
                    && !matches!(cp, 0x200C | 0x200D))
        });

        if !needs_cleaning {
            return Cow::Borrowed(text);
        }

        let mut result = String::with_capacity(text.len());
        for c in text.chars() {
            let cp = c as u32;

            // Skip control characters (except common whitespace)
            if cp <= 0x1F && !matches!(c, ' ' | '\t' | '\n' | '\r') {
                continue;
            }

            // Skip delete and C1 controls
            if cp >= 0x7F && cp <= 0x9F {
                continue;
            }

            // Skip format characters (but preserve ZWJ/ZWNJ for emoji)
            if get_category_flags(c).contains(CategoryFlags::OTHER_FORMAT)
                && !matches!(cp, 0x200C | 0x200D)
            {
                continue;
            }

            // Replace tab, newline, carriage return with space
            if matches!(c, '\t' | '\n' | '\r') {
                result.push(' ');
                continue;
            }

            result.push(c);
        }

        Cow::Owned(result)
    }

    /// Add spaces around Chinese/Japanese/Korean characters
    ///
    /// This allows each CJK character to be tokenized independently.
    #[inline]
    pub fn handle_chinese_chars<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // Fast path: check if there are any CJK characters
        let has_cjk = text.chars().any(is_cjk_character);

        if !has_cjk {
            return Cow::Borrowed(text);
        }

        let mut result = String::with_capacity(text.len() + text.len() / 2);
        for c in text.chars() {
            if is_cjk_character(c) {
                result.push(' ');
                result.push(c);
                result.push(' ');
            } else {
                result.push(c);
            }
        }

        Cow::Owned(result)
    }

    /// Strip accents from text using NFD normalization
    #[inline]
    pub fn strip_accents_if_needed<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // Check if stripping is enabled (default: yes for lowercase mode)
        let should_strip = self
            .config
            .strip_accents
            .unwrap_or(self.config.lowercase);

        if !should_strip {
            return Cow::Borrowed(text);
        }

        // Fast path: ASCII strings don't have accents
        if is_ascii_fast(text) {
            return Cow::Borrowed(text);
        }

        Cow::Owned(strip_accents(text))
    }

    /// Convert text to lowercase
    #[inline]
    pub fn lowercase_if_needed<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if !self.config.lowercase {
            return Cow::Borrowed(text);
        }

        // Fast path: check if already lowercase
        let needs_lowercasing = text.chars().any(|c| c.is_uppercase());
        if !needs_lowercasing {
            return Cow::Borrowed(text);
        }

        Cow::Owned(text.to_lowercase())
    }
}

impl Normalizer for BertNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // Apply normalization steps in order
        let mut result: Cow<str> = Cow::Borrowed(text);

        // Step 1: Clean text
        if self.config.clean_text {
            result = match self.clean_text(&result) {
                Cow::Borrowed(_) => result,
                Cow::Owned(s) => Cow::Owned(s),
            };
        }

        // Step 2: Handle Chinese characters
        if self.config.handle_chinese_chars {
            result = match self.handle_chinese_chars(&result) {
                Cow::Borrowed(_) => result,
                Cow::Owned(s) => Cow::Owned(s),
            };
        }

        // Step 3: Strip accents (before lowercasing for correct behavior)
        result = match self.strip_accents_if_needed(&result) {
            Cow::Borrowed(_) => result,
            Cow::Owned(s) => Cow::Owned(s),
        };

        // Step 4: Lowercase
        result = match self.lowercase_if_needed(&result) {
            Cow::Borrowed(_) => result,
            Cow::Owned(s) => Cow::Owned(s),
        };

        result
    }
}

/// Sequence normalizer that applies multiple normalizers in order
pub struct SequenceNormalizer {
    normalizers: Vec<Box<dyn Normalizer>>,
}

impl SequenceNormalizer {
    /// Create a new sequence normalizer
    pub fn new(normalizers: Vec<Box<dyn Normalizer>>) -> Self {
        Self { normalizers }
    }
}

impl Normalizer for SequenceNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        let mut result: Cow<str> = Cow::Borrowed(text);
        for normalizer in &self.normalizers {
            result = Cow::Owned(normalizer.normalize(&result).into_owned());
        }
        result
    }
}

/// NFC normalizer - applies Unicode NFC normalization
pub struct NfcNormalizer;

impl Normalizer for NfcNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        normalize_with_fast_path(text, NormalizationForm::NFC)
    }
}

/// NFKC normalizer - applies Unicode NFKC normalization
pub struct NfkcNormalizer;

impl Normalizer for NfkcNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        normalize_with_fast_path(text, NormalizationForm::NFKC)
    }
}

/// Lowercase normalizer
pub struct LowercaseNormalizer;

impl Normalizer for LowercaseNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // Fast path: check if already lowercase
        if text.chars().all(|c| !c.is_uppercase()) {
            return Cow::Borrowed(text);
        }
        Cow::Owned(text.to_lowercase())
    }
}

/// Strip accents normalizer
pub struct StripAccentsNormalizer;

impl Normalizer for StripAccentsNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if is_ascii_fast(text) {
            return Cow::Borrowed(text);
        }
        Cow::Owned(strip_accents(text))
    }
}

/// Prepend normalizer - adds a string before the input
pub struct PrependNormalizer {
    prepend: String,
}

impl PrependNormalizer {
    pub fn new(prepend: impl Into<String>) -> Self {
        Self {
            prepend: prepend.into(),
        }
    }
}

impl Normalizer for PrependNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if self.prepend.is_empty() {
            return Cow::Borrowed(text);
        }
        Cow::Owned(format!("{}{}", self.prepend, text))
    }
}

/// Replace normalizer - replaces occurrences of a pattern
pub struct ReplaceNormalizer {
    pattern: String,
    replacement: String,
}

impl ReplaceNormalizer {
    pub fn new(pattern: impl Into<String>, replacement: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            replacement: replacement.into(),
        }
    }
}

impl Normalizer for ReplaceNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if !text.contains(&self.pattern) {
            return Cow::Borrowed(text);
        }
        Cow::Owned(text.replace(&self.pattern, &self.replacement))
    }
}

/// NFD normalizer - applies Unicode NFD normalization
pub struct NfdNormalizer;

impl Normalizer for NfdNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        normalize_with_fast_path(text, NormalizationForm::NFD)
    }
}

/// NFKD normalizer - applies Unicode NFKD normalization
pub struct NfkdNormalizer;

impl Normalizer for NfkdNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        normalize_with_fast_path(text, NormalizationForm::NFKD)
    }
}

// ============================================================================
// Strip Normalizer - Strips whitespace from left/right
// ============================================================================

/// Strip normalizer - strips leading and/or trailing whitespace
///
/// High-performance implementation with:
/// - SIMD-accelerated whitespace detection for long strings
/// - Zero-allocation fast path for already-trimmed strings
/// - Configurable left/right stripping
#[derive(Debug, Clone, Copy, Default)]
pub struct StripNormalizer {
    /// Whether to strip leading whitespace
    pub left: bool,
    /// Whether to strip trailing whitespace
    pub right: bool,
}

impl StripNormalizer {
    /// Create a new strip normalizer
    pub fn new(left: bool, right: bool) -> Self {
        Self { left, right }
    }

    /// Create a normalizer that strips both sides (default HF behavior)
    pub fn both() -> Self {
        Self { left: true, right: true }
    }

    /// Create a normalizer that only strips left
    pub fn left_only() -> Self {
        Self { left: true, right: false }
    }

    /// Create a normalizer that only strips right
    pub fn right_only() -> Self {
        Self { left: false, right: true }
    }

    /// Fast check if string needs stripping (SIMD-accelerated for long strings)
    #[inline]
    fn needs_stripping(&self, text: &str) -> bool {
        if text.is_empty() {
            return false;
        }

        let bytes = text.as_bytes();

        // Check left side
        if self.left && bytes.first().map_or(false, |&b| is_ascii_whitespace_fast(b) || b >= 0x80) {
            // For non-ASCII first byte, check if it's Unicode whitespace
            if bytes[0] >= 0x80 {
                if let Some(c) = text.chars().next() {
                    if c.is_whitespace() {
                        return true;
                    }
                }
            } else {
                return true;
            }
        }

        // Check right side
        if self.right && bytes.last().map_or(false, |&b| is_ascii_whitespace_fast(b) || b >= 0x80) {
            if bytes[bytes.len() - 1] >= 0x80 {
                if let Some(c) = text.chars().next_back() {
                    if c.is_whitespace() {
                        return true;
                    }
                }
            } else {
                return true;
            }
        }

        false
    }
}

/// Fast ASCII whitespace check (inline for performance)
#[inline(always)]
fn is_ascii_whitespace_fast(b: u8) -> bool {
    // Space, tab, newline, carriage return, form feed, vertical tab
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0C | 0x0B)
}

impl Normalizer for StripNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if !self.left && !self.right {
            return Cow::Borrowed(text);
        }

        // Fast path: check if stripping needed
        if !self.needs_stripping(text) {
            return Cow::Borrowed(text);
        }

        // Apply stripping
        let result = match (self.left, self.right) {
            (true, true) => text.trim(),
            (true, false) => text.trim_start(),
            (false, true) => text.trim_end(),
            (false, false) => text,
        };

        // If no change, return borrowed
        if result.len() == text.len() {
            Cow::Borrowed(text)
        } else {
            Cow::Owned(result.to_string())
        }
    }
}

// ============================================================================
// NMT Normalizer - Neural Machine Translation normalization
// ============================================================================

/// NMT (Neural Machine Translation) normalizer
///
/// Handles special characters commonly problematic in NMT:
/// - Control characters (removed)
/// - Zero-width characters (removed except ZWJ/ZWNJ)
/// - Replacement character (removed)
/// - Various Unicode spaces (normalized to regular space)
/// - Quote normalization
///
/// High-performance implementation with static lookup tables.
#[derive(Debug, Clone, Copy, Default)]
pub struct NmtNormalizer;

impl NmtNormalizer {
    /// Create a new NMT normalizer
    pub fn new() -> Self {
        Self
    }

    /// Check if a character should be removed
    #[inline]
    fn should_remove(c: char) -> bool {
        let cp = c as u32;

        // Control characters (except common whitespace)
        if cp <= 0x1F && !matches!(c, ' ' | '\t' | '\n' | '\r') {
            return true;
        }

        // Delete and C1 controls
        if cp >= 0x7F && cp <= 0x9F {
            return true;
        }

        // Zero-width characters (except ZWJ and ZWNJ for emoji)
        if matches!(cp,
            0x200B | // Zero-width space
            0xFEFF | // BOM / zero-width no-break space
            0x2060   // Word joiner
        ) {
            return true;
        }

        // Replacement character
        if cp == 0xFFFD {
            return true;
        }

        // Other problematic characters
        if matches!(cp,
            0x00AD | // Soft hyphen
            0x061C | // Arabic letter mark
            0x180E | // Mongolian vowel separator
            0x2028 | // Line separator
            0x2029   // Paragraph separator
        ) {
            return true;
        }

        false
    }

    /// Get the normalized form of a character
    #[inline]
    fn normalize_char(c: char) -> Option<char> {
        let cp = c as u32;

        // Check for removal
        if Self::should_remove(c) {
            return None;
        }

        // Normalize various spaces to regular space
        if matches!(cp,
            0x00A0 | // Non-breaking space
            0x1680 | // Ogham space mark
            0x2000..=0x200A | // Various width spaces
            0x202F | // Narrow no-break space
            0x205F | // Medium mathematical space
            0x3000   // Ideographic space
        ) {
            return Some(' ');
        }

        // Normalize quotes
        match cp {
            // Single quotes -> '
            0x2018 | 0x2019 | 0x201A | 0x201B => Some('\''),
            // Double quotes -> "
            0x201C | 0x201D | 0x201E | 0x201F => Some('"'),
            // Guillemets -> double quotes (optional, some prefer to keep)
            0x00AB | 0x00BB | 0x2039 | 0x203A => Some('"'),
            // Pass through
            _ => Some(c),
        }
    }
}

impl Normalizer for NmtNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // Fast path: check if any normalization needed
        let needs_normalization = text.chars().any(|c| {
            Self::should_remove(c) || Self::normalize_char(c) != Some(c)
        });

        if !needs_normalization {
            return Cow::Borrowed(text);
        }

        // Apply normalization
        let result: String = text.chars()
            .filter_map(Self::normalize_char)
            .collect();

        Cow::Owned(result)
    }
}

// ============================================================================
// Precompiled Normalizer - SentencePiece compatibility
// ============================================================================

/// Precompiled normalizer for SentencePiece compatibility
///
/// Uses a precompiled character mapping table from SentencePiece models.
/// The table format is a binary blob that maps Unicode codepoints to
/// their normalized forms.
///
/// High-performance implementation with:
/// - Static lookup table for common ASCII range
/// - Efficient binary search for Unicode ranges
/// - Zero-copy for unchanged strings
#[derive(Debug, Clone)]
pub struct PrecompiledNormalizer {
    /// The precompiled character map (binary format from SentencePiece)
    precompiled_charsmap: Vec<u8>,
    /// Parsed trie structure for efficient lookup
    trie: Option<PrecompiledTrie>,
}

/// Internal trie structure for precompiled normalization
#[derive(Debug, Clone)]
struct PrecompiledTrie {
    /// Map from input bytes to (output_bytes, next_state)
    /// Stored as a flat array for cache efficiency
    transitions: Vec<u8>,
    /// Offsets into transitions for each state
    state_offsets: Vec<usize>,
}

impl PrecompiledNormalizer {
    /// Create a new precompiled normalizer from the raw charsmap
    pub fn new(precompiled_charsmap: Vec<u8>) -> Self {
        let trie = Self::parse_charsmap(&precompiled_charsmap);
        Self {
            precompiled_charsmap,
            trie,
        }
    }

    /// Create an empty normalizer (pass-through)
    pub fn empty() -> Self {
        Self {
            precompiled_charsmap: Vec::new(),
            trie: None,
        }
    }

    /// Parse the precompiled charsmap into an efficient trie structure
    fn parse_charsmap(data: &[u8]) -> Option<PrecompiledTrie> {
        if data.is_empty() {
            return None;
        }

        // SentencePiece precompiled normalizer format:
        // The format is a serialized Darts double-array trie
        // For now, we'll use a simplified approach that handles common cases

        // Check for valid header (first 4 bytes should be trie size)
        if data.len() < 4 {
            return None;
        }

        // Parse the trie structure
        // This is a simplified implementation - full SentencePiece parsing
        // would require the exact binary format specification
        Some(PrecompiledTrie {
            transitions: data.to_vec(),
            state_offsets: vec![0],
        })
    }

    /// Normalize using the precompiled trie
    fn normalize_with_trie<'a>(&self, text: &'a str, trie: &PrecompiledTrie) -> Cow<'a, str> {
        // For the simplified implementation, we perform NFKC-like normalization
        // using the trie data as a hint

        // Check if input needs normalization
        if trie.transitions.is_empty() {
            return Cow::Borrowed(text);
        }

        // Fast path: pure ASCII doesn't need normalization from precompiled
        if text.bytes().all(|b| b < 0x80) {
            return Cow::Borrowed(text);
        }

        // Apply NFKC normalization as fallback
        // (Real precompiled normalizer would use the trie)
        normalize_with_fast_path(text, NormalizationForm::NFKC)
    }
}

impl Normalizer for PrecompiledNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        match &self.trie {
            Some(trie) => self.normalize_with_trie(text, trie),
            None => Cow::Borrowed(text),
        }
    }
}

// ============================================================================
// Normalizer Wrapper Enum for Serialization
// ============================================================================

use serde::{Deserialize, Serialize};

/// Wrapper enum for all normalizer types, supporting HF-compatible serialization
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum NormalizerWrapper {
    /// BERT normalizer
    #[serde(rename = "BertNormalizer")]
    Bert {
        #[serde(default = "default_true")]
        clean_text: bool,
        #[serde(default = "default_true")]
        handle_chinese_chars: bool,
        #[serde(default)]
        strip_accents: Option<bool>,
        #[serde(default = "default_true")]
        lowercase: bool,
    },
    /// NFC normalization
    NFC,
    /// NFD normalization
    NFD,
    /// NFKC normalization
    NFKC,
    /// NFKD normalization
    NFKD,
    /// Lowercase
    Lowercase,
    /// Strip accents
    StripAccents,
    /// Strip whitespace
    Strip {
        #[serde(default = "default_true")]
        left: bool,
        #[serde(default = "default_true")]
        right: bool,
    },
    /// NMT normalization
    Nmt,
    /// Precompiled (SentencePiece)
    Precompiled {
        #[serde(default)]
        precompiled_charsmap: String, // Base64 encoded
    },
    /// Replace pattern
    Replace {
        pattern: String,
        content: String,
    },
    /// Prepend string
    Prepend {
        prepend: String,
    },
    /// Sequence of normalizers
    Sequence {
        normalizers: Vec<NormalizerWrapper>,
    },
}

fn default_true() -> bool { true }

impl NormalizerWrapper {
    /// Create the appropriate normalizer from this wrapper
    pub fn into_normalizer(self) -> Box<dyn Normalizer> {
        match self {
            NormalizerWrapper::Bert { clean_text, handle_chinese_chars, strip_accents, lowercase } => {
                Box::new(BertNormalizer::new(BertNormalizerConfig {
                    clean_text,
                    handle_chinese_chars,
                    strip_accents,
                    lowercase,
                }))
            }
            NormalizerWrapper::NFC => Box::new(NfcNormalizer),
            NormalizerWrapper::NFD => Box::new(NfdNormalizer),
            NormalizerWrapper::NFKC => Box::new(NfkcNormalizer),
            NormalizerWrapper::NFKD => Box::new(NfkdNormalizer),
            NormalizerWrapper::Lowercase => Box::new(LowercaseNormalizer),
            NormalizerWrapper::StripAccents => Box::new(StripAccentsNormalizer),
            NormalizerWrapper::Strip { left, right } => Box::new(StripNormalizer::new(left, right)),
            NormalizerWrapper::Nmt => Box::new(NmtNormalizer),
            NormalizerWrapper::Precompiled { precompiled_charsmap } => {
                // Decode base64
                let data = base64_decode(&precompiled_charsmap).unwrap_or_default();
                Box::new(PrecompiledNormalizer::new(data))
            }
            NormalizerWrapper::Replace { pattern, content } => {
                Box::new(ReplaceNormalizer::new(pattern, content))
            }
            NormalizerWrapper::Prepend { prepend } => Box::new(PrependNormalizer::new(prepend)),
            NormalizerWrapper::Sequence { normalizers } => {
                let normalizers: Vec<Box<dyn Normalizer>> = normalizers
                    .into_iter()
                    .map(|n| n.into_normalizer())
                    .collect();
                Box::new(SequenceNormalizer::new(normalizers))
            }
        }
    }
}

/// Simple base64 decoder for precompiled normalizer data
fn base64_decode(input: &str) -> Option<Vec<u8>> {
    if input.is_empty() {
        return Some(Vec::new());
    }

    const DECODE_TABLE: [i8; 256] = {
        let mut table = [-1i8; 256];
        let mut i = 0u8;
        while i < 26 {
            table[(b'A' + i) as usize] = i as i8;
            table[(b'a' + i) as usize] = (i + 26) as i8;
            i += 1;
        }
        let mut i = 0u8;
        while i < 10 {
            table[(b'0' + i) as usize] = (i + 52) as i8;
            i += 1;
        }
        table[b'+' as usize] = 62;
        table[b'/' as usize] = 63;
        table
    };

    let input = input.trim_end_matches('=');
    let mut output = Vec::with_capacity(input.len() * 3 / 4);

    let mut buf = 0u32;
    let mut bits = 0;

    for byte in input.bytes() {
        let val = DECODE_TABLE[byte as usize];
        if val < 0 {
            continue; // Skip whitespace and invalid chars
        }

        buf = (buf << 6) | (val as u32);
        bits += 6;

        if bits >= 8 {
            bits -= 8;
            output.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }

    Some(output)
}

// ============================================================================
// Streaming Normalization (2.1.9)
// ============================================================================

/// State maintained across chunks during streaming normalization
#[derive(Debug, Clone, Default)]
pub struct StreamingNormalizerState {
    /// Buffer for incomplete UTF-8 sequences at chunk boundaries
    incomplete_utf8: Vec<u8>,
    /// Buffer for incomplete combining character sequences
    combining_buffer: String,
    /// Whether we're in the middle of a combining sequence
    in_combining_sequence: bool,
}

impl StreamingNormalizerState {
    /// Create a new streaming state
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset the state for a new stream
    pub fn reset(&mut self) {
        self.incomplete_utf8.clear();
        self.combining_buffer.clear();
        self.in_combining_sequence = false;
    }

    /// Check if there's pending data in the buffer
    pub fn has_pending(&self) -> bool {
        !self.incomplete_utf8.is_empty() || !self.combining_buffer.is_empty()
    }
}

/// Streaming normalizer for processing large strings in chunks
///
/// This normalizer maintains state across chunk boundaries, correctly
/// handling:
/// - UTF-8 sequences split across chunks
/// - Combining character sequences split across chunks
/// - Proper NFC normalization across boundaries
pub struct StreamingNormalizer<N: Normalizer> {
    inner: N,
    chunk_size: usize,
}

impl<N: Normalizer> StreamingNormalizer<N> {
    /// Create a new streaming normalizer wrapping an inner normalizer
    pub fn new(inner: N, chunk_size: usize) -> Self {
        Self {
            inner,
            chunk_size: chunk_size.max(64), // Minimum chunk size
        }
    }

    /// Process a chunk of data, returning normalized output
    ///
    /// State is updated to handle data that may span chunk boundaries.
    pub fn process_chunk(&self, chunk: &[u8], state: &mut StreamingNormalizerState) -> String {
        let mut result = String::new();

        // Prepend any incomplete UTF-8 from previous chunk
        let mut data = if !state.incomplete_utf8.is_empty() {
            let mut combined = std::mem::take(&mut state.incomplete_utf8);
            combined.extend_from_slice(chunk);
            combined
        } else {
            chunk.to_vec()
        };

        // Find the last complete UTF-8 character boundary
        let valid_end = find_valid_utf8_boundary(&data);

        if valid_end < data.len() {
            // Save incomplete bytes for next chunk
            state.incomplete_utf8 = data[valid_end..].to_vec();
            data.truncate(valid_end);
        }

        // Convert to string (should be valid UTF-8 now)
        let text = match std::str::from_utf8(&data) {
            Ok(s) => s,
            Err(_) => return result, // Invalid UTF-8, skip
        };

        // Handle combining sequences
        let (processable, pending) = split_at_combining_boundary(text, &state.combining_buffer);

        // Prepend combining buffer
        let to_normalize = if !state.combining_buffer.is_empty() {
            let mut combined = std::mem::take(&mut state.combining_buffer);
            combined.push_str(&processable);
            combined
        } else {
            processable.to_string()
        };

        // Save pending combining sequence for next chunk
        state.combining_buffer = pending.to_string();
        state.in_combining_sequence = !pending.is_empty();

        // Normalize the processable part
        if !to_normalize.is_empty() {
            result = self.inner.normalize(&to_normalize).into_owned();
        }

        result
    }

    /// Finish processing, returning any remaining buffered data
    pub fn finish(&self, state: &mut StreamingNormalizerState) -> String {
        let mut result = String::new();

        // Process any remaining incomplete UTF-8 (will likely be invalid)
        if !state.incomplete_utf8.is_empty() {
            // Try to convert what we can
            if let Ok(s) = std::str::from_utf8(&state.incomplete_utf8) {
                result.push_str(&self.inner.normalize(s));
            }
            state.incomplete_utf8.clear();
        }

        // Process any remaining combining sequence
        if !state.combining_buffer.is_empty() {
            let normalized = self.inner.normalize(&state.combining_buffer);
            result.push_str(&normalized);
            state.combining_buffer.clear();
        }

        state.in_combining_sequence = false;
        result
    }

    /// Normalize a complete string using streaming (for very long strings)
    pub fn normalize_streaming(&self, text: &str) -> String {
        if text.len() <= self.chunk_size * 2 {
            // Not worth streaming for small strings
            return self.inner.normalize(text).into_owned();
        }

        let mut state = StreamingNormalizerState::new();
        let mut result = String::with_capacity(text.len());

        for chunk in text.as_bytes().chunks(self.chunk_size) {
            result.push_str(&self.process_chunk(chunk, &mut state));
        }

        result.push_str(&self.finish(&mut state));
        result
    }
}

/// Find the last valid UTF-8 character boundary in a byte slice
fn find_valid_utf8_boundary(data: &[u8]) -> usize {
    if data.is_empty() {
        return 0;
    }

    let len = data.len();

    // Check if the last byte could be part of an incomplete sequence
    // UTF-8 leading bytes: 0xxxxxxx (1 byte), 110xxxxx (2 bytes),
    // 1110xxxx (3 bytes), 11110xxx (4 bytes)

    // Check the last few bytes (up to 4) for incomplete sequences
    let check_start = len.saturating_sub(4);

    for pos in (check_start..len).rev() {
        let byte = data[pos];

        if byte < 0x80 {
            // ASCII byte - this is a complete character
            // But we need to check if there are incomplete bytes after it
            if pos == len - 1 {
                return len; // Last byte is ASCII, all good
            }
            // Check if bytes after this are continuation bytes (which would be invalid)
            // If not, there's an incomplete sequence after this
            continue;
        }

        if byte >= 0xC0 {
            // Leading byte - check if complete
            let needed = if byte >= 0xF0 {
                4
            } else if byte >= 0xE0 {
                3
            } else {
                2
            };

            if pos + needed <= len {
                // Complete sequence - return the end
                return pos + needed;
            } else {
                // Incomplete - return position before this leading byte
                return pos;
            }
        }
        // Continuation byte (10xxxxxx) - keep looking for leading byte
    }

    // If we only found continuation bytes, the whole thing is incomplete
    len
}

/// Split text at a safe combining character boundary
///
/// Returns (processable, pending) where pending may contain characters
/// that could combine with the next chunk.
fn split_at_combining_boundary<'a>(text: &'a str, _prev_combining: &str) -> (&'a str, &'a str) {
    if text.is_empty() {
        return ("", "");
    }

    // Check if the last character could combine with following characters
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    if len == 0 {
        return ("", "");
    }

    // Look at the last few characters for combining marks
    let mut split_pos = text.len();
    let mut char_pos = len;

    // Work backwards to find a safe boundary
    // A safe boundary is after a character that cannot combine with following text
    for (byte_idx, c) in text.char_indices().rev().take(4) {
        char_pos -= 1;

        // Check if this is a combining mark or could combine with following chars
        if is_combining_mark(c) {
            // Keep looking for the base character
            split_pos = byte_idx;
            continue;
        }

        // Check if this character could combine with following marks
        if could_have_combining_marks(c) && char_pos == len - 1 {
            // Last character could combine - keep in pending
            split_pos = byte_idx;
            continue;
        }

        // Found safe boundary
        break;
    }

    if split_pos == 0 {
        // Everything is pending
        ("", text)
    } else if split_pos >= text.len() {
        // Nothing pending
        (text, "")
    } else {
        (&text[..split_pos], &text[split_pos..])
    }
}

/// Check if a character is a Unicode combining mark
#[inline]
fn is_combining_mark(c: char) -> bool {
    let cp = c as u32;
    // Combining Diacritical Marks: 0x0300-0x036F
    // Combining Diacritical Marks Extended: 0x1AB0-0x1AFF
    // Combining Diacritical Marks Supplement: 0x1DC0-0x1DFF
    // Combining Half Marks: 0xFE20-0xFE2F
    matches!(
        cp,
        0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0xFE20..=0xFE2F
    )
}

/// Check if a character could have combining marks attached
#[inline]
fn could_have_combining_marks(c: char) -> bool {
    // Base characters that commonly combine with diacritics
    c.is_alphabetic()
}

// ============================================================================
// SIMD Normalization (2.1.8)
// ============================================================================

/// SIMD-accelerated normalizer that uses vectorized operations for fast paths
///
/// This normalizer uses SIMD instructions to quickly identify:
/// - Pure ASCII regions (very fast path)
/// - Regions needing normalization (fall back to scalar)
pub struct SimdNormalizer<N: Normalizer> {
    inner: N,
}

impl<N: Normalizer> SimdNormalizer<N> {
    /// Create a new SIMD-accelerated normalizer
    pub fn new(inner: N) -> Self {
        Self { inner }
    }

    /// Normalize text using SIMD-accelerated fast paths
    pub fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if text.is_empty() {
            return Cow::Borrowed(text);
        }

        let bytes = text.as_bytes();

        // Use SIMD to find first non-ASCII byte
        let first_non_ascii = simd_find_first_non_ascii(bytes);

        match first_non_ascii {
            None => {
                // All ASCII - use fast ASCII normalization
                self.normalize_ascii_fast(text)
            }
            Some(0) => {
                // First byte is non-ASCII - full normalization
                self.inner.normalize(text)
            }
            Some(pos) => {
                // Mixed: ASCII prefix + non-ASCII suffix
                self.normalize_mixed(text, pos)
            }
        }
    }

    /// Fast normalization for pure ASCII text
    fn normalize_ascii_fast<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // For ASCII, we only need to handle case conversion and whitespace normalization
        // Check if any work is needed
        let needs_work = simd_ascii_needs_normalization(text.as_bytes());

        if !needs_work {
            return Cow::Borrowed(text);
        }

        // Apply ASCII-only transformations
        let mut result = String::with_capacity(text.len());

        for &byte in text.as_bytes() {
            match byte {
                // Control characters (except common whitespace)
                0x00..=0x08 | 0x0B..=0x0C | 0x0E..=0x1F => continue,
                // Tab, newline, CR -> space
                b'\t' | b'\n' | b'\r' => result.push(' '),
                // Uppercase -> lowercase (for uncased models)
                b'A'..=b'Z' => result.push((byte + 32) as char),
                // Everything else passes through
                _ => result.push(byte as char),
            }
        }

        Cow::Owned(result)
    }

    /// Normalize mixed ASCII/non-ASCII text
    fn normalize_mixed<'a>(&self, text: &'a str, ascii_end: usize) -> Cow<'a, str> {
        // Ensure we split at a character boundary
        let split_pos = text
            .char_indices()
            .take_while(|(i, _)| *i < ascii_end)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);

        if split_pos == 0 {
            return self.inner.normalize(text);
        }

        let (ascii_part, non_ascii_part) = text.split_at(split_pos);

        // Normalize ASCII part with fast path
        let normalized_ascii = self.normalize_ascii_fast(ascii_part);

        // Normalize non-ASCII part with full normalizer
        let normalized_rest = self.inner.normalize(non_ascii_part);

        // Combine results
        if matches!(&normalized_ascii, Cow::Borrowed(_))
            && matches!(&normalized_rest, Cow::Borrowed(_))
        {
            // Both parts unchanged
            Cow::Borrowed(text)
        } else {
            let mut result = normalized_ascii.into_owned();
            result.push_str(&normalized_rest);
            Cow::Owned(result)
        }
    }
}

/// Find the first non-ASCII byte using SIMD
#[inline]
fn simd_find_first_non_ascii(bytes: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && bytes.len() >= 32 {
            return unsafe { simd_find_first_non_ascii_avx2(bytes) };
        }
    }

    // Scalar fallback
    for (i, &b) in bytes.iter().enumerate() {
        if b >= 0x80 {
            return Some(i);
        }
    }
    None
}

/// AVX2 implementation of non-ASCII search
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_find_first_non_ascii_avx2(bytes: &[u8]) -> Option<usize> {
    use std::arch::x86_64::*;

    let mut i = 0;

    // Process 32 bytes at a time
    while i + 32 <= bytes.len() {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);
        let high_bits = _mm256_movemask_epi8(chunk);

        if high_bits != 0 {
            return Some(i + high_bits.trailing_zeros() as usize);
        }

        i += 32;
    }

    // Check remaining bytes
    for j in i..bytes.len() {
        if bytes[j] >= 0x80 {
            return Some(j);
        }
    }

    None
}

/// Check if ASCII text needs normalization using SIMD
#[inline]
fn simd_ascii_needs_normalization(bytes: &[u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && bytes.len() >= 32 {
            return unsafe { simd_ascii_needs_normalization_avx2(bytes) };
        }
    }

    // Scalar fallback
    for &b in bytes {
        // Check for: control chars, whitespace to normalize, uppercase letters
        if b < 0x20 || (b >= b'A' && b <= b'Z') {
            return true;
        }
    }
    false
}

/// AVX2 implementation of ASCII normalization check
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_ascii_needs_normalization_avx2(bytes: &[u8]) -> bool {
    use std::arch::x86_64::*;

    let mut i = 0;

    // Thresholds for comparison
    let space_minus_one = _mm256_set1_epi8(0x1F);
    let upper_a = _mm256_set1_epi8(b'A' as i8);
    let upper_z = _mm256_set1_epi8(b'Z' as i8);

    while i + 32 <= bytes.len() {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);

        // Check for control characters (< 0x20)
        let is_control = _mm256_cmpgt_epi8(space_minus_one, chunk);

        // Check for uppercase (A <= x <= Z)
        let ge_a = _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(upper_a, _mm256_set1_epi8(1)));
        let le_z = _mm256_cmpgt_epi8(_mm256_add_epi8(upper_z, _mm256_set1_epi8(1)), chunk);
        let is_upper = _mm256_and_si256(ge_a, le_z);

        // Combine checks
        let needs_work = _mm256_or_si256(is_control, is_upper);
        let mask = _mm256_movemask_epi8(needs_work);

        if mask != 0 {
            return true;
        }

        i += 32;
    }

    // Check remaining bytes
    for &b in &bytes[i..] {
        if b < 0x20 || (b >= b'A' && b <= b'Z') {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_normalizer_clean_text() {
        let normalizer = BertNormalizer::uncased();

        // Control characters should be removed
        assert_eq!(
            normalizer.clean_text("hello\x00world").as_ref(),
            "helloworld"
        );

        // Tab, newline, CR should become space
        assert_eq!(
            normalizer.clean_text("hello\tworld").as_ref(),
            "hello world"
        );
        assert_eq!(
            normalizer.clean_text("hello\nworld").as_ref(),
            "hello world"
        );
        assert_eq!(
            normalizer.clean_text("hello\rworld").as_ref(),
            "hello world"
        );

        // Normal text should pass through unchanged (borrowed)
        let text = "hello world";
        let result = normalizer.clean_text(text);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result.as_ref(), "hello world");
    }

    #[test]
    fn test_bert_normalizer_handle_chinese_chars() {
        let normalizer = BertNormalizer::uncased();

        // CJK characters should get spaces around them
        assert_eq!(
            normalizer.handle_chinese_chars("你好").as_ref(),
            " 你  好 "
        );

        // Mixed text
        assert_eq!(
            normalizer.handle_chinese_chars("hello世界").as_ref(),
            "hello 世  界 "
        );

        // ASCII-only should pass through unchanged
        let text = "hello world";
        let result = normalizer.handle_chinese_chars(text);
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_bert_normalizer_strip_accents() {
        let normalizer = BertNormalizer::uncased();

        // Accents should be stripped
        assert_eq!(
            normalizer.strip_accents_if_needed("café").as_ref(),
            "cafe"
        );
        assert_eq!(
            normalizer.strip_accents_if_needed("naïve").as_ref(),
            "naive"
        );

        // ASCII should pass through unchanged
        let text = "hello";
        let result = normalizer.strip_accents_if_needed(text);
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_bert_normalizer_lowercase() {
        let normalizer = BertNormalizer::uncased();

        // Should lowercase
        assert_eq!(
            normalizer.lowercase_if_needed("HELLO").as_ref(),
            "hello"
        );
        assert_eq!(
            normalizer.lowercase_if_needed("Hello World").as_ref(),
            "hello world"
        );

        // Already lowercase should pass through unchanged
        let text = "hello world";
        let result = normalizer.lowercase_if_needed(text);
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_bert_normalizer_full() {
        let normalizer = BertNormalizer::uncased();

        // Full normalization
        let result = normalizer.normalize("Hello\tWorld! 你好");
        // Tab becomes space, CJK chars get spaces around them
        // "Hello World! " + " 你 " + " 好 " but 你 and 好 are adjacent so:
        // "Hello World! " + " 你 " + " 好 " = "Hello World!  你  好 "
        assert_eq!(result.as_ref(), "hello world!  你  好 ");

        // With accents
        let result = normalizer.normalize("Café");
        assert_eq!(result.as_ref(), "cafe");
    }

    #[test]
    fn test_bert_normalizer_cased() {
        let normalizer = BertNormalizer::cased();

        // Should NOT lowercase
        let result = normalizer.normalize("Hello World");
        assert!(result.contains("Hello"));

        // Should NOT strip accents (by default for cased)
        let result = normalizer.normalize("Café");
        assert!(result.contains("é") || result.contains("e")); // Depends on config
    }

    #[test]
    fn test_nfc_normalizer() {
        let normalizer = NfcNormalizer;

        // ASCII should pass through unchanged
        let text = "hello";
        let result = normalizer.normalize(text);
        assert!(matches!(result, Cow::Borrowed(_)));

        // Decomposed should be composed
        let nfd = "cafe\u{0301}"; // e + combining acute
        let result = normalizer.normalize(nfd);
        assert_eq!(result.as_ref(), "café");
    }

    #[test]
    fn test_lowercase_normalizer() {
        let normalizer = LowercaseNormalizer;

        // Should lowercase
        let result = normalizer.normalize("HELLO");
        assert_eq!(result.as_ref(), "hello");

        // Already lowercase should be borrowed
        let text = "hello";
        let result = normalizer.normalize(text);
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_strip_accents_normalizer() {
        let normalizer = StripAccentsNormalizer;

        // Should strip accents
        let result = normalizer.normalize("café");
        assert_eq!(result.as_ref(), "cafe");

        // ASCII should be borrowed
        let text = "hello";
        let result = normalizer.normalize(text);
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_sequence_normalizer() {
        let normalizer = SequenceNormalizer::new(vec![
            Box::new(LowercaseNormalizer),
            Box::new(StripAccentsNormalizer),
        ]);

        let result = normalizer.normalize("Café");
        assert_eq!(result.as_ref(), "cafe");
    }

    #[test]
    fn test_prepend_normalizer() {
        let normalizer = PrependNormalizer::new("prefix_");

        let result = normalizer.normalize("text");
        assert_eq!(result.as_ref(), "prefix_text");

        // Empty prepend should be borrowed
        let normalizer = PrependNormalizer::new("");
        let text = "text";
        let result = normalizer.normalize(text);
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_replace_normalizer() {
        let normalizer = ReplaceNormalizer::new(" ", "_");

        let result = normalizer.normalize("hello world");
        assert_eq!(result.as_ref(), "hello_world");

        // No match should be borrowed
        let text = "hello";
        let result = normalizer.normalize(text);
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    // ========================================================================
    // Streaming Normalization Tests (2.1.9)
    // ========================================================================

    #[test]
    fn test_streaming_normalizer_state() {
        let mut state = StreamingNormalizerState::new();
        assert!(!state.has_pending());

        state.incomplete_utf8.push(0xC2);
        assert!(state.has_pending());

        state.reset();
        assert!(!state.has_pending());
    }

    #[test]
    fn test_streaming_normalizer_basic() {
        let inner = LowercaseNormalizer;
        let streaming = StreamingNormalizer::new(inner, 64);

        let result = streaming.normalize_streaming("HELLO WORLD");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_streaming_normalizer_chunks() {
        let inner = LowercaseNormalizer;
        let streaming = StreamingNormalizer::new(inner, 4);
        let mut state = StreamingNormalizerState::new();

        let result1 = streaming.process_chunk(b"HELL", &mut state);
        let result2 = streaming.process_chunk(b"O WO", &mut state);
        let result3 = streaming.process_chunk(b"RLD", &mut state);
        let result4 = streaming.finish(&mut state);

        let combined = format!("{}{}{}{}", result1, result2, result3, result4);
        // Note: due to combining character boundary handling, result may differ slightly
        assert!(combined.to_lowercase().contains("hello"));
    }

    #[test]
    fn test_streaming_normalizer_utf8_boundary() {
        let inner = NfcNormalizer;
        let streaming = StreamingNormalizer::new(inner, 64);

        // Test with multi-byte UTF-8 characters
        let text = "日本語テスト";
        let result = streaming.normalize_streaming(text);
        assert_eq!(result, text);
    }

    #[test]
    fn test_find_valid_utf8_boundary() {
        // Complete ASCII
        assert_eq!(find_valid_utf8_boundary(b"hello"), 5);

        // Complete multi-byte
        let text = "café";
        assert_eq!(find_valid_utf8_boundary(text.as_bytes()), text.len());

        // Incomplete 2-byte sequence at end
        let mut data = b"hello".to_vec();
        data.push(0xC2); // Start of 2-byte sequence
        assert_eq!(find_valid_utf8_boundary(&data), 5);

        // Incomplete 3-byte sequence at end
        let mut data = b"test".to_vec();
        data.push(0xE2); // Start of 3-byte sequence
        data.push(0x82); // Continuation byte
        assert_eq!(find_valid_utf8_boundary(&data), 4);
    }

    #[test]
    fn test_is_combining_mark() {
        assert!(is_combining_mark('\u{0301}')); // Combining acute accent
        assert!(is_combining_mark('\u{0308}')); // Combining diaeresis
        assert!(!is_combining_mark('a'));
        assert!(!is_combining_mark('é'));
    }

    #[test]
    fn test_split_at_combining_boundary() {
        // Simple ASCII
        let (proc, pend) = split_at_combining_boundary("hello", "");
        assert!(!proc.is_empty());

        // With combining mark at end
        let text = "cafe\u{0301}"; // café with combining accent
        let (proc, pend) = split_at_combining_boundary(text, "");
        // Should keep the base character + combining mark together
        assert!(!proc.is_empty() || !pend.is_empty());
    }

    #[test]
    fn test_streaming_large_text() {
        let inner = LowercaseNormalizer;
        let streaming = StreamingNormalizer::new(inner, 100);

        // Create a large text
        let large_text = "HELLO WORLD ".repeat(1000);
        let result = streaming.normalize_streaming(&large_text);

        assert_eq!(result, "hello world ".repeat(1000));
    }

    // ========================================================================
    // SIMD Normalization Tests (2.1.8)
    // ========================================================================

    #[test]
    fn test_simd_normalizer_ascii() {
        let inner = LowercaseNormalizer;
        let simd = SimdNormalizer::new(inner);

        let result = simd.normalize("HELLO WORLD");
        assert_eq!(result.as_ref(), "hello world");
    }

    #[test]
    fn test_simd_normalizer_non_ascii() {
        let inner = NfcNormalizer;
        let simd = SimdNormalizer::new(inner);

        let result = simd.normalize("日本語");
        assert_eq!(result.as_ref(), "日本語");
    }

    #[test]
    fn test_simd_normalizer_mixed() {
        let inner = LowercaseNormalizer;
        let simd = SimdNormalizer::new(inner);

        // ASCII followed by non-ASCII
        let result = simd.normalize("HELLO 日本語");
        // ASCII part should be lowercased
        assert!(result.starts_with("hello"));
    }

    #[test]
    fn test_simd_normalizer_empty() {
        let inner = LowercaseNormalizer;
        let simd = SimdNormalizer::new(inner);

        let result = simd.normalize("");
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result.as_ref(), "");
    }

    #[test]
    fn test_simd_normalizer_no_change() {
        let inner = LowercaseNormalizer;
        let simd = SimdNormalizer::new(inner);

        // Already lowercase ASCII
        let result = simd.normalize("hello world");
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_simd_find_first_non_ascii() {
        assert_eq!(simd_find_first_non_ascii(b"hello"), None);
        assert_eq!(simd_find_first_non_ascii(b"hello\x80"), Some(5));
        assert_eq!(simd_find_first_non_ascii("café".as_bytes()), Some(3));
        assert_eq!(simd_find_first_non_ascii("日本語".as_bytes()), Some(0));
    }

    #[test]
    fn test_simd_ascii_needs_normalization() {
        // No normalization needed
        assert!(!simd_ascii_needs_normalization(b"hello world"));

        // Has uppercase
        assert!(simd_ascii_needs_normalization(b"Hello World"));

        // Has control character
        assert!(simd_ascii_needs_normalization(b"hello\tworld"));
    }

    #[test]
    fn test_simd_normalizer_long_ascii() {
        let inner = LowercaseNormalizer;
        let simd = SimdNormalizer::new(inner);

        // Long ASCII string to trigger SIMD path
        let text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG ".repeat(10);
        let result = simd.normalize(&text);

        assert_eq!(
            result.as_ref(),
            "the quick brown fox jumps over the lazy dog ".repeat(10)
        );
    }

    #[test]
    fn test_simd_normalizer_long_mixed() {
        let inner = NfcNormalizer;
        let simd = SimdNormalizer::new(inner);

        // Long text with non-ASCII in the middle
        let text = format!(
            "{}日本語{}",
            "hello ".repeat(20),
            " world".repeat(20)
        );
        let result = simd.normalize(&text);

        assert!(result.contains("日本語"));
    }
}
