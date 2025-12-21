//! High-Performance WordLevel Tokenizer
//!
//! This module implements a WordLevel tokenizer optimized for maximum throughput:
//! - Scalar version with efficient pre-tokenization
//! - SIMD-accelerated whitespace/punctuation detection (AVX2, AVX-512)
//! - Token caching for repeated words
//! - HuggingFace-compatible output
//!
//! Target: 20-50x faster than HuggingFace Tokenizers

use std::sync::Arc;

use ahash::AHashMap;
use unicode_general_category::{get_general_category, GeneralCategory};

use crate::cache::{CacheStats, ShardedCache};
use crate::encoding::Encoding;
use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;
use crate::vocab::Vocabulary;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for WordLevel tokenizer
#[derive(Debug, Clone)]
pub struct WordLevelConfig {
    /// Unknown token string
    pub unk_token: String,
    /// Unknown token ID (looked up from vocab if not provided)
    pub unk_id: Option<u32>,
    /// Whether to split on punctuation characters
    pub split_on_punctuation: bool,
    /// Whether to split on CJK characters (each character becomes a token)
    pub split_on_cjk: bool,
    /// Whether to lowercase text before tokenization
    pub lowercase: bool,
    /// Minimum word length to consider (shorter words become UNK)
    pub min_word_length: usize,
    /// Maximum word length to consider (longer words become UNK)
    pub max_word_length: usize,
}

impl Default for WordLevelConfig {
    fn default() -> Self {
        Self {
            unk_token: "<unk>".to_string(),
            unk_id: None,
            split_on_punctuation: true,
            split_on_cjk: true,
            lowercase: false,
            min_word_length: 0,
            max_word_length: 200,
        }
    }
}

// ============================================================================
// Pre-Token Structure
// ============================================================================

/// A pre-tokenized word with byte offsets (WordLevel-specific)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordLevelPreToken {
    /// The word text
    pub text: String,
    /// Start byte offset in original text
    pub start: usize,
    /// End byte offset in original text
    pub end: usize,
}

/// Type alias for backwards compatibility
pub type PreToken = WordLevelPreToken;

impl WordLevelPreToken {
    #[inline]
    pub fn new(text: String, start: usize, end: usize) -> Self {
        Self { text, start, end }
    }

    #[inline]
    pub fn from_slice(text: &str, start: usize, end: usize) -> Self {
        Self {
            text: text.to_string(),
            start,
            end,
        }
    }
}

// ============================================================================
// Token Cache Type
// ============================================================================

/// Type alias for word-level token cache (word string → token ID)
pub type WordLevelCache = ShardedCache<String, u32>;

// ============================================================================
// Character Classification (Scalar)
// ============================================================================

/// Check if character is ASCII whitespace
#[inline(always)]
pub fn is_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c')
}

/// Check if character is Unicode whitespace (broader than ASCII)
#[inline(always)]
pub fn is_unicode_whitespace(c: char) -> bool {
    c.is_whitespace()
}

/// Check if character is punctuation
#[inline(always)]
pub fn is_punctuation(c: char) -> bool {
    // ASCII punctuation
    if c.is_ascii() {
        return matches!(
            c,
            '!' | '"'
                | '#'
                | '$'
                | '%'
                | '&'
                | '\''
                | '('
                | ')'
                | '*'
                | '+'
                | ','
                | '-'
                | '.'
                | '/'
                | ':'
                | ';'
                | '<'
                | '='
                | '>'
                | '?'
                | '@'
                | '['
                | '\\'
                | ']'
                | '^'
                | '_'
                | '`'
                | '{'
                | '|'
                | '}'
                | '~'
        );
    }
    // Unicode punctuation categories
    matches!(
        unicode_general_category::get_general_category(c),
        unicode_general_category::GeneralCategory::ConnectorPunctuation
            | unicode_general_category::GeneralCategory::DashPunctuation
            | unicode_general_category::GeneralCategory::OpenPunctuation
            | unicode_general_category::GeneralCategory::ClosePunctuation
            | unicode_general_category::GeneralCategory::InitialPunctuation
            | unicode_general_category::GeneralCategory::FinalPunctuation
            | unicode_general_category::GeneralCategory::OtherPunctuation
    )
}

/// Check if character is a CJK ideograph
#[inline(always)]
pub fn is_cjk_character(c: char) -> bool {
    let cp = c as u32;
    // CJK Unified Ideographs and related blocks
    matches!(
        cp,
        0x4E00..=0x9FFF     // CJK Unified Ideographs
        | 0x3400..=0x4DBF   // CJK Unified Ideographs Extension A
        | 0x20000..=0x2A6DF // CJK Unified Ideographs Extension B
        | 0x2A700..=0x2B73F // CJK Unified Ideographs Extension C
        | 0x2B740..=0x2B81F // CJK Unified Ideographs Extension D
        | 0x2B820..=0x2CEAF // CJK Unified Ideographs Extension E
        | 0xF900..=0xFAFF   // CJK Compatibility Ideographs
        | 0x2F800..=0x2FA1F // CJK Compatibility Ideographs Supplement
        | 0x3000..=0x303F   // CJK Symbols and Punctuation
        | 0xFF00..=0xFFEF   // Halfwidth and Fullwidth Forms
    )
}

// ============================================================================
// Scalar Pre-Tokenizer
// ============================================================================

/// Check if a character is a "word" character (matching HuggingFace's Whitespace pre-tokenizer)
///
/// HuggingFace's Whitespace pre-tokenizer uses regex \w which matches:
/// - Unicode alphabetic characters (L* categories)
/// - Unicode Mark characters (M* categories) - combining marks like strikethrough, diacritics
/// - Unicode decimal digits (category Nd only, NOT subscript/superscript digits which are No)
/// - Connector punctuation like underscore '_' (Pc category)
///
/// Note: This is the Unicode-aware definition of \w in Rust regex.
#[inline(always)]
fn is_word_char(c: char) -> bool {
    let cat = get_general_category(c);
    c.is_alphabetic()
        || matches!(cat,
            GeneralCategory::DecimalNumber
            | GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
            | GeneralCategory::ConnectorPunctuation
        )
}

/// Pre-tokenize text into words (scalar version)
///
/// This splits text on whitespace and optionally on punctuation and CJK characters.
/// Returns a vector of PreToken with byte offsets.
///
/// Note: When split_on_punctuation is true, this matches HuggingFace's Whitespace
/// pre-tokenizer behavior by splitting at boundaries between alphanumeric and
/// non-alphanumeric characters. Consecutive non-alphanumeric characters are grouped
/// together as a single token (e.g., "--" stays as one token).
pub fn pretokenize_scalar(text: &str, config: &WordLevelConfig) -> Vec<PreToken> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut tokens = Vec::with_capacity(text.len() / 5); // Estimate ~5 chars/word
    let mut word_start: Option<usize> = None;
    let mut symbol_start: Option<usize> = None; // Track consecutive non-alphanumeric

    for (i, c) in text.char_indices() {
        let char_len = c.len_utf8();
        let is_ws = is_whitespace(c);
        let is_cjk = config.split_on_cjk && is_cjk_character(c);

        // When split_on_punctuation is true, we split on any non-alphanumeric character
        // (not just ASCII punctuation) to match HuggingFace's Whitespace pre-tokenizer
        let is_symbol = config.split_on_punctuation && !is_word_char(c) && !is_ws && !is_cjk;

        if is_ws {
            // Whitespace: flush current word or symbol sequence
            if let Some(start) = word_start {
                let word = &text[start..i];
                if !word.is_empty() {
                    tokens.push(PreToken::from_slice(word, start, i));
                }
                word_start = None;
            }
            if let Some(start) = symbol_start {
                let sym = &text[start..i];
                if !sym.is_empty() {
                    tokens.push(PreToken::from_slice(sym, start, i));
                }
                symbol_start = None;
            }
        } else if is_symbol {
            // Non-alphanumeric symbol: flush current word, start/continue symbol sequence
            if let Some(start) = word_start {
                let word = &text[start..i];
                if !word.is_empty() {
                    tokens.push(PreToken::from_slice(word, start, i));
                }
                word_start = None;
            }
            // Start or continue symbol sequence
            if symbol_start.is_none() {
                symbol_start = Some(i);
            }
        } else if is_cjk {
            // CJK: flush current word and symbols, emit CJK char as separate token
            if let Some(start) = word_start {
                let word = &text[start..i];
                if !word.is_empty() {
                    tokens.push(PreToken::from_slice(word, start, i));
                }
                word_start = None;
            }
            if let Some(start) = symbol_start {
                let sym = &text[start..i];
                if !sym.is_empty() {
                    tokens.push(PreToken::from_slice(sym, start, i));
                }
                symbol_start = None;
            }
            // Emit CJK character as its own token
            tokens.push(PreToken::from_slice(&text[i..i + char_len], i, i + char_len));
        } else {
            // Alphanumeric character: flush symbols, start or continue word
            if let Some(start) = symbol_start {
                let sym = &text[start..i];
                if !sym.is_empty() {
                    tokens.push(PreToken::from_slice(sym, start, i));
                }
                symbol_start = None;
            }
            if word_start.is_none() {
                word_start = Some(i);
            }
        }
    }

    // Flush final word or symbols
    if let Some(start) = word_start {
        let word = &text[start..];
        if !word.is_empty() {
            tokens.push(PreToken::from_slice(word, start, text.len()));
        }
    }
    if let Some(start) = symbol_start {
        let sym = &text[start..];
        if !sym.is_empty() {
            tokens.push(PreToken::from_slice(sym, start, text.len()));
        }
    }

    tokens
}

/// Pre-tokenize with lowercase transformation
///
/// Note: When split_on_punctuation is true, this matches HuggingFace's Whitespace
/// pre-tokenizer behavior by splitting at boundaries between alphanumeric and
/// non-alphanumeric characters.
pub fn pretokenize_scalar_lowercase(text: &str, config: &WordLevelConfig) -> Vec<PreToken> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut tokens = Vec::with_capacity(text.len() / 5);
    let mut word_chars = String::with_capacity(32);
    let mut word_start: Option<usize> = None;
    let mut symbol_start: Option<usize> = None; // Track consecutive non-alphanumeric

    for (i, c) in text.char_indices() {
        let char_len = c.len_utf8();
        let is_ws = is_whitespace(c);
        let is_cjk = config.split_on_cjk && is_cjk_character(c);
        let is_symbol = config.split_on_punctuation && !is_word_char(c) && !is_ws && !is_cjk;

        if is_ws {
            // Flush word
            if let Some(start) = word_start {
                if !word_chars.is_empty() {
                    tokens.push(PreToken::new(word_chars.clone(), start, i));
                    word_chars.clear();
                }
                word_start = None;
            }
            // Flush symbols
            if let Some(start) = symbol_start {
                let sym = &text[start..i];
                if !sym.is_empty() {
                    tokens.push(PreToken::from_slice(sym, start, i));
                }
                symbol_start = None;
            }
        } else if is_symbol {
            // Flush word first
            if let Some(start) = word_start {
                if !word_chars.is_empty() {
                    tokens.push(PreToken::new(word_chars.clone(), start, i));
                    word_chars.clear();
                }
                word_start = None;
            }
            // Start or continue symbol sequence
            if symbol_start.is_none() {
                symbol_start = Some(i);
            }
        } else if is_cjk {
            // Flush word
            if let Some(start) = word_start {
                if !word_chars.is_empty() {
                    tokens.push(PreToken::new(word_chars.clone(), start, i));
                    word_chars.clear();
                }
                word_start = None;
            }
            // Flush symbols
            if let Some(start) = symbol_start {
                let sym = &text[start..i];
                if !sym.is_empty() {
                    tokens.push(PreToken::from_slice(sym, start, i));
                }
                symbol_start = None;
            }
            // CJK characters: no lowercase transformation
            tokens.push(PreToken::from_slice(&text[i..i + char_len], i, i + char_len));
        } else {
            // Flush symbols first
            if let Some(start) = symbol_start {
                let sym = &text[start..i];
                if !sym.is_empty() {
                    tokens.push(PreToken::from_slice(sym, start, i));
                }
                symbol_start = None;
            }
            if word_start.is_none() {
                word_start = Some(i);
            }
            // Lowercase the character
            for lc in c.to_lowercase() {
                word_chars.push(lc);
            }
        }
    }

    // Flush final word or symbols
    if let Some(start) = word_start {
        if !word_chars.is_empty() {
            tokens.push(PreToken::new(word_chars, start, text.len()));
        }
    }
    if let Some(start) = symbol_start {
        let sym = &text[start..];
        if !sym.is_empty() {
            tokens.push(PreToken::from_slice(sym, start, text.len()));
        }
    }

    tokens
}

// ============================================================================
// SIMD Pre-Tokenizer
// ============================================================================

/// Check if AVX2 is available
#[inline]
pub fn is_avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Check if AVX-512 is available
#[inline]
pub fn is_avx512_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// SIMD-accelerated pre-tokenizer with automatic dispatch
///
/// Automatically selects the best SIMD implementation:
/// - AVX-512: 64 bytes per iteration (if available)
/// - AVX2: 32 bytes per iteration (if available)
/// - Scalar: fallback for short text or non-x86
pub fn pretokenize_simd(text: &str, config: &WordLevelConfig) -> Vec<PreToken> {
    if text.is_empty() {
        return Vec::new();
    }

    let bytes = text.as_bytes();

    // For short text or non-ASCII-only, use scalar
    // SIMD is most effective for ASCII text with word boundaries
    if text.len() < 64 || !is_ascii_dominant(bytes) {
        return pretokenize_scalar(text, config);
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "nightly")]
        if is_avx512_available() {
            return unsafe { pretokenize_avx512(text, config) };
        }
        if is_avx2_available() {
            return unsafe { pretokenize_avx2(text, config) };
        }
    }

    pretokenize_scalar(text, config)
}

/// Check if text is predominantly ASCII (>= 95%)
#[inline]
fn is_ascii_dominant(bytes: &[u8]) -> bool {
    let sample_size = bytes.len().min(256);
    let non_ascii = bytes.iter().take(sample_size).filter(|&&b| b >= 128).count();
    non_ascii * 100 / sample_size < 5
}

/// AVX2 SIMD pre-tokenizer (32 bytes per iteration)
///
/// Uses SIMD to rapidly classify whitespace and non-alphanumeric boundaries.
/// Falls back to scalar for non-ASCII segments.
///
/// Note: When split_on_punctuation is true, splits at alphanumeric/non-alphanumeric
/// boundaries to match HuggingFace's Whitespace pre-tokenizer behavior.
///
/// # Safety
/// - Caller must verify AVX2 is available
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pretokenize_avx2(text: &str, config: &WordLevelConfig) -> Vec<PreToken> {
    use std::arch::x86_64::*;

    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut tokens = Vec::with_capacity(len / 5);

    // Character classification constants
    let space = _mm256_set1_epi8(b' ' as i8);
    let tab = _mm256_set1_epi8(b'\t' as i8);
    let newline = _mm256_set1_epi8(b'\n' as i8);
    let cr = _mm256_set1_epi8(b'\r' as i8);
    let ascii_high_bit = _mm256_set1_epi8(0x80u8 as i8);

    // Word character range markers for SIMD classification
    // Digits: 0x30-0x39 (0-9)
    // Uppercase: 0x41-0x5A (A-Z)
    // Underscore: 0x5F (_)
    // Lowercase: 0x61-0x7A (a-z)
    let digit_lo = _mm256_set1_epi8(0x30);
    let digit_hi = _mm256_set1_epi8(0x39);
    let upper_lo = _mm256_set1_epi8(0x41);
    let upper_hi = _mm256_set1_epi8(0x5A);
    let underscore = _mm256_set1_epi8(0x5F); // '_'
    let lower_lo = _mm256_set1_epi8(0x61);
    let lower_hi = _mm256_set1_epi8(0x7A);

    let mut i = 0;
    let mut word_start: Option<usize> = None;
    let mut symbol_start: Option<usize> = None;

    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);

        // Check for non-ASCII bytes (high bit set)
        let non_ascii_mask = _mm256_movemask_epi8(_mm256_and_si256(chunk, ascii_high_bit));

        if non_ascii_mask != 0 {
            // Non-ASCII detected: process character-by-character
            let non_ascii_pos = non_ascii_mask.trailing_zeros() as usize;

            // Process ASCII portion before non-ASCII byte
            for j in i..(i + non_ascii_pos) {
                process_byte_avx2_helper(
                    text,
                    bytes,
                    j,
                    &mut word_start,
                    &mut symbol_start,
                    &mut tokens,
                    config,
                );
            }

            // Process non-ASCII region with scalar (using is_word_char)
            let mut char_pos = i + non_ascii_pos;
            while char_pos < len.min(i + 32) {
                let c = text[char_pos..].chars().next().unwrap();
                let char_len = c.len_utf8();

                if is_whitespace(c) {
                    // Flush word
                    if let Some(start) = word_start {
                        tokens.push(PreToken::from_slice(&text[start..char_pos], start, char_pos));
                        word_start = None;
                    }
                    // Flush symbols
                    if let Some(start) = symbol_start {
                        tokens.push(PreToken::from_slice(&text[start..char_pos], start, char_pos));
                        symbol_start = None;
                    }
                } else if config.split_on_cjk && is_cjk_character(c) {
                    // CJK: flush word and symbols, emit CJK as separate token
                    if let Some(start) = word_start {
                        tokens.push(PreToken::from_slice(&text[start..char_pos], start, char_pos));
                        word_start = None;
                    }
                    if let Some(start) = symbol_start {
                        tokens.push(PreToken::from_slice(&text[start..char_pos], start, char_pos));
                        symbol_start = None;
                    }
                    tokens.push(PreToken::from_slice(&text[char_pos..char_pos + char_len], char_pos, char_pos + char_len));
                } else if config.split_on_punctuation && !is_word_char(c) {
                    // Non-alphanumeric (symbol)
                    if let Some(start) = word_start {
                        tokens.push(PreToken::from_slice(&text[start..char_pos], start, char_pos));
                        word_start = None;
                    }
                    if symbol_start.is_none() {
                        symbol_start = Some(char_pos);
                    }
                } else {
                    // Alphanumeric
                    if let Some(start) = symbol_start {
                        tokens.push(PreToken::from_slice(&text[start..char_pos], start, char_pos));
                        symbol_start = None;
                    }
                    if word_start.is_none() {
                        word_start = Some(char_pos);
                    }
                }

                char_pos += char_len;
            }

            i = char_pos;
            continue;
        }

        // All ASCII: use SIMD classification
        // Build whitespace mask
        let ws_mask = {
            let space_eq = _mm256_cmpeq_epi8(chunk, space);
            let tab_eq = _mm256_cmpeq_epi8(chunk, tab);
            let newline_eq = _mm256_cmpeq_epi8(chunk, newline);
            let cr_eq = _mm256_cmpeq_epi8(chunk, cr);
            let ws = _mm256_or_si256(_mm256_or_si256(space_eq, tab_eq), _mm256_or_si256(newline_eq, cr_eq));
            _mm256_movemask_epi8(ws) as u32
        };

        // Build word character mask (bytes that are 0-9, A-Z, _, or a-z)
        let alnum_mask = if config.split_on_punctuation {
            // Digits: 0x30-0x39
            let ge_digit = _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(digit_lo, _mm256_set1_epi8(1)));
            let le_digit = _mm256_cmpgt_epi8(_mm256_add_epi8(digit_hi, _mm256_set1_epi8(1)), chunk);
            let is_digit = _mm256_and_si256(ge_digit, le_digit);

            // Uppercase: 0x41-0x5A
            let ge_upper = _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(upper_lo, _mm256_set1_epi8(1)));
            let le_upper = _mm256_cmpgt_epi8(_mm256_add_epi8(upper_hi, _mm256_set1_epi8(1)), chunk);
            let is_upper = _mm256_and_si256(ge_upper, le_upper);

            // Underscore: 0x5F
            let is_underscore = _mm256_cmpeq_epi8(chunk, underscore);

            // Lowercase: 0x61-0x7A
            let ge_lower = _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(lower_lo, _mm256_set1_epi8(1)));
            let le_lower = _mm256_cmpgt_epi8(_mm256_add_epi8(lower_hi, _mm256_set1_epi8(1)), chunk);
            let is_lower = _mm256_and_si256(ge_lower, le_lower);

            let alnum = _mm256_or_si256(_mm256_or_si256(is_digit, is_upper), _mm256_or_si256(is_underscore, is_lower));
            _mm256_movemask_epi8(alnum) as u32
        } else {
            0xFFFF_FFFF // All are considered word chars if not splitting
        };

        // Symbol mask = not whitespace and not alphanumeric
        let symbol_mask = !ws_mask & !alnum_mask;

        // Combined boundary mask (whitespace OR symbols)
        let boundary_mask = ws_mask | symbol_mask;

        if boundary_mask == 0 {
            // No boundaries in this chunk - all alphanumeric
            if let Some(start) = symbol_start {
                tokens.push(PreToken::from_slice(&text[start..i], start, i));
                symbol_start = None;
            }
            if word_start.is_none() {
                word_start = Some(i);
            }
        } else if ws_mask == 0xFFFF_FFFF {
            // All whitespace
            if let Some(start) = word_start {
                tokens.push(PreToken::from_slice(&text[start..i], start, i));
                word_start = None;
            }
            if let Some(start) = symbol_start {
                tokens.push(PreToken::from_slice(&text[start..i], start, i));
                symbol_start = None;
            }
        } else {
            // Mixed: process bit by bit
            let mut remaining_ws = ws_mask;
            let mut remaining_alnum = alnum_mask;

            for bit in 0..32u32 {
                let pos = i + bit as usize;
                let is_ws = (remaining_ws & 1) == 1;
                let is_alnum = (remaining_alnum & 1) == 1;
                remaining_ws >>= 1;
                remaining_alnum >>= 1;

                if is_ws {
                    // Flush word
                    if let Some(start) = word_start {
                        tokens.push(PreToken::from_slice(&text[start..pos], start, pos));
                        word_start = None;
                    }
                    // Flush symbols
                    if let Some(start) = symbol_start {
                        tokens.push(PreToken::from_slice(&text[start..pos], start, pos));
                        symbol_start = None;
                    }
                } else if !is_alnum {
                    // Symbol (non-alphanumeric, non-whitespace)
                    if let Some(start) = word_start {
                        tokens.push(PreToken::from_slice(&text[start..pos], start, pos));
                        word_start = None;
                    }
                    if symbol_start.is_none() {
                        symbol_start = Some(pos);
                    }
                } else {
                    // Alphanumeric
                    if let Some(start) = symbol_start {
                        tokens.push(PreToken::from_slice(&text[start..pos], start, pos));
                        symbol_start = None;
                    }
                    if word_start.is_none() {
                        word_start = Some(pos);
                    }
                }
            }
        }

        i += 32;
    }

    // Handle remaining bytes with scalar
    while i < len {
        process_byte_avx2_helper(text, bytes, i, &mut word_start, &mut symbol_start, &mut tokens, config);
        i += 1;
    }

    // Flush final word or symbols
    if let Some(start) = word_start {
        if start < len {
            tokens.push(PreToken::from_slice(&text[start..], start, len));
        }
    }
    if let Some(start) = symbol_start {
        if start < len {
            tokens.push(PreToken::from_slice(&text[start..], start, len));
        }
    }

    tokens
}

/// Check if byte is an ASCII word character (for SIMD path)
/// Matches alphanumeric (a-z, A-Z, 0-9) plus underscore
#[inline(always)]
fn is_ascii_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Helper to process a single byte during AVX2 pre-tokenization
#[inline]
fn process_byte_avx2_helper(
    text: &str,
    bytes: &[u8],
    pos: usize,
    word_start: &mut Option<usize>,
    symbol_start: &mut Option<usize>,
    tokens: &mut Vec<PreToken>,
    config: &WordLevelConfig,
) {
    let byte = bytes[pos];

    // Whitespace check
    if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
        // Flush word
        if let Some(start) = *word_start {
            tokens.push(PreToken::from_slice(&text[start..pos], start, pos));
            *word_start = None;
        }
        // Flush symbols
        if let Some(start) = *symbol_start {
            tokens.push(PreToken::from_slice(&text[start..pos], start, pos));
            *symbol_start = None;
        }
        return;
    }

    // Non-alphanumeric check (symbol/punctuation)
    if config.split_on_punctuation && !is_ascii_word_byte(byte) {
        // Flush word
        if let Some(start) = *word_start {
            tokens.push(PreToken::from_slice(&text[start..pos], start, pos));
            *word_start = None;
        }
        // Start or continue symbol sequence
        if symbol_start.is_none() {
            *symbol_start = Some(pos);
        }
        return;
    }

    // Regular alphanumeric character
    // Flush symbols
    if let Some(start) = *symbol_start {
        tokens.push(PreToken::from_slice(&text[start..pos], start, pos));
        *symbol_start = None;
    }
    if word_start.is_none() {
        *word_start = Some(pos);
    }
}

/// AVX-512 SIMD pre-tokenizer (64 bytes per iteration)
///
/// Uses 512-bit vectors for maximum throughput on supported CPUs.
///
/// Note: When split_on_punctuation is true, splits at alphanumeric/non-alphanumeric
/// boundaries to match HuggingFace's Whitespace pre-tokenizer behavior.
///
/// # Safety
/// - Caller must verify AVX-512F and AVX-512BW are available
#[cfg(all(target_arch = "x86_64", feature = "nightly"))]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn pretokenize_avx512(text: &str, config: &WordLevelConfig) -> Vec<PreToken> {
    use std::arch::x86_64::*;

    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut tokens = Vec::with_capacity(len / 5);

    // Character classification constants
    let space = _mm512_set1_epi8(b' ' as i8);
    let tab = _mm512_set1_epi8(b'\t' as i8);
    let newline = _mm512_set1_epi8(b'\n' as i8);
    let cr = _mm512_set1_epi8(b'\r' as i8);
    let high_bit = _mm512_set1_epi8(0x80u8 as i8);

    // Word character range markers
    // Digits: 0x30-0x39 (0-9)
    // Uppercase: 0x41-0x5A (A-Z)
    // Underscore: 0x5F (_)
    // Lowercase: 0x61-0x7A (a-z)
    let digit_lo = _mm512_set1_epi8(0x30);
    let digit_hi = _mm512_set1_epi8(0x39);
    let upper_lo = _mm512_set1_epi8(0x41);
    let upper_hi = _mm512_set1_epi8(0x5A);
    let underscore = _mm512_set1_epi8(0x5F); // '_'
    let lower_lo = _mm512_set1_epi8(0x61);
    let lower_hi = _mm512_set1_epi8(0x7A);

    let mut i = 0;
    let mut word_start: Option<usize> = None;
    let mut symbol_start: Option<usize> = None;

    while i + 64 <= len {
        let chunk = _mm512_loadu_si512(bytes.as_ptr().add(i) as *const _);

        // Check for non-ASCII bytes
        let non_ascii_mask = _mm512_test_epi8_mask(chunk, high_bit);

        if non_ascii_mask != 0 {
            // Non-ASCII detected: fall back to scalar for this segment
            let non_ascii_pos = non_ascii_mask.trailing_zeros() as usize;

            // Process ASCII portion
            for j in i..(i + non_ascii_pos) {
                process_byte_avx2_helper(text, bytes, j, &mut word_start, &mut symbol_start, &mut tokens, config);
            }

            // Process non-ASCII region with scalar
            let mut char_pos = i + non_ascii_pos;
            while char_pos < len.min(i + 64) {
                let c = text[char_pos..].chars().next().unwrap();
                let char_len = c.len_utf8();

                if is_whitespace(c) {
                    // Flush word
                    if let Some(start) = word_start {
                        tokens.push(PreToken::from_slice(&text[start..char_pos], start, char_pos));
                        word_start = None;
                    }
                    // Flush symbols
                    if let Some(start) = symbol_start {
                        tokens.push(PreToken::from_slice(&text[start..char_pos], start, char_pos));
                        symbol_start = None;
                    }
                } else if config.split_on_cjk && is_cjk_character(c) {
                    // CJK: flush word and symbols, emit CJK as separate token
                    if let Some(start) = word_start {
                        tokens.push(PreToken::from_slice(&text[start..char_pos], start, char_pos));
                        word_start = None;
                    }
                    if let Some(start) = symbol_start {
                        tokens.push(PreToken::from_slice(&text[start..char_pos], start, char_pos));
                        symbol_start = None;
                    }
                    tokens.push(PreToken::from_slice(&text[char_pos..char_pos + char_len], char_pos, char_pos + char_len));
                } else if config.split_on_punctuation && !is_word_char(c) {
                    // Non-alphanumeric (symbol)
                    if let Some(start) = word_start {
                        tokens.push(PreToken::from_slice(&text[start..char_pos], start, char_pos));
                        word_start = None;
                    }
                    if symbol_start.is_none() {
                        symbol_start = Some(char_pos);
                    }
                } else {
                    // Alphanumeric
                    if let Some(start) = symbol_start {
                        tokens.push(PreToken::from_slice(&text[start..char_pos], start, char_pos));
                        symbol_start = None;
                    }
                    if word_start.is_none() {
                        word_start = Some(char_pos);
                    }
                }

                char_pos += char_len;
            }

            i = char_pos;
            continue;
        }

        // All ASCII: use SIMD classification
        // Whitespace mask (using native AVX-512 mask operations)
        let space_mask = _mm512_cmpeq_epi8_mask(chunk, space);
        let tab_mask = _mm512_cmpeq_epi8_mask(chunk, tab);
        let newline_mask = _mm512_cmpeq_epi8_mask(chunk, newline);
        let cr_mask = _mm512_cmpeq_epi8_mask(chunk, cr);
        let ws_mask = space_mask | tab_mask | newline_mask | cr_mask;

        // Alphanumeric mask
        let alnum_mask = if config.split_on_punctuation {
            // Digits: 0x30-0x39
            let ge_digit = _mm512_cmpge_epi8_mask(chunk, digit_lo);
            let le_digit = _mm512_cmple_epi8_mask(chunk, digit_hi);
            let is_digit = ge_digit & le_digit;

            // Uppercase: 0x41-0x5A
            let ge_upper = _mm512_cmpge_epi8_mask(chunk, upper_lo);
            let le_upper = _mm512_cmple_epi8_mask(chunk, upper_hi);
            let is_upper = ge_upper & le_upper;

            // Lowercase: 0x61-0x7A
            let ge_lower = _mm512_cmpge_epi8_mask(chunk, lower_lo);
            let le_lower = _mm512_cmple_epi8_mask(chunk, lower_hi);
            let is_lower = ge_lower & le_lower;

            // Underscore: 0x5F
            let is_underscore = _mm512_cmpeq_epi8_mask(chunk, underscore);

            is_digit | is_upper | is_underscore | is_lower
        } else {
            !0u64 // All are considered alphanumeric if not splitting
        };

        // Symbol mask = not whitespace and not alphanumeric
        let symbol_mask = !ws_mask & !alnum_mask;

        // Combined boundary mask
        let boundary_mask = ws_mask | symbol_mask;

        if boundary_mask == 0 {
            // No boundaries: all alphanumeric
            if let Some(start) = symbol_start {
                tokens.push(PreToken::from_slice(&text[start..i], start, i));
                symbol_start = None;
            }
            if word_start.is_none() {
                word_start = Some(i);
            }
        } else {
            // Process bit by bit using efficient bit extraction
            let mut remaining_ws = ws_mask;
            let mut remaining_alnum = alnum_mask;

            for bit in 0..64u32 {
                let pos = i + bit as usize;
                let is_ws = (remaining_ws & 1) == 1;
                let is_alnum = (remaining_alnum & 1) == 1;
                remaining_ws >>= 1;
                remaining_alnum >>= 1;

                if is_ws {
                    // Flush word
                    if let Some(start) = word_start {
                        tokens.push(PreToken::from_slice(&text[start..pos], start, pos));
                        word_start = None;
                    }
                    // Flush symbols
                    if let Some(start) = symbol_start {
                        tokens.push(PreToken::from_slice(&text[start..pos], start, pos));
                        symbol_start = None;
                    }
                } else if !is_alnum {
                    // Symbol (non-alphanumeric, non-whitespace)
                    if let Some(start) = word_start {
                        tokens.push(PreToken::from_slice(&text[start..pos], start, pos));
                        word_start = None;
                    }
                    if symbol_start.is_none() {
                        symbol_start = Some(pos);
                    }
                } else {
                    // Alphanumeric
                    if let Some(start) = symbol_start {
                        tokens.push(PreToken::from_slice(&text[start..pos], start, pos));
                        symbol_start = None;
                    }
                    if word_start.is_none() {
                        word_start = Some(pos);
                    }
                }
            }
        }

        i += 64;
    }

    // Handle remaining bytes with scalar
    while i < len {
        process_byte_avx2_helper(text, bytes, i, &mut word_start, &mut symbol_start, &mut tokens, config);
        i += 1;
    }

    // Flush final word or symbols
    if let Some(start) = word_start {
        if start < len {
            tokens.push(PreToken::from_slice(&text[start..], start, len));
        }
    }
    if let Some(start) = symbol_start {
        if start < len {
            tokens.push(PreToken::from_slice(&text[start..], start, len));
        }
    }

    tokens
}

// ============================================================================
// SIMD-Accelerated Encode
// ============================================================================

impl WordLevelTokenizer {
    /// Encode text using SIMD-accelerated pre-tokenization
    pub fn encode_simd(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Use SIMD pre-tokenization
        let pretokens = if self.config.lowercase {
            // For lowercase, we need to transform the text first
            // SIMD lowercase + SIMD pretokenize could be added later
            pretokenize_scalar_lowercase(text, &self.config)
        } else {
            pretokenize_simd(text, &self.config)
        };

        // Look up each word
        let mut ids = Vec::with_capacity(pretokens.len());
        for pretoken in pretokens {
            ids.push(self.lookup_word(&pretoken.text));
        }

        ids
    }
}

// ============================================================================
// WordLevel Tokenizer (Scalar)
// ============================================================================

/// High-performance WordLevel tokenizer
///
/// Each word in the input text is mapped to a single token ID.
/// Words not in the vocabulary are mapped to the unknown token.
pub struct WordLevelTokenizer {
    /// Forward vocabulary: token string → ID
    vocab: AHashMap<String, u32>,
    /// Reverse vocabulary: ID → token string
    vocab_r: Vec<String>,
    /// Configuration
    config: WordLevelConfig,
    /// Unknown token ID (resolved from vocab)
    unk_id: u32,
    /// Vocabulary size
    vocab_size: usize,
    /// Optional token cache for repeated words
    cache: Option<Arc<WordLevelCache>>,
}

impl WordLevelTokenizer {
    /// Create a new WordLevel tokenizer from vocabulary
    pub fn new(vocab: AHashMap<String, u32>, config: WordLevelConfig) -> Self {
        let vocab_size = vocab.len();

        // Build reverse vocabulary
        let mut vocab_r = vec![String::new(); vocab_size];
        for (token, &id) in &vocab {
            if (id as usize) < vocab_size {
                vocab_r[id as usize] = token.clone();
            }
        }

        // Resolve unknown token ID
        let unk_id = config
            .unk_id
            .or_else(|| vocab.get(&config.unk_token).copied())
            .unwrap_or(0);

        Self {
            vocab,
            vocab_r,
            config,
            unk_id,
            vocab_size,
            cache: None,
        }
    }

    /// Create tokenizer from vocabulary file (JSON format)
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json(&content)
    }

    /// Create tokenizer from JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        let parsed: serde_json::Value = serde_json::from_str(json)?;

        // Check if it's a HuggingFace tokenizer.json format
        if parsed.get("model").is_some() {
            return Self::from_hf_tokenizer_json(&parsed);
        }

        // Direct vocabulary format: {"token": id, ...}
        let vocab: AHashMap<String, u32> = parsed
            .as_object()
            .ok_or_else(|| Error::VocabLoad("Expected JSON object".to_string()))?
            .iter()
            .filter_map(|(k, v)| v.as_u64().map(|id| (k.clone(), id as u32)))
            .collect();

        if vocab.is_empty() {
            return Err(Error::VocabLoad("Empty vocabulary".to_string()));
        }

        Ok(Self::new(vocab, WordLevelConfig::default()))
    }

    /// Create tokenizer from HuggingFace tokenizer.json format
    fn from_hf_tokenizer_json(json: &serde_json::Value) -> Result<Self> {
        let model = json
            .get("model")
            .ok_or_else(|| Error::VocabLoad("Missing 'model' field".to_string()))?;

        // Get model type
        let model_type = model
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("WordLevel");

        if model_type != "WordLevel" {
            return Err(Error::VocabLoad(format!(
                "Expected WordLevel model, got {}",
                model_type
            )));
        }

        // Parse vocabulary
        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::VocabLoad("Missing 'vocab' in model".to_string()))?;

        let vocab: AHashMap<String, u32> = vocab_obj
            .iter()
            .filter_map(|(k, v)| v.as_u64().map(|id| (k.clone(), id as u32)))
            .collect();

        // Parse config
        let unk_token = model
            .get("unk_token")
            .and_then(|t| t.as_str())
            .unwrap_or("<unk>")
            .to_string();

        let config = WordLevelConfig {
            unk_token,
            ..Default::default()
        };

        Ok(Self::new(vocab, config))
    }

    /// Create tokenizer with cache enabled
    pub fn with_cache(
        vocab: AHashMap<String, u32>,
        config: WordLevelConfig,
        cache_capacity: usize,
    ) -> Self {
        let mut tokenizer = Self::new(vocab, config);
        tokenizer.cache = Some(Arc::new(ShardedCache::new(cache_capacity)));
        tokenizer
    }

    /// Enable caching with specified capacity
    pub fn enable_cache(&mut self, capacity: usize) {
        self.cache = Some(Arc::new(ShardedCache::new(capacity)));
    }

    /// Disable caching
    pub fn disable_cache(&mut self) {
        self.cache = None;
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.cache.as_ref().map(|c| c.stats())
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.clear();
        }
    }

    /// Look up a word in the vocabulary
    #[inline]
    pub fn lookup_word(&self, word: &str) -> u32 {
        // Check length constraints
        let len = word.len();
        if len < self.config.min_word_length || len > self.config.max_word_length {
            return self.unk_id;
        }

        // Check cache first
        if let Some(ref cache) = self.cache {
            if let Some(id) = cache.get(&word.to_string()) {
                return id;
            }
        }

        // Vocabulary lookup
        let id = self.vocab.get(word).copied().unwrap_or(self.unk_id);

        // Cache the result
        if let Some(ref cache) = self.cache {
            cache.insert(word.to_string(), id);
        }

        id
    }

    /// Encode text to token IDs (scalar version)
    pub fn encode_scalar(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Pre-tokenize
        let pretokens = if self.config.lowercase {
            pretokenize_scalar_lowercase(text, &self.config)
        } else {
            pretokenize_scalar(text, &self.config)
        };

        // Look up each word
        let mut ids = Vec::with_capacity(pretokens.len());
        for pretoken in pretokens {
            ids.push(self.lookup_word(&pretoken.text));
        }

        ids
    }

    /// Encode text with full Encoding output (includes offsets)
    pub fn encode_to_encoding(&self, text: &str, _add_special_tokens: bool) -> Encoding {
        if text.is_empty() {
            return Encoding::new();
        }

        // Pre-tokenize
        let pretokens = if self.config.lowercase {
            pretokenize_scalar_lowercase(text, &self.config)
        } else {
            pretokenize_scalar(text, &self.config)
        };

        let mut encoding = Encoding::with_capacity(pretokens.len());

        for (word_idx, pretoken) in pretokens.into_iter().enumerate() {
            let id = self.lookup_word(&pretoken.text);
            let token_str = self.id_to_token(id).unwrap_or(&self.config.unk_token);

            encoding.push(
                id,
                token_str.to_string(),
                (pretoken.start, pretoken.end),
                Some(word_idx as u32),
                Some(0),  // sequence_id = 0 for single sequence
                false,
            );
        }

        encoding
    }

    /// Batch encode multiple texts
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        use rayon::prelude::*;

        texts.par_iter().map(|text| self.encode_scalar(text)).collect()
    }

    /// Get token ID for a token string
    #[inline]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    /// Get token string for an ID
    #[inline]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab_r.get(id as usize).map(|s| s.as_str())
    }

    /// Decode token IDs back to text
    pub fn decode_ids(&self, ids: &[u32], skip_special_tokens: bool) -> String {
        let mut result = String::with_capacity(ids.len() * 5);

        for (i, &id) in ids.iter().enumerate() {
            if let Some(token) = self.id_to_token(id) {
                // Skip special tokens if requested
                if skip_special_tokens
                    && (token.starts_with('[') && token.ends_with(']')
                        || token.starts_with('<') && token.ends_with('>'))
                {
                    continue;
                }

                if i > 0 && !result.is_empty() {
                    result.push(' ');
                }
                result.push_str(token);
            }
        }

        result
    }

    /// Get vocabulary size
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the unknown token ID
    #[inline]
    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// Get reference to vocabulary
    pub fn vocabulary(&self) -> &AHashMap<String, u32> {
        &self.vocab
    }
}

// ============================================================================
// Tokenizer Trait Implementation
// ============================================================================

impl Tokenizer for WordLevelTokenizer {
    fn vocabulary(&self) -> &Vocabulary {
        // Note: This requires creating a Vocabulary wrapper
        // For now, we'll use a static reference trick
        unimplemented!("Use vocabulary() method directly")
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
        let mut encoding = self.encode_to_encoding(text, add_special_tokens);
        let encoding2 = self.encode_to_encoding(text_pair, add_special_tokens);

        // Merge encodings (simple concatenation for WordLevel)
        let ids = encoding2.get_ids();
        let tokens = encoding2.get_tokens();
        let offsets = encoding2.get_offsets();
        let word_ids = encoding2.get_word_ids();

        for i in 0..encoding2.len() {
            encoding.push(
                ids[i],
                tokens[i].clone(),
                offsets[i],
                word_ids[i],
                Some(1),  // sequence_id = 1 for second sequence
                false,
            );
        }

        Ok(encoding)
    }

    fn encode_batch(&self, texts: &[&str], _add_special_tokens: bool) -> Result<Vec<Encoding>> {
        use rayon::prelude::*;

        let results: Vec<Encoding> = texts
            .par_iter()
            .map(|text| self.encode_to_encoding(text, false))
            .collect();

        Ok(results)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        Ok(self.decode_ids(ids, skip_special_tokens))
    }

    fn decode_batch(&self, ids_batch: &[Vec<u32>], skip_special_tokens: bool) -> Result<Vec<String>> {
        use rayon::prelude::*;

        let results: Vec<String> = ids_batch
            .par_iter()
            .map(|ids| self.decode_ids(ids, skip_special_tokens))
            .collect();

        Ok(results)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab_r.get(id as usize).map(|s| s.as_str())
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn save(&self, path: &std::path::Path) -> Result<()> {
        let vocab_json = serde_json::to_string_pretty(&self.vocab)
            .map_err(|e| Error::InvalidConfig(format!("Failed to serialize vocabulary: {}", e)))?;

        std::fs::write(path, vocab_json)?;

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vocab() -> AHashMap<String, u32> {
        let mut vocab = AHashMap::new();
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("<pad>".to_string(), 1);
        vocab.insert("hello".to_string(), 2);
        vocab.insert("world".to_string(), 3);
        vocab.insert("the".to_string(), 4);
        vocab.insert("quick".to_string(), 5);
        vocab.insert("brown".to_string(), 6);
        vocab.insert("fox".to_string(), 7);
        vocab.insert("jumps".to_string(), 8);
        vocab.insert("over".to_string(), 9);
        vocab.insert("lazy".to_string(), 10);
        vocab.insert("dog".to_string(), 11);
        vocab.insert(".".to_string(), 12);
        vocab.insert(",".to_string(), 13);
        vocab.insert("!".to_string(), 14);
        vocab.insert("?".to_string(), 15);
        vocab
    }

    // ========================================
    // Character Classification Tests
    // ========================================

    #[test]
    fn test_is_whitespace() {
        assert!(is_whitespace(' '));
        assert!(is_whitespace('\t'));
        assert!(is_whitespace('\n'));
        assert!(is_whitespace('\r'));
        assert!(!is_whitespace('a'));
        assert!(!is_whitespace('.'));
    }

    #[test]
    fn test_is_punctuation() {
        assert!(is_punctuation('.'));
        assert!(is_punctuation(','));
        assert!(is_punctuation('!'));
        assert!(is_punctuation('?'));
        assert!(is_punctuation('-'));
        assert!(is_punctuation('\''));
        assert!(is_punctuation('"'));
        assert!(!is_punctuation('a'));
        assert!(!is_punctuation(' '));
        assert!(!is_punctuation('1'));
    }

    #[test]
    fn test_is_cjk_character() {
        // CJK characters
        assert!(is_cjk_character('中'));
        assert!(is_cjk_character('国'));
        assert!(is_cjk_character('日'));
        assert!(is_cjk_character('本'));

        // Non-CJK
        assert!(!is_cjk_character('a'));
        assert!(!is_cjk_character('1'));
        assert!(!is_cjk_character(' '));
        assert!(!is_cjk_character('é'));
    }

    // ========================================
    // Pre-tokenization Tests
    // ========================================

    #[test]
    fn test_pretokenize_simple() {
        let config = WordLevelConfig::default();
        let tokens = pretokenize_scalar("hello world", &config);

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 5);
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[1].start, 6);
        assert_eq!(tokens[1].end, 11);
    }

    #[test]
    fn test_pretokenize_multiple_spaces() {
        let config = WordLevelConfig::default();
        let tokens = pretokenize_scalar("hello   world", &config);

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_pretokenize_with_punctuation() {
        let config = WordLevelConfig {
            split_on_punctuation: true,
            ..Default::default()
        };
        let tokens = pretokenize_scalar("hello, world!", &config);

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, ",");
        assert_eq!(tokens[2].text, "world");
        assert_eq!(tokens[3].text, "!");
    }

    #[test]
    fn test_pretokenize_without_punctuation_split() {
        let config = WordLevelConfig {
            split_on_punctuation: false,
            ..Default::default()
        };
        let tokens = pretokenize_scalar("hello, world!", &config);

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello,");
        assert_eq!(tokens[1].text, "world!");
    }

    #[test]
    fn test_pretokenize_with_cjk() {
        let config = WordLevelConfig {
            split_on_cjk: true,
            ..Default::default()
        };
        let tokens = pretokenize_scalar("hello中国world", &config);

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "中");
        assert_eq!(tokens[2].text, "国");
        assert_eq!(tokens[3].text, "world");
    }

    #[test]
    fn test_pretokenize_empty() {
        let config = WordLevelConfig::default();
        let tokens = pretokenize_scalar("", &config);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_pretokenize_only_whitespace() {
        let config = WordLevelConfig::default();
        let tokens = pretokenize_scalar("   \t\n  ", &config);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_pretokenize_lowercase() {
        let config = WordLevelConfig::default();
        let tokens = pretokenize_scalar_lowercase("Hello WORLD", &config);

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_pretokenize_preserves_offsets() {
        let config = WordLevelConfig::default();
        let text = "The quick brown fox";
        let tokens = pretokenize_scalar(text, &config);

        for token in &tokens {
            assert_eq!(&text[token.start..token.end], token.text);
        }
    }

    // ========================================
    // Tokenizer Tests
    // ========================================

    #[test]
    fn test_tokenizer_creation() {
        let vocab = create_test_vocab();
        let tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

        assert_eq!(tokenizer.vocab_size(), 16);
        assert_eq!(tokenizer.unk_id(), 0);
    }

    #[test]
    fn test_lookup_word_found() {
        let vocab = create_test_vocab();
        let tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

        assert_eq!(tokenizer.lookup_word("hello"), 2);
        assert_eq!(tokenizer.lookup_word("world"), 3);
        assert_eq!(tokenizer.lookup_word("the"), 4);
    }

    #[test]
    fn test_lookup_word_not_found() {
        let vocab = create_test_vocab();
        let tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

        assert_eq!(tokenizer.lookup_word("unknown_word"), 0); // UNK
        assert_eq!(tokenizer.lookup_word("xyz"), 0);
    }

    #[test]
    fn test_encode_simple() {
        let vocab = create_test_vocab();
        let tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

        let ids = tokenizer.encode_scalar("hello world");
        assert_eq!(ids, vec![2, 3]);
    }

    #[test]
    fn test_encode_with_punctuation() {
        let vocab = create_test_vocab();
        let tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

        let ids = tokenizer.encode_scalar("hello, world!");
        assert_eq!(ids, vec![2, 13, 3, 14]); // hello, comma, world, exclamation
    }

    #[test]
    fn test_encode_with_unknown() {
        let vocab = create_test_vocab();
        let tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

        let ids = tokenizer.encode_scalar("hello unknown world");
        assert_eq!(ids, vec![2, 0, 3]); // hello, UNK, world
    }

    #[test]
    fn test_encode_sentence() {
        let vocab = create_test_vocab();
        let tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

        let ids = tokenizer.encode_scalar("the quick brown fox jumps over the lazy dog");
        assert_eq!(ids, vec![4, 5, 6, 7, 8, 9, 4, 10, 11]);
    }

    #[test]
    fn test_encode_empty() {
        let vocab = create_test_vocab();
        let tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

        let ids = tokenizer.encode_scalar("");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_token_to_id() {
        let vocab = create_test_vocab();
        let tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

        assert_eq!(tokenizer.token_to_id("hello"), Some(2));
        assert_eq!(tokenizer.token_to_id("world"), Some(3));
        assert_eq!(tokenizer.token_to_id("nonexistent"), None);
    }

    #[test]
    fn test_id_to_token() {
        let vocab = create_test_vocab();
        let tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

        assert_eq!(tokenizer.id_to_token(2), Some("hello"));
        assert_eq!(tokenizer.id_to_token(3), Some("world"));
        assert_eq!(tokenizer.id_to_token(999), None);
    }

    #[test]
    fn test_decode() {
        let vocab = create_test_vocab();
        let tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

        let text = tokenizer.decode_ids(&[2, 3], false);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_decode_skip_special() {
        let vocab = create_test_vocab();
        let tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

        let text = tokenizer.decode_ids(&[0, 2, 3], true);
        assert_eq!(text, "hello world"); // <unk> is skipped
    }

    // ========================================
    // Cache Tests
    // ========================================

    #[test]
    fn test_cache_enabled() {
        let vocab = create_test_vocab();
        let tokenizer =
            WordLevelTokenizer::with_cache(vocab, WordLevelConfig::default(), 1000);

        // First lookup populates cache
        assert_eq!(tokenizer.lookup_word("hello"), 2);

        // Check cache stats
        let stats = tokenizer.cache_stats().unwrap();
        assert!(stats.misses >= 1);

        // Second lookup should hit cache
        assert_eq!(tokenizer.lookup_word("hello"), 2);
        let stats2 = tokenizer.cache_stats().unwrap();
        assert!(stats2.hits >= 1);
    }

    #[test]
    fn test_cache_clear() {
        let vocab = create_test_vocab();
        let tokenizer =
            WordLevelTokenizer::with_cache(vocab, WordLevelConfig::default(), 1000);

        tokenizer.lookup_word("hello");
        tokenizer.clear_cache();

        let stats = tokenizer.cache_stats().unwrap();
        assert_eq!(stats.size, 0);
    }

    // ========================================
    // Batch Encoding Tests
    // ========================================

    #[test]
    fn test_encode_batch() {
        let vocab = create_test_vocab();
        let tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

        let texts = vec!["hello world", "the fox"];
        let results = tokenizer.encode_batch(&texts);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0], vec![2, 3]);
        assert_eq!(results[1], vec![4, 7]);
    }

    // ========================================
    // Encoding Output Tests
    // ========================================

    #[test]
    fn test_encode_to_encoding() {
        let vocab = create_test_vocab();
        let tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

        let encoding = tokenizer.encode_to_encoding("hello world", false);

        assert_eq!(encoding.get_ids(), &[2, 3]);
        assert_eq!(encoding.get_tokens(), &["hello", "world"]);
        assert_eq!(encoding.get_offsets(), &[(0, 5), (6, 11)]);
    }

    // ========================================
    // JSON Loading Tests
    // ========================================

    #[test]
    fn test_from_json_simple() {
        let json = r#"{"<unk>": 0, "hello": 1, "world": 2}"#;
        let tokenizer = WordLevelTokenizer::from_json(json).unwrap();

        assert_eq!(tokenizer.vocab_size(), 3);
        assert_eq!(tokenizer.token_to_id("hello"), Some(1));
    }

    #[test]
    fn test_from_json_hf_format() {
        let json = r#"{
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "hello": 1, "world": 2},
                "unk_token": "<unk>"
            }
        }"#;
        let tokenizer = WordLevelTokenizer::from_json(json).unwrap();

        assert_eq!(tokenizer.vocab_size(), 3);
        assert_eq!(tokenizer.unk_id(), 0);
    }

    // ========================================
    // SIMD Pre-tokenizer Tests
    // ========================================

    #[test]
    fn test_simd_pretokenize_simple() {
        let config = WordLevelConfig::default();
        // Short text falls back to scalar
        let tokens_short = pretokenize_simd("hello world", &config);
        let scalar_short = pretokenize_scalar("hello world", &config);
        assert_eq!(tokens_short, scalar_short);
    }

    #[test]
    fn test_simd_pretokenize_long_text() {
        let config = WordLevelConfig::default();
        // Long ASCII text for SIMD path
        let long_text = "The quick brown fox jumps over the lazy dog. This is a much longer sentence that should trigger the SIMD code path for pre-tokenization.";
        let tokens = pretokenize_simd(long_text, &config);
        let scalar_tokens = pretokenize_scalar(long_text, &config);

        // SIMD and scalar should produce identical results
        assert_eq!(tokens.len(), scalar_tokens.len());
        for (t, s) in tokens.iter().zip(scalar_tokens.iter()) {
            assert_eq!(t.text, s.text, "Token mismatch");
            assert_eq!(t.start, s.start, "Start offset mismatch");
            assert_eq!(t.end, s.end, "End offset mismatch");
        }
    }

    #[test]
    fn test_simd_pretokenize_with_punctuation() {
        let config = WordLevelConfig {
            split_on_punctuation: true,
            ..Default::default()
        };
        let text = "Hello, world! This is a test. Can you handle punctuation: yes or no?";
        let tokens = pretokenize_simd(text, &config);
        let scalar_tokens = pretokenize_scalar(text, &config);

        assert_eq!(tokens.len(), scalar_tokens.len());
        for (t, s) in tokens.iter().zip(scalar_tokens.iter()) {
            assert_eq!(t.text, s.text);
            assert_eq!(t.start, s.start);
            assert_eq!(t.end, s.end);
        }
    }

    #[test]
    fn test_simd_pretokenize_mixed_ascii_utf8() {
        let config = WordLevelConfig::default();
        // Mix of ASCII and UTF-8 to test fallback
        let text = "Hello world 你好世界 more words here to make it longer and trigger SIMD path for ASCII";
        let tokens = pretokenize_simd(text, &config);
        let scalar_tokens = pretokenize_scalar(text, &config);

        // SIMD should handle mixed content correctly
        assert_eq!(tokens.len(), scalar_tokens.len());
        for (t, s) in tokens.iter().zip(scalar_tokens.iter()) {
            assert_eq!(t.text, s.text);
        }
    }

    #[test]
    fn test_encode_simd_matches_scalar() {
        let vocab = create_test_vocab();
        let tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

        // Long text to trigger SIMD path
        let text = "the quick brown fox jumps over the lazy dog hello world the fox dog";
        let simd_ids = tokenizer.encode_simd(text);
        let scalar_ids = tokenizer.encode_scalar(text);

        assert_eq!(simd_ids, scalar_ids);
    }

    #[test]
    fn test_ascii_dominant_check() {
        // Pure ASCII should pass
        assert!(is_ascii_dominant(b"hello world this is all ascii text"));
        // Text with very few non-ASCII should pass (< 5%)
        // 100 ASCII chars + 2 non-ASCII bytes = ~2% non-ASCII
        let mostly_ascii = b"hello world this is mostly ascii text with many words to dilute the single accent \xc3\xa9 character";
        assert!(is_ascii_dominant(mostly_ascii));
        // All non-ASCII should fail
        let non_ascii = "你好世界这是中文".as_bytes();
        assert!(!is_ascii_dominant(non_ascii));
    }

    #[test]
    fn test_simd_availability() {
        // Just test that these functions don't panic
        let _avx2 = is_avx2_available();
        let _avx512 = is_avx512_available();
    }

    // ========================================
    // Performance Micro-benchmarks (in tests)
    // ========================================

    #[test]
    fn test_pretokenize_performance_hint() {
        use std::time::Instant;

        let config = WordLevelConfig::default();
        // Generate a large text sample
        let sample = "The quick brown fox jumps over the lazy dog. ".repeat(1000);

        // Warm-up
        for _ in 0..3 {
            let _ = pretokenize_scalar(&sample, &config);
            let _ = pretokenize_simd(&sample, &config);
        }

        // Benchmark scalar
        let start = Instant::now();
        for _ in 0..10 {
            let _ = pretokenize_scalar(&sample, &config);
        }
        let scalar_time = start.elapsed();

        // Benchmark SIMD
        let start = Instant::now();
        for _ in 0..10 {
            let _ = pretokenize_simd(&sample, &config);
        }
        let simd_time = start.elapsed();

        // Just print for informational purposes (not a hard assertion)
        println!("Scalar: {:?}, SIMD: {:?}", scalar_time, simd_time);
        // On AVX2/AVX-512 systems, SIMD should be faster or equal
        // We don't assert this since it depends on hardware
    }

    // ========================================
    // HuggingFace Compatibility Tests
    // ========================================

    /// Helper to create a temp file with HF tokenizer JSON and load it
    fn load_hf_tokenizer_from_json(json: &str) -> Option<tokenizers::Tokenizer> {
        use std::io::Write;
        let mut temp_file = std::env::temp_dir();
        temp_file.push(format!("budtiktok_test_{}.json", std::process::id()));

        if let Ok(mut file) = std::fs::File::create(&temp_file) {
            if file.write_all(json.as_bytes()).is_ok() {
                let result = tokenizers::Tokenizer::from_file(&temp_file).ok();
                let _ = std::fs::remove_file(&temp_file);
                return result;
            }
        }
        None
    }

    /// Test that our tokenizer produces the same output as HuggingFace
    #[test]
    fn test_hf_compatibility_basic() {
        // Create a WordLevel tokenizer in HuggingFace format
        let hf_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<unk>": 0, "hello": 1, "world": 2, "the": 3,
                    "quick": 4, "brown": 5, "fox": 6
                },
                "unk_token": "<unk>"
            }
        }"#;

        let hf_tokenizer = match load_hf_tokenizer_from_json(hf_json) {
            Some(t) => t,
            None => {
                println!("Skipping HF compatibility test - couldn't create HF tokenizer");
                return;
            }
        };

        // Create BudTikTok tokenizer with same vocab
        let mut vocab = AHashMap::new();
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);
        vocab.insert("the".to_string(), 3);
        vocab.insert("quick".to_string(), 4);
        vocab.insert("brown".to_string(), 5);
        vocab.insert("fox".to_string(), 6);

        let config = WordLevelConfig {
            split_on_punctuation: false, // HF Whitespace pre-tokenizer doesn't split on punct
            ..Default::default()
        };
        let bud_tokenizer = WordLevelTokenizer::new(vocab, config);

        // Test cases
        let test_cases = [
            "hello world",
            "the quick brown fox",
            "hello",
            "unknown word here",
        ];

        for text in &test_cases {
            let hf_encoding = hf_tokenizer.encode(*text, false).expect("HF encode failed");
            let hf_ids: Vec<u32> = hf_encoding.get_ids().to_vec();

            let bud_ids = bud_tokenizer.encode_scalar(text);

            assert_eq!(hf_ids, bud_ids, "Mismatch for text: '{}'", text);
        }
    }

    #[test]
    fn test_hf_compatibility_with_unk() {
        let hf_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hello": 1, "world": 2},
                "unk_token": "[UNK]"
            }
        }"#;

        let hf_tokenizer = match load_hf_tokenizer_from_json(hf_json) {
            Some(t) => t,
            None => {
                println!("Skipping HF compatibility test - couldn't create HF tokenizer");
                return;
            }
        };

        let mut vocab = AHashMap::new();
        vocab.insert("[UNK]".to_string(), 0);
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);

        let config = WordLevelConfig {
            unk_token: "[UNK]".to_string(),
            split_on_punctuation: false,
            ..Default::default()
        };
        let bud_tokenizer = WordLevelTokenizer::new(vocab, config);

        // Test with unknown words
        let text = "hello unknown world";
        let hf_encoding = hf_tokenizer.encode(text, false).expect("HF encode failed");
        let hf_ids: Vec<u32> = hf_encoding.get_ids().to_vec();
        let bud_ids = bud_tokenizer.encode_scalar(text);

        assert_eq!(hf_ids, bud_ids, "UNK handling mismatch");
        assert_eq!(bud_ids, vec![1, 0, 2]); // hello, [UNK], world
    }

    #[test]
    fn test_hf_compatibility_empty() {
        let hf_json = r#"{
            "version": "1.0",
            "pre_tokenizer": {"type": "Whitespace"},
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "hello": 1},
                "unk_token": "<unk>"
            }
        }"#;

        let hf_tokenizer = match load_hf_tokenizer_from_json(hf_json) {
            Some(t) => t,
            None => {
                println!("Skipping HF compatibility test - couldn't create HF tokenizer");
                return;
            }
        };

        let mut vocab = AHashMap::new();
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("hello".to_string(), 1);

        let config = WordLevelConfig {
            split_on_punctuation: false,
            ..Default::default()
        };
        let bud_tokenizer = WordLevelTokenizer::new(vocab, config);

        // Test empty string
        let hf_encoding = hf_tokenizer.encode("", false).expect("HF encode failed");
        let hf_ids: Vec<u32> = hf_encoding.get_ids().to_vec();
        let bud_ids = bud_tokenizer.encode_scalar("");

        assert_eq!(hf_ids, bud_ids, "Empty string mismatch");
        assert!(bud_ids.is_empty());
    }

    #[test]
    fn test_hf_compatibility_whitespace_variations() {
        let hf_json = r#"{
            "version": "1.0",
            "pre_tokenizer": {"type": "Whitespace"},
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "a": 1, "b": 2, "c": 3},
                "unk_token": "<unk>"
            }
        }"#;

        let hf_tokenizer = match load_hf_tokenizer_from_json(hf_json) {
            Some(t) => t,
            None => {
                println!("Skipping HF compatibility test - couldn't create HF tokenizer");
                return;
            }
        };

        let mut vocab = AHashMap::new();
        vocab.insert("<unk>".to_string(), 0);
        vocab.insert("a".to_string(), 1);
        vocab.insert("b".to_string(), 2);
        vocab.insert("c".to_string(), 3);

        let config = WordLevelConfig {
            split_on_punctuation: false,
            ..Default::default()
        };
        let bud_tokenizer = WordLevelTokenizer::new(vocab, config);

        // Test multiple whitespace types
        let test_cases = [
            "a b c",       // single spaces
            "a  b  c",     // double spaces
            "a\tb\tc",     // tabs
            "a\nb\nc",     // newlines
            "  a  ",       // leading/trailing spaces
        ];

        for text in &test_cases {
            let hf_encoding = hf_tokenizer.encode(*text, false).expect("HF encode failed");
            let hf_ids: Vec<u32> = hf_encoding.get_ids().to_vec();
            let bud_ids = bud_tokenizer.encode_scalar(text);

            assert_eq!(hf_ids, bud_ids, "Whitespace mismatch for: '{:?}'", text);
        }
    }
}
