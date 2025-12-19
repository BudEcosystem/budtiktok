//! Pre-tokenization for splitting text into words
//!
//! Pre-tokenizers split input text into smaller units (usually words) before
//! the actual tokenization algorithm is applied. This module provides:
//! - BERT pre-tokenizer: Split on whitespace and punctuation
//! - Whitespace pre-tokenizer: Split on whitespace only
//! - Metaspace pre-tokenizer: Replace spaces with special character
//! - Byte-level pre-tokenizer: For GPT-2 style tokenization

use crate::unicode::{is_cjk_character, is_punctuation, is_whitespace};

/// A span of text with its byte offsets
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreToken {
    /// The text content
    pub text: String,
    /// Start byte offset in original text
    pub start: usize,
    /// End byte offset in original text
    pub end: usize,
}

impl PreToken {
    /// Create a new pre-token
    pub fn new(text: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            text: text.into(),
            start,
            end,
        }
    }
}

/// Trait for pre-tokenizers
pub trait PreTokenizer: Send + Sync {
    /// Pre-tokenize the input text into a sequence of pre-tokens
    fn pre_tokenize(&self, text: &str) -> Vec<PreToken>;
}

/// BERT-style pre-tokenizer
///
/// Splits text on:
/// - Whitespace characters
/// - Punctuation characters (each punctuation becomes its own token)
/// - CJK characters (each CJK character becomes its own token)
#[derive(Debug, Clone, Default)]
pub struct BertPreTokenizer {
    /// Whether to handle CJK characters specially
    pub handle_chinese_chars: bool,
}

impl BertPreTokenizer {
    /// Create a new BERT pre-tokenizer
    pub fn new() -> Self {
        Self {
            handle_chinese_chars: true,
        }
    }

    /// Create without CJK handling
    pub fn without_chinese_chars() -> Self {
        Self {
            handle_chinese_chars: false,
        }
    }
}

impl PreTokenizer for BertPreTokenizer {
    fn pre_tokenize(&self, text: &str) -> Vec<PreToken> {
        let mut tokens = Vec::new();
        let mut current_word = String::new();
        let mut word_start = 0;

        for (i, c) in text.char_indices() {
            let char_len = c.len_utf8();

            if is_whitespace(c) {
                // Whitespace ends current word
                if !current_word.is_empty() {
                    tokens.push(PreToken::new(
                        std::mem::take(&mut current_word),
                        word_start,
                        i,
                    ));
                }
                word_start = i + char_len;
            } else if is_punctuation(c) {
                // Punctuation ends current word and is its own token
                if !current_word.is_empty() {
                    tokens.push(PreToken::new(
                        std::mem::take(&mut current_word),
                        word_start,
                        i,
                    ));
                }
                tokens.push(PreToken::new(c.to_string(), i, i + char_len));
                word_start = i + char_len;
            } else if self.handle_chinese_chars && is_cjk_character(c) {
                // CJK character ends current word and is its own token
                if !current_word.is_empty() {
                    tokens.push(PreToken::new(
                        std::mem::take(&mut current_word),
                        word_start,
                        i,
                    ));
                }
                tokens.push(PreToken::new(c.to_string(), i, i + char_len));
                word_start = i + char_len;
            } else {
                // Regular character - add to current word
                if current_word.is_empty() {
                    word_start = i;
                }
                current_word.push(c);
            }
        }

        // Don't forget the last word
        if !current_word.is_empty() {
            tokens.push(PreToken::new(current_word, word_start, text.len()));
        }

        tokens
    }
}

/// Whitespace pre-tokenizer
///
/// Simply splits on whitespace characters.
#[derive(Debug, Clone, Default)]
pub struct WhitespacePreTokenizer;

impl WhitespacePreTokenizer {
    pub fn new() -> Self {
        Self
    }
}

impl PreTokenizer for WhitespacePreTokenizer {
    fn pre_tokenize(&self, text: &str) -> Vec<PreToken> {
        let mut tokens = Vec::new();
        let mut current_word = String::new();
        let mut word_start = 0;

        for (i, c) in text.char_indices() {
            let char_len = c.len_utf8();

            if is_whitespace(c) {
                if !current_word.is_empty() {
                    tokens.push(PreToken::new(
                        std::mem::take(&mut current_word),
                        word_start,
                        i,
                    ));
                }
                word_start = i + char_len;
            } else {
                if current_word.is_empty() {
                    word_start = i;
                }
                current_word.push(c);
            }
        }

        if !current_word.is_empty() {
            tokens.push(PreToken::new(current_word, word_start, text.len()));
        }

        tokens
    }
}

/// Metaspace pre-tokenizer (used by SentencePiece/Unigram)
///
/// Replaces spaces with the special metaspace character (U+2581 = ▁)
/// and optionally adds a prefix space.
#[derive(Debug, Clone)]
pub struct MetaspacePreTokenizer {
    /// The metaspace replacement character (default: ▁)
    pub replacement: char,
    /// Whether to prepend the replacement to the string
    pub prepend_scheme: PrependScheme,
    /// Whether to split on the metaspace character
    pub split: bool,
}

/// How to prepend the metaspace character
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrependScheme {
    /// Never prepend
    Never,
    /// Always prepend
    Always,
    /// Prepend only to first word
    First,
}

impl Default for MetaspacePreTokenizer {
    fn default() -> Self {
        Self {
            replacement: '▁',
            prepend_scheme: PrependScheme::Always,
            split: true,
        }
    }
}

impl MetaspacePreTokenizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_replacement(mut self, replacement: char) -> Self {
        self.replacement = replacement;
        self
    }

    pub fn with_prepend_scheme(mut self, scheme: PrependScheme) -> Self {
        self.prepend_scheme = scheme;
        self
    }

    pub fn with_split(mut self, split: bool) -> Self {
        self.split = split;
        self
    }

    /// Transform text by replacing spaces with metaspace
    fn transform(&self, text: &str, is_first: bool) -> String {
        let mut result = String::with_capacity(text.len() + 1);

        // Prepend metaspace if needed
        let should_prepend = match self.prepend_scheme {
            PrependScheme::Never => false,
            PrependScheme::Always => !text.starts_with(' '),
            PrependScheme::First => is_first && !text.starts_with(' '),
        };

        if should_prepend {
            result.push(self.replacement);
        }

        // Replace spaces with metaspace
        for c in text.chars() {
            if c == ' ' {
                result.push(self.replacement);
            } else {
                result.push(c);
            }
        }

        result
    }
}

impl PreTokenizer for MetaspacePreTokenizer {
    fn pre_tokenize(&self, text: &str) -> Vec<PreToken> {
        if !self.split {
            // Just transform, don't split
            let transformed = self.transform(text, true);
            return vec![PreToken::new(transformed, 0, text.len())];
        }

        // Split on whitespace, transform each piece
        let mut tokens = Vec::new();
        let mut is_first = true;

        for (i, word) in text.split_whitespace().enumerate() {
            // Find the actual byte position in the original text
            let start = if i == 0 {
                text.find(word).unwrap_or(0)
            } else {
                // This is a simplification; proper offset tracking would be more complex
                text[tokens.last().map(|t: &PreToken| t.end).unwrap_or(0)..]
                    .find(word)
                    .map(|pos| tokens.last().map(|t: &PreToken| t.end).unwrap_or(0) + pos)
                    .unwrap_or(0)
            };
            let end = start + word.len();

            let transformed = self.transform(word, is_first);
            tokens.push(PreToken::new(transformed, start, end));
            is_first = false;
        }

        tokens
    }
}

/// Byte-level pre-tokenizer (GPT-2 style)
///
/// Maps bytes to printable characters and splits on whitespace.
/// Ensures all tokens can be safely displayed.
#[derive(Debug, Clone)]
pub struct ByteLevelPreTokenizer {
    /// Whether to add a prefix space
    pub add_prefix_space: bool,
    /// Whether to trim offsets to exclude prefix space
    pub trim_offsets: bool,
    /// Whether to use the standard GPT-2 byte mapping
    pub use_regex: bool,
}

impl Default for ByteLevelPreTokenizer {
    fn default() -> Self {
        Self {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        }
    }
}

impl ByteLevelPreTokenizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the byte-to-char mapping (GPT-2 style)
    fn byte_to_char(byte: u8) -> char {
        match byte {
            // Printable ASCII (0x21-0x7E) and some extended (0xA1-0xFF)
            0x21..=0x7E | 0xA1..=0xFF => byte as char,
            // Map other bytes to Unicode range starting at U+0100
            b => {
                // Count how many bytes come before this one that need mapping
                let offset = match b {
                    0x00..=0x20 => b,               // First 33 bytes (0-32)
                    0x7F..=0xA0 => b - 0x7F + 33,   // Next 34 bytes (127-160)
                    _ => unreachable!(),
                };
                char::from_u32(0x100 + offset as u32).unwrap()
            }
        }
    }

    /// Convert bytes to the GPT-2 character representation
    fn bytes_to_chars(bytes: &[u8]) -> String {
        bytes.iter().map(|&b| Self::byte_to_char(b)).collect()
    }
}

impl PreTokenizer for ByteLevelPreTokenizer {
    fn pre_tokenize(&self, text: &str) -> Vec<PreToken> {
        let text = if self.add_prefix_space && !text.starts_with(' ') {
            format!(" {}", text)
        } else {
            text.to_string()
        };

        // For GPT-2 style, we split on whitespace but keep leading spaces with words
        if !self.use_regex {
            // Simple whitespace splitting
            let whitespace = WhitespacePreTokenizer::new();
            let pretokens = whitespace.pre_tokenize(&text);
            return pretokens
                .into_iter()
                .map(|pt| {
                    let chars = Self::bytes_to_chars(pt.text.as_bytes());
                    PreToken::new(chars, pt.start, pt.end)
                })
                .collect();
        }

        // GPT-2 style: keep leading space with following word
        let mut tokens = Vec::new();
        let mut current_word = String::new();
        let mut word_start = 0;
        let mut in_word = false;

        for (i, c) in text.char_indices() {
            let char_len = c.len_utf8();

            if c.is_whitespace() {
                if in_word {
                    // End of word - emit the word
                    let bytes = current_word.as_bytes();
                    let chars = Self::bytes_to_chars(bytes);
                    tokens.push(PreToken::new(chars, word_start, i));
                    current_word.clear();
                    in_word = false;
                }
                // Start new word with this space
                word_start = i;
                current_word.push(c);
            } else {
                if !in_word && current_word.is_empty() {
                    word_start = i;
                }
                current_word.push(c);
                in_word = true;
            }
        }

        // Last word
        if !current_word.is_empty() {
            let bytes = current_word.as_bytes();
            let chars = Self::bytes_to_chars(bytes);
            tokens.push(PreToken::new(chars, word_start, text.len()));
        }

        tokens
    }
}

/// Split pre-tokenizer - splits on a specific pattern
#[derive(Debug, Clone)]
pub struct SplitPreTokenizer {
    /// Pattern to split on
    pub pattern: String,
    /// Behavior when pattern is found
    pub behavior: SplitBehavior,
    /// Whether to invert the pattern (keep delimiter, remove rest)
    pub invert: bool,
}

/// How to handle the delimiter when splitting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitBehavior {
    #[default]
    /// Remove the delimiter
    Removed,
    /// Keep delimiter as separate token
    Isolated,
    /// Merge delimiter with previous token
    MergedWithPrevious,
    /// Merge delimiter with next token
    MergedWithNext,
}

impl SplitPreTokenizer {
    pub fn new(pattern: impl Into<String>, behavior: SplitBehavior) -> Self {
        Self {
            pattern: pattern.into(),
            behavior,
            invert: false,
        }
    }

    pub fn with_invert(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }
}

impl PreTokenizer for SplitPreTokenizer {
    fn pre_tokenize(&self, text: &str) -> Vec<PreToken> {
        if self.pattern.is_empty() {
            return vec![PreToken::new(text.to_string(), 0, text.len())];
        }

        let mut tokens = Vec::new();
        let mut last_end = 0;

        for (start, part) in text.match_indices(&self.pattern) {
            // Add text before the match
            if start > last_end {
                tokens.push(PreToken::new(
                    text[last_end..start].to_string(),
                    last_end,
                    start,
                ));
            }

            // Handle the delimiter based on behavior
            match self.behavior {
                SplitBehavior::Removed => {
                    // Don't add delimiter
                }
                SplitBehavior::Isolated => {
                    tokens.push(PreToken::new(part.to_string(), start, start + part.len()));
                }
                SplitBehavior::MergedWithPrevious => {
                    if let Some(last) = tokens.last_mut() {
                        last.text.push_str(part);
                        last.end = start + part.len();
                    } else {
                        tokens.push(PreToken::new(part.to_string(), start, start + part.len()));
                    }
                }
                SplitBehavior::MergedWithNext => {
                    // Will be handled by adding it to the start of next token
                    // For now, just track it
                    tokens.push(PreToken::new(part.to_string(), start, start + part.len()));
                }
            }

            last_end = start + part.len();
        }

        // Add remaining text
        if last_end < text.len() {
            tokens.push(PreToken::new(
                text[last_end..].to_string(),
                last_end,
                text.len(),
            ));
        }

        tokens
    }
}

/// Punctuation pre-tokenizer - isolates punctuation
#[derive(Debug, Clone, Copy, Default)]
pub struct PunctuationPreTokenizer {
    /// Behavior for punctuation
    pub behavior: SplitBehavior,
}

impl PunctuationPreTokenizer {
    pub fn new(behavior: SplitBehavior) -> Self {
        Self { behavior }
    }
}

impl PreTokenizer for PunctuationPreTokenizer {
    fn pre_tokenize(&self, text: &str) -> Vec<PreToken> {
        let mut tokens = Vec::new();
        let mut current_word = String::new();
        let mut word_start = 0;

        for (i, c) in text.char_indices() {
            let char_len = c.len_utf8();

            if is_punctuation(c) {
                // Handle current word
                if !current_word.is_empty() {
                    tokens.push(PreToken::new(
                        std::mem::take(&mut current_word),
                        word_start,
                        i,
                    ));
                }

                // Handle punctuation based on behavior
                match self.behavior {
                    SplitBehavior::Removed => {}
                    SplitBehavior::Isolated => {
                        tokens.push(PreToken::new(c.to_string(), i, i + char_len));
                    }
                    SplitBehavior::MergedWithPrevious => {
                        if let Some(last) = tokens.last_mut() {
                            last.text.push(c);
                            last.end = i + char_len;
                        } else {
                            tokens.push(PreToken::new(c.to_string(), i, i + char_len));
                        }
                    }
                    SplitBehavior::MergedWithNext => {
                        current_word.push(c);
                        word_start = i;
                    }
                }

                if !matches!(self.behavior, SplitBehavior::MergedWithNext) {
                    word_start = i + char_len;
                }
            } else {
                if current_word.is_empty() {
                    word_start = i;
                }
                current_word.push(c);
            }
        }

        if !current_word.is_empty() {
            tokens.push(PreToken::new(current_word, word_start, text.len()));
        }

        tokens
    }
}

/// Digits pre-tokenizer - isolates digits
#[derive(Debug, Clone, Copy, Default)]
pub struct DigitsPreTokenizer {
    /// Whether to split individual digits
    pub individual_digits: bool,
}

impl DigitsPreTokenizer {
    pub fn new(individual_digits: bool) -> Self {
        Self { individual_digits }
    }
}

impl PreTokenizer for DigitsPreTokenizer {
    fn pre_tokenize(&self, text: &str) -> Vec<PreToken> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let mut current_start = 0;
        let mut in_digits = false;

        for (i, c) in text.char_indices() {
            let char_len = c.len_utf8();
            let is_digit = c.is_ascii_digit();

            if is_digit != in_digits || (self.individual_digits && is_digit) {
                // State change or individual digit mode
                if !current.is_empty() {
                    tokens.push(PreToken::new(
                        std::mem::take(&mut current),
                        current_start,
                        i,
                    ));
                }
                current_start = i;
                in_digits = is_digit;
            }

            current.push(c);
        }

        if !current.is_empty() {
            tokens.push(PreToken::new(current, current_start, text.len()));
        }

        tokens
    }
}

/// Sequence pre-tokenizer - applies multiple pre-tokenizers in order
pub struct SequencePreTokenizer {
    pretokenizers: Vec<Box<dyn PreTokenizer>>,
}

impl SequencePreTokenizer {
    pub fn new(pretokenizers: Vec<Box<dyn PreTokenizer>>) -> Self {
        Self { pretokenizers }
    }
}

impl PreTokenizer for SequencePreTokenizer {
    fn pre_tokenize(&self, text: &str) -> Vec<PreToken> {
        let mut tokens = vec![PreToken::new(text.to_string(), 0, text.len())];

        for pretokenizer in &self.pretokenizers {
            let mut new_tokens = Vec::new();
            for token in tokens {
                let sub_tokens = pretokenizer.pre_tokenize(&token.text);
                for mut sub in sub_tokens {
                    // Adjust offsets
                    sub.start += token.start;
                    sub.end = sub.start + sub.text.len();
                    new_tokens.push(sub);
                }
            }
            tokens = new_tokens;
        }

        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_pretokenizer_basic() {
        let pretokenizer = BertPreTokenizer::new();

        let tokens = pretokenizer.pre_tokenize("hello world");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_bert_pretokenizer_punctuation() {
        let pretokenizer = BertPreTokenizer::new();

        let tokens = pretokenizer.pre_tokenize("hello, world!");
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, ",");
        assert_eq!(tokens[2].text, "world");
        assert_eq!(tokens[3].text, "!");
    }

    #[test]
    fn test_bert_pretokenizer_cjk() {
        let pretokenizer = BertPreTokenizer::new();

        let tokens = pretokenizer.pre_tokenize("hello世界");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "世");
        assert_eq!(tokens[2].text, "界");
    }

    #[test]
    fn test_bert_pretokenizer_offsets() {
        let pretokenizer = BertPreTokenizer::new();

        let text = "hello world";
        let tokens = pretokenizer.pre_tokenize(text);

        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens[0].end, 5);
        assert_eq!(&text[tokens[0].start..tokens[0].end], "hello");

        assert_eq!(tokens[1].start, 6);
        assert_eq!(tokens[1].end, 11);
        assert_eq!(&text[tokens[1].start..tokens[1].end], "world");
    }

    #[test]
    fn test_whitespace_pretokenizer() {
        let pretokenizer = WhitespacePreTokenizer::new();

        let tokens = pretokenizer.pre_tokenize("hello  world\tfoo");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
        assert_eq!(tokens[2].text, "foo");
    }

    #[test]
    fn test_metaspace_pretokenizer() {
        let pretokenizer = MetaspacePreTokenizer::new();

        let tokens = pretokenizer.pre_tokenize("hello world");
        assert_eq!(tokens.len(), 2);
        assert!(tokens[0].text.starts_with('▁'));
        assert!(tokens[1].text.starts_with('▁'));
    }

    #[test]
    fn test_metaspace_pretokenizer_prepend_first() {
        let pretokenizer = MetaspacePreTokenizer::new().with_prepend_scheme(PrependScheme::First);

        let tokens = pretokenizer.pre_tokenize("hello world");
        assert_eq!(tokens.len(), 2);
        assert!(tokens[0].text.starts_with('▁')); // First word has prefix
        assert!(!tokens[1].text.starts_with('▁')); // Second word doesn't
    }

    #[test]
    fn test_byte_level_pretokenizer() {
        let pretokenizer = ByteLevelPreTokenizer::new();

        // Simple test
        let tokens = pretokenizer.pre_tokenize("hello");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_punctuation_pretokenizer() {
        let pretokenizer = PunctuationPreTokenizer::new(SplitBehavior::Isolated);

        let tokens = pretokenizer.pre_tokenize("hello,world");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, ",");
        assert_eq!(tokens[2].text, "world");
    }

    #[test]
    fn test_digits_pretokenizer() {
        let pretokenizer = DigitsPreTokenizer::new(false);

        let tokens = pretokenizer.pre_tokenize("hello123world");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "123");
        assert_eq!(tokens[2].text, "world");
    }

    #[test]
    fn test_digits_pretokenizer_individual() {
        let pretokenizer = DigitsPreTokenizer::new(true);

        let tokens = pretokenizer.pre_tokenize("a12b");
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "a");
        assert_eq!(tokens[1].text, "1");
        assert_eq!(tokens[2].text, "2");
        assert_eq!(tokens[3].text, "b");
    }

    #[test]
    fn test_sequence_pretokenizer() {
        let pretokenizer = SequencePreTokenizer::new(vec![
            Box::new(WhitespacePreTokenizer::new()),
            Box::new(PunctuationPreTokenizer::new(SplitBehavior::Isolated)),
        ]);

        let tokens = pretokenizer.pre_tokenize("hello, world!");
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, ",");
        assert_eq!(tokens[2].text, "world");
        assert_eq!(tokens[3].text, "!");
    }
}
