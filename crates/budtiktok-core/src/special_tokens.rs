//! Special token handling with Aho-Corasick automaton
//!
//! This module provides efficient special token matching using the Aho-Corasick
//! algorithm. It allows finding all occurrences of special tokens in a text in
//! a single pass, which is essential for correct tokenization.
//!
//! # Example
//!
//! ```rust,ignore
//! use budtiktok_core::special_tokens::{SpecialTokenMatcher, AddedToken};
//!
//! let tokens = vec![
//!     AddedToken::new("[CLS]", 101).special(),
//!     AddedToken::new("[SEP]", 102).special(),
//!     AddedToken::new("[UNK]", 100).special(),
//! ];
//!
//! let matcher = SpecialTokenMatcher::new(&tokens);
//! let matches = matcher.find_all("Hello [SEP] World");
//! ```

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use serde::{Deserialize, Serialize};

/// Represents an added token (special token)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddedToken {
    /// The token content/string
    pub content: String,
    /// The token ID in the vocabulary
    pub id: u32,
    /// Whether this is a special token
    pub special: bool,
    /// Whether the token should match only at word boundaries (single word)
    pub single_word: bool,
    /// Whether to strip left whitespace
    pub lstrip: bool,
    /// Whether to strip right whitespace
    pub rstrip: bool,
    /// Whether the token should be normalized before matching
    pub normalized: bool,
}

impl AddedToken {
    /// Create a new added token
    pub fn new(content: impl Into<String>, id: u32) -> Self {
        Self {
            content: content.into(),
            id,
            special: false,
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: true,
        }
    }

    /// Mark as special token
    pub fn special(mut self) -> Self {
        self.special = true;
        self.normalized = false; // Special tokens typically aren't normalized
        self
    }

    /// Set single word matching
    pub fn single_word(mut self, value: bool) -> Self {
        self.single_word = value;
        self
    }

    /// Set left strip
    pub fn lstrip(mut self, value: bool) -> Self {
        self.lstrip = value;
        self
    }

    /// Set right strip
    pub fn rstrip(mut self, value: bool) -> Self {
        self.rstrip = value;
        self
    }

    /// Set normalized
    pub fn normalized(mut self, value: bool) -> Self {
        self.normalized = value;
        self
    }
}

/// A match found by the special token matcher
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokenMatch {
    /// Start byte offset in the text
    pub start: usize,
    /// End byte offset in the text
    pub end: usize,
    /// Index of the matched token in the original token list
    pub token_idx: usize,
    /// Token ID
    pub token_id: u32,
}

impl SpecialTokenMatch {
    /// Get the length of the match
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Check if the match is empty
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Efficient special token matcher using Aho-Corasick automaton
///
/// This matcher finds all occurrences of special tokens in a single pass
/// through the text, using the LeftmostLongest match semantics to prefer
/// longer matches when tokens overlap.
#[derive(Debug)]
pub struct SpecialTokenMatcher {
    /// The Aho-Corasick automaton
    automaton: AhoCorasick,
    /// The added tokens (indexed by pattern ID)
    tokens: Vec<AddedToken>,
    /// Whether any token requires normalized matching
    has_normalized: bool,
    /// Whether any token requires non-normalized matching
    has_non_normalized: bool,
}

impl SpecialTokenMatcher {
    /// Create a new matcher from a list of added tokens
    pub fn new(tokens: &[AddedToken]) -> Self {
        // Separate normalized and non-normalized tokens
        let patterns: Vec<&str> = tokens.iter().map(|t| t.content.as_str()).collect();

        let automaton = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .expect("Failed to build Aho-Corasick automaton");

        let has_normalized = tokens.iter().any(|t| t.normalized);
        let has_non_normalized = tokens.iter().any(|t| !t.normalized);

        Self {
            automaton,
            tokens: tokens.to_vec(),
            has_normalized,
            has_non_normalized,
        }
    }

    /// Create an empty matcher
    pub fn empty() -> Self {
        let empty_patterns: &[&str] = &[];
        Self {
            automaton: AhoCorasick::new(empty_patterns).unwrap(),
            tokens: Vec::new(),
            has_normalized: false,
            has_non_normalized: false,
        }
    }

    /// Check if the matcher is empty (no patterns)
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Get the number of patterns
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Find all non-overlapping matches in the text
    ///
    /// Returns matches in order of their start position, with longer matches
    /// preferred when tokens overlap.
    pub fn find_all(&self, text: &str) -> Vec<SpecialTokenMatch> {
        if self.is_empty() {
            return Vec::new();
        }

        self.automaton
            .find_iter(text)
            .filter_map(|m| {
                let token_idx = m.pattern().as_usize();
                let token = &self.tokens[token_idx];

                // Check single_word constraint
                if token.single_word {
                    let start = m.start();
                    let end = m.end();

                    // Check if preceded by word character
                    if start > 0 {
                        let prev_char = text[..start].chars().last();
                        if prev_char.map(|c| c.is_alphanumeric()).unwrap_or(false) {
                            return None;
                        }
                    }

                    // Check if followed by word character
                    if end < text.len() {
                        let next_char = text[end..].chars().next();
                        if next_char.map(|c| c.is_alphanumeric()).unwrap_or(false) {
                            return None;
                        }
                    }
                }

                Some(SpecialTokenMatch {
                    start: m.start(),
                    end: m.end(),
                    token_idx,
                    token_id: token.id,
                })
            })
            .collect()
    }

    /// Find all matches and return the text split around them
    ///
    /// Returns a sequence of (text_segment, is_special, Option<token_id>)
    pub fn split_with_matches<'a>(&self, text: &'a str) -> Vec<(&'a str, bool, Option<u32>)> {
        let matches = self.find_all(text);
        let mut result = Vec::with_capacity(matches.len() * 2 + 1);
        let mut last_end = 0;

        for m in &matches {
            // Add text before the match
            if m.start > last_end {
                result.push((&text[last_end..m.start], false, None));
            }

            // Add the match itself
            result.push((&text[m.start..m.end], true, Some(m.token_id)));

            last_end = m.end;
        }

        // Add remaining text after last match
        if last_end < text.len() {
            result.push((&text[last_end..], false, None));
        }

        // Handle empty text
        if result.is_empty() && !text.is_empty() {
            result.push((text, false, None));
        }

        result
    }

    /// Get the token at the given index
    pub fn get_token(&self, idx: usize) -> Option<&AddedToken> {
        self.tokens.get(idx)
    }

    /// Get all tokens
    pub fn get_tokens(&self) -> &[AddedToken] {
        &self.tokens
    }

    /// Check if the matcher has any normalized tokens
    pub fn has_normalized(&self) -> bool {
        self.has_normalized
    }

    /// Check if the matcher has any non-normalized tokens
    pub fn has_non_normalized(&self) -> bool {
        self.has_non_normalized
    }

    /// Extract text segments and normalize non-special parts
    ///
    /// This method splits the text around special tokens and optionally
    /// normalizes the non-special segments. Special tokens are never normalized.
    ///
    /// # Arguments
    /// * `text` - The input text to process
    /// * `normalizer` - Optional normalizer for non-special segments
    ///
    /// # Returns
    /// A vector of (segment, is_special, token_id) tuples where normalized
    /// segments are returned as owned Strings.
    pub fn extract_and_normalize<N>(
        &self,
        text: &str,
        normalizer: Option<&N>,
    ) -> Vec<ExtractedSegment>
    where
        N: crate::normalizer::Normalizer,
    {
        let matches = self.find_all(text);
        let mut result = Vec::with_capacity(matches.len() * 2 + 1);
        let mut last_end = 0;

        for m in &matches {
            // Add text before the match (normalized if normalizer provided)
            if m.start > last_end {
                let segment = &text[last_end..m.start];
                let content = if let Some(norm) = normalizer {
                    norm.normalize(segment).into_owned()
                } else {
                    segment.to_string()
                };

                if !content.is_empty() {
                    result.push(ExtractedSegment {
                        content,
                        is_special: false,
                        token_id: None,
                        token_idx: None,
                    });
                }
            }

            // Add the special token (never normalized)
            let token = &self.tokens[m.token_idx];
            result.push(ExtractedSegment {
                content: token.content.clone(),
                is_special: true,
                token_id: Some(m.token_id),
                token_idx: Some(m.token_idx),
            });

            last_end = m.end;
        }

        // Add remaining text after last match
        if last_end < text.len() {
            let segment = &text[last_end..];
            let content = if let Some(norm) = normalizer {
                norm.normalize(segment).into_owned()
            } else {
                segment.to_string()
            };

            if !content.is_empty() {
                result.push(ExtractedSegment {
                    content,
                    is_special: false,
                    token_id: None,
                    token_idx: None,
                });
            }
        }

        // Handle text with no matches
        if result.is_empty() && !text.is_empty() {
            let content = if let Some(norm) = normalizer {
                norm.normalize(text).into_owned()
            } else {
                text.to_string()
            };

            if !content.is_empty() {
                result.push(ExtractedSegment {
                    content,
                    is_special: false,
                    token_id: None,
                    token_idx: None,
                });
            }
        }

        result
    }

    /// Find a token by its content
    pub fn find_token(&self, content: &str) -> Option<&AddedToken> {
        self.tokens.iter().find(|t| t.content == content)
    }

    /// Get token ID for a given content string
    pub fn token_to_id(&self, content: &str) -> Option<u32> {
        self.find_token(content).map(|t| t.id)
    }

    /// Get token content for a given ID
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokens.iter()
            .find(|t| t.id == id)
            .map(|t| t.content.as_str())
    }
}

/// An extracted segment from special token processing
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedSegment {
    /// The segment content (owned string)
    pub content: String,
    /// Whether this is a special token
    pub is_special: bool,
    /// The token ID if this is a special token
    pub token_id: Option<u32>,
    /// The token index in the matcher
    pub token_idx: Option<usize>,
}

impl ExtractedSegment {
    /// Check if the segment is empty
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Get the length of the content
    pub fn len(&self) -> usize {
        self.content.len()
    }
}

impl Default for SpecialTokenMatcher {
    fn default() -> Self {
        Self::empty()
    }
}

/// Builder for creating special token matchers
#[derive(Debug, Default)]
pub struct SpecialTokenMatcherBuilder {
    tokens: Vec<AddedToken>,
}

impl SpecialTokenMatcherBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a token
    pub fn add_token(mut self, token: AddedToken) -> Self {
        self.tokens.push(token);
        self
    }

    /// Add multiple tokens
    pub fn add_tokens(mut self, tokens: impl IntoIterator<Item = AddedToken>) -> Self {
        self.tokens.extend(tokens);
        self
    }

    /// Add a special token by content and ID
    pub fn add_special(mut self, content: impl Into<String>, id: u32) -> Self {
        self.tokens.push(AddedToken::new(content, id).special());
        self
    }

    /// Build the matcher
    pub fn build(self) -> SpecialTokenMatcher {
        SpecialTokenMatcher::new(&self.tokens)
    }
}

/// Create a default BERT special token matcher
pub fn bert_special_tokens(
    cls_id: u32,
    sep_id: u32,
    pad_id: u32,
    unk_id: u32,
    mask_id: u32,
) -> SpecialTokenMatcher {
    SpecialTokenMatcherBuilder::new()
        .add_special("[CLS]", cls_id)
        .add_special("[SEP]", sep_id)
        .add_special("[PAD]", pad_id)
        .add_special("[UNK]", unk_id)
        .add_special("[MASK]", mask_id)
        .build()
}

/// Create a default GPT-2 special token matcher
pub fn gpt2_special_tokens(
    bos_id: u32,
    eos_id: u32,
    unk_id: Option<u32>,
) -> SpecialTokenMatcher {
    let mut builder = SpecialTokenMatcherBuilder::new()
        .add_special("<|endoftext|>", eos_id);

    if bos_id != eos_id {
        builder = builder.add_special("<|startoftext|>", bos_id);
    }

    if let Some(unk) = unk_id {
        builder = builder.add_special("<|unk|>", unk);
    }

    builder.build()
}

/// Create a default LLaMA special token matcher
pub fn llama_special_tokens(
    bos_id: u32,
    eos_id: u32,
    unk_id: u32,
) -> SpecialTokenMatcher {
    SpecialTokenMatcherBuilder::new()
        .add_special("<s>", bos_id)
        .add_special("</s>", eos_id)
        .add_special("<unk>", unk_id)
        .build()
}

/// Create a default T5 special token matcher
pub fn t5_special_tokens(
    pad_id: u32,
    eos_id: u32,
    unk_id: u32,
) -> SpecialTokenMatcher {
    let mut builder = SpecialTokenMatcherBuilder::new()
        .add_special("<pad>", pad_id)
        .add_special("</s>", eos_id)
        .add_special("<unk>", unk_id);

    // T5 has extra sentinel tokens <extra_id_0> through <extra_id_99>
    // We add the first few commonly used ones
    for i in 0..10 {
        builder = builder.add_special(format!("<extra_id_{}>", i), unk_id + 1 + i);
    }

    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_added_token() {
        let token = AddedToken::new("[CLS]", 101).special();
        assert_eq!(token.content, "[CLS]");
        assert_eq!(token.id, 101);
        assert!(token.special);
        assert!(!token.normalized);
    }

    #[test]
    fn test_matcher_basic() {
        let tokens = vec![
            AddedToken::new("[CLS]", 101).special(),
            AddedToken::new("[SEP]", 102).special(),
        ];

        let matcher = SpecialTokenMatcher::new(&tokens);
        assert_eq!(matcher.len(), 2);
        assert!(!matcher.is_empty());
    }

    #[test]
    fn test_find_all() {
        let tokens = vec![
            AddedToken::new("[CLS]", 101).special(),
            AddedToken::new("[SEP]", 102).special(),
        ];

        let matcher = SpecialTokenMatcher::new(&tokens);
        let matches = matcher.find_all("[CLS] Hello [SEP] World [SEP]");

        assert_eq!(matches.len(), 3);

        // [CLS] at position 0
        assert_eq!(matches[0].start, 0);
        assert_eq!(matches[0].end, 5);
        assert_eq!(matches[0].token_id, 101);

        // First [SEP] at position 12
        assert_eq!(matches[1].start, 12);
        assert_eq!(matches[1].end, 17);
        assert_eq!(matches[1].token_id, 102);

        // Second [SEP] at position 24
        assert_eq!(matches[2].start, 24);
        assert_eq!(matches[2].end, 29);
        assert_eq!(matches[2].token_id, 102);
    }

    #[test]
    fn test_split_with_matches() {
        let tokens = vec![
            AddedToken::new("[CLS]", 101).special(),
            AddedToken::new("[SEP]", 102).special(),
        ];

        let matcher = SpecialTokenMatcher::new(&tokens);
        let splits = matcher.split_with_matches("[CLS] Hello [SEP]");

        // "[CLS]" + " Hello " + "[SEP]" = 3 segments (no trailing text)
        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0], ("[CLS]", true, Some(101)));
        assert_eq!(splits[1], (" Hello ", false, None));
        assert_eq!(splits[2], ("[SEP]", true, Some(102)));

        // Test with trailing text
        let splits = matcher.split_with_matches("[CLS] Hello [SEP] World");
        assert_eq!(splits.len(), 4);
        assert_eq!(splits[0], ("[CLS]", true, Some(101)));
        assert_eq!(splits[1], (" Hello ", false, None));
        assert_eq!(splits[2], ("[SEP]", true, Some(102)));
        assert_eq!(splits[3], (" World", false, None));
    }

    #[test]
    fn test_single_word() {
        let tokens = vec![AddedToken::new("test", 100).single_word(true)];

        let matcher = SpecialTokenMatcher::new(&tokens);

        // Should match standalone "test"
        let matches = matcher.find_all("This is test here");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].start, 8);
        assert_eq!(matches[0].end, 12);

        // Should NOT match "test" in "testing"
        let matches = matcher.find_all("This is testing");
        assert_eq!(matches.len(), 0);

        // Should NOT match "test" in "atest"
        let matches = matcher.find_all("This is atest");
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_leftmost_longest() {
        // Test that longer matches are preferred
        let tokens = vec![
            AddedToken::new("[SEP]", 102).special(),
            AddedToken::new("[SEPARATOR]", 103).special(),
        ];

        let matcher = SpecialTokenMatcher::new(&tokens);
        let matches = matcher.find_all("Hello [SEPARATOR] World");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].token_id, 103); // Should match the longer one
    }

    #[test]
    fn test_empty_matcher() {
        let matcher = SpecialTokenMatcher::empty();
        assert!(matcher.is_empty());
        assert_eq!(matcher.len(), 0);

        let matches = matcher.find_all("Hello World");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_no_matches() {
        let tokens = vec![AddedToken::new("[CLS]", 101).special()];

        let matcher = SpecialTokenMatcher::new(&tokens);
        let matches = matcher.find_all("Hello World");

        assert!(matches.is_empty());
    }

    #[test]
    fn test_builder() {
        let matcher = SpecialTokenMatcherBuilder::new()
            .add_special("[CLS]", 101)
            .add_special("[SEP]", 102)
            .add_token(AddedToken::new("[UNK]", 100).special().single_word(true))
            .build();

        assert_eq!(matcher.len(), 3);

        let matches = matcher.find_all("[CLS] test [SEP]");
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_bert_special_tokens() {
        let matcher = bert_special_tokens(101, 102, 0, 100, 103);
        assert_eq!(matcher.len(), 5);

        let matches = matcher.find_all("Input: [MASK] is a [UNK]");
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_llama_special_tokens() {
        let matcher = llama_special_tokens(1, 2, 0);
        assert_eq!(matcher.len(), 3);

        let matches = matcher.find_all("<s>Hello</s>");
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_adjacent_special_tokens() {
        let tokens = vec![
            AddedToken::new("[CLS]", 101).special(),
            AddedToken::new("[SEP]", 102).special(),
        ];

        let matcher = SpecialTokenMatcher::new(&tokens);
        let matches = matcher.find_all("[CLS][SEP]");

        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].token_id, 101);
        assert_eq!(matches[1].token_id, 102);
    }

    #[test]
    fn test_unicode_special_tokens() {
        let tokens = vec![
            AddedToken::new("▁", 1000).special(), // Metaspace
            AddedToken::new("⁇", 1001).special(), // Double question mark
        ];

        let matcher = SpecialTokenMatcher::new(&tokens);
        let matches = matcher.find_all("Hello▁World⁇End");

        assert_eq!(matches.len(), 2);
    }
}
