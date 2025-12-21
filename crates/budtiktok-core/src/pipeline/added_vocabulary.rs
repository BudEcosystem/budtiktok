//! AddedVocabulary for dynamic token management
//!
//! This module provides an enhanced AddedVocabulary implementation that supports
//! dynamic token addition at runtime, compatible with HuggingFace tokenizers.
//! It uses dual Aho-Corasick automata for efficient token matching:
//! - One for non-normalized tokens (matched first, on original text)
//! - One for normalized tokens (matched after normalization)

use ahash::{AHashMap, AHashSet};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use serde::{Deserialize, Serialize};

use crate::normalizer::Normalizer;
use crate::special_tokens::AddedToken;

/// Segment extracted from text during added token processing
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedSegment {
    /// The segment content (may be normalized)
    pub content: String,
    /// Whether this is an added token
    pub is_added_token: bool,
    /// The token ID if this is an added token
    pub token_id: Option<u32>,
    /// Whether this is a special token
    pub is_special: bool,
}

impl ExtractedSegment {
    /// Create a new regular text segment
    pub fn text(content: String) -> Self {
        Self {
            content,
            is_added_token: false,
            token_id: None,
            is_special: false,
        }
    }

    /// Create a new added token segment
    pub fn added_token(content: String, id: u32, is_special: bool) -> Self {
        Self {
            content,
            is_added_token: true,
            token_id: Some(id),
            is_special,
        }
    }
}

/// Dynamic vocabulary for user-added tokens
///
/// This structure manages tokens that are added to the vocabulary at runtime,
/// separate from the base model vocabulary. It uses dual Aho-Corasick automata
/// for efficient matching:
/// - Non-normalized tokens are matched first on the original text
/// - Normalized tokens are matched after normalization on remaining segments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddedVocabulary {
    /// Token content -> ID mapping
    added_tokens_map: AHashMap<String, u32>,

    /// ID -> AddedToken mapping (for decode)
    #[serde(skip)]
    added_tokens_map_r: AHashMap<u32, AddedToken>,

    /// Regular added tokens (ordered by insertion)
    added_tokens: Vec<AddedToken>,

    /// Special tokens (ordered by insertion)
    special_tokens: Vec<AddedToken>,

    /// O(1) special token lookup
    special_tokens_set: AHashSet<String>,

    /// Aho-Corasick for non-normalized tokens
    #[serde(skip)]
    split_trie: Option<(AhoCorasick, Vec<u32>)>,

    /// Aho-Corasick for normalized tokens
    #[serde(skip)]
    split_normalized_trie: Option<(AhoCorasick, Vec<u32>)>,

    /// Whether to encode special tokens (default: false)
    #[serde(default)]
    encode_special_tokens: bool,
}

impl Default for AddedVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

impl AddedVocabulary {
    /// Create a new empty AddedVocabulary
    pub fn new() -> Self {
        Self {
            added_tokens_map: AHashMap::new(),
            added_tokens_map_r: AHashMap::new(),
            added_tokens: Vec::new(),
            special_tokens: Vec::new(),
            special_tokens_set: AHashSet::new(),
            split_trie: None,
            split_normalized_trie: None,
            encode_special_tokens: false,
        }
    }

    /// Create from a list of tokens
    pub fn from_tokens(tokens: &[AddedToken], special: &[AddedToken]) -> Self {
        let mut vocab = Self::new();

        // Add special tokens first
        for token in special {
            vocab.add_token_internal(token.clone(), true);
        }

        // Add regular tokens
        for token in tokens {
            vocab.add_token_internal(token.clone(), false);
        }

        vocab.rebuild_tries();
        vocab
    }

    /// Add a token internally without rebuilding tries
    fn add_token_internal(&mut self, mut token: AddedToken, is_special: bool) {
        // Skip if already exists with same ID
        if let Some(&existing_id) = self.added_tokens_map.get(&token.content) {
            if existing_id == token.id {
                return;
            }
        }

        // Mark as special if needed
        if is_special {
            token.special = true;
        }

        // Add to maps
        self.added_tokens_map.insert(token.content.clone(), token.id);
        self.added_tokens_map_r.insert(token.id, token.clone());

        // Add to appropriate list
        if token.special {
            self.special_tokens_set.insert(token.content.clone());
            self.special_tokens.push(token);
        } else {
            self.added_tokens.push(token);
        }
    }

    /// Rebuild the Aho-Corasick tries
    fn rebuild_tries(&mut self) {
        // Collect non-normalized tokens (owned strings to avoid borrow issues)
        let non_normalized: Vec<(String, u32)> = self.all_tokens()
            .filter(|t| !t.normalized)
            .map(|t| (t.content.clone(), t.id))
            .collect();

        // Collect normalized tokens (owned strings to avoid borrow issues)
        let normalized: Vec<(String, u32)> = self.all_tokens()
            .filter(|t| t.normalized)
            .map(|t| (t.content.clone(), t.id))
            .collect();

        // Build non-normalized trie
        self.split_trie = if non_normalized.is_empty() {
            None
        } else {
            let patterns: Vec<&str> = non_normalized.iter().map(|(s, _)| s.as_str()).collect();
            let ids: Vec<u32> = non_normalized.iter().map(|(_, id)| *id).collect();

            AhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build(&patterns)
                .ok()
                .map(|ac| (ac, ids))
        };

        // Build normalized trie
        self.split_normalized_trie = if normalized.is_empty() {
            None
        } else {
            let patterns: Vec<&str> = normalized.iter().map(|(s, _)| s.as_str()).collect();
            let ids: Vec<u32> = normalized.iter().map(|(_, id)| *id).collect();

            AhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build(&patterns)
                .ok()
                .map(|ac| (ac, ids))
        };
    }

    /// Add tokens to the vocabulary (HF-compatible)
    ///
    /// Returns the number of tokens actually added (excluding duplicates).
    pub fn add_tokens<N: Normalizer>(
        &mut self,
        tokens: &[AddedToken],
        _model_vocab_size: usize,
        _normalizer: Option<&N>,
    ) -> usize {
        let initial_count = self.len();

        for token in tokens {
            self.add_token_internal(token.clone(), false);
        }

        // Rebuild tries after adding all tokens
        self.rebuild_tries();

        self.len() - initial_count
    }

    /// Add special tokens to the vocabulary (HF-compatible)
    ///
    /// Returns the number of tokens actually added (excluding duplicates).
    pub fn add_special_tokens<N: Normalizer>(
        &mut self,
        tokens: &[AddedToken],
        _model_vocab_size: usize,
        _normalizer: Option<&N>,
    ) -> usize {
        let initial_count = self.len();

        for token in tokens {
            self.add_token_internal(token.clone(), true);
        }

        // Rebuild tries after adding all tokens
        self.rebuild_tries();

        self.len() - initial_count
    }

    /// Extract and normalize text, identifying added tokens
    ///
    /// This method performs a two-pass extraction:
    /// 1. First pass: Find non-normalized tokens in the original text
    /// 2. Second pass: Normalize remaining segments and find normalized tokens
    pub fn extract_and_normalize<N: Normalizer>(
        &self,
        text: &str,
        normalizer: Option<&N>,
    ) -> Vec<ExtractedSegment> {
        if self.is_empty() {
            // No added tokens, just normalize the whole text
            let content = match normalizer {
                Some(n) => n.normalize(text).into_owned(),
                None => text.to_string(),
            };
            return vec![ExtractedSegment::text(content)];
        }

        // First pass: find non-normalized tokens
        let mut segments = Vec::new();

        if let Some((ref ac, ref ids)) = self.split_trie {
            let mut last_end = 0;

            for m in ac.find_iter(text) {
                let token_id = ids[m.pattern().as_usize()];
                let token = self.id_to_token(token_id);
                let is_special = token.map(|t| t.special).unwrap_or(false);

                // Check single_word constraint
                if let Some(t) = token {
                    if t.single_word && !self.check_word_boundary(text, m.start(), m.end()) {
                        continue;
                    }
                }

                // Add text before this match (will be processed in second pass)
                if m.start() > last_end {
                    segments.push(ExtractedSegment::text(text[last_end..m.start()].to_string()));
                }

                // Add the matched token
                segments.push(ExtractedSegment::added_token(
                    text[m.start()..m.end()].to_string(),
                    token_id,
                    is_special,
                ));

                last_end = m.end();
            }

            // Add remaining text
            if last_end < text.len() {
                segments.push(ExtractedSegment::text(text[last_end..].to_string()));
            }
        } else {
            // No non-normalized tokens
            segments.push(ExtractedSegment::text(text.to_string()));
        }

        // Second pass: normalize text segments and find normalized tokens
        let mut final_segments = Vec::new();

        for segment in segments {
            if segment.is_added_token {
                // Pass through added tokens unchanged
                final_segments.push(segment);
            } else {
                // Normalize and search for normalized tokens
                let normalized_text = match normalizer {
                    Some(n) => n.normalize(&segment.content).into_owned(),
                    None => segment.content.clone(),
                };

                if let Some((ref ac, ref ids)) = self.split_normalized_trie {
                    let mut last_end = 0;

                    for m in ac.find_iter(&normalized_text) {
                        let token_id = ids[m.pattern().as_usize()];
                        let token = self.id_to_token(token_id);
                        let is_special = token.map(|t| t.special).unwrap_or(false);

                        // Check single_word constraint
                        if let Some(t) = token {
                            if t.single_word && !self.check_word_boundary(&normalized_text, m.start(), m.end()) {
                                continue;
                            }
                        }

                        // Add text before this match
                        if m.start() > last_end {
                            final_segments.push(ExtractedSegment::text(
                                normalized_text[last_end..m.start()].to_string(),
                            ));
                        }

                        // Add the matched token
                        final_segments.push(ExtractedSegment::added_token(
                            normalized_text[m.start()..m.end()].to_string(),
                            token_id,
                            is_special,
                        ));

                        last_end = m.end();
                    }

                    // Add remaining text
                    if last_end < normalized_text.len() {
                        final_segments.push(ExtractedSegment::text(
                            normalized_text[last_end..].to_string(),
                        ));
                    }
                } else {
                    // No normalized tokens, just add the normalized text
                    if !normalized_text.is_empty() {
                        final_segments.push(ExtractedSegment::text(normalized_text));
                    }
                }
            }
        }

        // Handle empty result
        if final_segments.is_empty() {
            let content = match normalizer {
                Some(n) => n.normalize(text).into_owned(),
                None => text.to_string(),
            };
            if !content.is_empty() {
                final_segments.push(ExtractedSegment::text(content));
            }
        }

        final_segments
    }

    /// Check if a match is at word boundaries
    fn check_word_boundary(&self, text: &str, start: usize, end: usize) -> bool {
        // Check character before
        if start > 0 {
            if let Some(c) = text[..start].chars().last() {
                if c.is_alphanumeric() {
                    return false;
                }
            }
        }

        // Check character after
        if end < text.len() {
            if let Some(c) = text[end..].chars().next() {
                if c.is_alphanumeric() {
                    return false;
                }
            }
        }

        true
    }

    /// Get token ID for a given content string
    pub fn token_to_id(&self, content: &str) -> Option<u32> {
        self.added_tokens_map.get(content).copied()
    }

    /// Get AddedToken for a given ID
    pub fn id_to_token(&self, id: u32) -> Option<&AddedToken> {
        self.added_tokens_map_r.get(&id)
    }

    /// Get token content string for a given ID
    pub fn id_to_content(&self, id: u32) -> Option<&str> {
        self.added_tokens_map_r.get(&id).map(|t| t.content.as_str())
    }

    /// Check if an ID is a special token
    pub fn is_special_token(&self, id: u32) -> bool {
        self.added_tokens_map_r
            .get(&id)
            .map(|t| t.special)
            .unwrap_or(false)
    }

    /// Check if a content string is a special token
    pub fn is_special(&self, content: &str) -> bool {
        self.special_tokens_set.contains(content)
    }

    /// Get the number of added tokens (including special tokens)
    pub fn len(&self) -> usize {
        self.added_tokens.len() + self.special_tokens.len()
    }

    /// Check if the vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.added_tokens.is_empty() && self.special_tokens.is_empty()
    }

    /// Iterate over all added tokens
    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> {
        self.added_tokens_map.iter().map(|(k, &v)| (k.as_str(), v))
    }

    /// Iterate over all AddedToken objects
    pub fn all_tokens(&self) -> impl Iterator<Item = &AddedToken> {
        self.special_tokens.iter().chain(self.added_tokens.iter())
    }

    /// Get all special tokens
    pub fn get_special_tokens(&self) -> &[AddedToken] {
        &self.special_tokens
    }

    /// Get all regular added tokens
    pub fn get_added_tokens(&self) -> &[AddedToken] {
        &self.added_tokens
    }

    /// Set whether to encode special tokens
    pub fn set_encode_special_tokens(&mut self, encode: bool) {
        self.encode_special_tokens = encode;
    }

    /// Get whether to encode special tokens
    pub fn encode_special_tokens(&self) -> bool {
        self.encode_special_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_added_vocabulary_new() {
        let vocab = AddedVocabulary::new();
        assert!(vocab.is_empty());
        assert_eq!(vocab.len(), 0);
    }

    #[test]
    fn test_add_special_tokens() {
        let mut vocab = AddedVocabulary::new();

        let tokens = vec![
            AddedToken::new("[CLS]", 101).special(),
            AddedToken::new("[SEP]", 102).special(),
        ];

        let added = vocab.add_special_tokens::<crate::normalizer::NfcNormalizer>(&tokens, 30000, None);
        assert_eq!(added, 2);
        assert_eq!(vocab.len(), 2);
        assert!(vocab.is_special("[CLS]"));
        assert!(vocab.is_special("[SEP]"));
    }

    #[test]
    fn test_add_regular_tokens() {
        let mut vocab = AddedVocabulary::new();

        let tokens = vec![
            AddedToken::new("<custom1>", 30000),
            AddedToken::new("<custom2>", 30001),
        ];

        let added = vocab.add_tokens::<crate::normalizer::NfcNormalizer>(&tokens, 30000, None);
        assert_eq!(added, 2);
        assert_eq!(vocab.len(), 2);
        assert!(!vocab.is_special("<custom1>"));
    }

    #[test]
    fn test_token_lookup() {
        let mut vocab = AddedVocabulary::new();

        let tokens = vec![AddedToken::new("[CLS]", 101).special()];
        vocab.add_special_tokens::<crate::normalizer::NfcNormalizer>(&tokens, 30000, None);

        assert_eq!(vocab.token_to_id("[CLS]"), Some(101));
        assert!(vocab.is_special_token(101));
        assert_eq!(vocab.id_to_content(101), Some("[CLS]"));
    }

    #[test]
    fn test_extract_simple() {
        let mut vocab = AddedVocabulary::new();

        let tokens = vec![
            AddedToken::new("[CLS]", 101).special().normalized(false),
            AddedToken::new("[SEP]", 102).special().normalized(false),
        ];
        vocab.add_special_tokens::<crate::normalizer::NfcNormalizer>(&tokens, 30000, None);

        let segments = vocab.extract_and_normalize::<crate::normalizer::NfcNormalizer>(
            "[CLS] Hello World [SEP]",
            None,
        );

        assert_eq!(segments.len(), 3);
        assert!(segments[0].is_added_token);
        assert_eq!(segments[0].token_id, Some(101));
        assert!(!segments[1].is_added_token);
        assert_eq!(segments[1].content, " Hello World ");
        assert!(segments[2].is_added_token);
        assert_eq!(segments[2].token_id, Some(102));
    }

    #[test]
    fn test_extract_with_normalization() {
        let mut vocab = AddedVocabulary::new();

        // Add a normalized token
        let tokens = vec![AddedToken::new("hello", 1000).normalized(true)];
        vocab.add_tokens::<crate::normalizer::NfcNormalizer>(&tokens, 30000, None);

        // Without normalization, "HELLO" shouldn't match "hello"
        let segments = vocab.extract_and_normalize::<crate::normalizer::NfcNormalizer>(
            "HELLO world",
            None,
        );

        // The token should not match because we're not normalizing the input
        assert!(!segments.iter().any(|s| s.is_added_token && s.token_id == Some(1000)));
    }

    #[test]
    fn test_duplicate_tokens() {
        let mut vocab = AddedVocabulary::new();

        let tokens = vec![AddedToken::new("[CLS]", 101).special()];
        vocab.add_special_tokens::<crate::normalizer::NfcNormalizer>(&tokens, 30000, None);

        // Add same token again
        let added = vocab.add_special_tokens::<crate::normalizer::NfcNormalizer>(&tokens, 30000, None);
        assert_eq!(added, 0); // No new tokens added
        assert_eq!(vocab.len(), 1);
    }

    #[test]
    fn test_from_tokens() {
        let special = vec![AddedToken::new("[PAD]", 0).special()];
        let regular = vec![AddedToken::new("<custom>", 30000)];

        let vocab = AddedVocabulary::from_tokens(&regular, &special);
        assert_eq!(vocab.len(), 2);
        assert!(vocab.is_special("[PAD]"));
        assert!(!vocab.is_special("<custom>"));
    }
}
