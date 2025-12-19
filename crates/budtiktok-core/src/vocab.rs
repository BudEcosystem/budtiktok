//! Vocabulary management for tokenizers
//!
//! This module handles vocabulary loading, lookup, and management for
//! different tokenizer types (WordPiece, BPE, Unigram).

use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{Error, Result};

/// Vocabulary structure for token-to-ID and ID-to-token mappings
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Token string to token ID mapping
    token_to_id: AHashMap<String, u32>,
    /// Token ID to token string mapping
    id_to_token: Vec<String>,
    /// Special tokens configuration
    special_tokens: SpecialTokens,
}

/// Special tokens configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpecialTokens {
    /// Unknown token
    pub unk_token: Option<String>,
    /// Padding token
    pub pad_token: Option<String>,
    /// Classification token (e.g., [CLS])
    pub cls_token: Option<String>,
    /// Separator token (e.g., [SEP])
    pub sep_token: Option<String>,
    /// Mask token (e.g., [MASK])
    pub mask_token: Option<String>,
    /// Beginning of sequence token
    pub bos_token: Option<String>,
    /// End of sequence token
    pub eos_token: Option<String>,
}

impl Vocabulary {
    /// Create a new vocabulary from a token-to-ID mapping
    pub fn new(token_to_id: AHashMap<String, u32>, special_tokens: SpecialTokens) -> Self {
        let mut id_to_token = vec![String::new(); token_to_id.len()];
        for (token, &id) in &token_to_id {
            if (id as usize) < id_to_token.len() {
                id_to_token[id as usize] = token.clone();
            }
        }

        Self {
            token_to_id,
            id_to_token,
            special_tokens,
        }
    }

    /// Load vocabulary from a JSON file (HuggingFace format)
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Self::from_json(&content)
    }

    /// Load vocabulary from a JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        // Try to parse as a simple {token: id} mapping
        let token_to_id: AHashMap<String, u32> = serde_json::from_str(json)
            .map_err(|e| Error::vocab_load(format!("Failed to parse vocabulary JSON: {}", e)))?;

        Ok(Self::new(token_to_id, SpecialTokens::default()))
    }

    /// Get the token ID for a token string
    #[inline]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get the token string for a token ID
    #[inline]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// Get the vocabulary size
    #[inline]
    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }

    /// Check if vocabulary is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }

    /// Check if a token exists in the vocabulary
    #[inline]
    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Get the unknown token ID
    pub fn unk_token_id(&self) -> Option<u32> {
        self.special_tokens
            .unk_token
            .as_ref()
            .and_then(|t| self.token_to_id(t))
    }

    /// Get the padding token ID
    pub fn pad_token_id(&self) -> Option<u32> {
        self.special_tokens
            .pad_token
            .as_ref()
            .and_then(|t| self.token_to_id(t))
    }

    /// Get the CLS token ID
    pub fn cls_token_id(&self) -> Option<u32> {
        self.special_tokens
            .cls_token
            .as_ref()
            .and_then(|t| self.token_to_id(t))
    }

    /// Get the SEP token ID
    pub fn sep_token_id(&self) -> Option<u32> {
        self.special_tokens
            .sep_token
            .as_ref()
            .and_then(|t| self.token_to_id(t))
    }

    /// Get the special tokens configuration
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Set special tokens
    pub fn set_special_tokens(&mut self, special_tokens: SpecialTokens) {
        self.special_tokens = special_tokens;
    }

    /// Iterate over all tokens
    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> {
        self.token_to_id.iter().map(|(k, &v)| (k.as_str(), v))
    }

    /// Get all tokens starting with a prefix (useful for WordPiece continuation tokens)
    pub fn tokens_with_prefix(&self, prefix: &str) -> Vec<(&str, u32)> {
        self.token_to_id
            .iter()
            .filter(|(token, _)| token.starts_with(prefix))
            .map(|(k, &v)| (k.as_str(), v))
            .collect()
    }
}

/// Builder for creating vocabularies
pub struct VocabularyBuilder {
    tokens: Vec<String>,
    special_tokens: SpecialTokens,
}

impl VocabularyBuilder {
    /// Create a new vocabulary builder
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            special_tokens: SpecialTokens::default(),
        }
    }

    /// Add a token to the vocabulary
    pub fn add_token(mut self, token: impl Into<String>) -> Self {
        self.tokens.push(token.into());
        self
    }

    /// Add multiple tokens
    pub fn add_tokens(mut self, tokens: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tokens.extend(tokens.into_iter().map(|t| t.into()));
        self
    }

    /// Set the unknown token
    pub fn unk_token(mut self, token: impl Into<String>) -> Self {
        self.special_tokens.unk_token = Some(token.into());
        self
    }

    /// Set the padding token
    pub fn pad_token(mut self, token: impl Into<String>) -> Self {
        self.special_tokens.pad_token = Some(token.into());
        self
    }

    /// Set the CLS token
    pub fn cls_token(mut self, token: impl Into<String>) -> Self {
        self.special_tokens.cls_token = Some(token.into());
        self
    }

    /// Set the SEP token
    pub fn sep_token(mut self, token: impl Into<String>) -> Self {
        self.special_tokens.sep_token = Some(token.into());
        self
    }

    /// Build the vocabulary
    pub fn build(self) -> Vocabulary {
        let mut token_to_id = AHashMap::with_capacity(self.tokens.len());
        for (id, token) in self.tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
        }
        Vocabulary::new(token_to_id, self.special_tokens)
    }
}

impl Default for VocabularyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_basic() {
        let vocab = VocabularyBuilder::new()
            .add_tokens(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world"])
            .unk_token("[UNK]")
            .pad_token("[PAD]")
            .cls_token("[CLS]")
            .sep_token("[SEP]")
            .build();

        assert_eq!(vocab.len(), 6);
        assert_eq!(vocab.token_to_id("hello"), Some(4));
        assert_eq!(vocab.id_to_token(4), Some("hello"));
        assert!(vocab.contains("world"));
        assert!(!vocab.contains("foo"));
    }

    #[test]
    fn test_special_tokens() {
        let vocab = VocabularyBuilder::new()
            .add_tokens(["[UNK]", "[CLS]", "[SEP]"])
            .unk_token("[UNK]")
            .cls_token("[CLS]")
            .sep_token("[SEP]")
            .build();

        assert_eq!(vocab.unk_token_id(), Some(0));
        assert_eq!(vocab.cls_token_id(), Some(1));
        assert_eq!(vocab.sep_token_id(), Some(2));
    }
}
