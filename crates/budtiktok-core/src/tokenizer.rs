//! Main tokenizer trait and implementations
//!
//! This module defines the core `Tokenizer` trait that all tokenizer
//! implementations must satisfy.

use std::path::Path;

use crate::encoding::Encoding;
use crate::error::Result;
use crate::vocab::Vocabulary;

/// Padding strategy for batch tokenization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// No padding
    None,
    /// Pad to longest sequence in batch
    Longest,
    /// Pad to a fixed length
    Fixed(usize),
}

/// Truncation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// No truncation
    None,
    /// Truncate only the first sequence
    OnlyFirst,
    /// Truncate only the second sequence
    OnlySecond,
    /// Truncate the longest sequence
    LongestFirst,
}

/// Tokenizer configuration
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Maximum sequence length
    pub max_length: Option<usize>,
    /// Padding strategy
    pub padding: PaddingStrategy,
    /// Truncation strategy
    pub truncation: TruncationStrategy,
    /// Stride for overflow tokens
    pub stride: usize,
    /// Whether to add special tokens
    pub add_special_tokens: bool,
    /// Return attention mask
    pub return_attention_mask: bool,
    /// Return token type IDs
    pub return_token_type_ids: bool,
    /// Return offsets mapping
    pub return_offsets_mapping: bool,
    /// Return special tokens mask
    pub return_special_tokens_mask: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            max_length: None,
            padding: PaddingStrategy::None,
            truncation: TruncationStrategy::None,
            stride: 0,
            add_special_tokens: true,
            return_attention_mask: true,
            return_token_type_ids: true,
            return_offsets_mapping: false,
            return_special_tokens_mask: false,
        }
    }
}

/// Core tokenizer trait (dyn-compatible)
pub trait Tokenizer: Send + Sync {
    /// Get the tokenizer's vocabulary
    fn vocabulary(&self) -> &Vocabulary;

    /// Encode a single text into tokens
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding>;

    /// Encode a pair of texts (for sequence classification, etc.)
    fn encode_pair(
        &self,
        text: &str,
        text_pair: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding>;

    /// Encode a batch of texts
    fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        texts
            .iter()
            .map(|t| self.encode(t, add_special_tokens))
            .collect()
    }

    /// Decode token IDs back to text
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Decode a batch of token ID sequences
    fn decode_batch(
        &self,
        ids_batch: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        ids_batch
            .iter()
            .map(|ids| self.decode(ids, skip_special_tokens))
            .collect()
    }

    /// Get a token ID for a token string
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocabulary().token_to_id(token)
    }

    /// Get a token string for a token ID
    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocabulary().id_to_token(id)
    }

    /// Get the vocabulary size
    fn vocab_size(&self) -> usize {
        self.vocabulary().len()
    }

    /// Save the tokenizer to a file
    fn save(&self, path: &Path) -> Result<()>;
}

/// Builder for tokenizer configuration
pub struct TokenizerConfigBuilder {
    config: TokenizerConfig,
}

impl TokenizerConfigBuilder {
    /// Create a new config builder
    pub fn new() -> Self {
        Self {
            config: TokenizerConfig::default(),
        }
    }

    /// Set maximum sequence length
    pub fn max_length(mut self, length: usize) -> Self {
        self.config.max_length = Some(length);
        self
    }

    /// Set padding strategy
    pub fn padding(mut self, strategy: PaddingStrategy) -> Self {
        self.config.padding = strategy;
        self
    }

    /// Set truncation strategy
    pub fn truncation(mut self, strategy: TruncationStrategy) -> Self {
        self.config.truncation = strategy;
        self
    }

    /// Set stride for overflow
    pub fn stride(mut self, stride: usize) -> Self {
        self.config.stride = stride;
        self
    }

    /// Set whether to add special tokens
    pub fn add_special_tokens(mut self, add: bool) -> Self {
        self.config.add_special_tokens = add;
        self
    }

    /// Build the configuration
    pub fn build(self) -> TokenizerConfig {
        self.config
    }
}

impl Default for TokenizerConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = TokenizerConfigBuilder::new()
            .max_length(512)
            .padding(PaddingStrategy::Longest)
            .truncation(TruncationStrategy::LongestFirst)
            .stride(128)
            .build();

        assert_eq!(config.max_length, Some(512));
        assert_eq!(config.padding, PaddingStrategy::Longest);
        assert_eq!(config.truncation, TruncationStrategy::LongestFirst);
        assert_eq!(config.stride, 128);
    }
}
