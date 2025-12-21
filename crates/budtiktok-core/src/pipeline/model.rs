//! TokenizerModel trait and model wrappers
//!
//! This module defines the core tokenization model trait that provides a unified
//! interface for all tokenizer types. Instead of wrapping each tokenizer type
//! individually, we use the existing `Tokenizer` trait to provide a consistent API.

use std::collections::HashMap;

use crate::error::Result;
use crate::tokenizer::Tokenizer;
use crate::vocab::Vocabulary;

/// A single token produced by a model
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    /// The token ID
    pub id: u32,
    /// The token string value
    pub value: String,
    /// The byte offsets (start, end) in the original text
    pub offsets: (usize, usize),
}

impl Token {
    /// Create a new token
    pub fn new(id: u32, value: impl Into<String>, offsets: (usize, usize)) -> Self {
        Self {
            id,
            value: value.into(),
            offsets,
        }
    }
}

/// Core tokenization model trait
///
/// This provides the core tokenization capability. The pipeline uses this trait
/// to perform tokenization while managing added tokens, normalization, and
/// post-processing separately.
pub trait TokenizerModel: Send + Sync {
    /// Tokenize a single pre-processed segment of text
    fn tokenize(&self, text: &str) -> Result<Vec<Token>>;

    /// Get the token ID for a token string
    fn token_to_id(&self, token: &str) -> Option<u32>;

    /// Get the token string for an ID
    fn id_to_token(&self, id: u32) -> Option<String>;

    /// Get the vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get the vocabulary reference
    fn get_vocabulary(&self) -> &Vocabulary;

    /// Get the vocabulary as a HashMap
    fn vocab_to_hashmap(&self) -> HashMap<String, u32> {
        self.get_vocabulary().iter().map(|(k, v)| (k.to_string(), v)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_creation() {
        let token = Token::new(42, "hello", (0, 5));
        assert_eq!(token.id, 42);
        assert_eq!(token.value, "hello");
        assert_eq!(token.offsets, (0, 5));
    }
}
