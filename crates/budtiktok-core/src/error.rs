//! Error types for BudTikTok core
//!
//! This module defines all error types used throughout the tokenization pipeline.

use thiserror::Error;

/// Result type alias for BudTikTok operations
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for BudTikTok
#[derive(Error, Debug)]
pub enum Error {
    /// Failed to load vocabulary file
    #[error("Failed to load vocabulary: {0}")]
    VocabLoad(String),

    /// Failed to parse tokenizer configuration
    #[error("Invalid tokenizer configuration: {0}")]
    InvalidConfig(String),

    /// Token not found in vocabulary
    #[error("Unknown token: {0}")]
    UnknownToken(String),

    /// Token ID not found in vocabulary
    #[error("Unknown token ID: {0}")]
    UnknownTokenId(u32),

    /// Input text exceeds maximum length
    #[error("Input exceeds maximum length: {0} > {1}")]
    InputTooLong(usize, usize),

    /// Invalid UTF-8 encoding
    #[error("Invalid UTF-8: {0}")]
    InvalidUtf8(String),

    /// Normalization error
    #[error("Normalization error: {0}")]
    Normalization(String),

    /// Pre-tokenization error
    #[error("Pre-tokenization error: {0}")]
    PreTokenization(String),

    /// Model-specific tokenization error
    #[error("Tokenization error: {0}")]
    Tokenization(String),

    /// Post-processing error
    #[error("Post-processing error: {0}")]
    PostProcessing(String),

    /// Decoding error
    #[error("Decoding error: {0}")]
    Decoding(String),

    /// Cache error
    #[error("Cache error: {0}")]
    Cache(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl Error {
    /// Create a vocab load error
    pub fn vocab_load(msg: impl Into<String>) -> Self {
        Error::VocabLoad(msg.into())
    }

    /// Create an invalid config error
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Error::InvalidConfig(msg.into())
    }

    /// Create an unknown token error
    pub fn unknown_token(token: impl Into<String>) -> Self {
        Error::UnknownToken(token.into())
    }

    /// Create an unknown token ID error
    pub fn unknown_token_id(id: u32) -> Self {
        Error::UnknownTokenId(id)
    }

    /// Create an input too long error
    pub fn input_too_long(actual: usize, max: usize) -> Self {
        Error::InputTooLong(actual, max)
    }

    /// Create a tokenization error
    pub fn tokenization(msg: impl Into<String>) -> Self {
        Error::Tokenization(msg.into())
    }
}
