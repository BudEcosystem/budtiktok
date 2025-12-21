//! Error types compatible with HuggingFace tokenizers

use thiserror::Error;

/// Error type compatible with `tokenizers::Error`
#[derive(Error, Debug)]
pub enum Error {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Encoding error
    #[error("Encoding error: {0}")]
    Encoding(String),

    /// Decoding error
    #[error("Decoding error: {0}")]
    Decoding(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// BudTikTok core error
    #[error("Tokenizer error: {0}")]
    Core(#[from] budtiktok_core::Error),
}

impl Error {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub fn encoding(msg: impl Into<String>) -> Self {
        Self::Encoding(msg.into())
    }

    pub fn decoding(msg: impl Into<String>) -> Self {
        Self::Decoding(msg.into())
    }
}

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;
