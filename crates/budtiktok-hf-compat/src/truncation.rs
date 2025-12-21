//! Truncation types compatible with HuggingFace tokenizers

use serde::{Deserialize, Serialize};

/// Truncation direction - compatible with `tokenizers::TruncationDirection`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TruncationDirection {
    /// Truncate from the left
    Left,
    /// Truncate from the right (default)
    #[default]
    Right,
}

/// Truncation strategy - compatible with `tokenizers::TruncationStrategy`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TruncationStrategy {
    /// Truncate the longest sequence first
    #[default]
    LongestFirst,
    /// Only truncate the first sequence
    OnlyFirst,
    /// Only truncate the second sequence
    OnlySecond,
}

/// Truncation parameters - compatible with `tokenizers::TruncationParams`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TruncationParams {
    /// Maximum length
    pub max_length: usize,
    /// Truncation strategy
    pub strategy: TruncationStrategy,
    /// Stride for overflowing tokens
    pub stride: usize,
    /// Truncation direction
    pub direction: TruncationDirection,
}

impl Default for TruncationParams {
    fn default() -> Self {
        Self {
            max_length: 512,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
            direction: TruncationDirection::Right,
        }
    }
}

impl TruncationParams {
    /// Create new truncation parameters
    pub fn new(max_length: usize) -> Self {
        Self {
            max_length,
            ..Default::default()
        }
    }
}
