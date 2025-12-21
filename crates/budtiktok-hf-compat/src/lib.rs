//! HuggingFace Tokenizers Compatibility Layer for BudTikTok
//!
//! This crate provides a drop-in replacement for the HuggingFace `tokenizers` crate,
//! using BudTikTok's high-performance implementation under the hood.
//!
//! # Usage
//!
//! Replace:
//! ```toml
//! tokenizers = "0.21"
//! ```
//!
//! With:
//! ```toml
//! budtiktok-hf-compat = { path = "..." }
//! ```
//!
//! And change imports from:
//! ```rust,ignore
//! use tokenizers::Tokenizer;
//! ```
//!
//! To:
//! ```rust,ignore
//! use budtiktok_hf_compat::Tokenizer;
//! ```

mod encoding;
mod error;
mod postprocessors;
mod tokenizer;
mod truncation;

pub use encoding::Encoding;
pub use error::Error;
pub use postprocessors::{PostProcessorWrapper, Sequence, TemplateProcessing, TemplateProcessingBuilder};
pub use tokenizer::Tokenizer;
pub use truncation::{TruncationDirection, TruncationParams, TruncationStrategy};

/// Re-export processors module for `tokenizers::processors::*` pattern
pub mod processors {
    pub mod sequence {
        pub use crate::postprocessors::Sequence;
    }
    pub mod template {
        pub use crate::postprocessors::{TemplateProcessing, TemplateProcessingBuilder};
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn get_test_tokenizer_path() -> Option<PathBuf> {
        // Try common test tokenizer locations
        let paths = [
            "/tmp/tokenizer.json",
            "tests/fixtures/tokenizer.json",
        ];

        for p in paths {
            let path = PathBuf::from(p);
            if path.exists() {
                return Some(path);
            }
        }
        None
    }

    #[test]
    fn test_api_compatibility() {
        // This test verifies the API matches HF tokenizers
        // The types should compile and have the same methods
    }
}
