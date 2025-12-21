//! Tokenizer compatible with HuggingFace tokenizers
//!
//! This module provides a `Tokenizer` struct that matches the API of
//! `tokenizers::Tokenizer` from the HuggingFace tokenizers crate.
//!
//! # Auto-Configuration
//!
//! On first tokenizer load, BudTikTok automatically detects hardware capabilities
//! and configures optimal settings:
//! - SIMD backend selection (AVX2/AVX-512/NEON)
//! - Thread pool sizing based on CPU cores
//! - Cache sizing based on available memory
//!
//! This happens transparently - no code changes required.

use std::path::Path;
use std::sync::Arc;
use std::sync::Once;

use budtiktok_core::pipeline::TokenizerPipeline;
use budtiktok_core::get_auto_config;

use crate::encoding::Encoding;
use crate::error::{Error, Result};
use crate::postprocessors::PostProcessorWrapper;
use crate::truncation::TruncationParams;

/// Log auto-config info once on first tokenizer load
static LOG_AUTO_CONFIG: Once = Once::new();

/// Inner shared state for the tokenizer
///
/// This is wrapped in Arc for O(1) cloning. The tokenizer pipeline itself
/// is thread-safe (uses internal RwLock for added vocabulary).
struct TokenizerInner {
    /// The core BudTikTok tokenizer pipeline
    pipeline: TokenizerPipeline,
    /// Source JSON for serialization (optional)
    source_json: String,
}

/// Tokenizer compatible with HuggingFace `tokenizers::Tokenizer`
///
/// This struct wraps a BudTikTok tokenizer and provides the same API
/// as the HuggingFace tokenizers crate.
///
/// # Thread Safety
///
/// This tokenizer is thread-safe and can be cloned cheaply (O(1)) for use
/// across multiple threads. Cloning only increments a reference counter.
pub struct Tokenizer {
    /// Shared inner state (Arc for cheap cloning)
    inner: Arc<TokenizerInner>,
    /// Per-instance truncation parameters (can differ between clones)
    truncation: Option<TruncationParams>,
    /// Per-instance padding flag (can differ between clones)
    padding: bool,
    /// Per-instance post-processor (can differ between clones)
    post_processor: Option<PostProcessorWrapper>,
}

impl Clone for Tokenizer {
    fn clone(&self) -> Self {
        // O(1) clone - just increment Arc reference counter
        Self {
            inner: Arc::clone(&self.inner),
            truncation: self.truncation.clone(),
            padding: self.padding,
            post_processor: self.post_processor.clone(),
        }
    }
}

impl Tokenizer {
    /// Load a tokenizer from a file
    ///
    /// This is the primary way to create a tokenizer, matching
    /// `Tokenizer::from_file()` from the HuggingFace crate.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = std::fs::read_to_string(path.as_ref())?;
        Self::from_str(&json)
    }

    /// Load a tokenizer from a JSON string
    ///
    /// On first tokenizer load, auto-configuration is initialized to detect
    /// optimal SIMD and parallelism settings for the current hardware.
    pub fn from_str(json: &str) -> Result<Self> {
        // Initialize auto-config on first load (logs hardware detection)
        LOG_AUTO_CONFIG.call_once(|| {
            let config = get_auto_config();
            // Log at trace level to avoid noise in production
            #[cfg(feature = "tracing")]
            tracing::debug!(
                "BudTikTok auto-config: ISA={}, cores={}, SIMD_pretok={}, cache_size={}",
                config.best_isa,
                config.physical_cores,
                config.use_simd_pretokenizer,
                config.cache_size
            );
            let _ = config; // Suppress unused warning when tracing is disabled
        });

        let pipeline = TokenizerPipeline::from_str(json)?;
        Ok(Self {
            inner: Arc::new(TokenizerInner {
                pipeline,
                source_json: json.to_string(),
            }),
            truncation: None,
            padding: false,
            post_processor: None,
        })
    }

    /// Set truncation parameters
    ///
    /// Returns `&mut Self` to allow method chaining like HF tokenizers.
    pub fn with_truncation(&mut self, params: Option<TruncationParams>) -> Result<&mut Self> {
        self.truncation = params;
        Ok(self)
    }

    /// Set padding
    ///
    /// Returns `&mut Self` to allow method chaining.
    pub fn with_padding(&mut self, padding: Option<()>) -> &mut Self {
        self.padding = padding.is_some();
        self
    }

    /// Get the current post-processor
    ///
    /// Returns a reference to the post-processor if one is set.
    pub fn get_post_processor(&self) -> Option<&PostProcessorWrapper> {
        self.post_processor.as_ref()
    }

    /// Set post-processor
    ///
    /// Returns `&mut Self` to allow method chaining.
    /// Accepts anything that can be converted into a PostProcessorWrapper.
    pub fn with_post_processor<P: Into<PostProcessorWrapper>>(&mut self, post_processor: Option<P>) -> &mut Self {
        self.post_processor = post_processor.map(|p| p.into());
        self
    }

    /// Encode a single text
    ///
    /// This matches the HF tokenizers API:
    /// `tokenizer.encode::<&str>(text, add_special_tokens)`
    pub fn encode<S>(&self, input: S, add_special_tokens: bool) -> Result<Encoding>
    where
        S: AsEncodable,
    {
        input.encode_with(self, add_special_tokens)
    }

    /// Encode text internally
    fn encode_single(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> {
        let mut encoding: Encoding = self.inner.pipeline.encode(text, add_special_tokens)?.into();

        // Apply truncation if set
        if let Some(ref trunc) = self.truncation {
            encoding.truncate(trunc.max_length, trunc.stride);
        }

        Ok(encoding)
    }

    /// Encode a pair of texts
    fn encode_pair(&self, text1: String, text2: String, add_special_tokens: bool) -> Result<Encoding> {
        let mut encoding: Encoding = self.inner.pipeline.encode_pair(&text1, &text2, add_special_tokens)?.into();

        // Apply truncation if set
        if let Some(ref trunc) = self.truncation {
            encoding.truncate(trunc.max_length, trunc.stride);
        }

        Ok(encoding)
    }

    /// Encode a batch of texts
    ///
    /// This matches the HF tokenizers API:
    /// `tokenizer.encode_batch(texts, add_special_tokens)`
    ///
    /// Uses parallel processing for better throughput on large batches.
    pub fn encode_batch<S>(&self, inputs: Vec<S>, add_special_tokens: bool) -> Result<Vec<Encoding>>
    where
        S: AsRef<str> + Send + Sync,
    {
        let text_refs: Vec<&str> = inputs.iter().map(|s| s.as_ref()).collect();
        let encodings = self.inner.pipeline.encode_batch(&text_refs, add_special_tokens)?;

        let mut results: Vec<Encoding> = encodings.into_iter().map(|e| e.into()).collect();

        // Apply truncation if set
        if let Some(ref trunc) = self.truncation {
            for encoding in &mut results {
                encoding.truncate(trunc.max_length, trunc.stride);
            }
        }

        Ok(results)
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner.pipeline.decode(ids, skip_special_tokens).map_err(Error::from)
    }

    /// Get the vocabulary size
    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        self.inner.pipeline.get_vocab_size(with_added_tokens)
    }

    /// Convert a token to its ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.pipeline.token_to_id(token)
    }

    /// Convert an ID to its token
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.pipeline.id_to_token(id)
    }
}

/// Trait for types that can be encoded
///
/// This enables the generic `encode::<T>()` API that HF tokenizers uses.
pub trait AsEncodable {
    fn encode_with(self, tokenizer: &Tokenizer, add_special_tokens: bool) -> Result<Encoding>;
}

impl AsEncodable for &str {
    fn encode_with(self, tokenizer: &Tokenizer, add_special_tokens: bool) -> Result<Encoding> {
        tokenizer.encode_single(self, add_special_tokens)
    }
}

impl AsEncodable for String {
    fn encode_with(self, tokenizer: &Tokenizer, add_special_tokens: bool) -> Result<Encoding> {
        tokenizer.encode_single(&self, add_special_tokens)
    }
}

impl AsEncodable for &String {
    fn encode_with(self, tokenizer: &Tokenizer, add_special_tokens: bool) -> Result<Encoding> {
        tokenizer.encode_single(self, add_special_tokens)
    }
}

impl AsEncodable for (String, String) {
    fn encode_with(self, tokenizer: &Tokenizer, add_special_tokens: bool) -> Result<Encoding> {
        tokenizer.encode_pair(self.0, self.1, add_special_tokens)
    }
}

impl AsEncodable for (&str, &str) {
    fn encode_with(self, tokenizer: &Tokenizer, add_special_tokens: bool) -> Result<Encoding> {
        tokenizer.encode_pair(self.0.to_string(), self.1.to_string(), add_special_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_signature() {
        // This test just verifies the API compiles correctly
        // It doesn't run actual tokenization
    }
}
