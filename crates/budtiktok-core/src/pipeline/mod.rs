//! TokenizerPipeline - HuggingFace-compatible tokenizer pipeline
//!
//! This module provides a high-performance tokenizer pipeline that is fully compatible
//! with HuggingFace tokenizers while maintaining BudTikTok's performance advantages.
//!
//! # Features
//!
//! - **Full HF Compatibility**: Same API surface, same JSON format
//! - **Dynamic Token Addition**: Add tokens at runtime with `add_tokens()` and `add_special_tokens()`
//! - **Component Access**: Get and set normalizers, pre-tokenizers, post-processors, and decoders
//! - **Performance**: Preserves all BudTikTok optimizations (SIMD, caching, O(n) algorithms)
//!
//! # Example
//!
//! ```rust,ignore
//! use budtiktok_core::pipeline::TokenizerPipeline;
//! use budtiktok_core::special_tokens::AddedToken;
//!
//! let mut pipeline = TokenizerPipeline::from_file("tokenizer.json")?;
//!
//! // Dynamic token addition
//! pipeline.add_special_tokens(&[
//!     AddedToken::new("<custom>", 30000).special(),
//! ]);
//!
//! // Encode
//! let encoding = pipeline.encode("Hello world", true)?;
//!
//! // Get vocabulary
//! let vocab = pipeline.get_vocab(true);
//! ```

pub mod added_vocabulary;
pub mod model;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use rayon::prelude::*;

use crate::autoconfig::get_auto_config;
use crate::decoder::Decoder;
use crate::encoding::{Encoding, PaddingParams, TruncationParams, pad_encodings, truncate_encoding};
use crate::error::{Error, Result};
use crate::normalizer::Normalizer;
use crate::postprocessor::PostProcessor;
use crate::pretokenizer::PreTokenizer;
use crate::special_tokens::AddedToken;
use crate::tokenizer::Tokenizer;
use crate::vocab::Vocabulary;

pub use added_vocabulary::{AddedVocabulary, ExtractedSegment};
pub use model::{Token, TokenizerModel};

/// Input type for encoding operations
#[derive(Debug, Clone)]
pub enum EncodeInput<'a> {
    /// Single text input
    Single(&'a str),
    /// Pair of texts (for sequence classification, etc.)
    Dual(&'a str, &'a str),
}

impl<'a> From<&'a str> for EncodeInput<'a> {
    fn from(text: &'a str) -> Self {
        Self::Single(text)
    }
}

impl<'a> From<(&'a str, &'a str)> for EncodeInput<'a> {
    fn from((text, text_pair): (&'a str, &'a str)) -> Self {
        Self::Dual(text, text_pair)
    }
}

impl<'a> EncodeInput<'a> {
    /// Get the first (or only) text
    pub fn text(&self) -> &str {
        match self {
            Self::Single(t) => t,
            Self::Dual(t, _) => t,
        }
    }

    /// Get the second text if present
    pub fn text_pair(&self) -> Option<&str> {
        match self {
            Self::Single(_) => None,
            Self::Dual(_, t) => Some(t),
        }
    }
}

/// High-performance tokenizer pipeline with HF-compatible API
///
/// This struct provides a complete tokenization pipeline that mirrors the HuggingFace
/// tokenizers architecture while preserving BudTikTok's performance optimizations.
pub struct TokenizerPipeline {
    /// The core tokenizer (wraps existing implementations)
    inner: Box<dyn Tokenizer>,

    /// Optional normalizer (applied before tokenization)
    normalizer: Option<Arc<dyn Normalizer + Send + Sync>>,

    /// Optional pre-tokenizer (splits text before model)
    pre_tokenizer: Option<Arc<dyn PreTokenizer + Send + Sync>>,

    /// Optional post-processor (adds special tokens)
    post_processor: Option<Arc<dyn PostProcessor + Send + Sync>>,

    /// Optional decoder (converts IDs back to text)
    decoder: Option<Arc<dyn Decoder + Send + Sync>>,

    /// Dynamic vocabulary for added tokens (thread-safe)
    added_vocabulary: RwLock<AddedVocabulary>,

    /// Truncation configuration
    truncation: Option<TruncationParams>,

    /// Padding configuration
    padding: Option<PaddingParams>,
}

impl TokenizerPipeline {
    /// Create a pipeline from an existing tokenizer
    pub fn new(tokenizer: Box<dyn Tokenizer>) -> Self {
        Self {
            inner: tokenizer,
            normalizer: None,
            pre_tokenizer: None,
            post_processor: None,
            decoder: None,
            added_vocabulary: RwLock::new(AddedVocabulary::new()),
            truncation: None,
            padding: None,
        }
    }

    /// Load a tokenizer pipeline from a JSON file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path.as_ref())
            .map_err(|e| Error::VocabLoad(format!("Failed to read tokenizer file: {}", e)))?;
        Self::from_str(&json)
    }

    /// Load a tokenizer pipeline from a JSON string
    pub fn from_str(json: &str) -> Result<Self> {
        use crate::config::TokenizerConfig;
        use crate::postprocessor::{RobertaPostProcessor, BertPostProcessor, SpecialToken};

        // Parse the full config to extract all components
        let config = TokenizerConfig::from_json(json)?;

        // Load the tokenizer
        let tokenizer = crate::loader::load_tokenizer_from_str(json)?;
        let mut pipeline = Self::new(tokenizer);

        // Set post-processor if configured
        if let Some(ref pp_config) = config.post_processor {
            match pp_config.processor_type.as_str() {
                "RobertaProcessing" => {
                    // RoBERTa-style: <s> content </s>
                    // Used by MPNet, RoBERTa, ALBERT, etc.
                    let cls_info = pp_config.cls.as_ref();
                    let sep_info = pp_config.sep.as_ref();

                    let (cls_token, cls_id) = cls_info
                        .map(|(t, id)| (t.as_str(), *id))
                        .unwrap_or(("<s>", 0));
                    let (sep_token, sep_id) = sep_info
                        .map(|(t, id)| (t.as_str(), *id))
                        .unwrap_or(("</s>", 2));

                    let cls = SpecialToken::new(cls_token, cls_id);
                    let sep = SpecialToken::new(sep_token, sep_id);
                    let processor = RobertaPostProcessor::new(cls, sep);
                    pipeline.with_post_processor(processor);
                }
                "BertProcessing" => {
                    // BERT-style: [CLS] content [SEP]
                    let cls_info = pp_config.cls.as_ref();
                    let sep_info = pp_config.sep.as_ref();

                    let (cls_token, cls_id) = cls_info
                        .map(|(t, id)| (t.as_str(), *id))
                        .unwrap_or(("[CLS]", 101));
                    let (sep_token, sep_id) = sep_info
                        .map(|(t, id)| (t.as_str(), *id))
                        .unwrap_or(("[SEP]", 102));

                    let cls = SpecialToken::new(cls_token, cls_id);
                    let sep = SpecialToken::new(sep_token, sep_id);
                    let processor = BertPostProcessor::new(cls, sep);
                    pipeline.with_post_processor(processor);
                }
                "TemplateProcessing" => {
                    // Template-based post-processing
                    // Parse the template to determine the correct special tokens
                    if let Some(ref single) = pp_config.single {
                        // Check if template contains [CLS] or <s>
                        let uses_roberta = single.iter().any(|item| {
                            if let crate::config::TemplateItem::SpecialToken { special_token } = item {
                                special_token.id == "<s>"
                            } else {
                                false
                            }
                        });
                        let uses_bert = single.iter().any(|item| {
                            if let crate::config::TemplateItem::SpecialToken { special_token } = item {
                                special_token.id.contains("CLS")
                            } else {
                                false
                            }
                        });

                        if uses_roberta {
                            // RoBERTa-style: <s> content </s>
                            if let Some(ref special_tokens) = pp_config.special_tokens {
                                let cls_id = special_tokens.get("<s>")
                                    .and_then(|st| st.ids.first().copied())
                                    .unwrap_or(0);
                                let sep_id = special_tokens.get("</s>")
                                    .and_then(|st| st.ids.first().copied())
                                    .unwrap_or(2);

                                let cls = SpecialToken::new("<s>", cls_id);
                                let sep = SpecialToken::new("</s>", sep_id);
                                let processor = RobertaPostProcessor::new(cls, sep);
                                pipeline.with_post_processor(processor);
                            }
                        } else if uses_bert {
                            // BERT-style: [CLS] content [SEP]
                            if let Some(ref special_tokens) = pp_config.special_tokens {
                                let cls_id = special_tokens.get("[CLS]")
                                    .and_then(|st| st.ids.first().copied())
                                    .unwrap_or(101);
                                let sep_id = special_tokens.get("[SEP]")
                                    .and_then(|st| st.ids.first().copied())
                                    .unwrap_or(102);

                                let processor = BertPostProcessor::with_ids(cls_id, sep_id);
                                pipeline.with_post_processor(processor);
                            }
                        }
                    }
                }
                _ => {
                    // Unknown processor type - skip silently
                    // This allows graceful degradation for unsupported processor types
                }
            }
        }

        Ok(pipeline)
    }

    // ========================================================================
    // Encode Methods (HF-Compatible)
    // ========================================================================

    /// Encode text into tokens (HF-compatible signature)
    pub fn encode<'s, E>(&self, input: E, add_special_tokens: bool) -> Result<Encoding>
    where
        E: Into<EncodeInput<'s>>,
    {
        let input = input.into();

        // Record statistics for auto-tuning
        let auto_config = get_auto_config();
        auto_config.record_encode(input.text().len());

        // Get added vocabulary read lock
        let added_vocab = self.added_vocabulary.read();

        // Check for added tokens in text
        let text = input.text();

        // If we have added tokens, we need to handle them specially
        if !added_vocab.is_empty() {
            // Extract segments, splitting on added tokens
            let segments = added_vocab.extract_and_normalize::<NoopNormalizer>(
                text,
                None,
            );

            // Build encoding from segments
            let mut encoding = Encoding::with_capacity(16);
            let mut word_id: u32 = 0;

            for segment in segments {
                if segment.is_added_token {
                    // Added token - add directly
                    if let Some(token_id) = segment.token_id {
                        encoding.push(
                            token_id,
                            segment.content.clone(),
                            (0, segment.content.len()),
                            None,
                            Some(0),
                            segment.is_special,
                        );
                    }
                } else {
                    // Regular text - use inner tokenizer
                    let inner_encoding = self.inner.encode(&segment.content, false)?;
                    for i in 0..inner_encoding.len() {
                        encoding.push(
                            inner_encoding.get_ids()[i],
                            inner_encoding.get_tokens()[i].clone(),
                            inner_encoding.get_offsets()[i],
                            Some(word_id),
                            Some(0),
                            inner_encoding.get_special_tokens_mask()[i] == 1,
                        );
                    }
                    if !inner_encoding.is_empty() {
                        word_id += 1;
                    }
                }
            }

            // Post-process if needed
            if add_special_tokens {
                if let Some(ref pp) = self.post_processor {
                    encoding = pp.process(encoding, add_special_tokens);
                }
            }

            // Apply truncation
            if let Some(ref trunc) = self.truncation {
                truncate_encoding(&mut encoding, trunc);
            }

            Ok(encoding)
        } else {
            // No added tokens - use inner tokenizer directly
            // Pass add_special_tokens=false to inner tokenizer if we have a post-processor
            let has_post_processor = self.post_processor.is_some();
            let inner_add_special = add_special_tokens && !has_post_processor;
            let mut encoding = self.inner.encode(text, inner_add_special)?;

            // Apply post-processor if present
            if add_special_tokens {
                if let Some(ref pp) = self.post_processor {
                    encoding = pp.process(encoding, add_special_tokens);
                }
            }

            // Apply truncation
            if let Some(ref trunc) = self.truncation {
                truncate_encoding(&mut encoding, trunc);
            }

            Ok(encoding)
        }
    }

    /// Encode a pair of texts
    pub fn encode_pair(
        &self,
        text: &str,
        text_pair: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        self.inner.encode_pair(text, text_pair, add_special_tokens)
    }

    /// Batch encode with optional parallelization
    ///
    /// Automatically uses optimal parallelization based on hardware and batch size.
    /// This is the optimized batch path that minimizes per-item overhead.
    pub fn encode_batch<'s>(
        &self,
        inputs: &[&'s str],
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        // Record statistics once for the whole batch (not per-item)
        let auto_config = get_auto_config();
        auto_config.record_batch_encode(inputs.len());

        // Check added vocabulary once (not per-item)
        let has_added_tokens = !self.added_vocabulary.read().is_empty();
        let has_post_processor = self.post_processor.is_some();
        let has_truncation = self.truncation.is_some();

        // Fast path: no added tokens, no post-processor, no truncation
        // Use inner tokenizer's batch method directly for maximum performance
        if !has_added_tokens && !has_post_processor && !has_truncation {
            return self.inner.encode_batch(inputs, add_special_tokens);
        }

        // Slow path: need per-item processing for added tokens/post-processing
        // But still avoid repeated lock acquisition by reading once
        let added_vocab = self.added_vocabulary.read();

        let mut encodings: Vec<Encoding> = if added_vocab.is_empty() {
            // No added tokens - use parallel inner encoding
            // If we have a post-processor, don't let inner tokenizer add special tokens
            let inner_add_special = add_special_tokens && !has_post_processor;
            inputs
                .par_iter()
                .map(|&text| {
                    let mut encoding = self.inner.encode(text, inner_add_special)?;

                    // Apply post-processing if needed
                    if add_special_tokens {
                        if let Some(ref pp) = self.post_processor {
                            encoding = pp.process(encoding, add_special_tokens);
                        }
                    }

                    // Apply truncation
                    if let Some(ref trunc) = self.truncation {
                        truncate_encoding(&mut encoding, trunc);
                    }

                    Ok(encoding)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            // Has added tokens - need segment extraction
            inputs
                .par_iter()
                .map(|&text| {
                    let segments = added_vocab.extract_and_normalize::<NoopNormalizer>(
                        text,
                        None,
                    );

                    let mut encoding = Encoding::with_capacity(16);
                    let mut word_id: u32 = 0;

                    for segment in segments {
                        if segment.is_added_token {
                            if let Some(token_id) = segment.token_id {
                                encoding.push(
                                    token_id,
                                    segment.content.clone(),
                                    (0, segment.content.len()),
                                    None,
                                    Some(0),
                                    segment.is_special,
                                );
                            }
                        } else {
                            let inner_encoding = self.inner.encode(&segment.content, false)?;
                            for i in 0..inner_encoding.len() {
                                encoding.push(
                                    inner_encoding.get_ids()[i],
                                    inner_encoding.get_tokens()[i].clone(),
                                    inner_encoding.get_offsets()[i],
                                    Some(word_id),
                                    Some(0),
                                    inner_encoding.get_special_tokens_mask()[i] == 1,
                                );
                            }
                            if !inner_encoding.is_empty() {
                                word_id += 1;
                            }
                        }
                    }

                    if add_special_tokens {
                        if let Some(ref pp) = self.post_processor {
                            encoding = pp.process(encoding, add_special_tokens);
                        }
                    }

                    if let Some(ref trunc) = self.truncation {
                        truncate_encoding(&mut encoding, trunc);
                    }

                    Ok(encoding)
                })
                .collect::<Result<Vec<_>>>()?
        };

        // Apply padding after batch (needs all lengths)
        if let Some(ref pad) = self.padding {
            pad_encodings(&mut encodings, pad);
        }

        Ok(encodings)
    }

    // ========================================================================
    // Decode Methods (HF-Compatible)
    // ========================================================================

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let added_vocab = self.added_vocabulary.read();

        if added_vocab.is_empty() {
            // No added tokens - use inner tokenizer directly
            return self.inner.decode(ids, skip_special_tokens);
        }

        // Build tokens list, handling added tokens
        let tokens: Vec<String> = ids.iter().filter_map(|&id| {
            // Check added vocabulary first
            if let Some(token) = added_vocab.id_to_token(id) {
                if skip_special_tokens && token.special {
                    return None;
                }
                return Some(token.content.clone());
            }
            // Fall back to inner tokenizer vocabulary
            self.inner.id_to_token(id).map(|s| s.to_string())
        }).collect();

        // Apply decoder if we have one
        if let Some(ref decoder) = self.decoder {
            decoder.decode(tokens)
        } else {
            Ok(tokens.join(""))
        }
    }

    /// Decode a batch of token ID sequences
    pub fn decode_batch(
        &self,
        ids_batch: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        ids_batch
            .par_iter()
            .map(|ids| self.decode(ids, skip_special_tokens))
            .collect()
    }

    // ========================================================================
    // Dynamic Token Addition (HF-Compatible)
    // ========================================================================

    /// Add tokens to vocabulary (returns count of newly added)
    pub fn add_tokens(&self, tokens: &[AddedToken]) -> usize {
        let mut added_vocab = self.added_vocabulary.write();
        added_vocab.add_tokens::<NoopNormalizer>(
            tokens,
            self.inner.vocab_size(),
            None,
        )
    }

    /// Add special tokens (returns count of newly added)
    pub fn add_special_tokens(&self, tokens: &[AddedToken]) -> usize {
        let mut added_vocab = self.added_vocabulary.write();
        added_vocab.add_special_tokens::<NoopNormalizer>(
            tokens,
            self.inner.vocab_size(),
            None,
        )
    }

    /// Get vocabulary with optional added tokens
    pub fn get_vocab(&self, with_added_tokens: bool) -> HashMap<String, u32> {
        let mut vocab: HashMap<String, u32> = self.inner.vocabulary()
            .iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        if with_added_tokens {
            let added = self.added_vocabulary.read();
            for (token, id) in added.iter() {
                vocab.insert(token.to_string(), id);
            }
        }

        vocab
    }

    /// Get total vocabulary size
    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        let base = self.inner.vocab_size();
        if with_added_tokens {
            base + self.added_vocabulary.read().len()
        } else {
            base
        }
    }

    // ========================================================================
    // Component Accessors/Setters (HF-Compatible)
    // ========================================================================

    /// Set normalizer
    pub fn with_normalizer<N>(&mut self, normalizer: N) -> &mut Self
    where
        N: Normalizer + Send + Sync + 'static,
    {
        self.normalizer = Some(Arc::new(normalizer));
        self
    }

    /// Get normalizer
    pub fn get_normalizer(&self) -> Option<&(dyn Normalizer + Send + Sync)> {
        self.normalizer.as_ref().map(|n| n.as_ref())
    }

    /// Set pre-tokenizer
    pub fn with_pre_tokenizer<PT>(&mut self, pre_tokenizer: PT) -> &mut Self
    where
        PT: PreTokenizer + Send + Sync + 'static,
    {
        self.pre_tokenizer = Some(Arc::new(pre_tokenizer));
        self
    }

    /// Get pre-tokenizer
    pub fn get_pre_tokenizer(&self) -> Option<&(dyn PreTokenizer + Send + Sync)> {
        self.pre_tokenizer.as_ref().map(|pt| pt.as_ref())
    }

    /// Set post-processor
    pub fn with_post_processor<PP>(&mut self, post_processor: PP) -> &mut Self
    where
        PP: PostProcessor + Send + Sync + 'static,
    {
        self.post_processor = Some(Arc::new(post_processor));
        self
    }

    /// Get post-processor
    pub fn get_post_processor(&self) -> Option<&(dyn PostProcessor + Send + Sync)> {
        self.post_processor.as_ref().map(|pp| pp.as_ref())
    }

    /// Set decoder
    pub fn with_decoder<D>(&mut self, decoder: D) -> &mut Self
    where
        D: Decoder + Send + Sync + 'static,
    {
        self.decoder = Some(Arc::new(decoder));
        self
    }

    /// Get decoder
    pub fn get_decoder(&self) -> Option<&(dyn Decoder + Send + Sync)> {
        self.decoder.as_ref().map(|d| d.as_ref())
    }

    /// Set truncation parameters
    pub fn with_truncation(&mut self, params: Option<TruncationParams>) -> &mut Self {
        self.truncation = params;
        self
    }

    /// Get truncation parameters
    pub fn get_truncation(&self) -> Option<&TruncationParams> {
        self.truncation.as_ref()
    }

    /// Set padding parameters
    pub fn with_padding(&mut self, params: Option<PaddingParams>) -> &mut Self {
        self.padding = params;
        self
    }

    /// Get padding parameters
    pub fn get_padding(&self) -> Option<&PaddingParams> {
        self.padding.as_ref()
    }

    // ========================================================================
    // Token/ID Conversion
    // ========================================================================

    /// Convert a token to its ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        // Check added vocabulary first
        let added = self.added_vocabulary.read();
        if let Some(id) = added.token_to_id(token) {
            return Some(id);
        }
        // Fall back to inner tokenizer
        self.inner.token_to_id(token)
    }

    /// Convert an ID to its token
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        // Check added vocabulary first
        let added = self.added_vocabulary.read();
        if let Some(content) = added.id_to_content(id) {
            return Some(content.to_string());
        }
        // Fall back to inner tokenizer
        self.inner.id_to_token(id).map(|s| s.to_string())
    }

    /// Get the inner tokenizer
    pub fn inner(&self) -> &dyn Tokenizer {
        self.inner.as_ref()
    }

    // ========================================================================
    // Serialization
    // ========================================================================

    /// Save the tokenizer to a file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.inner.save(path.as_ref())
    }
}

// ============================================================================
// NoopNormalizer for internal use
// ============================================================================

/// A no-op normalizer that returns text unchanged
#[derive(Debug, Clone, Default)]
struct NoopNormalizer;

impl Normalizer for NoopNormalizer {
    fn normalize<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        std::borrow::Cow::Borrowed(text)
    }
}

// ============================================================================
// Implement Tokenizer trait for TokenizerPipeline
// ============================================================================

impl Tokenizer for TokenizerPipeline {
    fn vocabulary(&self) -> &Vocabulary {
        self.inner.vocabulary()
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> {
        TokenizerPipeline::encode(self, text, add_special_tokens)
    }

    fn encode_pair(
        &self,
        text: &str,
        text_pair: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        TokenizerPipeline::encode_pair(self, text, text_pair, add_special_tokens)
    }

    fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        TokenizerPipeline::encode_batch(self, texts, add_special_tokens)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        TokenizerPipeline::decode(self, ids, skip_special_tokens)
    }

    fn save(&self, path: &Path) -> Result<()> {
        TokenizerPipeline::save(self, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_input_from_str() {
        let input: EncodeInput = "Hello world".into();
        assert_eq!(input.text(), "Hello world");
        assert!(input.text_pair().is_none());
    }

    #[test]
    fn test_encode_input_from_pair() {
        let input: EncodeInput = ("Hello", "World").into();
        assert_eq!(input.text(), "Hello");
        assert_eq!(input.text_pair(), Some("World"));
    }
}
