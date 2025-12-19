//! GPU-accelerated WordPiece tokenizer
//!
//! High-performance batch tokenization using CUDA:
//! - Parallel pre-tokenization on GPU
//! - Batched vocabulary lookups
//! - Efficient memory transfers with pinned memory

use crate::backend::GpuError;

#[cfg(feature = "cuda")]
use crate::cuda::CudaContext;
#[cfg(feature = "cuda")]
use crate::kernels::{PreTokenizeKernel, VocabLookupKernel, WordPieceKernel};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Configuration for GPU WordPiece tokenizer
#[derive(Debug, Clone)]
pub struct GpuWordPieceConfig {
    /// Continuation prefix (e.g., "##")
    pub continuation_prefix: String,
    /// Maximum word length in characters
    pub max_word_length: usize,
    /// Unknown token
    pub unk_token: String,
    /// Whether to lowercase
    pub do_lower_case: bool,
    /// Minimum batch size to use GPU (below this, use CPU)
    pub min_gpu_batch_size: usize,
    /// Maximum batch size per kernel launch
    pub max_batch_size: usize,
}

impl Default for GpuWordPieceConfig {
    fn default() -> Self {
        Self {
            continuation_prefix: "##".to_string(),
            max_word_length: 100,
            unk_token: "[UNK]".to_string(),
            do_lower_case: true,
            min_gpu_batch_size: 16,
            max_batch_size: 4096,
        }
    }
}

impl GpuWordPieceConfig {
    /// Maximum texts to process in a single batch to limit memory usage
    /// Controls how many texts are pre-tokenized before GPU processing
    pub const MAX_TEXTS_PER_BATCH: usize = 100;

    /// Maximum total words to collect before GPU processing
    /// Prevents memory exhaustion from large batches with long texts
    pub const MAX_WORDS_PER_GPU_BATCH: usize = 50_000;
}

/// GPU-accelerated WordPiece tokenizer
#[cfg(feature = "cuda")]
pub struct GpuWordPieceTokenizer {
    context: Arc<CudaContext>,
    pre_tokenize_kernel: PreTokenizeKernel,
    wordpiece_kernel: WordPieceKernel,
    config: GpuWordPieceConfig,
    cls_id: Option<u32>,
    sep_id: Option<u32>,
}

#[cfg(not(feature = "cuda"))]
pub struct GpuWordPieceTokenizer {
    config: GpuWordPieceConfig,
}

#[cfg(feature = "cuda")]
impl GpuWordPieceTokenizer {
    /// Create a new GPU WordPiece tokenizer
    pub fn new(
        context: Arc<CudaContext>,
        vocab: &[(&str, u32)],
        config: GpuWordPieceConfig,
    ) -> Result<Self, GpuError> {
        let pre_tokenize_kernel = PreTokenizeKernel::new(&context)?;
        let wordpiece_kernel = WordPieceKernel::new(&context, vocab, &config.continuation_prefix)?;

        // Find special token IDs
        let cls_id = vocab.iter().find(|(s, _)| *s == "[CLS]").map(|(_, id)| *id);
        let sep_id = vocab.iter().find(|(s, _)| *s == "[SEP]").map(|(_, id)| *id);

        Ok(Self {
            context,
            pre_tokenize_kernel,
            wordpiece_kernel,
            config,
            cls_id,
            sep_id,
        })
    }

    /// Create from a vocabulary map
    pub fn from_vocab_map(
        context: Arc<CudaContext>,
        vocab_map: &std::collections::HashMap<String, u32>,
        config: GpuWordPieceConfig,
    ) -> Result<Self, GpuError> {
        let vocab: Vec<(&str, u32)> = vocab_map.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        Self::new(context, &vocab, config)
    }

    /// Normalize text (lowercase if configured)
    #[inline]
    fn normalize<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        if self.config.do_lower_case {
            if text.bytes().any(|b| b >= b'A' && b <= b'Z') {
                std::borrow::Cow::Owned(text.to_lowercase())
            } else {
                std::borrow::Cow::Borrowed(text)
            }
        } else {
            std::borrow::Cow::Borrowed(text)
        }
    }

    /// Pre-tokenize text into words using GPU
    fn pre_tokenize(&self, text: &str) -> Result<Vec<String>, GpuError> {
        let boundaries = self.pre_tokenize_kernel.find_word_boundaries(text.as_bytes())?;
        Ok(boundaries
            .into_iter()
            .map(|(start, end)| text[start..end].to_string())
            .collect())
    }

    /// Encode a single text
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, GpuError> {
        let normalized = self.normalize(text);
        let words = self.pre_tokenize(&normalized)?;

        if words.is_empty() {
            return Ok(Vec::new());
        }

        let word_refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        let word_tokens = self.wordpiece_kernel.tokenize_words(&word_refs)?;

        let total_tokens: usize = word_tokens.iter().map(|t| t.len()).sum();
        let mut result = Vec::with_capacity(total_tokens);

        for tokens in word_tokens {
            result.extend(tokens);
        }

        Ok(result)
    }

    /// Encode with special tokens ([CLS], [SEP])
    pub fn encode_with_special(&self, text: &str) -> Result<Vec<u32>, GpuError> {
        let mut tokens = Vec::with_capacity(text.len() / 4 + 3);

        if let Some(cls_id) = self.cls_id {
            tokens.push(cls_id);
        }

        tokens.extend(self.encode(text)?);

        if let Some(sep_id) = self.sep_id {
            tokens.push(sep_id);
        }

        Ok(tokens)
    }

    /// Encode a batch of texts using GPU acceleration
    /// Processes texts in batches to prevent memory exhaustion
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, GpuError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // For small batches, use sequential processing
        if texts.len() < self.config.min_gpu_batch_size {
            return texts.iter().map(|t| self.encode(t)).collect();
        }

        let mut results = Vec::with_capacity(texts.len());

        // Process texts in batches to limit memory usage
        let mut batch_start = 0;
        while batch_start < texts.len() {
            // Determine batch size based on both text count and estimated word count
            let mut batch_end = batch_start;
            let mut estimated_words = 0usize;

            while batch_end < texts.len() {
                // Rough estimate: 1 word per 6 characters
                let text_words = texts[batch_end].len() / 6 + 1;
                if batch_end > batch_start
                    && (batch_end - batch_start >= GpuWordPieceConfig::MAX_TEXTS_PER_BATCH
                        || estimated_words + text_words > GpuWordPieceConfig::MAX_WORDS_PER_GPU_BATCH)
                {
                    break;
                }
                estimated_words += text_words;
                batch_end += 1;
            }

            // Process this batch
            let batch_results = self.encode_batch_internal(&texts[batch_start..batch_end])?;
            results.extend(batch_results);

            batch_start = batch_end;
        }

        Ok(results)
    }

    /// Internal: Process a bounded batch of texts
    fn encode_batch_internal(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, GpuError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Collect all words from texts in this batch
        let mut all_words: Vec<String> = Vec::new();
        let mut text_word_ranges: Vec<(usize, usize)> = Vec::new();

        for text in texts {
            let normalized = self.normalize(text);
            let start = all_words.len();
            let words = self.pre_tokenize(&normalized)?;
            all_words.extend(words);
            let end = all_words.len();
            text_word_ranges.push((start, end));
        }

        if all_words.is_empty() {
            return Ok(vec![Vec::new(); texts.len()]);
        }

        // GPU lookup for all words in this batch
        let word_refs: Vec<&str> = all_words.iter().map(|s| s.as_str()).collect();
        let all_word_tokens = self.wordpiece_kernel.tokenize_words(&word_refs)?;

        // Reassemble results per text
        let mut results = Vec::with_capacity(texts.len());

        for (start, end) in text_word_ranges {
            let mut text_tokens = Vec::new();
            for word_tokens in &all_word_tokens[start..end] {
                text_tokens.extend(word_tokens);
            }
            results.push(text_tokens);
        }

        Ok(results)
    }

    /// Encode batch with special tokens
    pub fn encode_batch_with_special(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, GpuError> {
        let base_results = self.encode_batch(texts)?;

        let cls_id = self.cls_id;
        let sep_id = self.sep_id;

        Ok(base_results
            .into_iter()
            .map(|mut tokens| {
                let mut result = Vec::with_capacity(tokens.len() + 2);
                if let Some(cls) = cls_id {
                    result.push(cls);
                }
                result.append(&mut tokens);
                if let Some(sep) = sep_id {
                    result.push(sep);
                }
                result
            })
            .collect())
    }

    /// Get device information
    pub fn device_info(&self) -> &crate::cuda::CudaDevice {
        self.context.device_info()
    }

    /// Synchronize GPU operations
    pub fn synchronize(&self) -> Result<(), GpuError> {
        self.context.synchronize()
    }
}

#[cfg(not(feature = "cuda"))]
impl GpuWordPieceTokenizer {
    pub fn new(
        _vocab: &[(&str, u32)],
        config: GpuWordPieceConfig,
    ) -> Result<Self, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn encode(&self, _text: &str) -> Result<Vec<u32>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn encode_batch(&self, _texts: &[&str]) -> Result<Vec<Vec<u32>>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn encode_with_special(&self, _text: &str) -> Result<Vec<u32>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn encode_batch_with_special(&self, _texts: &[&str]) -> Result<Vec<Vec<u32>>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = GpuWordPieceConfig::default();
        assert_eq!(config.continuation_prefix, "##");
        assert_eq!(config.max_word_length, 100);
        assert!(config.do_lower_case);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_tokenizer_creation() {
        use crate::cuda::is_cuda_available;

        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return;
        }

        let ctx = Arc::new(CudaContext::new(0).expect("Failed to create context"));

        let vocab = vec![
            ("[PAD]", 0u32),
            ("[UNK]", 1),
            ("[CLS]", 2),
            ("[SEP]", 3),
            ("hello", 4),
            ("world", 5),
            ("##ing", 6),
        ];

        let tokenizer = GpuWordPieceTokenizer::new(ctx, &vocab, GpuWordPieceConfig::default());
        assert!(tokenizer.is_ok());
    }
}
