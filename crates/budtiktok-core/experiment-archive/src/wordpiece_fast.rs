//! Fast WordPiece Tokenizer
//!
//! Optimized implementation with:
//! - LinMaxMatch O(n) algorithm
//! - SIMD-accelerated pre-tokenization
//! - Parallel batch processing
//! - Zero-allocation inner loops
//! - Cache-friendly data structures

use ahash::AHashMap;
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;

use crate::cache::{ClockCache, TokenCache};
use crate::encoding::Encoding;
use crate::error::{Error, Result};
use crate::linmaxmatch::{FastPreTokenizer, LinMaxMatchTokenizer};
use crate::simd_classify::SimdClassifier;
use crate::tokenizer::Tokenizer;
use crate::vocab::Vocabulary;

/// Configuration for the fast WordPiece tokenizer
#[derive(Debug, Clone)]
pub struct FastWordPieceConfig {
    /// Continuation prefix (default: "##")
    pub continuing_subword_prefix: String,
    /// Maximum characters per word
    pub max_input_chars_per_word: usize,
    /// Unknown token
    pub unk_token: String,
    /// Whether to lowercase text
    pub do_lower_case: bool,
    /// Whether to strip accents
    pub strip_accents: bool,
    /// Whether to tokenize CJK characters individually
    pub tokenize_chinese_chars: bool,
    /// Minimum batch size for parallel processing
    pub parallel_threshold: usize,
    /// Cache size (number of entries)
    pub cache_size: usize,
}

impl Default for FastWordPieceConfig {
    fn default() -> Self {
        Self {
            continuing_subword_prefix: "##".to_string(),
            max_input_chars_per_word: 200,
            unk_token: "[UNK]".to_string(),
            do_lower_case: true,
            strip_accents: true,
            tokenize_chinese_chars: true,
            parallel_threshold: 8, // Use parallelism for batches >= 8
            cache_size: 100_000,
        }
    }
}

/// Fast WordPiece tokenizer with O(n) complexity
pub struct FastWordPieceTokenizer {
    /// LinMaxMatch tokenizer for O(n) word tokenization
    linmaxmatch: Arc<LinMaxMatchTokenizer>,
    /// Fast pre-tokenizer with lookup tables
    pre_tokenizer: FastPreTokenizer,
    /// SIMD classifier for parallel character classification
    simd_classifier: SimdClassifier,
    /// Vocabulary for special tokens and reverse lookup
    vocabulary: Vocabulary,
    /// Configuration
    config: FastWordPieceConfig,
    /// Token cache
    cache: TokenCache,
    /// Pre-allocated buffer for normalization
    norm_buffer: std::cell::UnsafeCell<String>,
}

// Safety: The UnsafeCell is only accessed from &self methods which are
// not called concurrently on the same instance
unsafe impl Sync for FastWordPieceTokenizer {}
unsafe impl Send for FastWordPieceTokenizer {}

impl FastWordPieceTokenizer {
    /// Create a new fast WordPiece tokenizer
    pub fn new(vocabulary: Vocabulary, config: FastWordPieceConfig) -> Self {
        // Build vocabulary map for LinMaxMatch
        let mut vocab_map = AHashMap::new();
        for (token, id) in vocabulary.iter() {
            vocab_map.insert(token.to_string(), id);
        }

        let linmaxmatch = Arc::new(LinMaxMatchTokenizer::new(
            &vocab_map,
            &config.continuing_subword_prefix,
            &config.unk_token,
        ));

        Self {
            linmaxmatch,
            pre_tokenizer: FastPreTokenizer::new(),
            simd_classifier: SimdClassifier::new(),
            vocabulary,
            config: config.clone(),
            cache: ClockCache::new(config.cache_size),
            norm_buffer: std::cell::UnsafeCell::new(String::with_capacity(4096)),
        }
    }

    /// Normalize text in-place (lowercase, strip accents)
    #[inline]
    fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // Fast path: check if normalization is needed
        if !self.config.do_lower_case && !self.config.strip_accents {
            return Cow::Borrowed(text);
        }

        // Check if text is already normalized (ASCII lowercase, no accents)
        let needs_work = text.bytes().any(|b| {
            b >= b'A' && b <= b'Z' || // uppercase
            b >= 0x80 // non-ASCII (potential accents)
        });

        if !needs_work {
            return Cow::Borrowed(text);
        }

        // Perform normalization
        let mut result = if self.config.do_lower_case {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        if self.config.strip_accents {
            result = self.strip_accents_fast(&result);
        }

        Cow::Owned(result)
    }

    /// Fast accent stripping using lookup tables for common cases
    #[inline]
    fn strip_accents_fast(&self, text: &str) -> String {
        use unicode_normalization::UnicodeNormalization;

        // Fast path: check if any non-ASCII characters exist
        if text.bytes().all(|b| b < 0x80) {
            return text.to_string();
        }

        // NFD decomposition and filter combining marks
        text.nfd()
            .filter(|c| !matches!(unicode_general_category::get_general_category(*c),
                unicode_general_category::GeneralCategory::NonspacingMark))
            .collect()
    }

    /// Tokenize a single word using LinMaxMatch (O(n) complexity)
    #[inline]
    fn tokenize_word_to_ids(&self, word: &str) -> Vec<u32> {
        // Check length constraint
        let byte_len = word.len();
        if byte_len > self.config.max_input_chars_per_word {
            let char_count = word.chars().count();
            if char_count > self.config.max_input_chars_per_word {
                return vec![self.vocabulary.unk_token_id().unwrap_or(0)];
            }
        }

        // Only use cache for medium-length words where cache overhead is worthwhile
        // Skip caching for very short words (< 16 bytes) - already fast without cache
        // Skip caching for very long words (> 128 bytes) - cache key allocation overhead
        let use_cache = byte_len >= 16 && byte_len <= 128;

        if use_cache {
            // Allocate cache key only when needed
            let cache_key = word.to_string();
            if let Some(cached) = self.cache.get(&cache_key) {
                return cached;
            }

            // Use LinMaxMatch for O(n) tokenization
            let ids = self.linmaxmatch.tokenize_word(word.as_bytes());
            self.cache.insert(cache_key, ids.clone());
            return ids;
        }

        // Fast path: directly tokenize without cache
        self.linmaxmatch.tokenize_word(word.as_bytes())
    }

    /// Encode a single text (fast path)
    #[inline]
    pub fn encode_fast(&self, text: &str) -> Vec<u32> {
        let normalized = self.normalize(text);
        let norm_bytes = normalized.as_bytes();

        // Use fast pre-tokenizer
        let word_bounds = self.pre_tokenizer.pre_tokenize(norm_bytes);

        // Pre-allocate result vector
        let mut ids = Vec::with_capacity(word_bounds.len() * 2);

        for (start, end) in word_bounds {
            let word = &normalized[start..end];
            let word_ids = self.tokenize_word_to_ids(word);
            ids.extend(word_ids);
        }

        ids
    }

    /// Encode a batch of texts with parallel processing
    pub fn encode_batch_fast(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        let batch_size = texts.len();

        if batch_size == 0 {
            return Vec::new();
        }

        // Use parallelism for batches > 1 (matching BlazeText behavior)
        // Rayon's work-stealing handles load balancing automatically
        if batch_size > 1 {
            texts.par_iter().map(|text| self.encode_fast(text)).collect()
        } else {
            texts.iter().map(|text| self.encode_fast(text)).collect()
        }
    }

    /// Encode with special tokens
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 4 + 3);

        // Add [CLS]
        if let Some(cls_id) = self.vocabulary.cls_token_id() {
            ids.push(cls_id);
        }

        // Encode text
        ids.extend(self.encode_fast(text));

        // Add [SEP]
        if let Some(sep_id) = self.vocabulary.sep_token_id() {
            ids.push(sep_id);
        }

        ids
    }

    /// Encode batch with special tokens
    pub fn encode_batch_with_special(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        let batch_size = texts.len();

        if batch_size == 0 {
            return Vec::new();
        }

        // Use parallelism for batches > 1 (matching BlazeText behavior)
        // Rayon's work-stealing handles load balancing automatically
        if batch_size > 1 {
            texts.par_iter().map(|text| self.encode_with_special(text)).collect()
        } else {
            texts.iter().map(|text| self.encode_with_special(text)).collect()
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> crate::cache::CacheStats {
        self.cache.stats()
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

impl Tokenizer for FastWordPieceTokenizer {
    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> {
        let normalized = self.normalize(text);
        let norm_bytes = normalized.as_bytes();

        // Use fast pre-tokenizer
        let word_bounds = self.pre_tokenizer.pre_tokenize(norm_bytes);

        // Create encoding
        let mut encoding = Encoding::with_capacity(word_bounds.len() * 2);
        let mut word_idx = 0u32;

        for (start, end) in word_bounds {
            let word = &normalized[start..end];
            let word_ids = self.tokenize_word_to_ids(word);

            for (i, &id) in word_ids.iter().enumerate() {
                let token = self
                    .vocabulary
                    .id_to_token(id)
                    .unwrap_or(&self.config.unk_token);

                encoding.push(
                    id,
                    token.to_string(),
                    (start, end),
                    Some(word_idx),
                    Some(0),
                    false,
                );
            }

            word_idx += 1;
        }

        if add_special_tokens {
            self.add_special_tokens(&mut encoding);
        }

        Ok(encoding)
    }

    fn encode_pair(
        &self,
        text: &str,
        text_pair: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let mut encoding = self.encode(text, false)?;
        let encoding_pair = self.encode(text_pair, false)?;

        // Update sequence IDs for second sequence
        let mut pair_with_seq_id = Encoding::with_capacity(encoding_pair.len());
        for i in 0..encoding_pair.len() {
            pair_with_seq_id.push(
                encoding_pair.get_ids()[i],
                encoding_pair.get_tokens()[i].clone(),
                encoding_pair.get_offsets()[i],
                encoding_pair.get_word_ids()[i],
                Some(1),
                false,
            );
        }

        encoding.merge(pair_with_seq_id, true);

        if add_special_tokens {
            self.add_special_tokens(&mut encoding);
            if let Some(sep_id) = self.vocabulary.sep_token_id() {
                let last_offset = encoding.get_offsets().last().map(|(_, e)| *e).unwrap_or(0);
                encoding.push(
                    sep_id,
                    "[SEP]".to_string(),
                    (last_offset, last_offset),
                    None,
                    Some(1),
                    true,
                );
            }
        }

        Ok(encoding)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut tokens = Vec::with_capacity(ids.len());

        for &id in ids {
            if let Some(token) = self.vocabulary.id_to_token(id) {
                if skip_special_tokens && (token.starts_with('[') && token.ends_with(']')) {
                    continue;
                }

                let token = if token.starts_with(&self.config.continuing_subword_prefix) {
                    &token[self.config.continuing_subword_prefix.len()..]
                } else {
                    token
                };

                tokens.push(token.to_string());
            } else {
                tokens.push(self.config.unk_token.clone());
            }
        }

        let mut result = String::new();
        for (i, token) in tokens.iter().enumerate() {
            if i > 0 && !token.is_empty() {
                result.push(' ');
            }
            result.push_str(token);
        }

        Ok(result)
    }

    fn save(&self, _path: &std::path::Path) -> Result<()> {
        Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Saving not yet implemented",
        )))
    }
}

impl FastWordPieceTokenizer {
    fn add_special_tokens(&self, encoding: &mut Encoding) {
        let cls_id = self.vocabulary.cls_token_id();
        let sep_id = self.vocabulary.sep_token_id();

        if let (Some(cls), Some(sep)) = (cls_id, sep_id) {
            let mut new_encoding = Encoding::with_capacity(encoding.len() + 2);

            new_encoding.push(cls, "[CLS]".to_string(), (0, 0), None, Some(0), true);

            for i in 0..encoding.len() {
                new_encoding.push(
                    encoding.get_ids()[i],
                    encoding.get_tokens()[i].clone(),
                    encoding.get_offsets()[i],
                    encoding.get_word_ids()[i],
                    Some(0),
                    false,
                );
            }

            let last_offset = encoding.get_offsets().last().map(|(_, e)| *e).unwrap_or(0);
            new_encoding.push(
                sep,
                "[SEP]".to_string(),
                (last_offset, last_offset),
                None,
                Some(0),
                true,
            );

            *encoding = new_encoding;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::VocabularyBuilder;

    fn create_test_tokenizer() -> FastWordPieceTokenizer {
        let vocab = VocabularyBuilder::new()
            .add_tokens([
                "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "##ing", "##ed",
                "un", "play", "##play", "##er", "the", "a",
            ])
            .unk_token("[UNK]")
            .pad_token("[PAD]")
            .cls_token("[CLS]")
            .sep_token("[SEP]")
            .build();

        FastWordPieceTokenizer::new(vocab, FastWordPieceConfig::default())
    }

    #[test]
    fn test_fast_encode() {
        let tokenizer = create_test_tokenizer();
        let ids = tokenizer.encode_fast("hello world");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_fast_batch_encode() {
        let tokenizer = create_test_tokenizer();
        let texts = vec!["hello world", "the quick brown fox"];
        let results = tokenizer.encode_batch_fast(&texts);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_encode_with_special() {
        let tokenizer = create_test_tokenizer();
        let encoding = tokenizer.encode("hello", true).unwrap();
        assert_eq!(encoding.get_tokens()[0], "[CLS]");
        assert_eq!(encoding.get_tokens().last().unwrap(), "[SEP]");
    }

    #[test]
    fn test_parallel_batch() {
        let tokenizer = create_test_tokenizer();
        let texts: Vec<&str> = (0..100).map(|_| "hello world test").collect();
        let results = tokenizer.encode_batch_fast(&texts);
        assert_eq!(results.len(), 100);
    }
}
