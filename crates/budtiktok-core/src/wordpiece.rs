//! WordPiece tokenization algorithm
//!
//! Implementation of the WordPiece algorithm used by BERT and related models.
//! Uses a greedy longest-match-first strategy with the continuation prefix "##".

use std::path::Path;

use crate::cache::{ClockCache, TokenCache};
use crate::encoding::Encoding;
use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;
use crate::unicode::{is_cjk_character, is_punctuation, is_whitespace, normalize, normalize_bert_text, NormalizationForm, get_category_flags, CategoryFlags};
use crate::vocab::Vocabulary;

/// WordPiece tokenizer configuration
#[derive(Debug, Clone)]
pub struct WordPieceConfig {
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
    /// Whether to tokenize CJK characters
    pub tokenize_chinese_chars: bool,
}

impl Default for WordPieceConfig {
    fn default() -> Self {
        Self {
            continuing_subword_prefix: "##".to_string(),
            // Match HuggingFace's default (100, not 200)
            max_input_chars_per_word: 100,
            unk_token: "[UNK]".to_string(),
            do_lower_case: true,
            strip_accents: true,
            tokenize_chinese_chars: true,
        }
    }
}

/// WordPiece tokenizer implementation
pub struct WordPieceTokenizer {
    vocabulary: Vocabulary,
    config: WordPieceConfig,
    cache: TokenCache,
}

impl WordPieceTokenizer {
    /// Create a new WordPiece tokenizer with vocabulary and config
    pub fn new(vocabulary: Vocabulary, config: WordPieceConfig) -> Self {
        Self {
            vocabulary,
            config,
            cache: ClockCache::new(100_000),
        }
    }

    /// Create from a vocabulary file
    pub fn from_file(vocab_path: impl AsRef<Path>) -> Result<Self> {
        let vocabulary = Vocabulary::from_file(vocab_path)?;
        Ok(Self::new(vocabulary, WordPieceConfig::default()))
    }

    /// Create from a pretrained model name (loads from HuggingFace)
    pub fn from_pretrained(name: &str) -> Result<Self> {
        // TODO: Download from HuggingFace Hub
        Err(Error::vocab_load(format!(
            "Pretrained model loading not yet implemented: {}",
            name
        )))
    }

    /// Normalize text according to configuration
    ///
    /// Matches HuggingFace's BERT BasicTokenizer normalization:
    /// 1. Clean text (remove control chars, normalize whitespace)
    /// 2. Unicode NFC normalization
    /// 3. Lowercase (if enabled)
    /// 4. Accent stripping (if enabled)
    fn normalize(&self, text: &str) -> String {
        // Clean text first (control chars, no-break spaces, etc.)
        let mut result = normalize_bert_text(text);

        // Unicode NFC normalization
        result = normalize(&result, NormalizationForm::NFC);

        if self.config.do_lower_case {
            result = result.to_lowercase();
        }

        if self.config.strip_accents {
            result = crate::unicode::strip_accents(&result);
        }

        result
    }

    /// Check if a character should be treated as a word separator
    ///
    /// BERT's BasicTokenizer splits on:
    /// - All punctuation (Po, Pd, Ps, Pe, Pi, Pf, Pc)
    /// - ASCII math symbols only (+, =, <, >, etc.) - NOT Unicode ones like ×, ÷
    /// - ASCII modifier symbols only
    /// - Dollar sign ($) specifically (but NOT other currency symbols like £, €)
    ///
    /// This matches HuggingFace's BertPreTokenizer behavior exactly.
    #[inline]
    fn is_bert_separator(c: char) -> bool {
        let flags = get_category_flags(c);

        // Always split on punctuation
        if flags.is_punctuation() {
            return true;
        }

        // For symbols, only split on ASCII symbols
        // Unicode symbols like × (multiplication), ÷ (division) are kept within words
        if flags.is_symbol() && (c as u32) < 128 {
            // Dollar sign is the only currency symbol that splits
            if c == '$' {
                return true;
            }
            // Split on ASCII math symbols (+, =, <, >, etc.)
            if flags.contains(CategoryFlags::SYMBOL_MATH) {
                return true;
            }
            // Split on ASCII modifier symbols (^, etc.)
            if flags.contains(CategoryFlags::SYMBOL_MODIFIER) {
                return true;
            }
            // Split on other ASCII symbols
            return true;
        }

        false
    }

    /// Pre-tokenize text into words
    ///
    /// Matches HuggingFace's BertPreTokenizer behavior:
    /// - Splits on whitespace
    /// - Splits on punctuation (each punct becomes its own token)
    /// - Splits on symbols (including $, €, etc.)
    /// - Handles CJK characters as individual tokens
    fn pre_tokenize(&self, text: &str) -> Vec<(String, usize, usize)> {
        let mut words = Vec::new();
        let mut current_word = String::new();
        let mut word_start = 0;

        for (i, c) in text.char_indices() {
            let char_len = c.len_utf8();

            if is_whitespace(c) {
                if !current_word.is_empty() {
                    words.push((current_word.clone(), word_start, i));
                    current_word.clear();
                }
                word_start = i + char_len;
            } else if self.config.tokenize_chinese_chars && is_cjk_character(c) {
                // Each CJK character is its own word
                if !current_word.is_empty() {
                    words.push((current_word.clone(), word_start, i));
                    current_word.clear();
                }
                words.push((c.to_string(), i, i + char_len));
                word_start = i + char_len;
            } else if Self::is_bert_separator(c) {
                // Punctuation and symbols are their own words
                if !current_word.is_empty() {
                    words.push((current_word.clone(), word_start, i));
                    current_word.clear();
                }
                words.push((c.to_string(), i, i + char_len));
                word_start = i + char_len;
            } else {
                if current_word.is_empty() {
                    word_start = i;
                }
                current_word.push(c);
            }
        }

        if !current_word.is_empty() {
            words.push((current_word, word_start, text.len()));
        }

        words
    }

    /// Maximum word length for caching (3.1.6)
    /// Words longer than this are not cached to avoid memory pressure
    const MAX_CACHE_WORD_LEN: usize = 256;

    /// Tokenize a single word to token IDs using WordPiece algorithm with caching (3.1.6)
    ///
    /// Uses byte-length optimization (3.1.2): if byte_len <= max_chars,
    /// we can skip character counting because char_count <= byte_len in UTF-8.
    /// Only count characters when byte_len exceeds the limit.
    ///
    /// Caches results for words <= 256 bytes to avoid repeated tokenization.
    fn tokenize_word_to_ids(&self, word: &str) -> Vec<u32> {
        // Byte-length optimization: byte_len >= char_len in UTF-8
        // If byte_len <= max, we're safe. Only count chars if byte_len > max.
        if word.len() > self.config.max_input_chars_per_word
            && word.chars().count() > self.config.max_input_chars_per_word
        {
            return vec![self.vocabulary.token_to_id(&self.config.unk_token).unwrap_or(0)];
        }

        // Caching integration (3.1.6): check cache for cache-worthy strings
        let use_cache = word.len() <= Self::MAX_CACHE_WORD_LEN;

        if use_cache {
            if let Some(cached) = self.cache.get(&word.to_string()) {
                return cached;
            }
        }

        // Perform tokenization
        let mut ids = Vec::new();
        let mut start = 0;
        let unk_id = self.vocabulary.token_to_id(&self.config.unk_token).unwrap_or(0);

        while start < word.len() {
            let mut end = word.len();
            let mut found = false;

            while start < end {
                let substr = if start > 0 {
                    format!("{}{}", self.config.continuing_subword_prefix, &word[start..end])
                } else {
                    word[start..end].to_string()
                };

                if let Some(id) = self.vocabulary.token_to_id(&substr) {
                    ids.push(id);
                    found = true;
                    break;
                }

                // Move end back by one character (handle UTF-8)
                let mut new_end = end - 1;
                while new_end > start && !word.is_char_boundary(new_end) {
                    new_end -= 1;
                }
                end = new_end;
            }

            if !found {
                let result = vec![unk_id];
                if use_cache {
                    self.cache.insert(word.to_string(), result.clone());
                }
                return result;
            }

            start = end;
        }

        // Caching integration (3.1.6): cache results for cache-worthy strings
        if use_cache {
            self.cache.insert(word.to_string(), ids.clone());
        }

        ids
    }

    /// Tokenize a single word using WordPiece algorithm
    ///
    /// Returns token strings. For better performance with caching,
    /// use `tokenize_word_to_ids` which returns token IDs.
    fn tokenize_word(&self, word: &str) -> Vec<String> {
        self.tokenize_word_to_ids(word)
            .iter()
            .map(|&id| {
                self.vocabulary
                    .id_to_token(id)
                    .unwrap_or(&self.config.unk_token)
                    .to_string()
            })
            .collect()
    }

    /// Add special tokens to encoding
    fn add_special_tokens(&self, encoding: &mut Encoding) {
        let cls_id = self.vocabulary.cls_token_id();
        let sep_id = self.vocabulary.sep_token_id();

        if let (Some(cls), Some(sep)) = (cls_id, sep_id) {
            let mut new_encoding = Encoding::with_capacity(encoding.len() + 2);

            // Add [CLS]
            new_encoding.push(
                cls,
                "[CLS]".to_string(),
                (0, 0),
                None,
                Some(0),
                true,
            );

            // Add original tokens
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

            // Add [SEP]
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

impl Tokenizer for WordPieceTokenizer {
    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> {
        // Normalize text
        let normalized = self.normalize(text);

        // Pre-tokenize into words
        let words = self.pre_tokenize(&normalized);

        // Create encoding
        let mut encoding = Encoding::new();
        let mut word_idx = 0u32;

        for (word, start, end) in words {
            let tokens = self.tokenize_word(&word);

            for (i, token) in tokens.iter().enumerate() {
                let id = self.vocabulary.token_to_id(token)
                    .unwrap_or_else(|| self.vocabulary.unk_token_id().unwrap_or(0));

                let token_start = if i == 0 { start } else { start };
                let token_end = if i == tokens.len() - 1 { end } else { end };

                encoding.push(
                    id,
                    token.clone(),
                    (token_start, token_end),
                    Some(word_idx),
                    Some(0),
                    false,
                );
            }

            word_idx += 1;
        }

        // Add special tokens if requested
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
                Some(1), // Second sequence
                false,
            );
        }

        encoding.merge(pair_with_seq_id, true);

        if add_special_tokens {
            self.add_special_tokens(&mut encoding);
            // Add another [SEP] for the pair
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
                // Skip special tokens if requested
                if skip_special_tokens && (token.starts_with('[') && token.ends_with(']')) {
                    continue;
                }

                // Remove continuation prefix
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

        // Join tokens, handling continuation prefix
        let mut result = String::new();
        for (i, token) in tokens.iter().enumerate() {
            if i > 0 && !token.is_empty() {
                // Add space unless it's a continuation
                result.push(' ');
            }
            result.push_str(token);
        }

        Ok(result)
    }

    /// Parallel batch encoding for better throughput
    fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        use rayon::prelude::*;

        if texts.is_empty() {
            return Ok(Vec::new());
        }

        if texts.len() == 1 {
            return Ok(vec![self.encode(texts[0], add_special_tokens)?]);
        }

        texts
            .par_iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }

    fn save(&self, _path: &Path) -> Result<()> {
        // TODO: Implement saving
        Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Saving not yet implemented",
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::VocabularyBuilder;

    fn create_test_tokenizer() -> WordPieceTokenizer {
        let vocab = VocabularyBuilder::new()
            .add_tokens([
                "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                "hello", "world", "##ing", "##ed", "un",
                "play", "##play", "##er", "the", "a",
            ])
            .unk_token("[UNK]")
            .pad_token("[PAD]")
            .cls_token("[CLS]")
            .sep_token("[SEP]")
            .build();

        WordPieceTokenizer::new(vocab, WordPieceConfig::default())
    }

    #[test]
    fn test_basic_tokenization() {
        let tokenizer = create_test_tokenizer();
        let encoding = tokenizer.encode("hello world", false).unwrap();

        assert!(!encoding.is_empty());
        // Should contain "hello" and "world" tokens
    }

    #[test]
    fn test_with_special_tokens() {
        let tokenizer = create_test_tokenizer();
        let encoding = tokenizer.encode("hello", true).unwrap();

        // Should start with [CLS] and end with [SEP]
        assert_eq!(encoding.get_tokens()[0], "[CLS]");
        assert_eq!(encoding.get_tokens().last().unwrap(), "[SEP]");
    }

    #[test]
    fn test_pre_tokenize() {
        let tokenizer = create_test_tokenizer();
        let words = tokenizer.pre_tokenize("hello world");

        assert_eq!(words.len(), 2);
        assert_eq!(words[0].0, "hello");
        assert_eq!(words[1].0, "world");
    }

    // ===== Byte-Length Optimization Tests (3.1.2) =====

    #[test]
    fn test_byte_length_optimization_multibyte() {
        // Create tokenizer with small max_chars limit
        // Need to include the full word to match WordPiece algorithm behavior
        let vocab = VocabularyBuilder::new()
            .add_tokens(["[UNK]", "日本語", "日", "本", "語"])
            .unk_token("[UNK]")
            .build();

        let config = WordPieceConfig {
            max_input_chars_per_word: 5, // 5 chars max
            ..Default::default()
        };

        let tokenizer = WordPieceTokenizer::new(vocab, config);

        // "日本語" is 3 chars but 9 bytes (3 bytes per CJK char)
        // byte_len (9) > max_chars (5), but char_count (3) <= max_chars (5)
        // Should NOT return UNK due to length
        let word = "日本語";
        assert_eq!(word.len(), 9); // 9 bytes
        assert_eq!(word.chars().count(), 3); // 3 chars

        let tokens = tokenizer.tokenize_word(word);
        // Should find the word in vocab, not return UNK
        assert!(!tokens.contains(&"[UNK]".to_string()));
        assert_eq!(tokens, vec!["日本語".to_string()]);
    }

    #[test]
    fn test_byte_length_optimization_too_many_chars() {
        // Create tokenizer with small max_chars limit
        let vocab = VocabularyBuilder::new()
            .add_tokens(["[UNK]", "a", "b", "c", "d", "e", "f"])
            .unk_token("[UNK]")
            .build();

        let config = WordPieceConfig {
            max_input_chars_per_word: 3, // 3 chars max
            ..Default::default()
        };

        let tokenizer = WordPieceTokenizer::new(vocab, config);

        // "abcdef" is 6 chars (also 6 bytes)
        // char_count (6) > max_chars (3)
        // Should return UNK
        let word = "abcdef";
        assert_eq!(word.len(), 6); // 6 bytes
        assert_eq!(word.chars().count(), 6); // 6 chars

        let tokens = tokenizer.tokenize_word(word);
        assert_eq!(tokens, vec!["[UNK]".to_string()]);
    }

    #[test]
    fn test_byte_length_optimization_ascii_under_limit() {
        // ASCII word under limit - should not trigger char count
        let vocab = VocabularyBuilder::new()
            .add_tokens(["[UNK]", "hello"])
            .unk_token("[UNK]")
            .build();

        let config = WordPieceConfig {
            max_input_chars_per_word: 10,
            ..Default::default()
        };

        let tokenizer = WordPieceTokenizer::new(vocab, config);

        // "hello" is 5 chars and 5 bytes
        // byte_len (5) <= max_chars (10)
        // Optimization: no char count needed
        let word = "hello";
        assert_eq!(word.len(), 5);

        let tokens = tokenizer.tokenize_word(word);
        assert_eq!(tokens, vec!["hello".to_string()]);
    }

    #[test]
    fn test_byte_length_optimization_emoji() {
        // Emoji test: high byte count, low char count
        let vocab = VocabularyBuilder::new()
            .add_tokens(["[UNK]"])
            .unk_token("[UNK]")
            .build();

        let config = WordPieceConfig {
            max_input_chars_per_word: 5,
            ..Default::default()
        };

        let tokenizer = WordPieceTokenizer::new(vocab, config);

        // "😀😀" is 2 chars but 8 bytes (4 bytes per emoji)
        // byte_len (8) > max_chars (5), but char_count (2) <= max_chars (5)
        // Should try to tokenize (will return UNK since emoji not in vocab)
        let word = "😀😀";
        assert_eq!(word.len(), 8);
        assert_eq!(word.chars().count(), 2);

        let tokens = tokenizer.tokenize_word(word);
        // Will be UNK because emoji isn't in vocab, not because of length
        assert_eq!(tokens, vec!["[UNK]".to_string()]);
    }

    #[test]
    fn test_byte_length_optimization_mixed_utf8() {
        // Mix of ASCII and multibyte
        let vocab = VocabularyBuilder::new()
            .add_tokens(["[UNK]", "café"])
            .unk_token("[UNK]")
            .build();

        let config = WordPieceConfig {
            max_input_chars_per_word: 10,
            ..Default::default()
        };

        let tokenizer = WordPieceTokenizer::new(vocab, config);

        // "café" is 4 chars but 5 bytes (é is 2 bytes)
        let word = "café";
        assert_eq!(word.len(), 5);
        assert_eq!(word.chars().count(), 4);

        let tokens = tokenizer.tokenize_word(word);
        assert_eq!(tokens, vec!["café".to_string()]);
    }

    // ===== Caching Integration Tests (3.1.6) =====

    #[test]
    fn test_caching_integration_basic() {
        let vocab = VocabularyBuilder::new()
            .add_tokens(["[UNK]", "hello", "world"])
            .unk_token("[UNK]")
            .build();

        let tokenizer = WordPieceTokenizer::new(vocab, WordPieceConfig::default());

        // First call - should tokenize and cache
        let result1 = tokenizer.tokenize_word_to_ids("hello");
        assert!(!result1.is_empty());

        // Second call - should hit cache
        let result2 = tokenizer.tokenize_word_to_ids("hello");
        assert_eq!(result1, result2);

        // Check cache stats
        let stats = tokenizer.cache.stats();
        assert!(stats.hits >= 1); // At least one cache hit
    }

    #[test]
    fn test_caching_integration_cache_worthy() {
        let vocab = VocabularyBuilder::new()
            .add_tokens(["[UNK]", "test"])
            .unk_token("[UNK]")
            .build();

        let tokenizer = WordPieceTokenizer::new(vocab, WordPieceConfig::default());

        // Short word should be cached
        let short_word = "test";
        assert!(short_word.len() <= WordPieceTokenizer::MAX_CACHE_WORD_LEN);

        tokenizer.tokenize_word_to_ids(short_word);
        tokenizer.tokenize_word_to_ids(short_word);

        let stats = tokenizer.cache.stats();
        assert!(stats.hits >= 1);
    }

    #[test]
    fn test_caching_integration_long_words_not_cached() {
        let vocab = VocabularyBuilder::new()
            .add_tokens(["[UNK]"])
            .unk_token("[UNK]")
            .build();

        let tokenizer = WordPieceTokenizer::new(vocab, WordPieceConfig::default());

        // Create a word longer than MAX_CACHE_WORD_LEN
        let long_word: String = "a".repeat(300);
        assert!(long_word.len() > WordPieceTokenizer::MAX_CACHE_WORD_LEN);

        tokenizer.tokenize_word_to_ids(&long_word);
        tokenizer.tokenize_word_to_ids(&long_word);

        // Long words should not be cached, so size should be 0
        let stats = tokenizer.cache.stats();
        assert_eq!(stats.size, 0);
    }

    #[test]
    fn test_caching_integration_different_words() {
        let vocab = VocabularyBuilder::new()
            .add_tokens(["[UNK]", "hello", "world", "foo", "bar"])
            .unk_token("[UNK]")
            .build();

        let tokenizer = WordPieceTokenizer::new(vocab, WordPieceConfig::default());

        // Tokenize multiple different words
        tokenizer.tokenize_word_to_ids("hello");
        tokenizer.tokenize_word_to_ids("world");
        tokenizer.tokenize_word_to_ids("foo");
        tokenizer.tokenize_word_to_ids("bar");

        // All should be in cache
        let stats = tokenizer.cache.stats();
        assert!(stats.size >= 4);

        // Access them again - should be cache hits
        tokenizer.tokenize_word_to_ids("hello");
        tokenizer.tokenize_word_to_ids("world");

        let stats_after = tokenizer.cache.stats();
        assert!(stats_after.hits >= 2);
    }

    #[test]
    fn test_caching_integration_unk_cached() {
        let vocab = VocabularyBuilder::new()
            .add_tokens(["[UNK]"])
            .unk_token("[UNK]")
            .build();

        let tokenizer = WordPieceTokenizer::new(vocab, WordPieceConfig::default());

        // Word not in vocab should return UNK and be cached
        let result1 = tokenizer.tokenize_word_to_ids("unknownword");
        assert!(!result1.is_empty());

        // Second call should hit cache
        let result2 = tokenizer.tokenize_word_to_ids("unknownword");
        assert_eq!(result1, result2);

        let stats = tokenizer.cache.stats();
        assert!(stats.hits >= 1);
    }

    #[test]
    fn test_caching_integration_subword_tokenization() {
        let vocab = VocabularyBuilder::new()
            .add_tokens(["[UNK]", "play", "##ing", "##ed"])
            .unk_token("[UNK]")
            .build();

        let tokenizer = WordPieceTokenizer::new(vocab, WordPieceConfig::default());

        // Word that requires subword tokenization
        let result1 = tokenizer.tokenize_word_to_ids("playing");
        assert_eq!(result1.len(), 2); // "play" + "##ing"

        // Should be cached
        let result2 = tokenizer.tokenize_word_to_ids("playing");
        assert_eq!(result1, result2);

        let stats = tokenizer.cache.stats();
        assert!(stats.hits >= 1);
    }

    #[test]
    fn test_caching_integration_cache_stats() {
        let vocab = VocabularyBuilder::new()
            .add_tokens(["[UNK]", "test"])
            .unk_token("[UNK]")
            .build();

        let tokenizer = WordPieceTokenizer::new(vocab, WordPieceConfig::default());

        // Clear any initial state
        tokenizer.cache.clear();

        // First call: miss + insertion
        tokenizer.tokenize_word_to_ids("test");
        let stats1 = tokenizer.cache.stats();
        assert_eq!(stats1.misses, 1);
        assert_eq!(stats1.size, 1);

        // Second call: hit
        tokenizer.tokenize_word_to_ids("test");
        let stats2 = tokenizer.cache.stats();
        assert_eq!(stats2.hits, 1);
        assert!(stats2.hit_rate > 0.0);
    }
}
