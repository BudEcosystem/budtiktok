//! HuggingFace tokenizer.json parser
//!
//! This module parses the HuggingFace tokenizer.json format and extracts
//! all configuration needed to instantiate a tokenizer.

use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{Error, Result};

/// Root structure of tokenizer.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Version of the tokenizer format
    #[serde(default)]
    pub version: String,

    /// Truncation configuration
    #[serde(default)]
    pub truncation: Option<TruncationConfig>,

    /// Padding configuration
    #[serde(default)]
    pub padding: Option<PaddingConfig>,

    /// Added tokens (special tokens and user-defined tokens)
    #[serde(default)]
    pub added_tokens: Vec<AddedToken>,

    /// Normalizer configuration
    #[serde(default)]
    pub normalizer: Option<NormalizerConfig>,

    /// Pre-tokenizer configuration
    #[serde(default)]
    pub pre_tokenizer: Option<PreTokenizerConfig>,

    /// Post-processor configuration
    #[serde(default)]
    pub post_processor: Option<PostProcessorConfig>,

    /// Decoder configuration
    #[serde(default)]
    pub decoder: Option<DecoderConfig>,

    /// Model configuration (WordPiece, BPE, or Unigram)
    pub model: ModelConfig,
}

impl TokenizerConfig {
    /// Load tokenizer configuration from a file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| Error::VocabLoad(format!("Failed to read tokenizer.json: {}", e)))?;
        Self::from_json(&content)
    }

    /// Parse tokenizer configuration from JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| Error::VocabLoad(format!("Failed to parse tokenizer.json: {}", e)))
    }

    /// Get the model type
    pub fn model_type(&self) -> &str {
        &self.model.model_type
    }

    /// Check if this is a WordPiece tokenizer
    pub fn is_wordpiece(&self) -> bool {
        self.model.model_type == "WordPiece"
    }

    /// Check if this is a BPE tokenizer
    pub fn is_bpe(&self) -> bool {
        self.model.model_type == "BPE"
    }

    /// Check if this is a Unigram tokenizer
    pub fn is_unigram(&self) -> bool {
        self.model.model_type == "Unigram"
    }

    /// Get special tokens by role
    pub fn get_special_token(&self, role: &str) -> Option<&AddedToken> {
        self.added_tokens.iter().find(|t| {
            if !t.special {
                return false;
            }
            let lower = t.content.to_lowercase();
            match role {
                "unk" => lower.contains("unk"),
                "pad" => lower.contains("pad"),
                "cls" => lower.contains("cls"),
                "sep" => lower.contains("sep"),
                "mask" => lower.contains("mask"),
                "bos" => lower.contains("bos") || t.content == "<s>",
                "eos" => lower.contains("eos") || t.content == "</s>",
                _ => false,
            }
        })
    }
}

/// Truncation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncationConfig {
    /// Maximum length
    #[serde(default = "default_max_length")]
    pub max_length: usize,

    /// Truncation strategy
    #[serde(default)]
    pub strategy: String,

    /// Stride for overflow tokens
    #[serde(default)]
    pub stride: usize,

    /// Direction (left or right)
    #[serde(default)]
    pub direction: String,
}

fn default_max_length() -> usize {
    512
}

/// Padding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaddingConfig {
    /// Padding strategy
    #[serde(default)]
    pub strategy: PaddingStrategy,

    /// Direction (left or right)
    #[serde(default)]
    pub direction: String,

    /// Pad to multiple of this value
    #[serde(default)]
    pub pad_to_multiple_of: Option<usize>,

    /// Pad token
    #[serde(default)]
    pub pad_token: Option<String>,

    /// Pad token type ID
    #[serde(default)]
    pub pad_type_id: u32,

    /// Pad ID
    #[serde(default)]
    pub pad_id: u32,
}

/// Padding strategy
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "PascalCase")]
pub enum PaddingStrategy {
    #[default]
    BatchLongest,
    Fixed(usize),
}

/// Added token configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddedToken {
    /// Token ID
    pub id: u32,

    /// Token content/string
    pub content: String,

    /// Whether this is a single word token
    #[serde(default)]
    pub single_word: bool,

    /// Whether to left-strip
    #[serde(default)]
    pub lstrip: bool,

    /// Whether to right-strip
    #[serde(default)]
    pub rstrip: bool,

    /// Whether this is normalized
    #[serde(default = "default_true")]
    pub normalized: bool,

    /// Whether this is a special token
    #[serde(default)]
    pub special: bool,
}

fn default_true() -> bool {
    true
}

/// Normalizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizerConfig {
    /// Normalizer type
    #[serde(rename = "type")]
    pub normalizer_type: String,

    /// Whether to clean text (remove control chars)
    #[serde(default)]
    pub clean_text: Option<bool>,

    /// Whether to handle Chinese characters
    #[serde(default)]
    pub handle_chinese_chars: Option<bool>,

    /// Whether to strip accents
    #[serde(default)]
    pub strip_accents: Option<bool>,

    /// Whether to lowercase
    #[serde(default)]
    pub lowercase: Option<bool>,

    /// Normalizers for Sequence type
    #[serde(default)]
    pub normalizers: Option<Vec<NormalizerConfig>>,

    /// Replacement character for Replace type
    #[serde(default)]
    pub pattern: Option<PatternConfig>,

    /// Replacement content
    #[serde(default)]
    pub content: Option<String>,
}

/// Pattern configuration for normalizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConfig {
    /// Regex pattern
    #[serde(rename = "Regex")]
    pub regex: Option<String>,

    /// String pattern
    #[serde(rename = "String")]
    pub string: Option<String>,
}

/// Pre-tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreTokenizerConfig {
    /// Pre-tokenizer type
    #[serde(rename = "type")]
    pub pretokenizer_type: String,

    /// Whether to add prefix space
    #[serde(default)]
    pub add_prefix_space: Option<bool>,

    /// Replacement character (for Metaspace)
    #[serde(default)]
    pub replacement: Option<char>,

    /// Whether to use regex for splitting
    #[serde(default)]
    pub use_regex: Option<bool>,

    /// Pre-tokenizers for Sequence type
    #[serde(default)]
    pub pretokenizers: Option<Vec<PreTokenizerConfig>>,

    /// Split pattern
    #[serde(default)]
    pub pattern: Option<PatternConfig>,

    /// Split behavior
    #[serde(default)]
    pub behavior: Option<String>,

    /// Whether to invert the pattern
    #[serde(default)]
    pub invert: Option<bool>,
}

/// Post-processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostProcessorConfig {
    /// Post-processor type
    #[serde(rename = "type")]
    pub processor_type: String,

    /// Single sequence template
    #[serde(default)]
    pub single: Option<Vec<TemplateItem>>,

    /// Pair sequence template
    #[serde(default)]
    pub pair: Option<Vec<TemplateItem>>,

    /// Special tokens mapping
    #[serde(default)]
    pub special_tokens: Option<AHashMap<String, SpecialTokenConfig>>,

    /// CLS token (for BERT-style)
    #[serde(default)]
    pub cls: Option<(String, u32)>,

    /// SEP token (for BERT-style)
    #[serde(default)]
    pub sep: Option<(String, u32)>,
}

/// Template item for post-processor
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TemplateItem {
    /// Sequence reference
    Sequence {
        #[serde(rename = "Sequence")]
        sequence: SequenceRef
    },
    /// Special token reference
    SpecialToken {
        #[serde(rename = "SpecialToken")]
        special_token: SpecialTokenRef
    },
}

/// Sequence reference in template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceRef {
    pub id: String,
    pub type_id: u32,
}

/// Special token reference in template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokenRef {
    pub id: String,
    pub type_id: u32,
}

/// Special token configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokenConfig {
    pub id: String,
    pub ids: Vec<u32>,
    pub tokens: Vec<String>,
}

/// Decoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderConfig {
    /// Decoder type
    #[serde(rename = "type")]
    pub decoder_type: String,

    /// Prefix for word continuation (WordPiece)
    #[serde(default)]
    pub prefix: Option<String>,

    /// Whether to cleanup whitespace
    #[serde(default)]
    pub cleanup: Option<bool>,

    /// Replacement character (for Metaspace)
    #[serde(default)]
    pub replacement: Option<char>,

    /// Whether to add prefix space
    #[serde(default)]
    pub add_prefix_space: Option<bool>,

    /// Decoders for Sequence type
    #[serde(default)]
    pub decoders: Option<Vec<DecoderConfig>>,
}

/// Model configuration (core tokenization algorithm)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model type: "WordPiece", "BPE", or "Unigram"
    #[serde(rename = "type")]
    pub model_type: String,

    /// Unknown token
    #[serde(default)]
    pub unk_token: Option<String>,

    /// Continuing subword prefix (for WordPiece: "##")
    #[serde(default)]
    pub continuing_subword_prefix: Option<String>,

    /// Max input characters per word
    #[serde(default)]
    pub max_input_chars_per_word: Option<usize>,

    /// Vocabulary: token -> id mapping
    #[serde(default)]
    pub vocab: AHashMap<String, u32>,

    /// Merge rules (for BPE): list of (first, second) pairs
    #[serde(default)]
    pub merges: Option<Vec<String>>,

    /// End of word suffix (for some BPE models)
    #[serde(default)]
    pub end_of_word_suffix: Option<String>,

    /// Fuse unknown tokens
    #[serde(default)]
    pub fuse_unk: Option<bool>,

    /// Byte fallback (for BPE)
    #[serde(default)]
    pub byte_fallback: Option<bool>,

    /// Dropout (for BPE training)
    #[serde(default)]
    pub dropout: Option<f32>,
}

impl ModelConfig {
    /// Get vocabulary as owned HashMap
    pub fn get_vocab(&self) -> AHashMap<String, u32> {
        self.vocab.clone()
    }

    /// Parse BPE merge rules from string format "first second"
    pub fn parse_merges(&self) -> Vec<(String, String)> {
        self.merges
            .as_ref()
            .map(|merges| {
                merges
                    .iter()
                    .filter_map(|line| {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            Some((parts[0].to_string(), parts[1].to_string()))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_wordpiece_config() {
        let json = r###"{
            "version": "1.0",
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
                "vocab": {
                    "[PAD]": 0,
                    "[UNK]": 1,
                    "[CLS]": 2,
                    "[SEP]": 3,
                    "hello": 4,
                    "world": 5
                }
            },
            "added_tokens": [
                {"id": 0, "content": "[PAD]", "special": true},
                {"id": 1, "content": "[UNK]", "special": true},
                {"id": 2, "content": "[CLS]", "special": true},
                {"id": 3, "content": "[SEP]", "special": true}
            ]
        }"###;

        let config = TokenizerConfig::from_json(json).unwrap();
        assert!(config.is_wordpiece());
        assert_eq!(config.model.vocab.len(), 6);
        assert_eq!(config.model.unk_token, Some("[UNK]".to_string()));
        assert_eq!(config.model.continuing_subword_prefix, Some("##".to_string()));
        assert_eq!(config.added_tokens.len(), 4);
    }

    #[test]
    fn test_parse_bpe_config() {
        let json = r#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "unk_token": "<unk>",
                "vocab": {
                    "<unk>": 0,
                    "hello": 1,
                    "world": 2,
                    "helloworld": 3
                },
                "merges": [
                    "h e",
                    "he llo",
                    "hello world"
                ]
            }
        }"#;

        let config = TokenizerConfig::from_json(json).unwrap();
        assert!(config.is_bpe());
        assert_eq!(config.model.vocab.len(), 4);

        let merges = config.model.parse_merges();
        assert_eq!(merges.len(), 3);
        assert_eq!(merges[0], ("h".to_string(), "e".to_string()));
    }

    #[test]
    fn test_parse_normalizer_sequence() {
        let json = r#"{
            "version": "1.0",
            "model": {
                "type": "WordPiece",
                "vocab": {}
            },
            "normalizer": {
                "type": "Sequence",
                "normalizers": [
                    {"type": "NFD"},
                    {"type": "Lowercase"},
                    {"type": "StripAccents"}
                ]
            }
        }"#;

        let config = TokenizerConfig::from_json(json).unwrap();
        let normalizer = config.normalizer.unwrap();
        assert_eq!(normalizer.normalizer_type, "Sequence");
        assert!(normalizer.normalizers.is_some());
        assert_eq!(normalizer.normalizers.unwrap().len(), 3);
    }
}
