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

    /// Parse tokenizer configuration from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| Error::VocabLoad(format!("Failed to parse tokenizer.json from bytes: {}", e)))
    }

    /// Serialize to JSON string
    ///
    /// # Arguments
    /// * `pretty` - If true, output will be pretty-printed with indentation
    pub fn to_string(&self, pretty: bool) -> Result<String> {
        if pretty {
            serde_json::to_string_pretty(self)
                .map_err(|e| Error::InvalidConfig(format!("Failed to serialize config: {}", e)))
        } else {
            serde_json::to_string(self)
                .map_err(|e| Error::InvalidConfig(format!("Failed to serialize config: {}", e)))
        }
    }

    /// Save configuration to a file
    ///
    /// # Arguments
    /// * `path` - Path to save the configuration to
    /// * `pretty` - If true, output will be pretty-printed with indentation
    pub fn save(&self, path: impl AsRef<Path>, pretty: bool) -> Result<()> {
        let json = self.to_string(pretty)?;
        std::fs::write(path.as_ref(), json)
            .map_err(|e| Error::VocabLoad(format!("Failed to write tokenizer.json: {}", e)))
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
    /// Detects both explicit type and vocab format
    pub fn is_unigram(&self) -> bool {
        self.model.is_unigram()
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

/// Vocabulary format - either dict (WordPiece/BPE) or list (Unigram)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum VocabFormat {
    /// WordPiece/BPE format: {"token": id, ...}
    Dict(AHashMap<String, u32>),
    /// Unigram format: [["token", score], ...]
    List(Vec<(String, f64)>),
}

impl Default for VocabFormat {
    fn default() -> Self {
        VocabFormat::Dict(AHashMap::new())
    }
}

impl VocabFormat {
    /// Get vocabulary as token -> id mapping
    /// For Unigram, converts list indices to IDs
    pub fn as_token_to_id(&self) -> AHashMap<String, u32> {
        match self {
            VocabFormat::Dict(map) => map.clone(),
            VocabFormat::List(list) => {
                list.iter()
                    .enumerate()
                    .map(|(idx, (token, _score))| (token.clone(), idx as u32))
                    .collect()
            }
        }
    }

    /// Get Unigram pieces with scores (returns None for Dict format)
    pub fn as_unigram_pieces(&self) -> Option<Vec<(String, f64)>> {
        match self {
            VocabFormat::Dict(_) => None,
            VocabFormat::List(list) => Some(list.clone()),
        }
    }

    /// Check if this is Unigram format
    pub fn is_unigram_format(&self) -> bool {
        matches!(self, VocabFormat::List(_))
    }

    /// Get the vocabulary size
    pub fn len(&self) -> usize {
        match self {
            VocabFormat::Dict(map) => map.len(),
            VocabFormat::List(list) => list.len(),
        }
    }

    /// Check if vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Model configuration (core tokenization algorithm)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model type: "WordPiece", "BPE", or "Unigram"
    /// Note: Some Unigram models may not have this field set
    #[serde(rename = "type", default)]
    pub model_type: String,

    /// Unknown token (string)
    #[serde(default)]
    pub unk_token: Option<String>,

    /// Unknown token ID (for Unigram models)
    #[serde(default)]
    pub unk_id: Option<u32>,

    /// Continuing subword prefix (for WordPiece: "##")
    #[serde(default)]
    pub continuing_subword_prefix: Option<String>,

    /// Max input characters per word
    #[serde(default)]
    pub max_input_chars_per_word: Option<usize>,

    /// Vocabulary: either dict format or list format
    #[serde(default)]
    pub vocab: VocabFormat,

    /// Merge rules (for BPE): list of (first, second) pairs
    #[serde(default)]
    pub merges: Option<Vec<String>>,

    /// End of word suffix (for some BPE models)
    #[serde(default)]
    pub end_of_word_suffix: Option<String>,

    /// Fuse unknown tokens
    #[serde(default)]
    pub fuse_unk: Option<bool>,

    /// Byte fallback (for BPE/Unigram)
    #[serde(default)]
    pub byte_fallback: Option<bool>,

    /// Dropout (for BPE training)
    #[serde(default)]
    pub dropout: Option<f32>,
}

impl ModelConfig {
    /// Get vocabulary as owned HashMap
    pub fn get_vocab(&self) -> AHashMap<String, u32> {
        self.vocab.as_token_to_id()
    }

    /// Get Unigram pieces with scores (token, log_prob)
    /// Returns None if not a Unigram model or vocab is in dict format
    pub fn get_unigram_pieces(&self) -> Option<Vec<(String, f64)>> {
        self.vocab.as_unigram_pieces()
    }

    /// Check if this appears to be a Unigram model
    /// (either by explicit type or by vocab format)
    pub fn is_unigram(&self) -> bool {
        self.model_type == "Unigram" || self.vocab.is_unigram_format()
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

    #[test]
    fn test_parse_unigram_config() {
        // Unigram models use a list format: [["token", score], ...]
        let json = r#"{
            "version": "1.0",
            "model": {
                "unk_id": 0,
                "vocab": [
                    ["<unk>", 0.0],
                    ["<s>", 0.0],
                    ["</s>", 0.0],
                    ["▁hello", -5.5],
                    ["▁world", -6.2],
                    ["lo", -8.0],
                    ["he", -8.5]
                ]
            },
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true},
                {"id": 1, "content": "<s>", "special": true},
                {"id": 2, "content": "</s>", "special": true}
            ]
        }"#;

        let config = TokenizerConfig::from_json(json).unwrap();

        // Should detect as Unigram based on vocab format
        assert!(config.is_unigram());
        assert!(!config.is_wordpiece());
        assert!(!config.is_bpe());

        // Vocabulary should be accessible
        let vocab = config.model.get_vocab();
        assert_eq!(vocab.len(), 7);
        assert_eq!(vocab.get("<unk>"), Some(&0));
        assert_eq!(vocab.get("▁hello"), Some(&3));
        assert_eq!(vocab.get("▁world"), Some(&4));

        // Unigram pieces with scores should be available
        let pieces = config.model.get_unigram_pieces().unwrap();
        assert_eq!(pieces.len(), 7);
        assert_eq!(pieces[0], ("<unk>".to_string(), 0.0));
        assert_eq!(pieces[3], ("▁hello".to_string(), -5.5));
        assert_eq!(pieces[4], ("▁world".to_string(), -6.2));

        // UNK ID should be parsed
        assert_eq!(config.model.unk_id, Some(0));
    }

    #[test]
    fn test_parse_unigram_with_explicit_type() {
        let json = r#"{
            "model": {
                "type": "Unigram",
                "unk_id": 0,
                "vocab": [
                    ["<unk>", 0.0],
                    ["hello", -2.0]
                ]
            }
        }"#;

        let config = TokenizerConfig::from_json(json).unwrap();
        assert!(config.is_unigram());
        assert_eq!(config.model.model_type, "Unigram");
    }

    #[test]
    fn test_vocab_format_detection() {
        // Dict format (WordPiece/BPE)
        let dict_format = VocabFormat::Dict(
            [("hello".to_string(), 0), ("world".to_string(), 1)]
                .into_iter()
                .collect(),
        );
        assert!(!dict_format.is_unigram_format());
        assert!(dict_format.as_unigram_pieces().is_none());

        // List format (Unigram)
        let list_format = VocabFormat::List(vec![
            ("hello".to_string(), -2.0),
            ("world".to_string(), -3.0),
        ]);
        assert!(list_format.is_unigram_format());
        let pieces = list_format.as_unigram_pieces().unwrap();
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0].1, -2.0);

        // Convert list to token_to_id
        let vocab = list_format.as_token_to_id();
        assert_eq!(vocab.get("hello"), Some(&0));
        assert_eq!(vocab.get("world"), Some(&1));
    }

    #[test]
    fn test_unigram_special_tokens() {
        let json = r#"{
            "model": {
                "unk_id": 0,
                "vocab": [
                    ["<unk>", 0.0],
                    ["<s>", 0.0],
                    ["</s>", 0.0],
                    ["<pad>", 0.0],
                    ["hello", -5.0]
                ]
            },
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true},
                {"id": 1, "content": "<s>", "special": true},
                {"id": 2, "content": "</s>", "special": true},
                {"id": 3, "content": "<pad>", "special": true}
            ]
        }"#;

        let config = TokenizerConfig::from_json(json).unwrap();

        // Check special token detection
        assert!(config.get_special_token("unk").is_some());
        assert_eq!(config.get_special_token("unk").unwrap().content, "<unk>");
        assert!(config.get_special_token("bos").is_some());
        assert_eq!(config.get_special_token("bos").unwrap().content, "<s>");
        assert!(config.get_special_token("eos").is_some());
        assert_eq!(config.get_special_token("eos").unwrap().content, "</s>");
        assert!(config.get_special_token("pad").is_some());
    }
}
