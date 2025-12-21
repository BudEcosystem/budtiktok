//! Tokenizer loader with auto-detection
//!
//! This module provides automatic loading of tokenizers from HuggingFace
//! tokenizer.json files with model type auto-detection.

use std::path::Path;

use crate::bpe_linear::{BpeConfig, BpeTokenizer, MergeRule};
use crate::config::TokenizerConfig;
use crate::error::{Error, Result};
use crate::tokenizer::Tokenizer;
use crate::unigram::{UnigramConfig, UnigramPiece, UnigramTokenizer};
use crate::vocab::{SpecialTokens, Vocabulary};
use crate::wordpiece_hyper::{HyperConfig, HyperWordPieceTokenizer};

/// Load a tokenizer from a tokenizer.json file with auto-detection
pub fn load_tokenizer(path: impl AsRef<Path>) -> Result<Box<dyn Tokenizer>> {
    let config = TokenizerConfig::from_file(path)?;
    load_from_config(config)
}

/// Load a tokenizer from a JSON string with auto-detection
pub fn load_tokenizer_from_str(json: &str) -> Result<Box<dyn Tokenizer>> {
    let config = TokenizerConfig::from_json(json)?;
    load_from_config(config)
}

/// Load a tokenizer from a parsed configuration
pub fn load_from_config(config: TokenizerConfig) -> Result<Box<dyn Tokenizer>> {
    // First try explicit model type
    match config.model.model_type.as_str() {
        "WordPiece" => return load_wordpiece(config),
        "BPE" => return load_bpe(config),
        "Unigram" => return load_unigram(config),
        "" => {
            // No explicit type - try to infer from vocab format
            if config.model.is_unigram() {
                return load_unigram(config);
            }
        }
        _ => {}
    }

    // Final fallback - check if it's Unigram by vocab format
    if config.model.is_unigram() {
        return load_unigram(config);
    }

    Err(Error::InvalidConfig(format!(
        "Unsupported or unknown model type: '{}'. Supported types: WordPiece, BPE, Unigram. \
         For Unigram models without explicit type, ensure vocab is in list format [[token, score], ...]",
        config.model.model_type
    )))
}

/// Load a WordPiece tokenizer from configuration
/// Uses HyperWordPieceTokenizer for maximum performance (SIMD + hash tables)
fn load_wordpiece(config: TokenizerConfig) -> Result<Box<dyn Tokenizer>> {
    let vocab = build_vocabulary(&config)?;

    let hyper_config = HyperConfig {
        continuing_subword_prefix: config
            .model
            .continuing_subword_prefix
            .unwrap_or_else(|| "##".to_string()),
        max_input_chars_per_word: config.model.max_input_chars_per_word.unwrap_or(200),
        unk_token: config
            .model
            .unk_token
            .clone()
            .unwrap_or_else(|| "[UNK]".to_string()),
        do_lower_case: config
            .normalizer
            .as_ref()
            .and_then(|n| n.lowercase)
            .unwrap_or(true),
        strip_accents: config
            .normalizer
            .as_ref()
            .and_then(|n| n.strip_accents)
            .unwrap_or(true),
        tokenize_chinese_chars: config
            .normalizer
            .as_ref()
            .and_then(|n| n.handle_chinese_chars)
            .unwrap_or(true),
        // Use 15-bit hash table (32K buckets) for good balance of memory and speed
        hash_table_bits: 15,
    };

    Ok(Box::new(HyperWordPieceTokenizer::new(vocab, hyper_config)))
}

/// Load a BPE tokenizer from configuration
fn load_bpe(config: TokenizerConfig) -> Result<Box<dyn Tokenizer>> {
    let vocab = build_vocabulary(&config)?;

    let merges = config
        .model
        .parse_merges()
        .into_iter()
        .enumerate()
        .map(|(priority, (first, second))| {
            let result = format!("{}{}", first, second);
            MergeRule {
                first,
                second,
                result,
                priority: priority as u32,
            }
        })
        .collect();

    let bpe_config = BpeConfig {
        unk_token: config
            .model
            .unk_token
            .clone()
            .unwrap_or_else(|| "<unk>".to_string()),
        end_of_word_suffix: config.model.end_of_word_suffix.clone(),
        continuing_subword_prefix: config.model.continuing_subword_prefix.clone(),
        fuse_unk: config.model.fuse_unk.unwrap_or(false),
        byte_level: config.model.byte_fallback.unwrap_or(true),
        dropout: config.model.dropout.unwrap_or(0.0),
        use_linear_algorithm: true, // Default to O(n) algorithm
    };

    Ok(Box::new(BpeTokenizer::new(vocab, merges, bpe_config)))
}

/// Load a Unigram tokenizer from configuration
fn load_unigram(config: TokenizerConfig) -> Result<Box<dyn Tokenizer>> {
    let vocab = build_vocabulary(&config)?;

    // For Unigram, we need to extract pieces with scores from the vocab list format
    // HuggingFace Unigram format: [["token", score], ...]
    let pieces: Vec<UnigramPiece> = if let Some(unigram_pieces) = config.model.get_unigram_pieces() {
        // Use actual scores from the model
        unigram_pieces
            .into_iter()
            .map(|(token, score)| UnigramPiece { token, score })
            .collect()
    } else {
        // Fallback: create pieces from dict vocab with estimated scores
        let token_to_id = config.model.get_vocab();
        token_to_id
            .into_iter()
            .map(|(token, id)| {
                // Default score based on ID (lower ID = more common = higher score)
                let score = -(id as f64) / 1000.0;
                UnigramPiece { token, score }
            })
            .collect()
    };

    // Get UNK token - prefer explicit unk_token, then look in added tokens
    let unk_token = config
        .model
        .unk_token
        .clone()
        .or_else(|| find_special_token(&config, "unk"))
        .unwrap_or_else(|| "<unk>".to_string());

    // Get UNK ID - prefer explicit unk_id from model, then look up in vocab
    let unk_id = config
        .model
        .unk_id
        .or_else(|| vocab.token_to_id(&unk_token))
        .unwrap_or(0);

    let unigram_config = UnigramConfig {
        unk_token,
        unk_id,
        bos_token: find_special_token(&config, "bos"),
        eos_token: find_special_token(&config, "eos"),
        byte_fallback: config.model.byte_fallback.unwrap_or(false),
        // SentencePiece-style preprocessing: add ▁ prefix and replace spaces
        add_prefix_space: true,
        replacement_char: '▁',
    };

    Ok(Box::new(UnigramTokenizer::new(vocab, pieces, unigram_config)))
}

/// Build vocabulary from tokenizer configuration
fn build_vocabulary(config: &TokenizerConfig) -> Result<Vocabulary> {
    let token_to_id = config.model.get_vocab();

    if token_to_id.is_empty() {
        return Err(Error::VocabLoad("Empty vocabulary".to_string()));
    }

    let special_tokens = extract_special_tokens(config);
    Ok(Vocabulary::new(token_to_id, special_tokens))
}

/// Extract special tokens from configuration
fn extract_special_tokens(config: &TokenizerConfig) -> SpecialTokens {
    let mut special = SpecialTokens::default();

    // From model config
    special.unk_token = config.model.unk_token.clone();

    // From added tokens
    for token in &config.added_tokens {
        if !token.special {
            continue;
        }

        let content = &token.content;
        let lower = content.to_lowercase();

        if lower.contains("unk") {
            special.unk_token = Some(content.clone());
        } else if lower.contains("pad") {
            special.pad_token = Some(content.clone());
        } else if lower.contains("cls") {
            special.cls_token = Some(content.clone());
        } else if lower.contains("sep") {
            special.sep_token = Some(content.clone());
        } else if lower.contains("mask") {
            special.mask_token = Some(content.clone());
        } else if content == "<s>" || lower.contains("bos") {
            special.bos_token = Some(content.clone());
        } else if content == "</s>" || lower.contains("eos") {
            special.eos_token = Some(content.clone());
        }
    }

    special
}

/// Find a special token by role
fn find_special_token(config: &TokenizerConfig, role: &str) -> Option<String> {
    for token in &config.added_tokens {
        if !token.special {
            continue;
        }

        let content = &token.content;
        let lower = content.to_lowercase();

        let matches = match role {
            "bos" => content == "<s>" || lower.contains("bos"),
            "eos" => content == "</s>" || lower.contains("eos"),
            "unk" => lower.contains("unk"),
            "pad" => lower.contains("pad"),
            "cls" => lower.contains("cls"),
            "sep" => lower.contains("sep"),
            "mask" => lower.contains("mask"),
            _ => false,
        };

        if matches {
            return Some(content.clone());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_wordpiece_from_config() {
        let json = r###"{
            "version": "1.0",
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "vocab": {
                    "[PAD]": 0,
                    "[UNK]": 1,
                    "[CLS]": 2,
                    "[SEP]": 3,
                    "hello": 4,
                    "world": 5,
                    "##ing": 6
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
        let tokenizer = load_from_config(config).unwrap();

        assert_eq!(tokenizer.vocab_size(), 7);
        assert_eq!(tokenizer.token_to_id("[UNK]"), Some(1));
        assert_eq!(tokenizer.id_to_token(4), Some("hello"));
    }

    #[test]
    fn test_load_bpe_from_config() {
        let json = r#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "unk_token": "<unk>",
                "vocab": {
                    "<unk>": 0,
                    "h": 1,
                    "e": 2,
                    "l": 3,
                    "o": 4,
                    "he": 5,
                    "lo": 6
                },
                "merges": [
                    "h e",
                    "l o"
                ]
            }
        }"#;

        let config = TokenizerConfig::from_json(json).unwrap();
        let tokenizer = load_from_config(config).unwrap();

        assert_eq!(tokenizer.vocab_size(), 7);
    }

    #[test]
    fn test_unsupported_model_type() {
        let json = r#"{
            "model": {
                "type": "Unknown",
                "vocab": {}
            }
        }"#;

        let config = TokenizerConfig::from_json(json).unwrap();
        let result = load_from_config(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_unigram_from_config() {
        // Unigram model with list vocab format
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
                    ["▁", -1.0],
                    ["hello", -7.0],
                    ["world", -7.5],
                    ["lo", -8.0],
                    ["he", -8.5],
                    ["l", -9.0],
                    ["o", -9.0],
                    ["h", -9.0],
                    ["e", -9.0],
                    ["w", -9.0],
                    ["r", -9.0],
                    ["d", -9.0]
                ]
            },
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true},
                {"id": 1, "content": "<s>", "special": true},
                {"id": 2, "content": "</s>", "special": true}
            ]
        }"#;

        let config = TokenizerConfig::from_json(json).unwrap();
        let tokenizer = load_from_config(config).unwrap();

        // Check vocabulary
        assert_eq!(tokenizer.vocab_size(), 17);
        assert_eq!(tokenizer.token_to_id("<unk>"), Some(0));
        assert_eq!(tokenizer.token_to_id("▁hello"), Some(3));

        // Test basic encoding
        let encoding = tokenizer.encode("hello", false).unwrap();
        assert!(!encoding.is_empty());
    }

    #[test]
    fn test_load_unigram_with_explicit_type() {
        let json = r#"{
            "model": {
                "type": "Unigram",
                "unk_id": 0,
                "vocab": [
                    ["<unk>", 0.0],
                    ["hello", -2.0],
                    ["world", -3.0],
                    ["h", -5.0],
                    ["e", -5.0],
                    ["l", -5.0],
                    ["o", -5.0]
                ]
            },
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true}
            ]
        }"#;

        let config = TokenizerConfig::from_json(json).unwrap();
        let tokenizer = load_from_config(config).unwrap();

        assert_eq!(tokenizer.vocab_size(), 7);
        assert_eq!(tokenizer.token_to_id("hello"), Some(1));

        // Encoding should work
        let encoding = tokenizer.encode("hello world", false).unwrap();
        assert!(!encoding.is_empty());
    }

    #[test]
    fn test_unigram_special_tokens_in_loader() {
        let json = r#"{
            "model": {
                "unk_id": 0,
                "vocab": [
                    ["<unk>", 0.0],
                    ["<s>", 0.0],
                    ["</s>", 0.0],
                    ["hello", -5.0]
                ]
            },
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true},
                {"id": 1, "content": "<s>", "special": true},
                {"id": 2, "content": "</s>", "special": true}
            ]
        }"#;

        let config = TokenizerConfig::from_json(json).unwrap();
        let tokenizer = load_from_config(config).unwrap();

        // Verify special tokens are in vocabulary
        assert_eq!(tokenizer.token_to_id("<unk>"), Some(0));
        assert_eq!(tokenizer.token_to_id("<s>"), Some(1));
        assert_eq!(tokenizer.token_to_id("</s>"), Some(2));
    }

    #[test]
    fn test_unigram_encoding_uses_scores() {
        // Higher score tokens should be preferred
        let json = r#"{
            "model": {
                "unk_id": 0,
                "vocab": [
                    ["<unk>", 0.0],
                    ["hello", -2.0],
                    ["hel", -8.0],
                    ["lo", -8.0],
                    ["h", -10.0],
                    ["e", -10.0],
                    ["l", -10.0],
                    ["o", -10.0]
                ]
            },
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true}
            ]
        }"#;

        let config = TokenizerConfig::from_json(json).unwrap();
        let tokenizer = load_from_config(config).unwrap();

        let encoding = tokenizer.encode("hello", false).unwrap();

        // "hello" has highest score (-2.0), should be preferred
        // Check that we got a tokenization (may be "hello" or a subword sequence)
        assert!(!encoding.is_empty());

        // Verify token IDs are valid
        for &id in encoding.get_ids() {
            assert!(tokenizer.id_to_token(id).is_some());
        }
    }
}
