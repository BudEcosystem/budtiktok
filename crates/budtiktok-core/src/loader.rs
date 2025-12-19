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
use crate::wordpiece::{WordPieceConfig, WordPieceTokenizer};

/// Load a tokenizer from a tokenizer.json file with auto-detection
pub fn load_tokenizer(path: impl AsRef<Path>) -> Result<Box<dyn Tokenizer>> {
    let config = TokenizerConfig::from_file(path)?;
    load_from_config(config)
}

/// Load a tokenizer from a parsed configuration
pub fn load_from_config(config: TokenizerConfig) -> Result<Box<dyn Tokenizer>> {
    match config.model.model_type.as_str() {
        "WordPiece" => load_wordpiece(config),
        "BPE" => load_bpe(config),
        "Unigram" => load_unigram(config),
        other => Err(Error::InvalidConfig(format!(
            "Unsupported model type: {}. Supported types: WordPiece, BPE, Unigram",
            other
        ))),
    }
}

/// Load a WordPiece tokenizer from configuration
fn load_wordpiece(config: TokenizerConfig) -> Result<Box<dyn Tokenizer>> {
    let vocab = build_vocabulary(&config)?;

    let wp_config = WordPieceConfig {
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
    };

    Ok(Box::new(WordPieceTokenizer::new(vocab, wp_config)))
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

    // For Unigram, we need to extract pieces with scores
    // The vocabulary contains token->id, but we need scores
    // In a real implementation, these would come from the model
    let pieces: Vec<UnigramPiece> = config
        .model
        .vocab
        .iter()
        .map(|(token, &id)| {
            // Default score based on ID (lower ID = more common = higher score)
            let score = -(id as f64) / 1000.0;
            UnigramPiece {
                token: token.clone(),
                score,
            }
        })
        .collect();

    let unk_token = config
        .model
        .unk_token
        .clone()
        .unwrap_or_else(|| "<unk>".to_string());

    let unk_id = config.model.vocab.get(&unk_token).copied().unwrap_or(0);

    let unigram_config = UnigramConfig {
        unk_token,
        unk_id,
        bos_token: find_special_token(&config, "bos"),
        eos_token: find_special_token(&config, "eos"),
        byte_fallback: config.model.byte_fallback.unwrap_or(false),
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
}
