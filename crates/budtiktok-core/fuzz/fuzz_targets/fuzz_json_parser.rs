//! Fuzzing Target: tokenizer.json Parser (9.5.2)
//!
//! Fuzzes the JSON parser to ensure no panics on malformed input.
//! Tests the tokenizer configuration loading.
//!
//! Run with: cargo +nightly fuzz run fuzz_json_parser

#![no_main]

use libfuzzer_sys::fuzz_target;
use budtiktok_core::vocab::{Vocabulary, SpecialTokens};

fuzz_target!(|data: &[u8]| {
    // Only test valid UTF-8 (JSON is always UTF-8)
    if let Ok(json_str) = std::str::from_utf8(data) {
        fuzz_json_config(json_str);
        fuzz_vocabulary_json(json_str);
    }
});

/// Fuzz tokenizer configuration parsing
fn fuzz_json_config(json_str: &str) {
    // Try to parse as a tokenizer config
    // Even if parsing fails, it should not panic

    // Try parsing as a vocabulary JSON (word -> id mapping)
    // This calls Vocabulary::from_json internally
    let _ = Vocabulary::from_json(json_str);

    // Try parsing as special tokens config
    if let Ok(parsed) = serde_json::from_str::<SpecialTokensConfig>(json_str) {
        // Successfully parsed special tokens
        let _ = parsed.unk_token;
        let _ = parsed.pad_token;
    }

    // Try parsing as a generic JSON value
    if let Ok(_value) = serde_json::from_str::<serde_json::Value>(json_str) {
        // Successfully parsed as JSON, but may not be valid config
    }
}

/// Fuzz vocabulary JSON parsing
fn fuzz_vocabulary_json(json_str: &str) {
    // Try different vocabulary formats

    // Format 1: Simple word -> id mapping using from_json
    if let Ok(vocab) = Vocabulary::from_json(json_str) {
        // Test vocabulary operations
        let _ = vocab.len();
        let _ = vocab.is_empty();

        // Try some lookups
        let _ = vocab.token_to_id("[PAD]");
        let _ = vocab.token_to_id("[UNK]");
        let _ = vocab.id_to_token(0);
        let _ = vocab.id_to_token(1);
    }

    // Format 2: Array of tokens (try to construct vocabulary)
    if let Ok(tokens) = serde_json::from_str::<Vec<String>>(json_str) {
        if tokens.len() < 100_000 {
            // Build a map from the array
            let vocab_map: ahash::AHashMap<String, u32> = tokens
                .into_iter()
                .enumerate()
                .map(|(i, t)| (t, i as u32))
                .collect();
            let vocab = Vocabulary::new(vocab_map, SpecialTokens::default());
            let _ = vocab.len();
        }
    }

    // Format 3: BPE merges format (array of string pairs)
    if let Ok(merges) = serde_json::from_str::<Vec<String>>(json_str) {
        // Try to parse merge rules
        let mut parsed_merges = Vec::new();
        for merge_str in merges.iter().take(1000) {
            if let Some((a, b)) = merge_str.split_once(' ') {
                parsed_merges.push((a.to_string(), b.to_string()));
            }
        }
        // Merges parsed, could be used with BPE tokenizer
    }
}

/// Configuration structure for special tokens
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct SpecialTokensConfig {
    pad_token: Option<String>,
    unk_token: Option<String>,
    cls_token: Option<String>,
    sep_token: Option<String>,
    mask_token: Option<String>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}
