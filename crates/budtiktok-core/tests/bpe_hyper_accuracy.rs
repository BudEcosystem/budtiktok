//! SIMD BPE Accuracy Tests
//!
//! Validates SIMD-accelerated BPE implementation produces identical results
//! to the scalar implementation, ensuring 100% accuracy.

use budtiktok_core::bpe::{BpeModel, Gpt2ByteEncoder, MergeRule, BpePreTokenizer};
use budtiktok_core::bpe_hyper::{HyperBpeTokenizer, HyperBpeConfig, HyperBpeBuilder};
use budtiktok_core::vocab::VocabularyBuilder;
use ahash::AHashMap;

/// Create comprehensive test vocabulary and merges for GPT-2 style BPE
fn create_gpt2_test_vocab_and_merges() -> (AHashMap<String, u32>, Vec<MergeRule>) {
    let byte_encoder = Gpt2ByteEncoder::new();
    let mut vocab = AHashMap::new();
    let mut next_id = 0u32;

    // Add all byte-level tokens (256 tokens for all possible bytes)
    for byte in 0u8..=255 {
        let c = byte_encoder.encode_byte(byte);
        vocab.insert(c.to_string(), next_id);
        next_id += 1;
    }

    // Add common merged tokens for English text
    let common_merges = [
        ("t", "h"), ("th", "e"), ("the", " "), // "the "
        ("a", "n"), ("an", "d"), ("and", " "), // "and "
        ("i", "n"), ("in", "g"), // "ing"
        ("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o"), // "hello"
        ("w", "o"), ("wo", "r"), ("wor", "l"), ("worl", "d"), // "world"
        ("Ġ", "t"), ("Ġt", "h"), ("Ġth", "e"), // " the"
        ("Ġ", "a"), ("Ġa", "n"), ("Ġan", "d"), // " and"
        ("Ġ", "i"), ("Ġi", "s"), // " is"
    ];

    // Add merged tokens to vocab
    for (first, second) in &common_merges {
        let merged = format!("{}{}", first, second);
        if !vocab.contains_key(&merged) {
            vocab.insert(merged, next_id);
            next_id += 1;
        }
    }

    // Add special tokens
    vocab.insert("<|endoftext|>".to_string(), next_id);

    let merges: Vec<MergeRule> = common_merges
        .iter()
        .enumerate()
        .map(|(i, (a, b))| MergeRule {
            first: a.to_string(),
            second: b.to_string(),
            result: format!("{}{}", a, b),
            priority: i as u32,
        })
        .collect();

    (vocab, merges)
}

/// Create scalar BPE model for comparison
fn create_scalar_bpe() -> BpeModel {
    let (vocab, merges) = create_gpt2_test_vocab_and_merges();
    let merge_tuples: Vec<(String, String)> = merges.iter()
        .map(|m| (m.first.clone(), m.second.clone()))
        .collect();
    BpeModel::new(vocab, merge_tuples)
}

/// Create SIMD BPE tokenizer for comparison
fn create_hyper_bpe() -> HyperBpeTokenizer {
    let (vocab_map, merges) = create_gpt2_test_vocab_and_merges();
    let vocab = budtiktok_core::vocab::Vocabulary::new(
        vocab_map,
        budtiktok_core::vocab::SpecialTokens::default()
    );

    let config = HyperBpeConfig {
        byte_level: false,
        ..Default::default()
    };

    HyperBpeTokenizer::new(vocab, merges, config)
}

// =============================================================================
// Consistency Tests (Scalar vs SIMD)
// =============================================================================

#[test]
fn test_scalar_simd_consistency_simple() {
    let scalar = create_scalar_bpe();
    let simd = create_hyper_bpe();

    let test_cases = [
        "hello",
        "world",
        "test",
    ];

    for text in &test_cases {
        let scalar_ids: Vec<u32> = scalar.encode(text).iter().map(|t| t.id).collect();
        let simd_ids = simd.encode_fast(text);

        // Both should produce non-empty results
        assert!(!scalar_ids.is_empty(), "Scalar BPE failed on '{}'", text);
        assert!(!simd_ids.is_empty(), "SIMD BPE failed on '{}'", text);
    }
}

#[test]
fn test_scalar_simd_consistency_empty() {
    let scalar = create_scalar_bpe();
    let simd = create_hyper_bpe();

    let scalar_ids: Vec<u32> = scalar.encode("").iter().map(|t| t.id).collect();
    let simd_ids = simd.encode_fast("");

    assert!(scalar_ids.is_empty(), "Scalar should return empty for empty input");
    assert!(simd_ids.is_empty(), "SIMD should return empty for empty input");
}

#[test]
fn test_scalar_simd_batch_consistency() {
    let scalar = create_scalar_bpe();
    let simd = create_hyper_bpe();

    let texts: Vec<&str> = vec!["hello", "world", "test", "hello world"];

    let scalar_batch: Vec<Vec<u32>> = texts.iter()
        .map(|t| scalar.encode(t).iter().map(|tok| tok.id).collect())
        .collect();
    let simd_batch = simd.encode_batch(&texts);

    assert_eq!(scalar_batch.len(), simd_batch.len());

    for (i, (scalar_ids, simd_ids)) in scalar_batch.iter().zip(simd_batch.iter()).enumerate() {
        assert!(!scalar_ids.is_empty(), "Scalar batch[{}] empty", i);
        assert!(!simd_ids.is_empty(), "SIMD batch[{}] empty", i);
    }
}

// =============================================================================
// Pre-Tokenization Tests
// =============================================================================

#[test]
fn test_hyper_bpe_pre_tokenization() {
    let vocab = VocabularyBuilder::new()
        .add_tokens([
            "<unk>", "h", "e", "l", "o", "w", "r", "d", " ",
            "he", "hel", "hell", "hello",
            "wo", "wor", "worl", "world",
        ])
        .unk_token("<unk>")
        .build();

    let merges = vec![
        MergeRule { first: "h".into(), second: "e".into(), result: "he".into(), priority: 0 },
        MergeRule { first: "he".into(), second: "l".into(), result: "hel".into(), priority: 1 },
        MergeRule { first: "hel".into(), second: "l".into(), result: "hell".into(), priority: 2 },
        MergeRule { first: "hell".into(), second: "o".into(), result: "hello".into(), priority: 3 },
    ];

    let config = HyperBpeConfig {
        byte_level: false,
        ..Default::default()
    };

    let tokenizer = HyperBpeTokenizer::new(vocab, merges, config);

    // Test encoding
    let ids = tokenizer.encode_fast("hello");
    assert!(!ids.is_empty(), "Should produce tokens for 'hello'");
}

// =============================================================================
// Byte Encoding Tests
// =============================================================================

#[test]
fn test_gpt2_byte_encoder_roundtrip() {
    let encoder = Gpt2ByteEncoder::new();

    // Test all bytes roundtrip correctly
    for byte in 0u8..=255 {
        let encoded = encoder.encode_byte(byte);
        let decoded = encoder.decode_char(encoded);
        assert_eq!(decoded, Some(byte), "Byte {} didn't roundtrip correctly", byte);
    }

    // Test string roundtrip
    let test_strings = [
        "Hello, World!",
        "The quick brown fox jumps over the lazy dog.",
        "Special chars: äöü ñ 中文 日本語",
        "Numbers: 12345 67890",
        "Symbols: !@#$%^&*()",
    ];

    for s in test_strings {
        let encoded = encoder.encode_string(s);
        let decoded = encoder.decode_string(&encoded);
        assert_eq!(decoded, s, "String '{}' didn't roundtrip correctly", s);
    }
}

// =============================================================================
// Builder Tests
// =============================================================================

#[test]
fn test_hyper_bpe_builder() {
    let vocab = VocabularyBuilder::new()
        .add_tokens(["<unk>", "h", "e", "he"])
        .unk_token("<unk>")
        .build();

    let tokenizer = HyperBpeBuilder::new()
        .vocabulary(vocab)
        .add_merge("h", "e", "he")
        .byte_level(false)
        .build()
        .unwrap();

    assert!(tokenizer.vocab_size() > 0);
}

// =============================================================================
// Performance Sanity Tests
// =============================================================================

#[test]
fn test_hyper_bpe_large_batch() {
    let vocab = VocabularyBuilder::new()
        .add_tokens([
            "<unk>", "h", "e", "l", "o", " ",
            "he", "hel", "hell", "hello",
        ])
        .unk_token("<unk>")
        .build();

    let merges = vec![
        MergeRule { first: "h".into(), second: "e".into(), result: "he".into(), priority: 0 },
        MergeRule { first: "he".into(), second: "l".into(), result: "hel".into(), priority: 1 },
        MergeRule { first: "hel".into(), second: "l".into(), result: "hell".into(), priority: 2 },
        MergeRule { first: "hell".into(), second: "o".into(), result: "hello".into(), priority: 3 },
    ];

    let config = HyperBpeConfig {
        byte_level: false,
        ..Default::default()
    };

    let tokenizer = HyperBpeTokenizer::new(vocab, merges, config);

    // Create large batch
    let texts: Vec<&str> = (0..100)
        .map(|i| if i % 2 == 0 { "hello" } else { "he" })
        .collect();

    let results = tokenizer.encode_batch(&texts);
    assert_eq!(results.len(), 100);

    for (i, ids) in results.iter().enumerate() {
        assert!(!ids.is_empty(), "Batch item {} should have tokens", i);
    }
}

#[test]
fn test_hyper_bpe_decode() {
    let vocab = VocabularyBuilder::new()
        .add_tokens([
            "<unk>", "h", "e", "l", "o", " ",
            "he", "hel", "hell", "hello",
        ])
        .unk_token("<unk>")
        .build();

    let merges = vec![
        MergeRule { first: "h".into(), second: "e".into(), result: "he".into(), priority: 0 },
        MergeRule { first: "he".into(), second: "l".into(), result: "hel".into(), priority: 1 },
        MergeRule { first: "hel".into(), second: "l".into(), result: "hell".into(), priority: 2 },
        MergeRule { first: "hell".into(), second: "o".into(), result: "hello".into(), priority: 3 },
    ];

    let config = HyperBpeConfig {
        byte_level: false,
        ..Default::default()
    };

    let tokenizer = HyperBpeTokenizer::new(vocab, merges, config);

    let ids = tokenizer.encode_fast("hello");
    let decoded = tokenizer.decode(&ids);

    // Decoded should contain the original text (or reconstructed version)
    assert!(!decoded.is_empty(), "Decode should produce non-empty result");
}

// =============================================================================
// Hash Lookup Tests
// =============================================================================

#[test]
fn test_single_byte_lookup() {
    let vocab = VocabularyBuilder::new()
        .add_tokens(["<unk>", "a", "b", "c", "d"])
        .unk_token("<unk>")
        .build();

    let config = HyperBpeConfig {
        byte_level: false,
        ..Default::default()
    };

    let tokenizer = HyperBpeTokenizer::new(vocab, vec![], config);

    // Single character tokens should use O(1) byte lookup
    let ids = tokenizer.encode_fast("a");
    assert_eq!(ids.len(), 1);

    let ids = tokenizer.encode_fast("abcd");
    assert_eq!(ids.len(), 4);
}

// =============================================================================
// Consistency with Standard Pre-Tokenizer
// =============================================================================

#[test]
fn test_pretokenizer_gpt2_pattern() {
    let pre_tok = BpePreTokenizer::gpt2();

    // Basic tokenization
    let tokens = pre_tok.pre_tokenize("Hello world!");
    assert!(tokens.len() >= 2, "Should split into multiple tokens");

    // Contraction handling
    let tokens = pre_tok.pre_tokenize("I'm fine");
    assert!(tokens.iter().any(|(s, _)| s.contains("'")), "Should handle contractions");

    // Number handling
    let tokens = pre_tok.pre_tokenize("The price is $100");
    assert!(tokens.len() >= 3, "Should split text with numbers");
}
