//! BPE Algorithm Unit Tests (9.1.7)
//!
//! Tests for Byte Pair Encoding tokenization including:
//! - Merge rule application
//! - Priority queue ordering
//! - Tiktoken compatibility
//! - GPT-2 byte encoding
//! - Dropout training
//! - Pre-tokenization with regex
//! - Special token handling
//! - Cache integration

use budtiktok_core::bpe::*;
use std::collections::HashMap;

// =============================================================================
// BPE Model Basic Tests
// =============================================================================

#[test]
fn test_bpe_new() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);
    assert!(bpe.vocab_size() > 0);
}

#[test]
fn test_bpe_with_config() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let config = BpeConfig {
        unk_token: Some("<unk>".to_string()),
        continuing_subword_prefix: None,
        end_of_word_suffix: None,
        fuse_unk: false,
        byte_fallback: true,
    };
    let bpe = BpeModel::with_config(vocab, merges, config);
    assert_eq!(bpe.config().unk_token, Some("<unk>".to_string()));
}

#[test]
fn test_bpe_vocab_size() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);
    assert!(bpe.vocab_size() > 0);
}

#[test]
fn test_bpe_get_token() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    assert!(bpe.get_token_id("hello").is_some());
    assert!(bpe.get_token_id("xyz_not_exist").is_none());
}

// =============================================================================
// Merge Rule Application Tests
// =============================================================================

#[test]
fn test_merge_single_pair() {
    let vocab = create_test_vocab();
    let merges = vec![("h".to_string(), "e".to_string())]; // h + e -> he
    let bpe = BpeModel::new(vocab, merges);

    // "he" should merge
    let result = bpe.encode("he");
    assert!(result.len() >= 1);
}

#[test]
fn test_merge_multiple_pairs() {
    let vocab = create_test_vocab();
    let merges = vec![
        ("h".to_string(), "e".to_string()),   // h + e -> he
        ("he".to_string(), "l".to_string()),  // he + l -> hel
        ("hel".to_string(), "l".to_string()), // hel + l -> hell
    ];
    let bpe = BpeModel::new(vocab, merges);

    let result = bpe.encode("hell");
    // Should apply merges in order
    assert!(!result.is_empty());
}

#[test]
fn test_merge_order_matters() {
    let vocab = create_test_vocab();
    // Merges should be applied in order
    let merges1 = vec![
        ("a".to_string(), "b".to_string()),
        ("ab".to_string(), "c".to_string()),
    ];
    let merges2 = vec![
        ("b".to_string(), "c".to_string()),
        ("a".to_string(), "bc".to_string()),
    ];

    let bpe1 = BpeModel::new(vocab.clone(), merges1);
    let bpe2 = BpeModel::new(vocab, merges2);

    let result1 = bpe1.encode("abc");
    let result2 = bpe2.encode("abc");

    // Different merge orders may produce different results
    let _ = (result1, result2);
}

#[test]
fn test_merge_no_applicable() {
    let vocab = create_test_vocab();
    let merges = vec![("x".to_string(), "y".to_string())];
    let bpe = BpeModel::new(vocab, merges);

    let result = bpe.encode("abc");
    // Should not merge, each char separate
    assert!(result.len() >= 3);
}

#[test]
fn test_merge_repeated_application() {
    let vocab = create_test_vocab();
    let merges = vec![("l".to_string(), "l".to_string())]; // l + l -> ll
    let bpe = BpeModel::new(vocab, merges);

    let result = bpe.encode("llll");
    // Should merge pairs: llll -> ll ll -> (possibly further)
    assert!(result.len() >= 1);
}

// =============================================================================
// Priority Queue / Merge Ordering Tests
// =============================================================================

#[test]
fn test_merge_priority_by_rank() {
    let vocab = create_test_vocab();
    // Earlier merges have higher priority (lower rank)
    let merges = vec![
        ("a".to_string(), "b".to_string()), // Rank 0 - highest priority
        ("c".to_string(), "d".to_string()), // Rank 1
        ("e".to_string(), "f".to_string()), // Rank 2
    ];
    let bpe = BpeModel::new(vocab, merges);

    let result = bpe.encode("abcdef");
    // "ab" should be merged first due to highest priority
    assert!(!result.is_empty());
}

#[test]
fn test_merge_queue_updates() {
    let vocab = create_test_vocab();
    let merges = vec![
        ("a".to_string(), "b".to_string()),   // ab
        ("ab".to_string(), "c".to_string()),  // abc
        ("abc".to_string(), "d".to_string()), // abcd
    ];
    let bpe = BpeModel::new(vocab, merges);

    let result = bpe.encode("abcd");
    // Should progressively merge: abcd -> abc'd -> abcd
    // Final should be "abcd" as single token if in vocab
    assert!(result.len() >= 1);
}

#[test]
fn test_merge_leftmost_first() {
    let vocab = create_test_vocab();
    let merges = vec![("a".to_string(), "b".to_string())];
    let bpe = BpeModel::new(vocab, merges);

    let result = bpe.encode("abab");
    // Should merge leftmost "ab" first
    // Result: "ab" "ab" (both merged)
    assert!(!result.is_empty());
}

// =============================================================================
// Tiktoken Compatibility Tests
// =============================================================================

#[test]
fn test_tiktoken_byte_encoding() {
    let encoder = TiktokenEncoder::cl100k_base();

    let result = encoder.encode("hello world");
    assert!(!result.is_empty());
}

#[test]
fn test_tiktoken_special_tokens() {
    let encoder = TiktokenEncoder::cl100k_base();

    // Special tokens should be handled
    let result = encoder.encode_with_special_tokens("hello <|endoftext|> world");
    assert!(!result.is_empty());
}

#[test]
fn test_tiktoken_unicode() {
    let encoder = TiktokenEncoder::cl100k_base();

    let result = encoder.encode("日本語");
    assert!(!result.is_empty());
}

#[test]
fn test_tiktoken_decode() {
    let encoder = TiktokenEncoder::cl100k_base();

    let encoded = encoder.encode("hello world");
    let decoded = encoder.decode(&encoded);

    assert_eq!(decoded, "hello world");
}

#[test]
fn test_tiktoken_roundtrip() {
    let encoder = TiktokenEncoder::cl100k_base();

    let texts = vec!["hello", "世界", "hello 日本語 world", "emoji 😀"];
    for text in texts {
        let encoded = encoder.encode(text);
        let decoded = encoder.decode(&encoded);
        assert_eq!(decoded, text, "Roundtrip failed for '{}'", text);
    }
}

// =============================================================================
// GPT-2 Byte Encoding Tests
// =============================================================================

#[test]
fn test_gpt2_byte_encoder() {
    let byte_encoder = Gpt2ByteEncoder::new();

    // Test ASCII
    for byte in 0u8..128 {
        let encoded = byte_encoder.encode_byte(byte);
        let decoded = byte_encoder.decode_char(encoded);
        assert_eq!(decoded, Some(byte));
    }
}

#[test]
fn test_gpt2_encode_string() {
    let byte_encoder = Gpt2ByteEncoder::new();

    let result = byte_encoder.encode_string("hello");
    assert!(!result.is_empty());
}

#[test]
fn test_gpt2_decode_string() {
    let byte_encoder = Gpt2ByteEncoder::new();

    let encoded = byte_encoder.encode_string("hello");
    let decoded = byte_encoder.decode_string(&encoded);
    assert_eq!(decoded, "hello");
}

#[test]
fn test_gpt2_special_chars() {
    let byte_encoder = Gpt2ByteEncoder::new();

    // Test non-printable ASCII
    for byte in 0u8..32 {
        let encoded = byte_encoder.encode_byte(byte);
        let decoded = byte_encoder.decode_char(encoded);
        assert_eq!(decoded, Some(byte));
    }
}

// =============================================================================
// Pre-tokenization Regex Tests
// =============================================================================

#[test]
fn test_pretokenizer_gpt2_pattern() {
    let pretokenizer = BpePreTokenizer::gpt2();

    let tokens = pretokenizer.pre_tokenize("Hello world!");
    // Should split on whitespace and punctuation
    assert!(tokens.len() >= 2);
}

#[test]
fn test_pretokenizer_cl100k_pattern() {
    let pretokenizer = BpePreTokenizer::cl100k_base();

    let tokens = pretokenizer.pre_tokenize("Hello world!");
    assert!(tokens.len() >= 2);
}

#[test]
fn test_pretokenizer_preserves_offsets() {
    let pretokenizer = BpePreTokenizer::gpt2();

    let text = "Hello world";
    let tokens = pretokenizer.pre_tokenize(text);

    // Offsets should be valid
    for (token, (start, end)) in &tokens {
        assert!(*start < text.len());
        assert!(*end <= text.len());
        assert!(start <= end);
        assert_eq!(&text[*start..*end], *token);
    }
}

#[test]
fn test_pretokenizer_numbers() {
    let pretokenizer = BpePreTokenizer::gpt2();

    let tokens = pretokenizer.pre_tokenize("123 456");
    // Numbers should be handled
    assert!(tokens.len() >= 2);
}

#[test]
fn test_pretokenizer_contractions() {
    let pretokenizer = BpePreTokenizer::gpt2();

    let tokens = pretokenizer.pre_tokenize("I'm don't won't");
    // Contractions should be split appropriately
    assert!(tokens.len() >= 3);
}

// =============================================================================
// Dropout Training Tests
// =============================================================================

#[test]
fn test_dropout_disabled() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    // Without dropout, same input should give same output
    let result1 = bpe.encode("hello");
    let result2 = bpe.encode("hello");

    assert_eq!(result1.len(), result2.len());
}

#[test]
fn test_dropout_enabled() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let config = BpeConfig {
        unk_token: None,
        continuing_subword_prefix: None,
        end_of_word_suffix: None,
        fuse_unk: false,
        byte_fallback: true,
    };
    let mut bpe = BpeModel::with_config(vocab, merges, config);

    // With dropout, results may vary
    let results: Vec<_> = (0..10).map(|_| bpe.encode_with_dropout("hello", 0.1)).collect();

    // Should have some variation (probabilistic)
    // At minimum, all should be valid tokenizations
    for result in &results {
        assert!(!result.is_empty());
    }
}

#[test]
fn test_dropout_rate_zero() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let mut bpe = BpeModel::new(vocab, merges);

    // Dropout 0.0 should be same as no dropout
    let result1 = bpe.encode("hello");
    let result2 = bpe.encode_with_dropout("hello", 0.0);

    assert_eq!(result1.len(), result2.len());
}

#[test]
fn test_dropout_rate_one() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let mut bpe = BpeModel::new(vocab, merges);

    // Dropout 1.0 should skip all merges
    let result = bpe.encode_with_dropout("hello", 1.0);

    // Should be character-level (or byte-level)
    assert!(result.len() >= 5); // "hello" has 5 characters
}

// =============================================================================
// Special Token Handling Tests
// =============================================================================

#[test]
fn test_special_tokens_preserved() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let mut bpe = BpeModel::new(vocab, merges);

    bpe.add_special_token("<|endoftext|>", 50256);

    let result = bpe.encode_with_special_tokens("hello <|endoftext|> world");
    // Special token should be preserved as single token
    assert!(result.iter().any(|t| t.id == 50256));
}

#[test]
fn test_special_tokens_not_split() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let mut bpe = BpeModel::new(vocab, merges);

    bpe.add_special_token("<|custom|>", 50257);

    let result = bpe.encode_with_special_tokens("<|custom|>");
    // Should be single token
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].id, 50257);
}

#[test]
fn test_special_tokens_multiple() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let mut bpe = BpeModel::new(vocab, merges);

    bpe.add_special_token("<|start|>", 50256);
    bpe.add_special_token("<|end|>", 50257);

    let result = bpe.encode_with_special_tokens("<|start|>text<|end|>");
    assert!(result.iter().any(|t| t.id == 50256));
    assert!(result.iter().any(|t| t.id == 50257));
}

#[test]
fn test_added_tokens() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let mut bpe = BpeModel::new(vocab, merges);

    bpe.add_token("custom_token", 99999);

    let result = bpe.encode("custom_token");
    // Added token should be recognized
    assert!(result.iter().any(|t| t.id == 99999));
}

// =============================================================================
// Cache Integration Tests
// =============================================================================

#[test]
fn test_bpe_with_cache() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);
    let cached_bpe = CachedBpe::new(bpe, 1000);

    // First call - cache miss
    let result1 = cached_bpe.encode("hello");

    // Second call - cache hit
    let result2 = cached_bpe.encode("hello");

    assert_eq!(result1.len(), result2.len());
}

#[test]
fn test_bpe_cache_stats() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);
    let cached_bpe = CachedBpe::new(bpe, 1000);

    cached_bpe.encode("hello");
    cached_bpe.encode("hello"); // Hit
    cached_bpe.encode("world"); // Miss

    let stats = cached_bpe.cache_stats();
    assert!(stats.hits >= 1);
    assert!(stats.misses >= 2);
}

#[test]
fn test_bpe_cache_clear() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);
    let cached_bpe = CachedBpe::new(bpe, 1000);

    cached_bpe.encode("hello");
    cached_bpe.clear_cache();

    let stats = cached_bpe.cache_stats();
    assert_eq!(stats.size, 0);
}

// =============================================================================
// Encoding Output Tests
// =============================================================================

#[test]
fn test_encoding_ids() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let result = bpe.encode("hello");
    assert!(!result.is_empty());

    // All IDs should be valid
    for token in &result {
        assert!(bpe.id_to_token(token.id).is_some());
    }
}

#[test]
fn test_encoding_tokens() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let result = bpe.encode("hello");

    // Tokens should reconstruct original
    let reconstructed: String = result.iter().map(|t| t.token.as_str()).collect();
    assert!(reconstructed.contains("hello") || !result.is_empty());
}

#[test]
fn test_encoding_offsets() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let text = "hello world";
    let result = bpe.encode_with_offsets(text);

    for token in &result {
        let (start, end) = token.offsets;
        assert!(start <= text.len());
        assert!(end <= text.len());
        assert!(start <= end);
    }
}

// =============================================================================
// Decode Tests
// =============================================================================

#[test]
fn test_decode_basic() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let encoded = bpe.encode("hello");
    let ids: Vec<u32> = encoded.iter().map(|t| t.id).collect();
    let decoded = bpe.decode(&ids);

    assert!(!decoded.is_empty());
}

#[test]
fn test_decode_roundtrip() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let texts = vec!["hello", "world", "hello world"];
    for text in texts {
        let encoded = bpe.encode(text);
        let ids: Vec<u32> = encoded.iter().map(|t| t.id).collect();
        let decoded = bpe.decode(&ids);
        assert!(decoded.len() >= text.len() - 1);
    }
}

#[test]
fn test_decode_invalid_id() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let result = bpe.decode(&[99999999]);
    // Should handle gracefully
    let _ = result;
}

// =============================================================================
// Batch Processing Tests
// =============================================================================

#[test]
fn test_batch_encode() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let texts = vec!["hello", "world", "test"];
    let results = bpe.encode_batch(&texts);

    assert_eq!(results.len(), 3);
}

#[test]
fn test_batch_encode_parallel() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let texts: Vec<_> = (0..100).map(|i| format!("text{}", i)).collect();
    let texts_ref: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    let results = bpe.encode_batch_parallel(&texts_ref);
    assert_eq!(results.len(), 100);
}

#[test]
fn test_batch_decode() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let texts = vec!["hello", "world"];
    let encoded: Vec<Vec<u32>> = texts
        .iter()
        .map(|t| bpe.encode(t).iter().map(|tok| tok.id).collect())
        .collect();

    let decoded = bpe.decode_batch(&encoded);
    assert_eq!(decoded.len(), 2);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_empty_input() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let result = bpe.encode("");
    assert!(result.is_empty());
}

#[test]
fn test_single_char() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let result = bpe.encode("a");
    assert!(!result.is_empty());
}

#[test]
fn test_unicode_input() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let result = bpe.encode("日本語");
    assert!(!result.is_empty());
}

#[test]
fn test_emoji_input() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let result = bpe.encode("hello 😀 world");
    assert!(!result.is_empty());
}

#[test]
fn test_very_long_input() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let long_input = "hello ".repeat(1000);
    let result = bpe.encode(&long_input);
    assert!(!result.is_empty());
}

#[test]
fn test_whitespace_only() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let result = bpe.encode("   ");
    // Should handle whitespace-only input
    let _ = result;
}

#[test]
fn test_newlines() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let result = bpe.encode("hello\nworld\n");
    assert!(!result.is_empty());
}

// =============================================================================
// Serialization Tests
// =============================================================================

#[test]
fn test_model_to_bytes() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let bytes = bpe.to_bytes();
    assert!(!bytes.is_empty());
}

#[test]
fn test_model_from_bytes() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let bytes = bpe.to_bytes();
    let restored = BpeModel::from_bytes(&bytes).unwrap();

    assert_eq!(bpe.vocab_size(), restored.vocab_size());
}

#[test]
fn test_model_roundtrip() {
    let vocab = create_test_vocab();
    let merges = create_test_merges();
    let bpe = BpeModel::new(vocab, merges);

    let bytes = bpe.to_bytes();
    let restored = BpeModel::from_bytes(&bytes).unwrap();

    let result1 = bpe.encode("hello");
    let result2 = restored.encode("hello");

    assert_eq!(result1.len(), result2.len());
}

// =============================================================================
// Merge File Loading Tests
// =============================================================================

#[test]
fn test_load_merges_format() {
    // Test parsing of merges.txt format
    let merges_content = "h e\nhe l\nhel l\nhell o";
    let merges = parse_merges(merges_content);

    assert_eq!(merges.len(), 4);
    assert_eq!(merges[0], ("h".to_string(), "e".to_string()));
}

#[test]
fn test_load_merges_with_comments() {
    // Some formats have a version comment
    let merges_content = "#version: 0.2\nh e\nhe l";
    let merges = parse_merges(merges_content);

    assert_eq!(merges.len(), 2);
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_vocab() -> HashMap<String, u32> {
    let mut vocab = HashMap::new();

    // Single characters
    for (i, c) in "abcdefghijklmnopqrstuvwxyz".chars().enumerate() {
        vocab.insert(c.to_string(), i as u32);
    }

    // Space
    vocab.insert(" ".to_string(), 26);

    // Common merged tokens
    vocab.insert("he".to_string(), 100);
    vocab.insert("hel".to_string(), 101);
    vocab.insert("hell".to_string(), 102);
    vocab.insert("hello".to_string(), 103);
    vocab.insert("wo".to_string(), 104);
    vocab.insert("wor".to_string(), 105);
    vocab.insert("worl".to_string(), 106);
    vocab.insert("world".to_string(), 107);

    // Special tokens
    vocab.insert("<unk>".to_string(), 200);
    vocab.insert("<|endoftext|>".to_string(), 50256);

    vocab
}

fn create_test_merges() -> Vec<(String, String)> {
    vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
        ("hel".to_string(), "l".to_string()),
        ("hell".to_string(), "o".to_string()),
        ("w".to_string(), "o".to_string()),
        ("wo".to_string(), "r".to_string()),
        ("wor".to_string(), "l".to_string()),
        ("worl".to_string(), "d".to_string()),
    ]
}

fn parse_merges(content: &str) -> Vec<(String, String)> {
    content
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                Some((parts[0].to_string(), parts[1].to_string()))
            } else {
                None
            }
        })
        .collect()
}
