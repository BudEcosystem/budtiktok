//! Comprehensive accuracy tests for Unigram tokenizer
#![allow(dead_code)]
//!
//! This test suite verifies 100% accuracy match between BudTikTok's UnigramFast
//! and HuggingFace Tokenizers' Unigram implementation.
//!
//! Run with: cargo test --test unigram_hf_accuracy --release

use budtiktok_core::unigram::UnigramPiece;
use budtiktok_core::unigram_fast::{UnigramFast, UnigramFastConfig};
use budtiktok_core::vocab::{Vocabulary, SpecialTokens};
use tokenizers::Tokenizer as HfTokenizer;
use ahash::AHashMap;

const XLNET_PATH: &str = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/xlnet/tokenizer.json";

// ============================================================================
// Test Fixtures
// ============================================================================

fn load_tokenizers() -> Option<(UnigramFast, HfTokenizer)> {
    if !std::path::Path::new(XLNET_PATH).exists() {
        return None;
    }

    let hf_tok = HfTokenizer::from_file(XLNET_PATH).ok()?;

    let json_content = std::fs::read_to_string(XLNET_PATH).ok()?;
    let json: serde_json::Value = serde_json::from_str(&json_content).ok()?;

    let model = json.get("model")?;
    let vocab_array = model.get("vocab")?.as_array()?;

    let mut vocab = Vec::with_capacity(vocab_array.len());
    let mut pieces = Vec::with_capacity(vocab_array.len());

    for item in vocab_array {
        let arr = item.as_array()?;
        let token = arr.get(0)?.as_str()?;
        let score = arr.get(1)?.as_f64()?;
        vocab.push(token.to_string());
        pieces.push(UnigramPiece {
            token: token.to_string(),
            score,
        });
    }

    // Debug: Check what's at position 24
    if vocab.len() > 24 {
        eprintln!("DEBUG: vocab[24] = '{}' (bytes: {:?})", vocab[24], vocab[24].as_bytes());
        eprintln!("DEBUG: pieces[24].token = '{}' (bytes: {:?})", pieces[24].token, pieces[24].token.as_bytes());
    }

    // Debug: Build a test hashmap and check lookup
    let test_map: std::collections::HashMap<&str, u32> = vocab
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i as u32))
        .collect();
    eprintln!("DEBUG: test_map.get(\"▁a\") = {:?}", test_map.get("▁a"));
    eprintln!("DEBUG: test_map.get(vocab[24].as_str()) = {:?}", test_map.get(vocab[24].as_str()));
    eprintln!("DEBUG: vocab[24] == \"▁a\" ? {}", vocab[24] == "▁a");

    let unk_id = model.get("unk_id")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(0);

    let config = UnigramFastConfig {
        unk_id,
        add_prefix_space: true,
        fuse_unk: true,
        ..Default::default()
    };

    let mut vocab_map = AHashMap::with_capacity(vocab.len());
    for (i, token) in vocab.iter().enumerate() {
        vocab_map.insert(token.clone(), i as u32);
    }
    let vocabulary = Vocabulary::new(vocab_map, SpecialTokens {
        unk_token: model.get("unk_token").and_then(|v| v.as_str()).map(|s| s.to_string()),
        ..Default::default()
    });

    let fast_tok = UnigramFast::new(vocabulary, pieces, config);
    Some((fast_tok, hf_tok))
}

fn compare_tokenization(fast_tok: &UnigramFast, hf_tok: &HfTokenizer, text: &str) -> bool {
    let hf_enc = match hf_tok.encode(text, false) {
        Ok(enc) => enc,
        Err(_) => return false,
    };

    let hf_ids: Vec<u32> = hf_enc.get_ids().to_vec();
    let fast_ids = fast_tok.encode_hf_compat(text);

    if hf_ids != fast_ids {
        eprintln!("=== Mismatch for: '{}' ===", text);
        eprintln!("HF IDs:   {:?}", hf_ids);
        eprintln!("Fast IDs: {:?}", fast_ids);

        let hf_tokens: Vec<_> = hf_ids.iter()
            .filter_map(|&id| hf_tok.id_to_token(id))
            .collect();
        let fast_tokens: Vec<_> = fast_ids.iter()
            .filter_map(|&id| fast_tok.id_to_token(id))
            .collect();

        eprintln!("HF Tokens:   {:?}", hf_tokens);
        eprintln!("Fast Tokens: {:?}", fast_tokens);
        return false;
    }

    true
}

// ============================================================================
// Basic Accuracy Tests
// ============================================================================

#[test]
fn test_single_word() {
    let (fast_tok, hf_tok) = match load_tokenizers() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: XLNet tokenizer not found");
            return;
        }
    };

    let test_cases = vec![
        "Hello",
        "hello",
        "world",
        "test",
        "tokenization",
        "machine",
        "learning",
        "artificial",
        "intelligence",
    ];

    let mut passed = 0;
    let mut failed = 0;

    for text in test_cases {
        if compare_tokenization(&fast_tok, &hf_tok, text) {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    println!("Single word tests: {} passed, {} failed", passed, failed);
    assert_eq!(failed, 0, "Some single word tests failed");
}

#[test]
fn test_multi_word() {
    let (fast_tok, hf_tok) = match load_tokenizers() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: XLNet tokenizer not found");
            return;
        }
    };

    let test_cases = vec![
        "Hello world",
        "hello world",
        "The quick brown fox",
        "jumps over the lazy dog",
        "Machine learning is powerful",
        "Natural language processing",
        "This is a test sentence",
    ];

    let mut passed = 0;
    let mut failed = 0;

    for text in test_cases {
        if compare_tokenization(&fast_tok, &hf_tok, text) {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    println!("Multi-word tests: {} passed, {} failed", passed, failed);
    assert_eq!(failed, 0, "Some multi-word tests failed");
}

#[test]
fn test_punctuation() {
    let (fast_tok, hf_tok) = match load_tokenizers() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: XLNet tokenizer not found");
            return;
        }
    };

    let test_cases = vec![
        "Hello, world!",
        "Hello, world.",
        "What's up?",
        "It's a beautiful day!",
        "Question? Answer.",
        "Multiple... dots...",
        "Semi;colons:and,commas",
        "Parentheses (like this) work",
        "Quotes \"are\" 'important'",
    ];

    let mut passed = 0;
    let mut failed = 0;

    for text in test_cases {
        if compare_tokenization(&fast_tok, &hf_tok, text) {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    println!("Punctuation tests: {} passed, {} failed", passed, failed);
    assert_eq!(failed, 0, "Some punctuation tests failed");
}

#[test]
fn test_numbers() {
    let (fast_tok, hf_tok) = match load_tokenizers() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: XLNet tokenizer not found");
            return;
        }
    };

    let test_cases = vec![
        "123",
        "12345",
        "1.5",
        "3.14159",
        "$100",
        "50%",
        "2024",
        "100,000",
        "The year 2024 is here",
        "Price: $99.99",
    ];

    let mut passed = 0;
    let mut failed = 0;

    for text in test_cases {
        if compare_tokenization(&fast_tok, &hf_tok, text) {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    println!("Number tests: {} passed, {} failed", passed, failed);
    assert_eq!(failed, 0, "Some number tests failed");
}

#[test]
fn test_special_characters() {
    let (fast_tok, hf_tok) = match load_tokenizers() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: XLNet tokenizer not found");
            return;
        }
    };

    let test_cases = vec![
        "@mention",
        "#hashtag",
        "email@example.com",
        "https://example.com",
        "path/to/file",
        "C:\\Windows\\System32",
        "x + y = z",
        "a * b / c",
        "<tag>content</tag>",
    ];

    let mut passed = 0;
    let mut failed = 0;

    for text in test_cases {
        if compare_tokenization(&fast_tok, &hf_tok, text) {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    println!("Special character tests: {} passed, {} failed", passed, failed);
    assert_eq!(failed, 0, "Some special character tests failed");
}

#[test]
fn test_unicode() {
    let (fast_tok, hf_tok) = match load_tokenizers() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: XLNet tokenizer not found");
            return;
        }
    };

    let test_cases = vec![
        "café",
        "naïve",
        "résumé",
        "日本語",
        "中文",
        "한국어",
        "العربية",
        "🎉",
        "Hello 世界",
        "Mixed: café and 日本語",
    ];

    let mut passed = 0;
    let mut failed = 0;

    for text in test_cases {
        if compare_tokenization(&fast_tok, &hf_tok, text) {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    println!("Unicode tests: {} passed, {} failed", passed, failed);
    assert_eq!(failed, 0, "Some unicode tests failed");
}

#[test]
fn test_edge_cases() {
    let (fast_tok, hf_tok) = match load_tokenizers() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: XLNet tokenizer not found");
            return;
        }
    };

    let test_cases = vec![
        "a",
        " ",
        "  ",
        "   hello   ",
        "",
        "a b c d e",
        "aaaaaaaaaa",
        "ALLCAPS",
        "MiXeD CaSe",
        "under_score",
        "hyphen-ated",
        "CamelCase",
        "snake_case",
    ];

    let mut passed = 0;
    let mut failed = 0;

    for text in test_cases {
        if compare_tokenization(&fast_tok, &hf_tok, text) {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    println!("Edge case tests: {} passed, {} failed", passed, failed);
    assert_eq!(failed, 0, "Some edge case tests failed");
}

#[test]
fn test_long_text() {
    let (fast_tok, hf_tok) = match load_tokenizers() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: XLNet tokenizer not found");
            return;
        }
    };

    let test_cases = vec![
        "The quick brown fox jumps over the lazy dog. This is a longer sentence to test the tokenizer.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "Natural language processing (NLP) is a field of computer science and artificial intelligence concerned with the interactions between computers and human language.",
    ];

    let mut passed = 0;
    let mut failed = 0;

    for text in test_cases {
        if compare_tokenization(&fast_tok, &hf_tok, text) {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    println!("Long text tests: {} passed, {} failed", passed, failed);
    assert_eq!(failed, 0, "Some long text tests failed");
}

// ============================================================================
// Comprehensive Random Tests
// ============================================================================

#[test]
fn test_comprehensive_accuracy() {
    let (fast_tok, hf_tok) = match load_tokenizers() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: XLNet tokenizer not found");
            return;
        }
    };

    // Large set of test cases for comprehensive coverage
    let test_cases = vec![
        // Simple words
        "hello", "world", "test", "data", "code",
        // Multiple words
        "hello world", "test data", "code review",
        // Sentences
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning enables computers to learn.",
        "Natural language processing is a key AI component.",
        // Mixed content
        "Version 2.0 released on 2024-01-01",
        "Price: $99.99 (50% off!)",
        "Email: test@example.com",
        // Unicode
        "Hello 世界", "café résumé", "日本語テスト",
        // Special patterns
        "UPPERCASE", "lowercase", "MiXeD",
        "under_score", "hyphen-ated", "dot.separated",
        // Numbers
        "123", "3.14159", "1,000,000",
        // Punctuation heavy
        "What?! Really... Wow!",
        "Question: Answer; More: Details.",
        // Whitespace variations
        "  spaced  out  ", "no    breaks",
        // Code-like
        "function() { return true; }",
        "if (x > 0) { print(x); }",
        // URLs and paths
        "https://example.com/path?query=value",
        "/usr/local/bin/program",
        // Edge cases
        "a", "ab", "abc",
        "", " ", "  ",
    ];

    let mut passed = 0;
    let mut failed = 0;
    let mut failures = Vec::new();

    for text in &test_cases {
        if compare_tokenization(&fast_tok, &hf_tok, text) {
            passed += 1;
        } else {
            failed += 1;
            failures.push(text.to_string());
        }
    }

    let accuracy = (passed as f64 / test_cases.len() as f64) * 100.0;
    println!("\n=== Comprehensive Accuracy Results ===");
    println!("Total: {} tests", test_cases.len());
    println!("Passed: {}", passed);
    println!("Failed: {}", failed);
    println!("Accuracy: {:.2}%", accuracy);

    if !failures.is_empty() {
        println!("\nFailed test cases:");
        for (i, text) in failures.iter().enumerate().take(10) {
            println!("  {}. '{}'", i + 1, text);
        }
        if failures.len() > 10 {
            println!("  ... and {} more", failures.len() - 10);
        }
    }

    assert_eq!(accuracy, 100.0, "Accuracy must be 100% for HuggingFace compatibility");
}

// ============================================================================
// SIMD Variant Tests
// ============================================================================

#[test]
fn test_simd_matches_scalar() {
    let (fast_tok, _) = match load_tokenizers() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: XLNet tokenizer not found");
            return;
        }
    };

    let test_cases = vec![
        "Hello world",
        "The quick brown fox",
        "Machine learning is powerful",
        "日本語テスト",
        "Mixed: café and 世界",
        "Numbers: 123, 456.78, $99.99",
    ];

    for text in test_cases {
        let scalar_ids = fast_tok.encode_hf_compat(text);
        let simd_ids = fast_tok.encode_hf_compat_simd(text);

        assert_eq!(
            scalar_ids, simd_ids,
            "SIMD and scalar should produce identical results for: '{}'",
            text
        );
    }

    println!("SIMD matches scalar: All tests passed");
}

#[test]
fn test_batch_encoding() {
    let (fast_tok, hf_tok) = match load_tokenizers() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: XLNet tokenizer not found");
            return;
        }
    };

    let texts: Vec<&str> = vec![
        "Hello world",
        "The quick brown fox",
        "Machine learning",
        "Natural language processing",
        "Artificial intelligence",
    ];

    let batch_results = fast_tok.encode_batch_hf_compat(&texts);

    for (i, text) in texts.iter().enumerate() {
        let hf_enc = hf_tok.encode(*text, false).unwrap();
        let hf_ids: Vec<u32> = hf_enc.get_ids().to_vec();

        assert_eq!(
            batch_results[i], hf_ids,
            "Batch encoding mismatch for: '{}'",
            text
        );
    }

    println!("Batch encoding: All tests passed");
}

// ============================================================================
// Debug Tests
// ============================================================================

#[test]
fn test_debug_trie() {
    let (fast_tok, hf_tok) = match load_tokenizers() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: XLNet tokenizer not found");
            return;
        }
    };

    // Test the trie directly
    println!("\n=== Trie Debug Test ===");

    // First, let's build a simple trie manually to verify the trie builder works
    use budtiktok_core::unigram_fast::DoubleArrayTrie;
    let test_vocab = vec![
        ("▁", 17u32),
        ("▁a", 24),
        ("a", 101),
    ];
    let test_trie = DoubleArrayTrie::from_vocab(test_vocab.into_iter());
    println!("\nManual trie test:");
    println!("  get('▁') = {:?}", test_trie.get("▁".as_bytes()));
    println!("  get('▁a') = {:?}", test_trie.get("▁a".as_bytes()));
    println!("  get('a') = {:?}", test_trie.get("a".as_bytes()));

    // Print vocabulary stats
    println!("\nVocab stats:");
    println!("  vocab_size() = {}", fast_tok.vocab_size());

    // Check if id_to_token works
    println!("\nChecking id_to_token:");
    println!("  id 17 -> {:?}", fast_tok.id_to_token(17));
    println!("  id 24 -> {:?}", fast_tok.id_to_token(24));
    println!("  id 101 -> {:?}", fast_tok.id_to_token(101));

    // Check token_to_id
    println!("\nChecking token_to_id:");
    println!("  '▁' -> {:?}", fast_tok.token_to_id("▁"));
    println!("  '▁a' -> {:?}", fast_tok.token_to_id("▁a"));
    println!("  'a' -> {:?}", fast_tok.token_to_id("a"));

    // Check if ▁a exists in the trie
    let spiece_a = "▁a";
    let spiece = "▁";
    let a = "a";

    println!("Token lookup test:");
    println!("  '▁' ({:?}) -> ID {:?}", spiece.as_bytes(), fast_tok.trie_lookup(spiece.as_bytes()));
    println!("  '▁a' ({:?}) -> ID {:?}", spiece_a.as_bytes(), fast_tok.trie_lookup(spiece_a.as_bytes()));
    println!("  'a' ({:?}) -> ID {:?}", a.as_bytes(), fast_tok.trie_lookup(a.as_bytes()));

    // Check score lookup
    if let Some(id_a) = fast_tok.trie_lookup(spiece_a.as_bytes()) {
        println!("\nScore for '▁a' (ID {}): this confirms trie has it", id_a);
    } else {
        println!("\nERROR: '▁a' NOT FOUND in trie!");
    }

    // Now trace through what happens for "aaaaaaaaaa"
    let text = "aaaaaaaaaa";
    println!("\n--- Encoding '{}' ---", text);

    let hf_enc = hf_tok.encode(text, false).unwrap();
    let fast_ids = fast_tok.encode_hf_compat(text);

    println!("HF IDs: {:?}", hf_enc.get_ids());
    println!("Fast IDs: {:?}", fast_ids);

    let hf_tokens: Vec<_> = hf_enc.get_ids().iter()
        .filter_map(|&id| hf_tok.id_to_token(id))
        .collect();
    let fast_tokens: Vec<_> = fast_ids.iter()
        .filter_map(|&id| fast_tok.id_to_token(id))
        .collect();

    println!("HF Tokens: {:?}", hf_tokens);
    println!("Fast Tokens: {:?}", fast_tokens);

    // Check preprocessing
    let preprocessed = budtiktok_core::unigram_fast::preprocess_text(text, true);
    let preprocessed_str = String::from_utf8_lossy(&preprocessed);
    println!("\nPreprocessed: '{}' (bytes: {:?})", preprocessed_str, &preprocessed);

    // Verify the tokens match
    assert_eq!(hf_enc.get_ids(), fast_ids.as_slice(),
        "Token IDs should match for '{}'", text);
}

// ============================================================================
// Performance Sanity Check
// ============================================================================

#[test]
fn test_performance_sanity() {
    let (fast_tok, hf_tok) = match load_tokenizers() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: XLNet tokenizer not found");
            return;
        }
    };

    let text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let iterations = 100;

    // Warm up
    for _ in 0..10 {
        let _ = fast_tok.encode_hf_compat(&text);
        let _ = hf_tok.encode(text.as_str(), false);
    }

    // Benchmark BudTikTok
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = fast_tok.encode_hf_compat(&text);
    }
    let budtiktok_time = start.elapsed();

    // Benchmark HuggingFace
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = hf_tok.encode(text.as_str(), false);
    }
    let hf_time = start.elapsed();

    let speedup = hf_time.as_secs_f64() / budtiktok_time.as_secs_f64();

    println!("\n=== Performance Sanity Check ===");
    println!("Text length: {} chars", text.len());
    println!("Iterations: {}", iterations);
    println!("BudTikTok (scalar): {:.2}ms", budtiktok_time.as_millis());
    println!("HuggingFace: {:.2}ms", hf_time.as_millis());
    println!("Speedup: {:.1}x", speedup);

    // Sanity check: should be faster (not a strict requirement for accuracy tests)
    assert!(speedup > 1.0, "BudTikTok should be faster than HuggingFace");
}

#[test]
fn test_simd_performance_sanity() {
    let (fast_tok, hf_tok) = match load_tokenizers() {
        Some(t) => t,
        None => {
            eprintln!("Skipping: XLNet tokenizer not found");
            return;
        }
    };

    let text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let iterations = 100;

    // Warm up
    for _ in 0..10 {
        let _ = fast_tok.encode_hf_compat_simd(&text);
    }

    // Benchmark BudTikTok SIMD
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = fast_tok.encode_hf_compat_simd(&text);
    }
    let simd_time = start.elapsed();

    // Benchmark HuggingFace
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = hf_tok.encode(text.as_str(), false);
    }
    let hf_time = start.elapsed();

    let speedup = hf_time.as_secs_f64() / simd_time.as_secs_f64();

    println!("\n=== SIMD Performance Sanity Check ===");
    println!("Text length: {} chars", text.len());
    println!("Iterations: {}", iterations);
    println!("BudTikTok (SIMD): {:.2}ms", simd_time.as_millis());
    println!("HuggingFace: {:.2}ms", hf_time.as_millis());
    println!("Speedup: {:.1}x", speedup);
}
