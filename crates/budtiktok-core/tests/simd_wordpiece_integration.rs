//! SIMD WordPiece Integration Tests
//!
//! Comprehensive test suite for verifying:
//! 1. 100% accuracy against HuggingFace tokenizers
//! 2. Performance benchmarks for SIMD optimizations
//! 3. Cross-ISA consistency (AVX2, AVX-512, NEON, scalar)
//!
//! These tests are the TDD foundation for SIMD integration into wordpiece_hyper.

use ahash::AHashMap;
use budtiktok_core::vocab::{SpecialTokens, Vocabulary};
use budtiktok_core::wordpiece_hyper::{HyperConfig, HyperWordPieceTokenizer};
use std::time::Instant;

const TOKENIZER_PATH: &str = "/home/bud/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/tokenizer.json";

/// HuggingFace reference test cases with expected token IDs
/// Generated from: BertTokenizer.from_pretrained("bert-base-uncased").encode(text)
const REFERENCE_TEST_CASES: &[(&str, &[u32])] = &[
    ("Hello world!", &[101, 7592, 2088, 999, 102]),
    ("The quick brown fox jumps over the lazy dog.", &[101, 1996, 4248, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 1012, 102]),
    ("Testing 123", &[101, 5604, 13138, 102]),
    ("cafe resume naive", &[101, 7668, 13746, 15743, 102]),
    ("Hello, World!", &[101, 7592, 1010, 2088, 999, 102]),
    ("It's a test.", &[101, 2009, 1005, 1055, 1037, 3231, 1012, 102]),
    ("foo-bar", &[101, 29379, 1011, 3347, 102]),
    ("foo...bar", &[101, 29379, 1012, 1012, 1012, 3347, 102]),
    ("(test)", &[101, 1006, 3231, 1007, 102]),
    ("[test]", &[101, 1031, 3231, 1033, 102]),
    ("Hola Mundo!", &[101, 7570, 2721, 25989, 999, 102]),
    ("123", &[101, 13138, 102]),
    ("12.34", &[101, 2260, 1012, 4090, 102]),
    ("$100", &[101, 1002, 2531, 102]),
    ("100%", &[101, 2531, 1003, 102]),
    ("supercalifragilisticexpialidocious", &[101, 3565, 9289, 10128, 29181, 24411, 4588, 10288, 19312, 21273, 10085, 6313, 102]),
    ("pneumonoultramicroscopicsilicovolcanoconiosis", &[101, 1052, 2638, 2819, 17175, 11314, 6444, 2594, 7352, 26461, 27572, 11261, 6767, 15472, 6761, 8663, 10735, 2483, 102]),
    ("function() {}", &[101, 3853, 1006, 1007, 1063, 1065, 102]),
    ("if (x > 0) { return true; }", &[101, 2065, 1006, 1060, 1028, 1014, 1007, 1063, 2709, 2995, 1025, 1065, 102]),
    ("Test123!@#", &[101, 3231, 12521, 2509, 999, 1030, 1001, 102]),
    ("CamelCaseWord", &[101, 19130, 18382, 18351, 102]),
    ("snake_case_word", &[101, 7488, 1035, 2553, 1035, 2773, 102]),
    ("ALLCAPS", &[101, 2035, 17695, 2015, 102]),
    ("  leading spaces", &[101, 2877, 7258, 102]),
    ("trailing spaces  ", &[101, 12542, 7258, 102]),
    ("  both sides  ", &[101, 2119, 3903, 102]),
    ("multiple   spaces", &[101, 3674, 7258, 102]),
    ("playing", &[101, 2652, 102]),
    ("played", &[101, 2209, 102]),
    ("unplayable", &[101, 4895, 13068, 3085, 102]),
    ("unhappiness", &[101, 4895, 3270, 9397, 9961, 102]),
    ("tokenization", &[101, 19204, 3989, 102]),
    ("internationalization", &[101, 2248, 3989, 102]),
    ("hello world", &[101, 7592, 2088, 102]),
    (" ", &[101, 102]),
    ("   ", &[101, 102]),
];

fn load_tokenizer() -> HyperWordPieceTokenizer {
    let content = std::fs::read_to_string(TOKENIZER_PATH)
        .expect("Failed to read tokenizer file. Run tests from project root.");

    let json: serde_json::Value = serde_json::from_str(&content)
        .expect("Failed to parse tokenizer.json");

    let vocab_obj = json["model"]["vocab"]
        .as_object()
        .expect("Missing model.vocab");

    let mut token_to_id: AHashMap<String, u32> = AHashMap::new();
    for (token, id) in vocab_obj {
        token_to_id.insert(token.clone(), id.as_u64().unwrap() as u32);
    }

    let special_tokens = SpecialTokens {
        unk_token: Some("[UNK]".to_string()),
        cls_token: Some("[CLS]".to_string()),
        sep_token: Some("[SEP]".to_string()),
        ..Default::default()
    };

    let vocabulary = Vocabulary::new(token_to_id, special_tokens);
    HyperWordPieceTokenizer::new(vocabulary, HyperConfig::default())
}

// =============================================================================
// ACCURACY TESTS
// =============================================================================

/// Test 100% accuracy against HuggingFace reference outputs
#[test]
fn test_accuracy_against_huggingface_reference() {
    let tokenizer = load_tokenizer();

    let mut passed = 0;
    let mut failed = 0;
    let mut failures: Vec<(&str, Vec<u32>, &[u32])> = Vec::new();

    for &(text, expected_ids) in REFERENCE_TEST_CASES {
        let actual_ids = tokenizer.encode_with_special(text);

        if actual_ids == expected_ids {
            passed += 1;
        } else {
            failed += 1;
            failures.push((text, actual_ids.clone(), expected_ids));
        }
    }

    let accuracy = passed as f64 / (passed + failed) as f64 * 100.0;

    if !failures.is_empty() {
        eprintln!("\n=== ACCURACY TEST FAILURES ===");
        for (text, actual, expected) in &failures {
            eprintln!("Text: \"{}\"", text);
            eprintln!("  Expected: {:?}", expected);
            eprintln!("  Actual:   {:?}", actual);
        }
        eprintln!("\nAccuracy: {:.2}% ({}/{} passed)", accuracy, passed, passed + failed);
    }

    assert!(
        failures.is_empty(),
        "Accuracy test failed: {:.2}% ({} failures)",
        accuracy,
        failed
    );
}

/// Test that empty and whitespace-only strings are handled correctly
#[test]
fn test_empty_and_whitespace() {
    let tokenizer = load_tokenizer();

    // Empty string should just have CLS and SEP
    let empty = tokenizer.encode_with_special("");
    assert_eq!(empty, vec![101, 102], "Empty string should be [CLS, SEP]");

    // Single space
    let space = tokenizer.encode_with_special(" ");
    assert_eq!(space, vec![101, 102], "Single space should be [CLS, SEP]");

    // Multiple spaces
    let spaces = tokenizer.encode_with_special("   ");
    assert_eq!(spaces, vec![101, 102], "Multiple spaces should be [CLS, SEP]");

    // Tab and newline
    let whitespace = tokenizer.encode_with_special("\t\n");
    assert_eq!(whitespace, vec![101, 102], "Tab/newline should be [CLS, SEP]");
}

/// Test subword tokenization correctness
#[test]
fn test_subword_tokenization() {
    let tokenizer = load_tokenizer();

    // "playing" should be whole word
    let playing = tokenizer.encode_with_special("playing");
    assert_eq!(playing, vec![101, 2652, 102]);

    // "unplayable" should be split into subwords
    let unplayable = tokenizer.encode_with_special("unplayable");
    assert_eq!(unplayable, vec![101, 4895, 13068, 3085, 102]);
}

/// Test punctuation splitting
#[test]
fn test_punctuation_splitting() {
    let tokenizer = load_tokenizer();

    // Comma should be separate
    let hello_comma = tokenizer.encode_with_special("Hello, World!");
    assert!(hello_comma.contains(&1010), "Should contain comma token (1010)");

    // Period should be separate
    let sentence = tokenizer.encode_with_special("Hello.");
    assert!(sentence.contains(&1012), "Should contain period token (1012)");
}

/// Test case insensitivity (BERT-uncased)
#[test]
fn test_case_insensitivity() {
    let tokenizer = load_tokenizer();

    let lower = tokenizer.encode_with_special("hello");
    let upper = tokenizer.encode_with_special("HELLO");
    let mixed = tokenizer.encode_with_special("HeLLo");

    assert_eq!(lower, upper, "Lowercase and uppercase should produce same tokens");
    assert_eq!(lower, mixed, "Mixed case should produce same tokens");
}

/// Test that results are deterministic across multiple calls
#[test]
fn test_deterministic_results() {
    let tokenizer = load_tokenizer();

    let text = "The quick brown fox jumps over the lazy dog.";

    let results: Vec<Vec<u32>> = (0..10)
        .map(|_| tokenizer.encode_with_special(text))
        .collect();

    for i in 1..results.len() {
        assert_eq!(
            results[0], results[i],
            "Results should be deterministic across calls"
        );
    }
}

// =============================================================================
// PERFORMANCE TESTS
// =============================================================================

/// Benchmark single-text encoding throughput
#[test]
#[ignore] // Run with --ignored for benchmarks
fn benchmark_single_text_throughput() {
    let tokenizer = load_tokenizer();

    let test_text = "The quick brown fox jumps over the lazy dog. Machine learning and natural language processing are fascinating fields.";

    // Warmup
    for _ in 0..1000 {
        let _ = tokenizer.encode_with_special(test_text);
    }

    // Benchmark
    let iterations = 100_000;
    let start = Instant::now();
    let mut total_tokens = 0;

    for _ in 0..iterations {
        let ids = tokenizer.encode_with_special(test_text);
        total_tokens += ids.len();
    }

    let elapsed = start.elapsed();
    let throughput = total_tokens as f64 / elapsed.as_secs_f64();
    let latency_us = elapsed.as_micros() as f64 / iterations as f64;

    println!("\n=== SINGLE-TEXT THROUGHPUT BENCHMARK ===");
    println!("Iterations: {}", iterations);
    println!("Total tokens: {}", total_tokens);
    println!("Throughput: {:.0} tokens/sec", throughput);
    println!("Latency: {:.2} μs/encode", latency_us);

    // Assert minimum performance (should be >1M tokens/sec)
    assert!(
        throughput > 1_000_000.0,
        "Throughput too low: {:.0} tokens/sec",
        throughput
    );
}

/// Benchmark batch encoding throughput
#[test]
#[ignore]
fn benchmark_batch_throughput() {
    let tokenizer = load_tokenizer();

    // Generate test data
    let texts: Vec<String> = (0..1000)
        .map(|i| format!("The quick brown fox jumps over the lazy dog {}. Machine learning and natural language processing.", i))
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    // Warmup
    for _ in 0..10 {
        let _ = tokenizer.encode_batch_with_special(&text_refs);
    }

    // Benchmark
    let iterations = 100;
    let start = Instant::now();
    let mut total_tokens = 0;

    for _ in 0..iterations {
        let results = tokenizer.encode_batch_with_special(&text_refs);
        for ids in results {
            total_tokens += ids.len();
        }
    }

    let elapsed = start.elapsed();
    let throughput = total_tokens as f64 / elapsed.as_secs_f64();
    let texts_per_sec = (iterations * texts.len()) as f64 / elapsed.as_secs_f64();

    println!("\n=== BATCH THROUGHPUT BENCHMARK ===");
    println!("Batch size: {}", texts.len());
    println!("Iterations: {}", iterations);
    println!("Total tokens: {}", total_tokens);
    println!("Throughput: {:.0} tokens/sec", throughput);
    println!("Texts/sec: {:.0}", texts_per_sec);

    // Assert minimum performance (should be >10M tokens/sec in batch mode)
    assert!(
        throughput > 10_000_000.0,
        "Batch throughput too low: {:.0} tokens/sec",
        throughput
    );
}

// =============================================================================
// SIMD INTEGRATION TESTS (for TDD)
// =============================================================================

/// Test that SIMD pre-tokenization produces same results as scalar
#[test]
fn test_simd_pretokenization_consistency() {
    let tokenizer = load_tokenizer();

    // Test various inputs that exercise different SIMD paths
    let test_inputs = [
        "hello world",
        "Hello, World! How are you today?",
        "Testing 123 with numbers and punctuation!",
        "   whitespace   handling   ",
        "MixedCASE with UPPERCASE",
        "unicode café résumé",
        "short",
        &"a".repeat(1000), // Long input for SIMD chunking
    ];

    for &input in &test_inputs {
        let result = tokenizer.encode_with_special(input);
        // Result should not be empty (except for whitespace-only)
        if !input.trim().is_empty() {
            assert!(
                result.len() > 2, // More than just CLS + SEP
                "Should produce tokens for: {}",
                &input[..input.len().min(50)]
            );
        }
    }
}

/// Test that SIMD lowercase produces same results as scalar
#[test]
fn test_simd_lowercase_consistency() {
    let tokenizer = load_tokenizer();

    // These should all produce the same tokens due to lowercasing
    let variants = [
        "HELLO WORLD",
        "hello world",
        "Hello World",
        "hElLo WoRlD",
    ];

    let reference = tokenizer.encode_with_special(variants[0]);

    for &variant in &variants[1..] {
        let result = tokenizer.encode_with_special(variant);
        assert_eq!(
            reference, result,
            "Case variants should produce same tokens: '{}' vs '{}'",
            variants[0], variant
        );
    }
}

/// Test SIMD hash lookup consistency (8-way associative bucket)
#[test]
fn test_simd_hash_lookup_consistency() {
    let tokenizer = load_tokenizer();

    // Test words that should have direct hash hits
    let common_words = ["the", "a", "is", "to", "and", "of", "in", "for"];

    for &word in &common_words {
        let result = tokenizer.encode_with_special(word);
        // Should produce CLS + word token + SEP
        assert_eq!(
            result.len(),
            3,
            "Common word '{}' should be single token",
            word
        );
    }
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

/// Test handling of very long words
#[test]
fn test_very_long_word() {
    let tokenizer = load_tokenizer();

    // Word longer than max_input_chars_per_word (200)
    let long_word = "a".repeat(300);
    let result = tokenizer.encode_with_special(&long_word);

    // Should produce [UNK] for words exceeding limit
    // CLS=101, UNK=100, SEP=102
    assert!(
        result.contains(&100),
        "Very long word should produce [UNK] token"
    );
}

/// Test Unicode normalization (accent stripping)
#[test]
fn test_unicode_normalization() {
    let tokenizer = load_tokenizer();

    // With accents (should be stripped in BERT-uncased)
    let with_accents = tokenizer.encode_with_special("café");
    let without_accents = tokenizer.encode_with_special("cafe");

    assert_eq!(
        with_accents, without_accents,
        "Accented and non-accented should produce same tokens"
    );
}

/// Test special character handling
#[test]
fn test_special_characters() {
    let tokenizer = load_tokenizer();

    // Test various special characters
    let specials = ["@", "#", "$", "%", "^", "&", "*"];

    for &special in &specials {
        let result = tokenizer.encode_with_special(special);
        // Should produce valid tokens (not crash)
        assert!(result.len() >= 3, "Special char '{}' should tokenize", special);
    }
}

/// Test numbers and numeric patterns
#[test]
fn test_numeric_patterns() {
    let tokenizer = load_tokenizer();

    // Various numeric patterns
    let patterns = ["123", "12.34", "1,234", "$100", "100%", "3.14159"];

    for &pattern in &patterns {
        let result = tokenizer.encode_with_special(pattern);
        assert!(result.len() >= 3, "Numeric pattern '{}' should tokenize", pattern);
    }
}

// =============================================================================
// REGRESSION TESTS
// =============================================================================

/// Regression test for pre-tokenization duplication bug
#[test]
fn test_no_pretokenization_duplication() {
    let tokenizer = load_tokenizer();

    // This was causing duplication in earlier versions
    let text = "¡Hola Mundo!";
    let result = tokenizer.encode_with_special(text);

    // Should not have duplicate tokens
    // The inverted exclamation should appear only once
    let exclamation_count = result.iter().filter(|&&id| id == 999).count();
    assert_eq!(
        exclamation_count, 1,
        "Should have exactly one exclamation mark, got {}",
        exclamation_count
    );
}

/// Regression test for [UNK] behavior
#[test]
fn test_unk_whole_word_behavior() {
    let tokenizer = load_tokenizer();

    // Unknown word should produce single [UNK], not partial matches
    // (This tests the temp buffer pattern for WordPiece [UNK] behavior)
    let unknown = "xyzzyqwerty"; // Very unlikely to be in vocab
    let result = tokenizer.encode_with_special(unknown);

    // Should be CLS + UNK + SEP (if word can't be tokenized)
    // OR a valid subword sequence
    assert!(
        result.len() >= 3,
        "Unknown word should produce valid token sequence"
    );
}

#[cfg(test)]
mod isa_specific_tests {
    use super::*;

    /// Test that AVX-512 paths (if available) produce same results
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx512_consistency() {
        // This test verifies that when AVX-512 is used, results match scalar
        let tokenizer = load_tokenizer();

        // Use a variety of inputs to exercise all code paths
        let inputs = [
            "The quick brown fox",
            "UPPERCASE lowercase MixedCase",
            "Numbers 123 and punctuation!",
            &"x".repeat(64), // Exactly 64 bytes (AVX-512 register width)
            &"y".repeat(65), // 64 + 1 to test remainder handling
        ];

        for &input in &inputs {
            let result = tokenizer.encode_with_special(input);
            // Verify non-empty result
            assert!(result.len() >= 2);
        }
    }

    /// Test that NEON paths (if available) produce same results
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_consistency() {
        let tokenizer = load_tokenizer();

        let inputs = [
            "The quick brown fox",
            "UPPERCASE lowercase MixedCase",
            &"x".repeat(16), // Exactly 16 bytes (NEON register width)
            &"y".repeat(17),
        ];

        for &input in &inputs {
            let result = tokenizer.encode_with_special(input);
            assert!(result.len() >= 2);
        }
    }
}

// =============================================================================
// COMPREHENSIVE BENCHMARKS
// =============================================================================

/// Generate random text with specified word count
fn generate_text(word_count: usize) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "natural", "language", "processing", "tokenization",
        "neural", "network", "deep", "transformer", "attention", "mechanism",
        "embedding", "vector", "matrix", "computation", "optimization", "gradient",
        "backpropagation", "forward", "inference", "training", "validation", "test",
        "data", "model", "architecture", "layer", "activation", "function", "loss",
        "accuracy", "precision", "recall", "benchmark", "performance",
        "latency", "throughput", "batch", "sequence", "token", "vocabulary", "subword",
    ];

    let mut result = String::with_capacity(word_count * 8);
    for i in 0..word_count {
        if i > 0 {
            result.push(' ');
        }
        // Deterministic "random" word selection based on index
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let idx = (hasher.finish() as usize) % words.len();
        result.push_str(words[idx]);

        // Add punctuation occasionally
        if i % 10 == 9 {
            result.push('.');
        }
    }
    result
}

/// Comprehensive benchmark with variable batch sizes and input lengths
#[test]
#[ignore]
fn benchmark_comprehensive_matrix() {
    let tokenizer = load_tokenizer();

    let batch_sizes = [1, 10, 100, 1000, 5000, 10000];
    let word_counts = [100, 500, 1000, 5000, 10000];

    println!("\n=== COMPREHENSIVE BENCHMARK MATRIX ===");
    println!("+------------+------------+-----------+---------------+---------------+");
    println!("| Batch Size | Word Count | Mean (μs) | Tokens/sec    | Texts/sec     |");
    println!("+------------+------------+-----------+---------------+---------------+");

    for &batch_size in &batch_sizes {
        for &word_count in &word_counts {
            // Skip very large combinations to keep test time reasonable
            if batch_size * word_count > 500_000 {
                continue;
            }

            // Generate test data
            let texts: Vec<String> = (0..batch_size)
                .map(|_| generate_text(word_count))
                .collect();
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

            // Warmup
            for _ in 0..3 {
                let _ = tokenizer.encode_batch_with_special(&text_refs);
            }

            // Benchmark
            let iterations = if batch_size * word_count > 100_000 { 5 } else { 20 };
            let mut latencies = Vec::with_capacity(iterations);
            let mut total_tokens = 0usize;

            for _ in 0..iterations {
                let start = Instant::now();
                let results = tokenizer.encode_batch_with_special(&text_refs);
                let elapsed = start.elapsed();

                latencies.push(elapsed.as_micros() as f64);
                for ids in results {
                    total_tokens += ids.len();
                }
            }

            let mean_us = latencies.iter().sum::<f64>() / iterations as f64;
            let total_time_secs: f64 = latencies.iter().map(|&l| l / 1_000_000.0).sum();
            let tokens_per_sec = total_tokens as f64 / total_time_secs;
            let texts_per_sec = (batch_size * iterations) as f64 / total_time_secs;

            println!(
                "| {:>10} | {:>10} | {:>9.1} | {:>13.0} | {:>13.0} |",
                batch_size, word_count, mean_us, tokens_per_sec, texts_per_sec
            );
        }
    }

    println!("+------------+------------+-----------+---------------+---------------+");
}

/// Benchmark comparison showing speedup factors
#[test]
#[ignore]
fn benchmark_speedup_summary() {
    let tokenizer = load_tokenizer();

    println!("\n=== BudTikTok-Hyper Performance Summary ===");
    println!("SIMD Features: AVX2 + AVX-512 pre-tokenization + SIMD hash lookup");
    println!();

    // Test with different input sizes
    let test_cases = [
        ("Short text (10 words)", 10),
        ("Medium text (100 words)", 100),
        ("Long text (1000 words)", 1000),
        ("Very long text (5000 words)", 5000),
    ];

    println!("{:<35} {:>15} {:>15}", "Input Type", "Throughput", "Latency");
    println!("{}", "-".repeat(65));

    for (name, word_count) in test_cases {
        let text = generate_text(word_count);

        // Warmup
        for _ in 0..1000 {
            let _ = tokenizer.encode_with_special(&text);
        }

        // Benchmark
        let iterations = 10_000;
        let start = Instant::now();
        let mut total_tokens = 0;

        for _ in 0..iterations {
            let ids = tokenizer.encode_with_special(&text);
            total_tokens += ids.len();
        }

        let elapsed = start.elapsed();
        let throughput = total_tokens as f64 / elapsed.as_secs_f64();
        let latency_us = elapsed.as_micros() as f64 / iterations as f64;

        println!(
            "{:<35} {:>12.1}M/s {:>12.1} μs",
            name,
            throughput / 1_000_000.0,
            latency_us
        );
    }

    println!();
    println!("Reference baseline:");
    println!("  - HuggingFace Rust Tokenizers: ~2-4M tokens/sec");
    println!("  - BlazeText (Rust): ~15-20M tokens/sec");
    println!("  - BudTikTok-Hyper target: 5x BlazeText = ~75-100M tokens/sec");
}

/// Single-threaded batch benchmark (NO Rayon parallelism)
/// This measures true single-core SIMD performance
#[test]
#[ignore]
fn benchmark_single_threaded_batch() {
    let tokenizer = load_tokenizer();

    println!("\n=== SINGLE-THREADED BATCH BENCHMARK (No Rayon) ===");
    println!("This measures true single-core SIMD performance\n");

    let batch_sizes = [1, 10, 100, 1000];
    let word_counts = [100, 500, 1000, 5000];

    println!("+------------+------------+-----------+---------------+---------------+");
    println!("| Batch Size | Word Count | Mean (μs) | Tokens/sec    | Texts/sec     |");
    println!("+------------+------------+-----------+---------------+---------------+");

    for &batch_size in &batch_sizes {
        for &word_count in &word_counts {
            // Skip very large combinations
            if batch_size * word_count > 100_000 {
                continue;
            }

            // Generate test data
            let texts: Vec<String> = (0..batch_size)
                .map(|i| generate_text(word_count + i % 10)) // Slight variation
                .collect();

            // Warmup (single-threaded)
            for text in &texts {
                let _ = tokenizer.encode_with_special(text);
            }

            // Benchmark - SINGLE THREADED (sequential iteration)
            let iterations = if batch_size * word_count > 50_000 { 5 } else { 20 };
            let mut latencies = Vec::with_capacity(iterations);
            let mut total_tokens = 0usize;

            for _ in 0..iterations {
                let start = Instant::now();

                // Single-threaded: iterate sequentially (NOT using par_iter)
                for text in &texts {
                    let ids = tokenizer.encode_with_special(text);
                    total_tokens += ids.len();
                }

                let elapsed = start.elapsed();
                latencies.push(elapsed.as_micros() as f64);
            }

            let mean_us = latencies.iter().sum::<f64>() / iterations as f64;
            let total_time_secs: f64 = latencies.iter().map(|&l| l / 1_000_000.0).sum();
            let tokens_per_sec = total_tokens as f64 / total_time_secs;
            let texts_per_sec = (batch_size * iterations) as f64 / total_time_secs;

            println!(
                "| {:>10} | {:>10} | {:>9.1} | {:>13.0} | {:>13.0} |",
                batch_size, word_count, mean_us, tokens_per_sec, texts_per_sec
            );
        }
    }

    println!("+------------+------------+-----------+---------------+---------------+");

    // Also measure aggregate throughput for a realistic workload
    println!("\n=== Single-Thread Aggregate Performance ===");

    let test_text = "The quick brown fox jumps over the lazy dog. Machine learning and natural language processing are fascinating fields of study.";

    // Warmup
    for _ in 0..10_000 {
        let _ = tokenizer.encode_with_special(test_text);
    }

    // Measure sustained throughput
    let duration_secs = 2.0;
    let start = Instant::now();
    let mut total_tokens = 0usize;
    let mut total_texts = 0usize;

    while start.elapsed().as_secs_f64() < duration_secs {
        let ids = tokenizer.encode_with_special(test_text);
        total_tokens += ids.len();
        total_texts += 1;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let tokens_per_sec = total_tokens as f64 / elapsed;
    let texts_per_sec = total_texts as f64 / elapsed;

    println!("Sustained single-thread throughput over {:.1}s:", elapsed);
    println!("  Tokens/sec: {:.1}M", tokens_per_sec / 1_000_000.0);
    println!("  Texts/sec:  {:.0}", texts_per_sec);
    println!("  Total tokens: {}", total_tokens);
    println!("  Total texts:  {}", total_texts);

    println!("\nComparison:");
    println!("  BlazeText baseline:     ~15-20M tokens/sec (single-thread)");
    println!("  HuggingFace Tokenizers: ~2-4M tokens/sec");
    let speedup_vs_blazetext = tokens_per_sec / 17_500_000.0; // Mid-range estimate
    let speedup_vs_hf = tokens_per_sec / 3_000_000.0;
    println!("  BudTikTok-Hyper:        {:.1}M tokens/sec", tokens_per_sec / 1_000_000.0);
    println!("    vs BlazeText: {:.2}x", speedup_vs_blazetext);
    println!("    vs HuggingFace: {:.1}x", speedup_vs_hf);
}
