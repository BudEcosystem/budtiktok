//! BPE Accuracy Tests
//!
//! Validates budtiktok BPE implementations against HuggingFace tokenizers.
//! All implementations (scalar, SIMD, CUDA) must produce identical output
//! to HuggingFace tokenizers for 100% compatibility.

use ahash::AHashMap;
use budtiktok_core::bpe_fast::{FastBpeEncoder, SimdBpeEncoder};
use budtiktok_core::bpe_simd::SimdOptimizedBpeEncoder;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::Tokenizer as HfTokenizer;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;

/// Test result for a single comparison
#[derive(Debug, Clone)]
struct ComparisonResult {
    input: String,
    hf_ids: Vec<u32>,
    bud_ids: Vec<u32>,
    matches: bool,
    encoder_name: String,
}

/// GPT-2 style byte encoder
fn gpt2_byte_encoder() -> [char; 256] {
    let mut encode = ['\0'; 256];
    let mut n = 0u32;

    for b in 0u8..=255 {
        let c = if (b'!'..=b'~').contains(&b)
            || (0xa1..=0xac).contains(&b)
            || (0xae..=0xff).contains(&b)
        {
            b as char
        } else {
            let c = char::from_u32(256 + n).unwrap_or('?');
            n += 1;
            c
        };
        encode[b as usize] = c;
    }

    encode
}

/// Create GPT-2 style byte-level vocabulary and merges
fn create_gpt2_vocab_and_merges() -> (
    AHashMap<String, u32>,
    std::collections::HashMap<String, u32>,
    Vec<(String, String)>,
) {
    let byte_encoder = gpt2_byte_encoder();
    let mut vocab_ahash: AHashMap<String, u32> = AHashMap::new();
    let mut vocab_std: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut next_id = 0u32;

    // Add base byte tokens (256 tokens) - using GPT-2 byte encoding
    for b in 0u8..=255 {
        let c = byte_encoder[b as usize];
        let token = c.to_string();
        vocab_ahash.insert(token.clone(), next_id);
        vocab_std.insert(token, next_id);
        next_id += 1;
    }

    // Space character in GPT-2 encoding is at position 256+32 = char 288 = 'Ġ'
    let space_char = byte_encoder[b' ' as usize];

    // Add common merge pairs using byte-encoded tokens
    // Helper to byte-encode a string
    let byte_encode = |s: &str| -> String {
        s.bytes().map(|b| byte_encoder[b as usize]).collect()
    };

    let common_merges = [
        // Basic letter merges (byte-encoded)
        ("t", "h"), ("th", "e"),
        ("a", "n"), ("an", "d"),
        ("i", "n"), ("in", "g"),
        ("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o"),
        ("w", "o"), ("wo", "r"), ("wor", "l"), ("worl", "d"),
        ("e", "r"), ("o", "n"),
        ("i", "t"), ("o", "f"),
        ("t", "o"), ("a", "t"),
        ("o", "r"), ("e", "n"),
        ("a", "l"), ("r", "e"),
        ("c", "o"), ("co", "m"),
        ("s", "t"), ("st", "a"),
        ("m", "a"), ("ma", "n"),
        ("l", "e"), ("le", "a"),
        ("p", "r"), ("pr", "o"),
        ("b", "e"), ("be", "n"),
        ("q", "u"), ("qu", "i"), ("qui", "c"), ("quic", "k"),
        ("b", "r"), ("br", "o"), ("bro", "w"), ("brow", "n"),
        ("f", "o"), ("fo", "x"),
        ("j", "u"), ("ju", "m"), ("jum", "p"), ("jump", "s"),
        ("o", "v"), ("ov", "e"), ("ove", "r"),
        ("l", "a"), ("la", "z"), ("laz", "y"),
        ("d", "o"), ("do", "g"),
    ];

    // Byte-encode merge pairs and add to vocab
    let mut merge_tuples: Vec<(String, String)> = Vec::new();
    for (first, second) in &common_merges {
        let first_enc = byte_encode(first);
        let second_enc = byte_encode(second);
        let merged_enc = format!("{}{}", first_enc, second_enc);

        if !vocab_ahash.contains_key(&merged_enc) {
            vocab_ahash.insert(merged_enc.clone(), next_id);
            vocab_std.insert(merged_enc, next_id);
            next_id += 1;
        }
        merge_tuples.push((first_enc, second_enc));
    }

    // Add merges with space prefix (Ġ) for words after first word
    let space_prefixed_merges = [
        // Ġ + first letter -> Ġ letter
        (" ", "t"), (" ", "a"), (" ", "i"), (" ", "h"),
        (" ", "w"), (" ", "e"), (" ", "o"), (" ", "c"),
        (" ", "s"), (" ", "m"), (" ", "l"), (" ", "p"),
        (" ", "b"), (" ", "q"), (" ", "f"), (" ", "j"),
        (" ", "d"), (" ", "n"), (" ", "r"),
        // Ġt -> Ġth, Ġthe etc
        (" t", "h"), (" th", "e"),
        (" a", "n"), (" an", "d"),
        (" i", "n"), (" in", "g"),
        (" h", "e"), (" he", "l"), (" hel", "l"), (" hell", "o"),
        (" w", "o"), (" wo", "r"), (" wor", "l"), (" worl", "d"),
        (" q", "u"), (" qu", "i"), (" qui", "c"), (" quic", "k"),
        (" b", "r"), (" br", "o"), (" bro", "w"), (" brow", "n"),
        (" f", "o"), (" fo", "x"),
        (" j", "u"), (" ju", "m"), (" jum", "p"), (" jump", "s"),
        (" o", "v"), (" ov", "e"), (" ove", "r"),
        (" l", "a"), (" la", "z"), (" laz", "y"),
        (" d", "o"), (" do", "g"),
    ];

    for (first, second) in &space_prefixed_merges {
        let first_enc = byte_encode(first);
        let second_enc = byte_encode(second);
        let merged_enc = format!("{}{}", first_enc, second_enc);

        if !vocab_ahash.contains_key(&merged_enc) {
            vocab_ahash.insert(merged_enc.clone(), next_id);
            vocab_std.insert(merged_enc, next_id);
            next_id += 1;
        }
        merge_tuples.push((first_enc, second_enc));
    }

    // Add [UNK] token
    vocab_ahash.insert("<unk>".to_string(), next_id);
    vocab_std.insert("<unk>".to_string(), next_id);

    (vocab_ahash, vocab_std, merge_tuples)
}

/// Generate test sentences
fn generate_test_sentences() -> Vec<String> {
    vec![
        // Basic sentences
        "hello".to_string(),
        "world".to_string(),
        "hello world".to_string(),
        "the quick".to_string(),
        "the quick brown".to_string(),
        "the quick brown fox".to_string(),
        "the quick brown fox jumps".to_string(),
        "the quick brown fox jumps over".to_string(),
        "the quick brown fox jumps over the".to_string(),
        "the quick brown fox jumps over the lazy".to_string(),
        "the quick brown fox jumps over the lazy dog".to_string(),

        // Multiple spaces (edge case)
        "hello  world".to_string(),
        "  hello world  ".to_string(),

        // Single words
        "a".to_string(),
        "an".to_string(),
        "and".to_string(),
        "the".to_string(),
        "in".to_string(),
        "it".to_string(),

        // Repeated words
        "the the the".to_string(),
        "hello hello hello".to_string(),

        // Various lengths
        "a b c d e f".to_string(),
        "one two three four five".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test FastBpeEncoder accuracy against HuggingFace with ByteLevel
    #[test]
    fn test_fast_bpe_accuracy_with_bytelevel() {
        let (vocab_ahash, vocab_std, merges) = create_gpt2_vocab_and_merges();

        // Create budtiktok fast BPE
        let fast_bpe = FastBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");

        // Create HF BPE tokenizer WITH ByteLevel pre-tokenizer
        let hf_bpe = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab_std.clone(), merges.clone())
            .unk_token("<unk>".to_string())
            .build()
            .expect("Failed to build HF BPE");

        let mut hf_tokenizer = HfTokenizer::new(hf_bpe);

        // Set ByteLevel pre-tokenizer to match budtiktok behavior
        hf_tokenizer.with_pre_tokenizer(PreTokenizerWrapper::ByteLevel(
            ByteLevel::new(false, true, false)  // add_prefix_space=false, trim_offsets=true
        ));

        let test_sentences = generate_test_sentences();
        let mut mismatches = Vec::new();
        let mut total_tests = 0;
        let mut passed_tests = 0;

        for text in &test_sentences {
            if text.is_empty() {
                continue;
            }

            total_tests += 1;

            // Get HF encoding
            let hf_result = hf_tokenizer.encode(text.as_str(), false);
            if hf_result.is_err() {
                println!("HF failed to encode: {}", text);
                continue;
            }
            let hf_encoding = hf_result.unwrap();
            let hf_ids: Vec<u32> = hf_encoding.get_ids().to_vec();

            // Get budtiktok encoding
            let bud_ids = fast_bpe.encode(text);

            if hf_ids == bud_ids {
                passed_tests += 1;
            } else {
                mismatches.push(ComparisonResult {
                    input: text.clone(),
                    hf_ids: hf_ids.clone(),
                    bud_ids: bud_ids.clone(),
                    matches: false,
                    encoder_name: "FastBpeEncoder".to_string(),
                });
            }
        }

        println!("\n=== FastBpeEncoder Accuracy Test (ByteLevel) ===");
        println!("Total tests: {}", total_tests);
        println!("Passed: {}", passed_tests);
        println!("Failed: {}", mismatches.len());
        if total_tests > 0 {
            println!("Accuracy: {:.2}%", (passed_tests as f64 / total_tests as f64) * 100.0);
        }

        if !mismatches.is_empty() {
            println!("\nMismatches (first 10):");
            for mismatch in mismatches.iter().take(10) {
                println!("  Input: \"{}\"", mismatch.input);
                println!("    HF IDs:  {:?}", &mismatch.hf_ids[..mismatch.hf_ids.len().min(15)]);
                println!("    Bud IDs: {:?}", &mismatch.bud_ids[..mismatch.bud_ids.len().min(15)]);
            }
        }

        assert!(
            mismatches.is_empty(),
            "FastBpeEncoder has {} mismatches out of {} tests",
            mismatches.len(),
            total_tests
        );
    }

    /// Test internal consistency between all budtiktok encoders
    #[test]
    fn test_internal_consistency() {
        let (vocab_ahash, _, merges) = create_gpt2_vocab_and_merges();

        let fast_bpe = FastBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");
        let simd_bpe = SimdBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");
        let simd_opt_bpe = SimdOptimizedBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");

        let test_sentences = generate_test_sentences();
        let mut inconsistencies = Vec::new();

        for text in &test_sentences {
            if text.is_empty() {
                continue;
            }

            let fast_ids = fast_bpe.encode(text);
            let simd_ids = simd_bpe.encode(text);
            let simd_opt_ids = simd_opt_bpe.encode(text);

            if fast_ids != simd_ids || simd_ids != simd_opt_ids {
                inconsistencies.push((
                    text.clone(),
                    fast_ids.clone(),
                    simd_ids.clone(),
                    simd_opt_ids.clone(),
                ));
            }
        }

        println!("\n=== Internal Consistency Test ===");
        println!("Tested {} sentences", test_sentences.len());
        println!("Inconsistencies: {}", inconsistencies.len());

        if !inconsistencies.is_empty() {
            println!("\nInconsistencies:");
            for (text, fast, simd, simd_opt) in &inconsistencies {
                println!("  Input: \"{}\"", text);
                println!("    Fast:     {:?}", &fast[..fast.len().min(15)]);
                println!("    SIMD:     {:?}", &simd[..simd.len().min(15)]);
                println!("    SIMD Opt: {:?}", &simd_opt[..simd_opt.len().min(15)]);
            }
        }

        assert!(
            inconsistencies.is_empty(),
            "Found {} internal inconsistencies",
            inconsistencies.len()
        );
    }

    /// Test SimdBpeEncoder accuracy against HuggingFace with ByteLevel
    #[test]
    fn test_simd_bpe_accuracy_with_bytelevel() {
        let (vocab_ahash, vocab_std, merges) = create_gpt2_vocab_and_merges();

        let simd_bpe = SimdBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");

        let hf_bpe = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab_std.clone(), merges.clone())
            .unk_token("<unk>".to_string())
            .build()
            .expect("Failed to build HF BPE");

        let mut hf_tokenizer = HfTokenizer::new(hf_bpe);
        hf_tokenizer.with_pre_tokenizer(PreTokenizerWrapper::ByteLevel(
            ByteLevel::new(false, true, false)
        ));

        let test_sentences = generate_test_sentences();
        let mut mismatches = Vec::new();
        let mut total_tests = 0;
        let mut passed_tests = 0;

        for text in &test_sentences {
            if text.is_empty() {
                continue;
            }

            total_tests += 1;

            let hf_result = hf_tokenizer.encode(text.as_str(), false);
            if hf_result.is_err() {
                continue;
            }
            let hf_encoding = hf_result.unwrap();
            let hf_ids: Vec<u32> = hf_encoding.get_ids().to_vec();

            let bud_ids = simd_bpe.encode(text);

            if hf_ids == bud_ids {
                passed_tests += 1;
            } else {
                mismatches.push(ComparisonResult {
                    input: text.clone(),
                    hf_ids,
                    bud_ids,
                    matches: false,
                    encoder_name: "SimdBpeEncoder".to_string(),
                });
            }
        }

        println!("\n=== SimdBpeEncoder Accuracy Test (ByteLevel) ===");
        println!("Total tests: {}", total_tests);
        println!("Passed: {}", passed_tests);
        println!("Failed: {}", mismatches.len());
        if total_tests > 0 {
            println!("Accuracy: {:.2}%", (passed_tests as f64 / total_tests as f64) * 100.0);
        }

        if !mismatches.is_empty() {
            println!("\nMismatches (first 10):");
            for mismatch in mismatches.iter().take(10) {
                println!("  Input: \"{}\"", mismatch.input);
                println!("    HF IDs:  {:?}", &mismatch.hf_ids[..mismatch.hf_ids.len().min(15)]);
                println!("    Bud IDs: {:?}", &mismatch.bud_ids[..mismatch.bud_ids.len().min(15)]);
            }
        }

        assert!(
            mismatches.is_empty(),
            "SimdBpeEncoder has {} mismatches out of {} tests",
            mismatches.len(),
            total_tests
        );
    }

    /// Test SimdOptimizedBpeEncoder accuracy against HuggingFace with ByteLevel
    #[test]
    fn test_simd_optimized_bpe_accuracy_with_bytelevel() {
        let (vocab_ahash, vocab_std, merges) = create_gpt2_vocab_and_merges();

        let simd_opt_bpe = SimdOptimizedBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");
        println!("Using SIMD level: {:?}", simd_opt_bpe.simd_level());

        let hf_bpe = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab_std.clone(), merges.clone())
            .unk_token("<unk>".to_string())
            .build()
            .expect("Failed to build HF BPE");

        let mut hf_tokenizer = HfTokenizer::new(hf_bpe);
        hf_tokenizer.with_pre_tokenizer(PreTokenizerWrapper::ByteLevel(
            ByteLevel::new(false, true, false)
        ));

        let test_sentences = generate_test_sentences();
        let mut mismatches = Vec::new();
        let mut total_tests = 0;
        let mut passed_tests = 0;

        for text in &test_sentences {
            if text.is_empty() {
                continue;
            }

            total_tests += 1;

            let hf_result = hf_tokenizer.encode(text.as_str(), false);
            if hf_result.is_err() {
                continue;
            }
            let hf_encoding = hf_result.unwrap();
            let hf_ids: Vec<u32> = hf_encoding.get_ids().to_vec();

            let bud_ids = simd_opt_bpe.encode(text);

            if hf_ids == bud_ids {
                passed_tests += 1;
            } else {
                mismatches.push(ComparisonResult {
                    input: text.clone(),
                    hf_ids,
                    bud_ids,
                    matches: false,
                    encoder_name: "SimdOptimizedBpeEncoder".to_string(),
                });
            }
        }

        println!("\n=== SimdOptimizedBpeEncoder Accuracy Test (ByteLevel) ===");
        println!("Total tests: {}", total_tests);
        println!("Passed: {}", passed_tests);
        println!("Failed: {}", mismatches.len());
        if total_tests > 0 {
            println!("Accuracy: {:.2}%", (passed_tests as f64 / total_tests as f64) * 100.0);
        }

        if !mismatches.is_empty() {
            println!("\nMismatches (first 10):");
            for mismatch in mismatches.iter().take(10) {
                println!("  Input: \"{}\"", mismatch.input);
                println!("    HF IDs:  {:?}", &mismatch.hf_ids[..mismatch.hf_ids.len().min(15)]);
                println!("    Bud IDs: {:?}", &mismatch.bud_ids[..mismatch.bud_ids.len().min(15)]);
            }
        }

        assert!(
            mismatches.is_empty(),
            "SimdOptimizedBpeEncoder has {} mismatches out of {} tests",
            mismatches.len(),
            total_tests
        );
    }

    /// Batch encoding consistency test
    #[test]
    fn test_batch_encoding_consistency() {
        let (vocab_ahash, _, merges) = create_gpt2_vocab_and_merges();

        let fast_bpe = FastBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");
        let simd_bpe = SimdBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");
        let simd_opt_bpe = SimdOptimizedBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");

        let test_texts: Vec<&str> = vec![
            "hello world",
            "the quick brown fox",
            "jumps over the lazy dog",
            "hello",
            "world",
        ];

        let fast_batch = fast_bpe.encode_batch(&test_texts);
        let simd_batch = simd_bpe.encode_batch(&test_texts);
        let simd_opt_batch = simd_opt_bpe.encode_batch(&test_texts);

        let mut mismatches = 0;
        for (i, ((fast, simd), simd_opt)) in fast_batch.iter()
            .zip(simd_batch.iter())
            .zip(simd_opt_batch.iter())
            .enumerate()
        {
            if fast != simd || simd != simd_opt {
                println!("Batch mismatch at index {}: \"{}\"", i, test_texts[i]);
                println!("  Fast:     {:?}", fast);
                println!("  SIMD:     {:?}", simd);
                println!("  SIMD Opt: {:?}", simd_opt);
                mismatches += 1;
            }
        }

        println!("\n=== Batch Encoding Consistency Test ===");
        println!("Tested {} texts", test_texts.len());
        println!("Mismatches: {}", mismatches);

        assert_eq!(mismatches, 0, "Found {} batch encoding inconsistencies", mismatches);
    }
}
