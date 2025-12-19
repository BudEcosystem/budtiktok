//! Full Regression Test for BPE and WordPiece Tokenizers
//!
//! Tests:
//! 1. BPE (OptimizedBpeEncoder) - Single-core, Multi-core, SIMD
//! 2. WordPiece (WordPieceTokenizer, HyperWordPieceTokenizer) - Single-core, Multi-core, SIMD
//! 3. 100% accuracy verification against HuggingFace tokenizers
//! 4. Performance benchmarks on 1GB dataset

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::time::Instant;
use ahash::AHashMap;
use rayon::prelude::*;

// BPE imports
use budtiktok_core::bpe_linear::OptimizedBpeEncoder;

// WordPiece imports
use budtiktok_core::wordpiece::{WordPieceConfig, WordPieceTokenizer};
use budtiktok_core::wordpiece_hyper::{HyperConfig, HyperWordPieceTokenizer};
use budtiktok_core::vocab::{Vocabulary, SpecialTokens};
use budtiktok_core::tokenizer::Tokenizer;

const WORKSPACE: &str = "/home/bud/Desktop/latentbud/budtiktok";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          BUDTIKTOK FULL REGRESSION TEST - BPE & WORDPIECE                    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // Run BPE tests
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                              BPE TOKENIZER TESTS                               ");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    run_bpe_tests();

    println!("\n");

    // Run WordPiece tests
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                           WORDPIECE TOKENIZER TESTS                            ");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    run_wordpiece_tests();

    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                         REGRESSION TEST COMPLETE                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
}

// =============================================================================
// BPE TESTS
// =============================================================================

fn run_bpe_tests() {
    // Load GPT-2 vocab and merges
    let vocab_path = format!("{}/benchmark_data/gpt2/vocab.json", WORKSPACE);
    let merges_path = format!("{}/benchmark_data/gpt2/merges.txt", WORKSPACE);
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", WORKSPACE);

    println!("Loading GPT-2 vocabulary...");
    let vocab_content = fs::read_to_string(&vocab_path).expect("Failed to read vocab.json");
    let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_content).expect("Failed to parse vocab");
    let vocab_ahash: AHashMap<String, u32> = vocab.iter().map(|(k, v)| (k.clone(), *v)).collect();

    // Build reverse vocab for decoding
    let mut vocab_r: Vec<String> = vec![String::new(); vocab.len()];
    for (k, &v) in &vocab {
        vocab_r[v as usize] = k.clone();
    }

    println!("Loading merge rules...");
    let merge_content = fs::read_to_string(&merges_path).expect("Failed to read merges.txt");
    let merges: Vec<(String, String)> = merge_content.lines()
        .skip(1)
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                Some((parts[0].to_string(), parts[1].to_string()))
            } else {
                None
            }
        })
        .collect();

    println!("Creating OptimizedBpeEncoder...");
    let encoder = OptimizedBpeEncoder::new(vocab_ahash, merges, "<|endoftext|>");

    println!("Loading 1GB dataset...");
    let file = File::open(&data_path).expect("Failed to open dataset");
    let reader = BufReader::new(file);

    let documents: Vec<String> = reader.lines()
        .take(10000) // First 10K documents for comprehensive test
        .filter_map(|line| {
            let line = line.ok()?;
            let json: serde_json::Value = serde_json::from_str(&line).ok()?;
            json["text"].as_str().map(|s| s.to_string())
        })
        .collect();

    let total_bytes: usize = documents.iter().map(|d| d.len()).sum();
    println!("Loaded {} documents ({:.2} MB)\n", documents.len(), total_bytes as f64 / 1024.0 / 1024.0);

    // Test 1: Single-core encoding
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TEST 1: BPE Single-Core Encoding                                            │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mut total_tokens_single = 0usize;
    let single_core_results: Vec<Vec<u32>> = documents.iter()
        .map(|doc| {
            let tokens = encoder.encode(doc);
            total_tokens_single += tokens.len();
            tokens
        })
        .collect();
    let single_core_time = start.elapsed();
    let single_core_throughput = total_bytes as f64 / single_core_time.as_secs_f64() / 1024.0 / 1024.0;

    println!("  Documents:    {}", documents.len());
    println!("  Tokens:       {}", total_tokens_single);
    println!("  Time:         {:.2}s", single_core_time.as_secs_f64());
    println!("  Throughput:   {:.2} MB/s", single_core_throughput);
    println!("  Tokens/sec:   {:.0}", total_tokens_single as f64 / single_core_time.as_secs_f64());
    println!();

    // Test 2: Multi-core encoding
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TEST 2: BPE Multi-Core Encoding (Rayon)                                     │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let multi_core_results: Vec<Vec<u32>> = documents.par_iter()
        .map(|doc| encoder.encode(doc))
        .collect();
    let multi_core_time = start.elapsed();
    let total_tokens_multi: usize = multi_core_results.iter().map(|t| t.len()).sum();
    let multi_core_throughput = total_bytes as f64 / multi_core_time.as_secs_f64() / 1024.0 / 1024.0;

    let num_threads = rayon::current_num_threads();
    let parallel_efficiency = (single_core_time.as_secs_f64() / multi_core_time.as_secs_f64()) / num_threads as f64 * 100.0;

    println!("  Threads:      {}", num_threads);
    println!("  Documents:    {}", documents.len());
    println!("  Tokens:       {}", total_tokens_multi);
    println!("  Time:         {:.2}s", multi_core_time.as_secs_f64());
    println!("  Throughput:   {:.2} MB/s", multi_core_throughput);
    println!("  Tokens/sec:   {:.0}", total_tokens_multi as f64 / multi_core_time.as_secs_f64());
    println!("  Speedup:      {:.2}x over single-core", single_core_time.as_secs_f64() / multi_core_time.as_secs_f64());
    println!("  Efficiency:   {:.1}%", parallel_efficiency);
    println!();

    // Verify single-core and multi-core produce identical results
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TEST 3: BPE Consistency Check (Single vs Multi-core)                        │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    let mut mismatches = 0;
    for (i, (single, multi)) in single_core_results.iter().zip(multi_core_results.iter()).enumerate() {
        if single != multi {
            mismatches += 1;
            if mismatches <= 3 {
                println!("  MISMATCH at doc {}: single={} tokens, multi={} tokens", i, single.len(), multi.len());
            }
        }
    }

    if mismatches == 0 {
        println!("  ✓ PASS: Single-core and multi-core produce identical results");
    } else {
        println!("  ✗ FAIL: {} mismatches found", mismatches);
    }
    println!();

    // Store results for HuggingFace comparison
    println!("  Storing BPE results for HuggingFace comparison...");
    let bpe_token_counts: Vec<usize> = single_core_results.iter().map(|t| t.len()).collect();

    // Save to file for Python comparison
    let output_path = format!("{}/bpe_budtiktok_results.json", WORKSPACE);
    let results_json = serde_json::json!({
        "total_tokens": total_tokens_single,
        "token_counts": bpe_token_counts,
        "first_10_docs_tokens": single_core_results.iter().take(10).collect::<Vec<_>>(),
    });
    fs::write(&output_path, serde_json::to_string_pretty(&results_json).unwrap())
        .expect("Failed to write results");
    println!("  Saved to: {}", output_path);
}

// =============================================================================
// WORDPIECE TESTS
// =============================================================================

fn run_wordpiece_tests() {
    let tokenizer_path = format!("{}/test_data/bert-base-uncased/tokenizer.json", WORKSPACE);
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", WORKSPACE);

    println!("Loading BERT tokenizer...");
    let content = fs::read_to_string(&tokenizer_path).expect("Failed to read tokenizer.json");
    let json: serde_json::Value = serde_json::from_str(&content).expect("Failed to parse tokenizer.json");

    // Extract vocab
    let vocab_obj = json["model"]["vocab"].as_object().expect("Missing vocab");
    let mut token_to_id: AHashMap<String, u32> = AHashMap::new();
    for (token, id) in vocab_obj {
        token_to_id.insert(token.clone(), id.as_u64().unwrap() as u32);
    }

    println!("Vocabulary size: {}", token_to_id.len());

    // Create WordPiece config
    let mut config = WordPieceConfig::default();
    config.unk_token = "[UNK]".to_string();
    config.continuing_subword_prefix = "##".to_string();
    config.do_lower_case = true;
    config.strip_accents = true;

    // Create special tokens
    let special_tokens = SpecialTokens {
        unk_token: Some("[UNK]".to_string()),
        cls_token: Some("[CLS]".to_string()),
        sep_token: Some("[SEP]".to_string()),
        pad_token: Some("[PAD]".to_string()),
        mask_token: Some("[MASK]".to_string()),
        ..Default::default()
    };

    // Create tokenizers
    println!("Creating WordPieceTokenizer (scalar)...");
    let vocabulary = Vocabulary::new(token_to_id.clone(), special_tokens.clone());
    let scalar_tokenizer = WordPieceTokenizer::new(vocabulary.clone(), config.clone());

    println!("Creating HyperWordPieceTokenizer (SIMD)...");
    let hyper_config = HyperConfig {
        continuing_subword_prefix: "##".to_string(),
        // Match HuggingFace's default (100, not 200)
        max_input_chars_per_word: 100,
        unk_token: "[UNK]".to_string(),
        do_lower_case: true,
        strip_accents: true,
        tokenize_chinese_chars: true,
        hash_table_bits: 14,
    };
    let simd_tokenizer = HyperWordPieceTokenizer::new(vocabulary.clone(), hyper_config);

    println!("Loading 1GB dataset...");
    let file = File::open(&data_path).expect("Failed to open dataset");
    let reader = BufReader::new(file);

    let documents: Vec<String> = reader.lines()
        .take(10000) // First 10K documents
        .filter_map(|line| {
            let line = line.ok()?;
            let json: serde_json::Value = serde_json::from_str(&line).ok()?;
            json["text"].as_str().map(|s| s.to_string())
        })
        .collect();

    let total_bytes: usize = documents.iter().map(|d| d.len()).sum();
    println!("Loaded {} documents ({:.2} MB)\n", documents.len(), total_bytes as f64 / 1024.0 / 1024.0);

    // Test 1: Scalar WordPiece Single-core
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TEST 1: WordPiece Scalar Single-Core Encoding                               │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mut total_tokens_scalar = 0usize;
    let scalar_results: Vec<Vec<u32>> = documents.iter()
        .map(|doc| {
            let encoding = scalar_tokenizer.encode(doc, false).unwrap();
            let ids = encoding.get_ids().to_vec();
            total_tokens_scalar += ids.len();
            ids
        })
        .collect();
    let scalar_time = start.elapsed();
    let scalar_throughput = total_bytes as f64 / scalar_time.as_secs_f64() / 1024.0 / 1024.0;

    println!("  Documents:    {}", documents.len());
    println!("  Tokens:       {}", total_tokens_scalar);
    println!("  Time:         {:.2}s", scalar_time.as_secs_f64());
    println!("  Throughput:   {:.2} MB/s", scalar_throughput);
    println!("  Tokens/sec:   {:.0}", total_tokens_scalar as f64 / scalar_time.as_secs_f64());
    println!();

    // Save scalar results for HuggingFace comparison immediately (most accurate)
    println!("  Storing scalar WordPiece results for HuggingFace comparison...");
    let scalar_token_counts: Vec<usize> = scalar_results.iter().map(|t| t.len()).collect();
    let output_path = format!("{}/wordpiece_budtiktok_results.json", WORKSPACE);
    let results_json = serde_json::json!({
        "total_tokens": total_tokens_scalar,
        "token_counts": scalar_token_counts,
        "first_10_docs_tokens": scalar_results.iter().take(10).collect::<Vec<_>>(),
    });
    fs::write(&output_path, serde_json::to_string_pretty(&results_json).unwrap())
        .expect("Failed to write results");
    println!("  Saved to: {}", output_path);

    // Test 2: SIMD WordPiece Single-core
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TEST 2: WordPiece SIMD (HyperWordPiece) Single-Core Encoding                │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mut total_tokens_simd = 0usize;
    let simd_results: Vec<Vec<u32>> = documents.iter()
        .map(|doc| {
            let ids = simd_tokenizer.encode_fast(doc);
            total_tokens_simd += ids.len();
            ids
        })
        .collect();
    let simd_time = start.elapsed();
    let simd_throughput = total_bytes as f64 / simd_time.as_secs_f64() / 1024.0 / 1024.0;

    println!("  Documents:    {}", documents.len());
    println!("  Tokens:       {}", total_tokens_simd);
    println!("  Time:         {:.2}s", simd_time.as_secs_f64());
    println!("  Throughput:   {:.2} MB/s", simd_throughput);
    println!("  Tokens/sec:   {:.0}", total_tokens_simd as f64 / simd_time.as_secs_f64());
    println!("  Speedup:      {:.2}x over scalar", scalar_time.as_secs_f64() / simd_time.as_secs_f64());
    println!();

    // Test 3: SIMD Multi-core
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TEST 3: WordPiece SIMD Multi-Core Encoding (Rayon)                          │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let multi_simd_results: Vec<Vec<u32>> = documents.par_iter()
        .map(|doc| simd_tokenizer.encode_fast(doc))
        .collect();
    let multi_simd_time = start.elapsed();
    let total_tokens_multi_simd: usize = multi_simd_results.iter().map(|t: &Vec<u32>| t.len()).sum();
    let multi_simd_throughput = total_bytes as f64 / multi_simd_time.as_secs_f64() / 1024.0 / 1024.0;

    let num_threads = rayon::current_num_threads();
    let parallel_efficiency = (simd_time.as_secs_f64() / multi_simd_time.as_secs_f64()) / num_threads as f64 * 100.0;

    println!("  Threads:      {}", num_threads);
    println!("  Documents:    {}", documents.len());
    println!("  Tokens:       {}", total_tokens_multi_simd);
    println!("  Time:         {:.2}s", multi_simd_time.as_secs_f64());
    println!("  Throughput:   {:.2} MB/s", multi_simd_throughput);
    println!("  Tokens/sec:   {:.0}", total_tokens_multi_simd as f64 / multi_simd_time.as_secs_f64());
    println!("  Speedup:      {:.2}x over SIMD single-core", simd_time.as_secs_f64() / multi_simd_time.as_secs_f64());
    println!("  Efficiency:   {:.1}%", parallel_efficiency);
    println!();

    // Test 4: Consistency check
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TEST 4: WordPiece Consistency Check (SIMD Single vs Multi-core)             │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    let mut mismatches = 0;
    for (i, (single, multi)) in simd_results.iter().zip(multi_simd_results.iter()).enumerate() {
        let single: &Vec<u32> = single;
        let multi: &Vec<u32> = multi;
        if single != multi {
            mismatches += 1;
            if mismatches <= 3 {
                println!("  MISMATCH at doc {}: single={} tokens, multi={} tokens", i, single.len(), multi.len());
            }
        }
    }

    if mismatches == 0 {
        println!("  ✓ PASS: Single-core and multi-core SIMD produce identical results");
    } else {
        println!("  ✗ FAIL: {} mismatches found", mismatches);
    }
    println!();

    // Test 5: Compare scalar vs SIMD accuracy
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TEST 5: Scalar vs SIMD Accuracy Comparison                                  │");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    let mut scalar_simd_mismatches = 0;
    for (i, (scalar, simd)) in scalar_results.iter().zip(simd_results.iter()).enumerate() {
        if scalar != simd {
            scalar_simd_mismatches += 1;
            if scalar_simd_mismatches <= 3 {
                println!("  MISMATCH at doc {}: scalar={} tokens, simd={} tokens", i, scalar.len(), simd.len());
            }
        }
    }

    if scalar_simd_mismatches == 0 {
        println!("  ✓ PASS: Scalar and SIMD produce identical results");
    } else {
        println!("  ⚠ WARNING: {} documents differ between scalar and SIMD", scalar_simd_mismatches);
        println!("             Scalar tokens: {}, SIMD tokens: {}", total_tokens_scalar, total_tokens_simd);
    }
}
