//! Benchmark comparing SIMD BPE vs Standard BPE
//!
//! Run with: cargo run --example bpe_simd_benchmark --release
//!
//! This benchmark compares:
//! 1. OptimizedBpeEncoder (current production)
//! 2. SimdBpeEncoder greedy mode (new SIMD implementation)
//! 3. SimdBpeEncoder minimal mode (wavefront DP)

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::time::Instant;
use ahash::AHashMap;

use budtiktok_core::bpe_linear::OptimizedBpeEncoder;
use budtiktok_core::bpe_simd::SimdBpeEncoder;

const WORKSPACE: &str = "/home/bud/Desktop/latentbud/budtiktok";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║              BPE SIMD BENCHMARK: Standard vs SIMD Implementation                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

    // Load GPT-2 vocabulary
    let vocab_path = format!("{}/benchmark_data/gpt2/vocab.json", WORKSPACE);
    let merges_path = format!("{}/benchmark_data/gpt2/merges.txt", WORKSPACE);
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", WORKSPACE);

    println!("Loading GPT-2 vocabulary...");
    let vocab_content = fs::read_to_string(&vocab_path).expect("Failed to read vocab.json");
    let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_content).expect("Failed to parse vocab");
    let vocab_ahash: AHashMap<String, u32> = vocab.iter().map(|(k, v)| (k.clone(), *v)).collect();
    println!("  Vocabulary size: {}", vocab.len());

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
    println!("  Merge rules: {}", merges.len());

    println!("\nCreating encoders...");

    // Standard encoder (production)
    let start = Instant::now();
    let standard_encoder = OptimizedBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<|endoftext|>");
    let standard_build_time = start.elapsed();
    println!("  OptimizedBpeEncoder built in {:.2}ms", standard_build_time.as_secs_f64() * 1000.0);

    // SIMD encoder (experimental)
    let start = Instant::now();
    let simd_encoder = SimdBpeEncoder::new(vocab_ahash, merges, "<|endoftext|>");
    let simd_build_time = start.elapsed();
    println!("  SimdBpeEncoder built in {:.2}ms", simd_build_time.as_secs_f64() * 1000.0);

    // Load test data
    println!("\nLoading test data...");
    let file = File::open(&data_path).expect("Failed to open dataset");
    let reader = BufReader::new(file);

    let documents: Vec<String> = reader.lines()
        .take(1000) // First 1000 documents for quick benchmark
        .filter_map(|line| {
            let line = line.ok()?;
            let json: serde_json::Value = serde_json::from_str(&line).ok()?;
            json["text"].as_str().map(|s| s.to_string())
        })
        .collect();

    let total_bytes: usize = documents.iter().map(|d| d.len()).sum();
    let avg_doc_len = total_bytes as f64 / documents.len() as f64;
    println!("  Documents: {}", documents.len());
    println!("  Total bytes: {} ({:.2} MB)", total_bytes, total_bytes as f64 / 1024.0 / 1024.0);
    println!("  Average document length: {:.1} bytes", avg_doc_len);

    // Warmup
    println!("\nWarming up...");
    for doc in documents.iter().take(10) {
        let _ = standard_encoder.encode(doc);
        let _ = simd_encoder.encode_greedy(doc);
    }

    // =========================================================================
    // Benchmark 1: Standard BPE
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Benchmark 1: OptimizedBpeEncoder (Standard BPE)                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mut standard_tokens = 0usize;
    let standard_results: Vec<Vec<u32>> = documents.iter()
        .map(|doc| {
            let tokens = standard_encoder.encode(doc);
            standard_tokens += tokens.len();
            tokens
        })
        .collect();
    let standard_time = start.elapsed();
    let standard_throughput = total_bytes as f64 / standard_time.as_secs_f64() / 1024.0 / 1024.0;

    println!("  Time:         {:.3}s", standard_time.as_secs_f64());
    println!("  Throughput:   {:.1} MB/s", standard_throughput);
    println!("  Tokens:       {}", standard_tokens);
    println!("  Tokens/doc:   {:.1}", standard_tokens as f64 / documents.len() as f64);

    // =========================================================================
    // Benchmark 2: SIMD BPE GPT-2 Compatible (with pre-tokenization)
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Benchmark 2: SimdBpeEncoder GPT-2 Mode (Pre-tokenize + Greedy)                  │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mut gpt2_tokens = 0usize;
    let gpt2_results: Vec<Vec<u32>> = documents.iter()
        .map(|doc| {
            let tokens = simd_encoder.encode_gpt2(doc);
            gpt2_tokens += tokens.len();
            tokens
        })
        .collect();
    let gpt2_time = start.elapsed();
    let gpt2_throughput = total_bytes as f64 / gpt2_time.as_secs_f64() / 1024.0 / 1024.0;

    println!("  Time:         {:.3}s", gpt2_time.as_secs_f64());
    println!("  Throughput:   {:.1} MB/s", gpt2_throughput);
    println!("  Tokens:       {}", gpt2_tokens);
    println!("  Tokens/doc:   {:.1}", gpt2_tokens as f64 / documents.len() as f64);
    println!("  Speedup:      {:.2}x vs standard", standard_time.as_secs_f64() / gpt2_time.as_secs_f64());

    // =========================================================================
    // Benchmark 3: SIMD BPE Compatible (proper merge ordering)
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Benchmark 3: SimdBpeEncoder BPE Mode (100% Compatible with Standard)            │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mut bpe_tokens = 0usize;
    let bpe_results: Vec<Vec<u32>> = documents.iter()
        .map(|doc| {
            let tokens = simd_encoder.encode_bpe(doc);
            bpe_tokens += tokens.len();
            tokens
        })
        .collect();
    let bpe_time = start.elapsed();
    let bpe_throughput = total_bytes as f64 / bpe_time.as_secs_f64() / 1024.0 / 1024.0;

    println!("  Time:         {:.3}s", bpe_time.as_secs_f64());
    println!("  Throughput:   {:.1} MB/s", bpe_throughput);
    println!("  Tokens:       {}", bpe_tokens);
    println!("  Tokens/doc:   {:.1}", bpe_tokens as f64 / documents.len() as f64);
    println!("  Speedup:      {:.2}x vs standard", standard_time.as_secs_f64() / bpe_time.as_secs_f64());

    // =========================================================================
    // Benchmark 4: SIMD BPE Greedy Raw (no pre-tokenization)
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Benchmark 4: SimdBpeEncoder Greedy Raw (No Pre-tokenization)                    │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mut greedy_tokens = 0usize;
    let greedy_results: Vec<Vec<u32>> = documents.iter()
        .map(|doc| {
            let tokens = simd_encoder.encode_greedy(doc);
            greedy_tokens += tokens.len();
            tokens
        })
        .collect();
    let greedy_time = start.elapsed();
    let greedy_throughput = total_bytes as f64 / greedy_time.as_secs_f64() / 1024.0 / 1024.0;

    println!("  Time:         {:.3}s", greedy_time.as_secs_f64());
    println!("  Throughput:   {:.1} MB/s", greedy_throughput);
    println!("  Tokens:       {}", greedy_tokens);
    println!("  Tokens/doc:   {:.1}", greedy_tokens as f64 / documents.len() as f64);
    println!("  Speedup:      {:.2}x vs standard", standard_time.as_secs_f64() / greedy_time.as_secs_f64());

    // =========================================================================
    // Benchmark 5: SIMD BPE Minimal (Wavefront DP) - Skip if too slow
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Benchmark 5: SimdBpeEncoder Minimal (Wavefront DP) - First 100 docs only       │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    // Only run on first 100 docs since it's slow
    let minimal_docs = &documents[..100.min(documents.len())];
    let minimal_bytes: usize = minimal_docs.iter().map(|d| d.len()).sum();

    let start = Instant::now();
    let mut minimal_tokens = 0usize;
    let _minimal_results: Vec<Vec<u32>> = minimal_docs.iter()
        .map(|doc| {
            let tokens = simd_encoder.encode_minimal(doc);
            minimal_tokens += tokens.len();
            tokens
        })
        .collect();
    let minimal_time = start.elapsed();
    let minimal_throughput = minimal_bytes as f64 / minimal_time.as_secs_f64() / 1024.0 / 1024.0;

    println!("  Docs tested:  {}", minimal_docs.len());
    println!("  Time:         {:.3}s", minimal_time.as_secs_f64());
    println!("  Throughput:   {:.1} MB/s", minimal_throughput);
    println!("  Tokens:       {}", minimal_tokens);

    // =========================================================================
    // Accuracy Comparison
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Accuracy Comparison (BPE Mode should be 100% compatible)                        │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    // Compare token counts
    let token_diff_bpe = (bpe_tokens as i64 - standard_tokens as i64).abs();
    let token_diff_gpt2 = (gpt2_tokens as i64 - standard_tokens as i64).abs();
    let token_diff_greedy = (greedy_tokens as i64 - standard_tokens as i64).abs();

    println!("  Token count differences (vs standard BPE):");
    println!("    BPE Mode:   {:+} tokens ({:.2}% difference)",
             bpe_tokens as i64 - standard_tokens as i64,
             token_diff_bpe as f64 / standard_tokens as f64 * 100.0);
    println!("    GPT-2 Mode: {:+} tokens ({:.2}% difference)",
             gpt2_tokens as i64 - standard_tokens as i64,
             token_diff_gpt2 as f64 / standard_tokens as f64 * 100.0);
    println!("    Greedy Raw: {:+} tokens ({:.2}% difference)",
             greedy_tokens as i64 - standard_tokens as i64,
             token_diff_greedy as f64 / standard_tokens as f64 * 100.0);

    // Check exact matches
    let mut exact_match_bpe = 0;
    let mut exact_match_gpt2 = 0;
    let mut exact_match_greedy = 0;

    for i in 0..documents.len() {
        if standard_results[i] == bpe_results[i] {
            exact_match_bpe += 1;
        }
        if standard_results[i] == gpt2_results[i] {
            exact_match_gpt2 += 1;
        }
        if standard_results[i] == greedy_results[i] {
            exact_match_greedy += 1;
        }
    }

    println!("\n  Exact match rate (same tokenization):");
    println!("    BPE Mode:   {}/{} ({:.1}%)",
             exact_match_bpe, documents.len(),
             exact_match_bpe as f64 / documents.len() as f64 * 100.0);
    println!("    GPT-2 Mode: {}/{} ({:.1}%)",
             exact_match_gpt2, documents.len(),
             exact_match_gpt2 as f64 / documents.len() as f64 * 100.0);
    println!("    Greedy Raw: {}/{} ({:.1}%)",
             exact_match_greedy, documents.len(),
             exact_match_greedy as f64 / documents.len() as f64 * 100.0);

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                    SUMMARY                                       ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Encoder              │ Throughput │ Speedup │ Token Diff │ Exact Match          ║");
    println!("╠──────────────────────┼────────────┼─────────┼────────────┼──────────────────────╣");
    println!("║ Standard BPE         │ {:>7.1} MB/s │   1.00x │      0     │    100.0%            ║",
             standard_throughput);
    println!("║ SIMD BPE Mode        │ {:>7.1} MB/s │  {:>5.2}x │ {:>+9} │    {:>5.1}%            ║",
             bpe_throughput,
             standard_time.as_secs_f64() / bpe_time.as_secs_f64(),
             bpe_tokens as i64 - standard_tokens as i64,
             exact_match_bpe as f64 / documents.len() as f64 * 100.0);
    println!("║ SIMD GPT-2 Mode      │ {:>7.1} MB/s │  {:>5.2}x │ {:>+9} │    {:>5.1}%            ║",
             gpt2_throughput,
             standard_time.as_secs_f64() / gpt2_time.as_secs_f64(),
             gpt2_tokens as i64 - standard_tokens as i64,
             exact_match_gpt2 as f64 / documents.len() as f64 * 100.0);
    println!("║ SIMD Greedy Raw      │ {:>7.1} MB/s │  {:>5.2}x │ {:>+9} │    {:>5.1}%            ║",
             greedy_throughput,
             standard_time.as_secs_f64() / greedy_time.as_secs_f64(),
             greedy_tokens as i64 - standard_tokens as i64,
             exact_match_greedy as f64 / documents.len() as f64 * 100.0);
    println!("║ SIMD Minimal (DP)    │ {:>7.1} MB/s │   N/A   │    N/A     │    N/A               ║",
             minimal_throughput);
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝");

    // Show sample differences for GPT-2 mode
    println!("\nSample tokenization differences GPT-2 Mode vs Standard (first 3 non-matching):");
    let mut shown = 0;
    for i in 0..documents.len().min(100) {
        if standard_results[i] != gpt2_results[i] && shown < 3 {
            println!("\n  Doc {} (first 50 chars): \"{}\"", i,
                     &documents[i][..documents[i].len().min(50)]);
            println!("    Standard: {} tokens, first 5: {:?}",
                     standard_results[i].len(),
                     &standard_results[i][..standard_results[i].len().min(5)]);
            println!("    GPT-2:    {} tokens, first 5: {:?}",
                     gpt2_results[i].len(),
                     &gpt2_results[i][..gpt2_results[i].len().min(5)]);
            shown += 1;
        }
    }

    if shown == 0 {
        println!("\n  All documents match exactly between Standard and GPT-2 Mode!");
    }
}
