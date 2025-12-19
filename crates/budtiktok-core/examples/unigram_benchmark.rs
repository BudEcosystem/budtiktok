//! Unigram Tokenizer Benchmark
//!
//! Compares BudTikTok Unigram with HuggingFace Tokenizers
//!
//! Run with: cargo run --example unigram_benchmark --release

use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::time::Instant;

use rayon::prelude::*;
use tokenizers::Tokenizer as HfTokenizer;

const WORKSPACE: &str = "/home/bud/Desktop/latentbud/budtiktok";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    UNIGRAM TOKENIZER BENCHMARK                                   ║");
    println!("║          BudTikTok vs HuggingFace Tokenizers (XLNet/SentencePiece)               ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

    let num_cpus = num_cpus::get();
    println!("System: {} CPUs\n", num_cpus);

    // =========================================================================
    // Load XLNet tokenizer (HuggingFace)
    // =========================================================================
    let xlnet_path = format!("{}/benchmark_data/xlnet/tokenizer.json", WORKSPACE);

    println!("Loading HuggingFace XLNet tokenizer...");
    let start = Instant::now();
    let hf_tokenizer = HfTokenizer::from_file(&xlnet_path)
        .expect("Failed to load HuggingFace tokenizer");
    let hf_load_time = start.elapsed();
    println!("  HuggingFace load time: {:.2}ms", hf_load_time.as_secs_f64() * 1000.0);

    // Get vocabulary info
    let vocab_size = hf_tokenizer.get_vocab_size(true);
    println!("  Vocabulary size: {}", vocab_size);

    // =========================================================================
    // Load test data
    // =========================================================================
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", WORKSPACE);

    println!("\nLoading test data...");
    let file = File::open(&data_path).expect("Failed to open dataset");
    let reader = BufReader::new(file);

    let documents: Vec<String> = reader.lines()
        .take(10000)  // 10K documents for benchmark
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
        let _ = hf_tokenizer.encode(doc.as_str(), false);
    }

    // =========================================================================
    // Benchmark: HuggingFace Single-Core
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ HuggingFace Tokenizers - Single-Core                                            │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mut hf_tokens_single = 0usize;
    let hf_results_single: Vec<Vec<u32>> = documents.iter()
        .map(|doc| {
            let encoding = hf_tokenizer.encode(doc.as_str(), false).unwrap();
            let ids: Vec<u32> = encoding.get_ids().to_vec();
            hf_tokens_single += ids.len();
            ids
        })
        .collect();
    let hf_time_single = start.elapsed();
    let hf_throughput_single = total_bytes as f64 / hf_time_single.as_secs_f64() / 1024.0 / 1024.0;

    println!("  Time:         {:.3}s", hf_time_single.as_secs_f64());
    println!("  Throughput:   {:.1} MB/s", hf_throughput_single);
    println!("  Tokens:       {}", hf_tokens_single);
    println!("  Tokens/doc:   {:.1}", hf_tokens_single as f64 / documents.len() as f64);

    // =========================================================================
    // Benchmark: HuggingFace Multi-Core
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ HuggingFace Tokenizers - Multi-Core ({} threads)                          │", num_cpus);
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let hf_results_multi: Vec<Vec<u32>> = documents.par_iter()
        .map(|doc| {
            let encoding = hf_tokenizer.encode(doc.as_str(), false).unwrap();
            encoding.get_ids().to_vec()
        })
        .collect();
    let hf_time_multi = start.elapsed();

    let hf_tokens_multi: usize = hf_results_multi.iter().map(|v| v.len()).sum();
    let hf_throughput_multi = total_bytes as f64 / hf_time_multi.as_secs_f64() / 1024.0 / 1024.0;

    println!("  Time:         {:.3}s", hf_time_multi.as_secs_f64());
    println!("  Throughput:   {:.1} MB/s", hf_throughput_multi);
    println!("  Tokens:       {}", hf_tokens_multi);
    println!("  Speedup:      {:.2}x over single-core", hf_time_single.as_secs_f64() / hf_time_multi.as_secs_f64());
    let hf_efficiency = (hf_time_single.as_secs_f64() / hf_time_multi.as_secs_f64()) / num_cpus as f64 * 100.0;
    println!("  Efficiency:   {:.1}%", hf_efficiency);

    // =========================================================================
    // Verify consistency
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Consistency Check                                                               │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let mut mismatches = 0;
    for i in 0..documents.len() {
        if hf_results_single[i] != hf_results_multi[i] {
            mismatches += 1;
            if mismatches <= 3 {
                println!("  MISMATCH at doc {}: single={} tokens, multi={} tokens",
                    i, hf_results_single[i].len(), hf_results_multi[i].len());
            }
        }
    }

    if mismatches == 0 {
        println!("  ✓ PASS: Single-core and multi-core produce identical results");
    } else {
        println!("  ✗ FAIL: {} documents differ between single and multi-core", mismatches);
    }

    // =========================================================================
    // Token Statistics
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Token Statistics                                                                │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let chars_per_token = documents.iter()
        .map(|d| d.chars().count())
        .sum::<usize>() as f64 / hf_tokens_single as f64;

    let bytes_per_token = total_bytes as f64 / hf_tokens_single as f64;

    println!("  Total tokens:     {}", hf_tokens_single);
    println!("  Chars/token:      {:.2}", chars_per_token);
    println!("  Bytes/token:      {:.2}", bytes_per_token);
    println!("  Compression:      {:.2}x", bytes_per_token);

    // Sample tokenization
    println!("\n  Sample tokenizations:");
    let samples = ["Hello, world!", "The quick brown fox jumps over the lazy dog.", "Machine learning is fascinating."];

    for sample in &samples {
        let encoding = hf_tokenizer.encode(*sample, false).unwrap();
        let tokens: Vec<String> = encoding.get_tokens().iter().map(|s| s.to_string()).collect();
        println!("    \"{}\"", sample);
        println!("      -> {:?}", &tokens[..tokens.len().min(10)]);
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              SUMMARY                                             ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Configuration          │ Throughput │ Tokens/s     │ Efficiency                 ║");
    println!("╠────────────────────────┼────────────┼──────────────┼────────────────────────────╣");
    println!("║ HF Single-Core         │ {:>7.1} MB/s │ {:>10.0}   │      -                     ║",
             hf_throughput_single,
             hf_tokens_single as f64 / hf_time_single.as_secs_f64());
    println!("║ HF Multi-Core ({:>2} thr) │ {:>7.1} MB/s │ {:>10.0}   │    {:>5.1}%                 ║",
             num_cpus,
             hf_throughput_multi,
             hf_tokens_multi as f64 / hf_time_multi.as_secs_f64(),
             hf_efficiency);
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝");

    println!("\n┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ BudTikTok Unigram Status                                                        │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");
    println!("  NOTE: BudTikTok's Unigram implementation exists (src/unigram.rs) but requires");
    println!("        additional work to load HuggingFace tokenizer.json format with scores.");
    println!("");
    println!("  Current capabilities:");
    println!("    - Viterbi optimal segmentation");
    println!("    - N-best decoding");
    println!("    - Stochastic sampling");
    println!("    - Byte fallback");
    println!("");
    println!("  Missing for full benchmark:");
    println!("    - HuggingFace tokenizer.json Unigram model loader");
    println!("    - Score parsing from vocab list format [[token, score], ...]");
    println!("");
    println!("  HuggingFace Tokenizers (Rust) serves as the gold standard baseline.");
}
