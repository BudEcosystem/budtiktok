//! Unigram Tokenizer Benchmark
//!
//! Compares BudTikTok Unigram with HuggingFace Tokenizers
//!
//! Run with: cargo run --example unigram_benchmark --release

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use rayon::prelude::*;
use tokenizers::Tokenizer as HfTokenizer;

use budtiktok_core::{load_tokenizer, Tokenizer};

const WORKSPACE: &str = "/home/bud/Desktop/latentbud/budtiktok";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    UNIGRAM TOKENIZER BENCHMARK                                   ║");
    println!("║                  BudTikTok vs HuggingFace Tokenizers                             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

    let num_cpus = num_cpus::get();
    println!("System: {} CPUs\n", num_cpus);

    let xlnet_path = format!("{}/benchmark_data/xlnet/tokenizer.json", WORKSPACE);

    // =========================================================================
    // Load HuggingFace XLNet tokenizer
    // =========================================================================
    println!("Loading HuggingFace XLNet tokenizer...");
    let start = Instant::now();
    let hf_tokenizer = HfTokenizer::from_file(&xlnet_path)
        .expect("Failed to load HuggingFace tokenizer");
    let hf_load_time = start.elapsed();
    println!("  HuggingFace load time: {:.2}ms", hf_load_time.as_secs_f64() * 1000.0);

    let vocab_size = hf_tokenizer.get_vocab_size(true);
    println!("  Vocabulary size: {}", vocab_size);

    // =========================================================================
    // Load BudTikTok Unigram tokenizer
    // =========================================================================
    println!("\nLoading BudTikTok Unigram tokenizer...");
    let start = Instant::now();
    let bud_tokenizer = load_tokenizer(&xlnet_path)
        .expect("Failed to load BudTikTok tokenizer");
    let bud_load_time = start.elapsed();
    println!("  BudTikTok load time: {:.2}ms", bud_load_time.as_secs_f64() * 1000.0);
    println!("  Vocabulary size: {}", bud_tokenizer.vocab_size());
    println!("  Load speedup: {:.2}x", hf_load_time.as_secs_f64() / bud_load_time.as_secs_f64());

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
        let _ = bud_tokenizer.encode(doc, false);
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
    // Benchmark: BudTikTok Single-Core
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ BudTikTok Unigram - Single-Core                                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mut bud_tokens_single = 0usize;
    let bud_results_single: Vec<Vec<u32>> = documents.iter()
        .map(|doc| {
            let encoding = bud_tokenizer.encode(doc, false).unwrap();
            let ids: Vec<u32> = encoding.get_ids().to_vec();
            bud_tokens_single += ids.len();
            ids
        })
        .collect();
    let bud_time_single = start.elapsed();
    let bud_throughput_single = total_bytes as f64 / bud_time_single.as_secs_f64() / 1024.0 / 1024.0;

    println!("  Time:         {:.3}s", bud_time_single.as_secs_f64());
    println!("  Throughput:   {:.1} MB/s", bud_throughput_single);
    println!("  Tokens:       {}", bud_tokens_single);
    println!("  Tokens/doc:   {:.1}", bud_tokens_single as f64 / documents.len() as f64);
    println!("  vs HF:        {:.2}x", bud_throughput_single / hf_throughput_single);

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
    // Benchmark: BudTikTok Multi-Core
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ BudTikTok Unigram - Multi-Core ({} threads)                               │", num_cpus);
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let bud_results_multi: Vec<Vec<u32>> = documents.par_iter()
        .map(|doc| {
            let encoding = bud_tokenizer.encode(doc, false).unwrap();
            encoding.get_ids().to_vec()
        })
        .collect();
    let bud_time_multi = start.elapsed();

    let bud_tokens_multi: usize = bud_results_multi.iter().map(|v| v.len()).sum();
    let bud_throughput_multi = total_bytes as f64 / bud_time_multi.as_secs_f64() / 1024.0 / 1024.0;

    println!("  Time:         {:.3}s", bud_time_multi.as_secs_f64());
    println!("  Throughput:   {:.1} MB/s", bud_throughput_multi);
    println!("  Tokens:       {}", bud_tokens_multi);
    println!("  Speedup:      {:.2}x over single-core", bud_time_single.as_secs_f64() / bud_time_multi.as_secs_f64());
    let bud_efficiency = (bud_time_single.as_secs_f64() / bud_time_multi.as_secs_f64()) / num_cpus as f64 * 100.0;
    println!("  Efficiency:   {:.1}%", bud_efficiency);
    println!("  vs HF Multi:  {:.2}x", bud_throughput_multi / hf_throughput_multi);

    // =========================================================================
    // Accuracy Comparison (HF as gold standard)
    // =========================================================================
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Accuracy Comparison (HuggingFace as Gold Standard)                              │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let mut exact_matches = 0;
    let mut token_count_matches = 0;
    let mut total_hf_tokens = 0;
    let mut total_bud_tokens = 0;

    for i in 0..documents.len().min(1000) {
        let hf_ids = &hf_results_single[i];
        let bud_ids = &bud_results_single[i];

        total_hf_tokens += hf_ids.len();
        total_bud_tokens += bud_ids.len();

        if hf_ids == bud_ids {
            exact_matches += 1;
        }
        if hf_ids.len() == bud_ids.len() {
            token_count_matches += 1;
        }
    }

    let sample_size = documents.len().min(1000);
    println!("  Sample size:          {} documents", sample_size);
    println!("  Exact matches:        {} ({:.1}%)", exact_matches, exact_matches as f64 / sample_size as f64 * 100.0);
    println!("  Token count matches:  {} ({:.1}%)", token_count_matches, token_count_matches as f64 / sample_size as f64 * 100.0);
    println!("  HF avg tokens/doc:    {:.1}", total_hf_tokens as f64 / sample_size as f64);
    println!("  Bud avg tokens/doc:   {:.1}", total_bud_tokens as f64 / sample_size as f64);
    println!("  Token ratio:          {:.3}", total_bud_tokens as f64 / total_hf_tokens as f64);

    // Show sample differences
    println!("\n  Sample tokenization differences:");
    let mut shown = 0;
    for i in 0..documents.len().min(100) {
        if hf_results_single[i] != bud_results_single[i] && shown < 3 {
            let doc = &documents[i];
            let preview: String = doc.chars().take(50).collect();
            println!("\n    Document {}: \"{}...\"", i, preview);
            println!("      HF:  {} tokens - {:?}", hf_results_single[i].len(), &hf_results_single[i][..hf_results_single[i].len().min(10)]);
            println!("      Bud: {} tokens - {:?}", bud_results_single[i].len(), &bud_results_single[i][..bud_results_single[i].len().min(10)]);
            shown += 1;
        }
    }

    // Token Statistics
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Token Statistics                                                                │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let chars_per_token_hf = documents.iter()
        .map(|d| d.chars().count())
        .sum::<usize>() as f64 / hf_tokens_single as f64;

    let chars_per_token_bud = documents.iter()
        .map(|d| d.chars().count())
        .sum::<usize>() as f64 / bud_tokens_single as f64;

    let bytes_per_token_hf = total_bytes as f64 / hf_tokens_single as f64;
    let bytes_per_token_bud = total_bytes as f64 / bud_tokens_single as f64;

    println!("  HuggingFace:");
    println!("    Total tokens:     {}", hf_tokens_single);
    println!("    Chars/token:      {:.2}", chars_per_token_hf);
    println!("    Bytes/token:      {:.2}", bytes_per_token_hf);
    println!("  BudTikTok:");
    println!("    Total tokens:     {}", bud_tokens_single);
    println!("    Chars/token:      {:.2}", chars_per_token_bud);
    println!("    Bytes/token:      {:.2}", bytes_per_token_bud);

    // Sample tokenization
    println!("\n  Sample tokenizations:");
    let samples = ["Hello, world!", "The quick brown fox jumps over the lazy dog.", "Machine learning is fascinating."];

    for sample in &samples {
        let hf_encoding = hf_tokenizer.encode(*sample, false).unwrap();
        let hf_tokens: Vec<String> = hf_encoding.get_tokens().iter().map(|s| s.to_string()).collect();

        let bud_encoding = bud_tokenizer.encode(*sample, false).unwrap();
        let bud_tokens: Vec<String> = bud_encoding.get_tokens().iter().cloned().collect();

        println!("    \"{}\"", sample);
        println!("      HF:  {:?}", &hf_tokens[..hf_tokens.len().min(15)]);
        println!("      Bud: {:?}", &bud_tokens[..bud_tokens.len().min(15)]);
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              SUMMARY                                             ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Tokenizer            │ Single-Core │ Multi-Core │ Speedup  │ Efficiency         ║");
    println!("╠──────────────────────┼─────────────┼────────────┼──────────┼────────────────────╣");
    println!("║ HuggingFace          │ {:>7.1} MB/s │ {:>7.1} MB/s │ {:>6.2}x  │    {:>5.1}%           ║",
             hf_throughput_single,
             hf_throughput_multi,
             hf_time_single.as_secs_f64() / hf_time_multi.as_secs_f64(),
             hf_efficiency);
    println!("║ BudTikTok            │ {:>7.1} MB/s │ {:>7.1} MB/s │ {:>6.2}x  │    {:>5.1}%           ║",
             bud_throughput_single,
             bud_throughput_multi,
             bud_time_single.as_secs_f64() / bud_time_multi.as_secs_f64(),
             bud_efficiency);
    println!("╠──────────────────────┴─────────────┴────────────┴──────────┴────────────────────╣");
    println!("║ BudTikTok vs HF:     │ {:>6.2}x faster (single) │ {:>6.2}x faster (multi)       ║",
             bud_throughput_single / hf_throughput_single,
             bud_throughput_multi / hf_throughput_multi);
    println!("║ Accuracy:            │ {:.1}% exact match                                        ║",
             exact_matches as f64 / sample_size as f64 * 100.0);
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝");

    if bud_throughput_single < hf_throughput_single {
        println!("\n⚠ BudTikTok is slower than HuggingFace - investigation needed!");
    } else {
        println!("\n✓ BudTikTok is faster than HuggingFace!");
    }

    if (exact_matches as f64 / sample_size as f64) < 0.95 {
        println!("⚠ Accuracy is below 95% - tokenization differs from HuggingFace");
        println!("  Note: This is expected due to different Viterbi implementations and tie-breaking");
    }
}
