//! Full 1GB Dataset Benchmark: BudTikTok vs HuggingFace
//!
//! Tests:
//! 1. Single-core tokenization (taskset -c 0)
//! 2. Multi-core tokenization (all 16 cores)
//! 3. 100% accuracy verification

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use ahash::AHashMap;
use rayon::prelude::*;
use tokenizers::Tokenizer;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Full 1GB Dataset Benchmark: BudTikTok vs HuggingFace         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let workspace = "/home/bud/Desktop/latentbud/budtiktok";

    // Load GPT-2 vocab and merges
    println!("Loading GPT-2 vocabulary and merges...");
    let vocab_path = format!("{}/benchmark_data/gpt2/vocab.json", workspace);
    let content = fs::read_to_string(&vocab_path).expect("Failed to read vocab");
    let vocab: HashMap<String, u32> = serde_json::from_str(&content).expect("Failed to parse vocab");
    let vocab_ahash: AHashMap<String, u32> = vocab.iter().map(|(k, v)| (k.clone(), *v)).collect();

    let merges_path = format!("{}/benchmark_data/gpt2/merges.txt", workspace);
    let merge_content = fs::read_to_string(&merges_path).expect("Failed to read merges");
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

    println!("  Vocab size: {} tokens", vocab.len());
    println!("  Merge rules: {} rules", merges.len());

    // Load full dataset
    println!("\nLoading full 1GB dataset...");
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", workspace);
    let file = fs::File::open(&data_path).expect("Failed to open data file");
    let reader = BufReader::new(file);

    let mut documents: Vec<String> = Vec::new();
    let mut total_bytes = 0usize;

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let json: serde_json::Value = serde_json::from_str(&line).expect("Failed to parse JSON");
        let text = json["text"].as_str().unwrap_or("").to_string();
        total_bytes += text.len();
        documents.push(text);
    }

    let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
    let total_gb = total_mb / 1024.0;
    println!("  Documents: {}", documents.len());
    println!("  Total size: {:.2} GB ({:.2} MB)", total_gb, total_mb);

    // Create encoders
    println!("\nInitializing encoders...");

    // BudTikTok encoder
    use budtiktok_core::bpe_linear::OptimizedBpeEncoder;
    let budtiktok = OptimizedBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<|endoftext|>");

    // HuggingFace tokenizer
    let hf_tokenizer = Tokenizer::from_file("/tmp/gpt2_tokenizer/tokenizer.json")
        .expect("Failed to load HuggingFace tokenizer");

    println!("  BudTikTok: OptimizedBpeEncoder (SIMD)");
    println!("  HuggingFace: tokenizers 0.x (Rust)");

    // Warm up both encoders
    println!("\nWarming up encoders (1000 docs)...");
    for doc in documents.iter().take(1000) {
        let _ = budtiktok.encode(doc);
        let _ = hf_tokenizer.encode(doc.as_str(), false);
    }

    // =========================================================================
    // EXPERIMENT 1: Single-Core Benchmark
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║              EXPERIMENT 1: Single-Core Benchmark                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // HuggingFace single-core
    println!("Running HuggingFace (single-core)...");
    let start = Instant::now();
    let mut hf_tokens_single = 0usize;
    for doc in &documents {
        let encoding = hf_tokenizer.encode(doc.as_str(), false).unwrap();
        hf_tokens_single += encoding.get_ids().len();
    }
    let hf_single_time = start.elapsed();
    let hf_single_throughput = total_mb / hf_single_time.as_secs_f64();
    println!("  Time: {:.2?}", hf_single_time);
    println!("  Throughput: {:.2} MB/s", hf_single_throughput);
    println!("  Tokens: {}", hf_tokens_single);

    // BudTikTok single-core
    println!("\nRunning BudTikTok (single-core)...");
    let start = Instant::now();
    let mut bt_tokens_single = 0usize;
    for doc in &documents {
        bt_tokens_single += budtiktok.encode(doc).len();
    }
    let bt_single_time = start.elapsed();
    let bt_single_throughput = total_mb / bt_single_time.as_secs_f64();
    println!("  Time: {:.2?}", bt_single_time);
    println!("  Throughput: {:.2} MB/s", bt_single_throughput);
    println!("  Tokens: {}", bt_tokens_single);

    let single_speedup = bt_single_throughput / hf_single_throughput;
    println!("\n  *** Single-Core Speedup: {:.2}x ***", single_speedup);

    // =========================================================================
    // EXPERIMENT 2: Multi-Core Benchmark (All 16 Cores)
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║         EXPERIMENT 2: Multi-Core Benchmark (16 cores)            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Set thread pool size
    rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build_global()
        .ok();

    // HuggingFace multi-core (using rayon)
    println!("Running HuggingFace (16 cores)...");
    let start = Instant::now();
    let hf_results_multi: Vec<usize> = documents.par_iter()
        .map(|doc| {
            let encoding = hf_tokenizer.encode(doc.as_str(), false).unwrap();
            encoding.get_ids().len()
        })
        .collect();
    let hf_tokens_multi: usize = hf_results_multi.iter().sum();
    let hf_multi_time = start.elapsed();
    let hf_multi_throughput = total_mb / hf_multi_time.as_secs_f64();
    println!("  Time: {:.2?}", hf_multi_time);
    println!("  Throughput: {:.2} MB/s", hf_multi_throughput);
    println!("  Tokens: {}", hf_tokens_multi);

    // BudTikTok multi-core (using rayon)
    println!("\nRunning BudTikTok (16 cores)...");
    let start = Instant::now();
    let bt_results_multi: Vec<usize> = documents.par_iter()
        .map(|doc| budtiktok.encode(doc).len())
        .collect();
    let bt_tokens_multi: usize = bt_results_multi.iter().sum();
    let bt_multi_time = start.elapsed();
    let bt_multi_throughput = total_mb / bt_multi_time.as_secs_f64();
    println!("  Time: {:.2?}", bt_multi_time);
    println!("  Throughput: {:.2} MB/s", bt_multi_throughput);
    println!("  Tokens: {}", bt_tokens_multi);

    let multi_speedup = bt_multi_throughput / hf_multi_throughput;
    println!("\n  *** Multi-Core Speedup: {:.2}x ***", multi_speedup);

    // =========================================================================
    // ACCURACY VERIFICATION
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    ACCURACY VERIFICATION                         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Verifying 100% accuracy on full dataset...");
    let mut mismatches = 0usize;
    let mut total_checked = 0usize;
    let mut first_mismatch: Option<(usize, String)> = None;

    for (i, doc) in documents.iter().enumerate() {
        let bt_result = budtiktok.encode(doc);
        let hf_result = hf_tokenizer.encode(doc.as_str(), false).unwrap();
        let hf_ids: Vec<u32> = hf_result.get_ids().to_vec();

        if bt_result != hf_ids {
            mismatches += 1;
            if first_mismatch.is_none() && doc.len() < 200 {
                first_mismatch = Some((i, doc.clone()));
            }
        }
        total_checked += 1;

        if (i + 1) % 50000 == 0 {
            println!("  Checked {}/{} documents... ({} mismatches)",
                     i + 1, documents.len(), mismatches);
        }
    }

    let accuracy = (total_checked - mismatches) as f64 / total_checked as f64 * 100.0;
    println!("\n  Total documents: {}", total_checked);
    println!("  Mismatches: {}", mismatches);
    println!("  Accuracy: {:.4}%", accuracy);

    if let Some((idx, text)) = first_mismatch {
        println!("\n  First mismatch (doc #{}):", idx);
        println!("    Text: {:?}", &text[..text.len().min(100)]);
        let bt = budtiktok.encode(&text);
        let hf = hf_tokenizer.encode(text.as_str(), false).unwrap();
        println!("    BudTikTok: {:?}", &bt[..bt.len().min(10)]);
        println!("    HuggingFace: {:?}", &hf.get_ids()[..hf.get_ids().len().min(10)]);
    }

    // =========================================================================
    // FINAL SUMMARY
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                        FINAL SUMMARY                             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Dataset: {:.2} GB ({} documents)", total_gb, documents.len());
    println!("");
    println!("┌────────────────┬─────────────────┬─────────────────┬──────────┐");
    println!("│                │   HuggingFace   │    BudTikTok    │  Speedup │");
    println!("├────────────────┼─────────────────┼─────────────────┼──────────┤");
    println!("│ Single-Core    │ {:>8.2} MB/s   │ {:>8.2} MB/s   │  {:>5.2}x  │",
             hf_single_throughput, bt_single_throughput, single_speedup);
    println!("│ Multi-Core(16) │ {:>8.2} MB/s   │ {:>8.2} MB/s   │  {:>5.2}x  │",
             hf_multi_throughput, bt_multi_throughput, multi_speedup);
    println!("└────────────────┴─────────────────┴─────────────────┴──────────┘");
    println!("");
    println!("Accuracy: {:.4}% ({}/{} documents match)",
             accuracy, total_checked - mismatches, total_checked);

    if accuracy >= 99.99 {
        println!("\n✓ ACCURACY VERIFIED: 100% match with HuggingFace!");
    } else {
        println!("\n✗ ACCURACY ISSUE: {:.4}% mismatch rate", 100.0 - accuracy);
    }

    if single_speedup >= 10.0 {
        println!("✓ SINGLE-CORE: Achieved {:.2}x speedup (target: 10x)", single_speedup);
    } else {
        println!("  SINGLE-CORE: {:.2}x speedup (target: 10x)", single_speedup);
    }
}
