//! BudTikTok 1GB Dataset Benchmark
//! Tests single-core and multi-core performance

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use ahash::AHashMap;
use rayon::prelude::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║          BudTikTok 1GB Dataset Benchmark                         ║");
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

    // Create encoder
    println!("\nInitializing BudTikTok encoder...");
    use budtiktok_core::bpe_linear::OptimizedBpeEncoder;
    let encoder = OptimizedBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<|endoftext|>");

    // Warm up
    println!("\nWarming up (1000 docs)...");
    for doc in documents.iter().take(1000) {
        let _ = encoder.encode(doc);
    }

    // =========================================================================
    // EXPERIMENT 1: Single-Core Benchmark
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║              EXPERIMENT 1: Single-Core Benchmark                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Running BudTikTok (single-core) on {:.2} GB...", total_gb);
    let start = Instant::now();
    let mut total_tokens_single = 0usize;
    for doc in &documents {
        total_tokens_single += encoder.encode(doc).len();
    }
    let single_time = start.elapsed();
    let single_throughput = total_mb / single_time.as_secs_f64();

    println!("  Time: {:.2?}", single_time);
    println!("  Throughput: {:.2} MB/s", single_throughput);
    println!("  Tokens: {}", total_tokens_single);
    println!("  Tokens/sec: {:.0}", total_tokens_single as f64 / single_time.as_secs_f64());

    // =========================================================================
    // EXPERIMENT 2: Multi-Core Benchmark
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║         EXPERIMENT 2: Multi-Core Benchmark (16 cores)            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Set thread pool size to 16
    rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build_global()
        .ok();

    println!("Running BudTikTok (16 cores) on {:.2} GB...", total_gb);
    let start = Instant::now();
    let results: Vec<usize> = documents.par_iter()
        .map(|doc| encoder.encode(doc).len())
        .collect();
    let total_tokens_multi: usize = results.iter().sum();
    let multi_time = start.elapsed();
    let multi_throughput = total_mb / multi_time.as_secs_f64();

    println!("  Time: {:.2?}", multi_time);
    println!("  Throughput: {:.2} MB/s", multi_throughput);
    println!("  Tokens: {}", total_tokens_multi);
    println!("  Tokens/sec: {:.0}", total_tokens_multi as f64 / multi_time.as_secs_f64());
    println!("  Parallel efficiency: {:.1}%", (multi_throughput / single_throughput / 16.0) * 100.0);

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                     BUDTIKTOK RESULTS                            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Dataset: {:.2} GB ({} documents)", total_gb, documents.len());
    println!("");
    println!("┌────────────────┬─────────────────┬─────────────────┐");
    println!("│    Mode        │   Throughput    │   Tokens/sec    │");
    println!("├────────────────┼─────────────────┼─────────────────┤");
    println!("│ Single-Core    │ {:>8.2} MB/s   │ {:>12.0}    │", single_throughput, total_tokens_single as f64 / single_time.as_secs_f64());
    println!("│ Multi-Core(16) │ {:>8.2} MB/s   │ {:>12.0}    │", multi_throughput, total_tokens_multi as f64 / multi_time.as_secs_f64());
    println!("└────────────────┴─────────────────┴─────────────────┘");
    println!("");

    // Output for comparison script
    println!("\n[RESULTS_FOR_COMPARISON]");
    println!("BUDTIKTOK_SINGLE_THROUGHPUT={:.2}", single_throughput);
    println!("BUDTIKTOK_MULTI_THROUGHPUT={:.2}", multi_throughput);
    println!("BUDTIKTOK_SINGLE_TOKENS={}", total_tokens_single);
    println!("BUDTIKTOK_MULTI_TOKENS={}", total_tokens_multi);
}
