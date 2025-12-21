//! Token cache benchmark - measures cache hit rates and performance improvement
//!
//! Compares tokenization speed with and without caching on various workloads.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use budtiktok_core::unigram::UnigramPiece;
use budtiktok_core::unigram_fast::{UnigramFast, UnigramFastConfig};
use serde::Deserialize;

const DATASET_PATH: &str = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/openwebtext_1gb.jsonl";
const XLNET_PATH: &str = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/xlnet/tokenizer.json";

#[derive(Deserialize)]
struct TextRecord { text: String }

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║              TOKEN CACHE BENCHMARK                               ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Load tokenizer
    println!("Loading tokenizer...");
    let json_content = std::fs::read_to_string(XLNET_PATH).expect("Failed to read tokenizer");
    let json: serde_json::Value = serde_json::from_str(&json_content).unwrap();
    let model = json.get("model").unwrap();
    let vocab_array = model.get("vocab").unwrap().as_array().unwrap();

    let mut vocab = Vec::with_capacity(vocab_array.len());
    let mut pieces = Vec::with_capacity(vocab_array.len());

    for item in vocab_array {
        let arr = item.as_array().unwrap();
        let token = arr[0].as_str().unwrap();
        let score = arr[1].as_f64().unwrap();
        vocab.push(token.to_string());
        pieces.push(UnigramPiece { token: token.to_string(), score });
    }

    let unk_id = model.get("unk_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let config = UnigramFastConfig {
        unk_id,
        add_prefix_space: true,
        fuse_unk: true,
        ..Default::default()
    };

    // Create tokenizers - one with cache, one without
    let tok_no_cache = UnigramFast::new(vocab.clone(), pieces.clone(), config.clone());
    let tok_with_cache = UnigramFast::with_cache(vocab, pieces, config, 100_000);

    println!("Tokenizer loaded. Cache enabled: {} entries capacity\n", 100_000);

    // Load test texts
    println!("Loading test data...");
    let file = File::open(DATASET_PATH).expect("Failed to open dataset");
    let reader = BufReader::with_capacity(1024 * 1024, file);

    let texts: Vec<String> = reader
        .lines()
        .take(10_000)
        .filter_map(|line| line.ok())
        .filter_map(|l| serde_json::from_str::<TextRecord>(&l).ok().map(|r| r.text))
        .collect();

    let total_bytes: usize = texts.iter().map(|t| t.len()).sum();
    println!("Loaded {} texts ({:.2} MB)\n", texts.len(), total_bytes as f64 / (1024.0 * 1024.0));

    // === Benchmark 1: First pass (cold cache) ===
    println!("═══════════════════════════════════════════════════════════════════");
    println!("BENCHMARK 1: First Pass (Cold Cache)");
    println!("═══════════════════════════════════════════════════════════════════");

    // Without cache
    let start = Instant::now();
    let mut total_tokens_no_cache = 0usize;
    for text in &texts {
        let ids = tok_no_cache.encode_hf_compat_simd_v2(text);
        total_tokens_no_cache += ids.len();
    }
    let elapsed_no_cache = start.elapsed();
    let throughput_no_cache = total_bytes as f64 / elapsed_no_cache.as_secs_f64() / (1024.0 * 1024.0);

    println!("  Without cache: {:>8.2} ms | {:>6.2} MB/s | {} tokens",
             elapsed_no_cache.as_secs_f64() * 1000.0, throughput_no_cache, total_tokens_no_cache);

    // With cache (first pass populates)
    let start = Instant::now();
    let mut total_tokens_with_cache = 0usize;
    for text in &texts {
        let ids = tok_with_cache.encode_hf_compat_simd_v2(text);
        total_tokens_with_cache += ids.len();
    }
    let elapsed_with_cache = start.elapsed();
    let throughput_with_cache = total_bytes as f64 / elapsed_with_cache.as_secs_f64() / (1024.0 * 1024.0);

    if let Some(stats) = tok_with_cache.cache_stats() {
        println!("  With cache:    {:>8.2} ms | {:>6.2} MB/s | {} tokens",
                 elapsed_with_cache.as_secs_f64() * 1000.0, throughput_with_cache, total_tokens_with_cache);
        println!("  Cache stats:   {} entries | {:.2}% hit rate ({} hits, {} misses)",
                 stats.size, stats.hit_rate * 100.0, stats.hits, stats.misses);
    }

    // === Benchmark 2: Second pass (warm cache) ===
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("BENCHMARK 2: Second Pass (Warm Cache)");
    println!("═══════════════════════════════════════════════════════════════════");

    // Without cache (same as before)
    let start = Instant::now();
    for text in &texts {
        let _ = tok_no_cache.encode_hf_compat_simd_v2(text);
    }
    let elapsed_no_cache2 = start.elapsed();
    let throughput_no_cache2 = total_bytes as f64 / elapsed_no_cache2.as_secs_f64() / (1024.0 * 1024.0);

    println!("  Without cache: {:>8.2} ms | {:>6.2} MB/s",
             elapsed_no_cache2.as_secs_f64() * 1000.0, throughput_no_cache2);

    // With cache (should hit cache)
    let start = Instant::now();
    for text in &texts {
        let _ = tok_with_cache.encode_hf_compat_simd_v2(text);
    }
    let elapsed_with_cache2 = start.elapsed();
    let throughput_with_cache2 = total_bytes as f64 / elapsed_with_cache2.as_secs_f64() / (1024.0 * 1024.0);

    if let Some(stats) = tok_with_cache.cache_stats() {
        println!("  With cache:    {:>8.2} ms | {:>6.2} MB/s",
                 elapsed_with_cache2.as_secs_f64() * 1000.0, throughput_with_cache2);
        println!("  Cache stats:   {} entries | {:.2}% hit rate ({} hits, {} misses)",
                 stats.size, stats.hit_rate * 100.0, stats.hits, stats.misses);
    }

    let speedup = throughput_with_cache2 / throughput_no_cache2;
    println!("  Speedup:       {:.2}x", speedup);

    // === Benchmark 3: Repeated text (best case for cache) ===
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("BENCHMARK 3: Repeated Text (Cache Best Case)");
    println!("═══════════════════════════════════════════════════════════════════");

    // Take first 100 texts and repeat them 100 times
    let repeated_texts: Vec<&String> = texts.iter().take(100).cycle().take(10_000).collect();
    let repeated_bytes: usize = repeated_texts.iter().map(|t| t.len()).sum();

    // Clear cache for fresh test
    tok_with_cache.clear_cache();

    // Without cache
    let start = Instant::now();
    for text in &repeated_texts {
        let _ = tok_no_cache.encode_hf_compat_simd_v2(text);
    }
    let elapsed_repeated_no_cache = start.elapsed();
    let throughput_repeated_no_cache = repeated_bytes as f64 / elapsed_repeated_no_cache.as_secs_f64() / (1024.0 * 1024.0);

    println!("  Without cache: {:>8.2} ms | {:>6.2} MB/s",
             elapsed_repeated_no_cache.as_secs_f64() * 1000.0, throughput_repeated_no_cache);

    // With cache (should hit 99% after first 100)
    let start = Instant::now();
    for text in &repeated_texts {
        let _ = tok_with_cache.encode_hf_compat_simd_v2(text);
    }
    let elapsed_repeated_with_cache = start.elapsed();
    let throughput_repeated_with_cache = repeated_bytes as f64 / elapsed_repeated_with_cache.as_secs_f64() / (1024.0 * 1024.0);

    if let Some(stats) = tok_with_cache.cache_stats() {
        println!("  With cache:    {:>8.2} ms | {:>6.2} MB/s",
                 elapsed_repeated_with_cache.as_secs_f64() * 1000.0, throughput_repeated_with_cache);
        println!("  Cache stats:   {} entries | {:.2}% hit rate ({} hits, {} misses)",
                 stats.size, stats.hit_rate * 100.0, stats.hits, stats.misses);
    }

    let speedup_repeated = throughput_repeated_with_cache / throughput_repeated_no_cache;
    println!("  Speedup:       {:.2}x", speedup_repeated);

    // === Summary ===
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Unique texts:   {:.2}x speedup with warm cache", throughput_with_cache2 / throughput_no_cache2);
    println!("  Repeated texts: {:.2}x speedup (99% hit rate scenario)", speedup_repeated);
    println!();
    println!("Note: Cache is most effective when the same texts are encoded multiple times,");
    println!("such as in batch processing with data augmentation or repeated inference.");
}
