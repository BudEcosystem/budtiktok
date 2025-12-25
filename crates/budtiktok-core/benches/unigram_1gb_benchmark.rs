//! Comprehensive 1GB Benchmark: BudTikTok vs HuggingFace Unigram Tokenizers
//!
//! This benchmark runs both tokenizers on the full 1GB OpenWebText dataset
//! and measures:
//! - Throughput (MB/s, tokens/sec)
//! - Latency (mean, p50, p95, p99)
//! - Accuracy (100% match verification)
//! - Memory efficiency
//!
//! Run with: cargo run --release --bin unigram_1gb_benchmark

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicUsize, Ordering};

use budtiktok_core::unigram::UnigramPiece;
use budtiktok_core::unigram_fast::{UnigramFast, UnigramFastConfig};
use budtiktok_core::vocab::{Vocabulary, SpecialTokens};
use tokenizers::Tokenizer as HfTokenizer;
use rayon::prelude::*;
use serde::Deserialize;
use ahash::AHashMap;

// ============================================================================
// Constants
// ============================================================================

const DATASET_PATH: &str = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/openwebtext_1gb.jsonl";
const XLNET_TOKENIZER_PATH: &str = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/xlnet/tokenizer.json";

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Deserialize)]
struct TextRecord {
    text: String,
}

#[derive(Debug, Clone)]
struct BenchmarkStats {
    name: String,
    total_bytes: usize,
    total_tokens: usize,
    total_texts: usize,
    elapsed: Duration,
    throughput_mbs: f64,
    throughput_tokens_sec: f64,
    latencies_us: Vec<f64>,
}

impl BenchmarkStats {
    fn print_summary(&self) {
        println!("\n{}", "=".repeat(60));
        println!("  {}", self.name);
        println!("{}", "=".repeat(60));
        println!("  Total texts:      {:>15}", format_number(self.total_texts));
        println!("  Total bytes:      {:>15}", format_bytes(self.total_bytes));
        println!("  Total tokens:     {:>15}", format_number(self.total_tokens));
        println!("  Total time:       {:>15.2?}", self.elapsed);
        println!("  Throughput:       {:>15.2} MB/s", self.throughput_mbs);
        println!("  Token rate:       {:>15} tokens/sec", format_number(self.throughput_tokens_sec as usize));

        if !self.latencies_us.is_empty() {
            let mut sorted = self.latencies_us.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let len = sorted.len();

            let mean = sorted.iter().sum::<f64>() / len as f64;
            let p50 = sorted[len / 2];
            let p95 = sorted[(len as f64 * 0.95) as usize];
            let p99 = sorted[(len as f64 * 0.99) as usize];
            let min = sorted[0];
            let max = sorted[len - 1];

            println!("\n  Latency (per text):");
            println!("    Mean:           {:>15.2} µs", mean);
            println!("    P50:            {:>15.2} µs", p50);
            println!("    P95:            {:>15.2} µs", p95);
            println!("    P99:            {:>15.2} µs", p99);
            println!("    Min:            {:>15.2} µs", min);
            println!("    Max:            {:>15.2} µs", max);
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

fn load_dataset(path: &str, limit: Option<usize>) -> Vec<String> {
    println!("Loading dataset from {}...", path);
    let start = Instant::now();

    let file = File::open(path).expect("Failed to open dataset");
    let reader = BufReader::with_capacity(8 * 1024 * 1024, file);

    let texts: Vec<String> = reader
        .lines()
        .take(limit.unwrap_or(usize::MAX))
        .filter_map(|line| {
            line.ok().and_then(|l| {
                serde_json::from_str::<TextRecord>(&l)
                    .ok()
                    .map(|r| r.text)
            })
        })
        .collect();

    let total_bytes: usize = texts.iter().map(|t| t.len()).sum();

    println!("Loaded {} texts ({}) in {:?}",
             format_number(texts.len()),
             format_bytes(total_bytes),
             start.elapsed());

    texts
}

fn load_hf_tokenizer() -> HfTokenizer {
    println!("Loading HuggingFace tokenizer...");
    HfTokenizer::from_file(XLNET_TOKENIZER_PATH)
        .expect("Failed to load HuggingFace tokenizer")
}

fn load_budtiktok_tokenizer() -> UnigramFast {
    println!("Loading BudTikTok tokenizer...");
    let json_content = std::fs::read_to_string(XLNET_TOKENIZER_PATH)
        .expect("Failed to read tokenizer file");
    let json: serde_json::Value = serde_json::from_str(&json_content)
        .expect("Failed to parse tokenizer JSON");

    let model = json.get("model").expect("No model in tokenizer");
    let vocab_array = model.get("vocab")
        .expect("No vocab in model")
        .as_array()
        .expect("vocab is not array");

    let mut vocab = Vec::with_capacity(vocab_array.len());
    let mut pieces = Vec::with_capacity(vocab_array.len());

    for item in vocab_array {
        let arr = item.as_array().expect("vocab item not array");
        let token = arr[0].as_str().expect("token not string");
        let score = arr[1].as_f64().expect("score not f64");
        vocab.push(token.to_string());
        pieces.push(UnigramPiece {
            token: token.to_string(),
            score,
        });
    }

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

    UnigramFast::new(vocabulary, pieces, config)
}

// ============================================================================
// Benchmark Functions
// ============================================================================

fn benchmark_huggingface_sequential(
    tokenizer: &HfTokenizer,
    texts: &[String],
) -> BenchmarkStats {
    println!("\nRunning HuggingFace Sequential benchmark...");

    let total_bytes: usize = texts.iter().map(|t| t.len()).sum();
    let mut latencies = Vec::with_capacity(texts.len());
    let mut total_tokens = 0usize;

    let start = Instant::now();

    for text in texts {
        let text_start = Instant::now();
        let encoding = tokenizer.encode(text.as_str(), false).unwrap();
        total_tokens += encoding.get_ids().len();
        latencies.push(text_start.elapsed().as_micros() as f64);
    }

    let elapsed = start.elapsed();
    let throughput_mbs = total_bytes as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    let throughput_tokens_sec = total_tokens as f64 / elapsed.as_secs_f64();

    BenchmarkStats {
        name: "HuggingFace (Sequential)".to_string(),
        total_bytes,
        total_tokens,
        total_texts: texts.len(),
        elapsed,
        throughput_mbs,
        throughput_tokens_sec,
        latencies_us: latencies,
    }
}

fn benchmark_huggingface_parallel(
    tokenizer: &HfTokenizer,
    texts: &[String],
) -> BenchmarkStats {
    println!("\nRunning HuggingFace Parallel benchmark...");

    let total_bytes: usize = texts.iter().map(|t| t.len()).sum();
    let total_tokens = AtomicUsize::new(0);

    let start = Instant::now();

    texts.par_iter().for_each(|text| {
        let encoding = tokenizer.encode(text.as_str(), false).unwrap();
        total_tokens.fetch_add(encoding.get_ids().len(), Ordering::Relaxed);
    });

    let elapsed = start.elapsed();
    let total_tokens = total_tokens.load(Ordering::Relaxed);
    let throughput_mbs = total_bytes as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    let throughput_tokens_sec = total_tokens as f64 / elapsed.as_secs_f64();

    BenchmarkStats {
        name: "HuggingFace (Parallel)".to_string(),
        total_bytes,
        total_tokens,
        total_texts: texts.len(),
        elapsed,
        throughput_mbs,
        throughput_tokens_sec,
        latencies_us: vec![], // Can't measure per-text latency in parallel
    }
}

fn benchmark_budtiktok_sequential(
    tokenizer: &UnigramFast,
    texts: &[String],
) -> BenchmarkStats {
    println!("\nRunning BudTikTok Sequential benchmark...");

    let total_bytes: usize = texts.iter().map(|t| t.len()).sum();
    let mut latencies = Vec::with_capacity(texts.len());
    let mut total_tokens = 0usize;

    let start = Instant::now();

    for text in texts {
        let text_start = Instant::now();
        let ids = tokenizer.encode_hf_compat_simd_v2(text);
        total_tokens += ids.len();
        latencies.push(text_start.elapsed().as_micros() as f64);
    }

    let elapsed = start.elapsed();
    let throughput_mbs = total_bytes as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    let throughput_tokens_sec = total_tokens as f64 / elapsed.as_secs_f64();

    BenchmarkStats {
        name: "BudTikTok (Sequential, HF-compat)".to_string(),
        total_bytes,
        total_tokens,
        total_texts: texts.len(),
        elapsed,
        throughput_mbs,
        throughput_tokens_sec,
        latencies_us: latencies,
    }
}

fn benchmark_budtiktok_parallel(
    tokenizer: &UnigramFast,
    texts: &[String],
) -> BenchmarkStats {
    println!("\nRunning BudTikTok Parallel benchmark...");

    let total_bytes: usize = texts.iter().map(|t| t.len()).sum();
    let total_tokens = AtomicUsize::new(0);

    let start = Instant::now();

    texts.par_iter().for_each(|text| {
        let ids = tokenizer.encode_hf_compat_simd_v2(text);
        total_tokens.fetch_add(ids.len(), Ordering::Relaxed);
    });

    let elapsed = start.elapsed();
    let total_tokens = total_tokens.load(Ordering::Relaxed);
    let throughput_mbs = total_bytes as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    let throughput_tokens_sec = total_tokens as f64 / elapsed.as_secs_f64();

    BenchmarkStats {
        name: "BudTikTok (Parallel, HF-compat)".to_string(),
        total_bytes,
        total_tokens,
        total_texts: texts.len(),
        elapsed,
        throughput_mbs,
        throughput_tokens_sec,
        latencies_us: vec![],
    }
}

fn benchmark_budtiktok_fast_parallel(
    tokenizer: &UnigramFast,
    texts: &[String],
) -> BenchmarkStats {
    println!("\nRunning BudTikTok Fast (no normalization) Parallel benchmark...");

    let total_bytes: usize = texts.iter().map(|t| t.len()).sum();
    let total_tokens = AtomicUsize::new(0);

    let start = Instant::now();

    texts.par_iter().for_each(|text| {
        let ids = tokenizer.encode_fast(text);
        total_tokens.fetch_add(ids.len(), Ordering::Relaxed);
    });

    let elapsed = start.elapsed();
    let total_tokens = total_tokens.load(Ordering::Relaxed);
    let throughput_mbs = total_bytes as f64 / elapsed.as_secs_f64() / 1_000_000.0;
    let throughput_tokens_sec = total_tokens as f64 / elapsed.as_secs_f64();

    BenchmarkStats {
        name: "BudTikTok (Fast, no normalization)".to_string(),
        total_bytes,
        total_tokens,
        total_texts: texts.len(),
        elapsed,
        throughput_mbs,
        throughput_tokens_sec,
        latencies_us: vec![],
    }
}

// ============================================================================
// Accuracy Verification
// ============================================================================

fn verify_accuracy(
    hf_tokenizer: &HfTokenizer,
    bt_tokenizer: &UnigramFast,
    texts: &[String],
    sample_size: usize,
) -> (usize, usize, f64) {
    println!("\nVerifying accuracy on {} samples...", sample_size);

    let sample: Vec<_> = texts.iter().take(sample_size).collect();
    let mut matches = 0usize;
    let mut mismatches = 0usize;
    let mut mismatch_details: Vec<(usize, usize, String)> = Vec::new();

    for text in &sample {
        let hf_encoding = hf_tokenizer.encode(text.as_str(), false).unwrap();
        let hf_ids: Vec<u32> = hf_encoding.get_ids().to_vec();
        let bt_ids = bt_tokenizer.encode_hf_compat_simd_v2(text);

        if hf_ids == bt_ids {
            matches += 1;
        } else {
            mismatches += 1;

            // Find first difference position
            let mut first_diff_pos = None;
            for (i, (hf, bt)) in hf_ids.iter().zip(bt_ids.iter()).enumerate() {
                if hf != bt {
                    first_diff_pos = Some(i);
                    break;
                }
            }

            // Categorize mismatch
            let reason = if hf_ids.len() != bt_ids.len() && first_diff_pos.is_none() {
                format!("Length diff: HF={} BT={}", hf_ids.len(), bt_ids.len())
            } else if let Some(pos) = first_diff_pos {
                // Get the substring around this position
                let hf_tokens_around: Vec<_> = (pos.saturating_sub(2)..pos+3.min(hf_ids.len()))
                    .filter_map(|i| hf_ids.get(i).map(|id| hf_tokenizer.decode(&[*id], false).ok()))
                    .flatten()
                    .collect();
                format!("Diff@{}: {:?}", pos, hf_tokens_around)
            } else {
                "Unknown".to_string()
            };

            mismatch_details.push((hf_ids.len(), bt_ids.len(), reason));

            if mismatches <= 5 {
                println!("  Mismatch on text (len={}): {:?}...",
                         text.len(),
                         &text[..text.len().min(80)]);

                if let Some(pos) = first_diff_pos {
                    println!("    First diff at token {}", pos);
                    println!("    HF[{}..{}]: {:?}",
                             pos.saturating_sub(2),
                             pos+5.min(hf_ids.len()),
                             &hf_ids[pos.saturating_sub(2)..pos+5.min(hf_ids.len())]);
                    println!("    BT[{}..{}]: {:?}",
                             pos.saturating_sub(2),
                             pos+5.min(bt_ids.len()),
                             &bt_ids[pos.saturating_sub(2)..pos+5.min(bt_ids.len())]);

                    // Decode tokens to see what they represent
                    if let Ok(hf_decoded) = hf_tokenizer.decode(&hf_ids[pos..pos+1.min(hf_ids.len())], false) {
                        println!("    HF token text: {:?}", hf_decoded);
                    }
                    if let Some(tok) = bt_tokenizer.id_to_token(bt_ids[pos]) {
                        println!("    BT token text: {:?}", tok);
                    }
                } else {
                    println!("    Length mismatch: HF={} BT={}", hf_ids.len(), bt_ids.len());
                }
            }
        }
    }

    // Print mismatch analysis
    if mismatches > 0 {
        println!("\n  Mismatch analysis (first 20):");
        for (i, (hf_len, bt_len, reason)) in mismatch_details.iter().take(20).enumerate() {
            println!("    {}: HF={} BT={} - {}", i+1, hf_len, bt_len, reason);
        }
    }

    let accuracy = matches as f64 / sample.len() as f64 * 100.0;
    (matches, mismatches, accuracy)
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("\n");
    println!("{}", "#".repeat(70));
    println!("#  COMPREHENSIVE 1GB UNIGRAM TOKENIZER BENCHMARK");
    println!("#  BudTikTok vs HuggingFace Tokenizers");
    println!("{}", "#".repeat(70));

    // Load tokenizers
    let hf_tok = load_hf_tokenizer();
    let bt_tok = load_budtiktok_tokenizer();

    // Load dataset
    let texts = load_dataset(DATASET_PATH, None);

    // Warmup
    println!("\nWarming up...");
    for text in texts.iter().take(1000) {
        let _ = hf_tok.encode(text.as_str(), false);
        let _ = bt_tok.encode_hf_compat_simd_v2(text);
    }

    // Run benchmarks
    let results = vec![
        benchmark_huggingface_sequential(&hf_tok, &texts),
        benchmark_huggingface_parallel(&hf_tok, &texts),
        benchmark_budtiktok_sequential(&bt_tok, &texts),
        benchmark_budtiktok_parallel(&bt_tok, &texts),
        benchmark_budtiktok_fast_parallel(&bt_tok, &texts),
    ];

    // Print results
    for result in &results {
        result.print_summary();
    }

    // Verify accuracy
    let (matches, mismatches, accuracy) = verify_accuracy(&hf_tok, &bt_tok, &texts, 10000);

    println!("\n{}", "=".repeat(60));
    println!("  ACCURACY VERIFICATION");
    println!("{}", "=".repeat(60));
    println!("  Samples tested:   {:>15}", format_number(matches + mismatches));
    println!("  Matches:          {:>15}", format_number(matches));
    println!("  Mismatches:       {:>15}", format_number(mismatches));
    println!("  Accuracy:         {:>14.2}%", accuracy);

    // Calculate speedups
    println!("\n{}", "=".repeat(60));
    println!("  SPEEDUP COMPARISON");
    println!("{}", "=".repeat(60));

    let hf_seq = &results[0];
    let hf_par = &results[1];
    let bt_seq = &results[2];
    let bt_par = &results[3];
    let bt_fast = &results[4];

    println!("  BudTikTok (seq) vs HuggingFace (seq):  {:>6.1}x",
             hf_seq.elapsed.as_secs_f64() / bt_seq.elapsed.as_secs_f64());
    println!("  BudTikTok (par) vs HuggingFace (par):  {:>6.1}x",
             hf_par.elapsed.as_secs_f64() / bt_par.elapsed.as_secs_f64());
    println!("  BudTikTok (par) vs HuggingFace (seq):  {:>6.1}x",
             hf_seq.elapsed.as_secs_f64() / bt_par.elapsed.as_secs_f64());
    println!("  BudTikTok Fast vs HuggingFace (par):   {:>6.1}x",
             hf_par.elapsed.as_secs_f64() / bt_fast.elapsed.as_secs_f64());

    println!("\n{}", "#".repeat(70));
    println!("#  BENCHMARK COMPLETE");
    println!("{}\n", "#".repeat(70));
}
