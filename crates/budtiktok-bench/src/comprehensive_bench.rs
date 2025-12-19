//! Comprehensive Benchmarking Module
//!
//! Tests with:
//! - Batch sizes: 1, 10, 100, 1000, 5000, 10000
//! - Input lengths: 100, 500, 1000, 5000, 10000, 50000, 100000 words

use std::time::{Duration, Instant};
use rand::Rng;

/// Generate random text with specified word count
pub fn generate_text(word_count: usize) -> String {
    let words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "natural", "language", "processing", "tokenization",
        "neural", "network", "deep", "transformer", "attention", "mechanism",
        "embedding", "vector", "matrix", "computation", "optimization", "gradient",
        "backpropagation", "forward", "inference", "training", "validation", "test",
        "data", "model", "architecture", "layer", "activation", "function", "loss",
        "accuracy", "precision", "recall", "f1", "score", "benchmark", "performance",
        "latency", "throughput", "batch", "sequence", "token", "vocabulary", "subword",
    ];

    let mut rng = rand::thread_rng();
    let mut result = String::with_capacity(word_count * 8);

    for i in 0..word_count {
        if i > 0 {
            result.push(' ');
        }
        result.push_str(words[rng.gen_range(0..words.len())]);

        // Add punctuation occasionally
        if rng.gen_ratio(1, 10) {
            result.push_str(if rng.gen_bool(0.5) { "." } else { "," });
        }
    }

    result
}

/// Benchmark result for a single configuration
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub batch_size: usize,
    pub word_count: usize,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p99_us: f64,
    pub tokens_per_sec: f64,
    pub texts_per_sec: f64,
}

/// Run benchmark with specific configuration
pub fn run_benchmark<F>(
    batch_size: usize,
    word_count: usize,
    iterations: usize,
    warmup: usize,
    tokenize_fn: F,
) -> BenchResult
where
    F: Fn(&[String]) -> Vec<Vec<u32>>,
{
    // Generate test data
    let texts: Vec<String> = (0..batch_size)
        .map(|_| generate_text(word_count))
        .collect();

    // Warmup
    for _ in 0..warmup {
        let _ = tokenize_fn(&texts);
    }

    // Benchmark
    let mut latencies = Vec::with_capacity(iterations);
    let mut total_tokens = 0usize;

    for _ in 0..iterations {
        let start = Instant::now();
        let results = tokenize_fn(&texts);
        let elapsed = start.elapsed();

        latencies.push(elapsed);
        total_tokens += results.iter().map(|r| r.len()).sum::<usize>();
    }

    // Calculate statistics
    latencies.sort();

    let mean = latencies.iter().map(|d| d.as_micros() as f64).sum::<f64>() / iterations as f64;
    let p50 = latencies[iterations / 2].as_micros() as f64;
    let p99 = latencies[(iterations as f64 * 0.99) as usize].as_micros() as f64;

    let total_time_secs: f64 = latencies.iter().map(|d| d.as_secs_f64()).sum();
    let tokens_per_sec = total_tokens as f64 / total_time_secs;
    let texts_per_sec = (batch_size * iterations) as f64 / total_time_secs;

    BenchResult {
        batch_size,
        word_count,
        mean_us: mean,
        p50_us: p50,
        p99_us: p99,
        tokens_per_sec,
        texts_per_sec,
    }
}

/// Print benchmark results in table format
pub fn print_results(backend_name: &str, results: &[BenchResult]) {
    println!("\n=== {} Results ===", backend_name);
    println!("+------------+------------+-----------+----------+----------+---------------+---------------+");
    println!("| Batch Size | Word Count | Mean (μs) | P50 (μs) | P99 (μs) | Tokens/sec    | Texts/sec     |");
    println!("+------------+------------+-----------+----------+----------+---------------+---------------+");

    for r in results {
        println!(
            "| {:>10} | {:>10} | {:>9.1} | {:>8.1} | {:>8.1} | {:>13.0} | {:>13.0} |",
            r.batch_size,
            r.word_count,
            r.mean_us,
            r.p50_us,
            r.p99_us,
            r.tokens_per_sec,
            r.texts_per_sec,
        );
    }
    println!("+------------+------------+-----------+----------+----------+---------------+---------------+");
}
