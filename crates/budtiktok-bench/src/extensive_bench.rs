//! Extensive Single-Core Benchmark Module
//!
//! Comprehensive benchmarking across:
//! - Batch sizes: 1 to 15000
//! - Sequence lengths: 100 to 100000 words
//! - Variable and fixed-length inputs
//! - With and without SIMD vectorization

use std::time::{Duration, Instant};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for extensive benchmark
#[derive(Debug, Clone)]
pub struct ExtensiveBenchConfig {
    /// Batch sizes to test
    pub batch_sizes: Vec<usize>,
    /// Sequence lengths (in words) to test
    pub seq_lengths: Vec<usize>,
    /// Whether to use variable-length inputs within each batch
    pub variable_length: bool,
    /// Number of iterations per configuration
    pub iterations: usize,
    /// Warmup iterations
    pub warmup: usize,
    /// Duration limit per configuration (seconds)
    pub duration_limit_secs: u64,
}

impl Default for ExtensiveBenchConfig {
    fn default() -> Self {
        Self {
            batch_sizes: vec![1, 10, 50, 100, 500, 1000, 2000, 5000, 10000, 15000],
            seq_lengths: vec![100, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000],
            variable_length: false,
            iterations: 10,
            warmup: 3,
            duration_limit_secs: 60,
        }
    }
}

/// Result for a single benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensiveResult {
    pub backend: String,
    pub batch_size: usize,
    pub seq_length: usize,
    pub variable_length: bool,
    pub iterations: usize,
    pub total_tokens: usize,
    pub total_texts: usize,

    // Timing in microseconds
    pub mean_latency_us: f64,
    pub p50_latency_us: f64,
    pub p95_latency_us: f64,
    pub p99_latency_us: f64,
    pub min_latency_us: f64,
    pub max_latency_us: f64,

    // Throughput
    pub tokens_per_sec: f64,
    pub texts_per_sec: f64,
    pub batches_per_sec: f64,

    // Per-token and per-text metrics
    pub us_per_token: f64,
    pub us_per_text: f64,
}

/// Generate text with specified word count
pub fn generate_text_with_words(word_count: usize) -> String {
    let words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "natural", "language", "processing", "tokenization",
        "neural", "network", "deep", "transformer", "attention", "mechanism",
        "embedding", "vector", "matrix", "computation", "optimization", "gradient",
        "backpropagation", "forward", "inference", "training", "validation", "test",
        "data", "model", "architecture", "layer", "activation", "function", "loss",
        "accuracy", "precision", "recall", "score", "benchmark", "performance",
        "latency", "throughput", "batch", "sequence", "token", "vocabulary", "subword",
        "encoder", "decoder", "hidden", "state", "memory", "cache", "buffer",
        "parallel", "distributed", "efficient", "fast", "optimized", "scalable",
    ];

    let mut rng = rand::thread_rng();
    let mut result = String::with_capacity(word_count * 8);

    for i in 0..word_count {
        if i > 0 {
            result.push(' ');
        }
        result.push_str(words[rng.gen_range(0..words.len())]);

        // Add punctuation occasionally
        if rng.gen_ratio(1, 15) {
            result.push_str(match rng.gen_range(0..4) {
                0 => ".",
                1 => ",",
                2 => "!",
                _ => "?",
            });
        }
    }

    result
}

/// Generate batch of texts with fixed or variable length
pub fn generate_batch(batch_size: usize, target_words: usize, variable: bool) -> Vec<String> {
    let mut rng = rand::thread_rng();

    (0..batch_size)
        .map(|_| {
            let words = if variable {
                // Variable: 50% to 150% of target
                let min = (target_words as f64 * 0.5) as usize;
                let max = (target_words as f64 * 1.5) as usize;
                rng.gen_range(min.max(10)..max.max(20))
            } else {
                target_words
            };
            generate_text_with_words(words)
        })
        .collect()
}

/// Run benchmark for a single configuration
pub fn run_single_config<F>(
    backend_name: &str,
    batch_size: usize,
    seq_length: usize,
    variable_length: bool,
    iterations: usize,
    warmup: usize,
    duration_limit: Duration,
    encode_batch_fn: F,
) -> ExtensiveResult
where
    F: Fn(&[String]) -> Vec<Vec<u32>>,
{
    // Generate test data
    let texts = generate_batch(batch_size, seq_length, variable_length);

    // Warmup
    for _ in 0..warmup {
        let _ = encode_batch_fn(&texts);
    }

    // Benchmark
    let mut latencies = Vec::with_capacity(iterations);
    let mut total_tokens = 0usize;
    let start_time = Instant::now();
    let mut actual_iterations = 0usize;

    for _ in 0..iterations {
        if start_time.elapsed() > duration_limit {
            break;
        }

        let iter_start = Instant::now();
        let results = encode_batch_fn(&texts);
        let elapsed = iter_start.elapsed();

        latencies.push(elapsed.as_micros() as f64);
        for result in &results {
            total_tokens += result.len();
        }
        actual_iterations += 1;
    }

    // Calculate statistics
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = latencies.len().max(1);

    let mean = latencies.iter().sum::<f64>() / n as f64;
    let p50 = latencies.get(n / 2).copied().unwrap_or(0.0);
    let p95 = latencies.get(n * 95 / 100).copied().unwrap_or(0.0);
    let p99 = latencies.get(n * 99 / 100).copied().unwrap_or(0.0);
    let min = latencies.first().copied().unwrap_or(0.0);
    let max = latencies.last().copied().unwrap_or(0.0);

    let total_time_us: f64 = latencies.iter().sum();
    let total_time_sec = total_time_us / 1_000_000.0;
    let total_texts = batch_size * actual_iterations;

    let tokens_per_sec = if total_time_sec > 0.0 { total_tokens as f64 / total_time_sec } else { 0.0 };
    let texts_per_sec = if total_time_sec > 0.0 { total_texts as f64 / total_time_sec } else { 0.0 };
    let batches_per_sec = if total_time_sec > 0.0 { actual_iterations as f64 / total_time_sec } else { 0.0 };

    let us_per_token = if total_tokens > 0 { total_time_us / total_tokens as f64 } else { 0.0 };
    let us_per_text = if total_texts > 0 { total_time_us / total_texts as f64 } else { 0.0 };

    ExtensiveResult {
        backend: backend_name.to_string(),
        batch_size,
        seq_length,
        variable_length,
        iterations: actual_iterations,
        total_tokens,
        total_texts,
        mean_latency_us: mean,
        p50_latency_us: p50,
        p95_latency_us: p95,
        p99_latency_us: p99,
        min_latency_us: min,
        max_latency_us: max,
        tokens_per_sec,
        texts_per_sec,
        batches_per_sec,
        us_per_token,
        us_per_text,
    }
}

/// Print results in table format
pub fn print_results_table(results: &[ExtensiveResult], baseline_backend: &str) {
    println!("\n{:=<140}", "");
    println!("{:^140}", "EXTENSIVE SINGLE-CORE BENCHMARK RESULTS");
    println!("{:=<140}\n", "");

    // Group by batch_size and seq_length
    let mut grouped: std::collections::BTreeMap<(usize, usize, bool), Vec<&ExtensiveResult>> =
        std::collections::BTreeMap::new();

    for result in results {
        grouped
            .entry((result.batch_size, result.seq_length, result.variable_length))
            .or_default()
            .push(result);
    }

    // Print header
    println!("{:-<140}", "");
    println!(
        "{:>6} | {:>8} | {:>8} | {:>18} | {:>12} | {:>12} | {:>10} | {:>10} | {:>10}",
        "Batch", "SeqLen", "VarLen", "Backend", "Tokens/s", "Texts/s", "Mean(μs)", "P99(μs)", "Speedup"
    );
    println!("{:-<140}", "");

    for ((batch_size, seq_len, variable), group) in &grouped {
        // Find baseline throughput
        let baseline_throughput = group
            .iter()
            .find(|r| r.backend == baseline_backend)
            .map(|r| r.tokens_per_sec)
            .unwrap_or(1.0);

        for (i, result) in group.iter().enumerate() {
            let speedup = result.tokens_per_sec / baseline_throughput;
            let var_str = if *variable { "Yes" } else { "No" };

            // Only print batch/seq/var on first row of group
            let (b_str, s_str, v_str) = if i == 0 {
                (format!("{}", batch_size), format!("{}", seq_len), var_str.to_string())
            } else {
                ("".to_string(), "".to_string(), "".to_string())
            };

            println!(
                "{:>6} | {:>8} | {:>8} | {:>18} | {:>12.0} | {:>12.0} | {:>10.1} | {:>10.1} | {:>10.2}x",
                b_str, s_str, v_str,
                result.backend,
                result.tokens_per_sec,
                result.texts_per_sec,
                result.mean_latency_us,
                result.p99_latency_us,
                speedup
            );
        }
        println!("{:-<140}", "");
    }
}

/// Export results to CSV
pub fn export_csv(results: &[ExtensiveResult], path: &str) -> std::io::Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(path)?;

    // Header
    writeln!(
        file,
        "backend,batch_size,seq_length,variable_length,iterations,total_tokens,total_texts,\
         mean_latency_us,p50_latency_us,p95_latency_us,p99_latency_us,min_latency_us,max_latency_us,\
         tokens_per_sec,texts_per_sec,batches_per_sec,us_per_token,us_per_text"
    )?;

    // Data rows
    for r in results {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.6},{:.6}",
            r.backend, r.batch_size, r.seq_length, r.variable_length,
            r.iterations, r.total_tokens, r.total_texts,
            r.mean_latency_us, r.p50_latency_us, r.p95_latency_us, r.p99_latency_us,
            r.min_latency_us, r.max_latency_us,
            r.tokens_per_sec, r.texts_per_sec, r.batches_per_sec,
            r.us_per_token, r.us_per_text
        )?;
    }

    Ok(())
}

/// Summary statistics across all configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSummary {
    pub backend: String,
    pub total_configurations: usize,
    pub avg_tokens_per_sec: f64,
    pub max_tokens_per_sec: f64,
    pub min_tokens_per_sec: f64,
    pub avg_speedup_vs_baseline: f64,
    pub max_speedup_vs_baseline: f64,
}

/// Calculate summary statistics for each backend
pub fn calculate_summaries(results: &[ExtensiveResult], baseline_backend: &str) -> Vec<BackendSummary> {
    let mut backend_results: std::collections::HashMap<String, Vec<&ExtensiveResult>> =
        std::collections::HashMap::new();

    for result in results {
        backend_results
            .entry(result.backend.clone())
            .or_default()
            .push(result);
    }

    // Get baseline results for speedup calculation
    let baseline_map: std::collections::HashMap<(usize, usize, bool), f64> = results
        .iter()
        .filter(|r| r.backend == baseline_backend)
        .map(|r| ((r.batch_size, r.seq_length, r.variable_length), r.tokens_per_sec))
        .collect();

    let mut summaries = Vec::new();

    for (backend, results) in &backend_results {
        let throughputs: Vec<f64> = results.iter().map(|r| r.tokens_per_sec).collect();
        let n = throughputs.len() as f64;

        let avg_throughput = throughputs.iter().sum::<f64>() / n;
        let max_throughput = throughputs.iter().cloned().fold(0.0, f64::max);
        let min_throughput = throughputs.iter().cloned().fold(f64::INFINITY, f64::min);

        // Calculate speedups
        let speedups: Vec<f64> = results
            .iter()
            .filter_map(|r| {
                baseline_map
                    .get(&(r.batch_size, r.seq_length, r.variable_length))
                    .map(|baseline| r.tokens_per_sec / baseline)
            })
            .collect();

        let avg_speedup = if speedups.is_empty() {
            1.0
        } else {
            speedups.iter().sum::<f64>() / speedups.len() as f64
        };
        let max_speedup = speedups.iter().cloned().fold(1.0, f64::max);

        summaries.push(BackendSummary {
            backend: backend.clone(),
            total_configurations: results.len(),
            avg_tokens_per_sec: avg_throughput,
            max_tokens_per_sec: max_throughput,
            min_tokens_per_sec: min_throughput,
            avg_speedup_vs_baseline: avg_speedup,
            max_speedup_vs_baseline: max_speedup,
        });
    }

    // Sort by avg throughput descending
    summaries.sort_by(|a, b| b.avg_tokens_per_sec.partial_cmp(&a.avg_tokens_per_sec).unwrap());

    summaries
}

/// Print summary table
pub fn print_summary_table(summaries: &[BackendSummary]) {
    println!("\n{:=<100}", "");
    println!("{:^100}", "BACKEND SUMMARY");
    println!("{:=<100}\n", "");

    println!(
        "{:<20} | {:>12} | {:>14} | {:>14} | {:>14} | {:>10} | {:>10}",
        "Backend", "Configs", "Avg Tokens/s", "Max Tokens/s", "Min Tokens/s", "Avg Speedup", "Max Speedup"
    );
    println!("{:-<100}", "");

    for s in summaries {
        println!(
            "{:<20} | {:>12} | {:>14.0} | {:>14.0} | {:>14.0} | {:>10.2}x | {:>10.2}x",
            s.backend,
            s.total_configurations,
            s.avg_tokens_per_sec,
            s.max_tokens_per_sec,
            s.min_tokens_per_sec,
            s.avg_speedup_vs_baseline,
            s.max_speedup_vs_baseline
        );
    }
    println!("{:-<100}", "");
}
