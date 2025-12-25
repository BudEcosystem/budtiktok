//! Extensive Single-Core Benchmark Runner
//!
//! Comprehensive comparison of tokenizer backends across:
//! - Batch sizes: 1 to 15000
//! - Sequence lengths: 100 to 100000 words
//! - Variable and fixed-length inputs
//! - BlazeText, BudTikTok (no SIMD), BudTikTok-Hyper (SIMD), HuggingFace

use ahash;
use anyhow::{Context, Result};
#[cfg(feature = "blazetext")]
use blazetext_wordpiece::BertWordPieceTokenizer;
use budtiktok_core::vocab::{SpecialTokens, Vocabulary};
use budtiktok_core::wordpiece::{WordPieceConfig, WordPieceTokenizer};
use budtiktok_core::wordpiece_hyper::{HyperConfig, HyperWordPieceTokenizer};
use budtiktok_core::tokenizer::Tokenizer;
use clap::Parser;
use colored::Colorize;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer as HfTokenizer;

#[derive(Parser)]
#[command(name = "extensive-benchmark")]
#[command(about = "Run extensive single-core tokenizer benchmark")]
struct Cli {
    /// Path to tokenizer.json file
    #[arg(short, long)]
    tokenizer: String,

    /// Output CSV file path
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Comma-separated batch sizes (default: 1,10,100,500,1000,5000,10000)
    #[arg(long, value_delimiter = ',')]
    batch_sizes: Option<Vec<usize>>,

    /// Comma-separated sequence lengths in words (default: 100,500,1000,5000,10000,50000)
    #[arg(long, value_delimiter = ',')]
    seq_lengths: Option<Vec<usize>>,

    /// Include variable-length tests
    #[arg(long)]
    include_variable: bool,

    /// Iterations per configuration
    #[arg(long, default_value = "10")]
    iterations: usize,

    /// Warmup iterations
    #[arg(long, default_value = "3")]
    warmup: usize,

    /// Max duration per config in seconds
    #[arg(long, default_value = "30")]
    max_duration: u64,

    /// Skip specific backends (comma-separated: huggingface,blazetext,budtiktok,budtiktok-hyper)
    #[arg(long, value_delimiter = ',')]
    skip: Option<Vec<String>>,

    /// Run quick test with smaller parameters
    #[arg(long)]
    quick: bool,

    /// Maximum memory usage in GB (default: 8)
    #[arg(long, default_value = "8")]
    max_memory_gb: f64,

    /// Enable CUDA GPU benchmarks (requires --features cuda)
    #[arg(long)]
    cuda: bool,
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
    pub mean_latency_us: f64,
    pub p50_latency_us: f64,
    pub p95_latency_us: f64,
    pub p99_latency_us: f64,
    pub min_latency_us: f64,
    pub max_latency_us: f64,
    pub tokens_per_sec: f64,
    pub texts_per_sec: f64,
    pub batches_per_sec: f64,
    pub us_per_token: f64,
    pub us_per_text: f64,
}

/// Summary statistics for a backend
#[derive(Debug, Clone)]
pub struct BackendSummary {
    pub backend: String,
    pub total_configurations: usize,
    pub avg_tokens_per_sec: f64,
    pub max_tokens_per_sec: f64,
    pub min_tokens_per_sec: f64,
    pub avg_speedup_vs_baseline: f64,
    pub max_speedup_vs_baseline: f64,
}

/// Generate text with specified word count
fn generate_text_with_words(word_count: usize) -> String {
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
fn generate_batch(batch_size: usize, target_words: usize, variable: bool) -> Vec<String> {
    let mut rng = rand::thread_rng();

    (0..batch_size)
        .map(|_| {
            let words = if variable {
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
fn run_single_config<F>(
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

fn load_vocabulary(tokenizer_path: &str) -> Result<(ahash::AHashMap<String, u32>, WordPieceConfig)> {
    let content = std::fs::read_to_string(tokenizer_path)
        .with_context(|| format!("Failed to read tokenizer file: {}", tokenizer_path))?;

    let json: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| "Failed to parse tokenizer.json")?;

    let vocab_obj = json
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow::anyhow!("Missing model.vocab in tokenizer.json"))?;

    let mut token_to_id = ahash::AHashMap::new();
    for (token, id) in vocab_obj {
        let id = id.as_u64()
            .ok_or_else(|| anyhow::anyhow!("Invalid token ID for '{}'", token))? as u32;
        token_to_id.insert(token.clone(), id);
    }

    let mut config = WordPieceConfig::default();
    if let Some(normalizer) = json.get("normalizer") {
        config.do_lower_case = normalizer.get("lowercase").and_then(|v| v.as_bool()).unwrap_or(true);
        config.strip_accents = normalizer.get("strip_accents").and_then(|v| v.as_bool()).unwrap_or(true);
        config.tokenize_chinese_chars = normalizer.get("handle_chinese_chars").and_then(|v| v.as_bool()).unwrap_or(true);
    }

    if let Some(model) = json.get("model") {
        if let Some(unk) = model.get("unk_token").and_then(|v| v.as_str()) {
            config.unk_token = unk.to_string();
        }
        if let Some(prefix) = model.get("continuing_subword_prefix").and_then(|v| v.as_str()) {
            config.continuing_subword_prefix = prefix.to_string();
        }
    }

    Ok((token_to_id, config))
}

fn print_results_table(results: &[ExtensiveResult], baseline_backend: &str) {
    println!("\n{}", "=".repeat(150));
    println!("{:^150}", "EXTENSIVE SINGLE-CORE BENCHMARK RESULTS");
    println!("{}\n", "=".repeat(150));

    // Group by batch_size, seq_length, variable
    let mut grouped: BTreeMap<(usize, usize, bool), Vec<&ExtensiveResult>> = BTreeMap::new();

    for result in results {
        grouped
            .entry((result.batch_size, result.seq_length, result.variable_length))
            .or_default()
            .push(result);
    }

    println!("{:-<150}", "");
    println!(
        "{:>8} | {:>8} | {:>6} | {:>18} | {:>14} | {:>12} | {:>12} | {:>12} | {:>10}",
        "Batch", "SeqLen", "Var", "Backend", "Tokens/sec", "Texts/sec", "Mean(μs)", "P99(μs)", "Speedup"
    );
    println!("{:-<150}", "");

    for ((batch_size, seq_len, variable), group) in &grouped {
        let baseline_throughput = group
            .iter()
            .find(|r| r.backend == baseline_backend)
            .map(|r| r.tokens_per_sec)
            .unwrap_or(1.0);

        for (i, result) in group.iter().enumerate() {
            let speedup = result.tokens_per_sec / baseline_throughput;
            let var_str = if *variable { "Y" } else { "N" };

            let (b_str, s_str, v_str) = if i == 0 {
                (format!("{}", batch_size), format!("{}", seq_len), var_str.to_string())
            } else {
                ("".to_string(), "".to_string(), "".to_string())
            };

            println!(
                "{:>8} | {:>8} | {:>6} | {:>18} | {:>14.0} | {:>12.0} | {:>12.1} | {:>12.1} | {:>10.2}x",
                b_str, s_str, v_str,
                result.backend,
                result.tokens_per_sec,
                result.texts_per_sec,
                result.mean_latency_us,
                result.p99_latency_us,
                speedup
            );
        }
        println!("{:-<150}", "");
    }
}

fn calculate_summaries(results: &[ExtensiveResult], baseline_backend: &str) -> Vec<BackendSummary> {
    let mut backend_results: HashMap<String, Vec<&ExtensiveResult>> = HashMap::new();

    for result in results {
        backend_results
            .entry(result.backend.clone())
            .or_default()
            .push(result);
    }

    let baseline_map: HashMap<(usize, usize, bool), f64> = results
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

    summaries.sort_by(|a, b| b.avg_tokens_per_sec.partial_cmp(&a.avg_tokens_per_sec).unwrap());
    summaries
}

fn print_summary_table(summaries: &[BackendSummary]) {
    println!("\n{}", "=".repeat(110));
    println!("{:^110}", "BACKEND SUMMARY");
    println!("{}\n", "=".repeat(110));

    println!(
        "{:<20} | {:>8} | {:>14} | {:>14} | {:>14} | {:>12} | {:>12}",
        "Backend", "Configs", "Avg Tok/s", "Max Tok/s", "Min Tok/s", "Avg Speedup", "Max Speedup"
    );
    println!("{:-<110}", "");

    for s in summaries {
        println!(
            "{:<20} | {:>8} | {:>14.0} | {:>14.0} | {:>14.0} | {:>12.2}x | {:>12.2}x",
            s.backend,
            s.total_configurations,
            s.avg_tokens_per_sec,
            s.max_tokens_per_sec,
            s.min_tokens_per_sec,
            s.avg_speedup_vs_baseline,
            s.max_speedup_vs_baseline
        );
    }
    println!("{:-<110}", "");
}

/// Estimate memory usage for a given configuration in bytes
/// Conservative estimate: ~8 bytes per word (avg word + space) for input
/// Plus ~4 bytes per token output, assuming ~1.5 tokens per word
fn estimate_memory_bytes(batch_size: usize, seq_length_words: usize) -> usize {
    let input_bytes = batch_size * seq_length_words * 8; // ~8 bytes per word
    let output_bytes = batch_size * seq_length_words * 6; // ~1.5 tokens × 4 bytes
    let buffer_overhead = (input_bytes + output_bytes) / 2; // conservative overhead
    input_bytes + output_bytes + buffer_overhead
}

fn export_csv(results: &[ExtensiveResult], path: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;

    writeln!(
        file,
        "backend,batch_size,seq_length,variable_length,iterations,total_tokens,total_texts,\
         mean_latency_us,p50_latency_us,p95_latency_us,p99_latency_us,min_latency_us,max_latency_us,\
         tokens_per_sec,texts_per_sec,batches_per_sec,us_per_token,us_per_text"
    )?;

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

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("{}", "=".repeat(80).green());
    println!("{:^80}", "EXTENSIVE SINGLE-CORE TOKENIZER BENCHMARK".bold());
    println!("{}", "=".repeat(80).green());
    println!();

    let batch_sizes = if cli.quick {
        vec![1, 10, 100, 1000]
    } else {
        cli.batch_sizes.unwrap_or_else(|| vec![1, 10, 50, 100, 500, 1000, 2000, 5000])
    };

    let seq_lengths = if cli.quick {
        vec![100, 1000, 10000]
    } else {
        cli.seq_lengths.unwrap_or_else(|| vec![100, 500, 1000, 2000, 5000, 10000, 25000])
    };

    let max_memory_bytes = (cli.max_memory_gb * 1024.0 * 1024.0 * 1024.0) as usize;

    let iterations = if cli.quick { 5 } else { cli.iterations };
    let warmup = if cli.quick { 2 } else { cli.warmup };
    let max_duration = Duration::from_secs(cli.max_duration);

    let skipped: HashSet<String> = cli.skip
        .unwrap_or_default()
        .into_iter()
        .map(|s| s.to_lowercase())
        .collect();

    println!("Configuration:");
    println!("  Tokenizer:     {}", cli.tokenizer.cyan());
    println!("  Batch sizes:   {:?}", batch_sizes);
    println!("  Seq lengths:   {:?}", seq_lengths);
    println!("  Variable len:  {}", cli.include_variable);
    println!("  Iterations:    {}", iterations);
    println!("  Max duration:  {}s per config", cli.max_duration);
    println!("  Max memory:    {:.1} GB", cli.max_memory_gb);
    println!();

    println!("{}", "Loading tokenizers...".yellow());

    let (vocab_map, fast_config) = load_vocabulary(&cli.tokenizer)?;

    let special_tokens = SpecialTokens {
        unk_token: Some(fast_config.unk_token.clone()),
        cls_token: Some("[CLS]".to_string()),
        sep_token: Some("[SEP]".to_string()),
        ..Default::default()
    };

    let mut all_results: Vec<ExtensiveResult> = Vec::new();

    let variable_modes: Vec<bool> = if cli.include_variable {
        vec![false, true]
    } else {
        vec![false]
    };

    let total_configs = batch_sizes.len() * seq_lengths.len() * variable_modes.len();
    let active_backends: Vec<&str> = ["blazetext", "budtiktok", "budtiktok-hyper", "huggingface"]
        .iter()
        .filter(|b| !skipped.contains(&b.to_lowercase()))
        .copied()
        .collect();

    println!("Running {} backends × {} configurations = {} total tests\n",
             active_backends.len(), total_configs, active_backends.len() * total_configs);

    let mut skipped_configs = 0usize;

    // ===== BLAZETEXT =====
    #[cfg(feature = "blazetext")]
    if !skipped.contains("blazetext") {
        println!("{}", "▶ BlazeText".cyan().bold());
        let blazetext = BertWordPieceTokenizer::from_file(&cli.tokenizer)
            .map_err(|e| anyhow::anyhow!("Failed to load BlazeText: {}", e))?;

        for &variable in &variable_modes {
            for &seq_len in &seq_lengths {
                for &batch_size in &batch_sizes {
                    let est_mem = estimate_memory_bytes(batch_size, seq_len);
                    if est_mem > max_memory_bytes {
                        println!("  batch={:<6} seq={:<6} var={:<5} ... {} (est {:.1} GB > {:.1} GB limit)",
                                 batch_size, seq_len, variable,
                                 "SKIPPED".yellow(),
                                 est_mem as f64 / 1024.0 / 1024.0 / 1024.0,
                                 cli.max_memory_gb);
                        skipped_configs += 1;
                        continue;
                    }

                    print!("  batch={:<6} seq={:<6} var={:<5} ... ", batch_size, seq_len, variable);
                    std::io::stdout().flush()?;

                    let result = run_single_config(
                        "BlazeText",
                        batch_size,
                        seq_len,
                        variable,
                        iterations,
                        warmup,
                        max_duration,
                        |texts| {
                            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                            blazetext.encode_batch_fast(&refs, true)
                        },
                    );

                    println!("{:.0} tok/s", result.tokens_per_sec);
                    all_results.push(result);
                }
            }
        }
        println!();
    }

    // ===== BUDTIKTOK (WordPiece) =====
    if !skipped.contains("budtiktok") {
        println!("{}", "▶ BudTikTok (WordPiece)".cyan().bold());
        let vocab2 = Vocabulary::new(vocab_map.clone(), special_tokens.clone());
        let budtiktok = WordPieceTokenizer::new(vocab2, fast_config.clone());

        for &variable in &variable_modes {
            for &seq_len in &seq_lengths {
                for &batch_size in &batch_sizes {
                    let est_mem = estimate_memory_bytes(batch_size, seq_len);
                    if est_mem > max_memory_bytes {
                        println!("  batch={:<6} seq={:<6} var={:<5} ... {} (est {:.1} GB > {:.1} GB limit)",
                                 batch_size, seq_len, variable,
                                 "SKIPPED".yellow(),
                                 est_mem as f64 / 1024.0 / 1024.0 / 1024.0,
                                 cli.max_memory_gb);
                        skipped_configs += 1;
                        continue;
                    }

                    print!("  batch={:<6} seq={:<6} var={:<5} ... ", batch_size, seq_len, variable);
                    std::io::stdout().flush()?;

                    let result = run_single_config(
                        "BudTikTok",
                        batch_size,
                        seq_len,
                        variable,
                        iterations,
                        warmup,
                        max_duration,
                        |texts| {
                            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                            let encodings = budtiktok.encode_batch(&refs, true).unwrap_or_default();
                            encodings.iter().map(|e| e.get_ids().to_vec()).collect()
                        },
                    );

                    println!("{:.0} tok/s", result.tokens_per_sec);
                    all_results.push(result);
                }
            }
        }
        println!();
    }

    // ===== BUDTIKTOK-HYPER (with SIMD) =====
    if !skipped.contains("budtiktok-hyper") {
        println!("{}", "▶ BudTikTok-Hyper (SIMD optimized)".cyan().bold());
        let vocab3 = Vocabulary::new(vocab_map.clone(), special_tokens.clone());
        let hyper_config = HyperConfig {
            continuing_subword_prefix: fast_config.continuing_subword_prefix.clone(),
            max_input_chars_per_word: fast_config.max_input_chars_per_word,
            unk_token: fast_config.unk_token.clone(),
            do_lower_case: fast_config.do_lower_case,
            strip_accents: fast_config.strip_accents,
            tokenize_chinese_chars: fast_config.tokenize_chinese_chars,
            hash_table_bits: 14,
        };
        let hyper = HyperWordPieceTokenizer::new(vocab3, hyper_config);

        for &variable in &variable_modes {
            for &seq_len in &seq_lengths {
                for &batch_size in &batch_sizes {
                    let est_mem = estimate_memory_bytes(batch_size, seq_len);
                    if est_mem > max_memory_bytes {
                        println!("  batch={:<6} seq={:<6} var={:<5} ... {} (est {:.1} GB > {:.1} GB limit)",
                                 batch_size, seq_len, variable,
                                 "SKIPPED".yellow(),
                                 est_mem as f64 / 1024.0 / 1024.0 / 1024.0,
                                 cli.max_memory_gb);
                        skipped_configs += 1;
                        continue;
                    }

                    print!("  batch={:<6} seq={:<6} var={:<5} ... ", batch_size, seq_len, variable);
                    std::io::stdout().flush()?;

                    let result = run_single_config(
                        "BudTikTok-Hyper",
                        batch_size,
                        seq_len,
                        variable,
                        iterations,
                        warmup,
                        max_duration,
                        |texts| {
                            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                            hyper.encode_batch_with_special(&refs)
                        },
                    );

                    println!("{:.0} tok/s", result.tokens_per_sec);
                    all_results.push(result);
                }
            }
        }
        println!();
    }

    // ===== HUGGINGFACE =====
    if !skipped.contains("huggingface") {
        println!("{}", "▶ HuggingFace Tokenizers".cyan().bold());
        let hf_tokenizer = HfTokenizer::from_file(&cli.tokenizer)
            .map_err(|e| anyhow::anyhow!("Failed to load HuggingFace tokenizer: {}", e))?;

        for &variable in &variable_modes {
            for &seq_len in &seq_lengths {
                for &batch_size in &batch_sizes {
                    let est_mem = estimate_memory_bytes(batch_size, seq_len);
                    if est_mem > max_memory_bytes {
                        println!("  batch={:<6} seq={:<6} var={:<5} ... {} (est {:.1} GB > {:.1} GB limit)",
                                 batch_size, seq_len, variable,
                                 "SKIPPED".yellow(),
                                 est_mem as f64 / 1024.0 / 1024.0 / 1024.0,
                                 cli.max_memory_gb);
                        skipped_configs += 1;
                        continue;
                    }

                    print!("  batch={:<6} seq={:<6} var={:<5} ... ", batch_size, seq_len, variable);
                    std::io::stdout().flush()?;

                    let result = run_single_config(
                        "HuggingFace",
                        batch_size,
                        seq_len,
                        variable,
                        iterations,
                        warmup,
                        max_duration,
                        |texts| {
                            let encodings = hf_tokenizer.encode_batch(texts.to_vec(), true).unwrap();
                            encodings.iter().map(|e| e.get_ids().to_vec()).collect()
                        },
                    );

                    println!("{:.0} tok/s", result.tokens_per_sec);
                    all_results.push(result);
                }
            }
        }
        println!();
    }

    // ===== CUDA GPU =====
    #[cfg(feature = "cuda")]
    if cli.cuda && !skipped.contains("cuda") {
        use budtiktok_gpu::{GpuWordPieceTokenizer, GpuWordPieceConfig};
        use budtiktok_gpu::cuda::{CudaContext, is_cuda_available};
        use std::sync::Arc;

        if is_cuda_available() {
            println!("{}", "▶ BudTikTok-CUDA (GPU)".cyan().bold());

            let ctx = Arc::new(CudaContext::new(0)
                .map_err(|e| anyhow::anyhow!("Failed to create CUDA context: {}", e))?);

            println!("  GPU: {}", ctx.device_info().name);

            let gpu_vocab: Vec<(&str, u32)> = vocab_map.iter()
                .map(|(k, v)| (k.as_str(), *v))
                .collect();

            let gpu_config = GpuWordPieceConfig {
                continuation_prefix: fast_config.continuing_subword_prefix.clone(),
                max_word_length: fast_config.max_input_chars_per_word,
                unk_token: fast_config.unk_token.clone(),
                do_lower_case: fast_config.do_lower_case,
                min_gpu_batch_size: 8,
                max_batch_size: 4096,
            };

            let gpu_tokenizer = GpuWordPieceTokenizer::new(ctx.clone(), &gpu_vocab, gpu_config)
                .map_err(|e| anyhow::anyhow!("Failed to create GPU tokenizer: {}", e))?;

            for &variable in &variable_modes {
                for &seq_len in &seq_lengths {
                    for &batch_size in &batch_sizes {
                        let est_mem = estimate_memory_bytes(batch_size, seq_len);
                        if est_mem > max_memory_bytes {
                            println!("  batch={:<6} seq={:<6} var={:<5} ... {} (est {:.1} GB > {:.1} GB limit)",
                                     batch_size, seq_len, variable,
                                     "SKIPPED".yellow(),
                                     est_mem as f64 / 1024.0 / 1024.0 / 1024.0,
                                     cli.max_memory_gb);
                            skipped_configs += 1;
                            continue;
                        }

                        print!("  batch={:<6} seq={:<6} var={:<5} ... ", batch_size, seq_len, variable);
                        std::io::stdout().flush()?;

                        let result = run_single_config(
                            "BudTikTok-CUDA",
                            batch_size,
                            seq_len,
                            variable,
                            iterations,
                            warmup,
                            max_duration,
                            |texts| {
                                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                                gpu_tokenizer.encode_batch_with_special(&refs).unwrap_or_else(|_| {
                                    refs.iter().map(|_| vec![0u32]).collect()
                                })
                            },
                        );

                        println!("{:.0} tok/s", result.tokens_per_sec);
                        all_results.push(result);
                    }
                }
            }
            println!();
        } else {
            println!("{}", "⚠ CUDA not available, skipping GPU benchmarks".yellow());
        }
    }

    // ===== CUDA GPU-NATIVE (Full GPU Pipeline) =====
    #[cfg(feature = "cuda")]
    if cli.cuda && !skipped.contains("cuda-native") {
        use budtiktok_gpu::{GpuNativeTokenizer, GpuNativeConfig};
        use budtiktok_gpu::cuda::{CudaContext, is_cuda_available};
        use std::sync::Arc;

        if is_cuda_available() {
            println!("{}", "▶ BudTikTok-CUDA-Native (Full GPU Pipeline)".cyan().bold());

            let ctx = Arc::new(CudaContext::new(0)
                .map_err(|e| anyhow::anyhow!("Failed to create CUDA context: {}", e))?);

            println!("  GPU: {} (Full pipeline on GPU)", ctx.device_info().name);

            // Use smaller buffer sizes for GPU-native to reduce memory transfer overhead
            // These will be sufficient for most benchmark configurations
            let native_config = GpuNativeConfig {
                max_seq_bytes: 131072,  // 128KB per sequence
                max_tokens_per_seq: 2048,
                max_batch_size: 128,    // Smaller batches, will process in chunks
                do_lower_case: fast_config.do_lower_case,
                continuation_prefix: fast_config.continuing_subword_prefix.clone(),
                max_word_chars: fast_config.max_input_chars_per_word,
                block_size: 256,
            };

            // Convert AHashMap to HashMap for GpuNativeTokenizer
            let std_vocab: std::collections::HashMap<String, u32> = vocab_map.iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect();

            match GpuNativeTokenizer::new(ctx.clone(), &std_vocab, native_config) {
                Ok(mut gpu_native) => {
                    for &variable in &variable_modes {
                        for &seq_len in &seq_lengths {
                            for &batch_size in &batch_sizes {
                                // GPU-native has different memory characteristics
                                // Max batch is limited by GPU memory and config
                                if batch_size > 128 {
                                    println!("  batch={:<6} seq={:<6} var={:<5} ... {} (batch > 128 GPU limit)",
                                             batch_size, seq_len, variable,
                                             "SKIPPED".yellow());
                                    skipped_configs += 1;
                                    continue;
                                }

                                let est_mem = estimate_memory_bytes(batch_size, seq_len);
                                if est_mem > max_memory_bytes {
                                    println!("  batch={:<6} seq={:<6} var={:<5} ... {} (est {:.1} GB > {:.1} GB limit)",
                                             batch_size, seq_len, variable,
                                             "SKIPPED".yellow(),
                                             est_mem as f64 / 1024.0 / 1024.0 / 1024.0,
                                             cli.max_memory_gb);
                                    skipped_configs += 1;
                                    continue;
                                }

                                print!("  batch={:<6} seq={:<6} var={:<5} ... ", batch_size, seq_len, variable);
                                std::io::stdout().flush()?;

                                let texts_batch: Vec<String> = generate_batch(batch_size, seq_len, variable);
                                let text_refs: Vec<&str> = texts_batch.iter().map(|s| s.as_str()).collect();

                                // Warmup
                                for _ in 0..warmup {
                                    let _ = gpu_native.encode_batch(&text_refs);
                                }
                                let _ = ctx.synchronize();

                                // Benchmark
                                let mut latencies = Vec::with_capacity(iterations);
                                let mut total_tokens = 0usize;
                                let start_time = Instant::now();
                                let mut actual_iterations = 0usize;

                                for _ in 0..iterations {
                                    if start_time.elapsed() > max_duration {
                                        break;
                                    }

                                    let iter_start = Instant::now();
                                    match gpu_native.encode_batch(&text_refs) {
                                        Ok(results) => {
                                            let _ = ctx.synchronize();
                                            let elapsed = iter_start.elapsed();
                                            latencies.push(elapsed.as_micros() as f64);
                                            for result in &results {
                                                total_tokens += result.len();
                                            }
                                            actual_iterations += 1;
                                        }
                                        Err(e) => {
                                            println!("{} ({})", "ERROR".red(), e);
                                            break;
                                        }
                                    }
                                }

                                if actual_iterations > 0 {
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

                                    let result = ExtensiveResult {
                                        backend: "BudTikTok-CUDA-Native".to_string(),
                                        batch_size,
                                        seq_length: seq_len,
                                        variable_length: variable,
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
                                    };

                                    println!("{:.0} tok/s", result.tokens_per_sec);
                                    all_results.push(result);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("  {} Failed to create GPU-native tokenizer: {}", "⚠".yellow(), e);
                }
            }
            println!();
        }
    }

    #[cfg(not(feature = "cuda"))]
    if cli.cuda {
        println!("{}", "⚠ CUDA support not compiled in. Rebuild with --features cuda".yellow());
    }

    if skipped_configs > 0 {
        println!("{} {} configurations skipped due to memory limits ({:.1} GB max)",
                 "⚠".yellow(), skipped_configs, cli.max_memory_gb);
        println!("Use --max-memory-gb to increase limit (requires more RAM)\n");
    }

    // Print results
    print_results_table(&all_results, "BlazeText");

    // Print summary
    let summaries = calculate_summaries(&all_results, "BlazeText");
    print_summary_table(&summaries);

    // Export to CSV if requested
    if let Some(output_path) = cli.output {
        export_csv(&all_results, output_path.to_str().unwrap())?;
        println!("\n{} Results exported to: {}", "✓".green(), output_path.display());
    }

    // Print quick comparison
    println!("\n{}", "=".repeat(80).green());
    println!("{:^80}", "QUICK COMPARISON vs BlazeText");
    println!("{}", "=".repeat(80).green());

    for summary in &summaries {
        let emoji = if summary.avg_speedup_vs_baseline > 1.0 { "🚀" } else { "🐢" };
        println!(
            "{} {:<20}: {:.2}x average speedup, {:.2}x max speedup",
            emoji,
            summary.backend,
            summary.avg_speedup_vs_baseline,
            summary.max_speedup_vs_baseline
        );
    }

    Ok(())
}
