//! Comprehensive benchmark comparing HuggingFace tokenizers vs BudTikTok
//!
//! Measures:
//! - Tokenization latency (single thread, multi-thread)
//! - Throughput at various concurrency levels
//! - Memory consumption
//! - Token accuracy comparison

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use budtiktok_hf_compat::Tokenizer as BudTikTokTokenizer;
use tokenizers::Tokenizer as HFTokenizer;

// Sample texts of varying lengths
const SHORT_TEXTS: &[&str] = &[
    "Hello, world!",
    "The quick brown fox.",
    "Testing tokenization.",
    "Machine learning rocks!",
    "Natural language processing.",
];

const MEDIUM_TEXTS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog. This is a classic pangram used for testing.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Natural language processing focuses on the interaction between computers and humans using natural language.",
    "Deep learning neural networks have multiple layers that progressively extract higher-level features from input.",
    "Tokenization is the process of breaking down text into smaller units called tokens for processing.",
];

const LONG_TEXTS: &[&str] = &[
    "Artificial intelligence has transformed numerous industries over the past decade. From healthcare to finance, from transportation to entertainment, AI systems are now integral to how we live and work. Machine learning algorithms can now diagnose diseases, predict market trends, drive cars, and create art. The rapid advancement of deep learning, particularly transformer-based models, has accelerated this transformation significantly.",
    "The history of computing is a fascinating journey from mechanical calculators to quantum computers. Charles Babbage conceived the first programmable computer in the 1830s, though it was never completed. Ada Lovelace, often considered the first programmer, wrote algorithms for Babbage's Analytical Engine. The electronic computer era began in the 1940s with machines like ENIAC. The invention of the transistor in 1947 revolutionized electronics, leading to smaller and faster computers.",
];

#[derive(Debug, Clone)]
struct BenchmarkResult {
    name: String,
    concurrency: usize,
    num_requests: usize,
    total_time_ms: f64,
    throughput_rps: f64,
    throughput_tokens_per_sec: f64,
    latency_mean_us: f64,
    latency_p50_us: f64,
    latency_p90_us: f64,
    latency_p99_us: f64,
    memory_kb: usize,
}

fn get_memory_usage() -> usize {
    // Read from /proc/self/statm on Linux
    if let Ok(content) = std::fs::read_to_string("/proc/self/statm") {
        if let Some(resident) = content.split_whitespace().nth(1) {
            if let Ok(pages) = resident.parse::<usize>() {
                return pages * 4; // Pages are typically 4KB
            }
        }
    }
    0
}

fn benchmark_hf_tokenizer(
    tokenizer: &HFTokenizer,
    texts: &[&str],
    concurrency: usize,
    num_requests: usize,
) -> BenchmarkResult {
    let total_tokens = Arc::new(AtomicU64::new(0));

    let mem_before = get_memory_usage();

    // Build request indices
    let request_indices: Vec<usize> = (0..num_requests).collect();

    let start = Instant::now();

    // Use rayon to parallelize - this uses a fixed thread pool
    // The "concurrency" parameter affects workload distribution, not thread count
    let latencies: Vec<f64> = if concurrency == 1 {
        // Single-threaded benchmark
        request_indices.iter().map(|&i| {
            let text = texts[i % texts.len()];
            let t0 = Instant::now();
            let encoding = tokenizer.encode(text, true).unwrap();
            total_tokens.fetch_add(encoding.get_ids().len() as u64, Ordering::Relaxed);
            t0.elapsed().as_micros() as f64
        }).collect()
    } else {
        // Multi-threaded benchmark using rayon's thread pool
        // This safely handles high concurrency by using a fixed thread pool
        let total_tokens_clone = Arc::clone(&total_tokens);

        request_indices.par_iter().map(|&i| {
            let text = texts[i % texts.len()];
            let t0 = Instant::now();
            let encoding = tokenizer.encode(text, true).unwrap();
            total_tokens_clone.fetch_add(encoding.get_ids().len() as u64, Ordering::Relaxed);
            t0.elapsed().as_micros() as f64
        }).collect()
    };

    let total_time = start.elapsed();
    let mem_after = get_memory_usage();

    // Calculate statistics
    let mut sorted_latencies = latencies.clone();
    sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let latency_mean = sorted_latencies.iter().sum::<f64>() / sorted_latencies.len() as f64;
    let latency_p50 = sorted_latencies[sorted_latencies.len() / 2];
    let latency_p90 = sorted_latencies[(sorted_latencies.len() as f64 * 0.9) as usize];
    let latency_p99 = sorted_latencies[(sorted_latencies.len() as f64 * 0.99) as usize];

    let total_time_ms = total_time.as_secs_f64() * 1000.0;
    let throughput_rps = num_requests as f64 / total_time.as_secs_f64();
    let tokens = total_tokens.load(Ordering::Relaxed);
    let throughput_tokens = tokens as f64 / total_time.as_secs_f64();

    BenchmarkResult {
        name: "HuggingFace".to_string(),
        concurrency,
        num_requests,
        total_time_ms,
        throughput_rps,
        throughput_tokens_per_sec: throughput_tokens,
        latency_mean_us: latency_mean,
        latency_p50_us: latency_p50,
        latency_p90_us: latency_p90,
        latency_p99_us: latency_p99,
        memory_kb: mem_after.saturating_sub(mem_before),
    }
}

fn benchmark_budtiktok_tokenizer(
    tokenizer: &BudTikTokTokenizer,
    texts: &[&str],
    concurrency: usize,
    num_requests: usize,
) -> BenchmarkResult {
    let total_tokens = Arc::new(AtomicU64::new(0));

    let mem_before = get_memory_usage();

    // Build request indices
    let request_indices: Vec<usize> = (0..num_requests).collect();

    let start = Instant::now();

    // Use rayon to parallelize - this uses a fixed thread pool (based on CPU cores)
    // This safely handles high concurrency without spawning excessive threads
    let latencies: Vec<f64> = if concurrency == 1 {
        // Single-threaded benchmark
        request_indices.iter().map(|&i| {
            let text = texts[i % texts.len()];
            let t0 = Instant::now();
            let encoding = tokenizer.encode(text, true).unwrap();
            total_tokens.fetch_add(encoding.get_ids().len() as u64, Ordering::Relaxed);
            t0.elapsed().as_micros() as f64
        }).collect()
    } else {
        // Multi-threaded benchmark using rayon's thread pool
        // This safely handles high concurrency by using a fixed thread pool
        let total_tokens_clone = Arc::clone(&total_tokens);

        request_indices.par_iter().map(|&i| {
            let text = texts[i % texts.len()];
            let t0 = Instant::now();
            let encoding = tokenizer.encode(text, true).unwrap();
            total_tokens_clone.fetch_add(encoding.get_ids().len() as u64, Ordering::Relaxed);
            t0.elapsed().as_micros() as f64
        }).collect()
    };

    let total_time = start.elapsed();
    let mem_after = get_memory_usage();

    // Calculate statistics
    let mut sorted_latencies = latencies.clone();
    sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let latency_mean = sorted_latencies.iter().sum::<f64>() / sorted_latencies.len() as f64;
    let latency_p50 = sorted_latencies[sorted_latencies.len() / 2];
    let latency_p90 = sorted_latencies[(sorted_latencies.len() as f64 * 0.9) as usize];
    let latency_p99 = sorted_latencies[(sorted_latencies.len() as f64 * 0.99) as usize];

    let total_time_ms = total_time.as_secs_f64() * 1000.0;
    let throughput_rps = num_requests as f64 / total_time.as_secs_f64();
    let tokens = total_tokens.load(Ordering::Relaxed);
    let throughput_tokens = tokens as f64 / total_time.as_secs_f64();

    BenchmarkResult {
        name: "BudTikTok".to_string(),
        concurrency,
        num_requests,
        total_time_ms,
        throughput_rps,
        throughput_tokens_per_sec: throughput_tokens,
        latency_mean_us: latency_mean,
        latency_p50_us: latency_p50,
        latency_p90_us: latency_p90,
        latency_p99_us: latency_p99,
        memory_kb: mem_after.saturating_sub(mem_before),
    }
}

fn compare_accuracy(hf: &HFTokenizer, bt: &BudTikTokTokenizer, texts: &[&str]) {
    println!("\n{:=<80}", "");
    println!(" ACCURACY COMPARISON");
    println!("{:=<80}", "");

    let mut exact_matches = 0;
    let mut mismatches = Vec::new();

    for text in texts {
        let hf_encoding = hf.encode(*text, true).unwrap();
        let bt_encoding = bt.encode(*text, true).unwrap();

        // Get raw IDs without padding (filter out padding tokens which are 0)
        let hf_ids_raw = hf_encoding.get_ids();
        // Filter HF padding: take until we hit a 0 (pad token)
        let hf_ids: Vec<u32> = hf_ids_raw.iter()
            .take_while(|&&id| id != 0 || hf_ids_raw[0] == 0) // Handle edge case where 0 is first token
            .cloned()
            .collect();
        // Alternative: use attention mask if available, but filtering 0s works for this benchmark
        let bt_ids = bt_encoding.get_ids();

        if hf_ids.as_slice() == bt_ids {
            exact_matches += 1;
        } else {
            mismatches.push((
                text.chars().take(40).collect::<String>(),
                hf_ids.len(),
                bt_ids.len(),
                format!("{:?}", &hf_ids[..hf_ids.len().min(10)]),
                format!("{:?}", &bt_ids[..bt_ids.len().min(10)]),
            ));
        }
    }

    println!("Total texts: {}", texts.len());
    println!("Exact matches: {} ({:.1}%)", exact_matches, exact_matches as f64 / texts.len() as f64 * 100.0);

    if !mismatches.is_empty() {
        println!("\nMismatches (first 5):");
        for (i, (text, hf_len, bt_len, hf_ids, bt_ids)) in mismatches.iter().take(5).enumerate() {
            println!("  {}. \"{}...\"", i + 1, text);
            println!("     HF: {} tokens - {}", hf_len, hf_ids);
            println!("     BT: {} tokens - {}", bt_len, bt_ids);
        }
    }
}

fn print_results_table(hf_results: &[BenchmarkResult], bt_results: &[BenchmarkResult]) {
    println!("\n{:=<120}", "");
    println!(" PERFORMANCE COMPARISON");
    println!("{:=<120}", "");

    // Header
    println!(
        "{:>6} | {:>12} | {:>12} | {:>8} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8}",
        "Conc", "HF RPS", "BT RPS", "Speedup",
        "HF P50(us)", "BT P50(us)", "HF P99(us)", "BT P99(us)", "Mem(KB)"
    );
    println!("{:-<120}", "");

    for (hf, bt) in hf_results.iter().zip(bt_results.iter()) {
        let speedup = bt.throughput_rps / hf.throughput_rps;
        let latency_improvement = (hf.latency_p50_us - bt.latency_p50_us) / hf.latency_p50_us * 100.0;

        println!(
            "{:>6} | {:>12.0} | {:>12.0} | {:>7.2}x | {:>10.1} | {:>10.1} | {:>10.1} | {:>10.1} | {:>8}",
            hf.concurrency,
            hf.throughput_rps,
            bt.throughput_rps,
            speedup,
            hf.latency_p50_us,
            bt.latency_p50_us,
            hf.latency_p99_us,
            bt.latency_p99_us,
            bt.memory_kb,
        );
    }

    // Summary
    println!("\n{:=<120}", "");
    println!(" SUMMARY");
    println!("{:=<120}", "");

    let avg_speedup: f64 = hf_results.iter().zip(bt_results.iter())
        .map(|(hf, bt)| bt.throughput_rps / hf.throughput_rps)
        .sum::<f64>() / hf_results.len() as f64;

    let avg_latency_improvement: f64 = hf_results.iter().zip(bt_results.iter())
        .map(|(hf, bt)| (hf.latency_p50_us - bt.latency_p50_us) / hf.latency_p50_us * 100.0)
        .sum::<f64>() / hf_results.len() as f64;

    println!("Average throughput speedup: {:.2}x", avg_speedup);
    println!("Average latency improvement: {:.1}%", avg_latency_improvement);
}

fn main() {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let tokenizer_path = args.get(1).map(|s| s.as_str()).unwrap_or(
        "/home/bud/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer.json"
    );

    println!("Loading tokenizers from: {}", tokenizer_path);

    // Load tokenizers
    let hf_tokenizer = HFTokenizer::from_file(tokenizer_path).expect("Failed to load HF tokenizer");
    let bt_tokenizer = BudTikTokTokenizer::from_file(tokenizer_path).expect("Failed to load BudTikTok tokenizer");

    // Prepare test texts
    let mut all_texts: Vec<&str> = Vec::new();
    all_texts.extend_from_slice(SHORT_TEXTS);
    all_texts.extend_from_slice(MEDIUM_TEXTS);
    all_texts.extend_from_slice(LONG_TEXTS);

    // Print system info
    println!("\n{:=<80}", "");
    println!(" SYSTEM INFORMATION");
    println!("{:=<80}", "");
    println!("CPU cores: {}", num_cpus::get());
    println!("Physical cores: {}", num_cpus::get_physical());

    // Check SIMD support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            println!("AVX2: Supported");
        }
        if is_x86_feature_detected!("avx512f") {
            println!("AVX-512: Supported");
        } else {
            println!("AVX-512: Not supported (using scalar/AVX2 fallback)");
        }
    }

    // Warm up
    println!("\nWarming up...");
    for text in all_texts.iter().take(100) {
        let _ = hf_tokenizer.encode(*text, true);
        let _ = bt_tokenizer.encode(*text, true);
    }

    // Accuracy comparison
    compare_accuracy(&hf_tokenizer, &bt_tokenizer, &all_texts);

    // Benchmark at different concurrency levels
    // Extended to include high concurrency (256, 512, 1000) to test stability
    let concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000];
    let requests_per_level = 10000;

    println!("\n{:=<80}", "");
    println!(" RUNNING BENCHMARKS");
    println!("{:=<80}", "");
    println!("Requests per concurrency level: {}", requests_per_level);

    let mut hf_results = Vec::new();
    let mut bt_results = Vec::new();

    for &conc in &concurrency_levels {
        print!("  c={:>3}: ", conc);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let hf_result = benchmark_hf_tokenizer(&hf_tokenizer, &all_texts, conc, requests_per_level);
        print!("HF={:.0} rps, ", hf_result.throughput_rps);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let bt_result = benchmark_budtiktok_tokenizer(&bt_tokenizer, &all_texts, conc, requests_per_level);
        println!("BT={:.0} rps ({:.2}x)", bt_result.throughput_rps, bt_result.throughput_rps / hf_result.throughput_rps);

        hf_results.push(hf_result);
        bt_results.push(bt_result);
    }

    // Print results table
    print_results_table(&hf_results, &bt_results);

    // Variable vs Fixed length comparison
    println!("\n{:=<80}", "");
    println!(" VARIABLE vs FIXED LENGTH INPUT");
    println!("{:=<80}", "");

    // Fixed length (short)
    let hf_short = benchmark_hf_tokenizer(&hf_tokenizer, SHORT_TEXTS, 16, 10000);
    let bt_short = benchmark_budtiktok_tokenizer(&bt_tokenizer, SHORT_TEXTS, 16, 10000);
    println!("Short texts (avg ~15 chars):");
    println!("  HF: {:.0} rps, P50={:.1}us", hf_short.throughput_rps, hf_short.latency_p50_us);
    println!("  BT: {:.0} rps, P50={:.1}us ({:.2}x faster)", bt_short.throughput_rps, bt_short.latency_p50_us, bt_short.throughput_rps / hf_short.throughput_rps);

    // Fixed length (medium)
    let hf_medium = benchmark_hf_tokenizer(&hf_tokenizer, MEDIUM_TEXTS, 16, 10000);
    let bt_medium = benchmark_budtiktok_tokenizer(&bt_tokenizer, MEDIUM_TEXTS, 16, 10000);
    println!("\nMedium texts (avg ~100 chars):");
    println!("  HF: {:.0} rps, P50={:.1}us", hf_medium.throughput_rps, hf_medium.latency_p50_us);
    println!("  BT: {:.0} rps, P50={:.1}us ({:.2}x faster)", bt_medium.throughput_rps, bt_medium.latency_p50_us, bt_medium.throughput_rps / hf_medium.throughput_rps);

    // Fixed length (long)
    let hf_long = benchmark_hf_tokenizer(&hf_tokenizer, LONG_TEXTS, 16, 10000);
    let bt_long = benchmark_budtiktok_tokenizer(&bt_tokenizer, LONG_TEXTS, 16, 10000);
    println!("\nLong texts (avg ~400 chars):");
    println!("  HF: {:.0} rps, P50={:.1}us", hf_long.throughput_rps, hf_long.latency_p50_us);
    println!("  BT: {:.0} rps, P50={:.1}us ({:.2}x faster)", bt_long.throughput_rps, bt_long.latency_p50_us, bt_long.throughput_rps / hf_long.throughput_rps);

    // Variable length
    let hf_var = benchmark_hf_tokenizer(&hf_tokenizer, &all_texts, 16, 10000);
    let bt_var = benchmark_budtiktok_tokenizer(&bt_tokenizer, &all_texts, 16, 10000);
    println!("\nVariable length (mixed):");
    println!("  HF: {:.0} rps, P50={:.1}us", hf_var.throughput_rps, hf_var.latency_p50_us);
    println!("  BT: {:.0} rps, P50={:.1}us ({:.2}x faster)", bt_var.throughput_rps, bt_var.latency_p50_us, bt_var.throughput_rps / hf_var.throughput_rps);

    println!("\n{:=<80}", "");
    println!(" BENCHMARK COMPLETE");
    println!("{:=<80}", "");
}
