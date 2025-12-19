//! Comprehensive SIMD vs Scalar Benchmark for BPE and WordPiece
//!
//! Tests all combinations:
//! - BPE: Scalar/SIMD × Single-core/Multi-core
//! - WordPiece: Scalar/SIMD × Single-core/Multi-core
//!
//! Run with: cargo run --example simd_vs_scalar_benchmark --release

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};
use ahash::AHashMap;
use rayon::prelude::*;

// BPE imports
use budtiktok_core::bpe_linear::OptimizedBpeEncoder;

// WordPiece imports
use budtiktok_core::wordpiece::{WordPieceConfig, WordPieceTokenizer};
use budtiktok_core::wordpiece_hyper::{HyperConfig, HyperWordPieceTokenizer};
use budtiktok_core::vocab::{Vocabulary, SpecialTokens};
use budtiktok_core::tokenizer::Tokenizer;

// NUMA support
use budtiktok_core::{NumaTopology, set_thread_affinity};

const WORKSPACE: &str = "/home/bud/Desktop/latentbud/budtiktok";

#[derive(Clone, Copy)]
struct BenchResult {
    time_secs: f64,
    throughput_mbs: f64,
    tokens: usize,
    threads: usize,
}

impl BenchResult {
    fn speedup_over(&self, baseline: &BenchResult) -> f64 {
        baseline.time_secs / self.time_secs
    }

    fn parallel_efficiency(&self, single_core: &BenchResult) -> f64 {
        if self.threads <= 1 {
            100.0
        } else {
            (single_core.time_secs / self.time_secs) / self.threads as f64 * 100.0
        }
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║           COMPREHENSIVE SIMD vs SCALAR BENCHMARK - BPE & WORDPIECE               ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

    // Detect system info
    let num_cpus = num_cpus::get();
    let numa = NumaTopology::detect();
    println!("System: {} CPUs, {} NUMA nodes", num_cpus, numa.num_nodes);
    println!();

    // Run BPE benchmarks
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                                   BPE BENCHMARKS                                  ");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    let bpe_results = run_bpe_benchmarks(&numa);

    println!("\n");

    // Run WordPiece benchmarks
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("                                WORDPIECE BENCHMARKS                               ");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    let wp_results = run_wordpiece_benchmarks(&numa);

    // Print summary
    println!("\n");
    print_summary(&bpe_results, &wp_results);
}

fn run_bpe_benchmarks(numa: &NumaTopology) -> (BenchResult, BenchResult, BenchResult, BenchResult) {
    // Load GPT-2 vocab and merges
    let vocab_path = format!("{}/benchmark_data/gpt2/vocab.json", WORKSPACE);
    let merges_path = format!("{}/benchmark_data/gpt2/merges.txt", WORKSPACE);
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", WORKSPACE);

    println!("Loading GPT-2 vocabulary...");
    let vocab_content = fs::read_to_string(&vocab_path).expect("Failed to read vocab.json");
    let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_content).expect("Failed to parse vocab");
    let vocab_ahash: AHashMap<String, u32> = vocab.iter().map(|(k, v)| (k.clone(), *v)).collect();

    println!("Loading merge rules ({} merges)...", 50257);
    let merge_content = fs::read_to_string(&merges_path).expect("Failed to read merges.txt");
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

    println!("Creating OptimizedBpeEncoder...");
    let encoder = OptimizedBpeEncoder::new(vocab_ahash, merges, "<|endoftext|>");

    println!("Loading 1GB dataset...");
    let file = File::open(&data_path).expect("Failed to open dataset");
    let reader = BufReader::new(file);

    let documents: Vec<String> = reader.lines()
        .filter_map(|line| {
            let line = line.ok()?;
            let json: serde_json::Value = serde_json::from_str(&line).ok()?;
            json["text"].as_str().map(|s| s.to_string())
        })
        .collect();

    let total_bytes: usize = documents.iter().map(|d| d.len()).sum();
    println!("Loaded {} documents ({:.2} MB)\n", documents.len(), total_bytes as f64 / 1024.0 / 1024.0);

    let num_threads = rayon::current_num_threads();

    // Warmup
    println!("Warming up...");
    for doc in documents.iter().take(100) {
        let _ = encoder.encode(doc);
        let _ = encoder.encode_scalar(doc);
    }

    // -------------------------------------------------------------------------
    // Test 1: Scalar Single-Core
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ BPE Scalar Single-Core                                                          │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mut tokens_scalar_single = 0usize;
    for doc in &documents {
        tokens_scalar_single += encoder.encode_scalar(doc).len();
    }
    let time_scalar_single = start.elapsed();
    let throughput_scalar_single = total_bytes as f64 / time_scalar_single.as_secs_f64() / 1024.0 / 1024.0;

    let scalar_single = BenchResult {
        time_secs: time_scalar_single.as_secs_f64(),
        throughput_mbs: throughput_scalar_single,
        tokens: tokens_scalar_single,
        threads: 1,
    };

    println!("  Time:         {:.2}s", scalar_single.time_secs);
    println!("  Throughput:   {:.1} MB/s", scalar_single.throughput_mbs);
    println!("  Tokens:       {}", scalar_single.tokens);
    println!();

    // -------------------------------------------------------------------------
    // Test 2: SIMD Single-Core
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ BPE SIMD Single-Core                                                            │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mut tokens_simd_single = 0usize;
    for doc in &documents {
        tokens_simd_single += encoder.encode(doc).len();
    }
    let time_simd_single = start.elapsed();
    let throughput_simd_single = total_bytes as f64 / time_simd_single.as_secs_f64() / 1024.0 / 1024.0;

    let simd_single = BenchResult {
        time_secs: time_simd_single.as_secs_f64(),
        throughput_mbs: throughput_simd_single,
        tokens: tokens_simd_single,
        threads: 1,
    };

    println!("  Time:         {:.2}s", simd_single.time_secs);
    println!("  Throughput:   {:.1} MB/s", simd_single.throughput_mbs);
    println!("  Tokens:       {}", simd_single.tokens);
    println!("  SIMD Speedup: {:.2}x over scalar", simd_single.speedup_over(&scalar_single));
    println!();

    // Verify consistency
    assert_eq!(tokens_scalar_single, tokens_simd_single,
               "BPE: Scalar and SIMD token counts must match");

    // -------------------------------------------------------------------------
    // Test 3: Scalar Multi-Core
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ BPE Scalar Multi-Core ({} threads)                                       │", num_threads);
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    // Create NUMA-aware thread pool
    let thread_counter = AtomicUsize::new(0);
    let numa_clone = numa.clone();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .start_handler(move |_| {
            let idx = thread_counter.fetch_add(1, Ordering::SeqCst);
            let cpu = numa_clone.thread_to_cpu(idx);
            set_thread_affinity(cpu);
        })
        .build()
        .unwrap();

    let start = Instant::now();
    let tokens_scalar_multi: usize = pool.install(|| {
        documents.par_iter()
            .map(|doc| encoder.encode_scalar(doc).len())
            .sum()
    });
    let time_scalar_multi = start.elapsed();
    let throughput_scalar_multi = total_bytes as f64 / time_scalar_multi.as_secs_f64() / 1024.0 / 1024.0;

    let scalar_multi = BenchResult {
        time_secs: time_scalar_multi.as_secs_f64(),
        throughput_mbs: throughput_scalar_multi,
        tokens: tokens_scalar_multi,
        threads: num_threads,
    };

    println!("  Time:         {:.2}s", scalar_multi.time_secs);
    println!("  Throughput:   {:.1} MB/s", scalar_multi.throughput_mbs);
    println!("  Tokens:       {}", scalar_multi.tokens);
    println!("  Parallel Speedup: {:.2}x", scalar_multi.speedup_over(&scalar_single));
    println!("  Efficiency:   {:.1}%", scalar_multi.parallel_efficiency(&scalar_single));
    println!();

    // -------------------------------------------------------------------------
    // Test 4: SIMD Multi-Core
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ BPE SIMD Multi-Core ({} threads)                                         │", num_threads);
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let thread_counter = AtomicUsize::new(0);
    let numa_clone = numa.clone();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .start_handler(move |_| {
            let idx = thread_counter.fetch_add(1, Ordering::SeqCst);
            let cpu = numa_clone.thread_to_cpu(idx);
            set_thread_affinity(cpu);
        })
        .build()
        .unwrap();

    let start = Instant::now();
    let tokens_simd_multi: usize = pool.install(|| {
        documents.par_iter()
            .map(|doc| encoder.encode(doc).len())
            .sum()
    });
    let time_simd_multi = start.elapsed();
    let throughput_simd_multi = total_bytes as f64 / time_simd_multi.as_secs_f64() / 1024.0 / 1024.0;

    let simd_multi = BenchResult {
        time_secs: time_simd_multi.as_secs_f64(),
        throughput_mbs: throughput_simd_multi,
        tokens: tokens_simd_multi,
        threads: num_threads,
    };

    println!("  Time:         {:.2}s", simd_multi.time_secs);
    println!("  Throughput:   {:.1} MB/s", simd_multi.throughput_mbs);
    println!("  Tokens:       {}", simd_multi.tokens);
    println!("  SIMD Speedup:     {:.2}x over scalar multi-core", simd_multi.speedup_over(&scalar_multi));
    println!("  Parallel Speedup: {:.2}x over SIMD single-core", simd_multi.speedup_over(&simd_single));
    println!("  Efficiency:   {:.1}%", simd_multi.parallel_efficiency(&simd_single));
    println!("  Total Speedup:    {:.2}x over scalar single-core", simd_multi.speedup_over(&scalar_single));
    println!();

    (scalar_single, simd_single, scalar_multi, simd_multi)
}

fn run_wordpiece_benchmarks(numa: &NumaTopology) -> (BenchResult, BenchResult, BenchResult, BenchResult) {
    let tokenizer_path = format!("{}/test_data/bert-base-uncased/tokenizer.json", WORKSPACE);
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", WORKSPACE);

    println!("Loading BERT tokenizer...");
    let content = fs::read_to_string(&tokenizer_path).expect("Failed to read tokenizer.json");
    let json: serde_json::Value = serde_json::from_str(&content).expect("Failed to parse tokenizer.json");

    // Extract vocab
    let vocab_obj = json["model"]["vocab"].as_object().expect("Missing vocab");
    let mut token_to_id: AHashMap<String, u32> = AHashMap::new();
    for (token, id) in vocab_obj {
        token_to_id.insert(token.clone(), id.as_u64().unwrap() as u32);
    }

    println!("Vocabulary size: {}", token_to_id.len());

    // Create WordPiece config
    let mut config = WordPieceConfig::default();
    config.unk_token = "[UNK]".to_string();
    config.continuing_subword_prefix = "##".to_string();
    config.do_lower_case = true;
    config.strip_accents = true;

    // Create special tokens
    let special_tokens = SpecialTokens {
        unk_token: Some("[UNK]".to_string()),
        cls_token: Some("[CLS]".to_string()),
        sep_token: Some("[SEP]".to_string()),
        pad_token: Some("[PAD]".to_string()),
        mask_token: Some("[MASK]".to_string()),
        ..Default::default()
    };

    // Create tokenizers
    println!("Creating WordPieceTokenizer (scalar)...");
    let vocabulary = Vocabulary::new(token_to_id.clone(), special_tokens.clone());
    let scalar_tokenizer = WordPieceTokenizer::new(vocabulary.clone(), config.clone());

    println!("Creating HyperWordPieceTokenizer (SIMD)...");
    let hyper_config = HyperConfig {
        continuing_subword_prefix: "##".to_string(),
        max_input_chars_per_word: 100,
        unk_token: "[UNK]".to_string(),
        do_lower_case: true,
        strip_accents: true,
        tokenize_chinese_chars: true,
        hash_table_bits: 14,
    };
    let simd_tokenizer = HyperWordPieceTokenizer::new(vocabulary.clone(), hyper_config);

    println!("Loading 1GB dataset...");
    let file = File::open(&data_path).expect("Failed to open dataset");
    let reader = BufReader::new(file);

    let documents: Vec<String> = reader.lines()
        .filter_map(|line| {
            let line = line.ok()?;
            let json: serde_json::Value = serde_json::from_str(&line).ok()?;
            json["text"].as_str().map(|s| s.to_string())
        })
        .collect();

    let total_bytes: usize = documents.iter().map(|d| d.len()).sum();
    println!("Loaded {} documents ({:.2} MB)\n", documents.len(), total_bytes as f64 / 1024.0 / 1024.0);

    let num_threads = rayon::current_num_threads();

    // Warmup
    println!("Warming up...");
    for doc in documents.iter().take(100) {
        let _ = scalar_tokenizer.encode(doc, false);
        let _ = simd_tokenizer.encode_fast(doc);
    }

    // -------------------------------------------------------------------------
    // Test 1: Scalar Single-Core
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ WordPiece Scalar Single-Core                                                    │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mut tokens_scalar_single = 0usize;
    for doc in &documents {
        let encoding = scalar_tokenizer.encode(doc, false).unwrap();
        tokens_scalar_single += encoding.get_ids().len();
    }
    let time_scalar_single = start.elapsed();
    let throughput_scalar_single = total_bytes as f64 / time_scalar_single.as_secs_f64() / 1024.0 / 1024.0;

    let scalar_single = BenchResult {
        time_secs: time_scalar_single.as_secs_f64(),
        throughput_mbs: throughput_scalar_single,
        tokens: tokens_scalar_single,
        threads: 1,
    };

    println!("  Time:         {:.2}s", scalar_single.time_secs);
    println!("  Throughput:   {:.1} MB/s", scalar_single.throughput_mbs);
    println!("  Tokens:       {}", scalar_single.tokens);
    println!();

    // -------------------------------------------------------------------------
    // Test 2: SIMD Single-Core
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ WordPiece SIMD Single-Core                                                      │");
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let start = Instant::now();
    let mut tokens_simd_single = 0usize;
    for doc in &documents {
        tokens_simd_single += simd_tokenizer.encode_fast(doc).len();
    }
    let time_simd_single = start.elapsed();
    let throughput_simd_single = total_bytes as f64 / time_simd_single.as_secs_f64() / 1024.0 / 1024.0;

    let simd_single = BenchResult {
        time_secs: time_simd_single.as_secs_f64(),
        throughput_mbs: throughput_simd_single,
        tokens: tokens_simd_single,
        threads: 1,
    };

    println!("  Time:         {:.2}s", simd_single.time_secs);
    println!("  Throughput:   {:.1} MB/s", simd_single.throughput_mbs);
    println!("  Tokens:       {}", simd_single.tokens);
    println!("  SIMD Speedup: {:.2}x over scalar", simd_single.speedup_over(&scalar_single));
    println!();

    // -------------------------------------------------------------------------
    // Test 3: Scalar Multi-Core
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ WordPiece Scalar Multi-Core ({} threads)                                 │", num_threads);
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let thread_counter = AtomicUsize::new(0);
    let numa_clone = numa.clone();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .start_handler(move |_| {
            let idx = thread_counter.fetch_add(1, Ordering::SeqCst);
            let cpu = numa_clone.thread_to_cpu(idx);
            set_thread_affinity(cpu);
        })
        .build()
        .unwrap();

    let start = Instant::now();
    let tokens_scalar_multi: usize = pool.install(|| {
        documents.par_iter()
            .map(|doc| scalar_tokenizer.encode(doc, false).unwrap().get_ids().len())
            .sum()
    });
    let time_scalar_multi = start.elapsed();
    let throughput_scalar_multi = total_bytes as f64 / time_scalar_multi.as_secs_f64() / 1024.0 / 1024.0;

    let scalar_multi = BenchResult {
        time_secs: time_scalar_multi.as_secs_f64(),
        throughput_mbs: throughput_scalar_multi,
        tokens: tokens_scalar_multi,
        threads: num_threads,
    };

    println!("  Time:         {:.2}s", scalar_multi.time_secs);
    println!("  Throughput:   {:.1} MB/s", scalar_multi.throughput_mbs);
    println!("  Tokens:       {}", scalar_multi.tokens);
    println!("  Parallel Speedup: {:.2}x", scalar_multi.speedup_over(&scalar_single));
    println!("  Efficiency:   {:.1}%", scalar_multi.parallel_efficiency(&scalar_single));
    println!();

    // -------------------------------------------------------------------------
    // Test 4: SIMD Multi-Core
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ WordPiece SIMD Multi-Core ({} threads)                                   │", num_threads);
    println!("└─────────────────────────────────────────────────────────────────────────────────┘");

    let thread_counter = AtomicUsize::new(0);
    let numa_clone = numa.clone();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .start_handler(move |_| {
            let idx = thread_counter.fetch_add(1, Ordering::SeqCst);
            let cpu = numa_clone.thread_to_cpu(idx);
            set_thread_affinity(cpu);
        })
        .build()
        .unwrap();

    let start = Instant::now();
    let tokens_simd_multi: usize = pool.install(|| {
        documents.par_iter()
            .map(|doc| simd_tokenizer.encode_fast(doc).len())
            .sum()
    });
    let time_simd_multi = start.elapsed();
    let throughput_simd_multi = total_bytes as f64 / time_simd_multi.as_secs_f64() / 1024.0 / 1024.0;

    let simd_multi = BenchResult {
        time_secs: time_simd_multi.as_secs_f64(),
        throughput_mbs: throughput_simd_multi,
        tokens: tokens_simd_multi,
        threads: num_threads,
    };

    println!("  Time:         {:.2}s", simd_multi.time_secs);
    println!("  Throughput:   {:.1} MB/s", simd_multi.throughput_mbs);
    println!("  Tokens:       {}", simd_multi.tokens);
    println!("  SIMD Speedup:     {:.2}x over scalar multi-core", simd_multi.speedup_over(&scalar_multi));
    println!("  Parallel Speedup: {:.2}x over SIMD single-core", simd_multi.speedup_over(&simd_single));
    println!("  Efficiency:   {:.1}%", simd_multi.parallel_efficiency(&simd_single));
    println!("  Total Speedup:    {:.2}x over scalar single-core", simd_multi.speedup_over(&scalar_single));
    println!();

    (scalar_single, simd_single, scalar_multi, simd_multi)
}

fn print_summary(
    bpe: &(BenchResult, BenchResult, BenchResult, BenchResult),
    wp: &(BenchResult, BenchResult, BenchResult, BenchResult),
) {
    let (bpe_scalar_single, bpe_simd_single, bpe_scalar_multi, bpe_simd_multi) = bpe;
    let (wp_scalar_single, wp_simd_single, wp_scalar_multi, wp_simd_multi) = wp;

    println!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              SUMMARY COMPARISON                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

    println!("┌───────────────────────────────────────────────────────────────────────────────────┐");
    println!("│                               THROUGHPUT (MB/s)                                   │");
    println!("├─────────────────────┬──────────────────────┬──────────────────────────────────────┤");
    println!("│     Configuration   │         BPE          │           WordPiece                  │");
    println!("├─────────────────────┼──────────────────────┼──────────────────────────────────────┤");
    println!("│ Scalar Single-Core  │ {:>18.1}  │ {:>18.1}                    │",
             bpe_scalar_single.throughput_mbs, wp_scalar_single.throughput_mbs);
    println!("│ SIMD Single-Core    │ {:>18.1}  │ {:>18.1}                    │",
             bpe_simd_single.throughput_mbs, wp_simd_single.throughput_mbs);
    println!("│ Scalar Multi-Core   │ {:>18.1}  │ {:>18.1}                    │",
             bpe_scalar_multi.throughput_mbs, wp_scalar_multi.throughput_mbs);
    println!("│ SIMD Multi-Core     │ {:>18.1}  │ {:>18.1}                    │",
             bpe_simd_multi.throughput_mbs, wp_simd_multi.throughput_mbs);
    println!("└─────────────────────┴──────────────────────┴──────────────────────────────────────┘\n");

    println!("┌───────────────────────────────────────────────────────────────────────────────────┐");
    println!("│                                  SPEEDUPS                                         │");
    println!("├─────────────────────────────────────────────────────────────────────────────────┬─┤");
    println!("│                Metric                       │    BPE    │  WordPiece │          │");
    println!("├─────────────────────────────────────────────┼───────────┼────────────┤          │");
    println!("│ SIMD vs Scalar (Single-Core)                │  {:>6.2}x  │   {:>6.2}x   │          │",
             bpe_simd_single.speedup_over(bpe_scalar_single),
             wp_simd_single.speedup_over(wp_scalar_single));
    println!("│ SIMD vs Scalar (Multi-Core)                 │  {:>6.2}x  │   {:>6.2}x   │          │",
             bpe_simd_multi.speedup_over(bpe_scalar_multi),
             wp_simd_multi.speedup_over(wp_scalar_multi));
    println!("│ Scalar Multi vs Scalar Single               │  {:>6.2}x  │   {:>6.2}x   │          │",
             bpe_scalar_multi.speedup_over(bpe_scalar_single),
             wp_scalar_multi.speedup_over(wp_scalar_single));
    println!("│ SIMD Multi vs SIMD Single                   │  {:>6.2}x  │   {:>6.2}x   │          │",
             bpe_simd_multi.speedup_over(bpe_simd_single),
             wp_simd_multi.speedup_over(wp_simd_single));
    println!("├─────────────────────────────────────────────┼───────────┼────────────┤          │");
    println!("│ Total: SIMD Multi vs Scalar Single          │  {:>6.2}x  │   {:>6.2}x   │          │",
             bpe_simd_multi.speedup_over(bpe_scalar_single),
             wp_simd_multi.speedup_over(wp_scalar_single));
    println!("└─────────────────────────────────────────────┴───────────┴────────────┴──────────┘\n");

    println!("┌───────────────────────────────────────────────────────────────────────────────────┐");
    println!("│                           PARALLEL EFFICIENCY                                     │");
    println!("├─────────────────────────────────────────┬───────────┬────────────────────────────┤");
    println!("│            Configuration                │    BPE    │       WordPiece            │");
    println!("├─────────────────────────────────────────┼───────────┼────────────────────────────┤");
    println!("│ Scalar Multi-Core                       │  {:>6.1}%  │        {:>6.1}%             │",
             bpe_scalar_multi.parallel_efficiency(bpe_scalar_single),
             wp_scalar_multi.parallel_efficiency(wp_scalar_single));
    println!("│ SIMD Multi-Core                         │  {:>6.1}%  │        {:>6.1}%             │",
             bpe_simd_multi.parallel_efficiency(bpe_simd_single),
             wp_simd_multi.parallel_efficiency(wp_simd_single));
    println!("└─────────────────────────────────────────┴───────────┴────────────────────────────┘\n");

    println!("Analysis:");
    println!("- SIMD provides {:.1}x speedup for BPE and {:.1}x for WordPiece on single core",
             bpe_simd_single.speedup_over(bpe_scalar_single),
             wp_simd_single.speedup_over(wp_scalar_single));
    println!("- Multi-core scaling shows {:.1}% efficiency for BPE SIMD and {:.1}% for WordPiece SIMD",
             bpe_simd_multi.parallel_efficiency(bpe_simd_single),
             wp_simd_multi.parallel_efficiency(wp_simd_single));
    println!("- Total speedup (SIMD+parallel): {:.1}x for BPE, {:.1}x for WordPiece",
             bpe_simd_multi.speedup_over(bpe_scalar_single),
             wp_simd_multi.speedup_over(wp_scalar_single));
}
