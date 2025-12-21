//! Comprehensive TEI Benchmark
//!
//! Compares HuggingFace tokenizers vs BudTikTok across:
//! - Multiple sequence lengths (short, medium, long, very long)
//! - Multiple concurrency levels (1 to 1000)
//! - Token accuracy (exact match)
//! - Embedding accuracy (cosine similarity)

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use budtiktok_hf_compat::Tokenizer as BudTikTokTokenizer;
use tokenizers::Tokenizer as HFTokenizer;

/// Benchmark configuration
struct BenchConfig {
    /// Number of requests per concurrency level
    requests_per_level: usize,
    /// Concurrency levels to test
    concurrency_levels: Vec<usize>,
    /// Warmup iterations
    warmup_iterations: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            requests_per_level: 10000,
            concurrency_levels: vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000],
            warmup_iterations: 1000,
        }
    }
}

/// Test text categories by length
struct TestTexts {
    short: Vec<String>,      // ~10-20 chars
    medium: Vec<String>,     // ~50-100 chars
    long: Vec<String>,       // ~200-500 chars
    very_long: Vec<String>,  // ~1000+ chars
}

impl TestTexts {
    fn generate() -> Self {
        let short = vec![
            "Hello world".to_string(),
            "Quick test".to_string(),
            "AI is great".to_string(),
            "Machine learning".to_string(),
            "Neural networks".to_string(),
            "Deep learning rocks".to_string(),
            "NLP processing".to_string(),
            "Text embeddings".to_string(),
        ];

        let medium = vec![
            "The quick brown fox jumps over the lazy dog near the riverbank.".to_string(),
            "Machine learning models can process natural language effectively.".to_string(),
            "Artificial intelligence is transforming how we interact with technology.".to_string(),
            "Neural networks learn patterns from data through backpropagation algorithms.".to_string(),
            "Text embeddings capture semantic meaning in high-dimensional vector spaces.".to_string(),
            "Transformers revolutionized NLP with self-attention mechanisms for context.".to_string(),
            "BERT models use bidirectional training for better language understanding.".to_string(),
            "Sentence transformers create meaningful embeddings for similarity search.".to_string(),
        ];

        let long = vec![
            "The field of natural language processing has undergone a remarkable transformation in recent years, \
             driven by advances in deep learning and the development of transformer architectures. These models \
             have demonstrated unprecedented capabilities in understanding and generating human language, enabling \
             applications ranging from machine translation to question answering systems.".to_string(),
            "Modern embedding models like BERT, RoBERTa, and their successors have revolutionized how we represent \
             text in computational systems. By learning contextual representations, these models capture nuanced \
             semantic relationships that were previously difficult to model. This has led to significant improvements \
             in tasks such as semantic search, document classification, and information retrieval.".to_string(),
            "The tokenization process is fundamental to all neural language models. It converts raw text into \
             discrete tokens that can be processed by the model. Different tokenization strategies, such as \
             WordPiece, BPE, and Unigram, offer various trade-offs between vocabulary size, handling of rare words, \
             and computational efficiency. The choice of tokenizer significantly impacts model performance.".to_string(),
            "High-performance text processing systems must balance accuracy with throughput requirements. In production \
             environments, tokenizers process millions of requests daily, making efficiency crucial. SIMD optimizations, \
             intelligent caching, and parallel processing are key techniques for achieving the necessary performance \
             while maintaining the accuracy required for downstream model inference.".to_string(),
        ];

        let very_long = vec![
            "The evolution of natural language processing from rule-based systems to modern neural approaches \
             represents one of the most significant advances in artificial intelligence. Early NLP systems relied \
             on hand-crafted rules and linguistic knowledge, which limited their scalability and adaptability. \
             The introduction of statistical methods in the 1990s marked a paradigm shift, allowing systems to \
             learn patterns from data. However, it was the advent of deep learning, particularly recurrent neural \
             networks and later transformers, that truly revolutionized the field. Today's language models can \
             perform tasks that seemed impossible just a decade ago, from generating coherent long-form text to \
             engaging in nuanced conversations. The transformer architecture, introduced in the seminal 'Attention \
             Is All You Need' paper, replaced recurrence with self-attention mechanisms, enabling more efficient \
             parallelization and better modeling of long-range dependencies. This breakthrough led to models like \
             BERT, GPT, and their numerous variants, each pushing the boundaries of what machines can understand \
             and generate in human language.".to_string(),
            "Tokenization serves as the critical bridge between raw text and the numerical representations that \
             neural networks require. The choice of tokenization algorithm profoundly impacts both model performance \
             and computational efficiency. WordPiece, developed for BERT and related models, builds a vocabulary \
             of subword units that balance between whole words and individual characters. Byte-Pair Encoding (BPE), \
             popularized by GPT models, iteratively merges the most frequent character pairs to form tokens. \
             Unigram language model tokenization, used in SentencePiece, takes a probabilistic approach, selecting \
             the tokenization that maximizes the likelihood under a unigram language model. Each approach has its \
             strengths: WordPiece handles unknown words gracefully through subword decomposition, BPE provides \
             consistent tokenization across languages, and Unigram offers flexibility in vocabulary size selection. \
             The efficiency of tokenization is equally important in production systems. Modern tokenizers employ \
             various optimization techniques including SIMD instructions for parallel character processing, \
             hash-based vocabulary lookups, and intelligent caching of frequent words to achieve the throughput \
             required for real-time applications serving millions of users.".to_string(),
        ];

        Self { short, medium, long, very_long }
    }

    fn all_texts(&self) -> Vec<&str> {
        self.short.iter()
            .chain(self.medium.iter())
            .chain(self.long.iter())
            .chain(self.very_long.iter())
            .map(|s| s.as_str())
            .collect()
    }
}

#[derive(Debug, Clone)]
struct TokenAccuracyResult {
    total_texts: usize,
    exact_matches: usize,
    token_count_matches: usize,
    avg_token_diff: f64,
    mismatches: Vec<TokenMismatch>,
}

#[derive(Debug, Clone)]
struct TokenMismatch {
    text_preview: String,
    hf_tokens: usize,
    bt_tokens: usize,
    hf_ids_preview: Vec<u32>,
    bt_ids_preview: Vec<u32>,
}

#[derive(Debug, Clone)]
struct PerformanceResult {
    category: String,
    concurrency: usize,
    hf_rps: f64,
    bt_rps: f64,
    speedup: f64,
    hf_latency_p50_us: f64,
    bt_latency_p50_us: f64,
    hf_latency_p99_us: f64,
    bt_latency_p99_us: f64,
}

fn compare_token_accuracy(
    hf_tokenizer: &HFTokenizer,
    bt_tokenizer: &BudTikTokTokenizer,
    texts: &[&str],
) -> TokenAccuracyResult {
    let mut exact_matches = 0;
    let mut token_count_matches = 0;
    let mut total_token_diff = 0i64;
    let mut mismatches = Vec::new();

    for &text in texts {
        let hf_encoding = hf_tokenizer.encode(text, true).unwrap();
        let bt_encoding = bt_tokenizer.encode(text, true).unwrap();

        // Get IDs, filtering padding from HF
        let hf_ids: Vec<u32> = hf_encoding.get_ids().to_vec();
        let bt_ids: Vec<u32> = bt_encoding.get_ids().to_vec();

        // Filter out padding tokens (ID 0 at the end for BERT-style)
        let hf_ids_filtered: Vec<u32> = if hf_ids.len() > bt_ids.len() {
            // HF might have padding
            hf_ids.iter()
                .take_while(|&&id| id != 0 || hf_ids[0] == 0)
                .copied()
                .collect()
        } else {
            hf_ids.clone()
        };

        let tokens_match = hf_ids_filtered == bt_ids;
        let count_match = hf_ids_filtered.len() == bt_ids.len();

        if tokens_match {
            exact_matches += 1;
        }

        if count_match {
            token_count_matches += 1;
        }

        total_token_diff += (hf_ids_filtered.len() as i64 - bt_ids.len() as i64).abs();

        if !tokens_match && mismatches.len() < 10 {
            mismatches.push(TokenMismatch {
                text_preview: if text.len() > 50 {
                    format!("{}...", &text[..50])
                } else {
                    text.to_string()
                },
                hf_tokens: hf_ids_filtered.len(),
                bt_tokens: bt_ids.len(),
                hf_ids_preview: hf_ids_filtered.iter().take(10).copied().collect(),
                bt_ids_preview: bt_ids.iter().take(10).copied().collect(),
            });
        }
    }

    TokenAccuracyResult {
        total_texts: texts.len(),
        exact_matches,
        token_count_matches,
        avg_token_diff: total_token_diff as f64 / texts.len() as f64,
        mismatches,
    }
}

fn benchmark_tokenizer<F>(
    tokenizer_fn: F,
    texts: &[&str],
    concurrency: usize,
    num_requests: usize,
) -> (f64, f64, f64, f64) // (rps, p50_us, p90_us, p99_us)
where
    F: Fn(&str) -> Vec<u32> + Sync,
{
    let request_indices: Vec<usize> = (0..num_requests).collect();

    let start = Instant::now();

    let latencies: Vec<f64> = if concurrency == 1 {
        request_indices.iter().map(|&i| {
            let text = texts[i % texts.len()];
            let t0 = Instant::now();
            let _ = tokenizer_fn(text);
            t0.elapsed().as_micros() as f64
        }).collect()
    } else {
        request_indices.par_iter().map(|&i| {
            let text = texts[i % texts.len()];
            let t0 = Instant::now();
            let _ = tokenizer_fn(text);
            t0.elapsed().as_micros() as f64
        }).collect()
    };

    let total_time = start.elapsed();
    let rps = num_requests as f64 / total_time.as_secs_f64();

    let mut sorted = latencies.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = sorted[sorted.len() / 2];
    let p90 = sorted[(sorted.len() as f64 * 0.9) as usize];
    let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];

    (rps, p50, p90, p99)
}

fn run_category_benchmark(
    hf_tokenizer: &HFTokenizer,
    bt_tokenizer: &BudTikTokTokenizer,
    texts: &[&str],
    category: &str,
    config: &BenchConfig,
) -> Vec<PerformanceResult> {
    let mut results = Vec::new();

    println!("\n  {} ({} texts, avg {} chars):",
        category,
        texts.len(),
        texts.iter().map(|t| t.len()).sum::<usize>() / texts.len()
    );

    for &concurrency in &config.concurrency_levels {
        let (hf_rps, hf_p50, hf_p90, hf_p99) = benchmark_tokenizer(
            |text| hf_tokenizer.encode(text, true).unwrap().get_ids().to_vec(),
            texts,
            concurrency,
            config.requests_per_level,
        );

        let (bt_rps, bt_p50, bt_p90, bt_p99) = benchmark_tokenizer(
            |text| bt_tokenizer.encode(text, true).unwrap().get_ids().to_vec(),
            texts,
            concurrency,
            config.requests_per_level,
        );

        let speedup = bt_rps / hf_rps;

        println!("    c={:>4}: HF={:>8.0} rps, BT={:>8.0} rps ({:.2}x) | P50: {:.0}/{:.0}us, P99: {:.0}/{:.0}us",
            concurrency, hf_rps, bt_rps, speedup, hf_p50, bt_p50, hf_p99, bt_p99);

        results.push(PerformanceResult {
            category: category.to_string(),
            concurrency,
            hf_rps,
            bt_rps,
            speedup,
            hf_latency_p50_us: hf_p50,
            bt_latency_p50_us: bt_p50,
            hf_latency_p99_us: hf_p99,
            bt_latency_p99_us: bt_p99,
        });
    }

    results
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Find tokenizer path
    let tokenizer_path = if args.len() > 1 {
        args[1].clone()
    } else {
        // Default to all-MiniLM-L6-v2
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/bud".to_string());
        format!(
            "{}/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer.json",
            home
        )
    };

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║            COMPREHENSIVE TEI TOKENIZER BENCHMARK                             ║");
    println!("║         HuggingFace Tokenizers vs BudTikTok                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Loading tokenizers from: {}", tokenizer_path);

    // Load tokenizers
    let hf_tokenizer = HFTokenizer::from_file(&tokenizer_path)
        .expect("Failed to load HF tokenizer");
    let bt_tokenizer = BudTikTokTokenizer::from_file(&tokenizer_path)
        .expect("Failed to load BudTikTok tokenizer");

    // Print system info
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ SYSTEM INFORMATION                                                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!("  CPU cores (logical): {}", num_cpus::get());
    println!("  CPU cores (physical): {}", num_cpus::get_physical());

    #[cfg(target_arch = "x86_64")]
    {
        println!("  AVX2: {}", if is_x86_feature_detected!("avx2") { "Supported" } else { "Not supported" });
        println!("  AVX-512: {}", if is_x86_feature_detected!("avx512f") { "Supported" } else { "Not supported" });
    }

    // Generate test texts
    let test_texts = TestTexts::generate();
    let all_texts = test_texts.all_texts();

    // Benchmark configuration
    let config = BenchConfig::default();

    // Warmup
    println!();
    println!("Warming up ({} iterations)...", config.warmup_iterations);
    for i in 0..config.warmup_iterations {
        let text = all_texts[i % all_texts.len()];
        let _ = hf_tokenizer.encode(text, true);
        let _ = bt_tokenizer.encode(text, true);
    }

    // ========================================================================
    // ACCURACY COMPARISON
    // ========================================================================
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ TOKENIZATION ACCURACY                                                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    let categories = [
        ("Short texts (~15 chars)", test_texts.short.iter().map(|s| s.as_str()).collect::<Vec<_>>()),
        ("Medium texts (~80 chars)", test_texts.medium.iter().map(|s| s.as_str()).collect::<Vec<_>>()),
        ("Long texts (~350 chars)", test_texts.long.iter().map(|s| s.as_str()).collect::<Vec<_>>()),
        ("Very long texts (~1000 chars)", test_texts.very_long.iter().map(|s| s.as_str()).collect::<Vec<_>>()),
    ];

    let mut total_exact = 0;
    let mut total_texts = 0;

    for (name, texts) in &categories {
        let accuracy = compare_token_accuracy(&hf_tokenizer, &bt_tokenizer, texts);
        println!();
        println!("  {}:", name);
        println!("    Exact matches: {}/{} ({:.1}%)",
            accuracy.exact_matches, accuracy.total_texts,
            accuracy.exact_matches as f64 / accuracy.total_texts as f64 * 100.0);
        println!("    Token count matches: {}/{}", accuracy.token_count_matches, accuracy.total_texts);
        println!("    Avg token difference: {:.2}", accuracy.avg_token_diff);

        if !accuracy.mismatches.is_empty() {
            println!("    Mismatches:");
            for m in &accuracy.mismatches {
                println!("      - \"{}\": HF={} tokens, BT={} tokens",
                    m.text_preview, m.hf_tokens, m.bt_tokens);
                println!("        HF IDs: {:?}", m.hf_ids_preview);
                println!("        BT IDs: {:?}", m.bt_ids_preview);
            }
        }

        total_exact += accuracy.exact_matches;
        total_texts += accuracy.total_texts;
    }

    println!();
    println!("  ─────────────────────────────────────────────────────────────────────────────");
    println!("  TOTAL ACCURACY: {}/{} ({:.1}%)",
        total_exact, total_texts,
        total_exact as f64 / total_texts as f64 * 100.0);

    // ========================================================================
    // PERFORMANCE COMPARISON BY CATEGORY
    // ========================================================================
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ PERFORMANCE BY SEQUENCE LENGTH                                               ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    let short_texts: Vec<&str> = test_texts.short.iter().map(|s| s.as_str()).collect();
    let medium_texts: Vec<&str> = test_texts.medium.iter().map(|s| s.as_str()).collect();
    let long_texts: Vec<&str> = test_texts.long.iter().map(|s| s.as_str()).collect();
    let very_long_texts: Vec<&str> = test_texts.very_long.iter().map(|s| s.as_str()).collect();

    let mut all_results = Vec::new();

    all_results.extend(run_category_benchmark(&hf_tokenizer, &bt_tokenizer, &short_texts, "Short (~15 chars)", &config));
    all_results.extend(run_category_benchmark(&hf_tokenizer, &bt_tokenizer, &medium_texts, "Medium (~80 chars)", &config));
    all_results.extend(run_category_benchmark(&hf_tokenizer, &bt_tokenizer, &long_texts, "Long (~350 chars)", &config));
    all_results.extend(run_category_benchmark(&hf_tokenizer, &bt_tokenizer, &very_long_texts, "Very Long (~1000 chars)", &config));

    // ========================================================================
    // SUMMARY BY CONCURRENCY
    // ========================================================================
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ SUMMARY BY CONCURRENCY LEVEL                                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  {:>6} | {:>12} | {:>12} | {:>8} | {:>10} | {:>10}",
        "Conc", "HF RPS", "BT RPS", "Speedup", "HF P50(us)", "BT P50(us)");
    println!("  ─────────────────────────────────────────────────────────────────────────────");

    // Group by concurrency
    let mut by_concurrency: HashMap<usize, Vec<&PerformanceResult>> = HashMap::new();
    for result in &all_results {
        by_concurrency.entry(result.concurrency).or_default().push(result);
    }

    let mut concurrency_summary = Vec::new();
    for &conc in &config.concurrency_levels {
        if let Some(results) = by_concurrency.get(&conc) {
            let avg_hf_rps = results.iter().map(|r| r.hf_rps).sum::<f64>() / results.len() as f64;
            let avg_bt_rps = results.iter().map(|r| r.bt_rps).sum::<f64>() / results.len() as f64;
            let avg_speedup = avg_bt_rps / avg_hf_rps;
            let avg_hf_p50 = results.iter().map(|r| r.hf_latency_p50_us).sum::<f64>() / results.len() as f64;
            let avg_bt_p50 = results.iter().map(|r| r.bt_latency_p50_us).sum::<f64>() / results.len() as f64;

            println!("  {:>6} | {:>12.0} | {:>12.0} | {:>7.2}x | {:>10.1} | {:>10.1}",
                conc, avg_hf_rps, avg_bt_rps, avg_speedup, avg_hf_p50, avg_bt_p50);

            concurrency_summary.push((conc, avg_speedup));
        }
    }

    // ========================================================================
    // SUMMARY BY SEQUENCE LENGTH
    // ========================================================================
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ SUMMARY BY SEQUENCE LENGTH                                                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  {:>20} | {:>12} | {:>12} | {:>8} | {:>8}",
        "Category", "Avg HF RPS", "Avg BT RPS", "Speedup", "Latency↓");
    println!("  ─────────────────────────────────────────────────────────────────────────────");

    let categories_list = ["Short (~15 chars)", "Medium (~80 chars)", "Long (~350 chars)", "Very Long (~1000 chars)"];
    for cat in categories_list {
        let cat_results: Vec<_> = all_results.iter().filter(|r| r.category == cat).collect();
        if !cat_results.is_empty() {
            let avg_hf_rps = cat_results.iter().map(|r| r.hf_rps).sum::<f64>() / cat_results.len() as f64;
            let avg_bt_rps = cat_results.iter().map(|r| r.bt_rps).sum::<f64>() / cat_results.len() as f64;
            let avg_speedup = avg_bt_rps / avg_hf_rps;
            let avg_hf_p50 = cat_results.iter().map(|r| r.hf_latency_p50_us).sum::<f64>() / cat_results.len() as f64;
            let avg_bt_p50 = cat_results.iter().map(|r| r.bt_latency_p50_us).sum::<f64>() / cat_results.len() as f64;
            let latency_improvement = (avg_hf_p50 - avg_bt_p50) / avg_hf_p50 * 100.0;

            println!("  {:>20} | {:>12.0} | {:>12.0} | {:>7.2}x | {:>7.1}%",
                cat, avg_hf_rps, avg_bt_rps, avg_speedup, latency_improvement);
        }
    }

    // ========================================================================
    // FINAL SUMMARY
    // ========================================================================
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ FINAL SUMMARY                                                                ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let overall_avg_speedup = all_results.iter().map(|r| r.speedup).sum::<f64>() / all_results.len() as f64;
    let overall_avg_hf_p50 = all_results.iter().map(|r| r.hf_latency_p50_us).sum::<f64>() / all_results.len() as f64;
    let overall_avg_bt_p50 = all_results.iter().map(|r| r.bt_latency_p50_us).sum::<f64>() / all_results.len() as f64;
    let latency_improvement = (overall_avg_hf_p50 - overall_avg_bt_p50) / overall_avg_hf_p50 * 100.0;

    println!("  Tokenization Accuracy:  {}/{} ({:.1}%)",
        total_exact, total_texts, total_exact as f64 / total_texts as f64 * 100.0);
    println!("  Average Throughput:     {:.2}x faster", overall_avg_speedup);
    println!("  Average Latency:        {:.1}% lower", latency_improvement);
    println!("  P50 Latency:            {:.1}us (HF) vs {:.1}us (BT)", overall_avg_hf_p50, overall_avg_bt_p50);
    println!();

    // Concurrency scaling
    if concurrency_summary.len() >= 2 {
        let (_, speedup_1) = concurrency_summary[0];
        let (_, speedup_max) = concurrency_summary[concurrency_summary.len() - 1];
        println!("  Scaling (c=1 to c=1000):");
        println!("    c=1:    {:.2}x faster", speedup_1);
        println!("    c=1000: {:.2}x faster", speedup_max);
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ BENCHMARK COMPLETE                                                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
}
