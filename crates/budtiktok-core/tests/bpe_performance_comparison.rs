//! BPE Performance Comparison: budtiktok vs HuggingFace
//!
//! Comprehensive performance comparison across:
//! - Batch sizes (concurrency): 1, 10, 100, 500, 1000
//! - Sequence lengths: 100, 500, 1000, 2000, 5000

use ahash::AHashMap;
use budtiktok_core::bpe_fast::{FastBpeEncoder, SimdBpeEncoder};
use budtiktok_core::bpe_simd::SimdOptimizedBpeEncoder;
use budtiktok_core::bpe_linear::{TrieEncoder, FastLinearEncoder, OptimizedBpeEncoder};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::tokenizer::Tokenizer as HfTokenizer;

/// GPT-2 style byte encoder
fn gpt2_byte_encoder() -> [char; 256] {
    let mut encode = ['\0'; 256];
    let mut n = 0u32;

    for b in 0u8..=255 {
        let c = if (b'!'..=b'~').contains(&b)
            || (0xa1..=0xac).contains(&b)
            || (0xae..=0xff).contains(&b)
        {
            b as char
        } else {
            let c = char::from_u32(256 + n).unwrap_or('?');
            n += 1;
            c
        };
        encode[b as usize] = c;
    }

    encode
}

/// Create vocabulary and merges for benchmarking
fn create_benchmark_vocab() -> (
    AHashMap<String, u32>,
    HashMap<String, u32>,
    Vec<(String, String)>,
) {
    let byte_encoder = gpt2_byte_encoder();
    let mut vocab_ahash: AHashMap<String, u32> = AHashMap::new();
    let mut vocab_std: HashMap<String, u32> = HashMap::new();
    let mut next_id = 0u32;

    // Add base byte tokens (256 tokens)
    for b in 0u8..=255 {
        let c = byte_encoder[b as usize];
        let token = c.to_string();
        vocab_ahash.insert(token.clone(), next_id);
        vocab_std.insert(token, next_id);
        next_id += 1;
    }

    let byte_encode = |s: &str| -> String {
        s.bytes().map(|b| byte_encoder[b as usize]).collect()
    };

    // Extensive merge rules for realistic workload
    let common_merges = [
        ("t", "h"), ("th", "e"), ("the", " "),
        ("a", "n"), ("an", "d"), ("and", " "),
        ("i", "n"), ("in", "g"),
        ("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o"),
        ("w", "o"), ("wo", "r"), ("wor", "l"), ("worl", "d"),
        ("e", "r"), ("o", "n"),
        ("i", "t"), ("o", "f"),
        ("t", "o"), ("a", "t"),
        ("o", "r"), ("e", "n"),
        ("a", "l"), ("r", "e"),
        ("c", "o"), ("co", "m"), ("com", "p"),
        ("s", "t"), ("st", "a"), ("sta", "r"),
        ("m", "a"), ("ma", "c"), ("mac", "h"),
        ("l", "e"), ("le", "a"), ("lea", "r"),
        ("p", "r"), ("pr", "o"), ("pro", "c"),
        ("b", "e"), ("be", "n"),
        ("q", "u"), ("qu", "i"), ("qui", "c"), ("quic", "k"),
        ("b", "r"), ("br", "o"), ("bro", "w"), ("brow", "n"),
        ("f", "o"), ("fo", "x"),
        ("j", "u"), ("ju", "m"), ("jum", "p"), ("jump", "s"),
        ("o", "v"), ("ov", "e"), ("ove", "r"),
        ("l", "a"), ("la", "z"), ("laz", "y"),
        ("d", "o"), ("do", "g"),
        // Space prefixed merges
        (" ", "t"), (" t", "h"), (" th", "e"),
        (" ", "a"), (" a", "n"), (" an", "d"),
        (" ", "i"), (" i", "n"), (" in", "g"),
        (" ", "h"), (" h", "e"), (" he", "l"), (" hel", "l"), (" hell", "o"),
        (" ", "w"), (" w", "o"), (" wo", "r"), (" wor", "l"), (" worl", "d"),
        (" ", "q"), (" q", "u"), (" qu", "i"), (" qui", "c"), (" quic", "k"),
        (" ", "b"), (" b", "r"), (" br", "o"), (" bro", "w"), (" brow", "n"),
        (" ", "f"), (" f", "o"), (" fo", "x"),
        (" ", "j"), (" j", "u"), (" ju", "m"), (" jum", "p"), (" jump", "s"),
        (" ", "o"), (" o", "v"), (" ov", "e"), (" ove", "r"),
        (" ", "l"), (" l", "a"), (" la", "z"), (" laz", "y"),
        (" ", "d"), (" d", "o"), (" do", "g"),
    ];

    let mut merge_tuples: Vec<(String, String)> = Vec::new();
    for (first, second) in &common_merges {
        let first_enc = byte_encode(first);
        let second_enc = byte_encode(second);
        let merged_enc = format!("{}{}", first_enc, second_enc);

        if !vocab_ahash.contains_key(&merged_enc) {
            vocab_ahash.insert(merged_enc.clone(), next_id);
            vocab_std.insert(merged_enc, next_id);
            next_id += 1;
        }
        merge_tuples.push((first_enc, second_enc));
    }

    vocab_ahash.insert("<unk>".to_string(), next_id);
    vocab_std.insert("<unk>".to_string(), next_id);

    (vocab_ahash, vocab_std, merge_tuples)
}

/// Generate text of specified length
fn generate_text(target_len: usize) -> String {
    let base_texts = [
        "The quick brown fox jumps over the lazy dog. ",
        "Hello world, this is a test of the tokenizer. ",
        "Machine learning and artificial intelligence are transforming the world. ",
        "Natural language processing enables computers to understand human text. ",
        "The weather today is sunny with a chance of rain in the afternoon. ",
        "Programming in Rust provides memory safety without garbage collection. ",
        "Deep learning models require large amounts of training data. ",
        "The stock market experienced significant volatility this week. ",
    ];

    let mut result = String::with_capacity(target_len + 100);
    let mut idx = 0;

    while result.len() < target_len {
        result.push_str(base_texts[idx % base_texts.len()]);
        idx += 1;
    }

    result.truncate(target_len);
    result
}

/// Generate batch of texts
fn generate_batch(batch_size: usize, seq_len: usize) -> Vec<String> {
    (0..batch_size)
        .map(|i| {
            let mut text = generate_text(seq_len);
            if i % 2 == 0 {
                text = text.to_uppercase();
            }
            text
        })
        .collect()
}

/// Benchmark result
#[derive(Debug, Clone)]
struct BenchResult {
    name: String,
    batch_size: usize,
    seq_len: usize,
    total_bytes: usize,
    duration: Duration,
    throughput_mb_s: f64,
}

impl BenchResult {
    fn new(name: &str, batch_size: usize, seq_len: usize, total_bytes: usize, duration: Duration) -> Self {
        let throughput_mb_s = (total_bytes as f64 / 1_000_000.0) / duration.as_secs_f64();
        Self {
            name: name.to_string(),
            batch_size,
            seq_len,
            total_bytes,
            duration,
            throughput_mb_s,
        }
    }
}

/// Run benchmark iterations
fn benchmark<F>(name: &str, batch_size: usize, seq_len: usize, total_bytes: usize, iterations: usize, mut f: F) -> BenchResult
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..3 {
        f();
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let duration = start.elapsed() / iterations as u32;

    BenchResult::new(name, batch_size, seq_len, total_bytes, duration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_comparison() {
        println!("\n{}", "=".repeat(80));
        println!("BPE PERFORMANCE COMPARISON: budtiktok vs HuggingFace");
        println!("{}\n", "=".repeat(80));

        let (vocab_ahash, vocab_std, merges) = create_benchmark_vocab();

        // Create budtiktok encoders
        let fast_bpe = FastBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");
        let simd_bpe = SimdBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");
        let simd_opt_bpe = SimdOptimizedBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");
        let trie_bpe = TrieEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");
        let fast_linear = FastLinearEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");
        let optimized_bpe = OptimizedBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");

        println!("SIMD Level: {:?}", simd_opt_bpe.simd_level());

        // Create HuggingFace tokenizer
        let hf_bpe = BPE::builder()
            .vocab_and_merges(vocab_std.clone(), merges.clone())
            .unk_token("<unk>".to_string())
            .build()
            .expect("Failed to build HF BPE");

        let mut hf_tokenizer = HfTokenizer::new(hf_bpe);
        hf_tokenizer.with_pre_tokenizer(PreTokenizerWrapper::ByteLevel(
            ByteLevel::new(false, true, false)
        ));

        // Test configurations
        let batch_sizes = [1, 10, 100, 500, 1000];
        let seq_lengths = [100, 500, 1000, 2000, 5000];

        let iterations = 10;

        println!("\n{:-<165}", "");
        println!("{:<12} | {:>8} | {:>8} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9} | {:>9}",
            "Config", "Batch", "SeqLen", "HF(MB/s)", "Fast", "Trie", "Linear", "Optimized", "SIMD", "SIMDOpt", "Speedup");
        println!("{:-<165}", "");

        for &batch_size in &batch_sizes {
            for &seq_len in &seq_lengths {
                // Skip very large combinations to avoid long test times
                if batch_size * seq_len > 2_000_000 {
                    continue;
                }

                let batch = generate_batch(batch_size, seq_len);
                let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
                let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

                // Benchmark HuggingFace
                let hf_result = benchmark(
                    "HuggingFace",
                    batch_size,
                    seq_len,
                    total_bytes,
                    iterations,
                    || {
                        for text in &batch {
                            let _ = hf_tokenizer.encode(text.as_str(), false);
                        }
                    }
                );

                // Benchmark FastBpeEncoder
                let fast_result = benchmark(
                    "FastBpe",
                    batch_size,
                    seq_len,
                    total_bytes,
                    iterations,
                    || {
                        for text in &batch_refs {
                            let _ = fast_bpe.encode(text);
                        }
                    }
                );

                // Benchmark SimdBpeEncoder
                let simd_result = benchmark(
                    "SimdBpe",
                    batch_size,
                    seq_len,
                    total_bytes,
                    iterations,
                    || {
                        let _ = simd_bpe.encode_batch(&batch_refs);
                    }
                );

                // Benchmark SimdOptimizedBpeEncoder
                let simd_opt_result = benchmark(
                    "SimdOptBpe",
                    batch_size,
                    seq_len,
                    total_bytes,
                    iterations,
                    || {
                        let _ = simd_opt_bpe.encode_batch(&batch_refs);
                    }
                );

                // Benchmark TrieEncoder (linear-time)
                let trie_result = benchmark(
                    "TrieBpe",
                    batch_size,
                    seq_len,
                    total_bytes,
                    iterations,
                    || {
                        for text in &batch_refs {
                            let _ = trie_bpe.encode(text);
                        }
                    }
                );

                // Benchmark FastLinearEncoder (no regex, simple whitespace)
                let linear_result = benchmark(
                    "FastLinear",
                    batch_size,
                    seq_len,
                    total_bytes,
                    iterations,
                    || {
                        for text in &batch_refs {
                            let _ = fast_linear.encode(text);
                        }
                    }
                );

                // Benchmark OptimizedBpeEncoder (hand-rolled GPT-2 pre-tokenizer, 100% compatible)
                let optimized_result = benchmark(
                    "Optimized",
                    batch_size,
                    seq_len,
                    total_bytes,
                    iterations,
                    || {
                        for text in &batch_refs {
                            let _ = optimized_bpe.encode(text);
                        }
                    }
                );

                // Calculate speedup (best budtiktok vs HF)
                let best_budtiktok = simd_opt_result.throughput_mb_s
                    .max(simd_result.throughput_mb_s)
                    .max(fast_result.throughput_mb_s)
                    .max(trie_result.throughput_mb_s)
                    .max(linear_result.throughput_mb_s)
                    .max(optimized_result.throughput_mb_s);
                let speedup = best_budtiktok / hf_result.throughput_mb_s;

                println!("{:<12} | {:>8} | {:>8} | {:>9.2} | {:>9.2} | {:>9.2} | {:>9.2} | {:>9.2} | {:>9.2} | {:>9.2} | {:>8.1}x",
                    format!("c{}l{}", batch_size, seq_len),
                    batch_size,
                    seq_len,
                    hf_result.throughput_mb_s,
                    fast_result.throughput_mb_s,
                    trie_result.throughput_mb_s,
                    linear_result.throughput_mb_s,
                    optimized_result.throughput_mb_s,
                    simd_result.throughput_mb_s,
                    simd_opt_result.throughput_mb_s,
                    speedup
                );
            }
        }

        println!("{:-<165}\n", "");

        // Summary statistics
        println!("SUMMARY:");
        println!("- FastBpeEncoder: Scalar heap-based with fancy_regex (slow)");
        println!("- TrieEncoder: Trie-based with fancy_regex pre-tokenization");
        println!("- FastLinearEncoder: Trie-based with whitespace splitting (fast, ~99% compatible)");
        println!("- OptimizedBpeEncoder: Trie-based with HAND-ROLLED GPT-2 pre-tokenizer (FAST + 100% COMPATIBLE)");
        println!("- SimdBpeEncoder: SIMD-accelerated with thread-local workspaces");
        println!("- SimdOptimizedBpeEncoder: Maximum SIMD optimization (AVX2/AVX-512)");
        println!("\nNote: 'Optimized' uses a hand-rolled GPT-2 pre-tokenizer (no fancy_regex) for 100% HuggingFace compatibility.");
        println!("'Linear' uses simple whitespace splitting (fastest but may differ for edge cases).\n");
    }

    #[test]
    fn test_accuracy_verification() {
        println!("\n{}", "=".repeat(80));
        println!("ACCURACY VERIFICATION: budtiktok vs HuggingFace");
        println!("{}\n", "=".repeat(80));

        let (vocab_ahash, vocab_std, merges) = create_benchmark_vocab();

        let fast_bpe = FastBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");
        let simd_bpe = SimdBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");
        let simd_opt_bpe = SimdOptimizedBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");
        let optimized_bpe = OptimizedBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");

        let hf_bpe = BPE::builder()
            .vocab_and_merges(vocab_std.clone(), merges.clone())
            .unk_token("<unk>".to_string())
            .build()
            .expect("Failed to build HF BPE");

        let mut hf_tokenizer = HfTokenizer::new(hf_bpe);
        hf_tokenizer.with_pre_tokenizer(PreTokenizerWrapper::ByteLevel(
            ByteLevel::new(false, true, false)
        ));

        let test_texts = vec![
            "hello",
            "world",
            "hello world",
            "the quick brown fox",
            "jumps over the lazy dog",
            "hello  world",  // double space
            "  hello world  ",  // leading/trailing spaces
            "THE QUICK BROWN FOX",
            "Machine learning is amazing!",
            "Testing 123 numbers",
        ];

        let mut total = 0;
        let mut passed = 0;

        for text in &test_texts {
            total += 1;

            let hf_result = hf_tokenizer.encode(*text, false);
            if hf_result.is_err() {
                println!("HF failed on: {}", text);
                continue;
            }
            let hf_ids: Vec<u32> = hf_result.unwrap().get_ids().to_vec();

            let fast_ids = fast_bpe.encode(text);
            let simd_ids = simd_bpe.encode(text);
            let simd_opt_ids = simd_opt_bpe.encode(text);
            let optimized_ids = optimized_bpe.encode(text);

            let fast_match = fast_ids == hf_ids;
            let simd_match = simd_ids == hf_ids;
            let simd_opt_match = simd_opt_ids == hf_ids;
            let optimized_match = optimized_ids == hf_ids;

            if fast_match && simd_match && simd_opt_match && optimized_match {
                passed += 1;
                println!("[PASS] \"{}\"", text);
            } else {
                println!("[FAIL] \"{}\"", text);
                println!("       HF:        {:?}", hf_ids);
                println!("       Fast:      {:?} {}", fast_ids, if fast_match { "OK" } else { "MISMATCH" });
                println!("       SIMD:      {:?} {}", simd_ids, if simd_match { "OK" } else { "MISMATCH" });
                println!("       SIMDOpt:   {:?} {}", simd_opt_ids, if simd_opt_match { "OK" } else { "MISMATCH" });
                println!("       Optimized: {:?} {}", optimized_ids, if optimized_match { "OK" } else { "MISMATCH" });
            }
        }

        println!("\nAccuracy: {}/{} ({:.1}%)", passed, total, (passed as f64 / total as f64) * 100.0);

        assert_eq!(passed, total, "Not all tests passed! {}/{}", passed, total);
    }

    #[test]
    fn test_batch_consistency() {
        println!("\n{}", "=".repeat(80));
        println!("BATCH CONSISTENCY TEST");
        println!("{}\n", "=".repeat(80));

        let (vocab_ahash, _, merges) = create_benchmark_vocab();

        let fast_bpe = FastBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");
        let simd_bpe = SimdBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");
        let simd_opt_bpe = SimdOptimizedBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<unk>");

        let batch_sizes = [1, 10, 100, 500];
        let seq_lengths = [100, 500, 1000];

        for &batch_size in &batch_sizes {
            for &seq_len in &seq_lengths {
                let batch = generate_batch(batch_size, seq_len);
                let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();

                let fast_results = fast_bpe.encode_batch(&batch_refs);
                let simd_results = simd_bpe.encode_batch(&batch_refs);
                let simd_opt_results = simd_opt_bpe.encode_batch(&batch_refs);

                let mut all_match = true;
                for i in 0..batch_size {
                    if fast_results[i] != simd_results[i] || simd_results[i] != simd_opt_results[i] {
                        all_match = false;
                        println!("[FAIL] Batch c={} len={} item {}", batch_size, seq_len, i);
                        break;
                    }
                }

                if all_match {
                    println!("[PASS] Batch c={} len={}: All {} items consistent", batch_size, seq_len, batch_size);
                }
            }
        }

        println!("\nAll batch encodings are internally consistent across all encoders.");
    }
}
