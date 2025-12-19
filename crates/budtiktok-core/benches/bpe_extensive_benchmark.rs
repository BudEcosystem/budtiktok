//! Extensive BPE Performance Benchmarks
//!
//! Comprehensive benchmarking across:
//! - Batch sizes (c): 1, 10, 100, 500, 1000, 2000, 5000, 10000
//! - Sequence lengths: 100, 250, 500, 1000, 2000, 5000, 10000

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use budtiktok_core::bpe::{BpeModel, Gpt2ByteEncoder, MergeRule};
use budtiktok_core::bpe_hyper::{HyperBpeTokenizer, HyperBpeConfig};
use ahash::AHashMap;
use std::time::Duration;

/// Create comprehensive test vocabulary (GPT-2 style with more merges)
fn create_benchmark_vocab() -> (AHashMap<String, u32>, Vec<(String, String)>, Vec<MergeRule>) {
    let byte_encoder = Gpt2ByteEncoder::new();
    let mut vocab = AHashMap::new();
    let mut next_id = 0u32;

    // Add all byte-level tokens (256 tokens)
    for byte in 0u8..=255 {
        let c = byte_encoder.encode_byte(byte);
        vocab.insert(c.to_string(), next_id);
        next_id += 1;
    }

    // Add extensive merged tokens for realistic workload
    let common_merges = [
        // Common word parts
        ("t", "h"), ("th", "e"), ("the", " "),
        ("a", "n"), ("an", "d"), ("and", " "),
        ("i", "n"), ("in", "g"),
        ("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o"),
        ("w", "o"), ("wo", "r"), ("wor", "l"), ("worl", "d"),
        ("Ġ", "t"), ("Ġt", "h"), ("Ġth", "e"),
        ("Ġ", "a"), ("Ġa", "n"), ("Ġan", "d"),
        ("Ġ", "i"), ("Ġi", "s"), ("Ġi", "n"),
        ("e", "r"), ("er", " "), ("er", "s"),
        ("o", "n"), ("on", " "), ("on", "e"),
        ("i", "t"), ("it", " "), ("it", "h"),
        ("o", "f"), ("of", " "),
        ("t", "o"), ("to", " "),
        ("a", "t"), ("at", " "), ("at", "e"),
        ("o", "r"), ("or", " "), ("or", "e"),
        ("e", "n"), ("en", " "), ("en", "t"),
        ("a", "l"), ("al", " "), ("al", "l"),
        ("r", "e"), ("re", " "), ("re", "s"),
        ("c", "o"), ("co", "m"), ("com", "p"),
        ("s", "t"), ("st", "a"), ("sta", "t"),
        ("p", "r"), ("pr", "o"), ("pro", "c"),
        ("Ġ", "w"), ("Ġw", "i"), ("Ġwi", "t"), ("Ġwit", "h"),
        ("Ġ", "f"), ("Ġf", "o"), ("Ġfo", "r"),
        ("Ġ", "b"), ("Ġb", "e"), ("Ġbe", " "),
        ("Ġ", "o"), ("Ġo", "f"), ("Ġof", " "),
        ("Ġ", "s"), ("Ġs", "o"), ("Ġso", "m"),
        (".", " "), (",", " "), ("!", " "), ("?", " "),
    ];

    for (first, second) in &common_merges {
        let merged = format!("{}{}", first, second);
        if !vocab.contains_key(&merged) {
            vocab.insert(merged, next_id);
            next_id += 1;
        }
    }

    let merge_tuples: Vec<(String, String)> = common_merges
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();

    let merge_rules: Vec<MergeRule> = common_merges
        .iter()
        .enumerate()
        .map(|(i, (a, b))| MergeRule {
            first: a.to_string(),
            second: b.to_string(),
            result: format!("{}{}", a, b),
            priority: i as u32,
        })
        .collect();

    (vocab, merge_tuples, merge_rules)
}

/// Create scalar BPE model
fn create_scalar_bpe() -> BpeModel {
    let (vocab, merge_tuples, _) = create_benchmark_vocab();
    BpeModel::new(vocab, merge_tuples)
}

/// Create SIMD BPE tokenizer
fn create_hyper_bpe() -> HyperBpeTokenizer {
    let (vocab_map, _, merge_rules) = create_benchmark_vocab();
    let vocab = budtiktok_core::vocab::Vocabulary::new(
        vocab_map,
        budtiktok_core::vocab::SpecialTokens::default()
    );

    let config = HyperBpeConfig {
        byte_level: false,
        ..Default::default()
    };

    HyperBpeTokenizer::new(vocab, merge_rules, config)
}

/// Generate realistic text of specified length
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
            // Vary text slightly to avoid caching effects
            let mut text = generate_text(seq_len);
            if i % 2 == 0 {
                text = text.to_uppercase();
            }
            text
        })
        .collect()
}

// =============================================================================
// Batch Size Benchmarks (varying c with fixed seq_len=500)
// =============================================================================

fn bench_batch_sizes_scalar(c: &mut Criterion) {
    let bpe = create_scalar_bpe();
    let seq_len = 500;
    let batch_sizes = [1, 10, 100, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("scalar_batch_size_seq500");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for &batch_size in &batch_sizes {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("c", batch_size),
            &batch_refs,
            |b, batch| {
                b.iter(|| {
                    let results: Vec<_> = batch.iter()
                        .map(|text| black_box(bpe.encode(text)))
                        .collect();
                    results
                })
            }
        );
    }

    group.finish();
}

fn bench_batch_sizes_simd(c: &mut Criterion) {
    let bpe = create_hyper_bpe();
    let seq_len = 500;
    let batch_sizes = [1, 10, 100, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("simd_batch_size_seq500");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    for &batch_size in &batch_sizes {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("c", batch_size),
            &batch_refs,
            |b, batch| {
                b.iter(|| black_box(bpe.encode_batch(batch)))
            }
        );
    }

    group.finish();
}

// =============================================================================
// Sequence Length Benchmarks (varying seq_len with fixed c=100)
// =============================================================================

fn bench_seq_lengths_scalar(c: &mut Criterion) {
    let bpe = create_scalar_bpe();
    let batch_size = 100;
    let seq_lengths = [100, 250, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("scalar_seq_length_c100");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(30);

    for &seq_len in &seq_lengths {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("len", seq_len),
            &batch_refs,
            |b, batch| {
                b.iter(|| {
                    let results: Vec<_> = batch.iter()
                        .map(|text| black_box(bpe.encode(text)))
                        .collect();
                    results
                })
            }
        );
    }

    group.finish();
}

fn bench_seq_lengths_simd(c: &mut Criterion) {
    let bpe = create_hyper_bpe();
    let batch_size = 100;
    let seq_lengths = [100, 250, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("simd_seq_length_c100");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(30);

    for &seq_len in &seq_lengths {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("len", seq_len),
            &batch_refs,
            |b, batch| {
                b.iter(|| black_box(bpe.encode_batch(batch)))
            }
        );
    }

    group.finish();
}

// =============================================================================
// Combined Matrix Benchmarks (c × seq_len)
// =============================================================================

fn bench_matrix_simd(c: &mut Criterion) {
    let bpe = create_hyper_bpe();

    // Key combinations
    let configs = [
        (1, 100),
        (1, 1000),
        (1, 10000),
        (10, 100),
        (10, 1000),
        (10, 10000),
        (100, 100),
        (100, 1000),
        (100, 10000),
        (1000, 100),
        (1000, 1000),
        (1000, 10000),
        (10000, 100),
        (10000, 1000),
    ];

    let mut group = c.benchmark_group("simd_matrix");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    for &(batch_size, seq_len) in &configs {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("c{}_len{}", batch_size, seq_len), total_bytes),
            &batch_refs,
            |b, batch| {
                b.iter(|| black_box(bpe.encode_batch(batch)))
            }
        );
    }

    group.finish();
}

fn bench_matrix_scalar(c: &mut Criterion) {
    let bpe = create_scalar_bpe();

    // Smaller subset for scalar (it's slower)
    let configs = [
        (1, 100),
        (1, 1000),
        (10, 100),
        (10, 1000),
        (100, 100),
        (100, 1000),
        (1000, 100),
    ];

    let mut group = c.benchmark_group("scalar_matrix");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    for &(batch_size, seq_len) in &configs {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("c{}_len{}", batch_size, seq_len), total_bytes),
            &batch_refs,
            |b, batch| {
                b.iter(|| {
                    let results: Vec<_> = batch.iter()
                        .map(|text| black_box(bpe.encode(text)))
                        .collect();
                    results
                })
            }
        );
    }

    group.finish();
}

// =============================================================================
// Throughput Scaling Benchmarks
// =============================================================================

fn bench_throughput_scaling(c: &mut Criterion) {
    let bpe = create_hyper_bpe();

    // Total bytes constant (~1MB), vary batch_size vs seq_len
    let target_bytes = 1_000_000;
    let configs = [
        (10000, 100),   // Many short sequences
        (2000, 500),    // Medium
        (1000, 1000),   // Balanced
        (200, 5000),    // Fewer long sequences
        (100, 10000),   // Few very long sequences
    ];

    let mut group = c.benchmark_group("throughput_scaling_1MB");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(20);
    group.throughput(Throughput::Bytes(target_bytes as u64));

    for &(batch_size, seq_len) in &configs {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();

        group.bench_with_input(
            BenchmarkId::new(format!("c{}_len{}", batch_size, seq_len), batch_size * seq_len),
            &batch_refs,
            |b, batch| {
                b.iter(|| black_box(bpe.encode_batch(batch)))
            }
        );
    }

    group.finish();
}

// =============================================================================
// Single Sequence Performance (baseline)
// =============================================================================

fn bench_single_sequence(c: &mut Criterion) {
    let scalar_bpe = create_scalar_bpe();
    let simd_bpe = create_hyper_bpe();

    let seq_lengths = [100, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("single_sequence");
    group.measurement_time(Duration::from_secs(10));

    for &seq_len in &seq_lengths {
        let text = generate_text(seq_len);

        group.throughput(Throughput::Bytes(seq_len as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar", seq_len),
            &text,
            |b, text| {
                b.iter(|| black_box(scalar_bpe.encode(text)))
            }
        );

        group.bench_with_input(
            BenchmarkId::new("simd", seq_len),
            &text,
            |b, text| {
                b.iter(|| black_box(simd_bpe.encode_fast(text)))
            }
        );
    }

    group.finish();
}

// =============================================================================
// Latency Percentiles (for production use)
// =============================================================================

fn bench_latency_p99(c: &mut Criterion) {
    let bpe = create_hyper_bpe();

    // Production-like workload: c=100, varying seq lengths
    let batch_size = 100;
    let seq_lengths = [256, 512, 1024, 2048];

    let mut group = c.benchmark_group("latency_c100");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(100);

    for &seq_len in &seq_lengths {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("simd", seq_len),
            &batch_refs,
            |b, batch| {
                b.iter(|| black_box(bpe.encode_batch(batch)))
            }
        );
    }

    group.finish();
}

criterion_group!(
    name = batch_benches;
    config = Criterion::default().significance_level(0.05).noise_threshold(0.02);
    targets = bench_batch_sizes_scalar, bench_batch_sizes_simd
);

criterion_group!(
    name = seq_benches;
    config = Criterion::default().significance_level(0.05).noise_threshold(0.02);
    targets = bench_seq_lengths_scalar, bench_seq_lengths_simd
);

criterion_group!(
    name = matrix_benches;
    config = Criterion::default().significance_level(0.05).noise_threshold(0.02);
    targets = bench_matrix_simd, bench_matrix_scalar
);

criterion_group!(
    name = throughput_benches;
    config = Criterion::default().significance_level(0.05).noise_threshold(0.02);
    targets = bench_throughput_scaling, bench_single_sequence, bench_latency_p99
);

criterion_main!(batch_benches, seq_benches, matrix_benches, throughput_benches);
