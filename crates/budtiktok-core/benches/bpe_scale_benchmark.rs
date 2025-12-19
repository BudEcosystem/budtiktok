//! BPE Scale Performance Benchmarks
//!
//! Comprehensive benchmarking across:
//! - Batch sizes (c): 1, 10, 100, 500, 1000, 2000, 5000, 10000
//! - Sequence lengths: 100, 250, 500, 1000, 2000, 5000, 10000
//!
//! Optimized for fast execution while maintaining statistical validity.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use budtiktok_core::bpe::{BpeModel, Gpt2ByteEncoder, MergeRule};
use budtiktok_core::bpe_hyper::{HyperBpeTokenizer, HyperBpeConfig};
use budtiktok_core::bpe_fast::{FastBpeEncoder, SimdBpeEncoder};
use budtiktok_core::bpe_simd::{SimdOptimizedBpeEncoder, detected_simd_level};
use ahash::AHashMap;
use std::time::Duration;

/// Create test vocabulary with GPT-2 style merges
fn create_benchmark_vocab() -> (AHashMap<String, u32>, Vec<(String, String)>, Vec<MergeRule>) {
    let byte_encoder = Gpt2ByteEncoder::new();
    let mut vocab = AHashMap::new();
    let mut next_id = 0u32;

    for byte in 0u8..=255 {
        let c = byte_encoder.encode_byte(byte);
        vocab.insert(c.to_string(), next_id);
        next_id += 1;
    }

    let common_merges = [
        ("t", "h"), ("th", "e"), ("the", " "),
        ("a", "n"), ("an", "d"), ("and", " "),
        ("i", "n"), ("in", "g"), ("h", "e"),
        ("he", "l"), ("hel", "l"), ("hell", "o"),
        ("w", "o"), ("wo", "r"), ("wor", "l"), ("worl", "d"),
        ("Ġ", "t"), ("Ġt", "h"), ("Ġth", "e"),
        ("Ġ", "a"), ("Ġa", "n"), ("Ġan", "d"),
        ("Ġ", "i"), ("Ġi", "s"), ("Ġi", "n"),
        ("e", "r"), ("er", " "), ("o", "n"), ("on", " "),
        ("i", "t"), ("it", " "), ("o", "f"), ("of", " "),
        ("t", "o"), ("to", " "), ("a", "t"), ("at", " "),
        ("o", "r"), ("or", " "), ("e", "n"), ("en", " "),
        ("a", "l"), ("al", " "), ("r", "e"), ("re", " "),
        (".", " "), (",", " "),
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

fn create_scalar_bpe() -> BpeModel {
    let (vocab, merge_tuples, _) = create_benchmark_vocab();
    BpeModel::new(vocab, merge_tuples)
}

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

fn create_fast_bpe() -> FastBpeEncoder {
    let (vocab, merge_tuples, _) = create_benchmark_vocab();
    FastBpeEncoder::new(vocab, merge_tuples, "<unk>")
}

fn create_simd_fast_bpe() -> SimdBpeEncoder {
    let (vocab, merge_tuples, _) = create_benchmark_vocab();
    SimdBpeEncoder::new(vocab, merge_tuples, "<unk>")
}

fn create_simd_optimized_bpe() -> SimdOptimizedBpeEncoder {
    let (vocab, merge_tuples, _) = create_benchmark_vocab();
    SimdOptimizedBpeEncoder::new(vocab, merge_tuples, "<unk>")
}

fn generate_text(target_len: usize) -> String {
    let base_texts = [
        "The quick brown fox jumps over the lazy dog. ",
        "Hello world, this is a test of the tokenizer. ",
        "Machine learning enables computers to learn. ",
        "Natural language processing is important. ",
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

// =============================================================================
// SIMD Batch Size Benchmarks (c = 1 to 10000, seq_len = 500)
// =============================================================================

fn bench_simd_batch_sizes(c: &mut Criterion) {
    let bpe = create_hyper_bpe();
    let seq_len = 500;
    let batch_sizes = [1, 10, 100, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("simd_c_scale_seq500");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    for &batch_size in &batch_sizes {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("c", batch_size),
            &batch_refs,
            |b, batch| b.iter(|| black_box(bpe.encode_batch(batch)))
        );
    }

    group.finish();
}

// =============================================================================
// SIMD Sequence Length Benchmarks (seq_len = 100 to 10000, c = 100)
// =============================================================================

fn bench_simd_seq_lengths(c: &mut Criterion) {
    let bpe = create_hyper_bpe();
    let batch_size = 100;
    let seq_lengths = [100, 250, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("simd_len_scale_c100");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    for &seq_len in &seq_lengths {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("len", seq_len),
            &batch_refs,
            |b, batch| b.iter(|| black_box(bpe.encode_batch(batch)))
        );
    }

    group.finish();
}

// =============================================================================
// SIMD Full Matrix (c × seq_len)
// =============================================================================

fn bench_simd_matrix(c: &mut Criterion) {
    let bpe = create_hyper_bpe();

    let configs = [
        // Small batches, varying lengths
        (1, 100), (1, 500), (1, 1000), (1, 5000), (1, 10000),
        // Medium batches
        (10, 100), (10, 500), (10, 1000), (10, 5000), (10, 10000),
        (100, 100), (100, 500), (100, 1000), (100, 5000), (100, 10000),
        // Large batches
        (500, 100), (500, 500), (500, 1000), (500, 5000),
        (1000, 100), (1000, 500), (1000, 1000), (1000, 5000),
        (2000, 100), (2000, 500), (2000, 1000),
        (5000, 100), (5000, 500), (5000, 1000),
        (10000, 100), (10000, 500),
    ];

    let mut group = c.benchmark_group("simd_matrix");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &(batch_size, seq_len) in &configs {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("c{}_len{}", batch_size, seq_len), total_bytes),
            &batch_refs,
            |b, batch| b.iter(|| black_box(bpe.encode_batch(batch)))
        );
    }

    group.finish();
}

// =============================================================================
// Scalar Baseline (smaller sample for comparison)
// =============================================================================

fn bench_scalar_baseline(c: &mut Criterion) {
    let bpe = create_scalar_bpe();

    // Representative subset for scalar comparison
    let configs = [
        (1, 100), (1, 500), (1, 1000),
        (10, 100), (10, 500),
        (100, 100), (100, 500),
    ];

    let mut group = c.benchmark_group("scalar_baseline");
    group.measurement_time(Duration::from_secs(5));
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
// Single Sequence Performance
// =============================================================================

fn bench_single_sequence(c: &mut Criterion) {
    let scalar_bpe = create_scalar_bpe();
    let simd_bpe = create_hyper_bpe();
    let fast_bpe = create_fast_bpe();

    let seq_lengths = [100, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("single_seq");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    for &seq_len in &seq_lengths {
        let text = generate_text(seq_len);

        group.throughput(Throughput::Bytes(seq_len as u64));

        group.bench_with_input(
            BenchmarkId::new("simd", seq_len),
            &text,
            |b, text| b.iter(|| black_box(simd_bpe.encode_fast(text)))
        );

        group.bench_with_input(
            BenchmarkId::new("fast", seq_len),
            &text,
            |b, text| b.iter(|| black_box(fast_bpe.encode(text)))
        );
    }

    // Scalar only for smaller sequences (it's slow)
    for &seq_len in &[100, 500, 1000] {
        let text = generate_text(seq_len);

        group.throughput(Throughput::Bytes(seq_len as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar", seq_len),
            &text,
            |b, text| b.iter(|| black_box(scalar_bpe.encode(text)))
        );
    }

    group.finish();
}

// =============================================================================
// Fast BPE Performance (comparison with HF tokenizers)
// =============================================================================

fn bench_fast_bpe(c: &mut Criterion) {
    let fast_bpe = create_fast_bpe();
    let seq_lengths = [100, 250, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("fast_bpe_single");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    for &seq_len in &seq_lengths {
        let text = generate_text(seq_len);

        group.throughput(Throughput::Bytes(seq_len as u64));

        group.bench_with_input(
            BenchmarkId::new("len", seq_len),
            &text,
            |b, text| b.iter(|| black_box(fast_bpe.encode(text)))
        );
    }

    group.finish();
}

fn bench_fast_bpe_batch(c: &mut Criterion) {
    let fast_bpe = create_fast_bpe();
    let batch_size = 100;
    let seq_lengths = [100, 250, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("fast_bpe_batch_c100");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    for &seq_len in &seq_lengths {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("len", seq_len),
            &batch_refs,
            |b, batch| b.iter(|| black_box(fast_bpe.encode_batch(batch)))
        );
    }

    group.finish();
}

fn bench_fast_bpe_batch_sizes(c: &mut Criterion) {
    let fast_bpe = create_fast_bpe();
    let seq_len = 500;
    let batch_sizes = [1, 10, 100, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("fast_bpe_c_scale_seq500");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    for &batch_size in &batch_sizes {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("c", batch_size),
            &batch_refs,
            |b, batch| b.iter(|| black_box(fast_bpe.encode_batch(batch)))
        );
    }

    group.finish();
}

fn bench_simd_fast_bpe_batch(c: &mut Criterion) {
    let simd_bpe = create_simd_fast_bpe();
    let batch_size = 100;
    let seq_lengths = [100, 250, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("simd_fast_bpe_batch_c100");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    for &seq_len in &seq_lengths {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("len", seq_len),
            &batch_refs,
            |b, batch| b.iter(|| black_box(simd_bpe.encode_batch(batch)))
        );
    }

    group.finish();
}

// =============================================================================
// SIMD Optimized BPE (AVX-512/AVX2/SSE4.2/NEON with auto-detection)
// =============================================================================

fn bench_simd_optimized_bpe_batch(c: &mut Criterion) {
    let simd_bpe = create_simd_optimized_bpe();
    println!("SIMD Optimized BPE using: {:?}", detected_simd_level());

    let batch_size = 100;
    let seq_lengths = [100, 250, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("simd_optimized_bpe_batch_c100");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    for &seq_len in &seq_lengths {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("len", seq_len),
            &batch_refs,
            |b, batch| b.iter(|| black_box(simd_bpe.encode_batch(batch)))
        );
    }

    group.finish();
}

fn bench_simd_optimized_vs_fast(c: &mut Criterion) {
    let fast_bpe = create_fast_bpe();
    let simd_optimized_bpe = create_simd_optimized_bpe();

    println!("Comparing FastBPE vs SimdOptimizedBPE ({:?})", detected_simd_level());

    let batch_size = 100;
    let seq_lengths = [1000, 5000, 10000];

    let mut group = c.benchmark_group("fast_vs_simd_optimized");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    for &seq_len in &seq_lengths {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("fast_bpe", seq_len),
            &batch_refs,
            |b, batch| b.iter(|| black_box(fast_bpe.encode_batch(batch)))
        );

        group.bench_with_input(
            BenchmarkId::new("simd_optimized", seq_len),
            &batch_refs,
            |b, batch| b.iter(|| black_box(simd_optimized_bpe.encode_batch(batch)))
        );
    }

    group.finish();
}

fn bench_simd_optimized_single(c: &mut Criterion) {
    let simd_bpe = create_simd_optimized_bpe();
    let seq_lengths = [100, 500, 1000, 5000, 10000];

    let mut group = c.benchmark_group("simd_optimized_single");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    for &seq_len in &seq_lengths {
        let text = generate_text(seq_len);

        group.throughput(Throughput::Bytes(seq_len as u64));
        group.bench_with_input(
            BenchmarkId::new("len", seq_len),
            &text,
            |b, text| b.iter(|| black_box(simd_bpe.encode(text)))
        );
    }

    group.finish();
}

// =============================================================================
// Throughput at Scale (fixed ~10MB total)
// =============================================================================

fn bench_throughput_10mb(c: &mut Criterion) {
    let bpe = create_hyper_bpe();

    // ~10MB total data, different configurations
    let configs = [
        (10000, 1000),  // Many medium sequences
        (5000, 2000),   // Balanced
        (2000, 5000),   // Fewer longer sequences
        (1000, 10000),  // Few very long sequences
    ];

    let mut group = c.benchmark_group("throughput_10mb");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(15);

    for &(batch_size, seq_len) in &configs {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("c{}_len{}", batch_size, seq_len), total_bytes),
            &batch_refs,
            |b, batch| b.iter(|| black_box(bpe.encode_batch(batch)))
        );
    }

    group.finish();
}

criterion_group!(
    name = scale_benches;
    config = Criterion::default()
        .significance_level(0.05)
        .noise_threshold(0.03);
    targets =
        bench_simd_batch_sizes,
        bench_simd_seq_lengths,
        bench_simd_matrix,
        bench_scalar_baseline,
        bench_single_sequence,
        bench_fast_bpe,
        bench_fast_bpe_batch,
        bench_fast_bpe_batch_sizes,
        bench_simd_fast_bpe_batch,
        bench_simd_optimized_bpe_batch,
        bench_simd_optimized_vs_fast,
        bench_simd_optimized_single,
        bench_throughput_10mb
);

criterion_main!(scale_benches);
