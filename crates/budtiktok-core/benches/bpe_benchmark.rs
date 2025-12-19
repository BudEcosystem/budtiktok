//! BPE Performance Benchmarks
//!
//! Benchmarks scalar and SIMD BPE implementations against various workloads.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use budtiktok_core::bpe::{BpeModel, Gpt2ByteEncoder, MergeRule};
use budtiktok_core::bpe_hyper::{HyperBpeTokenizer, HyperBpeConfig};
use ahash::AHashMap;

/// Create comprehensive test vocabulary (GPT-2 style)
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

    // Add common merged tokens
    let common_merges = [
        ("t", "h"), ("th", "e"), ("the", " "),
        ("a", "n"), ("an", "d"), ("and", " "),
        ("i", "n"), ("in", "g"),
        ("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o"),
        ("w", "o"), ("wo", "r"), ("wor", "l"), ("worl", "d"),
        ("Ġ", "t"), ("Ġt", "h"), ("Ġth", "e"),
        ("Ġ", "a"), ("Ġa", "n"), ("Ġan", "d"),
        ("Ġ", "i"), ("Ġi", "s"),
        ("e", "r"), ("er", " "),
        ("o", "n"), ("on", " "),
        ("i", "t"), ("it", " "),
        ("o", "f"), ("of", " "),
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

/// Sample texts of various lengths
fn sample_texts() -> Vec<&'static str> {
    vec![
        "hello",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "This is a longer sentence with more words and punctuation marks, including commas!",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    ]
}

fn bench_scalar_encode(c: &mut Criterion) {
    let bpe = create_scalar_bpe();
    let texts = sample_texts();

    let mut group = c.benchmark_group("scalar_bpe_encode");

    for text in &texts {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("encode", text.len()),
            text,
            |b, text| {
                b.iter(|| black_box(bpe.encode(text)))
            }
        );
    }

    group.finish();
}

fn bench_hyper_encode(c: &mut Criterion) {
    let bpe = create_hyper_bpe();
    let texts = sample_texts();

    let mut group = c.benchmark_group("simd_bpe_encode");

    for text in &texts {
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("encode", text.len()),
            text,
            |b, text| {
                b.iter(|| black_box(bpe.encode_fast(text)))
            }
        );
    }

    group.finish();
}

fn bench_batch_encode(c: &mut Criterion) {
    let scalar_bpe = create_scalar_bpe();
    let hyper_bpe = create_hyper_bpe();

    let batch: Vec<&str> = (0..100)
        .map(|i| match i % 5 {
            0 => "hello",
            1 => "Hello, world!",
            2 => "The quick brown fox jumps over the lazy dog.",
            3 => "This is a test sentence.",
            _ => "Lorem ipsum dolor sit amet.",
        })
        .collect();

    let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

    let mut group = c.benchmark_group("batch_encode_100");
    group.throughput(Throughput::Bytes(total_bytes as u64));

    group.bench_function("scalar", |b| {
        b.iter(|| {
            let results: Vec<_> = batch.iter()
                .map(|text| black_box(scalar_bpe.encode(text)))
                .collect();
            results
        })
    });

    group.bench_function("simd_parallel", |b| {
        b.iter(|| black_box(hyper_bpe.encode_batch(&batch)))
    });

    group.finish();
}

fn bench_throughput_comparison(c: &mut Criterion) {
    let scalar_bpe = create_scalar_bpe();
    let hyper_bpe = create_hyper_bpe();

    // Generate realistic workload
    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let text_bytes = long_text.len();

    let mut group = c.benchmark_group("throughput_comparison");
    group.throughput(Throughput::Bytes(text_bytes as u64));

    group.bench_function("scalar_long_text", |b| {
        b.iter(|| black_box(scalar_bpe.encode(&long_text)))
    });

    group.bench_function("simd_long_text", |b| {
        b.iter(|| black_box(hyper_bpe.encode_fast(&long_text)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_scalar_encode,
    bench_hyper_encode,
    bench_batch_encode,
    bench_throughput_comparison,
);

criterion_main!(benches);
