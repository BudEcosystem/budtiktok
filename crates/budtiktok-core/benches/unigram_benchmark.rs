//! Unigram Tokenizer Benchmark
//!
//! Compares performance of:
//! - UnigramFast (optimized implementation with DAT, single-pass Viterbi, SIMD)
//! - UnigramTokenizer (original implementation)
//! - HuggingFace tokenizers (baseline)
//!
//! Benchmarks:
//! - Single sequence encoding at various lengths
//! - Batch encoding with different batch sizes
//! - Multi-core parallel batch encoding

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;
use std::path::Path;

// BudTikTok implementations
use budtiktok_core::unigram::UnigramPiece;
use budtiktok_core::unigram_fast::{UnigramFast, UnigramFastConfig};

// HuggingFace tokenizers for comparison
use tokenizers::Tokenizer as HfTokenizer;

// =============================================================================
// Constants
// =============================================================================

const XLNET_TOKENIZER_PATH: &str = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/xlnet/tokenizer.json";

// =============================================================================
// Test Data Generation
// =============================================================================

fn generate_text(target_len: usize) -> String {
    let base_texts = [
        "The quick brown fox jumps over the lazy dog. ",
        "Hello world, this is a test of the tokenizer. ",
        "Machine learning enables computers to learn from data. ",
        "Natural language processing is a key component of AI. ",
        "Transformers have revolutionized the field of NLP. ",
        "Attention mechanisms allow models to focus on relevant parts. ",
        "Pre-training on large corpora improves downstream performance. ",
        "Fine-tuning adapts models to specific tasks efficiently. ",
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
            // Add some variation
            if i % 3 == 0 {
                text = text.to_uppercase();
            } else if i % 3 == 1 {
                text = text.replace(" ", "  ");
            }
            text
        })
        .collect()
}

// =============================================================================
// Tokenizer Loading
// =============================================================================

fn load_hf_tokenizer() -> Option<HfTokenizer> {
    if Path::new(XLNET_TOKENIZER_PATH).exists() {
        HfTokenizer::from_file(XLNET_TOKENIZER_PATH).ok()
    } else {
        None
    }
}

fn load_unigram_fast_tokenizer() -> Option<UnigramFast> {
    if Path::new(XLNET_TOKENIZER_PATH).exists() {
        // Load via the tokenizer.json format
        let json_content = std::fs::read_to_string(XLNET_TOKENIZER_PATH).ok()?;
        let json: serde_json::Value = serde_json::from_str(&json_content).ok()?;

        // Extract vocab and pieces from JSON
        let model = json.get("model")?;
        let vocab_array = model.get("vocab")?.as_array()?;

        let mut vocab = Vec::with_capacity(vocab_array.len());
        let mut pieces = Vec::with_capacity(vocab_array.len());

        for item in vocab_array {
            let arr = item.as_array()?;
            let token = arr.get(0)?.as_str()?;
            let score = arr.get(1)?.as_f64()?;
            vocab.push(token.to_string());
            pieces.push(UnigramPiece {
                token: token.to_string(),
                score,
            });
        }

        // Get unk_id
        let unk_id = model.get("unk_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(0);

        let config = UnigramFastConfig {
            unk_id,
            add_prefix_space: true,
            ..Default::default()
        };

        Some(UnigramFast::new(vocab, pieces, config))
    } else {
        None
    }
}

// =============================================================================
// Single Sequence Benchmarks
// =============================================================================

fn bench_single_sequence(c: &mut Criterion) {
    let hf_tok = match load_hf_tokenizer() {
        Some(t) => t,
        None => {
            eprintln!("Skipping benchmark: XLNet tokenizer not found at {}", XLNET_TOKENIZER_PATH);
            return;
        }
    };

    let fast_tok = load_unigram_fast_tokenizer().expect("Failed to load UnigramFast");

    let seq_lengths = [100, 250, 500, 1000, 2000, 5000];

    let mut group = c.benchmark_group("unigram_single_seq");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(100);

    for &seq_len in &seq_lengths {
        let text = generate_text(seq_len);

        group.throughput(Throughput::Bytes(seq_len as u64));

        // HuggingFace baseline
        group.bench_with_input(
            BenchmarkId::new("hf", seq_len),
            &text,
            |b, text| b.iter(|| black_box(hf_tok.encode(text.as_str(), false).unwrap()))
        );

        // Optimized UnigramFast (fast mode, no normalization)
        group.bench_with_input(
            BenchmarkId::new("unigram_fast", seq_len),
            &text,
            |b, text| b.iter(|| black_box(fast_tok.encode_fast(text)))
        );

        // HF-compatible (100% accuracy match, scalar)
        group.bench_with_input(
            BenchmarkId::new("unigram_hf_compat", seq_len),
            &text,
            |b, text| b.iter(|| black_box(fast_tok.encode_hf_compat(text)))
        );

        // HF-compatible with SIMD optimizations
        group.bench_with_input(
            BenchmarkId::new("unigram_hf_compat_simd", seq_len),
            &text,
            |b, text| b.iter(|| black_box(fast_tok.encode_hf_compat_simd(text)))
        );
    }

    group.finish();
}

// =============================================================================
// Batch Encoding Benchmarks (Single-threaded)
// =============================================================================

fn bench_batch_single_thread(c: &mut Criterion) {
    let hf_tok = match load_hf_tokenizer() {
        Some(t) => t,
        None => {
            eprintln!("Skipping benchmark: XLNet tokenizer not found");
            return;
        }
    };

    let fast_tok = load_unigram_fast_tokenizer().expect("Failed to load UnigramFast");

    let batch_sizes = [1, 10, 50, 100, 500, 1000];
    let seq_len = 500;

    let mut group = c.benchmark_group("unigram_batch_single");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    for &batch_size in &batch_sizes {
        let batch = generate_batch(batch_size, seq_len);
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));

        // HuggingFace baseline (sequential)
        group.bench_with_input(
            BenchmarkId::new("hf", batch_size),
            &batch,
            |b, batch| b.iter(|| {
                batch.iter()
                    .map(|text| hf_tok.encode(text.as_str(), false).unwrap())
                    .collect::<Vec<_>>()
            })
        );

        // UnigramFast sequential
        group.bench_with_input(
            BenchmarkId::new("unigram_fast", batch_size),
            &batch,
            |b, batch| b.iter(|| {
                batch.iter()
                    .map(|text| fast_tok.encode_fast(text))
                    .collect::<Vec<_>>()
            })
        );
    }

    group.finish();
}

// =============================================================================
// Batch Encoding Benchmarks (Multi-threaded)
// =============================================================================

fn bench_batch_parallel(c: &mut Criterion) {
    use tokenizers::tokenizer::EncodeInput;

    let hf_tok = match load_hf_tokenizer() {
        Some(t) => t,
        None => {
            eprintln!("Skipping benchmark: XLNet tokenizer not found");
            return;
        }
    };

    let fast_tok = load_unigram_fast_tokenizer().expect("Failed to load UnigramFast");

    let batch_sizes = [100, 500, 1000, 2000, 5000];
    let seq_len = 500;

    let mut group = c.benchmark_group("unigram_batch_parallel");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(30);

    for &batch_size in &batch_sizes {
        let batch = generate_batch(batch_size, seq_len);
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));

        // HuggingFace parallel (uses encode_batch with internal parallelism)
        let batch_refs: Vec<EncodeInput> = batch.iter()
            .map(|s| EncodeInput::Single(s.clone().into()))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("hf_parallel", batch_size),
            &batch_refs,
            |b, batch| b.iter(|| {
                black_box(hf_tok.encode_batch(batch.clone(), false).unwrap())
            })
        );

        // UnigramFast parallel using Rayon
        let batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        group.bench_with_input(
            BenchmarkId::new("unigram_fast_parallel", batch_size),
            &batch_refs,
            |b, batch| b.iter(|| {
                black_box(fast_tok.encode_batch(batch))
            })
        );
    }

    group.finish();
}

// =============================================================================
// Throughput Benchmarks (~10MB total data)
// =============================================================================

fn bench_throughput_10mb(c: &mut Criterion) {
    use tokenizers::tokenizer::EncodeInput;

    let hf_tok = match load_hf_tokenizer() {
        Some(t) => t,
        None => {
            eprintln!("Skipping benchmark: XLNet tokenizer not found");
            return;
        }
    };

    let fast_tok = load_unigram_fast_tokenizer().expect("Failed to load UnigramFast");

    // ~10MB total data, different configurations
    let configs = [
        (10000, 1000),  // Many medium sequences
        (5000, 2000),   // Balanced
        (2000, 5000),   // Fewer longer sequences
    ];

    let mut group = c.benchmark_group("unigram_throughput_10mb");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);

    for &(batch_size, seq_len) in &configs {
        let batch = generate_batch(batch_size, seq_len);
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();
        let total_mb = total_bytes as f64 / 1_000_000.0;

        group.throughput(Throughput::Bytes(total_bytes as u64));

        // HuggingFace parallel
        let batch_refs: Vec<EncodeInput> = batch.iter()
            .map(|s| EncodeInput::Single(s.clone().into()))
            .collect();

        group.bench_with_input(
            BenchmarkId::new(format!("hf_c{}_len{}_{}mb", batch_size, seq_len, total_mb as u64), total_bytes),
            &batch_refs,
            |b, batch| b.iter(|| {
                black_box(hf_tok.encode_batch(batch.clone(), false).unwrap())
            })
        );

        // UnigramFast parallel
        let fast_batch_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        group.bench_with_input(
            BenchmarkId::new(format!("fast_c{}_len{}_{}mb", batch_size, seq_len, total_mb as u64), total_bytes),
            &fast_batch_refs,
            |b, batch| b.iter(|| {
                black_box(fast_tok.encode_batch(batch))
            })
        );
    }

    group.finish();
}

// =============================================================================
// Component Benchmarks (for optimization analysis)
// =============================================================================

fn bench_components(c: &mut Criterion) {
    let fast_tok = match load_unigram_fast_tokenizer() {
        Some(t) => t,
        None => {
            eprintln!("Skipping benchmark: XLNet tokenizer not found");
            return;
        }
    };

    let seq_lengths = [100, 500, 1000, 5000];

    let mut group = c.benchmark_group("unigram_components");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(100);

    for &seq_len in &seq_lengths {
        let text = generate_text(seq_len);

        group.throughput(Throughput::Bytes(seq_len as u64));

        // Full encoding pipeline
        group.bench_with_input(
            BenchmarkId::new("full", seq_len),
            &text,
            |b, text| b.iter(|| black_box(fast_tok.encode_fast(text)))
        );

        // Just preprocessing (space replacement)
        group.bench_with_input(
            BenchmarkId::new("preprocess", seq_len),
            &text,
            |b, text| b.iter(|| {
                black_box(budtiktok_core::unigram_fast::preprocess_text(text, true))
            })
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    name = unigram_benches;
    config = Criterion::default()
        .significance_level(0.05)
        .noise_threshold(0.03);
    targets =
        bench_single_sequence,
        bench_batch_single_thread,
        bench_batch_parallel,
        bench_throughput_10mb,
        bench_components
);

criterion_main!(unigram_benches);
