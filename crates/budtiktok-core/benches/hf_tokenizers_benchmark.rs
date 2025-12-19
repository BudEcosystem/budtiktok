//! HuggingFace Tokenizers BPE Scale Benchmark
//!
//! Benchmarks HuggingFace tokenizers library (used by text-embeddings-inference)
//! with the same configurations as budtiktok BPE benchmark:
//! - Batch sizes (c): 1, 10, 100, 500, 1000, 2000, 5000, 10000
//! - Sequence lengths: 100, 250, 500, 1000, 2000, 5000, 10000

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use tokenizers::tokenizer::{Tokenizer, EncodeInput};
use tokenizers::models::bpe::BpeBuilder;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use std::time::Duration;
use std::collections::HashMap;

/// GPT-2 style byte encoder
fn gpt2_byte_encoder() -> HashMap<u8, char> {
    let mut encoder = HashMap::new();
    let mut n = 0u32;

    // Printable bytes - map to themselves
    for b in b'!'..=b'~' {
        encoder.insert(b, b as char);
    }
    for b in 0xa1u8..=0xacu8 {
        encoder.insert(b, b as char);
    }
    for b in 0xaeu8..=0xffu8 {
        encoder.insert(b, b as char);
    }

    // Non-printable bytes - map to unicode range starting at 256
    for b in 0u8..=255u8 {
        if !encoder.contains_key(&b) {
            encoder.insert(b, char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }

    encoder
}

/// Create a GPT-2 style BPE tokenizer for benchmarking
fn create_hf_bpe_tokenizer() -> Tokenizer {
    let byte_encoder = gpt2_byte_encoder();

    // Create vocabulary - GPT-2 style byte-level BPE
    let mut vocab: HashMap<String, u32> = HashMap::new();
    let mut merges: Vec<(String, String)> = Vec::new();

    // Add base byte tokens (256 tokens)
    for b in 0u8..=255 {
        let c = byte_encoder[&b];
        vocab.insert(c.to_string(), b as u32);
    }

    let mut next_id = 256u32;

    // Get encoded space character (Ġ in GPT-2)
    let space_char = byte_encoder[&b' '].to_string();

    // Add common merge pairs (GPT-2 style) - using encoded characters
    let common_merges = [
        ("t", "h"), ("th", "e"),
        ("a", "n"), ("an", "d"),
        ("i", "n"), ("in", "g"), ("h", "e"),
        ("he", "l"), ("hel", "l"), ("hell", "o"),
        ("w", "o"), ("wo", "r"), ("wor", "l"), ("worl", "d"),
        ("e", "r"), ("o", "n"),
        ("i", "t"), ("o", "f"),
        ("t", "o"), ("a", "t"),
        ("o", "r"), ("e", "n"),
        ("a", "l"), ("r", "e"),
        // More common English merges
        ("c", "o"), ("co", "m"), ("com", "p"),
        ("s", "t"), ("st", "a"), ("sta", "r"),
        ("m", "a"), ("ma", "c"), ("mac", "h"),
        ("l", "e"), ("le", "a"), ("lea", "r"),
        ("p", "r"), ("pr", "o"), ("pro", "c"),
        ("b", "e"), ("be", "n"), ("ben", "c"),
        ("u", "i"), ("ui", "c"), ("uic", "k"),
        ("b", "r"), ("br", "o"), ("bro", "w"), ("brow", "n"),
        ("f", "o"), ("fo", "x"),
        ("j", "u"), ("ju", "m"), ("jum", "p"), ("jump", "s"),
        ("o", "v"), ("ov", "e"), ("ove", "r"),
        ("l", "a"), ("la", "z"), ("laz", "y"),
        ("d", "o"), ("do", "g"),
    ];

    for (first, second) in &common_merges {
        let merged = format!("{}{}", first, second);
        if !vocab.contains_key(&merged) {
            vocab.insert(merged.clone(), next_id);
            next_id += 1;
        }
        // Only add merge if both tokens are in vocab
        if vocab.contains_key(*first) && vocab.contains_key(*second) {
            merges.push((first.to_string(), second.to_string()));
        }
    }

    // Add space-prefixed merges (Ġ + word)
    let space_merge_targets = ["t", "a", "i", "o", "e", "s", "n", "r", "l"];

    for second in &space_merge_targets {
        let merged = format!("{}{}", space_char, second);
        if !vocab.contains_key(&merged) {
            vocab.insert(merged.clone(), next_id);
            next_id += 1;
        }
        if vocab.contains_key(&space_char) && vocab.contains_key(*second) {
            merges.push((space_char.clone(), second.to_string()));
        }
    }

    // Add [UNK] token
    vocab.insert("[UNK]".to_string(), next_id);

    // Build BPE model
    let bpe = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .unk_token("[UNK]".to_string())
        .build()
        .expect("Failed to build BPE model");

    let mut tokenizer = Tokenizer::new(bpe);

    // Add byte-level pre-tokenizer (GPT-2 style)
    tokenizer.with_pre_tokenizer(ByteLevel::default());
    tokenizer.with_decoder(ByteLevelDecoder::default());
    tokenizer.with_post_processor(ByteLevel::default());

    tokenizer
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
// HF Tokenizers Batch Size Benchmarks (c = 1 to 10000, seq_len = 500)
// =============================================================================

fn bench_hf_batch_sizes(c: &mut Criterion) {
    let tokenizer = create_hf_bpe_tokenizer();
    let seq_len = 500;
    let batch_sizes = [1, 10, 100, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("hf_c_scale_seq500");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    for &batch_size in &batch_sizes {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<EncodeInput> = batch.iter()
            .map(|s| EncodeInput::Single(s.clone().into()))
            .collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("c", batch_size),
            &batch_refs,
            |b, batch| b.iter(|| {
                black_box(tokenizer.encode_batch(batch.clone(), false).unwrap())
            })
        );
    }

    group.finish();
}

// =============================================================================
// HF Tokenizers Sequence Length Benchmarks (seq_len = 100 to 10000, c = 100)
// =============================================================================

fn bench_hf_seq_lengths(c: &mut Criterion) {
    let tokenizer = create_hf_bpe_tokenizer();
    let batch_size = 100;
    let seq_lengths = [100, 250, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("hf_len_scale_c100");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(30);

    for &seq_len in &seq_lengths {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<EncodeInput> = batch.iter()
            .map(|s| EncodeInput::Single(s.clone().into()))
            .collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("len", seq_len),
            &batch_refs,
            |b, batch| b.iter(|| {
                black_box(tokenizer.encode_batch(batch.clone(), false).unwrap())
            })
        );
    }

    group.finish();
}

// =============================================================================
// HF Tokenizers Full Matrix (c × seq_len)
// =============================================================================

fn bench_hf_matrix(c: &mut Criterion) {
    let tokenizer = create_hf_bpe_tokenizer();

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

    let mut group = c.benchmark_group("hf_matrix");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(20);

    for &(batch_size, seq_len) in &configs {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<EncodeInput> = batch.iter()
            .map(|s| EncodeInput::Single(s.clone().into()))
            .collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("c{}_len{}", batch_size, seq_len), total_bytes),
            &batch_refs,
            |b, batch| b.iter(|| {
                black_box(tokenizer.encode_batch(batch.clone(), false).unwrap())
            })
        );
    }

    group.finish();
}

// =============================================================================
// Single Sequence Performance (HF vs budtiktok comparison)
// =============================================================================

fn bench_hf_single_sequence(c: &mut Criterion) {
    let tokenizer = create_hf_bpe_tokenizer();
    let seq_lengths = [100, 500, 1000, 2000, 5000, 10000];

    let mut group = c.benchmark_group("hf_single_seq");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    for &seq_len in &seq_lengths {
        let text = generate_text(seq_len);

        group.throughput(Throughput::Bytes(seq_len as u64));

        group.bench_with_input(
            BenchmarkId::new("hf", seq_len),
            &text,
            |b, text| b.iter(|| black_box(tokenizer.encode(text.as_str(), false).unwrap()))
        );
    }

    group.finish();
}

// =============================================================================
// Throughput at Scale (fixed ~10MB total)
// =============================================================================

fn bench_hf_throughput_10mb(c: &mut Criterion) {
    let tokenizer = create_hf_bpe_tokenizer();

    // ~10MB total data, different configurations
    let configs = [
        (10000, 1000),  // Many medium sequences
        (5000, 2000),   // Balanced
        (2000, 5000),   // Fewer longer sequences
        (1000, 10000),  // Few very long sequences
    ];

    let mut group = c.benchmark_group("hf_throughput_10mb");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(15);

    for &(batch_size, seq_len) in &configs {
        let batch = generate_batch(batch_size, seq_len);
        let batch_refs: Vec<EncodeInput> = batch.iter()
            .map(|s| EncodeInput::Single(s.clone().into()))
            .collect();
        let total_bytes: usize = batch.iter().map(|s| s.len()).sum();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("c{}_len{}", batch_size, seq_len), total_bytes),
            &batch_refs,
            |b, batch| b.iter(|| {
                black_box(tokenizer.encode_batch(batch.clone(), false).unwrap())
            })
        );
    }

    group.finish();
}

criterion_group!(
    name = hf_scale_benches;
    config = Criterion::default()
        .significance_level(0.05)
        .noise_threshold(0.03);
    targets =
        bench_hf_batch_sizes,
        bench_hf_seq_lengths,
        bench_hf_matrix,
        bench_hf_single_sequence,
        bench_hf_throughput_10mb
);

criterion_main!(hf_scale_benches);
