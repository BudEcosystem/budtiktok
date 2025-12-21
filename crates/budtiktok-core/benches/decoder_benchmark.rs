//! Benchmark for decoder performance vs HuggingFace Tokenizers
//!
//! Run with: cargo bench --bench decoder_benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::time::Duration;

// BudTikTok decoders
use budtiktok_core::{
    Decoder, ByteLevelDecoder, MetaspaceDecoder, WordPieceDecoder,
    ByteFallbackDecoder, FuseDecoder, CTCDecoder,
    byte_to_gpt2_char,
};

// Generate test data
fn generate_gpt2_tokens(count: usize) -> Vec<String> {
    // Generate tokens that look like GPT-2 byte-level encoded text
    let mut tokens = Vec::with_capacity(count);
    let sample_tokens = [
        "Hello", "Ġworld", "Ġhow", "Ġare", "Ġyou", "Ġdoing", "Ġtoday", "?",
        "ĠI", "Ġam", "Ġfine", ",", "Ġthank", "Ġyou", "!", "ĠThe", "Ġquick",
        "Ġbrown", "Ġfox", "Ġjumps", "Ġover", "Ġthe", "Ġlazy", "Ġdog", ".",
    ];
    for i in 0..count {
        tokens.push(sample_tokens[i % sample_tokens.len()].to_string());
    }
    tokens
}

fn generate_metaspace_tokens(count: usize) -> Vec<String> {
    let mut tokens = Vec::with_capacity(count);
    let sample_tokens = [
        "▁Hello", "▁world", "▁how", "▁are", "▁you", "▁doing", "▁today",
        "▁I", "▁am", "▁fine", "▁thank", "▁you", "▁The", "▁quick",
        "▁brown", "▁fox", "▁jumps", "▁over", "▁the", "▁lazy", "▁dog",
    ];
    for i in 0..count {
        tokens.push(sample_tokens[i % sample_tokens.len()].to_string());
    }
    tokens
}

fn generate_wordpiece_tokens(count: usize) -> Vec<String> {
    let mut tokens = Vec::with_capacity(count);
    let sample_tokens = [
        "hello", "##world", "how", "##are", "you", "##doing", "today",
        "I", "am", "##fine", "thank", "##you", "The", "##quick",
        "brown", "##fox", "jump", "##s", "over", "the", "lazy", "##dog",
    ];
    for i in 0..count {
        tokens.push(sample_tokens[i % sample_tokens.len()].to_string());
    }
    tokens
}

fn generate_byte_fallback_tokens(count: usize) -> Vec<String> {
    let mut tokens = Vec::with_capacity(count);
    // Mix of regular tokens and byte fallback tokens
    let sample_tokens = [
        "Hello", "<0xE5>", "<0x8f>", "<0xab>", "world", // 叫
        "test", "<0x61>", "<0x62>", "<0x63>", // abc
        "foo", "bar", "baz",
    ];
    for i in 0..count {
        tokens.push(sample_tokens[i % sample_tokens.len()].to_string());
    }
    tokens
}

fn generate_ctc_tokens(count: usize) -> Vec<String> {
    let mut tokens = Vec::with_capacity(count);
    // CTC tokens with duplicates and pad tokens
    let sample_tokens = [
        "<pad>", "H", "H", "e", "l", "l", "<pad>", "o", "o",
        "|", "w", "o", "r", "l", "d", "<pad>",
    ];
    for i in 0..count {
        tokens.push(sample_tokens[i % sample_tokens.len()].to_string());
    }
    tokens
}

fn bench_bytelevel_decoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("ByteLevel Decoder");
    group.measurement_time(Duration::from_secs(5));

    for size in [100, 1000, 10000] {
        let tokens = generate_gpt2_tokens(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(format!("budtiktok_{}", size), |b| {
            let decoder = ByteLevelDecoder::new();
            b.iter(|| {
                decoder.decode(black_box(tokens.clone())).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_metaspace_decoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("Metaspace Decoder");
    group.measurement_time(Duration::from_secs(5));

    for size in [100, 1000, 10000] {
        let tokens = generate_metaspace_tokens(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(format!("budtiktok_{}", size), |b| {
            let decoder = MetaspaceDecoder::default();
            b.iter(|| {
                decoder.decode(black_box(tokens.clone())).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_wordpiece_decoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("WordPiece Decoder");
    group.measurement_time(Duration::from_secs(5));

    for size in [100, 1000, 10000] {
        let tokens = generate_wordpiece_tokens(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(format!("budtiktok_{}", size), |b| {
            let decoder = WordPieceDecoder::default();
            b.iter(|| {
                decoder.decode(black_box(tokens.clone())).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_byte_fallback_decoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("ByteFallback Decoder");
    group.measurement_time(Duration::from_secs(5));

    for size in [100, 1000, 10000] {
        let tokens = generate_byte_fallback_tokens(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(format!("budtiktok_{}", size), |b| {
            let decoder = ByteFallbackDecoder::new();
            b.iter(|| {
                decoder.decode(black_box(tokens.clone())).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_fuse_decoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("Fuse Decoder");
    group.measurement_time(Duration::from_secs(5));

    for size in [100, 1000, 10000] {
        let tokens = generate_gpt2_tokens(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(format!("budtiktok_{}", size), |b| {
            let decoder = FuseDecoder::new();
            b.iter(|| {
                decoder.decode(black_box(tokens.clone())).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_ctc_decoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("CTC Decoder");
    group.measurement_time(Duration::from_secs(5));

    for size in [100, 1000, 10000] {
        let tokens = generate_ctc_tokens(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(format!("budtiktok_{}", size), |b| {
            let decoder = CTCDecoder::default();
            b.iter(|| {
                decoder.decode(black_box(tokens.clone())).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_lookup_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("Lookup Table");

    // Benchmark the static lookup table vs HashMap approach
    group.bench_function("static_table_byte_to_char", |b| {
        b.iter(|| {
            let mut sum = 0u32;
            for byte in 0..=255u8 {
                let ch = byte_to_gpt2_char(byte);
                sum += ch as u32;
            }
            black_box(sum)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_bytelevel_decoder,
    bench_metaspace_decoder,
    bench_wordpiece_decoder,
    bench_byte_fallback_decoder,
    bench_fuse_decoder,
    bench_ctc_decoder,
    bench_lookup_table,
);

criterion_main!(benches);
