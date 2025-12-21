//! WordLevel Tokenizer Benchmark
//!
//! Compares BudTikTok WordLevel tokenizer performance with HuggingFace Tokenizers.
//! Tests both scalar and SIMD implementations.
//!
//! Run with: cargo bench --bench wordlevel_benchmark

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::str::FromStr;
use ahash::AHashMap;

use budtiktok_core::{
    WordLevelTokenizer, WordLevelConfig, WordLevelCache,
    pretokenize_scalar, pretokenize_simd,
    is_avx512_available,
};
use budtiktok_core::wordlevel::is_avx2_available;

// ============================================================================
// Test Data Generation
// ============================================================================

fn generate_test_texts() -> Vec<String> {
    vec![
        // Short sentences
        "Hello, world!".to_string(),
        "The quick brown fox jumps over the lazy dog.".to_string(),

        // Medium sentences
        "This is a test sentence with multiple words and punctuation marks, including commas, periods, and exclamation points!".to_string(),

        // Long paragraph
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The result is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them.".to_string(),

        // Technical text with many punctuation marks
        "function foo(x: i32, y: i32) -> bool { return x > 0 && y < 100; }".to_string(),

        // Repeated words (good for cache testing)
        "the the the the the quick quick quick brown brown fox fox fox".to_string(),
    ]
}

fn generate_large_text(size_kb: usize) -> String {
    let base = "The quick brown fox jumps over the lazy dog. ";
    let repeats = (size_kb * 1024) / base.len() + 1;
    base.repeat(repeats)
}

fn create_test_vocab() -> AHashMap<String, u32> {
    let mut vocab = AHashMap::new();
    // Common words
    let words = [
        "<unk>", "<pad>", "the", "a", "an", "is", "are", "was", "were",
        "hello", "world", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "this", "that", "with", "for", "and", "but", "or", "not", "be", "to",
        "of", "in", "it", "on", "at", "as", "by", "from", "up", "about",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "too", "very", "can", "will",
        "just", "should", "now", "language", "processing", "natural", "computer",
        "science", "data", "analysis", "program", "test", "sentence", "multiple",
        "words", "punctuation", "marks", "including", "commas", "periods",
        // Punctuation
        ".", ",", "!", "?", ":", ";", "-", "(", ")", "[", "]", "{", "}",
        "'", "\"", "/", "\\", "@", "#", "$", "%", "^", "&", "*", "+", "=",
    ];

    for (id, word) in words.iter().enumerate() {
        vocab.insert(word.to_string(), id as u32);
    }

    vocab
}

// ============================================================================
// Pre-tokenization Benchmarks
// ============================================================================

fn bench_pretokenize(c: &mut Criterion) {
    let mut group = c.benchmark_group("pretokenize");

    let config = WordLevelConfig::default();

    // Short text
    let short_text = "Hello, world! This is a test.";
    group.throughput(Throughput::Bytes(short_text.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("scalar/short", short_text.len()),
        short_text,
        |b, text| {
            b.iter(|| pretokenize_scalar(black_box(text), &config));
        },
    );
    group.bench_with_input(
        BenchmarkId::new("simd/short", short_text.len()),
        short_text,
        |b, text| {
            b.iter(|| pretokenize_simd(black_box(text), &config));
        },
    );

    // Medium text
    let medium_text = generate_large_text(1); // 1 KB
    group.throughput(Throughput::Bytes(medium_text.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("scalar/1kb", medium_text.len()),
        &medium_text,
        |b, text| {
            b.iter(|| pretokenize_scalar(black_box(text), &config));
        },
    );
    group.bench_with_input(
        BenchmarkId::new("simd/1kb", medium_text.len()),
        &medium_text,
        |b, text| {
            b.iter(|| pretokenize_simd(black_box(text), &config));
        },
    );

    // Large text
    let large_text = generate_large_text(100); // 100 KB
    group.throughput(Throughput::Bytes(large_text.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("scalar/100kb", large_text.len()),
        &large_text,
        |b, text| {
            b.iter(|| pretokenize_scalar(black_box(text), &config));
        },
    );
    group.bench_with_input(
        BenchmarkId::new("simd/100kb", large_text.len()),
        &large_text,
        |b, text| {
            b.iter(|| pretokenize_simd(black_box(text), &config));
        },
    );

    group.finish();
}

// ============================================================================
// Full Tokenization Benchmarks
// ============================================================================

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");

    let vocab = create_test_vocab();
    let tokenizer = WordLevelTokenizer::new(vocab.clone(), WordLevelConfig::default());
    let tokenizer_with_cache = WordLevelTokenizer::with_cache(vocab, WordLevelConfig::default(), 10000);

    let texts = generate_test_texts();
    let total_bytes: usize = texts.iter().map(|t| t.len()).sum();

    group.throughput(Throughput::Bytes(total_bytes as u64));

    // Scalar encode
    group.bench_function("scalar/batch", |b| {
        b.iter(|| {
            for text in &texts {
                black_box(tokenizer.encode_scalar(text));
            }
        });
    });

    // SIMD encode
    group.bench_function("simd/batch", |b| {
        b.iter(|| {
            for text in &texts {
                black_box(tokenizer.encode_simd(text));
            }
        });
    });

    // With cache (warm)
    // Warm up the cache first
    for text in &texts {
        tokenizer_with_cache.encode_scalar(text);
    }

    group.bench_function("scalar_cached/batch", |b| {
        b.iter(|| {
            for text in &texts {
                black_box(tokenizer_with_cache.encode_scalar(text));
            }
        });
    });

    group.finish();
}

// ============================================================================
// HuggingFace Comparison
// ============================================================================

fn bench_vs_huggingface(c: &mut Criterion) {
    use tokenizers::Tokenizer;

    let mut group = c.benchmark_group("vs_huggingface");

    // Create a WordLevel tokenizer in HuggingFace format
    let hf_json = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {
                "<unk>": 0, "the": 1, "a": 2, "is": 3, "quick": 4, "brown": 5,
                "fox": 6, "jumps": 7, "over": 8, "lazy": 9, "dog": 10, ".": 11,
                ",": 12, "!": 13, "?": 14, "hello": 15, "world": 16, "this": 17,
                "test": 18, "with": 19, "multiple": 20, "words": 21
            },
            "unk_token": "<unk>"
        }
    }"#;

    let hf_tokenizer: Option<Tokenizer> = match Tokenizer::from_str(hf_json) {
        Ok(t) => Some(t),
        Err(e) => {
            eprintln!("Failed to create HF tokenizer: {}", e);
            None
        }
    };

    // Create BudTikTok tokenizer with same vocab
    let mut vocab = AHashMap::new();
    vocab.insert("<unk>".to_string(), 0);
    vocab.insert("the".to_string(), 1);
    vocab.insert("a".to_string(), 2);
    vocab.insert("is".to_string(), 3);
    vocab.insert("quick".to_string(), 4);
    vocab.insert("brown".to_string(), 5);
    vocab.insert("fox".to_string(), 6);
    vocab.insert("jumps".to_string(), 7);
    vocab.insert("over".to_string(), 8);
    vocab.insert("lazy".to_string(), 9);
    vocab.insert("dog".to_string(), 10);
    vocab.insert(".".to_string(), 11);
    vocab.insert(",".to_string(), 12);
    vocab.insert("!".to_string(), 13);
    vocab.insert("?".to_string(), 14);
    vocab.insert("hello".to_string(), 15);
    vocab.insert("world".to_string(), 16);
    vocab.insert("this".to_string(), 17);
    vocab.insert("test".to_string(), 18);
    vocab.insert("with".to_string(), 19);
    vocab.insert("multiple".to_string(), 20);
    vocab.insert("words".to_string(), 21);

    let bud_tokenizer = WordLevelTokenizer::new(vocab, WordLevelConfig::default());

    // Test text
    let test_text = "the quick brown fox jumps over the lazy dog";

    // Single text benchmark
    group.throughput(Throughput::Bytes(test_text.len() as u64));

    if let Some(ref hf_tok) = hf_tokenizer {
        group.bench_function("huggingface/single", |b| {
            b.iter(|| {
                black_box(hf_tok.encode(test_text, false).unwrap());
            });
        });
    }

    group.bench_function("budtiktok_scalar/single", |b| {
        b.iter(|| {
            black_box(bud_tokenizer.encode_scalar(test_text));
        });
    });

    group.bench_function("budtiktok_simd/single", |b| {
        b.iter(|| {
            black_box(bud_tokenizer.encode_simd(test_text));
        });
    });

    // Batch benchmark
    let batch: Vec<&str> = vec![test_text; 100];
    let batch_bytes: usize = batch.iter().map(|t| t.len()).sum();
    group.throughput(Throughput::Bytes(batch_bytes as u64));

    if let Some(ref hf_tok) = hf_tokenizer {
        group.bench_function("huggingface/batch100", |b| {
            b.iter(|| {
                for text in &batch {
                    black_box(hf_tok.encode(*text, false).unwrap());
                }
            });
        });
    }

    group.bench_function("budtiktok_scalar/batch100", |b| {
        b.iter(|| {
            for text in &batch {
                black_box(bud_tokenizer.encode_scalar(*text));
            }
        });
    });

    group.bench_function("budtiktok_simd/batch100", |b| {
        b.iter(|| {
            for text in &batch {
                black_box(bud_tokenizer.encode_simd(*text));
            }
        });
    });

    // Parallel batch
    group.bench_function("budtiktok/parallel_batch100", |b| {
        b.iter(|| {
            black_box(bud_tokenizer.encode_batch(&batch));
        });
    });

    group.finish();
}

// ============================================================================
// Large Scale Benchmark
// ============================================================================

fn bench_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale");
    group.sample_size(10); // Fewer samples for large benchmarks

    let vocab = create_test_vocab();
    let tokenizer = WordLevelTokenizer::new(vocab.clone(), WordLevelConfig::default());
    let tokenizer_cached = WordLevelTokenizer::with_cache(vocab, WordLevelConfig::default(), 100_000);

    // 1 MB of text
    let large_text = generate_large_text(1024);
    group.throughput(Throughput::Bytes(large_text.len() as u64));

    group.bench_function("scalar/1mb", |b| {
        b.iter(|| black_box(tokenizer.encode_scalar(&large_text)));
    });

    group.bench_function("simd/1mb", |b| {
        b.iter(|| black_box(tokenizer.encode_simd(&large_text)));
    });

    // Warm cache for cached benchmark
    tokenizer_cached.encode_scalar(&large_text);

    group.bench_function("cached/1mb", |b| {
        b.iter(|| black_box(tokenizer_cached.encode_scalar(&large_text)));
    });

    group.finish();
}

// ============================================================================
// Main
// ============================================================================

fn print_simd_info() {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║              WORDLEVEL TOKENIZER BENCHMARK                       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("SIMD Capabilities:");
    println!("  AVX2:    {}", if is_avx2_available() { "✓ Available" } else { "✗ Not available" });
    println!("  AVX-512: {}", if is_avx512_available() { "✓ Available" } else { "✗ Not available" });
    println!();
}

criterion_group!(
    benches,
    bench_pretokenize,
    bench_encode,
    bench_vs_huggingface,
    bench_large_scale,
);

criterion_main!(benches);
