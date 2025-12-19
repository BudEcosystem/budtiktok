//! Scalar tokenization benchmarks
//!
//! Validates that scalar fallback implementations are competitive with or better
//! than BlazeText's optimizations (6.6x-10x over HuggingFace baseline).
//!
//! Key optimizations validated:
//! - ASCII fast path with O(1) lookup
//! - Thread-local Unicode category cache
//! - SWAR (SIMD Within A Register) for 8-byte parallel processing
//! - Branchless character classification
//! - Loop unrolling for whitespace detection

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use budtiktok_core::swar::{
    find_first_whitespace_unrolled, is_all_ascii_unrolled,
    is_ascii_alphanumeric_branchless, is_ascii_punctuation_branchless,
    is_ascii_whitespace_branchless, to_lowercase_ascii_unrolled,
};
use budtiktok_core::unicode::{
    get_category_flags, is_ascii_fast, is_cjk_character, is_punctuation, is_whitespace,
    normalize_with_fast_path, NormalizationForm,
};

// =============================================================================
// Test Data
// =============================================================================

fn ascii_text() -> &'static str {
    "The quick brown fox jumps over the lazy dog. This is a test sentence with various punctuation marks: commas, periods, and exclamation points!"
}

fn unicode_text() -> &'static str {
    "日本語テスト。これは日本語のテキストです。The quick brown fox. 中文测试。"
}

fn mixed_text() -> &'static str {
    "Hello world! 你好世界! Привет мир! مرحبا بالعالم! This is mixed text with ASCII, CJK, Cyrillic, and Arabic."
}

fn long_ascii_text() -> String {
    ascii_text().repeat(100)
}

fn long_unicode_text() -> String {
    unicode_text().repeat(100)
}

// =============================================================================
// ASCII Fast Path Benchmarks
// =============================================================================

fn bench_is_ascii_fast(c: &mut Criterion) {
    let mut group = c.benchmark_group("ascii_fast_path");

    let ascii = ascii_text();
    let unicode = unicode_text();
    let long_ascii = long_ascii_text();

    group.throughput(Throughput::Bytes(ascii.len() as u64));
    group.bench_function("is_ascii_fast_short_ascii", |b| {
        b.iter(|| is_ascii_fast(black_box(ascii)))
    });

    group.throughput(Throughput::Bytes(unicode.len() as u64));
    group.bench_function("is_ascii_fast_short_unicode", |b| {
        b.iter(|| is_ascii_fast(black_box(unicode)))
    });

    group.throughput(Throughput::Bytes(long_ascii.len() as u64));
    group.bench_function("is_ascii_fast_long_ascii", |b| {
        b.iter(|| is_ascii_fast(black_box(&long_ascii)))
    });

    group.finish();
}

// =============================================================================
// SWAR Benchmarks (SIMD Within A Register)
// =============================================================================

fn bench_swar_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("swar_operations");

    let ascii = ascii_text().as_bytes();
    let long_ascii = long_ascii_text();
    let long_bytes = long_ascii.as_bytes();

    // Whitespace detection with unrolling
    group.throughput(Throughput::Bytes(ascii.len() as u64));
    group.bench_function("find_whitespace_unrolled_short", |b| {
        b.iter(|| find_first_whitespace_unrolled(black_box(ascii)))
    });

    group.throughput(Throughput::Bytes(long_bytes.len() as u64));
    group.bench_function("find_whitespace_unrolled_long", |b| {
        b.iter(|| find_first_whitespace_unrolled(black_box(long_bytes)))
    });

    // ASCII check with unrolling
    group.throughput(Throughput::Bytes(long_bytes.len() as u64));
    group.bench_function("is_all_ascii_unrolled", |b| {
        b.iter(|| is_all_ascii_unrolled(black_box(long_bytes)))
    });

    // Lowercase conversion
    let mut buffer = long_bytes.to_vec();
    group.throughput(Throughput::Bytes(buffer.len() as u64));
    group.bench_function("to_lowercase_unrolled", |b| {
        b.iter(|| {
            to_lowercase_ascii_unrolled(black_box(&mut buffer));
        })
    });

    group.finish();
}

// =============================================================================
// Branchless Classification Benchmarks
// =============================================================================

fn bench_branchless_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("branchless_classification");

    // Test all printable ASCII characters
    let test_bytes: Vec<u8> = (0u8..128).collect();

    group.throughput(Throughput::Elements(test_bytes.len() as u64));

    group.bench_function("is_whitespace_branchless", |b| {
        b.iter(|| {
            test_bytes.iter().map(|&byte| {
                is_ascii_whitespace_branchless(black_box(byte))
            }).sum::<u8>()
        })
    });

    group.bench_function("is_alphanumeric_branchless", |b| {
        b.iter(|| {
            test_bytes.iter().map(|&byte| {
                is_ascii_alphanumeric_branchless(black_box(byte))
            }).sum::<u8>()
        })
    });

    group.bench_function("is_punctuation_branchless", |b| {
        b.iter(|| {
            test_bytes.iter().map(|&byte| {
                is_ascii_punctuation_branchless(black_box(byte))
            }).sum::<u8>()
        })
    });

    // Compare with standard library (branching)
    group.bench_function("std_is_whitespace", |b| {
        b.iter(|| {
            test_bytes.iter().map(|&byte| {
                (byte as char).is_whitespace() as u8
            }).sum::<u8>()
        })
    });

    group.finish();
}

// =============================================================================
// Unicode Category Cache Benchmarks
// =============================================================================

fn bench_unicode_category(c: &mut Criterion) {
    let mut group = c.benchmark_group("unicode_category");

    // ASCII characters - should hit fast path
    let ascii_chars: Vec<char> = "Hello, World! 123".chars().collect();

    // Non-ASCII characters - exercises cache
    let unicode_chars: Vec<char> = "日本語中文한국어".chars().collect();

    // Mixed - exercises both paths
    let mixed_chars: Vec<char> = "Hello 你好 World 世界".chars().collect();

    group.throughput(Throughput::Elements(ascii_chars.len() as u64));
    group.bench_function("category_flags_ascii", |b| {
        b.iter(|| {
            ascii_chars.iter().map(|&c| get_category_flags(black_box(c))).count()
        })
    });

    group.throughput(Throughput::Elements(unicode_chars.len() as u64));
    group.bench_function("category_flags_unicode", |b| {
        b.iter(|| {
            unicode_chars.iter().map(|&c| get_category_flags(black_box(c))).count()
        })
    });

    group.throughput(Throughput::Elements(mixed_chars.len() as u64));
    group.bench_function("category_flags_mixed", |b| {
        b.iter(|| {
            mixed_chars.iter().map(|&c| get_category_flags(black_box(c))).count()
        })
    });

    // Repeated access to same character (cache hit test)
    group.bench_function("category_flags_cached", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let _ = get_category_flags(black_box('中'));
            }
        })
    });

    group.finish();
}

// =============================================================================
// Character Classification Benchmarks
// =============================================================================

fn bench_char_classification(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_classification");

    let mixed = mixed_text();
    let chars: Vec<char> = mixed.chars().collect();

    group.throughput(Throughput::Elements(chars.len() as u64));

    group.bench_function("is_whitespace", |b| {
        b.iter(|| {
            chars.iter().filter(|&&c| is_whitespace(black_box(c))).count()
        })
    });

    group.bench_function("is_punctuation", |b| {
        b.iter(|| {
            chars.iter().filter(|&&c| is_punctuation(black_box(c))).count()
        })
    });

    group.bench_function("is_cjk_character", |b| {
        b.iter(|| {
            chars.iter().filter(|&&c| is_cjk_character(black_box(c))).count()
        })
    });

    group.finish();
}

// =============================================================================
// Normalization Fast Path Benchmarks
// =============================================================================

fn bench_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization");

    let ascii = ascii_text();
    let unicode = unicode_text();
    let long_ascii = long_ascii_text();

    // ASCII fast path - should return Cow::Borrowed (zero-copy)
    group.throughput(Throughput::Bytes(ascii.len() as u64));
    group.bench_function("normalize_ascii_nfc", |b| {
        b.iter(|| normalize_with_fast_path(black_box(ascii), NormalizationForm::NFC))
    });

    // Unicode requires actual normalization
    group.throughput(Throughput::Bytes(unicode.len() as u64));
    group.bench_function("normalize_unicode_nfc", |b| {
        b.iter(|| normalize_with_fast_path(black_box(unicode), NormalizationForm::NFC))
    });

    // Long ASCII - validates scaling of fast path
    group.throughput(Throughput::Bytes(long_ascii.len() as u64));
    group.bench_function("normalize_long_ascii_nfc", |b| {
        b.iter(|| normalize_with_fast_path(black_box(&long_ascii), NormalizationForm::NFC))
    });

    group.finish();
}

// =============================================================================
// End-to-End Pre-tokenization Benchmark
// =============================================================================

fn bench_pretokenization(c: &mut Criterion) {
    let mut group = c.benchmark_group("pretokenization");

    let text = mixed_text();
    let long_text = text.repeat(50);

    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("split_words_short", |b| {
        b.iter(|| {
            let words: Vec<&str> = black_box(text).split_whitespace().collect();
            words.len()
        })
    });

    group.throughput(Throughput::Bytes(long_text.len() as u64));
    group.bench_function("split_words_long", |b| {
        b.iter(|| {
            let words: Vec<&str> = black_box(&long_text).split_whitespace().collect();
            words.len()
        })
    });

    // Custom word split with character classification
    group.bench_function("custom_word_split", |b| {
        b.iter(|| {
            let mut words = Vec::new();
            let mut word_start = None;

            for (i, c) in black_box(text).char_indices() {
                if is_whitespace(c) || is_punctuation(c) {
                    if let Some(start) = word_start {
                        words.push(&text[start..i]);
                        word_start = None;
                    }
                    if is_punctuation(c) {
                        // Punctuation is its own token
                    }
                } else if word_start.is_none() {
                    word_start = Some(i);
                }
            }

            if let Some(start) = word_start {
                words.push(&text[start..]);
            }

            words.len()
        })
    });

    group.finish();
}

// =============================================================================
// Criterion Groups
// =============================================================================

criterion_group!(
    benches,
    bench_is_ascii_fast,
    bench_swar_operations,
    bench_branchless_classification,
    bench_unicode_category,
    bench_char_classification,
    bench_normalization,
    bench_pretokenization,
);

criterion_main!(benches);
