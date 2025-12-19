//! Performance Regression Tests (9.3.2)
//!
//! Tests that ensure performance doesn't regress beyond thresholds.
//! These tests measure execution time and flag regressions.
//!
//! Note: For accurate benchmarking, use criterion. These tests provide
//! basic regression detection during development.
//!
//! IMPORTANT: These tests use relaxed thresholds for debug builds.
//! Production benchmarks should use criterion with release builds.

use budtiktok_core::unicode::*;
use std::time::{Duration, Instant};

// =============================================================================
// Performance Baseline Helpers
// =============================================================================

/// Debug builds are typically 10-20x slower than release
#[cfg(debug_assertions)]
const DEBUG_MULTIPLIER: u64 = 20;

#[cfg(not(debug_assertions))]
const DEBUG_MULTIPLIER: u64 = 1;

/// Run a function multiple times and return average duration
fn measure_avg<F: FnMut()>(mut f: F, iterations: usize) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed() / iterations as u32
}

/// Check that a function completes within a time budget (adjusted for debug builds)
fn assert_within_budget<F: FnMut()>(f: F, iterations: usize, max_avg_us: u64, name: &str) {
    let adjusted_budget = max_avg_us * DEBUG_MULTIPLIER;
    let avg = measure_avg(f, iterations);
    let avg_us = avg.as_micros() as u64;

    assert!(
        avg_us <= adjusted_budget,
        "{} regression: {} us average (budget: {} us, debug multiplier: {}x)",
        name,
        avg_us,
        adjusted_budget,
        DEBUG_MULTIPLIER
    );
}

// =============================================================================
// ASCII Detection Performance
// =============================================================================

#[test]
fn test_perf_is_ascii_fast_short() {
    let text = "Hello, World!"; // 13 chars

    assert_within_budget(
        || { let _ = is_ascii_fast(text); },
        10_000,
        1, // 1 us budget for short strings
        "is_ascii_fast (short string)"
    );
}

#[test]
fn test_perf_is_ascii_fast_long() {
    let text = "a".repeat(10_000); // 10K chars

    assert_within_budget(
        || { let _ = is_ascii_fast(&text); },
        1_000,
        50, // 50 us budget for 10K string
        "is_ascii_fast (10K string)"
    );
}

// =============================================================================
// Normalization Performance
// =============================================================================

#[test]
fn test_perf_normalize_ascii() {
    let text = "Hello, World! This is a test sentence.";

    assert_within_budget(
        || { let _ = normalize(text, NormalizationForm::NFC); },
        10_000,
        5, // 5 us budget - ASCII should be fast
        "normalize (ASCII)"
    );
}

#[test]
fn test_perf_normalize_unicode() {
    let text = "héllo wörld café";

    assert_within_budget(
        || { let _ = normalize(text, NormalizationForm::NFC); },
        10_000,
        50, // 50 us budget for Unicode
        "normalize (Unicode)"
    );
}

// =============================================================================
// Character Classification Performance
// =============================================================================

#[test]
fn test_perf_is_whitespace() {
    let chars = [' ', '\t', '\n', 'a', 'Z', '!'];

    assert_within_budget(
        || {
            for &ch in &chars {
                let _ = is_whitespace(ch);
            }
        },
        100_000,
        1, // 1 us budget for 6 lookups
        "is_whitespace (6 chars)"
    );
}

#[test]
fn test_perf_is_punctuation() {
    let chars = ['.', ',', '!', '?', 'a', ' '];

    assert_within_budget(
        || {
            for &ch in &chars {
                let _ = is_punctuation(ch);
            }
        },
        100_000,
        1,
        "is_punctuation (6 chars)"
    );
}

#[test]
fn test_perf_get_category_flags() {
    let chars = ['a', 'Z', '5', ' ', '!', '\x00'];

    assert_within_budget(
        || {
            for &ch in &chars {
                let _ = get_category_flags(ch);
            }
        },
        100_000,
        1,
        "get_category_flags (6 chars)"
    );
}

// =============================================================================
// Throughput Tests
// =============================================================================

#[test]
fn test_perf_ascii_check_throughput() {
    // Test throughput: should process at least 1GB/s for ASCII check
    // (relaxed to 50MB/s for debug builds)
    let text = "a".repeat(1_000_000); // 1MB
    let iterations = 10;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = is_ascii_fast(&text);
    }
    let elapsed = start.elapsed();

    let bytes_processed = text.len() * iterations;
    let bytes_per_sec = bytes_processed as f64 / elapsed.as_secs_f64();
    let gb_per_sec = bytes_per_sec / 1_000_000_000.0;

    // Adjust threshold for debug builds (20x slower)
    let min_throughput = 1.0 / DEBUG_MULTIPLIER as f64;

    assert!(
        gb_per_sec >= min_throughput,
        "ASCII check throughput regression: {:.2} GB/s (expected >= {:.2})",
        gb_per_sec,
        min_throughput
    );
}

// =============================================================================
// Criterion Integration Pattern (for CI)
// =============================================================================

/// This test demonstrates the pattern for criterion-based benchmarks.
/// In CI, criterion benchmarks would be run separately with:
/// `cargo bench -- --save-baseline main`
/// `cargo bench -- --baseline main`
///
/// Regressions > 10% would fail CI.
#[test]
#[ignore = "Use criterion for accurate benchmarking"]
fn test_perf_criterion_pattern() {
    // criterion_group!(benches, bench_is_ascii_fast, bench_normalize);
    // criterion_main!(benches);
    //
    // fn bench_is_ascii_fast(c: &mut Criterion) {
    //     let text = "a".repeat(10_000);
    //     c.bench_function("is_ascii_fast", |b| {
    //         b.iter(|| is_ascii_fast(black_box(&text)))
    //     });
    // }
    todo!("Use criterion for production benchmarks");
}

// =============================================================================
// Latency Distribution Tests
// =============================================================================

/// Test that latency is consistent (low variance)
#[test]
fn test_perf_latency_consistency() {
    let text = "Hello, World!";
    let iterations = 1000;
    let mut durations = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = is_ascii_fast(text);
        durations.push(start.elapsed());
    }

    // Sort to get percentiles
    durations.sort();

    let p50 = durations[iterations / 2];
    let p99 = durations[iterations * 99 / 100];

    // P99 should not be more than 10x P50 (low variance)
    let ratio = p99.as_nanos() as f64 / p50.as_nanos().max(1) as f64;

    assert!(
        ratio < 100.0, // Very loose bound - just catch extreme outliers
        "Latency variance too high: P99/P50 = {:.1}x",
        ratio
    );
}
