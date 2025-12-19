//! SIMD Validation Tests
//!
//! Tests that verify SIMD implementations (SSE4.2, AVX2) produce identical
//! results to scalar baselines, and benchmarks performance improvements.

use std::time::{Duration, Instant};

// =============================================================================
// Test Data Generation
// =============================================================================

fn generate_ascii_text(len: usize) -> String {
    (0..len)
        .map(|i| {
            let chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 \t\n";
            chars[i % chars.len()] as char
        })
        .collect()
}

fn generate_unicode_text(len: usize) -> String {
    let patterns = ["Hello", "世界", "émoji", "日本語", " ", "\t", "Ω", "αβγ"];
    let mut result = String::new();
    let mut i = 0;
    while result.len() < len {
        result.push_str(patterns[i % patterns.len()]);
        i += 1;
    }
    result.truncate(len);
    result
}

fn generate_whitespace_heavy(len: usize) -> String {
    (0..len)
        .map(|i| {
            if i % 5 == 0 { ' ' }
            else if i % 7 == 0 { '\t' }
            else if i % 11 == 0 { '\n' }
            else { ('a' as u8 + (i % 26) as u8) as char }
        })
        .collect()
}

// =============================================================================
// Scalar Baseline Implementations
// =============================================================================

fn scalar_is_all_ascii(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b < 128)
}

fn scalar_count_whitespace(bytes: &[u8]) -> usize {
    bytes.iter().filter(|&&b| b == b' ' || b == b'\t' || b == b'\n' || b == b'\r').count()
}

fn scalar_to_lowercase(bytes: &mut [u8]) {
    for b in bytes.iter_mut() {
        if *b >= b'A' && *b <= b'Z' {
            *b += 32;
        }
    }
}

fn scalar_find_non_ascii(bytes: &[u8]) -> Option<usize> {
    bytes.iter().position(|&b| b >= 128)
}

fn scalar_validate_utf8(bytes: &[u8]) -> bool {
    std::str::from_utf8(bytes).is_ok()
}

fn scalar_count_code_points(s: &str) -> usize {
    s.chars().count()
}

// =============================================================================
// SIMD Feature Detection
// =============================================================================

fn has_sse42() -> bool {
    #[cfg(target_arch = "x86_64")]
    { is_x86_feature_detected!("sse4.2") }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    { is_x86_feature_detected!("avx2") }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

// =============================================================================
// Correctness Tests - SSE4.2
// =============================================================================

#[test]
fn test_sse42_available() {
    let available = has_sse42();
    println!("SSE4.2 available: {}", available);
    // This test just reports availability, doesn't fail
}

#[test]
fn test_sse42_is_all_ascii_correctness() {
    use budtiktok_core::is_all_ascii;

    if !has_sse42() {
        println!("Skipping SSE4.2 test - not available");
        return;
    }

    // Test various sizes including SIMD boundaries - pure ASCII
    for size in [0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1000, 10000] {
        let ascii = generate_ascii_text(size);
        let ascii_bytes = ascii.as_bytes();

        let scalar_result = scalar_is_all_ascii(ascii_bytes);
        let simd_result = is_all_ascii(ascii_bytes);

        assert_eq!(
            scalar_result, simd_result,
            "ASCII check mismatch at size {}: scalar={}, simd={}",
            size, scalar_result, simd_result
        );
    }

    // Test non-ASCII detection at various positions
    for size in [16, 17, 18, 19, 20, 24, 31, 32, 33, 48, 64, 100, 256] {
        for pos in 0..size {
            let mut bytes: Vec<u8> = vec![b'a'; size];
            bytes[pos] = 200; // Non-ASCII byte

            let scalar_result = scalar_is_all_ascii(&bytes);
            let simd_result = is_all_ascii(&bytes);

            assert_eq!(
                scalar_result, simd_result,
                "Non-ASCII detection mismatch at size={}, pos={}: scalar={}, simd={}",
                size, pos, scalar_result, simd_result
            );
        }
    }

    println!("✓ SSE4.2 is_all_ascii correctness verified (including non-ASCII detection)");
}

#[test]
fn test_sse42_find_whitespace_correctness() {
    use budtiktok_core::find_first_whitespace;

    if !has_sse42() {
        println!("Skipping SSE4.2 test - not available");
        return;
    }

    for size in [16, 32, 64, 128, 256, 1000] {
        let text = generate_whitespace_heavy(size);
        let bytes = text.as_bytes();

        // Find first whitespace using scalar
        let scalar_pos = bytes.iter().position(|&b| b == b' ' || b == b'\t' || b == b'\n' || b == b'\r');
        let simd_pos = find_first_whitespace(bytes);

        assert_eq!(
            scalar_pos, simd_pos,
            "Whitespace position mismatch at size {}", size
        );
    }

    println!("✓ SSE4.2 find_whitespace correctness verified");
}

#[test]
fn test_sse42_count_whitespace_correctness() {
    use budtiktok_core::count_whitespace;

    if !has_sse42() {
        println!("Skipping SSE4.2 test - not available");
        return;
    }

    for size in [16, 32, 64, 128, 256, 1000] {
        let text = generate_whitespace_heavy(size);
        let bytes = text.as_bytes();

        let scalar_count = scalar_count_whitespace(bytes);
        let simd_count = count_whitespace(bytes);

        assert_eq!(
            scalar_count, simd_count,
            "Whitespace count mismatch at size {}: scalar={}, simd={}",
            size, scalar_count, simd_count
        );
    }

    println!("✓ SSE4.2 count_whitespace correctness verified");
}

// =============================================================================
// Correctness Tests - AVX2
// =============================================================================

#[test]
fn test_avx2_available() {
    let available = has_avx2();
    println!("AVX2 available: {}", available);
}

#[test]
fn test_avx2_utf8_validation_correctness() {
    use budtiktok_core::is_valid_utf8;

    if !has_avx2() {
        println!("Skipping AVX2 test - not available");
        return;
    }

    // Valid UTF-8 strings
    let valid_cases = [
        "",
        "Hello, World!",
        "日本語テスト",
        "émoji 🎉 αβγδ",
        &"a".repeat(1000),
        &generate_unicode_text(10000),
    ];

    for (i, case) in valid_cases.iter().enumerate() {
        let bytes = case.as_bytes();
        let scalar_valid = scalar_validate_utf8(bytes);
        let simd_valid = is_valid_utf8(bytes);

        assert_eq!(
            scalar_valid, simd_valid,
            "UTF-8 validation mismatch for valid case {}: scalar={}, simd={}",
            i, scalar_valid, simd_valid
        );
    }

    // Invalid UTF-8 sequences
    let invalid_cases: Vec<Vec<u8>> = vec![
        vec![0xFF],                           // Invalid byte
        vec![0xC0, 0x80],                     // Overlong encoding
        vec![0xED, 0xA0, 0x80],               // Surrogate half
        vec![0xF4, 0x90, 0x80, 0x80],         // Above U+10FFFF
        vec![b'a', b'b', 0x80, b'c'],         // Continuation without start
        vec![0xE0, 0x80],                     // Incomplete sequence
    ];

    for (i, case) in invalid_cases.iter().enumerate() {
        let scalar_valid = scalar_validate_utf8(case);
        let simd_valid = is_valid_utf8(case);

        assert_eq!(
            scalar_valid, simd_valid,
            "UTF-8 validation mismatch for invalid case {}: scalar={}, simd={}",
            i, scalar_valid, simd_valid
        );
    }

    println!("✓ AVX2 UTF-8 validation correctness verified");
}

#[test]
fn test_avx2_count_code_points_correctness() {
    use budtiktok_core::count_code_points;

    if !has_avx2() {
        println!("Skipping AVX2 test - not available");
        return;
    }

    let test_cases = [
        "",
        "Hello",
        "日本語",
        "Hello 世界 émoji 🎉",
        &"a".repeat(1000),
        &"日".repeat(500),
        &generate_unicode_text(5000),
    ];

    for (i, case) in test_cases.iter().enumerate() {
        let scalar_count = scalar_count_code_points(case);
        let simd_count = count_code_points(case.as_bytes());

        assert_eq!(
            scalar_count, simd_count,
            "Code point count mismatch for case {}: scalar={}, simd={}",
            i, scalar_count, simd_count
        );
    }

    println!("✓ AVX2 count_code_points correctness verified");
}

#[test]
fn test_avx2_find_non_ascii_correctness() {
    use budtiktok_core::find_first_non_ascii;

    if !has_avx2() {
        println!("Skipping AVX2 test - not available");
        return;
    }

    // All ASCII
    for size in [32, 64, 128, 256, 1000] {
        let ascii = generate_ascii_text(size);
        let bytes = ascii.as_bytes();

        let scalar_pos = scalar_find_non_ascii(bytes);
        let simd_pos = find_first_non_ascii(bytes);

        assert_eq!(
            scalar_pos, simd_pos,
            "Non-ASCII position mismatch for pure ASCII at size {}", size
        );
    }

    // With non-ASCII at various positions
    for size in [64, 128, 256] {
        for insert_pos in [0, 16, 31, 32, 33, 63, 64, size/2, size-1] {
            if insert_pos >= size { continue; }

            let mut bytes: Vec<u8> = (0..size).map(|i| b'a' + (i % 26) as u8).collect();
            bytes[insert_pos] = 0xC0; // Non-ASCII start byte
            if insert_pos + 1 < size {
                bytes[insert_pos + 1] = 0x80; // Continuation byte
            }

            let scalar_pos = scalar_find_non_ascii(&bytes);
            let simd_pos = find_first_non_ascii(&bytes);

            assert_eq!(
                scalar_pos, simd_pos,
                "Non-ASCII position mismatch at size {}, insert_pos {}: scalar={:?}, simd={:?}",
                size, insert_pos, scalar_pos, simd_pos
            );
        }
    }

    println!("✓ AVX2 find_non_ascii correctness verified");
}

// =============================================================================
// SWAR Correctness Tests
// =============================================================================

#[test]
fn test_swar_has_zero_byte_correctness() {
    use budtiktok_core::has_zero_byte;

    // No zeros
    let no_zeros: u64 = 0x0101010101010101;
    assert!(!has_zero_byte(no_zeros));

    // All zeros
    assert!(has_zero_byte(0u64));

    // Zero in each position
    for pos in 0..8 {
        let mut word = 0x0101010101010101u64;
        word &= !(0xFFu64 << (pos * 8)); // Clear byte at position
        assert!(has_zero_byte(word), "Failed to detect zero at position {}", pos);
    }

    // Random values with zeros
    for _ in 0..1000 {
        let bytes: [u8; 8] = rand_bytes();
        let word = u64::from_ne_bytes(bytes);

        let scalar_has_zero = bytes.iter().any(|&b| b == 0);
        let swar_has_zero = has_zero_byte(word);

        assert_eq!(scalar_has_zero, swar_has_zero);
    }

    println!("✓ SWAR has_zero_byte correctness verified");
}

#[test]
fn test_swar_has_byte_correctness() {
    use budtiktok_core::has_byte;

    for target in [0u8, 32, 65, 127, 255] {
        for _ in 0..100 {
            let bytes: [u8; 8] = rand_bytes();
            let word = u64::from_ne_bytes(bytes);

            let scalar_has = bytes.iter().any(|&b| b == target);
            let swar_has = has_byte(word, target);

            assert_eq!(
                scalar_has, swar_has,
                "has_byte mismatch for target {}: bytes={:?}", target, bytes
            );
        }
    }

    println!("✓ SWAR has_byte correctness verified");
}

#[test]
fn test_swar_is_all_ascii_unrolled_correctness() {
    use budtiktok_core::is_all_ascii_unrolled;

    for size in [0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 100, 256, 1000] {
        // Pure ASCII
        let ascii: Vec<u8> = (0..size).map(|i| (i % 128) as u8).collect();
        assert_eq!(
            scalar_is_all_ascii(&ascii),
            is_all_ascii_unrolled(&ascii),
            "ASCII check mismatch for pure ASCII at size {}", size
        );

        // With non-ASCII
        if size > 0 {
            let mut non_ascii = ascii.clone();
            non_ascii[size / 2] = 200;
            assert_eq!(
                scalar_is_all_ascii(&non_ascii),
                is_all_ascii_unrolled(&non_ascii),
                "ASCII check mismatch for non-ASCII at size {}", size
            );
        }
    }

    println!("✓ SWAR is_all_ascii_unrolled correctness verified");
}

// Helper for random bytes
fn rand_bytes() -> [u8; 8] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    hasher.finish().to_ne_bytes()
}

// =============================================================================
// Performance Benchmarks
// =============================================================================

struct BenchResult {
    name: String,
    scalar_ns: u64,
    simd_ns: u64,
    speedup: f64,
    size: usize,
}

impl BenchResult {
    fn print(&self) {
        println!(
            "  {:40} | {:>10} | {:>10} | {:>8} | {:>6.2}x",
            self.name,
            format!("{} ns", self.scalar_ns),
            format!("{} ns", self.simd_ns),
            format!("{} bytes", self.size),
            self.speedup
        );
    }
}

fn bench<F1, F2, R>(name: &str, size: usize, iterations: usize, scalar_fn: F1, simd_fn: F2) -> BenchResult
where
    F1: Fn() -> R,
    F2: Fn() -> R,
{
    // Warmup
    for _ in 0..100 {
        let _ = scalar_fn();
        let _ = simd_fn();
    }

    // Benchmark scalar
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(scalar_fn());
    }
    let scalar_duration = start.elapsed();

    // Benchmark SIMD
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(simd_fn());
    }
    let simd_duration = start.elapsed();

    let scalar_ns = scalar_duration.as_nanos() as u64 / iterations as u64;
    let simd_ns = simd_duration.as_nanos() as u64 / iterations as u64;
    let speedup = scalar_ns as f64 / simd_ns.max(1) as f64;

    BenchResult {
        name: name.to_string(),
        scalar_ns,
        simd_ns,
        speedup,
        size,
    }
}

#[test]
fn bench_ascii_check_performance() {
    use budtiktok_core::is_all_ascii;

    println!("\n{}", "=".repeat(80));
    println!("ASCII Check Performance: Scalar vs SIMD");
    println!("{}", "=".repeat(80));
    println!("  {:40} | {:>10} | {:>10} | {:>8} | Speedup", "Operation", "Scalar", "SIMD", "Size");
    println!("  {:-<40}-+-{:-<10}-+-{:-<10}-+-{:-<8}-+--------", "", "", "", "");

    for &size in &[64, 256, 1024, 4096, 16384, 65536] {
        let data = generate_ascii_text(size);
        let bytes = data.as_bytes();

        let result = bench(
            &format!("is_all_ascii ({})", size),
            size,
            10000,
            || scalar_is_all_ascii(bytes),
            || is_all_ascii(bytes),
        );
        result.print();
    }

    println!();
}

#[test]
fn bench_utf8_validation_performance() {
    use budtiktok_core::is_valid_utf8;

    println!("\n{}", "=".repeat(80));
    println!("UTF-8 Validation Performance: Scalar vs SIMD (AVX2)");
    println!("{}", "=".repeat(80));
    println!("  {:40} | {:>10} | {:>10} | {:>8} | Speedup", "Operation", "Scalar", "SIMD", "Size");
    println!("  {:-<40}-+-{:-<10}-+-{:-<10}-+-{:-<8}-+--------", "", "", "", "");

    if !has_avx2() {
        println!("  AVX2 not available - skipping");
        return;
    }

    // ASCII text
    for &size in &[256, 1024, 4096, 16384, 65536] {
        let data = generate_ascii_text(size);
        let bytes = data.as_bytes();

        let result = bench(
            &format!("validate_utf8 ASCII ({})", size),
            size,
            5000,
            || scalar_validate_utf8(bytes),
            || is_valid_utf8(bytes),
        );
        result.print();
    }

    // Unicode text
    for &size in &[256, 1024, 4096, 16384] {
        let data = generate_unicode_text(size);
        let bytes = data.as_bytes();

        let result = bench(
            &format!("validate_utf8 Unicode ({})", size),
            bytes.len(),
            5000,
            || scalar_validate_utf8(bytes),
            || is_valid_utf8(bytes),
        );
        result.print();
    }

    println!();
}

#[test]
fn bench_code_point_count_performance() {
    use budtiktok_core::count_code_points;

    println!("\n{}", "=".repeat(80));
    println!("Code Point Count Performance: Scalar vs SIMD (AVX2)");
    println!("{}", "=".repeat(80));
    println!("  {:40} | {:>10} | {:>10} | {:>8} | Speedup", "Operation", "Scalar", "SIMD", "Size");
    println!("  {:-<40}-+-{:-<10}-+-{:-<10}-+-{:-<8}-+--------", "", "", "", "");

    if !has_avx2() {
        println!("  AVX2 not available - skipping");
        return;
    }

    // ASCII
    for &size in &[256, 1024, 4096, 16384] {
        let data = generate_ascii_text(size);
        let bytes = data.as_bytes();

        let result = bench(
            &format!("count_code_points ASCII ({})", size),
            size,
            5000,
            || scalar_count_code_points(&data),
            || count_code_points(bytes),
        );
        result.print();
    }

    // Unicode
    for &size in &[256, 1024, 4096] {
        let data = generate_unicode_text(size);
        let bytes = data.as_bytes();

        let result = bench(
            &format!("count_code_points Unicode ({})", data.len()),
            data.len(),
            5000,
            || scalar_count_code_points(&data),
            || count_code_points(bytes),
        );
        result.print();
    }

    println!();
}

#[test]
fn bench_whitespace_count_performance() {
    use budtiktok_core::count_whitespace;

    println!("\n{}", "=".repeat(80));
    println!("Whitespace Count Performance: Scalar vs SIMD (SSE4.2)");
    println!("{}", "=".repeat(80));
    println!("  {:40} | {:>10} | {:>10} | {:>8} | Speedup", "Operation", "Scalar", "SIMD", "Size");
    println!("  {:-<40}-+-{:-<10}-+-{:-<10}-+-{:-<8}-+--------", "", "", "", "");

    if !has_sse42() {
        println!("  SSE4.2 not available - skipping");
        return;
    }

    for &size in &[256, 1024, 4096, 16384, 65536] {
        let data = generate_whitespace_heavy(size);
        let bytes = data.as_bytes();

        let result = bench(
            &format!("count_whitespace ({})", size),
            size,
            10000,
            || scalar_count_whitespace(bytes),
            || count_whitespace(bytes),
        );
        result.print();
    }

    println!();
}

#[test]
fn bench_swar_performance() {
    use budtiktok_core::{has_zero_byte, broadcast_byte, is_all_ascii_unrolled};

    println!("\n{}", "=".repeat(80));
    println!("SWAR Performance (64-bit register tricks)");
    println!("{}", "=".repeat(80));
    println!("  {:40} | {:>10} | {:>10} | {:>8} | Speedup", "Operation", "Scalar", "SWAR", "Size");
    println!("  {:-<40}-+-{:-<10}-+-{:-<10}-+-{:-<8}-+--------", "", "", "", "");

    // has_zero_byte
    let words: Vec<u64> = (0..1000).map(|i| 0x0101010101010101u64 + i).collect();
    let result = bench(
        "has_zero_byte (1000 words)",
        8000,
        1000,
        || words.iter().map(|&w| {
            let bytes = w.to_ne_bytes();
            bytes.iter().any(|&b| b == 0)
        }).filter(|&x| x).count(),
        || words.iter().map(|&w| has_zero_byte(w)).filter(|&x| x).count(),
    );
    result.print();

    // is_all_ascii_unrolled
    for &size in &[64, 256, 1024, 4096, 16384] {
        let data: Vec<u8> = (0..size).map(|i| (i % 128) as u8).collect();

        let result = bench(
            &format!("is_all_ascii_unrolled ({})", size),
            size,
            10000,
            || scalar_is_all_ascii(&data),
            || is_all_ascii_unrolled(&data),
        );
        result.print();
    }

    println!();
}

// =============================================================================
// Summary Test
// =============================================================================

#[test]
fn test_simd_summary() {
    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║                         SIMD VALIDATION SUMMARY                            ║");
    println!("╠════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Feature Detection:                                                         ║");
    println!("║   SSE4.2:    {:5}                                                         ║", if has_sse42() { "✓" } else { "✗" });
    println!("║   AVX2:      {:5}                                                         ║", if has_avx2() { "✓" } else { "✗" });

    #[cfg(target_arch = "x86_64")]
    {
        let avx512 = is_x86_feature_detected!("avx512f");
        println!("║   AVX-512F:  {:5}                                                         ║", if avx512 { "✓" } else { "✗" });
    }

    println!("╠════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Run individual tests for detailed correctness and performance results.     ║");
    println!("║                                                                            ║");
    println!("║ Commands:                                                                  ║");
    println!("║   cargo test --test simd_validation -- --nocapture                         ║");
    println!("║   cargo test --test simd_validation bench_ -- --nocapture                  ║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝");
    println!();
}
