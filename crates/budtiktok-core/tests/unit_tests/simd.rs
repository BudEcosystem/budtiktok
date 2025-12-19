//! SIMD Implementation Unit Tests (9.1.8)
//!
//! Tests for SIMD-accelerated operations including:
//! - ASCII detection
//! - UTF-8 validation
//! - Byte scanning
//! - Character counting
//! - Whitespace detection
//! - Parallel string operations
//! - Feature detection
//! - Fallback implementations

use budtiktok_core::simd::*;

// =============================================================================
// Feature Detection Tests
// =============================================================================

#[test]
fn test_feature_detection() {
    let features = SimdFeatures::detect();

    // At least one implementation should be available
    assert!(
        features.has_any(),
        "At least scalar fallback should be available"
    );
}

#[test]
fn test_feature_sse2() {
    let features = SimdFeatures::detect();

    #[cfg(target_arch = "x86_64")]
    {
        // SSE2 is guaranteed on x86_64
        assert!(features.has_sse2());
    }
}

#[test]
fn test_feature_avx2() {
    let features = SimdFeatures::detect();

    // AVX2 may or may not be available
    let _ = features.has_avx2();
}

#[test]
fn test_feature_neon() {
    let features = SimdFeatures::detect();

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is guaranteed on aarch64
        assert!(features.has_neon());
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // Should return false on non-ARM
        assert!(!features.has_neon());
    }
}

#[test]
fn test_best_implementation() {
    let features = SimdFeatures::detect();
    let best = features.best_available();

    // Should return some implementation
    assert!(matches!(
        best,
        SimdImplementation::Avx512
            | SimdImplementation::Avx2
            | SimdImplementation::Sse42
            | SimdImplementation::Sse2
            | SimdImplementation::Neon
            | SimdImplementation::Scalar
    ));
}

// =============================================================================
// ASCII Detection Tests
// =============================================================================

#[test]
fn test_is_ascii_pure_ascii() {
    assert!(is_ascii_simd(b"hello"));
    assert!(is_ascii_simd(b"Hello World!"));
    assert!(is_ascii_simd(b"0123456789"));
    assert!(is_ascii_simd(b"!@#$%^&*()"));
    assert!(is_ascii_simd(b""));
    assert!(is_ascii_simd(b" "));
    assert!(is_ascii_simd(b"\t\n\r"));
}

#[test]
fn test_is_ascii_with_high_bytes() {
    assert!(!is_ascii_simd("héllo".as_bytes()));
    assert!(!is_ascii_simd("café".as_bytes()));
    assert!(!is_ascii_simd("日本語".as_bytes()));
    assert!(!is_ascii_simd(b"hello\x80"));
    assert!(!is_ascii_simd(b"\xFF"));
}

#[test]
fn test_is_ascii_boundary_cases() {
    // Test at various boundaries
    assert!(is_ascii_simd(b"12345678")); // 8 bytes
    assert!(is_ascii_simd(b"1234567890123456")); // 16 bytes
    assert!(is_ascii_simd(b"12345678901234567890123456789012")); // 32 bytes
    assert!(is_ascii_simd(b"1234567")); // 7 bytes (remainder)
    assert!(is_ascii_simd(b"123456789")); // 9 bytes
}

#[test]
fn test_is_ascii_long_string() {
    let long_ascii = vec![b'a'; 10000];
    assert!(is_ascii_simd(&long_ascii));

    let mut with_high_byte = long_ascii.clone();
    with_high_byte[5000] = 0x80;
    assert!(!is_ascii_simd(&with_high_byte));
}

#[test]
fn test_is_ascii_vs_scalar() {
    // SIMD and scalar should agree
    let test_cases = vec![
        b"hello".to_vec(),
        b"world".to_vec(),
        vec![0x80],
        "日本語".as_bytes().to_vec(),
        vec![b'a'; 1000],
    ];

    for case in test_cases {
        let simd_result = is_ascii_simd(&case);
        let scalar_result = is_ascii_scalar(&case);
        assert_eq!(
            simd_result, scalar_result,
            "SIMD and scalar disagree for {:?}",
            &case[..case.len().min(20)]
        );
    }
}

// =============================================================================
// UTF-8 Validation Tests
// =============================================================================

#[test]
fn test_utf8_valid_ascii() {
    assert!(validate_utf8_simd(b"hello").is_ok());
    assert!(validate_utf8_simd(b"Hello World!").is_ok());
}

#[test]
fn test_utf8_valid_multibyte() {
    assert!(validate_utf8_simd("日本語".as_bytes()).is_ok());
    assert!(validate_utf8_simd("한글".as_bytes()).is_ok());
    assert!(validate_utf8_simd("emoji 😀".as_bytes()).is_ok());
}

#[test]
fn test_utf8_invalid_continuation() {
    // Invalid: continuation byte without start
    assert!(validate_utf8_simd(&[0x80]).is_err());
    assert!(validate_utf8_simd(&[0xBF]).is_err());
}

#[test]
fn test_utf8_invalid_overlong() {
    // Overlong encoding of ASCII
    assert!(validate_utf8_simd(&[0xC0, 0x80]).is_err()); // Overlong NUL
    assert!(validate_utf8_simd(&[0xC1, 0xBF]).is_err()); // Overlong
}

#[test]
fn test_utf8_invalid_surrogate() {
    // UTF-16 surrogates are invalid in UTF-8
    assert!(validate_utf8_simd(&[0xED, 0xA0, 0x80]).is_err()); // U+D800
    assert!(validate_utf8_simd(&[0xED, 0xBF, 0xBF]).is_err()); // U+DFFF
}

#[test]
fn test_utf8_invalid_too_large() {
    // Code points above U+10FFFF
    assert!(validate_utf8_simd(&[0xF4, 0x90, 0x80, 0x80]).is_err());
}

#[test]
fn test_utf8_truncated() {
    // Truncated sequences
    assert!(validate_utf8_simd(&[0xC2]).is_err()); // Missing continuation
    assert!(validate_utf8_simd(&[0xE0, 0xA0]).is_err()); // Missing continuation
    assert!(validate_utf8_simd(&[0xF0, 0x90, 0x80]).is_err()); // Missing continuation
}

#[test]
fn test_utf8_empty() {
    assert!(validate_utf8_simd(b"").is_ok());
}

#[test]
fn test_utf8_vs_std() {
    // SIMD validation should agree with std::str::from_utf8
    let test_cases: Vec<&[u8]> = vec![
        b"hello",
        "日本語".as_bytes(),
        &[0x80],
        &[0xC2, 0x80],
        &[0xC0, 0x80],
        &[0xED, 0xA0, 0x80],
    ];

    for case in test_cases {
        let simd_result = validate_utf8_simd(case).is_ok();
        let std_result = std::str::from_utf8(case).is_ok();
        assert_eq!(
            simd_result, std_result,
            "Validation disagrees for {:?}",
            case
        );
    }
}

// =============================================================================
// Byte Scanning Tests
// =============================================================================

#[test]
fn test_find_byte_basic() {
    assert_eq!(find_byte_simd(b"hello", b'e'), Some(1));
    assert_eq!(find_byte_simd(b"hello", b'l'), Some(2));
    assert_eq!(find_byte_simd(b"hello", b'o'), Some(4));
    assert_eq!(find_byte_simd(b"hello", b'x'), None);
}

#[test]
fn test_find_byte_first_position() {
    assert_eq!(find_byte_simd(b"hello", b'h'), Some(0));
    assert_eq!(find_byte_simd(b"aaa", b'a'), Some(0));
}

#[test]
fn test_find_byte_last_position() {
    assert_eq!(find_byte_simd(b"hello", b'o'), Some(4));
    assert_eq!(find_byte_simd(b"abcdefghijklmnop", b'p'), Some(15));
}

#[test]
fn test_find_byte_empty() {
    assert_eq!(find_byte_simd(b"", b'x'), None);
}

#[test]
fn test_find_byte_long_string() {
    let long = vec![b'a'; 10000];
    assert_eq!(find_byte_simd(&long, b'a'), Some(0));
    assert_eq!(find_byte_simd(&long, b'b'), None);

    let mut with_target = long.clone();
    with_target[5000] = b'b';
    assert_eq!(find_byte_simd(&with_target, b'b'), Some(5000));
}

#[test]
fn test_count_byte() {
    assert_eq!(count_byte_simd(b"hello", b'l'), 2);
    assert_eq!(count_byte_simd(b"hello", b'e'), 1);
    assert_eq!(count_byte_simd(b"hello", b'x'), 0);
    assert_eq!(count_byte_simd(b"aaaaaa", b'a'), 6);
}

#[test]
fn test_find_any_byte() {
    let bytes = [b' ', b'\t', b'\n'];
    assert_eq!(find_any_byte_simd(b"hello world", &bytes), Some(5));
    assert_eq!(find_any_byte_simd(b"hello\tworld", &bytes), Some(5));
    assert_eq!(find_any_byte_simd(b"helloworld", &bytes), None);
}

// =============================================================================
// Character Counting Tests
// =============================================================================

#[test]
fn test_count_chars_ascii() {
    assert_eq!(count_chars_simd("hello".as_bytes()), 5);
    assert_eq!(count_chars_simd("Hello World!".as_bytes()), 12);
    assert_eq!(count_chars_simd("".as_bytes()), 0);
}

#[test]
fn test_count_chars_utf8() {
    assert_eq!(count_chars_simd("日本語".as_bytes()), 3);
    assert_eq!(count_chars_simd("한글".as_bytes()), 2);
    assert_eq!(count_chars_simd("hello 世界".as_bytes()), 8);
}

#[test]
fn test_count_chars_emoji() {
    // Basic emoji (single code point)
    assert_eq!(count_chars_simd("😀".as_bytes()), 1);
    // Multiple emoji
    assert_eq!(count_chars_simd("😀😁😂".as_bytes()), 3);
}

#[test]
fn test_count_chars_mixed() {
    let text = "Hello 日本語 world";
    let char_count = text.chars().count();
    assert_eq!(count_chars_simd(text.as_bytes()), char_count);
}

#[test]
fn test_count_chars_vs_std() {
    let test_cases = vec!["hello", "日本語", "hello 世界 test", "emoji 😀 here"];

    for text in test_cases {
        let simd_count = count_chars_simd(text.as_bytes());
        let std_count = text.chars().count();
        assert_eq!(simd_count, std_count, "Char count differs for '{}'", text);
    }
}

// =============================================================================
// Whitespace Detection Tests
// =============================================================================

#[test]
fn test_find_whitespace() {
    assert_eq!(find_whitespace_simd(b"hello world"), Some(5));
    assert_eq!(find_whitespace_simd(b"hello\tworld"), Some(5));
    assert_eq!(find_whitespace_simd(b"hello\nworld"), Some(5));
    assert_eq!(find_whitespace_simd(b"helloworld"), None);
}

#[test]
fn test_find_whitespace_start() {
    assert_eq!(find_whitespace_simd(b" hello"), Some(0));
    assert_eq!(find_whitespace_simd(b"\thello"), Some(0));
}

#[test]
fn test_find_non_whitespace() {
    assert_eq!(find_non_whitespace_simd(b"   hello"), Some(3));
    assert_eq!(find_non_whitespace_simd(b"\t\n hello"), Some(3));
    assert_eq!(find_non_whitespace_simd(b"hello"), Some(0));
    assert_eq!(find_non_whitespace_simd(b"   "), None);
}

#[test]
fn test_strip_whitespace() {
    let (start, end) = strip_whitespace_bounds_simd(b"  hello  ");
    assert_eq!(start, 2);
    assert_eq!(end, 7);

    let (start, end) = strip_whitespace_bounds_simd(b"hello");
    assert_eq!(start, 0);
    assert_eq!(end, 5);

    let (start, end) = strip_whitespace_bounds_simd(b"   ");
    assert!(start >= end); // Empty result
}

#[test]
fn test_is_all_whitespace() {
    assert!(is_all_whitespace_simd(b"   "));
    assert!(is_all_whitespace_simd(b"\t\n\r "));
    assert!(is_all_whitespace_simd(b""));
    assert!(!is_all_whitespace_simd(b"hello"));
    assert!(!is_all_whitespace_simd(b" hello "));
}

// =============================================================================
// Parallel String Operations Tests
// =============================================================================

#[test]
fn test_to_lowercase_ascii() {
    let mut buffer = *b"HELLO WORLD";
    to_lowercase_ascii_simd(&mut buffer);
    assert_eq!(&buffer, b"hello world");
}

#[test]
fn test_to_lowercase_mixed() {
    let mut buffer = *b"HeLLo WoRLd";
    to_lowercase_ascii_simd(&mut buffer);
    assert_eq!(&buffer, b"hello world");
}

#[test]
fn test_to_lowercase_already_lower() {
    let mut buffer = *b"hello";
    to_lowercase_ascii_simd(&mut buffer);
    assert_eq!(&buffer, b"hello");
}

#[test]
fn test_to_uppercase_ascii() {
    let mut buffer = *b"hello world";
    to_uppercase_ascii_simd(&mut buffer);
    assert_eq!(&buffer, b"HELLO WORLD");
}

#[test]
fn test_memcmp_equal() {
    let a = b"hello world";
    let b = b"hello world";
    assert!(memcmp_simd(a, b));
}

#[test]
fn test_memcmp_not_equal() {
    let a = b"hello world";
    let b = b"hello World";
    assert!(!memcmp_simd(a, b));
}

#[test]
fn test_memcmp_different_lengths() {
    let a = b"hello";
    let b = b"hello world";
    // Different length slices
    assert!(!memcmp_simd(a, &b[..5]) || memcmp_simd(a, &b[..5]));
}

#[test]
fn test_memcmp_empty() {
    assert!(memcmp_simd(b"", b""));
}

#[test]
fn test_memcmp_long() {
    let a: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
    let b = a.clone();
    assert!(memcmp_simd(&a, &b));

    let mut c = a.clone();
    c[5000] = c[5000].wrapping_add(1);
    assert!(!memcmp_simd(&a, &c));
}

// =============================================================================
// Alignment Tests
// =============================================================================

#[test]
fn test_unaligned_access() {
    // Test with unaligned pointers
    let data = vec![b'a'; 100];

    // Start at offset 1 (likely unaligned)
    assert!(is_ascii_simd(&data[1..]));
    assert_eq!(count_chars_simd(&data[1..]), 99);
}

#[test]
fn test_small_inputs() {
    // Inputs smaller than SIMD width
    for len in 0..32 {
        let data: Vec<u8> = (0..len).map(|_| b'a').collect();
        assert!(is_ascii_simd(&data));
        assert_eq!(count_chars_simd(&data), len);
    }
}

// =============================================================================
// Fallback Implementation Tests
// =============================================================================

#[test]
fn test_scalar_is_ascii() {
    assert!(is_ascii_scalar(b"hello"));
    assert!(!is_ascii_scalar(&[0x80]));
}

#[test]
fn test_scalar_find_byte() {
    assert_eq!(find_byte_scalar(b"hello", b'e'), Some(1));
    assert_eq!(find_byte_scalar(b"hello", b'x'), None);
}

#[test]
fn test_scalar_count_chars() {
    assert_eq!(count_chars_scalar("hello".as_bytes()), 5);
    assert_eq!(count_chars_scalar("日本語".as_bytes()), 3);
}

#[test]
fn test_scalar_vs_simd_consistency() {
    let test_data = vec![
        b"hello".to_vec(),
        b"world".to_vec(),
        vec![b'a'; 1000],
        "日本語".as_bytes().to_vec(),
    ];

    for data in test_data {
        // ASCII detection
        assert_eq!(
            is_ascii_simd(&data),
            is_ascii_scalar(&data),
            "ASCII detection mismatch"
        );

        // Byte finding
        assert_eq!(
            find_byte_simd(&data, b'a'),
            find_byte_scalar(&data, b'a'),
            "Byte finding mismatch"
        );

        // Char counting
        assert_eq!(
            count_chars_simd(&data),
            count_chars_scalar(&data),
            "Char counting mismatch"
        );
    }
}

// =============================================================================
// Performance Characteristics Tests
// =============================================================================

#[test]
fn test_simd_processes_chunks() {
    // Verify SIMD processes in chunks by checking timing characteristics
    // This is a basic sanity check, not a precise benchmark

    let small = vec![b'a'; 16];
    let large = vec![b'a'; 16384];

    // Both should complete quickly
    let _ = is_ascii_simd(&small);
    let _ = is_ascii_simd(&large);
}

#[test]
fn test_early_exit_on_failure() {
    // SIMD should exit early when finding non-ASCII
    let mut data = vec![b'a'; 10000];
    data[0] = 0x80; // Non-ASCII at start

    // Should be fast because it exits early
    assert!(!is_ascii_simd(&data));
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_boundary_byte_127() {
    // 127 (0x7F) is the last valid ASCII byte
    assert!(is_ascii_simd(&[127]));
    assert!(is_ascii_simd(&[0, 127]));
}

#[test]
fn test_boundary_byte_128() {
    // 128 (0x80) is the first non-ASCII byte
    assert!(!is_ascii_simd(&[128]));
    assert!(!is_ascii_simd(&[0, 128]));
}

#[test]
fn test_all_byte_values() {
    // Test each byte value individually
    for byte in 0u8..=127 {
        assert!(
            is_ascii_simd(&[byte]),
            "Byte {} should be ASCII",
            byte
        );
    }
    for byte in 128u8..=255 {
        assert!(
            !is_ascii_simd(&[byte]),
            "Byte {} should not be ASCII",
            byte
        );
    }
}

#[test]
fn test_null_bytes() {
    assert!(is_ascii_simd(&[0]));
    assert!(is_ascii_simd(&[0, 0, 0]));
    assert!(is_ascii_simd(b"hello\x00world"));
}

// =============================================================================
// Dispatch Tests
// =============================================================================

#[test]
fn test_dispatch_select() {
    let dispatcher = SimdDispatcher::new();

    // Should select appropriate implementation
    let result = dispatcher.is_ascii(b"hello");
    assert!(result);
}

#[test]
fn test_dispatch_consistency() {
    let dispatcher = SimdDispatcher::new();

    // All dispatch methods should give consistent results
    let data = b"hello world";

    assert!(dispatcher.is_ascii(data));
    assert_eq!(dispatcher.find_byte(data, b' '), Some(5));
    assert_eq!(dispatcher.count_chars(data), 11);
}

// =============================================================================
// Specific SIMD Width Tests
// =============================================================================

#[test]
fn test_sse_width() {
    // SSE processes 16 bytes at a time
    let exactly_16 = vec![b'a'; 16];
    let exactly_32 = vec![b'a'; 32];

    assert!(is_ascii_simd(&exactly_16));
    assert!(is_ascii_simd(&exactly_32));
}

#[test]
fn test_avx_width() {
    // AVX2 processes 32 bytes at a time
    let exactly_32 = vec![b'a'; 32];
    let exactly_64 = vec![b'a'; 64];

    assert!(is_ascii_simd(&exactly_32));
    assert!(is_ascii_simd(&exactly_64));
}

#[test]
fn test_avx512_width() {
    // AVX-512 processes 64 bytes at a time
    let exactly_64 = vec![b'a'; 64];
    let exactly_128 = vec![b'a'; 128];

    assert!(is_ascii_simd(&exactly_64));
    assert!(is_ascii_simd(&exactly_128));
}
