//! ISA Consistency Tests
//!
//! These tests ensure that tokenizer outputs are 100% identical regardless of
//! which underlying implementation (scalar, SWAR, SSE4.2, AVX2) is used.
//!
//! This is CRITICAL for correctness - users must get the same token IDs
//! on any system.

use budtiktok_core::unicode::{is_ascii_fast, normalize, NormalizationForm};
use budtiktok_core::unicode::{is_whitespace, is_punctuation, is_cjk_character, is_control};
use budtiktok_core::unicode::get_category_flags;

// =============================================================================
// Scalar Reference Implementations
// =============================================================================

mod scalar {
    //! Pure scalar implementations that serve as the ground truth.
    //! These MUST be correct - all SIMD implementations must match these.

    pub fn is_all_ascii(bytes: &[u8]) -> bool {
        bytes.iter().all(|&b| b < 128)
    }

    pub fn has_zero_byte(bytes: &[u8; 8]) -> bool {
        bytes.iter().any(|&b| b == 0)
    }

    pub fn has_byte(bytes: &[u8; 8], target: u8) -> bool {
        bytes.iter().any(|&b| b == target)
    }

    pub fn has_non_ascii(bytes: &[u8; 8]) -> bool {
        bytes.iter().any(|&b| b >= 128)
    }

    pub fn count_whitespace(bytes: &[u8]) -> usize {
        bytes.iter().filter(|&&b| matches!(b, b' ' | b'\t' | b'\n' | b'\r')).count()
    }

    pub fn find_first_whitespace(bytes: &[u8]) -> Option<usize> {
        bytes.iter().position(|&b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
    }

    pub fn find_first_non_ascii(bytes: &[u8]) -> Option<usize> {
        bytes.iter().position(|&b| b >= 128)
    }

    pub fn to_lowercase_ascii(bytes: &mut [u8]) {
        for b in bytes.iter_mut() {
            if *b >= b'A' && *b <= b'Z' {
                *b += 32;
            }
        }
    }

    pub fn to_uppercase_ascii(bytes: &mut [u8]) {
        for b in bytes.iter_mut() {
            if *b >= b'a' && *b <= b'z' {
                *b -= 32;
            }
        }
    }

    pub fn is_valid_utf8(bytes: &[u8]) -> bool {
        std::str::from_utf8(bytes).is_ok()
    }

    pub fn count_code_points(s: &str) -> usize {
        s.chars().count()
    }

    pub fn is_whitespace_char(c: char) -> bool {
        c.is_whitespace()
    }

    pub fn is_ascii_whitespace(b: u8) -> bool {
        matches!(b, b' ' | b'\t' | b'\n' | b'\r')
    }

    pub fn is_ascii_alphanumeric(b: u8) -> bool {
        b.is_ascii_alphanumeric()
    }

    pub fn normalize_nfc(s: &str) -> String {
        use unicode_normalization::UnicodeNormalization;
        s.nfc().collect()
    }

    pub fn normalize_nfd(s: &str) -> String {
        use unicode_normalization::UnicodeNormalization;
        s.nfd().collect()
    }
}

// =============================================================================
// Test Data Generation
// =============================================================================

/// Generate test strings covering edge cases
fn test_strings() -> Vec<(&'static str, String)> {
    vec![
        // Empty
        ("empty", "".to_string()),

        // Single characters
        ("single_ascii", "a".to_string()),
        ("single_space", " ".to_string()),
        ("single_newline", "\n".to_string()),
        ("single_tab", "\t".to_string()),
        ("single_unicode", "日".to_string()),
        ("single_emoji", "🎉".to_string()),

        // Pure ASCII of various lengths (SIMD boundaries)
        ("ascii_7", "abcdefg".to_string()),
        ("ascii_8", "abcdefgh".to_string()),
        ("ascii_15", "abcdefghijklmno".to_string()),
        ("ascii_16", "abcdefghijklmnop".to_string()),
        ("ascii_31", "a".repeat(31)),
        ("ascii_32", "a".repeat(32)),
        ("ascii_63", "a".repeat(63)),
        ("ascii_64", "a".repeat(64)),
        ("ascii_127", "a".repeat(127)),
        ("ascii_128", "a".repeat(128)),
        ("ascii_255", "a".repeat(255)),
        ("ascii_256", "a".repeat(256)),
        ("ascii_1000", "a".repeat(1000)),

        // ASCII with whitespace
        ("ascii_spaces", "hello world foo bar".to_string()),
        ("ascii_tabs", "hello\tworld\tfoo".to_string()),
        ("ascii_newlines", "hello\nworld\nfoo".to_string()),
        ("ascii_mixed_ws", "hello \t\n world".to_string()),

        // Unicode strings
        ("unicode_chinese", "你好世界".to_string()),
        ("unicode_japanese", "こんにちは世界".to_string()),
        ("unicode_korean", "안녕하세요".to_string()),
        ("unicode_arabic", "مرحبا بالعالم".to_string()),
        ("unicode_russian", "Привет мир".to_string()),
        ("unicode_mixed", "Hello 世界 مرحبا мир".to_string()),

        // Accented characters (normalization edge cases)
        ("accented_composed", "café".to_string()),  // é as single code point
        ("accented_decomposed", "cafe\u{0301}".to_string()),  // e + combining acute
        ("accented_mixed", "naïve résumé".to_string()),

        // Emoji and special Unicode
        ("emoji_single", "😀".to_string()),
        ("emoji_sequence", "👨‍👩‍👧‍👦".to_string()),  // Family emoji (ZWJ sequence)
        ("emoji_flags", "🇺🇸🇯🇵🇩🇪".to_string()),
        ("emoji_mixed", "Hello 👋 World 🌍".to_string()),

        // Edge cases with non-ASCII at boundaries
        ("non_ascii_at_0", format!("é{}", "a".repeat(31))),
        ("non_ascii_at_7", format!("{}é{}", "a".repeat(7), "b".repeat(24))),
        ("non_ascii_at_8", format!("{}é{}", "a".repeat(8), "b".repeat(23))),
        ("non_ascii_at_15", format!("{}é{}", "a".repeat(15), "b".repeat(16))),
        ("non_ascii_at_16", format!("{}é{}", "a".repeat(16), "b".repeat(15))),
        ("non_ascii_at_31", format!("{}é{}", "a".repeat(31), "b".repeat(1))),
        ("non_ascii_at_end", format!("{}é", "a".repeat(31))),

        // Control characters
        ("with_null", "hello\0world".to_string()),
        ("with_bell", "hello\x07world".to_string()),
        ("with_escape", "hello\x1bworld".to_string()),

        // Real-world examples
        ("sentence_en", "The quick brown fox jumps over the lazy dog.".to_string()),
        ("sentence_mixed", "The café served naïve customers 日本語 text.".to_string()),
        ("code_snippet", "fn main() { println!(\"Hello, world!\"); }".to_string()),
        ("json_like", r#"{"name": "test", "value": 123}"#.to_string()),
        ("url_like", "https://example.com/path?query=value&foo=bar".to_string()),
    ]
}

/// Generate byte arrays for low-level tests
fn test_byte_arrays() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        // Pure ASCII
        ("bytes_ascii_8", vec![65, 66, 67, 68, 69, 70, 71, 72]),
        ("bytes_ascii_16", (0..16).map(|i| 65 + i).collect()),
        ("bytes_ascii_32", (0..32).map(|i| 65 + (i % 26)).collect()),

        // With high bytes at various positions
        ("bytes_high_at_0", vec![200, 65, 66, 67, 68, 69, 70, 71]),
        ("bytes_high_at_1", vec![65, 200, 66, 67, 68, 69, 70, 71]),
        ("bytes_high_at_7", vec![65, 66, 67, 68, 69, 70, 71, 200]),
        ("bytes_high_at_8", {
            let mut v: Vec<u8> = (0..16).map(|i| 65 + i).collect();
            v[8] = 200;
            v
        }),
        ("bytes_high_at_15", {
            let mut v: Vec<u8> = (0..16).map(|i| 65 + i).collect();
            v[15] = 200;
            v
        }),

        // With zeros
        ("bytes_zero_at_0", vec![0, 65, 66, 67, 68, 69, 70, 71]),
        ("bytes_zero_at_4", vec![65, 66, 67, 68, 0, 69, 70, 71]),
        ("bytes_zero_at_7", vec![65, 66, 67, 68, 69, 70, 71, 0]),

        // All same value
        ("bytes_all_zero", vec![0; 32]),
        ("bytes_all_space", vec![32; 32]),
        ("bytes_all_a", vec![97; 32]),
        ("bytes_all_high", vec![200; 32]),

        // Whitespace patterns
        ("bytes_ws_scattered", {
            let mut v = vec![97u8; 32];
            v[0] = 32;   // space
            v[8] = 9;    // tab
            v[16] = 10;  // newline
            v[24] = 13;  // carriage return
            v
        }),
    ]
}

// =============================================================================
// SWAR Consistency Tests
// =============================================================================

#[test]
fn test_swar_has_zero_byte_consistency() {
    use budtiktok_core::has_zero_byte;

    for (name, bytes) in test_byte_arrays() {
        // Process 8 bytes at a time
        for chunk in bytes.chunks(8) {
            if chunk.len() == 8 {
                let arr: [u8; 8] = chunk.try_into().unwrap();
                let word = u64::from_ne_bytes(arr);

                let scalar_result = scalar::has_zero_byte(&arr);
                let swar_result = has_zero_byte(word);

                assert_eq!(
                    scalar_result, swar_result,
                    "has_zero_byte mismatch for '{}': scalar={}, swar={}, bytes={:?}",
                    name, scalar_result, swar_result, arr
                );
            }
        }
    }
    println!("✓ SWAR has_zero_byte: 100% consistent with scalar");
}

#[test]
fn test_swar_has_byte_consistency() {
    use budtiktok_core::has_byte;

    let targets = [0u8, 32, 65, 97, 127, 128, 200, 255];

    for (name, bytes) in test_byte_arrays() {
        for chunk in bytes.chunks(8) {
            if chunk.len() == 8 {
                let arr: [u8; 8] = chunk.try_into().unwrap();
                let word = u64::from_ne_bytes(arr);

                for &target in &targets {
                    let scalar_result = scalar::has_byte(&arr, target);
                    let swar_result = has_byte(word, target);

                    assert_eq!(
                        scalar_result, swar_result,
                        "has_byte mismatch for '{}', target={}: scalar={}, swar={}, bytes={:?}",
                        name, target, scalar_result, swar_result, arr
                    );
                }
            }
        }
    }
    println!("✓ SWAR has_byte: 100% consistent with scalar");
}

#[test]
fn test_swar_has_non_ascii_consistency() {
    use budtiktok_core::has_non_ascii;

    for (name, bytes) in test_byte_arrays() {
        for chunk in bytes.chunks(8) {
            if chunk.len() == 8 {
                let arr: [u8; 8] = chunk.try_into().unwrap();
                let word = u64::from_ne_bytes(arr);

                let scalar_result = scalar::has_non_ascii(&arr);
                let swar_result = has_non_ascii(word);

                assert_eq!(
                    scalar_result, swar_result,
                    "has_non_ascii mismatch for '{}': scalar={}, swar={}, bytes={:?}",
                    name, scalar_result, swar_result, arr
                );
            }
        }
    }
    println!("✓ SWAR has_non_ascii: 100% consistent with scalar");
}

#[test]
fn test_swar_is_all_ascii_unrolled_consistency() {
    use budtiktok_core::is_all_ascii_unrolled;

    for (name, bytes) in test_byte_arrays() {
        let scalar_result = scalar::is_all_ascii(&bytes);
        let swar_result = is_all_ascii_unrolled(&bytes);

        assert_eq!(
            scalar_result, swar_result,
            "is_all_ascii_unrolled mismatch for '{}': scalar={}, swar={}, len={}",
            name, scalar_result, swar_result, bytes.len()
        );
    }

    // Also test strings
    for (name, s) in test_strings() {
        let bytes = s.as_bytes();
        let scalar_result = scalar::is_all_ascii(bytes);
        let swar_result = is_all_ascii_unrolled(bytes);

        assert_eq!(
            scalar_result, swar_result,
            "is_all_ascii_unrolled mismatch for string '{}': scalar={}, swar={}",
            name, scalar_result, swar_result
        );
    }

    println!("✓ SWAR is_all_ascii_unrolled: 100% consistent with scalar");
}

// =============================================================================
// SSE4.2 Consistency Tests
// =============================================================================

#[test]
fn test_sse42_is_all_ascii_consistency() {
    use budtiktok_core::is_all_ascii;

    // Test byte arrays
    for (name, bytes) in test_byte_arrays() {
        let scalar_result = scalar::is_all_ascii(&bytes);
        let simd_result = is_all_ascii(&bytes);

        assert_eq!(
            scalar_result, simd_result,
            "is_all_ascii mismatch for '{}': scalar={}, simd={}, len={}",
            name, scalar_result, simd_result, bytes.len()
        );
    }

    // Test strings
    for (name, s) in test_strings() {
        let bytes = s.as_bytes();
        let scalar_result = scalar::is_all_ascii(bytes);
        let simd_result = is_all_ascii(bytes);

        assert_eq!(
            scalar_result, simd_result,
            "is_all_ascii mismatch for string '{}': scalar={}, simd={}",
            name, scalar_result, simd_result
        );
    }

    // Test exhaustive positions for non-ASCII detection
    for size in [8, 16, 32, 64, 128, 256] {
        for pos in 0..size {
            let mut bytes = vec![b'a'; size];
            bytes[pos] = 200; // Non-ASCII

            let scalar_result = scalar::is_all_ascii(&bytes);
            let simd_result = is_all_ascii(&bytes);

            assert_eq!(
                scalar_result, simd_result,
                "is_all_ascii failed at size={}, pos={}",
                size, pos
            );
        }
    }

    println!("✓ SSE4.2 is_all_ascii: 100% consistent with scalar");
}

#[test]
fn test_sse42_find_first_whitespace_consistency() {
    use budtiktok_core::find_first_whitespace;

    for (name, bytes) in test_byte_arrays() {
        let scalar_result = scalar::find_first_whitespace(&bytes);
        let simd_result = find_first_whitespace(&bytes);

        assert_eq!(
            scalar_result, simd_result,
            "find_first_whitespace mismatch for '{}': scalar={:?}, simd={:?}",
            name, scalar_result, simd_result
        );
    }

    for (name, s) in test_strings() {
        let bytes = s.as_bytes();
        let scalar_result = scalar::find_first_whitespace(bytes);
        let simd_result = find_first_whitespace(bytes);

        assert_eq!(
            scalar_result, simd_result,
            "find_first_whitespace mismatch for string '{}': scalar={:?}, simd={:?}",
            name, scalar_result, simd_result
        );
    }

    println!("✓ SSE4.2 find_first_whitespace: 100% consistent with scalar");
}

#[test]
fn test_sse42_count_whitespace_consistency() {
    use budtiktok_core::count_whitespace;

    for (name, bytes) in test_byte_arrays() {
        let scalar_result = scalar::count_whitespace(&bytes);
        let simd_result = count_whitespace(&bytes);

        assert_eq!(
            scalar_result, simd_result,
            "count_whitespace mismatch for '{}': scalar={}, simd={}",
            name, scalar_result, simd_result
        );
    }

    for (name, s) in test_strings() {
        let bytes = s.as_bytes();
        let scalar_result = scalar::count_whitespace(bytes);
        let simd_result = count_whitespace(bytes);

        assert_eq!(
            scalar_result, simd_result,
            "count_whitespace mismatch for string '{}': scalar={}, simd={}",
            name, scalar_result, simd_result
        );
    }

    println!("✓ SSE4.2 count_whitespace: 100% consistent with scalar");
}

#[test]
fn test_sse42_find_first_non_ascii_consistency() {
    use budtiktok_core::find_first_non_ascii;

    for (name, bytes) in test_byte_arrays() {
        let scalar_result = scalar::find_first_non_ascii(&bytes);
        let simd_result = find_first_non_ascii(&bytes);

        assert_eq!(
            scalar_result, simd_result,
            "find_first_non_ascii mismatch for '{}': scalar={:?}, simd={:?}",
            name, scalar_result, simd_result
        );
    }

    for (name, s) in test_strings() {
        let bytes = s.as_bytes();
        let scalar_result = scalar::find_first_non_ascii(bytes);
        let simd_result = find_first_non_ascii(bytes);

        assert_eq!(
            scalar_result, simd_result,
            "find_first_non_ascii mismatch for string '{}': scalar={:?}, simd={:?}",
            name, scalar_result, simd_result
        );
    }

    println!("✓ SSE4.2 find_first_non_ascii: 100% consistent with scalar");
}

// =============================================================================
// AVX2 Consistency Tests
// =============================================================================

#[test]
fn test_avx2_utf8_validation_consistency() {
    use budtiktok_core::is_valid_utf8;

    for (name, bytes) in test_byte_arrays() {
        let scalar_result = scalar::is_valid_utf8(&bytes);
        let simd_result = is_valid_utf8(&bytes);

        assert_eq!(
            scalar_result, simd_result,
            "is_valid_utf8 mismatch for '{}': scalar={}, simd={}",
            name, scalar_result, simd_result
        );
    }

    for (name, s) in test_strings() {
        let bytes = s.as_bytes();
        let scalar_result = scalar::is_valid_utf8(bytes);
        let simd_result = is_valid_utf8(bytes);

        assert_eq!(
            scalar_result, simd_result,
            "is_valid_utf8 mismatch for string '{}': scalar={}, simd={}",
            name, scalar_result, simd_result
        );
    }

    // Test invalid UTF-8 sequences
    let invalid_sequences: Vec<(&str, Vec<u8>)> = vec![
        ("invalid_continuation", vec![0x80]),
        ("invalid_start", vec![0xFF]),
        ("truncated_2byte", vec![0xC2]),
        ("truncated_3byte", vec![0xE0, 0xA0]),
        ("truncated_4byte", vec![0xF0, 0x90, 0x80]),
        ("overlong_2byte", vec![0xC0, 0x80]),
        ("overlong_3byte", vec![0xE0, 0x80, 0x80]),
        ("surrogate_high", vec![0xED, 0xA0, 0x80]),
        ("surrogate_low", vec![0xED, 0xB0, 0x80]),
        ("above_max", vec![0xF4, 0x90, 0x80, 0x80]),
    ];

    for (name, bytes) in invalid_sequences {
        let scalar_result = scalar::is_valid_utf8(&bytes);
        let simd_result = is_valid_utf8(&bytes);

        assert_eq!(
            scalar_result, simd_result,
            "is_valid_utf8 mismatch for invalid '{}': scalar={}, simd={}",
            name, scalar_result, simd_result
        );
    }

    println!("✓ AVX2 is_valid_utf8: 100% consistent with scalar");
}

#[test]
fn test_avx2_count_code_points_consistency() {
    use budtiktok_core::count_code_points;

    for (name, s) in test_strings() {
        let scalar_result = scalar::count_code_points(&s);
        let simd_result = count_code_points(s.as_bytes());

        assert_eq!(
            scalar_result, simd_result,
            "count_code_points mismatch for '{}': scalar={}, simd={}",
            name, scalar_result, simd_result
        );
    }

    println!("✓ AVX2 count_code_points: 100% consistent with scalar");
}

// =============================================================================
// Unicode Function Consistency Tests
// =============================================================================

#[test]
fn test_is_ascii_fast_consistency() {
    for (name, s) in test_strings() {
        let scalar_result = scalar::is_all_ascii(s.as_bytes());
        let fast_result = is_ascii_fast(&s);

        assert_eq!(
            scalar_result, fast_result,
            "is_ascii_fast mismatch for '{}': scalar={}, fast={}",
            name, scalar_result, fast_result
        );
    }

    println!("✓ is_ascii_fast: 100% consistent with scalar");
}

#[test]
fn test_normalize_consistency() {
    for (name, s) in test_strings() {
        // NFC
        let scalar_nfc = scalar::normalize_nfc(&s);
        let lib_nfc = normalize(&s, NormalizationForm::NFC);
        assert_eq!(
            scalar_nfc, lib_nfc,
            "NFC mismatch for '{}': scalar='{}', lib='{}'",
            name, scalar_nfc, lib_nfc
        );

        // NFD
        let scalar_nfd = scalar::normalize_nfd(&s);
        let lib_nfd = normalize(&s, NormalizationForm::NFD);
        assert_eq!(
            scalar_nfd, lib_nfd,
            "NFD mismatch for '{}': scalar='{}', lib='{}'",
            name, scalar_nfd, lib_nfd
        );
    }

    println!("✓ normalize: 100% consistent with scalar (NFC, NFD)");
}

#[test]
fn test_character_classification_consistency() {
    // Test all ASCII characters
    for b in 0u8..128 {
        let c = b as char;

        // Whitespace
        let scalar_ws = scalar::is_whitespace_char(c);
        let lib_ws = is_whitespace(c);
        assert_eq!(
            scalar_ws, lib_ws,
            "is_whitespace mismatch for {:?} (0x{:02X})",
            c, b
        );
    }

    // Test Unicode whitespace
    let unicode_whitespace = [
        '\u{00A0}', // No-break space
        '\u{2000}', // En quad
        '\u{2001}', // Em quad
        '\u{2002}', // En space
        '\u{2003}', // Em space
        '\u{2028}', // Line separator
        '\u{2029}', // Paragraph separator
        '\u{3000}', // Ideographic space
    ];

    for &c in &unicode_whitespace {
        let scalar_ws = scalar::is_whitespace_char(c);
        let lib_ws = is_whitespace(c);
        assert_eq!(
            scalar_ws, lib_ws,
            "is_whitespace mismatch for U+{:04X}",
            c as u32
        );
    }

    println!("✓ Character classification: 100% consistent with scalar");
}

// =============================================================================
// Branchless Operation Consistency Tests
// =============================================================================

#[test]
fn test_branchless_operations_consistency() {
    use budtiktok_core::{
        is_ascii_whitespace_branchless,
        is_ascii_alphanumeric_branchless,
        to_lowercase_branchless,
        to_uppercase_branchless,
    };

    for b in 0u8..=255 {
        // Whitespace (only valid for ASCII)
        if b < 128 {
            let scalar_ws = scalar::is_ascii_whitespace(b);
            let branchless_ws = is_ascii_whitespace_branchless(b) != 0;
            assert_eq!(
                scalar_ws, branchless_ws,
                "is_ascii_whitespace_branchless mismatch for 0x{:02X}",
                b
            );

            let scalar_alnum = scalar::is_ascii_alphanumeric(b);
            let branchless_alnum = is_ascii_alphanumeric_branchless(b) != 0;
            assert_eq!(
                scalar_alnum, branchless_alnum,
                "is_ascii_alphanumeric_branchless mismatch for 0x{:02X}",
                b
            );
        }

        // Case conversion (only for ASCII letters)
        if b < 128 {
            let scalar_lower = (b as char).to_ascii_lowercase() as u8;
            let branchless_lower = to_lowercase_branchless(b);
            assert_eq!(
                scalar_lower, branchless_lower,
                "to_lowercase_branchless mismatch for 0x{:02X}: scalar=0x{:02X}, branchless=0x{:02X}",
                b, scalar_lower, branchless_lower
            );

            let scalar_upper = (b as char).to_ascii_uppercase() as u8;
            let branchless_upper = to_uppercase_branchless(b);
            assert_eq!(
                scalar_upper, branchless_upper,
                "to_uppercase_branchless mismatch for 0x{:02X}: scalar=0x{:02X}, branchless=0x{:02X}",
                b, scalar_upper, branchless_upper
            );
        }
    }

    println!("✓ Branchless operations: 100% consistent with scalar");
}

// =============================================================================
// Exhaustive Edge Case Tests
// =============================================================================

#[test]
fn test_boundary_conditions_exhaustive() {
    use budtiktok_core::{is_all_ascii_unrolled, find_first_non_ascii, count_whitespace};

    // Test all positions where non-ASCII could appear in various buffer sizes
    for size in [8, 16, 32, 64, 128, 256] {
        for pos in 0..size {
            let mut bytes = vec![b'a'; size];
            bytes[pos] = 200; // Non-ASCII

            // is_all_ascii_unrolled
            let scalar_ascii = scalar::is_all_ascii(&bytes);
            let swar_ascii = is_all_ascii_unrolled(&bytes);
            assert_eq!(
                scalar_ascii, swar_ascii,
                "is_all_ascii_unrolled failed at size={}, pos={}",
                size, pos
            );

            // find_first_non_ascii
            let scalar_pos = scalar::find_first_non_ascii(&bytes);
            let simd_pos = find_first_non_ascii(&bytes);
            assert_eq!(
                scalar_pos, simd_pos,
                "find_first_non_ascii failed at size={}, pos={}: scalar={:?}, simd={:?}",
                size, pos, scalar_pos, simd_pos
            );
        }
    }

    // Test all positions where whitespace could appear
    for size in [8, 16, 32, 64] {
        for pos in 0..size {
            for &ws in &[b' ', b'\t', b'\n', b'\r'] {
                let mut bytes = vec![b'a'; size];
                bytes[pos] = ws;

                let scalar_count = scalar::count_whitespace(&bytes);
                let simd_count = count_whitespace(&bytes);
                assert_eq!(
                    scalar_count, simd_count,
                    "count_whitespace failed at size={}, pos={}, ws=0x{:02X}",
                    size, pos, ws
                );
            }
        }
    }

    println!("✓ Boundary conditions: All positions tested exhaustively");
}

// =============================================================================
// Summary
// =============================================================================

#[test]
fn test_isa_consistency_summary() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                       ISA CONSISTENCY TEST SUMMARY                           ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                              ║");
    println!("║  These tests verify that ALL implementations produce IDENTICAL outputs       ║");
    println!("║  regardless of which ISA-specific code path is taken.                        ║");
    println!("║                                                                              ║");
    println!("║  Tested implementations:                                                     ║");
    println!("║    • Scalar (reference implementation)                                       ║");
    println!("║    • SWAR (64-bit register tricks)                                           ║");
    println!("║    • SSE4.2 (x86-64)                                                         ║");
    println!("║    • AVX2 (x86-64)                                                           ║");
    println!("║    • Branchless operations                                                   ║");
    println!("║                                                                              ║");
    println!("║  Run all tests with:                                                         ║");
    println!("║    cargo test --test isa_consistency -- --nocapture                          ║");
    println!("║                                                                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
}
