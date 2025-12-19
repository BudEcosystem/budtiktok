//! SIMD-accelerated UTF-8 Validation (4.2.3)
//!
//! High-performance UTF-8 validation using SIMD instructions.
//! Based on the simdutf algorithm for processing 64 bytes per iteration with AVX-512
//! or 32 bytes with AVX2.
//!
//! UTF-8 encoding rules:
//! - ASCII: 0xxxxxxx (0x00-0x7F)
//! - 2-byte: 110xxxxx 10xxxxxx (0xC0-0xDF, 0x80-0xBF)
//! - 3-byte: 1110xxxx 10xxxxxx 10xxxxxx (0xE0-0xEF, 0x80-0xBF, 0x80-0xBF)
//! - 4-byte: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx (0xF0-0xF7, 0x80-0xBF, 0x80-0xBF, 0x80-0xBF)

/// Result of UTF-8 validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Utf8ValidationResult {
    /// Valid UTF-8
    Valid,
    /// Invalid UTF-8 at the given byte offset
    Invalid(usize),
    /// Incomplete sequence at end (may need more data)
    Incomplete(usize),
}

/// Validate a byte slice as UTF-8
///
/// Uses SIMD acceleration when available.
pub fn validate_utf8(input: &[u8]) -> Utf8ValidationResult {
    if input.is_empty() {
        return Utf8ValidationResult::Valid;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { validate_utf8_avx2(input) };
        }
    }

    validate_utf8_scalar(input)
}

/// Check if a byte slice is valid UTF-8
#[inline]
pub fn is_valid_utf8(input: &[u8]) -> bool {
    matches!(validate_utf8(input), Utf8ValidationResult::Valid)
}

/// Find the first invalid byte position (or None if valid)
#[inline]
pub fn find_invalid_utf8(input: &[u8]) -> Option<usize> {
    match validate_utf8(input) {
        Utf8ValidationResult::Invalid(pos) => Some(pos),
        Utf8ValidationResult::Incomplete(pos) => Some(pos),
        Utf8ValidationResult::Valid => None,
    }
}

/// Scalar UTF-8 validation (fallback)
fn validate_utf8_scalar(input: &[u8]) -> Utf8ValidationResult {
    let mut i = 0;

    while i < input.len() {
        let byte = input[i];

        if byte < 0x80 {
            // ASCII
            i += 1;
        } else if byte < 0xC2 {
            // Invalid leading byte (0x80-0xBF are continuation, 0xC0-0xC1 are overlong)
            return Utf8ValidationResult::Invalid(i);
        } else if byte < 0xE0 {
            // 2-byte sequence
            if i + 1 >= input.len() {
                return Utf8ValidationResult::Incomplete(i);
            }
            if !is_continuation(input[i + 1]) {
                return Utf8ValidationResult::Invalid(i + 1);
            }
            i += 2;
        } else if byte < 0xF0 {
            // 3-byte sequence
            if i + 2 >= input.len() {
                return Utf8ValidationResult::Incomplete(i);
            }
            if !is_continuation(input[i + 1]) || !is_continuation(input[i + 2]) {
                return Utf8ValidationResult::Invalid(i + 1);
            }
            // Check for overlong and surrogate
            let c1 = input[i + 1];
            if byte == 0xE0 && c1 < 0xA0 {
                return Utf8ValidationResult::Invalid(i); // Overlong
            }
            if byte == 0xED && c1 >= 0xA0 {
                return Utf8ValidationResult::Invalid(i); // Surrogate
            }
            i += 3;
        } else if byte < 0xF5 {
            // 4-byte sequence
            if i + 3 >= input.len() {
                return Utf8ValidationResult::Incomplete(i);
            }
            if !is_continuation(input[i + 1])
                || !is_continuation(input[i + 2])
                || !is_continuation(input[i + 3])
            {
                return Utf8ValidationResult::Invalid(i + 1);
            }
            // Check for overlong and out of range
            let c1 = input[i + 1];
            if byte == 0xF0 && c1 < 0x90 {
                return Utf8ValidationResult::Invalid(i); // Overlong
            }
            if byte == 0xF4 && c1 >= 0x90 {
                return Utf8ValidationResult::Invalid(i); // Above U+10FFFF
            }
            i += 4;
        } else {
            return Utf8ValidationResult::Invalid(i);
        }
    }

    Utf8ValidationResult::Valid
}

/// Check if a byte is a UTF-8 continuation byte (10xxxxxx)
#[inline]
fn is_continuation(byte: u8) -> bool {
    (byte & 0xC0) == 0x80
}

/// AVX2-accelerated UTF-8 validation
///
/// Uses SIMD for fast ASCII detection, falls back to scalar for non-ASCII.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn validate_utf8_avx2(input: &[u8]) -> Utf8ValidationResult {
    use std::arch::x86_64::*;

    // For short inputs, use scalar
    if input.len() < 32 {
        return validate_utf8_scalar(input);
    }

    let mut i = 0;

    // Process 32 bytes at a time, looking for all-ASCII chunks
    while i + 32 <= input.len() {
        let chunk = _mm256_loadu_si256(input.as_ptr().add(i) as *const __m256i);

        // Check if all ASCII (high bit = 0 for all bytes)
        let high_bits = _mm256_movemask_epi8(chunk);
        if high_bits == 0 {
            // All ASCII, continue
            i += 32;
        } else {
            // Found non-ASCII, use scalar from this point
            break;
        }
    }

    // Validate remaining bytes with scalar (handles multi-byte sequences correctly)
    if i < input.len() {
        match validate_utf8_scalar(&input[i..]) {
            Utf8ValidationResult::Valid => Utf8ValidationResult::Valid,
            Utf8ValidationResult::Invalid(pos) => Utf8ValidationResult::Invalid(i + pos),
            Utf8ValidationResult::Incomplete(pos) => Utf8ValidationResult::Incomplete(i + pos),
        }
    } else {
        Utf8ValidationResult::Valid
    }
}

/// Count the number of UTF-8 code points in a valid UTF-8 string
///
/// Uses SIMD for fast counting.
pub fn count_code_points(input: &[u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && input.len() >= 32 {
            return unsafe { count_code_points_avx2(input) };
        }
    }

    count_code_points_scalar(input)
}

/// Scalar code point counting
fn count_code_points_scalar(input: &[u8]) -> usize {
    // Count non-continuation bytes (bytes that don't match 10xxxxxx)
    input.iter().filter(|&&b| (b & 0xC0) != 0x80).count()
}

/// AVX2-accelerated code point counting
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_code_points_avx2(input: &[u8]) -> usize {
    use std::arch::x86_64::*;

    let mut count = 0usize;
    let mut i = 0;

    // Process 32 bytes at a time
    let cont_mask = _mm256_set1_epi8(0xC0u8 as i8);
    let cont_pattern = _mm256_set1_epi8(0x80u8 as i8);

    while i + 32 <= input.len() {
        let chunk = _mm256_loadu_si256(input.as_ptr().add(i) as *const __m256i);

        // Check which bytes are continuation bytes (10xxxxxx)
        let masked = _mm256_and_si256(chunk, cont_mask);
        let is_cont = _mm256_cmpeq_epi8(masked, cont_pattern);
        let cont_bits = _mm256_movemask_epi8(is_cont) as u32;

        // Count non-continuation bytes (32 - popcount of continuation)
        count += 32 - cont_bits.count_ones() as usize;
        i += 32;
    }

    // Handle remaining bytes
    count += count_code_points_scalar(&input[i..]);

    count
}

/// Find the byte index of the nth code point
pub fn nth_code_point_index(input: &[u8], n: usize) -> Option<usize> {
    let mut count = 0;
    for (i, &byte) in input.iter().enumerate() {
        if (byte & 0xC0) != 0x80 {
            if count == n {
                return Some(i);
            }
            count += 1;
        }
    }
    None
}

/// Statistics about UTF-8 content
#[derive(Debug, Clone, Default)]
pub struct Utf8Stats {
    /// Total bytes
    pub byte_count: usize,
    /// Number of ASCII bytes (0x00-0x7F)
    pub ascii_count: usize,
    /// Number of 2-byte sequences
    pub two_byte_count: usize,
    /// Number of 3-byte sequences
    pub three_byte_count: usize,
    /// Number of 4-byte sequences
    pub four_byte_count: usize,
    /// Total code points
    pub code_point_count: usize,
}

/// Compute statistics about UTF-8 content
pub fn compute_utf8_stats(input: &[u8]) -> Utf8Stats {
    let mut stats = Utf8Stats {
        byte_count: input.len(),
        ..Default::default()
    };

    let mut i = 0;
    while i < input.len() {
        let byte = input[i];
        if byte < 0x80 {
            stats.ascii_count += 1;
            stats.code_point_count += 1;
            i += 1;
        } else if byte < 0xE0 {
            stats.two_byte_count += 1;
            stats.code_point_count += 1;
            i += 2;
        } else if byte < 0xF0 {
            stats.three_byte_count += 1;
            stats.code_point_count += 1;
            i += 3;
        } else {
            stats.four_byte_count += 1;
            stats.code_point_count += 1;
            i += 4;
        }
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_ascii() {
        assert_eq!(validate_utf8(b"hello world"), Utf8ValidationResult::Valid);
        assert!(is_valid_utf8(b"hello world"));
    }

    #[test]
    fn test_valid_multibyte() {
        assert_eq!(validate_utf8("日本語".as_bytes()), Utf8ValidationResult::Valid);
        assert_eq!(validate_utf8("café".as_bytes()), Utf8ValidationResult::Valid);
        assert_eq!(validate_utf8("🎉".as_bytes()), Utf8ValidationResult::Valid);
    }

    #[test]
    fn test_invalid_continuation() {
        // Continuation byte without leading byte
        assert!(matches!(
            validate_utf8(&[0x80]),
            Utf8ValidationResult::Invalid(0)
        ));
    }

    #[test]
    fn test_invalid_overlong() {
        // Overlong 2-byte encoding of ASCII
        assert!(matches!(
            validate_utf8(&[0xC0, 0x80]),
            Utf8ValidationResult::Invalid(0)
        ));
    }

    #[test]
    fn test_invalid_surrogate() {
        // UTF-16 surrogate encoded in UTF-8 (invalid)
        assert!(matches!(
            validate_utf8(&[0xED, 0xA0, 0x80]),
            Utf8ValidationResult::Invalid(0)
        ));
    }

    #[test]
    fn test_invalid_too_large() {
        // Above U+10FFFF
        assert!(matches!(
            validate_utf8(&[0xF4, 0x90, 0x80, 0x80]),
            Utf8ValidationResult::Invalid(0)
        ));
    }

    #[test]
    fn test_incomplete_sequence() {
        // Incomplete 2-byte sequence
        assert!(matches!(
            validate_utf8(&[0xC2]),
            Utf8ValidationResult::Incomplete(0)
        ));

        // Incomplete 3-byte sequence
        assert!(matches!(
            validate_utf8(&[0xE2, 0x82]),
            Utf8ValidationResult::Incomplete(0)
        ));
    }

    #[test]
    fn test_empty() {
        assert_eq!(validate_utf8(b""), Utf8ValidationResult::Valid);
    }

    #[test]
    fn test_find_invalid() {
        assert_eq!(find_invalid_utf8(b"hello"), None);
        assert_eq!(find_invalid_utf8(&[0x80]), Some(0));
    }

    #[test]
    fn test_count_code_points() {
        assert_eq!(count_code_points(b"hello"), 5);
        assert_eq!(count_code_points("日本語".as_bytes()), 3);
        assert_eq!(count_code_points("café".as_bytes()), 4);
        assert_eq!(count_code_points("🎉".as_bytes()), 1);
    }

    #[test]
    fn test_count_code_points_mixed() {
        let s = "Hello 日本語 🎉";
        assert_eq!(count_code_points(s.as_bytes()), s.chars().count());
    }

    #[test]
    fn test_nth_code_point_index() {
        let s = "日本語";
        assert_eq!(nth_code_point_index(s.as_bytes(), 0), Some(0));
        assert_eq!(nth_code_point_index(s.as_bytes(), 1), Some(3)); // 日 is 3 bytes
        assert_eq!(nth_code_point_index(s.as_bytes(), 2), Some(6));
        assert_eq!(nth_code_point_index(s.as_bytes(), 3), None);
    }

    #[test]
    fn test_utf8_stats() {
        let stats = compute_utf8_stats("Hello 日本".as_bytes());
        assert_eq!(stats.ascii_count, 6); // 'H', 'e', 'l', 'l', 'o', ' '
        assert_eq!(stats.three_byte_count, 2); // 日, 本
        assert_eq!(stats.code_point_count, 8);
    }

    #[test]
    fn test_long_valid_string() {
        // Test with string longer than 32 bytes to trigger SIMD path
        let s = "The quick brown fox jumps over the lazy dog. 日本語テスト。";
        assert_eq!(validate_utf8(s.as_bytes()), Utf8ValidationResult::Valid);
    }

    #[test]
    fn test_long_invalid_string() {
        // Create a string with invalid byte in the middle
        let mut bytes = vec![b'a'; 50];
        bytes[40] = 0x80; // Invalid continuation byte
        assert!(matches!(
            validate_utf8(&bytes),
            Utf8ValidationResult::Invalid(40)
        ));
    }

    #[test]
    fn test_all_ascii_long() {
        let bytes = vec![b'a'; 100];
        assert_eq!(validate_utf8(&bytes), Utf8ValidationResult::Valid);
        assert_eq!(count_code_points(&bytes), 100);
    }

    #[test]
    fn test_boundary_sequences() {
        // Test boundaries between chunks (32 bytes)
        let mut s = String::new();
        for _ in 0..10 {
            s.push('日'); // 3-byte char
        }
        s.push_str("abc"); // End with ASCII
        assert_eq!(validate_utf8(s.as_bytes()), Utf8ValidationResult::Valid);
    }
}
