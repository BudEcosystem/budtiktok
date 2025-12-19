//! SIMD-accelerated whitespace detection
//!
//! This module provides high-performance whitespace detection and splitting
//! using SIMD intrinsics (AVX2, AVX-512, NEON).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Find whitespace positions using SIMD (auto-detects best backend)
pub fn find_whitespace_simd(bytes: &[u8]) -> Vec<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { find_whitespace_avx2(bytes) };
        }
        if std::arch::is_x86_feature_detected!("sse4.2") {
            return unsafe { find_whitespace_sse42(bytes) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { find_whitespace_neon(bytes) };
    }

    // Scalar fallback
    find_whitespace_scalar(bytes)
}

/// Scalar implementation of whitespace detection
fn find_whitespace_scalar(bytes: &[u8]) -> Vec<usize> {
    bytes
        .iter()
        .enumerate()
        .filter(|(_, &b)| b == b' ' || b == b'\t' || b == b'\n' || b == b'\r')
        .map(|(i, _)| i)
        .collect()
}

/// AVX2 implementation of whitespace detection (32 bytes at a time)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_whitespace_avx2(bytes: &[u8]) -> Vec<usize> {
    let mut positions = Vec::new();
    let len = bytes.len();

    // Process 32 bytes at a time using AVX2
    let space = _mm256_set1_epi8(b' ' as i8);
    let tab = _mm256_set1_epi8(b'\t' as i8);
    let newline = _mm256_set1_epi8(b'\n' as i8);
    let carriage = _mm256_set1_epi8(b'\r' as i8);

    let mut i = 0;
    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);

        // Check for each whitespace character
        let space_mask = _mm256_cmpeq_epi8(chunk, space);
        let tab_mask = _mm256_cmpeq_epi8(chunk, tab);
        let newline_mask = _mm256_cmpeq_epi8(chunk, newline);
        let carriage_mask = _mm256_cmpeq_epi8(chunk, carriage);

        // Combine all whitespace masks
        let combined1 = _mm256_or_si256(space_mask, tab_mask);
        let combined2 = _mm256_or_si256(newline_mask, carriage_mask);
        let combined = _mm256_or_si256(combined1, combined2);

        // Extract bitmask
        let mask = _mm256_movemask_epi8(combined) as u32;

        // Extract positions from bitmask
        if mask != 0 {
            let mut m = mask;
            while m != 0 {
                let pos = m.trailing_zeros() as usize;
                positions.push(i + pos);
                m &= m - 1; // Clear lowest set bit
            }
        }

        i += 32;
    }

    // Handle remaining bytes with scalar
    for j in i..len {
        let b = bytes[j];
        if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
            positions.push(j);
        }
    }

    positions
}

/// SSE4.2 implementation of whitespace detection (16 bytes at a time)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn find_whitespace_sse42(bytes: &[u8]) -> Vec<usize> {
    let mut positions = Vec::new();
    let len = bytes.len();

    // Process 16 bytes at a time using SSE4.2
    let space = _mm_set1_epi8(b' ' as i8);
    let tab = _mm_set1_epi8(b'\t' as i8);
    let newline = _mm_set1_epi8(b'\n' as i8);
    let carriage = _mm_set1_epi8(b'\r' as i8);

    let mut i = 0;
    while i + 16 <= len {
        let chunk = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);

        // Check for each whitespace character
        let space_mask = _mm_cmpeq_epi8(chunk, space);
        let tab_mask = _mm_cmpeq_epi8(chunk, tab);
        let newline_mask = _mm_cmpeq_epi8(chunk, newline);
        let carriage_mask = _mm_cmpeq_epi8(chunk, carriage);

        // Combine all whitespace masks
        let combined1 = _mm_or_si128(space_mask, tab_mask);
        let combined2 = _mm_or_si128(newline_mask, carriage_mask);
        let combined = _mm_or_si128(combined1, combined2);

        // Extract bitmask
        let mask = _mm_movemask_epi8(combined) as u32;

        // Extract positions from bitmask
        if mask != 0 {
            let mut m = mask;
            while m != 0 {
                let pos = m.trailing_zeros() as usize;
                positions.push(i + pos);
                m &= m - 1; // Clear lowest set bit
            }
        }

        i += 16;
    }

    // Handle remaining bytes with scalar
    for j in i..len {
        let b = bytes[j];
        if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
            positions.push(j);
        }
    }

    positions
}

/// NEON implementation of whitespace detection (16 bytes at a time)
#[cfg(target_arch = "aarch64")]
unsafe fn find_whitespace_neon(bytes: &[u8]) -> Vec<usize> {
    let mut positions = Vec::new();
    let len = bytes.len();

    // Process 16 bytes at a time using NEON
    let space = vdupq_n_u8(b' ');
    let tab = vdupq_n_u8(b'\t');
    let newline = vdupq_n_u8(b'\n');
    let carriage = vdupq_n_u8(b'\r');

    let mut i = 0;
    while i + 16 <= len {
        let chunk = vld1q_u8(bytes.as_ptr().add(i));

        // Check for each whitespace character
        let space_mask = vceqq_u8(chunk, space);
        let tab_mask = vceqq_u8(chunk, tab);
        let newline_mask = vceqq_u8(chunk, newline);
        let carriage_mask = vceqq_u8(chunk, carriage);

        // Combine all whitespace masks
        let combined1 = vorrq_u8(space_mask, tab_mask);
        let combined2 = vorrq_u8(newline_mask, carriage_mask);
        let combined = vorrq_u8(combined1, combined2);

        // Check if any bytes matched
        let max_val = vmaxvq_u8(combined);
        if max_val != 0 {
            // Extract individual positions
            let mut result = [0u8; 16];
            vst1q_u8(result.as_mut_ptr(), combined);
            for (j, &r) in result.iter().enumerate() {
                if r == 0xFF {
                    positions.push(i + j);
                }
            }
        }

        i += 16;
    }

    // Handle remaining bytes with scalar
    for j in i..len {
        let b = bytes[j];
        if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
            positions.push(j);
        }
    }

    positions
}

/// Split text on whitespace using SIMD
pub fn split_whitespace_simd(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let positions = find_whitespace_simd(bytes);

    if positions.is_empty() {
        return if text.is_empty() { vec![] } else { vec![text] };
    }

    let mut result = Vec::with_capacity(positions.len() + 1);
    let mut start = 0;

    for &pos in &positions {
        if pos > start {
            result.push(&text[start..pos]);
        }
        start = pos + 1;
    }

    if start < text.len() {
        result.push(&text[start..]);
    }

    result
}

/// Check if byte slice contains only ASCII whitespace using SIMD
pub fn is_all_whitespace_simd(bytes: &[u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { is_all_whitespace_avx2(bytes) };
        }
    }

    // Scalar fallback
    bytes.iter().all(|&b| b == b' ' || b == b'\t' || b == b'\n' || b == b'\r')
}

/// AVX2 implementation of all-whitespace check
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn is_all_whitespace_avx2(bytes: &[u8]) -> bool {
    let len = bytes.len();

    let space = _mm256_set1_epi8(b' ' as i8);
    let tab = _mm256_set1_epi8(b'\t' as i8);
    let newline = _mm256_set1_epi8(b'\n' as i8);
    let carriage = _mm256_set1_epi8(b'\r' as i8);

    let mut i = 0;
    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);

        // Check for each whitespace character
        let space_mask = _mm256_cmpeq_epi8(chunk, space);
        let tab_mask = _mm256_cmpeq_epi8(chunk, tab);
        let newline_mask = _mm256_cmpeq_epi8(chunk, newline);
        let carriage_mask = _mm256_cmpeq_epi8(chunk, carriage);

        // Combine all whitespace masks
        let combined1 = _mm256_or_si256(space_mask, tab_mask);
        let combined2 = _mm256_or_si256(newline_mask, carriage_mask);
        let combined = _mm256_or_si256(combined1, combined2);

        // Check if all bytes are whitespace
        let mask = _mm256_movemask_epi8(combined) as u32;
        if mask != 0xFFFF_FFFF {
            return false;
        }

        i += 32;
    }

    // Handle remaining bytes with scalar
    for j in i..len {
        let b = bytes[j];
        if b != b' ' && b != b'\t' && b != b'\n' && b != b'\r' {
            return false;
        }
    }

    true
}

/// Check if a string is pure ASCII using SIMD (processes 32 bytes at a time with AVX2)
pub fn is_ascii_simd(bytes: &[u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { is_ascii_avx2(bytes) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { is_ascii_neon(bytes) };
    }

    // Scalar fallback
    bytes.iter().all(|&b| b < 128)
}

/// AVX2 ASCII check
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn is_ascii_avx2(bytes: &[u8]) -> bool {
    let len = bytes.len();
    let high_bit = _mm256_set1_epi8(0x80_u8 as i8);

    let mut i = 0;
    while i + 32 <= len {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);
        let result = _mm256_and_si256(chunk, high_bit);
        let mask = _mm256_movemask_epi8(result);
        if mask != 0 {
            return false;
        }
        i += 32;
    }

    // Check remainder
    for j in i..len {
        if bytes[j] >= 128 {
            return false;
        }
    }

    true
}

/// NEON ASCII check
#[cfg(target_arch = "aarch64")]
unsafe fn is_ascii_neon(bytes: &[u8]) -> bool {
    let len = bytes.len();
    let high_bit = vdupq_n_u8(0x80);

    let mut i = 0;
    while i + 16 <= len {
        let chunk = vld1q_u8(bytes.as_ptr().add(i));
        let result = vandq_u8(chunk, high_bit);
        let max_val = vmaxvq_u8(result);
        if max_val != 0 {
            return false;
        }
        i += 16;
    }

    // Check remainder
    for j in i..len {
        if bytes[j] >= 128 {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_whitespace_simd() {
        let text = b"hello world this is a test";
        let positions = find_whitespace_simd(text);
        assert_eq!(positions, vec![5, 11, 16, 19, 21]);
    }

    #[test]
    fn test_find_whitespace_scalar() {
        let text = b"hello world";
        let positions = find_whitespace_scalar(text);
        assert_eq!(positions, vec![5]);
    }

    #[test]
    fn test_split_whitespace_simd() {
        let text = "hello world this is a test";
        let parts = split_whitespace_simd(text);
        assert_eq!(parts, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_split_whitespace_empty() {
        let text = "";
        let parts = split_whitespace_simd(text);
        assert!(parts.is_empty());
    }

    #[test]
    fn test_split_whitespace_no_spaces() {
        let text = "hello";
        let parts = split_whitespace_simd(text);
        assert_eq!(parts, vec!["hello"]);
    }

    #[test]
    fn test_is_all_whitespace_simd() {
        assert!(is_all_whitespace_simd(b"   "));
        assert!(is_all_whitespace_simd(b" \t\n\r"));
        assert!(!is_all_whitespace_simd(b"hello"));
        assert!(!is_all_whitespace_simd(b" hello "));
    }

    #[test]
    fn test_is_ascii_simd() {
        assert!(is_ascii_simd(b"hello world"));
        assert!(is_ascii_simd(b""));
        assert!(is_ascii_simd(b"The quick brown fox"));
        assert!(!is_ascii_simd("café".as_bytes()));
        assert!(!is_ascii_simd("日本語".as_bytes()));
    }

    #[test]
    fn test_find_whitespace_with_tabs_newlines() {
        let text = b"hello\tworld\nthis\ris";
        let positions = find_whitespace_simd(text);
        assert_eq!(positions, vec![5, 11, 16]);
    }
}
