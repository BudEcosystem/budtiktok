//! ARM NEON SIMD Implementation (4.5.1, 4.5.2)
//!
//! NEON SIMD support for ARM processors (AArch64).
//! Provides character classification, UTF-8 validation, and other operations
//! optimized for ARM architecture.
//!
//! Key features:
//! - 128-bit vectors (16 bytes per iteration)
//! - vtbl for table lookups
//! - Parallel byte classification

/// Check if NEON is available
#[inline]
pub fn is_neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on AArch64
        true
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

// ============================================================================
// Character Classification (4.5.1)
// ============================================================================

/// Classify whitespace characters using NEON
///
/// Returns a bitmask where bit i is set if byte i is whitespace.
pub fn classify_whitespace(input: &[u8]) -> u16 {
    if input.is_empty() {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if input.len() >= 16 {
            return unsafe { classify_whitespace_neon(input) };
        }
    }

    classify_whitespace_scalar(input)
}

/// Scalar whitespace classification
fn classify_whitespace_scalar(input: &[u8]) -> u16 {
    let mut mask = 0u16;
    for (i, &byte) in input.iter().take(16).enumerate() {
        if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
            mask |= 1 << i;
        }
    }
    mask
}

/// NEON whitespace classification
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn classify_whitespace_neon(input: &[u8]) -> u16 {
    use std::arch::aarch64::*;

    let chunk = vld1q_u8(input.as_ptr());

    // Compare against whitespace characters
    let space = vdupq_n_u8(b' ');
    let tab = vdupq_n_u8(b'\t');
    let newline = vdupq_n_u8(b'\n');
    let cr = vdupq_n_u8(b'\r');

    let is_space = vceqq_u8(chunk, space);
    let is_tab = vceqq_u8(chunk, tab);
    let is_newline = vceqq_u8(chunk, newline);
    let is_cr = vceqq_u8(chunk, cr);

    // Combine results
    let combined = vorrq_u8(vorrq_u8(is_space, is_tab), vorrq_u8(is_newline, is_cr));

    // Extract bitmask
    neon_movemask(combined)
}

/// Extract bitmask from NEON comparison result
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_movemask(v: std::arch::aarch64::uint8x16_t) -> u16 {
    use std::arch::aarch64::*;

    // Use bit manipulation to extract one bit per byte
    // Shift each byte's MSB to position 0-15
    let shift = vld1q_u8([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].as_ptr());
    let bits = vshrq_n_u8(v, 7); // Get MSB of each byte
    let shifted = vshlq_u8(bits, vreinterpretq_s8_u8(shift));

    // Sum across vector to get the mask
    let low = vget_low_u8(shifted);
    let high = vget_high_u8(shifted);

    // Horizontal add to combine bits
    let sum_low = vaddv_u8(low);
    let sum_high = vaddv_u8(high);

    (sum_high as u16) << 8 | sum_low as u16
}

/// Classify ASCII alphanumeric characters
pub fn classify_alphanumeric(input: &[u8]) -> u16 {
    if input.is_empty() {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if input.len() >= 16 {
            return unsafe { classify_alphanumeric_neon(input) };
        }
    }

    classify_alphanumeric_scalar(input)
}

/// Scalar alphanumeric classification
fn classify_alphanumeric_scalar(input: &[u8]) -> u16 {
    let mut mask = 0u16;
    for (i, &byte) in input.iter().take(16).enumerate() {
        if byte.is_ascii_alphanumeric() {
            mask |= 1 << i;
        }
    }
    mask
}

/// NEON alphanumeric classification
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn classify_alphanumeric_neon(input: &[u8]) -> u16 {
    use std::arch::aarch64::*;

    let chunk = vld1q_u8(input.as_ptr());

    // Check digits: '0' (0x30) <= x <= '9' (0x39)
    let digit_lo = vdupq_n_u8(0x30);
    let digit_hi = vdupq_n_u8(0x39);
    let ge_digit_lo = vcgeq_u8(chunk, digit_lo);
    let le_digit_hi = vcleq_u8(chunk, digit_hi);
    let is_digit = vandq_u8(ge_digit_lo, le_digit_hi);

    // Check uppercase: 'A' (0x41) <= x <= 'Z' (0x5A)
    let upper_lo = vdupq_n_u8(0x41);
    let upper_hi = vdupq_n_u8(0x5A);
    let ge_upper_lo = vcgeq_u8(chunk, upper_lo);
    let le_upper_hi = vcleq_u8(chunk, upper_hi);
    let is_upper = vandq_u8(ge_upper_lo, le_upper_hi);

    // Check lowercase: 'a' (0x61) <= x <= 'z' (0x7A)
    let lower_lo = vdupq_n_u8(0x61);
    let lower_hi = vdupq_n_u8(0x7A);
    let ge_lower_lo = vcgeq_u8(chunk, lower_lo);
    let le_lower_hi = vcleq_u8(chunk, lower_hi);
    let is_lower = vandq_u8(ge_lower_lo, le_lower_hi);

    // Combine
    let combined = vorrq_u8(vorrq_u8(is_digit, is_upper), is_lower);

    neon_movemask(combined)
}

/// Find first non-ASCII byte position
pub fn find_non_ascii(input: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "aarch64")]
    {
        if input.len() >= 16 {
            return unsafe { find_non_ascii_neon(input) };
        }
    }

    find_non_ascii_scalar(input)
}

/// Scalar non-ASCII search
fn find_non_ascii_scalar(input: &[u8]) -> Option<usize> {
    for (i, &byte) in input.iter().enumerate() {
        if byte >= 0x80 {
            return Some(i);
        }
    }
    None
}

/// NEON non-ASCII search
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn find_non_ascii_neon(input: &[u8]) -> Option<usize> {
    use std::arch::aarch64::*;

    let mut i = 0;
    let ascii_max = vdupq_n_u8(0x7F);

    while i + 16 <= input.len() {
        let chunk = vld1q_u8(input.as_ptr().add(i));

        // Check if any byte > 0x7F
        let high_bits = vcgtq_u8(chunk, ascii_max);
        let mask = neon_movemask(high_bits);

        if mask != 0 {
            return Some(i + mask.trailing_zeros() as usize);
        }

        i += 16;
    }

    // Handle remaining bytes
    for j in i..input.len() {
        if input[j] >= 0x80 {
            return Some(j);
        }
    }

    None
}

// ============================================================================
// UTF-8 Validation (4.5.2)
// ============================================================================

/// Validate UTF-8 using NEON
pub fn validate_utf8_neon_available(input: &[u8]) -> Result<(), usize> {
    #[cfg(target_arch = "aarch64")]
    {
        if input.len() >= 16 {
            return unsafe { validate_utf8_neon(input) };
        }
    }

    validate_utf8_scalar(input)
}

/// Scalar UTF-8 validation
fn validate_utf8_scalar(input: &[u8]) -> Result<(), usize> {
    let mut i = 0;

    while i < input.len() {
        let byte = input[i];

        if byte < 0x80 {
            i += 1;
        } else if byte < 0xC2 {
            return Err(i);
        } else if byte < 0xE0 {
            if i + 1 >= input.len() || (input[i + 1] & 0xC0) != 0x80 {
                return Err(i);
            }
            i += 2;
        } else if byte < 0xF0 {
            if i + 2 >= input.len() {
                return Err(i);
            }
            let c1 = input[i + 1];
            let c2 = input[i + 2];
            if (c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 {
                return Err(i);
            }
            if byte == 0xE0 && c1 < 0xA0 {
                return Err(i);
            }
            if byte == 0xED && c1 >= 0xA0 {
                return Err(i);
            }
            i += 3;
        } else if byte < 0xF5 {
            if i + 3 >= input.len() {
                return Err(i);
            }
            let c1 = input[i + 1];
            let c2 = input[i + 2];
            let c3 = input[i + 3];
            if (c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80 {
                return Err(i);
            }
            if byte == 0xF0 && c1 < 0x90 {
                return Err(i);
            }
            if byte == 0xF4 && c1 >= 0x90 {
                return Err(i);
            }
            i += 4;
        } else {
            return Err(i);
        }
    }

    Ok(())
}

/// NEON UTF-8 validation
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn validate_utf8_neon(input: &[u8]) -> Result<(), usize> {
    use std::arch::aarch64::*;

    let mut i = 0;
    let ascii_max = vdupq_n_u8(0x7F);

    // Fast path: scan for all-ASCII chunks
    while i + 16 <= input.len() {
        let chunk = vld1q_u8(input.as_ptr().add(i));

        // Check if all ASCII
        let high_bits = vcgtq_u8(chunk, ascii_max);
        let mask = neon_movemask(high_bits);

        if mask == 0 {
            // All ASCII, continue
            i += 16;
        } else {
            // Found non-ASCII, validate with scalar
            break;
        }
    }

    // Validate remaining bytes with scalar
    validate_utf8_scalar(&input[i..]).map_err(|pos| i + pos)
}

// ============================================================================
// Count Operations
// ============================================================================

/// Count code points using NEON
pub fn count_code_points_neon_available(input: &[u8]) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        if input.len() >= 16 {
            return unsafe { count_code_points_neon(input) };
        }
    }

    count_code_points_scalar(input)
}

/// Scalar code point counting
fn count_code_points_scalar(input: &[u8]) -> usize {
    input.iter().filter(|&&b| (b & 0xC0) != 0x80).count()
}

/// NEON code point counting
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn count_code_points_neon(input: &[u8]) -> usize {
    use std::arch::aarch64::*;

    let mut count = 0usize;
    let mut i = 0;

    let cont_mask = vdupq_n_u8(0xC0);
    let cont_pattern = vdupq_n_u8(0x80);

    while i + 16 <= input.len() {
        let chunk = vld1q_u8(input.as_ptr().add(i));

        // Find continuation bytes (10xxxxxx)
        let masked = vandq_u8(chunk, cont_mask);
        let is_cont = vceqq_u8(masked, cont_pattern);
        let cont_bits = neon_movemask(is_cont);

        // Count non-continuation bytes
        count += 16 - cont_bits.count_ones() as usize;
        i += 16;
    }

    // Handle remaining bytes
    count += count_code_points_scalar(&input[i..]);

    count
}

/// Count whitespace characters
pub fn count_whitespace(input: &[u8]) -> usize {
    let mut total = 0;
    let mut i = 0;

    while i + 16 <= input.len() {
        let mask = classify_whitespace(&input[i..]);
        total += mask.count_ones() as usize;
        i += 16;
    }

    // Handle remaining bytes
    for &byte in &input[i..] {
        if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
            total += 1;
        }
    }

    total
}

// ============================================================================
// Lowercase Conversion
// ============================================================================

/// Convert ASCII uppercase to lowercase using NEON
pub fn to_lowercase_ascii(input: &mut [u8]) {
    #[cfg(target_arch = "aarch64")]
    {
        if input.len() >= 16 {
            unsafe { to_lowercase_ascii_neon(input) };
            return;
        }
    }

    to_lowercase_ascii_scalar(input);
}

/// Scalar lowercase conversion
fn to_lowercase_ascii_scalar(input: &mut [u8]) {
    for byte in input {
        if *byte >= b'A' && *byte <= b'Z' {
            *byte += 32;
        }
    }
}

/// NEON lowercase conversion
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn to_lowercase_ascii_neon(input: &mut [u8]) {
    use std::arch::aarch64::*;

    let mut i = 0;
    let upper_a = vdupq_n_u8(b'A');
    let upper_z = vdupq_n_u8(b'Z');
    let diff = vdupq_n_u8(32);

    while i + 16 <= input.len() {
        let chunk = vld1q_u8(input.as_ptr().add(i));

        // Find uppercase letters
        let ge_a = vcgeq_u8(chunk, upper_a);
        let le_z = vcleq_u8(chunk, upper_z);
        let is_upper = vandq_u8(ge_a, le_z);

        // Add 32 only to uppercase letters
        let add_mask = vandq_u8(is_upper, diff);
        let result = vaddq_u8(chunk, add_mask);

        vst1q_u8(input.as_mut_ptr().add(i), result);
        i += 16;
    }

    // Handle remaining bytes
    to_lowercase_ascii_scalar(&mut input[i..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_neon_available() {
        let _ = is_neon_available();
    }

    #[test]
    fn test_classify_whitespace() {
        let input = b"hello world test";
        let mask = classify_whitespace(input);
        // Space at position 5 and 11
        assert!(mask & (1 << 5) != 0);
        assert!(mask & (1 << 11) != 0);
    }

    #[test]
    fn test_classify_whitespace_tabs() {
        let input = b"a\tb\nc\rd";
        let mask = classify_whitespace(input);
        assert!(mask & (1 << 1) != 0); // tab
        assert!(mask & (1 << 3) != 0); // newline
        assert!(mask & (1 << 5) != 0); // carriage return
    }

    #[test]
    fn test_classify_alphanumeric() {
        let input = b"abc123!@#XYZ";
        let mask = classify_alphanumeric(input);
        // a,b,c,1,2,3,X,Y,Z are alphanumeric
        assert!(mask & (1 << 0) != 0); // a
        assert!(mask & (1 << 1) != 0); // b
        assert!(mask & (1 << 2) != 0); // c
        assert!(mask & (1 << 3) != 0); // 1
        assert!(mask & (1 << 4) != 0); // 2
        assert!(mask & (1 << 5) != 0); // 3
        assert!(mask & (1 << 6) == 0); // !
        assert!(mask & (1 << 9) != 0); // X
    }

    #[test]
    fn test_find_non_ascii() {
        assert_eq!(find_non_ascii(b"hello"), None);
        assert_eq!(find_non_ascii(b"hello\x80world"), Some(5));
        assert_eq!(find_non_ascii("café".as_bytes()), Some(3)); // é starts at byte 3
    }

    #[test]
    fn test_validate_utf8_valid() {
        assert!(validate_utf8_neon_available(b"hello").is_ok());
        assert!(validate_utf8_neon_available("日本語".as_bytes()).is_ok());
        assert!(validate_utf8_neon_available("🎉".as_bytes()).is_ok());
    }

    #[test]
    fn test_validate_utf8_invalid() {
        assert!(validate_utf8_neon_available(&[0x80]).is_err());
        assert!(validate_utf8_neon_available(&[0xC0, 0x80]).is_err());
    }

    #[test]
    fn test_count_code_points() {
        assert_eq!(count_code_points_neon_available(b"hello"), 5);
        assert_eq!(count_code_points_neon_available("日本語".as_bytes()), 3);
        assert_eq!(count_code_points_neon_available("🎉".as_bytes()), 1);
    }

    #[test]
    fn test_count_whitespace() {
        assert_eq!(count_whitespace(b"hello world"), 1);
        assert_eq!(count_whitespace(b"a b c d"), 3);
        assert_eq!(count_whitespace(b"nospace"), 0);
    }

    #[test]
    fn test_to_lowercase() {
        let mut input = b"Hello World ABC".to_vec();
        to_lowercase_ascii(&mut input);
        assert_eq!(&input, b"hello world abc");
    }

    #[test]
    fn test_to_lowercase_mixed() {
        let mut input = b"HeLLo123WoRLD".to_vec();
        to_lowercase_ascii(&mut input);
        assert_eq!(&input, b"hello123world");
    }

    #[test]
    fn test_long_input() {
        // Test with input longer than 16 bytes
        let input = b"The quick brown fox jumps over the lazy dog";
        assert_eq!(count_whitespace(input), 8);
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(classify_whitespace(b""), 0);
        assert_eq!(find_non_ascii(b""), None);
        assert!(validate_utf8_neon_available(b"").is_ok());
        assert_eq!(count_code_points_neon_available(b""), 0);
    }
}
