//! SSE4.2 Implementation (4.4.2)
//!
//! SSE4.2 provides PCMPISTR* instructions for string comparison and range checks.
//! These are particularly useful for Unicode character classification.
//!
//! Key features:
//! - PCMPISTRM with ranges mode for character class detection
//! - Efficient detection of ASCII, whitespace, punctuation, etc.
//! - Processes 16 bytes per iteration

/// Check if SSE4.2 is available at runtime
#[inline]
pub fn is_sse42_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("sse4.2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Result of range check operation
#[derive(Debug, Clone, Copy)]
pub struct RangeCheckResult {
    /// Mask of matching bytes (bit i = 1 if byte i matches)
    pub mask: u16,
    /// Number of matching bytes
    pub count: u32,
}

/// Character class definitions as ranges
pub mod char_classes {
    /// ASCII digits: 0x30-0x39 ('0'-'9')
    pub const DIGITS: &[u8] = &[0x30, 0x39];

    /// ASCII lowercase: 0x61-0x7A ('a'-'z')
    pub const LOWERCASE: &[u8] = &[0x61, 0x7A];

    /// ASCII uppercase: 0x41-0x5A ('A'-'Z')
    pub const UPPERCASE: &[u8] = &[0x41, 0x5A];

    /// ASCII letters: 'A'-'Z' and 'a'-'z'
    pub const LETTERS: &[u8] = &[0x41, 0x5A, 0x61, 0x7A];

    /// ASCII alphanumeric: '0'-'9', 'A'-'Z', 'a'-'z'
    pub const ALPHANUMERIC: &[u8] = &[0x30, 0x39, 0x41, 0x5A, 0x61, 0x7A];

    /// ASCII whitespace: space (0x20), tab (0x09), newline (0x0A), carriage return (0x0D)
    pub const WHITESPACE: &[u8] = &[0x09, 0x0D, 0x20, 0x20];

    /// ASCII control characters: 0x00-0x1F
    pub const CONTROL: &[u8] = &[0x00, 0x1F];

    /// ASCII printable: 0x20-0x7E
    pub const PRINTABLE: &[u8] = &[0x20, 0x7E];

    /// Non-ASCII (high bytes): 0x80-0xFF
    pub const NON_ASCII: &[u8] = &[0x80, 0xFF];

    /// ASCII punctuation (common subset)
    pub const PUNCTUATION: &[u8] = &[
        0x21, 0x2F, // !"#$%&'()*+,-./
        0x3A, 0x40, // :;<=>?@
        0x5B, 0x60, // [\]^_`
        0x7B, 0x7E, // {|}~
    ];
}

/// Check if bytes match any of the given ranges using SSE4.2
///
/// Each pair of bytes in `ranges` defines a range [low, high].
/// Returns mask of bytes that fall within any range.
pub fn check_ranges(input: &[u8], ranges: &[u8]) -> RangeCheckResult {
    if input.is_empty() {
        return RangeCheckResult { mask: 0, count: 0 };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.2") && input.len() >= 16 {
            return unsafe { check_ranges_sse42(input, ranges) };
        }
    }

    check_ranges_scalar(input, ranges)
}

/// Scalar implementation of range check
fn check_ranges_scalar(input: &[u8], ranges: &[u8]) -> RangeCheckResult {
    let mut mask = 0u16;
    let mut count = 0u32;

    let check_len = input.len().min(16);

    for (i, &byte) in input.iter().take(check_len).enumerate() {
        let mut matches = false;
        for pair in ranges.chunks_exact(2) {
            if byte >= pair[0] && byte <= pair[1] {
                matches = true;
                break;
            }
        }
        if matches {
            mask |= 1 << i;
            count += 1;
        }
    }

    RangeCheckResult { mask, count }
}

/// SSE4.2 implementation using PCMPISTRM
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn check_ranges_sse42(input: &[u8], ranges: &[u8]) -> RangeCheckResult {
    use std::arch::x86_64::*;

    // Load input (up to 16 bytes)
    let input_vec = if input.len() >= 16 {
        _mm_loadu_si128(input.as_ptr() as *const __m128i)
    } else {
        // Pad with zeros for shorter input
        let mut buf = [0u8; 16];
        buf[..input.len()].copy_from_slice(input);
        _mm_loadu_si128(buf.as_ptr() as *const __m128i)
    };

    // Load ranges (up to 16 bytes = 8 ranges)
    let ranges_vec = if ranges.len() >= 16 {
        _mm_loadu_si128(ranges.as_ptr() as *const __m128i)
    } else {
        let mut buf = [0u8; 16];
        let copy_len = ranges.len().min(16);
        buf[..copy_len].copy_from_slice(&ranges[..copy_len]);
        _mm_loadu_si128(buf.as_ptr() as *const __m128i)
    };

    // PCMPISTRM with ranges mode
    // _SIDD_UBYTE_OPS: unsigned byte comparison
    // _SIDD_CMP_RANGES: compare against ranges (pairs)
    // _SIDD_BIT_MASK: return bit mask
    // Mode must be a compile-time constant for intrinsic
    const MODE: i32 = _SIDD_UBYTE_OPS | _SIDD_CMP_RANGES | _SIDD_BIT_MASK;

    // Note: PCMPISTRM returns result in XMM0, we use _mm_cmpistrm to get it
    let result = _mm_cmpistrm(ranges_vec, input_vec, MODE);

    // Extract the mask from the lower 16 bits
    let mask_full = _mm_extract_epi32(result, 0) as u32;
    let mask = (mask_full & 0xFFFF) as u16;

    // Limit mask to actual input length
    let valid_mask = if input.len() >= 16 {
        mask
    } else {
        mask & ((1u16 << input.len()) - 1)
    };

    let count = valid_mask.count_ones();

    RangeCheckResult {
        mask: valid_mask,
        count,
    }
}

/// Find positions of bytes matching ranges
pub fn find_matching_positions(input: &[u8], ranges: &[u8]) -> Vec<usize> {
    let mut positions = Vec::new();

    let mut offset = 0;
    while offset < input.len() {
        let chunk = &input[offset..];
        let result = check_ranges(chunk, ranges);

        // Extract positions from mask
        let mut mask = result.mask;
        let mut bit_pos = 0;
        while mask != 0 {
            if mask & 1 != 0 {
                let pos = offset + bit_pos;
                if pos < input.len() {
                    positions.push(pos);
                }
            }
            mask >>= 1;
            bit_pos += 1;
        }

        offset += 16;
    }

    positions
}

/// Count bytes matching ranges in entire input
pub fn count_matching_bytes(input: &[u8], ranges: &[u8]) -> usize {
    let mut total = 0;

    let mut offset = 0;
    while offset < input.len() {
        let chunk = &input[offset..];
        let chunk_len = chunk.len().min(16);
        let result = check_ranges(chunk, ranges);

        // Only count valid positions
        let valid_mask = if chunk_len < 16 {
            result.mask & ((1u16 << chunk_len) - 1)
        } else {
            result.mask
        };

        total += valid_mask.count_ones() as usize;
        offset += 16;
    }

    total
}

/// Check if all bytes in input match the given ranges
pub fn all_match_ranges(input: &[u8], ranges: &[u8]) -> bool {
    let mut offset = 0;
    while offset < input.len() {
        let chunk = &input[offset..];
        let chunk_len = chunk.len().min(16);
        let result = check_ranges(chunk, ranges);

        // Check if all bytes in this chunk match
        let expected_mask = if chunk_len < 16 {
            (1u16 << chunk_len) - 1
        } else {
            0xFFFF
        };

        if result.mask != expected_mask {
            return false;
        }

        offset += 16;
    }

    true
}

/// Check if any byte in input matches the given ranges
pub fn any_match_ranges(input: &[u8], ranges: &[u8]) -> bool {
    let mut offset = 0;
    while offset < input.len() {
        let chunk = &input[offset..];
        let result = check_ranges(chunk, ranges);
        if result.mask != 0 {
            return true;
        }
        offset += 16;
    }

    false
}

/// Find first position of a byte matching ranges
pub fn find_first_matching(input: &[u8], ranges: &[u8]) -> Option<usize> {
    let mut offset = 0;
    while offset < input.len() {
        let chunk = &input[offset..];
        let result = check_ranges(chunk, ranges);
        if result.mask != 0 {
            // Find lowest set bit
            let bit_pos = result.mask.trailing_zeros() as usize;
            let pos = offset + bit_pos;
            if pos < input.len() {
                return Some(pos);
            }
        }
        offset += 16;
    }

    None
}

/// Check if input contains only ASCII characters
#[inline]
pub fn is_all_ascii(input: &[u8]) -> bool {
    // ASCII range is 0x00-0x7F, so we check for absence of high bytes
    !any_match_ranges(input, char_classes::NON_ASCII)
}

/// Check if input contains only ASCII alphanumeric characters
#[inline]
pub fn is_all_alphanumeric(input: &[u8]) -> bool {
    all_match_ranges(input, char_classes::ALPHANUMERIC)
}

/// Find first non-ASCII byte
#[inline]
pub fn find_first_non_ascii(input: &[u8]) -> Option<usize> {
    find_first_matching(input, char_classes::NON_ASCII)
}

/// Find first whitespace character
#[inline]
pub fn find_first_whitespace(input: &[u8]) -> Option<usize> {
    find_first_matching(input, char_classes::WHITESPACE)
}

/// Count whitespace characters
#[inline]
pub fn count_whitespace(input: &[u8]) -> usize {
    count_matching_bytes(input, char_classes::WHITESPACE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_sse42_available() {
        // Just verify the function doesn't crash
        let _ = is_sse42_available();
    }

    #[test]
    fn test_check_ranges_digits() {
        let input = b"abc123def";
        let result = check_ranges(input, char_classes::DIGITS);

        // Digits are at positions 3, 4, 5
        assert!(result.mask & (1 << 3) != 0);
        assert!(result.mask & (1 << 4) != 0);
        assert!(result.mask & (1 << 5) != 0);
        assert_eq!(result.count, 3);
    }

    #[test]
    fn test_check_ranges_lowercase() {
        let input = b"ABCdef";
        let result = check_ranges(input, char_classes::LOWERCASE);

        // Lowercase at positions 3, 4, 5
        assert!(result.mask & (1 << 3) != 0);
        assert!(result.mask & (1 << 4) != 0);
        assert!(result.mask & (1 << 5) != 0);
        assert_eq!(result.count, 3);
    }

    #[test]
    fn test_check_ranges_whitespace() {
        let input = b"hello world";
        let result = check_ranges(input, char_classes::WHITESPACE);

        // Space at position 5
        assert!(result.mask & (1 << 5) != 0);
        assert_eq!(result.count, 1);
    }

    #[test]
    fn test_check_ranges_empty() {
        let result = check_ranges(b"", char_classes::DIGITS);
        assert_eq!(result.mask, 0);
        assert_eq!(result.count, 0);
    }

    #[test]
    fn test_find_matching_positions() {
        let input = b"a1b2c3d4";
        let positions = find_matching_positions(input, char_classes::DIGITS);

        assert_eq!(positions, vec![1, 3, 5, 7]);
    }

    #[test]
    fn test_count_matching_bytes() {
        let input = b"hello world 123";
        let count = count_matching_bytes(input, char_classes::DIGITS);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_all_match_ranges() {
        assert!(all_match_ranges(b"abc", char_classes::LOWERCASE));
        assert!(!all_match_ranges(b"aBc", char_classes::LOWERCASE));
        assert!(all_match_ranges(b"123", char_classes::DIGITS));
    }

    #[test]
    fn test_any_match_ranges() {
        assert!(any_match_ranges(b"hello", char_classes::LOWERCASE));
        assert!(!any_match_ranges(b"12345", char_classes::LOWERCASE));
    }

    #[test]
    fn test_find_first_matching() {
        let input = b"HELLO world";
        let pos = find_first_matching(input, char_classes::LOWERCASE);
        assert_eq!(pos, Some(6)); // 'w'
    }

    #[test]
    fn test_is_all_ascii() {
        assert!(is_all_ascii(b"hello world"));
        assert!(!is_all_ascii("héllo".as_bytes()));
    }

    #[test]
    fn test_is_all_alphanumeric() {
        assert!(is_all_alphanumeric(b"abc123"));
        assert!(!is_all_alphanumeric(b"abc 123"));
    }

    #[test]
    fn test_find_first_non_ascii() {
        assert_eq!(find_first_non_ascii(b"hello"), None);
        assert_eq!(find_first_non_ascii("héllo".as_bytes()), Some(1));
    }

    #[test]
    fn test_find_first_whitespace() {
        assert_eq!(find_first_whitespace(b"hello world"), Some(5));
        assert_eq!(find_first_whitespace(b"helloworld"), None);
    }

    #[test]
    fn test_count_whitespace() {
        assert_eq!(count_whitespace(b"hello world"), 1);
        assert_eq!(count_whitespace(b"a b c d"), 3);
        assert_eq!(count_whitespace(b"nospace"), 0);
    }

    #[test]
    fn test_long_input() {
        // Test with input longer than 16 bytes
        let input = b"The quick brown fox jumps over the lazy dog 12345";
        let count = count_matching_bytes(input, char_classes::DIGITS);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_short_input() {
        // Test with input shorter than 16 bytes
        let input = b"abc";
        let result = check_ranges(input, char_classes::LOWERCASE);
        assert_eq!(result.count, 3);
    }

    #[test]
    fn test_alphanumeric_count() {
        let input = b"Hello, World! 123";
        let count = count_matching_bytes(input, char_classes::ALPHANUMERIC);
        // 'H', 'e', 'l', 'l', 'o', 'W', 'o', 'r', 'l', 'd', '1', '2', '3' = 13
        assert_eq!(count, 13);
    }

    #[test]
    fn test_multiple_ranges() {
        // Test with letters range (uppercase + lowercase)
        let input = b"Hello123";
        let count = count_matching_bytes(input, char_classes::LETTERS);
        assert_eq!(count, 5); // H, e, l, l, o
    }
}
