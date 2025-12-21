//! AVX-512 Implementation (4.2.1, 4.2.2)
//!
//! AVX-512 SIMD support for high-performance character processing.
//! Processes 64 bytes per iteration using 512-bit vectors.
//!
//! Key features:
//! - 64 bytes per iteration (vs 32 for AVX2)
//! - Native mask registers for comparisons
//! - Efficient bit manipulation with mask operations
//!
//! Note: AVX-512 intrinsics require the `nightly` feature flag as they use
//! unstable Rust features. On stable Rust, scalar fallbacks are used.

/// Check if AVX-512 (F and BW) is available
#[inline]
pub fn is_avx512_available() -> bool {
    #[cfg(all(target_arch = "x86_64", feature = "nightly"))]
    {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "nightly")))]
    {
        false
    }
}

// ============================================================================
// Character Classification (4.2.1)
// ============================================================================

/// Classify whitespace characters using AVX-512
///
/// Returns a 64-bit mask where bit i is set if byte i is whitespace.
pub fn classify_whitespace(input: &[u8]) -> u64 {
    if input.is_empty() {
        return 0;
    }

    #[cfg(all(target_arch = "x86_64", feature = "nightly"))]
    {
        if is_avx512_available() && input.len() >= 64 {
            return unsafe { classify_whitespace_avx512(input) };
        }
    }

    classify_whitespace_scalar(input)
}

/// Scalar whitespace classification
fn classify_whitespace_scalar(input: &[u8]) -> u64 {
    let mut mask = 0u64;
    for (i, &byte) in input.iter().take(64).enumerate() {
        if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
            mask |= 1u64 << i;
        }
    }
    mask
}

/// AVX-512 whitespace classification
///
/// # Safety
/// - Input must have at least 64 bytes
/// - AVX-512F and AVX-512BW must be available (checked by caller)
#[cfg(all(target_arch = "x86_64", feature = "nightly"))]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn classify_whitespace_avx512(input: &[u8]) -> u64 {
    use std::arch::x86_64::*;

    // Safety check: input must be at least 64 bytes
    debug_assert!(
        input.len() >= 64,
        "AVX-512 requires at least 64 bytes, got {}",
        input.len()
    );

    let chunk = _mm512_loadu_si512(input.as_ptr() as *const _);

    let space = _mm512_set1_epi8(b' ' as i8);
    let tab = _mm512_set1_epi8(b'\t' as i8);
    let newline = _mm512_set1_epi8(b'\n' as i8);
    let cr = _mm512_set1_epi8(b'\r' as i8);

    let space_mask = _mm512_cmpeq_epi8_mask(chunk, space);
    let tab_mask = _mm512_cmpeq_epi8_mask(chunk, tab);
    let newline_mask = _mm512_cmpeq_epi8_mask(chunk, newline);
    let cr_mask = _mm512_cmpeq_epi8_mask(chunk, cr);

    space_mask | tab_mask | newline_mask | cr_mask
}

/// Classify ASCII alphanumeric characters
pub fn classify_alphanumeric(input: &[u8]) -> u64 {
    if input.is_empty() {
        return 0;
    }

    #[cfg(all(target_arch = "x86_64", feature = "nightly"))]
    {
        if is_avx512_available() && input.len() >= 64 {
            return unsafe { classify_alphanumeric_avx512(input) };
        }
    }

    classify_alphanumeric_scalar(input)
}

/// Scalar alphanumeric classification
fn classify_alphanumeric_scalar(input: &[u8]) -> u64 {
    let mut mask = 0u64;
    for (i, &byte) in input.iter().take(64).enumerate() {
        if byte.is_ascii_alphanumeric() {
            mask |= 1u64 << i;
        }
    }
    mask
}

/// AVX-512 alphanumeric classification
///
/// # Safety
/// - Input must have at least 64 bytes
/// - AVX-512F and AVX-512BW must be available (checked by caller)
#[cfg(all(target_arch = "x86_64", feature = "nightly"))]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn classify_alphanumeric_avx512(input: &[u8]) -> u64 {
    use std::arch::x86_64::*;

    debug_assert!(
        input.len() >= 64,
        "AVX-512 requires at least 64 bytes, got {}",
        input.len()
    );

    let chunk = _mm512_loadu_si512(input.as_ptr() as *const _);

    // Check digits: '0' (0x30) <= x <= '9' (0x39)
    let digit_lo = _mm512_set1_epi8(0x30);
    let digit_hi = _mm512_set1_epi8(0x39);
    let ge_digit_lo = _mm512_cmpge_epi8_mask(chunk, digit_lo);
    let le_digit_hi = _mm512_cmple_epi8_mask(chunk, digit_hi);
    let is_digit = ge_digit_lo & le_digit_hi;

    // Check uppercase: 'A' (0x41) <= x <= 'Z' (0x5A)
    let upper_lo = _mm512_set1_epi8(0x41);
    let upper_hi = _mm512_set1_epi8(0x5A);
    let ge_upper_lo = _mm512_cmpge_epi8_mask(chunk, upper_lo);
    let le_upper_hi = _mm512_cmple_epi8_mask(chunk, upper_hi);
    let is_upper = ge_upper_lo & le_upper_hi;

    // Check lowercase: 'a' (0x61) <= x <= 'z' (0x7A)
    let lower_lo = _mm512_set1_epi8(0x61);
    let lower_hi = _mm512_set1_epi8(0x7A);
    let ge_lower_lo = _mm512_cmpge_epi8_mask(chunk, lower_lo);
    let le_lower_hi = _mm512_cmple_epi8_mask(chunk, lower_hi);
    let is_lower = ge_lower_lo & le_lower_hi;

    is_digit | is_upper | is_lower
}

/// Classify punctuation characters
pub fn classify_punctuation(input: &[u8]) -> u64 {
    if input.is_empty() {
        return 0;
    }

    #[cfg(all(target_arch = "x86_64", feature = "nightly"))]
    {
        if is_avx512_available() && input.len() >= 64 {
            return unsafe { classify_punctuation_avx512(input) };
        }
    }

    classify_punctuation_scalar(input)
}

/// Scalar punctuation classification
fn classify_punctuation_scalar(input: &[u8]) -> u64 {
    let mut mask = 0u64;
    for (i, &byte) in input.iter().take(64).enumerate() {
        if byte.is_ascii_punctuation() {
            mask |= 1u64 << i;
        }
    }
    mask
}

/// AVX-512 punctuation classification
#[cfg(all(target_arch = "x86_64", feature = "nightly"))]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn classify_punctuation_avx512(input: &[u8]) -> u64 {
    use std::arch::x86_64::*;

    let chunk = _mm512_loadu_si512(input.as_ptr() as *const _);

    // Punctuation ranges: 0x21-0x2F, 0x3A-0x40, 0x5B-0x60, 0x7B-0x7E

    // Range 1: 0x21-0x2F (!"#$%&'()*+,-./)
    let r1_lo = _mm512_set1_epi8(0x21);
    let r1_hi = _mm512_set1_epi8(0x2F);
    let ge_r1_lo = _mm512_cmpge_epi8_mask(chunk, r1_lo);
    let le_r1_hi = _mm512_cmple_epi8_mask(chunk, r1_hi);
    let in_r1 = ge_r1_lo & le_r1_hi;

    // Range 2: 0x3A-0x40 (:;<=>?@)
    let r2_lo = _mm512_set1_epi8(0x3A);
    let r2_hi = _mm512_set1_epi8(0x40);
    let ge_r2_lo = _mm512_cmpge_epi8_mask(chunk, r2_lo);
    let le_r2_hi = _mm512_cmple_epi8_mask(chunk, r2_hi);
    let in_r2 = ge_r2_lo & le_r2_hi;

    // Range 3: 0x5B-0x60 ([\]^_`)
    let r3_lo = _mm512_set1_epi8(0x5B);
    let r3_hi = _mm512_set1_epi8(0x60);
    let ge_r3_lo = _mm512_cmpge_epi8_mask(chunk, r3_lo);
    let le_r3_hi = _mm512_cmple_epi8_mask(chunk, r3_hi);
    let in_r3 = ge_r3_lo & le_r3_hi;

    // Range 4: 0x7B-0x7E ({|}~)
    let r4_lo = _mm512_set1_epi8(0x7B);
    let r4_hi = _mm512_set1_epi8(0x7E);
    let ge_r4_lo = _mm512_cmpge_epi8_mask(chunk, r4_lo);
    let le_r4_hi = _mm512_cmple_epi8_mask(chunk, r4_hi);
    let in_r4 = ge_r4_lo & le_r4_hi;

    in_r1 | in_r2 | in_r3 | in_r4
}

/// Find non-ASCII bytes
pub fn classify_non_ascii(input: &[u8]) -> u64 {
    if input.is_empty() {
        return 0;
    }

    #[cfg(all(target_arch = "x86_64", feature = "nightly"))]
    {
        if is_avx512_available() && input.len() >= 64 {
            return unsafe { classify_non_ascii_avx512(input) };
        }
    }

    classify_non_ascii_scalar(input)
}

/// Scalar non-ASCII classification
fn classify_non_ascii_scalar(input: &[u8]) -> u64 {
    let mut mask = 0u64;
    for (i, &byte) in input.iter().take(64).enumerate() {
        if byte >= 0x80 {
            mask |= 1u64 << i;
        }
    }
    mask
}

/// AVX-512 non-ASCII classification
#[cfg(all(target_arch = "x86_64", feature = "nightly"))]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn classify_non_ascii_avx512(input: &[u8]) -> u64 {
    use std::arch::x86_64::*;

    let chunk = _mm512_loadu_si512(input.as_ptr() as *const _);
    let ascii_max = _mm512_set1_epi8(0x7F);

    // Bytes > 0x7F are non-ASCII (using unsigned comparison)
    _mm512_cmpgt_epi8_mask(chunk, ascii_max)
}

// ============================================================================
// Pre-tokenization (4.2.2)
// ============================================================================

/// Pre-tokenize input using AVX-512
///
/// Returns byte ranges (start, end) for each token.
/// Tokens are separated by whitespace.
pub fn pretokenize(input: &[u8]) -> Vec<(usize, usize)> {
    if input.is_empty() {
        return Vec::new();
    }

    #[cfg(all(target_arch = "x86_64", feature = "nightly"))]
    {
        if is_avx512_available() && input.len() >= 64 {
            return unsafe { pretokenize_avx512(input) };
        }
    }

    pretokenize_scalar(input)
}

/// Scalar pre-tokenization
fn pretokenize_scalar(input: &[u8]) -> Vec<(usize, usize)> {
    let mut tokens = Vec::new();
    let mut token_start: Option<usize> = None;

    for (i, &byte) in input.iter().enumerate() {
        let is_whitespace = byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r';

        match (token_start, is_whitespace) {
            (Some(start), true) => {
                // End of token
                tokens.push((start, i));
                token_start = None;
            }
            (None, false) => {
                // Start of token
                token_start = Some(i);
            }
            _ => {}
        }
    }

    // Handle trailing token
    if let Some(start) = token_start {
        tokens.push((start, input.len()));
    }

    tokens
}

/// AVX-512 pre-tokenization
#[cfg(all(target_arch = "x86_64", feature = "nightly"))]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn pretokenize_avx512(input: &[u8]) -> Vec<(usize, usize)> {
    let mut tokens = Vec::new();
    let mut i = 0;
    let mut in_token = false;
    let mut token_start = 0usize;

    // Process 64 bytes at a time
    while i + 64 <= input.len() {
        let ws_mask = classify_whitespace_avx512(&input[i..]);

        // Process each bit in the mask
        let mut bit_pos = 0;
        let mut remaining_mask = ws_mask;

        while bit_pos < 64 {
            let is_ws = (remaining_mask & 1) != 0;
            let abs_pos = i + bit_pos;

            if in_token && is_ws {
                // End of token
                tokens.push((token_start, abs_pos));
                in_token = false;
            } else if !in_token && !is_ws {
                // Start of token
                token_start = abs_pos;
                in_token = true;
            }

            remaining_mask >>= 1;
            bit_pos += 1;
        }

        i += 64;
    }

    // Handle remaining bytes with scalar
    while i < input.len() {
        let byte = input[i];
        let is_ws = byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r';

        if in_token && is_ws {
            tokens.push((token_start, i));
            in_token = false;
        } else if !in_token && !is_ws {
            token_start = i;
            in_token = true;
        }

        i += 1;
    }

    // Handle trailing token
    if in_token {
        tokens.push((token_start, input.len()));
    }

    tokens
}

/// Find word boundaries (whitespace positions)
///
/// Returns positions where whitespace occurs.
pub fn find_word_boundaries(input: &[u8]) -> Vec<usize> {
    let mut positions = Vec::new();
    let mut i = 0;

    while i + 64 <= input.len() {
        let mask = classify_whitespace(&input[i..]);
        extract_positions_from_mask(mask, i, &mut positions);
        i += 64;
    }

    // Handle remaining bytes
    while i < input.len() {
        let byte = input[i];
        if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
            positions.push(i);
        }
        i += 1;
    }

    positions
}

/// Extract positions from a 64-bit mask
#[inline]
fn extract_positions_from_mask(mut mask: u64, base: usize, positions: &mut Vec<usize>) {
    while mask != 0 {
        let pos = mask.trailing_zeros() as usize;
        positions.push(base + pos);
        mask &= mask - 1; // Clear lowest set bit
    }
}

/// Count whitespace bytes in the entire input
pub fn count_whitespace(input: &[u8]) -> usize {
    let mut total = 0;
    let mut i = 0;

    while i + 64 <= input.len() {
        let mask = classify_whitespace(&input[i..]);
        total += mask.count_ones() as usize;
        i += 64;
    }

    // Handle remaining bytes
    for &byte in &input[i..] {
        if byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r' {
            total += 1;
        }
    }

    total
}

/// Check if all bytes are ASCII
pub fn is_all_ascii(input: &[u8]) -> bool {
    let mut i = 0;

    while i + 64 <= input.len() {
        let mask = classify_non_ascii(&input[i..]);
        if mask != 0 {
            return false;
        }
        i += 64;
    }

    // Check remaining bytes
    for &byte in &input[i..] {
        if byte >= 0x80 {
            return false;
        }
    }

    true
}

/// Find the first non-ASCII byte
pub fn find_first_non_ascii(input: &[u8]) -> Option<usize> {
    let mut i = 0;

    while i + 64 <= input.len() {
        let mask = classify_non_ascii(&input[i..]);
        if mask != 0 {
            return Some(i + mask.trailing_zeros() as usize);
        }
        i += 64;
    }

    // Check remaining bytes
    for j in i..input.len() {
        if input[j] >= 0x80 {
            return Some(j);
        }
    }

    None
}

/// Convert ASCII uppercase to lowercase using AVX-512
pub fn to_lowercase_ascii(input: &mut [u8]) {
    #[cfg(all(target_arch = "x86_64", feature = "nightly"))]
    {
        if is_avx512_available() && input.len() >= 64 {
            unsafe { to_lowercase_ascii_avx512(input) };
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

/// AVX-512 lowercase conversion
#[cfg(all(target_arch = "x86_64", feature = "nightly"))]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn to_lowercase_ascii_avx512(input: &mut [u8]) {
    use std::arch::x86_64::*;

    let mut i = 0;
    let upper_a = _mm512_set1_epi8(b'A' as i8);
    let upper_z = _mm512_set1_epi8(b'Z' as i8);
    let diff = _mm512_set1_epi8(32);

    while i + 64 <= input.len() {
        let chunk = _mm512_loadu_si512(input.as_ptr().add(i) as *const _);

        // Find uppercase letters: A <= x <= Z
        let ge_a = _mm512_cmpge_epi8_mask(chunk, upper_a);
        let le_z = _mm512_cmple_epi8_mask(chunk, upper_z);
        let is_upper = ge_a & le_z;

        // Add 32 to uppercase letters using mask blend
        let lowered = _mm512_add_epi8(chunk, diff);
        let result = _mm512_mask_blend_epi8(is_upper, chunk, lowered);

        _mm512_storeu_si512(input.as_mut_ptr().add(i) as *mut _, result);
        i += 64;
    }

    // Handle remaining bytes
    to_lowercase_ascii_scalar(&mut input[i..]);
}

/// Convert ASCII lowercase to uppercase using AVX-512
pub fn to_uppercase_ascii(input: &mut [u8]) {
    #[cfg(all(target_arch = "x86_64", feature = "nightly"))]
    {
        if is_avx512_available() && input.len() >= 64 {
            unsafe { to_uppercase_ascii_avx512(input) };
            return;
        }
    }

    to_uppercase_ascii_scalar(input);
}

/// Scalar uppercase conversion
fn to_uppercase_ascii_scalar(input: &mut [u8]) {
    for byte in input {
        if *byte >= b'a' && *byte <= b'z' {
            *byte -= 32;
        }
    }
}

/// AVX-512 uppercase conversion
#[cfg(all(target_arch = "x86_64", feature = "nightly"))]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn to_uppercase_ascii_avx512(input: &mut [u8]) {
    use std::arch::x86_64::*;

    let mut i = 0;
    let lower_a = _mm512_set1_epi8(b'a' as i8);
    let lower_z = _mm512_set1_epi8(b'z' as i8);
    let diff = _mm512_set1_epi8(32);

    while i + 64 <= input.len() {
        let chunk = _mm512_loadu_si512(input.as_ptr().add(i) as *const _);

        // Find lowercase letters: a <= x <= z
        let ge_a = _mm512_cmpge_epi8_mask(chunk, lower_a);
        let le_z = _mm512_cmple_epi8_mask(chunk, lower_z);
        let is_lower = ge_a & le_z;

        // Subtract 32 from lowercase letters using mask blend
        let uppered = _mm512_sub_epi8(chunk, diff);
        let result = _mm512_mask_blend_epi8(is_lower, chunk, uppered);

        _mm512_storeu_si512(input.as_mut_ptr().add(i) as *mut _, result);
        i += 64;
    }

    // Handle remaining bytes
    to_uppercase_ascii_scalar(&mut input[i..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_avx512_available() {
        let _ = is_avx512_available();
    }

    #[test]
    fn test_classify_whitespace() {
        let input = b"hello world test data here now";
        let mask = classify_whitespace(input);
        // Space at position 5, 11, 16, 21, 26
        assert!(mask & (1 << 5) != 0);
        assert!(mask & (1 << 11) != 0);
        assert!(mask & (1 << 16) != 0);
        assert!(mask & (1 << 21) != 0);
        assert!(mask & (1 << 26) != 0);
    }

    #[test]
    fn test_classify_whitespace_tabs_newlines() {
        let input = b"a\tb\nc\rd";
        let mask = classify_whitespace(input);
        assert!(mask & (1 << 1) != 0); // tab
        assert!(mask & (1 << 3) != 0); // newline
        assert!(mask & (1 << 5) != 0); // carriage return
    }

    #[test]
    fn test_classify_alphanumeric() {
        let input = b"abc123!@#XYZ456def";
        let mask = classify_alphanumeric(input);
        // a,b,c,1,2,3 at 0-5, X,Y,Z at 9-11, 4,5,6 at 12-14, d,e,f at 15-17
        assert!(mask & (1 << 0) != 0); // a
        assert!(mask & (1 << 5) != 0); // 3
        assert!(mask & (1 << 6) == 0); // !
        assert!(mask & (1 << 9) != 0); // X
    }

    #[test]
    fn test_classify_punctuation() {
        let input = b"Hello, World! Test: data.";
        let mask = classify_punctuation(input);
        // Punctuation at: , (5), ! (12), : (18), . (24)
        assert!(mask & (1 << 5) != 0);  // ,
        assert!(mask & (1 << 12) != 0); // !
        assert!(mask & (1 << 18) != 0); // :
        assert!(mask & (1 << 24) != 0); // .
    }

    #[test]
    fn test_classify_non_ascii() {
        let input = b"hello";
        assert_eq!(classify_non_ascii(input), 0);

        let mut input2 = b"hello".to_vec();
        input2.push(0x80);
        let mask = classify_non_ascii(&input2);
        assert!(mask & (1 << 5) != 0);
    }

    #[test]
    fn test_pretokenize() {
        let input = b"hello world test";
        let tokens = pretokenize(input);
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], (0, 5));   // "hello"
        assert_eq!(tokens[1], (6, 11));  // "world"
        assert_eq!(tokens[2], (12, 16)); // "test"
    }

    #[test]
    fn test_pretokenize_multiple_spaces() {
        let input = b"hello  world";
        let tokens = pretokenize(input);
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], (0, 5));
        assert_eq!(tokens[1], (7, 12));
    }

    #[test]
    fn test_pretokenize_leading_trailing() {
        let input = b"  hello world  ";
        let tokens = pretokenize(input);
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], (2, 7));
        assert_eq!(tokens[1], (8, 13));
    }

    #[test]
    fn test_pretokenize_empty() {
        let tokens = pretokenize(b"");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_pretokenize_only_whitespace() {
        let tokens = pretokenize(b"   \t\n  ");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_find_word_boundaries() {
        let input = b"hello world test";
        let boundaries = find_word_boundaries(input);
        assert_eq!(boundaries, vec![5, 11]);
    }

    #[test]
    fn test_count_whitespace() {
        let input = b"hello world test";
        assert_eq!(count_whitespace(input), 2);
    }

    #[test]
    fn test_is_all_ascii() {
        assert!(is_all_ascii(b"hello world"));
        assert!(!is_all_ascii("héllo".as_bytes()));
    }

    #[test]
    fn test_find_first_non_ascii() {
        assert_eq!(find_first_non_ascii(b"hello"), None);
        assert_eq!(find_first_non_ascii("café".as_bytes()), Some(3));
    }

    #[test]
    fn test_to_lowercase() {
        let mut input = b"Hello World ABC".to_vec();
        to_lowercase_ascii(&mut input);
        assert_eq!(&input, b"hello world abc");
    }

    #[test]
    fn test_to_uppercase() {
        let mut input = b"Hello World abc".to_vec();
        to_uppercase_ascii(&mut input);
        assert_eq!(&input, b"HELLO WORLD ABC");
    }

    #[test]
    fn test_long_input() {
        // Test with input longer than 64 bytes
        let input = b"The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";
        assert_eq!(count_whitespace(input), 16);
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(classify_whitespace(b""), 0);
        assert_eq!(classify_alphanumeric(b""), 0);
        assert!(pretokenize(b"").is_empty());
    }

    #[test]
    fn test_extract_positions() {
        let mut positions = Vec::new();
        extract_positions_from_mask(0b10101, 0, &mut positions);
        assert_eq!(positions, vec![0, 2, 4]);
    }

    #[test]
    fn test_pretokenize_long() {
        // Test with input that triggers AVX-512 path (if available)
        let input = b"The quick brown fox jumps over the lazy dog and then continues running through the forest";
        let tokens = pretokenize(input);
        assert!(!tokens.is_empty());
        // Verify first and last token
        assert_eq!(&input[tokens[0].0..tokens[0].1], b"The");
        assert_eq!(&input[tokens.last().unwrap().0..tokens.last().unwrap().1], b"forest");
    }
}
