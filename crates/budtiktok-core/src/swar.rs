//! SWAR (SIMD Within A Register) byte operations
//!
//! High-performance byte manipulation using u64 word operations.
//! These techniques allow processing 8 bytes at a time without SIMD intrinsics,
//! providing portable performance improvements across all platforms.
//!
//! Based on techniques from:
//! - "Hacker's Delight" by Henry S. Warren Jr.
//! - BlazeText and simdutf implementations
//! - TEI tokenization optimizations

/// Magic constants for SWAR operations
const LO_BYTE: u64 = 0x0101_0101_0101_0101;
const HI_BYTE: u64 = 0x8080_8080_8080_8080;

// ============================================================================
// Core SWAR Operations (4.6.1)
// ============================================================================

/// Check if a u64 word contains a zero byte.
/// This is the foundation for many string operations.
///
/// Algorithm: For each byte in the word:
/// - Subtracting 0x01 from a non-zero byte won't set the high bit
/// - Subtracting 0x01 from a zero byte will set the high bit (0x00 - 0x01 = 0xFF)
/// - ANDing with !x ensures we only detect zero bytes, not bytes >= 0x80
///
/// # Example
/// ```rust,ignore
/// assert!(has_zero_byte(0x0048656C6C6F0000)); // "Hello\0\0"
/// assert!(!has_zero_byte(0x48656C6C6F576F72)); // "HelloWor"
/// ```
#[inline(always)]
pub const fn has_zero_byte(x: u64) -> bool {
    x.wrapping_sub(LO_BYTE) & !x & HI_BYTE != 0
}

/// Check if a u64 word contains a specific byte value.
/// XORing with the broadcast byte converts matches to zeros.
///
/// # Example
/// ```rust,ignore
/// let word = u64::from_ne_bytes(*b"hello wo");
/// assert!(has_byte(word, b' ')); // Contains space
/// assert!(!has_byte(word, b'x')); // No 'x'
/// ```
#[inline(always)]
pub const fn has_byte(x: u64, byte: u8) -> bool {
    has_zero_byte(x ^ (LO_BYTE * byte as u64))
}

/// Broadcast a byte value to all 8 positions in a u64.
#[inline(always)]
pub const fn broadcast_byte(byte: u8) -> u64 {
    LO_BYTE * byte as u64
}

/// Check if any byte in a u64 has its high bit set (non-ASCII).
#[inline(always)]
pub const fn has_non_ascii(x: u64) -> bool {
    x & HI_BYTE != 0
}

/// Count the number of bytes less than a given value.
/// Useful for counting ASCII characters, digits, etc.
#[inline(always)]
pub fn count_bytes_less_than(x: u64, threshold: u8) -> u32 {
    // Subtract threshold from each byte (with saturation-like behavior)
    // Bytes < threshold will have high bit set after adjustment
    let threshold_broadcast = broadcast_byte(threshold);
    let adjusted = x.wrapping_sub(threshold_broadcast);
    // Count bytes where high bit is NOT set (those were >= threshold originally)
    // So we count bytes where high bit IS set in the adjusted value
    let mask = adjusted & HI_BYTE;
    (mask.count_ones() / 8) as u32
}

/// Find the position of the first occurrence of a byte (0-7), or 8 if not found.
#[inline(always)]
pub fn find_byte_position(x: u64, byte: u8) -> usize {
    let xored = x ^ broadcast_byte(byte);
    let has_zero = xored.wrapping_sub(LO_BYTE) & !xored & HI_BYTE;
    if has_zero == 0 {
        8
    } else {
        // Find position of first zero byte
        #[cfg(target_endian = "little")]
        {
            has_zero.trailing_zeros() as usize / 8
        }
        #[cfg(target_endian = "big")]
        {
            has_zero.leading_zeros() as usize / 8
        }
    }
}

/// Find the position of the first whitespace character (space, tab, newline, CR).
#[inline(always)]
pub fn find_whitespace_position(x: u64) -> usize {
    // Check for each whitespace character
    let space_match = x ^ broadcast_byte(b' ');
    let tab_match = x ^ broadcast_byte(b'\t');
    let newline_match = x ^ broadcast_byte(b'\n');
    let cr_match = x ^ broadcast_byte(b'\r');

    // Detect zeros in each
    let space_zero = space_match.wrapping_sub(LO_BYTE) & !space_match & HI_BYTE;
    let tab_zero = tab_match.wrapping_sub(LO_BYTE) & !tab_match & HI_BYTE;
    let newline_zero = newline_match.wrapping_sub(LO_BYTE) & !newline_match & HI_BYTE;
    let cr_zero = cr_match.wrapping_sub(LO_BYTE) & !cr_match & HI_BYTE;

    // Combine all matches
    let any_ws = space_zero | tab_zero | newline_zero | cr_zero;

    if any_ws == 0 {
        8
    } else {
        #[cfg(target_endian = "little")]
        {
            any_ws.trailing_zeros() as usize / 8
        }
        #[cfg(target_endian = "big")]
        {
            any_ws.leading_zeros() as usize / 8
        }
    }
}

/// Create a mask of bytes that match any of the given characters.
/// Returns a u64 where each byte is 0xFF if it matched, 0x00 otherwise.
#[inline(always)]
pub fn match_any_of_4(x: u64, a: u8, b: u8, c: u8, d: u8) -> u64 {
    let match_a = x ^ broadcast_byte(a);
    let match_b = x ^ broadcast_byte(b);
    let match_c = x ^ broadcast_byte(c);
    let match_d = x ^ broadcast_byte(d);

    let zero_a = match_a.wrapping_sub(LO_BYTE) & !match_a & HI_BYTE;
    let zero_b = match_b.wrapping_sub(LO_BYTE) & !match_b & HI_BYTE;
    let zero_c = match_c.wrapping_sub(LO_BYTE) & !match_c & HI_BYTE;
    let zero_d = match_d.wrapping_sub(LO_BYTE) & !match_d & HI_BYTE;

    // Convert high-bit masks to full-byte masks
    let combined = zero_a | zero_b | zero_c | zero_d;
    // Spread high bit to entire byte: 0x80 -> 0xFF
    (combined >> 7) * 0xFF
}

// ============================================================================
// Branchless Classification (4.6.2)
// ============================================================================

/// Branchless check if a byte is ASCII whitespace.
/// Returns 1 if whitespace, 0 otherwise.
#[inline(always)]
pub const fn is_ascii_whitespace_branchless(b: u8) -> u8 {
    // space=32, tab=9, newline=10, cr=13
    let is_space = (b == b' ') as u8;
    let is_tab = (b == b'\t') as u8;
    let is_nl = (b == b'\n') as u8;
    let is_cr = (b == b'\r') as u8;
    is_space | is_tab | is_nl | is_cr
}

/// Branchless check if a byte is ASCII alphanumeric.
#[inline(always)]
pub const fn is_ascii_alphanumeric_branchless(b: u8) -> u8 {
    // Lowercase: 'a'-'z' (97-122)
    // Uppercase: 'A'-'Z' (65-90)
    // Digit: '0'-'9' (48-57)
    let lower = ((b >= b'a') as u8) & ((b <= b'z') as u8);
    let upper = ((b >= b'A') as u8) & ((b <= b'Z') as u8);
    let digit = ((b >= b'0') as u8) & ((b <= b'9') as u8);
    lower | upper | digit
}

/// Branchless check if a byte is ASCII punctuation.
#[inline(always)]
pub const fn is_ascii_punctuation_branchless(b: u8) -> u8 {
    // Punctuation ranges: 33-47, 58-64, 91-96, 123-126
    let r1 = ((b >= 33) as u8) & ((b <= 47) as u8);
    let r2 = ((b >= 58) as u8) & ((b <= 64) as u8);
    let r3 = ((b >= 91) as u8) & ((b <= 96) as u8);
    let r4 = ((b >= 123) as u8) & ((b <= 126) as u8);
    r1 | r2 | r3 | r4
}

/// Branchless lowercase conversion.
/// Converts A-Z to a-z, leaves other bytes unchanged.
#[inline(always)]
pub const fn to_lowercase_branchless(b: u8) -> u8 {
    // If byte is A-Z (65-90), add 32 to make it lowercase
    let is_upper = ((b >= b'A') as u8) & ((b <= b'Z') as u8);
    b + (is_upper * 32)
}

/// Branchless uppercase conversion.
#[inline(always)]
pub const fn to_uppercase_branchless(b: u8) -> u8 {
    let is_lower = ((b >= b'a') as u8) & ((b <= b'z') as u8);
    b - (is_lower * 32)
}

/// Classify a byte into categories using branchless operations.
/// Returns a bitfield: bit 0 = whitespace, bit 1 = alphanumeric, bit 2 = punctuation, bit 3 = control
#[inline(always)]
pub const fn classify_byte_branchless(b: u8) -> u8 {
    let whitespace = is_ascii_whitespace_branchless(b);
    let alphanumeric = is_ascii_alphanumeric_branchless(b);
    let punctuation = is_ascii_punctuation_branchless(b);
    let control = ((b < 32) as u8) & (1 - whitespace); // Control but not whitespace

    whitespace | (alphanumeric << 1) | (punctuation << 2) | (control << 3)
}

// ============================================================================
// Loop Unrolling Utilities (4.6.3)
// ============================================================================

/// Process bytes in chunks of 8 using SWAR, with loop unrolling.
/// Returns the index of the first whitespace character, or the length if none found.
pub fn find_first_whitespace_unrolled(bytes: &[u8]) -> usize {
    let len = bytes.len();
    let mut i = 0;

    // Process 32 bytes at a time (4x8-byte words)
    while i + 32 <= len {
        let ptr = bytes.as_ptr();

        // SAFETY: We've verified i + 32 <= len
        unsafe {
            let w0 = (ptr.add(i) as *const u64).read_unaligned();
            let w1 = (ptr.add(i + 8) as *const u64).read_unaligned();
            let w2 = (ptr.add(i + 16) as *const u64).read_unaligned();
            let w3 = (ptr.add(i + 24) as *const u64).read_unaligned();

            // Check first word
            let pos0 = find_whitespace_position(w0);
            if pos0 < 8 {
                return i + pos0;
            }

            // Check second word
            let pos1 = find_whitespace_position(w1);
            if pos1 < 8 {
                return i + 8 + pos1;
            }

            // Check third word
            let pos2 = find_whitespace_position(w2);
            if pos2 < 8 {
                return i + 16 + pos2;
            }

            // Check fourth word
            let pos3 = find_whitespace_position(w3);
            if pos3 < 8 {
                return i + 24 + pos3;
            }
        }

        i += 32;
    }

    // Process 8 bytes at a time
    while i + 8 <= len {
        let ptr = bytes.as_ptr();
        unsafe {
            let word = (ptr.add(i) as *const u64).read_unaligned();
            let pos = find_whitespace_position(word);
            if pos < 8 {
                return i + pos;
            }
        }
        i += 8;
    }

    // Process remaining bytes
    while i < len {
        if is_ascii_whitespace_branchless(bytes[i]) != 0 {
            return i;
        }
        i += 1;
    }

    len
}

/// Count whitespace characters using SWAR with loop unrolling.
pub fn count_whitespace_unrolled(bytes: &[u8]) -> usize {
    let len = bytes.len();
    let mut count = 0usize;
    let mut i = 0;

    // Process 32 bytes at a time
    while i + 32 <= len {
        let ptr = bytes.as_ptr();

        unsafe {
            let w0 = (ptr.add(i) as *const u64).read_unaligned();
            let w1 = (ptr.add(i + 8) as *const u64).read_unaligned();
            let w2 = (ptr.add(i + 16) as *const u64).read_unaligned();
            let w3 = (ptr.add(i + 24) as *const u64).read_unaligned();

            // Count whitespace in each word
            count += count_whitespace_in_word(w0) as usize;
            count += count_whitespace_in_word(w1) as usize;
            count += count_whitespace_in_word(w2) as usize;
            count += count_whitespace_in_word(w3) as usize;
        }

        i += 32;
    }

    // Process remaining bytes
    while i < len {
        count += is_ascii_whitespace_branchless(bytes[i]) as usize;
        i += 1;
    }

    count
}

/// Count whitespace characters in a single u64 word.
#[inline(always)]
fn count_whitespace_in_word(x: u64) -> u32 {
    // Check for each whitespace character
    let space_match = x ^ broadcast_byte(b' ');
    let tab_match = x ^ broadcast_byte(b'\t');
    let newline_match = x ^ broadcast_byte(b'\n');
    let cr_match = x ^ broadcast_byte(b'\r');

    // Detect zeros (matches) - each match sets the high bit (0x80) of that byte
    let space_zero = space_match.wrapping_sub(LO_BYTE) & !space_match & HI_BYTE;
    let tab_zero = tab_match.wrapping_sub(LO_BYTE) & !tab_match & HI_BYTE;
    let newline_zero = newline_match.wrapping_sub(LO_BYTE) & !newline_match & HI_BYTE;
    let cr_zero = cr_match.wrapping_sub(LO_BYTE) & !cr_match & HI_BYTE;

    // Each match contributes exactly one bit (at position 0x80 of that byte)
    // So count_ones() directly gives us the count of whitespace characters
    let combined = space_zero | tab_zero | newline_zero | cr_zero;
    combined.count_ones()
}

/// Check if all bytes in a slice are ASCII using SWAR with unrolling.
pub fn is_all_ascii_unrolled(bytes: &[u8]) -> bool {
    let len = bytes.len();
    let mut i = 0;

    // Process 32 bytes at a time
    while i + 32 <= len {
        let ptr = bytes.as_ptr();

        unsafe {
            let w0 = (ptr.add(i) as *const u64).read_unaligned();
            let w1 = (ptr.add(i + 8) as *const u64).read_unaligned();
            let w2 = (ptr.add(i + 16) as *const u64).read_unaligned();
            let w3 = (ptr.add(i + 24) as *const u64).read_unaligned();

            // Check if any word has non-ASCII
            if has_non_ascii(w0 | w1 | w2 | w3) {
                return false;
            }
        }

        i += 32;
    }

    // Process remaining with 8-byte chunks
    while i + 8 <= len {
        let ptr = bytes.as_ptr();
        unsafe {
            let word = (ptr.add(i) as *const u64).read_unaligned();
            if has_non_ascii(word) {
                return false;
            }
        }
        i += 8;
    }

    // Check remaining bytes
    while i < len {
        if bytes[i] >= 128 {
            return false;
        }
        i += 1;
    }

    true
}

/// Convert ASCII bytes to lowercase in-place using SWAR.
pub fn to_lowercase_ascii_unrolled(bytes: &mut [u8]) {
    let len = bytes.len();
    let mut i = 0;

    // Process 8 bytes at a time
    while i + 8 <= len {
        let ptr = bytes.as_mut_ptr();

        unsafe {
            let word = (ptr.add(i) as *const u64).read_unaligned();

            // For each byte, if it's A-Z (65-90), add 32
            // This uses the fact that adding 32 is equivalent to OR'ing with 0x20
            // But we only want to do this for A-Z

            // Create mask for uppercase letters
            let lower_bound = word.wrapping_sub(broadcast_byte(b'A'));
            let upper_bound = broadcast_byte(b'Z').wrapping_sub(word);

            // Bytes in range will have high bit clear in both
            let in_range = !(lower_bound | upper_bound) & HI_BYTE;

            // Convert mask to OR mask (0x20 for each uppercase letter)
            let or_mask = in_range >> 2; // Shift 0x80 to 0x20

            let lowercased = word | or_mask;
            (ptr.add(i) as *mut u64).write_unaligned(lowercased);
        }

        i += 8;
    }

    // Handle remaining bytes
    while i < len {
        bytes[i] = to_lowercase_branchless(bytes[i]);
        i += 1;
    }
}

/// Split bytes on whitespace, returning byte ranges.
/// Uses SWAR for efficient whitespace detection.
pub fn split_on_whitespace_ranges(bytes: &[u8]) -> Vec<(usize, usize)> {
    let len = bytes.len();
    if len == 0 {
        return vec![];
    }

    let mut ranges = Vec::new();
    let mut start: Option<usize> = None;
    let mut i = 0;

    // Process 8 bytes at a time
    while i + 8 <= len {
        let ptr = bytes.as_ptr();

        unsafe {
            let word = (ptr.add(i) as *const u64).read_unaligned();

            // Create whitespace mask for each byte
            let space_match = word ^ broadcast_byte(b' ');
            let tab_match = word ^ broadcast_byte(b'\t');
            let newline_match = word ^ broadcast_byte(b'\n');
            let cr_match = word ^ broadcast_byte(b'\r');

            let space_zero = space_match.wrapping_sub(LO_BYTE) & !space_match & HI_BYTE;
            let tab_zero = tab_match.wrapping_sub(LO_BYTE) & !tab_match & HI_BYTE;
            let newline_zero = newline_match.wrapping_sub(LO_BYTE) & !newline_match & HI_BYTE;
            let cr_zero = cr_match.wrapping_sub(LO_BYTE) & !cr_match & HI_BYTE;

            let ws_mask = space_zero | tab_zero | newline_zero | cr_zero;

            // Process each byte in the word
            for j in 0..8 {
                let byte_idx = i + j;
                #[cfg(target_endian = "little")]
                let is_ws = (ws_mask >> (j * 8)) & 0x80 != 0;
                #[cfg(target_endian = "big")]
                let is_ws = (ws_mask >> ((7 - j) * 8)) & 0x80 != 0;

                if is_ws {
                    if let Some(s) = start {
                        ranges.push((s, byte_idx));
                        start = None;
                    }
                } else if start.is_none() {
                    start = Some(byte_idx);
                }
            }
        }

        i += 8;
    }

    // Handle remaining bytes
    while i < len {
        let is_ws = is_ascii_whitespace_branchless(bytes[i]) != 0;
        if is_ws {
            if let Some(s) = start {
                ranges.push((s, i));
                start = None;
            }
        } else if start.is_none() {
            start = Some(i);
        }
        i += 1;
    }

    // Handle final range
    if let Some(s) = start {
        ranges.push((s, len));
    }

    ranges
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========== SWAR Core Tests (4.6.1) ==========

    #[test]
    fn test_has_zero_byte() {
        // Word with zero byte
        assert!(has_zero_byte(0x0048656C6C6F0000));
        // Word without zero byte
        assert!(!has_zero_byte(0x48656C6C6F576F72));
        // Zero at different positions
        assert!(has_zero_byte(0x0000000000000000));
        assert!(has_zero_byte(0x00FFFFFFFFFFFFFF));
        assert!(has_zero_byte(0xFF00FFFFFFFFFFFF));
        assert!(has_zero_byte(0xFFFFFFFFFFFFFF00));
    }

    #[test]
    fn test_has_byte() {
        let word = u64::from_ne_bytes(*b"hello wo");
        assert!(has_byte(word, b'h'));
        assert!(has_byte(word, b'e'));
        assert!(has_byte(word, b'l'));
        assert!(has_byte(word, b'o'));
        assert!(has_byte(word, b' '));
        assert!(has_byte(word, b'w'));
        assert!(!has_byte(word, b'x'));
        assert!(!has_byte(word, b'z'));
    }

    #[test]
    fn test_has_non_ascii() {
        assert!(!has_non_ascii(u64::from_ne_bytes(*b"hello wo")));
        assert!(!has_non_ascii(0x7F7F7F7F7F7F7F7F));
        assert!(has_non_ascii(0x8000000000000000));
        assert!(has_non_ascii(0x0000000000000080));
        assert!(has_non_ascii(0xFFFFFFFFFFFFFFFF));
    }

    #[test]
    fn test_find_byte_position() {
        let word = u64::from_ne_bytes(*b"hello wo");
        assert_eq!(find_byte_position(word, b'h'), 0);
        assert_eq!(find_byte_position(word, b' '), 5);
        assert_eq!(find_byte_position(word, b'x'), 8); // Not found
    }

    #[test]
    fn test_find_whitespace_position() {
        let word1 = u64::from_ne_bytes(*b"hello wo");
        assert_eq!(find_whitespace_position(word1), 5);

        let word2 = u64::from_ne_bytes(*b"helloXXX");
        assert_eq!(find_whitespace_position(word2), 8);

        let word3 = u64::from_ne_bytes(*b" hello  ");
        assert_eq!(find_whitespace_position(word3), 0);

        let word4 = u64::from_ne_bytes(*b"hello\txx");
        assert_eq!(find_whitespace_position(word4), 5);
    }

    // ========== Branchless Classification Tests (4.6.2) ==========

    #[test]
    fn test_is_ascii_whitespace_branchless() {
        assert_eq!(is_ascii_whitespace_branchless(b' '), 1);
        assert_eq!(is_ascii_whitespace_branchless(b'\t'), 1);
        assert_eq!(is_ascii_whitespace_branchless(b'\n'), 1);
        assert_eq!(is_ascii_whitespace_branchless(b'\r'), 1);
        assert_eq!(is_ascii_whitespace_branchless(b'a'), 0);
        assert_eq!(is_ascii_whitespace_branchless(b'Z'), 0);
        assert_eq!(is_ascii_whitespace_branchless(b'0'), 0);
    }

    #[test]
    fn test_is_ascii_alphanumeric_branchless() {
        assert_eq!(is_ascii_alphanumeric_branchless(b'a'), 1);
        assert_eq!(is_ascii_alphanumeric_branchless(b'z'), 1);
        assert_eq!(is_ascii_alphanumeric_branchless(b'A'), 1);
        assert_eq!(is_ascii_alphanumeric_branchless(b'Z'), 1);
        assert_eq!(is_ascii_alphanumeric_branchless(b'0'), 1);
        assert_eq!(is_ascii_alphanumeric_branchless(b'9'), 1);
        assert_eq!(is_ascii_alphanumeric_branchless(b' '), 0);
        assert_eq!(is_ascii_alphanumeric_branchless(b'.'), 0);
    }

    #[test]
    fn test_is_ascii_punctuation_branchless() {
        assert_eq!(is_ascii_punctuation_branchless(b'!'), 1);
        assert_eq!(is_ascii_punctuation_branchless(b'.'), 1);
        assert_eq!(is_ascii_punctuation_branchless(b','), 1);
        assert_eq!(is_ascii_punctuation_branchless(b'?'), 1);
        assert_eq!(is_ascii_punctuation_branchless(b'['), 1);
        assert_eq!(is_ascii_punctuation_branchless(b'{'), 1);
        assert_eq!(is_ascii_punctuation_branchless(b'a'), 0);
        assert_eq!(is_ascii_punctuation_branchless(b' '), 0);
    }

    #[test]
    fn test_to_lowercase_branchless() {
        assert_eq!(to_lowercase_branchless(b'A'), b'a');
        assert_eq!(to_lowercase_branchless(b'Z'), b'z');
        assert_eq!(to_lowercase_branchless(b'M'), b'm');
        assert_eq!(to_lowercase_branchless(b'a'), b'a'); // Already lowercase
        assert_eq!(to_lowercase_branchless(b'0'), b'0'); // Not a letter
        assert_eq!(to_lowercase_branchless(b' '), b' '); // Not a letter
    }

    #[test]
    fn test_to_uppercase_branchless() {
        assert_eq!(to_uppercase_branchless(b'a'), b'A');
        assert_eq!(to_uppercase_branchless(b'z'), b'Z');
        assert_eq!(to_uppercase_branchless(b'm'), b'M');
        assert_eq!(to_uppercase_branchless(b'A'), b'A'); // Already uppercase
        assert_eq!(to_uppercase_branchless(b'0'), b'0'); // Not a letter
    }

    #[test]
    fn test_classify_byte_branchless() {
        // Whitespace
        let ws_class = classify_byte_branchless(b' ');
        assert_eq!(ws_class & 1, 1); // Whitespace bit set

        // Alphanumeric
        let alpha_class = classify_byte_branchless(b'a');
        assert_eq!(alpha_class & 2, 2); // Alphanumeric bit set
        assert_eq!(alpha_class & 1, 0); // Not whitespace

        // Punctuation
        let punct_class = classify_byte_branchless(b'!');
        assert_eq!(punct_class & 4, 4); // Punctuation bit set
    }

    // ========== Loop Unrolling Tests (4.6.3) ==========

    #[test]
    fn test_find_first_whitespace_unrolled() {
        assert_eq!(find_first_whitespace_unrolled(b"hello world"), 5);
        assert_eq!(find_first_whitespace_unrolled(b"helloworld"), 10);
        assert_eq!(find_first_whitespace_unrolled(b" hello"), 0);
        assert_eq!(find_first_whitespace_unrolled(b"hello\tworld"), 5);
        assert_eq!(find_first_whitespace_unrolled(b"hello\nworld"), 5);
        assert_eq!(find_first_whitespace_unrolled(b""), 0);

        // Long string to test unrolling
        let long = b"abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz test";
        assert_eq!(find_first_whitespace_unrolled(long), 52);
    }

    #[test]
    fn test_count_whitespace_unrolled() {
        assert_eq!(count_whitespace_unrolled(b"hello world"), 1);
        assert_eq!(count_whitespace_unrolled(b"hello  world"), 2);
        assert_eq!(count_whitespace_unrolled(b"   "), 3);
        assert_eq!(count_whitespace_unrolled(b"hello\t\n\r"), 3);
        assert_eq!(count_whitespace_unrolled(b"nospaces"), 0);
        assert_eq!(count_whitespace_unrolled(b""), 0);

        // Long string
        let long = b"a b c d e f g h i j k l m n o p q r s t u v w x y z";
        assert_eq!(count_whitespace_unrolled(long), 25);
    }

    #[test]
    fn test_is_all_ascii_unrolled() {
        assert!(is_all_ascii_unrolled(b"hello world"));
        assert!(is_all_ascii_unrolled(b"The quick brown fox"));
        assert!(is_all_ascii_unrolled(b""));
        assert!(is_all_ascii_unrolled(b"123456789"));
        assert!(!is_all_ascii_unrolled("café".as_bytes()));
        assert!(!is_all_ascii_unrolled("日本語".as_bytes()));

        // Long ASCII string
        let long: Vec<u8> = (0..1000).map(|i| (i % 128) as u8).collect();
        assert!(is_all_ascii_unrolled(&long));

        // Long string with non-ASCII at end
        let mut long_non_ascii = vec![b'a'; 100];
        long_non_ascii.push(0x80);
        assert!(!is_all_ascii_unrolled(&long_non_ascii));
    }

    #[test]
    fn test_to_lowercase_ascii_unrolled() {
        let mut bytes = b"HELLO WORLD".to_vec();
        to_lowercase_ascii_unrolled(&mut bytes);
        assert_eq!(&bytes, b"hello world");

        let mut mixed = b"HeLLo WoRLD 123!".to_vec();
        to_lowercase_ascii_unrolled(&mut mixed);
        assert_eq!(&mixed, b"hello world 123!");

        // Long string
        let mut long: Vec<u8> = (b'A'..=b'Z').cycle().take(100).collect();
        to_lowercase_ascii_unrolled(&mut long);
        let expected: Vec<u8> = (b'a'..=b'z').cycle().take(100).collect();
        assert_eq!(long, expected);
    }

    #[test]
    fn test_split_on_whitespace_ranges() {
        let ranges = split_on_whitespace_ranges(b"hello world test");
        assert_eq!(ranges, vec![(0, 5), (6, 11), (12, 16)]);

        let ranges2 = split_on_whitespace_ranges(b"  hello  ");
        assert_eq!(ranges2, vec![(2, 7)]);

        let ranges3 = split_on_whitespace_ranges(b"nospace");
        assert_eq!(ranges3, vec![(0, 7)]);

        let ranges4 = split_on_whitespace_ranges(b"");
        assert_eq!(ranges4, vec![] as Vec<(usize, usize)>);

        let ranges5 = split_on_whitespace_ranges(b"   ");
        assert_eq!(ranges5, vec![] as Vec<(usize, usize)>);

        // With tabs and newlines
        let ranges6 = split_on_whitespace_ranges(b"a\tb\nc");
        assert_eq!(ranges6, vec![(0, 1), (2, 3), (4, 5)]);
    }

    // ========== Property-based tests ==========

    #[test]
    fn test_swar_consistency_with_scalar() {
        // Verify SWAR functions match scalar implementations
        for b in 0u8..=255 {
            let is_ws_swar = is_ascii_whitespace_branchless(b) != 0;
            let is_ws_scalar = b == b' ' || b == b'\t' || b == b'\n' || b == b'\r';
            assert_eq!(is_ws_swar, is_ws_scalar, "Whitespace mismatch for byte {}", b);

            let lower_swar = to_lowercase_branchless(b);
            let lower_scalar = if b >= b'A' && b <= b'Z' { b + 32 } else { b };
            assert_eq!(lower_swar, lower_scalar, "Lowercase mismatch for byte {}", b);
        }
    }

    #[test]
    fn test_find_whitespace_consistency() {
        let test_cases = [
            b"hello world".as_slice(),
            b"no_whitespace",
            b" leading",
            b"trailing ",
            b"\ttab\there",
            b"a b c d e f g h i j k l m n o p",
        ];

        for bytes in test_cases {
            let swar_result = find_first_whitespace_unrolled(bytes);
            let scalar_result = bytes.iter().position(|&b| {
                b == b' ' || b == b'\t' || b == b'\n' || b == b'\r'
            }).unwrap_or(bytes.len());
            assert_eq!(swar_result, scalar_result, "Mismatch for {:?}", bytes);
        }
    }
}
