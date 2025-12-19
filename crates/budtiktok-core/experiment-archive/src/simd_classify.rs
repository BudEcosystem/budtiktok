//! SIMD-Accelerated Character Classification
//!
//! Uses AVX2/AVX-512 instructions to classify multiple characters at once
//! for whitespace, punctuation, and CJK detection.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Classification results as bitmasks
#[derive(Debug, Clone, Copy)]
pub struct ClassificationResult {
    /// Bitmask of whitespace positions
    pub whitespace: u64,
    /// Bitmask of punctuation positions
    pub punctuation: u64,
    /// Bitmask of non-ASCII positions (potential CJK or multi-byte)
    pub non_ascii: u64,
}

/// SIMD character classifier
pub struct SimdClassifier {
    /// Whitespace lookup table (for pshufb)
    #[cfg(target_arch = "x86_64")]
    ws_lookup_lo: [u8; 16],
    #[cfg(target_arch = "x86_64")]
    ws_lookup_hi: [u8; 16],
    /// Punctuation lookup tables
    #[cfg(target_arch = "x86_64")]
    punct_lookup_lo: [u8; 16],
    #[cfg(target_arch = "x86_64")]
    punct_lookup_hi: [u8; 16],
}

impl SimdClassifier {
    pub fn new() -> Self {
        // Build lookup tables for SIMD classification
        // Using pshufb technique: split byte into high/low nibbles
        // and use two table lookups

        #[cfg(target_arch = "x86_64")]
        {
            // Whitespace: ' ', '\t', '\n', '\r' (0x20, 0x09, 0x0A, 0x0D)
            let mut ws_lo = [0u8; 16];
            let mut ws_hi = [0u8; 16];

            // For space (0x20): high nibble=2, low nibble=0
            ws_hi[2] = 1; // high nibble 2
            ws_lo[0] = 1; // low nibble 0

            // For tab (0x09): high nibble=0, low nibble=9
            ws_hi[0] |= 2;
            ws_lo[9] |= 2;

            // For newline (0x0A): high nibble=0, low nibble=A
            ws_hi[0] |= 4;
            ws_lo[0xA] |= 4;

            // For carriage return (0x0D): high nibble=0, low nibble=D
            ws_hi[0] |= 8;
            ws_lo[0xD] |= 8;

            // Punctuation lookup (simplified for common ASCII punctuation)
            let mut punct_lo = [0u8; 16];
            let mut punct_hi = [0u8; 16];

            // Mark common punctuation ranges
            for &c in b"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" {
                let hi = (c >> 4) as usize;
                let lo = (c & 0x0F) as usize;
                if hi < 16 && lo < 16 {
                    punct_hi[hi] |= 1 << (lo & 7);
                    punct_lo[lo] |= 1 << (hi & 7);
                }
            }

            Self {
                ws_lookup_lo: ws_lo,
                ws_lookup_hi: ws_hi,
                punct_lookup_lo: punct_lo,
                punct_lookup_hi: punct_hi,
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        Self {}
    }

    /// Classify 32 bytes at once using AVX2
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn classify_avx2(&self, data: &[u8]) -> ClassificationResult {
        debug_assert!(data.len() >= 32);

        // Load 32 bytes
        let input = _mm256_loadu_si256(data.as_ptr() as *const __m256i);

        // Check for non-ASCII (high bit set)
        let high_bit_mask = _mm256_set1_epi8(0x80u8 as i8);
        let non_ascii = _mm256_and_si256(input, high_bit_mask);
        let non_ascii_mask = _mm256_movemask_epi8(non_ascii) as u64;

        // Classify using lookup tables
        // Split into high and low nibbles
        let lo_nibble_mask = _mm256_set1_epi8(0x0F);
        let lo_nibbles = _mm256_and_si256(input, lo_nibble_mask);
        let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(input, 4), lo_nibble_mask);

        // Whitespace detection
        let ws_lo_table = _mm256_broadcastsi128_si256(_mm_loadu_si128(
            self.ws_lookup_lo.as_ptr() as *const __m128i,
        ));
        let ws_hi_table = _mm256_broadcastsi128_si256(_mm_loadu_si128(
            self.ws_lookup_hi.as_ptr() as *const __m128i,
        ));

        let ws_lo_result = _mm256_shuffle_epi8(ws_lo_table, lo_nibbles);
        let ws_hi_result = _mm256_shuffle_epi8(ws_hi_table, hi_nibbles);
        let ws_combined = _mm256_and_si256(ws_lo_result, ws_hi_result);
        let ws_mask = _mm256_movemask_epi8(_mm256_cmpgt_epi8(ws_combined, _mm256_setzero_si256()));

        // Punctuation detection (simplified - check specific ranges)
        let punct_start = _mm256_set1_epi8(0x21); // '!'
        let punct_end1 = _mm256_set1_epi8(0x2F);  // '/'
        let punct_start2 = _mm256_set1_epi8(0x3A); // ':'
        let punct_end2 = _mm256_set1_epi8(0x40);  // '@'
        let punct_start3 = _mm256_set1_epi8(0x5B); // '['
        let punct_end3 = _mm256_set1_epi8(0x60);  // '`'
        let punct_start4 = _mm256_set1_epi8(0x7B); // '{'
        let punct_end4 = _mm256_set1_epi8(0x7E);  // '~'

        // Check ranges
        let in_range1 = _mm256_and_si256(
            _mm256_cmpgt_epi8(input, _mm256_sub_epi8(punct_start, _mm256_set1_epi8(1))),
            _mm256_cmpgt_epi8(_mm256_add_epi8(punct_end1, _mm256_set1_epi8(1)), input),
        );
        let in_range2 = _mm256_and_si256(
            _mm256_cmpgt_epi8(input, _mm256_sub_epi8(punct_start2, _mm256_set1_epi8(1))),
            _mm256_cmpgt_epi8(_mm256_add_epi8(punct_end2, _mm256_set1_epi8(1)), input),
        );
        let in_range3 = _mm256_and_si256(
            _mm256_cmpgt_epi8(input, _mm256_sub_epi8(punct_start3, _mm256_set1_epi8(1))),
            _mm256_cmpgt_epi8(_mm256_add_epi8(punct_end3, _mm256_set1_epi8(1)), input),
        );
        let in_range4 = _mm256_and_si256(
            _mm256_cmpgt_epi8(input, _mm256_sub_epi8(punct_start4, _mm256_set1_epi8(1))),
            _mm256_cmpgt_epi8(_mm256_add_epi8(punct_end4, _mm256_set1_epi8(1)), input),
        );

        let punct_combined = _mm256_or_si256(
            _mm256_or_si256(in_range1, in_range2),
            _mm256_or_si256(in_range3, in_range4),
        );
        let punct_mask = _mm256_movemask_epi8(punct_combined) as u64;

        ClassificationResult {
            whitespace: ws_mask as u64,
            punctuation: punct_mask,
            non_ascii: non_ascii_mask,
        }
    }

    /// Scalar fallback for classification
    pub fn classify_scalar(&self, data: &[u8]) -> ClassificationResult {
        let mut whitespace = 0u64;
        let mut punctuation = 0u64;
        let mut non_ascii = 0u64;

        for (i, &byte) in data.iter().enumerate().take(64) {
            let bit = 1u64 << i;

            if byte >= 0x80 {
                non_ascii |= bit;
            } else {
                match byte {
                    b' ' | b'\t' | b'\n' | b'\r' => whitespace |= bit,
                    b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~' => punctuation |= bit,
                    _ => {}
                }
            }
        }

        ClassificationResult {
            whitespace,
            punctuation,
            non_ascii,
        }
    }

    /// Classify with automatic dispatch based on CPU features
    #[inline]
    pub fn classify(&self, data: &[u8]) -> ClassificationResult {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && data.len() >= 32 {
                return unsafe { self.classify_avx2(data) };
            }
        }
        self.classify_scalar(data)
    }
}

impl Default for SimdClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Fast whitespace scanning using SIMD
/// Returns the index of the first whitespace character, or None
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn find_whitespace_avx2(data: &[u8]) -> Option<usize> {
    let len = data.len();
    let mut offset = 0;

    // Process 32 bytes at a time
    while offset + 32 <= len {
        let chunk = _mm256_loadu_si256(data.as_ptr().add(offset) as *const __m256i);

        // Compare with space, tab, newline, carriage return
        let space = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b' ' as i8));
        let tab = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\t' as i8));
        let newline = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\n' as i8));
        let carriage = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\r' as i8));

        let ws = _mm256_or_si256(_mm256_or_si256(space, tab), _mm256_or_si256(newline, carriage));
        let mask = _mm256_movemask_epi8(ws) as u32;

        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }

        offset += 32;
    }

    // Scalar fallback for remaining bytes
    for (i, &byte) in data[offset..].iter().enumerate() {
        if matches!(byte, b' ' | b'\t' | b'\n' | b'\r') {
            return Some(offset + i);
        }
    }

    None
}

/// Fast non-whitespace scanning using SIMD
/// Skips leading whitespace and returns the index of first non-whitespace
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn skip_whitespace_avx2(data: &[u8]) -> usize {
    let len = data.len();
    let mut offset = 0;

    // Process 32 bytes at a time
    while offset + 32 <= len {
        let chunk = _mm256_loadu_si256(data.as_ptr().add(offset) as *const __m256i);

        // Compare with space, tab, newline, carriage return
        let space = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b' ' as i8));
        let tab = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\t' as i8));
        let newline = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\n' as i8));
        let carriage = _mm256_cmpeq_epi8(chunk, _mm256_set1_epi8(b'\r' as i8));

        let ws = _mm256_or_si256(_mm256_or_si256(space, tab), _mm256_or_si256(newline, carriage));
        let mask = _mm256_movemask_epi8(ws) as u32;

        // If not all whitespace, find first non-whitespace
        if mask != 0xFFFFFFFF {
            let non_ws_mask = !mask;
            if non_ws_mask != 0 {
                return offset + non_ws_mask.trailing_zeros() as usize;
            }
        }

        offset += 32;
    }

    // Scalar fallback for remaining bytes
    for (i, &byte) in data[offset..].iter().enumerate() {
        if !matches!(byte, b' ' | b'\t' | b'\n' | b'\r') {
            return offset + i;
        }
    }

    len
}

/// Portable scalar versions
pub fn find_whitespace_scalar(data: &[u8]) -> Option<usize> {
    data.iter()
        .position(|&b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
}

pub fn skip_whitespace_scalar(data: &[u8]) -> usize {
    data.iter()
        .position(|&b| !matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
        .unwrap_or(data.len())
}

/// Auto-dispatching whitespace finder
#[inline]
pub fn find_whitespace(data: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { find_whitespace_avx2(data) };
        }
    }
    find_whitespace_scalar(data)
}

/// Auto-dispatching whitespace skipper
#[inline]
pub fn skip_whitespace(data: &[u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { skip_whitespace_avx2(data) };
        }
    }
    skip_whitespace_scalar(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_whitespace() {
        assert_eq!(find_whitespace(b"hello world"), Some(5));
        assert_eq!(find_whitespace(b"hello"), None);
        assert_eq!(find_whitespace(b" hello"), Some(0));
        assert_eq!(find_whitespace(b"hello\tworld"), Some(5));
    }

    #[test]
    fn test_skip_whitespace() {
        assert_eq!(skip_whitespace(b"  hello"), 2);
        assert_eq!(skip_whitespace(b"hello"), 0);
        assert_eq!(skip_whitespace(b"   "), 3);
        assert_eq!(skip_whitespace(b"\t\n  hello"), 4);
    }

    #[test]
    fn test_classifier_scalar() {
        let classifier = SimdClassifier::new();
        let result = classifier.classify_scalar(b"hello world!");

        // Check whitespace at position 5
        assert_ne!(result.whitespace & (1 << 5), 0);
        // Check punctuation at position 11
        assert_ne!(result.punctuation & (1 << 11), 0);
    }
}
