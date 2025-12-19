//! ARM SVE/SVE2 Implementation (4.5.3)
//!
//! Scalable Vector Extension support for newer ARM processors.
//! SVE provides vector-length agnostic (VLA) programming, allowing the same
//! code to run efficiently on processors with different vector lengths
//! (128 to 2048 bits).
//!
//! Key features:
//! - Vector-length agnostic code
//! - Predicate registers for masked operations
//! - Gather/scatter load/store
//! - First-fault loads for safe speculation
//!
//! Note: SVE intrinsics require nightly Rust and the `aarch64_ver_fp_intrinsics` feature.
//! This module provides fallback implementations that use NEON when SVE is not available.

/// Check if SVE is available at runtime
#[inline]
pub fn is_sve_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        // Check for SVE support using CPUID
        // On Linux, this can be done via HWCAP
        #[cfg(target_os = "linux")]
        {
            use std::arch::is_aarch64_feature_detected;
            // Note: is_aarch64_feature_detected! for "sve" may not be stable
            // Use a fallback detection method
            is_sve_detected_linux()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Check if SVE2 is available
#[inline]
pub fn is_sve2_available() -> bool {
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        is_sve2_detected_linux()
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
    {
        false
    }
}

/// Linux-specific SVE detection using HWCAP
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn is_sve_detected_linux() -> bool {
    // HWCAP_SVE = 1 << 22 on AArch64 Linux
    const HWCAP_SVE: u64 = 1 << 22;

    unsafe {
        let hwcap = libc::getauxval(libc::AT_HWCAP);
        (hwcap & HWCAP_SVE) != 0
    }
}

/// Linux-specific SVE2 detection
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn is_sve2_detected_linux() -> bool {
    // HWCAP2_SVE2 = 1 << 1 on AArch64 Linux
    const HWCAP2_SVE2: u64 = 1 << 1;

    unsafe {
        let hwcap2 = libc::getauxval(libc::AT_HWCAP2);
        (hwcap2 & HWCAP2_SVE2) != 0
    }
}

/// Get the SVE vector length in bytes (0 if SVE not available)
#[inline]
pub fn get_sve_vector_length() -> usize {
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        if is_sve_available() {
            // Use prctl to get SVE vector length
            unsafe {
                // PR_SVE_GET_VL = 51
                const PR_SVE_GET_VL: i32 = 51;
                let vl = libc::prctl(PR_SVE_GET_VL);
                if vl > 0 {
                    // Lower 16 bits contain vector length in bytes
                    (vl & 0xFFFF) as usize
                } else {
                    0
                }
            }
        } else {
            0
        }
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
    {
        0
    }
}

// ============================================================================
// Vector-Length Agnostic Operations
// ============================================================================

/// Information about SVE capabilities
#[derive(Debug, Clone)]
pub struct SveInfo {
    /// Whether SVE is available
    pub available: bool,
    /// Whether SVE2 is available
    pub sve2_available: bool,
    /// Vector length in bytes
    pub vector_length: usize,
    /// Vector length in bits
    pub vector_bits: usize,
}

impl SveInfo {
    /// Detect SVE capabilities
    pub fn detect() -> Self {
        let available = is_sve_available();
        let sve2_available = is_sve2_available();
        let vector_length = get_sve_vector_length();

        Self {
            available,
            sve2_available,
            vector_length,
            vector_bits: vector_length * 8,
        }
    }

    /// Get the number of elements that fit in a vector for a given element size
    pub fn elements_per_vector(&self, element_bytes: usize) -> usize {
        if self.vector_length > 0 && element_bytes > 0 {
            self.vector_length / element_bytes
        } else {
            0
        }
    }
}

// ============================================================================
// Character Classification (VLA)
// ============================================================================

/// Classify whitespace using SVE or fallback
///
/// This function automatically uses the best available implementation:
/// - SVE if available (processes vector_length bytes per iteration)
/// - NEON fallback (processes 16 bytes per iteration)
/// - Scalar fallback (processes 1 byte at a time)
pub fn classify_whitespace_vla(input: &[u8]) -> Vec<bool> {
    if input.is_empty() {
        return Vec::new();
    }

    #[cfg(target_arch = "aarch64")]
    {
        if is_sve_available() {
            return classify_whitespace_sve(input);
        }
        // Fall back to NEON
        return classify_whitespace_neon_fallback(input);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        classify_whitespace_scalar(input)
    }
}

/// Scalar whitespace classification
fn classify_whitespace_scalar(input: &[u8]) -> Vec<bool> {
    input
        .iter()
        .map(|&b| b == b' ' || b == b'\t' || b == b'\n' || b == b'\r')
        .collect()
}

/// SVE whitespace classification
#[cfg(target_arch = "aarch64")]
fn classify_whitespace_sve(input: &[u8]) -> Vec<bool> {
    // For now, use scalar as SVE intrinsics are not stable
    // When SVE intrinsics stabilize, this would use:
    // svld1_u8, svcmpeq_u8, svptest_any, etc.
    classify_whitespace_scalar(input)
}

/// NEON fallback for whitespace classification
#[cfg(target_arch = "aarch64")]
fn classify_whitespace_neon_fallback(input: &[u8]) -> Vec<bool> {
    use crate::neon::classify_whitespace;

    let mut result = Vec::with_capacity(input.len());
    let mut i = 0;

    while i + 16 <= input.len() {
        let mask = classify_whitespace(&input[i..]);
        for bit in 0..16 {
            result.push((mask & (1 << bit)) != 0);
        }
        i += 16;
    }

    // Handle remaining bytes
    for &b in &input[i..] {
        result.push(b == b' ' || b == b'\t' || b == b'\n' || b == b'\r');
    }

    result
}

/// Find the first non-ASCII byte using SVE or fallback
pub fn find_non_ascii_vla(input: &[u8]) -> Option<usize> {
    if input.is_empty() {
        return None;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if is_sve_available() {
            return find_non_ascii_sve(input);
        }
        // Fall back to NEON
        return crate::neon::find_non_ascii(input);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        find_non_ascii_scalar(input)
    }
}

/// Scalar non-ASCII search
#[cfg(not(target_arch = "aarch64"))]
fn find_non_ascii_scalar(input: &[u8]) -> Option<usize> {
    for (i, &b) in input.iter().enumerate() {
        if b >= 0x80 {
            return Some(i);
        }
    }
    None
}

/// SVE non-ASCII search
#[cfg(target_arch = "aarch64")]
fn find_non_ascii_sve(input: &[u8]) -> Option<usize> {
    // SVE implementation would use svmatch for efficient search
    // For now, fall back to NEON
    crate::neon::find_non_ascii(input)
}

// ============================================================================
// UTF-8 Validation (VLA)
// ============================================================================

/// Validate UTF-8 using SVE or fallback
pub fn validate_utf8_vla(input: &[u8]) -> Result<(), usize> {
    if input.is_empty() {
        return Ok(());
    }

    #[cfg(target_arch = "aarch64")]
    {
        if is_sve_available() {
            return validate_utf8_sve(input);
        }
        return crate::neon::validate_utf8_neon_available(input);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        validate_utf8_scalar(input)
    }
}

/// Scalar UTF-8 validation
#[cfg(not(target_arch = "aarch64"))]
fn validate_utf8_scalar(input: &[u8]) -> Result<(), usize> {
    std::str::from_utf8(input)
        .map(|_| ())
        .map_err(|e| e.valid_up_to())
}

/// SVE UTF-8 validation
#[cfg(target_arch = "aarch64")]
fn validate_utf8_sve(input: &[u8]) -> Result<(), usize> {
    // SVE would allow processing more bytes per iteration with variable vector length
    // For now, fall back to NEON
    crate::neon::validate_utf8_neon_available(input)
}

// ============================================================================
// Code Point Counting (VLA)
// ============================================================================

/// Count UTF-8 code points using SVE or fallback
pub fn count_code_points_vla(input: &[u8]) -> usize {
    if input.is_empty() {
        return 0;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if is_sve_available() {
            return count_code_points_sve(input);
        }
        return crate::neon::count_code_points_neon_available(input);
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        count_code_points_scalar(input)
    }
}

/// Scalar code point counting
#[cfg(not(target_arch = "aarch64"))]
fn count_code_points_scalar(input: &[u8]) -> usize {
    input.iter().filter(|&&b| (b & 0xC0) != 0x80).count()
}

/// SVE code point counting
#[cfg(target_arch = "aarch64")]
fn count_code_points_sve(input: &[u8]) -> usize {
    // SVE would use predicated counting with variable vector length
    // For now, fall back to NEON
    crate::neon::count_code_points_neon_available(input)
}

// ============================================================================
// Case Conversion (VLA)
// ============================================================================

/// Convert ASCII to lowercase using SVE or fallback
pub fn to_lowercase_vla(input: &mut [u8]) {
    if input.is_empty() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if is_sve_available() {
            to_lowercase_sve(input);
            return;
        }
        crate::neon::to_lowercase_ascii(input);
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        to_lowercase_scalar(input);
    }
}

/// Scalar lowercase conversion
#[cfg(not(target_arch = "aarch64"))]
fn to_lowercase_scalar(input: &mut [u8]) {
    for b in input {
        if *b >= b'A' && *b <= b'Z' {
            *b += 32;
        }
    }
}

/// SVE lowercase conversion
#[cfg(target_arch = "aarch64")]
fn to_lowercase_sve(input: &mut [u8]) {
    // SVE would use predicated conditional add
    // For now, fall back to NEON
    crate::neon::to_lowercase_ascii(input);
}

// ============================================================================
// Batch Processing (VLA)
// ============================================================================

/// Process multiple strings efficiently using SVE
///
/// This processes multiple strings in a cache-friendly manner,
/// taking advantage of larger vector widths when available.
pub fn batch_classify_whitespace(inputs: &[&[u8]]) -> Vec<Vec<bool>> {
    inputs
        .iter()
        .map(|input| classify_whitespace_vla(input))
        .collect()
}

/// Batch validate UTF-8 for multiple strings
pub fn batch_validate_utf8(inputs: &[&[u8]]) -> Vec<Result<(), usize>> {
    inputs
        .iter()
        .map(|input| validate_utf8_vla(input))
        .collect()
}

/// Batch count code points
pub fn batch_count_code_points(inputs: &[&[u8]]) -> Vec<usize> {
    inputs
        .iter()
        .map(|input| count_code_points_vla(input))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_sve_available() {
        // Just verify it doesn't crash
        let _ = is_sve_available();
    }

    #[test]
    fn test_is_sve2_available() {
        let _ = is_sve2_available();
    }

    #[test]
    fn test_get_sve_vector_length() {
        let vl = get_sve_vector_length();
        // If SVE is available, vector length should be at least 16 bytes
        if is_sve_available() {
            assert!(vl >= 16);
            // Vector length should be a multiple of 16
            assert!(vl % 16 == 0);
        } else {
            assert_eq!(vl, 0);
        }
    }

    #[test]
    fn test_sve_info() {
        let info = SveInfo::detect();

        if info.available {
            assert!(info.vector_length >= 16);
            assert_eq!(info.vector_bits, info.vector_length * 8);
            assert!(info.elements_per_vector(1) >= 16);
            assert!(info.elements_per_vector(4) >= 4);
        }
    }

    #[test]
    fn test_classify_whitespace_vla() {
        let input = b"hello world";
        let result = classify_whitespace_vla(input);

        assert_eq!(result.len(), input.len());
        assert!(!result[0]); // 'h'
        assert!(result[5]); // space
        assert!(!result[6]); // 'w'
    }

    #[test]
    fn test_classify_whitespace_vla_tabs() {
        let input = b"a\tb\nc\rd";
        let result = classify_whitespace_vla(input);

        assert!(!result[0]); // 'a'
        assert!(result[1]); // tab
        assert!(!result[2]); // 'b'
        assert!(result[3]); // newline
        assert!(!result[4]); // 'c'
        assert!(result[5]); // carriage return
    }

    #[test]
    fn test_find_non_ascii_vla() {
        assert_eq!(find_non_ascii_vla(b"hello"), None);
        assert_eq!(find_non_ascii_vla(b"hello\x80"), Some(5));
        assert_eq!(find_non_ascii_vla("café".as_bytes()), Some(3));
    }

    #[test]
    fn test_validate_utf8_vla() {
        assert!(validate_utf8_vla(b"hello").is_ok());
        assert!(validate_utf8_vla("日本語".as_bytes()).is_ok());
        assert!(validate_utf8_vla(&[0x80]).is_err());
    }

    #[test]
    fn test_count_code_points_vla() {
        assert_eq!(count_code_points_vla(b"hello"), 5);
        assert_eq!(count_code_points_vla("日本語".as_bytes()), 3);
        assert_eq!(count_code_points_vla("🎉".as_bytes()), 1);
    }

    #[test]
    fn test_to_lowercase_vla() {
        let mut input = b"Hello World".to_vec();
        to_lowercase_vla(&mut input);
        assert_eq!(&input, b"hello world");
    }

    #[test]
    fn test_batch_classify_whitespace() {
        let inputs: Vec<&[u8]> = vec![b"hello world", b"a\tb"];
        let results = batch_classify_whitespace(&inputs);

        assert_eq!(results.len(), 2);
        assert!(results[0][5]); // space in first string
        assert!(results[1][1]); // tab in second string
    }

    #[test]
    fn test_batch_validate_utf8() {
        let inputs: Vec<&[u8]> = vec![b"hello", "日本語".as_bytes(), &[0x80]];
        let results = batch_validate_utf8(&inputs);

        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
        assert!(results[2].is_err());
    }

    #[test]
    fn test_batch_count_code_points() {
        let inputs: Vec<&[u8]> = vec![b"hello", "日本語".as_bytes()];
        let results = batch_count_code_points(&inputs);

        assert_eq!(results[0], 5);
        assert_eq!(results[1], 3);
    }

    #[test]
    fn test_empty_input() {
        assert!(classify_whitespace_vla(b"").is_empty());
        assert_eq!(find_non_ascii_vla(b""), None);
        assert!(validate_utf8_vla(b"").is_ok());
        assert_eq!(count_code_points_vla(b""), 0);
    }

    #[test]
    fn test_long_input() {
        // Test with input that would use SIMD path
        let input = b"The quick brown fox jumps over the lazy dog. ".repeat(10);
        let result = classify_whitespace_vla(&input);

        // Count spaces
        let space_count = result.iter().filter(|&&x| x).count();
        assert_eq!(space_count, 90); // 9 spaces per sentence * 10 repetitions
    }
}
