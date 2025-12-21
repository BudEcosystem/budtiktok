//! Platform-Specific SIMD Backends for Ultra-Fast Tokenization
//!
//! Provides optimized implementations for:
//! - x86_64: SSE4.2, AVX2, AVX-512
//! - ARM: NEON, SVE, SVE2
//!
//! Based on StringZilla's approach of using different algorithms per platform
//! and needle length for optimal performance.

use std::sync::atomic::{AtomicBool, Ordering};

// ============================================================================
// Platform Detection
// ============================================================================

/// Detected SIMD capabilities
#[derive(Debug, Clone, Copy)]
pub struct SimdCapabilities {
    // x86_64
    pub sse42: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vbmi: bool,

    // ARM
    pub neon: bool,
    pub sve: bool,
    pub sve2: bool,
    pub sve_vector_length: usize,
}

impl SimdCapabilities {
    /// Detect SIMD capabilities at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                sse42: is_x86_feature_detected!("sse4.2"),
                avx2: is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                avx512bw: is_x86_feature_detected!("avx512bw"),
                avx512vbmi: is_x86_feature_detected!("avx512vbmi"),
                neon: false,
                sve: false,
                sve2: false,
                sve_vector_length: 0,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on aarch64
            let sve = Self::detect_sve();
            let sve2 = Self::detect_sve2();
            let sve_len = if sve { Self::get_sve_vector_length() } else { 0 };

            Self {
                sse42: false,
                avx2: false,
                avx512f: false,
                avx512bw: false,
                avx512vbmi: false,
                neon: true,
                sve,
                sve2,
                sve_vector_length: sve_len,
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                sse42: false,
                avx2: false,
                avx512f: false,
                avx512bw: false,
                avx512vbmi: false,
                neon: false,
                sve: false,
                sve2: false,
                sve_vector_length: 0,
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_sve() -> bool {
        #[cfg(target_feature = "sve")]
        {
            true
        }
        #[cfg(not(target_feature = "sve"))]
        {
            // Runtime detection via HWCAP
            #[cfg(target_os = "linux")]
            {
                const AT_HWCAP: u64 = 16;
                const HWCAP_SVE: u64 = 1 << 22;

                unsafe {
                    let hwcap = libc::getauxval(AT_HWCAP as libc::c_ulong);
                    (hwcap & HWCAP_SVE) != 0
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                false
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_sve2() -> bool {
        #[cfg(target_feature = "sve2")]
        {
            true
        }
        #[cfg(not(target_feature = "sve2"))]
        {
            #[cfg(target_os = "linux")]
            {
                const AT_HWCAP2: u64 = 26;
                const HWCAP2_SVE2: u64 = 1 << 1;

                unsafe {
                    let hwcap2 = libc::getauxval(AT_HWCAP2 as libc::c_ulong);
                    (hwcap2 & HWCAP2_SVE2) != 0
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                false
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn get_sve_vector_length() -> usize {
        #[cfg(target_feature = "sve")]
        {
            use std::arch::aarch64::*;
            unsafe { svcntb() }
        }
        #[cfg(not(target_feature = "sve"))]
        {
            // Read from /proc/sys/abi/sve_default_vector_length on Linux
            #[cfg(target_os = "linux")]
            {
                std::fs::read_to_string("/proc/sys/abi/sve_default_vector_length")
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0)
            }
            #[cfg(not(target_os = "linux"))]
            {
                0
            }
        }
    }

    /// Get the best available backend name
    pub fn best_backend(&self) -> &'static str {
        #[cfg(target_arch = "x86_64")]
        {
            if self.avx512bw {
                "AVX-512"
            } else if self.avx2 {
                "AVX2"
            } else if self.sse42 {
                "SSE4.2"
            } else {
                "Scalar"
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.sve2 {
                "SVE2"
            } else if self.sve {
                "SVE"
            } else {
                "NEON"
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            "Scalar"
        }
    }
}

// Global cached capabilities
static CAPABILITIES_DETECTED: AtomicBool = AtomicBool::new(false);
static mut CAPABILITIES: Option<SimdCapabilities> = None;

/// Get cached SIMD capabilities
pub fn get_capabilities() -> SimdCapabilities {
    if !CAPABILITIES_DETECTED.load(Ordering::Relaxed) {
        let caps = SimdCapabilities::detect();
        unsafe {
            CAPABILITIES = Some(caps);
        }
        CAPABILITIES_DETECTED.store(true, Ordering::Release);
    }
    unsafe { CAPABILITIES.unwrap() }
}

// ============================================================================
// x86_64 Implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use std::arch::x86_64::*;

    /// Find first occurrence of any byte from needle in haystack (AVX2)
    #[target_feature(enable = "avx2")]
    pub unsafe fn find_first_of_avx2(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        if haystack.is_empty() || needle.is_empty() {
            return None;
        }

        let len = haystack.len();
        let mut offset = 0;

        // Build needle mask - broadcast each needle byte and combine
        // For small needles, this is very efficient
        if needle.len() <= 4 {
            let n0 = _mm256_set1_epi8(needle[0] as i8);
            let n1 = if needle.len() > 1 { _mm256_set1_epi8(needle[1] as i8) } else { n0 };
            let n2 = if needle.len() > 2 { _mm256_set1_epi8(needle[2] as i8) } else { n0 };
            let n3 = if needle.len() > 3 { _mm256_set1_epi8(needle[3] as i8) } else { n0 };

            while offset + 32 <= len {
                let chunk = _mm256_loadu_si256(haystack.as_ptr().add(offset) as *const __m256i);

                let m0 = _mm256_cmpeq_epi8(chunk, n0);
                let m1 = _mm256_cmpeq_epi8(chunk, n1);
                let m2 = _mm256_cmpeq_epi8(chunk, n2);
                let m3 = _mm256_cmpeq_epi8(chunk, n3);

                let combined = _mm256_or_si256(
                    _mm256_or_si256(m0, m1),
                    _mm256_or_si256(m2, m3)
                );

                let mask = _mm256_movemask_epi8(combined) as u32;
                if mask != 0 {
                    return Some(offset + mask.trailing_zeros() as usize);
                }

                offset += 32;
            }
        }

        // Scalar fallback for remaining bytes
        for i in offset..len {
            if needle.contains(&haystack[i]) {
                return Some(i);
            }
        }

        None
    }

    /// Find first occurrence of any whitespace character (AVX2)
    #[target_feature(enable = "avx2")]
    pub unsafe fn find_whitespace_avx2(data: &[u8]) -> Option<usize> {
        let len = data.len();
        let mut offset = 0;

        let space = _mm256_set1_epi8(b' ' as i8);
        let tab = _mm256_set1_epi8(b'\t' as i8);
        let newline = _mm256_set1_epi8(b'\n' as i8);
        let carriage = _mm256_set1_epi8(b'\r' as i8);

        while offset + 32 <= len {
            let chunk = _mm256_loadu_si256(data.as_ptr().add(offset) as *const __m256i);

            let m_space = _mm256_cmpeq_epi8(chunk, space);
            let m_tab = _mm256_cmpeq_epi8(chunk, tab);
            let m_newline = _mm256_cmpeq_epi8(chunk, newline);
            let m_carriage = _mm256_cmpeq_epi8(chunk, carriage);

            let combined = _mm256_or_si256(
                _mm256_or_si256(m_space, m_tab),
                _mm256_or_si256(m_newline, m_carriage)
            );

            let mask = _mm256_movemask_epi8(combined) as u32;
            if mask != 0 {
                return Some(offset + mask.trailing_zeros() as usize);
            }

            offset += 32;
        }

        // Scalar fallback
        for i in offset..len {
            match data[i] {
                b' ' | b'\t' | b'\n' | b'\r' => return Some(i),
                _ => {}
            }
        }

        None
    }

    /// Skip whitespace using AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn skip_whitespace_avx2(data: &[u8]) -> usize {
        let len = data.len();
        let mut offset = 0;

        let space = _mm256_set1_epi8(b' ' as i8);
        let tab = _mm256_set1_epi8(b'\t' as i8);
        let newline = _mm256_set1_epi8(b'\n' as i8);
        let carriage = _mm256_set1_epi8(b'\r' as i8);

        while offset + 32 <= len {
            let chunk = _mm256_loadu_si256(data.as_ptr().add(offset) as *const __m256i);

            let m_space = _mm256_cmpeq_epi8(chunk, space);
            let m_tab = _mm256_cmpeq_epi8(chunk, tab);
            let m_newline = _mm256_cmpeq_epi8(chunk, newline);
            let m_carriage = _mm256_cmpeq_epi8(chunk, carriage);

            let ws = _mm256_or_si256(
                _mm256_or_si256(m_space, m_tab),
                _mm256_or_si256(m_newline, m_carriage)
            );

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

        // Scalar fallback
        for i in offset..len {
            match data[i] {
                b' ' | b'\t' | b'\n' | b'\r' => {}
                _ => return i,
            }
        }

        len
    }

    /// Vectorized lowercase conversion (AVX2)
    #[target_feature(enable = "avx2")]
    pub unsafe fn to_lowercase_avx2(data: &mut [u8]) {
        let len = data.len();
        let mut offset = 0;

        let a_upper = _mm256_set1_epi8(b'A' as i8);
        let z_upper = _mm256_set1_epi8(b'Z' as i8);
        let case_bit = _mm256_set1_epi8(0x20);

        while offset + 32 <= len {
            let chunk = _mm256_loadu_si256(data.as_ptr().add(offset) as *const __m256i);

            // Check if in range A-Z
            let ge_a = _mm256_cmpgt_epi8(chunk, _mm256_sub_epi8(a_upper, _mm256_set1_epi8(1)));
            let le_z = _mm256_cmpgt_epi8(_mm256_add_epi8(z_upper, _mm256_set1_epi8(1)), chunk);
            let is_upper = _mm256_and_si256(ge_a, le_z);

            // Add 0x20 to uppercase letters
            let to_add = _mm256_and_si256(is_upper, case_bit);
            let result = _mm256_add_epi8(chunk, to_add);

            _mm256_storeu_si256(data.as_mut_ptr().add(offset) as *mut __m256i, result);

            offset += 32;
        }

        // Scalar fallback
        for byte in &mut data[offset..] {
            if *byte >= b'A' && *byte <= b'Z' {
                *byte += 0x20;
            }
        }
    }

    /// Parallel string comparison (AVX2) - returns matching length
    #[target_feature(enable = "avx2")]
    pub unsafe fn compare_strings_avx2(a: &[u8], b: &[u8]) -> usize {
        let len = a.len().min(b.len());
        let mut offset = 0;

        while offset + 32 <= len {
            let chunk_a = _mm256_loadu_si256(a.as_ptr().add(offset) as *const __m256i);
            let chunk_b = _mm256_loadu_si256(b.as_ptr().add(offset) as *const __m256i);

            let cmp = _mm256_cmpeq_epi8(chunk_a, chunk_b);
            let mask = _mm256_movemask_epi8(cmp) as u32;

            if mask != 0xFFFFFFFF {
                // Found mismatch
                return offset + (!mask).trailing_zeros() as usize;
            }

            offset += 32;
        }

        // Scalar fallback
        for i in offset..len {
            if a[i] != b[i] {
                return i;
            }
        }

        len
    }

    /// AVX-512 find whitespace (even faster with 64-byte chunks)
    #[cfg(all(target_feature = "avx512f", feature = "nightly"))]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    pub unsafe fn find_whitespace_avx512(data: &[u8]) -> Option<usize> {
        let len = data.len();
        let mut offset = 0;

        let space = _mm512_set1_epi8(b' ' as i8);
        let tab = _mm512_set1_epi8(b'\t' as i8);
        let newline = _mm512_set1_epi8(b'\n' as i8);
        let carriage = _mm512_set1_epi8(b'\r' as i8);

        while offset + 64 <= len {
            let chunk = _mm512_loadu_si512(data.as_ptr().add(offset) as *const i32);

            let m_space = _mm512_cmpeq_epi8_mask(chunk, space);
            let m_tab = _mm512_cmpeq_epi8_mask(chunk, tab);
            let m_newline = _mm512_cmpeq_epi8_mask(chunk, newline);
            let m_carriage = _mm512_cmpeq_epi8_mask(chunk, carriage);

            let mask = m_space | m_tab | m_newline | m_carriage;
            if mask != 0 {
                return Some(offset + mask.trailing_zeros() as usize);
            }

            offset += 64;
        }

        // Fall back to AVX2 for remaining bytes
        if offset < len {
            if let Some(pos) = find_whitespace_avx2(&data[offset..]) {
                return Some(offset + pos);
            }
        }

        None
    }
}

// ============================================================================
// ARM Implementations
// ============================================================================

#[cfg(target_arch = "aarch64")]
pub mod arm {
    use std::arch::aarch64::*;

    /// Find first whitespace using NEON
    #[inline]
    pub unsafe fn find_whitespace_neon(data: &[u8]) -> Option<usize> {
        let len = data.len();
        let mut offset = 0;

        let space = vdupq_n_u8(b' ');
        let tab = vdupq_n_u8(b'\t');
        let newline = vdupq_n_u8(b'\n');
        let carriage = vdupq_n_u8(b'\r');

        while offset + 16 <= len {
            let chunk = vld1q_u8(data.as_ptr().add(offset));

            let m_space = vceqq_u8(chunk, space);
            let m_tab = vceqq_u8(chunk, tab);
            let m_newline = vceqq_u8(chunk, newline);
            let m_carriage = vceqq_u8(chunk, carriage);

            let ws = vorrq_u8(
                vorrq_u8(m_space, m_tab),
                vorrq_u8(m_newline, m_carriage)
            );

            // Check if any matches
            let max = vmaxvq_u8(ws);
            if max != 0 {
                // Find first match position
                let mask = neon_to_bitmask(ws);
                if mask != 0 {
                    return Some(offset + mask.trailing_zeros() as usize);
                }
            }

            offset += 16;
        }

        // Scalar fallback
        for i in offset..len {
            match data[i] {
                b' ' | b'\t' | b'\n' | b'\r' => return Some(i),
                _ => {}
            }
        }

        None
    }

    /// Skip whitespace using NEON
    #[inline]
    pub unsafe fn skip_whitespace_neon(data: &[u8]) -> usize {
        let len = data.len();
        let mut offset = 0;

        let space = vdupq_n_u8(b' ');
        let tab = vdupq_n_u8(b'\t');
        let newline = vdupq_n_u8(b'\n');
        let carriage = vdupq_n_u8(b'\r');

        while offset + 16 <= len {
            let chunk = vld1q_u8(data.as_ptr().add(offset));

            let m_space = vceqq_u8(chunk, space);
            let m_tab = vceqq_u8(chunk, tab);
            let m_newline = vceqq_u8(chunk, newline);
            let m_carriage = vceqq_u8(chunk, carriage);

            let ws = vorrq_u8(
                vorrq_u8(m_space, m_tab),
                vorrq_u8(m_newline, m_carriage)
            );

            // Check if all whitespace
            let min = vminvq_u8(ws);
            if min == 0 {
                // Not all whitespace - find first non-whitespace
                let not_ws = vmvnq_u8(ws);
                let mask = neon_to_bitmask(not_ws);
                if mask != 0 {
                    return offset + mask.trailing_zeros() as usize;
                }
            }

            offset += 16;
        }

        // Scalar fallback
        for i in offset..len {
            match data[i] {
                b' ' | b'\t' | b'\n' | b'\r' => {}
                _ => return i,
            }
        }

        len
    }

    /// Vectorized lowercase conversion (NEON)
    #[inline]
    pub unsafe fn to_lowercase_neon(data: &mut [u8]) {
        let len = data.len();
        let mut offset = 0;

        let a_upper = vdupq_n_u8(b'A');
        let z_upper = vdupq_n_u8(b'Z');
        let case_bit = vdupq_n_u8(0x20);

        while offset + 16 <= len {
            let chunk = vld1q_u8(data.as_ptr().add(offset));

            // Check if in range A-Z
            let ge_a = vcgeq_u8(chunk, a_upper);
            let le_z = vcleq_u8(chunk, z_upper);
            let is_upper = vandq_u8(ge_a, le_z);

            // Add 0x20 to uppercase letters
            let to_add = vandq_u8(is_upper, case_bit);
            let result = vaddq_u8(chunk, to_add);

            vst1q_u8(data.as_mut_ptr().add(offset), result);

            offset += 16;
        }

        // Scalar fallback
        for byte in &mut data[offset..] {
            if *byte >= b'A' && *byte <= b'Z' {
                *byte += 0x20;
            }
        }
    }

    /// Convert NEON comparison result to bitmask
    #[inline]
    unsafe fn neon_to_bitmask(v: uint8x16_t) -> u16 {
        // Use SHRN to extract high bits
        let high_bits = vshrn_n_u16(vreinterpretq_u16_u8(v), 4);

        // Use lookup table to pack bits
        let shifted = vshl_n_u8(high_bits, 0);

        // Manual extraction for each byte
        let mut mask = 0u16;
        let arr: [u8; 8] = std::mem::transmute(shifted);
        for i in 0..8 {
            if arr[i] != 0 {
                mask |= 1 << (i * 2);
            }
        }

        // Simple approach: check each byte
        let bytes: [u8; 16] = std::mem::transmute(v);
        let mut result = 0u16;
        for (i, &b) in bytes.iter().enumerate() {
            if b != 0 {
                result |= 1 << i;
            }
        }
        result
    }

    // SVE implementations would go here when targeting SVE-capable hardware
    // SVE provides predicated operations which are excellent for string processing
}

// ============================================================================
// Unified Dispatcher
// ============================================================================

/// Find first whitespace character
#[inline]
pub fn find_whitespace(data: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { x86::find_whitespace_avx2(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { arm::find_whitespace_neon(data) };
    }

    // Scalar fallback
    data.iter().position(|&b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
}

/// Skip leading whitespace
#[inline]
pub fn skip_whitespace(data: &[u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { x86::skip_whitespace_avx2(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { arm::skip_whitespace_neon(data) };
    }

    // Scalar fallback
    data.iter()
        .position(|&b| !matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
        .unwrap_or(data.len())
}

/// Convert to lowercase in-place
#[inline]
pub fn to_lowercase_inplace(data: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { x86::to_lowercase_avx2(data) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { arm::to_lowercase_neon(data) };
        return;
    }

    // Scalar fallback
    for byte in data {
        if *byte >= b'A' && *byte <= b'Z' {
            *byte += 0x20;
        }
    }
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
    fn test_to_lowercase() {
        let mut data = b"Hello WORLD".to_vec();
        to_lowercase_inplace(&mut data);
        assert_eq!(&data, b"hello world");

        let mut data = b"already lowercase".to_vec();
        to_lowercase_inplace(&mut data);
        assert_eq!(&data, b"already lowercase");
    }

    #[test]
    fn test_capabilities() {
        let caps = get_capabilities();
        println!("SIMD capabilities: {:?}", caps);
        println!("Best backend: {}", caps.best_backend());
    }
}
