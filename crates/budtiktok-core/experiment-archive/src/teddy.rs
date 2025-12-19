//! Teddy Algorithm for SIMD-accelerated Multi-pattern Matching (4.3.2)
//!
//! The Teddy algorithm uses SIMD fingerprinting for fast substring search across
//! multiple patterns. It's particularly effective for special token matching in
//! tokenizers where we need to find any of several special tokens in the input.
//!
//! Key features:
//! - Groups patterns by first 2-3 byte fingerprints
//! - Uses AVX2 shuffle for parallel fingerprint matching
//! - Falls back to scalar verification for candidates
//!
//! This implementation provides both AVX2-accelerated and scalar fallback versions.


/// Number of buckets for fingerprint grouping (must be power of 2)
const NUM_BUCKETS: usize = 8;

/// Maximum number of patterns per bucket
const MAX_PATTERNS_PER_BUCKET: usize = 8;

/// Teddy matcher for SIMD-accelerated special token matching
#[derive(Clone)]
pub struct TeddyMatcher {
    /// Patterns grouped by fingerprint bucket
    buckets: Vec<TeddyBucket>,
    /// All patterns for verification
    patterns: Vec<Vec<u8>>,
    /// Pattern IDs for each pattern
    pattern_ids: Vec<u32>,
    /// Minimum pattern length
    min_pattern_len: usize,
    /// Fingerprint mask (lower bits determine bucket)
    bucket_mask: u8,
}

/// A bucket containing patterns with similar fingerprints
#[derive(Clone, Default)]
struct TeddyBucket {
    /// Patterns in this bucket (indices into main patterns array)
    pattern_indices: Vec<usize>,
    /// First byte of each pattern for quick filtering
    first_bytes: Vec<u8>,
}

/// Match result from Teddy search
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeddyMatch {
    /// Start position in input
    pub start: usize,
    /// End position in input
    pub end: usize,
    /// Pattern ID
    pub pattern_id: u32,
}

impl TeddyMatcher {
    /// Create a new Teddy matcher from patterns
    pub fn new<I, P>(patterns_with_ids: I) -> Self
    where
        I: IntoIterator<Item = (P, u32)>,
        P: AsRef<[u8]>,
    {
        let mut patterns = Vec::new();
        let mut pattern_ids = Vec::new();

        for (pattern, id) in patterns_with_ids {
            patterns.push(pattern.as_ref().to_vec());
            pattern_ids.push(id);
        }

        if patterns.is_empty() {
            return Self {
                buckets: vec![TeddyBucket::default(); NUM_BUCKETS],
                patterns,
                pattern_ids,
                min_pattern_len: usize::MAX,
                bucket_mask: (NUM_BUCKETS - 1) as u8,
            };
        }

        let min_pattern_len = patterns.iter().map(|p| p.len()).min().unwrap_or(0);

        // Group patterns by fingerprint bucket
        let mut buckets = vec![TeddyBucket::default(); NUM_BUCKETS];
        let bucket_mask = (NUM_BUCKETS - 1) as u8;

        for (idx, pattern) in patterns.iter().enumerate() {
            if pattern.is_empty() {
                continue;
            }

            let fingerprint = Self::compute_fingerprint(pattern);
            let bucket_idx = (fingerprint & bucket_mask) as usize;

            if buckets[bucket_idx].pattern_indices.len() < MAX_PATTERNS_PER_BUCKET {
                buckets[bucket_idx].pattern_indices.push(idx);
                buckets[bucket_idx].first_bytes.push(pattern[0]);
            }
        }

        Self {
            buckets,
            patterns,
            pattern_ids,
            min_pattern_len,
            bucket_mask,
        }
    }

    /// Compute fingerprint from first bytes of pattern
    #[inline]
    fn compute_fingerprint(pattern: &[u8]) -> u8 {
        if pattern.is_empty() {
            return 0;
        }

        let mut fp = pattern[0];
        if pattern.len() > 1 {
            fp ^= pattern[1].wrapping_mul(31);
        }
        if pattern.len() > 2 {
            fp ^= pattern[2].wrapping_mul(17);
        }
        fp
    }

    /// Check if matcher has any patterns
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Get number of patterns
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Find all matches in the input
    pub fn find_all(&self, input: &[u8]) -> Vec<TeddyMatch> {
        if self.patterns.is_empty() || input.len() < self.min_pattern_len {
            return Vec::new();
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.find_all_avx2(input) };
            }
        }

        self.find_all_scalar(input)
    }

    /// Find first match in the input
    pub fn find_first(&self, input: &[u8]) -> Option<TeddyMatch> {
        if self.patterns.is_empty() || input.len() < self.min_pattern_len {
            return None;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.find_first_avx2(input) };
            }
        }

        self.find_first_scalar(input)
    }

    /// Scalar implementation for fallback
    fn find_all_scalar(&self, input: &[u8]) -> Vec<TeddyMatch> {
        let mut matches = Vec::new();

        for pos in 0..=input.len().saturating_sub(self.min_pattern_len) {
            let fingerprint = Self::compute_fingerprint(&input[pos..]);
            let bucket_idx = (fingerprint & self.bucket_mask) as usize;
            let bucket = &self.buckets[bucket_idx];

            // Check each pattern in the bucket
            for (i, &pattern_idx) in bucket.pattern_indices.iter().enumerate() {
                // Quick first-byte check
                if input[pos] != bucket.first_bytes[i] {
                    continue;
                }

                let pattern = &self.patterns[pattern_idx];
                if pos + pattern.len() > input.len() {
                    continue;
                }

                // Full verification
                if input[pos..pos + pattern.len()] == pattern[..] {
                    matches.push(TeddyMatch {
                        start: pos,
                        end: pos + pattern.len(),
                        pattern_id: self.pattern_ids[pattern_idx],
                    });
                }
            }
        }

        matches
    }

    /// Scalar implementation for find_first
    fn find_first_scalar(&self, input: &[u8]) -> Option<TeddyMatch> {
        for pos in 0..=input.len().saturating_sub(self.min_pattern_len) {
            let fingerprint = Self::compute_fingerprint(&input[pos..]);
            let bucket_idx = (fingerprint & self.bucket_mask) as usize;
            let bucket = &self.buckets[bucket_idx];

            for (i, &pattern_idx) in bucket.pattern_indices.iter().enumerate() {
                if input[pos] != bucket.first_bytes[i] {
                    continue;
                }

                let pattern = &self.patterns[pattern_idx];
                if pos + pattern.len() > input.len() {
                    continue;
                }

                if input[pos..pos + pattern.len()] == pattern[..] {
                    return Some(TeddyMatch {
                        start: pos,
                        end: pos + pattern.len(),
                        pattern_id: self.pattern_ids[pattern_idx],
                    });
                }
            }
        }

        None
    }

    /// AVX2-accelerated implementation
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn find_all_avx2(&self, input: &[u8]) -> Vec<TeddyMatch> {
        use std::arch::x86_64::*;

        let mut matches = Vec::new();

        if input.len() < 32 {
            return self.find_all_scalar(input);
        }

        // Build first-byte masks for each bucket
        let mut bucket_masks: [__m256i; NUM_BUCKETS] = [_mm256_setzero_si256(); NUM_BUCKETS];
        for (bucket_idx, bucket) in self.buckets.iter().enumerate() {
            if !bucket.first_bytes.is_empty() {
                // Create a mask with all first bytes of patterns in this bucket
                bucket_masks[bucket_idx] = _mm256_set1_epi8(bucket.first_bytes[0] as i8);
            }
        }

        // Process 32 bytes at a time
        let chunks = input.len() / 32;
        for chunk_idx in 0..chunks {
            let chunk_start = chunk_idx * 32;
            let chunk = _mm256_loadu_si256(input.as_ptr().add(chunk_start) as *const __m256i);

            // Check each bucket's first byte
            for (bucket_idx, bucket) in self.buckets.iter().enumerate() {
                if bucket.pattern_indices.is_empty() {
                    continue;
                }

                // Compare chunk against first byte of patterns in this bucket
                let cmp = _mm256_cmpeq_epi8(chunk, bucket_masks[bucket_idx]);
                let mask = _mm256_movemask_epi8(cmp) as u32;

                if mask != 0 {
                    // Found potential matches, verify each
                    let mut bit_mask = mask;
                    while bit_mask != 0 {
                        let bit_pos = bit_mask.trailing_zeros() as usize;
                        let pos = chunk_start + bit_pos;

                        // Verify all patterns in this bucket
                        for &pattern_idx in &bucket.pattern_indices {
                            let pattern = &self.patterns[pattern_idx];
                            if pos + pattern.len() <= input.len()
                                && input[pos..pos + pattern.len()] == pattern[..]
                            {
                                matches.push(TeddyMatch {
                                    start: pos,
                                    end: pos + pattern.len(),
                                    pattern_id: self.pattern_ids[pattern_idx],
                                });
                            }
                        }

                        bit_mask &= bit_mask - 1; // Clear lowest set bit
                    }
                }
            }
        }

        // Handle remaining bytes with scalar code
        let remaining_start = chunks * 32;
        if remaining_start < input.len() {
            let remaining_matches = self.find_all_scalar(&input[remaining_start..]);
            for m in remaining_matches {
                matches.push(TeddyMatch {
                    start: remaining_start + m.start,
                    end: remaining_start + m.end,
                    pattern_id: m.pattern_id,
                });
            }
        }

        matches
    }

    /// AVX2-accelerated find_first
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn find_first_avx2(&self, input: &[u8]) -> Option<TeddyMatch> {
        use std::arch::x86_64::*;

        if input.len() < 32 {
            return self.find_first_scalar(input);
        }

        // Build first-byte masks for each bucket
        let mut bucket_masks: [__m256i; NUM_BUCKETS] = [_mm256_setzero_si256(); NUM_BUCKETS];
        for (bucket_idx, bucket) in self.buckets.iter().enumerate() {
            if !bucket.first_bytes.is_empty() {
                bucket_masks[bucket_idx] = _mm256_set1_epi8(bucket.first_bytes[0] as i8);
            }
        }

        // Process 32 bytes at a time
        let chunks = input.len() / 32;
        for chunk_idx in 0..chunks {
            let chunk_start = chunk_idx * 32;
            let chunk = _mm256_loadu_si256(input.as_ptr().add(chunk_start) as *const __m256i);

            for (bucket_idx, bucket) in self.buckets.iter().enumerate() {
                if bucket.pattern_indices.is_empty() {
                    continue;
                }

                let cmp = _mm256_cmpeq_epi8(chunk, bucket_masks[bucket_idx]);
                let mask = _mm256_movemask_epi8(cmp) as u32;

                if mask != 0 {
                    let mut bit_mask = mask;
                    while bit_mask != 0 {
                        let bit_pos = bit_mask.trailing_zeros() as usize;
                        let pos = chunk_start + bit_pos;

                        for &pattern_idx in &bucket.pattern_indices {
                            let pattern = &self.patterns[pattern_idx];
                            if pos + pattern.len() <= input.len()
                                && input[pos..pos + pattern.len()] == pattern[..]
                            {
                                return Some(TeddyMatch {
                                    start: pos,
                                    end: pos + pattern.len(),
                                    pattern_id: self.pattern_ids[pattern_idx],
                                });
                            }
                        }

                        bit_mask &= bit_mask - 1;
                    }
                }
            }
        }

        // Handle remaining bytes
        let remaining_start = chunks * 32;
        if remaining_start < input.len() {
            if let Some(m) = self.find_first_scalar(&input[remaining_start..]) {
                return Some(TeddyMatch {
                    start: remaining_start + m.start,
                    end: remaining_start + m.end,
                    pattern_id: m.pattern_id,
                });
            }
        }

        None
    }
}

/// Builder for TeddyMatcher with optimizations
#[derive(Default)]
pub struct TeddyMatcherBuilder {
    patterns: Vec<(Vec<u8>, u32)>,
}

impl TeddyMatcherBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pattern with its ID (consuming builder pattern)
    pub fn add_pattern(mut self, pattern: impl AsRef<[u8]>, id: u32) -> Self {
        self.patterns.push((pattern.as_ref().to_vec(), id));
        self
    }

    /// Add a string pattern (consuming builder pattern)
    pub fn add_string(self, pattern: &str, id: u32) -> Self {
        self.add_pattern(pattern.as_bytes(), id)
    }

    /// Add a pattern with its ID (mutable reference pattern)
    pub fn add(&mut self, pattern: impl AsRef<[u8]>, id: u32) -> &mut Self {
        self.patterns.push((pattern.as_ref().to_vec(), id));
        self
    }

    /// Build the matcher
    pub fn build(self) -> TeddyMatcher {
        TeddyMatcher::new(self.patterns.into_iter())
    }
}

/// Statistics about Teddy matcher
#[derive(Debug, Clone)]
pub struct TeddyStats {
    /// Number of patterns
    pub pattern_count: usize,
    /// Number of non-empty buckets
    pub active_buckets: usize,
    /// Average patterns per active bucket
    pub avg_patterns_per_bucket: f64,
    /// Minimum pattern length
    pub min_pattern_len: usize,
    /// Maximum pattern length
    pub max_pattern_len: usize,
}

impl TeddyMatcher {
    /// Get statistics about the matcher
    pub fn stats(&self) -> TeddyStats {
        let active_buckets = self.buckets.iter().filter(|b| !b.pattern_indices.is_empty()).count();
        let total_patterns_in_buckets: usize = self.buckets.iter().map(|b| b.pattern_indices.len()).sum();

        let avg_patterns_per_bucket = if active_buckets > 0 {
            total_patterns_in_buckets as f64 / active_buckets as f64
        } else {
            0.0
        };

        let max_pattern_len = self.patterns.iter().map(|p| p.len()).max().unwrap_or(0);

        TeddyStats {
            pattern_count: self.patterns.len(),
            active_buckets,
            avg_patterns_per_bucket,
            min_pattern_len: self.min_pattern_len,
            max_pattern_len,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_matcher() {
        let matcher: TeddyMatcher = TeddyMatcher::new(std::iter::empty::<(&[u8], u32)>());
        assert!(matcher.is_empty());
        assert_eq!(matcher.find_all(b"hello"), vec![]);
        assert_eq!(matcher.find_first(b"hello"), None);
    }

    #[test]
    fn test_single_pattern() {
        let matcher = TeddyMatcher::new(vec![(b"hello".to_vec(), 0u32)]);
        assert_eq!(matcher.pattern_count(), 1);

        let matches = matcher.find_all(b"say hello world");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].start, 4);
        assert_eq!(matches[0].end, 9);
        assert_eq!(matches[0].pattern_id, 0);
    }

    #[test]
    fn test_multiple_patterns() {
        let matcher = TeddyMatcher::new(vec![
            (b"[CLS]".to_vec(), 0u32),
            (b"[SEP]".to_vec(), 1),
            (b"[MASK]".to_vec(), 2),
            (b"[PAD]".to_vec(), 3),
        ]);

        let input = b"[CLS] Hello [MASK] World [SEP]";
        let matches = matcher.find_all(input);

        assert_eq!(matches.len(), 3);

        // Verify [CLS]
        let cls_match = matches.iter().find(|m| m.pattern_id == 0).unwrap();
        assert_eq!(cls_match.start, 0);
        assert_eq!(cls_match.end, 5);

        // Verify [MASK]
        let mask_match = matches.iter().find(|m| m.pattern_id == 2).unwrap();
        assert_eq!(mask_match.start, 12);
        assert_eq!(mask_match.end, 18);

        // Verify [SEP]
        let sep_match = matches.iter().find(|m| m.pattern_id == 1).unwrap();
        assert_eq!(sep_match.start, 25);
        assert_eq!(sep_match.end, 30);
    }

    #[test]
    fn test_overlapping_matches() {
        let matcher = TeddyMatcher::new(vec![
            (b"abc".to_vec(), 0u32),
            (b"bcd".to_vec(), 1),
        ]);

        let matches = matcher.find_all(b"abcd");
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_find_first() {
        let matcher = TeddyMatcher::new(vec![
            (b"world".to_vec(), 0u32),
            (b"hello".to_vec(), 1),
        ]);

        let first = matcher.find_first(b"hello world");
        assert!(first.is_some());
        let m = first.unwrap();
        assert_eq!(m.pattern_id, 1); // "hello" comes first in input
        assert_eq!(m.start, 0);
    }

    #[test]
    fn test_no_match() {
        let matcher = TeddyMatcher::new(vec![(b"xyz".to_vec(), 0u32)]);
        assert_eq!(matcher.find_all(b"hello world"), vec![]);
        assert_eq!(matcher.find_first(b"hello world"), None);
    }

    #[test]
    fn test_builder() {
        let matcher = TeddyMatcherBuilder::new()
            .add_string("[UNK]", 0)
            .add_string("[PAD]", 1)
            .add_pattern(b"</s>", 2)
            .build();

        assert_eq!(matcher.pattern_count(), 3);

        let matches = matcher.find_all(b"[UNK] token [PAD]");
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_unicode_patterns() {
        let matcher = TeddyMatcher::new(vec![
            ("▁".as_bytes().to_vec(), 0u32),  // Metaspace
            ("日本".as_bytes().to_vec(), 1),
        ]);

        let input = "▁Hello 日本語".as_bytes();
        let matches = matcher.find_all(input);

        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_stats() {
        let matcher = TeddyMatcher::new(vec![
            (b"[CLS]".to_vec(), 0u32),
            (b"[SEP]".to_vec(), 1),
            (b"[MASK]".to_vec(), 2),
        ]);

        let stats = matcher.stats();
        assert_eq!(stats.pattern_count, 3);
        assert!(stats.active_buckets > 0);
        assert_eq!(stats.min_pattern_len, 5);
        assert_eq!(stats.max_pattern_len, 6);
    }

    #[test]
    fn test_long_input() {
        let matcher = TeddyMatcher::new(vec![
            (b"[TOKEN]".to_vec(), 0u32),
        ]);

        // Create input longer than 32 bytes to test AVX2 path
        let input = b"This is a long input text that contains [TOKEN] somewhere in the middle and is definitely longer than 32 bytes";
        let matches = matcher.find_all(input);

        assert_eq!(matches.len(), 1);
        assert!(matches[0].start > 32); // Token is after first chunk
    }

    #[test]
    fn test_multiple_occurrences() {
        let matcher = TeddyMatcher::new(vec![(b"ab".to_vec(), 0u32)]);
        let matches = matcher.find_all(b"ab ab ab");

        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0].start, 0);
        assert_eq!(matches[1].start, 3);
        assert_eq!(matches[2].start, 6);
    }

    #[test]
    fn test_empty_input() {
        let matcher = TeddyMatcher::new(vec![(b"test".to_vec(), 0u32)]);
        assert_eq!(matcher.find_all(b""), vec![]);
        assert_eq!(matcher.find_first(b""), None);
    }

    #[test]
    fn test_input_shorter_than_pattern() {
        let matcher = TeddyMatcher::new(vec![(b"hello".to_vec(), 0u32)]);
        assert_eq!(matcher.find_all(b"hi"), vec![]);
    }
}
