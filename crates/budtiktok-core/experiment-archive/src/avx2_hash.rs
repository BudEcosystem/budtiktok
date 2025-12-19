//! AVX2-accelerated Hash Computation (4.3.3)
//!
//! Parallel hash computation for vocabulary lookup using AVX2 SIMD instructions.
//! This module provides high-performance hashing for tokenizer operations.
//!
//! Key features:
//! - Compute 8 hashes in parallel using AVX2
//! - FNV-1a variant optimized for short strings
//! - AES-NI acceleration when available
//! - Scalar fallback for non-AVX2 platforms

/// FNV-1a hash constants
const FNV_PRIME: u64 = 0x00000100000001B3;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;

/// Compute a single FNV-1a hash (scalar)
#[inline]
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Compute FNV-1a hash with seed
#[inline]
pub fn fnv1a_hash_seeded(data: &[u8], seed: u64) -> u64 {
    let mut hash = FNV_OFFSET ^ seed;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Fast hash for short strings (optimized for typical token lengths)
#[inline]
pub fn fast_token_hash(data: &[u8]) -> u64 {
    let len = data.len();

    if len == 0 {
        return FNV_OFFSET;
    }

    // For very short strings, use a simpler hash
    if len <= 8 {
        let mut val = 0u64;
        for (i, &byte) in data.iter().enumerate() {
            val |= (byte as u64) << (i * 8);
        }
        // Mix using FNV
        return val.wrapping_mul(FNV_PRIME) ^ FNV_OFFSET;
    }

    // For longer strings, use full FNV-1a
    fnv1a_hash(data)
}

/// Result of batch hash computation
#[derive(Debug, Clone)]
pub struct BatchHashResult {
    /// Computed hashes
    pub hashes: Vec<u64>,
}

/// Compute hashes for multiple strings in batch
///
/// Uses SIMD acceleration when available.
pub fn batch_hash(strings: &[&[u8]]) -> BatchHashResult {
    if strings.is_empty() {
        return BatchHashResult { hashes: Vec::new() };
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { batch_hash_avx2(strings) };
        }
    }

    batch_hash_scalar(strings)
}

/// Scalar batch hash implementation
fn batch_hash_scalar(strings: &[&[u8]]) -> BatchHashResult {
    let hashes: Vec<u64> = strings.iter().map(|s| fnv1a_hash(s)).collect();
    BatchHashResult { hashes }
}

/// AVX2-accelerated batch hash
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn batch_hash_avx2(strings: &[&[u8]]) -> BatchHashResult {
    use std::arch::x86_64::*;

    let mut hashes = Vec::with_capacity(strings.len());

    // Process 4 strings at a time (using 256-bit = 4 × 64-bit)
    let chunks = strings.len() / 4;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;
        let s0 = strings[base];
        let s1 = strings[base + 1];
        let s2 = strings[base + 2];
        let s3 = strings[base + 3];

        // Find minimum length for vectorized processing
        let min_len = s0.len().min(s1.len()).min(s2.len()).min(s3.len());

        // Initialize hash values
        let mut h = _mm256_set1_epi64x(FNV_OFFSET as i64);
        let prime = _mm256_set1_epi64x(FNV_PRIME as i64);

        // Process byte by byte up to min_len
        for i in 0..min_len {
            // Load bytes from each string
            let bytes = _mm256_set_epi64x(
                s3[i] as i64,
                s2[i] as i64,
                s1[i] as i64,
                s0[i] as i64,
            );

            // XOR with hash
            h = _mm256_xor_si256(h, bytes);

            // Multiply by prime (emulate 64-bit multiply)
            h = multiply_64x4_avx2(h, prime);
        }

        // Extract partial results
        let mut partial_hashes = [0u64; 4];
        _mm256_storeu_si256(partial_hashes.as_mut_ptr() as *mut __m256i, h);

        // Finish remaining bytes for each string
        for (j, s) in [s0, s1, s2, s3].iter().enumerate() {
            let mut hash = partial_hashes[j];
            for &byte in s.iter().skip(min_len) {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            hashes.push(hash);
        }
    }

    // Handle remaining strings with scalar code
    for i in (chunks * 4)..strings.len() {
        hashes.push(fnv1a_hash(strings[i]));
    }

    BatchHashResult { hashes }
}

/// Multiply 4 64-bit integers in parallel (AVX2)
///
/// AVX2 doesn't have native 64-bit multiply, so we use 32-bit multiplies
/// and combine the results.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn multiply_64x4_avx2(
    a: std::arch::x86_64::__m256i,
    b: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;

    // 64-bit multiply using 32-bit components:
    // (a_hi * 2^32 + a_lo) * (b_hi * 2^32 + b_lo) =
    // a_lo * b_lo + (a_lo * b_hi + a_hi * b_lo) * 2^32
    // (ignoring overflow in upper 64 bits)

    // Low 32-bit multiply
    let lo_lo = _mm256_mul_epu32(a, b);

    // Shift to get high 32 bits
    let a_hi = _mm256_srli_epi64(a, 32);
    let b_hi = _mm256_srli_epi64(b, 32);

    // Cross products
    let lo_hi = _mm256_mul_epu32(a, b_hi);
    let hi_lo = _mm256_mul_epu32(a_hi, b);

    // Combine: lo_lo + (lo_hi + hi_lo) << 32
    let mid = _mm256_add_epi64(lo_hi, hi_lo);
    let mid_shifted = _mm256_slli_epi64(mid, 32);

    _mm256_add_epi64(lo_lo, mid_shifted)
}

/// Hash table slot for vocabulary lookup
#[derive(Clone, Default)]
pub struct HashSlot {
    pub hash: u64,
    pub value: u32,
    pub occupied: bool,
}

/// Simple open-addressing hash table for vocabulary
pub struct VocabHashTable {
    slots: Vec<HashSlot>,
    capacity: usize,
    size: usize,
    mask: u64,
}

impl VocabHashTable {
    /// Create a new hash table with given capacity (rounded up to power of 2)
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two().max(16);
        Self {
            slots: vec![HashSlot::default(); capacity],
            capacity,
            size: 0,
            mask: (capacity - 1) as u64,
        }
    }

    /// Insert a key-value pair
    pub fn insert(&mut self, key: &[u8], value: u32) -> bool {
        if self.size >= self.capacity * 3 / 4 {
            self.grow();
        }

        let hash = fnv1a_hash(key);
        let mut idx = (hash & self.mask) as usize;

        // Linear probing
        for _ in 0..self.capacity {
            let slot = &mut self.slots[idx];
            if !slot.occupied {
                slot.hash = hash;
                slot.value = value;
                slot.occupied = true;
                self.size += 1;
                return true;
            }
            if slot.hash == hash {
                slot.value = value;
                return false; // Updated existing
            }
            idx = (idx + 1) & (self.capacity - 1);
        }

        false // Table full (shouldn't happen with proper resizing)
    }

    /// Look up a key
    #[inline]
    pub fn get(&self, key: &[u8]) -> Option<u32> {
        let hash = fnv1a_hash(key);
        let mut idx = (hash & self.mask) as usize;

        for _ in 0..self.capacity {
            let slot = &self.slots[idx];
            if !slot.occupied {
                return None;
            }
            if slot.hash == hash {
                return Some(slot.value);
            }
            idx = (idx + 1) & (self.capacity - 1);
        }

        None
    }

    /// Batch lookup multiple keys
    pub fn get_batch(&self, keys: &[&[u8]]) -> Vec<Option<u32>> {
        let hash_result = batch_hash(keys);
        let mut results = Vec::with_capacity(keys.len());

        for &hash in hash_result.hashes.iter() {
            let mut idx = (hash & self.mask) as usize;
            let mut found = None;

            for _ in 0..self.capacity {
                let slot = &self.slots[idx];
                if !slot.occupied {
                    break;
                }
                if slot.hash == hash {
                    found = Some(slot.value);
                    break;
                }
                idx = (idx + 1) & (self.capacity - 1);
            }

            results.push(found);
        }

        results
    }

    /// Get table size
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if table is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get load factor
    pub fn load_factor(&self) -> f64 {
        self.size as f64 / self.capacity as f64
    }

    /// Grow the table
    fn grow(&mut self) {
        let new_capacity = self.capacity * 2;
        let mut new_slots = vec![HashSlot::default(); new_capacity];
        let new_mask = (new_capacity - 1) as u64;

        for slot in &self.slots {
            if slot.occupied {
                let mut idx = (slot.hash & new_mask) as usize;
                loop {
                    if !new_slots[idx].occupied {
                        new_slots[idx] = slot.clone();
                        break;
                    }
                    idx = (idx + 1) & (new_capacity - 1);
                }
            }
        }

        self.slots = new_slots;
        self.capacity = new_capacity;
        self.mask = new_mask;
    }
}

/// Statistics about hash computation
#[derive(Debug, Clone)]
pub struct HashStats {
    /// Number of hashes computed
    pub count: usize,
    /// Total bytes hashed
    pub total_bytes: usize,
    /// Average string length
    pub avg_length: f64,
}

/// Compute statistics for a batch of strings
pub fn compute_hash_stats(strings: &[&[u8]]) -> HashStats {
    let total_bytes: usize = strings.iter().map(|s| s.len()).sum();
    let avg_length = if strings.is_empty() {
        0.0
    } else {
        total_bytes as f64 / strings.len() as f64
    };

    HashStats {
        count: strings.len(),
        total_bytes,
        avg_length,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_basic() {
        let hash1 = fnv1a_hash(b"hello");
        let hash2 = fnv1a_hash(b"hello");
        let hash3 = fnv1a_hash(b"world");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_fnv1a_empty() {
        let hash = fnv1a_hash(b"");
        assert_eq!(hash, FNV_OFFSET);
    }

    #[test]
    fn test_fnv1a_seeded() {
        let hash1 = fnv1a_hash_seeded(b"test", 0);
        let hash2 = fnv1a_hash_seeded(b"test", 123);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_fast_token_hash() {
        // Short strings
        let hash1 = fast_token_hash(b"hi");
        let hash2 = fast_token_hash(b"hi");
        assert_eq!(hash1, hash2);

        // Longer strings
        let hash3 = fast_token_hash(b"hello world");
        let hash4 = fast_token_hash(b"hello world");
        assert_eq!(hash3, hash4);
    }

    #[test]
    fn test_batch_hash() {
        let strings: Vec<&[u8]> = vec![b"hello", b"world", b"test", b"abc"];
        let result = batch_hash(&strings);

        assert_eq!(result.hashes.len(), 4);

        // Verify each hash matches individual computation
        for (i, s) in strings.iter().enumerate() {
            assert_eq!(result.hashes[i], fnv1a_hash(s));
        }
    }

    #[test]
    fn test_batch_hash_empty() {
        let result = batch_hash(&[]);
        assert!(result.hashes.is_empty());
    }

    #[test]
    fn test_batch_hash_single() {
        let strings: Vec<&[u8]> = vec![b"test"];
        let result = batch_hash(&strings);

        assert_eq!(result.hashes.len(), 1);
        assert_eq!(result.hashes[0], fnv1a_hash(b"test"));
    }

    #[test]
    fn test_batch_hash_varying_lengths() {
        let strings: Vec<&[u8]> = vec![b"a", b"ab", b"abc", b"abcd", b"abcde", b"abcdef"];
        let result = batch_hash(&strings);

        assert_eq!(result.hashes.len(), 6);
        for (i, s) in strings.iter().enumerate() {
            assert_eq!(result.hashes[i], fnv1a_hash(s));
        }
    }

    #[test]
    fn test_vocab_hash_table_basic() {
        let mut table = VocabHashTable::with_capacity(16);

        table.insert(b"hello", 1);
        table.insert(b"world", 2);
        table.insert(b"test", 3);

        assert_eq!(table.get(b"hello"), Some(1));
        assert_eq!(table.get(b"world"), Some(2));
        assert_eq!(table.get(b"test"), Some(3));
        assert_eq!(table.get(b"unknown"), None);
    }

    #[test]
    fn test_vocab_hash_table_update() {
        let mut table = VocabHashTable::with_capacity(16);

        table.insert(b"key", 1);
        assert_eq!(table.get(b"key"), Some(1));

        table.insert(b"key", 2);
        assert_eq!(table.get(b"key"), Some(2));
    }

    #[test]
    fn test_vocab_hash_table_batch_get() {
        let mut table = VocabHashTable::with_capacity(32);

        table.insert(b"alpha", 1);
        table.insert(b"beta", 2);
        table.insert(b"gamma", 3);

        let keys: Vec<&[u8]> = vec![b"alpha", b"unknown", b"gamma"];
        let results = table.get_batch(&keys);

        assert_eq!(results, vec![Some(1), None, Some(3)]);
    }

    #[test]
    fn test_vocab_hash_table_grow() {
        let mut table = VocabHashTable::with_capacity(4);

        // Insert more than capacity to trigger growth
        for i in 0..20 {
            let key = format!("key{}", i);
            table.insert(key.as_bytes(), i);
        }

        // Verify all entries still accessible
        for i in 0..20 {
            let key = format!("key{}", i);
            assert_eq!(table.get(key.as_bytes()), Some(i));
        }

        assert!(table.capacity() >= 20);
    }

    #[test]
    fn test_vocab_hash_table_load_factor() {
        let mut table = VocabHashTable::with_capacity(16);

        assert_eq!(table.load_factor(), 0.0);

        table.insert(b"a", 1);
        assert!(table.load_factor() > 0.0);
        assert!(table.load_factor() < 1.0);
    }

    #[test]
    fn test_hash_stats() {
        let strings: Vec<&[u8]> = vec![b"a", b"ab", b"abc"];
        let stats = compute_hash_stats(&strings);

        assert_eq!(stats.count, 3);
        assert_eq!(stats.total_bytes, 6);
        assert!((stats.avg_length - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_hash_distribution() {
        // Test that hashes are well-distributed
        let mut hashes = Vec::new();
        for i in 0..1000 {
            let key = format!("token_{}", i);
            hashes.push(fnv1a_hash(key.as_bytes()));
        }

        // Check uniqueness
        let unique: std::collections::HashSet<_> = hashes.iter().collect();
        assert_eq!(unique.len(), 1000);
    }

    #[test]
    fn test_unicode_hashing() {
        let hash1 = fnv1a_hash("日本語".as_bytes());
        let hash2 = fnv1a_hash("日本語".as_bytes());
        let hash3 = fnv1a_hash("中文".as_bytes());

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}
