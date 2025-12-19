//! Token caching with CLOCK eviction algorithm
//!
//! This module provides a high-performance cache for tokenization results
//! using the CLOCK (Second Chance) algorithm for O(1) amortized eviction.

use ahash::AHashMap;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Cache entry with reference bit for CLOCK algorithm
struct CacheEntry<V> {
    value: V,
    referenced: AtomicBool,
}

impl<V: Clone> CacheEntry<V> {
    fn new(value: V) -> Self {
        Self {
            value,
            // Start with referenced=false so CLOCK can distinguish
            // between "just inserted" and "accessed since insertion"
            referenced: AtomicBool::new(false),
        }
    }

    fn get(&self) -> V {
        self.referenced.store(true, Ordering::Relaxed);
        self.value.clone()
    }

    fn is_referenced(&self) -> bool {
        self.referenced.load(Ordering::Relaxed)
    }

    fn clear_reference(&self) {
        self.referenced.store(false, Ordering::Relaxed);
    }
}

/// CLOCK cache implementation for token results
///
/// The CLOCK algorithm provides O(1) amortized eviction while maintaining
/// good cache hit rates for tokenization workloads.
pub struct ClockCache<K, V> {
    entries: RwLock<AHashMap<K, CacheEntry<V>>>,
    keys: RwLock<Vec<K>>,
    hand: AtomicUsize,
    capacity: usize,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl<K, V> ClockCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new CLOCK cache with the specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: RwLock::new(AHashMap::with_capacity(capacity)),
            keys: RwLock::new(Vec::with_capacity(capacity)),
            hand: AtomicUsize::new(0),
            capacity,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    /// Get a value from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        let entries = self.entries.read();
        if let Some(entry) = entries.get(key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.get())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert a value into the cache
    pub fn insert(&self, key: K, value: V) {
        let mut entries = self.entries.write();
        let mut keys = self.keys.write();

        // If key already exists, update the value in place (no change to keys vector)
        if let Some(entry) = entries.get_mut(&key) {
            // Update the entry directly to avoid allocation
            *entry = CacheEntry::new(value);
            entry.referenced.store(true, Ordering::Relaxed);
            return;
        }

        // If at capacity, evict using CLOCK algorithm
        if entries.len() >= self.capacity {
            self.evict(&mut entries, &mut keys);
        }

        // Insert new entry - key goes in both maps
        keys.push(key.clone());
        entries.insert(key, CacheEntry::new(value));
    }

    /// Evict an entry using the CLOCK algorithm
    ///
    /// Uses the "second chance" algorithm: entries with the reference bit set
    /// get their bit cleared and are skipped. Entries without the bit are evicted.
    fn evict(&self, entries: &mut AHashMap<K, CacheEntry<V>>, keys: &mut Vec<K>) {
        if keys.is_empty() {
            return;
        }

        let len = keys.len();
        // Ensure hand is within bounds after potential concurrent modifications
        let mut hand = self.hand.load(Ordering::Relaxed);
        if hand >= len {
            hand = 0;
        }
        let start_hand = hand;

        loop {
            // Safety: hand is guaranteed to be < len at this point
            debug_assert!(hand < keys.len(), "hand {} >= keys.len() {}", hand, keys.len());

            let key = &keys[hand];

            if let Some(entry) = entries.get(key) {
                if !entry.is_referenced() {
                    // Found victim - evict it
                    let victim_key = keys.swap_remove(hand);
                    entries.remove(&victim_key);

                    // After swap_remove, if hand was pointing to the last element,
                    // we need to adjust. If keys is now empty or hand >= new len,
                    // reset to 0.
                    let new_len = keys.len();
                    let new_hand = if new_len == 0 || hand >= new_len {
                        0
                    } else {
                        hand
                    };
                    self.hand.store(new_hand, Ordering::Relaxed);
                    return;
                }
                // Give second chance
                entry.clear_reference();
            } else {
                // Entry doesn't exist in map (shouldn't happen, but handle gracefully)
                // Remove the orphan key
                keys.swap_remove(hand);
                let new_len = keys.len();
                if new_len == 0 {
                    self.hand.store(0, Ordering::Relaxed);
                    return;
                }
                // Don't increment hand since swap_remove brought a new element here
                if hand >= new_len {
                    hand = 0;
                }
                continue;
            }

            // Move to next position
            hand = (hand + 1) % len;

            // Safety: prevent infinite loop if all entries are referenced
            if hand == start_hand {
                // All entries have been given second chance, evict current
                // Recalculate len in case it changed
                let current_len = keys.len();
                if current_len == 0 {
                    self.hand.store(0, Ordering::Relaxed);
                    return;
                }
                if hand >= current_len {
                    hand = 0;
                }

                let victim_key = keys.swap_remove(hand);
                entries.remove(&victim_key);

                let new_len = keys.len();
                let new_hand = if new_len == 0 || hand >= new_len { 0 } else { hand };
                self.hand.store(new_hand, Ordering::Relaxed);
                return;
            }
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        CacheStats {
            hits,
            misses,
            hit_rate,
            size: self.entries.read().len(),
            capacity: self.capacity,
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        let mut entries = self.entries.write();
        let mut keys = self.keys.write();
        entries.clear();
        keys.clear();
        self.hand.store(0, Ordering::Relaxed);
    }

    /// Get the current size of the cache
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub hit_rate: f64,
    pub size: usize,
    pub capacity: usize,
}

/// Token cache specialized for tokenization results
pub type TokenCache = ClockCache<String, Vec<u32>>;

/// Create a new token cache with default capacity
pub fn new_token_cache() -> TokenCache {
    // Default: 100K entries, ~50MB assuming average 100 bytes per entry
    ClockCache::new(100_000)
}

/// Create a token cache with specified capacity
pub fn new_token_cache_with_capacity(capacity: usize) -> TokenCache {
    ClockCache::new(capacity)
}

// ============================================================================
// Sharded Cache - Reduces lock contention for concurrent access
// ============================================================================

/// Number of shards for the sharded cache (power of 2 for efficient modulo)
const DEFAULT_SHARD_COUNT: usize = 16;

/// Sharded cache implementation for reduced lock contention
///
/// Distributes entries across multiple shards to reduce lock contention
/// under concurrent access. Each shard is an independent CLOCK cache.
pub struct ShardedCache<K, V> {
    shards: Vec<ClockCache<K, V>>,
    shard_count: usize,
    hasher: ahash::RandomState,
}

impl<K, V> ShardedCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new sharded cache with the specified total capacity
    pub fn new(capacity: usize) -> Self {
        Self::with_shards(capacity, DEFAULT_SHARD_COUNT)
    }

    /// Create a new sharded cache with specified capacity and shard count
    pub fn with_shards(capacity: usize, shard_count: usize) -> Self {
        let shard_count = shard_count.max(1).next_power_of_two();
        let shard_capacity = (capacity / shard_count).max(1);

        let shards = (0..shard_count)
            .map(|_| ClockCache::new(shard_capacity))
            .collect();

        Self {
            shards,
            shard_count,
            hasher: ahash::RandomState::new(),
        }
    }

    /// Get the shard index for a key
    #[inline]
    fn shard_index(&self, key: &K) -> usize {
        use std::hash::{BuildHasher, Hasher};
        let mut hasher = self.hasher.build_hasher();
        key.hash(&mut hasher);
        hasher.finish() as usize & (self.shard_count - 1)
    }

    /// Get a value from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        let shard_idx = self.shard_index(key);
        self.shards[shard_idx].get(key)
    }

    /// Insert a value into the cache
    pub fn insert(&self, key: K, value: V) {
        let shard_idx = self.shard_index(&key);
        self.shards[shard_idx].insert(key, value);
    }

    /// Get combined cache statistics
    pub fn stats(&self) -> CacheStats {
        let mut total_hits = 0;
        let mut total_misses = 0;
        let mut total_size = 0;
        let mut total_capacity = 0;

        for shard in &self.shards {
            let shard_stats = shard.stats();
            total_hits += shard_stats.hits;
            total_misses += shard_stats.misses;
            total_size += shard_stats.size;
            total_capacity += shard_stats.capacity;
        }

        let total = total_hits + total_misses;
        let hit_rate = if total > 0 {
            total_hits as f64 / total as f64
        } else {
            0.0
        };

        CacheStats {
            hits: total_hits,
            misses: total_misses,
            hit_rate,
            size: total_size,
            capacity: total_capacity,
        }
    }

    /// Clear all shards
    pub fn clear(&self) {
        for shard in &self.shards {
            shard.clear();
        }
    }

    /// Get the total size across all shards
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Check if all shards are empty
    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.is_empty())
    }

    /// Get the number of shards
    pub fn shard_count(&self) -> usize {
        self.shard_count
    }
}

// ============================================================================
// Multi-Level Cache - L1 (fast) + L2 (large)
// ============================================================================

/// Configuration for multi-level cache
#[derive(Debug, Clone)]
pub struct MultiLevelCacheConfig {
    /// L1 cache capacity (small, fast)
    pub l1_capacity: usize,
    /// L2 cache capacity (larger, slower)
    pub l2_capacity: usize,
    /// Whether to promote L2 hits to L1
    pub promote_on_hit: bool,
}

impl Default for MultiLevelCacheConfig {
    fn default() -> Self {
        Self {
            l1_capacity: 1_000,     // 1K entries in L1
            l2_capacity: 100_000,    // 100K entries in L2
            promote_on_hit: true,
        }
    }
}

/// Multi-level cache with L1 (fast, small) and L2 (slower, larger)
///
/// L1 uses a simple HashMap with RwLock for fastest access.
/// L2 uses a sharded CLOCK cache for larger capacity.
///
/// On L1 miss but L2 hit, entry is promoted to L1 (if configured).
pub struct MultiLevelCache<K, V> {
    l1: RwLock<AHashMap<K, V>>,
    l2: ShardedCache<K, V>,
    config: MultiLevelCacheConfig,
    l1_hits: AtomicUsize,
    l2_hits: AtomicUsize,
    misses: AtomicUsize,
}

impl<K, V> MultiLevelCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new multi-level cache with default configuration
    pub fn new() -> Self {
        Self::with_config(MultiLevelCacheConfig::default())
    }

    /// Create a new multi-level cache with the specified configuration
    pub fn with_config(config: MultiLevelCacheConfig) -> Self {
        Self {
            l1: RwLock::new(AHashMap::with_capacity(config.l1_capacity)),
            l2: ShardedCache::new(config.l2_capacity),
            config,
            l1_hits: AtomicUsize::new(0),
            l2_hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    /// Get a value from the cache (checks L1 first, then L2)
    pub fn get(&self, key: &K) -> Option<V> {
        // Check L1 first
        {
            let l1 = self.l1.read();
            if let Some(value) = l1.get(key) {
                self.l1_hits.fetch_add(1, Ordering::Relaxed);
                return Some(value.clone());
            }
        }

        // Check L2
        if let Some(value) = self.l2.get(key) {
            self.l2_hits.fetch_add(1, Ordering::Relaxed);

            // Promote to L1 if configured
            if self.config.promote_on_hit {
                self.insert_l1(key.clone(), value.clone());
            }

            return Some(value);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Insert a value into the cache (goes to both L1 and L2)
    pub fn insert(&self, key: K, value: V) {
        self.insert_l1(key.clone(), value.clone());
        self.l2.insert(key, value);
    }

    /// Insert only into L1 (for promotions)
    fn insert_l1(&self, key: K, value: V) {
        let mut l1 = self.l1.write();

        // Simple FIFO eviction for L1 if full
        if l1.len() >= self.config.l1_capacity && !l1.contains_key(&key) {
            // Remove a random entry (HashMap doesn't maintain order)
            if let Some(evict_key) = l1.keys().next().cloned() {
                l1.remove(&evict_key);
            }
        }

        l1.insert(key, value);
    }

    /// Get cache statistics for both levels
    pub fn stats(&self) -> MultiLevelCacheStats {
        let l1_hits = self.l1_hits.load(Ordering::Relaxed);
        let l2_hits = self.l2_hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = l1_hits + l2_hits + misses;

        let l2_stats = self.l2.stats();

        MultiLevelCacheStats {
            l1_hits,
            l2_hits,
            misses,
            l1_hit_rate: if total > 0 { l1_hits as f64 / total as f64 } else { 0.0 },
            l2_hit_rate: if total > 0 { l2_hits as f64 / total as f64 } else { 0.0 },
            total_hit_rate: if total > 0 { (l1_hits + l2_hits) as f64 / total as f64 } else { 0.0 },
            l1_size: self.l1.read().len(),
            l1_capacity: self.config.l1_capacity,
            l2_size: l2_stats.size,
            l2_capacity: l2_stats.capacity,
        }
    }

    /// Clear both cache levels
    pub fn clear(&self) {
        self.l1.write().clear();
        self.l2.clear();
    }

    /// Get the total size across both levels
    pub fn len(&self) -> usize {
        self.l1.read().len() + self.l2.len()
    }

    /// Check if both levels are empty
    pub fn is_empty(&self) -> bool {
        self.l1.read().is_empty() && self.l2.is_empty()
    }
}

impl<K, V> Default for MultiLevelCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for multi-level cache
#[derive(Debug, Clone)]
pub struct MultiLevelCacheStats {
    pub l1_hits: usize,
    pub l2_hits: usize,
    pub misses: usize,
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub total_hit_rate: f64,
    pub l1_size: usize,
    pub l1_capacity: usize,
    pub l2_size: usize,
    pub l2_capacity: usize,
}

/// Sharded token cache for concurrent access
pub type ShardedTokenCache = ShardedCache<String, Vec<u32>>;

/// Multi-level token cache
pub type MultiLevelTokenCache = MultiLevelCache<String, Vec<u32>>;

/// Create a new sharded token cache with default settings
pub fn new_sharded_token_cache() -> ShardedTokenCache {
    ShardedCache::new(100_000)
}

/// Create a new multi-level token cache with default settings
pub fn new_multi_level_token_cache() -> MultiLevelTokenCache {
    MultiLevelCache::new()
}

// ============================================================================
// Cache Warmup (2.3.5)
// ============================================================================

/// Result of cache warmup operation
#[derive(Debug, Clone)]
pub struct WarmupResult {
    /// Number of entries successfully warmed up
    pub entries_warmed: usize,
    /// Number of entries that failed
    pub entries_failed: usize,
    /// Time taken for warmup in milliseconds
    pub duration_ms: u128,
}

/// Configuration for cache warmup
#[derive(Debug, Clone)]
pub struct WarmupConfig {
    /// Maximum number of entries to warm up
    pub max_entries: usize,
    /// Minimum token frequency to include (0.0-1.0, 0 = include all)
    pub min_frequency: f64,
    /// Number of parallel warmup threads (0 = sequential)
    pub parallel_threads: usize,
    /// Batch size for parallel warmup
    pub batch_size: usize,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,      // Top 10K most frequent
            min_frequency: 0.0,       // Include all
            parallel_threads: 4,      // 4 threads by default
            batch_size: 256,          // Process 256 tokens per batch
        }
    }
}

/// Cache warmer for pre-populating caches with frequent tokens
///
/// The warmer tokenizes frequent vocabulary entries and caches
/// the results for faster lookup during inference.
pub struct CacheWarmer<C> {
    cache: C,
    config: WarmupConfig,
}

impl<C> CacheWarmer<C>
where
    C: CacheInsertable,
{
    /// Create a new cache warmer with the given cache
    pub fn new(cache: C) -> Self {
        Self {
            cache,
            config: WarmupConfig::default(),
        }
    }

    /// Create a new cache warmer with custom configuration
    pub fn with_config(cache: C, config: WarmupConfig) -> Self {
        Self { cache, config }
    }

    /// Warm up the cache with tokens from vocabulary
    ///
    /// Takes a tokenization function that converts strings to token IDs.
    /// The vocabulary is provided as an iterator of (token, frequency) pairs,
    /// sorted by frequency in descending order.
    pub fn warmup<F, I>(&self, vocab_entries: I, tokenize_fn: F) -> WarmupResult
    where
        I: IntoIterator<Item = (String, f64)>,
        F: Fn(&str) -> Vec<u32>,
    {
        let start = std::time::Instant::now();
        let mut entries_warmed = 0;
        let mut entries_failed = 0;

        for (token, frequency) in vocab_entries.into_iter().take(self.config.max_entries) {
            // Skip if below minimum frequency
            if frequency < self.config.min_frequency {
                continue;
            }

            // Tokenize and cache
            let ids = tokenize_fn(&token);
            if ids.is_empty() {
                entries_failed += 1;
            } else {
                self.cache.cache_insert(token, ids);
                entries_warmed += 1;
            }
        }

        WarmupResult {
            entries_warmed,
            entries_failed,
            duration_ms: start.elapsed().as_millis(),
        }
    }

    /// Warm up the cache using parallel processing
    ///
    /// Uses rayon for parallel tokenization. Suitable for large vocabularies.
    #[cfg(feature = "parallel")]
    pub fn warmup_parallel<F, I>(&self, vocab_entries: I, tokenize_fn: F) -> WarmupResult
    where
        I: IntoIterator<Item = (String, f64)>,
        F: Fn(&str) -> Vec<u32> + Sync,
    {
        use rayon::prelude::*;

        let start = std::time::Instant::now();

        let entries: Vec<_> = vocab_entries
            .into_iter()
            .take(self.config.max_entries)
            .filter(|(_, freq)| *freq >= self.config.min_frequency)
            .collect();

        let results: Vec<_> = entries
            .par_chunks(self.config.batch_size)
            .flat_map(|batch| {
                batch.iter().map(|(token, _)| {
                    let ids = tokenize_fn(token);
                    (token.clone(), ids)
                })
            })
            .collect();

        let mut entries_warmed = 0;
        let mut entries_failed = 0;

        for (token, ids) in results {
            if ids.is_empty() {
                entries_failed += 1;
            } else {
                self.cache.cache_insert(token, ids);
                entries_warmed += 1;
            }
        }

        WarmupResult {
            entries_warmed,
            entries_failed,
            duration_ms: start.elapsed().as_millis(),
        }
    }

    /// Warm up the cache from a vocabulary file
    ///
    /// Expects a file with one token per line, optionally followed by
    /// a tab and frequency count.
    pub fn warmup_from_vocab_file<F, P>(&self, path: P, tokenize_fn: F) -> std::io::Result<WarmupResult>
    where
        P: AsRef<std::path::Path>,
        F: Fn(&str) -> Vec<u32>,
    {
        use std::io::{BufRead, BufReader};
        use std::fs::File;

        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut entries: Vec<(String, f64)> = Vec::new();
        let mut max_freq: u64 = 1;

        // First pass: read and find max frequency
        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split('\t').collect();

            let token = parts[0].to_string();
            let freq = if parts.len() > 1 {
                parts[1].parse::<u64>().unwrap_or(1)
            } else {
                1
            };

            max_freq = max_freq.max(freq);
            entries.push((token, freq as f64));
        }

        // Normalize frequencies to 0.0-1.0
        let entries: Vec<_> = entries
            .into_iter()
            .map(|(t, f)| (t, f / max_freq as f64))
            .collect();

        // Sort by frequency descending
        let mut sorted = entries;
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(self.warmup(sorted.into_iter(), tokenize_fn))
    }

    /// Warm up the cache from a simple token list
    ///
    /// Assigns equal frequency to all tokens (1.0).
    pub fn warmup_from_tokens<F, I, S>(&self, tokens: I, tokenize_fn: F) -> WarmupResult
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
        F: Fn(&str) -> Vec<u32>,
    {
        let entries = tokens.into_iter().map(|t| (t.into(), 1.0));
        self.warmup(entries, tokenize_fn)
    }

    /// Get a reference to the underlying cache
    pub fn cache(&self) -> &C {
        &self.cache
    }

    /// Get the warmup configuration
    pub fn config(&self) -> &WarmupConfig {
        &self.config
    }
}

/// Trait for caches that support the warmup insert operation
pub trait CacheInsertable {
    fn cache_insert(&self, key: String, value: Vec<u32>);
}

impl CacheInsertable for ClockCache<String, Vec<u32>> {
    fn cache_insert(&self, key: String, value: Vec<u32>) {
        self.insert(key, value);
    }
}

impl CacheInsertable for ShardedCache<String, Vec<u32>> {
    fn cache_insert(&self, key: String, value: Vec<u32>) {
        self.insert(key, value);
    }
}

impl CacheInsertable for MultiLevelCache<String, Vec<u32>> {
    fn cache_insert(&self, key: String, value: Vec<u32>) {
        self.insert(key, value);
    }
}

/// Convenience function to warm up a CLOCK cache with vocabulary entries
pub fn warmup_cache<F, I>(
    cache: &ClockCache<String, Vec<u32>>,
    vocab_entries: I,
    tokenize_fn: F,
) -> WarmupResult
where
    I: IntoIterator<Item = (String, f64)>,
    F: Fn(&str) -> Vec<u32>,
{
    let start = std::time::Instant::now();
    let mut entries_warmed = 0;
    let mut entries_failed = 0;

    for (token, _) in vocab_entries {
        let ids = tokenize_fn(&token);
        if ids.is_empty() {
            entries_failed += 1;
        } else {
            cache.insert(token, ids);
            entries_warmed += 1;
        }
    }

    WarmupResult {
        entries_warmed,
        entries_failed,
        duration_ms: start.elapsed().as_millis(),
    }
}

/// Convenience function to warm up a sharded cache with vocabulary entries
pub fn warmup_sharded_cache<F, I>(
    cache: &ShardedCache<String, Vec<u32>>,
    vocab_entries: I,
    tokenize_fn: F,
) -> WarmupResult
where
    I: IntoIterator<Item = (String, f64)>,
    F: Fn(&str) -> Vec<u32>,
{
    let start = std::time::Instant::now();
    let mut entries_warmed = 0;
    let mut entries_failed = 0;

    for (token, _) in vocab_entries {
        let ids = tokenize_fn(&token);
        if ids.is_empty() {
            entries_failed += 1;
        } else {
            cache.insert(token, ids);
            entries_warmed += 1;
        }
    }

    WarmupResult {
        entries_warmed,
        entries_failed,
        duration_ms: start.elapsed().as_millis(),
    }
}

/// Convenience function to warm up a multi-level cache with vocabulary entries
pub fn warmup_multi_level_cache<F, I>(
    cache: &MultiLevelCache<String, Vec<u32>>,
    vocab_entries: I,
    tokenize_fn: F,
) -> WarmupResult
where
    I: IntoIterator<Item = (String, f64)>,
    F: Fn(&str) -> Vec<u32>,
{
    let start = std::time::Instant::now();
    let mut entries_warmed = 0;
    let mut entries_failed = 0;

    for (token, _) in vocab_entries {
        let ids = tokenize_fn(&token);
        if ids.is_empty() {
            entries_failed += 1;
        } else {
            cache.insert(token, ids);
            entries_warmed += 1;
        }
    }

    WarmupResult {
        entries_warmed,
        entries_failed,
        duration_ms: start.elapsed().as_millis(),
    }
}

/// Extract frequent tokens from a vocabulary map for warmup
///
/// Returns tokens sorted by ID (lower ID = more frequent in most tokenizers).
pub fn extract_frequent_tokens<I>(vocab: I, max_tokens: usize) -> Vec<(String, f64)>
where
    I: IntoIterator<Item = (String, u32)>,
{
    let mut entries: Vec<_> = vocab.into_iter().collect();

    // Sort by ID ascending (lower ID = more frequent in most vocabularies)
    entries.sort_by_key(|(_, id)| *id);

    // Take top N and convert ID to normalized frequency
    let max_id = entries.last().map(|(_, id)| *id).unwrap_or(1) as f64;

    entries
        .into_iter()
        .take(max_tokens)
        .map(|(token, id)| {
            // Lower ID = higher frequency
            let freq = 1.0 - (id as f64 / max_id);
            (token, freq)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic() {
        let cache: ClockCache<String, Vec<u32>> = ClockCache::new(10);

        cache.insert("hello".to_string(), vec![1, 2, 3]);
        assert_eq!(cache.get(&"hello".to_string()), Some(vec![1, 2, 3]));
        assert_eq!(cache.get(&"world".to_string()), None);
    }

    #[test]
    fn test_cache_eviction() {
        let cache: ClockCache<u32, u32> = ClockCache::new(3);

        cache.insert(1, 10);
        cache.insert(2, 20);
        cache.insert(3, 30);

        // Access 1 and 2 to set their reference bits
        cache.get(&1);
        cache.get(&2);

        // Insert 4, should evict 3 (not referenced)
        cache.insert(4, 40);

        assert!(cache.get(&1).is_some());
        assert!(cache.get(&2).is_some());
        assert!(cache.get(&4).is_some());
        // 3 should have been evicted
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_cache_stats() {
        let cache: ClockCache<String, u32> = ClockCache::new(10);

        cache.insert("a".to_string(), 1);
        cache.get(&"a".to_string()); // hit
        cache.get(&"b".to_string()); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.5).abs() < 0.001);
    }

    // ===== Sharded Cache Tests =====

    #[test]
    fn test_sharded_cache_basic() {
        let cache: ShardedCache<String, u32> = ShardedCache::new(100);

        cache.insert("hello".to_string(), 1);
        cache.insert("world".to_string(), 2);

        assert_eq!(cache.get(&"hello".to_string()), Some(1));
        assert_eq!(cache.get(&"world".to_string()), Some(2));
        assert_eq!(cache.get(&"missing".to_string()), None);
    }

    #[test]
    fn test_sharded_cache_distribution() {
        // Use higher capacity to avoid eviction during test
        let cache: ShardedCache<u32, u32> = ShardedCache::with_shards(1000, 4);

        // Insert entries - fewer than capacity to avoid evictions
        for i in 0..50 {
            cache.insert(i, i * 10);
        }

        // All should be retrievable
        for i in 0..50 {
            assert_eq!(cache.get(&i), Some(i * 10));
        }

        assert_eq!(cache.shard_count(), 4);
        assert!(cache.len() <= 50);
    }

    #[test]
    fn test_sharded_cache_stats() {
        let cache: ShardedCache<String, u32> = ShardedCache::new(100);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);

        cache.get(&"a".to_string()); // hit
        cache.get(&"c".to_string()); // miss

        let stats = cache.stats();
        assert!(stats.hits >= 1);
        assert!(stats.misses >= 1);
    }

    // ===== Multi-Level Cache Tests =====

    #[test]
    fn test_multi_level_cache_basic() {
        let cache: MultiLevelCache<String, u32> = MultiLevelCache::new();

        cache.insert("hello".to_string(), 1);
        assert_eq!(cache.get(&"hello".to_string()), Some(1));
        assert_eq!(cache.get(&"missing".to_string()), None);
    }

    #[test]
    fn test_multi_level_cache_promotion() {
        let config = MultiLevelCacheConfig {
            l1_capacity: 2,
            l2_capacity: 1000, // Large enough for 16 shards
            promote_on_hit: true,
        };
        let cache: MultiLevelCache<String, u32> = MultiLevelCache::with_config(config);

        // Insert entries
        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3);

        // Clear L1 to force L2 lookup
        cache.l1.write().clear();

        // Get from L2 should promote to L1
        assert_eq!(cache.get(&"a".to_string()), Some(1));

        // Now it should be in L1
        let stats = cache.stats();
        assert!(stats.l2_hits >= 1);
    }

    #[test]
    fn test_multi_level_cache_stats() {
        let cache: MultiLevelCache<String, u32> = MultiLevelCache::new();

        cache.insert("a".to_string(), 1);

        // L1 hit
        cache.get(&"a".to_string());
        // Miss
        cache.get(&"missing".to_string());

        let stats = cache.stats();
        assert_eq!(stats.l1_hits, 1);
        assert_eq!(stats.misses, 1);
        assert!(stats.total_hit_rate > 0.0);
    }

    #[test]
    fn test_multi_level_cache_clear() {
        let cache: MultiLevelCache<String, u32> = MultiLevelCache::new();

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);

        assert!(!cache.is_empty());

        cache.clear();

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.get(&"a".to_string()), None);
    }

    // ===== Cache Warmup Tests (2.3.5) =====

    #[test]
    fn test_warmup_basic() {
        let cache: ClockCache<String, Vec<u32>> = ClockCache::new(100);

        // Simple tokenizer that returns a single token ID
        let tokenize = |s: &str| vec![s.len() as u32];

        let vocab = vec![
            ("hello".to_string(), 1.0),
            ("world".to_string(), 0.9),
            ("test".to_string(), 0.8),
        ];

        let result = warmup_cache(&cache, vocab.into_iter(), tokenize);

        assert_eq!(result.entries_warmed, 3);
        assert_eq!(result.entries_failed, 0);

        // Verify entries are cached
        assert_eq!(cache.get(&"hello".to_string()), Some(vec![5]));
        assert_eq!(cache.get(&"world".to_string()), Some(vec![5]));
        assert_eq!(cache.get(&"test".to_string()), Some(vec![4]));
    }

    #[test]
    fn test_cache_warmer_with_config() {
        let cache: ClockCache<String, Vec<u32>> = ClockCache::new(100);
        let config = WarmupConfig {
            max_entries: 2, // Only warm up 2 entries
            min_frequency: 0.5, // Only entries with freq >= 0.5
            parallel_threads: 0,
            batch_size: 1,
        };

        let warmer = CacheWarmer::with_config(cache, config);

        let vocab = vec![
            ("high".to_string(), 1.0),
            ("medium".to_string(), 0.6),
            ("low".to_string(), 0.3), // Below min_frequency
        ];

        let tokenize = |s: &str| vec![s.len() as u32];
        let result = warmer.warmup(vocab.into_iter(), tokenize);

        // Only 2 entries (high and medium), low is below min_frequency
        assert_eq!(result.entries_warmed, 2);
        assert!(warmer.cache().get(&"high".to_string()).is_some());
        assert!(warmer.cache().get(&"medium".to_string()).is_some());
        assert!(warmer.cache().get(&"low".to_string()).is_none());
    }

    #[test]
    fn test_warmup_sharded() {
        let cache: ShardedCache<String, Vec<u32>> = ShardedCache::new(100);

        let tokenize = |s: &str| {
            // Return multiple tokens for multi-byte strings
            s.bytes().map(|b| b as u32).collect()
        };

        let vocab = vec![
            ("abc".to_string(), 1.0),
            ("def".to_string(), 0.9),
        ];

        let result = warmup_sharded_cache(&cache, vocab.into_iter(), tokenize);

        assert_eq!(result.entries_warmed, 2);
        assert_eq!(cache.get(&"abc".to_string()), Some(vec![97, 98, 99]));
        assert_eq!(cache.get(&"def".to_string()), Some(vec![100, 101, 102]));
    }

    #[test]
    fn test_warmup_multi_level() {
        let cache: MultiLevelCache<String, Vec<u32>> = MultiLevelCache::new();

        let tokenize = |s: &str| vec![s.len() as u32, 0];

        let vocab = vec![
            ("one".to_string(), 1.0),
            ("two".to_string(), 0.8),
        ];

        let result = warmup_multi_level_cache(&cache, vocab.into_iter(), tokenize);

        assert_eq!(result.entries_warmed, 2);
        assert_eq!(cache.get(&"one".to_string()), Some(vec![3, 0]));
        assert_eq!(cache.get(&"two".to_string()), Some(vec![3, 0]));

        // Check L1 hit
        let stats = cache.stats();
        assert!(stats.l1_hits >= 2);
    }

    #[test]
    fn test_warmup_with_failures() {
        let cache: ClockCache<String, Vec<u32>> = ClockCache::new(100);

        // Tokenizer that fails for strings starting with 'x'
        let tokenize = |s: &str| {
            if s.starts_with('x') {
                vec![] // Empty = failure
            } else {
                vec![s.len() as u32]
            }
        };

        let vocab = vec![
            ("good".to_string(), 1.0),
            ("xbad".to_string(), 0.9), // Will fail
            ("also_good".to_string(), 0.8),
        ];

        let result = warmup_cache(&cache, vocab.into_iter(), tokenize);

        assert_eq!(result.entries_warmed, 2);
        assert_eq!(result.entries_failed, 1);
    }

    #[test]
    fn test_warmup_from_tokens() {
        let cache: ClockCache<String, Vec<u32>> = ClockCache::new(100);
        let warmer = CacheWarmer::new(cache);

        let tokens = vec!["hello", "world", "test"];
        let tokenize = |s: &str| vec![s.len() as u32];

        let result = warmer.warmup_from_tokens(tokens, tokenize);

        assert_eq!(result.entries_warmed, 3);
        assert!(warmer.cache().get(&"hello".to_string()).is_some());
    }

    #[test]
    fn test_extract_frequent_tokens() {
        let vocab = vec![
            ("the".to_string(), 0u32),  // Most frequent (lowest ID)
            ("a".to_string(), 1),
            ("of".to_string(), 2),
            ("rare".to_string(), 1000), // Least frequent (highest ID)
        ];

        let frequent = extract_frequent_tokens(vocab.into_iter(), 2);

        assert_eq!(frequent.len(), 2);
        assert_eq!(frequent[0].0, "the");
        assert_eq!(frequent[1].0, "a");

        // Check frequencies are normalized (lower ID = higher freq)
        assert!(frequent[0].1 > frequent[1].1);
    }

    #[test]
    fn test_warmup_result() {
        let result = WarmupResult {
            entries_warmed: 100,
            entries_failed: 5,
            duration_ms: 50,
        };

        assert_eq!(result.entries_warmed, 100);
        assert_eq!(result.entries_failed, 5);
        assert_eq!(result.duration_ms, 50);
    }

    #[test]
    fn test_warmup_config_default() {
        let config = WarmupConfig::default();

        assert_eq!(config.max_entries, 10_000);
        assert_eq!(config.min_frequency, 0.0);
        assert_eq!(config.parallel_threads, 4);
        assert_eq!(config.batch_size, 256);
    }

    #[test]
    fn test_cache_insertable_trait() {
        // Test that all cache types implement CacheInsertable
        let clock_cache: ClockCache<String, Vec<u32>> = ClockCache::new(10);
        clock_cache.cache_insert("test".to_string(), vec![1, 2, 3]);
        assert_eq!(clock_cache.get(&"test".to_string()), Some(vec![1, 2, 3]));

        let sharded_cache: ShardedCache<String, Vec<u32>> = ShardedCache::new(100);
        sharded_cache.cache_insert("test".to_string(), vec![4, 5, 6]);
        assert_eq!(sharded_cache.get(&"test".to_string()), Some(vec![4, 5, 6]));

        let ml_cache: MultiLevelCache<String, Vec<u32>> = MultiLevelCache::new();
        ml_cache.cache_insert("test".to_string(), vec![7, 8, 9]);
        assert_eq!(ml_cache.get(&"test".to_string()), Some(vec![7, 8, 9]));
    }
}
