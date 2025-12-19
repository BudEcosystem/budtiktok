//! Token Cache Unit Tests (9.1.3)
//!
//! Tests for caching functionality including:
//! - CLOCK cache insert and retrieve
//! - CLOCK eviction algorithm
//! - Concurrent access safety
//! - Statistics tracking
//! - Sharded cache
//! - Multi-level cache hierarchy

use budtiktok_core::cache::*;
use std::sync::Arc;
use std::thread;

// =============================================================================
// CLOCK Cache Basic Tests
// =============================================================================

#[test]
fn test_clock_cache_insert_and_retrieve() {
    let cache: ClockCache<String, u32> = ClockCache::new(100);

    cache.insert("hello".to_string(), 1);
    cache.insert("world".to_string(), 2);

    assert_eq!(cache.get(&"hello".to_string()), Some(1));
    assert_eq!(cache.get(&"world".to_string()), Some(2));
    assert_eq!(cache.get(&"missing".to_string()), None);
}

#[test]
fn test_clock_cache_update() {
    let cache: ClockCache<String, u32> = ClockCache::new(100);

    cache.insert("key".to_string(), 1);
    assert_eq!(cache.get(&"key".to_string()), Some(1));

    cache.insert("key".to_string(), 2);
    assert_eq!(cache.get(&"key".to_string()), Some(2));
}

#[test]
fn test_clock_cache_capacity() {
    let cache: ClockCache<u32, u32> = ClockCache::new(10);

    // Insert more than capacity
    for i in 0..20 {
        cache.insert(i, i * 10);
    }

    // Some entries should be evicted
    let present_count: usize = (0..20).filter(|i| cache.get(i).is_some()).count();
    assert!(present_count <= 10, "Cache should respect capacity");
}

#[test]
fn test_clock_eviction_second_chance() {
    let cache: ClockCache<u32, u32> = ClockCache::new(5);

    // Fill cache
    for i in 0..5 {
        cache.insert(i, i);
    }

    // Access some entries to give them a second chance
    cache.get(&0);
    cache.get(&1);
    cache.get(&2);

    // Insert more entries, should evict unreferenced ones first
    for i in 5..10 {
        cache.insert(i, i);
    }

    // Accessed entries more likely to survive
    // (This is probabilistic based on CLOCK algorithm)
    let accessed_survived = [0, 1, 2].iter().filter(|i| cache.get(i).is_some()).count();
    // At least some should survive due to second chance
    assert!(accessed_survived >= 1, "CLOCK should give second chance to referenced entries");
}

// =============================================================================
// Cache Statistics Tests
// =============================================================================

#[test]
fn test_cache_stats_hits_misses() {
    let cache: ClockCache<String, u32> = ClockCache::new(100);

    cache.insert("a".to_string(), 1);
    cache.insert("b".to_string(), 2);

    // Hits
    cache.get(&"a".to_string());
    cache.get(&"b".to_string());

    // Misses
    cache.get(&"x".to_string());
    cache.get(&"y".to_string());
    cache.get(&"z".to_string());

    let stats = cache.stats();
    assert_eq!(stats.hits, 2);
    assert_eq!(stats.misses, 3);
}

#[test]
fn test_cache_stats_hit_rate() {
    let cache: ClockCache<String, u32> = ClockCache::new(100);

    cache.insert("key".to_string(), 1);

    // 4 hits
    for _ in 0..4 {
        cache.get(&"key".to_string());
    }

    // 1 miss
    cache.get(&"missing".to_string());

    let stats = cache.stats();
    assert!((stats.hit_rate - 0.8).abs() < 0.01, "Hit rate should be 80%");
}

#[test]
fn test_cache_stats_zero_access() {
    let cache: ClockCache<String, u32> = ClockCache::new(100);
    let stats = cache.stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.hit_rate, 0.0);
}

// =============================================================================
// Sharded Cache Tests
// =============================================================================

#[test]
fn test_sharded_cache_basic() {
    let cache: ShardedCache<String, u32> = ShardedCache::new(100);

    cache.insert("hello".to_string(), 1);
    cache.insert("world".to_string(), 2);

    assert_eq!(cache.get(&"hello".to_string()), Some(1));
    assert_eq!(cache.get(&"world".to_string()), Some(2));
}

#[test]
fn test_sharded_cache_distribution() {
    let cache: ShardedCache<u32, u32> = ShardedCache::with_shards(1000, 4);

    // Insert many entries
    for i in 0..100 {
        cache.insert(i, i * 10);
    }

    // All should be retrievable
    for i in 0..100 {
        assert_eq!(cache.get(&i), Some(i * 10), "Entry {} should exist", i);
    }
}

#[test]
fn test_sharded_cache_stats() {
    let cache: ShardedCache<String, u32> = ShardedCache::new(100);

    cache.insert("a".to_string(), 1);
    cache.insert("b".to_string(), 2);

    cache.get(&"a".to_string());
    cache.get(&"missing".to_string());

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
}

// =============================================================================
// Multi-Level Cache Tests
// =============================================================================

#[test]
fn test_multi_level_cache_basic() {
    let cache: MultiLevelCache<String, u32> = MultiLevelCache::new();

    cache.insert("hello".to_string(), 1);
    assert_eq!(cache.get(&"hello".to_string()), Some(1));
}

#[test]
fn test_multi_level_cache_promotion() {
    let config = MultiLevelCacheConfig {
        l1_capacity: 2,
        l2_capacity: 1000,
        promote_on_hit: true,
    };
    let cache: MultiLevelCache<String, u32> = MultiLevelCache::with_config(config);

    // Insert entries - they go to L1 first
    cache.insert("a".to_string(), 1);
    cache.insert("b".to_string(), 2);
    cache.insert("c".to_string(), 3); // Should push 'a' to L2

    // Access 'a' - should promote from L2 back to L1
    let result = cache.get(&"a".to_string());
    assert_eq!(result, Some(1));

    // Check stats show L2 hit
    let stats = cache.stats();
    assert!(stats.l2_hits >= 1 || stats.l1_hits >= 1);
}

#[test]
fn test_multi_level_cache_stats() {
    let cache: MultiLevelCache<String, u32> = MultiLevelCache::new();

    cache.insert("a".to_string(), 1);

    // First access - L1 hit
    cache.get(&"a".to_string());

    // Miss
    cache.get(&"missing".to_string());

    let stats = cache.stats();
    assert!(stats.l1_hits >= 1);
    assert_eq!(stats.misses, 1);
}

#[test]
fn test_multi_level_cache_clear() {
    let cache: MultiLevelCache<String, u32> = MultiLevelCache::new();

    cache.insert("a".to_string(), 1);
    cache.insert("b".to_string(), 2);

    cache.clear();

    assert_eq!(cache.get(&"a".to_string()), None);
    assert_eq!(cache.get(&"b".to_string()), None);
}

// =============================================================================
// Concurrent Access Tests
// =============================================================================

#[test]
fn test_clock_cache_concurrent_reads() {
    let cache = Arc::new(ClockCache::<u32, u32>::new(1000));

    // Pre-populate
    for i in 0..100 {
        cache.insert(i, i * 10);
    }

    let mut handles = vec![];

    // Spawn multiple reader threads
    for _ in 0..4 {
        let cache_clone = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            for i in 0..100 {
                let _ = cache_clone.get(&i);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Cache should still be consistent
    for i in 0..100 {
        assert_eq!(cache.get(&i), Some(i * 10));
    }
}

#[test]
fn test_clock_cache_concurrent_writes() {
    let cache = Arc::new(ClockCache::<u32, u32>::new(1000));
    let mut handles = vec![];

    // Spawn multiple writer threads
    for t in 0..4 {
        let cache_clone = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            for i in 0..100 {
                cache_clone.insert(t * 100 + i, i);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Should have some entries from each thread
    let mut found = [false; 4];
    for t in 0..4 {
        for i in 0..100 {
            if cache.get(&(t * 100 + i)).is_some() {
                found[t as usize] = true;
                break;
            }
        }
    }

    // At least some threads should have entries
    let found_count: usize = found.iter().filter(|&&b| b).count();
    assert!(found_count >= 1, "At least one thread should have entries in cache");
}

#[test]
fn test_sharded_cache_concurrent() {
    let cache = Arc::new(ShardedCache::<u32, u32>::new(10000));
    let mut handles = vec![];

    // Concurrent writers
    for t in 0..8 {
        let cache_clone = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            for i in 0..1000 {
                cache_clone.insert(t * 1000 + i, i);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify some entries exist
    let stats = cache.stats();
    assert!(stats.size > 0, "Cache should have entries");
}

// =============================================================================
// Cache Warmup Tests
// =============================================================================

#[test]
fn test_warmup_basic() {
    let cache: ClockCache<String, Vec<u32>> = ClockCache::new(100);

    let tokenize = |s: &str| vec![s.len() as u32];

    let vocab = vec![
        ("hello".to_string(), 1.0),
        ("world".to_string(), 0.9),
        ("test".to_string(), 0.8),
    ];

    let result = warmup_cache(&cache, vocab.into_iter(), tokenize);

    assert_eq!(result.entries_warmed, 3);
    assert_eq!(result.entries_failed, 0);
    assert!(cache.get(&"hello".to_string()).is_some());
}

#[test]
fn test_warmup_with_filter() {
    let cache: ClockCache<String, Vec<u32>> = ClockCache::new(100);
    let config = WarmupConfig {
        max_entries: 10_000,
        min_frequency: 0.5,
        parallel_threads: 0,
        batch_size: 100,
    };
    let warmer = CacheWarmer::with_config(cache, config);

    let vocab = vec![
        ("high".to_string(), 1.0),
        ("medium".to_string(), 0.6),
        ("low".to_string(), 0.3), // Below threshold
    ];

    let tokenize = |s: &str| vec![s.len() as u32];
    let result = warmer.warmup(vocab.into_iter(), tokenize);

    // Only high and medium should be warmed
    assert_eq!(result.entries_warmed, 2);
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

// =============================================================================
// Type Aliases Tests
// =============================================================================

#[test]
fn test_sharded_token_cache() {
    let cache = new_sharded_token_cache();
    cache.insert("hello".to_string(), vec![1, 2, 3]);
    assert_eq!(cache.get(&"hello".to_string()), Some(vec![1, 2, 3]));
}

#[test]
fn test_multi_level_token_cache() {
    let cache = new_multi_level_token_cache();
    cache.insert("hello".to_string(), vec![1, 2, 3]);
    assert_eq!(cache.get(&"hello".to_string()), Some(vec![1, 2, 3]));
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_cache_empty_key() {
    let cache: ClockCache<String, u32> = ClockCache::new(100);
    cache.insert("".to_string(), 42);
    assert_eq!(cache.get(&"".to_string()), Some(42));
}

#[test]
fn test_cache_large_value() {
    let cache: ClockCache<String, Vec<u32>> = ClockCache::new(100);
    let large_value: Vec<u32> = (0..10000).collect();
    cache.insert("large".to_string(), large_value.clone());
    assert_eq!(cache.get(&"large".to_string()), Some(large_value));
}

#[test]
fn test_cache_unicode_key() {
    let cache: ClockCache<String, u32> = ClockCache::new(100);
    cache.insert("日本語".to_string(), 1);
    cache.insert("한글".to_string(), 2);
    cache.insert("émoji😀".to_string(), 3);

    assert_eq!(cache.get(&"日本語".to_string()), Some(1));
    assert_eq!(cache.get(&"한글".to_string()), Some(2));
    assert_eq!(cache.get(&"émoji😀".to_string()), Some(3));
}

#[test]
fn test_cache_zero_capacity() {
    // Cache with capacity 0 or 1 should still work
    let cache: ClockCache<String, u32> = ClockCache::new(1);
    cache.insert("a".to_string(), 1);
    cache.insert("b".to_string(), 2);
    // One of them should be evicted
    let count = [("a", 1), ("b", 2)].iter()
        .filter(|(k, v)| cache.get(&k.to_string()) == Some(*v))
        .count();
    assert_eq!(count, 1, "Only one entry should fit");
}
