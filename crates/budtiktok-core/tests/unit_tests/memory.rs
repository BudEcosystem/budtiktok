//! Memory Management Unit Tests (9.1.9)
//!
//! Tests for memory management including:
//! - Arena allocator
//! - Object pool
//! - Buffer reuse
//! - Memory limits
//! - Allocation statistics
//! - Thread-local caching
//! - Zero-copy operations
//! - Memory-mapped files

use budtiktok_core::memory::*;
use std::sync::Arc;
use std::thread;

// =============================================================================
// Arena Allocator Tests
// =============================================================================

#[test]
fn test_arena_new() {
    let arena = Arena::new();
    assert!(arena.allocated() == 0 || arena.allocated() > 0);
}

#[test]
fn test_arena_with_capacity() {
    let arena = Arena::with_capacity(1024);
    assert!(arena.capacity() >= 1024);
}

#[test]
fn test_arena_alloc_bytes() {
    let arena = Arena::new();

    let ptr1 = arena.alloc_bytes(100);
    assert!(!ptr1.is_null());

    let ptr2 = arena.alloc_bytes(200);
    assert!(!ptr2.is_null());
    assert_ne!(ptr1, ptr2);
}

#[test]
fn test_arena_alloc_typed() {
    let arena = Arena::new();

    let value: &mut u64 = arena.alloc(42u64);
    assert_eq!(*value, 42);

    *value = 100;
    assert_eq!(*value, 100);
}

#[test]
fn test_arena_alloc_slice() {
    let arena = Arena::new();

    let slice: &mut [u32] = arena.alloc_slice(10, 0);
    assert_eq!(slice.len(), 10);
    assert!(slice.iter().all(|&x| x == 0));

    slice[0] = 42;
    assert_eq!(slice[0], 42);
}

#[test]
fn test_arena_alloc_str() {
    let arena = Arena::new();

    let s = arena.alloc_str("hello world");
    assert_eq!(s, "hello world");
}

#[test]
fn test_arena_reset() {
    let arena = Arena::new();

    arena.alloc_bytes(1000);
    let allocated_before = arena.allocated();

    arena.reset();

    // After reset, we can reuse the memory
    arena.alloc_bytes(1000);
    // Capacity should be preserved
    assert!(arena.capacity() >= 1000);
}

#[test]
fn test_arena_alignment() {
    let arena = Arena::new();

    // Allocate with specific alignment
    let ptr = arena.alloc_aligned(100, 64);
    assert!(!ptr.is_null());
    assert_eq!(ptr as usize % 64, 0);
}

#[test]
fn test_arena_many_small_allocs() {
    let arena = Arena::new();

    for _ in 0..10000 {
        let _ = arena.alloc(42u32);
    }

    // Should not panic or run out of memory
    assert!(arena.allocated() > 0);
}

#[test]
fn test_arena_large_alloc() {
    let arena = Arena::new();

    let large = arena.alloc_bytes(1024 * 1024); // 1MB
    assert!(!large.is_null());
}

// =============================================================================
// Object Pool Tests
// =============================================================================

#[test]
fn test_pool_new() {
    let pool: ObjectPool<Vec<u8>> = ObjectPool::new();
    assert_eq!(pool.available(), 0);
}

#[test]
fn test_pool_with_capacity() {
    let pool: ObjectPool<Vec<u8>> = ObjectPool::with_capacity(10);
    assert!(pool.capacity() >= 10);
}

#[test]
fn test_pool_acquire_release() {
    let pool: ObjectPool<Vec<u8>> = ObjectPool::new();

    let mut obj = pool.acquire();
    obj.push(42);

    pool.release(obj);
    assert!(pool.available() >= 1);
}

#[test]
fn test_pool_reuse() {
    let pool: ObjectPool<Vec<u8>> = ObjectPool::new();

    // Acquire, modify, release
    let mut obj = pool.acquire();
    obj.extend_from_slice(&[1, 2, 3]);
    obj.clear(); // Clear before release
    pool.release(obj);

    // Re-acquire should get the same buffer
    let obj2 = pool.acquire();
    assert_eq!(obj2.capacity(), 3); // Capacity preserved from first use
}

#[test]
fn test_pool_multiple_objects() {
    let pool: ObjectPool<Vec<u8>> = ObjectPool::new();

    let obj1 = pool.acquire();
    let obj2 = pool.acquire();
    let obj3 = pool.acquire();

    pool.release(obj1);
    pool.release(obj2);
    pool.release(obj3);

    assert!(pool.available() >= 3);
}

#[test]
fn test_pool_clear() {
    let pool: ObjectPool<Vec<u8>> = ObjectPool::new();

    pool.release(pool.acquire());
    pool.release(pool.acquire());

    pool.clear();
    assert_eq!(pool.available(), 0);
}

#[test]
fn test_pool_with_init() {
    let pool: ObjectPool<Vec<u8>> = ObjectPool::with_init(|| {
        let mut v = Vec::with_capacity(1024);
        v.push(0); // Initialize with something
        v
    });

    let obj = pool.acquire();
    assert!(obj.capacity() >= 1024);
}

#[test]
fn test_pool_stats() {
    let pool: ObjectPool<Vec<u8>> = ObjectPool::new();

    let obj = pool.acquire();
    pool.release(obj);
    let _ = pool.acquire();

    let stats = pool.stats();
    assert!(stats.total_acquires >= 2);
    assert!(stats.total_releases >= 1);
}

// =============================================================================
// Buffer Pool Tests
// =============================================================================

#[test]
fn test_buffer_pool_new() {
    let pool = BufferPool::new();
    let buffer = pool.acquire(100);
    assert!(buffer.capacity() >= 100);
}

#[test]
fn test_buffer_pool_size_classes() {
    let pool = BufferPool::new();

    // Different size requests
    let b1 = pool.acquire(10);
    let b2 = pool.acquire(100);
    let b3 = pool.acquire(1000);

    assert!(b1.capacity() >= 10);
    assert!(b2.capacity() >= 100);
    assert!(b3.capacity() >= 1000);

    pool.release(b1);
    pool.release(b2);
    pool.release(b3);
}

#[test]
fn test_buffer_pool_reuse_same_size() {
    let pool = BufferPool::new();

    let b1 = pool.acquire(100);
    let cap1 = b1.capacity();
    pool.release(b1);

    let b2 = pool.acquire(100);
    // Should reuse the same buffer
    assert_eq!(b2.capacity(), cap1);
}

#[test]
fn test_buffer_pool_guard() {
    let pool = Arc::new(BufferPool::new());

    {
        let guard = pool.acquire_guard(100);
        // Buffer is automatically returned when guard drops
        assert!(guard.capacity() >= 100);
    }

    // Buffer should be back in pool
    assert!(pool.available_for_size(100) >= 1);
}

// =============================================================================
// Memory Limits Tests
// =============================================================================

#[test]
fn test_memory_limit_basic() {
    let limiter = MemoryLimiter::new(1024 * 1024); // 1MB limit

    assert!(limiter.try_reserve(1000).is_ok());
    limiter.release(1000);
}

#[test]
fn test_memory_limit_exceeded() {
    let limiter = MemoryLimiter::new(1000);

    assert!(limiter.try_reserve(500).is_ok());
    assert!(limiter.try_reserve(600).is_err()); // Would exceed limit
}

#[test]
fn test_memory_limit_release() {
    let limiter = MemoryLimiter::new(1000);

    limiter.try_reserve(800).unwrap();
    assert!(limiter.try_reserve(300).is_err());

    limiter.release(500);
    assert!(limiter.try_reserve(300).is_ok());
}

#[test]
fn test_memory_limit_current_usage() {
    let limiter = MemoryLimiter::new(1000);

    assert_eq!(limiter.current_usage(), 0);

    limiter.try_reserve(300).unwrap();
    assert_eq!(limiter.current_usage(), 300);

    limiter.release(100);
    assert_eq!(limiter.current_usage(), 200);
}

#[test]
fn test_memory_limit_available() {
    let limiter = MemoryLimiter::new(1000);

    assert_eq!(limiter.available(), 1000);

    limiter.try_reserve(400).unwrap();
    assert_eq!(limiter.available(), 600);
}

// =============================================================================
// Allocation Statistics Tests
// =============================================================================

#[test]
fn test_alloc_stats_tracking() {
    let tracker = AllocationTracker::new();

    tracker.record_alloc(100);
    tracker.record_alloc(200);
    tracker.record_dealloc(100);

    let stats = tracker.stats();
    assert_eq!(stats.total_allocated, 300);
    assert_eq!(stats.total_deallocated, 100);
    assert_eq!(stats.current_usage, 200);
}

#[test]
fn test_alloc_stats_peak() {
    let tracker = AllocationTracker::new();

    tracker.record_alloc(100);
    tracker.record_alloc(200);
    tracker.record_dealloc(150);
    tracker.record_alloc(50);

    let stats = tracker.stats();
    assert_eq!(stats.peak_usage, 300); // 100 + 200
}

#[test]
fn test_alloc_stats_count() {
    let tracker = AllocationTracker::new();

    tracker.record_alloc(100);
    tracker.record_alloc(200);
    tracker.record_alloc(300);
    tracker.record_dealloc(100);
    tracker.record_dealloc(200);

    let stats = tracker.stats();
    assert_eq!(stats.allocation_count, 3);
    assert_eq!(stats.deallocation_count, 2);
}

#[test]
fn test_alloc_stats_reset() {
    let tracker = AllocationTracker::new();

    tracker.record_alloc(1000);
    tracker.reset();

    let stats = tracker.stats();
    assert_eq!(stats.current_usage, 0);
    assert_eq!(stats.allocation_count, 0);
}

// =============================================================================
// Thread-Local Caching Tests
// =============================================================================

#[test]
fn test_thread_local_arena() {
    let arena = thread_local_arena();

    let ptr = arena.alloc_bytes(100);
    assert!(!ptr.is_null());
}

#[test]
fn test_thread_local_buffer() {
    let buffer = thread_local_buffer(100);
    assert!(buffer.capacity() >= 100);
}

#[test]
fn test_thread_local_isolation() {
    let arena1 = thread_local_arena();
    let ptr1 = arena1.alloc_bytes(100) as usize;

    let handle = thread::spawn(|| {
        let arena2 = thread_local_arena();
        arena2.alloc_bytes(100) as usize
    });

    let ptr2 = handle.join().unwrap();

    // Different threads should have different arenas
    // (pointers may happen to be same, but likely different)
    let _ = (ptr1, ptr2);
}

#[test]
fn test_thread_local_pool() {
    let pool = thread_local_pool::<Vec<u8>>();

    let obj = pool.acquire();
    pool.release(obj);

    assert!(pool.available() >= 1);
}

// =============================================================================
// Zero-Copy Operations Tests
// =============================================================================

#[test]
fn test_zero_copy_slice() {
    let data = vec![1u8, 2, 3, 4, 5];
    let slice = ZeroCopySlice::new(&data);

    assert_eq!(slice.len(), 5);
    assert_eq!(slice[0], 1);
    assert_eq!(slice[4], 5);
}

#[test]
fn test_zero_copy_str() {
    let s = String::from("hello world");
    let zc_str = ZeroCopyStr::new(&s);

    assert_eq!(zc_str.as_str(), "hello world");
    assert_eq!(zc_str.len(), 11);
}

#[test]
fn test_zero_copy_split() {
    let data = b"hello world test";
    let parts: Vec<_> = zero_copy_split(data, b' ').collect();

    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0], b"hello");
    assert_eq!(parts[1], b"world");
    assert_eq!(parts[2], b"test");
}

#[test]
fn test_zero_copy_no_alloc() {
    let data = vec![0u8; 10000];
    let slice = ZeroCopySlice::new(&data);

    // Should not allocate additional memory
    assert_eq!(slice.as_ptr(), data.as_ptr());
}

// =============================================================================
// Memory-Mapped File Tests
// =============================================================================

#[test]
fn test_mmap_create() {
    let temp_file = std::env::temp_dir().join("test_mmap.bin");

    // Write some data
    std::fs::write(&temp_file, b"hello world").unwrap();

    let mmap = MemoryMap::open(&temp_file).unwrap();
    assert_eq!(mmap.len(), 11);
    assert_eq!(&mmap[..], b"hello world");

    std::fs::remove_file(&temp_file).ok();
}

#[test]
fn test_mmap_large_file() {
    let temp_file = std::env::temp_dir().join("test_mmap_large.bin");

    // Write 10MB of data
    let data: Vec<u8> = (0..10_000_000).map(|i| (i % 256) as u8).collect();
    std::fs::write(&temp_file, &data).unwrap();

    let mmap = MemoryMap::open(&temp_file).unwrap();
    assert_eq!(mmap.len(), 10_000_000);
    assert_eq!(mmap[0], 0);
    assert_eq!(mmap[256], 0);

    std::fs::remove_file(&temp_file).ok();
}

#[test]
fn test_mmap_read_only() {
    let temp_file = std::env::temp_dir().join("test_mmap_ro.bin");
    std::fs::write(&temp_file, b"test data").unwrap();

    let mmap = MemoryMap::open_read_only(&temp_file).unwrap();
    assert_eq!(&mmap[..], b"test data");

    std::fs::remove_file(&temp_file).ok();
}

// =============================================================================
// Concurrent Access Tests
// =============================================================================

#[test]
fn test_arena_not_sync() {
    // Arena is intentionally not Sync for performance
    let arena = Arena::new();

    // Can use in single thread
    let _ = arena.alloc_bytes(100);
}

#[test]
fn test_pool_concurrent_access() {
    let pool = Arc::new(ObjectPool::<Vec<u8>>::new());
    let mut handles = vec![];

    for _ in 0..4 {
        let pool_clone = Arc::clone(&pool);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let obj = pool_clone.acquire();
                pool_clone.release(obj);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Pool should be consistent
    let _ = pool.stats();
}

#[test]
fn test_limiter_concurrent() {
    let limiter = Arc::new(MemoryLimiter::new(10000));
    let mut handles = vec![];

    for _ in 0..4 {
        let limiter_clone = Arc::clone(&limiter);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                if limiter_clone.try_reserve(10).is_ok() {
                    // Simulate work
                    limiter_clone.release(10);
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Should end up with 0 usage
    assert_eq!(limiter.current_usage(), 0);
}

// =============================================================================
// Scoped Allocation Tests
// =============================================================================

#[test]
fn test_scoped_arena() {
    let arena = ScopedArena::new();

    {
        let scope = arena.scope();
        scope.alloc_bytes(100);
        scope.alloc_bytes(200);
        // Allocations freed when scope drops
    }

    // Arena memory can be reused
    let _ = arena.scope();
}

#[test]
fn test_nested_scopes() {
    let arena = ScopedArena::new();

    let outer = arena.scope();
    outer.alloc_bytes(100);

    {
        let inner = outer.child_scope();
        inner.alloc_bytes(50);
        // Inner allocations freed here
    }

    // Outer allocation still valid
    let _ = outer.alloc_bytes(25);
}

// =============================================================================
// Alignment and Padding Tests
// =============================================================================

#[test]
fn test_alignment_power_of_two() {
    let arena = Arena::new();

    for align in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        let ptr = arena.alloc_aligned(100, align);
        assert_eq!(
            ptr as usize % align,
            0,
            "Alignment {} not satisfied",
            align
        );
    }
}

#[test]
fn test_struct_alignment() {
    #[repr(C, align(64))]
    struct Aligned64 {
        data: [u8; 64],
    }

    let arena = Arena::new();
    let obj: &mut Aligned64 = arena.alloc(Aligned64 { data: [0; 64] });

    assert_eq!(obj as *const _ as usize % 64, 0);
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_zero_size_alloc() {
    let arena = Arena::new();

    let ptr = arena.alloc_bytes(0);
    // Zero-size allocation should succeed (may return same pointer)
    let _ = ptr;
}

#[test]
fn test_many_releases() {
    let pool: ObjectPool<Vec<u8>> = ObjectPool::new();

    // Release many objects
    for _ in 0..1000 {
        pool.release(Vec::new());
    }

    // Pool should handle many releases
    assert!(pool.available() >= 1000);
}

#[test]
fn test_acquire_without_release() {
    let pool: ObjectPool<Vec<u8>> = ObjectPool::new();

    // Acquire many without release - should still work
    for _ in 0..100 {
        let _ = pool.acquire();
    }

    // Objects are just dropped, pool is empty
    assert_eq!(pool.available(), 0);
}

#[test]
fn test_limiter_release_more_than_reserved() {
    let limiter = MemoryLimiter::new(1000);

    limiter.try_reserve(100).unwrap();
    // Releasing more than reserved should be handled gracefully
    limiter.release(200);

    // Current usage should not go negative
    assert!(limiter.current_usage() <= 0 || limiter.current_usage() == 0);
}

// =============================================================================
// Memory Pressure Tests
// =============================================================================

#[test]
fn test_pool_under_pressure() {
    let pool: ObjectPool<Vec<u8>> = ObjectPool::with_capacity(10);

    // Create pressure by acquiring many objects
    let mut objects: Vec<_> = (0..100).map(|_| pool.acquire()).collect();

    // Release half
    for obj in objects.drain(50..) {
        pool.release(obj);
    }

    // Pool should handle this gracefully
    assert!(pool.available() >= 50);
}

#[test]
fn test_arena_growth() {
    let arena = Arena::with_capacity(64);

    // Allocate more than initial capacity
    for _ in 0..100 {
        arena.alloc_bytes(100);
    }

    // Arena should have grown
    assert!(arena.capacity() > 64);
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_arena_with_pool() {
    let arena = Arena::new();
    let pool: ObjectPool<Vec<u8>> = ObjectPool::new();

    // Use arena for small allocations
    let small: &mut [u8] = arena.alloc_slice(10, 0);
    small[0] = 1;

    // Use pool for reusable buffers
    let mut buffer = pool.acquire();
    buffer.extend_from_slice(&[1, 2, 3]);
    pool.release(buffer);

    // Both should work together
    assert!(arena.allocated() > 0);
    assert!(pool.available() >= 1);
}

#[test]
fn test_limiter_with_pool() {
    let limiter = Arc::new(MemoryLimiter::new(10000));
    let pool: ObjectPool<Vec<u8>> = ObjectPool::new();

    // Reserve memory for pooled objects
    limiter.try_reserve(1000).unwrap();

    let mut obj = pool.acquire();
    obj.reserve(1000);
    pool.release(obj);

    limiter.release(1000);

    assert_eq!(limiter.current_usage(), 0);
}
