//! Memory Regression Tests (9.3.3)
//!
//! Tests that ensure memory usage doesn't regress.
//! These tests check for memory leaks and excessive allocation.
//!
//! Note: For production, use miri or heaptrack. These tests provide
//! basic sanity checks during development.

use budtiktok_core::unicode::*;
use budtiktok_core::memory::*;

// =============================================================================
// Arena Allocator Tests
// =============================================================================

#[test]
fn test_memory_arena_basic() {
    // Arena should allocate and reset without leaking
    with_arena(|arena| {
        // Allocate some data
        let s1: &str = arena.alloc_str("hello");
        let s2: &str = arena.alloc_str("world");

        assert_eq!(s1, "hello");
        assert_eq!(s2, "world");
    });

    // After with_arena, memory should be released/reset
}

#[test]
fn test_memory_arena_repeated_use() {
    // Repeated arena use should not accumulate memory
    for _ in 0..1000 {
        with_arena(|arena| {
            // Allocate 1KB per iteration
            for _ in 0..10 {
                let _s: &str = arena.alloc_str(&"x".repeat(100));
            }
        });
    }

    // If this completes without OOM, memory is being reclaimed
}

// =============================================================================
// String Interner Tests
// =============================================================================

#[test]
fn test_memory_interner_deduplication() {
    // Interner should deduplicate strings
    let s1 = intern("hello");
    let s2 = intern("hello");
    let s3 = intern("world");

    // Same string should return same reference
    assert_eq!(s1, s2);
    assert_ne!(s1, s3);

    // Check stats show deduplication
    let stats = global_interner_stats();
    // At least some strings should be interned
    assert!(stats.interned >= 2);
}

#[test]
fn test_memory_interner_no_duplicate_storage() {
    // Interning the same string many times should not grow memory
    let initial_stats = global_interner_stats();

    for _ in 0..1000 {
        let _ = intern("repeated_string");
    }

    let final_stats = global_interner_stats();

    // Storage should only increase by 1 string, not 1000
    // Note: exact behavior depends on interner implementation
    assert!(
        final_stats.interned <= initial_stats.interned + 10,
        "Interner appears to be duplicating strings: {} -> {}",
        initial_stats.interned,
        final_stats.interned
    );
}

// =============================================================================
// Buffer Pool Tests
// =============================================================================

#[test]
fn test_memory_buffer_pool_reuse() {
    // Buffer pool should reuse allocations
    let initial_capacity = {
        with_u32_buffer(|buf| {
            buf.capacity()
        })
    };

    // Use and return buffer multiple times
    for _ in 0..100 {
        with_u32_buffer(|buf| {
            buf.extend(0..100);
            buf.len()
        });
    }

    // Capacity should be stable (reusing same buffer)
    let final_capacity = {
        with_u32_buffer(|buf| {
            buf.capacity()
        })
    };

    // Allow some growth but not unlimited
    assert!(
        final_capacity <= initial_capacity.max(100) * 2,
        "Buffer pool appears to not be reusing buffers"
    );
}

#[test]
fn test_memory_buffer_pool_cleanup() {
    // Use buffer with large allocation
    with_u32_buffer(|buf| {
        buf.extend(0..10000);
    });

    // Next use should get clean buffer
    with_u32_buffer(|buf| {
        assert!(buf.is_empty(), "Buffer not cleaned between uses");
    });
}

// =============================================================================
// Encoding Memory Tests
// =============================================================================

#[test]
fn test_memory_encoding_growth() {
    use budtiktok_core::Encoding;

    // Create encoding and grow it
    let mut encoding = Encoding::with_capacity(10);

    for i in 0..100 {
        encoding.push(
            i,
            format!("token{}", i),
            (i as usize, (i + 1) as usize),
            Some(i),
            Some(0),
            false,
        );
    }

    assert_eq!(encoding.len(), 100);
}

#[test]
fn test_memory_encoding_truncate_releases() {
    use budtiktok_core::Encoding;

    // Truncation should not leak memory
    for _ in 0..100 {
        let mut encoding = Encoding::with_capacity(100);

        for i in 0..100 {
            encoding.push(
                i,
                format!("token{}", i),
                (i as usize, (i + 1) as usize),
                Some(i),
                Some(0),
                false,
            );
        }

        encoding.truncate(10, 0);
        assert_eq!(encoding.len(), 10);
    }

    // If this completes without OOM, truncation is working
}

// =============================================================================
// Normalization Memory Tests
// =============================================================================

#[test]
fn test_memory_normalize_ascii_no_alloc() {
    // ASCII normalization with fast path should be zero-copy
    use std::borrow::Cow;

    let text = "Hello, World!";
    let result = normalize_with_fast_path(text, NormalizationForm::NFC);

    // Should be borrowed (no allocation)
    assert!(
        matches!(result, Cow::Borrowed(_)),
        "ASCII normalization should use fast path (borrowed)"
    );
}

#[test]
fn test_memory_normalize_repeated() {
    // Repeated normalization should not accumulate memory
    let text = "héllo wörld";

    for _ in 0..10000 {
        let _ = normalize(text, NormalizationForm::NFC);
    }

    // If this completes without OOM, memory is being released
}

// =============================================================================
// Large Input Memory Tests
// =============================================================================

#[test]
fn test_memory_large_ascii_check() {
    // Large ASCII check should complete without issues
    let text = "a".repeat(10_000_000); // 10MB

    let result = is_ascii_fast(&text);

    assert!(result);
    // Memory for result (bool) is trivial
}

#[test]
fn test_memory_large_normalization() {
    // Large normalization should handle memory gracefully
    let text = "hello ".repeat(100_000); // ~600KB

    let result = normalize(&text, NormalizationForm::NFC);

    assert_eq!(result.len(), text.len());
    // Memory should be released after function returns
}

// =============================================================================
// Miri Integration Pattern
// =============================================================================

/// This test demonstrates the pattern for miri memory safety checks.
/// In CI, miri would be run with:
/// `cargo +nightly miri test`
///
/// Miri detects:
/// - Use after free
/// - Memory leaks
/// - Invalid pointer dereference
/// - Data races
#[test]
fn test_memory_miri_compatible() {
    // Simple operations that miri can verify

    // String allocation
    let s = String::from("hello");
    assert_eq!(s.len(), 5);

    // Vec operations
    let mut v = Vec::new();
    v.push(1);
    v.push(2);
    assert_eq!(v.len(), 2);

    // Arena usage
    with_arena(|arena| {
        let s: &str = arena.alloc_str("test");
        assert_eq!(s, "test");
    });
}

// =============================================================================
// Memory Limit Tests
// =============================================================================

#[test]
#[ignore = "Requires memory limit configuration"]
fn test_memory_limit_enforcement() {
    // Test that memory limits are enforced
    // let allocator = BoundedAllocator::new(1024 * 1024); // 1MB limit
    //
    // // This should fail with OOM
    // let result = allocator.try_alloc(2 * 1024 * 1024);
    // assert!(result.is_err());
    todo!("Implement memory limit enforcement test");
}

#[test]
#[ignore = "Requires heaptrack or similar"]
fn test_memory_no_leaks() {
    // This would be run with heaptrack to detect leaks
    // heaptrack ./target/release/budtiktok-bench
    //
    // Expected: 0 bytes leaked after processing
    todo!("Use heaptrack for leak detection");
}
