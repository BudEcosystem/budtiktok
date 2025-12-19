//! Memory allocator configuration
//!
//! This module configures the global memory allocator based on compile-time features.
//!
//! # Allocator Selection
//!
//! BudTikTok supports multiple allocators that can be selected via Cargo features:
//!
//! - **jemalloc** (Linux, recommended for servers): Excellent for multi-threaded workloads,
//!   provides thread-local caches and reduced fragmentation.
//! - **mimalloc** (cross-platform): Good performance across all platforms, especially
//!   on Windows and macOS.
//! - **system** (default): Uses the system allocator.
//!
//! # Usage
//!
//! Enable an allocator feature in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! budtiktok-core = { version = "*", features = ["jemalloc"] }
//! ```
//!
//! Or via the command line:
//!
//! ```bash
//! cargo build --release --features jemalloc
//! ```
//!
//! # Performance Considerations
//!
//! For tokenization workloads, alternative allocators typically provide:
//! - 10-30% throughput improvement on multi-threaded workloads
//! - Reduced memory fragmentation for long-running services
//! - Better scaling under high allocation pressure
//!
//! # Thread-Local Caches
//!
//! Both jemalloc and mimalloc use thread-local caches to reduce contention.
//! This is particularly beneficial for tokenization where each thread
//! frequently allocates small objects (token buffers, strings).

/// Configure jemalloc as the global allocator
///
/// jemalloc is recommended for Linux server deployments due to:
/// - Excellent multi-threaded performance
/// - Thread-local caches reduce contention
/// - Better memory fragmentation characteristics
/// - Tunable configuration options
#[cfg(all(feature = "jemalloc", not(feature = "mimalloc")))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Configure mimalloc as the global allocator
///
/// mimalloc is recommended for:
/// - Cross-platform deployments (Windows, macOS, Linux)
/// - Workloads with many small allocations
/// - When jemalloc is not available
#[cfg(all(feature = "mimalloc", not(feature = "jemalloc")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// When both features are enabled, prefer jemalloc on Unix, mimalloc on Windows
#[cfg(all(feature = "jemalloc", feature = "mimalloc", unix))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(feature = "jemalloc", feature = "mimalloc", windows))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Information about the configured allocator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocatorKind {
    /// System default allocator
    System,
    /// jemalloc allocator
    Jemalloc,
    /// mimalloc allocator
    MiMalloc,
}

impl AllocatorKind {
    /// Returns the name of the allocator
    pub fn name(&self) -> &'static str {
        match self {
            AllocatorKind::System => "system",
            AllocatorKind::Jemalloc => "jemalloc",
            AllocatorKind::MiMalloc => "mimalloc",
        }
    }
}

impl std::fmt::Display for AllocatorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Get the currently configured allocator
pub fn current_allocator() -> AllocatorKind {
    #[cfg(all(feature = "jemalloc", not(feature = "mimalloc")))]
    {
        AllocatorKind::Jemalloc
    }

    #[cfg(all(feature = "mimalloc", not(feature = "jemalloc")))]
    {
        AllocatorKind::MiMalloc
    }

    #[cfg(all(feature = "jemalloc", feature = "mimalloc", unix))]
    {
        AllocatorKind::Jemalloc
    }

    #[cfg(all(feature = "jemalloc", feature = "mimalloc", windows))]
    {
        AllocatorKind::MiMalloc
    }

    #[cfg(not(any(feature = "jemalloc", feature = "mimalloc")))]
    {
        AllocatorKind::System
    }
}

/// Check if a custom allocator is configured
pub fn has_custom_allocator() -> bool {
    current_allocator() != AllocatorKind::System
}

/// Allocator statistics (when available)
#[derive(Debug, Clone, Default)]
pub struct AllocatorStats {
    /// Total bytes allocated
    pub allocated: usize,
    /// Total bytes in active pages
    pub active: usize,
    /// Total bytes mapped from the OS
    pub mapped: usize,
    /// Total bytes retained (not returned to OS)
    pub retained: usize,
    /// Total bytes in resident pages
    pub resident: usize,
}

impl AllocatorStats {
    /// Get current allocator statistics
    ///
    /// Note: Statistics are only available for jemalloc.
    /// Returns default/zero values for other allocators.
    pub fn get() -> Self {
        #[cfg(feature = "jemalloc")]
        {
            get_jemalloc_stats()
        }

        #[cfg(not(feature = "jemalloc"))]
        {
            Self::default()
        }
    }
}

/// Get jemalloc statistics
#[cfg(feature = "jemalloc")]
fn get_jemalloc_stats() -> AllocatorStats {
    use tikv_jemallocator::Jemalloc;
    use std::alloc::{GlobalAlloc, Layout};

    // jemalloc stats require mallctl, which isn't exposed directly
    // For now, return defaults - full stats would require jemalloc-sys
    let _ = Jemalloc;

    AllocatorStats::default()
}

/// Memory pool hint for the allocator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationHint {
    /// Small, short-lived allocation (< 256 bytes)
    SmallTransient,
    /// Small, long-lived allocation (< 256 bytes)
    SmallPersistent,
    /// Medium allocation (256 bytes - 64KB)
    Medium,
    /// Large allocation (> 64KB)
    Large,
    /// Huge allocation (> 1MB, consider mmap)
    Huge,
}

impl AllocationHint {
    /// Get the recommended allocation strategy
    pub fn recommend_for_size(size: usize) -> Self {
        match size {
            0..=255 => AllocationHint::SmallTransient,
            256..=65535 => AllocationHint::Medium,
            65536..=1048575 => AllocationHint::Large,
            _ => AllocationHint::Huge,
        }
    }
}

/// Arena-based allocation pool for tokenization
///
/// Uses bumpalo for arena allocation which is much faster than
/// individual allocations for temporary tokenization buffers.
pub struct TokenArena {
    arena: bumpalo::Bump,
    allocation_count: std::sync::atomic::AtomicUsize,
}

impl TokenArena {
    /// Create a new token arena with default capacity
    pub fn new() -> Self {
        Self {
            arena: bumpalo::Bump::new(),
            allocation_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Create a new token arena with specified initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            arena: bumpalo::Bump::with_capacity(capacity),
            allocation_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Allocate a slice in the arena
    pub fn alloc_slice<T: Copy>(&self, slice: &[T]) -> &[T] {
        self.allocation_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.arena.alloc_slice_copy(slice)
    }

    /// Allocate a string in the arena
    pub fn alloc_str(&self, s: &str) -> &str {
        self.allocation_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.arena.alloc_str(s)
    }

    /// Allocate a Vec-like collection in the arena
    pub fn alloc_vec<T>(&self) -> bumpalo::collections::Vec<'_, T> {
        bumpalo::collections::Vec::new_in(&self.arena)
    }

    /// Get bytes allocated in this arena
    pub fn allocated_bytes(&self) -> usize {
        self.arena.allocated_bytes()
    }

    /// Get number of allocations made
    pub fn allocation_count(&self) -> usize {
        self.allocation_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Reset the arena, freeing all allocations
    pub fn reset(&mut self) {
        self.arena.reset();
        self.allocation_count.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Default for TokenArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_current_allocator() {
        let kind = current_allocator();
        // Should return something
        assert!(!kind.name().is_empty());
    }

    #[test]
    fn test_allocator_kind_display() {
        assert_eq!(AllocatorKind::System.to_string(), "system");
        assert_eq!(AllocatorKind::Jemalloc.to_string(), "jemalloc");
        assert_eq!(AllocatorKind::MiMalloc.to_string(), "mimalloc");
    }

    #[test]
    fn test_allocator_stats() {
        let stats = AllocatorStats::get();
        // Stats might be zero for system allocator, but should not panic
        let _ = stats.allocated;
    }

    #[test]
    fn test_allocation_hint() {
        assert_eq!(AllocationHint::recommend_for_size(64), AllocationHint::SmallTransient);
        assert_eq!(AllocationHint::recommend_for_size(1024), AllocationHint::Medium);
        assert_eq!(AllocationHint::recommend_for_size(100_000), AllocationHint::Large);
        assert_eq!(AllocationHint::recommend_for_size(10_000_000), AllocationHint::Huge);
    }

    #[test]
    fn test_token_arena() {
        let arena = TokenArena::new();

        // Allocate strings
        let s1 = arena.alloc_str("hello");
        let s2 = arena.alloc_str("world");

        assert_eq!(s1, "hello");
        assert_eq!(s2, "world");
        assert_eq!(arena.allocation_count(), 2);
        assert!(arena.allocated_bytes() > 0);
    }

    #[test]
    fn test_token_arena_reset() {
        let mut arena = TokenArena::new();

        arena.alloc_str("hello");
        assert_eq!(arena.allocation_count(), 1);

        arena.reset();
        assert_eq!(arena.allocation_count(), 0);
    }

    #[test]
    fn test_token_arena_slice() {
        let arena = TokenArena::new();

        let ids: &[u32] = &[1, 2, 3, 4, 5];
        let allocated = arena.alloc_slice(ids);

        assert_eq!(allocated, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_token_arena_vec() {
        let arena = TokenArena::new();

        let mut vec = arena.alloc_vec::<u32>();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        assert_eq!(&vec[..], &[1, 2, 3]);
    }

    #[test]
    fn test_has_custom_allocator() {
        let has_custom = has_custom_allocator();
        let kind = current_allocator();

        assert_eq!(has_custom, kind != AllocatorKind::System);
    }
}
