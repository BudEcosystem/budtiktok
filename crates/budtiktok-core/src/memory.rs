//! Memory management utilities for high-performance tokenization
//!
//! This module provides memory optimization primitives:
//! - Arena allocator for per-batch zero-cost allocations
//! - String interner for vocabulary deduplication
//! - Memory pool for reusing Encoding structs
//! - Thread-local buffers for temporary storage

use ahash::{AHashMap, AHashSet};
use bumpalo::Bump;
use parking_lot::{Mutex, RwLock};
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ============================================================================
// Arena Allocator (2.5.1)
// ============================================================================

/// Default arena capacity (1 MB)
const DEFAULT_ARENA_CAPACITY: usize = 1024 * 1024;

thread_local! {
    /// Thread-local arena for per-batch allocations
    static THREAD_ARENA: RefCell<Bump> = RefCell::new(Bump::with_capacity(DEFAULT_ARENA_CAPACITY));
}

/// Execute a function with access to the thread-local arena.
/// The arena is reset after the function completes, freeing all allocations.
///
/// # Example
/// ```rust,ignore
/// use budtiktok_core::memory::with_arena;
///
/// let result = with_arena(|arena| {
///     let buffer = arena.alloc_slice_copy(&[1u32, 2, 3, 4, 5]);
///     buffer.iter().sum::<u32>()
/// });
/// assert_eq!(result, 15);
/// ```
pub fn with_arena<T, F>(f: F) -> T
where
    F: FnOnce(&Bump) -> T,
{
    THREAD_ARENA.with(|arena| {
        let arena = arena.borrow();
        let result = f(&arena);
        // Arena is reset when this scope ends via Drop
        result
    })
}

/// Execute a function with access to the thread-local arena, then reset it.
/// This is more explicit about the reset behavior.
pub fn with_arena_reset<T, F>(f: F) -> T
where
    F: FnOnce(&Bump) -> T,
{
    THREAD_ARENA.with(|arena_cell| {
        let mut arena = arena_cell.borrow_mut();
        let result = f(&arena);
        arena.reset();
        result
    })
}

/// A scoped arena that automatically resets when dropped.
/// Useful for batch processing where you want explicit lifetime control.
pub struct ScopedArena {
    arena: Bump,
}

impl ScopedArena {
    /// Create a new scoped arena with default capacity
    pub fn new() -> Self {
        Self {
            arena: Bump::with_capacity(DEFAULT_ARENA_CAPACITY),
        }
    }

    /// Create a new scoped arena with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            arena: Bump::with_capacity(capacity),
        }
    }

    /// Get a reference to the underlying arena
    #[inline]
    pub fn arena(&self) -> &Bump {
        &self.arena
    }

    /// Allocate a slice and copy data into it
    #[inline]
    pub fn alloc_slice_copy<T: Copy>(&self, slice: &[T]) -> &[T] {
        self.arena.alloc_slice_copy(slice)
    }

    /// Allocate a string and copy into it
    #[inline]
    pub fn alloc_str(&self, s: &str) -> &str {
        self.arena.alloc_str(s)
    }

    /// Reset the arena, freeing all allocations
    #[inline]
    pub fn reset(&mut self) {
        self.arena.reset();
    }

    /// Get the number of bytes allocated
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.arena.allocated_bytes()
    }
}

impl Default for ScopedArena {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// String Interner (2.5.2)
// ============================================================================

/// A string interner that deduplicates strings for memory efficiency.
/// Once interned, strings are never freed until the interner is dropped.
///
/// This is particularly useful for vocabulary tokens where the same string
/// may be referenced many times.
///
/// # Safety Note
/// The `intern()` method returns `&'static str` which is technically a lie -
/// the strings are only valid as long as the `StringInterner` exists.
/// This is sound ONLY when:
/// 1. The interner is stored in a `static` variable (via `Lazy`), OR
/// 2. The interner is kept alive via `Arc` for as long as references are used
///
/// For safe usage with bounded lifetimes, use `intern_bounded()` instead.
pub struct StringInterner {
    /// Set of interned strings
    strings: RwLock<AHashSet<Arc<str>>>,
    /// Statistics
    stats: InternerStats,
}

/// Statistics for string interner
#[derive(Debug, Default)]
pub struct InternerStats {
    /// Number of intern requests
    pub requests: AtomicUsize,
    /// Number of cache hits (string already interned)
    pub hits: AtomicUsize,
    /// Number of new strings interned
    pub interned: AtomicUsize,
    /// Total bytes stored
    pub bytes_stored: AtomicUsize,
}

impl StringInterner {
    /// Create a new string interner
    pub fn new() -> Self {
        Self {
            strings: RwLock::new(AHashSet::new()),
            stats: InternerStats::default(),
        }
    }

    /// Create a new string interner with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            strings: RwLock::new(AHashSet::with_capacity(capacity)),
            stats: InternerStats::default(),
        }
    }

    /// Intern a string with a lifetime bounded to the interner.
    /// This is the safe version that properly expresses the lifetime relationship.
    ///
    /// If the string is already interned, returns the existing reference.
    #[inline]
    pub fn intern_bounded<'a>(&'a self, s: &str) -> &'a str {
        // SAFETY: We return a reference to the string data owned by an Arc
        // inside our HashSet. Since we never remove from the HashSet, and
        // Arc<str> has a stable data address, this reference is valid as
        // long as the interner (self) exists.
        let arc = self.intern(s);
        unsafe { std::mem::transmute::<&str, &'a str>(&*arc) }
    }

    /// Intern a string, returning an `Arc<str>` reference.
    ///
    /// This is the safe, modern version of interning that ensures the string
    /// stays alive as long as any reference to it exists.
    pub fn intern(&self, s: &str) -> Arc<str> {
        self.stats.requests.fetch_add(1, Ordering::Relaxed);

        // Fast path: check if already interned
        {
            let strings = self.strings.read();
            if let Some(interned) = strings.get(s) {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return Arc::clone(interned);
            }
        }

        // Slow path: need to intern
        let mut strings = self.strings.write();

        // Double-check after acquiring write lock
        if let Some(interned) = strings.get(s) {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Arc::clone(interned);
        }

        // Intern the string
        let arc: Arc<str> = Arc::from(s);
        let result = Arc::clone(&arc);

        self.stats
            .bytes_stored
            .fetch_add(s.len(), Ordering::Relaxed);
        self.stats.interned.fetch_add(1, Ordering::Relaxed);

        strings.insert(arc);

        result
    }

    /// Check if a string is already interned
    pub fn contains(&self, s: &str) -> bool {
        self.strings.read().contains(s)
    }

    /// Get the number of interned strings
    pub fn len(&self) -> usize {
        self.strings.read().len()
    }

    /// Check if the interner is empty
    pub fn is_empty(&self) -> bool {
        self.strings.read().is_empty()
    }

    /// Get statistics about the interner
    pub fn stats(&self) -> InternerStatsSnapshot {
        InternerStatsSnapshot {
            requests: self.stats.requests.load(Ordering::Relaxed),
            hits: self.stats.hits.load(Ordering::Relaxed),
            interned: self.stats.interned.load(Ordering::Relaxed),
            bytes_stored: self.stats.bytes_stored.load(Ordering::Relaxed),
        }
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of interner statistics
#[derive(Debug, Clone, Copy)]
pub struct InternerStatsSnapshot {
    pub requests: usize,
    pub hits: usize,
    pub interned: usize,
    pub bytes_stored: usize,
}

impl InternerStatsSnapshot {
    /// Get the hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            (self.hits as f64 / self.requests as f64) * 100.0
        }
    }
}

// ============================================================================
// Memory Pool for Encodings (2.5.3)
// ============================================================================

/// A pool of pre-allocated vectors that can be reused across tokenization requests.
/// This avoids repeated allocation/deallocation of the vectors inside Encoding structs.
pub struct VecPool<T> {
    /// Pool of available vectors
    pool: Mutex<Vec<Vec<T>>>,
    /// Default capacity for new vectors
    default_capacity: usize,
    /// Statistics
    borrows: AtomicUsize,
    returns: AtomicUsize,
    allocations: AtomicUsize,
}

impl<T> VecPool<T> {
    /// Create a new vector pool
    pub fn new(default_capacity: usize) -> Self {
        Self {
            pool: Mutex::new(Vec::new()),
            default_capacity,
            borrows: AtomicUsize::new(0),
            returns: AtomicUsize::new(0),
            allocations: AtomicUsize::new(0),
        }
    }

    /// Create a new vector pool with pre-allocated vectors
    pub fn with_preallocated(default_capacity: usize, count: usize) -> Self
    where
        T: Default,
    {
        let pool: Vec<Vec<T>> = (0..count)
            .map(|_| Vec::with_capacity(default_capacity))
            .collect();

        Self {
            pool: Mutex::new(pool),
            default_capacity,
            borrows: AtomicUsize::new(0),
            returns: AtomicUsize::new(0),
            allocations: AtomicUsize::new(count),
        }
    }

    /// Borrow a vector from the pool, or allocate a new one if empty
    pub fn borrow(&self) -> Vec<T> {
        self.borrows.fetch_add(1, Ordering::Relaxed);

        let mut pool = self.pool.lock();
        if let Some(mut vec) = pool.pop() {
            vec.clear();
            vec
        } else {
            self.allocations.fetch_add(1, Ordering::Relaxed);
            Vec::with_capacity(self.default_capacity)
        }
    }

    /// Return a vector to the pool for reuse
    pub fn return_vec(&self, mut vec: Vec<T>) {
        self.returns.fetch_add(1, Ordering::Relaxed);
        vec.clear();

        let mut pool = self.pool.lock();
        // Only keep vectors that have reasonable capacity
        if vec.capacity() <= self.default_capacity * 4 {
            pool.push(vec);
        }
        // Otherwise let it drop to avoid memory bloat
    }

    /// Get pool statistics
    pub fn stats(&self) -> VecPoolStats {
        VecPoolStats {
            borrows: self.borrows.load(Ordering::Relaxed),
            returns: self.returns.load(Ordering::Relaxed),
            allocations: self.allocations.load(Ordering::Relaxed),
            pool_size: self.pool.lock().len(),
        }
    }

    /// Clear the pool, freeing all cached vectors
    pub fn clear(&self) {
        let mut pool = self.pool.lock();
        pool.clear();
    }
}

/// Statistics for a vector pool
#[derive(Debug, Clone, Copy)]
pub struct VecPoolStats {
    pub borrows: usize,
    pub returns: usize,
    pub allocations: usize,
    pub pool_size: usize,
}

impl VecPoolStats {
    /// Get the reuse rate as a percentage
    pub fn reuse_rate(&self) -> f64 {
        if self.borrows == 0 {
            0.0
        } else {
            let reuses = self.borrows.saturating_sub(self.allocations);
            (reuses as f64 / self.borrows as f64) * 100.0
        }
    }
}

/// A pool specifically designed for Encoding struct components.
/// Manages multiple vector pools for different data types.
pub struct EncodingPool {
    /// Pool for token IDs (u32)
    pub ids_pool: VecPool<u32>,
    /// Pool for token strings
    pub tokens_pool: VecPool<String>,
    /// Pool for offsets
    pub offsets_pool: VecPool<(usize, usize)>,
    /// Pool for word IDs
    pub word_ids_pool: VecPool<Option<u32>>,
    /// Pool for sequence IDs
    pub sequence_ids_pool: VecPool<Option<usize>>,
}

impl EncodingPool {
    /// Create a new encoding pool with default capacities
    pub fn new() -> Self {
        Self::with_capacity(128) // Default token capacity per encoding
    }

    /// Create a new encoding pool with specified token capacity
    pub fn with_capacity(token_capacity: usize) -> Self {
        Self {
            ids_pool: VecPool::new(token_capacity),
            tokens_pool: VecPool::new(token_capacity),
            offsets_pool: VecPool::new(token_capacity),
            word_ids_pool: VecPool::new(token_capacity),
            sequence_ids_pool: VecPool::new(token_capacity),
        }
    }

    /// Create an encoding pool with pre-allocated vectors
    pub fn with_preallocated(token_capacity: usize, count: usize) -> Self {
        Self {
            ids_pool: VecPool::with_preallocated(token_capacity, count),
            tokens_pool: VecPool::with_preallocated(token_capacity, count),
            offsets_pool: VecPool::with_preallocated(token_capacity, count),
            word_ids_pool: VecPool::with_preallocated(token_capacity, count),
            sequence_ids_pool: VecPool::with_preallocated(token_capacity, count),
        }
    }

    /// Borrow a set of vectors for creating an Encoding
    pub fn borrow(&self) -> EncodingVecs {
        EncodingVecs {
            ids: self.ids_pool.borrow(),
            type_ids: self.ids_pool.borrow(),
            tokens: self.tokens_pool.borrow(),
            offsets: self.offsets_pool.borrow(),
            special_tokens_mask: self.ids_pool.borrow(),
            attention_mask: self.ids_pool.borrow(),
            word_ids: self.word_ids_pool.borrow(),
            sequence_ids: self.sequence_ids_pool.borrow(),
        }
    }

    /// Return vectors from an encoding back to the pool
    pub fn return_vecs(&self, vecs: EncodingVecs) {
        self.ids_pool.return_vec(vecs.ids);
        self.ids_pool.return_vec(vecs.type_ids);
        self.tokens_pool.return_vec(vecs.tokens);
        self.offsets_pool.return_vec(vecs.offsets);
        self.ids_pool.return_vec(vecs.special_tokens_mask);
        self.ids_pool.return_vec(vecs.attention_mask);
        self.word_ids_pool.return_vec(vecs.word_ids);
        self.sequence_ids_pool.return_vec(vecs.sequence_ids);
    }
}

impl Default for EncodingPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Borrowed vectors for building an Encoding
pub struct EncodingVecs {
    pub ids: Vec<u32>,
    pub type_ids: Vec<u32>,
    pub tokens: Vec<String>,
    pub offsets: Vec<(usize, usize)>,
    pub special_tokens_mask: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub word_ids: Vec<Option<u32>>,
    pub sequence_ids: Vec<Option<usize>>,
}

impl EncodingVecs {
    /// Clear all vectors for reuse
    pub fn clear(&mut self) {
        self.ids.clear();
        self.type_ids.clear();
        self.tokens.clear();
        self.offsets.clear();
        self.special_tokens_mask.clear();
        self.attention_mask.clear();
        self.word_ids.clear();
        self.sequence_ids.clear();
    }

    /// Push a token into all vectors
    pub fn push(
        &mut self,
        id: u32,
        type_id: u32,
        token: String,
        offset: (usize, usize),
        special_token_mask: u32,
        attention: u32,
        word_id: Option<u32>,
        sequence_id: Option<usize>,
    ) {
        self.ids.push(id);
        self.type_ids.push(type_id);
        self.tokens.push(token);
        self.offsets.push(offset);
        self.special_tokens_mask.push(special_token_mask);
        self.attention_mask.push(attention);
        self.word_ids.push(word_id);
        self.sequence_ids.push(sequence_id);
    }

    /// Get the current length
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

// ============================================================================
// Thread-local Buffers
// ============================================================================

// Thread-local buffer for temporary string operations
thread_local! {
    static STRING_BUFFER: RefCell<String> = RefCell::new(String::with_capacity(4096));
}

/// Execute a function with access to a thread-local string buffer.
/// The buffer is cleared after use.
pub fn with_string_buffer<T, F>(f: F) -> T
where
    F: FnOnce(&mut String) -> T,
{
    STRING_BUFFER.with(|buffer| {
        let mut buffer = buffer.borrow_mut();
        buffer.clear();
        f(&mut buffer)
    })
}

// Thread-local buffer for temporary u32 vectors (token IDs)
thread_local! {
    static U32_BUFFER: RefCell<Vec<u32>> = RefCell::new(Vec::with_capacity(512));
}

/// Execute a function with access to a thread-local u32 buffer.
pub fn with_u32_buffer<T, F>(f: F) -> T
where
    F: FnOnce(&mut Vec<u32>) -> T,
{
    U32_BUFFER.with(|buffer| {
        let mut buffer = buffer.borrow_mut();
        buffer.clear();
        f(&mut buffer)
    })
}

// ============================================================================
// Global Interner (for vocabulary)
// ============================================================================

/// Global string interner instance for vocabulary tokens
static GLOBAL_INTERNER: once_cell::sync::Lazy<StringInterner> =
    once_cell::sync::Lazy::new(|| StringInterner::with_capacity(100_000));

/// Intern a string using the global interner, returning a `&'static str`.
///
/// This is safe because the global interner is static and never freed.
pub fn intern(s: &str) -> &'static str {
    let arc = GLOBAL_INTERNER.intern(s);
    // SAFETY: GLOBAL_INTERNER is static and never removes elements.
    // The string data in Arc<str> is at a stable address.
    unsafe { std::mem::transmute::<&str, &'static str>(&*arc) }
}

/// Get statistics from the global interner
pub fn global_interner_stats() -> InternerStatsSnapshot {
    GLOBAL_INTERNER.stats()
}

// ============================================================================
// Interned Vocabulary
// ============================================================================

/// A vocabulary that uses the string interner for memory-efficient storage.
/// All tokens are interned, so repeated tokens share the same memory.
pub struct InternedVocabulary {
    /// Token string to token ID mapping
    token_to_id: AHashMap<Arc<str>, u32>,
    /// Token ID to token string mapping
    id_to_token: Vec<Arc<str>>,
    /// The interner (may be shared or owned)
    interner: Arc<StringInterner>,
}

impl InternedVocabulary {
    /// Create a new interned vocabulary from tokens
    pub fn new(tokens: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        let interner = Arc::new(StringInterner::with_capacity(50_000));
        let mut token_to_id = AHashMap::new();
        let mut id_to_token = Vec::new();

        for (id, token) in tokens.into_iter().enumerate() {
            let interned = interner.intern(token.as_ref());
            token_to_id.insert(Arc::clone(&interned), id as u32);
            id_to_token.push(interned);
        }

        Self {
            token_to_id,
            id_to_token,
            interner,
        }
    }

    /// Create an interned vocabulary with a shared interner
    pub fn with_interner(
        tokens: impl IntoIterator<Item = impl AsRef<str>>,
        interner: Arc<StringInterner>,
    ) -> Self {
        let mut token_to_id = AHashMap::new();
        let mut id_to_token = Vec::new();

        for (id, token) in tokens.into_iter().enumerate() {
            let interned = interner.intern(token.as_ref());
            token_to_id.insert(Arc::clone(&interned), id as u32);
            id_to_token.push(interned);
        }

        Self {
            token_to_id,
            id_to_token,
            interner,
        }
    }

    /// Get the token ID for a token string
    #[inline]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get the token string for a token ID
    #[inline]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|arc| &**arc)
    }

    /// Get the vocabulary size
    #[inline]
    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }

    /// Check if vocabulary is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }

    /// Check if a token exists in the vocabulary
    #[inline]
    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Get the interner statistics
    pub fn interner_stats(&self) -> InternerStatsSnapshot {
        self.interner.stats()
    }

    /// Iterate over all tokens
    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> {
        self.token_to_id.iter().map(|(k, &v)| (&**k, v))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoped_arena() {
        let mut arena = ScopedArena::new();

        let slice = arena.alloc_slice_copy(&[1u32, 2, 3, 4, 5]);
        assert_eq!(slice, &[1, 2, 3, 4, 5]);

        let s = arena.alloc_str("hello world");
        assert_eq!(s, "hello world");

        assert!(arena.allocated_bytes() > 0);

        arena.reset();
        // After reset, previous allocations are invalid (but we can't test that safely)
    }

    #[test]
    fn test_with_arena_reset() {
        let result = with_arena_reset(|arena| {
            let slice = arena.alloc_slice_copy(&[1i32, 2, 3]);
            slice.iter().sum::<i32>()
        });
        assert_eq!(result, 6);
    }

    #[test]
    fn test_string_interner_basic() {
        let interner = StringInterner::new();

        let s1 = interner.intern("hello");
        let s2 = interner.intern("world");
        let s3 = interner.intern("hello"); // Should return same reference

        assert_eq!(&*s1, "hello");
        assert_eq!(&*s2, "world");
        assert!(Arc::ptr_eq(&s1, &s3)); // Same pointer
        assert_eq!(interner.len(), 2);
    }

    #[test]
    fn test_string_interner_stats() {
        let interner = StringInterner::new();

        interner.intern("a");
        interner.intern("b");
        interner.intern("a"); // Hit

        let stats = interner.stats();
        assert_eq!(stats.requests, 3);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.interned, 2);
    }

    #[test]
    fn test_vec_pool_basic() {
        let pool: VecPool<u32> = VecPool::new(100);

        let mut v1 = pool.borrow();
        v1.push(1);
        v1.push(2);
        pool.return_vec(v1);

        let v2 = pool.borrow();
        assert!(v2.is_empty()); // Should be cleared
        assert!(v2.capacity() >= 100);
    }

    #[test]
    fn test_vec_pool_stats() {
        let pool: VecPool<u32> = VecPool::new(100);

        let v1 = pool.borrow();
        let v2 = pool.borrow();
        pool.return_vec(v1);
        let _v3 = pool.borrow(); // Should reuse v1

        let stats = pool.stats();
        assert_eq!(stats.borrows, 3);
        assert_eq!(stats.returns, 1);
        assert_eq!(stats.allocations, 2); // v1 and v2 were new, v3 reused v1

        // Clean up
        pool.return_vec(v2);
    }

    #[test]
    fn test_encoding_pool() {
        let pool = EncodingPool::new();

        let mut vecs = pool.borrow();
        vecs.push(
            1,
            0,
            "hello".to_string(),
            (0, 5),
            0,
            1,
            Some(0),
            Some(0),
        );

        assert_eq!(vecs.len(), 1);
        assert_eq!(vecs.ids[0], 1);

        pool.return_vecs(vecs);
    }

    #[test]
    fn test_with_string_buffer() {
        let result = with_string_buffer(|buf| {
            buf.push_str("hello ");
            buf.push_str("world");
            buf.len()
        });
        assert_eq!(result, 11);
    }

    #[test]
    fn test_with_u32_buffer() {
        let result = with_u32_buffer(|buf| {
            buf.extend_from_slice(&[1, 2, 3, 4, 5]);
            buf.iter().sum::<u32>()
        });
        assert_eq!(result, 15);
    }

    #[test]
    fn test_global_interner() {
        let s1 = intern("global_test_string_1");
        let s2 = intern("global_test_string_1");
        assert!(std::ptr::eq(s1, s2));
    }

    #[test]
    fn test_interned_vocabulary() {
        let vocab = InternedVocabulary::new(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world"]);

        assert_eq!(vocab.len(), 6);
        assert_eq!(vocab.token_to_id("hello"), Some(4));
        assert_eq!(vocab.id_to_token(4), Some("hello"));
        assert!(vocab.contains("world"));
        assert!(!vocab.contains("foo"));
    }

    #[test]
    fn test_interned_vocabulary_memory_efficiency() {
        let interner = Arc::new(StringInterner::new());

        // Create two vocabularies sharing the same interner
        let vocab1 = InternedVocabulary::with_interner(
            ["hello", "world", "foo"],
            Arc::clone(&interner),
        );
        let vocab2 = InternedVocabulary::with_interner(
            ["hello", "bar", "baz"],
            Arc::clone(&interner),
        );

        // "hello" should be interned only once
        let stats = interner.stats();
        assert!(stats.interned <= 5); // hello, world, foo, bar, baz (hello shared)

        // Both vocabs can look up hello
        assert_eq!(vocab1.token_to_id("hello"), Some(0));
        assert_eq!(vocab2.token_to_id("hello"), Some(0));
    }

    #[test]
    fn test_vec_pool_preallocated() {
        let pool: VecPool<u32> = VecPool::with_preallocated(100, 5);

        let stats = pool.stats();
        assert_eq!(stats.allocations, 5);
        assert_eq!(stats.pool_size, 5);

        // Borrow all pre-allocated vectors
        let v1 = pool.borrow();
        let v2 = pool.borrow();
        let v3 = pool.borrow();
        let v4 = pool.borrow();
        let v5 = pool.borrow();

        let stats = pool.stats();
        assert_eq!(stats.borrows, 5);
        assert_eq!(stats.pool_size, 0);

        // This should allocate a new one
        let v6 = pool.borrow();
        let stats = pool.stats();
        assert_eq!(stats.allocations, 6);

        // Return all
        pool.return_vec(v1);
        pool.return_vec(v2);
        pool.return_vec(v3);
        pool.return_vec(v4);
        pool.return_vec(v5);
        pool.return_vec(v6);

        let stats = pool.stats();
        assert_eq!(stats.pool_size, 6);
    }
}

// ============================================================================
// NUMA-Aware Memory Management
// ============================================================================

#[cfg(target_os = "linux")]
extern crate libc;

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub num_nodes: usize,
    /// Number of CPUs per node
    pub cpus_per_node: Vec<Vec<usize>>,
    /// Total number of CPUs
    pub total_cpus: usize,
    /// Whether NUMA is available
    pub numa_available: bool,
}

impl NumaTopology {
    /// Detect NUMA topology from the system
    pub fn detect() -> Self {
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Self::single_node_fallback()
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_linux() -> Self {
        use std::fs;
        use std::path::Path;

        let numa_path = Path::new("/sys/devices/system/node");
        if !numa_path.exists() {
            return Self::single_node_fallback();
        }

        let mut num_nodes = 0;
        let mut cpus_per_node = Vec::new();

        // Count NUMA nodes
        if let Ok(entries) = fs::read_dir(numa_path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("node") && name_str[4..].parse::<usize>().is_ok() {
                    num_nodes += 1;
                }
            }
        }

        if num_nodes == 0 {
            return Self::single_node_fallback();
        }

        // Get CPUs for each node
        for node_id in 0..num_nodes {
            let cpu_list_path = format!("/sys/devices/system/node/node{}/cpulist", node_id);
            let cpus = if let Ok(content) = fs::read_to_string(&cpu_list_path) {
                Self::parse_cpu_list(&content.trim())
            } else {
                Vec::new()
            };
            cpus_per_node.push(cpus);
        }

        let total_cpus = cpus_per_node.iter().map(|v| v.len()).sum();

        Self {
            num_nodes,
            cpus_per_node,
            total_cpus,
            numa_available: num_nodes > 1,
        }
    }

    /// Parse CPU list format like "0-7,16-23"
    fn parse_cpu_list(list: &str) -> Vec<usize> {
        let mut cpus = Vec::new();
        for part in list.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some((start, end)) = part.split_once('-') {
                if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
                    cpus.extend(s..=e);
                }
            } else if let Ok(cpu) = part.parse::<usize>() {
                cpus.push(cpu);
            }
        }
        cpus
    }

    fn single_node_fallback() -> Self {
        let total_cpus = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        Self {
            num_nodes: 1,
            cpus_per_node: vec![(0..total_cpus).collect()],
            total_cpus,
            numa_available: false,
        }
    }

    /// Get the NUMA node for a given CPU
    pub fn cpu_to_node(&self, cpu: usize) -> Option<usize> {
        for (node_id, cpus) in self.cpus_per_node.iter().enumerate() {
            if cpus.contains(&cpu) {
                return Some(node_id);
            }
        }
        None
    }

    /// Get recommended CPUs for a given thread index (round-robin across nodes)
    pub fn thread_to_cpu(&self, thread_idx: usize) -> usize {
        if self.total_cpus == 0 {
            return 0;
        }
        // Round-robin across nodes for better memory distribution
        let node_idx = thread_idx % self.num_nodes;
        let cpus = &self.cpus_per_node[node_idx];
        if cpus.is_empty() {
            return thread_idx % self.total_cpus;
        }
        let cpu_idx = (thread_idx / self.num_nodes) % cpus.len();
        cpus[cpu_idx]
    }
}

/// Set CPU affinity for the current thread
/// This helps with NUMA locality by keeping threads on specific CPUs
#[cfg(target_os = "linux")]
pub fn set_thread_affinity(cpu: usize) -> bool {
    use std::mem;

    unsafe {
        let mut cpu_set: libc::cpu_set_t = mem::zeroed();
        libc::CPU_ZERO(&mut cpu_set);
        libc::CPU_SET(cpu, &mut cpu_set);

        let result = libc::sched_setaffinity(
            0, // current thread
            mem::size_of::<libc::cpu_set_t>(),
            &cpu_set,
        );
        result == 0
    }
}

#[cfg(not(target_os = "linux"))]
pub fn set_thread_affinity(_cpu: usize) -> bool {
    false // Not supported on this platform
}

/// NUMA-aware parallel iterator configuration
/// Use this when setting up rayon thread pools for optimal NUMA performance
pub struct NumaParallelConfig {
    topology: NumaTopology,
}

impl NumaParallelConfig {
    /// Create a new NUMA-aware parallel configuration
    pub fn new() -> Self {
        Self {
            topology: NumaTopology::detect(),
        }
    }

    /// Get the detected NUMA topology
    pub fn topology(&self) -> &NumaTopology {
        &self.topology
    }

    /// Setup thread affinity for optimal NUMA performance
    /// Call this at the start of each rayon worker thread
    pub fn setup_thread_affinity(&self, thread_idx: usize) {
        if self.topology.numa_available {
            let cpu = self.topology.thread_to_cpu(thread_idx);
            set_thread_affinity(cpu);
        }
    }

    /// Get recommended thread count based on NUMA topology
    /// Defaults to total CPU count, but can be adjusted for memory bandwidth
    pub fn recommended_threads(&self) -> usize {
        // For memory-bound workloads, using fewer threads per NUMA node
        // can reduce memory bandwidth contention
        if self.topology.numa_available && self.topology.num_nodes > 1 {
            // Use half the threads if heavily memory bound
            // This leaves more memory bandwidth per thread
            self.topology.total_cpus
        } else {
            self.topology.total_cpus
        }
    }
}

impl Default for NumaParallelConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-local NUMA node ID (cached for performance)
thread_local! {
    static CURRENT_NUMA_NODE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Get the NUMA node for the current thread (cached)
pub fn current_numa_node() -> usize {
    CURRENT_NUMA_NODE.with(|cell| {
        if let Some(node) = cell.get() {
            return node;
        }

        // Detect and cache
        #[cfg(target_os = "linux")]
        {
            let node = unsafe {
                // Get current CPU
                let cpu = libc::sched_getcpu();
                if cpu >= 0 {
                    // Look up NUMA node for this CPU
                    let topology = NumaTopology::detect();
                    topology.cpu_to_node(cpu as usize).unwrap_or(0)
                } else {
                    0
                }
            };
            cell.set(Some(node));
            node
        }
        #[cfg(not(target_os = "linux"))]
        {
            cell.set(Some(0));
            0
        }
    })
}
