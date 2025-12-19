//! Custom Arena Allocator for Zero-Allocation Token Buffers
//!
//! Provides thread-local arena allocation to eliminate allocation overhead
//! in the tokenization hot path. Based on bump allocation with periodic resets.

use std::cell::UnsafeCell;
use std::mem::{self, MaybeUninit};
use std::ptr::NonNull;

/// Default arena size (1MB - fits in L3 cache on most CPUs)
const DEFAULT_ARENA_SIZE: usize = 1024 * 1024;

/// Token buffer pre-allocation size
const DEFAULT_TOKEN_BUFFER_CAPACITY: usize = 256;

/// Arena allocator for fast bump allocation
pub struct Arena {
    /// Start of the arena memory
    start: NonNull<u8>,
    /// Current allocation pointer
    current: UnsafeCell<*mut u8>,
    /// End of the arena memory
    end: *mut u8,
    /// Total capacity
    capacity: usize,
}

impl Arena {
    /// Create a new arena with given capacity
    pub fn new(capacity: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(capacity, 64)
            .expect("Invalid arena layout");

        let ptr = unsafe { std::alloc::alloc(layout) };
        let start = NonNull::new(ptr).expect("Arena allocation failed");

        Self {
            start,
            current: UnsafeCell::new(ptr),
            end: unsafe { ptr.add(capacity) },
            capacity,
        }
    }

    /// Allocate memory from the arena
    #[inline]
    pub fn alloc<T>(&self, count: usize) -> Option<&mut [MaybeUninit<T>]> {
        let size = mem::size_of::<T>() * count;
        let align = mem::align_of::<T>();

        unsafe {
            let current = *self.current.get();

            // Align the current pointer
            let aligned = (current as usize + align - 1) & !(align - 1);
            let new_current = aligned + size;

            if new_current > self.end as usize {
                return None;
            }

            *self.current.get() = new_current as *mut u8;

            let ptr = aligned as *mut MaybeUninit<T>;
            Some(std::slice::from_raw_parts_mut(ptr, count))
        }
    }

    /// Allocate and initialize a slice
    #[inline]
    pub fn alloc_slice<T: Copy>(&self, values: &[T]) -> Option<&mut [T]> {
        let slice = self.alloc::<T>(values.len())?;
        for (dst, src) in slice.iter_mut().zip(values.iter()) {
            dst.write(*src);
        }
        Some(unsafe {
            std::slice::from_raw_parts_mut(
                slice.as_mut_ptr() as *mut T,
                values.len()
            )
        })
    }

    /// Allocate a Vec-like buffer with given capacity
    #[inline]
    pub fn alloc_vec<T>(&self, capacity: usize) -> Option<ArenaVec<T>> {
        let slice = self.alloc::<T>(capacity)?;
        Some(ArenaVec {
            ptr: slice.as_mut_ptr() as *mut T,
            len: 0,
            capacity,
        })
    }

    /// Reset the arena (invalidates all allocations!)
    #[inline]
    pub fn reset(&self) {
        unsafe {
            *self.current.get() = self.start.as_ptr();
        }
    }

    /// Get current usage
    #[inline]
    pub fn used(&self) -> usize {
        unsafe {
            (*self.current.get() as usize) - (self.start.as_ptr() as usize)
        }
    }

    /// Get remaining capacity
    #[inline]
    pub fn remaining(&self) -> usize {
        self.capacity - self.used()
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.capacity, 64)
            .expect("Invalid arena layout");

        unsafe {
            std::alloc::dealloc(self.start.as_ptr(), layout);
        }
    }
}

// Safety: Arena is thread-local by design
unsafe impl Send for Arena {}

/// Vec-like container allocated from arena
pub struct ArenaVec<T> {
    ptr: *mut T,
    len: usize,
    capacity: usize,
}

impl<T> ArenaVec<T> {
    /// Push an element
    #[inline]
    pub fn push(&mut self, value: T) -> bool {
        if self.len >= self.capacity {
            return false;
        }
        unsafe {
            self.ptr.add(self.len).write(value);
        }
        self.len += 1;
        true
    }

    /// Extend from iterator
    #[inline]
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            if !self.push(item) {
                break;
            }
        }
    }

    /// Get length
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get as slice
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get as mutable slice
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Convert to owned Vec (copies data)
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.as_slice().to_vec()
    }

    /// Clear the vector
    #[inline]
    pub fn clear(&mut self) {
        // Note: doesn't call drop on elements
        self.len = 0;
    }
}

impl<T> std::ops::Deref for ArenaVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> std::ops::DerefMut for ArenaVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

/// Thread-local arena pool for tokenization
pub struct ArenaPool {
    /// Pool of arenas for parallel processing
    arenas: Vec<Arena>,
}

impl ArenaPool {
    /// Create a pool with given number of arenas
    pub fn new(num_arenas: usize, arena_size: usize) -> Self {
        let arenas = (0..num_arenas)
            .map(|_| Arena::new(arena_size))
            .collect();
        Self { arenas }
    }

    /// Get arena for given thread index
    #[inline]
    pub fn get(&self, thread_id: usize) -> &Arena {
        &self.arenas[thread_id % self.arenas.len()]
    }

    /// Reset all arenas
    pub fn reset_all(&self) {
        for arena in &self.arenas {
            arena.reset();
        }
    }
}

impl Default for ArenaPool {
    fn default() -> Self {
        let num_threads = rayon::current_num_threads().max(1);
        Self::new(num_threads, DEFAULT_ARENA_SIZE)
    }
}

/// Pre-allocated token buffer for batch processing
#[repr(C, align(64))]
pub struct TokenBuffer {
    /// Token IDs
    ids: Vec<u32>,
    /// Word boundaries (start, end pairs)
    word_bounds: Vec<(usize, usize)>,
    /// Token offsets within words
    token_offsets: Vec<usize>,
}

impl TokenBuffer {
    /// Create a new token buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            ids: Vec::with_capacity(capacity),
            word_bounds: Vec::with_capacity(capacity / 4),
            token_offsets: Vec::with_capacity(capacity),
        }
    }

    /// Clear the buffer for reuse
    #[inline]
    pub fn clear(&mut self) {
        self.ids.clear();
        self.word_bounds.clear();
        self.token_offsets.clear();
    }

    /// Get token IDs
    #[inline]
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Push a token ID
    #[inline]
    pub fn push_id(&mut self, id: u32) {
        self.ids.push(id);
    }

    /// Extend with token IDs
    #[inline]
    pub fn extend_ids(&mut self, ids: impl IntoIterator<Item = u32>) {
        self.ids.extend(ids);
    }

    /// Push word boundary
    #[inline]
    pub fn push_word_bound(&mut self, start: usize, end: usize) {
        self.word_bounds.push((start, end));
    }

    /// Get word boundaries
    #[inline]
    pub fn word_bounds(&self) -> &[(usize, usize)] {
        &self.word_bounds
    }

    /// Reserve capacity
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.ids.reserve(additional);
    }
}

impl Default for TokenBuffer {
    fn default() -> Self {
        Self::new(DEFAULT_TOKEN_BUFFER_CAPACITY)
    }
}

/// Pool of reusable token buffers for batch processing
pub struct TokenBufferPool {
    /// Available buffers
    buffers: parking_lot::Mutex<Vec<TokenBuffer>>,
    /// Buffer capacity
    buffer_capacity: usize,
}

impl TokenBufferPool {
    /// Create a new pool
    pub fn new(initial_buffers: usize, buffer_capacity: usize) -> Self {
        let buffers = (0..initial_buffers)
            .map(|_| TokenBuffer::new(buffer_capacity))
            .collect();
        Self {
            buffers: parking_lot::Mutex::new(buffers),
            buffer_capacity,
        }
    }

    /// Acquire a buffer from the pool
    #[inline]
    pub fn acquire(&self) -> TokenBuffer {
        let mut buffers = self.buffers.lock();
        buffers.pop().unwrap_or_else(|| TokenBuffer::new(self.buffer_capacity))
    }

    /// Return a buffer to the pool
    #[inline]
    pub fn release(&self, mut buffer: TokenBuffer) {
        buffer.clear();
        let mut buffers = self.buffers.lock();
        if buffers.len() < 256 {  // Limit pool size
            buffers.push(buffer);
        }
    }
}

impl Default for TokenBufferPool {
    fn default() -> Self {
        Self::new(8, DEFAULT_TOKEN_BUFFER_CAPACITY)
    }
}

/// RAII guard for token buffer
pub struct TokenBufferGuard<'a> {
    buffer: Option<TokenBuffer>,
    pool: &'a TokenBufferPool,
}

impl<'a> TokenBufferGuard<'a> {
    /// Create a new guard
    pub fn new(pool: &'a TokenBufferPool) -> Self {
        Self {
            buffer: Some(pool.acquire()),
            pool,
        }
    }

    /// Get mutable reference to buffer
    #[inline]
    pub fn buffer(&mut self) -> &mut TokenBuffer {
        self.buffer.as_mut().unwrap()
    }
}

impl Drop for TokenBufferGuard<'_> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer);
        }
    }
}

impl std::ops::Deref for TokenBufferGuard<'_> {
    type Target = TokenBuffer;

    fn deref(&self) -> &TokenBuffer {
        self.buffer.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for TokenBufferGuard<'_> {
    fn deref_mut(&mut self) -> &mut TokenBuffer {
        self.buffer.as_mut().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        let arena = Arena::new(1024);

        let slice1 = arena.alloc::<u32>(10).unwrap();
        assert_eq!(slice1.len(), 10);

        let slice2 = arena.alloc::<u64>(5).unwrap();
        assert_eq!(slice2.len(), 5);

        assert!(arena.used() > 0);
        assert!(arena.remaining() < 1024);
    }

    #[test]
    fn test_arena_reset() {
        let arena = Arena::new(1024);

        let _ = arena.alloc::<u32>(100);
        assert!(arena.used() > 0);

        arena.reset();
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_arena_vec() {
        let arena = Arena::new(1024);

        let mut vec = arena.alloc_vec::<u32>(10).unwrap();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        assert_eq!(vec.len(), 3);
        assert_eq!(vec.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_token_buffer() {
        let mut buffer = TokenBuffer::new(100);

        buffer.push_id(1);
        buffer.push_id(2);
        buffer.push_id(3);

        assert_eq!(buffer.ids(), &[1, 2, 3]);

        buffer.clear();
        assert!(buffer.ids().is_empty());
    }

    #[test]
    fn test_buffer_pool() {
        let pool = TokenBufferPool::new(2, 100);

        let buffer1 = pool.acquire();
        let buffer2 = pool.acquire();

        pool.release(buffer1);
        pool.release(buffer2);

        // Should reuse buffers
        let _buffer3 = pool.acquire();
        let _buffer4 = pool.acquire();
    }
}
