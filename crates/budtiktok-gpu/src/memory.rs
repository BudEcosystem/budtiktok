//! GPU memory management
//!
//! Provides efficient memory allocation and pooling for GPU operations:
//! - Device memory buffers
//! - Pinned host memory for fast transfers
//! - Memory pools for reuse

use crate::backend::GpuError;

#[cfg(feature = "cuda")]
use crate::cuda::CudaContext;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, ValidAsZeroBits};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// GPU memory pool for efficient allocation reuse
#[derive(Debug)]
pub struct GpuMemoryPool {
    total_bytes: usize,
    allocated_bytes: usize,
    #[cfg(feature = "cuda")]
    free_buffers: Vec<(usize, Vec<u8>)>, // (size, buffer) pairs
}

impl GpuMemoryPool {
    /// Create a new memory pool with specified maximum size
    pub fn new(size_bytes: usize) -> Self {
        Self {
            total_bytes: size_bytes,
            allocated_bytes: 0,
            #[cfg(feature = "cuda")]
            free_buffers: Vec::new(),
        }
    }

    /// Get available memory
    pub fn available(&self) -> usize {
        self.total_bytes.saturating_sub(self.allocated_bytes)
    }

    /// Get total memory capacity
    pub fn total(&self) -> usize {
        self.total_bytes
    }

    /// Get currently allocated memory
    pub fn allocated(&self) -> usize {
        self.allocated_bytes
    }

    /// Reset the pool (mark all memory as available)
    pub fn reset(&mut self) {
        self.allocated_bytes = 0;
        #[cfg(feature = "cuda")]
        {
            self.free_buffers.clear();
        }
    }
}

/// GPU buffer wrapper for device memory
#[cfg(feature = "cuda")]
pub struct GpuBuffer<T: DeviceRepr> {
    device: Arc<CudaDevice>,
    slice: CudaSlice<T>,
    len: usize,
}

#[cfg(feature = "cuda")]
impl<T: DeviceRepr + ValidAsZeroBits + Clone + Default + Unpin> GpuBuffer<T> {
    /// Allocate a new buffer on the GPU
    pub fn new(ctx: &CudaContext, len: usize) -> Result<Self, GpuError> {
        let device = ctx.device().clone();
        let slice = ctx.alloc::<T>(len)?;
        Ok(Self { device, slice, len })
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Copy data from host to device
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<(), GpuError> {
        if data.len() != self.len {
            return Err(GpuError::MemoryAllocation(format!(
                "Size mismatch: expected {}, got {}",
                self.len,
                data.len()
            )));
        }
        self.device
            .htod_sync_copy_into(data, &mut self.slice)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))
    }

    /// Copy data from device to host
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<(), GpuError> {
        if data.len() != self.len {
            return Err(GpuError::MemoryAllocation(format!(
                "Size mismatch: expected {}, got {}",
                self.len,
                data.len()
            )));
        }
        self.device
            .dtoh_sync_copy_into(&self.slice, data)
            .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))
    }

    /// Get the underlying slice
    pub fn slice(&self) -> &CudaSlice<T> {
        &self.slice
    }

    /// Get mutable underlying slice
    pub fn slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }
}

/// Pinned (page-locked) host memory for fast GPU transfers
#[cfg(feature = "cuda")]
pub struct PinnedBuffer<T> {
    data: Vec<T>,
    len: usize,
}

#[cfg(feature = "cuda")]
impl<T: Clone + Default> PinnedBuffer<T> {
    /// Allocate pinned memory
    pub fn new(_ctx: &CudaContext, len: usize) -> Result<Self, GpuError> {
        // Note: cudarc doesn't have direct pinned memory API
        // In production, use cudaHostAlloc via raw CUDA
        // For now, use regular memory (still works, just slower)
        let data = vec![T::default(); len];
        Ok(Self { data, len })
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get slice of the data
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get mutable slice of the data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

/// Double buffer for async pipeline
#[cfg(feature = "cuda")]
pub struct DoubleBuffer<T: DeviceRepr + ValidAsZeroBits + Clone + Default + Unpin> {
    buffer_a: GpuBuffer<T>,
    buffer_b: GpuBuffer<T>,
    active: bool, // false = A, true = B
}

#[cfg(feature = "cuda")]
impl<T: DeviceRepr + ValidAsZeroBits + Clone + Default + Unpin> DoubleBuffer<T> {
    /// Create a new double buffer
    pub fn new(ctx: &CudaContext, len: usize) -> Result<Self, GpuError> {
        let buffer_a = GpuBuffer::new(ctx, len)?;
        let buffer_b = GpuBuffer::new(ctx, len)?;
        Ok(Self {
            buffer_a,
            buffer_b,
            active: false,
        })
    }

    /// Get the current active buffer
    pub fn active_buffer(&self) -> &GpuBuffer<T> {
        if self.active {
            &self.buffer_b
        } else {
            &self.buffer_a
        }
    }

    /// Get the current inactive buffer (for staging)
    pub fn staging_buffer(&self) -> &GpuBuffer<T> {
        if self.active {
            &self.buffer_a
        } else {
            &self.buffer_b
        }
    }

    /// Get mutable staging buffer
    pub fn staging_buffer_mut(&mut self) -> &mut GpuBuffer<T> {
        if self.active {
            &mut self.buffer_a
        } else {
            &mut self.buffer_b
        }
    }

    /// Swap active and staging buffers
    pub fn swap(&mut self) {
        self.active = !self.active;
    }
}

/// Memory arena for batch allocations
#[derive(Debug)]
pub struct GpuArena {
    capacity: usize,
    offset: usize,
}

impl GpuArena {
    /// Create a new arena with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self { capacity, offset: 0 }
    }

    /// Allocate from the arena
    pub fn alloc(&mut self, size: usize, align: usize) -> Option<usize> {
        // Align offset
        let aligned_offset = (self.offset + align - 1) & !(align - 1);

        if aligned_offset + size > self.capacity {
            return None;
        }

        let result = aligned_offset;
        self.offset = aligned_offset + size;
        Some(result)
    }

    /// Reset the arena
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Get remaining capacity
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.offset)
    }
}

/// Batch allocator for tokenization
#[cfg(feature = "cuda")]
pub struct TokenizationBuffers {
    /// Input text bytes
    pub input_bytes: GpuBuffer<u8>,
    /// Word offsets
    pub word_offsets: GpuBuffer<u32>,
    /// Word lengths
    pub word_lengths: GpuBuffer<u32>,
    /// Output token IDs
    pub output_ids: GpuBuffer<u32>,
    /// Attention mask
    pub attention_mask: GpuBuffer<u32>,
    /// Token count per sequence
    pub token_counts: GpuBuffer<u32>,

    max_batch_size: usize,
    max_seq_len: usize,
}

#[cfg(feature = "cuda")]
impl TokenizationBuffers {
    /// Create buffers for batch tokenization
    pub fn new(
        ctx: &CudaContext,
        max_batch_size: usize,
        max_seq_len: usize,
    ) -> Result<Self, GpuError> {
        let max_total_tokens = max_batch_size * max_seq_len;
        let max_input_bytes = max_batch_size * max_seq_len * 4; // UTF-8 worst case

        Ok(Self {
            input_bytes: GpuBuffer::new(ctx, max_input_bytes)?,
            word_offsets: GpuBuffer::new(ctx, max_total_tokens)?,
            word_lengths: GpuBuffer::new(ctx, max_total_tokens)?,
            output_ids: GpuBuffer::new(ctx, max_total_tokens)?,
            attention_mask: GpuBuffer::new(ctx, max_total_tokens)?,
            token_counts: GpuBuffer::new(ctx, max_batch_size)?,
            max_batch_size,
            max_seq_len,
        })
    }

    /// Get maximum batch size
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let pool = GpuMemoryPool::new(1024 * 1024);
        assert_eq!(pool.total(), 1024 * 1024);
        assert_eq!(pool.available(), 1024 * 1024);
        assert_eq!(pool.allocated(), 0);
    }

    #[test]
    fn test_memory_pool_reset() {
        let mut pool = GpuMemoryPool::new(1024);
        pool.reset();
        assert_eq!(pool.available(), 1024);
    }

    #[test]
    fn test_arena_allocation() {
        let mut arena = GpuArena::new(1024);

        let offset1 = arena.alloc(100, 8);
        assert!(offset1.is_some());
        assert_eq!(offset1.unwrap(), 0);

        let offset2 = arena.alloc(100, 8);
        assert!(offset2.is_some());
        assert_eq!(offset2.unwrap(), 104); // Aligned to 8

        assert_eq!(arena.remaining(), 1024 - 204);
    }

    #[test]
    fn test_arena_overflow() {
        let mut arena = GpuArena::new(100);

        let offset1 = arena.alloc(50, 1);
        assert!(offset1.is_some());

        let offset2 = arena.alloc(60, 1);
        assert!(offset2.is_none()); // Would overflow
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = GpuArena::new(1024);
        arena.alloc(500, 1);
        assert_eq!(arena.remaining(), 524);

        arena.reset();
        assert_eq!(arena.remaining(), 1024);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_buffer() {
        use crate::cuda::{is_cuda_available, CudaContext};

        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return;
        }

        let ctx = CudaContext::new(0).expect("Context failed");
        let buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1024).expect("Alloc failed");

        assert_eq!(buffer.len(), 1024);
        assert_eq!(buffer.size_bytes(), 1024 * 4);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tokenization_buffers() {
        use crate::cuda::{is_cuda_available, CudaContext};

        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return;
        }

        let ctx = CudaContext::new(0).expect("Context failed");
        let buffers = TokenizationBuffers::new(&ctx, 32, 512).expect("Buffers failed");

        assert_eq!(buffers.max_batch_size(), 32);
        assert_eq!(buffers.max_seq_len(), 512);
    }
}
