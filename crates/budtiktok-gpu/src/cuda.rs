//! CUDA backend implementation
//!
//! Provides GPU acceleration using NVIDIA CUDA.

use crate::backend::GpuError;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice as CudarcDevice, CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// CUDA device information
#[derive(Debug, Clone)]
pub struct CudaDevice {
    /// Device ID (0-indexed)
    pub id: i32,
    /// Device name (e.g., "NVIDIA GeForce RTX 3080")
    pub name: String,
    /// Compute capability (major, minor)
    pub compute_capability: (i32, i32),
    /// Total device memory in bytes
    pub total_memory: usize,
    /// Number of multiprocessors
    pub multiprocessor_count: i32,
    /// Maximum threads per block
    pub max_threads_per_block: i32,
    /// Warp size
    pub warp_size: i32,
}

/// CUDA execution context
#[cfg(feature = "cuda")]
pub struct CudaContext {
    device: Arc<CudarcDevice>,
    device_id: i32,
    device_info: CudaDevice,
}

#[cfg(not(feature = "cuda"))]
pub struct CudaContext {
    device_id: i32,
}

/// Check if CUDA is available on this system
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        CudarcDevice::new(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get list of available CUDA devices
pub fn get_cuda_devices() -> Vec<CudaDevice> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::{result, sys};

        let mut devices = Vec::new();

        // Initialize CUDA
        if result::init().is_err() {
            return devices;
        }

        // Get device count
        let count = match result::device::get_count() {
            Ok(c) => c,
            Err(_) => return devices,
        };

        for i in 0..count {
            // Get raw device handle
            let cu_device = match result::device::get(i) {
                Ok(d) => d,
                Err(_) => continue,
            };

            // Get device name using raw CUDA call
            let name = {
                let mut name_buf = [0i8; 256];
                unsafe {
                    if sys::cuDeviceGetName(name_buf.as_mut_ptr(), 256, cu_device)
                        .result()
                        .is_ok()
                    {
                        let c_str = std::ffi::CStr::from_ptr(name_buf.as_ptr());
                        c_str.to_string_lossy().into_owned()
                    } else {
                        format!("CUDA Device {}", i)
                    }
                }
            };

            // Get total memory
            let total_memory = unsafe { result::device::total_mem(cu_device).unwrap_or(0) };

            // Get compute capability
            let compute_major = unsafe {
                result::device::get_attribute(
                    cu_device,
                    sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                )
                .unwrap_or(7)
            };
            let compute_minor = unsafe {
                result::device::get_attribute(
                    cu_device,
                    sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                )
                .unwrap_or(5)
            };

            // Get multiprocessor count
            let multiprocessor_count = unsafe {
                result::device::get_attribute(
                    cu_device,
                    sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                )
                .unwrap_or(0)
            };

            // Get max threads per block
            let max_threads_per_block = unsafe {
                result::device::get_attribute(
                    cu_device,
                    sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                )
                .unwrap_or(1024)
            };

            // Get warp size
            let warp_size = unsafe {
                result::device::get_attribute(
                    cu_device,
                    sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                )
                .unwrap_or(32)
            };

            let device = CudaDevice {
                id: i,
                name,
                compute_capability: (compute_major, compute_minor),
                total_memory,
                multiprocessor_count,
                max_threads_per_block,
                warp_size,
            };

            devices.push(device);
        }

        devices
    }
    #[cfg(not(feature = "cuda"))]
    {
        Vec::new()
    }
}

#[cfg(feature = "cuda")]
impl CudaContext {
    /// Create a new CUDA context on the specified device
    pub fn new(device_id: i32) -> Result<Self, GpuError> {
        use cudarc::driver::{result, sys};

        let device = CudarcDevice::new(device_id as usize)
            .map_err(|e| GpuError::Cuda(format!("Failed to create device: {}", e)))?;

        // Get raw device handle for property queries
        let cu_device = *device.cu_device();

        // Get device name
        let name = {
            let mut name_buf = [0i8; 256];
            unsafe {
                if sys::cuDeviceGetName(name_buf.as_mut_ptr(), 256, cu_device)
                    .result()
                    .is_ok()
                {
                    let c_str = std::ffi::CStr::from_ptr(name_buf.as_ptr());
                    c_str.to_string_lossy().into_owned()
                } else {
                    format!("CUDA Device {}", device_id)
                }
            }
        };

        // Get total memory
        let total_memory = unsafe { result::device::total_mem(cu_device).unwrap_or(0) };

        // Get compute capability
        let compute_major = unsafe {
            result::device::get_attribute(
                cu_device,
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
            .unwrap_or(7)
        };
        let compute_minor = unsafe {
            result::device::get_attribute(
                cu_device,
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
            .unwrap_or(5)
        };

        // Get multiprocessor count
        let multiprocessor_count = unsafe {
            result::device::get_attribute(
                cu_device,
                sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .unwrap_or(0)
        };

        let device_info = CudaDevice {
            id: device_id,
            name,
            compute_capability: (compute_major, compute_minor),
            total_memory,
            multiprocessor_count,
            max_threads_per_block: 1024,
            warp_size: 32,
        };

        Ok(Self {
            device,
            device_id,
            device_info,
        })
    }

    /// Get the device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Get device information
    pub fn device_info(&self) -> &CudaDevice {
        &self.device_info
    }

    /// Get free memory on the device
    pub fn free_memory(&self) -> usize {
        // cudarc doesn't have a direct free memory query, estimate from total
        self.device_info.total_memory / 2 // Conservative estimate
    }

    /// Get the underlying cudarc device
    pub fn device(&self) -> &Arc<CudarcDevice> {
        &self.device
    }

    /// Synchronize the device (wait for all operations to complete)
    pub fn synchronize(&self) -> Result<(), GpuError> {
        self.device
            .synchronize()
            .map_err(|e| GpuError::Cuda(format!("Sync failed: {}", e)))
    }

    /// Allocate device memory
    pub fn alloc<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, GpuError> {
        self.device
            .alloc_zeros::<T>(len)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))
    }

    /// Copy data from host to device
    pub fn htod_copy<T: cudarc::driver::DeviceRepr + Clone + Unpin>(
        &self,
        data: &[T],
    ) -> Result<CudaSlice<T>, GpuError> {
        self.device
            .htod_copy(data.to_vec())
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))
    }

    /// Copy data from device to host
    pub fn dtoh_copy<T: cudarc::driver::DeviceRepr + Clone + Unpin>(
        &self,
        slice: &CudaSlice<T>,
    ) -> Result<Vec<T>, GpuError> {
        self.device
            .dtoh_sync_copy(slice)
            .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))
    }

    /// Load a PTX module
    pub fn load_ptx(
        &self,
        ptx: Ptx,
        module_name: &'static str,
        func_names: &[&'static str],
    ) -> Result<(), GpuError> {
        self.device
            .load_ptx(ptx, module_name, func_names)
            .map_err(|e| GpuError::Cuda(format!("PTX load failed: {}", e)))
    }

    /// Get a kernel function
    pub fn get_func(
        &self,
        module: &str,
        func: &str,
    ) -> Result<cudarc::driver::CudaFunction, GpuError> {
        self.device
            .get_func(module, func)
            .ok_or_else(|| GpuError::Cuda(format!("Function {} not found in {}", func, module)))
    }
}

#[cfg(not(feature = "cuda"))]
impl CudaContext {
    pub fn new(_device_id: i32) -> Result<Self, GpuError> {
        Err(GpuError::NotAvailable(
            "CUDA support not compiled in".into(),
        ))
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn free_memory(&self) -> usize {
        0
    }

    pub fn synchronize(&self) -> Result<(), GpuError> {
        Err(GpuError::NotAvailable(
            "CUDA support not compiled in".into(),
        ))
    }
}

// =============================================================================
// CUDA Kernel Source Code (PTX)
// =============================================================================

/// PTX kernel source for pre-tokenization (whitespace detection)
#[cfg(feature = "cuda")]
pub const PRETOKENIZE_KERNEL_SRC: &str = r#"
extern "C" __global__ void find_whitespace(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ output,
    unsigned int* __restrict__ count,
    unsigned int input_len
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_len) {
        unsigned char c = input[idx];
        // Check for whitespace: space, tab, newline, carriage return
        bool is_ws = (c == ' ') || (c == '\t') || (c == '\n') || (c == '\r');

        if (is_ws) {
            unsigned int pos = atomicAdd(count, 1);
            output[pos] = idx;
        }
    }
}

extern "C" __global__ void find_word_boundaries(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ starts,
    unsigned int* __restrict__ ends,
    unsigned int* __restrict__ word_count,
    unsigned int input_len
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_len) {
        unsigned char c = input[idx];
        bool is_ws = (c == ' ') || (c == '\t') || (c == '\n') || (c == '\r');

        // Check if this is a word start (non-whitespace after whitespace or at position 0)
        bool prev_is_ws = (idx == 0) ||
            (input[idx-1] == ' ') || (input[idx-1] == '\t') ||
            (input[idx-1] == '\n') || (input[idx-1] == '\r');

        if (!is_ws && prev_is_ws) {
            unsigned int word_idx = atomicAdd(word_count, 1);
            starts[word_idx] = idx;
        }

        // Check if this is a word end (whitespace after non-whitespace)
        bool next_is_ws = (idx + 1 >= input_len) ||
            (input[idx+1] == ' ') || (input[idx+1] == '\t') ||
            (input[idx+1] == '\n') || (input[idx+1] == '\r');

        if (!is_ws && next_is_ws) {
            // Find the word index by counting starts before this position
            // This is a simplified version; production would use prefix sums
        }
    }
}
"#;

/// PTX kernel source for vocabulary lookup (hash-based)
#[cfg(feature = "cuda")]
pub const VOCAB_LOOKUP_KERNEL_SRC: &str = r#"
// FNV-1a hash function
__device__ unsigned long long fnv1a_hash(const unsigned char* data, unsigned int len) {
    unsigned long long hash = 14695981039346656037ULL;
    for (unsigned int i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

extern "C" __global__ void vocab_lookup(
    const unsigned char* __restrict__ words,       // Packed word data
    const unsigned int* __restrict__ word_offsets, // Start offset of each word
    const unsigned int* __restrict__ word_lengths, // Length of each word
    const unsigned long long* __restrict__ vocab_hashes, // Vocabulary hashes
    const unsigned int* __restrict__ vocab_ids,    // Vocabulary token IDs
    unsigned int* __restrict__ output,             // Output token IDs
    unsigned int num_words,
    unsigned int vocab_size,
    unsigned int unk_id
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_words) {
        unsigned int offset = word_offsets[idx];
        unsigned int len = word_lengths[idx];

        // Compute hash
        unsigned long long hash = fnv1a_hash(&words[offset], len);

        // Linear probing in hash table
        unsigned int slot = hash % vocab_size;
        unsigned int found_id = unk_id;

        for (unsigned int probe = 0; probe < 64; probe++) {
            unsigned int probe_slot = (slot + probe) % vocab_size;
            if (vocab_hashes[probe_slot] == hash) {
                found_id = vocab_ids[probe_slot];
                break;
            }
            if (vocab_hashes[probe_slot] == 0) {
                break; // Empty slot, not found
            }
        }

        output[idx] = found_id;
    }
}
"#;

/// PTX kernel source for BPE merge operations (BlockBPE-style parallel algorithm)
/// Based on research from "BlockBPE: Parallel BPE Tokenization" (arxiv 2507.11941)
#[cfg(feature = "cuda")]
pub const BPE_MERGE_KERNEL_SRC: &str = r#"
// FNV-1a hash for merge pair lookup
__device__ unsigned long long bpe_pair_hash(unsigned int first_id, unsigned int second_id) {
    unsigned long long hash = 14695981039346656037ULL;
    // Hash the first ID (4 bytes)
    for (int i = 0; i < 4; i++) {
        hash ^= (first_id >> (i * 8)) & 0xFF;
        hash *= 1099511628211ULL;
    }
    // Hash the second ID (4 bytes)
    for (int i = 0; i < 4; i++) {
        hash ^= (second_id >> (i * 8)) & 0xFF;
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Lookup merge rule in hash table
// Returns merged_id if found, UINT_MAX if not found
__device__ unsigned int lookup_merge(
    unsigned int first_id,
    unsigned int second_id,
    const unsigned long long* __restrict__ merge_hashes,
    const unsigned int* __restrict__ merge_first_ids,
    const unsigned int* __restrict__ merge_second_ids,
    const unsigned int* __restrict__ merge_result_ids,
    const unsigned int* __restrict__ merge_priorities,
    unsigned int table_size,
    unsigned int* out_priority
) {
    unsigned long long hash = bpe_pair_hash(first_id, second_id);
    unsigned int slot = hash % table_size;

    for (unsigned int probe = 0; probe < 64; probe++) {
        unsigned int probe_slot = (slot + probe) % table_size;
        if (merge_hashes[probe_slot] == 0) {
            return 0xFFFFFFFF; // Empty slot, not found
        }
        if (merge_hashes[probe_slot] == hash &&
            merge_first_ids[probe_slot] == first_id &&
            merge_second_ids[probe_slot] == second_id) {
            *out_priority = merge_priorities[probe_slot];
            return merge_result_ids[probe_slot];
        }
    }
    return 0xFFFFFFFF; // Not found after max probes
}

// Single BPE merge pass - finds best merge across all adjacent pairs
// Each thread handles one token position
extern "C" __global__ void bpe_find_best_merge(
    const unsigned int* __restrict__ token_ids,     // Current token IDs
    const unsigned int* __restrict__ token_valid,   // 1 if token is valid, 0 if merged away
    const unsigned long long* __restrict__ merge_hashes,
    const unsigned int* __restrict__ merge_first_ids,
    const unsigned int* __restrict__ merge_second_ids,
    const unsigned int* __restrict__ merge_result_ids,
    const unsigned int* __restrict__ merge_priorities,
    unsigned int* __restrict__ best_positions,      // Output: positions with best merge per block
    unsigned int* __restrict__ best_priorities,     // Output: priorities of best merges
    unsigned int* __restrict__ best_results,        // Output: result token IDs
    unsigned int num_tokens,
    unsigned int table_size
) {
    extern __shared__ unsigned int shared_data[];
    unsigned int* s_best_pos = shared_data;
    unsigned int* s_best_pri = shared_data + blockDim.x;
    unsigned int* s_best_res = shared_data + 2 * blockDim.x;

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize with invalid values
    s_best_pos[tid] = 0xFFFFFFFF;
    s_best_pri[tid] = 0xFFFFFFFF;
    s_best_res[tid] = 0xFFFFFFFF;

    __syncthreads();

    // Each thread checks if position idx can merge with idx+1
    if (idx < num_tokens - 1 && token_valid[idx] && token_valid[idx + 1]) {
        unsigned int first_id = token_ids[idx];
        unsigned int second_id = token_ids[idx + 1];
        unsigned int priority;

        unsigned int result_id = lookup_merge(
            first_id, second_id,
            merge_hashes, merge_first_ids, merge_second_ids,
            merge_result_ids, merge_priorities,
            table_size, &priority
        );

        if (result_id != 0xFFFFFFFF) {
            s_best_pos[tid] = idx;
            s_best_pri[tid] = priority;
            s_best_res[tid] = result_id;
        }
    }

    __syncthreads();

    // Parallel reduction to find block-minimum priority
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_best_pri[tid + stride] < s_best_pri[tid]) {
                s_best_pos[tid] = s_best_pos[tid + stride];
                s_best_pri[tid] = s_best_pri[tid + stride];
                s_best_res[tid] = s_best_res[tid + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes block result
    if (tid == 0) {
        best_positions[blockIdx.x] = s_best_pos[0];
        best_priorities[blockIdx.x] = s_best_pri[0];
        best_results[blockIdx.x] = s_best_res[0];
    }
}

// Apply merge at specified positions
extern "C" __global__ void bpe_apply_merge(
    unsigned int* __restrict__ token_ids,
    unsigned int* __restrict__ token_valid,
    const unsigned int* __restrict__ merge_positions,  // Positions to merge
    const unsigned int* __restrict__ merge_results,    // Result token IDs
    unsigned int num_merges
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_merges) {
        unsigned int pos = merge_positions[idx];
        if (pos != 0xFFFFFFFF) {
            // Apply merge: update token at pos, invalidate token at pos+1
            token_ids[pos] = merge_results[idx];
            token_valid[pos + 1] = 0;
        }
    }
}

// Compact tokens after merges (remove invalidated tokens)
// Uses exclusive prefix sum pattern
extern "C" __global__ void bpe_compact_tokens(
    const unsigned int* __restrict__ token_ids,
    const unsigned int* __restrict__ token_valid,
    unsigned int* __restrict__ output_ids,
    unsigned int* __restrict__ output_count,
    unsigned int num_tokens
) {
    extern __shared__ unsigned int s_prefix[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load validity into shared memory
    s_prefix[tid] = (idx < num_tokens) ? token_valid[idx] : 0;
    __syncthreads();

    // Exclusive prefix sum (Blelloch scan)
    // Up-sweep
    for (unsigned int d = 1; d < blockDim.x; d *= 2) {
        unsigned int ai = (tid + 1) * 2 * d - 1;
        if (ai < blockDim.x) {
            s_prefix[ai] += s_prefix[ai - d];
        }
        __syncthreads();
    }

    // Set last element to 0 for exclusive scan
    if (tid == 0) {
        s_prefix[blockDim.x - 1] = 0;
    }
    __syncthreads();

    // Down-sweep
    for (unsigned int d = blockDim.x / 2; d >= 1; d /= 2) {
        unsigned int ai = (tid + 1) * 2 * d - 1;
        if (ai < blockDim.x) {
            unsigned int temp = s_prefix[ai - d];
            s_prefix[ai - d] = s_prefix[ai];
            s_prefix[ai] += temp;
        }
        __syncthreads();
    }

    // Write compacted output
    if (idx < num_tokens && token_valid[idx]) {
        unsigned int write_pos = s_prefix[tid];
        // Add block offset (from previous blocks)
        // For multi-block, would need global prefix sum
        output_ids[write_pos] = token_ids[idx];
    }

    // Last thread updates count
    if (tid == blockDim.x - 1 && blockIdx.x == 0) {
        *output_count = s_prefix[tid] + token_valid[idx];
    }
}

// Character to initial token mapping (byte-level BPE)
extern "C" __global__ void bpe_char_to_tokens(
    const unsigned char* __restrict__ input,
    const unsigned int* __restrict__ byte_to_token,  // 256-entry lookup table
    unsigned int* __restrict__ output_ids,
    unsigned int input_len
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_len) {
        output_ids[idx] = byte_to_token[input[idx]];
    }
}

// ============================================================================
// GPU-Optimized BPE: Apply ALL non-conflicting merges in parallel
// This is the key optimization - avoid CPU roundtrips
// ============================================================================

// Find all mergeable pairs and mark the BEST merge at each position
// Uses "leftmost wins" strategy for conflicts
extern "C" __global__ void bpe_find_all_merges(
    const unsigned int* __restrict__ token_ids,
    const unsigned int* __restrict__ next_indices,  // Linked list: next valid token index
    const unsigned long long* __restrict__ merge_hashes,
    const unsigned int* __restrict__ merge_first_ids,
    const unsigned int* __restrict__ merge_second_ids,
    const unsigned int* __restrict__ merge_result_ids,
    const unsigned int* __restrict__ merge_priorities,
    unsigned int* __restrict__ can_merge,           // Output: 1 if this position can merge
    unsigned int* __restrict__ merge_results,       // Output: result token if merging
    unsigned int* __restrict__ merge_pris,          // Output: priority of merge
    unsigned int num_tokens,
    unsigned int table_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_tokens) return;

    unsigned int next_idx = next_indices[idx];
    if (next_idx == 0xFFFFFFFF) {
        can_merge[idx] = 0;
        return;
    }

    unsigned int first_id = token_ids[idx];
    unsigned int second_id = token_ids[next_idx];

    // Hash lookup
    unsigned long long hash = 14695981039346656037ULL;
    for (int i = 0; i < 4; i++) {
        hash ^= (first_id >> (i * 8)) & 0xFF;
        hash *= 1099511628211ULL;
    }
    for (int i = 0; i < 4; i++) {
        hash ^= (second_id >> (i * 8)) & 0xFF;
        hash *= 1099511628211ULL;
    }

    unsigned int slot = hash % table_size;
    unsigned int found = 0;

    for (unsigned int probe = 0; probe < 64; probe++) {
        unsigned int probe_slot = (slot + probe) % table_size;
        if (merge_hashes[probe_slot] == 0) break;
        if (merge_hashes[probe_slot] == hash &&
            merge_first_ids[probe_slot] == first_id &&
            merge_second_ids[probe_slot] == second_id) {
            can_merge[idx] = 1;
            merge_results[idx] = merge_result_ids[probe_slot];
            merge_pris[idx] = merge_priorities[probe_slot];
            found = 1;
            break;
        }
    }

    if (!found) {
        can_merge[idx] = 0;
    }
}

// Resolve conflicts: for overlapping merges, keep only lowest priority (best)
// A merge at position i conflicts with merge at position next[i]
extern "C" __global__ void bpe_resolve_conflicts(
    const unsigned int* __restrict__ next_indices,
    unsigned int* __restrict__ can_merge,
    const unsigned int* __restrict__ merge_pris,
    unsigned int num_tokens
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_tokens) return;
    if (can_merge[idx] == 0) return;

    unsigned int next_idx = next_indices[idx];
    if (next_idx == 0xFFFFFFFF) return;

    // Check if next position also wants to merge
    if (can_merge[next_idx] == 1) {
        // Conflict! Lower priority wins (ties: left wins)
        if (merge_pris[idx] > merge_pris[next_idx]) {
            can_merge[idx] = 0;  // We lose
        }
        // If equal priority, leftmost wins, so next loses
        // That will be handled when next_idx runs this kernel
    }

    // Also check if previous position's merge would consume us
    // This requires checking if we are someone's "next"
    // For simplicity, we use two passes or atomic operations
}

// Apply all approved merges and update linked list
extern "C" __global__ void bpe_apply_all_merges(
    unsigned int* __restrict__ token_ids,
    unsigned int* __restrict__ next_indices,
    const unsigned int* __restrict__ can_merge,
    const unsigned int* __restrict__ merge_results,
    unsigned int num_tokens
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_tokens) return;
    if (can_merge[idx] == 0) return;

    unsigned int next_idx = next_indices[idx];
    if (next_idx == 0xFFFFFFFF) return;

    // Apply merge: update token, skip next in linked list
    token_ids[idx] = merge_results[idx];
    next_indices[idx] = next_indices[next_idx];  // Skip merged token
}

// Count remaining tokens (parallel reduction)
extern "C" __global__ void bpe_count_valid(
    const unsigned int* __restrict__ next_indices,
    unsigned int* __restrict__ count,
    unsigned int start_idx
) {
    // Single thread walks the linked list
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int cnt = 0;
    unsigned int idx = start_idx;
    while (idx != 0xFFFFFFFF) {
        cnt++;
        idx = next_indices[idx];
    }
    *count = cnt;
}
"#;

/// PTX kernel source for Unigram/Viterbi tokenization (parallel lattice construction)
#[cfg(feature = "cuda")]
pub const UNIGRAM_VITERBI_KERNEL_SRC: &str = r#"
// Unigram trie node structure (flattened for GPU)
// Each node has: char, is_end, token_id, score, children_start, children_count
// Children are stored contiguously in a separate array

// Parallel lattice construction - each thread handles one starting position
extern "C" __global__ void unigram_build_lattice(
    const unsigned char* __restrict__ input,
    const unsigned int* __restrict__ trie_chars,       // Character at each trie node
    const unsigned int* __restrict__ trie_is_end,      // 1 if node is end of token
    const unsigned int* __restrict__ trie_token_ids,   // Token ID if is_end
    const float* __restrict__ trie_scores,             // Log probability if is_end
    const unsigned int* __restrict__ trie_child_start, // Start index of children
    const unsigned int* __restrict__ trie_child_count, // Number of children
    float* __restrict__ lattice_scores,                // Output: best score to reach each position
    unsigned int* __restrict__ lattice_prev_pos,       // Output: previous position for backtracking
    unsigned int* __restrict__ lattice_token_id,       // Output: token ID used to reach this position
    unsigned int input_len,
    unsigned int trie_size
) {
    extern __shared__ float s_scores[];

    unsigned int start = blockIdx.x * blockDim.x + threadIdx.x;

    if (start >= input_len) return;

    // Traverse trie from this starting position
    unsigned int node = 0;  // Root node
    unsigned int pos = start;

    while (pos < input_len) {
        unsigned char c = input[pos];

        // Find child with matching character
        unsigned int child_start = trie_child_start[node];
        unsigned int child_count = trie_child_count[node];
        unsigned int next_node = 0xFFFFFFFF;

        for (unsigned int i = 0; i < child_count; i++) {
            unsigned int child = child_start + i;
            if (child < trie_size && trie_chars[child] == c) {
                next_node = child;
                break;
            }
        }

        if (next_node == 0xFFFFFFFF) break;  // No matching child

        node = next_node;
        pos++;

        // If this is a valid token ending, update lattice
        if (trie_is_end[node]) {
            unsigned int token_id = trie_token_ids[node];
            float score = trie_scores[node];

            // Read previous best score at start position
            float prev_score = (start == 0) ? 0.0f : lattice_scores[start];
            float new_score = prev_score + score;

            // Atomic max update for best score at end position
            // Note: CUDA doesn't have atomicMaxFloat, so we use atomicCAS loop
            float old_score = lattice_scores[pos];
            while (new_score > old_score) {
                float assumed = old_score;
                old_score = __int_as_float(atomicCAS(
                    (unsigned int*)&lattice_scores[pos],
                    __float_as_int(assumed),
                    __float_as_int(new_score)
                ));
                if (old_score == assumed) {
                    // Successfully updated, also update prev_pos and token_id
                    lattice_prev_pos[pos] = start;
                    lattice_token_id[pos] = token_id;
                    break;
                }
            }
        }
    }
}

// Forward pass: compute best scores in parallel waves
extern "C" __global__ void unigram_forward_pass(
    const float* __restrict__ token_scores,      // Score for each (start, end, token_id) tuple
    const unsigned int* __restrict__ token_starts,
    const unsigned int* __restrict__ token_ends,
    const unsigned int* __restrict__ token_ids,
    float* __restrict__ best_scores,             // Best score to reach each position
    unsigned int* __restrict__ best_prev,        // Previous position for backtracking
    unsigned int* __restrict__ best_token,       // Token ID used
    unsigned int num_tokens,
    unsigned int input_len,
    unsigned int wave                            // Current wave (position being computed)
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process all tokens ending at position 'wave'
    if (idx < num_tokens && token_ends[idx] == wave) {
        unsigned int start = token_starts[idx];
        float score = token_scores[idx];

        // Get best score to reach start position
        float prev_score = (start == 0) ? 0.0f : best_scores[start];
        float new_score = prev_score + score;

        // Atomic update if this is better
        float old = atomicMax((int*)&best_scores[wave], __float_as_int(new_score));
        if (__int_as_float(old) < new_score) {
            best_prev[wave] = start;
            best_token[wave] = token_ids[idx];
        }
    }
}

// Viterbi backtrack (sequential on CPU is more efficient)
// But we provide a GPU version for completeness
extern "C" __global__ void unigram_backtrack(
    const unsigned int* __restrict__ best_prev,
    const unsigned int* __restrict__ best_token,
    unsigned int* __restrict__ output_tokens,
    unsigned int* __restrict__ output_count,
    unsigned int end_pos
) {
    // Single thread performs backtracking
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int count = 0;
    unsigned int pos = end_pos;

    // First pass: count tokens
    while (pos > 0) {
        count++;
        pos = best_prev[pos];
    }

    *output_count = count;

    // Second pass: write tokens in reverse order
    pos = end_pos;
    unsigned int write_pos = count;
    while (pos > 0 && write_pos > 0) {
        write_pos--;
        output_tokens[write_pos] = best_token[pos];
        pos = best_prev[pos];
    }
}
"#;

/// PTX kernel source for Character tokenization (SentencePiece Character mode)
/// Each character/byte becomes a separate token - simplest tokenization
#[cfg(feature = "cuda")]
pub const CHARACTER_TOKENIZE_KERNEL_SRC: &str = r#"
// Character tokenization with vocabulary lookup
// Maps each byte/character to its token ID via lookup table

extern "C" __global__ void char_tokenize(
    const unsigned char* __restrict__ input,
    const unsigned int* __restrict__ char_to_token,  // 256-entry lookup for bytes
    unsigned int* __restrict__ output_ids,
    unsigned int input_len
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_len) {
        output_ids[idx] = char_to_token[input[idx]];
    }
}

// UTF-8 aware character tokenization
// Handles multi-byte UTF-8 sequences
extern "C" __global__ void char_tokenize_utf8(
    const unsigned char* __restrict__ input,
    const unsigned int* __restrict__ char_start,     // Start byte mask (1 if char starts here)
    const unsigned long long* __restrict__ vocab_hashes,
    const unsigned int* __restrict__ vocab_ids,
    unsigned int* __restrict__ output_ids,
    unsigned int* __restrict__ output_positions,     // Which input positions have tokens
    unsigned int* __restrict__ output_count,
    unsigned int input_len,
    unsigned int vocab_size,
    unsigned int unk_id
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= input_len) return;
    if (char_start[idx] == 0) return;  // Not a character start

    // Find character length (UTF-8)
    unsigned char c = input[idx];
    unsigned int char_len = 1;
    if ((c & 0x80) == 0) {
        char_len = 1;  // ASCII
    } else if ((c & 0xE0) == 0xC0) {
        char_len = 2;  // 2-byte UTF-8
    } else if ((c & 0xF0) == 0xE0) {
        char_len = 3;  // 3-byte UTF-8
    } else if ((c & 0xF8) == 0xF0) {
        char_len = 4;  // 4-byte UTF-8
    }

    if (idx + char_len > input_len) char_len = input_len - idx;

    // Hash the character bytes
    unsigned long long hash = 14695981039346656037ULL;
    for (unsigned int i = 0; i < char_len; i++) {
        hash ^= input[idx + i];
        hash *= 1099511628211ULL;
    }

    // Lookup in vocab
    unsigned int slot = hash % vocab_size;
    unsigned int token_id = unk_id;

    for (unsigned int probe = 0; probe < 64; probe++) {
        unsigned int probe_slot = (slot + probe) % vocab_size;
        if (vocab_hashes[probe_slot] == hash) {
            token_id = vocab_ids[probe_slot];
            break;
        }
        if (vocab_hashes[probe_slot] == 0) break;
    }

    // Write output
    unsigned int pos = atomicAdd(output_count, 1);
    output_ids[pos] = token_id;
    output_positions[pos] = idx;
}

// Mark UTF-8 character start positions
extern "C" __global__ void mark_utf8_char_starts(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ char_start,
    unsigned int input_len
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= input_len) return;

    unsigned char c = input[idx];

    // UTF-8 continuation bytes have pattern 10xxxxxx
    // All other bytes are character starts
    if ((c & 0xC0) == 0x80) {
        char_start[idx] = 0;  // Continuation byte
    } else {
        char_start[idx] = 1;  // Character start
    }
}
"#;

/// PTX kernel source for Word tokenization (SentencePiece Word mode)
/// Whitespace-delimited tokenization with vocabulary lookup
#[cfg(feature = "cuda")]
pub const WORD_TOKENIZE_KERNEL_SRC: &str = r#"
// FNV-1a hash for variable-length strings
__device__ unsigned long long word_hash(const unsigned char* data, unsigned int len) {
    unsigned long long hash = 14695981039346656037ULL;
    for (unsigned int i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Find word boundaries (whitespace delimited)
// Each thread marks if current position is a word start or end
extern "C" __global__ void word_find_boundaries(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ word_starts,    // Output: 1 if word starts here
    unsigned int* __restrict__ word_ends,      // Output: 1 if word ends here
    unsigned int input_len
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= input_len) return;

    unsigned char c = input[idx];
    bool is_ws = (c == ' ') || (c == '\t') || (c == '\n') || (c == '\r');

    word_starts[idx] = 0;
    word_ends[idx] = 0;

    if (!is_ws) {
        // Check if this is a word start (non-ws after ws or at position 0)
        bool prev_is_ws = (idx == 0) ||
            (input[idx-1] == ' ') || (input[idx-1] == '\t') ||
            (input[idx-1] == '\n') || (input[idx-1] == '\r');

        if (prev_is_ws) {
            word_starts[idx] = 1;
        }

        // Check if this is a word end (non-ws before ws or at end)
        bool next_is_ws = (idx + 1 >= input_len) ||
            (input[idx+1] == ' ') || (input[idx+1] == '\t') ||
            (input[idx+1] == '\n') || (input[idx+1] == '\r');

        if (next_is_ws) {
            word_ends[idx] = 1;
        }
    }
}

// Prefix sum to compute word indices
extern "C" __global__ void word_prefix_sum(
    const unsigned int* __restrict__ word_starts,
    unsigned int* __restrict__ word_indices,   // Output: word index for each start position
    unsigned int input_len
) {
    extern __shared__ unsigned int s_prefix[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    s_prefix[tid] = (idx < input_len) ? word_starts[idx] : 0;
    __syncthreads();

    // Inclusive prefix sum (Hillis-Steele)
    for (unsigned int d = 1; d < blockDim.x; d *= 2) {
        unsigned int val = 0;
        if (tid >= d) {
            val = s_prefix[tid - d];
        }
        __syncthreads();
        s_prefix[tid] += val;
        __syncthreads();
    }

    // Write exclusive sum (shift by 1)
    if (idx < input_len) {
        word_indices[idx] = (tid == 0) ? 0 : s_prefix[tid - 1];
    }
}

// Tokenize words using vocabulary hash table
extern "C" __global__ void word_tokenize_lookup(
    const unsigned char* __restrict__ input,
    const unsigned int* __restrict__ word_start_positions,  // Positions where words start
    const unsigned int* __restrict__ word_lengths,          // Length of each word
    const unsigned long long* __restrict__ vocab_hashes,
    const unsigned int* __restrict__ vocab_ids,
    unsigned int* __restrict__ output_ids,
    unsigned int num_words,
    unsigned int vocab_size,
    unsigned int unk_id
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_words) return;

    unsigned int start = word_start_positions[idx];
    unsigned int len = word_lengths[idx];

    // Hash the word
    unsigned long long hash = word_hash(&input[start], len);

    // Lookup in vocab
    unsigned int slot = hash % vocab_size;
    unsigned int token_id = unk_id;

    for (unsigned int probe = 0; probe < 64; probe++) {
        unsigned int probe_slot = (slot + probe) % vocab_size;
        if (vocab_hashes[probe_slot] == hash) {
            token_id = vocab_ids[probe_slot];
            break;
        }
        if (vocab_hashes[probe_slot] == 0) break;
    }

    output_ids[idx] = token_id;
}

// Compact word boundaries into start positions and lengths
extern "C" __global__ void word_compact_boundaries(
    const unsigned int* __restrict__ word_starts,
    const unsigned int* __restrict__ word_ends,
    const unsigned int* __restrict__ prefix_sums,
    unsigned int* __restrict__ start_positions,
    unsigned int* __restrict__ lengths,
    unsigned int* __restrict__ word_count,
    unsigned int input_len
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= input_len) return;

    if (word_starts[idx] == 1) {
        unsigned int word_idx = prefix_sums[idx];
        start_positions[word_idx] = idx;

        // Find length (scan to find matching end)
        unsigned int len = 0;
        for (unsigned int i = idx; i < input_len; i++) {
            len++;
            if (word_ends[i] == 1) break;
        }
        lengths[word_idx] = len;

        // Update word count atomically (only first thread)
        if (word_idx + 1 > *word_count) {
            atomicMax(word_count, word_idx + 1);
        }
    }
}

// Combined word tokenizer - single pass for small inputs
extern "C" __global__ void word_tokenize_simple(
    const unsigned char* __restrict__ input,
    const unsigned long long* __restrict__ vocab_hashes,
    const unsigned int* __restrict__ vocab_ids,
    unsigned int* __restrict__ output_ids,
    unsigned int* __restrict__ output_count,
    unsigned int input_len,
    unsigned int vocab_size,
    unsigned int unk_id
) {
    // Single thread processes entire input (for short texts)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int count = 0;
    unsigned int word_start = 0;
    bool in_word = false;

    for (unsigned int i = 0; i <= input_len; i++) {
        bool is_ws = (i == input_len) ||
            (input[i] == ' ') || (input[i] == '\t') ||
            (input[i] == '\n') || (input[i] == '\r');

        if (!is_ws && !in_word) {
            // Word start
            word_start = i;
            in_word = true;
        } else if (is_ws && in_word) {
            // Word end - tokenize it
            unsigned int word_len = i - word_start;

            // Hash the word
            unsigned long long hash = 14695981039346656037ULL;
            for (unsigned int j = word_start; j < i; j++) {
                hash ^= input[j];
                hash *= 1099511628211ULL;
            }

            // Lookup
            unsigned int slot = hash % vocab_size;
            unsigned int token_id = unk_id;

            for (unsigned int probe = 0; probe < 64; probe++) {
                unsigned int probe_slot = (slot + probe) % vocab_size;
                if (vocab_hashes[probe_slot] == hash) {
                    token_id = vocab_ids[probe_slot];
                    break;
                }
                if (vocab_hashes[probe_slot] == 0) break;
            }

            output_ids[count++] = token_id;
            in_word = false;
        }
    }

    *output_count = count;
}
"#;

/// PTX kernel source for ASCII check (SIMD-style on GPU)
#[cfg(feature = "cuda")]
pub const ASCII_CHECK_KERNEL_SRC: &str = r#"
extern "C" __global__ void is_all_ascii(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ result,
    unsigned int input_len
) {
    __shared__ unsigned int block_result;

    if (threadIdx.x == 0) {
        block_result = 1; // Assume all ASCII
    }
    __syncthreads();

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_len) {
        if (input[idx] >= 128) {
            atomicAnd(&block_result, 0);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAnd(result, block_result);
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_available_check() {
        let available = is_cuda_available();
        println!("CUDA available: {}", available);
    }

    #[test]
    fn test_get_devices() {
        let devices = get_cuda_devices();
        println!("Found {} CUDA devices", devices.len());
        for dev in &devices {
            println!("  Device {}: {}", dev.id, dev.name);
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_context_creation() {
        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return;
        }

        let ctx = CudaContext::new(0);
        assert!(ctx.is_ok(), "Failed to create context: {:?}", ctx.err());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_memory_allocation() {
        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return;
        }

        let ctx = CudaContext::new(0).expect("Context creation failed");
        let slice = ctx.alloc::<f32>(1024);
        assert!(slice.is_ok(), "Allocation failed: {:?}", slice.err());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_host_device_copy() {
        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return;
        }

        let ctx = CudaContext::new(0).expect("Context creation failed");

        let host_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let device_data = ctx.htod_copy(&host_data).expect("HtoD copy failed");

        let result = ctx.dtoh_copy(&device_data).expect("DtoH copy failed");

        assert_eq!(host_data, result);
    }
}
