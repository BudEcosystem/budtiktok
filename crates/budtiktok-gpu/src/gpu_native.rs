//! GPU-Native Tokenization Pipeline
//!
//! High-performance tokenization that runs ENTIRELY on GPU, based on state-of-the-art research:
//!
//! - **BlockBPE** (arxiv:2507.11941): GPU-resident hashmaps, block-level prefix scans
//! - **RAPIDS cuDF nvtext**: 483x faster WordPiece via perfect hashing
//!
//! Key optimizations:
//! - All data stays on GPU (zero CPU<->GPU transfers during tokenization)
//! - Parallel pre-tokenization using warp-level primitives
//! - GPU-optimized hash table for vocabulary lookup (O(1) per token)
//! - Prefix scan for result compaction
//! - Coalesced memory access patterns
//!
//! Architecture:
//! ```text
//! Input Text (Host) -> [H2D Transfer] -> GPU Pipeline -> [D2H Transfer] -> Token IDs (Host)
//!                                           |
//!                                           v
//!                      +-------------------------------------------+
//!                      |  1. Normalize (lowercase, unicode)        |
//!                      |  2. Pre-tokenize (find word boundaries)   |
//!                      |  3. WordPiece (hash lookup, greedy match) |
//!                      |  4. Compact results (prefix scan)         |
//!                      +-------------------------------------------+
//!                             All stages run on GPU
//! ```

use crate::backend::GpuError;

#[cfg(feature = "cuda")]
use crate::cuda::CudaContext;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Configuration for GPU-native tokenizer
#[derive(Debug, Clone)]
pub struct GpuNativeConfig {
    /// Maximum sequence length in bytes
    pub max_seq_bytes: usize,
    /// Maximum tokens per sequence
    pub max_tokens_per_seq: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Whether to lowercase text
    pub do_lower_case: bool,
    /// Continuation prefix for subwords (e.g., "##")
    pub continuation_prefix: String,
    /// Maximum word length before treating as [UNK]
    pub max_word_chars: usize,
    /// Block size for CUDA kernels (256 or 512 recommended)
    pub block_size: u32,
}

impl Default for GpuNativeConfig {
    fn default() -> Self {
        Self {
            max_seq_bytes: 8192,
            max_tokens_per_seq: 512,
            max_batch_size: 1024,
            do_lower_case: true,
            continuation_prefix: "##".to_string(),
            max_word_chars: 100,
            block_size: 256,
        }
    }
}

/// GPU-resident vocabulary for O(1) token lookup
/// Uses perfect hashing similar to RAPIDS cuDF approach
#[cfg(feature = "cuda")]
pub struct GpuVocabulary {
    /// Hash table: hash -> (token_string_offset, token_id)
    hash_table_hashes: CudaSlice<u64>,
    hash_table_ids: CudaSlice<u32>,
    /// Packed token strings for collision resolution
    token_strings: CudaSlice<u8>,
    token_offsets: CudaSlice<u32>,
    token_lengths: CudaSlice<u16>,
    /// Table size (power of 2)
    table_size: usize,
    /// Special token IDs
    unk_id: u32,
    cls_id: u32,
    sep_id: u32,
    pad_id: u32,
    /// Continuation prefix info
    prefix_hash: u64,
    prefix_len: usize,
}

/// Kernel configuration as a u32 array (16 elements)
/// Layout:
/// [0] table_size, [1] unk_id, [2] cls_id, [3] sep_id,
/// [4] prefix_hash_lo, [5] prefix_hash_hi, [6] prefix_len, [7] num_sequences,
/// [8] max_words_per_seq, [9] max_tokens_per_seq, [10] max_word_chars, [11-15] reserved
pub const KERNEL_CONFIG_SIZE: usize = 16;

/// GPU-native tokenizer that runs the entire pipeline on GPU
#[cfg(feature = "cuda")]
pub struct GpuNativeTokenizer {
    context: Arc<CudaContext>,
    vocab: GpuVocabulary,
    config: GpuNativeConfig,
    // Pre-allocated buffers for batch processing
    d_input_bytes: CudaSlice<u8>,
    d_input_offsets: CudaSlice<u32>,
    d_input_lengths: CudaSlice<u32>,
    d_normalized: CudaSlice<u8>,
    d_word_bounds: CudaSlice<u32>, // Interleaved: [start0, end0, start1, end1, ...]
    d_word_count: CudaSlice<u32>,
    d_output_ids: CudaSlice<u32>,
    d_output_lengths: CudaSlice<u32>,
    d_kernel_config: CudaSlice<u32>,
}

// =============================================================================
// CUDA Kernel Source Code for GPU-Native Pipeline
// =============================================================================

/// Complete GPU-native tokenization kernels
/// Based on BlockBPE (arxiv:2507.11941) and RAPIDS cuDF approaches
#[cfg(feature = "cuda")]
pub const GPU_NATIVE_KERNELS: &str = r#"
// =============================================================================
// FNV-1a Hash Function (GPU-optimized)
// =============================================================================

__device__ __forceinline__ unsigned long long fnv1a_hash_gpu(
    const unsigned char* data,
    unsigned int len
) {
    unsigned long long hash = 14695981039346656037ULL;
    #pragma unroll 4
    for (unsigned int i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// Hash with continuation prefix prepended
__device__ __forceinline__ unsigned long long fnv1a_hash_with_prefix(
    const unsigned char* data,
    unsigned int len,
    unsigned long long prefix_hash,
    unsigned int prefix_len
) {
    // Start with prefix hash state
    unsigned long long hash = prefix_hash;
    #pragma unroll 4
    for (unsigned int i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// =============================================================================
// Stage 1: Normalization Kernel (Lowercase + Basic Cleanup)
// =============================================================================

extern "C" __global__ void normalize_text(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const unsigned int* __restrict__ seq_offsets,  // Start offset of each sequence
    const unsigned int* __restrict__ seq_lengths,  // Length of each sequence
    unsigned int num_sequences,
    bool do_lower_case
) {
    unsigned int seq_idx = blockIdx.x;
    if (seq_idx >= num_sequences) return;

    unsigned int seq_start = seq_offsets[seq_idx];
    unsigned int seq_len = seq_lengths[seq_idx];

    // Each thread handles multiple characters (coalesced access)
    for (unsigned int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        unsigned int pos = seq_start + i;
        unsigned char c = input[pos];

        // Lowercase ASCII A-Z
        if (do_lower_case && c >= 'A' && c <= 'Z') {
            c = c + 32;  // 'a' - 'A' = 32
        }

        // Replace control characters with space (except newline/tab)
        if (c < 32 && c != '\n' && c != '\t' && c != '\r') {
            c = ' ';
        }

        output[pos] = c;
    }
}

// =============================================================================
// Stage 2: Pre-tokenization Kernel (Find Word Boundaries)
// Uses parallel prefix scan pattern from BlockBPE
// =============================================================================

// Classify each byte as word boundary indicator
__device__ __forceinline__ int is_word_boundary(unsigned char c) {
    // Whitespace characters
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return 1;
    // Common punctuation that splits words
    if (c == '.' || c == ',' || c == '!' || c == '?' ||
        c == ';' || c == ':' || c == '"' || c == '\'') return 1;
    if (c == '(' || c == ')' || c == '[' || c == ']' ||
        c == '{' || c == '}' || c == '<' || c == '>') return 1;
    return 0;
}

// Find word boundaries for a single sequence
// Output: word_bounds array (interleaved: [start0, end0, start1, end1, ...]) with word count
extern "C" __global__ void find_word_boundaries(
    const unsigned char* __restrict__ input,
    unsigned int* __restrict__ word_bounds,     // Output: interleaved [start, end] pairs
    unsigned int* __restrict__ word_counts,     // Output: number of words per sequence
    const unsigned int* __restrict__ seq_offsets,
    const unsigned int* __restrict__ seq_lengths,
    unsigned int num_sequences,
    unsigned int max_words_per_seq
) {
    extern __shared__ unsigned int s_word_data[];
    unsigned int* s_is_start = s_word_data;
    unsigned int* s_prefix_sum = s_word_data + blockDim.x;

    unsigned int seq_idx = blockIdx.x;
    if (seq_idx >= num_sequences) return;

    unsigned int seq_start = seq_offsets[seq_idx];
    unsigned int seq_len = seq_lengths[seq_idx];
    unsigned int tid = threadIdx.x;

    // Output offset for this sequence's words (2 u32 per word: start, end)
    unsigned int word_out_offset = seq_idx * max_words_per_seq * 2;

    // Process in chunks if sequence is longer than block size
    unsigned int local_word_count = 0;

    for (unsigned int chunk_start = 0; chunk_start < seq_len; chunk_start += blockDim.x) {
        unsigned int pos = chunk_start + tid;
        unsigned int global_pos = seq_start + pos;

        // Determine if this position is a word start
        int is_start = 0;
        if (pos < seq_len) {
            unsigned char c = input[global_pos];
            bool is_boundary = is_word_boundary(c);

            // Word starts after a boundary or at position 0
            if (!is_boundary) {
                if (pos == 0) {
                    is_start = 1;
                } else {
                    unsigned char prev = input[global_pos - 1];
                    if (is_word_boundary(prev)) {
                        is_start = 1;
                    }
                }
            }
        }

        s_is_start[tid] = is_start;
        __syncthreads();

        // Parallel prefix sum to find word indices
        s_prefix_sum[tid] = s_is_start[tid];
        __syncthreads();

        // Hillis-Steele prefix sum
        for (unsigned int d = 1; d < blockDim.x; d *= 2) {
            unsigned int val = 0;
            if (tid >= d) {
                val = s_prefix_sum[tid - d];
            }
            __syncthreads();
            s_prefix_sum[tid] += val;
            __syncthreads();
        }

        // Convert to exclusive prefix sum
        unsigned int exclusive_sum = (tid == 0) ? 0 : s_prefix_sum[tid - 1];
        __syncthreads();

        // Write word boundaries (interleaved format)
        if (is_start && pos < seq_len) {
            unsigned int word_idx = local_word_count + exclusive_sum;
            if (word_idx < max_words_per_seq) {
                // Find word end (scan forward to next boundary)
                unsigned int end_pos = global_pos;
                while (end_pos < seq_start + seq_len && !is_word_boundary(input[end_pos])) {
                    end_pos++;
                }
                // Write interleaved: [start, end]
                word_bounds[word_out_offset + word_idx * 2] = global_pos;
                word_bounds[word_out_offset + word_idx * 2 + 1] = end_pos;
            }
        }

        // Update local word count from this chunk
        __syncthreads();
        if (tid == blockDim.x - 1) {
            local_word_count += s_prefix_sum[tid];
        }
        __syncthreads();

        // Broadcast updated count
        local_word_count = __shfl_sync(0xFFFFFFFF, local_word_count, blockDim.x - 1);
    }

    // Write final word count for this sequence
    if (tid == 0) {
        word_counts[seq_idx] = min(local_word_count, max_words_per_seq);
    }
}

// =============================================================================
// Stage 3: WordPiece Tokenization Kernel
// GPU-optimized greedy longest-match with hash table lookup
// Based on RAPIDS cuDF's 483x speedup approach
// =============================================================================

// Lookup a token in the hash table
// Returns token ID if found, UNK_ID otherwise
__device__ unsigned int vocab_lookup(
    const unsigned char* word,
    unsigned int word_len,
    const unsigned long long* __restrict__ hash_table_hashes,
    const unsigned int* __restrict__ hash_table_ids,
    const unsigned char* __restrict__ token_strings,
    const unsigned int* __restrict__ token_offsets,
    const unsigned short* __restrict__ token_lengths,
    unsigned int table_size,
    unsigned int unk_id,
    bool use_prefix,
    unsigned long long prefix_hash,
    unsigned int prefix_len
) {
    // Compute hash of the word (with optional prefix for subwords)
    unsigned long long hash;
    if (use_prefix) {
        hash = fnv1a_hash_with_prefix(word, word_len, prefix_hash, prefix_len);
    } else {
        hash = fnv1a_hash_gpu(word, word_len);
    }

    unsigned int slot = hash % table_size;

    // Linear probing with max 64 probes
    #pragma unroll 4
    for (unsigned int probe = 0; probe < 64; probe++) {
        unsigned int probe_slot = (slot + probe) % table_size;
        unsigned long long stored_hash = hash_table_hashes[probe_slot];

        if (stored_hash == 0) {
            return unk_id;  // Empty slot, not found
        }

        if (stored_hash == hash) {
            // Verify string match (handle hash collisions)
            unsigned int token_offset = token_offsets[probe_slot];
            unsigned short token_len = token_lengths[probe_slot];

            // For prefix tokens, effective length includes prefix
            unsigned int effective_len = use_prefix ? (word_len + prefix_len) : word_len;

            if (token_len == effective_len) {
                // Compare strings
                bool match = true;
                const unsigned char* token_str = &token_strings[token_offset];

                if (use_prefix) {
                    // Skip prefix in stored token, compare rest
                    for (unsigned int i = 0; i < word_len && match; i++) {
                        if (token_str[prefix_len + i] != word[i]) {
                            match = false;
                        }
                    }
                } else {
                    for (unsigned int i = 0; i < word_len && match; i++) {
                        if (token_str[i] != word[i]) {
                            match = false;
                        }
                    }
                }

                if (match) {
                    return hash_table_ids[probe_slot];
                }
            }
        }
    }

    return unk_id;
}

// Config array layout:
// [0] table_size, [1] unk_id, [2] cls_id, [3] sep_id,
// [4] prefix_hash_lo, [5] prefix_hash_hi, [6] prefix_len, [7] num_sequences,
// [8] max_words_per_seq, [9] max_tokens_per_seq, [10] max_word_chars, [11] padding

// WordPiece tokenization for a batch of words
// Each block processes one sequence's words
// Uses config array and interleaved word_bounds to fit in 12 parameters for cudarc
extern "C" __global__ void wordpiece_tokenize(
    const unsigned char* __restrict__ input,
    const unsigned int* __restrict__ word_bounds,   // Interleaved: [start0, end0, start1, end1, ...]
    const unsigned int* __restrict__ word_counts,
    unsigned int* __restrict__ output_ids,
    unsigned int* __restrict__ output_lengths,
    const unsigned long long* __restrict__ hash_table_hashes,
    const unsigned int* __restrict__ hash_table_ids,
    const unsigned char* __restrict__ token_strings,
    const unsigned int* __restrict__ token_offsets,
    const unsigned short* __restrict__ token_lengths,
    const unsigned int* __restrict__ config
) {
    // Unpack config array
    unsigned int table_size = config[0];
    unsigned int unk_id = config[1];
    unsigned int cls_id = config[2];
    unsigned int sep_id = config[3];
    unsigned long long prefix_hash = ((unsigned long long)config[5] << 32) | config[4];
    unsigned int prefix_len = config[6];
    unsigned int num_sequences = config[7];
    unsigned int max_words_per_seq = config[8];
    unsigned int max_tokens_per_seq = config[9];
    unsigned int max_word_chars = config[10];

    unsigned int seq_idx = blockIdx.x;
    if (seq_idx >= num_sequences) return;

    // word_bounds is interleaved, so offset is 2 * max_words_per_seq per sequence
    unsigned int word_bound_offset = seq_idx * max_words_per_seq * 2;
    unsigned int out_offset = seq_idx * max_tokens_per_seq;
    unsigned int num_words = word_counts[seq_idx];

    unsigned int tid = threadIdx.x;

    // Thread 0 handles special tokens and coordinates
    __shared__ unsigned int s_token_count;

    if (tid == 0) {
        s_token_count = 0;

        // Add [CLS] token
        if (cls_id != 0xFFFFFFFF) {
            output_ids[out_offset + s_token_count] = cls_id;
            s_token_count++;
        }
    }
    __syncthreads();

    // Process words in parallel (each thread handles multiple words)
    // Use atomics to track output position
    for (unsigned int w = tid; w < num_words; w += blockDim.x) {
        // Read from interleaved word_bounds: [start0, end0, start1, end1, ...]
        unsigned int word_start = word_bounds[word_bound_offset + w * 2];
        unsigned int word_end = word_bounds[word_bound_offset + w * 2 + 1];
        unsigned int word_len = word_end - word_start;

        if (word_len == 0 || word_len > max_word_chars) {
            // Empty word or too long - emit [UNK]
            unsigned int pos = atomicAdd(&s_token_count, 1);
            if (pos < max_tokens_per_seq - 1) {  // Reserve space for [SEP]
                output_ids[out_offset + pos] = unk_id;
            }
            continue;
        }

        const unsigned char* word = &input[word_start];

        // Try whole word first
        unsigned int whole_word_id = vocab_lookup(
            word, word_len,
            hash_table_hashes, hash_table_ids,
            token_strings, token_offsets, token_lengths,
            table_size, unk_id, false, 0, 0
        );

        if (whole_word_id != unk_id) {
            // Whole word found
            unsigned int pos = atomicAdd(&s_token_count, 1);
            if (pos < max_tokens_per_seq - 1) {
                output_ids[out_offset + pos] = whole_word_id;
            }
            continue;
        }

        // WordPiece greedy longest-match
        unsigned int char_pos = 0;
        bool is_first_subword = true;
        bool failed = false;

        while (char_pos < word_len && !failed) {
            // Try longest subword first
            unsigned int best_len = 0;
            unsigned int best_id = unk_id;

            // Greedy search from longest to shortest
            for (unsigned int end = word_len; end > char_pos; end--) {
                unsigned int subword_len = end - char_pos;
                const unsigned char* subword = &word[char_pos];

                unsigned int token_id;
                if (is_first_subword) {
                    token_id = vocab_lookup(
                        subword, subword_len,
                        hash_table_hashes, hash_table_ids,
                        token_strings, token_offsets, token_lengths,
                        table_size, unk_id, false, 0, 0
                    );
                } else {
                    // Use continuation prefix hash
                    token_id = vocab_lookup(
                        subword, subword_len,
                        hash_table_hashes, hash_table_ids,
                        token_strings, token_offsets, token_lengths,
                        table_size, unk_id, true, prefix_hash, prefix_len
                    );
                }

                if (token_id != unk_id) {
                    best_len = subword_len;
                    best_id = token_id;
                    break;  // Found longest match
                }
            }

            if (best_len == 0) {
                // No match found - emit [UNK] for whole word
                failed = true;
                break;
            }

            // Emit token
            unsigned int pos = atomicAdd(&s_token_count, 1);
            if (pos < max_tokens_per_seq - 1) {
                output_ids[out_offset + pos] = best_id;
            }

            char_pos += best_len;
            is_first_subword = false;
        }

        if (failed) {
            // Emit [UNK] for failed word
            unsigned int pos = atomicAdd(&s_token_count, 1);
            if (pos < max_tokens_per_seq - 1) {
                output_ids[out_offset + pos] = unk_id;
            }
        }
    }

    __syncthreads();

    // Thread 0 adds [SEP] and writes final length
    if (tid == 0) {
        if (sep_id != 0xFFFFFFFF && s_token_count < max_tokens_per_seq) {
            output_ids[out_offset + s_token_count] = sep_id;
            s_token_count++;
        }
        output_lengths[seq_idx] = s_token_count;
    }
}

// =============================================================================
// Optimized single-sequence tokenization (for batch size 1)
// Uses warp-level primitives for maximum efficiency
// =============================================================================

extern "C" __global__ void wordpiece_tokenize_single(
    const unsigned char* __restrict__ input,
    unsigned int input_len,
    unsigned int* __restrict__ output_ids,
    unsigned int* __restrict__ output_length,
    const unsigned long long* __restrict__ hash_table_hashes,
    const unsigned int* __restrict__ hash_table_ids,
    const unsigned char* __restrict__ token_strings,
    const unsigned int* __restrict__ token_offsets,
    const unsigned short* __restrict__ token_lengths,
    unsigned int table_size,
    unsigned int unk_id,
    unsigned int cls_id,
    unsigned int sep_id,
    unsigned long long prefix_hash,
    unsigned int prefix_len,
    unsigned int max_tokens,
    unsigned int max_word_chars,
    bool do_lower_case
) {
    __shared__ unsigned int s_token_count;
    __shared__ unsigned char s_normalized[8192];  // Shared memory for normalized text

    unsigned int tid = threadIdx.x;

    // Step 1: Normalize text in parallel
    for (unsigned int i = tid; i < input_len; i += blockDim.x) {
        unsigned char c = input[i];
        if (do_lower_case && c >= 'A' && c <= 'Z') {
            c = c + 32;
        }
        s_normalized[i] = c;
    }
    __syncthreads();

    // Step 2: Thread 0 performs sequential tokenization
    // (For single sequence, sequential is often faster than parallel coordination)
    if (tid == 0) {
        s_token_count = 0;

        // Add [CLS]
        if (cls_id != 0xFFFFFFFF) {
            output_ids[s_token_count++] = cls_id;
        }

        // Find and tokenize words
        unsigned int pos = 0;
        while (pos < input_len && s_token_count < max_tokens - 1) {
            // Skip whitespace
            while (pos < input_len && is_word_boundary(s_normalized[pos])) {
                pos++;
            }

            if (pos >= input_len) break;

            // Find word end
            unsigned int word_start = pos;
            while (pos < input_len && !is_word_boundary(s_normalized[pos])) {
                pos++;
            }
            unsigned int word_len = pos - word_start;

            if (word_len == 0) continue;
            if (word_len > max_word_chars) {
                output_ids[s_token_count++] = unk_id;
                continue;
            }

            const unsigned char* word = &s_normalized[word_start];

            // Try whole word
            unsigned int whole_id = vocab_lookup(
                word, word_len,
                hash_table_hashes, hash_table_ids,
                token_strings, token_offsets, token_lengths,
                table_size, unk_id, false, 0, 0
            );

            if (whole_id != unk_id) {
                output_ids[s_token_count++] = whole_id;
                continue;
            }

            // WordPiece subword tokenization
            unsigned int char_pos = 0;
            bool is_first = true;
            bool failed = false;

            while (char_pos < word_len && s_token_count < max_tokens - 1) {
                unsigned int best_len = 0;
                unsigned int best_id = unk_id;

                for (unsigned int end = word_len; end > char_pos; end--) {
                    unsigned int sublen = end - char_pos;
                    const unsigned char* sub = &word[char_pos];

                    unsigned int tok_id = vocab_lookup(
                        sub, sublen,
                        hash_table_hashes, hash_table_ids,
                        token_strings, token_offsets, token_lengths,
                        table_size, unk_id, !is_first, prefix_hash, prefix_len
                    );

                    if (tok_id != unk_id) {
                        best_len = sublen;
                        best_id = tok_id;
                        break;
                    }
                }

                if (best_len == 0) {
                    failed = true;
                    break;
                }

                output_ids[s_token_count++] = best_id;
                char_pos += best_len;
                is_first = false;
            }

            if (failed) {
                output_ids[s_token_count++] = unk_id;
            }
        }

        // Add [SEP]
        if (sep_id != 0xFFFFFFFF && s_token_count < max_tokens) {
            output_ids[s_token_count++] = sep_id;
        }

        *output_length = s_token_count;
    }
}
"#;

// =============================================================================
// Implementation
// =============================================================================

/// FNV-1a hash (CPU version for vocabulary building)
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 14695981039346656037;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

/// Compute partial hash for continuation prefix
fn compute_prefix_hash(prefix: &str) -> u64 {
    fnv1a_hash(prefix.as_bytes())
}

#[cfg(feature = "cuda")]
impl GpuVocabulary {
    /// Build GPU vocabulary from token-to-id mapping
    pub fn new(
        ctx: &CudaContext,
        vocab: &std::collections::HashMap<String, u32>,
        continuation_prefix: &str,
    ) -> Result<Self, GpuError> {
        // Find special tokens
        let unk_id = vocab.get("[UNK]").copied().unwrap_or(0);
        let cls_id = vocab.get("[CLS]").copied().unwrap_or(0xFFFFFFFF);
        let sep_id = vocab.get("[SEP]").copied().unwrap_or(0xFFFFFFFF);
        let pad_id = vocab.get("[PAD]").copied().unwrap_or(0);

        // Build hash table with 2x capacity for low collisions
        let table_size = (vocab.len() * 2).next_power_of_two().max(1024);
        let mut hashes = vec![0u64; table_size];
        let mut ids = vec![0u32; table_size];

        // Pack token strings
        let mut packed_strings: Vec<u8> = Vec::new();
        let mut offsets = vec![0u32; table_size];
        let mut lengths = vec![0u16; table_size];

        for (token, &id) in vocab.iter() {
            let hash = fnv1a_hash(token.as_bytes());
            let mut slot = (hash as usize) % table_size;

            // Linear probing
            for _ in 0..64 {
                if hashes[slot] == 0 {
                    hashes[slot] = hash;
                    ids[slot] = id;
                    offsets[slot] = packed_strings.len() as u32;
                    lengths[slot] = token.len() as u16;
                    packed_strings.extend_from_slice(token.as_bytes());
                    break;
                }
                slot = (slot + 1) % table_size;
            }
        }

        // Copy to GPU
        let device = ctx.device();
        let hash_table_hashes = device
            .htod_copy(hashes)
            .map_err(|e| GpuError::Cuda(format!("Hash table copy failed: {}", e)))?;
        let hash_table_ids = device
            .htod_copy(ids)
            .map_err(|e| GpuError::Cuda(format!("ID table copy failed: {}", e)))?;
        let token_strings = device
            .htod_copy(packed_strings)
            .map_err(|e| GpuError::Cuda(format!("Token strings copy failed: {}", e)))?;
        let token_offsets = device
            .htod_copy(offsets)
            .map_err(|e| GpuError::Cuda(format!("Offsets copy failed: {}", e)))?;
        let token_lengths = device
            .htod_copy(lengths)
            .map_err(|e| GpuError::Cuda(format!("Lengths copy failed: {}", e)))?;

        let prefix_hash = compute_prefix_hash(continuation_prefix);

        Ok(Self {
            hash_table_hashes,
            hash_table_ids,
            token_strings,
            token_offsets,
            token_lengths,
            table_size,
            unk_id,
            cls_id,
            sep_id,
            pad_id,
            prefix_hash,
            prefix_len: continuation_prefix.len(),
        })
    }
}

#[cfg(feature = "cuda")]
impl GpuNativeTokenizer {
    const MODULE_NAME: &'static str = "gpu_native";
    const FUNC_NORMALIZE: &'static str = "normalize_text";
    const FUNC_FIND_WORDS: &'static str = "find_word_boundaries";
    const FUNC_WORDPIECE: &'static str = "wordpiece_tokenize";
    const FUNC_WORDPIECE_SINGLE: &'static str = "wordpiece_tokenize_single";

    /// Create a new GPU-native tokenizer
    pub fn new(
        ctx: Arc<CudaContext>,
        vocab: &std::collections::HashMap<String, u32>,
        config: GpuNativeConfig,
    ) -> Result<Self, GpuError> {
        // Compile and load kernels
        let ptx = compile_ptx(GPU_NATIVE_KERNELS)
            .map_err(|e| GpuError::Cuda(format!("Kernel compilation failed: {}", e)))?;

        ctx.device()
            .load_ptx(
                ptx,
                Self::MODULE_NAME,
                &[
                    Self::FUNC_NORMALIZE,
                    Self::FUNC_FIND_WORDS,
                    Self::FUNC_WORDPIECE,
                    Self::FUNC_WORDPIECE_SINGLE,
                ],
            )
            .map_err(|e| GpuError::Cuda(format!("PTX load failed: {}", e)))?;

        // Build GPU vocabulary
        let gpu_vocab = GpuVocabulary::new(&ctx, vocab, &config.continuation_prefix)?;

        // Pre-allocate buffers
        let device = ctx.device();
        let max_input_bytes = config.max_batch_size * config.max_seq_bytes;
        let max_words = config.max_batch_size * config.max_seq_bytes / 4; // Avg 4 chars per word
        let max_output = config.max_batch_size * config.max_tokens_per_seq;

        let d_input_bytes = device
            .alloc_zeros::<u8>(max_input_bytes)
            .map_err(|e| GpuError::MemoryAllocation(format!("Input buffer: {}", e)))?;
        let d_input_offsets = device
            .alloc_zeros::<u32>(config.max_batch_size + 1)
            .map_err(|e| GpuError::MemoryAllocation(format!("Offsets buffer: {}", e)))?;
        let d_input_lengths = device
            .alloc_zeros::<u32>(config.max_batch_size)
            .map_err(|e| GpuError::MemoryAllocation(format!("Lengths buffer: {}", e)))?;
        let d_normalized = device
            .alloc_zeros::<u8>(max_input_bytes)
            .map_err(|e| GpuError::MemoryAllocation(format!("Normalized buffer: {}", e)))?;
        // Word bounds is interleaved: [start0, end0, start1, end1, ...] so 2x size
        let d_word_bounds = device
            .alloc_zeros::<u32>(max_words * 2)
            .map_err(|e| GpuError::MemoryAllocation(format!("Word bounds buffer: {}", e)))?;
        let d_word_count = device
            .alloc_zeros::<u32>(config.max_batch_size)
            .map_err(|e| GpuError::MemoryAllocation(format!("Word count buffer: {}", e)))?;
        let d_output_ids = device
            .alloc_zeros::<u32>(max_output)
            .map_err(|e| GpuError::MemoryAllocation(format!("Output buffer: {}", e)))?;
        let d_output_lengths = device
            .alloc_zeros::<u32>(config.max_batch_size)
            .map_err(|e| GpuError::MemoryAllocation(format!("Output lengths buffer: {}", e)))?;
        let d_kernel_config = device
            .alloc_zeros::<u32>(KERNEL_CONFIG_SIZE)
            .map_err(|e| GpuError::MemoryAllocation(format!("Config buffer: {}", e)))?;

        Ok(Self {
            context: ctx,
            vocab: gpu_vocab,
            config,
            d_input_bytes,
            d_input_offsets,
            d_input_lengths,
            d_normalized,
            d_word_bounds,
            d_word_count,
            d_output_ids,
            d_output_lengths,
            d_kernel_config,
        })
    }

    /// Tokenize a batch of texts entirely on GPU
    pub fn encode_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<u32>>, GpuError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let num_sequences = texts.len();
        if num_sequences > self.config.max_batch_size {
            return Err(GpuError::InvalidConfig(format!(
                "Batch size {} exceeds max {}",
                num_sequences, self.config.max_batch_size
            )));
        }

        // Pack input texts
        let max_input_bytes = self.config.max_batch_size * self.config.max_seq_bytes;
        let mut packed_input: Vec<u8> = Vec::with_capacity(max_input_bytes);
        let mut offsets: Vec<u32> = Vec::with_capacity(self.config.max_batch_size + 1);
        offsets.push(0);
        let mut lengths: Vec<u32> = Vec::with_capacity(self.config.max_batch_size);

        for text in texts {
            let bytes = text.as_bytes();
            let len = bytes.len().min(self.config.max_seq_bytes);
            lengths.push(len as u32);
            packed_input.extend_from_slice(&bytes[..len]);
            offsets.push(packed_input.len() as u32);
        }

        // Pad vectors to match pre-allocated GPU buffer sizes (required by htod_sync_copy_into)
        packed_input.resize(max_input_bytes, 0);
        offsets.resize(self.config.max_batch_size + 1, offsets.last().copied().unwrap_or(0));
        lengths.resize(self.config.max_batch_size, 0);

        // Upload to GPU
        let device = self.context.device();
        device
            .htod_sync_copy_into(&packed_input, &mut self.d_input_bytes)
            .map_err(|e| GpuError::Cuda(format!("Input upload failed: {}", e)))?;
        device
            .htod_sync_copy_into(&offsets, &mut self.d_input_offsets)
            .map_err(|e| GpuError::Cuda(format!("Offsets upload failed: {}", e)))?;
        device
            .htod_sync_copy_into(&lengths, &mut self.d_input_lengths)
            .map_err(|e| GpuError::Cuda(format!("Lengths upload failed: {}", e)))?;

        // Stage 1: Normalize
        let normalize_func = device
            .get_func(Self::MODULE_NAME, Self::FUNC_NORMALIZE)
            .ok_or_else(|| GpuError::Cuda("Normalize function not found".into()))?;

        let launch_cfg = LaunchConfig {
            grid_dim: (num_sequences as u32, 1, 1),
            block_dim: (self.config.block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            normalize_func.launch(
                launch_cfg,
                (
                    &self.d_input_bytes,
                    &mut self.d_normalized,
                    &self.d_input_offsets,
                    &self.d_input_lengths,
                    num_sequences as u32,
                    self.config.do_lower_case,
                ),
            )
        }
        .map_err(|e| GpuError::KernelExecution(format!("Normalize kernel failed: {}", e)))?;

        // Stage 2: Find word boundaries
        let find_words_func = device
            .get_func(Self::MODULE_NAME, Self::FUNC_FIND_WORDS)
            .ok_or_else(|| GpuError::Cuda("Find words function not found".into()))?;

        let max_words_per_seq = self.config.max_seq_bytes / 4; // Conservative estimate
        let shared_mem = (self.config.block_size as usize * 2 * std::mem::size_of::<u32>()) as u32;

        let launch_cfg_words = LaunchConfig {
            grid_dim: (num_sequences as u32, 1, 1),
            block_dim: (self.config.block_size, 1, 1),
            shared_mem_bytes: shared_mem,
        };

        unsafe {
            find_words_func.launch(
                launch_cfg_words,
                (
                    &self.d_normalized,
                    &mut self.d_word_bounds,
                    &mut self.d_word_count,
                    &self.d_input_offsets,
                    &self.d_input_lengths,
                    num_sequences as u32,
                    max_words_per_seq as u32,
                ),
            )
        }
        .map_err(|e| GpuError::KernelExecution(format!("Find words kernel failed: {}", e)))?;

        // Stage 3: WordPiece tokenization
        let wordpiece_func = device
            .get_func(Self::MODULE_NAME, Self::FUNC_WORDPIECE)
            .ok_or_else(|| GpuError::Cuda("WordPiece function not found".into()))?;

        // Prepare kernel config as u32 array (16 elements)
        // Layout: [0] table_size, [1] unk_id, [2] cls_id, [3] sep_id,
        //         [4] prefix_hash_lo, [5] prefix_hash_hi, [6] prefix_len, [7] num_sequences,
        //         [8] max_words_per_seq, [9] max_tokens_per_seq, [10] max_word_chars, [11-15] reserved
        let kernel_config: [u32; KERNEL_CONFIG_SIZE] = [
            self.vocab.table_size as u32,
            self.vocab.unk_id,
            self.vocab.cls_id,
            self.vocab.sep_id,
            (self.vocab.prefix_hash & 0xFFFFFFFF) as u32,
            (self.vocab.prefix_hash >> 32) as u32,
            self.vocab.prefix_len as u32,
            num_sequences as u32,
            max_words_per_seq as u32,
            self.config.max_tokens_per_seq as u32,
            self.config.max_word_chars as u32,
            0, 0, 0, 0, 0, // reserved/padding
        ];

        // Upload config to GPU
        device
            .htod_sync_copy_into(&kernel_config, &mut self.d_kernel_config)
            .map_err(|e| GpuError::Cuda(format!("Config upload failed: {}", e)))?;

        // WordPiece kernel: 11 parameters (within cudarc's 12 param limit)
        unsafe {
            wordpiece_func.launch(
                launch_cfg,
                (
                    &self.d_normalized,
                    &self.d_word_bounds,
                    &self.d_word_count,
                    &mut self.d_output_ids,
                    &mut self.d_output_lengths,
                    &self.vocab.hash_table_hashes,
                    &self.vocab.hash_table_ids,
                    &self.vocab.token_strings,
                    &self.vocab.token_offsets,
                    &self.vocab.token_lengths,
                    &self.d_kernel_config,
                ),
            )
        }
        .map_err(|e| GpuError::KernelExecution(format!("WordPiece kernel failed: {}", e)))?;

        // Synchronize
        self.context
            .synchronize()
            .map_err(|e| GpuError::Cuda(format!("Sync failed: {}", e)))?;

        // Download results
        let output_lengths: Vec<u32> = device
            .dtoh_sync_copy(&self.d_output_lengths)
            .map_err(|e| GpuError::Cuda(format!("Output lengths download failed: {}", e)))?;

        let all_output_ids: Vec<u32> = device
            .dtoh_sync_copy(&self.d_output_ids)
            .map_err(|e| GpuError::Cuda(format!("Output IDs download failed: {}", e)))?;

        // Unpack results
        let mut results = Vec::with_capacity(num_sequences);
        for i in 0..num_sequences {
            let len = output_lengths[i] as usize;
            let start = i * self.config.max_tokens_per_seq;
            let end = start + len.min(self.config.max_tokens_per_seq);
            results.push(all_output_ids[start..end].to_vec());
        }

        Ok(results)
    }

    /// Get device info
    pub fn device_info(&self) -> &crate::cuda::CudaDevice {
        self.context.device_info()
    }

    /// Synchronize GPU operations
    pub fn synchronize(&self) -> Result<(), GpuError> {
        self.context.synchronize()
    }
}

// =============================================================================
// Stub implementation when CUDA is not available
// =============================================================================

#[cfg(not(feature = "cuda"))]
pub struct GpuVocabulary;

#[cfg(not(feature = "cuda"))]
pub struct GpuNativeTokenizer {
    config: GpuNativeConfig,
}

#[cfg(not(feature = "cuda"))]
impl GpuNativeTokenizer {
    pub fn new(
        _vocab: &std::collections::HashMap<String, u32>,
        config: GpuNativeConfig,
    ) -> Result<Self, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn encode_batch(&mut self, _texts: &[&str]) -> Result<Vec<Vec<u32>>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_hash() {
        let hash1 = fnv1a_hash(b"hello");
        let hash2 = fnv1a_hash(b"hello");
        let hash3 = fnv1a_hash(b"world");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_config_default() {
        let config = GpuNativeConfig::default();
        assert_eq!(config.max_seq_bytes, 8192);
        assert_eq!(config.max_tokens_per_seq, 512);
        assert!(config.do_lower_case);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_native_creation() {
        use crate::cuda::{is_cuda_available, CudaContext};
        use std::collections::HashMap;
        use std::sync::Arc;

        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return;
        }

        let ctx = Arc::new(CudaContext::new(0).expect("Context failed"));

        let mut vocab = HashMap::new();
        vocab.insert("[PAD]".to_string(), 0);
        vocab.insert("[UNK]".to_string(), 1);
        vocab.insert("[CLS]".to_string(), 2);
        vocab.insert("[SEP]".to_string(), 3);
        vocab.insert("hello".to_string(), 4);
        vocab.insert("world".to_string(), 5);
        vocab.insert("##ing".to_string(), 6);

        let tokenizer = GpuNativeTokenizer::new(ctx, &vocab, GpuNativeConfig::default());
        assert!(tokenizer.is_ok());
    }
}
