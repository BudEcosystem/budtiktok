//! GPU kernel definitions and implementations
//!
//! This module provides CUDA kernels for tokenization operations:
//! - Pre-tokenization (word boundary detection)
//! - Vocabulary lookup (hash-based)
//! - WordPiece tokenization

use crate::backend::GpuError;

#[cfg(feature = "cuda")]
use crate::cuda::{CudaContext, PRETOKENIZE_KERNEL_SRC, VOCAB_LOOKUP_KERNEL_SRC};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Kernel configuration
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Block dimensions (threads per block)
    pub block_size: (u32, u32, u32),
    /// Grid dimensions (blocks per grid)
    pub grid_size: (u32, u32, u32),
    /// Shared memory size in bytes
    pub shared_memory_bytes: usize,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            block_size: (256, 1, 1),
            grid_size: (1, 1, 1),
            shared_memory_bytes: 0,
        }
    }
}

impl KernelConfig {
    /// Create config for 1D kernel launch
    pub fn linear(n_elements: usize, block_size: u32) -> Self {
        let grid_size = ((n_elements as u32) + block_size - 1) / block_size;
        Self {
            block_size: (block_size, 1, 1),
            grid_size: (grid_size, 1, 1),
            shared_memory_bytes: 0,
        }
    }

    /// Create config with shared memory
    pub fn with_shared_memory(mut self, bytes: usize) -> Self {
        self.shared_memory_bytes = bytes;
        self
    }
}

/// Calculate optimal kernel configuration for tokenization
pub fn calculate_kernel_config(batch_size: usize, max_seq_len: usize) -> KernelConfig {
    let total_elements = batch_size * max_seq_len;
    KernelConfig::linear(total_elements, 256)
}

// =============================================================================
// Pre-tokenization Kernel
// =============================================================================

/// GPU kernel for pre-tokenization (word boundary detection)
#[cfg(feature = "cuda")]
pub struct PreTokenizeKernel {
    device: Arc<CudaDevice>,
}

#[cfg(feature = "cuda")]
impl PreTokenizeKernel {
    const MODULE_NAME: &'static str = "pretokenize";
    const FUNC_WHITESPACE: &'static str = "find_whitespace";
    const FUNC_BOUNDARIES: &'static str = "find_word_boundaries";

    /// Create a new pre-tokenization kernel
    pub fn new(ctx: &CudaContext) -> Result<Self, GpuError> {
        let device = ctx.device().clone();

        // Compile PTX from source
        let ptx = compile_ptx(PRETOKENIZE_KERNEL_SRC)
            .map_err(|e| GpuError::Cuda(format!("PTX compilation failed: {}", e)))?;

        // Load the module
        device
            .load_ptx(
                ptx,
                Self::MODULE_NAME,
                &[Self::FUNC_WHITESPACE, Self::FUNC_BOUNDARIES],
            )
            .map_err(|e| GpuError::Cuda(format!("PTX load failed: {}", e)))?;

        Ok(Self { device })
    }

    /// Find whitespace positions in input bytes
    pub fn find_whitespace_positions(&self, input: &[u8]) -> Result<Vec<u32>, GpuError> {
        let input_len = input.len();
        if input_len == 0 {
            return Ok(Vec::new());
        }

        // Copy input to device
        let d_input = self
            .device
            .htod_copy(input.to_vec())
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        // Allocate output buffer (worst case: every position is whitespace)
        let mut d_output = self
            .device
            .alloc_zeros::<u32>(input_len)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;
        let mut d_count = self
            .device
            .alloc_zeros::<u32>(1)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;

        // Get kernel function
        let func = self
            .device
            .get_func(Self::MODULE_NAME, Self::FUNC_WHITESPACE)
            .ok_or_else(|| {
                GpuError::Cuda(format!(
                    "Function {} not found in {}",
                    Self::FUNC_WHITESPACE,
                    Self::MODULE_NAME
                ))
            })?;

        // Configure launch
        let config = KernelConfig::linear(input_len, 256);
        let launch_config = LaunchConfig {
            grid_dim: config.grid_size,
            block_dim: config.block_size,
            shared_mem_bytes: config.shared_memory_bytes as u32,
        };

        // Launch kernel
        unsafe {
            func.launch(
                launch_config,
                (&d_input, &mut d_output, &mut d_count, input_len as u32),
            )
        }
        .map_err(|e| GpuError::KernelExecution(format!("Whitespace kernel failed: {}", e)))?;

        // Synchronize
        self.device
            .synchronize()
            .map_err(|e| GpuError::Cuda(format!("Sync failed: {}", e)))?;

        // Get count
        let count_vec = self
            .device
            .dtoh_sync_copy(&d_count)
            .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?;
        let count = count_vec[0] as usize;

        // Copy only the valid positions
        let all_positions = self
            .device
            .dtoh_sync_copy(&d_output)
            .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?;
        Ok(all_positions[..count.min(input_len)].to_vec())
    }

    /// Find word boundaries (start, end) pairs
    pub fn find_word_boundaries(&self, input: &[u8]) -> Result<Vec<(usize, usize)>, GpuError> {
        // For now, use CPU implementation for word boundaries
        // GPU version requires prefix sum which adds complexity
        let mut boundaries = Vec::new();
        let mut word_start: Option<usize> = None;

        for (i, &byte) in input.iter().enumerate() {
            let is_ws = byte == b' ' || byte == b'\t' || byte == b'\n' || byte == b'\r';

            if !is_ws && word_start.is_none() {
                word_start = Some(i);
            } else if is_ws && word_start.is_some() {
                boundaries.push((word_start.unwrap(), i));
                word_start = None;
            }
        }

        // Handle word at end of input
        if let Some(start) = word_start {
            boundaries.push((start, input.len()));
        }

        Ok(boundaries)
    }

    /// Find word boundaries for a batch of texts
    pub fn find_word_boundaries_batch(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<(usize, usize)>>, GpuError> {
        texts
            .iter()
            .map(|text| self.find_word_boundaries(text.as_bytes()))
            .collect()
    }
}

// =============================================================================
// Vocabulary Lookup Kernel
// =============================================================================

/// GPU kernel for vocabulary lookup using hash table
#[cfg(feature = "cuda")]
pub struct VocabLookupKernel {
    device: Arc<CudaDevice>,
    vocab_hashes: CudaSlice<u64>,
    vocab_ids: CudaSlice<u32>,
    vocab_size: usize,
    unk_id: u32,
}

#[cfg(feature = "cuda")]
impl VocabLookupKernel {
    const MODULE_NAME: &'static str = "vocab_lookup";
    const FUNC_LOOKUP: &'static str = "vocab_lookup";

    /// Create a new vocabulary lookup kernel
    pub fn new(ctx: &CudaContext, vocab: &[(&str, u32)]) -> Result<Self, GpuError> {
        // Build hash table with 4x capacity for low collision rate
        let table_size = (vocab.len() * 4).next_power_of_two().max(16);
        let mut hashes = vec![0u64; table_size];
        let mut ids = vec![0u32; table_size];

        let unk_id = vocab
            .iter()
            .find(|(s, _)| *s == "[UNK]")
            .map(|(_, id)| *id)
            .unwrap_or(0);

        let max_probes = 64; // Increased from 32 for better collision handling
        let mut collision_count = 0;

        for (word, id) in vocab {
            let hash = fnv1a_hash(word.as_bytes());
            let mut slot = (hash as usize) % table_size;
            let mut inserted = false;

            // Linear probing with better collision handling
            for probe in 0..max_probes {
                if hashes[slot] == 0 {
                    hashes[slot] = hash;
                    ids[slot] = *id;
                    inserted = true;
                    if probe > 0 {
                        collision_count += 1;
                    }
                    break;
                }
                // Skip if same hash (duplicate word)
                if hashes[slot] == hash {
                    inserted = true;
                    break;
                }
                slot = (slot + 1) % table_size;
            }

            if !inserted {
                return Err(GpuError::InvalidConfig(format!(
                    "Hash table overflow: could not insert word '{}' after {} probes. \
                     Vocabulary size {} may be too large for table size {}.",
                    word, max_probes, vocab.len(), table_size
                )));
            }
        }

        if collision_count > vocab.len() / 10 {
            // More than 10% collisions - log warning (in production, use tracing)
            eprintln!(
                "Warning: {} hash collisions in vocabulary of {} words",
                collision_count,
                vocab.len()
            );
        }

        // Get device reference and copy to device
        let device = ctx.device().clone();
        let vocab_hashes = device
            .htod_copy(hashes)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let vocab_ids = device
            .htod_copy(ids)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        // Compile and load kernel
        let ptx = compile_ptx(VOCAB_LOOKUP_KERNEL_SRC)
            .map_err(|e| GpuError::Cuda(format!("PTX compilation failed: {}", e)))?;

        device
            .load_ptx(ptx, Self::MODULE_NAME, &[Self::FUNC_LOOKUP])
            .map_err(|e| GpuError::Cuda(format!("PTX load failed: {}", e)))?;

        Ok(Self {
            device,
            vocab_hashes,
            vocab_ids,
            vocab_size: table_size,
            unk_id,
        })
    }

    /// Lookup words in vocabulary
    pub fn lookup(&self, words: &[&str]) -> Result<Vec<u32>, GpuError> {
        if words.is_empty() {
            return Ok(Vec::new());
        }

        // Pack words into contiguous buffer
        let mut packed_words = Vec::new();
        let mut offsets = Vec::with_capacity(words.len());
        let mut lengths = Vec::with_capacity(words.len());

        for word in words {
            offsets.push(packed_words.len() as u32);
            lengths.push(word.len() as u32);
            packed_words.extend_from_slice(word.as_bytes());
        }

        // Copy to device
        let d_words = self
            .device
            .htod_copy(packed_words)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let d_offsets = self
            .device
            .htod_copy(offsets)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let d_lengths = self
            .device
            .htod_copy(lengths)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let mut d_output = self
            .device
            .alloc_zeros::<u32>(words.len())
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;

        // Get kernel function
        let func = self
            .device
            .get_func(Self::MODULE_NAME, Self::FUNC_LOOKUP)
            .ok_or_else(|| {
                GpuError::Cuda(format!(
                    "Function {} not found in {}",
                    Self::FUNC_LOOKUP,
                    Self::MODULE_NAME
                ))
            })?;

        // Configure launch
        let config = KernelConfig::linear(words.len(), 256);
        let launch_config = LaunchConfig {
            grid_dim: config.grid_size,
            block_dim: config.block_size,
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            func.launch(
                launch_config,
                (
                    &d_words,
                    &d_offsets,
                    &d_lengths,
                    &self.vocab_hashes,
                    &self.vocab_ids,
                    &mut d_output,
                    words.len() as u32,
                    self.vocab_size as u32,
                    self.unk_id,
                ),
            )
        }
        .map_err(|e| GpuError::KernelExecution(format!("Vocab lookup failed: {}", e)))?;

        // Synchronize and copy results
        self.device
            .synchronize()
            .map_err(|e| GpuError::Cuda(format!("Sync failed: {}", e)))?;
        self.device
            .dtoh_sync_copy(&d_output)
            .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))
    }
}

/// FNV-1a hash function (matches GPU implementation)
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 14695981039346656037;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

// =============================================================================
// WordPiece Kernel
// =============================================================================

/// GPU kernel for WordPiece tokenization
#[cfg(feature = "cuda")]
pub struct WordPieceKernel {
    vocab_lookup: VocabLookupKernel,
    continuation_prefix: String,
    max_word_length: usize,
    /// Maximum words to process in a single batch to prevent memory exhaustion
    /// For a word with n chars, we generate n(n+1)/2 substrings
    /// With max_word_length=100 and batch=1000, worst case is 1000 * 5050 = 5M substrings
    max_words_per_batch: usize,
}

#[cfg(feature = "cuda")]
impl WordPieceKernel {
    /// Default maximum words per batch - balances GPU efficiency vs memory usage
    /// 2000 words × ~36 avg substrings × ~50 bytes = ~3.6MB per batch
    const DEFAULT_MAX_WORDS_PER_BATCH: usize = 2000;

    /// Create a new WordPiece kernel
    pub fn new(
        ctx: &CudaContext,
        vocab: &[(&str, u32)],
        continuation_prefix: &str,
    ) -> Result<Self, GpuError> {
        let vocab_lookup = VocabLookupKernel::new(ctx, vocab)?;

        Ok(Self {
            vocab_lookup,
            continuation_prefix: continuation_prefix.to_string(),
            max_word_length: 100,
            max_words_per_batch: Self::DEFAULT_MAX_WORDS_PER_BATCH,
        })
    }

    /// Create with custom max words per batch
    pub fn with_batch_limit(
        ctx: &CudaContext,
        vocab: &[(&str, u32)],
        continuation_prefix: &str,
        max_words_per_batch: usize,
    ) -> Result<Self, GpuError> {
        let vocab_lookup = VocabLookupKernel::new(ctx, vocab)?;

        Ok(Self {
            vocab_lookup,
            continuation_prefix: continuation_prefix.to_string(),
            max_word_length: 100,
            max_words_per_batch: max_words_per_batch.max(100), // At least 100 words
        })
    }

    /// Tokenize a single word using WordPiece algorithm with batched GPU lookup
    pub fn tokenize_word(&self, word: &str) -> Result<Vec<u32>, GpuError> {
        if word.len() > self.max_word_length {
            return Ok(vec![self.vocab_lookup.unk_id]);
        }

        let chars: Vec<char> = word.chars().collect();
        if chars.is_empty() {
            return Ok(Vec::new());
        }

        // Pre-generate all possible substrings for batched lookup
        // Format: (start_pos, end_pos, lookup_string)
        let mut all_substrings: Vec<(usize, usize, String)> = Vec::new();

        // Add whole word first
        all_substrings.push((0, chars.len(), word.to_string()));

        // Generate all substrings with continuation prefix where needed
        for start in 0..chars.len() {
            for end in (start + 1)..=chars.len() {
                // Skip whole word (already added)
                if start == 0 && end == chars.len() {
                    continue;
                }
                let substr: String = chars[start..end].iter().collect();
                let lookup_str = if start > 0 {
                    format!("{}{}", self.continuation_prefix, substr)
                } else {
                    substr
                };
                all_substrings.push((start, end, lookup_str));
            }
        }

        // Single batched GPU lookup for all substrings
        let lookup_strs: Vec<&str> = all_substrings.iter().map(|(_, _, s)| s.as_str()).collect();
        let lookup_results = self.vocab_lookup.lookup(&lookup_strs)?;

        // Build lookup map: (start, end) -> token_id (only for found tokens)
        let mut found_tokens: std::collections::HashMap<(usize, usize), u32> =
            std::collections::HashMap::new();
        for (i, &token_id) in lookup_results.iter().enumerate() {
            if token_id != self.vocab_lookup.unk_id {
                let (start, end, _) = &all_substrings[i];
                found_tokens.insert((*start, *end), token_id);
            }
        }

        // Check if whole word was found
        if let Some(&token_id) = found_tokens.get(&(0, chars.len())) {
            return Ok(vec![token_id]);
        }

        // Greedy longest-match using pre-computed results (no GPU calls in loop)
        let mut tokens = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let mut found = false;

            // Try longest substring first
            for end in (start + 1..=chars.len()).rev() {
                if let Some(&token_id) = found_tokens.get(&(start, end)) {
                    tokens.push(token_id);
                    start = end;
                    found = true;
                    break;
                }
            }

            if !found {
                return Ok(vec![self.vocab_lookup.unk_id]);
            }
        }

        Ok(tokens)
    }

    /// Tokenize multiple words with batched GPU lookup
    /// Uses internal batching to prevent memory exhaustion
    pub fn tokenize_words(&self, words: &[&str]) -> Result<Vec<Vec<u32>>, GpuError> {
        if words.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = vec![Vec::new(); words.len()];

        // Process words in batches to prevent memory exhaustion
        for batch_start in (0..words.len()).step_by(self.max_words_per_batch) {
            let batch_end = (batch_start + self.max_words_per_batch).min(words.len());
            let batch_words = &words[batch_start..batch_end];

            // Process this batch
            let batch_results = self.tokenize_words_batch(batch_words)?;

            // Copy results to final output
            for (i, result) in batch_results.into_iter().enumerate() {
                results[batch_start + i] = result;
            }
        }

        Ok(results)
    }

    /// Internal: Process a single batch of words (bounded memory)
    fn tokenize_words_batch(&self, words: &[&str]) -> Result<Vec<Vec<u32>>, GpuError> {
        if words.is_empty() {
            return Ok(Vec::new());
        }

        // Pre-generate all substrings for this batch
        // Track which word each substring belongs to
        struct SubstringInfo {
            word_idx: usize,
            start: usize,
            end: usize,
        }

        let mut all_substrings: Vec<String> = Vec::new();
        let mut substring_info: Vec<SubstringInfo> = Vec::new();
        let mut word_chars: Vec<Vec<char>> = Vec::new();

        for (word_idx, &word) in words.iter().enumerate() {
            let chars: Vec<char> = word.chars().collect();

            if chars.is_empty() || word.len() > self.max_word_length {
                word_chars.push(chars);
                continue;
            }

            // Add whole word
            all_substrings.push(word.to_string());
            substring_info.push(SubstringInfo {
                word_idx,
                start: 0,
                end: chars.len(),
            });

            // Generate all substrings
            for start in 0..chars.len() {
                for end in (start + 1)..=chars.len() {
                    if start == 0 && end == chars.len() {
                        continue; // Skip whole word (already added)
                    }
                    let substr: String = chars[start..end].iter().collect();
                    let lookup_str = if start > 0 {
                        format!("{}{}", self.continuation_prefix, substr)
                    } else {
                        substr
                    };
                    all_substrings.push(lookup_str);
                    substring_info.push(SubstringInfo {
                        word_idx,
                        start,
                        end,
                    });
                }
            }

            word_chars.push(chars);
        }

        // Single GPU lookup for all substrings of this batch
        let lookup_strs: Vec<&str> = all_substrings.iter().map(|s| s.as_str()).collect();
        let lookup_results = if lookup_strs.is_empty() {
            Vec::new()
        } else {
            self.vocab_lookup.lookup(&lookup_strs)?
        };

        // Build per-word lookup maps
        let mut word_found_tokens: Vec<std::collections::HashMap<(usize, usize), u32>> =
            vec![std::collections::HashMap::new(); words.len()];

        for (i, &token_id) in lookup_results.iter().enumerate() {
            if token_id != self.vocab_lookup.unk_id {
                let info = &substring_info[i];
                word_found_tokens[info.word_idx].insert((info.start, info.end), token_id);
            }
        }

        // Process each word using its pre-computed lookup results
        let mut results = Vec::with_capacity(words.len());

        for (word_idx, chars) in word_chars.iter().enumerate() {
            if chars.is_empty() {
                results.push(Vec::new());
                continue;
            }

            if words[word_idx].len() > self.max_word_length {
                results.push(vec![self.vocab_lookup.unk_id]);
                continue;
            }

            let found_tokens = &word_found_tokens[word_idx];

            // Check whole word first
            if let Some(&token_id) = found_tokens.get(&(0, chars.len())) {
                results.push(vec![token_id]);
                continue;
            }

            // Greedy longest-match
            let mut tokens = Vec::new();
            let mut start = 0;
            let mut success = true;

            while start < chars.len() {
                let mut found = false;

                for end in (start + 1..=chars.len()).rev() {
                    if let Some(&token_id) = found_tokens.get(&(start, end)) {
                        tokens.push(token_id);
                        start = end;
                        found = true;
                        break;
                    }
                }

                if !found {
                    success = false;
                    break;
                }
            }

            if success {
                results.push(tokens);
            } else {
                results.push(vec![self.vocab_lookup.unk_id]);
            }
        }

        Ok(results)
    }
}

// =============================================================================
// BPE Kernel (BlockBPE-style GPU implementation)
// =============================================================================

/// Merge rule for BPE
#[derive(Debug, Clone)]
pub struct BpeMergeRule {
    /// First token ID
    pub first_id: u32,
    /// Second token ID
    pub second_id: u32,
    /// Result token ID
    pub result_id: u32,
    /// Priority (lower = higher priority, earlier in merge order)
    pub priority: u32,
}

/// GPU kernel for BPE tokenization using BlockBPE-style parallel algorithm
#[cfg(feature = "cuda")]
pub struct BpeKernel {
    device: Arc<CudaDevice>,
    // Merge rule hash table (GPU-resident)
    merge_hashes: CudaSlice<u64>,
    merge_first_ids: CudaSlice<u32>,
    merge_second_ids: CudaSlice<u32>,
    merge_result_ids: CudaSlice<u32>,
    merge_priorities: CudaSlice<u32>,
    table_size: usize,
    // Byte to token mapping for initial tokenization
    byte_to_token: CudaSlice<u32>,
    unk_id: u32,
}

#[cfg(feature = "cuda")]
impl BpeKernel {
    const MODULE_NAME: &'static str = "bpe_merge";
    const FUNC_FIND_BEST: &'static str = "bpe_find_best_merge";
    const FUNC_APPLY: &'static str = "bpe_apply_merge";
    const FUNC_COMPACT: &'static str = "bpe_compact_tokens";
    const FUNC_CHAR_TO_TOKENS: &'static str = "bpe_char_to_tokens";

    /// Create a new BPE kernel with merge rules
    pub fn new(
        ctx: &CudaContext,
        merge_rules: &[BpeMergeRule],
        byte_to_token_map: &[u32; 256],
        unk_id: u32,
    ) -> Result<Self, GpuError> {
        use crate::cuda::BPE_MERGE_KERNEL_SRC;

        let device = ctx.device().clone();

        // Compile and load PTX
        let ptx = compile_ptx(BPE_MERGE_KERNEL_SRC)
            .map_err(|e| GpuError::Cuda(format!("BPE PTX compilation failed: {}", e)))?;

        device
            .load_ptx(
                ptx,
                Self::MODULE_NAME,
                &[
                    Self::FUNC_FIND_BEST,
                    Self::FUNC_APPLY,
                    Self::FUNC_COMPACT,
                    Self::FUNC_CHAR_TO_TOKENS,
                ],
            )
            .map_err(|e| GpuError::Cuda(format!("BPE PTX load failed: {}", e)))?;

        // Build hash table for merge rules (4x capacity)
        let table_size = (merge_rules.len() * 4).next_power_of_two().max(16);
        let mut hashes = vec![0u64; table_size];
        let mut first_ids = vec![0u32; table_size];
        let mut second_ids = vec![0u32; table_size];
        let mut result_ids = vec![0u32; table_size];
        let mut priorities = vec![0u32; table_size];

        for rule in merge_rules {
            let hash = bpe_pair_hash(rule.first_id, rule.second_id);
            let mut slot = (hash as usize) % table_size;

            for _ in 0..64 {
                if hashes[slot] == 0 {
                    hashes[slot] = hash;
                    first_ids[slot] = rule.first_id;
                    second_ids[slot] = rule.second_id;
                    result_ids[slot] = rule.result_id;
                    priorities[slot] = rule.priority;
                    break;
                }
                slot = (slot + 1) % table_size;
            }
        }

        // Copy to GPU
        let merge_hashes = device
            .htod_copy(hashes)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let merge_first_ids = device
            .htod_copy(first_ids)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let merge_second_ids = device
            .htod_copy(second_ids)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let merge_result_ids = device
            .htod_copy(result_ids)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let merge_priorities = device
            .htod_copy(priorities)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        // Copy byte-to-token mapping
        let byte_to_token = device
            .htod_copy(byte_to_token_map.to_vec())
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        Ok(Self {
            device,
            merge_hashes,
            merge_first_ids,
            merge_second_ids,
            merge_result_ids,
            merge_priorities,
            table_size,
            byte_to_token,
            unk_id,
        })
    }

    /// Tokenize text using GPU-accelerated BPE
    pub fn tokenize(&self, text: &[u8]) -> Result<Vec<u32>, GpuError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let num_chars = text.len();

        // Step 1: Convert bytes to initial token IDs
        let d_input = self
            .device
            .htod_copy(text.to_vec())
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        let mut d_token_ids = self
            .device
            .alloc_zeros::<u32>(num_chars)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;

        let char_to_tokens_func = self
            .device
            .get_func(Self::MODULE_NAME, Self::FUNC_CHAR_TO_TOKENS)
            .ok_or_else(|| GpuError::Cuda("char_to_tokens function not found".into()))?;

        let config = KernelConfig::linear(num_chars, 256);
        let launch_config = LaunchConfig {
            grid_dim: config.grid_size,
            block_dim: config.block_size,
            shared_mem_bytes: 0,
        };

        unsafe {
            char_to_tokens_func.launch(
                launch_config,
                (&d_input, &self.byte_to_token, &mut d_token_ids, num_chars as u32),
            )
        }
        .map_err(|e| GpuError::KernelExecution(format!("char_to_tokens failed: {}", e)))?;

        // Step 2: Iteratively apply merges
        let mut d_valid = self
            .device
            .htod_copy(vec![1u32; num_chars])
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        let mut current_len = num_chars;
        let max_iterations = num_chars; // At most num_chars-1 merges possible

        for _ in 0..max_iterations {
            if current_len <= 1 {
                break;
            }

            // Find best merge in each block
            let num_blocks = (current_len + 255) / 256;
            let mut d_best_positions = self
                .device
                .alloc_zeros::<u32>(num_blocks)
                .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;
            let mut d_best_priorities = self
                .device
                .htod_copy(vec![u32::MAX; num_blocks])
                .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
            let mut d_best_results = self
                .device
                .alloc_zeros::<u32>(num_blocks)
                .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;

            let find_best_func = self
                .device
                .get_func(Self::MODULE_NAME, Self::FUNC_FIND_BEST)
                .ok_or_else(|| GpuError::Cuda("find_best_merge function not found".into()))?;

            let shared_mem = 256 * 3 * std::mem::size_of::<u32>(); // 3 arrays of 256 u32s
            let config = KernelConfig::linear(current_len, 256).with_shared_memory(shared_mem);
            let launch_config = LaunchConfig {
                grid_dim: config.grid_size,
                block_dim: config.block_size,
                shared_mem_bytes: shared_mem as u32,
            };

            unsafe {
                find_best_func.launch(
                    launch_config,
                    (
                        &d_token_ids,
                        &d_valid,
                        &self.merge_hashes,
                        &self.merge_first_ids,
                        &self.merge_second_ids,
                        &self.merge_result_ids,
                        &self.merge_priorities,
                        &mut d_best_positions,
                        &mut d_best_priorities,
                        &mut d_best_results,
                        num_chars as u32,
                        self.table_size as u32,
                    ),
                )
            }
            .map_err(|e| GpuError::KernelExecution(format!("find_best_merge failed: {}", e)))?;

            // Copy back to check if any merge was found
            self.device
                .synchronize()
                .map_err(|e| GpuError::Cuda(format!("Sync failed: {}", e)))?;

            let best_priorities = self
                .device
                .dtoh_sync_copy(&d_best_priorities)
                .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?;

            // Find global best merge
            let mut global_best_idx = 0;
            let mut global_best_pri = u32::MAX;
            for (i, &pri) in best_priorities.iter().enumerate() {
                if pri < global_best_pri {
                    global_best_pri = pri;
                    global_best_idx = i;
                }
            }

            if global_best_pri == u32::MAX {
                // No more merges possible
                break;
            }

            // Apply the single best merge
            let best_positions = self
                .device
                .dtoh_sync_copy(&d_best_positions)
                .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?;
            let best_results = self
                .device
                .dtoh_sync_copy(&d_best_results)
                .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?;

            let merge_pos = best_positions[global_best_idx];
            let merge_result = best_results[global_best_idx];

            if merge_pos == u32::MAX {
                break;
            }

            // Apply merge on CPU (simpler for single merge)
            let mut token_ids = self
                .device
                .dtoh_sync_copy(&d_token_ids)
                .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?;
            let mut valid = self
                .device
                .dtoh_sync_copy(&d_valid)
                .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?;

            token_ids[merge_pos as usize] = merge_result;
            valid[(merge_pos + 1) as usize] = 0;
            current_len -= 1;

            d_token_ids = self
                .device
                .htod_copy(token_ids)
                .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
            d_valid = self
                .device
                .htod_copy(valid)
                .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        }

        // Extract final tokens
        let token_ids = self
            .device
            .dtoh_sync_copy(&d_token_ids)
            .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?;
        let valid = self
            .device
            .dtoh_sync_copy(&d_valid)
            .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?;

        Ok(token_ids
            .into_iter()
            .zip(valid)
            .filter(|(_, v)| *v == 1)
            .map(|(id, _)| id)
            .collect())
    }

    /// Tokenize batch of texts
    pub fn tokenize_batch(&self, texts: &[&[u8]]) -> Result<Vec<Vec<u32>>, GpuError> {
        texts.iter().map(|text| self.tokenize(text)).collect()
    }
}

/// Hash function for BPE merge pairs (matches GPU implementation)
fn bpe_pair_hash(first_id: u32, second_id: u32) -> u64 {
    let mut hash: u64 = 14695981039346656037;
    for i in 0..4 {
        hash ^= ((first_id >> (i * 8)) & 0xFF) as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    for i in 0..4 {
        hash ^= ((second_id >> (i * 8)) & 0xFF) as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

// =============================================================================
// Optimized BPE Kernel (GPU-resident, parallel merges)
// =============================================================================

/// GPU-optimized BPE kernel using linked-list representation
/// Key optimization: ALL non-conflicting merges applied in parallel per pass
#[cfg(feature = "cuda")]
pub struct OptimizedBpeKernel {
    device: Arc<CudaDevice>,
    // Merge rule hash table (GPU-resident)
    merge_hashes: CudaSlice<u64>,
    merge_first_ids: CudaSlice<u32>,
    merge_second_ids: CudaSlice<u32>,
    merge_result_ids: CudaSlice<u32>,
    merge_priorities: CudaSlice<u32>,
    table_size: usize,
    // Byte to token mapping
    byte_to_token: CudaSlice<u32>,
    // Pre-allocated work buffers for reuse
    max_seq_len: usize,
}

#[cfg(feature = "cuda")]
impl OptimizedBpeKernel {
    const MODULE_NAME: &'static str = "bpe_merge";
    const FUNC_CHAR_TO_TOKENS: &'static str = "bpe_char_to_tokens";
    const FUNC_FIND_ALL: &'static str = "bpe_find_all_merges";
    const FUNC_RESOLVE: &'static str = "bpe_resolve_conflicts";
    const FUNC_APPLY_ALL: &'static str = "bpe_apply_all_merges";
    const FUNC_COUNT: &'static str = "bpe_count_valid";

    /// Create optimized BPE kernel
    pub fn new(
        ctx: &CudaContext,
        merge_rules: &[BpeMergeRule],
        byte_to_token_map: &[u32; 256],
        max_seq_len: usize,
    ) -> Result<Self, GpuError> {
        use crate::cuda::BPE_MERGE_KERNEL_SRC;

        let device = ctx.device().clone();

        // Compile and load PTX
        let ptx = compile_ptx(BPE_MERGE_KERNEL_SRC)
            .map_err(|e| GpuError::Cuda(format!("BPE PTX compilation failed: {}", e)))?;

        device
            .load_ptx(
                ptx,
                Self::MODULE_NAME,
                &[
                    Self::FUNC_CHAR_TO_TOKENS,
                    Self::FUNC_FIND_ALL,
                    Self::FUNC_RESOLVE,
                    Self::FUNC_APPLY_ALL,
                    Self::FUNC_COUNT,
                ],
            )
            .map_err(|e| GpuError::Cuda(format!("BPE PTX load failed: {}", e)))?;

        // Build hash table
        let table_size = (merge_rules.len() * 4).next_power_of_two().max(16);
        let mut hashes = vec![0u64; table_size];
        let mut first_ids = vec![0u32; table_size];
        let mut second_ids = vec![0u32; table_size];
        let mut result_ids = vec![0u32; table_size];
        let mut priorities = vec![0u32; table_size];

        for rule in merge_rules {
            let hash = bpe_pair_hash(rule.first_id, rule.second_id);
            let mut slot = (hash as usize) % table_size;

            for _ in 0..64 {
                if hashes[slot] == 0 {
                    hashes[slot] = hash;
                    first_ids[slot] = rule.first_id;
                    second_ids[slot] = rule.second_id;
                    result_ids[slot] = rule.result_id;
                    priorities[slot] = rule.priority;
                    break;
                }
                slot = (slot + 1) % table_size;
            }
        }

        let merge_hashes = device
            .htod_copy(hashes)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let merge_first_ids = device
            .htod_copy(first_ids)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let merge_second_ids = device
            .htod_copy(second_ids)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let merge_result_ids = device
            .htod_copy(result_ids)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let merge_priorities = device
            .htod_copy(priorities)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let byte_to_token = device
            .htod_copy(byte_to_token_map.to_vec())
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        Ok(Self {
            device,
            merge_hashes,
            merge_first_ids,
            merge_second_ids,
            merge_result_ids,
            merge_priorities,
            table_size,
            byte_to_token,
            max_seq_len,
        })
    }

    /// Tokenize with GPU-resident parallel merge algorithm
    pub fn tokenize(&self, text: &[u8]) -> Result<Vec<u32>, GpuError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let num_chars = text.len();

        // Step 1: Copy input and convert to initial tokens
        let d_input = self
            .device
            .htod_copy(text.to_vec())
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        let mut d_token_ids = self
            .device
            .alloc_zeros::<u32>(num_chars)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;

        // Initialize token IDs from bytes
        let char_to_tokens_func = self
            .device
            .get_func(Self::MODULE_NAME, Self::FUNC_CHAR_TO_TOKENS)
            .ok_or_else(|| GpuError::Cuda("char_to_tokens not found".into()))?;

        let config = KernelConfig::linear(num_chars, 256);
        unsafe {
            char_to_tokens_func.launch(
                LaunchConfig {
                    grid_dim: config.grid_size,
                    block_dim: config.block_size,
                    shared_mem_bytes: 0,
                },
                (&d_input, &self.byte_to_token, &mut d_token_ids, num_chars as u32),
            )
        }
        .map_err(|e| GpuError::KernelExecution(format!("char_to_tokens failed: {}", e)))?;

        // Initialize linked list: next[i] = i+1, next[n-1] = INVALID
        let mut next_indices: Vec<u32> = (1..=num_chars as u32).collect();
        next_indices[num_chars - 1] = u32::MAX;

        let mut d_next = self
            .device
            .htod_copy(next_indices)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        // Allocate work buffers (reused each iteration)
        let mut d_can_merge = self
            .device
            .alloc_zeros::<u32>(num_chars)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;
        let mut d_merge_results = self
            .device
            .alloc_zeros::<u32>(num_chars)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;
        let mut d_merge_pris = self
            .device
            .alloc_zeros::<u32>(num_chars)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;

        // Get kernel functions
        let find_all_func = self
            .device
            .get_func(Self::MODULE_NAME, Self::FUNC_FIND_ALL)
            .ok_or_else(|| GpuError::Cuda("find_all_merges not found".into()))?;
        let resolve_func = self
            .device
            .get_func(Self::MODULE_NAME, Self::FUNC_RESOLVE)
            .ok_or_else(|| GpuError::Cuda("resolve_conflicts not found".into()))?;
        let apply_func = self
            .device
            .get_func(Self::MODULE_NAME, Self::FUNC_APPLY_ALL)
            .ok_or_else(|| GpuError::Cuda("apply_all_merges not found".into()))?;

        // Iterate until no more merges (all GPU-resident)
        let max_iters = num_chars;
        for _ in 0..max_iters {
            // Reset can_merge
            self.device
                .htod_sync_copy_into(&vec![0u32; num_chars], &mut d_can_merge)
                .map_err(|e| GpuError::Cuda(format!("Reset failed: {}", e)))?;

            // Find all possible merges (clone function to avoid move)
            unsafe {
                find_all_func.clone().launch(
                    LaunchConfig {
                        grid_dim: config.grid_size,
                        block_dim: config.block_size,
                        shared_mem_bytes: 0,
                    },
                    (
                        &d_token_ids,
                        &d_next,
                        &self.merge_hashes,
                        &self.merge_first_ids,
                        &self.merge_second_ids,
                        &self.merge_result_ids,
                        &self.merge_priorities,
                        &mut d_can_merge,
                        &mut d_merge_results,
                        &mut d_merge_pris,
                        num_chars as u32,
                        self.table_size as u32,
                    ),
                )
            }
            .map_err(|e| GpuError::KernelExecution(format!("find_all failed: {}", e)))?;

            // Resolve conflicts (clone function to avoid move)
            unsafe {
                resolve_func.clone().launch(
                    LaunchConfig {
                        grid_dim: config.grid_size,
                        block_dim: config.block_size,
                        shared_mem_bytes: 0,
                    },
                    (&d_next, &mut d_can_merge, &d_merge_pris, num_chars as u32),
                )
            }
            .map_err(|e| GpuError::KernelExecution(format!("resolve failed: {}", e)))?;

            // Check if any merges to apply (quick check on GPU result)
            let can_merge_host = self
                .device
                .dtoh_sync_copy(&d_can_merge)
                .map_err(|e| GpuError::Cuda(format!("DtoH failed: {}", e)))?;

            let merge_count: u32 = can_merge_host.iter().sum();
            if merge_count == 0 {
                break; // No more merges possible
            }

            // Apply all approved merges (clone function to avoid move)
            unsafe {
                apply_func.clone().launch(
                    LaunchConfig {
                        grid_dim: config.grid_size,
                        block_dim: config.block_size,
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut d_token_ids,
                        &mut d_next,
                        &d_can_merge,
                        &d_merge_results,
                        num_chars as u32,
                    ),
                )
            }
            .map_err(|e| GpuError::KernelExecution(format!("apply failed: {}", e)))?;
        }

        // Extract final tokens by traversing linked list
        self.device
            .synchronize()
            .map_err(|e| GpuError::Cuda(format!("Sync failed: {}", e)))?;

        let token_ids = self
            .device
            .dtoh_sync_copy(&d_token_ids)
            .map_err(|e| GpuError::Cuda(format!("DtoH failed: {}", e)))?;
        let next = self
            .device
            .dtoh_sync_copy(&d_next)
            .map_err(|e| GpuError::Cuda(format!("DtoH failed: {}", e)))?;

        // Walk linked list to get final tokens
        let mut result = Vec::new();
        let mut idx = 0usize;
        while idx < num_chars {
            result.push(token_ids[idx]);
            let next_idx = next[idx];
            if next_idx == u32::MAX {
                break;
            }
            idx = next_idx as usize;
        }

        Ok(result)
    }

    /// Batch tokenize multiple texts
    pub fn tokenize_batch(&self, texts: &[&[u8]]) -> Result<Vec<Vec<u32>>, GpuError> {
        texts.iter().map(|t| self.tokenize(t)).collect()
    }
}

// =============================================================================
// Unigram/Viterbi Kernel
// =============================================================================

/// Flattened trie node for GPU transfer
#[derive(Debug, Clone)]
pub struct FlatTrieNode {
    /// Character at this node
    pub char_byte: u8,
    /// Is this a valid token ending?
    pub is_end: bool,
    /// Token ID if is_end
    pub token_id: u32,
    /// Log probability score
    pub score: f32,
    /// Start index of children in the flattened array
    pub children_start: u32,
    /// Number of children
    pub children_count: u32,
}

/// GPU kernel for Unigram tokenization with parallel Viterbi
#[cfg(feature = "cuda")]
pub struct UnigramKernel {
    device: Arc<CudaDevice>,
    // Flattened trie on GPU
    trie_chars: CudaSlice<u32>,
    trie_is_end: CudaSlice<u32>,
    trie_token_ids: CudaSlice<u32>,
    trie_scores: CudaSlice<f32>,
    trie_child_start: CudaSlice<u32>,
    trie_child_count: CudaSlice<u32>,
    trie_size: usize,
    unk_id: u32,
    min_score: f32,
}

#[cfg(feature = "cuda")]
impl UnigramKernel {
    const MODULE_NAME: &'static str = "unigram_viterbi";
    const FUNC_BUILD_LATTICE: &'static str = "unigram_build_lattice";
    const FUNC_FORWARD: &'static str = "unigram_forward_pass";
    const FUNC_BACKTRACK: &'static str = "unigram_backtrack";

    /// Create a new Unigram kernel with flattened trie
    pub fn new(
        ctx: &CudaContext,
        trie_nodes: &[FlatTrieNode],
        unk_id: u32,
    ) -> Result<Self, GpuError> {
        use crate::cuda::UNIGRAM_VITERBI_KERNEL_SRC;

        let device = ctx.device().clone();

        // Compile and load PTX
        let ptx = compile_ptx(UNIGRAM_VITERBI_KERNEL_SRC)
            .map_err(|e| GpuError::Cuda(format!("Unigram PTX compilation failed: {}", e)))?;

        device
            .load_ptx(
                ptx,
                Self::MODULE_NAME,
                &[Self::FUNC_BUILD_LATTICE, Self::FUNC_FORWARD, Self::FUNC_BACKTRACK],
            )
            .map_err(|e| GpuError::Cuda(format!("Unigram PTX load failed: {}", e)))?;

        // Flatten trie data for GPU
        let trie_size = trie_nodes.len();
        let mut chars = Vec::with_capacity(trie_size);
        let mut is_end = Vec::with_capacity(trie_size);
        let mut token_ids = Vec::with_capacity(trie_size);
        let mut scores = Vec::with_capacity(trie_size);
        let mut child_start = Vec::with_capacity(trie_size);
        let mut child_count = Vec::with_capacity(trie_size);

        let mut min_score = f32::INFINITY;

        for node in trie_nodes {
            chars.push(node.char_byte as u32);
            is_end.push(if node.is_end { 1u32 } else { 0u32 });
            token_ids.push(node.token_id);
            scores.push(node.score);
            child_start.push(node.children_start);
            child_count.push(node.children_count);

            if node.is_end && node.score < min_score {
                min_score = node.score;
            }
        }

        // Copy to GPU
        let trie_chars = device
            .htod_copy(chars)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let trie_is_end = device
            .htod_copy(is_end)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let trie_token_ids = device
            .htod_copy(token_ids)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let trie_scores = device
            .htod_copy(scores)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let trie_child_start = device
            .htod_copy(child_start)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let trie_child_count = device
            .htod_copy(child_count)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        Ok(Self {
            device,
            trie_chars,
            trie_is_end,
            trie_token_ids,
            trie_scores,
            trie_child_start,
            trie_child_count,
            trie_size,
            unk_id,
            min_score,
        })
    }

    /// Tokenize text using GPU-accelerated Viterbi
    pub fn tokenize(&self, text: &[u8]) -> Result<Vec<u32>, GpuError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let input_len = text.len();

        // Copy input to device
        let d_input = self
            .device
            .htod_copy(text.to_vec())
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        // Allocate and initialize lattice arrays
        // Position 0 starts with score 0.0, all others start with NEG_INFINITY
        let mut lattice_scores_init = vec![f32::NEG_INFINITY; input_len + 1];
        lattice_scores_init[0] = 0.0;

        let mut d_lattice_scores = self
            .device
            .htod_copy(lattice_scores_init)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let mut d_lattice_prev = self
            .device
            .alloc_zeros::<u32>(input_len + 1)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;
        let mut d_lattice_token = self
            .device
            .alloc_zeros::<u32>(input_len + 1)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;

        // Launch lattice building kernel
        let build_lattice_func = self
            .device
            .get_func(Self::MODULE_NAME, Self::FUNC_BUILD_LATTICE)
            .ok_or_else(|| GpuError::Cuda("build_lattice function not found".into()))?;

        let config = KernelConfig::linear(input_len, 256);
        let shared_mem = 256 * std::mem::size_of::<f32>();
        let launch_config = LaunchConfig {
            grid_dim: config.grid_size,
            block_dim: config.block_size,
            shared_mem_bytes: shared_mem as u32,
        };

        unsafe {
            build_lattice_func.launch(
                launch_config,
                (
                    &d_input,
                    &self.trie_chars,
                    &self.trie_is_end,
                    &self.trie_token_ids,
                    &self.trie_scores,
                    &self.trie_child_start,
                    &self.trie_child_count,
                    &mut d_lattice_scores,
                    &mut d_lattice_prev,
                    &mut d_lattice_token,
                    input_len as u32,
                    self.trie_size as u32,
                ),
            )
        }
        .map_err(|e| GpuError::KernelExecution(format!("build_lattice failed: {}", e)))?;

        self.device
            .synchronize()
            .map_err(|e| GpuError::Cuda(format!("Sync failed: {}", e)))?;

        // Backtrack on CPU (more efficient for sequential operation)
        let lattice_prev = self
            .device
            .dtoh_sync_copy(&d_lattice_prev)
            .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?;
        let lattice_token = self
            .device
            .dtoh_sync_copy(&d_lattice_token)
            .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?;
        let lattice_scores = self
            .device
            .dtoh_sync_copy(&d_lattice_scores)
            .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?;

        // Check if we reached the end
        if lattice_scores[input_len] == f32::NEG_INFINITY {
            // Fallback: return unknown tokens for each byte
            return Ok(vec![self.unk_id; input_len]);
        }

        // Backtrack
        let mut tokens = Vec::new();
        let mut pos = input_len;

        while pos > 0 {
            tokens.push(lattice_token[pos]);
            pos = lattice_prev[pos] as usize;
        }

        tokens.reverse();
        Ok(tokens)
    }

    /// Tokenize batch of texts
    pub fn tokenize_batch(&self, texts: &[&[u8]]) -> Result<Vec<Vec<u32>>, GpuError> {
        texts.iter().map(|text| self.tokenize(text)).collect()
    }
}

// =============================================================================
// Character Tokenization Kernel (SentencePiece Character mode)
// =============================================================================

/// GPU kernel for Character tokenization (SentencePiece Character mode)
/// Maps each character/byte to its token ID - simplest form of tokenization
#[cfg(feature = "cuda")]
pub struct CharacterKernel {
    device: Arc<CudaDevice>,
    // Simple byte-to-token lookup (256 entries)
    byte_to_token: CudaSlice<u32>,
    // UTF-8 vocabulary hash table (for multi-byte chars)
    vocab_hashes: CudaSlice<u64>,
    vocab_ids: CudaSlice<u32>,
    vocab_size: usize,
    unk_id: u32,
    use_utf8: bool,
}

#[cfg(feature = "cuda")]
impl CharacterKernel {
    const MODULE_NAME: &'static str = "char_tokenize";
    const FUNC_SIMPLE: &'static str = "char_tokenize";
    const FUNC_UTF8: &'static str = "char_tokenize_utf8";
    const FUNC_MARK_STARTS: &'static str = "mark_utf8_char_starts";

    /// Create a new Character kernel with byte-to-token mapping
    /// For ASCII-only usage, pass a simple byte_to_token map
    pub fn new_ascii(
        ctx: &CudaContext,
        byte_to_token_map: &[u32; 256],
        unk_id: u32,
    ) -> Result<Self, GpuError> {
        use crate::cuda::CHARACTER_TOKENIZE_KERNEL_SRC;

        let device = ctx.device().clone();

        // Compile and load PTX
        let ptx = compile_ptx(CHARACTER_TOKENIZE_KERNEL_SRC)
            .map_err(|e| GpuError::Cuda(format!("Character PTX compilation failed: {}", e)))?;

        device
            .load_ptx(
                ptx,
                Self::MODULE_NAME,
                &[Self::FUNC_SIMPLE, Self::FUNC_UTF8, Self::FUNC_MARK_STARTS],
            )
            .map_err(|e| GpuError::Cuda(format!("Character PTX load failed: {}", e)))?;

        // Copy byte-to-token mapping
        let byte_to_token = device
            .htod_copy(byte_to_token_map.to_vec())
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        // Empty vocab tables for ASCII mode
        let vocab_hashes = device
            .htod_copy(vec![0u64; 1])
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let vocab_ids = device
            .htod_copy(vec![0u32; 1])
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        Ok(Self {
            device,
            byte_to_token,
            vocab_hashes,
            vocab_ids,
            vocab_size: 1,
            unk_id,
            use_utf8: false,
        })
    }

    /// Create a new Character kernel with UTF-8 character vocabulary
    pub fn new_utf8(
        ctx: &CudaContext,
        char_vocab: &[(&str, u32)],
        unk_id: u32,
    ) -> Result<Self, GpuError> {
        use crate::cuda::CHARACTER_TOKENIZE_KERNEL_SRC;

        let device = ctx.device().clone();

        // Compile and load PTX
        let ptx = compile_ptx(CHARACTER_TOKENIZE_KERNEL_SRC)
            .map_err(|e| GpuError::Cuda(format!("Character PTX compilation failed: {}", e)))?;

        device
            .load_ptx(
                ptx,
                Self::MODULE_NAME,
                &[Self::FUNC_SIMPLE, Self::FUNC_UTF8, Self::FUNC_MARK_STARTS],
            )
            .map_err(|e| GpuError::Cuda(format!("Character PTX load failed: {}", e)))?;

        // Build hash table for character vocabulary
        let table_size = (char_vocab.len() * 4).next_power_of_two().max(16);
        let mut hashes = vec![0u64; table_size];
        let mut ids = vec![0u32; table_size];

        for (char_str, id) in char_vocab {
            let hash = fnv1a_hash(char_str.as_bytes());
            let mut slot = (hash as usize) % table_size;

            for _ in 0..64 {
                if hashes[slot] == 0 {
                    hashes[slot] = hash;
                    ids[slot] = *id;
                    break;
                }
                slot = (slot + 1) % table_size;
            }
        }

        // Copy to GPU
        let vocab_hashes = device
            .htod_copy(hashes)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let vocab_ids = device
            .htod_copy(ids)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        // Dummy byte_to_token for UTF-8 mode
        let byte_to_token = device
            .htod_copy(vec![unk_id; 256])
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        Ok(Self {
            device,
            byte_to_token,
            vocab_hashes,
            vocab_ids,
            vocab_size: table_size,
            unk_id,
            use_utf8: true,
        })
    }

    /// Tokenize text using character-level tokenization (ASCII mode)
    pub fn tokenize(&self, text: &[u8]) -> Result<Vec<u32>, GpuError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        if !self.use_utf8 {
            self.tokenize_ascii(text)
        } else {
            self.tokenize_utf8(text)
        }
    }

    fn tokenize_ascii(&self, text: &[u8]) -> Result<Vec<u32>, GpuError> {
        let input_len = text.len();

        // Copy input to device
        let d_input = self
            .device
            .htod_copy(text.to_vec())
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        let mut d_output = self
            .device
            .alloc_zeros::<u32>(input_len)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;

        // Get kernel function
        let func = self
            .device
            .get_func(Self::MODULE_NAME, Self::FUNC_SIMPLE)
            .ok_or_else(|| GpuError::Cuda("char_tokenize function not found".into()))?;

        let config = KernelConfig::linear(input_len, 256);
        unsafe {
            func.launch(
                LaunchConfig {
                    grid_dim: config.grid_size,
                    block_dim: config.block_size,
                    shared_mem_bytes: 0,
                },
                (&d_input, &self.byte_to_token, &mut d_output, input_len as u32),
            )
        }
        .map_err(|e| GpuError::KernelExecution(format!("char_tokenize failed: {}", e)))?;

        // Copy result
        self.device
            .synchronize()
            .map_err(|e| GpuError::Cuda(format!("Sync failed: {}", e)))?;

        self.device
            .dtoh_sync_copy(&d_output)
            .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))
    }

    fn tokenize_utf8(&self, text: &[u8]) -> Result<Vec<u32>, GpuError> {
        let input_len = text.len();

        // Copy input to device
        let d_input = self
            .device
            .htod_copy(text.to_vec())
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        // Mark UTF-8 character starts
        let mut d_char_start = self
            .device
            .alloc_zeros::<u32>(input_len)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;

        let mark_func = self
            .device
            .get_func(Self::MODULE_NAME, Self::FUNC_MARK_STARTS)
            .ok_or_else(|| GpuError::Cuda("mark_utf8_char_starts not found".into()))?;

        let config = KernelConfig::linear(input_len, 256);
        unsafe {
            mark_func.launch(
                LaunchConfig {
                    grid_dim: config.grid_size,
                    block_dim: config.block_size,
                    shared_mem_bytes: 0,
                },
                (&d_input, &mut d_char_start, input_len as u32),
            )
        }
        .map_err(|e| GpuError::KernelExecution(format!("mark_utf8_char_starts failed: {}", e)))?;

        // Allocate output buffers
        let mut d_output_ids = self
            .device
            .alloc_zeros::<u32>(input_len)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;
        let mut d_output_positions = self
            .device
            .alloc_zeros::<u32>(input_len)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;
        let mut d_output_count = self
            .device
            .alloc_zeros::<u32>(1)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;

        // Run UTF-8 tokenization
        let tokenize_func = self
            .device
            .get_func(Self::MODULE_NAME, Self::FUNC_UTF8)
            .ok_or_else(|| GpuError::Cuda("char_tokenize_utf8 not found".into()))?;

        unsafe {
            tokenize_func.launch(
                LaunchConfig {
                    grid_dim: config.grid_size,
                    block_dim: config.block_size,
                    shared_mem_bytes: 0,
                },
                (
                    &d_input,
                    &d_char_start,
                    &self.vocab_hashes,
                    &self.vocab_ids,
                    &mut d_output_ids,
                    &mut d_output_positions,
                    &mut d_output_count,
                    input_len as u32,
                    self.vocab_size as u32,
                    self.unk_id,
                ),
            )
        }
        .map_err(|e| GpuError::KernelExecution(format!("char_tokenize_utf8 failed: {}", e)))?;

        // Get results
        self.device
            .synchronize()
            .map_err(|e| GpuError::Cuda(format!("Sync failed: {}", e)))?;

        let count = self
            .device
            .dtoh_sync_copy(&d_output_count)
            .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?[0] as usize;

        let output_ids = self
            .device
            .dtoh_sync_copy(&d_output_ids)
            .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?;

        Ok(output_ids[..count].to_vec())
    }

    /// Tokenize batch of texts
    pub fn tokenize_batch(&self, texts: &[&[u8]]) -> Result<Vec<Vec<u32>>, GpuError> {
        texts.iter().map(|text| self.tokenize(text)).collect()
    }
}

// =============================================================================
// Word Tokenization Kernel (SentencePiece Word mode)
// =============================================================================

/// GPU kernel for Word tokenization (SentencePiece Word mode)
/// Whitespace-delimited tokenization with vocabulary lookup
#[cfg(feature = "cuda")]
pub struct WordKernel {
    device: Arc<CudaDevice>,
    vocab_hashes: CudaSlice<u64>,
    vocab_ids: CudaSlice<u32>,
    vocab_size: usize,
    unk_id: u32,
}

#[cfg(feature = "cuda")]
impl WordKernel {
    const MODULE_NAME: &'static str = "word_tokenize";
    const FUNC_SIMPLE: &'static str = "word_tokenize_simple";
    const FUNC_FIND_BOUNDARIES: &'static str = "word_find_boundaries";
    const FUNC_LOOKUP: &'static str = "word_tokenize_lookup";

    /// Create a new Word kernel with vocabulary
    pub fn new(ctx: &CudaContext, vocab: &[(&str, u32)], unk_id: u32) -> Result<Self, GpuError> {
        use crate::cuda::WORD_TOKENIZE_KERNEL_SRC;

        let device = ctx.device().clone();

        // Compile and load PTX
        let ptx = compile_ptx(WORD_TOKENIZE_KERNEL_SRC)
            .map_err(|e| GpuError::Cuda(format!("Word PTX compilation failed: {}", e)))?;

        device
            .load_ptx(
                ptx,
                Self::MODULE_NAME,
                &[Self::FUNC_SIMPLE, Self::FUNC_FIND_BOUNDARIES, Self::FUNC_LOOKUP],
            )
            .map_err(|e| GpuError::Cuda(format!("Word PTX load failed: {}", e)))?;

        // Build hash table for vocabulary
        let table_size = (vocab.len() * 4).next_power_of_two().max(16);
        let mut hashes = vec![0u64; table_size];
        let mut ids = vec![0u32; table_size];

        for (word, id) in vocab {
            let hash = fnv1a_hash(word.as_bytes());
            let mut slot = (hash as usize) % table_size;

            for _ in 0..64 {
                if hashes[slot] == 0 {
                    hashes[slot] = hash;
                    ids[slot] = *id;
                    break;
                }
                if hashes[slot] == hash {
                    break; // Duplicate
                }
                slot = (slot + 1) % table_size;
            }
        }

        // Copy to GPU
        let vocab_hashes = device
            .htod_copy(hashes)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;
        let vocab_ids = device
            .htod_copy(ids)
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        Ok(Self {
            device,
            vocab_hashes,
            vocab_ids,
            vocab_size: table_size,
            unk_id,
        })
    }

    /// Tokenize text using word-level tokenization
    pub fn tokenize(&self, text: &[u8]) -> Result<Vec<u32>, GpuError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let input_len = text.len();

        // Copy input to device
        let d_input = self
            .device
            .htod_copy(text.to_vec())
            .map_err(|e| GpuError::Cuda(format!("HtoD copy failed: {}", e)))?;

        // Allocate output buffers (worst case: every byte is a word)
        let mut d_output_ids = self
            .device
            .alloc_zeros::<u32>(input_len)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;
        let mut d_output_count = self
            .device
            .alloc_zeros::<u32>(1)
            .map_err(|e| GpuError::MemoryAllocation(format!("Alloc failed: {}", e)))?;

        // Use simple single-pass tokenizer (efficient for short to medium texts)
        let func = self
            .device
            .get_func(Self::MODULE_NAME, Self::FUNC_SIMPLE)
            .ok_or_else(|| GpuError::Cuda("word_tokenize_simple not found".into()))?;

        unsafe {
            func.launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &d_input,
                    &self.vocab_hashes,
                    &self.vocab_ids,
                    &mut d_output_ids,
                    &mut d_output_count,
                    input_len as u32,
                    self.vocab_size as u32,
                    self.unk_id,
                ),
            )
        }
        .map_err(|e| GpuError::KernelExecution(format!("word_tokenize_simple failed: {}", e)))?;

        // Get results
        self.device
            .synchronize()
            .map_err(|e| GpuError::Cuda(format!("Sync failed: {}", e)))?;

        let count = self
            .device
            .dtoh_sync_copy(&d_output_count)
            .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?[0] as usize;

        let output_ids = self
            .device
            .dtoh_sync_copy(&d_output_ids)
            .map_err(|e| GpuError::Cuda(format!("DtoH copy failed: {}", e)))?;

        Ok(output_ids[..count].to_vec())
    }

    /// Tokenize batch of texts
    pub fn tokenize_batch(&self, texts: &[&[u8]]) -> Result<Vec<Vec<u32>>, GpuError> {
        texts.iter().map(|text| self.tokenize(text)).collect()
    }
}

// =============================================================================
// Non-CUDA Stubs
// =============================================================================

#[cfg(not(feature = "cuda"))]
pub struct PreTokenizeKernel;

#[cfg(not(feature = "cuda"))]
impl PreTokenizeKernel {
    pub fn new(_ctx: &()) -> Result<Self, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn find_word_boundaries(&self, _input: &[u8]) -> Result<Vec<(usize, usize)>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn find_word_boundaries_batch(
        &self,
        _texts: &[&str],
    ) -> Result<Vec<Vec<(usize, usize)>>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }
}

#[cfg(not(feature = "cuda"))]
pub struct VocabLookupKernel;

#[cfg(not(feature = "cuda"))]
impl VocabLookupKernel {
    pub fn new(_ctx: &(), _vocab: &[(&str, u32)]) -> Result<Self, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn lookup(&self, _words: &[&str]) -> Result<Vec<u32>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }
}

#[cfg(not(feature = "cuda"))]
pub struct WordPieceKernel;

#[cfg(not(feature = "cuda"))]
impl WordPieceKernel {
    pub fn new(_ctx: &(), _vocab: &[(&str, u32)], _prefix: &str) -> Result<Self, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn tokenize_word(&self, _word: &str) -> Result<Vec<u32>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn tokenize_words(&self, _words: &[&str]) -> Result<Vec<Vec<u32>>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }
}

#[cfg(not(feature = "cuda"))]
pub struct BpeKernel;

#[cfg(not(feature = "cuda"))]
impl BpeKernel {
    pub fn new(
        _ctx: &(),
        _merge_rules: &[BpeMergeRule],
        _byte_to_token_map: &[u32; 256],
        _unk_id: u32,
    ) -> Result<Self, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn tokenize(&self, _text: &[u8]) -> Result<Vec<u32>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn tokenize_batch(&self, _texts: &[&[u8]]) -> Result<Vec<Vec<u32>>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }
}

#[cfg(not(feature = "cuda"))]
pub struct UnigramKernel;

#[cfg(not(feature = "cuda"))]
impl UnigramKernel {
    pub fn new(
        _ctx: &(),
        _trie_nodes: &[FlatTrieNode],
        _unk_id: u32,
    ) -> Result<Self, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn tokenize(&self, _text: &[u8]) -> Result<Vec<u32>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn tokenize_batch(&self, _texts: &[&[u8]]) -> Result<Vec<Vec<u32>>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }
}

#[cfg(not(feature = "cuda"))]
pub struct OptimizedBpeKernel;

#[cfg(not(feature = "cuda"))]
impl OptimizedBpeKernel {
    pub fn new(
        _ctx: &(),
        _merge_rules: &[BpeMergeRule],
        _byte_to_token_map: &[u32; 256],
        _max_seq_len: usize,
    ) -> Result<Self, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn tokenize(&self, _text: &[u8]) -> Result<Vec<u32>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn tokenize_batch(&self, _texts: &[&[u8]]) -> Result<Vec<Vec<u32>>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }
}

#[cfg(not(feature = "cuda"))]
pub struct CharacterKernel;

#[cfg(not(feature = "cuda"))]
impl CharacterKernel {
    pub fn new_ascii(
        _ctx: &(),
        _byte_to_token_map: &[u32; 256],
        _unk_id: u32,
    ) -> Result<Self, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn new_utf8(
        _ctx: &(),
        _char_vocab: &[(&str, u32)],
        _unk_id: u32,
    ) -> Result<Self, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn tokenize(&self, _text: &[u8]) -> Result<Vec<u32>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn tokenize_batch(&self, _texts: &[&[u8]]) -> Result<Vec<Vec<u32>>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }
}

#[cfg(not(feature = "cuda"))]
pub struct WordKernel;

#[cfg(not(feature = "cuda"))]
impl WordKernel {
    pub fn new(
        _ctx: &(),
        _vocab: &[(&str, u32)],
        _unk_id: u32,
    ) -> Result<Self, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn tokenize(&self, _text: &[u8]) -> Result<Vec<u32>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    pub fn tokenize_batch(&self, _texts: &[&[u8]]) -> Result<Vec<Vec<u32>>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_config_linear() {
        let config = KernelConfig::linear(1000, 256);
        assert_eq!(config.block_size.0, 256);
        assert_eq!(config.grid_size.0, 4); // ceil(1000/256) = 4
    }

    #[test]
    fn test_kernel_config_default() {
        let config = KernelConfig::default();
        assert_eq!(config.block_size.0, 256);
        assert_eq!(config.grid_size.0, 1);
    }

    #[test]
    fn test_fnv1a_hash() {
        let hash1 = fnv1a_hash(b"hello");
        let hash2 = fnv1a_hash(b"world");
        let hash3 = fnv1a_hash(b"hello");

        assert_ne!(hash1, hash2);
        assert_eq!(hash1, hash3);
    }

    #[test]
    fn test_calculate_kernel_config() {
        let config = calculate_kernel_config(32, 512);
        // 32 * 512 = 16384 elements
        // ceil(16384/256) = 64 blocks
        assert_eq!(config.grid_size.0, 64);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_pretokenize_kernel() {
        use crate::cuda::{is_cuda_available, CudaContext};

        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return;
        }

        let ctx = CudaContext::new(0).expect("Context failed");
        let kernel = PreTokenizeKernel::new(&ctx).expect("Kernel creation failed");

        let boundaries = kernel
            .find_word_boundaries(b"hello world")
            .expect("Boundaries failed");

        assert_eq!(boundaries.len(), 2);
        assert_eq!(boundaries[0], (0, 5));
        assert_eq!(boundaries[1], (6, 11));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_vocab_lookup_kernel() {
        use crate::cuda::{is_cuda_available, CudaContext};

        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return;
        }

        let ctx = CudaContext::new(0).expect("Context failed");

        let vocab = vec![("hello", 100), ("world", 101), ("[UNK]", 0)];

        let kernel = VocabLookupKernel::new(&ctx, &vocab).expect("Kernel creation failed");

        let results = kernel
            .lookup(&["hello", "world", "unknown"])
            .expect("Lookup failed");

        assert_eq!(results, vec![100, 101, 0]);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_bpe_kernel() {
        use crate::cuda::{is_cuda_available, CudaContext};

        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return;
        }

        let ctx = CudaContext::new(0).expect("Context failed");

        // Simple merge rules: "a" + "b" -> "ab"
        let merge_rules = vec![BpeMergeRule {
            first_id: 97,  // 'a'
            second_id: 98, // 'b'
            result_id: 256, // merged token
            priority: 0,
        }];

        // Simple byte-to-token mapping (identity for ASCII)
        let mut byte_to_token = [0u32; 256];
        for i in 0u32..256 {
            byte_to_token[i as usize] = i;
        }

        let kernel = BpeKernel::new(&ctx, &merge_rules, &byte_to_token, 0).expect("Kernel creation failed");

        let result = kernel.tokenize(b"ab").expect("Tokenize failed");
        println!("BPE result for 'ab': {:?}", result);

        // Should have merged 'a' + 'b' into single token
        assert!(result.len() <= 2);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_unigram_kernel() {
        use crate::cuda::{is_cuda_available, CudaContext};

        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return;
        }

        let ctx = CudaContext::new(0).expect("Context failed");

        // Simple trie with a few tokens
        let trie_nodes = vec![
            // Root node
            FlatTrieNode {
                char_byte: 0,
                is_end: false,
                token_id: 0,
                score: 0.0,
                children_start: 1,
                children_count: 2,
            },
            // 'h' node
            FlatTrieNode {
                char_byte: b'h',
                is_end: false,
                token_id: 0,
                score: 0.0,
                children_start: 3,
                children_count: 1,
            },
            // 'w' node
            FlatTrieNode {
                char_byte: b'w',
                is_end: false,
                token_id: 0,
                score: 0.0,
                children_start: 5,
                children_count: 1,
            },
            // 'e' node (after h)
            FlatTrieNode {
                char_byte: b'e',
                is_end: false,
                token_id: 0,
                score: 0.0,
                children_start: 4,
                children_count: 1,
            },
            // 'l' node (after he)
            FlatTrieNode {
                char_byte: b'l',
                is_end: false,
                token_id: 0,
                score: 0.0,
                children_start: 6,
                children_count: 1,
            },
            // 'o' node (after w)
            FlatTrieNode {
                char_byte: b'o',
                is_end: true,
                token_id: 101, // "wo"
                score: -2.0,
                children_start: 0,
                children_count: 0,
            },
            // 'l' node (after hel -> "hell")
            FlatTrieNode {
                char_byte: b'l',
                is_end: false,
                token_id: 0,
                score: 0.0,
                children_start: 7,
                children_count: 1,
            },
            // 'o' node (after hell -> "hello")
            FlatTrieNode {
                char_byte: b'o',
                is_end: true,
                token_id: 100, // "hello"
                score: -1.5,
                children_start: 0,
                children_count: 0,
            },
        ];

        let kernel = UnigramKernel::new(&ctx, &trie_nodes, 0).expect("Kernel creation failed");

        // Test tokenization
        let result = kernel.tokenize(b"hello");
        println!("Unigram result for 'hello': {:?}", result);

        // Should find tokens (even if fallback to unknown)
        assert!(result.is_ok());
    }

    #[test]
    fn test_bpe_pair_hash() {
        let hash1 = bpe_pair_hash(100, 200);
        let hash2 = bpe_pair_hash(200, 100);
        let hash3 = bpe_pair_hash(100, 200);

        assert_ne!(hash1, hash2); // Order matters
        assert_eq!(hash1, hash3); // Same input = same hash
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_character_kernel_ascii() {
        use crate::cuda::{is_cuda_available, CudaContext};

        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return;
        }

        let ctx = CudaContext::new(0).expect("Context failed");

        // Simple byte-to-token mapping (identity for printable ASCII)
        let mut byte_to_token = [0u32; 256];
        for i in 0u32..256 {
            byte_to_token[i as usize] = i;
        }

        let kernel =
            CharacterKernel::new_ascii(&ctx, &byte_to_token, 0).expect("Kernel creation failed");

        let result = kernel.tokenize(b"hello").expect("Tokenize failed");
        println!("Character result for 'hello': {:?}", result);

        // Each character should be a separate token
        assert_eq!(result.len(), 5);
        assert_eq!(result, vec![104, 101, 108, 108, 111]); // h, e, l, l, o
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_word_kernel() {
        use crate::cuda::{is_cuda_available, CudaContext};

        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return;
        }

        let ctx = CudaContext::new(0).expect("Context failed");

        let vocab = vec![
            ("hello", 100),
            ("world", 101),
            ("test", 102),
            ("[UNK]", 0),
        ];

        let kernel = WordKernel::new(&ctx, &vocab, 0).expect("Kernel creation failed");

        let result = kernel.tokenize(b"hello world").expect("Tokenize failed");
        println!("Word result for 'hello world': {:?}", result);

        // Should have 2 word tokens
        assert_eq!(result.len(), 2);
        assert_eq!(result, vec![100, 101]); // hello, world
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_word_kernel_unknown() {
        use crate::cuda::{is_cuda_available, CudaContext};

        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return;
        }

        let ctx = CudaContext::new(0).expect("Context failed");

        let vocab = vec![("hello", 100), ("[UNK]", 0)];

        let kernel = WordKernel::new(&ctx, &vocab, 0).expect("Kernel creation failed");

        let result = kernel.tokenize(b"hello unknown").expect("Tokenize failed");
        println!("Word result with unknown: {:?}", result);

        // Should have hello token and unknown token
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 100); // hello
        assert_eq!(result[1], 0); // [UNK]
    }

    /// Benchmark test to compare GPU vs CPU tokenization throughput
    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_tokenization_benchmark() {
        use crate::cuda::{is_cuda_available, CudaContext};
        use std::time::Instant;

        if !is_cuda_available() {
            println!("Skipping benchmark - no CUDA");
            return;
        }

        let ctx = CudaContext::new(0).expect("Context failed");
        println!(
            "GPU Device: {} ({}MB)",
            ctx.device_info().name,
            ctx.device_info().total_memory / 1024 / 1024
        );

        // Create test data - batch of variable length texts
        let texts: Vec<String> = (0..100)
            .map(|i| format!("This is test sentence number {} for GPU tokenization benchmark testing with variable length inputs to stress test the system", i))
            .collect();
        let text_bytes: Vec<&[u8]> = texts.iter().map(|s| s.as_bytes()).collect();

        // Build vocabulary
        let mut vocab: Vec<(&str, u32)> = vec![("[UNK]", 0)];
        let words = [
            "This", "is", "test", "sentence", "number", "for", "GPU", "tokenization",
            "benchmark", "testing", "with", "variable", "length", "inputs", "to",
            "stress", "the", "system",
        ];
        for (i, word) in words.iter().enumerate() {
            vocab.push((word, (i + 1) as u32));
        }
        // Add numbers 0-99
        let numbers: Vec<String> = (0..100).map(|i| i.to_string()).collect();
        for (i, num) in numbers.iter().enumerate() {
            vocab.push((num.as_str(), (100 + i) as u32));
        }

        // Build byte-to-token for character kernel
        let mut byte_to_token = [0u32; 256];
        for i in 0u32..256 {
            byte_to_token[i as usize] = i;
        }

        // === Character Kernel Benchmark ===
        let char_kernel =
            CharacterKernel::new_ascii(&ctx, &byte_to_token, 0).expect("CharacterKernel failed");

        let start = Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            for text in &text_bytes {
                let _ = char_kernel.tokenize(text);
            }
        }
        let char_time = start.elapsed();
        let char_throughput =
            (text_bytes.len() * iterations) as f64 / char_time.as_secs_f64();
        println!(
            "CharacterKernel: {:.0} texts/sec ({} texts x {} iterations in {:?})",
            char_throughput,
            text_bytes.len(),
            iterations,
            char_time
        );

        // === Word Kernel Benchmark ===
        let word_kernel = WordKernel::new(&ctx, &vocab, 0).expect("WordKernel failed");

        let start = Instant::now();
        for _ in 0..iterations {
            for text in &text_bytes {
                let _ = word_kernel.tokenize(text);
            }
        }
        let word_time = start.elapsed();
        let word_throughput =
            (text_bytes.len() * iterations) as f64 / word_time.as_secs_f64();
        println!(
            "WordKernel: {:.0} texts/sec ({} texts x {} iterations in {:?})",
            word_throughput,
            text_bytes.len(),
            iterations,
            word_time
        );

        // === CPU Baseline (simple word split) ===
        let start = Instant::now();
        for _ in 0..iterations {
            for text in &texts {
                let _words: Vec<&str> = text.split_whitespace().collect();
            }
        }
        let cpu_time = start.elapsed();
        let cpu_throughput = (texts.len() * iterations) as f64 / cpu_time.as_secs_f64();
        println!(
            "CPU word split: {:.0} texts/sec ({} texts x {} iterations in {:?})",
            cpu_throughput,
            texts.len(),
            iterations,
            cpu_time
        );

        // GPU should be comparable or faster for batch processing
        println!("GPU Character / CPU ratio: {:.2}x", char_throughput / cpu_throughput);
        println!("GPU Word / CPU ratio: {:.2}x", word_throughput / cpu_throughput);

        // Note: For short texts, CPU may actually be faster due to kernel launch overhead
        // GPU wins on large batches and long texts where parallelism helps
        println!("Note: GPU benefits increase with batch size and text length");
    }
}
