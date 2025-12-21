# BudTikTok Implementation Plan: 10x+ Performance HuggingFace Drop-in Replacement

## Executive Summary

This document provides a detailed implementation plan to make BudTikTok a complete, 100% accurate drop-in replacement for HuggingFace Tokenizers with **guaranteed 10x+ performance improvement**. The plan is based on extensive research of SOTA algorithms, academic papers, and high-performance implementations.

### Performance Targets

| Metric | Current BudTikTok | Target | HuggingFace Baseline |
|--------|-------------------|--------|---------------------|
| Single-thread throughput | 41 MB/s | **100+ MB/s** | 8 MB/s |
| Multi-thread throughput | 443 MB/s | **800+ MB/s** | 46 MB/s |
| Speedup vs HF (single) | 5x | **12-15x** | 1x |
| Speedup vs HF (multi) | 9.6x | **15-20x** | 1x |
| Accuracy | 99.96% | **100%** | 100% |

---

## Table of Contents

1. [Research Summary](#1-research-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 1: Core Algorithm Optimizations](#3-phase-1-core-algorithm-optimizations)
4. [Phase 2: SIMD Acceleration](#4-phase-2-simd-acceleration)
5. [Phase 3: Multi-Core Parallelization](#5-phase-3-multi-core-parallelization)
6. [Phase 4: Hardware Autodetection](#6-phase-4-hardware-autodetection)
7. [Phase 5: Complete Feature Parity](#7-phase-5-complete-feature-parity)
8. [Phase 6: Accuracy Verification](#8-phase-6-accuracy-verification)
9. [Implementation Timeline](#9-implementation-timeline)
10. [References](#10-references)

---

## 1. Research Summary

### 1.1 SOTA Tokenization Algorithms

#### Linear-Time BPE (GitHub rust-gems)
**Source:** [github/rust-gems](https://github.com/github/rust-gems) | [Blog Post](https://github.blog/ai-and-ml/llms/so-many-tokens-so-little-time-introducing-a-faster-more-flexible-byte-pair-tokenizer/)

**Key Innovation:** Dynamic programming with bitfield tracking
- Tracks encodings of ALL text prefixes in O(n) space
- Exploits property: knowing final token uniquely determines entire sequence
- Uses compatibility rule: `is_compatible(token_a, token_b)` with at most 14 lookups
- **Performance:** 10x faster than HuggingFace, 4x faster than tiktoken

**Algorithm:**
```rust
// For each position, track the last token of the optimal encoding
fn encode_linear(text: &[u8], vocab: &Vocabulary) -> Vec<u32> {
    let n = text.len();
    let mut last_token = vec![None; n + 1];  // O(n) space
    let mut valid_positions = BitVec::new(n + 1);  // Bitfield tracking
    valid_positions.set(0, true);

    for end in 1..=n {
        for start in (0..end).rev() {
            if !valid_positions.get(start) { continue; }
            if let Some(token) = vocab.lookup(&text[start..end]) {
                if is_compatible(last_token[start], token) {
                    last_token[end] = Some(token);
                    valid_positions.set(end, true);
                    break;  // Greedy: take longest compatible token
                }
            }
        }
    }

    // Backtrack to reconstruct encoding
    reconstruct_path(&last_token, n)
}
```

#### LinMaxMatch WordPiece (Google Research)
**Source:** [EMNLP 2021 Paper](https://aclanthology.org/2021.emnlp-main.160/) | [Google Blog](https://research.google/blog/a-fast-wordpiece-tokenization-system/)

**Key Innovation:** Aho-Corasick-inspired failure links on vocabulary trie
- O(n) complexity vs O(n²) or O(nm) for traditional algorithms
- **Performance:** 3x faster average, 4.5x faster at 95th percentile
- Combines pre-tokenization and WordPiece into single linear pass

**Algorithm:**
```rust
// Build vocabulary trie with failure links
struct LinMaxMatchTrie {
    trie: DoubleArrayTrie,      // Compact trie representation
    failure_links: Vec<u32>,    // Aho-Corasick-style fallback
    output_links: Vec<u32>,     // Token output at each node
}

impl LinMaxMatchTrie {
    fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut node = 0;  // Start at root
        let mut i = 0;

        while i < text.len() {
            let c = text.as_bytes()[i];

            // Try to advance in trie
            if let Some(next) = self.trie.transition(node, c) {
                node = next;
                if let Some(token) = self.output_links[node] {
                    tokens.push(token);
                    node = 0;  // Reset to root after match
                }
                i += 1;
            } else {
                // Use failure link (Aho-Corasick style)
                node = self.failure_links[node];
                if node == 0 && !tokens.is_empty() {
                    // No match possible, emit [UNK]
                    tokens.push(UNK_ID);
                    i += 1;
                }
            }
        }
        tokens
    }
}
```

#### O(n) Unigram Viterbi (SentencePiece)
**Source:** [SentencePiece unigram_model.cc](https://github.com/google/sentencepiece/blob/master/src/unigram_model.cc)

**Key Innovation:** Memory-efficient Viterbi without storing full lattice
- Only stores best_path[i] for each position i
- Reduces O(n*k) memory to O(n), where k = max tokens per position
- No dynamic lattice node pool needed

**Algorithm:**
```rust
struct UnigramViterbi {
    vocab: Vec<(String, f64)>,  // (token, log_prob)
    trie: DoubleArrayTrie,      // For fast prefix matching
}

impl UnigramViterbi {
    fn encode(&self, text: &str) -> Vec<u32> {
        let n = text.len();
        let mut best_score = vec![f64::NEG_INFINITY; n + 1];
        let mut best_token = vec![0u32; n + 1];
        let mut best_start = vec![0usize; n + 1];

        best_score[0] = 0.0;

        // Forward pass: O(n * max_token_len)
        for end in 1..=n {
            // Find all tokens ending at position `end`
            for (start, token_id, score) in self.find_tokens_ending_at(text, end) {
                let candidate_score = best_score[start] + score;
                if candidate_score > best_score[end] {
                    best_score[end] = candidate_score;
                    best_token[end] = token_id;
                    best_start[end] = start;
                }
            }
        }

        // Backward pass: O(n)
        let mut tokens = Vec::new();
        let mut pos = n;
        while pos > 0 {
            tokens.push(best_token[pos]);
            pos = best_start[pos];
        }
        tokens.reverse();
        tokens
    }
}
```

### 1.2 SIMD Optimization Research

#### simdutf Library
**Source:** [simdutf/simdutf](https://github.com/simdutf/simdutf) | [Documentation](https://simdutf.github.io/simdutf/index.html)

**Performance:** Billions of characters per second
- Used in Node.js, WebKit/Safari, Chromium, Cloudflare Workers, Bun
- 3-10x faster than ICU on non-ASCII, 20x faster on ASCII
- Supports SSE2, AVX2, AVX-512, NEON, RISC-V Vector

**Key Functions:**
- `validate_utf8()` - Fast UTF-8 validation
- `convert_utf8_to_utf16()` - Transcoding at 10+ GB/s

#### memchr Library
**Source:** [BurntSushi/memchr](https://github.com/BurntSushi/memchr)

**Features:**
- SIMD-accelerated byte search (1, 2, or 3 bytes)
- memmem: SIMD substring search
- Platform support: x86_64 (SSE2, AVX2), aarch64 (NEON), wasm32

**Key Insight:** Dynamic CPU feature detection at runtime allows portable binaries with optimal performance.

#### Teddy Algorithm
**Source:** [aho-corasick/teddy](https://github.com/BurntSushi/aho-corasick/blob/master/src/packed/teddy/README.md)

**Innovation:** SIMD-accelerated multiple substring matching
- Scans 16/32 bytes at a time using SIMD fingerprints
- "Completely blows away competition for short substrings"
- Originated from Intel's Hyperscan project

#### AVX-512 UTF-8 Processing
**Source:** [Daniel Lemire's Research](https://lemire.me/blog/2023/09/13/transcoding-unicode-strings-at-crazy-speeds-with-avx-512/)

**Key Paper:** "Transcoding Unicode Characters with AVX-512 Instructions" (Clausecker & Lemire, 2023)
- 2x faster than previous best with AVX-512
- Efficient validation against overlong sequences and illegal code points
- Requires AVX512-VBMI2 (Ice Lake+, Zen 4+)

### 1.3 Multi-Core Parallelization Research

#### Rayon Work-Stealing
**Source:** [rayon-rs/rayon](https://github.com/rayon-rs/rayon) | [Optimization Blog](https://gendignoux.com/blog/2024/11/18/rust-rayon-optimized.html)

**Key Concepts:**
1. **Work Stealing:** Idle threads steal from busy threads' queues
2. **Adaptive Chunking:** Dynamic chunk sizes based on workload
3. **Bulk Stealing:** Steal in bulk to minimize synchronization overhead
4. **Binary Tree Splitting:** Recursive subdivision for optimal parallelism

**Best Practices:**
```rust
// Optimal parallel encoding pattern
fn encode_batch_parallel(texts: &[&str], tokenizer: &Tokenizer) -> Vec<Encoding> {
    texts
        .par_iter()
        .with_min_len(8)  // Minimum chunk size to amortize overhead
        .map(|text| tokenizer.encode(text))
        .collect()
}
```

**Load Balancing Consideration:**
- Fixed partitioning by item count may cause imbalance
- Use adaptive strategies: `par_iter()` with work-stealing
- For variable-length texts, sort by length and interleave

#### BlockBPE: GPU-Parallel BPE
**Source:** [arXiv:2507.11941](https://arxiv.org/html/2507.11941v1)

**Innovation:** Near-linear parallel BPE for GPU
- 2x higher throughput than tiktoken on high-batch workloads
- 2.5x higher than HuggingFace Tokenizers
- Optimized for high-throughput batch inference

### 1.4 Double-Array Trie Data Structure

#### daachorse Library
**Source:** [daac-tools/daachorse](https://github.com/daac-tools/daachorse) | [Paper](https://arxiv.org/abs/2207.13870)

**Performance:**
- 3.0-5.2x faster than aho-corasick crate
- 56-60% smaller memory footprint
- Constant-time state-to-state traversal
- 12 bytes per state

**Key Features:**
- LeftmostLongest matching for tokenization
- Overlapping and non-overlapping search modes
- Optimized for large pattern sets (tested with 675K patterns)

### 1.5 Cache Optimization Research

**Key Principles:**
1. **Cache Line Awareness:** 64 bytes per line on most CPUs
2. **False Sharing Prevention:** Align data to cache lines for multi-threaded code
3. **Struct of Arrays (SoA):** Better cache utilization than Array of Structs
4. **Linear Data Structures:** Avoid pointer chasing; prefer contiguous memory

**C++17 Constants:**
- `std::hardware_destructive_interference_size` - Avoid false sharing
- `std::hardware_constructive_interference_size` - Promote true sharing

---

## 2. Architecture Overview

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                      BudTikTok Architecture                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
│  │   Input     │───▶│ Dispatcher  │───▶│   Output    │              │
│  │   Text      │    │  (Runtime)  │    │  Encoding   │              │
│  └─────────────┘    └──────┬──────┘    └─────────────┘              │
│                            │                                         │
│           ┌────────────────┼────────────────┐                       │
│           ▼                ▼                ▼                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │   Scalar    │  │  AVX2/NEON  │  │   AVX-512   │                 │
│  │   Path      │  │    Path     │  │    Path     │                 │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │
│         │                │                │                         │
│         └────────────────┼────────────────┘                         │
│                          ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    Shared Components                           │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐               │ │
│  │  │ Double-    │  │ Vocabulary │  │ Special    │               │ │
│  │  │ Array Trie │  │   Cache    │  │ Token      │               │ │
│  │  │ (daachorse)│  │  (LRU)     │  │ Matcher    │               │ │
│  │  └────────────┘  └────────────┘  └────────────┘               │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    Rayon Thread Pool                           │ │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐               │ │
│  │  │Thread 1│  │Thread 2│  │Thread 3│  │Thread N│  Work Stealing│ │
│  │  └────────┘  └────────┘  └────────┘  └────────┘               │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Module Structure

```
budtiktok-core/
├── src/
│   ├── lib.rs                 # Public API
│   ├── tokenizer.rs           # Unified Tokenizer trait
│   │
│   ├── models/
│   │   ├── mod.rs
│   │   ├── bpe_linear.rs      # O(n) BPE (rust-gems style)
│   │   ├── wordpiece_fast.rs  # LinMaxMatch WordPiece
│   │   ├── unigram_viterbi.rs # O(n) memory Viterbi
│   │   └── wordlevel.rs       # SIMD WordLevel
│   │
│   ├── simd/
│   │   ├── mod.rs             # Dispatcher
│   │   ├── scalar.rs          # Fallback implementation
│   │   ├── avx2.rs            # AVX2 implementations
│   │   ├── avx512.rs          # AVX-512 implementations
│   │   ├── neon.rs            # ARM NEON implementations
│   │   └── detect.rs          # Runtime CPU detection
│   │
│   ├── trie/
│   │   ├── mod.rs
│   │   ├── double_array.rs    # Double-array trie
│   │   └── failure_links.rs   # Aho-Corasick failure links
│   │
│   ├── parallel/
│   │   ├── mod.rs
│   │   ├── batch.rs           # Batch processing
│   │   ├── adaptive.rs        # Adaptive chunking
│   │   └── cache_aligned.rs   # Cache-line aligned data
│   │
│   ├── normalizers/           # All HF normalizers
│   ├── pretokenizers/         # All HF pre-tokenizers
│   ├── postprocessors/        # All HF post-processors
│   ├── decoders/              # All HF decoders
│   │
│   ├── encoding.rs            # Encoding structure + methods
│   ├── vocab.rs               # Vocabulary management
│   ├── config.rs              # Configuration + serialization
│   └── loader.rs              # HF-compatible loading
│
├── benches/
│   ├── accuracy_test.rs       # 100% accuracy verification
│   ├── performance_1gb.rs     # Full benchmark suite
│   └── simd_dispatch.rs       # SIMD path benchmarks
│
└── tests/
    ├── hf_compatibility/      # HF round-trip tests
    └── edge_cases/            # Unicode edge cases
```

---

## 3. Phase 1: Core Algorithm Optimizations

### 3.1 Linear-Time BPE Implementation

**Goal:** Replace current BPE with O(n) algorithm from rust-gems research

**Implementation Details:**

```rust
// src/models/bpe_linear.rs

use bitvec::prelude::*;

/// Linear-time BPE encoder using dynamic programming with bitfield tracking
pub struct LinearBpeEncoder {
    /// Token vocabulary: token_bytes -> token_id
    vocab: FxHashMap<Vec<u8>, u32>,

    /// Reverse vocabulary for compatibility checking
    id_to_token: Vec<Vec<u8>>,

    /// Merge priority: (pair) -> priority (lower = earlier merge)
    merge_priority: FxHashMap<(u32, u32), u32>,

    /// Precomputed compatibility cache for common pairs
    compat_cache: FxHashMap<(u32, u32), bool>,
}

impl LinearBpeEncoder {
    /// O(n) encoding using prefix-tracking DP
    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        let n = text.len();
        if n == 0 { return Vec::new(); }

        // Track last token for each prefix position
        let mut last_token: Vec<Option<u32>> = vec![None; n + 1];

        // Bitfield for valid tokenization positions
        let mut valid: BitVec = bitvec![0; n + 1];
        valid.set(0, true);

        // DP: for each end position, find longest compatible token
        for end in 1..=n {
            // Iterate backwards from end to find tokens
            for start in (0..end).rev() {
                if !valid[start] { continue; }

                // Check if text[start..end] is a valid token
                if let Some(&token_id) = self.vocab.get(&text[start..end]) {
                    // Check compatibility with previous token
                    if self.is_compatible(last_token[start], token_id) {
                        last_token[end] = Some(token_id);
                        valid.set(end, true);
                        break;  // Greedy: take longest compatible
                    }
                }

                // Early termination if token too long
                if end - start > self.max_token_len() { break; }
            }
        }

        // Backtrack to reconstruct encoding
        self.reconstruct(&last_token, n)
    }

    /// Check if token_b can follow token_a in a valid BPE encoding
    /// This is the key insight: at most 14 lookups needed
    #[inline]
    fn is_compatible(&self, prev: Option<u32>, next: u32) -> bool {
        let Some(prev_id) = prev else { return true; };

        // Check cache first
        if let Some(&result) = self.compat_cache.get(&(prev_id, next)) {
            return result;
        }

        // Full compatibility check (at most 14 lookups)
        self.compute_compatibility(prev_id, next)
    }

    fn compute_compatibility(&self, prev: u32, next: u32) -> bool {
        let prev_bytes = &self.id_to_token[prev as usize];
        let next_bytes = &self.id_to_token[next as usize];

        // Check all possible splits of the combined sequence
        let combined: Vec<u8> = prev_bytes.iter()
            .chain(next_bytes.iter())
            .copied()
            .collect();

        // If any alternative tokenization has higher priority, incompatible
        for split in 1..combined.len() {
            let left = &combined[..split];
            let right = &combined[split..];

            if let (Some(&left_id), Some(&right_id)) =
                (self.vocab.get(left), self.vocab.get(right))
            {
                // Check if this alternative has higher priority
                if self.has_higher_priority(left_id, right_id, prev, next) {
                    return false;
                }
            }
        }

        true
    }
}
```

**Effort:** 2 weeks
**Expected Improvement:** 3-5x over current BPE

### 3.2 LinMaxMatch WordPiece

**Goal:** Implement Google's O(n) WordPiece algorithm

```rust
// src/models/wordpiece_fast.rs

use crate::trie::{DoubleArrayTrie, FailureLinks};

/// Fast WordPiece using LinMaxMatch algorithm
pub struct LinMaxMatchWordPiece {
    /// Double-array trie for vocabulary
    trie: DoubleArrayTrie,

    /// Failure links (Aho-Corasick style)
    failure: FailureLinks,

    /// Output token IDs at each trie node
    output: Vec<Option<u32>>,

    /// Continuation prefix (e.g., "##")
    prefix: Vec<u8>,

    /// Unknown token ID
    unk_id: u32,
}

impl LinMaxMatchWordPiece {
    /// Build trie with failure links from vocabulary
    pub fn from_vocab(vocab: &[(String, u32)], prefix: &str, unk_id: u32) -> Self {
        let mut trie_builder = DoubleArrayTrieBuilder::new();

        // Add all vocabulary tokens to trie
        for (token, id) in vocab {
            trie_builder.insert(token.as_bytes(), *id);
        }

        let trie = trie_builder.build();
        let failure = FailureLinks::build(&trie);

        Self {
            trie,
            failure,
            output: Self::compute_outputs(vocab),
            prefix: prefix.as_bytes().to_vec(),
            unk_id,
        }
    }

    /// O(n) tokenization using failure links
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let n = bytes.len();

        let mut tokens = Vec::with_capacity(n / 4);  // Estimate
        let mut i = 0;
        let mut is_continuation = false;

        while i < n {
            // Skip whitespace
            while i < n && bytes[i].is_ascii_whitespace() {
                i += 1;
                is_continuation = false;
            }
            if i >= n { break; }

            // Tokenize word
            let word_start = i;
            while i < n && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }

            // Process word through trie
            self.tokenize_word(&bytes[word_start..i], &mut tokens, is_continuation);
            is_continuation = true;
        }

        tokens
    }

    fn tokenize_word(&self, word: &[u8], tokens: &mut Vec<u32>, continuation: bool) {
        let mut node = 0;  // Trie root
        let mut last_match: Option<(usize, u32)> = None;
        let mut start = 0;

        // If continuation, try with prefix first
        let effective_word = if continuation {
            let mut prefixed = self.prefix.clone();
            prefixed.extend_from_slice(word);
            prefixed
        } else {
            word.to_vec()
        };

        let mut i = 0;
        while i < effective_word.len() {
            let byte = effective_word[i];

            // Try to advance in trie
            loop {
                if let Some(next) = self.trie.transition(node, byte) {
                    node = next;
                    if let Some(token_id) = self.output[node as usize] {
                        last_match = Some((i + 1, token_id));
                    }
                    break;
                } else if node == 0 {
                    // At root, no match possible
                    break;
                } else {
                    // Use failure link
                    node = self.failure.get(node);
                }
            }

            i += 1;

            // If no progress and we have a match, emit it
            if node == 0 && last_match.is_some() {
                let (end, token_id) = last_match.take().unwrap();
                tokens.push(token_id);
                i = start + end;
                start = i;
                node = 0;
            }
        }

        // Handle remaining
        if let Some((_, token_id)) = last_match {
            tokens.push(token_id);
        } else if start < effective_word.len() {
            tokens.push(self.unk_id);
        }
    }
}
```

**Effort:** 2 weeks
**Expected Improvement:** 3-4x over current WordPiece

### 3.3 Memory-Efficient Unigram Viterbi

**Goal:** Implement O(n) memory Viterbi from SentencePiece

```rust
// src/models/unigram_viterbi.rs

/// Memory-efficient Unigram tokenizer with O(n) Viterbi
pub struct EfficientUnigramTokenizer {
    /// Vocabulary trie for fast prefix matching
    trie: DoubleArrayTrie,

    /// Log probabilities indexed by token ID
    scores: Vec<f32>,  // Use f32 for cache efficiency

    /// Unknown token handling
    unk_id: u32,
    unk_score: f32,

    /// Byte fallback tokens (optional)
    byte_fallback: Option<[u32; 256]>,
}

impl EfficientUnigramTokenizer {
    /// O(n) Viterbi without storing full lattice
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let n = bytes.len();

        if n == 0 { return Vec::new(); }

        // O(n) space: only store best ending at each position
        let mut best_score = vec![f32::NEG_INFINITY; n + 1];
        let mut best_edge = vec![(0usize, 0u32); n + 1];  // (start, token_id)

        best_score[0] = 0.0;

        // Forward pass
        for end in 1..=n {
            // Find all tokens ending at position `end`
            self.find_tokens_ending_at(bytes, end, |start, token_id, score| {
                let candidate = best_score[start] + score;
                if candidate > best_score[end] {
                    best_score[end] = candidate;
                    best_edge[end] = (start, token_id);
                }
            });

            // Byte fallback if no token found
            if best_score[end] == f32::NEG_INFINITY {
                if let Some(byte_tokens) = &self.byte_fallback {
                    let byte = bytes[end - 1];
                    let token_id = byte_tokens[byte as usize];
                    let score = self.scores[token_id as usize];
                    let candidate = best_score[end - 1] + score;
                    if candidate > best_score[end] {
                        best_score[end] = candidate;
                        best_edge[end] = (end - 1, token_id);
                    }
                }
            }
        }

        // Backward pass: reconstruct path
        let mut tokens = Vec::with_capacity(n / 3);
        let mut pos = n;
        while pos > 0 {
            let (start, token_id) = best_edge[pos];
            tokens.push(token_id);
            pos = start;
        }
        tokens.reverse();
        tokens
    }

    /// Efficiently find all tokens ending at position `end`
    #[inline]
    fn find_tokens_ending_at<F>(&self, bytes: &[u8], end: usize, mut callback: F)
    where
        F: FnMut(usize, u32, f32),
    {
        // Use trie to find all matching prefixes ending at `end`
        // Iterate backwards from `end` with early termination

        let max_len = std::cmp::min(end, self.max_token_len());

        for len in 1..=max_len {
            let start = end - len;
            let slice = &bytes[start..end];

            // Trie lookup
            if let Some(token_id) = self.trie.exact_match(slice) {
                let score = self.scores[token_id as usize];
                callback(start, token_id, score);
            }
        }
    }
}
```

**Effort:** 1.5 weeks
**Expected Improvement:** 2-3x over current Unigram

---

## 4. Phase 2: SIMD Acceleration

### 4.1 SIMD Dispatcher Architecture

```rust
// src/simd/mod.rs

/// SIMD capability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    Scalar = 0,
    Sse2 = 1,
    Avx2 = 2,
    Avx512 = 3,
    #[cfg(target_arch = "aarch64")]
    Neon = 4,
}

/// Detect best available SIMD level at runtime
pub fn detect_simd_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vbmi2") {
            return SimdLevel::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdLevel::Avx2;
        }
        if is_x86_feature_detected!("sse2") {
            return SimdLevel::Sse2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return SimdLevel::Neon;
    }

    SimdLevel::Scalar
}

/// Thread-local SIMD level cache
thread_local! {
    static SIMD_LEVEL: SimdLevel = detect_simd_level();
}

/// Dispatch to appropriate SIMD implementation
macro_rules! simd_dispatch {
    ($func:ident, $($arg:expr),*) => {
        SIMD_LEVEL.with(|level| {
            match *level {
                SimdLevel::Avx512 => avx512::$func($($arg),*),
                SimdLevel::Avx2 => avx2::$func($($arg),*),
                SimdLevel::Neon => neon::$func($($arg),*),
                _ => scalar::$func($($arg),*),
            }
        })
    };
}
```

### 4.2 AVX2 Character Classification

```rust
// src/simd/avx2.rs

use std::arch::x86_64::*;

/// AVX2-accelerated character classification for pre-tokenization
#[target_feature(enable = "avx2")]
pub unsafe fn classify_chars_avx2(text: &[u8]) -> Vec<CharClass> {
    let n = text.len();
    let mut result = Vec::with_capacity(n);

    // Process 32 bytes at a time
    let mut i = 0;

    // Precomputed lookup tables
    let whitespace_low = _mm256_set_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0x80, 0x80, 0, 0x80, 0x80, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0x80, 0x80, 0, 0x80, 0x80, 0, 0,
    );

    while i + 32 <= n {
        let chunk = _mm256_loadu_si256(text.as_ptr().add(i) as *const __m256i);

        // Classify ASCII characters using SIMD
        let is_whitespace = classify_whitespace_avx2(chunk);
        let is_letter = classify_letter_avx2(chunk);
        let is_digit = classify_digit_avx2(chunk);
        let is_punct = classify_punct_avx2(chunk);
        let is_ascii = _mm256_cmpgt_epi8(_mm256_set1_epi8(0), chunk); // MSB not set

        // Store classification results
        let ws_mask = _mm256_movemask_epi8(is_whitespace);
        let letter_mask = _mm256_movemask_epi8(is_letter);
        let digit_mask = _mm256_movemask_epi8(is_digit);
        let punct_mask = _mm256_movemask_epi8(is_punct);

        // Unpack masks to result vector
        for j in 0..32 {
            let bit = 1u32 << j;
            if ws_mask & bit != 0 {
                result.push(CharClass::Whitespace);
            } else if letter_mask & bit != 0 {
                result.push(CharClass::Letter);
            } else if digit_mask & bit != 0 {
                result.push(CharClass::Digit);
            } else if punct_mask & bit != 0 {
                result.push(CharClass::Punctuation);
            } else {
                result.push(CharClass::Other);
            }
        }

        i += 32;
    }

    // Handle remainder with scalar
    while i < n {
        result.push(classify_char_scalar(text[i]));
        i += 1;
    }

    result
}

/// Classify whitespace using SIMD shuffle
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn classify_whitespace_avx2(chunk: __m256i) -> __m256i {
    // Whitespace: 0x09 (tab), 0x0A (LF), 0x0D (CR), 0x20 (space)
    let tab = _mm256_set1_epi8(0x09);
    let lf = _mm256_set1_epi8(0x0A);
    let cr = _mm256_set1_epi8(0x0D);
    let space = _mm256_set1_epi8(0x20);

    let is_tab = _mm256_cmpeq_epi8(chunk, tab);
    let is_lf = _mm256_cmpeq_epi8(chunk, lf);
    let is_cr = _mm256_cmpeq_epi8(chunk, cr);
    let is_space = _mm256_cmpeq_epi8(chunk, space);

    _mm256_or_si256(
        _mm256_or_si256(is_tab, is_lf),
        _mm256_or_si256(is_cr, is_space),
    )
}

/// Classify ASCII letters [A-Za-z]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn classify_letter_avx2(chunk: __m256i) -> __m256i {
    // Convert to uppercase by clearing bit 5
    let upper = _mm256_and_si256(chunk, _mm256_set1_epi8(!0x20));

    // Check if in range [A-Z] (0x41-0x5A)
    let ge_a = _mm256_cmpgt_epi8(upper, _mm256_set1_epi8(0x40));
    let le_z = _mm256_cmpgt_epi8(_mm256_set1_epi8(0x5B), upper);

    _mm256_and_si256(ge_a, le_z)
}
```

### 4.3 AVX-512 UTF-8 Validation

```rust
// src/simd/avx512.rs

use std::arch::x86_64::*;

/// Ultra-fast UTF-8 validation using AVX-512
/// Based on Daniel Lemire's research
#[target_feature(enable = "avx512bw", enable = "avx512vbmi2")]
pub unsafe fn validate_utf8_avx512(bytes: &[u8]) -> bool {
    let n = bytes.len();
    let mut i = 0;

    // Error accumulator
    let mut error = _mm512_setzero_si512();

    // Previous chunk state for continuation validation
    let mut prev_incomplete = _mm512_setzero_si512();

    while i + 64 <= n {
        let chunk = _mm512_loadu_si512(bytes.as_ptr().add(i) as *const __m512i);

        // Check for ASCII fast path (all bytes < 128)
        let high_bits = _mm512_movepi8_mask(chunk);
        if high_bits == 0 {
            // Pure ASCII - skip ahead
            i += 64;
            prev_incomplete = _mm512_setzero_si512();
            continue;
        }

        // Full UTF-8 validation
        let validation_error = validate_utf8_chunk_avx512(
            chunk,
            &mut prev_incomplete,
        );

        error = _mm512_or_si512(error, validation_error);
        i += 64;
    }

    // Check for accumulated errors
    let has_error = _mm512_test_epi8_mask(error, error) != 0;

    // Validate remainder with scalar
    let remainder_valid = validate_utf8_scalar(&bytes[i..]);

    !has_error && remainder_valid
}

#[inline]
#[target_feature(enable = "avx512bw")]
unsafe fn validate_utf8_chunk_avx512(
    chunk: __m512i,
    prev_incomplete: &mut __m512i,
) -> __m512i {
    // Classify bytes by leading bits
    let byte_1_high = _mm512_cmpge_epu8_mask(chunk, _mm512_set1_epi8(0xC0u8 as i8));
    let byte_2_high = _mm512_cmpge_epu8_mask(chunk, _mm512_set1_epi8(0xE0u8 as i8));
    let byte_3_high = _mm512_cmpge_epu8_mask(chunk, _mm512_set1_epi8(0xF0u8 as i8));

    // Continuation bytes (10xxxxxx)
    let is_cont = _mm512_mask_blend_epi8(
        _mm512_cmpge_epu8_mask(chunk, _mm512_set1_epi8(0x80u8 as i8)),
        _mm512_setzero_si512(),
        _mm512_set1_epi8(0xFF_u8 as i8),
    );
    let is_cont = _mm512_andnot_si512(
        _mm512_mask_blend_epi8(
            _mm512_cmpge_epu8_mask(chunk, _mm512_set1_epi8(0xC0u8 as i8)),
            _mm512_setzero_si512(),
            _mm512_set1_epi8(0xFF_u8 as i8),
        ),
        is_cont,
    );

    // Validate continuation byte sequences
    // ... (full implementation continues)

    _mm512_setzero_si512()  // Return error mask
}
```

### 4.4 SIMD Pre-Tokenization

```rust
// src/simd/pretokenize.rs

/// SIMD-accelerated whitespace splitting
#[target_feature(enable = "avx2")]
pub unsafe fn split_whitespace_simd(text: &str) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let n = bytes.len();

    let mut spans = Vec::with_capacity(n / 5);  // Estimate ~5 chars per word
    let mut word_start: Option<usize> = None;
    let mut i = 0;

    // Whitespace detection masks
    let space = _mm256_set1_epi8(b' ' as i8);
    let tab = _mm256_set1_epi8(b'\t' as i8);
    let newline = _mm256_set1_epi8(b'\n' as i8);
    let cr = _mm256_set1_epi8(b'\r' as i8);

    while i + 32 <= n {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);

        // Detect whitespace characters
        let is_space = _mm256_cmpeq_epi8(chunk, space);
        let is_tab = _mm256_cmpeq_epi8(chunk, tab);
        let is_nl = _mm256_cmpeq_epi8(chunk, newline);
        let is_cr = _mm256_cmpeq_epi8(chunk, cr);

        let is_ws = _mm256_or_si256(
            _mm256_or_si256(is_space, is_tab),
            _mm256_or_si256(is_nl, is_cr),
        );

        let ws_mask = _mm256_movemask_epi8(is_ws) as u32;

        // Find word boundaries using bit manipulation
        if ws_mask == 0 {
            // No whitespace - entire chunk is part of word
            if word_start.is_none() {
                word_start = Some(i);
            }
        } else if ws_mask == 0xFFFFFFFF {
            // All whitespace - end any current word
            if let Some(start) = word_start.take() {
                spans.push((start, i));
            }
        } else {
            // Mixed - process bit by bit
            for j in 0..32 {
                let is_whitespace = (ws_mask >> j) & 1 != 0;

                if is_whitespace {
                    if let Some(start) = word_start.take() {
                        spans.push((start, i + j));
                    }
                } else if word_start.is_none() {
                    word_start = Some(i + j);
                }
            }
        }

        i += 32;
    }

    // Handle remainder
    while i < n {
        let is_whitespace = bytes[i].is_ascii_whitespace();

        if is_whitespace {
            if let Some(start) = word_start.take() {
                spans.push((start, i));
            }
        } else if word_start.is_none() {
            word_start = Some(i);
        }

        i += 1;
    }

    // Final word
    if let Some(start) = word_start {
        spans.push((start, n));
    }

    spans
}
```

**Effort:** 3 weeks total for SIMD implementations
**Expected Improvement:** 2-3x additional speedup

---

## 5. Phase 3: Multi-Core Parallelization

### 5.1 Adaptive Batch Processing

```rust
// src/parallel/batch.rs

use rayon::prelude::*;

/// Configuration for parallel batch encoding
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Minimum texts per thread to avoid overhead
    pub min_texts_per_thread: usize,

    /// Whether to use adaptive chunking
    pub adaptive: bool,

    /// Whether to sort by length for load balancing
    pub length_sorted: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            min_texts_per_thread: 8,
            adaptive: true,
            length_sorted: true,
        }
    }
}

/// Parallel batch encoder with optimal load balancing
pub struct ParallelBatchEncoder<T: Tokenizer + Sync> {
    tokenizer: T,
    config: BatchConfig,
}

impl<T: Tokenizer + Sync> ParallelBatchEncoder<T> {
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Encoding> {
        let n = texts.len();
        let num_threads = rayon::current_num_threads();

        // Single-threaded for small batches
        if n < self.config.min_texts_per_thread * 2 {
            return texts.iter()
                .map(|t| self.tokenizer.encode(t))
                .collect();
        }

        if self.config.length_sorted {
            // Sort indices by text length for better load balancing
            self.encode_length_sorted(texts)
        } else {
            self.encode_simple_parallel(texts)
        }
    }

    fn encode_length_sorted(&self, texts: &[&str]) -> Vec<Encoding> {
        // Create (index, length) pairs and sort by length
        let mut indexed: Vec<(usize, usize)> = texts.iter()
            .enumerate()
            .map(|(i, t)| (i, t.len()))
            .collect();

        // Sort by length descending (longest first)
        indexed.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        // Encode in parallel with work stealing
        let results: Vec<(usize, Encoding)> = indexed
            .par_iter()
            .map(|(i, _)| (*i, self.tokenizer.encode(texts[*i])))
            .collect();

        // Restore original order
        let mut ordered = vec![Encoding::default(); texts.len()];
        for (i, encoding) in results {
            ordered[i] = encoding;
        }

        ordered
    }

    fn encode_simple_parallel(&self, texts: &[&str]) -> Vec<Encoding> {
        texts
            .par_iter()
            .with_min_len(self.config.min_texts_per_thread)
            .map(|t| self.tokenizer.encode(t))
            .collect()
    }
}
```

### 5.2 Cache-Aligned Data Structures

```rust
// src/parallel/cache_aligned.rs

use std::cell::UnsafeCell;

/// Cache line size (64 bytes on most modern CPUs)
const CACHE_LINE_SIZE: usize = 64;

/// Cache-line aligned wrapper to prevent false sharing
#[repr(align(64))]
pub struct CacheAligned<T>(pub T);

impl<T: Default> Default for CacheAligned<T> {
    fn default() -> Self {
        Self(T::default())
    }
}

/// Per-thread encoding workspace to avoid allocation
#[repr(align(64))]
pub struct ThreadLocalWorkspace {
    /// Reusable token buffer
    tokens: Vec<u32>,

    /// Reusable score buffer (for Unigram)
    scores: Vec<f32>,

    /// Reusable string buffer
    string_buf: String,

    /// Padding to ensure cache line separation
    _pad: [u8; 16],
}

impl ThreadLocalWorkspace {
    pub fn new() -> Self {
        Self {
            tokens: Vec::with_capacity(1024),
            scores: Vec::with_capacity(1024),
            string_buf: String::with_capacity(4096),
            _pad: [0; 16],
        }
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.scores.clear();
        self.string_buf.clear();
    }
}

/// Thread-local workspace pool
pub struct WorkspacePool {
    workspaces: Vec<CacheAligned<UnsafeCell<ThreadLocalWorkspace>>>,
}

impl WorkspacePool {
    pub fn new(num_threads: usize) -> Self {
        Self {
            workspaces: (0..num_threads)
                .map(|_| CacheAligned(UnsafeCell::new(ThreadLocalWorkspace::new())))
                .collect(),
        }
    }

    /// Get workspace for current thread (lock-free)
    pub fn get(&self, thread_id: usize) -> &mut ThreadLocalWorkspace {
        unsafe {
            &mut *self.workspaces[thread_id % self.workspaces.len()].0.get()
        }
    }
}
```

### 5.3 Lock-Free Vocabulary Access

```rust
// src/parallel/vocab.rs

use parking_lot::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Lock-free vocabulary with optimistic reads
pub struct ConcurrentVocabulary {
    /// Main vocabulary (read-heavy, rarely modified)
    vocab: RwLock<FxHashMap<Vec<u8>, u32>>,

    /// Read counter for statistics
    read_count: AtomicUsize,

    /// Small hot cache for frequent tokens
    hot_cache: [AtomicU64; 256],  // Key hash -> packed (hash, token_id)
}

impl ConcurrentVocabulary {
    /// Lock-free lookup with hot cache
    #[inline]
    pub fn lookup(&self, token: &[u8]) -> Option<u32> {
        // Try hot cache first (lock-free)
        let hash = fxhash::hash64(token);
        let cache_idx = (hash as usize) % self.hot_cache.len();

        let cached = self.hot_cache[cache_idx].load(Ordering::Relaxed);
        if cached != 0 {
            let cached_hash = cached >> 32;
            let cached_id = cached as u32;

            if cached_hash as u64 == (hash >> 32) {
                self.read_count.fetch_add(1, Ordering::Relaxed);
                return Some(cached_id);
            }
        }

        // Fall back to main vocabulary (read lock)
        let vocab = self.vocab.read();
        let result = vocab.get(token).copied();

        // Update hot cache on hit
        if let Some(id) = result {
            let packed = ((hash >> 32) << 32) | (id as u64);
            self.hot_cache[cache_idx].store(packed, Ordering::Relaxed);
        }

        result
    }
}
```

**Effort:** 2 weeks
**Expected Improvement:** Near-linear scaling with cores

---

## 6. Phase 4: Hardware Autodetection

### 6.1 Comprehensive CPU Detection

```rust
// src/simd/detect.rs

use std::sync::OnceLock;

/// Detected CPU capabilities
#[derive(Debug, Clone)]
pub struct CpuCapabilities {
    // x86_64 features
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vbmi: bool,
    pub avx512vbmi2: bool,

    // ARM features
    pub neon: bool,
    pub sve: bool,
    pub sve2: bool,

    // Cache info
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub l3_cache_size: usize,
    pub cache_line_size: usize,

    // Core info
    pub physical_cores: usize,
    pub logical_cores: usize,
}

static CPU_CAPS: OnceLock<CpuCapabilities> = OnceLock::new();

impl CpuCapabilities {
    /// Detect CPU capabilities (cached after first call)
    pub fn detect() -> &'static Self {
        CPU_CAPS.get_or_init(|| Self::detect_inner())
    }

    fn detect_inner() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                sse2: is_x86_feature_detected!("sse2"),
                sse3: is_x86_feature_detected!("sse3"),
                ssse3: is_x86_feature_detected!("ssse3"),
                sse4_1: is_x86_feature_detected!("sse4.1"),
                sse4_2: is_x86_feature_detected!("sse4.2"),
                avx: is_x86_feature_detected!("avx"),
                avx2: is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                avx512bw: is_x86_feature_detected!("avx512bw"),
                avx512vbmi: is_x86_feature_detected!("avx512vbmi"),
                avx512vbmi2: is_x86_feature_detected!("avx512vbmi2"),
                neon: false,
                sve: false,
                sve2: false,
                l1_cache_size: Self::detect_cache_size(1),
                l2_cache_size: Self::detect_cache_size(2),
                l3_cache_size: Self::detect_cache_size(3),
                cache_line_size: Self::detect_cache_line_size(),
                physical_cores: num_cpus::get_physical(),
                logical_cores: num_cpus::get(),
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                sse2: false,
                sse3: false,
                ssse3: false,
                sse4_1: false,
                sse4_2: false,
                avx: false,
                avx2: false,
                avx512f: false,
                avx512bw: false,
                avx512vbmi: false,
                avx512vbmi2: false,
                neon: true,  // Always available on aarch64
                sve: std::arch::is_aarch64_feature_detected!("sve"),
                sve2: std::arch::is_aarch64_feature_detected!("sve2"),
                l1_cache_size: Self::detect_cache_size(1),
                l2_cache_size: Self::detect_cache_size(2),
                l3_cache_size: Self::detect_cache_size(3),
                cache_line_size: Self::detect_cache_line_size(),
                physical_cores: num_cpus::get_physical(),
                logical_cores: num_cpus::get(),
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::fallback()
        }
    }

    /// Get optimal SIMD width for this CPU
    pub fn optimal_simd_width(&self) -> usize {
        if self.avx512bw { 64 }
        else if self.avx2 { 32 }
        else if self.neon { 16 }
        else if self.sse2 { 16 }
        else { 1 }
    }

    /// Get recommended number of threads
    pub fn recommended_threads(&self) -> usize {
        // Use physical cores for CPU-bound work
        self.physical_cores
    }

    /// Get optimal chunk size for parallel processing
    pub fn optimal_chunk_size(&self) -> usize {
        // Aim for each chunk to fit in L2 cache
        std::cmp::max(self.l2_cache_size / 4, 64 * 1024)
    }

    #[cfg(target_os = "linux")]
    fn detect_cache_size(level: u8) -> usize {
        // Read from sysfs
        let path = format!(
            "/sys/devices/system/cpu/cpu0/cache/index{}/size",
            level - 1
        );
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| {
                let s = s.trim();
                if s.ends_with('K') {
                    s[..s.len()-1].parse::<usize>().ok().map(|n| n * 1024)
                } else if s.ends_with('M') {
                    s[..s.len()-1].parse::<usize>().ok().map(|n| n * 1024 * 1024)
                } else {
                    s.parse().ok()
                }
            })
            .unwrap_or(match level {
                1 => 32 * 1024,
                2 => 256 * 1024,
                3 => 8 * 1024 * 1024,
                _ => 0,
            })
    }

    fn detect_cache_line_size() -> usize {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string(
                "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
            )
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(64)
        }

        #[cfg(not(target_os = "linux"))]
        { 64 }  // Safe default
    }
}
```

### 6.2 Dynamic Dispatch Table

```rust
// src/simd/dispatch.rs

use super::detect::CpuCapabilities;

/// Function pointer type for tokenization
type EncodeFn = fn(&[u8], &Vocabulary) -> Vec<u32>;
type PreTokenizeFn = unsafe fn(&str) -> Vec<(usize, usize)>;
type ValidateUtf8Fn = unsafe fn(&[u8]) -> bool;

/// Dispatch table with resolved function pointers
pub struct DispatchTable {
    pub encode_bpe: EncodeFn,
    pub encode_wordpiece: EncodeFn,
    pub pre_tokenize: PreTokenizeFn,
    pub validate_utf8: ValidateUtf8Fn,
    pub simd_level: &'static str,
}

impl DispatchTable {
    /// Build dispatch table based on CPU capabilities
    pub fn build() -> Self {
        let caps = CpuCapabilities::detect();

        #[cfg(target_arch = "x86_64")]
        {
            if caps.avx512vbmi2 {
                return Self::avx512();
            }
            if caps.avx2 {
                return Self::avx2();
            }
            if caps.sse2 {
                return Self::sse2();
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if caps.neon {
                return Self::neon();
            }
        }

        Self::scalar()
    }

    fn scalar() -> Self {
        Self {
            encode_bpe: scalar::encode_bpe,
            encode_wordpiece: scalar::encode_wordpiece,
            pre_tokenize: scalar::pre_tokenize,
            validate_utf8: scalar::validate_utf8,
            simd_level: "scalar",
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn avx2() -> Self {
        Self {
            encode_bpe: avx2::encode_bpe,
            encode_wordpiece: avx2::encode_wordpiece,
            pre_tokenize: avx2::pre_tokenize,
            validate_utf8: avx2::validate_utf8,
            simd_level: "avx2",
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn avx512() -> Self {
        Self {
            encode_bpe: avx512::encode_bpe,
            encode_wordpiece: avx512::encode_wordpiece,
            pre_tokenize: avx512::pre_tokenize,
            validate_utf8: avx512::validate_utf8,
            simd_level: "avx512",
        }
    }
}

/// Global dispatch table (initialized once)
static DISPATCH: OnceLock<DispatchTable> = OnceLock::new();

pub fn dispatch() -> &'static DispatchTable {
    DISPATCH.get_or_init(DispatchTable::build)
}
```

**Effort:** 1 week
**Expected Improvement:** Automatic optimal path selection

---

## 7. Phase 5: Complete Feature Parity

### 7.1 Missing Components Checklist

| Component | Priority | Effort | Status |
|-----------|----------|--------|--------|
| **Decoders** | | | |
| ByteLevelDecoder | Critical | 4h | TODO |
| MetaspaceDecoder | Critical | 2h | TODO |
| WordPieceDecoder | Critical | 2h | TODO |
| SequenceDecoder | Critical | 1h | TODO |
| BPEDecoder | High | 2h | TODO |
| ByteFallbackDecoder | Medium | 2h | TODO |
| CTCDecoder | Low | 4h | TODO |
| **Pre-Tokenizers** | | | |
| Split (full regex) | High | 4h | Partial |
| CharDelimiterSplit | Medium | 1h | TODO |
| UnicodeScripts | Low | 4h | TODO |
| **Normalizers** | | | |
| Strip | Low | 1h | TODO |
| Precompiled | Low | 4h | TODO |
| **Encoding Methods** | | | |
| char_to_token() | Medium | 1h | TODO |
| char_to_word() | Medium | 1h | TODO |
| token_to_chars() | Medium | 1h | TODO |
| token_to_word() | Medium | 1h | TODO |
| word_to_chars() | Medium | 1h | TODO |
| word_to_tokens() | Medium | 1h | TODO |
| **Padding/Truncation** | | | |
| Left padding | Medium | 2h | TODO |
| Left truncation | Medium | 2h | TODO |
| Batch padding | Medium | 4h | TODO |
| pad_to_multiple_of | Low | 1h | TODO |
| **Serialization** | | | |
| Complete save() | High | 8h | TODO |
| from_pretrained() | Medium | 16h | TODO |
| **API** | | | |
| encode_pair() | High | 4h | TODO |
| decode_batch() | Medium | 2h | TODO |
| enable_padding() | Medium | 2h | TODO |
| enable_truncation() | Medium | 2h | TODO |

### 7.2 Decoder Implementation

```rust
// src/decoders/mod.rs

/// Trait for token-to-text decoders
pub trait Decoder: Send + Sync {
    /// Decode a sequence of tokens to text
    fn decode(&self, tokens: &[String]) -> String;

    /// Decode keeping tokens separate (for chaining)
    fn decode_chain(&self, tokens: &[String]) -> Vec<String> {
        tokens.to_vec()  // Default: no-op
    }
}

/// Byte-level decoder (GPT-2 style)
pub struct ByteLevelDecoder;

impl Decoder for ByteLevelDecoder {
    fn decode(&self, tokens: &[String]) -> String {
        let bytes: Vec<u8> = tokens.iter()
            .flat_map(|t| t.chars().map(char_to_byte))
            .collect();

        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Reverse GPT-2 byte mapping
fn char_to_byte(c: char) -> u8 {
    let cp = c as u32;

    match cp {
        // Direct mapping range
        0x21..=0x7E | 0xA1..=0xFF => cp as u8,

        // Mapped range (U+0100+)
        0x100..=0x120 => (cp - 0x100) as u8,          // 0x00-0x20
        0x121..=0x142 => (cp - 0x121 + 0x7F) as u8,   // 0x7F-0xA0

        _ => b'?',  // Unknown
    }
}

/// Metaspace decoder (SentencePiece style)
pub struct MetaspaceDecoder {
    replacement: char,
    add_prefix_space: bool,
}

impl Default for MetaspaceDecoder {
    fn default() -> Self {
        Self {
            replacement: '\u{2581}',  // ▁
            add_prefix_space: true,
        }
    }
}

impl Decoder for MetaspaceDecoder {
    fn decode(&self, tokens: &[String]) -> String {
        let mut result: String = tokens.join("");

        // Replace metaspace with space
        result = result.replace(self.replacement, " ");

        // Remove leading space if added
        if self.add_prefix_space && result.starts_with(' ') {
            result.remove(0);
        }

        result
    }
}

/// Sequence decoder: chain multiple decoders
pub struct SequenceDecoder {
    decoders: Vec<Box<dyn Decoder>>,
}

impl Decoder for SequenceDecoder {
    fn decode(&self, tokens: &[String]) -> String {
        let mut current = tokens.to_vec();

        for decoder in &self.decoders {
            current = decoder.decode_chain(&current);
        }

        current.join("")
    }

    fn decode_chain(&self, tokens: &[String]) -> Vec<String> {
        let mut current = tokens.to_vec();

        for decoder in &self.decoders {
            current = decoder.decode_chain(&current);
        }

        current
    }
}
```

**Effort:** 3 weeks for complete feature parity

---

## 8. Phase 6: Accuracy Verification

### 8.1 Comprehensive Test Suite

```rust
// tests/hf_compatibility/mod.rs

use tokenizers::Tokenizer as HfTokenizer;
use budtiktok::Tokenizer as BudTokenizer;
use proptest::prelude::*;

/// Test 100% accuracy against HuggingFace
#[test]
fn test_exact_accuracy() {
    let test_cases = load_test_corpus();  // 10K+ diverse texts

    let hf = HfTokenizer::from_file("tokenizer.json").unwrap();
    let bud = BudTokenizer::from_file("tokenizer.json").unwrap();

    let mut mismatches = 0;

    for (i, text) in test_cases.iter().enumerate() {
        let hf_result = hf.encode(text, false).unwrap();
        let bud_result = bud.encode(text, false).unwrap();

        if hf_result.get_ids() != bud_result.get_ids() {
            println!("MISMATCH #{}: {:?}", i, text);
            println!("  HF:  {:?}", hf_result.get_ids());
            println!("  Bud: {:?}", bud_result.get_ids());
            mismatches += 1;
        }

        // Also verify other fields
        assert_eq!(
            hf_result.get_offsets(),
            bud_result.get_offsets(),
            "Offset mismatch for: {:?}", text
        );

        assert_eq!(
            hf_result.get_attention_mask(),
            bud_result.get_attention_mask(),
            "Attention mask mismatch for: {:?}", text
        );
    }

    assert_eq!(mismatches, 0, "Found {} mismatches", mismatches);
}

/// Property-based testing for edge cases
proptest! {
    #![proptest_config(ProptestConfig::with_cases(10000))]

    #[test]
    fn prop_roundtrip(text in ".*") {
        let tokenizer = BudTokenizer::from_file("tokenizer.json").unwrap();

        let encoded = tokenizer.encode(&text, false).unwrap();
        let decoded = tokenizer.decode(encoded.get_ids(), false).unwrap();

        // Decoded should match original (modulo normalization)
        let normalized_input = tokenizer.normalizer().normalize(&text);
        prop_assert_eq!(decoded, normalized_input);
    }

    #[test]
    fn prop_hf_match(text in "[a-zA-Z0-9 .,!?]{1,100}") {
        let hf = HfTokenizer::from_file("tokenizer.json").unwrap();
        let bud = BudTokenizer::from_file("tokenizer.json").unwrap();

        let hf_enc = hf.encode(&text, false).unwrap();
        let bud_enc = bud.encode(&text, false).unwrap();

        prop_assert_eq!(hf_enc.get_ids(), bud_enc.get_ids());
    }
}
```

### 8.2 Edge Case Coverage

```rust
// tests/edge_cases/mod.rs

/// Test Unicode edge cases
#[test]
fn test_unicode_edge_cases() {
    let cases = vec![
        // Combining characters
        ("café", vec!["cafe\u{0301}"]),
        ("g\u{0336}r\u{0336}o\u{0336}s\u{0336}s", vec!["g̶r̶o̶s̶s̶"]),

        // Zero-width characters
        ("a\u{200D}b", vec!["a\u{200D}b"]),  // ZWJ
        ("a\u{200C}b", vec!["a\u{200C}b"]),  // ZWNJ

        // Bidirectional text
        ("\u{202E}secret", vec!["\u{202E}secret"]),  // RTL override

        // Emoji with modifiers
        ("👨\u{200D}👩\u{200D}👧", vec!["👨‍👩‍👧"]),

        // CJK
        ("你好世界", vec!["你", "好", "世", "界"]),

        // Mixed scripts
        ("Hello世界", vec!["Hello", "世", "界"]),

        // Subscripts/superscripts
        ("H₂O", vec!["H", "₂", "O"]),

        // Empty and whitespace
        ("", vec![]),
        ("   ", vec![]),
        ("\t\n\r", vec![]),

        // Very long words
        (&"a".repeat(10000), vec![&"a".repeat(10000)]),
    ];

    let tokenizer = setup_tokenizer();

    for (input, expected_tokens) in cases {
        let result = tokenizer.encode(input, false).unwrap();
        assert_eq!(
            result.get_tokens().len(),
            expected_tokens.len(),
            "Token count mismatch for: {:?}",
            input
        );
    }
}

/// Test format character handling
#[test]
fn test_format_characters() {
    // These are the 4 edge cases from the WordLevel benchmark
    let hf = HfTokenizer::from_file("tokenizer.json").unwrap();
    let bud = BudTokenizer::from_file("tokenizer.json").unwrap();

    let problematic = vec![
        "test\u{200C}text",  // ZWNJ
        "test\u{200D}text",  // ZWJ
        "test\u{202E}text",  // RTL Override
        "test\u{FEFF}text",  // BOM
    ];

    for text in problematic {
        let hf_result = hf.encode(text, false).unwrap();
        let bud_result = bud.encode(text, false).unwrap();

        assert_eq!(
            hf_result.get_ids(),
            bud_result.get_ids(),
            "Mismatch for format char: {:?}",
            text
        );
    }
}
```

### 8.3 Continuous Accuracy Verification

```rust
// benches/accuracy_continuous.rs

/// Accuracy verification that runs with every benchmark
pub fn verify_accuracy_sample(tokenizer: &impl Tokenizer, sample_size: usize) -> f64 {
    let hf = HfTokenizer::from_file("tokenizer.json").unwrap();
    let test_texts = load_random_sample(sample_size);

    let mut correct = 0;

    for text in &test_texts {
        let hf_ids = hf.encode(text, false).unwrap().get_ids().to_vec();
        let our_ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();

        if hf_ids == our_ids {
            correct += 1;
        }
    }

    (correct as f64) / (sample_size as f64) * 100.0
}

/// Fail benchmark if accuracy drops below 100%
#[bench]
fn bench_with_accuracy_check(b: &mut Bencher) {
    let tokenizer = BudTokenizer::from_file("tokenizer.json").unwrap();

    // Verify accuracy first
    let accuracy = verify_accuracy_sample(&tokenizer, 1000);
    if accuracy < 100.0 {
        panic!("Accuracy dropped to {:.2}% - FAILING BENCHMARK", accuracy);
    }

    // Then run performance benchmark
    b.iter(|| tokenizer.encode("test text", false));
}
```

**Effort:** 1 week for comprehensive test suite
**Goal:** 100% accuracy on all test cases

---

## 9. Implementation Timeline

### Phase Overview

```
Week 1-2:   Phase 1A - Linear BPE Implementation
Week 3-4:   Phase 1B - LinMaxMatch WordPiece
Week 5:     Phase 1C - Memory-Efficient Unigram
Week 6-7:   Phase 2A - AVX2 SIMD (core functions)
Week 8:     Phase 2B - AVX-512 SIMD
Week 9:     Phase 3A - Parallel Batch Processing
Week 10:    Phase 3B - Cache Optimization
Week 11:    Phase 4 - Hardware Autodetection
Week 12-13: Phase 5A - Decoders
Week 14:    Phase 5B - Missing Components
Week 15:    Phase 6 - Accuracy Verification
Week 16:    Integration & Polish
```

### Milestones

| Milestone | Week | Deliverables |
|-----------|------|--------------|
| M1: Core Algorithms | 5 | O(n) BPE, WordPiece, Unigram |
| M2: SIMD Complete | 8 | AVX2/AVX-512/NEON acceleration |
| M3: Parallel Complete | 10 | Multi-core with work-stealing |
| M4: Feature Parity | 14 | All HF features implemented |
| M5: 100% Accuracy | 15 | Full test suite passing |
| M6: Release Ready | 16 | Documentation, benchmarks |

### Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Algorithm complexity | Medium | High | Use proven algorithms (rust-gems, LinMaxMatch) |
| SIMD portability | Medium | Medium | Fallback to scalar, test on multiple CPUs |
| Accuracy edge cases | High | Critical | Extensive fuzz testing, HF comparison |
| Performance regression | Low | Medium | CI benchmarks, regression tests |

---

## 10. References

### Papers

1. **Linear-Time BPE**
   - GitHub Blog: [Faster BPE Tokenizer](https://github.blog/ai-and-ml/llms/so-many-tokens-so-little-time-introducing-a-faster-more-flexible-byte-pair-tokenizer/)
   - Code: [github/rust-gems](https://github.com/github/rust-gems)

2. **Fast WordPiece (LinMaxMatch)**
   - Paper: [EMNLP 2021](https://aclanthology.org/2021.emnlp-main.160/)
   - Song et al., "Fast WordPiece Tokenization"

3. **Double-Array Aho-Corasick**
   - Paper: [SPE 2023](https://onlinelibrary.wiley.com/doi/10.1002/spe.3190)
   - Kanda et al., "Engineering faster double-array Aho-Corasick automata"
   - Code: [daac-tools/daachorse](https://github.com/daac-tools/daachorse)

4. **SIMD UTF-8**
   - Paper: [SPE 2023](https://onlinelibrary.wiley.com/doi/full/10.1002/spe.3261)
   - Clausecker & Lemire, "Transcoding Unicode Characters with AVX-512"
   - Code: [simdutf/simdutf](https://github.com/simdutf/simdutf)

5. **Teddy Algorithm**
   - Documentation: [aho-corasick/teddy](https://github.com/BurntSushi/aho-corasick/blob/master/src/packed/teddy/README.md)
   - Origin: Intel Hyperscan

### Libraries

| Library | Purpose | Link |
|---------|---------|------|
| bpe | O(n) BPE | [crates.io/crates/bpe](https://crates.io/crates/bpe) |
| daachorse | Fast Aho-Corasick | [crates.io/crates/daachorse](https://crates.io/crates/daachorse) |
| simdutf | SIMD UTF-8 | [github.com/simdutf/simdutf](https://github.com/simdutf/simdutf) |
| memchr | SIMD search | [crates.io/crates/memchr](https://crates.io/crates/memchr) |
| rayon | Parallelism | [crates.io/crates/rayon](https://crates.io/crates/rayon) |

### Benchmarks

| Implementation | Throughput (single) | Throughput (multi) | vs HF |
|----------------|--------------------|--------------------|-------|
| HuggingFace | 8 MB/s | 46 MB/s | 1x |
| tiktoken | 24 MB/s | - | 3x |
| rust-gems bpe | 80 MB/s | - | 10x |
| **BudTikTok Target** | **100+ MB/s** | **800+ MB/s** | **12-20x** |

---

## Appendix A: Performance Optimization Checklist

### Algorithm Level
- [ ] O(n) BPE with bitfield tracking
- [ ] LinMaxMatch WordPiece with failure links
- [ ] O(n) memory Viterbi for Unigram
- [ ] Double-array trie for vocabulary

### SIMD Level
- [ ] AVX2 character classification
- [ ] AVX-512 UTF-8 validation
- [ ] SIMD whitespace splitting
- [ ] Vectorized byte search (memchr)

### Memory Level
- [ ] Cache-line aligned structures
- [ ] Thread-local workspaces
- [ ] Lock-free vocabulary access
- [ ] Minimize allocations

### Parallelism Level
- [ ] Work-stealing batch processing
- [ ] Length-sorted load balancing
- [ ] Adaptive chunk sizing
- [ ] False sharing prevention

### Build Level
- [ ] LTO enabled
- [ ] Native CPU target
- [ ] Profile-guided optimization
- [ ] Optimal codegen settings

---

## Appendix B: Accuracy Test Corpus

### Required Test Categories

1. **Basic ASCII** (10K samples)
   - English sentences
   - Code snippets
   - Numbers and punctuation

2. **Unicode** (10K samples)
   - Accented characters (café, naïve)
   - CJK text (Chinese, Japanese, Korean)
   - Arabic, Hebrew, Thai
   - Emoji (including ZWJ sequences)

3. **Edge Cases** (1K samples)
   - Empty strings
   - Whitespace-only
   - Single characters
   - Very long strings (100K+ chars)
   - Format characters (ZWJ, ZWNJ, RLO)

4. **Real-World** (10K samples)
   - Wikipedia articles
   - News articles
   - Social media posts
   - Code repositories

5. **Adversarial** (1K samples)
   - Malformed UTF-8 (handled gracefully)
   - Unusual Unicode combinations
   - Repeated patterns
   - Mixed scripts

---

*Document Version: 1.0*
*Last Updated: 2025-12-21*
*Authors: BudTikTok Team*
