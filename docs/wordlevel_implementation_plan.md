# WordLevel Tokenizer Implementation Plan for BudTikTok

## Executive Summary

This document outlines the research findings and implementation plan for a high-performance WordLevel tokenizer in BudTikTok, targeting both scalar and SIMD-optimized versions.

---

## 1. HuggingFace WordLevel Implementation Analysis

### 1.1 Core Algorithm

The [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) WordLevel implementation is remarkably simple:

```rust
pub struct WordLevel {
    vocab: AHashMap<String, u32>,      // token → ID
    vocab_r: AHashMap<u32, String>,    // ID → token (reverse)
    pub unk_token: String,              // unknown token
}

fn tokenize(&self, token: &str) -> Result<Vec<Token>> {
    if let Some(&id) = self.vocab.get(token) {
        Ok(vec![Token { id, value: token.to_owned(), offsets: (0, token.len()) }])
    } else if let Some(&unk_id) = self.vocab.get(&self.unk_token) {
        Ok(vec![Token { id: unk_id, value: self.unk_token.clone(), offsets: (0, token.len()) }])
    } else {
        Err(Box::new(Error::MissingUnkToken))
    }
}
```

### 1.2 Key Characteristics

| Aspect | HuggingFace Approach |
|--------|---------------------|
| **Vocabulary Lookup** | AHashMap (O(1) average) |
| **Pre-tokenization** | External (separate step) |
| **Unknown Handling** | Single UNK token fallback |
| **Memory** | Dual HashMap (forward + reverse) |
| **Complexity** | O(n) for n words (assumes pre-tokenized) |

### 1.3 Bottlenecks in HF Implementation

1. **Pre-tokenization is external** - Requires separate pass over text
2. **String allocations** - `token.to_owned()` allocates for every token
3. **No caching** - Repeated words are looked up every time
4. **No SIMD** - Character-by-character whitespace detection

---

## 2. Optimization Opportunities

### 2.1 Pre-tokenization (Text → Words)

This is the primary bottleneck for WordLevel tokenization. Key optimizations:

#### A. SIMD Whitespace Detection

Based on [StringZilla](https://ashvardanian.com/posts/splitting-strings-cpp/) and [simdjson](https://github.com/simdjson/simdjson) techniques:

```
Traditional:  for (c in text) { if is_whitespace(c) { split } }  → O(n), 1 byte/cycle
SIMD AVX-512: Load 64 bytes → parallel comparison → bitmask     → O(n), 64 bytes/cycle
```

**Technique: 256-slot Bitset Classification**

```rust
// Build bitset for delimiter chars (space, tab, newline, punctuation)
let delimiters: [u64; 4] = build_bitset(&[' ', '\t', '\n', '\r', '.', ',', '!', '?']);

// SIMD check: which bytes are delimiters?
fn find_delimiters_avx512(bytes: &[u8; 64], delimiters: &[u64; 4]) -> u64 {
    // Split each byte into nibbles
    // Use vpshufb for table lookup
    // Combine with AND to get match mask
}
```

**Performance**: [10x faster than C++ STL](https://ashvardanian.com/posts/splitting-strings-cpp/) for string splitting.

#### B. SWAR (SIMD Within A Register)

For platforms without AVX-512, process 8 bytes at a time using u64:

```rust
const LO: u64 = 0x0101_0101_0101_0101;
const HI: u64 = 0x8080_8080_8080_8080;

fn has_whitespace_u64(word: u64) -> bool {
    let space = word ^ (LO * b' ' as u64);
    let tab = word ^ (LO * b'\t' as u64);
    let newline = word ^ (LO * b'\n' as u64);

    (space.wrapping_sub(LO) & !space & HI) != 0 ||
    (tab.wrapping_sub(LO) & !tab & HI) != 0 ||
    (newline.wrapping_sub(LO) & !newline & HI) != 0
}
```

### 2.2 Vocabulary Lookup

#### A. Perfect Hash Functions

For static vocabularies, [perfect hash functions](https://mainmatter.com/blog/2022/06/23/the-perfect-hash-function/) provide:

| Library | Build Time (100K) | Lookup Time | Bits/Entry |
|---------|-------------------|-------------|------------|
| [rust-phf](https://github.com/rust-phf/rust-phf) | 0.4s | O(1) | ~10 |
| [quickphf](https://docs.rs/quickphf) | 0.04s | O(1) | <8 |
| AHashMap | 0.01s | O(1) avg | ~64 |

**Recommendation**: Use `quickphf` for compile-time vocabularies, AHashMap for runtime-loaded.

#### B. Trie-based Lookup (for prefix matching)

If supporting partial matches or longest-prefix, use Double-Array Trie:

```rust
// O(m) lookup where m = word length
fn lookup(&self, word: &[u8]) -> Option<u32> {
    let mut state = 0;
    for &byte in word {
        let next = self.base[state] + byte as i32;
        if self.check[next] != state { return None; }
        state = next;
    }
    self.token_ids[state]
}
```

### 2.3 LinMaxMatch Algorithm (from [Google Research](https://research.google/blog/a-fast-wordpiece-tokenization-system/))

While designed for WordPiece, the core ideas apply to WordLevel:

1. **Single-pass tokenization**: Combine pre-tokenization + token lookup
2. **Failure links**: Pre-computed fallback transitions
3. **Linear complexity**: O(n) guaranteed

For WordLevel, we can simplify this to:
- Process text character-by-character
- Accumulate word until delimiter
- Single hash lookup per word

---

## 3. Implementation Architecture

### 3.1 Scalar Version

```rust
pub struct WordLevelTokenizer {
    // Vocabulary (token string → ID)
    vocab: AHashMap<String, u32>,
    // Reverse vocabulary (ID → token string)
    vocab_r: Vec<String>,
    // Configuration
    config: WordLevelConfig,
    // Token cache (optional, for repeated words)
    cache: Option<Arc<ShardedCache<String, u32>>>,
}

pub struct WordLevelConfig {
    pub unk_token: String,
    pub unk_id: u32,
    // Pre-tokenizer configuration
    pub split_on_punctuation: bool,
    pub split_on_cjk: bool,
    pub lowercase: bool,
}
```

#### Scalar Tokenization Algorithm

```rust
fn encode(&self, text: &str) -> Vec<u32> {
    let mut result = Vec::with_capacity(text.len() / 5);  // Estimate: avg 5 chars/word
    let mut word_start = 0;
    let mut in_word = false;

    for (i, c) in text.char_indices() {
        let is_delimiter = c.is_whitespace() ||
                          (self.config.split_on_punctuation && is_punctuation(c));

        if is_delimiter {
            if in_word {
                // Flush word
                let word = &text[word_start..i];
                result.push(self.lookup_word(word));
                in_word = false;
            }
            // Optionally emit delimiter token (for punctuation)
            if self.config.split_on_punctuation && !c.is_whitespace() {
                let punct = &text[i..i + c.len_utf8()];
                result.push(self.lookup_word(punct));
            }
        } else if !in_word {
            word_start = i;
            in_word = true;
        }
    }

    // Flush final word
    if in_word {
        let word = &text[word_start..];
        result.push(self.lookup_word(word));
    }

    result
}

#[inline]
fn lookup_word(&self, word: &str) -> u32 {
    // Check cache first
    if let Some(ref cache) = self.cache {
        if let Some(id) = cache.get(&word.to_string()) {
            return id;
        }
    }

    // Vocabulary lookup
    let id = self.vocab.get(word).copied().unwrap_or(self.config.unk_id);

    // Cache result
    if let Some(ref cache) = self.cache {
        cache.insert(word.to_string(), id);
    }

    id
}
```

### 3.2 SIMD Version

```rust
pub struct WordLevelTokenizerSimd {
    // Same as scalar
    vocab: AHashMap<String, u32>,
    vocab_r: Vec<String>,
    config: WordLevelConfig,
    cache: Option<Arc<ShardedCache<String, u32>>>,

    // SIMD-specific: precomputed delimiter bitset
    delimiter_bitset: [u64; 4],  // 256-bit for all ASCII chars
}
```

#### SIMD Pre-tokenization (AVX-512)

```rust
#[cfg(target_feature = "avx512f")]
fn find_word_boundaries_avx512(&self, text: &[u8]) -> Vec<(usize, usize)> {
    use std::arch::x86_64::*;

    let mut boundaries = Vec::with_capacity(text.len() / 5);
    let mut pos = 0;
    let mut word_start: Option<usize> = None;

    // Process 64 bytes at a time
    while pos + 64 <= text.len() {
        let chunk = unsafe { _mm512_loadu_si512(text.as_ptr().add(pos) as *const _) };

        // Classify each byte using vpshufb nibble lookup
        let delimiter_mask = self.classify_delimiters_avx512(chunk);

        // Process delimiter positions
        let mut mask = delimiter_mask;
        while mask != 0 {
            let offset = mask.trailing_zeros() as usize;
            let abs_pos = pos + offset;

            if let Some(start) = word_start {
                boundaries.push((start, abs_pos));
                word_start = None;
            }

            // Check if next position starts a word
            if offset + 1 < 64 && (mask & (1u64 << (offset + 1))) == 0 {
                word_start = Some(abs_pos + 1);
            }

            mask &= mask - 1;  // Clear lowest bit
        }

        // Handle no-delimiter case
        if delimiter_mask == 0 && word_start.is_none() {
            word_start = Some(pos);
        }

        pos += 64;
    }

    // Handle remainder with scalar code
    // ... (same as scalar version)

    boundaries
}

#[cfg(target_feature = "avx512f")]
#[inline]
unsafe fn classify_delimiters_avx512(&self, data: __m512i) -> u64 {
    // Split bytes into nibbles
    let lo_nibbles = _mm512_and_si512(data, _mm512_set1_epi8(0x0f));
    let hi_nibbles = _mm512_srli_epi16(data, 4);
    let hi_nibbles = _mm512_and_si512(hi_nibbles, _mm512_set1_epi8(0x0f));

    // Load bitset tables
    let lo_table = _mm512_broadcast_i32x4(_mm_loadu_si128(
        self.delimiter_bitset.as_ptr() as *const _
    ));
    let hi_table = _mm512_broadcast_i32x4(_mm_loadu_si128(
        self.delimiter_bitset.as_ptr().add(2) as *const _
    ));

    // Nibble lookup
    let lo_bits = _mm512_shuffle_epi8(lo_table, lo_nibbles);
    let hi_bits = _mm512_shuffle_epi8(hi_table, hi_nibbles);

    // Combine with AND (both nibbles must match)
    let matches = _mm512_and_si512(lo_bits, hi_bits);

    // Convert to bitmask
    _mm512_test_epi8_mask(matches, matches)
}
```

#### AVX2 Fallback

```rust
#[cfg(target_feature = "avx2")]
fn find_word_boundaries_avx2(&self, text: &[u8]) -> Vec<(usize, usize)> {
    use std::arch::x86_64::*;

    let mut boundaries = Vec::with_capacity(text.len() / 5);
    let mut pos = 0;

    // Precompute comparison vectors
    let space_vec = unsafe { _mm256_set1_epi8(b' ' as i8) };
    let tab_vec = unsafe { _mm256_set1_epi8(b'\t' as i8) };
    let newline_vec = unsafe { _mm256_set1_epi8(b'\n' as i8) };
    let cr_vec = unsafe { _mm256_set1_epi8(b'\r' as i8) };

    while pos + 32 <= text.len() {
        let chunk = unsafe { _mm256_loadu_si256(text.as_ptr().add(pos) as *const _) };

        // Compare with each delimiter
        let space_eq = unsafe { _mm256_cmpeq_epi8(chunk, space_vec) };
        let tab_eq = unsafe { _mm256_cmpeq_epi8(chunk, tab_vec) };
        let newline_eq = unsafe { _mm256_cmpeq_epi8(chunk, newline_vec) };
        let cr_eq = unsafe { _mm256_cmpeq_epi8(chunk, cr_vec) };

        // OR all matches together
        let any_ws = unsafe {
            _mm256_or_si256(
                _mm256_or_si256(space_eq, tab_eq),
                _mm256_or_si256(newline_eq, cr_eq)
            )
        };

        // Convert to bitmask
        let mask = unsafe { _mm256_movemask_epi8(any_ws) } as u32;

        // Process matches (similar to AVX-512 version)
        // ...

        pos += 32;
    }

    boundaries
}
```

### 3.3 Batch Processing with Rayon

```rust
fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
    texts.par_iter()
        .map(|text| self.encode(text))
        .collect()
}

fn encode_batch_simd(&self, texts: &[&str]) -> Vec<Vec<u32>> {
    // Sort by length for better SIMD utilization
    let mut indexed: Vec<_> = texts.iter().enumerate().collect();
    indexed.sort_by_key(|(_, t)| std::cmp::Reverse(t.len()));

    // Process in parallel
    let results: Vec<_> = indexed.par_iter()
        .map(|(i, text)| (*i, self.encode_simd(text)))
        .collect();

    // Restore original order
    let mut output = vec![Vec::new(); texts.len()];
    for (i, result) in results {
        output[i] = result;
    }
    output
}
```

---

## 4. Data Structure Recommendations

### 4.1 Vocabulary Storage

| Use Case | Recommended Structure | Lookup Time |
|----------|----------------------|-------------|
| Runtime-loaded vocab | AHashMap | O(1) avg |
| Compile-time vocab | quickphf::PhfMap | O(1) guaranteed |
| Prefix matching needed | DoubleArrayTrie | O(m) |
| Large vocab (>1M) | ShardedMap | O(1) avg, low contention |

### 4.2 Memory Layout

```rust
// Cache-friendly vocabulary layout
pub struct OptimizedVocab {
    // Token IDs packed densely
    token_to_id: AHashMap<Box<str>, u32>,  // Use Box<str> to reduce indirection

    // Reverse lookup as Vec (ID is array index)
    id_to_token: Vec<Box<str>>,

    // Token lengths for quick filtering
    token_lengths: Vec<u8>,  // Most tokens < 256 bytes

    // Precomputed hash for common tokens
    common_token_hashes: Vec<u64>,
}
```

---

## 5. Expected Performance

Based on research and benchmarks:

| Component | Scalar | SIMD (AVX2) | SIMD (AVX-512) |
|-----------|--------|-------------|----------------|
| Pre-tokenization | 500 MB/s | 3 GB/s | 8 GB/s |
| Vocab lookup | 50M lookups/s | 50M lookups/s | 50M lookups/s |
| Cache hit | 200M hits/s | 200M hits/s | 200M hits/s |
| **End-to-end** | **300 MB/s** | **2 GB/s** | **5 GB/s** |

Comparison with HuggingFace:
- HuggingFace WordLevel: ~100 MB/s
- BudTikTok Scalar: ~300 MB/s (3x faster)
- BudTikTok SIMD: ~2-5 GB/s (20-50x faster)

---

## 6. Implementation Phases

### Phase 1: Scalar Implementation
1. Create `WordLevelTokenizer` struct
2. Implement `Tokenizer` trait
3. Add pre-tokenizer (whitespace + punctuation)
4. Integrate cache from existing infrastructure
5. Add HuggingFace compatibility tests

### Phase 2: SIMD Optimization
1. Implement AVX2 whitespace detection
2. Implement AVX-512 bitset classification
3. Add runtime feature detection
4. Benchmark against scalar

### Phase 3: Advanced Features
1. Perfect hash vocabulary (quickphf integration)
2. Compile-time vocabulary generation
3. Cache warmup from vocabulary
4. NEON support for ARM

---

## 7. File Structure

```
crates/budtiktok-core/src/
├── wordlevel.rs              # Scalar implementation
├── wordlevel_simd.rs         # SIMD implementation
├── wordlevel_pretokenize.rs  # Pre-tokenization logic
└── wordlevel_bench.rs        # Benchmarks
```

---

## 8. References

1. [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) - Reference implementation
2. [Fast WordPiece Tokenization (Google)](https://research.google/blog/a-fast-wordpiece-tokenization-system/) - LinMaxMatch algorithm
3. [StringZilla](https://ashvardanian.com/posts/splitting-strings-cpp/) - SIMD string splitting
4. [simdjson](https://github.com/simdjson/simdjson) - SIMD text processing patterns
5. [rust-phf](https://github.com/rust-phf/rust-phf) - Compile-time perfect hashing
6. [quickphf](https://docs.rs/quickphf) - Fast perfect hash maps
7. [Daniel Lemire's Blog](https://lemire.me/blog/) - SIMD optimization techniques
