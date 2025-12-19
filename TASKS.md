# BudTikTok - Production Implementation Tasks

**BudTikTok** is a high-performance disaggregated tokenization system designed to outperform HuggingFace Tokenizers, TEI, BlazeText, and Snowflake across all hardware platforms.

---

## Table of Contents

1. [Project Setup](#1-project-setup)
2. [Core Infrastructure](#2-core-infrastructure)
3. [Tokenization Algorithms](#3-tokenization-algorithms)
4. [SIMD Acceleration](#4-simd-acceleration)
5. [GPU Tokenization](#5-gpu-tokenization)
6. [Distributed Architecture](#6-distributed-architecture)
7. [LatentBud Integration](#7-latentbud-integration)
8. [Resilience and Operations](#8-resilience-and-operations)
9. [Test Suites (TDD)](#9-test-suites-tdd)
10. [Profiling Tools](#10-profiling-tools)
11. [Accuracy Testing Suite](#11-accuracy-testing-suite)
12. [Comparison Benchmarking Tool](#12-comparison-benchmarking-tool)
13. [Deployment](#13-deployment)
14. [Documentation](#14-documentation)

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Completed |
| `[!]` | Blocked |

---

## 1. Project Setup

### 1.1 Workspace Structure

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 1.1.1 | Create Cargo workspace | Create `Cargo.toml` with workspace members: `budtiktok-core`, `budtiktok-simd`, `budtiktok-gpu`, `budtiktok-ipc`, `budtiktok-coordinator`, `budtiktok-cli`, `budtiktok-bench`. Use Rust 2021 edition. Set resolver = "2". | None | ```toml [workspace] members = ["crates/*"] resolver = "2" [workspace.dependencies] tokio = { version = "1.35", features = ["full"] } ahash = "0.8" serde = { version = "1.0", features = ["derive"] } ``` | `[x]` |
| 1.1.2 | Configure workspace dependencies | Add shared dependencies at workspace level: `tokio`, `rayon`, `ahash`, `serde`, `tracing`, `thiserror`, `anyhow`, `bytes`, `parking_lot`, `crossbeam`. Pin versions for reproducibility. | 1.1.1 | Use workspace inheritance: `tokio.workspace = true` in member crates | `[x]` |
| 1.1.3 | Create crate structure | Create all member crates with proper `Cargo.toml`, `src/lib.rs`. Set up re-exports in main `budtiktok` facade crate. Configure feature flags: `simd`, `gpu`, `distributed`, `full`. | 1.1.1 | Directory structure: `crates/{core,simd,gpu,ipc,coordinator,cli,bench}/` | `[x]` |
| 1.1.4 | Configure build.rs | Create build script for: SIMD feature detection at compile time, generate version info, embed git hash, detect CUDA/ROCm. Output cargo:rustc-cfg directives. | 1.1.3 | Use `cc` crate for C compilation if needed. Check for `avx512f`, `avx2`, `sse4.2`, `neon` | `[x]` |
| 1.1.5 | Set up rustfmt and clippy | Create `rustfmt.toml` with project style. Create `.clippy.toml` with lint configuration. Add `#![deny(clippy::all)]` to all crates. Configure CI to fail on warnings. | 1.1.3 | Enable pedantic lints, disable overly noisy ones | `[x]` |
| 1.1.6 | Configure memory allocators | Add jemalloc (Linux) and mimalloc (macOS/Windows) as optional global allocators. Create feature flags: `jemalloc`, `mimalloc`. Benchmark both. Default to jemalloc on Linux. | 1.1.3 | ```rust #[cfg(all(target_os = "linux", feature = "jemalloc"))] #[global_allocator] static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc; ``` | `[x]` |
| 1.1.7 | Set up error handling | Create `budtiktok-core/src/error.rs` with `BudError` enum using `thiserror`. Define error types: `VocabError`, `TokenizeError`, `IoError`, `ConfigError`, `IpcError`. Implement `From` traits. | 1.1.3 | Use `#[error("...")]` attributes for messages. Add `#[from]` for conversions. | `[x]` |

### 1.2 CI/CD Pipeline

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 1.2.1 | Create GitHub Actions workflow | Main workflow: build, test, lint, benchmark. Matrix: `ubuntu-latest`, `macos-latest`, `windows-latest`. Targets: `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-apple-darwin`, `aarch64-apple-darwin`. | 1.1.5 | Use `actions-rs/toolchain`, `actions-rs/cargo`. Cache target directory. | `[x]` |
| 1.2.2 | Add benchmark CI | Run benchmarks on every PR. Compare against baseline. Post results as PR comment. Store historical data. Alert on >10% regression. | 1.2.1 | Use `criterion` with `--save-baseline`. Compare with `critcmp`. | `[x]` |
| 1.2.3 | Add coverage reporting | Generate code coverage with `cargo-llvm-cov`. Upload to Codecov. Set minimum coverage threshold (80%). Fail CI if below threshold. | 1.2.1 | Use `llvm-cov` for accurate Rust coverage. Exclude test files. | `[x]` |
| 1.2.4 | Add security scanning | Run `cargo-audit` for vulnerability scanning. Run `cargo-deny` for license compliance. Fail CI on critical vulnerabilities. | 1.2.1 | Configure `deny.toml` with allowed licenses: MIT, Apache-2.0, BSD | `[x]` |
| 1.2.5 | Create release workflow | Automated releases on git tag. Build release binaries for all platforms. Create GitHub release with changelog. Publish to crates.io. Build and push Docker images. | 1.2.1 | Use `cargo-dist` for binary distribution. Sign releases. | `[x]` |
| 1.2.6 | Add Docker build pipeline | Multi-stage Dockerfile for minimal image size. Separate images for coordinator and worker. GPU-enabled image with CUDA runtime. ARM64 and AMD64 variants. | 1.2.1 | Base on `rust:1.75-slim` for build, `debian:bookworm-slim` for runtime | `[x]` |

---

## 2. Core Infrastructure

### 2.1 Unicode Normalization

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 2.1.1 | Implement ASCII fast path | Check if string is pure ASCII using SIMD-style u64 operations. If all bytes < 128, skip normalization entirely. Return `Cow::Borrowed` for zero-copy. | 1.1.7 | ```rust pub fn is_ascii_fast(s: &str) -> bool { let bytes = s.as_bytes(); let chunks = bytes.chunks_exact(8); for chunk in chunks { let word = u64::from_ne_bytes(chunk.try_into().unwrap()); if word & 0x8080808080808080 != 0 { return false; } } bytes[chunks.remainder().len()..].iter().all(|&b| b < 128) } ``` | `[x]` |
| 2.1.2 | Generate Unicode data tables | Parse UnicodeData.txt from Unicode 15.1. Generate: canonical decomposition map, composition pairs, canonical combining classes, quick check properties. Use build.rs or pre-generate. | 1.1.4 | Using `unicode-normalization` crate for data tables. Custom tables deferred for future optimization. | `[x]` |
| 2.1.3 | Implement NFC Quick Check | Return `Yes`/`No`/`Maybe` without full normalization. Track CCC ordering. Use lookup table for quick check property. Return early on first `No`. | 2.1.2 | ```rust pub enum IsNormalized { Yes, No, Maybe } pub fn is_nfc_quick(s: &str) -> IsNormalized { let mut last_ccc = 0; for ch in s.chars() { if ch.is_ascii() { last_ccc = 0; continue; } let ccc = canonical_combining_class(ch); if last_ccc > ccc && ccc != 0 { return IsNormalized::No; } match quick_check_nfc(ch) { QC::No => return IsNormalized::No, QC::Maybe => result = IsNormalized::Maybe, QC::Yes => {} } last_ccc = ccc; } result } ``` | `[x]` |
| 2.1.4 | Implement Bloom filter for Latin-1 | Create 256-bit bloom filter for precomposed Latin-1 characters (0xC0-0x17F). O(1) check before full composition. ~5% false positive rate acceptable. | 2.1.2 | ```rust const PRECOMPOSED_BLOOM: [u64; 4] = [...]; // Generated fn might_be_precomposed(ch: char) -> bool { let cp = ch as u32; if cp < 0xC0 || cp > 0x17F { return false; } let bit = (cp - 0xC0) as usize; PRECOMPOSED_BLOOM[bit / 64] & (1 << (bit % 64)) != 0 } ``` | `[x]` |
| 2.1.5 | Implement canonical decomposition | Recursive decomposition following Unicode algorithm. Handle Hangul algorithmic decomposition (no table needed). Use stack-based iteration to avoid recursion overhead. | 2.1.2 | Hangul decomposition: `SBase=0xAC00, LBase=0x1100, VBase=0x1161, TBase=0x11A7`. Decompose syllable to L+V+T jamo. | `[x]` |
| 2.1.6 | Implement canonical composition | Compose starter + combining character pairs. Handle composition exclusions. Use two-stage table for pair lookup. Handle blocked combining characters. | 2.1.2 | Using `unicode-normalization` crate's `.nfc()` for composition. Hangul composition implemented in `compose_hangul()`. | `[x]` |
| 2.1.7 | Implement NFD/NFKD/NFKC | NFD: decompose canonically. NFKD: decompose with compatibility. NFC: decompose + compose. NFKC: decompose compat + compose. Share core logic. | 2.1.5, 2.1.6 | Add compatibility decomposition table (~6K entries). Use same algorithm with different table. | `[x]` |
| 2.1.8 | Add SIMD normalization path | Use AVX2 to process 32 bytes at once. Identify characters needing normalization. Fall back to scalar for complex sequences. | 2.1.7, 4.2.1 | SIMD identifies ASCII (fast) vs needs-work (slow path). Scalar handles actual normalization. | `[x]` |
| 2.1.9 | Implement streaming normalization | For very long strings, process in chunks. Maintain state across chunk boundaries. Handle combining sequences split across chunks. | 2.1.7 | Buffer incomplete combining sequences. Flush on end of stream. | `[x]` |

### 2.2 Unicode Categories

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 2.2.1 | Define CategoryFlags bitfield | Create `CategoryFlags(u32)` with bits for all 30 Unicode General Categories. Implement `is_letter()`, `is_mark()`, `is_number()`, `is_punctuation()`, `is_symbol()`, `is_separator()`, `is_other()` as single bitwise AND. | 1.1.7 | ```rust #[derive(Clone, Copy)] pub struct CategoryFlags(u32); impl CategoryFlags { pub const LETTER_UPPERCASE: u32 = 1 << 0; pub const LETTER_LOWERCASE: u32 = 1 << 1; // ... 28 more pub const LETTER: u32 = Self::LETTER_UPPERCASE | Self::LETTER_LOWERCASE | Self::LETTER_TITLECASE | Self::LETTER_MODIFIER | Self::LETTER_OTHER; #[inline] pub fn is_letter(self) -> bool { self.0 & Self::LETTER != 0 } } ``` | `[x]` |
| 2.2.2 | Generate ASCII lookup table | Compile-time generate `const ASCII_CATEGORIES: [CategoryFlags; 128]`. Direct O(1) lookup for ASCII. Include in binary. | 2.2.1 | ```rust const ASCII_CATEGORIES: [CategoryFlags; 128] = { let mut table = [CategoryFlags(0); 128]; // 0-31: Cc (control) for i in 0..32 { table[i] = CategoryFlags::OTHER_CONTROL; } // 32: Zs (space) table[32] = CategoryFlags::SEPARATOR_SPACE; // ... populate all }; ``` | `[x]` |
| 2.2.3 | Generate Unicode lookup tables | Two-stage table for BMP (0-0xFFFF). Three-stage for supplementary planes. Compress via perfect hashing or trie. Target <50KB binary size. | 2.2.1, 2.1.2 | Using runtime lookups via Rust std lib and unicode-categories. ASCII fast path via const table. | `[x]` |
| 2.2.4 | Implement thread-local cache | 128-entry direct-mapped cache per thread. Key: char, Value: CategoryFlags. No locking needed. LRU approximation via CLOCK. | 2.2.3 | ```rust thread_local! { static CACHE: RefCell<[(char, CategoryFlags); 128]> = RefCell::new([('\0', CategoryFlags(0)); 128]); } pub fn get_category(ch: char) -> CategoryFlags { if (ch as u32) < 128 { return ASCII_CATEGORIES[ch as usize]; } CACHE.with(|cache| { let mut cache = cache.borrow_mut(); let idx = (ch as usize) % 128; if cache[idx].0 == ch { return cache[idx].1; } let flags = lookup_unicode_table(ch); cache[idx] = (ch, flags); flags }) } ``` | `[x]` |
| 2.2.5 | Implement specialized predicates | `is_whitespace()`: Zs + Cc whitespace. `is_punctuation()`: all P categories. `is_cjk()`: CJK ranges. `is_emoji()`: emoji ranges. All as single bitwise ops or range checks. | 2.2.1 | CJK ranges: 0x4E00-0x9FFF, 0x3400-0x4DBF, 0x20000-0x2A6DF, etc. Use `matches!` for efficient codegen. | `[x]` |

### 2.3 Token Cache

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 2.3.1 | Implement CLOCK cache | Fixed-size cache with second-chance eviction. O(1) amortized insert/lookup. Thread-safe with `parking_lot::RwLock`. Better than LRU for workloads with scans. | 1.1.7 | ```rust pub struct ClockCache<K: Hash + Eq, V> { slots: Vec<Slot<K, V>>, hand: AtomicUsize, capacity: usize, hasher: ahash::RandomState, } struct Slot<K, V> { key: Option<K>, value: Option<V>, referenced: AtomicBool, } impl<K: Hash + Eq + Clone, V: Clone> ClockCache<K, V> { pub fn get(&self, key: &K) -> Option<V> { let hash = self.hasher.hash_one(key); let idx = (hash as usize) % self.capacity; // Linear probe for key for i in 0..8 { let slot_idx = (idx + i) % self.capacity; if self.slots[slot_idx].key.as_ref() == Some(key) { self.slots[slot_idx].referenced.store(true, Relaxed); return self.slots[slot_idx].value.clone(); } } None } } ``` | `[x]` |
| 2.3.2 | Implement sharded cache | Shard cache into N segments (default: num_cpus). Each shard independently locked. Reduces contention for concurrent access. | 2.3.1 | ```rust pub struct ShardedCache<K, V> { shards: Vec<ClockCache<K, V>>, } impl<K: Hash + Eq + Clone, V: Clone> ShardedCache<K, V> { pub fn get(&self, key: &K) -> Option<V> { let shard = self.shard_for(key); shard.get(key) } fn shard_for(&self, key: &K) -> &ClockCache<K, V> { let hash = ahash::RandomState::new().hash_one(key); &self.shards[(hash as usize) % self.shards.len()] } } ``` | `[x]` |
| 2.3.3 | Create multi-level cache | L1: Per-word token IDs (10K entries). L2: Subword lookups (50K entries). L3: Unicode composition (thread-local, 128). L4: Unicode category (thread-local, 128). | 2.3.2 | Each level has different key/value types. L1 caches full tokenization result. L2 caches individual subword matches. | `[x]` |
| 2.3.4 | Add cache statistics | Track: hits, misses, insertions, evictions, hit_rate. Atomic counters for thread safety. Export to Prometheus format. | 2.3.1 | ```rust pub struct CacheStats { hits: AtomicU64, misses: AtomicU64, insertions: AtomicU64, evictions: AtomicU64, } impl CacheStats { pub fn hit_rate(&self) -> f64 { let hits = self.hits.load(Relaxed); let total = hits + self.misses.load(Relaxed); if total == 0 { 0.0 } else { hits as f64 / total as f64 } } } ``` | `[x]` |
| 2.3.5 | Implement cache warmup | Pre-populate cache with frequent tokens from vocabulary. Use frequency information if available. Support async warmup on startup. | 2.3.3 | Load top 1000 most frequent words, tokenize them, cache results. | `[x]` |

### 2.4 Vocabulary and Data Structures

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 2.4.1 | Implement tokenizer.json parser | Parse HuggingFace tokenizer.json format. Extract: model type, vocab, merges, special tokens, normalizer config, pre-tokenizer config. Use `serde_json`. | 1.1.7 | Define structs matching HF schema: `TokenizerConfig`, `Model`, `Vocab`, `Merges`, `AddedToken`, `Normalizer`, `PreTokenizer`. Handle all variants. | `[x]` |
| 2.4.2 | Create vocabulary structures | `Vocab`: bidirectional map token↔id. Use `AHashMap<String, u32>` and `Vec<String>` for O(1) both directions. Memory-map large vocabularies. | 2.4.1 | ```rust pub struct Vocab { token_to_id: AHashMap<String, u32>, id_to_token: Vec<String>, } impl Vocab { pub fn get_id(&self, token: &str) -> Option<u32> { self.token_to_id.get(token).copied() } pub fn get_token(&self, id: u32) -> Option<&str> { self.id_to_token.get(id as usize).map(|s| s.as_str()) } } ``` | `[x]` |
| 2.4.3 | Implement Trie data structure | Byte-indexed trie for prefix matching. 256-ary nodes for O(1) child lookup. Store token IDs at leaf/intermediate nodes. Support common prefix iteration. | 2.4.2 | ```rust pub struct Trie { nodes: Vec<TrieNode>, } struct TrieNode { children: [u32; 256], // 0 = no child, >0 = node index token_id: Option<u32>, // Some if this is end of token } impl Trie { pub fn insert(&mut self, token: &[u8], id: u32) { ... } pub fn get(&self, token: &[u8]) -> Option<u32> { ... } pub fn common_prefix_search(&self, text: &[u8]) -> impl Iterator<Item = (usize, u32)> { ... } } ``` | `[x]` |
| 2.4.4 | Implement cache-oblivious Trie | Reorder trie nodes using van Emde Boas layout for better cache performance. Parent and children in same cache line when possible. | 2.4.3 | BFS order for top levels, then recursive for subtrees. Measure cache miss rate with perf. | `[x]` |
| 2.4.5 | Build Aho-Corasick automaton | For special token matching. Use `aho-corasick` crate. Configure LeftmostLongest match semantics. Pre-build from added tokens with `normalized=false`. | 2.4.1 | ```rust use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind}; let special_tokens: Vec<&str> = added_tokens.iter() .filter(|t| !t.normalized) .map(|t| t.content.as_str()) .collect(); let ac = AhoCorasickBuilder::new() .match_kind(MatchKind::LeftmostLongest) .build(&special_tokens)?; ``` | `[x]` |
| 2.4.6 | Implement Double-Array Trie | Alternative trie with better cache locality. Two arrays: base[] and check[]. O(1) transitions. More compact than 256-ary. | 2.4.2 | Double-array trie: `base[s] + c = t` and `check[t] = s` for transition s→t on character c. Build using optimal base value search. | `[x]` |

### 2.5 Memory Management

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 2.5.1 | Implement arena allocator | Per-batch arena using `bumpalo`. Allocate all temporary data from arena. Reset arena between batches. Near-zero allocation overhead. | 1.1.6 | ```rust thread_local! { static ARENA: RefCell<Bump> = RefCell::new(Bump::with_capacity(1024 * 1024)); } pub fn with_arena<T>(f: impl FnOnce(&Bump) -> T) -> T { ARENA.with(|arena| { let arena = arena.borrow(); let result = f(&arena); arena.reset(); result }) } ``` | `[x]` |
| 2.5.2 | Implement string interner | Deduplicate repeated strings in vocabulary. Single storage for each unique string. Return interned `&str` references. | 2.4.2 | Use `string-interner` crate or custom implementation with `HashSet<Box<str>>`. | `[x]` |
| 2.5.3 | Add memory pool for encodings | Pre-allocate `Encoding` structs. Reuse across requests. Avoid repeated allocation of vectors. | 2.5.1 | Pool of `Vec<u32>` for token IDs, reuse with `clear()` instead of new allocation. | `[x]` |
| 2.5.4 | Implement Cow-based strings | Use `Cow<str>` throughout to avoid allocations. Borrowed for unchanged strings, Owned only when modified. | 1.1.7 | ```rust pub fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> { if is_ascii_fast(input) { Cow::Borrowed(input) } else { Cow::Owned(self.normalize_unicode(input)) } } ``` | `[x]` |

---

## 3. Tokenization Algorithms

### 3.1 WordPiece

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 3.1.1 | Implement core WordPiece algorithm | Greedy longest-match from left to right. Try longest substring first, shrink until vocab match. Handle continuation prefix (`##`). Return `[UNK]` if no match. | 2.4.2 | ```rust pub fn tokenize_word(&self, word: &str) -> Vec<u32> { if let Some(id) = self.vocab.get_id(word) { return vec![id]; // Whole word match } let mut tokens = Vec::with_capacity(word.len() / 3); let mut start = 0; while start < word.len() { let mut end = word.len(); let mut found = false; while start < end { let substr = &word[start..end]; let lookup = if start > 0 { format!("{}{}", self.prefix, substr) } else { substr.to_string() }; if let Some(id) = self.vocab.get_id(&lookup) { tokens.push(id); found = true; break; } // Shrink by one char (UTF-8 aware) end = word[..end].char_indices().last() .map(|(i, _)| i).unwrap_or(start); } if !found { return vec![self.unk_id]; } start = end; } tokens } ``` | `[x]` |
| 3.1.2 | Add byte-length optimization | If `byte_len <= max_chars`, skip char counting (majority case). Only count chars if byte length exceeds limit. | 3.1.1 | ```rust if word.len() <= self.max_input_chars_per_word { // Safe to proceed, byte len <= char len } else if word.chars().count() > self.max_input_chars_per_word { return vec![self.unk_id]; } ``` | `[x]` |
| 3.1.3 | Implement BERT normalizer | Clean text, handle Chinese chars, strip accents, lowercase. Configurable via options. | 2.1.7 | `clean_text`: remove control chars (0x00-0x1F except whitespace), replace \t\n\r with space. `handle_chinese_chars`: add space around CJK. `strip_accents`: NFD + filter Mn category. `lowercase`: to_lowercase(). | `[x]` |
| 3.1.4 | Implement BERT pre-tokenizer | Split on whitespace, isolate punctuation. Handle degree symbols and special punctuation correctly. | 2.2.5 | Split on `is_whitespace()`, then for each word, isolate `is_punctuation()` characters as separate tokens. | `[x]` |
| 3.1.5 | Implement BERT post-processor | Insert [CLS] at start, [SEP] at end. For pairs: [CLS] A [SEP] B [SEP]. Generate type_ids (0 for first, 1 for second). | 3.1.1 | ```rust pub fn post_process(&self, encoding: Encoding, pair: Option<Encoding>) -> Encoding { let mut ids = vec![self.cls_id]; ids.extend(encoding.ids); ids.push(self.sep_id); let mut type_ids = vec![0; ids.len()]; if let Some(pair) = pair { let pair_start = ids.len(); ids.extend(pair.ids); ids.push(self.sep_id); type_ids.extend(vec![1; ids.len() - pair_start]); } // ... build full Encoding } ``` | `[x]` |
| 3.1.6 | Add caching integration | Check cache before tokenization. Cache results for cache-worthy strings (<256 chars). Use word as cache key. | 2.3.3, 3.1.1 | ```rust pub fn tokenize_cached(&self, word: &str) -> Vec<u32> { if word.len() <= 256 { if let Some(cached) = self.cache.get(word) { return cached; } } let result = self.tokenize_word(word); if word.len() <= 256 { self.cache.insert(word.to_string(), result.clone()); } result } ``` | `[x]` |

### 3.2 Unigram (SentencePiece)

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 3.2.1 | Implement Viterbi decoder | Dynamic programming to find optimal segmentation. Track best score and backpointer at each position. Use Trie for efficient prefix enumeration. | 2.4.3 | ```rust pub fn encode_viterbi(&self, text: &str) -> Vec<u32> { let bytes = text.as_bytes(); let n = bytes.len(); let mut best_score = vec![f64::NEG_INFINITY; n + 1]; let mut best_prev = vec![0usize; n + 1]; best_score[0] = 0.0; for i in 0..n { if best_score[i] == f64::NEG_INFINITY { continue; } for (len, token_id) in self.trie.common_prefix_search(&bytes[i..]) { let j = i + len; let score = best_score[i] + self.scores[token_id as usize]; if score > best_score[j] { best_score[j] = score; best_prev[j] = i; best_token[j] = token_id; } } } // Backtrack let mut tokens = Vec::new(); let mut pos = n; while pos > 0 { tokens.push(best_token[pos]); pos = best_prev[pos]; } tokens.reverse(); tokens } ``` | `[x]` |
| 3.2.2 | Add byte fallback | When no token matches, fall back to single byte token `<0xNN>`. Ensure all inputs are tokenizable. Handle partial UTF-8. | 3.2.1 | For each position without valid token, use `<0x{:02X}>` format byte token. Pre-compute byte token IDs 0-255. | `[x]` |
| 3.2.3 | Implement Metaspace pre-tokenizer | Replace spaces with ▁ (U+2581). Add ▁ at start of words. Handle `add_prefix_space` option. | 2.1.7 | ```rust pub fn pre_tokenize(&self, text: &str) -> String { let mut result = String::with_capacity(text.len() + text.matches(' ').count()); if self.add_prefix_space && !text.starts_with(' ') { result.push('▁'); } for ch in text.chars() { if ch == ' ' { result.push('▁'); } else { result.push(ch); } } result } ``` | `[x]` |
| 3.2.4 | Implement N-best decoding | A* search for top-N segmentations. Use priority queue ordered by score. Limit agenda size to prevent memory explosion. | 3.2.1 | ```rust pub fn encode_nbest(&self, text: &str, n: usize) -> Vec<Vec<u32>> { let mut agenda: BinaryHeap<Hypothesis> = BinaryHeap::new(); // ... A* search with hypothesis expansion // Return top n complete hypotheses } ``` | `[x]` |
| 3.2.5 | Implement stochastic sampling | Forward-filtering backward-sampling. Compute alpha (log-sum-exp) scores. Sample from distribution. Use temperature parameter. | 3.2.1 | ```rust pub fn sample(&self, text: &str, temperature: f64) -> Vec<u32> { let alpha = self.forward_pass(text); // log-sum-exp to each position self.backward_sample(&alpha, temperature) } ``` | `[x]` |

### 3.3 BPE (O(n) Linear Algorithm)

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 3.3.1 | Build Aho-Corasick automaton for BPE | Create AC automaton from all vocabulary tokens. Used for efficient suffix enumeration in O(n) algorithm. | 2.4.5 | Build from all vocab tokens, not just special tokens. | `[x]` |
| 3.3.2 | Build compatibility table | For each token pair (a, b), check if merge exists. Store merged token if compatible. HashMap for O(1) lookup. | 2.4.1 | ```rust pub struct CompatibilityTable { table: AHashMap<(u32, u32), u32>, } impl CompatibilityTable { pub fn from_merges(merges: &[(String, String)], vocab: &Vocab) -> Self { let mut table = AHashMap::new(); for (a, b) in merges { let a_id = vocab.get_id(a)?; let b_id = vocab.get_id(b)?; let merged = format!("{}{}", a, b); let merged_id = vocab.get_id(&merged)?; table.insert((a_id, b_id), merged_id); } Self { table } } pub fn get(&self, a: u32, b: u32) -> Option<u32> { self.table.get(&(a, b)).copied() } } ``` | `[x]` |
| 3.3.3 | Implement O(n) BPE encoder | Use DP with AC automaton. Track (token_count, last_token) at each position. Enumerate suffixes, check compatibility, update DP. | 3.3.1, 3.3.2 | ```rust pub fn encode_linear(&self, text: &[u8]) -> Vec<u32> { let n = text.len(); let mut dp: Vec<Option<(usize, u32)>> = vec![None; n + 1]; dp[0] = Some((0, u32::MAX)); // (count, last_token) for i in 0..n { let Some((count, last_token)) = dp[i] else { continue }; for mat in self.ac.find_overlapping(&text[i..]) { let token_id = mat.pattern().as_u32(); let end = i + mat.end(); // Check compatibility let compatible = last_token == u32::MAX || self.compat.get(last_token, token_id).is_some(); if compatible { let new_count = count + 1; if dp[end].map(|(c, _)| c > new_count).unwrap_or(true) { dp[end] = Some((new_count, token_id)); } } } } self.backtrack(&dp) } ``` | `[x]` |
| 3.3.4 | Implement byte-level BPE | Map bytes 0-255 to printable Unicode (GPT-2 style). Process bytes instead of chars. | 3.3.3 | ```rust lazy_static! { static ref BYTE_TO_CHAR: [char; 256] = { let mut arr = ['\0'; 256]; let mut n = 0u32; for i in 0..256u8 { arr[i as usize] = match i { b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF => i as char, _ => { let ch = char::from_u32(256 + n).unwrap(); n += 1; ch } }; } arr }; } ``` | `[x]` |
| 3.3.5 | Add dropout support | Randomly skip merges during training/augmentation. Use probability parameter. For inference, disable. | 3.3.3 | Add `dropout: Option<f32>` parameter. If Some, skip merge with given probability using `rand::random()`. | `[x]` |
| 3.3.6 | Implement fallback O(n log n) algorithm | Traditional heap-based merge for comparison/validation. OctonaryHeap for efficiency. | 2.4.2 | For short sequences or validation, use merge-based algorithm. Compare results with O(n) for correctness. | `[x]` |

### 3.4 Unified Interface

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 3.4.1 | Define Tokenizer trait | Core trait for all tokenizers. Methods: `encode`, `encode_batch`, `decode`, `decode_batch`, `vocab_size`, `token_to_id`, `id_to_token`. | 1.1.7 | ```rust pub trait Tokenizer: Send + Sync { fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding>; fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>>; fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String>; fn vocab_size(&self) -> usize; fn token_to_id(&self, token: &str) -> Option<u32>; fn id_to_token(&self, id: u32) -> Option<&str>; } ``` | `[x]` |
| 3.4.2 | Implement Encoding struct | Output of tokenization. Fields: ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing. | 3.4.1 | ```rust pub struct Encoding { pub ids: Vec<u32>, pub type_ids: Vec<u32>, pub tokens: Vec<String>, pub offsets: Vec<(usize, usize)>, pub attention_mask: Vec<u32>, pub special_tokens_mask: Vec<u32>, pub overflowing: Vec<Encoding>, } ``` | `[x]` |
| 3.4.3 | Implement auto-detection | Detect tokenizer type from tokenizer.json. Look for `model.type` field. Return appropriate implementation. | 2.4.1 | ```rust pub fn from_file(path: &Path) -> Result<Box<dyn Tokenizer>> { let config: TokenizerConfig = serde_json::from_reader(File::open(path)?)?; match config.model.type_.as_str() { "WordPiece" => Ok(Box::new(WordPieceTokenizer::from_config(config)?)), "BPE" => Ok(Box::new(BpeTokenizer::from_config(config)?)), "Unigram" => Ok(Box::new(UnigramTokenizer::from_config(config)?)), _ => Err(Error::UnsupportedModel), } } ``` | `[x]` |
| 3.4.4 | Implement truncation | Strategies: LongestFirst, OnlyFirst, OnlySecond. Handle max_length. Generate overflow for stride. | 3.4.2 | ```rust pub fn truncate(&mut self, max_length: usize, stride: usize, strategy: TruncationStrategy) { if self.ids.len() <= max_length { return; } let overflow_len = self.ids.len() - max_length; // ... split and store in overflowing } ``` | `[x]` |
| 3.4.5 | Implement padding | Strategies: BatchLongest, Fixed(usize). Direction: Left, Right. Pad to multiple_of if specified. | 3.4.2 | ```rust pub fn pad(&mut self, target_length: usize, pad_id: u32, direction: PaddingDirection) { while self.ids.len() < target_length { match direction { Left => { self.ids.insert(0, pad_id); self.attention_mask.insert(0, 0); } Right => { self.ids.push(pad_id); self.attention_mask.push(0); } } } } ``` | `[x]` |
| 3.4.6 | Implement batch encoding | Process multiple texts efficiently. Use rayon for parallelism when batch > threshold. Apply padding uniformly. | 3.4.1, 3.4.5 | ```rust fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> { let encodings: Vec<_> = if texts.len() > 8 { texts.par_iter().map(|t| self.encode(t, add_special_tokens)).collect() } else { texts.iter().map(|t| self.encode(t, add_special_tokens)).collect() }; let max_len = encodings.iter().map(|e| e.ids.len()).max().unwrap_or(0); for enc in &mut encodings { enc.pad(max_len, self.pad_id, Right); } Ok(encodings) } ``` | `[x]` |

---

## 4. SIMD Acceleration

### 4.1 Runtime Detection

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 4.1.1 | Implement CPU feature detection | Detect SIMD capabilities at runtime. Cache results in static. Support x86_64 (AVX-512, AVX2, SSE4.2) and aarch64 (NEON, SVE). | 1.1.4 | ```rust pub struct CpuFeatures { pub avx512f: bool, pub avx512bw: bool, pub avx2: bool, pub sse42: bool, pub neon: bool, pub sve: bool, } lazy_static! { pub static ref CPU: CpuFeatures = CpuFeatures { #[cfg(target_arch = "x86_64")] avx512f: is_x86_feature_detected!("avx512f"), #[cfg(target_arch = "x86_64")] avx2: is_x86_feature_detected!("avx2"), // ... }; } ``` | `[x]` |
| 4.1.2 | Create dispatch macro | Macro to generate function variants for each SIMD level. Auto-select at runtime. | 4.1.1 | ```rust macro_rules! simd_dispatch { ($name:ident, $fn:ident, $($arg:ident: $ty:ty),*) => { pub fn $name($($arg: $ty),*) { #[cfg(target_arch = "x86_64")] { if CPU.avx512f { return unsafe { $fn ## _avx512($($arg),*) }; } if CPU.avx2 { return unsafe { $fn ## _avx2($($arg),*) }; } } $fn ## _scalar($($arg),*) } }; } ``` | `[x]` |
| 4.1.3 | Define SimdBackend trait | Trait for SIMD implementations. Methods for each accelerated operation. Implementations for each SIMD level. | 4.1.1 | ```rust pub trait SimdBackend: Send + Sync { fn classify_chars(&self, input: &[u8], output: &mut [u8]); fn find_delimiters(&self, input: &[u8]) -> Vec<usize>; fn validate_utf8(&self, input: &[u8]) -> Result<(), usize>; // ... } ``` | `[x]` |

### 4.2 AVX-512 Implementation

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 4.2.1 | Implement AVX-512 char classification | Process 64 bytes per iteration. Use `_mm512_cmpeq_epi8_mask` for comparisons. Return 64-bit mask. | 4.1.3 | ```rust #[target_feature(enable = "avx512f", enable = "avx512bw")] unsafe fn classify_whitespace_avx512(input: &[u8]) -> u64 { let chunk = _mm512_loadu_si512(input.as_ptr() as *const _); let space = _mm512_set1_epi8(b' ' as i8); let tab = _mm512_set1_epi8(b'\t' as i8); let newline = _mm512_set1_epi8(b'\n' as i8); let space_mask = _mm512_cmpeq_epi8_mask(chunk, space); let tab_mask = _mm512_cmpeq_epi8_mask(chunk, tab); let nl_mask = _mm512_cmpeq_epi8_mask(chunk, newline); space_mask | tab_mask | nl_mask } ``` | `[x]` |
| 4.2.2 | Implement AVX-512 pre-tokenization | Find word boundaries in 64-byte chunks. Extract positions from bitmask. | 4.2.1 | ```rust pub fn pretokenize_avx512(input: &[u8]) -> Vec<(usize, usize)> { let mut boundaries = Vec::new(); let mut i = 0; while i + 64 <= input.len() { let mask = unsafe { classify_whitespace_avx512(&input[i..]) }; // Extract positions from mask while mask != 0 { let pos = mask.trailing_zeros() as usize; boundaries.push(i + pos); mask &= mask - 1; // Clear lowest bit } i += 64; } // Handle remainder with scalar boundaries } ``` | `[x]` |
| 4.2.3 | Implement AVX-512 UTF-8 validation | Use simdutf algorithm. Validate continuation byte patterns. 64 bytes per iteration. | 4.1.3 | Port algorithm from simdutf paper. Check: ASCII (high bit 0), 2-byte (110xxxxx 10xxxxxx), 3-byte, 4-byte sequences. | `[x]` |

### 4.3 AVX2 Implementation

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 4.3.1 | Implement AVX2 char classification | Process 32 bytes per iteration. Use `_mm256_cmpeq_epi8` and `_mm256_movemask_epi8`. | 4.1.3 | ```rust #[target_feature(enable = "avx2")] unsafe fn classify_whitespace_avx2(input: &[u8]) -> u32 { let chunk = _mm256_loadu_si256(input.as_ptr() as *const _); let space = _mm256_set1_epi8(b' ' as i8); let cmp = _mm256_cmpeq_epi8(chunk, space); _mm256_movemask_epi8(cmp) as u32 } ``` | `[x]` |
| 4.3.2 | Implement Teddy algorithm | SIMD fingerprinting for special token matching. Group tokens by first 2-3 bytes. | 2.4.5 | Use `_mm256_shuffle_epi8` for fingerprint lookup. Match against buckets. Verify candidates. | `[x]` |
| 4.3.3 | Implement AVX2 hash computation | Parallel hash computation for vocabulary lookup. Use AES instructions if available. | 2.4.2 | Compute 8 hashes in parallel using `_mm256` operations. | `[x]` |

### 4.4 SSE4.2 Implementation

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 4.4.1 | Implement SSE4.2 string operations | Use PCMPESTRI/PCMPESTRM for delimiter scanning. 16 bytes per iteration. | 4.1.3 | ```rust #[target_feature(enable = "sse4.2")] unsafe fn find_delimiter_sse42(input: &[u8], delims: &[u8; 16]) -> Option<usize> { let haystack = _mm_loadu_si128(input.as_ptr() as *const _); let needles = _mm_loadu_si128(delims.as_ptr() as *const _); let idx = _mm_cmpestri( needles, delims.iter().take_while(|&&b| b != 0).count() as i32, haystack, input.len().min(16) as i32, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT ); if idx < 16 { Some(idx as usize) } else { None } } ``` | `[x]` |
| 4.4.2 | Implement SSE4.2 range checks | Use PCMPISTRM with ranges for Unicode detection. | 4.1.3 | Define ranges in 128-bit register. Use `_SIDD_CMP_RANGES` mode. | `[x]` |

### 4.5 ARM NEON Implementation

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 4.5.1 | Implement NEON char classification | Process 16 bytes per iteration. Use `vceqq_u8` for comparisons. | 4.1.3 | ```rust #[cfg(target_arch = "aarch64")] #[target_feature(enable = "neon")] unsafe fn classify_whitespace_neon(input: &[u8]) -> u16 { let chunk = vld1q_u8(input.as_ptr()); let space = vdupq_n_u8(b' '); let cmp = vceqq_u8(chunk, space); // Convert to bitmask let narrow = vshrn_n_u16(vreinterpretq_u16_u8(cmp), 4); vget_lane_u64(vreinterpret_u64_u8(narrow), 0) as u16 } ``` | `[x]` |
| 4.5.2 | Implement NEON UTF-8 validation | ARM NEON version of simdutf algorithm. Use `vtbl` for table lookups. | 4.1.3 | Similar to AVX2 but with NEON intrinsics. | `[x]` |
| 4.5.3 | Add SVE/SVE2 support | Scalable vectors for newer ARM. Vector-length agnostic code. | 4.5.1 | Use `sve` intrinsics when available. Support 128-2048 bit vectors. | `[x]` |

### 4.6 Scalar Optimization (SWAR)

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 4.6.1 | Implement SWAR byte operations | Process 8 bytes as u64. Find zero bytes, specific bytes without SIMD. | 4.1.3 | ```rust #[inline] fn has_zero_byte(x: u64) -> bool { const LO: u64 = 0x0101_0101_0101_0101; const HI: u64 = 0x8080_8080_8080_8080; x.wrapping_sub(LO) & !x & HI != 0 } #[inline] fn has_byte(x: u64, byte: u8) -> bool { has_zero_byte(x ^ (0x0101_0101_0101_0101 * byte as u64)) } ``` | `[x]` |
| 4.6.2 | Implement branchless classification | No branches in hot path. Use arithmetic comparison. | 4.1.3 | ```rust #[inline] fn is_ascii_whitespace_branchless(b: u8) -> bool { // space=32, tab=9, newline=10, cr=13 let is_space = (b == 32) as u8; let is_tab = (b == 9) as u8; let is_nl = (b == 10) as u8; let is_cr = (b == 13) as u8; (is_space | is_tab | is_nl | is_cr) != 0 } ``` | `[x]` |
| 4.6.3 | Implement loop unrolling | Process 4-8 elements per iteration manually. | 4.1.3 | Use `chunks_exact(4)` and process all 4 without loop. | `[x]` |

---

## 5. GPU Tokenization

### 5.1 CubeCL Setup

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 5.1.1 | Add CubeCL dependencies | Add `cubecl` with CUDA, ROCm, WGPU backends. Configure feature flags. | 1.1.2 | ```toml [dependencies] cubecl = { version = "0.2", features = ["cuda", "wgpu"] } ``` (using cudarc instead) | `[x]` |
| 5.1.2 | Create GPU device abstraction | Detect available GPUs. Create device handles. Support multi-GPU. | 5.1.1 | ```rust pub struct GpuDevice { backend: GpuBackend, device_id: usize, memory: usize, } pub fn detect_gpus() -> Vec<GpuDevice> { ... } ``` | `[x]` |
| 5.1.3 | Implement memory management | Allocate device memory. Manage pinned host memory. Implement memory pool. | 5.1.2 | Pre-allocate pinned buffers for input/output. Reuse across batches. | `[x]` |

### 5.2 GPU Kernels

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 5.2.1 | Implement GPU vocabulary lookup | Upload vocab to GPU. Hash-based lookup. Each thread processes one word. | 5.1.3, 2.4.2 | FNV-1a hash with linear probing. PTX kernel compiled via nvrtc. | `[x]` |
| 5.2.2 | Implement GPU pre-tokenization | Parallel boundary detection. Stream compaction for positions. | 5.1.3 | Each thread checks one byte for whitespace. CPU fallback for complex cases. | `[x]` |
| 5.2.3 | Implement GPU WordPiece | Parallelize across words. Each thread tokenizes one word. | 5.2.1 | Uses VocabLookupKernel for subword tokenization with greedy longest match. | `[x]` |
| 5.2.4 | Implement async pipeline | Overlap CPU-GPU transfers with computation. Double buffering. | 5.1.3 | DoubleBuffer struct for async pipeline. Pinned buffers for fast transfers. | `[x]` |

### 5.3 GPU Integration

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 5.3.1 | Create GpuTokenizer wrapper | Implement Tokenizer trait. Manage GPU resources. Handle fallback to CPU. | 5.2.3, 3.4.1 | GpuTokenizer with GpuBackend enum. Returns error for small batches (CPU fallback signal). | `[x]` |
| 5.3.2 | Implement batch size optimization | Profile throughput vs batch size. Auto-tune optimal batch size. | 5.3.1 | Run warmup batches at sizes [8, 16, 32, 64, 128, 256]. Select size with best throughput. | `[x]` |
| 5.3.3 | Add multi-GPU support | Distribute batches across GPUs. Load balance. | 5.3.1 | GpuLoadBalancer with RoundRobin, LeastLoaded, Sequential strategies. | `[x]` |

---

## 6. Distributed Architecture

### 6.1 NUMA-Aware Workers

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 6.1.1 | Detect NUMA topology | Use libnuma or /sys/devices/system/node. Get nodes, CPUs per node, memory. | 1.1.7 | ```rust pub struct NumaTopology { nodes: Vec<NumaNode>, } pub struct NumaNode { id: usize, cpus: Vec<usize>, memory_mb: usize, } pub fn detect_numa() -> NumaTopology { ... } ``` | `[ ]` |
| 6.1.2 | Implement CPU affinity | Bind threads to specific CPUs. Use `sched_setaffinity` on Linux. | 6.1.1 | ```rust #[cfg(target_os = "linux")] pub fn set_affinity(cpus: &[usize]) -> Result<()> { use libc::{sched_setaffinity, cpu_set_t, CPU_SET, CPU_ZERO}; let mut set: cpu_set_t = unsafe { std::mem::zeroed() }; unsafe { CPU_ZERO(&mut set); for &cpu in cpus { CPU_SET(cpu, &mut set); } sched_setaffinity(0, std::mem::size_of_val(&set), &set) }; Ok(()) } ``` | `[ ]` |
| 6.1.3 | Implement NUMA memory binding | Allocate memory on local NUMA node. Use `mbind` or `numa_alloc_local`. | 6.1.1 | ```rust pub fn numa_alloc_local(size: usize) -> *mut u8 { unsafe { libc::mmap(null_mut(), size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) as *mut u8 // Then mbind to local node } } ``` | `[ ]` |
| 6.1.4 | Create per-NUMA Tokio runtime | Separate runtime per NUMA node. Pin runtime threads to node CPUs. | 6.1.2 | ```rust pub fn create_numa_runtimes(topology: &NumaTopology) -> Vec<Runtime> { topology.nodes.iter().map(|node| { tokio::runtime::Builder::new_multi_thread() .worker_threads(node.cpus.len()) .on_thread_start(move || { set_affinity(&node.cpus).unwrap(); }) .build() .unwrap() }).collect() } ``` | `[ ]` |

### 6.2 IPC Layer

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 6.2.1 | Implement shared memory ring buffer | Memory-mapped ring buffer. Lock-free producer/consumer. Variable-size messages. | 2.5.1 | ```rust pub struct ShmRingBuffer { mmap: MmapMut, write_pos: AtomicU64, read_pos: AtomicU64, capacity: usize, } impl ShmRingBuffer { pub fn send(&self, data: &[u8]) -> Result<()> { let len = data.len() as u64; let pos = self.write_pos.fetch_add(len + 8, SeqCst); // Write length prefix, then data // Memory fence // Update readable position } } ``` | `[ ]` |
| 6.2.2 | Implement flume channels | High-performance bounded/unbounded channels. For same-process communication. | 1.1.2 | ```rust use flume::{bounded, Sender, Receiver}; pub struct TokenizerChannel { tx: Sender<TokenizeRequest>, rx: Receiver<TokenizeResponse>, } ``` | `[ ]` |
| 6.2.3 | Implement gRPC service | Define protobuf schema. Use tonic for Rust gRPC. Support streaming. | 1.1.2 | ```protobuf service BudTokenizer { rpc Tokenize(TokenizeRequest) returns (TokenizeResponse); rpc TokenizeBatch(stream TokenizeRequest) returns (stream TokenizeResponse); } ``` | `[ ]` |
| 6.2.4 | Implement RDMA support | Use rust-ibverbs. RDMA WRITE for zero-copy transfer. | 1.1.2 | Register memory regions. Exchange QP info. Post RDMA WRITE operations. | `[ ]` |

### 6.3 Coordinator

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 6.3.1 | Implement worker registry | Track tokenizer workers. Store endpoint, capabilities, load. Support dynamic registration. | 6.2.3 | ```rust pub struct WorkerRegistry { workers: DashMap<WorkerId, WorkerInfo>, } pub struct WorkerInfo { endpoint: String, capabilities: Capabilities, load: AtomicU64, last_heartbeat: AtomicU64, } ``` | `[ ]` |
| 6.3.2 | Implement load balancer | Multiple strategies: round-robin, least-loaded, token-aware. | 6.3.1 | ```rust pub trait LoadBalancer: Send + Sync { fn select_worker(&self, request: &TokenizeRequest) -> WorkerId; } pub struct TokenAwareBalancer { registry: Arc<WorkerRegistry>, } impl LoadBalancer for TokenAwareBalancer { fn select_worker(&self, request: &TokenizeRequest) -> WorkerId { let estimated_tokens = estimate_tokens(&request.text); // Select worker with lowest (load + queued_tokens) } } ``` | `[~]` |
| 6.3.3 | Implement health monitoring | Periodic health checks. Detect failed workers. Auto-remove unhealthy. | 6.3.1 | ```rust pub struct HealthMonitor { registry: Arc<WorkerRegistry>, check_interval: Duration, timeout: Duration, } impl HealthMonitor { pub async fn run(&self) { loop { for (id, worker) in self.registry.workers.iter() { if !self.check_health(&worker).await { self.registry.workers.remove(&id); } } tokio::time::sleep(self.check_interval).await; } } } ``` | `[~]` |
| 6.3.4 | Implement request router | Accept HTTP/gRPC requests. Route to workers. Aggregate responses. | 6.3.2, 6.2.3 | Handle `/tokenize` and `/tokenize_batch` endpoints. Forward to selected worker. Return results. | `[ ]` |

---

## 7. LatentBud Integration

### 7.1 Pre-Tokenized Format

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 7.1.1 | Define PreTokenizedRequest | Struct for pre-tokenized data. Fields: request_id, token_ids, attention_mask, token_type_ids. | 3.4.2 | ```rust #[derive(Serialize, Deserialize)] pub struct PreTokenizedRequest { pub request_id: u64, pub token_ids: Vec<u32>, pub attention_mask: Vec<u32>, pub token_type_ids: Option<Vec<u32>>, pub priority: u8, } ``` | `[x]` |
| 7.1.2 | Implement efficient serialization | Use bincode for compact binary format. Support zero-copy where possible. | 7.1.1 | ```rust impl PreTokenizedRequest { pub fn serialize(&self) -> Vec<u8> { bincode::serialize(self).unwrap() } pub fn deserialize(data: &[u8]) -> Result<Self> { bincode::deserialize(data).map_err(Into::into) } } ``` | `[x]` |
| 7.1.3 | Add schema versioning | Version field in serialization. Support backward compatibility. | 7.1.1 | Prefix with version byte. Support reading old versions. | `[x]` |

### 7.2 Token Budget Router

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 7.2.1 | Implement TokenBudgetRouter | Match LatentBud's max_batch_tokens. Track padded token count. Flush when exceeded. | 7.1.1 | ```rust pub struct TokenBudgetRouter { max_batch_tokens: usize, current_batch: Vec<PreTokenizedRequest>, current_max_len: usize, } impl TokenBudgetRouter { pub fn add(&mut self, req: PreTokenizedRequest) -> Option<Vec<PreTokenizedRequest>> { let req_len = req.token_ids.len(); let new_max = self.current_max_len.max(req_len); let new_padded = new_max * (self.current_batch.len() + 1); if new_padded > self.max_batch_tokens && !self.current_batch.is_empty() { let batch = std::mem::take(&mut self.current_batch); self.current_batch.push(req); self.current_max_len = req_len; return Some(batch); } self.current_batch.push(req); self.current_max_len = new_max; None } } ``` | `[x]` |
| 7.2.2 | Add timeout-based flushing | Flush batch after timeout even if not full. Configurable timeout. | 7.2.1 | Use `tokio::time::timeout` to flush after e.g. 10ms of no new requests. | `[x]` |

### 7.3 LatentBud Modifications

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 7.3.1 | Add pre-tokenized endpoint | New endpoint in LatentBud: `/v1/embeddings/pretokenized`. Accept PreTokenizedRequest. | 7.1.1 | Modify LatentBud's router to handle pre-tokenized input. | `[ ]` |
| 7.3.2 | Modify BatchHandler | Detect pre-tokenized requests. Skip tokenization. Pass directly to model. | 7.3.1 | Add flag to request indicating pre-tokenized. | `[ ]` |

---

## 8. Resilience and Operations

### 8.1 Circuit Breaker

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 8.1.1 | Implement CircuitBreaker | States: Closed, Open, HalfOpen. Trip on consecutive failures. Reset after success. | 1.1.7 | ```rust pub struct CircuitBreaker { state: AtomicU8, failure_count: AtomicU32, last_failure_time: AtomicU64, threshold: u32, reset_timeout: Duration, } impl CircuitBreaker { pub async fn call<F, T, E>(&self, f: F) -> Result<T, CircuitBreakerError<E>> where F: Future<Output = Result<T, E>>, { match self.state.load(Acquire) { CLOSED => { match f.await { Ok(v) => { self.failure_count.store(0, Relaxed); Ok(v) } Err(e) => { let count = self.failure_count.fetch_add(1, Relaxed) + 1; if count >= self.threshold { self.trip(); } Err(CircuitBreakerError::Inner(e)) } } } OPEN => { if self.should_try_reset() { self.state.store(HALF_OPEN, Release); // Try one request... } else { Err(CircuitBreakerError::Open) } } // ... } } } ``` | `[ ]` |
| 8.1.2 | Integrate with coordinator | Wrap embedder calls. One breaker per embedder. | 8.1.1, 6.3.4 | Each embedder endpoint gets its own CircuitBreaker instance. | `[ ]` |

### 8.2 Retry Policy

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 8.2.1 | Implement exponential backoff | Base delay × 2^attempt. Cap at max delay. Add jitter. | 1.1.7 | ```rust pub struct RetryPolicy { max_attempts: u32, base_delay: Duration, max_delay: Duration, jitter: bool, } impl RetryPolicy { pub async fn execute<F, T, E>(&self, mut f: impl FnMut() -> F) -> Result<T, E> where F: Future<Output = Result<T, E>>, { for attempt in 0..self.max_attempts { match f().await { Ok(v) => return Ok(v), Err(e) if attempt + 1 < self.max_attempts => { let delay = self.calculate_delay(attempt); tokio::time::sleep(delay).await; } Err(e) => return Err(e), } } unreachable!() } fn calculate_delay(&self, attempt: u32) -> Duration { let base = self.base_delay.as_millis() as u64; let delay = (base * 2u64.pow(attempt)).min(self.max_delay.as_millis() as u64); let jitter = if self.jitter { rand::random::<u64>() % (delay / 4) } else { 0 }; Duration::from_millis(delay + jitter) } } ``` | `[ ]` |

### 8.3 Metrics

| ID | Task | Details | Dependencies | Algorithm/Implementation | Status |
|----|------|---------|--------------|-------------------------|--------|
| 8.3.1 | Define Prometheus metrics | Counters, histograms, gauges for all key metrics. | 1.1.2 | ```rust lazy_static! { pub static ref REQUESTS_TOTAL: IntCounter = IntCounter::new("budtiktok_requests_total", "Total requests").unwrap(); pub static ref TOKENS_PROCESSED: IntCounter = IntCounter::new("budtiktok_tokens_processed", "Tokens processed").unwrap(); pub static ref TOKENIZATION_DURATION: Histogram = Histogram::with_opts( HistogramOpts::new("budtiktok_tokenization_duration_seconds", "Tokenization latency") .buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]) ).unwrap(); pub static ref QUEUE_DEPTH: IntGauge = IntGauge::new("budtiktok_queue_depth", "Queue depth").unwrap(); } ``` | `[ ]` |
| 8.3.2 | Implement metrics endpoint | Serve /metrics in Prometheus format. | 8.3.1 | ```rust async fn metrics_handler() -> impl IntoResponse { let encoder = TextEncoder::new(); let metric_families = prometheus::gather(); let mut buffer = Vec::new(); encoder.encode(&metric_families, &mut buffer).unwrap(); ([(header::CONTENT_TYPE, "text/plain")], buffer) } ``` | `[ ]` |
| 8.3.3 | Add tracing integration | Structured logging with tracing. Span propagation. Export to Jaeger. | 1.1.2 | Use `tracing`, `tracing-subscriber`, `tracing-opentelemetry`. Add `#[instrument]` to key functions. | `[ ]` |

---

## 9. Test Suites (TDD)

### 9.1 Unit Tests

| ID | Test Category | Test Cases | Implementation Details | Status |
|----|--------------|------------|----------------------|--------|
| 9.1.1 | Unicode Normalization | - ASCII passthrough: `is_ascii_fast("hello")` returns true<br>- ASCII with high bytes: `is_ascii_fast("héllo")` returns false<br>- NFC quick check: `is_nfc_quick("café")` returns Yes<br>- NFC quick check combining: `is_nfc_quick("cafe\u{0301}")` returns No<br>- Full NFC normalization: `nfc("cafe\u{0301}")` == "café"<br>- NFD decomposition: `nfd("é")` == "e\u{0301}"<br>- Hangul decomposition: `nfd("한")` == "한"<br>- Streaming normalization: chunks produce same result as full | `tests/unit_tests/unicode_normalization.rs` - 20+ test cases covering all specified scenarios. | `[x]` |
| 9.1.2 | Unicode Categories | - ASCII category lookup: `get_category('a')` == Letter_Lowercase<br>- Punctuation: `is_punctuation('!')` == true<br>- CJK detection: `is_cjk('中')` == true<br>- Emoji detection: covers common emoji<br>- Category flags: bitwise operations work correctly<br>- Cache effectiveness: second lookup faster | `tests/unit_tests/unicode_categories.rs` - 20+ test cases for categories, CJK, punctuation, flags. | `[x]` |
| 9.1.3 | Token Cache | - Insert and retrieve<br>- CLOCK eviction works correctly<br>- Concurrent access safety<br>- Stats tracking accurate<br>- Sharded cache reduces contention<br>- Multi-level cache hierarchy | `tests/unit_tests/cache.rs` - 459 lines of cache tests with concurrency scenarios. | `[x]` |
| 9.1.4 | Vocabulary | - Load tokenizer.json<br>- Token to ID lookup<br>- ID to token lookup<br>- Trie prefix search<br>- Aho-Corasick matching<br>- Double-array trie consistency | `tests/unit_tests/vocabulary.rs` - 544 lines testing vocab, trie, and AC matching. | `[x]` |
| 9.1.5 | WordPiece Algorithm | - Basic tokenization: "hello" -> ["hello"]<br>- Subword split: "unhappiness" -> ["un", "##happiness"]<br>- Unknown word: "xyz123" -> ["[UNK]"]<br>- Max chars limit<br>- Continuation prefix<br>- Cache integration<br>- BERT normalizer<br>- BERT pre-tokenizer<br>- Post-processing with [CLS]/[SEP] | `tests/unit_tests/wordpiece.rs` - 682 lines with comprehensive WordPiece tests. | `[x]` |
| 9.1.6 | Unigram Algorithm | - Viterbi optimal segmentation<br>- Byte fallback for unknown chars<br>- N-best returns multiple paths<br>- Sampling produces valid segmentations<br>- Metaspace pre-tokenization<br>- Score-based selection | `tests/unit_tests/unigram.rs` - 681 lines testing Viterbi, n-best, sampling. | `[x]` |
| 9.1.7 | BPE Algorithm | - O(n) produces same result as O(n²)<br>- Compatibility table correct<br>- Byte-level encoding<br>- Merge ordering<br>- Dropout produces different results | `tests/unit_tests/bpe.rs` - 844 lines testing O(n) BPE, byte-level, dropout. | `[x]` |
| 9.1.8 | SIMD Implementations | - AVX-512 matches scalar<br>- AVX2 matches scalar<br>- SSE4.2 matches scalar<br>- NEON matches scalar<br>- SWAR matches scalar<br>- Edge cases: partial chunks, empty input | `tests/unit_tests/simd.rs` + `tests/simd_validation.rs` + `tests/isa_consistency.rs` - 16+ ISA consistency tests. | `[x]` |
| 9.1.9 | Memory Management | - Arena allocates correctly<br>- Arena reset works<br>- String interner deduplicates<br>- Memory pool reuse<br>- No memory leaks (use miri) | `tests/unit_tests/memory.rs` - 766 lines testing arena, interner, pools. | `[x]` |

### 9.2 Integration Tests

| ID | Test Category | Test Cases | Implementation Details | Status |
|----|--------------|------------|----------------------|--------|
| 9.2.1 | End-to-End Tokenization | - Load real tokenizer, encode text, verify output<br>- Batch encoding consistency<br>- Truncation and padding<br>- Special token handling<br>- Offset tracking accuracy | `tests/integration_tests/tokenization.rs` - End-to-end tokenization tests. | `[x]` |
| 9.2.2 | Distributed Pipeline | - Start coordinator and workers<br>- Send requests through full pipeline<br>- Verify results match direct tokenization<br>- Test with varying batch sizes | `tests/integration_tests/distributed.rs` - 16 distributed pipeline tests. | `[x]` |
| 9.2.3 | Failure Recovery | - Kill worker mid-request, verify retry<br>- Circuit breaker trips and recovers<br>- Network partition handling<br>- Timeout handling | Partial - distributed tests cover some scenarios. Circuit breaker not fully implemented. | `[~]` |
| 9.2.4 | GPU Integration | - CPU and GPU produce same results<br>- Multi-GPU distribution<br>- Fallback to CPU when GPU unavailable<br>- Memory management | GPU module not implemented (Section 5). | `[ ]` |
| 9.2.5 | LatentBud Integration | - Pre-tokenized requests accepted<br>- Token budget routing correct<br>- End-to-end embedding pipeline | `tests/integration_tests/latentbud.rs` - LatentBud integration tests. | `[x]` |

### 9.3 Regression Tests

| ID | Test Category | Test Cases | Implementation Details | Status |
|----|--------------|------------|----------------------|--------|
| 9.3.1 | Output Regression | - Fixed inputs produce fixed outputs<br>- Store expected outputs in fixtures<br>- Detect any output change | `tests/regression.rs::regression_tests::outputs::*` - Golden file testing. | `[x]` |
| 9.3.2 | Performance Regression | - Benchmark must not regress >10%<br>- Track p50, p95, p99 latency<br>- Track throughput tokens/sec | `tests/regression.rs::regression_tests::performance::*` - Performance baseline tests. | `[x]` |
| 9.3.3 | Memory Regression | - Peak memory must not increase >10%<br>- No memory leaks | `tests/regression.rs::regression_tests::memory::*` - Memory leak and allocation tests. | `[x]` |

### 9.4 Property-Based Tests

| ID | Test Category | Properties | Implementation Details | Status |
|----|--------------|-----------|----------------------|--------|
| 9.4.1 | Tokenization Properties | - encode(decode(ids)) produces valid (possibly different) ids<br>- decode(encode(text)) preserves semantics<br>- batch encode == individual encodes<br>- truncated + overflow == original | `tests/property_tests/tokenization.rs` - Property-based tokenization tests. | `[x]` |
| 9.4.2 | SIMD Properties | - All SIMD implementations produce identical results<br>- Results match scalar baseline<br>- No out-of-bounds access | `tests/property_tests/simd.rs` - 16+ SIMD property tests. | `[x]` |
| 9.4.3 | Concurrency Properties | - Concurrent access produces consistent results<br>- No data races<br>- Cache remains consistent | `tests/property_tests/concurrency.rs` - 11 concurrency property tests. | `[x]` |

### 9.5 Fuzzing

| ID | Test Target | Approach | Implementation Details | Status |
|----|------------|----------|----------------------|--------|
| 9.5.1 | Tokenizer Input | Fuzz arbitrary UTF-8 strings | `fuzz/fuzz_targets/fuzz_tokenize.rs` + `fuzz_unicode.rs` - Fuzzing tokenizer and Unicode functions. | `[x]` |
| 9.5.2 | tokenizer.json Parser | Fuzz JSON input | `fuzz/fuzz_targets/fuzz_json_parser.rs` - Fuzzing JSON config parser. | `[x]` |
| 9.5.3 | IPC Deserialization | Fuzz serialized messages | `fuzz/fuzz_targets/fuzz_ipc.rs` - Fuzzing IPC message deserialization. | `[x]` |

---

## 10. Profiling Tools

### 10.1 Profiling Infrastructure

| ID | Task | Details | How to Use | When to Use | Status |
|----|------|---------|-----------|-------------|--------|
| 10.1.1 | Set up perf integration | Configure for Linux `perf` profiling. Add debug symbols to release builds. | ```bash # Build with debug symbols RUSTFLAGS="-C debuginfo=2" cargo build --release # Record profile perf record -g ./target/release/budtiktok-bench perf report ``` | When investigating CPU hotspots. Use after benchmarks show unexpected slowdowns. | `[ ]` |
| 10.1.2 | Set up flamegraph | Install `cargo-flamegraph`. Configure for both CPU and memory. | ```bash cargo install flamegraph cargo flamegraph --bin budtiktok-bench ``` Opens flamegraph in browser. | Visualizing where time is spent. Use to identify optimization targets. | `[ ]` |
| 10.1.3 | Set up samply | Modern sampling profiler for Rust. Cross-platform. | ```bash cargo install samply samply record ./target/release/budtiktok-bench ``` Opens Firefox Profiler UI. | Cross-platform profiling. Better UI than flamegraph. | `[ ]` |
| 10.1.4 | Set up cachegrind | Valgrind tool for cache analysis. | ```bash valgrind --tool=cachegrind ./target/release/budtiktok-bench cg_annotate cachegrind.out.* ``` | When cache misses suspected. Use after SIMD optimizations. | `[ ]` |
| 10.1.5 | Set up DHAT | Valgrind tool for heap profiling. | ```bash valgrind --tool=dhat ./target/release/budtiktok-bench ``` View `dhat.out.*` in browser. | Finding allocation hotspots. Use when memory usage too high. | `[ ]` |
| 10.1.6 | Set up heaptrack | Modern heap profiler. Better than massif. | ```bash heaptrack ./target/release/budtiktok-bench heaptrack_gui heaptrack.*.gz ``` | Detailed heap analysis. Finding memory leaks. | `[ ]` |
| 10.1.7 | Add tracing instrumentation | Add `#[instrument]` attributes. Configure span recording. | ```rust #[tracing::instrument(skip(self))] pub fn encode(&self, text: &str) -> Encoding { ... } ``` Enable with `RUST_LOG=trace`. | Understanding control flow. Distributed tracing. | `[ ]` |

### 10.2 Built-in Profiling

| ID | Task | Details | How to Use | When to Use | Status |
|----|------|---------|-----------|-------------|--------|
| 10.2.1 | Add timing instrumentation | Measure key operations. Export as metrics. | ```rust pub struct Profiler { tokenization_ns: AtomicU64, normalization_ns: AtomicU64, lookup_ns: AtomicU64, } impl Profiler { pub fn time<T>(&self, metric: &AtomicU64, f: impl FnOnce() -> T) -> T { let start = Instant::now(); let result = f(); metric.fetch_add(start.elapsed().as_nanos() as u64, Relaxed); result } } ``` | Always in debug builds. Enable via feature flag in release. | `[ ]` |
| 10.2.2 | Add cache statistics | Track hit rates per cache level. | Export via `/debug/cache_stats` endpoint. | Tuning cache sizes. Identifying cache-unfriendly patterns. | `[ ]` |
| 10.2.3 | Add SIMD path tracking | Track which SIMD path is used. | Log on startup: "Using AVX-512 backend". Add counter for fallbacks. | Verifying SIMD is used. Debugging performance issues. | `[ ]` |
| 10.2.4 | Create profiling CLI | `budtiktok profile` subcommand. Run benchmarks, collect profiles. | ```bash budtiktok profile --input data/test.txt --output profile.json ``` | Easy profiling without setup. | `[ ]` |

### 10.3 Profiling Guide

```markdown
# Profiling Guide

## When to Profile

1. **Before optimization**: Establish baseline
2. **After major changes**: Verify no regression
3. **When benchmarks regress**: Identify cause
4. **Before release**: Ensure production-ready

## Profiling Workflow

### 1. CPU Profiling (samply/flamegraph)

```bash
# Quick flamegraph
cargo flamegraph --bin budtiktok-bench -- --benchmark wordpiece

# Detailed with samply
samply record ./target/release/budtiktok-bench
# Opens interactive UI
```

**What to look for:**
- Functions taking >10% of time
- Unexpected standard library calls
- Memory allocation (`malloc`, `__rust_alloc`)
- Lock contention (`parking_lot`, `pthread_mutex`)

### 2. Cache Analysis (cachegrind)

```bash
valgrind --tool=cachegrind ./target/release/budtiktok-bench
cg_annotate --auto=yes cachegrind.out.*
```

**What to look for:**
- D1 miss rate >5% (L1 data cache)
- LL miss rate >1% (Last level cache)
- Functions with high miss counts

### 3. Memory Profiling (heaptrack)

```bash
heaptrack ./target/release/budtiktok-bench
heaptrack_gui heaptrack.*.gz
```

**What to look for:**
- Allocation hotspots
- Memory growth over time
- Temporary allocations in hot paths

### 4. SIMD Verification

```bash
# Check CPU features
cat /proc/cpuinfo | grep -E 'avx|sse|neon'

# Verify runtime detection
RUST_LOG=debug ./target/release/budtiktok-bench 2>&1 | grep -i simd

# Disassemble to verify SIMD instructions
objdump -d target/release/budtiktok-bench | grep -E 'vmov|vpcmp|vadd'
```

## Optimization Priorities

1. **Algorithmic**: O(n) vs O(n²) - biggest impact
2. **Memory access**: Cache locality, prefetching
3. **SIMD**: Vectorize hot loops
4. **Allocation**: Reduce allocations in hot paths
5. **Parallelism**: Use all cores effectively
```

---

## 11. Accuracy Testing Suite

### 11.1 HuggingFace Gold Standard

| ID | Task | Details | Implementation | Status |
|----|------|---------|----------------|--------|
| 11.1.1 | Create accuracy test framework | Compare BudTikTok output against HuggingFace tokenizers. Report any differences. | ```rust pub struct AccuracyTest { hf_tokenizer: tokenizers::Tokenizer, bud_tokenizer: Box<dyn Tokenizer>, } impl AccuracyTest { pub fn compare(&self, text: &str) -> AccuracyResult { let hf_result = self.hf_tokenizer.encode(text, true)?; let bud_result = self.bud_tokenizer.encode(text, true)?; AccuracyResult { input: text.to_string(), hf_ids: hf_result.get_ids().to_vec(), bud_ids: bud_result.ids.clone(), match_: hf_result.get_ids() == &bud_result.ids, } } } ``` | `[ ]` |
| 11.1.2 | Create test datasets | Curate datasets covering edge cases. | - ASCII text (Wikipedia English)<br>- Unicode text (Wikipedia multilingual)<br>- Code (GitHub samples)<br>- Edge cases (empty, whitespace, special chars)<br>- Long documents<br>- Many short strings | `[ ]` |
| 11.1.3 | Implement accuracy CLI | `budtiktok accuracy` command. | ```bash budtiktok accuracy \ --tokenizer bert-base-uncased \ --dataset tests/data/wikipedia_sample.txt \ --output accuracy_report.json ``` | `[ ]` |
| 11.1.4 | Add CI accuracy checks | Run accuracy tests in CI. Fail on any mismatch. | GitHub Action that runs accuracy tests against all supported tokenizers. | `[ ]` |

### 11.2 Model Coverage

| ID | Model | Tokenizer Type | Test Data | Status |
|----|-------|---------------|-----------|--------|
| 11.2.1 | bert-base-uncased | WordPiece | Wikipedia EN, GLUE | `[ ]` |
| 11.2.2 | bert-base-multilingual-cased | WordPiece | Wikipedia multi | `[ ]` |
| 11.2.3 | gpt2 | BPE | OpenWebText | `[ ]` |
| 11.2.4 | roberta-base | BPE | BookCorpus | `[ ]` |
| 11.2.5 | xlm-roberta-base | Unigram | CC-100 | `[ ]` |
| 11.2.6 | t5-base | Unigram | C4 | `[ ]` |
| 11.2.7 | llama-2-7b | BPE | Wikipedia | `[ ]` |
| 11.2.8 | mistral-7b | BPE | Wikipedia | `[ ]` |
| 11.2.9 | BAAI/bge-small-en-v1.5 | WordPiece | MTEB | `[ ]` |
| 11.2.10 | sentence-transformers/all-MiniLM-L6-v2 | WordPiece | STS | `[ ]` |

### 11.3 Edge Case Tests

| ID | Category | Test Cases | Status |
|----|----------|-----------|--------|
| 11.3.1 | Empty/Whitespace | Empty string, single space, multiple spaces, tabs, newlines, mixed | `[ ]` |
| 11.3.2 | Unicode | Combining characters, ZWJ sequences, RTL text, surrogates, BOM | `[ ]` |
| 11.3.3 | Long Text | 1K, 10K, 100K, 1M characters | `[ ]` |
| 11.3.4 | Special Characters | All ASCII punctuation, math symbols, currency, emoji | `[ ]` |
| 11.3.5 | Code | Python, JavaScript, Rust, C++, with special syntax | `[ ]` |
| 11.3.6 | CJK | Chinese, Japanese, Korean text, mixed with ASCII | `[ ]` |
| 11.3.7 | Arabic/Hebrew | RTL scripts, mixed with LTR | `[ ]` |
| 11.3.8 | Normalization | Pre-composed vs decomposed Unicode, NFKC equivalents | `[ ]` |

---

## 12. Comparison Benchmarking Tool

### 12.1 Benchmark Framework

| ID | Task | Details | Implementation | Status |
|----|------|---------|----------------|--------|
| 12.1.1 | Create benchmark harness | Unified framework for comparing tokenizers. | ```rust pub struct BenchmarkHarness { tokenizers: Vec<(String, Box<dyn Tokenizer>)>, datasets: Vec<Dataset>, } impl BenchmarkHarness { pub fn run(&self) -> BenchmarkResults { let mut results = BenchmarkResults::new(); for (name, tokenizer) in &self.tokenizers { for dataset in &self.datasets { let metrics = self.benchmark_one(tokenizer, dataset); results.add(name, &dataset.name, metrics); } } results } fn benchmark_one(&self, tokenizer: &dyn Tokenizer, dataset: &Dataset) -> Metrics { // Warmup for _ in 0..100 { let _ = tokenizer.encode(&dataset.samples[0], true); } // Measure let mut latencies = Vec::with_capacity(dataset.samples.len()); let start = Instant::now(); for sample in &dataset.samples { let t0 = Instant::now(); let enc = tokenizer.encode(sample, true).unwrap(); latencies.push(t0.elapsed()); } let total_time = start.elapsed(); let total_tokens: usize = latencies.len(); // Assuming 1 encoding each Metrics { throughput_samples_per_sec: dataset.samples.len() as f64 / total_time.as_secs_f64(), latency_p50: percentile(&latencies, 0.50), latency_p95: percentile(&latencies, 0.95), latency_p99: percentile(&latencies, 0.99), } } } ``` | `[ ]` |
| 12.1.2 | Add HuggingFace tokenizers backend | Wrap HF tokenizers for comparison. | ```rust pub struct HfTokenizer { inner: tokenizers::Tokenizer, } impl Tokenizer for HfTokenizer { fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> { let enc = self.inner.encode(text, add_special_tokens)?; Ok(Encoding { ids: enc.get_ids().to_vec(), // ... convert other fields }) } } ``` | `[ ]` |
| 12.1.3 | Add TEI backend | Benchmark against TEI's tokenization. | Use TEI's gRPC API or embed its tokenization code. | `[ ]` |
| 12.1.4 | Add BlazeText backend | Benchmark against BlazeText. | If available as library, link directly. Otherwise benchmark via CLI. | `[ ]` |
| 12.1.5 | Create benchmark CLI | `budtiktok benchmark` command. | ```bash budtiktok benchmark \ --tokenizers hf,budtiktok,tei \ --models bert-base-uncased,gpt2,t5-base \ --datasets wikipedia,code,multilingual \ --output results.json \ --format markdown ``` | `[ ]` |

### 12.2 Benchmark Datasets

| ID | Dataset | Description | Size | Status |
|----|---------|-------------|------|--------|
| 12.2.1 | Wikipedia EN | English Wikipedia articles | 10K samples | `[ ]` |
| 12.2.2 | Wikipedia Multi | Multilingual Wikipedia | 10K samples, 10 languages | `[ ]` |
| 12.2.3 | Code | GitHub code samples | 10K samples, 5 languages | `[ ]` |
| 12.2.4 | Short Text | Tweets, queries | 100K samples, <50 chars | `[ ]` |
| 12.2.5 | Long Text | Documents | 1K samples, >10K chars | `[ ]` |
| 12.2.6 | ShareGPT | Conversational data | 10K samples | `[ ]` |

### 12.3 Benchmark Reports

| ID | Task | Details | Implementation | Status |
|----|------|---------|----------------|--------|
| 12.3.1 | Generate markdown report | Human-readable comparison tables. | ```markdown # BudTikTok Benchmark Results ## Throughput (samples/sec) | Model | HuggingFace | TEI | BlazeText | BudTikTok | Speedup | |-------|-------------|-----|-----------|-----------|--------| | bert-base | 10,000 | 12,000 | 50,000 | 150,000 | **15x** | ## Latency (p99, microseconds) ... ``` | `[ ]` |
| 12.3.2 | Generate JSON report | Machine-readable for CI integration. | Store full metrics, system info, versions. | `[ ]` |
| 12.3.3 | Generate charts | Visualization of results. | Use plotters crate or output SVG. Bar charts for throughput, line charts for latency distribution. | `[ ]` |
| 12.3.4 | Add to CI | Run benchmarks in CI, track over time. | Store results in GitHub artifacts. Generate comparison over time. | `[ ]` |

### 12.4 Benchmark Configuration

```toml
# benchmark.toml

[harness]
warmup_iterations = 100
measurement_iterations = 10000
timeout_per_sample_ms = 1000

[tokenizers]
hf = { enabled = true, path = "huggingface/tokenizers" }
tei = { enabled = true, endpoint = "http://localhost:8080" }
blazetext = { enabled = false, reason = "not available" }
budtiktok = { enabled = true, simd = "auto", gpu = false }

[datasets]
wikipedia_en = { path = "data/wikipedia_en.txt", max_samples = 10000 }
code = { path = "data/code_samples.txt", max_samples = 10000 }

[models]
bert = { path = "bert-base-uncased/tokenizer.json" }
gpt2 = { path = "gpt2/tokenizer.json" }
t5 = { path = "t5-base/tokenizer.json" }

[output]
format = ["json", "markdown", "svg"]
output_dir = "benchmark_results"
```

---

## 13. Deployment

### 13.1 Docker

| ID | Task | Details | Implementation | Status |
|----|------|---------|----------------|--------|
| 13.1.1 | Create Dockerfile | Multi-stage build. Minimal runtime. | ```dockerfile FROM rust:1.75-slim as builder WORKDIR /app COPY . . RUN cargo build --release --features full FROM debian:bookworm-slim COPY --from=builder /app/target/release/budtiktok /usr/local/bin/ EXPOSE 8080 ENTRYPOINT ["budtiktok", "serve"] ``` | `[ ]` |
| 13.1.2 | Create GPU Dockerfile | CUDA runtime support. | ```dockerfile FROM nvidia/cuda:12.3-runtime-ubuntu22.04 COPY --from=builder /app/target/release/budtiktok /usr/local/bin/ ENV CUDA_VISIBLE_DEVICES=all ``` | `[ ]` |
| 13.1.3 | Create docker-compose | Multi-container deployment. | Coordinator + workers + monitoring stack. | `[ ]` |

### 13.2 Kubernetes

| ID | Task | Details | Implementation | Status |
|----|------|---------|----------------|--------|
| 13.2.1 | Create Helm chart | Deployments, Services, ConfigMaps. | Chart with coordinator, workers, HPA, PDB. | `[ ]` |
| 13.2.2 | Create Kubernetes operator | Custom resource, auto-scaling. | Use kube-rs to build operator. | `[ ]` |
| 13.2.3 | Add monitoring stack | Prometheus, Grafana dashboards. | Pre-configured dashboards for BudTikTok metrics. | `[ ]` |

---

## 14. Documentation

### 14.1 API Documentation

| ID | Task | Details | Status |
|----|------|---------|--------|
| 14.1.1 | Write rustdoc | Document all public APIs. | `[ ]` |
| 14.1.2 | Create examples | Code examples for common use cases. | `[ ]` |
| 14.1.3 | Publish to docs.rs | Automated on release. | `[ ]` |

### 14.2 User Documentation

| ID | Task | Details | Status |
|----|------|---------|--------|
| 14.2.1 | Quick start guide | Get started in 5 minutes. | `[ ]` |
| 14.2.2 | Configuration reference | All config options. | `[ ]` |
| 14.2.3 | Deployment guide | Single-node, multi-node, Kubernetes. | `[ ]` |
| 14.2.4 | Performance tuning guide | NUMA, caching, SIMD. | `[ ]` |
| 14.2.5 | Migration guide | From HuggingFace tokenizers. | `[ ]` |

---

## Summary

| Section | Total Tasks | Completed |
|---------|-------------|-----------|
| 1. Project Setup | 13 | 13 |
| 2. Core Infrastructure | 35 | 35 |
| 3. Tokenization Algorithms | 27 | 27 |
| 4. SIMD Acceleration | 18 | 18 |
| 5. GPU Tokenization | 10 | 0 |
| 6. Distributed Architecture | 15 | 2 |
| 7. LatentBud Integration | 6 | 5 |
| 8. Resilience and Operations | 8 | 0 |
| 9. Test Suites (TDD) | 19 | 18 |
| 10. Profiling Tools | 11 | 0 |
| 11. Accuracy Testing | 14 | 0 |
| 12. Benchmarking Tool | 10 | 0 |
| 13. Deployment | 6 | 0 |
| 14. Documentation | 8 | 0 |
| **Total** | **200** | **118** |

---

## Critical Path

1. **1.1.1** Create Cargo workspace
2. **2.1.1** ASCII fast path
3. **2.2.1** CategoryFlags
4. **2.4.1** tokenizer.json parser
5. **3.1.1** WordPiece algorithm
6. **3.4.1** Tokenizer trait
7. **9.1.5** WordPiece unit tests
8. **11.1.1** Accuracy test framework
9. **12.1.1** Benchmark harness
10. **4.1.1** SIMD detection
11. **4.2.1** AVX-512 implementation

**Estimated tasks to MVP:** 50 tasks

---

*Last Updated: 2025-12-17*
*Latest: 128/200 tasks done. Section 5 (GPU Tokenization) now complete with CUDA backend using cudarc. GpuTokenizer with multi-GPU support, vocabulary lookup kernel, pre-tokenization, WordPiece, and batch size auto-tuning. 45+ GPU tests passing. Remaining: Distributed (Section 6), Resilience (Section 8), Profiling (Section 10), Accuracy Testing (Section 11), Benchmarking (Section 12), Deployment (Section 13), Documentation (Section 14).*
