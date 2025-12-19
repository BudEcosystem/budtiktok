# BudTikTok

A high-performance tokenization library for NLP/LLM applications, written in Rust.

BudTikTok provides blazing-fast tokenization with support for multiple tokenizer types, SIMD acceleration, and multi-core parallelism.

## Features

- **Multiple Tokenizer Types**: WordPiece (BERT), BPE (GPT-2/GPT-3/GPT-4), Unigram (SentencePiece)
- **SIMD Acceleration**: Automatic detection and use of SSE4.2, AVX2, AVX-512 (x86_64) and NEON, SVE, SVE2 (ARM)
- **Multi-Core Parallelism**: Configurable thread count with automatic scaling
- **HuggingFace Compatible**: Load tokenizers directly from `tokenizer.json` files
- **Zero-Copy Design**: Minimal allocations for maximum throughput
- **Production Ready**: Extensively tested with regression and property-based tests

## Benchmark Results

Tested on 1GB OpenWebText dataset (215K documents) on a 16-core CPU:

### BPE (GPT-2) Throughput

| Configuration | Throughput | Speedup |
|--------------|-----------|---------|
| Scalar Single-Core | 21.8 MB/s | 1.00x |
| SIMD Single-Core | 22.0 MB/s | 1.01x |
| Scalar Multi-Core (16 threads) | 236.1 MB/s | 10.81x |
| **SIMD Multi-Core (16 threads)** | **239.4 MB/s** | **10.96x** |

### WordPiece (BERT) Throughput

| Configuration | Throughput | Speedup |
|--------------|-----------|---------|
| Scalar Single-Core | 8.1 MB/s | 1.00x |
| SIMD Single-Core | 29.6 MB/s | 3.63x |
| Scalar Multi-Core (16 threads) | 33.4 MB/s | 4.09x |
| **SIMD Multi-Core (16 threads)** | **266.0 MB/s** | **32.64x** |

### Key Insights

- **WordPiece**: SIMD provides 3.6x speedup on single-core, 32.6x total with multi-core
- **BPE**: Multi-core parallelism is the primary speedup (10.9x); SIMD helps with pre-tokenization
- **Parallel Efficiency**: 68% for BPE, 56% for WordPiece

## Supported Tokenizers

### WordPiece (BERT-style)
- Used by: BERT, DistilBERT, ELECTRA, RoBERTa (with modifications)
- Algorithm: Greedy longest-match with subword prefix (`##`)
- SIMD: Accelerated trie lookup and whitespace detection

### BPE (Byte Pair Encoding)
- Used by: GPT-2, GPT-3, GPT-4, LLaMA, Mistral, Claude
- Algorithm: Merge-based encoding with priority ordering
- Optimizations: Thread-local caching, direct word lookup, Aho-Corasick automaton

### Unigram (SentencePiece)
- Used by: T5, ALBERT, XLNet, mBART
- Algorithm: Probabilistic subword segmentation
- Implementation: Viterbi decoding for optimal segmentation

## Supported ISAs (Instruction Set Architectures)

### x86_64
| ISA | Vector Width | Features |
|-----|--------------|----------|
| SSE4.2 | 128-bit | Baseline SIMD |
| AVX2 | 256-bit | Recommended for most CPUs |
| AVX-512 | 512-bit | Best for server CPUs (Xeon, EPYC) |

### ARM (aarch64)
| ISA | Vector Width | Features |
|-----|--------------|----------|
| NEON | 128-bit | Standard on all ARM64 |
| SVE | Scalable (128-2048 bit) | Arm v8.2+ |
| SVE2 | Scalable | Arm v9+ with enhanced ops |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
budtiktok-core = { path = "crates/budtiktok-core" }
```

### Optional Features

```toml
[dependencies]
budtiktok-core = { path = "crates/budtiktok-core", features = ["jemalloc"] }
```

| Feature | Description |
|---------|-------------|
| `jemalloc` | Use jemalloc allocator (recommended for Linux servers) |
| `mimalloc` | Use mimalloc allocator (good cross-platform performance) |
| `all-allocators` | Enable all allocator options for benchmarking |

## Quick Start

### Basic Usage

```rust
use budtiktok_core::{load_tokenizer, Tokenizer};

// Load a tokenizer from HuggingFace format
let tokenizer = load_tokenizer("path/to/tokenizer.json")?;

// Encode text
let encoding = tokenizer.encode("Hello, world!", true)?;
println!("Token IDs: {:?}", encoding.ids);
println!("Tokens: {:?}", encoding.tokens);

// Decode back to text
let text = tokenizer.decode(&encoding.ids, true)?;
println!("Decoded: {}", text);
```

### With Runtime Configuration

```rust
use budtiktok_core::{
    RuntimeConfig, IsaSelection, ParallelismConfig,
    load_tokenizer, apply_config,
};

// Auto-detect everything (recommended)
let config = RuntimeConfig::auto();
apply_config(&config);

// Or customize settings
let config = RuntimeConfig::new()
    .with_isa(IsaSelection::Avx2)      // Force AVX2
    .with_threads(8)                    // Use 8 threads
    .with_cache(true)                   // Enable word cache
    .with_cache_size(50000);            // Cache 50K words

apply_config(&config);
println!("{}", config);

let tokenizer = load_tokenizer("path/to/tokenizer.json")?;
```

### ISA Selection

```rust
use budtiktok_core::{IsaSelection, RuntimeConfig};

// Auto-detect best ISA (recommended)
let config = RuntimeConfig::new().with_isa(IsaSelection::Auto);

// Force specific ISA
let config = RuntimeConfig::new().with_isa(IsaSelection::Avx2);
let config = RuntimeConfig::new().with_isa(IsaSelection::Avx512);
let config = RuntimeConfig::new().with_isa(IsaSelection::Neon);

// Disable SIMD entirely (scalar mode)
let config = RuntimeConfig::scalar();

// Check what's available
println!("Best ISA: {}", IsaSelection::best_available());
println!("AVX2 available: {}", IsaSelection::Avx2.is_available());
```

### Thread Configuration

```rust
use budtiktok_core::{ParallelismConfig, RuntimeConfig};

// Single-threaded (for embedding in other apps)
let config = RuntimeConfig::single_threaded();

// Specific thread count
let config = RuntimeConfig::new().with_threads(4);

// Use all CPU cores
let config = RuntimeConfig::new()
    .with_parallelism(ParallelismConfig::AllCores);

// Auto-detect optimal thread count
let config = RuntimeConfig::new()
    .with_parallelism(ParallelismConfig::Auto);
```

### Batch Encoding (Multi-Core)

```rust
use budtiktok_core::{load_tokenizer, Tokenizer, RuntimeConfig, apply_config};
use rayon::prelude::*;

// Configure for maximum throughput
let config = RuntimeConfig::server();
apply_config(&config);

let tokenizer = load_tokenizer("path/to/tokenizer.json")?;

let texts = vec![
    "First document",
    "Second document",
    "Third document",
];

// Parallel encoding
let encodings: Vec<_> = texts
    .par_iter()
    .map(|text| tokenizer.encode(text, true).unwrap())
    .collect();
```

### System Information

```rust
use budtiktok_core::system_info;

let info = system_info();
println!("{}", info);

// Output:
// System Information
//   Architecture: x86_64
//   OS: linux
//   Physical Cores: 8
//   Logical Cores: 16
//   Best ISA: AVX2
//   SIMD Capabilities:
//     SSE4.2: true
//     AVX2: true
//     AVX-512F: false
//     AVX-512BW: false
```

## API Reference

### RuntimeConfig

Configuration for runtime behavior.

```rust
// Preset configurations
RuntimeConfig::auto()           // Auto-detect everything (recommended)
RuntimeConfig::single_threaded() // Single-threaded mode
RuntimeConfig::scalar()         // No SIMD, pure scalar
RuntimeConfig::server()         // High-throughput server mode

// Builder methods
config.with_isa(isa)                    // Set ISA selection
config.with_threads(n)                  // Set thread count
config.with_parallelism(parallelism)    // Set parallelism config
config.with_cache(enabled)              // Enable/disable word cache
config.with_cache_size(size)            // Set cache size
config.with_batch_size(size)            // Set batch size

// Inspection methods
config.effective_isa()          // Get resolved ISA
config.effective_threads()      // Get resolved thread count
config.simd_enabled()           // Check if SIMD is active
config.validate()               // Get configuration warnings
```

### IsaSelection

ISA selection enumeration.

```rust
IsaSelection::Auto      // Auto-detect best
IsaSelection::Scalar    // No SIMD
IsaSelection::Sse42     // SSE 4.2
IsaSelection::Avx2      // AVX2
IsaSelection::Avx512    // AVX-512
IsaSelection::Neon      // ARM NEON
IsaSelection::Sve       // ARM SVE
IsaSelection::Sve2      // ARM SVE2

// Methods
isa.is_available()      // Check availability
isa.effective()         // Resolve Auto to actual
isa.name()              // Human-readable name
IsaSelection::from_str("avx2")  // Parse from string
IsaSelection::best_available()  // Get best for platform
```

### ParallelismConfig

Thread configuration.

```rust
ParallelismConfig::SingleThreaded   // 1 thread
ParallelismConfig::Threads(n)       // n threads
ParallelismConfig::Auto             // Auto-detect
ParallelismConfig::AllCores         // Use all cores

// Methods
parallelism.effective_threads()     // Get actual thread count
parallelism.is_parallel()           // Check if > 1 thread
ParallelismConfig::from_str("4")    // Parse from string
```

## Optimizations

### BPE Optimizations
- **Thread-Local LRU Cache**: Common words are cached per-thread
- **Direct Trie Lookup**: Complete words are matched directly without merge iterations
- **Aho-Corasick Automaton**: Fast multi-pattern matching for merges
- **O(n) Linear Algorithm**: Efficient merge application

### WordPiece Optimizations
- **SIMD Trie Lookup**: Vectorized prefix matching
- **Batch Classification**: SIMD character classification
- **Prefetching**: Cache-friendly memory access patterns
- **Double-Array Trie**: Compact, cache-efficient data structure

### Memory Optimizations
- **Arena Allocator**: Bulk allocation for temporary data
- **String Interning**: Deduplicated token storage
- **Pool Allocator**: Reusable encoding buffers
- **NUMA-Aware**: Optimized for multi-socket systems

## Running Benchmarks

```bash
# Run the full benchmark suite
cargo run --release --example simd_vs_scalar_benchmark

# Run criterion benchmarks
cargo bench --bench scalar_tokenization

# Run regression tests
cargo run --release --example full_regression_test
```

## Testing

```bash
# Run all tests
cargo test --release -p budtiktok-core

# Run specific test modules
cargo test --lib -p budtiktok-core bpe --release
cargo test --lib -p budtiktok-core wordpiece --release
cargo test --lib -p budtiktok-core runtime --release

# Run regression tests
cargo test --test regression -p budtiktok-core --release
```

## Project Structure

```
budtiktok/
├── crates/
│   └── budtiktok-core/          # Core library
│       ├── src/
│       │   ├── lib.rs           # Library entry point
│       │   ├── runtime.rs       # Runtime configuration
│       │   ├── config.rs        # HuggingFace config parser
│       │   ├── tokenizer.rs     # Tokenizer trait
│       │   ├── wordpiece.rs     # Scalar WordPiece
│       │   ├── wordpiece_hyper.rs # SIMD WordPiece
│       │   ├── bpe_linear.rs    # Production BPE
│       │   ├── bpe_fast.rs      # BPE utilities
│       │   ├── unigram.rs       # Unigram tokenizer
│       │   ├── simd_backends.rs # SIMD implementations
│       │   └── ...
│       ├── benches/             # Criterion benchmarks
│       ├── examples/            # Example programs
│       └── tests/               # Integration tests
├── benchmark_data/              # Test data (GPT-2 vocab, etc.)
└── docs/                        # Documentation
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BUDTIKTOK_ISA` | Force ISA selection (auto, scalar, avx2, etc.) | auto |
| `BUDTIKTOK_THREADS` | Force thread count | auto |
| `RAYON_NUM_THREADS` | Rayon thread pool size | all cores |

## Comparison with Other Libraries

| Feature | BudTikTok | HuggingFace Tokenizers | tiktoken |
|---------|-----------|------------------------|----------|
| Language | Rust | Rust | Rust/Python |
| WordPiece | Yes | Yes | No |
| BPE | Yes | Yes | Yes |
| Unigram | Yes | Yes | No |
| SIMD | Auto-detect | Limited | No |
| Multi-Core | Configurable | Yes | No |
| ISA Selection | Yes | No | No |

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) - Reference implementation
- [tiktoken](https://github.com/openai/tiktoken) - BPE inspiration
- [StringZilla](https://github.com/ashvardanian/StringZilla) - SIMD techniques
- [rust-bpe](https://github.com/github/rust-gems) - O(n) BPE algorithm
