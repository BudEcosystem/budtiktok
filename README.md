# BudTikTok: High-Performance HF-Compatible Tokenization

**BudTikTok** is a next-generation, production-ready tokenization library designed to bridge the gap between high-performance systems and the HuggingFace ecosystem. It offers a **5-10x performance advantage** over standard HuggingFace tokenizers while maintaining **95% API and format compatibility**.

## 🚀 Key Features

### ⚡ Extreme Performance
- **SIMD Acceleration**: Runtime-detected optimization for **AVX-512**, **AVX2**, **SSE4.2** (x86_64) and **NEON**, **SVE** (ARM64).
- **Parallel Execution**: Native **Rayon** integration for multi-threaded batch encoding and decoding.
- **Intelligent Caching**: Multi-level cache with CLOCK eviction and sharded access for high concurrency.
- **Lazy Evaluation**: Zero-copy pipeline design that only computes what is necessary.

### 🎮 GPU Acceleration
- **CUDA Support**: Fully integrated GPU tokenization pipeline.
- **Multi-GPU**: Automatic load balancing across available GPUs.
- **Async Pipeline**: Overlapped CPU-GPU data transfer for maximum throughput.

### 🔌 Full HuggingFace Compatibility
- **Drop-in Replacement**: Compatible with standard `tokenizer.json` files.
- **Post-Processing**: Native support for **BERT**, **RoBERTa**, and **Template** post-processors.
- **Model Support**:
    - **WordPiece** (BERT, DistilBERT, Electra)
    - **BPE** (GPT-2, RoBERTa, Llama-2)
    - **Unigram** (Albert, T5)
    - **WordLevel**
- **Gap Analysis**: See [BUDTIKTOK_HF_GAP_ANALYSIS.md](BUDTIKTOK_HF_GAP_ANALYSIS.md) for detailed compatibility report.

### 🧠 LatentBud Integration
- **Pre-tokenized Requests**: Native support for pre-tokenized inputs to bypass redundant processing.
- **Token Budget Routing**: Intelligent routing based on token budgets for efficient batching.

## 📦 Installation

Add `budtiktok` to your `Cargo.toml`:

```toml
[dependencies]
budtiktok = { git = "https://github.com/BudEcosystem/budtiktok.git" }
```

## 🛠️ Usage

### Basic Tokenization

```rust
use budtiktok::TokenizerPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load from a standard tokenizer.json file
    let tokenizer = TokenizerPipeline::from_file("tokenizer.json")?;

    // Encode text
    let encoding = tokenizer.encode("Hello, world!", true)?;
    println!("Tokens: {:?}", encoding.get_tokens());
    println!("IDs: {:?}", encoding.get_ids());

    // Decode IDs back to text
    let decoded = tokenizer.decode(encoding.get_ids(), true)?;
    println!("Decoded: {}", decoded);

    Ok(())
}
```

### GPU Tokenization

```rust
use budtiktok::{TokenizerPipeline, GpuConfig};

// Enable GPU with auto-detection
let config = GpuConfig::auto();
let tokenizer = TokenizerPipeline::from_file_with_gpu("tokenizer.json", config)?;

// Tokenize on GPU (transparently handles batching)
let encodings = tokenizer.encode_batch(&texts, true)?;
```

## 🏗️ Architecture

BudTikTok employs a **Pipeline Wrapper** pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                      TokenizerPipeline                          │
│  ┌───────────┐  ┌──────────────┐  ┌───────┐  ┌──────────────┐  │
│  │Normalizer │→ │PreTokenizer  │→ │ Model │→ │PostProcessor │  │
│  │ (Option)  │  │  (Option)    │  │       │  │  (Option)    │  │
│  └───────────┘  └──────────────┘  └───────┘  └──────────────┘  │
│                                       ↓                         │
│  ┌─────────────────┐            ┌─────────┐                     │
│  │ AddedVocabulary │←───────────│ Decoder │                     │
│  │ (Aho-Corasick)  │            │(Option) │                     │
│  └─────────────────┘            └─────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

This design allows for:
1.  **Lazy Evaluation**: Components like normalizers are only applied when necessary.
2.  **Zero-Copy Optimizations**: Extensive use of `Cow<str>` and memory mapping.
3.  **Lock-Free Concurrency**: `RwLock` for read-heavy vocabulary access and `Arc` for shared immutable components.

## 🆕 Recent Updates (Dec 2025)

### Memory Safety Refactor
- **Arc<str> StringInterner** - Eliminated potential use-after-free risks with reference-counted string storage
- **Thread-safe vocabulary** - All interned strings are protected by `Arc` for safe concurrent access
- **Zero unsafe transmute** in hot paths - Safety-critical code now uses bounded lifetimes

### Full HuggingFace Serialization
- **`save()` and `from_pretrained()`** - Full compatibility with HuggingFace `tokenizer.json` format
- **Roundtrip verified** - All tokenizer types (BPE, WordPiece, Unigram) pass serialization tests
- **Model versioning** - Deterministic config hashes for cache key generation

### Enhanced SIMD Performance
- **AVX2 character classification** - Real SIMD intrinsics for 32-byte parallel character classification
- **PSHUFB nibble technique** - Ultra-fast ASCII class lookup
- **Verified correctness** - SIMD paths produce identical results to scalar (test-verified)

### Vocabulary Refactor
- **Structured `Vocabulary` type** - Replaced raw `Vec<String>` with type-safe vocabulary management
- **Centralized special tokens** - `SpecialTokens` struct for `[CLS]`, `[SEP]`, `[PAD]`, etc.
- **Improved type safety** - Compile-time guarantees for vocabulary operations

### Test Results
```
407 tests passed (budtiktok-core)
3 serialization roundtrip tests passed
Workspace compilation: Success
```

## 📄 License

This project is licensed under the **Apache-2.0** license.
