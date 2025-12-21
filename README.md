# BudTikTok: High-Performance HF-Compatible Tokenization

**BudTikTok** is a next-generation tokenization library designed to bridge the gap between high-performance systems and the HuggingFace ecosystem. It offers a **5-10x performance advantage** over standard HuggingFace tokenizers while maintaining full API and format compatibility.

## 🚀 Key Features

- **Extreme Performance**: Built with Rust, utilizing SIMD (AVX-512), intelligent caching, and lazy evaluation to achieve state-of-the-art speeds.
- **Full HuggingFace Compatibility**:
    - Drop-in replacement for HF tokenizers.
    - Uses standard `tokenizer.json` format.
    - Compatible API surface for seamless integration.
- **Hybrid Architecture**: unique `TokenizerPipeline` design that wraps ultra-fast core models (WordPiece, BPE, Unigram) with flexible, composable pipeline stages.
- **Comprehensive Model Support**:
    - **WordPiece** (BERT, DistilBERT, Electra)
    - **BPE** (GPT-2, RoBERTa)
    - **Unigram** (Albert, T5)
    - **WordLevel**
- **Production Ready**: Thread-safe, robust error handling, and designed for high-concurrency environments.

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

### Dynamic Token Management

BudTikTok supports dynamic vocabulary modification, fully compatible with HF's `AddedVocabulary`.

```rust
use budtiktok::{TokenizerPipeline, AddedToken};

let mut tokenizer = TokenizerPipeline::from_file("tokenizer.json")?;

// Add a special token
tokenizer.add_special_tokens(&[
    AddedToken::from("<custom_token>", true),
]);

// The new token is now recognized and handled correctly
let encoding = tokenizer.encode("This has a <custom_token>.", true)?;
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

## 📄 License

This project is licensed under the **Apache-2.0** license.
