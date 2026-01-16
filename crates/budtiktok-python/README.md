# BudTikTok Python Bindings

Ultra-fast HuggingFace-compatible tokenizer with SIMD and multi-core support.

## Features

- **4-20x faster** than HuggingFace tokenizers
- **Drop-in replacement** for HuggingFace tokenizers API
- **SIMD acceleration** (AVX2/AVX-512/NEON)
- **Multi-core parallelism** via Rayon (work-stealing thread pool)
- **GIL release** during tokenization for true Python parallelism
- **Zero-copy** where possible

## Installation

```bash
pip install budtiktok
```

Or build from source:

```bash
cd crates/budtiktok-python
maturin develop --release
```

## Usage

### Basic Usage

```python
from budtiktok import Tokenizer

# Load from tokenizer.json
tokenizer = Tokenizer.from_file("path/to/tokenizer.json")

# Single encoding
encoding = tokenizer.encode("Hello, world!", add_special_tokens=True)
print(encoding.ids)  # [101, 7592, 117, 2088, 106, 102]

# Batch encoding (parallel)
encodings = tokenizer.encode_batch(["Hello", "World"], add_special_tokens=True)
for enc in encodings:
    print(enc.ids)
```

### HuggingFace-Compatible Interface

```python
from budtiktok import Tokenizer

tokenizer = Tokenizer.from_pretrained("path/to/model")

# Just like HuggingFace tokenizers
result = tokenizer(
    ["Hello, world!", "How are you?"],
    max_length=512,
    padding="longest",
    truncation=True,
    return_tensors="np",  # or "pt" for PyTorch
)

print(result["input_ids"].shape)       # (2, max_len)
print(result["attention_mask"].shape)  # (2, max_len)
```

### Token Length Estimation (for Token-Budget Batching)

```python
# Fast path for getting just token lengths
lengths = tokenizer.get_token_lengths(texts, add_special_tokens=True)
```

### Configuration Info

```python
import budtiktok

config = budtiktok.get_config()
print(f"ISA: {config['best_isa']}")
print(f"Physical cores: {config['physical_cores']}")
print(f"SIMD pretokenizer: {config['use_simd_pretokenizer']}")
```

## Performance

Benchmarks on Intel i9-13900K with BERT tokenizer:

| Batch Size | HuggingFace | BudTikTok | Speedup |
|------------|-------------|-----------|---------|
| 1          | 100 µs      | 5 µs      | 20x     |
| 32         | 2000 µs     | 200 µs    | 10x     |
| 1024       | 40000 µs    | 4000 µs   | 10x     |

## Integration with LatentBud

BudTikTok is designed for seamless integration with LatentBud:

```python
from infinity_emb.inference.optimizations.budtiktok_tokenizer import (
    create_budtiktok_tokenizer,
    BUDTIKTOK_AVAILABLE,
)

# Automatically uses BudTikTok if available, falls back to HF
tokenizer = create_budtiktok_tokenizer(model_path, use_budtiktok=True)
```

## Development

### Building from Source

**Prerequisites:**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin
```

**Development build:**
```bash
cd crates/budtiktok-python
maturin develop --release
```

**Build wheel:**
```bash
maturin build --release
# Output: target/wheels/budtiktok-*.whl
```

### Installation Options

1. **From PyPI (recommended):**
   ```bash
   pip install budtiktok
   ```

2. **From GitHub:**
   ```bash
   pip install git+https://github.com/BudEcosystem/budtiktok.git#subdirectory=crates/budtiktok-python
   ```

3. **From local source:**
   ```bash
   git clone https://github.com/BudEcosystem/budtiktok.git
   cd budtiktok/crates/budtiktok-python
   pip install maturin
   maturin develop --release
   ```

### Platform Support

| Platform | Architecture | Status |
|----------|-------------|---------|
| Linux | x86_64 | ✅ Fully supported |
| Linux | aarch64 | ✅ Fully supported |
| macOS | x86_64 (Intel) | ✅ Fully supported |
| macOS | aarch64 (Apple Silicon) | ✅ Fully supported |
| Windows | x64 | ✅ Fully supported |

**Minimum requirements:**
- Python 3.8+
- glibc 2.28+ (Linux)
- macOS 10.12+ (Sierra)
- Windows 10+

### Running Tests

```bash
# Run Python tests (if available)
pytest crates/budtiktok-python/tests -v

# Run Rust tests
cargo test -p budtiktok-python

# Quick smoke test
python -c "from budtiktok import Tokenizer, get_config; print(get_config())"
```

## Contributing

Contributions are welcome! Please see the main repository for contribution guidelines.

For releases and CI/CD documentation, see [RELEASE.md](../../.github/RELEASE.md).

## License

Apache-2.0
