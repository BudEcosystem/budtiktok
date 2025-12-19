# BudTikTok Core Fuzz Tests

This directory contains fuzz tests for BudTikTok Core using `cargo-fuzz` and `libFuzzer`.

## Prerequisites

1. Install cargo-fuzz (requires nightly Rust):
```bash
cargo install cargo-fuzz
```

2. Ensure you have nightly Rust installed:
```bash
rustup install nightly
```

## Available Fuzz Targets

| Target | Description | Task ID |
|--------|-------------|---------|
| `fuzz_tokenize` | Fuzzes tokenization pipeline with arbitrary UTF-8 strings | 9.5.1 |
| `fuzz_json_parser` | Fuzzes JSON config/vocabulary parser | 9.5.2 |
| `fuzz_ipc` | Fuzzes IPC binary deserialization | 9.5.3 |
| `fuzz_unicode` | Fuzzes Unicode normalization and classification | - |

## Running Fuzz Tests

### Run a specific fuzz target:
```bash
cd crates/budtiktok-core
cargo +nightly fuzz run fuzz_tokenize
```

### Run with a time limit (e.g., 60 seconds):
```bash
cargo +nightly fuzz run fuzz_tokenize -- -max_total_time=60
```

### Run with specific number of jobs (parallel):
```bash
cargo +nightly fuzz run fuzz_tokenize -- -jobs=4
```

### List all available fuzz targets:
```bash
cargo +nightly fuzz list
```

### View coverage report:
```bash
cargo +nightly fuzz coverage fuzz_tokenize
```

## Corpus Management

Each fuzz target has a corpus directory in `fuzz/corpus/<target_name>/`.
The fuzzer will use existing corpus files as seeds and save new interesting inputs.

### Add seed inputs:
```bash
mkdir -p fuzz/corpus/fuzz_tokenize
echo "Hello, World!" > fuzz/corpus/fuzz_tokenize/hello
echo "日本語テスト" > fuzz/corpus/fuzz_tokenize/japanese
```

## Crash Reproduction

If a crash is found, it will be saved in `fuzz/artifacts/<target_name>/`.
To reproduce:
```bash
cargo +nightly fuzz run fuzz_tokenize fuzz/artifacts/fuzz_tokenize/crash-<hash>
```

## Writing New Fuzz Targets

1. Create a new file in `fuzz/fuzz_targets/`
2. Add it to `fuzz/Cargo.toml` as a `[[bin]]` entry
3. Use the `fuzz_target!` macro from `libfuzzer-sys`

Example:
```rust
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Your fuzzing code here
    // Any panic will be caught and reported as a bug
});
```

## Best Practices

1. **Limit input size**: Prevent memory exhaustion with early returns for large inputs
2. **Handle errors gracefully**: Operations should not panic on any input
3. **Focus on parsing/deserialization**: These are common sources of bugs
4. **Use structured fuzzing**: The `arbitrary` crate can generate structured inputs
