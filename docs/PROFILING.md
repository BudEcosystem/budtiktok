# BudTikTok Profiling Guide

This guide covers how to profile BudTikTok to identify performance bottlenecks and optimize the codebase.

## Table of Contents

1. [Profiling Tools Overview](#profiling-tools-overview)
2. [Installation](#installation)
3. [CPU Profiling](#cpu-profiling)
4. [Memory Profiling](#memory-profiling)
5. [Cache Analysis](#cache-analysis)
6. [SIMD Verification](#simd-verification)
7. [Distributed System Profiling](#distributed-system-profiling)
8. [When to Profile](#when-to-profile)
9. [Interpreting Results](#interpreting-results)
10. [Common Bottlenecks](#common-bottlenecks)

---

## Profiling Tools Overview

| Tool | Purpose | Platform | When to Use |
|------|---------|----------|-------------|
| **samply** | CPU sampling profiler | Linux, macOS | General performance analysis |
| **flamegraph** | CPU flame graphs | Linux, macOS | Visualizing call stacks |
| **perf** | Low-level CPU profiling | Linux | Detailed CPU analysis |
| **cachegrind** | Cache simulation | Linux | Cache optimization |
| **DHAT** | Heap profiling | Linux | Finding allocation hotspots |
| **heaptrack** | Heap analysis | Linux | Memory leak detection |
| **miri** | Memory safety | Any | Detecting undefined behavior |
| **loom** | Concurrency testing | Any | Finding race conditions |

---

## Installation

### Linux

```bash
# Install system tools
sudo apt-get install linux-tools-common linux-tools-generic valgrind heaptrack

# Install Rust tools
cargo install flamegraph samply cargo-llvm-cov

# For criterion benchmarks
cargo install critcmp
```

### macOS

```bash
# Install via Homebrew
brew install flamegraph

# Install Rust tools
cargo install flamegraph samply cargo-llvm-cov

# Note: Instruments.app is also available on macOS
```

### Build Configuration

Add to your `~/.cargo/config.toml` for better profiling:

```toml
[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "force-frame-pointers=yes"]
```

Build with debug symbols:

```bash
# Release build with debug info
cargo build --profile release-with-debug

# Or set environment variable
RUSTFLAGS="-C debuginfo=2" cargo build --release
```

---

## CPU Profiling

### Using samply (Recommended)

samply is a modern sampling profiler with an excellent UI.

```bash
# Profile the benchmark binary
samply record ./target/release/budtiktok-bench --bench wordpiece

# Opens Firefox Profiler in browser automatically
```

**What to look for:**
- Functions with high "self time" are optimization targets
- Look for unexpected standard library calls (`alloc`, `clone`, `format!`)
- Check for lock contention (`parking_lot`, `Mutex::lock`)

### Using flamegraph

Generate SVG flame graphs:

```bash
# Basic flamegraph
cargo flamegraph --bin budtiktok-bench -- --bench wordpiece

# With specific features
cargo flamegraph --features simd --bin budtiktok-bench

# Output to specific file
cargo flamegraph -o profile.svg --bin budtiktok-bench
```

**Reading flamegraphs:**
- Width = time spent (wider = more time)
- Y-axis = call stack depth
- Look for wide bars in the middle (hot functions)
- Hover for exact percentages

### Using perf (Linux)

Low-level CPU profiling with hardware counters:

```bash
# Record profile
perf record -g --call-graph dwarf ./target/release/budtiktok-bench

# Generate report
perf report --hierarchy

# Show annotated source
perf annotate

# Record specific events
perf record -e cycles,instructions,cache-misses ./target/release/budtiktok-bench
```

**Useful perf events:**
- `cycles` - CPU cycles
- `instructions` - Instructions executed
- `cache-references` - Total cache accesses
- `cache-misses` - Cache misses
- `branch-misses` - Branch mispredictions

### Built-in Timing

BudTikTok includes built-in timing instrumentation:

```bash
# Enable detailed timing
BUDTIKTOK_PROFILE=1 ./target/release/budtiktok-bench

# Output includes:
# - Normalization time
# - Pre-tokenization time
# - Model tokenization time
# - Post-processing time
```

---

## Memory Profiling

### Using heaptrack (Recommended)

Best for finding allocation hotspots and memory growth:

```bash
# Record allocations
heaptrack ./target/release/budtiktok-bench

# Analyze with GUI
heaptrack_gui heaptrack.budtiktok-bench.*.gz

# Or analyze in terminal
heaptrack_print heaptrack.budtiktok-bench.*.gz
```

**What to look for:**
- Peak memory usage
- Allocation count per function
- Temporary allocations (allocate then quickly free)
- Memory growth over time (potential leaks)

### Using DHAT

Valgrind tool for detailed heap profiling:

```bash
# Run with DHAT
valgrind --tool=dhat ./target/release/budtiktok-bench

# View results in browser
# Open dhat.out.* file with DHAT viewer
```

**What to look for:**
- "max-blocks" - Peak allocation count
- "total-bytes" - Total bytes allocated
- Short-lived allocations in hot paths

### Using miri

Detect memory safety issues:

```bash
# Install miri
rustup +nightly component add miri

# Run tests under miri
cargo +nightly miri test

# Run specific test
cargo +nightly miri test test_wordpiece_basic
```

**Note:** miri is slow, run on specific tests, not full benchmarks.

---

## Cache Analysis

### Using cachegrind

Simulate CPU cache behavior:

```bash
# Run cachegrind
valgrind --tool=cachegrind ./target/release/budtiktok-bench

# Annotate results
cg_annotate --auto=yes cachegrind.out.*

# Show per-function breakdown
cg_annotate --threshold=1 cachegrind.out.*
```

**Key metrics:**
- **D1 miss rate** - L1 data cache misses (target: <5%)
- **LL miss rate** - Last-level cache misses (target: <1%)
- **Bc** - Conditional branches
- **Bcm** - Mispredicted conditional branches

### Interpreting Cache Results

```
==12345== D   refs:      100,000,000
==12345== D1  misses:      1,000,000  (1.0%)
==12345== LLd misses:        100,000  (0.1%)
```

Good cache behavior:
- D1 miss rate < 5%
- LL miss rate < 1%
- Spatial locality: access memory sequentially
- Temporal locality: reuse recently accessed data

---

## SIMD Verification

### Verify SIMD Detection

```bash
# Check CPU features
cat /proc/cpuinfo | grep -E 'avx|sse|neon' | head -5

# Check runtime detection
RUST_LOG=debug ./target/release/budtiktok 2>&1 | grep -i 'simd\|avx\|sse\|neon'
```

### Verify SIMD Code Generation

```bash
# Disassemble and look for SIMD instructions
objdump -d target/release/budtiktok-bench | grep -E 'vmov|vpcmp|vadd|vpand' | head -20

# Count SIMD instructions
objdump -d target/release/budtiktok-bench | grep -cE 'vmov|vpcmp|vadd'
```

### Benchmark SIMD vs Scalar

```bash
# Run benchmarks with different backends
BUDTIKTOK_SIMD=avx512 ./target/release/budtiktok-bench
BUDTIKTOK_SIMD=avx2 ./target/release/budtiktok-bench
BUDTIKTOK_SIMD=scalar ./target/release/budtiktok-bench
```

---

## Distributed System Profiling

### Tracing

Enable distributed tracing:

```bash
# Start with tracing enabled
RUST_LOG=budtiktok=trace ./target/release/budtiktok serve

# Or export to Jaeger
OTEL_EXPORTER_JAEGER_ENDPOINT=http://localhost:14268/api/traces ./target/release/budtiktok serve
```

### Latency Breakdown

Use built-in latency breakdown:

```bash
# Enable latency breakdown
curl http://localhost:8080/debug/latency

# Output:
# {
#   "tokenization_p50_us": 150,
#   "tokenization_p99_us": 500,
#   "ipc_transfer_p50_us": 20,
#   "queue_wait_p50_us": 10
# }
```

### Worker Utilization

Monitor worker metrics:

```bash
# Get worker stats
curl http://localhost:8080/metrics | grep worker

# Output:
# budtiktok_worker_utilization{worker="0"} 0.75
# budtiktok_worker_queue_depth{worker="0"} 50
```

---

## When to Profile

### Profile Before Optimization

Always establish a baseline before making changes:

```bash
# Save baseline
cargo bench -- --save-baseline before

# Make changes...

# Compare
cargo bench -- --baseline before
critcmp before after
```

### Profile After Major Changes

Run after:
- Adding new algorithms
- Changing data structures
- Modifying hot paths
- Updating dependencies

### Profile Before Release

Ensure no regressions:

```bash
# Full profiling suite
./scripts/profile-all.sh

# Generates:
# - Flamegraph
# - Memory analysis
# - Cache analysis
# - Benchmark comparison
```

---

## Interpreting Results

### Identifying Hotspots

A function is a hotspot if:
- >10% of total time
- Called millions of times
- High cache miss rate

### Common Patterns

**Pattern: Allocation in hot path**
```
50% - core::ptr::drop_in_place
30% - alloc::alloc::alloc
```
Solution: Use arena allocators, pre-allocate, avoid cloning

**Pattern: Hash table overhead**
```
40% - hashbrown::raw::RawTable::find
```
Solution: Use perfect hashing, cache lookups, switch to Trie

**Pattern: Lock contention**
```
30% - parking_lot::RawMutex::lock_slow
```
Solution: Reduce lock scope, use sharding, lock-free structures

**Pattern: Branch misprediction**
```
High Bcm (branch misprediction) count
```
Solution: Use branchless code, sort data for predictability

---

## Common Bottlenecks

### 1. String Allocation

**Symptom:** High allocation count, `String::from` in flamegraph

**Solution:**
```rust
// Instead of
let s = format!("{}{}", prefix, word);

// Use
let mut s = String::with_capacity(prefix.len() + word.len());
s.push_str(prefix);
s.push_str(word);

// Or use Cow<str> to avoid allocation when possible
```

### 2. Hash Table Lookups

**Symptom:** `HashMap::get` dominating profile

**Solution:**
- Use `AHashMap` instead of `HashMap`
- Pre-compute hashes
- Use Trie for prefix matching
- Cache frequent lookups

### 3. UTF-8 Iteration

**Symptom:** `str::chars` or `char_indices` taking significant time

**Solution:**
- Process bytes when possible
- Use SIMD for ASCII detection
- Batch character processing

### 4. Cache Misses

**Symptom:** High L1/LL miss rate in cachegrind

**Solution:**
- Improve data locality (Structure of Arrays)
- Prefetch data
- Reduce pointer chasing
- Use cache-oblivious algorithms

### 5. Memory Bandwidth

**Symptom:** Throughput plateaus despite low CPU usage

**Solution:**
- Reduce memory footprint
- Use smaller data types
- Compress data
- Process in-place

---

## Profiling Checklist

Before submitting optimizations:

- [ ] Baseline benchmark recorded
- [ ] Flamegraph shows improvement in target function
- [ ] No memory leaks (heaptrack)
- [ ] Cache miss rate not increased (cachegrind)
- [ ] SIMD still being used (objdump)
- [ ] All tests pass
- [ ] Benchmark shows expected improvement
- [ ] No regression in other benchmarks

---

## Scripts

### Full Profiling Script

Create `scripts/profile-all.sh`:

```bash
#!/bin/bash
set -e

OUTPUT_DIR="profiling_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Building release with debug symbols..."
RUSTFLAGS="-C debuginfo=2" cargo build --release

echo "Running benchmarks..."
cargo bench -- --noplot --save-baseline current > "$OUTPUT_DIR/benchmark.txt"

echo "Generating flamegraph..."
cargo flamegraph -o "$OUTPUT_DIR/flamegraph.svg" --bin budtiktok-bench -- --bench wordpiece 2>/dev/null

echo "Running heaptrack..."
heaptrack ./target/release/budtiktok-bench > /dev/null 2>&1
mv heaptrack.*.gz "$OUTPUT_DIR/"

echo "Running cachegrind..."
valgrind --tool=cachegrind ./target/release/budtiktok-bench > /dev/null 2>&1
mv cachegrind.out.* "$OUTPUT_DIR/"

echo "Done! Results in $OUTPUT_DIR"
```

---

*Last Updated: 2025-12-17*
