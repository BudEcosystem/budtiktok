# Experimental Implementations (Archived)

This folder contains experimental implementations that were not successful enough for production use.

## bpe_simd.rs

SIMD-accelerated BPE tokenizer experiments. Archived because:

### Results Summary

| Encoder | Throughput | vs Standard | Accuracy |
|---------|-----------|-------------|----------|
| Standard BPE (scalar) | 26.4 MB/s | 1.00x | 100% |
| SIMD BPE Mode | 24.4 MB/s | **0.92x (slower)** | 100% |
| SIMD GPT-2 Mode | 23.3 MB/s | 0.88x (slower) | 0.9% |
| SIMD Greedy Raw | 51.1 MB/s | 1.93x (faster) | 0.2% |

### Why It Failed

1. **BPE's merge algorithm is inherently sequential**: You must find the single best merge, apply it, then repeat. SIMD can parallelize lookups but doesn't change the O(n^2) complexity.

2. **The standard encoder has optimizations that matter more**:
   - Thread-local LRU cache for common words
   - Direct trie match for complete words (skips merge entirely)
   - These provide more speedup than SIMD parallelism

3. **Greedy mode is fast but inaccurate**: The 1.93x speedup comes at the cost of different tokenization (only 0.2% exact match).

### Key Learnings

- For BPE, algorithmic optimizations (caching, direct word lookup) beat SIMD parallelism
- The "Train It and Forget It" paper's finding that greedy is acceptable for inference may not apply when exact compatibility is required
- SIMD shines in pre-tokenization and normalization, not in the merge phase

### If Revisiting

To make SIMD BPE actually faster while maintaining compatibility:
1. Add thread-local word cache (like OptimizedBpeEncoder)
2. Add direct trie match for complete words
3. Consider BlockBPE-style parallel merge passes for very long sequences
