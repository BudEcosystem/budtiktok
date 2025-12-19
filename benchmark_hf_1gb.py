#!/usr/bin/env python3
"""
HuggingFace Tokenizers 1GB Benchmark
Tests single-core and multi-core performance
"""

import json
import time
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

# Use HuggingFace tokenizers (Rust-backed)
from tokenizers import Tokenizer
from transformers import AutoTokenizer

def load_dataset(path: str, max_docs: int = None):
    """Load JSONL dataset"""
    documents = []
    total_bytes = 0

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            data = json.loads(line)
            text = data.get('text', '')
            documents.append(text)
            total_bytes += len(text.encode('utf-8'))

    return documents, total_bytes

def tokenize_single(args):
    """Tokenize a single document (for multiprocessing)"""
    text, tokenizer_path = args
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return len(tokenizer.encode(text))

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        HuggingFace Tokenizers 1GB Benchmark                      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    workspace = "/home/bud/Desktop/latentbud/budtiktok"
    data_path = f"{workspace}/benchmark_data/openwebtext_1gb.jsonl"

    # Load dataset
    print("Loading full 1GB dataset...")
    documents, total_bytes = load_dataset(data_path)
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_mb / 1024
    print(f"  Documents: {len(documents)}")
    print(f"  Total size: {total_gb:.2f} GB ({total_mb:.2f} MB)")

    # Load tokenizer
    print("\nInitializing HuggingFace tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    print(f"  Vocab size: {len(tokenizer)}")

    # Warm up
    print("\nWarming up (1000 docs)...")
    for doc in documents[:1000]:
        _ = tokenizer.encode(doc)

    # =========================================================================
    # EXPERIMENT 1: Single-Core Benchmark
    # =========================================================================
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║              EXPERIMENT 1: Single-Core Benchmark                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    print(f"Running HuggingFace (single-core) on {total_gb:.2f} GB...")
    start = time.perf_counter()
    total_tokens_single = 0
    for doc in documents:
        total_tokens_single += len(tokenizer.encode(doc))
    single_time = time.perf_counter() - start
    single_throughput = total_mb / single_time

    print(f"  Time: {single_time:.2f}s")
    print(f"  Throughput: {single_throughput:.2f} MB/s")
    print(f"  Tokens: {total_tokens_single}")
    print(f"  Tokens/sec: {total_tokens_single / single_time:.0f}")

    # =========================================================================
    # EXPERIMENT 2: Multi-Core Benchmark (using batch encoding)
    # =========================================================================
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         EXPERIMENT 2: Multi-Core Benchmark (16 cores)            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    num_cores = 16
    print(f"Running HuggingFace (batch encoding with {num_cores} threads)...")

    # HuggingFace tokenizers library supports internal parallelism
    # Use batch encoding which leverages Rust parallelism internally
    batch_size = 10000

    start = time.perf_counter()
    total_tokens_multi = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        # batch_encode_plus uses internal parallelism
        results = tokenizer(batch, add_special_tokens=False, return_length=True)
        total_tokens_multi += sum(results['length'])

    multi_time = time.perf_counter() - start
    multi_throughput = total_mb / multi_time

    print(f"  Time: {multi_time:.2f}s")
    print(f"  Throughput: {multi_throughput:.2f} MB/s")
    print(f"  Tokens: {total_tokens_multi}")
    print(f"  Tokens/sec: {total_tokens_multi / multi_time:.0f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                   HUGGINGFACE RESULTS                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    print(f"Dataset: {total_gb:.2f} GB ({len(documents)} documents)")
    print()
    print("┌────────────────┬─────────────────┬─────────────────┐")
    print("│    Mode        │   Throughput    │   Tokens/sec    │")
    print("├────────────────┼─────────────────┼─────────────────┤")
    print(f"│ Single-Core    │ {single_throughput:>8.2f} MB/s   │ {total_tokens_single / single_time:>12.0f}    │")
    print(f"│ Multi-Core(16) │ {multi_throughput:>8.2f} MB/s   │ {total_tokens_multi / multi_time:>12.0f}    │")
    print("└────────────────┴─────────────────┴─────────────────┘")
    print()

    # Output for comparison
    print("\n[RESULTS_FOR_COMPARISON]")
    print(f"HF_SINGLE_THROUGHPUT={single_throughput:.2f}")
    print(f"HF_MULTI_THROUGHPUT={multi_throughput:.2f}")
    print(f"HF_SINGLE_TOKENS={total_tokens_single}")
    print(f"HF_MULTI_TOKENS={total_tokens_multi}")

if __name__ == "__main__":
    main()
