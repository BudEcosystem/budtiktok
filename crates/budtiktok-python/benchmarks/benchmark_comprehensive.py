#!/usr/bin/env python3
"""
Comprehensive BudTikTok vs HuggingFace Tokenizers Benchmark

Measures:
- Performance across different batch sizes
- Performance across different input lengths
- Performance with varying concurrency
- Accuracy validation against HuggingFace

Usage:
    python benchmark_comprehensive.py [--model-path PATH] [--quick]
"""

import argparse
import gc
import json
import os
import random
import statistics
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# Try to import both tokenizers
try:
    import budtiktok
    BUDTIKTOK_AVAILABLE = True
except ImportError:
    BUDTIKTOK_AVAILABLE = False
    print("Warning: BudTikTok not available, skipping BudTikTok benchmarks")

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: HuggingFace transformers not available, skipping HF benchmarks")


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    batch_size: int
    input_length: int
    num_threads: int
    total_texts: int
    total_tokens: int
    time_seconds: float
    tokens_per_second: float
    texts_per_second: float
    time_per_text_us: float


def generate_texts(count: int, min_length: int, max_length: int, seed: int = 42) -> List[str]:
    """Generate random texts of varying lengths."""
    random.seed(seed)
    texts = []

    # Common English words for more realistic text
    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
        "machine", "learning", "neural", "network", "embedding", "vector", "transformer",
        "attention", "model", "training", "inference", "optimization", "batch", "token",
        "sentence", "document", "semantic", "similarity", "search", "retrieval", "encoder"
    ]

    for _ in range(count):
        length = random.randint(min_length, max_length)
        text_words = [random.choice(words) for _ in range(length)]
        texts.append(" ".join(text_words))

    return texts


def benchmark_budtiktok(
    tokenizer,
    texts: List[str],
    batch_size: int,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
) -> BenchmarkResult:
    """Benchmark BudTikTok tokenizer."""
    total_tokens = 0

    # Warmup
    for _ in range(warmup_runs):
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            _ = tokenizer.encode_batch(batch, add_special_tokens=True)

    gc.collect()

    # Benchmark
    times = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encodings = tokenizer.encode_batch(batch, add_special_tokens=True)
            if total_tokens == 0:  # Only count on first run
                total_tokens += sum(len(e) for e in encodings)
        end = time.perf_counter()
        times.append(end - start)

    total_tokens = total_tokens or sum(len(tokenizer.encode(t, add_special_tokens=True)) for t in texts)

    avg_time = statistics.mean(times)
    return BenchmarkResult(
        name="BudTikTok",
        batch_size=batch_size,
        input_length=sum(len(t.split()) for t in texts) // len(texts),
        num_threads=1,
        total_texts=len(texts),
        total_tokens=total_tokens,
        time_seconds=avg_time,
        tokens_per_second=total_tokens / avg_time,
        texts_per_second=len(texts) / avg_time,
        time_per_text_us=avg_time * 1_000_000 / len(texts),
    )


def benchmark_huggingface(
    tokenizer,
    texts: List[str],
    batch_size: int,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
) -> BenchmarkResult:
    """Benchmark HuggingFace tokenizer."""
    total_tokens = 0

    # Warmup
    for _ in range(warmup_runs):
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            _ = tokenizer(batch, add_special_tokens=True, padding=False, truncation=False)

    gc.collect()

    # Benchmark
    times = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = tokenizer(batch, add_special_tokens=True, padding=False, truncation=False)
            if total_tokens == 0:
                total_tokens += sum(len(ids) for ids in result["input_ids"])
        end = time.perf_counter()
        times.append(end - start)

    total_tokens = total_tokens or sum(len(ids) for ids in tokenizer(texts)["input_ids"])

    avg_time = statistics.mean(times)
    return BenchmarkResult(
        name="HuggingFace",
        batch_size=batch_size,
        input_length=sum(len(t.split()) for t in texts) // len(texts),
        num_threads=1,
        total_texts=len(texts),
        total_tokens=total_tokens,
        time_seconds=avg_time,
        tokens_per_second=total_tokens / avg_time,
        texts_per_second=len(texts) / avg_time,
        time_per_text_us=avg_time * 1_000_000 / len(texts),
    )


def validate_accuracy(budtiktok_tok, hf_tok, texts: List[str]) -> Dict[str, Any]:
    """Validate that BudTikTok produces same results as HuggingFace."""
    mismatches = []
    total_tokens_budtiktok = 0
    total_tokens_hf = 0

    for i, text in enumerate(texts[:1000]):  # Check first 1000
        # BudTikTok
        bt_enc = budtiktok_tok.encode(text, add_special_tokens=True)
        bt_ids = bt_enc.ids

        # HuggingFace
        hf_enc = hf_tok(text, add_special_tokens=True)
        hf_ids = hf_enc["input_ids"]

        total_tokens_budtiktok += len(bt_ids)
        total_tokens_hf += len(hf_ids)

        if bt_ids != hf_ids:
            mismatches.append({
                "index": i,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "budtiktok_ids": bt_ids[:20],
                "hf_ids": hf_ids[:20],
                "budtiktok_len": len(bt_ids),
                "hf_len": len(hf_ids),
            })

    return {
        "total_checked": min(len(texts), 1000),
        "mismatches": len(mismatches),
        "match_rate": 1.0 - len(mismatches) / min(len(texts), 1000),
        "total_tokens_budtiktok": total_tokens_budtiktok,
        "total_tokens_hf": total_tokens_hf,
        "sample_mismatches": mismatches[:5],
    }


def run_batch_size_benchmark(
    budtiktok_tok,
    hf_tok,
    texts: List[str],
    batch_sizes: List[int],
) -> List[Tuple[BenchmarkResult, Optional[BenchmarkResult]]]:
    """Run benchmarks across different batch sizes."""
    results = []

    for batch_size in batch_sizes:
        print(f"  Batch size {batch_size}...", end=" ", flush=True)

        bt_result = None
        hf_result = None

        if BUDTIKTOK_AVAILABLE and budtiktok_tok:
            bt_result = benchmark_budtiktok(budtiktok_tok, texts, batch_size)

        if HF_AVAILABLE and hf_tok:
            hf_result = benchmark_huggingface(hf_tok, texts, batch_size)

        if bt_result and hf_result:
            speedup = hf_result.time_seconds / bt_result.time_seconds
            print(f"BT: {bt_result.time_per_text_us:.1f}µs, HF: {hf_result.time_per_text_us:.1f}µs, Speedup: {speedup:.2f}x")
        elif bt_result:
            print(f"BT: {bt_result.time_per_text_us:.1f}µs")
        elif hf_result:
            print(f"HF: {hf_result.time_per_text_us:.1f}µs")

        results.append((bt_result, hf_result))

    return results


def run_input_length_benchmark(
    budtiktok_tok,
    hf_tok,
    length_configs: List[Tuple[int, int]],  # (min_words, max_words)
    num_texts: int = 1000,
    batch_size: int = 32,
) -> List[Tuple[BenchmarkResult, Optional[BenchmarkResult]]]:
    """Run benchmarks across different input lengths."""
    results = []

    for min_words, max_words in length_configs:
        print(f"  Input length {min_words}-{max_words} words...", end=" ", flush=True)

        texts = generate_texts(num_texts, min_words, max_words)

        bt_result = None
        hf_result = None

        if BUDTIKTOK_AVAILABLE and budtiktok_tok:
            bt_result = benchmark_budtiktok(budtiktok_tok, texts, batch_size)

        if HF_AVAILABLE and hf_tok:
            hf_result = benchmark_huggingface(hf_tok, texts, batch_size)

        if bt_result and hf_result:
            speedup = hf_result.time_seconds / bt_result.time_seconds
            print(f"BT: {bt_result.tokens_per_second:.0f} tok/s, HF: {hf_result.tokens_per_second:.0f} tok/s, Speedup: {speedup:.2f}x")
        elif bt_result:
            print(f"BT: {bt_result.tokens_per_second:.0f} tok/s")
        elif hf_result:
            print(f"HF: {hf_result.tokens_per_second:.0f} tok/s")

        results.append((bt_result, hf_result))

    return results


def run_concurrent_benchmark(
    budtiktok_tok,
    hf_tok,
    texts: List[str],
    thread_counts: List[int],
    batch_size: int = 32,
) -> Dict[str, List[BenchmarkResult]]:
    """Run benchmarks with different thread pool sizes."""
    results = {"budtiktok": [], "huggingface": []}

    def tokenize_batch_budtiktok(batch):
        return budtiktok_tok.encode_batch(batch, add_special_tokens=True)

    def tokenize_batch_hf(batch):
        return hf_tok(batch, add_special_tokens=True, padding=False, truncation=False)

    for num_threads in thread_counts:
        print(f"  {num_threads} threads...", end=" ", flush=True)

        # Split texts into batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        # BudTikTok with thread pool
        if BUDTIKTOK_AVAILABLE and budtiktok_tok:
            gc.collect()
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                list(executor.map(tokenize_batch_budtiktok, batches))
            bt_time = time.perf_counter() - start

            total_tokens = sum(len(e) for batch in batches for e in tokenize_batch_budtiktok(batch))
            bt_result = BenchmarkResult(
                name="BudTikTok",
                batch_size=batch_size,
                input_length=sum(len(t.split()) for t in texts) // len(texts),
                num_threads=num_threads,
                total_texts=len(texts),
                total_tokens=total_tokens,
                time_seconds=bt_time,
                tokens_per_second=total_tokens / bt_time,
                texts_per_second=len(texts) / bt_time,
                time_per_text_us=bt_time * 1_000_000 / len(texts),
            )
            results["budtiktok"].append(bt_result)

        # HuggingFace with thread pool
        if HF_AVAILABLE and hf_tok:
            gc.collect()
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                list(executor.map(tokenize_batch_hf, batches))
            hf_time = time.perf_counter() - start

            total_tokens = sum(len(ids) for batch in batches for ids in tokenize_batch_hf(batch)["input_ids"])
            hf_result = BenchmarkResult(
                name="HuggingFace",
                batch_size=batch_size,
                input_length=sum(len(t.split()) for t in texts) // len(texts),
                num_threads=num_threads,
                total_texts=len(texts),
                total_tokens=total_tokens,
                time_seconds=hf_time,
                tokens_per_second=total_tokens / hf_time,
                texts_per_second=len(texts) / hf_time,
                time_per_text_us=hf_time * 1_000_000 / len(texts),
            )
            results["huggingface"].append(hf_result)

        if results["budtiktok"] and results["huggingface"]:
            speedup = results["huggingface"][-1].time_seconds / results["budtiktok"][-1].time_seconds
            print(f"BT: {results['budtiktok'][-1].texts_per_second:.0f} texts/s, "
                  f"HF: {results['huggingface'][-1].texts_per_second:.0f} texts/s, "
                  f"Speedup: {speedup:.2f}x")
        else:
            print("Done")

    return results


def print_summary(all_results: Dict[str, Any]):
    """Print a summary of all benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Batch size results
    if "batch_size" in all_results:
        print("\n📊 Batch Size Scaling:")
        print("-" * 60)
        print(f"{'Batch Size':>12} │ {'BudTikTok (µs)':>15} │ {'HuggingFace (µs)':>17} │ {'Speedup':>8}")
        print("-" * 60)
        for bt, hf in all_results["batch_size"]:
            if bt and hf:
                speedup = hf.time_per_text_us / bt.time_per_text_us
                print(f"{bt.batch_size:>12} │ {bt.time_per_text_us:>15.1f} │ {hf.time_per_text_us:>17.1f} │ {speedup:>7.2f}x")

    # Input length results
    if "input_length" in all_results:
        print("\n📏 Input Length Scaling:")
        print("-" * 60)
        print(f"{'Input Length':>12} │ {'BudTikTok (tok/s)':>18} │ {'HuggingFace (tok/s)':>19} │ {'Speedup':>8}")
        print("-" * 60)
        for bt, hf in all_results["input_length"]:
            if bt and hf:
                speedup = bt.tokens_per_second / hf.tokens_per_second
                print(f"{bt.input_length:>12} │ {bt.tokens_per_second:>18,.0f} │ {hf.tokens_per_second:>19,.0f} │ {speedup:>7.2f}x")

    # Concurrency results
    if "concurrency" in all_results:
        print("\n🧵 Concurrency Scaling:")
        print("-" * 60)
        print(f"{'Threads':>12} │ {'BudTikTok (texts/s)':>20} │ {'HuggingFace (texts/s)':>21} │ {'Speedup':>8}")
        print("-" * 60)
        bt_results = all_results["concurrency"].get("budtiktok", [])
        hf_results = all_results["concurrency"].get("huggingface", [])
        for bt, hf in zip(bt_results, hf_results):
            if bt and hf:
                speedup = bt.texts_per_second / hf.texts_per_second
                print(f"{bt.num_threads:>12} │ {bt.texts_per_second:>20,.0f} │ {hf.texts_per_second:>21,.0f} │ {speedup:>7.2f}x")

    # Accuracy results
    if "accuracy" in all_results:
        acc = all_results["accuracy"]
        print("\n✅ Accuracy Validation:")
        print("-" * 60)
        print(f"  Texts checked: {acc['total_checked']}")
        print(f"  Mismatches: {acc['mismatches']}")
        print(f"  Match rate: {acc['match_rate'] * 100:.2f}%")
        print(f"  Total tokens (BudTikTok): {acc['total_tokens_budtiktok']:,}")
        print(f"  Total tokens (HuggingFace): {acc['total_tokens_hf']:,}")

        if acc['sample_mismatches']:
            print("\n  Sample mismatches:")
            for m in acc['sample_mismatches'][:3]:
                print(f"    Text: {m['text']}")
                print(f"    BT: {m['budtiktok_ids']}")
                print(f"    HF: {m['hf_ids']}")

    # Hardware info
    if BUDTIKTOK_AVAILABLE:
        config = budtiktok.get_config()
        print("\n🖥️ Hardware Configuration:")
        print("-" * 60)
        print(f"  SIMD ISA: {config['best_isa']}")
        print(f"  Physical cores: {config['physical_cores']}")
        print(f"  Logical cores: {config['logical_cores']}")
        print(f"  SIMD pretokenizer: {config['use_simd_pretokenizer']}")
        print(f"  Recommended batch size: {config['recommended_batch_size']}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive BudTikTok benchmark")
    parser.add_argument(
        "--model-path",
        default="/home/bud/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        help="Path to model directory with tokenizer.json"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer iterations")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()

    print("=" * 80)
    print("BudTikTok vs HuggingFace Tokenizers - Comprehensive Benchmark")
    print("=" * 80)

    # Load tokenizers
    budtiktok_tok = None
    hf_tok = None

    tokenizer_json = os.path.join(args.model_path, "tokenizer.json")

    if BUDTIKTOK_AVAILABLE:
        print(f"\n📦 Loading BudTikTok from: {tokenizer_json}")
        budtiktok_tok = budtiktok.Tokenizer.from_file(tokenizer_json)
        print(f"   Vocab size: {budtiktok_tok.vocab_size}")
        config = budtiktok.get_config()
        print(f"   SIMD ISA: {config['best_isa']}, Cores: {config['physical_cores']}")

    if HF_AVAILABLE:
        print(f"\n📦 Loading HuggingFace from: {args.model_path}")
        hf_tok = AutoTokenizer.from_pretrained(args.model_path)
        print(f"   Vocab size: {len(hf_tok)}")

    if not budtiktok_tok and not hf_tok:
        print("Error: Neither tokenizer could be loaded!")
        sys.exit(1)

    # Generate test data
    num_texts = 500 if args.quick else 5000
    print(f"\n📝 Generating {num_texts} test texts...")
    texts = generate_texts(num_texts, 10, 100)

    all_results = {}

    # 1. Batch size benchmark
    print("\n🔬 Running batch size benchmark...")
    batch_sizes = [1, 4, 8, 16, 32, 64, 128] if not args.quick else [1, 16, 64]
    all_results["batch_size"] = run_batch_size_benchmark(budtiktok_tok, hf_tok, texts, batch_sizes)

    # 2. Input length benchmark
    print("\n🔬 Running input length benchmark...")
    length_configs = [
        (5, 10),     # Short
        (20, 40),    # Medium
        (50, 100),   # Long
        (100, 200),  # Very long
    ] if not args.quick else [(10, 30), (50, 100)]
    all_results["input_length"] = run_input_length_benchmark(
        budtiktok_tok, hf_tok, length_configs,
        num_texts=500 if args.quick else 2000
    )

    # 3. Concurrency benchmark
    print("\n🔬 Running concurrency benchmark...")
    thread_counts = [1, 2, 4, 8] if not args.quick else [1, 4]
    all_results["concurrency"] = run_concurrent_benchmark(
        budtiktok_tok, hf_tok, texts[:1000], thread_counts
    )

    # 4. Accuracy validation
    if budtiktok_tok and hf_tok:
        print("\n🔬 Running accuracy validation...")
        all_results["accuracy"] = validate_accuracy(budtiktok_tok, hf_tok, texts)

    # Print summary
    print_summary(all_results)

    # Save results
    if args.output:
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in all_results.items():
            if key in ["batch_size", "input_length"]:
                json_results[key] = [
                    {
                        "budtiktok": vars(bt) if bt else None,
                        "huggingface": vars(hf) if hf else None,
                    }
                    for bt, hf in value
                ]
            elif key == "concurrency":
                json_results[key] = {
                    k: [vars(r) for r in v]
                    for k, v in value.items()
                }
            else:
                json_results[key] = value

        with open(args.output, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
