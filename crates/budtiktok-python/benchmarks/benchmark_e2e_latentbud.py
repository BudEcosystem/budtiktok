#!/usr/bin/env python3
"""
End-to-End LatentBud Benchmark with BudTikTok Integration

This benchmark measures the full embedding pipeline performance:
- Tokenization phase (BudTikTok vs HuggingFace)
- Full embedding throughput (requests/sec)
- Latency distribution (P50, P95, P99)
- Token-budget batching efficiency
- Accuracy validation

Usage:
    python benchmark_e2e_latentbud.py [--model MODEL] [--quick]
"""

import argparse
import asyncio
import gc
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Add LatentBud to path
LATENTBUD_PATH = Path("/home/bud/Desktop/latentbud/LatentBud/libs/infinity_emb")
if str(LATENTBUD_PATH) not in sys.path:
    sys.path.insert(0, str(LATENTBUD_PATH))

# Check available packages
BUDTIKTOK_AVAILABLE = False
HF_TOKENIZERS_AVAILABLE = False
LATENTBUD_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import budtiktok
    BUDTIKTOK_AVAILABLE = True
    print(f"✓ BudTikTok available: ISA={budtiktok.get_config()['best_isa']}")
except ImportError as e:
    print(f"✗ BudTikTok not available: {e}")

try:
    from transformers import AutoTokenizer
    HF_TOKENIZERS_AVAILABLE = True
    print("✓ HuggingFace transformers available")
except ImportError:
    print("✗ HuggingFace transformers not available")

try:
    import torch
    TORCH_AVAILABLE = True
    print(f"✓ PyTorch available: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except ImportError:
    print("✗ PyTorch not available")

try:
    from infinity_emb import AsyncEmbeddingEngine, EngineArgs
    from infinity_emb.inference.token_budget_queue import TokenBudgetQueue
    from infinity_emb.primitives import PrioritizedQueueItem, EmbeddingSingle, EmbeddingInner
    LATENTBUD_AVAILABLE = True
    print("✓ LatentBud available")
except ImportError as e:
    print(f"✗ LatentBud not available: {e}")
    # Define dummy classes if not available
    TokenBudgetQueue = None
    PrioritizedQueueItem = None
    EmbeddingInner = None
    EmbeddingSingle = None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    model_path: str = "/home/bud/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    num_texts: int = 5000
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32, 64, 128])
    input_lengths: List[Tuple[int, int]] = field(default_factory=lambda: [(5, 10), (20, 40), (50, 100), (100, 200)])
    warmup_runs: int = 3
    benchmark_runs: int = 10
    concurrent_requests: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])


@dataclass
class TokenizationResult:
    """Results from tokenization benchmark."""
    name: str
    batch_size: int
    num_texts: int
    total_tokens: int
    time_seconds: float
    tokens_per_second: float
    texts_per_second: float
    time_per_text_us: float


@dataclass
class LatencyResult:
    """Latency distribution results."""
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass
class ThroughputResult:
    """Throughput benchmark results."""
    name: str
    concurrent_requests: int
    batch_size: int
    total_requests: int
    total_embeddings: int
    time_seconds: float
    requests_per_second: float
    embeddings_per_second: float
    latency: LatencyResult


def generate_texts(count: int, min_words: int, max_words: int, seed: int = 42) -> List[str]:
    """Generate random texts of varying lengths."""
    random.seed(seed)
    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "machine", "learning", "neural", "network", "embedding", "vector",
        "transformer", "attention", "model", "training", "inference", "batch",
        "sentence", "document", "semantic", "similarity", "search", "retrieval",
    ]
    texts = []
    for _ in range(count):
        length = random.randint(min_words, max_words)
        text_words = [random.choice(words) for _ in range(length)]
        texts.append(" ".join(text_words))
    return texts


class TokenizationBenchmark:
    """Benchmark tokenization performance."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.budtiktok_tok = None
        self.hf_tok = None

        # Load tokenizers
        if BUDTIKTOK_AVAILABLE:
            tokenizer_json = os.path.join(model_path, "tokenizer.json")
            if os.path.exists(tokenizer_json):
                self.budtiktok_tok = budtiktok.Tokenizer.from_file(tokenizer_json)
                print(f"  BudTikTok loaded: vocab_size={self.budtiktok_tok.vocab_size}")

        if HF_TOKENIZERS_AVAILABLE:
            self.hf_tok = AutoTokenizer.from_pretrained(model_path)
            print(f"  HuggingFace loaded: vocab_size={len(self.hf_tok)}")

    def benchmark_budtiktok(
        self,
        texts: List[str],
        batch_size: int,
        warmup: int = 3,
        runs: int = 10,
    ) -> Optional[TokenizationResult]:
        """Benchmark BudTikTok tokenizer."""
        if self.budtiktok_tok is None:
            return None

        total_tokens = 0

        # Warmup
        for _ in range(warmup):
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                _ = self.budtiktok_tok.encode_batch(batch, add_special_tokens=True)

        gc.collect()

        # Benchmark
        times = []
        for run_idx in range(runs):
            start = time.perf_counter()
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                encodings = self.budtiktok_tok.encode_batch(batch, add_special_tokens=True)
                if run_idx == 0:
                    total_tokens += sum(len(e) for e in encodings)
            times.append(time.perf_counter() - start)

        avg_time = statistics.mean(times)
        return TokenizationResult(
            name="BudTikTok",
            batch_size=batch_size,
            num_texts=len(texts),
            total_tokens=total_tokens,
            time_seconds=avg_time,
            tokens_per_second=total_tokens / avg_time,
            texts_per_second=len(texts) / avg_time,
            time_per_text_us=avg_time * 1_000_000 / len(texts),
        )

    def benchmark_huggingface(
        self,
        texts: List[str],
        batch_size: int,
        warmup: int = 3,
        runs: int = 10,
    ) -> Optional[TokenizationResult]:
        """Benchmark HuggingFace tokenizer."""
        if self.hf_tok is None:
            return None

        total_tokens = 0

        # Warmup
        for _ in range(warmup):
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                _ = self.hf_tok(batch, add_special_tokens=True, padding=False, truncation=False)

        gc.collect()

        # Benchmark
        times = []
        for run_idx in range(runs):
            start = time.perf_counter()
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                result = self.hf_tok(batch, add_special_tokens=True, padding=False, truncation=False)
                if run_idx == 0:
                    total_tokens += sum(len(ids) for ids in result["input_ids"])
            times.append(time.perf_counter() - start)

        avg_time = statistics.mean(times)
        return TokenizationResult(
            name="HuggingFace",
            batch_size=batch_size,
            num_texts=len(texts),
            total_tokens=total_tokens,
            time_seconds=avg_time,
            tokens_per_second=total_tokens / avg_time,
            texts_per_second=len(texts) / avg_time,
            time_per_text_us=avg_time * 1_000_000 / len(texts),
        )

    def validate_accuracy(self, texts: List[str]) -> Dict[str, Any]:
        """Validate BudTikTok produces same results as HuggingFace."""
        if self.budtiktok_tok is None or self.hf_tok is None:
            return {"error": "Both tokenizers required for validation"}

        mismatches = []
        total_tokens_bt = 0
        total_tokens_hf = 0

        for i, text in enumerate(texts[:1000]):
            bt_enc = self.budtiktok_tok.encode(text, add_special_tokens=True)
            bt_ids = bt_enc.ids

            hf_enc = self.hf_tok(text, add_special_tokens=True)
            hf_ids = hf_enc["input_ids"]

            total_tokens_bt += len(bt_ids)
            total_tokens_hf += len(hf_ids)

            if bt_ids != hf_ids:
                mismatches.append({
                    "index": i,
                    "text": text[:50],
                    "bt_len": len(bt_ids),
                    "hf_len": len(hf_ids),
                })

        return {
            "total_checked": min(len(texts), 1000),
            "mismatches": len(mismatches),
            "match_rate": 1.0 - len(mismatches) / min(len(texts), 1000),
            "total_tokens_budtiktok": total_tokens_bt,
            "total_tokens_huggingface": total_tokens_hf,
            "sample_mismatches": mismatches[:3],
        }


class TokenBudgetQueueBenchmark:
    """Benchmark token-budget queue with different tokenizers."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.budtiktok_tok = None
        self.hf_tok = None

        if BUDTIKTOK_AVAILABLE:
            tokenizer_json = os.path.join(model_path, "tokenizer.json")
            if os.path.exists(tokenizer_json):
                self.budtiktok_tok = budtiktok.Tokenizer.from_file(tokenizer_json)

        if HF_TOKENIZERS_AVAILABLE:
            self.hf_tok = AutoTokenizer.from_pretrained(model_path)

    def get_token_lengths_budtiktok(self, texts: List[str]) -> List[int]:
        """Get token lengths using BudTikTok."""
        if self.budtiktok_tok is None:
            raise ValueError("BudTikTok not available")
        return self.budtiktok_tok.get_token_lengths(texts, add_special_tokens=True)

    def get_token_lengths_hf(self, texts: List[str]) -> List[int]:
        """Get token lengths using HuggingFace."""
        if self.hf_tok is None:
            raise ValueError("HuggingFace not available")
        encodings = self.hf_tok(texts, add_special_tokens=True, padding=False, truncation=False)
        return [len(ids) for ids in encodings["input_ids"]]

    def benchmark_queue_formation(
        self,
        texts: List[str],
        max_batch_tokens: int = 16384,
        max_batch_size: int = 64,
        use_budtiktok: bool = True,
    ) -> Dict[str, Any]:
        """Benchmark queue formation with token-budget batching."""
        if TokenBudgetQueue is None:
            return {"error": "LatentBud not available for queue benchmarks"}

        # Get token lengths
        start = time.perf_counter()
        if use_budtiktok and self.budtiktok_tok:
            lengths = self.get_token_lengths_budtiktok(texts)
            method = "BudTikTok"
        else:
            lengths = self.get_token_lengths_hf(texts)
            method = "HuggingFace"
        length_time = time.perf_counter() - start

        # Create queue and add items
        queue = TokenBudgetQueue()

        items = []
        import asyncio

        # Create a temporary event loop for futures
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        for i, (text, length) in enumerate(zip(texts, lengths)):
            content = EmbeddingSingle(sentence=text)
            inner = EmbeddingInner(content=content, future=loop.create_future())
            item = PrioritizedQueueItem(priority=length, item=inner)
            items.append(item)

        start = time.perf_counter()
        queue.extend(items)

        # Pop batches
        batches = []
        batch_stats = []
        for batch in queue.pop_optimal_batches_by_tokens(
            max_tokens=max_batch_tokens,
            max_size=max_batch_size,
            max_n_batches=100,
        ):
            batches.append(len(batch))

        queue_time = time.perf_counter() - start

        return {
            "method": method,
            "num_texts": len(texts),
            "length_estimation_time_ms": length_time * 1000,
            "queue_formation_time_ms": queue_time * 1000,
            "total_time_ms": (length_time + queue_time) * 1000,
            "num_batches": len(batches),
            "avg_batch_size": sum(batches) / len(batches) if batches else 0,
            "total_tokens": sum(lengths),
            "avg_tokens_per_text": sum(lengths) / len(lengths),
        }


async def run_full_embedding_benchmark(
    model_path: str,
    texts: List[str],
    concurrent_requests: int = 1,
    batch_size: int = 32,
    use_budtiktok: bool = True,  # Note: Currently always uses BudTikTok if available
) -> Optional[ThroughputResult]:
    """Run full embedding pipeline benchmark.

    Note: BudTikTok usage is determined by the model's initialization,
    defaulting to True when available. The use_budtiktok param here
    is for labeling purposes only.
    """
    if not LATENTBUD_AVAILABLE or not TORCH_AVAILABLE:
        return None

    try:
        # Configure engine - EngineArgs is frozen, so pass all params at construction
        engine_args = EngineArgs(
            model_name_or_path=model_path,
            engine="torch",
            dtype="float16" if torch.cuda.is_available() else "float32",
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_warmup=True,
        )

        engine = AsyncEmbeddingEngine.from_args(engine_args)

        async with engine:
            # Warmup
            for _ in range(3):
                await engine.embed(sentences=texts[:batch_size])

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Benchmark with concurrent requests
            latencies = []
            total_embeddings = 0

            # Split texts into batches
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

            start = time.perf_counter()

            if concurrent_requests == 1:
                # Sequential
                for batch in batches:
                    batch_start = time.perf_counter()
                    embeddings, _ = await engine.embed(sentences=batch)
                    latencies.append((time.perf_counter() - batch_start) * 1000)
                    total_embeddings += len(embeddings)
            else:
                # Concurrent
                sem = asyncio.Semaphore(concurrent_requests)

                async def embed_batch(batch):
                    async with sem:
                        batch_start = time.perf_counter()
                        embeddings, _ = await engine.embed(sentences=batch)
                        return (time.perf_counter() - batch_start) * 1000, len(embeddings)

                results = await asyncio.gather(*[embed_batch(b) for b in batches])
                for latency_ms, count in results:
                    latencies.append(latency_ms)
                    total_embeddings += count

            total_time = time.perf_counter() - start

            # Calculate latency distribution
            latencies.sort()
            latency_result = LatencyResult(
                p50_ms=latencies[int(len(latencies) * 0.50)],
                p95_ms=latencies[int(len(latencies) * 0.95)],
                p99_ms=latencies[int(len(latencies) * 0.99)],
                mean_ms=statistics.mean(latencies),
                min_ms=min(latencies),
                max_ms=max(latencies),
            )

            return ThroughputResult(
                name="BudTikTok" if use_budtiktok else "HuggingFace",
                concurrent_requests=concurrent_requests,
                batch_size=batch_size,
                total_requests=len(batches),
                total_embeddings=total_embeddings,
                time_seconds=total_time,
                requests_per_second=len(batches) / total_time,
                embeddings_per_second=total_embeddings / total_time,
                latency=latency_result,
            )

    except Exception as e:
        print(f"  Error running embedding benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_tokenization_comparison(results: List[Tuple[Optional[TokenizationResult], Optional[TokenizationResult]]]):
    """Print tokenization benchmark comparison."""
    print("\n" + "=" * 80)
    print("TOKENIZATION BENCHMARK")
    print("=" * 80)
    print(f"\n{'Batch Size':>12} │ {'BudTikTok (µs/text)':>20} │ {'HuggingFace (µs/text)':>22} │ {'Speedup':>10}")
    print("-" * 72)

    for bt, hf in results:
        if bt and hf:
            speedup = hf.time_per_text_us / bt.time_per_text_us
            print(f"{bt.batch_size:>12} │ {bt.time_per_text_us:>20.1f} │ {hf.time_per_text_us:>22.1f} │ {speedup:>9.2f}x")
        elif bt:
            print(f"{bt.batch_size:>12} │ {bt.time_per_text_us:>20.1f} │ {'N/A':>22} │ {'N/A':>10}")
        elif hf:
            print(f"{hf.batch_size:>12} │ {'N/A':>20} │ {hf.time_per_text_us:>22.1f} │ {'N/A':>10}")


def print_queue_comparison(results: Dict[str, Dict[str, Any]]):
    """Print queue formation benchmark comparison."""
    print("\n" + "=" * 80)
    print("TOKEN-BUDGET QUEUE FORMATION")
    print("=" * 80)

    for name, stats in results.items():
        print(f"\n{name}:")
        print(f"  Length estimation: {stats['length_estimation_time_ms']:.2f} ms")
        print(f"  Queue formation:   {stats['queue_formation_time_ms']:.2f} ms")
        print(f"  Total time:        {stats['total_time_ms']:.2f} ms")
        print(f"  Batches formed:    {stats['num_batches']}")
        print(f"  Avg batch size:    {stats['avg_batch_size']:.1f}")


def print_throughput_comparison(results: List[Tuple[Optional[ThroughputResult], Optional[ThroughputResult]]]):
    """Print throughput benchmark comparison."""
    print("\n" + "=" * 80)
    print("FULL EMBEDDING THROUGHPUT (with BudTikTok)")
    print("=" * 80)

    print(f"\n{'Concurrency':>12} │ {'Embeddings/sec':>16} │ {'Requests/sec':>14} │ {'P50 (ms)':>10} │ {'P99 (ms)':>10}")
    print("-" * 72)

    for bt, _ in results:
        if bt:
            print(f"{bt.concurrent_requests:>12} │ {bt.embeddings_per_second:>16,.0f} │ {bt.requests_per_second:>14.1f} │ {bt.latency.p50_ms:>10.1f} │ {bt.latency.p99_ms:>10.1f}")

    print("\n" + "-" * 72)
    print("LATENCY DISTRIBUTION (ms)")
    print("-" * 72)
    print(f"{'Concurrency':>12} │ {'Min':>8} │ {'P50':>8} │ {'P95':>8} │ {'P99':>8} │ {'Max':>8}")
    print("-" * 72)

    for bt, _ in results:
        if bt:
            lat = bt.latency
            print(f"{bt.concurrent_requests:>12} │ {lat.min_ms:>8.1f} │ {lat.p50_ms:>8.1f} │ {lat.p95_ms:>8.1f} │ {lat.p99_ms:>8.1f} │ {lat.max_ms:>8.1f}")


async def main():
    parser = argparse.ArgumentParser(description="E2E LatentBud Benchmark with BudTikTok")
    parser.add_argument(
        "--model",
        default="/home/bud/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        help="Model path"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--tokenizer-only", action="store_true", help="Only run tokenizer benchmarks")
    args = parser.parse_args()

    config = BenchmarkConfig(model_path=args.model)
    if args.quick:
        config.num_texts = 1000
        config.batch_sizes = [1, 32]
        config.concurrent_requests = [1, 4]
        config.benchmark_runs = 5

    print("\n" + "=" * 80)
    print("E2E LatentBud Benchmark with BudTikTok Integration")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_path}")
    print(f"  Texts: {config.num_texts}")
    print(f"  Batch sizes: {config.batch_sizes}")
    print(f"  Quick mode: {args.quick}")

    all_results = {}

    # Generate test texts
    print(f"\n📝 Generating {config.num_texts} test texts...")
    texts = generate_texts(config.num_texts, 10, 100)

    # 1. Tokenization Benchmark
    print("\n🔬 Running tokenization benchmark...")
    tok_bench = TokenizationBenchmark(config.model_path)

    tokenization_results = []
    for batch_size in config.batch_sizes:
        print(f"  Batch size {batch_size}...", end=" ", flush=True)
        bt = tok_bench.benchmark_budtiktok(texts, batch_size, runs=config.benchmark_runs)
        hf = tok_bench.benchmark_huggingface(texts, batch_size, runs=config.benchmark_runs)
        tokenization_results.append((bt, hf))

        if bt and hf:
            speedup = hf.time_per_text_us / bt.time_per_text_us
            print(f"BT: {bt.time_per_text_us:.1f}µs, HF: {hf.time_per_text_us:.1f}µs, Speedup: {speedup:.2f}x")
        else:
            print("Done")

    all_results["tokenization"] = tokenization_results

    # 2. Accuracy Validation
    print("\n🔬 Running accuracy validation...")
    accuracy = tok_bench.validate_accuracy(texts)
    all_results["accuracy"] = accuracy
    print(f"  Match rate: {accuracy['match_rate'] * 100:.2f}%")
    print(f"  Mismatches: {accuracy['mismatches']} / {accuracy['total_checked']}")

    # 3. Token-Budget Queue Benchmark
    print("\n🔬 Running token-budget queue benchmark...")
    queue_bench = TokenBudgetQueueBenchmark(config.model_path)

    queue_results = {}
    if LATENTBUD_AVAILABLE:
        if BUDTIKTOK_AVAILABLE and queue_bench.budtiktok_tok:
            queue_results["BudTikTok"] = queue_bench.benchmark_queue_formation(texts, use_budtiktok=True)
        if HF_TOKENIZERS_AVAILABLE and queue_bench.hf_tok:
            queue_results["HuggingFace"] = queue_bench.benchmark_queue_formation(texts, use_budtiktok=False)
    else:
        print("  ⚠️ Skipping queue benchmark (LatentBud not available)")

    all_results["queue_formation"] = queue_results

    if ("BudTikTok" in queue_results and "HuggingFace" in queue_results and
        "error" not in queue_results.get("BudTikTok", {}) and
        "error" not in queue_results.get("HuggingFace", {})):
        bt_time = queue_results["BudTikTok"]["total_time_ms"]
        hf_time = queue_results["HuggingFace"]["total_time_ms"]
        print(f"  BudTikTok: {bt_time:.2f}ms, HuggingFace: {hf_time:.2f}ms, Speedup: {hf_time/bt_time:.2f}x")

    # 4. Full Embedding Pipeline (if not tokenizer-only)
    throughput_results = []
    if not args.tokenizer_only and LATENTBUD_AVAILABLE and TORCH_AVAILABLE:
        print("\n🔬 Running full embedding pipeline benchmark (with BudTikTok)...")
        print("  Note: BudTikTok is enabled by default when available")

        for concurrent in config.concurrent_requests:
            print(f"  Concurrency {concurrent}...", end=" ", flush=True)

            # Run with BudTikTok (default when available)
            bt_result = await run_full_embedding_benchmark(
                config.model_path,
                texts[:min(1000, len(texts))],  # Use subset for speed
                concurrent_requests=concurrent,
                batch_size=32,
                use_budtiktok=True,
            )

            throughput_results.append((bt_result, None))

            if bt_result:
                lat = bt_result.latency
                print(f"{bt_result.embeddings_per_second:.0f} emb/s, "
                      f"P50={lat.p50_ms:.1f}ms, P99={lat.p99_ms:.1f}ms")
            else:
                print("Skipped")

        all_results["throughput"] = throughput_results
    else:
        print("\n⚠️ Skipping full embedding benchmark (--tokenizer-only or missing dependencies)")

    # Print summaries
    print_tokenization_comparison(tokenization_results)
    print_queue_comparison(queue_results)
    if throughput_results:
        print_throughput_comparison(throughput_results)

    # Hardware info
    if BUDTIKTOK_AVAILABLE:
        config_info = budtiktok.get_config()
        print("\n" + "=" * 80)
        print("HARDWARE CONFIGURATION")
        print("=" * 80)
        print(f"  SIMD ISA: {config_info['best_isa']}")
        print(f"  Physical cores: {config_info['physical_cores']}")
        print(f"  Logical cores: {config_info['logical_cores']}")
        print(f"  SIMD pretokenizer: {config_info['use_simd_pretokenizer']}")

    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Save results
    if args.output:
        # Convert results to JSON-serializable format
        def serialize_result(r):
            if r is None:
                return None
            if isinstance(r, (TokenizationResult, ThroughputResult, LatencyResult)):
                return asdict(r)
            return r

        json_results = {
            "tokenization": [
                {"budtiktok": serialize_result(bt), "huggingface": serialize_result(hf)}
                for bt, hf in tokenization_results
            ],
            "accuracy": accuracy,
            "queue_formation": queue_results,
        }
        if throughput_results:
            json_results["throughput"] = [
                {"budtiktok": serialize_result(bt), "huggingface": serialize_result(hf)}
                for bt, hf in throughput_results
            ]

        with open(args.output, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())
