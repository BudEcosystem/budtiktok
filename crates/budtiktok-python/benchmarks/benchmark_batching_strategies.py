#!/usr/bin/env python3
"""
Comprehensive Batching Strategies Benchmark

Tests different batching algorithms and tokenizer configurations:
- Token-budget batching vs size-based batching
- HACC (Hardware-Aware Cost-Constrained) batching
- BudTikTok vs HuggingFace tokenizers
- GPU utilization monitoring

Usage:
    python benchmark_batching_strategies.py [--model MODEL] [--quick]
"""

import argparse
import asyncio
import gc
import json
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Add LatentBud to path
LATENTBUD_PATH = Path("/home/bud/Desktop/latentbud/LatentBud/libs/infinity_emb")
if str(LATENTBUD_PATH) not in sys.path:
    sys.path.insert(0, str(LATENTBUD_PATH))

# Check available packages
TORCH_AVAILABLE = False
PYNVML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except Exception:
    pass


@dataclass
class GPUMetrics:
    """GPU utilization metrics."""
    utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_percent: float = 0.0
    power_watts: float = 0.0
    temperature_c: float = 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    config_name: str
    batch_strategy: str
    use_budtiktok: bool
    concurrent_requests: int
    batch_size: int
    max_batch_tokens: Optional[int]
    total_texts: int
    total_embeddings: int
    total_time_seconds: float
    throughput_embeddings_per_sec: float
    throughput_requests_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_min_ms: float
    latency_max_ms: float
    gpu_utilization_mean: float = 0.0
    gpu_utilization_max: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    tokenization_time_percent: float = 0.0
    error: Optional[str] = None


class GPUMonitor:
    """Monitor GPU metrics during benchmark."""

    def __init__(self, device_id: int = 0, sample_interval: float = 0.1):
        self.device_id = device_id
        self.sample_interval = sample_interval
        self.samples: List[GPUMetrics] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start monitoring in background thread."""
        if not PYNVML_AVAILABLE:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return aggregated stats."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

        if not self.samples:
            return {}

        utilizations = [s.utilization_percent for s in self.samples]
        memory_used = [s.memory_used_mb for s in self.samples]

        return {
            "gpu_utilization_mean": statistics.mean(utilizations),
            "gpu_utilization_max": max(utilizations),
            "gpu_utilization_min": min(utilizations),
            "gpu_memory_peak_mb": max(memory_used),
            "gpu_memory_mean_mb": statistics.mean(memory_used),
            "sample_count": len(self.samples),
        }

    def _monitor_loop(self):
        """Background monitoring loop."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            while self._running:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except Exception:
                        power = 0.0

                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except Exception:
                        temp = 0.0

                    self.samples.append(GPUMetrics(
                        utilization_percent=util.gpu,
                        memory_used_mb=mem.used / (1024 * 1024),
                        memory_total_mb=mem.total / (1024 * 1024),
                        memory_percent=(mem.used / mem.total) * 100,
                        power_watts=power,
                        temperature_c=temp,
                    ))
                except Exception:
                    pass

                time.sleep(self.sample_interval)
        except Exception:
            pass


def generate_texts(count: int, min_words: int, max_words: int, seed: int = 42) -> List[str]:
    """Generate random texts of varying lengths."""
    import random
    random.seed(seed)

    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "machine", "learning", "neural", "network", "embedding", "vector",
        "transformer", "attention", "model", "training", "inference", "batch",
        "sentence", "document", "semantic", "similarity", "search", "retrieval",
        "natural", "language", "processing", "deep", "representation", "encoder",
    ]

    texts = []
    for _ in range(count):
        length = random.randint(min_words, max_words)
        text_words = [random.choice(words) for _ in range(length)]
        texts.append(" ".join(text_words))
    return texts


async def run_benchmark(
    model_path: str,
    texts: List[str],
    batch_strategy: str = "auto",
    max_batch_tokens: Optional[int] = None,
    max_batch_size: int = 32,
    concurrent_requests: int = 1,
    use_budtiktok: bool = True,
    warmup_batches: int = 3,
) -> Optional[BenchmarkResult]:
    """Run a single benchmark configuration."""

    try:
        from infinity_emb import AsyncEmbeddingEngine, EngineArgs
    except ImportError as e:
        return BenchmarkResult(
            config_name=f"{batch_strategy}_bt{use_budtiktok}_c{concurrent_requests}",
            batch_strategy=batch_strategy,
            use_budtiktok=use_budtiktok,
            concurrent_requests=concurrent_requests,
            batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
            total_texts=0,
            total_embeddings=0,
            total_time_seconds=0,
            throughput_embeddings_per_sec=0,
            throughput_requests_per_sec=0,
            latency_p50_ms=0,
            latency_p95_ms=0,
            latency_p99_ms=0,
            latency_mean_ms=0,
            latency_min_ms=0,
            latency_max_ms=0,
            error=f"Import error: {e}",
        )

    tok_label = "budtiktok" if use_budtiktok else "huggingface"
    config_name = f"{batch_strategy}_{tok_label}_c{concurrent_requests}"
    if max_batch_tokens:
        config_name = f"{batch_strategy}_tok{max_batch_tokens//1024}k_{tok_label}_c{concurrent_requests}"

    # Control BudTikTok usage by patching the module
    if not use_budtiktok:
        import infinity_emb.inference.optimizations.budtiktok_tokenizer as bt_module
        original_available = getattr(bt_module, 'BUDTIKTOK_AVAILABLE', True)
        bt_module.BUDTIKTOK_AVAILABLE = False
    else:
        original_available = None

    try:
        # Build engine args with batch strategy and max_batch_tokens
        # Note: Disable prefetch for BudTikTok - its fast tokenization doesn't
        # benefit from prefetch overhead. HuggingFace benefits from prefetch
        # because its slower tokenization can overlap with GPU inference.
        use_prefetch = not use_budtiktok  # Only use prefetch with HuggingFace

        engine_args = EngineArgs(
            model_name_or_path=model_path,
            engine="torch",
            dtype="float16" if TORCH_AVAILABLE and torch.cuda.is_available() else "float32",
            device="cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
            model_warmup=True,
            batch_size=max_batch_size,
            batch_strategy=batch_strategy,
            max_batch_tokens=max_batch_tokens if max_batch_tokens else 0,
            prefetch_pipeline=use_prefetch,
            multiprocess_tokenizer=False if use_budtiktok else True,
            parallel_tokenizer=False if use_budtiktok else True,
        )

        engine = AsyncEmbeddingEngine.from_args(engine_args)

        # Start GPU monitoring
        gpu_monitor = GPUMonitor()

        async with engine:
            # Warmup
            batch_size = max_batch_size
            warmup_texts = texts[:batch_size * warmup_batches]
            for i in range(0, len(warmup_texts), batch_size):
                batch = warmup_texts[i:i + batch_size]
                await engine.embed(sentences=batch)

            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            # Start GPU monitoring
            gpu_monitor.start()

            # Benchmark
            latencies = []
            total_embeddings = 0
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

            start_time = time.perf_counter()

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

            total_time = time.perf_counter() - start_time

            # Stop GPU monitoring
            gpu_stats = gpu_monitor.stop()

            # Calculate latency distribution
            latencies.sort()
            n = len(latencies)

            result = BenchmarkResult(
                config_name=config_name,
                batch_strategy=batch_strategy,
                use_budtiktok=use_budtiktok,
                concurrent_requests=concurrent_requests,
                batch_size=max_batch_size,
                max_batch_tokens=max_batch_tokens,
                total_texts=len(texts),
                total_embeddings=total_embeddings,
                total_time_seconds=total_time,
                throughput_embeddings_per_sec=total_embeddings / total_time,
                throughput_requests_per_sec=len(batches) / total_time,
                latency_p50_ms=latencies[int(n * 0.50)],
                latency_p95_ms=latencies[int(n * 0.95)],
                latency_p99_ms=latencies[int(n * 0.99)] if n > 0 else latencies[-1],
                latency_mean_ms=statistics.mean(latencies),
                latency_min_ms=min(latencies),
                latency_max_ms=max(latencies),
                gpu_utilization_mean=gpu_stats.get("gpu_utilization_mean", 0),
                gpu_utilization_max=gpu_stats.get("gpu_utilization_max", 0),
                gpu_memory_peak_mb=gpu_stats.get("gpu_memory_peak_mb", 0),
            )
            return result

    except Exception as e:
        import traceback
        return BenchmarkResult(
            config_name=config_name,
            batch_strategy=batch_strategy,
            use_budtiktok=use_budtiktok,
            concurrent_requests=concurrent_requests,
            batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
            total_texts=0,
            total_embeddings=0,
            total_time_seconds=0,
            throughput_embeddings_per_sec=0,
            throughput_requests_per_sec=0,
            latency_p50_ms=0,
            latency_p95_ms=0,
            latency_p99_ms=0,
            latency_mean_ms=0,
            latency_min_ms=0,
            latency_max_ms=0,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )
    finally:
        # Restore original BudTikTok availability
        if original_available is not None:
            import infinity_emb.inference.optimizations.budtiktok_tokenizer as bt_module
            bt_module.BUDTIKTOK_AVAILABLE = original_available


def print_results_table(results: List[BenchmarkResult]):
    """Print results as a formatted table."""
    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS")
    print("=" * 120)

    # Header
    print(f"\n{'Config':<35} │ {'Throughput':>12} │ {'P50':>8} │ {'P99':>8} │ {'GPU Util':>10} │ {'GPU Mem':>10}")
    print(f"{'':35} │ {'(emb/s)':>12} │ {'(ms)':>8} │ {'(ms)':>8} │ {'(%)':>10} │ {'(MB)':>10}")
    print("-" * 120)

    for r in results:
        if r.error:
            print(f"{r.config_name:<35} │ {'ERROR':>12} │ {'-':>8} │ {'-':>8} │ {'-':>10} │ {'-':>10}")
            continue

        print(f"{r.config_name:<35} │ {r.throughput_embeddings_per_sec:>12,.0f} │ "
              f"{r.latency_p50_ms:>8.1f} │ {r.latency_p99_ms:>8.1f} │ "
              f"{r.gpu_utilization_mean:>10.1f} │ {r.gpu_memory_peak_mb:>10.0f}")


def print_comparison(results: List[BenchmarkResult]):
    """Print comparison analysis."""
    print("\n" + "=" * 120)
    print("ANALYSIS")
    print("=" * 120)

    # Filter out errors
    valid_results = [r for r in results if not r.error]
    if not valid_results:
        print("\n  No valid results to analyze")
        return

    # Group by batch strategy
    by_strategy = {}
    for r in valid_results:
        key = r.batch_strategy
        if key not in by_strategy:
            by_strategy[key] = []
        by_strategy[key].append(r)

    # Compare strategies
    print("\n📊 Batch Strategy Comparison (best throughput per strategy):")
    print("-" * 80)

    best_per_strategy = {}
    for strategy, rs in by_strategy.items():
        best = max(rs, key=lambda x: x.throughput_embeddings_per_sec)
        best_per_strategy[strategy] = best
        tok = "BT" if best.use_budtiktok else "HF"
        print(f"  {strategy:20}: {best.throughput_embeddings_per_sec:>10,.0f} emb/s "
              f"(c={best.concurrent_requests}, {tok}, P99={best.latency_p99_ms:.1f}ms)")

    # Find overall best
    if best_per_strategy:
        overall_best = max(best_per_strategy.values(), key=lambda x: x.throughput_embeddings_per_sec)
        print(f"\n  🏆 Best: {overall_best.batch_strategy} with {overall_best.throughput_embeddings_per_sec:,.0f} emb/s")

    # BudTikTok vs HuggingFace comparison
    print("\n📊 BudTikTok vs HuggingFace Tokenizer Impact:")
    print("-" * 80)

    budtiktok_results = [r for r in valid_results if r.use_budtiktok]
    huggingface_results = [r for r in valid_results if not r.use_budtiktok]

    if budtiktok_results and huggingface_results:
        # Compare matching configurations
        for hf_r in huggingface_results:
            # Find matching BudTikTok result
            for bt_r in budtiktok_results:
                if (bt_r.batch_strategy == hf_r.batch_strategy and
                    bt_r.max_batch_tokens == hf_r.max_batch_tokens and
                    bt_r.batch_size == hf_r.batch_size and
                    bt_r.concurrent_requests == hf_r.concurrent_requests):

                    speedup = bt_r.throughput_embeddings_per_sec / hf_r.throughput_embeddings_per_sec if hf_r.throughput_embeddings_per_sec > 0 else 0
                    latency_improvement = (hf_r.latency_p99_ms - bt_r.latency_p99_ms) / hf_r.latency_p99_ms * 100 if hf_r.latency_p99_ms > 0 else 0

                    strategy_desc = bt_r.batch_strategy
                    if bt_r.max_batch_tokens:
                        strategy_desc += f"_tok{bt_r.max_batch_tokens//1024}k"

                    print(f"  {strategy_desc:25} (c={bt_r.concurrent_requests}): "
                          f"BT={bt_r.throughput_embeddings_per_sec:>8,.0f} vs HF={hf_r.throughput_embeddings_per_sec:>8,.0f} "
                          f"({speedup:.2f}x speedup, {latency_improvement:+.1f}% P99)")
                    break

        # Summary
        avg_bt_throughput = statistics.mean([r.throughput_embeddings_per_sec for r in budtiktok_results])
        avg_hf_throughput = statistics.mean([r.throughput_embeddings_per_sec for r in huggingface_results]) if huggingface_results else 0
        if avg_hf_throughput > 0:
            overall_speedup = avg_bt_throughput / avg_hf_throughput
            print(f"\n  📈 Average Impact: {overall_speedup:.2f}x throughput improvement with BudTikTok")
    else:
        if not huggingface_results:
            print("  (No HuggingFace results for comparison)")
        if not budtiktok_results:
            print("  (No BudTikTok results for comparison)")

    # GPU utilization analysis
    print("\n📊 GPU Utilization Analysis:")
    print("-" * 80)

    has_gpu_data = False
    for r in valid_results:
        if r.gpu_utilization_mean > 0:
            has_gpu_data = True
            util_status = "🟢 Good" if r.gpu_utilization_mean > 80 else "🟡 Medium" if r.gpu_utilization_mean > 50 else "🔴 Low"
            print(f"  {r.config_name:40}: {r.gpu_utilization_mean:5.1f}% mean, {r.gpu_utilization_max:5.1f}% max ({util_status})")

    if not has_gpu_data:
        print("  (No GPU utilization data collected - pynvml not available or no GPU)")

    # Recommendations
    print("\n💡 Recommendations:")
    print("-" * 80)

    if best_per_strategy:
        # Find best throughput and best latency
        best_throughput = max(valid_results, key=lambda x: x.throughput_embeddings_per_sec)
        best_latency = min(valid_results, key=lambda x: x.latency_p99_ms)

        print(f"  • Maximum Throughput: {best_throughput.config_name}")
        print(f"    → {best_throughput.throughput_embeddings_per_sec:,.0f} emb/s")

        print(f"  • Lowest Latency: {best_latency.config_name}")
        print(f"    → P99={best_latency.latency_p99_ms:.1f}ms")

        # Best efficiency (throughput / concurrent)
        best_efficiency = max(valid_results, key=lambda x: x.throughput_embeddings_per_sec / x.concurrent_requests)
        print(f"  • Best Efficiency: {best_efficiency.config_name}")
        print(f"    → {best_efficiency.throughput_embeddings_per_sec / best_efficiency.concurrent_requests:.0f} emb/s per concurrent request")


async def main():
    parser = argparse.ArgumentParser(description="Batching Strategies Benchmark")
    parser.add_argument(
        "--model",
        default="/home/bud/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        help="Model path"
    )
    parser.add_argument("--quick", action="store_true", help="Quick benchmark with fewer configs")
    parser.add_argument("--num-texts", type=int, default=2000, help="Number of texts to benchmark")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    print("=" * 120)
    print("Batching Strategies Benchmark")
    print("=" * 120)

    # Check environment
    print(f"\n📦 Environment:")
    print(f"  PyTorch: {'✓ ' + torch.__version__ if TORCH_AVAILABLE else '✗'}")
    if TORCH_AVAILABLE:
        print(f"  CUDA: {'✓' if torch.cuda.is_available() else '✗'}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Monitoring: {'✓' if PYNVML_AVAILABLE else '✗'}")

    # Generate test data
    num_texts = 500 if args.quick else args.num_texts
    print(f"\n📝 Generating {num_texts} test texts...")
    texts = generate_texts(num_texts, 10, 100)

    # Define configurations to test
    # Format: (batch_strategy, max_batch_tokens, max_batch_size, concurrent_requests, use_budtiktok)
    if args.quick:
        configs = [
            # BudTikTok configs
            ("auto", None, 32, 1, True),
            ("auto", None, 32, 4, True),
            ("tokens", 16384, 64, 4, True),
            # HuggingFace comparison
            ("auto", None, 32, 4, False),
            ("tokens", 16384, 64, 4, False),
        ]
    else:
        configs = [
            # === BudTikTok Configurations ===
            # Size-based batching
            ("size", None, 32, 1, True),
            ("size", None, 32, 4, True),
            ("size", None, 64, 4, True),
            # Token-budget batching
            ("tokens", 8192, 64, 1, True),
            ("tokens", 8192, 64, 4, True),
            ("tokens", 16384, 64, 1, True),
            ("tokens", 16384, 64, 4, True),
            ("tokens", 16384, 64, 8, True),
            ("tokens", 32768, 128, 4, True),
            # HACC batching
            ("hacc", 16384, 64, 1, True),
            ("hacc", 16384, 64, 4, True),
            ("hacc", 16384, 64, 8, True),
            # Optimized HACC
            ("optimized_hacc", 16384, 64, 4, True),
            ("optimized_hacc_v2", 16384, 64, 4, True),
            # Bucketed priority
            ("bucketed_priority", 16384, 64, 4, True),
            # Auto mode
            ("auto", None, 32, 4, True),
            ("auto", 16384, 64, 4, True),

            # === HuggingFace Comparison (selected configs) ===
            ("size", None, 32, 4, False),
            ("tokens", 16384, 64, 4, False),
            ("hacc", 16384, 64, 4, False),
            ("auto", 16384, 64, 4, False),
        ]

    results = []
    total_configs = len(configs)

    print(f"\n🔬 Running {total_configs} benchmark configurations...")

    for i, (batch_strategy, max_batch_tokens, batch_size, concurrent, use_budtiktok) in enumerate(configs):
        tok_label = "budtiktok" if use_budtiktok else "huggingface"
        config_name = f"{batch_strategy}"
        if max_batch_tokens:
            config_name += f"_tok{max_batch_tokens//1024}k"
        config_name += f"_bs{batch_size}_{tok_label}_c{concurrent}"

        print(f"\n[{i+1}/{total_configs}] {config_name}...", end=" ", flush=True)

        result = await run_benchmark(
            model_path=args.model,
            texts=texts,
            batch_strategy=batch_strategy,
            max_batch_tokens=max_batch_tokens,
            max_batch_size=batch_size,
            concurrent_requests=concurrent,
            use_budtiktok=use_budtiktok,
        )

        if result.error:
            print(f"ERROR: {result.error[:50]}...")
        else:
            print(f"{result.throughput_embeddings_per_sec:,.0f} emb/s, "
                  f"P99={result.latency_p99_ms:.1f}ms, "
                  f"GPU={result.gpu_utilization_mean:.0f}%")

        results.append(result)

        # Clear GPU memory between runs
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    # Print results
    print_results_table(results)
    print_comparison(results)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())
