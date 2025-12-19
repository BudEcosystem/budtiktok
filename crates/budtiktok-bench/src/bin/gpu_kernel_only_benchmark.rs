//! GPU Kernel-Only Benchmark
//! Measures GPU tokenization with properly sized buffers (simulates cuDF scenario)

use anyhow::Result;
use std::time::Instant;

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    use budtiktok_gpu::cuda::{CudaContext, is_cuda_available};
    use budtiktok_gpu::{GpuNativeTokenizer, GpuNativeConfig};
    use std::sync::Arc;
    use std::collections::HashMap;

    if !is_cuda_available() {
        println!("CUDA not available");
        return Ok(());
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   GPU BENCHMARK WITH PROPERLY SIZED BUFFERS                  ║");
    println!("║   (Simulates cuDF scenario with minimal transfer overhead)   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load vocabulary
    let tokenizer_path = std::env::args().nth(1)
        .unwrap_or_else(|| "test_data/bert-base-uncased/tokenizer.json".to_string());

    let content = std::fs::read_to_string(&tokenizer_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;
    let vocab_obj = json["model"]["vocab"].as_object().unwrap();

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (token, id) in vocab_obj {
        vocab.insert(token.clone(), id.as_u64().unwrap() as u32);
    }

    let ctx = Arc::new(CudaContext::new(0)?);
    println!("GPU: {}", ctx.device_info().name);
    println!("Vocabulary: {} tokens\n", vocab.len());

    // Reference CPU throughput (BudTikTok-Hyper approximate)
    let cpu_simd_throughput = 250_000_000.0; // 250M tok/s

    // Test configurations: (batch_size, seq_length_chars)
    let configs = [
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 512),
        (1024, 256),
        (2048, 128),
    ];

    println!("┌─────────┬──────────┬──────────┬──────────────┬──────────────┬─────────────┐");
    println!("│  Batch  │ Seq Len  │  Tokens  │   Time/batch │   Tok/sec    │  vs CPU*    │");
    println!("├─────────┼──────────┼──────────┼──────────────┼──────────────┼─────────────┤");

    for (batch_size, seq_len) in configs {
        // Generate test data
        let texts: Vec<String> = (0..batch_size)
            .map(|i| {
                format!("The quick brown fox jumps over the lazy dog number {}. ", i)
                    .repeat(seq_len / 50 + 1)
                    .chars()
                    .take(seq_len)
                    .collect()
            })
            .collect();

        let total_chars: usize = texts.iter().map(|t| t.len()).sum();
        let est_tokens = (total_chars as f64 / 5.0 * 1.3) as usize;
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // Create tokenizer with EXACT buffer sizes (minimal padding)
        let config = GpuNativeConfig {
            max_seq_bytes: seq_len + 64,
            max_tokens_per_seq: seq_len,
            max_batch_size: batch_size + 1,
            do_lower_case: true,
            continuation_prefix: "##".to_string(),
            max_word_chars: 100,
            block_size: 256,
        };

        let mut tokenizer = match GpuNativeTokenizer::new(ctx.clone(), &vocab, config) {
            Ok(t) => t,
            Err(e) => {
                println!("│ {:>7} │ {:>8} │ FAILED: {} │", batch_size, seq_len, e);
                continue;
            }
        };

        // Warmup
        for _ in 0..5 {
            let _ = tokenizer.encode_batch(&text_refs);
        }
        ctx.synchronize()?;

        // Benchmark
        let iterations = 50;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = tokenizer.encode_batch(&text_refs);
        }
        ctx.synchronize()?;

        let total_time = start.elapsed();
        let avg_time_us = total_time.as_micros() as f64 / iterations as f64;
        let tokens_per_sec = est_tokens as f64 * iterations as f64 / total_time.as_secs_f64();
        let vs_cpu = tokens_per_sec / cpu_simd_throughput;

        println!("│ {:>7} │ {:>8} │ {:>8} │ {:>10.0} µs │ {:>12.0} │ {:>10.2}x │",
                 batch_size, seq_len, est_tokens, avg_time_us, tokens_per_sec, vs_cpu);
    }

    println!("└─────────┴──────────┴──────────┴──────────────┴──────────────┴─────────────┘");
    println!("\n* vs CPU = comparison against BudTikTok-Hyper (~250M tok/s)");

    // Large batch detailed analysis
    println!("\n");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  LARGE BATCH DETAILED ANALYSIS (1024 texts × 512 chars)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let batch_size = 1024;
    let seq_len = 512;

    let texts: Vec<String> = (0..batch_size)
        .map(|i| {
            format!("Machine learning natural language processing tokenization benchmark sentence {}. ", i)
                .repeat(seq_len / 80 + 1)
                .chars()
                .take(seq_len)
                .collect()
        })
        .collect();

    let total_chars: usize = texts.iter().map(|t| t.len()).sum();
    let est_tokens = (total_chars as f64 / 5.0 * 1.3) as usize;
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    let config = GpuNativeConfig {
        max_seq_bytes: seq_len + 64,
        max_tokens_per_seq: seq_len,
        max_batch_size: batch_size + 1,
        do_lower_case: true,
        continuation_prefix: "##".to_string(),
        max_word_chars: 100,
        block_size: 256,
    };

    let mut tokenizer = GpuNativeTokenizer::new(ctx.clone(), &vocab, config)?;

    // Warmup
    for _ in 0..10 {
        let _ = tokenizer.encode_batch(&text_refs);
    }
    ctx.synchronize()?;

    // Benchmark with many iterations
    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = tokenizer.encode_batch(&text_refs)?;
    }
    ctx.synchronize()?;

    let total_time = start.elapsed();
    let avg_time_us = total_time.as_micros() as f64 / iterations as f64;
    let throughput = est_tokens as f64 * iterations as f64 / total_time.as_secs_f64();

    println!("Batch size:       {}", batch_size);
    println!("Sequence length:  {} chars", seq_len);
    println!("Total chars:      {}", total_chars);
    println!("Est. tokens:      {} per batch", est_tokens);
    println!();
    println!("Average time:     {:.0} µs per batch", avg_time_us);
    println!("Throughput:       {:.0} tokens/sec", throughput);
    println!("vs CPU SIMD:      {:.2}x", throughput / cpu_simd_throughput);

    // Estimate transfer overhead
    let h2d_bytes = total_chars + (batch_size + 1) * 4 + batch_size * 4; // input + offsets + lengths
    let d2h_bytes = batch_size * seq_len * 4 + batch_size * 4; // output IDs + lengths
    let transfer_bytes = h2d_bytes + d2h_bytes;

    let pcie_bandwidth = 20e9; // 20 GB/s practical
    let transfer_us = (transfer_bytes as f64 / pcie_bandwidth) * 1e6;
    let kernel_only_us = (avg_time_us - transfer_us).max(1.0);
    let kernel_only_throughput = est_tokens as f64 / (kernel_only_us / 1e6);

    println!();
    println!("─── Transfer Analysis ───");
    println!("H2D transfer:     {:.2} MB", h2d_bytes as f64 / 1e6);
    println!("D2H transfer:     {:.2} MB", d2h_bytes as f64 / 1e6);
    println!("Transfer time:    {:.0} µs (@ 20 GB/s PCIe)", transfer_us);
    println!("Transfer %:       {:.1}%", transfer_us / avg_time_us * 100.0);
    println!();
    println!("─── If Data Stayed on GPU (cuDF scenario) ───");
    println!("Kernel-only time: {:.0} µs", kernel_only_us);
    println!("Kernel throughput: {:.0} tokens/sec", kernel_only_throughput);
    println!("vs CPU SIMD:      {:.2}x", kernel_only_throughput / cpu_simd_throughput);

    println!("\n");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  CONCLUSION");
    println!("═══════════════════════════════════════════════════════════════\n");
    println!("cuDF achieves 483x speedup because:");
    println!("  1. Data is ALREADY in GPU memory (GPU DataFrames)");
    println!("  2. Output STAYS in GPU memory for downstream ML");
    println!("  3. Zero H2D/D2H transfers in their benchmark");
    println!("  4. They benchmark at scale (millions of rows)");
    println!();
    println!("When to use GPU tokenization:");
    println!("  ✓ ML training pipeline where data stays on GPU");
    println!("  ✓ Processing millions of texts in GPU DataFrame");
    println!("  ✗ Standalone tokenizer API (CPU SIMD is faster)");
    println!("  ✗ Low-latency inference (transfer overhead dominates)");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("Compile with --features cuda");
}
