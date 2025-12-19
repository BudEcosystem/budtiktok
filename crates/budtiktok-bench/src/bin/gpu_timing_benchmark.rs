//! GPU Timing Breakdown Benchmark
//! Shows where time is spent: H2D transfer, kernels, D2H transfer

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

    println!("=== GPU Timing Breakdown ===\n");

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

    println!("Vocabulary size: {}\n", vocab.len());

    let ctx = Arc::new(CudaContext::new(0)?);
    println!("GPU: {}\n", ctx.device_info().name);

    // Test different configurations
    let configs = [
        ("Small batch", 16, 128),      // 16 texts, ~128 chars each
        ("Medium batch", 64, 512),     // 64 texts, ~512 chars each
        ("Large batch", 128, 2048),    // 128 texts, ~2048 chars each
    ];

    for (name, batch_size, chars_per_text) in configs {
        println!("--- {} (batch={}, chars={}) ---", name, batch_size, chars_per_text);

        // Generate test data
        let texts: Vec<String> = (0..batch_size)
            .map(|i| format!("The quick brown fox jumps over the lazy dog. Sentence number {}. ", i).repeat(chars_per_text / 60 + 1))
            .map(|s| s.chars().take(chars_per_text).collect())
            .collect();

        let total_chars: usize = texts.iter().map(|t| t.len()).sum();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // Create tokenizer with appropriately sized buffers
        let config = GpuNativeConfig {
            max_seq_bytes: chars_per_text + 1024,  // A bit more than needed
            max_tokens_per_seq: chars_per_text / 2,
            max_batch_size: batch_size + 1,
            do_lower_case: true,
            continuation_prefix: "##".to_string(),
            max_word_chars: 100,
            block_size: 256,
        };

        let max_batch = config.max_batch_size;
        let max_seq = config.max_seq_bytes;
        let max_tokens = config.max_tokens_per_seq;

        let mut tokenizer = match GpuNativeTokenizer::new(ctx.clone(), &vocab, config) {
            Ok(t) => t,
            Err(e) => {
                println!("  Failed to create tokenizer: {}\n", e);
                continue;
            }
        };

        // Warmup
        for _ in 0..3 {
            let _ = tokenizer.encode_batch(&text_refs);
        }
        let _ = ctx.synchronize();

        // Benchmark
        let iterations = 10;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = tokenizer.encode_batch(&text_refs);
        }
        let _ = ctx.synchronize();

        let total_time = start.elapsed();
        let avg_time_us = total_time.as_micros() as f64 / iterations as f64;

        // Estimate tokens (rough: 1.3 tokens per word, 5 chars per word)
        let est_tokens = (total_chars as f64 / 5.0 * 1.3) as usize;
        let tokens_per_sec = est_tokens as f64 * iterations as f64 / total_time.as_secs_f64();

        println!("  Total chars:     {}", total_chars);
        println!("  Est. tokens:     {}", est_tokens);
        println!("  Avg time:        {:.0} µs", avg_time_us);
        println!("  Throughput:      {:.0} tokens/sec", tokens_per_sec);

        // Calculate theoretical transfer overhead
        // PCIe 4.0 x16: ~25 GB/s theoretical, ~20 GB/s practical
        let buffer_size = max_batch * max_seq;
        let h2d_bytes = buffer_size + (max_batch + 1) * 4 + max_batch * 4;
        let d2h_bytes = max_batch * max_tokens * 4 + max_batch * 4;
        let transfer_bytes = h2d_bytes + d2h_bytes;

        let pcie_bandwidth = 20e9; // 20 GB/s practical
        let theoretical_transfer_us = (transfer_bytes as f64 / pcie_bandwidth) * 1e6;

        println!("  Buffer sizes:");
        println!("    H2D:           {:.2} MB", h2d_bytes as f64 / 1e6);
        println!("    D2H:           {:.2} MB", d2h_bytes as f64 / 1e6);
        println!("    Total:         {:.2} MB", transfer_bytes as f64 / 1e6);
        println!("  Transfer overhead (theoretical @ 20 GB/s): {:.0} µs", theoretical_transfer_us);
        println!("  Transfer % of total: {:.1}%", theoretical_transfer_us / avg_time_us * 100.0);
        println!();
    }

    println!("\n=== Key Insight ===");
    println!("cuDF achieves 483x speedup because:");
    println!("1. Data is ALREADY in GPU memory (GPU DataFrames)");
    println!("2. Output STAYS in GPU memory for downstream ML");
    println!("3. No H2D or D2H transfers in their benchmark");
    println!("4. They benchmark at massive scale (millions of rows)");
    println!();
    println!("Our benchmark includes full H2D + D2H round-trip,");
    println!("which is the realistic standalone tokenizer use case.");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("Compile with --features cuda");
}
