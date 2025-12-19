//! SPWP (Speculative Parallel WordPiece) Benchmark
//! Compares SPWP+HBFC against CPU SIMD implementation

use anyhow::Result;
use std::time::Instant;
use std::collections::HashMap;
use ahash::AHashMap;

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    use budtiktok_gpu::cuda::{CudaContext, is_cuda_available};
    use budtiktok_gpu::{SpwpTokenizer, SpwpConfig};
    use budtiktok_core::wordpiece_hyper::{HyperWordPieceTokenizer, HyperConfig};
    use budtiktok_core::vocab::{Vocabulary, SpecialTokens};
    use std::sync::Arc;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   SPWP (Speculative Parallel WordPiece) BENCHMARK            ║");
    println!("║   Target: 30x speedup over CPU SIMD                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    if !is_cuda_available() {
        println!("CUDA not available");
        return Ok(());
    }

    // Load vocabulary
    let tokenizer_path = std::env::args().nth(1)
        .unwrap_or_else(|| "test_data/bert-base-uncased/tokenizer.json".to_string());

    println!("Loading vocabulary from: {}", tokenizer_path);
    let content = std::fs::read_to_string(&tokenizer_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;
    let vocab_obj = json["model"]["vocab"].as_object()
        .ok_or_else(|| anyhow::anyhow!("No vocab in tokenizer.json"))?;

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (token, id) in vocab_obj {
        vocab.insert(token.clone(), id.as_u64().unwrap() as u32);
    }
    println!("Vocabulary size: {} tokens\n", vocab.len());

    // Create AHashMap version for budtiktok
    let vocab_map: AHashMap<String, u32> = vocab.iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    // Special tokens
    let special_tokens = SpecialTokens {
        unk_token: Some("[UNK]".to_string()),
        cls_token: Some("[CLS]".to_string()),
        sep_token: Some("[SEP]".to_string()),
        pad_token: Some("[PAD]".to_string()),
        mask_token: Some("[MASK]".to_string()),
        bos_token: None,
        eos_token: None,
    };

    // Initialize GPU
    let ctx = Arc::new(CudaContext::new(0)?);
    println!("GPU: {}", ctx.device_info().name);
    println!("Compute Capability: {}.{}",
             ctx.device_info().compute_capability.0, ctx.device_info().compute_capability.1);
    println!();

    // Initialize SPWP tokenizer
    println!("Initializing SPWP tokenizer...");
    let spwp_config = SpwpConfig {
        max_input_bytes: 1024 * 1024,
        max_output_tokens: 256 * 1024,
        do_lower_case: true,
        continuation_prefix: "##".to_string(),
        block_size: 256,
    };

    let mut spwp = SpwpTokenizer::new(ctx.clone(), &vocab, spwp_config)?;

    // Print bloom filter stats
    let (total_tokens, total_bits, expected_fp) = spwp.get_bloom_stats();
    println!("HBFC Bloom Filter:");
    println!("  Tokens indexed: {}", total_tokens);
    println!("  Total bits: {}", total_bits);
    println!("  Expected FP rate: {:.4}%\n", expected_fp * 100.0);

    // Initialize CPU SIMD tokenizer for comparison
    println!("Initializing CPU SIMD tokenizer (HyperWordPiece)...");
    let vocab_obj = Vocabulary::new(vocab_map.clone(), special_tokens.clone());
    let hyper_config = HyperConfig {
        continuing_subword_prefix: "##".to_string(),
        max_input_chars_per_word: 100,
        unk_token: "[UNK]".to_string(),
        do_lower_case: true,
        strip_accents: false,
        tokenize_chinese_chars: true,
        hash_table_bits: 14,
    };
    let hyper = HyperWordPieceTokenizer::new(vocab_obj, hyper_config);
    println!("CPU tokenizer ready\n");

    // Test configurations
    let test_configs = [
        ("Short text", 64, 100),
        ("Medium text", 256, 50),
        ("Long text", 1024, 20),
        ("Very long text", 4096, 10),
        ("Paragraph", 8192, 5),
    ];

    println!("═══════════════════════════════════════════════════════════════");
    println!("  CORRECTNESS TEST");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Test correctness first
    let test_text = "The quick brown fox jumps over the lazy dog.";
    println!("Test text: \"{}\"", test_text);

    let spwp_tokens = spwp.encode(test_text)?;
    let hyper_tokens = hyper.encode_fast(test_text);

    println!("SPWP tokens: {:?}", spwp_tokens);
    println!("Hyper tokens: {:?}", hyper_tokens);

    let match_status = if spwp_tokens == hyper_tokens {
        "✓ MATCH"
    } else {
        "✗ MISMATCH"
    };
    println!("Status: {}\n", match_status);

    println!("═══════════════════════════════════════════════════════════════");
    println!("  PERFORMANCE BENCHMARK");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("┌─────────────────┬─────────┬────────────────┬────────────────┬─────────────┐");
    println!("│ Configuration   │ Chars   │  CPU SIMD      │  SPWP GPU      │  Speedup    │");
    println!("├─────────────────┼─────────┼────────────────┼────────────────┼─────────────┤");

    let base_sentence = "The quick brown fox jumps over the lazy dog. Machine learning and natural language processing are transforming how we interact with technology. ";

    for (name, char_count, iterations) in test_configs {
        // Generate test text
        let text: String = base_sentence.repeat(char_count / base_sentence.len() + 1)
            .chars()
            .take(char_count)
            .collect();

        // Warmup
        for _ in 0..3 {
            let _ = spwp.encode(&text);
            let _ = hyper.encode_fast(&text);
        }
        ctx.synchronize()?;

        // Benchmark CPU SIMD
        let cpu_start = Instant::now();
        let mut cpu_tokens = 0usize;
        for _ in 0..iterations {
            let tokens = hyper.encode_fast(&text);
            cpu_tokens += tokens.len();
        }
        let cpu_time = cpu_start.elapsed();
        let cpu_us = cpu_time.as_micros() as f64 / iterations as f64;
        let cpu_tok_per_sec = (cpu_tokens as f64 / cpu_time.as_secs_f64()) as u64;

        // Benchmark GPU SPWP
        let gpu_start = Instant::now();
        let mut gpu_tokens = 0usize;
        for _ in 0..iterations {
            let tokens = spwp.encode(&text)?;
            gpu_tokens += tokens.len();
        }
        ctx.synchronize()?;
        let gpu_time = gpu_start.elapsed();
        let gpu_us = gpu_time.as_micros() as f64 / iterations as f64;
        let gpu_tok_per_sec = (gpu_tokens as f64 / gpu_time.as_secs_f64()) as u64;

        let speedup = cpu_us / gpu_us;

        println!("│ {:15} │ {:7} │ {:>10} t/s │ {:>10} t/s │ {:>9.2}x │",
                 name, char_count,
                 format_number(cpu_tok_per_sec),
                 format_number(gpu_tok_per_sec),
                 speedup);
    }

    println!("└─────────────────┴─────────┴────────────────┴────────────────┴─────────────┘");

    // Batch benchmark
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  BATCH BENCHMARK");
    println!("═══════════════════════════════════════════════════════════════\n");

    let batch_sizes = [256, 512, 1024, 2048, 4096];
    let text_len = 256;

    println!("Text length: {} chars per text", text_len);
    println!();
    println!("┌─────────────────┬────────────────┬────────────────┬─────────────┐");
    println!("│ Batch Size      │  CPU SIMD      │  SPWP GPU      │  Speedup    │");
    println!("├─────────────────┼────────────────┼────────────────┼─────────────┤");

    for batch_size in batch_sizes {
        let texts: Vec<String> = (0..batch_size)
            .map(|i| {
                format!("Document {} contains important information about machine learning and AI. ", i)
                    .repeat(text_len / 70 + 1)
                    .chars()
                    .take(text_len)
                    .collect()
            })
            .collect();

        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // Warmup
        for _ in 0..3 {
            let _ = spwp.encode_batch(&text_refs);  // Use batched encoding
            for t in &text_refs {
                let _ = hyper.encode_fast(t);
            }
        }
        ctx.synchronize()?;

        let iterations = 20;

        // CPU benchmark
        let cpu_start = Instant::now();
        let mut cpu_tokens = 0usize;
        for _ in 0..iterations {
            for t in &text_refs {
                cpu_tokens += hyper.encode_fast(t).len();
            }
        }
        let cpu_time = cpu_start.elapsed();
        let cpu_tok_per_sec = (cpu_tokens as f64 / cpu_time.as_secs_f64()) as u64;

        // GPU benchmark - USE BATCHED ENCODING (single kernel for all texts)
        let gpu_start = Instant::now();
        let mut gpu_tokens = 0usize;
        for _ in 0..iterations {
            let results = spwp.encode_batch(&text_refs)?;
            for r in &results {
                gpu_tokens += r.len();
            }
        }
        ctx.synchronize()?;
        let gpu_time = gpu_start.elapsed();
        let gpu_tok_per_sec = (gpu_tokens as f64 / gpu_time.as_secs_f64()) as u64;

        let speedup = gpu_tok_per_sec as f64 / cpu_tok_per_sec as f64;

        println!("│ {:15} │ {:>10} t/s │ {:>10} t/s │ {:>9.2}x │",
                 format!("{} texts", batch_size),
                 format_number(cpu_tok_per_sec),
                 format_number(gpu_tok_per_sec),
                 speedup);
    }

    println!("└─────────────────┴────────────────┴────────────────┴─────────────┘");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Target speedup: 30x");
    println!();
    println!("If speedup < 30x, optimization opportunities:");
    println!("  1. Implement true batch processing (single kernel for all texts)");
    println!("  2. Use warp-cooperative vocabulary broadcast (WCVB)");
    println!("  3. Optimize memory coalescing in bloom filter checks");
    println!("  4. Use texture memory for vocabulary hash table");
    println!("  5. Implement multi-level bloom filter hierarchy");
    println!("  6. Use persistent kernels to avoid launch overhead");

    Ok(())
}

fn format_number(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("Compile with --features cuda");
}
