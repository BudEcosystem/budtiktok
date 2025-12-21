//! Profile Bottleneck Analysis
//!
//! Identifies where time is being spent in the tokenization pipeline.

use anyhow::Result;
use budtiktok_core::pipeline::TokenizerPipeline;
use budtiktok_core::wordpiece_hyper::{HyperConfig, HyperWordPieceTokenizer};
use budtiktok_core::vocab::{SpecialTokens, Vocabulary};
use clap::Parser;
use colored::Colorize;
use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;
use tokenizers::Tokenizer as HfTokenizer;

#[derive(Parser)]
#[command(name = "profile-bottleneck")]
struct Cli {
    #[arg(short, long)]
    tokenizer: String,
}

fn generate_batch(batch_size: usize, avg_words: usize) -> Vec<String> {
    let words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "machine", "learning", "natural", "language", "processing",
    ];
    let mut rng = rand::thread_rng();
    (0..batch_size)
        .map(|_| {
            (0..avg_words)
                .map(|_| words[rng.gen_range(0..words.len())])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("{}", "═".repeat(80).cyan());
    println!("{:^80}", "BOTTLENECK PROFILING".bold().cyan());
    println!("{}", "═".repeat(80).cyan());
    println!();

    let json = std::fs::read_to_string(&cli.tokenizer)?;

    // Load HF tokenizer
    let hf_tokenizer = HfTokenizer::from_file(&cli.tokenizer)
        .map_err(|e| anyhow::anyhow!("HF load failed: {}", e))?;

    // Load pipeline tokenizer
    let pipeline = TokenizerPipeline::from_str(&json)?;

    // Load HyperWordPiece directly (bypasses pipeline overhead)
    let hf_json: serde_json::Value = serde_json::from_str(&json)?;
    let vocab_obj = hf_json["model"]["vocab"].as_object().unwrap();
    let mut vocab_map: HashMap<String, u32> = HashMap::new();
    for (k, v) in vocab_obj {
        vocab_map.insert(k.clone(), v.as_u64().unwrap() as u32);
    }
    let special = SpecialTokens {
        cls_token: Some("[CLS]".to_string()),
        sep_token: Some("[SEP]".to_string()),
        unk_token: Some("[UNK]".to_string()),
        ..Default::default()
    };
    let vocab = Vocabulary::new(vocab_map.into_iter().collect(), special);
    let hyper_config = HyperConfig {
        continuing_subword_prefix: "##".to_string(),
        max_input_chars_per_word: 100,
        unk_token: "[UNK]".to_string(),
        do_lower_case: true,
        strip_accents: true,
        tokenize_chinese_chars: true,
        hash_table_bits: 14,
    };
    let hyper = HyperWordPieceTokenizer::new(vocab, hyper_config);

    let batch_sizes = [1, 8, 32, 128, 512, 1024];
    let iterations = 20;
    let warmup = 5;

    println!("{:<10} | {:>15} | {:>15} | {:>15} | {:>10} | {:>10}",
             "Batch", "HF (µs)", "Pipeline (µs)", "Hyper (µs)", "Pipe/HF", "Hyper/HF");
    println!("{}", "-".repeat(90));

    for &batch_size in &batch_sizes {
        let texts = generate_batch(batch_size, 20);
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        // Warmup
        for _ in 0..warmup {
            let _ = hf_tokenizer.encode_batch(texts.clone(), true);
            let _ = pipeline.encode_batch(&text_refs, true);
            let _ = hyper.encode_batch_with_special(&text_refs);
        }

        // Benchmark HF
        let mut hf_times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = hf_tokenizer.encode_batch(texts.clone(), true);
            hf_times.push(start.elapsed().as_micros() as f64);
        }
        let hf_mean = hf_times.iter().sum::<f64>() / hf_times.len() as f64;

        // Benchmark Pipeline (with overhead)
        let mut pipe_times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = pipeline.encode_batch(&text_refs, true);
            pipe_times.push(start.elapsed().as_micros() as f64);
        }
        let pipe_mean = pipe_times.iter().sum::<f64>() / pipe_times.len() as f64;

        // Benchmark Hyper directly (no pipeline overhead)
        let mut hyper_times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = hyper.encode_batch_with_special(&text_refs);
            hyper_times.push(start.elapsed().as_micros() as f64);
        }
        let hyper_mean = hyper_times.iter().sum::<f64>() / hyper_times.len() as f64;

        let pipe_ratio = hf_mean / pipe_mean;
        let hyper_ratio = hf_mean / hyper_mean;

        let pipe_color = if pipe_ratio >= 1.0 { format!("{:.2}x", pipe_ratio).green() } else { format!("{:.2}x", pipe_ratio).red() };
        let hyper_color = if hyper_ratio >= 1.0 { format!("{:.2}x", hyper_ratio).green() } else { format!("{:.2}x", hyper_ratio).red() };

        println!("{:<10} | {:>15.1} | {:>15.1} | {:>15.1} | {:>10} | {:>10}",
                 batch_size, hf_mean, pipe_mean, hyper_mean, pipe_color, hyper_color);
    }

    println!();
    println!("{}", "═".repeat(80).cyan());
    println!("{:^80}", "ANALYSIS".bold().cyan());
    println!("{}", "═".repeat(80).cyan());
    println!();
    println!("If 'Hyper' is much faster than 'Pipeline', the overhead is in:");
    println!("  - RwLock contention on added_vocabulary");
    println!("  - Encoding struct construction overhead");
    println!("  - Post-processing overhead");
    println!();
    println!("If 'Pipeline' is similar to 'Hyper', the core tokenization is the bottleneck.");
    println!();

    Ok(())
}
