//! Real-World BPE Benchmark: budtiktok vs HuggingFace
//!
//! This test compares tokenization accuracy and performance on a real-world
//! dataset (1GB+ of OpenWebText) using the actual GPT-2 tokenizer.

use ahash::AHashMap;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::{Duration, Instant};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::tokenizer::Tokenizer as HfTokenizer;

// Import budtiktok encoders
use budtiktok_core::bpe_fast::{FastBpeEncoder, Gpt2ByteEncoderFast};
use budtiktok_core::bpe_linear::OptimizedBpeEncoder;

/// Load GPT-2 vocabulary from vocab.json
fn load_gpt2_vocab(path: &str) -> (AHashMap<String, u32>, HashMap<String, u32>) {
    let content = fs::read_to_string(path).expect("Failed to read vocab.json");
    let vocab: HashMap<String, u32> = serde_json::from_str(&content).expect("Failed to parse vocab.json");

    let vocab_ahash: AHashMap<String, u32> = vocab.iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();

    (vocab_ahash, vocab)
}

/// Load GPT-2 merges from merges.txt
fn load_gpt2_merges(path: &str) -> Vec<(String, String)> {
    let content = fs::read_to_string(path).expect("Failed to read merges.txt");
    let lines: Vec<&str> = content.lines().collect();

    // Skip the first line (version header)
    let merges: Vec<(String, String)> = lines[1..]
        .iter()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                Some((parts[0].to_string(), parts[1].to_string()))
            } else {
                None
            }
        })
        .collect();

    merges
}

/// Load documents from JSONL file
fn load_documents(path: &str, limit: Option<usize>) -> Vec<String> {
    let file = File::open(path).expect("Failed to open JSONL file");
    let reader = BufReader::new(file);

    let mut documents = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        if let Some(max) = limit {
            if i >= max {
                break;
            }
        }

        if let Ok(line) = line {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some(text) = json.get("text").and_then(|t| t.as_str()) {
                    documents.push(text.to_string());
                }
            }
        }
    }

    documents
}

/// Format duration nicely
fn format_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 1.0 {
        format!("{:.2}ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{:.2}s", secs)
    } else {
        format!("{:.2}m", secs / 60.0)
    }
}

#[test]
fn test_realworld_accuracy_and_performance() {
    println!("\n{}", "=".repeat(80));
    println!("REAL-WORLD BPE BENCHMARK: budtiktok vs HuggingFace");
    println!("Dataset: OpenWebText (~1GB)");
    println!("{}\n", "=".repeat(80));

    // Check if data files exist - use workspace root path
    let workspace_root = env!("CARGO_MANIFEST_DIR").replace("/crates/budtiktok-core", "");
    let vocab_path = format!("{}/benchmark_data/gpt2/vocab.json", workspace_root);
    let merges_path = format!("{}/benchmark_data/gpt2/merges.txt", workspace_root);
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", workspace_root);

    if !Path::new(&vocab_path).exists() || !Path::new(&merges_path).exists() {
        println!("GPT-2 tokenizer files not found at: {}", vocab_path);
        println!("Please download them first:");
        println!("  python3 -c \"from transformers import GPT2Tokenizer; t = GPT2Tokenizer.from_pretrained('gpt2'); t.save_pretrained('benchmark_data/gpt2')\"");
        return;
    }

    if !Path::new(&data_path).exists() {
        println!("OpenWebText data not found at: {}", data_path);
        println!("Please download the dataset first.");
        return;
    }

    // Load GPT-2 tokenizer components
    println!("Loading GPT-2 tokenizer...");
    let (vocab_ahash, vocab_std) = load_gpt2_vocab(&vocab_path);
    let merges = load_gpt2_merges(&merges_path);

    println!("  Vocabulary size: {}", vocab_ahash.len());
    println!("  Merge rules: {}", merges.len());

    // Create HuggingFace tokenizer
    println!("\nCreating HuggingFace BPE tokenizer...");
    let hf_bpe = BPE::builder()
        .vocab_and_merges(vocab_std.clone(), merges.clone())
        .unk_token("<|endoftext|>".to_string())
        .build()
        .expect("Failed to build HF BPE");

    let mut hf_tokenizer = HfTokenizer::new(hf_bpe);
    hf_tokenizer.with_pre_tokenizer(PreTokenizerWrapper::ByteLevel(
        ByteLevel::new(false, true, true)  // add_prefix_space=false, trim_offsets=true, use_regex=true (GPT-2 pattern)
    ));

    // Create budtiktok OptimizedBpeEncoder
    println!("Creating budtiktok OptimizedBpeEncoder...");
    let optimized_encoder = OptimizedBpeEncoder::new(
        vocab_ahash.clone(),
        merges.clone(),
        "<|endoftext|>"
    );

    // Also create FastBpeEncoder for comparison
    println!("Creating budtiktok FastBpeEncoder (fancy_regex baseline)...");
    let fast_encoder = FastBpeEncoder::new(
        vocab_ahash.clone(),
        merges.clone(),
        "<|endoftext|>"
    );

    // Load documents
    println!("\nLoading documents...");
    let documents = load_documents(&data_path, None);  // Load all documents
    let total_docs = documents.len();
    let total_bytes: usize = documents.iter().map(|d| d.len()).sum();

    println!("  Total documents: {}", total_docs);
    println!("  Total size: {:.2} GB", total_bytes as f64 / (1024.0 * 1024.0 * 1024.0));

    // =========================================================================
    // ACCURACY TEST: Compare outputs for every document
    // =========================================================================
    println!("\n{}", "-".repeat(80));
    println!("ACCURACY TEST: Comparing tokenization output for ALL documents");
    println!("{}", "-".repeat(80));

    let mut accuracy_matches = 0;
    let mut accuracy_mismatches = 0;
    let mut mismatch_examples: Vec<(usize, String, Vec<u32>, Vec<u32>)> = Vec::new();

    let accuracy_start = Instant::now();

    for (i, doc) in documents.iter().enumerate() {
        // Tokenize with HuggingFace
        let hf_result = hf_tokenizer.encode(doc.as_str(), false);
        let hf_ids: Vec<u32> = match hf_result {
            Ok(encoding) => encoding.get_ids().to_vec(),
            Err(_) => continue,  // Skip documents that fail to encode
        };

        // Tokenize with budtiktok OptimizedBpeEncoder
        let optimized_ids = optimized_encoder.encode(doc);

        // Compare
        if hf_ids == optimized_ids {
            accuracy_matches += 1;
        } else {
            accuracy_mismatches += 1;
            if mismatch_examples.len() < 10 {
                mismatch_examples.push((
                    i,
                    if doc.len() > 100 { format!("{}...", &doc[..100]) } else { doc.clone() },
                    hf_ids,
                    optimized_ids
                ));
            }
        }

        // Progress update
        if (i + 1) % 50000 == 0 || i + 1 == total_docs {
            let elapsed = accuracy_start.elapsed();
            let rate = (i + 1) as f64 / elapsed.as_secs_f64();
            println!("  Progress: {}/{} documents ({:.1}%), {:.0} docs/s, matches: {}, mismatches: {}",
                i + 1, total_docs,
                (i + 1) as f64 / total_docs as f64 * 100.0,
                rate,
                accuracy_matches,
                accuracy_mismatches
            );
        }
    }

    let accuracy_elapsed = accuracy_start.elapsed();
    let accuracy_pct = accuracy_matches as f64 / (accuracy_matches + accuracy_mismatches) as f64 * 100.0;

    println!("\nAccuracy Results:");
    println!("  Matches: {} ({:.4}%)", accuracy_matches, accuracy_pct);
    println!("  Mismatches: {}", accuracy_mismatches);
    println!("  Time: {}", format_duration(accuracy_elapsed));

    if !mismatch_examples.is_empty() {
        println!("\nMismatch Examples (first {}):", mismatch_examples.len());
        for (i, (doc_idx, text, hf_ids, opt_ids)) in mismatch_examples.iter().enumerate() {
            println!("\n  Example {}:", i + 1);
            println!("    Doc index: {}", doc_idx);
            println!("    Text: \"{}\"", text);
            println!("    HF tokens ({} ids): {:?}", hf_ids.len(), &hf_ids[..hf_ids.len().min(20)]);
            println!("    Optimized tokens ({} ids): {:?}", opt_ids.len(), &opt_ids[..opt_ids.len().min(20)]);
        }
    }

    // =========================================================================
    // PERFORMANCE TEST: Time full dataset tokenization
    // =========================================================================
    println!("\n{}", "-".repeat(80));
    println!("PERFORMANCE TEST: Tokenizing entire dataset");
    println!("{}", "-".repeat(80));

    // Warm-up runs
    println!("\nWarm-up runs...");
    for doc in documents.iter().take(1000) {
        let _ = hf_tokenizer.encode(doc.as_str(), false);
        let _ = optimized_encoder.encode(doc);
        let _ = fast_encoder.encode(doc);
    }

    // HuggingFace benchmark
    println!("\nBenchmarking HuggingFace tokenizer...");
    let hf_start = Instant::now();
    let mut hf_total_tokens = 0usize;
    for doc in &documents {
        if let Ok(encoding) = hf_tokenizer.encode(doc.as_str(), false) {
            hf_total_tokens += encoding.get_ids().len();
        }
    }
    let hf_elapsed = hf_start.elapsed();
    let hf_throughput = total_bytes as f64 / hf_elapsed.as_secs_f64() / (1024.0 * 1024.0);

    println!("  HuggingFace:");
    println!("    Time: {}", format_duration(hf_elapsed));
    println!("    Throughput: {:.2} MB/s", hf_throughput);
    println!("    Total tokens: {}", hf_total_tokens);

    // budtiktok OptimizedBpeEncoder benchmark
    println!("\nBenchmarking budtiktok OptimizedBpeEncoder...");
    let opt_start = Instant::now();
    let mut opt_total_tokens = 0usize;
    for doc in &documents {
        let tokens = optimized_encoder.encode(doc);
        opt_total_tokens += tokens.len();
    }
    let opt_elapsed = opt_start.elapsed();
    let opt_throughput = total_bytes as f64 / opt_elapsed.as_secs_f64() / (1024.0 * 1024.0);

    println!("  OptimizedBpeEncoder (hand-rolled GPT-2 pre-tokenizer):");
    println!("    Time: {}", format_duration(opt_elapsed));
    println!("    Throughput: {:.2} MB/s", opt_throughput);
    println!("    Total tokens: {}", opt_total_tokens);
    println!("    Speedup vs HF: {:.2}x", hf_elapsed.as_secs_f64() / opt_elapsed.as_secs_f64());

    // budtiktok FastBpeEncoder benchmark (fancy_regex baseline)
    println!("\nBenchmarking budtiktok FastBpeEncoder (fancy_regex)...");
    let fast_start = Instant::now();
    let mut fast_total_tokens = 0usize;
    for doc in &documents {
        let tokens = fast_encoder.encode(doc);
        fast_total_tokens += tokens.len();
    }
    let fast_elapsed = fast_start.elapsed();
    let fast_throughput = total_bytes as f64 / fast_elapsed.as_secs_f64() / (1024.0 * 1024.0);

    println!("  FastBpeEncoder (fancy_regex):");
    println!("    Time: {}", format_duration(fast_elapsed));
    println!("    Throughput: {:.2} MB/s", fast_throughput);
    println!("    Total tokens: {}", fast_total_tokens);
    println!("    Speedup vs HF: {:.2}x", hf_elapsed.as_secs_f64() / fast_elapsed.as_secs_f64());

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("\n{}", "=".repeat(80));
    println!("SUMMARY");
    println!("{}", "=".repeat(80));

    println!("\nDataset: {:.2} GB, {} documents",
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0), total_docs);

    println!("\nAccuracy:");
    println!("  OptimizedBpeEncoder vs HuggingFace: {:.4}% match ({}/{} documents)",
        accuracy_pct, accuracy_matches, accuracy_matches + accuracy_mismatches);

    println!("\nPerformance:");
    println!("  {:<35} {:>12} {:>12} {:>10}",
        "Encoder", "Time", "Throughput", "Speedup");
    println!("  {:-<35} {:->12} {:->12} {:->10}", "", "", "", "");
    println!("  {:<35} {:>12} {:>10.2} MB/s {:>9}",
        "HuggingFace (baseline)", format_duration(hf_elapsed), hf_throughput, "1.0x");
    println!("  {:<35} {:>12} {:>10.2} MB/s {:>8.2}x",
        "FastBpeEncoder (fancy_regex)", format_duration(fast_elapsed), fast_throughput,
        hf_elapsed.as_secs_f64() / fast_elapsed.as_secs_f64());
    println!("  {:<35} {:>12} {:>10.2} MB/s {:>8.2}x",
        "OptimizedBpeEncoder (hand-rolled)", format_duration(opt_elapsed), opt_throughput,
        hf_elapsed.as_secs_f64() / opt_elapsed.as_secs_f64());

    // Assert 100% accuracy
    assert_eq!(
        accuracy_mismatches, 0,
        "ACCURACY TEST FAILED: {} documents had mismatched tokenization",
        accuracy_mismatches
    );
}

#[test]
fn test_realworld_accuracy_sample() {
    // Quick accuracy test on a smaller sample for faster iteration
    println!("\n{}", "=".repeat(80));
    println!("QUICK ACCURACY TEST: Sample of 1000 documents");
    println!("{}\n", "=".repeat(80));

    let workspace_root = env!("CARGO_MANIFEST_DIR").replace("/crates/budtiktok-core", "");
    let vocab_path = format!("{}/benchmark_data/gpt2/vocab.json", workspace_root);
    let merges_path = format!("{}/benchmark_data/gpt2/merges.txt", workspace_root);
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", workspace_root);

    if !Path::new(&vocab_path).exists() || !Path::new(&data_path).exists() {
        println!("Required files not found at: {}", vocab_path);
        println!("Skipping test.");
        return;
    }

    let (vocab_ahash, vocab_std) = load_gpt2_vocab(&vocab_path);
    let merges = load_gpt2_merges(&merges_path);

    let hf_bpe = BPE::builder()
        .vocab_and_merges(vocab_std.clone(), merges.clone())
        .unk_token("<|endoftext|>".to_string())
        .build()
        .expect("Failed to build HF BPE");

    let mut hf_tokenizer = HfTokenizer::new(hf_bpe);
    hf_tokenizer.with_pre_tokenizer(PreTokenizerWrapper::ByteLevel(
        ByteLevel::new(false, true, true)  // use_regex=true for GPT-2 pattern
    ));

    let optimized_encoder = OptimizedBpeEncoder::new(
        vocab_ahash,
        merges,
        "<|endoftext|>"
    );

    let documents = load_documents(&data_path, Some(1000));

    let mut matches = 0;
    let mut mismatches = 0;

    for (i, doc) in documents.iter().enumerate() {
        let hf_ids: Vec<u32> = match hf_tokenizer.encode(doc.as_str(), false) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(_) => continue,
        };

        let opt_ids = optimized_encoder.encode(doc);

        if hf_ids == opt_ids {
            matches += 1;
        } else {
            mismatches += 1;
            if mismatches <= 3 {
                println!("\nMismatch at doc {}:", i);
                println!("  Text (first 200 chars): {:?}", &doc[..doc.len().min(200)]);
                println!("  HF: {} tokens, first 10: {:?}", hf_ids.len(), &hf_ids[..hf_ids.len().min(10)]);
                println!("  Opt: {} tokens, first 10: {:?}", opt_ids.len(), &opt_ids[..opt_ids.len().min(10)]);
            }
        }
    }

    println!("\nResults: {}/{} matches ({:.2}%)",
        matches, matches + mismatches,
        matches as f64 / (matches + mismatches) as f64 * 100.0);

    // Debug: Find exact divergence in first mismatch
    if mismatches > 0 {
        // Re-tokenize first mismatched document for detailed debugging
        let first_mismatch_idx = documents.iter().enumerate()
            .find(|(_, doc)| {
                let hf_ids: Vec<u32> = hf_tokenizer.encode(doc.as_str(), false).unwrap().get_ids().to_vec();
                let opt_ids = optimized_encoder.encode(doc);
                hf_ids != opt_ids
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        println!("\nDebugging first mismatch at doc index: {}", first_mismatch_idx);
        let doc = &documents[first_mismatch_idx];
        let hf_ids: Vec<u32> = hf_tokenizer.encode(doc.as_str(), false).unwrap().get_ids().to_vec();
        let opt_ids = optimized_encoder.encode(doc);

        // Build reverse vocab
        let vocab_r: std::collections::HashMap<u32, String> = vocab_std.iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        // Find first divergence
        let min_len = hf_ids.len().min(opt_ids.len());
        for i in 0..min_len {
            if hf_ids[i] != opt_ids[i] {
                println!("\n=== FIRST DIVERGENCE at token {} ===", i);
                println!("  HF token: {} ({})", hf_ids[i], vocab_r.get(&hf_ids[i]).unwrap_or(&"?".to_string()));
                println!("  Opt token: {} ({})", opt_ids[i], vocab_r.get(&opt_ids[i]).unwrap_or(&"?".to_string()));

                println!("\nHF tokens around divergence:");
                for j in i.saturating_sub(5)..i+10.min(hf_ids.len()) {
                    let mark = if j == i { " <-- DIVERGES" } else { "" };
                    println!("  {}: {} ({:?}){}", j, hf_ids[j], vocab_r.get(&hf_ids[j]).unwrap_or(&"?".to_string()), mark);
                }

                println!("\nOpt tokens around divergence:");
                for j in i.saturating_sub(5)..i+10.min(opt_ids.len()) {
                    let mark = if j == i { " <-- DIVERGES" } else { "" };
                    println!("  {}: {} ({:?}){}", j, opt_ids[j], vocab_r.get(&opt_ids[j]).unwrap_or(&"?".to_string()), mark);
                }
                break;
            }
        }

        if hf_ids.len() != opt_ids.len() && hf_ids[..min_len] == opt_ids[..min_len] {
            println!("\n=== No divergence in common prefix ===");
            println!("Difference is at end. HF has {} tokens, Opt has {} tokens", hf_ids.len(), opt_ids.len());
            println!("Last 10 HF tokens:");
            for j in hf_ids.len().saturating_sub(10)..hf_ids.len() {
                println!("  {}: {} ({:?})", j, hf_ids[j], vocab_r.get(&hf_ids[j]).unwrap_or(&"?".to_string()));
            }
            println!("Last 10 Opt tokens:");
            for j in opt_ids.len().saturating_sub(10)..opt_ids.len() {
                println!("  {}: {} ({:?})", j, opt_ids[j], vocab_r.get(&opt_ids[j]).unwrap_or(&"?".to_string()));
            }
        }
    }

    assert_eq!(mismatches, 0, "Accuracy test failed with {} mismatches", mismatches);
}

#[test]
fn test_unicode_edge_case() {
    let workspace_root = env!("CARGO_MANIFEST_DIR").replace("/crates/budtiktok-core", "");
    let vocab_path = format!("{}/benchmark_data/gpt2/vocab.json", workspace_root);
    let merges_path = format!("{}/benchmark_data/gpt2/merges.txt", workspace_root);

    let (vocab_ahash, vocab_std) = load_gpt2_vocab(&vocab_path);
    let merges = load_gpt2_merges(&merges_path);
    let vocab_r: std::collections::HashMap<u32, String> = vocab_std.iter()
        .map(|(k, v)| (*v, k.clone()))
        .collect();

    let hf_bpe = BPE::builder()
        .vocab_and_merges(vocab_std.clone(), merges.clone())
        .unk_token("<|endoftext|>".to_string())
        .build()
        .expect("Failed to build HF BPE");

    let mut hf_tokenizer = HfTokenizer::new(hf_bpe);
    hf_tokenizer.with_pre_tokenizer(PreTokenizerWrapper::ByteLevel(
        ByteLevel::new(false, true, true)
    ));

    let optimized = OptimizedBpeEncoder::new(vocab_ahash, merges, "<|endoftext|>");

    let test_cases = vec![
        "ღvvtೋ",
        "Glaღdf ღvvtೋ",
        " ღvvtೋ",  // With leading space
    ];

    for test in test_cases {
        println!("\n=== Testing: {:?} ===", test);
        
        let hf_ids: Vec<u32> = hf_tokenizer.encode(test, false).unwrap().get_ids().to_vec();
        let opt_ids = optimized.encode(test);
        
        println!("HF tokens ({}):", hf_ids.len());
        for id in &hf_ids {
            println!("  {} ({:?})", id, vocab_r.get(id).unwrap_or(&"?".to_string()));
        }
        
        println!("Opt tokens ({}):", opt_ids.len());
        for id in &opt_ids {
            println!("  {} ({:?})", id, vocab_r.get(id).unwrap_or(&"?".to_string()));
        }
        
        if hf_ids == opt_ids {
            println!("MATCH!");
        } else {
            println!("MISMATCH - finding divergence...");
            for i in 0..hf_ids.len().min(opt_ids.len()) {
                if hf_ids[i] != opt_ids[i] {
                    println!("  Diverges at position {}: HF {} vs Opt {}", i, hf_ids[i], opt_ids[i]);
                    break;
                }
            }
        }
    }
}

#[test]
fn test_doc_147863() {
    let workspace_root = env!("CARGO_MANIFEST_DIR").replace("/crates/budtiktok-core", "");
    let vocab_path = format!("{}/benchmark_data/gpt2/vocab.json", workspace_root);
    let merges_path = format!("{}/benchmark_data/gpt2/merges.txt", workspace_root);
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", workspace_root);

    let (vocab_ahash, vocab_std) = load_gpt2_vocab(&vocab_path);
    let merges = load_gpt2_merges(&merges_path);
    let vocab_r: std::collections::HashMap<u32, String> = vocab_std.iter()
        .map(|(k, v)| (*v, k.clone()))
        .collect();

    let documents = load_documents(&data_path, Some(147864));
    let doc = &documents[147863];

    let hf_bpe = BPE::builder()
        .vocab_and_merges(vocab_std.clone(), merges.clone())
        .unk_token("<|endoftext|>".to_string())
        .build()
        .expect("Failed to build HF BPE");

    let mut hf_tokenizer = HfTokenizer::new(hf_bpe);
    hf_tokenizer.with_pre_tokenizer(PreTokenizerWrapper::ByteLevel(
        ByteLevel::new(false, true, true)
    ));

    let optimized = OptimizedBpeEncoder::new(vocab_ahash, merges, "<|endoftext|>");

    let hf_ids: Vec<u32> = hf_tokenizer.encode(doc.as_str(), false).unwrap().get_ids().to_vec();
    let opt_ids = optimized.encode(doc);

    println!("HF tokens: {}, Opt tokens: {}", hf_ids.len(), opt_ids.len());

    // Find first divergence
    for i in 0..hf_ids.len().min(opt_ids.len()) {
        if hf_ids[i] != opt_ids[i] {
            println!("\nFirst divergence at token {}:", i);
            println!("  HF: {} ({:?})", hf_ids[i], vocab_r.get(&hf_ids[i]));
            println!("  Opt: {} ({:?})", opt_ids[i], vocab_r.get(&opt_ids[i]));

            // Show context
            println!("\nHF tokens [{}..{}]:", i.saturating_sub(5), i+10.min(hf_ids.len()));
            for j in i.saturating_sub(5)..i+10.min(hf_ids.len()) {
                let mark = if j == i { " <-- DIVERGES" } else { "" };
                println!("  {}: {} ({:?}){}", j, hf_ids[j], vocab_r.get(&hf_ids[j]), mark);
            }

            println!("\nOpt tokens [{}..{}]:", i.saturating_sub(5), i+10.min(opt_ids.len()));
            for j in i.saturating_sub(5)..i+10.min(opt_ids.len()) {
                let mark = if j == i { " <-- DIVERGES" } else { "" };
                println!("  {}: {} ({:?}){}", j, opt_ids[j], vocab_r.get(&opt_ids[j]), mark);
            }
            break;
        }
    }

    if hf_ids.len() != opt_ids.len() && hf_ids[..hf_ids.len().min(opt_ids.len())] == opt_ids[..hf_ids.len().min(opt_ids.len())] {
        println!("\nNo divergence in common tokens. Length difference.");
        println!("HF last 10:");
        for j in hf_ids.len().saturating_sub(10)..hf_ids.len() {
            println!("  {}: {} ({:?})", j, hf_ids[j], vocab_r.get(&hf_ids[j]));
        }
        println!("Opt last 10:");
        for j in opt_ids.len().saturating_sub(10)..opt_ids.len() {
            println!("  {}: {} ({:?})", j, opt_ids[j], vocab_r.get(&opt_ids[j]));
        }
    }
}

#[test]
fn test_kannada_apostrophe() {
    let workspace_root = env!("CARGO_MANIFEST_DIR").replace("/crates/budtiktok-core", "");
    let vocab_path = format!("{}/benchmark_data/gpt2/vocab.json", workspace_root);
    let merges_path = format!("{}/benchmark_data/gpt2/merges.txt", workspace_root);

    let (vocab_ahash, vocab_std) = load_gpt2_vocab(&vocab_path);
    let merges = load_gpt2_merges(&merges_path);
    let vocab_r: std::collections::HashMap<u32, String> = vocab_std.iter()
        .map(|(k, v)| (*v, k.clone()))
        .collect();

    let hf_bpe = BPE::builder()
        .vocab_and_merges(vocab_std.clone(), merges.clone())
        .unk_token("<|endoftext|>".to_string())
        .build()
        .expect("Failed to build HF BPE");

    let mut hf_tokenizer = HfTokenizer::new(hf_bpe);
    hf_tokenizer.with_pre_tokenizer(PreTokenizerWrapper::ByteLevel(
        ByteLevel::new(false, true, true)
    ));

    let optimized = OptimizedBpeEncoder::new(vocab_ahash, merges, "<|endoftext|>");

    // Test cases
    let test_cases = vec![
        "ೋ's",           // Kannada + 's
        "x's",            // ASCII + 's (should be contraction)
        "ღ's",           // Georgian + 's
        "日's",           // CJK + 's
        "test's",         // Normal word + 's (should be contraction)
    ];

    for test in test_cases {
        println!("\n=== Testing: {:?} ===", test);
        
        let hf_ids: Vec<u32> = hf_tokenizer.encode(test, false).unwrap().get_ids().to_vec();
        let opt_ids = optimized.encode(test);
        
        println!("HF tokens ({}):", hf_ids.len());
        for id in &hf_ids {
            println!("  {} ({:?})", id, vocab_r.get(id));
        }
        
        println!("Opt tokens ({}):", opt_ids.len());
        for id in &opt_ids {
            println!("  {} ({:?})", id, vocab_r.get(id));
        }
        
        if hf_ids == opt_ids {
            println!("MATCH!");
        } else {
            println!("MISMATCH!");
        }
    }
}

#[test]
fn test_fast_encoder_accuracy() {
    let workspace_root = env!("CARGO_MANIFEST_DIR").replace("/crates/budtiktok-core", "");
    let vocab_path = format!("{}/benchmark_data/gpt2/vocab.json", workspace_root);
    let merges_path = format!("{}/benchmark_data/gpt2/merges.txt", workspace_root);
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", workspace_root);

    let (vocab_ahash, vocab_std) = load_gpt2_vocab(&vocab_path);
    let merges = load_gpt2_merges(&merges_path);

    let hf_bpe = BPE::builder()
        .vocab_and_merges(vocab_std.clone(), merges.clone())
        .unk_token("<|endoftext|>".to_string())
        .build()
        .expect("Failed to build HF BPE");

    let mut hf_tokenizer = HfTokenizer::new(hf_bpe);
    hf_tokenizer.with_pre_tokenizer(PreTokenizerWrapper::ByteLevel(
        ByteLevel::new(false, true, true)
    ));

    let fast_encoder = FastBpeEncoder::new(vocab_ahash, merges, "<|endoftext|>");

    let documents = load_documents(&data_path, Some(1000));

    let mut matches = 0;
    let mut mismatches = 0;
    let mut total_hf_tokens = 0usize;
    let mut total_fast_tokens = 0usize;

    for (i, doc) in documents.iter().enumerate() {
        let hf_ids: Vec<u32> = match hf_tokenizer.encode(doc.as_str(), false) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(_) => continue,
        };
        let fast_ids = fast_encoder.encode(doc);

        total_hf_tokens += hf_ids.len();
        total_fast_tokens += fast_ids.len();

        if hf_ids == fast_ids {
            matches += 1;
        } else {
            mismatches += 1;
            if mismatches <= 3 {
                println!("\nFastBpeEncoder mismatch at doc {}:", i);
                println!("  HF tokens: {}, Fast tokens: {}", hf_ids.len(), fast_ids.len());
                println!("  Text (first 100): {:?}", &doc[..doc.len().min(100)]);
            }
        }
    }

    println!("\n=== FastBpeEncoder Accuracy Test ===");
    println!("Documents: {} matches, {} mismatches", matches, mismatches);
    println!("Accuracy: {:.2}%", matches as f64 / (matches + mismatches) as f64 * 100.0);
    println!("Total HF tokens: {}", total_hf_tokens);
    println!("Total Fast tokens: {}", total_fast_tokens);
    println!("Token difference: {} ({:.4}%)", 
        total_fast_tokens as i64 - total_hf_tokens as i64,
        (total_fast_tokens as f64 - total_hf_tokens as f64) / total_hf_tokens as f64 * 100.0);
}

#[test]
fn test_pretokenizer_completeness() {
    use budtiktok_core::bpe_linear::Gpt2PreTokenizer;
    
    let workspace_root = env!("CARGO_MANIFEST_DIR").replace("/crates/budtiktok-core", "");
    let vocab_path = format!("{}/benchmark_data/gpt2/vocab.json", workspace_root);
    let merges_path = format!("{}/benchmark_data/gpt2/merges.txt", workspace_root);

    let (_, vocab_std) = load_gpt2_vocab(&vocab_path);
    let merges = load_gpt2_merges(&merges_path);

    let hf_bpe = BPE::builder()
        .vocab_and_merges(vocab_std.clone(), merges.clone())
        .unk_token("<|endoftext|>".to_string())
        .build()
        .expect("Failed to build HF BPE");

    let mut hf_tokenizer = HfTokenizer::new(hf_bpe);
    hf_tokenizer.with_pre_tokenizer(PreTokenizerWrapper::ByteLevel(
        ByteLevel::new(false, true, true)
    ));

    // GPT-2 pattern: '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    
    let test_cases = vec![
        // Pattern 1: Contractions '(?:[sdmt]|ll|ve|re)
        ("don't", "Contraction 't"),
        ("I'm", "Contraction 'm"),
        ("he's", "Contraction 's"),
        ("we'd", "Contraction 'd"),
        ("they'll", "Contraction 'll"),
        ("we've", "Contraction 've"),
        ("you're", "Contraction 're"),
        ("DON'T", "Uppercase - should NOT match contraction"),
        ("'Strange", "Apostrophe before uppercase"),
        
        // Pattern 2: ` ?\p{L}+` - optional space + letters
        ("hello", "Simple word"),
        (" hello", "Space + word"),
        ("  hello", "Double space + word"),
        ("日本語", "CJK characters (letters)"),
        (" 日本語", "Space + CJK"),
        ("café", "Accented letters"),
        
        // Pattern 3: ` ?\p{N}+` - optional space + numbers
        ("123", "Numbers"),
        (" 456", "Space + numbers"),
        ("①②③", "Circled numbers (Unicode numbers)"),
        
        // Pattern 4: ` ?[^\s\p{L}\p{N}]+` - optional space + other
        ("!!!", "Punctuation"),
        (" ---", "Space + punctuation"),
        ("@#$%", "Symbols"),
        (" @@@", "Space + symbols"),
        
        // Pattern 5 & 6: `\s+(?!\S)|\s+` - whitespace
        ("a\tb", "Tab between words"),
        ("a\nb", "Newline between words"),
        ("a  b", "Double space between words"),
        ("hello   ", "Trailing spaces"),
        ("\n\n", "Multiple newlines"),
        ("\t\t", "Multiple tabs"),
        
        // Edge cases
        ("", "Empty string"),
        (" ", "Single space"),
        ("  ", "Double space"),
        ("\n", "Single newline"),
        ("a", "Single letter"),
        ("1", "Single digit"),
        (".", "Single punctuation"),
        
        // Mixed content
        ("Hello, world!", "Mixed sentence"),
        ("I'm 123 years old.", "Mixed with contractions"),
        ("Price: $99.99", "Symbols and numbers"),
        ("fn main() { println!(\"Hello\"); }", "Code-like"),
        
        // Unicode edge cases
        ("ೋ's", "Kannada mark + apostrophe (known edge case)"),
        ("ღ's", "Georgian letter + apostrophe"),
        ("测试test", "Mixed CJK and ASCII"),
        (" \u{00A0}test", "Non-breaking space"),
        
        // Whitespace varieties
        ("a\r\nb", "CRLF"),
        ("a\rb", "CR only"),
        ("test\u{000B}test", "Vertical tab"),
        ("test\u{000C}test", "Form feed"),
    ];

    println!("\n=== GPT-2 Pre-tokenizer Completeness Test ===\n");
    
    let mut all_passed = true;
    let mut passed = 0;
    let mut failed = 0;

    for (text, description) in &test_cases {
        let hf_enc = hf_tokenizer.encode(*text, false);
        let hf_ids: Vec<u32> = match hf_enc {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(e) => {
                println!("SKIP: {} - HF encoding error: {:?}", description, e);
                continue;
            }
        };

        // Use OptimizedBpeEncoder which uses our pre-tokenizer
        let vocab_ahash: ahash::AHashMap<String, u32> = vocab_std.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let opt = OptimizedBpeEncoder::new(vocab_ahash, merges.clone(), "<|endoftext|>");
        let opt_ids = opt.encode(text);

        if hf_ids == opt_ids {
            passed += 1;
            println!("PASS: {} - {:?}", description, text);
        } else {
            failed += 1;
            all_passed = false;
            println!("FAIL: {} - {:?}", description, text);
            println!("      HF:  {:?} ({} tokens)", &hf_ids[..hf_ids.len().min(10)], hf_ids.len());
            println!("      Opt: {:?} ({} tokens)", &opt_ids[..opt_ids.len().min(10)], opt_ids.len());
        }
    }

    println!("\n=== Summary ===");
    println!("Passed: {}/{}", passed, passed + failed);
    println!("Failed: {}", failed);

    assert!(all_passed, "Some pre-tokenizer tests failed");
}

#[test]
fn test_newline_edge_case() {
    let workspace_root = env!("CARGO_MANIFEST_DIR").replace("/crates/budtiktok-core", "");
    let vocab_path = format!("{}/benchmark_data/gpt2/vocab.json", workspace_root);
    let merges_path = format!("{}/benchmark_data/gpt2/merges.txt", workspace_root);

    let (vocab_ahash, vocab_std) = load_gpt2_vocab(&vocab_path);
    let merges = load_gpt2_merges(&merges_path);
    
    let vocab_r: std::collections::HashMap<u32, String> = vocab_std.iter()
        .map(|(k, v)| (*v, k.clone()))
        .collect();

    let hf_bpe = BPE::builder()
        .vocab_and_merges(vocab_std.clone(), merges.clone())
        .unk_token("<|endoftext|>".to_string())
        .build()
        .expect("Failed to build HF BPE");

    let mut hf_tokenizer = HfTokenizer::new(hf_bpe);
    hf_tokenizer.with_pre_tokenizer(PreTokenizerWrapper::ByteLevel(
        ByteLevel::new(false, true, true)
    ));

    let opt = OptimizedBpeEncoder::new(vocab_ahash, merges, "<|endoftext|>");

    let test_cases = vec![
        "\n\n",
        "\n\n\n",
        "hello\n\nworld",
        "test\n\n\n\ntest",
        "\t\t",
        "\r\r",
    ];

    println!("\n=== Consecutive Whitespace Edge Cases ===\n");
    
    for text in test_cases {
        let hf_ids: Vec<u32> = hf_tokenizer.encode(text, false).unwrap().get_ids().to_vec();
        let opt_ids = opt.encode(text);
        
        let hf_decoded: Vec<String> = hf_ids.iter()
            .map(|id| vocab_r.get(id).cloned().unwrap_or_else(|| format!("?{}", id)))
            .collect();
        let opt_decoded: Vec<String> = opt_ids.iter()
            .map(|id| vocab_r.get(id).cloned().unwrap_or_else(|| format!("?{}", id)))
            .collect();
        
        let matches = hf_ids == opt_ids;
        let status = if matches { "✓ MATCH" } else { "✗ DIFFER" };
        
        println!("Input: {:?}", text);
        println!("  HF:  {:?} -> {:?}", hf_ids, hf_decoded);
        println!("  Opt: {:?} -> {:?}", opt_ids, opt_decoded);
        println!("  Status: {}\n", status);
    }
}
