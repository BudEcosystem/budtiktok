//! Test different parallel optimization strategies

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};
use rayon::prelude::*;
use serde_json::Value;
use ahash::AHashMap;

// BPE imports
use budtiktok_core::bpe_linear::OptimizedBpeEncoder;

fn main() {
    // Load vocabulary
    let vocab_path = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/gpt2/vocab.json";
    let merges_path = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/gpt2/merges.txt";

    println!("Loading GPT-2 tokenizer...");

    let vocab_json = std::fs::read_to_string(vocab_path).expect("Failed to read vocab");
    let vocab_map: serde_json::Map<String, Value> = serde_json::from_str(&vocab_json).expect("Failed to parse vocab");
    let vocab_ahash: AHashMap<String, u32> = vocab_map
        .into_iter()
        .map(|(k, v)| (k, v.as_u64().unwrap() as u32))
        .collect();

    let merges_content = std::fs::read_to_string(merges_path).expect("Failed to read merges");
    let merges: Vec<(String, String)> = merges_content
        .lines()
        .skip(1)
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                Some((parts[0].to_string(), parts[1].to_string()))
            } else {
                None
            }
        })
        .collect();

    let encoder = OptimizedBpeEncoder::new(vocab_ahash, merges, "<|endoftext|>");

    // Load documents
    let data_path = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/openwebtext_1gb.jsonl";
    let file = File::open(data_path).expect("Failed to open dataset");
    let reader = BufReader::new(file);

    let documents: Vec<String> = reader
        .lines()
        .take(10000)
        .filter_map(|line| {
            let line = line.ok()?;
            let v: Value = serde_json::from_str(&line).ok()?;
            v.get("text")?.as_str().map(String::from)
        })
        .collect();

    let total_bytes: usize = documents.iter().map(|d| d.len()).sum();
    let num_threads = rayon::current_num_threads();

    println!("Loaded {} docs ({:.2} MB)", documents.len(), total_bytes as f64 / 1024.0 / 1024.0);
    println!("Threads: {}", num_threads);
    println!();

    // Baseline: single-core
    println!("=== SINGLE-CORE BASELINE ===");
    let start = Instant::now();
    let mut single_tokens = 0usize;
    for doc in &documents {
        single_tokens += encoder.encode(doc).len();
    }
    let single_time = start.elapsed();
    println!("  Time: {:.3}s, Tokens: {}, Throughput: {:.1} MB/s",
        single_time.as_secs_f64(), single_tokens,
        total_bytes as f64 / single_time.as_secs_f64() / 1024.0 / 1024.0);
    println!();

    // Strategy 1: Simple par_iter (current)
    println!("=== STRATEGY 1: par_iter().map().collect() ===");
    let start = Instant::now();
    let results: Vec<Vec<u32>> = documents.par_iter()
        .map(|doc| encoder.encode(doc))
        .collect();
    let time1 = start.elapsed();
    let tokens1: usize = results.iter().map(|v: &Vec<u32>| v.len()).sum();
    let efficiency1 = (single_time.as_secs_f64() / time1.as_secs_f64()) / num_threads as f64 * 100.0;
    println!("  Time: {:.3}s, Tokens: {}, Throughput: {:.1} MB/s",
        time1.as_secs_f64(), tokens1,
        total_bytes as f64 / time1.as_secs_f64() / 1024.0 / 1024.0);
    println!("  Speedup: {:.2}x, Efficiency: {:.1}%",
        single_time.as_secs_f64() / time1.as_secs_f64(), efficiency1);
    println!();

    // Strategy 2: Pre-allocated with par_iter_mut
    println!("=== STRATEGY 2: Pre-allocated indexed ===");
    let mut results2: Vec<Vec<u32>> = vec![Vec::new(); documents.len()];
    let start = Instant::now();
    results2.par_iter_mut()
        .enumerate()
        .for_each(|(i, result)| {
            *result = encoder.encode(&documents[i]);
        });
    let time2 = start.elapsed();
    let tokens2: usize = results2.iter().map(|v: &Vec<u32>| v.len()).sum();
    let efficiency2 = (single_time.as_secs_f64() / time2.as_secs_f64()) / num_threads as f64 * 100.0;
    println!("  Time: {:.3}s, Tokens: {}, Throughput: {:.1} MB/s",
        time2.as_secs_f64(), tokens2,
        total_bytes as f64 / time2.as_secs_f64() / 1024.0 / 1024.0);
    println!("  Speedup: {:.2}x, Efficiency: {:.1}%",
        single_time.as_secs_f64() / time2.as_secs_f64(), efficiency2);
    println!();

    // Strategy 3: par_chunks to reduce scheduling overhead
    println!("=== STRATEGY 3: par_chunks (chunk_size=100) ===");
    let start = Instant::now();
    let results3: Vec<Vec<u32>> = documents.par_chunks(100)
        .flat_map(|chunk| {
            chunk.iter().map(|doc| encoder.encode(doc)).collect::<Vec<_>>()
        })
        .collect();
    let time3 = start.elapsed();
    let tokens3: usize = results3.iter().map(|v: &Vec<u32>| v.len()).sum();
    let efficiency3 = (single_time.as_secs_f64() / time3.as_secs_f64()) / num_threads as f64 * 100.0;
    println!("  Time: {:.3}s, Tokens: {}, Throughput: {:.1} MB/s",
        time3.as_secs_f64(), tokens3,
        total_bytes as f64 / time3.as_secs_f64() / 1024.0 / 1024.0);
    println!("  Speedup: {:.2}x, Efficiency: {:.1}%",
        single_time.as_secs_f64() / time3.as_secs_f64(), efficiency3);
    println!();

    // Strategy 4: Sort by size for better load balance (descending)
    println!("=== STRATEGY 4: Size-sorted (descending) + par_iter ===");
    let mut indexed: Vec<(usize, &String)> = documents.iter().enumerate().collect();
    indexed.sort_by_key(|(_, d)| std::cmp::Reverse(d.len()));

    let start = Instant::now();
    let mut results4_indexed: Vec<(usize, Vec<u32>)> = indexed.par_iter()
        .map(|(i, doc)| (*i, encoder.encode(doc)))
        .collect();
    results4_indexed.sort_by_key(|(i, _)| *i);
    let time4 = start.elapsed();
    let tokens4: usize = results4_indexed.iter().map(|(_, v): &(usize, Vec<u32>)| v.len()).sum();
    let efficiency4 = (single_time.as_secs_f64() / time4.as_secs_f64()) / num_threads as f64 * 100.0;
    println!("  Time: {:.3}s, Tokens: {}, Throughput: {:.1} MB/s",
        time4.as_secs_f64(), tokens4,
        total_bytes as f64 / time4.as_secs_f64() / 1024.0 / 1024.0);
    println!("  Speedup: {:.2}x, Efficiency: {:.1}%",
        single_time.as_secs_f64() / time4.as_secs_f64(), efficiency4);
    println!();

    // Strategy 5: Atomic counting (no result collection)
    println!("=== STRATEGY 5: Atomic counting only (no Vec collect) ===");
    let token_count = AtomicUsize::new(0);
    let start = Instant::now();
    documents.par_iter()
        .for_each(|doc| {
            let tokens = encoder.encode(doc);
            token_count.fetch_add(tokens.len(), Ordering::Relaxed);
        });
    let time5 = start.elapsed();
    let tokens5 = token_count.load(Ordering::Relaxed);
    let efficiency5 = (single_time.as_secs_f64() / time5.as_secs_f64()) / num_threads as f64 * 100.0;
    println!("  Time: {:.3}s, Tokens: {}, Throughput: {:.1} MB/s",
        time5.as_secs_f64(), tokens5,
        total_bytes as f64 / time5.as_secs_f64() / 1024.0 / 1024.0);
    println!("  Speedup: {:.2}x, Efficiency: {:.1}%",
        single_time.as_secs_f64() / time5.as_secs_f64(), efficiency5);
    println!();

    // Strategy 6: Size-balanced interleaving
    println!("=== STRATEGY 6: Size-balanced interleaving ===");
    let mut by_size: Vec<(usize, &String)> = documents.iter().enumerate().collect();
    by_size.sort_by_key(|(_, d)| d.len());
    // Interleave: take from front and back alternately
    let mut interleaved: Vec<(usize, &String)> = Vec::with_capacity(documents.len());
    let mid = by_size.len() / 2;
    for i in 0..mid {
        interleaved.push(by_size[i].clone());
        interleaved.push(by_size[by_size.len() - 1 - i].clone());
    }
    if by_size.len() % 2 == 1 {
        interleaved.push(by_size[mid].clone());
    }

    let start = Instant::now();
    let mut results6: Vec<(usize, Vec<u32>)> = interleaved.par_iter()
        .map(|(i, doc)| (*i, encoder.encode(doc)))
        .collect();
    results6.sort_by_key(|(i, _)| *i);
    let time6 = start.elapsed();
    let tokens6: usize = results6.iter().map(|(_, v): &(usize, Vec<u32>)| v.len()).sum();
    let efficiency6 = (single_time.as_secs_f64() / time6.as_secs_f64()) / num_threads as f64 * 100.0;
    println!("  Time: {:.3}s, Tokens: {}, Throughput: {:.1} MB/s",
        time6.as_secs_f64(), tokens6,
        total_bytes as f64 / time6.as_secs_f64() / 1024.0 / 1024.0);
    println!("  Speedup: {:.2}x, Efficiency: {:.1}%",
        single_time.as_secs_f64() / time6.as_secs_f64(), efficiency6);
    println!();

    println!("=== SUMMARY ===");
    println!("Strategy 1 (par_iter):          {:.1}% efficiency", efficiency1);
    println!("Strategy 2 (pre-alloc):         {:.1}% efficiency", efficiency2);
    println!("Strategy 3 (par_chunks):        {:.1}% efficiency", efficiency3);
    println!("Strategy 4 (size-sorted desc):  {:.1}% efficiency", efficiency4);
    println!("Strategy 5 (atomic count):      {:.1}% efficiency", efficiency5);
    println!("Strategy 6 (interleaved):       {:.1}% efficiency", efficiency6);
}
