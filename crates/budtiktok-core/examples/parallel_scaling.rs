//! Test parallel scaling with different thread counts

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};
use rayon::prelude::*;
use serde_json::Value;
use ahash::AHashMap;

use budtiktok_core::bpe_linear::OptimizedBpeEncoder;
use budtiktok_core::{NumaTopology, set_thread_affinity};

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

    println!("Loaded {} docs ({:.2} MB)", documents.len(), total_bytes as f64 / 1024.0 / 1024.0);
    println!();

    // Single-core baseline
    println!("=== SINGLE-CORE BASELINE ===");
    let start = Instant::now();
    let mut single_tokens = 0usize;
    for doc in &documents {
        single_tokens += encoder.encode(doc).len();
    }
    let single_time = start.elapsed();
    let single_throughput = total_bytes as f64 / single_time.as_secs_f64() / 1024.0 / 1024.0;
    println!("  Time: {:.3}s, Throughput: {:.1} MB/s", single_time.as_secs_f64(), single_throughput);
    println!();

    // Test with different thread counts
    println!("=== SCALING TEST ===");
    println!("{:>8} {:>12} {:>12} {:>12} {:>12}", "Threads", "Time (s)", "Throughput", "Speedup", "Efficiency");
    println!("{:-<8} {:-<12} {:-<12} {:-<12} {:-<12}", "", "", "", "", "");

    // Detect NUMA topology
    let numa = NumaTopology::detect();
    println!("NUMA topology: {} nodes, {} CPUs total", numa.num_nodes, numa.total_cpus);
    if numa.numa_available {
        println!("NUMA is available - testing with thread affinity");
    }
    println!();

    for num_threads in [1, 2, 4, 8, 12, 16] {
        // Configure rayon thread pool with NUMA-aware thread affinity
        let thread_counter = AtomicUsize::new(0);
        let numa_clone = numa.clone();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .start_handler(move |_thread_idx| {
                // Set thread affinity based on NUMA topology
                let idx = thread_counter.fetch_add(1, Ordering::SeqCst);
                let cpu = numa_clone.thread_to_cpu(idx);
                set_thread_affinity(cpu);
            })
            .build()
            .unwrap();

        let start = Instant::now();
        let tokens: usize = pool.install(|| {
            documents.par_iter()
                .map(|doc| encoder.encode(doc).len())
                .sum()
        });
        let time = start.elapsed();

        let throughput = total_bytes as f64 / time.as_secs_f64() / 1024.0 / 1024.0;
        let speedup = single_time.as_secs_f64() / time.as_secs_f64();
        let efficiency = speedup / num_threads as f64 * 100.0;

        assert_eq!(tokens, single_tokens);

        println!("{:>8} {:>12.3} {:>10.1} MB/s {:>10.2}x {:>10.1}%",
            num_threads, time.as_secs_f64(), throughput, speedup, efficiency);
    }

    println!();
    println!("=== ANALYSIS ===");
    println!("If efficiency decreases significantly with more threads, the bottleneck is likely:");
    println!("  - Memory bandwidth (all threads competing for RAM access)");
    println!("  - Cache contention (thrashing shared L3 cache)");
    println!("  - NUMA effects (threads accessing remote memory)");
}
