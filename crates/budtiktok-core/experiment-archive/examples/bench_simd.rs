//! SIMD Pre-tokenizer Benchmark
use std::collections::HashMap;
use std::fs;
use std::time::Instant;
use ahash::AHashMap;

fn main() {
    let workspace = "/home/bud/Desktop/latentbud/budtiktok";
    
    // Load vocab
    let vocab_path = format!("{}/benchmark_data/gpt2/vocab.json", workspace);
    let content = fs::read_to_string(&vocab_path).expect("Failed to read vocab");
    let vocab: HashMap<String, u32> = serde_json::from_str(&content).expect("Failed to parse vocab");
    let vocab_ahash: AHashMap<String, u32> = vocab.iter().map(|(k, v)| (k.clone(), *v)).collect();
    
    // Load merges
    let merges_path = format!("{}/benchmark_data/gpt2/merges.txt", workspace);
    let merge_content = fs::read_to_string(&merges_path).expect("Failed to read merges");
    let merges: Vec<(String, String)> = merge_content.lines()
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
    
    // Load sample data
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", workspace);
    let mut documents = Vec::new();
    let mut total_bytes = 0usize;
    
    let file = std::fs::File::open(&data_path).expect("Failed to open data file");
    let reader = std::io::BufReader::new(file);
    use std::io::BufRead;
    
    // Load first 10000 documents
    for line in reader.lines().take(10000) {
        let line = line.expect("Failed to read line");
        let json: serde_json::Value = serde_json::from_str(&line).expect("Failed to parse JSON");
        let text = json["text"].as_str().unwrap_or("").to_string();
        total_bytes += text.len();
        documents.push(text);
    }
    
    let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
    println!("Loaded {} documents ({:.2} MB)", documents.len(), total_mb);
    
    // Create encoder
    use budtiktok_core::bpe_linear::OptimizedBpeEncoder;
    let encoder = OptimizedBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<|endoftext|>");
    
    // Warm up
    println!("\nWarm-up...");
    for doc in documents.iter().take(100) {
        let _ = encoder.encode(doc);
    }
    
    // Benchmark
    println!("\nBenchmarking OptimizedBpeEncoder (SIMD pre-tokenizer)...");
    let start = Instant::now();
    let mut total_tokens = 0usize;
    for doc in &documents {
        total_tokens += encoder.encode(doc).len();
    }
    let elapsed = start.elapsed();
    let throughput = total_mb / elapsed.as_secs_f64();
    println!("  Time: {:.2?}", elapsed);
    println!("  Throughput: {:.2} MB/s", throughput);
    println!("  Tokens: {}", total_tokens);
    
    // Second run (cache warmed)
    println!("\nSecond run (cache warmed)...");
    let start = Instant::now();
    let mut total_tokens2 = 0usize;
    for doc in &documents {
        total_tokens2 += encoder.encode(doc).len();
    }
    let elapsed2 = start.elapsed();
    let throughput2 = total_mb / elapsed2.as_secs_f64();
    println!("  Time: {:.2?}", elapsed2);
    println!("  Throughput: {:.2} MB/s", throughput2);
    println!("  Tokens: {}", total_tokens2);
    
    // HuggingFace baseline: ~2.67 MB/s
    println!("\n=== Results ===");
    println!("SIMD Optimized: {:.2} MB/s ({:.2}x vs HF 2.67 MB/s)", throughput, throughput / 2.67);
    println!("Cache-warmed:   {:.2} MB/s ({:.2}x vs HF 2.67 MB/s)", throughput2, throughput2 / 2.67);
}
