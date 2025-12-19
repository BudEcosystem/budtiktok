//! Detailed performance benchmark for BPE encoding
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
    
    // Load sample data - first 10000 documents
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", workspace);
    let mut documents = Vec::new();
    let mut total_bytes = 0usize;
    
    let file = std::fs::File::open(&data_path).expect("Failed to open data file");
    let reader = std::io::BufReader::new(file);
    use std::io::BufRead;
    
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
    use budtiktok_core::bpe_linear::gpt2_pre_tokenize_simd;
    use budtiktok_core::bpe_fast::Gpt2ByteEncoderFast;
    
    let encoder = OptimizedBpeEncoder::new(vocab_ahash.clone(), merges.clone(), "<|endoftext|>");
    let byte_encoder = Gpt2ByteEncoderFast::new();
    
    // Warm up
    println!("\nWarm-up...");
    for doc in documents.iter().take(100) {
        let _ = encoder.encode(doc);
    }
    
    // Phase 1: Pre-tokenization only
    println!("\n=== Phase Breakdown ===\n");
    
    println!("1. Pre-tokenization only:");
    let start = Instant::now();
    let mut total_pre_tokens = 0usize;
    for doc in &documents {
        let pre_tokens = gpt2_pre_tokenize_simd(doc);
        total_pre_tokens += pre_tokens.len();
    }
    let pretok_time = start.elapsed();
    println!("   Time: {:.2?}", pretok_time);
    println!("   Pre-tokens: {}", total_pre_tokens);
    
    // Phase 2: Pre-tokenization + byte encoding
    println!("\n2. Pre-tokenization + byte encoding:");
    let start = Instant::now();
    let mut total_byte_encoded = 0usize;
    for doc in &documents {
        let pre_tokens = gpt2_pre_tokenize_simd(doc);
        for pt in pre_tokens {
            let be = byte_encoder.encode_string(pt);
            total_byte_encoded += be.len();
        }
    }
    let pretok_be_time = start.elapsed();
    println!("   Time: {:.2?}", pretok_be_time);
    println!("   Byte encoding overhead: {:.2?}", pretok_be_time - pretok_time);
    
    // Phase 3: Full encoding
    println!("\n3. Full encoding:");
    let start = Instant::now();
    let mut total_tokens = 0usize;
    for doc in &documents {
        total_tokens += encoder.encode(doc).len();
    }
    let full_time = start.elapsed();
    let throughput = total_mb / full_time.as_secs_f64();
    println!("   Time: {:.2?}", full_time);
    println!("   Throughput: {:.2} MB/s", throughput);
    println!("   BPE merge overhead: {:.2?}", full_time - pretok_be_time);
    
    // Summary
    println!("\n=== Summary ===");
    let pretok_pct = pretok_time.as_secs_f64() / full_time.as_secs_f64() * 100.0;
    let be_pct = (pretok_be_time - pretok_time).as_secs_f64() / full_time.as_secs_f64() * 100.0;
    let bpe_pct = (full_time - pretok_be_time).as_secs_f64() / full_time.as_secs_f64() * 100.0;
    
    println!("Pre-tokenization: {:.1}%", pretok_pct);
    println!("Byte encoding:    {:.1}%", be_pct);
    println!("BPE merge:        {:.1}%", bpe_pct);
    println!("\nTotal throughput: {:.2} MB/s ({:.2}x vs HF 2.67 MB/s)", throughput, throughput / 2.67);
}
