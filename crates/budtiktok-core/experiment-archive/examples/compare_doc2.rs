//! Compare tokenization of doc 2 chunk by chunk to find divergence

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use ahash::AHashMap;

fn main() {
    let workspace = "/home/bud/Desktop/latentbud/budtiktok";

    // Load vocab and merges
    let vocab_path = format!("{}/benchmark_data/gpt2/vocab.json", workspace);
    let content = fs::read_to_string(&vocab_path).unwrap();
    let vocab: HashMap<String, u32> = serde_json::from_str(&content).unwrap();
    let vocab_ahash: AHashMap<String, u32> = vocab.iter().map(|(k, v)| (k.clone(), *v)).collect();

    let merges_path = format!("{}/benchmark_data/gpt2/merges.txt", workspace);
    let merge_content = fs::read_to_string(&merges_path).unwrap();
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

    use budtiktok_core::bpe_linear::OptimizedBpeEncoder;
    let encoder = OptimizedBpeEncoder::new(vocab_ahash, merges, "<|endoftext|>");

    // Load doc 2
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", workspace);
    let file = fs::File::open(&data_path).unwrap();
    let reader = BufReader::new(file);

    let doc2: String = reader.lines()
        .skip(2)
        .take(1)
        .map(|line| {
            let line = line.unwrap();
            let json: serde_json::Value = serde_json::from_str(&line).unwrap();
            json["text"].as_str().unwrap_or("").to_string()
        })
        .next()
        .unwrap();

    // HF expected token counts for 500-char chunks
    // (We'll need to compare these with Python output)
    println!("Doc 2: {} chars", doc2.len());
    println!("\nTokenizing in 500-char chunks:");

    let chunk_size = 500;
    let mut total_tokens = 0;

    for (i, chunk_start) in (0..doc2.len()).step_by(chunk_size).enumerate() {
        let chunk_end = (chunk_start + chunk_size).min(doc2.len());
        let chunk = &doc2[chunk_start..chunk_end];
        let tokens = encoder.encode(chunk);
        total_tokens += tokens.len();
        println!("Chunk {} (chars {}..{}): {} tokens",
                 i, chunk_start, chunk_end, tokens.len());
    }

    println!("\nTotal tokens (chunked): {}", total_tokens);

    // Full document tokenization
    let full_tokens = encoder.encode(&doc2);
    println!("Full document tokens: {}", full_tokens.len());

    // Print first 50 tokens
    println!("\nFirst 50 token IDs: {:?}", &full_tokens[..50.min(full_tokens.len())]);
}
