//! Full comparison of doc 2 tokenization

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

    // Build reverse vocab
    let mut vocab_r: Vec<String> = vec![String::new(); vocab.len()];
    for (k, &v) in &vocab {
        vocab_r[v as usize] = k.clone();
    }

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

    // Tokenize full document
    let tokens = encoder.encode(&doc2);

    // HF expected tokens (hardcoded for comparison)
    let hf_expected: Vec<u32> = vec![464, 9317, 6241, 416, 5721, 1023, 389, 511, 898, 290, 466, 407, 2380, 262, 5009, 286, 8329, 18323, 13, 785, 13, 198, 198, 1639, 423, 284, 1577, 1992, 8732, 2486, 3884, 329, 530, 1517, 25, 15794, 13, 10528, 318, 1683, 465, 8046, 13, 10528, 481, 1683, 307, 465, 8046, 13];

    println!("BudTikTok: {} tokens", tokens.len());
    println!("Expected (first 50): {} tokens", hf_expected.len());

    // Print all tokens
    println!("\nAll BudTikTok tokens:");
    for (i, &tok) in tokens.iter().enumerate() {
        let decoded = vocab_r.get(tok as usize).map(|s| s.as_str()).unwrap_or("<unk>");
        if i > 0 && i % 20 == 0 {
            println!();
        }
        print!("{}", decoded);
    }
    println!("\n");

    // Print token IDs for comparison
    println!("\nToken IDs (first 100):");
    for (i, &tok) in tokens.iter().take(100).enumerate() {
        print!("{:5} ", tok);
        if (i + 1) % 10 == 0 {
            println!();
        }
    }
}
