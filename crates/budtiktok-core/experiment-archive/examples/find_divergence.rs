//! Find exact divergence between BudTikTok and HuggingFace tokenization

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

    // Load doc 2 and 8
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", workspace);
    let file = fs::File::open(&data_path).unwrap();
    let reader = BufReader::new(file);

    let docs: Vec<String> = reader.lines()
        .take(10)
        .map(|line| {
            let line = line.unwrap();
            let json: serde_json::Value = serde_json::from_str(&line).unwrap();
            json["text"].as_str().unwrap_or("").to_string()
        })
        .collect();

    // Test short samples to find divergence
    let test_cases = vec![
        // From doc 2 - check different segments
        &docs[2][..200],
        &docs[2][200..400],
        &docs[2][400..600],
        // From doc 8
        &docs[8][..200],
        &docs[8][200..400],
    ];

    println!("Testing short segments to find divergence...\n");

    for (i, text) in test_cases.iter().enumerate() {
        let tokens = encoder.encode(text);
        println!("Segment {}: {} chars -> {} BudTikTok tokens", i, text.len(), tokens.len());

        // Decode tokens to verify
        let decoded: Vec<&str> = tokens.iter()
            .map(|&id| vocab_r.get(id as usize).map(|s| s.as_str()).unwrap_or("<unk>"))
            .collect();
        println!("  First 10 tokens: {:?}", &decoded[..decoded.len().min(10)]);
    }

    // Test specific strings that might cause issues
    println!("\n\nTesting specific patterns...\n");
    let patterns = vec![
        "Hello world",
        "Hello, world!",
        "consistency.",
        "fault.",
        "fault.\n",
        "fault.\n\n",
        "Townhall.com.",
        "Townhall.com.\n",
        "\n\n",
        "Kim Jong Un.",
        "AP Images / Business",
    ];

    for pattern in patterns {
        let tokens = encoder.encode(pattern);
        let decoded: Vec<&str> = tokens.iter()
            .map(|&id| vocab_r.get(id as usize).map(|s| s.as_str()).unwrap_or("<unk>"))
            .collect();
        println!("'{}' -> {} tokens: {:?}", pattern.escape_default(), tokens.len(), decoded);
    }
}
