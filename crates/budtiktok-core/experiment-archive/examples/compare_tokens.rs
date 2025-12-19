//! Compare token counts with HuggingFace

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

    // Load first 10 documents
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", workspace);
    let file = fs::File::open(&data_path).unwrap();
    let reader = BufReader::new(file);

    // Expected HF token counts from Python test
    let hf_expected = [1217, 2187, 1278, 1521, 671, 795, 532, 659, 845, 1101];

    let mut total_tokens = 0usize;
    let mut total_diff = 0i64;

    for (i, line) in reader.lines().take(10).enumerate() {
        let line = line.unwrap();
        let json: serde_json::Value = serde_json::from_str(&line).unwrap();
        let text = json["text"].as_str().unwrap_or("");
        let tokens = encoder.encode(text);
        let bt_count = tokens.len();
        let hf_count = hf_expected[i];
        let diff = bt_count as i64 - hf_count as i64;
        total_diff += diff;
        total_tokens += bt_count;

        println!("Doc {}: {} chars | BT: {} | HF: {} | diff: {:+}",
                 i, text.len(), bt_count, hf_count, diff);
    }
    println!("\nTotal BudTikTok: {} | Total HF: {} | Total diff: {:+}",
             total_tokens, hf_expected.iter().sum::<usize>(), total_diff);
}
