//! Debug accuracy issues between HF and BudTikTok

use budtiktok_hf_compat::Tokenizer;
use tokenizers::Tokenizer as HfTokenizer;
use std::env;

fn main() {
    println!("{}", "=".repeat(80));
    println!("{:^80}", "TOKEN ACCURACY DEBUG");
    println!("{}", "=".repeat(80));
    println!();

    let tokenizer_path = format!(
        "{}/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer.json",
        env::var("HOME").unwrap()
    );

    println!("Loading tokenizers from: {}", tokenizer_path);
    println!();

    let json = std::fs::read_to_string(&tokenizer_path).unwrap();

    // Load HF tokenizer
    let hf_tokenizer = HfTokenizer::from_file(&tokenizer_path).expect("Failed to load HF tokenizer");

    // Load BudTikTok tokenizer
    let bud_tokenizer = Tokenizer::from_str(&json).expect("Failed to load BudTikTok tokenizer");

    let texts = [
        "Hello world",
        "The quick brown fox",
        "Machine learning is fascinating",
        "Natural language processing enables computers to understand human text.",
    ];

    for text in &texts {
        println!("Text: '{}'", text);
        println!("{}", "-".repeat(60));

        // HF tokenization
        let hf_encoding = hf_tokenizer.encode(*text, true).expect("HF encode failed");
        let hf_ids: Vec<u32> = hf_encoding.get_ids().to_vec();
        let hf_tokens: Vec<&str> = hf_encoding.get_tokens().iter().map(|s| s.as_str()).collect();

        println!("HF tokens: {:?}", hf_tokens);
        println!("HF IDs:    {:?}", hf_ids);

        // BudTikTok tokenization
        let bud_encoding = bud_tokenizer.encode(*text, true).expect("BudTikTok encode failed");
        let bud_ids = bud_encoding.get_ids().to_vec();

        println!("Bud IDs:   {:?}", bud_ids);

        if hf_ids == bud_ids {
            println!("✓ MATCH");
        } else {
            println!("✗ MISMATCH");
            // Show difference
            for (i, (hf, bud)) in hf_ids.iter().zip(bud_ids.iter()).enumerate() {
                if hf != bud {
                    println!("  Position {}: HF={}, Bud={}", i, hf, bud);
                }
            }
            if hf_ids.len() != bud_ids.len() {
                println!("  Length: HF={}, Bud={}", hf_ids.len(), bud_ids.len());
            }
        }
        println!();
    }
}
