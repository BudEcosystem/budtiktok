#!/usr/bin/env python3
"""Debug accuracy issues between HF and BudTikTok tokenizers."""

import subprocess
import json

def main():
    from transformers import AutoTokenizer

    print("=" * 80)
    print("TOKEN ACCURACY DEBUG")
    print("=" * 80)
    print()

    # Load HF tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    test_texts = [
        "Hello world",
        "The quick brown fox",
        "Machine learning is fascinating",
        "Natural language processing enables computers to understand human text.",
    ]

    for text in test_texts:
        print(f"\nText: '{text}'")
        print("-" * 60)

        # HF tokenization
        hf_result = tokenizer(text, return_tensors=None, add_special_tokens=True)
        hf_ids = hf_result['input_ids']
        hf_tokens = tokenizer.convert_ids_to_tokens(hf_ids)

        print(f"HF tokens: {hf_tokens}")
        print(f"HF IDs:    {hf_ids}")
        print()

    print()
    print("=" * 80)
    print("CHECKING RUST BUDTIKTOK OUTPUT")
    print("=" * 80)

    # Try running a quick Rust test
    rust_test = '''
use budtiktok_hf_compat::Tokenizer;

fn main() {
    let json = std::fs::read_to_string(
        std::env::var("HOME").unwrap() +
        "/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/tokenizer.json"
    ).unwrap();

    let tokenizer = Tokenizer::from_str(&json).unwrap();

    let texts = [
        "Hello world",
        "The quick brown fox",
        "Machine learning is fascinating",
    ];

    for text in &texts {
        let encoding = tokenizer.encode(*text, true).unwrap();
        println!("Text: '{}'", text);
        println!("  IDs: {:?}", encoding.get_ids());
        println!();
    }
}
'''
    print("(Run the Rust debug binary to see BudTikTok output)")

if __name__ == "__main__":
    main()
