//! Debug specific documents that mismatch

use budtiktok_core::wordpiece::WordPieceTokenizer;
use budtiktok_core::Tokenizer;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    // Load BERT tokenizer
    let bert_vocab_path = "/home/bud/Desktop/latentbud/budtiktok/crates/budtiktok-core/data/bert-base-uncased-vocab.txt";
    let tokenizer = WordPieceTokenizer::from_pretrained(bert_vocab_path)
        .expect("Failed to load tokenizer");
    
    // Load documents
    let data_path = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/openwebtext_1gb.jsonl";
    let file = File::open(data_path).expect("Failed to open data file");
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
    
    // Debug specific documents
    for &doc_idx in &[6886usize, 9300usize] {
        let doc = &documents[doc_idx];
        let encoding = tokenizer.encode(doc, false).unwrap();
        let tokens = encoding.get_ids();

        println!("Document {}: {} tokens", doc_idx, tokens.len());

        // Print first 50 tokens as strings
        let vocab = tokenizer.vocabulary();
        let token_strs: Vec<&str> = tokens.iter()
            .take(50)
            .map(|&id| vocab.id_to_token(id).unwrap_or("[UNK]"))
            .collect();
        println!("First 50: {:?}", token_strs);

        // Print last 50 tokens
        let start = if tokens.len() > 50 { tokens.len() - 50 } else { 0 };
        let token_strs: Vec<&str> = tokens[start..].iter()
            .map(|&id| vocab.id_to_token(id).unwrap_or("[UNK]"))
            .collect();
        println!("Last 50: {:?}", token_strs);
        
        // Print document content near end
        let doc_end = if doc.len() > 500 { &doc[doc.len()-500..] } else { doc };
        println!("Document end (last 500 chars): {:?}", doc_end);
        println!();
    }
}
