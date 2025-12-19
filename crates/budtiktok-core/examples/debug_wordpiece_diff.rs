//! Debug WordPiece differences between scalar and SIMD implementations

use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use ahash::AHashMap;

use budtiktok_core::wordpiece::{WordPieceConfig, WordPieceTokenizer};
use budtiktok_core::wordpiece_hyper::{HyperConfig, HyperWordPieceTokenizer};
use budtiktok_core::vocab::{Vocabulary, SpecialTokens};
use budtiktok_core::tokenizer::Tokenizer;

const WORKSPACE: &str = "/home/bud/Desktop/latentbud/budtiktok";

fn main() {
    let tokenizer_path = format!("{}/test_data/bert-base-uncased/tokenizer.json", WORKSPACE);
    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", WORKSPACE);

    // Load vocabulary
    let content = fs::read_to_string(&tokenizer_path).expect("Failed to read tokenizer.json");
    let json: serde_json::Value = serde_json::from_str(&content).expect("Failed to parse");
    let vocab_obj = json["model"]["vocab"].as_object().expect("Missing vocab");
    let mut token_to_id: AHashMap<String, u32> = AHashMap::new();
    for (token, id) in vocab_obj {
        token_to_id.insert(token.clone(), id.as_u64().unwrap() as u32);
    }

    // Build reverse vocab
    let mut id_to_token: Vec<String> = vec![String::new(); token_to_id.len()];
    for (token, &id) in &token_to_id {
        if (id as usize) < id_to_token.len() {
            id_to_token[id as usize] = token.clone();
        }
    }

    // Create tokenizers
    let config = WordPieceConfig::default();
    let special_tokens = SpecialTokens {
        unk_token: Some("[UNK]".to_string()),
        cls_token: Some("[CLS]".to_string()),
        sep_token: Some("[SEP]".to_string()),
        pad_token: Some("[PAD]".to_string()),
        mask_token: Some("[MASK]".to_string()),
        ..Default::default()
    };
    let vocabulary = Vocabulary::new(token_to_id.clone(), special_tokens.clone());
    let scalar_tokenizer = WordPieceTokenizer::new(vocabulary.clone(), config);

    let hyper_config = HyperConfig {
        continuing_subword_prefix: "##".to_string(),
        max_input_chars_per_word: 100,
        unk_token: "[UNK]".to_string(),
        do_lower_case: true,
        strip_accents: true,
        tokenize_chinese_chars: true,
        hash_table_bits: 14,
    };
    let simd_tokenizer = HyperWordPieceTokenizer::new(vocabulary.clone(), hyper_config);

    // Load documents
    let file = File::open(&data_path).expect("Failed to open dataset");
    let reader = BufReader::new(file);
    let documents: Vec<String> = reader.lines()
        .take(10000) // All docs
        .filter_map(|line| {
            let line = line.ok()?;
            let json: serde_json::Value = serde_json::from_str(&line).ok()?;
            json["text"].as_str().map(|s| s.to_string())
        })
        .collect();

    println!("Analyzing {} documents...\n", documents.len());

    // Specific docs to check
    let docs_to_check: Vec<usize> = (0..documents.len()).collect();

    let mut total_diffs = 0;
    for i in 0..documents.len() {
        let doc = &documents[i];
        let scalar_encoding = scalar_tokenizer.encode(doc, false).unwrap();
        let scalar_ids = scalar_encoding.get_ids();
        let simd_ids = simd_tokenizer.encode_fast(doc);

        if scalar_ids != &simd_ids[..] {
            total_diffs += 1;
            if total_diffs <= 10 {
                println!("=== Document {} ===", i);
                println!("Scalar: {} tokens, SIMD: {} tokens", scalar_ids.len(), simd_ids.len());

                // Find first difference
                let mut first_diff_idx = 0;
                for j in 0..scalar_ids.len().max(simd_ids.len()) {
                    let s = scalar_ids.get(j);
                    let d = simd_ids.get(j);
                    if s != d {
                        first_diff_idx = j;
                        break;
                    }
                }

                // Show context around difference
                let start = first_diff_idx.saturating_sub(3);
                let end_scalar = (first_diff_idx + 5).min(scalar_ids.len());
                let end_simd = (first_diff_idx + 5).min(simd_ids.len());

                println!("First difference at index {}:", first_diff_idx);

                print!("  Scalar tokens: ");
                for j in start..end_scalar {
                    let id = scalar_ids[j];
                    let token = id_to_token.get(id as usize).map(|s| s.as_str()).unwrap_or("[?]");
                    if j == first_diff_idx {
                        print!("[>>>{}<<<] ", token);
                    } else {
                        print!("{} ", token);
                    }
                }
                println!();

                print!("  SIMD tokens:   ");
                for j in start..end_simd {
                    let id = simd_ids[j];
                    let token = id_to_token.get(id as usize).map(|s| s.as_str()).unwrap_or("[?]");
                    if j == first_diff_idx {
                        print!("[>>>{}<<<] ", token);
                    } else {
                        print!("{} ", token);
                    }
                }
                println!();

                // Try to find the source text that caused the difference
                // Get character position by summing token lengths up to the difference
                let mut char_pos = 0;
                for j in 0..first_diff_idx {
                    if let Some(&id) = scalar_ids.get(j) {
                        if let Some(token) = id_to_token.get(id as usize) {
                            let t = if token.starts_with("##") { &token[2..] } else { token };
                            char_pos += t.len();
                        }
                    }
                }

                // Show surrounding text
                let text_start = char_pos.saturating_sub(20);
                let text_end = (char_pos + 40).min(doc.len());
                if text_end > text_start {
                    let context = &doc[text_start..text_end];
                    println!("  Source text around pos {}: {:?}", char_pos, context);
                }
                println!();
            }
        }
    }

    println!("\nTotal documents with differences: {} / {}", total_diffs, documents.len());
}
