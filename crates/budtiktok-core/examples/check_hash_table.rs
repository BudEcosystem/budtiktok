//! Check hash table for missing tokens

use std::fs;
use ahash::AHashMap;

use budtiktok_core::wordpiece_hyper::{HyperConfig, HyperWordPieceTokenizer};
use budtiktok_core::vocab::{Vocabulary, SpecialTokens};

const WORKSPACE: &str = "/home/bud/Desktop/latentbud/budtiktok";

fn main() {
    let tokenizer_path = format!("{}/test_data/bert-base-uncased/tokenizer.json", WORKSPACE);

    // Load vocabulary
    let content = fs::read_to_string(&tokenizer_path).expect("Failed to read tokenizer.json");
    let json: serde_json::Value = serde_json::from_str(&content).expect("Failed to parse");
    let vocab_obj = json["model"]["vocab"].as_object().expect("Missing vocab");
    let mut token_to_id: AHashMap<String, u32> = AHashMap::new();
    for (token, id) in vocab_obj {
        token_to_id.insert(token.clone(), id.as_u64().unwrap() as u32);
    }

    // Check if words are in vocabulary
    println!("Vocabulary size: {}", token_to_id.len());
    let words_to_check = ["justified", "just", "street", "st", "fires", "fire", "57", "b", "40", "z", "##ae", "##46", "##3", "##e", "##6"];
    for word in words_to_check {
        if let Some(id) = token_to_id.get(word) {
            println!("'{}' found in vocab with id: {}", word, id);
        } else {
            println!("'{}' NOT found in vocab", word);
        }
    }

    // Create tokenizer
    let special_tokens = SpecialTokens {
        unk_token: Some("[UNK]".to_string()),
        cls_token: Some("[CLS]".to_string()),
        sep_token: Some("[SEP]".to_string()),
        pad_token: Some("[PAD]".to_string()),
        mask_token: Some("[MASK]".to_string()),
        ..Default::default()
    };
    let vocabulary = Vocabulary::new(token_to_id.clone(), special_tokens.clone());

    // Create tokenizer with debug output
    let hyper_config = HyperConfig {
        continuing_subword_prefix: "##".to_string(),
        max_input_chars_per_word: 100,
        unk_token: "[UNK]".to_string(),
        do_lower_case: true,
        strip_accents: true,
        tokenize_chinese_chars: true,
        hash_table_bits: 14, // 16K buckets
    };
    let simd_tokenizer = HyperWordPieceTokenizer::new(vocabulary.clone(), hyper_config);

    // Test tokenization
    let test_words = ["justified", "just", "hello", "world", "street", "st", "fires", "fire", "57", "b", "40", "z", "5", "7"];
    for word in test_words {
        let tokens = simd_tokenizer.encode_fast(word);
        let token_strs: Vec<String> = tokens.iter()
            .map(|&id| {
                vocabulary.id_to_token(id).unwrap_or("[?]").to_string()
            })
            .collect();
        println!("'{}' -> {:?} (ids: {:?})", word, token_strs, tokens);
    }

    // Test with more context - from the debug output
    let test_sentences = [
        "is on a justified mission",
        "was entirely justified",
        "This is justified.",
        "the main street plaza",
        "a downtown toronto street",
        "57ae463e",
        "bhfmcvvt",
        "zmuxoty",
        "57ae463e6bc53a38512c58a878370338dcfe0fb59eeedfd9b3e7959fe7c149d1",
        "\"hash\": \"57ae463e6bc53a38512c58a878370338dcfe0fb59eeedfd9b3e7959fe7c149d1\"",
    ];
    println!("\nSentence tests:");
    for sentence in test_sentences {
        let tokens = simd_tokenizer.encode_fast(sentence);
        let token_strs: Vec<String> = tokens.iter()
            .map(|&id| {
                vocabulary.id_to_token(id).unwrap_or("[?]").to_string()
            })
            .collect();
        println!("'{}' -> {:?}", sentence, token_strs);
    }

    // Count non-## tokens and check bucket distribution
    let mut non_cont_count = 0;
    for (token, _) in token_to_id.iter() {
        if !token.starts_with("##") {
            non_cont_count += 1;
        }
    }
    println!("\nNon-continuation tokens: {}", non_cont_count);
    println!("Hash table size: {} buckets (14 bits)", 1 << 14);
    println!("Average tokens per bucket: {:.2}", non_cont_count as f64 / (1 << 14) as f64);

    // Debug: Check hash values for problematic tokens
    println!("\nHash debug:");
    let problematic_tokens = ["street", "fires", "justified", "57", "40", "b", "z", "d"];
    for token in problematic_tokens {
        if let Some(&id) = token_to_id.get(token) {
            // FNV-1a hash
            let h = fast_hash(token.as_bytes());
            let bucket_idx = (h & ((1u64 << 14) - 1)) as usize;
            println!("  '{}' (id={}, len={}) -> hash={:016x} -> bucket {}", token, id, token.len(), h, bucket_idx);
        }
    }

    // Find all tokens that hash to bucket 9022
    println!("\nAll tokens hashing to bucket 9022:");
    let target_bucket = 9022;
    let mask = (1u64 << 14) - 1;
    for (token, &id) in &token_to_id {
        if token.starts_with("##") {
            continue; // Skip continuation tokens
        }
        let h = fast_hash(token.as_bytes());
        let bucket_idx = (h & mask) as usize;
        if bucket_idx == target_bucket {
            println!("  '{}' (id={}, len={}) -> hash={:016x}", token, id, token.len(), h);
        }
    }

    // Check bucket 9023 too (the overflow bucket)
    println!("\nAll tokens hashing to bucket 9023:");
    let target_bucket = 9023;
    for (token, &id) in &token_to_id {
        if token.starts_with("##") {
            continue;
        }
        let h = fast_hash(token.as_bytes());
        let bucket_idx = (h & mask) as usize;
        if bucket_idx == target_bucket {
            println!("  '{}' (id={}, len={}) -> hash={:016x}", token, id, token.len(), h);
        }
    }
}

fn fast_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
