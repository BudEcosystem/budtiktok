//! Debug hash lookup

use std::fs;
use ahash::AHashMap;

use budtiktok_core::wordpiece_hyper::{HyperConfig, HyperWordPieceTokenizer};
use budtiktok_core::vocab::{Vocabulary, SpecialTokens};

const WORKSPACE: &str = "/home/bud/Desktop/latentbud/budtiktok";

fn fast_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

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

    let special_tokens = SpecialTokens {
        unk_token: Some("[UNK]".to_string()),
        cls_token: Some("[CLS]".to_string()),
        sep_token: Some("[SEP]".to_string()),
        pad_token: Some("[PAD]".to_string()),
        mask_token: Some("[MASK]".to_string()),
        ..Default::default()
    };
    let vocabulary = Vocabulary::new(token_to_id.clone(), special_tokens.clone());

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

    // Test the exact hex string
    let hex_string = "57ae463e6bc53a38512c58a878370338dcfe0fb59eeedfd9b3e7959fe7c149d1";
    println!("Testing: {}", hex_string);
    println!("Length: {} chars", hex_string.len());

    // Compute incremental hashes like speculative_hash_lookup does
    let bytes = hex_string.as_bytes();
    let max_try = bytes.len().min(24);
    println!("\nIncremental hashes (max_try={}):", max_try);

    let mut h: u64 = 0xcbf29ce484222325;
    for i in 0..max_try {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(0x100000001b3);

        let next_pos = i + 1;
        let is_boundary = next_pos >= bytes.len() || (bytes[next_pos] & 0xC0) != 0x80;

        if is_boundary {
            let prefix = std::str::from_utf8(&bytes[..next_pos]).unwrap();
            let bucket = (h & ((1u64 << 14) - 1)) as usize;
            let in_vocab = token_to_id.get(prefix).is_some();
            println!("  len={:2} prefix='{}' hash={:016x} bucket={:5} in_vocab={}",
                     next_pos, &prefix[..prefix.len().min(20)], h, bucket, in_vocab);
        }
    }

    // Check what "57" hash should be
    let hash_57 = fast_hash("57".as_bytes());
    let bucket_57 = (hash_57 & ((1u64 << 14) - 1)) as usize;
    println!("\nDirect hash of '57': {:016x} bucket={}", hash_57, bucket_57);

    // Check all tokens hashing to bucket 2021
    println!("\nTokens hashing to bucket {} (and adjacent):", bucket_57);
    let mask = (1u64 << 14) - 1;
    for (token, &id) in &token_to_id {
        if token.starts_with("##") {
            continue; // Skip continuation tokens - they're in a different table
        }
        let h = fast_hash(token.as_bytes());
        let bucket = (h & mask) as usize;
        if bucket >= bucket_57 && bucket <= bucket_57 + 2 {
            println!("  '{}' (id={}, len={}) -> bucket {} hash={:016x}", token, id, token.len(), bucket, h);
        }
    }

    // Also check what's actually in the hash table bucket
    // We need to access the tokenizer's internal hash table
    // Since it's private, let's verify by checking specific lookups
    println!("\nDirect lookup tests:");
    for test in &["5", "57", "goofy", "chihuahua"] {
        let result = simd_tokenizer.encode_fast(test);
        let strs: Vec<String> = result.iter()
            .map(|&id| vocabulary.id_to_token(id).unwrap_or("[?]").to_string())
            .collect();
        println!("  '{}' -> {:?}", test, strs);
    }

    // Tokenize and see result
    let result = simd_tokenizer.encode_fast(hex_string);
    let token_strs: Vec<String> = result.iter()
        .map(|&id| vocabulary.id_to_token(id).unwrap_or("[?]").to_string())
        .collect();
    println!("\nTokenization result: {:?}", token_strs);
    println!("IDs: {:?}", result);

    // Also test a shorter hex string
    let short_hex = "57ae463e";
    let short_result = simd_tokenizer.encode_fast(short_hex);
    let short_strs: Vec<String> = short_result.iter()
        .map(|&id| vocabulary.id_to_token(id).unwrap_or("[?]").to_string())
        .collect();
    println!("\nShort test '{}': {:?}", short_hex, short_strs);

    // Test continuation tokens
    println!("\nContinuation token tests:");
    // When looking up continuation, we strip "##" and lookup the suffix
    for test in &["ae", "46", "3", "e", "6", "bc", "53", "a", "38"] {
        // The continuation lookup searches for "##ae" etc.
        let cont_token = format!("##{}", test);
        if let Some(&id) = token_to_id.get(&cont_token) {
            println!("  '{}' exists in vocab (id={})", cont_token, id);
        } else {
            println!("  '{}' NOT in vocab!", cont_token);
        }
    }

    // Also test incrementally
    println!("\nIncremental tokenization test:");
    let prefixes = ["5", "57", "57a", "57ae", "57ae4", "57ae46", "57ae463", "57ae463e"];
    for prefix in &prefixes {
        let result = simd_tokenizer.encode_fast(prefix);
        let strs: Vec<String> = result.iter()
            .map(|&id| vocabulary.id_to_token(id).unwrap_or("[?]").to_string())
            .collect();
        println!("  '{}' -> {:?}", prefix, strs);
    }

    // Test word lengths around the issue
    println!("\nLength tests (finding boundary):");
    for len in 40..=55 {
        let test_str = &hex_string[..len];
        let result = simd_tokenizer.encode_fast(test_str);
        let unk_count = result.iter().filter(|&&id| id == 100).count();
        let marker = if unk_count > 0 { " <-- FAILS" } else { "" };
        println!("  len={}: tokens={}, UNKs={}{}", len, result.len(), unk_count, marker);
    }
}
