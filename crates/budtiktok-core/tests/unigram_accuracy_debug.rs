//! Debug test for Unigram tokenization accuracy issues

use budtiktok_core::unigram::UnigramPiece;
use budtiktok_core::unigram_fast::{UnigramFast, UnigramFastConfig, preprocess_text};
use budtiktok_core::vocab::{Vocabulary, SpecialTokens};
use ahash::AHashMap;
use tokenizers::Tokenizer as HfTokenizer;

const XLNET_PATH: &str = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/xlnet/tokenizer.json";

fn load_unigram_fast(path: &str) -> Option<UnigramFast> {
    let json_content = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&json_content).ok()?;

    let model = json.get("model")?;
    let vocab_array = model.get("vocab")?.as_array()?;

    let mut vocab = Vec::with_capacity(vocab_array.len());
    let mut pieces = Vec::with_capacity(vocab_array.len());

    for item in vocab_array {
        let arr = item.as_array()?;
        let token = arr.get(0)?.as_str()?;
        let score = arr.get(1)?.as_f64()?;
        vocab.push(token.to_string());
        pieces.push(UnigramPiece {
            token: token.to_string(),
            score,
        });
    }

    let unk_id = model.get("unk_id")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(0);

    let config = UnigramFastConfig {
        unk_id,
        add_prefix_space: true,
        ..Default::default()
    };

    let mut vocab_map = AHashMap::with_capacity(vocab.len());
    for (i, token) in vocab.iter().enumerate() {
        vocab_map.insert(token.clone(), i as u32);
    }
    let vocabulary = Vocabulary::new(vocab_map, SpecialTokens {
        unk_token: model.get("unk_token").and_then(|v| v.as_str()).map(|s| s.to_string()),
        ..Default::default()
    });

    Some(UnigramFast::new(vocabulary, pieces, config))
}

#[test]
fn debug_tokenization_differences() {
    if !std::path::Path::new(XLNET_PATH).exists() {
        println!("Skipping: tokenizer not found");
        return;
    }

    let hf_tok = HfTokenizer::from_file(XLNET_PATH).expect("Failed to load HF tokenizer");
    let fast_tok = load_unigram_fast(XLNET_PATH).expect("Failed to load UnigramFast");

    // Check what specific tokens map to
    println!("=== Token ID Mappings ===");
    for id in [0, 17, 22, 24, 101, 133, 136, 10103, 11368, 17211] {
        let hf_token = hf_tok.id_to_token(id);
        let fast_token = fast_tok.id_to_token(id);
        println!("ID {}: HF={:?}, Fast={:?}", id, hf_token, fast_token);
    }

    // Test a simple case
    let text = "Hello";
    println!("\n=== Encoding '{}' ===", text);

    let hf_enc = hf_tok.encode(text, false).expect("HF encode failed");
    let fast_ids = fast_tok.encode_fast(text);

    println!("HF IDs: {:?}", hf_enc.get_ids());
    println!("HF tokens: {:?}", hf_enc.get_tokens());
    println!("Fast IDs: {:?}", fast_ids);

    let fast_tokens: Vec<_> = fast_ids.iter()
        .filter_map(|&id| fast_tok.id_to_token(id))
        .collect();
    println!("Fast tokens: {:?}", fast_tokens);

    // Check preprocessing
    println!("\n=== Preprocessing Debug ===");
    let preprocessed = preprocess_text(text, true);
    println!("Input: '{}'", text);
    println!("Preprocessed bytes: {:?}", preprocessed);
    println!("Preprocessed str: '{}'", String::from_utf8_lossy(&preprocessed));

    // Check what ▁Hello maps to
    println!("\n=== Token Lookups ===");
    let test_tokens = ["▁Hello", "Hello", "▁", "▁hello", "hello"];
    for token in test_tokens {
        let hf_id = hf_tok.token_to_id(token);
        let fast_id = fast_tok.token_to_id(token);
        println!("'{}': HF={:?}, Fast={:?}", token, hf_id, fast_id);
    }

    // Debug: Check the trie lookup directly
    println!("\n=== Trie Direct Lookup ===");
    let test_bytes = ["▁Hello", "Hello", "▁hello", "hello"];
    for token in test_bytes {
        let trie_id = fast_tok.trie_lookup(token.as_bytes());
        println!("Trie lookup '{}': {:?}", token, trie_id);
    }
}

#[test]
fn debug_complex_tokenization() {
    if !std::path::Path::new(XLNET_PATH).exists() {
        println!("Skipping: tokenizer not found");
        return;
    }

    let hf_tok = HfTokenizer::from_file(XLNET_PATH).expect("Failed to load HF tokenizer");
    let fast_tok = load_unigram_fast(XLNET_PATH).expect("Failed to load UnigramFast");

    let text = "Hello, world!";
    println!("=== Encoding '{}' ===", text);

    let hf_enc = hf_tok.encode(text, false).expect("HF encode failed");
    let fast_ids = fast_tok.encode_fast(text);

    println!("HF IDs: {:?}", hf_enc.get_ids());
    println!("HF tokens: {:?}", hf_enc.get_tokens());
    println!("Fast IDs: {:?}", fast_ids);

    let fast_tokens: Vec<_> = fast_ids.iter()
        .filter_map(|&id| fast_tok.id_to_token(id))
        .collect();
    println!("Fast tokens: {:?}", fast_tokens);

    // Check preprocessing
    let preprocessed = preprocess_text(text, true);
    println!("\nPreprocessed: '{}'", String::from_utf8_lossy(&preprocessed));

    // Trace through token by token
    println!("\n=== Token by Token Analysis ===");
    for (i, hf_id) in hf_enc.get_ids().iter().enumerate() {
        let hf_token = hf_tok.id_to_token(*hf_id);
        println!("Position {}: HF ID {} = {:?}", i, hf_id, hf_token);
    }

    println!("\nFast tokens:");
    for (i, &fast_id) in fast_ids.iter().enumerate() {
        let fast_token = fast_tok.id_to_token(fast_id);
        println!("Position {}: Fast ID {} = {:?}", i, fast_id, fast_token);
    }
}

#[test]
fn debug_normalizer_check() {
    if !std::path::Path::new(XLNET_PATH).exists() {
        println!("Skipping: tokenizer not found");
        return;
    }

    // Load the tokenizer.json and check what normalizer/pre_tokenizer is configured
    let json_content = std::fs::read_to_string(XLNET_PATH).expect("Failed to read file");
    let json: serde_json::Value = serde_json::from_str(&json_content).expect("Failed to parse JSON");

    println!("=== Tokenizer Configuration ===");

    if let Some(normalizer) = json.get("normalizer") {
        println!("Normalizer: {}", serde_json::to_string_pretty(normalizer).unwrap());
    } else {
        println!("Normalizer: None");
    }

    if let Some(pre_tokenizer) = json.get("pre_tokenizer") {
        println!("\nPre-tokenizer: {}", serde_json::to_string_pretty(pre_tokenizer).unwrap());
    } else {
        println!("\nPre-tokenizer: None");
    }

    if let Some(model) = json.get("model") {
        if let Some(model_type) = model.get("type") {
            println!("\nModel type: {}", model_type);
        }
    }
}
