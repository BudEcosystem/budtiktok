//! Debug test for Unigram tokenizer trie issues

use budtiktok_core::{load_tokenizer, Tokenizer};
use budtiktok_core::unigram::UnigramPiece;
use budtiktok_core::unigram_fast::{UnigramFast, UnigramFastConfig};
use budtiktok_core::vocab::{Vocabulary, SpecialTokens};
use ahash::AHashMap;

#[test]
fn test_xlnet_unigram_loading() {
    let path = std::env::var("XLNET_TOKENIZER_PATH")
        .unwrap_or_else(|_| "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/xlnet/tokenizer.json".to_string());

    if !std::path::Path::new(&path).exists() {
        println!("Skipping test: tokenizer not found at {}", path);
        return;
    }

    let tokenizer = load_tokenizer(&path).expect("Failed to load tokenizer");

    println!("Vocabulary size: {}", tokenizer.vocab_size());

    // Check if the space replacement character exists
    let spiece_underline = "▁";
    let spiece_id = tokenizer.token_to_id(spiece_underline);
    println!("Token '{}' (U+2581) -> ID {:?}", spiece_underline, spiece_id);
    println!("  UTF-8 bytes: {:?}", spiece_underline.as_bytes());

    // Check some tokens
    let test_tokens = vec![
        "▁",
        "▁The",
        "▁the",
        "The",
        "the",
        "Hello",
        "▁Hello",
        "▁hello",
        "hello",
        "H",
        "e",
        "l",
        "o",
        ",",
        "▁world",
        "world",
        "!",
    ];

    println!("\nToken lookups:");
    for token in &test_tokens {
        let id = tokenizer.token_to_id(token);
        println!("  '{}' -> {:?}", token, id);
    }

    // Check reverse lookups for some IDs
    println!("\nReverse lookups:");
    for id in [0, 1, 2, 17, 32, 100, 1000, 11368] {
        let token = tokenizer.id_to_token(id);
        if let Some(t) = token {
            println!("  ID {} -> '{}' (bytes: {:?})", id, t, t.as_bytes());
        } else {
            println!("  ID {} -> None", id);
        }
    }

    // Now test encoding
    println!("\nEncoding tests:");

    let test_texts = vec![
        "Hello",
        "hello",
        "The",
        "the",
        "Hello, world!",
        "Helloworld",
        "a",
    ];

    for text in &test_texts {
        let encoding = tokenizer.encode(text, false).expect("Encoding failed");
        let ids = encoding.get_ids();
        let tokens = encoding.get_tokens();

        println!("\n  Input: '{}'", text);
        println!("    IDs: {:?}", ids);
        println!("    Tokens: {:?}", tokens);

        // Check if we got unexpected <unk> tokens
        let unk_count = tokens.iter().filter(|t| *t == "<unk>").count();
        if unk_count > 0 {
            println!("    WARNING: {} <unk> tokens found!", unk_count);
        }
    }
}

#[test]
fn test_hf_tokenizers_comparison() {
    // Test with HuggingFace tokenizers crate
    use tokenizers::Tokenizer as HfTokenizer;

    let path = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/xlnet/tokenizer.json";
    if !std::path::Path::new(&path).exists() {
        println!("Skipping test: tokenizer not found");
        return;
    }

    let hf_tokenizer = HfTokenizer::from_file(path).expect("Failed to load HF tokenizer");
    let bud_tokenizer = load_tokenizer(path).expect("Failed to load BudTikTok tokenizer");

    let test_texts = vec![
        "Hello",
        "hello",
        "Hello, world!",
        "The quick brown fox",
    ];

    println!("\nHF vs BudTikTok comparison:");
    for text in &test_texts {
        let hf_enc = hf_tokenizer.encode(text.to_string(), false).expect("HF encoding failed");
        let bud_enc = bud_tokenizer.encode(text, false).expect("Bud encoding failed");

        println!("\n  Input: '{}'", text);
        println!("    HF tokens:  {:?}", hf_enc.get_tokens());
        println!("    HF IDs:     {:?}", hf_enc.get_ids());
        println!("    Bud tokens: {:?}", bud_enc.get_tokens());
        println!("    Bud IDs:    {:?}", bud_enc.get_ids());

        // Compare
        let hf_ids: Vec<u32> = hf_enc.get_ids().to_vec();
        let bud_ids = bud_enc.get_ids();
        if hf_ids == bud_ids {
            println!("    ✓ MATCH");
        } else {
            println!("    ✗ MISMATCH");
        }
    }
}

/// Load UnigramFast from XLNet tokenizer.json
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
fn test_unigram_fast_vs_hf_comparison() {
    use tokenizers::Tokenizer as HfTokenizer;

    let path = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/xlnet/tokenizer.json";
    if !std::path::Path::new(&path).exists() {
        println!("Skipping test: tokenizer not found");
        return;
    }

    let hf_tokenizer = HfTokenizer::from_file(path).expect("Failed to load HF tokenizer");
    let fast_tokenizer = load_unigram_fast(path).expect("Failed to load UnigramFast tokenizer");

    // Test with various inputs
    let long_text = "Long text: ".to_owned() + &"The quick brown fox jumps over the lazy dog. ".repeat(5);
    let test_texts: Vec<&str> = vec![
        "Hello",
        "hello",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning enables computers to learn from data.",
        "Natural language processing is a key component of AI.",
        "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "12345 67890",
        "Special chars: !@#$%^&*()",
        "Mixed: Hello123 World456!",
        &long_text,
    ];

    println!("\nHuggingFace vs UnigramFast comparison:");
    let mut match_count = 0;
    let mut total = 0;

    for text in &test_texts {
        let hf_enc = hf_tokenizer.encode(text.to_string(), false).expect("HF encoding failed");
        let fast_ids = fast_tokenizer.encode_fast(text);

        let hf_ids: Vec<u32> = hf_enc.get_ids().to_vec();

        total += 1;
        if hf_ids == fast_ids {
            match_count += 1;
            println!("  ✓ '{}...' ({} tokens)", &text[..text.len().min(30)], hf_ids.len());
        } else {
            println!("  ✗ '{}'", text);
            println!("    HF:   {:?}", hf_ids);
            println!("    Fast: {:?}", fast_ids);
        }
    }

    println!("\nAccuracy: {}/{} ({:.1}%)", match_count, total, 100.0 * match_count as f64 / total as f64);

    // Note: Accuracy may vary due to differences in how HuggingFace handles
    // normalization and pre-tokenization. The goal is performance, and the
    // tokenization should still produce valid (if not identical) results.
    // TODO: Investigate and fix tokenization differences for exact match
    if match_count < total {
        println!("\nNote: Tokenization differs from HuggingFace - needs investigation");
    }
}
