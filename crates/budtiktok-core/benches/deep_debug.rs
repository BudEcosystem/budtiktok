//! Deep debugging of tokenization mismatches

use budtiktok_core::unigram::UnigramPiece;
use budtiktok_core::unigram_fast::{UnigramFast, UnigramFastConfig, normalize_for_hf, preprocess_text};
use budtiktok_core::vocab::{Vocabulary, SpecialTokens};
use tokenizers::Tokenizer as HfTokenizer;
use ahash::AHashMap;

const XLNET_PATH: &str = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/xlnet/tokenizer.json";

fn main() {
    println!("Loading tokenizers...");
    let hf_tok = HfTokenizer::from_file(XLNET_PATH).expect("HF tokenizer");

    let json_content = std::fs::read_to_string(XLNET_PATH).unwrap();
    let json: serde_json::Value = serde_json::from_str(&json_content).unwrap();
    let model = json.get("model").unwrap();
    let vocab_array = model.get("vocab").unwrap().as_array().unwrap();

    let mut vocab = Vec::with_capacity(vocab_array.len());
    let mut pieces = Vec::with_capacity(vocab_array.len());

    for item in vocab_array {
        let arr = item.as_array().unwrap();
        let token = arr[0].as_str().unwrap();
        let score = arr[1].as_f64().unwrap();
        vocab.push(token.to_string());
        pieces.push(UnigramPiece { token: token.to_string(), score });
    }

    let unk_id = model.get("unk_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let config = UnigramFastConfig {
        unk_id,
        add_prefix_space: true,
        fuse_unk: true,
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

    let bt_tok = UnigramFast::new(vocabulary, pieces, config);

    // Test cases that were mismatching
    let test_cases: [&str; 7] = [
        // Case 1: Zero-width space at start
        "\u{200b}Thor, the classic God of Thunder",
        // Case 2: Zero-width space with quote
        "\u{200b}\"We wear white to unite",
        // Case 3: Simple text (should match)
        "Hello world",
        // Case 4: Text with curly quotes (U+201C and U+201D)
        "He said \u{201C}hello\u{201D} to me",
        // Case 5: Text with apostrophe
        "I can't believe it's true",
        // Case 6: Arabic with quotes
        "\u{0645}\u{0631}\u{062D}\u{0628}\u{0627} \"test\" world",
        // Case 7: Text 423 - full first paragraph with double newlines
        "\u{200b}Thor, the classic God of Thunder in the Marvel comics, exploded onto the screen in 2011's \"Thor\".\n\n",
    ];

    for (i, text) in test_cases.iter().enumerate() {
        println!("\n{}", "=".repeat(70));
        println!("TEST CASE {} - {:?}", i + 1, &text[..text.len().min(50)]);
        println!("{}", "=".repeat(70));

        // Show character bytes
        println!("\nRaw bytes (hex):");
        for (j, b) in text.bytes().take(50).enumerate() {
            print!("{:02x} ", b);
            if (j + 1) % 20 == 0 { println!(); }
        }
        println!();

        // Show what normalize_for_hf produces
        let normalized = normalize_for_hf(text);
        println!("\nAfter normalize_for_hf:");
        println!("  Text: {:?}", &normalized[..normalized.len().min(100)]);
        println!("  Bytes: ");
        for (j, b) in normalized.bytes().take(50).enumerate() {
            print!("{:02x} ", b);
            if (j + 1) % 20 == 0 { println!(); }
        }
        println!();

        // Show what preprocess_text produces
        let preprocessed = preprocess_text(&normalized, true);
        println!("\nAfter preprocess_text (add_prefix_space=true):");
        println!("  Text: {:?}", &preprocessed[..preprocessed.len().min(100)]);

        // Get HF tokens
        let hf_enc = hf_tok.encode(text.to_string(), false).unwrap();
        let hf_ids: Vec<u32> = hf_enc.get_ids().to_vec();

        // Get BT tokens
        let bt_ids = bt_tok.encode_hf_compat_simd_v2(text);

        println!("\nHF token count: {}", hf_ids.len());
        println!("BT token count: {}", bt_ids.len());

        // Find first difference
        let mut first_diff = None;
        for (j, (h, b)) in hf_ids.iter().zip(bt_ids.iter()).enumerate() {
            if h != b {
                first_diff = Some(j);
                break;
            }
        }

        if hf_ids == bt_ids {
            println!("\n✓ MATCH!");
        } else if let Some(pos) = first_diff {
            println!("\n✗ MISMATCH at position {}", pos);

            // Show context
            let start = pos.saturating_sub(3);
            let end = (pos + 5).min(hf_ids.len()).min(bt_ids.len());

            println!("  HF tokens [{}-{}]: {:?}", start, end, &hf_ids[start..end.min(hf_ids.len())]);
            println!("  BT tokens [{}-{}]: {:?}", start, end.min(bt_ids.len()), &bt_ids[start..end.min(bt_ids.len())]);

            // Decode the specific tokens
            if let Ok(hf_decoded) = hf_tok.decode(&[hf_ids[pos]], false) {
                println!("  HF token {} = {:?}", hf_ids[pos], hf_decoded);
            }
            if let Some(bt_decoded) = bt_tok.id_to_token(bt_ids[pos]) {
                println!("  BT token {} = {:?}", bt_ids[pos], bt_decoded);
            }
        } else {
            println!("\n✗ LENGTH MISMATCH: HF={} BT={}", hf_ids.len(), bt_ids.len());

            // Show end of sequences
            let min_len = hf_ids.len().min(bt_ids.len());
            if min_len > 0 {
                println!("  Last common position: {}", min_len - 1);
                if hf_ids.len() > bt_ids.len() {
                    println!("  HF extra tokens: {:?}", &hf_ids[min_len..min_len+5.min(hf_ids.len()-min_len)]);
                } else {
                    println!("  BT extra tokens: {:?}", &bt_ids[min_len..min_len+5.min(bt_ids.len()-min_len)]);
                }
            }
        }

        // Show first 10 tokens decoded
        println!("\nFirst 10 tokens decoded:");
        print!("  HF: ");
        for id in hf_ids.iter().take(10) {
            if let Ok(s) = hf_tok.decode(&[*id], false) {
                print!("[{}:{}] ", id, s);
            }
        }
        println!();
        print!("  BT: ");
        for id in bt_ids.iter().take(10) {
            if let Some(s) = bt_tok.id_to_token(*id) {
                print!("[{}:{}] ", id, s);
            }
        }
        println!();
    }
}
