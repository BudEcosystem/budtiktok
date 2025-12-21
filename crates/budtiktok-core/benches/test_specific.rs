//! Test specific problematic texts

use budtiktok_core::unigram::UnigramPiece;
use budtiktok_core::unigram_fast::{UnigramFast, UnigramFastConfig, normalize_for_hf};
use tokenizers::Tokenizer as HfTokenizer;

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

    let bt_tok = UnigramFast::new(vocab, pieces, config);

    // Read test text
    let text = std::fs::read_to_string("/tmp/test_text_423.txt").expect("Read test file");
    println!("Text length: {} chars", text.len());

    // Normalize
    let normalized = normalize_for_hf(&text);
    println!("Normalized length: {} chars", normalized.len());

    // Get tokens
    let hf_enc = hf_tok.encode(text.clone(), false).unwrap();
    let hf_ids: Vec<u32> = hf_enc.get_ids().to_vec();
    let bt_ids = bt_tok.encode_hf_compat_simd_v2(&text);

    println!("HF tokens: {}", hf_ids.len());
    println!("BT tokens: {}", bt_ids.len());

    if hf_ids == bt_ids {
        println!("\n✓ MATCH!");
        return;
    }

    // Find first difference
    let mut first_diff = None;
    for (i, (h, b)) in hf_ids.iter().zip(bt_ids.iter()).enumerate() {
        if h != b {
            first_diff = Some(i);
            break;
        }
    }

    if let Some(pos) = first_diff {
        println!("\n✗ First token diff at position {}", pos);

        // Show context
        let start = pos.saturating_sub(5);
        let end = (pos + 10).min(hf_ids.len()).min(bt_ids.len());

        println!("HF tokens [{}-{}]: {:?}", start, end, &hf_ids[start..end.min(hf_ids.len())]);
        println!("BT tokens [{}-{}]: {:?}", start, end.min(bt_ids.len()), &bt_ids[start..end.min(bt_ids.len())]);

        // Decode around the difference
        println!("\nDecoded HF tokens around diff:");
        for i in start..end.min(hf_ids.len()) {
            let marker = if i == pos { ">>>" } else { "   " };
            if let Ok(s) = hf_tok.decode(&[hf_ids[i]], false) {
                println!("{} [{}] {} = {:?}", marker, i, hf_ids[i], s);
            }
        }
        println!("\nDecoded BT tokens around diff:");
        for i in start..end.min(bt_ids.len()) {
            let marker = if i == pos { ">>>" } else { "   " };
            if let Some(s) = bt_tok.id_to_token(bt_ids[i]) {
                println!("{} [{}] {} = {:?}", marker, i, bt_ids[i], s);
            }
        }
    } else if hf_ids.len() != bt_ids.len() {
        println!("\n✗ Length mismatch: HF={} BT={}", hf_ids.len(), bt_ids.len());

        let min_len = hf_ids.len().min(bt_ids.len());
        println!("\nAll {} common tokens match!", min_len);

        // Show the extra tokens
        if hf_ids.len() > bt_ids.len() {
            println!("\nHF has {} extra tokens at end:", hf_ids.len() - min_len);
            for i in min_len..hf_ids.len() {
                if let Ok(s) = hf_tok.decode(&[hf_ids[i]], false) {
                    println!("  [{}] {} = {:?}", i, hf_ids[i], s);
                }
            }
            // Show what the last few common tokens were
            println!("\nLast 5 common tokens:");
            for i in min_len.saturating_sub(5)..min_len {
                if let Ok(s) = hf_tok.decode(&[hf_ids[i]], false) {
                    println!("  [{}] {} = {:?}", i, hf_ids[i], s);
                }
            }
        } else {
            println!("\nBT has {} extra tokens at end:", bt_ids.len() - min_len);
            for i in min_len..bt_ids.len() {
                if let Some(s) = bt_tok.id_to_token(bt_ids[i]) {
                    println!("  [{}] {} = {:?}", i, bt_ids[i], s);
                }
            }
        }
    }
}
