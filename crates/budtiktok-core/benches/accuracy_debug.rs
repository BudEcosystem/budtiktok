//! Quick accuracy debug script

use std::fs::File;
use std::io::{BufRead, BufReader};

use budtiktok_core::unigram::UnigramPiece;
use budtiktok_core::unigram_fast::{UnigramFast, UnigramFastConfig};
use budtiktok_core::vocab::{Vocabulary, SpecialTokens};
use tokenizers::Tokenizer as HfTokenizer;
use serde::Deserialize;
use ahash::AHashMap;

const DATASET_PATH: &str = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/openwebtext_1gb.jsonl";
const XLNET_PATH: &str = "/home/bud/Desktop/latentbud/budtiktok/benchmark_data/xlnet/tokenizer.json";

#[derive(Deserialize)]
struct TextRecord { text: String }

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

    println!("Loading texts...");
    let file = File::open(DATASET_PATH).unwrap();
    let reader = BufReader::new(file);

    let texts: Vec<String> = reader.lines().take(10000).filter_map(|line| {
        line.ok().and_then(|l| serde_json::from_str::<TextRecord>(&l).ok().map(|r| r.text))
    }).collect();

    println!("Checking {} texts...", texts.len());

    let mut matches = 0;
    let mut mismatches = 0;

    for (idx, text) in texts.iter().enumerate() {
        let hf_enc = hf_tok.encode(text.as_str(), false).unwrap();
        let hf_ids: Vec<u32> = hf_enc.get_ids().to_vec();
        let bt_ids = bt_tok.encode_hf_compat_simd_v2(text);

        if hf_ids == bt_ids {
            matches += 1;
        } else {
            mismatches += 1;

            if mismatches <= 5 {
                println!("\n=== MISMATCH #{} (text {}) ===", mismatches, idx);

                // Find first diff
                let mut first_diff = None;
                for (i, (h, b)) in hf_ids.iter().zip(bt_ids.iter()).enumerate() {
                    if h != b {
                        first_diff = Some(i);
                        break;
                    }
                }

                if let Some(pos) = first_diff {
                    println!("First diff at position {}", pos);

                    // Try to find the text at this position
                    // Decode tokens before and at the difference
                    let context_start = pos.saturating_sub(3);
                    let context_end = (pos + 3).min(hf_ids.len());

                    if let Ok(hf_decoded) = hf_tok.decode(&hf_ids[context_start..context_end], false) {
                        println!("HF decoded context: {:?}", hf_decoded);
                    }

                    // Show the specific differing tokens
                    let hf_token_decoded = hf_tok.decode(&[hf_ids[pos]], false).ok();
                    let bt_token = bt_tok.id_to_token(bt_ids[pos]);
                    println!("HF token at {}: id={} text={:?}", pos, hf_ids[pos], hf_token_decoded);
                    println!("BT token at {}: id={} text={:?}", pos, bt_ids[pos], bt_token);

                    // Show context tokens
                    println!("HF ids [{}-{}]: {:?}", context_start, context_end, &hf_ids[context_start..context_end]);
                    println!("BT ids [{}-{}]: {:?}", context_start, context_end.min(bt_ids.len()), &bt_ids[context_start..context_end.min(bt_ids.len())]);
                } else if hf_ids.len() != bt_ids.len() {
                    println!("Length mismatch only: HF={} BT={}", hf_ids.len(), bt_ids.len());
                    let min_len = hf_ids.len().min(bt_ids.len());
                    if min_len > 0 {
                        if let Ok(hf_end) = hf_tok.decode(&hf_ids[min_len-1..min_len], false) {
                            println!("HF last common token: {:?}", hf_end);
                        }
                    }
                    if hf_ids.len() > bt_ids.len() {
                        if let Ok(extra) = hf_tok.decode(&hf_ids[min_len..min_len+3.min(hf_ids.len()-min_len)], false) {
                            println!("HF extra tokens: {:?}", extra);
                        }
                    }
                }

                // Show text snippet around where we might expect the problem
                if text.len() <= 200 {
                    println!("Full text: {:?}", text);
                } else {
                    println!("Text start: {:?}...", &text[..100.min(text.len())]);
                }
            }
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));
    println!("Matches:    {}", matches);
    println!("Mismatches: {}", mismatches);
    println!("Accuracy:   {:.2}%", matches as f64 / texts.len() as f64 * 100.0);
}
