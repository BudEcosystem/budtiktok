//! Detailed analysis of remaining accuracy issues

use std::fs::File;
use std::io::{BufRead, BufReader};

use budtiktok_core::unigram::UnigramPiece;
use budtiktok_core::unigram_fast::{UnigramFast, UnigramFastConfig, normalize_for_hf, preprocess_text};
use tokenizers::Tokenizer as HfTokenizer;
use serde::Deserialize;

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
    let mut scores: Vec<f64> = Vec::with_capacity(vocab_array.len());

    for item in vocab_array {
        let arr = item.as_array().unwrap();
        let token = arr[0].as_str().unwrap();
        let score = arr[1].as_f64().unwrap();
        vocab.push(token.to_string());
        scores.push(score);
        pieces.push(UnigramPiece { token: token.to_string(), score });
    }

    let unk_id = model.get("unk_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let config = UnigramFastConfig {
        unk_id,
        add_prefix_space: true,
        fuse_unk: true,
        ..Default::default()
    };

    let bt_tok = UnigramFast::new(vocab.clone(), pieces, config);

    // Load texts and find mismatches
    println!("Loading texts...");
    let file = File::open(DATASET_PATH).unwrap();
    let reader = BufReader::new(file);

    let texts: Vec<String> = reader.lines().take(10000).filter_map(|line| {
        line.ok().and_then(|l| serde_json::from_str::<TextRecord>(&l).ok().map(|r| r.text))
    }).collect();

    println!("Analyzing {} texts for mismatches...\n", texts.len());

    let mut mismatch_count = 0;

    for (idx, text) in texts.iter().enumerate() {
        let hf_enc = hf_tok.encode(text.as_str(), false).unwrap();
        let hf_ids: Vec<u32> = hf_enc.get_ids().to_vec();
        let bt_ids = bt_tok.encode_hf_compat_simd_v2(text);

        if hf_ids != bt_ids {
            mismatch_count += 1;

            if mismatch_count <= 10 {
                println!("{}", "=".repeat(80));
                println!("MISMATCH #{} (text index {})", mismatch_count, idx);
                println!("{}", "=".repeat(80));

                // Find first difference
                let mut first_diff = None;
                for (i, (h, b)) in hf_ids.iter().zip(bt_ids.iter()).enumerate() {
                    if h != b {
                        first_diff = Some(i);
                        break;
                    }
                }

                if let Some(pos) = first_diff {
                    println!("\nFirst difference at position {}", pos);

                    // Get the text that was tokenized around this position
                    // Decode tokens before and after
                    let start = pos.saturating_sub(5);
                    let end = (pos + 10).min(hf_ids.len()).min(bt_ids.len());

                    println!("\nHF tokens around diff:");
                    for i in start..end.min(hf_ids.len()) {
                        let marker = if i == pos { ">>> " } else { "    " };
                        let token_str = hf_tok.decode(&[hf_ids[i]], false).unwrap_or_default();
                        let score = if (hf_ids[i] as usize) < scores.len() {
                            scores[hf_ids[i] as usize]
                        } else {
                            0.0
                        };
                        println!("{}[{}] id={:<6} score={:>10.4} text={:?}",
                                 marker, i, hf_ids[i], score, token_str);
                    }

                    println!("\nBT tokens around diff:");
                    for i in start..end.min(bt_ids.len()) {
                        let marker = if i == pos { ">>> " } else { "    " };
                        let token_str = bt_tok.id_to_token(bt_ids[i]).unwrap_or_default();
                        let score = if (bt_ids[i] as usize) < scores.len() {
                            scores[bt_ids[i] as usize]
                        } else {
                            0.0
                        };
                        println!("{}[{}] id={:<6} score={:>10.4} text={:?}",
                                 marker, i, bt_ids[i], score, token_str);
                    }

                    // Show what the normalized and preprocessed text looks like
                    let normalized = normalize_for_hf(text);
                    let preprocessed = preprocess_text(&normalized, true);

                    // Try to find the relevant part of the text
                    // Decode a range of HF tokens to get approximate text position
                    if pos > 0 {
                        let context_tokens: Vec<u32> = hf_ids[start..pos+3.min(hf_ids.len()-pos)]
                            .to_vec();
                        if let Ok(context) = hf_tok.decode(&context_tokens, false) {
                            println!("\nContext text (from HF decode): {:?}", context);
                        }
                    }

                    // Check if this is a Viterbi tie-breaking issue
                    // Compare scores of the different paths
                    let hf_token = hf_ids[pos];
                    let bt_token = bt_ids[pos];
                    let hf_score = if (hf_token as usize) < scores.len() { scores[hf_token as usize] } else { 0.0 };
                    let bt_score = if (bt_token as usize) < scores.len() { scores[bt_token as usize] } else { 0.0 };

                    println!("\nScore comparison at diff position:");
                    println!("  HF token {} score: {:.6}", hf_token, hf_score);
                    println!("  BT token {} score: {:.6}", bt_token, bt_score);

                    if (hf_score - bt_score).abs() < 0.0001 {
                        println!("  >>> LIKELY TIE-BREAKING ISSUE (scores nearly equal)");
                    } else if hf_score > bt_score {
                        println!("  >>> HF chose LOWER score token (unexpected)");
                    } else {
                        println!("  >>> BT chose LOWER score token (Viterbi path difference)");
                    }

                } else if hf_ids.len() != bt_ids.len() {
                    println!("\nLength mismatch: HF={} BT={}", hf_ids.len(), bt_ids.len());
                    let min_len = hf_ids.len().min(bt_ids.len());
                    println!("All {} common tokens match", min_len);

                    if hf_ids.len() > bt_ids.len() {
                        println!("\nHF extra tokens:");
                        for i in min_len..hf_ids.len() {
                            let token_str = hf_tok.decode(&[hf_ids[i]], false).unwrap_or_default();
                            println!("  [{}] {} = {:?}", i, hf_ids[i], token_str);
                        }
                    } else {
                        println!("\nBT extra tokens:");
                        for i in min_len..bt_ids.len() {
                            let token_str = bt_tok.id_to_token(bt_ids[i]).unwrap_or_default();
                            println!("  [{}] {} = {:?}", i, bt_ids[i], token_str);
                        }
                    }
                }

                // Show first 200 chars of text
                println!("\nText (first 200 chars): {:?}", &text[..text.len().min(200)]);
            }
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("SUMMARY: {} mismatches in {} texts ({:.2}% accuracy)",
             mismatch_count, texts.len(),
             (texts.len() - mismatch_count) as f64 / texts.len() as f64 * 100.0);
}
