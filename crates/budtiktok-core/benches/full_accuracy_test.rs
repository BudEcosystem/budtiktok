//! Full 1GB corpus accuracy test - document by document comparison with HuggingFace

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

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
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     FULL 1GB CORPUS ACCURACY TEST - Document by Document         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Load tokenizers
    println!("Loading tokenizers...");
    let hf_tok = HfTokenizer::from_file(XLNET_PATH).expect("Failed to load HF tokenizer");

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
    println!("Tokenizers loaded successfully.\n");

    // Load and process all documents
    println!("Loading documents from 1GB corpus...");
    let file = File::open(DATASET_PATH).expect("Failed to open dataset");
    let reader = BufReader::with_capacity(1024 * 1024, file);

    let mut total_docs = 0usize;
    let mut matching_docs = 0usize;
    let mut mismatching_docs = 0usize;
    let mut total_tokens_hf = 0usize;
    let mut total_tokens_bt = 0usize;
    let mut total_bytes = 0usize;

    // Track mismatch categories
    let mut viterbi_tie_mismatches = 0usize;
    let mut unk_mismatches = 0usize;
    let mut length_mismatches = 0usize;
    let mut other_mismatches = 0usize;

    // Store first few mismatches for reporting
    let mut mismatch_examples: Vec<(usize, String, usize, usize)> = Vec::new();

    let start_time = Instant::now();
    let mut last_report = Instant::now();

    for (idx, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };

        let record: TextRecord = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let text = &record.text;
        if text.is_empty() {
            continue;
        }

        total_docs += 1;
        total_bytes += text.len();

        // Tokenize with both
        let hf_enc = hf_tok.encode(text.as_str(), false).unwrap();
        let hf_ids: Vec<u32> = hf_enc.get_ids().to_vec();
        let bt_ids = bt_tok.encode_hf_compat_simd_v2(text);

        total_tokens_hf += hf_ids.len();
        total_tokens_bt += bt_ids.len();

        if hf_ids == bt_ids {
            matching_docs += 1;
        } else {
            mismatching_docs += 1;

            // Categorize the mismatch
            let category = categorize_mismatch(&hf_ids, &bt_ids, &hf_tok, &bt_tok);
            match category {
                MismatchCategory::ViterbiTie => viterbi_tie_mismatches += 1,
                MismatchCategory::UnkDifference => unk_mismatches += 1,
                MismatchCategory::LengthDifference => length_mismatches += 1,
                MismatchCategory::Other => other_mismatches += 1,
            }

            // Store first 20 mismatch examples
            if mismatch_examples.len() < 20 {
                let first_diff = find_first_diff(&hf_ids, &bt_ids);
                let preview: String = text.chars().take(80).collect();
                mismatch_examples.push((idx, preview, first_diff, hf_ids.len()));
            }
        }

        // Progress report every 5 seconds
        if last_report.elapsed().as_secs() >= 5 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let docs_per_sec = total_docs as f64 / elapsed;
            let mb_processed = total_bytes as f64 / (1024.0 * 1024.0);
            let accuracy = if total_docs > 0 {
                matching_docs as f64 / total_docs as f64 * 100.0
            } else {
                0.0
            };

            println!(
                "Progress: {:>8} docs | {:>7.2} MB | {:>6.0} docs/s | Accuracy: {:.4}% ({} mismatches)",
                total_docs, mb_processed, docs_per_sec, accuracy, mismatching_docs
            );
            last_report = Instant::now();
        }
    }

    let total_time = start_time.elapsed();

    // Final report
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         FINAL RESULTS                            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    println!("CORPUS STATISTICS:");
    println!("  Total documents:        {:>12}", total_docs);
    println!("  Total bytes:            {:>12} ({:.2} GB)", total_bytes, total_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("  Total tokens (HF):      {:>12}", total_tokens_hf);
    println!("  Total tokens (BT):      {:>12}", total_tokens_bt);
    println!("  Processing time:        {:>12.2}s", total_time.as_secs_f64());
    println!("  Throughput:             {:>12.2} MB/s", total_bytes as f64 / total_time.as_secs_f64() / (1024.0 * 1024.0));
    println!();

    let accuracy = matching_docs as f64 / total_docs as f64 * 100.0;
    println!("ACCURACY:");
    println!("  Matching documents:     {:>12}", matching_docs);
    println!("  Mismatching documents:  {:>12}", mismatching_docs);
    println!("  Document accuracy:      {:>12.6}%", accuracy);
    println!();

    println!("MISMATCH BREAKDOWN:");
    println!("  Viterbi tie-breaking:   {:>12} ({:.2}%)",
             viterbi_tie_mismatches,
             viterbi_tie_mismatches as f64 / total_docs as f64 * 100.0);
    println!("  UNK token differences:  {:>12} ({:.2}%)",
             unk_mismatches,
             unk_mismatches as f64 / total_docs as f64 * 100.0);
    println!("  Length differences:     {:>12} ({:.2}%)",
             length_mismatches,
             length_mismatches as f64 / total_docs as f64 * 100.0);
    println!("  Other differences:      {:>12} ({:.2}%)",
             other_mismatches,
             other_mismatches as f64 / total_docs as f64 * 100.0);
    println!();

    if !mismatch_examples.is_empty() {
        println!("SAMPLE MISMATCHES (first {}):", mismatch_examples.len());
        for (i, (doc_idx, preview, first_diff, total_tokens)) in mismatch_examples.iter().enumerate() {
            println!("  {}. Doc #{}: first diff at token {}/{}",
                     i + 1, doc_idx, first_diff, total_tokens);
            println!("     Text: \"{}...\"", preview);
        }
        println!();
    }

    // Token-level accuracy
    let token_matches = calculate_token_accuracy(total_tokens_hf, total_tokens_bt, mismatching_docs);
    println!("TOKEN-LEVEL ESTIMATE:");
    println!("  Estimated token accuracy: ~{:.4}%", token_matches);
    println!();

    println!("══════════════════════════════════════════════════════════════════");
    if accuracy >= 99.9 {
        println!("  ✓ EXCELLENT: {:.4}% document-level accuracy achieved!", accuracy);
    } else if accuracy >= 99.0 {
        println!("  ✓ GOOD: {:.4}% document-level accuracy", accuracy);
    } else {
        println!("  ⚠ NEEDS IMPROVEMENT: {:.4}% document-level accuracy", accuracy);
    }
    println!("══════════════════════════════════════════════════════════════════");
}

#[derive(Debug, Clone, Copy)]
enum MismatchCategory {
    ViterbiTie,
    UnkDifference,
    LengthDifference,
    Other,
}

fn categorize_mismatch(
    hf_ids: &[u32],
    bt_ids: &[u32],
    _hf_tok: &HfTokenizer,
    _bt_tok: &UnigramFast,
) -> MismatchCategory {
    // Length difference
    if hf_ids.len() != bt_ids.len() {
        // Check if it's just off by a small amount (likely Viterbi tie)
        let diff = (hf_ids.len() as i64 - bt_ids.len() as i64).abs();
        if diff <= 2 {
            return MismatchCategory::ViterbiTie;
        }
        return MismatchCategory::LengthDifference;
    }

    // Find first difference
    let mut unk_involved = false;
    let mut consecutive_swap = false;

    for i in 0..hf_ids.len().min(bt_ids.len()) {
        if hf_ids[i] != bt_ids[i] {
            // Check if UNK (id=0) is involved
            if hf_ids[i] == 0 || bt_ids[i] == 0 {
                unk_involved = true;
            }

            // Check for consecutive swap pattern (Viterbi tie-breaking)
            // e.g., HF=[A,B] vs BT=[B,A] where order is swapped
            if i + 1 < hf_ids.len() && i + 1 < bt_ids.len() {
                if hf_ids[i] == bt_ids[i + 1] && hf_ids[i + 1] == bt_ids[i] {
                    consecutive_swap = true;
                }
            }
            break;
        }
    }

    if consecutive_swap {
        return MismatchCategory::ViterbiTie;
    }
    if unk_involved {
        return MismatchCategory::UnkDifference;
    }
    MismatchCategory::Other
}

fn find_first_diff(hf_ids: &[u32], bt_ids: &[u32]) -> usize {
    for i in 0..hf_ids.len().min(bt_ids.len()) {
        if hf_ids[i] != bt_ids[i] {
            return i;
        }
    }
    hf_ids.len().min(bt_ids.len())
}

fn calculate_token_accuracy(total_hf: usize, total_bt: usize, mismatches: usize) -> f64 {
    // Rough estimate: assume average mismatch affects ~2 tokens
    let estimated_wrong_tokens = mismatches * 2;
    let total_tokens = total_hf.max(total_bt);
    if total_tokens == 0 {
        return 100.0;
    }
    (1.0 - estimated_wrong_tokens as f64 / total_tokens as f64) * 100.0
}
