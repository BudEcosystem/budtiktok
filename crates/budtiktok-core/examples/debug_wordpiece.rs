//! Debug WordPiece tokenization to find exact divergence from HuggingFace

use std::fs;
use ahash::AHashMap;
use budtiktok_core::wordpiece::{WordPieceConfig, WordPieceTokenizer};
use budtiktok_core::vocab::{Vocabulary, SpecialTokens};
use budtiktok_core::tokenizer::Tokenizer;

const WORKSPACE: &str = "/home/bud/Desktop/latentbud/budtiktok";

fn main() {
    // Load BERT tokenizer
    let tokenizer_path = format!("{}/test_data/bert-base-uncased/tokenizer.json", WORKSPACE);
    let content = fs::read_to_string(&tokenizer_path).expect("Failed to read tokenizer.json");
    let json: serde_json::Value = serde_json::from_str(&content).expect("Failed to parse tokenizer.json");

    // Extract vocab
    let vocab_obj = json["model"]["vocab"].as_object().expect("Missing vocab");
    let mut token_to_id: AHashMap<String, u32> = AHashMap::new();
    for (token, id) in vocab_obj {
        token_to_id.insert(token.clone(), id.as_u64().unwrap() as u32);
    }

    println!("Vocabulary size: {}", token_to_id.len());

    // Create WordPiece config
    let mut config = WordPieceConfig::default();
    config.unk_token = "[UNK]".to_string();
    config.continuing_subword_prefix = "##".to_string();
    config.do_lower_case = true;
    config.strip_accents = true;

    // Create special tokens
    let special_tokens = SpecialTokens {
        unk_token: Some("[UNK]".to_string()),
        cls_token: Some("[CLS]".to_string()),
        sep_token: Some("[SEP]".to_string()),
        pad_token: Some("[PAD]".to_string()),
        mask_token: Some("[MASK]".to_string()),
        ..Default::default()
    };

    let vocabulary = Vocabulary::new(token_to_id.clone(), special_tokens);
    let tokenizer = WordPieceTokenizer::new(vocabulary, config);

    // Test specific cases
    println!("\n=== Testing Specific Cases ===\n");

    let test_cases = vec![
        ("en-dash", "word–word"),
        ("em-dash", "word—word"),
        ("ellipsis", "word…word"),
        ("curly apos", "it's"),
        ("straight apos", "it's"),
        ("mixed", "test–one…two's"),
        ("soft hyphen", "soft\u{00AD}hyphen"),
        ("nbspace", "word\u{00A0}word"),
    ];

    for (name, text) in &test_cases {
        let encoding = tokenizer.encode(text, false).unwrap();
        let tokens: Vec<&str> = encoding.get_tokens().iter().map(|s| s.as_str()).collect();
        println!("{:15} | Input: {:25} | Tokens: {:?}", name, format!("{:?}", text), tokens);
    }

    // Check vocabulary for specific tokens
    println!("\n=== Vocabulary Check ===\n");

    let chars_to_check = vec![
        "'",      // Straight apostrophe
        "'",      // Curly apostrophe U+2019
        "–",      // En-dash U+2013
        "—",      // Em-dash U+2014
        "…",      // Ellipsis U+2026
        "-",      // Hyphen-minus
    ];

    for c in chars_to_check {
        let id = token_to_id.get(c);
        println!("Token {:?} (U+{:04X}): {:?}", c, c.chars().next().unwrap() as u32, id);
    }

    // Test a document from the dataset
    println!("\n=== Testing Real Document ===\n");

    let data_path = format!("{}/benchmark_data/openwebtext_1gb.jsonl", WORKSPACE);
    let file = std::fs::File::open(&data_path).expect("Failed to open dataset");
    let reader = std::io::BufReader::new(file);

    use std::io::BufRead;
    let documents: Vec<String> = reader.lines()
        .take(50)
        .filter_map(|line| {
            let line = line.ok()?;
            let json: serde_json::Value = serde_json::from_str(&line).ok()?;
            json["text"].as_str().map(|s| s.to_string())
        })
        .collect();

    // Test document 3 (known to have discrepancy)
    let doc = &documents[3];
    let encoding = tokenizer.encode(doc, false).unwrap();

    println!("Document 3:");
    println!("  Length: {} chars", doc.len());
    println!("  Token count: {}", encoding.len());
    println!("  First 50 tokens: {:?}", &encoding.get_tokens()[..50.min(encoding.len())]);

    // Find unusual characters
    println!("\n  Unusual characters:");
    for (i, c) in doc.char_indices() {
        if c as u32 > 127 || c == '\u{2013}' || c == '\u{2014}' || c == '\u{2026}' || c == '\u{2018}' || c == '\u{2019}' {
            if i < 500 {
                let start = i.saturating_sub(5);
                let end = (i + 10).min(doc.len());
                println!("    pos {}: U+{:04X} {:?}", i, c as u32, &doc[start..end]);
            }
        }
    }
}
