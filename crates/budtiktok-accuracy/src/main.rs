//! BudTikTok Accuracy Testing Suite
//!
//! Validates BudTikTok tokenization output against HuggingFace tokenizers
//! as the gold standard. Ensures 100% compatibility for production use.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use tabled::{Table, Tabled};
use tokenizers::Tokenizer as HfTokenizer;
use budtiktok_core::wordpiece_hyper::{HyperConfig, HyperWordPieceTokenizer};

/// BudTikTok Accuracy Testing Suite
#[derive(Parser)]
#[command(name = "budtiktok-accuracy")]
#[command(about = "Validate BudTikTok against HuggingFace tokenizers")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run accuracy tests against a single tokenizer
    Test {
        /// Path to tokenizer.json or HuggingFace model name
        #[arg(short, long)]
        tokenizer: String,

        /// Path to test data file (one text per line)
        #[arg(short, long)]
        data: PathBuf,

        /// Output report file (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Maximum number of samples to test
        #[arg(short, long, default_value = "10000")]
        max_samples: usize,

        /// Fail on first mismatch
        #[arg(long)]
        fail_fast: bool,
    },

    /// Run accuracy tests against multiple tokenizers
    TestAll {
        /// Directory containing tokenizer configs
        #[arg(short, long)]
        tokenizers_dir: PathBuf,

        /// Path to test data file
        #[arg(short, long)]
        data: PathBuf,

        /// Output directory for reports
        #[arg(short, long)]
        output_dir: PathBuf,
    },

    /// Generate test dataset from various sources
    GenerateData {
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Data source type
        #[arg(short, long, value_enum)]
        source: DataSource,

        /// Number of samples to generate
        #[arg(short, long, default_value = "10000")]
        count: usize,
    },

    /// Show detailed comparison for a single input
    Debug {
        /// Path to tokenizer.json
        #[arg(short, long)]
        tokenizer: String,

        /// Text to tokenize
        #[arg(short, long)]
        text: String,
    },
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum DataSource {
    Wikipedia,
    Code,
    Multilingual,
    EdgeCases,
    Mixed,
}

/// Result of comparing a single tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub input: String,
    pub hf_ids: Vec<u32>,
    pub bud_ids: Vec<u32>,
    pub hf_tokens: Vec<String>,
    pub bud_tokens: Vec<String>,
    pub matches: bool,
    pub mismatch_details: Option<MismatchDetails>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MismatchDetails {
    pub first_diff_position: usize,
    pub hf_token_at_diff: String,
    pub bud_token_at_diff: String,
    pub length_diff: i32,
}

/// Summary of accuracy test run
#[derive(Debug, Clone, Serialize, Deserialize, Tabled)]
pub struct AccuracySummary {
    pub tokenizer: String,
    pub total_samples: usize,
    pub matching_samples: usize,
    pub mismatched_samples: usize,
    #[tabled(display_with = "display_percentage")]
    pub accuracy_percentage: f64,
    pub total_tokens_hf: usize,
    pub total_tokens_bud: usize,
}

fn display_percentage(p: &f64) -> String {
    format!("{:.4}%", p)
}

/// Full accuracy report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyReport {
    pub tokenizer_path: String,
    pub dataset_path: String,
    pub summary: AccuracySummary,
    pub mismatches: Vec<ComparisonResult>,
    pub timestamp: String,
}

/// Accuracy tester that compares BudTikTok against HuggingFace
pub struct AccuracyTester {
    hf_tokenizer: HfTokenizer,
    bud_tokenizer: HyperWordPieceTokenizer,
    tokenizer_name: String,
}

impl AccuracyTester {
    /// Create a new accuracy tester from a tokenizer path or model name
    pub fn new(tokenizer_path: &str) -> Result<Self> {
        let hf_tokenizer = HfTokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load HF tokenizer from {}: {}", tokenizer_path, e))?;

        // Load BudTikTok Hyper tokenizer
        let bud_tokenizer = Self::load_hyper_tokenizer(tokenizer_path)?;

        Ok(Self {
            hf_tokenizer,
            bud_tokenizer,
            tokenizer_name: tokenizer_path.to_string(),
        })
    }

    fn load_hyper_tokenizer(model_path: &str) -> Result<HyperWordPieceTokenizer> {
        let content = std::fs::read_to_string(model_path)
            .with_context(|| format!("Failed to read tokenizer file: {}", model_path))?;

        let json: serde_json::Value = serde_json::from_str(&content)
            .with_context(|| "Failed to parse tokenizer.json")?;

        let vocab_obj = json
            .get("model")
            .and_then(|m| m.get("vocab"))
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow::anyhow!("Missing model.vocab in tokenizer.json"))?;

        let mut token_to_id = ahash::AHashMap::new();
        for (token, id) in vocab_obj {
            let id = id.as_u64()
                .ok_or_else(|| anyhow::anyhow!("Invalid token ID for '{}'", token))? as u32;
            token_to_id.insert(token.clone(), id);
        }

        let mut config = HyperConfig::default();
        if let Some(normalizer) = json.get("normalizer") {
            config.do_lower_case = normalizer.get("lowercase").and_then(|v| v.as_bool()).unwrap_or(true);
            config.strip_accents = normalizer.get("strip_accents").and_then(|v| v.as_bool()).unwrap_or(true);
            config.tokenize_chinese_chars = normalizer.get("handle_chinese_chars").and_then(|v| v.as_bool()).unwrap_or(true);
        }

        if let Some(model) = json.get("model") {
            if let Some(unk) = model.get("unk_token").and_then(|v| v.as_str()) {
                config.unk_token = unk.to_string();
            }
            if let Some(prefix) = model.get("continuing_subword_prefix").and_then(|v| v.as_str()) {
                config.continuing_subword_prefix = prefix.to_string();
            }
        }

        let special_tokens = budtiktok_core::vocab::SpecialTokens {
            unk_token: Some(config.unk_token.clone()),
            cls_token: Some("[CLS]".to_string()),
            sep_token: Some("[SEP]".to_string()),
            ..Default::default()
        };

        let vocabulary = budtiktok_core::vocab::Vocabulary::new(token_to_id, special_tokens);
        Ok(HyperWordPieceTokenizer::new(vocabulary, config))
    }

    /// Compare tokenization of a single text
    pub fn compare(&self, text: &str, add_special_tokens: bool) -> Result<ComparisonResult> {
        // Get HuggingFace result
        let hf_encoding = self
            .hf_tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("HF tokenization failed: {}", e))?;

        let hf_ids: Vec<u32> = hf_encoding.get_ids().to_vec();
        let hf_tokens: Vec<String> = hf_encoding.get_tokens().to_vec();

        // Get BudTikTok Hyper result
        let bud_ids = if add_special_tokens {
            self.bud_tokenizer.encode_with_special(text)
        } else {
            self.bud_tokenizer.encode_fast(text)
        };

        // Convert IDs to tokens for display (using HF tokenizer's vocab for consistency)
        let bud_tokens: Vec<String> = bud_ids
            .iter()
            .map(|&id| {
                self.hf_tokenizer
                    .id_to_token(id)
                    .unwrap_or_else(|| format!("[UNK:{}]", id))
            })
            .collect();

        let matches = hf_ids == bud_ids;

        let mismatch_details = if !matches {
            let first_diff = hf_ids
                .iter()
                .zip(bud_ids.iter())
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| hf_ids.len().min(bud_ids.len()));

            Some(MismatchDetails {
                first_diff_position: first_diff,
                hf_token_at_diff: hf_tokens.get(first_diff).cloned().unwrap_or_default(),
                bud_token_at_diff: bud_tokens.get(first_diff).cloned().unwrap_or_default(),
                length_diff: bud_ids.len() as i32 - hf_ids.len() as i32,
            })
        } else {
            None
        };

        Ok(ComparisonResult {
            input: text.to_string(),
            hf_ids,
            bud_ids,
            hf_tokens,
            bud_tokens,
            matches,
            mismatch_details,
        })
    }

    /// Run accuracy test on a dataset
    pub fn run_test(
        &self,
        data_path: &Path,
        max_samples: usize,
        fail_fast: bool,
    ) -> Result<AccuracyReport> {
        let file = File::open(data_path)
            .with_context(|| format!("Failed to open data file: {:?}", data_path))?;
        let reader = BufReader::new(file);

        let samples: Vec<String> = reader
            .lines()
            .take(max_samples)
            .filter_map(|l| l.ok())
            .filter(|l| !l.trim().is_empty())
            .collect();

        let pb = ProgressBar::new(samples.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut results: Vec<ComparisonResult> = Vec::with_capacity(samples.len());
        let mut total_tokens_hf = 0usize;
        let mut total_tokens_bud = 0usize;

        for sample in &samples {
            let result = self.compare(sample, true)?;

            total_tokens_hf += result.hf_ids.len();
            total_tokens_bud += result.bud_ids.len();

            if !result.matches && fail_fast {
                pb.finish_with_message("Stopped on first mismatch");
                return Err(anyhow::anyhow!(
                    "Mismatch found for input: {}",
                    sample.chars().take(50).collect::<String>()
                ));
            }

            results.push(result);
            pb.inc(1);
        }

        pb.finish_with_message("Done");

        let matching = results.iter().filter(|r| r.matches).count();
        let mismatched = results.iter().filter(|r| !r.matches).count();

        let summary = AccuracySummary {
            tokenizer: self.tokenizer_name.clone(),
            total_samples: results.len(),
            matching_samples: matching,
            mismatched_samples: mismatched,
            accuracy_percentage: (matching as f64 / results.len() as f64) * 100.0,
            total_tokens_hf,
            total_tokens_bud,
        };

        let mismatches: Vec<ComparisonResult> =
            results.into_iter().filter(|r| !r.matches).collect();

        Ok(AccuracyReport {
            tokenizer_path: self.tokenizer_name.clone(),
            dataset_path: data_path.to_string_lossy().to_string(),
            summary,
            mismatches,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }
}

/// Generate test data
pub fn generate_test_data(source: DataSource, count: usize, output: &Path) -> Result<()> {
    let samples: Vec<String> = match source {
        DataSource::EdgeCases => generate_edge_cases(count),
        DataSource::Wikipedia => {
            println!("Note: Wikipedia data generation requires downloading. Using synthetic data.");
            generate_synthetic_text(count)
        }
        DataSource::Code => generate_code_samples(count),
        DataSource::Multilingual => generate_multilingual_samples(count),
        DataSource::Mixed => {
            let mut samples = Vec::new();
            samples.extend(generate_edge_cases(count / 4));
            samples.extend(generate_synthetic_text(count / 4));
            samples.extend(generate_code_samples(count / 4));
            samples.extend(generate_multilingual_samples(count / 4));
            samples
        }
    };

    let mut file = File::create(output)?;
    for sample in samples {
        writeln!(file, "{}", sample)?;
    }

    println!("Generated {} samples to {:?}", count, output);
    Ok(())
}

fn generate_edge_cases(count: usize) -> Vec<String> {
    let long_a = "a".repeat(1000);
    let long_ab = "ab ".repeat(100);

    let cases: Vec<&str> = vec![
        // Empty and whitespace
        "",
        " ",
        "   ",
        "\t",
        "\n",
        "\r\n",
        "  \t  \n  ",
        // Single characters
        "a",
        "Z",
        "0",
        "!",
        ".",
        ",",
        // Unicode edge cases
        "é",          // Precomposed
        "e\u{0301}",  // Decomposed (e + combining acute)
        "ñ",
        "中",
        "日本語",
        "한국어",
        "العربية",
        "עברית",
        "🎉",
        "👨‍👩‍👧‍👦",  // ZWJ sequence
        "\u{FEFF}",   // BOM
        "\u{200B}",   // Zero-width space
        "\u{00A0}",   // Non-breaking space
        // Punctuation edge cases
        "Hello, world!",
        "It's a test.",
        "foo-bar",
        "foo--bar",
        "foo...bar",
        "(test)",
        "[test]",
        "{test}",
        "\"test\"",
        "'test'",
        "test's",
        "don't",
        "can't",
        "won't",
        // Numbers
        "123",
        "12.34",
        "1,234",
        "$100",
        "100%",
        "3.14159",
        // Long words
        "supercalifragilisticexpialidocious",
        "pneumonoultramicroscopicsilicovolcanoconiosis",
        // URLs and emails
        "https://example.com/path?query=value",
        "user@example.com",
        // Code-like
        "function() {}",
        "if (x > 0) { return true; }",
        "def main():",
        "console.log('test')",
        // Mixed
        "Hello 你好 مرحبا",
        "Test123!@#",
        "CamelCaseWord",
        "snake_case_word",
        "kebab-case-word",
        "ALLCAPS",
        "  leading spaces",
        "trailing spaces  ",
        "  both sides  ",
        // Very long
        &long_a,
        &long_ab,
    ];

    cases
        .into_iter()
        .map(|s| s.to_string())
        .cycle()
        .take(count)
        .collect()
}

fn generate_synthetic_text(count: usize) -> Vec<String> {
    let sentences = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
        "The five boxing wizards jump quickly.",
        "Sphinx of black quartz, judge my vow.",
        "Two driven jocks help fax my big quiz.",
        "The jay, pig, fox, zebra and my wolves quack!",
        "Sympathizing would fix Quaker objectives.",
        "A wizard's job is to vex chumps quickly in fog.",
        "Watch Jeopardy!, Alex Trebek's fun TV quiz game.",
    ];

    (0..count)
        .map(|i| {
            let num_sentences = (i % 5) + 1;
            (0..num_sentences)
                .map(|j| sentences[(i + j) % sentences.len()])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

fn generate_code_samples(count: usize) -> Vec<String> {
    let samples = vec![
        "def hello_world():\n    print(\"Hello, World!\")",
        "function add(a, b) { return a + b; }",
        "pub fn main() { println!(\"Hello\"); }",
        "#include <stdio.h>\nint main() { return 0; }",
        "class MyClass { constructor() { this.value = 0; } }",
        "SELECT * FROM users WHERE id = 1;",
        "import numpy as np\narr = np.array([1, 2, 3])",
        "const express = require('express');",
        "type Result<T> = Ok<T> | Err<string>;",
        "async function fetchData() { await fetch(url); }",
    ];

    samples
        .into_iter()
        .map(|s| s.to_string())
        .cycle()
        .take(count)
        .collect()
}

fn generate_multilingual_samples(count: usize) -> Vec<String> {
    let samples = vec![
        "Hello, World!",
        "Bonjour le monde!",
        "Hallo Welt!",
        "¡Hola Mundo!",
        "Ciao mondo!",
        "Olá Mundo!",
        "Привет мир!",
        "你好世界！",
        "こんにちは世界！",
        "안녕하세요 세계!",
        "مرحبا بالعالم!",
        "שלום עולם!",
        "Γειά σου Κόσμε!",
        "Xin chào thế giới!",
        "สวัสดีโลก!",
    ];

    samples
        .into_iter()
        .map(|s| s.to_string())
        .cycle()
        .take(count)
        .collect()
}

fn print_debug_comparison(result: &ComparisonResult) {
    println!("\n{}", "=".repeat(80));
    println!("{}", "Debug Comparison".bold());
    println!("{}", "=".repeat(80));

    println!("\n{}:", "Input".bold());
    println!("  \"{}\"", result.input);

    println!("\n{}:", "HuggingFace Tokens".bold().green());
    for (i, (id, token)) in result.hf_ids.iter().zip(&result.hf_tokens).enumerate() {
        println!("  [{:3}] {:6} \"{}\"", i, id, token);
    }

    println!("\n{}:", "BudTikTok Tokens".bold().blue());
    for (i, (id, token)) in result.bud_ids.iter().zip(&result.bud_tokens).enumerate() {
        let status = if i < result.hf_ids.len() && result.hf_ids[i] == *id {
            "✓".green()
        } else {
            "✗".red()
        };
        println!("  [{:3}] {:6} \"{}\" {}", i, id, token, status);
    }

    if result.matches {
        println!("\n{}", "✓ MATCH".bold().green());
    } else {
        println!("\n{}", "✗ MISMATCH".bold().red());
        if let Some(ref details) = result.mismatch_details {
            println!("  First diff at position: {}", details.first_diff_position);
            println!("  HF token: \"{}\"", details.hf_token_at_diff);
            println!("  Bud token: \"{}\"", details.bud_token_at_diff);
            println!("  Length diff: {:+}", details.length_diff);
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Test {
            tokenizer,
            data,
            output,
            max_samples,
            fail_fast,
        } => {
            println!("{}", "BudTikTok Accuracy Test".bold());
            println!("Tokenizer: {}", tokenizer);
            println!("Data: {:?}", data);
            println!("Max samples: {}", max_samples);
            println!();

            let tester = AccuracyTester::new(&tokenizer)?;
            let report = tester.run_test(&data, max_samples, fail_fast)?;

            // Print summary table
            let table = Table::new(vec![report.summary.clone()]).to_string();
            println!("\n{}", "Summary".bold());
            println!("{}", table);

            // Print status
            if report.summary.accuracy_percentage == 100.0 {
                println!("\n{}", "✓ All samples match!".bold().green());
            } else {
                println!(
                    "\n{} {} mismatches found",
                    "✗".red(),
                    report.mismatches.len()
                );

                // Show first few mismatches
                println!("\n{}", "First mismatches:".bold());
                for (i, mismatch) in report.mismatches.iter().take(5).enumerate() {
                    println!(
                        "  {}. \"{}...\"",
                        i + 1,
                        mismatch.input.chars().take(50).collect::<String>()
                    );
                }
            }

            // Save report
            if let Some(output_path) = output {
                let json = serde_json::to_string_pretty(&report)?;
                std::fs::write(&output_path, json)?;
                println!("\nReport saved to {:?}", output_path);
            }

            // Exit with error code if mismatches
            if report.summary.mismatched_samples > 0 {
                std::process::exit(1);
            }
        }

        Commands::TestAll {
            tokenizers_dir,
            data,
            output_dir,
        } => {
            std::fs::create_dir_all(&output_dir)?;

            let tokenizer_files: Vec<_> = std::fs::read_dir(&tokenizers_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map_or(false, |ext| ext == "json"))
                .collect();

            println!("Found {} tokenizer configs", tokenizer_files.len());

            for entry in tokenizer_files {
                let path = entry.path();
                println!("\nTesting {:?}...", path);

                match AccuracyTester::new(&path.to_string_lossy()) {
                    Ok(tester) => {
                        let report = tester.run_test(&data, 10000, false)?;
                        let output_file =
                            output_dir.join(format!("{}.json", path.file_stem().unwrap().to_string_lossy()));
                        let json = serde_json::to_string_pretty(&report)?;
                        std::fs::write(&output_file, json)?;

                        if report.summary.accuracy_percentage == 100.0 {
                            println!("  {} 100% accuracy", "✓".green());
                        } else {
                            println!(
                                "  {} {:.2}% accuracy ({} mismatches)",
                                "✗".red(),
                                report.summary.accuracy_percentage,
                                report.summary.mismatched_samples
                            );
                        }
                    }
                    Err(e) => {
                        println!("  {} Failed to load: {}", "✗".red(), e);
                    }
                }
            }
        }

        Commands::GenerateData {
            output,
            source,
            count,
        } => {
            generate_test_data(source, count, &output)?;
        }

        Commands::Debug { tokenizer, text } => {
            let tester = AccuracyTester::new(&tokenizer)?;
            let result = tester.compare(&text, true)?;
            print_debug_comparison(&result);
        }
    }

    Ok(())
}

// Stub for chrono until we add the dependency
mod chrono {
    pub struct Utc;
    impl Utc {
        pub fn now() -> DateTime { DateTime }
    }
    pub struct DateTime;
    impl DateTime {
        pub fn to_rfc3339(&self) -> String {
            "2025-12-17T00:00:00Z".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_case_generation() {
        let cases = generate_edge_cases(100);
        assert_eq!(cases.len(), 100);
    }

    #[test]
    fn test_accuracy_tester_creation() {
        // This will fail without a real tokenizer file, but tests the structure
        let result = AccuracyTester::new("nonexistent.json");
        assert!(result.is_err());
    }
}
