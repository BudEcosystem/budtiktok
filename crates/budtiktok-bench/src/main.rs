//! BudTikTok Comparison Benchmarking Tool
//!
//! Comprehensive benchmarking suite for comparing tokenizer performance across:
//! - BudTikTok (this project)
//! - HuggingFace Tokenizers (Rust)
//! - HuggingFace TEI (Text Embeddings Inference server)
//! - BlazeText
//!
//! ## Usage
//!
//! ```bash
//! # Run all benchmarks
//! budtiktok-bench run --tokenizer bert-base-uncased --dataset wiki
//!
//! # Compare specific backends
//! budtiktok-bench compare --backends budtiktok,huggingface,tei
//!
//! # Generate report
//! budtiktok-bench report --format markdown --output results.md
//! ```

use ahash;
use anyhow::{Context, Result};
#[cfg(feature = "blazetext")]
use blazetext_wordpiece::BertWordPieceTokenizer;
use budtiktok_core::tokenizer::Tokenizer;
use budtiktok_core::wordpiece::{WordPieceConfig, WordPieceTokenizer};
use budtiktok_core::wordpiece_hyper::{HyperConfig, HyperWordPieceTokenizer};
use clap::{Parser, Subcommand};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tabled::{Table, Tabled};
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{debug, info, warn};

/// BudTikTok Comparison Benchmarking Tool
#[derive(Parser)]
#[command(name = "budtiktok-bench")]
#[command(about = "Benchmark BudTikTok against other tokenizers")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
}

#[derive(Subcommand)]
enum Commands {
    /// Run benchmarks for a single backend
    Run {
        /// Backend to benchmark
        #[arg(short, long, value_enum)]
        backend: Backend,

        /// Tokenizer model name or path
        #[arg(short, long)]
        tokenizer: String,

        /// Dataset to use for benchmarking
        #[arg(short, long, value_enum, default_value = "synthetic")]
        dataset: Dataset,

        /// Number of iterations per benchmark
        #[arg(short, long, default_value = "1000")]
        iterations: usize,

        /// Number of warmup iterations
        #[arg(short, long, default_value = "100")]
        warmup: usize,

        /// Batch sizes to test
        #[arg(long, value_delimiter = ',', default_value = "1,8,32,64,128")]
        batch_sizes: Vec<usize>,

        /// Output format
        #[arg(long, value_enum, default_value = "table")]
        format: OutputFormat,

        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Compare multiple backends
    Compare {
        /// Backends to compare
        #[arg(short, long, value_delimiter = ',')]
        backends: Vec<Backend>,

        /// Tokenizer model name or path
        #[arg(short, long)]
        tokenizer: String,

        /// Dataset to use
        #[arg(short, long, value_enum, default_value = "synthetic")]
        dataset: Dataset,

        /// Number of iterations
        #[arg(short, long, default_value = "1000")]
        iterations: usize,

        /// Number of warmup iterations
        #[arg(short, long, default_value = "100")]
        warmup: usize,

        /// Batch sizes to test
        #[arg(long, value_delimiter = ',', default_value = "1,8,32,64,128")]
        batch_sizes: Vec<usize>,

        /// Output format
        #[arg(long, value_enum, default_value = "table")]
        format: OutputFormat,

        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Generate a comprehensive report
    Report {
        /// Input benchmark results (JSON files)
        #[arg(short, long, num_args = 1..)]
        inputs: Vec<PathBuf>,

        /// Output format
        #[arg(long, value_enum, default_value = "markdown")]
        format: OutputFormat,

        /// Output file
        #[arg(short, long)]
        output: PathBuf,

        /// Include charts (requires 'charts' feature)
        #[arg(long)]
        charts: bool,
    },

    /// List available datasets
    Datasets,

    /// Generate synthetic test data
    GenerateData {
        /// Output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Number of samples to generate
        #[arg(short, long, default_value = "10000")]
        samples: usize,

        /// Distribution of text lengths
        #[arg(long, value_enum, default_value = "mixed")]
        distribution: LengthDistribution,
    },

    /// Run latency benchmark (detailed latency analysis)
    Latency {
        /// Backend to benchmark
        #[arg(short, long, value_enum)]
        backend: Backend,

        /// Tokenizer model
        #[arg(short, long)]
        tokenizer: String,

        /// Number of samples
        #[arg(short, long, default_value = "10000")]
        samples: usize,

        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Run throughput benchmark (tokens/second)
    Throughput {
        /// Backend to benchmark
        #[arg(short, long, value_enum)]
        backend: Backend,

        /// Tokenizer model
        #[arg(short, long)]
        tokenizer: String,

        /// Duration in seconds
        #[arg(short, long, default_value = "30")]
        duration: u64,

        /// Number of parallel workers
        #[arg(short, long, default_value = "1")]
        workers: usize,

        /// Batch size
        #[arg(long, default_value = "32")]
        batch_size: usize,
    },

    /// Run memory benchmark
    Memory {
        /// Backend to benchmark
        #[arg(short, long, value_enum)]
        backend: Backend,

        /// Tokenizer model
        #[arg(short, long)]
        tokenizer: String,

        /// Number of samples
        #[arg(short, long, default_value = "1000")]
        samples: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
enum Backend {
    /// BudTikTok (this project) - FastWordPiece
    Budtiktok,
    /// BudTikTok Hyper - Hash table bypass + SIMD (fastest, production)
    BudtiktokHyper,
    /// HuggingFace Tokenizers (Rust)
    Huggingface,
    /// HuggingFace TEI server
    Tei,
    /// BlazeText
    Blazetext,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Budtiktok => write!(f, "BudTikTok"),
            Backend::BudtiktokHyper => write!(f, "BudTikTok-Hyper"),
            Backend::Huggingface => write!(f, "HuggingFace"),
            Backend::Tei => write!(f, "TEI"),
            Backend::Blazetext => write!(f, "BlazeText"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
enum Dataset {
    /// Synthetic data with varying lengths
    Synthetic,
    /// Wikipedia articles
    Wiki,
    /// Code snippets
    Code,
    /// Social media posts (short)
    Social,
    /// Long documents
    LongDoc,
    /// Mixed real-world data
    Mixed,
    /// Custom dataset from file
    Custom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum OutputFormat {
    /// ASCII table
    Table,
    /// JSON
    Json,
    /// CSV
    Csv,
    /// Markdown
    Markdown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum LengthDistribution {
    /// Short texts (< 128 tokens)
    Short,
    /// Medium texts (128-512 tokens)
    Medium,
    /// Long texts (512-2048 tokens)
    Long,
    /// Very long texts (2048+ tokens)
    VeryLong,
    /// Mixed distribution
    Mixed,
}

/// Benchmark result for a single configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub backend: String,
    pub tokenizer: String,
    pub dataset: String,
    pub batch_size: usize,
    pub iterations: usize,
    pub total_samples: usize,
    pub total_tokens: usize,

    // Timing metrics (in microseconds)
    pub latency_mean_us: f64,
    pub latency_median_us: f64,
    pub latency_p50_us: f64,
    pub latency_p95_us: f64,
    pub latency_p99_us: f64,
    pub latency_min_us: f64,
    pub latency_max_us: f64,
    pub latency_std_us: f64,

    // Throughput metrics
    pub throughput_samples_per_sec: f64,
    pub throughput_tokens_per_sec: f64,
    pub throughput_batches_per_sec: f64,

    // Memory metrics (in bytes)
    pub memory_peak_bytes: Option<u64>,
    pub memory_avg_bytes: Option<u64>,

    // Metadata
    pub timestamp: String,
    pub cpu_info: String,
    pub os_info: String,
}

#[derive(Debug, Clone, Tabled)]
struct ResultRow {
    #[tabled(rename = "Backend")]
    backend: String,
    #[tabled(rename = "Batch")]
    batch_size: usize,
    #[tabled(rename = "Mean (μs)")]
    latency_mean: String,
    #[tabled(rename = "P50 (μs)")]
    latency_p50: String,
    #[tabled(rename = "P99 (μs)")]
    latency_p99: String,
    #[tabled(rename = "Tokens/s")]
    throughput: String,
    #[tabled(rename = "Speedup")]
    speedup: String,
}

/// Tokenizer backend abstraction
trait TokenizerBackend: Send + Sync {
    fn name(&self) -> &str;
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<u32>>>;
    fn decode(&self, ids: &[u32]) -> Result<String>;
}

/// HuggingFace Tokenizers backend
struct HuggingFaceBackend {
    tokenizer: HfTokenizer,
}

impl HuggingFaceBackend {
    fn new(tokenizer_path: &str) -> Result<Self> {
        let tokenizer = HfTokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load HuggingFace tokenizer from {}: {}", tokenizer_path, e))?;
        Ok(Self { tokenizer })
    }
}

impl TokenizerBackend for HuggingFaceBackend {
    fn name(&self) -> &str {
        "HuggingFace"
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<u32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Batch encoding failed: {}", e))?;
        Ok(encodings.iter().map(|e| e.get_ids().to_vec()).collect())
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
    }
}

/// BudTikTok backend using budtiktok-core with WordPieceTokenizer
struct BudTikTokBackend {
    tokenizer: WordPieceTokenizer,
}

impl BudTikTokBackend {
    fn new(model_path: &str) -> Result<Self> {
        // Load tokenizer.json and extract vocab
        let content = std::fs::read_to_string(model_path)
            .with_context(|| format!("Failed to read tokenizer file: {}", model_path))?;

        let json: serde_json::Value = serde_json::from_str(&content)
            .with_context(|| "Failed to parse tokenizer.json")?;

        // Extract vocab from model section
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

        // Extract config from normalizer section
        let mut config = WordPieceConfig::default();
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
        let tokenizer = WordPieceTokenizer::new(vocabulary, config);

        info!("Loaded BudTikTok WordPiece tokenizer from {}", model_path);
        Ok(Self { tokenizer })
    }
}

impl TokenizerBackend for BudTikTokBackend {
    fn name(&self) -> &str {
        "BudTikTok"
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Use the Tokenizer trait's encode method
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("BudTikTok encoding failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<u32>>> {
        // Use the Tokenizer trait's encode_batch method
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let encodings = self.tokenizer.encode_batch(&text_refs, true)
            .map_err(|e| anyhow::anyhow!("BudTikTok batch encoding failed: {}", e))?;
        Ok(encodings.iter().map(|e| e.get_ids().to_vec()).collect())
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, true)
            .map_err(|e| anyhow::anyhow!("BudTikTok decoding failed: {}", e))
    }
}

/// BudTikTok Hyper backend using hash table bypass for common tokens
struct BudTikTokHyperBackend {
    tokenizer: HyperWordPieceTokenizer,
}

impl BudTikTokHyperBackend {
    fn new(model_path: &str) -> Result<Self> {
        // Load tokenizer.json and extract vocab
        let content = std::fs::read_to_string(model_path)
            .with_context(|| format!("Failed to read tokenizer file: {}", model_path))?;

        let json: serde_json::Value = serde_json::from_str(&content)
            .with_context(|| "Failed to parse tokenizer.json")?;

        // Extract vocab from model section
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

        // Extract config from normalizer section
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
        let tokenizer = HyperWordPieceTokenizer::new(vocabulary, config);

        info!("Loaded BudTikTok Hyper tokenizer from {} (hash table bypass)", model_path);
        Ok(Self { tokenizer })
    }
}

impl TokenizerBackend for BudTikTokHyperBackend {
    fn name(&self) -> &str {
        "BudTikTok-Hyper"
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode_with_special(text))
    }

    fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<u32>>> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        Ok(self.tokenizer.encode_batch_with_special(&text_refs))
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(self.tokenizer.decode(ids, true))
    }
}

/// TEI (Text Embeddings Inference) backend via HTTP
struct TeiBackend {
    client: reqwest::blocking::Client,
    base_url: String,
}

impl TeiBackend {
    fn new(base_url: &str) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            base_url: base_url.to_string(),
        })
    }
}

impl TokenizerBackend for TeiBackend {
    fn name(&self) -> &str {
        "TEI"
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        #[derive(Serialize)]
        struct TokenizeRequest {
            inputs: String,
        }

        #[derive(Deserialize)]
        struct TokenizeResponse {
            token_ids: Vec<u32>,
        }

        let response: TokenizeResponse = self
            .client
            .post(format!("{}/tokenize", self.base_url))
            .json(&TokenizeRequest {
                inputs: text.to_string(),
            })
            .send()
            .context("TEI request failed")?
            .json()
            .context("Failed to parse TEI response")?;

        Ok(response.token_ids)
    }

    fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<u32>>> {
        // TEI typically handles batching internally, but for fair comparison
        // we'll call encode for each text
        texts.iter().map(|t| self.encode(t)).collect()
    }

    fn decode(&self, _ids: &[u32]) -> Result<String> {
        // TEI doesn't typically expose decode endpoint
        Err(anyhow::anyhow!("TEI decode not supported"))
    }
}

/// BlazeText backend using blazetext_wordpiece crate
#[cfg(feature = "blazetext")]
struct BlazeTextBackend {
    tokenizer: BertWordPieceTokenizer,
}

#[cfg(feature = "blazetext")]
impl BlazeTextBackend {
    fn new(model_path: &str) -> Result<Self> {
        let tokenizer = BertWordPieceTokenizer::from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load BlazeText tokenizer from {}: {}", model_path, e))?;
        info!("Loaded BlazeText tokenizer from {}", model_path);
        Ok(Self { tokenizer })
    }
}

#[cfg(feature = "blazetext")]
impl TokenizerBackend for BlazeTextBackend {
    fn name(&self) -> &str {
        "BlazeText"
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Use encode_fast for best performance (no offset computation)
        Ok(self.tokenizer.encode_fast(text, true))
    }

    fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<u32>>> {
        // Convert String to &str for blazetext API
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        Ok(self.tokenizer.encode_batch_fast(&text_refs, true))
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(self.tokenizer.decode_ids(ids))
    }
}

/// Create backend from enum
fn create_backend(backend: Backend, tokenizer: &str) -> Result<Box<dyn TokenizerBackend>> {
    match backend {
        Backend::Budtiktok => Ok(Box::new(BudTikTokBackend::new(tokenizer)?)),
        Backend::BudtiktokHyper => Ok(Box::new(BudTikTokHyperBackend::new(tokenizer)?)),
        Backend::Huggingface => Ok(Box::new(HuggingFaceBackend::new(tokenizer)?)),
        Backend::Tei => {
            let tei_url = std::env::var("TEI_URL").unwrap_or_else(|_| "http://localhost:8080".into());
            Ok(Box::new(TeiBackend::new(&tei_url)?))
        }
        #[cfg(feature = "blazetext")]
        Backend::Blazetext => Ok(Box::new(BlazeTextBackend::new(tokenizer)?)),
        #[cfg(not(feature = "blazetext"))]
        Backend::Blazetext => Err(anyhow::anyhow!("BlazeText backend not enabled. Recompile with --features blazetext")),
    }
}

/// Generate test data based on dataset type
fn generate_test_data(dataset: Dataset, count: usize) -> Vec<String> {
    match dataset {
        Dataset::Synthetic => generate_synthetic_data(count, LengthDistribution::Mixed),
        Dataset::Wiki => generate_wiki_like_data(count),
        Dataset::Code => generate_code_data(count),
        Dataset::Social => generate_social_data(count),
        Dataset::LongDoc => generate_long_doc_data(count),
        Dataset::Mixed => generate_mixed_data(count),
        Dataset::Custom => {
            warn!("Custom dataset not loaded, using synthetic");
            generate_synthetic_data(count, LengthDistribution::Mixed)
        }
    }
}

fn generate_synthetic_data(count: usize, distribution: LengthDistribution) -> Vec<String> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "hello", "world", "machine", "learning", "artificial", "intelligence",
        "natural", "language", "processing", "deep", "neural", "network",
        "transformer", "attention", "mechanism", "embedding", "tokenization",
        "vocabulary", "subword", "byte", "pair", "encoding", "wordpiece",
        "unigram", "sentencepiece", "performance", "optimization", "cache",
        "memory", "throughput", "latency", "benchmark", "comparison",
    ];

    (0..count)
        .map(|_| {
            let word_count = match distribution {
                LengthDistribution::Short => rng.gen_range(5..30),
                LengthDistribution::Medium => rng.gen_range(30..100),
                LengthDistribution::Long => rng.gen_range(100..400),
                LengthDistribution::VeryLong => rng.gen_range(400..1000),
                LengthDistribution::Mixed => match rng.gen_range(0..4) {
                    0 => rng.gen_range(5..30),
                    1 => rng.gen_range(30..100),
                    2 => rng.gen_range(100..400),
                    _ => rng.gen_range(400..800),
                },
            };

            (0..word_count)
                .map(|_| words[rng.gen_range(0..words.len())])
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

fn generate_wiki_like_data(count: usize) -> Vec<String> {
    // Generate Wikipedia-like article snippets
    let templates = [
        "In {year}, {person} made significant contributions to the field of {field}. \
         This work laid the foundation for modern {application}.",
        "The {noun} is a {adjective} structure found in {location}. \
         It was built during the {era} period and serves as a testament to {quality}.",
        "{topic} refers to the study of {subject}. \
         Researchers have identified several key aspects including {aspect1}, {aspect2}, and {aspect3}.",
        "According to historical records, the {event} occurred in {year}. \
         This had profound implications for {consequence}.",
    ];

    let fields = ["computer science", "physics", "biology", "mathematics", "linguistics"];
    let persons = ["Dr. Smith", "Professor Johnson", "Marie Curie", "Alan Turing", "Ada Lovelace"];
    let years = ["1950", "1972", "1989", "2001", "2015"];
    let applications = ["machine learning", "quantum computing", "genome sequencing", "cryptography"];

    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..count)
        .map(|_| {
            let template = templates[rng.gen_range(0..templates.len())];
            template
                .replace("{year}", years[rng.gen_range(0..years.len())])
                .replace("{person}", persons[rng.gen_range(0..persons.len())])
                .replace("{field}", fields[rng.gen_range(0..fields.len())])
                .replace("{application}", applications[rng.gen_range(0..applications.len())])
                .replace("{noun}", "building")
                .replace("{adjective}", "remarkable")
                .replace("{location}", "Europe")
                .replace("{era}", "Renaissance")
                .replace("{quality}", "human ingenuity")
                .replace("{topic}", "Computational linguistics")
                .replace("{subject}", "language and computation")
                .replace("{aspect1}", "syntax")
                .replace("{aspect2}", "semantics")
                .replace("{aspect3}", "pragmatics")
                .replace("{event}", "scientific revolution")
                .replace("{consequence}", "modern society")
        })
        .collect()
}

fn generate_code_data(count: usize) -> Vec<String> {
    let code_snippets = [
        "fn main() { println!(\"Hello, world!\"); }",
        "def calculate_sum(numbers): return sum(numbers)",
        "const result = await fetch(url).then(r => r.json());",
        "SELECT * FROM users WHERE active = true ORDER BY created_at DESC;",
        "class TokenizerConfig { public int maxLength = 512; }",
        "impl Iterator for TokenIterator { fn next(&mut self) -> Option<Token> { self.tokens.pop() } }",
        "async function processTokens(text: string): Promise<number[]> { return tokenizer.encode(text); }",
        "for i in range(len(tokens)): output.append(vocab[tokens[i]])",
    ];

    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..count)
        .map(|_| {
            let snippet_count = rng.gen_range(1..5);
            (0..snippet_count)
                .map(|_| code_snippets[rng.gen_range(0..code_snippets.len())])
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect()
}

fn generate_social_data(count: usize) -> Vec<String> {
    let social_templates = [
        "Just tried the new #tokenizer and it's amazing! 10x faster than before",
        "Can't believe how fast this processes text. Mind = blown",
        "Anyone else having issues with the latest update? @support",
        "TIL about byte pair encoding. Pretty cool stuff!",
        "Working on some NLP stuff today. Progress is slow but steady",
        "This benchmark is wild: 1M tokens/sec on a single CPU core",
        "Finally got my model training properly. Time for coffee",
        "Hot take: tokenization is the most underrated part of NLP",
    ];

    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..count)
        .map(|_| social_templates[rng.gen_range(0..social_templates.len())].to_string())
        .collect()
}

fn generate_long_doc_data(count: usize) -> Vec<String> {
    // Generate longer documents by combining multiple paragraphs
    let paragraphs = [
        "Natural language processing has seen tremendous advances in recent years, \
         driven primarily by the transformer architecture and large-scale pre-training. \
         These models have achieved state-of-the-art results across a wide range of tasks.",
        "Tokenization is a critical preprocessing step that converts raw text into a \
         sequence of tokens that can be processed by neural networks. The choice of \
         tokenization algorithm can significantly impact model performance.",
        "Byte Pair Encoding (BPE) is one of the most popular subword tokenization \
         algorithms. It iteratively merges the most frequent pairs of characters or \
         character sequences to build a vocabulary of subword units.",
        "WordPiece is another subword tokenization algorithm commonly used in models \
         like BERT. It uses a greedy longest-match-first strategy to segment words \
         into subword units from a learned vocabulary.",
        "Unigram tokenization takes a probabilistic approach, learning a vocabulary \
         that maximizes the likelihood of the training data. During tokenization, \
         it finds the most probable segmentation using dynamic programming.",
    ];

    use rand::Rng;
    let mut rng = rand::thread_rng();

    (0..count)
        .map(|_| {
            let para_count = rng.gen_range(5..15);
            (0..para_count)
                .map(|_| paragraphs[rng.gen_range(0..paragraphs.len())])
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .collect()
}

fn generate_mixed_data(count: usize) -> Vec<String> {
    let mut data = Vec::with_capacity(count);
    let per_type = count / 5;

    data.extend(generate_synthetic_data(per_type, LengthDistribution::Mixed));
    data.extend(generate_wiki_like_data(per_type));
    data.extend(generate_code_data(per_type));
    data.extend(generate_social_data(per_type));
    data.extend(generate_long_doc_data(count - 4 * per_type));

    // Shuffle the data
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    data.shuffle(&mut rng);

    data
}

/// Run a benchmark for a single backend and configuration
fn run_benchmark(
    backend: &dyn TokenizerBackend,
    data: &[String],
    batch_size: usize,
    iterations: usize,
    warmup: usize,
) -> Result<BenchmarkResult> {
    let total_samples = data.len();

    // Warmup phase
    debug!("Running {} warmup iterations...", warmup);
    for _ in 0..warmup {
        for chunk in data.chunks(batch_size) {
            let _ = backend.encode_batch(&chunk.to_vec())?;
        }
    }

    // Benchmark phase
    let mut latencies: Vec<f64> = Vec::with_capacity(iterations * (total_samples / batch_size + 1));
    let mut total_tokens = 0usize;

    let pb = ProgressBar::new(iterations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap(),
    );
    pb.set_message("Running benchmark...");

    for _ in 0..iterations {
        for chunk in data.chunks(batch_size) {
            let texts: Vec<String> = chunk.to_vec();
            let start = Instant::now();
            let results = backend.encode_batch(&texts)?;
            let elapsed = start.elapsed();

            latencies.push(elapsed.as_secs_f64() * 1_000_000.0); // Convert to microseconds

            for result in &results {
                total_tokens += result.len();
            }
        }
        pb.inc(1);
    }
    pb.finish_with_message("Done!");

    // Calculate statistics
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = latencies.len();

    let mean = latencies.iter().sum::<f64>() / n as f64;
    let median = latencies[n / 2];
    let p50 = latencies[n * 50 / 100];
    let p95 = latencies[n * 95 / 100];
    let p99 = latencies[n * 99 / 100];
    let min = latencies[0];
    let max = latencies[n - 1];

    let variance = latencies.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();

    let total_time_us: f64 = latencies.iter().sum();
    let total_time_sec = total_time_us / 1_000_000.0;
    let samples_per_sec = (total_samples * iterations) as f64 / total_time_sec;
    let tokens_per_sec = total_tokens as f64 / total_time_sec;
    let batches_per_sec = n as f64 / total_time_sec;

    Ok(BenchmarkResult {
        backend: backend.name().to_string(),
        tokenizer: String::new(), // Will be filled by caller
        dataset: String::new(),   // Will be filled by caller
        batch_size,
        iterations,
        total_samples: total_samples * iterations,
        total_tokens,
        latency_mean_us: mean,
        latency_median_us: median,
        latency_p50_us: p50,
        latency_p95_us: p95,
        latency_p99_us: p99,
        latency_min_us: min,
        latency_max_us: max,
        latency_std_us: std,
        throughput_samples_per_sec: samples_per_sec,
        throughput_tokens_per_sec: tokens_per_sec,
        throughput_batches_per_sec: batches_per_sec,
        memory_peak_bytes: None,
        memory_avg_bytes: None,
        timestamp: chrono::Utc::now().to_rfc3339(),
        cpu_info: get_cpu_info(),
        os_info: get_os_info(),
    })
}

fn get_cpu_info() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in content.lines() {
                if line.starts_with("model name") {
                    if let Some(name) = line.split(':').nth(1) {
                        return name.trim().to_string();
                    }
                }
            }
        }
    }
    "Unknown CPU".to_string()
}

fn get_os_info() -> String {
    format!(
        "{} {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::env::consts::FAMILY
    )
}

/// Format results for output
fn format_results(results: &[BenchmarkResult], format: OutputFormat) -> String {
    match format {
        OutputFormat::Table => format_table(results),
        OutputFormat::Json => serde_json::to_string_pretty(results).unwrap_or_default(),
        OutputFormat::Csv => format_csv(results),
        OutputFormat::Markdown => format_markdown(results),
    }
}

fn format_table(results: &[BenchmarkResult]) -> String {
    // Find baseline (HuggingFace) for speedup calculation
    let baseline_throughput: HashMap<usize, f64> = results
        .iter()
        .filter(|r| r.backend == "HuggingFace")
        .map(|r| (r.batch_size, r.throughput_tokens_per_sec))
        .collect();

    let rows: Vec<ResultRow> = results
        .iter()
        .map(|r| {
            let speedup = baseline_throughput
                .get(&r.batch_size)
                .map(|baseline| r.throughput_tokens_per_sec / baseline)
                .unwrap_or(1.0);

            ResultRow {
                backend: r.backend.clone(),
                batch_size: r.batch_size,
                latency_mean: format!("{:.1}", r.latency_mean_us),
                latency_p50: format!("{:.1}", r.latency_p50_us),
                latency_p99: format!("{:.1}", r.latency_p99_us),
                throughput: format!("{:.0}", r.throughput_tokens_per_sec),
                speedup: if r.backend == "HuggingFace" {
                    "1.00x (baseline)".to_string()
                } else {
                    format!("{:.2}x", speedup)
                },
            }
        })
        .collect();

    Table::new(rows).to_string()
}

fn format_csv(results: &[BenchmarkResult]) -> String {
    let mut wtr = csv::Writer::from_writer(vec![]);

    // Write header
    wtr.write_record([
        "backend",
        "batch_size",
        "latency_mean_us",
        "latency_p50_us",
        "latency_p95_us",
        "latency_p99_us",
        "throughput_tokens_per_sec",
        "total_samples",
        "total_tokens",
    ])
    .unwrap();

    for r in results {
        wtr.write_record([
            &r.backend,
            &r.batch_size.to_string(),
            &format!("{:.2}", r.latency_mean_us),
            &format!("{:.2}", r.latency_p50_us),
            &format!("{:.2}", r.latency_p95_us),
            &format!("{:.2}", r.latency_p99_us),
            &format!("{:.2}", r.throughput_tokens_per_sec),
            &r.total_samples.to_string(),
            &r.total_tokens.to_string(),
        ])
        .unwrap();
    }

    String::from_utf8(wtr.into_inner().unwrap()).unwrap_or_default()
}

fn format_markdown(results: &[BenchmarkResult]) -> String {
    let mut output = String::new();

    output.push_str("# BudTikTok Benchmark Results\n\n");
    output.push_str(&format!("**Generated:** {}\n\n", chrono::Utc::now().to_rfc3339()));

    if let Some(first) = results.first() {
        output.push_str("## System Information\n\n");
        output.push_str(&format!("- **CPU:** {}\n", first.cpu_info));
        output.push_str(&format!("- **OS:** {}\n", first.os_info));
        output.push_str(&format!("- **Tokenizer:** {}\n", first.tokenizer));
        output.push_str(&format!("- **Dataset:** {}\n\n", first.dataset));
    }

    output.push_str("## Results\n\n");
    output.push_str("| Backend | Batch | Mean (μs) | P50 (μs) | P99 (μs) | Tokens/s | Speedup |\n");
    output.push_str("|---------|-------|-----------|----------|----------|----------|----------|\n");

    // Find baseline
    let baseline_throughput: HashMap<usize, f64> = results
        .iter()
        .filter(|r| r.backend == "HuggingFace")
        .map(|r| (r.batch_size, r.throughput_tokens_per_sec))
        .collect();

    for r in results {
        let speedup = baseline_throughput
            .get(&r.batch_size)
            .map(|baseline| r.throughput_tokens_per_sec / baseline)
            .unwrap_or(1.0);

        output.push_str(&format!(
            "| {} | {} | {:.1} | {:.1} | {:.1} | {:.0} | {:.2}x |\n",
            r.backend,
            r.batch_size,
            r.latency_mean_us,
            r.latency_p50_us,
            r.latency_p99_us,
            r.throughput_tokens_per_sec,
            speedup
        ));
    }

    output.push_str("\n## Notes\n\n");
    output.push_str("- Latencies are per-batch in microseconds\n");
    output.push_str("- Speedup is relative to HuggingFace baseline\n");
    output.push_str("- Throughput is measured in tokens per second\n");

    output
}

fn output_results(results: &str, output: &Option<PathBuf>) -> Result<()> {
    match output {
        Some(path) => {
            let mut file = fs::File::create(path)?;
            file.write_all(results.as_bytes())?;
            info!("Results written to {}", path.display());
        }
        None => {
            println!("{}", results);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let filter = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();

    match cli.command {
        Commands::Run {
            backend,
            tokenizer,
            dataset,
            iterations,
            warmup,
            batch_sizes,
            format,
            output,
        } => {
            println!(
                "{} Running benchmark for {} with {}",
                ">>>".green().bold(),
                backend.to_string().cyan(),
                tokenizer.yellow()
            );

            let backend_impl = create_backend(backend, &tokenizer)?;
            let data = generate_test_data(dataset, 1000);

            let mut results = Vec::new();

            for &batch_size in &batch_sizes {
                println!(
                    "  {} Batch size: {}",
                    "->".blue(),
                    batch_size.to_string().yellow()
                );

                let mut result = run_benchmark(&*backend_impl, &data, batch_size, iterations, warmup)?;
                result.tokenizer = tokenizer.clone();
                result.dataset = format!("{:?}", dataset);
                results.push(result);
            }

            let formatted = format_results(&results, format);
            output_results(&formatted, &output)?;
        }

        Commands::Compare {
            backends,
            tokenizer,
            dataset,
            iterations,
            warmup,
            batch_sizes,
            format,
            output,
        } => {
            println!(
                "{} Comparing {} backends with {}",
                ">>>".green().bold(),
                backends.len().to_string().cyan(),
                tokenizer.yellow()
            );

            let data = generate_test_data(dataset, 1000);
            let mut all_results = Vec::new();

            for backend in &backends {
                println!(
                    "\n{} Benchmarking {}...",
                    ">>>".green(),
                    backend.to_string().cyan()
                );

                let backend_impl = create_backend(*backend, &tokenizer)?;

                for &batch_size in &batch_sizes {
                    println!(
                        "  {} Batch size: {}",
                        "->".blue(),
                        batch_size.to_string().yellow()
                    );

                    let mut result =
                        run_benchmark(&*backend_impl, &data, batch_size, iterations, warmup)?;
                    result.tokenizer = tokenizer.clone();
                    result.dataset = format!("{:?}", dataset);
                    all_results.push(result);
                }
            }

            let formatted = format_results(&all_results, format);
            output_results(&formatted, &output)?;
        }

        Commands::Report {
            inputs,
            format,
            output,
            charts: _,
        } => {
            println!("{} Generating report...", ">>>".green().bold());

            let mut all_results: Vec<BenchmarkResult> = Vec::new();

            for input in &inputs {
                let content = fs::read_to_string(input)
                    .with_context(|| format!("Failed to read {}", input.display()))?;
                let results: Vec<BenchmarkResult> = serde_json::from_str(&content)
                    .with_context(|| format!("Failed to parse {}", input.display()))?;
                all_results.extend(results);
            }

            let formatted = format_results(&all_results, format);
            output_results(&formatted, &Some(output))?;
        }

        Commands::Datasets => {
            println!("{} Available datasets:\n", ">>>".green().bold());
            println!("  {} - Synthetic data with varying lengths", "synthetic".cyan());
            println!("  {} - Wikipedia-like articles", "wiki".cyan());
            println!("  {} - Code snippets", "code".cyan());
            println!("  {} - Short social media posts", "social".cyan());
            println!("  {} - Long documents (5-15 paragraphs)", "longdoc".cyan());
            println!("  {} - Mix of all types", "mixed".cyan());
        }

        Commands::GenerateData {
            output,
            samples,
            distribution,
        } => {
            println!(
                "{} Generating {} samples with {:?} distribution...",
                ">>>".green().bold(),
                samples.to_string().cyan(),
                distribution
            );

            fs::create_dir_all(&output)?;

            let data = generate_synthetic_data(samples, distribution);

            for (i, text) in data.iter().enumerate() {
                let path = output.join(format!("sample_{:06}.txt", i));
                fs::write(&path, text)?;
            }

            // Also save as JSON for easy loading
            let json_path = output.join("data.json");
            let json = serde_json::to_string_pretty(&data)?;
            fs::write(&json_path, json)?;

            println!("{} Generated {} samples in {}", "OK".green(), samples, output.display());
        }

        Commands::Latency {
            backend,
            tokenizer,
            samples,
            output,
        } => {
            println!(
                "{} Running latency analysis for {}...",
                ">>>".green().bold(),
                backend.to_string().cyan()
            );

            let backend_impl = create_backend(backend, &tokenizer)?;
            let data = generate_test_data(Dataset::Mixed, samples);

            let mut latencies: Vec<f64> = Vec::with_capacity(samples);

            let pb = ProgressBar::new(samples as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len}")
                    .unwrap(),
            );

            for text in &data {
                let start = Instant::now();
                let _ = backend_impl.encode(text)?;
                latencies.push(start.elapsed().as_secs_f64() * 1_000_000.0);
                pb.inc(1);
            }
            pb.finish();

            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = latencies.len();

            let report = format!(
                r#"Latency Analysis Report
========================

Backend: {}
Tokenizer: {}
Samples: {}

Statistics:
  Min:    {:.2} μs
  Max:    {:.2} μs
  Mean:   {:.2} μs
  Median: {:.2} μs
  P50:    {:.2} μs
  P90:    {:.2} μs
  P95:    {:.2} μs
  P99:    {:.2} μs
  P99.9:  {:.2} μs

Histogram:
  <10μs:    {} ({:.1}%)
  10-50μs:  {} ({:.1}%)
  50-100μs: {} ({:.1}%)
  100-500μs: {} ({:.1}%)
  >500μs:   {} ({:.1}%)
"#,
                backend,
                tokenizer,
                samples,
                latencies[0],
                latencies[n - 1],
                latencies.iter().sum::<f64>() / n as f64,
                latencies[n / 2],
                latencies[n * 50 / 100],
                latencies[n * 90 / 100],
                latencies[n * 95 / 100],
                latencies[n * 99 / 100],
                latencies[n * 999 / 1000],
                latencies.iter().filter(|&&x| x < 10.0).count(),
                latencies.iter().filter(|&&x| x < 10.0).count() as f64 / n as f64 * 100.0,
                latencies.iter().filter(|&&x| x >= 10.0 && x < 50.0).count(),
                latencies.iter().filter(|&&x| x >= 10.0 && x < 50.0).count() as f64 / n as f64 * 100.0,
                latencies.iter().filter(|&&x| x >= 50.0 && x < 100.0).count(),
                latencies.iter().filter(|&&x| x >= 50.0 && x < 100.0).count() as f64 / n as f64 * 100.0,
                latencies.iter().filter(|&&x| x >= 100.0 && x < 500.0).count(),
                latencies.iter().filter(|&&x| x >= 100.0 && x < 500.0).count() as f64 / n as f64 * 100.0,
                latencies.iter().filter(|&&x| x >= 500.0).count(),
                latencies.iter().filter(|&&x| x >= 500.0).count() as f64 / n as f64 * 100.0,
            );

            output_results(&report, &output)?;
        }

        Commands::Throughput {
            backend,
            tokenizer,
            duration,
            workers,
            batch_size,
        } => {
            println!(
                "{} Running throughput benchmark for {} seconds with {} workers...",
                ">>>".green().bold(),
                duration.to_string().cyan(),
                workers.to_string().yellow()
            );

            let backend_impl = create_backend(backend, &tokenizer)?;
            let data = generate_test_data(Dataset::Mixed, 10000);

            let start = Instant::now();
            let target_duration = Duration::from_secs(duration);

            let total_tokens = std::sync::atomic::AtomicUsize::new(0);
            let total_batches = std::sync::atomic::AtomicUsize::new(0);

            // Use rayon for parallel processing
            rayon::scope(|s| {
                for _ in 0..workers {
                    s.spawn(|_| {
                        let mut local_tokens = 0usize;
                        let mut local_batches = 0usize;

                        while start.elapsed() < target_duration {
                            for chunk in data.chunks(batch_size) {
                                if start.elapsed() >= target_duration {
                                    break;
                                }

                                if let Ok(results) = backend_impl.encode_batch(&chunk.to_vec()) {
                                    for result in &results {
                                        local_tokens += result.len();
                                    }
                                    local_batches += 1;
                                }
                            }
                        }

                        total_tokens.fetch_add(local_tokens, std::sync::atomic::Ordering::Relaxed);
                        total_batches.fetch_add(local_batches, std::sync::atomic::Ordering::Relaxed);
                    });
                }
            });

            let elapsed = start.elapsed();
            let tokens = total_tokens.load(std::sync::atomic::Ordering::Relaxed);
            let batches = total_batches.load(std::sync::atomic::Ordering::Relaxed);

            println!(
                "\n{} Throughput Results:",
                "===".green().bold()
            );
            println!("  Duration:      {:.2}s", elapsed.as_secs_f64());
            println!("  Workers:       {}", workers);
            println!("  Batch size:    {}", batch_size);
            println!("  Total tokens:  {}", tokens);
            println!("  Total batches: {}", batches);
            println!(
                "  {} {:.0} tokens/sec",
                "Throughput:".green().bold(),
                tokens as f64 / elapsed.as_secs_f64()
            );
            println!(
                "  {} {:.0} batches/sec",
                "Batch rate:".green(),
                batches as f64 / elapsed.as_secs_f64()
            );
        }

        Commands::Memory {
            backend,
            tokenizer,
            samples,
        } => {
            println!(
                "{} Running memory benchmark for {}...",
                ">>>".green().bold(),
                backend.to_string().cyan()
            );

            // Memory profiling is best done with external tools like heaptrack
            // This provides a simple baseline measurement
            let backend_impl = create_backend(backend, &tokenizer)?;
            let data = generate_test_data(Dataset::Mixed, samples);

            println!("  {} Loading tokenizer and warming up...", "->".blue());

            // Warmup
            for text in data.iter().take(100) {
                let _ = backend_impl.encode(text)?;
            }

            println!("  {} Processing {} samples...", "->".blue(), samples);

            let pb = ProgressBar::new(samples as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len}")
                    .unwrap(),
            );

            for text in &data {
                let _ = backend_impl.encode(text)?;
                pb.inc(1);
            }
            pb.finish();

            println!(
                "\n{} Memory analysis complete.",
                "OK".green()
            );
            println!("  For detailed memory profiling, use:");
            println!("    heaptrack ./target/release/budtiktok-bench run ...");
            println!("    valgrind --tool=dhat ./target/release/budtiktok-bench run ...");
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_synthetic_data() {
        let data = generate_synthetic_data(10, LengthDistribution::Short);
        assert_eq!(data.len(), 10);
        for text in &data {
            assert!(!text.is_empty());
        }
    }

    #[test]
    fn test_generate_mixed_data() {
        let data = generate_mixed_data(100);
        assert_eq!(data.len(), 100);
    }

    #[test]
    fn test_format_csv() {
        let results = vec![BenchmarkResult {
            backend: "Test".to_string(),
            tokenizer: "test-tokenizer".to_string(),
            dataset: "synthetic".to_string(),
            batch_size: 32,
            iterations: 100,
            total_samples: 1000,
            total_tokens: 50000,
            latency_mean_us: 100.0,
            latency_median_us: 95.0,
            latency_p50_us: 95.0,
            latency_p95_us: 150.0,
            latency_p99_us: 200.0,
            latency_min_us: 50.0,
            latency_max_us: 300.0,
            latency_std_us: 25.0,
            throughput_samples_per_sec: 10000.0,
            throughput_tokens_per_sec: 500000.0,
            throughput_batches_per_sec: 312.5,
            memory_peak_bytes: None,
            memory_avg_bytes: None,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            cpu_info: "Test CPU".to_string(),
            os_info: "Linux".to_string(),
        }];

        let csv = format_csv(&results);
        assert!(csv.contains("backend"));
        assert!(csv.contains("Test"));
        assert!(csv.contains("500000"));
    }

    #[test]
    fn test_backend_display() {
        assert_eq!(Backend::Budtiktok.to_string(), "BudTikTok");
        assert_eq!(Backend::BudtiktokHyper.to_string(), "BudTikTok-Hyper");
        assert_eq!(Backend::Huggingface.to_string(), "HuggingFace");
        assert_eq!(Backend::Tei.to_string(), "TEI");
        assert_eq!(Backend::Blazetext.to_string(), "BlazeText");
    }
}
