//! BudTikTok CLI - Command-line tokenizer
//!
//! # Usage
//!
//! ```bash
//! # Tokenize text
//! budtiktok encode "Hello, world!" --model bert-base-uncased
//!
//! # Decode tokens
//! budtiktok decode 101 7592 1010 2088 999 102 --model bert-base-uncased
//!
//! # Interactive mode
//! budtiktok interactive --model bert-base-uncased
//!
//! # Benchmark
//! budtiktok benchmark --model bert-base-uncased --input corpus.txt
//! ```

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use tracing::info;

/// BudTikTok - High-performance tokenizer
#[derive(Parser)]
#[command(name = "budtiktok")]
#[command(about = "High-performance tokenization for ML workloads")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode text to tokens
    Encode {
        /// Text to encode
        text: String,

        /// Model name or path
        #[arg(short, long)]
        model: String,

        /// Add special tokens
        #[arg(long, default_value = "true")]
        add_special_tokens: bool,

        /// Output format (ids, tokens, json)
        #[arg(short, long, default_value = "ids")]
        format: String,
    },

    /// Decode tokens to text
    Decode {
        /// Token IDs to decode
        ids: Vec<u32>,

        /// Model name or path
        #[arg(short, long)]
        model: String,

        /// Skip special tokens
        #[arg(long, default_value = "true")]
        skip_special_tokens: bool,
    },

    /// Interactive tokenization mode
    Interactive {
        /// Model name or path
        #[arg(short, long)]
        model: String,
    },

    /// Tokenize file(s)
    File {
        /// Input file(s)
        #[arg(short, long, num_args = 1..)]
        input: Vec<PathBuf>,

        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Model name or path
        #[arg(short, long)]
        model: String,

        /// Number of parallel workers
        #[arg(short, long, default_value = "1")]
        workers: usize,
    },

    /// Show model information
    Info {
        /// Model name or path
        #[arg(short, long)]
        model: String,
    },

    /// List available models
    List,

    /// Benchmark tokenization performance
    Benchmark {
        /// Model name or path
        #[arg(short, long)]
        model: String,

        /// Input file for benchmark
        #[arg(short, long)]
        input: PathBuf,

        /// Number of iterations
        #[arg(short = 'n', long, default_value = "100")]
        iterations: usize,
    },

    /// Start tokenization server
    #[cfg(feature = "server")]
    Serve {
        /// Model name or path
        #[arg(short, long)]
        model: String,

        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Number of workers
        #[arg(short, long, default_value = "4")]
        workers: usize,
    },
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
        Commands::Encode {
            text,
            model,
            add_special_tokens,
            format,
        } => {
            info!("Loading model: {}", model);
            // TODO: Load BudTikTok tokenizer
            // let tokenizer = budtiktok_core::Tokenizer::from_pretrained(&model)?;
            // let encoding = tokenizer.encode(&text, add_special_tokens)?;

            // Placeholder output
            println!(
                "{} Encode with model '{}' (special_tokens={})",
                ">>>".green(),
                model.cyan(),
                add_special_tokens
            );
            println!("  Input: {}", text.yellow());
            println!("  Format: {}", format);
            println!("  {} Implementation pending - core tokenizer not yet complete", "Note:".red());
        }

        Commands::Decode {
            ids,
            model,
            skip_special_tokens,
        } => {
            info!("Loading model: {}", model);
            // TODO: Load BudTikTok tokenizer and decode

            println!(
                "{} Decode with model '{}' (skip_special={})",
                ">>>".green(),
                model.cyan(),
                skip_special_tokens
            );
            println!("  IDs: {:?}", ids);
            println!("  {} Implementation pending", "Note:".red());
        }

        Commands::Interactive { model } => {
            println!(
                "{} Interactive mode with model '{}'",
                ">>>".green(),
                model.cyan()
            );
            println!("Type text to tokenize, or 'quit' to exit.\n");

            // TODO: Load tokenizer once
            // let tokenizer = budtiktok_core::Tokenizer::from_pretrained(&model)?;

            let stdin = io::stdin();
            let mut stdout = io::stdout();

            loop {
                print!("{} ", ">".green());
                stdout.flush()?;

                let mut line = String::new();
                stdin.lock().read_line(&mut line)?;
                let line = line.trim();

                if line == "quit" || line == "exit" {
                    break;
                }

                if line.is_empty() {
                    continue;
                }

                // TODO: Tokenize and print results
                // let encoding = tokenizer.encode(line, true)?;
                // println!("IDs: {:?}", encoding.get_ids());
                // println!("Tokens: {:?}", encoding.get_tokens());

                println!("  {} Tokenization pending", "Note:".yellow());
            }
        }

        Commands::File {
            input,
            output,
            model,
            workers,
        } => {
            println!(
                "{} Processing {} file(s) with {} workers",
                ">>>".green(),
                input.len(),
                workers
            );
            println!("  Model: {}", model.cyan());
            if let Some(ref out) = output {
                println!("  Output: {}", out.display());
            }
            println!("  {} Implementation pending", "Note:".red());
        }

        Commands::Info { model } => {
            println!("{} Model information for '{}'", ">>>".green(), model.cyan());
            println!("  {} Implementation pending", "Note:".red());
            // TODO: Load and display model info
            // - Vocabulary size
            // - Algorithm type (WordPiece, BPE, Unigram)
            // - Special tokens
            // - Max length
        }

        Commands::List => {
            println!("{} Available models:", ">>>".green());
            println!("  {} Model listing pending", "Note:".red());
            // TODO: List available models
        }

        Commands::Benchmark {
            model,
            input,
            iterations,
        } => {
            println!(
                "{} Benchmarking '{}' with {} iterations",
                ">>>".green(),
                model.cyan(),
                iterations
            );
            println!("  Input: {}", input.display());
            println!("  {} Use budtiktok-bench for comprehensive benchmarks", "Tip:".blue());
            println!("  {} Implementation pending", "Note:".red());
        }

        #[cfg(feature = "server")]
        Commands::Serve {
            model,
            host,
            port,
            workers,
        } => {
            println!(
                "{} Starting server at {}:{} with {} workers",
                ">>>".green(),
                host,
                port,
                workers
            );
            println!("  Model: {}", model.cyan());
            println!("  {} Server implementation pending", "Note:".red());
        }
    }

    Ok(())
}
