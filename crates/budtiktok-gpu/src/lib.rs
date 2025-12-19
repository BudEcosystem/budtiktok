//! BudTikTok GPU - GPU-accelerated tokenization
//!
//! This crate provides GPU acceleration for batch tokenization:
//! - CUDA backend for NVIDIA GPUs
//! - Metal backend for Apple Silicon (not yet implemented)
//! - Vulkan backend for cross-platform support (not yet implemented)
//!
//! # GPU-Native Pipeline (Recommended)
//!
//! The `GpuNativeTokenizer` runs the ENTIRE tokenization pipeline on GPU with zero
//! CPU round-trips during tokenization. Based on state-of-the-art research:
//! - BlockBPE (arxiv:2507.11941): GPU-resident hashmaps, parallel prefix scans
//! - RAPIDS cuDF nvtext: 483x faster WordPiece via perfect hashing
//!
//! ```rust,ignore
//! use budtiktok_gpu::{GpuNativeTokenizer, GpuNativeConfig};
//! use budtiktok_gpu::cuda::CudaContext;
//! use std::sync::Arc;
//! use std::collections::HashMap;
//!
//! let ctx = Arc::new(CudaContext::new(0)?);
//! let vocab: HashMap<String, u32> = load_vocabulary();
//! let mut tokenizer = GpuNativeTokenizer::new(ctx, &vocab, GpuNativeConfig::default())?;
//!
//! let texts = vec!["Hello world", "GPU tokenization is fast"];
//! let results = tokenizer.encode_batch(&texts)?;
//! ```
//!
//! # Hybrid Pipeline (Legacy)
//!
//! The `GpuWordPieceTokenizer` uses CPU for pre-tokenization and GPU for vocabulary
//! lookup. Useful when GPU-native pipeline isn't needed.
//!
//! ```rust,ignore
//! use budtiktok_gpu::{GpuWordPieceTokenizer, GpuWordPieceConfig};
//! use budtiktok_gpu::cuda::CudaContext;
//! use std::sync::Arc;
//!
//! let ctx = Arc::new(CudaContext::new(0)?);
//! let vocab = vec![("[PAD]", 0), ("[UNK]", 1), ("hello", 2), ("world", 3)];
//! let tokenizer = GpuWordPieceTokenizer::new(ctx, &vocab, GpuWordPieceConfig::default())?;
//!
//! let batch = vec!["Hello world", "Another text"];
//! let results = tokenizer.encode_batch(&batch)?;
//! ```

#![deny(clippy::all)]
#![warn(clippy::pedantic)]

pub mod backend;
pub mod cuda;
pub mod memory;
pub mod kernels;
pub mod wordpiece;
pub mod gpu_native;
pub mod spwp;
pub mod bpe;

pub use backend::{GpuBackend, GpuTokenizer, GpuError};
pub use wordpiece::{GpuWordPieceTokenizer, GpuWordPieceConfig};
pub use gpu_native::{GpuNativeTokenizer, GpuNativeConfig};
pub use spwp::{SpwpTokenizer, SpwpConfig};
pub use bpe::{GpuBpeTokenizer, GpuBpeConfig, GpuBpeEncoding};
