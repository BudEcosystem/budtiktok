//! BudTikTok SIMD - Hardware-accelerated text processing
//!
//! This crate provides SIMD-accelerated implementations of:
//! - UTF-8 validation and processing
//! - Whitespace detection and splitting
//! - ASCII case conversion
//! - Pattern matching primitives
//!
//! # Architecture Support
//!
//! - **x86_64**: AVX-512, AVX2, SSE4.2
//! - **ARM**: NEON, SVE
//! - **RISC-V**: RVV (Vector Extension)
//!
//! # Example
//!
//! ```rust,ignore
//! use budtiktok_simd::{detect_whitespace, SimdBackend};
//!
//! let backend = SimdBackend::auto_detect();
//! let text = b"Hello world this is a test";
//! let splits = backend.find_whitespace(text);
//! ```

#![deny(clippy::all)]
#![warn(clippy::pedantic)]

pub mod detect;
pub mod utf8;
pub mod whitespace;
pub mod patterns;
pub mod backend;

pub use backend::{SimdBackend, SimdCapabilities};

/// Runtime CPU feature detection
pub fn detect_capabilities() -> SimdCapabilities {
    SimdCapabilities::detect()
}
