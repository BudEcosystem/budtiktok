//! Integration Tests for BudTikTok Core (Section 9.2)
//!
//! This module contains integration tests covering:
//! - End-to-end tokenization workflows
//! - Distributed pipeline testing
//! - Failure recovery scenarios
//! - GPU integration (when available)
//! - LatentBud integration

pub mod tokenization;

// TDD specification tests - contain #[ignore] tests for features not yet implemented:
pub mod distributed;
pub mod resilience;
pub mod gpu;
pub mod latentbud;
