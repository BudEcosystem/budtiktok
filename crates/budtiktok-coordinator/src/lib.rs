//! BudTikTok Coordinator - Distributed worker fleet management
//!
//! This crate provides coordination for distributed tokenization:
//! - Worker registration and health monitoring
//! - Load balancing across workers
//! - NUMA-aware request routing
//! - Metrics and monitoring
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │                  Coordinator                     │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │
//! │  │   Router    │  │   Health    │  │ Metrics │  │
//! │  │             │  │   Monitor   │  │         │  │
//! │  └──────┬──────┘  └──────┬──────┘  └────┬────┘  │
//! └─────────┼────────────────┼──────────────┼───────┘
//!           │                │              │
//!    ┌──────┴──────┐  ┌──────┴──────┐      │
//!    │   Worker 0  │  │   Worker 1  │      │
//!    │  (NUMA 0)   │  │  (NUMA 1)   │      │
//!    └─────────────┘  └─────────────┘      │
//!                                          │
//!                              ┌───────────┴───────────┐
//!                              │   Prometheus/Grafana  │
//!                              └───────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use budtiktok_coordinator::{Coordinator, CoordinatorConfig};
//!
//! let config = CoordinatorConfig::builder()
//!     .num_workers(4)
//!     .numa_aware(true)
//!     .build();
//!
//! let coordinator = Coordinator::new(config)?;
//! coordinator.start().await?;
//! ```

#![deny(clippy::all)]
#![warn(clippy::pedantic)]

pub mod config;
pub mod coordinator;
pub mod health;
pub mod load_balancer;
pub mod metrics;
pub mod router;
pub mod token_budget;
pub mod worker;

pub use config::CoordinatorConfig;
pub use coordinator::Coordinator;
pub use token_budget::{
    TokenBudgetConfig, TokenBudgetRouter, TokenBudgetBatch, TokenBudgetStats,
    PendingRequest, BatchResult, AsyncTokenBudgetRouter,
};
pub use worker::WorkerHandle;
