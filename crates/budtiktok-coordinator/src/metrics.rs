//! Metrics collection and export

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Metrics collector
pub struct MetricsCollector {
    requests_total: AtomicU64,
    tokens_total: AtomicU64,
    errors_total: AtomicU64,
    latency_sum_us: AtomicU64,
    start_time: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            tokens_total: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            latency_sum_us: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Record a request
    pub fn record_request(&self, tokens: u64, latency_us: u64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.tokens_total.fetch_add(tokens, Ordering::Relaxed);
        self.latency_sum_us.fetch_add(latency_us, Ordering::Relaxed);
    }

    /// Record an error
    pub fn record_error(&self) {
        self.errors_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Get metrics snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        let requests = self.requests_total.load(Ordering::Relaxed);
        let tokens = self.tokens_total.load(Ordering::Relaxed);
        let errors = self.errors_total.load(Ordering::Relaxed);
        let latency_sum = self.latency_sum_us.load(Ordering::Relaxed);
        let uptime = self.start_time.elapsed();

        MetricsSnapshot {
            requests_total: requests,
            tokens_total: tokens,
            errors_total: errors,
            avg_latency_us: if requests > 0 { latency_sum / requests } else { 0 },
            requests_per_second: if uptime.as_secs() > 0 {
                requests as f64 / uptime.as_secs_f64()
            } else {
                0.0
            },
            tokens_per_second: if uptime.as_secs() > 0 {
                tokens as f64 / uptime.as_secs_f64()
            } else {
                0.0
            },
            uptime_seconds: uptime.as_secs(),
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics snapshot
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub requests_total: u64,
    pub tokens_total: u64,
    pub errors_total: u64,
    pub avg_latency_us: u64,
    pub requests_per_second: f64,
    pub tokens_per_second: f64,
    pub uptime_seconds: u64,
}
