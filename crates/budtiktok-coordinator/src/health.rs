//! Health monitoring for workers

use std::time::{Duration, Instant};

/// Health status of a worker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    pub status: HealthStatus,
    pub latency: Duration,
    pub timestamp: Instant,
    pub message: Option<String>,
}

impl HealthCheckResult {
    /// Create a healthy result
    pub fn healthy(latency: Duration) -> Self {
        Self {
            status: HealthStatus::Healthy,
            latency,
            timestamp: Instant::now(),
            message: None,
        }
    }

    /// Create an unhealthy result
    pub fn unhealthy(message: String) -> Self {
        Self {
            status: HealthStatus::Unhealthy,
            latency: Duration::ZERO,
            timestamp: Instant::now(),
            message: Some(message),
        }
    }
}

/// Health monitor for tracking worker health
pub struct HealthMonitor {
    check_interval: Duration,
    timeout: Duration,
    last_check: Option<Instant>,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(check_interval: Duration, timeout: Duration) -> Self {
        Self {
            check_interval,
            timeout,
            last_check: None,
        }
    }

    /// Check if a health check is due
    pub fn should_check(&self) -> bool {
        match self.last_check {
            Some(last) => last.elapsed() >= self.check_interval,
            None => true,
        }
    }

    /// Record a health check
    pub fn record_check(&mut self) {
        self.last_check = Some(Instant::now());
    }
}
