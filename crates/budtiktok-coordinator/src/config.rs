//! Coordinator configuration

/// Coordinator configuration
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Number of worker processes
    pub num_workers: usize,
    /// Enable NUMA-aware scheduling
    pub numa_aware: bool,
    /// Maximum batch size per worker
    pub max_batch_size: usize,
    /// Worker timeout in milliseconds
    pub worker_timeout_ms: u64,
    /// Health check interval in milliseconds
    pub health_check_interval_ms: u64,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics port
    pub metrics_port: u16,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
            numa_aware: true,
            max_batch_size: 128,
            worker_timeout_ms: 30000,
            health_check_interval_ms: 1000,
            enable_metrics: true,
            metrics_port: 9090,
        }
    }
}

impl CoordinatorConfig {
    /// Create a builder for coordinator configuration
    pub fn builder() -> CoordinatorConfigBuilder {
        CoordinatorConfigBuilder::new()
    }
}

/// Builder for coordinator configuration
pub struct CoordinatorConfigBuilder {
    config: CoordinatorConfig,
}

impl CoordinatorConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: CoordinatorConfig::default(),
        }
    }

    /// Set number of workers
    pub fn num_workers(mut self, n: usize) -> Self {
        self.config.num_workers = n;
        self
    }

    /// Enable/disable NUMA-aware scheduling
    pub fn numa_aware(mut self, enabled: bool) -> Self {
        self.config.numa_aware = enabled;
        self
    }

    /// Set maximum batch size
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.config.max_batch_size = size;
        self
    }

    /// Build the configuration
    pub fn build(self) -> CoordinatorConfig {
        self.config
    }
}

impl Default for CoordinatorConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Stub for num_cpus - in real implementation would use the crate
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    }
}
