//! Load balancing for worker distribution

use std::sync::atomic::{AtomicUsize, Ordering};

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least-connections
    LeastConnections,
    /// Random
    Random,
    /// NUMA-aware (prefer local NUMA node)
    NumaAware,
}

/// Load balancer for distributing requests across workers
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    next_worker: AtomicUsize,
    num_workers: usize,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(strategy: LoadBalancingStrategy, num_workers: usize) -> Self {
        Self {
            strategy,
            next_worker: AtomicUsize::new(0),
            num_workers,
        }
    }

    /// Select next worker for request
    pub fn select_worker(&self) -> usize {
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let current = self.next_worker.fetch_add(1, Ordering::Relaxed);
                current % self.num_workers
            }
            LoadBalancingStrategy::Random => {
                // Simple pseudo-random using fetch_add
                let seed = self.next_worker.fetch_add(1, Ordering::Relaxed);
                (seed * 1103515245 + 12345) % self.num_workers
            }
            LoadBalancingStrategy::LeastConnections => {
                // TODO: Track connections per worker
                self.next_worker.fetch_add(1, Ordering::Relaxed) % self.num_workers
            }
            LoadBalancingStrategy::NumaAware => {
                // TODO: Implement NUMA-aware selection
                self.next_worker.fetch_add(1, Ordering::Relaxed) % self.num_workers
            }
        }
    }
}
