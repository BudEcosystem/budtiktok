//! Request routing

use crate::load_balancer::{LoadBalancer, LoadBalancingStrategy};

/// Request router
pub struct Router {
    load_balancer: LoadBalancer,
}

impl Router {
    /// Create a new router
    pub fn new(num_workers: usize, strategy: LoadBalancingStrategy) -> Self {
        Self {
            load_balancer: LoadBalancer::new(strategy, num_workers),
        }
    }

    /// Route a request to a worker
    pub fn route(&self) -> usize {
        self.load_balancer.select_worker()
    }
}
