//! Main coordinator implementation

use crate::config::CoordinatorConfig;
use crate::worker::WorkerHandle;
use std::sync::Arc;

/// Coordinator for managing worker fleet
pub struct Coordinator {
    config: CoordinatorConfig,
    workers: Vec<WorkerHandle>,
}

impl Coordinator {
    /// Create a new coordinator
    pub fn new(config: CoordinatorConfig) -> Result<Self, CoordinatorError> {
        Ok(Self {
            config,
            workers: Vec::new(),
        })
    }

    /// Start the coordinator and spawn workers
    pub async fn start(&mut self) -> Result<(), CoordinatorError> {
        // TODO: Spawn worker processes
        for i in 0..self.config.num_workers {
            let worker = WorkerHandle::spawn(i, self.config.clone())?;
            self.workers.push(worker);
        }
        Ok(())
    }

    /// Stop all workers
    pub async fn stop(&mut self) -> Result<(), CoordinatorError> {
        for worker in &mut self.workers {
            worker.stop()?;
        }
        self.workers.clear();
        Ok(())
    }

    /// Get number of active workers
    pub fn active_workers(&self) -> usize {
        self.workers.iter().filter(|w| w.is_healthy()).count()
    }

    /// Get coordinator configuration
    pub fn config(&self) -> &CoordinatorConfig {
        &self.config
    }
}

/// Coordinator error type
#[derive(Debug, thiserror::Error)]
pub enum CoordinatorError {
    #[error("Failed to spawn worker: {0}")]
    SpawnFailed(String),
    #[error("Worker communication error: {0}")]
    CommunicationError(String),
    #[error("Worker timeout")]
    Timeout,
    #[error("All workers failed")]
    AllWorkersFailed,
}
