//! GPU backend abstraction
//!
//! Provides a unified interface for GPU-accelerated tokenization with:
//! - CPU fallback for small batches
//! - Automatic batch size optimization
//! - Multi-GPU support

use thiserror::Error;

#[cfg(feature = "cuda")]
use crate::cuda::{get_cuda_devices, is_cuda_available, CudaContext, CudaDevice};
#[cfg(feature = "cuda")]
use crate::kernels::{PreTokenizeKernel, VocabLookupKernel, WordPieceKernel};
#[cfg(feature = "cuda")]
use crate::memory::TokenizationBuffers;
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::time::Instant;

/// GPU backend error
#[derive(Error, Debug)]
pub enum GpuError {
    #[error("GPU not available: {0}")]
    NotAvailable(String),
    #[error("CUDA error: {0}")]
    Cuda(String),
    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),
    #[error("Kernel execution failed: {0}")]
    KernelExecution(String),
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// GPU backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// NVIDIA CUDA
    Cuda,
    /// Apple Metal (not yet implemented)
    Metal,
    /// Vulkan compute (not yet implemented)
    Vulkan,
}

/// GPU tokenizer configuration
#[derive(Debug, Clone)]
pub struct GpuTokenizerConfig {
    /// Minimum batch size to use GPU (below this, use CPU)
    pub min_batch_size: usize,
    /// Maximum batch size per GPU
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Device IDs to use (empty = auto-detect)
    pub device_ids: Vec<i32>,
    /// Enable automatic batch size tuning
    pub auto_tune: bool,
}

impl Default for GpuTokenizerConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 8,
            max_batch_size: 256,
            max_seq_len: 512,
            device_ids: Vec::new(),
            auto_tune: true,
        }
    }
}

/// GPU tokenizer for batch processing
#[cfg(feature = "cuda")]
pub struct GpuTokenizer {
    backend: GpuBackend,
    config: GpuTokenizerConfig,
    contexts: Vec<Arc<CudaContext>>,
    optimal_batch_size: usize,
}

#[cfg(not(feature = "cuda"))]
pub struct GpuTokenizer {
    backend: GpuBackend,
    config: GpuTokenizerConfig,
}

impl GpuTokenizer {
    /// Create a new GPU tokenizer with default configuration
    #[cfg(feature = "cuda")]
    pub fn new(backend: GpuBackend) -> Result<Self, GpuError> {
        Self::with_config(backend, GpuTokenizerConfig::default())
    }

    /// Create a new GPU tokenizer with custom configuration
    #[cfg(feature = "cuda")]
    pub fn with_config(backend: GpuBackend, config: GpuTokenizerConfig) -> Result<Self, GpuError> {
        match backend {
            GpuBackend::Cuda => {
                if !is_cuda_available() {
                    return Err(GpuError::NotAvailable("CUDA not available".into()));
                }

                // Determine which devices to use
                let device_ids = if config.device_ids.is_empty() {
                    // Auto-detect: use all available devices
                    let devices = get_cuda_devices();
                    if devices.is_empty() {
                        return Err(GpuError::NotAvailable("No CUDA devices found".into()));
                    }
                    devices.iter().map(|d| d.id).collect()
                } else {
                    config.device_ids.clone()
                };

                // Create contexts for each device
                let mut contexts = Vec::with_capacity(device_ids.len());
                for &device_id in &device_ids {
                    let ctx = CudaContext::new(device_id)?;
                    contexts.push(Arc::new(ctx));
                }

                let mut tokenizer = Self {
                    backend,
                    config,
                    contexts,
                    optimal_batch_size: 32, // Default, will be tuned
                };

                // Auto-tune batch size if enabled
                if tokenizer.config.auto_tune {
                    tokenizer.optimal_batch_size = tokenizer.find_optimal_batch_size();
                }

                Ok(tokenizer)
            }
            GpuBackend::Metal => {
                Err(GpuError::NotAvailable("Metal backend not yet implemented".into()))
            }
            GpuBackend::Vulkan => {
                Err(GpuError::NotAvailable("Vulkan backend not yet implemented".into()))
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(backend: GpuBackend) -> Result<Self, GpuError> {
        Self::with_config(backend, GpuTokenizerConfig::default())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn with_config(backend: GpuBackend, config: GpuTokenizerConfig) -> Result<Self, GpuError> {
        Err(GpuError::NotAvailable(format!(
            "{:?} backend not compiled in (enable 'cuda' feature)",
            backend
        )))
    }

    /// Get the backend type
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// Get the configuration
    #[cfg(feature = "cuda")]
    pub fn config(&self) -> &GpuTokenizerConfig {
        &self.config
    }

    /// Encode a single text
    #[cfg(feature = "cuda")]
    pub fn encode(&self, _text: &str) -> Result<Vec<u32>, GpuError> {
        // For single text, CPU is usually faster
        // Return placeholder - actual implementation would use CPU tokenizer
        Err(GpuError::NotAvailable(
            "Single encoding should use CPU fallback".into(),
        ))
    }

    #[cfg(not(feature = "cuda"))]
    pub fn encode(&self, _text: &str) -> Result<Vec<u32>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    /// Encode a batch of texts on GPU
    #[cfg(feature = "cuda")]
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, GpuError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Use CPU for small batches
        if texts.len() < self.config.min_batch_size {
            return Err(GpuError::NotAvailable(format!(
                "Batch size {} below threshold {}, use CPU",
                texts.len(),
                self.config.min_batch_size
            )));
        }

        // Distribute across GPUs if multiple available
        if self.contexts.len() > 1 {
            self.encode_batch_multi_gpu(texts)
        } else {
            self.encode_batch_single_gpu(texts, &self.contexts[0])
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn encode_batch(&self, _texts: &[&str]) -> Result<Vec<Vec<u32>>, GpuError> {
        Err(GpuError::NotAvailable("CUDA not enabled".into()))
    }

    /// Encode batch on a single GPU
    #[cfg(feature = "cuda")]
    fn encode_batch_single_gpu(
        &self,
        texts: &[&str],
        ctx: &CudaContext,
    ) -> Result<Vec<Vec<u32>>, GpuError> {
        // Placeholder implementation
        // In production, this would:
        // 1. Pre-tokenize texts (find word boundaries)
        // 2. Upload words to GPU
        // 3. Run vocabulary lookup kernel
        // 4. Run WordPiece kernel for OOV words
        // 5. Download results

        // For now, return placeholder
        Ok(texts.iter().map(|_| vec![0u32; 10]).collect())
    }

    /// Encode batch across multiple GPUs
    #[cfg(feature = "cuda")]
    fn encode_batch_multi_gpu(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, GpuError> {
        let num_gpus = self.contexts.len();
        let chunk_size = (texts.len() + num_gpus - 1) / num_gpus;

        let mut all_results = Vec::with_capacity(texts.len());

        // Process chunks on each GPU
        // In production, this would use async streams for overlap
        for (i, chunk) in texts.chunks(chunk_size).enumerate() {
            let ctx = &self.contexts[i % num_gpus];
            let chunk_results = self.encode_batch_single_gpu(chunk, ctx)?;
            all_results.extend(chunk_results);
        }

        Ok(all_results)
    }

    /// Find optimal batch size through benchmarking
    #[cfg(feature = "cuda")]
    pub fn find_optimal_batch_size(&self) -> usize {
        let test_sizes = [8, 16, 32, 64, 128, 256];
        let mut best_size = 32;
        let mut best_throughput = 0.0f64;

        // Generate test data
        let test_text = "This is a test sentence for benchmarking tokenization throughput.";

        for &size in &test_sizes {
            if size > self.config.max_batch_size {
                break;
            }

            let texts: Vec<&str> = vec![test_text; size];

            // Warmup
            let _ = self.encode_batch(&texts);
            let _ = self.encode_batch(&texts);

            // Benchmark
            let start = Instant::now();
            let iterations = 10;
            for _ in 0..iterations {
                let _ = self.encode_batch(&texts);
            }
            let elapsed = start.elapsed();

            let throughput = (size * iterations) as f64 / elapsed.as_secs_f64();

            if throughput > best_throughput {
                best_throughput = throughput;
                best_size = size;
            }
        }

        best_size
    }

    #[cfg(not(feature = "cuda"))]
    pub fn find_optimal_batch_size(&self) -> usize {
        32 // Default
    }

    /// Get the number of available GPUs
    #[cfg(feature = "cuda")]
    pub fn num_gpus(&self) -> usize {
        self.contexts.len()
    }

    #[cfg(not(feature = "cuda"))]
    pub fn num_gpus(&self) -> usize {
        0
    }

    /// Get device information
    #[cfg(feature = "cuda")]
    pub fn device_info(&self) -> Vec<CudaDevice> {
        self.contexts
            .iter()
            .map(|ctx| ctx.device_info().clone())
            .collect()
    }

    /// Synchronize all GPU operations
    #[cfg(feature = "cuda")]
    pub fn synchronize(&self) -> Result<(), GpuError> {
        for ctx in &self.contexts {
            ctx.synchronize()?;
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn synchronize(&self) -> Result<(), GpuError> {
        Ok(())
    }
}

/// Multi-GPU load balancer
#[cfg(feature = "cuda")]
pub struct GpuLoadBalancer {
    contexts: Vec<Arc<CudaContext>>,
    pending_work: Vec<usize>, // Number of pending items per GPU
    strategy: LoadBalanceStrategy,
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalanceStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least loaded GPU first
    LeastLoaded,
    /// Fill one GPU before moving to next
    Sequential,
}

#[cfg(feature = "cuda")]
impl GpuLoadBalancer {
    /// Create a new load balancer
    pub fn new(contexts: Vec<Arc<CudaContext>>, strategy: LoadBalanceStrategy) -> Self {
        let pending_work = vec![0; contexts.len()];
        Self {
            contexts,
            pending_work,
            strategy,
        }
    }

    /// Get the next GPU to use
    pub fn next_gpu(&mut self) -> &CudaContext {
        let idx = match self.strategy {
            LoadBalanceStrategy::RoundRobin => {
                // Simple round-robin
                let min_idx = self
                    .pending_work
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &w)| w)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                min_idx
            }
            LoadBalanceStrategy::LeastLoaded => {
                // Find GPU with least pending work
                self.pending_work
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &w)| w)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
            LoadBalanceStrategy::Sequential => {
                // Fill GPUs in order
                self.pending_work
                    .iter()
                    .enumerate()
                    .find(|(_, &w)| w == 0)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
        };

        self.pending_work[idx] += 1;
        &self.contexts[idx]
    }

    /// Mark work as complete on a GPU
    pub fn work_complete(&mut self, device_id: i32) {
        for (i, ctx) in self.contexts.iter().enumerate() {
            if ctx.device_id() == device_id {
                self.pending_work[i] = self.pending_work[i].saturating_sub(1);
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_enum() {
        assert_ne!(GpuBackend::Cuda, GpuBackend::Metal);
        assert_ne!(GpuBackend::Metal, GpuBackend::Vulkan);
    }

    #[test]
    fn test_config_default() {
        let config = GpuTokenizerConfig::default();
        assert_eq!(config.min_batch_size, 8);
        assert_eq!(config.max_batch_size, 256);
        assert_eq!(config.max_seq_len, 512);
        assert!(config.auto_tune);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tokenizer_creation() {
        use crate::cuda::is_cuda_available;

        if !is_cuda_available() {
            println!("Skipping - no CUDA");
            return;
        }

        let tokenizer = GpuTokenizer::new(GpuBackend::Cuda);
        assert!(tokenizer.is_ok());

        let tokenizer = tokenizer.unwrap();
        assert_eq!(tokenizer.backend(), GpuBackend::Cuda);
        assert!(tokenizer.num_gpus() >= 1);
    }

    #[test]
    fn test_metal_not_available() {
        let result = GpuTokenizer::new(GpuBackend::Metal);
        assert!(result.is_err());
    }

    #[test]
    fn test_vulkan_not_available() {
        let result = GpuTokenizer::new(GpuBackend::Vulkan);
        assert!(result.is_err());
    }
}
