//! Auto-Configuration for Maximum Performance
//!
//! This module provides automatic configuration that:
//! - Detects hardware capabilities (SIMD, cores, cache sizes)
//! - Selects optimal algorithms based on hardware
//! - Auto-tunes based on observed workload patterns
//! - Maintains 100% HF tokenizer API compatibility
//!
//! # Usage
//!
//! Auto-configuration is applied automatically when loading tokenizers.
//! No code changes are required.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

use crate::runtime::{IsaSelection, ParallelismConfig, RuntimeConfig};
use crate::simd_backends::{get_capabilities, SimdCapabilities};

/// Global auto-configuration instance (initialized once on first use)
static AUTO_CONFIG: OnceLock<AutoConfig> = OnceLock::new();

/// Get the global auto-configuration
pub fn get_auto_config() -> &'static AutoConfig {
    AUTO_CONFIG.get_or_init(AutoConfig::detect)
}

/// Initialize auto-configuration with custom settings
pub fn init_auto_config(config: AutoConfig) -> Result<(), AutoConfig> {
    AUTO_CONFIG.set(config)
}

/// Auto-detected and auto-tuned configuration
#[derive(Debug)]
pub struct AutoConfig {
    /// Detected SIMD capabilities
    pub simd: SimdCapabilities,

    /// Best available ISA
    pub best_isa: IsaSelection,

    /// Number of physical CPU cores
    pub physical_cores: usize,

    /// Number of logical CPU cores (with hyperthreading)
    pub logical_cores: usize,

    /// Recommended batch size for parallel processing
    pub recommended_batch_size: usize,

    /// Whether to use SIMD for pretokenization
    pub use_simd_pretokenizer: bool,

    /// Whether to use SIMD for normalization
    pub use_simd_normalizer: bool,

    /// Cache size for word-level caching
    pub cache_size: usize,

    /// Runtime statistics for auto-tuning
    stats: RuntimeStats,
}

/// Runtime statistics for workload-aware tuning
#[derive(Debug, Default)]
struct RuntimeStats {
    /// Total encode calls
    encode_calls: AtomicU64,

    /// Total batch encode calls
    batch_encode_calls: AtomicU64,

    /// Average batch size observed
    avg_batch_size: AtomicUsize,

    /// Average text length observed
    avg_text_length: AtomicUsize,

    /// Peak concurrent requests observed
    peak_concurrency: AtomicUsize,
}

impl AutoConfig {
    /// Auto-detect optimal configuration for current hardware
    pub fn detect() -> Self {
        let simd = get_capabilities();
        let best_isa = IsaSelection::best_available();
        let physical_cores = num_cpus::get_physical();
        let logical_cores = num_cpus::get();

        // Determine optimal batch size based on cores
        // Use physical cores for CPU-bound work, logical for I/O-bound
        let recommended_batch_size = Self::calculate_batch_size(physical_cores, logical_cores);

        // Enable SIMD pretokenizer if we have at least SSE4.2 (x86) or NEON (ARM)
        let use_simd_pretokenizer = Self::should_use_simd_pretokenizer(&simd);

        // Enable SIMD normalizer for ASCII-heavy workloads (lowercase, etc.)
        let use_simd_normalizer = Self::should_use_simd_normalizer(&simd);

        // Cache size based on available memory and cores
        let cache_size = Self::calculate_cache_size(physical_cores);

        Self {
            simd,
            best_isa,
            physical_cores,
            logical_cores,
            recommended_batch_size,
            use_simd_pretokenizer,
            use_simd_normalizer,
            cache_size,
            stats: RuntimeStats::default(),
        }
    }

    /// Calculate optimal batch size
    fn calculate_batch_size(physical_cores: usize, logical_cores: usize) -> usize {
        // For tokenization, we want to balance:
        // - Too small: overhead of parallel dispatch
        // - Too large: memory pressure and cache thrashing

        // Good heuristic: 64-256 items per core for tokenization
        let items_per_core = if physical_cores >= 8 {
            128 // More cores = smaller batches per core to reduce contention
        } else if physical_cores >= 4 {
            192
        } else {
            256
        };

        (physical_cores * items_per_core).max(64).min(4096)
    }

    /// Determine if SIMD pretokenizer should be used
    fn should_use_simd_pretokenizer(simd: &SimdCapabilities) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            // Use SIMD if we have at least AVX2 (256-bit)
            simd.avx2
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on aarch64
            simd.neon
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    /// Determine if SIMD normalizer should be used
    fn should_use_simd_normalizer(simd: &SimdCapabilities) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            simd.avx2
        }

        #[cfg(target_arch = "aarch64")]
        {
            simd.neon
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    /// Calculate optimal cache size
    fn calculate_cache_size(physical_cores: usize) -> usize {
        // Base cache size per core, scaled by core count
        // More cores = more potential concurrent lookups
        let base_size = 2048;
        let scaled = base_size * physical_cores;

        // Cap at reasonable maximum (memory usage)
        scaled.max(4096).min(100_000)
    }

    /// Record an encode call for statistics
    #[inline]
    pub fn record_encode(&self, text_length: usize) {
        self.stats.encode_calls.fetch_add(1, Ordering::Relaxed);

        // Update average text length with exponential moving average
        let current = self.stats.avg_text_length.load(Ordering::Relaxed);
        if current == 0 {
            self.stats.avg_text_length.store(text_length, Ordering::Relaxed);
        } else {
            // EMA with alpha = 0.1
            let new_avg = (current * 9 + text_length) / 10;
            self.stats.avg_text_length.store(new_avg, Ordering::Relaxed);
        }
    }

    /// Record a batch encode call for statistics
    #[inline]
    pub fn record_batch_encode(&self, batch_size: usize) {
        self.stats.batch_encode_calls.fetch_add(1, Ordering::Relaxed);

        // Update average batch size with EMA
        let current = self.stats.avg_batch_size.load(Ordering::Relaxed);
        if current == 0 {
            self.stats.avg_batch_size.store(batch_size, Ordering::Relaxed);
        } else {
            let new_avg = (current * 9 + batch_size) / 10;
            self.stats.avg_batch_size.store(new_avg, Ordering::Relaxed);
        }
    }

    /// Update peak concurrency observation
    #[inline]
    pub fn record_concurrency(&self, current_concurrency: usize) {
        let peak = self.stats.peak_concurrency.load(Ordering::Relaxed);
        if current_concurrency > peak {
            self.stats.peak_concurrency.store(current_concurrency, Ordering::Relaxed);
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> AutoConfigStats {
        AutoConfigStats {
            encode_calls: self.stats.encode_calls.load(Ordering::Relaxed),
            batch_encode_calls: self.stats.batch_encode_calls.load(Ordering::Relaxed),
            avg_batch_size: self.stats.avg_batch_size.load(Ordering::Relaxed),
            avg_text_length: self.stats.avg_text_length.load(Ordering::Relaxed),
            peak_concurrency: self.stats.peak_concurrency.load(Ordering::Relaxed),
        }
    }

    /// Get optimal parallelism for current workload
    pub fn get_optimal_parallelism(&self, batch_size: usize) -> ParallelismConfig {
        if batch_size <= 1 {
            // Single item - no parallelism benefit
            ParallelismConfig::SingleThreaded
        } else if batch_size < self.physical_cores {
            // Small batch - use as many threads as items
            ParallelismConfig::Threads(batch_size)
        } else {
            // Large batch - use all physical cores
            ParallelismConfig::Threads(self.physical_cores)
        }
    }

    /// Get optimal RuntimeConfig for current workload
    pub fn get_runtime_config(&self) -> RuntimeConfig {
        RuntimeConfig {
            isa: self.best_isa,
            parallelism: ParallelismConfig::Auto,
            simd_pretokenization: self.use_simd_pretokenizer,
            simd_normalization: self.use_simd_normalizer,
            enable_cache: true,
            cache_size: self.cache_size,
            batch_size: self.recommended_batch_size,
        }
    }

    /// Check if SIMD should be used for a given text length
    #[inline]
    pub fn should_use_simd_for_text(&self, text_length: usize) -> bool {
        // SIMD has overhead - only worth it for texts >= 32 bytes
        // For x86 AVX2: 32-byte vectors, for ARM NEON: 16-byte vectors
        #[cfg(target_arch = "x86_64")]
        {
            self.use_simd_pretokenizer && text_length >= 32
        }

        #[cfg(target_arch = "aarch64")]
        {
            self.use_simd_pretokenizer && text_length >= 16
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }
}

/// Statistics snapshot
#[derive(Debug, Clone)]
pub struct AutoConfigStats {
    pub encode_calls: u64,
    pub batch_encode_calls: u64,
    pub avg_batch_size: usize,
    pub avg_text_length: usize,
    pub peak_concurrency: usize,
}

impl std::fmt::Display for AutoConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "BudTikTok Auto-Configuration")?;
        writeln!(f, "  ISA: {} (best available)", self.best_isa)?;
        writeln!(f, "  Physical cores: {}", self.physical_cores)?;
        writeln!(f, "  Logical cores: {}", self.logical_cores)?;
        writeln!(f, "  SIMD pretokenizer: {}", self.use_simd_pretokenizer)?;
        writeln!(f, "  SIMD normalizer: {}", self.use_simd_normalizer)?;
        writeln!(f, "  Recommended batch size: {}", self.recommended_batch_size)?;
        writeln!(f, "  Cache size: {}", self.cache_size)?;

        #[cfg(target_arch = "x86_64")]
        {
            writeln!(f, "  SIMD Capabilities:")?;
            writeln!(f, "    SSE4.2: {}", self.simd.sse42)?;
            writeln!(f, "    AVX2: {}", self.simd.avx2)?;
            writeln!(f, "    AVX-512F: {}", self.simd.avx512f)?;
            writeln!(f, "    AVX-512BW: {}", self.simd.avx512bw)?;
        }

        #[cfg(target_arch = "aarch64")]
        {
            writeln!(f, "  SIMD Capabilities:")?;
            writeln!(f, "    NEON: {}", self.simd.neon)?;
            writeln!(f, "    SVE: {}", self.simd.sve)?;
            writeln!(f, "    SVE2: {}", self.simd.sve2)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_detect() {
        let config = AutoConfig::detect();

        assert!(config.physical_cores >= 1);
        assert!(config.logical_cores >= config.physical_cores);
        assert!(config.recommended_batch_size >= 64);
        assert!(config.cache_size >= 4096);

        println!("{}", config);
    }

    #[test]
    fn test_get_global_config() {
        let config = get_auto_config();
        assert!(config.physical_cores >= 1);
    }

    #[test]
    fn test_record_stats() {
        let config = AutoConfig::detect();

        config.record_encode(100);
        config.record_encode(200);
        config.record_batch_encode(32);

        let stats = config.get_stats();
        assert_eq!(stats.encode_calls, 2);
        assert_eq!(stats.batch_encode_calls, 1);
    }

    #[test]
    fn test_optimal_parallelism() {
        let config = AutoConfig::detect();

        // Single item - no parallelism
        let p1 = config.get_optimal_parallelism(1);
        assert_eq!(p1.effective_threads(), 1);

        // Large batch - use all cores
        let p100 = config.get_optimal_parallelism(100);
        assert!(p100.effective_threads() >= 1);
    }
}
