//! Runtime Configuration for BudTikTok
//!
//! This module provides configuration options for:
//! - ISA (Instruction Set Architecture) selection
//! - Parallelism and thread control
//! - Performance tuning
//!
//! # Example
//!
//! ```rust,ignore
//! use budtiktok_core::{RuntimeConfig, IsaSelection, ParallelismConfig};
//!
//! // Auto-detect everything (recommended)
//! let config = RuntimeConfig::auto();
//!
//! // Force specific ISA
//! let config = RuntimeConfig::new()
//!     .with_isa(IsaSelection::Avx2)
//!     .with_threads(4);
//!
//! // Single-threaded mode
//! let config = RuntimeConfig::single_threaded();
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use crate::simd_backends::{SimdCapabilities, get_capabilities};

/// Instruction Set Architecture selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsaSelection {
    /// Automatically detect and use the best available ISA
    #[default]
    Auto,

    /// Force scalar (no SIMD) implementation
    Scalar,

    // x86_64 ISAs
    /// SSE 4.2 (128-bit vectors)
    Sse42,
    /// AVX2 (256-bit vectors) - recommended for most x86_64
    Avx2,
    /// AVX-512 (512-bit vectors) - best for server CPUs
    Avx512,

    // ARM ISAs
    /// NEON (128-bit vectors) - standard on all aarch64
    Neon,
    /// SVE (Scalable Vector Extension) - Arm v8.2+
    Sve,
    /// SVE2 (Scalable Vector Extension 2) - Arm v9+
    Sve2,
}

impl IsaSelection {
    /// Check if this ISA is available on the current platform
    pub fn is_available(&self) -> bool {
        let caps = get_capabilities();
        match self {
            IsaSelection::Auto => true,
            IsaSelection::Scalar => true,
            IsaSelection::Sse42 => caps.sse42,
            IsaSelection::Avx2 => caps.avx2,
            IsaSelection::Avx512 => caps.avx512bw,
            IsaSelection::Neon => caps.neon,
            IsaSelection::Sve => caps.sve,
            IsaSelection::Sve2 => caps.sve2,
        }
    }

    /// Get the best available ISA for the current platform
    pub fn best_available() -> Self {
        let caps = get_capabilities();

        #[cfg(target_arch = "x86_64")]
        {
            if caps.avx512bw {
                return IsaSelection::Avx512;
            }
            if caps.avx2 {
                return IsaSelection::Avx2;
            }
            if caps.sse42 {
                return IsaSelection::Sse42;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if caps.sve2 {
                return IsaSelection::Sve2;
            }
            if caps.sve {
                return IsaSelection::Sve;
            }
            if caps.neon {
                return IsaSelection::Neon;
            }
        }

        IsaSelection::Scalar
    }

    /// Get a human-readable name for this ISA
    pub fn name(&self) -> &'static str {
        match self {
            IsaSelection::Auto => "Auto",
            IsaSelection::Scalar => "Scalar",
            IsaSelection::Sse42 => "SSE4.2",
            IsaSelection::Avx2 => "AVX2",
            IsaSelection::Avx512 => "AVX-512",
            IsaSelection::Neon => "NEON",
            IsaSelection::Sve => "SVE",
            IsaSelection::Sve2 => "SVE2",
        }
    }

    /// Get the effective ISA (resolves Auto to actual ISA)
    pub fn effective(&self) -> Self {
        match self {
            IsaSelection::Auto => Self::best_available(),
            other => *other,
        }
    }

    /// Parse from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "auto" => Some(IsaSelection::Auto),
            "scalar" | "none" => Some(IsaSelection::Scalar),
            "sse42" | "sse4.2" => Some(IsaSelection::Sse42),
            "avx2" => Some(IsaSelection::Avx2),
            "avx512" | "avx-512" => Some(IsaSelection::Avx512),
            "neon" => Some(IsaSelection::Neon),
            "sve" => Some(IsaSelection::Sve),
            "sve2" => Some(IsaSelection::Sve2),
            _ => None,
        }
    }
}

impl std::fmt::Display for IsaSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Parallelism configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismConfig {
    /// Single-threaded execution
    SingleThreaded,

    /// Use a specific number of threads
    Threads(usize),

    /// Automatically determine the best number of threads
    /// Uses all available cores by default
    Auto,

    /// Use all available CPU cores
    AllCores,
}

impl Default for ParallelismConfig {
    fn default() -> Self {
        ParallelismConfig::Auto
    }
}

impl ParallelismConfig {
    /// Get the effective number of threads
    pub fn effective_threads(&self) -> usize {
        match self {
            ParallelismConfig::SingleThreaded => 1,
            ParallelismConfig::Threads(n) => (*n).max(1),
            ParallelismConfig::Auto | ParallelismConfig::AllCores => {
                num_cpus::get()
            }
        }
    }

    /// Check if parallelism is enabled
    pub fn is_parallel(&self) -> bool {
        self.effective_threads() > 1
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "single" | "1" | "single-threaded" => Some(ParallelismConfig::SingleThreaded),
            "auto" => Some(ParallelismConfig::Auto),
            "all" | "all-cores" => Some(ParallelismConfig::AllCores),
            _ => s.parse::<usize>().ok().map(ParallelismConfig::Threads),
        }
    }
}

impl std::fmt::Display for ParallelismConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParallelismConfig::SingleThreaded => write!(f, "Single-threaded"),
            ParallelismConfig::Threads(n) => write!(f, "{} threads", n),
            ParallelismConfig::Auto => write!(f, "Auto ({} threads)", self.effective_threads()),
            ParallelismConfig::AllCores => write!(f, "All cores ({} threads)", self.effective_threads()),
        }
    }
}

/// Runtime configuration for tokenization
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// ISA selection for SIMD operations
    pub isa: IsaSelection,

    /// Parallelism configuration
    pub parallelism: ParallelismConfig,

    /// Enable SIMD for pre-tokenization (splitting text into words)
    pub simd_pretokenization: bool,

    /// Enable SIMD for normalization (lowercasing, etc.)
    pub simd_normalization: bool,

    /// Enable word-level caching for BPE
    pub enable_cache: bool,

    /// Maximum cache size (number of entries)
    pub cache_size: usize,

    /// Batch size for parallel processing
    pub batch_size: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::auto()
    }
}

impl RuntimeConfig {
    /// Create a new configuration with default settings
    pub fn new() -> Self {
        Self {
            isa: IsaSelection::Auto,
            parallelism: ParallelismConfig::Auto,
            simd_pretokenization: true,
            simd_normalization: true,
            enable_cache: true,
            cache_size: 10000,
            batch_size: 1000,
        }
    }

    /// Create an auto-detecting configuration (recommended)
    ///
    /// This configuration:
    /// - Automatically detects the best ISA for your CPU
    /// - Uses all available CPU cores
    /// - Enables all optimizations
    pub fn auto() -> Self {
        Self::new()
    }

    /// Create a single-threaded configuration
    ///
    /// Useful for:
    /// - Embedding in single-threaded applications
    /// - Debugging and profiling
    /// - Environments where threading is not allowed
    pub fn single_threaded() -> Self {
        Self {
            parallelism: ParallelismConfig::SingleThreaded,
            ..Self::new()
        }
    }

    /// Create a scalar-only configuration (no SIMD)
    ///
    /// Useful for:
    /// - Maximum compatibility
    /// - Debugging SIMD issues
    /// - Baseline benchmarking
    pub fn scalar() -> Self {
        Self {
            isa: IsaSelection::Scalar,
            simd_pretokenization: false,
            simd_normalization: false,
            ..Self::new()
        }
    }

    /// Create a high-throughput server configuration
    ///
    /// Optimized for:
    /// - Maximum throughput on server workloads
    /// - Large batch processing
    pub fn server() -> Self {
        Self {
            isa: IsaSelection::Auto,
            parallelism: ParallelismConfig::AllCores,
            simd_pretokenization: true,
            simd_normalization: true,
            enable_cache: true,
            cache_size: 50000,
            batch_size: 5000,
        }
    }

    /// Set the ISA selection
    pub fn with_isa(mut self, isa: IsaSelection) -> Self {
        self.isa = isa;
        self
    }

    /// Set the number of threads
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.parallelism = if threads <= 1 {
            ParallelismConfig::SingleThreaded
        } else {
            ParallelismConfig::Threads(threads)
        };
        self
    }

    /// Set parallelism configuration
    pub fn with_parallelism(mut self, parallelism: ParallelismConfig) -> Self {
        self.parallelism = parallelism;
        self
    }

    /// Enable or disable SIMD pre-tokenization
    pub fn with_simd_pretokenization(mut self, enable: bool) -> Self {
        self.simd_pretokenization = enable;
        self
    }

    /// Enable or disable SIMD normalization
    pub fn with_simd_normalization(mut self, enable: bool) -> Self {
        self.simd_normalization = enable;
        self
    }

    /// Enable or disable word-level caching
    pub fn with_cache(mut self, enable: bool) -> Self {
        self.enable_cache = enable;
        self
    }

    /// Set cache size
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    /// Set batch size for parallel processing
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Get the effective ISA being used
    pub fn effective_isa(&self) -> IsaSelection {
        self.isa.effective()
    }

    /// Get the effective number of threads
    pub fn effective_threads(&self) -> usize {
        self.parallelism.effective_threads()
    }

    /// Check if SIMD is enabled
    pub fn simd_enabled(&self) -> bool {
        self.isa != IsaSelection::Scalar &&
        (self.simd_pretokenization || self.simd_normalization)
    }

    /// Get a summary of the configuration
    pub fn summary(&self) -> String {
        format!(
            "ISA: {}, Threads: {}, Cache: {}, SIMD Pre-tok: {}, SIMD Norm: {}",
            self.effective_isa(),
            self.effective_threads(),
            if self.enable_cache { "enabled" } else { "disabled" },
            self.simd_pretokenization,
            self.simd_normalization
        )
    }

    /// Validate the configuration and return any warnings
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // Check ISA availability
        if !self.isa.is_available() {
            warnings.push(format!(
                "Selected ISA {} is not available on this platform. Will fall back to {}.",
                self.isa,
                IsaSelection::best_available()
            ));
        }

        // Check thread count
        let cpu_count = num_cpus::get();
        if let ParallelismConfig::Threads(n) = self.parallelism {
            if n > cpu_count * 2 {
                warnings.push(format!(
                    "Thread count ({}) exceeds 2x CPU count ({}). This may cause performance degradation.",
                    n, cpu_count
                ));
            }
        }

        // Check cache settings
        if self.enable_cache && self.cache_size < 100 {
            warnings.push("Cache size is very small. Consider increasing for better performance.".to_string());
        }

        warnings
    }
}

impl std::fmt::Display for RuntimeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "BudTikTok Runtime Configuration")?;
        writeln!(f, "  ISA: {} (effective: {})", self.isa, self.effective_isa())?;
        writeln!(f, "  Parallelism: {}", self.parallelism)?;
        writeln!(f, "  SIMD Pre-tokenization: {}", self.simd_pretokenization)?;
        writeln!(f, "  SIMD Normalization: {}", self.simd_normalization)?;
        writeln!(f, "  Cache: {} (size: {})",
            if self.enable_cache { "enabled" } else { "disabled" },
            self.cache_size)?;
        writeln!(f, "  Batch Size: {}", self.batch_size)?;
        Ok(())
    }
}

// Global runtime configuration
static GLOBAL_THREADS: AtomicUsize = AtomicUsize::new(0);

/// Set the global thread pool size for Rayon
pub fn set_global_threads(threads: usize) {
    GLOBAL_THREADS.store(threads, Ordering::SeqCst);

    // Initialize Rayon thread pool if not already done
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();
}

/// Get the current global thread count
pub fn get_global_threads() -> usize {
    let stored = GLOBAL_THREADS.load(Ordering::SeqCst);
    if stored == 0 {
        num_cpus::get()
    } else {
        stored
    }
}

/// Apply runtime configuration globally
pub fn apply_config(config: &RuntimeConfig) {
    set_global_threads(config.effective_threads());
}

/// Get system information
pub fn system_info() -> SystemInfo {
    SystemInfo::detect()
}

/// System information
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// Number of physical CPU cores
    pub physical_cores: usize,
    /// Number of logical CPU cores (with hyperthreading)
    pub logical_cores: usize,
    /// Detected SIMD capabilities
    pub simd_capabilities: SimdCapabilities,
    /// Best available ISA
    pub best_isa: IsaSelection,
    /// Target architecture
    pub arch: &'static str,
    /// Target OS
    pub os: &'static str,
}

impl SystemInfo {
    /// Detect system information
    pub fn detect() -> Self {
        let logical_cores = num_cpus::get();
        let physical_cores = num_cpus::get_physical();
        let simd_capabilities = get_capabilities();
        let best_isa = IsaSelection::best_available();

        Self {
            physical_cores,
            logical_cores,
            simd_capabilities,
            best_isa,
            arch: std::env::consts::ARCH,
            os: std::env::consts::OS,
        }
    }
}

impl std::fmt::Display for SystemInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "System Information")?;
        writeln!(f, "  Architecture: {}", self.arch)?;
        writeln!(f, "  OS: {}", self.os)?;
        writeln!(f, "  Physical Cores: {}", self.physical_cores)?;
        writeln!(f, "  Logical Cores: {}", self.logical_cores)?;
        writeln!(f, "  Best ISA: {}", self.best_isa)?;
        writeln!(f, "  SIMD Capabilities:")?;

        #[cfg(target_arch = "x86_64")]
        {
            writeln!(f, "    SSE4.2: {}", self.simd_capabilities.sse42)?;
            writeln!(f, "    AVX2: {}", self.simd_capabilities.avx2)?;
            writeln!(f, "    AVX-512F: {}", self.simd_capabilities.avx512f)?;
            writeln!(f, "    AVX-512BW: {}", self.simd_capabilities.avx512bw)?;
        }

        #[cfg(target_arch = "aarch64")]
        {
            writeln!(f, "    NEON: {}", self.simd_capabilities.neon)?;
            writeln!(f, "    SVE: {}", self.simd_capabilities.sve)?;
            writeln!(f, "    SVE2: {}", self.simd_capabilities.sve2)?;
            if self.simd_capabilities.sve {
                writeln!(f, "    SVE Vector Length: {} bytes", self.simd_capabilities.sve_vector_length)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isa_selection_auto() {
        let isa = IsaSelection::Auto;
        assert!(isa.is_available());
        let effective = isa.effective();
        assert_ne!(effective, IsaSelection::Auto);
    }

    #[test]
    fn test_isa_selection_from_str() {
        assert_eq!(IsaSelection::from_str("auto"), Some(IsaSelection::Auto));
        assert_eq!(IsaSelection::from_str("AVX2"), Some(IsaSelection::Avx2));
        assert_eq!(IsaSelection::from_str("scalar"), Some(IsaSelection::Scalar));
        assert_eq!(IsaSelection::from_str("invalid"), None);
    }

    #[test]
    fn test_parallelism_config() {
        assert_eq!(ParallelismConfig::SingleThreaded.effective_threads(), 1);
        assert_eq!(ParallelismConfig::Threads(4).effective_threads(), 4);
        assert!(ParallelismConfig::Auto.effective_threads() >= 1);
    }

    #[test]
    fn test_runtime_config_default() {
        let config = RuntimeConfig::default();
        assert_eq!(config.isa, IsaSelection::Auto);
        assert!(config.simd_pretokenization);
        assert!(config.enable_cache);
    }

    #[test]
    fn test_runtime_config_builder() {
        let config = RuntimeConfig::new()
            .with_isa(IsaSelection::Avx2)
            .with_threads(4)
            .with_cache(false);

        assert_eq!(config.isa, IsaSelection::Avx2);
        assert_eq!(config.parallelism, ParallelismConfig::Threads(4));
        assert!(!config.enable_cache);
    }

    #[test]
    fn test_system_info() {
        let info = SystemInfo::detect();
        assert!(info.physical_cores >= 1);
        assert!(info.logical_cores >= info.physical_cores);
        println!("{}", info);
    }

    #[test]
    fn test_config_validate() {
        let config = RuntimeConfig::new()
            .with_threads(1000)
            .with_cache_size(10);

        let warnings = config.validate();
        assert!(!warnings.is_empty());
    }
}
