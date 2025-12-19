//! SIMD backend abstraction

use crate::detect::{CpuFeatures, SimdBackendType};

/// SIMD capabilities
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    pub features: CpuFeatures,
    pub backend_type: SimdBackendType,
}

impl SimdCapabilities {
    /// Detect SIMD capabilities at runtime
    pub fn detect() -> Self {
        let features = CpuFeatures::detect();
        let backend_type = features.best_backend();
        Self {
            features,
            backend_type,
        }
    }

    /// Check if AVX-512 is available
    pub fn has_avx512(&self) -> bool {
        self.features.avx512f && self.features.avx512bw
    }

    /// Check if AVX2 is available
    pub fn has_avx2(&self) -> bool {
        self.features.avx2
    }

    /// Check if NEON is available
    pub fn has_neon(&self) -> bool {
        self.features.neon
    }
}

/// SIMD backend for text processing
pub struct SimdBackend {
    capabilities: SimdCapabilities,
}

impl SimdBackend {
    /// Create a new SIMD backend with auto-detection
    pub fn auto_detect() -> Self {
        Self {
            capabilities: SimdCapabilities::detect(),
        }
    }

    /// Create a backend with specific capabilities
    pub fn with_capabilities(capabilities: SimdCapabilities) -> Self {
        Self { capabilities }
    }

    /// Get the backend type
    pub fn backend_type(&self) -> SimdBackendType {
        self.capabilities.backend_type
    }

    /// Find whitespace positions
    pub fn find_whitespace(&self, text: &[u8]) -> Vec<usize> {
        match self.capabilities.backend_type {
            SimdBackendType::Avx512 => self.find_whitespace_avx512(text),
            SimdBackendType::Avx2 => self.find_whitespace_avx2(text),
            SimdBackendType::Sse42 => self.find_whitespace_sse42(text),
            SimdBackendType::Neon => self.find_whitespace_neon(text),
            _ => self.find_whitespace_scalar(text),
        }
    }

    fn find_whitespace_avx512(&self, text: &[u8]) -> Vec<usize> {
        // TODO: Implement AVX-512 whitespace detection
        self.find_whitespace_scalar(text)
    }

    fn find_whitespace_avx2(&self, text: &[u8]) -> Vec<usize> {
        // TODO: Implement AVX2 whitespace detection
        self.find_whitespace_scalar(text)
    }

    fn find_whitespace_sse42(&self, text: &[u8]) -> Vec<usize> {
        // TODO: Implement SSE4.2 whitespace detection
        self.find_whitespace_scalar(text)
    }

    fn find_whitespace_neon(&self, text: &[u8]) -> Vec<usize> {
        // TODO: Implement NEON whitespace detection
        self.find_whitespace_scalar(text)
    }

    fn find_whitespace_scalar(&self, text: &[u8]) -> Vec<usize> {
        crate::whitespace::find_whitespace_simd(text)
    }
}
