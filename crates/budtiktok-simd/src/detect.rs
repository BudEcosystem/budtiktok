//! CPU feature detection for SIMD capabilities

/// Detected CPU features
#[derive(Debug, Clone, Default)]
pub struct CpuFeatures {
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub avx2: bool,
    pub sse42: bool,
    pub neon: bool,
    pub sve: bool,
    pub rvv: bool,
}

impl CpuFeatures {
    /// Detect CPU features at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
                avx512vl: std::arch::is_x86_feature_detected!("avx512vl"),
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                sse42: std::arch::is_x86_feature_detected!("sse4.2"),
                neon: false,
                sve: false,
                rvv: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx512f: false,
                avx512bw: false,
                avx512vl: false,
                avx2: false,
                sse42: false,
                neon: true, // NEON is mandatory on aarch64
                sve: false, // Would need runtime detection
                rvv: false,
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }

    /// Get the best available SIMD backend
    pub fn best_backend(&self) -> SimdBackendType {
        if self.avx512f && self.avx512bw {
            SimdBackendType::Avx512
        } else if self.avx2 {
            SimdBackendType::Avx2
        } else if self.sse42 {
            SimdBackendType::Sse42
        } else if self.neon {
            SimdBackendType::Neon
        } else if self.sve {
            SimdBackendType::Sve
        } else if self.rvv {
            SimdBackendType::Rvv
        } else {
            SimdBackendType::Scalar
        }
    }
}

/// SIMD backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackendType {
    Avx512,
    Avx2,
    Sse42,
    Neon,
    Sve,
    Rvv,
    Scalar,
}
