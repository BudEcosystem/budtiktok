//! Version and build information
//!
//! This module provides compile-time version information embedded by build.rs.

/// Version string from Cargo.toml
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Target architecture
#[cfg(target_arch = "x86_64")]
pub const TARGET_ARCH: &str = "x86_64";
#[cfg(target_arch = "aarch64")]
pub const TARGET_ARCH: &str = "aarch64";
#[cfg(target_arch = "x86")]
pub const TARGET_ARCH: &str = "x86";
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "x86")))]
pub const TARGET_ARCH: &str = "unknown";

/// Target operating system
#[cfg(target_os = "linux")]
pub const TARGET_OS: &str = "linux";
#[cfg(target_os = "macos")]
pub const TARGET_OS: &str = "macos";
#[cfg(target_os = "windows")]
pub const TARGET_OS: &str = "windows";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
pub const TARGET_OS: &str = "unknown";

/// Get git commit hash (short form)
pub fn git_hash() -> &'static str {
    option_env!("BUDTIKTOK_GIT_HASH").unwrap_or("unknown")
}

/// Check if git working directory was dirty during build
pub fn git_dirty() -> bool {
    option_env!("BUDTIKTOK_GIT_DIRTY") == Some("true")
}

/// Get build timestamp in ISO 8601 format
pub fn build_timestamp() -> &'static str {
    option_env!("BUDTIKTOK_BUILD_TIMESTAMP").unwrap_or("unknown")
}

/// Get build profile (debug/release)
pub fn build_profile() -> &'static str {
    option_env!("BUDTIKTOK_BUILD_PROFILE").unwrap_or("unknown")
}

/// Get Rust compiler version used for build
pub fn rustc_version() -> &'static str {
    option_env!("BUDTIKTOK_RUSTC_VERSION").unwrap_or("unknown")
}

// Keep legacy constants for backwards compatibility
pub const GIT_HASH: &str = "see git_hash()";
pub const BUILD_TIMESTAMP: &str = "see build_timestamp()";

/// Full version string including git hash
pub fn full_version() -> String {
    if git_dirty() {
        format!("{}-{}+dirty", VERSION, git_hash())
    } else {
        format!("{}-{}", VERSION, git_hash())
    }
}

/// Build information summary
#[derive(Debug, Clone)]
pub struct BuildInfo {
    pub version: &'static str,
    pub git_hash: &'static str,
    pub git_dirty: bool,
    pub build_timestamp: &'static str,
    pub build_profile: &'static str,
    pub rustc_version: &'static str,
    pub target_arch: &'static str,
    pub target_os: &'static str,
    pub simd_features: SimdFeatures,
}

impl BuildInfo {
    /// Get build information
    pub fn get() -> Self {
        Self {
            version: VERSION,
            git_hash: git_hash(),
            git_dirty: git_dirty(),
            build_timestamp: build_timestamp(),
            build_profile: build_profile(),
            rustc_version: rustc_version(),
            target_arch: TARGET_ARCH,
            target_os: TARGET_OS,
            simd_features: SimdFeatures::detect(),
        }
    }
}

impl std::fmt::Display for BuildInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "BudTikTok Core v{}", self.version)?;
        writeln!(f, "  Git: {}{}", self.git_hash, if self.git_dirty { " (dirty)" } else { "" })?;
        writeln!(f, "  Built: {}", self.build_timestamp)?;
        writeln!(f, "  Profile: {}", self.build_profile)?;
        writeln!(f, "  Rustc: {}", self.rustc_version)?;
        writeln!(f, "  Target: {} / {}", self.target_arch, self.target_os)?;
        write!(f, "  SIMD: {}", self.simd_features)
    }
}

/// SIMD feature detection
#[derive(Debug, Clone, Default)]
pub struct SimdFeatures {
    // x86_64 features
    pub avx512: bool,
    pub avx2: bool,
    pub sse42: bool,
    pub sse2: bool,

    // ARM features
    pub neon: bool,
    pub sve: bool,
    pub sve2: bool,
}

impl SimdFeatures {
    /// Detect available SIMD features at runtime
    pub fn detect() -> Self {
        let mut features = Self::default();

        #[cfg(target_arch = "x86_64")]
        {
            features.sse2 = true; // Always available on x86_64

            if is_x86_feature_detected!("sse4.2") {
                features.sse42 = true;
            }
            if is_x86_feature_detected!("avx2") {
                features.avx2 = true;
            }
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                features.avx512 = true;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            features.neon = true; // Always available on AArch64

            // SVE detection via library functions
            features.sve = crate::sve::is_sve_available();
            features.sve2 = crate::sve::is_sve2_available();
        }

        features
    }

    /// Get the best available SIMD tier
    pub fn best_tier(&self) -> SimdTier {
        if self.avx512 {
            SimdTier::Avx512
        } else if self.avx2 {
            SimdTier::Avx2
        } else if self.sve2 {
            SimdTier::Sve2
        } else if self.sve {
            SimdTier::Sve
        } else if self.neon {
            SimdTier::Neon
        } else if self.sse42 {
            SimdTier::Sse42
        } else if self.sse2 {
            SimdTier::Sse2
        } else {
            SimdTier::Scalar
        }
    }

    /// Get bytes processed per iteration for the best SIMD tier
    pub fn bytes_per_iteration(&self) -> usize {
        match self.best_tier() {
            SimdTier::Avx512 => 64,
            SimdTier::Avx2 => 32,
            SimdTier::Sve2 | SimdTier::Sve => {
                // SVE is variable length, return minimum
                #[cfg(target_arch = "aarch64")]
                {
                    let vl = crate::sve::get_sve_vector_length();
                    if vl > 0 { vl } else { 16 }
                }
                #[cfg(not(target_arch = "aarch64"))]
                { 16 }
            }
            SimdTier::Neon => 16,
            SimdTier::Sse42 | SimdTier::Sse2 => 16,
            SimdTier::Scalar => 8, // SWAR using u64
        }
    }
}

impl std::fmt::Display for SimdFeatures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut features = Vec::new();

        if self.avx512 {
            features.push("AVX-512");
        }
        if self.avx2 {
            features.push("AVX2");
        }
        if self.sse42 {
            features.push("SSE4.2");
        }
        if self.sse2 {
            features.push("SSE2");
        }
        if self.neon {
            features.push("NEON");
        }
        if self.sve2 {
            features.push("SVE2");
        } else if self.sve {
            features.push("SVE");
        }

        if features.is_empty() {
            write!(f, "scalar only")
        } else {
            write!(f, "{}", features.join(", "))
        }
    }
}

/// SIMD tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdTier {
    Scalar = 0,
    Sse2 = 1,
    Sse42 = 2,
    Neon = 3,
    Sve = 4,
    Avx2 = 5,
    Sve2 = 6,
    Avx512 = 7,
}

impl SimdTier {
    /// Get the name of this SIMD tier
    pub fn name(&self) -> &'static str {
        match self {
            SimdTier::Scalar => "Scalar",
            SimdTier::Sse2 => "SSE2",
            SimdTier::Sse42 => "SSE4.2",
            SimdTier::Neon => "NEON",
            SimdTier::Sve => "SVE",
            SimdTier::Avx2 => "AVX2",
            SimdTier::Sve2 => "SVE2",
            SimdTier::Avx512 => "AVX-512",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_full_version() {
        let v = full_version();
        assert!(v.starts_with(VERSION));
    }

    #[test]
    fn test_build_info() {
        let info = BuildInfo::get();
        assert_eq!(info.version, VERSION);
        println!("{}", info);
    }

    #[test]
    fn test_simd_features() {
        let features = SimdFeatures::detect();

        // At minimum, we should have scalar support
        let tier = features.best_tier();
        assert!(tier >= SimdTier::Scalar);

        // Check bytes per iteration is reasonable
        let bytes = features.bytes_per_iteration();
        assert!(bytes >= 8);
        assert!(bytes <= 64);
    }

    #[test]
    fn test_simd_tier_ordering() {
        assert!(SimdTier::Avx512 > SimdTier::Avx2);
        assert!(SimdTier::Avx2 > SimdTier::Sse42);
        assert!(SimdTier::Sse42 > SimdTier::Scalar);
    }

    #[test]
    fn test_simd_features_display() {
        let features = SimdFeatures::detect();
        let s = features.to_string();
        // Should contain at least something
        assert!(!s.is_empty());
    }
}
