//! Build script for budtiktok-core (1.1.4)
//!
//! This build script handles:
//! - SIMD feature detection at compile time
//! - Version information embedding
//! - Git hash embedding
//! - Target architecture detection

use std::env;
use std::process::Command;

fn main() {
    // Rerun if build script changes
    println!("cargo:rerun-if-changed=build.rs");

    // Detect and emit SIMD capabilities
    detect_simd_features();

    // Embed version information
    embed_version_info();

    // Embed git hash
    embed_git_hash();

    // Detect target architecture
    detect_target_arch();
}

/// Detect SIMD features available on the target
fn detect_simd_features() {
    let target = env::var("TARGET").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    // x86_64 SIMD features
    if target_arch == "x86_64" {
        // Check for AVX-512 support
        // Note: Runtime detection is still needed, but we can enable compilation
        if target_supports_feature(&target, "avx512f") {
            println!("cargo:rustc-cfg=has_avx512");
        }

        // AVX2 is widely available on modern x86_64
        if target_supports_feature(&target, "avx2") {
            println!("cargo:rustc-cfg=has_avx2");
        }

        // SSE4.2 is available on virtually all x86_64 CPUs
        if target_supports_feature(&target, "sse4.2") {
            println!("cargo:rustc-cfg=has_sse42");
        }

        // Always available on x86_64
        println!("cargo:rustc-cfg=has_sse2");
    }

    // ARM64 SIMD features
    if target_arch == "aarch64" {
        // NEON is mandatory on AArch64
        println!("cargo:rustc-cfg=has_neon");

        // SVE detection (compile-time hint, runtime check still needed)
        if target_supports_feature(&target, "sve") {
            println!("cargo:rustc-cfg=has_sve");
        }

        // SVE2 detection
        if target_supports_feature(&target, "sve2") {
            println!("cargo:rustc-cfg=has_sve2");
        }
    }

    // Emit target feature summary
    println!(
        "cargo:warning=Building for target: {} (arch: {})",
        target, target_arch
    );
}

/// Check if a target supports a specific CPU feature
fn target_supports_feature(target: &str, feature: &str) -> bool {
    // Check CARGO_CFG_TARGET_FEATURE environment variable
    if let Ok(features) = env::var("CARGO_CFG_TARGET_FEATURE") {
        if features.split(',').any(|f| f.trim() == feature) {
            return true;
        }
    }

    // Check RUSTFLAGS for target-feature
    if let Ok(rustflags) = env::var("RUSTFLAGS") {
        if rustflags.contains(&format!("+{}", feature)) {
            return true;
        }
    }

    // Default features for known targets
    match (target, feature) {
        // Modern x86_64 targets typically support these
        (t, "sse4.2") if t.contains("x86_64") => true,
        (t, "avx2") if t.contains("x86_64") && !t.contains("i686") => {
            // Most modern x86_64 CPUs support AVX2
            // But don't assume for generic targets
            false
        }
        // AArch64 always has NEON
        (t, "neon") if t.contains("aarch64") => true,
        _ => false,
    }
}

/// Embed version information
fn embed_version_info() {
    // Package version from Cargo.toml
    let version = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=BUDTIKTOK_VERSION={}", version);

    // Build timestamp
    let timestamp = chrono_lite_timestamp();
    println!("cargo:rustc-env=BUDTIKTOK_BUILD_TIMESTAMP={}", timestamp);

    // Build profile
    let profile = env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=BUDTIKTOK_BUILD_PROFILE={}", profile);

    // Rust version
    if let Some(rustc_version) = get_rustc_version() {
        println!("cargo:rustc-env=BUDTIKTOK_RUSTC_VERSION={}", rustc_version);
    }
}

/// Embed git hash for traceability
fn embed_git_hash() {
    // Rerun if git HEAD changes
    if std::path::Path::new(".git/HEAD").exists() {
        println!("cargo:rerun-if-changed=.git/HEAD");
    }

    // Get git commit hash
    let git_hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=BUDTIKTOK_GIT_HASH={}", git_hash);

    // Check if working directory is dirty
    let is_dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|output| !output.stdout.is_empty())
        .unwrap_or(false);

    println!(
        "cargo:rustc-env=BUDTIKTOK_GIT_DIRTY={}",
        if is_dirty { "true" } else { "false" }
    );
}

/// Detect and emit target architecture information
fn detect_target_arch() {
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    println!("cargo:rustc-env=BUDTIKTOK_TARGET_ARCH={}", target_arch);
    println!("cargo:rustc-env=BUDTIKTOK_TARGET_OS={}", target_os);
    println!("cargo:rustc-env=BUDTIKTOK_TARGET_ENV={}", target_env);

    // Emit cfg for specific architectures
    match target_arch.as_str() {
        "x86_64" => println!("cargo:rustc-cfg=target_arch_x86_64"),
        "aarch64" => println!("cargo:rustc-cfg=target_arch_aarch64"),
        "x86" => println!("cargo:rustc-cfg=target_arch_x86"),
        _ => {}
    }

    // Emit cfg for specific OS
    match target_os.as_str() {
        "linux" => println!("cargo:rustc-cfg=target_os_linux"),
        "macos" => println!("cargo:rustc-cfg=target_os_macos"),
        "windows" => println!("cargo:rustc-cfg=target_os_windows"),
        _ => {}
    }
}

/// Get rustc version
fn get_rustc_version() -> Option<String> {
    Command::new("rustc")
        .args(["--version"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .map(|s| s.trim().to_string())
}

/// Simple timestamp without external dependencies
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    // Convert to a simple ISO-like format
    let secs = duration.as_secs();

    // Calculate date components (simplified)
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;

    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Simplified year calculation (good enough for build timestamps)
    let mut year = 1970;
    let mut remaining_days = days_since_epoch as i64;

    while remaining_days >= days_in_year(year) {
        remaining_days -= days_in_year(year);
        year += 1;
    }

    let mut month = 1;
    while remaining_days >= days_in_month(year, month) {
        remaining_days -= days_in_month(year, month);
        month += 1;
    }

    let day = remaining_days + 1;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn days_in_year(year: i64) -> i64 {
    if is_leap_year(year) {
        366
    } else {
        365
    }
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 => 31,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        3 => 31,
        4 => 30,
        5 => 31,
        6 => 30,
        7 => 31,
        8 => 31,
        9 => 30,
        10 => 31,
        11 => 30,
        12 => 31,
        _ => 30,
    }
}
