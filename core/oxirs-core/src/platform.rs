//! Platform capabilities detection for OxiRS
//!
//! This module provides unified platform detection across the OxiRS ecosystem.
//! All platform-specific code must use this module for capability detection.

use std::sync::OnceLock;

/// Platform capabilities detection result
#[derive(Debug, Clone)]
pub struct PlatformCapabilities {
    /// SIMD support available
    pub simd_available: bool,
    /// GPU support available
    pub gpu_available: bool,
    /// CUDA support available
    pub cuda_available: bool,
    /// OpenCL support available
    pub opencl_available: bool,
    /// Metal support available (macOS)
    pub metal_available: bool,
    /// AVX2 instructions available
    pub avx2_available: bool,
    /// AVX512 instructions available
    pub avx512_available: bool,
    /// ARM NEON instructions available
    pub neon_available: bool,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// CPU architecture
    pub arch: String,
    /// Operating system
    pub os: String,
}

// Cache the detected capabilities
static CAPABILITIES: OnceLock<PlatformCapabilities> = OnceLock::new();

impl PlatformCapabilities {
    /// Detect platform capabilities
    pub fn detect() -> &'static PlatformCapabilities {
        CAPABILITIES.get_or_init(|| {
            let mut caps = PlatformCapabilities {
                simd_available: false,
                gpu_available: false,
                cuda_available: false,
                opencl_available: false,
                metal_available: false,
                avx2_available: false,
                avx512_available: false,
                neon_available: false,
                cpu_cores: num_cpus::get(),
                arch: std::env::consts::ARCH.to_string(),
                os: std::env::consts::OS.to_string(),
            };

            // Detect SIMD capabilities
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                caps.simd_available = is_x86_feature_detected!("sse2");
                caps.avx2_available = is_x86_feature_detected!("avx2");
                caps.avx512_available = is_x86_feature_detected!("avx512f");
            }

            #[cfg(target_arch = "aarch64")]
            {
                caps.simd_available = true; // NEON is mandatory on aarch64
                caps.neon_available = true;
            }

            // Detect GPU capabilities
            caps.gpu_available = Self::detect_gpu();

            // Detect CUDA
            #[cfg(feature = "cuda")]
            {
                caps.cuda_available = Self::detect_cuda();
            }

            // Detect OpenCL
            #[cfg(feature = "opencl")]
            {
                caps.opencl_available = Self::detect_opencl();
            }

            // Detect Metal (macOS only)
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                caps.metal_available = Self::detect_metal();
            }

            caps
        })
    }

    /// Get a human-readable summary of capabilities
    pub fn summary(&self) -> String {
        let mut features = Vec::new();

        if self.simd_available {
            features.push("SIMD");

            if self.avx2_available {
                features.push("AVX2");
            }
            if self.avx512_available {
                features.push("AVX512");
            }
            if self.neon_available {
                features.push("NEON");
            }
        }

        if self.gpu_available {
            features.push("GPU");

            if self.cuda_available {
                features.push("CUDA");
            }
            if self.opencl_available {
                features.push("OpenCL");
            }
            if self.metal_available {
                features.push("Metal");
            }
        }

        format!(
            "{} ({} cores, {})",
            features.join(", "),
            self.cpu_cores,
            self.arch
        )
    }

    /// Check if any GPU is available
    fn detect_gpu() -> bool {
        // Simple heuristic - check for common GPU environment variables
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
            || std::env::var("GPU_DEVICE_ORDINAL").is_ok()
            || std::env::var("ROCR_VISIBLE_DEVICES").is_ok()
    }

    /// Check if CUDA is available
    #[cfg(feature = "cuda")]
    fn detect_cuda() -> bool {
        // Check for CUDA runtime
        std::env::var("CUDA_PATH").is_ok()
            || std::path::Path::new("/usr/local/cuda").exists()
            || std::path::Path::new("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA")
                .exists()
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(dead_code)]
    fn detect_cuda() -> bool {
        false
    }

    /// Check if OpenCL is available
    #[cfg(feature = "opencl")]
    #[allow(dead_code)]
    fn detect_opencl() -> bool {
        // Check for OpenCL libraries
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/usr/lib/libOpenCL.so").exists()
                || std::path::Path::new("/usr/lib64/libOpenCL.so").exists()
        }
        #[cfg(target_os = "windows")]
        {
            std::path::Path::new("C:\\Windows\\System32\\OpenCL.dll").exists()
        }
        #[cfg(target_os = "macos")]
        {
            true // OpenCL is included in macOS
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            false
        }
    }

    #[cfg(not(feature = "opencl"))]
    #[allow(dead_code)]
    fn detect_opencl() -> bool {
        false
    }

    /// Check if Metal is available
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[allow(dead_code)]
    fn detect_metal() -> bool {
        // Metal is available on all modern macOS systems
        true
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    #[allow(dead_code)]
    fn detect_metal() -> bool {
        false
    }
}

/// Auto-optimizer for selecting best implementation based on problem size
pub struct AutoOptimizer {
    capabilities: &'static PlatformCapabilities,
}

impl AutoOptimizer {
    /// Create a new auto-optimizer
    pub fn new() -> Self {
        Self {
            capabilities: PlatformCapabilities::detect(),
        }
    }

    /// Determine if GPU should be used based on problem size
    pub fn should_use_gpu(&self, problem_size: usize) -> bool {
        // Use GPU for large problems when available
        self.capabilities.gpu_available && problem_size > 100_000
    }

    /// Determine if SIMD should be used based on problem size
    pub fn should_use_simd(&self, problem_size: usize) -> bool {
        // Use SIMD for medium to large problems
        self.capabilities.simd_available && problem_size > 1000
    }

    /// Determine if parallel processing should be used
    pub fn should_use_parallel(&self, problem_size: usize) -> bool {
        // Use parallel processing for large problems on multi-core systems
        self.capabilities.cpu_cores > 1 && problem_size > 10_000
    }

    /// Get recommended chunk size for parallel processing
    pub fn recommended_chunk_size(&self, total_size: usize) -> usize {
        // Balance between parallelism overhead and work distribution
        let ideal_chunks = self.capabilities.cpu_cores * 4;
        let chunk_size = total_size / ideal_chunks;

        // Ensure reasonable chunk size
        chunk_size.clamp(1000, 100_000)
    }
}

impl Default for AutoOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let caps = PlatformCapabilities::detect();

        // Should have at least 1 CPU core
        assert!(caps.cpu_cores >= 1);

        // Should have valid architecture
        assert!(!caps.arch.is_empty());

        // Should have valid OS
        assert!(!caps.os.is_empty());

        println!("Platform capabilities: {}", caps.summary());
    }

    #[test]
    fn test_auto_optimizer() {
        let optimizer = AutoOptimizer::new();

        // Small problem sizes should not use GPU
        assert!(!optimizer.should_use_gpu(100));

        // Medium problem sizes might use SIMD
        let _ = optimizer.should_use_simd(5000);

        // Get chunk size recommendation
        let chunk_size = optimizer.recommended_chunk_size(1_000_000);
        assert!(chunk_size > 0);
    }
}
