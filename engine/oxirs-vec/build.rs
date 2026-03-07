fn main() {
    // Declare custom cfg to avoid unexpected_cfgs warnings
    println!("cargo:rustc-check-cfg=cfg(cuda_runtime_available)");

    // Link against libnuma on Linux
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=numa");
    }

    // Detect CUDA availability and set cfg flag
    // This allows conditional compilation for CUDA-dependent code
    // Per COOLJAPAN Pure Rust Policy: CUDA is optional and must be feature-gated
    #[cfg(feature = "cuda")]
    {
        // Check if CUDA is actually available
        let cuda_available = std::process::Command::new("nvcc")
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);

        if cuda_available {
            println!("cargo:rustc-cfg=cuda_runtime_available");
            println!("cargo:warning=CUDA toolkit detected - GPU acceleration will be available");
        } else {
            // CUDA not available - build will succeed with CPU fallbacks
            println!("cargo:warning=CUDA feature enabled but toolkit not found");
            println!("cargo:warning=GPU operations will fall back to CPU implementations");
            println!("cargo:warning=For GPU acceleration, install CUDA toolkit from https://developer.nvidia.com/cuda-downloads");
            println!("cargo:warning=Or build without CUDA feature: cargo build -p oxirs-vec (default features are Pure Rust)");
        }
    }
}
