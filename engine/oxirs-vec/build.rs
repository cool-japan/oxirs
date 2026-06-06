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
            // Informational only — emitted as plain build-script output (visible
            // with `cargo build -vv`) rather than `cargo:warning=` so it does not
            // violate the no-warnings policy.
            println!("oxirs-vec: CUDA toolkit detected - GPU acceleration will be available");
        } else {
            // CUDA not available - build will succeed with CPU fallbacks.
            // Informational only (see note above); not emitted as `cargo:warning=`.
            println!("oxirs-vec: CUDA feature enabled but toolkit not found");
            println!("oxirs-vec: GPU operations will fall back to CPU implementations");
            println!("oxirs-vec: For GPU acceleration, install CUDA toolkit from https://developer.nvidia.com/cuda-downloads");
            println!("oxirs-vec: Or build without CUDA feature (default features are Pure Rust)");
        }
    }
}
