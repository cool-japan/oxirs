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
        } else {
            println!("cargo:warning=CUDA feature enabled but CUDA toolkit not found - CUDA-dependent tests/examples will be skipped");
            println!("cargo:warning=Compilation will likely fail due to missing CUDA libraries");
            println!(
                "cargo:warning=To fix: either install CUDA toolkit or build without --all-features"
            );
            // cuda-runtime-sys will fail to link - this is expected behavior
            // Users should not enable cuda feature without CUDA toolkit
        }
    }
}
