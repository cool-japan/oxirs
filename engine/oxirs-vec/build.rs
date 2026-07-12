fn main() {
    // Link against libnuma on Linux
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=numa");
    }

    // NOTE: CUDA detection/linking has been removed from oxirs-vec. Real NVIDIA
    // CUDA acceleration now lives in the quarantined `oxirs-vec-adapter-cuda`
    // crate (publish = false), keeping oxirs-vec's published --all-features
    // surface free of `cuda-runtime-sys` per the COOLJAPAN Pure Rust Policy v2.
}
