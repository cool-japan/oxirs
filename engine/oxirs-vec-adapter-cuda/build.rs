fn main() {
    // Declare the custom cfg so the FFI gating below does not trigger
    // unexpected_cfgs warnings.
    println!("cargo:rustc-check-cfg=cfg(cuda_runtime_available)");

    // Detect a usable CUDA toolkit (`nvcc`). When present we compile the real
    // `cuda-runtime-sys` FFI paths; otherwise host-memory fallbacks are compiled
    // so the crate's own Rust still builds on machines without CUDA. (The
    // `cuda-runtime-sys` -sys crate only links `libcudart` when a final binary or
    // test is produced, so an rlib/`cargo check` build succeeds either way.)
    let cuda_available = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    if cuda_available {
        println!("cargo:rustc-cfg=cuda_runtime_available");
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
}
