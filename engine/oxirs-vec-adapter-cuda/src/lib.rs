//! # oxirs-vec-adapter-cuda
//!
//! Quarantined **NVIDIA CUDA** GPU-acceleration backend for [`oxirs_vec`].
//!
//! This crate isolates the [`cuda-runtime-sys`] C FFI (the raw CUDA Runtime API
//! bindings) into a `publish = false` adapter so that the published `oxirs-vec`
//! crate keeps a 100% Pure-Rust `--all-features` dependency surface, per the
//! **COOLJAPAN Pure Rust Policy v2**.
//!
//! The published `oxirs-vec` crate ships CPU-backed reference implementations of
//! its GPU types (`GpuBuffer`, `GpuDevice`, `GpuAccelerator`, ...). This adapter
//! provides the real CUDA-backed equivalents (`cudaMalloc`/`cudaMemcpy`/
//! `cudaFree`, device enumeration, streams, kernel launch) and reuses
//! [`oxirs_vec::gpu::GpuDevice`] as its public device record so the two stay
//! API-compatible.
//!
//! ## Build behavior
//!
//! The crate's [`build.rs`] probes for the CUDA toolkit (`nvcc`). When found, the
//! `cuda_runtime_available` cfg is set and the real FFI paths are compiled; when
//! absent, host-memory fallbacks are compiled so the crate's own Rust still
//! builds. Note that `cuda-runtime-sys` links `libcudart` only when a *final*
//! binary/test is linked — so `cargo build`/`cargo check` of this library
//! succeed even without a CUDA toolkit, while `cargo test` (which links a test
//! binary) requires `libcudart` to be present.
//!
//! [`cuda-runtime-sys`]: https://crates.io/crates/cuda-runtime-sys

// Raw CUDA Runtime API access via `cuda-runtime-sys` is inherently `unsafe`; this
// adapter exists precisely to confine that unsafety. The workspace lints set
// `unsafe_code = "warn"`, which we deliberately allow here.
#![allow(unsafe_code)]

pub mod buffer;
pub mod device;
pub mod kernel;
pub mod stream;

pub use buffer::CudaBuffer;
pub use device::{cuda_all_devices, cuda_device_count, cuda_device_info};
pub use kernel::{launch_kernel, CudaKernel};
pub use stream::{device_memory_usage, device_synchronize, CudaStream};

/// Returns `true` if this crate was compiled with a usable CUDA toolkit
/// (`nvcc` detected at build time) and therefore exercises the real
/// `cuda-runtime-sys` FFI paths rather than host-memory fallbacks.
pub fn cuda_build_supported() -> bool {
    cfg!(cuda_runtime_available)
}

/// Returns `true` if at least one CUDA device is currently visible to the driver.
///
/// Always `false` on builds without a CUDA toolkit.
pub fn is_cuda_available() -> bool {
    cuda_device_count().map(|n| n > 0).unwrap_or(false)
}
