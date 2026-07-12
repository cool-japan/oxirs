//! CUDA device enumeration backed by `cuda-runtime-sys`.
//!
//! Results are returned as [`oxirs_vec::gpu::GpuDevice`], the same public record
//! the Pure-Rust `oxirs-vec` build produces, so callers can treat CUDA and
//! simulated devices uniformly.

use anyhow::{anyhow, Result};
use oxirs_vec::gpu::GpuDevice;

/// Number of CUDA devices visible to the driver.
///
/// Returns `Ok(0)` on builds without a CUDA toolkit.
pub fn cuda_device_count() -> Result<i32> {
    #[cfg(cuda_runtime_available)]
    {
        use cuda_runtime_sys::*;
        let mut count: i32 = 0;
        // SAFETY: `cudaGetDeviceCount` writes a single `i32` through the pointer.
        let result = unsafe { cudaGetDeviceCount(&mut count) };
        if result != cudaError_t::cudaSuccess {
            return Err(anyhow!("cudaGetDeviceCount failed: {result:?}"));
        }
        Ok(count)
    }
    #[cfg(not(cuda_runtime_available))]
    {
        Ok(0)
    }
}

/// Query a single CUDA device's properties.
///
/// Errors on builds without a CUDA toolkit.
#[allow(unused_variables)]
pub fn cuda_device_info(device_id: i32) -> Result<GpuDevice> {
    #[cfg(cuda_runtime_available)]
    {
        use cuda_runtime_sys::*;

        // SAFETY: a zeroed `cudaDeviceProp` is a valid initial POD state that
        // `cudaGetDeviceProperties` fully overwrites.
        let mut props: cudaDeviceProp = unsafe { std::mem::zeroed() };
        // SAFETY: writes device properties into `props` for a valid device id.
        let result = unsafe { cudaGetDeviceProperties(&mut props, device_id) };
        if result != cudaError_t::cudaSuccess {
            return Err(anyhow!(
                "cudaGetDeviceProperties failed for device {device_id}: {result:?}"
            ));
        }

        // SAFETY: selects the device before querying its memory info.
        let result = unsafe { cudaSetDevice(device_id) };
        if result != cudaError_t::cudaSuccess {
            return Err(anyhow!(
                "cudaSetDevice failed for device {device_id}: {result:?}"
            ));
        }

        let mut free_memory: usize = 0;
        let mut total_memory: usize = 0;
        // SAFETY: writes two `usize` values through valid pointers.
        let result = unsafe { cudaMemGetInfo(&mut free_memory, &mut total_memory) };
        if result != cudaError_t::cudaSuccess {
            return Err(anyhow!(
                "cudaMemGetInfo failed for device {device_id}: {result:?}"
            ));
        }

        // SAFETY: `props.name` is a NUL-terminated C string filled by the driver.
        let name = unsafe {
            std::ffi::CStr::from_ptr(props.name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        let cores = estimate_cuda_cores(&props);
        let clock_rate_ghz = props.clockRate as f64 / 1_000_000.0; // kHz -> GHz
        let peak_flops = cores as f64 * clock_rate_ghz * 2.0 * 1_000_000_000.0;

        Ok(GpuDevice {
            device_id,
            name,
            compute_capability: (props.major, props.minor),
            total_memory,
            free_memory,
            max_threads_per_block: props.maxThreadsPerBlock,
            max_blocks_per_grid: props.maxGridSize[0],
            warp_size: props.warpSize,
            memory_bandwidth: props.memoryBusWidth as f32 * props.memoryClockRate as f32 * 2.0
                / 8.0
                / 1_000_000.0, // GB/s
            peak_flops,
        })
    }
    #[cfg(not(cuda_runtime_available))]
    {
        Err(anyhow!(
            "oxirs-vec-adapter-cuda was built without a CUDA toolkit (nvcc not found); \
             real CUDA device info is unavailable on this build"
        ))
    }
}

/// Enumerate all visible CUDA devices.
///
/// Returns an empty vector on builds without a CUDA toolkit.
pub fn cuda_all_devices() -> Result<Vec<GpuDevice>> {
    let count = cuda_device_count()?;
    let mut devices = Vec::new();
    for device_id in 0..count {
        if let Ok(device) = cuda_device_info(device_id) {
            devices.push(device);
        }
    }
    Ok(devices)
}

/// Estimate the number of CUDA cores from device properties (compute-capability
/// based heuristic).
#[cfg(cuda_runtime_available)]
fn estimate_cuda_cores(props: &cuda_runtime_sys::cudaDeviceProp) -> i32 {
    let sm_count = props.multiProcessorCount;
    match (props.major, props.minor) {
        (7, 5) => sm_count * 64,  // Turing
        (7, 0) => sm_count * 64,  // Volta
        (6, 1) => sm_count * 128, // Pascal
        (6, 0) => sm_count * 64,  // Pascal
        (5, 2) => sm_count * 128, // Maxwell
        (5, 0) => sm_count * 128, // Maxwell
        (3, 7) => sm_count * 192, // Kepler
        (3, 5) => sm_count * 192, // Kepler
        _ => sm_count * 64,       // Default estimate
    }
}
