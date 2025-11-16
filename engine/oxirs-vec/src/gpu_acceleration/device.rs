//! GPU device management and information

use anyhow::{anyhow, Result};

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub device_id: i32,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory: usize,
    pub free_memory: usize,
    pub max_threads_per_block: i32,
    pub max_blocks_per_grid: i32,
    pub warp_size: i32,
    pub memory_bandwidth: f32,
    pub peak_flops: f64,
}

impl GpuDevice {
    /// Get device performance score for selection
    pub fn performance_score(&self) -> f64 {
        let memory_score = self.free_memory as f64 / (1024.0 * 1024.0 * 1024.0); // GB
        let compute_score = self.peak_flops / 1e12; // TFLOPS
        let bandwidth_score = self.memory_bandwidth as f64 / 1000.0; // TB/s

        // Weighted combination of different performance factors
        memory_score * 0.3 + compute_score * 0.5 + bandwidth_score * 0.2
    }

    /// Check if device supports required compute capability
    pub fn supports_compute_capability(&self, required: (i32, i32)) -> bool {
        self.compute_capability.0 > required.0 ||
        (self.compute_capability.0 == required.0 && self.compute_capability.1 >= required.1)
    }
}

/// Query available GPU devices
pub fn query_gpu_devices() -> Result<Vec<GpuDevice>> {
    #[cfg(all(feature = "cuda", cuda_runtime_available))]
    {
        get_cuda_devices()
    }

    #[cfg(not(all(feature = "cuda", cuda_runtime_available)))]
    {
        // Return empty list if CUDA is not available
        Ok(Vec::new())
    }
}

/// Get the best GPU device based on performance and memory
pub fn get_best_gpu_device() -> Result<GpuDevice> {
    let devices = query_gpu_devices()?;

    if devices.is_empty() {
        return Err(anyhow!("No GPU devices available"));
    }

    // Find device with highest performance score
    let best_device = devices
        .into_iter()
        .max_by(|a, b| a.performance_score().partial_cmp(&b.performance_score()).unwrap())
        .unwrap();

    Ok(best_device)
}

/// Get GPU device by ID
pub fn get_gpu_device(device_id: i32) -> Result<GpuDevice> {
    let devices = query_gpu_devices()?;
    devices
        .into_iter()
        .find(|d| d.device_id == device_id)
        .ok_or_else(|| anyhow!("GPU device {} not found", device_id))
}

#[cfg(all(feature = "cuda", cuda_runtime_available))]
fn get_cuda_devices() -> Result<Vec<GpuDevice>> {
    use cuda_runtime_sys::*;

    let mut device_count: i32 = 0;
    unsafe {
        let result = cudaGetDeviceCount(&mut device_count);
        if result != cudaError_t::cudaSuccess {
            return Err(anyhow!("Failed to get CUDA device count"));
        }
    }

    let mut devices = Vec::new();
    for device_id in 0..device_count {
        if let Ok(device) = get_cuda_device_info(device_id) {
            devices.push(device);
        }
    }

    Ok(devices)
}

#[cfg(all(feature = "cuda", cuda_runtime_available))]
fn get_cuda_device_info(device_id: i32) -> Result<GpuDevice> {
    use cuda_runtime_sys::*;

    let mut props: cudaDeviceProp = unsafe { std::mem::zeroed() };
    unsafe {
        let result = cudaGetDeviceProperties(&mut props, device_id);
        if result != cudaError_t::cudaSuccess {
            return Err(anyhow!("Failed to get device properties for device {}", device_id));
        }
    }

    // Get memory info
    let mut free_memory: usize = 0;
    let mut total_memory: usize = 0;
    unsafe {
        let result = cudaSetDevice(device_id);
        if result != cudaError_t::cudaSuccess {
            return Err(anyhow!("Failed to set device {}", device_id));
        }

        let result = cudaMemGetInfo(&mut free_memory, &mut total_memory);
        if result != cudaError_t::cudaSuccess {
            return Err(anyhow!("Failed to get memory info for device {}", device_id));
        }
    }

    // Convert device name from C string
    let name = unsafe {
        std::ffi::CStr::from_ptr(props.name.as_ptr())
            .to_string_lossy()
            .into_owned()
    };

    // Calculate peak FLOPS (simplified estimation)
    let cores = estimate_cuda_cores(&props);
    let clock_rate_ghz = props.clockRate as f64 / 1_000_000.0; // Convert from kHz to GHz
    let peak_flops = cores as f64 * clock_rate_ghz * 2.0 * 1_000_000_000.0; // 2 operations per clock

    Ok(GpuDevice {
        device_id,
        name,
        compute_capability: (props.major, props.minor),
        total_memory,
        free_memory,
        max_threads_per_block: props.maxThreadsPerBlock,
        max_blocks_per_grid: props.maxGridSize[0],
        warp_size: props.warpSize,
        memory_bandwidth: props.memoryBusWidth as f32 * props.memoryClockRate as f32 * 2.0 / 8.0 / 1_000_000.0, // GB/s
        peak_flops,
    })
}

#[cfg(all(feature = "cuda", cuda_runtime_available))]
fn estimate_cuda_cores(props: &cuda_runtime_sys::cudaDeviceProp) -> i32 {
    // Simplified CUDA core estimation based on compute capability
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