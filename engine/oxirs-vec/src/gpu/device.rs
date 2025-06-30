//\! GPU device information and management

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
    /// Get information about a specific GPU device
    pub fn get_device_info(device_id: i32) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            use cuda_runtime_sys::*;
            unsafe {
                let result = cudaSetDevice(device_id);
                if result \!= cudaError_t::cudaSuccess {
                    return Err(anyhow\!("Failed to set CUDA device {}", device_id));
                }

                let mut props: cudaDeviceProp = std::mem::zeroed();
                let result = cudaGetDeviceProperties(&mut props, device_id);
                if result \!= cudaError_t::cudaSuccess {
                    return Err(anyhow\!("Failed to get device properties"));
                }

                let mut free_mem: usize = 0;
                let mut total_mem: usize = 0;
                let result = cudaMemGetInfo(&mut free_mem, &mut total_mem);
                if result \!= cudaError_t::cudaSuccess {
                    return Err(anyhow\!("Failed to get memory info"));
                }

                Ok(Self {
                    device_id,
                    name: std::ffi::CStr::from_ptr(props.name.as_ptr())
                        .to_string_lossy()
                        .to_string(),
                    compute_capability: (props.major, props.minor),
                    total_memory: total_mem,
                    free_memory: free_mem,
                    max_threads_per_block: props.maxThreadsPerBlock,
                    max_blocks_per_grid: props.maxGridSize[0],
                    warp_size: props.warpSize,
                    memory_bandwidth: props.memoryBusWidth as f32 * props.memoryClockRate as f32 * 2.0 / 8.0 / 1e6,
                    peak_flops: props.clockRate as f64 * props.multiProcessorCount as f64 * props.maxThreadsPerMultiProcessor as f64 / 1e6,
                })
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback for testing without CUDA
            Ok(Self {
                device_id,
                name: format\!("Simulated GPU {}", device_id),
                compute_capability: (7, 5), // Simulate modern GPU
                total_memory: 8 * 1024 * 1024 * 1024, // 8GB
                free_memory: 6 * 1024 * 1024 * 1024,  // 6GB free
                max_threads_per_block: 1024,
                max_blocks_per_grid: 65535,
                warp_size: 32,
                memory_bandwidth: 900.0, // GB/s
                peak_flops: 14000.0, // GFLOPS
            })
        }
    }

    /// Get information about all available GPU devices
    pub fn get_all_devices() -> Result<Vec<Self>> {
        #[cfg(feature = "cuda")]
        {
            use cuda_runtime_sys::*;
            unsafe {
                let mut device_count: i32 = 0;
                let result = cudaGetDeviceCount(&mut device_count);
                if result \!= cudaError_t::cudaSuccess {
                    return Err(anyhow\!("Failed to get device count"));
                }

                let mut devices = Vec::new();
                for i in 0..device_count {
                    if let Ok(device) = Self::get_device_info(i) {
                        devices.push(device);
                    }
                }
                Ok(devices)
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback: simulate 2 GPUs for testing
            Ok(vec\![
                Self::get_device_info(0)?,
                Self::get_device_info(1)?,
            ])
        }
    }

    /// Check if this device supports a specific compute capability
    pub fn supports_compute_capability(&self, major: i32, minor: i32) -> bool {
        self.compute_capability.0 > major || 
        (self.compute_capability.0 == major && self.compute_capability.1 >= minor)
    }

    /// Get theoretical peak memory bandwidth in GB/s
    pub fn peak_memory_bandwidth(&self) -> f32 {
        self.memory_bandwidth
    }

    /// Get theoretical peak compute performance in GFLOPS
    pub fn peak_compute_performance(&self) -> f64 {
        self.peak_flops
    }

    /// Calculate optimal thread block configuration for given problem size
    pub fn calculate_optimal_block_config(&self, problem_size: usize) -> (i32, i32) {
        let optimal_threads = (self.max_threads_per_block as f32 * 0.75) as i32; // Use 75% of max
        let blocks_needed = ((problem_size as f32) / (optimal_threads as f32)).ceil() as i32;
        let blocks = blocks_needed.min(self.max_blocks_per_grid);
        (blocks, optimal_threads)
    }
}
EOF < /dev/null
