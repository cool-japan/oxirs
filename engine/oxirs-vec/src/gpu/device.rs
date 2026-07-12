//! GPU device information and management
//!
//! The published `oxirs-vec` build reports a simulated device (Pure Rust). Real
//! CUDA device enumeration (`cudaGetDeviceProperties`, `cudaMemGetInfo`, ...) lives
//! in the quarantined `oxirs-vec-adapter-cuda` crate (publish = false) per the
//! COOLJAPAN Pure Rust Policy v2.

use anyhow::Result;

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
    /// Create a simulated GPU device for testing or when no CUDA device is available
    fn simulated(device_id: i32) -> Self {
        Self {
            device_id,
            name: format!("Simulated GPU {device_id}"),
            compute_capability: (7, 5),
            total_memory: 8 * 1024 * 1024 * 1024,
            free_memory: 6 * 1024 * 1024 * 1024,
            max_threads_per_block: 1024,
            max_blocks_per_grid: 65535,
            warp_size: 32,
            memory_bandwidth: 900.0,
            peak_flops: 14000.0,
        }
    }

    /// Get information about a specific GPU device
    pub fn get_device_info(device_id: i32) -> Result<Self> {
        // Pure Rust build: report a simulated device. CUDA-backed device
        // enumeration is provided by oxirs-vec-adapter-cuda.
        tracing::warn!("CUDA not available - using simulated GPU device");
        Ok(Self::simulated(device_id))
    }

    /// Get information about all available GPU devices
    pub fn get_all_devices() -> Result<Vec<Self>> {
        // Pure Rust build: simulate two GPUs. CUDA-backed enumeration is provided
        // by oxirs-vec-adapter-cuda.
        tracing::warn!("CUDA not available - using simulated GPU devices");
        Ok(vec![Self::get_device_info(0)?, Self::get_device_info(1)?])
    }

    /// Check if this device supports a specific compute capability
    pub fn supports_compute_capability(&self, major: i32, minor: i32) -> bool {
        self.compute_capability.0 > major
            || (self.compute_capability.0 == major && self.compute_capability.1 >= minor)
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
