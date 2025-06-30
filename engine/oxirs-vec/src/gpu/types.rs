//! Common GPU types and structures

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

/// GPU execution configuration
#[derive(Debug, Clone)]
pub struct GpuExecutionConfig {
    pub block_size: u32,
    pub grid_size: u32,
    pub shared_memory_size: u32,
    pub stream_id: Option<u32>,
}

impl Default for GpuExecutionConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            grid_size: 256,
            shared_memory_size: 0,
            stream_id: None,
        }
    }
}