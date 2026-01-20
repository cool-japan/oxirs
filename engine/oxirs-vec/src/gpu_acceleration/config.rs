//! GPU acceleration configuration

/// GPU acceleration configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub device_id: i32,
    pub enable_mixed_precision: bool,
    pub enable_tensor_cores: bool,
    pub batch_size: usize,
    pub memory_pool_size: usize,
    pub stream_count: usize,
    pub enable_peer_access: bool,
    pub enable_unified_memory: bool,
    pub enable_async_execution: bool,
    pub enable_multi_gpu: bool,
    pub preferred_gpu_ids: Vec<i32>,
    pub dynamic_batch_sizing: bool,
    pub enable_memory_compression: bool,
    pub kernel_cache_size: usize,
    pub optimization_level: GpuOptimization,
    pub precision_mode: GpuPrecision,
}

/// GPU optimization levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuOptimization {
    Debug,       // Maximum debugging, minimal optimization
    Balanced,    // Good balance of performance and debugging
    Performance, // Maximum performance, minimal debugging
    Extreme,     // Aggressive optimizations, may reduce precision
}

/// Precision modes for GPU computations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuPrecision {
    FP32,     // Single precision
    FP16,     // Half precision
    Mixed,    // Mixed precision (FP16 for compute, FP32 for storage)
    INT8,     // 8-bit integer quantization
    Adaptive, // Adaptive precision based on data characteristics
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            enable_mixed_precision: true,
            enable_tensor_cores: true,
            batch_size: 1024,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            stream_count: 4,
            enable_peer_access: false,
            enable_unified_memory: false,
            enable_async_execution: true,
            enable_multi_gpu: false,
            preferred_gpu_ids: vec![0],
            dynamic_batch_sizing: true,
            enable_memory_compression: false,
            kernel_cache_size: 100, // Cache up to 100 compiled kernels
            optimization_level: GpuOptimization::Balanced,
            precision_mode: GpuPrecision::FP32,
        }
    }
}

impl GpuConfig {
    /// Create a performance-optimized configuration
    pub fn performance() -> Self {
        Self {
            optimization_level: GpuOptimization::Performance,
            enable_mixed_precision: true,
            enable_tensor_cores: true,
            batch_size: 2048,
            enable_async_execution: true,
            dynamic_batch_sizing: true,
            ..Default::default()
        }
    }

    /// Create a debug-friendly configuration
    pub fn debug() -> Self {
        Self {
            optimization_level: GpuOptimization::Debug,
            enable_mixed_precision: false,
            enable_tensor_cores: false,
            batch_size: 128,
            enable_async_execution: false,
            dynamic_batch_sizing: false,
            ..Default::default()
        }
    }

    /// Create a memory-efficient configuration
    pub fn memory_efficient() -> Self {
        Self {
            enable_memory_compression: true,
            memory_pool_size: 512 * 1024 * 1024, // 512MB
            precision_mode: GpuPrecision::FP16,
            batch_size: 512,
            ..Default::default()
        }
    }

    /// Validate configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.batch_size == 0 {
            return Err("Batch size must be greater than 0".to_string());
        }

        if self.memory_pool_size < 1024 * 1024 {
            return Err("Memory pool size must be at least 1MB".to_string());
        }

        if self.stream_count == 0 {
            return Err("Stream count must be greater than 0".to_string());
        }

        if self.kernel_cache_size == 0 {
            return Err("Kernel cache size must be greater than 0".to_string());
        }

        if self.preferred_gpu_ids.is_empty() {
            return Err("At least one preferred GPU ID must be specified".to_string());
        }

        Ok(())
    }
}

impl GpuOptimization {
    /// Get the optimization level as a numeric value
    pub fn level(&self) -> u32 {
        match self {
            GpuOptimization::Debug => 0,
            GpuOptimization::Balanced => 1,
            GpuOptimization::Performance => 2,
            GpuOptimization::Extreme => 3,
        }
    }

    /// Check if debugging is enabled for this optimization level
    pub fn debug_enabled(&self) -> bool {
        matches!(self, GpuOptimization::Debug | GpuOptimization::Balanced)
    }
}

impl GpuPrecision {
    /// Get the bytes per element for this precision mode
    pub fn bytes_per_element(&self) -> usize {
        match self {
            GpuPrecision::FP32 => 4,
            GpuPrecision::FP16 => 2,
            GpuPrecision::Mixed => 4, // Storage precision
            GpuPrecision::INT8 => 1,
            GpuPrecision::Adaptive => 4, // Default to FP32
        }
    }

    /// Check if this precision mode supports tensor cores
    pub fn supports_tensor_cores(&self) -> bool {
        matches!(self, GpuPrecision::FP16 | GpuPrecision::Mixed)
    }
}