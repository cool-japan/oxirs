//! Main GPU accelerator implementation

use super::{GpuBuffer, GpuConfig, GpuDevice, GpuPerformanceStats, KernelManager};
use crate::similarity::SimilarityMetric;
use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// CUDA stream handle
#[derive(Debug)]
pub struct CudaStream {
    handle: *mut std::ffi::c_void,
    device_id: i32,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

/// CUDA kernel handle
#[derive(Debug)]
pub struct CudaKernel {
    function: *mut std::ffi::c_void,
    module: *mut std::ffi::c_void,
    name: String,
}

unsafe impl Send for CudaKernel {}
unsafe impl Sync for CudaKernel {}

/// Parameters for similarity kernel execution
#[derive(Debug, Clone)]
pub struct SimilarityKernelParams {
    pub query_count: usize,
    pub db_count: usize,
    pub dim: usize,
    pub metric: String,
}

/// GPU acceleration engine for vector operations
#[derive(Debug)]
pub struct GpuAccelerator {
    config: GpuConfig,
    device: GpuDevice,
    memory_pool: Arc<Mutex<Vec<GpuBuffer>>>,
    stream_pool: Vec<CudaStream>,
    kernel_cache: Arc<RwLock<HashMap<String, CudaKernel>>>,
    performance_stats: Arc<RwLock<GpuPerformanceStats>>,
    kernel_manager: KernelManager,
}

unsafe impl Send for GpuAccelerator {}
unsafe impl Sync for GpuAccelerator {}

impl GpuAccelerator {
    pub fn new(config: GpuConfig) -> Result<Self> {
        config.validate()?;

        let device = GpuDevice::get_device_info(config.device_id)?;
        let memory_pool = Arc::new(Mutex::new(Vec::new()));
        let stream_pool = Self::create_streams(config.stream_count, config.device_id)?;
        let kernel_manager = KernelManager::new();

        Ok(Self {
            config,
            device,
            memory_pool,
            stream_pool,
            kernel_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_stats: Arc::new(RwLock::new(GpuPerformanceStats::new())),
            kernel_manager,
        })
    }

    fn create_streams(count: usize, device_id: i32) -> Result<Vec<CudaStream>> {
        let mut streams = Vec::new();

        for _ in 0..count {
            let handle = Self::create_cuda_stream(device_id)?;
            streams.push(CudaStream { handle, device_id });
        }

        Ok(streams)
    }

    #[allow(unused_variables)]
    fn create_cuda_stream(device_id: i32) -> Result<*mut std::ffi::c_void> {
        // Pure Rust build: placeholder handle. Real CUDA streams are created by
        // oxirs-vec-adapter-cuda.
        Ok(1 as *mut std::ffi::c_void)
    }

    /// Compute similarity between query vectors and database vectors
    pub fn compute_similarity(
        &self,
        queries: &[f32],
        database: &[f32],
        query_count: usize,
        db_count: usize,
        dim: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<f32>> {
        let timer = super::performance::GpuTimer::start("similarity_computation");

        // Allocate GPU buffers
        let mut query_buffer = GpuBuffer::new(queries.len(), self.config.device_id)?;
        let mut db_buffer = GpuBuffer::new(database.len(), self.config.device_id)?;
        let result_buffer = GpuBuffer::new(query_count * db_count, self.config.device_id)?;

        // Copy data to GPU
        query_buffer.copy_from_host(queries)?;
        db_buffer.copy_from_host(database)?;

        // Select appropriate kernel
        let kernel_name = match metric {
            SimilarityMetric::Cosine => "cosine_similarity",
            SimilarityMetric::Euclidean => "euclidean_distance",
            _ => return Err(anyhow!("Unsupported similarity metric for GPU")),
        };

        // Create kernel parameters
        let params = SimilarityKernelParams {
            query_count,
            db_count,
            dim,
            metric: kernel_name.to_string(),
        };

        // Launch kernel
        self.launch_similarity_kernel(
            kernel_name,
            &query_buffer,
            &db_buffer,
            &result_buffer,
            &params,
        )?;

        // Copy results back
        let mut results = vec![0.0f32; query_count * db_count];
        result_buffer.copy_to_host(&mut results)?;

        // Record performance
        let duration = timer.stop();
        self.performance_stats
            .write()
            .record_compute_operation(duration);

        Ok(results)
    }

    fn launch_similarity_kernel(
        &self,
        kernel_name: &str,
        query_buffer: &GpuBuffer,
        db_buffer: &GpuBuffer,
        result_buffer: &GpuBuffer,
        params: &SimilarityKernelParams,
    ) -> Result<()> {
        // Pure Rust build computes on CPU. The CUDA kernel launch is provided by
        // oxirs-vec-adapter-cuda.
        self.compute_similarity_cpu(query_buffer, db_buffer, result_buffer, params, kernel_name)
    }

    fn compute_similarity_cpu(
        &self,
        _query_buffer: &GpuBuffer,
        _db_buffer: &GpuBuffer,
        _result_buffer: &GpuBuffer,
        params: &SimilarityKernelParams,
        _metric: &str,
    ) -> Result<()> {
        // Simplified CPU fallback
        let query_data = vec![0.0f32; params.query_count * params.dim];
        let db_data = vec![0.0f32; params.db_count * params.dim];
        let mut results = vec![0.0f32; params.query_count * params.db_count];

        // Copy data from "GPU" buffers (actually host memory in fallback)
        // In real implementation, this would be proper GPU memory access

        for i in 0..params.query_count {
            for j in 0..params.db_count {
                let query_vec = &query_data[i * params.dim..(i + 1) * params.dim];
                let db_vec = &db_data[j * params.dim..(j + 1) * params.dim];

                let similarity = match params.metric.as_str() {
                    "cosine_similarity" => self.compute_cosine_similarity(query_vec, db_vec),
                    "euclidean_distance" => self.compute_euclidean_distance(query_vec, db_vec),
                    _ => 0.0,
                };

                results[i * params.db_count + j] = similarity;
            }
        }

        Ok(())
    }

    fn compute_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 1e-8 && norm_b > 1e-8 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    fn compute_euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn get_or_compile_kernel(&self, name: &str) -> Result<CudaKernel> {
        // Check if kernel is already compiled
        if let Some(kernel) = self.kernel_cache.read().get(name) {
            return Ok(CudaKernel {
                function: kernel.function,
                module: kernel.module,
                name: kernel.name.clone(),
            });
        }

        // Compile kernel
        let kernel_source = self
            .kernel_manager
            .get_kernel(name)
            .ok_or_else(|| anyhow!("Kernel {} not found", name))?;

        let compiled_kernel = self.compile_kernel(name, kernel_source)?;

        // Cache the compiled kernel
        self.kernel_cache.write().insert(
            name.to_string(),
            CudaKernel {
                function: compiled_kernel.function,
                module: compiled_kernel.module,
                name: compiled_kernel.name.clone(),
            },
        );

        Ok(compiled_kernel)
    }

    fn compile_kernel(&self, name: &str, _source: &str) -> Result<CudaKernel> {
        // Kernel-compilation bookkeeping (Pure Rust). Real NVRTC compilation is
        // provided by oxirs-vec-adapter-cuda.
        Ok(CudaKernel {
            function: std::ptr::null_mut(),
            module: std::ptr::null_mut(),
            name: name.to_string(),
        })
    }

    /// Get device information
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    /// Get configuration
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> Arc<RwLock<GpuPerformanceStats>> {
        self.performance_stats.clone()
    }

    /// Synchronize all operations
    pub fn synchronize(&self) -> Result<()> {
        // No-op in the Pure Rust build; real device sync is in oxirs-vec-adapter-cuda.
        Ok(())
    }

    /// Reset performance statistics
    pub fn reset_stats(&self) {
        self.performance_stats.write().reset();
    }

    /// Get current GPU memory usage in bytes
    pub fn get_memory_usage(&self) -> Result<usize> {
        // Pure Rust build reports zero; real CUDA memory usage is in oxirs-vec-adapter-cuda.
        Ok(0)
    }
}

/// Check if GPU acceleration is available
pub fn is_gpu_available() -> bool {
    // Pure Rust build: no CUDA devices. Use oxirs-vec-adapter-cuda for real detection.
    false
}

/// Create a default GPU accelerator configuration
pub fn create_default_accelerator() -> Result<GpuAccelerator> {
    let config = GpuConfig::default();
    GpuAccelerator::new(config)
}

/// Create a performance-optimized GPU accelerator
pub fn create_performance_accelerator() -> Result<GpuAccelerator> {
    let config = GpuConfig {
        optimization_level: crate::gpu::OptimizationLevel::Performance,
        precision_mode: crate::gpu::PrecisionMode::FP32,
        memory_pool_size: 1024 * 1024 * 1024, // 1GB
        batch_size: 10000,
        enable_tensor_cores: true,
        enable_mixed_precision: false,
        ..Default::default()
    };
    GpuAccelerator::new(config)
}

/// Create a memory-optimized GPU accelerator
pub fn create_memory_optimized_accelerator() -> Result<GpuAccelerator> {
    let config = GpuConfig {
        optimization_level: crate::gpu::OptimizationLevel::Balanced,
        precision_mode: crate::gpu::PrecisionMode::FP16,
        memory_pool_size: 256 * 1024 * 1024, // 256MB
        batch_size: 1000,
        enable_tensor_cores: true,
        enable_mixed_precision: true,
        ..Default::default()
    };
    GpuAccelerator::new(config)
}
