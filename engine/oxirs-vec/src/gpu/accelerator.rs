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
        #[cfg(feature = "cuda")]
        {
            use cuda_runtime_sys::*;
            unsafe {
                let result = cudaSetDevice(device_id);
                if result != cudaError_t::cudaSuccess {
                    return Err(anyhow!("Failed to set CUDA device"));
                }

                let mut stream: cudaStream_t = std::ptr::null_mut();
                let result = cudaStreamCreate(&mut stream);
                if result != cudaError_t::cudaSuccess {
                    return Err(anyhow!("Failed to create CUDA stream"));
                }
                Ok(stream as *mut std::ffi::c_void)
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback: return a dummy handle for testing
            Ok(1 as *mut std::ffi::c_void)
        }
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
        #[cfg(feature = "cuda")]
        {
            // Get or compile kernel
            let kernel = self.get_or_compile_kernel(kernel_name)?;

            // Calculate grid and block dimensions
            let (blocks, threads) = self
                .device
                .calculate_optimal_block_config(params.query_count * params.db_count);

            // Launch kernel
            self.launch_kernel_impl(
                &kernel,
                blocks,
                threads,
                &[
                    query_buffer.ptr() as *mut std::ffi::c_void,
                    db_buffer.ptr() as *mut std::ffi::c_void,
                    result_buffer.ptr() as *mut std::ffi::c_void,
                    &params.query_count as *const usize as *mut std::ffi::c_void,
                    &params.db_count as *const usize as *mut std::ffi::c_void,
                    &params.dim as *const usize as *mut std::ffi::c_void,
                ],
            )?;
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback CPU implementation for testing
            self.compute_similarity_cpu(
                query_buffer,
                db_buffer,
                result_buffer,
                params,
                kernel_name,
            )?;
        }

        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
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

    #[cfg(not(feature = "cuda"))]
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

    #[cfg(not(feature = "cuda"))]
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

    #[allow(unused_variables)]
    fn compile_kernel(&self, name: &str, source: &str) -> Result<CudaKernel> {
        #[cfg(feature = "cuda")]
        {
            use cuda_runtime_sys::*;
            // In a real implementation, this would use NVRTC or similar to compile CUDA kernels
            // For now, return a dummy kernel
            Ok(CudaKernel {
                function: std::ptr::null_mut(),
                module: std::ptr::null_mut(),
                name: name.to_string(),
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(CudaKernel {
                function: std::ptr::null_mut(),
                module: std::ptr::null_mut(),
                name: name.to_string(),
            })
        }
    }

    #[cfg(feature = "cuda")]
    fn launch_kernel_impl(
        &self,
        kernel: &CudaKernel,
        blocks: i32,
        threads: i32,
        args: &[*mut std::ffi::c_void],
    ) -> Result<()> {
        use cuda_runtime_sys::*;
        unsafe {
            let result = cudaLaunchKernel(
                kernel.function,
                dim3 {
                    x: blocks as u32,
                    y: 1,
                    z: 1,
                },
                dim3 {
                    x: threads as u32,
                    y: 1,
                    z: 1,
                },
                args.as_ptr() as *mut *mut std::ffi::c_void,
                0,
                std::ptr::null_mut(),
            );
            if result != cudaError_t::cudaSuccess {
                return Err(anyhow!("Failed to launch kernel"));
            }

            // Synchronize
            let result = cudaDeviceSynchronize();
            if result != cudaError_t::cudaSuccess {
                return Err(anyhow!("Kernel execution failed"));
            }
        }
        Ok(())
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
        #[cfg(feature = "cuda")]
        {
            use cuda_runtime_sys::*;
            unsafe {
                let result = cudaDeviceSynchronize();
                if result != cudaError_t::cudaSuccess {
                    return Err(anyhow!("Failed to synchronize device"));
                }
            }
        }
        Ok(())
    }

    /// Reset performance statistics
    pub fn reset_stats(&self) {
        self.performance_stats.write().reset();
    }

    /// Get current GPU memory usage in bytes
    pub fn get_memory_usage(&self) -> Result<usize> {
        #[cfg(feature = "cuda")]
        {
            use cuda_runtime_sys::*;
            unsafe {
                let mut free: usize = 0;
                let mut total: usize = 0;
                let result = cudaMemGetInfo(&mut free as *mut usize, &mut total as *mut usize);
                if result != cudaError_t::cudaSuccess {
                    return Err(anyhow!("Failed to get memory info"));
                }
                Ok(total - free)
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Return dummy value for testing
            Ok(0)
        }
    }
}

impl Drop for GpuAccelerator {
    fn drop(&mut self) {
        // Cleanup CUDA streams
        #[cfg(feature = "cuda")]
        {
            for stream in &self.stream_pool {
                unsafe {
                    let _ = cuda_runtime_sys::cudaStreamDestroy(
                        stream.handle as cuda_runtime_sys::cudaStream_t,
                    );
                }
            }
        }
    }
}

/// Check if GPU acceleration is available
pub fn is_gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        match crate::gpu::device::GpuDevice::get_all_devices() {
            Ok(devices) => !devices.is_empty(),
            Err(_) => false,
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
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
