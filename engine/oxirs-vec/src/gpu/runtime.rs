//! CUDA runtime types and utilities

use std::time::Duration;

/// CUDA stream wrapper
#[derive(Debug)]
pub struct CudaStream {
    pub handle: *mut std::ffi::c_void,
    pub device_id: i32,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn new(device_id: i32) -> anyhow::Result<Self> {
        #[cfg(feature = "cuda")]
        {
            use cuda_runtime_sys::*;
            let mut stream: cudaStream_t = std::ptr::null_mut();
            unsafe {
                let result = cudaSetDevice(device_id);
                if result != cudaError_t::cudaSuccess {
                    return Err(anyhow::anyhow!("Failed to set CUDA device"));
                }

                let result = cudaStreamCreate(&mut stream);
                if result != cudaError_t::cudaSuccess {
                    return Err(anyhow::anyhow!("Failed to create CUDA stream"));
                }
            }
            Ok(Self {
                handle: stream as *mut std::ffi::c_void,
                device_id,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self {
                handle: std::ptr::null_mut(),
                device_id,
            })
        }
    }

    pub fn synchronize(&self) -> anyhow::Result<()> {
        #[cfg(feature = "cuda")]
        {
            use cuda_runtime_sys::*;
            unsafe {
                let result = cudaStreamSynchronize(self.handle as cudaStream_t);
                if result != cudaError_t::cudaSuccess {
                    return Err(anyhow::anyhow!("Failed to synchronize CUDA stream"));
                }
            }
        }
        Ok(())
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            use cuda_runtime_sys::*;
            unsafe {
                cudaStreamDestroy(self.handle as cudaStream_t);
            }
        }
    }
}

/// CUDA kernel wrapper
#[derive(Debug)]
pub struct CudaKernel {
    pub function: *mut std::ffi::c_void,
    pub module: *mut std::ffi::c_void,
    pub name: String,
}

unsafe impl Send for CudaKernel {}
unsafe impl Sync for CudaKernel {}

impl CudaKernel {
    pub fn load(ptx_code: &str, function_name: &str) -> anyhow::Result<Self> {
        #[cfg(feature = "cuda")]
        {
            use cuda_driver_sys::*;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                let ptx_cstring = std::ffi::CString::new(ptx_code)?;

                let result = cuModuleLoadData(&mut module, ptx_cstring.as_ptr() as *const _);
                if result != CUresult::CUDA_SUCCESS {
                    return Err(anyhow::anyhow!("Failed to load CUDA module"));
                }

                let mut function: CUfunction = std::ptr::null_mut();
                let func_cstring = std::ffi::CString::new(function_name)?;

                let result = cuModuleGetFunction(&mut function, module, func_cstring.as_ptr());
                if result != CUresult::CUDA_SUCCESS {
                    return Err(anyhow::anyhow!("Failed to get CUDA function"));
                }

                Ok(Self {
                    function: function as *mut std::ffi::c_void,
                    module: module as *mut std::ffi::c_void,
                    name: function_name.to_string(),
                })
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            Ok(Self {
                function: std::ptr::null_mut(),
                module: std::ptr::null_mut(),
                name: function_name.to_string(),
            })
        }
    }
}

impl Drop for CudaKernel {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            use cuda_driver_sys::*;
            unsafe {
                if !self.module.is_null() {
                    cuModuleUnload(self.module as CUmodule);
                }
            }
        }
    }
}

/// GPU performance statistics
#[derive(Debug, Default, Clone)]
pub struct GpuPerformanceStats {
    pub total_operations: u64,
    pub total_compute_time: Duration,
    pub total_memory_transfers: u64,
    pub total_transfer_time: Duration,
    pub peak_memory_usage: usize,
    pub current_memory_usage: usize,
}

impl GpuPerformanceStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_operation(&mut self, compute_time: Duration) {
        self.total_operations += 1;
        self.total_compute_time += compute_time;
    }

    pub fn record_transfer(&mut self, transfer_time: Duration) {
        self.total_memory_transfers += 1;
        self.total_transfer_time += transfer_time;
    }

    pub fn update_memory_usage(&mut self, current: usize) {
        self.current_memory_usage = current;
        if current > self.peak_memory_usage {
            self.peak_memory_usage = current;
        }
    }

    pub fn average_compute_time(&self) -> Duration {
        if self.total_operations > 0 {
            self.total_compute_time / self.total_operations as u32
        } else {
            Duration::ZERO
        }
    }

    pub fn average_transfer_time(&self) -> Duration {
        if self.total_memory_transfers > 0 {
            self.total_transfer_time / self.total_memory_transfers as u32
        } else {
            Duration::ZERO
        }
    }

    pub fn throughput_ops_per_sec(&self) -> f64 {
        if !self.total_compute_time.is_zero() {
            self.total_operations as f64 / self.total_compute_time.as_secs_f64()
        } else {
            0.0
        }
    }
}
