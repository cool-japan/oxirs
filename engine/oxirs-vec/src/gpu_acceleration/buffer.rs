//! GPU memory buffer management

use anyhow::{anyhow, Result};

/// GPU memory buffer for vector data
#[derive(Debug)]
pub struct GpuBuffer {
    ptr: *mut f32,
    size: usize,
    device_id: i32,
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl GpuBuffer {
    /// Create a new GPU buffer with specified size
    pub fn new(size: usize, device_id: i32) -> Result<Self> {
        let ptr = Self::allocate_gpu_memory(size * std::mem::size_of::<f32>(), device_id)?;
        Ok(Self {
            ptr: ptr as *mut f32,
            size,
            device_id,
        })
    }

    /// Get buffer size in elements
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Get raw pointer (unsafe)
    pub unsafe fn as_ptr(&self) -> *const f32 {
        self.ptr
    }

    /// Get mutable raw pointer (unsafe)
    pub unsafe fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr
    }

    /// Copy data from host to GPU buffer
    pub fn copy_from_host(&mut self, data: &[f32]) -> Result<()> {
        if data.len() > self.size {
            return Err(anyhow!("Data size exceeds buffer capacity"));
        }
        self.copy_host_to_device(
            data.as_ptr(),
            self.ptr,
            data.len() * std::mem::size_of::<f32>(),
        )
    }

    /// Copy data from GPU buffer to host
    pub fn copy_to_host(&self, data: &mut [f32]) -> Result<()> {
        if data.len() > self.size {
            return Err(anyhow!("Host buffer too small"));
        }
        self.copy_device_to_host(
            self.ptr,
            data.as_mut_ptr(),
            data.len() * std::mem::size_of::<f32>(),
        )
    }

    /// Copy data from another GPU buffer
    pub fn copy_from_buffer(&mut self, src: &GpuBuffer, count: usize) -> Result<()> {
        if count > self.size || count > src.size {
            return Err(anyhow!("Copy size exceeds buffer capacity"));
        }
        self.copy_device_to_device(src.ptr, self.ptr, count * std::mem::size_of::<f32>())
    }

    /// Zero out the buffer
    pub fn zero(&mut self) -> Result<()> {
        self.memset_device(self.ptr, 0, self.size * std::mem::size_of::<f32>())
    }

    fn allocate_gpu_memory(size: usize, device_id: i32) -> Result<*mut u8> {
        #[cfg(all(feature = "cuda", cuda_runtime_available))]
        {
            use cuda_runtime_sys::*;
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            unsafe {
                let result = cudaSetDevice(device_id);
                if result != cudaError_t::cudaSuccess {
                    return Err(anyhow!("Failed to set CUDA device"));
                }

                let result = cudaMalloc(&mut ptr, size);
                if result != cudaError_t::cudaSuccess {
                    return Err(anyhow!("Failed to allocate GPU memory"));
                }
            }
            Ok(ptr as *mut u8)
        }

        #[cfg(not(all(feature = "cuda", cuda_runtime_available)))]
        {
            // Fallback: allocate host memory for testing
            let layout = std::alloc::Layout::from_size_align(size, std::mem::align_of::<f32>())
                .map_err(|e| anyhow!("Invalid memory layout: {}", e))?;
            unsafe {
                let ptr = std::alloc::alloc(layout);
                if ptr.is_null() {
                    return Err(anyhow!("Failed to allocate memory"));
                }
                Ok(ptr)
            }
        }
    }

    fn copy_host_to_device(&self, src: *const f32, dst: *mut f32, size: usize) -> Result<()> {
        #[cfg(all(feature = "cuda", cuda_runtime_available))]
        {
            use cuda_runtime_sys::*;
            unsafe {
                let result = cudaMemcpy(
                    dst as *mut std::ffi::c_void,
                    src as *const std::ffi::c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                );
                if result != cudaError_t::cudaSuccess {
                    return Err(anyhow!("Failed to copy data to GPU"));
                }
            }
        }

        #[cfg(not(all(feature = "cuda", cuda_runtime_available)))]
        {
            // Fallback: simple memory copy for testing
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst, size / std::mem::size_of::<f32>());
            }
        }
        Ok(())
    }

    fn copy_device_to_host(&self, src: *const f32, dst: *mut f32, size: usize) -> Result<()> {
        #[cfg(all(feature = "cuda", cuda_runtime_available))]
        {
            use cuda_runtime_sys::*;
            unsafe {
                let result = cudaMemcpy(
                    dst as *mut std::ffi::c_void,
                    src as *const std::ffi::c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                );
                if result != cudaError_t::cudaSuccess {
                    return Err(anyhow!("Failed to copy data from GPU"));
                }
            }
        }

        #[cfg(not(all(feature = "cuda", cuda_runtime_available)))]
        {
            // Fallback: simple memory copy for testing
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst, size / std::mem::size_of::<f32>());
            }
        }
        Ok(())
    }

    fn copy_device_to_device(&self, src: *const f32, dst: *mut f32, size: usize) -> Result<()> {
        #[cfg(all(feature = "cuda", cuda_runtime_available))]
        {
            use cuda_runtime_sys::*;
            unsafe {
                let result = cudaMemcpy(
                    dst as *mut std::ffi::c_void,
                    src as *const std::ffi::c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                );
                if result != cudaError_t::cudaSuccess {
                    return Err(anyhow!("Failed to copy data between GPU buffers"));
                }
            }
        }

        #[cfg(not(all(feature = "cuda", cuda_runtime_available)))]
        {
            // Fallback: simple memory copy for testing
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst, size / std::mem::size_of::<f32>());
            }
        }
        Ok(())
    }

    fn memset_device(&self, ptr: *mut f32, value: i32, size: usize) -> Result<()> {
        #[cfg(all(feature = "cuda", cuda_runtime_available))]
        {
            use cuda_runtime_sys::*;
            unsafe {
                let result = cudaMemset(ptr as *mut std::ffi::c_void, value, size);
                if result != cudaError_t::cudaSuccess {
                    return Err(anyhow!("Failed to zero GPU memory"));
                }
            }
        }

        #[cfg(not(all(feature = "cuda", cuda_runtime_available)))]
        {
            // Fallback: zero host memory for testing
            unsafe {
                std::ptr::write_bytes(ptr as *mut u8, value as u8, size);
            }
        }
        Ok(())
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        #[cfg(all(feature = "cuda", cuda_runtime_available))]
        {
            use cuda_runtime_sys::*;
            unsafe {
                cudaFree(self.ptr as *mut std::ffi::c_void);
            }
        }

        #[cfg(not(all(feature = "cuda", cuda_runtime_available)))]
        {
            unsafe {
                let layout = std::alloc::Layout::from_size_align_unchecked(
                    self.size * std::mem::size_of::<f32>(),
                    std::mem::align_of::<f32>(),
                );
                std::alloc::dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}