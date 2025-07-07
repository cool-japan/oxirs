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
    pub fn new(size: usize, device_id: i32) -> Result<Self> {
        let ptr = Self::allocate_gpu_memory(size * std::mem::size_of::<f32>(), device_id)?;
        Ok(Self {
            ptr: ptr as *mut f32,
            size,
            device_id,
        })
    }

    pub fn copy_from_host(&mut self, data: &[f32]) -> Result<()> {
        if data.len() > self.size {
            return Err(anyhow!("Data size exceeds buffer capacity"));
        }
        self.copy_host_to_device(
            data.as_ptr(),
            self.ptr,
            std::mem::size_of_val(data),
        )
    }

    pub fn copy_to_host(&self, data: &mut [f32]) -> Result<()> {
        if data.len() > self.size {
            return Err(anyhow!("Host buffer too small"));
        }
        self.copy_device_to_host(
            self.ptr,
            data.as_mut_ptr(),
            std::mem::size_of_val(data),
        )
    }

    #[allow(unused_variables)]
    fn allocate_gpu_memory(size: usize, device_id: i32) -> Result<*mut u8> {
        // Simulate GPU memory allocation
        // In a real implementation, this would use CUDA runtime API
        #[cfg(feature = "cuda")]
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

        #[cfg(not(feature = "cuda"))]
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
        #[cfg(feature = "cuda")]
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
                    return Err(anyhow!("Failed to copy data to device"));
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback: simple memory copy for testing
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst, size / std::mem::size_of::<f32>());
            }
        }
        Ok(())
    }

    fn copy_device_to_host(&self, src: *const f32, dst: *mut f32, size: usize) -> Result<()> {
        #[cfg(feature = "cuda")]
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
                    return Err(anyhow!("Failed to copy data from device"));
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Fallback: simple memory copy for testing
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst, size / std::mem::size_of::<f32>());
            }
        }
        Ok(())
    }

    pub fn ptr(&self) -> *mut f32 {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn is_valid(&self) -> bool {
        !self.ptr.is_null()
    }

    /// Zero out the buffer
    pub fn zero(&mut self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            use cuda_runtime_sys::*;
            unsafe {
                let result = cudaMemset(
                    self.ptr as *mut std::ffi::c_void,
                    0,
                    self.size * std::mem::size_of::<f32>(),
                );
                if result != cudaError_t::cudaSuccess {
                    return Err(anyhow!("Failed to zero buffer"));
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            unsafe {
                std::ptr::write_bytes(self.ptr, 0, self.size);
            }
        }
        Ok(())
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            #[cfg(feature = "cuda")]
            {
                use cuda_runtime_sys::*;
                unsafe {
                    let _ = cudaFree(self.ptr as *mut std::ffi::c_void);
                }
            }

            #[cfg(not(feature = "cuda"))]
            {
                // Fallback: deallocate host memory
                let layout = std::alloc::Layout::from_size_align(
                    self.size * std::mem::size_of::<f32>(),
                    std::mem::align_of::<f32>(),
                );
                if let Ok(layout) = layout {
                    unsafe {
                        std::alloc::dealloc(self.ptr as *mut u8, layout);
                    }
                }
            }
        }
    }
}
