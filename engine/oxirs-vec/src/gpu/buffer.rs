//! GPU memory buffer management
//!
//! The published `oxirs-vec` build is 100% Pure Rust: this buffer is backed by
//! host memory. Real CUDA device memory (`cudaMalloc`/`cudaMemcpy`/`cudaFree`)
//! lives in the quarantined `oxirs-vec-adapter-cuda` crate (publish = false) per
//! the COOLJAPAN Pure Rust Policy v2.

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
        self.copy_host_to_device(data.as_ptr(), self.ptr, std::mem::size_of_val(data))
    }

    pub fn copy_to_host(&self, data: &mut [f32]) -> Result<()> {
        if data.len() > self.size {
            return Err(anyhow!("Host buffer too small"));
        }
        self.copy_device_to_host(self.ptr, data.as_mut_ptr(), std::mem::size_of_val(data))
    }

    #[allow(unused_variables)]
    fn allocate_gpu_memory(size: usize, device_id: i32) -> Result<*mut u8> {
        // Host-memory allocation (Pure Rust). CUDA-backed allocation is provided
        // by oxirs-vec-adapter-cuda.
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

    fn copy_host_to_device(&self, src: *const f32, dst: *mut f32, size: usize) -> Result<()> {
        // Pure Rust build: host-to-host copy.
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size / std::mem::size_of::<f32>());
        }
        Ok(())
    }

    fn copy_device_to_host(&self, src: *const f32, dst: *mut f32, size: usize) -> Result<()> {
        // Pure Rust build: host-to-host copy.
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size / std::mem::size_of::<f32>());
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
        unsafe {
            std::ptr::write_bytes(self.ptr, 0, self.size);
        }
        Ok(())
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
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
