//! CUDA device-memory buffer backed by `cuda-runtime-sys`.

use anyhow::{anyhow, Result};

/// A GPU memory buffer for `f32` vector data.
///
/// When the crate is built with a CUDA toolkit this wraps real device memory
/// (`cudaMalloc`/`cudaMemcpy`/`cudaFree`); otherwise it transparently falls back
/// to host memory so the type stays usable (and testable) without CUDA hardware.
#[derive(Debug)]
pub struct CudaBuffer {
    ptr: *mut f32,
    size: usize,
    device_id: i32,
}

// SAFETY: the buffer owns its allocation exclusively; raw-pointer access is
// internal. Sending/sharing the handle across threads is sound for the CUDA
// runtime (device pointers are process-global) and for the host fallback.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    /// Allocate a buffer holding `size` `f32` elements on `device_id`.
    pub fn new(size: usize, device_id: i32) -> Result<Self> {
        let ptr = Self::allocate(size * std::mem::size_of::<f32>(), device_id)?;
        Ok(Self {
            ptr: ptr as *mut f32,
            size,
            device_id,
        })
    }

    /// Number of `f32` elements the buffer can hold.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Device id the buffer is associated with.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Raw device pointer.
    pub fn ptr(&self) -> *mut f32 {
        self.ptr
    }

    /// Whether the underlying pointer is non-null.
    pub fn is_valid(&self) -> bool {
        !self.ptr.is_null()
    }

    /// Copy host data into the buffer.
    pub fn copy_from_host(&mut self, data: &[f32]) -> Result<()> {
        if data.len() > self.size {
            return Err(anyhow!("Data size exceeds buffer capacity"));
        }
        self.copy_host_to_device(data.as_ptr(), self.ptr, std::mem::size_of_val(data))
    }

    /// Copy buffer contents back to host.
    pub fn copy_to_host(&self, data: &mut [f32]) -> Result<()> {
        if data.len() > self.size {
            return Err(anyhow!("Host buffer too small"));
        }
        self.copy_device_to_host(self.ptr, data.as_mut_ptr(), std::mem::size_of_val(data))
    }

    /// Copy `count` `f32` elements from another buffer into this one.
    pub fn copy_from_buffer(&mut self, src: &CudaBuffer, count: usize) -> Result<()> {
        if count > self.size || count > src.size {
            return Err(anyhow!("Copy size exceeds buffer capacity"));
        }
        self.copy_device_to_device(src.ptr, self.ptr, count * std::mem::size_of::<f32>())
    }

    /// Zero the entire buffer.
    pub fn zero(&mut self) -> Result<()> {
        self.memset(self.ptr, 0, self.size * std::mem::size_of::<f32>())
    }

    #[allow(unused_variables)]
    fn allocate(size: usize, device_id: i32) -> Result<*mut u8> {
        #[cfg(cuda_runtime_available)]
        {
            use cuda_runtime_sys::*;
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            // SAFETY: selects the device, then allocates `size` bytes; `ptr` is
            // written by `cudaMalloc`.
            let result = unsafe { cudaSetDevice(device_id) };
            if result != cudaError_t::cudaSuccess {
                return Err(anyhow!("cudaSetDevice failed: {result:?}"));
            }
            let result = unsafe { cudaMalloc(&mut ptr, size) };
            if result != cudaError_t::cudaSuccess {
                return Err(anyhow!("cudaMalloc failed: {result:?}"));
            }
            Ok(ptr as *mut u8)
        }
        #[cfg(not(cuda_runtime_available))]
        {
            let layout = std::alloc::Layout::from_size_align(size, std::mem::align_of::<f32>())
                .map_err(|e| anyhow!("Invalid memory layout: {e}"))?;
            // SAFETY: `layout` has non-zero size for any non-empty buffer; the
            // null return is checked below.
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(anyhow!("Failed to allocate host memory"));
            }
            Ok(ptr)
        }
    }

    fn copy_host_to_device(&self, src: *const f32, dst: *mut f32, size: usize) -> Result<()> {
        #[cfg(cuda_runtime_available)]
        {
            use cuda_runtime_sys::*;
            // SAFETY: copies `size` bytes from a host pointer into a device pointer.
            let result = unsafe {
                cudaMemcpy(
                    dst as *mut std::ffi::c_void,
                    src as *const std::ffi::c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            };
            if result != cudaError_t::cudaSuccess {
                return Err(anyhow!("cudaMemcpy H2D failed: {result:?}"));
            }
            Ok(())
        }
        #[cfg(not(cuda_runtime_available))]
        {
            // SAFETY: host-to-host copy of `size` bytes (`size / 4` `f32` elements)
            // between non-overlapping allocations.
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst, size / std::mem::size_of::<f32>());
            }
            Ok(())
        }
    }

    fn copy_device_to_host(&self, src: *const f32, dst: *mut f32, size: usize) -> Result<()> {
        #[cfg(cuda_runtime_available)]
        {
            use cuda_runtime_sys::*;
            // SAFETY: copies `size` bytes from a device pointer into a host pointer.
            let result = unsafe {
                cudaMemcpy(
                    dst as *mut std::ffi::c_void,
                    src as *const std::ffi::c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            if result != cudaError_t::cudaSuccess {
                return Err(anyhow!("cudaMemcpy D2H failed: {result:?}"));
            }
            Ok(())
        }
        #[cfg(not(cuda_runtime_available))]
        {
            // SAFETY: host-to-host copy of `size` bytes between non-overlapping
            // allocations.
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst, size / std::mem::size_of::<f32>());
            }
            Ok(())
        }
    }

    fn copy_device_to_device(&self, src: *const f32, dst: *mut f32, size: usize) -> Result<()> {
        #[cfg(cuda_runtime_available)]
        {
            use cuda_runtime_sys::*;
            // SAFETY: copies `size` bytes between two device allocations.
            let result = unsafe {
                cudaMemcpy(
                    dst as *mut std::ffi::c_void,
                    src as *const std::ffi::c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                )
            };
            if result != cudaError_t::cudaSuccess {
                return Err(anyhow!("cudaMemcpy D2D failed: {result:?}"));
            }
            Ok(())
        }
        #[cfg(not(cuda_runtime_available))]
        {
            // SAFETY: host-to-host copy of `size` bytes between non-overlapping
            // allocations.
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst, size / std::mem::size_of::<f32>());
            }
            Ok(())
        }
    }

    fn memset(&self, ptr: *mut f32, value: i32, size: usize) -> Result<()> {
        #[cfg(cuda_runtime_available)]
        {
            use cuda_runtime_sys::*;
            // SAFETY: sets `size` bytes of the device allocation to `value`.
            let result = unsafe { cudaMemset(ptr as *mut std::ffi::c_void, value, size) };
            if result != cudaError_t::cudaSuccess {
                return Err(anyhow!("cudaMemset failed: {result:?}"));
            }
            Ok(())
        }
        #[cfg(not(cuda_runtime_available))]
        {
            // SAFETY: writes `size` bytes of the host allocation.
            unsafe {
                std::ptr::write_bytes(ptr as *mut u8, value as u8, size);
            }
            Ok(())
        }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        #[cfg(cuda_runtime_available)]
        {
            // SAFETY: frees a pointer obtained from `cudaMalloc`.
            unsafe {
                let _ = cuda_runtime_sys::cudaFree(self.ptr as *mut std::ffi::c_void);
            }
        }
        #[cfg(not(cuda_runtime_available))]
        {
            if let Ok(layout) = std::alloc::Layout::from_size_align(
                self.size * std::mem::size_of::<f32>(),
                std::mem::align_of::<f32>(),
            ) {
                // SAFETY: deallocates host memory allocated with the same layout.
                unsafe {
                    std::alloc::dealloc(self.ptr as *mut u8, layout);
                }
            }
        }
    }
}

#[cfg(all(test, not(cuda_runtime_available)))]
mod tests {
    use super::*;

    #[test]
    fn host_fallback_roundtrip() {
        // Without a CUDA toolkit the buffer is host-backed; verify the copy and
        // zero paths behave correctly.
        let mut buf = CudaBuffer::new(4, 0).expect("allocate host-backed buffer");
        assert!(buf.is_valid());
        assert_eq!(buf.size(), 4);

        buf.copy_from_host(&[1.0, 2.0, 3.0, 4.0])
            .expect("copy_from_host");

        let mut out = [0.0f32; 4];
        buf.copy_to_host(&mut out).expect("copy_to_host");
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);

        buf.zero().expect("zero");
        buf.copy_to_host(&mut out).expect("copy_to_host after zero");
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn rejects_oversized_copy() {
        let mut buf = CudaBuffer::new(2, 0).expect("allocate");
        assert!(buf.copy_from_host(&[1.0, 2.0, 3.0]).is_err());
    }
}
