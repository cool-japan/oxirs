//! CUDA streams and device-wide synchronization backed by `cuda-runtime-sys`.

use anyhow::{anyhow, Result};

/// A CUDA stream handle.
#[derive(Debug)]
pub struct CudaStream {
    handle: *mut std::ffi::c_void,
    device_id: i32,
}

// SAFETY: a CUDA stream handle is a process-global driver resource; passing the
// owning wrapper between threads is sound.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Create a new CUDA stream on `device_id`.
    ///
    /// Errors on builds without a CUDA toolkit.
    #[allow(unused_variables)]
    pub fn new(device_id: i32) -> Result<Self> {
        #[cfg(cuda_runtime_available)]
        {
            use cuda_runtime_sys::*;
            // SAFETY: selects the device, then creates a stream written via `&mut`.
            let result = unsafe { cudaSetDevice(device_id) };
            if result != cudaError_t::cudaSuccess {
                return Err(anyhow!("cudaSetDevice failed: {result:?}"));
            }
            let mut stream: cudaStream_t = std::ptr::null_mut();
            let result = unsafe { cudaStreamCreate(&mut stream) };
            if result != cudaError_t::cudaSuccess {
                return Err(anyhow!("cudaStreamCreate failed: {result:?}"));
            }
            Ok(Self {
                handle: stream as *mut std::ffi::c_void,
                device_id,
            })
        }
        #[cfg(not(cuda_runtime_available))]
        {
            Err(anyhow!(
                "oxirs-vec-adapter-cuda was built without a CUDA toolkit (nvcc not found); \
                 CUDA streams are unavailable on this build"
            ))
        }
    }

    /// Raw stream handle.
    pub fn handle(&self) -> *mut std::ffi::c_void {
        self.handle
    }

    /// Device id this stream belongs to.
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Block until all work queued on this stream has completed.
    pub fn synchronize(&self) -> Result<()> {
        #[cfg(cuda_runtime_available)]
        {
            use cuda_runtime_sys::*;
            // SAFETY: synchronizes a stream created by `cudaStreamCreate`.
            let result = unsafe { cudaStreamSynchronize(self.handle as cudaStream_t) };
            if result != cudaError_t::cudaSuccess {
                return Err(anyhow!("cudaStreamSynchronize failed: {result:?}"));
            }
            Ok(())
        }
        #[cfg(not(cuda_runtime_available))]
        {
            Ok(())
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        #[cfg(cuda_runtime_available)]
        {
            if !self.handle.is_null() {
                use cuda_runtime_sys::*;
                // SAFETY: destroys a stream created by `cudaStreamCreate`.
                unsafe {
                    let _ = cudaStreamDestroy(self.handle as cudaStream_t);
                }
            }
        }
    }
}

/// Block until all work on the current device has completed.
pub fn device_synchronize() -> Result<()> {
    #[cfg(cuda_runtime_available)]
    {
        use cuda_runtime_sys::*;
        // SAFETY: device-wide synchronization.
        let result = unsafe { cudaDeviceSynchronize() };
        if result != cudaError_t::cudaSuccess {
            return Err(anyhow!("cudaDeviceSynchronize failed: {result:?}"));
        }
        Ok(())
    }
    #[cfg(not(cuda_runtime_available))]
    {
        Ok(())
    }
}

/// Current device memory usage (`total - free`) in bytes.
///
/// Returns `Ok(0)` on builds without a CUDA toolkit.
pub fn device_memory_usage() -> Result<usize> {
    #[cfg(cuda_runtime_available)]
    {
        use cuda_runtime_sys::*;
        let mut free: usize = 0;
        let mut total: usize = 0;
        // SAFETY: writes two `usize` values through valid pointers.
        let result = unsafe { cudaMemGetInfo(&mut free, &mut total) };
        if result != cudaError_t::cudaSuccess {
            return Err(anyhow!("cudaMemGetInfo failed: {result:?}"));
        }
        Ok(total.saturating_sub(free))
    }
    #[cfg(not(cuda_runtime_available))]
    {
        Ok(0)
    }
}
