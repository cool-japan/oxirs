//! CUDA kernel handle and launch backed by `cuda-runtime-sys`.

use anyhow::{anyhow, Result};

/// A handle to a resolved CUDA kernel function.
#[derive(Debug)]
pub struct CudaKernel {
    function: *mut std::ffi::c_void,
    module: *mut std::ffi::c_void,
    name: String,
}

// SAFETY: the kernel handle wraps process-global driver resources.
unsafe impl Send for CudaKernel {}
unsafe impl Sync for CudaKernel {}

impl CudaKernel {
    /// Build a kernel handle from already-resolved function/module pointers.
    pub fn from_raw(
        function: *mut std::ffi::c_void,
        module: *mut std::ffi::c_void,
        name: impl Into<String>,
    ) -> Self {
        Self {
            function,
            module,
            name: name.into(),
        }
    }

    /// Kernel name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Raw function pointer.
    pub fn function(&self) -> *mut std::ffi::c_void {
        self.function
    }

    /// Raw module pointer.
    pub fn module(&self) -> *mut std::ffi::c_void {
        self.module
    }
}

/// Launch `kernel` with the given grid/block dimensions and argument pointers,
/// then synchronize the device.
///
/// Errors on builds without a CUDA toolkit.
#[allow(unused_variables)]
pub fn launch_kernel(
    kernel: &CudaKernel,
    blocks: i32,
    threads: i32,
    args: &[*mut std::ffi::c_void],
) -> Result<()> {
    #[cfg(cuda_runtime_available)]
    {
        use cuda_runtime_sys::*;
        // SAFETY: launches `kernel.function` with the provided 1-D launch
        // configuration and argument array.
        let result = unsafe {
            cudaLaunchKernel(
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
            )
        };
        if result != cudaError_t::cudaSuccess {
            return Err(anyhow!("cudaLaunchKernel failed: {result:?}"));
        }
        // SAFETY: device-wide synchronization after launch.
        let result = unsafe { cudaDeviceSynchronize() };
        if result != cudaError_t::cudaSuccess {
            return Err(anyhow!("kernel execution failed: {result:?}"));
        }
        Ok(())
    }
    #[cfg(not(cuda_runtime_available))]
    {
        Err(anyhow!(
            "oxirs-vec-adapter-cuda was built without a CUDA toolkit (nvcc not found); \
             kernel launch is unavailable on this build"
        ))
    }
}
