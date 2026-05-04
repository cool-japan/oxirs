//! GPU-accelerated kernels for physics simulations.
//!
//! When compiled with the `gpu` feature, this module exposes dispatchers that
//! attempt to offload finite-element stress assembly, the Navier-Stokes
//! pressure-Poisson solve, and heat-diffusion stencil updates onto a GPU
//! compute backend supplied by `scirs2_core`. When the `gpu` feature is
//! disabled (the default), every dispatcher returns
//! [`GpuError::BackendUnavailable`] immediately so callers can fall back to
//! the existing CPU paths without conditional compilation at the call site.
//!
//! This mirrors the SAMM W3-S12 GPU pattern: feature-gated, default off,
//! pure-Rust default surface, opt-in C/Fortran via the `gpu` feature.
//!
//! # Layout
//!
//! - [`stress_assembly`] — element stiffness and mass matrix assembly.
//! - [`navier_stokes_kernel`] — pressure-Poisson and Jacobi pressure solves.
//! - [`heat_kernel`] — explicit and ADI heat-diffusion stencils.
//!
//! # Feature gate
//!
//! Enable via `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! oxirs-physics = { version = "*", features = ["gpu"] }
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_physics::gpu::{GpuElementDescriptor, GpuError, StressAssemblyDispatcher};
//!
//! let dispatcher = StressAssemblyDispatcher::new();
//! let elements = vec![GpuElementDescriptor::default(); 8];
//! let result = dispatcher.dispatch_stiffness_assembly(&elements);
//! // Without `gpu` feature: Err(GpuError::BackendUnavailable)
//! ```

pub mod heat_kernel;
pub mod navier_stokes_kernel;
pub mod stress_assembly;

use thiserror::Error;

/// Errors that can arise when using a GPU dispatch path.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GpuError {
    /// The GPU backend is not compiled in or is unavailable at runtime.
    #[error("GPU backend is unavailable; falling back to CPU")]
    BackendUnavailable,

    /// A kernel produced an unexpected result.
    #[error("GPU kernel dispatch error: {0}")]
    DispatchError(String),

    /// Inputs to a GPU kernel were malformed.
    #[error("GPU kernel input error: {0}")]
    InvalidInput(String),
}

/// Result type for GPU dispatch operations.
pub type GpuResult<T> = Result<T, GpuError>;

/// Whether the underlying compute backend is available at runtime.
#[inline]
pub fn backend_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        true
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

pub use heat_kernel::HeatKernelDispatcher;
pub use navier_stokes_kernel::NavierStokesKernelDispatcher;
pub use stress_assembly::{
    FemElementKind, GpuElementContribution, GpuElementDescriptor, StressAssemblyDispatcher,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_available_matches_feature() {
        #[cfg(feature = "gpu")]
        assert!(backend_available());
        #[cfg(not(feature = "gpu"))]
        assert!(!backend_available());
    }

    #[test]
    fn gpu_error_display() {
        assert!(GpuError::BackendUnavailable
            .to_string()
            .contains("unavailable"));
        assert!(GpuError::DispatchError("foo".to_string())
            .to_string()
            .contains("foo"));
        assert!(GpuError::InvalidInput("bad".to_string())
            .to_string()
            .contains("bad"));
    }
}
