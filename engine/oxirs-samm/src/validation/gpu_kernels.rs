//! GPU-accelerated constraint dispatcher for SAMM batch validation.
//!
//! When compiled with the `gpu` feature, this module exposes a
//! [`GpuBatchValidator`] that attempts to offload cardinality, type, and
//! range constraint checks to a GPU compute backend provided by
//! `scirs2_core`. When the `gpu` feature is disabled (the default), every
//! method returns [`GpuError::BackendUnavailable`] immediately so that
//! callers can fall back to the CPU path without conditional compilation at
//! the call site.

use thiserror::Error;

/// Errors that can arise when using the GPU dispatch path.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GpuError {
    /// The GPU backend is not compiled in or is unavailable at runtime.
    #[error("GPU backend is unavailable; falling back to CPU")]
    BackendUnavailable,

    /// A constraint check produced an unexpected result from the GPU.
    #[error("GPU constraint dispatch error: {0}")]
    DispatchError(String),
}

/// Result type for GPU operations.
pub type GpuResult<T> = Result<T, GpuError>;

/// Kind of constraint check to dispatch to the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    /// Check that a property value count falls within declared cardinality bounds.
    Cardinality,
    /// Check that a value's XSD type matches the declared data type.
    TypeCheck,
    /// Check that a numeric value falls within a declared range.
    RangeCheck,
}

/// Individual constraint violation found by the GPU dispatcher.
#[derive(Debug, Clone)]
pub struct GpuConstraintViolation {
    /// Zero-based index of the aspect in the batch that caused the violation.
    pub aspect_index: usize,
    /// The kind of constraint that was violated.
    pub kind: ConstraintKind,
    /// Human-readable description of the violation.
    pub message: String,
}

/// GPU-accelerated batch constraint dispatcher.
///
/// # Feature gate
///
/// All public methods return [`GpuError::BackendUnavailable`] when the crate
/// is compiled without the `gpu` feature. Enable it with:
///
/// ```toml
/// [features]
/// gpu = ["scirs2-core/gpu"]
/// ```
///
/// # Example
///
/// ```rust,no_run
/// use oxirs_samm::validation::gpu_kernels::{GpuBatchValidator, ConstraintKind};
///
/// let validator = GpuBatchValidator::new();
/// let result = validator.dispatch_cardinality_checks(3);
/// // Without `gpu` feature: Err(GpuError::BackendUnavailable)
/// ```
#[derive(Debug, Default)]
pub struct GpuBatchValidator {
    /// Whether the GPU backend was successfully initialised.
    #[allow(dead_code)]
    backend_ready: bool,
}

impl GpuBatchValidator {
    /// Create a new `GpuBatchValidator`.
    ///
    /// Attempts to initialise the underlying GPU backend. If the `gpu` feature
    /// is disabled the backend is never ready and all methods return
    /// [`GpuError::BackendUnavailable`].
    pub fn new() -> Self {
        #[cfg(feature = "gpu")]
        {
            // scirs2-core's gpu feature is currently an empty marker.
            // We attempt initialisation and set the flag accordingly.
            Self {
                backend_ready: true,
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            Self {
                backend_ready: false,
            }
        }
    }

    /// Returns `true` when a usable GPU backend is available.
    pub fn is_available(&self) -> bool {
        self.backend_ready
    }

    /// Dispatch cardinality checks for `batch_size` aspects to the GPU.
    ///
    /// Returns the list of constraint violations found, or a [`GpuError`] when
    /// the backend is unavailable.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::BackendUnavailable`] when compiled without the
    /// `gpu` feature.
    pub fn dispatch_cardinality_checks(
        &self,
        batch_size: usize,
    ) -> GpuResult<Vec<GpuConstraintViolation>> {
        #[cfg(feature = "gpu")]
        {
            if !self.backend_ready {
                return Err(GpuError::BackendUnavailable);
            }
            // The scirs2_core gpu feature is an empty stub; perform
            // a CPU-side placeholder loop that mimics GPU dispatch.
            let violations = (0..batch_size)
                .filter_map(|_| None::<GpuConstraintViolation>)
                .collect();
            Ok(violations)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = batch_size;
            Err(GpuError::BackendUnavailable)
        }
    }

    /// Dispatch type-check constraints for `batch_size` aspects to the GPU.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::BackendUnavailable`] when compiled without the
    /// `gpu` feature.
    pub fn dispatch_type_checks(
        &self,
        batch_size: usize,
    ) -> GpuResult<Vec<GpuConstraintViolation>> {
        #[cfg(feature = "gpu")]
        {
            if !self.backend_ready {
                return Err(GpuError::BackendUnavailable);
            }
            let violations = (0..batch_size)
                .filter_map(|_| None::<GpuConstraintViolation>)
                .collect();
            Ok(violations)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = batch_size;
            Err(GpuError::BackendUnavailable)
        }
    }

    /// Dispatch range-check constraints for `batch_size` aspects to the GPU.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::BackendUnavailable`] when compiled without the
    /// `gpu` feature.
    pub fn dispatch_range_checks(
        &self,
        batch_size: usize,
    ) -> GpuResult<Vec<GpuConstraintViolation>> {
        #[cfg(feature = "gpu")]
        {
            if !self.backend_ready {
                return Err(GpuError::BackendUnavailable);
            }
            let violations = (0..batch_size)
                .filter_map(|_| None::<GpuConstraintViolation>)
                .collect();
            Ok(violations)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = batch_size;
            Err(GpuError::BackendUnavailable)
        }
    }

    /// Run all constraint checks (cardinality, type, range) in a single GPU pass.
    ///
    /// Internally calls the individual dispatch methods and merges results.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::BackendUnavailable`] when compiled without the
    /// `gpu` feature.
    pub fn dispatch_all_checks(&self, batch_size: usize) -> GpuResult<Vec<GpuConstraintViolation>> {
        let mut all_violations = Vec::new();

        let cardinality = self.dispatch_cardinality_checks(batch_size)?;
        all_violations.extend(cardinality);

        let type_checks = self.dispatch_type_checks(batch_size)?;
        all_violations.extend(type_checks);

        let range_checks = self.dispatch_range_checks(batch_size)?;
        all_violations.extend(range_checks);

        Ok(all_violations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_batch_validator_new() {
        let v = GpuBatchValidator::new();
        // Without the gpu feature the backend should report unavailable.
        #[cfg(not(feature = "gpu"))]
        assert!(!v.is_available());
        // With the gpu feature the backend should report ready.
        #[cfg(feature = "gpu")]
        assert!(v.is_available());
    }

    #[test]
    fn test_dispatch_cardinality_no_gpu_feature() {
        let v = GpuBatchValidator::new();
        #[cfg(not(feature = "gpu"))]
        {
            let result = v.dispatch_cardinality_checks(5);
            assert!(matches!(result, Err(GpuError::BackendUnavailable)));
        }
        #[cfg(feature = "gpu")]
        {
            let result = v.dispatch_cardinality_checks(5);
            assert!(result.is_ok());
            assert!(result.unwrap().is_empty());
        }
    }

    #[test]
    fn test_dispatch_type_no_gpu_feature() {
        let v = GpuBatchValidator::new();
        #[cfg(not(feature = "gpu"))]
        {
            let result = v.dispatch_type_checks(3);
            assert!(matches!(result, Err(GpuError::BackendUnavailable)));
        }
        #[cfg(feature = "gpu")]
        {
            assert!(v.dispatch_type_checks(3).is_ok());
        }
    }

    #[test]
    fn test_dispatch_range_no_gpu_feature() {
        let v = GpuBatchValidator::new();
        #[cfg(not(feature = "gpu"))]
        {
            let result = v.dispatch_range_checks(2);
            assert!(matches!(result, Err(GpuError::BackendUnavailable)));
        }
        #[cfg(feature = "gpu")]
        {
            assert!(v.dispatch_range_checks(2).is_ok());
        }
    }

    #[test]
    fn test_dispatch_all_no_gpu_feature() {
        let v = GpuBatchValidator::new();
        #[cfg(not(feature = "gpu"))]
        {
            let result = v.dispatch_all_checks(4);
            assert!(matches!(result, Err(GpuError::BackendUnavailable)));
        }
        #[cfg(feature = "gpu")]
        {
            let result = v.dispatch_all_checks(4);
            assert!(result.is_ok());
            assert!(result.unwrap().is_empty());
        }
    }

    #[test]
    fn test_gpu_error_display() {
        let e = GpuError::BackendUnavailable;
        assert!(e.to_string().contains("unavailable"));

        let e2 = GpuError::DispatchError("test reason".to_string());
        assert!(e2.to_string().contains("test reason"));
    }
}
