//! Batch SAMM aspect validation with optional GPU acceleration.
//!
//! [`BatchValidator`] validates a slice of [`Aspect`] references and returns
//! one [`ValidationReport`] per aspect.  By default all validation runs on
//! the CPU using the existing [`SammSchemaValidator`].  When built with the
//! `gpu` feature **and** `with_gpu(true)` is called, the validator attempts
//! to dispatch constraint checks to the GPU backend via
//! [`gpu_kernels::GpuBatchValidator`].  If the GPU path fails (e.g. the
//! backend is unavailable at runtime) the implementation transparently falls
//! back to the CPU path — it never panics.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_samm::metamodel::Aspect;
//! use oxirs_samm::validation::batch::BatchValidator;
//!
//! let a1 = Aspect::new("urn:samm:org.example:1.0.0#Foo".to_string());
//! let a2 = Aspect::new("urn:samm:org.example:1.0.0#Bar".to_string());
//! let reports = BatchValidator::new().validate_batch(&[&a1, &a2]);
//! assert_eq!(reports.len(), 2);
//! ```

use crate::metamodel::Aspect;
use crate::validation::gpu_kernels::{self, GpuBatchValidator};
use crate::validation::schema_validator::{SammSchemaValidator, ValidationReport};

/// Validates a batch of [`Aspect`] models, optionally using a GPU backend.
///
/// # Builder
///
/// ```rust,no_run
/// use oxirs_samm::validation::batch::BatchValidator;
///
/// let validator = BatchValidator::new()
///     .with_gpu(false)      // CPU-only (default)
///     .without_naming_checks();
/// ```
#[derive(Debug)]
pub struct BatchValidator {
    /// Whether to attempt GPU-accelerated validation.
    use_gpu: bool,
    /// Underlying schema validator used on the CPU path.
    schema_validator: SammSchemaValidator,
}

impl Default for BatchValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchValidator {
    /// Create a new `BatchValidator` with CPU-only validation and all checks
    /// enabled.
    pub fn new() -> Self {
        Self {
            use_gpu: false,
            schema_validator: SammSchemaValidator::new(),
        }
    }

    /// Enable or disable GPU acceleration.
    ///
    /// When `enabled` is `true` **and** the crate was compiled with the `gpu`
    /// feature, the validator will attempt to use the GPU backend.  If the
    /// backend is unavailable at runtime the CPU path is used automatically.
    ///
    /// Passing `false` (the default) always uses the CPU path regardless of
    /// the feature flag.
    pub fn with_gpu(mut self, enabled: bool) -> Self {
        self.use_gpu = enabled;
        self
    }

    /// Disable naming-convention checks in the underlying schema validator.
    pub fn without_naming_checks(mut self) -> Self {
        self.schema_validator = SammSchemaValidator::new().without_naming_checks();
        self
    }

    /// Enable warnings for missing preferred names.
    pub fn with_name_warnings(mut self) -> Self {
        self.schema_validator = SammSchemaValidator::new().with_name_warnings();
        self
    }

    /// Validate a batch of aspects and return one report per aspect.
    ///
    /// The order of the returned reports mirrors the order of `aspects`.  An
    /// empty input produces an empty output.
    ///
    /// When GPU acceleration is requested the validator first attempts to run
    /// bulk constraint checks on the GPU; if that fails it silently falls back
    /// to the CPU path.
    pub fn validate_batch(&self, aspects: &[&Aspect]) -> Vec<ValidationReport> {
        if aspects.is_empty() {
            return Vec::new();
        }

        if self.use_gpu {
            match self.try_gpu_batch(aspects) {
                Ok(reports) => return reports,
                Err(_) => {
                    // GPU path failed — fall through to CPU path silently.
                }
            }
        }

        self.cpu_batch(aspects)
    }

    // ------------------------------------------------------------------ //
    //  Private helpers                                                     //
    // ------------------------------------------------------------------ //

    /// CPU path: validate each aspect sequentially with the schema validator.
    fn cpu_batch(&self, aspects: &[&Aspect]) -> Vec<ValidationReport> {
        aspects
            .iter()
            .map(|aspect| self.schema_validator.validate_aspect(aspect))
            .collect()
    }

    /// GPU path: attempt GPU constraint dispatch, then merge with CPU reports.
    ///
    /// # Errors
    ///
    /// Returns the [`gpu_kernels::GpuError`] if the backend is unavailable.
    fn try_gpu_batch(
        &self,
        aspects: &[&Aspect],
    ) -> Result<Vec<ValidationReport>, gpu_kernels::GpuError> {
        let gpu = GpuBatchValidator::new();

        // Dispatch all constraint categories in a single GPU pass.
        // If this returns Err, the caller falls back to CPU.
        let gpu_violations = gpu.dispatch_all_checks(aspects.len())?;

        // The scirs2_core GPU stub does not produce real violations, so we
        // obtain the full structural reports from the CPU schema validator and
        // incorporate any GPU-detected violations on top.
        let mut reports: Vec<ValidationReport> = aspects
            .iter()
            .map(|aspect| self.schema_validator.validate_aspect(aspect))
            .collect();

        // Merge GPU-detected violations into the corresponding reports.
        for violation in gpu_violations {
            if let Some(report) = reports.get_mut(violation.aspect_index) {
                report.add_error(crate::validation::schema_validator::SchemaValidationError {
                    element_urn: String::new(),
                    rule: crate::validation::schema_validator::ValidationRule::Custom(format!(
                        "gpu:{:?}",
                        violation.kind
                    )),
                    message: violation.message,
                });
            }
        }

        Ok(reports)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::{Aspect, Characteristic, CharacteristicKind, Property};

    fn valid_aspect(base_urn: &str) -> Aspect {
        // Strip fragment from base_urn to build sub-element URNs
        let ns = base_urn.split('#').next().unwrap_or(base_urn);
        let mut aspect = Aspect::new(base_urn.to_string());
        let char = Characteristic::new(
            format!("{}#speedChar", ns),
            CharacteristicKind::Measurement {
                unit: "unit:kilometre".to_string(),
            },
        )
        .with_data_type("http://www.w3.org/2001/XMLSchema#float".to_string());
        // Property name (the fragment) must be camelCase, starting with lowercase.
        let prop = Property::new(format!("{}#speed", ns)).with_characteristic(char);
        aspect.add_property(prop);
        aspect
    }

    fn invalid_aspect(urn: &str) -> Aspect {
        // Aspect with no properties — always fails schema validation
        Aspect::new(urn.to_string())
    }

    #[test]
    fn test_batch_validator_cpu_basic() {
        let a1 = valid_aspect("urn:samm:org.example:1.0.0#Aspect1");
        let a2 = valid_aspect("urn:samm:org.example:1.0.0#Aspect2");
        let a3 = valid_aspect("urn:samm:org.example:1.0.0#Aspect3");

        let reports = BatchValidator::new().validate_batch(&[&a1, &a2, &a3]);
        assert_eq!(reports.len(), 3);
    }

    #[test]
    fn test_batch_validator_empty_batch() {
        let reports = BatchValidator::new().validate_batch(&[]);
        assert!(reports.is_empty());
    }

    #[test]
    fn test_batch_validator_cpu_mixed() {
        let valid = valid_aspect("urn:samm:org.example:1.0.0#Good");
        let invalid = invalid_aspect("urn:samm:org.example:1.0.0#Bad");

        let reports = BatchValidator::new().validate_batch(&[&valid, &invalid]);
        assert_eq!(reports.len(), 2);
        assert!(reports[0].is_valid, "first aspect should be valid");
        assert!(!reports[1].is_valid, "second aspect should be invalid");
    }

    #[test]
    fn test_batch_validator_with_gpu_false() {
        let a = valid_aspect("urn:samm:org.example:1.0.0#GpuFalse");
        let reports = BatchValidator::new().with_gpu(false).validate_batch(&[&a]);
        assert_eq!(reports.len(), 1);
    }

    #[test]
    fn test_batch_validator_gpu_true_falls_back_to_cpu() {
        // With gpu=true but no actual GPU backend compiled, it must fall back.
        let a = valid_aspect("urn:samm:org.example:1.0.0#GpuTrue");
        let reports = BatchValidator::new().with_gpu(true).validate_batch(&[&a]);
        assert_eq!(reports.len(), 1);
        // Result is identical to the CPU path
        let cpu_reports = BatchValidator::new().with_gpu(false).validate_batch(&[&a]);
        assert_eq!(reports[0].is_valid, cpu_reports[0].is_valid);
    }
}
