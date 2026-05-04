//! SAMM Model Validation
//!
//! This module provides advanced validation of SAMM Aspect Model definitions
//! against the SAMM 2.3.0 specification, extending the basic SHACL-based
//! validator with deep structural and semantic checks.
//!
//! # Modules
//!
//! - [`schema_validator`] – enforces SAMM spec constraints (naming, types,
//!   required fields, enumeration invariants, etc.)
//! - [`batch`] – validate multiple aspects in one call with optional GPU
//!   acceleration (see [`batch::BatchValidator`]).
//! - [`gpu_kernels`] – GPU constraint dispatcher used internally by
//!   [`batch::BatchValidator`].
//!
//! # Quick Example
//!
//! ```rust,no_run
//! use oxirs_samm::metamodel::Aspect;
//! use oxirs_samm::validation::SammSchemaValidator;
//! use oxirs_samm::validation::batch::BatchValidator;
//!
//! let aspect = Aspect::new("urn:samm:org.example:1.0.0#Movement".to_string());
//! let validator = SammSchemaValidator::new();
//! let report = validator.validate_aspect(&aspect);
//! println!("Valid: {}", report.is_valid);
//!
//! // Batch validation
//! let reports = BatchValidator::new().validate_batch(&[&aspect]);
//! assert_eq!(reports.len(), 1);
//! ```

pub mod batch;
pub mod gpu_kernels;
pub mod schema_validator;

pub use batch::BatchValidator;
pub use schema_validator::{
    SammSchemaValidator, SchemaValidationError, SchemaValidationWarning, ValidationReport,
    ValidationRule,
};
