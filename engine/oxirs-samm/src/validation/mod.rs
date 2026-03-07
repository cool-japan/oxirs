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
//!
//! # Quick Example
//!
//! ```rust,no_run
//! use oxirs_samm::metamodel::Aspect;
//! use oxirs_samm::validation::SammSchemaValidator;
//!
//! let aspect = Aspect::new("urn:samm:org.example:1.0.0#Movement".to_string());
//! let validator = SammSchemaValidator::new();
//! let report = validator.validate_aspect(&aspect);
//! println!("Valid: {}", report.is_valid);
//! ```

pub mod schema_validator;

pub use schema_validator::{
    SammSchemaValidator, SchemaValidationError, SchemaValidationWarning, ValidationReport,
    ValidationRule,
};
