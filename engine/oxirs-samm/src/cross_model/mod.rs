//! Cross-model reference validation for SAMM aspect models.
//!
//! SAMM models may reference types defined in **other** model files via
//! cross-model URN references (e.g. `urn:samm:other.namespace:1.0.0#SomeType`).
//! This module provides a [`CrossModelRegistry`] for loading and indexing
//! multiple model namespaces, and a [`CrossModelValidator`] for checking that
//! all such cross-model references resolve.
//!
//! # Example
//!
//! ```rust
//! use oxirs_samm::cross_model::{
//!     CrossModelRegistry, CrossModelValidator, CrossModelReference, ModelEntry,
//! };
//!
//! let mut registry = CrossModelRegistry::new();
//! registry.register_model(ModelEntry {
//!     namespace: "urn:samm:org.example.b:1.0.0".to_string(),
//!     file_path: None,
//!     exported_urns: vec![
//!         "urn:samm:org.example.b:1.0.0#TemperatureChar".to_string(),
//!     ],
//! }).unwrap();
//!
//! let references = CrossModelValidator::extract_cross_references(
//!     "urn:samm:org.example.a:1.0.0",
//!     &[("dataType", "urn:samm:org.example.b:1.0.0#TemperatureChar")],
//! );
//!
//! let validator = CrossModelValidator::new(&registry);
//! let report = validator.validate("urn:samm:org.example.a:1.0.0", &references);
//! assert!(report.is_valid());
//! ```

pub mod registry;
pub mod validator;

pub use registry::{CrossModelRegistry, ModelEntry};
pub use validator::{CrossModelReference, CrossModelValidator, ValidationReport};

/// Errors that can arise when working with the cross-model registry or
/// performing cross-model validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossModelError {
    /// A model with this namespace was already registered.
    DuplicateNamespace(String),

    /// A URN exported by a new model is already owned by an existing model.
    DuplicateUrn {
        /// The conflicting URN.
        urn: String,
        /// Namespace of the model that already owns the URN.
        existing_namespace: String,
    },

    /// The `namespace` field of a [`ModelEntry`] was empty.
    EmptyNamespace,

    /// The supplied string does not look like a valid SAMM URN.
    InvalidUrn(String),
}

impl std::fmt::Display for CrossModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateNamespace(ns) => {
                write!(f, "namespace already registered: {ns}")
            }
            Self::DuplicateUrn {
                urn,
                existing_namespace,
            } => write!(
                f,
                "URN '{urn}' already registered under namespace '{existing_namespace}'"
            ),
            Self::EmptyNamespace => write!(f, "model namespace must not be empty"),
            Self::InvalidUrn(urn) => write!(f, "invalid SAMM URN: '{urn}'"),
        }
    }
}

impl std::error::Error for CrossModelError {}
