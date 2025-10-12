//! # OxiRS SAMM - Semantic Aspect Meta Model Implementation
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0--alpha.2-orange)](https://github.com/cool-japan/oxirs/releases)
//!
//! **Status**: Alpha Release (v0.1.0-alpha.3)
//! ⚠️ APIs may change. Not recommended for production use.
//!
//! This crate provides a Rust implementation of the Semantic Aspect Meta Model (SAMM),
//! which enables the creation of models to describe the semantics of digital twins.
//!
//! ## Overview
//!
//! SAMM (formerly BAMM) is a meta model for defining domain-specific aspects of digital twins.
//! It provides a set of predefined objects that allow domain experts to define Aspect Models
//! and complement digital twins with a semantic foundation.
//!
//! ## Core Concepts
//!
//! - **Aspect**: The root element describing a digital twin's specific aspect
//! - **Property**: A named feature of an Aspect with a defined Characteristic
//! - **Characteristic**: Describes the semantics of a Property's value
//! - **Entity**: A complex data structure with multiple properties
//! - **Operation**: A function that can be performed on an Aspect
//! - **Event**: An occurrence that can be emitted by an Aspect
//!
//! ## Features
//!
//! - ✅ **SAMM 2.3.0 Support**: Full support for latest SAMM specification
//! - ✅ **RDF/Turtle Parsing**: Load SAMM models from Turtle files
//! - ✅ **SHACL Validation**: Validate models against SAMM shapes
//! - 🚧 **Code Generation**: Generate Rust code from SAMM models (coming soon)
//! - 🚧 **AAS Integration**: Asset Administration Shell support (coming soon)
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxirs_samm::metamodel::{Aspect, ModelElement};
//! use oxirs_samm::parser::parse_aspect_model;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Parse a SAMM model from a Turtle file
//! let aspect = parse_aspect_model("path/to/AspectModel.ttl").await?;
//!
//! println!("Aspect: {}", aspect.name());
//! for property in aspect.properties() {
//!     println!("  Property: {}", property.name());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## References
//!
//! - [SAMM Specification](https://eclipse-esmf.github.io/samm-specification/snapshot/index.html)
//! - [Eclipse ESMF](https://github.com/eclipse-esmf)
//! - [SAMM Java SDK](https://github.com/eclipse-esmf/esmf-sdk)

// TODO: Complete documentation for all public APIs (137 items - tracked for future work)
#![allow(missing_docs)]
#![warn(clippy::all)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

pub mod aas_parser;
pub mod error;
pub mod generators;
pub mod metamodel;
pub mod package;
pub mod parser;
pub mod performance;
pub mod production;
pub mod serializer;
pub mod templates;
pub mod validator;

// Re-exports for convenience
pub use error::{Result, SammError};
pub use metamodel::{Aspect, Characteristic, Entity, Operation, Property};
pub use performance::{BatchProcessor, ModelCache, PerformanceConfig};
pub use production::{
    health_check, init_production, HealthCheck, HealthStatus, MetricsCollector, OperationType,
    ProductionConfig,
};
pub use templates::{TemplateContext, TemplateEngine};

/// SAMM version supported by this implementation
pub const SAMM_VERSION: &str = "2.3.0";

/// SAMM namespace URN prefix
pub const SAMM_NAMESPACE: &str = "urn:samm:org.eclipse.esmf.samm";

/// SAMM characteristics namespace
pub const SAMM_C_NAMESPACE: &str = "urn:samm:org.eclipse.esmf.samm:characteristic";

/// SAMM entities namespace
pub const SAMM_E_NAMESPACE: &str = "urn:samm:org.eclipse.esmf.samm:entity";

/// SAMM units namespace
pub const SAMM_U_NAMESPACE: &str = "urn:samm:org.eclipse.esmf.samm:unit";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_samm_version() {
        assert_eq!(SAMM_VERSION, "2.3.0");
    }

    #[test]
    fn test_namespaces() {
        assert!(SAMM_NAMESPACE.starts_with("urn:samm:"));
        assert!(SAMM_C_NAMESPACE.contains("characteristic"));
        assert!(SAMM_E_NAMESPACE.contains("entity"));
        assert!(SAMM_U_NAMESPACE.contains("unit"));
    }
}
