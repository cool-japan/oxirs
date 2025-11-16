//! # OxiRS SAMM - Semantic Aspect Meta Model Implementation
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0--beta.1-blue)](https://github.com/cool-japan/oxirs/releases)
//!
//! **Status**: Beta.1 Release (v0.1.0-beta.1)
//! ✅ All public APIs documented. API stability guarantees in place.
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
//! - ✅ **Code Generation**: Generate code in 7+ languages (Rust, TypeScript, Python, Java, Scala, GraphQL, SQL)
//! - ✅ **AAS Integration**: Full Asset Administration Shell V3.0 support (XML, JSON, AASX packages)
//! - ✅ **Performance Optimization**: Parallel processing, caching, profiling utilities
//! - ✅ **Production Monitoring**: Metrics collection, health checks, structured logging
//!
//! ## Quick Start
//!
//! ### Basic Usage - Parse and Validate
//!
//! ```rust,no_run
//! use oxirs_samm::metamodel::{Aspect, ModelElement};
//! use oxirs_samm::parser::parse_aspect_model;
//! use oxirs_samm::validator::validate_aspect;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Parse a SAMM model from a Turtle file
//! let aspect = parse_aspect_model("path/to/AspectModel.ttl").await?;
//!
//! // Validate the aspect model
//! let validation_result = validate_aspect(&aspect).await?;
//! if !validation_result.is_valid {
//!     for error in &validation_result.errors {
//!         eprintln!("Validation error: {}", error.message);
//!     }
//! }
//!
//! // Access aspect properties
//! println!("Aspect: {}", aspect.name());
//! for property in aspect.properties() {
//!     println!("  Property: {} (type: {:?})",
//!              property.name(),
//!              property.characteristic.as_ref().map(|c| &c.data_type));
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Advanced Usage - Code Generation
//!
//! ```rust,no_run
//! use oxirs_samm::parser::parse_aspect_model;
//! use oxirs_samm::generators::{
//!     generate_typescript, TsOptions,
//!     generate_graphql,
//!     generate_sql, SqlDialect,
//! };
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let aspect = parse_aspect_model("Movement.ttl").await?;
//!
//! // Generate TypeScript interfaces
//! let ts_code = generate_typescript(&aspect, TsOptions::default())?;
//! std::fs::write("movement.ts", ts_code)?;
//!
//! // Generate GraphQL schema
//! let graphql_schema = generate_graphql(&aspect)?;
//! std::fs::write("movement.graphql", graphql_schema)?;
//!
//! // Generate SQL DDL
//! let sql_ddl = generate_sql(&aspect, SqlDialect::PostgreSql)?;
//! std::fs::write("movement.sql", sql_ddl)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Performance Tuning
//!
//! ```rust,no_run
//! use oxirs_samm::{PerformanceConfig, BatchProcessor, ModelCache};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure performance settings
//! let config = PerformanceConfig {
//!     parallel_processing: true,
//!     num_workers: 8,
//!     cache_size: 100,
//!     profiling_enabled: true,
//!     ..Default::default()
//! };
//!
//! // Use batch processor for multiple models
//! let processor = BatchProcessor::new(config);
//! let models = vec![
//!     "model1_content".to_string(),
//!     "model2_content".to_string(),
//!     "model3_content".to_string(),
//! ];
//! let results = processor.process_batch(models, |model| {
//!     // Process each model
//!     Ok(model.len())
//! }).await?;
//!
//! // Use model cache for frequent lookups
//! let cache = ModelCache::new(100);
//! if let Some(cached_model) = cache.get("urn:samm:org.example:1.0.0#Movement") {
//!     println!("Found cached model: {}", cached_model.len());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Production Monitoring
//!
//! ```rust,no_run
//! use oxirs_samm::{
//!     init_production, ProductionConfig,
//!     MetricsCollector, OperationType,
//!     health_check,
//! };
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize production monitoring
//! let config = ProductionConfig::default();
//! init_production(&config)?;
//!
//! // Collect metrics
//! let metrics = MetricsCollector::global();
//! metrics.record_operation(OperationType::Parse);
//! metrics.record_operation_with_duration(OperationType::Parse, 150.0); // ms
//!
//! // Perform health checks
//! let health_result = health_check();
//! println!("System health: {:?}", health_result);
//!
//! // Get metrics snapshot
//! let snapshot = metrics.snapshot();
//! println!("Total operations: {}", snapshot.operations_total);
//! println!("Parse operations: {}", snapshot.parse_operations);
//! println!("Errors: {}", snapshot.errors_total);
//! # Ok(())
//! # }
//! ```
//!
//! ## References
//!
//! - [SAMM Specification](https://eclipse-esmf.github.io/samm-specification/snapshot/index.html)
//! - [Eclipse ESMF](https://github.com/eclipse-esmf)
//! - [SAMM Java SDK](https://github.com/eclipse-esmf/esmf-sdk)

// Documentation and linting configuration for Beta.1 release
// All public APIs are now documented - enforcing strict documentation requirements
#![deny(missing_docs)]
#![warn(clippy::all)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

pub mod aas_parser;
pub mod cache;
pub mod comparison;
pub mod error;
pub mod generators;
pub mod metamodel;
pub mod migration;
pub mod package;
pub mod parser;
pub mod performance;
pub mod production;
pub mod query;
pub mod serializer;
pub mod simd_ops;
pub mod templates;
pub mod transformation;
pub mod utils;
pub mod validator;

// Re-exports for convenience
pub use comparison::{MetadataChange, MetadataChangeType, ModelComparison, PropertyChange};
pub use error::{ErrorCategory, Result, SammError, SourceLocation};
pub use generators::{GeneratedFile, MultiFileGenerator, MultiFileOptions, OutputLayout};
pub use metamodel::{Aspect, Characteristic, Entity, Operation, Property};
pub use migration::{MigrationOptions, MigrationResult, ModelMigrator, SammVersion};
pub use parser::{ErrorRecoveryStrategy, RecoveryAction, RecoveryContext, StreamingParser};
pub use performance::{BatchProcessor, ModelCache, PerformanceConfig};
pub use production::{
    health_check, init_production, HealthCheck, HealthStatus, MetricsCollector, OperationType,
    ProductionConfig,
};
pub use query::{ComplexityMetrics, Dependency, ModelQuery};
pub use templates::{
    PostRenderHook, PreRenderHook, TemplateContext, TemplateEngine, ValidationHook,
};
pub use transformation::{ModelTransformation, TransformationRule};

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
