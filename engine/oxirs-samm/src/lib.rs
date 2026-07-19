//! # OxiRS SAMM - Semantic Aspect Meta Model Implementation
//!
//! [![Version](https://img.shields.io/badge/version-0.3.3-blue)](https://github.com/cool-japan/oxirs/releases)
//!
//! **Status**: Production Release (v0.3.3)
//! ✅ All public APIs documented. Production-ready with API stability guarantees.
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
pub mod analytics;
/// Structural differ for SAMM Aspect Models.
pub mod aspect_differ;
pub mod cache;
pub mod cloud_backends;
/// AWS S3 (and S3-compatible) storage backend: [`cloud_backends_aws::S3Config`] / [`cloud_backends_aws::S3Backend`].
pub mod cloud_backends_aws;
/// Azure Blob Storage backend: [`cloud_backends_azure::AzureConfig`] / [`cloud_backends_azure::AzureBlobBackend`].
pub mod cloud_backends_azure;
/// Common HMAC/SHA-256, hex, URL and XML helpers shared by the cloud backends.
pub mod cloud_backends_common;
/// Google Cloud Storage backend: [`cloud_backends_gcp::GcsConfig`] / [`cloud_backends_gcp::GcsBackend`].
pub mod cloud_backends_gcp;
/// Generic HTTP REST storage backend: [`cloud_backends_http::HttpConfig`] / [`cloud_backends_http::HttpBackend`].
pub mod cloud_backends_http;
/// Backend aggregation plus the local-filesystem adapter [`cloud_backends_impl::LocalFsBackend`].
pub mod cloud_backends_impl;
/// Multi-backend replication and synchronization for cloud storage.
pub mod cloud_backends_sync;
#[cfg(test)]
mod cloud_backends_tests;
/// Cloud backend configuration envelopes and capability/access-policy enums.
pub mod cloud_backends_types;
pub mod cloud_client;
pub mod cloud_storage;
pub mod codegen;
pub mod comparison;
pub mod documentation;
pub mod dtdl_parser;
pub mod error;
pub mod generators;
pub mod graph_analytics;
/// JSON-LD 1.1 compaction and framing algorithms.
pub mod jsonld;
pub mod metamodel;
pub mod migration;
pub mod operation_mapper;
pub mod package;
pub mod parser;
pub mod performance;
pub mod production;
pub mod query;
pub mod query_cache;
pub mod serializer;
pub mod simd_ops;
/// SAMM submodel templates: reusable groups of properties for aspect models.
pub mod submodel_templates;
pub mod templates;
pub mod transformation;
pub mod utils;
pub mod validation;
pub mod validator;
pub mod versioning;
// SAMM vocabulary mapper (v1.1.0 round 5)
pub mod vocabulary_mapper;

// SAMM/BAMM unit catalog with SI and derived units (v1.1.0 round 6)
pub mod unit_catalog;

/// Registry of SAMM constraint types with validation support (v1.1.0 round 7).
pub mod constraint_registry;

/// SAMM aspect hierarchical property chain traversal (v1.1.0 round 8).
pub mod aspect_chain;

/// SAMM operation registry for aspect operations (v1.1.0 round 9).
pub mod operation_registry;

/// SAMM physical unit conversion engine (v1.1.0 round 10).
pub mod unit_converter;

/// SAMM event model for IoT events (v1.1.0 round 11).
pub mod event_model;

/// Aspect model serialization to JSON/YAML/Text (v1.1.0 round 12).
pub mod aspect_export;

/// SAMM characteristic validation (v1.1.0 round 13).
pub mod characteristic_validator;

/// SAMM aspect model validation — required properties, cardinality, constraints, cycles (v1.1.0 round 13).
pub mod aspect_validator;

/// SAMM aspect model serialization to JSON/YAML/Turtle/Compact formats (v1.1.0 round 14).
pub mod model_serializer;

/// SAMM entity resolution: hierarchy traversal, property flattening, and comparison (v1.1.0 round 12).
pub mod entity_resolver;

/// SAMM payload generation from aspect model definitions (v1.1.0 round 11).
pub mod payload_generator;

/// SAMM property mapping between aspect models and target schemas (v1.1.0 round 15).
pub mod property_mapper;

/// SAMM characteristic constraint validators (RangeConstraint, LengthConstraint, EncodingConstraint, RegularExpressionConstraint) (v1.1.0 round 16).
pub mod constraint_validator;

/// Cross-model URN reference validation for SAMM aspect models (v0.3.1).
///
/// Provides [`CrossModelRegistry`] for indexing multiple SAMM model namespaces
/// and [`CrossModelValidator`] for checking that all cross-model URN references
/// resolve.  See the [`cross_model`] module documentation for a full example.
pub mod cross_model;
pub use cross_model::{
    CrossModelError, CrossModelReference, CrossModelRegistry, CrossModelValidator, ModelEntry,
};
// Re-export ValidationReport from cross_model under a distinct name to avoid
// clashing with validation::ValidationReport.
pub use cross_model::ValidationReport as CrossModelValidationReport;

/// ESMF SDK 2.x feature parity matrix and status report (v0.3.0).
pub mod parity;
pub use parity::{
    generate_report, load_catalog, FeatureCategory, FeatureEntry, FeatureStatus, ImplStatus,
    ParityMatrix,
};

/// Command-line interface sub-command implementations (v0.3.0).
pub mod cli;

// Re-exports for convenience
pub use analytics::{
    Anomaly, AnomalyType, BatchCorrelationError, BatchCorrelationMatrix, BenchmarkComparison,
    BenchmarkLevel, BestPracticeCheck, BestPracticeReport, CheckCategory, ComplexityAssessment,
    ComplexityLevel, ConfidenceLevel, CorrelationDirection, CorrelationInsight,
    CorrelationStrength, DependencyMetrics, DistributionAnalysis, DistributionFit,
    DistributionParameters, DistributionStats, DistributionType, ModelAnalytics,
    PropertyCorrelationMatrix, QualityTest, Recommendation, RecommendationType, Severity,
    StatisticalAnomaly, StatisticalMetrics,
};
pub use cache::{
    AspectCache, CacheStatistics, CharacteristicCache, EntityCache, LruModelCache, OperationCache,
    PropertyCache, TtlCache, TtlCacheStatistics,
};
pub use cloud_client::{
    CloudStorageClient, CloudStorageError, MockCloudStorage, RetryableCloudClient,
};
pub use cloud_storage::{
    BatchResult, CacheStats, CloudModelStorage, CloudStorageBackend, MemoryBackend, ModelInfo,
    ObjectMetadata,
};
pub use codegen::{
    HttpMethod, JsonSchemaGenerator, JsonSchemaOptions, JsonSchemaValidator, OpenApiGenerator,
    OpenApiOptions, OpenApiVersion, PaginationConfig, ValidationError as JsonSchemaValidationError,
};
pub use comparison::{MetadataChange, MetadataChangeType, ModelComparison, PropertyChange};
pub use documentation::{DocumentationFormat, DocumentationGenerator, DocumentationStyle};
pub use dtdl_parser::parse_dtdl_interface;
pub use error::{ErrorCategory, Result, SammError, SourceLocation};
pub use generators::{GeneratedFile, MultiFileGenerator, MultiFileOptions, OutputLayout};
pub use graph_analytics::{
    CentralityMetrics, ChangeMagnitude, ColorScheme, Community, Cycle, CycleBreakSuggestion,
    GraphComparison, GraphMetrics, ImpactAnalysis, ModelGraph, RiskLevel, VisualizationStyle,
};
pub use metamodel::{Aspect, Characteristic, Entity, Operation, Property};
pub use migration::{MigrationOptions, MigrationResult, ModelMigrator, SammVersion};
pub use parser::{ErrorRecoveryStrategy, RecoveryAction, RecoveryContext, StreamingParser};
pub use performance::{BatchProcessor, ModelCache, PerformanceConfig};
pub use production::{
    health_check, init_production, HealthCheck, HealthStatus, MetricsCollector, OperationType,
    ProductionConfig,
};
pub use query::{ComplexityMetrics, Dependency, ModelQuery};
pub use query_cache::{CacheStatistics as QueryCacheStatistics, CachedModelQuery};
pub use serializer::{
    serialize_aspect_to_file, serialize_aspect_to_jsonld_file, serialize_aspect_to_jsonld_string,
    serialize_aspect_to_rdfxml_file, serialize_aspect_to_rdfxml_string, serialize_aspect_to_string,
    JsonLdSerializer, RdfXmlSerializer, TurtleSerializer,
};
pub use templates::{
    scaffolding::{ModelTemplate, TemplateBuilder, TemplateRegistry},
    PostRenderHook, PreRenderHook, TemplateContext, TemplateEngine, ValidationHook,
};
pub use transformation::{ModelTransformation, TransformationRule};
pub use validation::{
    BatchValidator, SammSchemaValidator, SchemaValidationError, SchemaValidationWarning,
    ValidationReport, ValidationRule,
};
pub use versioning::{AspectMigrationRegistry, AspectVersion, MigrationStep, VersionedAspect};

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
