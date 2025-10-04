//! # OxiRS SHACL - RDF Validation Engine
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0--alpha.2-orange)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-shacl/badge.svg)](https://docs.rs/oxirs-shacl)
//!
//! **Status**: Alpha Release (v0.1.0-alpha.2)
//! ⚠️ APIs may change. Not recommended for production use.
//!
//! SHACL (Shapes Constraint Language) validation engine for RDF data.
//! Provides comprehensive constraint validation with SHACL Core and SHACL-SPARQL support.
//!
//! ## Features
//!
//! - **SHACL Core** - Complete SHACL Core constraint validation
//! - **SHACL-SPARQL** - SPARQL-based constraints (experimental)
//! - **Property Paths** - Full property path evaluation
//! - **Logical Constraints** - sh:and, sh:or, sh:not, sh:xone
//! - **Validation Reports** - Comprehensive violation reporting
//! - **Performance** - Optimized validation engine
//!
//! ## See Also
//!
//! - [`oxirs-core`](https://docs.rs/oxirs-core) - RDF data model
//! - [`oxirs-arq`](https://docs.rs/oxirs-arq) - SPARQL query engine
//!
//! ## Basic Usage
//!
//! ```rust
//! use oxirs_shacl::{ShaclValidator, ValidationConfig, ValidationStrategy};
//! use oxirs_core::{Store, model::*};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a new SHACL validator
//! let mut validator = ShaclValidator::new();
//!
//! // Load SHACL shapes from Turtle
//! let shapes_ttl = r#"
//! @prefix sh: <http://www.w3.org/ns/shacl#> .
//! @prefix ex: <http://example.org/> .
//! @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
//!
//! ex:PersonShape
//!     a sh:NodeShape ;
//!     sh:targetClass ex:Person ;
//!     sh:property [
//!         sh:path ex:name ;
//!         sh:datatype xsd:string ;
//!         sh:maxCount 1 ;
//!         sh:minCount 1 ;
//!     ] ;
//!     sh:property [
//!         sh:path ex:age ;
//!         sh:datatype xsd:integer ;
//!         sh:minInclusive 0 ;
//!         sh:maxInclusive 150 ;
//!     ] .
//! "#;
//!
//! validator.load_shapes_from_turtle(shapes_ttl)?;
//!
//! // Create test data
//! let data_ttl = r#"
//! @prefix ex: <http://example.org/> .
//!
//! ex:john a ex:Person ;
//!     ex:name "John Doe" ;
//!     ex:age 30 .
//!
//! ex:jane a ex:Person ;
//!     ex:name "Jane Smith" ;
//!     ex:age 200 .  # This violates the age constraint
//! "#;
//!
//! // Validate the data
//! let config = ValidationConfig::default()
//!     .with_strategy(ValidationStrategy::Optimized)
//!     .with_inference_enabled(true);
//!
//! let report = validator.validate_turtle(data_ttl, config)?;
//!
//! // Check results
//! println!("Validation conforms: {}", report.conforms());
//! for violation in report.violations() {
//!     println!("Violation: {} at {}",
//!         violation.result_message.as_deref().unwrap_or("No message"),
//!         violation.focus_node
//!     );
//! }
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced Usage
//!
//! ### Custom Constraint Components
//!
//! ```rust
//! use oxirs_shacl::{ShaclValidator, CustomConstraintComponent, ValidationConfig};
//! use oxirs_core::model::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut validator = ShaclValidator::new();
//!
//! // Register a custom constraint component
//! let email_constraint = CustomConstraintComponent::new(
//!     "http://example.org/EmailConstraint",
//!     vec!["http://example.org/emailPattern"],
//!     |focus_node, params, data_graph| {
//!         // Custom validation logic for email format
//!         if let Term::Literal(lit) = focus_node {
//!             let email_regex = regex::Regex::new(r"^[^@]+@[^@]+\.[^@]+$").unwrap();
//!             if email_regex.is_match(&lit.value) {
//!                 Ok(vec![]) // No violations
//!             } else {
//!                 Ok(vec![
//!                     // Return validation violation
//!                 ])
//!             }
//!         } else {
//!             Ok(vec![]) // Non-literals are ignored
//!         }
//!     }
//! );
//!
//! validator.register_custom_component(email_constraint)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Parallel Validation
//!
//! ```rust
//! use oxirs_shacl::{ShaclValidator, ValidationConfig, ValidationStrategy};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut validator = ShaclValidator::new();
//!
//! // Configure for parallel validation of large datasets
//! let config = ValidationConfig::default()
//!     .with_strategy(ValidationStrategy::Parallel)
//!     .with_batch_size(1000)
//!     .with_thread_count(8)
//!     .with_memory_limit_mb(2048);
//!
//! // Validate large RDF dataset
//! // let report = validator.validate_large_dataset(&large_store, config)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Incremental Validation
//!
//! ```rust
//! use oxirs_shacl::{ShaclValidator, ValidationConfig, ValidationStrategy};
//! use oxirs_core::model::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut validator = ShaclValidator::new();
//!
//! // Enable incremental validation for real-time updates
//! let config = ValidationConfig::default()
//!     .with_strategy(ValidationStrategy::Incremental)
//!     .with_change_tracking(true);
//!
//! // Initial validation
//! // let initial_report = validator.validate_store(&store, config)?;
//!
//! // Add new data and validate only affected parts
//! let new_triple = Triple::new(
//!     NamedNode::new("http://example.org/newPerson")?,
//!     NamedNode::new("http://example.org/name")?,
//!     Literal::new_simple_literal("New Person")
//! );
//!
//! // Incremental validation of just the new data
//! // let incremental_report = validator.validate_incremental_change(
//! //     &store,
//! //     vec![new_triple],
//! //     vec![], // no deletions
//! //     config
//! // )?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Advanced Enterprise Features
//!
//! #### Streaming Validation for Large Datasets
//!
//! ```rust
//! use oxirs_shacl::{ShaclValidator, ValidationConfig, ValidationStrategy, ReportFormat};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut validator = ShaclValidator::new();
//!
//! // Configure for large dataset processing
//! let config = ValidationConfig::default()
//!     .with_strategy(ValidationStrategy::Streaming)
//!     .with_batch_size(10000)
//!     .with_memory_limit_mb(4096)
//!     .with_progress_reporting(true);
//!
//! // Process large RDF dataset with progress tracking
//! // let report = validator.validate_large_dataset(&large_store, config)?;
//!
//! // Export results in different formats
//! // let json_report = report.export_as(ReportFormat::Json)?;
//! // let html_report = report.export_as(ReportFormat::Html)?;
//! // let csv_report = report.export_as(ReportFormat::Csv)?;
//! # Ok(())
//! # }
//! ```
//!
//! #### Performance Analytics and Monitoring
//!
//! ```rust
//! use oxirs_shacl::{ShaclValidator, ValidationConfig, PerformanceAnalytics};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut validator = ShaclValidator::new();
//! let config = ValidationConfig::default()
//!     .with_performance_monitoring(true)
//!     .with_analytics_enabled(true);
//!
//! // Validate with performance tracking
//! // let report = validator.validate_store(&store, config)?;
//!
//! // Get detailed performance analytics
//! // let analytics = validator.get_performance_analytics()?;
//! // println!("Average validation time: {}ms", analytics.average_validation_time());
//! // println!("Memory usage: {}MB", analytics.peak_memory_usage_mb());
//! // println!("Bottlenecks: {:?}", analytics.identify_bottlenecks());
//! # Ok(())
//! # }
//! ```
//!
//! #### Builder Pattern for Complex Configurations
//!
//! ```rust
//! use oxirs_shacl::{ValidatorBuilder, EnhancedValidatorBuilder, ValidationStrategy};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Enhanced validator with all enterprise features
//! let validator = EnhancedValidatorBuilder::new()
//!     .with_strategy(ValidationStrategy::Parallel)
//!     .with_batch_size(5000)
//!     .with_thread_count(8)
//!     .with_memory_limit_mb(8192)
//!     .with_caching_enabled(true)
//!     .with_cache_size_mb(1024)
//!     .with_performance_monitoring(true)
//!     .with_error_recovery(true)
//!     .with_timeout_seconds(300)
//!     .build()?;
//!
//! // Configure shape loading with advanced options
//! let shape_config = ShapeLoaderBuilder::new()
//!     .with_format_detection(true)
//!     .with_async_loading(true)
//!     .with_validation_on_load(true)
//!     .build();
//!
//! // Load shapes from multiple sources
//! // validator.load_shapes_with_config(&shape_config).await?;
//! # Ok(())
//! # }
//! ```
//!
//! #### Advanced Report Generation
//!
//! ```rust
//! use oxirs_shacl::{ShaclValidator, ReportBuilder, ReportFormat, FilterCriteria};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let validator = ShaclValidator::new();
//! // let report = validator.validate_store(&store, config)?;
//!
//! // Generate customized reports with filtering
//! let filtered_report = ReportBuilder::new()
//!     .filter_by_severity(&[Severity::Violation, Severity::Warning])
//!     .filter_by_shape_pattern("http://example.org/shapes/*")
//!     .include_metadata(true)
//!     .include_statistics(true)
//!     .format(ReportFormat::Html)
//!     .build();
//!
//! // Export with custom formatting
//! // let html_output = filtered_report.generate(&report)?;
//! // let json_export = report.export_as(ReportFormat::Json)?;
//! // let prometheus_metrics = report.export_as(ReportFormat::Prometheus)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Enterprise Integration Patterns
//!
//! #### External System Integration
//!
//! ```rust
//! use oxirs_shacl::{ShaclValidator, ValidationConfig, IntegrationBuilder};
//! use std::time::Duration;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut validator = ShaclValidator::new();
//!
//! // Configure integration with external validation services
//! let integration_config = IntegrationBuilder::new()
//!     .with_external_validator("http://validation-service.example.org/api")
//!     .with_timeout(Duration::from_secs(30))
//!     .with_fallback_strategy(FallbackStrategy::LocalValidation)
//!     .with_circuit_breaker(CircuitBreakerConfig::default())
//!     .with_retry_policy(RetryPolicy::exponential_backoff(3))
//!     .build();
//!
//! let config = ValidationConfig::default()
//!     .with_external_integration(integration_config)
//!     .with_hybrid_validation(true);
//!
//! // Validate with external service integration
//! // let report = validator.validate_with_external_services(&store, config).await?;
//! # Ok(())
//! # }
//! ```
//!
//! #### Federated Validation Across Multiple Graphs
//!
//! ```rust
//! use oxirs_shacl::{FederatedValidator, GraphEndpoint, ValidationStrategy};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let federated_validator = FederatedValidator::new();
//!
//! // Configure multiple graph endpoints
//! let endpoints = vec![
//!     GraphEndpoint::new("http://graph1.example.org/sparql")
//!         .with_authentication("bearer", "token1")
//!         .with_timeout(Duration::from_secs(60)),
//!     GraphEndpoint::new("http://graph2.example.org/sparql")
//!         .with_authentication("basic", "user:pass")
//!         .with_load_balancing(true),
//!     GraphEndpoint::new("http://graph3.example.org/sparql")
//!         .with_failover_priority(1),
//! ];
//!
//! let config = FederatedValidationConfig::new()
//!     .with_endpoints(endpoints)
//!     .with_strategy(ValidationStrategy::DistributedParallel)
//!     .with_result_aggregation(AggregationStrategy::Union)
//!     .with_conflict_resolution(ConflictResolution::MajorityVote);
//!
//! // Validate across federated knowledge graphs
//! // let federated_report = federated_validator
//! //     .validate_federated_graphs(&shapes, config).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Security and Compliance
//!
//! #### Enterprise Security Configuration
//!
//! ```rust
//! use oxirs_shacl::{ShaclValidator, SecurityConfig, EncryptionConfig, AuditConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let security_config = SecurityConfig::new()
//!     .with_encryption(EncryptionConfig::Aes256Gcm {
//!         key_rotation_interval: Duration::from_days(90),
//!         key_derivation: KeyDerivation::Pbkdf2,
//!     })
//!     .with_access_control(AccessControlConfig::RoleBased {
//!         roles: vec!["validator", "admin", "readonly"],
//!         permissions: HashMap::from([
//!             ("validator", vec!["validate", "read"]),
//!             ("admin", vec!["validate", "read", "write", "configure"]),
//!             ("readonly", vec!["read"]),
//!         ]),
//!     })
//!     .with_audit_logging(AuditConfig::Comprehensive {
//!         log_all_operations: true,
//!         include_data_samples: false, // For GDPR compliance
//!         retention_period: Duration::from_days(2555), // 7 years
//!         encryption_at_rest: true,
//!     });
//!
//! let validator = ShaclValidator::with_security_config(security_config)?;
//! # Ok(())
//! # }
//! ```
//!
//! #### Data Privacy and GDPR Compliance
//!
//! ```rust
//! use oxirs_shacl::{ShaclValidator, PrivacyConfig, DataClassification};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let privacy_config = PrivacyConfig::new()
//!     .with_data_classification(DataClassification::Automatic)
//!     .with_pii_detection(true)
//!     .with_anonymization_on_export(true)
//!     .with_right_to_be_forgotten(true)
//!     .with_consent_tracking(true)
//!     .with_lawful_basis_validation(true);
//!
//! let validator = ShaclValidator::new()
//!     .with_privacy_compliance(privacy_config);
//!
//! // Shapes can specify privacy requirements
//! let privacy_aware_shapes = r#"
//! @prefix sh: <http://www.w3.org/ns/shacl#> .
//! @prefix privacy: <http://example.org/privacy#> .
//! @prefix ex: <http://example.org/> .
//!
//! ex:PersonShape
//!     a sh:NodeShape ;
//!     sh:targetClass ex:Person ;
//!     sh:property [
//!         sh:path ex:email ;
//!         privacy:classification privacy:PII ;
//!         privacy:processingBasis privacy:Consent ;
//!         privacy:retentionPeriod "P5Y" ;
//!     ] .
//! "#;
//!
//! // Validation includes privacy compliance checking
//! // let report = validator.validate_with_privacy_check(data, privacy_aware_shapes)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Production Deployment Patterns
//!
//! #### Cloud-Native Kubernetes Deployment
//!
//! ```rust
//! use oxirs_shacl::{ShaclValidator, CloudConfig, KubernetesConfig, ScalingConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let cloud_config = CloudConfig::kubernetes()
//!     .with_namespace("validation-service")
//!     .with_replica_count(3)
//!     .with_scaling(ScalingConfig::HorizontalPodAutoscaler {
//!         min_replicas: 2,
//!         max_replicas: 10,
//!         cpu_threshold: 70,
//!         memory_threshold: 80,
//!     })
//!     .with_resource_limits(ResourceLimits {
//!         cpu: "2000m",
//!         memory: "4Gi",
//!         storage: "20Gi",
//!     })
//!     .with_health_checks(HealthCheckConfig {
//!         liveness_probe: "/health/live",
//!         readiness_probe: "/health/ready",
//!         startup_probe: "/health/startup",
//!     })
//!     .with_service_mesh(ServiceMeshConfig::Istio {
//!         circuit_breaker: true,
//!         rate_limiting: true,
//!         observability: true,
//!     });
//!
//! let validator = ShaclValidator::with_cloud_config(cloud_config)?;
//! # Ok(())
//! # }
//! ```
//!
//! #### Error Recovery and Resilience Patterns
//!
//! ```rust
//! use oxirs_shacl::{ShaclValidator, ResilienceConfig, ErrorRecoveryStrategy};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let resilience_config = ResilienceConfig::new()
//!     .with_circuit_breaker(CircuitBreakerConfig {
//!         failure_threshold: 5,
//!         timeout: Duration::from_secs(60),
//!         half_open_max_calls: 3,
//!     })
//!     .with_bulkhead(BulkheadConfig {
//!         max_concurrent_calls: 100,
//!         max_wait_duration: Duration::from_secs(30),
//!     })
//!     .with_timeout(TimeoutConfig {
//!         default_timeout: Duration::from_secs(120),
//!         per_operation_timeouts: HashMap::from([
//!             ("validation", Duration::from_secs(300)),
//!             ("shape_loading", Duration::from_secs(60)),
//!         ]),
//!     })
//!     .with_retry(RetryConfig {
//!         max_attempts: 3,
//!         backoff_strategy: BackoffStrategy::ExponentialWithJitter {
//!             initial_delay: Duration::from_millis(100),
//!             max_delay: Duration::from_secs(30),
//!             multiplier: 2.0,
//!         },
//!         retryable_errors: vec![
//!             ErrorKind::NetworkTimeout,
//!             ErrorKind::ServiceUnavailable,
//!             ErrorKind::RateLimitExceeded,
//!         ],
//!     });
//!
//! let validator = ShaclValidator::with_resilience_config(resilience_config)?;
//!
//! // Validation with comprehensive error recovery
//! // match validator.validate_with_recovery(&store, &shapes, config).await {
//! //     Ok(report) => println!("Validation completed: {}", report.conforms()),
//! //     Err(e) if e.is_recoverable() => {
//! //         // Automatic retry with backoff
//! //         let recovery_report = validator.attempt_recovery(&e).await?;
//! //         println!("Recovered from error: {:?}", recovery_report);
//! //     }
//! //     Err(e) => return Err(e.into()),
//! // }
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

use indexmap::IndexMap;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use oxirs_core::OxirsError;

pub use crate::optimization::integration::ValidationStrategy;

pub mod analytics;
pub mod builders;
pub mod constraints;
pub mod custom_components;
pub mod federated_validation;
pub mod iri_resolver;
pub mod optimization;
pub mod paths;
pub mod report;
pub mod scirs_graph_integration;
pub mod security;
pub mod shape_import;
pub mod shape_inheritance;
pub mod shape_versioning;
pub mod shapes;
pub mod sparql;
pub mod targets;
pub mod validation;
pub mod vocabulary;
pub mod w3c_test_suite;

// Re-export key types for convenience - avoiding ambiguous glob re-exports
pub use analytics::ValidationAnalytics;
pub use builders::*;
pub use constraints::{Constraint, ConstraintContext, ConstraintEvaluationResult};
pub use custom_components::{
    ComponentMetadata, CustomConstraint, CustomConstraintRegistry, EmailValidationComponent,
    RangeConstraintComponent, RegexConstraintComponent,
};
pub use federated_validation::*;
pub use iri_resolver::*;
pub use optimization::{
    NegationOptimizer, OptimizationConfig, OptimizationResult, OptimizationStrategy,
    ValidationOptimizationEngine,
};
pub use paths::*;
pub use report::{
    ReportFormat, ReportGenerator, ReportMetadata, ValidationReport, ValidationSummary,
};
pub use scirs_graph_integration::{
    BasicMetrics, ConnectivityAnalysis, GraphValidationConfig, GraphValidationResult,
    SciRS2GraphValidator,
};
pub use security::{SecureSparqlExecutor, SecurityConfig, SecurityPolicy};
pub use shape_import::*;
pub use shape_inheritance::*;
// Import specific types from shapes to avoid conflicts
pub use shapes::{
    format_literal_for_sparql, format_term_for_sparql, ShapeCacheStats, ShapeFactory, ShapeParser,
    ShapeParsingConfig, ShapeParsingContext, ShapeParsingStats, ShapeValidationReport,
    ShapeValidator, SingleShapeValidationReport,
};
pub use sparql::*;
// Import specific types from targets to avoid conflicts
pub use targets::{
    Target, TargetCacheStats, TargetOptimizationConfig, TargetSelectionStats, TargetSelector,
};
pub use validation::{ValidationEngine, ValidationViolation};
pub use w3c_test_suite::*;

// Re-export optimization types (note: these are already imported above)

/// SHACL namespace IRI
pub static SHACL_NS: &str = "http://www.w3.org/ns/shacl#";

/// SHACL vocabulary terms
pub static SHACL_VOCAB: Lazy<vocabulary::ShaclVocabulary> =
    Lazy::new(vocabulary::ShaclVocabulary::new);

/// IRI resolver for validation and expansion
pub use iri_resolver::IriResolver;

/// Core error type for SHACL operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum ShaclError {
    #[error("Shape parsing error: {0}")]
    ShapeParsing(String),

    #[error("Constraint validation error: {0}")]
    ConstraintValidation(String),

    #[error("Target selection error: {0}")]
    TargetSelection(String),

    #[error("Property path error: {0}")]
    PropertyPath(String),

    #[error("Path evaluation error: {0}")]
    PathEvaluationError(String),

    #[error("SPARQL execution error: {0}")]
    SparqlExecution(String),

    #[error("Validation engine error: {0}")]
    ValidationEngine(String),

    #[error("Report generation error: {0}")]
    ReportGeneration(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("OxiRS core error: {0}")]
    Core(#[from] OxirsError),

    #[error("IO error: {0}")]
    Io(String),

    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),

    #[error("JSON error: {0}")]
    Json(String),

    #[error("IRI resolution error: {0}")]
    IriResolution(#[from] crate::iri_resolver::IriResolutionError),

    #[error("Security violation: {0}")]
    SecurityViolation(String),

    #[error("Shape validation error: {0}")]
    ShapeValidation(String),

    #[error("Validation timeout: {0}")]
    Timeout(String),

    #[error("Memory limit exceeded: {0}")]
    MemoryLimit(String),

    #[error("Recursion limit exceeded: {0}")]
    RecursionLimit(String),

    #[error("Memory pool error: {0}")]
    MemoryPool(String),

    #[error("Memory optimization error: {0}")]
    MemoryOptimization(String),

    #[error("Async operation error: {0}")]
    AsyncOperation(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Report error: {0}")]
    ReportError(String),
}

impl From<serde_json::Error> for ShaclError {
    fn from(err: serde_json::Error) -> Self {
        ShaclError::Json(err.to_string())
    }
}

impl From<std::io::Error> for ShaclError {
    fn from(err: std::io::Error) -> Self {
        ShaclError::Io(err.to_string())
    }
}

impl From<serde_yaml::Error> for ShaclError {
    fn from(err: serde_yaml::Error) -> Self {
        ShaclError::Json(err.to_string())
    }
}

impl From<anyhow::Error> for ShaclError {
    fn from(err: anyhow::Error) -> Self {
        ShaclError::ValidationEngine(err.to_string())
    }
}

impl From<std::fmt::Error> for ShaclError {
    fn from(err: std::fmt::Error) -> Self {
        ShaclError::ReportGeneration(err.to_string())
    }
}

/// Result type alias for SHACL operations
pub type Result<T> = std::result::Result<T, ShaclError>;

/// SHACL shape identifier
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ShapeId(pub String);

impl ShapeId {
    pub fn new(id: impl Into<String>) -> Self {
        ShapeId(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn generate() -> Self {
        ShapeId(format!("shape_{}", Uuid::new_v4()))
    }
}

impl fmt::Display for ShapeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for ShapeId {
    fn from(s: String) -> Self {
        ShapeId(s)
    }
}

impl From<&str> for ShapeId {
    fn from(s: &str) -> Self {
        ShapeId(s.to_string())
    }
}

/// SHACL constraint component identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ConstraintComponentId(pub String);

impl ConstraintComponentId {
    pub fn new(id: impl Into<String>) -> Self {
        ConstraintComponentId(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ConstraintComponentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for ConstraintComponentId {
    fn from(s: String) -> Self {
        ConstraintComponentId(s)
    }
}

impl From<&str> for ConstraintComponentId {
    fn from(s: &str) -> Self {
        ConstraintComponentId(s.to_string())
    }
}

/// SHACL shape type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeType {
    NodeShape,
    PropertyShape,
}

/// SHACL shape representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shape {
    /// Unique identifier for this shape
    pub id: ShapeId,

    /// Type of this shape (node or property)
    pub shape_type: ShapeType,

    /// Target definitions for this shape
    pub targets: Vec<Target>,

    /// Property path (for property shapes)
    pub path: Option<PropertyPath>,

    /// Constraints applied by this shape
    pub constraints: IndexMap<ConstraintComponentId, Constraint>,

    /// Whether this shape is deactivated
    pub deactivated: bool,

    /// Human-readable label
    pub label: Option<String>,

    /// Human-readable description
    pub description: Option<String>,

    /// Groups this shape belongs to
    pub groups: Vec<String>,

    /// Order for evaluation
    pub order: Option<i32>,

    /// Default severity for violations
    pub severity: Severity,

    /// Custom messages for this shape
    pub messages: IndexMap<String, String>, // language -> message

    /// --- Enhanced features ---

    /// Parent shapes for inheritance (sh:extends)
    pub extends: Vec<ShapeId>,

    /// Priority for conflict resolution (higher value = higher priority)
    pub priority: Option<i32>,

    /// Additional metadata
    pub metadata: ShapeMetadata,
}

impl Shape {
    pub fn new(id: ShapeId, shape_type: ShapeType) -> Self {
        Self {
            id,
            shape_type,
            targets: Vec::new(),
            path: None,
            constraints: IndexMap::new(),
            deactivated: false,
            label: None,
            description: None,
            groups: Vec::new(),
            order: None,
            severity: Severity::Violation,
            messages: IndexMap::new(),
            extends: Vec::new(),
            priority: None,
            metadata: ShapeMetadata::default(),
        }
    }

    pub fn node_shape(id: ShapeId) -> Self {
        Self::new(id, ShapeType::NodeShape)
    }

    pub fn property_shape(id: ShapeId, path: PropertyPath) -> Self {
        let mut shape = Self::new(id, ShapeType::PropertyShape);
        shape.path = Some(path);
        shape
    }

    pub fn add_constraint(&mut self, component_id: ConstraintComponentId, constraint: Constraint) {
        self.constraints.insert(component_id, constraint);
    }

    pub fn add_target(&mut self, target: Target) {
        self.targets.push(target);
    }

    pub fn is_active(&self) -> bool {
        !self.deactivated
    }

    pub fn is_node_shape(&self) -> bool {
        matches!(self.shape_type, ShapeType::NodeShape)
    }

    pub fn is_property_shape(&self) -> bool {
        matches!(self.shape_type, ShapeType::PropertyShape)
    }

    /// Set shape inheritance
    pub fn extends(&mut self, parent_shape_id: ShapeId) -> &mut Self {
        self.extends.push(parent_shape_id);
        self
    }

    /// Set shape priority
    pub fn with_priority(&mut self, priority: i32) -> &mut Self {
        self.priority = Some(priority);
        self
    }

    /// Set shape metadata
    pub fn with_metadata(&mut self, metadata: ShapeMetadata) -> &mut Self {
        self.metadata = metadata;
        self
    }

    /// Update metadata fields
    pub fn update_metadata<F>(&mut self, updater: F) -> &mut Self
    where
        F: FnOnce(&mut ShapeMetadata),
    {
        updater(&mut self.metadata);
        self
    }

    /// Get effective priority (defaults to 0 if not set)
    pub fn effective_priority(&self) -> i32 {
        self.priority.unwrap_or(0)
    }

    /// Check if this shape extends another shape
    pub fn extends_shape(&self, shape_id: &ShapeId) -> bool {
        self.extends.contains(shape_id)
    }

    /// Get all parent shape IDs
    pub fn parent_shapes(&self) -> &[ShapeId] {
        &self.extends
    }
}

impl Default for Shape {
    fn default() -> Self {
        Self::new(ShapeId("default:shape".to_string()), ShapeType::NodeShape)
    }
}

/// Shape metadata for tracking additional information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ShapeMetadata {
    /// Author of the shape
    pub author: Option<String>,

    /// Creation timestamp
    pub created: Option<chrono::DateTime<chrono::Utc>>,

    /// Last modification timestamp
    pub modified: Option<chrono::DateTime<chrono::Utc>>,

    /// Version string
    pub version: Option<String>,

    /// License information
    pub license: Option<String>,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// Custom properties
    pub custom: HashMap<String, String>,
}

/// Violation severity levels
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub enum Severity {
    Info,
    Warning,
    #[default]
    Violation,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Info => write!(f, "Info"),
            Severity::Warning => write!(f, "Warning"),
            Severity::Violation => write!(f, "Violation"),
        }
    }
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Maximum number of violations to report (0 = unlimited)
    pub max_violations: usize,

    /// Include violations with Info severity
    pub include_info: bool,

    /// Include violations with Warning severity
    pub include_warnings: bool,

    /// Stop validation on first violation
    pub fail_fast: bool,

    /// Maximum recursion depth for shape validation
    pub max_recursion_depth: usize,

    /// Timeout for validation in milliseconds
    pub timeout_ms: Option<u64>,

    /// Enable parallel validation
    pub parallel: bool,

    /// Custom validation context
    pub context: HashMap<String, String>,

    /// Validation strategy
    pub strategy: ValidationStrategy,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_violations: 0,
            include_info: true,
            include_warnings: true,
            fail_fast: false,
            max_recursion_depth: 50,
            timeout_ms: None,
            parallel: false,
            context: HashMap::new(),
            strategy: ValidationStrategy::default(),
        }
    }
}

impl ValidationConfig {
    /// Set the validation strategy
    pub fn with_strategy(mut self, strategy: ValidationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enable inference during validation
    pub fn with_inference_enabled(mut self, enabled: bool) -> Self {
        self.context
            .insert("inference_enabled".to_string(), enabled.to_string());
        self
    }
}


/// OxiRS SHACL version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
// Validator module
pub mod validator;

// Re-export validator types
pub use validator::{Validator, ValidationStats, ValidatorBuilder};


/// Initialize OxiRS SHACL with default configuration
pub fn init() -> Result<()> {
    tracing::info!("Initializing OxiRS SHACL v{}", VERSION);
    Ok(())
}
