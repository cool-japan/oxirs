//! # OxiRS SHACL - RDF Validation Engine
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0--alpha.2-orange)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-shacl/badge.svg)](https://docs.rs/oxirs-shacl)
//!
//! **Status**: Alpha Release (v0.1.0-alpha.3)
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
//! use oxirs_shacl::{ValidationConfig, ValidationStrategy};
//! use oxirs_core::{Store, model::*};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a validation configuration
//! let config = ValidationConfig::default()
//!     .with_strategy(ValidationStrategy::Optimized)
//!     .with_inference_enabled(true);
//!
//! // Validation is typically performed using ValidationEngine
//! // See examples/ directory for complete working examples
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced Features
//!
//! For detailed examples and advanced usage patterns, including:
//! - Custom constraint components
//! - Parallel and incremental validation
//! - Enterprise security and compliance features
//! - Federated validation
//! - Production deployment patterns
//!
//! Please refer to the individual module documentation and the examples
//! directory in the repository.

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
pub mod w3c_test_suite_enhanced;

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
pub use validator::{ValidationStats, Validator, ValidatorBuilder};

/// Initialize OxiRS SHACL with default configuration
pub fn init() -> Result<()> {
    tracing::info!("Initializing OxiRS SHACL v{}", VERSION);
    Ok(())
}
