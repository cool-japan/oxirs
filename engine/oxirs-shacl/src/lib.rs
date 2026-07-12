//! # OxiRS SHACL - RDF Validation Engine
//!
//! [![Version](https://img.shields.io/badge/version-0.3.3-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-shacl/badge.svg)](https://docs.rs/oxirs-shacl)
//!
//! **Status**: Production Release (v0.3.3)
//! **Stability**: Public APIs are stable. Production-ready with comprehensive testing.
//!
//! SHACL (Shapes Constraint Language) validation engine for RDF data.
//! Provides comprehensive constraint validation with SHACL Core and SHACL-SPARQL support.
//!
//! ## Features
//!
//! - **SHACL Core** - Complete SHACL Core constraint validation (27/27 W3C constraint types)
//! - **SHACL-SPARQL** - SPARQL-based constraints (experimental)
//! - **SHACL-AF** - Advanced Features (SPARQL targets, ASK validators, target types)
//! - **Property Paths** - Full property path evaluation including inverse, sequence, alternative,
//!   zero-or-more, one-or-more, zero-or-one operators
//! - **Logical Constraints** - `sh:and`, `sh:or`, `sh:not`, `sh:xone`
//! - **Validation Reports** - W3C-compliant violation reports with metadata
//! - **Performance** - Optimized validation engine with caching, parallelism, and incremental modes
//!
//! ## Companion Documentation
//!
//! In addition to this rustdoc reference, two companion documents live at the crate root:
//!
//! - `COOKBOOK.md` — task-oriented patterns (cardinality, string, datatype, qualified value,
//!   target chains, SHACL-AF SPARQL constraints) with Turtle examples and Rust API snippets.
//! - `SPEC_MAPPING.md` — exhaustive table mapping every SHACL Core / SHACL-AF construct to the
//!   Rust symbol that implements it.
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
//! ## Crate Layout
//!
//! - [`constraints`] — every SHACL Core constraint component
//!   ([`MinCountConstraint`](constraints::cardinality_constraints::MinCountConstraint),
//!   [`PatternConstraint`](constraints::string_constraints::PatternConstraint),
//!   [`NodeKindConstraint`](constraints::value_constraints::NodeKindConstraint), and so on).
//! - [`paths`] — property path evaluation and traversal.
//! - [`shapes`] — shape parsing, factories, and shape-level validation.
//! - [`targets`] — target selectors (`sh:targetClass`, `sh:targetSubjectsOf`, …).
//! - [`validation`] — the validation engine and reporting pipeline.
//! - [`sparql_af`] — SHACL Advanced Features built on SPARQL.
//! - [`report`] — W3C validation report generation in multiple formats.
//! - [`optimization`] — query rewriting and execution-plan optimisation.
//! - [`cache`] — validation result and parallel-validation caches.
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

pub mod advanced_features;
pub mod analytics;
pub mod builders;
pub mod cache;
pub mod constraints;
pub mod custom_components;
pub mod designer;
pub mod federated_validation;
pub mod incremental;
pub mod integration;
pub mod iri_resolver;
#[cfg(feature = "lsp")]
pub mod lsp;
pub mod node_expr_builtins;
pub mod node_expr_evaluator;
#[cfg(test)]
mod node_expr_tests;
pub mod node_expr_types;
pub mod node_expressions;
pub mod optimization;
pub mod paths;
pub mod report;
pub mod schema_import;
pub mod scirs_graph_integration;
pub mod security;
pub mod shaclc_parser;
pub mod shape_import;
pub mod shape_inheritance;
pub mod shape_map;
pub mod shape_versioning;
pub mod shapes;
pub mod sparql;
pub mod sparql_af;
pub mod targets;
pub mod templates;
pub mod testing;
pub mod validation;
pub mod visual_editor;
pub mod vocabulary;
pub mod w3c_test_suite;
pub mod w3c_test_suite_enhanced;

// Re-export key types for convenience - avoiding ambiguous glob re-exports
pub use advanced_features::{
    AdvancedTarget, AdvancedTargetSelector, ConditionalConstraint, ConditionalEvaluator,
    ConditionalResult, FunctionInvocation, FunctionParameter, FunctionRegistry, FunctionResult,
    InferenceStrategy, InferredShape, ParameterType, ReturnType, RuleEngine, RuleEngineStats,
    RuleExecutionResult, ShaclFunction, ShaclRule, ShapeInferenceConfig, ShapeInferenceEngine,
    ShapeRegistry,
};
pub use analytics::ValidationAnalytics;
pub use builders::*;
pub use cache::{
    CacheStats, CachedValidationResult, ParallelConstraintConfig, ParallelConstraintOutcome,
    ParallelConstraintValidator, ParallelValidationStats, ParallelValidationSummary, TripleKey,
    ValidationCache, ValidationCacheKey,
};
pub use constraints::{
    AdvancedSparqlConstraint, AlwaysViolatingEvaluator, Constraint, ConstraintContext,
    ConstraintEvaluationResult, ExpressionConstraintComponent, ExpressionConstraintResult,
    ExpressionContext, ExpressionEvaluator, FailingEvaluator, MockSparqlEvaluator,
    NodeKindConstraint, PathResolver, PropertyConstraint, ShaclExpression, ShaclValue,
    SparqlConstraintResult, SparqlConstraintSeverity, SparqlEvaluator,
};
pub use custom_components::{
    ComponentMetadata, CustomConstraint, CustomConstraintRegistry, EmailValidationComponent,
    RangeConstraintComponent, RegexConstraintComponent,
};
pub use federated_validation::*;
pub use incremental::{
    Changeset, GraphChange, IncrementalConfig, IncrementalStats, IncrementalValidator,
};
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
pub use testing::{
    ShapeTestSuite, TestAssertions, TestCase, TestExpectation, TestResult, TestStatus,
    TestSuiteResult, TestSummary,
};
pub use validation::{ValidationEngine, ValidationViolation};
pub use visual_editor::{
    ColorScheme, ExportFormat, LayoutDirection, ShapeVisualizer, VisualizerConfig,
};
pub use w3c_test_suite::*;

// Re-export SHACL-AF SPARQL target types
pub use sparql_af::{
    ask_validator::{
        FailingAskExecutor, MockAskExecutor, SparqlAskExecutor, SparqlAskResult,
        SparqlAskValidator, SparqlAskValidatorBuilder, SparqlAskViolation,
    },
    sparql_target::{
        ParameterBinding, SparqlAfTarget, SparqlAfTargetResult, SparqlTargetEvaluator,
        SparqlTargetMock,
    },
    target_type::{
        SparqlTargetParameter, SparqlTargetType, SparqlTargetTypeInstance, SparqlTargetTypeRegistry,
    },
    PrefixMap, SubstitutionContext,
};

// Re-export designer types
pub use designer::{
    ConstraintSpec, DesignIssue, DesignStep, DesignWizard, Domain, PropertyDesign, PropertyHint,
    RecommendationEngine, ShapeDesign, ShapeDesigner,
    ShapeInferenceEngine as DesignerInferenceEngine,
};

// Re-export optimization types (note: these are already imported above)

/// SHACL namespace IRI
pub static SHACL_NS: &str = "http://www.w3.org/ns/shacl#";

/// SHACL vocabulary terms
pub static SHACL_VOCAB: Lazy<vocabulary::ShaclVocabulary> =
    Lazy::new(vocabulary::ShaclVocabulary::new);

/// IRI resolver for validation and expansion
pub use iri_resolver::IriResolver;

/// Core error type for SHACL operations.
///
/// Every fallible API in this crate returns [`Result<T>`](crate::Result), which is a
/// type alias for `std::result::Result<T, ShaclError>`. The variants below cover the full
/// surface area of the validation pipeline — parsing, constraint evaluation, property
/// path traversal, SPARQL execution, reporting, and resource limits.
///
/// All variants derive [`thiserror::Error`] and provide a human-readable `Display`
/// implementation suitable for inclusion in user-facing diagnostics. Errors that wrap
/// foreign types (`oxirs_core::OxirsError`, `regex::Error`, `IriResolutionError`)
/// implement `From` so the `?` operator works without boilerplate.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ShaclError {
    /// A shape document failed to parse (malformed Turtle/RDF, missing required terms,
    /// or invalid SHACL syntax).
    #[error("Shape parsing error: {0}")]
    ShapeParsing(String),

    /// A constraint definition was rejected during structural validation
    /// (e.g. invalid `sh:pattern` regex, contradictory cardinalities).
    #[error("Constraint validation error: {0}")]
    ConstraintValidation(String),

    /// Target selection failed — typically a malformed `sh:target*` declaration
    /// or an inaccessible store.
    #[error("Target selection error: {0}")]
    TargetSelection(String),

    /// A property path expression could not be parsed.
    #[error("Property path error: {0}")]
    PropertyPath(String),

    /// A property path evaluated correctly but produced an unexpected result
    /// during traversal (cycles, infinite paths beyond the recursion limit).
    #[error("Path evaluation error: {0}")]
    PathEvaluationError(String),

    /// A SPARQL query (used by SHACL-SPARQL constraints, SHACL-AF targets,
    /// or the optimizer) failed to execute.
    #[error("SPARQL execution error: {0}")]
    SparqlExecution(String),

    /// The validation engine itself reported an internal error
    /// (orchestration failure, scheduler error, etc.).
    #[error("Validation engine error: {0}")]
    ValidationEngine(String),

    /// Producing a validation report failed (serialisation, IO, or formatter error).
    #[error("Report generation error: {0}")]
    ReportGeneration(String),

    /// A user-supplied `ValidationConfig` or builder argument is invalid.
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Wrapped error originating in the `oxirs-core` RDF data model.
    #[error("OxiRS core error: {0}")]
    Core(#[from] OxirsError),

    /// Wrapped `std::io::Error`.
    #[error("IO error: {0}")]
    Io(String),

    /// A `sh:pattern` regex failed to compile.
    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),

    /// JSON serialisation/deserialisation error
    /// (used by the JSON-LD report writer and YAML config loader).
    #[error("JSON error: {0}")]
    Json(String),

    /// IRI resolution failed (relative IRI without a base, malformed prefix, etc.).
    #[error("IRI resolution error: {0}")]
    IriResolution(#[from] crate::iri_resolver::IriResolutionError),

    /// A SHACL-SPARQL constraint was rejected by the security policy
    /// (forbidden function, query too expensive, network egress denied).
    #[error("Security violation: {0}")]
    SecurityViolation(String),

    /// A higher-level shape contract was violated
    /// (e.g. inheritance cycle, conflicting `sh:property` definitions).
    #[error("Shape validation error: {0}")]
    ShapeValidation(String),

    /// Validation exceeded the configured wall-clock timeout.
    #[error("Validation timeout: {0}")]
    Timeout(String),

    /// Validation exceeded the configured memory budget.
    #[error("Memory limit exceeded: {0}")]
    MemoryLimit(String),

    /// Recursive shape evaluation reached `ValidationConfig::max_recursion_depth`.
    #[error("Recursion limit exceeded: {0}")]
    RecursionLimit(String),

    /// Internal pool used for short-lived allocations failed.
    #[error("Memory pool error: {0}")]
    MemoryPool(String),

    /// Memory-aware optimisation pass aborted.
    #[error("Memory optimization error: {0}")]
    MemoryOptimization(String),

    /// An async runtime task failed (only when the `async` feature is enabled).
    #[error("Async operation error: {0}")]
    AsyncOperation(String),

    /// A construct was recognised but is not yet implemented in this build
    /// (typically gated behind a Cargo feature).
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Generic report-related failure (writer IO, missing template, etc.).
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

/// SHACL shape identifier.
///
/// A shape is identified by its IRI in the shapes graph; for blank-node shapes,
/// implementations must mint a stable identifier. `ShapeId` is the canonical
/// representation used throughout the engine to reference shapes in maps,
/// inheritance chains, and validation reports.
///
/// Shapes are typically named with the `sh:NodeShape` or `sh:PropertyShape` IRI:
///
/// ```rust
/// use oxirs_shacl::ShapeId;
///
/// let person = ShapeId::new("http://example.org/PersonShape");
/// assert_eq!(person.as_str(), "http://example.org/PersonShape");
/// ```
///
/// For blank-node shapes (anonymous shapes inlined in a property shape, etc.),
/// use [`ShapeId::generate`] to mint a fresh UUID-based identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ShapeId(pub String);

impl ShapeId {
    /// Construct a shape identifier from anything convertible into `String`.
    pub fn new(id: impl Into<String>) -> Self {
        ShapeId(id.into())
    }

    /// Return the underlying IRI/string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Generate a fresh, globally unique shape identifier (UUID-backed).
    ///
    /// Used for blank-node shapes and synthetic shapes constructed at runtime.
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

/// SHACL constraint component identifier.
///
/// Every SHACL constraint is associated with a *constraint component* — the
/// IRI that identifies which kind of constraint it is. For example, the
/// `sh:minCount` parameter activates the constraint component whose ID is
/// `sh:MinCountConstraintComponent`. These IDs surface in violation reports
/// (`sh:sourceConstraintComponent`) and let consumers route or filter
/// violations by constraint family.
///
/// ```rust
/// use oxirs_shacl::ConstraintComponentId;
///
/// let id = ConstraintComponentId::new("sh:MinCountConstraintComponent");
/// assert_eq!(id.as_str(), "sh:MinCountConstraintComponent");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ConstraintComponentId(pub String);

impl ConstraintComponentId {
    /// Construct a constraint component ID from anything convertible into `String`.
    pub fn new(id: impl Into<String>) -> Self {
        ConstraintComponentId(id.into())
    }

    /// Return the underlying IRI/string slice.
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

/// SHACL shape type.
///
/// SHACL distinguishes two shape kinds (SHACL Core §2.1):
///
/// - **Node shape** (`sh:NodeShape`) — places constraints on focus nodes themselves.
/// - **Property shape** (`sh:PropertyShape`) — declares a `sh:path` and places
///   constraints on the values reached via that path from a focus node.
///
/// The variant of this enum determines how a [`Shape`] is interpreted by the
/// validation engine: property shapes always have a `path`, node shapes do not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeType {
    /// `sh:NodeShape` — constraints applied directly to focus nodes.
    NodeShape,
    /// `sh:PropertyShape` — constraints applied to values reachable from focus nodes
    /// through the shape's `sh:path`.
    PropertyShape,
}

/// SHACL shape representation.
///
/// A `Shape` is the in-memory model of a single SHACL shape — either a node shape
/// (`sh:NodeShape`) or a property shape (`sh:PropertyShape`). Shapes group together
/// the targets that select focus nodes, the path (for property shapes), the
/// constraints to evaluate, and various metadata used for reporting and
/// inheritance.
///
/// Construct via [`Shape::node_shape`] or [`Shape::property_shape`]:
///
/// ```rust
/// use oxirs_shacl::{Shape, ShapeId};
///
/// let mut person = Shape::node_shape(ShapeId::new("http://example.org/PersonShape"));
/// person.label = Some("Person".into());
/// person.description = Some("A natural person".into());
/// assert!(person.is_node_shape());
/// ```
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

    /// Property shapes linked via sh:property (for NodeShapes)
    pub property_shapes: Vec<ShapeId>,

    /// Priority for conflict resolution (higher value = higher priority)
    pub priority: Option<i32>,

    /// Additional metadata
    pub metadata: ShapeMetadata,
}

impl Shape {
    /// Construct a new shape with the given ID and type, no targets, and no
    /// constraints. Defaults: severity = `Violation`, `deactivated = false`.
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
            property_shapes: Vec::new(),
            priority: None,
            metadata: ShapeMetadata::default(),
        }
    }

    /// Construct an empty `sh:NodeShape`.
    pub fn node_shape(id: ShapeId) -> Self {
        Self::new(id, ShapeType::NodeShape)
    }

    /// Construct an empty `sh:PropertyShape` with the given `sh:path`.
    pub fn property_shape(id: ShapeId, path: PropertyPath) -> Self {
        let mut shape = Self::new(id, ShapeType::PropertyShape);
        shape.path = Some(path);
        shape
    }

    /// Attach a constraint to this shape under the given component ID.
    /// Inserting twice with the same `component_id` overwrites the previous value.
    pub fn add_constraint(&mut self, component_id: ConstraintComponentId, constraint: Constraint) {
        self.constraints.insert(component_id, constraint);
    }

    /// Append a target declaration (`sh:targetClass`, `sh:targetNode`, …) to this shape.
    pub fn add_target(&mut self, target: Target) {
        self.targets.push(target);
    }

    /// Whether this shape participates in validation. A shape with `sh:deactivated true`
    /// is loaded into the shapes graph but skipped by the validation engine.
    pub fn is_active(&self) -> bool {
        !self.deactivated
    }

    /// True when [`shape_type`](Shape::shape_type) is [`ShapeType::NodeShape`].
    pub fn is_node_shape(&self) -> bool {
        matches!(self.shape_type, ShapeType::NodeShape)
    }

    /// True when [`shape_type`](Shape::shape_type) is [`ShapeType::PropertyShape`].
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

/// Violation severity levels (SHACL Core §2.1.4 `sh:severity`).
///
/// SHACL allows shape authors to mark violations as informational, warning, or
/// hard violation. The default is [`Severity::Violation`]. Tools (CI, IDE,
/// linters) typically filter or color-code reports by severity.
///
/// Severity values are totally ordered: `Info < Warning < Violation`. This
/// allows `max(severities)` to compute the worst severity in a report.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub enum Severity {
    /// `sh:Info` — purely informational; not a failure.
    Info,
    /// `sh:Warning` — soft violation; default-on but can be ignored.
    Warning,
    /// `sh:Violation` — hard violation (default for SHACL Core).
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

/// Validation configuration shared by all entry points into the engine.
///
/// `ValidationConfig` controls how the engine behaves during a single validation
/// run: how many violations to emit, which severities to include, whether to
/// stop after the first failure, recursion limits, time budgets, parallelism,
/// and which optimisation strategy to use.
///
/// ```rust
/// use oxirs_shacl::{ValidationConfig, ValidationStrategy, Severity};
///
/// let cfg = ValidationConfig::default()
///     .with_strategy(ValidationStrategy::Optimized)
///     .with_inference_enabled(true);
/// assert!(cfg.include_warnings);
/// assert_eq!(cfg.max_recursion_depth, 50);
/// ```
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
// SHACL node expression evaluator (v1.1.0 round 5)
pub mod node_expression_evaluator;

// SHACL target declaration evaluator (v1.1.0 round 6)
pub mod target_selector;

// SHACL sh:message template interpolation (v1.1.0 round 7)
pub mod message_formatter;

// SHACL SPARQL-based constraint validation (sh:SPARQLConstraintComponent) (v1.1.0 round 8)
pub mod sparql_constraint_validator;

// SHACL property path constraint checking (v1.1.0 round 9)
pub mod property_path_checker;

// SHACL shape graph loader and indexer (v1.1.0 round 10)
pub mod shape_graph_loader;

// SHACL entailment regime support: RDFS + OWL Direct subsets (v1.1.0 round 11)
pub mod entailment_regime;

// Pattern-based SHACL shape matching (v1.1.0 round 12)
pub mod shape_matcher;

// SHACL constraint inheritance via sh:and/sh:or/sh:not/sh:xone (v1.1.0 round 13)
pub mod constraint_inheritance;

// SHACL severity level handling (v1.1.0 round 12)
pub mod severity_handler;

// SHACL focus node selection (v1.1.0 round 11)
pub mod focus_node_selector;

// SHACL property path execution — full path operator evaluation (v1.1.0 round 13)
pub mod path_executor;

// SHACL sh:parameter / parameterized constraint components (v1.1.0 round 14)
pub mod constraint_parameter;

// SHACL sh:datatype constraint checker (v1.1.0 round 15)
pub mod datatype_checker;

// SHACL sh:node constraint component (v1.1.0 round 16)
pub mod node_constraint;

// Re-export validator types
pub use validator::{ValidationStats, Validator, ValidatorBuilder};

/// Initialize OxiRS SHACL with default configuration
pub fn init() -> Result<()> {
    tracing::info!("Initializing OxiRS SHACL v{}", VERSION);
    Ok(())
}
