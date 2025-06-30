//! # OxiRS SHACL
//!
//! SHACL Core + SHACL-SPARQL validator for RDF data validation.
//!
//! This crate provides comprehensive SHACL validation capabilities for RDF data,
//! including both SHACL Core constraints and SHACL-SPARQL extensions.
//!
//! ## Features
//!
//! - Complete SHACL Core constraint validation
//! - SHACL-SPARQL extensions support
//! - Property path evaluation
//! - Logical constraints (sh:and, sh:or, sh:not, sh:xone)
//! - Shape-based constraints and closed shapes
//! - Comprehensive validation reporting
//! - High-performance validation engine
//! - Parallel validation support
//! - Incremental validation capabilities
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

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use indexmap::IndexMap;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use oxirs_core::{
    graph::Graph,
    model::{BlankNode, Literal, NamedNode, Quad, Term, Triple, Variable},
    OxirsError, Store,
};

pub mod analytics;
pub mod builders;
pub mod constraints;
pub mod custom_components;
pub mod federated_validation;
pub mod iri_resolver;
pub mod optimization;
pub mod paths;
pub mod report;
pub mod security;
pub mod shape_import;
pub mod shape_inheritance;
pub mod shapes;
pub mod sparql;
pub mod targets;
pub mod validation;
pub mod vocabulary;

// Re-export key types for convenience
pub use analytics::*;
pub use builders::*;
pub use constraints::*;
pub use custom_components::*;
pub use federated_validation::*;
pub use iri_resolver::*;
pub use optimization::*;
pub use paths::*;
pub use report::*;
pub use security::*;
pub use shape_import::*;
pub use shape_inheritance::*;
pub use shapes::*;
pub use sparql::*;
pub use targets::*;
pub use validation::*;

// Re-export optimization types
pub use targets::{TargetCacheStats, TargetOptimizationConfig, TargetSelectionStats};

/// SHACL namespace IRI
pub static SHACL_NS: &str = "http://www.w3.org/ns/shacl#";

/// SHACL vocabulary terms
pub static SHACL_VOCAB: Lazy<vocabulary::ShaclVocabulary> =
    Lazy::new(vocabulary::ShaclVocabulary::new);

/// IRI resolver for validation and expansion  
pub use iri_resolver::IriResolver;

/// Core error type for SHACL operations
#[derive(Debug, thiserror::Error)]
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
    Io(#[from] std::io::Error),

    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

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
}

/// Result type alias for SHACL operations
pub type Result<T> = std::result::Result<T, ShaclError>;

/// SHACL shape identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Default for ShapeMetadata {
    fn default() -> Self {
        Self {
            author: None,
            created: None,
            modified: None,
            version: None,
            license: None,
            tags: Vec::new(),
            custom: HashMap::new(),
        }
    }
}

/// Violation severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Violation,
}

impl Default for Severity {
    fn default() -> Self {
        Severity::Violation
    }
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
        }
    }
}

/// SHACL validator
#[derive(Debug)]
pub struct Validator {
    /// All loaded shapes indexed by ID
    shapes: IndexMap<ShapeId, Shape>,

    /// Shape dependency graph for optimization
    shape_dependencies: petgraph::Graph<ShapeId, ()>,

    /// Target cache for performance
    target_cache: HashMap<String, Vec<Term>>,

    /// Configuration for validation
    config: ValidationConfig,
}

impl Validator {
    /// Create a new validator with default configuration
    pub fn new() -> Self {
        Self {
            shapes: IndexMap::new(),
            shape_dependencies: petgraph::Graph::new(),
            target_cache: HashMap::new(),
            config: ValidationConfig::default(),
        }
    }

    /// Create a new validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self {
            shapes: IndexMap::new(),
            shape_dependencies: petgraph::Graph::new(),
            target_cache: HashMap::new(),
            config,
        }
    }

    /// Add a shape to the validator
    pub fn add_shape(&mut self, shape: Shape) -> Result<()> {
        let shape_id = shape.id.clone();

        // Validate the shape itself
        self.validate_shape(&shape)?;

        // Add to shape collection
        self.shapes.insert(shape_id.clone(), shape);

        // Update dependency graph
        self.update_shape_dependencies(shape_id)?;

        // Clear target cache as it may be invalidated
        self.target_cache.clear();

        Ok(())
    }

    /// Remove a shape from the validator
    pub fn remove_shape(&mut self, shape_id: &ShapeId) -> Option<Shape> {
        let removed = self.shapes.remove(shape_id);
        if removed.is_some() {
            self.target_cache.clear();

            // Remove from dependency graph
            use petgraph::visit::IntoNodeReferences;
            let mut node_to_remove = None;

            // Find the node index for this shape
            for (node_idx, node_shape_id) in self.shape_dependencies.node_references() {
                if node_shape_id == shape_id {
                    node_to_remove = Some(node_idx);
                    break;
                }
            }

            // Remove the node if found
            if let Some(node_idx) = node_to_remove {
                self.shape_dependencies.remove_node(node_idx);
            }
        }
        removed
    }

    /// Get all shapes
    pub fn shapes(&self) -> &IndexMap<ShapeId, Shape> {
        &self.shapes
    }

    /// Get a specific shape by ID
    pub fn get_shape(&self, shape_id: &ShapeId) -> Option<&Shape> {
        self.shapes.get(shape_id)
    }

    /// Load shapes from an RDF graph in a store
    pub fn load_shapes_from_store(
        &mut self,
        store: &Store,
        graph_name: Option<&str>,
    ) -> Result<usize> {
        let mut parser = shapes::ShapeParser::new();
        let shapes = parser.parse_shapes_from_store(store, graph_name)?;

        let count = shapes.len();
        for shape in shapes {
            self.add_shape(shape)?;
        }

        Ok(count)
    }

    /// Load shapes from RDF data
    pub fn load_shapes_from_rdf(
        &mut self,
        rdf_data: &str,
        format: &str,
        base_iri: Option<&str>,
    ) -> Result<usize> {
        let mut parser = shapes::ShapeParser::new();
        let shapes = parser.parse_shapes_from_rdf(rdf_data, format, base_iri)?;

        let count = shapes.len();
        for shape in shapes {
            self.add_shape(shape)?;
        }

        Ok(count)
    }

    /// Validate data in a store against all loaded shapes
    pub fn validate_store(
        &self,
        store: &Store,
        config: Option<ValidationConfig>,
    ) -> Result<ValidationReport> {
        let config = config.unwrap_or_else(|| self.config.clone());
        let mut engine = validation::ValidationEngine::new(&self.shapes, config);
        engine.validate_store(store)
    }

    /// Validate specific nodes against a specific shape
    pub fn validate_nodes(
        &self,
        store: &Store,
        shape_id: &ShapeId,
        nodes: &[Term],
        config: Option<ValidationConfig>,
    ) -> Result<ValidationReport> {
        let shape = self.get_shape(shape_id).ok_or_else(|| {
            ShaclError::ValidationEngine(format!("Shape not found: {}", shape_id))
        })?;

        let config = config.unwrap_or_else(|| self.config.clone());
        let mut engine = validation::ValidationEngine::new(&self.shapes, config);
        engine.validate_nodes(store, shape, nodes)
    }

    /// Validate a single node against a specific shape
    pub fn validate_node(
        &self,
        store: &Store,
        shape_id: &ShapeId,
        node: &Term,
        config: Option<ValidationConfig>,
    ) -> Result<ValidationReport> {
        self.validate_nodes(store, shape_id, &[node.clone()], config)
    }

    /// Comprehensive shape validation to ensure shapes graphs are themselves valid
    fn validate_shape(&self, shape: &Shape) -> Result<()> {
        // 1. Basic shape type validation
        self.validate_shape_type(shape)?;

        // 2. Validate shape targets
        self.validate_shape_targets(shape)?;

        // 3. Validate property paths
        self.validate_shape_paths(shape)?;

        // 4. Validate individual constraints
        self.validate_shape_constraints(shape)?;

        // 5. Validate constraint combinations
        self.validate_constraint_combinations(shape)?;

        // 6. Validate metadata
        self.validate_shape_metadata(shape)?;

        // 7. Validate inheritance references
        self.validate_inheritance_references(shape)?;

        // 8. Enhanced SHACL specification compliance
        self.validate_shacl_compliance(shape)?;

        // 9. Validate constraint parameter restrictions
        self.validate_constraint_parameters(shape)?;

        // 10. Validate logical constraint consistency
        self.validate_logical_consistency(shape)?;

        Ok(())
    }

    /// Validate shape type requirements
    fn validate_shape_type(&self, shape: &Shape) -> Result<()> {
        match shape.shape_type {
            ShapeType::PropertyShape => {
                if shape.path.is_none() {
                    return Err(ShaclError::ShapeParsing(format!(
                        "Property shape '{}' must have a property path",
                        shape.id
                    )));
                }

                // Property shapes can have targets, though it's more common for node shapes
                // We'll allow all target types but log a warning for unusual combinations
                if !shape.targets.is_empty() {
                    for target in &shape.targets {
                        match target {
                            Target::Class(_) => {
                                tracing::debug!(
                                    "Property shape '{}' has a class target, which is valid but unusual",
                                    shape.id
                                );
                            }
                            _ => {} // All target types are valid for property shapes
                        }
                    }
                }
            }
            ShapeType::NodeShape => {
                if shape.path.is_some() {
                    return Err(ShaclError::ShapeParsing(format!(
                        "Node shape '{}' should not have a property path",
                        shape.id
                    )));
                }
            }
        }
        Ok(())
    }

    /// Validate shape targets
    fn validate_shape_targets(&self, shape: &Shape) -> Result<()> {
        for target in &shape.targets {
            match target {
                Target::Class(class_iri) => {
                    // Validate that the class IRI is a valid IRI
                    self.validate_iri_format(class_iri.as_str())?;
                }
                Target::Node(node_term) => {
                    // Validate that the node term is valid
                    if let Term::NamedNode(node_iri) = node_term {
                        self.validate_iri_format(node_iri.as_str())?;
                    }
                    // Other term types (blank nodes, literals) are also valid targets
                }
                Target::SubjectsOf(property_iri) => {
                    // Validate that the property IRI is a valid IRI
                    self.validate_iri_format(property_iri.as_str())?;
                }
                Target::ObjectsOf(property_iri) => {
                    // Validate that the property IRI is a valid IRI
                    self.validate_iri_format(property_iri.as_str())?;
                }
                Target::Sparql(sparql_target) => {
                    // Validate SPARQL query syntax
                    self.validate_sparql_query(&sparql_target.query)?;
                }
                Target::Implicit(class_iri) => {
                    // Validate that the implicit class IRI is a valid IRI
                    self.validate_iri_format(class_iri.as_str())?;
                }
            }
        }
        Ok(())
    }

    /// Validate property paths
    fn validate_shape_paths(&self, shape: &Shape) -> Result<()> {
        if let Some(path) = &shape.path {
            self.validate_property_path(path)?;
        }
        Ok(())
    }

    /// Validate property path structure
    fn validate_property_path(&self, path: &PropertyPath) -> Result<()> {
        match path {
            PropertyPath::Predicate(iri) => {
                self.validate_iri_format(iri.as_str())?;
            }
            PropertyPath::Inverse(inner_path) => {
                self.validate_property_path(inner_path)?;
            }
            PropertyPath::Sequence(paths) => {
                if paths.is_empty() {
                    return Err(ShaclError::ShapeParsing(
                        "Sequence path cannot be empty".to_string(),
                    ));
                }
                for p in paths {
                    self.validate_property_path(p)?;
                }
            }
            PropertyPath::Alternative(paths) => {
                if paths.len() < 2 {
                    return Err(ShaclError::ShapeParsing(
                        "Alternative path must have at least 2 alternatives".to_string(),
                    ));
                }
                for p in paths {
                    self.validate_property_path(p)?;
                }
            }
            PropertyPath::ZeroOrMore(inner_path)
            | PropertyPath::OneOrMore(inner_path)
            | PropertyPath::ZeroOrOne(inner_path) => {
                self.validate_property_path(inner_path)?;
            }
        }
        Ok(())
    }

    /// Validate shape constraints
    fn validate_shape_constraints(&self, shape: &Shape) -> Result<()> {
        for (component_id, constraint) in &shape.constraints {
            // Validate individual constraint
            constraint.validate().map_err(|e| {
                ShaclError::ShapeParsing(format!(
                    "Invalid constraint '{}' in shape '{}': {}",
                    component_id, shape.id, e
                ))
            })?;
        }
        Ok(())
    }

    /// Validate constraint combinations for logical consistency
    fn validate_constraint_combinations(&self, shape: &Shape) -> Result<()> {
        let constraints = &shape.constraints;

        // Check for conflicting cardinality constraints
        if let (Some(min_count), Some(max_count)) = (
            constraints.get(&ConstraintComponentId::new("minCount")),
            constraints.get(&ConstraintComponentId::new("maxCount")),
        ) {
            if let (Constraint::MinCount(min), Constraint::MaxCount(max)) = (min_count, max_count) {
                if min.min_count > max.max_count {
                    return Err(ShaclError::ShapeParsing(format!(
                        "Shape '{}' has conflicting cardinality: minCount ({}) > maxCount ({})",
                        shape.id, min.min_count, max.max_count
                    )));
                }
            }
        }

        // Check for conflicting string length constraints
        if let (Some(min_length), Some(max_length)) = (
            constraints.get(&ConstraintComponentId::new("minLength")),
            constraints.get(&ConstraintComponentId::new("maxLength")),
        ) {
            if let (Constraint::MinLength(min), Constraint::MaxLength(max)) =
                (min_length, max_length)
            {
                if min.min_length > max.max_length {
                    return Err(ShaclError::ShapeParsing(format!(
                        "Shape '{}' has conflicting string length: minLength ({}) > maxLength ({})",
                        shape.id, min.min_length, max.max_length
                    )));
                }
            }
        }

        // Check for conflicting numeric range constraints
        self.validate_numeric_range_consistency(shape, constraints)?;

        Ok(())
    }

    /// Validate numeric range constraint consistency
    fn validate_numeric_range_consistency(
        &self,
        shape: &Shape,
        constraints: &IndexMap<ConstraintComponentId, Constraint>,
    ) -> Result<()> {
        // Extract numeric bounds (as literals for basic consistency check)
        let mut min_inclusive: Option<&Literal> = None;
        let mut max_inclusive: Option<&Literal> = None;
        let mut min_exclusive: Option<&Literal> = None;
        let mut max_exclusive: Option<&Literal> = None;

        for (_, constraint) in constraints {
            match constraint {
                Constraint::MinInclusive(val) => min_inclusive = Some(&val.min_value),
                Constraint::MaxInclusive(val) => max_inclusive = Some(&val.max_value),
                Constraint::MinExclusive(val) => min_exclusive = Some(&val.min_value),
                Constraint::MaxExclusive(val) => max_exclusive = Some(&val.max_value),
                _ => {}
            }
        }

        // Basic consistency check - try to parse as numbers for simple validation
        if let (Some(min_inc), Some(max_inc)) = (min_inclusive, max_inclusive) {
            if let (Ok(min_val), Ok(max_val)) = (
                min_inc.value().parse::<f64>(),
                max_inc.value().parse::<f64>(),
            ) {
                if min_val > max_val {
                    return Err(ShaclError::ShapeParsing(format!(
                        "Shape '{}' has inconsistent range: minInclusive ({}) > maxInclusive ({})",
                        shape.id, min_inc, max_inc
                    )));
                }
            }
        }

        if let (Some(min_exc), Some(max_exc)) = (min_exclusive, max_exclusive) {
            if let (Ok(min_val), Ok(max_val)) = (
                min_exc.value().parse::<f64>(),
                max_exc.value().parse::<f64>(),
            ) {
                if min_val >= max_val {
                    return Err(ShaclError::ShapeParsing(format!(
                        "Shape '{}' has inconsistent range: minExclusive ({}) >= maxExclusive ({})",
                        shape.id, min_exc, max_exc
                    )));
                }
            }
        }

        Ok(())
    }

    /// Validate shape metadata
    fn validate_shape_metadata(&self, shape: &Shape) -> Result<()> {
        // Validate severity
        match shape.severity {
            Severity::Info | Severity::Warning | Severity::Violation => {
                // Valid severities
            }
        }

        // Validate messages (check for valid language tags)
        for (lang_tag, _message) in &shape.messages {
            if !self.is_valid_language_tag(lang_tag) {
                return Err(ShaclError::ShapeParsing(format!(
                    "Shape '{}' has invalid language tag: '{}'",
                    shape.id, lang_tag
                )));
            }
        }

        Ok(())
    }

    /// Validate inheritance references
    fn validate_inheritance_references(&self, shape: &Shape) -> Result<()> {
        for parent_shape_id in &shape.extends {
            // Check if the parent shape is known
            if !self.shapes.contains_key(parent_shape_id) {
                tracing::warn!(
                    "Shape '{}' extends unknown shape '{}'",
                    shape.id,
                    parent_shape_id
                );
                // This is a warning, not an error, as the parent shape might be defined later
            }
        }
        Ok(())
    }

    /// Enhanced SHACL specification compliance validation
    fn validate_shacl_compliance(&self, shape: &Shape) -> Result<()> {
        // Check for required properties on property shapes
        if shape.is_property_shape() {
            if shape.path.is_none() {
                return Err(ShaclError::ShapeParsing(format!(
                    "Property shape '{}' must have exactly one sh:path property",
                    shape.id
                )));
            }
        }

        // Validate that node shapes don't have sh:path
        if shape.is_node_shape() && shape.path.is_some() {
            return Err(ShaclError::ShapeParsing(format!(
                "Node shape '{}' must not have sh:path property",
                shape.id
            )));
        }

        // Check for valid target combinations
        self.validate_target_combinations(shape)?;

        // Validate constraint applicability to shape type
        self.validate_constraint_applicability(shape)?;

        // Check for deprecated or experimental features
        self.validate_feature_usage(shape)?;

        Ok(())
    }

    /// Validate target combinations
    fn validate_target_combinations(&self, shape: &Shape) -> Result<()> {
        // Node shapes should typically have targets, property shapes may or may not
        if shape.is_node_shape() && shape.targets.is_empty() {
            tracing::warn!(
                "Node shape '{}' has no targets - it will not validate any data",
                shape.id
            );
        }

        // Check for redundant targets
        let mut unique_targets = HashSet::new();
        for target in &shape.targets {
            let target_key = format!("{:?}", target);
            if unique_targets.contains(&target_key) {
                tracing::warn!("Shape '{}' has duplicate target: {:?}", shape.id, target);
            }
            unique_targets.insert(target_key);
        }

        Ok(())
    }

    /// Validate constraint applicability to shape type
    fn validate_constraint_applicability(&self, shape: &Shape) -> Result<()> {
        for (constraint_id, constraint) in &shape.constraints {
            // Some constraints only make sense for property shapes
            match constraint {
                Constraint::MinCount(_) | Constraint::MaxCount(_) => {
                    if shape.is_node_shape() {
                        tracing::warn!(
                            "Cardinality constraint '{}' in node shape '{}' has no effect",
                            constraint_id,
                            shape.id
                        );
                    }
                }
                Constraint::UniqueLang(_) => {
                    if shape.is_node_shape() {
                        tracing::warn!(
                            "UniqueLang constraint in node shape '{}' may not be meaningful",
                            shape.id
                        );
                    }
                }
                _ => {} // Other constraints are valid for both types
            }
        }
        Ok(())
    }

    /// Validate feature usage for warnings about deprecated/experimental features
    fn validate_feature_usage(&self, _shape: &Shape) -> Result<()> {
        // This could be expanded to check for deprecated SHACL features
        // or warn about experimental extensions
        Ok(())
    }

    /// Validate constraint parameters for correctness
    fn validate_constraint_parameters(&self, shape: &Shape) -> Result<()> {
        for (constraint_id, constraint) in &shape.constraints {
            match constraint {
                Constraint::Pattern(pattern_constraint) => {
                    // Validate regex pattern
                    if let Err(e) = regex::Regex::new(&pattern_constraint.pattern) {
                        return Err(ShaclError::ShapeParsing(format!(
                            "Invalid regex pattern in constraint '{}' for shape '{}': {}",
                            constraint_id, shape.id, e
                        )));
                    }
                }
                Constraint::LanguageIn(lang_constraint) => {
                    // Validate language tags
                    for lang_tag in &lang_constraint.languages {
                        if !self.is_valid_language_tag(lang_tag) {
                            return Err(ShaclError::ShapeParsing(format!(
                                "Invalid language tag '{}' in constraint '{}' for shape '{}'",
                                lang_tag, constraint_id, shape.id
                            )));
                        }
                    }
                }
                Constraint::In(in_constraint) => {
                    // Validate that the list is not empty
                    if in_constraint.values.is_empty() {
                        return Err(ShaclError::ShapeParsing(format!(
                            "sh:in constraint in shape '{}' cannot have empty value list",
                            shape.id
                        )));
                    }
                }
                Constraint::QualifiedValueShape(qvs_constraint) => {
                    // Validate qualified cardinality constraints
                    if qvs_constraint.qualified_min_count.is_none()
                        && qvs_constraint.qualified_max_count.is_none()
                    {
                        return Err(ShaclError::ShapeParsing(format!(
                            "QualifiedValueShape constraint in shape '{}' must have at least one of sh:qualifiedMinCount or sh:qualifiedMaxCount",
                            shape.id
                        )));
                    }

                    if let (Some(min), Some(max)) = (
                        qvs_constraint.qualified_min_count,
                        qvs_constraint.qualified_max_count,
                    ) {
                        if min > max {
                            return Err(ShaclError::ShapeParsing(format!(
                                "QualifiedValueShape constraint in shape '{}' has qualifiedMinCount ({}) > qualifiedMaxCount ({})",
                                shape.id, min, max
                            )));
                        }
                    }
                }
                _ => {} // Other constraints validated elsewhere
            }
        }
        Ok(())
    }

    /// Validate logical consistency of constraint combinations
    fn validate_logical_consistency(&self, shape: &Shape) -> Result<()> {
        let constraints = &shape.constraints;

        // Check for impossible combinations
        self.validate_impossible_combinations(shape, constraints)?;

        // Check for redundant constraints
        self.validate_redundant_constraints(shape, constraints)?;

        // Check for conflicting constraints
        self.validate_conflicting_constraints(shape, constraints)?;

        Ok(())
    }

    /// Check for impossible constraint combinations
    fn validate_impossible_combinations(
        &self,
        shape: &Shape,
        constraints: &IndexMap<ConstraintComponentId, Constraint>,
    ) -> Result<()> {
        // Check for NodeKind + Datatype conflicts
        if let (Some(node_kind), Some(datatype)) = (
            constraints.get(&ConstraintComponentId::new("nodeKind")),
            constraints.get(&ConstraintComponentId::new("datatype")),
        ) {
            if let (Constraint::NodeKind(nk), Constraint::Datatype(_)) = (node_kind, datatype) {
                use crate::constraints::value_constraints::NodeKind;
                match nk.node_kind {
                    NodeKind::Iri | NodeKind::BlankNode | NodeKind::BlankNodeOrIri => {
                        return Err(ShaclError::ShapeParsing(format!(
                            "Shape '{}' has conflicting nodeKind and datatype constraints - IRIs and blank nodes cannot have datatypes",
                            shape.id
                        )));
                    }
                    _ => {} // Literal node kinds can have datatypes
                }
            }
        }

        Ok(())
    }

    /// Check for redundant constraints
    fn validate_redundant_constraints(
        &self,
        shape: &Shape,
        _constraints: &IndexMap<ConstraintComponentId, Constraint>,
    ) -> Result<()> {
        // This could be expanded to detect more redundant patterns
        tracing::debug!(
            "Checking for redundant constraints in shape '{}'...",
            shape.id
        );
        Ok(())
    }

    /// Check for conflicting constraints beyond basic cases
    fn validate_conflicting_constraints(
        &self,
        shape: &Shape,
        constraints: &IndexMap<ConstraintComponentId, Constraint>,
    ) -> Result<()> {
        // Check for class/datatype conflicts
        if let (Some(Constraint::Class(_)), Some(Constraint::Datatype(_))) = (
            constraints.get(&ConstraintComponentId::new("class")),
            constraints.get(&ConstraintComponentId::new("datatype")),
        ) {
            tracing::warn!(
                "Shape '{}' has both class and datatype constraints - ensure they are compatible",
                shape.id
            );
        }

        Ok(())
    }

    /// Validate IRI format
    fn validate_iri_format(&self, iri: &str) -> Result<()> {
        use url::Url;

        // Check if it's a valid absolute IRI
        if iri.starts_with("http://") || iri.starts_with("https://") || iri.starts_with("urn:") {
            Url::parse(iri)
                .map_err(|e| ShaclError::ShapeParsing(format!("Invalid IRI '{}': {}", iri, e)))?;
        } else if iri.contains(':') {
            // Could be a prefixed name - validate prefix part
            if let Some(colon_pos) = iri.find(':') {
                let prefix = &iri[..colon_pos];
                if prefix.is_empty() {
                    return Err(ShaclError::ShapeParsing(format!(
                        "Invalid prefixed name '{}': empty prefix",
                        iri
                    )));
                }
            }
        } else {
            return Err(ShaclError::ShapeParsing(format!(
                "Invalid IRI '{}': must be absolute or prefixed",
                iri
            )));
        }

        Ok(())
    }

    /// Validate SPARQL query syntax (basic validation)
    fn validate_sparql_query(&self, query: &str) -> Result<()> {
        // Basic validation - check for required keywords
        let query_upper = query.to_uppercase();

        if !query_upper.contains("SELECT")
            && !query_upper.contains("ASK")
            && !query_upper.contains("CONSTRUCT")
            && !query_upper.contains("DESCRIBE")
        {
            return Err(ShaclError::ShapeParsing(
                "SPARQL query must contain a valid query form (SELECT, ASK, CONSTRUCT, or DESCRIBE)".to_string()
            ));
        }

        // Check for balanced braces
        let mut brace_count = 0;
        for char in query.chars() {
            match char {
                '{' => brace_count += 1,
                '}' => brace_count -= 1,
                _ => {}
            }
            if brace_count < 0 {
                return Err(ShaclError::ShapeParsing(
                    "SPARQL query has unbalanced braces".to_string(),
                ));
            }
        }

        if brace_count != 0 {
            return Err(ShaclError::ShapeParsing(
                "SPARQL query has unbalanced braces".to_string(),
            ));
        }

        Ok(())
    }

    /// Check if a language tag is valid (basic check)
    fn is_valid_language_tag(&self, tag: &str) -> bool {
        // Very basic validation - language tags should be 2-3 characters, possibly with region
        // Format: language[-region] e.g., "en", "en-US", "fr-CA"
        if tag.is_empty() || tag.len() > 8 {
            return false;
        }

        let parts: Vec<&str> = tag.split('-').collect();
        if parts.is_empty() || parts.len() > 2 {
            return false;
        }

        // First part should be 2-3 character language code
        let lang = parts[0];
        if lang.len() < 2 || lang.len() > 3 || !lang.chars().all(|c| c.is_alphabetic()) {
            return false;
        }

        // Optional second part should be 2-3 character region code
        if parts.len() == 2 {
            let region = parts[1];
            if region.len() < 2 || region.len() > 3 || !region.chars().all(|c| c.is_alphabetic()) {
                return false;
            }
        }

        true
    }

    /// Update shape dependency graph
    fn update_shape_dependencies(&mut self, shape_id: ShapeId) -> Result<()> {
        use petgraph::graph::NodeIndex;

        // Find or create node for this shape
        let shape_node = self.get_or_create_shape_node(&shape_id);

        // Collect all dependencies first to avoid borrow checker issues
        let mut all_dependencies = Vec::new();

        if let Some(shape) = self.shapes.get(&shape_id) {
            // Collect dependencies from inheritance
            all_dependencies.extend(shape.extends.iter().cloned());

            // Collect dependencies from constraints
            for (_, constraint) in &shape.constraints {
                let dependencies = self.extract_constraint_dependencies(constraint);
                all_dependencies.extend(dependencies);
            }
        }

        // Now add all the edges
        for dep_shape_id in all_dependencies {
            let dep_node = self.get_or_create_shape_node(&dep_shape_id);
            self.shape_dependencies.add_edge(shape_node, dep_node, ());
        }

        // Check for circular dependencies
        if petgraph::algo::is_cyclic_directed(&self.shape_dependencies) {
            return Err(ShaclError::ShapeParsing(
                "circular dependency detected in shape definitions".to_string(),
            ));
        }

        Ok(())
    }

    /// Get or create a node in the dependency graph for a shape
    fn get_or_create_shape_node(&mut self, shape_id: &ShapeId) -> petgraph::graph::NodeIndex {
        use petgraph::visit::IntoNodeReferences;

        // Check if node already exists
        for (node_idx, node_shape_id) in self.shape_dependencies.node_references() {
            if node_shape_id == shape_id {
                return node_idx;
            }
        }

        // Create new node
        self.shape_dependencies.add_node(shape_id.clone())
    }

    /// Extract shape dependencies from a constraint
    fn extract_constraint_dependencies(&self, constraint: &Constraint) -> Vec<ShapeId> {
        let mut dependencies = Vec::new();

        match constraint {
            Constraint::Node(node_constraint) => {
                dependencies.push(node_constraint.shape.clone());
            }
            Constraint::QualifiedValueShape(qualified_constraint) => {
                dependencies.push(qualified_constraint.shape.clone());
            }
            Constraint::Not(not_constraint) => {
                dependencies.push(not_constraint.shape.clone());
            }
            Constraint::And(and_constraint) => {
                dependencies.extend(and_constraint.shapes.clone());
            }
            Constraint::Or(or_constraint) => {
                dependencies.extend(or_constraint.shapes.clone());
            }
            Constraint::Xone(xone_constraint) => {
                dependencies.extend(xone_constraint.shapes.clone());
            }
            _ => {
                // Other constraints don't have shape dependencies
            }
        }

        dependencies
    }

    /// Get optimal shape evaluation order based on dependencies and sh:order property
    pub fn get_evaluation_order(&self) -> Vec<ShapeId> {
        use petgraph::algo::toposort;
        use petgraph::visit::IntoNodeReferences;

        // Perform topological sort on the dependency graph
        let mut shape_ids: Vec<ShapeId> = match toposort(&self.shape_dependencies, None) {
            Ok(sorted_nodes) => {
                // Map node indices back to shape IDs
                sorted_nodes
                    .into_iter()
                    .filter_map(|node_idx| self.shape_dependencies.node_weight(node_idx).cloned())
                    .collect()
            }
            Err(_) => {
                // If there's a cycle (shouldn't happen as we check during add),
                // fall back to arbitrary order
                tracing::warn!("Dependency cycle detected during evaluation order calculation");
                self.shapes.keys().cloned().collect()
            }
        };

        // Secondary sort by sh:order property for stable ordering
        shape_ids.sort_by(|a: &ShapeId, b: &ShapeId| {
            let shape_a = self.shapes.get(a);
            let shape_b = self.shapes.get(b);

            match (shape_a, shape_b) {
                (Some(s_a), Some(s_b)) => {
                    // Primary sort by order (lower values first)
                    s_a.order
                        .unwrap_or(0)
                        .cmp(&s_b.order.unwrap_or(0))
                        // Secondary sort by priority (higher values first)
                        .then_with(|| s_b.effective_priority().cmp(&s_a.effective_priority()))
                        // Tertiary sort by shape ID for stability
                        .then_with(|| a.as_str().cmp(b.as_str()))
                }
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => a.as_str().cmp(b.as_str()),
            }
        });

        shape_ids
    }

    /// Clear all internal caches
    pub fn clear_caches(&mut self) {
        self.target_cache.clear();
    }

    /// Resolve inherited constraints for a shape
    pub fn resolve_inherited_constraints(
        &self,
        shape_id: &ShapeId,
    ) -> Result<IndexMap<ConstraintComponentId, Constraint>> {
        let mut resolved_constraints = IndexMap::new();
        let mut visited = HashSet::new();

        self.collect_inherited_constraints(shape_id, &mut resolved_constraints, &mut visited)?;

        Ok(resolved_constraints)
    }

    /// Recursively collect constraints from a shape and its parents
    fn collect_inherited_constraints(
        &self,
        shape_id: &ShapeId,
        resolved_constraints: &mut IndexMap<ConstraintComponentId, Constraint>,
        visited: &mut HashSet<ShapeId>,
    ) -> Result<()> {
        // Avoid infinite recursion
        if visited.contains(shape_id) {
            return Ok(());
        }
        visited.insert(shape_id.clone());

        if let Some(shape) = self.shapes.get(shape_id) {
            // First process parent shapes (depth-first)
            for parent_id in &shape.extends {
                self.collect_inherited_constraints(parent_id, resolved_constraints, visited)?;
            }

            // Then add this shape's constraints, potentially overriding parent constraints
            for (component_id, constraint) in &shape.constraints {
                resolved_constraints.insert(component_id.clone(), constraint.clone());
            }
        }

        Ok(())
    }

    /// Resolve shapes with priority-based conflict resolution
    /// Returns shapes sorted by priority (highest first)
    pub fn resolve_shapes_by_priority(&self, shape_ids: &[ShapeId]) -> Vec<&Shape> {
        let mut shapes: Vec<&Shape> = shape_ids
            .iter()
            .filter_map(|id| self.shapes.get(id))
            .collect();

        // Sort by priority (highest first), then by ID for stability
        shapes.sort_by(|a, b| {
            b.effective_priority()
                .cmp(&a.effective_priority())
                .then_with(|| a.id.as_str().cmp(b.id.as_str()))
        });

        shapes
    }

    /// Get all shapes that a given shape inherits from (transitively)
    pub fn get_all_parent_shapes(&self, shape_id: &ShapeId) -> Result<Vec<ShapeId>> {
        let mut parents = Vec::new();
        let mut visited = HashSet::new();

        self.collect_parent_shapes(shape_id, &mut parents, &mut visited)?;

        Ok(parents)
    }

    fn collect_parent_shapes(
        &self,
        shape_id: &ShapeId,
        parents: &mut Vec<ShapeId>,
        visited: &mut HashSet<ShapeId>,
    ) -> Result<()> {
        if visited.contains(shape_id) {
            return Ok(());
        }
        visited.insert(shape_id.clone());

        if let Some(shape) = self.shapes.get(shape_id) {
            for parent_id in &shape.extends {
                parents.push(parent_id.clone());
                self.collect_parent_shapes(parent_id, parents, visited)?;
            }
        }

        Ok(())
    }

    /// Get validation statistics
    pub fn get_statistics(&self) -> ValidationStats {
        ValidationStats {
            total_shapes: self.shapes.len(),
            node_shapes: self.shapes.values().filter(|s| s.is_node_shape()).count(),
            property_shapes: self
                .shapes
                .values()
                .filter(|s| s.is_property_shape())
                .count(),
            active_shapes: self.shapes.values().filter(|s| s.is_active()).count(),
            deactivated_shapes: self.shapes.values().filter(|s| !s.is_active()).count(),
        }
    }
}

impl Default for Validator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStats {
    pub total_shapes: usize,
    pub node_shapes: usize,
    pub property_shapes: usize,
    pub active_shapes: usize,
    pub deactivated_shapes: usize,
}

/// Builder for creating validators with custom configuration
#[derive(Debug)]
pub struct ValidatorBuilder {
    config: ValidationConfig,
}

impl ValidatorBuilder {
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
        }
    }

    pub fn max_violations(mut self, max: usize) -> Self {
        self.config.max_violations = max;
        self
    }

    pub fn include_info(mut self, include: bool) -> Self {
        self.config.include_info = include;
        self
    }

    pub fn include_warnings(mut self, include: bool) -> Self {
        self.config.include_warnings = include;
        self
    }

    pub fn fail_fast(mut self, fail_fast: bool) -> Self {
        self.config.fail_fast = fail_fast;
        self
    }

    pub fn max_recursion_depth(mut self, depth: usize) -> Self {
        self.config.max_recursion_depth = depth;
        self
    }

    pub fn timeout_ms(mut self, timeout: Option<u64>) -> Self {
        self.config.timeout_ms = timeout;
        self
    }

    pub fn parallel(mut self, parallel: bool) -> Self {
        self.config.parallel = parallel;
        self
    }

    pub fn context(mut self, context: HashMap<String, String>) -> Self {
        self.config.context = context;
        self
    }

    pub fn build(self) -> Validator {
        Validator::with_config(self.config)
    }
}

impl Default for ValidatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Version information for OxiRS SHACL
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize OxiRS SHACL with default configuration
pub fn init() -> Result<()> {
    tracing::info!("Initializing OxiRS SHACL v{}", VERSION);
    Ok(())
}
