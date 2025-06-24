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
//! ## Usage
//! 
//! ```rust
//! use oxirs_shacl::{Validator, ValidationConfig};
//! use oxirs_core::store::Store;
//! 
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let store = Store::new()?;
//! let validator = Validator::new();
//! 
//! // Load SHACL shapes
//! // validator.load_shapes_from_store(&store, "shapes_graph")?;
//! 
//! // Validate data
//! // let result = validator.validate_store(&store, ValidationConfig::default())?;
//! 
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
    model::{NamedNode, Term, Triple, Quad, BlankNode, Literal, Variable},
    store::Store,
    graph::Graph,
    OxirsError,
};

pub mod shapes;
pub mod constraints;
pub mod validation;
pub mod report;
pub mod targets;
pub mod paths;
pub mod sparql;
pub mod vocabulary;

// Re-export key types for convenience
pub use shapes::*;
pub use constraints::*;
pub use validation::*;
pub use report::*;
pub use targets::*;
pub use paths::*;

/// SHACL namespace IRI
pub static SHACL_NS: &str = "http://www.w3.org/ns/shacl#";

/// SHACL vocabulary terms
pub static SHACL_VOCAB: Lazy<vocabulary::ShaclVocabulary> = Lazy::new(vocabulary::ShaclVocabulary::new);

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
}

/// Violation severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
            // TODO: Update dependency graph
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
    pub fn load_shapes_from_store(&mut self, store: &Store, graph_name: Option<&str>) -> Result<usize> {
        let mut parser = shapes::ShapeParser::new();
        let shapes = parser.parse_shapes_from_store(store, graph_name)?;
        
        let count = shapes.len();
        for shape in shapes {
            self.add_shape(shape)?;
        }
        
        Ok(count)
    }
    
    /// Load shapes from RDF data
    pub fn load_shapes_from_rdf(&mut self, rdf_data: &str, format: &str, base_iri: Option<&str>) -> Result<usize> {
        let mut parser = shapes::ShapeParser::new();
        let shapes = parser.parse_shapes_from_rdf(rdf_data, format, base_iri)?;
        
        let count = shapes.len();
        for shape in shapes {
            self.add_shape(shape)?;
        }
        
        Ok(count)
    }
    
    /// Validate data in a store against all loaded shapes
    pub fn validate_store(&self, store: &Store, config: Option<ValidationConfig>) -> Result<ValidationReport> {
        let config = config.unwrap_or_else(|| self.config.clone());
        let mut engine = validation::ValidationEngine::new(&self.shapes, config);
        engine.validate_store(store)
    }
    
    /// Validate specific nodes against a specific shape
    pub fn validate_nodes(&self, store: &Store, shape_id: &ShapeId, nodes: &[Term], config: Option<ValidationConfig>) -> Result<ValidationReport> {
        let shape = self.get_shape(shape_id)
            .ok_or_else(|| ShaclError::ValidationEngine(format!("Shape not found: {}", shape_id)))?;
        
        let config = config.unwrap_or_else(|| self.config.clone());
        let mut engine = validation::ValidationEngine::new(&self.shapes, config);
        engine.validate_nodes(store, shape, nodes)
    }
    
    /// Validate a single node against a specific shape
    pub fn validate_node(&self, store: &Store, shape_id: &ShapeId, node: &Term, config: Option<ValidationConfig>) -> Result<ValidationReport> {
        self.validate_nodes(store, shape_id, &[node.clone()], config)
    }
    
    /// Check if shapes are valid
    fn validate_shape(&self, shape: &Shape) -> Result<()> {
        // Basic shape validation
        if shape.is_property_shape() && shape.path.is_none() {
            return Err(ShaclError::ShapeParsing(
                "Property shape must have a property path".to_string()
            ));
        }
        
        // Validate constraints
        for (component_id, constraint) in &shape.constraints {
            constraint.validate()?;
        }
        
        Ok(())
    }
    
    /// Update shape dependency graph
    fn update_shape_dependencies(&mut self, _shape_id: ShapeId) -> Result<()> {
        // TODO: Implement dependency graph update
        // This is important for detecting circular dependencies and optimizing evaluation order
        Ok(())
    }
    
    /// Clear all internal caches
    pub fn clear_caches(&mut self) {
        self.target_cache.clear();
    }
    
    /// Get validation statistics
    pub fn get_statistics(&self) -> ValidationStats {
        ValidationStats {
            total_shapes: self.shapes.len(),
            node_shapes: self.shapes.values().filter(|s| s.is_node_shape()).count(),
            property_shapes: self.shapes.values().filter(|s| s.is_property_shape()).count(),
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