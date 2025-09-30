//! SHACL target type definitions
//!
//! This module contains all type definitions for SHACL target selection.

use serde::{Deserialize, Serialize};

use oxirs_core::model::{NamedNode, Term};

/// SHACL target types for selecting nodes to validate
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Target {
    /// sh:targetClass - selects all instances of a class
    Class(NamedNode),

    /// sh:targetNode - selects specific nodes
    Node(Term),

    /// sh:targetObjectsOf - selects objects of a property
    ObjectsOf(NamedNode),

    /// sh:targetSubjectsOf - selects subjects of a property
    SubjectsOf(NamedNode),

    /// sh:target with SPARQL SELECT query
    Sparql(SparqlTarget),

    /// Implicit target (shape IRI used as class)
    Implicit(NamedNode),

    /// Complex target combinations

    /// Union of multiple targets (all nodes that match any of the targets)
    Union(TargetUnion),

    /// Intersection of multiple targets (only nodes that match all targets)
    Intersection(TargetIntersection),

    /// Difference between targets (nodes in first target but not in second)
    Difference(TargetDifference),

    /// Conditional target (nodes that match primary target and satisfy condition)
    Conditional(ConditionalTarget),

    /// Hierarchical target (nodes related to other targets through properties)
    Hierarchical(HierarchicalTarget),

    /// Path-based target (nodes reachable through property paths from base targets)
    PathBased(PathBasedTarget),
}

/// SPARQL-based target definition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparqlTarget {
    /// SPARQL SELECT query that returns target nodes
    pub query: String,

    /// Optional prefixes for the query
    pub prefixes: Option<String>,
}

/// Union target combining multiple targets
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetUnion {
    /// List of targets to combine with union
    pub targets: Vec<Target>,
    /// Optional optimization hints
    pub optimization_hints: Option<UnionOptimizationHints>,
}

/// Intersection target requiring nodes to match all targets
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetIntersection {
    /// List of targets that must all match
    pub targets: Vec<Target>,
    /// Optional optimization hints
    pub optimization_hints: Option<IntersectionOptimizationHints>,
}

/// Difference target (first target minus second target)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetDifference {
    /// Primary target (nodes to include)
    pub primary_target: Box<Target>,
    /// Exclusion target (nodes to exclude)
    pub exclusion_target: Box<Target>,
}

/// Conditional target with additional constraints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConditionalTarget {
    /// Base target to start with
    pub base_target: Box<Target>,
    /// Condition that must be satisfied
    pub condition: TargetCondition,
    /// Optional evaluation context
    pub context: Option<TargetContext>,
}

/// Hierarchical target based on relationships
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalTarget {
    /// Root target nodes
    pub root_target: Box<Target>,
    /// Relationship to follow
    pub relationship: HierarchicalRelationship,
    /// Maximum depth to traverse (-1 for unlimited)
    pub max_depth: i32,
    /// Whether to include intermediate nodes
    pub include_intermediate: bool,
}

/// Path-based target using property paths
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathBasedTarget {
    /// Starting target nodes
    pub start_target: Box<Target>,
    /// Property path to follow
    pub path: crate::paths::PropertyPath,
    /// Direction to traverse the path
    pub direction: PathDirection,
    /// Optional filtering constraints
    pub filters: Vec<PathFilter>,
}

/// Optimization hints for union targets
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnionOptimizationHints {
    /// Whether to use SPARQL UNION optimization
    pub use_sparql_union: bool,
    /// Whether to deduplicate results
    pub deduplicate_results: bool,
    /// Estimated selectivity for each target
    pub target_selectivities: Vec<f64>,
}

/// Optimization hints for intersection targets
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntersectionOptimizationHints {
    /// Order targets by estimated selectivity (most selective first)
    pub order_by_selectivity: bool,
    /// Use early termination when possible
    pub use_early_termination: bool,
    /// Estimated selectivity for each target
    pub target_selectivities: Vec<f64>,
}

/// Condition for conditional targets
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetCondition {
    /// SPARQL ASK query condition
    SparqlAsk {
        query: String,
        prefixes: Option<String>,
    },
    /// Property existence condition
    PropertyExists {
        property: NamedNode,
        direction: PropertyDirection,
    },
    /// Property value condition
    PropertyValue {
        property: NamedNode,
        value: Term,
        direction: PropertyDirection,
    },
    /// Type condition
    HasType { class_iri: NamedNode },
    /// Cardinality condition
    Cardinality {
        property: NamedNode,
        min_count: Option<usize>,
        max_count: Option<usize>,
        direction: PropertyDirection,
    },
}

/// Hierarchical relationship types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HierarchicalRelationship {
    /// Follow property in forward direction
    Property(NamedNode),
    /// Follow property in inverse direction
    InverseProperty(NamedNode),
    /// Follow subclass relationship
    SubclassOf,
    /// Follow superclass relationship
    SuperclassOf,
    /// Follow any rdfs:subPropertyOf relationship
    SubpropertyOf,
    /// Follow any rdfs:subPropertyOf relationship in inverse
    SuperpropertyOf,
    /// Custom SPARQL path
    CustomPath(String),
}

/// Path traversal direction
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathDirection {
    /// Forward direction (subject to object)
    Forward,
    /// Backward direction (object to subject)
    Backward,
    /// Both directions
    Both,
}

/// Property direction for conditions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropertyDirection {
    /// Node is subject of the property
    Subject,
    /// Node is object of the property
    Object,
    /// Node can be either subject or object
    Either,
}

/// Filters for path-based targets
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathFilter {
    /// Filter by node type
    NodeType(NodeTypeFilter),
    /// Filter by property value
    PropertyValue { property: NamedNode, value: Term },
    /// Filter by SPARQL condition
    SparqlCondition {
        condition: String,
        prefixes: Option<String>,
    },
}

/// Node type filters
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeTypeFilter {
    /// Only IRI nodes
    IriOnly,
    /// Only blank nodes
    BlankNodeOnly,
    /// Only literal nodes
    LiteralOnly,
    /// Specific class instances
    InstanceOf(NamedNode),
}

/// Target evaluation context
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetContext {
    /// Additional variables bound in the context
    pub bindings: std::collections::HashMap<String, Term>,
    /// Custom prefixes for this context
    pub prefixes: std::collections::HashMap<String, String>,
}
