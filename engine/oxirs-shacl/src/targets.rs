//! SHACL target selection implementation
//!
//! This module handles target node selection according to SHACL specification.

use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

use oxirs_core::{
    model::{BlankNode, NamedNode, RdfTerm, Term, Triple},
    ConcreteStore, OxirsError, Store,
};

use crate::{Result, ShaclError, SHACL_VOCAB};

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

impl Target {
    /// Create a class target
    pub fn class(class_iri: NamedNode) -> Self {
        Target::Class(class_iri)
    }

    /// Create a node target
    pub fn node(node: Term) -> Self {
        Target::Node(node)
    }

    /// Create an objects-of target
    pub fn objects_of(property: NamedNode) -> Self {
        Target::ObjectsOf(property)
    }

    /// Create a subjects-of target
    pub fn subjects_of(property: NamedNode) -> Self {
        Target::SubjectsOf(property)
    }

    /// Create a SPARQL target
    pub fn sparql(query: String, prefixes: Option<String>) -> Self {
        Target::Sparql(SparqlTarget { query, prefixes })
    }

    /// Create an implicit target
    pub fn implicit(class_iri: NamedNode) -> Self {
        Target::Implicit(class_iri)
    }

    /// Create a union target combining multiple targets
    pub fn union(targets: Vec<Target>) -> Self {
        Target::Union(TargetUnion {
            targets,
            optimization_hints: None,
        })
    }

    /// Create a union target with optimization hints
    pub fn union_with_hints(targets: Vec<Target>, hints: UnionOptimizationHints) -> Self {
        Target::Union(TargetUnion {
            targets,
            optimization_hints: Some(hints),
        })
    }

    /// Create an intersection target requiring all targets to match
    pub fn intersection(targets: Vec<Target>) -> Self {
        Target::Intersection(TargetIntersection {
            targets,
            optimization_hints: None,
        })
    }

    /// Create an intersection target with optimization hints
    pub fn intersection_with_hints(
        targets: Vec<Target>,
        hints: IntersectionOptimizationHints,
    ) -> Self {
        Target::Intersection(TargetIntersection {
            targets,
            optimization_hints: Some(hints),
        })
    }

    /// Create a difference target (primary minus exclusion)
    pub fn difference(primary: Target, exclusion: Target) -> Self {
        Target::Difference(TargetDifference {
            primary_target: Box::new(primary),
            exclusion_target: Box::new(exclusion),
        })
    }

    /// Create a conditional target with a condition
    pub fn conditional(base: Target, condition: TargetCondition) -> Self {
        Target::Conditional(ConditionalTarget {
            base_target: Box::new(base),
            condition,
            context: None,
        })
    }

    /// Create a conditional target with context
    pub fn conditional_with_context(
        base: Target,
        condition: TargetCondition,
        context: TargetContext,
    ) -> Self {
        Target::Conditional(ConditionalTarget {
            base_target: Box::new(base),
            condition,
            context: Some(context),
        })
    }

    /// Create a hierarchical target following relationships
    pub fn hierarchical(
        root: Target,
        relationship: HierarchicalRelationship,
        max_depth: i32,
    ) -> Self {
        Target::Hierarchical(HierarchicalTarget {
            root_target: Box::new(root),
            relationship,
            max_depth,
            include_intermediate: false,
        })
    }

    /// Create a hierarchical target with intermediate nodes included
    pub fn hierarchical_with_intermediate(
        root: Target,
        relationship: HierarchicalRelationship,
        max_depth: i32,
    ) -> Self {
        Target::Hierarchical(HierarchicalTarget {
            root_target: Box::new(root),
            relationship,
            max_depth,
            include_intermediate: true,
        })
    }

    /// Create a path-based target following property paths
    pub fn path_based(
        start: Target,
        path: crate::paths::PropertyPath,
        direction: PathDirection,
    ) -> Self {
        Target::PathBased(PathBasedTarget {
            start_target: Box::new(start),
            path,
            direction,
            filters: Vec::new(),
        })
    }

    /// Create a path-based target with filters
    pub fn path_based_with_filters(
        start: Target,
        path: crate::paths::PropertyPath,
        direction: PathDirection,
        filters: Vec<PathFilter>,
    ) -> Self {
        Target::PathBased(PathBasedTarget {
            start_target: Box::new(start),
            path,
            direction,
            filters,
        })
    }

    /// Check if this target requires complex evaluation
    pub fn is_complex(&self) -> bool {
        matches!(
            self,
            Target::Union(_)
                | Target::Intersection(_)
                | Target::Difference(_)
                | Target::Conditional(_)
                | Target::Hierarchical(_)
                | Target::PathBased(_)
        )
    }

    /// Get the estimated complexity of this target (0-10 scale)
    pub fn complexity_score(&self) -> u8 {
        match self {
            Target::Node(_) => 1,
            Target::Class(_) | Target::Implicit(_) => 2,
            Target::ObjectsOf(_) | Target::SubjectsOf(_) => 3,
            Target::Sparql(_) => 4,
            Target::Union(u) => 5 + (u.targets.len() as u8).min(3),
            Target::Intersection(i) => 6 + (i.targets.len() as u8).min(3),
            Target::Difference(_) => 7,
            Target::Conditional(_) => 8,
            Target::Hierarchical(_) => 9,
            Target::PathBased(_) => 10,
        }
    }

    /// Get all nested targets within complex targets
    pub fn nested_targets(&self) -> Vec<&Target> {
        match self {
            Target::Union(u) => u.targets.iter().collect(),
            Target::Intersection(i) => i.targets.iter().collect(),
            Target::Difference(d) => vec![&d.primary_target, &d.exclusion_target],
            Target::Conditional(c) => vec![&c.base_target],
            Target::Hierarchical(h) => vec![&h.root_target],
            Target::PathBased(p) => vec![&p.start_target],
            _ => vec![],
        }
    }
}

/// Target selector for finding nodes that match target definitions
#[derive(Debug)]
pub struct TargetSelector {
    /// Cache for target results to improve performance
    cache: std::collections::HashMap<String, CachedTargetResult>,

    /// Optimization settings
    optimization_config: TargetOptimizationConfig,

    /// Performance statistics
    stats: TargetSelectionStats,

    /// Query plan cache for SPARQL targets
    query_plan_cache: std::collections::HashMap<String, QueryPlan>,

    /// Index usage statistics for adaptive optimization
    index_usage_stats: std::collections::HashMap<String, IndexUsageStats>,
}

/// Configuration for target selection optimization
#[derive(Debug, Clone)]
pub struct TargetOptimizationConfig {
    /// Enable result caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl: u64,
    /// Maximum cache size (number of entries)
    pub max_cache_size: usize,
    /// Enable query plan optimization
    pub enable_query_optimization: bool,
    /// Index hint threshold (use index if selectivity < threshold)
    pub index_hint_threshold: f64,
    /// Parallel execution threshold (execute in parallel if cardinality > threshold)
    pub parallel_threshold: usize,
    /// Enable adaptive optimization based on statistics
    pub enable_adaptive_optimization: bool,
}

impl Default for TargetOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_ttl: 300, // 5 minutes
            max_cache_size: 1000,
            enable_query_optimization: true,
            index_hint_threshold: 0.1,
            parallel_threshold: 10000,
            enable_adaptive_optimization: true,
        }
    }
}

/// Cached target result
#[derive(Debug, Clone)]
struct CachedTargetResult {
    /// Target nodes
    nodes: HashSet<Term>,
    /// Cache timestamp
    cached_at: std::time::Instant,
    /// Cache statistics
    stats: CacheStats,
}

/// Cache statistics
#[derive(Debug, Clone)]
struct CacheStats {
    /// Number of hits
    hits: usize,
    /// Number of misses
    misses: usize,
    /// Average query time
    avg_query_time: std::time::Duration,
}

/// Target selection statistics
#[derive(Debug, Clone)]
pub struct TargetSelectionStats {
    /// Total number of target evaluations
    pub total_evaluations: usize,
    /// Total time spent on target selection
    pub total_time: std::time::Duration,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average evaluation time
    pub avg_evaluation_time: std::time::Duration,
    /// Index usage statistics
    pub index_usage_rate: f64,
    /// Parallel execution rate
    pub parallel_execution_rate: f64,
}

/// Query plan for SPARQL targets
#[derive(Debug, Clone)]
struct QueryPlan {
    /// Optimized SPARQL query
    optimized_query: String,
    /// Estimated cardinality
    estimated_cardinality: usize,
    /// Index usage recommendations
    index_hints: Vec<IndexHint>,
    /// Execution strategy
    execution_strategy: ExecutionStrategy,
    /// Plan creation time
    created_at: std::time::Instant,
}

/// Index usage hint
#[derive(Debug, Clone)]
struct IndexHint {
    /// Index type
    index_type: String,
    /// Estimated selectivity
    selectivity: f64,
    /// Cost benefit
    cost_benefit: f64,
}

/// Execution strategy for target selection
#[derive(Debug, Clone)]
enum ExecutionStrategy {
    /// Sequential execution
    Sequential,
    /// Parallel execution
    Parallel,
    /// Index-driven execution
    IndexDriven,
    /// Hybrid approach
    Hybrid,
}

/// Index usage statistics
#[derive(Debug, Clone)]
struct IndexUsageStats {
    /// Number of times used
    usage_count: usize,
    /// Average performance gain
    avg_performance_gain: f64,
    /// Last used timestamp
    last_used: std::time::Instant,
}

/// Target cache statistics
#[derive(Debug, Clone)]
pub struct TargetCacheStats {
    /// Total cache hits
    pub hits: usize,
    /// Total cache misses
    pub misses: usize,
    /// Cache hit rate
    pub hit_rate: f64,
    /// Cache size
    pub cache_size: usize,
    /// Memory usage estimate
    pub memory_usage_bytes: usize,
}

/// Query optimization options
#[derive(Debug, Clone)]
pub struct QueryOptimizationOptions {
    /// Maximum number of results to return
    pub limit: Option<usize>,
    /// Ensure deterministic ordering of results
    pub deterministic_results: bool,
    /// Use index hints in generated queries
    pub use_index_hints: bool,
    /// Include performance monitoring hints
    pub include_performance_hints: bool,
    /// Use UNION optimization for batch queries
    pub use_union_optimization: bool,
    /// Custom optimization parameters
    pub custom_params: std::collections::HashMap<String, String>,
}

impl Default for QueryOptimizationOptions {
    fn default() -> Self {
        Self {
            limit: None,
            deterministic_results: false,
            use_index_hints: true,
            include_performance_hints: false,
            use_union_optimization: true,
            custom_params: std::collections::HashMap::new(),
        }
    }
}

/// Optimized query result
#[derive(Debug, Clone)]
pub struct OptimizedQuery {
    /// The optimized SPARQL query
    pub sparql: String,
    /// Estimated result cardinality
    pub estimated_cardinality: usize,
    /// Recommended execution strategy
    pub execution_strategy: ExecutionStrategy,
    /// Index usage hints
    pub index_hints: Vec<IndexHint>,
    /// Time spent on optimization
    pub optimization_time: std::time::Duration,
}

/// Execution plan for target selection
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Recommended execution strategy
    pub execution_strategy: ExecutionStrategy,
    /// Index usage hints
    pub index_hints: Vec<IndexHint>,
    /// Estimated cardinality
    pub estimated_cardinality: usize,
}

/// Batch query result
#[derive(Debug, Clone)]
pub struct BatchQueryResult {
    /// Individual optimized queries
    pub individual_queries: Vec<OptimizedQuery>,
    /// Optional union query combining all targets
    pub union_query: Option<String>,
    /// Total estimated cardinality across all queries
    pub total_estimated_cardinality: usize,
    /// Time spent on batch optimization
    pub batch_optimization_time: std::time::Duration,
}

impl Default for TargetSelectionStats {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            total_time: std::time::Duration::from_millis(0),
            cache_hit_rate: 0.0,
            avg_evaluation_time: std::time::Duration::from_millis(0),
            index_usage_rate: 0.0,
            parallel_execution_rate: 0.0,
        }
    }
}

impl TargetSelector {
    /// Create a new target selector
    pub fn new() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            optimization_config: TargetOptimizationConfig::default(),
            stats: TargetSelectionStats::default(),
            query_plan_cache: std::collections::HashMap::new(),
            index_usage_stats: std::collections::HashMap::new(),
        }
    }

    /// Create a new target selector with custom optimization config
    pub fn with_config(config: TargetOptimizationConfig) -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            optimization_config: config,
            stats: TargetSelectionStats::default(),
            query_plan_cache: std::collections::HashMap::new(),
            index_usage_stats: std::collections::HashMap::new(),
        }
    }

    /// Generate efficient SPARQL query for target selection
    pub fn generate_target_query(
        &self,
        target: &Target,
        graph_name: Option<&str>,
    ) -> Result<String> {
        match target {
            Target::Class(class_iri) => {
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{}> {{", graph)
                } else {
                    String::new()
                };

                let close_clause = if graph_name.is_some() { " }" } else { "" };

                Ok(format!(
                    "SELECT DISTINCT ?target WHERE {{ {} ?target <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{}> .{} }}",
                    graph_clause,
                    class_iri.as_str(),
                    close_clause
                ))
            }
            Target::Node(node) => {
                // For specific nodes, just return the node
                match node {
                    Term::NamedNode(nn) => {
                        Ok(format!("SELECT (<{}> AS ?target) WHERE {{ }}", nn.as_str()))
                    }
                    Term::BlankNode(bn) => {
                        Ok(format!("SELECT (?{} AS ?target) WHERE {{ }}", bn.as_str()))
                    }
                    Term::Literal(lit) => Ok(format!("SELECT ({} AS ?target) WHERE {{ }}", lit)),
                    _ => Err(ShaclError::ValidationEngine(
                        "Unsupported term type for node target".to_string(),
                    )),
                }
            }
            Target::ObjectsOf(property) => {
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{}> {{", graph)
                } else {
                    String::new()
                };

                let close_clause = if graph_name.is_some() { " }" } else { "" };

                Ok(format!(
                    "SELECT DISTINCT ?target WHERE {{ {} ?s <{}> ?target .{} }}",
                    graph_clause,
                    property.as_str(),
                    close_clause
                ))
            }
            Target::SubjectsOf(property) => {
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{}> {{", graph)
                } else {
                    String::new()
                };

                let close_clause = if graph_name.is_some() { " }" } else { "" };

                Ok(format!(
                    "SELECT DISTINCT ?target WHERE {{ {} ?target <{}> ?o .{} }}",
                    graph_clause,
                    property.as_str(),
                    close_clause
                ))
            }
            Target::Sparql(sparql_target) => {
                // Return the user-provided SPARQL query with optional prefixes
                let mut query = String::new();
                if let Some(prefixes) = &sparql_target.prefixes {
                    query.push_str(prefixes);
                    query.push('\n');
                }
                query.push_str(&sparql_target.query);
                Ok(query)
            }
            Target::Implicit(class_iri) => {
                // Same as class target
                let graph_clause = if let Some(graph) = graph_name {
                    format!("GRAPH <{}> {{", graph)
                } else {
                    String::new()
                };

                let close_clause = if graph_name.is_some() { " }" } else { "" };

                Ok(format!(
                    "SELECT DISTINCT ?target WHERE {{ {} ?target <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{}> .{} }}",
                    graph_clause,
                    class_iri.as_str(),
                    close_clause
                ))
            }

            // Complex target combinations
            Target::Union(union_target) => {
                self.generate_union_query(&union_target.targets, graph_name)
            }

            Target::Intersection(intersection_target) => {
                self.generate_intersection_query(&intersection_target.targets, graph_name)
            }

            Target::Difference(difference_target) => self.generate_difference_query(
                &difference_target.primary_target,
                &difference_target.exclusion_target,
                graph_name,
            ),

            Target::Conditional(conditional_target) => self.generate_conditional_query(
                &conditional_target.base_target,
                &conditional_target.condition,
                conditional_target.context.as_ref(),
                graph_name,
            ),

            Target::Hierarchical(hierarchical_target) => self.generate_hierarchical_query(
                &hierarchical_target.root_target,
                &hierarchical_target.relationship,
                hierarchical_target.max_depth,
                hierarchical_target.include_intermediate,
                graph_name,
            ),

            Target::PathBased(path_target) => self.generate_path_based_query(
                &path_target.start_target,
                &path_target.path,
                &path_target.direction,
                &path_target.filters,
                graph_name,
            ),
        }
    }

    /// Optimize SPARQL query for target selection with index hints
    pub fn optimize_target_query(&mut self, query: &str, target: &Target) -> Result<String> {
        if !self.optimization_config.enable_query_optimization {
            return Ok(query.to_string());
        }

        let mut optimized_query = query.to_string();

        // Add index hints based on target type
        match target {
            Target::Class(_) | Target::Implicit(_) => {
                // Add hint for type index if available
                if self.should_use_index_hint("type_index") {
                    optimized_query = format!("# Use type index\n{}", optimized_query);
                }
            }
            Target::ObjectsOf(_) => {
                // Add hint for predicate index
                if self.should_use_index_hint("predicate_index") {
                    optimized_query = format!("# Use predicate index\n{}", optimized_query);
                }
            }
            Target::SubjectsOf(_) => {
                // Add hint for subject index
                if self.should_use_index_hint("subject_index") {
                    optimized_query = format!("# Use subject index\n{}", optimized_query);
                }
            }
            _ => {} // No specific optimization for other types
        }

        // Add LIMIT clause if max_results is configured
        if self.optimization_config.parallel_threshold > 0 {
            if !optimized_query.to_uppercase().contains("LIMIT") {
                optimized_query = format!(
                    "{} LIMIT {}",
                    optimized_query, self.optimization_config.parallel_threshold
                );
            }
        }

        Ok(optimized_query)
    }

    /// Check if index hint should be used based on statistics
    fn should_use_index_hint(&self, index_type: &str) -> bool {
        if let Some(stats) = self.index_usage_stats.get(index_type) {
            stats.avg_performance_gain > self.optimization_config.index_hint_threshold
        } else {
            true // Default to using index hints if no statistics available
        }
    }

    /// Select all target nodes for a given target definition
    pub fn select_targets(
        &mut self,
        store: &dyn Store,
        target: &Target,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let start_time = std::time::Instant::now();
        let cache_key = self.create_cache_key(target, graph_name);

        // Check cache first
        if self.optimization_config.enable_caching {
            if let Some(cached_result) = self.cache.get(&cache_key) {
                let cache_age = cached_result.cached_at.elapsed();
                if cache_age.as_secs() <= self.optimization_config.cache_ttl {
                    // Clone the nodes before updating statistics to avoid borrow conflict
                    let nodes: Vec<_> = cached_result.nodes.iter().cloned().collect();

                    // Update statistics
                    self.stats.total_evaluations += 1;
                    self.record_cache_hit();

                    return Ok(nodes);
                }
            }
        }

        // Record cache miss
        self.record_cache_miss();

        // Choose execution strategy based on target type and optimization settings
        let result = match target {
            Target::Sparql(_) => {
                // Use advanced SPARQL optimization for SPARQL targets
                self.execute_sparql_target_optimized(store, target, graph_name)?
            }
            _ => {
                // Use direct store access for other target types
                self.execute_target_selection_direct(store, target, graph_name)?
            }
        };

        // Cache the result if beneficial
        if self.optimization_config.enable_caching && self.should_cache_result(&result) {
            let cached_result = CachedTargetResult {
                nodes: result.iter().cloned().collect(),
                cached_at: std::time::Instant::now(),
                stats: CacheStats {
                    hits: 0,
                    misses: 1,
                    avg_query_time: start_time.elapsed(),
                },
            };

            // Manage cache size with intelligent eviction
            self.manage_cache_size(&cache_key, cached_result);
        }

        // Update statistics
        self.update_execution_statistics(start_time.elapsed(), result.len());

        Ok(result)
    }

    /// Execute SPARQL target with advanced optimizations
    fn execute_sparql_target_optimized(
        &mut self,
        store: &dyn Store,
        target: &Target,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        if let Target::Sparql(sparql_target) = target {
            // Get or create optimized query plan
            let query_plan = self.get_or_create_query_plan(&sparql_target.query)?;

            // Choose execution strategy based on estimated cardinality
            match query_plan.execution_strategy {
                ExecutionStrategy::Sequential => {
                    self.execute_sparql_sequential(store, &query_plan.optimized_query, graph_name)
                }
                ExecutionStrategy::Parallel => {
                    self.execute_sparql_parallel(store, &query_plan.optimized_query, graph_name)
                }
                ExecutionStrategy::IndexDriven => {
                    self.execute_sparql_index_driven(store, &query_plan.optimized_query, graph_name)
                }
                ExecutionStrategy::Hybrid => {
                    self.execute_sparql_hybrid(store, &query_plan.optimized_query, graph_name)
                }
            }
        } else {
            Err(ShaclError::TargetSelection(
                "Expected SPARQL target".to_string(),
            ))
        }
    }

    /// Get or create optimized query plan for SPARQL query
    fn get_or_create_query_plan(&mut self, query: &str) -> Result<QueryPlan> {
        // Check cache first
        if let Some(plan) = self.query_plan_cache.get(query) {
            // Check if plan is still fresh (older than 1 hour)
            if plan.created_at.elapsed().as_secs() < 3600 {
                return Ok(plan.clone());
            }
        }

        // Create new optimized query plan
        let plan = self.create_optimized_query_plan(query)?;
        self.query_plan_cache
            .insert(query.to_string(), plan.clone());

        Ok(plan)
    }

    /// Create optimized query plan with cost estimation
    fn create_optimized_query_plan(&self, query: &str) -> Result<QueryPlan> {
        let mut optimized_query = query.to_string();
        let mut index_hints = Vec::new();

        // Analyze query patterns for optimization opportunities
        let estimated_cardinality = self.estimate_query_cardinality(query);

        // Add DISTINCT if not present to avoid duplicates
        if !query.to_uppercase().contains("DISTINCT") && query.to_uppercase().contains("SELECT") {
            optimized_query = optimized_query.replace("SELECT", "SELECT DISTINCT");
        }

        // Add index hints based on query patterns
        if query.contains("rdf:type") || query.contains("a ") {
            index_hints.push(IndexHint {
                index_type: "type_index".to_string(),
                selectivity: 0.1,
                cost_benefit: 0.8,
            });
        }

        // Determine execution strategy
        let execution_strategy =
            if estimated_cardinality > self.optimization_config.parallel_threshold {
                if self.optimization_config.enable_adaptive_optimization {
                    ExecutionStrategy::Hybrid
                } else {
                    ExecutionStrategy::Parallel
                }
            } else if !index_hints.is_empty() {
                ExecutionStrategy::IndexDriven
            } else {
                ExecutionStrategy::Sequential
            };

        // Add performance hints
        if self.optimization_config.enable_adaptive_optimization {
            optimized_query = format!(
                "# Query plan: {:?}, Est. cardinality: {}\n{}",
                execution_strategy, estimated_cardinality, optimized_query
            );
        }

        Ok(QueryPlan {
            optimized_query,
            estimated_cardinality,
            index_hints,
            execution_strategy,
            created_at: std::time::Instant::now(),
        })
    }

    /// Estimate query cardinality based on patterns
    fn estimate_query_cardinality(&self, query: &str) -> usize {
        // Simple heuristic-based estimation
        let mut estimated_cardinality = 1000; // Default estimate

        // Adjust based on query patterns
        if query.contains("rdf:type") {
            estimated_cardinality *= 10; // Type queries tend to be larger
        }

        if query.contains("OPTIONAL") {
            estimated_cardinality = (estimated_cardinality as f64 * 1.5) as usize;
        }

        if query.contains("UNION") {
            estimated_cardinality *= 2;
        }

        // Limit-based adjustment
        if let Some(limit_match) = query.to_uppercase().find("LIMIT") {
            let limit_part = &query[limit_match + 5..];
            if let Some(number) = limit_part.split_whitespace().next() {
                if let Ok(limit) = number.parse::<usize>() {
                    estimated_cardinality = estimated_cardinality.min(limit);
                }
            }
        }

        estimated_cardinality
    }

    /// Execute SPARQL query with sequential strategy
    fn execute_sparql_sequential(
        &self,
        store: &dyn Store,
        query: &str,
        _graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        // For now, return empty result as SPARQL execution would require
        // integration with oxirs-arq or similar SPARQL engine
        tracing::debug!("Executing SPARQL target query sequentially: {}", query);
        Ok(Vec::new())
    }

    /// Execute SPARQL query with parallel strategy
    fn execute_sparql_parallel(
        &self,
        store: &dyn Store,
        query: &str,
        _graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        // For now, return empty result as parallel SPARQL execution would require
        // advanced integration with the SPARQL engine
        tracing::debug!("Executing SPARQL target query in parallel: {}", query);
        Ok(Vec::new())
    }

    /// Execute SPARQL query with index-driven strategy
    fn execute_sparql_index_driven(
        &self,
        store: &dyn Store,
        query: &str,
        _graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        // For now, return empty result as index-driven execution would require
        // specific index optimization in the SPARQL engine
        tracing::debug!(
            "Executing SPARQL target query with index optimization: {}",
            query
        );
        Ok(Vec::new())
    }

    /// Execute SPARQL query with hybrid strategy
    fn execute_sparql_hybrid(
        &self,
        store: &dyn Store,
        query: &str,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        // Hybrid strategy: start with index-driven, fall back to parallel if needed
        let start_time = std::time::Instant::now();

        // Try index-driven first
        let result = self.execute_sparql_index_driven(store, query, graph_name)?;

        // If taking too long or no results, try parallel approach
        if start_time.elapsed().as_millis() > 1000 && result.is_empty() {
            tracing::debug!("Falling back to parallel execution for SPARQL target");
            return self.execute_sparql_parallel(store, query, graph_name);
        }

        Ok(result)
    }

    /// Check if result should be cached based on size and performance characteristics
    fn should_cache_result(&self, result: &[Term]) -> bool {
        // Don't cache very large results that might consume too much memory
        if result.len() > 10000 {
            return false;
        }

        // Don't cache empty results unless configuration allows it
        if result.is_empty() {
            return false;
        }

        // Cache results of medium size that are likely to be reused
        result.len() > 10 && result.len() < 1000
    }

    /// Manage cache size with intelligent eviction strategies
    fn manage_cache_size(&mut self, new_key: &str, new_result: CachedTargetResult) {
        if self.cache.len() >= self.optimization_config.max_cache_size {
            // Advanced eviction strategy: remove least valuable entries
            let mut removal_candidates = Vec::new();

            for (key, cached) in &self.cache {
                let age = cached.cached_at.elapsed().as_secs();
                let hit_rate = if cached.stats.hits + cached.stats.misses > 0 {
                    cached.stats.hits as f64 / (cached.stats.hits + cached.stats.misses) as f64
                } else {
                    0.0
                };

                // Score based on age, hit rate, and result size
                let score = (age as f64) * 0.5
                    + (1.0 - hit_rate) * 0.3
                    + (cached.nodes.len() as f64 * 0.01);
                removal_candidates.push((key.clone(), score));
            }

            // Sort by score (higher score = more likely to remove)
            removal_candidates
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Remove the worst candidate
            if let Some((key_to_remove, _)) = removal_candidates.first() {
                self.cache.remove(key_to_remove);
            }
        }

        self.cache.insert(new_key.to_string(), new_result);
    }

    /// Update execution statistics
    fn update_execution_statistics(&mut self, duration: std::time::Duration, result_count: usize) {
        self.stats.total_evaluations += 1;
        self.stats.total_time += duration;

        // Update average evaluation time
        if self.stats.total_evaluations > 0 {
            self.stats.avg_evaluation_time =
                self.stats.total_time / self.stats.total_evaluations as u32;
        }

        // Update adaptive thresholds based on performance
        if self.optimization_config.enable_adaptive_optimization {
            self.update_adaptive_thresholds(duration, result_count);
        }
    }

    /// Update adaptive optimization thresholds based on performance data
    fn update_adaptive_thresholds(&mut self, duration: std::time::Duration, result_count: usize) {
        // Adjust parallel threshold based on performance
        if duration.as_millis() > 500
            && result_count > self.optimization_config.parallel_threshold / 2
        {
            // Lower threshold if we're seeing slow performance on medium-sized results
            self.optimization_config.parallel_threshold =
                (self.optimization_config.parallel_threshold * 8) / 10;
        } else if duration.as_millis() < 100
            && result_count > self.optimization_config.parallel_threshold
        {
            // Raise threshold if we're handling large results quickly
            self.optimization_config.parallel_threshold =
                (self.optimization_config.parallel_threshold * 12) / 10;
        }

        // Adjust cache TTL based on hit patterns
        if self.stats.cache_hit_rate > 0.8 {
            // High hit rate, can afford longer TTL
            self.optimization_config.cache_ttl = (self.optimization_config.cache_ttl * 12) / 10;
        } else if self.stats.cache_hit_rate < 0.3 {
            // Low hit rate, reduce TTL to freshen cache more often
            self.optimization_config.cache_ttl = (self.optimization_config.cache_ttl * 8) / 10;
        }
    }

    /// Record cache hit
    fn record_cache_hit(&mut self) {
        self.update_cache_hit_rate();
    }

    /// Record cache miss  
    fn record_cache_miss(&mut self) {
        self.update_cache_hit_rate();
    }

    /// Execute target selection using direct store access instead of SPARQL due to oxirs-core limitations
    fn execute_target_query(
        &self,
        store: &dyn Store,
        _query: &str,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        // We'll implement direct store access based on the target type
        // This bypasses the SPARQL parser limitations with long IRIs
        Ok(Vec::new()) // Placeholder - will be implemented per target type
    }

    /// Execute target selection using direct store operations
    fn execute_target_selection_direct(
        &self,
        store: &dyn Store,
        target: &Target,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        use oxirs_core::model::{GraphName, NamedNode as CoreNamedNode, Object, Predicate, Quad};

        match target {
            Target::Class(class_iri) => {
                // Find all subjects that have rdf:type = class_iri
                let rdf_type = Predicate::NamedNode(
                    CoreNamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").map_err(
                        |e| ShaclError::TargetSelection(format!("Invalid RDF type IRI: {}", e)),
                    )?,
                );

                let mut target_nodes = Vec::new();

                // Iterate through all quads in the store
                for quad in store.find_quads(None, None, None, None)? {
                    // Check if this is a type triple and matches our target class
                    if quad.predicate() == &rdf_type
                        && matches!(quad.object(), Object::NamedNode(obj) if obj.as_str() == class_iri.as_str())
                    {
                        // Check graph name if specified
                        if let Some(graph) = graph_name {
                            if let GraphName::NamedNode(quad_graph) = quad.graph_name() {
                                if quad_graph.as_str() != graph {
                                    continue;
                                }
                            } else if !matches!(quad.graph_name(), GraphName::DefaultGraph) {
                                continue;
                            }
                        }

                        // Add the subject as a target node
                        match quad.subject() {
                            oxirs_core::model::Subject::NamedNode(node) => {
                                target_nodes.push(Term::NamedNode(node.clone()));
                            }
                            oxirs_core::model::Subject::BlankNode(blank) => {
                                target_nodes.push(Term::BlankNode(blank.clone()));
                            }
                            oxirs_core::model::Subject::Variable(var) => {
                                target_nodes.push(Term::Variable(var.clone()));
                            }
                            oxirs_core::model::Subject::QuotedTriple(qt) => {
                                target_nodes.push(Term::QuotedTriple(qt.clone()));
                            }
                        }
                    }
                }

                // Remove duplicates and sort for deterministic results
                self.sort_and_dedupe_targets(&mut target_nodes);

                tracing::debug!(
                    "Found {} target nodes for class {}",
                    target_nodes.len(),
                    class_iri.as_str()
                );
                Ok(target_nodes)
            }
            Target::Node(node) => {
                // For specific nodes, just return the node itself
                Ok(vec![node.clone()])
            }
            Target::ObjectsOf(property) => {
                let mut target_nodes = Vec::new();

                // Find all objects of the specified property
                for quad in store.find_quads(None, None, None, None)? {
                    if matches!(quad.predicate(), Predicate::NamedNode(pred) if pred.as_str() == property.as_str())
                    {
                        // Check graph name if specified
                        if let Some(graph) = graph_name {
                            if let GraphName::NamedNode(quad_graph) = quad.graph_name() {
                                if quad_graph.as_str() != graph {
                                    continue;
                                }
                            } else if !matches!(quad.graph_name(), GraphName::DefaultGraph) {
                                continue;
                            }
                        }

                        target_nodes.push(quad.object().clone().into());
                    }
                }

                self.sort_and_dedupe_targets(&mut target_nodes);

                tracing::debug!(
                    "Found {} target nodes for objectsOf {}",
                    target_nodes.len(),
                    property.as_str()
                );
                Ok(target_nodes)
            }
            Target::SubjectsOf(property) => {
                let mut target_nodes = Vec::new();

                // Find all subjects of the specified property
                for quad in store.find_quads(None, None, None, None)? {
                    if matches!(quad.predicate(), Predicate::NamedNode(pred) if pred.as_str() == property.as_str())
                    {
                        // Check graph name if specified
                        if let Some(graph) = graph_name {
                            if let GraphName::NamedNode(quad_graph) = quad.graph_name() {
                                if quad_graph.as_str() != graph {
                                    continue;
                                }
                            } else if !matches!(quad.graph_name(), GraphName::DefaultGraph) {
                                continue;
                            }
                        }

                        // Add the subject as a target node
                        match quad.subject() {
                            oxirs_core::model::Subject::NamedNode(node) => {
                                target_nodes.push(Term::NamedNode(node.clone()));
                            }
                            oxirs_core::model::Subject::BlankNode(blank) => {
                                target_nodes.push(Term::BlankNode(blank.clone()));
                            }
                            oxirs_core::model::Subject::Variable(var) => {
                                target_nodes.push(Term::Variable(var.clone()));
                            }
                            oxirs_core::model::Subject::QuotedTriple(qt) => {
                                target_nodes.push(Term::QuotedTriple(qt.clone()));
                            }
                        }
                    }
                }

                self.sort_and_dedupe_targets(&mut target_nodes);

                tracing::debug!(
                    "Found {} target nodes for subjectsOf {}",
                    target_nodes.len(),
                    property.as_str()
                );
                Ok(target_nodes)
            }
            Target::Sparql(_sparql_target) => {
                // For SPARQL targets, we still need to use the query engine
                // This might fail due to oxirs-core limitations, but let's try
                let query = self.generate_target_query(target, graph_name)?;
                self.execute_sparql_target_query(store, &query)
            }
            Target::Implicit(class_iri) => {
                // Same as class target
                let class_target = Target::Class(class_iri.clone());
                self.execute_target_selection_direct(store, &class_target, graph_name)
            }

            // Complex target combinations
            Target::Union(union_target) => {
                self.execute_union_target(store, &union_target.targets, graph_name)
            }

            Target::Intersection(intersection_target) => {
                self.execute_intersection_target(store, &intersection_target.targets, graph_name)
            }

            Target::Difference(difference_target) => self.execute_difference_target(
                store,
                &difference_target.primary_target,
                &difference_target.exclusion_target,
                graph_name,
            ),

            Target::Conditional(conditional_target) => self.execute_conditional_target(
                store,
                &conditional_target.base_target,
                &conditional_target.condition,
                conditional_target.context.as_ref(),
                graph_name,
            ),

            Target::Hierarchical(hierarchical_target) => self.execute_hierarchical_target(
                store,
                &hierarchical_target.root_target,
                &hierarchical_target.relationship,
                hierarchical_target.max_depth,
                hierarchical_target.include_intermediate,
                graph_name,
            ),

            Target::PathBased(path_target) => self.execute_path_based_target(
                store,
                &path_target.start_target,
                &path_target.path,
                &path_target.direction,
                &path_target.filters,
                graph_name,
            ),
        }
    }

    /// Execute SPARQL target query (may fail due to oxirs-core limitations)
    fn execute_sparql_target_query(&self, store: &dyn Store, query: &str) -> Result<Vec<Term>> {
        use oxirs_core::query::{QueryEngine, QueryResult};

        tracing::info!("Executing SPARQL target query: '{}'", query);

        let query_engine = QueryEngine::new();

        match query_engine.query(query, store) {
            Ok(QueryResult::Select {
                variables: _,
                bindings,
            }) => {
                let mut target_nodes = Vec::new();

                for binding in bindings {
                    // Look for ?this variable or first variable
                    if let Some(term) = binding.get("this").or_else(|| binding.values().next()) {
                        target_nodes.push(term.clone());
                    }
                }

                self.sort_and_dedupe_targets(&mut target_nodes);
                Ok(target_nodes)
            }
            Ok(_) => Err(ShaclError::SparqlExecution(
                "SPARQL target query must return SELECT results".to_string(),
            )),
            Err(e) => {
                tracing::error!("SPARQL target query execution failed: {}", e);
                Err(ShaclError::SparqlExecution(format!(
                    "SPARQL target query execution failed: {}",
                    e
                )))
            }
        }
    }

    /// Sort and remove duplicate target nodes
    fn sort_and_dedupe_targets(&self, target_nodes: &mut Vec<Term>) {
        target_nodes.sort_by(|a, b| match (a, b) {
            (Term::NamedNode(a_node), Term::NamedNode(b_node)) => {
                a_node.as_str().cmp(b_node.as_str())
            }
            (Term::BlankNode(a_blank), Term::BlankNode(b_blank)) => {
                a_blank.as_str().cmp(b_blank.as_str())
            }
            (Term::Literal(a_lit), Term::Literal(b_lit)) => a_lit.as_str().cmp(b_lit.as_str()),
            (Term::NamedNode(_), _) => std::cmp::Ordering::Less,
            (Term::BlankNode(_), Term::NamedNode(_)) => std::cmp::Ordering::Greater,
            (Term::BlankNode(_), _) => std::cmp::Ordering::Less,
            (Term::Literal(_), Term::NamedNode(_)) => std::cmp::Ordering::Greater,
            (Term::Literal(_), Term::BlankNode(_)) => std::cmp::Ordering::Greater,
            _ => std::cmp::Ordering::Equal,
        });
        target_nodes.dedup();
    }

    /// Create cache key for target and graph combination
    fn create_cache_key(&self, target: &Target, graph_name: Option<&str>) -> String {
        match target {
            Target::Class(class_iri) => {
                format!(
                    "class:{}:{}",
                    class_iri.as_str(),
                    graph_name.unwrap_or("default")
                )
            }
            Target::Node(node) => {
                format!(
                    "node:{}:{}",
                    format!("{:?}", node),
                    graph_name.unwrap_or("default")
                )
            }
            Target::ObjectsOf(property) => {
                format!(
                    "objects_of:{}:{}",
                    property.as_str(),
                    graph_name.unwrap_or("default")
                )
            }
            Target::SubjectsOf(property) => {
                format!(
                    "subjects_of:{}:{}",
                    property.as_str(),
                    graph_name.unwrap_or("default")
                )
            }
            Target::Sparql(sparql_target) => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                sparql_target.query.hash(&mut hasher);
                let query_hash = hasher.finish();

                format!("sparql:{}:{}", query_hash, graph_name.unwrap_or("default"))
            }
            Target::Implicit(class_iri) => {
                format!(
                    "implicit:{}:{}",
                    class_iri.as_str(),
                    graph_name.unwrap_or("default")
                )
            }
            Target::Union(union_target) => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                for target in &union_target.targets {
                    self.create_cache_key(target, graph_name).hash(&mut hasher);
                }
                let union_hash = hasher.finish();
                format!("union:{}:{}", union_hash, graph_name.unwrap_or("default"))
            }
            Target::Intersection(intersection_target) => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                for target in &intersection_target.targets {
                    self.create_cache_key(target, graph_name).hash(&mut hasher);
                }
                let intersection_hash = hasher.finish();
                format!(
                    "intersection:{}:{}",
                    intersection_hash,
                    graph_name.unwrap_or("default")
                )
            }
            Target::Difference(difference_target) => {
                let primary_key =
                    self.create_cache_key(&difference_target.primary_target, graph_name);
                let exclusion_key =
                    self.create_cache_key(&difference_target.exclusion_target, graph_name);
                format!(
                    "difference:{}:{}:{}",
                    primary_key,
                    exclusion_key,
                    graph_name.unwrap_or("default")
                )
            }
            Target::Conditional(conditional_target) => {
                let base_key = self.create_cache_key(&conditional_target.base_target, graph_name);
                format!(
                    "conditional:{}:{}",
                    base_key,
                    graph_name.unwrap_or("default")
                )
            }
            Target::Hierarchical(hierarchical_target) => {
                let root_key = self.create_cache_key(&hierarchical_target.root_target, graph_name);
                format!(
                    "hierarchical:{}:{}",
                    root_key,
                    graph_name.unwrap_or("default")
                )
            }
            Target::PathBased(path_based_target) => {
                let start_key = self.create_cache_key(&path_based_target.start_target, graph_name);
                format!(
                    "path_based:{}:{}",
                    start_key,
                    graph_name.unwrap_or("default")
                )
            }
        }
    }

    /// Find the oldest cache entry for eviction
    fn find_oldest_cache_entry(&self) -> Option<String> {
        self.cache
            .iter()
            .min_by_key(|(_, cached_result)| cached_result.cached_at)
            .map(|(key, _)| key.clone())
    }

    /// Update cache hit rate statistics
    fn update_cache_hit_rate(&mut self) {
        if self.stats.total_evaluations > 0 {
            let total_cache_operations = self
                .cache
                .values()
                .map(|cached_result| cached_result.stats.hits + cached_result.stats.misses)
                .sum::<usize>();

            let total_hits = self
                .cache
                .values()
                .map(|cached_result| cached_result.stats.hits)
                .sum::<usize>();

            if total_cache_operations > 0 {
                self.stats.cache_hit_rate = total_hits as f64 / total_cache_operations as f64;
            }
        }
    }

    /// Get current target selection statistics
    pub fn get_statistics(&self) -> &TargetSelectionStats {
        &self.stats
    }

    /// Get cache statistics
    pub fn get_cache_statistics(&self) -> TargetCacheStats {
        let total_hits = self
            .cache
            .values()
            .map(|cached_result| cached_result.stats.hits)
            .sum::<usize>();

        let total_operations = self
            .cache
            .values()
            .map(|cached_result| cached_result.stats.hits + cached_result.stats.misses)
            .sum::<usize>();

        let hit_rate = if total_operations > 0 {
            total_hits as f64 / total_operations as f64
        } else {
            0.0
        };

        // Estimate memory usage (rough calculation)
        let memory_usage = self.cache.len() * 1024; // Rough estimate: 1KB per cache entry

        TargetCacheStats {
            hits: total_hits,
            misses: total_operations - total_hits,
            hit_rate,
            cache_size: self.cache.len(),
            memory_usage_bytes: memory_usage,
        }
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Update index usage statistics for adaptive optimization
    pub fn update_index_usage_stats(&mut self, index_type: &str, performance_gain: f64) {
        let stats = self
            .index_usage_stats
            .entry(index_type.to_string())
            .or_insert_with(|| IndexUsageStats {
                usage_count: 0,
                avg_performance_gain: 0.0,
                last_used: std::time::Instant::now(),
            });

        stats.usage_count += 1;
        stats.avg_performance_gain = (stats.avg_performance_gain * (stats.usage_count - 1) as f64
            + performance_gain)
            / stats.usage_count as f64;
        stats.last_used = std::time::Instant::now();
    }

    /// Generate optimized SPARQL query with advanced features
    pub fn generate_optimized_target_query(
        &mut self,
        target: &Target,
        graph_name: Option<&str>,
        query_options: &QueryOptimizationOptions,
    ) -> Result<OptimizedQuery> {
        let start_time = std::time::Instant::now();

        // Check if we have a cached query plan
        let cache_key = self.create_query_plan_cache_key(target, graph_name, query_options);
        if let Some(cached_plan) = self.query_plan_cache.get(&cache_key) {
            if cached_plan.created_at.elapsed() < std::time::Duration::from_secs(300) {
                return Ok(OptimizedQuery {
                    sparql: cached_plan.optimized_query.clone(),
                    estimated_cardinality: cached_plan.estimated_cardinality,
                    execution_strategy: cached_plan.execution_strategy.clone(),
                    index_hints: cached_plan.index_hints.clone(),
                    optimization_time: start_time.elapsed(),
                });
            }
        }

        // Generate base query
        let base_query = self.generate_target_query(target, graph_name)?;

        // Apply optimizations
        let optimized_query = self.apply_query_optimizations(&base_query, target, query_options)?;

        // Create execution plan
        let execution_plan = self.create_execution_plan(target, query_options)?;

        // Estimate cardinality
        let estimated_cardinality = self.estimate_target_cardinality(target, query_options);

        // Cache the query plan
        let query_plan = QueryPlan {
            optimized_query: optimized_query.clone(),
            estimated_cardinality,
            index_hints: execution_plan.index_hints.clone(),
            execution_strategy: execution_plan.execution_strategy.clone(),
            created_at: std::time::Instant::now(),
        };

        if self.query_plan_cache.len() < 100 {
            self.query_plan_cache.insert(cache_key, query_plan);
        }

        Ok(OptimizedQuery {
            sparql: optimized_query,
            estimated_cardinality,
            execution_strategy: execution_plan.execution_strategy,
            index_hints: execution_plan.index_hints,
            optimization_time: start_time.elapsed(),
        })
    }

    /// Apply query optimizations based on target type and options
    fn apply_query_optimizations(
        &self,
        base_query: &str,
        target: &Target,
        options: &QueryOptimizationOptions,
    ) -> Result<String> {
        let mut optimized_query = base_query.to_string();

        // Add LIMIT if specified
        if let Some(limit) = options.limit {
            if !optimized_query.contains("LIMIT") {
                optimized_query.push_str(&format!(" LIMIT {}", limit));
            }
        }

        // Add ORDER BY for deterministic results if requested
        if options.deterministic_results {
            if !optimized_query.contains("ORDER BY") {
                optimized_query =
                    optimized_query.replace("SELECT DISTINCT ?target", "SELECT DISTINCT ?target");
                optimized_query.push_str(" ORDER BY ?target");
            }
        }

        // Add index hints based on target type
        if options.use_index_hints {
            optimized_query = self.add_index_hints(&optimized_query, target)?;
        }

        // Add performance monitoring hints if enabled
        if options.include_performance_hints {
            optimized_query = format!(
                "# Generated: {}\n{}",
                chrono::Utc::now().to_rfc3339(),
                optimized_query
            );
        }

        Ok(optimized_query)
    }

    /// Add index hints to the query based on target type
    fn add_index_hints(&self, query: &str, target: &Target) -> Result<String> {
        let mut optimized_query = query.to_string();

        match target {
            Target::Class(_) | Target::Implicit(_) => {
                if self.should_use_index_hint("type_index") {
                    optimized_query = format!(
                        "# HINT: Use type index for rdf:type lookups\n{}",
                        optimized_query
                    );
                }
            }
            Target::ObjectsOf(property) => {
                if self.should_use_index_hint("object_index") {
                    optimized_query = format!(
                        "# HINT: Use object index for property <{}>\n{}",
                        property.as_str(),
                        optimized_query
                    );
                }
            }
            Target::SubjectsOf(property) => {
                if self.should_use_index_hint("subject_index") {
                    optimized_query = format!(
                        "# HINT: Use subject index for property <{}>\n{}",
                        property.as_str(),
                        optimized_query
                    );
                }
            }
            _ => {}
        }

        Ok(optimized_query)
    }

    /// Create execution plan for target selection
    fn create_execution_plan(
        &self,
        target: &Target,
        options: &QueryOptimizationOptions,
    ) -> Result<ExecutionPlan> {
        let estimated_cardinality = self.estimate_target_cardinality(target, options);

        // Choose execution strategy based on estimated cardinality
        let execution_strategy =
            if estimated_cardinality > self.optimization_config.parallel_threshold {
                ExecutionStrategy::Parallel
            } else if self.should_use_index_strategy(target) {
                ExecutionStrategy::IndexDriven
            } else {
                ExecutionStrategy::Sequential
            };

        // Generate index hints
        let index_hints = self.generate_index_hints(target)?;

        Ok(ExecutionPlan {
            execution_strategy,
            index_hints,
            estimated_cardinality,
        })
    }

    /// Estimate cardinality for target selection
    fn estimate_target_cardinality(
        &self,
        target: &Target,
        _options: &QueryOptimizationOptions,
    ) -> usize {
        match target {
            Target::Node(_) => 1, // Single node
            Target::Class(_) | Target::Implicit(_) => {
                // Estimate based on typical class sizes
                1000 // Default estimate
            }
            Target::ObjectsOf(_) | Target::SubjectsOf(_) => {
                // Estimate based on property usage
                500 // Default estimate
            }
            Target::Sparql(_) => {
                // Can't estimate SPARQL queries easily
                100 // Conservative estimate
            }
            Target::Union(union_target) => {
                // Sum estimates for all targets in union
                union_target
                    .targets
                    .iter()
                    .map(|t| self.estimate_target_cardinality(t, _options))
                    .sum()
            }
            Target::Intersection(intersection_target) => {
                // Take minimum estimate for intersection
                intersection_target
                    .targets
                    .iter()
                    .map(|t| self.estimate_target_cardinality(t, _options))
                    .min()
                    .unwrap_or(0)
            }
            Target::Difference(difference_target) => {
                // Estimate as primary minus exclusion (but at least 0)
                let primary =
                    self.estimate_target_cardinality(&difference_target.primary_target, _options);
                let exclusion =
                    self.estimate_target_cardinality(&difference_target.exclusion_target, _options);
                primary.saturating_sub(exclusion)
            }
            Target::Conditional(conditional_target) => {
                // Estimate as base target with condition filtering (assume 50% selectivity)
                let base =
                    self.estimate_target_cardinality(&conditional_target.base_target, _options);
                base / 2
            }
            Target::Hierarchical(hierarchical_target) => {
                // Estimate based on hierarchy depth and branching factor
                let root =
                    self.estimate_target_cardinality(&hierarchical_target.root_target, _options);
                let depth = hierarchical_target.max_depth.max(1) as usize;
                root * depth // Simple estimate
            }
            Target::PathBased(path_based_target) => {
                // Estimate based on path traversal (assume moderate expansion)
                let start =
                    self.estimate_target_cardinality(&path_based_target.start_target, _options);
                start * 2 // Assume 2x expansion on average
            }
        }
    }

    /// Generate index hints for target type
    fn generate_index_hints(&self, target: &Target) -> Result<Vec<IndexHint>> {
        let mut hints = Vec::new();

        match target {
            Target::Class(_) | Target::Implicit(_) => {
                hints.push(IndexHint {
                    index_type: "type_index".to_string(),
                    selectivity: 0.1, // Assume 10% selectivity
                    cost_benefit: 0.8,
                });
            }
            Target::ObjectsOf(property) => {
                hints.push(IndexHint {
                    index_type: format!("object_index_{}", property.as_str()),
                    selectivity: 0.05,
                    cost_benefit: 0.9,
                });
            }
            Target::SubjectsOf(property) => {
                hints.push(IndexHint {
                    index_type: format!("subject_index_{}", property.as_str()),
                    selectivity: 0.05,
                    cost_benefit: 0.9,
                });
            }
            _ => {}
        }

        Ok(hints)
    }

    /// Check if index strategy should be used
    fn should_use_index_strategy(&self, target: &Target) -> bool {
        match target {
            Target::Class(_) | Target::Implicit(_) => self.should_use_index_hint("type_index"),
            Target::ObjectsOf(_) | Target::SubjectsOf(_) => {
                self.should_use_index_hint("property_index")
            }
            _ => false,
        }
    }

    /// Create cache key for query plans
    fn create_query_plan_cache_key(
        &self,
        target: &Target,
        graph_name: Option<&str>,
        options: &QueryOptimizationOptions,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash target
        match target {
            Target::Class(class_iri) => {
                "class".hash(&mut hasher);
                class_iri.as_str().hash(&mut hasher);
            }
            Target::Node(node) => {
                "node".hash(&mut hasher);
                format!("{:?}", node).hash(&mut hasher);
            }
            Target::ObjectsOf(property) => {
                "objects_of".hash(&mut hasher);
                property.as_str().hash(&mut hasher);
            }
            Target::SubjectsOf(property) => {
                "subjects_of".hash(&mut hasher);
                property.as_str().hash(&mut hasher);
            }
            Target::Sparql(sparql_target) => {
                "sparql".hash(&mut hasher);
                sparql_target.query.hash(&mut hasher);
            }
            Target::Implicit(class_iri) => {
                "implicit".hash(&mut hasher);
                class_iri.as_str().hash(&mut hasher);
            }
            Target::Union(union_target) => {
                "union".hash(&mut hasher);
                for target in &union_target.targets {
                    self.create_query_plan_cache_key(target, graph_name, options)
                        .hash(&mut hasher);
                }
            }
            Target::Intersection(intersection_target) => {
                "intersection".hash(&mut hasher);
                for target in &intersection_target.targets {
                    self.create_query_plan_cache_key(target, graph_name, options)
                        .hash(&mut hasher);
                }
            }
            Target::Difference(difference_target) => {
                "difference".hash(&mut hasher);
                self.create_query_plan_cache_key(
                    &difference_target.primary_target,
                    graph_name,
                    options,
                )
                .hash(&mut hasher);
                self.create_query_plan_cache_key(
                    &difference_target.exclusion_target,
                    graph_name,
                    options,
                )
                .hash(&mut hasher);
            }
            Target::Conditional(conditional_target) => {
                "conditional".hash(&mut hasher);
                self.create_query_plan_cache_key(
                    &conditional_target.base_target,
                    graph_name,
                    options,
                )
                .hash(&mut hasher);
            }
            Target::Hierarchical(hierarchical_target) => {
                "hierarchical".hash(&mut hasher);
                self.create_query_plan_cache_key(
                    &hierarchical_target.root_target,
                    graph_name,
                    options,
                )
                .hash(&mut hasher);
                hierarchical_target.max_depth.hash(&mut hasher);
            }
            Target::PathBased(path_based_target) => {
                "path_based".hash(&mut hasher);
                self.create_query_plan_cache_key(
                    &path_based_target.start_target,
                    graph_name,
                    options,
                )
                .hash(&mut hasher);
            }
        }

        // Hash graph name and options
        graph_name.hash(&mut hasher);
        options.limit.hash(&mut hasher);
        options.deterministic_results.hash(&mut hasher);
        options.use_index_hints.hash(&mut hasher);

        format!("plan_{:x}", hasher.finish())
    }

    /// Generate batch target queries for multiple targets
    pub fn generate_batch_target_queries(
        &mut self,
        targets: &[Target],
        graph_name: Option<&str>,
        options: &QueryOptimizationOptions,
    ) -> Result<BatchQueryResult> {
        let start_time = std::time::Instant::now();
        let mut optimized_queries = Vec::new();
        let mut total_estimated_cardinality = 0;

        for target in targets {
            let optimized_query =
                self.generate_optimized_target_query(target, graph_name, options)?;
            total_estimated_cardinality += optimized_query.estimated_cardinality;
            optimized_queries.push(optimized_query);
        }

        // Create union query if beneficial
        let union_query = if optimized_queries.len() > 1 && options.use_union_optimization {
            Some(self.create_union_query(&optimized_queries)?)
        } else {
            None
        };

        Ok(BatchQueryResult {
            individual_queries: optimized_queries,
            union_query,
            total_estimated_cardinality,
            batch_optimization_time: start_time.elapsed(),
        })
    }

    /// Create union query from multiple target queries
    fn create_union_query(&self, queries: &[OptimizedQuery]) -> Result<String> {
        let mut union_parts = Vec::new();

        for query in queries {
            // Extract the WHERE clause from each query
            if let Some(where_start) = query.sparql.find("WHERE {") {
                let where_clause = &query.sparql[where_start + 7..];
                if let Some(where_end) = where_clause.rfind('}') {
                    let where_content = &where_clause[..where_end].trim();
                    union_parts.push(format!("{{ {} }}", where_content));
                }
            }
        }

        if union_parts.is_empty() {
            return Err(ShaclError::ValidationEngine(
                "No valid WHERE clauses found for union query".to_string(),
            ));
        }

        Ok(format!(
            "SELECT DISTINCT ?target WHERE {{\n  {}\n}}",
            union_parts.join("\n  UNION\n  ")
        ))
    }

    /// Generate SPARQL query for union targets
    fn generate_union_query(&self, targets: &[Target], graph_name: Option<&str>) -> Result<String> {
        if targets.is_empty() {
            return Err(ShaclError::ValidationEngine(
                "Union target cannot be empty".to_string(),
            ));
        }

        let mut union_parts = Vec::new();
        for target in targets {
            let target_query = self.generate_target_query(target, graph_name)?;
            // Extract WHERE clause content
            if let Some(where_start) = target_query.find("WHERE {") {
                let where_clause = &target_query[where_start + 7..];
                if let Some(where_end) = where_clause.rfind('}') {
                    let where_content = where_clause[..where_end].trim();
                    union_parts.push(format!("{{ {} }}", where_content));
                }
            }
        }

        Ok(format!(
            "SELECT DISTINCT ?target WHERE {{\n  {}\n}}",
            union_parts.join("\n  UNION\n  ")
        ))
    }

    /// Generate SPARQL query for intersection targets
    fn generate_intersection_query(
        &self,
        targets: &[Target],
        graph_name: Option<&str>,
    ) -> Result<String> {
        if targets.is_empty() {
            return Err(ShaclError::ValidationEngine(
                "Intersection target cannot be empty".to_string(),
            ));
        }

        if targets.len() == 1 {
            return self.generate_target_query(&targets[0], graph_name);
        }

        // Generate subqueries for each target
        let mut subqueries = Vec::new();
        for (i, target) in targets.iter().enumerate() {
            let target_query = self.generate_target_query(target, graph_name)?;
            let var_name = format!("target{}", i);

            // Replace ?target with specific variable
            let modified_query = target_query.replace("?target", &format!("?{}", var_name));
            subqueries.push((var_name, modified_query));
        }

        // Create filter conditions to ensure all variables refer to the same node
        let mut filter_conditions = Vec::new();
        for i in 1..subqueries.len() {
            filter_conditions.push(format!("?target0 = ?target{}", i));
        }

        let filter_clause = if !filter_conditions.is_empty() {
            format!("FILTER({})", filter_conditions.join(" && "))
        } else {
            String::new()
        };

        // Extract WHERE clause content from each subquery
        let mut where_parts = Vec::new();
        for (_, query) in &subqueries {
            if let Some(where_start) = query.find("WHERE {") {
                let where_clause = &query[where_start + 7..];
                if let Some(where_end) = where_clause.rfind('}') {
                    let where_content = where_clause[..where_end].trim();
                    where_parts.push(where_content.to_string());
                }
            }
        }

        Ok(format!(
            "SELECT DISTINCT (?target0 AS ?target) WHERE {{\n  {}\n  {}\n}}",
            where_parts.join("\n  "),
            filter_clause
        ))
    }

    /// Generate SPARQL query for difference targets
    fn generate_difference_query(
        &self,
        primary: &Target,
        exclusion: &Target,
        graph_name: Option<&str>,
    ) -> Result<String> {
        let primary_query = self.generate_target_query(primary, graph_name)?;
        let exclusion_query = self.generate_target_query(exclusion, graph_name)?;

        // Extract WHERE clause content from primary query
        let primary_where = if let Some(where_start) = primary_query.find("WHERE {") {
            let where_clause = &primary_query[where_start + 7..];
            if let Some(where_end) = where_clause.rfind('}') {
                where_clause[..where_end].trim()
            } else {
                return Err(ShaclError::ValidationEngine(
                    "Invalid primary query structure".to_string(),
                ));
            }
        } else {
            return Err(ShaclError::ValidationEngine(
                "Primary query missing WHERE clause".to_string(),
            ));
        };

        // Extract WHERE clause content from exclusion query
        let exclusion_where = if let Some(where_start) = exclusion_query.find("WHERE {") {
            let where_clause = &exclusion_query[where_start + 7..];
            if let Some(where_end) = where_clause.rfind('}') {
                where_clause[..where_end]
                    .trim()
                    .replace("?target", "?excluded")
            } else {
                return Err(ShaclError::ValidationEngine(
                    "Invalid exclusion query structure".to_string(),
                ));
            }
        } else {
            return Err(ShaclError::ValidationEngine(
                "Exclusion query missing WHERE clause".to_string(),
            ));
        };

        Ok(format!(
            "SELECT DISTINCT ?target WHERE {{\n  {}\n  FILTER NOT EXISTS {{\n    {}\n    FILTER(?target = ?excluded)\n  }}\n}}",
            primary_where,
            exclusion_where
        ))
    }

    /// Generate SPARQL query for conditional targets
    fn generate_conditional_query(
        &self,
        base_target: &Target,
        condition: &TargetCondition,
        context: Option<&TargetContext>,
        graph_name: Option<&str>,
    ) -> Result<String> {
        let base_query = self.generate_target_query(base_target, graph_name)?;

        // Extract WHERE clause content from base query
        let base_where = if let Some(where_start) = base_query.find("WHERE {") {
            let where_clause = &base_query[where_start + 7..];
            if let Some(where_end) = where_clause.rfind('}') {
                where_clause[..where_end].trim()
            } else {
                return Err(ShaclError::ValidationEngine(
                    "Invalid base query structure".to_string(),
                ));
            }
        } else {
            return Err(ShaclError::ValidationEngine(
                "Base query missing WHERE clause".to_string(),
            ));
        };

        let condition_clause = match condition {
            TargetCondition::SparqlAsk { query, prefixes: _ } => {
                // Use the ASK query as a filter condition
                format!("FILTER EXISTS {{ {} }}", query)
            }
            TargetCondition::PropertyExists {
                property,
                direction,
            } => match direction {
                PropertyDirection::Subject => {
                    format!("?target <{}> ?value .", property.as_str())
                }
                PropertyDirection::Object => {
                    format!("?subject <{}> ?target .", property.as_str())
                }
                PropertyDirection::Either => {
                    format!(
                        "{{ ?target <{}> ?value . }} UNION {{ ?subject <{}> ?target . }}",
                        property.as_str(),
                        property.as_str()
                    )
                }
            },
            TargetCondition::PropertyValue {
                property,
                value,
                direction,
            } => {
                let value_str = match value {
                    Term::NamedNode(nn) => format!("<{}>", nn.as_str()),
                    Term::Literal(lit) => format!("\"{}\"", lit.value()),
                    _ => format!("{:?}", value),
                };

                match direction {
                    PropertyDirection::Subject => {
                        format!("?target <{}> {} .", property.as_str(), value_str)
                    }
                    PropertyDirection::Object => {
                        format!("{} <{}> ?target .", value_str, property.as_str())
                    }
                    PropertyDirection::Either => {
                        format!(
                            "{{ ?target <{}> {} . }} UNION {{ {} <{}> ?target . }}",
                            property.as_str(),
                            value_str,
                            value_str,
                            property.as_str()
                        )
                    }
                }
            }
            TargetCondition::HasType { class_iri } => {
                format!(
                    "?target <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{}> .",
                    class_iri.as_str()
                )
            }
            TargetCondition::Cardinality {
                property,
                min_count,
                max_count,
                direction,
            } => {
                let base_pattern = match direction {
                    PropertyDirection::Subject => format!("?target <{}> ?value", property.as_str()),
                    PropertyDirection::Object => {
                        format!("?subject <{}> ?target", property.as_str())
                    }
                    PropertyDirection::Either => format!(
                        "{{ ?target <{}> ?value }} UNION {{ ?subject <{}> ?target }}",
                        property.as_str(),
                        property.as_str()
                    ),
                };

                let mut filters = Vec::new();
                if let Some(min) = min_count {
                    filters.push(format!(
                        "(SELECT (COUNT(*) AS ?count) WHERE {{ {} }}) >= {}",
                        base_pattern, min
                    ));
                }
                if let Some(max) = max_count {
                    filters.push(format!(
                        "(SELECT (COUNT(*) AS ?count) WHERE {{ {} }}) <= {}",
                        base_pattern, max
                    ));
                }

                if !filters.is_empty() {
                    format!("FILTER({})", filters.join(" && "))
                } else {
                    base_pattern
                }
            }
        };

        // Add context bindings if provided
        let context_bindings = if let Some(ctx) = context {
            ctx.bindings
                .iter()
                .map(|(var, term)| {
                    let term_str = match term {
                        Term::NamedNode(nn) => format!("<{}>", nn.as_str()),
                        Term::Literal(lit) => format!("\"{}\"", lit.value()),
                        _ => format!("{:?}", term),
                    };
                    format!("BIND({} AS ?{})", term_str, var)
                })
                .collect::<Vec<_>>()
                .join("\n  ")
        } else {
            String::new()
        };

        Ok(format!(
            "SELECT DISTINCT ?target WHERE {{\n  {}\n  {}\n  {}\n}}",
            base_where, condition_clause, context_bindings
        ))
    }

    /// Generate SPARQL query for hierarchical targets
    fn generate_hierarchical_query(
        &self,
        root_target: &Target,
        relationship: &HierarchicalRelationship,
        max_depth: i32,
        include_intermediate: bool,
        graph_name: Option<&str>,
    ) -> Result<String> {
        let root_query = self.generate_target_query(root_target, graph_name)?;

        // Extract WHERE clause content from root query
        let root_where = if let Some(where_start) = root_query.find("WHERE {") {
            let where_clause = &root_query[where_start + 7..];
            if let Some(where_end) = where_clause.rfind('}') {
                where_clause[..where_end].trim().replace("?target", "?root")
            } else {
                return Err(ShaclError::ValidationEngine(
                    "Invalid root query structure".to_string(),
                ));
            }
        } else {
            return Err(ShaclError::ValidationEngine(
                "Root query missing WHERE clause".to_string(),
            ));
        };

        let relationship_pattern = match relationship {
            HierarchicalRelationship::Property(prop) => {
                format!("?root <{}>+ ?target", prop.as_str())
            }
            HierarchicalRelationship::InverseProperty(prop) => {
                format!("?target <{}>+ ?root", prop.as_str())
            }
            HierarchicalRelationship::SubclassOf => {
                "?root <http://www.w3.org/2000/01/rdf-schema#subClassOf>+ ?target".to_string()
            }
            HierarchicalRelationship::SuperclassOf => {
                "?target <http://www.w3.org/2000/01/rdf-schema#subClassOf>+ ?root".to_string()
            }
            HierarchicalRelationship::SubpropertyOf => {
                "?root <http://www.w3.org/2000/01/rdf-schema#subPropertyOf>+ ?target".to_string()
            }
            HierarchicalRelationship::SuperpropertyOf => {
                "?target <http://www.w3.org/2000/01/rdf-schema#subPropertyOf>+ ?root".to_string()
            }
            HierarchicalRelationship::CustomPath(path) => {
                path.replace("?start", "?root").replace("?end", "?target")
            }
        };

        // Add depth limit if specified
        let depth_limit = if max_depth > 0 {
            // This is a simplified approach - in practice, you'd need more sophisticated depth limiting
            format!("# Depth limited to {} levels", max_depth)
        } else {
            String::new()
        };

        let select_clause = if include_intermediate {
            "SELECT DISTINCT ?target"
        } else {
            "SELECT DISTINCT ?target"
        };

        Ok(format!(
            "{} WHERE {{\n  {}\n  {}\n  {}\n}}",
            select_clause, root_where, relationship_pattern, depth_limit
        ))
    }

    /// Generate SPARQL query for path-based targets
    fn generate_path_based_query(
        &self,
        start_target: &Target,
        path: &crate::paths::PropertyPath,
        direction: &PathDirection,
        filters: &[PathFilter],
        graph_name: Option<&str>,
    ) -> Result<String> {
        let start_query = self.generate_target_query(start_target, graph_name)?;

        // Extract WHERE clause content from start query
        let start_where = if let Some(where_start) = start_query.find("WHERE {") {
            let where_clause = &start_query[where_start + 7..];
            if let Some(where_end) = where_clause.rfind('}') {
                where_clause[..where_end]
                    .trim()
                    .replace("?target", "?start")
            } else {
                return Err(ShaclError::ValidationEngine(
                    "Invalid start query structure".to_string(),
                ));
            }
        } else {
            return Err(ShaclError::ValidationEngine(
                "Start query missing WHERE clause".to_string(),
            ));
        };

        // Generate property path pattern
        let path_pattern =
            self.generate_property_path_pattern(path, "?start", "?target", direction)?;

        // Generate filter conditions
        let filter_conditions = filters
            .iter()
            .map(|filter| self.generate_path_filter_condition(filter))
            .collect::<Result<Vec<_>>>()?
            .join("\n  ");

        Ok(format!(
            "SELECT DISTINCT ?target WHERE {{\n  {}\n  {}\n  {}\n}}",
            start_where, path_pattern, filter_conditions
        ))
    }

    /// Generate property path pattern for SPARQL
    fn generate_property_path_pattern(
        &self,
        path: &crate::paths::PropertyPath,
        start_var: &str,
        end_var: &str,
        direction: &PathDirection,
    ) -> Result<String> {
        use crate::paths::PropertyPath;

        match path {
            PropertyPath::Predicate(prop) => match direction {
                PathDirection::Forward => {
                    Ok(format!("{} <{}> {}", start_var, prop.as_str(), end_var))
                }
                PathDirection::Backward => {
                    Ok(format!("{} <{}> {}", end_var, prop.as_str(), start_var))
                }
                PathDirection::Both => Ok(format!(
                    "{{ {} <{}> {} }} UNION {{ {} <{}> {} }}",
                    start_var,
                    prop.as_str(),
                    end_var,
                    end_var,
                    prop.as_str(),
                    start_var
                )),
            },
            PropertyPath::Inverse(inner_path) => {
                // Reverse the direction for inverse paths
                let reversed_direction = match direction {
                    PathDirection::Forward => PathDirection::Backward,
                    PathDirection::Backward => PathDirection::Forward,
                    PathDirection::Both => PathDirection::Both,
                };
                self.generate_property_path_pattern(
                    inner_path,
                    start_var,
                    end_var,
                    &reversed_direction,
                )
            }
            PropertyPath::Sequence(paths) => {
                if paths.is_empty() {
                    return Err(ShaclError::ValidationEngine(
                        "Empty sequence path".to_string(),
                    ));
                }

                if paths.len() == 1 {
                    return self
                        .generate_property_path_pattern(&paths[0], start_var, end_var, direction);
                }

                // Generate intermediate variables
                let mut patterns = Vec::new();
                let mut current_start = start_var.to_string();

                for (i, path_segment) in paths.iter().enumerate() {
                    let current_end = if i == paths.len() - 1 {
                        end_var.to_string()
                    } else {
                        format!("?intermediate{}", i)
                    };

                    let pattern = self.generate_property_path_pattern(
                        path_segment,
                        &current_start,
                        &current_end,
                        direction,
                    )?;
                    patterns.push(pattern);
                    current_start = current_end;
                }

                Ok(patterns.join(" .\n  "))
            }
            PropertyPath::Alternative(paths) => {
                if paths.is_empty() {
                    return Err(ShaclError::ValidationEngine(
                        "Empty alternative path".to_string(),
                    ));
                }

                let patterns: Result<Vec<_>> = paths
                    .iter()
                    .map(|p| self.generate_property_path_pattern(p, start_var, end_var, direction))
                    .collect();

                Ok(format!("{{ {} }}", patterns?.join(" } UNION { ")))
            }
            PropertyPath::ZeroOrMore(inner_path) => {
                let inner_pattern =
                    self.generate_property_path_pattern(inner_path, start_var, end_var, direction)?;
                Ok(format!("({})* {}", inner_pattern, end_var))
            }
            PropertyPath::OneOrMore(inner_path) => {
                let inner_pattern =
                    self.generate_property_path_pattern(inner_path, start_var, end_var, direction)?;
                Ok(format!("({})+", inner_pattern))
            }
            PropertyPath::ZeroOrOne(inner_path) => {
                let inner_pattern =
                    self.generate_property_path_pattern(inner_path, start_var, end_var, direction)?;
                Ok(format!(
                    "{{ {} }} UNION {{ BIND({} AS {}) }}",
                    inner_pattern, start_var, end_var
                ))
            }
        }
    }

    /// Generate filter condition for path filters
    fn generate_path_filter_condition(&self, filter: &PathFilter) -> Result<String> {
        match filter {
            PathFilter::NodeType(node_type) => match node_type {
                NodeTypeFilter::IriOnly => Ok("FILTER(isIRI(?target))".to_string()),
                NodeTypeFilter::BlankNodeOnly => Ok("FILTER(isBlank(?target))".to_string()),
                NodeTypeFilter::LiteralOnly => Ok("FILTER(isLiteral(?target))".to_string()),
                NodeTypeFilter::InstanceOf(class_iri) => Ok(format!(
                    "?target <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{}> .",
                    class_iri.as_str()
                )),
            },
            PathFilter::PropertyValue { property, value } => {
                let value_str = match value {
                    Term::NamedNode(nn) => format!("<{}>", nn.as_str()),
                    Term::Literal(lit) => format!("\"{}\"", lit.value()),
                    _ => format!("{:?}", value),
                };
                Ok(format!("?target <{}> {} .", property.as_str(), value_str))
            }
            PathFilter::SparqlCondition {
                condition,
                prefixes: _,
            } => Ok(format!("FILTER({})", condition)),
        }
    }

    /// Execute union target by combining results from multiple targets
    fn execute_union_target(
        &self,
        store: &dyn Store,
        targets: &[Target],
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let mut all_nodes = HashSet::new();

        for target in targets {
            let target_nodes = self.execute_target_selection_direct(store, target, graph_name)?;
            all_nodes.extend(target_nodes);
        }

        let mut result: Vec<_> = all_nodes.into_iter().collect();
        self.sort_and_dedupe_targets(&mut result);
        Ok(result)
    }

    /// Execute intersection target by finding nodes that match all targets
    fn execute_intersection_target(
        &self,
        store: &dyn Store,
        targets: &[Target],
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        if targets.is_empty() {
            return Ok(Vec::new());
        }

        if targets.len() == 1 {
            return self.execute_target_selection_direct(store, &targets[0], graph_name);
        }

        // Get results for first target
        let mut result_sets: Vec<HashSet<Term>> = Vec::new();

        for target in targets {
            let target_nodes = self.execute_target_selection_direct(store, target, graph_name)?;
            result_sets.push(target_nodes.into_iter().collect());
        }

        // Find intersection of all sets
        if let Some(first_set) = result_sets.first() {
            let intersection: HashSet<_> = result_sets
                .iter()
                .skip(1)
                .fold(first_set.clone(), |acc, set| {
                    acc.intersection(set).cloned().collect()
                });

            let mut result: Vec<_> = intersection.into_iter().collect();
            self.sort_and_dedupe_targets(&mut result);
            Ok(result)
        } else {
            Ok(Vec::new())
        }
    }

    /// Execute difference target by subtracting exclusion target from primary target
    fn execute_difference_target(
        &self,
        store: &dyn Store,
        primary: &Target,
        exclusion: &Target,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let primary_nodes = self.execute_target_selection_direct(store, primary, graph_name)?;
        let exclusion_nodes: HashSet<_> = self
            .execute_target_selection_direct(store, exclusion, graph_name)?
            .into_iter()
            .collect();

        let mut result: Vec<_> = primary_nodes
            .into_iter()
            .filter(|node| !exclusion_nodes.contains(node))
            .collect();

        self.sort_and_dedupe_targets(&mut result);
        Ok(result)
    }

    /// Execute conditional target by filtering base target with condition  
    fn execute_conditional_target(
        &self,
        store: &dyn Store,
        base_target: &Target,
        condition: &TargetCondition,
        _context: Option<&TargetContext>,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let base_nodes = self.execute_target_selection_direct(store, base_target, graph_name)?;
        let mut filtered_nodes = Vec::new();

        for node in base_nodes {
            if self.evaluate_target_condition(store, &node, condition, graph_name)? {
                filtered_nodes.push(node);
            }
        }

        self.sort_and_dedupe_targets(&mut filtered_nodes);
        Ok(filtered_nodes)
    }

    /// Execute hierarchical target by following relationships from root targets
    fn execute_hierarchical_target(
        &self,
        store: &dyn Store,
        root_target: &Target,
        relationship: &HierarchicalRelationship,
        max_depth: i32,
        include_intermediate: bool,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let root_nodes = self.execute_target_selection_direct(store, root_target, graph_name)?;
        let mut result_nodes = HashSet::new();

        for root_node in root_nodes {
            let traversed_nodes = self.traverse_hierarchy(
                store,
                &root_node,
                relationship,
                max_depth,
                include_intermediate,
                graph_name,
            )?;
            result_nodes.extend(traversed_nodes);
        }

        let mut result: Vec<_> = result_nodes.into_iter().collect();
        self.sort_and_dedupe_targets(&mut result);
        Ok(result)
    }

    /// Execute path-based target by following property paths from start targets
    fn execute_path_based_target(
        &self,
        store: &dyn Store,
        start_target: &Target,
        path: &crate::paths::PropertyPath,
        direction: &PathDirection,
        filters: &[PathFilter],
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let start_nodes = self.execute_target_selection_direct(store, start_target, graph_name)?;
        let mut result_nodes = HashSet::new();

        for start_node in start_nodes {
            let path_nodes =
                self.evaluate_property_path(store, &start_node, path, direction, graph_name)?;

            // Apply filters
            for node in path_nodes {
                if self.apply_path_filters(store, &node, filters, graph_name)? {
                    result_nodes.insert(node);
                }
            }
        }

        let mut result: Vec<_> = result_nodes.into_iter().collect();
        self.sort_and_dedupe_targets(&mut result);
        Ok(result)
    }

    /// Evaluate a target condition for a specific node
    fn evaluate_target_condition(
        &self,
        store: &dyn Store,
        node: &Term,
        condition: &TargetCondition,
        graph_name: Option<&str>,
    ) -> Result<bool> {
        use oxirs_core::model::{
            GraphName, NamedNode as CoreNamedNode, Object, Predicate, Subject,
        };

        match condition {
            TargetCondition::SparqlAsk {
                query: _,
                prefixes: _,
            } => {
                // For now, simplified evaluation - in practice you'd substitute the node into the ASK query
                Ok(true)
            }
            TargetCondition::PropertyExists {
                property,
                direction,
            } => {
                let property_predicate =
                    Predicate::NamedNode(CoreNamedNode::new(property.as_str()).map_err(|e| {
                        ShaclError::TargetSelection(format!("Invalid property IRI: {}", e))
                    })?);

                let exists = match direction {
                    PropertyDirection::Subject => {
                        // Check if node is subject of the property
                        let subject = match node {
                            Term::NamedNode(nn) => Subject::NamedNode(nn.clone()),
                            Term::BlankNode(bn) => Subject::BlankNode(bn.clone()),
                            _ => return Ok(false),
                        };

                        !store
                            .find_quads(Some(&subject), Some(&property_predicate), None, None)?
                            .is_empty()
                    }
                    PropertyDirection::Object => {
                        // Check if node is object of the property
                        let object = match node {
                            Term::NamedNode(nn) => Object::NamedNode(nn.clone()),
                            Term::BlankNode(bn) => Object::BlankNode(bn.clone()),
                            Term::Literal(lit) => Object::Literal(lit.clone()),
                            _ => return Ok(false),
                        };

                        !store
                            .find_quads(None, Some(&property_predicate), Some(&object), None)?
                            .is_empty()
                    }
                    PropertyDirection::Either => {
                        // Check both directions
                        self.evaluate_target_condition(
                            store,
                            node,
                            &TargetCondition::PropertyExists {
                                property: property.clone(),
                                direction: PropertyDirection::Subject,
                            },
                            graph_name,
                        )? || self.evaluate_target_condition(
                            store,
                            node,
                            &TargetCondition::PropertyExists {
                                property: property.clone(),
                                direction: PropertyDirection::Object,
                            },
                            graph_name,
                        )?
                    }
                };

                Ok(exists)
            }
            TargetCondition::PropertyValue {
                property: _,
                value: _,
                direction: _,
            } => {
                // Similar to PropertyExists but check for specific value
                Ok(false) // Simplified for now
            }
            TargetCondition::HasType { class_iri } => {
                let rdf_type = Predicate::NamedNode(
                    CoreNamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").map_err(
                        |e| ShaclError::TargetSelection(format!("Invalid RDF type IRI: {}", e)),
                    )?,
                );
                let class_object = Object::NamedNode(class_iri.clone());
                let subject = match node {
                    Term::NamedNode(nn) => Subject::NamedNode(nn.clone()),
                    Term::BlankNode(bn) => Subject::BlankNode(bn.clone()),
                    _ => return Ok(false),
                };

                let has_type = !store
                    .find_quads(Some(&subject), Some(&rdf_type), Some(&class_object), None)?
                    .is_empty();
                Ok(has_type)
            }
            TargetCondition::Cardinality {
                property: _,
                min_count: _,
                max_count: _,
                direction: _,
            } => {
                // Simplified for now - would need to count occurrences
                Ok(true)
            }
        }
    }

    /// Traverse hierarchy from a node following relationships
    fn traverse_hierarchy(
        &self,
        store: &dyn Store,
        start_node: &Term,
        relationship: &HierarchicalRelationship,
        max_depth: i32,
        include_intermediate: bool,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back((start_node.clone(), 0));

        while let Some((current_node, depth)) = queue.pop_front() {
            if max_depth >= 0 && depth > max_depth {
                continue;
            }

            if visited.contains(&current_node) {
                continue;
            }
            visited.insert(current_node.clone());

            if depth > 0 && (include_intermediate || depth == max_depth || queue.is_empty()) {
                result.push(current_node.clone());
            }

            // Find related nodes based on relationship type
            let related_nodes =
                self.find_related_nodes(store, &current_node, relationship, graph_name)?;
            for related_node in related_nodes {
                if !visited.contains(&related_node) {
                    queue.push_back((related_node, depth + 1));
                }
            }
        }

        Ok(result)
    }

    /// Find nodes related to the current node through a specific relationship
    fn find_related_nodes(
        &self,
        store: &dyn Store,
        node: &Term,
        relationship: &HierarchicalRelationship,
        _graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        use oxirs_core::model::{NamedNode as CoreNamedNode, Object, Predicate, Subject};

        let subject = match node {
            Term::NamedNode(nn) => Subject::NamedNode(nn.clone()),
            Term::BlankNode(bn) => Subject::BlankNode(bn.clone()),
            _ => return Ok(Vec::new()),
        };

        let mut related_nodes = Vec::new();

        match relationship {
            HierarchicalRelationship::Property(prop) => {
                let predicate = Predicate::NamedNode(prop.clone());
                for quad in store.find_quads(Some(&subject), Some(&predicate), None, None)? {
                    match quad.object() {
                        Object::NamedNode(nn) => related_nodes.push(Term::NamedNode(nn.clone())),
                        Object::BlankNode(bn) => related_nodes.push(Term::BlankNode(bn.clone())),
                        _ => {}
                    }
                }
            }
            HierarchicalRelationship::InverseProperty(prop) => {
                let predicate = Predicate::NamedNode(prop.clone());
                let object = match node {
                    Term::NamedNode(nn) => Object::NamedNode(nn.clone()),
                    Term::BlankNode(bn) => Object::BlankNode(bn.clone()),
                    _ => return Ok(Vec::new()),
                };

                for quad in store.find_quads(None, Some(&predicate), Some(&object), None)? {
                    match quad.subject() {
                        Subject::NamedNode(nn) => related_nodes.push(Term::NamedNode(nn.clone())),
                        Subject::BlankNode(bn) => related_nodes.push(Term::BlankNode(bn.clone())),
                        _ => {}
                    }
                }
            }
            HierarchicalRelationship::SubclassOf => {
                let subclass_prop = Predicate::NamedNode(
                    CoreNamedNode::new("http://www.w3.org/2000/01/rdf-schema#subClassOf").map_err(
                        |e| ShaclError::TargetSelection(format!("Invalid subClassOf IRI: {}", e)),
                    )?,
                );

                for quad in store.find_quads(Some(&subject), Some(&subclass_prop), None, None)? {
                    match quad.object() {
                        Object::NamedNode(nn) => related_nodes.push(Term::NamedNode(nn.clone())),
                        Object::BlankNode(bn) => related_nodes.push(Term::BlankNode(bn.clone())),
                        _ => {}
                    }
                }
            }
            HierarchicalRelationship::SuperclassOf => {
                // Find all nodes that are subclasses of the current node
                let subclass_prop = Predicate::NamedNode(
                    CoreNamedNode::new("http://www.w3.org/2000/01/rdf-schema#subClassOf").map_err(
                        |e| ShaclError::TargetSelection(format!("Invalid subClassOf IRI: {}", e)),
                    )?,
                );
                let object = match node {
                    Term::NamedNode(nn) => Object::NamedNode(nn.clone()),
                    Term::BlankNode(bn) => Object::BlankNode(bn.clone()),
                    _ => return Ok(Vec::new()),
                };

                for quad in store.find_quads(None, Some(&subclass_prop), Some(&object), None)? {
                    match quad.subject() {
                        Subject::NamedNode(nn) => related_nodes.push(Term::NamedNode(nn.clone())),
                        Subject::BlankNode(bn) => related_nodes.push(Term::BlankNode(bn.clone())),
                        _ => {}
                    }
                }
            }
            HierarchicalRelationship::SubpropertyOf => {
                let subprop_prop = Predicate::NamedNode(
                    CoreNamedNode::new("http://www.w3.org/2000/01/rdf-schema#subPropertyOf")
                        .map_err(|e| {
                            ShaclError::TargetSelection(format!("Invalid subPropertyOf IRI: {}", e))
                        })?,
                );

                for quad in store.find_quads(Some(&subject), Some(&subprop_prop), None, None)? {
                    match quad.object() {
                        Object::NamedNode(nn) => related_nodes.push(Term::NamedNode(nn.clone())),
                        Object::BlankNode(bn) => related_nodes.push(Term::BlankNode(bn.clone())),
                        _ => {}
                    }
                }
            }
            HierarchicalRelationship::SuperpropertyOf => {
                let subprop_prop = Predicate::NamedNode(
                    CoreNamedNode::new("http://www.w3.org/2000/01/rdf-schema#subPropertyOf")
                        .map_err(|e| {
                            ShaclError::TargetSelection(format!("Invalid subPropertyOf IRI: {}", e))
                        })?,
                );
                let object = match node {
                    Term::NamedNode(nn) => Object::NamedNode(nn.clone()),
                    Term::BlankNode(bn) => Object::BlankNode(bn.clone()),
                    _ => return Ok(Vec::new()),
                };

                for quad in store.find_quads(None, Some(&subprop_prop), Some(&object), None)? {
                    match quad.subject() {
                        Subject::NamedNode(nn) => related_nodes.push(Term::NamedNode(nn.clone())),
                        Subject::BlankNode(bn) => related_nodes.push(Term::BlankNode(bn.clone())),
                        _ => {}
                    }
                }
            }
            HierarchicalRelationship::CustomPath(_path) => {
                // Custom path evaluation would require SPARQL execution
                // Simplified for now
            }
        }

        Ok(related_nodes)
    }

    /// Evaluate property path from a starting node
    fn evaluate_property_path(
        &self,
        store: &dyn Store,
        start_node: &Term,
        path: &crate::paths::PropertyPath,
        direction: &PathDirection,
        graph_name: Option<&str>,
    ) -> Result<Vec<Term>> {
        use crate::paths::PropertyPath;
        use oxirs_core::model::{NamedNode as CoreNamedNode, Object, Predicate, Subject};

        match path {
            PropertyPath::Predicate(prop) => {
                let mut result = Vec::new();
                let predicate = Predicate::NamedNode(prop.clone());

                match direction {
                    PathDirection::Forward => {
                        let subject = match start_node {
                            Term::NamedNode(nn) => Subject::NamedNode(nn.clone()),
                            Term::BlankNode(bn) => Subject::BlankNode(bn.clone()),
                            _ => return Ok(Vec::new()),
                        };

                        for quad in
                            store.find_quads(Some(&subject), Some(&predicate), None, None)?
                        {
                            match quad.object() {
                                Object::NamedNode(nn) => result.push(Term::NamedNode(nn.clone())),
                                Object::BlankNode(bn) => result.push(Term::BlankNode(bn.clone())),
                                Object::Literal(lit) => result.push(Term::Literal(lit.clone())),
                                _ => {}
                            }
                        }
                    }
                    PathDirection::Backward => {
                        let object = match start_node {
                            Term::NamedNode(nn) => Object::NamedNode(nn.clone()),
                            Term::BlankNode(bn) => Object::BlankNode(bn.clone()),
                            Term::Literal(lit) => Object::Literal(lit.clone()),
                            _ => return Ok(Vec::new()),
                        };

                        for quad in store.find_quads(None, Some(&predicate), Some(&object), None)? {
                            match quad.subject() {
                                Subject::NamedNode(nn) => result.push(Term::NamedNode(nn.clone())),
                                Subject::BlankNode(bn) => result.push(Term::BlankNode(bn.clone())),
                                _ => {}
                            }
                        }
                    }
                    PathDirection::Both => {
                        let mut forward_nodes = self.evaluate_property_path(
                            store,
                            start_node,
                            path,
                            &PathDirection::Forward,
                            graph_name,
                        )?;
                        let backward_nodes = self.evaluate_property_path(
                            store,
                            start_node,
                            path,
                            &PathDirection::Backward,
                            graph_name,
                        )?;
                        result.append(&mut forward_nodes);
                        result.extend(backward_nodes);
                    }
                }

                Ok(result)
            }
            PropertyPath::Inverse(inner_path) => {
                let reversed_direction = match direction {
                    PathDirection::Forward => PathDirection::Backward,
                    PathDirection::Backward => PathDirection::Forward,
                    PathDirection::Both => PathDirection::Both,
                };
                self.evaluate_property_path(
                    store,
                    start_node,
                    inner_path,
                    &reversed_direction,
                    graph_name,
                )
            }
            PropertyPath::Sequence(paths) => {
                if paths.is_empty() {
                    return Ok(Vec::new());
                }

                let mut current_nodes = vec![start_node.clone()];

                for path_segment in paths {
                    let mut next_nodes = Vec::new();
                    for node in current_nodes {
                        let segment_nodes = self.evaluate_property_path(
                            store,
                            &node,
                            path_segment,
                            direction,
                            graph_name,
                        )?;
                        next_nodes.extend(segment_nodes);
                    }
                    current_nodes = next_nodes;
                }

                Ok(current_nodes)
            }
            PropertyPath::Alternative(paths) => {
                let mut result = Vec::new();
                for path_alternative in paths {
                    let alt_nodes = self.evaluate_property_path(
                        store,
                        start_node,
                        path_alternative,
                        direction,
                        graph_name,
                    )?;
                    result.extend(alt_nodes);
                }
                Ok(result)
            }
            PropertyPath::ZeroOrMore(_inner_path) => {
                // Simplified - would need proper transitive closure
                Ok(vec![start_node.clone()])
            }
            PropertyPath::OneOrMore(_inner_path) => {
                // Simplified - would need proper transitive closure
                Ok(Vec::new())
            }
            PropertyPath::ZeroOrOne(inner_path) => {
                let mut result = vec![start_node.clone()]; // Zero case
                let one_nodes = self
                    .evaluate_property_path(store, start_node, inner_path, direction, graph_name)?;
                result.extend(one_nodes); // One case
                Ok(result)
            }
        }
    }

    /// Apply path filters to a node
    fn apply_path_filters(
        &self,
        store: &dyn Store,
        node: &Term,
        filters: &[PathFilter],
        _graph_name: Option<&str>,
    ) -> Result<bool> {
        for filter in filters {
            match filter {
                PathFilter::NodeType(node_type) => {
                    let matches = match node_type {
                        NodeTypeFilter::IriOnly => matches!(node, Term::NamedNode(_)),
                        NodeTypeFilter::BlankNodeOnly => matches!(node, Term::BlankNode(_)),
                        NodeTypeFilter::LiteralOnly => matches!(node, Term::Literal(_)),
                        NodeTypeFilter::InstanceOf(class_iri) => self.evaluate_target_condition(
                            store,
                            node,
                            &TargetCondition::HasType {
                                class_iri: class_iri.clone(),
                            },
                            None,
                        )?,
                    };
                    if !matches {
                        return Ok(false);
                    }
                }
                PathFilter::PropertyValue { property, value } => {
                    let matches = self.evaluate_target_condition(
                        store,
                        node,
                        &TargetCondition::PropertyValue {
                            property: property.clone(),
                            value: value.clone(),
                            direction: PropertyDirection::Subject,
                        },
                        None,
                    )?;
                    if !matches {
                        return Ok(false);
                    }
                }
                PathFilter::SparqlCondition {
                    condition: _,
                    prefixes: _,
                } => {
                    // Simplified - would need SPARQL evaluation
                    // For now, assume it passes
                }
            }
        }
        Ok(true)
    }
}

impl Default for TargetSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_creation() {
        let class_iri = NamedNode::new("http://example.org/Person").unwrap();
        let target = Target::class(class_iri.clone());

        match target {
            Target::Class(iri) => assert_eq!(iri, class_iri),
            _ => panic!("Expected class target"),
        }
    }

    #[test]
    fn test_node_target() {
        let node = Term::NamedNode(NamedNode::new("http://example.org/john").unwrap());
        let target = Target::node(node.clone());

        match target {
            Target::Node(n) => assert_eq!(n, node),
            _ => panic!("Expected node target"),
        }
    }

    #[test]
    fn test_sparql_target() {
        let query = "SELECT ?this WHERE { ?this a ex:Person }".to_string();
        let prefixes = Some("PREFIX ex: <http://example.org/>".to_string());
        let target = Target::sparql(query.clone(), prefixes.clone());

        match target {
            Target::Sparql(sparql_target) => {
                assert_eq!(sparql_target.query, query);
                assert_eq!(sparql_target.prefixes, prefixes);
            }
            _ => panic!("Expected SPARQL target"),
        }
    }

    #[test]
    fn debug_target_query_generation() {
        let selector = TargetSelector::new();
        let class_iri = NamedNode::new("http://example.org/Person").unwrap();
        let target = Target::class(class_iri);

        let query = selector.generate_target_query(&target, None).unwrap();
        println!("Generated query: '{}'", query);
        eprintln!("Generated query: '{}'", query);

        assert!(query.contains("SELECT"));
        assert!(query.contains("?target"));
        assert!(query.contains("http://example.org/Person"));
    }

    #[test]
    fn debug_direct_sparql_execution() {
        use oxirs_core::model::{GraphName, Quad};
        use oxirs_core::{query::QueryEngine, Store};

        let mut store = ConcreteStore::new().unwrap();

        // Test simple query first
        let simple_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }";
        println!("Testing simple query: '{}'", simple_query);

        let query_engine = QueryEngine::new();
        let result = query_engine.query(simple_query, &store);

        match result {
            Ok(result) => println!("Simple query executed successfully: {:?}", result),
            Err(e) => {
                println!("Simple query failed: {}", e);
                eprintln!("Simple query failed: {}", e);
            }
        }

        // Test with simpler IRI first
        let simple_iri_query = "SELECT DISTINCT ?target WHERE { ?target <http://example.org/type> <http://example.org/Person> . }";
        println!("Testing simple IRI query: '{}'", simple_iri_query);

        let result2 = query_engine.query(simple_iri_query, &store);

        match result2 {
            Ok(result) => println!("Simple IRI query executed successfully: {:?}", result),
            Err(e) => {
                println!("Simple IRI query failed: {}", e);
                eprintln!("Simple IRI query failed: {}", e);
            }
        }

        // Test our specific query with long RDF type IRI
        let query = "SELECT DISTINCT ?target WHERE { ?target <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> . }";
        println!("Testing RDF type query: '{}'", query);

        let result3 = query_engine.query(query, &store);

        match result3 {
            Ok(result) => println!("RDF type query executed successfully: {:?}", result),
            Err(e) => {
                println!("RDF type query failed: {}", e);
                eprintln!("RDF type query failed: {}", e);
            }
        }
    }

    #[test]
    fn test_cache_key_generation() {
        let selector = TargetSelector::new();
        let class_iri = NamedNode::new("http://example.org/Person").unwrap();
        let target = Target::class(class_iri);

        let key1 = selector.create_cache_key(&target, None);
        let key2 = selector.create_cache_key(&target, None);
        let key3 = selector.create_cache_key(&target, Some("graph1"));

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_target_optimization_config() {
        let config = TargetOptimizationConfig::default();

        assert!(config.enable_caching);
        assert_eq!(config.cache_ttl, 300);
        assert_eq!(config.max_cache_size, 1000);
        assert!(config.enable_query_optimization);
    }

    #[test]
    fn test_target_selector_with_config() {
        let mut config = TargetOptimizationConfig::default();
        config.cache_ttl = 600;
        config.enable_caching = false;

        let selector = TargetSelector::with_config(config.clone());
        assert_eq!(selector.optimization_config.cache_ttl, 600);
        assert!(!selector.optimization_config.enable_caching);
    }
}
