//! Type definitions for BGP optimization
//!
//! This module contains all the data structures used in BGP optimization,
//! including selectivity information, index planning, and optimization results.

use crate::algebra::TriplePattern;
use crate::optimizer::index_types::IndexType;
use std::collections::HashMap;

/// BGP optimization result
#[derive(Debug, Clone)]
pub struct OptimizedBGP {
    /// Reordered triple patterns
    pub patterns: Vec<TriplePattern>,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Selectivity information
    pub selectivity_info: SelectivityInfo,
    /// Recommended index usage
    pub index_plan: IndexUsagePlan,
}

/// Selectivity information for BGP
#[derive(Debug, Clone)]
pub struct SelectivityInfo {
    /// Pattern selectivities
    pub pattern_selectivity: Vec<PatternSelectivity>,
    /// Join selectivities
    pub join_selectivity: HashMap<(usize, usize), f64>,
    /// Overall BGP selectivity
    pub overall_selectivity: f64,
}

/// Selectivity information for a single pattern
#[derive(Debug, Clone)]
pub struct PatternSelectivity {
    /// Triple pattern
    pub pattern: TriplePattern,
    /// Estimated selectivity (0.0 to 1.0)
    pub selectivity: f64,
    /// Estimated cardinality
    pub cardinality: usize,
    /// Contributing factors
    pub factors: SelectivityFactors,
}

/// Factors contributing to selectivity
#[derive(Debug, Clone)]
pub struct SelectivityFactors {
    /// Subject selectivity
    pub subject_selectivity: f64,
    /// Predicate selectivity
    pub predicate_selectivity: f64,
    /// Object selectivity
    pub object_selectivity: f64,
    /// Type selectivity
    pub type_selectivity: f64,
    /// Literal selectivity
    pub literal_selectivity: f64,
    /// Index availability factor
    pub index_factor: f64,
    /// Data distribution factor
    pub distribution_factor: f64,
}

/// Index usage plan for BGP execution
#[derive(Debug, Clone)]
pub struct IndexUsagePlan {
    /// Index assignments per pattern
    pub pattern_indexes: Vec<IndexAssignment>,
    /// Join index opportunities
    pub join_indexes: Vec<JoinIndexOpportunity>,
    /// Multi-index intersection opportunities
    pub index_intersections: Vec<IndexIntersection>,
    /// Bloom filter recommendations
    pub bloom_filter_candidates: Vec<BloomFilterCandidate>,
    /// Recommended indices
    pub recommended_indices: Vec<IndexType>,
    /// Access patterns
    pub access_patterns: Vec<String>,
    /// Estimated cost reduction
    pub estimated_cost_reduction: f64,
}

/// Index assignment for a pattern
#[derive(Debug, Clone)]
pub struct IndexAssignment {
    /// Pattern index
    pub pattern_idx: usize,
    /// Recommended index type
    pub index_type: IndexType,
    /// Expected scan cost
    pub scan_cost: f64,
}

/// Join index opportunity
#[derive(Debug, Clone)]
pub struct JoinIndexOpportunity {
    /// Left pattern index
    pub left_pattern_idx: usize,
    /// Right pattern index
    pub right_pattern_idx: usize,
    /// Join variable
    pub join_var: String,
    /// Potential speedup factor
    pub speedup_factor: f64,
}

/// Multi-index intersection for complex queries
#[derive(Debug, Clone)]
pub struct IndexIntersection {
    /// Pattern index for intersection
    pub pattern_idx: usize,
    /// Primary index type
    pub primary_index: IndexType,
    /// Secondary indexes for intersection
    pub secondary_indexes: Vec<IndexType>,
    /// Expected benefit from intersection
    pub selectivity_improvement: f64,
    /// Intersection algorithm to use
    pub intersection_algorithm: IntersectionAlgorithm,
}

/// Types of index intersections
#[derive(Debug, Clone)]
pub enum IntersectionType {
    /// Variable-based join intersection
    VariableJoin,
    /// Value-based intersection
    ValueIntersection,
    /// Spatial intersection
    SpatialIntersection,
    /// Temporal intersection
    TemporalIntersection,
}

/// Intersection algorithm types
#[derive(Debug, Clone)]
pub enum IntersectionAlgorithm {
    /// Bitmap intersection for dense results
    Bitmap,
    /// Hash-based intersection for sparse results
    Hash,
    /// Skip-list intersection for ordered indexes
    SkipList,
}

/// Bloom filter candidate for negative lookups
#[derive(Debug, Clone)]
pub struct BloomFilterCandidate {
    /// Pattern index
    pub pattern_idx: usize,
    /// Filter position (Subject, Predicate, Object)
    pub filter_position: TermPosition,
    /// Expected false positive rate
    pub false_positive_rate: f64,
    /// Memory footprint estimate (bytes)
    pub memory_footprint: usize,
    /// Expected performance gain
    pub performance_gain: f64,
}

/// Adaptive index selector for dynamic optimization
#[derive(Debug, Clone)]
pub struct AdaptiveIndexSelector {
    /// Query pattern frequency
    #[allow(dead_code)]
    pub(crate) pattern_frequency: HashMap<String, usize>,
    /// Index effectiveness history
    #[allow(dead_code)]
    pub(crate) index_effectiveness: HashMap<IndexType, f64>,
    /// Workload characteristics
    #[allow(dead_code)]
    pub(crate) workload_characteristics: WorkloadCharacteristics,
}

/// Workload characteristics for adaptive optimization
#[derive(Debug, Clone, Default)]
pub struct WorkloadCharacteristics {
    /// Average query complexity (number of patterns)
    pub avg_query_complexity: f64,
    /// Predicate diversity
    pub predicate_diversity: f64,
    /// Join frequency patterns
    pub join_frequency: HashMap<String, usize>,
    /// Temporal query patterns
    pub temporal_access_patterns: HashMap<String, Vec<std::time::Instant>>,
}

/// Term position enumeration
#[derive(Debug, Clone, Copy)]
pub enum TermPosition {
    Subject,
    Predicate,
    Object,
}

impl TermPosition {
    pub fn to_string(&self) -> &'static str {
        match self {
            TermPosition::Subject => "subject",
            TermPosition::Predicate => "predicate",
            TermPosition::Object => "object",
        }
    }
}

/// Operations that can be spilled to disk
#[derive(Debug, Clone)]
pub enum SpillOperation {
    /// Intermediate result sets
    IntermediateResults,
    /// Hash table for joins
    HashTable,
}

impl Default for AdaptiveIndexSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveIndexSelector {
    pub fn new() -> Self {
        Self {
            pattern_frequency: HashMap::new(),
            index_effectiveness: HashMap::new(),
            workload_characteristics: WorkloadCharacteristics::default(),
        }
    }
}