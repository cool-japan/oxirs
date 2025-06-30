//! Index Types and Structures
//!
//! Definitions for different types of indexes and their positions.

/// Index type specification for optimization
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum IndexType {
    /// Primary SPOC index (Subject, Predicate, Object, Context)
    SPOC,
    /// Secondary POSC index (Predicate, Object, Subject, Context)
    POSC,
    /// Secondary OSPC index (Object, Subject, Predicate, Context)
    OSPC,
    
    // Additional index types
    Hash,
    BTree,
    Bitmap,
    Bloom,
    
    // Advanced index types for enhanced optimization
    BTreeIndex(IndexPosition),
    HashIndex(IndexPosition),
    BitmapIndex(IndexPosition),
    SpatialRTree,
    TemporalBTree,
    MultiColumnBTree(Vec<IndexPosition>),
    BloomFilter(IndexPosition),
    Custom(String),
}

/// Index position specification
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum IndexPosition {
    Subject,
    Predicate,
    Object,
    SubjectPredicate,
    PredicateObject,
    SubjectObject,
    FullTriple,
}

/// Index statistics for optimization decisions
#[derive(Debug, Clone, Default)]
pub struct IndexStatistics {
    /// Number of distinct subjects
    pub subject_count: usize,
    /// Number of distinct predicates
    pub predicate_count: usize,
    /// Number of distinct objects
    pub object_count: usize,
    /// Total number of triples
    pub triple_count: usize,
    /// Average selectivity
    pub avg_selectivity: f64,
    /// Index access frequency
    pub access_frequency: usize,
}

/// Index union plan for OR conditions
#[derive(Debug, Clone)]
pub struct IndexUnionPlan {
    pub left_indexes: Vec<IndexType>,
    pub right_indexes: Vec<IndexType>,
    pub union_cost: f64,
    pub estimated_selectivity: f64,
}

/// Index filter plan for push-down optimization
#[derive(Debug, Clone)]
pub struct IndexFilterPlan {
    pub pattern_index: usize,
    pub filter_index: IndexType,
    pub push_down_conditions: Vec<crate::algebra::Expression>,
    pub estimated_cost: f64,
}